"""Пример для совета небольшим и средним МО: правило ``usefulness.example_small``.

Правило записано 04.10.2026 отдельным коммитом ДО выбора. Пример — иллюстрация совета, не проверка: вердиктов
нет, любой исход публикуется. По образцу ``usefulness.example`` (МО, типичное по разности ошибок «соседи
по региону − похожие по корзине»), но на тех же МО и целях, что число совета (``size_check.json`` →
``lower_four``):

1. **Пул:** МО общего набора T7 с населением ``pop_avg`` 2023 года не выше границы верхнего квинтиля
   (``size_check.json`` → последний элемент ``quintile_bounds``; на границе — в нижний квинтиль), у которых
   все три цели (общепит, доход 5-НДФЛ, отгрузка) входят в расчёт (``usefulness_by_type.target_table``,
   ``in_target`` после ``min_set``).
2. **Счёт:** по каждой цели d_t = err_B − err_D, u_t = (средний ранг d_t в пуле − 0,5) / n пула; s — среднее
   u_t по трём целям.
3. **Выбор:** МО с наименьшим |s − медиана s|; равенство — в пределах ``usefulness.rule_share.tie_tol``,
   из равных — меньший ``territory_id``.

Контроль (код 3): число МО ниже границы и средняя по трём целям доля «B ближе D» у них тем же кодом совпадают
с ``size_check.json`` → ``lower_four`` — то есть пример выбран на тех же МО, целях и наборах, что 56,3%.

Выход — ``<paths.outputs>/usefulness/example_small.json``; прежние выходы этапа не трогает. Вызывается после
``usefulness_size.run`` и ``usefulness_level.run`` последним шагом ``munnet.usefulness.run``
(подключён после коммита правила 28a2079).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from munnet import usefulness_by_type as BT
from munnet import usefulness_size as Z
from munnet.config import Config
from munnet.contracts import MissingInputError, QCError
from munnet.usefulness import compare_errors, percentile_of

log = logging.getLogger(__name__)

# Ключи подблока example_small, которые код исполняет буквально. Другое значение — код 3. Тексты
# (seen_before, words.rule, words.no_claim) не сверяются.
EXPECTED: dict[tuple[str, ...], Any] = {
    ("status",): "illustration",
    ("pool", "base"): "t7_common",
    ("pool", "size"): "lower_four",
    ("pool", "targets"): "by_type_test",
    ("pool", "require"): "all_three",
    ("pool", "empty"): "error",
    ("statistic", "d"): "err_B_minus_err_D",
    ("statistic", "rank"): "within_pool",
    ("statistic", "score"): "mean_of_three",
    ("statistic", "pick"): "closest_to_median",
    ("statistic", "seen_target"): "none",
    ("show",): ["who", "per_target", "n_closer", "score", "retail", "members"],
    ("words", "any_outcome"): "publish",
    ("qc", "lower_four_n"): "size_check",
    ("qc", "lower_four_share"): 1.0e-12,
    ("qc", "spec_frozen"): True,
}
TARGETS = BT.TARGETS  # общепит, доход 5-НДФЛ, отгрузка — цели числа совета
SEEN = BT.SEEN  # розница — показывается рядом, в выбор не входит
POP_COL = Z.POP_COL
YEAR = 2023  # usefulness.size_posthoc.size.year — сверяется в compute


def check_spec(block: Mapping | None) -> None:
    """Ключи подблока ``example_small`` равны записанным (``EXPECTED``); иначе ``QCError``."""
    if not isinstance(block, Mapping):
        raise QCError("usefulness.example_small: подблока нет в конфиге")
    bad = [
        f"{'.'.join(p)} = {BT._get(block, p)!r} (ожидалось {v!r})"
        for p, v in EXPECTED.items()
        if BT._get(block, p) != v
    ]
    if bad:
        raise QCError("usefulness.example_small: блок конфига не совпадает с кодом: " + "; ".join(bad))


# --- Правило ------------------------------------------------------------------------------------------------


def small_mask(pop: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    """МО не выше границы верхнего квинтиля (квинтили 1–4; на границе — в нижний, как ``assign_quantile``)."""
    q = Z.assign_quantile(pop, np.asarray(bounds, dtype=float))
    return (q >= 0) & (q < len(bounds))


def rank_scores(d: Mapping[str, np.ndarray]) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """u_t = (средний ранг d_t − 0,5) / n по каждой цели и s — среднее u_t по целям (массивы — МО пула)."""
    u = {}
    for t, v in d.items():
        v = np.asarray(v, dtype=float)
        if not np.isfinite(v).all():
            raise QCError(f"usefulness.example_small: у МО пула нет разности ошибок по цели {t}")
        u[t] = (rankdata(v, method="average") - 0.5) / len(v)
    return u, np.mean(np.stack(list(u.values())), axis=0)


def pick_closest_to_median(ids: np.ndarray, s: np.ndarray, tol: float) -> int:
    """МО с наименьшим |s − медиана s|; равные в пределах ``tol`` — меньший ``territory_id``."""
    ids = np.asarray(ids, dtype=np.int64)
    dist = np.abs(np.asarray(s, dtype=float) - np.median(s))
    return int(np.min(ids[dist <= dist.min() + tol]))


# --- Сборка -------------------------------------------------------------------------------------------------


def _read(cfg: Config) -> dict[str, Any]:
    out = cfg.dir("outputs")
    processed = cfg.dir("processed")
    paths = {
        "t7_errors": out / "interpret" / "t7_errors.csv",
        "node_comparable": out / "interpret" / "node_comparable.csv",
        "types": out / "interpret" / "types.csv",
        "size_check": out / "usefulness" / "size_check.json",
        "mo_flags": out / "usefulness" / "mo_flags.csv",
        "context": processed / "context_annual.parquet",
        "nodes": processed / "features_nodes.parquet",
    }
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise MissingInputError(
            f"usefulness.example_small: нет входов {missing} — сначала interpret и usefulness (size_posthoc)"
        )
    cols = ["territory_id", "year", *dict.fromkeys(BT.COLUMNS.values()), BT.NDFL_FILTER, POP_COL]
    return {
        "errors": pd.read_csv(paths["t7_errors"]),
        "comparable": pd.read_csv(paths["node_comparable"]),
        "types": pd.read_csv(paths["types"]),
        "size_check": json.loads(paths["size_check"].read_text(encoding="utf-8")),
        "mo_flags": pd.read_csv(paths["mo_flags"]),
        "context": pd.read_parquet(paths["context"], columns=cols),
        "nodes": pd.read_parquet(
            paths["nodes"], columns=["territory_id", "name", "name_short", "region_name", "region_group"]
        ),
    }


def _closer(eb: float, ed: float, tol: float) -> str:
    b, tie = compare_errors(np.array([eb]), np.array([ed]), tol)
    d, _ = compare_errors(np.array([ed]), np.array([eb]), tol)
    return "B" if b[0] else ("tie" if tie[0] else ("D" if d[0] else "none"))


def compute(cfg: Config, inp: Mapping[str, Any]) -> dict[str, Any]:
    """Выбор примера по правилу и всё, что о нём показывается (``show``)."""
    block = cfg["usefulness"]["example_small"]
    check_spec(block)
    bt_block = cfg["usefulness"]["by_type_test"]
    BT.check_spec(bt_block)
    if int(cfg["usefulness"]["size_posthoc"]["size"]["year"]) != YEAR:
        raise QCError("usefulness.example_small: год населения не совпадает с size_posthoc.size.year")
    min_set = int(bt_block["universe"]["min_set"])
    tol = float(cfg["usefulness"]["rule_share"]["tie_tol"])
    qc_tol = float(block["qc"]["lower_four_share"])
    sc = inp["size_check"]
    bounds = np.asarray(sc["quintile_bounds"], dtype=float)

    e = inp["errors"]
    ids = np.sort(e.loc[e["common"].astype(bool), "territory_id"].to_numpy(dtype=np.int64))
    ctx = inp["context"]
    pop_all = ctx.loc[ctx["year"] == YEAR].set_index("territory_id")[POP_COL].astype(float)
    pop = pop_all.reindex(ids).to_numpy(dtype=float)
    small = small_mask(pop, bounds)

    # цели, наборы и «B ближе D» — как в числе совета (usefulness_size / usefulness_by_type)
    tabs: dict[str, pd.DataFrame] = {}
    ys: dict[str, pd.Series] = {}
    for t in (*TARGETS, SEEN):
        y = BT.target_change(ctx, BT.COLUMNS[t], BT.NDFL_FILTER if t == "ndfl" else None)
        ys[t] = y
        tabs[t] = BT.target_table(y, inp["comparable"], ids, min_set)

    # контроль: те же МО и та же доля, что lower_four в size_check.json
    lf = sc["lower_four"]
    if int(small.sum()) != int(lf["n"]):
        raise QCError(
            f"usefulness.example_small: МО ниже границы {int(small.sum())} ≠ lower_four.n {lf['n']}"
        )
    shares = []
    for t in TARGETS:
        tab = tabs[t]
        m = tab["in_target"].to_numpy() & small
        eb = np.nan_to_num(tab["err_B"].to_numpy(), nan=np.inf)
        ed = np.nan_to_num(tab["err_D"].to_numpy(), nan=np.inf)
        shares.append(float((m & compare_errors(eb, ed, tol)[0]).sum() / m.sum()))
    share = float(np.mean(shares))
    if not abs(share - float(lf["share"])) <= qc_tol:
        raise QCError(f"usefulness.example_small: доля {share!r} ≠ lower_four.share {lf['share']!r}")

    # пул и выбор
    all_three = np.logical_and.reduce([tabs[t]["in_target"].to_numpy() for t in TARGETS])
    pool = small & all_three
    if not pool.any():
        raise QCError("usefulness.example_small: пул пуст (pool.empty = error)")
    pids = ids[pool]
    d = {t: (tabs[t]["err_B"].to_numpy() - tabs[t]["err_D"].to_numpy())[pool] for t in TARGETS}
    u, s = rank_scores(d)
    tid = pick_closest_to_median(pids, s, tol)
    k = int(np.flatnonzero(pids == tid)[0])
    dist = np.abs(s - np.median(s))
    order = np.lexsort((pids, dist))  # ранг расстояния до медианы: 1 — пример

    nodes = inp["nodes"].set_index("territory_id")
    types = inp["types"].set_index("territory_id")
    flags = inp["mo_flags"].set_index("territory_id")
    comp = inp["comparable"]

    def who(i: int) -> dict:
        n = nodes.loc[i]
        name = n["name_short"] if pd.notna(n["name_short"]) else n["name"]
        return {"territory_id": int(i), "name": str(name), "region": str(n["region_name"])}

    def target_block(t: str, with_rank: bool) -> dict | None:
        tab = tabs[t].set_index("territory_id")
        r = tab.loc[tid]
        if not bool(r["in_target"]):
            return None
        y = ys[t]  # медианы — по членам с известной целью, как в ошибках (usefulness_by_type.set_errors)
        mb = comp.loc[(comp["territory_id"] == tid) & (comp["set"] == "B"), "other_id"].to_numpy()
        md = comp.loc[(comp["territory_id"] == tid) & (comp["set"] == "D"), "other_id"].to_numpy()
        yb = y.reindex(mb).dropna().to_numpy(dtype=float)
        yd = y.reindex(md).dropna().to_numpy(dtype=float)
        in_pool = tabs[t]["territory_id"].isin(pids).to_numpy()
        out = {
            "own_change": float(np.expm1(r["y"])),
            "median_B_change": float(np.expm1(np.median(yb))),
            "median_D_change": float(np.expm1(np.median(yd))),
            "err_B": float(r["err_B"]),
            "err_D": float(r["err_D"]),
            "closer": _closer(float(r["err_B"]), float(r["err_D"]), tol),
            "size_B": int(r["size_B"]),
            "size_D": int(r["size_D"]),
        }
        if with_rank:
            out.update(
                {
                    "pct_B_pool": percentile_of(tabs[t]["err_B"].to_numpy()[in_pool], float(r["err_B"])),
                    "pct_D_pool": percentile_of(tabs[t]["err_D"].to_numpy()[in_pool], float(r["err_D"])),
                    "u": float(u[t][k]),
                }
            )
        return out

    def members(set_name: str) -> list[dict]:
        m = comp.loc[(comp["territory_id"] == tid) & (comp["set"] == set_name)].sort_values("rank")
        return [{**who(int(r.other_id)), "km": float(r.km)} for r in m.itertuples()]

    per_target = {t: target_block(t, True) for t in TARGETS}
    n_closer = int(sum(1 for t in TARGETS if per_target[t]["closer"] == "B"))
    fl = flags.loc[tid] if tid in flags.index else None
    q = int(Z.assign_quantile(np.array([pop_all.loc[tid]]), bounds)[0]) + 1
    return {
        "status": "done",
        "kind": "illustration",
        "note": (
            "правило usefulness.example_small записано 04.10.2026 до выбора; пример — иллюстрация совета, "
            "не проверка; любой исход публикуется"
        ),
        **who(tid),
        "type": int(types.loc[tid, "type"]),
        "type_name": str(types.loc[tid, "name"]),
        "flag_text": None if fl is None else str(fl["flag_text"]),
        "pop_avg_2023": float(pop_all.loc[tid]),
        "quintile": q,
        "per_target": per_target,
        "n_closer": n_closer,
        "score": {
            "s": float(s[k]),
            "median_s": float(np.median(s)),
            "abs_diff": float(dist[k]),
            "distance_rank": int(np.flatnonzero(pids[order] == tid)[0]) + 1,
            "n_pool": int(len(pids)),
        },
        "retail": {
            "label": "цель примера Йошкар-Олы, в выбор не входит",
            **(target_block(SEEN, False) or {}),
        },
        "members_B": members("B"),
        "members_D": members("D"),
        "pool": {
            "bound": float(bounds[-1]),
            "n_small": int(small.sum()),
            "n_pool": int(len(pids)),
            "n_small_missing_target": int((small & ~all_three).sum()),
        },
        "qc": {
            "lower_four_n": int(lf["n"]),
            "lower_four_share": float(lf["share"]),
            "share_recomputed": share,
            "abs_diff": abs(share - float(lf["share"])),
            "spec_frozen": True,
        },
    }


def run(cfg: Config) -> dict[str, Any]:
    """Выбор и запись ``example_small.json``; старый файл удаляется до расчёта."""
    path = cfg.dir("outputs") / "usefulness" / "example_small.json"
    path.unlink(missing_ok=True)
    res = compute(cfg, _read(cfg))
    _write_json(path, res)
    log.info(
        "usefulness.example_small: пример для совета небольшим и средним МО — %s (%s), "
        "соседи ближе в %d из 3 целей",
        res["name"], res["region"], res["n_closer"],
    )  # fmt: skip
    return res


def _write_json(path: Path, obj: Any) -> None:
    BT._write_json(path, obj)
