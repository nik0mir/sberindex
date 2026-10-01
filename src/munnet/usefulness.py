"""Этап usefulness — разведка после вскрытия: чем полезна типология отдельному МО.

Запуск: ``python -m munnet usefulness``.

Правила записаны в блоке ``usefulness`` конфига отдельным коммитом 01.10.2026 до первого расчёта (замечание
судьи критерия 5; черновик разобрал devils-advocate). Это не проверка тезиса: вердикты, тексты исходов этапа 5
и заголовки сайта этап не трогает, он только читает выходы этапа interpret и считает описательные результаты:

1. **Сверка с соседями по своему региону.** У каждого МО общего набора T7 (``t7_errors.csv``, ``common``) —
   ошибка сверки изменения розничного оборота 2023 → 2024 с медианой соседей по своему региону (набор B)
   и с медианой похожих по тратам МО других регионов (D, набор продукта T7), цель без поправки на регион
   (``err_abs_*``). Доля случаев, где B ошибается строго меньше D, разбор с допуском δ, медиана разности
   ошибок, бутстрап по группам региона, слова по интервалу и доли по типам; рядом — то же против C (случайные
   МО того же размера из других регионов) и против нового набора R (случайные МО своего региона). Ошибки B и D
   пересчитываются заново из ``context_annual`` и составов наборов (``node_comparable.csv``) и сверяются
   с выгрузкой этапа 5 (код 3). Пример — МО с разностью ошибок, ближайшей к медиане разности.
2. **Флаг устойчивости типа МО** (``node_seed.csv``, ``node_r1.csv``): «тип устойчив» — узел есть во всех
   вариантах R1 и повторах seed и везде тот же тип; тот же тип везде, где узел есть, но вариантов меньше —
   «проверен в N из 3 вариантов»; иначе — «тип зависит от варианта расчёта».

Выходы — ``<paths.outputs>/usefulness/``: ``facts.json``, ``rule_by_mo.csv``, ``mo_flags.csv``. После них —
проверка ``usefulness.by_type_test`` (модуль ``munnet.usefulness_by_type``) в свои файлы ``by_type.json``,
``by_type_runs.csv``, ``by_type_by_mo.csv``.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from munnet.config import Config
from munnet.contracts import MissingInputError, QCError

log = logging.getLogger(__name__)

# Что реализовано: значения ключей блока usefulness, записанные до расчёта. Другое значение — код 3, а не
# молчаливое исполнение другого правила.
EXPECTED: dict[tuple[str, ...], Any] = {
    ("rule_share", "source"): "outputs/interpret/t7_errors.csv",
    ("rule_share", "target"): "target_abs",
    ("rule_share", "rule_set"): "B",
    ("rule_share", "against"): "D",
    ("rule_share", "also_against"): ["C", "R"],
    ("rule_share", "groups"): "region_group",
    ("rule_share", "works"): "strict_less",
    ("rule_share", "share"): "works_over_n",
    ("rule_share", "gain"): "median_paired",
    ("rule_share", "per_mo"): "not_shown",
    ("rule_share", "by_type"): True,
    ("example", "rule"): "median_paired_diff",
    ("example", "keep"): "interpret.tests.T7_utility.example",
    ("type_flag", "stable"): "all_runs",
    ("type_flag", "also"): "majority",
    ("type_flag", "breakdown"): ["seeds_only", "per_variant", "all"],
    ("type_flag", "by_type"): True,
    ("type_flag", "display"): "separate_counts",
}
YEARS = (2023, 2024)  # interpret.windows.change_years — сверяется в run()
TARGET_COL = "retail_pc"  # interpret.tests.T7_utility.target_abs.source = context_annual.retail_pc
FLAG_WORDS = {
    "stable": "тип устойчив",
    "partial_check": "проверен в {n} из {m} вариантов",
    "depends": "тип зависит от варианта расчёта",
}


def _get(block: Mapping, path: tuple[str, ...]) -> Any:
    node: Any = block
    for k in path:
        node = node.get(k) if isinstance(node, Mapping) else None
    return node


def check_spec(block: Mapping) -> None:
    """Ключи блока ``usefulness`` равны записанным до расчёта (``EXPECTED``); иначе ``QCError``."""
    bad = [
        f"{'.'.join(p)} = {_get(block, p)!r} (ожидалось {v!r})"
        for p, v in EXPECTED.items()
        if _get(block, p) != v
    ]
    if bad:
        raise QCError("usefulness: блок конфига не совпадает с реализованным правилом: " + "; ".join(bad))


# --- Сверка с соседями по своему региону ----------------------------------------------------------------


def target_change(context: pd.DataFrame, years: tuple[int, int] = YEARS) -> pd.Series:
    """Цель target_abs T7: ln(1 + retail_pc) года b минус года a, индекс — territory_id (пропуск — NaN)."""
    ya, yb = years
    c = context.loc[context["year"].isin([ya, yb]), ["territory_id", "year", TARGET_COL]]
    w = c.pivot(index="territory_id", columns="year", values=TARGET_COL)
    return (np.log1p(w[yb].astype(float)) - np.log1p(w[ya].astype(float))).rename("y")


def set_medians(y: pd.Series, comparable: pd.DataFrame, set_name: str) -> pd.Series:
    """Медиана цели членов набора ``set_name`` у каждого МО (члены — только с известной целью)."""
    m = comparable.loc[comparable["set"] == set_name, ["territory_id", "other_id"]].copy()
    m["y"] = y.reindex(m["other_id"].to_numpy()).to_numpy()
    m = m.dropna(subset=["y"])
    return m.groupby("territory_id")["y"].median()


def recompute_errors(y: pd.Series, comparable: pd.DataFrame, ids: np.ndarray, set_name: str) -> np.ndarray:
    """Ошибка МО = |y − медиана y набора| (interpret.tests.T7_utility.error = abs_median)."""
    med = set_medians(y, comparable, set_name).reindex(ids).to_numpy()
    return np.abs(y.reindex(ids).to_numpy() - med)


def region_random_errors(
    y: pd.Series,
    ids: np.ndarray,
    sizes: np.ndarray,
    group_of: pd.Series,
    draws: int,
    seed: int,
) -> np.ndarray:
    """Набор R: у каждого МО ``ids`` — ``sizes`` случайных МО своей группы региона с известной целью
    (без самого МО), ``draws`` розыгрышей; ошибка — медиана по розыгрышам |y − медиана y набора| (как у C)."""
    rng = np.random.default_rng(seed)
    known = y.dropna()
    g_known = group_of.reindex(known.index)
    pools = {g: known.index[(g_known == g).to_numpy()].to_numpy() for g in pd.unique(g_known.dropna())}
    out = np.empty(len(ids))
    for k, (i, size) in enumerate(zip(ids, sizes, strict=True)):
        pool = pools[group_of.loc[i]]
        pool = pool[pool != i]
        m = int(min(size, len(pool)))
        if m < 1:
            raise QCError(f"usefulness: у МО {i} в своей группе региона нет других МО с известной целью")
        vals = known.reindex(pool).to_numpy()
        errs = [abs(y.loc[i] - np.median(rng.choice(vals, m, replace=False))) for _ in range(draws)]
        out[k] = float(np.median(errs))
    return out


def group_bootstrap(
    groups: np.ndarray, stats: Mapping[str, Callable[[np.ndarray], float]], n_boot: int, seed: int
) -> dict[str, np.ndarray]:
    """Бутстрап по группам: группы целиком, с возвращением; ``stats`` — имя -> функция индексов МО."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    members = [np.flatnonzero(groups == g) for g in uniq]
    out = {k: np.empty(n_boot) for k in stats}
    for b in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([members[i] for i in pick])
        for k, f in stats.items():
            out[k][b] = f(idx)
    return out


def ci(draws: np.ndarray, level: float) -> list[float]:
    a = (1.0 - level) / 2.0
    d = draws[np.isfinite(draws)]
    return [float(np.quantile(d, a)), float(np.quantile(d, 1.0 - a))]


def share_words(interval: list[float], words: Mapping[str, str]) -> str:
    """Слова о доле по интервалу: нижняя граница выше 0,5 — ``more_often``, верхняя ниже — ``less_often``."""
    lo, hi = interval
    if lo > 0.5:
        return str(words["more_often"])
    if hi < 0.5:
        return str(words["less_often"])
    return str(words["about_half"])


def gain_words(interval: list[float], words: Mapping[str, str]) -> str:
    """Слова о медиане разности d = ошибка B − ошибка D по её интервалу."""
    lo, hi = interval
    if hi < 0:
        return str(words["b_better"])
    if lo > 0:
        return str(words["d_better"])
    return str(words["no_difference"])


def pick_example(ids: np.ndarray, d: np.ndarray) -> int:
    """МО с d ближе всего к медиане d; при равенстве — меньший territory_id."""
    dist = np.abs(d - np.median(d))
    return int(np.min(ids[dist == dist.min()]))


def percentile_of(values: np.ndarray, v: float) -> float:
    """Доля МО с ошибкой не больше ``v`` (процентиль примера среди всех МО общего набора)."""
    return float(np.mean(values <= v))


# --- Флаг устойчивости типа МО ------------------------------------------------------------------------


def type_flags(node_seed: pd.DataFrame, typed_ids: np.ndarray, n_variants: int, n_seeds: int) -> pd.DataFrame:
    """Счёт вариантов R1 и повторов seed по узлам с типом и флаг: ``stable`` — узел есть во всех
    ``n_variants`` вариантах и ``n_seeds`` повторах и везде тот же тип; ``partial_check`` — тот же тип везде,
    где узел есть, но вариантов или повторов меньше; ``depends`` — иначе. ``majority`` — тот же тип во всех
    повторах и больше чем в половине вариантов, где узел есть (описание). Нет строки узла — ``QCError``."""
    p = node_seed.pivot(index="territory_id", columns="kind", values=["n_same", "n_runs"])
    p.columns = [f"{kind}_{v.replace('n_', '')}" for v, kind in p.columns]
    p = p.reindex(typed_ids)
    need = ["seed_same", "seed_runs", "variant_same", "variant_runs"]
    missing = [c for c in need if c not in p.columns]
    if missing or p[need].isna().any().any():
        raise QCError(
            f"usefulness: node_seed.csv — нет счёта для части узлов с типом ({missing or 'пропуски'})"
        )
    p = p[need].astype(np.int64)
    if (p["seed_runs"] < 1).any() or (p["variant_runs"] < 1).any():
        raise QCError("usefulness: node_seed.csv — у узла нет ни одного варианта или повтора")
    if (p["seed_runs"] > n_seeds).any() or (p["variant_runs"] > n_variants).any():
        raise QCError("usefulness: node_seed.csv — повторов или вариантов больше, чем записано в блоке")
    seeds_all = p["seed_same"] == p["seed_runs"]
    all_same = seeds_all & (p["variant_same"] == p["variant_runs"])
    full = (p["seed_runs"] == n_seeds) & (p["variant_runs"] == n_variants)
    p["flag"] = np.where(all_same & full, "stable", np.where(all_same, "partial_check", "depends"))
    p["stable"] = p["flag"] == "stable"
    p["majority"] = seeds_all & (2 * p["variant_same"] > p["variant_runs"])
    return p.reset_index().rename(columns={"index": "territory_id"})


def flag_text(row: Mapping, n_variants: int) -> str:
    return FLAG_WORDS[str(row["flag"])].format(n=int(row["variant_runs"]), m=n_variants)


# --- Этап ----------------------------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_inputs(cfg: Config) -> dict[str, Any]:
    out = Path(cfg["paths"]["outputs"]) / "interpret"
    processed = Path(cfg["paths"]["processed"])
    paths = {
        "facts": out / "facts.json",
        "t7_errors": out / "t7_errors.csv",
        "node_comparable": out / "node_comparable.csv",
        "node_seed": out / "node_seed.csv",
        "node_r1": out / "node_r1.csv",
        "types": out / "types.csv",
        "context": processed / "context_annual.parquet",
        "nodes": processed / "features_nodes.parquet",
    }
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise MissingInputError(f"usefulness: нет входов {missing} — сначала этап interpret")
    facts = json.loads(paths["facts"].read_text(encoding="utf-8"))
    if facts.get("status") != "done" or facts.get("blind"):
        raise QCError("usefulness: facts.json этапа interpret не завершён или от слепого прогона")
    return {
        "paths": paths,
        "facts": facts,
        "errors": pd.read_csv(paths["t7_errors"]),
        "comparable": pd.read_csv(paths["node_comparable"]),
        "seed": pd.read_csv(paths["node_seed"]),
        "r1": pd.read_csv(paths["node_r1"]),
        "types": pd.read_csv(paths["types"]),
        "context": pd.read_parquet(paths["context"], columns=["territory_id", "year", TARGET_COL]),
        "nodes": pd.read_parquet(
            paths["nodes"], columns=["territory_id", "name", "name_short", "region_name", "region_group"]
        ),
    }


def _qc_recompute(inp: Mapping[str, Any], e: pd.DataFrame, y: pd.Series, tol: float, product: str) -> None:
    t7 = inp["facts"]["t7"]
    if str(t7.get("product")) != product:
        raise QCError(
            f"usefulness: набор продукта T7 — {t7.get('product')!r}, а правило записано против {product!r}"
        )
    ids = e["territory_id"].to_numpy(dtype=np.int64)
    if len(ids) != int(t7["n_common"]):
        raise QCError(f"usefulness: в общем наборе {len(ids)} МО, в facts.t7.n_common — {t7['n_common']}")
    for s in ("B", "D"):
        rec = recompute_errors(y, inp["comparable"], ids, s)
        dev = float(np.nanmax(np.abs(rec - e[f"err_abs_{s}"].to_numpy()))) if len(ids) else 0.0
        if np.isnan(rec).any() or not dev <= tol:
            raise QCError(f"usefulness: пересчёт ошибок набора {s} расходится с t7_errors.csv (до {dev:.3g})")
    for s in ("B", "C", "D"):
        med = float(np.median(e[f"err_abs_{s}"]))
        if abs(med - float(t7["median_error_abs"][s])) > tol:
            raise QCError(f"usefulness: медиана ошибок {s} {med} ≠ facts.t7.median_error_abs[{s}]")


def rule_share(inp: Mapping[str, Any], block: Mapping, seed: int) -> tuple[dict, pd.DataFrame]:
    """Доля случаев, где соседи по региону (B) ошибаются меньше похожих по тратам (D); рядом — C и R."""
    rs = block["rule_share"]
    e = inp["errors"]
    e = e.loc[e["common"].astype(bool)].sort_values("territory_id").reset_index(drop=True)
    ids = e["territory_id"].to_numpy(dtype=np.int64)
    y = target_change(inp["context"])
    _qc_recompute(inp, e, y, float(rs["qc_recompute"]), str(rs["against"]))

    group_of = inp["nodes"].set_index("territory_id")["region_group"]
    groups = group_of.reindex(ids).to_numpy()
    if pd.isna(groups).any():
        raise QCError("usefulness: у части МО общего набора нет группы региона в features_nodes")
    sizes_b = (
        inp["comparable"]
        .loc[inp["comparable"]["set"] == "B"]
        .assign(known=lambda m: y.reindex(m["other_id"].to_numpy()).notna().to_numpy())
        .query("known")
        .groupby("territory_id")
        .size()
        .reindex(ids)
        .to_numpy()
    )
    er = region_random_errors(y, ids, sizes_b, group_of, int(rs["random_draws"]), seed)
    types = inp["types"].set_index("territory_id")["type"].reindex(ids).fillna(0).to_numpy(dtype=np.int64)
    eb, ed, ec = (e[f"err_abs_{s}"].to_numpy() for s in ("B", "D", "C"))
    d = eb - ed
    works = {"D": eb < ed, "C": eb < ec, "R": eb < er}
    delta = float(rs["delta"])
    type_ids = [int(t) for t in sorted(set(types.tolist())) if t > 0]
    stats: dict[str, Callable[[np.ndarray], float]] = {
        f"share_{k}": (lambda i, w=w: w[i].mean()) for k, w in works.items()
    }
    stats["gain"] = lambda i: float(np.median(d[i]))
    for t in type_ids:
        stats[f"type_{t}"] = lambda i, t=t: (
            works["D"][i][types[i] == t].mean() if (types[i] == t).any() else np.nan
        )
    n_boot, level = int(rs["bootstrap"]), float(rs["ci_level"])
    draws = group_bootstrap(groups, stats, n_boot, seed)

    def share(k: str) -> dict:
        interval = ci(draws[f"share_{k}"], level)
        return {
            "works": int(works[k].sum()),
            "ties": int((eb == {"D": ed, "C": ec, "R": er}[k]).sum()),
            "share": float(works[k].mean()),
            "share_ci": interval,
            "words": share_words(interval, rs["words"]),
        }

    gain_ci = ci(draws["gain"], level)
    res = {
        "n": len(ids),
        "n_groups": int(len(np.unique(groups))),
        "reference": str(rs["reference_always"]),
        "vs_D": share("D"),
        "vs_C": share("C"),
        "vs_R": share("R"),
        "delta": {
            "value": delta,
            "b_better": int((d < -delta).sum()),
            "within": int((np.abs(d) <= delta).sum()),
            "d_better": int((d > delta).sum()),
        },
        "gain_median": float(np.median(d)),
        "gain_ci": gain_ci,
        "gain_words": gain_words(gain_ci, rs["gain_words"]),
        "median_error_abs": {
            **{s: float(np.median(e[f"err_abs_{s}"])) for s in ("B", "C", "D")},
            "R": float(np.median(er)),
        },
        "bootstrap": {"n": n_boot, "level": level, "by": "region_group", "seed": seed},
        "random_draws": int(rs["random_draws"]),
        "by_type": [],
    }
    for t in type_ids:
        m = types == t
        interval = ci(draws[f"type_{t}"], level)
        res["by_type"].append(
            {
                "type": t,
                "n": int(m.sum()),
                "works": int(works["D"][m].sum()),
                "share": float(works["D"][m].mean()),
                "share_ci": interval,
                "words": share_words(interval, rs["words"]),
            }
        )
    res["edit"] = str(rs["edits"][_word_key(res["vs_D"]["words"], rs["words"])])
    if res["vs_R"]["words"] != rs["words"]["more_often"]:
        res["edit_R"] = str(rs["edits"]["vs_R_not_more_often"])
    table = pd.DataFrame(
        {
            "territory_id": ids,
            "type": types,
            "region_group": groups.astype(np.int64),
            "err_abs_B": eb,
            "err_abs_D": ed,
            "err_abs_C": ec,
            "err_abs_R": er,
            "d": d,
            "works": works["D"],
            "works_vs_C": works["C"],
            "works_vs_R": works["R"],
        }
    )
    return res, table


def _word_key(word: str, words: Mapping[str, str]) -> str:
    return next(k for k, v in words.items() if v == word)


def example(inp: Mapping[str, Any], table: pd.DataFrame) -> dict:
    """Пример пользы: МО с разностью ошибок, ближайшей к медиане разности, — изменение оборота у него
    и медианы изменения у соседей по региону и у похожих по тратам (в процентах), ошибки и их процентили,
    соседи по региону с км; рядом — пример этапа 5 по той же цели."""
    ids = table["territory_id"].to_numpy(dtype=np.int64)
    tid = pick_example(ids, table["d"].to_numpy())
    y = target_change(inp["context"])
    comp = inp["comparable"]
    nodes = inp["nodes"].set_index("territory_id")
    types = inp["types"].set_index("territory_id")
    tab = table.set_index("territory_id")
    row = tab.loc[tid]

    def who(i: int) -> dict:
        n = nodes.loc[i]
        name = n["name_short"] if pd.notna(n["name_short"]) else n["name"]
        return {"territory_id": int(i), "name": str(name), "region": str(n["region_name"])}

    def members(s: str) -> list[dict]:
        m = comp.loc[(comp["territory_id"] == tid) & (comp["set"] == s)].sort_values("rank")
        return [
            {**who(int(r.other_id)), "km": float(r.km), "known": bool(pd.notna(y.get(int(r.other_id))))}
            for r in m.itertuples()
        ]

    def errs(i: int) -> dict:
        r = tab.loc[i]
        return {
            "err_B": float(r["err_abs_B"]),
            "err_D": float(r["err_abs_D"]),
            "pct_B": percentile_of(table["err_abs_B"].to_numpy(), float(r["err_abs_B"])),
            "pct_D": percentile_of(table["err_abs_D"].to_numpy(), float(r["err_abs_D"])),
            "works": bool(r["works"]),
        }

    t5 = int(inp["facts"]["t7"]["example"]["territory_id"])
    return {
        **who(tid),
        "type": int(types.loc[tid, "type"]),
        "type_name": str(types.loc[tid, "name"]),
        "own_change": float(np.expm1(y.loc[tid])),
        "median_B_change": float(np.expm1(set_medians(y, comp, "B").loc[tid])),
        "median_D_change": float(np.expm1(set_medians(y, comp, "D").loc[tid])),
        **errs(tid),
        "d": float(row["d"]),
        "members_B": members("B"),
        "members_D": members("D"),
        "stage5_example": {**who(t5), **(errs(t5) if t5 in tab.index else {}), "target": "target_abs"},
    }


def flags_summary(flags: pd.DataFrame, r1: pd.DataFrame) -> dict:
    """Доли флага в целом и по типам; разбивка «тот же тип»: только повторы, каждый вариант, всё вместе."""

    def part(f: pd.DataFrame) -> dict:
        return {
            "n": len(f),
            **{k: int((f["flag"] == k).sum()) for k in FLAG_WORDS},
            "stable_share": float(f["stable"].mean()),
            "majority": int(f["majority"].sum()),
            "majority_share": float(f["majority"].mean()),
            "seeds_only_share": float((f["seed_same"] == f["seed_runs"]).mean()),
        }

    typed = set(flags["territory_id"].astype(int))
    v = r1.loc[(r1["kind"] == "variant") & r1["territory_id"].astype(int).isin(typed)]
    per_variant = [
        {"variant": str(name), "n": len(g), "same_share": float(g["same"].astype(bool).mean())}
        for name, g in v.groupby("variant", sort=True)
    ]
    return {
        **part(flags),
        "per_variant": per_variant,
        "by_type": [{"type": int(t), **part(g)} for t, g in flags.groupby("type")],
    }


def run(cfg: Config) -> dict:
    block = cfg["usefulness"]
    check_spec(block)
    if tuple(int(v) for v in cfg["interpret"]["windows"]["change_years"]) != YEARS:
        raise QCError("usefulness: interpret.windows.change_years не 2023, 2024 — цель T7 другая")
    rob = cfg["interpret"]["robustness"]
    seed = int(cfg["seed"])
    n_variants, n_seeds = int(block["type_flag"]["variants"]), int(block["type_flag"]["seeds"])
    if n_variants != len(rob["variants"]) or n_seeds != len([s for s in rob["seeds"] if int(s) != seed]):
        raise QCError("usefulness: type_flag.variants/seeds не совпадают с interpret.robustness")
    inp = _read_inputs(cfg)
    rule, table = rule_share(inp, block, seed)
    ex = example(inp, table)
    types = inp["types"]
    typed = types.loc[types["type"] > 0, ["territory_id", "type"]]
    flags = type_flags(inp["seed"], typed["territory_id"].to_numpy(dtype=np.int64), n_variants, n_seeds)
    flags = flags.merge(typed, on="territory_id", how="left")
    flags["flag_text"] = [flag_text(r, n_variants) for r in flags.to_dict("records")]
    facts = {
        "status": "done",
        "note": "разведка после вскрытия: правила записаны до расчёта (блок usefulness), вне вердиктов",
        "label": str(block["type_flag"]["label"]),
        "rule": rule,
        "example": ex,
        "type_flag": flags_summary(flags, inp["r1"]),
        "inputs_sha256": {k: _sha256(p) for k, p in inp["paths"].items()},
    }
    out = cfg.dir("outputs") / "usefulness"
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "rule_by_mo.csv", index=False)
    cols = [
        "territory_id",
        "type",
        "seed_same",
        "seed_runs",
        "variant_same",
        "variant_runs",
        "flag",
        "flag_text",
    ]
    flags[[*cols, "majority"]].to_csv(out / "mo_flags.csv", index=False)
    (out / "facts.json").write_text(json.dumps(facts, ensure_ascii=False, indent=1), encoding="utf-8")
    r = rule["vs_D"]
    log.info(
        "usefulness: соседи по региону ближе похожих по тратам в %d из %d случаев "
        "(%.3f, интервал %.3f–%.3f: %s); пример — %s (%s); тип устойчив у %d из %d узлов",
        r["works"], rule["n"], r["share"], *r["share_ci"], r["words"], ex["name"], ex["region"],
        facts["type_flag"]["stable"], facts["type_flag"]["n"],
    )  # fmt: skip
    # Проверка usefulness.by_type_test (предрегистрация 01.10, f744563) — после существующих расчётов, в свои
    # файлы by_type*.json/csv; facts.json, rule_by_mo.csv и mo_flags.csv выше уже записаны и не меняются.
    from munnet import usefulness_by_type

    usefulness_by_type.run(cfg, rule["by_type"])
    return facts
