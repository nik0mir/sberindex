"""Разведка ``usefulness.size_posthoc``: разница в пользе сверки со своим регионом — про размер МО или тип.

РАЗВЕДКА ПОСЛЕ ВСКРЫТИЯ, 02.10.2026: блок записан после того, как увидели исход ``usefulness.by_type_test``
(confirmed во всех 8 прогонах) и черновые расчёты devils-advocate (da1–da8). Не проверка: вердиктов нет,
числа и p — описание. Порог «крупные = верхний квинтиль населения» выбран участником до этого кода
по черновым числам.

Вызывается из ``munnet.usefulness.run`` после ``usefulness_by_type.run`` и пишет только свои файлы:
``<paths.outputs>/usefulness/size_check.json`` и ``size_by_mo.csv`` (узлы с типом и известным населением:
население, квинтиль по границам общего набора T7, флаг ``large``). Прежние выходы этапа не трогает.

Что считается (цели, наборы B и D, ``min_set`` и «B ближе D» — функциями ``munnet.usefulness_by_type``):

1. **Квинтили населения** ``pop_avg`` 2023 года по 1542 МО общего набора T7; границы публикуются.
2. **Доля «B ближе D»** по квинтилям — средняя по трём целям общая доля по МО группы (решающая статистика
   by_type_test) с интервалом бутстрапа по группам региона; рядом розница.
3. **Крупные против остальных** (верхний квинтиль): Δ, интервал, p перестановок метки внутри групп региона.
4. **Тип при равном размере:** Δ {3,4} − {1,2} внутри квинтилей (гармонические веса), p перестановок типов
   внутри ячеек «группа региона × квинтиль». **Размер при равном типе:** Δ крупных внутри групп типов,
   p перестановок квинтилей внутри ячеек «группа региона × группа типов».
5. **Бутстрап с общими членами D:** одна выборка групп региона взвешивает и МО, и членов D (медиана D — по
   членам, повторённым столько раз, сколько взята их группа) — для основной Δ by_type_test и для Δ крупных.
6. **Пересечение покрытий:** то же описание на МО, у которых известны все три цели.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from munnet import usefulness_by_type as BT
from munnet.config import Config
from munnet.contracts import MissingInputError, QCError

log = logging.getLogger(__name__)

# Ключи подблока size_posthoc, которые код исполняет буквально. Другое значение — код 3. Тексты (seen_before)
# не сверяются.
EXPECTED: dict[tuple[str, ...], Any] = {
    ("status",): "posthoc_exploration",
    ("size", "source"): "context_annual.pop_avg",
    ("size", "year"): 2023,
    ("quintiles", "n"): 5,
    ("quintiles", "base"): "t7_common",
    ("quintiles", "large"): "top",
    ("targets",): "by_type_test",
    ("statistic", "share"): "mean_of_three",
    ("statistic", "seen"): "retail_describe",
    ("statistic", "by_quintile"): True,
    ("statistic", "bootstrap"): {"n": 2000, "by": "region_group", "empty_group": "drop"},
    ("statistic", "ci_level"): 0.95,
    ("statistic", "large_vs_rest"): {"permutation": {"within": "region_group", "n": 2000}},
    ("statistic", "type_given_size"): {
        "weights": "harmonic",
        "permutation": {"within": "region_group_x_quintile", "n": 2000},
    },
    ("statistic", "size_given_type"): {
        "weights": "harmonic",
        "permutation": {"within": "region_group_x_type_group", "n": 2000},
    },
    ("statistic", "p_less"): "(#{Δ_perm ≤ Δ} + 1) / (n + 1)",
    ("shared_members_bootstrap", "n"): 2000,
    ("shared_members_bootstrap", "statistics"): ["types_main", "large_vs_rest"],
    ("all_three_targets",): "describe",
    ("qc", "types_main_vs_by_type"): 1.0e-12,
    ("qc", "spec_frozen"): True,
}
POP_COL = "pop_avg"  # size.source без префикса «context_annual.»
TARGETS = BT.TARGETS  # общепит, доход 5-НДФЛ, отгрузка — решающие в by_type_test
SEEN = BT.SEEN  # розница — описание рядом


def check_spec(block: Mapping | None) -> None:
    """Ключи подблока ``size_posthoc`` равны записанным (``EXPECTED``); иначе ``QCError``."""
    if not isinstance(block, Mapping):
        raise QCError("usefulness.size_posthoc: подблока нет в конфиге")
    bad = [
        f"{'.'.join(p)} = {BT._get(block, p)!r} (ожидалось {v!r})"
        for p, v in EXPECTED.items()
        if BT._get(block, p) != v
    ]
    if bad:
        raise QCError("usefulness.size_posthoc: блок конфига не совпадает с кодом: " + "; ".join(bad))


# --- Квинтили -----------------------------------------------------------------------------------------------


def quantile_bounds(values: np.ndarray, n_q: int) -> np.ndarray:
    """Внутренние границы n_q квантильных групп (n_q − 1 чисел), ``np.quantile`` с линейной интерполяцией —
    те же, что у ``pandas.qcut``."""
    v = np.asarray(values, dtype=float)
    if not np.isfinite(v).all():
        raise QCError("usefulness.size_posthoc: у части МО базы нет населения")
    return np.quantile(v, np.arange(1, n_q) / n_q)


def assign_quantile(values: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    """Номер группы 0…len(bounds): значение на границе — в нижнюю группу (правый край включён, как ``qcut``);
    пропуск — −1."""
    v = np.asarray(values, dtype=float)
    q = np.searchsorted(np.asarray(bounds, dtype=float), v, side="left")
    return np.where(np.isfinite(v), q, -1).astype(np.int64)


def cell_codes(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Код ячейки пары (a, b) — номер пары по возрастанию; для перестановок внутри ячеек."""
    pairs = np.stack([np.asarray(a, dtype=np.int64), np.asarray(b, dtype=np.int64)], axis=1)
    _, inv = np.unique(pairs, axis=0, return_inverse=True)
    return inv.ravel().astype(np.int64)


# --- Статистики (по одной цели — функции usefulness_by_type, затем среднее по целям) ------------------------


def mean_share(
    S: BT.Summer, member: np.ndarray, data: Mapping[str, Mapping[str, np.ndarray]], targets: Sequence[str]
) -> np.ndarray:
    """Средняя по целям общая доля «B ближе D» у МО группы ``member``."""
    return np.mean([BT.group_share(S, member, data[t]["works"], data[t]["mask"])[0] for t in targets], axis=0)


def mean_delta(
    S: BT.Summer,
    hi: np.ndarray,
    lo: np.ndarray,
    data: Mapping[str, Mapping[str, np.ndarray]],
    targets: Sequence[str],
) -> np.ndarray:
    """Средняя по целям Δ_t = доля у ``hi`` минус доля у ``lo`` (``usefulness_by_type.share_diff``)."""
    return np.mean(
        [BT.share_diff(S, hi, lo, data[t]["works"], data[t]["mask"])["delta"] for t in targets], axis=0
    )


def mean_strat_delta(
    S: BT.Summer,
    hi: np.ndarray,
    lo: np.ndarray,
    data: Mapping[str, Mapping[str, np.ndarray]],
    targets: Sequence[str],
    strata: Sequence[np.ndarray],
) -> np.ndarray:
    """Средняя по целям Δ внутри страт с гармоническими весами (``usefulness_by_type.status_delta``)."""
    return np.mean(
        [BT.status_delta(S, hi, lo, data[t]["works"], data[t]["mask"], strata) for t in targets], axis=0
    )


def empty_draws(
    S: BT.Summer,
    hi: np.ndarray,
    lo: np.ndarray,
    data: Mapping[str, Mapping[str, np.ndarray]],
    targets: Sequence[str],
) -> np.ndarray:
    """Выборки бутстрапа, где у какой-либо цели пуста группа ``hi`` или ``lo`` (отбрасываются)."""
    bad: np.ndarray | None = None
    for t in targets:
        d = BT.share_diff(S, hi, lo, data[t]["works"], data[t]["mask"])
        e = np.atleast_1d((d["n_hi"] == 0) | (d["n_lo"] == 0))
        bad = e if bad is None else bad | e
    if bad is None:
        raise QCError("usefulness.size_posthoc: нет целей")
    return bad


def _sd(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.std(x, ddof=1)) if len(x) > 1 else float("nan")


def perm_summary(null: np.ndarray, obs: float) -> dict[str, float]:
    """p_less и p_greater (``usefulness_by_type.perm_p``), среднее и SD нуля."""
    null = np.asarray(null, dtype=float)
    p = BT.perm_p(null, obs)
    fin = null[np.isfinite(null)]
    return {
        "p_less": p["p_less"],
        "p_greater": p["p_greater"],
        "n_undefined": p["n_undefined"],
        "null_mean": float(np.mean(fin)) if len(fin) else float("nan"),
        "null_sd": _sd(fin),
        "n_perm": int(len(null)),
    }


# --- Бутстрап с общими членами D ----------------------------------------------------------------------------


def d_member_matrix(
    y: pd.Series,
    comparable: pd.DataFrame,
    ids: np.ndarray,
    member_group: pd.Series,
    group_codes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Члены D с известной целью у каждого МО ``ids``: матрица значений цели (строка — МО, по возрастанию,
    хвост — NaN) и матрица индексов групп региона членов: 0…G−1 — номер группы в ``group_codes`` (группы
    общего набора, пересыпаются), G — группа вне общего набора (вес 1), G + 1 — пустая ячейка (вес 0)."""
    n_g = len(group_codes)
    m = comparable.loc[comparable["set"] == "D", ["territory_id", "other_id"]].copy()
    m["y"] = y.reindex(m["other_id"].to_numpy()).to_numpy()
    m = m.dropna(subset=["y"])
    m = m.loc[m["territory_id"].isin(ids)]
    pos = {int(g): i for i, g in enumerate(group_codes)}
    grp = member_group.reindex(m["other_id"].to_numpy()).to_numpy()
    m["gi"] = [pos.get(int(g), n_g) if pd.notna(g) else n_g for g in grp]
    m = m.sort_values(["territory_id", "y"], kind="mergesort")
    row = {int(t): i for i, t in enumerate(ids)}
    m["row"] = m["territory_id"].map(row).to_numpy()
    m["col"] = m.groupby("territory_id").cumcount().to_numpy()
    k = int(m["col"].max()) + 1 if len(m) else 1
    Y = np.full((len(ids), k), np.nan)
    G = np.full((len(ids), k), n_g + 1, dtype=np.int64)
    Y[m["row"].to_numpy(), m["col"].to_numpy()] = m["y"].to_numpy()
    G[m["row"].to_numpy(), m["col"].to_numpy()] = m["gi"].to_numpy()
    return Y, G


def weighted_median_rows(Y: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Медиана каждой строки, где значение ``Y[i, j]`` повторено ``W[i, j]`` раз (целые веса); строки ``Y`` —
    по возрастанию. Равна ``np.median(np.repeat(Y[i], W[i]))``; сумма весов 0 — NaN."""
    W = np.asarray(W, dtype=np.int64)
    T = W.sum(axis=1)
    C = np.cumsum(W, axis=1)
    p1, p2 = (T - 1) // 2, T // 2
    i1 = np.argmax(C > p1[:, None], axis=1)
    i2 = np.argmax(C > p2[:, None], axis=1)
    r = np.arange(len(Y))
    med = (Y[r, i1] + Y[r, i2]) / 2.0
    return np.where(T > 0, med, np.nan)


def shared_bootstrap(
    prep: Mapping[str, Mapping[str, np.ndarray]],
    gidx: np.ndarray,
    counts: np.ndarray,
    pairs: Mapping[str, tuple[np.ndarray, np.ndarray]],
    targets: Sequence[str],
) -> dict[str, np.ndarray]:
    """Бутстрап с общими членами D: в выборке b вес МО — сколько раз взята его группа, вес члена D — сколько
    раз взята группа члена (вне общего набора — 1). ``prep[цель]``: ``mask``, ``own`` (y МО), ``err_B``,
    ``Y``, ``G`` (``d_member_matrix``). ``pairs[имя] = (hi, lo)``. Возвращает по имени массив Δ (среднее
    по целям) по выборкам;
    выборка, где у какой-либо цели пуста одна из групп, — NaN."""
    n_b, n_g = counts.shape
    out = {k: np.full(n_b, np.nan) for k in pairs}
    for b in range(n_b):
        ext = np.concatenate([counts[b], [1.0, 0.0]]).astype(np.int64)
        w_mo = counts[b][gidx]
        deltas: dict[str, list[float]] = {k: [] for k in pairs}
        for t in targets:
            p = prep[t]
            med = weighted_median_rows(p["Y"], ext[p["G"]])
            err_d = np.abs(p["own"] - med)
            ok = p["mask"] & np.isfinite(err_d) & (w_mo > 0)
            works = ok & (p["err_B"] < np.where(np.isfinite(err_d), err_d, -np.inf))
            wt = w_mo * ok
            for k, (hi, lo) in pairs.items():
                n_hi, n_lo = (wt * hi).sum(), (wt * lo).sum()
                if n_hi > 0 and n_lo > 0:
                    deltas[k].append((wt * works * hi).sum() / n_hi - (wt * works * lo).sum() / n_lo)
                else:
                    deltas[k].append(np.nan)
        for k in pairs:
            out[k][b] = np.mean(deltas[k])  # NaN у одной цели — NaN у среднего (выборка отбрасывается)
    return out


# --- Сборка -------------------------------------------------------------------------------------------------


def _read(cfg: Config) -> dict[str, Any]:
    out = cfg.dir("outputs") / "interpret"
    processed = cfg.dir("processed")
    paths = {
        "t7_errors": out / "t7_errors.csv",
        "node_comparable": out / "node_comparable.csv",
        "types": out / "types.csv",
        "context": processed / "context_annual.parquet",
        "nodes": processed / "features_nodes.parquet",
    }
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise MissingInputError(f"usefulness.size_posthoc: нет входов {missing} — сначала этап interpret")
    cols = ["territory_id", "year", *dict.fromkeys(BT.COLUMNS.values()), BT.NDFL_FILTER, POP_COL]
    return {
        "errors": pd.read_csv(paths["t7_errors"]),
        "comparable": pd.read_csv(paths["node_comparable"]),
        "types": pd.read_csv(paths["types"]),
        "context": pd.read_parquet(paths["context"], columns=cols),
        "nodes": pd.read_parquet(paths["nodes"], columns=["territory_id", "region_group"]),
    }


def _ci(draws: np.ndarray, level: float) -> list[float]:
    return BT.ci(np.asarray(draws, dtype=float), level)


def _group_block(
    S_b: BT.Summer,
    member: np.ndarray,
    data: Mapping[str, Mapping[str, np.ndarray]],
    level: float,
) -> dict[str, Any]:
    """Доля «B ближе D» группы: среднее по трём целям и по каждой цели (с розницей) с интервалами."""
    per = {}
    for t in (*TARGETS, SEEN):
        d = data[t]
        s, n = BT.group_share(BT.point_sum, member, d["works"], d["mask"])
        sb, _ = BT.group_share(S_b, member, d["works"], d["mask"])
        per[t] = {"n": int(n), "works": int((member & d["mask"] & d["works"]).sum()), "share": float(s),
                  "share_ci": _ci(sb, level)}  # fmt: skip
    return {
        "share": float(mean_share(BT.point_sum, member, data, TARGETS)),
        "share_ci": _ci(mean_share(S_b, member, data, TARGETS), level),
        "per_target": per,
    }


def _delta_block(
    S_b: BT.Summer,
    hi: np.ndarray,
    lo: np.ndarray,
    data: Mapping[str, Mapping[str, np.ndarray]],
    perm_hi: np.ndarray | None,
    perm_lo: np.ndarray | None,
    level: float,
) -> dict[str, Any]:
    """Δ = доля у ``hi`` минус у ``lo``, среднее по трём целям: точка, интервал бутстрапа (выборки с пустой
    группой
    отброшены), по целям и розница; при ``perm_hi`` — p перестановок."""
    delta = float(mean_delta(BT.point_sum, hi, lo, data, TARGETS))
    draws = mean_delta(S_b, hi, lo, data, TARGETS)
    drop = empty_draws(S_b, hi, lo, data, TARGETS)
    per = {}
    for t in (*TARGETS, SEEN):
        d = data[t]
        pt = BT.share_diff(BT.point_sum, hi, lo, d["works"], d["mask"])
        bt = BT.share_diff(S_b, hi, lo, d["works"], d["mask"])
        per[t] = {"n_hi": int(pt["n_hi"]), "n_lo": int(pt["n_lo"]), "share_hi": float(pt["share_hi"]),
                  "share_lo": float(pt["share_lo"]), "delta": float(pt["delta"]),
                  "delta_ci": _ci(bt["delta"][~drop], level)}  # fmt: skip
    out = {
        "delta": delta,
        "delta_ci": _ci(draws[~drop], level),
        "sd_boot": _sd(draws[~drop]),
        "bootstrap_dropped": int(drop.sum()),
        "share_hi": float(mean_share(BT.point_sum, hi, data, TARGETS)),
        "share_lo": float(mean_share(BT.point_sum, lo, data, TARGETS)),
        "per_target": per,
    }
    if perm_hi is not None:
        null = mean_delta(BT.point_sum, perm_hi, perm_lo, data, TARGETS)
        out["permutation"] = perm_summary(null, delta)
    return out


def _restrict(
    data: Mapping[str, Mapping[str, np.ndarray]], keep: np.ndarray
) -> dict[str, dict[str, np.ndarray]]:
    return {t: {**d, "mask": d["mask"] & keep} for t, d in data.items()}


def compute(
    cfg: Config, inp: Mapping[str, Any], by_type_facts: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Весь расчёт блока на прочитанных входах. ``by_type_facts`` — выход ``usefulness_by_type.run`` (для QC:
    точечная Δ типов тем же кодом = основной прогон by_type_test). Возвращает ``{"facts", "by_mo"}``."""
    t0 = time.perf_counter()
    block = cfg["usefulness"]["size_posthoc"]
    check_spec(block)
    bt_block = cfg["usefulness"]["by_type_test"]
    BT.check_spec(bt_block)
    seed = int(cfg["seed"])
    st = block["statistic"]
    level = float(st["ci_level"])
    n_q = int(block["quintiles"]["n"])
    year = int(block["size"]["year"])
    high = [int(k) for k in bt_block["universe"]["groups"]["high"]]
    low = [int(k) for k in bt_block["universe"]["groups"]["low"]]
    min_set = int(bt_block["universe"]["min_set"])

    e = inp["errors"]
    ids = np.sort(e.loc[e["common"].astype(bool), "territory_id"].to_numpy(dtype=np.int64))
    group_of = inp["nodes"].set_index("territory_id")["region_group"]
    groups = group_of.reindex(ids).to_numpy()
    if pd.isna(groups).any():
        raise QCError("usefulness.size_posthoc: у части МО общего набора нет группы региона")
    groups = groups.astype(np.int64)
    types = inp["types"].set_index("territory_id")["type"]
    labels = types.reindex(ids)
    if labels.isna().any() or not set(labels.astype(int)) <= set(high) | set(low):
        raise QCError("usefulness.size_posthoc: у части МО общего набора нет типа из групп {3,4}/{1,2}")
    labels = labels.to_numpy(dtype=np.int64)
    hi_t, lo_t = np.isin(labels, high), np.isin(labels, low)

    ctx = inp["context"]
    pop_all = ctx.loc[ctx["year"] == year].set_index("territory_id")[POP_COL].astype(float)
    pop = pop_all.reindex(ids).to_numpy(dtype=float)
    bounds = quantile_bounds(pop, n_q)
    q = assign_quantile(pop, bounds)
    large = q == n_q - 1
    rest = ~large

    # цели — как by_type_test (те же функции и правило «сработало»)
    data: dict[str, dict[str, np.ndarray]] = {}
    prep: dict[str, dict[str, np.ndarray]] = {}
    uniq_groups = np.unique(groups)
    for t in (*TARGETS, SEEN):
        y = BT.target_change(ctx, BT.COLUMNS[t], BT.NDFL_FILTER if t == "ndfl" else None)
        tab = BT.target_table(y, inp["comparable"], ids, min_set)
        m = tab["in_target"].to_numpy()
        eb, ed = tab["err_B"].to_numpy(), tab["err_D"].to_numpy()
        works = m & (np.nan_to_num(eb, nan=np.inf) < np.nan_to_num(ed, nan=np.inf))
        data[t] = {"mask": m, "works": works}
        if t in TARGETS:
            Y, G = d_member_matrix(y, inp["comparable"], ids, group_of, uniq_groups)
            prep[t] = {"mask": m, "own": tab["y"].to_numpy(dtype=float), "err_B": eb, "Y": Y, "G": G}

    gidx, counts = BT.bootstrap_counts(groups, int(st["bootstrap"]["n"]), seed)
    S_b = BT.boot_sum(gidx, counts)

    # QC: точечная Δ типов тем же кодом = основной прогон by_type_test
    d_types = float(mean_delta(BT.point_sum, hi_t, lo_t, data, TARGETS))
    tol = float(block["qc"]["types_main_vs_by_type"])
    qc = {"types_main_delta": d_types}
    if by_type_facts is not None:
        ref = float(by_type_facts["runs"]["main"]["delta"])
        if not abs(ref - d_types) <= tol:
            raise QCError(
                f"usefulness.size_posthoc: Δ типов {d_types!r} ≠ by_type_test {ref!r} (допуск {tol})"
            )
        qc.update({"by_type_main_delta": ref, "abs_diff": abs(ref - d_types), "tol": tol})

    # 1–2. квинтили
    quint = []
    for k in range(n_q):
        mk = q == k
        lo_b = float(pop[mk].min())
        hi_b = float(pop[mk].max())
        quint.append(
            {
                "quintile": k + 1,
                "n_base": int(mk.sum()),
                "pop_min": lo_b,
                "pop_max": hi_b,
                "upper_bound": float(bounds[k]) if k < n_q - 1 else None,
                "share_types_hi": float(hi_t[mk].mean()),
                **_group_block(S_b, mk, data, level),
            }
        )

    # 3. крупные против остальных: перестановка метки внутри групп региона
    n_perm_l = int(st["large_vs_rest"]["permutation"]["n"])
    perm_g = BT.permutation_index(groups, n_perm_l, seed)
    large_block = _delta_block(S_b, large, rest, data, large[perm_g], rest[perm_g], level)
    large_block["n_large"] = int(large.sum())
    large_block["pop_from"] = float(bounds[-1])

    # основная Δ типов (как by_type_test) — тем же блоком, перестановка внутри групп региона (описание)
    types_block = _delta_block(S_b, hi_t, lo_t, data, hi_t[perm_g], lo_t[perm_g], level)

    # 4a. тип при равном размере
    strata_q = [q == k for k in range(n_q)]
    cell_q = cell_codes(groups, q)
    perm_q = BT.permutation_index(cell_q, int(st["type_given_size"]["permutation"]["n"]), seed)
    lab_p = labels[perm_q]
    tgs = float(mean_strat_delta(BT.point_sum, hi_t, lo_t, data, TARGETS, strata_q))
    tgs_b = mean_strat_delta(S_b, hi_t, lo_t, data, TARGETS, strata_q)
    tgs_null = mean_strat_delta(
        BT.point_sum, np.isin(lab_p, high), np.isin(lab_p, low), data, TARGETS, strata_q
    )
    type_given_size = {
        "delta": tgs,
        "delta_ci": _ci(tgs_b, level),
        "cells": int(len(np.unique(cell_q))),
        "permutation": perm_summary(tgs_null, tgs),
    }

    # 4b. размер при равном типе
    strata_t = [hi_t, lo_t]
    cell_t = cell_codes(groups, hi_t.astype(np.int64))
    perm_t = BT.permutation_index(cell_t, int(st["size_given_type"]["permutation"]["n"]), seed)
    q_p = q[perm_t]
    sgt = float(mean_strat_delta(BT.point_sum, large, rest, data, TARGETS, strata_t))
    sgt_b = mean_strat_delta(S_b, large, rest, data, TARGETS, strata_t)
    sgt_null = mean_strat_delta(BT.point_sum, q_p == n_q - 1, q_p != n_q - 1, data, TARGETS, strata_t)
    size_given_type = {
        "delta": sgt,
        "delta_ci": _ci(sgt_b, level),
        "cells": int(len(np.unique(cell_t))),
        "permutation": perm_summary(sgt_null, sgt),
    }

    # 5. бутстрап с общими членами D (те же счётчики групп)
    sm = block["shared_members_bootstrap"]
    n_sm = int(sm["n"])
    t1 = time.perf_counter()
    pairs = {"types_main": (hi_t, lo_t), "large_vs_rest": (large, rest)}
    sb = shared_bootstrap(prep, gidx, counts[:n_sm], pairs, TARGETS)
    t_sm = time.perf_counter() - t1
    # контроль: при весах «все группы по разу» медиана D — исходная, «B ближе D» — как в data
    ones = shared_bootstrap(prep, gidx, np.ones((1, counts.shape[1])), {"types_main": (hi_t, lo_t)}, TARGETS)
    if not abs(float(ones["types_main"][0]) - d_types) <= tol:
        raise QCError(
            "usefulness.size_posthoc: бутстрап с общими членами D при единичных весах не дал Δ типов"
        )
    n_outside = int(sum((prep[t]["G"] == len(uniq_groups)).sum() for t in TARGETS))
    shared = {
        "n_draws": n_sm,
        "seconds": round(t_sm, 1),
        "members_outside_groups": n_outside,
        "types_main": {
            "delta": d_types,
            "delta_ci": _ci(sb["types_main"], level),
            "sd_boot": _sd(sb["types_main"]),
            "dropped": int((~np.isfinite(sb["types_main"])).sum()),
            "ordinary_ci": types_block["delta_ci"],
        },
        "large_vs_rest": {
            "delta": large_block["delta"],
            "delta_ci": _ci(sb["large_vs_rest"], level),
            "sd_boot": _sd(sb["large_vs_rest"]),
            "dropped": int((~np.isfinite(sb["large_vs_rest"])).sum()),
            "ordinary_ci": large_block["delta_ci"],
        },
    }

    # 6. пересечение покрытий трёх целей
    inter = np.logical_and.reduce([data[t]["mask"] for t in TARGETS])
    di = _restrict(data, inter)
    all_three = {
        "n": int(inter.sum()),
        "n_types_hi": int((inter & hi_t).sum()),
        "n_types_lo": int((inter & lo_t).sum()),
        "n_large": int((inter & large).sum()),
        "types": _delta_block(S_b, hi_t, lo_t, di, hi_t[perm_g], lo_t[perm_g], level),
        "large_vs_rest": _delta_block(S_b, large, rest, di, large[perm_g], rest[perm_g], level),
        "by_quintile": [
            {
                "quintile": k + 1,
                "n": int((inter & (q == k)).sum()),
                "share": float(mean_share(BT.point_sum, q == k, di, TARGETS)),
                "share_ci": _ci(mean_share(S_b, q == k, di, TARGETS), level),
            }
            for k in range(n_q)
        ],
    }

    crosstab = {
        "types_hi_large": int((hi_t & large).sum()),
        "types_hi_rest": int((hi_t & rest).sum()),
        "types_lo_large": int((lo_t & large).sum()),
        "types_lo_rest": int((lo_t & rest).sum()),
    }

    # узлы с типом и известным населением: квинтиль по границам общего набора
    typed = types.loc[types.isin(set(high) | set(low))]
    pop_typed = pop_all.reindex(typed.index.to_numpy())
    known = pop_typed.notna().to_numpy()
    tid = typed.index.to_numpy(dtype=np.int64)[known]
    pv = pop_typed.to_numpy(dtype=float)[known]
    qv = assign_quantile(pv, bounds)
    by_mo = pd.DataFrame(
        {
            "territory_id": tid,
            "type": typed.to_numpy(dtype=np.int64)[known],
            "pop_avg_2023": pv,
            "quintile": qv + 1,
            "large": qv == n_q - 1,
            "in_t7_common": np.isin(tid, ids),
        }
    ).sort_values("territory_id", kind="mergesort")
    # квинтили общего набора в by_mo = квинтили расчёта
    chk = by_mo.set_index("territory_id")["quintile"].reindex(ids).to_numpy()
    if not np.array_equal(chk, q + 1):
        raise QCError("usefulness.size_posthoc: квинтили size_by_mo.csv расходятся с расчётом")

    facts = {
        "status": "done",
        "kind": "posthoc_exploration",
        "note": (
            "РАЗВЕДКА ПОСЛЕ ВСКРЫТИЯ, 02.10.2026: записано после исхода by_type_test и черновых расчётов "
            "devils-advocate (da1–da8); не проверка, вердиктов нет; числа и p — описание; порог «крупные = "
            "верхний квинтиль населения» выбран участником до этого кода по черновым числам"
        ),
        "targets": list(TARGETS),
        "seen_target": SEEN,
        "size": {"source": str(block["size"]["source"]), "year": year},
        "n_base": int(len(ids)),
        "n_groups": int(len(uniq_groups)),
        "quintile_bounds": [float(b) for b in bounds],
        "quintiles": quint,
        "large_vs_rest": large_block,
        "types_main": types_block,
        "type_given_size": type_given_size,
        "size_given_type": size_given_type,
        "shared_members_bootstrap": shared,
        "all_three_targets": all_three,
        "crosstab_types_large": crosstab,
        "by_mo": {
            "n": int(len(by_mo)),
            "n_typed_without_pop": int((~known).sum()),
            "n_large": int(by_mo["large"].sum()),
            "n_large_outside_common": int((by_mo["large"] & ~by_mo["in_t7_common"]).sum()),
        },
        "design": {
            "bootstrap": {"n": int(counts.shape[0]), "by": "region_group", "seed": seed},
            "permutation": {"n": n_perm_l, "seed": seed},
            "shared_members_bootstrap": {"n": n_sm, "seed": seed, "same_counts_as_bootstrap": True},
            "ci_level": level,
        },
        "qc": {**qc, "spec_frozen": True},
        "seconds": round(time.perf_counter() - t0, 1),
    }
    return {"facts": facts, "by_mo": by_mo}


def run(cfg: Config, by_type_facts: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Расчёт и запись ``size_check.json`` и ``size_by_mo.csv``; старые выходы блока удаляются до расчёта."""
    out = cfg.dir("outputs") / "usefulness"
    out.mkdir(parents=True, exist_ok=True)
    files = {"facts": out / "size_check.json", "by_mo": out / "size_by_mo.csv"}
    for p in files.values():
        p.unlink(missing_ok=True)
    res = compute(cfg, _read(cfg), by_type_facts)
    res["by_mo"].to_csv(files["by_mo"], index=False)
    _write_json(files["facts"], res["facts"])
    f = res["facts"]
    log.info(
        "usefulness.size_posthoc (разведка после вскрытия): доля «B ближе D» по квинтилям населения %s; "
        "крупные против остальных Δ %.4f; p типа при равном размере %.3f, размера при равном типе %.3f",
        [round(x["share"], 3) for x in f["quintiles"]], f["large_vs_rest"]["delta"],
        f["type_given_size"]["permutation"]["p_less"], f["size_given_type"]["permutation"]["p_less"],
    )  # fmt: skip
    return f


def _write_json(path: Path, obj: Any) -> None:
    BT._write_json(path, obj)
