"""T6 (сдвиг вверх вне данных банка) и T7 (польза «сопоставимых территорий») — на массивах.

T6: изменение оборота общепита Росстата (``log1p_rel``, 2023 → 2024) у надёжно перешедших вверх с двух нижних
ступеней против оставшихся того же исходного типа; дельта Клиффа только по парам одного исходного типа,
перестановки статуса внутри исходного типа.

T7: для каждого МО — наборы по 10 МО: A — ближайшие по корзине 2023 года того же типа из других регионов,
B — ближайшие по координатам в своей группе региона, C — случайные того же дециля населения из других регионов
(100 розыгрышей), D — ближайшие по корзине из других регионов без учёта типа.
Ошибка МО — |y − медиана y набора|,
сводка — медиана по МО; разности медианных ошибок — парный бутстрап МО.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from munnet.clustering.validation import BeyondAttributes
from munnet.interpret import stats as S

# --- T6 --------------------------------------------------------------------------------------------


def _weighted_delta(change: np.ndarray, groups: list[tuple[np.ndarray, np.ndarray]]) -> float:
    num = den = 0.0
    for mv, st in groups:
        if len(mv) and len(st):
            w = len(mv) * len(st)
            num += w * S.cliff_delta(change[mv], change[st])
            den += w
    return num / den if den else float("nan")


def t6_eval(
    change: np.ndarray,
    t_a: np.ndarray,
    t_b: np.ndarray,
    reliable: np.ndarray,
    flows: Sequence[tuple[int, int]],
    n_perm: int,
    n_boot: int,
    alpha: float,
    level: float,
    cliff_lower_min: float,
    rng: np.random.Generator,
) -> dict:
    """Правило T6: confirmed — p < alpha и нижняя граница δ > ``cliff_lower_min``; partial — одно из двух."""
    change = np.asarray(change, dtype=np.float64)
    ok = np.isfinite(change)
    t_a, t_b, reliable = np.asarray(t_a), np.asarray(t_b), np.asarray(reliable, dtype=bool)
    sources = sorted({a for a, _ in flows})
    groups, per_flow = [], []
    for s in sources:
        dests = [b for a, b in flows if a == s]
        mv = np.flatnonzero(ok & reliable & (t_a == s) & np.isin(t_b, dests))
        st = np.flatnonzero(ok & ~reliable & (t_a == s) & (t_b == s))
        groups.append((mv, st))
        for d in dests:
            m = np.flatnonzero(ok & reliable & (t_a == s) & (t_b == d))
            per_flow.append(
                {
                    "source": s,
                    "dest": d,
                    "n_movers": int(len(m)),
                    "n_stayers": int(len(st)),
                    "cliff": S.cliff_delta(change[m], change[st]),
                    "median_movers": float(np.median(change[m])) if len(m) else float("nan"),
                    "median_stayers": float(np.median(change[st])) if len(st) else float("nan"),
                }
            )
    delta = _weighted_delta(change, groups)
    null = np.empty(n_perm)
    for i in range(n_perm):
        pg = []
        for mv, st in groups:
            pool = np.concatenate([mv, st])
            perm = pool[rng.permutation(len(pool))]
            pg.append((perm[: len(mv)], perm[len(mv) :]))
        null[i] = _weighted_delta(change, pg)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        bg = [
            (mv[rng.integers(0, len(mv), len(mv))], st[rng.integers(0, len(st), len(st))])
            for mv, st in groups
        ]
        boots[i] = _weighted_delta(change, bg)
    p = S.p_greater(delta, null)
    ci = S.perc_ci(boots, level)
    c1 = bool(p < alpha)
    c2 = bool(ci[0] > cliff_lower_min)
    verdict = "confirmed" if (c1 and c2) else ("partial" if (c1 or c2) else "not")
    return {
        "verdict": verdict,
        "delta": delta,
        "ci": ci,
        "p": p,
        "n_movers": int(sum(len(mv) for mv, _ in groups)),
        "n_stayers": int(sum(len(st) for _, st in groups)),
        "n_nodes": int(ok.sum()),
        "per_flow": per_flow,
    }


# --- T7 --------------------------------------------------------------------------------------------


@dataclass
class T7Sets:
    members: dict[str, list[np.ndarray]]  # набор -> по МО позиции членов (для C — первый розыгрыш)
    err: dict[str, np.ndarray]  # набор -> ошибка МО (NaN — набор меньше min_set)
    err_abs: dict[str, np.ndarray]
    sizes: dict[str, np.ndarray]
    c_draws: list[list[np.ndarray]] = field(default_factory=list)


def _nearest(dist: np.ndarray, cand: np.ndarray, k: int) -> np.ndarray:
    idx = np.flatnonzero(cand)
    if len(idx) <= k:
        return idx[np.argsort(dist[idx], kind="stable")]
    part = idx[np.argpartition(dist[idx], k - 1)[:k]]
    return part[np.lexsort((part, dist[part]))]


def t7_sets(
    F: np.ndarray,
    xy: np.ndarray,
    groups: np.ndarray,
    types: np.ndarray,
    decile: np.ndarray,
    y: np.ndarray,
    y_abs: np.ndarray,
    k: int,
    min_set: int,
    draws: int,
    rng: np.random.Generator,
) -> T7Sets:
    """Наборы A, B, C, D для каждого МО (строки — МО с известной целью; члены — только из них)."""
    n = len(y)
    members: dict[str, list[np.ndarray]] = {s: [] for s in "ABCD"}
    err = {s: np.full(n, np.nan) for s in "ABCD"}
    err_abs = {s: np.full(n, np.nan) for s in "ABCD"}
    sizes = {s: np.zeros(n, dtype=np.int64) for s in "ABCD"}
    all_draws = []
    for i in range(n):
        other = groups != groups[i]
        d_b = np.sqrt(((F - F[i]) ** 2).sum(axis=1))
        d_g = np.sqrt(((xy - xy[i]) ** 2).sum(axis=1))
        not_self = np.arange(n) != i
        sets = {
            "A": _nearest(d_b, other & (types == types[i]), k),
            "B": _nearest(d_g, (groups == groups[i]) & not_self, k),
            "D": _nearest(d_b, other, k),
        }
        pool_c = np.flatnonzero(other & (decile == decile[i]))
        dr = []
        for _ in range(draws):
            dr.append(rng.choice(pool_c, min(k, len(pool_c)), replace=False) if len(pool_c) else pool_c)
        all_draws.append(dr)
        sets["C"] = dr[0] if dr else np.array([], dtype=np.int64)
        for s, mem in sets.items():
            members[s].append(mem)
            sizes[s][i] = len(mem)
            if len(mem) < min_set:
                continue
            if s == "C":
                err[s][i] = float(np.median([abs(y[i] - np.median(y[m])) for m in dr]))
                err_abs[s][i] = float(np.median([abs(y_abs[i] - np.median(y_abs[m])) for m in dr]))
            else:
                err[s][i] = abs(y[i] - float(np.median(y[mem])))
                err_abs[s][i] = abs(y_abs[i] - float(np.median(y_abs[mem])))
    return T7Sets(members=members, err=err, err_abs=err_abs, sizes=sizes, c_draws=all_draws)


def _diff_ci(e1: np.ndarray, e0: np.ndarray, idx_boot: list[np.ndarray], level: float) -> tuple[float, tuple]:
    point = float(np.median(e1) - np.median(e0))
    d = np.array([np.median(e1[b]) - np.median(e0[b]) for b in idx_boot])
    return point, S.perc_ci(d, level)


def t7_eval(
    sets: T7Sets,
    n_boot: int,
    level: float,
    lower_min: float,
    rng: np.random.Generator,
) -> dict:
    """Сравнение наборов на общих МО (все четыре набора не меньше min_set); правило ``T7.rule``."""
    common = np.ones(len(sets.err["A"]), dtype=bool)
    for s in "ABCD":
        common &= np.isfinite(sets.err[s])
    pos = np.flatnonzero(common)
    e = {s: sets.err[s][pos] for s in "ABCD"}
    ea = {s: sets.err_abs[s][pos] for s in "ABCD"}
    boots = [rng.integers(0, len(pos), len(pos)) for _ in range(n_boot)]
    med = {s: float(np.median(e[s])) for s in "ABCD"}
    med_abs = {s: float(np.median(ea[s])) for s in "ABCD"}
    diffs = {}
    for base in ("A", "D"):
        for other in ("B", "C"):
            diffs[f"{other}-{base}"] = _diff_ci(e[other], e[base], boots, level)
            diffs[f"{other}-{base}_abs"] = _diff_ci(ea[other], ea[base], boots, level)
    diffs["D-A"] = _diff_ci(e["D"], e["A"], boots, level)
    diffs["D-A_abs"] = _diff_ci(ea["D"], ea["A"], boots, level)

    def verdict(base: str) -> tuple[str, bool, bool]:
        vs_b = diffs[f"B-{base}"][1][0] > lower_min and diffs[f"B-{base}_abs"][1][0] > lower_min
        vs_c = diffs[f"C-{base}"][1][0] > lower_min
        v = "confirmed" if (vs_b and vs_c) else ("partial" if (vs_b or vs_c) else "not")
        return v, bool(vs_b), bool(vs_c)

    lo, hi = diffs["D-A"][1]
    gain = "adds" if lo > lower_min else ("hurts" if hi < lower_min else "neutral")
    product = "A" if gain == "adds" else "D"
    v_p, b_p, c_p = verdict(product)
    v_a, b_a, c_a = verdict("A")
    v_d, b_d, c_d = verdict("D")
    return {
        "n_common": int(len(pos)),
        "n_dropped": int((~common).sum()),
        "common_pos": pos,
        "median_error": med,
        "median_error_abs": med_abs,
        "diffs": diffs,
        "type_gain": gain,
        "product": product,
        "verdict": v_p,
        "against_B": b_p,
        "against_C": c_p,
        "verdict_A": v_a,
        "against_B_A": b_a,
        "against_C_A": c_a,
        "verdict_D": v_d,
        "against_B_D": b_d,
        "against_C_D": c_d,
    }


def beyond_place(
    y: np.ndarray, X: np.ndarray, labels: np.ndarray, folds: int, repeats: int, n_perm: int, seed: int, rng
) -> dict:
    """(b) T7: ΔR² на перекрёстной проверке от меток типа сверх признаков места (``BeyondAttributes`` этапа 3)
    и p по перестановкам меток типа."""
    ba = BeyondAttributes(np.asarray(y, dtype=np.float64), X, folds, repeats, seed)
    gain = ba.gain(labels)
    null = np.array([ba.gain(labels[rng.permutation(len(labels))]) for _ in range(n_perm)])
    return {"delta_r2": float(gain), "base_r2": float(ba.base_r2()), "p": S.p_greater(gain, null)}


def median_example(err: np.ndarray, ids: np.ndarray) -> int:
    """Позиция МО, чья ошибка ближе всего к медиане (``example: median_error_product``; равенство —
    меньший id)."""
    med = float(np.median(err))
    d = np.abs(err - med)
    order = np.lexsort((ids, d))
    return int(order[0])
