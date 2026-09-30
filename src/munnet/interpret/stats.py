"""Статистика этапа interpret: чистые функции на массивах, без чтения файлов.

- ``within_ranks`` — ранг показателя внутри страты, ``rank(method=average) / n_s`` (``strata.within_rank``).
- ``strat_spearman`` — ρ Спирмена ранга ступени минус его среднее в страте и ранга показателя внутри страты
  (``T1_ladder_external.conditional.statistic: stratified_spearman``).
- ``epsilon2`` — ε² Краскела — Уоллиса H / (n − 1), как во внешней проверке этапа 3
  (``munnet.clustering.validation.epsilon2``).
- ``cliff_delta`` — дельта Клиффа δ = P(x > y) − P(x < y) через U Манна — Уитни.
- ``monotone_ok`` — правило ``nondecreasing_top_gt_bottom``: медианы на соседних ступенях не убывают,
  верхняя строго выше нижней.
- p-значения перестановок — (1 + число не менее крайних) / (1 + число перестановок).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from munnet.clustering.validation import epsilon2 as _eps2
from munnet.clustering.validation import permute_within, spearman
from munnet.network.rules import bh_qvalues

__all__ = [
    "bh_qvalues",
    "cliff_delta",
    "codes",
    "epsilon2",
    "holm",
    "medians_by",
    "monotone_ok",
    "p_greater",
    "p_two_sided",
    "perc_ci",
    "permute_within",
    "spearman",
    "strat_spearman",
    "weighted_quantile",
    "within_ranks",
]

MAD_SCALE = 1.4826


def codes(labels: np.ndarray) -> tuple[np.ndarray, int]:
    """Коды 0…m−1 значений ``labels`` и их число."""
    _, inv = np.unique(np.asarray(labels), return_inverse=True)
    return inv.astype(np.int64), int(inv.max()) + 1 if len(inv) else 0


def within_ranks(y: np.ndarray, strata: np.ndarray) -> np.ndarray:
    """Ранг ``y`` внутри страты (средний при равенстве), делённый на число узлов страты: доли 0–1."""
    s = pd.Series(np.asarray(y, dtype=np.float64))
    g = np.asarray(strata)
    r = s.groupby(g).rank(method="average").to_numpy()
    n = s.groupby(g).transform("size").to_numpy(dtype=np.float64)
    return r / n


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else float("nan")


def strat_spearman(
    step: np.ndarray, yw: np.ndarray, strata: np.ndarray, ryw: np.ndarray | None = None
) -> float:
    """ρ Спирмена по всем узлам: ранг ступени минус его среднее в страте против ранга показателя внутри
    страты. ``ryw`` — готовые ранги ``yw`` (для бутстрапа и перестановок, где ``yw`` не меняется)."""
    step = np.asarray(step, dtype=np.float64)
    inv, m = codes(strata)
    cnt = np.bincount(inv, minlength=m).astype(np.float64)
    mean = np.bincount(inv, weights=step, minlength=m) / np.where(cnt > 0, cnt, 1.0)
    x = step - mean[inv]
    rx = rankdata(x)
    ry = rankdata(yw) if ryw is None else ryw
    return _pearson(rx, ry)


def epsilon2(y: np.ndarray, labels: np.ndarray, ranks: np.ndarray | None = None) -> float:
    return float(_eps2(np.asarray(y, dtype=np.float64), np.asarray(labels), ranks))


def cliff_delta(x: np.ndarray, y: np.ndarray) -> float:
    """δ = P(x > y) − P(x < y) по всем парам (ранги с поправкой на равенство)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0:
        return float("nan")
    r = rankdata(np.concatenate([x, y]))
    u = r[:n1].sum() - n1 * (n1 + 1) / 2.0
    return float(2.0 * u / (n1 * n2) - 1.0)


def monotone_ok(medians: Sequence[float]) -> bool:
    """Медианы по ступеням снизу вверх: соседние не убывают, верхняя строго выше нижней; пропуск — нет."""
    m = np.asarray(medians, dtype=np.float64)
    if len(m) < 2 or not np.isfinite(m).all():
        return False
    return bool(np.all(np.diff(m) >= 0) and m[-1] > m[0])


def medians_by(values: np.ndarray, labels: np.ndarray, order: Sequence[int]) -> list[float]:
    """Медианы ``values`` по значениям ``labels`` в порядке ``order`` (пустая группа — NaN)."""
    values = np.asarray(values, dtype=np.float64)
    labels = np.asarray(labels)
    out = []
    for g in order:
        v = values[labels == g]
        out.append(float(np.median(v)) if len(v) else float("nan"))
    return out


def p_two_sided(obs: float, null: np.ndarray) -> float:
    null = np.asarray(null, dtype=np.float64)
    null = null[np.isfinite(null)]
    if not np.isfinite(obs):
        return float("nan")
    return float((1 + np.sum(np.abs(null) >= abs(obs) - 1e-12)) / (1 + len(null)))


def p_greater(obs: float, null: np.ndarray) -> float:
    null = np.asarray(null, dtype=np.float64)
    null = null[np.isfinite(null)]
    if not np.isfinite(obs):
        return float("nan")
    return float((1 + np.sum(null >= obs - 1e-12)) / (1 + len(null)))


def perc_ci(samples: np.ndarray, level: float) -> tuple[float, float]:
    """Перцентильный интервал уровня ``level`` (пропуски выброшены)."""
    s = np.asarray(samples, dtype=np.float64)
    s = s[np.isfinite(s)]
    if not len(s):
        return float("nan"), float("nan")
    lo, hi = np.percentile(s, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return float(lo), float(hi)


def holm(p: Sequence[float]) -> np.ndarray:
    """Поправка Холма: скорректированные p (монотонные, не больше 1); пропуски остаются пропусками."""
    p = np.asarray(p, dtype=np.float64)
    out = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    m = int(ok.sum())
    if not m:
        return out
    pv = p[ok]
    order = np.argsort(pv, kind="stable")
    adj = np.maximum.accumulate((m - np.arange(m)) * pv[order])
    res = np.empty(m)
    res[order] = np.minimum(adj, 1.0)
    out[ok] = res
    return out


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    """Взвешенный квантиль: наименьшее значение, у которого накопленный вес не меньше доли ``q``."""
    v = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    ok = np.isfinite(v) & np.isfinite(w) & (w > 0)
    v, w = v[ok], w[ok]
    if not len(v):
        return float("nan")
    order = np.argsort(v, kind="stable")
    v, w = v[order], w[order]
    cw = np.cumsum(w) / w.sum()
    return float(v[np.searchsorted(cw, q - 1e-12)])


def mad_scale(values: np.ndarray) -> float:
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if not len(v):
        return float("nan")
    return float(np.median(np.abs(v - np.median(v))) * MAD_SCALE)
