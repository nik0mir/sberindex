"""Робастная статистика разведки: медианы и MAD, ранговые связи в целом и внутри регионов, η², CLR, SMD,
модель пропуска, I Морана с перестановочным тестом (spec_final, Б.1).

Все функции чистые: пропуски отбрасываются попарно, случайность — только через переданный ``rng``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.spatial import cKDTree

# MAD × 1,4826 — оценка стандартного отклонения, согласованная с нормальным распределением.
MAD_SCALE = 1.4826
# Меньше трёх пар — ранговая корреляция не определена.
MIN_PAIRS = 3
# Допуск сравнения накопленных весов (на одно наблюдение) во взвешенном квантиле.
_CUM_TOL = 1e-9


def _arr(x) -> np.ndarray:
    return np.asarray(pd.Series(x, dtype="float64") if not isinstance(x, np.ndarray) else x, dtype="float64")


def mad(x) -> float:
    """Медианное абсолютное отклонение (без множителя 1,4826); пропуски отбрасываются."""
    v = _arr(x)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return float("nan")
    return float(np.median(np.abs(v - np.median(v))))


def _robust_z(v: pd.Series) -> pd.Series:
    scale = MAD_SCALE * mad(v)
    if not np.isfinite(scale) or scale == 0:
        return pd.Series(np.nan, index=v.index)
    return (v - v.median()) / scale


def robust_z(x: pd.Series, groups: pd.Series | None = None) -> pd.Series:
    """Робастный z = (x − медиана) / (1,4826 · MAD), в целом или внутри групп.

    При MAD = 0 (половина значений одинакова) z не определён — NaN, а не бесконечность.
    """
    x = pd.Series(x, dtype="float64")
    if groups is None:
        return _robust_z(x)
    g = pd.Series(groups, index=x.index)
    return x.groupby(g, observed=True, dropna=False).transform(_robust_z)


def _pairs(x, w) -> tuple[np.ndarray, np.ndarray]:
    xv, wv = _arr(x), _arr(w)
    if xv.shape != wv.shape:
        raise ValueError("значения и веса разной длины")
    ok = ~np.isnan(xv) & ~np.isnan(wv)
    xv, wv = xv[ok], wv[ok]
    if (wv < 0).any():
        raise ValueError("отрицательные веса")
    keep = wv > 0
    return xv[keep], wv[keep]


def weighted_quantile(x, w, q: float) -> float:
    """Взвешенный квантиль: тип 7 с частотными весами, нормированными к сумме n (как ``wtd.quantile`` Hmisc).

    При равных весах совпадает с ``np.quantile`` (линейная интерполяция). Вес — «сколько раз» наблюдение
    входит в выборку: позиция h = (n − 1)·q + 1 ищется на накопленных весах, между соседними позициями —
    линейная интерполяция. Наблюдения с весом много меньше среднего почти не влияют на крайние квантили.
    """
    xv, wv = _pairs(x, w)
    if xv.size == 0:
        return float("nan")
    order = np.argsort(xv, kind="mergesort")
    xv, wv = xv[order], wv[order]
    n = xv.size
    cum = np.cumsum(wv * n / wv.sum())
    tol = _CUM_TOL * n

    def at(k: float) -> float:
        i = int(np.searchsorted(cum, k - tol, side="left"))
        return float(xv[min(i, n - 1)])

    h = (n - 1) * float(q) + 1
    lo = math.floor(h)
    frac = h - lo
    return at(lo) if frac == 0 else at(lo) + frac * (at(lo + 1) - at(lo))


def weighted_mean(x, w) -> float:
    """Взвешенное среднее; пары с пропуском отбрасываются."""
    xv, wv = _pairs(x, w)
    if wv.sum() == 0:
        return float("nan")
    return float(np.average(xv, weights=wv))


def _rank_corr(x: np.ndarray, y: np.ndarray) -> float:
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    return _pearson(rx, ry)


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a - a.mean(), b - b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / denom) if denom > 0 else float("nan")


def _complete(*cols) -> tuple[np.ndarray, ...]:
    arrs = [_arr(c) for c in cols]
    ok = np.logical_and.reduce([~np.isnan(a) for a in arrs])
    return tuple(a[ok] for a in arrs)


def spearman(x, y) -> tuple[float, int]:
    """Ранговая корреляция Спирмена по парам без пропусков: (ρ, n)."""
    xv, yv = _complete(x, y)
    n = int(xv.size)
    return (_rank_corr(xv, yv) if n >= MIN_PAIRS else float("nan")), n


def spearman_ci(
    x, y, n_boot: int, rng: np.random.Generator, level: float = 0.95
) -> tuple[float, float, float]:
    """ρ Спирмена и перцентильный бутстреп-интервал: (ρ, нижняя граница, верхняя граница)."""
    xv, yv = _complete(x, y)
    n = xv.size
    rho = _rank_corr(xv, yv) if n >= MIN_PAIRS else float("nan")
    if n < MIN_PAIRS:
        return rho, float("nan"), float("nan")
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = np.array([_rank_corr(xv[i], yv[i]) for i in idx])
    alpha = (1 - level) / 2
    lo, hi = np.nanquantile(boots, [alpha, 1 - alpha])
    return rho, float(lo), float(hi)


def _group_codes(groups, n: int) -> np.ndarray:
    g = pd.Series(groups).reset_index(drop=True)
    if len(g) != n:
        raise ValueError("группы и значения разной длины")
    codes, _ = pd.factorize(g, use_na_sentinel=True)
    return codes


def _center_by_group(v: np.ndarray, codes: np.ndarray) -> np.ndarray:
    s = pd.Series(v)
    return (s - s.groupby(codes).transform("mean")).to_numpy()


def _within_ranks(cols: list[np.ndarray], codes: np.ndarray) -> tuple[list[np.ndarray], int]:
    """Ранги по всей выборке, центрированные по группе; группы из одного наблюдения отбрасываются."""
    sizes = pd.Series(codes).map(pd.Series(codes).value_counts()).to_numpy()
    keep = (codes >= 0) & (sizes >= 2)
    ranks = [pd.Series(c[keep]).rank().to_numpy() for c in cols]
    return [_center_by_group(r, codes[keep]) for r in ranks], int(keep.sum())


def spearman_within(x, y, groups) -> tuple[float, int]:
    """ρ внутри групп: ранги всей выборки, центрированные по группе (регион), затем корреляция Пирсона.

    Чистый эффект группы (сдвиг уровня у всех её МО) такая связь не видит, если группы разнесены по шкале;
    при сильно перекрывающихся группах ранги всей выборки внутри группы нелинейны и остаток эффекта группы
    возможен. Позиционно: x, y и groups — в одном порядке строк. Возвращает (ρ, n).
    """
    xv, yv = _arr(x), _arr(y)
    codes = _group_codes(groups, xv.size)
    ok = ~np.isnan(xv) & ~np.isnan(yv) & (codes >= 0)
    (rx, ry), n = _within_ranks([xv[ok], yv[ok]], codes[ok])
    return (_pearson(rx, ry) if n >= MIN_PAIRS else float("nan")), n


def partial_spearman(x, y, controls: pd.DataFrame, groups=None) -> tuple[float, int]:
    """Частная ранговая корреляция x и y при контроле ``controls`` (и групп, если заданы).

    Ранги x, y и контролей (центрированные по группе при ``groups``) → остатки МНК x и y на контроли →
    корреляция остатков. Возвращает (ρ, n).
    """
    ctrl = pd.DataFrame(controls).reset_index(drop=True)
    cols = [_arr(x), _arr(y), *[_arr(ctrl[c]) for c in ctrl.columns]]
    ok = np.logical_and.reduce([~np.isnan(c) for c in cols])
    if groups is not None:
        codes = _group_codes(groups, cols[0].size)
        ok &= codes >= 0
        ranked, n = _within_ranks([c[ok] for c in cols], codes[ok])
    else:
        ranked = [pd.Series(c[ok]).rank().to_numpy() for c in cols]
        ranked = [r - r.mean() for r in ranked]
        n = int(ok.sum())
    if n < MIN_PAIRS + len(ctrl.columns):
        return float("nan"), n
    z = np.column_stack([np.ones(n), *ranked[2:]]) if len(ranked) > 2 else np.ones((n, 1))
    resid = []
    for target in ranked[:2]:
        beta, *_ = np.linalg.lstsq(z, target, rcond=None)
        resid.append(target - z @ beta)
    return _pearson(resid[0], resid[1]), n


def _eta2_parts(x, groups) -> tuple[float, int, int]:
    """(η², n, k): доля межгрупповой суммы квадратов, число наблюдений и групп без пропусков."""
    xv = _arr(x)
    codes = _group_codes(groups, xv.size)
    ok = ~np.isnan(xv) & (codes >= 0)
    xv, codes = xv[ok], codes[ok]
    n, k = int(xv.size), int(np.unique(codes).size)
    if n < 2:
        return float("nan"), n, k
    total = ((xv - xv.mean()) ** 2).sum()
    if total == 0:
        return float("nan"), n, k
    means = pd.Series(xv).groupby(codes).transform("mean").to_numpy()
    between = ((means - xv.mean()) ** 2).sum()
    return float(between / total), n, k


def eta2(x, groups) -> float:
    """η² — доля межгрупповой суммы квадратов в общей: 1 — всё различие между группами, 0 — внутри.

    Сырая доля смещена вверх: на чистом шуме она в среднем (k − 1)/(n − 1) — при 77 регионах и 2100 МО
    около 0,036. Рядом с порогами показывать и ``eta2_adj``.
    """
    return _eta2_parts(x, groups)[0]


def adjusted_r2(r2: float, n: int, n_params: int) -> float:
    """Скорректированный R²: 1 − (1 − R²)·(n − 1)/(n − p − 1), p — число предикторов без константы.

    На шуме в среднем около нуля (может быть меньше нуля); при n ≤ p + 1 не определён — NaN.
    """
    if not np.isfinite(r2) or n - n_params - 1 <= 0:
        return float("nan")
    return float(1.0 - (1.0 - r2) * (n - 1) / (n - n_params - 1))


def eta2_adj(x, groups) -> float:
    """η² с поправкой на число групп k: 1 − (1 − η²)·(n − 1)/(n − k) = η² − (k − 1)/(n − k)·(1 − η²).

    Это скорректированный R² регрессии на фиктивные переменные групп (оценка ε²): на шуме он в среднем около
    нуля (сырой η² — около (k − 1)/(n − 1)), при постоянстве внутри групп равен 1.
    """
    value, n, k = _eta2_parts(x, groups)
    return adjusted_r2(value, n, k - 1)


def closure(df: pd.DataFrame) -> pd.DataFrame:
    """Замыкание композиции: каждая строка делится на свою сумму (доли в сумме 1)."""
    return df.div(df.sum(axis=1), axis=0)


def clr(df: pd.DataFrame) -> pd.DataFrame:
    """Центрированное логарифмическое отношение: ln(доля / геометрическое среднее долей строки).

    Строки в сумме дают 0. Нули и отрицательные значения недопустимы (ValueError).
    """
    values = df.to_numpy(dtype="float64")
    finite = values[~np.isnan(values)]
    if (finite <= 0).any():
        raise ValueError("clr: доли должны быть больше нуля")
    logs = np.log(values)
    return pd.DataFrame(logs - logs.mean(axis=1, keepdims=True), index=df.index, columns=df.columns)


def smd(a, b) -> float:
    """Стандартизованная разность средних: (ср. a − ср. b) / √((var a + var b) / 2)."""
    av, bv = _arr(a), _arr(b)
    av, bv = av[~np.isnan(av)], bv[~np.isnan(bv)]
    if av.size < 2 or bv.size < 2:
        return float("nan")
    pooled = np.sqrt((av.var(ddof=1) + bv.var(ddof=1)) / 2)
    return float((av.mean() - bv.mean()) / pooled) if pooled > 0 else float("nan")


def missingness_auc(X: pd.DataFrame, y: pd.Series, seed: int, folds: int = 5) -> float:
    """AUC логистической модели пропуска на кросс-валидации: ≈ 0,5 — пропуски случайны.

    Модель: медианное заполнение пропусков признаков → стандартизация → логистическая регрессия;
    ``folds`` стратифицированных фолдов с перемешиванием (``random_state = seed``).
    """
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    target = pd.Series(y).astype(int).to_numpy()
    if len(np.unique(target)) < 2:
        return float("nan")
    features = pd.DataFrame(X).astype("float64").to_numpy()
    model = make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=1000)
    )
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    proba = cross_val_predict(model, features, target, cv=cv, method="predict_proba")[:, 1]
    return float(roc_auc_score(target, proba))


def knn_weights(xy: np.ndarray, k: int) -> sp.csr_matrix:
    """Матрица весов k ближайших соседей (без самой точки), строки нормированы на 1."""
    pts = np.asarray(xy, dtype="float64")
    n = pts.shape[0]
    if k >= n:
        raise ValueError(f"k = {k} не меньше числа точек {n}")
    _, idx = cKDTree(pts).query(pts, k=k + 1)
    rows, cols = [], []
    for i in range(n):
        neigh = [j for j in idx[i] if j != i][:k]
        rows.extend([i] * len(neigh))
        cols.extend(neigh)
    data = np.full(len(rows), 1.0 / k)
    return sp.csr_matrix((data, (rows, cols)), shape=(n, n))


@dataclass(frozen=True)
class MoranResult:
    """I Морана: значение, перестановочное p (одностороннее, в сторону наблюдаемого), z и число МО."""

    I: float  # noqa: E741 — общепринятое обозначение статистики
    p: float
    z: float
    n: int


def morans_i(
    y: pd.Series, xy: pd.DataFrame, k: int, permutations: int, rng: np.random.Generator
) -> MoranResult:
    """Глобальный I Морана на весах kNN (строки нормированы) с перестановочным тестом.

    МО с пропуском в ``y`` или координатах отбрасываются до построения весов. p = (1 + число
    перестановок не слабее наблюдаемого) / (перестановок + 1); z — по среднему и SD перестановок.
    Ожидание при отсутствии связи −1/(n − 1).
    """
    yv = _arr(y)
    pts = np.asarray(pd.DataFrame(xy), dtype="float64")
    ok = ~np.isnan(yv) & ~np.isnan(pts).any(axis=1)
    yv, pts = yv[ok], pts[ok]
    n = int(yv.size)
    if n <= k + 1:
        return MoranResult(float("nan"), float("nan"), float("nan"), n)
    w = knn_weights(pts, k)
    z = yv - yv.mean()
    denom = float(z @ z)
    if denom == 0:
        return MoranResult(float("nan"), float("nan"), float("nan"), n)
    observed = float(z @ (w @ z)) / denom
    perm = np.array([rng.permutation(z) for _ in range(permutations)])
    sims = np.einsum("ij,ij->i", perm, (w @ perm.T).T) / denom
    expected = -1.0 / (n - 1)
    if observed >= expected:
        extreme = int((sims >= observed).sum())
    else:
        extreme = int((sims <= observed).sum())
    p = (1 + extreme) / (permutations + 1)
    sd = sims.std(ddof=1)
    zscore = (observed - sims.mean()) / sd if sd > 0 else float("nan")
    return MoranResult(float(observed), float(p), float(zscore), n)
