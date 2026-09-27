"""Правила рёбер: сходство S_ij пары узлов, нулевые модели и значимость.

Все матрицы — ``float32`` (n × n, n ≈ 1800 узлов), считаются векторно; диагональ — NaN (петель нет).
Обозначения (docs/network.md, раздел «Правила»):

- s_i ∈ R⁶ — корзина узла за окно: CLR шести долей (пять категорий и «Прочее») минус центр своей группы
  региона (``features_windows.clr_rel_*``); Σ s_i = 0;
- r_i(t) — свой ритм: ln трат узла в месяц t минус линейный тренд узла a_i + b_i·t минус общий ритм
  календарного месяца g(t) (медиана по узлам), ``features_rhythm_monthly.own``;
- d_ij — дорожное расстояние, км; P_i — население.

Правила (больше S — ближе):

1. ``basket_cosine``: S = ⟨s_i, s_j⟩ / (‖s_i‖·‖s_j‖);
2. ``basket_distance``: d = ‖s_i − s_j‖ (расстояние Эйчисона между корзинами относительно региона),
   S = exp(−d²/σ²), σ — медиана d по парам;
3. ``rhythm_corr``: S = ρ_ij = corr(r_i, r_j) по месяцам окна (Пирсон; Спирмен — проверка);
4. ``rhythm_lag``: S = max_{|ℓ| ≤ L} ρ_ij(ℓ), ρ_ij(ℓ) = corr(r_i(t), r_j(t + ℓ)) по общим месяцам;
   ℓ* > 0 — i опережает j на ℓ* месяцев. S симметрично: ρ_ij(ℓ) = ρ_ji(−ℓ);
5. ``rhythm_dtw``: d = DTW(z(r_i), z(r_j)) с окном Сакоэ — Чибы w месяцев, S = exp(−d²/σ²);
6. ``road``: S = exp(−d²/σ²) по дорожному расстоянию;
7. ``gravity``: ln S = ln P_i + ln P_j − β·ln max(d, d_min).

Нулевая модель правил по рядам — циклический сдвиг ряда одного узла пары на s месяцев: автокорреляция
и распределение значений ряда сохраняются, совпадение во времени разрушается. Сдвиги, близкие к 0 или
к 12 месяцам (сезон совпадает), исключены: |s| mod 12 не ближе ``reach + guard`` к 0 и 12, где ``reach`` —
досягаемость правила (L у лагов, w у DTW, 0 у корреляции). Значимость пары — эмпирическое p-значение
по нулевому распределению, поправка на множественность — Бенджамини — Хохберг.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

FLOAT = np.float32
PERIOD = 12  # месяцев в сезонном цикле


# --- Вспомогательные ------------------------------------------------------------------------------


def _check_finite(X: np.ndarray, name: str) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"{name}: нужна матрица узлы × координаты, получено измерений {X.ndim}")
    if not np.isfinite(X).all():
        raise ValueError(f"{name}: пропуски или бесконечности во входе ({int((~np.isfinite(X)).sum())})")
    return X


def _no_loops(S: np.ndarray, square: bool) -> np.ndarray:
    S = S.astype(FLOAT, copy=False)
    if square:
        np.fill_diagonal(S, np.nan)
    return S


def zscore_rows(X: np.ndarray) -> np.ndarray:
    """Строки с нулевым средним и единичным стандартным отклонением (ddof = 0); постоянная строка — NaN."""
    X = np.asarray(X, dtype=np.float64)
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (X - mu) / np.where(sd > 0, sd, np.nan)


def upper(S: np.ndarray) -> np.ndarray:
    """Значения над диагональю (пары i < j) в порядке ``np.triu_indices``."""
    return S[np.triu_indices(S.shape[0], k=1)]


def median_offdiag(D: np.ndarray) -> float:
    """Медиана конечных значений над диагональю — масштаб σ ядра exp(−d²/σ²)."""
    v = upper(D)
    v = v[np.isfinite(v)]
    if not len(v):
        raise ValueError("нет ни одной конечной пары для масштаба σ")
    return float(np.median(v))


def gaussian_similarity(D: np.ndarray, sigma: float) -> np.ndarray:
    """S = exp(−d²/σ²); пропуск расстояния остаётся пропуском."""
    if not sigma > 0:
        raise ValueError(f"σ = {sigma}: нужен положительный масштаб")
    D = np.asarray(D, dtype=np.float64)
    return np.exp(-((D / sigma) ** 2)).astype(FLOAT)


# --- Корзина -------------------------------------------------------------------------------------


def cosine_matrix(X: np.ndarray) -> np.ndarray:
    """S_ij = ⟨x_i, x_j⟩ / (‖x_i‖·‖x_j‖); у нулевого вектора направления нет — его строка NaN."""
    X = _check_finite(X, "cosine_matrix")
    norm = np.linalg.norm(X, axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        U = X / np.where(norm > 0, norm, np.nan)
    S = np.clip(U @ U.T, -1.0, 1.0)
    return _no_loops(S, True)


def euclid_matrix(X: np.ndarray) -> np.ndarray:
    """d_ij = ‖x_i − x_j‖ (для CLR-векторов — расстояние Эйчисона); диагональ — NaN."""
    X = _check_finite(X, "euclid_matrix")
    sq = (X**2).sum(axis=1)
    D2 = np.maximum(sq[:, None] + sq[None, :] - 2.0 * X @ X.T, 0.0)
    return _no_loops(np.sqrt(D2), True)


# --- Ряды ----------------------------------------------------------------------------------------


def corr_matrix(X: np.ndarray, Y: np.ndarray | None = None) -> np.ndarray:
    """ρ_ij = corr(x_i, y_j) Пирсона по всем столбцам; ``Y`` не задан — ``Y = X``, диагональ NaN."""
    X = _check_finite(X, "corr_matrix")
    Zx = zscore_rows(X)
    Zy = Zx if Y is None else zscore_rows(_check_finite(Y, "corr_matrix"))
    S = np.clip(Zx @ Zy.T / X.shape[1], -1.0, 1.0)
    return _no_loops(S, Y is None)


def rank_rows(X: np.ndarray) -> np.ndarray:
    """Ранги значений в каждой строке (средние ранги у равных)."""
    from scipy.stats import rankdata

    return rankdata(_check_finite(X, "rank_rows"), axis=1)


def spearman_matrix(X: np.ndarray, Y: np.ndarray | None = None) -> np.ndarray:
    """ρ Спирмена: Пирсон рангов строк."""
    return corr_matrix(rank_rows(X), None if Y is None else rank_rows(Y))


def lag_order(max_lag: int) -> list[int]:
    """Сдвиги 0, +1, −1, +2, −2, …: при равных корреляциях выигрывает меньший |ℓ|."""
    out = [0]
    for lag in range(1, max_lag + 1):
        out += [lag, -lag]
    return out


def _stacked_z(mats: list[np.ndarray]) -> np.ndarray:
    """Сцепленные z-ряды категорий, делённые на √(число столбцов): Z_x·Z_yᵀ = среднее корреляций."""
    Z = np.hstack([zscore_rows(m) for m in mats])
    return Z / np.sqrt(Z.shape[1])


def _lag_corr_mean(Xs: list[np.ndarray], Ys: list[np.ndarray], lag: int) -> np.ndarray:
    """Среднее по категориям corr(x_i(t), y_j(t + ℓ)) по n − |ℓ| общим месяцам (свои средние и отклонения)."""
    n = Xs[0].shape[1]
    if lag >= 0:
        a, b = [x[:, : n - lag] for x in Xs], [y[:, lag:] for y in Ys]
    else:
        a, b = [x[:, -lag:] for x in Xs], [y[:, : n + lag] for y in Ys]
    return np.clip(_stacked_z(a) @ _stacked_z(b).T, -1.0, 1.0)


def lagged_corr(
    X: np.ndarray | list[np.ndarray], max_lag: int, Y: np.ndarray | list[np.ndarray] | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """S_ij = max_{|ℓ| ≤ L} ρ_ij(ℓ), ρ_ij(ℓ) = corr(x_i(t), y_j(t + ℓ)) по общим месяцам; ℓ* — аргмаксимум.

    ℓ* > 0: движение ряда i повторяется в ряду j через ℓ* месяцев (i опережает j). Несколько категорий
    (список матриц) — на каждом сдвиге среднее корреляций по категориям, затем максимум: сдвиг один для
    всех категорий пары. Без ``Y`` матрица симметрична (ρ_ij(ℓ) = ρ_ji(−ℓ)), а ℓ* антисимметрична.
    """
    Xs = [_check_finite(x, "lagged_corr") for x in (X if isinstance(X, list) else [X])]
    Ys = Xs if Y is None else [_check_finite(y, "lagged_corr") for y in (Y if isinstance(Y, list) else [Y])]
    n = Xs[0].shape[1]
    if not 0 <= max_lag <= n - 3:
        raise ValueError(f"max_lag = {max_lag}: нужно 0…{n - 3} при длине ряда {n}")
    best = np.full((Xs[0].shape[0], Ys[0].shape[0]), -np.inf)
    arg = np.zeros(best.shape, dtype=np.int8)
    for lag in lag_order(max_lag):
        C = _lag_corr_mean(Xs, Ys, lag)
        better = C > best
        best = np.where(better, C, best)
        arg = np.where(better, np.int8(lag), arg)
    S = _no_loops(np.clip(best, -1.0, 1.0), Y is None)
    if Y is None:
        np.fill_diagonal(arg, 0)
    return S, arg


def dtw_matrix(X: np.ndarray, window: int, Y: np.ndarray | None = None) -> np.ndarray:
    """d_ij = DTW(z(x_i), z(y_j)) c окном Сакоэ — Чибы: путь выравнивания не отходит от диагонали дальше
    ``window`` месяцев (в dtaidistance это параметр ``window = w + 1``). Ряды z-нормируются по окну.
    Считает C-версия ``dtaidistance`` (``distance_matrix_fast``); без неё — ``ImportError``."""
    import logging

    from dtaidistance import dtw

    logging.getLogger("be.kuleuven.dtai.distance").setLevel(logging.WARNING)  # строка лога на каждый вызов
    if not dtw.try_import_c():
        raise ImportError("dtaidistance без C-расширения: DTW на всех парах слишком медленный")
    Zx = zscore_rows(_check_finite(X, "dtw_matrix"))
    if not np.isfinite(Zx).all():
        raise ValueError("dtw_matrix: постоянный ряд — z-нормировка не определена")
    if Y is None:
        D = dtw.distance_matrix_fast(np.ascontiguousarray(Zx), window=window + 1)
        D = np.asarray(D, dtype=np.float64)
        D = np.minimum(D, D.T)  # на случай незаполненного нижнего треугольника
        return _no_loops(D, True)
    Zy = zscore_rows(_check_finite(Y, "dtw_matrix"))
    n = Zx.shape[0]
    stacked = np.ascontiguousarray(np.vstack([Zx, Zy]))
    D = dtw.distance_matrix_fast(stacked, window=window + 1, block=((0, n), (n, n + Zy.shape[0])))
    return np.asarray(D, dtype=np.float64)[:n, n:].astype(FLOAT)


# --- Нулевая модель: циклический сдвиг -----------------------------------------------------------


def allowed_shifts(n: int, reach: int, guard: int, period: int = PERIOD) -> np.ndarray:
    """Сдвиги s ∈ 1…n−1, у которых min(s mod p, p − s mod p) ≥ reach + guard (сезон не совпадает)."""
    s = np.arange(1, n)
    m = s % period
    circ = np.minimum(m, period - m)
    out = s[circ >= reach + guard]
    if not len(out):
        raise ValueError(f"нет допустимых сдвигов: длина {n}, досягаемость {reach}, запас {guard}")
    return out


def shift_rows(X: np.ndarray, shifts: np.ndarray) -> np.ndarray:
    """Строка i циклически сдвинута на shifts[i]: y_i(t) = x_i(t − s_i mod n)."""
    X = np.asarray(X)
    n = X.shape[1]
    idx = (np.arange(n)[None, :] - np.asarray(shifts)[:, None]) % n
    return np.take_along_axis(X, idx, axis=1)


def null_similarities(
    cross: Callable[[np.ndarray, np.ndarray], np.ndarray],
    X: np.ndarray,
    shifts: np.ndarray,
    repeats: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Нулевое распределение сходства: ``cross(сдвинутые X, X)`` для всех пар i ≠ j, ``repeats`` раз со
    случайным допустимым сдвигом каждого узла. Возвращает отсортированный массив (float32)."""
    parts = []
    n = X.shape[0]
    off = ~np.eye(n, dtype=bool)
    for _ in range(repeats):
        s = rng.choice(shifts, size=n)
        M = np.asarray(cross(shift_rows(X, s), X), dtype=FLOAT)
        v = M[off]
        parts.append(v[np.isfinite(v)])
    return np.sort(np.concatenate(parts))


def empirical_p(values: np.ndarray, null_sorted: np.ndarray) -> np.ndarray:
    """p = (1 + #{нуль ≥ S}) / (1 + размер нуля): верхний хвост, никогда не 0."""
    values = np.asarray(values, dtype=np.float64)
    ge = len(null_sorted) - np.searchsorted(null_sorted, values, side="left")
    p = (1.0 + ge) / (1.0 + len(null_sorted))
    return np.where(np.isfinite(values), p, np.nan)


def bh_qvalues(p: np.ndarray) -> np.ndarray:
    """q-значения Бенджамини — Хохберга: q_(i) = min_{j ≥ i} p_(j)·m / j; пропуски не считаются."""
    p = np.asarray(p, dtype=np.float64)
    q = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    m = int(ok.sum())
    if not m:
        return q
    pv = p[ok]
    order = np.argsort(pv, kind="stable")
    ranked = pv[order] * m / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(ranked, 1.0)
    q[ok] = out
    return q


# --- Несколько категорий -------------------------------------------------------------------------


def corr_multi(
    Xs: list[np.ndarray], Ys: list[np.ndarray] | None = None, method: str = "pearson"
) -> np.ndarray:
    """ρ_ij по нескольким категориям своего ритма: среднее корреляций по категориям, ρ_ij = (1/C)·Σ_c ρ^c_ij.
    Считается одним произведением сцепленных z-рядов (сцепленный вектор с равным весом категорий).
    ``method`` — ``pearson`` или ``spearman`` (ранги внутри каждой категории — проверка на выбросы)."""
    prep = (lambda m: _check_finite(m, "corr_multi")) if method == "pearson" else rank_rows
    Zx = _stacked_z([prep(x) for x in Xs])
    Zy = Zx if Ys is None else _stacked_z([prep(y) for y in Ys])
    return _no_loops(np.clip(Zx @ Zy.T, -1.0, 1.0), Ys is None)


def dtw_multi(Xs: list[np.ndarray], window: int, Ys: list[np.ndarray] | None = None) -> np.ndarray:
    """Среднее по категориям расстояний DTW (у каждой категории своё выравнивание)."""
    ys = [None] * len(Xs) if Ys is None else Ys
    return np.mean([dtw_matrix(x, window, y) for x, y in zip(Xs, ys, strict=True)], axis=0).astype(FLOAT)
