"""Матрицы сходства правил на окне месяцев и нулевые модели правил по рядам."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np

from munnet.network import geo
from munnet.network import rules as R
from munnet.network.data import NodeSet, window_clr
from munnet.network.params import NetworkParams, RuleSpec

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Similarity:
    """Матрица сходства правила: S (n × n, float32, диагональ NaN), сдвиг ℓ* (лаговое правило), масштаб σ
    ядра exp(−d²/σ²) и время расчёта (с)."""

    rule: str
    S: np.ndarray
    lag: np.ndarray | None = None
    sigma: float | None = None
    seconds: float = 0.0


class GeoCache:
    """Дорожные расстояния узлов считаются один раз на набор узлов."""

    def __init__(self, pairs, city_distance: str) -> None:
        self.pairs = pairs
        self.how = city_distance
        self._cache: dict[tuple, np.ndarray] = {}

    def distance(self, ns: NodeSet) -> np.ndarray:
        key = (ns.mode, ns.n)
        if key not in self._cache:
            self._cache[key] = geo.node_distance_matrix(self.pairs, ns.ids, ns.members, ns.mo_pop, self.how)
        return self._cache[key]


def similarity(rule: RuleSpec, ns: NodeSet, months, geo_cache: GeoCache | None = None) -> Similarity:
    """S правила ``rule`` на месяцах ``months`` (номера t). Правила географии от окна не зависят."""
    t0 = time.perf_counter()
    lag = sigma = None
    months = np.asarray(months)
    if rule.kind == "basket_cosine":
        S = R.cosine_matrix(window_clr(ns, months))
    elif rule.kind == "basket_distance":
        D = R.euclid_matrix(window_clr(ns, months))
        sigma = R.median_offdiag(D)
        S = R.gaussian_similarity(D, sigma)
    elif rule.kind == "rhythm_corr":
        S = R.corr_multi(ns.series(rule.categories, months))
    elif rule.kind == "rhythm_lag":
        S, lag = R.lagged_corr(ns.series(rule.categories, months), rule.max_lag)
    elif rule.kind == "rhythm_dtw":
        D = R.dtw_multi(ns.series(rule.categories, months), rule.window)
        sigma = R.median_offdiag(D)
        S = R.gaussian_similarity(D, sigma)
    elif rule.kind == "road":
        D = geo_cache.distance(ns)
        sigma = R.median_offdiag(D)
        S = R.gaussian_similarity(D, sigma)
    elif rule.kind == "gravity":
        S = geo.gravity_log(geo_cache.distance(ns), ns.pop, rule.beta, rule.min_km)
    else:  # pragma: no cover — вид проверяет NetworkParams
        raise ValueError(rule.kind)
    return Similarity(rule.name, S, lag, sigma, time.perf_counter() - t0)


def spearman_similarity(rule: RuleSpec, ns: NodeSet, months) -> np.ndarray:
    """Корреляционное правило с ρ Спирмена вместо Пирсона — проверка на выбросы."""
    return R.corr_multi(ns.series(rule.categories, np.asarray(months)), method="spearman")


def series_null(
    rule: RuleSpec, ns: NodeSet, months, p: NetworkParams, rng: np.random.Generator, sigma: float | None
) -> np.ndarray:
    """Нулевое распределение S правила по рядам: ряды всех категорий узла сдвинуты циклически на один
    случайный допустимый сдвиг (``rules.allowed_shifts``), сходство — со всеми несдвинутыми рядами."""
    months = np.asarray(months)
    Xs = ns.series(rule.categories, months)
    shifts = R.allowed_shifts(len(months), rule.reach, p.shift_guard)
    stacked = np.stack(Xs, axis=0)  # категории × узлы × месяцы: сдвиг один на все категории узла
    parts = []
    n = ns.n
    off = ~np.eye(n, dtype=bool)
    for _ in range(p.shift_repeats):
        s = rng.choice(shifts, size=n)
        shifted = [R.shift_rows(x, s) for x in stacked]
        if rule.kind == "rhythm_corr":
            M = R.corr_multi(shifted, list(stacked))
        elif rule.kind == "rhythm_lag":
            M, _ = R.lagged_corr(shifted, rule.max_lag, list(stacked))
        else:
            M = R.gaussian_similarity(R.dtw_multi(shifted, rule.window, list(stacked)), sigma)
        v = np.asarray(M, dtype=np.float32)[off]
        parts.append(v[np.isfinite(v)])
    return np.sort(np.concatenate(parts))


def qvalues(S: np.ndarray, null_sorted: np.ndarray) -> np.ndarray:
    """q-значения Бенджамини — Хохберга всех пар i < j (симметричная матрица, диагональ NaN)."""
    n = S.shape[0]
    iu = np.triu_indices(n, 1)
    q = R.bh_qvalues(R.empirical_p(S[iu], null_sorted))
    Q = np.full((n, n), np.nan)
    Q[iu] = q
    Q[(iu[1], iu[0])] = q
    return Q
