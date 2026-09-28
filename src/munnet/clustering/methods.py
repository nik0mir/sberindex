"""Методы кластеризации трёх семейств и базовая линия — единый интерфейс «входы, K или параметр, seed →
метки».

По признакам X: K-means, Ward, GMM, HDBSCAN. По графу G: Leiden, Louvain (модульность с разрешением γ, веса
рёбер), спектральная (affinity — матрица весов G). На (G, X): KEFRiN Шалилеха и Миркина (``kefrin``) и гибрид
«спектральное вложение G размерности K (нормированный лапласиан) ⊕ X, K-means». Базовая линия
``kmeans_joint`` —
K-means по координатам корзины (из них построен G) ⊕ X. В гибриде и базовой линии у двух блоков равный
суммарный вес: блок делится на корень своей суммарной дисперсии и умножается на √α и √(1 − α).

``fit`` — один запуск с одним seed (так считается бутстрап: у каждой выборки свой seed); ``fit_protocol`` —
итог метода по ``protocol.seeds``: лучший запуск по собственному критерию (инерция K-means, правдоподобие GMM,
модульность Leiden и Louvain, критерий F KEFRiN), у спектральной — медоид по ARI. ``tune_param`` подбирает
разрешение γ и min_cluster_size только по получившемуся K (без критериев выбора).
"""

from __future__ import annotations

import logging
import random
import time
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.sparse as sp

from munnet.clustering import kefrin as KF

log = logging.getLogger(__name__)

NOISE = -1
SCORE_TOL = 1e-9  # относительный допуск равенства собственного критерия у запусков разных seed


@dataclass(frozen=True)
class Fit:
    labels: np.ndarray
    score: float  # собственный критерий метода: больше — лучше (NaN — нет)
    extra: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class ProtocolFit:
    labels: np.ndarray
    score: float
    seed: int | None
    seed_ari: float  # медианный ARI запусков разных seed (NaN у детерминированных)
    seconds: float
    extra: dict[str, float] = field(default_factory=dict)


# --- Вспомогательные ------------------------------------------------------------------------------


def canonical(labels: np.ndarray) -> np.ndarray:
    """Метки 0…K−1 по убыванию размера кластера (при равенстве — по первому узлу); шум −1 сохраняется."""
    labels = np.asarray(labels)
    out = np.full(len(labels), NOISE, dtype=np.int64)
    ok = labels != NOISE
    vals, first, counts = np.unique(labels[ok], return_index=True, return_counts=True)
    order = sorted(range(len(vals)), key=lambda i: (-counts[i], first[i]))
    mapping = {vals[i]: r for r, i in enumerate(order)}
    out[ok] = [mapping[v] for v in labels[ok]]
    return out


def n_clusters(labels: np.ndarray) -> int:
    return int(len(np.unique(labels[labels != NOISE])))


def block_join(blocks: Sequence[np.ndarray], weights: Sequence[float]) -> np.ndarray:
    """Блоки с суммарным весом ``weights``: блок / √(сумма дисперсий столбцов) × √вес."""
    parts = []
    for B, w in zip(blocks, weights, strict=True):
        if w <= 0:
            continue
        B = np.asarray(B, dtype=np.float64)
        B = B - B.mean(axis=0)
        tv = float(B.var(axis=0).sum())
        parts.append(B / np.sqrt(tv if tv > 0 else 1.0) * np.sqrt(w))
    return np.hstack(parts)


def to_igraph(A: sp.csr_matrix):
    import igraph as ig

    U = sp.triu(A, k=1).tocoo()
    g = ig.Graph(n=A.shape[0], edges=list(zip(U.row.tolist(), U.col.tolist(), strict=True)))
    g.es["weight"] = U.data.astype(float).tolist()
    return g


def spectral_embed(A: sp.csr_matrix, k: int, seed: int) -> np.ndarray:
    """Вложение, как в ``SpectralClustering``: нормированный лапласиан, первые k векторов
    (``drop_first=False``)."""
    from sklearn.manifold import spectral_embedding

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # несвязный подграф бутстрапа: предупреждение, вложение считается
        return spectral_embedding(A, n_components=k, drop_first=False, random_state=seed)


def _kmeans(Z: np.ndarray, k: int, seed: int, n_init: int) -> Fit:
    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=k, n_init=n_init, random_state=seed).fit(Z)
    return Fit(km.labels_, -float(km.inertia_))


# --- Методы ---------------------------------------------------------------------------------------


def fit(
    method: str,
    inp,
    k: int,
    param: float | None,
    seed: int,
    impl: Mapping[str, Any],
    cache=None,
    alpha: float = 0.5,
) -> Fit:
    """Один запуск метода; ``k`` — число кластеров (у Leiden, Louvain и HDBSCAN не используется: K задаёт
    ``param``); ``alpha`` — вес графа в гибриде. ``cache`` — словарь на одни входы (дерево Ward,
    стандартизованная сеть KEFRiN, граф igraph)."""
    cache = {} if cache is None else cache
    n_init = int(impl.get("kmeans_n_init", 20))
    X = inp.X
    if method == "kmeans":
        return _kmeans(X, k, seed, n_init)
    if method == "kmeans_joint":
        if "joint" not in cache:
            cache["joint"] = block_join([inp.B, X], [0.5, 0.5])
        return _kmeans(cache["joint"], k, seed, n_init)
    if method == "ward":
        from scipy.cluster.hierarchy import cut_tree, linkage

        if "ward" not in cache:
            cache["ward"] = linkage(X, method="ward")
        return Fit(cut_tree(cache["ward"], n_clusters=k).ravel(), float("nan"))
    if method == "gmm":
        from sklearn.mixture import GaussianMixture

        gm = GaussianMixture(n_components=k, random_state=seed, **dict(impl.get("gmm") or {}))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gm.fit(X)
        return Fit(gm.predict(X), float(gm.score(X)), {"bic": float(gm.bic(X))})
    if method == "hdbscan":
        from sklearn.cluster import HDBSCAN

        labels = HDBSCAN(min_cluster_size=int(param), copy=True).fit(X).labels_
        return Fit(labels, float("nan"))
    if method in ("leiden", "louvain"):
        if "igraph" not in cache:
            cache["igraph"] = to_igraph(inp.A)
        g = cache["igraph"]
        gamma = float(param)
        if method == "leiden":
            import leidenalg

            part = leidenalg.find_partition(
                g,
                leidenalg.RBConfigurationVertexPartition,
                weights="weight",
                resolution_parameter=gamma,
                seed=int(seed),
                n_iterations=-1,
            )
            memb = np.asarray(part.membership)
        else:
            import igraph as ig

            ig.set_random_number_generator(random.Random(int(seed)))
            try:
                memb = np.asarray(g.community_multilevel(weights="weight", resolution=gamma).membership)
            finally:
                ig.set_random_number_generator(random)
        q = float(g.modularity(memb.tolist(), weights="weight", resolution=gamma))
        return Fit(memb, q)
    if method == "spectral":
        from sklearn.cluster import SpectralClustering

        sc = SpectralClustering(
            n_clusters=k,
            affinity="precomputed",
            random_state=seed,
            assign_labels=str((impl.get("spectral") or {}).get("assign_labels", "kmeans")),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            labels = sc.fit(inp.A).labels_
        return Fit(labels, float("nan"))
    if method == "hybrid":
        E = spectral_embed(inp.A, k, seed) if alpha > 0 else np.zeros((inp.n, 0))
        Z = block_join([E, X], [alpha, 1.0 - alpha])
        return _kmeans(Z, k, seed, n_init)
    if method == "shalileh_mirkin":
        sm = impl.get("shalileh_mirkin") or {}
        if "kefrin_P" not in cache:
            if str(sm.get("network", "modularity")) != "modularity":
                raise ValueError("KEFRiN: реализована стандартизация сети modularity (M)")
            cache["kefrin_P"] = KF.ModularityLinks(inp.A)  # (M) без плотной матрицы n × n
            Y = X
            if str(sm.get("features", "inputs")) == "zscore":
                Y = (X - X.mean(axis=0)) / np.where(X.std(axis=0) > 0, X.std(axis=0), 1.0)
            cache["kefrin_Y"] = Y
        first = int(np.random.default_rng(seed).integers(inp.n))
        res = KF.kefrin(
            cache["kefrin_Y"],
            cache["kefrin_P"],
            k,
            first,
            rho=float(sm.get("rho", 1.0)),
            xi=float(sm.get("xi", 1.0)),
            max_iter=int(sm.get("max_iter", 100)),
        )
        return Fit(
            res.labels,
            -res.criterion,
            {"iterations": float(res.iterations), "converged": float(res.converged)},
        )
    raise ValueError(f"неизвестный метод {method}")


def _median_ari(runs: list[np.ndarray]) -> float:
    from sklearn.metrics import adjusted_rand_score

    if len(runs) < 2:
        return float("nan")
    v = [adjusted_rand_score(runs[a], runs[b]) for a in range(len(runs)) for b in range(a + 1, len(runs))]
    return float(np.median(v))


def fit_protocol(
    method: str,
    inp,
    k: int,
    param: float | None,
    seeds: Sequence[int],
    impl,
    random_methods,
    cache=None,
    alpha: float = 0.5,
) -> ProtocolFit:
    """Итог метода: у детерминированных — один запуск; у случайных — ``seeds`` запусков, лучший по
    собственному
    критерию (равенство — меньший seed), у методов без критерия — медоид по ARI."""
    from sklearn.metrics import adjusted_rand_score

    cache = {} if cache is None else cache
    t0 = time.perf_counter()
    use = list(seeds) if method in random_methods else [int(seeds[0])]
    fits = [fit(method, inp, k, param, s, impl, cache, alpha) for s in use]
    runs = [f.labels for f in fits]
    scores = np.array([f.score for f in fits])
    if len(fits) == 1:
        best = 0
    elif np.isfinite(scores).all():
        # равные с точностью до округления (сумма BLAS в разных потоках) — меньший seed: итог повторяем
        top = float(scores.max())
        best = int(np.flatnonzero(scores >= top - SCORE_TOL * max(1.0, abs(top)))[0])
    else:  # медоид: наибольшая сумма ARI с остальными запусками
        M = np.array([[adjusted_rand_score(a, b) for b in runs] for a in runs])
        best = int(np.argmax(M.sum(axis=1)))
    return ProtocolFit(
        labels=canonical(runs[best]),
        score=float(scores[best]),
        seed=int(use[best]) if method in random_methods else None,
        seed_ari=_median_ari(runs),
        seconds=time.perf_counter() - t0,
        extra=dict(fits[best].extra),
    )


# --- Подбор параметра, задающего K ---------------------------------------------------------------


@dataclass
class ParamScan:
    method: str
    points: dict[float, int] = field(default_factory=dict)  # параметр -> K итога протокола

    def k_of(self, value: float) -> int:
        return self.points[value]


def _pick(values: list[float]) -> float:
    """Из значений параметра с одним K — медианное (нижняя середина): не граница диапазона."""
    values = sorted(values)
    return values[(len(values) - 1) // 2]


def tune_param(
    method: str, inp, k_grid: Sequence[int], seeds, impl, random_methods
) -> tuple[dict[int, float], ParamScan]:
    """Параметр, дающий каждое K сетки: логарифмическая сетка, затем деление пополам между соседними точками
    с K по обе стороны от нужного. K — у итога протокола (лучший из seeds). Недостижимое K — нет в ответе."""
    scan = ParamScan(method)
    cache: dict = {}

    def k_at(value: float) -> int:
        if value not in scan.points:
            pf = fit_protocol(method, inp, 2, value, seeds, impl, random_methods, cache)
            scan.points[value] = n_clusters(pf.labels)
        return scan.points[value]

    if method == "hdbscan":
        g = impl.get("hdbscan_grid") or {}
        lo, hi, pts = int(g.get("min_size", 5)), int(g.get("max_size", 400)), int(g.get("points", 40))
        hi = min(hi, max(lo + 1, inp.n // 2))
        grid = sorted({int(round(v)) for v in np.geomspace(lo, hi, pts)})
        refine = 12
        integer = True
    else:
        g = impl.get("leiden_grid") or {}
        grid = list(
            np.geomspace(
                float(g.get("gamma_min", 0.005)), float(g.get("gamma_max", 5.0)), int(g.get("points", 40))
            )
        )
        refine = int(g.get("refine", 12))
        integer = False
    for v in grid:
        k_at(v)
    for target in k_grid:
        if any(kk == target for kk in scan.points.values()):
            continue
        pts_sorted = sorted(scan.points.items())
        for (a, ka), (b, kb) in zip(pts_sorted, pts_sorted[1:], strict=False):
            if (ka - target) * (kb - target) >= 0:
                continue
            for _ in range(refine):
                if integer:
                    if b - a <= 1:
                        break
                    mid = float((int(a) + int(b)) // 2)
                else:
                    mid = float(np.sqrt(a * b))
                km = k_at(mid)
                if km == target:
                    break
                if (ka - target) * (km - target) < 0:
                    b, kb = mid, km
                else:
                    a, ka = mid, km
            if any(kk == target for kk in scan.points.values()):
                break
    chosen = {}
    for target in k_grid:
        vals = [v for v, kk in scan.points.items() if kk == target]
        if vals:
            chosen[target] = _pick(vals)
    return chosen, scan


# --- Ориентир K: iK-means Миркина ----------------------------------------------------------------


def anomalous_patterns(Z: np.ndarray) -> list[np.ndarray]:
    """Аномальные кластеры по одному (iK-means, Mirkin, 2005/2012): данные центрируются в общем центре; из
    оставшихся точек берётся самая далёкая от центра; кластер — точки, которые ближе к его центру, чем
    к общему центру; центр — среднее кластера; повтор до стабилизации; кластер удаляется, шаг повторяется."""
    Z = np.asarray(Z, dtype=np.float64)
    Z = Z - Z.mean(axis=0)
    remaining = np.ones(len(Z), dtype=bool)
    norms = (Z**2).sum(axis=1)
    out = []
    while remaining.any():
        idx = np.flatnonzero(remaining)
        c = Z[idx[np.argmax(norms[idx])]].copy()
        members = idx[[int(np.argmax(norms[idx]))]]
        for _ in range(1000):
            d = ((Z[idx] - c) ** 2).sum(axis=1)
            new = idx[d < norms[idx]]
            if not len(new):
                new = members
            c_new = Z[new].mean(axis=0)
            if np.array_equal(new, members) and np.allclose(c_new, c):
                break
            members, c = new, c_new
        out.append(members)
        remaining[members] = False
    return out


def ik_means_k(Z: np.ndarray, min_size: int) -> tuple[int, list[int]]:
    """Ориентир K: число аномальных кластеров не меньше ``min_size``; второе — размеры всех кластеров."""
    sizes = [len(s) for s in anomalous_patterns(Z)]
    return int(sum(s >= min_size for s in sizes)), sizes
