"""Разрежение матрицы сходства в сеть, паспорт сети, зонд кластеров и сравнение сетей.

Рёбра — неориентированные пары (i, j), i < j, по номерам узлов 0…n−1; вес — сходство S_ij правила.

Разрежение:

- ``knn`` — объединение списков k ближайших: ребро, если j среди k самых похожих на i или i среди k
  самых похожих на j (у каждого узла степень не меньше k — изолятов нет);
- ``mutual_knn`` — пересечение (взаимные k ближайших): ребро, только если оба в списках друг друга;
- ``threshold`` — все пары из маски (значимые после поправки Бенджамини — Хохберга или выше квантиля).

k-встречаемость N_k(x) — сколько раз узел x входит в списки k ближайших других узлов; асимметрия её
распределения — признак хабов (Radovanović и др., 2010).
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

EARTH_KM = 6371.0088


# --- Разрежение ----------------------------------------------------------------------------------


def knn_lists(S: np.ndarray, k: int) -> np.ndarray:
    """Номера k самых похожих узлов для каждого узла (n × k); пропуск сходства и сам узел не берутся,
    равные сходства — по меньшему номеру (устойчивая сортировка). Недостающие места — −1."""
    S = np.asarray(S, dtype=np.float64)
    n = S.shape[0]
    if not 1 <= k < n:
        raise ValueError(f"k = {k}: нужно 1…{n - 1}")
    key = np.where(np.isfinite(S), -S, np.inf)
    np.fill_diagonal(key, np.inf)
    # k наименьших ключей строки (argpartition), затем порядок внутри них: ключ, при равенстве — номер
    part = np.argpartition(key, k - 1, axis=1)[:, :k]
    vals = np.take_along_axis(key, part, axis=1)
    order = np.lexsort((part, vals), axis=1)
    top = np.take_along_axis(part, order, axis=1)
    valid = np.isfinite(np.take_along_axis(key, top, axis=1))
    return np.where(valid, top, -1)


def kocc(lists: np.ndarray, n: int) -> np.ndarray:
    """k-встречаемость N_k: сколько раз узел встречается в чужих списках k ближайших."""
    v = lists[lists >= 0]
    return np.bincount(v, minlength=n)


def _pairs(rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    a, b = np.minimum(rows, cols), np.maximum(rows, cols)
    return np.stack([a, b], axis=1)


def knn_edges(S: np.ndarray, k: int, mutual: bool = False) -> pd.DataFrame:
    """Рёбра kNN (объединение) или взаимного kNN: колонки ``i``, ``j`` (i < j), ``weight`` = S_ij."""
    lists = knn_lists(S, k)
    n = lists.shape[0]
    rows = np.repeat(np.arange(n), lists.shape[1])
    cols = lists.ravel()
    ok = cols >= 0
    pairs = _pairs(rows[ok], cols[ok])
    uniq, counts = np.unique(pairs, axis=0, return_counts=True)
    if mutual:
        uniq = uniq[counts == 2]
    return edge_frame(uniq, S)


def threshold_edges(S: np.ndarray, mask: np.ndarray) -> pd.DataFrame:
    """Рёбра всех пар i < j, где ``mask`` истинна (маска считается симметричной)."""
    iu = np.triu_indices(S.shape[0], k=1)
    keep = np.asarray(mask)[iu] & np.isfinite(np.asarray(S)[iu])
    pairs = np.stack([iu[0][keep], iu[1][keep]], axis=1)
    return edge_frame(pairs, S)


def top_pairs_mask(S: np.ndarray, n_edges: int) -> np.ndarray:
    """Маска ``n_edges`` самых похожих пар (порог-квантиль той же плотности, что у сети для сравнения)."""
    v = np.asarray(S, dtype=np.float64)[np.triu_indices(S.shape[0], k=1)]
    v = v[np.isfinite(v)]
    if n_edges <= 0 or not len(v):
        return np.zeros(S.shape, dtype=bool)
    thr = np.sort(v)[::-1][min(n_edges, len(v)) - 1]
    return np.asarray(S) >= thr


def edge_frame(pairs: np.ndarray, S: np.ndarray) -> pd.DataFrame:
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    w = np.asarray(S)[pairs[:, 0], pairs[:, 1]].astype(np.float64)
    out = pd.DataFrame({"i": pairs[:, 0], "j": pairs[:, 1], "weight": w})
    return out.sort_values(["i", "j"], kind="mergesort").reset_index(drop=True)


def edge_set(edges: pd.DataFrame, ids: Sequence[int] | None = None) -> set[tuple[int, int]]:
    """Множество рёбер; с ``ids`` — в ключах узлов (territory_id), чтобы сравнивать сети разных узлов."""
    i, j = edges["i"].to_numpy(dtype=np.int64), edges["j"].to_numpy(dtype=np.int64)
    if ids is not None:
        ids = np.asarray(ids, dtype=np.int64)
        i, j = ids[i], ids[j]
    return set(zip(np.minimum(i, j).tolist(), np.maximum(i, j).tolist(), strict=True))


def jaccard(a: set, b: set) -> float:
    union = len(a | b)
    return len(a & b) / union if union else float("nan")


# --- Паспорт -------------------------------------------------------------------------------------


def to_igraph(n: int, edges: pd.DataFrame):
    import igraph as ig

    g = ig.Graph(n=n, edges=list(zip(edges["i"].tolist(), edges["j"].tolist(), strict=True)))
    g.es["weight"] = edges["weight"].astype(float).tolist()
    return g


def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * EARTH_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _skew(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    sd = x.std()
    return float(((x - x.mean()) ** 3).mean() / sd**3) if sd > 0 else 0.0


@dataclass(frozen=True)
class NodeInfo:
    """Сведения об узлах сети в порядке номеров 0…n−1."""

    ids: np.ndarray  # territory_id
    region: np.ndarray  # группа региона
    lat: np.ndarray
    lon: np.ndarray
    attrs: pd.DataFrame  # атрибуты места (индекс 0…n−1), могут быть пропуски


def assortativity(g, values: np.ndarray) -> float:
    """Ассортативность Ньюмена по числовому атрибуту на подграфе узлов без пропуска."""
    values = np.asarray(values, dtype=np.float64)
    keep = np.flatnonzero(np.isfinite(values))
    if len(keep) < 3:
        return float("nan")
    sub = g.induced_subgraph(keep.tolist())
    if sub.ecount() < 2 or np.nanstd(values[keep]) == 0:
        return float("nan")
    return float(sub.assortativity(values[keep].tolist(), directed=False))


def passport(
    edges: pd.DataFrame,
    info: NodeInfo,
    lists: np.ndarray | None = None,
    attr_columns: Sequence[str] = (),
) -> dict[str, float]:
    """Паспорт сети: рёбра, плотность, компоненты, степени, k-встречаемость, кластеризация, ассортативность
    по группе региона и атрибутам, доля рёбер внутри группы региона, медианная длина ребра (км по прямой)."""
    n = len(info.ids)
    g = to_igraph(n, edges)
    deg = np.asarray(g.degree())
    comps = g.connected_components()
    sizes = np.asarray(comps.sizes())
    i, j = edges["i"].to_numpy(), edges["j"].to_numpy()
    km = haversine_km(info.lat[i], info.lon[i], info.lat[j], info.lon[j]) if len(edges) else np.array([])
    _, region_codes = np.unique(info.region, return_inverse=True)
    out: dict[str, float] = {
        "n_nodes": n,
        "n_edges": len(edges),
        "density": 2 * len(edges) / (n * (n - 1)) if n > 1 else float("nan"),
        "n_components": len(sizes),
        "giant_share": float(sizes.max() / n) if n else float("nan"),
        "isolates": int((deg == 0).sum()),
        "deg_median": float(np.median(deg)),
        "deg_max": int(deg.max()) if n else 0,
        "clustering": float(g.transitivity_avglocal_undirected(mode="zero")),
        "assort_region": float(g.assortativity_nominal(region_codes.tolist(), directed=False))
        if len(edges)
        else float("nan"),
        "within_region": float((info.region[i] == info.region[j]).mean()) if len(edges) else float("nan"),
        "edge_km_median": float(np.median(km)) if len(km) else float("nan"),
    }
    if lists is not None:
        occ = kocc(lists, n)
        out["kocc_skew"] = _skew(occ)
        out["kocc_max"] = int(occ.max())
    for col in attr_columns:
        out[f"assort_{col}"] = assortativity(g, info.attrs[col].to_numpy(dtype=np.float64))
    if attr_columns:
        vals = [out[f"assort_{c}"] for c in attr_columns]
        out["attr_consistency"] = float(np.nanmean(vals)) if np.isfinite(vals).any() else float("nan")
    return out


def degree_of(edges: pd.DataFrame, n: int) -> np.ndarray:
    return np.bincount(edges["i"].to_numpy(), minlength=n) + np.bincount(edges["j"].to_numpy(), minlength=n)


# --- Зонд кластеров: Leiden и нулевая модель -----------------------------------------------------


def leiden(g, seed: int, resolution: float) -> tuple[np.ndarray, float]:
    """Разбиение Leiden (модульность с разрешением γ, без весов) и его модульность Q."""
    import leidenalg

    part = leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,
        resolution_parameter=resolution,
        seed=int(seed),
        n_iterations=-1,
    )
    return np.asarray(part.membership), float(g.modularity(part.membership, resolution=resolution))


def rewired(g, seed: int, factor: int):
    """Копия сети с перемешанными рёбрами: степени узлов сохранены, петель и кратных рёбер нет."""
    import igraph as ig

    h = g.copy()
    ig.set_random_number_generator(random.Random(seed))
    try:
        h.rewire(n=factor * h.ecount(), allowed_edge_types="simple")
    finally:
        ig.set_random_number_generator(random)
    return h


@dataclass(frozen=True)
class ProbeResult:
    membership: np.ndarray  # лучшее по Q разбиение из seeds
    q: float
    q_null_mean: float
    q_null_sd: float
    n_comm: int
    ari_seeds: float  # медианный ARI разбиений разных seed

    @property
    def q_z(self) -> float:
        return (self.q - self.q_null_mean) / self.q_null_sd if self.q_null_sd > 0 else float("nan")

    @property
    def q_excess(self) -> float:
        return self.q - self.q_null_mean


def probe(
    n: int, edges: pd.DataFrame, seeds: Sequence[int], resolution: float, null_repeats: int, factor: int
) -> ProbeResult:
    """Фиксированный зонд: Leiden с разными seed; модульность сравнивается с Leiden на перемешанных сетях
    с теми же степенями (сырая модульность сетей разной плотности несопоставима)."""
    from sklearn.metrics import adjusted_rand_score

    g = to_igraph(n, edges)
    runs = [leiden(g, s, resolution) for s in seeds]
    best = max(range(len(runs)), key=lambda r: runs[r][1])
    aris = [
        adjusted_rand_score(runs[a][0], runs[b][0]) for a in range(len(runs)) for b in range(a + 1, len(runs))
    ]
    nulls = [
        leiden(rewired(g, 10_000 + r + int(seeds[0]), factor), int(seeds[0]), resolution)[1]
        for r in range(null_repeats)
    ]
    return ProbeResult(
        membership=runs[best][0],
        q=runs[best][1],
        q_null_mean=float(np.mean(nulls)) if nulls else float("nan"),
        q_null_sd=float(np.std(nulls, ddof=1)) if len(nulls) > 1 else float("nan"),
        n_comm=int(len(np.unique(runs[best][0]))),
        ari_seeds=float(np.median(aris)) if aris else float("nan"),
    )


# --- Сравнение сетей -----------------------------------------------------------------------------


def rank_corr_upper(A: np.ndarray, B: np.ndarray) -> float:
    """ρ Спирмена сходств двух правил по всем парам i < j, где оба сходства определены."""
    from scipy.stats import rankdata

    a, b = A[np.triu_indices(A.shape[0], 1)], B[np.triu_indices(B.shape[0], 1)]
    ok = np.isfinite(a) & np.isfinite(b)
    ra, rb = rankdata(a[ok]), rankdata(b[ok])
    return float(np.corrcoef(ra, rb)[0, 1])


def mantel_ranks(
    A: np.ndarray, B: np.ndarray, permutations: int, rng: np.random.Generator
) -> tuple[float, float]:
    """Ранговая корреляция сходств двух правил по парам i < j и её перестановочное p (тест Мантела:
    перестановка узлов второго правила): p = (1 + #{|ρ_perm| ≥ |ρ|}) / (1 + перестановок).

    Ранги — по парам, где определены оба сходства; пропуск (например, нет дорожного расстояния) получает
    средний ранг и в корреляцию не вносит вклада. Перестановка узлов переставляет пары, не меняя набора
    значений, поэтому среднее и норма рангов второго правила от неё не зависят: ρ_perm — скалярное
    произведение."""
    from scipy.stats import rankdata

    n = A.shape[0]
    iu = np.triu_indices(n, 1)
    a, b = A[iu].astype(np.float64), B[iu].astype(np.float64)
    ok = np.isfinite(a) & np.isfinite(b)

    def centered(v: np.ndarray) -> np.ndarray:
        r = np.zeros(len(v))
        r[ok] = rankdata(v[ok])
        r[ok] -= r[ok].mean()
        return r

    ra, rb = centered(a), centered(b)
    RB = np.zeros((n, n))
    RB[iu] = rb
    RB = RB + RB.T
    norm = np.sqrt((ra**2).sum() * (rb**2).sum())
    obs = float(ra @ rb / norm)
    hits = 0
    for _ in range(permutations):
        q = rng.permutation(n)
        if abs(ra @ RB[q[iu[0]], q[iu[1]]] / norm) >= abs(obs):
            hits += 1
    return obs, (1 + hits) / (1 + permutations)


def ari_on_common(a: Mapping[int, int], b: Mapping[int, int]) -> float:
    """ARI двух разбиений на общих узлах (ключи — territory_id)."""
    from sklearn.metrics import adjusted_rand_score

    common = sorted(set(a) & set(b))
    if len(common) < 2:
        return float("nan")
    return float(adjusted_rand_score([a[c] for c in common], [b[c] for c in common]))
