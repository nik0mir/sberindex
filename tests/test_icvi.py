"""Тесты индексов качества munnet.icvi на синтетике с известным ответом (без data/ и сети).

Ручные значения AVI и AVU — по формулам (18)–(21) из Shalileh, Antonov, Tsyplakova, Doklady Mathematics,
2025, т. 112, № 3, с. 555: внутренние рёбра — упорядоченные пары (сумма a_ij по i, j из кластера),
AVU = (1/K)·Σ_{k≠l}.
S_Dbw — Halkidi, Vazirgiannis, ICDM 2001, в изложении самих авторов: Halkidi, Batistakis, Vazirgiannis,
SIGMOD Record, 2002, т. 31, № 3, формулы (11), (14)–(17).
"""

import math

import igraph as ig
import networkx as nx
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from sklearn.datasets import make_blobs
from sklearn.metrics import calinski_harabasz_score, silhouette_score

from munnet import icvi
from munnet.config import load_config

ALL = ["sw", "ch", "s_dbw", "avi", "avu", "mq"]


# ---------------------------------------------------------------------------------------------
# Синтетика


def blobs(n_per=50, k=4, d=5, std=1.0, seed=0):
    x, y = make_blobs(
        n_samples=[n_per] * k, n_features=d, cluster_std=std, center_box=(-10, 10), random_state=seed
    )
    return x, y


def planted(sizes, p_in, p_out, seed=0, weighted=False):
    """Граф с посаженным разбиением (стохастическая блочная модель): симметричная CSR и истинные метки."""
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(len(sizes)), sizes)
    n = labels.size
    prob = np.where(labels[:, None] == labels[None, :], p_in, p_out)
    upper = np.triu(rng.random((n, n)) < prob, k=1)
    w = rng.uniform(0.1, 1.0, size=(n, n)) if weighted else np.ones((n, n))
    a = np.where(upper, w, 0.0)
    a = a + a.T
    return sp.csr_matrix(a), labels


def two_triangles(bridge=1.0, inner=1.0):
    """Два треугольника {0,1,2} и {3,4,5}, соединённые ребром 2–3."""
    edges = [(0, 1, inner), (0, 2, inner), (1, 2, inner), (3, 4, inner), (3, 5, inner), (4, 5, inner)]
    edges.append((2, 3, bridge))
    return _from_edges(6, edges), np.array([0, 0, 0, 1, 1, 1])


def _from_edges(n, edges):
    a = np.zeros((n, n))
    for i, j, w in edges:
        a[i, j] = a[j, i] = w
    return sp.csr_matrix(a)


def _nx_graph(a):
    g = nx.Graph()
    g.add_nodes_from(range(a.shape[0]))
    coo = sp.triu(a, k=1).tocoo()
    g.add_weighted_edges_from(zip(coo.row.tolist(), coo.col.tolist(), coo.data.tolist(), strict=True))
    return g


def _ig_graph(a):
    coo = sp.triu(a, k=1).tocoo()
    g = ig.Graph(n=a.shape[0], edges=list(zip(coo.row.tolist(), coo.col.tolist(), strict=True)))
    g.es["weight"] = coo.data.tolist()
    return g


# ---------------------------------------------------------------------------------------------
# Конфиг и контракт


def test_metric_specs_from_default_config():
    specs = icvi.metric_specs(load_config())
    assert list(specs) == ALL
    assert {n: (s.space, s.better) for n, s in specs.items()} == {
        "sw": ("features", "max"),
        "ch": ("features", "max"),
        "s_dbw": ("features", "min"),
        "avi": ("graph", "max"),
        "avu": ("graph", "min"),
        "mq": ("graph", "max"),
    }
    assert specs["avu"].name == "avu"


def test_metric_specs_rejects_unknown_space_or_direction():
    bad = {"icvi": {"metrics": {"sw": {"space": "time", "better": "max"}}}}
    with pytest.raises(ValueError, match="space"):
        icvi.metric_specs(bad)
    bad = {"icvi": {"metrics": {"sw": {"space": "features", "better": "up"}}}}
    with pytest.raises(ValueError, match="better"):
        icvi.metric_specs(bad)
    bad = {"icvi": {"metrics": {"dbi": {"space": "features", "better": "min"}}}}
    with pytest.raises(ValueError, match="dbi"):
        icvi.metric_specs(bad)


def test_config_directions_match_module_directions():
    """Направления в конфиге и в модуле не расходятся: zscore и выбор этапа 3 читают одно и то же."""
    specs = icvi.metric_specs(load_config())
    assert {n: s.better for n, s in specs.items()} == icvi.BETTER


def test_run_is_deferred_until_cluster():
    with pytest.raises(NotImplementedError, match="после cluster"):
        icvi.run(load_config())


# ---------------------------------------------------------------------------------------------
# Матрица смежности из таблицы рёбер


def test_adjacency_symmetric_in_node_order():
    edges = pd.DataFrame({"source": [10, 10, 20], "target": [20, 30, 30], "weight": [0.5, 1.0, 2.0]})
    a = icvi.adjacency(edges, [30, 20, 10])
    assert sp.issparse(a) and a.format == "csr"
    expected = np.array([[0, 2.0, 1.0], [2.0, 0, 0.5], [1.0, 0.5, 0]])
    np.testing.assert_allclose(a.toarray(), expected)


def test_adjacency_keeps_isolated_nodes_and_custom_weight():
    edges = pd.DataFrame({"source": [1], "target": [2], "w": [3.0]})
    a = icvi.adjacency(edges, [1, 2, 3], weight="w")
    assert a.shape == (3, 3)
    assert a[2].nnz == 0
    assert a[0, 1] == a[1, 0] == 3.0


@pytest.mark.parametrize(
    ("source", "target", "weight", "match"),
    [
        ([1, 1], [2, 2], [1.0, 1.0], "дубл"),
        ([1], [1], [1.0], "петл"),
        ([2], [1], [1.0], "source < target"),
        ([1], [9], [1.0], "нет среди узлов"),
        ([1], [2], [-1.0], "отрицательн"),
        ([1], [2], [np.nan], "пропуск"),
    ],
)
def test_adjacency_rejects_bad_edges(source, target, weight, match):
    edges = pd.DataFrame({"source": source, "target": target, "weight": weight})
    with pytest.raises(ValueError, match=match):
        icvi.adjacency(edges, [1, 2, 3])


# ---------------------------------------------------------------------------------------------
# SW и CH — сверка со sklearn


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_silhouette_and_ch_match_sklearn(seed):
    x, y = blobs(n_per=40, k=4, d=6, std=2.5, seed=seed)
    assert icvi.silhouette(x, y) == pytest.approx(silhouette_score(x, y), abs=1e-10)
    assert icvi.calinski_harabasz(x, y) == pytest.approx(calinski_harabasz_score(x, y), rel=1e-10)


def test_silhouette_singleton_cluster_matches_sklearn():
    """Rousseeuw, 1987, с. 55: у объекта единственного в кластере s(i) = 0."""
    x, y = blobs(n_per=20, k=3, d=2, seed=3)
    y = y.copy()
    y[0] = 7  # синглтон
    assert icvi.silhouette(x, y) == pytest.approx(silhouette_score(x, y), abs=1e-10)
    assert icvi.calinski_harabasz(x, y) == pytest.approx(calinski_harabasz_score(x, y), rel=1e-10)


# ---------------------------------------------------------------------------------------------
# S_Dbw — ручные примеры (Halkidi, Vazirgiannis, 2001)


def test_s_dbw_hand_example_separated():
    """Два кластера {0, 1, 2} и {10, 11, 12} на прямой.

    σ(X) = 154/6 = 77/3, ||σ(C_k)|| = 2/3 → Scat = (1/2)·2·(2/3)/(77/3) = 2/77;
    stdev = (1/2)·√(4/3) ≈ 0,577; в окрестности середины 6 нет точек → Dens_bw = 0; S_Dbw = 2/77.
    """
    x = np.array([0, 1, 2, 10, 11, 12], dtype=float)[:, None]
    y = np.array([0, 0, 0, 1, 1, 1])
    assert icvi.s_dbw(x, y) == pytest.approx(2 / 77, abs=1e-12)


def test_s_dbw_hand_example_touching():
    """{0, 1, 2} и {3, 4, 5}: σ(X) = 35/12 → Scat = (2/3)/(35/12) = 8/35; stdev ≈ 0,577.

    Середина 2,5: точки 2 и 3 (расстояние 0,5) → density(u) = 2; у центров 1 и 4 — по одной точке (сами 1 и 4)
    → отношение 2 для обеих упорядоченных пар; Dens_bw = (2 + 2)/(2·1) = 2; S_Dbw = 2 + 8/35 = 78/35.
    """
    x = np.arange(6, dtype=float)[:, None]
    y = np.array([0, 0, 0, 1, 1, 1])
    assert icvi.s_dbw(x, y) == pytest.approx(78 / 35, abs=1e-12)
    assert icvi.s_dbw(x, y, density="own") == pytest.approx(78 / 35, abs=1e-12)


def test_s_dbw_density_variants_differ_by_hand():
    """{0, 1, 2} и {1,3; 7; 9,7}: вариант плотности центра меняет ответ.

    Центры 1 и 6; ||σ(C_0)|| = 2/3, ||σ(C_1)|| = 36,78/3 = 12,26; stdev = (1/2)·√(2/3 + 12,26) ≈ 1,798.
    Центр 1: точки 0, 1, 2 своего кластера и 1,3 чужого → «pair» (C_0 ∪ C_1, буквально формула (15)) = 4,
    «own» (только свой кластер, как в Doklady, 2025, формула (15)) = 3. Центр 6: одна точка 7
    в обоих вариантах.
    Середина 3,5: одна точка 2. Dens_bw = 1/4 («pair») или 1/3 («own»).
    σ(X) = 76,28/6 → Scat = (1/2)·(2/3 + 12,26)/(76,28/6).
    """
    x = np.array([0, 1, 2, 1.3, 7, 9.7])[:, None]
    y = np.array([0, 0, 0, 1, 1, 1])
    scat = 0.5 * (2 / 3 + 36.78 / 3) / (76.28 / 6)
    assert icvi.s_dbw(x, y) == pytest.approx(scat + 1 / 4, abs=1e-10)
    assert icvi.s_dbw(x, y, density="pair") == pytest.approx(scat + 1 / 4, abs=1e-10)
    assert icvi.s_dbw(x, y, density="own") == pytest.approx(scat + 1 / 3, abs=1e-10)
    with pytest.raises(ValueError, match="density"):
        icvi.s_dbw(x, y, density="kim")


# ---------------------------------------------------------------------------------------------
# AVI и AVU — ручные примеры


def test_avi_avu_two_triangles():
    """Внутренние рёбра — упорядоченные пары: изолируемость = 6/(6 + 1) ≈ 0,857, а не 3/(3 + 1) = 0,75.

    Объединяемость пары: M_01/(out_0 + out_1 − M_01) = 1/(1 + 1 − 1) = 1; AVU = (1/K)·(U_01 + U_10) = 1.
    При K = 2 и связном графе AVU ≡ 1 по построению (всё внешнее у обоих — друг к другу).
    """
    a, y = two_triangles()
    assert icvi.avi(a, y) == pytest.approx(6 / 7, abs=1e-12)
    assert icvi.avi(a, y) != pytest.approx(0.75)
    assert icvi.avu(a, y) == pytest.approx(1.0, abs=1e-12)


def test_avi_weighted_two_triangles():
    """Веса — суммы весов (Howie и др., PVLDB, 2023, т. 16, № 11, с. 3175, формулы (36)–(37), p(u, v)).

    Рёбра треугольников весом 2, мост весом 1: изолируемость = 12/(12 + 1).
    """
    a, y = two_triangles(bridge=1.0, inner=2.0)
    assert icvi.avi(a, y) == pytest.approx(12 / 13, abs=1e-12)
    assert icvi.avu(a, y) == pytest.approx(1.0, abs=1e-12)


def test_avi_avu_chain_of_three_triangles():
    """Треугольники T0–T1–T2 цепочкой (мосты 2–3 и 5–6).

    out = (1, 2, 1); U_01 = 1/(1 + 2 − 1) = 1/2, U_12 = 1/2, U_02 = 0 → AVU = (1/3)·2·(1/2 + 1/2) = 2/3.
    Изолируемость: 6/7, 6/8, 6/7.
    """
    tri = [(0, 1), (0, 2), (1, 2)]
    edges = [(i + s, j + s, 1.0) for s in (0, 3, 6) for i, j in tri] + [(2, 3, 1.0), (5, 6, 1.0)]
    a = _from_edges(9, edges)
    y = np.repeat([0, 1, 2], 3)
    assert icvi.avi(a, y) == pytest.approx((6 / 7 + 6 / 8 + 6 / 7) / 3, abs=1e-12)
    assert icvi.avu(a, y) == pytest.approx(2 / 3, abs=1e-12)


def test_disconnected_components_as_clusters():
    """Кластеры — компоненты связности: AVI = 1; знаменатель AVU = 0 у каждой пары → объединяемость 0
    (соглашение модуля, с предупреждением): предел «смешивания нет» из Doklady, 2025, с. 560."""
    tri = [(0, 1), (0, 2), (1, 2)]
    a = _from_edges(6, [(i + s, j + s, 1.0) for s in (0, 3) for i, j in tri])
    y = np.array([0, 0, 0, 1, 1, 1])
    assert icvi.avi(a, y) == pytest.approx(1.0)
    with pytest.warns(icvi.IcviWarning, match="знаменатель"):
        assert icvi.avu(a, y) == 0.0


# ---------------------------------------------------------------------------------------------
# MQ — модульность: networkx против igraph


@pytest.mark.parametrize("weighted", [False, True])
def test_modularity_matches_networkx_and_igraph(weighted):
    a, y = planted([30, 25, 35, 20], 0.3, 0.05, seed=4, weighted=weighted)
    ours = icvi.modularity(a, y)
    comms = [set(np.flatnonzero(y == k).tolist()) for k in np.unique(y)]
    ref_nx = nx.community.modularity(_nx_graph(a), comms, weight="weight", resolution=1.0)
    ref_ig = _ig_graph(a).modularity(y.tolist(), weights="weight", resolution=1.0)
    assert ref_nx == pytest.approx(ref_ig, abs=1e-9)
    assert ours == pytest.approx(ref_nx, abs=1e-9)
    assert ours == pytest.approx(ref_ig, abs=1e-9)


def test_modularity_resolution_matches_networkx():
    a, y = planted([20, 20, 20], 0.4, 0.05, seed=5, weighted=True)
    comms = [set(np.flatnonzero(y == k).tolist()) for k in np.unique(y)]
    for gamma in (0.5, 2.0):
        ref = nx.community.modularity(_nx_graph(a), comms, weight="weight", resolution=gamma)
        assert icvi.modularity(a, y, resolution=gamma) == pytest.approx(ref, abs=1e-9)


# ---------------------------------------------------------------------------------------------
# Истинное разбиение против случайного и смешивание


def _baseline(labels, x=None, a=None, names=None, n=200, seed=0):
    df = icvi.random_baseline(labels, X=x, A=a, names=names, n_perm=n, rng=np.random.default_rng(seed))
    return df.set_index("name")


def test_blobs_truth_beats_random_on_feature_indices():
    x, y = blobs(n_per=50, k=4, d=5, std=1.5, seed=6)
    base = _baseline(y, x=x, names=["sw", "ch", "s_dbw"], n=100)
    for name in ("sw", "ch", "s_dbw"):
        value = icvi.compute(name, y, X=x)
        z = icvi.zscore(value, base.loc[name, "mean"], base.loc[name, "sd"], icvi.BETTER[name])
        assert z > 3, (name, value, base.loc[name].to_dict())


def test_planted_truth_beats_random_on_avi_and_mq():
    a, y = planted([40, 40, 40, 40], 0.25, 0.02, seed=7)
    base = _baseline(y, a=a, names=["avi", "avu", "mq"], n=200)
    for name in ("avi", "mq"):
        value = icvi.compute(name, y, A=a)
        z = icvi.zscore(value, base.loc[name, "mean"], base.loc[name, "sd"], icvi.BETTER[name])
        assert z > 3, (name, value, base.loc[name].to_dict())


def test_avu_does_not_separate_truth_from_random_in_homogeneous_sbm():
    """Свойство формулы, а не ошибка: U_kl не содержит внутренних рёбер, а в однородной SBM внешние рёбра
    распределены по парам кластеров так же, как при случайных метках, — у обоих AVU ≈ (K − 1)/(2K − 3)
    (для K = 4 это 0,6; ср. табл. 3 в Doklady, 2025: 0,67; 0,56; 0,55; 0,53; 0,52 при K = 3…15).
    Поэтому z-оценка AVU против случайного базиса не бывает большой положительной на таком графе."""
    a, y = planted([60, 60, 60, 60], 0.25, 0.05, seed=8)
    base = _baseline(y, a=a, names=["avu"], n=200)
    value = icvi.avu(a, y)
    assert value == pytest.approx(3 / 5, abs=0.03)
    assert base.loc["avu", "mean"] == pytest.approx(3 / 5, abs=0.01)
    assert icvi.zscore(value, base.loc["avu", "mean"], base.loc["avu", "sd"], "min") < 2


def test_oversplit_partition_is_worse_on_graph_indices_with_same_k():
    """Одинаковое K = 4: истина против «два блока слиты, третий расщеплён пополам».

    Половинки расщеплённого блока отдают почти все внешние рёбра друг другу → объединяемость пары ≈ 1:
    AVU растёт (хуже), AVI и MQ падают (хуже)."""
    a, y = planted([50, 50, 50, 50], 0.3, 0.02, seed=9)
    bad = y.copy()
    bad[bad == 1] = 0
    block2 = np.flatnonzero(y == 2)
    bad[block2[:25]] = 1
    for name in ("avi", "mq"):
        assert icvi.compute(name, y, A=a) > icvi.compute(name, bad, A=a) + 0.05, name
    assert icvi.avu(a, bad) > icvi.avu(a, y) + 0.05  # половинки: U ≈ 187/(262 + 262 − 187) ≈ 0,55


def test_feature_indices_worsen_with_mixing():
    values = {n: [] for n in ("sw", "ch", "s_dbw")}
    for std in (0.5, 1.5, 3.0, 6.0):
        x, y = blobs(n_per=60, k=4, d=4, std=std, seed=10)
        for name in values:
            values[name].append(icvi.compute(name, y, X=x))
    assert np.all(np.diff(values["sw"]) < 0), values["sw"]
    assert np.all(np.diff(values["ch"]) < 0), values["ch"]
    assert np.all(np.diff(values["s_dbw"]) > 0), values["s_dbw"]


def test_graph_indices_worsen_with_mixing():
    """p_out растёт при общей средней степени: AVI и MQ падают; AVU меняется слабо (см. тест выше)."""
    avi_v, mq_v = [], []
    for p_out in (0.005, 0.02, 0.05, 0.1):
        a, y = planted([60, 60, 60, 60], 0.3, p_out, seed=11)
        avi_v.append(icvi.avi(a, y))
        mq_v.append(icvi.modularity(a, y))
    assert np.all(np.diff(avi_v) < 0), avi_v
    assert np.all(np.diff(mq_v) < 0), mq_v


def test_random_labels_avi_is_about_one_over_k():
    """Случайный базис AVI (Doklady, 2025, с. 559: «убывает примерно как 1/K»): при счёте упорядоченных пар
    изолируемость случайного кластера ≈ его доля объёма = 1/K, а не 1/(2K − 1) (неупорядоченные рёбра)."""
    a, _ = planted([300], 0.05, 0.05, seed=12)  # граф Эрдёша — Реньи, 300 узлов
    for k in (3, 5, 8):
        y = np.arange(300) % k
        base = _baseline(y, a=a, names=["avi", "avu"], n=200, seed=k)
        assert base.loc["avi", "mean"] == pytest.approx(1 / k, abs=0.01)
        assert abs(base.loc["avi", "mean"] - 1 / (2 * k - 1)) > 0.04
        assert base.loc["avu", "mean"] == pytest.approx((k - 1) / (2 * k - 3), abs=0.02)


# ---------------------------------------------------------------------------------------------
# Инвариантность и края


def test_relabelling_does_not_change_any_index():
    x, y = blobs(n_per=30, k=3, d=3, std=2.0, seed=13)
    a, _ = planted([30, 30, 30], 0.3, 0.05, seed=13, weighted=True)
    renamed = np.array([7, 2, 40])[y]
    before = icvi.evaluate(y, X=x, A=a)
    after = icvi.evaluate(renamed, X=x, A=a)
    assert list(before) == ALL
    for name in ALL:
        assert after[name] == pytest.approx(before[name], abs=1e-12), name


def test_single_cluster_gives_nan_with_warning():
    x, _ = blobs(n_per=10, k=2, d=2, seed=14)
    a, _ = planted([10, 10], 0.5, 0.1, seed=14)
    y = np.zeros(20, dtype=int)
    with pytest.warns(icvi.IcviWarning, match="K = 1"):
        values = icvi.evaluate(y, X=x, A=a)
    assert all(math.isnan(v) for v in values.values())


def test_too_many_clusters_for_features_gives_nan():
    x = np.arange(6, dtype=float).reshape(3, 2)
    with pytest.warns(icvi.IcviWarning, match="K"):
        assert math.isnan(icvi.silhouette(x, np.array([0, 1, 2])))


def test_noise_label_is_dropped_from_features_and_graph():
    x, y = blobs(n_per=30, k=3, d=3, std=2.0, seed=15)
    a, _ = planted([30, 30, 30], 0.3, 0.05, seed=15)
    noisy = y.copy()
    noisy[::7] = icvi.NOISE
    keep = noisy != icvi.NOISE
    full = icvi.evaluate(noisy, X=x, A=a)
    reduced = icvi.evaluate(noisy[keep], X=x[keep], A=a[keep][:, keep])
    for name in ALL:
        assert full[name] == pytest.approx(reduced[name], abs=1e-12), name


def test_other_negative_labels_are_rejected():
    x, y = blobs(n_per=10, k=2, d=2, seed=16)
    y = y.copy()
    y[0] = -2
    with pytest.raises(ValueError, match="-2"):
        icvi.silhouette(x, y)


def test_length_mismatch_is_an_error():
    x, y = blobs(n_per=10, k=2, d=2, seed=16)
    with pytest.raises(ValueError, match="длин"):
        icvi.silhouette(x, y[:-1])
    a, _ = planted([10, 10], 0.5, 0.1)
    with pytest.raises(ValueError, match="длин"):
        icvi.avi(a, y[:-1])


def test_all_noise_gives_nan_with_warning():
    x, _ = blobs(n_per=10, k=2, d=2, seed=17)
    with pytest.warns(icvi.IcviWarning):
        assert math.isnan(icvi.silhouette(x, np.full(20, icvi.NOISE)))


def test_isolates_and_singletons_on_graph():
    """Изолят внутри кластера не ломает счёт; кластер-синглтон со степенью > 0 — изолируемость 0."""
    a, y = two_triangles()
    a = sp.block_diag([a, sp.csr_matrix((1, 1))]).tocsr()  # узел 6 — изолят
    y_iso = np.append(y, 1)
    assert icvi.avi(a, y_iso) == pytest.approx(6 / 7)
    assert icvi.modularity(a, y_iso) == pytest.approx(icvi.modularity(*two_triangles()), abs=1e-12)
    y_single = np.array([0, 0, 0, 1, 1, 2, 1])  # узел 5 — синглтон
    assert math.isfinite(icvi.avi(a, y_single))


def test_cluster_of_only_isolates_gives_nan_avi():
    a, y = two_triangles()
    a = sp.block_diag([a, sp.csr_matrix((2, 2))]).tocsr()
    y = np.append(y, [2, 2])
    with pytest.warns(icvi.IcviWarning, match="нет рёбер"):
        assert math.isnan(icvi.avi(a, y))


def test_graph_without_edges_gives_nan():
    a = sp.csr_matrix((6, 6))
    y = np.array([0, 0, 0, 1, 1, 1])
    for fn in (icvi.avi, icvi.avu, icvi.modularity):
        with pytest.warns(icvi.IcviWarning, match="без рёбер"):
            assert math.isnan(fn(a, y))


def test_graph_must_be_symmetric_without_loops():
    a, y = two_triangles()
    a = a.tolil()
    a[0, 5] = 1.0
    with pytest.raises(ValueError, match="симметр"):
        icvi.avi(a.tocsr(), y)
    b, _ = two_triangles()
    b = b.tolil()
    b[0, 0] = 1.0
    with pytest.raises(ValueError, match="петл"):
        icvi.modularity(b.tocsr(), y)


def test_compute_requires_its_space():
    x, y = blobs(n_per=10, k=2, d=2, seed=18)
    with pytest.raises(ValueError, match="A"):
        icvi.compute("avi", y, X=x)
    with pytest.raises(ValueError, match="X"):
        icvi.compute("sw", y)
    with pytest.raises(ValueError, match="dbi"):
        icvi.compute("dbi", y, X=x)


def test_evaluate_defaults_to_available_spaces():
    x, y = blobs(n_per=10, k=2, d=2, seed=19)
    assert list(icvi.evaluate(y, X=x)) == ["sw", "ch", "s_dbw"]
    a, _ = planted([10, 10], 0.5, 0.1)
    assert list(icvi.evaluate(y, A=a)) == ["avi", "avu", "mq"]
    assert list(icvi.evaluate(y, X=x, A=a, names=["mq", "sw"])) == ["mq", "sw"]


# ---------------------------------------------------------------------------------------------
# Случайный базис и z-оценка


def test_random_baseline_matches_single_metric_on_same_permutations():
    """Векторный счёт базиса = поштучный счёт тех же перестановок (одна и та же последовательность rng)."""
    x, y = blobs(n_per=20, k=4, d=3, std=3.0, seed=20)
    a, _ = planted([20, 20, 20, 20], 0.3, 0.08, seed=20, weighted=True)
    y = y.copy()
    y[[3, 17]] = icvi.NOISE
    values = icvi.baseline_values(y, X=x, A=a, names=ALL, n_perm=6, rng=np.random.default_rng(1))
    rng = np.random.default_rng(1)
    keep = y != icvi.NOISE
    for p in range(6):
        perm = y.copy()
        perm[keep] = rng.permutation(y[keep])
        for name in ALL:
            assert values[name][p] == pytest.approx(icvi.compute(name, perm, X=x, A=a), abs=1e-10), name


def test_random_baseline_table_and_determinism():
    x, y = blobs(n_per=20, k=3, d=3, seed=21)
    a, _ = planted([20, 20, 20], 0.3, 0.05, seed=21)
    one = icvi.random_baseline(y, X=x, A=a, n_perm=30, rng=np.random.default_rng(5))
    two = icvi.random_baseline(y, X=x, A=a, n_perm=30, rng=np.random.default_rng(5))
    assert list(one.columns) == ["name", "mean", "sd", "n_perm", "n_undefined"]
    assert one["name"].tolist() == ALL
    assert (one["n_perm"] + one["n_undefined"] == 30).all()
    pd.testing.assert_frame_equal(one, two)


def test_random_baseline_keeps_noise_in_place():
    """Шум в перестановках не участвует: базис с шумом = базис на данных без шумовых строк."""
    x, y = blobs(n_per=20, k=3, d=3, seed=22)
    a, _ = planted([20, 20, 20], 0.3, 0.05, seed=22)
    noisy = y.copy()
    noisy[::5] = icvi.NOISE
    keep = noisy != icvi.NOISE
    with_noise = icvi.random_baseline(noisy, X=x, A=a, n_perm=20, rng=np.random.default_rng(3))
    without = icvi.random_baseline(
        noisy[keep], X=x[keep], A=a[keep][:, keep], n_perm=20, rng=np.random.default_rng(3)
    )
    pd.testing.assert_frame_equal(with_noise, without)


def test_zscore_sign_follows_direction():
    assert icvi.zscore(5.0, 3.0, 2.0, "max") == pytest.approx(1.0)
    assert icvi.zscore(5.0, 3.0, 2.0, "min") == pytest.approx(-1.0)
    assert icvi.zscore(1.0, 3.0, 2.0, "min") == pytest.approx(1.0)
    assert math.isnan(icvi.zscore(1.0, 3.0, 0.0, "max"))
    with pytest.raises(ValueError, match="better"):
        icvi.zscore(1.0, 0.0, 1.0, "up")


def test_all_six_on_one_candidate_are_finite():
    x, y = blobs(n_per=25, k=4, d=4, std=2.0, seed=23)
    a, _ = planted([25, 25, 25, 25], 0.3, 0.05, seed=23)
    values = icvi.evaluate(y, X=x, A=a)
    assert list(values) == ALL
    assert all(math.isfinite(v) for v in values.values())


def test_config_s_dbw_variant_is_module_default():
    assert load_config()["icvi"]["s_dbw_density"] == icvi.S_DBW_DENSITY


def test_options_from_config():
    assert icvi.options(load_config()) == {"s_dbw_density": "pair"}
    assert icvi.options({"icvi": {"s_dbw_density": "own"}}) == {"s_dbw_density": "own"}
    assert icvi.options({"icvi": {}}) == {"s_dbw_density": icvi.S_DBW_DENSITY}
    with pytest.raises(ValueError, match="s_dbw_density"):
        icvi.options({"icvi": {"s_dbw_density": "kim"}})
    # результат передаётся как **options(cfg)
    x = np.array([0, 1, 2, 1.3, 7, 9.7])[:, None]
    y = np.array([0, 0, 0, 1, 1, 1])
    own = icvi.options({"icvi": {"s_dbw_density": "own"}})
    assert icvi.compute("s_dbw", y, X=x, **own) == pytest.approx(icvi.s_dbw(x, y, density="own"))
    assert icvi.evaluate(y, X=x, **own)["s_dbw"] == pytest.approx(icvi.s_dbw(x, y, density="own"))


# ---------------------------------------------------------------------------------------------
# Вырожденный базис AVU при K = 2 и K = 3


def _random_weighted_graph(n, p, seed):
    rng = np.random.default_rng(seed)
    upper = np.triu(rng.random((n, n)) < p, k=1) * rng.uniform(0.1, 5.0, size=(n, n))
    return sp.csr_matrix(upper + upper.T)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_avu_is_two_thirds_at_k3_and_zscore_is_nan(seed):
    """K = 3: out_1 + out_2 − M_12 = M_12 + M_13 + M_23 у каждой пары → Σ_{k≠l} U_kl = 2, AVU ≡ 2/3
    при любых весах и размерах; базис вырожден (SD ~1e-16) → z = NaN, а не шум округления."""
    rng = np.random.default_rng(seed)
    a = _random_weighted_graph(60, 0.15, seed)
    y = rng.choice(3, size=60, p=[0.5, 0.3, 0.2])
    assert icvi.avu(a, y) == pytest.approx(2 / 3, abs=1e-12)
    base = _baseline(y, a=a, names=["avu"], n=50, seed=seed)
    assert base.loc["avu", "mean"] == pytest.approx(2 / 3, abs=1e-12)
    assert base.loc["avu", "sd"] < 1e-12
    assert math.isnan(icvi.zscore(icvi.avu(a, y), base.loc["avu", "mean"], base.loc["avu", "sd"], "min"))


@pytest.mark.parametrize("seed", [0, 1])
def test_avu_is_one_at_k2_on_connected_graph_and_zscore_is_nan(seed):
    rng = np.random.default_rng(seed)
    a = _random_weighted_graph(50, 0.2, seed)
    y = rng.choice(2, size=50, p=[0.7, 0.3])
    assert icvi.avu(a, y) == pytest.approx(1.0, abs=1e-12)
    base = _baseline(y, a=a, names=["avu"], n=50, seed=seed)
    assert math.isnan(icvi.zscore(icvi.avu(a, y), base.loc["avu", "mean"], base.loc["avu", "sd"], "min"))


def test_zscore_relative_tolerance_for_degenerate_baseline():
    assert math.isnan(icvi.zscore(2 / 3, 2 / 3, 1e-16, "min"))
    assert math.isnan(icvi.zscore(1000.0, 1000.0, 1e-10, "max"))  # порог относительный: 1e-12 · 1000
    assert icvi.zscore(1000.0 + 1e-6, 1000.0, 1e-8, "max") == pytest.approx(100.0)
    assert icvi.zscore(0.5, 0.4, 1e-3, "max") == pytest.approx(100.0)


def test_random_baseline_counts_undefined_permutations():
    """S_Dbw даёт NaN, если у середины есть точки, а у центров пары — нет: такие перестановки отброшены
    и посчитаны в n_undefined (n_perm + n_undefined = запрошенное число)."""
    x = np.random.default_rng(2).standard_t(2, size=(60, 3))  # тяжёлые хвосты: центры в пустых областях
    y = np.arange(60) % 6
    base = icvi.random_baseline(y, X=x, names=["s_dbw"], n_perm=40, rng=np.random.default_rng(0))
    row = base.set_index("name").loc["s_dbw"]
    values = icvi.baseline_values(y, X=x, names=["s_dbw"], n_perm=40, rng=np.random.default_rng(0))["s_dbw"]
    assert row["n_undefined"] == int(np.isnan(values).sum()) > 0
    assert row["n_perm"] + row["n_undefined"] == 40
    assert row["mean"] == pytest.approx(np.nanmean(values))


def test_random_baseline_sd_is_sample_sd_ddof1():
    """SD базиса — выборочное (ddof = 1) по тем же перестановкам; при n_perm = 4 отличие от ddof = 0 — 15%."""
    x, y = blobs(n_per=20, k=3, d=3, std=3.0, seed=25)
    a, _ = planted([20, 20, 20], 0.3, 0.08, seed=25, weighted=True)
    names = ["sw", "ch", "s_dbw", "avi", "mq"]
    base = icvi.random_baseline(y, X=x, A=a, names=names, n_perm=4, rng=np.random.default_rng(7))
    values = icvi.baseline_values(y, X=x, A=a, names=names, n_perm=4, rng=np.random.default_rng(7))
    base = base.set_index("name")
    for name in names:
        v = values[name]
        assert base.loc[name, "mean"] == pytest.approx(np.mean(v), abs=1e-12), name
        assert base.loc[name, "sd"] == pytest.approx(np.std(v, ddof=1), rel=1e-12), name
        assert base.loc[name, "sd"] != pytest.approx(np.std(v, ddof=0), rel=1e-3), name


@pytest.mark.parametrize(
    "labels", [np.array(["a", "b"] * 10), np.array([0, 1] * 10, dtype=object), np.array([True, False] * 10)]
)
def test_non_integer_label_types_are_rejected_clearly(labels):
    x, _ = blobs(n_per=10, k=2, d=2, seed=26)
    with pytest.raises(ValueError, match="метки должны быть целыми числами"):
        icvi.silhouette(x, labels)


def test_float_labels_must_be_whole_numbers():
    x, y = blobs(n_per=10, k=2, d=2, seed=27)
    assert icvi.silhouette(x, y.astype(float)) == pytest.approx(icvi.silhouette(x, y))
    for bad in (y + 0.5, np.where(y == 0, np.nan, 1.0)):
        with pytest.raises(ValueError, match="метки должны быть целыми числами"):
            icvi.silhouette(x, bad)
