"""Правила рёбер, нулевые модели и разрежение на синтетике с известным ответом."""

import numpy as np
import pandas as pd
import pytest

from munnet.network import geo, graph
from munnet.network import rules as R


def blocks(n_per: int = 15, t: int = 24, rho: float = 0.8, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Два блока рядов: внутри блока общий фактор (корреляция пар ≈ rho), между блоками — независимы."""
    rng = np.random.default_rng(seed)
    f = rng.normal(size=(2, t))
    rows, labels = [], []
    for b in range(2):
        for _ in range(n_per):
            rows.append(np.sqrt(rho) * f[b] + np.sqrt(1 - rho) * rng.normal(size=t))
            labels.append(b)
    return np.asarray(rows), np.asarray(labels)


# --- Формулы против прямого расчёта ---------------------------------------------------------------


def test_corr_matches_numpy_and_has_no_loops():
    X, _ = blocks()
    S = R.corr_matrix(X)
    ref = np.corrcoef(X)
    np.fill_diagonal(ref, np.nan)
    assert S.dtype == np.float32
    assert np.allclose(S, ref, atol=1e-6, equal_nan=True)
    assert np.isnan(np.diag(S)).all()
    assert np.allclose(S, S.T, equal_nan=True)


def test_spearman_matches_scipy():
    from scipy.stats import spearmanr

    X, _ = blocks(n_per=4)
    S = R.spearman_matrix(X)
    assert S[0, 5] == pytest.approx(spearmanr(X[0], X[5])[0], abs=1e-6)


def test_cosine_and_euclid_match_direct():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(6, 6))
    C = R.cosine_matrix(X)
    D = R.euclid_matrix(X)
    a, b = X[1], X[4]
    assert C[1, 4] == pytest.approx(a @ b / np.linalg.norm(a) / np.linalg.norm(b), abs=1e-6)
    assert D[1, 4] == pytest.approx(np.linalg.norm(a - b), abs=1e-5)
    K = R.gaussian_similarity(D, 2.0)
    assert K[1, 4] == pytest.approx(np.exp(-((np.linalg.norm(a - b) / 2.0) ** 2)), abs=1e-6)
    zero = np.vstack([X, np.zeros(6)])
    assert np.isnan(R.cosine_matrix(zero)[-1]).all()  # у нулевого вектора направления нет


def test_nan_input_is_rejected():
    X = np.ones((3, 5))
    X[1, 2] = np.nan
    with pytest.raises(ValueError, match="пропуски"):
        R.corr_matrix(X)


def test_corr_multi_is_mean_of_category_correlations():
    X, _ = blocks(n_per=5, seed=1)
    Y, _ = blocks(n_per=5, seed=2)
    S = R.corr_multi([X, Y])
    ref = (R.corr_matrix(X) + R.corr_matrix(Y)) / 2
    assert np.allclose(S, ref, atol=1e-5, equal_nan=True)
    assert np.allclose(R.corr_multi([X]), R.corr_matrix(X), atol=1e-6, equal_nan=True)
    L, _ = R.lagged_corr([X, Y], 0)
    assert np.allclose(L, ref, atol=1e-5, equal_nan=True)


# --- Заложенная структура восстанавливается --------------------------------------------------------


def test_correlation_knn_finds_blocks_and_probe_recovers_them():
    from sklearn.metrics import adjusted_rand_score

    X, labels = blocks()
    S = R.corr_matrix(X)
    edges = graph.knn_edges(S, 4)
    assert (labels[edges["i"]] == labels[edges["j"]]).all()
    res = graph.probe(len(X), edges, seeds=[0, 1, 2], resolution=1.0, null_repeats=5, factor=10)
    # каждое сообщество — из одного блока (Leiden может дробить блок, но не смешивает блоки)
    for c in np.unique(res.membership):
        assert len(set(labels[res.membership == c])) == 1
    dense = graph.probe(
        len(X), graph.knn_edges(S, 10), seeds=[0, 1, 2], resolution=1.0, null_repeats=5, factor=10
    )
    assert adjusted_rand_score(labels, dense.membership) == 1.0
    assert res.q_z > 2  # модульность выше, чем у сети с теми же степенями и перемешанными рёбрами


def test_lag_is_recovered_with_sign():
    rng = np.random.default_rng(3)
    base = rng.normal(size=40)
    x = base[2:26]  # x(t) = base(t + 2)
    y = base[0:24]  # y(t) = base(t) = x(t − 2): y повторяет x через 2 месяца
    S, lag = R.lagged_corr(np.vstack([x, y]), 3)
    assert S[0, 1] == pytest.approx(1.0, abs=1e-6)
    assert lag[0, 1] == 2 and lag[1, 0] == -2  # x опережает y
    assert S[0, 1] == pytest.approx(S[1, 0])


def test_lag_zero_wins_ties_and_multi_category_shares_the_lag():
    rng = np.random.default_rng(4)
    a = rng.normal(size=(3, 24))
    S, lag = R.lagged_corr([a, a[::-1].copy()], 2)
    assert np.allclose(np.diag(lag), 0)
    S1, _ = R.lagged_corr(a, 2)
    assert S1.shape == S.shape == (3, 3)


def test_dtw_tolerates_shift_within_window_only():
    t = np.arange(24, dtype=float)
    peak = lambda c: np.exp(-((t - c) ** 2) / 2.0)  # noqa: E731
    X = np.vstack([peak(10), peak(11), peak(14)])
    D = R.dtw_matrix(X, window=2)
    assert D[0, 1] < 0.5 * D[0, 2]  # сдвиг на 1 месяц внутри окна почти не штрафуется
    euclid = np.linalg.norm(R.zscore_rows(X)[0] - R.zscore_rows(X)[1])
    assert D[0, 1] < euclid
    D0 = R.dtw_matrix(X, window=0)  # без окна сдвигов — евклидово расстояние z-рядов
    assert D0[0, 1] == pytest.approx(euclid, rel=1e-6)


def test_dtw_cross_block_matches_full_matrix():
    X, _ = blocks(n_per=4)
    full = R.dtw_matrix(X, window=2)
    cross = R.dtw_matrix(X, window=2, Y=X)
    off = ~np.eye(len(X), dtype=bool)
    assert np.allclose(full[off], cross[off], atol=1e-5)


# --- Нулевая модель и поправка ----------------------------------------------------------------------


def test_allowed_shifts_avoid_season_alignment():
    s = R.allowed_shifts(24, reach=3, guard=2)
    m = s % 12
    assert (np.minimum(m, 12 - m) >= 5).all()
    assert 12 not in s and 1 not in s
    assert list(R.allowed_shifts(12, 0, 2)) == list(range(2, 11))


def test_shift_rows_is_circular():
    X = np.arange(10).reshape(2, 5)
    Y = R.shift_rows(X, np.array([1, 2]))
    assert list(Y[0]) == [4, 0, 1, 2, 3]
    assert list(Y[1]) == [8, 9, 5, 6, 7]


def test_bh_matches_manual():
    p = np.array([0.01, 0.04, 0.03, 0.2])
    # отсортированные: 0,01·4/1 = 0,04; 0,03·4/2 = 0,06; 0,04·4/3 = 0,0533 -> монотонно 0,0533; 0,2
    q = R.bh_qvalues(p)
    assert np.allclose(q, [0.04, 0.0533333, 0.0533333, 0.2], atol=1e-6)


def test_null_gives_no_structure_on_noise_and_finds_blocks():
    rng = np.random.default_rng(5)
    noise = rng.normal(size=(40, 24))
    shifts = R.allowed_shifts(24, 0, 2)
    null = R.null_similarities(lambda a, b: R.corr_matrix(a, b), noise, shifts, 5, rng)
    S = R.corr_matrix(noise)
    q = R.bh_qvalues(R.empirical_p(R.upper(S), null))
    assert (q < 0.05).sum() <= 2  # на шуме значимых пар почти нет
    X, labels = blocks(n_per=20)
    null_b = R.null_similarities(lambda a, b: R.corr_matrix(a, b), X, shifts, 5, rng)
    Sb = R.corr_matrix(X)
    qb = R.bh_qvalues(R.empirical_p(R.upper(Sb), null_b))
    iu = np.triu_indices(len(X), 1)
    same = labels[iu[0]] == labels[iu[1]]
    assert (qb[same] < 0.05).mean() > 0.9
    assert (qb[~same] < 0.05).mean() < 0.05


def test_empirical_p_never_zero():
    null = np.sort(np.array([0.1, 0.2, 0.3]))
    p = R.empirical_p(np.array([0.5, 0.2, np.nan]), null)
    assert p[0] == pytest.approx(0.25) and p[1] == pytest.approx(0.75) and np.isnan(p[2])


# --- Разрежение ----------------------------------------------------------------------------------


def test_knn_union_and_mutual():
    S = np.array(
        [
            [np.nan, 0.9, 0.1, 0.0],
            [0.9, np.nan, 0.8, 0.2],
            [0.1, 0.8, np.nan, 0.7],
            [0.0, 0.2, 0.7, np.nan],
        ]
    )
    union = graph.knn_edges(S, 1)
    mutual = graph.knn_edges(S, 1, mutual=True)
    # списки: 0->1, 1->0, 2->1, 3->2
    assert set(map(tuple, union[["i", "j"]].to_numpy())) == {(0, 1), (1, 2), (2, 3)}
    assert set(map(tuple, mutual[["i", "j"]].to_numpy())) == {(0, 1)}
    assert (union["i"] < union["j"]).all()
    assert union.loc[union["i"] == 1, "weight"].item() == pytest.approx(0.8)
    lists = graph.knn_lists(S, 1)
    assert list(graph.kocc(lists, 4)) == [1, 2, 1, 0]


def test_knn_skips_missing_similarities():
    S = np.array([[np.nan, np.nan, 0.5], [np.nan, np.nan, np.nan], [0.5, np.nan, np.nan]])
    lists = graph.knn_lists(S, 2)
    assert list(lists[1]) == [-1, -1]
    edges = graph.knn_edges(S, 2)
    assert set(map(tuple, edges[["i", "j"]].to_numpy())) == {(0, 2)}


def test_top_pairs_mask_density():
    rng = np.random.default_rng(6)
    X = rng.normal(size=(20, 10))
    S = R.corr_matrix(X)
    edges = graph.threshold_edges(S, graph.top_pairs_mask(S, 15))
    assert len(edges) == 15


def test_passport_on_two_triangles():
    edges = pd.DataFrame({"i": [0, 0, 1, 3, 3, 4, 2], "j": [1, 2, 2, 4, 5, 5, 3], "weight": 1.0})
    info = graph.NodeInfo(
        ids=np.arange(6),
        region=np.array([1, 1, 1, 2, 2, 2]),
        lat=np.full(6, 55.0),
        lon=np.linspace(37, 38, 6),
        attrs=pd.DataFrame({"a": [1.0, 1, 1, 5, 5, 5]}),
    )
    p = graph.passport(edges, info, attr_columns=["a"])
    assert p["n_edges"] == 7 and p["n_components"] == 1 and p["isolates"] == 0
    assert p["within_region"] == pytest.approx(6 / 7)
    assert p["assort_a"] > 0.5 and p["assort_region"] > 0.5
    assert p["deg_max"] == 3


def test_mantel_identical_and_independent():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(25, 12))
    A = R.corr_matrix(X)
    rho, p = graph.mantel_ranks(A, A, 19, rng)
    assert rho == pytest.approx(1.0) and p == pytest.approx(1 / 20)
    B = R.corr_matrix(rng.normal(size=(25, 12)))
    rho2, p2 = graph.mantel_ranks(A, B, 19, rng)
    assert abs(rho2) < 0.2 and p2 > 0.05


# --- Дороги ---------------------------------------------------------------------------------------


def conn_frame() -> pd.DataFrame:
    rows = [
        (1, 2, 10.0, "highway"),
        (3, 1, 30.0, "highway"),
        (3, 2, 20.0, "highway"),
        (4, 1, 40.0, "highway"),
        (4, 2, 50.0, "highway"),
        (4, 3, 60.0, "highway"),
        (1, 2, 7.0, "railway"),
        (2, 1, 7.0, "railway"),  # железная дорога — в обе стороны
    ]
    return pd.DataFrame(rows, columns=["territory_id_x", "territory_id_y", "distance", "type"])


def test_railway_pair_both_directions_is_one_edge():
    rail = geo.road_pairs(conn_frame(), "railway")
    assert len(rail) == 1 and rail["distance"].item() == 7.0
    road = geo.road_pairs(conn_frame(), "highway")
    assert len(road) == 6 and (road["a"] < road["b"]).all()


def test_city_distance_population_mean_and_min():
    pairs = geo.road_pairs(conn_frame(), "highway")
    # узел 90 = МО 3 и 4 (город), узлы 1 и 2 — сами себе
    members = pd.DataFrame({"territory_id": [1, 2, 3, 4], "node_id": [1, 2, 90, 90]})
    weights = pd.Series({1: 1.0, 2: 1.0, 3: 1.0, 4: 3.0})
    D = geo.node_distance_matrix(pairs, np.array([1, 2, 90]), members, weights, "pop_mean")
    assert D[0, 1] == 10.0
    assert D[0, 2] == pytest.approx((1 * 30 + 3 * 40) / 4)  # 37,5: среднее с весами населения
    assert D[1, 2] == pytest.approx((1 * 20 + 3 * 50) / 4)
    assert np.allclose(D, D.T, equal_nan=True) and np.isnan(np.diag(D)).all()
    Dm = geo.node_distance_matrix(pairs, np.array([1, 2, 90]), members, weights, "min")
    assert Dm[0, 2] == 30.0  # пара 3–4 внутри города исчезла, петли нет


def test_gravity_formula():
    D = np.array([[np.nan, 10.0], [10.0, np.nan]])
    S = geo.gravity_log(D, np.array([100.0, 1000.0]), beta=2.0, min_km=1.0)
    assert S[0, 1] == pytest.approx(np.log(100) + np.log(1000) - 2 * np.log(10), rel=1e-6)
    D0 = np.array([[np.nan, 0.0], [0.0, np.nan]])
    assert np.isfinite(geo.gravity_log(D0, np.array([1.0, 1.0]), 2.0, 1.0)[0, 1])


# --- Корзина: блоки находятся, на шуме надёжность на уровне случая ---------------------------------


def basket_blocks(n_per: int = 20, seed: int = 0, signal: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Векторы корзины относительно региона (сумма 0): два типа с противоположным отличием от региона."""
    rng = np.random.default_rng(seed)
    direction = np.array([1.0, -1.0, 0.5, -0.5, 0.0, 0.0])
    rows, labels = [], []
    for b, sign in enumerate((1.0, -1.0)):
        for _ in range(n_per):
            v = sign * signal * direction + rng.normal(0, 0.3, 6)
            rows.append(v - v.mean())
            labels.append(b)
    return np.asarray(rows), np.asarray(labels)


@pytest.mark.parametrize("rule", ["cosine", "distance"])
def test_basket_rules_find_types(rule):
    X, labels = basket_blocks()
    if rule == "cosine":
        S = R.cosine_matrix(X)
    else:
        D = R.euclid_matrix(X)
        S = R.gaussian_similarity(D, R.median_offdiag(D))
    edges = graph.knn_edges(S, 5)
    assert (labels[edges["i"]] == labels[edges["j"]]).mean() > 0.95


def test_reliability_is_at_chance_on_noise_and_high_with_structure():
    rng = np.random.default_rng(8)
    n = 60
    noise_a, noise_b = rng.normal(size=(n, 6)), rng.normal(size=(n, 6))
    j_noise = graph.jaccard(
        graph.edge_set(graph.knn_edges(R.cosine_matrix(noise_a), 5)),
        graph.edge_set(graph.knn_edges(R.cosine_matrix(noise_b), 5)),
    )
    base = rng.normal(size=(n, 6))
    y1, y2 = base + rng.normal(0, 0.1, (n, 6)), base + rng.normal(0, 0.1, (n, 6))
    j_struct = graph.jaccard(
        graph.edge_set(graph.knn_edges(R.cosine_matrix(y1), 5)),
        graph.edge_set(graph.knn_edges(R.cosine_matrix(y2), 5)),
    )
    assert j_noise < 0.2 < 0.5 < j_struct
