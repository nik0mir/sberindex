"""Методы этапа cluster на синтетике с известной истиной: KEFRiN по статье, единый интерфейс методов,
подбор параметра под K, устойчивость, внешняя проверка, генератор синтетики."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import kruskal
from sklearn.metrics import adjusted_rand_score

from munnet.clustering import kefrin as KF
from munnet.clustering import methods as M
from munnet.clustering import stability as ST
from munnet.clustering import synthetic as SY
from munnet.clustering import validation as V
from munnet.clustering.params import RANDOM_METHODS

IMPL = {
    "kmeans_n_init": 5,
    "shalileh_mirkin": {"rho": 1.0, "xi": 1.0, "max_iter": 100},
    "leiden_grid": {"gamma_min": 0.01, "gamma_max": 5.0, "points": 15, "refine": 10},
    "hdbscan_grid": {"min_size": 5, "max_size": 80, "points": 12},
}


@pytest.fixture(scope="module")
def strong():
    """Сильный сигнал и в графе, и в признаках: 4 группы по 60 узлов."""
    return SY.make_knn(240, 4, 4, 3, 6, graph_signal=3.0, feature_signal=6.0, knn_k=10, seed=1)


def test_modularity_standardization_rows_sum_to_zero():
    rng = np.random.default_rng(0)
    P = rng.random((7, 7))
    P = P + P.T
    np.fill_diagonal(P, 0)
    Z = KF.modularity_standardize(P)
    assert np.allclose(Z.sum(axis=1), 0) and np.allclose(Z.sum(axis=0), 0)


def test_kefrin_criterion_matches_formula_5_matrix_form(strong):
    inp, truth = strong
    P = KF.modularity_standardize(inp.A)
    res = KF.kefrin(inp.X, P, 4, first=0, rho=1.0, xi=2.0)
    # формула 5 в матричной форме: F = ρ Tr[(Y − SC)ᵀ(Y − SC)] + ξ Tr[(P − SΛ)ᵀ(P − SΛ)], центры — средние (7)
    S = np.eye(4)[res.labels]
    C = np.linalg.pinv(S) @ inp.X
    L = np.linalg.pinv(S) @ P
    F = 1.0 * np.trace((inp.X - S @ C).T @ (inp.X - S @ C)) + 2.0 * np.trace((P - S @ L).T @ (P - S @ L))
    assert res.criterion == pytest.approx(F, rel=1e-9)
    assert res.converged


def test_kefrin_minimum_distance_is_fixed_point(strong):
    inp, _ = strong
    P = KF.modularity_standardize(inp.A)
    res = KF.kefrin(inp.X, P, 4, first=3)
    C = np.vstack([inp.X[res.labels == k].mean(axis=0) for k in range(4)])
    L = np.vstack([P[res.labels == k].mean(axis=0) for k in range(4)])
    d = KF.combined_distances(inp.X, P, C, L, 1.0, 1.0)
    assert np.array_equal(np.argmin(d, axis=1), res.labels)


def test_sparse_modularity_links_equal_dense_standardization(strong):
    inp, _ = strong
    dense = KF.kefrin(inp.X, KF.modularity_standardize(inp.A), 4, first=7)
    fast = KF.kefrin(inp.X, inp.A, 4, first=7)
    assert np.array_equal(dense.labels, fast.labels)
    assert fast.criterion == pytest.approx(dense.criterion, rel=1e-9)
    links = KF.ModularityLinks(inp.A)
    P = KF.modularity_standardize(inp.A)
    assert np.allclose(links.norms2(), (P**2).sum(axis=1))
    L = links.centers(dense.labels, 4, links.seeds([0, 1, 2, 3]))
    Ld = np.vstack([P[dense.labels == k].mean(axis=0) for k in range(4)])
    assert np.allclose(links.dots(L), P @ Ld.T) and np.allclose(links.center_norms2(L), (Ld**2).sum(axis=1))


def test_maxmin_seeds_are_distinct_and_start_with_first(strong):
    inp, _ = strong
    P = KF.modularity_standardize(inp.A)
    s = KF.maxmin_seeds(inp.X, P, 4, first=5, rho=1.0, xi=1.0)
    assert s[0] == 5 and len(set(s.tolist())) == 4


def test_kefrin_recovers_planted_partition(strong):
    inp, truth = strong
    pf = M.fit_protocol("shalileh_mirkin", inp, 4, None, (0, 1, 2), IMPL, RANDOM_METHODS)
    assert adjusted_rand_score(truth, pf.labels) > 0.9


def test_kefrin_uses_graph_when_features_are_noise():
    inp, truth = SY.make_knn(240, 4, 6, 3, 6, graph_signal=3.0, feature_signal=0.0, knn_k=10, seed=2)
    km = M.fit_protocol("kmeans", inp, 4, None, (0, 1), IMPL, RANDOM_METHODS)
    sm = M.fit_protocol("shalileh_mirkin", inp, 4, None, (0, 1, 2), IMPL, RANDOM_METHODS)
    assert adjusted_rand_score(truth, km.labels) < 0.2
    assert adjusted_rand_score(truth, sm.labels) > 0.6


@pytest.mark.parametrize(
    "method", ["kmeans", "ward", "gmm", "spectral", "shalileh_mirkin", "hybrid", "kmeans_joint"]
)
def test_methods_with_given_k_return_k_clusters_and_recover_truth(strong, method):
    inp, truth = strong
    pf = M.fit_protocol(method, inp, 4, None, (0, 1), IMPL, RANDOM_METHODS)
    assert M.n_clusters(pf.labels) == 4
    assert adjusted_rand_score(truth, pf.labels) > 0.8
    # метки канонические: 0 — самый крупный кластер
    sizes = np.bincount(pf.labels)
    assert (np.diff(sizes) <= 0).all()


@pytest.mark.parametrize("method", ["leiden", "louvain"])
def test_resolution_is_tuned_only_to_reach_k(strong, method):
    inp, truth = strong
    chosen, scan = M.tune_param(method, inp, [3, 4, 6], (0, 1), IMPL, RANDOM_METHODS)
    assert 4 in chosen
    pf = M.fit_protocol(method, inp, 4, chosen[4], (0, 1), IMPL, RANDOM_METHODS)
    assert M.n_clusters(pf.labels) == 4
    assert adjusted_rand_score(truth, pf.labels) > 0.8
    for k, g in chosen.items():
        assert scan.points[g] == k


def test_hdbscan_param_and_noise(strong):
    inp, _ = strong
    chosen, scan = M.tune_param("hdbscan", inp, [4], (0,), IMPL, RANDOM_METHODS)
    assert chosen and all(scan.points[v] == k for k, v in chosen.items())


def test_fit_is_reproducible(strong):
    inp, _ = strong
    for m in ("kmeans", "leiden", "louvain", "shalileh_mirkin", "hybrid"):
        param = 1.0 if m in ("leiden", "louvain") else None
        a = M.fit(m, inp, 4, param, 7, IMPL)
        b = M.fit(m, inp, 4, param, 7, IMPL)
        assert np.array_equal(a.labels, b.labels), m


def test_canonical_and_block_join():
    assert M.canonical(np.array([5, 5, 2, 2, 2, -1, 9])).tolist() == [1, 1, 0, 0, 0, -1, 2]
    rng = np.random.default_rng(0)
    Z = M.block_join([rng.normal(size=(50, 2)) * 10, rng.normal(size=(50, 5))], [0.5, 0.5])
    assert Z[:, :2].var(axis=0).sum() == pytest.approx(0.5)
    assert Z[:, 2:].var(axis=0).sum() == pytest.approx(0.5)


def test_best_match_jaccard():
    full = np.array([0, 0, 0, 1, 1, 1])
    assert ST.best_match_jaccard(full, np.array([3, 3, 3, 4, 4, 4])).tolist() == [1.0, 1.0]
    j = ST.best_match_jaccard(full, np.array([3, 3, 4, 4, 4, 4]))
    assert j[0] == pytest.approx(2 / 3) and j[1] == pytest.approx(3 / 4)
    assert np.isnan(ST.best_match_jaccard(np.array([0, 0, 2]), np.array([1, 1, 1]))[1])


def test_subsample_deterministic_and_size():
    a, b = ST.subsample(100, 0.8, 42, 3), ST.subsample(100, 0.8, 42, 3)
    assert np.array_equal(a, b) and len(a) == 80 and not np.array_equal(a, ST.subsample(100, 0.8, 42, 4))


def test_tree_kappa_high_when_type_is_a_threshold():
    rng = np.random.default_rng(0)
    Xm = rng.normal(size=(300, 3))
    labels = (Xm[:, 0] > 0).astype(int) + (Xm[:, 1] > 0.5).astype(int) * 2
    assert ST.tree_kappa(Xm, labels, 3, 5, 0) > 0.9
    assert ST.tree_kappa(Xm, rng.integers(0, 3, 300), 3, 5, 0) < 0.2


def test_epsilon2_matches_kruskal():
    rng = np.random.default_rng(1)
    y = rng.normal(size=90)
    lab = rng.integers(0, 3, 90)
    y[lab == 2] += 1.0
    H = kruskal(*[y[lab == g] for g in range(3)]).statistic
    assert V.epsilon2(y, lab) == pytest.approx(H / (len(y) - 1), rel=1e-9)


def test_type_mean_ranks_equal_rankdata():
    from scipy.stats import rankdata

    rng = np.random.default_rng(2)
    lab = rng.integers(0, 5, 200)
    ax = rng.normal(size=200)
    ax[lab == 4] = 0.0
    ax[lab == 3] = 0.0  # два типа с равным средним — связанные ранги
    assert np.allclose(V.type_mean_ranks(ax, lab), rankdata(V.type_mean(ax, lab)))


def test_permute_within_keeps_group_composition():
    rng = np.random.default_rng(0)
    groups = np.repeat([1, 2, 3], 10)
    lab = rng.integers(0, 4, 30)
    p = V.permute_within(lab, groups, rng)
    for g in (1, 2, 3):
        assert sorted(p[groups == g]) == sorted(lab[groups == g])


def test_validation_detects_signal_and_sign():
    rng = np.random.default_rng(3)
    n = 300
    groups = np.repeat(np.arange(10), 30)
    lab = rng.integers(0, 3, n)
    axis = lab + rng.normal(scale=0.3, size=n)
    y = lab * 0.8 + rng.normal(size=n)
    X = rng.normal(size=(n, 4))
    ind = pd.DataFrame({"good": y, "noise": rng.normal(size=n)})
    axes = pd.DataFrame({"ax": axis})
    cfg = {
        "permutations": 199,
        "alpha": 0.05,
        "expected": {"good": {"ax": 1}, "noise": {"ax": 1}},
        "beyond_attributes": {"folds": 5, "repeats": 3},
        "wage_link_max": 0.5,
    }
    out = V.validate(lab, ind, axes, X, rng.normal(size=n), groups, cfg, seed=0, beyond_permutations=99)
    good = out.set_index(["indicator", "check"])
    assert good.loc[("good", "difference"), "significant"]
    assert good.loc[("good", "sign"), "sign_ok"] and good.loc[("good", "sign"), "significant"]
    assert good.loc[("good", "beyond_attributes"), "stat"] > 0.05
    assert good.loc[("good", "beyond_attributes"), "significant"]
    assert not good.loc[("noise", "difference"), "significant"]


def test_beyond_attributes_fast_equals_direct_refit():
    rng = np.random.default_rng(5)
    n = 120
    X = rng.normal(size=(n, 3))
    lab = rng.integers(0, 3, n)
    y = X[:, 0] + lab + rng.normal(size=n)
    model = V.BeyondAttributes(y, X, 4, 2, seed=1)
    D = V._design(lab)
    Y = model.Y
    direct = []
    for r in range(2):
        sse = sst = 0.0
        for part in model.parts[r * 4 : (r + 1) * 4]:
            tr, te = part["tr"], part["te"]
            A = np.column_stack([np.ones(len(tr)), X[tr], D[tr]])
            beta, *_ = np.linalg.lstsq(A, Y[tr], rcond=None)
            pred = np.column_stack([np.ones(len(te)), X[te], D[te]]) @ beta
            sse += ((Y[te] - pred) ** 2).sum()
            sst += ((Y[te] - Y[tr].mean()) ** 2).sum()
        direct.append(1 - sse / sst)
    assert model._r2(D) == pytest.approx(np.mean(direct), rel=1e-9)


def test_synthetic_generators_match_design():
    inp, truth = SY.make_knn(300, 4, 11, 3, 6, 1.0, 1.0, 10, seed=0)
    deg = inp.A.getnnz(axis=1)
    assert deg.min() >= 10 and 12 <= deg.mean() <= 17
    inp2, truth2 = SY.make_sbm(400, 4, 11, 3, 1.0, mixing=0.3, degree=14, seed=0)
    A = inp2.A.tocoo()
    between = (truth2[A.row] != truth2[A.col]).mean()
    assert abs(between - 0.3) < 0.05 and 12 <= inp2.A.getnnz(axis=1).mean() <= 16


def test_synthetic_summary_winner_tie_and_none():
    """Победитель ячейки: интервал лучшего не перекрыт — метод; перекрыт — «ничья» с набором; лучший средний
    ARI ниже порога — «нет»."""
    rng = np.random.default_rng(0)
    rows = []
    for cell, (a, b) in {"clear": (0.9, 0.3), "close": (0.6, 0.59), "weak": (0.02, 0.01)}.items():
        for _ in range(20):
            rows.append(
                {
                    "design": cell,
                    "graph_signal": 1.0,
                    "feature_signal": 1.0,
                    "a": a + rng.normal(0, 0.02),
                    "b": b + rng.normal(0, 0.02),
                }
            )
    s = SY.summarize(pd.DataFrame(rows), ["a", "b"], {"a": "features", "b": "attributed"}, 0.05, 0.95)
    s = s.set_index("design")
    assert s.loc["clear", "winner"] == "a" and s.loc["clear", "best"] == "a"
    assert s.loc["close", "winner"] == SY.TIE and set(s.loc["close", "winner_set"].split(",")) == {"a", "b"}
    assert s.loc["weak", "winner"] == SY.NONE and s.loc["weak", "best"] == "a"
    assert (s["n_sets"] == 20).all() and (s["a_lo"] < s["a"]).all() and (s["a"] < s["a_hi"]).all()


def test_report_tradeoff_and_synthetic_counts():
    """Отчёт: итог против кандидатов фронта — лучше и хуже сверх допуска; счёт ячеек синтетики по методам."""
    from types import SimpleNamespace

    from munnet.clustering import report as R

    cands = pd.DataFrame({"method": ["hybrid", "leiden"], "k": [4, 3]}, index=["hybrid_k04", "leiden_k03"])
    lw = pd.DataFrame(
        {
            "crit_quality_features": [0.2, 0.8],
            "crit_quality_graph": [0.9, 0.1],
            "crit_stability": [0.91, 0.62],
            "crit_interpretability": [0.80, 0.81],
            "on_front": [True, True],
        },
        index=["hybrid_k04", "leiden_k03"],
    )
    cp = SimpleNamespace(
        tie={"quality_features": 0.0, "quality_graph": 0.0, "stability": 0.02, "interpretability": 0.02}
    )
    text = R._tradeoff(lw, "hybrid_k04", cands, cp)
    assert text.startswith("против «Leiden, K = 3» гибрид лучше по качеству на G, устойчивости")
    assert text.endswith("хуже — по качеству в X")  # разница объяснимости 0,01 меньше допуска — ничья
    s = pd.DataFrame(
        {
            "best": ["kmeans", "hybrid", "kmeans"],
            "winner": ["kmeans", "tie", "none"],
            "winner_set": ["kmeans", "hybrid,kmeans", ""],
            "kmeans": [0.9, 0.5, 0.01],
            "hybrid": [0.1, 0.52, 0.0],
        }
    )
    assert R._syn_counts(s) == "K-means — 1, 1 и 1; Гибрид — 1, 0 и 1"
