import numpy as np
import pandas as pd
import pytest

from munnet.eda import stats


def rng(i: int = 0) -> np.random.Generator:
    return np.random.default_rng([42, i])


def test_mad_and_robust_z_on_known_sample():
    x = pd.Series([1.0, 2.0, 3.0, 4.0, 100.0])
    assert stats.mad(x) == 1.0  # |x − 3| = 2, 1, 0, 1, 97
    z = stats.robust_z(x)
    assert z.iloc[2] == 0.0
    assert z.iloc[4] == pytest.approx(97 / 1.4826)
    assert stats.robust_z(pd.Series([5.0, 5.0, 5.0, 6.0])).isna().all()  # MAD = 0 -> не бесконечность


def test_robust_z_within_groups():
    x = pd.Series([1.0, 2.0, 3.0, 101.0, 102.0, 103.0], index=list("abcdef"))
    g = pd.Series(["x", "x", "x", "y", "y", "y"], index=list("fedcba"))[::-1]  # тот же порядок, другой индекс
    z = stats.robust_z(x, g.to_numpy())
    assert z.tolist() == pytest.approx([-1 / 1.4826, 0.0, 1 / 1.4826] * 2)


def test_weighted_quantile_equals_numpy_with_equal_weights():
    x = rng().normal(size=57)
    for q in (0.0, 0.1, 0.25, 0.5, 0.9, 1.0):
        assert stats.weighted_quantile(x, np.ones_like(x), q) == pytest.approx(np.quantile(x, q))
        assert stats.weighted_quantile(x, np.full_like(x, 3.5), q) == pytest.approx(np.quantile(x, q))


def test_weighted_quantile_and_mean_follow_weights():
    x = np.array([10.0, 20.0, 30.0, np.nan])
    w = np.array([1.0, 1.0, 100.0, 5.0])
    assert stats.weighted_quantile(x, w, 0.5) == 30.0  # 98% веса у 30
    assert stats.weighted_quantile([1.0, 2.0], [1.0, 1.0], 0.25) == pytest.approx(1.25)
    assert stats.weighted_mean(x, w) == pytest.approx((10 + 20 + 3000) / 102)
    with pytest.raises(ValueError):
        stats.weighted_mean([1.0, 2.0], [1.0, -1.0])


def test_eta2_extremes():
    groups = np.repeat(np.arange(10), 50)
    assert stats.eta2(groups * 1.0, groups) == pytest.approx(1.0)
    assert stats.eta2(rng(1).normal(size=500), groups) < 0.05


def test_spearman_drops_missing_pairs():
    x = [1.0, 2.0, 3.0, np.nan, 5.0]
    y = [2.0, 4.0, 6.0, 8.0, np.nan]
    assert stats.spearman(x, y) == (pytest.approx(1.0), 3)
    rho, n = stats.spearman([1.0, 2.0], [2.0, 1.0])
    assert np.isnan(rho) and n == 2


def test_spearman_within_removes_pure_group_effect():
    r = rng(2)
    groups = np.repeat(np.arange(8), 60)
    effect = r.normal(0, 5, 8)[groups]
    x = effect + r.normal(size=groups.size)
    y = effect + r.normal(size=groups.size)
    assert stats.spearman(x, y)[0] > 0.8
    rho_within, n = stats.spearman_within(x, y, groups)
    assert abs(rho_within) < 0.15 and n == groups.size
    z = r.normal(size=groups.size)
    assert stats.spearman_within(effect + z, effect + z + 0.1 * r.normal(size=groups.size), groups)[0] > 0.9


def test_spearman_within_drops_singleton_groups():
    rho, n = stats.spearman_within([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], ["a", "a", "b", "b", "c"])
    assert n == 4 and rho == pytest.approx(1.0)


def test_partial_spearman_controls_common_driver():
    r = rng(3)
    z = r.normal(size=800)
    x, y = z + 0.3 * r.normal(size=800), z + 0.3 * r.normal(size=800)
    assert stats.spearman(x, y)[0] > 0.7
    rho, n = stats.partial_spearman(x, y, pd.DataFrame({"z": z}))
    assert abs(rho) < 0.1 and n == 800
    groups = np.repeat(np.arange(8), 100)
    shift = 10.0 * groups  # регионы разнесены по шкале: ранги всей выборки внутри региона — его собственные
    rho_g, _ = stats.partial_spearman(x + shift, y + shift, pd.DataFrame({"z": z}), groups=groups)
    assert abs(rho_g) < 0.1


def test_spearman_ci_contains_estimate():
    r = rng(4)
    x = r.normal(size=200)
    y = x + r.normal(size=200)
    rho, lo, hi = stats.spearman_ci(x, y, n_boot=200, rng=rng(5))
    assert lo < rho < hi


def test_closure_and_clr():
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 2.0], "c": [6.0, 4.0]})
    closed = stats.closure(df)
    assert closed.sum(axis=1).tolist() == pytest.approx([1.0, 1.0])
    clr = stats.clr(closed)
    assert clr.sum(axis=1).tolist() == pytest.approx([0.0, 0.0], abs=1e-12)
    assert stats.clr(df).to_numpy() == pytest.approx(clr.to_numpy())  # CLR не зависит от масштаба строки
    with pytest.raises(ValueError):
        stats.clr(pd.DataFrame({"a": [0.0], "b": [1.0]}))


def test_smd():
    assert stats.smd([0.0, 2.0], [1.0, 3.0]) == pytest.approx(-1 / np.sqrt(2))


def test_missingness_auc_noise_and_signal():
    r = rng(6)
    n = 400
    X = pd.DataFrame({"a": r.normal(size=n), "b": r.normal(size=n)})
    y_noise = pd.Series(r.random(n) < 0.2)
    assert 0.35 < stats.missingness_auc(X, y_noise, seed=42) < 0.65
    y_signal = pd.Series(X["a"] + 0.3 * r.normal(size=n) < -0.8)
    X.loc[0, "b"] = np.nan  # пропуск в признаке не ломает модель
    assert stats.missingness_auc(X, y_signal, seed=42) > 0.8


def test_knn_weights_rows_normalised_without_self():
    pts = rng(7).random((30, 2))
    w = stats.knn_weights(pts, 4)
    assert w.shape == (30, 30)
    assert np.allclose(np.asarray(w.sum(axis=1)).ravel(), 1.0)
    assert w.diagonal().sum() == 0
    assert (w.getnnz(axis=1) == 4).all()


def grid(n: int = 20) -> tuple[np.ndarray, np.ndarray]:
    ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    return ii.ravel(), jj.ravel()


def test_morans_i_gradient_checkerboard_noise():
    ii, jj = grid()
    xy = pd.DataFrame({"x": ii, "y": jj}, dtype=float)
    smooth = stats.morans_i(pd.Series(ii + jj, dtype=float), xy, k=4, permutations=199, rng=rng(8))
    assert smooth.I > 0.5 and smooth.p < 0.01 and smooth.n == ii.size
    checker = stats.morans_i(pd.Series((ii + jj) % 2, dtype=float), xy, k=4, permutations=199, rng=rng(9))
    assert checker.I < 0
    noise = stats.morans_i(pd.Series(rng(10).normal(size=ii.size)), xy, k=8, permutations=199, rng=rng(11))
    assert abs(noise.I - (-1 / (ii.size - 1))) < 0.05
    assert noise.p > 0.01


def test_morans_i_drops_missing_before_weights():
    ii, jj = grid(10)
    y = pd.Series(ii + jj, dtype=float)
    y.iloc[:5] = np.nan
    res = stats.morans_i(y, pd.DataFrame({"x": ii, "y": jj}, dtype=float), k=4, permutations=99, rng=rng(12))
    assert res.n == 95 and res.I > 0.5
