"""Этап evaluate на синтетике: поддельные метки этапа cluster на маленьком графе и признаках (без data/)."""

import copy
import json
import math

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

from munnet import icvi, icvi_report, icvi_stage
from munnet.config import Config, load_config
from munnet.contracts import CLUSTER_LABELS, MissingInputError, QCError, coerce, validate, write_table

METRICS = ["sw", "ch", "s_dbw", "avi", "avu", "mq"]


def graph_truth(truth, n_per=30):
    """Разбиение сети — сдвинутое на полблока разбиение признаков: X и G тянут в разные стороны."""
    return np.roll(truth, n_per // 2)


def synth(n_per=30, k=4, seed=0):
    """Признаки с разбиением на k облаков и граф с посаженным другим разбиением; id узлов не подряд."""
    rng = np.random.default_rng(seed)
    x, truth = make_blobs(
        n_samples=[n_per] * k, n_features=5, cluster_std=2.0, center_box=(-8, 8), random_state=seed
    )
    n = x.shape[0]
    tg = graph_truth(truth, n_per)
    prob = np.where(tg[:, None] == tg[None, :], 0.3, 0.03)
    upper = np.triu(rng.random((n, n)) < prob, k=1) * rng.uniform(0.5, 1.0, size=(n, n))
    a = sp.csr_matrix(upper + upper.T)
    ids = 1000 + 3 * np.arange(n)
    return ids, x, a, truth


def fake_labels(ids, x, truth, seed=0):
    """Кандидаты «метод × K» в схеме CLUSTER_LABELS: K-means по X (K = 3, 4, 5), «граф» = истина (K = 4)
    и перемешанная истина (K = 3), HDBSCAN с шумом (K = 4)."""
    rng = np.random.default_rng(seed)
    cands = {}
    for k in (3, 4, 5):
        cands[("kmeans", k)] = KMeans(k, n_init=5, random_state=0).fit_predict(x)
    cands[("leiden", 4)] = graph_truth(truth)
    shuffled = rng.permutation(truth)
    shuffled[shuffled == 3] = 2
    cands[("leiden", 3)] = shuffled
    noisy = truth.copy()
    noisy[rng.choice(len(truth), size=len(truth) // 10, replace=False)] = icvi.NOISE
    cands[("hdbscan", 4)] = noisy
    winners = {("kmeans", 4), ("leiden", 4), ("hdbscan", 4)}
    rows = []
    for (method, k), lab in cands.items():
        rows.append(
            pd.DataFrame(
                {
                    "candidate": f"{method}_k{k:02d}",
                    "territory_id": ids.astype("int32"),
                    "method": method,
                    "k": np.int16(k),
                    "param": float(k) if method != "leiden" else 0.5,
                    "label": lab.astype("int16"),
                    "feasible": (method, k) != ("leiden", 3),
                    "is_method_winner": (method, k) in winners,
                    "is_final": (method, k) == ("kmeans", 4),
                }
            )
        )
    return coerce(pd.concat(rows, ignore_index=True), CLUSTER_LABELS)


def params(**kw):
    cfg = load_config()
    base = icvi_stage.EvalParams.from_config(cfg)
    return icvi_stage.EvalParams(**{**base.__dict__, "permutations": 20, "bootstrap": 12, "workers": 1, **kw})


@pytest.fixture(scope="module")
def result():
    ids, x, a, truth = synth()
    tab = fake_labels(ids, x, truth)
    return ids, x, a, tab, icvi_stage.evaluate_candidates(ids, x, a, tab, params())


# ---------------------------------------------------------------------------------------------


def test_params_from_default_config():
    p = icvi_stage.EvalParams.from_config(load_config())
    assert p.permutations == 200
    assert p.bootstrap >= 100
    assert p.ci_level == 0.95
    assert 0 < p.subsample < 1
    assert p.options == {"s_dbw_density": "pair"}
    assert list(p.specs) == METRICS
    assert p.families["kmeans"] == "features" and p.families["leiden"] == "graph"


def test_long_table_schema_and_values(result):
    ids, x, a, tab, res = result
    long = res.long
    assert list(long.columns) == [
        "candidate",
        "method",
        "k",
        "space",
        "metric",
        "value",
        "ci_low",
        "ci_high",
        "baseline_mean",
        "baseline_sd",
        "z",
        "n_undefined",
        "percentile",
    ]
    assert set(long["metric"]) == {*METRICS, "anui"}
    assert len(long) == 6 * 7
    validate(long, icvi_stage.ICVI_LONG)
    lab = tab.loc[tab["candidate"] == "kmeans_k04"].set_index("territory_id").loc[ids, "label"].to_numpy()
    expected = icvi.evaluate(lab, X=x, A=a)
    got = long.loc[long["candidate"] == "kmeans_k04"].set_index("metric")
    for name in METRICS:
        assert got.loc[name, "value"] == pytest.approx(expected[name], abs=1e-12), name
    assert got.loc["anui", "value"] == pytest.approx(icvi.anui(expected["avi"], expected["avu"]))
    assert (got.loc[["sw", "ch", "s_dbw"], "space"] == "features").all()
    assert (got.loc[["avi", "avu", "mq", "anui"], "space"] == "graph").all()


def test_z_uses_same_permutations_as_cluster_stage(result):
    """Базис — тот же рецепт seed, что у этапа cluster (compute.icvi_task): z совпадают поэлементно."""
    ids, x, a, tab, res = result
    import zlib

    p = params()
    lab = tab.loc[tab["candidate"] == "leiden_k04"].set_index("territory_id").loc[ids, "label"].to_numpy()
    rng = np.random.default_rng([p.seed, 17, zlib.crc32(b"leiden_k04")])
    base = icvi.random_baseline(lab, X=x, A=a, names=METRICS, n_perm=20, rng=rng, **p.options)
    base = base.set_index("name")
    got = res.long.loc[res.long["candidate"] == "leiden_k04"].set_index("metric")
    for name in METRICS:
        assert got.loc[name, "baseline_mean"] == pytest.approx(base.loc[name, "mean"], abs=1e-12)
        assert got.loc[name, "baseline_sd"] == pytest.approx(base.loc[name, "sd"], abs=1e-12)
        assert got.loc[name, "n_undefined"] == base.loc[name, "n_undefined"]
        z = icvi.zscore(
            got.loc[name, "value"], base.loc[name, "mean"], base.loc[name, "sd"], icvi.BETTER[name]
        )
        assert (math.isnan(z) and math.isnan(got.loc[name, "z"])) or got.loc[name, "z"] == pytest.approx(z)


def test_intervals_contain_value_and_avu_k3_is_degenerate(result):
    _, _, _, _, res = result
    long = res.long
    fin = long.dropna(subset=["ci_low", "ci_high"])
    assert len(fin) > 0.8 * len(long)
    assert ((fin["ci_low"] <= fin["value"] + 1e-12) & (fin["value"] <= fin["ci_high"] + 1e-12)).all()
    k3 = long.loc[(long["metric"] == "avu") & (long["k"] == 3)]
    assert np.allclose(k3["value"], 2 / 3)
    assert k3["z"].isna().all()


def test_candidates_table_flags_and_noise(result):
    _, _, _, _, res = result
    c = res.cands.set_index("candidate")
    assert c.loc["hdbscan_k04", "noise_share"] == pytest.approx(0.1, abs=0.01)
    assert c.loc["kmeans_k03", "noise_share"] == 0
    assert bool(c.loc["kmeans_k04", "is_final"]) and not bool(c.loc["leiden_k03", "feasible"])
    assert c.loc["kmeans_k04", "family"] == "features"
    assert c.loc["leiden_k04", "family"] == "graph"
    assert {"scat", "dens_bw", "stdev"} <= set(c.columns)


def test_subsample_interval_matches_sampling_ci_for_a_mean():
    """Подвыборка m = 0,8·n без возвращения: разброс среднего подвыборки в √((n − m)/m) = 1/2 раза меньше
    разброса среднего выборки; после масштаба √(m/(n − m)) ширина ≈ 2·1,96·σ/√n."""
    rng = np.random.default_rng(0)
    n, m = 2000, 1600
    sample = rng.normal(size=n)
    boot = np.array([rng.choice(sample, size=m, replace=False).mean() for _ in range(400)])
    lo, hi = icvi_stage.subsample_interval(sample.mean(), boot, m=m, n=n, level=0.95)
    assert lo < sample.mean() < hi
    assert hi - lo == pytest.approx(2 * 1.96 / math.sqrt(n), rel=0.15)
    assert icvi_stage.subsample_interval(1.0, np.array([np.nan] * 5), m=m, n=n, level=0.95) == (
        pytest.approx(float("nan"), nan_ok=True),
        pytest.approx(float("nan"), nan_ok=True),
    )


def test_agreement_is_symmetric_with_unit_diagonal(result):
    _, _, _, _, res = result
    tau = icvi_stage.agreement(res.long, res.cands, value="z")
    assert list(tau.index) == METRICS and list(tau.columns) == METRICS
    np.testing.assert_allclose(np.diag(tau.to_numpy()), 1.0)
    np.testing.assert_allclose(tau.to_numpy(), tau.to_numpy().T, equal_nan=True)
    raw = icvi_stage.agreement(res.long, res.cands, value="value")
    assert raw.shape == (6, 6)


def test_agreement_signs_min_metrics():
    """Ранжирование «лучше — выше»: у метрики «меньше — лучше» сырые значения берутся со знаком минус."""
    long = pd.DataFrame(
        {
            "candidate": [f"m_k0{i}" for i in range(4)] * 2,
            "metric": ["sw"] * 4 + ["s_dbw"] * 4,
            "value": [0.1, 0.2, 0.3, 0.4, 4.0, 3.0, 2.0, 1.0],
            "z": [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0],
        }
    )
    cands = pd.DataFrame({"candidate": [f"m_k0{i}" for i in range(4)], "feasible": True})
    tau = icvi_stage.agreement(long, cands, value="value", metrics=["sw", "s_dbw"])
    assert tau.loc["sw", "s_dbw"] == pytest.approx(1.0)


def test_space_ranks_and_k_dependence(result):
    _, _, _, _, res = result
    q = icvi_stage.space_ranks(res.long, res.cands).set_index("candidate")
    assert q[["q_features", "q_graph"]].stack().between(0, 1).all()
    # K-means оптимизирует X, «граф» = посаженное разбиение сети: каждый выше в своём пространстве
    assert q.loc["kmeans_k04", "q_features"] >= q.loc["leiden_k04", "q_features"]
    assert q.loc["leiden_k04", "q_graph"] >= q.loc["kmeans_k03", "q_graph"]
    kd = icvi_stage.k_dependence(res.long, res.cands)
    assert set(kd["metric"]) == set(METRICS)
    assert {"rho_value_k", "rho_z_k"} <= set(kd.columns)


def test_final_unifiability_table(result):
    ids, x, a, tab, res = result
    u = res.final_u
    assert list(u["type"]) == [1, 2, 3, 4]
    assert {"isolability", "type_1", "type_4", "size"} <= set(u.columns)
    avu_final = res.long.set_index(["candidate", "metric"]).loc[("kmeans_k04", "avu"), "value"]
    assert np.nansum(u[[f"type_{j}" for j in range(1, 5)]].to_numpy()) / 4 == pytest.approx(avu_final)


def test_labels_must_cover_the_same_nodes(result):
    ids, x, a, tab, _ = result
    bad = tab.loc[tab["territory_id"] != ids[0]]
    with pytest.raises(QCError, match="узл"):
        icvi_stage.evaluate_candidates(ids, x, a, bad, params())


# ---------------------------------------------------------------------------------------------
# Этап целиком: run(cfg) во временной папке


def _cfg(tmp_path) -> Config:
    data = copy.deepcopy(load_config().data)
    for key in ("interim", "processed", "outputs"):
        data["paths"][key] = str(tmp_path / key)
    data["icvi"]["baseline_permutations"] = 20
    data["icvi"]["bootstrap"] = 12
    data["icvi"]["report"] = str(tmp_path / "docs" / "icvi.md")
    data["icvi"]["report_images"] = str(tmp_path / "docs" / "img" / "icvi")
    data["clustering"]["impl"]["workers"] = 1
    return Config(data=data, path=tmp_path / "check.yaml")


def test_run_without_cluster_labels_is_missing_input(tmp_path, monkeypatch):
    ids, x, a, _ = synth()
    monkeypatch.setattr(icvi_stage, "load_xa", lambda cfg: (ids, x, a))
    with pytest.raises(MissingInputError, match="cluster"):
        icvi_stage.run(_cfg(tmp_path))


def test_run_end_to_end(tmp_path, monkeypatch):
    ids, x, a, truth = synth()
    cfg = _cfg(tmp_path)
    write_table(fake_labels(ids, x, truth), CLUSTER_LABELS, tmp_path / "processed" / "cluster_labels.parquet")
    monkeypatch.setattr(icvi_stage, "load_xa", lambda cfg: (ids, x, a))
    monkeypatch.setattr(icvi_report, "CLAIMS", {})  # утверждения о реальных данных; проверены ниже отдельно
    icvi.run(cfg)
    out = tmp_path / "outputs" / "evaluate"
    long = pd.read_csv(out / "icvi_long.csv")
    assert len(long) == 42
    assert (out / "icvi_long.parquet").exists()
    for name in (
        "candidates.csv",
        "agreement_z.csv",
        "agreement_value.csv",
        "agreement_within_k.csv",
        "k_dependence.csv",
        "k_fair.csv",
    ):
        assert (out / name).exists(), name
    facts = json.loads((out / "report_facts.json").read_text(encoding="utf-8"))
    assert facts["icvi.n_candidates"]["value"] == 6
    pngs = sorted(p.name for p in (tmp_path / "docs" / "img" / "icvi").glob("*.png"))
    assert len(pngs) == 5
    md = (tmp_path / "docs" / "icvi.md").read_text(encoding="utf-8")
    assert "{{" not in md and "<!--" not in md
    assert "Biswas" in md and "Шалилех" not in md.split("Biswas")[0][-200:]


def test_all_claims_are_evaluable_on_synthetic_facts(tmp_path, monkeypatch):
    """Каждое утверждение шаблона вычислимо (нужные факты существуют); верность — дело реальных данных."""
    ids, x, a, truth = synth()
    cfg = _cfg(tmp_path)
    write_table(fake_labels(ids, x, truth), CLUSTER_LABELS, tmp_path / "processed" / "cluster_labels.parquet")
    monkeypatch.setattr(icvi_stage, "load_xa", lambda cfg: (ids, x, a))
    claims = dict(icvi_report.CLAIMS)
    monkeypatch.setattr(icvi_report, "CLAIMS", {})
    icvi.run(cfg)
    facts = icvi_report.load_facts(tmp_path / "outputs" / "evaluate")
    for text, fn in claims.items():
        assert isinstance(bool(fn(facts)), bool), text
    assert claims, "в шаблоне должны быть проверяемые утверждения"


def test_broken_claim_stops_the_report(tmp_path, monkeypatch):
    ids, x, a, truth = synth()
    cfg = _cfg(tmp_path)
    write_table(fake_labels(ids, x, truth), CLUSTER_LABELS, tmp_path / "processed" / "cluster_labels.parquet")
    monkeypatch.setattr(icvi_stage, "load_xa", lambda cfg: (ids, x, a))
    monkeypatch.setattr(icvi_report, "CLAIMS", {"заведомо неверное": lambda f: False})
    with pytest.raises(QCError, match="заведомо неверное"):
        icvi.run(cfg)
    assert not (tmp_path / "docs" / "icvi.md").exists()


def test_z_match_cluster(result, tmp_path):
    _, _, _, _, res = result
    assert icvi_stage.z_match_cluster(tmp_path / "нет.csv", res.long)["n"] == 0
    w = res.long.pivot(index="candidate", columns="metric", values="z")
    cand = pd.DataFrame({"cand": w.index, **{f"z_{m}": w[m].round(4).to_numpy() for m in METRICS}})
    path = tmp_path / "candidates.csv"
    cand.to_csv(path, index=False)
    ok = icvi_stage.z_match_cluster(path, res.long)
    assert ok["n"] > 0 and ok["nan_mismatch"] == 0 and ok["rel"] < 1e-4
    cand.loc[0, "z_sw"] += 1.0
    cand.to_csv(path, index=False)
    assert icvi_stage.z_match_cluster(path, res.long)["rel"] > 0.01


def test_percentile_in_baseline_direction_and_ties():
    base = np.array([1.0, 2.0, 3.0, 4.0, np.nan])
    assert icvi_stage.percentile_in_baseline(3.0, base, "max") == pytest.approx((2 + 0.5) / 4)
    assert icvi_stage.percentile_in_baseline(3.0, base, "min") == pytest.approx((1 + 0.5) / 4)
    assert icvi_stage.percentile_in_baseline(9.0, base, "max") == 1.0
    assert math.isnan(icvi_stage.percentile_in_baseline(float("nan"), base, "max"))


def test_agreement_within_k_removes_the_k_confound():
    """Метрики A и B противоположно зависят от K, но внутри каждого K упорядочивают кандидатов одинаково:
    τ по всем кандидатам отрицателен, τ внутри K равен 1."""
    rows, cands = [], []
    for k in (3, 6, 9):
        for j in range(5):
            cand = f"m{j}_k{k:02d}"
            cands.append({"candidate": cand, "k_eff": k, "feasible": True})
            rows.append({"candidate": cand, "metric": "sw", "value": -10 * k + j, "z": float(j)})
            rows.append({"candidate": cand, "metric": "avi", "value": 10 * k + j, "z": float(j)})
    long = pd.DataFrame(rows)
    cands = pd.DataFrame(cands)
    pooled = icvi_stage.agreement(long, cands, value="value", metrics=["sw", "avi"])
    assert pooled.loc["sw", "avi"] < 0
    within, by_k = icvi_stage.agreement_within_k(long, cands, value="value", min_n=5)
    assert within.loc["sw", "avi"] == pytest.approx(1.0)
    assert sorted(by_k.loc[(by_k["a"] == "sw") & (by_k["b"] == "avi"), "k"]) == [3, 6, 9]
    assert math.isnan(within.loc["sw", "mq"])  # пары без данных — NaN, а не 0


def test_k_fair_table(result):
    _, _, _, _, res = result
    kf = icvi_stage.k_fair(res.long, res.cands).set_index("candidate")
    r = kf.loc["leiden_k04"]
    assert r["avi_adj"] == pytest.approx((r["avi"] - r["avi_base"]) / (1 - r["avi_base"]))
    assert r["avi_base"] == pytest.approx(0.25, abs=0.03)  # случайный базис AVI ≈ 1/K
    assert kf.loc[kf["k_eff"] == 4, "rank_in_k_avi"].min() == 1
    assert kf.loc["leiden_k04", "rank_in_k_avi"] == 1  # посаженное разбиение сети — лучшее при K = 4
    assert kf["pct_avi"].between(0, 1).all()
