"""Этап cluster на синтетике: выходы по контрактам, выбор по предрегистрированному правилу, проверки
чувствительности, варианты входов, повторяемость."""

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from synth import make_config, make_processed
from test_network_stage import light, write_connection

from munnet import clustering, features, network
from munnet.clustering.params import CRITERIA, ClusterParams
from munnet.config import Config
from munnet.contracts import CLUSTER_SCHEMAS, read_table


def light_cluster(cfg: Config) -> Config:
    """Малые повторы и сетки: логика этапа та же, считается за секунды."""
    data = copy.deepcopy(cfg.data)
    c = data["clustering"]
    c["inputs"]["graph"]["k"] = data["network"]["sparsify"]["k"]
    c["n_clusters"] = [3, 5]
    c["feasible"]["k_range"] = [3, 5]
    c["protocol"].update({"seeds": 2, "alpha_curve": [0.0, 0.5, 1.0]})
    c["stability"].update({"bootstrap": 3, "bootstrap_variants": 2})
    c["validation"].update({"permutations": 19, "beyond_attributes": {"folds": 3, "repeats": 2}})
    c["impl"].update(
        {
            "workers": 1,
            "kmeans_n_init": 3,
            "leiden_grid": {"gamma_min": 0.05, "gamma_max": 5.0, "points": 10, "refine": 4},
            "hdbscan_grid": {"min_size": 3, "max_size": 20, "points": 6},
            "edge_perturbation": {"drop": 0.1, "repeats": 2},
        }
    )
    c["synthetic"].update(
        {
            "n_nodes": 60,
            "k_true": 3,
            "repeats": 1,
            "graph_signal": [0.0, 2.0],
            "feature_signal": [0.0, 2.0],
            "sbm_mixing": [0.3],
            "sbm_degree": 8,
            "seeds": 1,
            "methods": ["kmeans", "leiden", "shalileh_mirkin", "hybrid", "kmeans_joint"],
        }
    )
    c["report"] = ""
    data["icvi"]["baseline_permutations"] = 10
    return Config(data=data, path=cfg.path)


@pytest.fixture(scope="module")
def run_cluster(tmp_path_factory):
    root = tmp_path_factory.mktemp("cluster")
    make_processed(root)
    cfg = make_config(root)
    features.run(cfg)
    write_connection(cfg)
    cfg = light(cfg, root)
    network.run(cfg)
    cfg = light_cluster(cfg)
    clustering.run(cfg)
    return cfg


def _out(cfg: Config) -> Path:
    return Path(cfg["paths"]["outputs"]) / "cluster"


def test_outputs_follow_contracts_and_final_is_eligible(run_cluster):
    processed = Path(run_cluster["paths"]["processed"])
    tables = {n: read_table(processed / f"{n}.parquet", s) for n, s in CLUSTER_SCHEMAS.items()}
    labels, final = tables["cluster_labels"], tables["cluster_final"]
    cp = ClusterParams.from_config(run_cluster)
    fj = json.loads((_out(run_cluster) / "final.json").read_text(encoding="utf-8"))
    assert fj["method"] in cp.eligible_methods
    chosen = labels.loc[labels["is_final"]]
    assert chosen["candidate"].nunique() == 1 and chosen["candidate"].iloc[0] == fj["candidate"]
    assert chosen["feasible"].all()
    # итоговый тип = метка кандидата + 1, на тех же узлах
    m = final.merge(chosen, on="territory_id")
    assert len(m) == len(final) and (m["type"] == m["label"] + 1).all()
    assert final["k"].eq(fj["k"]).all() and final["type"].nunique() == fj["k"]
    assert final["type_jaccard"].between(0, 1).all()


def test_candidate_table_has_all_criteria_and_feasibility(run_cluster):
    cands = pd.read_csv(_out(run_cluster) / "candidates.csv")
    cp = ClusterParams.from_config(run_cluster)
    assert set(cands["method"]) <= set(cp.methods)
    for col in (*[c for c in CRITERIA if not c.startswith("quality")], "feasible", "k", "seed_ari"):
        assert col in cands.columns, col
    for name in cp.icvi_metrics:
        assert f"z_{name}" in cands.columns and f"z_{name}_nc" in cands.columns
    assert "z_s_dbw_own" in cands.columns and "n_metrics_graph" in cands.columns
    assert cands["stability"].between(-1, 1).all()
    bad = cands.loc[~cands["feasible"].astype(bool)]
    assert bad["infeasible_reason"].fillna("").str.len().gt(0).all()


def test_selection_intermediate_and_sensitivity(run_cluster):
    out = _out(run_cluster)
    levels = pd.read_csv(out / "selection_levels.csv")
    assert set(levels["level"]) == {1, 2}
    assert levels.groupby(["level", "set"])["winner"].sum().eq(1).all()  # один победитель в каждом наборе
    win2 = levels.loc[(levels["level"] == 2) & levels["winner"]]
    assert bool(win2["on_front"].iloc[0])
    checks = pd.read_csv(out / "sensitivity.csv")
    expected = {"main", "borda", "copeland_all", "set_icvi_only", "set_plan", "tie_x0", "tie_x2"}
    assert expected | {"joint", "all_eligible"} <= set(checks["check"])
    assert len(pd.read_csv(out / "sensitivity_orders.csv")) == 24
    icvi = pd.read_csv(out / "sensitivity_icvi.csv")
    assert {"no_avu", "avi_mq_one", "no_s_dbw", "s_dbw_own", "nan_worst", "noise_cluster"} <= set(
        icvi["check"]
    )
    edges = pd.read_csv(out / "k_edges.csv")
    assert edges["k_on_edge"].isin([True, False]).all()


def test_variants_and_extra_checks(run_cluster):
    v = pd.read_csv(_out(run_cluster) / "variants.csv")
    assert list(v["variant"]) == ["graph_basket_cos", "no_level", "nodes_separate", "x_clipped"]
    sep = v.set_index("variant").loc["nodes_separate"]
    # в синтетике у города один район: узлов столько же; на данных районов больше (242 против 2)
    assert sep["n_nodes"] >= v.set_index("variant").loc["no_level", "n_nodes"]
    assert v.set_index("variant").loc["no_level", "n_features"] == 10
    assert v["ari_winner_vs_main"].between(-1, 1).all()
    extra = pd.read_csv(_out(run_cluster) / "extra_checks.csv")
    assert set(extra["check"]) == {"min_share_01", "min_share_01_all"}


def test_validation_synthetic_and_stability_tables(run_cluster):
    out = _out(run_cluster)
    val = pd.read_csv(out / "validation.csv")
    fj = json.loads((out / "final.json").read_text(encoding="utf-8"))
    fin = val.loc[val["cand"] == fj["candidate"]]
    assert set(fin["check"]) == {"difference", "sign", "beyond_attributes"}
    assert fin["p"].between(0, 1).all() and fin["q"].between(0, 1).all()
    syn = pd.read_csv(out / "synthetic_summary.csv")
    assert set(syn["design"]) == {"knn", "sbm"} and syn["winner"].notna().all()
    raw = pd.read_csv(out / "synthetic_raw.csv")
    assert raw.loc[raw["design"] == "sbm", "kmeans_joint"].isna().all()  # у блочного графа нет координат
    jac = pd.read_csv(out / "stability_clusters.csv")
    assert jac["jaccard"].dropna().between(0, 1).all()
    assert len(pd.read_csv(out / "alpha_curve.csv")) == 3
    U = pd.read_csv(out / "avu_matrix.csv")
    assert len(U) == fj["k"]


def _hashes(cfg: Config) -> dict[str, str]:
    paths = sorted(Path(cfg["paths"]["processed"]).glob("cluster_*.parquet"))
    out = _out(cfg)
    paths += [out / "facts.json", out / "candidates.csv", out / "sensitivity.csv", out / "validation.csv"]
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def test_rerun_is_identical(run_cluster):
    before = _hashes(run_cluster)
    clustering.run(run_cluster)
    assert _hashes(run_cluster) == before


def test_workers_do_not_change_result(run_cluster, tmp_path):
    """Итог не зависит от числа процессов: seed задачи задан её номером."""
    data = copy.deepcopy(run_cluster.data)
    data["clustering"]["impl"]["workers"] = 2
    data["clustering"]["synthetic"]["repeats"] = 0
    data["clustering"]["selection"]["sensitivity"]["variants"] = {}
    data["clustering"]["impl"]["extra_checks"] = {"min_cluster_share": 0.01}
    data["paths"]["outputs"] = str(tmp_path / "outputs")
    cfg2 = Config(data=data, path=run_cluster.path)
    from munnet.clustering import compute as CO
    from munnet.clustering.inputs import load_inputs

    cp = ClusterParams.from_config(cfg2)
    inp = load_inputs(cfg2, cp).inputs
    one = CO.evaluate_set(inp, ClusterParams.from_config(run_cluster), 3)
    two = CO.evaluate_set(inp, cp, 3)
    for c in one.labels:
        assert np.array_equal(one.labels[c], two.labels[c]), c
    pd.testing.assert_series_equal(one.cands["stability"], two.cands["stability"])
