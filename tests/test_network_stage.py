"""Этап network на синтетике: выходы по контрактам, выбор из кандидатов, повторяемость, отчёт."""

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from synth import make_config, make_processed

from munnet import features, network
from munnet.config import Config
from munnet.contracts import FEATURES_NODES, NETWORK_SCHEMAS, TERRITORIES, QCError, read_table
from munnet.eda.base import make_fact
from munnet.network import report
from munnet.network.graph import haversine_km
from munnet.network.params import NetworkParams


def write_connection(cfg: Config) -> None:
    """``connection.parquet``: автодороги — каждая пара один раз (x > y), расстояние = 1,3 × по прямой;
    железные дороги — у части пар в обе стороны, как в источнике."""
    ter = read_table(Path(cfg["paths"]["processed"]) / "territories.parquet", TERRITORIES)
    ids = ter["territory_id"].to_numpy()
    lat, lon = ter["point_lat"].to_numpy(), ter["point_lon"].to_numpy()
    i, j = np.triu_indices(len(ids), 1)
    km = 1.3 * haversine_km(lat[i], lon[i], lat[j], lon[j])
    hw = pd.DataFrame({"territory_id_x": ids[j], "territory_id_y": ids[i], "distance": np.round(km, 1)})
    hw["type"] = "highway"
    rail = hw.iloc[::3].copy()
    rail["type"] = "railway"
    back = rail.rename(columns={"territory_id_x": "territory_id_y", "territory_id_y": "territory_id_x"})
    conn = pd.concat([hw, rail, back], ignore_index=True).astype(
        {"territory_id_x": "int64", "territory_id_y": "int64"}
    )
    path = Path(cfg["paths"]["raw"]) / cfg["panel"]["connection"]
    path.parent.mkdir(parents=True, exist_ok=True)
    conn.to_parquet(path, index=False)


def light(cfg: Config, root: Path) -> Config:
    """Лёгкие параметры для синтетики: меньше повторов, k под десятки узлов, без отчёта (его утверждения
    написаны под реальные данные — проверяются отдельно)."""
    data = copy.deepcopy(cfg.data)
    net = data["network"]
    net["sparsify"].update({"k": 4, "k_grid": [3, 4]})
    net["nulls"].update({"shift_repeats": 2, "rewire_repeats": 3, "mantel_permutations": 5})
    net["bootstrap"] = 3
    net["probe"]["seeds"] = 3
    net["report"] = ""
    net["report_images"] = str(root / "docs" / "img" / "network")
    return Config(data=data, path=cfg.path)


@pytest.fixture(scope="module")
def run_net(tmp_path_factory):
    root = tmp_path_factory.mktemp("network")
    make_processed(root)
    cfg = make_config(root)
    features.run(cfg)
    write_connection(cfg)
    cfg = light(cfg, root)
    network.run(cfg)
    return cfg


def test_outputs_follow_contracts(run_net):
    processed = Path(run_net["paths"]["processed"])
    tables = {name: read_table(processed / f"{name}.parquet", s) for name, s in NETWORK_SCHEMAS.items()}
    edges = tables["network_edges"]
    assert (edges["source"] < edges["target"]).all()  # симметрия: ребро записано один раз, петель нет
    assert np.isfinite(edges["weight"]).all()
    p = NetworkParams.from_config(run_net)
    main = edges.loc[edges["is_main"]]
    assert set(main["rule"]) == set(p.rules)
    # у лагового правила есть сдвиг, у остальных — нет
    assert main.loc[main["rule"] == "rhythm_lag", "lag"].notna().all()
    assert main.loc[main["rule"] == "basket_dist", "lag"].isna().all()
    assert main.loc[main["rule"].str.startswith("rhythm"), "q_value"].between(0, 1).all()
    win = tables["network_windows"]
    assert set(win["window_kind"].astype(str)) == set(p.dyn_kinds)
    assert win.loc[win["is_main_kind"], "window_kind"].astype(str).eq(p.dyn_main).all()
    nodes = tables["network_window_nodes"]
    assert nodes.groupby(["window_kind", "window"]).size().nunique() == 1  # в каждом окне все узлы


def test_knn_degree_at_least_k_and_no_isolates_for_spending_rules(run_net):
    processed = Path(run_net["paths"]["processed"])
    nodes = read_table(processed / "network_nodes.parquet", NETWORK_SCHEMAS["network_nodes"])
    p = NetworkParams.from_config(run_net)
    spend = nodes.loc[nodes["rule"].isin(p.candidates) & (nodes["rule"] != "basket_cos")]
    assert (spend["degree"] >= p.k).all()
    # косинус не определён для нулевого вектора: узел — единственный в своей группе региона (корзина равна
    # центру группы) остаётся изолятом; расстояние такой узел связывает
    fn = read_table(processed / "features_nodes.parquet", FEATURES_NODES)
    fn = fn.loc[fn["is_node"]]
    single = fn.groupby("region_group")["territory_id"].transform("size") == 1
    lonely = set(fn.loc[single, "territory_id"])
    cos = nodes.loc[nodes["rule"] == "basket_cos"].set_index("territory_id")["degree"]
    assert set(cos.index[cos == 0]) <= lonely


def test_selection_and_tables(run_net):
    out = Path(run_net["paths"]["outputs"]) / "network"
    comp = pd.read_csv(out / "comparison.csv")
    p = NetworkParams.from_config(run_net)
    assert set(comp["rule"]) == set(p.rules)
    facts = json.loads((out / "facts.json").read_text(encoding="utf-8"))
    assert facts["chosen"] in p.candidates
    sets = pd.read_csv(out / "selection_sets.csv")
    assert set(sets["set"]) == set(p.criteria_sets) and set(sets["tie"]) == {True, False}
    main = sets.loc[sets["set"] == "main"]
    assert (main["orders"] == 24).all()  # 4! порядков основного набора
    assert set(facts["tolerance"]) >= {"reliability", "attribute_consistency"}
    assert facts["tolerance"]["simplicity"] == 0.0
    pairs = pd.read_csv(out / "rule_pairs.csv")
    assert (pairs["rank_corr_q"] >= pairs["rank_corr_p"] - 1e-12).all()  # поправка не уменьшает p
    # абляции есть в сравнении, но не в выборе; текстовые колонки — в конце таблицы
    assert {"basket_dist_abs", "rhythm_corr_rel"} <= set(comp["rule"])
    assert list(comp.columns[-3:]) == ["time_bucket", "top_hubs", "meaning"]
    assert set(pd.read_csv(out / "selection_criteria.csv")["rule"]) == set(p.candidates)
    road = comp.set_index("rule").loc["geo_road"]
    assert road["geo_jaccard"] == 1.0  # контроль совпадает сам с собой
    for name in ("sensitivity_k", "sparsify", "rule_pairs", "selection_orders", "modes", "windows_noise"):
        assert len(pd.read_csv(out / f"{name}.csv")) > 0, name
    modes = pd.read_csv(out / "modes.csv")
    assert (modes["mode"] == "separate").all() and (modes["n_inner"] > 0).all()
    for fid in ("N01", "N02", "N03", "N04"):
        assert list((out / "figures").glob(f"{fid}_*.png")), fid


def _hashes(cfg: Config) -> dict[str, str]:
    paths = sorted(Path(cfg["paths"]["processed"]).glob("network_*.parquet"))
    paths += sorted(
        p for p in (Path(cfg["paths"]["outputs"]) / "network").glob("*.*") if p.name != "timing.csv"
    )
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def test_run_is_repeatable(run_net):
    before = _hashes(run_net)
    network.run(run_net)
    assert _hashes(run_net) == before


def test_params_reject_bad_values(run_net):
    data = copy.deepcopy(run_net.data)
    data["network"]["sparsify"]["k"] = 7  # не входит в k_grid
    with pytest.raises(ValueError, match="k_grid"):
        NetworkParams.from_config(Config(data=data, path=run_net.path))
    data = copy.deepcopy(run_net.data)
    data["network"]["selection"]["priority"] = ["reliability"]
    with pytest.raises(ValueError, match="priority"):
        NetworkParams.from_config(Config(data=data, path=run_net.path))


def test_report_claims_stop_the_build_when_numbers_change():
    facts = {
        f"net.{k}": make_fact(f"net.{k}", v, "str") for k, v in {"chosen_label": "Косинус корзин"}.items()
    }
    with pytest.raises(QCError, match="выбрано правило"):
        report.check_claims(facts)
