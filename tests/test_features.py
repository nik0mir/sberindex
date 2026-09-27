"""Этап features: окна корзины, замена почти нулей, «относительно группы региона», ритм, место, запуск."""

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from nodes_synth import (
    city_context,
    context_annual,
    node_config,
    okved_shares,
    panel_wide,
    territories,
)
from synth import make_config, make_processed

from munnet import features, nodes
from munnet.config import Config
from munnet.contracts import FEATURE_SCHEMAS, PARTS, TERRITORIES, read_table

MSK = 90077


def params(**over) -> features.FeatureParams:
    cfg = make_config(Path("."))
    data = copy.deepcopy(cfg.data)
    data["features"].update(over)
    return features.FeatureParams.from_config(Config(data=data, path=cfg.path))


@pytest.fixture(scope="module")
def tiny():
    prm = nodes.NodeParams.from_section(node_config(), recipients_ratio_ok=(0.05, 3.0))
    nd = nodes.build_nodes(panel_wide(), territories(), context_annual(), okved_shares(), city_context(), prm)
    groups = nd.nodes.set_index("territory_id")["region_group"]
    return nd, groups


# --- Композиции ----------------------------------------------------------------------------------


def test_replace_small_is_multiplicative_and_keeps_closure():
    shares = pd.DataFrame([[0.0005, 0.4995, 0.5], [0.2, 0.3, 0.5]], columns=list("abc"))
    out, k = features.replace_small(shares, 0.001)
    assert out.iloc[0, 0] == pytest.approx(0.001)
    assert np.allclose(out.sum(axis=1), 1.0)
    # остальные доли сжаты в одно и то же число раз: их отношение не меняется
    assert out.iloc[0, 1] / out.iloc[0, 2] == pytest.approx(0.4995 / 0.5)
    assert list(k) == [1, 0]
    assert np.allclose(out.iloc[1], shares.iloc[1])  # строку без почти нулей не трогаем
    same, k0 = features.replace_small(shares, 0.0)
    assert np.allclose(same, shares) and (k0 == 0).all()


def test_relative_clr_is_zero_sum_and_centered_by_group():
    rng = np.random.default_rng(0)
    shares = pd.DataFrame(rng.dirichlet(np.ones(6), size=12), columns=list(PARTS))
    clr, _ = features.clr_frame(shares, 0.0)
    groups = pd.Series([1] * 5 + [2] * 7)
    for how in ("mean", "median"):
        rel = features.relative_clr(clr, groups, how)
        assert np.allclose(rel.sum(axis=1), 0.0)
    rel = features.relative_clr(clr, groups, "mean")
    assert np.allclose(rel.groupby(groups.to_numpy()).mean(), 0.0)  # центр группы вычтен


def test_relative_values_excludes_flagged_from_center():
    v = pd.Series([1.0, 3.0, 100.0])
    g = pd.Series([1, 1, 1])
    use = pd.Series([True, True, False])
    rel = features.relative_values(v, g, "median", use)
    assert list(rel) == pytest.approx(
        [-1.0, 1.0, 98.0]
    )  # выброс не сдвигает центр, но своё значение получает


# --- Узлы и окна ---------------------------------------------------------------------------------


def test_window_list_counts():
    names = [w[0] for w in features.window_list(["year", "half", "quarter"])]
    assert names[:2] == ["2023", "2024"] and len(names) == 2 + 4 + 8
    assert "2024Q3" in names and "2023H2" in names


def test_select_nodes_keeps_everyone_with_reason(tiny):
    nd, _ = tiny
    t = features.select_nodes(nd.nodes, 24).set_index("territory_id")
    assert not t.loc[5, "is_node"] and "12 мес." in t.loc[5, "drop_reason"]  # только 2024 год
    assert t.loc[MSK, "is_node"] and pd.isna(t.loc[MSK, "drop_reason"])
    assert len(t) == len(nd.nodes)  # никто не выпал из справочника
    assert features.select_nodes(nd.nodes, 12).set_index("territory_id").loc[5, "is_node"]


def test_window_features_complete_windows_yoy_and_groups(tiny):
    nd, groups = tiny
    w = features.window_features(nd.panel_wide, groups, params())
    w = w.set_index(["territory_id", "window"])
    # у МО 5 нет 2023 года: окон 2023 нет, окна 2024 есть, прироста к прошлому году нет
    assert (5, "2023") not in w.index and (5, "2023Q1") not in w.index and (5, "2024") in w.index
    assert pd.isna(w.loc[(5, "2024"), "mp_pp_yoy"])
    # у района 12 нет марта 2023, но Москва — узел-город с данными района 11: окно 2023Q1 есть
    assert (MSK, "2023Q1") in w.index
    d = w.loc[(1, "2024")]
    assert d["mp_pp_yoy"] == pytest.approx(100 * (d["sh_marketplace"] - w.loc[(1, "2023"), "sh_marketplace"]))
    assert np.allclose(w[[f"sh_{p}" for p in PARTS]].sum(axis=1), 1.0)
    # Москва в группе Московской области: уровень относительно среднего ln уровня МО 1, 2 и самой Москвы
    y = w.xs("2024", level="window")
    grp = y.loc[y["region_group"] == 50, "log_level"]
    assert set(grp.index) == {1, 2, MSK}
    assert y.loc[MSK, "log_level_rel"] == pytest.approx(y.loc[MSK, "log_level"] - grp.mean())


def test_place_features_relative_to_group_without_outlier(tiny):
    nd, groups = tiny
    t = features.select_nodes(nd.nodes, 12)
    ids = pd.Index(sorted(t.loc[t["is_node"], "territory_id"]), name="territory_id")
    # центр — среднее, чтобы выброс сдвинул его, если бы входил (медиана трёх значений его и так не видит)
    prm = params(place_center="mean")
    pl = features.place_features(nd.context_annual, t, ids, prm).set_index(["territory_id", "year"])
    # зарплата Москвы — выброс: центр группы 50 — среднее ln зарплаты МО 1 и 2 (по 40 000), Москва — от него
    assert pl.loc[(1, 2023), "log_wage_rel"] == pytest.approx(0.0)
    assert pl.loc[(MSK, 2023), "log_wage_rel"] == pytest.approx(np.log(87_500 / 40_000))
    assert bool(pl.loc[(MSK, 2023), "wage_outlier"])
    assert np.isnan(pl.loc[(MSK, 2023), "ndfl_income_pc"]) and not pl.loc[(MSK, 2023), "ndfl_ok"]
    assert bool(pl.loc[(3, 2023), "north"]) and not bool(pl.loc[(1, 2023), "north"])
    assert pl.loc[(MSK, 2023), "emp_sh_primary"] == pytest.approx(110 / 400)


# --- Синтетика разведки: ритм и запуск этапа -----------------------------------------------------


@pytest.fixture(scope="module")
def run_synth(tmp_path_factory):
    root = tmp_path_factory.mktemp("features")
    make_processed(root)
    cfg = make_config(root)
    features.run(cfg)
    return cfg


def test_run_writes_tables_by_contract(run_synth):
    processed = Path(run_synth["paths"]["processed"])
    tables = {name: read_table(processed / f"{name}.parquet", s) for name, s in FEATURE_SCHEMAS.items()}
    nodes_t = tables["features_nodes"]
    assert nodes_t["is_node"].sum() == tables["features_windows"]["territory_id"].nunique()
    assert (~nodes_t["is_node"]).sum() > 0 and nodes_t.loc[~nodes_t["is_node"], "drop_reason"].notna().all()
    assert set(tables["features_rhythm_monthly"]["category"].astype(str)) == set(
        run_synth["features"]["rhythm_categories"]
    )
    members = tables["features_members"]
    ter = read_table(processed / "territories.parquet", TERRITORIES)
    assert set(members["territory_id"]) == set(ter["territory_id"])  # каждое МО панели — в связи с узлом
    inner = set(ter.loc[ter["is_inner_city"], "territory_id"])
    assert set(members.loc[members["role"].astype(str) == "city_member", "territory_id"]) == inner
    # узлы-города есть и в признаках, и в рядах для рёбер
    cities = set(nodes_t.loc[nodes_t["is_city_node"], "territory_id"])
    assert cities and cities <= set(tables["features_basket_monthly"]["territory_id"])


def test_rhythm_features_find_northern_summer(run_synth):
    processed = Path(run_synth["paths"]["processed"])
    rh = read_table(processed / "features_rhythm.parquet", FEATURE_SCHEMAS["features_rhythm"])
    north, south = rh.loc[rh["north"]], rh.loc[~rh["north"]]
    assert north["own_reliable"].mean() > south["own_reliable"].mean()  # свой ритм заложен только у Севера
    assert north["own_summer"].median() > south["own_summer"].median()
    assert (rh["own_peak_month"].between(1, 12)).all()


def test_checks_json_has_story_numbers(run_synth):
    out = Path(run_synth["paths"]["outputs"]) / "features"
    checks = json.loads((out / "checks.json").read_text(encoding="utf-8"))
    story = checks["story"]
    for key in ("ami_region", "tree_kappa", "ari_years", "moved_share", "noise_share", "offline_2023"):
        assert key in story and key in checks["story_seed_range"]
    assert 0 <= story["moved_share"] <= 1 and -1 <= story["ari_years"] <= 1
    assert checks["node_mode"] == "collapse" and checks["n_nodes"] <= checks["n_nodes_all"]
    # сверочные числа конфига и черновик 26.09 записаны раздельно, у черновика — пояснение
    assert set(checks["reference_ok"]) == set(checks["reference"]) and checks["draft_26_09"]["note"]
    assert set(checks["draft_26_09"]["diff"]) == set(checks["draft_26_09"]["values"])
    ft = pd.read_csv(out / "feature_table.csv")
    assert {"coverage", "reliability", "eta2_region_group", "max_abs_rho", "collinear_08"} <= set(ft.columns)
    for name in ("types_relative", "types_fixed", "types_transitions"):
        assert (out / f"{name}.csv").exists()


def _hashes(cfg: Config) -> dict[str, str]:
    paths = sorted(Path(cfg["paths"]["processed"]).glob("features_*.parquet"))
    paths += sorted((Path(cfg["paths"]["outputs"]) / "features").glob("*"))
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def test_run_is_idempotent(run_synth):
    before = _hashes(run_synth)
    features.run(run_synth)
    assert _hashes(run_synth) == before


def test_params_reject_bad_values():
    with pytest.raises(ValueError, match="windows"):
        params(windows=["month"])
    with pytest.raises(ValueError, match="min_months"):
        params(min_months=30)
    with pytest.raises(ValueError, match="region_center"):
        params(region_center="mode")


# --- Отчёт docs/features.md ----------------------------------------------------------------------


def test_report_is_written_from_facts(run_synth):
    report = Path(run_synth["features"]["report"])
    text = report.read_text(encoding="utf-8")
    assert text.startswith("# Признаки узлов сети") and "{{" not in text and "<!--" not in text
    facts = json.loads(
        (Path(run_synth["paths"]["outputs"]) / "features" / "report_facts.json").read_text(encoding="utf-8")
    )
    assert facts["feat.n_nodes"]["text"] in text
    ft = pd.read_csv(Path(run_synth["paths"]["outputs"]) / "features" / "feature_table.csv")
    assert {"role", "moran_i", "check_ok", "check_note", "max_abs_rho_edges"} <= set(ft.columns)
    roles = dict(zip(ft["feature"], ft["role"], strict=True))
    assert (
        roles["clr_rel_food"] == "edges" and roles["own_summer"] == "layer" and roles["dec_peak"] == "outside"
    )


def _claims_true(ft: pd.DataFrame) -> pd.DataFrame:
    """Таблица признаков, где выполнены все утверждения шаблона (как на реальных данных 27.09)."""
    t = ft.set_index("feature").copy()
    t["max_abs_rho"] = 0.3
    t.loc[["summer_excess", "own_summer"], "max_abs_rho"] = 1.0
    t.loc["summer_excess", "max_abs_rho_with"] = "own_summer"
    t.loc["own_summer", "max_abs_rho_with"] = "summer_excess"
    t.loc[["clr_rel_food", "clr_rel_cafe"], "max_abs_rho"] = 0.85
    t.loc["dec_peak", "reliability"] = 0.25
    t.loc[[f"clr_rel_{q}" for q in PARTS], "eta2_region_group"] = 0.0
    t.loc[["market_access", "age_old_share", "own_summer"], "eta2_region_group"] = 0.7
    t["max_abs_rho_edges"] = np.where(t["role"] == "attributes", 0.2, np.nan)
    t.loc["log_level_rel", "max_abs_rho_edges"] = 0.73
    return t.reset_index()


def test_report_claims_are_checked(run_synth):
    from munnet import features_report

    ft = pd.read_csv(Path(run_synth["paths"]["outputs"]) / "features" / "feature_table.csv")
    checks = {"story": {"tree_kappa": 0.3, "tree_kappa_clean": 0.1, "moved_share": 0.2, "noise_share": 0.07}}
    p = params()
    good = _claims_true(ft)
    assert features_report.check_claims(good, checks, p, 0.6) == []
    bad = good.set_index("feature")
    bad.loc["dec_peak", "reliability"] = 0.9  # декабрьский пик стал надёжным — текст «это шум» неверен
    problems = features_report.check_claims(bad.reset_index(), checks, p, 0.6)
    assert problems == ["декабрьский пик ненадёжен"]
    worse = {"story": {**checks["story"], "tree_kappa": 0.05}}
    assert "каппа корзины к региону > очищенной" in features_report.check_claims(good, worse, p, 0.6)


def test_strict_report_stops_on_wrong_claims(tmp_path):
    from munnet.contracts import QCError

    make_processed(tmp_path)
    cfg = make_config(tmp_path)
    data = copy.deepcopy(cfg.data)
    data["features"]["report_strict"] = True
    cfg = Config(data=data, path=cfg.path)
    report = Path(cfg["features"]["report"])
    # синтетика не выполняет утверждений, сформулированных по реальным данным (например, доступность рынков
    # в ней не региональна): строгий режим останавливает этап и не пишет отчёт с неверным текстом
    with pytest.raises(QCError, match="утверждения шаблона"):
        features.run(cfg)
    assert not report.exists()
