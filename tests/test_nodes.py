"""Узлы сети: свёртка Москвы и Петербурга в узлы-города, режимы separate и exclude (munnet.nodes)."""

import logging

import numpy as np
import pandas as pd
import pytest
from nodes_synth import (
    CITY_RECIPIENTS,
    MO,
    city_context,
    context_annual,
    node_config,
    okved_shares,
    panel_wide,
    territories,
)

from munnet import nodes
from munnet.contracts import NODE_CONTEXT, NODE_MEMBERS, NODE_PANEL, NODES, PARTS, validate


def build(mode: str = "collapse", **over) -> nodes.NodeData:
    cfg = {**node_config(mode), **over}
    params = nodes.NodeParams.from_section(cfg, recipients_ratio_ok=(0.05, 3.0))
    return nodes.build_nodes(
        panel_wide(), territories(), context_annual(), okved_shares(), city_context(), params
    )


MSK, SPB = 90077, 90078


def test_collapse_node_ids_members_and_groups(caplog):
    with caplog.at_level(logging.WARNING, logger="munnet.nodes"):
        nd = build()
    ids = set(nd.nodes["territory_id"])
    assert ids == {1, 2, 3, 4, 5, MSK, SPB}
    for table, schema in ((nd.nodes, NODES), (nd.members, NODE_MEMBERS)):
        validate(table, schema)
    m = nd.members.set_index("territory_id")
    assert m.loc[11, "node_id"] == MSK and m.loc[12, "node_id"] == MSK and m.loc[21, "node_id"] == SPB
    assert (m.loc[[11, 12, 21], "role"].astype(str) == "city_member").all()
    assert m.loc[3, "role"] == "self" and m.loc[3, "node_id"] == 3
    n = nd.nodes.set_index("territory_id")
    # Москва — в одной группе с Московской областью; у области то же название группы
    assert n.loc[MSK, "region_group"] == 50 and n.loc[1, "region_group"] == 50
    assert n.loc[MSK, "region_group_name"] == n.loc[2, "region_group_name"] == "Москва и Московская область"
    assert n.loc[MSK, "region_code"] == 77  # свой субъект остаётся
    # Ленинградской области в мире нет: Петербург — сам себе группа, и об этом предупреждение
    assert n.loc[SPB, "region_group"] == 78
    assert "Ленинград" in caplog.text or "47" in caplog.text
    assert n.loc[MSK, "is_city_node"] and n.loc[MSK, "mo_type"] == "city" and n.loc[MSK, "n_members"] == 2
    assert not n.loc[MSK, "is_inner_city"]
    assert n.loc[MSK, "name"] == "Москва"
    assert n.loc[MSK, "oktmo_2024"] == "45000000"


def test_collapse_spending_is_population_weighted_mean_of_present_districts():
    nd = build()
    validate(nd.panel_wide, NODE_PANEL)
    w = panel_wide().set_index(["territory_id", "date"])
    city = nd.panel_wide.loc[nd.panel_wide["territory_id"] == MSK].set_index("date")
    d = "2023-05"
    expected = (1000 * w.loc[(11, d), "v_food"] + 3000 * w.loc[(12, d), "v_food"]) / 4000
    assert city.loc[d, "v_food"] == pytest.approx(expected)
    assert city.loc[d, "pop_coverage"] == pytest.approx(1.0) and city.loc[d, "n_members"] == 2
    # март 2023 года: района 12 нет — только район 11, покрытие — четверть населения города
    d = "2023-03"
    assert city.loc[d, "v_all"] == pytest.approx(w.loc[(11, d), "v_all"])
    assert city.loc[d, "pop_coverage"] == pytest.approx(0.25) and city.loc[d, "n_members"] == 1
    parts = city[[f"v_{p}" for p in PARTS]].sum(axis=1)
    assert np.allclose(parts, city["v_all"], rtol=1e-12)
    assert np.allclose(city[[f"sh_{p}" for p in PARTS]].sum(axis=1), 1.0)
    n = nd.nodes.set_index("territory_id")
    assert n.loc[MSK, "n_months"] == 24 and n.loc[MSK, "series_status"] == "full"
    # МО вне городов — как в панели, только в float
    mo3 = nd.panel_wide.loc[nd.panel_wide["territory_id"] == 3].set_index("date")
    assert np.allclose(mo3["v_all"], w.loc[3, "v_all"].astype(float))
    assert (mo3["pop_coverage"] == 1.0).all()


def test_collapse_territory_attributes_population_weighted():
    n = build().nodes.set_index("territory_id")
    t = territories().set_index("territory_id")
    assert n.loc[MSK, "market_access"] == pytest.approx(
        (1000 * t.loc[11, "market_access"] + 3000 * t.loc[12, "market_access"]) / 4000
    )
    assert n.loc[MSK, "point_lat"] == pytest.approx((1000 * 55.75 + 3000 * 55.70) / 4000)
    assert n.loc[MSK, "area_km2"] == pytest.approx(t.loc[[11, 12], "area_km2"].sum())
    assert np.isnan(n.loc[MSK, "dist_capital_km"])  # у районов расстояния нет — пропуск, не ноль


def test_collapse_context_sums_weights_and_city_row():
    nd = build()
    validate(nd.context_annual, NODE_CONTEXT)
    c = nd.context_annual.set_index(["territory_id", "year"])
    msk = c.loc[(MSK, 2023)]
    assert msk["pop_jan1"] == 4000 and msk["pop_avg"] == 4000
    assert msk["employees"] == 400
    assert msk["wage"] == pytest.approx((100 * 50_000 + 300 * 100_000) / 400)  # веса работников
    assert np.isnan(msk["retail_krub"])  # розница только у района с четвертью населения — пропуск
    assert msk["ndfl_income_rub"] == pytest.approx(4e6)  # доход — сумма районов
    assert msk["ndfl_recipients"] == CITY_RECIPIENTS  # получатели — строка «Субъект РФ», не сумма районов
    assert msk["recipients_to_pop"] == pytest.approx(CITY_RECIPIENTS / 4000)
    assert bool(msk["wage_outlier"]) and bool(msk["ndfl_outlier"])
    assert not bool(msk["ndfl_ok"]) and np.isnan(msk["ndfl_income_pc"])  # выброс: доход на жителя не считаем
    assert not bool(msk["workplace_based"])
    assert msk["payroll_pc"] == pytest.approx(msk["payroll_krub"] * 1000 / 4000 / 12)
    spb = c.loc[(SPB, 2024)]
    assert np.isnan(spb["ndfl_recipients"]) and not bool(spb["ndfl_ok"])  # строки «Свод» нет
    assert not bool(spb["wage_outlier"])
    assert c.loc[(3, 2023), "n_members"] == 1 and not bool(c.loc[(3, 2023), "wage_outlier"])


def test_collapse_okved_shares_weighted_by_employees():
    nd = build()
    s = nd.okved_shares.set_index(["territory_id", "year", "section"])["share"]
    assert s.loc[(MSK, 2023, "A")] == pytest.approx((50 + 60) / 400)
    assert s.loc[(MSK, 2023, "C")] == pytest.approx(240 / 400)
    assert s.loc[(MSK, 2023, "unallocated")] == pytest.approx(50 / 400)
    assert np.isnan(s.loc[(MSK, 2023, "B")])  # не раскрыт ни в одном районе — пропуск, а не ноль
    total = nd.okved_shares.groupby(["territory_id", "year"])["share"].sum()
    assert np.allclose(total, 1.0)
    c = nd.context_annual.set_index(["territory_id", "year"])
    assert c.loc[(MSK, 2023), "emp_sh_A"] == pytest.approx(110 / 400)
    assert c.loc[(MSK, 2023), "emp_sh_primary"] == pytest.approx(110 / 400)  # A + B, B скрыт


def test_separate_keeps_all_mo_with_own_region():
    nd = build("separate")
    assert set(nd.nodes["territory_id"]) == set(MO)
    assert (nd.nodes["region_group"] == nd.nodes["region_code"]).all()
    assert (nd.members["role"].astype(str) == "self").all()
    assert not nd.nodes["is_city_node"].any()


def test_exclude_drops_inner_city_with_reason():
    nd = build("exclude")
    assert set(nd.nodes["territory_id"]) == {1, 2, 3, 4, 5}
    m = nd.members.set_index("territory_id")
    assert (m.loc[[11, 12, 21], "role"].astype(str) == "excluded").all()
    assert m.loc[[11, 12, 21], "node_id"].isna().all()
    assert set(nd.panel_wide["territory_id"]) == {1, 2, 3, 4, 5}


def test_params_reject_unknown_mode_and_outlier():
    with pytest.raises(ValueError, match="mode"):
        nodes.NodeParams.from_section(node_config("merge"), recipients_ratio_ok=(0.05, 3.0))
    bad = node_config()
    bad["cities"][77]["outliers"] = ["retail"]
    with pytest.raises(ValueError, match="outliers"):
        nodes.NodeParams.from_section(bad, recipients_ratio_ok=(0.05, 3.0))


def test_node_long_panel_matches_wide():
    nd = build()
    long = nodes.panel_long(nd.panel_wide)
    assert len(long) == len(nd.panel_wide) * 7
    sub = long.loc[(long["territory_id"] == MSK) & (long["date"] == "2024-01")].set_index("category")["value"]
    wide = nd.panel_wide.loc[
        (nd.panel_wide["territory_id"] == MSK) & (nd.panel_wide["date"] == "2024-01")
    ].iloc[0]
    assert sub["food"] == pytest.approx(wide["v_food"])
    assert pd.api.types.is_float_dtype(long["value"])
