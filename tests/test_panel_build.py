"""Панель трат, качество рядов, контрольные числа и прогон этапа panel на мини-дереве (spec_final, А)."""

import hashlib
import json
import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest
from panel_synth import (
    CATEGORY_NAMES,
    EXPECTED_HARD,
    EXPECTED_SOFT,
    MONTHS,
    consumption_frame,
    make_config,
    make_raw_tree,
)

from munnet.contracts import (
    CONTEXT_ANNUAL,
    CONTEXT_LONG,
    GEO_COLUMNS,
    GEO_OPTIONAL,
    OKVED_SHARES,
    PANEL_LONG,
    PANEL_WIDE,
    TERRITORIES,
    MissingInputError,
    QCError,
    read_table,
)
from munnet.panel import run
from munnet.panel.consumption import add_other, build_panel, read_consumption, series_quality, to_wide
from munnet.panel.controls import check_controls

CATS = make_config(".")["panel"]["categories"]


def _raw(rows):
    """Траты в формате ``read_consumption``: (territory_id, date, код, value)."""
    df = pd.DataFrame(rows, columns=["territory_id", "date", "category", "value"])
    df["territory_id"] = df["territory_id"].astype("int32")
    df["value"] = df["value"].astype("int32")
    return df


def _month(tid, date, total=1000, parts=(300, 100, 100, 50, 50)):
    codes = ["food", "marketplace", "transport", "health", "cafe"]
    return [(tid, date, "all", total), *[(tid, date, c, v) for c, v in zip(codes, parts, strict=True)]]


# --- Панель -------------------------------------------------------------------------------------


def test_read_consumption_skips_index_column_and_maps_codes(tmp_path):
    path = tmp_path / "consumption.parquet"
    consumption_frame().to_parquet(path)
    assert "__index_level_0__" in pq.read_schema(path).names  # как в источнике (Л23)
    df = read_consumption(path, CATS)
    assert list(df.columns) == ["territory_id", "date", "category", "value"]
    assert set(df["category"]) == set(CATEGORY_NAMES)
    assert df["territory_id"].dtype == np.int32 and df["value"].dtype == np.int32


def test_read_consumption_rejects_unknown_category(tmp_path):
    frame = consumption_frame().iloc[:6].copy()
    frame.loc[frame.index[0], "category"] = "Одежда"
    path = tmp_path / "c.parquet"
    frame.to_parquet(path)
    with pytest.raises(ValueError, match="Одежда"):
        read_consumption(path, CATS)


def test_other_is_all_minus_five_and_shares_sum_to_one():
    raw = _raw(_month(1, "2023-01") + _month(1, "2023-02", total=2000))
    long = add_other(raw)
    other = long[long["category"] == "other"].set_index("date")["value"]
    assert other.to_dict() == {"2023-01": 400, "2023-02": 1400}
    assert long.loc[long["category"] == "other", "is_derived"].all()
    assert not long.loc[long["category"] != "other", "is_derived"].any()
    assert list(long["category"].cat.categories) == [
        "all",
        "food",
        "marketplace",
        "transport",
        "health",
        "cafe",
        "other",
    ]
    wide = to_wide(long)
    shares = wide[[c for c in wide.columns if c.startswith("sh_")]]
    assert shares.shape[1] == 6 and np.allclose(shares.sum(axis=1), 1.0, atol=1e-12)
    assert np.allclose(wide["log_all"], np.log(wide["v_all"]))
    assert wide[["t", "year", "month"]].iloc[1].tolist() == [1, 2023, 2]


def test_nonpositive_other_raises_qc_with_place():
    raw = _raw(_month(7, "2024-03", total=600))  # сумма пяти = 600
    with pytest.raises(QCError, match=r"\(7, '2024-03', 0\)"):
        add_other(raw)


def test_incomplete_month_raises_qc():
    raw = _raw(_month(7, "2024-02") + _month(7, "2024-03")[:-1])
    with pytest.raises(QCError, match=r"не все категории: \(7, '2024-03'\)"):
        add_other(raw)


def test_series_status_and_internal_gap():
    rows = []
    for tid, months in {
        1: MONTHS,
        2: MONTHS[:12],
        3: MONTHS[12:],
        4: MONTHS[:3],  # только январь–март 2023 — partial, а не only_2023
        5: MONTHS[:4] + MONTHS[7:],
        6: MONTHS[1:13],  # ровно 12 месяцев, но не календарный год
    }.items():
        for d in months:
            rows.append((tid, d))
    q = series_quality(pd.DataFrame(rows, columns=["territory_id", "date"])).set_index("territory_id")
    assert q["series_status"].to_dict() == {
        1: "full",
        2: "only_2023",
        3: "only_2024",
        4: "partial",
        5: "partial",
        6: "partial",
    }
    assert q.loc[5, "has_internal_gap"] and q.loc[5, "longest_gap"] == 3
    assert not q.loc[[1, 2, 3, 4, 6], "has_internal_gap"].any()
    assert q.loc[5, "coverage_pattern"] == "1111" + "000" + "1" * 17
    assert q.loc[6, ["n_2023", "n_2024", "first_date", "last_date"]].tolist() == [11, 1, "2023-02", "2024-01"]


def test_build_panel_row_counts():
    raw = _raw(_month(1, "2023-01") + _month(2, "2023-01") + _month(2, "2024-12"))
    panel_long, panel_wide, quality = build_panel(raw)
    assert len(panel_long) == 3 * 7 and len(panel_wide) == 3 and len(quality) == 2
    assert panel_long["t"].max() == 23


# --- Контрольные числа --------------------------------------------------------------------------


def _ctl_cfg(tmp_path):
    cfg = make_config(tmp_path)
    cfg["panel"]["controls"]["hard"] = {"territories": 16, "months": 24}
    cfg["panel"]["controls"]["soft"] = {"pop_jan1_2023": 100}
    return cfg


def test_hard_mismatch_raises_after_writing_json(tmp_path):
    out = tmp_path / "controls.json"
    with pytest.raises(QCError, match="territories: ожидалось 16, получено 15"):
        check_controls({"territories": 15, "months": 24, "pop_jan1_2023": 100}, _ctl_cfg(tmp_path), out)
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["territories"] == {"expected": 16, "actual": 15, "kind": "hard", "ok": False}
    assert report["months"]["ok"]


def test_soft_mismatch_only_warns(tmp_path, caplog):
    out = tmp_path / "controls.json"
    with caplog.at_level(logging.WARNING):
        warnings = check_controls(
            {"territories": 16, "months": 24, "pop_jan1_2023": 98, "extra": 1.5}, _ctl_cfg(tmp_path), out
        )
    assert len(warnings) == 1 and "pop_jan1_2023" in warnings[0]
    assert "pop_jan1_2023" in caplog.text
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["pop_jan1_2023"]["kind"] == "soft" and not report["pop_jan1_2023"]["ok"]
    assert report["extra"] == {"expected": None, "actual": 1.5, "kind": "info", "ok": True}
    # В пределах допуска 1% — без предупреждения.
    assert (
        check_controls({"territories": 16, "months": 24, "pop_jan1_2023": 99.5}, _ctl_cfg(tmp_path), out)
        == []
    )


# --- Прогон этапа на мини-дереве ----------------------------------------------------------------


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = tmp_path_factory.mktemp("panel_run")
    cfg = make_raw_tree(root)
    run(cfg)
    return cfg


def _hashes(cfg):
    out = {}
    for d in (
        cfg["paths"]["processed"],
        cfg["paths"]["interim"],
        str(Path(cfg["paths"]["outputs"]) / "panel"),
    ):
        for p in sorted(Path(d).glob("*")):
            out[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def test_run_writes_all_outputs_by_contract(built):
    processed = Path(built["paths"]["processed"])
    tables = {
        "panel_long": PANEL_LONG,
        "panel_wide": PANEL_WIDE,
        "territories": TERRITORIES,
        "context_long": CONTEXT_LONG,
        "context_annual": CONTEXT_ANNUAL,
        "okved_shares": OKVED_SHARES,
    }
    got = {name: read_table(processed / f"{name}.parquet", schema) for name, schema in tables.items()}
    assert len(got["panel_long"]) == EXPECTED_HARD["panel_long_rows"]
    assert len(got["territories"]) == EXPECTED_HARD["territories"]
    assert len(got["context_annual"]) == EXPECTED_HARD["territories"] * 2
    geo = gpd.read_parquet(processed / "territories_geo.parquet")
    assert set(GEO_COLUMNS) <= set(geo.columns) and len(geo) == 23
    assert geo["in_panel"].sum() == EXPECTED_HARD["territories"]
    assert geo["territory_id"].dtype == np.int32 and (geo.geom_type == "MultiPolygon").all()
    for name in ("bdmo_rows", "ndfl_rows"):
        assert (Path(built["paths"]["interim"]) / f"{name}.parquet").exists()


def test_run_outputs_follow_consumer_contracts(built):
    """Стыки с разведкой: подписи регионов карты, формат потерь, контрольных чисел и флагов контекста."""
    processed = Path(built["paths"]["processed"])
    panel_out = Path(built["paths"]["outputs"]) / "panel"
    geo = gpd.read_parquet(processed / "territories_geo.parquet")
    assert set(GEO_OPTIONAL) <= set(geo.columns) and geo["region_name"].notna().all()
    names = geo.set_index("territory_id")["region_name"]
    assert names[22] == "Рязанская область" and not geo.set_index("territory_id").loc[22, "in_panel"]
    ter = read_table(processed / "territories.parquet", TERRITORIES)
    assert list(ter["mo_type"].cat.categories) == ["mr", "mo", "go", "vgt"]
    assert ter["successor_id"].dtype == "Int32" and ter["point_lon"].notna().all()
    un = pd.read_csv(panel_out / "unmatched.csv", dtype={"oktmo": str})
    assert list(un.columns[:4]) == ["territory_id", "year", "indicator", "reason"]
    assert un["reason"].notna().all()
    report = json.loads((panel_out / "controls.json").read_text(encoding="utf-8"))
    assert all(set(v) == {"expected", "actual", "kind", "ok"} for v in report.values())
    assert {v["kind"] for v in report.values()} == {"hard", "soft", "info"}
    cl = pd.read_parquet(processed / "context_long.parquet", columns=["oktmo_used", "flag"])
    assert cl["oktmo_used"].str.fullmatch(r"\d{8}").all()
    assert cl["flag"].notna().all() and cl["flag"].str.fullmatch(r"([a-z_]+(;[a-z_]+)*)?").all()
    assert (cl["flag"] == "").any() and (cl["flag"] != "").any()


def test_run_controls_match_synthetic_world(built):
    report = json.loads(
        (Path(built["paths"]["outputs"]) / "panel" / "controls.json").read_text(encoding="utf-8")
    )
    for name in (*EXPECTED_HARD, *EXPECTED_SOFT):
        assert report[name]["ok"], (name, report[name])
    assert report["stable_union_values"]["kind"] == "info"


def test_run_reference_cases(built):
    processed = Path(built["paths"]["processed"])
    ter = read_table(processed / "territories.parquet", TERRITORIES).set_index("territory_id")
    ann = read_table(processed / "context_annual.parquet", CONTEXT_ANNUAL).set_index(["territory_id", "year"])
    cl = read_table(processed / "context_long.parquet", CONTEXT_LONG)
    pop = cl[cl["indicator"] == "population"].set_index(["territory_id", "year"])
    assert pop.loc[(3, 2023), "method"] == "version"  # Магадан (Л4)
    assert pop.loc[(7, 2024), "value"] == 937364  # Саратов: прямой код, части не суммируются (Л10)
    assert pop.loc[(6, 2024), "value"] == 122210 and (4, 2024) not in pop.index  # Павлово-Посадский (Л8)
    assert pop.loc[(8, 2024), "method"] == "stable"
    assert pop.loc[(11, 2023), "method"] == "carried" and pop.loc[(11, 2023), "flag"] == "zero_replaced"
    assert pop.loc[(10, 2024), "method"] == "carried"  # Рязань (Л6)
    assert pop.loc[(12, 2023), "flag"] == "fallback_period"  # Кемерово (Л5)
    assert ann.loc[(9, 2024), "pop_avg_method"] == "jan1_only"  # 2025: multi_row
    assert (
        ter.loc[13, "is_inner_city"]
        and np.isnan(ter.loc[13, "center_lat"])
        and not np.isnan(ter.loc[13, "point_lat"])
    )
    assert ann.loc[(13, 2023), "workplace_based"] and np.isnan(ann.loc[(13, 2023), "retail_pc"])
    assert ter.loc[4, "lineage_role"] == "union_predecessor" and ter.loc[4, "successor_id"] == 6
    assert ter.loc[10, "dist_capital_km"] == 25.0 and ter.loc[1, "dist_capital_km"] == 0.0
    assert ann.loc[(1, 2024), "orgs"] == 3000  # запасной снимок «На 1 июля»
    # Центр субъекта — городской округ 1; район 2 помечен центром в справочнике, но столицей не считается.
    assert ter.loc[1, "is_capital"] and ter.loc[1, "mo_status"] == "capital"
    assert not ter.loc[2, "is_capital"] and pd.isna(ter.loc[2, "mo_status"])
    assert ter.loc[2, "dist_capital_km"] == 30.0
    # Как Сасовский: у преемника 23 в 2024 году строк нет и переносить нечего; роли объединения.
    assert (23, 2024) not in pop.index and np.isnan(ann.loc[(23, 2024), "pop_jan1"])
    assert ter.loc[21, "lineage_role"] == "union_predecessor" and ter.loc[21, "successor_id"] == 23
    assert ter.loc[23, "lineage_role"] == "union_successor"
    # 5-НДФЛ: новый код МО 11 найден мостом по паре из строк 2016 года; объединение под кодом части 4.
    ndfl = cl[cl["indicator"] == "ndfl_income"].set_index(["territory_id", "year"])
    assert ndfl.loc[(11, 2023), ["method", "oktmo_used"]].tolist() == ["stable", "42635000"]
    assert ndfl.loc[(4, 2023), "flag"] == "union_merged" and np.isnan(ann.loc[(4, 2023), "ndfl_income_rub"])


def test_run_unmatched_lists_every_panel_mo_without_population(built):
    processed = Path(built["paths"]["processed"])
    un = pd.read_csv(Path(built["paths"]["outputs"]) / "panel" / "unmatched.csv", dtype={"oktmo": str})
    ann = read_table(processed / "context_annual.parquet", CONTEXT_ANNUAL)
    missing = ann.loc[ann["pop_jan1"].isna(), ["territory_id", "year"]]
    pop_un = un[un["indicator"] == "population"]
    listed = set(map(tuple, pop_un[["territory_id", "year"]].to_numpy()))
    assert set(map(tuple, missing.to_numpy())) <= listed
    reasons = pop_un.set_index(["territory_id", "year"])["reason"]
    assert reasons.loc[(6, 2023)] == "not_in_slice"
    assert reasons.loc[(9, 2025)] == "multi_row" and reasons.loc[(10, 2025)] == "region_no_rows"
    assert reasons.loc[(23, 2024)] == "region_no_rows" and reasons.loc[(21, 2024)] == "not_in_slice"
    assert un["oktmo"].dropna().str.fullmatch(r"\d{8}").all()
    ndfl = un[un["indicator"] == "ndfl_income"].set_index(["territory_id", "year"])["reason"]
    assert ndfl.loc[(4, 2023)] == "union_merged" and ndfl.loc[(5, 2023)] == "union_banned"


def test_run_outputs_feed_eda_loader(built):
    from munnet.eda.data import load

    data = load(built)
    assert len(data.mo) == EXPECTED_HARD["territories"]
    assert data.controls["territories"]["ok"]


def test_run_is_idempotent(built):
    before = _hashes(built)
    run(built)
    assert _hashes(built) == before


def test_run_without_raw_data_reports_missing_inputs(tmp_path):
    with pytest.raises(MissingInputError, match="сначала запустите этап data"):
        run(make_config(tmp_path))
