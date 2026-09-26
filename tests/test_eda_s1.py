"""Раздел разведки E1 «Что есть в данных»: чистые функции на малых примерах и раздел целиком на синтетике."""

import dataclasses

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import shapely
from pyproj import Transformer
from synth import make_config, make_eda_data, make_section_context

from munnet import maps, style
from munnet.eda import eda_dir, section_rng
from munnet.eda import s1_coverage as s1
from munnet.eda.base import PLACEHOLDER, SectionContext

NBSP = " "
CRS = "+proj=aea +lat_1=52 +lat_2=64 +lat_0=0 +lon_0=100 +x_0=0 +y_0=0 +ellps=WGS84 +units=m +no_defs"

# Обязательные факты раздела (spec_final, Б.4, E1).
REQUIRED_FACTS = (
    "n_mo",
    "n_full",
    "n_only_2023",
    "n_only_2024",
    "n_partial",
    "n_regions",
    "n_absent_regions",
    "absent_regions",
    "n_missing_outside",
    "n_inner_city",
    "pop_share_inner_city",
    "pop_median_full",
    "pop_median_incomplete",
    "missing_mw_p",
    "missing_auc",
    "cov_population_2023",
    "cov_population_2024",
    "cov_wage_2023",
    "cov_wage_2024",
    "cov_ndfl_income_2023",
    "cov_ndfl_income_2024",
    "controls_hard_ok",
    "controls_soft_warnings",
)


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    return make_eda_data(tmp_path_factory.mktemp("e1"))


@pytest.fixture(scope="module")
def finding(tmp_path_factory, data):
    """Раздел целиком на синтетике (один прогон на модуль тестов)."""
    ctx = make_section_context(tmp_path_factory.mktemp("e1_run"), "e1", data=data)
    with style.use():
        result = s1.run_section(ctx)
    return ctx, result


def _run(tmp_path, data, **coverage):
    cfg = make_config(tmp_path)
    if coverage:
        cfg.data["eda"]["coverage"] = coverage
    ctx = SectionContext(cfg, data, eda_dir(cfg), "e1", section_rng(cfg, 1))
    with style.use():
        return ctx, s1.run_section(ctx)


def _territories(statuses, months, n23, n24, gaps=None, regions=None):
    n = len(statuses)
    return pd.DataFrame(
        {
            "territory_id": np.arange(1, n + 1, dtype="int32"),
            "series_status": statuses,
            "n_months": months,
            "n_2023": n23,
            "n_2024": n24,
            "has_internal_gap": gaps if gaps is not None else [False] * n,
            "region_code": regions if regions is not None else [1] * n,
            "is_inner_city": [False] * n,
        }
    )


def _square(lon, lat, side_km=40):
    x, y = Transformer.from_crs("EPSG:4326", CRS, always_xy=True).transform(lon, lat)
    h = side_km * 500
    return shapely.MultiPolygon([shapely.box(x - h, y - h, x + h, y + h)])


def _geo(rows):
    """Кадр карты: (territory_id, region_code, in_panel, lon, lat[, region_name])."""
    records = [
        {
            "territory_id": r[0],
            "region_code": r[1],
            "in_panel": r[2],
            "year_from": 2018,
            "year_to": 9999,
            "region_name": r[5] if len(r) > 5 else None,
            "geometry": _square(r[3], r[4]),
        }
        for r in rows
    ]
    return gpd.GeoDataFrame(records, geometry="geometry", crs=CRS)


# --- Помощники -------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("n", "form"),
    [(1, "регион"), (2, "региона"), (4, "региона"), (5, "регионов"), (11, "регионов"), (12, "регионов")]
    + [(21, "регион"), (22, "региона"), (77, "регионов"), (111, "регионов"), (101, "регион")],
)
def test_plural(n, form):
    assert s1.plural(n, "регион", "региона", "регионов") == form


@pytest.mark.parametrize(("n", "form"), [(1, "региона"), (21, "региона"), (11, "регионов"), (77, "регионов")])
def test_genitive_plural(n, form):
    assert s1.genitive_plural(n, "региона", "регионов") == form


def test_months_word_and_month_name():
    assert s1.months_word(24) == f"24{NBSP}месяца"
    assert s1.months_word(12) == f"12{NBSP}месяцев"
    assert s1.month_name("2024-01") == "январь 2024"
    assert s1.month_name("2023-12") == "декабрь 2023"
    with pytest.raises(ValueError, match="ГГГГ-ММ"):
        s1.month_name("2024-1")


def test_join_regions_groups_oblasts_alphabetically():
    names = [
        "Курская область",
        "Краснодарский край",
        "Брянская область",
        "Севастополь",
        "Белгородская область",
    ]
    assert (
        s1.join_regions(names) == "Белгородская, Брянская и Курская области, Краснодарский край, Севастополь"
    )
    assert s1.join_regions(["Курская область", "Республика Крым"]) == "Курская область, Республика Крым"
    assert s1.join_regions(["Республика Крым"]) == "Республика Крым"
    assert s1.join_regions([]) == ""


def test_table_number_formats():
    assert s1._fmt_exact(120.48) == "120,48" and s1._fmt_exact(2190) == "2190"
    assert s1._fmt_exact(303126) == "303 126" and s1._fmt_exact(None) == "—"
    assert s1._fmt_auto(51251.6) == "51 252" and s1._fmt_auto(8.48) == "8,5" and s1._fmt_auto(21.0) == "21"


def test_fact_key_is_safe():
    assert s1.fact_key("ndfl_income") == "ndfl_income"
    assert s1.fact_key("Catering-Turnover 2") == "catering_turnover_2"


def test_coverage_params_config_overrides_defaults(tmp_path):
    cfg = make_config(tmp_path)
    assert s1.coverage_params(cfg) == s1.COVERAGE_DEFAULTS
    cfg.data["eda"]["coverage"] = {"text_regions": 5}
    params = s1.coverage_params(cfg)
    assert params["text_regions"] == 5 and params["full_share_min"] == s1.COVERAGE_DEFAULTS["full_share_min"]


# --- Ряды трат ------------------------------------------------------------------------------------


def test_series_summary_counts_statuses_and_breakdown():
    ter = _territories(
        ["full", "full", "only_2023", "only_2024", "partial", "partial", "partial"],
        [24, 24, 12, 12, 3, 12, 20],
        [12, 12, 12, 0, 3, 6, 10],
        [12, 12, 0, 12, 0, 6, 10],
        gaps=[False, False, False, False, False, True, True],
        regions=[1, 1, 2, 2, 3, 3, 3],
    )
    s = s1.series_summary(ter)
    assert (s["n_mo"], s["n_full"], s["n_only_2023"], s["n_only_2024"], s["n_partial"]) == (7, 2, 1, 1, 3)
    assert s["n_incomplete"] == 5 and s["n_internal_gap"] == 2 and s["n_regions"] == 3
    # ровно 12 месяцев — не только «ровно год»: ряд 6 + 6 месяцев тоже
    assert s["n_exactly_12"] == 3
    # «только 2023» по месяцам: и весь год, и январь–март
    assert (s["n_only_year_2023"], s["n_only_year_2024"], s["n_both_years_incomplete"]) == (2, 1, 2)


def test_status_groups_and_unknown_status():
    ter = _territories(
        ["full", "only_2023", "only_2024", "partial"], [24, 12, 12, 5], [12, 12, 0, 5], [12, 0, 12, 0]
    )
    groups = s1.status_groups(ter)
    assert groups.to_dict() == {1: s1.FULL, 2: s1.ONE_YEAR, 3: s1.ONE_YEAR, 4: s1.PARTIAL}
    ter.loc[0, "series_status"] = "strange"
    with pytest.raises(ValueError, match="strange"):
        s1.status_groups(ter)


def test_months_coverage_counts_mo_per_month():
    wide = pd.DataFrame(
        {"territory_id": [1, 2, 1, 1, 2], "date": ["2023-02", "2023-02", "2023-01", "2023-03", "2023-03"]}
    )
    out = s1.months_coverage(wide)
    assert out["date"].tolist() == ["2023-01", "2023-02", "2023-03"] and out["n_mo"].tolist() == [1, 2, 2]


def test_month_ladder_counts_and_population_share():
    mo = pd.DataFrame({"n_months": [24, 24, 18, 12, 3], "weight": [100.0, 300.0, 50.0, 50.0, np.nan]})
    ladder = s1.month_ladder(mo, [24, 12, 18]).set_index("min_months")
    assert ladder.index.tolist() == [12, 18, 24]
    assert ladder["n_mo"].tolist() == [4, 3, 2]
    assert ladder.loc[24, "pop_share"] == pytest.approx(400 / 500)
    assert ladder.loc[12, "pop_share"] == pytest.approx(1.0)
    assert ladder.loc[18, "share_mo"] == pytest.approx(3 / 5)


def test_inner_city_shares():
    mo = pd.DataFrame(
        {
            "is_inner_city": [True, True, False, False],
            "weight": [100.0, 100.0, 600.0, 200.0],
            "region_name": ["Москва", "Санкт-Петербург", "Регион", "Регион"],
        }
    )
    inner = s1.inner_city(mo)
    assert inner["n"] == 2 and inner["node_share"] == 0.5 and inner["pop_share"] == pytest.approx(0.2)
    assert inner["by_region"].to_dict() == {"Москва": 1, "Санкт-Петербург": 1}


# --- Кого нет -------------------------------------------------------------------------------------


@pytest.fixture
def frame():
    """Регион 1 в панели (центр), регион 2 без трат на юго-западе, регион 3 наполовину без трат."""
    rows = [
        (1, 1, True, 60.0, 56.0, "Центральная область"),
        (2, 1, True, 61.0, 56.0, "Центральная область"),
        (3, 2, False, 35.0, 46.0, "Южный край"),
        (4, 2, False, 36.0, 46.0, "Южный край"),
        (5, 3, True, 80.0, 60.0, "Восточная область"),
        (6, 3, False, 81.0, 60.0, "Восточная область"),
        (7, 3, False, 82.0, 60.0, "Восточная область"),
    ]
    return _geo(rows)


def test_region_names_fall_back_to_territories_and_code(frame):
    assert s1.region_names(frame).to_dict() == {
        1: "Центральная область",
        2: "Южный край",
        3: "Восточная область",
    }
    no_names = frame.drop(columns="region_name")
    ter = pd.DataFrame({"region_code": [1], "region_name": ["Центр из справочника"]})
    names = s1.region_names(no_names, ter)
    assert names.to_dict() == {1: "Центр из справочника", 2: "регион 2", 3: "регион 3"}


def test_absent_regions_and_south_west(frame):
    names = s1.region_names(frame)
    absent = s1.absent_regions(frame, names)
    assert absent["region_code"].tolist() == [2] and absent["n_mo"].tolist() == [2]
    assert absent.loc[0, "region_name"] == "Южный край"
    # точка лежит внутри одного из двух квадратов региона (35° и 36° в. д., 46° с. ш.)
    assert absent.loc[0, "lat"] == pytest.approx(46.0, abs=0.5)
    assert 34.5 < absent.loc[0, "lon"] < 36.5
    lat, lon = pd.Series([56.0, 56.0, 60.0]), pd.Series([60.0, 61.0, 80.0])
    assert s1.south_west(absent, lat, lon)
    north_east = absent.assign(lat=70.0, lon=120.0)
    assert not s1.south_west(north_east, lat, lon)
    assert not s1.south_west(absent.iloc[:0], lat, lon)


def test_absent_regions_empty_when_every_region_has_panel(frame):
    frame = frame.loc[frame["region_code"] != 2]
    assert s1.absent_regions(frame, s1.region_names(frame)).empty


def test_missing_outside_counts_only_regions_with_panel(frame):
    out = s1.missing_outside(frame, s1.region_names(frame))
    assert out.to_dict("list") == {"region_code": [3], "region_name": ["Восточная область"], "n_missing": [2]}


def test_funnel_counts_slice_spend_full(frame):
    ter = _territories(["full", "partial", "full"], [24, 5, 24], [12, 5, 12], [12, 0, 12])
    ter["territory_id"] = np.array([1, 2, 99], dtype="int32")  # МО 99 нет в срезе
    steps = s1.funnel(frame, ter, 2023)
    assert steps["n_mo"].tolist() == [7, 2, 1]
    assert steps["label"].iloc[0] == "МО в справочнике на 2023 год"


# --- Пропуски -------------------------------------------------------------------------------------


def _mo_missing(n=200, seed=0, signal=True):
    rng = np.random.default_rng(seed)
    pop = np.exp(rng.normal(10, 1, n))
    incomplete = (pop < np.quantile(pop, 0.15)) if signal else (rng.random(n) < 0.15)
    return pd.DataFrame(
        {
            "series_status": np.where(incomplete, "partial", "full"),
            "pop_2023": pop,
            "weight": pop,
            "wage_2023": np.exp(rng.normal(10.8, 0.2, n)),
            "urban_share_2023": rng.uniform(0, 1, n),
            "point_lat": rng.uniform(45, 70, n),
            "market_access": rng.uniform(100, 1000, n),
            "mo_type": pd.Categorical(
                rng.choice(["mr", "mo", "go"], n), categories=["mr", "mo", "go", "vgt"]
            ),
            "region_name": rng.choice(["А", "Б", "В"], n),
        }
    )


def test_missingness_finds_planted_small_mo_signal():
    table, auc, p = s1.missingness(_mo_missing(), seed=42)
    t = table.set_index("feature")
    assert auc > 0.9 and p < 1e-6
    assert t.loc["ln_pop", "smd"] < -1 and t.loc["ln_pop", "incomplete"] < t.loc["ln_pop", "full"]
    assert set(t.index) == {
        "ln_pop",
        "ln_wage",
        "urban",
        "lat",
        "access",
        "type_mr",
        "type_mo",
        "type_go",
        "type_vgt",
    }
    assert t.loc["urban", "full"] > 1  # доля горожан показана в процентах
    assert t.loc["type_vgt", "full"] == 0 and np.isnan(
        t.loc["type_vgt", "smd"]
    )  # нет разброса — SMD не определён


def test_missingness_on_random_gaps_is_near_chance():
    _, auc, p = s1.missingness(_mo_missing(n=600, signal=False), seed=42)
    assert 0.3 < auc < 0.7 and p > 0.01


def test_missingness_log_ignores_nonpositive():
    mo = _mo_missing()
    mo.loc[0, "pop_2023"] = 0.0
    X = s1.missingness_features(mo)
    assert np.isnan(X.loc[0, "ln_pop"]) and np.isfinite(X["ln_wage"]).all()


def test_missingness_population_is_2023_average_without_2024_fallback():
    """Сравнение населения — по среднегодовому 2023 года: МО без него (появились в 2024 году) не входят,
    даже если у них есть вес 2024 года."""
    mo = _mo_missing()
    first_inc = mo.index[mo["series_status"] == "partial"][0]
    mo.loc[first_inc, "pop_2023"] = np.nan  # вес (2024 год) остаётся
    table, _, _ = s1.missingness(mo, seed=0)
    row = table.set_index("feature").loc["ln_pop"]
    assert row["n_incomplete"] == (mo["series_status"] == "partial").sum() - 1


def test_incomplete_by_region_sorted_by_count_then_share():
    mo = pd.DataFrame(
        {
            "region_name": ["А"] * 4 + ["Б"] * 2 + ["В"] * 3,
            "series_status": [
                "partial",
                "partial",
                "full",
                "full",
                "partial",
                "only_2023",
                "full",
                "full",
                "full",
            ],
        }
    )
    out = s1.incomplete_by_region(mo)
    assert out["region_name"].tolist() == ["Б", "А"]  # 2 из 2 выше, чем 2 из 4; В без неполных не попадает
    assert out["share_incomplete"].tolist() == [1.0, 0.5] and out["n_full"].tolist() == [0, 2]


def test_missingness_table_has_labelled_rows_only():
    feats, _, _ = s1.missingness(_mo_missing(), seed=0)
    regions = s1.incomplete_by_region(_mo_missing())
    table = s1.missingness_table(feats, regions)
    assert table.columns.tolist() == ["label", "full", "incomplete", "smd", "share_incomplete"]
    assert len(table) == len(feats) + len(regions)
    assert table["label"].iloc[-1].endswith(", число МО") and table["smd"].iloc[len(feats) :].isna().all()


# --- Контекст -------------------------------------------------------------------------------------


def _context_long():
    rows = [
        # занятость: итог и разделы — считается один раз, по итогу
        (1, 2023, "employees", "TOTAL", "direct", ""),
        (1, 2023, "employees", "A", "direct", ""),
        (2, 2023, "employees", "TOTAL", "stable", ""),
        # возраст без итога: три группы одного МО — одно МО, флаги собираются по всем группам
        (1, 2023, "age", "old", "direct", ""),
        (1, 2023, "age", "working", "direct", "anomaly"),
        (1, 2023, "age", "young", "direct", ""),
        # население: перенос и запасной период с двумя флагами
        (1, 2024, "population", "TOTAL", "carried", ""),
        (2, 2024, "population", "TOTAL", "direct", "fallback_period;dup_conflict"),
        (3, 2024, "population", "TOTAL", "version", ""),
        (3, 2023, "zz_unknown", "TOTAL", "direct", ""),
    ]
    df = pd.DataFrame(rows, columns=["territory_id", "year", "indicator", "dim", "method", "flag"])
    df["indicator"] = df["indicator"].astype("category")
    df["method"] = pd.Categorical(df["method"], categories=["direct", "version", "stable", "carried"])
    return df


def _unmatched():
    return pd.DataFrame(
        {
            "territory_id": [4, 5, 5, 6],
            "year": [2024, 2024, 2024, 2023],
            "indicator": ["population", "population", "population", "wage"],
            "reason": ["region_no_rows", "no_code", "no_code", "no_code"],
        }
    )


def test_context_coverage_counts_mo_methods_flags_and_losses():
    cov = s1.context_coverage(_context_long(), _unmatched(), n_mo=10).set_index(["indicator", "year"])
    assert cov.loc[("employees", 2023), "n_mo"] == 2
    assert cov.loc[("employees", 2023), ["method_direct", "method_stable"]].tolist() == [1, 1]
    assert cov.loc[("age", 2023), "n_mo"] == 1 and cov.loc[("age", 2023), "flag_anomaly"] == 1
    pop = cov.loc[("population", 2024)]
    assert pop["n_mo"] == 3 and pop["share"] == pytest.approx(0.3)
    assert (pop["method_direct"], pop["method_version"], pop["method_carried"]) == (1, 1, 1)
    assert pop["flag_fallback_period"] == 1 and pop["flag_dup_conflict"] == 1 and pop["n_flagged"] == 1
    assert pop["lost_region_no_rows"] == 1 and pop["lost_no_code"] == 1 and pop["n_lost"] == 2
    assert pop["lost"] == "в регионе нет годовых строк — 1; нет кода — 1"
    assert pop["n_excluded"] == 0 and pop["n_used"] == pop["n_mo"]
    assert pop["flags"] == "запасной период — 1; конфликт дублей — 1"
    # показатель только с потерями — в таблице с нулём МО
    wage = cov.loc[("wage", 2023)]
    assert wage["n_mo"] == 0 and wage["n_lost"] == 1 and wage["label"] == "Средняя зарплата"


def test_context_coverage_order_and_unknown_indicator():
    cov = s1.context_coverage(_context_long(), _unmatched())
    order = cov["indicator"].drop_duplicates().tolist()
    assert order == [
        "population",
        "age",
        "wage",
        "employees",
        "zz_unknown",
    ]  # порядок INDICATORS, чужие — в конце
    unknown = cov.loc[cov["indicator"] == "zz_unknown"].iloc[0]
    assert unknown["label"] == "zz_unknown" and unknown["source"] == "" and np.isnan(unknown["share"])


def test_context_coverage_without_unmatched():
    cov = s1.context_coverage(
        _context_long(), pd.DataFrame(columns=["territory_id", "year", "indicator", "reason"])
    )
    assert not any(c.startswith("lost_") for c in cov.columns) and (cov["n_lost"] == 0).all()
    assert s1.coverage_lookup(cov)[("population", 2024)] == 3


def test_context_coverage_splits_excluded_values_from_losses():
    """Значение с флагом, которое этап panel не взял в годовую таблицу (МО есть и в context_long, и в
    unmatched), — «исключено», а не «потеряно»: n_mo + n_lost — все проверенные МО."""
    rows = [
        (1, 2023, "ndfl_recipients", "TOTAL", "direct", ""),
        (2, 2023, "ndfl_recipients", "TOTAL", "version", ""),
        (3, 2023, "ndfl_recipients", "TOTAL", "direct", "recipients_jump"),
        (5, 2023, "ndfl_recipients", "TOTAL", "direct", "union_merged"),
    ]
    cl = pd.DataFrame(rows, columns=["territory_id", "year", "indicator", "dim", "method", "flag"])
    um = pd.DataFrame(
        {
            "territory_id": [3, 4, 5],
            "year": [2023, 2023, 2023],
            "indicator": ["ndfl_recipients"] * 3,
            "reason": ["recipients_jump", "no_code", "union_merged"],
        }
    )
    row = s1.context_coverage(cl, um, n_mo=5).iloc[0]
    assert (row["n_mo"], row["n_excluded"], row["n_used"], row["n_lost"]) == (4, 2, 2, 1)
    assert row["n_mo"] + row["n_lost"] == 5
    assert row["lost"] == "нет кода — 1" and "lost_recipients_jump" not in row.index
    assert "сбой единиц получателей в источнике, исключено — 1" in row["flags"]
    assert "значение уже за объединённое МО, исключено — 1" in row["flags"]


def test_labelled_counts_sorted_by_count_without_zeros():
    counts = {"region_no_rows": 3, "union_banned": 0, "no_code": 7, "not_in_slice": 3}
    text = s1._labelled_counts(counts, s1.REASON_LABELS)
    assert text == "нет кода — 7; в регионе нет годовых строк — 3; МО нет в срезе справочника — 3"


def test_context_table_puts_context_year_first():
    cov = s1.context_coverage(_context_long(), _unmatched())
    extra = cov.iloc[[0]].assign(year=2022)
    cov = pd.concat([cov, extra, cov.iloc[[0]].assign(year=2025)], ignore_index=True)
    t = s1.context_table(cov, [2023, 2024])
    assert t.columns.tolist() == list(s1.T02_COLUMNS)
    years = t["year"].tolist()
    assert years == sorted(years, key=lambda y: {2023: 0, 2024: 1}.get(y, 2 + y))
    # внутри года — порядок показателей из context_coverage
    assert t.loc[t["year"] == 2023, "indicator"].tolist() == ["age", "wage", "employees", "zz_unknown"]


def test_regions_without_rows_counts_region_no_rows_only():
    ter = pd.DataFrame(
        {
            "territory_id": [1, 2, 3, 4, 5],
            "region_name": ["Тверская область", "Тверская область", "Рязанская область", "Б", "Б"],
        }
    )
    um = pd.DataFrame(
        {
            "territory_id": [1, 2, 3, 4, 5, 3],
            "year": [2024, 2024, 2024, 2024, 2023, 2024],
            "indicator": ["wage", "wage", "wage", "wage", "wage", "retail"],
            "reason": ["region_no_rows"] * 3 + ["no_code", "region_no_rows", "region_no_rows"],
        }
    )
    out = s1.regions_without_rows(um, ter, "wage", 2024)
    assert out.to_dict("list") == {"region_name": ["Тверская область", "Рязанская область"], "n_mo": [2, 1]}
    assert s1.regions_without_rows(um.iloc[:0], ter, "wage", 2024).empty


# --- Контрольные числа ----------------------------------------------------------------------------


CONTROLS = {
    "pop_jan1_2023": {"expected": 2188, "actual": 2180, "kind": "soft", "ok": True},
    "wage_2024": {"expected": 2037, "actual": 1900, "kind": "soft", "ok": False},
    "territories": {"expected": 2190, "actual": 2190, "kind": "hard", "ok": True},
    "regions": {"expected": 77, "actual": 76, "kind": "hard", "ok": False},
    "zz_custom": {"expected": 0, "actual": 0, "kind": "hard", "ok": True},
    "stable_union_values": {"expected": None, "actual": 62, "kind": "info", "ok": True},
    "method_wage_2023_stable": {"expected": None, "actual": 4, "kind": "info", "ok": True},
}


def test_controls_table_orders_labels_and_statuses():
    t = s1.controls_table(CONTROLS, order=["territories", "regions", "wage_2024", "pop_jan1_2023"])
    assert t["name"].tolist() == [
        "territories",
        "regions",
        "zz_custom",
        "wage_2024",
        "pop_jan1_2023",
        "method_wage_2023_stable",
        "stable_union_values",
    ]
    assert (
        t["status"].tolist()
        == ["сошлось", "ошибка", "сошлось", "предупреждение", "в допуске"] + ["справочно"] * 2
    )
    assert t["kind"].tolist() == ["жёсткое"] * 3 + ["мягкое"] * 2 + ["справочное"] * 2
    by = t.set_index("name")
    assert by.loc["regions", "deviation"] == pytest.approx(76 / 77 - 1)
    assert np.isnan(by.loc["zz_custom", "deviation"])  # ожидался ноль
    assert np.isnan(by.loc["stable_union_values", "deviation"])  # ожидания нет
    assert by.loc["wage_2024", "label"] == "Средняя зарплата, 2024: МО со значением"
    assert by.loc["pop_jan1_2023", "label"] == "Население на 1 января, 2023: МО со значением"
    assert (
        by.loc["method_wage_2023_stable", "label"] == "Средняя зарплата, 2023: МО по шагу «мост oktmo_stable»"
    )
    assert by.loc["zz_custom", "label"] == "zz_custom"


def test_context_title_shows_union_bridge_check():
    assert s1.context_title(CONTROLS).endswith("значений моста у МО с объединением в истории ОКТМО — 62")
    assert "моста у МО" not in s1.context_title({})


def test_controls_summary_and_hard_ok_text():
    c = s1.controls_summary(CONTROLS)
    assert c == {"hard_n": 3, "hard_failed": 1, "soft_n": 2, "soft_warnings": 1}
    assert s1.hard_ok_text(c) == "нет"
    assert s1.hard_ok_text({**c, "hard_failed": 0}) == "да"
    assert s1.hard_ok_text(s1.controls_summary({})) == "нет данных"
    assert s1.controls_table({}).empty


# --- Текст ----------------------------------------------------------------------------------------


def _summary(**kw):
    s = {"n_regions": 77}
    base = dict(
        n_absent=8,
        absent_has_resorts=True,
        pop_smaller=True,
        nonrandom=True,
        thresholds=[12, 18, 24],
        ref_year=2024,
        slice_year=2023,
        context_year=2023,
        has_cov={"population": True, "wage": True, "ndfl_income": True},
        n_gap_regions=5,
        gap_indicator=s1.INDICATORS["wage"],
    )
    return s1.summary_md(s, **{**base, **kw})


def test_summary_claims_follow_data():
    text = _summary()
    assert "неслучайны" in text and "Курорты" in text and "у меньших МО" in text
    assert "о {{e1.n_regions}} регионах" in text and "не короче 12, 18 и 24 месяцев" in text
    calm = _summary(absent_has_resorts=False, nonrandom=False, pop_smaller=False, n_absent=0)
    assert "неслучайны" not in calm and "Курорты" not in calm and "меньших МО" not in calm
    assert "{{e1.n_absent_regions}}" not in calm
    no_ndfl = _summary(has_cov={"population": True, "wage": True, "ndfl_income": False})
    assert "cov_ndfl_income" not in no_ndfl


def test_summary_gap_sentence_follows_data():
    text = _summary()
    assert f"у{NBSP}Росстата нет годового значения показателя «средняя зарплата»" in text
    assert "{{e1.gap_regions_2024}}" in text and f"за{NBSP}2023 год" in text
    assert "gap_regions" not in _summary(n_gap_regions=0)
    assert "gap_regions" not in _summary(ref_year=2023)  # год трат совпадает с годом контекста


def test_summary_has_no_bare_numbers_or_ascii_typography():
    text = _summary()
    assert '"' not in text and " - " not in text and "..." not in text and "  " not in text
    keys = {k for _, k in PLACEHOLDER.findall(text)}
    assert {"n_mo", "n_full", "missing_auc", "pop_share_inner_city", "n_months_ge24"} <= keys


# --- Раздел целиком -------------------------------------------------------------------------------


def test_run_section_outputs_and_required_facts(finding):
    ctx, f = finding
    assert f.section == "e1" and f.title == s1.TITLE
    assert [r.fid for r in f.figures] == ["F01", "F02"]
    assert [r.slug for r in f.figures] == ["coverage_map", "coverage_funnel"]
    assert [t.tid for t in f.tables] == ["T00", "T01", "T02"]
    missing = [k for k in REQUIRED_FACTS if f"e1.{k}" not in f.facts]
    assert not missing
    assert f.headline_errors == []
    for rec in f.figures:
        for rel in (rec.png, rec.svg, rec.data_csv):
            assert (ctx.out_dir / rel).exists()
    for t in f.tables:
        assert (ctx.out_dir / t.csv).exists() and "nan" not in t.markdown.lower()
    assert f.indicators.columns.tolist() == ["territory_id"] and f.caveats


def test_run_section_facts_match_synthetic_truth(finding, data):
    _, f = finding
    fact = {k.split(".", 1)[1]: v for k, v in f.facts.items()}
    ter = data.territories
    status = ter["series_status"].astype(str).value_counts()
    assert fact["n_mo"].value == len(ter) == 74
    assert fact["n_full"].value == status["full"] and fact["n_partial"].value == status["partial"]
    assert (
        fact["n_only_2023"].value == status["only_2023"] and fact["n_only_2024"].value == status["only_2024"]
    )
    assert fact["n_regions"].value == ter["region_code"].nunique()
    assert fact["n_inner_city"].value == 2
    # выпавший регион синтетики — один, на юго-западе; вне него МО без данных нет
    assert fact["n_absent_regions"].value == 1 and fact["absent_regions"].value == "Выпавшая область"
    assert fact["n_missing_outside"].value == 0
    # контрольные числа синтетики: жёсткие сошлись, мягкое одно вне допуска
    assert fact["controls_hard_ok"].value == "да" and fact["controls_soft_warnings"].value == 1
    assert fact["cov_population_2023"].value == 74 and fact["cov_wage_2024"].value == 74
    assert fact["mo_per_month_min"].value <= fact["mo_per_month_max"].value <= 74


def test_run_section_detects_planted_missingness_signal(finding):
    """Сигнал синтетики: неполные ряды — у самых мелких МО."""
    _, f = finding
    assert f.facts["e1.missing_auc"].value > 0.7
    assert f.facts["e1.missing_mw_p"].value < 0.01
    assert f.facts["e1.pop_median_incomplete"].value < f.facts["e1.pop_median_full"].value
    assert f.facts["e1.smd_ln_pop"].value < 0
    assert "неслучайны" in f.summary_md


def test_run_section_titles_and_tables(finding):
    ctx, f = finding
    f01, f02 = f.figures
    assert f01.title == f"Траты есть по 74{NBSP}МО 8{NBSP}регионов; 1{NBSP}регион юго-запада выпал"
    assert f02.title.startswith(f"Полный ряд — у{NBSP}68{NBSP}МО, зарплата — у{NBSP}74")
    assert "Выпавшая область" in f01.alt
    t00, t01, t02 = f.tables
    assert len(t00.markdown.splitlines()) == 2 + 3  # шапка, разделитель и все три числа синтетики
    assert (
        "Среднегодовое население 2023 года, чел., медиана" in t01.markdown
        and "Регион 1, число МО" in t01.markdown
    )
    assert "Население на 1 января" in t02.markdown and "Год" in t02.markdown
    # в отчёт идут только строки года признаков контекста (2023), остальные годы — в CSV
    md_years = {line.split(" | ")[2] for line in t02.markdown.splitlines()[2:]}
    assert md_years == {"2023"} and "здесь 2023 год" in t02.title
    csv = pd.read_csv(ctx.out_dir / t02.csv)
    assert csv.columns.tolist() == list(s1.T02_COLUMNS) and set(csv["year"]) > {2023}
    # МО из unmatched нет в годовой таблице: если значение у него всё же есть (у преемника синтетики), это
    # исключение, а не потеря; так или иначе n_mo + n_lost — все 74 МО, в годовой таблице — 73
    pop = csv.loc[(csv["indicator"] == "population") & (csv["year"] == 2023)].iloc[0]
    assert pop["n_mo"] + pop["n_lost"] == 74 and pop["n_used"] == 73


def test_run_section_map_counts_only_drawn_mo(finding, data):
    """Легенда и подзаголовок F01 — о видимых МО: упразднённых к 2024 году МО в границах карты нет."""
    ctx, f = finding
    df = pd.read_csv(ctx.out_dir / f.figures[0].data_csv)
    drawn = int(df["in_panel"].sum())
    assert drawn < len(data.territories)  # предшественник объединения синтетики в границах 2024 года не виден
    assert f"на карте {drawn} из {len(data.territories)}" in f.figures[0].subtitle.replace(NBSP, " ")


def test_run_section_gap_facts_present(finding):
    _, f = finding
    assert f.facts["e1.n_gap_regions_2024"].value == 0 and f.facts["e1.gap_regions_2024"].value == "нет"
    assert "gap_regions" not in f.summary_md


def test_run_section_map_data_marks_absent_region(finding):
    ctx, f = finding
    df = pd.read_csv(ctx.out_dir / f.figures[0].data_csv)
    absent = df.loc[~df["in_panel"]]
    assert len(absent) == 4 and (absent["status"] == s1.NO_DATA).all()
    assert set(df.loc[df["in_panel"], "status"]) <= {s1.FULL, s1.ONE_YEAR, s1.PARTIAL}


def test_run_section_summary_placeholders_are_known_facts(finding):
    _, f = finding
    keys = {f"{s}.{k}" for s, k in PLACEHOLDER.findall(f.summary_md)}
    assert keys and keys <= set(f.facts)


def test_run_section_is_deterministic(tmp_path, data, finding):
    _, first = finding
    _, second = _run(tmp_path, data)
    assert {k: v.text for k, v in first.facts.items()} == {k: v.text for k, v in second.facts.items()}


def test_headline_f01_fails_when_absent_region_is_not_south_west(tmp_path, data):
    """Проверка заголовка F01: выпавший регион к северо-востоку от МО панели — ошибка заголовка."""
    geo = maps.load_geometry(data.geo_path)
    absent = ~geo["in_panel"].astype(bool)
    geo.loc[absent, "geometry"] = geo.loc[absent, "geometry"].translate(xoff=5e6, yoff=3e6)
    path = tmp_path / "geo_ne.parquet"
    geo.to_parquet(path)
    _, f = _run(tmp_path, dataclasses.replace(data, geo_path=path))
    assert len(f.headline_errors) == 1 and "F01" in f.headline_errors[0]


def test_headline_f02_fails_below_full_share(tmp_path, data):
    _, f = _run(tmp_path, data, full_share_min=0.99)
    assert len(f.headline_errors) == 1 and "F02" in f.headline_errors[0]


def test_run_section_without_ndfl_keeps_required_facts(tmp_path, data):
    cl = data.context_long
    no_ndfl = cl.loc[~cl["indicator"].astype(str).str.startswith("ndfl")].reset_index(drop=True)
    _, f = _run(tmp_path, dataclasses.replace(data, context_long=no_ndfl))
    assert f.facts["e1.cov_ndfl_income_2023"].value == 0
    assert "5-НДФЛ" not in f.figures[1].title and "cov_ndfl_income" not in f.summary_md
    assert f.headline_errors == []
