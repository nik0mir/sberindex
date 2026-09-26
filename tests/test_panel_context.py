"""Контекст Росстата и ФНС: чтение, отбор, население, доли, «на жителя», 5-НДФЛ (spec_final, А.7–А.10)."""

import numpy as np
import pandas as pd
import pytest
from panel_synth import (
    JAN1,
    JAN_DEC,
    LOW,
    TOTAL_OKVED,
    YEAR_VALUE,
    age_rows,
    brow,
    dictionary_frame,
    make_config,
    ndfl_frame,
    write_bdmo,
)

from munnet.panel.bdmo import (
    age_spec,
    check_age,
    okved_section,
    read_bdmo,
    read_bdmo_with_codes,
    select,
    stable_pairs,
)
from munnet.panel.context import build_annual, flag_recipient_jumps, population
from munnet.panel.ndfl import read_ndfl
from munnet.panel.territory import read_dictionary

CFG = make_config(".")
CX = CFG["context"]
YEARS = CX["read_years"]


def _read(tmp_path, code, rows):
    path = write_bdmo(tmp_path / f"{code}.csv.gz", code, rows)
    return read_bdmo(path, CX["level"], CX["na_codes"], YEARS)


def _spec(**kw):
    base = {"dims": {"okved2": TOTAL_OKVED}, "periods": [JAN_DEC, YEAR_VALUE], "years": [2023, 2024]}
    return {**base, **kw}


# --- Чтение БД ПМО ------------------------------------------------------------------------------


def test_read_bdmo_keeps_leading_zero_top_level_and_na_codes(tmp_path):
    rows = [
        brow("01512000", 2023, 12979, JAN1, mest="Все население"),
        brow("01512000", 2023, 5, JAN1, mest="Все население", level=LOW),
        brow("01513000", 2023, "", JAN1, mest="CD", stable="ND"),
        brow("01514000", 2019, 1, JAN1, mest="Все население"),
    ]
    df = _read(tmp_path, "Y48112027", rows)
    assert len(df) == 2  # нижний уровень и год вне read_years отброшены
    assert set(df["oktmo"]) == {"01512000", "01513000"}
    empty = df[df["oktmo"] == "01513000"].iloc[0]
    assert pd.isna(empty["mest"]) and pd.isna(empty["oktmo_stable"]) and np.isnan(empty["value"])
    assert pd.isna(df["comment"]).all()  # «CD» в комментарии — тоже пропуск
    assert df["year"].dtype == np.int16 and df["value"].dtype == np.float64


def test_bdmo_codes_are_collected_before_year_filter(tmp_path):
    rows = [
        brow("01512000", 2023, 12979, JAN1, mest="Все население"),
        brow("01514000", 2019, 1, JAN1, mest="Все население", stable="01515000"),  # год вне read_years
        brow("01516000", 2019, 1, JAN1, mest="Все население", level=LOW, stable="01517000"),
    ]
    path = write_bdmo(tmp_path / "Y48112027.csv.gz", "Y48112027", rows)
    df, codes = read_bdmo_with_codes(path, CX["level"], CX["na_codes"], YEARS)
    assert set(df["oktmo"]) == {"01512000"}
    got = set(map(tuple, codes[["oktmo", "oktmo_stable", "year"]].astype(str).to_numpy()))
    assert got == {("01512000", "01512000", "2023"), ("01514000", "01515000", "2019")}
    pairs = stable_pairs([codes])
    assert ("01514000", "01515000") in set(map(tuple, pairs.to_numpy()))


# --- Периоды, разрезы, дедупликация -------------------------------------------------------------


def test_select_periods_annual_fallback_and_never_quarters(tmp_path):
    rows = [
        brow("41701000", 2023, 100, JAN_DEC, okved2=TOTAL_OKVED),
        brow("41701000", 2023, 999, YEAR_VALUE, okved2=TOTAL_OKVED),
        brow("42640000", 2023, 80, YEAR_VALUE, okved2=TOTAL_OKVED),
        brow("41610000", 2023, 70, "Январь-сентябрь", okved2=TOTAL_OKVED),
        brow("41610000", 2023, 9, "Декабрь", okved2=TOTAL_OKVED),
    ]
    out = select(_read(tmp_path, "Y48401003", rows), _spec(), CX).set_index("oktmo")
    assert out.loc["41701000", "value"] == 100 and pd.isna(out.loc["41701000", "flag"])
    assert out.loc["42640000", "value"] == 80 and out.loc["42640000", "flag"] == "fallback_period"
    assert "41610000" not in out.index  # нарастающий итог и один месяц — никогда (Л12, Л13)


def test_select_two_totals_takes_surveyed(tmp_path):
    rows = [
        brow("41701000", 2023, 999, JAN_DEC, okved2="Всего по отдельным видам экономической деятельности"),
        brow("41701000", 2023, 100, JAN_DEC, okved2=TOTAL_OKVED),
    ]
    out = select(_read(tmp_path, "Y48401003", rows), _spec(), CX)
    assert out["value"].tolist() == [100.0]


def test_okved_sections_cyrillic_letters_and_totals():
    total, letters = CX["okved_total"], CX["okved_letters"]
    assert okved_section(TOTAL_OKVED, total, letters) == "TOTAL"
    assert okved_section("Всего по отдельным видам экономической деятельности", total, letters) is None
    for cyr, lat in (("А", "A"), ("В", "B"), ("Е", "E"), ("Н", "H")):
        assert okved_section(f"Раздел {cyr} Что-то", total, letters) == lat
    assert okved_section("Раздел C Обрабатывающие производства", total, letters) == "C"
    assert okved_section("Торговля розничная в неспециализированных магазинах", total, letters) is None


def test_dedup_after_filter_mun_type_priority_and_conflict(tmp_path):
    rows = [
        # МО 1: одинаковые значения схлопываются; строка района с тем же ОКТМО уступает городу.
        brow("41701000", 2023, 300, JAN1, mest="Все население"),
        brow("41701000", 2023, 300, JAN1, mest="Все население"),
        brow("41701000", 2023, 111, JAN1, mest="Все население", mun_type="Внутригородской район"),
        # МО 2: два равноправных значения — первое и флаг dup_conflict.
        brow("41610000", 2023, 50, JAN1, mest="Все население"),
        brow("41610000", 2023, 55, JAN1, mest="Все население"),
        # Конфликт в разрезе, который отфильтрован, не мешает.
        brow("41620000", 2023, 20, JAN1, mest="Все население"),
        brow("41620000", 2023, 1, JAN1, mest="Городское население"),
        brow("41620000", 2023, 2, JAN1, mest="Городское население"),
    ]
    spec = {"dims": {"mest": "Все население"}, "periods": [JAN1, YEAR_VALUE], "years": [2023]}
    out = select(_read(tmp_path, "Y48112027", rows), spec, CX).set_index("oktmo")
    assert out.loc["41701000", "value"] == 300 and pd.isna(out.loc["41701000", "flag"])
    assert out.loc["41610000", "value"] == 50 and out.loc["41610000", "flag"] == "dup_conflict"
    assert out.loc["41620000", "value"] == 20 and pd.isna(out.loc["41620000", "flag"])


def test_anomaly_comment_is_flag_and_value_kept(tmp_path):
    rows = [
        brow("41701000", 2023, 45000, JAN_DEC, okved2=TOTAL_OKVED, comment="Аномальное значение показателя")
    ]
    out = select(_read(tmp_path, "Y48423007", rows), _spec(periods=[JAN_DEC]), CX)
    assert out["value"].tolist() == [45000.0] and out["flag"].tolist() == ["anomaly"]


def test_age_groups_ignore_u2012_intervals_and_check_sum(tmp_path):
    frame = _read(tmp_path, "Y48112014_2023", age_rows(2023))
    rows = select(frame, age_spec(CX, 2023), CX)
    assert set(rows["dim"]) == {"TOTAL", "young", "working", "old"}  # «0‒4» и мужчины не попали
    checked = check_age(rows, CX["age"]["groups"].keys(), 0.01)
    assert set(checked["dim"]) == {"young", "working", "old"}
    flags = checked.groupby("oktmo")["flag"].first()
    assert pd.isna(flags["41701000"]) and flags["41610000"] == "age_sum_mismatch"


def test_stable_pairs_drop_codes_with_two_stable_codes():
    a = pd.DataFrame({"oktmo": ["1", "2", "3"], "oktmo_stable": ["1", "9", "3"]})
    b = pd.DataFrame({"oktmo": ["2", "3"], "oktmo_stable": ["8", "3"]})
    pairs = stable_pairs([a, b])
    assert pairs.to_dict("list") == {"oktmo": ["1", "3"], "oktmo_stable": ["1", "3"]}


# --- Население ----------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def dic(tmp_path_factory):
    path = tmp_path_factory.mktemp("dict") / "dict.xlsx"
    dictionary_frame().to_excel(path, index=False)
    return read_dictionary(path)


def _matched(rows):
    df = pd.DataFrame(
        rows,
        columns=["territory_id", "year", "value", "method", "flag", "reason", "oktmo_used", "period_used"],
    )
    df["territory_id"] = df["territory_id"].astype("int32")
    df["year"] = df["year"].astype("int16")
    df["dim"] = np.where(df["value"].isna(), None, "TOTAL")
    for col in ("method", "flag", "reason", "oktmo_used", "period_used", "dim"):
        df[col] = df[col].astype("str")
    return df


def test_population_zero_to_na_then_carried_only_for_slice(dic):
    m = _matched(
        [
            (11, 2022, 11882.0, "direct", None, None, "42630000", JAN1),
            (11, 2023, 0.0, "direct", None, None, "42630000", JAN1),  # Краснинский: ноль (Л7)
            (10, 2023, 25000.0, "direct", None, None, "61710000", JAN1),
            (10, 2024, np.nan, None, None, "region_no_rows", None, None),  # Рязань 2024 (Л6)
            (4, 2023, 91858.0, "direct", None, None, "41630000", JAN1),  # в 2024 году МО 4 нет в срезе
            (12, 2023, 40100.0, "direct", "fallback_period", None, "42640000", YEAR_VALUE),
            (12, 2024, np.nan, None, None, "no_code", None, None),
        ]
    )
    pop = population(m, dic, 2024, carry_years=[2023, 2024]).set_index(["territory_id", "year"])
    k = pop.loc[(11, 2023)]
    assert k["value"] == 11882.0 and k["method"] == "carried" and k["flag"] == "zero_replaced"
    r = pop.loc[(10, 2024)]
    assert r["value"] == 25000.0 and r["method"] == "carried" and r["oktmo_used"] == "61710000"
    assert (4, 2024) not in pop.index  # перенос только для МО среза
    kem = pop.loc[(12, 2024)]
    assert kem["method"] == "carried" and kem["flag"] == "fallback_period"  # происхождение сохраняется


def test_population_no_carry_outside_carry_years(dic):
    m = _matched(
        [
            (10, 2024, 25000.0, "direct", None, None, "61710000", JAN1),
            (10, 2025, np.nan, None, None, "region_no_rows", None, None),
        ]
    )
    pop = population(m, dic, 2024, carry_years=[2023, 2024]).set_index(["territory_id", "year"])
    assert np.isnan(pop.loc[(10, 2025), "value"])  # 1 января 2025 года — только наблюдение


# --- Годовая таблица ----------------------------------------------------------------------------


def _long(rows):
    df = pd.DataFrame(rows, columns=["territory_id", "year", "indicator", "dim", "value", "method"])
    df["territory_id"] = df["territory_id"].astype("int32")
    df["year"] = df["year"].astype("int16")
    for col in ("unit", "source_code", "period_used", "oktmo_used", "flag"):
        df[col] = pd.Series(np.nan, index=df.index, dtype="str")
    return df


def _territories(ids, inner=()):
    return pd.DataFrame(
        {"territory_id": np.array(ids, dtype="int32"), "is_inner_city": [i in inner for i in ids]}
    )


def test_per_capita_units_and_pop_avg():
    rows = [
        (1, 2023, "population", "TOTAL", 100.0, "direct"),
        (1, 2024, "population", "TOTAL", 100.0, "direct"),
        (1, 2023, "payroll", "TOTAL", 1200.0, "direct"),  # 1200 тыс. ₽ в год при 100 жителях
        (1, 2023, "ndfl_income", "TOTAL", 1_200_000.0, "direct"),
        (1, 2023, "ndfl_recipients", "TOTAL", 40.0, "direct"),
        (1, 2023, "orgs", "TOTAL", 5.0, "direct"),
        (1, 2023, "nights", "TOTAL", 300.0, "direct"),
        (2, 2024, "population", "TOTAL", 50.0, "direct"),
        (2, 2025, "population", "TOTAL", 70.0, "direct"),
        (3, 2024, "population", "TOTAL", 80.0, "carried"),
        (3, 2023, "population", "TOTAL", 80.0, "direct"),
    ]
    ann, _ = build_annual(_long(rows), _territories([1, 2, 3]), CFG)
    a = ann.set_index(["territory_id", "year"])
    assert a.loc[(1, 2023), "payroll_pc"] == pytest.approx(1000.0)  # ₽ в месяц на жителя
    assert a.loc[(1, 2023), "ndfl_income_pc"] == pytest.approx(1000.0)
    assert a.loc[(1, 2023), "ndfl_income_per_recipient"] == pytest.approx(2500.0)
    assert a.loc[(1, 2023), "recipients_to_pop"] == pytest.approx(0.4) and a.loc[(1, 2023), "ndfl_ok"]
    assert a.loc[(1, 2023), "orgs_per_1000"] == pytest.approx(50.0)
    assert a.loc[(1, 2023), "nights_pc"] == pytest.approx(3.0)
    assert a.loc[(2, 2024), "pop_avg"] == 60.0 and a.loc[(2, 2024), "pop_avg_method"] == "mean_jan1"
    assert a.loc[(3, 2023), "pop_avg"] == 80.0 and a.loc[(3, 2023), "pop_avg_method"] == "jan1_only"
    assert a.loc[(3, 2024), "pop_jan1_method"] == "carried"
    assert pd.isna(a.loc[(2, 2023), "pop_avg_method"])


def test_workplace_based_and_ndfl_ok_blank_per_capita():
    rows = [
        (13, 2023, "population", "TOTAL", 1000.0, "direct"),
        (13, 2023, "payroll", "TOTAL", 1200.0, "direct"),
        (13, 2023, "employees", "TOTAL", 900.0, "direct"),
        (13, 2023, "ndfl_income", "TOTAL", 1e9, "direct"),
        (13, 2023, "ndfl_recipients", "TOTAL", 500.0, "direct"),
        (2, 2023, "population", "TOTAL", 1000.0, "direct"),
        (2, 2023, "ndfl_income", "TOTAL", 1e7, "direct"),
        (2, 2023, "ndfl_recipients", "TOTAL", 10.0, "direct"),  # 1% жителей: вне интервала
    ]
    ann, _ = build_annual(_long(rows), _territories([2, 13], inner=(13,)), CFG)
    a = ann.set_index(["territory_id", "year"])
    city = a.loc[(13, 2023)]
    assert city["workplace_based"] and not city["ndfl_ok"]
    assert (
        np.isnan(city["payroll_pc"])
        and np.isnan(city["ndfl_income_pc"])
        and np.isnan(city["employees_to_working_age"])
    )
    assert city["employees"] == 900.0  # сам показатель остаётся
    low = a.loc[(2, 2023)]
    assert not low["ndfl_ok"] and np.isnan(low["ndfl_income_pc"])
    # Получателей меньше 5% жителей — так выглядит и дефект источника: доход на получателя не считаем.
    assert np.isnan(low["ndfl_income_per_recipient"]) and low["recipients_to_pop"] == pytest.approx(0.01)
    assert city["ndfl_income_per_recipient"] == pytest.approx(1e9 / 500 / 12)  # по месту работы — считаем


def _ndfl_matched(values):
    """Выход match_rows для 5-НДФЛ: (territory_id, year, value)."""
    df = pd.DataFrame(values, columns=["territory_id", "year", "value"])
    df["territory_id"] = df["territory_id"].astype("int32")
    df["year"] = df["year"].astype("int16")
    df["dim"] = pd.Series("TOTAL", index=df.index, dtype="str")
    df["flag"] = pd.Series(np.nan, index=df.index, dtype="str")
    return df


def test_recipient_jump_flags_only_units_defect():
    income = _ndfl_matched(
        [
            (1, 2023, 2.86e10), (1, 2024, 3.78e10),  # Тосненский: 5 получателей в 2023 году
            (2, 2023, 1.19e10), (2, 2024, 1.59e10),  # Кольцово: сотни тысяч получателей в 2024 году
            (3, 2023, 1.0e9), (3, 2024, 9.0e9),  # доход вырос в 9 раз — не дефект единиц
            (4, 2022, 4.0e9), (4, 2023, 4.4e9), (4, 2024, 4.8e9),  # обычный ряд
        ]
    )  # fmt: skip
    recipients = _ndfl_matched(
        [
            (1, 2023, 5.0), (1, 2024, 66705.0),
            (2, 2023, 14937.0), (2, 2024, 628748.0),
            (3, 2023, 100.0), (3, 2024, 9000.0),
            (4, 2022, 10000.0), (4, 2023, 10500.0), (4, 2024, 11000.0),
        ]
    )  # fmt: skip
    out = flag_recipient_jumps(income, recipients).set_index(["territory_id", "year"])["flag"]
    assert out.loc[(1, 2023)] == "recipients_jump"
    assert out.drop((1, 2023)).isna().all()


def test_flagged_values_are_excluded_from_annual():
    rows = [
        (1, 2023, "population", "TOTAL", 1000.0, "direct"),
        (1, 2023, "ndfl_income", "TOTAL", 1.2e7, "direct"),
        (1, 2023, "ndfl_recipients", "TOTAL", 5.0, "direct"),
        (2, 2023, "population", "TOTAL", 1000.0, "direct"),
        (2, 2023, "ndfl_income", "TOTAL", 1.2e7, "direct"),
        (2, 2023, "ndfl_recipients", "TOTAL", 400.0, "direct"),
    ]
    long = _long(rows)
    long.loc[(long["territory_id"] == 1) & (long["indicator"] == "ndfl_recipients"), "flag"] = (
        "recipients_jump"
    )
    long.loc[(long["territory_id"] == 2) & (long["indicator"] != "population"), "flag"] = "union_merged"
    a = build_annual(long, _territories([1, 2]), CFG)[0].set_index(["territory_id", "year"])
    one, two = a.loc[(1, 2023)], a.loc[(2, 2023)]
    assert one["ndfl_income_rub"] == 1.2e7 and np.isnan(one["ndfl_recipients"])
    assert np.isnan(one["recipients_to_pop"]) and np.isnan(one["ndfl_income_per_recipient"])
    assert np.isnan(two["ndfl_income_rub"]) and np.isnan(two["ndfl_income_pc"]) and not two["ndfl_ok"]


def test_ndfl_ok_from_neighbour_year_when_recipients_are_defect():
    rows = []
    for tid, rec_2024 in ((1, 400.0), (2, 10.0), (3, 400.0), (4, 400.0)):
        rows += [
            (tid, 2023, "population", "TOTAL", 1000.0, "direct"),
            (tid, 2024, "population", "TOTAL", 1000.0, "direct"),
            (tid, 2023, "ndfl_income", "TOTAL", 1.2e7, "direct"),
            (tid, 2024, "ndfl_income", "TOTAL", 1.3e7, "direct"),
            (tid, 2023, "ndfl_recipients", "TOTAL", 5.0, "direct"),
            (tid, 2024, "ndfl_recipients", "TOTAL", rec_2024, "direct"),
        ]
    long = _long(rows)
    jump = (long["indicator"] == "ndfl_recipients") & (long["year"] == 2023) & (long["territory_id"] != 4)
    long.loc[jump, "flag"] = "recipients_jump"  # у МО 4 пять получателей без флага — учёт по месту работы
    a = build_annual(long, _territories([1, 2, 3, 4], inner=(3,)), CFG)[0].set_index(["territory_id", "year"])
    one = a.loc[(1, 2023)]  # 2024 год: 40% жителей — в интервале
    assert one["ndfl_ok"] and one["ndfl_ok_neighbour"]
    assert one["ndfl_income_pc"] == pytest.approx(1000.0)
    assert np.isnan(one["recipients_to_pop"]) and np.isnan(one["ndfl_income_per_recipient"])
    assert a.loc[(1, 2024), "ndfl_ok"] and not a.loc[(1, 2024), "ndfl_ok_neighbour"]
    two = a.loc[(2, 2023)]  # сосед вне интервала (1% жителей)
    assert not two["ndfl_ok"] and not two["ndfl_ok_neighbour"] and np.isnan(two["ndfl_income_pc"])
    three = a.loc[(3, 2023)]  # внутригородская территория: по месту работы не считаем
    assert not three["ndfl_ok"] and not three["ndfl_ok_neighbour"]
    four = a.loc[(4, 2023)]  # без дефекта своё отношение 0,5% решает, сосед не смотрится
    assert not four["ndfl_ok"] and not four["ndfl_ok_neighbour"]
    assert four["recipients_to_pop"] == pytest.approx(0.005)


def test_employment_shares_close_to_one_with_unallocated():
    rows = [
        (1, 2023, "employees", "TOTAL", 1000.0, "direct"),
        (1, 2023, "employees", "A", 100.0, "direct"),
        (1, 2023, "employees", "B", 50.0, "direct"),
        (1, 2023, "employees", "C", 200.0, "direct"),
        (1, 2023, "employees", "P", 250.0, "direct"),
        (2, 2023, "employees", "TOTAL", 100.0, "direct"),
        (2, 2023, "employees", "A", 60.0, "direct"),
        (2, 2023, "employees", "C", 50.0, "direct"),  # сумма 110% — нормировка
        (3, 2023, "employees", "TOTAL", 1000.0, "direct"),
        (3, 2023, "employees", "P", 1005.0, "direct"),  # перебор 0,5% — в пределах допуска
    ]
    ann, shares = build_annual(_long(rows), _territories([1, 2, 3]), CFG)
    a = ann.set_index(["territory_id", "year"])
    one = a.loc[(1, 2023)]
    assert one["emp_sh_A"] == pytest.approx(0.1) and one["emp_sh_unallocated"] == pytest.approx(0.4)
    assert one["emp_sh_primary"] == pytest.approx(0.15) and one["emp_sh_public"] == pytest.approx(0.25)
    assert one["emp_n_disclosed"] == 4 and np.isnan(one["emp_sh_D"])
    two = a.loc[(2, 2023)]
    assert two["emp_sh_renormalized"] and two["emp_sh_A"] == pytest.approx(60 / 110)
    three = a.loc[(3, 2023)]  # перебор до 1%: тоже нормировка и флаг (отступление от А.4)
    assert three["emp_sh_renormalized"] and three["emp_sh_public"] == pytest.approx(1.0)
    assert three["emp_sh_unallocated"] == 0.0 and not one["emp_sh_renormalized"]
    sums = shares.groupby(["territory_id", "year"])["share"].sum()
    assert np.allclose(sums, 1.0)
    assert set(shares["section"]) == {*"ABCDEFGHIJKLMNOPQRSTU", "unallocated"}
    assert np.isnan(shares.set_index(["territory_id", "section"]).loc[(1, "D"), "employees"])


def test_urban_share_from_rural_when_urban_missing():
    rows = [
        (1, 2023, "population", "TOTAL", 100.0, "direct"),
        (1, 2023, "population_rural", "TOTAL", 40.0, "direct"),
        (2, 2023, "population", "TOTAL", 100.0, "direct"),
        (2, 2023, "population_urban", "TOTAL", 70.0, "direct"),
        (2, 2023, "population_rural", "TOTAL", 30.0, "direct"),
        (3, 2023, "population", "TOTAL", 100.0, "direct"),
    ]
    ann, _ = build_annual(_long(rows), _territories([1, 2, 3]), CFG)
    a = ann.set_index(["territory_id", "year"])["urban_share"]
    assert a.loc[(1, 2023)] == pytest.approx(0.6)
    assert a.loc[(2, 2023)] == pytest.approx(0.7)
    assert np.isnan(a.loc[(3, 2023)])


# --- 5-НДФЛ -------------------------------------------------------------------------------------


def test_read_ndfl_filters_indicator_rate_authority_and_zero(tmp_path):
    path = tmp_path / "ndfl.parquet"
    ndfl_frame().to_parquet(path, index=False)
    out = read_ndfl(path, CX["ndfl"], CX["na_codes"])
    inc = out["ndfl_income"].set_index(["oktmo", "year"])["value"]
    assert inc.loc[("41701000", 2023)] == 1.2e9  # Y777000006, ставка «Всего», своя инспекция, верхний уровень
    assert ("41610000", 2023) not in inc.index  # ноль дохода -> пропуск
    assert set(out["ndfl_income"]["source_code"]) == {"Y777000006"}  # Y777000030 игнорируется
    rec = out["ndfl_recipients"].set_index(["oktmo", "year"])["value"]
    assert rec.loc[("41701000", 2023)] == 120000.0  # без чужой инспекции и кода дохода 2000
    assert out["ndfl_income"]["flag"].isna().all()
    pairs = set(map(tuple, out["pairs"][["oktmo", "oktmo_stable"]].to_numpy()))
    assert ("42560000", "42650000") in pairs
    # Пара старого кода есть только в строке 2016 года другого показателя (Y777000030) — она тоже в мосте.
    assert ("42630000", "42635000") in pairs
    present = set(map(tuple, out["present"].astype({"year": "int64"}).to_numpy()))
    assert ("41631000", 2022) in present and ("41631000", 2023) not in present
    assert ("42630000", 2016) in present
