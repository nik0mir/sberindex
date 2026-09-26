"""Соединение по ОКТМО: три шага по срезу года и запреты моста (spec_final, А.6)."""

import numpy as np
import pandas as pd

from munnet.panel.crosswalk import match_rows

LAST = 2024


def _dic(rows):
    """Мини-справочник: (territory_id, oktmo8, year_from, year_to)."""
    df = pd.DataFrame(rows, columns=["territory_id", "oktmo8", "year_from", "year_to"])
    df["territory_id"] = df["territory_id"].astype("int32")
    df["year_from"] = df["year_from"].astype("int16")
    df["year_to"] = df["year_to"].astype("int16")
    df["oktmo8"] = df["oktmo8"].astype("str")
    return df


def _rows(rows):
    """Строки источника: (oktmo, oktmo_stable, year, value[, dim])."""
    out = []
    for r in rows:
        oktmo, stable, year, value, *dim = r
        out.append(
            {
                "oktmo": oktmo,
                "oktmo_stable": stable,
                "year": year,
                "dim": dim[0] if dim else "TOTAL",
                "value": float(value),
                "period_used": "На 1 января",
                "flag": np.nan,
            }
        )
    df = pd.DataFrame(out)
    df["year"] = df["year"].astype("int16")
    for col in ("oktmo", "oktmo_stable", "dim", "flag", "period_used"):
        df[col] = df[col].astype("str")
    return df


def _pairs(rows):
    return pd.DataFrame(rows, columns=["oktmo", "oktmo_stable"]).astype("str")


def _get(res, tid, year, dim="TOTAL"):
    sel = res[(res["territory_id"] == tid) & (res["year"] == year)]
    got = sel[sel["dim"] == dim]
    return got.iloc[0] if len(got) else sel.iloc[0]


def test_priority_direct_then_version_then_stable():
    dic = _dic(
        [
            (1, "41701000", 2018, 9999),
            (3, "41620000", 2018, 2023),  # как Магадан: «на 1 января 2023» под кодом 2022 года
            (3, "41520000", 2023, 9999),
            (8, "42650000", 2018, 9999),
        ]
    )
    rows = _rows(
        [
            ("41701000", "41701000", 2023, 100),
            ("41799000", "41701000", 2023, 999),  # мост тоже есть, но прямой код важнее
            ("41620000", "41520000", 2023, 20),
            ("42560000", "42650000", 2023, 15),  # нового кода нет в справочнике: только мост
        ]
    )
    pairs = _pairs([("41701000", "41701000"), ("41799000", "41701000"), ("41620000", "41520000")])
    pairs = pd.concat([pairs, _pairs([("42560000", "42650000"), ("42650000", "42650000")])])
    res = match_rows(rows, dic, pairs, [2023], LAST)
    assert _get(res, 1, 2023)[["value", "method", "oktmo_used"]].tolist() == [100.0, "direct", "41701000"]
    assert _get(res, 3, 2023)[["value", "method", "oktmo_used"]].tolist() == [20.0, "version", "41620000"]
    assert _get(res, 8, 2023)[["value", "method", "oktmo_used"]].tolist() == [15.0, "stable", "42560000"]


def test_version_takes_nearest_old_version():
    # Две старые версии со строками: берётся ближайшая к году (код Y − 1, Л4), а не первая по году (А.6).
    dic = _dic([(3, "41610000", 2016, 2019), (3, "41620000", 2019, 2023), (3, "41520000", 2023, 9999)])
    rows = _rows([("41610000", "41610000", 2023, 1), ("41620000", "41620000", 2023, 2)])
    pairs = _pairs([("41610000", "41610000"), ("41620000", "41620000")])
    res = match_rows(rows, dic, pairs, [2023], LAST)
    assert _get(res, 3, 2023)[["value", "oktmo_used", "method"]].tolist() == [2.0, "41620000", "version"]


def test_version_blocked_when_code_is_direct_code_of_other_mo():
    dic = _dic([(17, "42670000", 2018, 2022), (17, "42671000", 2022, 9999), (18, "42670000", 2022, 9999)])
    rows = _rows([("42670000", "42670000", 2023, 5000)])
    res = match_rows(rows, dic, _pairs([("42670000", "42670000")]), [2023], LAST)
    assert _get(res, 18, 2023)["method"] == "direct"
    lost = _get(res, 17, 2023)
    assert np.isnan(lost["value"]) and lost["reason"] == "no_code"


def test_one_code_two_mo_in_different_years():
    dic = _dic([(4, "41630000", 2018, 2024), (6, "41630000", 2024, 9999)])
    rows = _rows([("41630000", "41630000", 2023, 91858), ("41630000", "41630000", 2024, 122210)])
    res = match_rows(rows, dic, _pairs([("41630000", "41630000")]), [2023, 2024], LAST)
    assert _get(res, 4, 2023)["value"] == 91858.0
    assert _get(res, 6, 2024)["value"] == 122210.0
    assert set(res.loc[res["year"] == 2023, "territory_id"]) == {4}  # 6 нет в срезе 2023 года
    assert set(res.loc[res["year"] == 2024, "territory_id"]) == {6}


def test_bridge_banned_when_stable_code_belongs_to_two_mo():
    # Павловский Посад и Электрогорск ведут к одному стабильному коду (Л9): мост дал бы сумму частей.
    dic = _dic([(4, "41630000", 2018, 2024), (5, "41631000", 2018, 2024), (6, "41630000", 2024, 9999)])
    rows = _rows([("41630000", "41630000", 2023, 91858), ("41639000", "41630000", 2023, 121788)])
    pairs = _pairs([("41630000", "41630000"), ("41631000", "41630000"), ("41639000", "41630000")])
    res = match_rows(rows, dic, pairs, [2023], LAST)
    assert _get(res, 4, 2023)["value"] == 91858.0
    lost = _get(res, 5, 2023)
    assert np.isnan(lost["value"]) and lost["reason"] == "union_banned"


def _present(codes_years):
    return pd.DataFrame(codes_years, columns=["oktmo", "year"]).astype({"oktmo": "str", "year": "int16"})


def test_union_written_under_one_part_code_is_flagged():
    # 5-НДФЛ 2023: Павлово-Посадский округ уже под кодом Павловского Посада, кода Электрогорска за год нет.
    dic = _dic([(4, "41630000", 2018, 2024), (5, "41631000", 2018, 2024), (6, "41630000", 2024, 9999)])
    rows = _rows([("41630000", "41630000", 2023, 37105)])
    pairs = _pairs([("41630000", "41630000"), ("41631000", "41630000")])
    present = _present([("41630000", 2022), ("41631000", 2022), ("41630000", 2023)])
    res = match_rows(rows, dic, pairs, [2023], LAST, present=present)
    four = _get(res, 4, 2023)
    assert four[["value", "method", "flag"]].tolist() == [37105.0, "direct", "union_merged"]
    five = _get(res, 5, 2023)
    assert np.isnan(five["value"]) and five["reason"] == "union_banned"
    # Код части есть в источнике за год (другой показатель) — значение части 4 своё, флага нет.
    present_both = _present([("41630000", 2023), ("41631000", 2023)])
    res = match_rows(rows, dic, pairs, [2023], LAST, present=present_both)
    assert pd.isna(_get(res, 4, 2023)["flag"]) and _get(res, 5, 2023)["reason"] == "no_code"
    # Без кодов источника шаг не выполняется.
    assert pd.isna(_get(match_rows(rows, dic, pairs, [2023], LAST), 4, 2023)["flag"])


def test_direct_beats_bridge_and_parts_are_not_summed():
    dic = _dic([(7, "63701000", 2023, 9999)])  # Саратов после объединения (Л10)
    rows = _rows([("63701000", "63701000", 2024, 937364), ("63702000", "63701000", 2024, 23861)])
    pairs = _pairs([("63701000", "63701000"), ("63702000", "63701000")])
    res = match_rows(rows, dic, pairs, [2024], LAST)
    got = res[(res["territory_id"] == 7) & (res["year"] == 2024)]
    assert got["value"].tolist() == [937364.0]
    assert got["method"].tolist() == ["direct"]


def test_bridge_does_not_take_direct_rows_of_other_mo():
    dic = _dic([(19, "42680000", 2018, 9999), (20, "42690000", 2018, 9999)])
    # Строка с прямым кодом МО 20 несёт стабильный код МО 19: мост её не забирает.
    rows = _rows([("42690000", "42680000", 2023, 777)])
    res = match_rows(rows, dic, _pairs([("42680000", "42680000")]), [2023], LAST)
    assert _get(res, 20, 2023)["method"] == "direct"
    assert np.isnan(_get(res, 19, 2023)["value"])


def test_multi_row_and_own_code_preference():
    dic = _dic([(9, "42660000", 2018, 9999), (10, "42663000", 2018, 9999)])
    rows = _rows(
        [
            ("42661000", "42660000", 2025, 100),
            ("42662000", "42660000", 2025, 200),  # два кандидата с разными значениями
            ("42664000", "42663000", 2025, 300),
            ("42665000", "42663000", 2025, 400),
        ]
    )
    pairs = _pairs(
        [
            ("42660000", "42660000"),
            ("42661000", "42660000"),
            ("42662000", "42660000"),
            ("42663000", "42665000"),  # стабильный код МО 10 сам встречается кодом строки
            ("42664000", "42665000"),
            ("42665000", "42665000"),
        ]
    )
    rows.loc[rows["oktmo"].isin(["42664000", "42665000"]), "oktmo_stable"] = "42665000"
    res = match_rows(rows, dic, pairs, [2025], LAST)
    lost = _get(res, 9, 2025)
    assert np.isnan(lost["value"]) and lost["reason"] == "multi_row"
    assert _get(res, 10, 2025)[["value", "oktmo_used", "method"]].tolist() == [400.0, "42665000", "stable"]


def test_identical_bridge_candidates_are_taken():
    dic = _dic([(9, "42660000", 2018, 9999)])
    rows = _rows([("42661000", "42660000", 2025, 100), ("42662000", "42660000", 2025, 100)])
    pairs = _pairs([("42660000", "42660000")])
    res = match_rows(rows, dic, pairs, [2025], LAST)
    assert _get(res, 9, 2025)[["value", "method"]].tolist() == [100.0, "stable"]


def test_all_dims_of_one_code_and_reasons():
    dic = _dic([(1, "41701000", 2018, 9999), (10, "61710000", 2018, 9999), (11, "41702000", 2018, 9999)])
    rows = _rows([("41701000", "41701000", 2024, 1000, "TOTAL"), ("41701000", "41701000", 2024, 100, "A")])
    res = match_rows(rows, dic, _pairs([("41701000", "41701000")]), [2024], LAST)
    one = res[res["territory_id"] == 1].set_index("dim")["value"]
    assert one.to_dict() == {"A": 100.0, "TOTAL": 1000.0}
    assert _get(res, 10, 2024)["reason"] == "region_no_rows"  # у субъекта 61 нет строк за год
    assert _get(res, 11, 2024)["reason"] == "no_code"
    assert res["territory_id"].dtype == np.int32 and res["year"].dtype == np.int16


def test_empty_rows_give_all_slice_mo_without_values():
    dic = _dic([(1, "41701000", 2018, 9999)])
    res = match_rows(_rows([("41701000", "41701000", 2023, 1)]).iloc[:0], dic, _pairs([]), [2023], LAST)
    assert len(res) == 1 and res["reason"].iloc[0] == "region_no_rows"
