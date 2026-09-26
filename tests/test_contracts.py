import numpy as np
import pandas as pd
import pytest

from munnet.contracts import (
    CATEGORY_CODES,
    CONTEXT_ANNUAL,
    PANEL_LONG,
    PANEL_WIDE,
    PARTS,
    TERRITORIES,
    Col,
    SchemaError,
    TableSchema,
    read_table,
    validate,
    write_table,
)


def panel_long(n_mo: int = 2) -> pd.DataFrame:
    """Два МО × два месяца × 7 категорий, в обратном порядке строк (validate должен отсортировать)."""
    rows = []
    for tid in range(n_mo, 0, -1):
        for t, date in ((1, "2023-02"), (0, "2023-01")):
            for c in CATEGORY_CODES:
                rows.append((tid, date, t, 2023, t + 1, c, 100 + t, c == "other"))
    df = pd.DataFrame(
        rows, columns=["territory_id", "date", "t", "year", "month", "category", "value", "is_derived"]
    )
    return df.astype(
        {
            "territory_id": "int32",
            "t": "int8",
            "year": "int16",
            "month": "int8",
            "value": "int32",
            "category": pd.CategoricalDtype(list(CATEGORY_CODES), ordered=True),
        }
    ).set_index(pd.Index(range(100, 100 + len(rows))))


def test_valid_table_is_sorted_and_reindexed():
    df = panel_long()
    out = validate(df, PANEL_LONG)
    assert list(out.columns) == [c.name for c in PANEL_LONG.columns]
    assert list(out.index) == list(range(len(df)))
    keys = list(zip(out["territory_id"], out["date"], out["category"].cat.codes, strict=True))
    assert keys == sorted(keys)
    assert len(df) == len(out) and df.index[0] == 100  # исходная таблица не изменилась


def test_all_violations_reported_at_once():
    df = panel_long()
    df["value"] = df["value"].astype("int64")  # тип
    df["extra"] = 1  # лишняя колонка
    df.loc[df.index[0], "date"] = None  # пропуск
    df = pd.concat([df, df.iloc[[3]]])  # дубль ключа
    with pytest.raises(SchemaError) as err:
        validate(df, PANEL_LONG)
    msg = str(err.value)
    assert "value: тип int64, ожидался int32" in msg
    assert "лишние колонки ['extra']" in msg
    assert "date: 1 пропусков" in msg
    assert "повторяющимся ключом" in msg


def test_range_and_missing_column():
    df = panel_long().drop(columns=["is_derived"])
    df.loc[df.index[0], "value"] = 0  # траты > 0
    df.loc[df.index[1], "month"] = 13
    with pytest.raises(SchemaError) as err:
        validate(df, PANEL_LONG)
    msg = str(err.value)
    assert "нет колонок ['is_derived']" in msg
    assert "value: 1 значений вне" in msg and "month: 1 значений вне" in msg


def test_row_checks_catch_inconsistent_month_and_derived_flag():
    df = panel_long()
    df.loc[df.index[0], "t"] = 5
    with pytest.raises(SchemaError, match="месяц согласован"):
        validate(df, PANEL_LONG)
    df = panel_long()
    df.loc[df["category"] == "food", "is_derived"] = True
    with pytest.raises(SchemaError, match="is_derived только у other"):
        validate(df, PANEL_LONG)


def test_unordered_or_foreign_categories_rejected():
    df = panel_long()
    df["category"] = df["category"].cat.as_unordered()
    with pytest.raises(SchemaError, match="упорядоченная категория"):
        validate(df, PANEL_LONG)
    df = panel_long()
    df["category"] = df["category"].cat.add_categories(["Продовольствие"])
    with pytest.raises(SchemaError, match="недопустимые категории"):
        validate(df, PANEL_LONG)


def test_wide_shares_must_sum_to_one():
    long = panel_long(1)
    wide = long.pivot_table(
        index=["territory_id", "date", "t", "year", "month"],
        columns="category",
        values="value",
        observed=True,
    ).reset_index()
    wide.columns = [
        c if c in ("territory_id", "date", "t", "year", "month") else f"v_{c}" for c in wide.columns
    ]
    wide["v_all"] = wide[[f"v_{p}" for p in PARTS]].sum(axis=1)
    for c in CATEGORY_CODES:
        wide[f"v_{c}"] = wide[f"v_{c}"].astype("int32")
    for p in PARTS:
        wide[f"sh_{p}"] = wide[f"v_{p}"] / wide["v_all"]
    wide["log_all"] = np.log(wide["v_all"].astype(float))
    wide = wide.astype({"territory_id": "int32", "t": "int8", "year": "int16", "month": "int8"})
    validate(wide, PANEL_WIDE)
    wide.loc[0, "sh_food"] += 1e-6
    with pytest.raises(SchemaError, match="шесть долей в сумме 1"):
        validate(wide, PANEL_WIDE)


def test_pattern_catches_lost_leading_zero_of_oktmo():
    schema = TableSchema("t", (Col("id", "int32"), Col("oktmo", "string", pattern=r"\d{8}")), key=("id",))
    ok = pd.DataFrame({"id": np.array([1], dtype="int32"), "oktmo": ["01512000"]})
    validate(ok, schema)
    bad = pd.DataFrame({"id": np.array([1], dtype="int32"), "oktmo": ["1512000"]})
    with pytest.raises(SchemaError, match="не по шаблону"):
        validate(bad, schema)


def test_nullable_int_accepted_only_when_declared():
    schema = TableSchema("t", (Col("id", "int32"), Col("succ", "int32", nullable=True)), key=("id",))
    df = pd.DataFrame({"id": pd.array([1, 2], dtype="Int32"), "succ": pd.array([3, None], dtype="Int32")})
    validate(df, schema)
    df["id"] = pd.array([1, None], dtype="Int32")
    with pytest.raises(SchemaError, match="id: 1 пропусков"):
        validate(df, schema)


def test_write_table_does_not_touch_old_file_on_error(tmp_path):
    path = tmp_path / "panel_long.parquet"
    write_table(panel_long(), PANEL_LONG, path)
    before = path.read_bytes()
    bad = panel_long()
    bad["value"] = -1
    with pytest.raises(SchemaError):
        write_table(bad, PANEL_LONG, path)
    assert path.read_bytes() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["panel_long.parquet"]  # временных файлов нет


def test_read_table_skips_index_column_and_restores_types(tmp_path):
    path = tmp_path / "panel_long.parquet"
    panel_long().to_parquet(path, index=True)  # непоследовательный индекс -> __index_level_0__ (Л23)
    out = read_table(path, PANEL_LONG)
    assert "__index_level_0__" not in out.columns
    assert list(out.columns) == [c.name for c in PANEL_LONG.columns]
    assert out["category"].cat.ordered and tuple(out["category"].cat.categories) == CATEGORY_CODES
    assert out["territory_id"].dtype == np.int32


def test_nullable_int_survives_round_trip(tmp_path):
    schema = TableSchema("t", (Col("id", "int32"), Col("succ", "int32", nullable=True)), key=("id",))
    df = pd.DataFrame({"id": np.array([2, 1], dtype="int32"), "succ": pd.array([None, 7], dtype="Int32")})
    path = write_table(df, schema, tmp_path / "t.parquet")
    out = read_table(path, schema)
    assert str(out["succ"].dtype) == "Int32"
    assert out["succ"].isna().tolist() == [False, True]


def test_extra_columns_only_where_allowed():
    assert CONTEXT_ANNUAL.extra_allowed
    assert not TERRITORIES.extra_allowed and not PANEL_LONG.extra_allowed


def test_workplace_rule_in_context_annual(tmp_path):
    from synth import make_processed

    processed = make_processed(tmp_path)
    annual = read_table(processed / "context_annual.parquet", CONTEXT_ANNUAL)
    assert (
        annual.columns[len(CONTEXT_ANNUAL.columns) :].str.startswith("emp_sh_").all()
    )  # разделы — после схемы
    inner = annual.index[annual["workplace_based"]][0]
    annual.loc[inner, "payroll_pc"] = 1000.0  # показатель по месту работы у внутригородской территории
    with pytest.raises(SchemaError, match="workplace_based"):
        validate(annual, CONTEXT_ANNUAL)


def test_synthetic_tables_satisfy_contracts(tmp_path):
    from synth import make_processed

    from munnet.contracts import SCHEMAS

    processed = make_processed(tmp_path)
    for name, schema in SCHEMAS.items():
        df = read_table(processed / f"{name}.parquet", schema)
        assert len(df) > 0, name
