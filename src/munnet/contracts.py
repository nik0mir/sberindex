"""Контракты таблиц этапов panel и eda: схемы, проверка, запись и чтение parquet.

Схемы описывают выходы этапа panel (spec_final, часть А.4). ``validate`` проверяет набор колонок, типы,
пропуски, диапазоны, допустимые значения, шаблоны строк, уникальность ключа и межколоночные правила и
собирает все нарушения в одно сообщение ``SchemaError``. ``write_table`` пишет только проверенную таблицу
и не портит старый файл при ошибке; ``read_table`` читает явный список колонок (ловушка Л23) и
восстанавливает типы схемы.
"""

from __future__ import annotations

import math
import os
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# Коды категорий: единственный словарь проекта (spec_final, А.3). Подписи — style.LABELS.
CATEGORY_CODES: tuple[str, ...] = ("all", "food", "marketplace", "transport", "health", "cafe", "other")
# Шесть частей замкнутой композиции: пять категорий источника и остаток «Прочее».
PARTS: tuple[str, ...] = CATEGORY_CODES[1:]

# Период панели: годы и число месяцев (config: panel.years, period). Совпадение с конфигом проверяет тест.
YEARS: tuple[int, ...] = (2023, 2024)
N_MONTHS: int = 12 * len(YEARS)
# Годы годового контекста: население 2022–2025 (config: context.read_years), остальное — годы панели.
CONTEXT_YEAR_RANGE: tuple[float, float] = (2022.0, 2025.0)

# Допуск межколоночных равенств для float64: сумма шести долей = 1, log_all = ln(v_all) (А.4).
SUM_TOL = 1e-9
# Допуск границ диапазона для float64: доли вида x / y могут выйти за 1 на ошибку округления.
RANGE_TOL = 1e-9
# Сколько примеров нарушений показывать в сообщении на одно правило.
MAX_EXAMPLES = 5

INT32_MAX = float(np.iinfo(np.int32).max)
INF = math.inf

DATE_PATTERN = r"\d{4}-(0[1-9]|1[0-2])"
OKTMO_PATTERN = r"\d{8}"
FLAG_PATTERN = r"([a-z_]+(;[a-z_]+)*)?"

_INT_DTYPES = ("int8", "int16", "int32", "int64")
_DTYPES = (*_INT_DTYPES, "float64", "string", "bool", "category")


class SchemaError(ValueError):
    """Таблица не соответствует схеме; в сообщении перечислены все нарушения."""


class QCError(RuntimeError):
    """Не прошла проверка качества: жёсткое контрольное число или заголовок графика (код выхода 3)."""


class MissingInputError(FileNotFoundError):
    """Нет входов этапа: предыдущий этап не запускался (код выхода 1)."""


@dataclass(frozen=True)
class Col:
    """Колонка схемы.

    ``dtype`` — один из ``int8``, ``int16``, ``int32``, ``int64``, ``float64``, ``string``, ``bool``,
    ``category``. ``range`` — включительные границы для чисел; ``values`` — допустимые значения для
    ``category`` и ``string`` (у ``category`` с ``ordered=True`` это ещё и порядок категорий);
    ``pattern`` — регулярное выражение, которому целиком соответствует каждое непустое значение строки.
    """

    name: str
    dtype: str
    nullable: bool = False
    range: tuple[float, float] | None = None
    values: tuple[str, ...] | None = None
    pattern: str | None = None
    ordered: bool = False

    def __post_init__(self) -> None:
        if self.dtype not in _DTYPES:
            raise ValueError(f"колонка {self.name}: неизвестный тип {self.dtype!r}")


@dataclass(frozen=True)
class Check:
    """Межколоночное правило: ``func(df)`` возвращает bool-серию, False — строка нарушает правило."""

    name: str
    func: Callable[[pd.DataFrame], pd.Series]


@dataclass(frozen=True)
class TableSchema:
    """Схема таблицы: колонки в порядке записи, ключ, межколоночные правила."""

    name: str
    columns: tuple[Col, ...]
    key: tuple[str, ...]
    extra_allowed: bool = False
    checks: tuple[Check, ...] = ()

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.columns)

    def col(self, name: str) -> Col:
        for c in self.columns:
            if c.name == name:
                return c
        raise KeyError(f"{self.name}: нет колонки {name}")


# --- Проверка ------------------------------------------------------------------------------------


def _examples(values: pd.Series | pd.Index) -> str:
    vals = list(values[:MAX_EXAMPLES])
    more = f" и ещё {len(values) - MAX_EXAMPLES}" if len(values) > MAX_EXAMPLES else ""
    return ", ".join(repr(v.item() if hasattr(v, "item") else v) for v in vals) + more


def _dtype_ok(s: pd.Series, col: Col) -> bool:
    dt = s.dtype
    if col.dtype in _INT_DTYPES:
        # numpy intN или nullable IntN той же ширины: int64 вместо int32 — нарушение контракта.
        return str(dt).lower() == col.dtype
    if col.dtype == "float64":
        return dt == np.float64
    if col.dtype == "bool":
        return dt == np.bool_ or str(dt) == "boolean"
    if col.dtype == "category":
        return isinstance(dt, pd.CategoricalDtype)
    # string: StringDtype любого вида или object, где все непустые значения — строки.
    if isinstance(dt, pd.StringDtype):
        return True
    if dt == np.dtype(object):
        return bool(s.dropna().map(lambda v: isinstance(v, str)).all())
    return False


def _check_column(s: pd.Series, col: Col, table: str) -> list[str]:
    errors: list[str] = []
    where = f"{table}.{col.name}"
    if not _dtype_ok(s, col):
        errors.append(f"{where}: тип {s.dtype}, ожидался {col.dtype}")
        return errors  # остальные проверки на чужом типе бессмысленны
    na = s.isna()
    if not col.nullable and na.any():
        errors.append(f"{where}: {int(na.sum())} пропусков, пропуски запрещены")
    present = s[~na]
    if present.empty:
        return errors
    if col.range is not None and col.dtype not in ("string", "bool", "category"):
        lo, hi = col.range
        tol = RANGE_TOL if col.dtype == "float64" else 0.0
        vals = present.astype("float64")
        bad = present[(vals < lo - tol) | (vals > hi + tol)]
        if not bad.empty:
            errors.append(f"{where}: {len(bad)} значений вне [{lo}; {hi}]: {_examples(bad)}")
    if col.dtype == "category":
        cats = tuple(str(c) for c in s.cat.categories)
        if col.values is not None:
            unknown = sorted(set(cats) - set(col.values))
            if unknown:
                errors.append(f"{where}: недопустимые категории {unknown}, допустимы {list(col.values)}")
            if col.ordered and (not s.cat.ordered or cats != col.values):
                errors.append(f"{where}: ожидалась упорядоченная категория {list(col.values)}")
    elif col.dtype == "string" and col.values is not None:
        bad = present[~present.isin(col.values)]
        if not bad.empty:
            errors.append(f"{where}: {len(bad)} недопустимых значений: {_examples(bad.unique())}")
    if col.pattern is not None and col.dtype in ("string", "category"):
        strs = present.astype(str)
        bad = strs[~strs.str.fullmatch(col.pattern)]
        if not bad.empty:
            errors.append(
                f"{where}: {len(bad)} значений не по шаблону {col.pattern}: {_examples(bad.unique())}"
            )
    return errors


def validate(df: pd.DataFrame, schema: TableSchema) -> pd.DataFrame:
    """Проверяет таблицу по схеме и возвращает её нормализованную копию.

    Все нарушения (нет колонки, лишняя колонка, тип, пропуски, диапазон, допустимые значения, шаблон,
    дубль ключа, межколоночные правила) собираются в одно сообщение ``SchemaError``. Копия: колонки в
    порядке схемы (разрешённые лишние — после них), строки отсортированы по ключу, индекс 0…n−1.
    """
    errors: list[str] = []
    names = schema.names
    missing = [c for c in names if c not in df.columns]
    if missing:
        errors.append(f"{schema.name}: нет колонок {missing}")
    extra = [c for c in df.columns if c not in names]
    if extra and not schema.extra_allowed:
        errors.append(f"{schema.name}: лишние колонки {extra}")
    for col in schema.columns:
        if col.name in df.columns:
            errors.extend(_check_column(df[col.name], col, schema.name))
    key = [k for k in schema.key if k in df.columns]
    if len(key) == len(schema.key) and len(df):
        dup = df.duplicated(subset=key, keep=False)
        if dup.any():
            sample = df.loc[dup, key].drop_duplicates().head(MAX_EXAMPLES)
            examples = "; ".join(str(tuple(r)) for r in sample.itertuples(index=False))
            errors.append(f"{schema.name}: {int(dup.sum())} строк с повторяющимся ключом {key}: {examples}")
    if not missing and not errors:
        # Межколоночные правила имеют смысл только на таблице с верными колонками и типами.
        for check in schema.checks:
            ok = check.func(df).fillna(False).astype(bool)
            if not ok.all():
                bad = df.loc[~ok, list(schema.key)].head(MAX_EXAMPLES)
                examples = "; ".join(str(tuple(r)) for r in bad.itertuples(index=False))
                errors.append(
                    f"{schema.name}: правило «{check.name}» нарушено в {int((~ok).sum())} строках: {examples}"
                )
    if errors:
        raise SchemaError(f"{schema.name}: {len(errors)} нарушений схемы\n- " + "\n- ".join(errors))
    order = [*names, *[c for c in df.columns if c not in names]]
    out = df[order].sort_values(list(schema.key), kind="mergesort") if len(df) else df[order]
    return out.reset_index(drop=True)


# --- Запись и чтение -----------------------------------------------------------------------------


def write_table(df: pd.DataFrame, schema: TableSchema, path: Path) -> Path:
    """Проверяет таблицу и атомарно пишет parquet (pyarrow, snappy, без индекса).

    Сначала ``validate``: при нарушении схемы файл не трогается. Затем запись во временный файл рядом
    с целевым и ``os.replace``: старый файл не портится и при сбое записи.
    """
    table = validate(df, schema)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        table.to_parquet(tmp, engine="pyarrow", compression="snappy", index=False)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return path


def coerce(df: pd.DataFrame, schema: TableSchema) -> pd.DataFrame:
    """Приводит прочитанные колонки к типам схемы.

    Целые с пропусками — nullable IntN, bool с пропусками — boolean, категории — список и порядок схемы.
    """
    out = df.copy()
    for col in schema.columns:
        if col.name not in out.columns:
            continue
        s = out[col.name]
        if col.dtype in _INT_DTYPES:
            if s.isna().any() or isinstance(s.dtype, pd.api.extensions.ExtensionDtype):
                out[col.name] = s.astype(col.dtype.capitalize())  # int32 -> nullable Int32
            else:
                out[col.name] = s.astype(col.dtype)
        elif col.dtype == "float64":
            out[col.name] = s.astype("float64")
        elif col.dtype == "bool":
            out[col.name] = s.astype("boolean") if s.isna().any() else s.astype(bool)
        elif col.dtype == "category":
            if col.values is not None:
                out[col.name] = (
                    s.astype(str)
                    .where(s.notna())
                    .astype(pd.CategoricalDtype(list(col.values), ordered=col.ordered))
                )
            elif not isinstance(s.dtype, pd.CategoricalDtype):
                out[col.name] = s.astype("category")
    return out


def read_table(path: Path, schema: TableSchema) -> pd.DataFrame:
    """Читает parquet по схеме: явный список колонок (служебный ``__index_level_0__`` не поднимается, Л23),
    типы схемы, проверка ``validate``."""
    import pyarrow.parquet as pq

    path = Path(path)
    available = pq.read_schema(path).names
    columns = [c for c in schema.names if c in available]
    if schema.extra_allowed:
        columns += [c for c in available if c not in schema.names and not c.startswith("__index_level_")]
    df = pd.read_parquet(path, columns=columns)
    return validate(coerce(df, schema), schema)


# --- Схемы выходов этапа panel (А.4) -------------------------------------------------------------

SHARE = (0.0, 1.0)
NONNEG = (0.0, INF)
POSITIVE_INT = (1.0, INT32_MAX)
LAT = (-90.0, 90.0)
LON = (-180.0, 180.0)

_YEAR_RANGE = (float(min(YEARS)), float(max(YEARS)))
_MONTH_COLS = (
    Col("date", "string", pattern=DATE_PATTERN),
    Col("t", "int8", range=(0, N_MONTHS - 1)),
    Col("year", "int16", range=_YEAR_RANGE),
    Col("month", "int8", range=(1, 12)),
)


def _month_consistent(df: pd.DataFrame) -> pd.Series:
    """date = «ГГГГ-ММ» из year и month; t = номер месяца от первого месяца панели."""
    year = df["year"].astype("int64")
    month = df["month"].astype("int64")
    date_ok = df["date"].astype(str) == year.astype(str) + "-" + month.astype(str).str.zfill(2)
    t_ok = df["t"].astype("int64") == (year - min(YEARS)) * 12 + month - 1
    return date_ok & t_ok


PANEL_LONG = TableSchema(
    name="panel_long",
    columns=(
        Col("territory_id", "int32", range=POSITIVE_INT),
        *_MONTH_COLS,
        Col("category", "category", values=CATEGORY_CODES, ordered=True),
        Col("value", "int32", range=POSITIVE_INT),
        Col("is_derived", "bool"),
    ),
    key=("territory_id", "date", "category"),
    checks=(
        Check("месяц согласован: date, t, year, month", _month_consistent),
        Check(
            "is_derived только у other", lambda d: d["is_derived"] == (d["category"].astype(str) == "other")
        ),
    ),
)


def _shares_sum_to_one(df: pd.DataFrame) -> pd.Series:
    return (df[[f"sh_{p}" for p in PARTS]].sum(axis=1) - 1.0).abs() <= SUM_TOL


def _parts_sum_to_all(df: pd.DataFrame) -> pd.Series:
    return df[[f"v_{p}" for p in PARTS]].astype("int64").sum(axis=1) == df["v_all"].astype("int64")


def _log_all(df: pd.DataFrame) -> pd.Series:
    return (df["log_all"] - np.log(df["v_all"].astype("float64"))).abs() <= SUM_TOL


PANEL_WIDE = TableSchema(
    name="panel_wide",
    columns=(
        Col("territory_id", "int32", range=POSITIVE_INT),
        *_MONTH_COLS,
        *(Col(f"v_{c}", "int32", range=POSITIVE_INT) for c in CATEGORY_CODES),
        *(Col(f"sh_{p}", "float64", range=SHARE) for p in PARTS),
        Col("log_all", "float64"),
    ),
    key=("territory_id", "date"),
    checks=(
        Check("месяц согласован: date, t, year, month", _month_consistent),
        Check("шесть долей в сумме 1", _shares_sum_to_one),
        Check("шесть частей в сумме дают v_all", _parts_sum_to_all),
        Check("log_all = ln(v_all)", _log_all),
    ),
)

MO_TYPES = ("mr", "mo", "go", "vgt")
MO_STATUSES = ("capital", "zato", "federal_territory")
SERIES_STATUSES = ("full", "only_2023", "only_2024", "partial")
LINEAGE_ROLES = ("none", "union_predecessor", "union_successor")
MATCH_METHODS = ("direct", "version", "stable", "carried")
POP_AVG_METHODS = ("mean_jan1", "jan1_only")


def _territory_consistent(df: pd.DataFrame) -> pd.Series:
    months_ok = df["n_months"].astype("int64") == df["n_2023"].astype("int64") + df["n_2024"].astype("int64")
    pattern_ok = df["coverage_pattern"].astype(str).str.count("1") == df["n_months"].astype("int64")
    capital_ok = df["is_capital"] == (df["mo_status"].astype(str) == "capital")
    return months_ok & pattern_ok & capital_ok


TERRITORIES = TableSchema(
    name="territories",
    columns=(
        Col("territory_id", "int32", range=POSITIVE_INT),
        Col("name", "string"),
        Col("name_short", "string"),
        Col("region_code", "int16", range=(1, 99)),
        Col("region_name", "string"),
        Col("mo_type", "category", values=MO_TYPES),
        Col("mo_status", "category", nullable=True, values=MO_STATUSES),
        Col("is_capital", "bool"),
        Col("is_inner_city", "bool"),
        Col("oktmo_2023", "string", nullable=True, pattern=OKTMO_PATTERN),
        Col("oktmo_2024", "string", nullable=True, pattern=OKTMO_PATTERN),
        Col("shape", "int8", range=(1, 3)),
        Col("center_name", "string", nullable=True),
        Col("center_lat", "float64", nullable=True, range=LAT),
        Col("center_lon", "float64", nullable=True, range=LON),
        Col("point_lat", "float64", range=LAT),
        Col("point_lon", "float64", range=LON),
        Col("x_aea", "float64"),
        Col("y_aea", "float64"),
        Col("area_km2", "float64", range=(0.0, INF)),
        Col("market_access", "float64", nullable=True, range=(0.0, 1000.0)),
        Col("dist_capital_km", "float64", nullable=True, range=NONNEG),
        Col("n_months", "int8", range=(1, N_MONTHS)),
        Col("n_2023", "int8", range=(0, 12)),
        Col("n_2024", "int8", range=(0, 12)),
        Col("first_date", "string", pattern=DATE_PATTERN),
        Col("last_date", "string", pattern=DATE_PATTERN),
        Col("coverage_pattern", "string", pattern=rf"[01]{{{N_MONTHS}}}"),
        Col("series_status", "category", values=SERIES_STATUSES),
        Col("has_internal_gap", "bool"),
        Col("longest_gap", "int8", range=(0, N_MONTHS - 2)),
        Col("lineage_role", "category", values=LINEAGE_ROLES),
        Col("lineage_change_id", "string", nullable=True, pattern=r"union_\d+"),
        Col("successor_id", "int32", nullable=True, range=POSITIVE_INT),
    ),
    key=("territory_id",),
    checks=(
        Check(
            "n_months = n_2023 + n_2024 = число единиц; is_capital = статус capital", _territory_consistent
        ),
    ),
)

GEO_COLUMNS: tuple[str, ...] = ("territory_id", "year_from", "year_to", "region_code", "in_panel", "geometry")
# Необязательная колонка территорий карты: подпись выпавших регионов (если этап panel её пишет).
GEO_OPTIONAL: tuple[str, ...] = ("region_name",)

CONTEXT_LONG = TableSchema(
    name="context_long",
    columns=(
        Col("territory_id", "int32", range=POSITIVE_INT),
        Col("year", "int16", range=CONTEXT_YEAR_RANGE),
        Col("indicator", "category"),
        Col("dim", "string"),
        Col("value", "float64"),
        Col("unit", "string"),
        Col("source_code", "string"),
        Col("period_used", "string"),
        Col("oktmo_used", "string", pattern=OKTMO_PATTERN),
        Col("method", "category", values=MATCH_METHODS),
        Col("flag", "string", nullable=True, pattern=FLAG_PATTERN),
    ),
    key=("territory_id", "year", "indicator", "dim"),
)

# Разделы ОКВЭД2 латиницей (Л16) и «не раскрыто» (Л17).
OKVED_SECTIONS: tuple[str, ...] = tuple("ABCDEFGHIJKLMNOPQRSTU")
OKVED_GROUPS: tuple[str, ...] = ("primary", "industry", "trade_transport", "market_services", "public")


def _f(name: str, rng: tuple[float, float] | None = NONNEG) -> Col:
    """Числовая колонка контекста: float64, пропуск допустим (нет значения — не ноль)."""
    return Col(name, "float64", nullable=True, range=rng)


# Показатели по месту работы, которые на жителя не считаются у внутригородских территорий (Л3, А.10).
WORKPLACE_PER_CAPITA: tuple[str, ...] = (
    "payroll_pc",
    "retail_pc",
    "catering_turnover_pc",
    "shipments_pc",
    "ndfl_income_pc",
    "employees_to_working_age",
)


def _workplace_na(df: pd.DataFrame) -> pd.Series:
    return ~df["workplace_based"].astype(bool) | df[list(WORKPLACE_PER_CAPITA)].isna().all(axis=1)


def _ndfl_ok_consistent(df: pd.DataFrame) -> pd.Series:
    return ~(df["ndfl_ok"].astype(bool) & df["workplace_based"].astype(bool))


def _ndfl_pc_na(df: pd.DataFrame) -> pd.Series:
    return df["ndfl_ok"].astype(bool) | df["ndfl_income_pc"].isna()


CONTEXT_ANNUAL = TableSchema(
    name="context_annual",
    columns=(
        Col("territory_id", "int32", range=POSITIVE_INT),
        Col("year", "int16", range=_YEAR_RANGE),
        _f("pop_jan1", (1.0, INF)),
        Col("pop_jan1_method", "category", nullable=True, values=MATCH_METHODS),
        _f("pop_avg", (1.0, INF)),
        Col("pop_avg_method", "category", nullable=True, values=POP_AVG_METHODS),
        _f("urban_share", SHARE),
        _f("age_young_share", SHARE),
        _f("age_working_share", SHARE),
        _f("age_old_share", SHARE),
        _f("employees"),
        _f("payroll_krub"),
        _f("wage"),
        _f("emp_sh_A", SHARE),
        _f("emp_sh_B", SHARE),
        _f("emp_sh_C", SHARE),
        *(_f(f"emp_sh_{g}", SHARE) for g in OKVED_GROUPS),
        _f("emp_sh_unallocated", SHARE),
        Col("emp_n_disclosed", "int8", nullable=True, range=(0, len(OKVED_SECTIONS))),
        _f("retail_krub"),
        _f("catering_turnover_krub"),
        _f("shipments_krub"),
        _f("shipments_mining_share", SHARE),
        _f("orgs"),
        _f("ip"),
        _f("nights"),
        _f("beds"),
        _f("invest_krub"),
        _f("ndfl_income_rub"),
        _f("ndfl_recipients"),
        _f("payroll_pc"),
        _f("retail_pc"),
        _f("catering_turnover_pc"),
        _f("shipments_pc"),
        _f("ndfl_income_pc"),
        _f("ndfl_income_per_recipient"),
        _f("recipients_to_pop"),
        _f("employees_to_working_age"),
        _f("orgs_per_1000"),
        _f("ip_per_1000"),
        _f("beds_per_1000"),
        _f("nights_pc"),
        Col("workplace_based", "bool"),
        Col("ndfl_ok", "bool"),
    ),
    key=("territory_id", "year"),
    extra_allowed=True,  # emp_sh_<раздел> для остальных разделов ОКВЭД2 (сколько раскрыто в данных)
    checks=(
        Check("при workplace_based показатели по месту работы на жителя пусты", _workplace_na),
        Check("ndfl_ok исключает workplace_based", _ndfl_ok_consistent),
        Check("при ndfl_ok = false доход 5-НДФЛ на жителя пуст", _ndfl_pc_na),
    ),
)

OKVED_SHARES = TableSchema(
    name="okved_shares",
    columns=(
        Col("territory_id", "int32", range=POSITIVE_INT),
        Col("year", "int16", range=_YEAR_RANGE),
        Col("section", "string", values=(*OKVED_SECTIONS, "unallocated")),
        _f("employees"),
        _f("share", SHARE),
    ),
    key=("territory_id", "year", "section"),
)

SCHEMAS: dict[str, TableSchema] = {
    s.name: s for s in (PANEL_LONG, PANEL_WIDE, TERRITORIES, CONTEXT_LONG, CONTEXT_ANNUAL, OKVED_SHARES)
}
