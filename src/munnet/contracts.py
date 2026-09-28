"""Контракты таблиц этапов panel, eda, features, network и cluster: схемы, проверка, запись и чтение parquet.

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

# Строки «Субъект РФ» 5-НДФЛ городов федерального значения (узлы-города, ``munnet.nodes``): получатели
# дохода Москвы берутся из городской строки — сумма по районам считает людей дважды.
CITY_INDICATORS: tuple[str, ...] = ("ndfl_income", "ndfl_recipients")
CITY_CONTEXT = TableSchema(
    name="city_context",
    columns=(
        Col("region_code", "int16", range=(1, 99)),
        Col("year", "int16", range=CONTEXT_YEAR_RANGE),
        Col("indicator", "string", values=CITY_INDICATORS),
        Col("value", "float64", range=(0.0, INF)),
        Col("oktmo", "string", pattern=OKTMO_PATTERN),
        Col("source_code", "string"),
        Col("report_type", "string"),
        Col("flag", "string", nullable=True, pattern=FLAG_PATTERN),
    ),
    key=("region_code", "year", "indicator"),
)

SCHEMAS: dict[str, TableSchema] = {
    s.name: s
    for s in (PANEL_LONG, PANEL_WIDE, TERRITORIES, CONTEXT_LONG, CONTEXT_ANNUAL, OKVED_SHARES, CITY_CONTEXT)
}


# --- Схемы узлов сети и признаков узлов (этап 2: munnet.nodes, munnet.features) -----------------
#
# Узел — МО панели или узел-город (внутригородские территории Москвы и Петербурга, свёрнутые в два узла
# в режиме ``nodes.mode = collapse``). У узла-города свой ``territory_id`` (``nodes.city_id_base`` + код
# субъекта), которого нет в справочнике СберИндекса: связь с МО панели — таблица ``node_members``.

NODE_MODES: tuple[str, ...] = ("collapse", "separate", "exclude")
CITY_MO_TYPE = "city"  # тип узла-города: город федерального значения целиком
NODE_MO_TYPES: tuple[str, ...] = (*MO_TYPES, CITY_MO_TYPE)
MEMBER_ROLES: tuple[str, ...] = ("self", "city_member", "excluded")
MAX_MEMBERS = 999.0


def _node_consistent(df: pd.DataFrame) -> pd.Series:
    months_ok = df["n_months"].astype("int64") == df["n_2023"].astype("int64") + df["n_2024"].astype("int64")
    pattern_ok = df["coverage_pattern"].astype(str).str.count("1") == df["n_months"].astype("int64")
    city_ok = df["is_city_node"] == (df["mo_type"].astype(str) == CITY_MO_TYPE)
    members_ok = df["is_city_node"] | (df["n_members"].astype("int64") == 1)
    return months_ok & pattern_ok & city_ok & members_ok


NODES = TableSchema(
    name="nodes",
    columns=(
        Col("territory_id", "int32", range=POSITIVE_INT),
        Col("name", "string"),
        Col("name_short", "string"),
        Col("region_code", "int16", range=(1, 99)),
        Col("region_name", "string"),
        Col("region_group", "int16", range=(1, 99)),
        Col("region_group_name", "string"),
        Col("mo_type", "category", values=NODE_MO_TYPES),
        Col("mo_status", "category", nullable=True, values=MO_STATUSES),
        Col("is_capital", "bool"),
        Col("is_inner_city", "bool"),
        Col("is_city_node", "bool"),
        Col("n_members", "int16", range=(1, MAX_MEMBERS)),
        Col("oktmo_2023", "string", nullable=True, pattern=OKTMO_PATTERN),
        Col("oktmo_2024", "string", nullable=True, pattern=OKTMO_PATTERN),
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
    ),
    key=("territory_id",),
    extra_allowed=True,  # этап features добавляет is_node и drop_reason
    checks=(
        Check(
            "n_months = n_2023 + n_2024 = число единиц; узел-город ⇔ тип city; у МО один участник",
            _node_consistent,
        ),
    ),
)


def _member_consistent(df: pd.DataFrame) -> pd.Series:
    role = df["role"].astype(str)
    excluded_ok = (role == "excluded") == df["node_id"].isna()
    same = df["node_id"].astype("Int64") == df["territory_id"].astype("Int64")
    self_ok = (role != "self") | same.fillna(False).astype(bool)
    return excluded_ok & self_ok


NODE_MEMBERS = TableSchema(
    name="node_members",
    columns=(
        Col("territory_id", "int32", range=POSITIVE_INT),
        Col("node_id", "int32", nullable=True, range=POSITIVE_INT),
        Col("role", "category", values=MEMBER_ROLES),
        Col("region_code", "int16", range=(1, 99)),
    ),
    key=("territory_id",),
    checks=(Check("excluded ⇔ нет узла; self — узел это само МО", _member_consistent),),
)


def _node_parts_sum(df: pd.DataFrame) -> pd.Series:
    parts = df[[f"v_{p}" for p in PARTS]].sum(axis=1)
    return (parts - df["v_all"]).abs() <= SUM_TOL * df["v_all"].abs()


NODE_PANEL = TableSchema(
    name="node_panel",
    columns=(
        Col("territory_id", "int32", range=POSITIVE_INT),
        *_MONTH_COLS,
        *(Col(f"v_{c}", "float64", range=(0.0, INF)) for c in CATEGORY_CODES),
        *(Col(f"sh_{p}", "float64", range=SHARE) for p in PARTS),
        Col("log_all", "float64"),
        Col("n_members", "int16", range=(1, MAX_MEMBERS)),
        Col("pop_coverage", "float64", range=SHARE),
    ),
    key=("territory_id", "date"),
    checks=(
        Check("месяц согласован: date, t, year, month", _month_consistent),
        Check("шесть долей в сумме 1", _shares_sum_to_one),
        Check("шесть частей в сумме дают v_all (до 1e-9 относительно)", _node_parts_sum),
        Check("log_all = ln(v_all)", _log_all),
    ),
)

NODE_CONTEXT = TableSchema(
    name="node_context",
    columns=(
        *CONTEXT_ANNUAL.columns,
        Col("wage_outlier", "bool"),
        Col("ndfl_outlier", "bool"),
        Col("n_members", "int16", range=(1, MAX_MEMBERS)),
    ),
    key=("territory_id", "year"),
    extra_allowed=True,
    checks=CONTEXT_ANNUAL.checks,
)

NODE_OKVED = TableSchema(name="node_okved", columns=OKVED_SHARES.columns, key=OKVED_SHARES.key)

# Окна признаков: год, полугодие, квартал (``features.windows``).
WINDOW_KINDS: tuple[str, ...] = ("year", "half", "quarter")
WINDOW_PATTERN = r"\d{4}(H[12]|Q[1-4])?"
CLR_TOL = 1e-9  # строка CLR в сумме 0 (float64)


def _clr_rows_zero(prefix: str) -> Callable[[pd.DataFrame], pd.Series]:
    def check(df: pd.DataFrame) -> pd.Series:
        return df[[f"{prefix}_{p}" for p in PARTS]].sum(axis=1).abs() <= CLR_TOL

    return check


FEATURES_MEMBERS = TableSchema(
    name="features_members",
    columns=NODE_MEMBERS.columns,
    key=NODE_MEMBERS.key,
    checks=NODE_MEMBERS.checks,
)

FEATURES_NODES = TableSchema(
    name="features_nodes",
    columns=(
        *NODES.columns,
        Col("is_node", "bool"),
        Col("drop_reason", "string", nullable=True),
    ),
    key=("territory_id",),
    checks=(
        *NODES.checks,
        Check("причина указана ровно у не-узлов", lambda d: d["is_node"] == d["drop_reason"].isna()),
    ),
)

FEATURES_WINDOWS = TableSchema(
    name="features_windows",
    columns=(
        Col("territory_id", "int32", range=POSITIVE_INT),
        Col("window", "string", pattern=WINDOW_PATTERN),
        Col("window_kind", "category", values=WINDOW_KINDS, ordered=True),
        Col("year", "int16", range=_YEAR_RANGE),
        Col("n_months", "int8", range=(1, 12)),
        Col("region_group", "int16", range=(1, 99)),
        Col("level", "float64", range=(0.0, INF)),
        Col("log_level", "float64"),
        Col("log_level_rel", "float64"),
        *(Col(f"sh_{p}", "float64", range=SHARE) for p in PARTS),
        *(Col(f"clr_{p}", "float64") for p in PARTS),
        *(Col(f"clr_rel_{p}", "float64") for p in PARTS),
        Col("mp_pp_yoy", "float64", nullable=True),
    ),
    key=("territory_id", "window"),
    checks=(
        Check("шесть долей в сумме 1", _shares_sum_to_one),
        Check("CLR строки в сумме 0", _clr_rows_zero("clr")),
        Check("CLR относительно группы региона в сумме 0", _clr_rows_zero("clr_rel")),
    ),
)

FEATURES_RHYTHM = TableSchema(
    name="features_rhythm",
    columns=(
        Col("territory_id", "int32", range=POSITIVE_INT),
        Col("summer_excess", "float64"),
        Col("dec_peak", "float64"),
        Col("own_summer", "float64"),
        Col("own_amplitude", "float64", range=NONNEG),
        Col("own_r", "float64", nullable=True, range=(-1.0, 1.0)),
        Col("own_reliable", "bool"),
        Col("own_amplitude_shrunk", "float64", nullable=True, range=NONNEG),
        Col("own_peak_month", "int8", range=(1, 12)),
        Col("north", "bool"),
    ),
    key=("territory_id",),
)

FEATURES_PLACE = TableSchema(
    name="features_place",
    columns=(
        Col("territory_id", "int32", range=POSITIVE_INT),
        Col("year", "int16", range=_YEAR_RANGE),
        Col("region_group", "int16", range=(1, 99)),
        *(_f(f"emp_sh_{g}", SHARE) for g in OKVED_GROUPS),
        _f("emp_sh_unallocated", SHARE),
        _f("wage"),
        _f("log_wage_rel", None),
        Col("wage_outlier", "bool"),
        _f("urban_share", SHARE),
        _f("urban_share_rel", None),
        _f("age_old_share", SHARE),
        _f("age_old_share_rel", None),
        _f("pop_avg", (1.0, INF)),
        _f("log_pop", None),
        _f("log_pop_rel", None),
        _f("market_access", (0.0, 1000.0)),
        _f("market_access_rel", None),
        _f("dist_capital_km"),
        Col("north", "bool"),
        Col("ndfl_ok", "bool"),
        _f("ndfl_income_pc"),
        _f("log_ndfl_rel", None),
    ),
    key=("territory_id", "year"),
    checks=(
        Check(
            "доход 5-НДФЛ на жителя только при ndfl_ok", lambda d: d["ndfl_ok"] | d["ndfl_income_pc"].isna()
        ),
    ),
)

FEATURES_BASKET_MONTHLY = TableSchema(
    name="features_basket_monthly",
    columns=(
        Col("territory_id", "int32", range=POSITIVE_INT),
        *_MONTH_COLS,
        Col("region_group", "int16", range=(1, 99)),
        *(Col(f"clr_{p}", "float64") for p in PARTS),
        *(Col(f"clr_rel_{p}", "float64") for p in PARTS),
        Col("n_replaced", "int8", range=(0, len(PARTS) - 1)),
    ),
    key=("territory_id", "date"),
    checks=(
        Check("месяц согласован: date, t, year, month", _month_consistent),
        Check("CLR строки в сумме 0", _clr_rows_zero("clr")),
        Check("CLR относительно группы региона в сумме 0", _clr_rows_zero("clr_rel")),
    ),
)

FEATURES_RHYTHM_MONTHLY = TableSchema(
    name="features_rhythm_monthly",
    columns=(
        Col("territory_id", "int32", range=POSITIVE_INT),
        Col("category", "category", values=CATEGORY_CODES, ordered=True),
        *_MONTH_COLS,
        Col("log_value", "float64"),
        Col("detrended", "float64"),
        Col("own", "float64"),
    ),
    key=("territory_id", "category", "date"),
    checks=(Check("месяц согласован: date, t, year, month", _month_consistent),),
)

NODE_SCHEMAS: dict[str, TableSchema] = {
    s.name: s for s in (NODES, NODE_MEMBERS, NODE_PANEL, NODE_CONTEXT, NODE_OKVED)
}
FEATURE_SCHEMAS: dict[str, TableSchema] = {
    s.name: s
    for s in (
        FEATURES_NODES,
        FEATURES_MEMBERS,
        FEATURES_WINDOWS,
        FEATURES_RHYTHM,
        FEATURES_PLACE,
        FEATURES_BASKET_MONTHLY,
        FEATURES_RHYTHM_MONTHLY,
    )
}

# --- Схемы выходов этапа network ----------------------------------------------------------------
# Рёбра — неориентированные пары узлов: source < target (territory_id). Правила и способы разрежения —
# секция ``network`` конфига; формулы — ``munnet.network.rules`` и docs/network.md.

SPARSIFY_METHODS: tuple[str, ...] = ("knn", "mutual_knn", "threshold")
NETWORK_WINDOW_KINDS: tuple[str, ...] = ("quarter", "half", "rolling")
RULE_NAME_PATTERN = r"[a-z][a-z0-9_]*"


def _source_lt_target(df: pd.DataFrame) -> pd.Series:
    return df["source"].astype("int64") < df["target"].astype("int64")


# ``weight`` — сходство S_ij правила; у гравитации это ln S_ij = ln P_i + ln P_j − β·ln d (логарифм: сами
# значения P_i·P_j/d^β различаются на много порядков), у остальных правил — само S_ij.
NETWORK_EDGES = TableSchema(
    name="network_edges",
    columns=(
        Col("rule", "string", pattern=RULE_NAME_PATTERN),
        Col("sparsify", "category", values=SPARSIFY_METHODS),
        Col("k", "int16", range=(1, 999)),
        Col("source", "int32", range=POSITIVE_INT),
        Col("target", "int32", range=POSITIVE_INT),
        Col("weight", "float64"),
        Col("lag", "int8", nullable=True, range=(-12, 12)),
        Col("q_value", "float64", nullable=True, range=SHARE),
        Col("same_region", "bool"),
        Col("dist_km", "float64", range=NONNEG),
        Col("is_main", "bool"),
    ),
    key=("rule", "sparsify", "k", "source", "target"),
    checks=(Check("source < target: ребро неориентированное, петель нет", _source_lt_target),),
)

NETWORK_WINDOWS = TableSchema(
    name="network_windows",
    columns=(
        Col("rule", "string", pattern=RULE_NAME_PATTERN),
        Col("window_kind", "category", values=NETWORK_WINDOW_KINDS),
        Col("window", "string"),
        Col("n_months", "int8", range=(1, N_MONTHS)),
        Col("source", "int32", range=POSITIVE_INT),
        Col("target", "int32", range=POSITIVE_INT),
        Col("weight", "float64"),
        Col("is_main_kind", "bool"),
    ),
    key=("rule", "window_kind", "window", "source", "target"),
    checks=(Check("source < target: ребро неориентированное, петель нет", _source_lt_target),),
)

NETWORK_WINDOW_NODES = TableSchema(
    name="network_window_nodes",
    columns=(
        Col("window_kind", "category", values=NETWORK_WINDOW_KINDS),
        Col("window", "string"),
        Col("territory_id", "int32", range=POSITIVE_INT),
        Col("n_months", "int8", range=(1, N_MONTHS)),
        Col("first_date", "string", pattern=DATE_PATTERN),
        Col("last_date", "string", pattern=DATE_PATTERN),
        *(Col(f"clr_rel_{p}", "float64") for p in PARTS),
    ),
    key=("window_kind", "window", "territory_id"),
    checks=(Check("CLR относительно группы региона в сумме 0", _clr_rows_zero("clr_rel")),),
)

NETWORK_NODES = TableSchema(
    name="network_nodes",
    columns=(
        Col("territory_id", "int32", range=POSITIVE_INT),
        Col("rule", "string", pattern=RULE_NAME_PATTERN),
        Col("degree", "int32", range=(0, INT32_MAX)),
        Col("kocc", "int32", range=(0, INT32_MAX)),
        Col("community", "int32", range=(0, INT32_MAX)),
        Col("is_city_node", "bool"),
    ),
    key=("territory_id", "rule"),
)

NETWORK_SCHEMAS: dict[str, TableSchema] = {
    s.name: s for s in (NETWORK_EDGES, NETWORK_WINDOWS, NETWORK_WINDOW_NODES, NETWORK_NODES)
}

# --- Схемы выходов этапа cluster ----------------------------------------------------------------
# Кандидат — пара «метод, K» (``<метод>_k<K>``); метка −1 — шум HDBSCAN; метки кандидата 0…K−1 по убыванию
# размера кластера. Итоговый тип узла — 1…K (``cluster_final``).

CANDIDATE_PATTERN = r"[a-z][a-z_]*_k\d{2}"


def _final_consistent(df: pd.DataFrame) -> pd.Series:
    return (df["type"] >= 1) & (df["type"] <= df["k"])


CLUSTER_LABELS = TableSchema(
    name="cluster_labels",
    columns=(
        Col("candidate", "string", pattern=CANDIDATE_PATTERN),
        Col("territory_id", "int32", range=POSITIVE_INT),
        Col("method", "string", pattern=RULE_NAME_PATTERN),
        Col("k", "int16", range=(0, 999)),
        Col("param", "float64", nullable=True),
        Col("label", "int16", range=(-1, 998)),
        Col("feasible", "bool"),
        Col("is_method_winner", "bool"),
        Col("is_final", "bool"),
    ),
    key=("candidate", "territory_id"),
)

CLUSTER_FINAL = TableSchema(
    name="cluster_final",
    columns=(
        Col("territory_id", "int32", range=POSITIVE_INT),
        Col("type", "int16", range=(1, 999)),
        Col("k", "int16", range=(1, 999)),
        Col("method", "string", pattern=RULE_NAME_PATTERN),
        Col("candidate", "string", pattern=CANDIDATE_PATTERN),
        Col("type_jaccard", "float64", range=SHARE),
        Col("is_city_node", "bool"),
    ),
    key=("territory_id",),
    checks=(Check("тип в 1…K", _final_consistent),),
)

CLUSTER_SCHEMAS: dict[str, TableSchema] = {s.name: s for s in (CLUSTER_LABELS, CLUSTER_FINAL)}
