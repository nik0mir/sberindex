"""Узлы сети: МО панели и узлы-города Москва и Петербург (PLAN.md, этап 1 «Выбрать сюжет» и этап 2).

Режим ``nodes.mode`` конфига:

- ``collapse`` (решение 26.09) — 247 внутригородских территорий сворачиваются в два узла-города, остальные
  МО — сами себе узлы. Районы ведут себя как одна экономика: свой ритм районов одного города совпадает
  на 0,86–0,88, на них 74% коротких дорожных рёбер (проверка 26.09);
- ``separate`` — все МО панели отдельными узлами (проверка чувствительности);
- ``exclude`` — без внутригородских территорий (они остаются в ``node_members`` с ролью ``excluded``).

Правила свёртки (``nodes`` конфига):

- **траты** по категориям в месяце — среднее районов с весами населения ``pop_avg`` своего года (нет — другого
  года), только районы с данными этого месяца; ``pop_coverage`` — доля населения города у этих районов;
- **население** — сумма; **потоки Росстата** (работники, фонд оплаты, розница, отгрузка…) — суммы;
  **зарплата** — среднее с весами работников; **доли жителей** (горожане, возраст) — средние с весами
  населения; **доли занятости** — средние с весами работников (скрытый в районе раздел — не раскрыт
  и в городе); показатели «на жителя» пересчитываются из сумм так же, как на этапе panel. Показатель города
  считается, только если он есть у районов с долей населения не меньше ``min_pop_coverage``, иначе пропуск:
  частичная сумма — не город;
- **5-НДФЛ** — доход — сумма районов, получатели — строка «Субъект РФ» (``city_context`` этапа panel):
  сумма по районам считает людей дважды. Нет строки — пропуск и ``ndfl_ok`` = false; ``ndfl_ok`` — то же
  правило отношения получателей к жителям, что на этапе panel, и не выброс;
- **выбросы** (``nodes.cities.<код>.outliers``): ``wage`` → флаг ``wage_outlier``, ``ndfl`` → ``ndfl_outlier``
  и доход 5-НДФЛ на жителя не считается (у Москвы оба — зарплата и доход по месту работодателя);
- **доступность рынков, расстояние до столицы, точка** — средние районов с весами населения, площадь — сумма;
- **группа региона** ``region_group``: у города — его область (Москва + Московская область, Петербург +
  Ленинградская), у остальных узлов — свой регион. Всё «относительно региона» считается по группе: иначе
  у города, единственного узла своего субъекта, остаток тождественно нулевой.

``territory_id`` узла-города — ``nodes.city_id_base`` + код субъекта (90077, 90078): таких номеров нет
в справочнике СберИндекса (1…3101), а код субъекта виден в номере. Связь МО панели с узлами —
``node_members``. Каждое соединение — ``merge(..., validate=...)`` с числом строк в логе; выходы
проверяются контрактами ``munnet.contracts``: ``NODES``, ``NODE_MEMBERS``, ``NODE_PANEL``, ``NODE_CONTEXT``,
``NODE_OKVED``.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from munnet.config import Config
from munnet.contracts import (
    CATEGORY_CODES,
    CITY_CONTEXT,
    CITY_MO_TYPE,
    CONTEXT_ANNUAL,
    INT32_MAX,
    NODE_CONTEXT,
    NODE_MEMBERS,
    NODE_MO_TYPES,
    NODE_MODES,
    NODE_OKVED,
    NODE_PANEL,
    NODES,
    OKVED_GROUPS,
    OKVED_SECTIONS,
    OKVED_SHARES,
    PANEL_WIDE,
    PARTS,
    TERRITORIES,
    YEARS,
    MissingInputError,
    coerce,
    read_table,
    validate,
)
from munnet.panel.consumption import series_quality
from munnet.panel.context import (
    KRUB_TO_RUB,
    MONTHS_PER_YEAR,
    NDFL_OK_NEIGHBOUR,
    PER_CAPITA_KRUB,
    PER_THOUSAND,
    UNALLOCATED,
)

log = logging.getLogger(__name__)

OUTLIER_KINDS: tuple[str, ...] = ("wage", "ndfl")
# Режим узлов словами — для отчётов (``docs/eda.md``, этап features).
MODE_TEXT: dict[str, str] = {
    "collapse": "Москва и Петербург — два узла-города, остальные МО — сами себе узлы",
    "exclude": "без внутригородских территорий Москвы и Петербурга",
    "separate": "все МО панели отдельными узлами",
}
V_COLS: tuple[str, ...] = tuple(f"v_{c}" for c in CATEGORY_CODES)
# Потоки и запасы Росстата и ФНС, которые у города складываются по районам.
SUM_COLUMNS: tuple[str, ...] = (
    "employees",
    "payroll_krub",
    "retail_krub",
    "catering_turnover_krub",
    "shipments_krub",
    "orgs",
    "ip",
    "nights",
    "beds",
    "invest_krub",
    "ndfl_income_rub",
)
# Доли жителей: средние районов с весами населения на 1 января.
POP_SHARE_COLUMNS: tuple[str, ...] = ("urban_share", "age_young_share", "age_working_share", "age_old_share")
PANEL_HINT = "сначала запустите этап panel: python -m munnet panel"


# --- Параметры -----------------------------------------------------------------------------------


@dataclass(frozen=True)
class CitySpec:
    """Узел-город: код субъекта, название, ОКТМО города целиком, код области для группы региона, выбросы."""

    region_code: int
    name: str
    oktmo: str
    region_group: int
    outliers: tuple[str, ...]


@dataclass(frozen=True)
class NodeParams:
    """Параметры узлов из секции ``nodes`` конфига и порог 5-НДФЛ ``context.ndfl.recipients_ratio_ok``."""

    mode: str
    city_id_base: int
    min_pop_coverage: float
    cities: Mapping[int, CitySpec]
    recipients_ratio_ok: tuple[float, float]

    @classmethod
    def from_section(cls, section: Mapping[str, Any], recipients_ratio_ok: Any) -> NodeParams:
        """Проверяет секцию: режим из ``NODE_MODES``, коды субъектов 1…99, известные виды выбросов, доля
        покрытия в (0; 1]; номера узлов-городов помещаются в int32. Ошибка — ``ValueError``."""
        mode = str(section["mode"])
        problems = []
        if mode not in NODE_MODES:
            problems.append(f"nodes.mode = {mode!r}: допустимы {list(NODE_MODES)}")
        base = int(section["city_id_base"])
        cities = {}
        for code, spec in (section.get("cities") or {}).items():
            code = int(code)
            outliers = tuple(str(x) for x in spec.get("outliers") or ())
            unknown = sorted(set(outliers) - set(OUTLIER_KINDS))
            if unknown:
                problems.append(f"nodes.cities.{code}.outliers: {unknown}, допустимы {list(OUTLIER_KINDS)}")
            if not 1 <= code <= 99:
                problems.append(f"nodes.cities: код субъекта {code} вне 1…99")
            cities[code] = CitySpec(
                region_code=code,
                name=str(spec["name"]),
                oktmo=str(spec["oktmo"]),
                region_group=int(spec.get("region_group", code)),
                outliers=outliers,
            )
        if base <= 0 or base + 99 > INT32_MAX:
            problems.append(f"nodes.city_id_base = {base}: номер узла-города должен помещаться в int32")
        cover = float(section.get("min_pop_coverage", 1.0))
        if not 0 < cover <= 1:
            problems.append(f"nodes.min_pop_coverage = {cover}: нужна доля в (0; 1]")
        if problems:
            raise ValueError("; ".join(problems))
        lo, hi = (float(x) for x in recipients_ratio_ok)
        return cls(mode, base, cover, cities, (lo, hi))

    def node_id(self, region_code: int) -> int:
        return self.city_id_base + int(region_code)


def node_params(cfg: Config) -> NodeParams:
    """``NodeParams`` из конфига проекта."""
    return NodeParams.from_section(cfg["nodes"], cfg["context"]["ndfl"]["recipients_ratio_ok"])


@dataclass(frozen=True)
class NodeData:
    """Таблицы узлов: справочник узлов, связь МО с узлами, траты по месяцам, годовой контекст, доли ОКВЭД2."""

    params: NodeParams
    nodes: pd.DataFrame
    members: pd.DataFrame
    panel_wide: pd.DataFrame
    context_annual: pd.DataFrame
    okved_shares: pd.DataFrame

    @property
    def mode(self) -> str:
        return self.params.mode

    @property
    def city_ids(self) -> list[int]:
        return sorted(int(t) for t in self.nodes.loc[self.nodes["is_city_node"], "territory_id"])


# --- Состав узлов --------------------------------------------------------------------------------


def _city_members(territories: pd.DataFrame, params: NodeParams) -> dict[int, list[int]]:
    """Код субъекта -> внутригородские территории этого города (для режима collapse)."""
    inner = territories.loc[territories["is_inner_city"].astype(bool)]
    out = {}
    for code in params.cities:
        ids = sorted(int(t) for t in inner.loc[inner["region_code"] == code, "territory_id"])
        if ids:
            out[code] = ids
    orphans = inner.loc[~inner["region_code"].isin(list(out)), "territory_id"]
    if len(orphans):
        log.warning(
            "узлы: у %d внутригородских территорий нет узла-города в nodes.cities — остаются узлами",
            len(orphans),
        )
    return out


def members_table(
    territories: pd.DataFrame, params: NodeParams, cities: Mapping[int, list[int]]
) -> pd.DataFrame:
    """``node_members``: МО панели -> узел (``self``), узел-город (``city_member``) или ``excluded``."""
    ter = territories[["territory_id", "region_code", "is_inner_city"]].copy()
    ter["node_id"] = ter["territory_id"].astype("Int32")
    ter["role"] = "self"
    inner = ter["is_inner_city"].astype(bool)
    if params.mode == "collapse":
        for code, ids in cities.items():
            mask = ter["territory_id"].isin(ids)
            ter.loc[mask, "node_id"] = params.node_id(code)
            ter.loc[mask, "role"] = "city_member"
    elif params.mode == "exclude":
        ter.loc[inner, "node_id"] = pd.NA
        ter.loc[inner, "role"] = "excluded"
    out = ter[["territory_id", "node_id", "role", "region_code"]]
    return validate(coerce(out, NODE_MEMBERS), NODE_MEMBERS)


def _weights(context_annual: pd.DataFrame, ids: list[int]) -> pd.DataFrame:
    """Вес района по годам панели: ``pop_avg`` своего года, нет — другого года панели; строки (id, year)."""
    pop = context_annual.loc[context_annual["territory_id"].isin(ids)].pivot(
        index="territory_id", columns="year", values="pop_avg"
    )
    pop = pop.reindex(index=ids, columns=list(YEARS)).astype("float64")
    filled = pop.bfill(axis=1).ffill(axis=1)
    missing = filled.index[filled.isna().any(axis=1)].tolist()
    if missing:
        raise ValueError(f"узлы: у районов {missing[:5]} нет населения ни в одном году — вес не задать")
    n_fill = int((pop.isna() & filled.notna()).sum().sum())
    if n_fill:
        log.warning("узлы: у %d район-лет население взято из другого года панели", n_fill)
    out = filled.stack().rename("w").reset_index()
    out["territory_id"] = out["territory_id"].astype("int32")
    out["year"] = out["year"].astype("int16")
    return out


def _merge(left: pd.DataFrame, right: pd.DataFrame, what: str, **kw: Any) -> pd.DataFrame:
    """``merge`` с проверкой кратности и числом строк до и после в логе."""
    out = left.merge(right, **kw)
    log.info("узлы: соединение «%s»: %d строк -> %d", what, len(left), len(out))
    return out


# --- Траты по месяцам ----------------------------------------------------------------------------


def _as_node_panel(wide: pd.DataFrame) -> pd.DataFrame:
    """Панель МО как панель узлов: траты float64, один участник, покрытие 1."""
    out = wide[["territory_id", "date", "t", "year", "month", *V_COLS]].copy()
    for c in V_COLS:
        out[c] = out[c].astype("float64")
    out["n_members"] = np.int16(1)
    out["pop_coverage"] = 1.0
    return out


def city_panel(wide: pd.DataFrame, weights: pd.DataFrame, node_id: int) -> pd.DataFrame:
    """Помесячные траты города: Σ w·v / Σ w по районам с данными месяца; покрытие — Σ w / Σ w всех районов
    года."""
    w = _merge(
        wide[["territory_id", "date", "t", "year", "month", *V_COLS]],
        weights,
        f"траты районов × население, узел {node_id}",
        on=["territory_id", "year"],
        how="inner",
        validate="many_to_one",
    )
    num = w[list(V_COLS)].astype("float64").mul(w["w"], axis=0)
    keys = ["date", "t", "year", "month"]
    num[keys] = w[keys]
    num["w"] = w["w"]
    num["n_members"] = 1
    g = num.groupby(keys, as_index=False).sum()
    out = g[keys].copy()
    for c in V_COLS:
        out[c] = g[c] / g["w"]
    total = weights.groupby("year")["w"].sum()
    out["pop_coverage"] = (g["w"] / g["year"].map(total)).clip(upper=1.0)
    out["n_members"] = g["n_members"].astype("int16")
    out.insert(0, "territory_id", np.int32(node_id))
    return out


def _finish_panel(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    # Доли от «Все категории» частей, собранных из того же среднего: их сумма — 1 до ошибки округления.
    for p in PARTS:
        out[f"sh_{p}"] = out[f"v_{p}"] / out["v_all"]
    out["log_all"] = np.log(out["v_all"])
    out["territory_id"] = out["territory_id"].astype("int32")
    out["t"] = out["t"].astype("int8")
    out["year"] = out["year"].astype("int16")
    out["month"] = out["month"].astype("int8")
    out["n_members"] = out["n_members"].astype("int16")
    return validate(out[list(NODE_PANEL.names)], NODE_PANEL)


def panel_long(node_panel: pd.DataFrame) -> pd.DataFrame:
    """Длинная панель узлов (как ``panel_long``, траты float64): для ``eda.data.build_national``."""
    keys = ["territory_id", "date", "t", "year", "month"]
    long = node_panel.melt(id_vars=keys, value_vars=list(V_COLS), var_name="category", value_name="value")
    long["category"] = pd.Categorical(
        long["category"].str.slice(2), categories=list(CATEGORY_CODES), ordered=True
    )
    long["is_derived"] = long["category"].astype(str) == "other"
    return long.sort_values(["territory_id", "date", "category"], kind="mergesort").reset_index(drop=True)


# --- Справочник узлов ----------------------------------------------------------------------------


def _wavg(values: pd.Series, weights: pd.Series) -> float:
    ok = values.notna() & weights.notna() & (weights > 0)
    if not ok.any():
        return float("nan")
    return float(np.average(values[ok].astype("float64"), weights=weights[ok].astype("float64")))


def _group_names(nodes: pd.DataFrame, cities: Mapping[int, CitySpec], present: set[int]) -> dict[int, str]:
    """Название группы региона: регион или «Москва и Московская область» для группы с городом."""
    names = nodes.drop_duplicates("region_code").set_index("region_code")["region_name"].astype(str).to_dict()
    out = dict(names)
    for spec in cities.values():
        if (
            spec.region_code in present
            and spec.region_group in names
            and spec.region_group != spec.region_code
        ):
            out[spec.region_group] = f"{names[spec.region_code]} и {names[spec.region_group]}"
    return out


def city_node_row(
    territories: pd.DataFrame, weights: pd.DataFrame, panel: pd.DataFrame, spec: CitySpec, node_id: int
) -> dict[str, Any]:
    """Строка справочника узла-города: точка, координаты, доступность рынков и расстояние — средние районов
    с весами населения первого года панели, площадь — сумма, качество ряда — по месяцам с данными."""
    ids = weights["territory_id"].unique()
    ter = territories.set_index("territory_id").loc[ids]
    w0 = weights.loc[weights["year"] == YEARS[0]].set_index("territory_id")["w"].reindex(ter.index)
    q = series_quality(panel[["territory_id", "date"]]).iloc[0]
    row = {
        "territory_id": node_id,
        "name": spec.name,
        "name_short": spec.name,
        "region_code": spec.region_code,
        "region_name": str(ter["region_name"].iloc[0]),
        "mo_type": CITY_MO_TYPE,
        "mo_status": None,
        "is_capital": False,
        "is_inner_city": False,
        "is_city_node": True,
        "n_members": len(ids),
        "oktmo_2023": spec.oktmo,
        "oktmo_2024": spec.oktmo,
        "area_km2": float(ter["area_km2"].sum()),
        "lineage_role": "none",
    }
    for col in ("point_lat", "point_lon", "x_aea", "y_aea", "market_access", "dist_capital_km"):
        row[col] = _wavg(ter[col], w0)
    for col in (
        "n_months",
        "n_2023",
        "n_2024",
        "first_date",
        "last_date",
        "coverage_pattern",
        "series_status",
    ):
        row[col] = q[col]
    row["has_internal_gap"] = bool(q["has_internal_gap"])
    row["longest_gap"] = int(q["longest_gap"])
    return row


def _finish_nodes(nodes: pd.DataFrame) -> pd.DataFrame:
    out = nodes[list(NODES.names)].copy()
    out["mo_type"] = pd.Categorical(out["mo_type"].astype(str), categories=list(NODE_MO_TYPES))
    return validate(coerce(out, NODES), NODES)


# --- Годовой контекст ----------------------------------------------------------------------------


def _covered(mask: pd.Series, pop: pd.Series, min_cover: float) -> bool:
    """Есть ли показатель (``mask``) у районов с долей населения не меньше ``min_cover``."""
    total = float(pop.sum())
    return total > 0 and float(pop[mask.to_numpy()].sum()) / total >= min_cover - 1e-12


def city_context_rows(
    context_annual: pd.DataFrame,
    city_rows: pd.DataFrame,
    spec: CitySpec,
    ids: list[int],
    node_id: int,
    params: NodeParams,
) -> list[dict[str, Any]]:
    """Годовой контекст узла-города (правила — в описании модуля); строка на год панели."""
    rows = []
    lo, hi = params.recipients_ratio_ok
    cover = params.min_pop_coverage
    for year in YEARS:
        c = context_annual.loc[(context_annual["year"] == year) & context_annual["territory_id"].isin(ids)]
        c = c.set_index("territory_id")
        pop = c["pop_jan1"].astype("float64").fillna(0.0)
        r: dict[str, Any] = {col.name: np.nan for col in CONTEXT_ANNUAL.columns}
        r.update(territory_id=node_id, year=year, n_members=len(ids))
        for col in ("pop_jan1", "pop_avg"):
            r[col] = float(c[col].sum()) if _covered(c[col].notna(), pop, cover) else np.nan
        for col in ("pop_jan1_method", "pop_avg_method"):
            vals = c[col].dropna().astype(str).unique()
            r[col] = vals[0] if len(vals) == 1 else None
        for col in SUM_COLUMNS:
            r[col] = float(c[col].sum()) if _covered(c[col].notna(), pop, cover) else np.nan
        for col in POP_SHARE_COLUMNS:
            r[col] = _wavg(c[col], c["pop_jan1"]) if _covered(c[col].notna(), pop, cover) else np.nan
        wage_ok = _covered(c["wage"].notna() & c["employees"].notna(), pop, cover)
        r["wage"] = _wavg(c["wage"], c["employees"]) if wage_ok else np.nan
        if _covered(c["shipments_mining_share"].notna(), pop, cover) and np.isfinite(r["shipments_krub"]):
            r["shipments_mining_share"] = _wavg(c["shipments_mining_share"], c["shipments_krub"])
        emp_cols = [f"emp_sh_{s}" for s in OKVED_SECTIONS if f"emp_sh_{s}" in c.columns]
        emp_cols += [f"emp_sh_{g}" for g in OKVED_GROUPS] + ["emp_sh_unallocated"]
        has_total = c["emp_sh_unallocated"].notna() & c["employees"].notna()
        emp_ok = _covered(has_total, pop, cover)
        for col in emp_cols:
            if not emp_ok or not c.loc[has_total, col].notna().any():
                r[col] = np.nan
                continue
            share = c.loc[has_total, col].astype("float64").fillna(0.0)  # скрытый раздел района — не раскрыт
            r[col] = _wavg(share, c.loc[has_total, "employees"])
        sections = [f"emp_sh_{s}" for s in OKVED_SECTIONS if f"emp_sh_{s}" in c.columns]
        r["emp_n_disclosed"] = int(sum(np.isfinite(r[s]) for s in sections)) if emp_ok else None
        r["emp_sh_renormalized"] = bool(c.get("emp_sh_renormalized", pd.Series(False)).fillna(False).any())
        rec = city_rows.loc[
            (city_rows["region_code"] == spec.region_code)
            & (city_rows["year"] == year)
            & (city_rows["indicator"] == "ndfl_recipients"),
            "value",
        ]
        r["ndfl_recipients"] = float(rec.iloc[0]) if len(rec) else np.nan
        avg = r["pop_avg"]
        for pc_col, src in PER_CAPITA_KRUB.items():
            r[pc_col] = r[src] * KRUB_TO_RUB / avg / MONTHS_PER_YEAR
        ratio = r["ndfl_recipients"] / avg
        r["recipients_to_pop"] = ratio
        r["wage_outlier"] = "wage" in spec.outliers
        r["ndfl_outlier"] = "ndfl" in spec.outliers
        in_range = bool(np.isfinite(ratio) and lo <= ratio <= hi)
        r["ndfl_ok"] = in_range and not r["ndfl_outlier"]
        r["ndfl_income_pc"] = r["ndfl_income_rub"] / avg / MONTHS_PER_YEAR if r["ndfl_ok"] else np.nan
        few = bool(np.isfinite(ratio) and ratio < lo)
        per_rec = r["ndfl_income_rub"] / r["ndfl_recipients"] / MONTHS_PER_YEAR
        r["ndfl_income_per_recipient"] = np.nan if few else per_rec
        r["employees_to_working_age"] = r["employees"] / (r["pop_jan1"] * r["age_working_share"])
        for col in ("orgs", "ip", "beds"):
            r[f"{col}_per_1000"] = r[col] / avg * PER_THOUSAND
        r["nights_pc"] = r["nights"] / avg
        r["workplace_based"] = False
        r[NDFL_OK_NEIGHBOUR] = False
        rows.append(r)
    return rows


def _finish_context(frame: pd.DataFrame) -> pd.DataFrame:
    out = coerce(frame, NODE_CONTEXT)
    for col in (*CONTEXT_ANNUAL.names, "wage_outlier", "ndfl_outlier", "n_members"):
        if col not in out.columns:
            out[col] = np.nan
    floats = out.select_dtypes("float64").columns
    out[floats] = out[floats].where(np.isfinite(out[floats]))  # деление на ноль -> пропуск
    return validate(coerce(out, NODE_CONTEXT), NODE_CONTEXT)


def city_okved(okved: pd.DataFrame, ids: list[int], node_id: int) -> pd.DataFrame:
    """Доли разделов ОКВЭД2 города: Σ работников раздела по районам / Σ итогов районов; раздел, скрытый
    во всех районах, — пропуск; «не раскрыто» — то же по строкам ``unallocated``; в сумме по городу — 1."""
    part = okved.loc[okved["territory_id"].isin(ids)].copy()
    part["total"] = part.groupby(["territory_id", "year"])["employees"].transform(
        lambda s: s.sum(min_count=1)
    )
    rows = []
    for year, g in part.groupby("year"):
        totals = g.drop_duplicates("territory_id")["total"].sum()
        for section, s in g.groupby("section"):
            emp = s["employees"].sum(min_count=1)
            share = emp / totals if totals > 0 and pd.notna(emp) else np.nan
            rows.append((node_id, year, section, emp, share))
    out = pd.DataFrame(rows, columns=["territory_id", "year", "section", "employees", "share"])
    return out


# --- Сборка --------------------------------------------------------------------------------------


def build_nodes(
    panel_wide: pd.DataFrame,
    territories: pd.DataFrame,
    context_annual: pd.DataFrame,
    okved_shares: pd.DataFrame,
    city_rows: pd.DataFrame,
    params: NodeParams,
) -> NodeData:
    """Таблицы узлов по режиму ``params.mode`` (правила — в описании модуля); выходы проверены контрактами."""
    cities = _city_members(territories, params) if params.mode == "collapse" else {}
    members = members_table(territories, params, cities)
    keep = members.loc[members["role"].astype(str) == "self", "territory_id"]
    base_ter = territories.loc[territories["territory_id"].isin(keep)].copy()
    base_ter["is_city_node"] = False
    base_ter["n_members"] = 1
    base_ter["region_group"] = base_ter["region_code"]

    panels = [_as_node_panel(panel_wide.loc[panel_wide["territory_id"].isin(keep)])]
    node_rows, ctx_rows, okved_parts = [], [], []
    for code, ids in cities.items():
        spec = params.cities[code]
        node_id = params.node_id(code)
        weights = _weights(context_annual, ids)
        panel = city_panel(panel_wide.loc[panel_wide["territory_id"].isin(ids)], weights, node_id)
        panels.append(panel)
        node_rows.append(city_node_row(territories, weights, panel, spec, node_id))
        ctx_rows.extend(city_context_rows(context_annual, city_rows, spec, ids, node_id, params))
        okved_parts.append(city_okved(okved_shares, ids, node_id))
        log.info("узел-город %s (%d): %d районов", spec.name, node_id, len(ids))

    nodes = pd.concat([base_ter, pd.DataFrame(node_rows)], ignore_index=True) if node_rows else base_ter
    present = set(int(c) for c in territories["region_code"].unique())
    for spec in params.cities.values():
        node = params.node_id(spec.region_code)
        if node not in set(nodes["territory_id"]):
            continue
        if spec.region_group in present:
            nodes.loc[nodes["territory_id"] == node, "region_group"] = spec.region_group
        else:
            nodes.loc[nodes["territory_id"] == node, "region_group"] = spec.region_code
            log.warning(
                "узлы: области %d (группа региона города %s) нет в панели — город остаётся в своей группе",
                spec.region_group,
                spec.name,
            )
    names = _group_names(nodes, params.cities, set(cities))
    nodes["region_group_name"] = nodes["region_group"].astype(int).map(names)
    nodes = _finish_nodes(nodes)

    base_ctx = context_annual.loc[context_annual["territory_id"].isin(keep)].copy()
    base_ctx["wage_outlier"] = False
    base_ctx["ndfl_outlier"] = False
    base_ctx["n_members"] = 1
    context = pd.concat([base_ctx, pd.DataFrame(ctx_rows)], ignore_index=True) if ctx_rows else base_ctx
    okved = pd.concat(
        [okved_shares.loc[okved_shares["territory_id"].isin(keep)], *okved_parts], ignore_index=True
    )
    okved = okved.astype({"territory_id": "int32", "year": "int16", "section": "str"})
    okved.loc[okved["section"] == UNALLOCATED, "section"] = UNALLOCATED
    data = NodeData(
        params=params,
        nodes=nodes,
        members=members,
        panel_wide=_finish_panel(pd.concat(panels, ignore_index=True)),
        context_annual=_finish_context(context),
        okved_shares=validate(okved, NODE_OKVED),
    )
    log.info(
        "узлы (%s): %d узлов, из них узлов-городов %d; МО панели %d, исключено %d",
        params.mode,
        len(nodes),
        int(nodes["is_city_node"].sum()),
        len(members),
        int((members["role"].astype(str) == "excluded").sum()),
    )
    return data


def group_text(data: NodeData) -> str:
    """Как считается «относительно региона»: группы регионов с узлами-городами словами."""
    groups = data.nodes.loc[data.nodes["is_city_node"], "region_group_name"].astype(str).tolist()
    if not groups:
        return "по своему региону"
    names = ", ".join(f"«{g}»" for g in groups)
    return f"по группе региона: города вместе со своими областями ({names})"


def load_node_data(cfg: Config) -> NodeData:
    """Читает выходы этапа panel и строит узлы по ``nodes`` конфига."""
    processed = Path(cfg["paths"]["processed"])
    tables = {}
    for name, schema in (
        ("panel_wide", PANEL_WIDE),
        ("territories", TERRITORIES),
        ("context_annual", CONTEXT_ANNUAL),
        ("okved_shares", OKVED_SHARES),
        ("city_context", CITY_CONTEXT),
    ):
        path = processed / f"{name}.parquet"
        if not path.exists():
            raise MissingInputError(f"нет {path}: {PANEL_HINT}")
        tables[name] = read_table(path, schema)
    return build_nodes(
        tables["panel_wide"],
        tables["territories"],
        tables["context_annual"],
        tables["okved_shares"],
        tables["city_context"],
        node_params(cfg),
    )
