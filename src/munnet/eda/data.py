"""Вход разведки: таблицы этапа panel и две общие производные таблицы (spec_final, Б.2).

``load(cfg)`` читает ``data/processed`` и ``outputs/panel`` (разведка панель не пересобирает) и строит:

- ``mo`` — строка на МО панели: уровни, доли и CLR по годам, рост, сезонные показатели (после снятия
  линейного тренда МО), контекст 2023 года;
- ``national`` — месяц × категория: медианы и квартили трат, медиана долей по МО («типичное МО») и доля
  в тратах всех жителей (веса — среднегодовое население).

Разделы берут показатели отсюда и не пересчитывают их по-своему.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd

from munnet import nodes
from munnet.config import Config
from munnet.contracts import (
    CATEGORY_CODES,
    CITY_CONTEXT,
    CONTEXT_ANNUAL,
    CONTEXT_LONG,
    N_MONTHS,
    OKVED_GROUPS,
    OKVED_SHARES,
    PANEL_LONG,
    PANEL_WIDE,
    PARTS,
    TERRITORIES,
    YEARS,
    MissingInputError,
    read_table,
)
from munnet.eda import stats

log = logging.getLogger(__name__)

PANEL_HINT = "сначала запустите этап panel: python -m munnet panel"
MONTHS_IN_YEAR = 12
DECEMBER = 12

# Колонки справочника территорий, которые разведка берёт в ``mo`` (Б.2).
TERRITORY_COLUMNS: tuple[str, ...] = (
    "name",
    "region_name",
    "region_code",
    "mo_type",
    "is_capital",
    "is_inner_city",
    "point_lat",
    "point_lon",
    "x_aea",
    "y_aea",
    "area_km2",
    "series_status",
    "n_months",
    "n_2023",
    "n_2024",
    "market_access",
    "dist_capital_km",
    "lineage_role",
)
# Контекст 2023 года в ``mo``: колонка context_annual -> колонка mo без суффикса года.
CONTEXT_2023: tuple[str, ...] = (
    "urban_share",
    "age_old_share",
    "emp_sh_A",
    "emp_sh_B",
    "emp_sh_C",
    *(f"emp_sh_{g}" for g in OKVED_GROUPS),
    "emp_sh_unallocated",
    "nights_pc",
    "recipients_to_pop",
)
CONTEXT_YEAR = YEARS[0]  # год контекста разведки (2023): у 2024 года покрытие Росстата меньше (А.11)
# Строковые колонки ``outputs/panel/unmatched.csv``: без этого ОКТМО с ведущим нулём (01512000) читается
# числом 1512000.0.
UNMATCHED_STR: dict[str, type] = {
    "indicator": str,
    "reason": str,
    "oktmo": str,
    "name": str,
    "region_name": str,
}


@dataclass(frozen=True)
class EdaData:
    """Все входы разведки. Разделы читают только это, файлы сами не открывают."""

    panel_long: pd.DataFrame
    panel_wide: pd.DataFrame
    territories: pd.DataFrame
    context_long: pd.DataFrame
    context_annual: pd.DataFrame
    okved_shares: pd.DataFrame
    controls: dict
    unmatched: pd.DataFrame
    mo: pd.DataFrame
    national: pd.DataFrame
    geo_path: Path
    # 5-НДФЛ городов федерального значения целиком (этап panel, для узлов-городов); пусто — нет файла.
    city_context: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=list(CITY_CONTEXT.names)))


def _require(path: Path) -> Path:
    if not path.exists():
        raise MissingInputError(f"нет {path}: {PANEL_HINT}")
    return path


def load(cfg: Config) -> EdaData:
    """Читает выходы этапа panel по контрактам и строит ``mo`` и ``national``.

    Нет ``data/processed`` или нужного файла — ``MissingInputError`` (подкласс ``FileNotFoundError``)
    с подсказкой запустить этап panel.
    """
    processed = Path(cfg["paths"]["processed"])
    if not processed.is_dir():
        raise MissingInputError(f"нет каталога {processed}: {PANEL_HINT}")
    panel_out = Path(cfg["paths"]["outputs"]) / "panel"
    tables = {
        "panel_long": read_table(_require(processed / "panel_long.parquet"), PANEL_LONG),
        "panel_wide": read_table(_require(processed / "panel_wide.parquet"), PANEL_WIDE),
        "territories": read_table(_require(processed / "territories.parquet"), TERRITORIES),
        "context_long": read_table(_require(processed / "context_long.parquet"), CONTEXT_LONG),
        "context_annual": read_table(_require(processed / "context_annual.parquet"), CONTEXT_ANNUAL),
        "okved_shares": read_table(_require(processed / "okved_shares.parquet"), OKVED_SHARES),
    }
    geo_path = _require(processed / "territories_geo.parquet")
    city_path = processed / "city_context.parquet"
    if city_path.exists():
        city = read_table(city_path, CITY_CONTEXT)
    else:
        log.warning("нет %s: у узлов-городов не будет числа получателей 5-НДФЛ (%s)", city_path, PANEL_HINT)
        city = pd.DataFrame(columns=list(CITY_CONTEXT.names))
    with open(_require(panel_out / "controls.json"), encoding="utf-8") as f:
        controls = json.load(f)
    unmatched = pd.read_csv(_require(panel_out / "unmatched.csv"), dtype=UNMATCHED_STR)
    eda = cfg["eda"]
    mo = build_mo(
        tables["panel_wide"],
        tables["territories"],
        tables["context_annual"],
        summer_months=eda["summer_months"],
        rel_min_months=eda["level_rel_min_months"],
    )
    national = build_national(tables["panel_long"], tables["panel_wide"], tables["context_annual"])
    return EdaData(
        **tables,
        controls=controls,
        unmatched=unmatched,
        mo=mo,
        national=national,
        geo_path=geo_path,
        city_context=city,
    )


def node_view(data: EdaData, cfg: Config) -> tuple[EdaData, nodes.NodeData]:
    """Те же входы разведки, но на узлах сети (``nodes.mode`` конфига): для матрицы сюжетов и прототипа.

    Панель, справочник, контекст и доли ОКВЭД2 — таблицы ``munnet.nodes``; ``mo`` и ``national`` строятся теми
    же функциями, что и для МО. Регион в ``mo`` и ``territories`` — группа региона узла (``region_group``:
    Москва вместе с Московской областью, Петербург — с Ленинградской), иначе у узла-города всё «относительно
    региона» тождественно нулю. Значения с флагом выброса (зарплата и доход 5-НДФЛ Москвы) в ``mo`` — пропуск:
    критерии сюжетов их не видят. ``panel_long`` — траты float64, ``context_long``, ``unmatched`` и полигоны —
    как у МО (узлам-городам полигонов нет).
    """
    nd = nodes.build_nodes(
        data.panel_wide,
        data.territories,
        data.context_annual,
        data.okved_shares,
        data.city_context,
        nodes.node_params(cfg),
    )
    ter = nd.nodes.copy()
    ter["region_code"] = ter["region_group"]
    ter["region_name"] = ter["region_group_name"]
    ctx = nd.context_annual.copy()
    ctx["wage"] = ctx["wage"].where(~ctx["wage_outlier"].astype(bool))
    eda = cfg["eda"]
    mo = build_mo(
        nd.panel_wide,
        ter,
        ctx,
        summer_months=eda["summer_months"],
        rel_min_months=eda["level_rel_min_months"],
    )
    long = nodes.panel_long(nd.panel_wide)
    national = build_national(long, nd.panel_wide, ctx)
    view = replace(
        data,
        panel_long=long,
        panel_wide=nd.panel_wide,
        territories=ter,
        context_annual=ctx,
        okved_shares=nd.okved_shares,
        mo=mo,
        national=national,
    )
    return view, nd


# --- mo ------------------------------------------------------------------------------------------


def _year_levels(wide: pd.DataFrame, year: int) -> pd.DataFrame:
    """Уровень, доли и CLR года для МО с полным годом (12 месяцев)."""
    part = wide.loc[wide["year"] == year]
    g = part.groupby("territory_id")
    n = g.size()
    full = n.index[n == MONTHS_IN_YEAR]
    sums = g[[f"v_{c}" for c in CATEGORY_CODES]].sum().loc[full].astype("float64")
    out = pd.DataFrame(index=n.index)
    out[f"level_{year}"] = (sums["v_all"] / MONTHS_IN_YEAR).reindex(n.index)
    shares = pd.DataFrame({p: sums[f"v_{p}"] / sums["v_all"] for p in PARTS})
    for p in PARTS:
        out[f"sh_{p}_{year}"] = shares[p].reindex(n.index)
    clr = stats.clr(shares) if len(shares) else shares
    for p in PARTS:
        out[f"clr_{p}_{year}"] = clr[p].reindex(n.index) if len(clr) else np.nan
    return out


def _level_rel(wide: pd.DataFrame, year: int, min_months: int) -> pd.Series:
    """exp(среднего ln(v_all / медиана v_all всех МО месяца)) по месяцам года при ≥ ``min_months`` месяцах."""
    med = wide.groupby("date")["v_all"].transform("median")
    rel = np.log(wide["v_all"].astype("float64") / med)
    part = pd.DataFrame({"territory_id": wide["territory_id"], "year": wide["year"], "rel": rel})
    part = part.loc[part["year"] == year]
    g = part.groupby("territory_id")["rel"]
    mean, n = g.mean(), g.size()
    return np.exp(mean.where(n >= min_months))


def detrended_log(wide: pd.DataFrame) -> pd.DataFrame:
    """ln ``v_all`` без линейного тренда МО: остатки МНК ``ln v_it = a_i + b_i·t`` по всем 24 месяцам.

    Строки — МО со всеми месяцами панели, колонки — ``t`` 0…23. Так же тренд снимает раздел E3
    (``s3_rhythm.detrend``): сезонные показатели ``mo`` и свой ритм E3 считаются от одного ряда.
    """
    Y = wide.pivot(index="territory_id", columns="t", values="log_all").astype("float64")
    Y = Y.reindex(columns=range(N_MONTHS)).dropna()
    t = np.arange(N_MONTHS, dtype="float64")
    X = np.column_stack([np.ones_like(t), t])
    beta, *_ = np.linalg.lstsq(X, Y.to_numpy().T, rcond=None)
    return pd.DataFrame(Y.to_numpy() - (X @ beta).T, index=Y.index, columns=Y.columns)


def _seasonal(wide: pd.DataFrame, summer_months: Sequence[int]) -> pd.DataFrame:
    """Летний избыток и декабрьский пик по МО с полным рядом: ln ``v_all`` без тренда МО, среднее двух лет,
    затем exp − 1.

    Летний избыток: среднее за летние месяцы минус среднее за остальные месяцы без декабря. Декабрьский
    пик: декабрь минус среднее за январь–ноябрь. Тренд снимается до сравнения месяцев (spec_final, Б.1,
    п. 4): летние месяцы и декабрь в году позже остальных, и без этого показатели росли бы вместе с ростом
    трат МО (у декабря — на половину годового прироста).
    """
    D = detrended_log(wide)
    t = D.columns.to_numpy()
    month, year = t % MONTHS_IN_YEAR + 1, t // MONTHS_IN_YEAR
    summer = np.isin(month, list(summer_months))
    dec = month == DECEMBER
    v = D.to_numpy()
    per_year_s, per_year_d = [], []
    for y in np.unique(year):
        in_y = year == y
        per_year_s.append(v[:, in_y & summer].mean(axis=1) - v[:, in_y & ~summer & ~dec].mean(axis=1))
        per_year_d.append(v[:, in_y & dec].mean(axis=1) - v[:, in_y & ~dec].mean(axis=1))
    return pd.DataFrame(
        {
            "summer_excess": np.expm1(np.mean(per_year_s, axis=0)),
            "dec_peak": np.expm1(np.mean(per_year_d, axis=0)),
        },
        index=D.index,
    )


def build_mo(
    panel_wide: pd.DataFrame,
    territories: pd.DataFrame,
    context_annual: pd.DataFrame,
    *,
    summer_months: Sequence[int],
    rel_min_months: int,
) -> pd.DataFrame:
    """Таблица ``mo``: одна строка на МО панели, ключ ``territory_id`` (формулы — spec_final, Б.2).

    Уровень года — среднее ``v_all`` за 12 месяцев (NA, если месяцев меньше); доли года — Σ ``v_<часть>`` /
    Σ ``v_all`` за 12 месяцев; рост (номинал), летний избыток и декабрьский пик (без тренда МО) — только
    у МО с полным рядом.
    """
    ter = territories.set_index("territory_id")
    mo = ter[list(TERRITORY_COLUMNS)].copy()
    ids = mo.index
    ctx = context_annual.set_index(["territory_id", "year"])
    pop_avg = ctx["pop_avg"].unstack("year")
    for y in YEARS:
        mo[f"pop_{y}"] = pop_avg[y].reindex(ids).astype("float64") if y in pop_avg.columns else np.nan
    mo["weight"] = mo[f"pop_{YEARS[0]}"].fillna(mo[f"pop_{YEARS[1]}"])  # вес — население 2023, иначе 2024
    for year in YEARS:
        levels = _year_levels(panel_wide, year).reindex(ids)
        mo[f"level_{year}"] = levels[f"level_{year}"]
        mo[f"level_rel_{year}"] = _level_rel(panel_wide, year, rel_min_months).reindex(ids)
        mo[f"log_level_{year}"] = np.log(mo[f"level_{year}"])
        for p in PARTS:
            mo[f"sh_{p}_{year}"] = levels[f"sh_{p}_{year}"]
        for p in PARTS:
            mo[f"clr_{p}_{year}"] = levels[f"clr_{p}_{year}"]
    full = mo["series_status"].astype(str) == "full"
    y0, y1 = YEARS
    ratio = (mo[f"level_{y1}"] / mo[f"level_{y0}"]).where(full)
    mo["growth"] = ratio - 1
    mo["growth_log"] = np.log(ratio)
    mo["mp_pp_change"] = 100.0 * (mo[f"sh_marketplace_{y1}"] - mo[f"sh_marketplace_{y0}"])
    seasonal = _seasonal(panel_wide, summer_months).reindex(ids)
    mo["summer_excess"] = seasonal["summer_excess"].where(full)
    mo["dec_peak"] = seasonal["dec_peak"].where(full)

    c23 = ctx.xs(CONTEXT_YEAR, level="year").reindex(ids) if len(ctx) else pd.DataFrame(index=ids)
    sfx = f"_{CONTEXT_YEAR}"
    mo[f"wage{sfx}"] = c23["wage"].astype("float64")
    mo[f"spend_to_wage{sfx}"] = mo[f"level{sfx}"] / mo[f"wage{sfx}"]
    mo[f"ndfl_pc{sfx}"] = c23["ndfl_income_pc"].astype("float64")
    mo[f"spend_to_ndfl{sfx}"] = mo[f"level{sfx}"] / mo[f"ndfl_pc{sfx}"]
    for col in CONTEXT_2023:
        mo[f"{col}{sfx}"] = c23[col].astype("float64")
    mo[f"log_density{sfx}"] = np.log(mo[f"pop{sfx}"] / mo["area_km2"])
    mo["workplace_based"] = c23["workplace_based"].astype("boolean").fillna(False).astype(bool)
    mo["ndfl_ok"] = c23["ndfl_ok"].astype("boolean").fillna(False).astype(bool)
    floats = mo.select_dtypes("float64").columns
    mo[floats] = mo[floats].where(np.isfinite(mo[floats]))  # деление на ноль -> пропуск, не бесконечность
    mo = mo.reset_index()
    mo["territory_id"] = mo["territory_id"].astype("int32")
    return mo


# --- national ------------------------------------------------------------------------------------


def build_national(
    panel_long: pd.DataFrame, panel_wide: pd.DataFrame, context_annual: pd.DataFrame
) -> pd.DataFrame:
    """Таблица ``national``: ключ (``date``, ``category``), все 7 категорий.

    ``median_value``, ``q25_value``, ``q75_value`` — по МО месяца (₽); ``median_share`` — медиана долей по
    МО месяца («типичное МО»; у ``all`` — 1); ``resident_share`` — Σ ``v_c`` × ``pop_avg`` / Σ ``v_all`` ×
    ``pop_avg`` по МО с известным населением года («доля в тратах всех жителей»).
    """
    long = panel_long[["territory_id", "date", "year", "category", "value"]].copy()
    all_v = panel_wide[["territory_id", "date", "v_all"]]
    long = long.merge(all_v, on=["territory_id", "date"], how="left")
    long["share"] = long["value"].astype("float64") / long["v_all"].astype("float64")
    pop = context_annual[["territory_id", "year", "pop_avg"]]
    long = long.merge(pop, on=["territory_id", "year"], how="left")
    weighted = long.loc[long["pop_avg"].notna()].copy()
    weighted["wv"] = weighted["value"].astype("float64") * weighted["pop_avg"]
    weighted["wall"] = weighted["v_all"].astype("float64") * weighted["pop_avg"]
    keys = ["date", "category"]
    g = long.groupby(keys, observed=True)
    out = pd.DataFrame(
        {
            "n_mo": g["territory_id"].nunique(),
            "median_value": g["value"].median(),
            "q25_value": g["value"].quantile(0.25),
            "q75_value": g["value"].quantile(0.75),
            "median_share": g["share"].median(),
        }
    )
    wg = weighted.groupby(keys, observed=True)[["wv", "wall"]].sum()
    out["resident_share"] = (wg["wv"] / wg["wall"]).reindex(out.index)
    out = out.reset_index()
    out["category"] = out["category"].astype(pd.CategoricalDtype(list(CATEGORY_CODES), ordered=True))
    out["n_mo"] = out["n_mo"].astype("int32")
    return out.sort_values(keys).reset_index(drop=True)
