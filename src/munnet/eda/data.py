"""Вход разведки: таблицы этапа panel и две общие производные таблицы (spec_final, Б.2).

``load(cfg)`` читает ``data/processed`` и ``outputs/panel`` (разведка панель не пересобирает) и строит:

- ``mo`` — строка на МО панели: уровни, доли и CLR по годам, рост, сезонные показатели, контекст 2023 года;
- ``national`` — месяц × категория: медианы и квартили трат, медиана долей по МО («типичное МО») и доля
  в тратах всех жителей (веса — среднегодовое население).

Разделы берут показатели отсюда и не пересчитывают их по-своему.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from munnet.config import Config
from munnet.contracts import (
    CATEGORY_CODES,
    CONTEXT_ANNUAL,
    CONTEXT_LONG,
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
    with open(_require(panel_out / "controls.json"), encoding="utf-8") as f:
        controls = json.load(f)
    unmatched = pd.read_csv(_require(panel_out / "unmatched.csv"), dtype={"reason": str})
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
        **tables, controls=controls, unmatched=unmatched, mo=mo, national=national, geo_path=geo_path
    )


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


def _seasonal(wide: pd.DataFrame, summer_months: Sequence[int]) -> pd.DataFrame:
    """Летний избыток и декабрьский пик по МО (ряд трат ln v_all), среднее двух лет, затем exp − 1.

    Летний избыток: среднее ln за летние месяцы минус среднее ln за остальные месяцы без декабря.
    Декабрьский пик: ln декабря минус среднее ln за январь–ноябрь.
    """
    w = wide[["territory_id", "year", "month"]].copy()
    w["ln"] = wide["log_all"]
    summer = w["month"].isin(list(summer_months))
    dec = w["month"] == DECEMBER
    keys = ["territory_id", "year"]
    s_mean = w.loc[summer].groupby(keys)["ln"].mean()
    rest_mean = w.loc[~summer & ~dec].groupby(keys)["ln"].mean()
    dec_ln = w.loc[dec].groupby(keys)["ln"].mean()
    jan_nov = w.loc[~dec].groupby(keys)["ln"].mean()
    per_year = pd.DataFrame({"summer": s_mean - rest_mean, "dec": dec_ln - jan_nov})
    avg = per_year.groupby(level="territory_id").mean()
    return pd.DataFrame({"summer_excess": np.expm1(avg["summer"]), "dec_peak": np.expm1(avg["dec"])})


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
    Σ ``v_all`` за 12 месяцев; рост, летний избыток и декабрьский пик — только у МО с полным рядом (номинал).
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
