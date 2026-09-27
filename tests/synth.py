"""Синтетические выходы этапа panel для тестов разведки (разделы E1–E5 и сводка не ждут реальных данных).

``make_processed(root)`` пишет в ``root/data/processed`` все таблицы по контрактам ``munnet.contracts``,
``territories_geo.parquet`` (квадраты в проекции Альберса) и ``root/outputs/panel/controls.json``,
``unmatched.csv``. Заложенные сигналы (их проверяют тесты разделов):

- уровень трат: сильный эффект региона, связь с населением, столичный и внутригородской надбавки;
  зарплата и доход 5-НДФЛ связаны с уровнем (ρ > 0,5);
- годовой ритм: общий для всех (январь ниже, декабрь выше), у «северной» группы (широта > 60°) свой
  летний подъём, повторяемый в оба года; у остальных МО свой ритм — чистый шум;
- номинальный рост и рост доли маркетплейсов: доля растёт быстрее у бедных МО, стартовые доли близки;
- доля общепита растёт с уровнем, продовольствия — падает (корзина связана с уровнем);
- неполные ряды — у самых мелких МО (только 2023 год, только 2024 год, внутренние разрывы, 11 месяцев 2024);
- 2 внутригородских МО (Москва и Санкт-Петербург, по одному), объединение 2024 года (предшественник и
  преемник; у преемника в 2023 году причина ``not_in_slice`` в ``unmatched.csv``), выпавший регион без
  данных СберИндекса (только на карте); в ``controls.json`` — и справочные числа (``kind: info``);
- шум трат растёт у малых МО (множитель √(медиана населения / население)), у общепита и транспорта —
  сильнее: T07 видит шумные малые МО;
- контекст: доли занятости (аграрные МО беднее), ночёвки, пригороды столиц с низким доходом по месту работы.

Импорт в тестах: ``from synth import make_eda_data`` (pytest кладёт ``tests/`` в ``sys.path``).
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from pyproj import Transformer

from munnet.config import Config, load_config
from munnet.contracts import (
    CATEGORY_CODES,
    CITY_CONTEXT,
    CONTEXT_ANNUAL,
    CONTEXT_LONG,
    LINEAGE_ROLES,
    MATCH_METHODS,
    MO_STATUSES,
    MO_TYPES,
    N_MONTHS,
    OKVED_SHARES,
    PANEL_LONG,
    PANEL_WIDE,
    PARTS,
    POP_AVG_METHODS,
    SERIES_STATUSES,
    TERRITORIES,
    YEARS,
    write_table,
)
from munnet.eda.data import EdaData, load

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"

# Общий годовой ритм страны (лог-пункты по календарным месяцам): январь ниже, декабрь выше.
COMMON_RHYTHM = np.array([-0.15, -0.12, -0.03, -0.02, 0.0, 0.01, 0.03, 0.03, -0.01, 0.0, 0.0, 0.20])
NORTH_SUMMER = 0.12  # свой летний подъём северной группы, лог-пункты (июнь–август)
SUMMER = (6, 7, 8)
NOISE_SD = 0.03  # помесячный шум ln v_all у МО медианного населения
# Шум долей у МО медианного населения; общепит и транспорт шумят сильнее (как в данных, T07).
SHARE_NOISE_SD = {"food": 0.005, "marketplace": 0.003, "transport": 0.004, "health": 0.002, "cafe": 0.002}
# Шум растёт у малых МО: множитель √(медиана населения / население МО) в этих пределах. У ln v_all шум
# только растёт (не меньше NOISE_SD): иначе у южных МО появляются случайные «надёжные» ритмы, и заложенный
# сигнал E3 (свой ритм — только у Севера) размывается; у долей — в обе стороны.
SIZE_NOISE_CLIP = {"all": (1.0, 2.0), "share": (0.5, 3.0)}
SIZE_NOISY = ("transport", "cafe")  # доли, шум которых зависит от населения
SIDE_KM = 40.0  # сторона квадрата МО
CITY_SIDE_KM = 5.0  # сторона квадрата внутригородской территории
CITY_REGIONS = {77: "Москва", 78: "Санкт-Петербург"}
ABSENT_REGION = (31, "Выпавшая область")
ABSENT_MO = 4
UNION_ID = "union_22"
SECTIONS_OKVED = tuple("ABCDEFGHIJKLMNOPQRS")


def _months() -> pd.DataFrame:
    t = np.arange(N_MONTHS)
    year = YEARS[0] + t // 12
    month = t % 12 + 1
    return pd.DataFrame(
        {
            "t": t,
            "year": year,
            "month": month,
            "date": [f"{y}-{m:02d}" for y, m in zip(year, month, strict=True)],
        }
    )


def make_config(root: Path) -> Config:
    """Конфиг проекта с путями внутри ``root`` (данные, выходы, отчёт)."""
    base = load_config(DEFAULT_CONFIG)
    data = copy.deepcopy(base.data)
    root = Path(root)
    data["paths"] = {
        "raw": str(root / "data" / "raw"),
        "interim": str(root / "data" / "interim"),
        "processed": str(root / "data" / "processed"),
        "outputs": str(root / "outputs"),
    }
    data["eda"]["report"] = str(root / "docs" / "eda.md")
    data["eda"]["report_images"] = str(root / "docs" / "img" / "eda")
    data["features"]["report"] = str(root / "docs" / "features.md")
    # утверждения отчёта features сформулированы по реальным данным; синтетика их не обязана выполнять
    data["features"]["report_strict"] = False
    return Config(data=data, path=DEFAULT_CONFIG)


def _square(x: float, y: float, side_m: float) -> shapely.Polygon:
    h = side_m / 2
    return shapely.box(x - h, y - h, x + h, y + h)


def _territories(rng: np.random.Generator, n_regions: int, mo_per_region: int, crs: str) -> pd.DataFrame:
    """МО панели: регионы, положение, население, базовый уровень трат, скрытые «истинные» параметры."""
    to_aea = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    n_north = max(1, n_regions // 3)
    rows = []
    tid = 1
    for r in range(n_regions):
        north = r >= n_regions - n_north
        lat = 62.5 + 3.0 * (r - (n_regions - n_north)) if north else 50.0 + 2.5 * r
        lon = 35.0 + 15.0 * r
        cx, cy = to_aea.transform(lon, lat)
        cols = math.ceil(math.sqrt(mo_per_region))
        region_effect = rng.normal(0.0, 0.25)
        for j in range(mo_per_region):
            gx, gy = j % cols, j // cols
            rows.append(
                {
                    "territory_id": tid,
                    "region_code": r + 1,
                    "region_name": f"Регион {r + 1}",
                    "x_aea": cx + (gx - (cols - 1) / 2) * SIDE_KM * 1000,
                    "y_aea": cy + (gy - 1) * SIDE_KM * 1000,
                    "side_m": SIDE_KM * 1000,
                    "north": north,
                    "capital": j == 0,
                    "inner": False,
                    "region_effect": region_effect,
                }
            )
            tid += 1
    for code, name in CITY_REGIONS.items():
        lon, lat = (37.6, 55.75) if code == 77 else (30.3, 59.94)
        x, y = to_aea.transform(lon, lat)
        rows.append(
            {
                "territory_id": tid,
                "region_code": code,
                "region_name": name,
                "x_aea": x,
                "y_aea": y,
                "side_m": CITY_SIDE_KM * 1000,
                "north": False,
                "capital": False,
                "inner": True,
                "region_effect": 0.3,
            }
        )
        tid += 1
    ter = pd.DataFrame(rows)
    n = len(ter)
    log_pop = rng.normal(10.0, 0.8, n) + np.where(ter["capital"], 2.0, 0.0) + np.where(ter["inner"], 1.5, 0.0)
    ter["pop"] = np.round(np.exp(log_pop))
    z_pop = (log_pop - log_pop.mean()) / log_pop.std()
    ter["log_level"] = (
        10.05
        + ter["region_effect"]
        + 0.08 * z_pop
        + np.where(ter["capital"], 0.15, 0.0)
        + np.where(ter["inner"], 0.6, 0.0)
        + np.where(ter["north"], 0.15, 0.0)
        + rng.normal(0.0, 0.06, n)
    )
    z_level = (ter["log_level"] - ter["log_level"].mean()) / ter["log_level"].std()
    ter["z_level"] = z_level
    ter["growth"] = 0.14 + 0.02 * z_level + rng.normal(0.0, 0.015, n)
    ter["mp0"] = 0.10 + rng.normal(0.0, 0.008, n)
    ter["mp_gain"] = np.clip(0.09 - 0.025 * z_level + rng.normal(0.0, 0.006, n), 0.02, None)
    ter["cafe0"] = 0.026 * np.exp(0.35 * z_level + rng.normal(0.0, 0.05, n))
    ter["food0"] = np.clip(0.46 - 0.03 * z_level + rng.normal(0.0, 0.01, n), 0.35, 0.55)
    return ter


def _coverage(ter: pd.DataFrame) -> dict[int, np.ndarray]:
    """Какие месяцы есть у МО: неполные ряды — у самых мелких МО (не столицы и не внутригородские)."""
    cover = {int(t): np.ones(N_MONTHS, dtype=bool) for t in ter["territory_id"]}
    small = ter.loc[~ter["capital"] & ~ter["inner"]].sort_values("pop")["territory_id"].astype(int).tolist()
    only_2023 = small[0:2]
    only_2024 = small[2]
    gap_2023, gap_cross, eleven_2024 = small[3], small[4], small[5]
    for tid in only_2023:
        cover[tid][12:] = False
    cover[only_2024][:12] = False
    cover[gap_2023][4:7] = False
    cover[gap_cross][10:14] = False
    cover[eleven_2024][23] = False
    return cover


def _size_factor(pop: pd.Series, kind: str) -> np.ndarray:
    """Множитель шума МО: √(медиана населения / население), в пределах ``SIZE_NOISE_CLIP[kind]``."""
    lo, hi = SIZE_NOISE_CLIP[kind]
    return np.clip(np.sqrt(pop.median() / pop.to_numpy(dtype="float64")), lo, hi)


def _panel(rng: np.random.Generator, ter: pd.DataFrame, cover: dict[int, np.ndarray]) -> pd.DataFrame:
    """Помесячные траты: уровень, рост, общий ритм, летний подъём Севера и шум, растущий у малых МО."""
    months = _months()
    t = months["t"].to_numpy()
    m_idx = months["month"].to_numpy() - 1
    summer_bump = np.isin(months["month"], SUMMER) * NORTH_SUMMER
    summer_bump = summer_bump - summer_bump.mean()
    f_all, f_share = _size_factor(ter["pop"], "all"), _size_factor(ter["pop"], "share")
    frames = []
    for i, row in enumerate(ter.itertuples(index=False)):
        ln_all = (
            row.log_level
            + row.growth * (t - (N_MONTHS - 1) / 2) / 12
            + COMMON_RHYTHM[m_idx]
            + (summer_bump if row.north else 0.0)
            + rng.normal(0.0, NOISE_SD * f_all[i], N_MONTHS)
        )
        v_all = np.round(np.exp(ln_all)).astype(np.int64)
        mp = row.mp0 + row.mp_gain * t / (N_MONTHS - 1) + np.where(m_idx == 11, 0.02, 0.0)
        sd = {c: s * (f_share[i] if c in SIZE_NOISY else 1.0) for c, s in SHARE_NOISE_SD.items()}
        shares = {
            "food": row.food0 + rng.normal(0.0, sd["food"], N_MONTHS),
            "marketplace": mp + rng.normal(0.0, sd["marketplace"], N_MONTHS),
            "transport": 0.057 + rng.normal(0.0, sd["transport"], N_MONTHS),
            "health": 0.049 + rng.normal(0.0, sd["health"], N_MONTHS),
            "cafe": row.cafe0 * (1 + 0.3 * np.isin(months["month"], SUMMER))
            + rng.normal(0.0, sd["cafe"], N_MONTHS),
        }
        wide = months.copy()
        wide["territory_id"] = row.territory_id
        wide["v_all"] = v_all
        for c, sh in shares.items():
            wide[f"v_{c}"] = np.maximum(np.round(v_all * np.clip(sh, 0.005, None)), 1).astype(np.int64)
        wide["v_other"] = wide["v_all"] - wide[[f"v_{c}" for c in shares]].sum(axis=1)
        frames.append(wide.loc[cover[int(row.territory_id)]])
    wide = pd.concat(frames, ignore_index=True)
    for c in CATEGORY_CODES:
        wide[f"v_{c}"] = wide[f"v_{c}"].astype("int32")
    for p in PARTS:
        wide[f"sh_{p}"] = wide[f"v_{p}"] / wide["v_all"]
    wide["log_all"] = np.log(wide["v_all"].astype("float64"))
    wide["territory_id"] = wide["territory_id"].astype("int32")
    wide["t"] = wide["t"].astype("int8")
    wide["year"] = wide["year"].astype("int16")
    wide["month"] = wide["month"].astype("int8")
    cols = ["territory_id", "date", "t", "year", "month", *[f"v_{c}" for c in CATEGORY_CODES]]
    return wide[cols + [f"sh_{p}" for p in PARTS] + ["log_all"]]


def _long(wide: pd.DataFrame) -> pd.DataFrame:
    keys = ["territory_id", "date", "t", "year", "month"]
    long = wide.melt(id_vars=keys, value_vars=[f"v_{c}" for c in CATEGORY_CODES], var_name="category")
    long["category"] = pd.Categorical(long["category"].str[2:], categories=list(CATEGORY_CODES), ordered=True)
    long["value"] = long["value"].astype("int32")
    long["is_derived"] = long["category"].astype(str) == "other"
    return long


def _series_quality(cover: dict[int, np.ndarray]) -> pd.DataFrame:
    rows = []
    months = _months()
    for tid, c in cover.items():
        idx = np.flatnonzero(c)
        first, last = idx.min(), idx.max()
        inner = ~c[first : last + 1]
        gaps, run = [], 0
        for missing in inner:
            run = run + 1 if missing else 0
            gaps.append(run)
        n23, n24 = int(c[:12].sum()), int(c[12:].sum())
        if c.all():
            status = "full"
        elif n23 == 12 and n24 == 0:
            status = "only_2023"
        elif n23 == 0 and n24 == 12:
            status = "only_2024"
        else:
            status = "partial"
        rows.append(
            {
                "territory_id": tid,
                "n_months": int(c.sum()),
                "n_2023": n23,
                "n_2024": n24,
                "first_date": months["date"][first],
                "last_date": months["date"][last],
                "coverage_pattern": "".join("1" if x else "0" for x in c),
                "series_status": status,
                "has_internal_gap": bool(inner.any()),
                "longest_gap": max(gaps) if gaps else 0,
            }
        )
    return pd.DataFrame(rows)


def _union_pair(quality: pd.DataFrame) -> tuple[int, int]:
    """Предшественник (первый ряд «только 2023») и преемник (ряд «только 2024») объединения 2024 года."""
    status = quality.set_index("territory_id")["series_status"]
    return int(status.index[status == "only_2023"][0]), int(status.index[status == "only_2024"][0])


def _territories_table(
    ter: pd.DataFrame, quality: pd.DataFrame, crs: str, rng: np.random.Generator, pred: int, succ: int
) -> pd.DataFrame:
    to_wgs = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    lon, lat = to_wgs.transform(ter["x_aea"].to_numpy(), ter["y_aea"].to_numpy())
    out = pd.DataFrame({"territory_id": ter["territory_id"].astype("int32")})
    names = [
        f"Внутригородская территория {i}" if inner else f"Муниципальное образование {i}"
        for i, inner in zip(ter["territory_id"], ter["inner"], strict=True)
    ]
    out["name"] = names
    out["name_short"] = [f"МО {i}" for i in ter["territory_id"]]
    out["region_code"] = ter["region_code"].astype("int16")
    out["region_name"] = ter["region_name"]
    types = np.where(
        ter["inner"], "vgt", np.where(ter["capital"], "go", rng.choice(["mr", "mo", "go"], len(ter)))
    )
    out["mo_type"] = pd.Categorical(types, categories=list(MO_TYPES))
    out["mo_status"] = pd.Categorical(np.where(ter["capital"], "capital", None), categories=list(MO_STATUSES))
    out["is_capital"] = ter["capital"].to_numpy()
    out["is_inner_city"] = ter["inner"].to_numpy()
    oktmo = [
        f"{int(r):02d}{int(t):06d}" for r, t in zip(ter["region_code"], ter["territory_id"], strict=True)
    ]
    out["oktmo_2023"] = pd.array(oktmo, dtype="str")
    out["oktmo_2024"] = pd.array(oktmo, dtype="str")
    out["shape"] = np.int8(1)
    out["center_name"] = pd.array(
        [None if i else f"с. Центр {t}" for t, i in zip(ter["territory_id"], ter["inner"], strict=True)],
        dtype="str",
    )
    out["center_lat"] = np.where(ter["inner"], np.nan, lat)
    out["center_lon"] = np.where(ter["inner"], np.nan, lon)
    out["point_lat"], out["point_lon"] = lat, lon
    out["x_aea"], out["y_aea"] = ter["x_aea"].to_numpy(), ter["y_aea"].to_numpy()
    out["area_km2"] = (ter["side_m"] / 1000) ** 2
    access = 1000 / (1 + np.exp(-(ter["z_level"] + rng.normal(0, 0.5, len(ter)))))
    out["market_access"] = access.to_numpy()
    out.loc[out.index[-3], "market_access"] = np.nan  # индекс известен не у всех МО
    cap = ter.loc[ter["capital"]].set_index("region_code")[["x_aea", "y_aea"]]
    cx = ter["region_code"].map(cap["x_aea"])
    cy = ter["region_code"].map(cap["y_aea"])
    out["dist_capital_km"] = (np.hypot(ter["x_aea"] - cx, ter["y_aea"] - cy) / 1000).to_numpy()
    out = out.merge(quality, on="territory_id", how="left")
    for c in ("n_months", "n_2023", "n_2024", "longest_gap"):
        out[c] = out[c].astype("int8")
    out["series_status"] = pd.Categorical(out["series_status"], categories=list(SERIES_STATUSES))
    out["lineage_role"] = "none"
    out["lineage_change_id"] = pd.array([None] * len(out), dtype="str")
    out["successor_id"] = pd.array([None] * len(out), dtype="Int32")
    pred = out.index[out["territory_id"] == pred][0]
    succ = out.index[out["territory_id"] == succ][0]
    out.loc[pred, "lineage_role"] = "union_predecessor"
    out.loc[succ, "lineage_role"] = "union_successor"
    out.loc[[pred, succ], "lineage_change_id"] = UNION_ID
    out.loc[pred, "successor_id"] = out.loc[succ, "territory_id"]
    out.loc[pred, "oktmo_2024"] = None
    out.loc[succ, "oktmo_2023"] = None
    out["lineage_role"] = pd.Categorical(out["lineage_role"], categories=list(LINEAGE_ROLES))
    return out


def _okved(rng: np.random.Generator, ter: pd.DataFrame) -> list[tuple[int, dict[str, float]]]:
    """Доли занятости: аграрные МО беднее, добыча — у части северных, обработка — у части южных."""
    rows = []
    for row in ter.itertuples(index=False):
        base = rng.dirichlet(np.full(len(SECTIONS_OKVED), 2.0))
        shares = dict(zip(SECTIONS_OKVED, base * 0.4, strict=True))
        shares["A"] = float(np.clip(0.12 - 0.08 * row.z_level + rng.normal(0, 0.03), 0.0, 0.5))
        shares["B"] = 0.25 if (row.north and row.territory_id % 3 == 0) else 0.01
        shares["C"] = 0.5 if (not row.north and row.territory_id % 4 == 0) else 0.08  # обрабатывающие МО
        shares["O"] = shares["O"] + 0.1 + (0.6 if row.z_level < -0.8 else 0.0)  # бедная бюджетная периферия
        shares["P"] = shares["P"] + 0.08
        shares["Q"] = shares["Q"] + 0.06
        total = sum(shares.values())
        disclosed = {k: v / total * 0.95 for k, v in shares.items()}
        if row.inner:
            disclosed.pop("B")  # раздел скрыт
        rows.append((row.territory_id, disclosed))
    return rows


def _absent_keys(table: pd.DataFrame) -> set[tuple[int, int]]:
    """(МО, год) вне среза справочника года: преемник объединения в 2023 году (реальный этап panel пишет
    для него в ``unmatched.csv`` причину ``not_in_slice``)."""
    roles = table.set_index("territory_id")["lineage_role"].astype(str)
    return {(int(t), YEARS[0]) for t in roles.index[roles == "union_successor"]}


def _context(rng: np.random.Generator, ter: pd.DataFrame, table: pd.DataFrame, okved_rows, cfg: Config):
    groups = cfg["context"]["okved_groups"]
    lo, hi = cfg["context"]["ndfl"]["recipients_ratio_ok"]
    annual, shares_rows, long_rows = [], [], []
    wage_base = np.exp(ter["log_level"] + 0.8 + rng.normal(0, 0.08, len(ter))).to_numpy()
    suburb = ((table["dist_capital_km"] <= 50) & ~table["is_capital"] & ~table["is_inner_city"]).to_numpy()
    nights_pc = np.where(
        rng.random(len(ter)) < 0.15, rng.uniform(5, 40, len(ter)), rng.uniform(0, 2, len(ter))
    )
    for i, row in enumerate(ter.itertuples(index=False)):
        okved = dict(okved_rows[i][1])
        for y_i, year in enumerate(YEARS):
            pop_jan1 = row.pop * (1 - 0.005 * y_i)
            pop_avg = pop_jan1 * (1 - 0.0025)
            wage = wage_base[i] * (1.12**y_i)
            employees = pop_avg * 0.3
            payroll_krub = employees * wage * 12 / 1000
            recipients = pop_avg * rng.uniform(0.3, 0.5) * (0.4 if suburb[i] else 1.0)
            if row.inner:
                recipients = pop_avg * 4.0  # по месту работодателя: больше, чем жителей
            income = recipients * wage * 12 * rng.uniform(0.7, 0.9)
            ratio = recipients / pop_avg
            workplace = bool(row.inner)
            ndfl_ok = bool(lo <= ratio <= hi) and not workplace
            young = rng.uniform(0.15, 0.22)
            old = rng.uniform(0.2, 0.3) + (0.05 if row.z_level < 0 else 0.0)
            urban = float(
                np.clip(0.5 + 0.2 * row.z_level + (0.5 if row.capital or row.inner else 0), 0.05, 1.0)
            )
            disclosed_sum = sum(okved.values())
            rec = {
                "territory_id": row.territory_id,
                "year": year,
                "pop_jan1": pop_jan1,
                "pop_jan1_method": "direct",
                "pop_avg": pop_avg,
                "pop_avg_method": "mean_jan1",
                "urban_share": urban,
                "age_young_share": young,
                "age_working_share": 1 - young - old,
                "age_old_share": old,
                "employees": employees,
                "payroll_krub": payroll_krub,
                "wage": wage,
                "emp_sh_unallocated": 1 - disclosed_sum,
                "emp_n_disclosed": len(okved),
                "retail_krub": pop_avg * 150.0,
                "catering_turnover_krub": pop_avg * 8.0 if i % 4 else np.nan,
                "shipments_krub": pop_avg * 300.0,
                "shipments_mining_share": okved.get("B", 0.0),
                "orgs": pop_avg * 0.02 if year == YEARS[-1] else np.nan,
                "ip": pop_avg * 0.03 if year == YEARS[-1] else np.nan,
                "nights": nights_pc[i] * pop_avg if nights_pc[i] > 1 else np.nan,
                "beds": pop_avg * 0.01 if nights_pc[i] > 1 else np.nan,
                "invest_krub": pop_avg * 50.0 if year == YEARS[0] else np.nan,
                "ndfl_income_rub": income,
                "ndfl_recipients": recipients,
                "workplace_based": workplace,
                "ndfl_ok": ndfl_ok,
            }
            for letter in SECTIONS_OKVED:
                rec[f"emp_sh_{letter}"] = okved.get(letter, np.nan)
            for g, letters in groups.items():
                rec[f"emp_sh_{g}"] = sum(okved.get(x, 0.0) for x in letters)
            per_month = 1000 / pop_avg / 12
            pc = {
                "payroll_pc": payroll_krub * per_month,
                "retail_pc": rec["retail_krub"] * per_month,
                "catering_turnover_pc": rec["catering_turnover_krub"] * per_month,
                "shipments_pc": rec["shipments_krub"] * per_month,
                "ndfl_income_pc": income / pop_avg / 12,
                "employees_to_working_age": employees / (pop_jan1 * rec["age_working_share"]),
            }
            if workplace:
                pc = dict.fromkeys(pc, np.nan)
            if not ndfl_ok:
                pc["ndfl_income_pc"] = np.nan
            rec.update(pc)
            rec["ndfl_income_per_recipient"] = income / recipients / 12
            rec["recipients_to_pop"] = ratio
            rec["orgs_per_1000"] = rec["orgs"] / pop_avg * 1000
            rec["ip_per_1000"] = rec["ip"] / pop_avg * 1000
            rec["beds_per_1000"] = rec["beds"] / pop_avg * 1000
            rec["nights_pc"] = rec["nights"] / pop_avg
            annual.append(rec)
            for letter in SECTIONS_OKVED:
                share = okved.get(letter, np.nan)
                shares_rows.append((row.territory_id, year, letter, share * employees, share))
            shares_rows.append(
                (row.territory_id, year, "unallocated", (1 - disclosed_sum) * employees, 1 - disclosed_sum)
            )
            oktmo = f"{int(row.region_code):02d}{int(row.territory_id):06d}"
            method = "version" if row.territory_id % 17 == 0 else "direct"
            for ind, val, unit, code, period in (
                ("population", pop_jan1, "Человек", "Y48112027", "На 1 января"),
                ("wage", wage, "Рубль", "Y48423007", "Январь-декабрь"),
                ("employees", employees, "Человек", "Y48423005", "Январь-декабрь"),
                ("ndfl_income", income, "Рубль", "Y777000006", "Январь-декабрь"),
                ("ndfl_recipients", recipients, "Человек", "Y777000028", "Январь-декабрь"),
            ):
                long_rows.append(
                    (row.territory_id, year, ind, "TOTAL", val, unit, code, period, oktmo, method, "")
                )
    annual = pd.DataFrame(annual)
    annual["territory_id"] = annual["territory_id"].astype("int32")
    annual["year"] = annual["year"].astype("int16")
    annual["pop_jan1_method"] = pd.Categorical(annual["pop_jan1_method"], categories=list(MATCH_METHODS))
    annual["pop_avg_method"] = pd.Categorical(annual["pop_avg_method"], categories=list(POP_AVG_METHODS))
    annual["emp_n_disclosed"] = annual["emp_n_disclosed"].astype("int8")
    shares = pd.DataFrame(shares_rows, columns=["territory_id", "year", "section", "employees", "share"])
    shares["territory_id"] = shares["territory_id"].astype("int32")
    shares["year"] = shares["year"].astype("int16")
    cols = [
        "territory_id",
        "year",
        "indicator",
        "dim",
        "value",
        "unit",
        "source_code",
        "period_used",
        "oktmo_used",
        "method",
        "flag",
    ]
    long = pd.DataFrame(long_rows, columns=cols)
    long["territory_id"] = long["territory_id"].astype("int32")
    long["year"] = long["year"].astype("int16")
    long["indicator"] = long["indicator"].astype("category")
    long["method"] = pd.Categorical(long["method"], categories=list(MATCH_METHODS))
    return annual, shares, long


def _geo(ter: pd.DataFrame, table: pd.DataFrame, crs: str) -> gpd.GeoDataFrame:
    """Квадраты МО панели и выпавшего региона; у предшественника и преемника объединения — один квадрат."""
    roles = table.set_index("territory_id")["lineage_role"].astype(str)
    rows = []
    for row in ter.itertuples(index=False):
        tid = int(row.territory_id)
        x, y = row.x_aea, row.y_aea
        year_from = 2024 if roles[tid] == "union_successor" else 2018
        year_to = 2024 if roles[tid] == "union_predecessor" else 9999
        rows.append(
            (tid, year_from, year_to, row.region_code, True, row.region_name, _square(x, y, row.side_m))
        )
    x0, y0 = ter["x_aea"].min(), ter["y_aea"].min() - 3 * SIDE_KM * 1000
    code, name = ABSENT_REGION
    for j in range(ABSENT_MO):
        tid = int(ter["territory_id"].max()) + 1 + j
        rows.append(
            (tid, 2018, 9999, code, False, name, _square(x0 + j * SIDE_KM * 1000, y0, SIDE_KM * 1000))
        )
    geo = gpd.GeoDataFrame(
        rows,
        columns=[
            "territory_id",
            "year_from",
            "year_to",
            "region_code",
            "in_panel",
            "region_name",
            "geometry",
        ],
        geometry="geometry",
        crs=crs,
    )
    geo["territory_id"] = geo["territory_id"].astype("int32")
    geo["year_from"] = geo["year_from"].astype("int16")
    geo["year_to"] = geo["year_to"].astype("int16")
    geo["region_code"] = geo["region_code"].astype("int16")
    geo["geometry"] = geo.geometry.apply(lambda g: shapely.MultiPolygon([g]))
    return geo


def make_processed(root: Path, *, n_regions: int = 6, mo_per_region: int = 12, seed: int = 0) -> Path:
    """Пишет синтетические выходы этапа panel в ``root``; возвращает ``root/data/processed``."""
    root = Path(root)
    cfg = make_config(root)
    crs = cfg["panel"]["crs"]
    rng = np.random.default_rng(seed)
    ter = _territories(rng, n_regions, mo_per_region, crs)
    cover = _coverage(ter)
    wide = _panel(rng, ter, cover)
    long = _long(wide)
    quality = _series_quality(cover)
    pred, succ = _union_pair(quality)
    # Преемник объединения занимает место предшественника (на карте 2023 года — один, 2024-го — другой).
    succ_row, pred_row = ter.index[ter["territory_id"] == succ][0], ter.index[ter["territory_id"] == pred][0]
    ter.loc[succ_row, ["x_aea", "y_aea", "region_code", "region_name", "region_effect"]] = ter.loc[
        pred_row, ["x_aea", "y_aea", "region_code", "region_name", "region_effect"]
    ]
    table = _territories_table(ter, quality, crs, rng, pred, succ)
    okved_rows = _okved(rng, ter)
    annual, shares, context_long = _context(rng, ter, table, okved_rows, cfg)

    processed = cfg.dir("processed")
    write_table(long, PANEL_LONG, processed / "panel_long.parquet")
    write_table(wide, PANEL_WIDE, processed / "panel_wide.parquet")
    write_table(table, TERRITORIES, processed / "territories.parquet")
    write_table(context_long, CONTEXT_LONG, processed / "context_long.parquet")
    write_table(annual, CONTEXT_ANNUAL, processed / "context_annual.parquet")
    write_table(shares, OKVED_SHARES, processed / "okved_shares.parquet")
    write_table(_city_context(ter), CITY_CONTEXT, processed / "city_context.parquet")
    _geo(ter, table, crs).to_parquet(processed / "territories_geo.parquet")

    panel_out = cfg.dir("outputs") / "panel"
    panel_out.mkdir(parents=True, exist_ok=True)
    statuses = table["series_status"].astype(str).value_counts()
    controls = {
        "territories": {"expected": len(table), "actual": len(table), "kind": "hard", "ok": True},
        "series_full": {
            "expected": int(statuses.get("full", 0)),
            "actual": int(statuses.get("full", 0)),
            "kind": "hard",
            "ok": True,
        },
        "pop_jan1_2023": {"expected": len(table), "actual": len(table) - 1, "kind": "soft", "ok": False},
        # информационные числа: ожидаемого значения нет (как у реального этапа)
        "bdmo_stable_pairs": {"expected": None, "actual": len(table), "kind": "info", "ok": True},
        "method_population_2023_version": {
            "expected": None,
            "actual": int((context_long["method"].astype(str) == "version").sum()),
            "kind": "info",
            "ok": True,
        },
    }
    with open(panel_out / "controls.json", "w", encoding="utf-8") as f:
        json.dump(controls, f, ensure_ascii=False, sort_keys=True, indent=1)
    _unmatched(table).to_csv(panel_out / "unmatched.csv", index=False, lineterminator="\n")
    return processed


def _city_context(ter: pd.DataFrame) -> pd.DataFrame:
    """Строки 5-НДФЛ «город целиком» (как ``panel.ndfl.read_city_rows``): только у Москвы (77), у Петербурга
    строки «Свод» нет. Получателей в 1,4 раза больше жителей (работают в городе и жители области)."""
    pop = float(ter.loc[ter["region_code"] == 77, "pop"].sum())
    rows = []
    for year in YEARS:
        rows.append((77, year, "ndfl_recipients", 1.4 * pop))
        rows.append((77, year, "ndfl_income", 1.4 * pop * 12 * 90_000.0))
    out = pd.DataFrame(rows, columns=["region_code", "year", "indicator", "value"])
    out["oktmo"] = "45000000"
    out["source_code"] = np.where(out["indicator"] == "ndfl_income", "Y777000006", "Y777000028")
    out["report_type"] = "Свод"
    out["flag"] = pd.Series([None] * len(out), dtype="str")
    return out.astype({"region_code": "int16", "year": "int16"})


UNMATCHED_INDICATORS = ("population", "wage")  # показатели строк unmatched.csv у преемника в 2023 году


def _unmatched(table: pd.DataFrame) -> pd.DataFrame:
    """``unmatched.csv`` с колонками реального файла: преемник объединения в 2023 году, причина
    ``not_in_slice`` (МО нет в срезе справочника года), ОКТМО пуст. Значения 2023 года в синтетике у него
    остаются: раздел E1 считает их исключением, а не потерей."""
    info = table.set_index("territory_id")[["name", "region_code", "region_name"]]
    rows = [
        (tid, year, ind, "not_in_slice", None)
        for tid, year in sorted(_absent_keys(table))
        for ind in UNMATCHED_INDICATORS
    ]
    out = pd.DataFrame(rows, columns=["territory_id", "year", "indicator", "reason", "oktmo"])
    return out.join(info, on="territory_id")


def make_eda_data(tmp_path: Path, **kw) -> EdaData:
    """Синтетика + ``munnet.eda.data.load``: готовый ``EdaData`` для тестов разделов."""
    make_processed(tmp_path, **kw)
    return load(make_config(tmp_path))


def make_section_context(tmp_path: Path, section: str, data: EdaData | None = None, **kw):
    """``SectionContext`` раздела на синтетике: выходы — в ``tmp_path/outputs/eda``."""
    from munnet.eda import eda_dir, section_rng
    from munnet.eda.base import SECTION_IDS, SYNTHESIS_ID, SYNTHESIS_NUMBER, SectionContext

    cfg = make_config(tmp_path)
    if data is None:
        data = make_eda_data(tmp_path, **kw)
    number = SYNTHESIS_NUMBER if section == SYNTHESIS_ID else SECTION_IDS.index(section) + 1
    return SectionContext(cfg, data, eda_dir(cfg), section, section_rng(cfg, number))
