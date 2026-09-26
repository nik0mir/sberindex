"""Раздел разведки E2 «Сколько тратят и от чего это зависит» (spec_final, Б.3: Q4–Q6; Б.4; В.5).

Уровень трат — оценка СберИндекса средних безналичных потребительских трат жителя МО в месяц (₽, номинал),
среднее за 12 месяцев года (``EdaData.mo``, Б.2). Траты привязаны к жителям МО: траты приезжих в МО не видны,
а траты жителей вне своего МО (поездки, онлайн) по смыслу входят, но как именно — модель СберИндекса не
раскрывает.

Выходы:

- F03 ``level_distribution`` — распределение уровня года в лог-шкале с тремя отметками: медиана МО («типичное
  МО»), среднее МО и среднее по жителям («типичный житель», веса — среднегодовое население);
- F04 ``level_map`` — картограмма уровня к медиане МО того же месяца (квантильные классы), η² региона;
- F05 ``level_vs_wage`` — траты против зарплаты Росстата в лог-лог шкале, ρ Спирмена в целом и внутри
  регионов, подписи МО с крупнейшими остатками от робастной линии;
- T03 ``level_extremes`` — МО сверху и снизу по уровню к медиане страны: регион, отношение к медиане региона,
  робастный z внутри типа МО, подсказка из контекста (признаки с |z| выше порога), пояснение из
  ``eda.annotations``;
- T04 ``weighting`` — «типичное МО» против «типичного жителя» по семи категориям.

Расчёты для сюжета С2 «Ядра и периферия»: ρ уровня с индексом доступности рынков (со всеми МО, без
внутригородских территорий, без них и Севера, внутри регионов), частный ρ при контроле зарплаты внутри
регионов — критерий отказа С2 — с бутстреп-интервалом и проверкой без внутригородских территорий, ρ
с расстоянием до столицы региона внутри регионов.

Числа — чистыми функциями (``level_summary``, ``place_groups``, ``level_drivers``, ``partial_rho_ci``,
``wage_fit``, ``residual_labels``, ``context_hints``, ``extremes``, ``weighting_table``, ``map_classes``),
словесные выводы без чисел — тоже (``c2_verdict``, ``access_split``: фраза меняется вместе с данными),
раскладка подписей F05 — ``column_limits`` и ``column_order``; рисование — только через ``style`` и
``maps``. Параметры — ``eda.level`` конфига поверх ``DEFAULTS``; случайность (бутстреп) — только ``ctx.rng``.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from matplotlib import patheffects
from matplotlib.transforms import ScaledTranslation
from scipy.stats import theilslopes

from munnet import maps, style
from munnet.config import Config
from munnet.contracts import CATEGORY_CODES, MO_TYPES, YEARS
from munnet.eda import stats
from munnet.eda.base import FactKind, Finding, SectionContext
from munnet.eda.data import CONTEXT_YEAR, MONTHS_IN_YEAR

SECTION_ID = "e2"
TITLE = "Сколько тратят и от чего это зависит"

BASE_YEAR = YEARS[0]  # первый год панели: уровень для связи с зарплатой (контекст — 2023 год)

# Параметры раздела по умолчанию; конфиг ``eda.level`` их переопределяет (ключ за ключом, ``checks`` — тоже).
DEFAULTS: dict[str, Any] = {
    "map_classes": 7,  # квантильных классов на карте уровня (F04, spec_final Б.4)
    "hist_bins": 40,  # столбцов гистограммы уровня (F03), равных в логарифме
    "labels_per_side": 3,  # подписей МО выше и ниже линии на F05: всего 6 (правило В.2 — не больше 5–7)
    "hint_z": 2.0,  # |робастный z| признака контекста, чтобы попасть в подсказку T03 (Б.1, п. 8)
    "hint_max": 3,  # признаков в подсказке одного МО
    "md_per_side": 7,  # строк сверху и снизу в Markdown T03 (в отчёте ≤ 15 строк, полностью — в CSV)
    "names_in_fact": 3,  # МО в фактах top_names и bottom_names
    "rub_ticks": [10_000, 15_000, 20_000, 30_000, 50_000, 70_000, 100_000, 150_000],  # деления осей трат, ₽
    "wage_ticks": [20_000, 30_000, 50_000, 100_000, 200_000, 300_000, 500_000],  # деления оси зарплаты, ₽
    "axis_pad": 1.06,  # запас лог-оси за краями данных, множитель: ось не тянется до далёкого деления
    # Колонки подписей F05 с выносными линиями (доли осей): «выше линии» — сверху слева, стопкой вниз;
    # «ниже линии» — снизу справа, стопкой вверх. Там пусто при положительной связи трат и зарплаты.
    "label_columns": {"above": {"x": 0.02, "y": 0.98}, "below": {"x": 0.98, "y": 0.03}},
    "label_gap_pt": 6,  # просвет между подписями в колонке, pt
    "column_margin": 0.03,  # зазор между колонкой подписей F05 и ближайшей подписанной точкой, доля ширины
    "checks": {  # проверки заголовков (spec_final, В.5)
        "wmean_ratio_min": 1.2,  # F03: среднее по жителям ≥ 1,2 × медиана МО
        "eta2_min": 0.6,  # F04: η² региона для лог-уровня
        "rho_wage_min": 0.7,  # F05: ρ уровня с зарплатой
    },
}

# Короткие подписи типов МО (коды — ``contracts.MO_TYPES``).
MO_TYPE_LABELS: dict[str, str] = {
    "mr": "муниципальный район",
    "mo": "муниципальный округ",
    "go": "городской округ",
    "vgt": "внутригородская территория",
}

# Признаки контекста для подсказки T03: колонка ``mo`` -> (преобразование, «выше обычного», «ниже обычного»).
# Робастный z считается по всем МО с известным значением; «log» — для величин с длинным правым хвостом.
_Y = CONTEXT_YEAR
HINT_FEATURES: dict[str, tuple[str, str, str]] = {
    f"wage_{_Y}": ("log", "высокая зарплата", "низкая зарплата"),
    f"ndfl_pc_{_Y}": ("log", "высокий доход 5-НДФЛ на жителя", "низкий доход 5-НДФЛ на жителя"),
    f"urban_share_{_Y}": ("id", "много горожан", "мало горожан"),
    f"age_old_share_{_Y}": ("id", "много пожилых", "мало пожилых"),
    f"emp_sh_A_{_Y}": ("id", "много занятых в сельском хозяйстве", "мало занятых в сельском хозяйстве"),
    f"emp_sh_B_{_Y}": ("id", "много занятых в добыче", "мало занятых в добыче"),
    f"emp_sh_public_{_Y}": ("id", "много занятых в бюджетной сфере", "мало занятых в бюджетной сфере"),
    f"nights_pc_{_Y}": ("log1p", "много ночёвок приезжих", "мало ночёвок приезжих"),
    f"log_density_{_Y}": ("id", "высокая плотность населения", "низкая плотность населения"),
    "market_access": ("id", "высокая доступность рынков", "низкая доступность рынков"),
    "dist_capital_km": ("log1p", "далеко от столицы региона", "близко к столице региона"),
    "point_lat": ("id", "север", "юг"),
}
# Где признак подсказки не имеет смысла: колонка -> логические колонки ``mo``, у которых значение не берётся.
HINT_SKIP: dict[str, tuple[str, ...]] = {
    "dist_capital_km": ("is_capital", "is_inner_city"),
    f"wage_{_Y}": ("is_inner_city",),  # зарплата внутригородских — по месту работодателя, не жителей
}
# Метки подсказки без z: логическая колонка ``mo`` -> текст.
HINT_FLAGS: dict[str, str] = {"is_capital": "столица региона"}
_TRANSFORMS: dict[str, Callable[[pd.Series], pd.Series]] = {
    "id": lambda s: s,
    "log": lambda s: np.log(s.where(s > 0)),
    "log1p": lambda s: np.log1p(s.where(s >= 0)),
}

GROUP_TOP, GROUP_BOTTOM = "верх", "низ"
TIMES = "×"


# --- Параметры -----------------------------------------------------------------------------------


def params(cfg: Config) -> dict[str, Any]:
    """Параметры раздела: ``DEFAULTS``, поверх — ``eda.level`` конфига (вложенный ``checks`` — по ключам)."""
    out = copy.deepcopy(DEFAULTS)
    override = cfg["eda"].get("level") or {}
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), dict):
            out[key] = {**out[key], **value}
        else:
            out[key] = value
    return out


# --- Чистые функции ------------------------------------------------------------------------------


def _col(mo: pd.DataFrame, name: str) -> pd.Series:
    """Колонка ``mo`` как float64 (нет колонки — все пропуски)."""
    if name not in mo.columns:
        return pd.Series(np.nan, index=mo.index, dtype="float64")
    return pd.to_numeric(mo[name], errors="coerce").astype("float64")


def _flag(mo: pd.DataFrame, name: str) -> pd.Series:
    """Логическая колонка ``mo`` (нет колонки или пропуск — False)."""
    if name not in mo.columns:
        return pd.Series(False, index=mo.index)
    return mo[name].astype("boolean").fillna(False).astype(bool)


def year_weights(mo: pd.DataFrame, year: int) -> pd.Series:
    """Веса «типичного жителя» года: ``pop_<год>`` (среднегодовое население), пропуск — ``weight`` (Б.2)."""
    return _col(mo, f"pop_{year}").fillna(_col(mo, "weight"))


def _median(x: pd.Series) -> float:
    x = pd.Series(x, dtype="float64").dropna()
    return float(x.median()) if len(x) else float("nan")


def _ratio(a: float, b: float) -> float:
    return float(a / b) if np.isfinite(a) and np.isfinite(b) and b != 0 else float("nan")


def level_summary(
    mo: pd.DataFrame, *, year: int = YEARS[-1], base_year: int = BASE_YEAR, wage_year: int = CONTEXT_YEAR
) -> dict[str, float]:
    """Сколько тратят: медиана, среднее и среднее по жителям уровня двух лет, перцентили, η² региона, медианы
    по типам МО, доля трат в зарплате.

    Уровень года — ``level_<год>`` (только МО с 12 месяцами года). «Типичное МО» — медиана по МО без весов,
    «типичный житель» — среднее с весами ``year_weights`` (МО без населения в него не входят). η² региона
    считается на логарифме уровня (Б.1, п. 2) — со всеми МО и без внутригородских территорий Москвы
    и Петербурга. Ключи — ключи фактов раздела без префикса.
    """
    out: dict[str, float] = {}
    for y in (base_year, year):
        level = _col(mo, f"level_{y}")
        w = year_weights(mo, y)
        out[f"n_level_{y}"] = int(level.notna().sum())
        out[f"level_median_{y}"] = _median(level)
        out[f"level_mean_{y}"] = float(level.mean()) if level.notna().any() else float("nan")
        out[f"level_wmean_{y}"] = stats.weighted_mean(level, w)
        out[f"n_level_weighted_{y}"] = int((level.notna() & w.notna() & (w > 0)).sum())
    level = _col(mo, f"level_{year}")
    p10, p90 = (float(level.quantile(q)) if level.notna().any() else float("nan") for q in (0.1, 0.9))
    out[f"level_p10_{year}"], out[f"level_p90_{year}"] = p10, p90
    out[f"p90_p10_{year}"] = _ratio(p90, p10)
    out[f"wmean_to_median_{year}"] = _ratio(out[f"level_wmean_{year}"], out[f"level_median_{year}"])
    log_level = np.log(level.where(level > 0))
    region = mo["region_code"]
    inner = _flag(mo, "is_inner_city")
    out["eta2_region_level"] = stats.eta2(log_level, region)
    out["eta2_region_level_no_inner"] = stats.eta2(log_level[~inner], region[~inner])
    s2w = _col(mo, f"spend_to_wage_{wage_year}")
    out["spend_to_wage_median"] = _median(s2w)
    out["n_spend_to_wage"] = int(s2w.notna().sum())
    mo_type = mo["mo_type"].astype(str)
    for t in MO_TYPES:
        part = level[mo_type == t]
        out[f"median_level_{t}"] = _median(part)
        out[f"n_level_{t}"] = int(part.notna().sum())
    return out


def place_groups(mo: pd.DataFrame, north_lat: float, *, year: int = YEARS[-1]) -> dict[str, float]:
    """Три группы мест: внутригородские территории Москвы и Петербурга, Север (широта точки МО
    ≥ ``north_lat``, без внутригородских) и остальные МО.

    По группам — медиана уровня к медиане страны (``level_rel_<год>``) и медиана индекса доступности
    рынков; для остальных МО — ρ ``level_<год>`` с доступностью (``rho_level_access_rest``): видно,
    держится ли общая связь без столиц и Севера, которые тянут её в разные стороны.
    """
    rel = _col(mo, f"level_rel_{year}")
    access = _col(mo, "market_access")
    inner = _flag(mo, "is_inner_city")
    north = (_col(mo, "point_lat") >= north_lat) & ~inner
    rest = ~inner & ~north
    out: dict[str, float] = {"n_rel": int(rel.notna().sum())}
    for name, mask in (("inner", inner), ("north", north), ("rest", rest)):
        out[f"median_rel_{name}"] = _median(rel[mask])
        out[f"n_rel_{name}"] = int(rel[mask].notna().sum())
        out[f"median_access_{name}"] = _median(access[mask])
    level = _col(mo, f"level_{year}")
    rho, n = stats.spearman(level[rest], access[rest])
    out["rho_level_access_rest"], out["n_level_access_rest"] = float(rho), int(n)
    return out


def access_split(rho_all: float, rho_no_inner: float, rho_rest: float) -> str:
    """Хвост фразы о связи уровня с доступностью рынков: «: Москва с Петербургом поднимают связь, а Север
    опускает её» — только если данные это показывают (без внутригородских ρ ниже, чем со всеми МО, а без
    Севера — снова выше); иначе пустая строка, и в тексте остаются одни числа."""
    if not all(np.isfinite([rho_all, rho_no_inner, rho_rest])):
        return ""
    if rho_no_inner < rho_all and rho_no_inner < rho_rest:
        return ": Москва с\u00a0Петербургом поднимают связь, а\u00a0Север опускает её"
    return ""


def c2_verdict(est: float, lo: float, hi: float, est_no_inner: float, threshold: float) -> str:
    """Словесный вывод о критерии отказа С2 (|частный ρ| ≥ порога) по оценке, бутстреп-интервалу и
    проверке без внутригородских территорий. Без чисел: числа текста — только из фактов.

    Интервал переводится в модуль по знаку оценки. Весь интервал выше порога — «выполняется уверенно»,
    весь ниже — «не выполняется», иначе — «на границе» (с оговоркой, если без внутригородских территорий
    оценка по другую сторону порога).
    """
    if not all(np.isfinite([est, lo, hi, threshold])):
        return "критерий не посчитан: не хватает данных"
    a = abs(est)
    lo_m, hi_m = (lo, hi) if est >= 0 else (-hi, -lo)
    if lo_m >= threshold:
        return "весь интервал бутстрепа выше порога, критерий выполняется уверенно"
    if hi_m < threshold:
        return "весь интервал бутстрепа ниже порога, критерий не выполняется"
    text = f"оценка {'выше' if a >= threshold else 'ниже'} порога, но интервал бутстрепа накрывает порог"
    if np.isfinite(est_no_inner) and (abs(est_no_inner) >= threshold) != (a >= threshold):
        text += ", а без внутригородских территорий оценка по другую сторону порога"
    return text + ". Критерий на самой границе, и решение о сюжете не стоит строить на одном этом числе"


def level_drivers(
    mo: pd.DataFrame, *, year: int = YEARS[-1], wage_year: int = CONTEXT_YEAR
) -> dict[str, float]:
    """От чего зависит уровень: ранговые связи с зарплатой, доступностью рынков и расстоянием до столицы.

    - ``rho_level_wage`` — ρ Спирмена ``level_<wage_year>`` с ``wage_<wage_year>`` (год зарплаты); внутри
      регионов (ранги, центрированные по региону) и без внутригородских территорий (их зарплата — по месту
      работодателя);
    - ``rho_level_access`` — ρ ``level_<year>`` с ``market_access``; то же внутри регионов, без
      внутригородских территорий и с уровнем года зарплаты (``rho_level_access_<wage_year>``);
    - ``partial_rho_access_within`` — частный ρ ``level_<year>`` с ``market_access`` при контроле
      ``wage_<wage_year>`` внутри регионов (``stats.partial_spearman``): даёт ли доступность рынков что-то
      сверх зарплаты и региона — критерий отказа С2; ``_no_inner`` — то же без внутригородских территорий
      (у них зарплата по месту работодателя, а Москва и Петербург — два крупнейших «региона»):
      проверка устойчивости критерия;
    - ``rho_level_dist_capital_within`` — ρ ``level_<year>`` с автодорожным расстоянием до столицы региона
      внутри регионов;
    - ``n_access_wage`` — МО, где известны уровень, доступность рынков и зарплата (покрытие сюжета С2).
    """
    region = mo["region_code"].reset_index(drop=True)
    inner = _flag(mo, "is_inner_city").reset_index(drop=True)
    base = _col(mo, f"level_{wage_year}").reset_index(drop=True)
    wage = _col(mo, f"wage_{wage_year}").reset_index(drop=True)
    level = _col(mo, f"level_{year}").reset_index(drop=True)
    access = _col(mo, "market_access").reset_index(drop=True)
    dist = _col(mo, "dist_capital_km").reset_index(drop=True)
    out: dict[str, float] = {}
    out["rho_level_wage"], out["n_level_wage"] = stats.spearman(base, wage)
    out["rho_level_wage_within"], _ = stats.spearman_within(base, wage, region)
    out["rho_level_wage_no_inner"], out["n_level_wage_no_inner"] = stats.spearman(base[~inner], wage[~inner])
    out["rho_level_access"], out["n_level_access"] = stats.spearman(level, access)
    out["rho_level_access_within"], _ = stats.spearman_within(level, access, region)
    out["rho_level_access_no_inner"], _ = stats.spearman(level[~inner], access[~inner])
    out[f"rho_level_access_{wage_year}"], _ = stats.spearman(base, access)
    out["partial_rho_access_within"], out["n_partial_access"] = stats.partial_spearman(
        level, access, pd.DataFrame({"wage": wage}), groups=region
    )
    keep = ~inner
    out["partial_rho_access_within_no_inner"], out["n_partial_access_no_inner"] = stats.partial_spearman(
        level[keep].reset_index(drop=True),
        access[keep].reset_index(drop=True),
        pd.DataFrame({"wage": wage[keep].reset_index(drop=True)}),
        groups=region[keep].reset_index(drop=True),
    )
    out["n_access_wage"] = int((level.notna() & access.notna() & wage.notna()).sum())
    out["rho_level_dist_capital_within"], out["n_level_dist_capital"] = stats.spearman_within(
        level, dist, region
    )
    return {k: (int(v) if k.startswith("n_") else float(v)) for k, v in out.items()}


# Существительное к короткому названию-прилагательному: «Яльчикский» -> «Яльчикский округ».
TYPE_NOUNS: dict[str, str] = {"mr": "район", "mo": "округ"}
ADJECTIVE_ENDINGS: tuple[str, ...] = ("ий", "ый", "ой")


def display_name(name: str, mo_type: str) -> str:
    """Название МО для таблиц и подписей: к одному слову-прилагательному района или округа добавляется
    «район» или «округ» («Яльчикский» -> «Яльчикский округ»); остальные названия — как есть."""
    noun = TYPE_NOUNS.get(str(mo_type))
    if noun and " " not in name and name.endswith(ADJECTIVE_ENDINGS):
        return f"{name} {noun}"
    return name


def _names(mo: pd.DataFrame) -> pd.Series:
    """Название МО для людей: ``name_short`` (если есть, иначе ``name``) через ``display_name``."""
    base = mo["name_short"].fillna(mo["name"]) if "name_short" in mo.columns else mo["name"]
    types = mo["mo_type"].astype(str) if "mo_type" in mo.columns else pd.Series("", index=mo.index)
    return pd.Series(
        [display_name(str(n), t) for n, t in zip(base, types, strict=True)], index=mo.index, dtype="object"
    )


def names_with_regions(names: Sequence[str], regions: Sequence[str]) -> str:
    """Перечень МО с регионами без повторов: «Арбат и Хамовники (Москва), Анадырь (Чукотский автономный
    округ)». Регионы — в порядке первого появления, МО внутри региона — в исходном порядке."""
    groups: dict[str, list[str]] = {}
    for name, region in zip(names, regions, strict=True):
        groups.setdefault(str(region), []).append(str(name))
    parts = []
    for region, items in groups.items():
        listed = items[0] if len(items) == 1 else ", ".join(items[:-1]) + " и " + items[-1]
        parts.append(f"{listed} ({region})")
    return ", ".join(parts)


def wage_fit(
    mo: pd.DataFrame, *, year: int = CONTEXT_YEAR, exclude_inner: bool = True
) -> tuple[pd.DataFrame, float, float]:
    """Робастная линия ln уровня на ln зарплаты года (Тейл — Сен) и остатки всех МО с обоими значениями.

    Линию строят МО без внутригородских территорий (``exclude_inner``): их зарплата — по месту работодателя,
    а траты — жителей. Возвращает таблицу (``territory_id``, ``name``, ``region_name``, ``is_inner_city``,
    ``wage``, ``level``, ``fitted``, ``resid_log`` — ln(уровень / линия): плюс — тратят больше, чем
    «положено» по зарплате), наклон и свободный член в логарифмах.
    """
    wage, level = _col(mo, f"wage_{year}"), _col(mo, f"level_{year}")
    ok = (wage > 0) & (level > 0)
    d = pd.DataFrame(
        {
            "territory_id": mo.loc[ok, "territory_id"].to_numpy(),
            "name": _names(mo).loc[ok].to_numpy(),
            "region_name": mo.loc[ok, "region_name"].astype(str).to_numpy(),
            "is_inner_city": _flag(mo, "is_inner_city").loc[ok].to_numpy(),
            "wage": wage[ok].to_numpy(),
            "level": level[ok].to_numpy(),
        }
    )
    x, y = np.log(d["wage"].to_numpy()), np.log(d["level"].to_numpy())
    base = ~d["is_inner_city"].to_numpy() if exclude_inner else np.ones(len(d), dtype=bool)
    if base.sum() < 2 or np.ptp(x[base]) == 0:
        return d.assign(fitted=np.nan, resid_log=np.nan), float("nan"), float("nan")
    slope = float(theilslopes(y[base], x[base])[0])
    # Свободный член — медиана остатков (у theilslopes он median(y) − наклон · median(x) и сдвигается
    # несимметричными выбросами).
    intercept = float(np.median(y[base] - slope * x[base]))
    pred = intercept + slope * x
    d["fitted"] = np.exp(pred)
    d["resid_log"] = y - pred
    return d, float(slope), float(intercept)


def partial_rho_ci(
    mo: pd.DataFrame,
    rng: np.random.Generator,
    n_boot: int,
    *,
    year: int = YEARS[-1],
    wage_year: int = CONTEXT_YEAR,
    level: float = 0.95,
) -> tuple[float, float]:
    """Перцентильный бутстреп-интервал частного ρ уровня с доступностью рынков при контроле зарплаты внутри
    регионов (как в ``level_drivers``): МО выбираются с возвращением ``n_boot`` раз.

    Критерий отказа С2 — порог по этому числу, а оценка может лежать у самого порога: интервал показывает,
    насколько уверенно она по ту или другую сторону.
    """
    cols = pd.DataFrame(
        {
            "level": _col(mo, f"level_{year}"),
            "access": _col(mo, "market_access"),
            "wage": _col(mo, f"wage_{wage_year}"),
            "region": mo["region_code"],
        }
    ).dropna()
    n = len(cols)
    if n <= stats.MIN_PAIRS or n_boot < 1:
        return float("nan"), float("nan")
    lv, ac, wg, rg = (cols[c].to_numpy() for c in ("level", "access", "wage", "region"))
    boots = np.empty(n_boot)
    for b in range(n_boot):
        i = rng.integers(0, n, n)
        boots[b] = stats.partial_spearman(lv[i], ac[i], pd.DataFrame({"wage": wg[i]}), groups=rg[i])[0]
    alpha = (1 - level) / 2
    lo, hi = np.nanquantile(boots, [alpha, 1 - alpha])
    return float(lo), float(hi)


def residual_labels(fit: pd.DataFrame, per_side: int, *, include_inner: bool = False) -> pd.DataFrame:
    """МО для подписей на F05: ``per_side`` крупнейших положительных и отрицательных остатков от линии.

    Внутригородские территории по умолчанию не подписываются: их остаток — артефакт зарплаты по месту
    работодателя. Колонка ``side``: «выше» или «ниже». Порядок устойчив: по |остатку|, затем
    по ``territory_id``.
    """
    d = fit.dropna(subset=["resid_log"])
    if not include_inner:
        d = d.loc[~d["is_inner_city"].astype(bool)]
    order = ["resid_log", "territory_id"]
    above = d.loc[d["resid_log"] > 0].sort_values(order, ascending=[False, True], kind="mergesort")
    below = d.loc[d["resid_log"] < 0].sort_values(order, ascending=[True, True], kind="mergesort")
    return pd.concat(
        [above.head(per_side).assign(side="выше"), below.head(per_side).assign(side="ниже")],
        ignore_index=True,
    )


def context_hints(
    mo: pd.DataFrame,
    z_threshold: float,
    max_features: int,
    features: Mapping[str, tuple[str, str, str]] = HINT_FEATURES,
) -> pd.Series:
    """Подсказка к выбросу: чем МО выделяется в контексте — признаки с |робастный z| > ``z_threshold``.

    Робастный z признака считается по всем МО с известным значением (``HINT_SKIP`` — где признак не имеет
    смысла, например расстояние до столицы у самой столицы). До ``max_features`` признаков по убыванию |z|,
    перед ними — метки ``HINT_FLAGS`` («столица региона»). Текст: «высокая зарплата (z = 3,1); мало
    горожан (z = −2,4)». Это описание, а не причина: объяснение — гипотеза, пока не проверено. Индекс —
    ``territory_id``; пустая строка — ничего необычного (или контекста нет).
    """
    rows: list[pd.DataFrame] = []
    for col, (transform, high, low) in features.items():
        if col not in mo.columns:
            continue
        raw = _col(mo, col)
        for flag in HINT_SKIP.get(col, ()):
            raw = raw.mask(_flag(mo, flag))
        z = stats.robust_z(_TRANSFORMS[transform](raw))
        hit = z.abs() > z_threshold
        if hit.any():
            rows.append(
                pd.DataFrame(
                    {
                        "territory_id": mo.loc[hit, "territory_id"].to_numpy(),
                        "absz": z[hit].abs().to_numpy(),
                        "text": [
                            f"{high if v > 0 else low} (z{style.NBSP}={style.NBSP}{style.fmt_num(v, 1)})"
                            for v in z[hit]
                        ],
                    }
                )
            )
    ids = pd.Index(mo["territory_id"].to_numpy(), name="territory_id")
    flags = pd.Series("", index=ids, dtype="object")
    for flag, text in HINT_FLAGS.items():
        mask = _flag(mo, flag).to_numpy()
        flags[mask] = [f"{t}; {text}" if t else text for t in flags[mask]]
    if rows:
        hits = pd.concat(rows, ignore_index=True).sort_values(
            ["territory_id", "absz"], ascending=[True, False], kind="mergesort"
        )
        joined = hits.groupby("territory_id")["text"].agg(lambda s: "; ".join(s.head(max_features)))
        joined = joined.reindex(ids).fillna("")
    else:
        joined = pd.Series("", index=ids, dtype="object")
    both = [f"{a}; {b}" if a and b else a or b for a, b in zip(flags, joined, strict=True)]
    return pd.Series(both, index=ids, dtype="object")


def normalize_annotations(annotations: Mapping[Any, Any] | None) -> dict[int, str]:
    """``eda.annotations`` -> {territory_id (int): пояснение}; ключи YAML бывают и строками."""
    return {int(k): str(v) for k, v in (annotations or {}).items()}


def extremes(
    mo: pd.DataFrame,
    annotations: Mapping[Any, Any] | None,
    top_n: int,
    *,
    year: int = YEARS[-1],
    hint_z: float = DEFAULTS["hint_z"],
    hint_max: int = DEFAULTS["hint_max"],
    outlier_z: float = float("inf"),
) -> pd.DataFrame:
    """T03: ``top_n`` МО сверху и снизу по уровню к медиане страны ``level_rel_<год>``.

    Колонки: ``group`` (верх / низ), ``rank`` (1 — самый крайний), ``territory_id``, ``name``,
    ``region_name``, ``mo_type`` (подпись), ``level`` (₽, NA при неполном годе), ``level_rel`` (к медиане
    МО месяца), ``rel_region`` (к медиане ``level_rel`` своего региона), ``z_type`` (робастный z логарифма
    ``level_rel`` внутри типа МО), ``outlier`` (|z_type| > ``outlier_z``), ``hint`` (``context_hints``),
    ``annotation`` (``eda.annotations``). Порядок: верх по убыванию, низ по возрастанию; при равенстве —
    по ``territory_id``.
    """
    rel = _col(mo, f"level_rel_{year}")
    region = mo["region_code"]
    region_median = rel.groupby(region).transform("median")
    z_type = stats.robust_z(np.log(rel.where(rel > 0)), groups=mo["mo_type"].astype(str))
    hints = context_hints(mo, hint_z, hint_max)
    notes = normalize_annotations(annotations)
    table = pd.DataFrame(
        {
            "territory_id": mo["territory_id"].to_numpy(),
            "name": _names(mo).to_numpy(),
            "region_name": mo["region_name"].astype(str).to_numpy(),
            "mo_type": mo["mo_type"]
            .astype(str)
            .map(MO_TYPE_LABELS)
            .fillna(mo["mo_type"].astype(str))
            .to_numpy(),
            "level": _col(mo, f"level_{year}").to_numpy(),
            "level_rel": rel.to_numpy(),
            "rel_region": (rel / region_median).to_numpy(),
            "z_type": z_type.to_numpy(),
        }
    )
    table = table.loc[table["level_rel"].notna()]
    table["outlier"] = table["z_type"].abs() > outlier_z
    table["hint"] = table["territory_id"].map(hints).fillna("")
    table["annotation"] = table["territory_id"].map(notes).fillna("")
    parts = []
    for group, descending in ((GROUP_TOP, True), (GROUP_BOTTOM, False)):
        order = table.sort_values(
            ["level_rel", "territory_id"], ascending=[not descending, True], kind="mergesort"
        )
        part = order.head(top_n).copy()
        part.insert(0, "rank", np.arange(1, len(part) + 1))
        part.insert(0, "group", group)
        parts.append(part)
    out = pd.concat(parts, ignore_index=True)
    out["territory_id"] = out["territory_id"].astype("int32")
    return out


def display_order(table: pd.DataFrame, per_side: int) -> pd.DataFrame:
    """Порядок строк T03 для отчёта: первые ``per_side`` сверху, первые ``per_side`` снизу, затем остальные
    (Markdown показывает голову таблицы, CSV — всё; ``group`` и ``rank`` сохраняют исходный порядок)."""
    head = table["rank"] <= per_side
    key = pd.DataFrame({"tail": ~head, "bottom": table["group"] == GROUP_BOTTOM, "rank": table["rank"]})
    order = key.sort_values(["tail", "bottom", "rank"], kind="mergesort").index
    return table.loc[order].reset_index(drop=True)


def weighting_table(panel_wide: pd.DataFrame, mo: pd.DataFrame, *, year: int = YEARS[-1]) -> pd.DataFrame:
    """T04: «типичное МО» против «типичного жителя» по семи категориям (включая «Все категории» и «Прочее»).

    Среднее ``v_<категория>`` за 12 месяцев года — только МО со всеми 12 месяцами (как ``level_<год>``).
    По МО: медиана, среднее без весов, среднее с весами ``year_weights`` (МО без населения не входят)
    и отношение среднего по жителям к медиане МО.
    """
    part = panel_wide.loc[panel_wide["year"] == year]
    n = part.groupby("territory_id").size()
    full = n.index[n == MONTHS_IN_YEAR]
    cols = [f"v_{c}" for c in CATEGORY_CODES]
    means = part.loc[part["territory_id"].isin(full)].groupby("territory_id")[cols].mean().astype("float64")
    w = year_weights(mo, year).set_axis(mo["territory_id"].to_numpy()).reindex(means.index)
    rows = []
    for c in CATEGORY_CODES:
        v = means[f"v_{c}"]
        median = _median(v)
        wmean = stats.weighted_mean(v, w)
        rows.append(
            {
                "category": c,
                "label": style.LABELS[c],
                "n_mo": int(v.notna().sum()),
                "n_weighted": int((v.notna() & w.notna() & (w > 0)).sum()),
                "median_mo": median,
                "mean_mo": float(v.mean()) if len(v) else float("nan"),
                "wmean_residents": wmean,
                "ratio_wmean_median": _ratio(wmean, median),
            }
        )
    return pd.DataFrame(rows)


# Короткие предлоги и союзы, которые не остаются в конце строки: однобуквенные (правило ru-text) и
# двухбуквенные предлоги (типографская традиция: «в среднем по / жителям» читается хуже, чем
# «в среднем / по жителям»).
SHORT_WORDS: tuple[str, ...] = (
    "в",
    "к",
    "с",
    "о",
    "у",
    "и",
    "а",
    "по",
    "на",
    "до",
    "из",
    "от",
    "за",
    "об",
    "со",
)
_SHORT_WORD = re.compile(r"(?<![\w-])(" + "|".join(SHORT_WORDS) + r") ", re.IGNORECASE)


def typo(text: str) -> str:
    """Типографика подписей графика: неразрывный пробел перед «—» и после коротких предлогов и союзов
    (``SHORT_WORDS``), чтобы перенос строки в ``style.finish`` не оставлял их в конце строки."""
    return _SHORT_WORD.sub(lambda m: m.group(1) + style.NBSP, text.replace(" — ", f"{style.NBSP}— "))


# Сокращения в названиях регионов для подписей точек: «Астраханская область» -> «Астраханская обл.».
REGION_ABBREVIATIONS: tuple[tuple[str, str], ...] = (
    ("автономный округ", "АО"),
    ("автономная область", "АО"),
    ("Республика", "Респ."),
    ("область", "обл."),
)


def short_region(name: str) -> str:
    """Короткое название региона для подписи на графике: «обл.», «Респ.», «АО»."""
    for full, short in REGION_ABBREVIATIONS:
        name = name.replace(full, short)
    return name


def point_label(name: str, region: str) -> str:
    """Подпись МО на графике в две строки: название и короткий регион (одноимённые МО есть в разных
    регионах)."""
    return f"{name}\n{short_region(region)}"


def times_fmt(v: float, decimals: int = 2) -> str:
    """Отношение к медиане: 1.954 -> «1,95×»."""
    text = style.fmt_num(v, decimals)
    return text if text == style.NA_TEXT else text + TIMES


def map_classes(values: pd.Series, weights: pd.Series, k: int) -> tuple[pd.DataFrame, pd.Series]:
    """Классы карты F04 (те же квантильные границы, что у ``maps.russia_map``) и население в каждом.

    ``values`` и ``weights`` — с индексом ``territory_id``. Возвращает таблицу классов (``cls``, ``label``,
    ``lo``, ``hi``, ``n_mo``, ``population``, ``pop_share`` — доля жителей МО карты с известным населением)
    и номер класса каждого МО. Карта площадью «врёт» о людях: доля жителей в классе — для текста.
    """
    v = pd.Series(values, dtype="float64").dropna()
    edges, labels = maps.quantile_bins(v, k, times_fmt)
    cls = maps.classify(v, edges)
    w = pd.Series(weights, dtype="float64").reindex(v.index)
    rows = []
    for i, label in enumerate(labels):
        mask = cls == i
        rows.append(
            {
                "cls": i,
                "label": label,
                "lo": float(edges[i]),
                "hi": float(edges[i + 1]),
                "n_mo": int(mask.sum()),
                "population": float(w[mask].sum()),
            }
        )
    table = pd.DataFrame(rows)
    total = table["population"].sum()
    table["pop_share"] = table["population"] / total if total > 0 else np.nan
    return table, cls


def plural(n: int, forms: tuple[str, str, str]) -> str:
    """Слово при числе: 1 регион, 2 региона, 5 регионов, 21 регион, 11 регионов."""
    n = abs(int(n))
    if n % 10 == 1 and n % 100 != 11:
        return forms[0]
    if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
        return forms[1]
    return forms[2]


def na_label(min_months: int) -> str:
    """Пункт легенды F04 для МО панели без ``level_rel_<год>``: «меньше 6 месяцев данных» (порог —
    ``eda.level_rel_min_months``; год карты назван в подзаголовке, а длинный пункт выталкивает легенду
    за край рисунка)."""
    word = plural(min_months, ("месяца", "месяцев", "месяцев"))
    return f"меньше {min_months} {word} данных"


def axis_ticks(
    candidates: Sequence[float], lo: float, hi: float, pad: float = 1.0
) -> tuple[list[float], tuple[float, float]]:
    """Деления лог-оси из ``candidates`` и пределы оси: данные [lo; hi] с запасом ``pad`` (множитель).

    Пределы — [lo / pad; hi · pad], деления — кандидаты внутри них: ось не тянется до далёкого деления,
    и у края не остаётся пустой полосы. Если внутри меньше двух кандидатов, добавляются ближайшие снаружи
    (по одному с каждой стороны) и пределы расширяются до них; нет и таких — деления на краях данных.
    """
    c = sorted(float(t) for t in candidates)
    a, b = float(lo) / pad, float(hi) * pad
    ticks = [t for t in c if a <= t <= b]
    if len(ticks) < 2:
        below = [t for t in c if t < a]
        above = [t for t in c if t > b]
        ticks = ([below[-1]] if below else []) + ticks + ([above[0]] if above else [])
    if len(ticks) < 2:
        ticks = [float(lo), float(hi)]
    return ticks, (min(a, ticks[0]), max(b, ticks[-1]))


# --- Факты: вид и определение --------------------------------------------------------------------


def fact_kind(key: str) -> FactKind:
    """Вид форматирования факта раздела по ключу."""
    if key.startswith(("n_", "median_access_")):
        return "int"
    if key.startswith(("level_", "median_level_")):
        return "rub"
    if key.startswith("partial_rho_"):
        return "num3"  # критерий отказа С2 у самого порога: два знака не различат 0,204 и 0,198
    if key.startswith("rho_"):
        return "rho"
    if key.startswith(("pop_share_", "spend_to_wage")):
        return "pct"
    if key.startswith("p90_p10_"):
        return "num1"
    if key.startswith(("eta2_", "wmean_to_median_", "median_rel_")):
        return "num2"
    if key.endswith("_names"):
        return "str"
    raise KeyError(f"e2: не задан вид факта {key}")


def fact_note(key: str, year: int, wage_year: int) -> str:
    """Определение и выборка факта (поле ``note`` в ``facts.json``): сначала точный ключ, затем шаблон по
    префиксу (ключи с годом или типом МО)."""
    level = "уровень — среднее v_all за 12 месяцев года, ₽ на жителя в месяц, номинал; МО с 12 месяцами года"
    exact = {
        "eta2_region_level": f"η² региона (межрегиональная сумма квадратов / общая) для ln level_{year}",
        "eta2_region_level_no_inner": f"η² региона для ln level_{year} без внутригородских территорий",
        "spend_to_wage_median": f"медиана level_{wage_year} / wage_{wage_year} (зарплата Росстата, без МСП)",
        "n_spend_to_wage": f"МО с level_{wage_year} и wage_{wage_year}",
        "median_rel_inner": f"медиана level_rel_{year} внутригородских территорий Москвы и Петербурга",
        "median_rel_north": f"медиана level_rel_{year} МО с широтой ≥ eda.north_lat, без внутригородских",
        "median_rel_rest": f"медиана level_rel_{year} прочих МО",
        "n_rel": f"МО с level_rel_{year} (не меньше eda.level_rel_min_months месяцев года)",
        "n_rel_inner": f"внутригородские территории с level_rel_{year}",
        "n_rel_north": f"МО Севера с level_rel_{year}",
        "n_rel_rest": f"прочие МО с level_rel_{year}",
        "median_access_inner": "медиана индекса доступности рынков (0–1000) внутригородских территорий",
        "median_access_north": (
            "медиана индекса доступности рынков МО с широтой ≥ eda.north_lat, без внутригородских"
        ),
        "median_access_rest": "медиана индекса доступности рынков прочих МО (не Север и не внутригородские)",
        "rho_level_access_rest": (
            f"ρ level_{year} и индекса доступности рынков без внутригородских территорий и без Севера"
        ),
        "n_level_access_rest": f"МО с level_{year} и доступностью рынков без внутригородских и без Севера",
        "rho_level_wage": f"ρ Спирмена level_{wage_year} и wage_{wage_year} (зарплата по месту работы)",
        "rho_level_wage_within": (
            f"ρ level_{wage_year} и wage_{wage_year} внутри регионов (ранги минус среднее региона)"
        ),
        "rho_level_wage_no_inner": f"ρ level_{wage_year} и wage_{wage_year} без внутригородских территорий",
        "n_level_wage": (
            f"МО с level_{wage_year} и wage_{wage_year}; ориентир Б.4 (2161 МО, ρ 0,865) посчитан по "
            f"среднему v_all за любые месяцы {wage_year} года, здесь по Б.2 — только МО с 12 месяцами года"
        ),
        "n_level_wage_no_inner": f"МО с level_{wage_year} и wage_{wage_year} без внутригородских территорий",
        "rho_level_access": (
            f"ρ Спирмена level_{year} и индекса доступности рынков (по Б.4); ориентир Б.4 (0,14) посчитан "
            f"по среднему v_all за любые месяцы {wage_year} года; с level_{wage_year} — "
            f"rho_level_access_{wage_year}; с уровнем {year} года связь сильнее"
        ),
        "rho_level_access_within": f"ρ level_{year} и индекса доступности рынков внутри регионов",
        "rho_level_access_no_inner": (
            f"ρ level_{year} и индекса доступности рынков без внутригородских территорий Москвы и Петербурга"
        ),
        f"rho_level_access_{wage_year}": (
            f"ρ level_{wage_year} (12 месяцев года) и индекса доступности рынков; ориентир Б.4 (0,14) "
            f"посчитан так же, но по среднему v_all за любые месяцы {wage_year} года, включая неполный год"
        ),
        "n_level_access": f"МО с level_{year} и индексом доступности рынков",
        "partial_rho_access_within": (
            f"частный ρ level_{year} и доступности рынков при контроле wage_{wage_year} внутри регионов "
            "(критерий отказа С2)"
        ),
        "n_partial_access": "МО в частном ρ (регионы из одного МО отброшены)",
        "partial_rho_access_within_no_inner": (
            "partial_rho_access_within без внутригородских территорий Москвы и Петербурга: устойчивость "
            "критерия отказа С2"
        ),
        "n_partial_access_no_inner": "МО в частном ρ без внутригородских территорий",
        "partial_rho_threshold": "порог критерия отказа С2: eda.rejection.s2_partial_rho_min",
        "partial_rho_access_within_lo": (
            "нижняя граница 95%-го перцентильного бутстреп-интервала partial_rho_access_within"
        ),
        "partial_rho_access_within_hi": (
            "верхняя граница 95%-го перцентильного бутстреп-интервала partial_rho_access_within"
        ),
        "n_access_wage": f"МО с level_{year}, доступностью рынков и wage_{wage_year} (покрытие С2)",
        "rho_level_dist_capital_within": (
            f"ρ level_{year} и расстояния по дорогам до столицы региона внутри регионов"
        ),
        "n_level_dist_capital": (
            f"МО с level_{year} и расстоянием до столицы региона (регионы из одного МО отброшены)"
        ),
        "pop_share_top_class": f"доля жителей МО карты F04 в верхнем квантильном классе level_rel_{year}",
        "pop_share_bottom_class": f"доля жителей МО карты F04 в нижнем квантильном классе level_rel_{year}",
        "top_names": f"МО с наибольшим level_rel_{year} (T03), в скобках — регион",
        "bottom_names": f"МО с наименьшим level_rel_{year} (T03), в скобках — регион",
        "wage_above_names": (
            f"МО без внутригородских с наибольшим положительным остатком ln level_{wage_year} от робастной "
            f"линии по ln wage_{wage_year} (подписаны на F05)"
        ),
        "wage_below_names": (
            f"МО без внутригородских с наибольшим отрицательным остатком ln level_{wage_year} от робастной "
            f"линии по ln wage_{wage_year} (подписаны на F05)"
        ),
    }
    if key in exact:
        return exact[key]
    prefixes = {
        "level_median_": (
            f"медиана уровня по МО («типичное МО»); {level}; ориентир Б.4 посчитан по среднему за любые "
            "месяцы года, включая неполный год"
        ),
        "level_mean_": f"среднее уровня по МО без весов; {level}",
        "level_wmean_": f"среднее уровня с весами — население года («типичный житель»); {level}",
        "level_p10_": f"10-й перцентиль уровня по МО; {level}",
        "level_p90_": f"90-й перцентиль уровня по МО; {level}",
        "p90_p10_": "отношение 90-го перцентиля уровня к 10-му",
        "wmean_to_median_": "среднее по жителям / медиана МО",
        "n_level_weighted_": "МО с уровнем года и известным населением (входят в среднее по жителям)",
        "median_level_": f"медиана level_{year} по МО типа (код типа — в ключе)",
        "n_level_": f"МО с уровнем (12 месяцев года): всего за год в ключе или МО типа в ключе ({year})",
    }
    for prefix, note in prefixes.items():
        if key.startswith(prefix):
            return note
    return ""


# --- Графики -------------------------------------------------------------------------------------

MARK_STYLES = ("-", "--", ":")  # медиана МО, среднее МО, среднее по жителям
MARK_LABEL_Y = (0.97, 0.83, 0.69)  # высота подписей отметок F03, доля высоты осей
MARK_LABEL_PAD = 0.15  # поле белой плашки под подписью отметки F03, доля кегля
HEADROOM = 1.3  # запас над столбцами F03 под подписи отметок
LINE_WIDTH = 1.2
LEADER_WIDTH = 0.5  # выносные линии подписей F05
POINT_SIZE = 9
INNER_SIZE = 14
LABELED_SIZE = 18  # подписанные точки F05


HIGHLIGHT_LABEL = "внутригородские территории\nМосквы и Петербурга"  # подпись выделенных столбцов F03
HIGHLIGHT_LABEL_PAD_PT = 3  # от верха столбца до подписи, pt


def _log_axis(ax, axis: str, candidates: Sequence[float], lo: float, hi: float, pad: float) -> None:
    """Лог-ось трат с делениями в тысячах рублей и пределами по краям данных с запасом (``axis_ticks``)."""
    ticks, lim = axis_ticks(candidates, lo, hi, pad)
    style.log_rub_axis(ax, axis, ticks)
    getattr(ax, f"set_{axis}lim")(*lim)


def plot_distribution(
    level: pd.Series,
    marks: Sequence[tuple[str, float]],
    p: Mapping[str, Any],
    year: int,
    highlight: pd.Series | None = None,
    highlight_label: str = HIGHLIGHT_LABEL,
):
    """F03: гистограмма уровня в лог-шкале (столбцы равной ширины в логарифме) и вертикальные отметки
    с прямыми подписями: подпись самой левой отметки — слева от линии, остальных — справа, на разной
    высоте.

    ``highlight`` — логическая маска МО (тот же индекс, что у ``level``): их часть столбцов лежит снизу
    тёмным серым с прямой подписью над самым высоким таким столбцом (второй горб распределения —
    внутригородские территории Москвы и Петербурга).
    """
    level = pd.Series(level, dtype="float64")
    ok = level.notna()
    v = level[ok].to_numpy()
    fig, ax = style.new_figure("full")
    edges = np.geomspace(v.min(), v.max(), int(p["hist_bins"]) + 1)
    hl = (
        np.zeros(len(v), dtype=bool)
        if highlight is None
        else pd.Series(highlight, index=level.index).fillna(False).astype(bool)[ok].to_numpy()
    )
    if hl.any():
        ax.hist(
            [v[hl], v[~hl]],
            bins=edges,
            stacked=True,
            color=[style.TEXT2, style.CONTEXT],
            edgecolor="white",
            linewidth=0.4,
        )
    else:
        ax.hist(v, bins=edges, color=style.CONTEXT, edgecolor="white", linewidth=0.4)
    _log_axis(ax, "x", p["rub_ticks"], v.min(), v.max(), float(p["axis_pad"]))
    ax.set_ylim(0, ax.get_ylim()[1] * HEADROOM)
    if hl.any():
        counts_hl, _ = np.histogram(v[hl], bins=edges)
        counts_all, _ = np.histogram(v, bins=edges)
        i = int(np.argmax(counts_hl))
        ax.annotate(
            highlight_label,
            (float(np.sqrt(edges[i] * edges[i + 1])), float(counts_all[i])),
            xytext=(0, HIGHLIGHT_LABEL_PAD_PT),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=style.POINT_LABEL_PT,
            color=style.TEXT2,
            linespacing=style.LINE_SPACING,
            path_effects=[patheffects.withStroke(linewidth=2.5, foreground="white")],
        )
    trans = ax.get_xaxis_transform()
    leftmost = min(value for _, value in marks)
    for (label, value), ls, y in zip(marks, MARK_STYLES, MARK_LABEL_Y, strict=False):
        ax.axvline(value, color=style.ACCENT, linewidth=LINE_WIDTH, linestyle=ls)
        left = value == leftmost
        ax.annotate(
            f"{label}: {style.fmt_krub(value)}",
            (value, y),
            xycoords=trans,
            xytext=(-4 if left else 4, 0),
            textcoords="offset points",
            ha="right" if left else "left",
            va="top",
            fontsize=style.POINT_LABEL_PT,
            color=style.TEXT,
            zorder=4,
            # Белая плашка: отметки стоят тесно, и соседняя линия не должна идти по тексту подписи.
            bbox={"boxstyle": f"square,pad={MARK_LABEL_PAD}", "facecolor": "white", "edgecolor": "none"},
        )
    ax.set_xlabel(f"Траты на жителя в месяц, среднее за {year} год")
    ax.set_ylabel("Число МО")
    return fig


def plot_wage(fit: pd.DataFrame, labels: pd.DataFrame, slope: float, intercept: float, p: Mapping[str, Any]):
    """F05: траты против зарплаты (лог-лог), серые точки МО, квадраты — внутригородские территории, робастная
    линия и подписи крупнейших остатков."""
    fig, ax = style.new_figure("full")
    inner = fit["is_inner_city"].astype(bool)
    ax.scatter(
        fit.loc[~inner, "wage"], fit.loc[~inner, "level"], s=POINT_SIZE, color=style.CONTEXT, linewidths=0
    )
    if inner.any():
        ax.scatter(
            fit.loc[inner, "wage"],
            fit.loc[inner, "level"],
            s=INNER_SIZE,
            marker="s",
            facecolors="none",
            edgecolors=style.TEXT2,
            linewidths=0.6,
        )
    pad = float(p["axis_pad"])
    wage_lo, wage_hi = float(fit["wage"].min()), float(fit["wage"].max())
    _log_axis(ax, "x", p["wage_ticks"], wage_lo, wage_hi, pad)
    _log_axis(ax, "y", p["rub_ticks"], fit["level"].min(), fit["level"].max(), pad)
    ax.grid(True, axis="both")
    ax.set_xlabel(f"Средняя зарплата Росстата, {CONTEXT_YEAR} год")
    ax.set_ylabel(f"Траты на жителя в месяц, {CONTEXT_YEAR}")
    if len(labels):
        ax.scatter(
            labels["wage"], labels["level"], s=LABELED_SIZE, color=style.ACCENT, zorder=3, linewidths=0
        )
        cols = p["label_columns"]
        parts = {
            key: labels.loc[labels["side"] == side] for side, key in (("выше", "above"), ("ниже", "below"))
        }
        for key, part in parts.items():
            column_labels(ax, part, cols[key], key, float(p["label_gap_pt"]))
        # Колонка «выше» должна стоять левее своих точек, «ниже» — правее: тогда выносные линии уходят
        # от колонки в сторону точек и не пересекают соседние подписи. Не хватает места — ось зарплаты
        # раздвигается.
        widths = {
            key: max(
                (
                    _text_width(ax, point_label(n, r))
                    for n, r in zip(d["name"], d["region_name"], strict=True)
                ),
                default=0.0,
            )
            for key, d in parts.items()
        }
        x0, x1 = column_limits(
            ax.get_xlim(),
            parts["above"]["wage"].to_numpy(),
            parts["below"]["wage"].to_numpy(),
            (float(cols["above"]["x"]), widths["above"]),
            (float(cols["below"]["x"]), widths["below"]),
            float(p["column_margin"]),
        )
        if not np.allclose((x0, x1), ax.get_xlim()):
            _log_axis(ax, "x", p["wage_ticks"], min(wage_lo, x0 * pad), max(wage_hi, x1 / pad), pad)
    if np.isfinite(slope):
        xs = np.geomspace(wage_lo, wage_hi, 50)  # линия — только в пределах данных, не в поле подписей
        ylim = ax.get_ylim()
        ax.plot(xs, np.exp(intercept) * xs**slope, color=style.ACCENT, linewidth=LINE_WIDTH)
        ax.set_ylim(*ylim)  # линия не раздвигает ось трат
    return fig


def _text_width(ax, text: str) -> float:
    """Ширина подписи точки (кегль ``POINT_LABEL_PT``) в долях ширины поля графика."""
    t = ax.text(0, 0, text, fontsize=style.POINT_LABEL_PT, linespacing=style.LINE_SPACING)
    renderer = ax.figure.canvas.get_renderer()
    width = t.get_window_extent(renderer=renderer).width / ax.get_window_extent(renderer=renderer).width
    t.remove()
    return float(width)


def column_limits(
    xlim: tuple[float, float],
    above_x: Sequence[float],
    below_x: Sequence[float],
    above: tuple[float, float],
    below: tuple[float, float],
    margin: float,
    iterations: int = 10,
) -> tuple[float, float]:
    """Пределы лог-оси X, при которых колонка подписей «выше» стоит левее своих точек, а «ниже» — правее.

    ``above`` = (левый край колонки, ширина), ``below`` = (правый край колонки, ширина) — в долях ширины
    поля; ``margin`` — зазор между колонкой и ближайшей точкой, тоже в долях. Доли не зависят от пределов
    оси, а положение точек — зависит: при нехватке места левый предел уменьшается (точки «выше» уходят
    вправо) или правый увеличивается. Две правки влияют друг на друга, поэтому они повторяются до
    ``iterations`` раз. Пределы, где места хватает, возвращаются как есть.
    """
    l0, l1 = np.log(xlim[0]), np.log(xlim[1])
    fa = above[0] + above[1] + margin  # точки «выше» — не левее этой доли
    fb = below[0] - below[1] - margin  # точки «ниже» — не правее этой доли
    xa = np.log(np.min(above_x)) if len(above_x) else None
    xb = np.log(np.max(below_x)) if len(below_x) else None
    for _ in range(iterations):
        changed = False
        if xa is not None and fa < 1 and (xa - l0) / (l1 - l0) < fa:
            l0 = (xa - fa * l1) / (1 - fa)
            changed = True
        if xb is not None and fb > 0 and (xb - l0) / (l1 - l0) > fb:
            l1 = l0 + (xb - l0) / fb
            changed = True
        if not changed:
            break
    return float(np.exp(l0)), float(np.exp(l1))


def column_order(points: pd.DataFrame, stack: str) -> pd.DataFrame:
    """Порядок подписей в колонке, чтобы выносные линии не перекрещивались: колонка «above» идёт сверху
    вниз — первой подписывается самая высокая точка; «below» — снизу вверх, первой — самая низкая.
    При равенстве — по ``territory_id``."""
    descending = stack == "above"
    return points.sort_values(["level", "territory_id"], ascending=[not descending, True], kind="mergesort")


def column_labels(ax, points: pd.DataFrame, anchor: Mapping[str, float], stack: str, gap_pt: float) -> list:
    """Подписи МО стопкой в углу поля графика с выносными линиями к точкам (F05).

    ``anchor`` — угол колонки в долях осей: у «above» — левый верхний (стопка растёт вниз), у «below» —
    правый нижний (стопка растёт вверх). Шаг — высота подписи в строках плюс ``gap_pt``; смещения —
    в пунктах, поэтому колонка не зависит от размера поля после ``style.finish``.
    """
    down = stack == "above"
    fig = ax.figure
    shift = 0.0
    out = []
    for name, region, x, y in column_order(points, stack)[
        ["name", "region_name", "wage", "level"]
    ].itertuples(index=False):
        text = point_label(name, region)
        height = (text.count("\n") + 1) * style.POINT_LABEL_PT * style.LINE_SPACING
        offset = ScaledTranslation(0, (-shift if down else shift) / 72, fig.dpi_scale_trans)
        out.append(
            ax.annotate(
                text,
                (x, y),
                xytext=(anchor["x"], anchor["y"]),
                textcoords=ax.transAxes + offset,
                ha="left" if down else "right",
                va="top" if down else "bottom",
                fontsize=style.POINT_LABEL_PT,
                color=style.TEXT,
                linespacing=style.LINE_SPACING,
                zorder=4,
                path_effects=[patheffects.withStroke(linewidth=2.5, foreground="white")],
                arrowprops={
                    "arrowstyle": "-",
                    "color": style.TEXT2,
                    "linewidth": LEADER_WIDTH,
                    "shrinkA": 2,
                    "shrinkB": 3,
                    "relpos": (1.0, 0.5) if down else (0.0, 0.5),
                },
            )
        )
        shift += height + gap_pt
    return out


# --- Раздел --------------------------------------------------------------------------------------


def _or_dash(value: Any) -> str:
    """Пустая ячейка Markdown — «—», а не пустота (и двойной пробел в строке таблицы)."""
    return str(value) if isinstance(value, str) and value else style.NA_TEXT


def _fmt_check(value: float, decimals: int = 2) -> str:
    return style.fmt_num(value, decimals)


def run_section(ctx: SectionContext) -> Finding:
    """E2: факты уровня и его связей, графики F03–F05, таблицы T03–T04."""
    cfg, data = ctx.cfg, ctx.data
    eda = cfg["eda"]
    p = params(cfg)
    checks = p["checks"]
    year, base, wage_year = int(eda["reference_year"]), BASE_YEAR, CONTEXT_YEAR
    names = data.territories[["territory_id", "name_short"]]
    mo = data.mo.merge(names, on="territory_id", how="left")

    # Числа
    summary = level_summary(mo, year=year, base_year=base, wage_year=wage_year)
    groups = place_groups(mo, float(eda["north_lat"]), year=year)
    drivers = level_drivers(mo, year=year, wage_year=wage_year)
    table3 = extremes(
        mo,
        eda.get("annotations"),
        int(eda["top_n"]),
        year=year,
        hint_z=float(p["hint_z"]),
        hint_max=int(p["hint_max"]),
        outlier_z=float(eda["robust_z"]),
    )
    rel = _col(mo, f"level_rel_{year}").set_axis(mo["territory_id"].to_numpy())
    weights = year_weights(mo, year).set_axis(mo["territory_id"].to_numpy())
    classes, cls = map_classes(rel, weights, int(p["map_classes"]))

    k_names = int(p["names_in_fact"])

    def names_of(group: str) -> str:
        part = table3.loc[table3["group"] == group].head(k_names)
        return names_with_regions(part["name"], part["region_name"])

    values: dict[str, Any] = {**summary, **groups, **drivers}
    lo, hi = partial_rho_ci(mo, ctx.rng, int(eda["bootstrap"]), year=year, wage_year=wage_year)
    values["partial_rho_access_within_lo"], values["partial_rho_access_within_hi"] = lo, hi
    values["partial_rho_threshold"] = float(eda["rejection"]["s2_partial_rho_min"])
    values["pop_share_top_class"] = float(classes["pop_share"].iloc[-1]) if len(classes) else float("nan")
    values["pop_share_bottom_class"] = float(classes["pop_share"].iloc[0]) if len(classes) else float("nan")
    values["top_names"] = names_of(GROUP_TOP)
    values["bottom_names"] = names_of(GROUP_BOTTOM)
    fit, slope, intercept = wage_fit(mo, year=wage_year)
    labels = residual_labels(fit, int(p["labels_per_side"]))
    for side, key in (("выше", "wage_above_names"), ("ниже", "wage_below_names")):
        part = labels.loc[labels["side"] == side]
        values[key] = names_with_regions(part["name"], part["region_name"])
    for key, value in values.items():
        ctx.fact(key, value, fact_kind(key), fact_note(key, year, wage_year))
    v = values  # короче в заголовках

    # F03 — распределение уровня
    med, mean, wmean = v[f"level_median_{year}"], v[f"level_mean_{year}"], v[f"level_wmean_{year}"]
    ratio = v[f"wmean_to_median_{year}"]
    inner = _flag(mo, "is_inner_city")
    fig = plot_distribution(
        mo[f"level_{year}"],
        [("медиана МО", med), ("среднее МО", mean), ("среднее по жителям", wmean)],
        p,
        year,
        highlight=inner,
    )
    title = ctx.headline(
        typo(
            f"В типичном МО житель тратит {style.fmt_krub(med)} в месяц, а в среднем по жителям — "
            f"{style.fmt_krub(wmean)}"
        ),
        ratio >= float(checks["wmean_ratio_min"]) and wmean > mean,
        f"F03: level_wmean_{year} / level_median_{year} = {_fmt_check(ratio)}, "
        f"нужно ≥ {_fmt_check(checks['wmean_ratio_min'])}; среднее по жителям {_fmt_check(wmean, 0)} "
        f"должно быть выше среднего МО {_fmt_check(mean, 0)} (многолюдные МО тратят больше)",
    )
    n_level = v[f"n_level_{year}"]
    ctx.save_figure(
        fig,
        fid="F03",
        slug="level_distribution",
        title=title,
        subtitle=typo(
            f"Средние безналичные траты жителя в месяц, {year}, номинал; МО с 12 месяцами года, "
            f"n{style.NBSP}={style.NBSP}{style.fmt_num(n_level)}; шкала логарифмическая"
        ),
        alt=typo(
            f"Гистограмма трат на жителя по МО за {year} год: медиана МО {style.fmt_krub(med)}, среднее по "
            f"жителям {style.fmt_krub(wmean)} — в {style.fmt_num(ratio, 1)} раза выше: многолюдные МО "
            "тратят больше и весят больше"
        ),
        data=mo.loc[mo[f"level_{year}"].notna(), ["territory_id", "name_short", "region_name", "mo_type"]]
        .assign(
            is_inner_city=inner,
            level=mo[f"level_{year}"],
            weight=year_weights(mo, year),
        )
        .reset_index(drop=True),
        check=(
            f"level_wmean_{year} ≥ {_fmt_check(checks['wmean_ratio_min'], 1)} · level_median_{year} "
            f"и level_wmean_{year} > level_mean_{year}"
        ),
    )

    # F04 — карта уровня к медиане страны
    geo = maps.load_geometry(data.geo_path)
    frame = maps.map_frame(geo, year)
    n_absent = len(maps.absent_regions(frame))
    note = (
        f"{n_absent} {plural(n_absent, ('регион', 'региона', 'регионов'))}:\nнет данных СберИндекса"
        if n_absent
        else None
    )
    fig, _ = maps.russia_map(
        rel.dropna(),
        geo,
        year,
        kind="quantile",
        k=int(p["map_classes"]),
        fmt=times_fmt,
        legend_title="Траты к медиане МО того же месяца",
        insets=maps.insets_from_config(cfg),
        na_label=na_label(int(eda["level_rel_min_months"])),
        absent_note=note,
    )
    eta2 = v["eta2_region_level"]
    hot = float(np.min([v["median_rel_inner"], v["median_rel_north"]]))  # NaN группы не пропускается
    ok_map = eta2 >= float(checks["eta2_min"]) and np.isfinite(hot) and hot > max(v["median_rel_rest"], 1.0)
    title = ctx.headline(
        typo(
            f"Траты выше всего в Москве, Петербурге и на Севере; регион объясняет {style.fmt_pct(eta2, 0)} "
            "разброса"
        ),
        ok_map,
        f"F04: eta2_region_level = {_fmt_check(eta2)} (нужно ≥ {_fmt_check(checks['eta2_min'])}); медианы "
        f"к стране: внутригородские {_fmt_check(v['median_rel_inner'])}, Север "
        f"{_fmt_check(v['median_rel_north'])}, прочие {_fmt_check(v['median_rel_rest'])} (нужно: первые две "
        "выше прочих и выше 1)",
    )
    map_data = pd.DataFrame(
        {
            "territory_id": rel.dropna().index.astype("int32"),
            "level_rel": rel.dropna().to_numpy(),
            "cls": cls.reindex(rel.dropna().index).to_numpy(),
        }
    )
    map_data["class_label"] = map_data["cls"].map(classes.set_index("cls")["label"])
    map_data["population"] = map_data["territory_id"].map(weights)
    ctx.save_figure(
        fig,
        fid="F04",
        slug="level_map",
        title=title,
        subtitle=typo(
            f"Траты {year} года к медиане МО того же месяца, среднее по месяцам; "
            f"{int(p['map_classes'])} квантильных классов; n{style.NBSP}={style.NBSP}"
            f"{style.fmt_num(v['n_rel'])}; η² региона {_fmt_check(eta2)}"
        ),
        alt=typo(
            "Карта МО по тратам к медиане страны: выше всего — внутригородские территории Москвы "
            f"и Петербурга (медиана {times_fmt(v['median_rel_inner'])}) и Север "
            f"({times_fmt(v['median_rel_north'])}); регион объясняет {style.fmt_pct(eta2, 0)} разброса "
            "логарифма уровня"
        ),
        data=map_data,
        check=(
            f"eta2_region_level ≥ {_fmt_check(checks['eta2_min'], 1)}; median_rel_inner и median_rel_north "
            "> max(median_rel_rest, 1)"
        ),
    )

    # F05 — траты и зарплата
    fig = plot_wage(fit, labels, slope, intercept, p)
    rho, rho_w = v["rho_level_wage"], v["rho_level_wage_within"]
    title = ctx.headline(
        typo(
            f"Траты почти повторяют зарплату: ρ{style.NBSP}={style.NBSP}{style.fmt_rho(rho)}, внутри "
            f"регионов — {style.fmt_rho(rho_w)}"
        ),
        rho >= float(checks["rho_wage_min"]),
        f"F05: rho_level_wage = {_fmt_check(rho)}, нужно ≥ {_fmt_check(checks['rho_wage_min'])}",
    )
    fit_data = fit.assign(labeled=fit["territory_id"].isin(labels["territory_id"]))
    ctx.save_figure(
        fig,
        fid="F05",
        slug="level_vs_wage",
        title=title,
        subtitle=typo(
            f"{wage_year} год; зарплата Росстата — по месту работы, без МСП; шкалы логарифмические; "
            f"n{style.NBSP}={style.NBSP}{style.fmt_num(v['n_level_wage'])}; квадраты — Москва и Петербург"
        ),
        alt=typo(
            f"Точечный график: чем выше зарплата в МО, тем выше траты жителей, ρ = {style.fmt_rho(rho)}; "
            f"внутри регионов связь держится (ρ = {style.fmt_rho(rho_w)}); подписаны МО дальше всего "
            "от робастной линии, кроме внутригородских территорий"
        ),
        data=fit_data,
        check=f"rho_level_wage ≥ {_fmt_check(checks['rho_wage_min'], 1)}",
        source=style.join_sources(style.SOURCE_SBER, style.SOURCE_ROSSTAT),
    )

    # T03 — выбросы по уровню
    per_side = int(p["md_per_side"])
    ctx.save_table(
        display_order(table3, per_side),
        tid="T03",
        slug="level_extremes",
        title=f"МО с самыми высокими и самыми низкими тратами {year} года к медиане страны",
        md_rows=2 * per_side,
        md_formats={
            "level": lambda x: style.fmt_rub(x),
            "level_rel": times_fmt,
            "rel_region": times_fmt,
            "z_type": lambda x: style.fmt_num(x, 1),
            "hint": _or_dash,
            "annotation": _or_dash,
        },
        md_labels={
            "group": "Край",
            "rank": "№",
            "territory_id": "id",
            "name": "МО",
            "region_name": "Регион",
            "mo_type": "Тип",
            "level": f"Траты {year}, ₽ в месяц",
            "level_rel": "К медиане страны",
            "rel_region": "К медиане региона",
            "z_type": "z в своём типе МО",
            "outlier": "Выброс",
            "hint": "Чем выделяется (признаки контекста)",
            "annotation": "Пояснение",
        },
    )

    # T04 — типичное МО и типичный житель
    table4 = weighting_table(data.panel_wide, mo, year=year)
    ctx.save_table(
        table4,
        tid="T04",
        slug="weighting",
        title=f"Типичное МО и типичный житель: траты по категориям, {year}",
        md_formats={
            "median_mo": lambda x: style.fmt_rub(x),
            "mean_mo": lambda x: style.fmt_rub(x),
            "wmean_residents": lambda x: style.fmt_rub(x),
            "ratio_wmean_median": lambda x: style.fmt_num(x, 2),
        },
        md_labels={
            "category": "Код",
            "label": "Категория",
            "n_mo": "МО",
            "n_weighted": "МО с населением",
            "median_mo": "Медиана МО",
            "mean_mo": "Среднее МО",
            "wmean_residents": "Среднее по жителям",
            "ratio_wmean_median": "Жители / медиана",
        },
    )

    verdict = c2_verdict(
        v["partial_rho_access_within"],
        v["partial_rho_access_within_lo"],
        v["partial_rho_access_within_hi"],
        v["partial_rho_access_within_no_inner"],
        v["partial_rho_threshold"],
    )
    split = access_split(v["rho_level_access"], v["rho_level_access_no_inner"], v["rho_level_access_rest"])
    summary = SUMMARY_MD.format(y=year, verdict=verdict, access_split=split)
    return ctx.finding(title=TITLE, summary_md=summary, caveats=CAVEATS)


# «Что видно» и «Что это значит для сюжета»: числа — только {{e2.ключ}}; {y} — год отчёта.
# Неразрывные пробелы (\u00a0) — после однобуквенных предлогов и союзов, в «рис. N» и вокруг «=».
SUMMARY_MD = """\
**Что видно.** В\u00a0типичном МО (медиана по МО) житель в\u00a0{y} году тратил по карте \
{{{{e2.level_median_{y}}}}} в\u00a0месяц, а\u00a0в\u00a0среднем по всем жителям — \
{{{{e2.level_wmean_{y}}}}}: многолюдные МО тратят больше (рис.\u00a03, T04). Второй горб распределения — \
внутригородские территории Москвы и\u00a0Петербурга: их медиана — {{{{e2.median_level_vgt}}}}, \
у\u00a0муниципальных районов — {{{{e2.median_level_mr}}}}, у\u00a0городских округов — \
{{{{e2.median_level_go}}}}. В\u00a0МО 90-го перцентиля траты выше, чем в\u00a0МО 10-го, \
в\u00a0{{{{e2.p90_p10_{y}}}}} раза. Выше всех — {{{{e2.top_names}}}}; ниже всех — \
{{{{e2.bottom_names}}}}; полный список — в\u00a0T03.

Регион объясняет большую часть разброса: η² региона для логарифма уровня — \
{{{{e2.eta2_region_level}}}}, без внутригородских территорий — \
{{{{e2.eta2_region_level_no_inner}}}} (η² — доля разброса между регионами в\u00a0общем разбросе), \
рис.\u00a04. Карта площадью скрывает людей: в\u00a0верхнем классе карты живёт \
{{{{e2.pop_share_top_class}}}} жителей её МО, в\u00a0нижнем — {{{{e2.pop_share_bottom_class}}}}. \
Траты почти повторяют зарплату Росстата: ранговая корреляция Спирмена \
ρ\u00a0=\u00a0{{{{e2.rho_level_wage}}}} по {{{{e2.n_level_wage}}}} МО, внутри регионов — \
{{{{e2.rho_level_wage_within}}}} (рис.\u00a05). Медиана отношения трат жителя к\u00a0средней зарплате \
работника — {{{{e2.spend_to_wage_median}}}} (знаменатели разные: все жители против работников \
организаций без МСП). Дальше всего выше линии — {{{{e2.wage_above_names}}}}, ниже — \
{{{{e2.wage_below_names}}}}. Гипотеза: там зарплата по месту работы расходится с\u00a0доходом \
жителей (жители пригорода работают в\u00a0областном центре, а\u00a0у\u00a0крупного работодателя работают \
приезжие); проверка — в\u00a0разделе «Экономика места».

**Что это значит для сюжета.** Сырой уровень трат — почти карта регионов и\u00a0зарплат, поэтому \
сюжет С2 «ядра и\u00a0периферия» по уровню её повторит. С\u00a0доступностью рынков \
ρ\u00a0=\u00a0{{{{e2.rho_level_access}}}}, без внутригородских территорий — \
{{{{e2.rho_level_access_no_inner}}}}, без них и\u00a0без Севера — \
{{{{e2.rho_level_access_rest}}}}{access_split}. Медиана индекса доступности \
на Севере — {{{{e2.median_access_north}}}}, у\u00a0прочих МО — {{{{e2.median_access_rest}}}}, \
а\u00a0траты к\u00a0медиане страны — {{{{e2.median_rel_north}}}} против {{{{e2.median_rel_rest}}}}. \
Внутри регионов ρ\u00a0=\u00a0{{{{e2.rho_level_access_within}}}}. Сверх зарплаты внутри регионов \
доступность даёт частный ρ\u00a0=\u00a0{{{{e2.partial_rho_access_within}}}} (интервал бутстрепа — \
от\u00a0{{{{e2.partial_rho_access_within_lo}}}} до\u00a0{{{{e2.partial_rho_access_within_hi}}}}, \
{{{{e2.n_partial_access}}}} МО), без внутригородских территорий — \
{{{{e2.partial_rho_access_within_no_inner}}}}. Порог критерия отказа С2 — \
{{{{e2.partial_rho_threshold}}}}: {verdict}. С\u00a0расстоянием до столицы региона внутри регионов \
ρ\u00a0=\u00a0{{{{e2.rho_level_dist_capital_within}}}}. Для этапа 2 уровень трат лучше брать \
относительно региона или зарплаты, иначе типы МО повторят карту регионов. Корреляции здесь описывают \
связь, а\u00a0не\u00a0причину.
"""

CAVEATS: list[str] = [
    "Траты — оценка СберИндекса средних безналичных потребительских трат жителей МО в номинальных рублях; "
    "траты приезжих в МО не видны, а траты жителей вне своего МО (поездки, онлайн) по смыслу входят, но как "
    "их привязывают к МО, модель не раскрывает.",
    "Наличные расходы не видны: где доля наличных выше, уровень занижен (гипотеза, в данных не проверить).",
    "Зарплата Росстата — по месту работы и без малого бизнеса; у внутригородских территорий Москвы "
    "и Петербурга она описывает работодателей района, а не жителей.",
    "Среднее по жителям взвешено среднегодовым населением года; МО без населения в него не входят.",
    "Выводы — о регионах, где есть данные СберИндекса, а не о всей России.",
]
