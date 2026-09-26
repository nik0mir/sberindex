"""Раздел разведки E1 «Что есть в данных»: кто в панели, кого нет, случайны ли пропуски, хватит ли контекста.

Выходы (spec_final, Б.4, E1; каталог — В.5):

- F01 ``coverage_map`` — карта статуса ряда трат по МО в границах опорного года; МО без данных СберИндекса
  заштрихованы, выпавшие целиком регионы названы одной выноской, Москва и Петербург — во врезках;
- F02 ``coverage_funnel`` — воронка «МО справочника → с тратами → с полным рядом» и покрытие показателей
  контекста Росстата и ФНС по годам;
- T00 ``controls`` — контрольные числа этапа panel; T01 ``missingness`` — чем неполные ряды отличаются от
  полных (SMD, модель пропуска, регионы); T02 ``context_coverage`` — показатель × год: сколько МО, каким шагом
  соединения по ОКТМО найдено значение, какие флаги, сколько значений в годовой таблице и почему значения нет
  (в отчёт — строки года признаков контекста разведки, остальные годы — в CSV).

Вычисления — чистые функции этого модуля; ``run_section`` только собирает факты, графики и таблицы.
Раздел читает только ``ctx.data`` (полигоны — через ``maps.load_geometry(ctx.data.geo_path)``).
Параметры — секция ``eda.coverage`` конфига; пока её нет, действуют ``COVERAGE_DEFAULTS``.
"""

from __future__ import annotations

import logging
import re
import textwrap
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import geopandas as gpd
import matplotlib as mpl
import numpy as np
import pandas as pd
import shapely
from matplotlib.figure import Figure
from scipy import stats as sps

from munnet import maps, style
from munnet.config import Config
from munnet.contracts import MATCH_METHODS, MO_TYPES, N_MONTHS, YEARS
from munnet.eda import stats
from munnet.eda.base import Finding, SectionContext
from munnet.eda.data import CONTEXT_YEAR

log = logging.getLogger(__name__)

SECTION_ID = "e1"
TITLE = "Что есть в данных"

# Параметры раздела — секция ``eda.coverage`` конфига. Значения ниже совпадают с предложенными для конфига и
# действуют, только пока секции нет (раздел не падает на конфиге без неё).
COVERAGE_DEFAULTS: dict[str, Any] = {
    "full_share_min": 0.9,  # проверка заголовка F02: полный ряд не меньше чем у 90% МО панели
    "month_thresholds": [12, 18, 24],  # пороги длины ряда для узла сети: сколько МО и жителей остаётся (Б.6)
    "text_regions": 3,  # сколько регионов называть в тексте (больше всего неполных рядов, пропущенных МО)
    "missing_alpha": 0.01,  # уровень теста Манна — Уитни, ниже которого пропуски называются неслучайными
    "gap_indicator": "wage",  # показатель Росстата, по которому ищутся регионы без строк за опорный год
}

# --- Словари подписей -----------------------------------------------------------------------------

FULL, ONE_YEAR, PARTIAL, NO_DATA = "full", "one_year", "partial", "no_data"
STATUS_GROUP = {"full": FULL, "only_2023": ONE_YEAR, "only_2024": ONE_YEAR, "partial": PARTIAL}
STATUS_LABELS = {FULL: "полный ряд", ONE_YEAR: "ровно один год", PARTIAL: "неполный ряд"}
# Цвета статусов на карте — точки шкал стиля (cividis и PuOr различимы при протанопии и дейтеранопии):
# полный ряд — светлый фон, неполные — тёмные, чтобы глаз находил пропуски, а не норму; «один год» — не серый,
# чтобы его не путали с серой штриховкой «нет данных СберИндекса».
STATUS_COLOR_SPEC = {
    FULL: (style.SEQ_CMAP, 0.85),
    ONE_YEAR: (style.DIV_CMAP, 0.8),
    PARTIAL: (style.SEQ_CMAP, 0.0),
}

MO_TYPE_LABELS = {
    "mr": "муниципальный район",
    "mo": "муниципальный округ",
    "go": "городской округ",
    "vgt": "внутригородская территория",
}
MONTHS_NOM = (
    "январь",
    "февраль",
    "март",
    "апрель",
    "май",
    "июнь",
    "июль",
    "август",
    "сентябрь",
    "октябрь",
    "ноябрь",
    "декабрь",
)
ROSSTAT, FNS = "Росстат", "ФНС"
SOURCE_GENITIVE = {ROSSTAT: "Росстата", FNS: "ФНС"}  # «у Росстата нет строк»
YEAR_SPAN = f"{YEARS[0]}–{YEARS[-1]}"  # «2023–2024»
MAIN_DIM = "TOTAL"  # разрез-итог показателя в context_long


@dataclass(frozen=True)
class Indicator:
    """Показатель контекста: подпись, источник и показывать ли его на F02."""

    label: str
    source: str
    on_chart: bool = True


# Порядок — порядок строк T02; показатели не из словаря идут после, по алфавиту.
INDICATORS: dict[str, Indicator] = {
    "population": Indicator("Население на 1 января", ROSSTAT),
    "age": Indicator("Возрастные группы", ROSSTAT),
    "population_urban": Indicator("Городское население", ROSSTAT, on_chart=False),
    "population_rural": Indicator("Сельское население", ROSSTAT, on_chart=False),
    "wage": Indicator("Средняя зарплата", ROSSTAT),
    "employees": Indicator("Работники", ROSSTAT),
    "payroll": Indicator("Фонд оплаты труда", ROSSTAT, on_chart=False),
    "retail": Indicator("Оборот розницы", ROSSTAT),
    "catering_turnover": Indicator("Оборот общепита", ROSSTAT),
    "shipments": Indicator("Отгрузка товаров", ROSSTAT),
    "orgs": Indicator("Организации", ROSSTAT),
    "ip": Indicator("Индивидуальные предприниматели", ROSSTAT),
    "nights": Indicator("Ночёвки в средствах размещения", ROSSTAT),
    "beds": Indicator("Места в средствах размещения", ROSSTAT),
    "invest": Indicator("Инвестиции", ROSSTAT),
    "ndfl_income": Indicator("Доход по 5-НДФЛ", FNS),
    "ndfl_recipients": Indicator("Получатели дохода по 5-НДФЛ", FNS),
}
# Покрытие этих показателей — обязательные факты ``cov_<показатель>_<год>`` (Б.4), даже если их ещё нет.
KEY_INDICATORS = ("population", "wage", "ndfl_income")

# Флаги значений, которые этап panel оставляет в context_long, но не берёт в context_annual (значения таких МО
# попадают и в unmatched.csv с причиной, равной флагу): в T02 это не «потеря», а «исключено».
UNION_MERGED, RECIPIENTS_JUMP = "union_merged", "recipients_jump"
FLAG_LABELS = {
    "fallback_period": "запасной период",
    "zero_replaced": "ноль вместо значения",
    "dup_conflict": "конфликт дублей",
    "anomaly": "аномалия по Росстату",
    "age_sum_mismatch": "сумма возрастных групп расходится с «Всего»",
    UNION_MERGED: "значение уже за объединённое МО, исключено",
    RECIPIENTS_JUMP: "сбой единиц получателей в источнике, исключено",
}
REGION_NO_ROWS = "region_no_rows"
REASON_LABELS = {
    REGION_NO_ROWS: "в регионе нет годовых строк",
    "union_banned": "мост запрещён объединением",
    "multi_row": "несколько разных строк",
    "no_code": "нет кода",
    "not_in_slice": "МО нет в срезе справочника",
    "nonpositive": "население не больше нуля, переносить нечего",
    "unknown": "причина не указана",
}
METHOD_LABELS = {
    "direct": "Прямой код",
    "version": "Код другой версии",
    "stable": "Мост oktmo_stable",
    "carried": "Перенос прошлого года",
}

# Подписи контрольных чисел этапа panel (А.11); ключи «<показатель>_<год>» подписываются по INDICATORS.
CONTROL_LABELS = {
    "consumption_rows": "Строк в тратах СберИндекса",
    "territories": "МО в панели",
    "months": "Месяцев в панели",
    "mo_months": "Пар «МО × месяц»",
    "panel_long_rows": "Строк длинной панели",
    "dup_keys": "Дублей ключа",
    "other_nonpositive": "Случаев «Прочее» ≤ 0",
    "series_full": "Полных рядов",
    "series_only_2023": "Рядов ровно за 2023 год",
    "series_only_2024": "Рядов ровно за 2024 год",
    "series_partial": "Прочих неполных рядов",
    "exactly_12_months": "Рядов ровно из 12 месяцев",
    "internal_gap": "Рядов с разрывом внутри",
    "dict_versions": "Версий МО в справочнике",
    "dict_territories": "МО в справочнике",
    "in_slice_2023": "МО панели в срезе справочника 2023 года",
    "in_slice_2024": "МО панели в срезе справочника 2024 года",
    "inner_city": "Внутригородских территорий",
    "regions": "Регионов",
    "polygons_matched": "МО панели с полигоном",
    "market_access": "МО панели с индексом доступности рынков",
    "pop_avg_both_2024": "Среднегодовое население 2024 года по двум датам",
    "pop_carried_2024": "Население 2024 года: перенос прошлого года",
    "pop_fallback_period_2023": "Население 2023 года: запасной период",
    "pop_sum_2024_mln": "Население МО панели на 1 января 2024 года, млн",
    "bdmo_stable_pairs": "Пар «ОКТМО — стабильный код» в БД ПМО",
    "ndfl_stable_pairs": "Пар «ОКТМО — стабильный код» в 5-НДФЛ",
    "stable_union_values": "Значений моста у МО с объединением в истории ОКТМО (ручная проверка запрета)",
}
_CONTROL_ALIASES = {"pop_jan1": "population"}
KIND_ORDER = ("hard", "soft", "info")  # info — справочные числа без ожидания (методы соединения, пары моста)
KIND_LABELS = {"hard": "жёсткое", "soft": "мягкое", "info": "справочное"}
CHECKED_KINDS = ("hard", "soft")  # в Markdown отчёта — только числа с проверкой, справочные — в CSV
STATUS_TEXT = {
    ("hard", True): "сошлось",
    ("hard", False): "ошибка",
    ("soft", True): "в допуске",
    ("soft", False): "предупреждение",
    ("info", True): "справочно",
    ("info", False): "справочно",
}

SERIES_NOTES = {
    "n_mo": "МО панели (territories)",
    "n_full": "ряд за все 24 месяца",
    "n_only_2023": "ровно 12 месяцев 2023 года и ни одного месяца 2024-го",
    "n_only_2024": "ровно 12 месяцев 2024 года и ни одного месяца 2023-го",
    "n_partial": "прочие неполные ряды",
    "n_incomplete": "все ряды короче 24 месяцев",
    "n_exactly_12": "ряды ровно из 12 месяцев (любых)",
    "n_internal_gap": "ряды с пропуском внутри",
    "n_only_year_2023": "есть месяцы только 2023 года (любое их число)",
    "n_only_year_2024": "есть месяцы только 2024 года (любое их число)",
    "n_both_years_incomplete": "неполные ряды с месяцами обоих лет",
    "n_regions": "регионы с тратами (region_code справочника)",
    "n_inner_city": "внутригородские территории Москвы и Петербурга",
}

# Регионы с курортами Черноморского побережья: если их нет, курортный тип по тратам неполон.
RESORT_REGIONS = ("Краснодарский край", "Республика Крым", "Севастополь")

CAVEATS = [
    "Траты — оценка средних безналичных потребительских расходов жителей МО по моделям СберИндекса "
    "в номинальных рублях. Траты приезжих в данных не видны; траты жителей вне своего МО по описанию набора "
    "входят, но модель привязки не раскрыта.",
    "Показатели Росстата — без малого бизнеса; занятость и доход по 5-НДФЛ считаются по месту работодателя, "
    "поэтому у внутригородских территорий Москвы и Петербурга на жителя они не считаются.",
    "Объединения МО 2024 года: ряды предшественников и преемников не склеены, на карте 2024 года показаны "
    "преемники.",
    "«Ровно один календарный год» — все 12 месяцев 2023 или 2024 года и ни одного месяца другого года; "
    "прочие ряды с пропусками — «неполные».",
]

_DATE = re.compile(r"(\d{4})-(\d{2})")
_KEY_UNSAFE = re.compile(r"[^a-z0-9_]+")
_CONTROL_YEAR = re.compile(r"(.+)_(\d{4})")
_CONTROL_METHOD = re.compile(r"method_(.+)_(\d{4})_([a-z]+)")
STABLE_UNION_KEY = "stable_union_values"  # значения моста у МО с «Объединением» в истории ОКТМО (А.6)
OBLAST = " область"

# Вёрстка графиков
REGION_NOTE_WIDTH = 36  # знаков в строке выноски о выпавших регионах на карте
BAR_LABEL_PT = 7
BAR_HEIGHT = 0.8  # доля шага строки под столбцами (у пары лет — на двоих)
PAIR_ROW_RATIO = 1.4  # высота строки с парой лет относительно строки воронки на F02
XLIM_PAD = 1.1  # запас оси X справа под подписи чисел у концов столбцов
LABEL_PAD_PT = 3  # отступ подписей у столбцов, pt


# --- Помощники ------------------------------------------------------------------------------------


def coverage_params(cfg: Config) -> dict[str, Any]:
    """Параметры раздела: ``eda.coverage`` конфига поверх ``COVERAGE_DEFAULTS``."""
    return {**COVERAGE_DEFAULTS, **(cfg["eda"].get("coverage") or {})}


def plural(n: int, one: str, few: str, many: str) -> str:
    """Форма слова при числе: 1 регион, 2 региона, 5 регионов, 11 регионов, 21 регион."""
    n = abs(int(n))
    if n % 10 == 1 and n % 100 != 11:
        return one
    if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
        return few
    return many


def genitive_plural(n: int, one: str, many: str) -> str:
    """Родительный падеж при числе: «МО 1 региона», «МО 21 региона», «МО 77 регионов»."""
    n = abs(int(n))
    return one if n % 10 == 1 and n % 100 != 11 else many


def months_word(n: int) -> str:
    """«24 месяца», «12 месяцев», «1 месяц» (неразрывный пробел)."""
    return f"{n}{style.NBSP}{plural(n, 'месяц', 'месяца', 'месяцев')}"


def month_name(date: str) -> str:
    """«2024-01» → «январь 2024»."""
    m = _DATE.fullmatch(str(date))
    if m is None:
        raise ValueError(f"месяц {date!r} не вида ГГГГ-ММ")
    return f"{MONTHS_NOM[int(m.group(2)) - 1]} {m.group(1)}"


def join_regions(names: Iterable[str]) -> str:
    """Регионы для текста: сначала области одним перечнем, затем остальные, всё по алфавиту.

    «Курская область», «Краснодарский край», «Брянская область» → «Брянская и Курская области,
    Краснодарский край».
    """
    names = sorted({str(n) for n in names})
    oblasts = [n.removesuffix(OBLAST) for n in names if n.endswith(OBLAST)]
    others = [n for n in names if not n.endswith(OBLAST)]
    parts = []
    if len(oblasts) == 1:
        parts.append(oblasts[0] + OBLAST)
    elif oblasts:
        parts.append(f"{', '.join(oblasts[:-1])} и {oblasts[-1]} области")
    return ", ".join(parts + others)


def fact_key(text: str) -> str:
    """Часть ключа факта из кода показателя: латиница в нижнем регистре, цифры и «_»."""
    return _KEY_UNSAFE.sub("_", str(text).lower()).strip("_")


def _and_list(items: Iterable[str]) -> str:
    """«a», «a и b», «a, b и c»."""
    items = list(items)
    return items[0] if len(items) == 1 else f"{', '.join(items[:-1])} и {items[-1]}"


def counts_text(names: Sequence[str], counts: Sequence[int], k: int | None = None) -> str:
    """«Республика Бурятия — 21, Республика Тыва — 5» (первые ``k``, None — все)."""
    pairs = list(zip(names, counts, strict=True))[:k]
    return ", ".join(f"{n} — {style.fmt_num(c)}" for n, c in pairs)


def _share(part: float, total: float) -> float:
    return float(part / total) if pd.notna(total) and total > 0 else np.nan


def _fmt_auto(v: Any) -> str:
    """Число таблицы: у целых и больших — без дробной части, иначе один знак."""
    if v is None or (not isinstance(v, str) and pd.isna(v)):
        return style.NA_TEXT
    v = float(v)
    return style.fmt_num(v, 0 if abs(v) >= 100 else style.auto_decimals([v], max_decimals=1))


def _fmt_exact(v: Any) -> str:
    """Контрольное число без потери точности: 2190 → «2190», 120.48 → «120,48» (не больше двух знаков)."""
    if v is None or (not isinstance(v, str) and pd.isna(v)):
        return style.NA_TEXT
    return style.fmt_num(float(v), style.auto_decimals([float(v)], max_decimals=2))


def _fmt_smd(v: Any) -> str:
    return style.fmt_num(v, 2)


def _fmt_share(v: Any) -> str:
    return style.fmt_pct(v, 1)


def _fmt_deviation(v: Any) -> str:
    return style.fmt_pct(v, 1, sign=True)


# --- Ряды трат ------------------------------------------------------------------------------------


def series_summary(territories: pd.DataFrame) -> dict[str, int]:
    """Счётчики рядов трат по справочнику территорий (статусы — А.4).

    ``n_incomplete`` — всё, кроме ``full``; ``n_only_year_<год>`` — месяцы только одного года (любое их
    число); ``n_both_years_incomplete`` — ряд захватывает оба года, но не полный.
    """
    status = territories["series_status"].astype(str)
    n23, n24 = territories["n_2023"].astype(int), territories["n_2024"].astype(int)
    full = status == "full"
    return {
        "n_mo": int(len(territories)),
        "n_full": int(full.sum()),
        "n_only_2023": int((status == "only_2023").sum()),
        "n_only_2024": int((status == "only_2024").sum()),
        "n_partial": int((status == "partial").sum()),
        "n_incomplete": int((~full).sum()),
        "n_exactly_12": int((territories["n_months"].astype(int) == 12).sum()),
        "n_internal_gap": int(territories["has_internal_gap"].astype(bool).sum()),
        "n_only_year_2023": int(((n23 > 0) & (n24 == 0)).sum()),
        "n_only_year_2024": int(((n23 == 0) & (n24 > 0)).sum()),
        "n_both_years_incomplete": int(((n23 > 0) & (n24 > 0) & ~full).sum()),
        "n_regions": int(territories["region_code"].nunique()),
        "n_inner_city": int(territories["is_inner_city"].astype(bool).sum()),
    }


def status_groups(territories: pd.DataFrame) -> pd.Series:
    """Группа статуса ряда для карты по ``territory_id``: ``full``, ``one_year`` (ровно календарный год),
    ``partial``. Неизвестный статус — ``ValueError``."""
    groups = territories["series_status"].astype(str).map(STATUS_GROUP)
    if groups.isna().any():
        bad = sorted(territories.loc[groups.isna(), "series_status"].astype(str).unique())
        raise ValueError(f"неизвестные статусы ряда {bad}")
    return pd.Series(groups.to_numpy(), index=territories["territory_id"].to_numpy(), name="status")


def months_coverage(panel_wide: pd.DataFrame) -> pd.DataFrame:
    """Сколько МО с тратами в каждом месяце: ``date``, ``n_mo``, по возрастанию даты."""
    out = panel_wide.groupby("date")["territory_id"].nunique().rename("n_mo").reset_index()
    return out.sort_values("date").reset_index(drop=True)


def month_ladder(mo: pd.DataFrame, thresholds: Sequence[int]) -> pd.DataFrame:
    """Сколько МО и какая доля населения выборки останется при пороге «ряд не короче k месяцев».

    Население — ``weight`` (среднегодовое 2023 года, иначе 2024-го); доля — от населения всех МО с весом.
    """
    months = mo["n_months"].astype(int)
    weight = mo["weight"].astype("float64")
    total = weight.sum(min_count=1)
    rows = []
    for k in sorted({int(t) for t in thresholds}):
        keep = months >= k
        rows.append(
            {
                "min_months": k,
                "n_mo": int(keep.sum()),
                "share_mo": float(keep.mean()) if len(mo) else np.nan,
                "pop_share": _share(weight[keep].sum(), total),
            }
        )
    return pd.DataFrame(rows, columns=["min_months", "n_mo", "share_mo", "pop_share"])


def inner_city(mo: pd.DataFrame) -> dict[str, Any]:
    """Внутригородские территории: число, доля узлов, доля населения выборки, число по регионам."""
    inner = mo["is_inner_city"].astype(bool)
    weight = mo["weight"].astype("float64")
    return {
        "n": int(inner.sum()),
        "node_share": float(inner.mean()) if len(mo) else np.nan,
        "pop_share": _share(weight[inner].sum(), weight.sum(min_count=1)),
        "by_region": mo.loc[inner, "region_name"].astype(str).value_counts(),
    }


# --- Кого нет: срез справочника, выпавшие регионы -------------------------------------------------


def region_names(geo: gpd.GeoDataFrame, territories: pd.DataFrame | None = None) -> pd.Series:
    """Название региона по ``region_code``: колонка ``region_name`` полигонов (если этап panel её пишет),
    иначе справочник территорий (только регионы панели), иначе «регион <код>»."""
    codes = sorted(geo["region_code"].astype(int).unique())
    names = pd.Series(pd.NA, index=codes, dtype="object")
    sources = []
    if "region_name" in geo.columns:
        sources.append(geo[["region_code", "region_name"]])
    if territories is not None:
        sources.append(territories[["region_code", "region_name"]])
    for src in sources:
        known = src.dropna(subset=["region_name"]).drop_duplicates("region_code")
        names = names.fillna(known.set_index(known["region_code"].astype(int))["region_name"].astype(str))
    return names.fillna(pd.Series({c: f"регион {c}" for c in codes}))


def absent_regions(frame: gpd.GeoDataFrame, names: pd.Series) -> pd.DataFrame:
    """Регионы кадра карты без единого МО панели: код, название, число МО справочника, точка (WGS84).

    Точка — ``representative_point()`` объединения полигонов региона (всегда внутри региона); сортировка —
    по названию.
    """
    codes = maps.absent_regions(frame)
    columns = ["region_code", "region_name", "n_mo", "lat", "lon"]
    if not codes:
        return pd.DataFrame(columns=columns)
    points, sizes = [], []
    for code in codes:
        part = frame.loc[frame["region_code"].astype(int) == code]
        points.append(shapely.union_all(shapely.make_valid(part.geometry.values)).representative_point())
        sizes.append(len(part))
    wgs = gpd.GeoSeries(points, crs=frame.crs).to_crs(4326)
    out = pd.DataFrame(
        {
            "region_code": codes,
            "region_name": [str(names.get(c, f"регион {c}")) for c in codes],
            "n_mo": sizes,
            "lat": wgs.y.to_numpy(),
            "lon": wgs.x.to_numpy(),
        }
    )
    return out.sort_values(["region_name", "region_code"]).reset_index(drop=True)[columns]


def missing_outside(frame: gpd.GeoDataFrame, names: pd.Series) -> pd.DataFrame:
    """МО справочника без данных СберИндекса в регионах, где другие МО в панели есть: число по регионам,
    по убыванию."""
    in_panel = frame["in_panel"].astype(bool)
    has_panel = in_panel.groupby(frame["region_code"]).transform("any")
    counts = frame.loc[~in_panel & has_panel, "region_code"].astype(int).value_counts()
    out = pd.DataFrame({"region_code": counts.index.astype(int), "n_missing": counts.to_numpy(dtype=int)})
    out["region_name"] = out["region_code"].map(names).astype(str)
    out = out.sort_values(["n_missing", "region_name"], ascending=[False, True])
    return out[["region_code", "region_name", "n_missing"]].reset_index(drop=True)


def south_west(absent: pd.DataFrame, lat: pd.Series, lon: pd.Series) -> bool:
    """Все выпавшие регионы южнее и западнее медианной точки МО панели — проверка слов «на юго-западе»."""
    if absent.empty:
        return False
    ref_lat, ref_lon = float(np.nanmedian(lat)), float(np.nanmedian(lon))
    return bool(((absent["lat"] < ref_lat) & (absent["lon"] < ref_lon)).all())


def funnel(frame: gpd.GeoDataFrame, territories: pd.DataFrame, year: int) -> pd.DataFrame:
    """Воронка среза справочника ``year``: все МО среза → с тратами СберИндекса → с полным рядом."""
    ter = territories.loc[territories["territory_id"].astype(int).isin(frame["territory_id"].astype(int))]
    steps = [
        ("slice", f"МО в справочнике на {year} год", len(frame)),
        ("with_spend", "из них с тратами СберИндекса", len(ter)),
        (
            "full",
            f"из них с полным рядом, {months_word(N_MONTHS)}",
            int((ter["series_status"].astype(str) == "full").sum()),
        ),
    ]
    return pd.DataFrame(steps, columns=["step", "label", "n_mo"])


# --- Случайны ли пропуски -------------------------------------------------------------------------


# Население для сравнения неполных рядов с полными: среднегодовое 2023 года без подстановки 2024-го (у МО,
# появившихся в 2024 году, его нет — они в сравнение не входят). Доли населения выборки — по ``weight`` (Б.2).
POP_COLUMN = "pop_2023"


@dataclass(frozen=True)
class Feature:
    """Признак модели пропуска: колонка ``mo`` и подпись; ``log`` — в модель и SMD идёт логарифм,
    в таблицу — медиана в исходных единицах; ``percent`` — в таблицу × 100."""

    key: str
    label: str
    column: str
    log: bool = False
    percent: bool = False


FEATURES: tuple[Feature, ...] = (
    Feature("ln_pop", "Среднегодовое население 2023 года, чел., медиана", POP_COLUMN, log=True),
    Feature("ln_wage", "Средняя зарплата 2023 года, ₽, медиана", "wage_2023", log=True),
    Feature("urban", "Доля горожан 2023 года, %, медиана", "urban_share_2023", percent=True),
    Feature("lat", "Широта точки МО, °, медиана", "point_lat"),
    Feature("access", "Индекс доступности рынков (0–1000), медиана", "market_access"),
)


def missingness_features(mo: pd.DataFrame) -> pd.DataFrame:
    """Признаки модели пропуска: ln населения и зарплаты, доля горожан, широта, доступность рынков и
    индикаторы типа МО. Неположительные население и зарплата — пропуск (логарифм не определён)."""
    X = pd.DataFrame(index=mo.index)
    for f in FEATURES:
        v = mo[f.column].astype("float64")
        X[f.key] = np.log(v.where(v > 0)) if f.log else v
    mo_type = mo["mo_type"].astype(str)
    for t in MO_TYPES:
        X[f"type_{t}"] = (mo_type == t).astype("float64")
    return X


def missingness(mo: pd.DataFrame, seed: int, folds: int = 5) -> tuple[pd.DataFrame, float, float]:
    """Чем неполные ряды (``series_status != full``) отличаются от полных: (таблица, AUC, p).

    Таблица — признак, медиана (у типа МО — доля, %) у полных и неполных, SMD неполных против полных
    (у населения и зарплаты — на логарифме), число МО со значением. AUC — логистическая модель пропуска на
    всех признаках (медианное заполнение, стандартизация, ``folds`` стратифицированных фолдов,
    ``random_state = seed``), 0,5 — пропуски не предсказываются. p — двусторонний тест Манна — Уитни
    по среднегодовому населению 2023 года (``POP_COLUMN``).
    """
    incomplete = mo["series_status"].astype(str) != "full"
    X = missingness_features(mo)
    rows = []
    for f in FEATURES:
        shown = mo[f.column].astype("float64") * (100.0 if f.percent else 1.0)
        rows.append((f.key, f.label, shown[~incomplete].median(), shown[incomplete].median(), X[f.key]))
    for t in MO_TYPES:
        dummy = X[f"type_{t}"]
        label = f"Тип «{MO_TYPE_LABELS.get(t, t)}», % МО"
        rows.append(
            (f"type_{t}", label, 100.0 * dummy[~incomplete].mean(), 100.0 * dummy[incomplete].mean(), dummy)
        )
    table = pd.DataFrame(
        [
            {
                "feature": key,
                "label": label,
                "full": full,
                "incomplete": inc,
                "smd": stats.smd(x[incomplete], x[~incomplete]),
                "n_full": int(x[~incomplete].notna().sum()),
                "n_incomplete": int(x[incomplete].notna().sum()),
            }
            for key, label, full, inc, x in rows
        ]
    )
    auc = stats.missingness_auc(X, incomplete, seed=seed, folds=folds)
    pop = mo[POP_COLUMN].astype("float64")
    a, b = pop[incomplete].dropna(), pop[~incomplete].dropna()
    p = float(sps.mannwhitneyu(a, b, alternative="two-sided").pvalue) if len(a) and len(b) else np.nan
    return table, float(auc), p


def incomplete_by_region(mo: pd.DataFrame) -> pd.DataFrame:
    """Неполные ряды по регионам: всего МО, полных, неполных, доля неполных; только регионы с неполными,
    по убыванию числа неполных (при равенстве — доли, затем названия)."""
    g = pd.DataFrame(
        {"region_name": mo["region_name"].astype(str), "inc": mo["series_status"].astype(str) != "full"}
    )
    agg = g.groupby("region_name")["inc"].agg(n_mo="size", n_incomplete="sum").reset_index()
    agg["n_incomplete"] = agg["n_incomplete"].astype(int)
    agg["n_full"] = agg["n_mo"] - agg["n_incomplete"]
    agg["share_incomplete"] = agg["n_incomplete"] / agg["n_mo"]
    agg = agg.loc[agg["n_incomplete"] > 0]
    agg = agg.sort_values(["n_incomplete", "share_incomplete", "region_name"], ascending=[False, False, True])
    return agg[["region_name", "n_mo", "n_full", "n_incomplete", "share_incomplete"]].reset_index(drop=True)


def missingness_table(features: pd.DataFrame, regions: pd.DataFrame) -> pd.DataFrame:
    """T01: сначала признаки (медианы у полных и неполных, SMD), затем регионы (число полных и неполных МО,
    доля неполных). Подпись строки — в колонке ``label``, служебных кодов в таблице отчёта нет."""
    feat = features.assign(share_incomplete=np.nan)
    reg = pd.DataFrame(
        {
            "label": regions["region_name"].astype(str) + ", число МО",
            "full": regions["n_full"].astype("float64"),
            "incomplete": regions["n_incomplete"].astype("float64"),
            "smd": np.nan,
            "share_incomplete": regions["share_incomplete"].astype("float64"),
        }
    )
    cols = ["label", "full", "incomplete", "smd", "share_incomplete"]
    return pd.concat([feat[cols], reg[cols]], ignore_index=True)


# --- Контекст: покрытие, шаги соединения, потери --------------------------------------------------


def _split_flags(flags: pd.Series) -> pd.Series:
    """Флаги строки «a;b» → множество имён (пустая строка и пропуск — пустое множество)."""
    return flags.fillna("").astype(str).map(lambda s: frozenset(x for x in s.split(";") if x))


def _indicator_order(codes: Iterable[str]) -> list[str]:
    codes = set(codes)
    known = [c for c in INDICATORS if c in codes]
    return known + sorted(codes - set(known))


def _label_rank(name: str, labels: Mapping[str, str]) -> tuple[int, str]:
    """Порядок флагов и причин: как в словаре подписей, неизвестные — после, по алфавиту."""
    order = list(labels)
    return (order.index(name) if name in labels else len(order), name)


def _labelled_counts(counts: Mapping[str, int], labels: Mapping[str, str]) -> str:
    """«нет кода — 124; в регионе нет годовых строк — 98»: по убыванию числа, при равенстве — в порядке
    ``counts``; нули не показываются."""
    items = sorted(((k, int(v)) for k, v in counts.items() if int(v) > 0), key=lambda kv: -kv[1])
    return "; ".join(f"{labels.get(k, k)} — {style.fmt_num(v)}" for k, v in items)


def _per_mo(context_long: pd.DataFrame) -> pd.DataFrame:
    """Одна строка на (показатель, год, МО): метод строки-итога (разрез ``TOTAL``, у показателей без него —
    первый разрез по алфавиту) и флаги всех разрезов."""
    cl = context_long[["territory_id", "year", "indicator", "dim", "method", "flag"]].copy()
    cl["indicator"] = cl["indicator"].astype(str)
    cl["dim"] = cl["dim"].astype(str)
    cl["year"] = cl["year"].astype(int)
    cl["method"] = cl["method"].astype(str)
    cl["flags"] = _split_flags(cl["flag"])
    keys = ["indicator", "year", "territory_id"]
    has_main = cl.groupby("indicator")["dim"].transform(lambda s: (s == MAIN_DIM).any())
    main = cl.loc[~has_main | (cl["dim"] == MAIN_DIM)].sort_values([*keys, "dim"]).drop_duplicates(keys)
    flags = cl.groupby(keys)["flags"].agg(lambda s: frozenset().union(*s))
    out = main.set_index(keys)[["method"]]
    out["flags"] = flags.reindex(out.index)
    return out.reset_index()


def context_coverage(
    context_long: pd.DataFrame, unmatched: pd.DataFrame, n_mo: int | None = None
) -> pd.DataFrame:
    """T02: показатель × год — МО панели со значением, шаг соединения по ОКТМО, флаги и причины потерь.

    Значение МО за год — строка-итог (``_per_mo``); флаги собираются по всем разрезам. ``unmatched`` этапа
    panel перечисляет все МО панели, чьего значения нет в годовой таблице ``context_annual``, с причиной. Если
    у такого МО значение в ``context_long`` есть (флаги ``union_merged``, ``recipients_jump``), это не потеря,
    а исключение: оно считается в ``n_excluded`` и не попадает в ``n_lost``, поэтому ``n_mo + n_lost`` — все
    МО, которые этап panel проверял за год. Показатель без единого значения тоже попадает в таблицу.
    ``n_mo`` (аргумент) — знаменатель доли (все МО панели); None — доля не считается.

    Колонки: ``indicator``, ``label``, ``source``, ``year``, ``n_mo``, ``share``, ``method_<шаг>`` (все шаги
    ``MATCH_METHODS``), ``flag_<имя>`` и ``lost_<причина>`` (какие встретились), ``n_flagged``,
    ``n_excluded``, ``n_used`` (= ``n_mo − n_excluded``, в годовой таблице), ``n_lost`` и те же числа строкой
    по-русски: ``flags``, ``lost``.
    """
    per_mo = _per_mo(context_long)
    flag_names = sorted({f for s in per_mo["flags"] for f in s}, key=lambda f: _label_rank(f, FLAG_LABELS))
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    for (ind, year), part in per_mo.groupby(["indicator", "year"]):
        methods = part["method"].value_counts()
        row = {"n_mo": int(part["territory_id"].nunique()), "n_flagged": int(part["flags"].map(bool).sum())}
        row |= {f"method_{m}": int(methods.get(m, 0)) for m in MATCH_METHODS}
        row |= {f"flag_{f}": int(part["flags"].map(lambda s, f=f: f in s).sum()) for f in flag_names}
        rows[(str(ind), int(year))] = row

    reasons: list[str] = []
    if len(unmatched):
        um = unmatched[["territory_id", "year", "indicator", "reason"]].copy()
        um["indicator"] = um["indicator"].astype(str)
        um["year"] = um["year"].astype(int)
        um["territory_id"] = um["territory_id"].astype(int)
        um["reason"] = um["reason"].fillna("unknown").astype(str)
        keys = ["indicator", "year", "territory_id"]
        with_value = pd.MultiIndex.from_frame(per_mo[keys].astype({"year": int, "territory_id": int}))
        excluded = pd.MultiIndex.from_frame(um[keys]).isin(with_value)
        n_excluded = um.loc[excluded].groupby(["indicator", "year"])["territory_id"].nunique()
        for (ind, year), n in n_excluded.items():
            rows.setdefault((str(ind), int(year)), {})["n_excluded"] = int(n)
        lost = um.loc[~excluded]
        reasons = sorted(lost["reason"].unique(), key=lambda r: _label_rank(r, REASON_LABELS))
        for (ind, year, reason), n in (
            lost.groupby(["indicator", "year", "reason"])["territory_id"].nunique().items()
        ):
            rows.setdefault((str(ind), int(year)), {})[f"lost_{reason}"] = int(n)
        for (ind, year), n in lost.groupby(["indicator", "year"])["territory_id"].nunique().items():
            rows[(str(ind), int(year))]["n_lost"] = int(n)

    records = []
    for ind in _indicator_order(k[0] for k in rows):
        spec = INDICATORS.get(ind, Indicator(ind, ""))
        for year in sorted(y for i, y in rows if i == ind):
            row = rows[(ind, year)]
            rec: dict[str, Any] = {"indicator": ind, "label": spec.label, "source": spec.source, "year": year}
            rec["n_mo"] = row.get("n_mo", 0)
            rec["share"] = rec["n_mo"] / n_mo if n_mo else np.nan
            rec |= {f"method_{m}": row.get(f"method_{m}", 0) for m in MATCH_METHODS}
            rec |= {f"flag_{f}": row.get(f"flag_{f}", 0) for f in flag_names}
            rec |= {f"lost_{r}": row.get(f"lost_{r}", 0) for r in reasons}
            rec["n_flagged"], rec["n_excluded"] = row.get("n_flagged", 0), row.get("n_excluded", 0)
            rec["n_used"] = rec["n_mo"] - rec["n_excluded"]
            rec["n_lost"] = row.get("n_lost", 0)
            rec["flags"] = _labelled_counts({f: rec[f"flag_{f}"] for f in flag_names}, FLAG_LABELS)
            rec["lost"] = _labelled_counts({r: rec[f"lost_{r}"] for r in reasons}, REASON_LABELS)
            records.append(rec)
    columns = [
        "indicator",
        "label",
        "source",
        "year",
        "n_mo",
        "share",
        *(f"method_{m}" for m in MATCH_METHODS),
    ]
    columns += [f"flag_{f}" for f in flag_names] + [f"lost_{r}" for r in reasons]
    columns += ["n_flagged", "n_excluded", "n_used", "n_lost", "flags", "lost"]
    return pd.DataFrame(records, columns=columns)


# Колонки T02 в CSV и в отчёте: без колонок по отдельным флагам и причинам (они — в строках ``flags``,
# ``lost``) и без источника (5-НДФЛ — ФНС, остальное — Росстат, это видно по названию), иначе таблица
# отчёта не помещается по ширине.
T02_COLUMNS = (
    "indicator",
    "label",
    "year",
    "n_mo",
    *(f"method_{m}" for m in MATCH_METHODS),
    "flags",
    "n_used",
    "n_lost",
    "lost",
)


def context_table(coverage: pd.DataFrame, first_years: Sequence[int]) -> pd.DataFrame:
    """T02 в порядке чтения: сначала годы ``first_years`` (год признаков контекста разведки, затем опорный),
    внутри года — порядок показателей ``coverage``; остальные годы — в конце, по возрастанию."""
    first = [int(y) for y in dict.fromkeys(first_years)]
    rank = coverage["year"].astype(int).map(lambda y: first.index(y) if y in first else len(first) + y)
    order = np.lexsort((np.arange(len(coverage)), rank.to_numpy()))
    return coverage.iloc[order][list(T02_COLUMNS)].reset_index(drop=True)


def regions_without_rows(
    unmatched: pd.DataFrame, territories: pd.DataFrame, indicator: str, year: int
) -> pd.DataFrame:
    """Регионы, где у показателя за год нет годового значения ни у одного МО (причина ``region_no_rows`` в
    ``unmatched``: строк нужного периода у субъекта нет, хотя могут быть квартальные): название региона из
    справочника территорий и число МО панели, по убыванию числа, затем по названию."""
    columns = ["region_name", "n_mo"]
    if unmatched.empty:
        return pd.DataFrame(columns=columns)
    um = unmatched
    sel = (um["indicator"].astype(str) == indicator) & (um["year"].astype(int) == int(year))
    sel &= um["reason"].astype(str) == REGION_NO_ROWS
    names = territories.set_index(territories["territory_id"].astype(int))["region_name"].astype(str)
    ids = um.loc[sel, "territory_id"].astype(int).drop_duplicates()
    counts = ids.map(names).value_counts()
    out = pd.DataFrame({"region_name": counts.index.astype(str), "n_mo": counts.to_numpy(dtype=int)})
    return out.sort_values(["n_mo", "region_name"], ascending=[False, True]).reset_index(drop=True)[columns]


def coverage_lookup(coverage: pd.DataFrame) -> dict[tuple[str, int], int]:
    """(показатель, год) → число МО панели со значением."""
    rows = coverage[["indicator", "year", "n_mo"]].itertuples(index=False)
    return {(str(i), int(y)): int(n) for i, y, n in rows}


# --- Контрольные числа ----------------------------------------------------------------------------


def control_label(name: str) -> str:
    """Подпись контрольного числа: из ``CONTROL_LABELS`` или по шаблонам «method_<показатель>_<год>_<шаг>»
    и «<показатель>_<год>»; незнакомое имя остаётся как есть."""
    if name in CONTROL_LABELS:
        return CONTROL_LABELS[name]
    m = _CONTROL_METHOD.fullmatch(name)
    if m and m.group(1) in INDICATORS:
        step = METHOD_LABELS.get(m.group(3), m.group(3)).lower()
        return f"{INDICATORS[m.group(1)].label}, {m.group(2)}: МО по шагу «{step}»"
    m = _CONTROL_YEAR.fullmatch(name)
    if m:
        ind = _CONTROL_ALIASES.get(m.group(1), m.group(1))
        if ind in INDICATORS:
            return f"{INDICATORS[ind].label}, {m.group(2)}: МО со значением"
    return name


def _deviation(expected: Any, actual: Any) -> float:
    try:
        exp, act = float(expected), float(actual)
    except (TypeError, ValueError):
        return np.nan
    return act / exp - 1 if exp != 0 else np.nan


def controls_table(controls: Mapping[str, Mapping[str, Any]], order: Sequence[str] = ()) -> pd.DataFrame:
    """T00: контрольные числа ``controls.json`` — ключ, подпись, вид, ожидалось, получено, отклонение, статус.

    Сначала жёсткие, потом мягкие; внутри — порядок ``order`` (порядок конфига), остальные по алфавиту.
    Отклонение — ``получено / ожидалось − 1`` (при нулевом или нечисловом ожидании — пропуск).
    """
    rank = {name: i for i, name in enumerate(order)}
    columns = ["name", "label", "kind", "expected", "actual", "deviation", "status"]
    rows = []
    for name, rec in controls.items():
        kind, ok = str(rec.get("kind", "")), bool(rec.get("ok", False))
        rows.append(
            {
                "name": str(name),
                "label": control_label(str(name)),
                "kind": KIND_LABELS.get(kind, kind),
                "expected": rec.get("expected"),
                "actual": rec.get("actual"),
                "deviation": _deviation(rec.get("expected"), rec.get("actual")),
                "status": STATUS_TEXT.get((kind, ok), "сошлось" if ok else "не сошлось"),
                "_kind": KIND_ORDER.index(kind) if kind in KIND_ORDER else len(KIND_ORDER),
                "_rank": rank.get(str(name), len(rank)),
            }
        )
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows).sort_values(["_kind", "_rank", "name"])[columns].reset_index(drop=True)


def controls_summary(controls: Mapping[str, Mapping[str, Any]]) -> dict[str, int]:
    """Сколько жёстких и мягких контрольных чисел и сколько из них не сошлось."""
    kinds = [(str(r.get("kind", "")), bool(r.get("ok", False))) for r in controls.values()]
    return {
        "hard_n": sum(k == "hard" for k, _ in kinds),
        "hard_failed": sum(k == "hard" and not ok for k, ok in kinds),
        "soft_n": sum(k == "soft" for k, _ in kinds),
        "soft_warnings": sum(k == "soft" and not ok for k, ok in kinds),
    }


# --- Графики --------------------------------------------------------------------------------------


def status_colors() -> dict[str, str]:
    """Цвет группы статуса ряда: точки шкал стиля из ``STATUS_COLOR_SPEC``."""
    return {g: mpl.colors.to_hex(mpl.colormaps[cmap](pos)) for g, (cmap, pos) in STATUS_COLOR_SPEC.items()}


def status_legend(counts: Mapping[str, int]) -> dict[str, str]:
    """Подпись группы на карте с числом МО: «полный ряд — 2016». Подписи короткие: легенда под картой
    в одну строку, а слишком широкая легенда сжимает карту."""
    return {g: f"{STATUS_LABELS[g]} — {style.fmt_num(counts.get(g, 0))}" for g in STATUS_LABELS}


def absent_note(absent: pd.DataFrame) -> str | None:
    """Текст выноски о выпавших регионах: число и названия, перенос по ``REGION_NOTE_WIDTH`` знаков."""
    if absent.empty:
        return None
    n = len(absent)
    head = f"Без данных СберИндекса {n}{style.NBSP}{plural(n, 'регион', 'региона', 'регионов')}:"
    return "\n".join([head, *textwrap.wrap(join_regions(absent["region_name"]), REGION_NOTE_WIDTH)])


def plot_coverage_map(
    status: pd.Series,
    geo: gpd.GeoDataFrame,
    year: int,
    counts: Mapping[str, int],
    absent: pd.DataFrame,
    insets: Sequence[maps.Inset],
) -> Figure:
    """F01: карта групп статуса ряда в границах ``year``: вне панели — штриховка, выпавшие регионы —
    выноска."""
    labels, colors = status_legend(counts), status_colors()
    fig, _ = maps.russia_map(
        status.map(labels),
        geo,
        year,
        kind="categorical",
        cmap={labels[g]: colors[g] for g in STATUS_LABELS},
        legend_title=f"Ряд трат СберИндекса за {months_word(N_MONTHS)} {YEAR_SPAN} годов, число МО",
        insets=insets,
        absent_note=absent_note(absent),
    )
    return fig


def _hbar_axis(ax) -> None:
    """Горизонтальные столбцы: сетка только по оси значений, без левой рамки и засечек подписей."""
    ax.grid(False, axis="y")
    ax.grid(True, axis="x")
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)


def plot_funnel(steps: pd.DataFrame, coverage: pd.DataFrame, n_mo: int, years: Sequence[int]) -> Figure:
    """F02: вверху — воронка среза справочника, внизу — МО панели со значением показателей контекста по годам.

    Ось X общая (число МО), столбцы от нуля; пунктир — все МО панели; годы подписаны прямо на столбцах первой
    строки, показатели отсортированы по лучшему из годов покрытию.
    """
    shown = [i for i, spec in INDICATORS.items() if spec.on_chart]
    cov = coverage.loc[coverage["indicator"].isin(shown) & coverage["year"].isin(years)]
    best = cov.groupby("indicator")["n_mo"].max()
    order = sorted(best.index, key=lambda i: (-best[i], _indicator_order(best.index).index(i)))
    fig, (top, bottom) = style.new_figure(
        "tall",
        nrows=2,
        ncols=1,
        sharex=True,
        gridspec_kw={"height_ratios": [len(steps), PAIR_ROW_RATIO * max(len(order), 1)]},
    )
    colors = [style.CONTEXT] * (len(steps) - 1) + [style.ACCENT]
    bars = top.barh(np.arange(len(steps)), steps["n_mo"], height=BAR_HEIGHT, color=colors)
    top.bar_label(
        bars, [style.fmt_num(v) for v in steps["n_mo"]], padding=LABEL_PAD_PT, fontsize=BAR_LABEL_PT
    )
    top.set_yticks(np.arange(len(steps)), steps["label"])
    top.invert_yaxis()

    year_colors = (style.ACCENT, style.CONTEXT)
    height = BAR_HEIGHT / len(years)
    row = {ind: i for i, ind in enumerate(order)}
    for j, year in enumerate(years):
        part = cov.loc[cov["year"] == year]
        color = year_colors[j % len(year_colors)]
        yy = part["indicator"].map(row).to_numpy() - BAR_HEIGHT / 2 + height * (j + 0.5)
        bars = bottom.barh(yy, part["n_mo"], height=height, color=color)
        bottom.bar_label(
            bars, [style.fmt_num(v) for v in part["n_mo"]], padding=LABEL_PAD_PT, fontsize=BAR_LABEL_PT
        )
        first = part["indicator"].to_numpy() == order[0]
        if first.any():  # прямая подпись года на первой строке вместо легенды
            bottom.annotate(
                str(year),
                (0, yy[first][0]),
                xytext=(LABEL_PAD_PT, 0),
                textcoords="offset points",
                va="center",
                ha="left",
                fontsize=BAR_LABEL_PT,
                color="white" if color == style.ACCENT else style.TEXT,
            )
    bottom.set_yticks(np.arange(len(order)), [INDICATORS.get(i, Indicator(i, "")).label for i in order])
    bottom.invert_yaxis()
    bottom.axvline(n_mo, color=style.TEXT2, linewidth=0.8, linestyle=":")
    bottom.annotate(
        f"все МО панели: {style.fmt_num(n_mo)}",
        (n_mo, 1),
        xycoords=("data", "axes fraction"),
        xytext=(-LABEL_PAD_PT, LABEL_PAD_PT),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=BAR_LABEL_PT,
        color=style.TEXT2,
    )
    for ax in (top, bottom):
        _hbar_axis(ax)
    # У верхней панели шкала общая с нижней: без оси и засечек, чтобы не казалось, что шкала обрывается.
    top.spines["bottom"].set_visible(False)
    top.tick_params(axis="x", length=0)
    top.set_xlim(0, max(float(steps["n_mo"].max()), n_mo) * XLIM_PAD)
    bottom.set_xlabel("Число МО")
    return fig


def lowest_coverage(coverage: pd.DataFrame, year: int) -> str:
    """«оборот общепита (1402 МО)» — показатель графика F02 с наименьшим покрытием за год."""
    on_chart = [i for i, spec in INDICATORS.items() if spec.on_chart]
    part = coverage.loc[(coverage["year"] == year) & coverage["indicator"].isin(on_chart)]
    if part.empty:
        return "нет данных"
    row = part.sort_values("n_mo").iloc[0]
    return f"{str(row['label']).lower()} ({style.fmt_num(row['n_mo'])} МО)"


# --- Раздел ---------------------------------------------------------------------------------------


def _facts(ctx: SectionContext, items: Iterable[tuple]) -> None:
    """Регистрирует факты из кортежей (ключ, значение, вид[, пояснение])."""
    for key, value, kind, *note in items:
        ctx.fact(key, value, kind, note=note[0] if note else "")


def _series_facts(ctx: SectionContext, params: Mapping[str, Any], ref_year: int) -> dict[str, Any]:
    """Ряды трат: статусы, МО по месяцам, пороги длины ряда, внутригородские территории."""
    ter, mo = ctx.data.territories, ctx.data.mo
    s = series_summary(ter)
    _facts(ctx, ((k, v, "int", SERIES_NOTES.get(k, "")) for k, v in s.items()))
    per_month = months_coverage(ctx.data.panel_wide)
    lo, hi = per_month.loc[per_month["n_mo"].idxmin()], per_month.loc[per_month["n_mo"].idxmax()]
    ladder = month_ladder(mo, params["month_thresholds"])
    inner = inner_city(mo)
    weight_note = "вес — среднегодовое население 2023 года, иначе 2024-го"
    _facts(
        ctx,
        [
            ("full_share", s["n_full"] / s["n_mo"], "pct", "доля МО панели с полным рядом"),
            ("mo_per_month_min", lo["n_mo"], "int", "меньше всего МО с тратами в одном месяце"),
            ("mo_per_month_min_date", month_name(lo["date"]), "str"),
            ("mo_per_month_max", hi["n_mo"], "int", "больше всего МО с тратами в одном месяце"),
            ("mo_per_month_max_date", month_name(hi["date"]), "str"),
            (
                f"n_full_{ref_year}",
                int((ter[f"n_{ref_year}"].astype(int) == 12).sum()),
                "int",
                f"МО со всеми 12 месяцами {ref_year} года",
            ),
            ("inner_city_node_share", inner["node_share"], "pct", "доля внутригородских среди МО панели"),
            ("pop_share_inner_city", inner["pop_share"], "pct", f"доля населения выборки; {weight_note}"),
            (
                "inner_city_by_region",
                counts_text(inner["by_region"].index, inner["by_region"].to_numpy()),
                "str",
            ),
        ],
    )
    for row in ladder.itertuples(index=False):
        k = int(row.min_months)
        _facts(
            ctx,
            [
                (f"n_months_ge{k}", row.n_mo, "int", f"МО с рядом не короче {k} месяцев"),
                (
                    f"pop_share_months_ge{k}",
                    row.pop_share,
                    "pct",
                    f"их доля населения выборки; {weight_note}",
                ),
            ],
        )
    return {**s, "thresholds": [int(k) for k in ladder["min_months"]]}


def _absence_facts(
    ctx: SectionContext, frame: gpd.GeoDataFrame, names: pd.Series, year: int, k_text: int
) -> dict[str, Any]:
    """Кого нет: срез справочника ``year``, выпавшие регионы, МО без данных в остальных регионах."""
    ter = ctx.data.territories
    absent = absent_regions(frame, names)
    outside = missing_outside(frame, names)
    steps = funnel(frame, ter, year)
    n_absent = len(absent)
    _facts(
        ctx,
        [
            (f"n_slice_{year}", len(frame), "int", f"МО справочника в срезе {year} года"),
            (
                f"n_panel_in_slice_{year}",
                steps.loc[steps["step"] == "with_spend", "n_mo"].iloc[0],
                "int",
                f"МО среза {year} года с тратами СберИндекса",
            ),
            ("n_absent_regions", n_absent, "int", f"регионы среза {year} года без единого МО панели"),
            ("absent_regions", join_regions(absent["region_name"]) if n_absent else "нет", "str"),
            (
                "n_missing_outside",
                int(outside["n_missing"].sum()),
                "int",
                f"МО среза {year} года без трат в регионах, где другие МО есть",
            ),
            (
                "missing_outside_regions",
                counts_text(outside["region_name"], outside["n_missing"], k_text) or "нет",
                "str",
                "регионы с наибольшим числом МО без данных вне выпавших регионов",
            ),
        ],
    )
    return {"absent": absent, "steps": steps, "n_slice": len(frame)}


def _missing_facts(ctx: SectionContext, params: Mapping[str, Any]) -> dict[str, Any]:
    """Случайны ли пропуски: медианы населения, Манн — Уитни, модель пропуска, регионы с неполными рядами."""
    mo = ctx.data.mo
    feats, auc, p = missingness(mo, seed=int(ctx.cfg["seed"]))
    regions = incomplete_by_region(mo)
    incomplete = mo["series_status"].astype(str) != "full"
    pop = mo[POP_COLUMN].astype("float64")
    pop_full, pop_inc = float(pop[~incomplete].median()), float(pop[incomplete].median())
    weight = mo["weight"].astype("float64")
    note = "медиана среднегодового населения 2023 года (МО без него не входят)"
    # Ориентиры Б.4 посчитаны по населению на 1 января 2023 года; раздел берёт среднегодовое (оно же вес
    # «типичного жителя», Б.1), поэтому медианы немного отличаются от ориентиров.
    ref = "ориентир spec_final Б.4 посчитан по населению на 1 января 2023 года, поэтому немного отличается"
    _facts(
        ctx,
        [
            ("pop_median_full", pop_full, "int", f"{note}, МО с полным рядом; {ref}"),
            ("pop_median_incomplete", pop_inc, "int", f"{note}, МО с неполным рядом; {ref}"),
            (
                "pop_share_incomplete",
                _share(weight[incomplete].sum(), weight.sum()),
                "pct",
                "их доля населения выборки (вес — среднегодовое население 2023 года, иначе 2024-го)",
            ),
            ("missing_mw_p", p, "p", "Манн — Уитни, население неполных против полных, двусторонний"),
            ("missing_auc", auc, "num2", "логистическая модель пропуска, стратифицированная кросс-валидация"),
            ("smd_ln_pop", feats.set_index("feature").loc["ln_pop", "smd"], "num2", "SMD ln населения"),
            (
                "incomplete_regions",
                counts_text(regions["region_name"], regions["n_incomplete"], int(params["text_regions"]))
                or "нет",
                "str",
                "регионы с наибольшим числом неполных рядов",
            ),
        ],
    )
    return {
        "features": feats,
        "regions": regions,
        "auc": auc,
        "p": p,
        "pop_smaller": pop_inc < pop_full,
        "nonrandom": bool(p < float(params["missing_alpha"])),
    }


def _indicator_label(ind: str) -> str:
    return INDICATORS[ind].label if ind in INDICATORS else ind


def _context_facts(
    ctx: SectionContext, n_mo: int, params: Mapping[str, Any], ref_year: int
) -> dict[str, Any]:
    """Покрытие контекста: ``cov_<показатель>_<год>`` для всех показателей и обязательных (0, если нет);
    регионы, где у показателя ``gap_indicator`` за опорный год нет годового значения ни у одного МО."""
    coverage = context_coverage(ctx.data.context_long, ctx.data.unmatched, n_mo=n_mo)
    cov = coverage_lookup(coverage)
    used = coverage.set_index(["indicator", "year"])["n_used"]
    for (ind, year), n in sorted(cov.items()):
        note = f"МО панели со значением: {_indicator_label(ind)} ({ind}), {year} год"
        if used[(ind, year)] != n:
            note += f"; в годовой таблице context_annual — {used[(ind, year)]} (остальные исключены флагом)"
        ctx.fact(f"cov_{fact_key(ind)}_{year}", n, "int", note=note)
    for ind in KEY_INDICATORS:
        for year in YEARS:
            if (ind, year) not in cov:
                ctx.fact(
                    f"cov_{fact_key(ind)}_{year}", 0, "int", note=f"{ind} за {year} год нет в context_long"
                )
    gap_ind = str(params["gap_indicator"])
    gaps = regions_without_rows(ctx.data.unmatched, ctx.data.territories, gap_ind, ref_year)
    where = f"{_indicator_label(gap_ind).lower()} ({gap_ind}) за {ref_year} год"
    _facts(
        ctx,
        [
            (
                f"n_gap_regions_{ref_year}",
                len(gaps),
                "int",
                f"регионы, где нет годового значения ни у одного МО: {where}",
            ),
            (f"gap_regions_{ref_year}", join_regions(gaps["region_name"]) or "нет", "str"),
            (f"n_gap_mo_{ref_year}", int(gaps["n_mo"].sum()), "int", f"МО панели в этих регионах: {where}"),
        ],
    )
    return {"coverage": coverage, "gap_regions": gaps, "gap_indicator": gap_ind}


def hard_ok_text(summary: Mapping[str, int]) -> str:
    """«да» — все жёсткие контрольные числа сошлись, «нет» — хотя бы одно нет, «нет данных» — их нет вовсе."""
    if summary["hard_n"] == 0:
        return "нет данных"
    return "да" if summary["hard_failed"] == 0 else "нет"


def _controls_facts(ctx: SectionContext) -> dict[str, int]:
    """Контрольные числа этапа panel: сколько и сколько не сошлось."""
    c = controls_summary(ctx.data.controls)
    _facts(
        ctx,
        [
            ("controls_hard_ok", hard_ok_text(c), "str", "все ли жёсткие контрольные числа сошлись"),
            ("controls_hard_n", c["hard_n"], "int", "жёстких контрольных чисел в controls.json"),
            ("controls_hard_failed", c["hard_failed"], "int"),
            ("controls_soft_n", c["soft_n"], "int", "мягких контрольных чисел в controls.json"),
            ("controls_soft_warnings", c["soft_warnings"], "int", "мягких чисел вне допуска"),
            (
                "stable_union_values",
                ctx.data.controls.get(STABLE_UNION_KEY, {}).get("actual"),
                "int",
                "значения, найденные мостом oktmo_stable у МО с объединением в истории ОКТМО (А.6)",
            ),
        ],
    )
    return c


def _save_map(
    ctx: SectionContext,
    geo: gpd.GeoDataFrame,
    names: pd.Series,
    s: Mapping[str, Any],
    gaps: Mapping[str, Any],
    miss: Mapping[str, Any],
    year: int,
) -> None:
    """F01: карта статуса ряда в границах ``year``; заголовок проверяет «выпавшие регионы — на юго-западе»."""
    ter = ctx.data.territories
    absent = gaps["absent"]
    n_absent = len(absent)
    if n_absent:
        sw = south_west(absent, ter["point_lat"].astype("float64"), ter["point_lon"].astype("float64"))
        regions_word = plural(n_absent, "регион", "региона", "регионов")
        verb = plural(n_absent, "выпал", "выпали", "выпали")
        tail = f"{n_absent}{style.NBSP}{regions_word} юго-запада {verb}"
        ok, detail = sw, f"F01: выпавших регионов {n_absent}; все южнее и западнее медианы МО панели — {sw}"
    else:
        tail, ok, detail = "целиком выпавших регионов нет", True, "F01: выпавших регионов нет"
    n_mo, n_reg = s["n_mo"], s["n_regions"]
    title = ctx.headline(
        f"Траты есть по {n_mo}{style.NBSP}МО {n_reg}{style.NBSP}"
        f"{genitive_plural(n_reg, 'региона', 'регионов')}; {tail}",
        ok,
        detail,
    )
    status = status_groups(ter)
    frame = maps.map_frame(geo, year)
    # Легенда считает только видимые на карте МО: МО панели, упразднённых к году карты, в её границах нет.
    drawn = status[status.index.isin(frame.loc[frame["in_panel"].astype(bool), "territory_id"].astype(int))]
    fig = plot_coverage_map(
        status,
        geo,
        year,
        drawn.value_counts().to_dict(),
        absent_regions(frame, names),
        maps.insets_from_config(ctx.cfg),
    )
    by_id = ter.set_index("territory_id")
    data = frame[["territory_id", "region_code", "in_panel"]].copy()
    data["region_name"] = data["region_code"].astype(int).map(names)
    data["status"] = data["territory_id"].map(status).fillna(NO_DATA)
    data["series_status"] = data["territory_id"].map(by_id["series_status"].astype(str))
    data["n_months"] = data["territory_id"].map(by_id["n_months"].astype("Int64"))
    top = miss["regions"]["region_name"].iloc[0] if len(miss["regions"]) else "нет"
    absent_text = join_regions(absent["region_name"]) if n_absent else "нет"
    nb = style.NBSP
    # Заголовок и подзаголовок — в одну строку на ширине карты (длиннее — перенос и карта меньше).
    subtitle = f"Статус ряда трат за {YEAR_SPAN} годы; на карте {len(drawn)} из {n_mo}{nb}МО панели"
    if len(drawn) < n_mo:
        subtitle += f" в границах {year} года"
    ctx.save_figure(
        fig,
        fid="F01",
        slug="coverage_map",
        title=title,
        subtitle=subtitle,
        alt=(
            f"Карта России: траты есть по {n_mo} МО, полный ряд у {s['n_full']}; целиком без данных "
            f"{n_absent} {plural(n_absent, 'регион', 'региона', 'регионов')} ({absent_text}); "
            f"больше всего неполных рядов — {top}"
        ),
        data=data.sort_values("territory_id"),
        check=(
            "выпавшие регионы есть, и все они южнее и западнее медианной точки МО панели"
            if n_absent
            else "выпавших регионов нет, заголовок это и говорит"
        ),
    )


def _save_funnel(
    ctx: SectionContext,
    s: Mapping[str, Any],
    gaps: Mapping[str, Any],
    coverage: pd.DataFrame,
    params: Mapping[str, Any],
    slice_year: int,
    year: int,
) -> None:
    """F02: воронка среза справочника ``slice_year`` и покрытие контекста; в заголовке — покрытие года
    ``year`` (год признаков контекста разведки). Проверка заголовка: полный ряд — у подавляющего большинства
    МО."""
    cov = coverage_lookup(coverage)
    full_min = float(params["full_share_min"])
    nb = style.NBSP
    # Год покрытия — в подписях столбцов, не в заголовке: с ним заголовок не помещается в одну строку.
    parts = [f"Полный ряд — у{nb}{s['n_full']}{nb}МО"]
    if cov.get(("wage", year)):
        parts.append(f"зарплата — у{nb}{style.fmt_num(cov[('wage', year)])}")
    if cov.get(("ndfl_income", year)):
        parts.append(f"доход 5-НДФЛ — у{nb}{style.fmt_num(cov[('ndfl_income', year)])}")
    title = ctx.headline(
        ", ".join(parts),
        s["n_full"] >= full_min * s["n_mo"],
        f"F02: полный ряд у {s['n_full']} из {s['n_mo']} МО, нужно не меньше {style.fmt_pct(full_min, 0)}",
    )
    steps = gaps["steps"]
    fig = plot_funnel(steps, coverage, s["n_mo"], list(YEARS))
    cols = ["block", "step", "label", "year", "n_mo"]
    context = coverage.loc[coverage["year"].isin(YEARS)].rename(columns={"indicator": "step"})
    data = pd.concat(
        [steps.assign(block="funnel", year=slice_year)[cols], context.assign(block="context")[cols]],
        ignore_index=True,
    )
    ctx.save_figure(
        fig,
        fid="F02",
        slug="coverage_funnel",
        title=title,
        subtitle=(
            f"Число МО: вверху — справочник {slice_year} года, внизу — МО панели со значением показателя "
            "по годам"
        ),
        alt=(
            f"Из {gaps['n_slice']} МО справочника траты есть у {int(steps['n_mo'].iloc[1])}, полный ряд — "
            f"у {s['n_full']}; показатели Росстата и ФНС есть не у всех МО панели, меньше всего — "
            f"{lowest_coverage(coverage, year)}"
        ),
        data=data,
        check=f"n_full ≥ {style.fmt_num(full_min, style.auto_decimals([full_min]))} · n_mo",
        source=style.join_sources(style.SOURCE_SBER, style.SOURCE_ROSSTAT, style.SOURCE_FNS),
    )


def context_title(controls: Mapping[str, Mapping[str, Any]], shown_year: int | None = None) -> str:
    """Заголовок T02; ``shown_year`` — год, строки которого идут в отчёт первыми (остальные годы — в CSV);
    если этап panel записал ``stable_union_values`` — с этим числом (ручная проверка запрета моста при
    объединениях, А.6)."""
    title = "Контекст Росстата и ФНС: МО панели со значением, шаг соединения по ОКТМО, флаги и потери"
    if shown_year is not None:
        title += f"; здесь {shown_year} год — год признаков контекста разведки, остальные годы — в CSV"
    union = controls.get(STABLE_UNION_KEY, {}).get("actual")
    if union is None:
        return title
    return f"{title}; значений моста у МО с объединением в истории ОКТМО — {style.fmt_num(union)}"


def _save_tables(
    ctx: SectionContext,
    csum: Mapping[str, int],
    miss: Mapping[str, Any],
    coverage: pd.DataFrame,
    ref_year: int,
) -> None:
    """T00 (целиком в отчёте — «Как пересобрать и проверить»), T01, T02."""
    controls_cfg = ctx.cfg["panel"]["controls"]
    t00 = controls_table(ctx.data.controls, [*controls_cfg["hard"], *controls_cfg["soft"]])
    ctx.save_table(
        t00,
        tid="T00",
        slug="controls",
        title=(
            f"Контрольные числа этапа panel: жёстких {csum['hard_n']}, не сошлось {csum['hard_failed']}; "
            f"мягких {csum['soft_n']}, вне допуска {csum['soft_warnings']}"
        ),
        md_rows=int(t00["kind"].isin([KIND_LABELS[k] for k in CHECKED_KINDS]).sum()),
        md_formats={"expected": _fmt_exact, "actual": _fmt_exact, "deviation": _fmt_deviation},
        md_labels={
            "name": "Ключ",
            "label": "Что проверяется",
            "kind": "Вид",
            "expected": "Ожидалось",
            "actual": "Получено",
            "deviation": "Отклонение",
            "status": "Статус",
        },
    )
    nb = style.NBSP
    ctx.save_table(
        missingness_table(miss["features"], miss["regions"]),
        tid="T01",
        slug="missingness",
        title=(
            "Неполные ряды против полных: SMD по признакам МО; AUC модели пропуска "
            f"{style.fmt_num(miss['auc'], 2)}; тест Манна — Уитни по населению, "
            f"p{nb}={nb}{style.fmt_p(miss['p'])}"
        ),
        md_formats={
            "full": _fmt_auto,
            "incomplete": _fmt_auto,
            "smd": _fmt_smd,
            "share_incomplete": _fmt_share,
        },
        md_labels={
            "label": "Признак или регион",
            "full": "Полные ряды",
            "incomplete": "Неполные ряды",
            "smd": "SMD",
            "share_incomplete": "Доля неполных",
        },
    )
    # В отчёт идут строки года признаков контекста разведки (``mo`` берёт контекст этого года, Б.2): так
    # таблица читается целиком, а не обрывается на середине списка показателей.
    t02 = context_table(coverage, [CONTEXT_YEAR, ref_year])
    shown = int((t02["year"] == CONTEXT_YEAR).sum())
    ctx.save_table(
        t02,
        tid="T02",
        slug="context_coverage",
        title=context_title(ctx.data.controls, CONTEXT_YEAR if 0 < shown < len(t02) else None),
        md_rows=shown if shown else len(t02),
        md_labels={
            "indicator": "Код",
            "label": "Показатель",
            "year": "Год",
            "n_mo": "МО со значением",
            **{f"method_{m}": METHOD_LABELS.get(m, m) for m in MATCH_METHODS},
            "flags": "Флаги",
            "n_used": "В годовой таблице",
            "n_lost": "Нет значения",
            "lost": "Почему нет",
        },
    )


def run_section(ctx: SectionContext) -> Finding:
    """Раздел E1 на ``ctx.data``: факты, F01, F02, T00, T01, T02 и текст «Что видно / что это значит»."""
    params = coverage_params(ctx.cfg)
    slice_year, ref_year = YEARS[0], int(ctx.cfg["eda"]["reference_year"])
    geo = maps.load_geometry(ctx.data.geo_path)
    if "region_name" not in geo.columns:
        log.warning("В territories_geo нет region_name: регионы без МО панели будут подписаны кодами")
    names = region_names(geo, ctx.data.territories)

    s = _series_facts(ctx, params, ref_year)
    gaps = _absence_facts(
        ctx, maps.map_frame(geo, slice_year), names, slice_year, int(params["text_regions"])
    )
    miss = _missing_facts(ctx, params)
    context = _context_facts(ctx, s["n_mo"], params, ref_year)
    coverage = context["coverage"]
    csum = _controls_facts(ctx)

    _save_map(ctx, geo, names, s, gaps, miss, ref_year)
    _save_funnel(ctx, s, gaps, coverage, params, slice_year, CONTEXT_YEAR)
    _save_tables(ctx, csum, miss, coverage, ref_year)

    cov = coverage_lookup(coverage)
    gap_ind = context["gap_indicator"]
    summary = summary_md(
        s,
        n_absent=len(gaps["absent"]),
        absent_has_resorts=bool(gaps["absent"]["region_name"].isin(RESORT_REGIONS).any()),
        pop_smaller=miss["pop_smaller"],
        nonrandom=miss["nonrandom"],
        thresholds=s["thresholds"],
        ref_year=ref_year,
        slice_year=slice_year,
        context_year=CONTEXT_YEAR,
        has_cov={ind: (ind, CONTEXT_YEAR) in cov for ind in KEY_INDICATORS},
        n_gap_regions=len(context["gap_regions"]),
        gap_indicator=INDICATORS.get(gap_ind, Indicator(gap_ind, "")),
    )
    return ctx.finding(title=TITLE, summary_md=summary, caveats=CAVEATS)


def summary_md(
    s: Mapping[str, Any],
    *,
    n_absent: int,
    absent_has_resorts: bool,
    pop_smaller: bool,
    nonrandom: bool,
    thresholds: Sequence[int],
    ref_year: int,
    slice_year: int,
    context_year: int,
    has_cov: Mapping[str, bool],
    n_gap_regions: int = 0,
    gap_indicator: Indicator | None = None,
) -> str:
    """«Что видно» и «Что это значит для сюжета»: числа — только ``{{e1.ключ}}``; формы слов и утверждения
    выбираются по данным («неслучайны» — только при p ниже порога, «курорты» — только если их регионы
    выпали, фраза о регионах без строк источника — только если такие регионы есть).

    ``slice_year`` — срез справочника воронки, ``context_year`` — год признаков контекста разведки,
    ``ref_year`` — опорный год трат; ``n_gap_regions`` — регионы без строк показателя ``gap_indicator``
    за опорный год."""
    nb = style.NBSP

    def f(key: str) -> str:
        return f"{{{{{SECTION_ID}.{key}}}}}"

    regions_gen = genitive_plural(s["n_regions"], "региона", "регионов")
    seen = [
        f"Траты есть по {f('n_mo')} МО {f('n_regions')} {regions_gen}. Полный ряд за "
        f"{months_word(N_MONTHS)} — у{nb}{f('n_full')} МО ({f('full_share')}), ровно {YEARS[0]} год — "
        f"у{nb}{f(f'n_only_{YEARS[0]}')}, ровно {YEARS[1]} год — у{nb}{f(f'n_only_{YEARS[1]}')}, прочие "
        "неполные — "
        f"у{nb}{f('n_partial')}; больше всего неполных рядов "
        f"в{nb}регионах: {f('incomplete_regions')}. Меньше всего МО с{nb}тратами в{nb}одном месяце — "
        f"{f('mo_per_month_min')} ({f('mo_per_month_min_date')}), больше всего — {f('mo_per_month_max')} "
        f"({f('mo_per_month_max_date')})."
    ]
    if n_absent:
        seen.append(
            f"Целиком без данных СберИндекса {f('n_absent_regions')} "
            f"{plural(n_absent, 'регион', 'региона', 'регионов')}: {f('absent_regions')}; в{nb}остальных "
            f"регионах нет ещё {f('n_missing_outside')} МО справочника {slice_year} года (больше всего: "
            f"{f('missing_outside_regions')})."
        )
    tests = (
        f"тест Манна — Уитни, p{nb}={nb}{f('missing_mw_p')}); логистическая модель по населению, зарплате, "
        f"доле горожан, широте, доступности рынков и типу МО отличает неполные ряды от полных с{nb}AUC "
        f"{f('missing_auc')} (0,5 — не лучше угадывания)."
    )
    if pop_smaller:
        seen.append(
            f"Неполные ряды — у{nb}меньших МО: медиана населения {f('pop_median_incomplete')} человек против "
            f"{f('pop_median_full')} у{nb}полных ({tests}"
        )
    else:
        seen.append(
            f"Медиана населения МО с неполным рядом — {f('pop_median_incomplete')} человек, с полным — "
            f"{f('pop_median_full')} ({tests}"
        )
    cov_parts = []
    if has_cov.get("population"):
        cov_parts.append(f"население на 1{nb}января — у{nb}{f(f'cov_population_{context_year}')} МО панели")
    if has_cov.get("wage"):
        cov_parts.append(f"средняя зарплата — у{nb}{f(f'cov_wage_{context_year}')}")
    if has_cov.get("ndfl_income"):
        cov_parts.append(f"доход по 5-НДФЛ — у{nb}{f(f'cov_ndfl_income_{context_year}')}")
    if cov_parts:
        seen.append(f"Контекст {context_year} года: " + ", ".join(cov_parts) + " (таблица T02).")
    if n_gap_regions and gap_indicator is not None and ref_year != context_year:
        where = genitive_plural(n_gap_regions, "регионе", "регионах")
        who = SOURCE_GENITIVE.get(gap_indicator.source, gap_indicator.source)
        source = f"у{nb}{who} " if who else ""
        seen.append(
            f"В{nb}{ref_year} году {source}нет годового значения показателя «{gap_indicator.label.lower()}» "
            f"ни у{nb}одного МО в{nb}{f(f'n_gap_regions_{ref_year}')} {where}: "
            f"{f(f'gap_regions_{ref_year}')} ({f(f'n_gap_mo_{ref_year}')} МО панели), поэтому признаки "
            f"контекста разведки взяты за{nb}{context_year} год."
        )

    regions_prep = genitive_plural(s["n_regions"], "регионе", "регионах")  # предложный падеж: о 77 регионах
    means = [f"Выводы разведки — о{nb}{f('n_regions')} {regions_prep} с тратами, а не обо всей России."]
    if absent_has_resorts:
        means.append(
            "Курорты Черноморского побережья в данные не попали, а траты приезжих в данные курортного МО "
            "не входят: траты привязаны к жителям. Поэтому курортный тип по тратам не выделить, только по "
            "ночёвкам и местам в средствах размещения (Росстат)."
        )
    if nonrandom:
        means.append(
            "Пропуски неслучайны, поэтому неполные МО нельзя молча выбросить: выводы о типах — о покрытых "
            "МО, а неполные ряды показываются отдельно."
        )
    if thresholds:
        ks = [int(k) for k in thresholds]
        months = genitive_plural(ks[-1], "месяца", "месяцев")  # «не короче 12, 18 и 24 месяцев»
        means.append(
            "Порог длины ряда для узла сети выбирает этап 2: ряд не короче "
            f"{_and_list(map(str, ks))} {months} есть у{nb}{_and_list(f(f'n_months_ge{k}') for k in ks)} МО "
            f"({_and_list(f(f'pop_share_months_ge{k}') for k in ks)} населения выборки); все 12 месяцев "
            f"{ref_year} года — у{nb}{f(f'n_full_{ref_year}')} МО."
        )
    means.append(
        f"Москва и Петербург — {f('n_inner_city')} внутригородских территорий ({f('inner_city_by_region')}): "
        f"{f('inner_city_node_share')} узлов и {f('pop_share_inner_city')} населения выборки. Оставить ли их "
        "отдельными узлами или свернуть в два, решается до этапа 2 (раздел «Регион или место»)."
    )
    means.append(
        f"Контрольные числа этапа panel: жёстких {f('controls_hard_n')}, "
        f"не сошлось {f('controls_hard_failed')}; мягких {f('controls_soft_n')}, "
        f"вне допуска {f('controls_soft_warnings')} (таблица T00)."
    )
    return "**Что видно.** " + " ".join(seen) + "\n\n**Что это значит для сюжета.** " + " ".join(means)
