"""Раздел E5 разведки «Экономика места»: видна ли местная экономика в тратах и где зарабатывают, а где тратят.

Вопросы Q18 и Q19 (spec_final, Б.3 и Б.4):

- **F14** — ρ Спирмена внутри регионов между чертами места (доли занятости, горожане, возраст, плотность,
  доступность рынков, удалённость от столицы региона, ночёвки, зарплата) и чертами трат (уровень, CLR
  корзины, летний избыток, прирост доли маркетплейсов, рост). Связь, которая держится внутри регионов, —
  свойство места, а не карты регионов (Б.1, п. 9). Отдельный столбец — уровень трат при равной зарплате
  (частный ρ): черта, чья связь при этом почти исчезает (доля бюджетников), — та же ось, что зарплата, и в
  заголовке не называется;
- **F15, T12** — безналичные траты к доходу 5-НДФЛ на жителя против расстояния до столицы региона: пригороды,
  где живут, но работают в другом МО, и вахтовые МО, где работают приезжие; столицы — отдельно. Вывод о
  пригородах проверяется на вариантах правила (порог расстояния, только районы и округа), против столицы
  своего региона, при равной доле пожилых и бюджетников внутри региона, по Московской области (расстояние
  по прямой до Москвы) и по линии «траты ~ зарплата» раздела E2; потери 5-НДФЛ из-за дефекта источника
  «единицы вместо тысяч» (флаг пакета P) считаются отдельно;
- **T11** — гипотезы о типах местной экономики: правила ``eda.economy_types`` (строки ``DataFrame.query`` над
  ``EdaData.mo``), медианы черт трат каждого типа против всех МО. Типы пересекаются — это не классификация.

Для «Что видно» раздел находит по данным самые сильные связи корзины с чертами места, столбцы без связей и
черту, у которой связь с уровнем трат сильнее всего меняется при учёте региона (маскировка картой регионов),
и сверяет определение ориентиров Б.4 «без учёта регионов» (разделы ОКВЭД2 A, O, G по всем МО).

Как читать: траты СберИндекса — оценка средних безналичных трат жителей МО (траты приезжих в курортном МО
не видны), а доход 5-НДФЛ и занятость Росстата учитываются по месту работодателя. Поэтому отношение трат к
доходу — не «норма сбережения», а баланс «где живут и где работают»; модель привязки трат к МО СберИндекс
не раскрывает.

Вычисления — чистые функции этого модуля (тестируются отдельно); ``run_section`` только собирает факты,
графики и таблицы через ``SectionContext``. Параметры — ``eda.place`` конфига; пока пакет F не добавил секцию,
действуют значения ``PLACE_DEFAULTS`` (те же, что предложены для конфига).
"""

from __future__ import annotations

import dataclasses
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm, to_rgba
from matplotlib.figure import Figure
from matplotlib.ticker import NullLocator
from scipy.stats import mannwhitneyu, theilslopes

from munnet import style
from munnet.config import Config
from munnet.eda import stats
from munnet.eda.base import Finding, SectionContext, to_markdown
from munnet.eda.data import CONTEXT_YEAR
from munnet.panel.bdmo import TOTAL_DIM
from munnet.panel.context import NDFL_RECIPIENTS, RECIPIENTS_JUMP_FLAG

SECTION_ID = "e5"
TITLE = "Экономика места"

# Параметры раздела: секция ``eda.place`` конфига поверх этих значений (предложены пакету F для конфига).
PLACE_DEFAULTS: dict[str, Any] = {
    "faint_abs_rho": 0.1,  # F14: |ρ| меньше — клетка светлая, связи нет
    "fact_abs_rho": 0.3,  # клетки F14 с |ρ| не меньше — в факты rho_within_*
    "headline_abs_rho": 0.2,  # F14: связь, названная в заголовке и тексте, не слабее
    "headline_skip": ["wage"],  # F14: черты места, которых нет в заголовке (зарплата — тема E2)
    # F14: черта места-контроль и порог частного ρ с уровнем трат при её контроле внутри регионов: черта, чья
    # связь с тратами при равной зарплате почти исчезает (доля бюджетников), — та же ось, что зарплата, и в
    # заголовке как отдельная черта не называется.
    "headline_control": "wage",
    "headline_partial_min": 0.1,
    "exclude_workplace_based": True,  # F14: без МО, где занятость и доход считаются по месту работы
    "types_exclude_workplace_based": True,  # T11: то же для типов, правило которых не про такие МО явно
    "hint_z": 2.0,  # подсказка: черты контекста с робастным |z| больше
    "hint_max": 3,  # не больше стольких черт в подсказке
    "edge_n": 7,  # T12: МО с каждого края отношения «траты / доход 5-НДФЛ»
    "label_high": 4,  # F15: подписей МО с высоким отношением
    "label_low": 3,  # F15: подписей МО с низким отношением
    "examples_n": 3,  # T11: примеров на тип
    "examples_by": "weight",  # T11: примеры — МО типа с наибольшим значением этой колонки mo
    "mw_alpha": 0.01,  # F15: отличие пригородов надёжно при p Манна — Уитни меньше
    "suburb_type": "suburb",  # тип из eda.economy_types, по которому считаются факты suburb_*
    "suburb_sens_km": [25, 50, 100],  # чувствительность: пороги расстояния пригорода вместо порога, км
    "suburb_district_types": ["mr", "mo"],  # вариант «только районы и округа», без городов-спутников
    "check_sections": ["A", "O", "G"],  # разделы ОКВЭД2 для сверки с ориентиром Б.4 «без учёта регионов»
    "check_level": "level_rel_2023",  # уровень трат сверки: к медиане страны, есть и у неполных рядов
    # С6, скорректированное сравнение: ln(траты / доход 5-НДФЛ) очищается от региона (дамми) и этих колонок
    # mo, затем пригороды сравниваются с остальными по остаткам. Доля пожилых — пенсии, которых нет в 5-НДФЛ;
    # доля бюджетников — мало частных работодателей и получателей дохода по месту работы.
    "adjust_controls": ["age_old_share_2023", "emp_sh_public_2023"],
    # С6: регионы без столицы субъекта в справочнике (расстояние до столицы — пропуск) и центр притяжения —
    # регион-город; расстояние — по прямой от центра МО до взвешенного по населению центра города, км.
    "outer_centers": {
        "moscow_obl": {"region": "Московская область", "center": "Москва", "center_gen": "Москвы"},
    },
    "wage_top_n": 30,  # С6, проверка гипотезы E2: МО с наибольшим превышением линии «траты ~ зарплата»
}

# Перцентили распределений T12: ключи фактов spend_to_ndfl_p10, _median, _p90 заданы спецификацией (Б.4).
QUANTILES: tuple[float, ...] = (0.1, 0.5, 0.9)
MEDIAN_Q = 0.5
# Деления осей F15: расстояние, км (на оси — км + 1, чтобы столица с нулём встала на логарифмическую ось),
# и отношение трат к доходу, раз.
DIST_TICKS_KM: tuple[float, ...] = (0, 10, 30, 100, 300, 1000, 3000)
RATIO_TICKS: tuple[float, ...] = (0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50)
LOG_PAD_X, LOG_PAD_Y = 1.3, 1.8  # поля F15 на логарифмических осях (множитель): подписи краёв не обрезаются
# F15: пояснение цветов точек — в самом пустом углу осей (доли осей: отступ, шаг строки, размер угла) и вес
# подписанной точки при выборе угла (рядом с ней стоит подпись).
KEY_MARGIN, KEY_LINE = 0.01, 0.06
KEY_BOX = (0.45, 0.2)
KEY_LABEL_WEIGHT = 50
# F14: клетка слабой связи почти белая, текст на тёмной клетке — белый.
FAINT_FACE = "#F7F7F7"
Z_WHOLE = 10  # в подсказке |z| от 10 — без дробной части: «z +88», а не «z +88,1»
DARK_LUMINANCE = 0.5
CELL_PT = 8
GROUP_COL = "region_code"
YEAR = CONTEXT_YEAR  # год контекста в mo (EdaData.mo: *_2023) и год отношений «траты / доход»
NDFL_RATIO = f"spend_to_ndfl_{YEAR}"
WAGE_RATIO = f"spend_to_wage_{YEAR}"
RECIPIENTS = f"recipients_to_pop_{YEAR}"
DIST = "dist_capital_km"


@dataclass(frozen=True)
class Trait:
    """Черта МО для F14 и подсказок.

    ``code`` — короткий код для ключей фактов (латиница в нижнем регистре), ``column`` — колонка ``mo``,
    ``label`` — подпись на графике, ``where`` — фраза «где …» для заголовка (про высокие значения),
    ``hint_hi`` и ``hint_lo`` — подсказки о необычно высоком и низком значении (пусто — не подсказывать),
    ``transform`` — шкала для робастного z подсказки (``none``, ``log``, ``log1p``). ρ Спирмена от
    монотонного преобразования не зависит, поэтому для F14 оно не нужно.
    """

    code: str
    column: str
    label: str
    where: str = ""
    hint_hi: str = ""
    hint_lo: str = ""
    transform: str = "none"


# Строки F14 — черты места (spec_final, Б.4, E5). Доли занятости — от работников крупных и средних
# организаций по месту работы (Росстат без МСП). Порядок полей: code, column, label, where, hint_hi, hint_lo,
# transform.
# fmt: off
PLACE_TRAITS: tuple[Trait, ...] = (
    Trait("agri", "emp_sh_A_2023", "Занятые в сельском хозяйстве", "больше аграриев",
          "много занятых в сельском хозяйстве"),
    Trait("mining", "emp_sh_B_2023", "Занятые в добыче", "больше занятых в добыче", "много занятых в добыче"),
    Trait("manuf", "emp_sh_C_2023", "Занятые в обработке", "больше занятых в обработке",
          "много занятых в обработке"),
    Trait("trade", "emp_sh_trade_transport_2023", "Занятые в строительстве,\nторговле, транспорте",
          "больше занятых в строительстве, торговле и транспорте"),
    Trait("public", "emp_sh_public_2023", "Занятые в бюджетной сфере", "больше бюджетников",
          "много бюджетников"),
    Trait("urban", "urban_share_2023", "Доля горожан", "больше горожан"),
    Trait("old", "age_old_share_2023", "Старше трудоспособного возраста", "больше пожилых",
          "много пожилых", "мало пожилых"),
    Trait("density", "log_density_2023", "Плотность населения", "плотнее население",
          "плотное население", "редкое население"),
    Trait("access", "market_access", "Доступность рынков", "выше доступность рынков",
          "высокая доступность рынков", "низкая доступность рынков"),
    Trait("dist", DIST, "Расстояние до столицы региона", "дальше от столицы региона",
          "далеко от столицы региона", "", "log1p"),
    Trait("nights", "nights_pc_2023", "Ночёвки в средствах\nразмещения на жителя",
          "больше ночёвок в средствах размещения", "много ночёвок в средствах размещения", "", "log1p"),
    Trait("wage", "wage_2023", "Средняя зарплата", "выше зарплата",
          "высокая зарплата", "низкая зарплата", "log"),
)
# Столбцы F14 — черты трат; доли корзины — в логарифмах (CLR), это сказано в подзаголовке.
SPEND_TRAITS: tuple[Trait, ...] = (
    Trait("level", "log_level_2023", "Уровень\nтрат"),
    Trait("food", "clr_food_2023", "Доля\nпродо-\nвольствия"),
    Trait("cafe", "clr_cafe_2023", "Доля\nобщепита"),
    Trait("mp", "clr_marketplace_2023", "Доля\nмаркет-\nплейсов"),
    Trait("other", "clr_other_2023", "Доля\n«Прочего»"),
    Trait("summer", "summer_excess", "Летний\nизбыток"),
    Trait("mp_pp", "mp_pp_change", "Рост доли\nмаркет-\nплейсов"),
    Trait("growth", "growth", "Рост\nтрат\n(номинал)"),
)
# Черты для подсказок в таблицах: черты места плюс получатели дохода на жителя (вахта и пригороды).
HINT_TRAITS: tuple[Trait, ...] = (
    *PLACE_TRAITS,
    Trait("recipients", RECIPIENTS, "Получатели дохода 5-НДФЛ к жителям", "",
          "много получателей дохода 5-НДФЛ на жителя", "мало получателей дохода 5-НДФЛ на жителя", "log"),
)
# fmt: on
LEVEL_CODE = "level"
# Столбцы F14 для фраз «Что видно» о корзине: код черты трат -> (подлежащее, слово при ρ > 0, при ρ < 0).
LEAD_TEXT: dict[str, tuple[str, str, str]] = {
    "cafe": ("доля общепита в корзине", "выше", "ниже"),
    "food": ("доля продуктов", "выше", "ниже"),
    "mp_pp": ("прирост доли маркетплейсов", "выше", "ниже"),
}
# Столбцы F14, о которых говорится «связаны слабо», если их самая сильная связь слабее порога фактов.
WEAK_TEXT: dict[str, str] = {"summer": "летний избыток", "other": "доля «Прочего»"}

# Подписи типов T11; тип без подписи здесь и в конфиге подписывается своим ключом.
TYPE_LABELS: dict[str, str] = {
    "inner_city": "Внутригородские территории",
    "capital": "Столицы регионов",
    "suburb": "Пригороды столиц регионов",
    "north": "Север",
    "mining": "Добыча",
    "manufacturing": "Обрабатывающая промышленность",
    "agrarian": "Аграрные",
    "public_periphery": "Бюджетная периферия",
    "resort": "Курортные (по ночёвкам Росстата)",
}
# Правило типа T11 словами для отчёта (в CSV — сам код ``query``): тип -> (шаблон правила с «{x}» вместо
# числа, слова с «{x}», вид числа: ``num`` — как есть, ``pct`` — доля в процентах). Правило конфига, которое
# не совпало с шаблоном, показывается кодом. Доли занятых — от работников крупных и средних организаций.
RULE_WORDS: dict[str, tuple[str, str, str]] = {
    "inner_city": ("is_inner_city", "внутригородская территория Москвы или Петербурга", "num"),
    "capital": ("is_capital", "административный центр субъекта по справочнику", "num"),
    "suburb": (
        "dist_capital_km <= {x} and not is_capital and not is_inner_city",
        "до {x} км по дорогам от столицы региона, кроме самой столицы",
        "num",
    ),
    "north": ("point_lat >= {x}", "центр севернее {x}° с. ш.", "num"),
    "mining": ("emp_sh_B_2023 >= {x}", "занятых в добыче не меньше {x}", "pct"),
    "manufacturing": ("emp_sh_C_2023 >= {x}", "занятых в обработке не меньше {x}", "pct"),
    "agrarian": ("emp_sh_A_2023 >= {x}", "занятых в сельском хозяйстве не меньше {x}", "pct"),
    "public_periphery": (
        "emp_sh_public_2023 >= {x}",
        "занятых в бюджетной сфере не меньше {x} работников крупных и средних организаций: такая доля "
        "высока там, где мало частных работодателей; гипотеза о типе, а не отдельная черта места",
        "pct",
    ),
    "resort": (
        "nights_pc_2023 >= {x}",
        "не меньше {x} ночёвок в средствах размещения на жителя за год",
        "num",
    ),
}
ALL_TYPE = "all"
ALL_LABEL = "Все МО панели"
EDGE_LABELS = {"distribution": "распределение", "high": "высокое", "low": "низкое"}
QUANTILE_NAMES = {0.1: "10-й перцентиль", 0.5: "медиана", 0.9: "90-й перцентиль"}


# --- Параметры и текст --------------------------------------------------------------------------


def params(cfg: Config) -> dict[str, Any]:
    """Параметры раздела: ``eda.place`` конфига поверх ``PLACE_DEFAULTS``; неизвестный ключ — ошибка."""
    own = dict(cfg["eda"].get("place") or {})
    unknown = sorted(set(own) - set(PLACE_DEFAULTS))
    if unknown:
        raise KeyError(f"eda.place: неизвестные параметры {unknown}; допустимы {sorted(PLACE_DEFAULTS)}")
    return {**PLACE_DEFAULTS, **own}


# Однобуквенное слово: перед ним не буква, не цифра и не дефис («10-й и» — «й» не предлог).
_SINGLE_LETTER = re.compile(r"(?<![\w-])([А-Яа-яЁё])\s(?=\S)")


def typo(text: str) -> str:
    """Типографика ``ru-text``: неразрывный пробел после однобуквенных слов и перед тире."""
    text = _SINGLE_LETTER.sub(lambda m: m.group(1) + style.NBSP, text)
    return text.replace(" — ", f"{style.NBSP}— ")


def quantile_key(q: float) -> str:
    """Суффикс ключа факта для квантиля: 0,5 → «median», 0,1 → «p10»."""
    return "median" if math.isclose(q, MEDIAN_Q) else f"p{round(100 * q)}"


def _times(x: float) -> str:
    """Отношение для текста: «1,6» (один знак; «в 1,6 раза»)."""
    return style.fmt_num(x, 1)


# --- Типы местной экономики (T11) --------------------------------------------------------------


def select_type(mo: pd.DataFrame, rule: str, name: str = "") -> pd.Series:
    """Маска МО по правилу ``DataFrame.query`` (``eda.economy_types``); пропуск в условии — «не входит».

    Ошибка в правиле (нет колонки, синтаксис) — ``ValueError`` с именем типа и текстом правила.
    """
    try:
        chosen = mo.query(rule, engine="python")
    except Exception as e:  # noqa: BLE001 — любую ошибку правила переводим в понятное сообщение
        raise ValueError(f"тип {name or '?'}: правило {rule!r} не применяется к mo: {e}") from e
    return pd.Series(mo.index.isin(chosen.index), index=mo.index, name=name or "type")


_WORKPLACE_RULE = re.compile(r"\b(is_inner_city|workplace_based)\b")
# Правило-порог «колонка >= число» или «колонка > число»: примеры типа — МО с наибольшим значением колонки.
_THRESHOLD_RULE = re.compile(r"\s*([A-Za-z_]\w*)\s*>=?\s*[-+]?\d+(\.\d+)?\s*")


def rule_column(rule: str, columns: Sequence[str]) -> str | None:
    """Колонка порога правила вида «колонка >= число» (``emp_sh_B_2023 >= 0.2``); иначе None."""
    m = _THRESHOLD_RULE.fullmatch(rule)
    return m.group(1) if m and m.group(1) in set(columns) else None


def _type_label(key: str, spec: Mapping[str, Any]) -> str:
    return str(spec.get("label") or TYPE_LABELS.get(key, key))


def rule_words(key: str, rule: str) -> str | None:
    """Правило типа словами по ``RULE_WORDS``; правило, не совпавшее с шаблоном типа, — None.

    Число шаблона подставляется в слова: «emp_sh_B_2023 >= 0.2» → «занятых в добыче не меньше 20,0%».
    """
    entry = RULE_WORDS.get(key)
    if entry is None:
        return None
    pattern, words, kind = entry
    parts = [re.escape(p) for p in " ".join(pattern.split()).split("{x}")]
    m = re.fullmatch(f"({_NUMBER})".join(parts), " ".join(str(rule).split()))
    if m is None:
        return None
    if "{x}" not in pattern:
        return words
    x = float(m.group(1))
    text = style.fmt_pct(x, 0 if math.isclose(100 * x, round(100 * x)) else 1) if kind == "pct" else _plain(x)
    return words.replace("{x}", text)


def _plain(x: float) -> str:
    """Число правила словами: целое без дробной части, иначе — с нужным числом знаков после запятой."""
    return style.fmt_num(x, 0 if float(x).is_integer() else len(f"{x:g}".split(".")[-1]))


def applied_rule(rule: str, excluded: bool) -> str:
    """Правило, по которому тип отобран на деле: при исключении внутригородских — «(rule) and not …»."""
    return f"({rule}) and not workplace_based" if excluded else rule


def signature_columns(year: int) -> list[str]:
    """Колонки подписи типа T11 (год долей и уровня к региону — ``year``)."""
    return [
        f"level_to_region_{year}",
        f"sh_food_{year}",
        f"sh_other_{year}",
        f"sh_cafe_{year}",
        f"sh_marketplace_{year}",
        "summer_excess",
        WAGE_RATIO,
        NDFL_RATIO,
        "mp_pp_change",
    ]


def targets_workplace_based(rule: str) -> bool:
    """Правило явно про внутригородские территории (``is_inner_city`` или ``workplace_based``)."""
    return bool(_WORKPLACE_RULE.search(rule))


def type_signatures(
    mo: pd.DataFrame,
    types_cfg: Mapping[str, Mapping[str, Any]],
    *,
    year: int,
    examples_n: int,
    examples_by: str = "weight",
    names: pd.Series | None = None,
    exclude_workplace_based: bool = False,
) -> pd.DataFrame:
    """Подписи гипотетических типов местной экономики (T11): строка «все МО» и строка на тип.

    Для каждого типа: правило, число МО и доля от всех, медианы черт трат — уровень к медиане своего региона
    (``level_rel_<year>`` / медиана региона), доли продовольствия, «Прочего», общепита и маркетплейсов года
    ``year``, летний избыток, траты / зарплата (без МО, где зарплата считается по месту работы), траты / доход
    5-НДФЛ, прирост доли маркетплейсов в п. п.; ``examples_n`` примеров — самые выраженные МО типа: колонка
    ``examples_by`` в правиле типа, иначе колонка порога правила вида «колонка >= число»
    (``rule_column``), иначе ``examples_by`` раздела (по умолчанию население); по убыванию.
    ``names`` — короткие названия по ``territory_id`` (иначе ``mo.name``).

    ``exclude_workplace_based`` — МО, где занятость и доход считаются по месту работы (внутригородские
    территории), не входят в типы, правило которых не про них явно (``targets_workplace_based``): иначе
    район Петербурга с заводом или гостиницами попадает в «обрабатывающие» или «курортные» по работникам
    и приезжим, а не по жителям. Колонка ``n_inner`` — сколько таких МО осталось в типе, ``rule_applied`` —
    правило, по которому тип отобран на деле (``mo.query(rule_applied)`` даёт ``n`` МО).
    """
    mo = mo.reset_index(drop=True)
    workplace = mo["workplace_based"].astype(bool).to_numpy()
    rel = mo[f"level_rel_{year}"]
    base = pd.DataFrame(
        {
            f"level_to_region_{year}": rel / rel.groupby(mo[GROUP_COL]).transform("median"),
            **{f"sh_{p}_{year}": mo[f"sh_{p}_{year}"] for p in ("food", "other", "cafe", "marketplace")},
            "summer_excess": mo["summer_excess"],
            WAGE_RATIO: mo[WAGE_RATIO].where(~mo["workplace_based"].astype(bool)),
            NDFL_RATIO: mo[NDFL_RATIO],
            "mp_pp_change": mo["mp_pp_change"],
        }
    )
    cols = signature_columns(year)
    short = mo["name"] if names is None else mo["territory_id"].map(names).fillna(mo["name"])
    rows = [
        {
            "type": ALL_TYPE,
            "label": ALL_LABEL,
            "rule": "",
            "rule_applied": "",
            "n": len(mo),
            "share_mo": 1.0,
            "n_inner": int(workplace.sum()),
            **base[cols].median().to_dict(),
            "examples": "",
        }
    ]
    for key, spec in types_cfg.items():
        rule = str(spec["rule"])
        mask = select_type(mo, rule, key).to_numpy()
        excl = exclude_workplace_based and not targets_workplace_based(rule)
        if excl:
            mask = mask & ~workplace  # не «&=»: to_numpy() у pandas 3 — представление только для чтения
        by = str(spec.get("examples_by") or rule_column(rule, mo.columns) or examples_by)
        top = (
            mo.loc[mask]
            .assign(_order=mo.loc[mask, by])
            .sort_values("_order", ascending=False, kind="mergesort")
        )
        examples = "; ".join(short.loc[top.index[:examples_n]].astype(str))
        rows.append(
            {
                "type": key,
                "label": _type_label(key, spec),
                "rule": rule,
                "rule_applied": applied_rule(rule, excl),
                "n": int(mask.sum()),
                "share_mo": float(mask.mean()) if len(mo) else float("nan"),
                "n_inner": int((mask & workplace).sum()),
                **base.loc[mask, cols].median().to_dict(),
                "examples": examples,
            }
        )
    columns = ["type", "label", "rule", "rule_applied", "n", "share_mo", "n_inner", *cols, "examples"]
    out = pd.DataFrame(rows, columns=columns)
    out["n"] = out["n"].astype("int64")
    return out


# --- Черты места против черт трат (F14) --------------------------------------------------------


def within_region_matrix(
    mo: pd.DataFrame, rows: Sequence[str], cols: Sequence[str], group: str = GROUP_COL
) -> pd.DataFrame:
    """ρ Спирмена каждой пары «строка × столбец» внутри регионов и в целом (длинная таблица).

    Внутри регионов — ранги всей выборки, центрированные по региону (``stats.spearman_within``): чистый
    эффект региона такая связь не видит. Колонки: ``row``, ``col``, ``rho_within``, ``n_within``,
    ``rho_all``, ``n_all``. Пропуски отбрасываются попарно.
    """
    mo = mo.reset_index(drop=True)
    groups = mo[group]
    records = []
    for r in rows:
        x = mo[r].astype("float64")
        for c in cols:
            y = mo[c].astype("float64")
            rho_w, n_w = stats.spearman_within(x, y, groups)
            rho_a, n_a = stats.spearman(x, y)
            records.append((r, c, rho_w, n_w, rho_a, n_a))
    out = pd.DataFrame(records, columns=["row", "col", "rho_within", "n_within", "rho_all", "n_all"])
    out[["n_within", "n_all"]] = out[["n_within", "n_all"]].astype("int64")
    return out


def partial_within(
    mo: pd.DataFrame, rows: Sequence[str], target: str, control: str, group: str = GROUP_COL
) -> pd.DataFrame:
    """Черты ``rows`` против ``target`` при контроле ``control`` внутри регионов.

    Колонки: ``row``; ``rho_partial`` — частный ρ Спирмена черты с ``target`` при контроле ``control``
    (ранги, центрированные по региону, ``stats.partial_spearman``), ``n_partial``; ``rho_control`` — ρ
    черты с ``control`` внутри регионов. Черта, равная контролю, — пропуски. Связь, которая при контроле
    почти исчезает, — та же ось, что контроль (доля бюджетников и зарплата), а не отдельная черта места.
    """
    mo = mo.reset_index(drop=True)
    ctrl = mo[control].astype("float64")
    groups = mo[group]
    records = []
    for r in rows:
        if r == control:
            records.append((r, np.nan, 0, np.nan))
            continue
        x = mo[r].astype("float64")
        rho, n = stats.partial_spearman(
            x, mo[target].astype("float64"), pd.DataFrame({"c": ctrl}), groups=groups
        )
        rho_c, _ = stats.spearman_within(x, ctrl, groups)
        records.append((r, rho, n, rho_c))
    out = pd.DataFrame(records, columns=["row", "rho_partial", "n_partial", "rho_control"])
    out["n_partial"] = out["n_partial"].astype("int64")
    return out


def split_by_partial(cells: pd.DataFrame, min_abs: float) -> tuple[pd.DataFrame, pd.Series | None]:
    """Клетки уровня трат, годные для заголовка F14, и самая сильная из отброшенных.

    ``cells`` — колонки ``code``, ``rho``, ``partial``. Годна клетка, у которой |``partial``| ≥ ``min_abs``
    или частный ρ не посчитан (проверить нечем). Отброшенная клетка с наибольшим |``rho``| — «та же ось,
    что контроль» для текста; нет таких — None.
    """
    partial = cells["partial"].astype("float64")
    keep = partial.isna() | (partial.abs() >= min_abs)
    dropped = cells.loc[~keep].dropna(subset=["rho"])
    axis = None if dropped.empty else dropped.loc[dropped["rho"].abs().idxmax()]
    return cells.loc[keep], axis


@dataclass(frozen=True)
class Headline:
    """Заголовок-вывод с проверкой: ``ok`` — выполнено ли условие ``check`` на текущих числах."""

    text: str
    ok: bool
    check: str
    detail: str


def column_leaders(matrix: pd.DataFrame, cols: Sequence[str], skip: Sequence[str] = ()) -> pd.DataFrame:
    """Самая сильная по |ρ| внутри регионов черта места для каждого столбца F14 из ``cols``.

    ``matrix`` — выход ``within_region_matrix`` с колонками ``row_code``, ``row_where``, ``col_code``;
    черты ``skip`` не участвуют (зарплата — тема раздела E2). Колонки результата: ``col_code``,
    ``row_code``, ``where``, ``rho``, ``max_abs``; столбец без посчитанных ρ — строка с пропусками.
    """
    rows = []
    for col in cols:
        part = matrix.loc[(matrix["col_code"] == col) & ~matrix["row_code"].isin(list(skip))]
        part = part.dropna(subset=["rho_within"])
        if part.empty:
            rows.append({"col_code": col, "row_code": None, "where": None, "rho": np.nan, "max_abs": np.nan})
            continue
        best = part.loc[part["rho_within"].abs().idxmax()]
        rows.append(
            {
                "col_code": col,
                "row_code": best["row_code"],
                "where": best["row_where"],
                "rho": float(best["rho_within"]),
                "max_abs": abs(float(best["rho_within"])),
            }
        )
    return pd.DataFrame(rows, columns=["col_code", "row_code", "where", "rho", "max_abs"])


def masking_cell(cells: pd.DataFrame) -> pd.Series | None:
    """Черта места, у которой связь с уровнем трат сильнее всего меняется при учёте региона.

    ``cells`` — колонки ``code``, ``where``, ``rho_all`` (без учёта регионов), ``rho_within``; выбирается
    строка с наибольшим |ρ внутри − ρ в целом|. Так видно, где карта регионов маскирует свойство места
    (например, доступность рынков: в целом связь обратная, внутри регионов — прямая). Нет пар — None.
    """
    valid = cells.dropna(subset=["rho_all", "rho_within"])
    if valid.empty:
        return None
    return valid.loc[(valid["rho_within"] - valid["rho_all"]).abs().idxmax()]


def section_level_rho(
    mo: pd.DataFrame, okved_shares: pd.DataFrame, sections: Sequence[str], level_col: str, year: int
) -> pd.DataFrame:
    """ρ Спирмена доли работников раздела ОКВЭД2 с уровнем трат без учёта регионов, по всем МО ``mo``.

    Доля раздела — ``okved_shares.share`` года ``year`` (скрытый раздел — пропуск, а не ноль). Это
    определение ориентиров Б.4 «без учёта регионов» (разделы A, O, G по отдельности и все МО, с
    внутригородскими), в отличие от групп разделов и выборки рис. 14. Колонки: ``section``, ``rho``, ``n``.
    """
    shares = okved_shares.loc[okved_shares["year"] == year]
    rows = []
    for sec in sections:
        s = shares.loc[shares["section"] == sec].set_index("territory_id")["share"]
        rho, n = stats.spearman(mo["territory_id"].map(s).astype("float64"), mo[level_col].astype("float64"))
        rows.append({"section": sec, "rho": rho, "n": n})
    return pd.DataFrame(rows, columns=["section", "rho", "n"])


def signature_headline(cells: pd.DataFrame, *, min_abs: float, max_len: int) -> tuple[Headline, dict]:
    """Заголовок F14 по столбцу «уровень трат»: самая сильная прямая и обратная связь с чертой места.

    ``cells`` — колонки ``code``, ``where`` (фраза «где …»), ``rho``. Связь называется, только если
    |ρ| ≥ ``min_abs``. Варианты по порядку, первый не длиннее ``max_len`` знаков: обе связи — «траты выше,
    где …, и ниже, где …» (две формулировки); одна — самая сильная; ни одной — «слабо связан».
    Возвращает заголовок и клетки, прошедшие порог: ``{"pos": строка cells, "neg": строка cells}``
    (без стороны, которая порог не прошла), — они же идут в факты и текст, даже если в заголовок не влезли.
    """
    valid = cells.dropna(subset=["rho"])
    if valid.empty:
        text = typo("Внутри регионов связь уровня трат с экономикой места не посчитана: нет данных")
        return Headline(text, False, "есть хотя бы одна клетка", "все ρ — пропуски"), {}
    pos = valid.loc[valid["rho"].idxmax()]
    neg = valid.loc[valid["rho"].idxmin()]
    pos_ok, neg_ok = bool(pos["rho"] >= min_abs), bool(neg["rho"] <= -min_abs)
    lim = style.fmt_num(min_abs, 1)
    detail = f"ρ({pos['code']}) = {pos['rho']:.3f}, ρ({neg['code']}) = {neg['rho']:.3f}, порог {min_abs}"
    chosen = {k: v for k, v, ok in (("pos", pos, pos_ok), ("neg", neg, neg_ok)) if ok}
    if pos_ok and neg_ok:
        rp, rn = style.fmt_rho(pos["rho"], sign=True), style.fmt_rho(neg["rho"])
        check = f"ρ({pos['code']}) ≥ {lim} и ρ({neg['code']}) ≤ −{lim}"
        for text in (
            f"Внутри регионов траты выше, где {pos['where']} (ρ {rp}), и ниже, где {neg['where']} ({rn})",
            f"Внутри регионов траты выше, где {pos['where']}, ниже — где {neg['where']}: ρ {rp} и {rn}",
            f"Внутри регионов траты выше, где {pos['where']} ({rp}), ниже — где {neg['where']} ({rn})",
            f"Внутри регионов: {pos['where']} — траты выше ({rp}), {neg['where']} — ниже ({rn})",
        ):
            if len(typo(text)) <= max_len:
                return Headline(typo(text), True, check, detail), chosen
    if pos_ok or neg_ok:
        best = pos if (pos_ok and (not neg_ok or pos["rho"] >= -neg["rho"])) else neg
        word = "выше" if best["rho"] > 0 else "ниже"
        text = typo(f"Внутри регионов траты {word}, где {best['where']}: ρ = {style.fmt_rho(best['rho'])}")
        check = f"|ρ({best['code']})| ≥ {lim}, знак {'+' if best['rho'] > 0 else '−'}"
        return Headline(text, abs(best["rho"]) >= min_abs, check, detail), chosen
    top = valid["rho"].abs().max()
    text = typo(
        f"Внутри регионов уровень трат слабо связан с экономикой места: |ρ| не больше {style.fmt_rho(top)}"
    )
    return Headline(text, bool(top < min_abs), f"max |ρ| < {lim}", detail), chosen


def _cell_colors(values: np.ndarray, vmax: float, faint: float) -> np.ndarray:
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cmap = mpl.colormaps[style.DIV_CMAP]
    colors = cmap(norm(np.nan_to_num(values)))
    colors[np.abs(np.nan_to_num(values)) < faint] = to_rgba(FAINT_FACE)
    colors[np.isnan(values)] = to_rgba(style.NODATA)
    return colors


def _luminance(rgba: np.ndarray) -> float:
    r, g, b = rgba[:3]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def plot_signatures(
    pivot: pd.DataFrame,
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    *,
    faint: float,
    vmax: float,
    note: str = "",
) -> Figure:
    """Тепловая карта F14: ρ в клетках, расходящаяся шкала с нулём в центре, слабые связи светлые.

    ``note`` — пояснение под картой (как читать светлые и пустые клетки).
    """
    values = pivot.to_numpy(dtype="float64")
    fig, ax = style.new_figure("tall")
    colors = _cell_colors(values, vmax, faint)
    ax.imshow(colors, aspect="auto", interpolation="nearest")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            v = values[i, j]
            weak = np.isnan(v) or abs(v) < faint
            color = (
                style.TEXT2
                if weak
                else ("white" if _luminance(colors[i, j]) < DARK_LUMINANCE else style.TEXT)
            )
            ax.text(j, i, style.fmt_rho(v), ha="center", va="center", fontsize=CELL_PT, color=color)
    ax.set_xticks(range(values.shape[1]), list(col_labels), fontsize=CELL_PT)
    ax.set_yticks(range(values.shape[0]), list(row_labels))
    ax.xaxis.set_ticks_position("top")
    ax.tick_params(length=0)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    # Уровень трат и он же при контроле — первые два столбца: отделены белой линией от корзины и динамики.
    if values.shape[1] > 2:
        ax.axvline(1.5, color="white", lw=3)
    if note:
        ax.set_xlabel(note, loc="left", fontsize=style.POINT_LABEL_PT, color=style.TEXT2)
    return fig


# --- Где зарабатывают и где тратят (F15, T12) ---------------------------------------------------


def work_home(
    mo: pd.DataFrame,
    *,
    suburb_rule: str,
    recipients_ok: Sequence[float],
    quantiles: Sequence[float] = QUANTILES,
) -> dict[str, Any]:
    """Траты к доходу по месту работы: распределения, пригороды против остальных, края получателей.

    - пригодные МО (``usable``) — ``ndfl_ok`` и известное ``spend_to_ndfl_2023`` (траты 2023 года к доходу
      5-НДФЛ на жителя; у внутригородских и МО с аномальным числом получателей доход на жителя уже пропуск);
    - ``distributions`` — n и квантили ``spend_to_ndfl_2023`` (пригодные МО), ``spend_to_wage_2023`` (без МО,
      где зарплата считается по месту работы) и ``recipients_to_pop_2023`` (все МО с числом получателей);
    - ``n_recip_gt`` и ``n_recip_lt`` — МО, где получателей дохода больше ``recipients_ok[1]`` на жителя
      или меньше ``recipients_ok[0]`` (5-НДФЛ по месту работодателя); ``n_recip_gt_inner`` — сколько из
      первых внутригородских территорий (``is_inner_city``);
    - пригороды — правило ``suburb_rule``; ``suburb_ratio`` — медиана отношения у пригородов / у остальных
      пригодных МО с известным расстоянием до столицы региона (``eligible``: МО без столицы субъекта в
      справочнике, как Московская область, ни пригороды, ни «остальные»; их число — ``n_dist_unknown``),
      ``suburb_mw_p`` — двусторонний тест Манна — Уитни;
    - столицы регионов (``is_capital``) среди пригодных: ``capital_n`` и ``capital_median``.
    """
    mo = mo.reset_index(drop=True)
    ratio = mo[NDFL_RATIO].astype("float64")
    usable = mo["ndfl_ok"].astype(bool) & ratio.notna() & np.isfinite(ratio) & (ratio > 0)
    eligible = usable & mo[DIST].notna()
    capital = mo["is_capital"].astype(bool) & usable
    wage_ratio = mo[WAGE_RATIO].astype("float64").where(~mo["workplace_based"].astype(bool))
    recip = mo[RECIPIENTS].astype("float64")
    samples = {NDFL_RATIO: ratio[usable], WAGE_RATIO: wage_ratio.dropna(), RECIPIENTS: recip.dropna()}
    dist = pd.DataFrame(
        {
            name: {
                "n": float(s.size),
                **{quantile_key(q): s.quantile(q) if s.size else np.nan for q in quantiles},
            }
            for name, s in samples.items()
        }
    ).T
    lo, hi = (float(v) for v in recipients_ok)
    suburb = select_type(mo, suburb_rule, "suburb")
    cmp = compare_groups(ratio, eligible, suburb)
    inner = mo["is_inner_city"].astype(bool) if "is_inner_city" in mo else pd.Series(False, index=mo.index)
    return {
        "usable": usable,
        "eligible": eligible,
        "n_usable": int(usable.sum()),
        "n_dist_unknown": int((usable & ~eligible).sum()),
        "capital": capital,
        "capital_n": int(capital.sum()),
        "capital_median": float(ratio[capital].median()) if capital.any() else float("nan"),
        "distributions": dist,
        "n_recip_gt": int((recip > hi).sum()),
        "n_recip_gt_inner": int(((recip > hi) & inner).sum()),
        "n_recip_lt": int((recip < lo).sum()),
        "recip_bounds": (lo, hi),
        "suburb": suburb,
        "suburb_n": cmp["n"],
        "suburb_median": cmp["median_in"],
        "other_median": cmp["median_out"],
        "suburb_ratio": cmp["ratio"],
        "suburb_mw_p": cmp["p"],
        "other_n": int((eligible & ~suburb).sum()),
    }


def compare_groups(values: pd.Series, usable: pd.Series, group: pd.Series) -> dict[str, float]:
    """Медиана ``values`` в группе против остальных пригодных МО и двусторонний тест Манна — Уитни.

    Возвращает ``n`` (пригодных МО группы), ``median_in``, ``median_out``, ``ratio`` (= ``median_in`` /
    ``median_out``) и ``p``; пустая группа или пустой остаток — пропуски вместо чисел.
    """
    inside = values[(usable & group).to_numpy()]
    outside = values[(usable & ~group).to_numpy()]
    med_in = float(inside.median()) if inside.size else float("nan")
    med_out = float(outside.median()) if outside.size else float("nan")
    p = (
        float(mannwhitneyu(inside, outside, alternative="two-sided").pvalue)
        if inside.size and outside.size
        else float("nan")
    )
    return {
        "n": int(inside.size),
        "median_in": med_in,
        "median_out": med_out,
        "ratio": med_in / med_out if med_out else float("nan"),
        "p": p,
    }


def paired_by_region(
    values: pd.Series, usable: pd.Series, region: pd.Series, capital: pd.Series, group: pd.Series
) -> dict[str, float]:
    """Группа и прочие МО против столицы своего региона: сравнение внутри региона.

    Берутся регионы, где среди пригодных (``usable``) МО есть столица, МО группы и прочие МО; в каждом —
    медиана ``values`` группы, прочих МО и значение столицы. Возвращает ``n_regions``; ``group_to_capital``
    и ``other_to_capital`` — медианы по регионам отношений «группа / столица» и «прочие / столица»;
    ``group_above`` и ``other_above`` — в скольких регионах группа (прочие) выше столицы; ``group_to_other`` —
    медиана по регионам отношения «группа / прочие». Если группа выше столицы, а прочие — так же, это разрыв
    «столица — все остальные», а не свойство группы.
    """
    frame = pd.DataFrame(
        {
            "v": values.to_numpy(dtype="float64"),
            "r": region.to_numpy(),
            "g": np.where(capital.to_numpy(bool), "cap", np.where(group.to_numpy(bool), "grp", "oth")),
        }
    ).loc[usable.to_numpy(bool)]
    med = frame.groupby(["r", "g"])["v"].median().unstack().reindex(columns=["cap", "grp", "oth"])
    full = med.dropna()
    nan = float("nan")
    if full.empty:
        keys = ("group_to_capital", "other_to_capital", "group_to_other")
        return {"n_regions": 0, "group_above": 0, "other_above": 0, **dict.fromkeys(keys, nan)}
    return {
        "n_regions": len(full),
        "group_to_capital": float((full["grp"] / full["cap"]).median()),
        "other_to_capital": float((full["oth"] / full["cap"]).median()),
        "group_to_other": float((full["grp"] / full["oth"]).median()),
        "group_above": int((full["grp"] > full["cap"]).sum()),
        "other_above": int((full["oth"] > full["cap"]).sum()),
    }


def region_residuals(y: np.ndarray, region: np.ndarray, controls: np.ndarray) -> np.ndarray:
    """Остатки МНК ``y`` на дамми регионов и колонки ``controls`` (n × k).

    По теореме Фриша — Во — Ловелла: ``y`` и контроли центрируются по региону, остатки — от регрессии
    центрированного ``y`` на центрированные контроли. У региона из одного МО остаток — ноль.
    """
    codes = pd.factorize(pd.Series(region))[0]

    def center(a: np.ndarray) -> np.ndarray:
        frame = pd.DataFrame(a)
        return (frame - frame.groupby(codes).transform("mean")).to_numpy(dtype="float64")

    yc = center(np.asarray(y, dtype="float64").reshape(-1, 1))[:, 0]
    xc = center(np.asarray(controls, dtype="float64").reshape(len(yc), -1))
    if xc.shape[1] == 0:
        return yc
    beta, *_ = np.linalg.lstsq(xc, yc, rcond=None)
    return yc - xc @ beta


def adjusted_comparison(
    mo: pd.DataFrame,
    values: pd.Series,
    eligible: pd.Series,
    group: pd.Series,
    controls: Sequence[str],
    region: str = GROUP_COL,
) -> dict[str, float]:
    """Группа против остальных МО при равных контролях внутри региона.

    ln ``values`` пригодных МО (``eligible``, значение > 0, все ``controls`` известны) очищается от региона
    (дамми) и ``controls`` (``region_residuals``); затем ``compare_groups`` по exp(остатка): ``ratio`` — во
    сколько раз медиана группы выше медианы остальных при равных контролях, ``p`` — Манн — Уитни, ``n`` —
    МО группы, ``n_all`` — все МО сравнения.
    """
    mo = mo.reset_index(drop=True)
    v = pd.Series(values, dtype="float64").reset_index(drop=True)
    ok = pd.Series(eligible, dtype=bool).reset_index(drop=True) & (v > 0)
    for c in controls:
        ok &= mo[c].notna()
    idx = ok.to_numpy()
    if idx.sum() == 0:
        return {"n": 0, "n_all": 0, "ratio": float("nan"), "p": float("nan")}
    resid = region_residuals(
        np.log(v[idx].to_numpy()), mo.loc[idx, region].to_numpy(), mo.loc[idx, list(controls)].to_numpy()
    )
    grp = pd.Series(group, dtype=bool).reset_index(drop=True)[idx].reset_index(drop=True)
    cmp = compare_groups(pd.Series(np.exp(resid)), pd.Series(True, index=grp.index), grp)
    return {"n": cmp["n"], "n_all": int(idx.sum()), "ratio": cmp["ratio"], "p": cmp["p"]}


def outer_center_check(
    mo: pd.DataFrame, usable: pd.Series, region: str, center: str, near_km: float
) -> dict[str, float]:
    """Регион без столицы субъекта в справочнике: отношение трат к доходу против расстояния до города-центра.

    Центр — средний по населению (``pop_2023``) центр МО региона ``center`` в проекции Альберса
    (``x_aea``, ``y_aea``); расстояние — по прямой, км. Возвращает ``n`` пригодных МО региона ``region``,
    ``n_near`` (не дальше ``near_km``), медианы ``near`` и ``far`` и ``rho`` Спирмена отношения с расстоянием.
    """
    mo = mo.reset_index(drop=True)
    nan = float("nan")
    part = mo.loc[(mo["region_name"] == region).to_numpy() & usable.to_numpy(bool)]
    city = mo.loc[(mo["region_name"] == center) & mo["x_aea"].notna() & mo["y_aea"].notna()]
    if part.empty or city.empty:
        return {"n": len(part), "n_near": 0, "near": nan, "far": nan, "rho": nan}
    w = city["pop_2023"].fillna(0).to_numpy(dtype="float64") if "pop_2023" in city else None
    w = w if w is not None and w.sum() > 0 else None
    cx, cy = np.average(city["x_aea"], weights=w), np.average(city["y_aea"], weights=w)
    km = np.hypot(part["x_aea"] - cx, part["y_aea"] - cy) / 1000
    ratio = part[NDFL_RATIO].astype("float64")
    near = (km <= near_km).to_numpy()
    rho, _ = stats.spearman(ratio, km)
    return {
        "n": len(part),
        "n_near": int(near.sum()),
        "near": float(ratio[near].median()) if near.any() else nan,
        "far": float(ratio[~near].median()) if (~near).any() else nan,
        "rho": rho,
    }


def wage_line_check(mo: pd.DataFrame, sample: pd.Series, group: pd.Series, top_n: int) -> dict[str, float]:
    """Гипотеза раздела E2: группа (пригороды) тратит больше, чем «положено» по зарплате по месту работы.

    Линия ln уровня трат 2023 года на ln зарплаты 2023 года — Тейл — Сен по МО без внутригородских
    территорий, свободный член — медиана остатков (как у E2). Остаток ln(уровень / линия) сравнивается у
    группы и остальных МО ``sample`` (``compare_groups`` по exp остатка): ``ratio``, ``p``, ``n``. ``top_k`` —
    сколько МО группы среди ``top_n`` МО ``sample`` с наибольшим остатком, ``base_share`` — доля группы в
    ``sample`` (для сравнения).
    """
    mo = mo.reset_index(drop=True)
    wage, level = mo["wage_2023"].astype("float64"), mo["level_2023"].astype("float64")
    base = (~mo["workplace_based"].astype(bool) & (wage > 0) & (level > 0)).to_numpy()
    nan = float("nan")
    if base.sum() < 2:
        return {"ratio": nan, "p": nan, "n": 0, "top_k": 0, "top_n": 0, "base_share": nan}
    x, y = np.log(wage[base].to_numpy()), np.log(level[base].to_numpy())
    slope = float(theilslopes(y, x)[0])
    intercept = float(np.median(y - slope * x))
    resid = pd.Series(np.nan, index=mo.index)
    resid[base] = y - intercept - slope * x
    ok = pd.Series(sample, dtype=bool).reset_index(drop=True) & resid.notna()
    grp = pd.Series(group, dtype=bool).reset_index(drop=True)
    cmp = compare_groups(np.exp(resid), ok, grp)
    top = resid[ok].nlargest(top_n).index
    return {
        "ratio": cmp["ratio"],
        "p": cmp["p"],
        "n": cmp["n"],
        "top_k": int(grp[top].sum()),
        "top_n": len(top),
        "base_share": float(grp[ok].mean()) if ok.any() else nan,
    }


# Порог правила «колонка <= число» или «колонка >= число» внутри составного правила.
_NUMBER = r"[-+]?\d+(?:\.\d+)?"


def rule_threshold(rule: str, column: str) -> float | None:
    """Число из условия «``column`` <= N» (или ``<``, ``>=``, ``>``) правила ``query``; нет условия — None."""
    m = re.search(rf"\b{re.escape(column)}\s*[<>]=?\s*({_NUMBER})", rule)
    return float(m.group(1)) if m else None


def suburb_variants(
    rule: str, column: str, thresholds: Sequence[float], district_types: Sequence[str]
) -> list[tuple[str, str]]:
    """Варианты правила пригорода для проверки чувствительности: пары (подпись, правило ``query``).

    Порог условия по ``column`` заменяется на каждое значение ``thresholds`` (если в правиле такого условия
    нет — остаётся только исходное правило); каждый вариант берётся для всех МО и только для типов
    ``district_types`` (районы и округа — без городских округов-спутников со своими работодателями).
    """
    base = rule_threshold(rule, column)
    pattern = re.compile(rf"(\b{re.escape(column)}\s*[<>]=?\s*){_NUMBER}")
    if base is None:
        by_km = [(rule, None)]
    else:
        by_km = [(pattern.sub(lambda m, t=t: f"{m.group(1)}{t:g}", rule, count=1), t) for t in thresholds]
    variants = []
    types = list(district_types)
    for r, km in by_km:
        label = "правило конфига" if km is None else f"до {style.fmt_num(km)} км"
        variants.append((label, r))
        if types:
            variants.append((f"{label}, только районы и округа", f"({r}) and mo_type in {types!r}"))
    return variants


def suburb_sensitivity(
    mo: pd.DataFrame, values: pd.Series, usable: pd.Series, variants: Sequence[tuple[str, str]]
) -> pd.DataFrame:
    """Пригороды против остальных пригодных МО при каждом варианте правила (``compare_groups``).

    Колонки: ``variant``, ``rule``, ``n``, ``median_in``, ``median_out``, ``ratio``, ``p``.
    """
    mo = mo.reset_index(drop=True)
    values = values.reset_index(drop=True)
    usable = usable.reset_index(drop=True)
    rows = [
        {"variant": label, "rule": r, **compare_groups(values, usable, select_type(mo, r, label))}
        for label, r in variants
    ]
    return pd.DataFrame(rows, columns=["variant", "rule", "n", "median_in", "median_out", "ratio", "p"])


def recipient_defects(
    context_long: pd.DataFrame, context_annual: pd.DataFrame, year: int, low: float
) -> dict[str, int]:
    """Сколько МО панели потеряли число получателей 5-НДФЛ из-за дефекта источника «единицы вместо тысяч».

    Пакет P помечает такие значения флагом ``recipients_jump`` и не берёт их в ``context_annual``, поэтому у
    этих МО ``recipients_to_pop`` пуст и ``ndfl_ok`` ложен. ``n_jump`` — МО года ``year`` с таким флагом;
    ``n_lt_raw`` — МО, у которых получателей меньше ``low`` на жителя до снятия дефекта (сырые значения
    ``context_long`` к ``pop_avg``): так считался ориентир Л3.
    """
    rows = context_long.loc[
        (context_long["indicator"] == NDFL_RECIPIENTS)
        & (context_long["year"] == year)
        & (context_long["dim"] == TOTAL_DIM)
    ].drop_duplicates("territory_id")
    flags = rows["flag"].astype("string").fillna("")
    jump = flags.str.split(";").map(lambda parts: RECIPIENTS_JUMP_FLAG in parts)
    pop = context_annual.loc[context_annual["year"] == year].set_index("territory_id")["pop_avg"]
    raw = rows.set_index("territory_id")["value"] / pop.reindex(rows["territory_id"]).to_numpy()
    return {"n_jump": int(jump.sum()), "n_lt_raw": int((raw < low).sum())}


def transform_values(x: pd.Series, transform: str) -> pd.Series:
    """Шкала черты для робастного z: ``none``; ``log`` (≤ 0 → пропуск); ``log1p`` (< 0 → пропуск)."""
    x = pd.Series(x, dtype="float64")
    if transform == "none":
        return x
    if transform == "log":
        return np.log(x.where(x > 0))
    if transform == "log1p":
        return np.log1p(x.where(x >= 0))
    raise ValueError(f"неизвестная шкала {transform!r}: допустимы none, log, log1p")


def context_hints(
    mo: pd.DataFrame,
    traits: Sequence[Trait],
    *,
    z_thr: float,
    max_hints: int,
    annotations: Mapping[int, str] | None = None,
) -> pd.Series:
    """Подсказка к МО (Б.1, п. 8): пояснение из ``eda.annotations`` и черты контекста с |z| > ``z_thr``.

    z — робастный (медиана и MAD), считается по всем МО с известным значением на шкале
    ``Trait.transform``; подсказываются только стороны, у которых есть текст (``hint_hi``, ``hint_lo``);
    не больше ``max_hints`` черт, самые сильные первыми. Индекс результата — ``territory_id``. Это признаки,
    а не проверенные причины.
    """
    mo = mo.reset_index(drop=True)
    parts: list[list[tuple[float, str]]] = [[] for _ in range(len(mo))]
    for t in traits:
        z = stats.robust_z(transform_values(mo[t.column], t.transform))
        for i, zi in enumerate(z.to_numpy()):
            if np.isnan(zi) or abs(zi) <= z_thr:
                continue
            text = t.hint_hi if zi > 0 else t.hint_lo
            if text:
                digits = 0 if abs(zi) >= Z_WHOLE else 1
                parts[i].append((abs(zi), f"{text} (z {style.fmt_num(zi, digits, sign=True)})"))
    notes = dict(annotations or {})
    hints = []
    for tid, found in zip(mo["territory_id"], parts, strict=True):
        found.sort(key=lambda item: -item[0])
        items = [text for _, text in found[:max_hints]]
        note = notes.get(int(tid))
        hints.append("; ".join(([str(note)] if note else []) + items))
    return pd.Series(hints, index=mo["territory_id"].astype("int32").to_numpy(), name="hint")


def ndfl_edges(
    mo: pd.DataFrame, usable: pd.Series, n: int, hints: pd.Series, names: pd.Series | None = None
) -> pd.DataFrame:
    """По ``n`` пригодных МО с самым высоким и самым низким отношением трат к доходу 5-НДФЛ (T12)."""
    mo = mo.reset_index(drop=True)
    part = mo.loc[usable.to_numpy()]
    cols = ["territory_id", "name", "region_name", NDFL_RATIO, WAGE_RATIO, RECIPIENTS, DIST]
    # Равные отношения — по territory_id: порядок и состав краёв воспроизводимы.
    keys = [NDFL_RATIO, "territory_id"]
    high = part.sort_values(keys, ascending=[False, True]).head(n).assign(block="high")
    low = part.sort_values(keys, ascending=[True, True]).head(n).assign(block="low")
    out = pd.concat([high, low], ignore_index=True)[["block", *cols]]
    if names is not None:
        out["name"] = out["territory_id"].map(names).fillna(out["name"])
    wp = mo.set_index("territory_id")["workplace_based"].astype(bool)
    out[WAGE_RATIO] = out[WAGE_RATIO].where(~out["territory_id"].map(wp).fillna(False).astype(bool))
    out["hint"] = out["territory_id"].map(hints).fillna("")
    return out


def work_home_table(
    distributions: pd.DataFrame, edges: pd.DataFrame, quantiles: Sequence[float]
) -> pd.DataFrame:
    """T12: строки распределений (квантили трёх отношений), затем края отношения трат к доходу."""
    rows = []
    for q in quantiles:
        key = quantile_key(q)
        rows.append(
            {
                "block": "distribution",
                "territory_id": pd.NA,
                "name": QUANTILE_NAMES.get(q, key),
                "region_name": "",
                NDFL_RATIO: distributions.loc[NDFL_RATIO, key],
                WAGE_RATIO: distributions.loc[WAGE_RATIO, key],
                RECIPIENTS: distributions.loc[RECIPIENTS, key],
                DIST: np.nan,
                "hint": "",
            }
        )
    out = pd.concat([pd.DataFrame(rows), edges[list(rows[0])]], ignore_index=True)
    out["territory_id"] = out["territory_id"].astype("Int32")
    return out


def _log_ticks(ax, axis: str, ticks: Sequence[float], labels: Sequence[str], lo: float, hi: float) -> None:
    keep = [(t, lab) for t, lab in zip(ticks, labels, strict=True) if lo <= t <= hi]
    getattr(ax, f"set_{axis}ticks")([t for t, _ in keep], [lab for _, lab in keep])
    getattr(ax, f"{axis}axis").set_minor_locator(NullLocator())


def _axes_fraction(
    x: pd.Series, y: pd.Series, x_lim: tuple[float, float], y_lim: tuple[float, float]
) -> np.ndarray:
    """Координаты точек в долях осей при логарифмических шкалах ``x_lim``, ``y_lim``: массив n × 2."""
    lx, ly = np.log(x_lim), np.log(y_lim)
    fx = (np.log(np.asarray(x, dtype="float64")) - lx[0]) / (lx[1] - lx[0])
    fy = (np.log(np.asarray(y, dtype="float64")) - ly[0]) / (ly[1] - ly[0])
    return np.column_stack([fx, fy])


def emptiest_corner(
    points: np.ndarray, labeled: np.ndarray, box: tuple[float, float] = KEY_BOX
) -> tuple[str, str]:
    """Угол осей для пояснения цветов точек, где меньше всего точек: (``left``/``right``, ``top``/``bottom``).

    ``points`` и ``labeled`` — координаты в долях осей (n × 2); подписанная точка весит ``KEY_LABEL_WEIGHT``
    обычных, потому что рядом с ней стоит подпись. ``box`` — ширина и высота угла в долях осей. При равенстве
    — порядок: левый верхний, правый верхний, левый нижний, правый нижний.
    """
    w, h = box
    best, best_score = ("left", "top"), float("inf")
    for level in ("top", "bottom"):
        for side in ("left", "right"):

            def inside(a: np.ndarray, side=side, level=level) -> int:
                if a.size == 0:
                    return 0
                fx = a[:, 0] if side == "left" else 1 - a[:, 0]
                fy = 1 - a[:, 1] if level == "top" else a[:, 1]
                return int(((fx <= w) & (fy <= h)).sum())

            score = inside(points) + KEY_LABEL_WEIGHT * inside(labeled)
            if score < best_score:
                best, best_score = (side, level), score
    return best


def plot_work_home(
    points: pd.DataFrame,
    labeled: pd.DataFrame,
    median: float,
    suburb_note: str = "пригороды столиц регионов",
    capital_note: str = "столицы регионов",
) -> Figure:
    """F15: траты к доходу 5-НДФЛ против расстояния до столицы региона, обе шкалы логарифмические.

    ``points`` — колонки ``x`` (км + 1), ``y`` (отношение), ``is_suburb``, ``is_capital``; ``labeled`` —
    ``x``, ``y``, ``label``, ``is_suburb`` для подписей краёв; ``median`` — горизонтальная линия медианы всех
    пригодных МО; ``suburb_note`` и ``capital_note`` — что такое чёрные точки и ромбы (подписи в углу).
    Столицы — пустые ромбы у нуля: их отношение задаёт центр занятости, а не пригород. Подписанные МО не из
    пригородов — серые кольца, чтобы их нельзя было спутать с пригородами.
    """
    fig, ax = style.new_figure("full")
    cap_mask = points["is_capital"].astype(bool) if "is_capital" in points else pd.Series(False, points.index)
    other = points.loc[~points["is_suburb"] & ~cap_mask]
    sub = points.loc[points["is_suburb"] & ~cap_mask]
    caps = points.loc[cap_mask]
    ax.scatter(other["x"], other["y"], s=9, color=style.CONTEXT, linewidths=0, zorder=2)
    ax.scatter(sub["x"], sub["y"], s=12, color=style.ACCENT, linewidths=0, zorder=2)
    ax.scatter(
        caps["x"],
        caps["y"],
        s=26,
        marker="D",
        facecolors="white",
        edgecolors=style.ACCENT,
        linewidths=0.9,
        zorder=2.5,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.axhline(median, color=style.TEXT2, lw=0.8, ls="--", zorder=1)
    x_lim = (points["x"].min() / LOG_PAD_X, points["x"].max() * LOG_PAD_X)
    y_lim = (points["y"].min() / LOG_PAD_Y, points["y"].max() * LOG_PAD_Y)
    ax.set_xlim(*x_lim)
    ax.set_ylim(*y_lim)
    _log_ticks(ax, "x", [k + 1 for k in DIST_TICKS_KM], [style.fmt_num(k) for k in DIST_TICKS_KM], *x_lim)
    ratio_labels = [style.fmt_num(v, style.auto_decimals([v])) for v in RATIO_TICKS]
    _log_ticks(ax, "y", RATIO_TICKS, ratio_labels, *y_lim)
    ax.text(
        0.99,
        median,
        typo(f" медиана {style.fmt_num(median, 2)}"),
        transform=ax.get_yaxis_transform(),
        ha="right",
        va="bottom",
        fontsize=style.POINT_LABEL_PT,
        color=style.TEXT2,
    )
    frac = _axes_fraction(points["x"], points["y"], x_lim, y_lim)
    lab = _axes_fraction(labeled["x"], labeled["y"], x_lim, y_lim)
    side, level = emptiest_corner(frac, lab)
    x0, ha = (KEY_MARGIN, "left") if side == "left" else (1 - KEY_MARGIN, "right")
    lines = (
        (typo(f"чёрные точки — {suburb_note}"), style.ACCENT),
        (typo(f"ромбы — {capital_note}"), style.ACCENT),
        ("серые — остальные МО", style.TEXT2),
    )
    if caps.empty:
        lines = (lines[0], lines[2])
    for i, (text, color) in enumerate(lines):
        y0 = 1 - KEY_MARGIN - i * KEY_LINE if level == "top" else KEY_MARGIN + (len(lines) - 1 - i) * KEY_LINE
        ax.text(
            x0,
            y0,
            text,
            transform=ax.transAxes,
            ha=ha,
            va="top" if level == "top" else "bottom",
            fontsize=style.POINT_LABEL_PT,
            color=color,
        )
    if len(labeled):
        # Один вызов — подписи раздвигаются между собой; затем подписанные МО не из пригородов
        # перерисовываются серыми кольцами: чёрная заливка остаётся только у пригородов.
        style.label_points(ax, labeled["x"], labeled["y"], labeled["label"], max_labels=len(labeled))
        rest = labeled.loc[~labeled["is_suburb"].astype(bool)]
        ax.scatter(rest["x"], rest["y"], s=22, color="white", linewidths=0, zorder=3.4)
        ax.scatter(
            rest["x"],
            rest["y"],
            s=22,
            facecolors=style.CONTEXT,
            edgecolors=style.TEXT2,
            linewidths=1,
            zorder=3.5,
        )
    ax.set_xlabel(typo("Расстояние по дорогам до столицы региона, км; 0 — МО с центром в столице региона"))
    ax.set_ylabel("Траты / доход 5-НДФЛ на жителя, раз")
    return fig


# --- Сборка раздела ------------------------------------------------------------------------------


def _num0(v: Any) -> str:
    return style.fmt_num(v, 0)


def _num2(v: Any) -> str:
    return style.fmt_num(v, 2)


def _pct(v: Any) -> str:
    return style.fmt_pct(v, 1)


def _pct_signed(v: Any) -> str:
    return style.fmt_pct(v, 1, sign=True)


def _pp(v: Any) -> str:
    return style.fmt_pp(v, 1)


def _id(v: Any) -> str:
    return style.NA_TEXT if pd.isna(v) else str(int(v))


def _code(v: Any) -> str:
    """Правило ``query`` в Markdown — как код: точка в «0.2» не считается ошибкой типографики."""
    return f"`{v}`" if v else ""


def _annotations(cfg: Config) -> dict[int, str]:
    raw = cfg["eda"].get("annotations") or {}
    return {int(k): str(v) for k, v in raw.items()}


# Короткое название без слова (у муниципальных округов Петербурга — «№ 78») дополняется до «МО № 78».
_HAS_WORD = re.compile(r"[А-Яа-яЁёA-Za-z]{2,}")


def short_names(territories: pd.DataFrame) -> pd.Series:
    """Короткие названия МО для подписей по ``territory_id``: ``name_short``, пропуск — ``name``;
    название без слова (только номер) — «МО № 78»."""
    names = territories.set_index("territory_id")
    full = names["name"].astype("string")
    short = (names["name_short"] if "name_short" in names else names["name"]).astype("string").fillna(full)
    bare = ~short.str.contains(_HAS_WORD, na=False)
    return short.mask(bare, "МО " + short)


# Существительное к короткому названию-прилагательному района или округа (как в таблицах разделов E2 и E3).
TYPE_NOUNS: dict[str, str] = {"mr": "район", "mo": "округ"}
ADJECTIVE_ENDINGS: tuple[str, ...] = ("ий", "ый", "ой")


def display_names(territories: pd.DataFrame) -> pd.Series:
    """Названия МО для текста и таблиц по ``territory_id``: короткое (``short_names``), а одно
    слово-прилагательное района или округа — с типом: «Карагинский» (mr) → «Карагинский район»,
    «Тенькинский» (mo) → «Тенькинский округ». Так «Ханты-Мансийский район» не спутать с городом, а названия
    совпадают с таблицами разделов E2 и E3."""
    short = short_names(territories)
    types = territories.set_index("territory_id")["mo_type"].astype("string").reindex(short.index)
    out = [
        f"{name} {TYPE_NOUNS[kind]}"
        if kind in TYPE_NOUNS and " " not in str(name) and str(name).endswith(ADJECTIVE_ENDINGS)
        else str(name)
        for name, kind in zip(short, types.fillna(""), strict=True)
    ]
    return pd.Series(out, index=short.index, dtype="object")


def one_line(label: str) -> str:
    """Подпись графика в одну строку: перенос с дефисом склеивается («маркет-\\nплейсов»), прочий — пробел."""
    return label.replace("-\n", "").replace("\n", " ")


def _trait_frame(traits: Sequence[Trait], prefix: str) -> pd.DataFrame:
    return pd.DataFrame(
        [(t.column, t.code, one_line(t.label), t.where) for t in traits],
        columns=[prefix, f"{prefix}_code", f"{prefix}_label", f"{prefix}_where"],
    )


def _first_fitting(variants: Sequence[str], limit: int) -> str:
    """Первый вариант текста не длиннее ``limit`` знаков (после ``typo``); иначе — последний."""
    texts = [typo(v) for v in variants]
    return next((t for t in texts if len(t) <= limit), texts[-1])


# Подпись столбца F14 «уровень трат при контроле»: код черты-контроля -> подпись; фраза «при равной …» и
# «та же ось, что …» для текста.
CONTROL_TEXT: dict[str, tuple[str, str, str]] = {
    "wage": ("Уровень\nпри\nравной\nзарплате", "при равной зарплате", "зарплата"),
}


def _control_text(code: str, label: str) -> tuple[str, str, str]:
    return CONTROL_TEXT.get(code, (f"Уровень\nпри контроле\n{code}", f"при контроле черты «{label}»", label))


def _signatures(ctx: SectionContext, p: dict[str, Any]) -> dict[str, Any]:
    """F14 и факты о связях черт места и трат внутри регионов."""
    mo = ctx.data.mo
    sample_mo = mo.loc[~mo["workplace_based"].astype(bool)] if p["exclude_workplace_based"] else mo
    matrix = within_region_matrix(
        sample_mo, [t.column for t in PLACE_TRAITS], [t.column for t in SPEND_TRAITS]
    )
    matrix = matrix.merge(_trait_frame(PLACE_TRAITS, "row"), on="row").merge(
        _trait_frame(SPEND_TRAITS, "col"), on="col"
    )
    strong_min, faint = float(p["fact_abs_rho"]), float(p["faint_abs_rho"])
    by_code = {t.code: t for t in PLACE_TRAITS}
    if p["headline_control"] not in by_code:
        raise KeyError(f"eda.place.headline_control: нет черты места {p['headline_control']!r}")
    control = by_code[str(p["headline_control"])]
    control_label, control_equal, control_axis = _control_text(control.code, one_line(control.label))
    level_col = next(t.column for t in SPEND_TRAITS if t.code == LEVEL_CODE)
    partial = partial_within(sample_mo, [t.column for t in PLACE_TRAITS], level_col, control.column).merge(
        _trait_frame(PLACE_TRAITS, "row"), on="row"
    )
    # Доли занятости — группы разделов context.okved_groups (у «бюджетников» — O, P, Q; у торговли — F, G, H):
    # с ориентирами Б.4 «без учёта регионов» по отдельным разделам A, O, G и всем МО сравнивать с поправкой.
    who = "МО без внутригородских территорий" if p["exclude_workplace_based"] else "все МО"
    sample = f"{who}; доли занятости — группы разделов ОКВЭД2"
    for r in matrix.itertuples(index=False):
        note = f"ρ Спирмена {{}}: {r.row_label} ~ {r.col_label}; {sample}; n = {{}}"
        if r.col_code == LEVEL_CODE or abs(r.rho_within) >= strong_min:
            ctx.fact(
                f"rho_within_{r.row_code}_{r.col_code}",
                r.rho_within,
                "rho",
                note.format("внутри регионов", r.n_within),
            )
        if r.col_code == LEVEL_CODE:
            ctx.fact(
                f"rho_all_{r.row_code}_{r.col_code}",
                r.rho_all,
                "rho",
                note.format("без учёта регионов", r.n_all),
            )
    for r in partial.itertuples(index=False):
        if r.row_code == control.code:
            continue
        ctx.fact(
            f"partial_{r.row_code}_level_{control.code}",
            r.rho_partial,
            "rho",
            f"частный ρ Спирмена внутри регионов: {r.row_label} ~ уровень трат 2023 года {control_equal} "
            f"(контроль — {control.column}); {sample}; n = {r.n_partial}",
        )
    strong = int((matrix["rho_within"].abs() >= strong_min).sum())
    ctx.fact("n_cells", len(matrix), "int", "клеток F14: черты места × черты трат")
    ctx.fact(
        "n_cells_strong", strong, "int", f"клеток F14 с |ρ| внутри регионов ≥ {style.fmt_num(strong_min, 1)}"
    )
    ctx.fact("strong_rho_min", strong_min, "num1", "порог «сильной» связи для фактов rho_within_*")
    n_sig = int(matrix["n_within"].max()) if len(matrix) else 0
    ctx.fact("n_signature", n_sig, "int", "наибольшее число МО в клетке F14")

    level = matrix.loc[(matrix["col_code"] == LEVEL_CODE) & ~matrix["row_code"].isin(p["headline_skip"])]
    part_cols = partial[["row_code", "rho_partial", "rho_control"]].rename(
        columns={"row_code": "code", "rho_partial": "partial"}
    )
    cells = level.rename(columns={"row_code": "code", "row_where": "where", "rho_within": "rho"}).merge(
        part_cols, on="code", how="left"
    )
    partial_min = float(p["headline_partial_min"])
    eligible, axis = split_by_partial(cells, partial_min)
    head, chosen = signature_headline(eligible, min_abs=float(p["headline_abs_rho"]), max_len=ctx.title_max)
    for side in ("pos", "neg"):
        cell = chosen.get(side)
        which = "прямой" if side == "pos" else "обратной"
        ctx.fact(
            f"sig_{side}_where",
            None if cell is None else cell["where"],
            "str",
            f"черта места с самой сильной {which} связью с уровнем трат среди черт, чья связь держится "
            f"{control_equal} (|частный ρ| ≥ {style.fmt_num(partial_min, 1)})",
        )
        ctx.fact(
            f"sig_{side}_rho",
            np.nan if cell is None else cell["rho"],
            "rho",
            "ρ Спирмена внутри регионов с уровнем трат 2023 года",
        )
        ctx.fact(
            f"sig_{side}_partial",
            np.nan if cell is None else cell["partial"],
            "rho",
            f"частный ρ Спирмена той же черты с уровнем трат {control_equal}, внутри регионов",
        )
    ctx.fact(
        "axis_where",
        None if axis is None else axis["where"],
        "str",
        f"черта места с самой сильной связью с уровнем трат, которая {control_equal} почти исчезает "
        f"(|частный ρ| < {style.fmt_num(partial_min, 1)}): та же ось, что {control.column}",
    )
    for key, col, what in (
        ("axis_rho", "rho", "ρ Спирмена внутри регионов с уровнем трат 2023 года"),
        ("axis_partial", "partial", f"частный ρ с уровнем трат {control_equal}"),
        ("axis_rho_control", "rho_control", f"ρ Спирмена внутри регионов с {control.column}"),
    ):
        ctx.fact(key, np.nan if axis is None else axis[col], "rho", f"{what}; черта axis_where")
    check = f"{head.check}; названные черты — |частный ρ {control_equal}| ≥ {style.fmt_num(partial_min, 1)}"
    title = ctx.headline(head.text, head.ok, f"F14 economy_signatures: {check}; {head.detail}")
    extra = _signature_extras(ctx, p, matrix, level, sample, sample_mo)

    order = (
        matrix.loc[matrix["col_code"] == LEVEL_CODE]
        .sort_values("rho_within", ascending=False, na_position="last")["row"]
        .tolist()
    )
    pivot = matrix.pivot(index="row", columns="col", values="rho_within").loc[
        order, [t.column for t in SPEND_TRAITS]
    ]
    pivot.insert(1, "level_given_control", partial.set_index("row")["rho_partial"].reindex(order))
    labels = {t.column: t.label for t in PLACE_TRAITS}
    col_labels = [SPEND_TRAITS[0].label, control_label, *(t.label for t in SPEND_TRAITS[1:])]
    top = float(np.nanmax(np.abs(pivot.to_numpy()))) if np.isfinite(pivot.to_numpy()).any() else strong_min
    vmax = max(math.ceil(10 * top) / 10, strong_min)
    n_lo = int(matrix["n_within"].min()) if len(matrix) else 0
    n_text = style.fmt_num(n_sig) if n_lo == n_sig else style.fmt_range(n_lo, n_sig)
    # Почему n разное — под картой: в подзаголовок (≤ 120 знаков) вместе с выборкой и CLR не помещается.
    n_why = "" if n_lo == n_sig else f".\nn = {n_text}: доли занятых по отраслям известны не во всех МО"
    fig = plot_signatures(
        pivot,
        [labels[r] for r in order],
        col_labels,
        faint=faint,
        vmax=vmax,
        note=typo(f"Светлые клетки — |ρ| < {style.fmt_num(faint, 1)}; «—» — не считается{n_why}"),
    )
    subtitle = _first_fitting(
        [
            f"ρ Спирмена внутри регионов, 2023, {who}; доли корзины — в логарифмах (CLR); n = {n_text}",
            f"ρ внутри регионов, 2023, {who}; доли корзины — в логарифмах (CLR); n = {n_text}",
            f"ρ внутри регионов, 2023, {who}; доли — CLR; n = {n_text}",
        ],
        ctx.subtitle_max,
    )
    pos, neg = chosen.get("pos"), chosen.get("neg")
    alt_parts = [f"{strong} из {len(matrix)} связей по модулю не слабее {style.fmt_num(strong_min, 1)}"]
    if pos is not None:
        alt_parts.insert(
            0, f"уровень трат выше, где {pos['where']} (ρ {style.fmt_rho(pos['rho'], sign=True)})"
        )
    if neg is not None:
        alt_parts.insert(
            1 if pos is not None else 0, f"ниже, где {neg['where']} (ρ {style.fmt_rho(neg['rho'])})"
        )
    if axis is not None:
        alt_parts.append(
            f"связь там, где {axis['where']}, {control_equal} почти исчезает "
            f"(частный ρ {style.fmt_rho(axis['partial'])})"
        )
    alt = typo("Тепловая карта связей черт места и трат внутри регионов: " + "; ".join(alt_parts))
    data = matrix[
        ["row_code", "row_label", "col_code", "col_label", "rho_within", "n_within", "rho_all", "n_all"]
    ].merge(
        partial[["row_code", "rho_partial", "n_partial"]].rename(
            columns={
                "rho_partial": f"rho_within_given_{control.code}",
                "n_partial": f"n_given_{control.code}",
            }
        ),
        on="row_code",
        how="left",
    )
    is_level = data["col_code"] == LEVEL_CODE
    for col in (f"rho_within_given_{control.code}", f"n_given_{control.code}"):
        data[col] = data[col].where(is_level)
    ctx.save_figure(
        fig,
        fid="F14",
        slug="economy_signatures",
        title=title,
        subtitle=subtitle,
        alt=alt,
        data=data,
        check=check,
        source=style.join_sources(style.SOURCE_SBER, style.SOURCE_ROSSTAT),
    )
    return {
        "head": head,
        "chosen": chosen,
        "axis": axis,
        "strong": strong,
        "n_cells": len(matrix),
        "who": who,
        "who_dat": "МО без внутригородских территорий" if p["exclude_workplace_based"] else "всем МО",
        "control_equal": control_equal,
        "control_axis": control_axis,
        **extra,
    }


def _signature_extras(
    ctx: SectionContext,
    p: dict[str, Any],
    matrix: pd.DataFrame,
    level: pd.DataFrame,
    sample: str,
    sample_mo: pd.DataFrame,
) -> dict[str, Any]:
    """Факты для «Что видно» по F14: корзина, слабые столбцы, маскировка регионом, сверка с ориентиром."""
    skip = list(p["headline_skip"])
    skipped = ", ".join(skip) or "—"
    lead = column_leaders(matrix, [*LEAD_TEXT, *WEAK_TEXT], skip)
    for r in lead.itertuples(index=False):
        if r.col_code in LEAD_TEXT:
            ctx.fact(
                f"lead_{r.col_code}_where",
                None if pd.isna(r.where) else r.where,
                "str",
                f"черта места, сильнее всего связанная со столбцом {r.col_code} рис. 14; без черт {skipped}",
            )
            ctx.fact(f"lead_{r.col_code}_rho", r.rho, "rho", f"ρ Спирмена внутри регионов; {sample}")
        else:
            ctx.fact(
                f"max_abs_{r.col_code}",
                r.max_abs,
                "rho",
                f"наибольший |ρ| внутри регионов в столбце {r.col_code} рис. 14; без черт {skipped}",
            )
    cells = level.rename(columns={"row_code": "code", "row_where": "where"})
    mask = masking_cell(cells)
    ctx.fact(
        "mask_where",
        None if mask is None else mask["where"],
        "str",
        "черта места, у которой связь с уровнем трат сильнее всего меняется при учёте региона",
    )
    for key in ("rho_all", "rho_within"):
        ctx.fact(
            f"mask_{key}",
            np.nan if mask is None else mask[key],
            "rho",
            f"ρ Спирмена {'без учёта регионов' if key == 'rho_all' else 'внутри регионов'} для mask_where, "
            f"уровень трат 2023 года; {sample}",
        )
    # Та же связь без учёта регионов по всем МО, с внутригородскими: у раздела E2 выборка такая (и год 2024).
    mask_inner = np.nan
    if mask is not None and len(sample_mo) < len(ctx.data.mo):
        mo = ctx.data.mo
        mask_inner, n_inner = stats.spearman(mo[mask["row"]], mo[mask["col"]])
        ctx.fact(
            "mask_rho_all_inner",
            mask_inner,
            "rho",
            f"ρ Спирмена без учёта регионов для mask_where по всем МО с внутригородскими территориями, "
            f"уровень трат 2023 года; n = {n_inner}",
        )
    else:
        ctx.fact("mask_rho_all_inner", np.nan, "rho", "не считается: выборка рис. 14 — все МО")
    check = section_level_rho(
        ctx.data.mo, ctx.data.okved_shares, p["check_sections"], str(p["check_level"]), YEAR
    )
    for r in check.itertuples(index=False):
        ctx.fact(
            f"rho_all_sec_{r.section.lower()}_level",
            r.rho,
            "rho",
            f"ρ Спирмена без учёта регионов: доля работников раздела ОКВЭД2 {r.section} ~ "
            f"{p['check_level']}; все МО с внутригородскими; n = {r.n}; определение ориентира Б.4",
        )
    return {"lead": lead.set_index("col_code"), "mask": mask, "mask_inner": mask_inner}


def suburb_verdict(ratio: float, p_value: float, thr: float, alpha: float) -> str:
    """Вывод о пригородах по отношению медиан ``ratio`` и p Манна — Уитни.

    ``higher`` — ratio ≥ ``thr`` и p < ``alpha``; ``lower`` — ratio ≤ 1 / ``thr`` и p < ``alpha``; с суффиксом
    ``_weak`` — то же направление при p ≥ ``alpha`` (различие ненадёжно); ``similar`` — между порогами;
    ``unknown`` — отношение не посчитано (нет пригородов или остальных МО).
    """
    if not np.isfinite(ratio):
        return "unknown"
    if 1 / thr < ratio < thr:
        return "similar"
    direction = "higher" if ratio >= thr else "lower"
    return direction if (np.isfinite(p_value) and p_value < alpha) else f"{direction}_weak"


def effect_verdict(ratio: float, p_value: float, thr: float, alpha: float) -> str:
    """Вывод о скорректированном различии: ``strong`` — ratio ≥ ``thr`` (или ≤ 1 / ``thr``) при p < ``alpha``;
    ``weak`` — различие надёжно (p < ``alpha``), но слабее порога; ``none`` — ненадёжно; ``unknown`` — не
    посчитано. Направление — знак ``ratio`` − 1."""
    if not (np.isfinite(ratio) and np.isfinite(p_value)):
        return "unknown"
    if p_value >= alpha:
        return "none"
    return "strong" if (ratio >= thr or ratio <= 1 / thr) else "weak"


# Тексты вывода о пригородах: заголовок F15 ({r} — «в N раза», {r2} — отношение с двумя знаками), фраза
# альт-текста и фраза «Что видно» ({k} — начало ключа факта «{{e5.»).
_SUBURB_TEXT: dict[str, tuple[str, str, str]] = {
    "higher": (
        "В пригородах траты к доходу по месту работы в {r} раза выше, чем в остальных МО",
        "в пригородах столиц регионов траты к доходу по месту работы в {r} раза выше, чем в остальных МО",
        "В пригородах столиц регионов отношение в {k}suburb_ratio}} раза выше, чем в остальных МО "
        "(p Манна — Уитни {k}suburb_mw_p}})",
    ),
    "higher_weak": (
        "В пригородах траты к доходу по месту работы выше в {r} раза, но различие ненадёжно",
        "в пригородах столиц регионов траты к доходу по месту работы выше, но различие ненадёжно",
        "В пригородах столиц регионов отношение выше в {k}suburb_ratio}} раза, но различие ненадёжно "
        "(p Манна — Уитни {k}suburb_mw_p}})",
    ),
    "lower": (
        "В пригородах траты к доходу по месту работы ниже, чем в остальных МО: {r2}",
        "в пригородах столиц регионов траты к доходу по месту работы ниже, чем в остальных МО",
        "В пригородах столиц регионов отношение ниже, чем в остальных МО: {k}suburb_ratio}} "
        "(p Манна — Уитни {k}suburb_mw_p}})",
    ),
    "lower_weak": (
        "В пригородах траты к доходу по месту работы ниже ({r2}), но различие ненадёжно",
        "в пригородах столиц регионов траты к доходу по месту работы ниже, но различие ненадёжно",
        "В пригородах столиц регионов отношение ниже ({k}suburb_ratio}} к остальным МО), но различие "
        "ненадёжно (p Манна — Уитни {k}suburb_mw_p}})",
    ),
    "similar": (
        "Пригороды столиц не выделяются: траты к доходу по месту работы — {m1} против {m2}",
        "пригороды столиц регионов по тратам к доходу по месту работы не выделяются среди остальных МО",
        "В пригородах столиц регионов медиана — {k}suburb_median}}, в остальных МО с известным расстоянием "
        "до столицы — {k}other_median}} (p Манна — Уитни {k}suburb_mw_p}})",
    ),
    "unknown": (
        "Пригороды не сравнить с остальными МО по тратам к доходу: нет пригодных МО",
        "пригороды столиц регионов не с чем сравнить: нет пригодных МО",
        "Сравнить пригороды столиц регионов с остальными МО не удалось",
    ),
}
# Заголовок пункта «Что видно» о пригородах по выводу suburb_verdict.
_SUBURB_BULLET: dict[str, str] = {
    "higher": "Пригороды выделяются",
    "higher_weak": "Различие пригородов ненадёжно",
    "lower": "Пригороды ниже остальных МО",
    "lower_weak": "Различие пригородов ненадёжно",
    "similar": "Пригороды не выделяются",
    "unknown": "Пригороды не с чем сравнить",
}


def _suburb_text(
    verdict: str, part: int, ratio: float, medians: tuple[float, float] = (np.nan, np.nan)
) -> str:
    """Текст вывода ``verdict``: ``part`` 0 — заголовок, 1 — альт-текст, 2 — «Что видно» с ключами фактов.

    ``medians`` — медианы отношения у пригородов и у остальных МО (для вывода «не выделяются»).
    """
    template = _SUBURB_TEXT[verdict][part]
    k = "{{" + SECTION_ID + "."
    subs = {
        "{r}": _times(ratio),
        "{r2}": style.fmt_num(ratio, 2),
        "{m1}": style.fmt_num(medians[0], 2),
        "{m2}": style.fmt_num(medians[1], 2),
        "{k}": k,
    }
    for key, value in subs.items():
        template = template.replace(key, value)
    return template


def _suburb_headline(
    ratio: float, p_value: float, thr: float, alpha: float, medians: tuple[float, float] = (np.nan, np.nan)
) -> Headline:
    """Заголовок F15 по выводу ``suburb_verdict``; проверка — условие этого вывода на текущих числах."""
    verdict = suburb_verdict(ratio, p_value, thr, alpha)
    detail = f"suburb_ratio = {ratio:.3f}, suburb_mw_p = {p_value:.3g}, порог {thr}, α = {alpha}"
    thr_t, alpha_t = style.fmt_num(thr, 1), style.fmt_num(alpha, 2)
    checks = {
        "higher": f"suburb_ratio ≥ {thr_t} и suburb_mw_p < {alpha_t}",
        "higher_weak": f"suburb_ratio ≥ {thr_t} и suburb_mw_p ≥ {alpha_t}",
        "lower": f"suburb_ratio ≤ 1 / {thr_t} и suburb_mw_p < {alpha_t}",
        "lower_weak": f"suburb_ratio ≤ 1 / {thr_t} и suburb_mw_p ≥ {alpha_t}",
        "similar": f"1 / {thr_t} < suburb_ratio < {thr_t}",
        "unknown": "suburb_ratio посчитан",
    }
    text = typo(_suburb_text(verdict, 0, ratio, medians))
    return Headline(text, verdict != "unknown", checks[verdict], detail)


def _work_home(
    ctx: SectionContext, p: dict[str, Any], names: pd.Series, labels: pd.Series, hints: pd.Series
) -> dict[str, Any]:
    """F15, T12 и факты о тратах к доходу по месту работы.

    ``names`` — названия МО для текста и таблиц (``display_names``), ``labels`` — короткие подписи точек.
    """
    cfg_eda = ctx.cfg["eda"]
    mo = ctx.data.mo.reset_index(drop=True)
    types = cfg_eda["economy_types"]
    if p["suburb_type"] not in types:
        raise KeyError(f"в eda.economy_types нет типа {p['suburb_type']!r}: по нему считаются факты suburb_*")
    rule = str(types[p["suburb_type"]]["rule"])
    wh = work_home(
        mo,
        suburb_rule=rule,
        recipients_ok=ctx.cfg["context"]["ndfl"]["recipients_ratio_ok"],
        quantiles=QUANTILES,
    )
    _ratio_facts(ctx, mo, wh)
    thr = float(cfg_eda["rejection"]["s6_suburb_ratio_min"])
    alpha = float(p["mw_alpha"])
    km = rule_threshold(rule, DIST)
    extra = _suburb_facts(ctx, p, mo, wh, rule, km, thr, names)

    edges = ndfl_edges(mo, wh["usable"], int(p["edge_n"]), hints, names)
    for block, key in (("high", "ndfl_high_top"), ("low", "ndfl_low_top")):
        part = edges.loc[edges["block"] == block].head(int(p["examples_n"]))
        text = ", ".join(
            f"{n} {style.fmt_num(v, 2)}" for n, v in zip(part["name"], part[NDFL_RATIO], strict=True)
        )
        ctx.fact(
            key,
            text,
            "str",
            f"МО с самым {'высоким' if block == 'high' else 'низким'} отношением трат {YEAR} года к доходу "
            "5-НДФЛ на жителя среди n_ndfl_usable; без МО с неполным годом трат (n_ndfl_no_level)",
        )
    # Ночёвки: без внутригородских территорий — у центральных районов городов мало жителей и много гостиниц.
    resident = mo.loc[mo["nights_pc_2023"].notna() & ~mo["workplace_based"].astype(bool)]
    nights = resident.nlargest(int(p["examples_n"]), "nights_pc_2023")
    ctx.fact(
        "top_nights",
        ", ".join(
            f"{names.get(t, n)} {style.fmt_num(v, 0)}"
            for t, n, v in zip(nights["territory_id"], nights["name"], nights["nights_pc_2023"], strict=True)
        ),
        "str",
        "МО с наибольшим числом ночёвок в средствах размещения на жителя, 2023; без внутригородских "
        "территорий",
    )
    usable = wh["usable"].to_numpy()
    part = mo.loc[usable]
    rho, n = stats.spearman(part[NDFL_RATIO], part[DIST])
    ctx.fact(
        "rho_ndfl_dist", rho, "rho", f"ρ Спирмена spend_to_ndfl_2023 ~ dist_capital_km, пригодные МО; n = {n}"
    )
    rho_dist, n = stats.spearman_within(part[NDFL_RATIO], part[DIST], part[GROUP_COL])
    ctx.fact(
        "rho_ndfl_dist_within",
        rho_dist,
        "rho",
        f"ρ Спирмена внутри регионов: spend_to_ndfl_2023 ~ dist_capital_km, пригодные МО; n = {n}",
    )
    dist_weak = bool(np.isfinite(rho_dist) and abs(rho_dist) < float(p["headline_abs_rho"]))

    medians = (wh["suburb_median"], wh["other_median"])
    verdict = suburb_verdict(wh["suburb_ratio"], wh["suburb_mw_p"], thr, alpha)
    head = _suburb_headline(wh["suburb_ratio"], wh["suburb_mw_p"], thr, alpha, medians)
    title = ctx.headline(head.text, head.ok, f"F15 work_vs_home: {head.check}; {head.detail}")
    _plot_f15(ctx, mo, wh, labels, p, km, verdict, title, head.check)
    _t12(ctx, wh, edges)
    return {
        "wh": wh,
        "head": head,
        "verdict": verdict,
        "thr": thr,
        "alpha": alpha,
        "dist_weak": dist_weak,
        **extra,
    }


def _ratio_facts(ctx: SectionContext, mo: pd.DataFrame, wh: dict[str, Any]) -> None:
    """Факты о распределении отношений и о получателях дохода 5-НДФЛ."""
    lo, hi = wh["recip_bounds"]
    dist = wh["distributions"]
    ctx.fact(
        "n_ndfl_usable",
        wh["n_usable"],
        "int",
        "МО с ndfl_ok и известным отношением трат 2023 года к доходу 5-НДФЛ на жителя",
    )
    for name, key in (
        (NDFL_RATIO, "spend_to_ndfl"),
        (WAGE_RATIO, "spend_to_wage"),
        (RECIPIENTS, "recip_to_pop"),
    ):
        for q in QUANTILES:
            qk = quantile_key(q)
            note = f"{QUANTILE_NAMES.get(q, qk)} {name} по {int(dist.loc[name, 'n'])} МО"
            if key == "spend_to_ndfl":
                note += (
                    "; ориентир Б.4 посчитан без фильтра ndfl_ok (число получателей на жителя вне допустимых "
                    "пределов и дефект источника не исключались), поэтому на другой выборке МО"
                )
            ctx.fact(f"{key}_{qk}", dist.loc[name, qk], "num2", note)
    defects = recipient_defects(ctx.data.context_long, ctx.data.context_annual, YEAR, lo)
    ctx.fact(
        "n_recip_gt3",
        wh["n_recip_gt"],
        "int",
        f"МО панели, где получателей дохода 5-НДФЛ 2023 года больше {style.fmt_num(hi, 2)} на жителя; "
        f"из них внутригородских — {wh['n_recip_gt_inner']} (факт n_recip_gt3_inner). Ориентир Л3 посчитан "
        "по всем МО 5-НДФЛ, а не только по МО панели",
    )
    ctx.fact(
        "n_recip_gt3_inner",
        wh["n_recip_gt_inner"],
        "int",
        "из n_recip_gt3 — внутригородские территории Москвы и Петербурга (5-НДФЛ по месту работодателя)",
    )
    ctx.fact(
        "n_recip_lt005",
        wh["n_recip_lt"],
        "int",
        f"МО панели, где получателей дохода 5-НДФЛ 2023 года меньше {style.fmt_num(lo, 2)} на жителя, после "
        f"снятия дефекта источника «единицы вместо тысяч» (флаг {RECIPIENTS_JUMP_FLAG} пакета P, факт "
        f"n_recip_jump); до снятия — {defects['n_lt_raw']} (факт n_recip_lt005_raw). Ориентир Л3 посчитан "
        "до снятия дефекта и по всем МО 5-НДФЛ, а не только по МО панели",
    )
    ctx.fact(
        "n_recip_lt005_raw",
        defects["n_lt_raw"],
        "int",
        f"МО панели, где получателей дохода 5-НДФЛ 2023 года меньше {style.fmt_num(lo, 2)} на жителя "
        "по сырым значениям context_long, включая дефект единиц",
    )
    ctx.fact(
        "n_recip_jump",
        defects["n_jump"],
        "int",
        f"МО панели, у которых число получателей 5-НДФЛ 2023 года снято как дефект источника "
        f"(флаг {RECIPIENTS_JUMP_FLAG}: единицы вместо тысяч); у них ndfl_ok ложен",
    )
    no_level = mo["ndfl_ok"].astype(bool) & mo["ndfl_pc_2023"].notna() & mo["level_2023"].isna()
    ctx.fact(
        "n_ndfl_no_level",
        int(no_level.sum()),
        "int",
        "МО с пригодным доходом 5-НДФЛ на жителя, но без уровня трат 2023 года (меньше 12 месяцев в году, "
        "level_2023 по Б.2 пуст) — не входят в n_ndfl_usable и края T12",
    )
    part = mo.loc[wh["usable"].to_numpy()]
    rho, n = stats.spearman(part[NDFL_RATIO], part[RECIPIENTS])
    ctx.fact(
        "rho_ratio_recipients",
        rho,
        "rho",
        f"ρ Спирмена spend_to_ndfl_2023 ~ recipients_to_pop_2023, пригодные МО; n = {n}: отношение по рангу "
        "почти обратно числу получателей на жителя, получатели не годятся для его внешней проверки",
    )
    rho_w, n = stats.spearman_within(part[NDFL_RATIO], part[RECIPIENTS], part[GROUP_COL])
    ctx.fact(
        "rho_ratio_recipients_within",
        rho_w,
        "rho",
        f"то же внутри регионов; n = {n}",
    )


def _suburb_facts(
    ctx: SectionContext,
    p: dict[str, Any],
    mo: pd.DataFrame,
    wh: dict[str, Any],
    rule: str,
    km: float | None,
    thr: float,
    names: pd.Series,
) -> dict[str, Any]:
    """Факты С6 о пригородах: сравнение, чувствительность, столицы, поправка на структуру, проверки."""
    ratio = mo[NDFL_RATIO].astype("float64")
    eligible, suburb, capital = wh["eligible"], wh["suburb"], wh["capital"]
    ctx.fact("suburb_n", wh["suburb_n"], "int", f"пригодных МО по правилу пригорода «{rule}»")
    ctx.fact(
        "other_n",
        wh["other_n"],
        "int",
        "пригодных МО с известным расстоянием до столицы региона, не пригородов (с ними сравниваются "
        "пригороды)",
    )
    ctx.fact(
        "n_dist_unknown",
        wh["n_dist_unknown"],
        "int",
        "пригодных МО без расстояния до столицы региона (нет столицы субъекта в справочнике, как у "
        "Московской области, или нет дороги): не входят ни в пригороды, ни в остальные МО",
    )
    ctx.fact(
        "suburb_km", np.nan if km is None else km, "int", "порог расстояния правила пригорода, км по дорогам"
    )
    ctx.fact("suburb_ratio_min", thr, "num1", "порог критерия отказа С6 для suburb_ratio (eda.rejection)")
    sens = suburb_sensitivity(
        mo, ratio, eligible, suburb_variants(rule, DIST, p["suburb_sens_km"], p["suburb_district_types"])
    )
    _sensitivity_facts(ctx, sens, p["suburb_sens_km"] if km is not None else [])
    ctx.fact(
        "suburb_ratio",
        wh["suburb_ratio"],
        "num2",
        "медиана spend_to_ndfl_2023 пригородов / медиана остальных пригодных МО с известным расстоянием",
    )
    ctx.fact(
        "suburb_mw_p", wh["suburb_mw_p"], "p", "двусторонний тест Манна — Уитни: пригороды против остальных"
    )
    ctx.fact("suburb_median", wh["suburb_median"], "num2", "медиана spend_to_ndfl_2023 пригородов")
    ctx.fact(
        "other_median",
        wh["other_median"],
        "num2",
        "медиана spend_to_ndfl_2023 остальных пригодных МО (other_n)",
    )
    ctx.fact("capital_n", wh["capital_n"], "int", "пригодных столиц регионов (is_capital)")
    ctx.fact("capital_median", wh["capital_median"], "num2", "медиана spend_to_ndfl_2023 столиц регионов")

    pair = paired_by_region(ratio, eligible, mo[GROUP_COL], capital, suburb)
    where = "в регионах, где есть пригодные столица, пригороды и прочие МО"
    ctx.fact("n_regions_paired", pair["n_regions"], "int", f"регионов для сравнения со столицей: {where}")
    for key, value, note in (
        ("suburb_to_capital_within", pair["group_to_capital"], "медиана по регионам: пригороды / столица"),
        ("other_to_capital_within", pair["other_to_capital"], "медиана по регионам: прочие МО / столица"),
        ("suburb_to_other_within", pair["group_to_other"], "медиана по регионам: пригороды / прочие МО"),
    ):
        ctx.fact(key, value, "num2", f"{note} (медианы spend_to_ndfl_2023 в регионе); {where}")
    ctx.fact("suburb_above_capital_n", pair["group_above"], "int", "регионов, где пригороды выше столицы")
    ctx.fact("other_above_capital_n", pair["other_above"], "int", "регионов, где прочие МО выше столицы")

    controls = list(p["adjust_controls"])
    adj = adjusted_comparison(mo, ratio, eligible, suburb, controls)
    how = f"остатки ln spend_to_ndfl_2023 после дамми регионов и {', '.join(controls)}"
    ctx.fact(
        "suburb_ratio_adj",
        adj["ratio"],
        "num2",
        f"во сколько раз медиана пригородов выше медианы остальных МО при равной структуре: {how}; "
        f"n = {adj['n_all']}, пригородов {adj['n']}",
    )
    ctx.fact("suburb_adj_p", adj["p"], "p", f"Манн — Уитни по остаткам: пригороды против остальных; {how}")

    known = ~mo["workplace_based"].astype(bool) & mo[DIST].notna()
    wage = wage_line_check(mo, known, suburb, int(p["wage_top_n"]))
    line = "линия ln level_2023 ~ ln wage_2023 (Тейл — Сен, без внутригородских, как у E2)"
    ctx.fact(
        "suburb_wage_ratio",
        wage["ratio"],
        "num2",
        f"во сколько раз пригороды тратят сверх {line} больше остальных МО с известным расстоянием "
        f"(медианы exp остатка); пригородов {wage['n']}; гипотеза раздела E2",
    )
    ctx.fact("suburb_wage_p", wage["p"], "p", "Манн — Уитни по остаткам от линии «траты ~ зарплата»")
    ctx.fact("wage_top_n", wage["top_n"], "int", "МО с наибольшим остатком от линии «траты ~ зарплата»")
    ctx.fact(
        "suburb_wage_top_k", wage["top_k"], "int", "пригородов среди wage_top_n МО с наибольшим остатком"
    )
    ctx.fact("suburb_wage_base_share", wage["base_share"], "pct", "доля пригородов среди МО сравнения")

    outer = _outer_facts(ctx, p, mo, wh["usable"], km, thr)
    lows = mo.loc[(eligible & suburb).to_numpy()].nsmallest(2, NDFL_RATIO)
    ctx.fact(
        "suburb_low_names",
        ", ".join(
            f"{names.get(t, n)} ({style.fmt_num(d)} км, {style.fmt_num(v, 2)})"
            for t, n, d, v in zip(
                lows["territory_id"], lows["name"], lows[DIST], lows[NDFL_RATIO], strict=True
            )
        ),
        "str",
        "пригороды с самым низким spend_to_ndfl_2023: расстояние до столицы региона и отношение",
    )
    zero = mo[DIST].eq(0) & ~mo["is_capital"].astype(bool)
    ctx.fact(
        "n_zero_dist",
        int(zero.sum()),
        "int",
        "МО панели не столиц с расстоянием 0 км: районы и округа, чей административный центр — столица "
        "региона",
    )
    district_caps = mo.loc[mo["is_capital"].astype(bool) & (mo["mo_type"].astype(str) != "go")]
    ctx.fact(
        "capital_districts",
        "; ".join(
            f"{names.get(t, n)} ({r})"
            for t, n, r in zip(
                district_caps["territory_id"],
                district_caps["name"],
                district_caps["region_name"],
                strict=True,
            )
        )
        or None,
        "str",
        "столицы субъектов по справочнику, которые не городские округа",
    )
    return {"sens": sens, "adj": adj, "pair": pair, "wage": wage, "outer": outer}


def _outer_facts(
    ctx: SectionContext, p: dict[str, Any], mo: pd.DataFrame, usable: pd.Series, km: float | None, thr: float
) -> list[dict[str, Any]]:
    """Факты о регионах без столицы субъекта (``outer_centers``): отношение против расстояния до города."""
    near_km = float(km) if km is not None else float(max(p["suburb_sens_km"]))
    out = []
    for code, spec in dict(p["outer_centers"]).items():
        check = outer_center_check(mo, usable, str(spec["region"]), str(spec["center"]), near_km)
        base = f"{spec['region']}: пригодные МО, расстояние по прямой до центра региона «{spec['center']}»"
        ctx.fact(f"{code}_n", check["n"], "int", f"{base}; всего МО")
        ctx.fact(
            f"{code}_near_n", check["n_near"], "int", f"{base}; МО не дальше {style.fmt_num(near_km)} км"
        )
        ctx.fact(
            f"{code}_near_ratio", check["near"], "num2", f"{base}; медиана spend_to_ndfl_2023 ближних МО"
        )
        ctx.fact(f"{code}_far_ratio", check["far"], "num2", f"{base}; медиана spend_to_ndfl_2023 дальних МО")
        ctx.fact(f"{code}_rho_dist", check["rho"], "rho", f"{base}; ρ Спирмена отношения с расстоянием")
        out.append({"code": code, **dict(spec), **check})
    return out


def _plot_f15(
    ctx: SectionContext,
    mo: pd.DataFrame,
    wh: dict[str, Any],
    labels: pd.Series,
    p: dict[str, Any],
    km: float | None,
    verdict: str,
    title: str,
    check: str,
) -> None:
    """F15: точки пригодных МО с известным расстоянием, пригороды и столицы отдельно, подписи краёв."""
    usable = wh["usable"].to_numpy()
    pts = mo.loc[usable & mo[DIST].notna().to_numpy()].copy()
    pts["x"] = pts[DIST] + 1
    pts["y"] = pts[NDFL_RATIO]
    pts["is_suburb"] = wh["suburb"].to_numpy()[pts.index]
    pts["is_capital"] = pts["is_capital"].astype(bool)
    pts["name_short"] = pts["territory_id"].map(labels).fillna(pts["name"])
    high = pts.nlargest(int(p["label_high"]), "y")
    low = pts.nsmallest(int(p["label_low"]), "y")
    labeled = pd.concat([high, low]).drop_duplicates("territory_id")
    labeled = labeled.assign(label=labeled["name_short"])
    median = float(wh["distributions"].loc[NDFL_RATIO, quantile_key(MEDIAN_Q)])
    note = "пригороды столиц регионов" + ("" if km is None else f" (до {style.fmt_num(km)} км по дорогам)")
    cap_note = f"столицы регионов, медиана {style.fmt_num(wh['capital_median'], 2)}"
    fig = plot_work_home(pts, labeled, median, note, cap_note)
    n_pts = style.fmt_num(len(pts))
    subtitle = _first_fitting(
        [
            f"Траты / доход 5-НДФЛ, {YEAR}; доход — по месту работы и без пенсий, поэтому больше 1 — норма; "
            f"n = {n_pts}; оси логарифмические",
            f"Траты / доход 5-НДФЛ, {YEAR}; доход по месту работы и без пенсий, поэтому больше 1 — норма; "
            f"n = {n_pts}; оси логарифмические",
            f"Траты / доход 5-НДФЛ, {YEAR}; доход — по месту работы и без пенсий, поэтому больше 1 — норма; "
            f"n = {n_pts}",
        ],
        ctx.subtitle_max,
    )
    alt = typo(
        f"Точечный график: {_suburb_text(verdict, 1, wh['suburb_ratio'])}; ниже всего отношение в столицах "
        f"регионов (медиана {style.fmt_num(wh['capital_median'], 2)}); медиана всех МО — "
        f"{style.fmt_num(median, 2)}"
    )
    data = pts[
        ["territory_id", "name_short", "region_name", DIST, NDFL_RATIO, "is_suburb", "is_capital"]
    ].assign(labeled=pts["territory_id"].isin(labeled["territory_id"]))
    ctx.save_figure(
        fig,
        fid="F15",
        slug="work_vs_home",
        title=title,
        subtitle=subtitle,
        alt=alt,
        data=data,
        check=check,
        source=style.join_sources(style.SOURCE_SBER, style.SOURCE_FNS),
    )


def _t12(ctx: SectionContext, wh: dict[str, Any], edges: pd.DataFrame) -> None:
    table = work_home_table(wh["distributions"], edges, QUANTILES)
    ratio_fmt = dict.fromkeys((NDFL_RATIO, WAGE_RATIO, RECIPIENTS), _num2)
    ctx.save_table(
        table,
        tid="T12",
        slug="work_home",
        md_rows=len(table),
        title=typo("Траты к доходу по месту работы: распределения и крайние МО"),
        md_formats={**ratio_fmt, DIST: _num0, "territory_id": _id, "block": EDGE_LABELS.get},
        md_labels={
            "block": "Блок",
            "territory_id": "id",
            "name": "МО или статистика",
            "region_name": "Регион",
            NDFL_RATIO: "Траты / доход 5-НДФЛ",
            WAGE_RATIO: "Траты / зарплата",
            RECIPIENTS: "Получатели / жители",
            DIST: "До столицы региона, км",
            "hint": "Подсказка",
        },
    )


def _sensitivity_facts(ctx: SectionContext, sens: pd.DataFrame, km_list: Sequence[float]) -> None:
    """Факты о чувствительности вывода о пригородах к правилу; ``km_list`` — пороги вариантов, км."""
    valid = sens.dropna(subset=["ratio"])
    listing = "; ".join(
        f"{r.variant}: {style.fmt_num(r.ratio, 2)} (n = {r.n}, p = {style.fmt_p(r.p)})"
        for r in valid.itertuples()
    )
    ctx.fact(
        "suburb_sens_n", len(valid), "int", f"вариантов правила пригорода с посчитанным отношением: {listing}"
    )
    for side, fn in (("min", "idxmin"), ("max", "idxmax")):
        row = valid.loc[getattr(valid["ratio"], fn)()] if len(valid) else None
        ctx.fact(
            f"suburb_sens_{side}",
            np.nan if row is None else row["ratio"],
            "num2",
            f"{'наименьшее' if side == 'min' else 'наибольшее'} suburb_ratio по вариантам правила"
            + ("" if row is None else f": {row['variant']}, n = {row['n']}"),
        )
    km = [float(k) for k in km_list]
    ctx.fact(
        "suburb_sens_km_min", min(km) if km else np.nan, "int", "наименьший порог расстояния вариантов, км"
    )
    ctx.fact(
        "suburb_sens_km_max", max(km) if km else np.nan, "int", "наибольший порог расстояния вариантов, км"
    )


# Колонки T11 в отчёте (CSV — целиком, с кодом правила и числом внутригородских МО).
T11_MD_DROP: tuple[str, ...] = ("type", "rule", "rule_applied", "share_mo", "n_inner")


def _types(ctx: SectionContext, p: dict[str, Any], names: pd.Series) -> pd.DataFrame:
    """T11 и факты n_type_<тип>."""
    cfg_eda = ctx.cfg["eda"]
    year = int(cfg_eda["reference_year"])
    table = type_signatures(
        ctx.data.mo,
        cfg_eda["economy_types"],
        year=year,
        examples_n=int(p["examples_n"]),
        examples_by=str(p["examples_by"]),
        names=names,
        exclude_workplace_based=bool(p["types_exclude_workplace_based"]),
    )
    for row in table.itertuples(index=False):
        if row.type != ALL_TYPE:
            inner = f"; из них внутригородских — {row.n_inner}"
            ctx.fact(f"n_type_{row.type}", row.n, "int", f"МО типа «{row.label}»: {row.rule_applied}{inner}")
            value = getattr(row, NDFL_RATIO)
            why = (
                "; пропуск: у МО типа нет пригодного дохода 5-НДФЛ на жителя (по месту работы не считается)"
                if pd.isna(value)
                else ""
            )
            ctx.fact(
                f"ndfl_type_{row.type}",
                value,
                "num2",
                f"медиана spend_to_ndfl_2023 у МО типа «{row.label}»{why}",
            )
    # Тип с самым низким и самым высоким отношением трат к доходу по месту работы — для текста о С6.
    ranked = table.loc[table["type"] != ALL_TYPE].dropna(subset=[NDFL_RATIO]).sort_values(NDFL_RATIO)
    for side, row in (("min", ranked.head(1)), ("max", ranked.tail(1))):
        label = row["label"].iloc[0] if len(row) else None
        value = row[NDFL_RATIO].iloc[0] if len(row) else np.nan
        which = "низкой" if side == "min" else "высокой"
        ctx.fact(
            f"ndfl_type_{side}_label", label, "str", f"тип T11 с самой {which} медианой spend_to_ndfl_2023"
        )
        ctx.fact(
            f"ndfl_type_{side}", value, "num2", f"медиана spend_to_ndfl_2023 типа с самой {which} медианой"
        )
    formats = {
        "rule": _code,
        "share_mo": _pct,
        f"level_to_region_{year}": _num2,
        **{f"sh_{c}_{year}": _pct for c in ("food", "other", "cafe", "marketplace")},
        "summer_excess": _pct_signed,
        WAGE_RATIO: _num2,
        NDFL_RATIO: _num2,
        "mp_pp_change": _pp,
    }
    labels = {
        "label": "Тип",
        "rule_text": "Правило",
        "n": "МО",
        f"level_to_region_{year}": f"Уровень к региону, {year}",
        f"sh_food_{year}": "Продовольствие",
        f"sh_other_{year}": "«Прочее»",
        f"sh_cafe_{year}": "Общепит",
        f"sh_marketplace_{year}": "Маркетплейсы",
        "summer_excess": "Летний избыток",
        WAGE_RATIO: "Траты / зарплата",
        NDFL_RATIO: "Траты / доход 5-НДФЛ",
        "mp_pp_change": "Прирост доли маркетплейсов",
        "examples": "Примеры",
    }
    record = ctx.save_table(
        table,
        tid="T11",
        slug="economy_types",
        title=typo(
            "Гипотезы о типах местной экономики: медианы черт трат против всех МО; внутригородские "
            "территории входят только в свой тип, код правила — в CSV"
        ),
        md_formats=formats,
    )
    # В отчёте правило — словами (код ``query`` и столбцы для проверки остаются в CSV).
    specs = cfg_eda["economy_types"]
    words = [
        "" if t == ALL_TYPE else (str(specs[t].get("text") or "") or rule_words(t, r) or _code(r))
        for t, r in zip(table["type"], table["rule"], strict=True)
    ]
    md_table = table.assign(rule_text=words).drop(columns=list(T11_MD_DROP))
    md_table = md_table[
        ["label", "rule_text", *[c for c in md_table.columns if c not in ("label", "rule_text")]]
    ]
    record_md = dataclasses.replace(record, markdown=to_markdown(md_table, formats, labels))
    ctx.tables[ctx.tables.index(record)] = record_md
    return table


def _fact_ref(key: str) -> str:
    """Ссылка на факт раздела в тексте: «{{e5.key}}»."""
    return "{{" + f"{SECTION_ID}.{key}" + "}}"


def _signature_text(sig: dict[str, Any], min_abs: float, weak_max: float) -> list[str]:
    """Пункты «Что видно» по рис. 14: черты места внутри регионов и маскировка картой регионов.

    Слова выбираются по знаку и силе чисел, сами числа — только ссылки на факты. ``min_abs`` — порог
    названной связи, ``weak_max`` — порог «сильной» связи (столбцы слабее него названы «связаны слабо»).
    """
    f = _fact_ref
    chosen = sig["chosen"]
    equal = sig.get("control_equal", "при равной зарплате")
    if "pos" in chosen and "neg" in chosen:
        text = (
            f"Траты выше там, где {f('sig_pos_where')} (ρ = {f('sig_pos_rho')}), и ниже, где "
            f"{f('sig_neg_where')} (ρ = {f('sig_neg_rho')})"
        )
        partials = [chosen[s].get("partial", np.nan) for s in ("pos", "neg")]
        if all(np.isfinite(float(v)) for v in partials):
            text += (
                f"; обе связи держатся и {equal} (частный ρ = {f('sig_pos_partial')} и "
                f"{f('sig_neg_partial')}; рис. 14)."
            )
        else:
            text += " (рис. 14)."
    elif chosen:
        side = next(iter(chosen))
        word = "выше" if side == "pos" else "ниже"
        text = f"Траты {word} там, где {f(f'sig_{side}_where')} (ρ = {f(f'sig_{side}_rho')}; рис. 14)."
    else:
        text = "Уровень трат почти не связан с чертами места (рис. 14)."
    axis = sig.get("axis")
    if axis is not None:
        word = "выше" if axis["rho"] > 0 else "ниже"
        text += (
            f" Где {f('axis_where')}, траты {word} (ρ = {f('axis_rho')}), но это та же ось, что "
            f"{sig.get('control_axis', 'зарплата')} (с ней ρ = {f('axis_rho_control')}): {equal} связь почти "
            f"исчезает (частный ρ = {f('axis_partial')})."
        )
    lead = sig["lead"]
    named = [c for c in LEAD_TEXT if c in lead.index and np.isfinite(lead.loc[c, "rho"])]
    named = [c for c in named if abs(lead.loc[c, "rho"]) >= min_abs]
    if named:
        parts = []
        same = len({lead.loc[c, "row_code"] for c in named}) == 1
        for i, c in enumerate(named):
            subject, up, down = LEAD_TEXT[c]
            word = up if lead.loc[c, "rho"] > 0 else down
            rho = f"ρ = {f(f'lead_{c}_rho')}"
            if same:
                parts.append(f"{subject} {word} ({rho})" if i == 0 else f"{subject} — {word} ({rho})")
            else:
                parts.append(f"{subject} {word}, где {f(f'lead_{c}_where')} ({rho})")
        if same:
            text += f" Где {f(f'lead_{named[0]}_where')}, {', '.join(parts)}."
        else:
            text += f" Внутри регионов {'; '.join(parts)}."
    weak = [c for c in WEAK_TEXT if c in lead.index and lead.loc[c, "max_abs"] < weak_max]
    # «Из 96 связей … — 41»: число в конце, без согласования «41 связь» / «42 связи».
    total = f"из {f('n_cells')} связей по модулю не слабее {f('strong_rho_min')} — {f('n_cells_strong')}"
    if weak:
        subjects = " и ".join(WEAK_TEXT[c] for c in weak)
        values = " и ".join(f(f"max_abs_{c}") for c in weak)
        text += f" С чертами места слабо связаны {subjects} (|ρ| не больше {values}); {total}."
    else:
        text += f" {total[0].upper()}{total[1:]}."
    bullets = [f"**Черты места внутри регионов.** {text}"]

    mask = sig["mask"]
    if mask is not None and abs(mask["rho_within"] - mask["rho_all"]) >= min_abs:
        a, w = float(mask["rho_all"]), float(mask["rho_within"])
        who = sig.get("who_dat", "МО без внутригородских территорий")
        if np.sign(a) != np.sign(w):
            head = "Карта регионов маскирует место."
            text = (
                f"По {who} за 2023 год: где {f('mask_where')}, без учёта регионов траты "
                f"{'выше' if a > 0 else 'ниже'} (ρ = {f('mask_rho_all')}), а внутри регионов — "
                f"{'выше' if w > 0 else 'ниже'} (ρ = {f('mask_rho_within')})."
            )
        else:
            head = f"Карта регионов {'маскирует' if abs(w) > abs(a) else 'усиливает'} связи места."
            text = (
                f"По {who} за 2023 год: где {f('mask_where')}, без учёта регионов ρ = {f('mask_rho_all')}, "
                f"внутри регионов — {f('mask_rho_within')}."
            )
        inner = sig.get("mask_inner", np.nan)
        if np.isfinite(float(inner)) and np.sign(float(inner)) != np.sign(a):
            # Раздел E2 показывает связь уровня трат с доступностью рынков по всем МО: знак там другой.
            e2 = (
                "; поэтому в разделе 2, где они есть, связь с другим знаком"
                if mask.get("code") == "access"
                else ""
            )
            text += (
                f" Если добавить внутригородские территории Москвы и Петербурга, знак без учёта регионов "
                f"меняется на обратный (ρ = {f('mask_rho_all_inner')}){e2}."
            )
        bullets.append(f"**{head}** {text}")
    return bullets


def _sensitivity_text(sens: pd.DataFrame, verdict: str, thr: float) -> str:
    """Фраза о том, зависит ли вывод о пригородах от правила: слова — по числам вариантов."""
    f = _fact_ref
    valid = sens.dropna(subset=["ratio"])
    if len(valid) < 2:
        return ""
    span = f"от {f('suburb_sens_min')} до {f('suburb_sens_max')}"
    how = (
        f"в {f('suburb_sens_n')} вариантах (порог {f('suburb_sens_km_min')}–{f('suburb_sens_km_max')} км, "
        "все МО или только районы и округа)"
    )
    lo, hi = float(valid["ratio"].min()), float(valid["ratio"].max())
    if verdict != "higher" and hi < thr:
        return (
            f" Вывод не зависит от правила пригорода: {how} отношение медиан — {span}, ниже порога С6 "
            f"({f('suburb_ratio_min')})."
        )
    if verdict == "higher" and lo >= thr:
        return f" Вывод устойчив к правилу пригорода: {how} отношение медиан — {span}."
    return f" Вывод зависит от правила пригорода: {how} отношение медиан — {span}."


def _outer_text(outer: Sequence[Mapping[str, Any]], thr: float) -> str:
    """Фраза о регионах без столицы субъекта: ближние к городу-центру МО против дальних."""
    f = _fact_ref
    out = []
    for o in outer:
        if not (np.isfinite(o["near"]) and np.isfinite(o["far"])):
            continue
        code, gen = o["code"], o.get("center_gen", o["center"])
        rel = o["near"] / o["far"]
        verdict = (
            "но и по прямому расстоянию до {} ближние МО не выделяются"
            if rel < thr
            else ("а по прямому расстоянию до {} ближние МО выделяются")
        )
        out.append(
            f" {o['region']} в тест не входит (столицы субъекта в справочнике у неё нет), "
            f"{verdict.format(gen)}: не дальше {f('suburb_km')} км — "
            f"{f(f'{code}_near_ratio')}, дальше — {f(f'{code}_far_ratio')}."
        )
    return "".join(out)


def _structure_text(home: dict[str, Any], p: dict[str, Any]) -> tuple[str, str]:
    """Пункты «Что видно» о том, что задаёт отношение, и о сравнении при равной структуре."""
    f = _fact_ref
    thr, alpha = home["thr"], home["alpha"]
    wh, pair, adj, wage = home["wh"], home["pair"], home["adj"], home["wage"]
    rho = home.get("rho_recipients", np.nan)
    how = "почти обратно" if rho <= -0.7 else ("обратно связано с" if rho < 0 else "связано с")
    first = (
        f"**Отношение задают рабочие места.** По рангу отношение {how} числу получателей 5-НДФЛ на жителя "
        f"(ρ = {f('rho_ratio_recipients')}, внутри регионов {f('rho_ratio_recipients_within')}): это "
        "плотность рабочих мест с доходом по месту работы, поэтому получатели не годятся для его внешней "
        "проверки."
    )
    if np.isfinite(wh["capital_median"]):
        first += f" Ниже всего отношение в столицах регионов ({f('capital_median')})."
    if pair["n_regions"]:
        first += (
            f" Против столицы своего региона пригороды выше в {f('suburb_to_capital_within')} раза, но и "
            f"прочие МО — в {f('other_to_capital_within')} раза (регионов в сравнении — "
            f"{f('n_regions_paired')})"
        )
        both = pair["other_to_capital"] >= thr and 1 / thr < pair["group_to_other"] < thr
        first += ": это разрыв «столица — остальные», а не свойство пригородов." if both else "."
    first += (
        f" Между типами T11 медиана ниже всего у типа «{f('ndfl_type_min_label')}» ({f('ndfl_type_min')}), "
        f"выше всего — у типа «{f('ndfl_type_max_label')}» ({f('ndfl_type_max')})."
    )

    words = [CONTROL_WORDS.get(c, c) for c in p["adjust_controls"]]
    controls = ", ".join(words[:-1]) + (" и " if len(words) > 1 else "") + words[-1] if words else ""
    v_adj = effect_verdict(adj["ratio"], adj["p"], thr, alpha)
    up = adj["ratio"] >= 1
    adj_head = {
        "strong": "При равной структуре пригороды выделяются.",
        "weak": "При равной структуре сигнал пригородов слабый.",
        "none": "При равной структуре пригороды тоже не выделяются.",
        "unknown": "Сравнение при равной структуре не посчитано.",
    }[v_adj]
    second = f"**{adj_head}**"
    if v_adj != "unknown":
        second += (
            f" С поправкой на регион, {controls} пригороды {'выше' if up else 'ниже'} остальных МО в "
            f"{f('suburb_ratio_adj')} раза (p Манна — Уитни {f('suburb_adj_p')})"
        )
        second += {
            "strong": f": это не слабее порога С6 ({f('suburb_ratio_min')}).",
            "weak": f": различие надёжно, но слабее порога С6 ({f('suburb_ratio_min')}).",
            "none": ": различие ненадёжно.",
        }[v_adj]
    v_wage = effect_verdict(wage["ratio"], wage["p"], thr, alpha)
    if v_wage != "unknown":
        # Гипотеза — «выше линии»: надёжное различие вниз её не подтверждает.
        v_wage = v_wage if wage["ratio"] > 1 else "none"
        verdict = {
            "strong": "подтверждается",
            "weak": "подтверждается слабо",
            "none": "не подтверждается",
        }[v_wage]
        word = "выше" if wage["ratio"] >= 1 else "ниже"
        second += (
            f" Гипотеза раздела 2 (пригороды выше линии «траты ~ зарплата») {verdict}: относительно этой "
            f"линии траты пригородов в {f('suburb_wage_ratio')} раза {word}, чем у остальных МО (p "
            f"{f('suburb_wage_p')}), "
            f"а среди {f('wage_top_n')} МО, сильнее всего превышающих линию, пригородов "
            f"{f('suburb_wage_top_k')} при их доле в выборке {f('suburb_wage_base_share')}."
        )
    return first, second


# Контроли скорректированного сравнения словами (винительный падеж: «с поправкой на …»).
CONTROL_WORDS: dict[str, str] = {
    "age_old_share_2023": "долю пожилых",
    "emp_sh_public_2023": "долю бюджетников",
    "urban_share_2023": "долю горожан",
    "log_density_2023": "плотность населения",
}


def _summary(sig: dict[str, Any], home: dict[str, Any], p: dict[str, Any]) -> str:
    """«Что видно» и «Что это значит для сюжета»: слова — по знаку чисел, числа — только {{ключи}} фактов."""
    f = _fact_ref
    verdict, thr = home["verdict"], home["thr"]
    bullets = _signature_text(sig, float(p["headline_abs_rho"]), float(p["fact_abs_rho"]))
    dist_word = "почти не зависит" if home["dist_weak"] else "зависит"
    suburbs = (
        f"**{_SUBURB_BULLET[verdict]}.** В типичном МО безналичные траты в {f('spend_to_ndfl_median')} раза "
        "больше дохода 5-НДФЛ на жителя (распределение — в T12): доход считается по месту работы и без "
        f"пенсий, поэтому отношение больше единицы — норма. "
        f"{_suburb_text(verdict, 2, home['wh']['suburb_ratio'])}, и внутри регионов "
        f"отношение {dist_word} от расстояния до столицы (ρ = {f('rho_ndfl_dist_within')})."
        + _sensitivity_text(home["sens"], verdict, thr)
        + _outer_text(home["outer"], thr)
    )
    structure, adjusted = _structure_text(home, p)
    edges = (
        f"**Крайние МО — гипотезы.** Самое высокое отношение — {f('ndfl_high_top')}: вероятно, жители "
        "работают у работодателей из других МО или живут на пенсии и выплаты, которых нет в 5-НДФЛ. Самое "
        f"низкое — {f('ndfl_low_top')}: вероятно, там работают приезжие, в том числе вахтой (рис. 15 и T12). "
        f"Отношение не считается у {f('n_recip_jump')} МО, где число получателей 5-НДФЛ снято как дефект "
        f"источника (единицы вместо тысяч), и у {f('n_ndfl_no_level')} МО с неполным годом трат."
    )
    bullets += [suburbs, structure, adjusted, edges]

    v_adj = effect_verdict(home["adj"]["ratio"], home["adj"]["p"], thr, home["alpha"])
    if verdict == "higher":
        s6 = (
            "Пригороды выделяются и по этому показателю: рёбра «работа — жительство» — дороги от пригородов "
            "к центрам занятости."
        )
    else:
        raw = {
            "lower": "пригороды ниже остальных МО",
            "unknown": "пригороды не с чем сравнить",
        }.get(verdict, "против остальных МО пригороды не выделяются")
        adj_phrase = {
            "strong": f"при равной структуре — выше в {f('suburb_ratio_adj')} раза, и критерий стоит "
            "проверить на скорректированном сравнении",
            "weak": f"при равной структуре — выше лишь в {f('suburb_ratio_adj')} раза",
            "none": "при равной структуре — тоже",
            "unknown": "сравнение при равной структуре не посчитано",
        }[v_adj]
        s6 = (
            f"Критерий отказа С6 не выполнен: {raw}, а {adj_phrase}, так что рёбра «работа — жительство» от "
            "пригородов к центрам занятости этот показатель подкрепляет слабо. Различие «столицы — остальные "
            "МО» может остаться слоем."
        )
    meaning = (
        f"Для сюжета С6 «Где зарабатывают и где тратят» пригодны {f('n_ndfl_usable')} МО с надёжным 5-НДФЛ; "
        f"признак узла — ln(траты / доход 5-НДФЛ), в сводке `log_spend_to_ndfl`. {s6} Черты места, чья связь "
        f"с тратами держится внутри регионов и {sig.get('control_equal', 'при равной зарплате')}, годятся "
        "для объяснения типов на этапе 2; их лучше брать относительно региона."
    )
    listing = "\n".join(f"- {b}" for b in bullets)
    return typo(f"**Что видно.**\n\n{listing}\n\n**Что это значит для сюжета.** {meaning}")


CAVEATS: tuple[str, ...] = (
    "Доход 5-НДФЛ, занятость и зарплата Росстата учитываются по месту работодателя, а траты СберИндекса — "
    "по жителям МО. Поэтому высокое отношение трат к доходу, вероятно, значит, что жители работают у "
    "работодателей из других МО или живут на пенсии и выплаты, которых нет в 5-НДФЛ, а низкое — что в МО "
    "работают приезжие (вахта) или крупный работодатель платит сотрудникам из других МО. Это гипотезы.",
    "Траты — оценка СберИндекса средних безналичных трат жителя МО; знаменатель (все жители или держатели "
    "карт) и модель привязки трат к МО не раскрыты. Траты жителей вне своего МО (поездки, онлайн) по смыслу "
    "входят, траты приезжих в курортном МО не видны: курортный тип T11 задан ночёвками Росстата, а не "
    "тратами. Отношение трат к доходу — сравнительная мера, а не доля расходов в доходе.",
    "Тест пригородов. Пригодные МО без расстояния до столицы региона ({{e5.n_dist_unknown}}, в основном "
    "Московская область: столицы субъекта в справочнике у неё нет) в него не входят. У {{e5.n_zero_dist}} "
    "районов и округов с центром в столице региона расстояние — 0 км, и правило пригорода захватывает "
    "районы вокруг столицы со своими крупными работодателями: {{e5.suburb_low_names}}.",
    "Занятость и зарплата Росстата — без малого бизнеса; доли разделов ОКВЭД2 считаются от работников "
    "крупных и средних организаций. Скрытый раздел — пропуск, а не ноль: доли занятых в добыче, обработке "
    "и сельском хозяйстве известны не во всех МО, эти строки рис. 14 посчитаны на меньшей выборке. Доля "
    "бюджетной сферы высока там, где мало частных работодателей, поэтому она идёт вместе с низкой зарплатой.",
    "Внутригородские территории Москвы и Петербурга не входят в рис. 14 и в отношения к доходу: показатели "
    "по месту работы на их жителя не считаются. В типы T11 они входят только в свой тип.",
    "Связь внутри регионов — не причина: она может идти через третью черту, например зарплату; рост трат — "
    "номинальный.",
)
SUBURB_CAVEAT = next(i for i, c in enumerate(CAVEATS) if c.startswith("Тест пригородов"))


def run_section(ctx: SectionContext) -> Finding:
    """Раздел E5: F14, F15, T11, T12, факты ``e5.*`` и показатель МО ``log_spend_to_ndfl`` для сводки."""
    p = params(ctx.cfg)
    mo = ctx.data.mo
    names = display_names(ctx.data.territories)
    labels = short_names(ctx.data.territories)
    hints = context_hints(
        mo,
        HINT_TRAITS,
        z_thr=float(p["hint_z"]),
        max_hints=int(p["hint_max"]),
        annotations=_annotations(ctx.cfg),
    )
    sig = _signatures(ctx, p)
    home = _work_home(ctx, p, names, labels, hints)
    home["rho_recipients"] = ctx.facts[f"{SECTION_ID}.rho_ratio_recipients"].value
    _types(ctx, p, names)
    wh = home["wh"]
    usable = wh["usable"].to_numpy()
    indicators = pd.DataFrame(
        {
            "territory_id": mo.loc[usable, "territory_id"].astype("int32").to_numpy(),
            "log_spend_to_ndfl": np.log(mo.loc[usable, NDFL_RATIO].astype("float64")).to_numpy(),
        }
    )
    caveats = list(CAVEATS)
    if ctx.facts[f"{SECTION_ID}.capital_districts"].value:
        # Столица субъекта — район (Гатчинский у Ленинградской области): расстояние — до его центра.
        caveats[SUBURB_CAVEAT] += (
            " Столица субъекта по справочнику — район, а не город: {{e5.capital_districts}}; расстояние "
            "считается до центра района."
        )
    return ctx.finding(
        title=TITLE,
        summary_md=_summary(sig, home, p),
        indicators=indicators,
        indicator_labels={"log_spend_to_ndfl": f"ln(траты / доход 5-НДФЛ на жителя), {YEAR}"},
        caveats=[typo(c) for c in caveats],
    )
