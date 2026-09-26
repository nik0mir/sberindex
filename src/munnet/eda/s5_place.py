"""Раздел E5 разведки «Экономика места»: видна ли местная экономика в тратах и где зарабатывают, а где тратят.

Вопросы Q18 и Q19 (spec_final, Б.3 и Б.4):

- **F14** — ρ Спирмена внутри регионов между чертами места (доли занятости, горожане, возраст, плотность,
  доступность рынков, удалённость от столицы региона, ночёвки, зарплата) и чертами трат (уровень, CLR
  корзины, летний избыток, прирост доли маркетплейсов, рост). Связь, которая держится внутри регионов, —
  свойство места, а не карты регионов (Б.1, п. 9);
- **F15, T12** — траты по карте к доходу 5-НДФЛ на жителя против расстояния до столицы региона: пригороды,
  где живут, но работают в другом МО, и вахтовые МО, где работают приезжие; вывод о пригородах проверяется
  на вариантах правила (порог расстояния, только районы и округа), потери 5-НДФЛ из-за дефекта источника
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
from scipy.stats import mannwhitneyu

from munnet import style
from munnet.config import Config
from munnet.eda import stats
from munnet.eda.base import Finding, SectionContext
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
KEY_BOX = (0.45, 0.14)
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
    Trait("trade", "emp_sh_trade_transport_2023", "Стройка, торговля, транспорт",
          "больше торговли, стройки, транспорта"),
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
    Trait("nights", "nights_pc_2023", "Ночёвки в гостиницах на жителя", "больше гостиничных ночёвок",
          "много гостиничных ночёвок", "", "log1p"),
    Trait("wage", "wage_2023", "Средняя зарплата", "выше зарплата",
          "высокая зарплата", "низкая зарплата", "log"),
)
# Столбцы F14 — черты трат.
SPEND_TRAITS: tuple[Trait, ...] = (
    Trait("level", "log_level_2023", "Уровень\nтрат"),
    Trait("food", "clr_food_2023", "Продукты\n(CLR)"),
    Trait("cafe", "clr_cafe_2023", "Общепит\n(CLR)"),
    Trait("mp", "clr_marketplace_2023", "Маркет-\nплейсы\n(CLR)"),
    Trait("other", "clr_other_2023", "«Прочее»\n(CLR)"),
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
    и приезжим, а не по жителям. Колонка ``n_inner`` — сколько таких МО осталось в типе.
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
                "n": int(mask.sum()),
                "share_mo": float(mask.mean()) if len(mo) else float("nan"),
                "n_inner": int((mask & workplace).sum()),
                **base.loc[mask, cols].median().to_dict(),
                "examples": examples,
            }
        )
    columns = ["type", "label", "rule", "n", "share_mo", "n_inner", *cols, "examples"]
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
    pivot: pd.DataFrame, row_labels: Sequence[str], col_labels: Sequence[str], *, faint: float, vmax: float
) -> Figure:
    """Тепловая карта F14: ρ в клетках, расходящаяся шкала с нулём в центре, слабые связи светлые."""
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
      пригодных МО, ``suburb_mw_p`` — двусторонний тест Манна — Уитни.
    """
    mo = mo.reset_index(drop=True)
    ratio = mo[NDFL_RATIO].astype("float64")
    usable = mo["ndfl_ok"].astype(bool) & ratio.notna() & np.isfinite(ratio) & (ratio > 0)
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
    cmp = compare_groups(ratio, usable, suburb)
    inner = mo["is_inner_city"].astype(bool) if "is_inner_city" in mo else pd.Series(False, index=mo.index)
    return {
        "usable": usable,
        "n_usable": int(usable.sum()),
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
    points: pd.DataFrame, labeled: pd.DataFrame, median: float, suburb_note: str = "пригороды столиц регионов"
) -> Figure:
    """F15: траты к доходу 5-НДФЛ против расстояния до столицы региона, обе шкалы логарифмические.

    ``points`` — колонки ``x`` (км + 1), ``y`` (отношение), ``is_suburb``; ``labeled`` — ``x``, ``y``,
    ``label``, ``is_suburb`` для подписей краёв; ``median`` — горизонтальная линия медианы всех пригодных МО;
    ``suburb_note`` — что такое чёрные точки (подпись в углу). Подписанные МО не из пригородов — серые
    кольца, чтобы их нельзя было спутать с пригородами.
    """
    fig, ax = style.new_figure("full")
    other = points.loc[~points["is_suburb"]]
    sub = points.loc[points["is_suburb"]]
    ax.scatter(other["x"], other["y"], s=9, color=style.CONTEXT, linewidths=0, zorder=2)
    ax.scatter(sub["x"], sub["y"], s=12, color=style.ACCENT, linewidths=0, zorder=2)
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
    lines = ((typo(f"чёрные точки — {suburb_note}"), style.ACCENT), ("серые — остальные МО", style.TEXT2))
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
    ax.set_xlabel("Расстояние по дорогам до столицы региона, км")
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


def _trait_frame(traits: Sequence[Trait], prefix: str) -> pd.DataFrame:
    return pd.DataFrame(
        [(t.column, t.code, t.label.replace("\n", " "), t.where) for t in traits],
        columns=[prefix, f"{prefix}_code", f"{prefix}_label", f"{prefix}_where"],
    )


def _signatures(ctx: SectionContext, p: dict[str, Any]) -> dict[str, Any]:
    """F14 и факты о связях черт места и трат внутри регионов."""
    mo = ctx.data.mo
    sample = mo.loc[~mo["workplace_based"].astype(bool)] if p["exclude_workplace_based"] else mo
    matrix = within_region_matrix(sample, [t.column for t in PLACE_TRAITS], [t.column for t in SPEND_TRAITS])
    matrix = matrix.merge(_trait_frame(PLACE_TRAITS, "row"), on="row").merge(
        _trait_frame(SPEND_TRAITS, "col"), on="col"
    )
    strong_min, faint = float(p["fact_abs_rho"]), float(p["faint_abs_rho"])
    # Доли занятости — группы разделов context.okved_groups (у «бюджетников» — O, P, Q; у торговли — F, G, H):
    # с ориентирами Б.4 «без учёта регионов» по отдельным разделам A, O, G и всем МО сравнивать с поправкой.
    sample = (
        "МО без внутригородских; доли занятости — группы разделов ОКВЭД2"
        if p["exclude_workplace_based"]
        else "все МО; доли занятости — группы разделов ОКВЭД2"
    )
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
    strong = int((matrix["rho_within"].abs() >= strong_min).sum())
    ctx.fact("n_cells", len(matrix), "int", "клеток F14: черты места × черты трат")
    ctx.fact(
        "n_cells_strong", strong, "int", f"клеток F14 с |ρ| внутри регионов ≥ {style.fmt_num(strong_min, 1)}"
    )
    ctx.fact("strong_rho_min", strong_min, "num1", "порог «сильной» связи для фактов rho_within_*")
    n_sig = int(matrix["n_within"].max()) if len(matrix) else 0
    ctx.fact("n_signature", n_sig, "int", "наибольшее число МО в клетке F14")

    level = matrix.loc[(matrix["col_code"] == LEVEL_CODE) & ~matrix["row_code"].isin(p["headline_skip"])]
    cells = level.rename(columns={"row_code": "code", "row_where": "where", "rho_within": "rho"})
    head, chosen = signature_headline(cells, min_abs=float(p["headline_abs_rho"]), max_len=ctx.title_max)
    for side in ("pos", "neg"):
        cell = chosen.get(side)
        ctx.fact(
            f"sig_{side}_where",
            None if cell is None else cell["where"],
            "str",
            f"черта места с самой сильной {'прямой' if side == 'pos' else 'обратной'} связью с уровнем трат",
        )
        ctx.fact(
            f"sig_{side}_rho",
            np.nan if cell is None else cell["rho"],
            "rho",
            "ρ Спирмена внутри регионов с уровнем трат 2023 года",
        )
    title = ctx.headline(head.text, head.ok, f"F14 economy_signatures: {head.check}; {head.detail}")
    extra = _signature_extras(ctx, p, matrix, level, sample)

    order = (
        matrix.loc[matrix["col_code"] == LEVEL_CODE]
        .sort_values("rho_within", ascending=False, na_position="last")["row"]
        .tolist()
    )
    pivot = matrix.pivot(index="row", columns="col", values="rho_within").loc[
        order, [t.column for t in SPEND_TRAITS]
    ]
    labels = {t.column: t.label for t in PLACE_TRAITS}
    top = float(np.nanmax(np.abs(pivot.to_numpy()))) if np.isfinite(pivot.to_numpy()).any() else strong_min
    vmax = max(math.ceil(10 * top) / 10, strong_min)
    fig = plot_signatures(
        pivot, [labels[r] for r in order], [t.label for t in SPEND_TRAITS], faint=faint, vmax=vmax
    )
    n_lo = int(matrix["n_within"].min()) if len(matrix) else 0
    n_text = style.fmt_num(n_sig) if n_lo == n_sig else style.fmt_range(n_lo, n_sig)
    who = "МО без внутригородских" if p["exclude_workplace_based"] else "все МО"
    subtitle = typo(
        f"ρ Спирмена внутри регионов, {who}, 2023; n = {n_text}; |ρ| < {style.fmt_num(faint, 1)} — светлые"
    )
    pos, neg = chosen.get("pos"), chosen.get("neg")
    alt_parts = [f"связей с |ρ| от {style.fmt_num(strong_min, 1)} — {strong} из {len(matrix)}"]
    if pos is not None:
        alt_parts.insert(
            0, f"уровень трат выше, где {pos['where']} (ρ {style.fmt_rho(pos['rho'], sign=True)})"
        )
    if neg is not None:
        alt_parts.insert(
            1 if pos is not None else 0, f"ниже, где {neg['where']} (ρ {style.fmt_rho(neg['rho'])})"
        )
    alt = typo("Тепловая карта связей черт места и трат внутри регионов: " + "; ".join(alt_parts))
    data = matrix[
        ["row_code", "row_label", "col_code", "col_label", "rho_within", "n_within", "rho_all", "n_all"]
    ]
    ctx.save_figure(
        fig,
        fid="F14",
        slug="economy_signatures",
        title=title,
        subtitle=subtitle,
        alt=alt,
        data=data,
        check=head.check,
        source=style.join_sources(style.SOURCE_SBER, style.SOURCE_ROSSTAT),
    )
    return {"head": head, "chosen": chosen, "strong": strong, "n_cells": len(matrix), **extra}


def _signature_extras(
    ctx: SectionContext, p: dict[str, Any], matrix: pd.DataFrame, level: pd.DataFrame, sample: str
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
            f"ρ Спирмена {'без учёта регионов' if key == 'rho_all' else 'внутри регионов'} для mask_where",
        )
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
    return {"lead": lead.set_index("col_code"), "mask": mask}


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


# Тексты вывода о пригородах: заголовок F15 ({r} — «в N раза», {r2} — отношение с двумя знаками), фраза
# альт-текста и фраза «Что видно» ({k} — начало ключа факта «{{e5.»).
_SUBURB_TEXT: dict[str, tuple[str, str, str]] = {
    "higher": (
        "В пригородах траты к доходу по месту работы в {r} раза выше, чем в остальных МО",
        "в пригородах столиц регионов траты к доходу по месту работы в {r} раза выше, чем в остальных МО",
        "в пригородах столиц регионов отношение в {k}suburb_ratio}} раза выше, чем в остальных МО "
        "(p Манна — Уитни {k}suburb_mw_p}})",
    ),
    "higher_weak": (
        "В пригородах траты к доходу по месту работы выше в {r} раза, но различие ненадёжно",
        "в пригородах столиц регионов траты к доходу по месту работы выше, но различие ненадёжно",
        "в пригородах столиц регионов отношение выше в {k}suburb_ratio}} раза, но различие ненадёжно "
        "(p Манна — Уитни {k}suburb_mw_p}})",
    ),
    "lower": (
        "В пригородах траты к доходу по месту работы ниже, чем в остальных МО: {r2}",
        "в пригородах столиц регионов траты к доходу по месту работы ниже, чем в остальных МО",
        "в пригородах столиц регионов отношение ниже, чем в остальных МО: {k}suburb_ratio}} "
        "(p Манна — Уитни {k}suburb_mw_p}})",
    ),
    "lower_weak": (
        "В пригородах траты к доходу по месту работы ниже ({r2}), но различие ненадёжно",
        "в пригородах столиц регионов траты к доходу по месту работы ниже, но различие ненадёжно",
        "в пригородах столиц регионов отношение ниже ({k}suburb_ratio}} к остальным МО), но различие "
        "ненадёжно (p Манна — Уитни {k}suburb_mw_p}})",
    ),
    "similar": (
        "В пригородах траты к доходу по месту работы почти как в остальных МО: {m1} и {m2}",
        "в пригородах столиц регионов траты к доходу по месту работы почти как в остальных МО",
        "в пригородах столиц регионов отношение почти как в остальных МО: медианы {k}suburb_median}} "
        "и {k}other_median}} (p Манна — Уитни {k}suburb_mw_p}})",
    ),
    "unknown": (
        "Пригороды не сравнить с остальными МО по тратам к доходу: нет пригодных МО",
        "пригороды столиц регионов не с чем сравнить: нет пригодных МО",
        "сравнить пригороды столиц регионов с остальными МО не удалось",
    ),
}


def _suburb_text(
    verdict: str, part: int, ratio: float, medians: tuple[float, float] = (np.nan, np.nan)
) -> str:
    """Текст вывода ``verdict``: ``part`` 0 — заголовок, 1 — альт-текст, 2 — «Что видно» с ключами фактов.

    ``medians`` — медианы отношения у пригородов и у остальных МО (для вывода «почти как»).
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


def _work_home(ctx: SectionContext, p: dict[str, Any], names: pd.Series, hints: pd.Series) -> dict[str, Any]:
    """F15, T12 и факты о тратах к доходу по месту работы."""
    cfg_eda = ctx.cfg["eda"]
    mo = ctx.data.mo.reset_index(drop=True)
    types = cfg_eda["economy_types"]
    if p["suburb_type"] not in types:
        raise KeyError(f"в eda.economy_types нет типа {p['suburb_type']!r}: по нему считаются факты suburb_*")
    wh = work_home(
        mo,
        suburb_rule=str(types[p["suburb_type"]]["rule"]),
        recipients_ok=ctx.cfg["context"]["ndfl"]["recipients_ratio_ok"],
        quantiles=QUANTILES,
    )
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
            ctx.fact(
                f"{key}_{qk}",
                dist.loc[name, qk],
                "num2",
                f"{QUANTILE_NAMES.get(q, qk)} {name} по {int(dist.loc[name, 'n'])} МО",
            )
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
    rule = str(types[p["suburb_type"]]["rule"])
    ctx.fact("suburb_n", wh["suburb_n"], "int", f"пригодных МО по правилу пригорода «{rule}»")
    thr = float(cfg_eda["rejection"]["s6_suburb_ratio_min"])
    ctx.fact("suburb_ratio_min", thr, "num1", "порог критерия отказа С6 для suburb_ratio (eda.rejection)")
    sens = suburb_sensitivity(
        mo,
        mo[NDFL_RATIO].astype("float64"),
        wh["usable"],
        suburb_variants(rule, DIST, p["suburb_sens_km"], p["suburb_district_types"]),
    )
    km_list = p["suburb_sens_km"] if rule_threshold(rule, DIST) is not None else []
    _sensitivity_facts(ctx, sens, km_list)
    ctx.fact(
        "suburb_ratio",
        wh["suburb_ratio"],
        "num2",
        "медиана spend_to_ndfl_2023 пригородов / медиана остальных пригодных МО",
    )
    ctx.fact(
        "suburb_mw_p", wh["suburb_mw_p"], "p", "двусторонний тест Манна — Уитни: пригороды против остальных"
    )
    ctx.fact("suburb_median", wh["suburb_median"], "num2", "медиана spend_to_ndfl_2023 пригородов")
    ctx.fact("other_median", wh["other_median"], "num2", "медиана spend_to_ndfl_2023 остальных пригодных МО")

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
        "МО с наибольшим числом гостиничных ночёвок на жителя, 2023; без внутригородских территорий",
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

    alpha = float(p["mw_alpha"])
    medians = (wh["suburb_median"], wh["other_median"])
    verdict = suburb_verdict(wh["suburb_ratio"], wh["suburb_mw_p"], thr, alpha)
    head = _suburb_headline(wh["suburb_ratio"], wh["suburb_mw_p"], thr, alpha, medians)
    title = ctx.headline(head.text, head.ok, f"F15 work_vs_home: {head.check}; {head.detail}")
    pts = mo.loc[usable & mo[DIST].notna().to_numpy()].copy()
    pts["x"] = pts[DIST] + 1
    pts["y"] = pts[NDFL_RATIO]
    pts["is_suburb"] = wh["suburb"].to_numpy()[pts.index]
    pts["name_short"] = pts["territory_id"].map(names).fillna(pts["name"])
    high = pts.nlargest(int(p["label_high"]), "y")
    low = pts.nsmallest(int(p["label_low"]), "y")
    labeled = pd.concat([high, low]).drop_duplicates("territory_id")
    labeled = labeled.assign(label=labeled["name_short"])
    median = float(dist.loc[NDFL_RATIO, quantile_key(MEDIAN_Q)])
    km = rule_threshold(rule, DIST)
    note = "пригороды столиц регионов" + ("" if km is None else f" (до {style.fmt_num(km)} км по дорогам)")
    fig = plot_work_home(pts, labeled, median, note)
    subtitle = typo(
        f"Траты / доход 5-НДФЛ (по месту работы), {YEAR}; n = {style.fmt_num(len(pts))} МО; "
        "обе оси логарифмические"
    )
    alt = typo(
        f"Точечный график: {_suburb_text(verdict, 1, wh['suburb_ratio'], medians)}; медиана отношения трат "
        f"к доходу по месту работы у всех МО — {style.fmt_num(median, 2)}"
    )
    data = pts[["territory_id", "name_short", "region_name", DIST, NDFL_RATIO, "is_suburb"]].assign(
        labeled=pts["territory_id"].isin(labeled["territory_id"])
    )
    ctx.save_figure(
        fig,
        fid="F15",
        slug="work_vs_home",
        title=title,
        subtitle=subtitle,
        alt=alt,
        data=data,
        check=head.check,
        source=style.join_sources(style.SOURCE_SBER, style.SOURCE_FNS),
    )

    table = work_home_table(dist, edges, QUANTILES)
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
    return {"wh": wh, "head": head, "verdict": verdict, "sens": sens, "thr": thr, "dist_weak": dist_weak}


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
            ctx.fact(f"n_type_{row.type}", row.n, "int", f"МО типа «{row.label}»: {row.rule}{inner}")
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
    ctx.save_table(
        table,
        tid="T11",
        slug="economy_types",
        title=typo("Гипотезы о типах местной экономики: медианы черт трат против всех МО"),
        md_formats={
            "rule": _code,
            "share_mo": _pct,
            f"level_to_region_{year}": _num2,
            **{f"sh_{c}_{year}": _pct for c in ("food", "other", "cafe", "marketplace")},
            "summer_excess": _pct_signed,
            WAGE_RATIO: _num2,
            NDFL_RATIO: _num2,
            "mp_pp_change": _pp,
        },
        md_labels={
            "type": "Код",
            "label": "Тип",
            "rule": "Правило",
            "n": "МО",
            "share_mo": "Доля МО",
            "n_inner": "Из них внутригородских",
            f"level_to_region_{year}": f"Уровень к региону, {year}",
            f"sh_food_{year}": "Продукты",
            f"sh_other_{year}": "«Прочее»",
            f"sh_cafe_{year}": "Общепит",
            f"sh_marketplace_{year}": "Маркетплейсы",
            "summer_excess": "Летний избыток",
            WAGE_RATIO: "Траты / зарплата",
            NDFL_RATIO: "Траты / доход 5-НДФЛ",
            "mp_pp_change": "Прирост доли маркетплейсов",
            "examples": "Примеры",
        },
    )
    return table


def _fact_ref(key: str) -> str:
    """Ссылка на факт раздела в тексте: «{{e5.key}}»."""
    return "{{" + f"{SECTION_ID}.{key}" + "}}"


def _signature_text(sig: dict[str, Any], min_abs: float, weak_max: float) -> str:
    """«Что видно» по рис. 14: уровень, корзина, слабые столбцы, маскировка регионом.

    Слова выбираются по знаку и силе чисел, сами числа — только ссылки на факты. ``min_abs`` — порог
    названной связи, ``weak_max`` — порог «сильной» связи (столбцы слабее него названы «связаны слабо»).
    """
    f = _fact_ref
    chosen = sig["chosen"]
    if "pos" in chosen and "neg" in chosen:
        text = (
            f"Внутри регионов траты выше там, где {f('sig_pos_where')} (ρ = {f('sig_pos_rho')}), и ниже, "
            f"где {f('sig_neg_where')} (ρ = {f('sig_neg_rho')})"
        )
    elif chosen:
        side = next(iter(chosen))
        word = "выше" if side == "pos" else "ниже"
        text = f"Внутри регионов траты {word} там, где {f(f'sig_{side}_where')} (ρ = {f(f'sig_{side}_rho')})"
    else:
        text = "Внутри регионов уровень трат почти не связан с чертами места"
    text += (
        f"; связей не слабее |ρ| = {f('strong_rho_min')} — {f('n_cells_strong')} из {f('n_cells')} (рис. 14)."
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
    if weak:
        subjects = " и ".join(WEAK_TEXT[c] for c in weak)
        values = " и ".join(f(f"max_abs_{c}") for c in weak)
        text += f" С чертами места слабо связаны {subjects}: |ρ| не больше {values}."
    mask = sig["mask"]
    if mask is not None and abs(mask["rho_within"] - mask["rho_all"]) >= min_abs:
        a, w = float(mask["rho_all"]), float(mask["rho_within"])
        if np.sign(a) != np.sign(w):
            text += (
                f" Карта регионов маскирует свойства места: где {f('mask_where')}, без учёта регионов траты "
                f"{'выше' if a > 0 else 'ниже'} (ρ = {f('mask_rho_all')}), а внутри регионов — "
                f"{'выше' if w > 0 else 'ниже'} (ρ = {f('mask_rho_within')})."
            )
        else:
            text += (
                f" Карта регионов {'маскирует' if abs(w) > abs(a) else 'усиливает'} связи места: где "
                f"{f('mask_where')}, без учёта регионов ρ = {f('mask_rho_all')}, внутри регионов — "
                f"{f('mask_rho_within')}."
            )
    return text


def _sensitivity_text(sens: pd.DataFrame, verdict: str, thr: float) -> str:
    """Фраза о том, зависит ли вывод о пригородах от правила: слова — по числам вариантов."""
    f = _fact_ref
    valid = sens.dropna(subset=["ratio"])
    if len(valid) < 2:
        return ""
    span = f"от {f('suburb_sens_min')} до {f('suburb_sens_max')}"
    how = (
        f"в {f('suburb_sens_n')} вариантах правила (порог от {f('suburb_sens_km_min')} до "
        f"{f('suburb_sens_km_max')} км по дорогам; все МО или только районы и округа, без городских "
        "округов-спутников)"
    )
    lo, hi = float(valid["ratio"].min()), float(valid["ratio"].max())
    if verdict != "higher" and hi < thr:
        return (
            f" Вывод не зависит от правила пригорода: {how} отношение медиан — {span}, ниже порога "
            f"критерия отказа С6 ({f('suburb_ratio_min')})."
        )
    if verdict == "higher" and lo >= thr:
        return f" Вывод устойчив к правилу пригорода: {how} отношение медиан — {span}."
    return f" Вывод зависит от правила пригорода: {how} отношение медиан — {span}."


def _summary(sig: dict[str, Any], home: dict[str, Any], p: dict[str, Any]) -> str:
    """«Что видно» и «Что это значит для сюжета»: слова — по знаку чисел, числа — только {{ключи}} фактов."""
    f = _fact_ref
    verdict, thr = home["verdict"], home["thr"]
    f14 = _signature_text(sig, float(p["headline_abs_rho"]), float(p["fact_abs_rho"]))
    sub = _suburb_text(verdict, 2, home["wh"]["suburb_ratio"])
    dist_word = "почти не зависит" if home["dist_weak"] else "зависит"
    f15 = (
        f"В типичном МО траты по карте в {f('spend_to_ndfl_median')} раза больше дохода 5-НДФЛ на жителя "
        f"(10-й и 90-й перцентили — {f('spend_to_ndfl_p10')} и {f('spend_to_ndfl_p90')}); {sub}."
        + _sensitivity_text(home["sens"], verdict, thr)
        + f" Внутри регионов отношение {dist_word} от расстояния до столицы региона (ρ = "
        f"{f('rho_ndfl_dist_within')}); между типами T11 медиана ниже всего у типа "
        f"«{f('ndfl_type_min_label')}» ({f('ndfl_type_min')}), выше всего — у типа "
        f"«{f('ndfl_type_max_label')}» ({f('ndfl_type_max')}). Самое высокое отношение — "
        f"{f('ndfl_high_top')}: вероятно, жители работают у работодателей из других МО или живут на пенсии "
        "и выплаты, которых нет "
        f"в 5-НДФЛ; самое низкое — {f('ndfl_low_top')}: вероятно, в этих МО работают приезжие, в том числе "
        "вахтой (гипотезы, рис. 15 и T12). "
        f"Отношение не считается у {f('n_recip_jump')} МО, где число получателей 5-НДФЛ снято как дефект "
        f"источника (единицы вместо тысяч), и у {f('n_ndfl_no_level')} МО с неполным годом трат."
    )
    if verdict == "higher":
        edges = (
            "рёбра «работа — жительство» — дороги от пригородов к центрам занятости: пригороды выделяются "
            "и по этому показателю"
        )
    else:
        edges = (
            "критерий отказа С6 по пригородам не выполнен: пригороды по этому показателю не выделяются, "
            "поэтому рёбра «работа — жительство» от пригородов к центрам занятости им не подкреплены; "
            "различие «центры занятости — периферия» между типами местной экономики (T11) может остаться "
            "слоем"
        )
    meaning = (
        f"Для сюжета С6 «Где зарабатывают и где тратят» пригодны {f('n_ndfl_usable')} МО с надёжным "
        f"5-НДФЛ; признак узла — ln(траты / доход 5-НДФЛ), в сводке `log_spend_to_ndfl`; {edges}. "
        "Черты места, чья связь с тратами держится внутри регионов, годятся для объяснения типов на этапе 2: "
        "это свойство места, а не карты регионов, и признаки лучше брать относительно региона. Типы T11 — "
        "гипотезы для проверки, а не классификация."
    )
    return typo(f"**Что видно.** {f14} {f15}\n\n**Что это значит для сюжета.** {meaning}")


CAVEATS: tuple[str, ...] = (
    "Доход 5-НДФЛ, занятость и зарплата Росстата учитываются по месту работодателя, а траты СберИндекса — "
    "по жителям МО. Поэтому высокое отношение трат к доходу, вероятно, значит, что жители работают у "
    "работодателей из других МО или живут на пенсии и выплаты, которых нет в 5-НДФЛ, а низкое — что в МО "
    "работают приезжие (вахта) или крупный работодатель платит сотрудникам из других МО. Это гипотезы.",
    "Траты — оценка СберИндекса средних безналичных трат жителя МО; знаменатель (все жители или держатели "
    "карт) и модель привязки трат к МО не раскрыты. Траты жителей вне своего МО (поездки, онлайн) по смыслу "
    "входят, траты приезжих в курортном МО не видны: курортный тип T11 задан ночёвками Росстата, а не "
    "тратами. Отношение трат к доходу — сравнительная мера, а не доля расходов в доходе.",
    "Занятость и зарплата Росстата — без малого бизнеса; доли разделов ОКВЭД2 считаются от работников "
    "крупных и средних организаций. Скрытый раздел — пропуск, а не ноль: доли занятых в добыче, обработке "
    "и сельском хозяйстве известны не во всех МО, эти строки рис. 14 посчитаны на меньшей выборке.",
    "Внутригородские территории Москвы и Петербурга не входят в рис. 14 и в отношения к доходу: показатели "
    "по месту работы на их жителя не считаются.",
    "Связь внутри регионов — не причина: она может идти через третью черту, например зарплату; рост трат — "
    "номинальный.",
)


def run_section(ctx: SectionContext) -> Finding:
    """Раздел E5: F14, F15, T11, T12, факты ``e5.*`` и показатель МО ``log_spend_to_ndfl`` для сводки."""
    p = params(ctx.cfg)
    mo = ctx.data.mo
    names = short_names(ctx.data.territories)
    hints = context_hints(
        mo,
        HINT_TRAITS,
        z_thr=float(p["hint_z"]),
        max_hints=int(p["hint_max"]),
        annotations=_annotations(ctx.cfg),
    )
    sig = _signatures(ctx, p)
    home = _work_home(ctx, p, names, hints)
    _types(ctx, p, names)
    wh = home["wh"]
    usable = wh["usable"].to_numpy()
    indicators = pd.DataFrame(
        {
            "territory_id": mo.loc[usable, "territory_id"].astype("int32").to_numpy(),
            "log_spend_to_ndfl": np.log(mo.loc[usable, NDFL_RATIO].astype("float64")).to_numpy(),
        }
    )
    return ctx.finding(
        title=TITLE,
        summary_md=_summary(sig, home, p),
        indicators=indicators,
        indicator_labels={"log_spend_to_ndfl": f"ln(траты / доход 5-НДФЛ на жителя), {YEAR}"},
        caveats=[typo(c) for c in CAVEATS],
    )
