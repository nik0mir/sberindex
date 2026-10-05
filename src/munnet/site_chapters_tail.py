"""Главы 6, 7, 9 и 10 лендинга (§3.8, §3.10, §3.11 и «Источники» ``docs/landing_spec.md``): HTML и статичные
SVG, собранные при сборке, — страница читается без JS.

Как и в ``site_chapters``: заголовки и тексты выводов — только из ``story`` (слоты этапа 5,
``site.headlines``, ``site.roles``, ``site.build.texts``); здесь — нейтральный интерфейс ``UI`` (подписи осей,
ключи, описание
шагов метода), он проходит тот же линт, что шаблон (``landing.lint_templates``). Числа — из ``checks.json``,
``methods.json`` и ``story``; ничего не пересчитывается. Карты ячеек — точки в координатах карты экрана 0
(``geo``: центры ячеек по ``hq``, ``hr``); общий слой всех ячеек задан один раз (``<path id="cells-base">``)
и переиспользуется через ``<use>`` — четыре маленькие карты не повторяют 1945 точек.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from munnet import site_chapters as SC
from munnet import style as mstyle
from munnet.site_chapters import (
    ACCENT,
    GREY,
    INK2,
    VARIANT_WORDS,
    Esc,
    _e,
    _f,
    _fig_mark,
    _section,
    _svg,
    _table,
    _text,
    _wrap,
)
from munnet.site_chapters import UI as SC_UI

INK = "#1d1d1d"
BASE = "#cdc8bd"  # фон маленьких карт: все ячейки
RIVAL_BASE = "#ebe7df"  # фон пары карт главы 3: ячейки без типа (светлее самой светлой группы соперника)

UI: dict[str, str] = {
    # глава 6 (§3.8)
    "comp_read": (
        "Каждому муниципалитету подобраны четыре набора по {k} территорий. Сверяли, насколько изменение "
        "розничного оборота Росстата на жителя 2023 → 2024 у самого муниципалитета отличается от медианы его "
        "набора за тот же период. Чем меньше ошибка, тем ближе набор к муниципалитету."
    ),
    "set_A": "похожие по корзине трат из других регионов, тип совпадает",
    "set_B": "соседи по своему региону",
    "set_C": "случайные того же размера из других регионов",
    "set_D": "похожие по корзине трат из других регионов, без учёта типа",
    "short_A": "похожие, тип совпадает",
    "short_B": "соседи по своему региону",
    "short_C": "случайные того же размера",
    "short_D": "похожие, без учёта типа",
    "comp_title": "Медианная ошибка сверки, {n}",
    "comp_axis": "медианная ошибка, логарифм оборота на жителя (0,05 ≈ 5%)",
    "comp_rel": "цель относительно своего региона",
    "comp_abs": "цель без поправки на регион",
    "comp_aria": "Медианная ошибка сверки у четырёх наборов: {items}",
    "diff_title": "Разница медианных ошибок двух наборов, 95% интервал",
    "diff_axis": "левее нуля — у первого набора ошибка меньше",
    "diff_aria": "Разница медианных ошибок: {items}",
    "ex_title": "Пример: {name}, {region}",
    "ex_note": (
        "Муниципалитет с медианной ошибкой набора «{set}» — средний случай. "
        "Вывод сделан по всем муниципалитетам, пример его иллюстрирует."
    ),
    "ex_map_aria": "{name} и два его набора на карте: {n_p} территорий из других регионов и {n_b} соседей",
    "ex_err_aria": (
        "Ошибка сверки у муниципалитета {name} по четырём наборам рядом с медианой по всем муниципалитетам"
    ),
    "ex_key_self": "муниципалитет примера",
    "ex_key_p": "{set}: {n} территорий",
    "ex_key_b": "соседи по своему региону: {n}",
    "ex_mo": "у этого муниципалитета",
    "ex_all": "медиана по всем",
    "ex_open": "Открыть карточку муниципалитета",
    "roles_title": "Кому и зачем",
    "comp_col_set": "Набор",
    "comp_col_rel": "Медианная ошибка",
    "comp_col_abs": "Без поправки на регион",
    "comp_col_ex": "У муниципалитета примера",
    "diff_col": "Разница",
    "diff_col_v": "Оценка",
    "diff_col_ci": "95% интервал",
    "alt_comp": "Ошибки наборов таблицей",
    "alt_ex": "Территории примера таблицей",
    "ex_col_set": "Набор",
    "ex_col_name": "Территория",
    "ex_col_km": "км по прямой",
    "src_comp": (
        "Источник: СберИндекс (CC BY-SA 4.0); оборот розничной торговли — Росстат, БД ПМО в обработке "
        "«Если быть точным» (CC BY 4.0), без малого бизнеса; расчёт «Корзина и регион»."
    ),
    # глава 3: карта-соперник T5 (порция 6c, §4.3: при T5 ≠ confirmed)
    "rv_title": "Типы и самое близкое к ним деление без типов на одной раскладке",
    "rv_types": "Типы",
    "rv_rival": "Без типов: «{label}», {n}",
    "rv_sized": (
        "Муниципалитеты выстроены по признаку и нарезаны на группы тех же размеров, что типы: "
        "1 — {lo}, {k} — {hi}."
    ),
    "rv_lo_up": "наименьшие значения",
    "rv_hi_up": "наибольшие",
    "rv_lo_down": "наибольшие значения",
    "rv_hi_down": "наименьшие",
    "rv_cap": (
        "Чем ближе две карты, тем меньше типы добавляют к простому делению. Совпадение типов с этим "
        "делением — AMI {ami} (0 — как у случайного деления, 1 — полное совпадение); это наибольший AMI "
        "среди делений без типов. Цвет справа — номер группы этого деления. Самые светлые ячейки — "
        "муниципалитеты без типа."
    ),
    "rv_aria_types": "Карта типов: {items}",
    "rv_aria_rival": "Карта деления «{label}»: {items}",
    "rv_group": "группа {g}",
    "rv_col_type": "Тип",
    "alt_rv": "Сколько муниципалитетов каждого типа в каждой группе",
    # глава 7 (§3.10)
    "limits_kicker": "Чего данные не показывают",
    "r1_title": "Тип при других вариантах расчёта",
    "r1_same": "тот же тип у {pct} муниципалитетов с типом ({same} из {n})",
    "r1_ari": "ARI с основным расчётом — {ari}",
    "r1_key_same": "тот же тип, что в основном расчёте",
    "r1_key_diff": "другой тип",
    "r1_aria": "{variant}: {same}",
    "r1_unstable": "Выводы, которые меняются в этих вариантах, на странице помечены: «{label}».",
    "r1_col_var": "Вариант расчёта",
    "r1_col_same": "Тот же тип",
    "r1_col_ari": "ARI (0 — как у случайного деления, 1 — полное совпадение)",  # 6o (check-ux 05.10)
    "r1_ari_first": " (0 — как у случайного деления, 1 — полное совпадение)",
    "alt_r1": "Варианты расчёта таблицей",
    "limits_list": "Ограничения данных",
    "lim_fragile": "Хрупкость",
    "lim_visitors": "Приезжие",
    "lim_regions": "Регионы",
    "lim_untyped": "Без типа",
    "lim_nominal": "Рубли",
    "lim_workplace": "Место работы",
    "lim_scope": "Охват",
    # глава 9 (§3.11)
    "m_s1_h": "Траты жителей",
    "m_s1": (
        "Безналичные траты жителей {n_mo} муниципалитетов по месяцам 2023–2024 годов: итог и пять категорий."
    ),
    "m_s2_h": "Относительно своего региона",
    "m_s2": (
        "Доли корзины и уровень трат сравниваются с типичным муниципалитетом своего региона: так снят "
        "региональный фон."
    ),
    "m_s3_h": "Сеть сходства трат",
    "m_s3": (
        "Муниципалитеты связаны, если их корзины одинаково отличаются от своих регионов: {n_edges} связей "
        "между {n_nodes} муниципалитетами. Правила связей сравнивались между собой."
    ),
    "m_s4_h": "Типы",
    "m_s4": (
        "{n_types} — {method}. Выбраны по правилу, записанному до расчётов, из {n_cand} сочетаний метода "
        "и числа типов; качество — шесть индексов (SW, CH, S_Dbw, AVI, AVU, MQ)."
    ),
    "m_s5_h": "Смены типа и проверки",
    "m_s5": (
        "Тип считается по {n_windows} скользящим 12-месячным окнам. Выводы сверены с плацебо — теми же "
        "расчётами на перемешанном времени — и с вариантами расчёта."
    ),
    "m_steps_aria": "Пять шагов расчёта",
    "m_report": "отчёт, раздел {n}",
    "m_report_many": "отчёт, разделы {n}",
    "m_repro_h": "Воспроизводимость",
    "m_repro": "Все этапы с загрузкой данных запускает одна команда:",
    "m_repro_meta": "seed {seed} · сборка из коммита {sha} · отпечаток результатов {facts}",
    "m_prereg_h": "Что записано до расчётов",
    "m_repo": "Репозиторий с кодом и отчётом",
    "m_report_all": "методологический отчёт",
    "m_table_h": "Методы и индексы качества",
    "m_table_note": (
        "Строка — лучший по правилу выбора кандидат метода; методы без допустимых кандидатов ({n_none}) "
        "есть в полной таблице. Под значением — z: на сколько стандартных "
        "отклонений индекс лучше, чем у случайных меток того же размера."
    ),
    "m_sort_hint": "Нажмите на заголовок столбца, чтобы отсортировать.",
    "m_table_all": "Все кандидаты ({n})",
    "m_table_aria": "Методы кластеризации и индексы качества",
    "m_col_method": "Метод",
    "m_col_family": "Что видит",
    "m_col_k": "K",
    "m_col_ok": "Допустим",
    "m_final": "итог",
    "m_yes": "да",
    "m_no": "нет",
    "m_better_max": "больше — лучше",
    "m_better_min": "меньше — лучше",
    "m_gloss_h": "Индексы",
    # глава 10
    "src_downloads": "Выгрузки (CSV, UTF-8, CC BY-SA 4.0)",
    "src_code": "Код — лицензия MIT. Производные данные страницы и выгрузки — CC BY-SA 4.0.",
    "src_tools": "Программы и шрифты страницы",
    # таблица всех муниципалитетов (порция 6c): текстовая альтернатива карте
    "all_kicker": "Текстовая альтернатива карте",
    "all_title": "Все муниципалитеты таблицей",
    "all_read": (
        "У каждого муниципалитета — регион, тип, устойчивость типа и тип по годам. Название открывает "
        "карточку. "
        "Одинаковые названия бывают в разных регионах: их различает регион."
    ),
    "all_note": (
        "Устойчивость: «тип не зависит от варианта расчёта» — тот же тип во всех вариантах расчёта "
        "и повторах расчёта с другим случайным стартом. Тип, итог 2023–2024 — по всем 24 месяцам сразу; "
        "2023 и 2024 — тип по каждому году отдельно. У районов Москвы и Петербурга тип и годы — города "
        "целиком."
    ),
    "all_nojs": "Таблица строится в браузере. Без JavaScript её можно скачать одним файлом:",
    "all_nojs_tail": "UTF-8, открывается в Excel.",
    "all_caption": "Муниципалитеты: название, регион, тип, устойчивость типа, тип 2023 и 2024 годов",
    "all_q": "Название",
    "all_q_ph": "Начало названия",
    "all_region": "Регион",
    "all_type": "Тип",
    "all_any_region": "Все регионы",
    "all_any_type": "Все типы",
    "all_no_type": "Нет типа",
    "all_c_name": "Название",
    "all_c_region": "Регион",
    "all_c_type": "Тип, итог 2023–2024",
    "all_c_flag": "Устойчивость типа",
    "all_c_23": "2023",
    "all_c_24": "2024",
    "all_prev": "← Назад",
    "all_next": "Дальше →",
}

METHOD_WORDS = {
    "hybrid": "гибрид: место в сети сходства трат и признаки места",
    "spectral": "спектральная кластеризация сети",
    "leiden": "Лейден",
    "louvain": "Лувен",
    "kmeans": "k-средних",
    "kmeans_joint": "k-средних на сети и признаках",
    "gmm": "смесь нормальных распределений",
    "ward": "иерархическая, Уорд",
    "shalileh_mirkin": "Шалилех — Миркин",
}
FAMILY_WORDS = {
    "graph": "сеть",
    "features": "признаки",
    "attributed": "сеть и признаки",
    "fusion": "сеть и признаки",
}
ICVI_GLOSS = {
    "sw": "SW — силуэт: насколько муниципалитет ближе к своему типу, чем к соседнему, в признаках",
    "ch": "CH — Калински — Харабаш: разброс между типами к разбросу внутри типов, в признаках",
    "s_dbw": "S_Dbw — разброс внутри типов плюс плотность между ними, в признаках",
    "avi": "AVI — доля связей сети внутри своего типа, в среднем по типам",
    "avu": "AVU — насколько связаны между собой пары типов в сети",
    "mq": "MQ — модулярность: связей внутри типов больше, чем при случайной сети с теми же степенями",
    "anui": "ANUI — сводка AVI и AVU",
    "ok": "Допустим — кандидат прошёл пороги правила выбора, записанного до расчётов (docs/clustering.md)",
}
SOURCES = [
    (
        "СберИндекс",
        "безналичные траты жителей муниципалитетов, индекс доступности рынков, автодорожные связи, "
        "справочник границ",
        "CC BY-SA 4.0",
        "https://sberindex.ru/ru/research/data-sense-opisanie-nabora-dannikh-khakatona-sberindeksa-po-munitsipalnim-dannim",
    ),
    (
        "Росстат, база данных показателей муниципальных образований в обработке «Если быть точным»",
        "оборот розницы и общепита, население, занятость и зарплаты (без малого бизнеса)",
        "CC BY 4.0",
        "https://tochno.st/datasets/bdmo",
    ),
    (
        "ФНС, форма 5-НДФЛ в обработке «Если быть точным»",
        "доходы и число получателей",
        # порция 6f: на странице набора (проверено 02.10) лицензии нет — как в report.md, приложение Б
        "лицензия на странице набора не указана, цитирование по форме «Если быть точным»",
        "https://tochno.st/datasets/ndfl",
    ),
    (
        "OpenStreetMap",
        "полигоны границ справочника СберИндекса; © участники OpenStreetMap",
        "ODbL",
        "https://www.openstreetmap.org/copyright",
    ),
]
# программы и шрифты, которые загружает страница (порция 5a): название, для чего, лицензия, ссылка
TOOLS = [
    ("three.js 0.170.0", "объёмная карта первого экрана (файлы в vendor/)", "MIT", "https://threejs.org"),
    ("Golos Text", "шрифт текста (файлы Fontsource в vendor/fonts/)", "SIL Open Font License 1.1",
     "https://fontsource.org/fonts/golos-text"),
    ("Unbounded", "шрифт названия и крупных чисел (файлы Fontsource в vendor/fonts/)",
     "SIL Open Font License 1.1",
     "https://fontsource.org/fonts/unbounded"),
]  # fmt: skip
DOWNLOADS = [
    (
        "mo.csv",
        "все муниципалитеты: регион, тип, устойчивость типа, типы 2023 и 2024 годов, смена, причина, "
        "если типа нет",
    ),
    ("types.csv", "типы: название, число муниципалитетов, доля жителей"),
    ("flows.csv", "смены типа 2023 → 2024: всего и с согласием половин года"),
    ("README.txt", "описание выгрузок и лицензии"),
]


def ui_strings() -> list[str]:
    """Тексты интерфейса глав 6, 7, 9, 10 — для линта (``landing.lint_templates``)."""
    return (
        list(UI.values())
        + list(METHOD_WORDS.values())
        + list(FAMILY_WORDS.values())
        + list(ICVI_GLOSS.values())
        + [x for s in SOURCES for x in s[:3]]
        + [x for s in TOOLS for x in s[:3]]
        + [d for _, d in DOWNLOADS]
    )


def _num(x: float, d: int) -> str:
    return _f(x, d) if x is not None else "—"


def _signed(x: float, d: int = 3) -> str:
    s = _f(abs(x), d)
    return ("+" if x > 0 else "−" if x < 0 else "") + s


def _ticks(lo: float, hi: float, n: int = 4) -> list[float]:
    raw = (hi - lo) / max(n, 1)
    mag = 10 ** math.floor(math.log10(raw)) if raw > 0 else 1
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)
    t = math.ceil(lo / step - 1e-9) * step
    out = []
    while t <= hi + 1e-12:
        out.append(round(t, 10))
        t += step
    return out


def _dec(step: float) -> int:
    """Знаков после запятой, чтобы шаг шкалы (0,025, 0,01, 5) записывался точно."""
    return next((d for d in range(7) if abs(round(step * 10**d) - step * 10**d) < 1e-9), 6)


def _set_label(s: str, view: Mapping) -> str:
    base = UI[f"set_{s}"]
    if s == view.get("product_set") and view.get("set_name"):
        return f"{view['set_name']}: {base}"
    return base


# --- общий слой ячеек


def cells_defs(geo: Mapping | None) -> str:
    """Невидимая SVG с путём всех ячеек (точки нулевой длины): её переиспользуют маленькие карты."""
    if not geo or not geo.get("xy"):
        return ""
    return (
        '<svg class="defs" width="0" height="0" aria-hidden="true" focusable="false" '
        'style="position:absolute" xmlns="http://www.w3.org/2000/svg"><defs>'
        f'<path id="cells-base" d="{_dots(geo["xy"].values())}"/></defs></svg>'
    )


def _dots(pts: Any) -> str:
    """Точки нулевой длины одним путём, относительными переходами (короче абсолютных)."""
    out, px, py = [], 0, 0
    for x, y in sorted((round(x), round(y)) for x, y in pts):
        out.append(f"m{x - px} {y - py}h0" if out else f"M{x} {y}h0")
        px, py = x, y
    return "".join(out)


def _mini_map(geo: Mapping, layers: str, label: str, cls: str = "mini-map", base: str = BASE) -> str:
    w, h = geo["w"], geo["h"]
    use = f'<use href="#cells-base" stroke="{base}" stroke-width="8.5" stroke-linecap="round" fill="none"/>'
    body = use + layers
    return _svg(w, h, body, label, f"ch-svg {cls}")


# --- глава 3: карта-соперник T5 (порция 6c)

# Нейтральная последовательная шкала групп соперника — серые одного тона от светлого к тёмному: не цвета
# типов, чтобы группа не читалась как тип. Светлый конец заметно темнее фона ячеек без типа (BASE).
RIVAL_LIGHT, RIVAL_DARK = "#878d93", "#1e2328"  # светлый конец — контраст ≥ 3 : 1 к бумаге


def rival_ramp(k: int) -> list[str]:
    """``k`` оттенков от RIVAL_LIGHT к RIVAL_DARK (равные шаги по каналам sRGB)."""
    a = [int(RIVAL_LIGHT[i : i + 2], 16) for i in (1, 3, 5)]
    b = [int(RIVAL_DARK[i : i + 2], 16) for i in (1, 3, 5)]
    out = []
    for j in range(k):
        s = j / max(k - 1, 1)
        out.append("#" + "".join(f"{round(x + (y - x) * s):02x}" for x, y in zip(a, b, strict=True)))
    return out


def _groups_word(n: int) -> str:
    """«4 группы» — число и слово по правилу числительного."""
    a, b = n % 100, n % 10
    w = "групп" if 11 <= a <= 14 else "группа" if b == 1 else "группы" if 2 <= b <= 4 else "групп"
    return f"{n} {w}"


def rival_maps(rival: Mapping | None, story: Mapping, geo: Mapping | None, esc: Esc) -> str:
    """Малое множество из двух карт ячеек на одной раскладке (общий слой ``#cells-base``): слева — типы,
    справа — сильнейшее по AMI деление без типов (``rival``: ``label``, ``ami``, ``groups`` {id: группа},
    ``order`` — «+»/«-» у делений по одному признаку). Только при ``view.show_rival``; ничего
    не пересчитывает,
    таблица «тип × группа» — счёт тех же меток."""
    view = story["view"]
    if not rival or not geo or not view.get("show_rival") or not rival.get("groups"):
        return ""
    xy = geo["xy"]
    gr = {int(i): g for i, g in rival["groups"].items()}
    types = {int(i): int(t) for i, t in (rival.get("types") or {}).items()}
    order = [int(t) for t in view.get("legend_order") or sorted(set(types.values()))]
    keys = sorted(set(gr.values()), key=lambda g: (str(type(g)), g))
    ramp = dict(zip(keys, rival_ramp(len(keys)), strict=True))
    lay_t, lay_r, items_t, items_r = [], [], [], []
    for t in order:
        pts = [xy[i] for i, tt in types.items() if tt == t and i in xy]
        col = view["type_colors"].get(str(t), INK2)
        lay_t.append(
            f'<path d="{_dots(pts)}" stroke="{col}" stroke-width="8.5" stroke-linecap="round" fill="none"/>'
        )
        items_t.append(f"{SC._type_name(story, t)} — {_f(len(pts))}")
    for g in keys:
        pts = [xy[i] for i, gg in gr.items() if gg == g and i in xy]
        lay_r.append(
            f'<path d="{_dots(pts)}" stroke="{ramp[g]}" stroke-width="8.5" stroke-linecap="round" '
            'fill="none"/>'
        )
        items_r.append(f"{UI['rv_group'].format(g=g)} — {_f(len(pts))}")
    label = str(rival.get("label") or "")
    head_r = UI["rv_rival"].format(label=label, n=_groups_word(len(keys)))
    svg_t = _mini_map(geo, "".join(lay_t), UI["rv_aria_types"].format(items="; ".join(items_t)),
                      "mini-map rv-map", RIVAL_BASE)  # fmt: skip
    svg_r = _mini_map(geo, "".join(lay_r), UI["rv_aria_rival"].format(label=label, items="; ".join(items_r)),
                      "mini-map rv-map", RIVAL_BASE)  # fmt: skip
    key_t = (
        '<ul class="rv-key">'
        + "".join(f"<li>{_fig_mark(t, view)}{esc(SC._type_name(story, t))}</li>" for t in order)
        + "</ul>"
    )
    key_r = (
        '<ul class="rv-key rv-ramp">'
        + "".join(f'<li><i class="sw" style="background:{ramp[g]}"></i>{esc(str(g))}</li>' for g in keys)
        + "</ul>"
    )
    sized = ""
    if rival.get("order") in ("+", "-") and len(keys) > 1:
        lo, hi = UI["rv_lo_up"], UI["rv_hi_up"]
        if rival["order"] == "-":
            lo, hi = UI["rv_lo_down"], UI["rv_hi_down"]
        sized = f'<p class="note">{esc(UI["rv_sized"].format(lo=lo, k=keys[-1], hi=hi))}</p>'
    rows = []
    for t in order:
        ids = [i for i, tt in types.items() if tt == t and i in gr]
        rows.append(
            [f"{_fig_mark(t, view)}{esc(SC._type_name(story, t))}"]
            + [_f(sum(1 for i in ids if gr[i] == g)) for g in keys]
        )
    return (
        '<figure class="ch-fig rv-fig" id="rival">'
        f"<h3>{esc(UI['rv_title'])}</h3>"
        '<div class="rv-grid">'
        f'<div class="rv-card"><h4>{esc(UI["rv_types"])}</h4>{svg_t}{key_t}</div>'
        f'<div class="rv-card"><h4>{esc(head_r)}</h4>{svg_r}{key_r}{sized}</div>'
        "</div>"
        f'<p class="note rv-cap">{esc(UI["rv_cap"].format(ami=_f(float(rival["ami"]), 2)))}</p>'
        f'<figcaption class="source">{esc(SC_UI["src_sber"])}</figcaption>'
        + _table([UI["rv_col_type"], *[UI["rv_group"].format(g=g) for g in keys]], rows, esc, UI["alt_rv"])
        + "</figure>"
    )


# --- глава 6: сопоставимые территории (§3.8)


# Точечные графики главы 6: подпись строки — над своей линией, поэтому SVG узкая (W) и читается на 375 px.
W, X0, X1 = 380, 4, 336  # ширина, начало и конец шкалы; справа — подпись значения


def _axis_caption(s: str, w: float, y: float) -> list[str]:
    """Подпись оси у правого края шкалы; на узком холсте — переносом, чтобы не уйти за левый край."""
    lines = _wrap(s, int((w - 44) / 5.6))  # 10 единиц × 0,56 — средний знак
    return [_text(w - 44, y + j * 13, ln, "axis", "end") for j, ln in enumerate(lines)]


def comparable_svg(t7: Mapping, view: Mapping, ex_err: Mapping | None, W: float = W) -> tuple[str, float]:
    """Точечный график медианных ошибок наборов A, B, C, D: точка — цель относительно региона, кольцо —
    без поправки; шкала от нуля, строки — по возрастанию ошибки. Возвращает SVG и правый край шкалы
    (для графика примера — та же шкала)."""
    med, mab = t7["median_error"], t7.get("median_error_abs") or {}
    sets = sorted(med, key=lambda s: (med[s], s))
    vals = [v for v in [*med.values(), *mab.values(), *(ex_err or {}).values()] if v is not None]
    xmax = max(vals) * 1.08
    ticks = _ticks(0, xmax, 3)
    xmax = max(xmax, ticks[-1])
    sx = lambda v: X0 + v / xmax * ((W - 44) - X0)  # noqa: E731
    g, y = [], 0.0
    for s in sets:
        lines = _wrap(f"{s} · {_set_label(s, view)}", int(58 * W / 380))[:3]
        cls = "th" if s == view.get("product_set") else "lab"
        for j, ln in enumerate(lines):
            g.append(_text(X0, y + 11 + j * 13, ln, cls))
        yl = y + 13 * len(lines) + 9
        g.append(f'<line class="row-rule" x1="{X0}" x2="{(W - 44)}" y1="{yl:.1f}" y2="{yl:.1f}"/>')
        if mab.get(s) is not None:
            g.append(
                f'<circle cx="{sx(mab[s]):.1f}" cy="{yl:.1f}" r="4.5" fill="#fff" stroke="{INK2}" '
                'stroke-width="1.5"/>'
            )
        g.append(f'<circle cx="{sx(med[s]):.1f}" cy="{yl:.1f}" r="4.5" fill="{INK}"/>')
        g.append(_text(W, yl + 4, _num(med[s], 3), "val", "end"))
        y = yl + 10
    dec = _dec(ticks[1] - ticks[0]) if len(ticks) > 1 else 2
    for t in ticks:
        g.append(f'<line class="grid" x1="{sx(t):.1f}" x2="{sx(t):.1f}" y1="{y - 4:.1f}" y2="{y:.1f}"/>')
        lab = _num(t, dec)  # крайняя подпись у начала шкалы — от края, чтобы не выйти за холст
        g.append(_text(sx(t), y + 12, lab, "axis", "start" if sx(t) < len(lab) * 3 else "middle"))
    g += _axis_caption(UI["comp_axis"], W, y + 26)
    y += 13 * (len(_wrap(UI["comp_axis"], int((W - 44) / 5.6))) - 1)
    items = "; ".join(f"{s} — {_num(med[s], 3)}" for s in sets)
    return _svg(W, y + 32, "".join(g), UI["comp_aria"].format(items=items), "ch-svg comp dots"), xmax


def diff_rows(t7: Mapping) -> list[tuple[str, str, Mapping]]:
    """Строки разностей: соседи и случайные против набора продукта, затем вклад типа (D − A)."""
    p = t7.get("product") or "A"
    diffs = t7.get("diffs") or {}
    out = []
    for a, b in (("B", p), ("C", p), ("D", "A")):
        k = f"{a}-{b}"
        if k in diffs:
            out.append((a, b, {"rel": diffs[k], "abs": diffs.get(k + "_abs")}))
    return out


def diff_svg(rows: Sequence[tuple[str, str, Mapping]], view: Mapping, W: float = W) -> str:
    """Разности медианных ошибок с 95% интервалами у нуля: точка и отрезок — цель относительно региона,
    кольцо и отрезок ниже — без поправки."""
    if not rows:
        return ""
    vals = [0.0]
    for _, _, d in rows:
        for key in ("rel", "abs"):
            if d.get(key):
                vals += [d[key][0], *d[key][1]]
    span = max(abs(min(vals)), abs(max(vals))) * 1.12
    ticks = _ticks(-span, span, 4)
    lo, hi = min(ticks[0], -span), max(ticks[-1], span)
    sx = lambda v: X0 + (v - lo) / (hi - lo) * ((W - 44) - X0)  # noqa: E731
    g, y, zero_rows = [], 0.0, []
    for a, b, d in rows:
        head = _wrap(f"{a} − {b}: {UI['short_' + a]} − {UI['short_' + b]}", int(W / (11.5 * 0.56)))
        for j, ln in enumerate(head):
            g.append(_text(X0, y + 11 + j * 13, ln, "th"))
        yl = y + 24 + 13 * (len(head) - 1)
        zero_rows.append(yl)
        for key, dy, style in (("rel", 0, "dot"), ("abs", 9, "ring")):
            v = d.get(key)
            if not v:
                continue
            est, (c0, c1) = v[0], v[1]
            yy = yl + dy
            g.append(
                f'<line x1="{sx(c0):.1f}" x2="{sx(c1):.1f}" y1="{yy:.1f}" y2="{yy:.1f}" '
                f'stroke="{INK if style == "dot" else INK2}" stroke-width="{2 if style == "dot" else 1.4}"/>'
            )
            if style == "dot":
                g.append(f'<circle cx="{sx(est):.1f}" cy="{yy:.1f}" r="4" fill="{INK}"/>')
                g.append(_text(W, yy + 4, _signed(est), "val", "end"))
            else:
                g.append(
                    f'<circle cx="{sx(est):.1f}" cy="{yy:.1f}" r="3.6" fill="#fff" stroke="{INK2}" '
                    'stroke-width="1.4"/>'
                )
        y = yl + 20
    for yl in zero_rows:  # линия нуля — только в строках графика, не поверх подписей
        g.append(
            f'<line class="zero" x1="{sx(0):.1f}" x2="{sx(0):.1f}" y1="{yl - 8:.1f}" y2="{yl + 17:.1f}"/>'
        )
    dec = _dec(ticks[1] - ticks[0]) if len(ticks) > 1 else 3
    for t in ticks:
        g.append(_text(sx(t), y + 12, _signed(t, dec) if t else "0", "axis", "middle"))
    g += _axis_caption(UI["diff_axis"], W, y + 26)
    y += 13 * (len(_wrap(UI["diff_axis"], int((W - 44) / 5.6))) - 1)
    items = "; ".join(
        f"{a} − {b}: {_signed(d['rel'][0])}, интервал {_signed(d['rel'][1][0])}…{_signed(d['rel'][1][1])}"
        for a, b, d in rows
        if d.get("rel")
    )
    return _svg(W, y + 32, "".join(g), UI["diff_aria"].format(items=items), "ch-svg diff dots")


def example_err_svg(t7: Mapping, ex_err: Mapping, xmax: float, name: str, W: float = W) -> str:
    """Ошибка у муниципалитета примера (акцент) рядом с медианой по всем (серая черта), та же шкала."""
    med = t7["median_error"]
    sets = sorted(med, key=lambda s: (med[s], s))
    sx = lambda v: X0 + v / xmax * ((W - 44) - X0)  # noqa: E731
    g, y = [_text(X0, 11, UI["ex_mo"], "acc"), _text((W - 44), 11, UI["ex_all"], "note", "end")], 18.0
    for s in sets:
        g.append(_text(X0, y + 11, f"{s} · {UI['short_' + s]}", "lab"))
        yl = y + 22
        g.append(f'<line class="row-rule" x1="{X0}" x2="{(W - 44)}" y1="{yl:.1f}" y2="{yl:.1f}"/>')
        g.append(
            f'<line x1="{sx(med[s]):.1f}" x2="{sx(med[s]):.1f}" y1="{yl - 7:.1f}" y2="{yl + 7:.1f}" '
            f'stroke="{INK2}" stroke-width="2"/>'
        )
        if ex_err.get(s) is not None:
            g.append(f'<circle cx="{sx(ex_err[s]):.1f}" cy="{yl:.1f}" r="4.5" fill="{ACCENT}"/>')
            g.append(_text(W, yl + 4, _num(ex_err[s], 3), "val", "end"))
        y = yl + 8
    return _svg(W, y + 6, "".join(g), UI["ex_err_aria"].format(name=name), "ch-svg ex-err dots")


def example_map(ex: Mapping, geo: Mapping, name: str) -> str:
    """Мини-карта примера: линии к набору продукта (из других регионов) и к соседям по региону."""
    xy = geo["xy"]
    me = xy.get(int(ex["territory_id"]))
    if me is None:
        return ""
    mx, my = me
    prod = [xy[i] for i, _ in ex.get("members_p") or [] if i in xy]
    nbrs = [xy[i] for i, _ in ex.get("members_b") or [] if i in xy]
    g = [f'<g stroke="{INK}" stroke-width="1.1" opacity="0.8">']
    g += [f'<line x1="{mx:.0f}" y1="{my:.0f}" x2="{x:.0f}" y2="{y:.0f}"/>' for x, y in prod]
    g.append(f'</g><g stroke="{INK2}" stroke-width="1" stroke-dasharray="2 2">')
    g += [f'<line x1="{mx:.0f}" y1="{my:.0f}" x2="{x:.0f}" y2="{y:.0f}"/>' for x, y in nbrs]
    g.append("</g>")
    g += [f'<circle cx="{x:.0f}" cy="{y:.0f}" r="4.5" fill="{INK}"/>' for x, y in prod]
    g += [
        f'<circle cx="{x:.0f}" cy="{y:.0f}" r="3.2" fill="#fff" stroke="{INK2}" stroke-width="1.2"/>'
        for x, y in nbrs
    ]
    g.append(f'<circle cx="{mx:.0f}" cy="{my:.0f}" r="7" fill="{ACCENT}" stroke="#fff" stroke-width="2"/>')
    anc, tx = ("end", mx - 11) if mx > geo["w"] * 0.6 else ("start", mx + 11)
    g.append(_text(tx, my - 10, name, "acc map-lab", anc))
    label = UI["ex_map_aria"].format(name=name, n_p=len(prod), n_b=len(nbrs))
    return _mini_map(geo, "".join(g), label, "mini-map ex-map")


def chapter_comparable(
    story: Mapping, checks: Mapping, mo: Mapping[int, Mapping], geo: Mapping | None, esc: Esc
) -> str:
    ch = story["chapters"]["comparable"]
    view = story["view"]
    t7 = checks.get("t7")
    text = ""
    if ch.get("title_long"):  # 6o (check-ux 05.10): h2 короткий, утверждение с числами — первым абзацем
        text += f'<p class="sub">{esc(_dot_end(ch["title_long"]))}</p>'
    if ch.get("lead"):
        text += f'<p class="sub">{esc(ch["lead"])}</p>'
    k = (t7 or {}).get("k") or ""
    if ch.get("abs_note"):  # порция 6k: та же пара наборов без поправки на регион, как в совете
        text += f'<p class="sub">{esc(_dot_end(ch["abs_note"]))}</p>'
    text += f"<p>{esc(UI['comp_read'].format(k=k))}</p>"
    if ch.get("same_period"):
        text += f'<p class="note">{esc(ch["same_period"])}.</p>'
    if ch.get("text"):
        text += (
            f'<details class="how"><summary>{esc(SC_UI["how"])}</summary><p>{esc(ch["text"])}</p></details>'
        )
    roles = story.get("roles") or {}
    rl = [
        f"<li>{esc(roles[r])}"
        + (
            f' <small class="label-note">{esc(roles.get("business_label"))}</small>'
            if r == "business"
            else ""
        )
        + "</li>"
        for r in ("ministry", "analysts", "business")
        if roles.get(r)
    ]
    if rl:
        text += f'<h3 class="roles-h">{esc(UI["roles_title"])}</h3><ul class="roles">{"".join(rl)}</ul>'
    if not t7:
        return _section("comparable", ch["title"], text, "", esc)
    ex = t7.get("example") or {}
    ex_err = ex.get("errors") or {}
    svg, xmax = comparable_svg(t7, view, ex_err)
    svg = SC.phone_pair(svg, comparable_svg(t7, view, ex_err, SC.PHONE_W)[0])
    rows = diff_rows(t7)
    dsvg = SC.phone_pair(diff_svg(rows, view), diff_svg(rows, view, SC.PHONE_W))
    med, mab = t7["median_error"], t7.get("median_error_abs") or {}
    sets = sorted(med, key=lambda s: (med[s], s))
    trows = [
        [
            esc(f"{s} · {_set_label(s, view)}"),
            _num(med[s], 4),
            _num(mab.get(s), 4),
            _num(ex_err.get(s), 4),
        ]
        for s in sets
    ]
    drows = [
        [
            esc(f"{a} − {b}"),
            _signed(d["rel"][0], 4),
            f"{_signed(d['rel'][1][0], 4)} … {_signed(d['rel'][1][1], 4)}",
            (
                f"{_signed(d['abs'][0], 4)} ({_signed(d['abs'][1][0], 4)} … {_signed(d['abs'][1][1], 4)})"
                if d.get("abs")
                else "—"
            ),
        ]
        for a, b, d in rows
        if d.get("rel")
    ]
    key = (
        '<ul class="dot-key">'
        f'<li><i class="k-dot"></i>{esc(UI["comp_rel"])}</li>'
        f'<li><i class="k-ring"></i>{esc(UI["comp_abs"])}</li></ul>'
    )
    n_common = t7.get("n_common")
    fig = (
        '<figure class="ch-fig comp-fig">'
        f"<h3>{esc(UI['comp_title'].format(n=_mo_n(n_common) if n_common else '—'))}</h3>{key}{svg}"
        + (f'<p class="note comp-link">{esc(ch["comp_link"])}.</p>' if ch.get("comp_link") else "")
        + (f"<h3>{esc(UI['diff_title'])}</h3>{dsvg}" if dsvg else "")
    )
    r = mo.get(int(ex["territory_id"])) if ex.get("territory_id") is not None else None
    if r is not None:
        name = str(r.get("ns") or r.get("n"))
        pset = view.get("set_name") or t7.get("product") or ""
        fig += (
            f'<div class="ex-block"><h3>{_fig_mark(r.get("t"), view)}'
            f"{esc(UI['ex_title'].format(name=name, region=r.get('r', '')))}</h3>"
            f'<p class="note">{esc(UI["ex_note"].format(set=pset))}</p>'
        )
        n_p, n_b = len(ex.get("members_p") or []), len(ex.get("members_b") or [])
        if geo:
            fig += example_map(ex, geo, name)
            fig += (
                '<ul class="dot-key">'
                f'<li><i class="k-acc"></i>{esc(UI["ex_key_self"])}</li>'
                f'<li><i class="k-dot"></i>{esc(UI["ex_key_p"].format(set=pset, n=n_p))}</li>'
                f'<li><i class="k-ring"></i>{esc(UI["ex_key_b"].format(n=n_b))}</li>'
                "</ul>"
            )
            if (story.get("card") or {}).get("shifted"):
                card = story["card"]
                fig += f'<p class="note">{esc(card["shifted"])}. {esc(card.get("lines") or "")}.</p>'
        if ex_err:
            fig += SC.phone_pair(
                example_err_svg(t7, ex_err, xmax, name), example_err_svg(t7, ex_err, xmax, name, SC.PHONE_W)
            )
        rid = int(r["id"])
        fig += f'<p><a class="open-card" href="#mo={rid}" data-go="{rid}">{esc(UI["ex_open"])}</a></p>'
        exrows = [
            [esc(pset), esc(str((mo.get(i) or {}).get("ns") or (mo.get(i) or {}).get("n") or i)),
             esc(str((mo.get(i) or {}).get("r") or "")), _f(km)]
            for i, km in ex.get("members_p") or []
        ] + [
            [esc(UI["set_B"]), esc(str((mo.get(i) or {}).get("ns") or (mo.get(i) or {}).get("n") or i)),
             esc(str((mo.get(i) or {}).get("r") or "")), _f(km)]
            for i, km in ex.get("members_b") or []
        ]  # fmt: skip
        fig += _table([UI["ex_col_set"], UI["ex_col_name"], "", UI["ex_col_km"]], exrows, esc, UI["alt_ex"])
        fig += "</div>"
    fig += (
        f'<figcaption class="source">{esc(UI["src_comp"])}</figcaption>'
        + _table(
            [UI["comp_col_set"], UI["comp_col_rel"], UI["comp_col_abs"], UI["comp_col_ex"]],
            trows,
            esc,
            UI["alt_comp"],
        )
        + (
            _table([UI["diff_col"], UI["diff_col_v"], UI["diff_col_ci"], UI["comp_abs"]], drows, esc)
            if drows
            else ""
        )
        + "</figure>"
    )
    return _section("comparable", ch["title"], text, fig, esc)


# --- глава 7: чего данные не показывают (§3.10)


def r1_maps(r1: Mapping, geo: Mapping | None, esc: Esc) -> tuple[str, list[list[str]]]:
    """Три маленькие карты вариантов R1: серая точка — тот же тип, акцентная — другой."""
    out, rows = [], []
    for v in r1.get("variants") or []:
        word = VARIANT_WORDS.get(v["variant"], v["variant"])
        same = UI["r1_same"].format(pct=_pct0(v["same"] / v["n"]), same=_f(v["same"]), n=_f(v["n"]))
        ari = UI["r1_ari"].format(ari=_num(v.get("ari"), 2)) if v.get("ari") is not None else ""
        if ari and not out:  # 6o (check-ux 05.10): что такое ARI — при первом упоминании
            ari += UI["r1_ari_first"]
        rows.append([esc(word), esc(same), _num(v.get("ari"), 2)])
        if not geo:
            continue
        diff = [geo["xy"][i] for i in v.get("diff_ids") or [] if i in geo["xy"]]
        layer = (
            f'<path d="{_dots(diff)}" stroke="{ACCENT}" stroke-width="7" stroke-linecap="round" fill="none"/>'
        )
        svg = _mini_map(geo, layer, UI["r1_aria"].format(variant=word, same=same), "mini-map r1-map")
        out.append(
            f'<div class="r1-card"><h4>{esc(word)}</h4>{svg}'
            f'<p class="note">{esc(same)}' + (f"; {esc(ari)}" if ari else "") + "</p></div>"
        )
    return "".join(out), rows


def _dot_end(s: str) -> str:
    s = str(s).rstrip()
    return s if not s or s[-1] in ".!?…" else s + "."


def _mo_n(n: int) -> str:
    """«1542 муниципалитета» — число и слово по правилу числительного (порция 6b)."""
    from munnet.site_findings import _plural

    return f"{_f(n)} {_plural(int(n))}"


def _pct0(x: float) -> str:
    """Доля «тот же тип» — с одним знаком, как в блоке «Что устояло» (порция 6a: было 47% против 46,9%)."""
    return mstyle.fmt_pct(x, 1)


def chapter_limits(story: Mapping, checks: Mapping, geo: Mapping | None, esc: Esc) -> str:
    ch = story["chapters"]["limits"]
    text = f'<p class="outcome">{esc(ch["text"])}</p>' if ch.get("text") else ""
    if ch.get("plain") and text:
        # 6o (check-ux 05.10): пересказ исхода T6 названиями типов; дословный текст — под «Как проверяли»
        vl = f"<b>{esc(ch['verbatim_label'])}.</b> " if ch.get("verbatim_label") else ""
        text = (
            (f'<p class="sub">{esc(_dot_end(ch["context"]))}</p>' if ch.get("context") else "")
            + f'<p class="outcome">{esc(_dot_end(ch["plain"]))}</p>'
            + f'<details class="how"><summary>{esc(SC_UI["how"])}</summary>'
            + f'<p class="verbatim">{vl}{esc(ch["text"])}</p></details>'
        )
    elif ch.get("context") and text:  # порция 6j: перед дословным исходом T6 — что смена типа не подтверждена
        # порция 6k: «как читать» (типы названиями, МО, ε²) — тоже до текста исхода
        key = f'<p class="explain">{esc(_dot_end(ch["key"]))}</p>' if ch.get("key") else ""
        text = f'<p class="sub">{esc(_dot_end(ch["context"]))}</p>' + key + text
    elif ch.get("key"):  # порция 6e: «как читать» — типы названиями, МО, ε² (текст исхода выше — дословно)
        text += f'<p class="explain">{esc(_dot_end(ch["key"]))}</p>'
    out_text = str(ch.get("text") or "")
    for x in ch.get("lead") or []:
        if str(x).rstrip(".") in out_text:  # порция 6k: строка уже есть в тексте исхода — не повторять
            continue
        text += f'<p class="sub">{esc(x)}.</p>'
    items = [(UI[f"lim_{k}"], v) for k, v in (ch.get("items") or {}).items() if v and f"lim_{k}" in UI]
    if items:
        text += (
            f'<h3 class="roles-h">{esc(UI["limits_list"])}</h3><ul class="limits">'
            + "".join(f"<li><b>{esc(h)}.</b> {esc(v)}.</li>" for h, v in items)
            + "</ul>"
        )
    r1 = checks.get("r1") or {}
    maps, rows = r1_maps(r1, geo, esc)
    if not rows:
        return _section("limits", ch["title"], text, "", esc, kicker=UI["limits_kicker"])
    key = (
        '<ul class="dot-key">'
        f'<li><i class="k-base"></i>{esc(UI["r1_key_same"])}</li>'
        f'<li><i class="k-acc"></i>{esc(UI["r1_key_diff"])}</li></ul>'
    )
    lab = (story.get("view") or {}).get("unstable_label")
    fig = (
        '<figure class="ch-fig r1-fig">'
        f"<h3>{esc(UI['lim_fragile'])}: {esc(UI['r1_title'].lower())}</h3>{key}"
        f'<div class="r1-grid">{maps}</div>'
        + (f'<p class="note">{esc(UI["r1_unstable"].format(label=lab))}</p>' if lab else "")
        + f'<figcaption class="source">{esc(SC_UI["src_sber"])}</figcaption>'
        + _table([UI["r1_col_var"], UI["r1_col_same"], UI["r1_col_ari"]], rows, esc, UI["alt_r1"])
        + (
            f'<p class="note alt-note">{esc(_dot_end(r1["circularity"]))}</p>'
            if r1.get("circularity")
            else ""
        )
        + "</figure>"
    )
    return _section("limits", ch["title"], text, fig, esc, kicker=UI["limits_kicker"])


# --- глава 9: как сделано (§3.11)


def _glyph(i: int, view: Mapping) -> str:
    """Маленькая картинка шага (декоративная: смысл — в тексте рядом)."""
    if i == 0:
        hs = [30, 18, 12, 9, 7, 20]
        body = "".join(
            f'<rect x="{6 + j * 12}" y="{40 - h}" width="8" height="{h}" fill="{INK2}"/>'
            for j, h in enumerate(hs)
        )
    elif i == 1:
        vs = [8, -6, 12, -3, -10, 5]
        body = f'<line x1="4" x2="80" y1="24" y2="24" stroke="{INK}"/>' + "".join(
            f'<rect x="{6 + j * 12}" y="{24 - max(v, 0)}" width="8" height="{abs(v)}" fill="{INK2}"/>'
            for j, v in enumerate(vs)
        )
    elif i == 2:
        pts = [(10, 12), (30, 34), (46, 10), (64, 30), (76, 12), (22, 42)]
        edges = [(0, 1), (0, 2), (1, 3), (2, 3), (3, 4), (1, 5), (2, 4)]
        body = "".join(
            f'<line x1="{pts[a][0]}" y1="{pts[a][1]}" x2="{pts[b][0]}" y2="{pts[b][1]}" stroke="{INK2}"/>'
            for a, b in edges
        ) + "".join(f'<circle cx="{x}" cy="{y}" r="4" fill="{INK}"/>' for x, y in pts)
    elif i == 3:
        colors = view.get("type_colors") or {}
        shapes = view.get("shapes") or {}
        order = [str(t) for t in view.get("legend_order") or sorted(colors)]
        body = "".join(
            f'<text x="{8 + j * 19}" y="30" font-size="18" fill="{colors.get(t, INK)}">'
            f"{_e(shapes.get(t, '●'))}</text>"
            for j, t in enumerate(order[:4])
        )
    else:
        body = (
            "".join(
                f'<rect x="{4 + j * 6}" y="8" width="5" height="10" fill="{INK2 if j < 8 else GREY}"/>'
                for j in range(13)
            )
            + "".join(
                f'<circle cx="{10 + (j * 37) % 50}" cy="{30 + (j * 7) % 12}" r="1.6" fill="{INK2}"/>'
                for j in range(18)
            )
            + f'<line x1="74" x2="74" y1="24" y2="44" stroke="{ACCENT}" stroke-width="2.5"/>'
        )
    return (
        f'<svg class="glyph" viewBox="0 0 84 48" width="84" height="48" aria-hidden="true" focusable="false" '
        f'xmlns="http://www.w3.org/2000/svg">{body}</svg>'
    )


def _link(url: str, label: str, esc: Esc) -> str:
    return f'<a href="{_e(url)}" rel="noopener">{esc(label)}</a>'


def methods_table(methods: Mapping, rows: Sequence[Mapping], esc: Esc, tid: str) -> str:
    metrics = methods["metrics"]
    labels, better = methods["labels"], methods["better"]
    head = [
        ("method", UI["m_col_method"], "text"),
        ("family", UI["m_col_family"], "text"),
        ("k", UI["m_col_k"], "num"),
        ("ok", UI["m_col_ok"], "text"),
    ] + [(m, f"{labels[m]} {'↑' if better[m] == 'max' else '↓'}", "num") for m in metrics]
    th = "".join(
        f'<th scope="col" data-kind="{kind}"><button type="button" class="sort" data-col="{j}">'
        f"{esc(lab)}</button></th>"
        for j, (_, lab, kind) in enumerate(head)
    )
    body = []
    for r in rows:
        cells = [
            (
                esc(METHOD_WORDS.get(r["method"], r["method"]))
                + (f' <b class="final">{esc(UI["m_final"])}</b>' if r.get("final") else ""),
                METHOD_WORDS.get(r["method"], r["method"]),
            ),
            (esc(FAMILY_WORDS.get(r["family"], r["family"])), FAMILY_WORDS.get(r["family"], r["family"])),
            (_f(r["k"]), r["k"]),
            (esc(UI["m_yes"] if r["feasible"] else UI["m_no"]), int(bool(r["feasible"]))),
        ]
        for m in metrics:
            v, z = (r.get("v") or {}).get(m), (r.get("z") or {}).get(m)
            dec = 1 if m == "ch" else 3
            cells.append(
                (
                    "<span>"
                    + ("—" if v is None else _f(v, dec))
                    + (f"<small>z {_f(z, 1)}</small>" if z is not None else "")
                    + "</span>",
                    "" if v is None else v,
                )
            )
        tds = "".join(
            f'<td data-v="{_e(sv)}">{html}</td>' if head[j][2] == "num" else f"<td>{html}</td>"
            for j, (html, sv) in enumerate(cells)
        )
        cls = ' class="is-final"' if r.get("final") else ""
        body.append(f"<tr{cls}>{tds}</tr>")
    return (
        f'<div class="mt-wrap"><table class="mtable" id="{tid}" aria-label="{_e(UI["m_table_aria"])}">'
        f"<thead><tr>{th}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"
    )


def methods_labels_css(methods: Mapping) -> str:
    """Подписи ячеек для карточек на телефоне — одним правилом на столбец, а не атрибутом в каждой ячейке."""
    labels, better = methods["labels"], methods["better"]
    heads = [UI["m_col_family"], UI["m_col_k"], UI["m_col_ok"]] + [
        f"{labels[m]} {'↑' if better[m] == 'max' else '↓'}" for m in methods["metrics"]
    ]
    rules = "".join(f'.mtable td:nth-child({j + 2})::before{{content:"{h}"}}' for j, h in enumerate(heads))
    return f"<style>@media (max-width:700px){{{rules}}}</style>"


def chapter_method(story: Mapping, checks: Mapping, methods: Mapping | None, meta: Mapping, esc: Esc) -> str:
    ch = story["chapters"]["method"]
    view = story["view"]
    m = ch.get("numbers") or {}
    links = ch.get("links") or {}
    repo = str(links.get("repo") or "")
    blob = lambda p: f"{repo}/blob/HEAD/{p}"  # noqa: E731
    report = blob("report/report.md")
    steps = [
        ("m_s1", [("docs/eda.md", "docs/eda.md")], 2),
        ("m_s2", [("docs/features.md", "docs/features.md")], 3),
        ("m_s3", [("docs/network.md", "docs/network.md")], 4),
        ("m_s4", [("docs/clustering.md", "docs/clustering.md"), ("docs/icvi.md", "docs/icvi.md")], "5–6"),
        (
            "m_s5",
            [("docs/dynamics.md", "docs/dynamics.md"), ("docs/interpretation.md", "docs/interpretation.md")],
            "7–8",
        ),
    ]
    li = []
    for i, (key, docs, sec) in enumerate(steps):
        body = (
            UI[key].format(**{k: v for k, v in m.items()})
            if all(v not in (None, "") for v in m.values())
            else UI[key]
        )
        a = (
            " · ".join(
                [_link(report, UI["m_report_many" if "–" in str(sec) else "m_report"].format(n=sec), esc)]
                + [_link(blob(p), lab, esc) for p, lab in docs]
            )
            if repo
            else ""
        )
        li.append(
            f'<li class="mstep"><span class="mnum" aria-hidden="true">{i + 1}</span>{_glyph(i, view)}'
            f"<h3>{esc(UI[key + '_h'])}</h3><p>{esc(body)}</p>"
            + (f'<p class="mlinks">{a}</p>' if a else "")
            + "</li>"
        )
    aria = _e(UI["m_steps_aria"])
    fig = f'<figure class="ch-fig method-fig"><ol class="msteps" aria-label="{aria}">{"".join(li)}</ol>'
    # воспроизводимость и предрегистрация
    text = ""
    cmd = ch.get("command")
    if cmd:
        sha = str(meta.get("sha") or "")[:7] or "—"
        facts = str(meta.get("facts_sha256") or "")[:12] or "—"
        line = UI["m_repro_meta"].format(seed=meta.get("seed", "—"), sha=sha, facts=facts)
        text += (
            f'<h3 class="roles-h">{esc(UI["m_repro_h"])}</h3><p>{esc(UI["m_repro"])}</p>'
            f'<pre class="cmd"><code>{_e(cmd)}</code></pre><p class="note">{esc(line)}</p>'
        )
    prereg = ch.get("prereg") or []
    if prereg:
        items = "".join(
            f"<li>{esc(p['text'])} — "
            + (_link(f"{repo}/commit/{p['commit']}", p["commit"][:7], esc) if repo else esc(p["commit"][:7]))
            + "</li>"
            for p in prereg
        )
        text += f'<h3 class="roles-h">{esc(UI["m_prereg_h"])}</h3><ul class="prereg">{items}</ul>'
    if repo:
        rep = _link(report, UI["m_report_all"], esc)
        text += f"<p>{_link(repo, UI['m_repo'], esc)} · {rep}</p>"
    fig += f'<div class="mrepro">{text}</div>' if text else ""
    text = ""
    if methods and methods.get("rows"):
        rows = methods["rows"]
        n_none = _f(len({r["method"] for r in rows} - {r["method"] for r in rows if r.get("feasible")}))
        best = sorted(
            [r for r in rows if r.get("winner") or r.get("final")],
            key=lambda r: (not r.get("final"), r["family"], r["method"], r["k"]),
        )
        gloss = (
            "".join(
                f"<li>{esc(ICVI_GLOSS[k])}; {esc(UI['m_better_' + methods['better'][k]])}</li>"
                for k in methods["metrics"]
                if k in ICVI_GLOSS
            )
            + f"<li>{esc(ICVI_GLOSS['ok'])}</li>"
        )
        fig += (
            f"<h3>{esc(UI['m_table_h'])}</h3>"
            f'<p class="note">{esc(UI["m_table_note"].format(n_none=n_none))} '
            f'<span class="sort-hint">{esc(UI["m_sort_hint"])}</span></p>'
            + methods_labels_css(methods)
            + methods_table(methods, best, esc, "mtable-best")
            + f'<details class="alt"><summary>{esc(UI["m_table_all"].format(n=_f(len(rows))))}</summary>'
            + methods_table(
                methods, sorted(rows, key=lambda r: (r["family"], r["method"], r["k"])), esc, "mtable-all"
            )
            + "</details>"
            + f'<details class="alt"><summary>{esc(UI["m_gloss_h"])}</summary>'
            + f'<ul class="gloss-list">{gloss}</ul></details>'
        )
    fig += f'<figcaption class="source">{esc(SC_UI["src_sber"])}</figcaption></figure>'
    return _section("method", ch["title"], text, fig, esc, wide=True)


# --- таблица всех муниципалитетов (порция 6c)


def chapter_all_mo(downloads: Mapping[str, int] | None, esc: Esc, years_note: str | None = None) -> str:
    """Каркас таблицы всех муниципалитетов: подписи, фильтры и заголовки с сортировкой (``aria-sort``);
    строки рисует landing.js по ``data/mo.json`` страницами (2192 строки сразу в DOM не идут). Без JS —
    ссылка на ``mo.csv`` и подсказка. ``years_note`` (6q, совет 06.10; ``texts.dynamics.table_years``
    при исходе главы 5) — строка над таблицей, на неё ссылаются заголовки колонок 2023 и 2024
    (``aria-describedby``)."""
    size = (downloads or {}).get("mo.csv")
    kb = f" <small>({_f(max(1, round(size / 1024)))} КБ)</small>" if size else ""
    cols = [("all_c_name", "text"), ("all_c_region", "text"), ("all_c_type", "num"), ("all_c_flag", "num"),
            ("all_c_23", "num"), ("all_c_24", "num")]  # fmt: skip
    th = "".join(
        f'<th scope="col" data-kind="{kind}"'
        + (' aria-sort="ascending"' if j == 0 else "")
        + f'><button type="button" class="sort" data-col="{j}"'
        + (' aria-describedby="mo-years"' if years_note and k in ("all_c_23", "all_c_24") else "")
        + f">{esc(UI[k])}</button></th>"
        for j, (k, kind) in enumerate(cols)
    )
    text = (
        f"<p>{esc(UI['all_read'])}</p>"
        f'<p class="note">{esc(UI["all_note"])}</p>'
        f'<p class="nojs">{esc(UI["all_nojs"])} <a href="data/download/mo.csv" download>mo.csv</a>{kb}. '
        f"{esc(UI['all_nojs_tail'])}</p>"
    )
    ctl = (
        '<div class="mo-ctl" id="mo-ctl" hidden>'
        f'<label class="mo-f"><span>{esc(UI["all_q"])}</span><input id="mo-q" type="search" '
        'autocomplete="off" '
        f'spellcheck="false" placeholder="{_e(UI["all_q_ph"])}"></label>'
        f'<label class="mo-f"><span>{esc(UI["all_region"])}</span><select id="mo-region">'
        f'<option value="">{esc(UI["all_any_region"])}</option></select></label>'
        f'<label class="mo-f"><span>{esc(UI["all_type"])}</span><select id="mo-type">'
        f'<option value="">{esc(UI["all_any_type"])}</option></select></label>'
        "</div>"
    )
    fig = (
        '<div class="ch-fig mo-fig">'
        + ctl
        + '<p class="mo-status" id="mo-status" aria-live="polite"></p>'
        + (f'<p class="mo-years" id="mo-years">{esc(years_note.rstrip(".") + ".")}</p>' if years_note else "")
        + '<div class="mo-wrap" id="mo-wrap" hidden><table class="motable" id="mo-table">'
        + f'<caption class="sr-only">{esc(UI["all_caption"])}</caption>'
        + f"<thead><tr>{th}</tr></thead><tbody></tbody></table></div>"
        + '<div class="mo-pager" id="mo-pager" hidden>'
        f'<button type="button" class="tool tool-text" id="mo-prev">{esc(UI["all_prev"])}</button>'
        f'<button type="button" class="tool tool-text" id="mo-next">{esc(UI["all_next"])}</button></div>'
        + f'<p class="source">{esc(SC_UI["src_sber"])}</p>'
        + "</div>"
    )
    # 6q (check-ux 06.10): tabindex="-1" — после ссылки «Перейти к таблице…» фокус оказывается в разделе
    return _section("all-mo", UI["all_title"], text, fig, esc, wide=True, kicker=UI["all_kicker"], focus=True)


# --- глава 10: источники и лицензии


def sources_html(story: Mapping, downloads: Mapping[str, int], esc: Esc) -> str:
    """Подвал: источники с лицензиями и ссылками, выгрузки CSV с размерами, лицензия кода."""
    title = ((story.get("chapters") or {}).get("sources") or {}).get("title") or "Источники и лицензии"
    src = "".join(
        f"<li><b>{esc(name)}</b>: {esc(what)} — {esc(lic)}. "
        f"{_link(url, url.split('//', 1)[1].split('/', 1)[0], esc)}</li>"
        for name, what, lic, url in SOURCES
    )
    dl = "".join(
        f'<li><a href="data/download/{_e(f)}" download>{_e(f)}</a> — {esc(what)}'
        + (f" <small>({_f(max(1, round(downloads[f] / 1024)))} КБ)</small>" if f in downloads else "")
        + "</li>"
        for f, what in DOWNLOADS
        if f in downloads
    )
    return (
        f'<h2 id="sources-title">{esc(title)}</h2><ul id="sources-list">{src}</ul>'
        + (f'<h3>{esc(UI["src_downloads"])}</h3><ul class="downloads">{dl}</ul>' if dl else "")
        + f"<p>{esc(UI['src_code'])}</p>"
        + f'<h3>{esc(UI["src_tools"])}</h3><ul class="tools">'
        + "".join(
            f"<li><b>{esc(name)}</b>: {esc(what)} — {esc(lic)}. "
            f"{_link(url, url.split('//', 1)[1].split('/', 1)[0], esc)}</li>"
            for name, what, lic, url in TOOLS
        )
        + "</ul>"
    )


# --- сборка


def chapters_html(
    story: Mapping,
    checks: Mapping,
    mo: Mapping[int, Mapping],
    geo: Mapping | None,
    methods: Mapping | None,
    meta: Mapping,
    esc: Esc,
    downloads: Mapping[str, int] | None = None,
) -> str:
    """Главы 6, 7, таблица всех муниципалитетов (порция 6c), глава 9 в порядке страницы (§2) и общий слой
    ячеек для маленьких карт."""
    return "\n".join(
        [
            cells_defs(geo),
            chapter_comparable(story, checks, mo, geo, esc),
            chapter_limits(story, checks, geo, esc),
            chapter_all_mo(downloads, esc, story.get("table_years")),
            chapter_method(story, checks, methods, meta, esc),
        ]
    )
