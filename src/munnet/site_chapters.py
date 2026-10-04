"""Главы 1, 3, 4, 5 лендинга (§3.4–§3.7 ``docs/landing_spec.md``): HTML и статичные SVG, собранные при сборке.

Страница ничего не считает: всё, что нарисовано, лежит в ``story.json``, ``types.json``, ``checks.json`` и
``mo.json``. Заголовки и тексты выводов — только из ``story`` (слоты этапа 5 и ``site.headlines``); здесь —
лишь нейтральный интерфейс из словаря ``UI`` (подписи осей, ключи, кнопки), он проходит тот же линт, что
шаблон. Форма главы 4 — по ``view.t1_layout`` и ``view.line_by_turnover``, стиль потоков главы 5 —
по ``view.flows`` (вердикт T3), стрелки — по ``view.arrows`` (T2). JS только переключает примеры главы 1
и раскрывает список муниципалитетов потока главы 5.
"""

from __future__ import annotations

import html
import math
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from munnet import style

# Нейтральный интерфейс глав: без выводов, линтуется (landing.lint_templates) при текущих вердиктах.
UI: dict[str, str] = {
    "src_sber": "Источник: СберИндекс (CC BY-SA 4.0), расчёт «Корзина и регион».",
    "src_rosstat": (
        "Источник: СберИндекс (CC BY-SA 4.0); Росстат, БД ПМО в обработке «Если быть точным» (CC BY 4.0); "
        "расчёт «Корзина и регион»."
    ),
    # порция 6f: блок «Что устояло» с советами по размеру — ещё и доход по 5-НДФЛ (лицензия на странице набора
    # не указана, как в report.md, приложение Б)
    "src_rosstat_ndfl": (
        "Источник: СберИндекс (CC BY-SA 4.0); Росстат, БД ПМО в обработке «Если быть точным» (CC BY 4.0); "
        "ФНС, 5-НДФЛ в обработке «Если быть точным» (лицензия на странице набора не указана); "
        "расчёт «Корзина и регион»."
    ),
    "alt": "Данные графика таблицей",
    "how": "Как проверяли",
    "how_note": "Подробнее",
    "alt_ami": "Все деления без типов таблицей",
    "alt_turnover": "Медианы оборота таблицей",
    "alt_rivals": "Сила порядка у всех делений таблицей",
    "alt_flows": "Смены типа таблицей",
    "alt_placebo": "Наблюдение и плацебо таблицей",
    "types_read": (
        "Каждая панель — один тип. Строки — признаки, по которым строились типы: корзина трат и экономика "
        "места. Ниже у каждого типа — его размер, устойчивость и примеры: они открывают карточку "
        "муниципалитета."
    ),
    "order_read": (
        "Оборот розничной торговли и общепита Росстата на жителя в расчёт типов не входил, поэтому по нему "
        "проверяли, упорядочены ли типы. Проверяли дважды: в целом и при том же размере и доле горожан, — "
        "и сравнили с делениями муниципалитетов на четыре группы без типов."
    ),
    "dyn_read": (
        "Тип считали отдельно по 2023 и 2024 годам. Смену типа засчитывали, если с ней согласны обе половины "
        "года — нечётные и чётные месяцы. Чтобы узнать, сколько таких смен даёт случайность, тот же расчёт "
        "повторили на 200 «годах» из перемешанных месяцев — это плацебо."
    ),
    # глава 1
    "basket_p1": (
        "Каждая полоса — одна из шести частей безналичных трат жителей муниципалитета. Сравниваем долю этой "
        "части в корзине муниципалитета с её обычной долей в своём регионе: полоса вправо — больше, чем "
        "обычно в регионе, влево — меньше."
    ),
    "basket_p2": (
        "Так муниципалитет сравнивается с соседями по своему региону. "
        "Тип учитывает эту корзину и экономику места — зарплату, население, занятость, тоже относительно "
        "своего региона."
    ),
    "basket_steps": "Пример муниципалитета",
    "basket_head": "{name}, {region} — корзина {year} года",
    "basket_left": "← меньше, чем обычно в регионе",
    "basket_right": "больше →",
    "basket_zero": "как обычно в регионе",
    "basket_unst": "знак у типа неустойчив",
    "basket_open": "Открыть карточку муниципалитета",
    "basket_cap": (
        "Полоса — доля части трат относительно обычной доли в своём регионе; числа не показаны, важны "
        "знак и длина. «Знак у типа неустойчив» — у типа этого муниципалитета знак части меняется между "
        "вариантами расчёта."
    ),
    "basket_aria": "Корзина трат {name} относительно своего региона: {parts}",
    "basket_more": "больше",
    "basket_less": "меньше",
    "basket_same": "как обычно",
    # глава 3
    "types_scale": (
        "Точка — медиана типа, цветной отрезок — её 95% интервал, серая полоса — середина типа (половина его "
        "муниципалитетов), тонкая вертикаль — типичный муниципалитет (медиана всех). Шкала своя у каждой "
        "строки и общая для четырёх типов; всё — относительно своего региона, кроме долей занятости."
    ),
    "types_basket": "Корзина трат",
    "types_place": "Экономика места",
    "types_unst": "* знак неустойчив к вариантам расчёта",
    "types_nodes": "муниципалитетов",
    "types_people": "жителей",
    "types_jac": "Совпадение с типом при перезапусках (Жаккар) — {j}",
    "types_unstable": "неустойчивый тип",
    "types_typical": "Типичные",
    "types_border": "На границе с другим типом",
    "types_largest": "Крупнейший",
    "types_aria": "{name}: медианы признаков типа относительно типичного муниципалитета",
    # запасная плашка; на странице — story.stability_label (site.build.texts.descr_label)
    "types_note_label": "описание: заранее записанной проверки нет",
    "ami_title": "С каким делением без типов типы совпадают сильнее всего",
    "ami_cap": (
        "AMI — совпадение типов с делением муниципалитетов по одному признаку: 0 — как у случайного деления, "
        "1 — полное совпадение. Показаны {n} делений с наибольшим AMI."
    ),
    "ami_aria": "Совпадение типов с делениями без типов, AMI",
    # глава 4
    "order_a": "Оборот на жителя по типам, медиана",
    "order_a_note": "Оборот Росстата (без МСП) в кластеризации не участвовал",
    "order_axis": (
        "относительно своего региона, логарифмическая шкала: ×2 — вдвое больше, чем обычно в регионе"
    ),
    "order_b": "Так же ли сильно упорядочены деления на четыре группы без типов — при том же размере и доле "
    "горожан",
    "order_b_axis": "сила порядка внутри страт размера и доли горожан, ρ Спирмена",
    "order_key_types": "типы, отрезок — 95% интервал",
    "order_key_rivals": "деления без типов: по одному признаку места, уровню трат, части корзины; K-means по "
    "признакам места",
    "order_key_random": "случайные метки",
    "order_line_solid": "сплошная линия — проверка пройдена",
    "order_line_dashed": "пунктир — проверка пройдена в целом; при том же размере и доле горожан — нет",
    "order_line_none": "без линии — не пройдена",
    "catering": "общепит",
    "retail": "розница",
    "order_aria_a": "Медиана оборота {turnover} на жителя по типам относительно своего региона",
    "order_aria_b": "Сила порядка {turnover} внутри страт у типов и у делений без типов",
    "order_best": "сильнейшее: {label}",
    "order_types": "типы {rho}",
    # порция 6l: заголовок раскрывающегося описания — о чём оно; плашка — story.stability_label
    "order_proxies": "Городские округа, столицы регионов и плотность населения по типам",
    # глава 5
    "flows_a": "Тип в 2023 году",
    "flows_b": "Тип в 2024 году",
    "flows_stay": "тип не менялся",
    "flows_rel": "смена типа, обе половины года согласны",
    "flows_noise": "смена типа в пределах шума",
    "flows_hint": "Нажмите на поток или строку ниже — появится список его муниципалитетов.",
    # порция 6k (check-ux): размеры типов в потоках и в легенде карты различаются — почему
    "flows_sizes": (
        "Числа у типов здесь — тип по каждому году отдельно; в легенде карты и в паспортах типов — тип по "
        "всем 24 месяцам сразу, поэтому числа не совпадают."
    ),
    "flows_aria": "Потоки муниципалитетов между типами 2023 и 2024 годов",
    "flows_item": "{a} → {b}: {n}",
    "flows_changes": "Смены типа",
    "flows_col_from": "Тип 2023",
    "flows_col_to": "Тип 2024",
    "flows_col_n": "Муниципалитетов",
    "flows_col_rel": "Из них обе половины года согласны",
    "placebo_title": "Смен типа за год, в которых обе половины года согласны: наблюдение и плацебо",
    "placebo_final": "Самый строгий вариант расчёта — {variant}",
    # порция 6b (check-ux): итог под графиком плацебо словами; фразы — по вердиктам checks.t3, числа — там же
    "placebo_sum_head": "Заголовок главы — по самому строгому варианту расчёта.",
    "placebo_sum_final_not": (
        "При варианте «{variant}» смен не больше, чем на плацебо: {obs} при пороге {p95}."
    ),
    # порция 6l (check-ux 03.10): одно опорное число плацебо в главе — 95-й перцентиль (было «против медианы»)
    "placebo_sum_main_up": (
        "В основном расчёте смен больше, чем бывает на плацебо ({obs} при пороге {p95}), но "
        "этот результат не устоял к вариантам расчёта."
    ),
    "placebo_main": "Основной расчёт — {label}",
    "placebo_obs": "наблюдение {n}",
    "placebo_p95": "95-й перцентиль плацебо {n}",
    "placebo_key": (
        "Точка — одно плацебо: «год», собранный из перемешанных месяцев 2023 и 2024 годов. Пунктир — порог: "
        "больше этого числа смен дают только 5% плацебо (95-й перцентиль). Красная черта — наблюдение. "
        "Проверка пройдена, если черта правее пунктира."
    ),
    "placebo_aria": "Число смен типа: {label}; наблюдение {obs}, 95-й перцентиль плацебо {p95}",
    "placebo_col_run": "Расчёт",
    "placebo_col_obs": "Наблюдение",
    "placebo_col_med": "Медиана плацебо",
    "placebo_col_p95": "Порог: 95-й перцентиль плацебо",
    "placebo_axis": "число смен типа за год",
    "type_n": "Тип {t}",
    "col_types": "типы",
    "col_med": "медиана",
    "col_med_log": "медиана, логарифм",
    "col_ci": "95% интервал",
    "col_iqr": "середина типа",
    "types_emp": "Занятость, доли разделов",
}

VARIANT_WORDS = {
    "main": "основной расчёт",
    "variant:graph_basket_cos": "другое правило связей между муниципалитетами",
    "variant:no_level": "без признака уровня трат",
    "variant:nodes_separate": "районы Москвы и Петербурга отдельно",
    "tracking:fixed_prototypes": "другой способ прослеживания типов",
    "tracking:evolutionary": "другой способ прослеживания типов",
}

GREY = "#d6d2c8"
INK2 = "#595959"
ACCENT = "#a3172d"

Esc = Callable[[Any], str]


def ui_strings() -> list[str]:
    """Тексты интерфейса глав — для линта (``landing.lint_templates``)."""
    return list(UI.values()) + list(VARIANT_WORDS.values())


def _e(s: Any) -> str:
    return html.escape(str(s if s is not None else ""), quote=True)


def _f(x: float, d: int = 0) -> str:
    return style.fmt_num(x, d)


def _pct(x: float | None) -> str:
    return "—" if x is None else style.fmt_pct(x, 0)


def _uid(prefix: str, *parts: Any) -> str:
    return prefix + "-" + "-".join(re.sub(r"[^a-z0-9]+", "", str(p).lower()) for p in parts)


def _hatch(pid: str) -> str:
    return (
        f'<defs><pattern id="{pid}" width="5" height="5" patternUnits="userSpaceOnUse" '
        f'patternTransform="rotate(45)"><line x1="0" y1="0" x2="0" y2="5" stroke="#fff" stroke-width="2"/>'
        "</pattern></defs>"
    )


def _svg(w: float, h: float, body: str, label: str, cls: str) -> str:
    return (
        f'<svg class="{cls}" viewBox="0 0 {w:g} {h:g}" role="img" aria-label="{_e(label)}" '
        f'xmlns="http://www.w3.org/2000/svg">{body}</svg>'
    )


# Порция 6a: у каждого графика — вариант для телефона. Подписи остаются в тех же единицах (CSS ``.ch-svg``:
# 10–11,5), а ширина холста меньше: на экране 343 px подпись в 10 единиц — 343 / 280 × 10 ≈ 12,3 px.
PHONE_W = 280
PHONE_MIN_PX = 12.0  # нижняя граница подписи на телефоне (проверяет тест на собранной странице)


def _with_cls(svg: str, c: str) -> str:
    return svg.replace('<svg class="', f'<svg class="{c} ', 1) if svg else svg


def _clamp(cx: float, w: float, lo: float, hi: float) -> float:
    """Центр подписи шириной ``w`` у точки ``cx``, но целиком внутри [lo, hi]."""
    return min(max(cx, lo + w / 2), hi - w / 2) if hi - lo > w else (lo + hi) / 2


def _text_w(s: str, size: float) -> float:
    """Оценка ширины строки Golos Text в единицах SVG (средний знак ≈ 0,56 кегля) — для раскладки подписей."""
    return len(s) * size * 0.56


def phone_pair(wide: str, phone: str) -> str:
    """Широкий вариант (``vw``) и вариант для телефона (``vp``): CSS показывает один по ширине экрана
    (до 700 px — ``vp``). Скрытый ``display: none`` недоступен и экранному диктору, подпись не дублируется."""
    if not wide or not phone:
        return wide or phone
    return _with_cls(wide, "vw") + _with_cls(phone, "vp")


def _text(x: float, y: float, s: str, cls: str = "", anchor: str = "start", extra: str = "") -> str:
    c = f' class="{cls}"' if cls else ""
    a = f' text-anchor="{anchor}"' if anchor != "start" else ""
    return f'<text x="{x:.1f}" y="{y:.1f}"{c}{a}{extra}>{_e(_nb(s))}</text>'


def _nb(s: str) -> str:
    """Типографика ru-text в подписях SVG (порция 6j): неразрывный пробел после однобуквенных слов и перед
    тире — та же функция, что для HTML (``landing.nbsp``)."""
    from munnet.landing import nbsp

    return nbsp(str(s)) if s else s


def _wrap(s: str, n: int) -> list[str]:
    out, line = [], ""
    for w in s.split():
        if line and len(line) + 1 + len(w) > n:
            out.append(line)
            line = w
        else:
            line = f"{line} {w}".strip()
    return [*out, line] if line else out


def _table(head: Sequence[str], rows: Sequence[Sequence[str]], esc: Esc, summary: str | None = None) -> str:
    th = "".join(f'<th scope="col">{esc(h)}</th>' for h in head)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return (
        f'<details class="alt"><summary>{esc(summary or UI["alt"])}</summary><div class="alt-wrap"><table>'
        f"<thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div></details>"
    )


def _fig_mark(t: int | None, view: Mapping) -> str:
    if t is None:
        return '<i class="fig fig-t0" aria-hidden="true"></i>'
    return f'<i class="fig fig-t{int(t)}" aria-hidden="true">{_e(view["shapes"].get(str(int(t)), ""))}</i>'


def _type_name(story: Mapping, t: int) -> str:
    return str(story["names"]["final"].get(str(t)) or UI["type_n"].format(t=t))


def _type_label(story: Mapping, t: int, x: float, y: float, cls: str = "lab", n: int = 0) -> list[str]:
    """Фигура и название типа (как в легенде): цвет не единственный носитель; длинное — в ``n`` знаков
    строкой, переносы — ниже через 13 единиц."""
    view = story["view"]
    out = [_shape(t, x + 4.5, y - 4, 4.5, view["type_colors"].get(str(t), INK2), view)]
    lines = _wrap(_type_name(story, t), n) if n else [_type_name(story, t)]
    out += [_text(x + 14, y + j * 13, ln, cls) for j, ln in enumerate(lines)]
    return out


def _shape(t: int, x: float, y: float, r: float, color: str, view: Mapping, extra: str = "") -> str:
    """Фигура типа (● ▲ ■ ◆) — цвет не единственный носитель смысла (§3.1)."""
    s = view["shapes"].get(str(t), "●")
    if s == "▲":
        a, b, c = (
            f"{x:.1f},{y - r * 1.15:.1f}",
            f"{x + r * 1.1:.1f},{y + r * 0.8:.1f}",
            f"{x - r * 1.1:.1f},{y + r * 0.8:.1f}",
        )
        p = f"{a} {b} {c}"
        return f'<polygon points="{p}" fill="{color}"{extra}/>'
    if s == "■":
        return f'<rect x="{x - r * 0.85:.1f}" y="{y - r * 0.85:.1f}" width="{r * 1.7:.1f}" height="{r * 1.7:.1f}" fill="{color}"{extra}/>'  # noqa: E501
    if s == "◆":
        q = r * 1.2
        p = f"{x:.1f},{y - q:.1f} {x + q:.1f},{y:.1f} {x:.1f},{y + q:.1f} {x - q:.1f},{y:.1f}"
        return f'<polygon points="{p}" fill="{color}"{extra}/>'
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="{color}"{extra}/>'


def _shape_d(t: int, x: float, y: float, r: float, view: Mapping) -> str:
    """Та же фигура типа, что ``_shape``, как подпуть (много меток — один элемент ``path``)."""
    s = view["shapes"].get(str(t), "●")
    if s == "▲":
        return f"M{x:.1f} {y - r * 1.15:.1f}l{r * 1.1:.1f} {r * 1.95:.1f}h{-r * 2.2:.1f}z"
    if s == "■":
        return f"M{x - r * 0.85:.1f} {y - r * 0.85:.1f}h{r * 1.7:.1f}v{r * 1.7:.1f}h{-r * 1.7:.1f}z"
    if s == "◆":
        q = r * 1.2
        return f"M{x:.1f} {y - q:.1f}l{q:.1f} {q:.1f}l{-q:.1f} {q:.1f}l{-q:.1f} {-q:.1f}z"
    return f"M{x - r:.1f} {y:.1f}a{r:.1f} {r:.1f} 0 1 0 {2 * r:.1f} 0a{r:.1f} {r:.1f} 0 1 0 {-2 * r:.1f} 0"


def rel_key(story: Mapping, placebo: bool = True) -> str:
    """Ключ грамматики надёжности (``palette_rule.reliability_grammar.key``, заморожен). Порция 6b (check-ux):
    фраза о плацебо — только под графиками с плацебо; без них — первая фраза ключа (до «Плацебо»).
    Слова ключа не меняются, меняется только то, какая его часть видна."""
    key = str(story.get("reliability_key") or "")
    if placebo:
        return key
    head, sep, _ = key.partition(" Плацебо")
    return head.rstrip(".") if sep else key


def _section(
    cid: str, title: str, text_html: str, fig_html: str, esc: Esc, wide: bool = False, kicker: str = ""
) -> str:
    cls = "chapter wide" if wide else "chapter"
    kick = f'<p class="kicker">{esc(kicker)}</p>' if kicker else ""
    return (
        f'<section class="{cls}" id="{cid}" aria-labelledby="{cid}-title"><div class="ch-text">{kick}'
        f'<h2 id="{cid}-title">{esc(title)}</h2>{text_html}</div>{fig_html}</section>'
    )


# --- глава 1: корзина относительно своего региона (§3.4)


def basket_svg(
    b: Sequence[float | None], labels: Sequence[str], unstable: set[int], name: str, W: float = 360
) -> str:
    """Шесть горизонтальных полос от нуля: ноль — «как обычно в регионе»; значения не подписаны (§3.4).
    «Знак у типа неустойчив» — строкой под своей полосой (порция 6a: раньше налезал на подпись части)."""
    lab_w, row, top, note_h = 118, 30, 8, 13
    x0, x1 = lab_w + 6, W - 8
    zero = (x0 + x1) / 2
    lim = max([0.1, *[abs(v) for v in b if v is not None]]) * 1.08
    sx = lambda v: zero + v / lim * (x1 - zero)  # noqa: E731
    ys, y = [], top
    for i in range(len(labels)):
        ys.append(y)
        y += row + (note_h if i in unstable and i < len(b) and b[i] is not None else 0)
    yb = y
    H = yb + 34
    pid = _uid("hb", name, int(W))[:40]
    g = [_hatch(pid)]
    g.append(f'<line class="zero" x1="{zero:.1f}" x2="{zero:.1f}" y1="{top - 4}" y2="{yb}"/>')
    parts = []
    for i, lab in enumerate(labels):
        y = ys[i]
        v = b[i] if i < len(b) else None
        g.append(_text(lab_w, y + row / 2 + 4, lab, "lab", "end"))
        if v is None:
            continue
        xa, xb = sorted((zero, sx(v)))
        w = max(1.0, xb - xa)
        u = i in unstable
        fill = "#9a978f" if u else INK2
        g.append(f'<rect x="{xa:.1f}" y="{y + 7}" width="{w:.1f}" height="{row - 14}" fill="{fill}"/>')
        if u:
            g.append(
                f'<rect x="{xa:.1f}" y="{y + 7}" width="{w:.1f}" height="{row - 14}" fill="url(#{pid})"/>'
            )
            # своей строкой под полосой, от нуля в сторону полосы и в пределах холста
            wn = _text_w(UI["basket_unst"], 10)
            right = (zero + 4, "start") if zero + 4 + wn <= W else None
            left = (zero - 4, "end") if zero - 4 - wn >= 0 else None
            # не поперёк линии нуля: сначала сторона полосы, иначе другая, иначе — по краю холста
            tx, anc = (right or left or (W, "end")) if v >= 0 else (left or right or (0, "start"))
            g.append(_text(tx, y + row - 1 + 4, UI["basket_unst"], "note", anc))
        word = UI["basket_more"] if v > 0.02 else UI["basket_less"] if v < -0.02 else UI["basket_same"]
        parts.append(f"{lab} — {word}")
    g.append(_text(zero, yb + 14, UI["basket_zero"], "axis", "middle"))
    # «← меньше…» — от левого края, если до нуля не хватает места (телефон), иначе от начала шкалы
    g.append(_text(x0 if W >= 340 else 0, yb + 30, UI["basket_left"], "axis"))
    g.append(_text(x1, yb + 30, UI["basket_right"], "axis", "end"))
    return _svg(
        W, H, "".join(g), UI["basket_aria"].format(name=name, parts="; ".join(parts)), "ch-svg basket"
    )


def chapter_basket(story: Mapping, types: Sequence[Mapping], mo: Mapping[int, Mapping], esc: Esc) -> str:
    """Глава 1: один типичный муниципалитет каждого типа, шаги переключают пример (без JS виден первый)."""
    view = story["view"]
    part_keys = ["food", "marketplace", "transport", "health", "cafe", "other"]
    plabels = story.get("part_labels") or {}
    from munnet.landing import PART_LABELS  # общий словарь частей; импорт здесь — без цикла при загрузке

    labels = [plabels.get(p) or PART_LABELS[p] for p in part_keys]
    panes, steps, rows = [], [], []
    for ty in types:
        ids = (ty.get("examples") or {}).get("typical") or []
        r = next((mo[i] for i in ids if i in mo and (mo[i].get("b24") or mo[i].get("b23"))), None)
        if r is None:
            continue
        year, b = (2024, r["b24"]) if r.get("b24") else (2023, r["b23"])
        t = int(ty["t"])
        unst = {
            part_keys.index(p.replace("clr_rel_", ""))
            for p in ty.get("unstable_parts") or []
            if p.replace("clr_rel_", "") in part_keys
        }
        name = str(r.get("ns") or r.get("n"))
        k = len(panes)
        head = UI["basket_head"].format(name=name, region=r.get("r", ""), year=year)
        panes.append(
            f'<div class="step-pane" data-step="{k}"{" hidden" if k else ""}>'
            f'<p class="fig-head">{_fig_mark(t, view)}{esc(head)} · {esc(_type_name(story, t))}</p>'
            f"{phone_pair(basket_svg(b, labels, unst, name), basket_svg(b, labels, unst, name, PHONE_W))}"
            f'<p><a class="open-card" href="#mo={int(r["id"])}" data-go="{int(r["id"])}">'
            f"{esc(UI['basket_open'])}"
            "</a></p></div>"
        )
        steps.append(
            f'<button type="button" class="step" data-step="{k}" '
            f'aria-pressed="{"true" if not k else "false"}">'
            f"{_fig_mark(t, view)}<span>{esc(name)}</span></button>"
        )
        rows.append(
            [f"{esc(name)}, {esc(r.get('r', ''))}", esc(_type_name(story, t))]
            + [
                esc(
                    UI["basket_more"]
                    if (v or 0) > 0.02
                    else UI["basket_less"]
                    if (v or 0) < -0.02
                    else UI["basket_same"]
                )
                for v in b
            ]
        )
    text = f"<p>{esc(UI['basket_p1'])}</p><p>{esc(UI['basket_p2'])}</p>"
    if not panes:
        return _section("basket", story["chapters"]["basket"]["title"], text, "", esc)
    fig = (
        '<figure class="ch-fig steps-fig" id="basket-fig">'
        + (
            f'<div class="steps" role="group" aria-label="{_e(UI["basket_steps"])}">{"".join(steps)}</div>'
            if len(steps) > 1
            else ""
        )
        + "".join(panes)
        + f'<figcaption class="source">{esc(UI["basket_cap"])} {esc(UI["src_sber"])}</figcaption>'
        + _table(["", "", *labels], rows, esc)
        + "</figure>"
    )
    return _section("basket", story["chapters"]["basket"]["title"], text, fig, esc)


# --- глава 3: малые множества типов (§3.5)


def _row_domains(types: Sequence[Mapping], feats: Sequence[str], ref: Mapping[str, float]) -> dict:
    dom = {}
    for f in feats:
        vals = [ref[f]] if f in ref and ref[f] is not None else []
        for ty in types:
            for p in ty.get("profile") or []:
                if p["feature"] == f:
                    vals += [
                        v
                        for v in (p.get("q25"), p.get("q75"), p.get("lo"), p.get("hi"), p.get("median"))
                        if v is not None
                    ]
        if f.endswith("_rel") or f.startswith("clr_rel_"):
            vals.append(0.0)
        lo, hi = (min(vals), max(vals)) if vals else (-1.0, 1.0)
        pad = (hi - lo) * 0.06 or 0.1
        dom[f] = (lo - pad, hi + pad)
    return dom


def _profile_rows(ty: Mapping) -> dict[str, Mapping]:
    return {p["feature"]: p for p in ty.get("profile") or []}


def _feature_groups(types: Sequence[Mapping]) -> tuple[list[str], list[str], list[str]]:
    feats: list[str] = []
    for ty in types:
        for p in ty.get("profile") or []:
            if p["feature"] not in feats:
                feats.append(p["feature"])
    basket = [f for f in feats if f.startswith("clr_rel_")]
    emp = [f for f in feats if f.startswith("emp_sh_")]
    return basket, [f for f in feats if f not in basket and f not in emp], emp


def _short_label(p: Mapping) -> str:
    """Подпись строки без префикса группы; части корзины — теми же словами, что в главе 1."""
    from munnet.landing import PART_LABELS

    f = str(p["feature"])
    if f.startswith("clr_rel_") and f[8:] in PART_LABELS:
        return PART_LABELS[f[8:]]
    s = str(p.get("label") or f)
    s = s.split(": ", 1)[1] if ": " in s else s
    return s[:1].lower() + s[1:]


def _dots_panel(
    ty: Mapping,
    groups: Sequence[tuple[str, list[str]]],
    dom: Mapping,
    ref: Mapping,
    view: Mapping,
    x0: float,
    x1: float,
    y0: float,
    row: float,
    gap: float,
) -> list[str]:
    """Строки одного типа: серая полоса q25–q75, отрезок 95% интервала и фигура медианы цветом типа.
    Однотипные метки — одним путём на панель (вес страницы)."""
    t = int(ty["t"])
    color = ty.get("color") or view["type_colors"].get(str(t), INK2)
    pr = _profile_rows(ty)
    rules, refs, iqr, ci, med = [], [], [], [], []
    y = y0
    for _, feats in groups:
        y += gap
        for f in feats:
            lo, hi = dom[f]
            sx = lambda v, lo=lo, hi=hi: x0 + (v - lo) / (hi - lo) * (x1 - x0)  # noqa: E731
            cy = y + row / 2
            rules.append(f"M{x0:.0f} {cy:.1f}H{x1:.0f}")
            if f in ref and ref[f] is not None:
                refs.append(f"M{sx(ref[f]):.1f} {y + 2:.1f}V{y + row - 2:.1f}")
            p = pr.get(f)
            if p and p.get("median") is not None:
                if p.get("q25") is not None and p.get("q75") is not None:
                    a, b = sx(p["q25"]), sx(p["q75"])
                    iqr.append(f"M{a:.1f} {cy - 4:.1f}h{max(1, b - a):.1f}v8h{-max(1, b - a):.1f}z")
                if p.get("lo") is not None and p.get("hi") is not None:
                    ci.append(f"M{sx(p['lo']):.1f} {cy:.1f}H{sx(p['hi']):.1f}")
                med.append(_shape_d(t, sx(p["median"]), cy, 4.2, view))
            y += row
    return [
        f'<path class="row-rule" d="{"".join(rules)}"/>',
        f'<path class="ref" d="{"".join(refs)}"/>' if refs else "",
        f'<path class="iqr" d="{"".join(iqr)}"/>' if iqr else "",
        f'<path d="{"".join(ci)}" stroke="{color}" stroke-width="2.5" fill="none"/>' if ci else "",
        f'<path d="{"".join(med)}" fill="{color}" stroke="#fff" stroke-width="1"/>' if med else "",
    ]


def types_svgs(story: Mapping, types: Sequence[Mapping], ref: Mapping[str, float]) -> tuple[str, list[str]]:
    """Широкая SVG (подписи строк и 4 панели в ряд, ≥ 900 px) и по панели на тип (телефон)."""
    view = story["view"]
    basket, place, emp = _feature_groups(types)
    feats = basket + place + emp
    if not feats:
        return "", []
    labels: dict[str, str] = {}
    for ty in types:
        for p in ty.get("profile") or []:
            labels.setdefault(p["feature"], _short_label(p))
    dom = _row_domains(types, feats, ref)
    groups = [
        (n, fs)
        for n, fs in ((UI["types_basket"], basket), (UI["types_place"], place), (UI["types_emp"], emp))
        if fs
    ]
    row, gap, head = 19.0, 22.0, 44.0

    def labels_col(xr: float, y0: float, unst: set[str]) -> list[str]:
        g, y = [], y0
        for title, fs in groups:
            g.append(_text(8, y + gap - 7, title, "grp"))
            y += gap
            for f in fs:
                lab = labels[f] + (" *" if f in unst else "")
                g.append(_text(xr, y + row / 2 + 4, lab, "lab", "end"))
                y += row
        return g

    n_rows = len(feats)
    H = head + len(groups) * gap + n_rows * row + 8
    # широкая
    lab_w, col_w, pad = 176.0, 188.0, 14.0
    W = lab_w + col_w * len(types)
    g = labels_col(lab_w - 6, head, set())
    for i, ty in enumerate(types):
        x0 = lab_w + i * col_w + pad / 2
        x1 = x0 + col_w - pad
        t = int(ty["t"])
        for j, ln in enumerate(_wrap(_type_name(story, t), 26)[:2]):
            g.append(_text(x0, 14 + j * 14, ln, "th"))
        mark = [f.split(":", 1)[-1] for f in ty.get("unstable_parts") or []]
        g += _dots_panel(ty, groups, dom, ref, view, x0, x1, head, row, gap)
        yy = head
        for _, fs in groups:
            yy += gap
            for f in fs:
                if f in mark:
                    g.append(_text(x1 + 3, yy + row / 2 + 4, "*", "note"))
                yy += row
    wide = _svg(W, H, "".join(g), UI["types_scale"], "ch-svg sm-wide")
    # по типу
    panels = []
    for ty in types:
        t = int(ty["t"])
        mark = {f.split(":", 1)[-1] for f in ty.get("unstable_parts") or []}
        # порция 6a: холст 310 — подпись в 11 единиц на 343 px ≥ 12 px; колонка подписей 172 — длинная
        # подпись занятости («стройка, торговля, транспорт») не выходит за левый край
        lw, Wp = 172.0, 310.0
        g = labels_col(lw - 6, 4, mark)
        g += _dots_panel(ty, groups, dom, ref, view, lw + 4, Wp - 8, 4, row, gap)
        Hp = 4 + len(groups) * gap + n_rows * row + 8
        panels.append(
            _svg(Wp, Hp, "".join(g), UI["types_aria"].format(name=_type_name(story, t)), "ch-svg sm-panel")
        )
    return wide, panels


def ami_svg(ami: Sequence[Mapping], n: int = 8, W: float = 360) -> str:
    """Полосы AMI от нуля, сильнейшее деление — акцентом (§3.5, T5 показывается всегда). Узкий холст
    (телефон) — подпись над своей полосой, полоса во всю ширину."""
    rows = list(ami)[:n]
    if not rows:
        return ""
    stacked = True  # порция 6a: подпись над полосой и на широком экране — длинные деления без обрезки слева
    lab_w, row = (0, 32) if stacked else (186, 20)
    vmax = max(0.5, max(float(r["ami"] or 0) for r in rows)) * 1.05
    x0, x1 = (0 if stacked else lab_w + 6), W - 34
    g = []
    y_extra = 0  # порция 6j: добавка высоты от подписей в две строки
    for i, r in enumerate(rows):
        y = 4 + i * row + y_extra
        a = float(r["ami"] or 0)
        w = max(1.0, a / vmax * (x1 - x0))
        lab = str(r["label"])
        yb = y + (14 if stacked else 4)
        if stacked:  # порция 6j: длинная подпись — в две строки (было обрезано многоточием)
            n_ch = int(W / (11 * 0.6))
            lines = _wrap(lab, n_ch)
            if len(lines) > 2:
                lines = [lines[0], " ".join(lines[1:])]
                lines[1] = lines[1] if len(lines[1]) <= n_ch else lines[1][: n_ch - 1] + "…"
            for k, ln in enumerate(lines):
                g.append(_text(0, y + 10 + 13 * k, ln, "lab"))
            yb += 13 * (len(lines) - 1)
            y_extra += 13 * (len(lines) - 1)
        else:
            lab = lab if len(lab) <= 34 else lab[:33] + "…"
            g.append(_text(lab_w, y + 14, lab, "lab", "end"))
        g.append(f'<rect x="{x0}" y="{yb}" width="{w:.1f}" height="12" fill="{ACCENT if i == 0 else INK2}"/>')
        g.append(_text(x0 + w + 4, yb + 10, _f(a, 2), "val"))
    g.append(f'<line class="zero" x1="{x0}" x2="{x0}" y1="2" y2="{4 + len(rows) * row + y_extra}"/>')
    return _svg(W, 8 + len(rows) * row + y_extra, "".join(g), UI["ami_aria"], "ch-svg ami")


def chapter_types(
    story: Mapping,
    types: Sequence[Mapping],
    checks: Mapping,
    ref: Mapping[str, float],
    mo: Mapping[int, Mapping],
    esc: Esc,
    passports: str = "",
    detail: str | None = None,
    rival: str = "",
) -> str:
    ch = story["chapters"]["types"]
    view = story["view"]
    text = f"<p>{esc(UI['types_read'])}</p>"
    if ch.get("explain"):  # порция 6b: пересказ исхода T5 под заголовком (site.build.texts.types_explain)
        text = f'<p class="explain">{esc(ch["explain"])}.</p>' + text
    if ch.get("text"):
        text += f'<details class="how"><summary>{esc(UI["how"])}</summary><p>{esc(ch["text"])}</p></details>'
    if ch.get("note"):
        text += (
            f'<p class="note">{esc(ch["note"])} <span class="label-note">'
            f"{esc(story.get('stability_label') or UI['types_note_label'])}</span></p>"
            if not str(ch["note"]).lower().startswith("описание")
            else f'<p class="note">{esc(ch["note"])}</p>'
        )
    if ch.get("note_text"):
        text += (
            f'<details class="how"><summary>{esc(UI["how_note"])}</summary>'
            f"<p>{esc(ch['note_text'])}</p></details>"
        )
    wide, panels = types_svgs(story, types, ref)
    flag_words = (story.get("card") or {}).get("flag_words") or {}
    cards = []
    for ty, panel in zip(types, panels or [""] * len(types), strict=False):
        t = int(ty["t"])

        def links(ids: Sequence[int], flag: bool = False) -> str:
            out = []
            for i in ids or []:
                r = mo.get(int(i))
                if r:
                    # 03.10 (совет судей): у типичных примеров — флаг «тип зависит от варианта расчёта»
                    fl = str(flag_words.get("d") or "") if flag and r.get("fl") == "d" else ""
                    out.append(
                        f'<a href="#mo={int(i)}" data-go="{int(i)}">{esc(r.get("ns") or r.get("n"))}</a>'
                        f" <small>{esc(r.get('r', ''))}{f' ({esc(fl)})' if fl else ''}</small>"
                    )
            return ", ".join(out)

        ex = ty.get("examples") or {}
        exs = "".join(
            f"<p><b>{esc(UI[k])}:</b> {links(ex.get(key) or [], key == 'typical')}</p>"
            for k, key in (
                ("types_typical", "typical"),
                ("types_border", "borderline"),
                ("types_largest", "largest"),
            )
            if ex.get(key)
        )
        size = "".join(
            f'<span class="size-row"><span class="size-lab">{_pct(v)} {esc(UI[k])}</span>'
            f'<span class="size-track"><i style="width:{(v or 0) * 100:.1f}%;'
            f'background:{_e(ty.get("color") or "")}">'
            "</i></span></span>"
            for k, v in (("types_nodes", ty.get("share_nodes")), ("types_people", ty.get("pop_share")))
        )
        jac = ""
        if ty.get("jaccard") is not None:
            jac = f'<p class="jac">{esc(UI["types_jac"].format(j=_f(ty["jaccard"], 2)))}'
            jac += f' <b class="flag">{esc(UI["types_unstable"])}</b></p>' if ty.get("unstable") else "</p>"
        cards.append(
            f'<div class="sm-card"><h3>{_fig_mark(t, view)}{esc(_type_name(story, t))} '
            f'<small>{_f(ty.get("size") or 0)}</small></h3><p class="size">{size}</p>{jac}{panel}'
            f'<div class="sm-ex">{exs}</div></div>'
        )
    ami = ((checks or {}).get("t5") or {}).get("ami") or []
    ami_html = ""
    if ami:
        n = min(8, len(ami))
        ami_html = (
            f'<div class="ami-block"><h3>{esc(UI["ami_title"])}</h3>'
            f"{phone_pair(ami_svg(ami, n), ami_svg(ami, n, PHONE_W))}"
            f'<p class="source">{esc(UI["ami_cap"].format(n=n))} {esc(UI["src_rosstat"])}</p>'
            + _table(["", "AMI"], [[esc(r["label"]), _f(r["ami"], 3)] for r in ami], esc, UI["alt_ami"])
            + "</div>"
        )
    trows = []
    for ty in types:
        for p in ty.get("profile") or []:
            trows.append(
                [
                    esc(_type_name(story, int(ty["t"]))),
                    esc(_short_label(p)),
                    _f(p["median"], 2),
                    f"{_f(p['lo'], 2)}…{_f(p['hi'], 2)}" if p.get("lo") is not None else "—",
                    f"{_f(p['q25'], 2)}…{_f(p['q75'], 2)}" if p.get("q25") is not None else "—",
                ]
            )
    # порция 5b: паспорта типов сверху; малые множества всех признаков с интервалами — под «Подробнее»
    # (открыто без JS не прячется: <details> раскрывается кликом, текст в HTML)
    full_open = (
        f'<details class="types-full"><summary>{esc(detail)}</summary>' if passports and detail else ""
    )
    fig = (
        '<figure class="ch-fig types-fig">'
        + passports
        + full_open
        + (f'<p class="scale-note">{esc(UI["types_scale"])}</p>' if wide else "")
        + wide
        + f'<div class="sm-grid">{"".join(cards)}</div>'
        + (
            f'<p class="note">{esc(UI["types_unst"])}</p>'
            if any(ty.get("unstable_parts") for ty in types)
            else ""
        )
        + ("</details>" if full_open else "")
        + f'<figcaption class="source">{esc(UI["src_rosstat"])}</figcaption>'
        + (_table(["", "", UI["col_med"], UI["col_ci"], UI["col_iqr"]], trows, esc) if trows else "")
        + ami_html
        + "</figure>"
        + rival
    )
    return _section("types", ch["title"], text, fig, esc, wide=True)


# --- глава 4: проверка порядка (§3.6)


def _log_ticks(lo: float, hi: float) -> list[tuple[float, str]]:
    out = []
    for m, s in ((0.25, "×0,25"), (0.5, "×0,5"), (1, "×1"), (2, "×2"), (4, "×4"), (8, "×8")):
        v = math.log(m)
        if lo <= v <= hi:
            out.append((v, s))
    return out


def order_svgs(story: Mapping, checks: Mapping, W: float = 360) -> tuple[str, str, list[list[str]]]:
    """Панель A — медианы оборота по типам (строки по ``view.t1_layout``, линия по ``line_by_turnover``);
    панель B — сила порядка внутри страт у типов (акцент) и у делений без типов (серые)."""
    view = story["view"]
    t1 = checks.get("t1") or {}
    order = [int(x) for x in checks.get("t1_order") or view.get("legend_order") or []]
    names = [k for k in ("catering", "retail") if k in t1]
    if not t1 or not order:
        return "", "", []
    layout = view.get("t1_layout", "ladder")
    # строки сверху вниз: при «ступенях» — порядок ladder снизу вверх; иначе — по размеру
    rows_top = (
        list(reversed(order)) if layout != "columns_by_size" else [int(x) for x in view["legend_order"]]
    )
    # панель A: строка — фигура и название типа над своими точками (порция 6a: названия вместо «Тип N»)
    row, top = 40, 22
    pw = (W - 4) / max(1, len(names))
    vals = [v for k in names for v in t1[k]["med_a"] if v is not None]
    lo, hi = min([*vals, 0.0]), max([*vals, 0.0])
    lo, hi = lo - (hi - lo) * 0.08, hi + (hi - lo) * 0.08
    H = top + row * len(rows_top) + 18
    cy_of = lambda j: top + j * row + 28  # noqa: E731
    g = []
    labs = []  # подписи строк — поверх линий (с подложкой цвета бумаги, CSS .order-a .lab)
    holes = []  # порция 6i: прямоугольники подписей — в маске линии медиан (линия не пересекает текст)
    for j, t in enumerate(rows_top):
        labs += _type_label(story, t, 0, top + j * row + 12)
        yb = top + j * row + 12
        holes.append(
            f'<rect x="-2" y="{yb - 12}" width="{14 + _text_w(_type_name(story, t), 11) + 8:.1f}" '
            f'height="17" fill="#000"/>'
        )
    mask_id = f"order-a-mask-{int(W)}"  # у широкого и телефонного вариантов на странице — разные id
    table = []
    for i, k in enumerate(names):
        x0 = i * pw + 10
        x1 = x0 + pw - 20
        sx = lambda v, x0=x0, x1=x1: x0 + (v - lo) / (hi - lo) * (x1 - x0)  # noqa: E731
        g.append(_text((x0 + x1) / 2, 12, UI[k], "th", "middle"))
        for v, s_ in _log_ticks(lo, hi):
            for j in range(len(rows_top)):
                g.append(
                    f'<line class="grid" x1="{sx(v):.1f}" x2="{sx(v):.1f}" y1="{cy_of(j) - 8}" '
                    f'y2="{cy_of(j) + 8}"/>'
                )
            g.append(_text(sx(v), top + row * len(rows_top) + 10, s_, "axis", "middle"))
        med = dict(zip(order, t1[k]["med_a"], strict=False))
        line = (view.get("line_by_turnover") or {}).get(k) or t1[k].get("line", "none")
        pts = [(sx(med[t]), cy_of(j)) for j, t in enumerate(rows_top) if med.get(t) is not None]
        if line in ("solid", "dashed") and layout != "columns_by_size" and len(pts) > 1:
            d = "M" + "L".join(f"{x:.1f},{y:.1f}" for x, y in pts)
            dash = ' stroke-dasharray="4 3"' if line == "dashed" else ""
            # порция 6i (judge-c6 № 5): линия прерывается под подписями типов, а не идёт сквозь текст
            g.append(
                f'<path d="{d}" fill="none" stroke="{INK2}" stroke-width="1.5"{dash} mask="url(#{mask_id})"/>'
            )
        op = "" if line in ("solid", "dashed") else ' opacity="0.45"'
        for j, t in enumerate(rows_top):
            if med.get(t) is None:
                continue
            g.append(
                _shape(
                    t,
                    sx(med[t]),
                    cy_of(j),
                    5,
                    view["type_colors"].get(str(t), INK2),
                    view,
                    op + ' stroke="#fff" stroke-width="1"',
                )
            )
        for t in order:
            if med.get(t) is not None:
                table.append([_e(UI[k]), _e(_type_name(story, t)), _f(med[t], 2)])
    aria_a = UI["order_aria_a"].format(turnover=" и ".join(UI[k] for k in names))
    mask = (
        f'<defs><mask id="{mask_id}" maskUnits="userSpaceOnUse" x="0" y="0" width="{W:g}" height="{H:g}">'
        f'<rect x="0" y="0" width="{W:g}" height="{H:g}" fill="#fff"/>{"".join(holes)}</mask></defs>'
    )
    svg_a = _svg(W, H, mask + "".join(g + labs), aria_a, "ch-svg order-a")
    # панель B
    rivals = checks.get("t1_rivals") or []
    W2, lab2, row2, top2 = W, 64, 80, 8
    xs = [r.get(f"rho_b_{k}") for r in rivals for k in names] + [
        v for k in names for v in (t1[k].get("rho_b_ci") or []) + [t1[k].get("rho_b")]
    ]
    xs = [x for x in xs if x is not None]
    blo, bhi = min([*xs, -0.05]) - 0.02, max([*xs, 0.3]) + 0.03
    x0, x1 = lab2 + 6, W2 - 10
    sx = lambda v: x0 + (v - blo) / (bhi - blo) * (x1 - x0)  # noqa: E731
    g = []
    H2 = top2 + row2 * len(names) + 30
    for tick in [x / 10 for x in range(-1, 8)]:
        if blo <= tick <= bhi:
            g.append(
                f'<line class="grid" x1="{sx(tick):.1f}" x2="{sx(tick):.1f}" y1="{top2}" y2="{H2 - 26}"/>'
            )
            g.append(_text(sx(tick), H2 - 13, _f(tick, 1), "axis", "middle"))
    for i, k in enumerate(names):
        y = top2 + i * row2
        cy = y + 46  # облако делений; над ним — типы (акцент), под ним — подпись сильнейшего деления
        g.append(_text(lab2, cy + 4, UI[k], "lab", "end"))
        best, rnd = None, []
        for j, r in enumerate(rivals):
            v = r.get(f"rho_b_{k}")
            if v is None:
                continue
            jy = cy - 11 + ((j * 0.618) % 1) * 22
            if r.get("group") == "random":
                rnd.append(f"M{sx(v) - 2.6:.1f} {jy:.1f}a2.6 2.6 0 1 0 5.2 0a2.6 2.6 0 1 0 -5.2 0")
            else:
                g.append(
                    f'<circle cx="{sx(v):.1f}" cy="{jy:.1f}" r="3.2" fill="{INK2}" opacity="0.75"><title>'
                    f"{_e(r['label'])}: ρ {_f(v, 2)}</title></circle>"
                )
                if best is None or v > best[0]:
                    best = (v, r["label"], jy)
        if rnd:
            g.append(f'<path d="{"".join(rnd)}" fill="none" stroke="#9a978f"/>')
        ci = t1[k].get("rho_b_ci") or []
        yd = y + 22
        if len(ci) == 2:
            g.append(f'<path d="M{sx(ci[0]):.1f} {yd}H{sx(ci[1]):.1f}" stroke="{ACCENT}" stroke-width="2"/>')
        if t1[k].get("rho_b") is not None:
            rx = sx(t1[k]["rho_b"])
            g.append(f'<path d="M{rx:.1f} {yd - 6}l6 6l-6 6l-6 -6z" fill="{ACCENT}"/>')
            lt = UI["order_types"].format(rho=_f(t1[k]["rho_b"], 2))
            g.append(_text(_clamp(rx, _text_w(lt, 11), x0 - 6, W2), yd - 9, lt, "acc", "middle"))
        if best is not None:
            bx = sx(best[0])
            g.append(f'<circle cx="{bx:.1f}" cy="{best[2]:.1f}" r="5.5" fill="none" stroke="{INK2}"/>')
            bl = UI["order_best"].format(label=best[1])
            bl = bl if len(bl) <= 44 else bl[:43] + "…"
            g.append(_text(_clamp(bx, _text_w(bl, 10), 0, W2), y + 73, bl, "note", "middle"))
    aria_b = UI["order_aria_b"].format(turnover=" и ".join(UI[k] for k in names))
    svg_b = _svg(W2, H2, "".join(g), aria_b, "ch-svg order-b")
    return svg_a, svg_b, table


def chapter_order(story: Mapping, checks: Mapping, esc: Esc) -> str:
    ch = story["chapters"]["order"]
    view = story["view"]
    text = f"<p>{esc(UI['order_read'])}</p>"
    if ch.get("text"):
        text += f'<details class="how"><summary>{esc(UI["how"])}</summary><p>{esc(ch["text"])}</p></details>'
    if ch.get("proxies"):
        text += (
            f'<details class="how"><summary>{esc(UI["order_proxies"])}</summary>'
            f'<p>{esc(ch["proxies"])} <span class="label-note">'
            f"{esc(story.get('stability_label') or UI['types_note_label'])}</span></p></details>"
        )
    svg_a, svg_b, table = order_svgs(story, checks)
    pa, pb, _ = order_svgs(story, checks, PHONE_W)
    svg_a, svg_b = phone_pair(svg_a, pa), phone_pair(svg_b, pb)
    if not svg_a:
        return _section("order", ch["title"], text, "", esc)
    lines = {v for v in (view.get("line_by_turnover") or {}).values()}
    lkey = "".join(
        f"<li>{esc(UI['order_line_' + k])}</li>" for k in ("solid", "dashed", "none") if k in lines
    )
    t1 = checks.get("t1") or {}
    brows = []
    for k in ("catering", "retail"):
        if k not in t1:
            continue
        v = t1[k]
        ci = v.get("rho_b_ci") or [None, None]
        brows.append(
            [esc(UI[k]), esc(UI["col_types"]), _f(v.get("rho_b"), 2), f"{_f(ci[0], 2)}…{_f(ci[1], 2)}"]
        )
        for r in checks.get("t1_rivals") or []:
            if r.get(f"rho_b_{k}") is not None:
                brows.append([esc(UI[k]), esc(r["label"]), _f(r[f"rho_b_{k}"], 2), ""])
    fig = (
        '<figure class="ch-fig order-fig">'
        f'<h3>{esc(UI["order_a"])}</h3><p class="note">{esc(UI["order_a_note"])}.</p>{svg_a}'
        f'<p class="axis-note">{esc(UI["order_axis"])}</p>'
        + (f'<ul class="line-key">{lkey}</ul>' if lkey else "")
        + f"<h3>{esc(UI['order_b'])}</h3>{svg_b}"
        f'<ul class="dot-key"><li><i class="k-acc"></i>{esc(UI["order_key_types"])}</li>'
        f'<li><i class="k-dot"></i>{esc(UI["order_key_rivals"])}</li>'
        f'<li><i class="k-ring"></i>{esc(UI["order_key_random"])}</li></ul>'
        f'<p class="axis-note">{esc(UI["order_b_axis"])}</p>'
        f'<p class="rel-key">{esc(rel_key(story, placebo=False))}</p>'
        f'<figcaption class="source">{esc(UI["src_rosstat"])}</figcaption>'
        + _table(["", "", UI["col_med_log"]], table, esc, UI["alt_turnover"])
        + _table(["", "", "ρ", UI["col_ci"]], brows, esc, UI["alt_rivals"])
        + "</figure>"
    )
    return _section("order", ch["title"], text, fig, esc)


# --- глава 5: динамика (§3.7)


def _band(xa: float, ya0: float, ya1: float, xb: float, yb0: float, yb1: float) -> str:
    m = (xa + xb) / 2
    return (
        f"M{xa:.1f},{ya0:.2f}C{m:.1f},{ya0:.2f} {m:.1f},{yb0:.2f} {xb:.1f},{yb0:.2f}"
        f"L{xb:.1f},{yb1:.2f}C{m:.1f},{yb1:.2f} {m:.1f},{ya1:.2f} {xa:.1f},{ya1:.2f}Z"
    )


def flow_label(a: int, b: int, n: int, story: Mapping | None = None) -> str:
    nm = (lambda t: _type_name(story, t)) if story else (lambda t: UI["type_n"].format(t=t))
    return UI["flows_item"].format(a=nm(a), b=nm(b), n=n)


FLOW_STYLE = {
    "solid_saturated": {"opacity": 0.85, "hatch": False, "dash": False},
    "dashed": {"opacity": 0.55, "hatch": False, "dash": True},
    "pale_hatched": {"opacity": 0.4, "hatch": True, "dash": False},
    "grey": {"opacity": 1.0, "hatch": False, "dash": False},
}


def alluvial_svg(story: Mapping, flows: Mapping, W: float = 480) -> str:
    """Аллювиальная диаграмма 4 × 4: оставшиеся — серые, смены — стилем по вердикту T3 (``view.flows``).
    Блоки подписаны фигурой и названием типа (порция 6a); на узком холсте (телефон) справа — фигура и число,
    порядок блоков тот же, названия — слева и в ключе под графиком."""
    view = story["view"]
    order = [int(t) for t in view.get("legend_order") or flows.get("types") or []]
    tot = {k: int(v) for k, v in (flows.get("total") or {}).items()}
    rel = {k: int(v) for k, v in (flows.get("reliable") or {}).items()}
    if not order or not tot:
        return ""
    H, top, gapn = 440, 30, 14
    narrow = W < 400
    lab_w, wrap = (118, 16) if narrow else (150, 21)
    bw = 10 if narrow else 12
    xa = lab_w + 4
    xb = W - (46 if narrow else lab_w + 4) - bw
    left = {a: sum(tot.get(f"{a}-{b}", 0) for b in order) for a in order}
    right = {b: sum(tot.get(f"{a}-{b}", 0) for a in order) for b in order}
    n = max(1, sum(left.values()))
    k = (H - top - 20 - gapn * (len(order) - 1)) / n
    ly, ry, y = {}, {}, top
    for a in order:
        ly[a] = y
        y += left[a] * k + gapn
    y = top
    for b in order:
        ry[b] = y
        y += right[b] * k + gapn
    st = view.get("flows") or {}
    pid = "hflow" if not narrow else "hflowp"
    g = [_hatch(pid)]
    g.append(_text(xa + bw / 2, 14, UI["flows_a"], "th", "middle"))
    g.append(_text(xb + bw / 2, 14, UI["flows_b"], "th", "middle"))
    lo_off = dict.fromkeys(order, 0.0)
    ro_off = dict.fromkeys(order, 0.0)
    bands = []
    for a in order:
        for b in order:
            nt = tot.get(f"{a}-{b}", 0)
            if not nt:
                continue
            nr = rel.get(f"{a}-{b}", 0) if a != b else 0
            pieces = [("stayed", nt)] if a == b else [("reliable", nr), ("noise", nt - nr)]
            for kind, cnt in pieces:
                if cnt <= 0:
                    continue
                h = cnt * k
                y0a, y0b = ly[a] + lo_off[a], ry[b] + ro_off[b]
                lo_off[a] += h
                ro_off[b] += h
                hv = max(h, 0.8)
                bands.append((a, b, kind, cnt, _band(xa + bw, y0a, y0a + hv, xb, y0b, y0b + hv)))
    for a, b, kind, _cnt, d in bands:
        s = FLOW_STYLE.get(st.get(kind, "grey"), FLOW_STYLE["grey"])
        if kind == "stayed":
            g.append(f'<path d="{d}" fill="{GREY}" class="flow stay"/>')
            continue
        color = view["type_colors"].get(str(b), INK2)
        dash = ' stroke="' + color + '" stroke-width="0.8" stroke-dasharray="3 2"' if s["dash"] else ""
        attrs = (
            f' class="flow chg" data-a="{a}" data-b="{b}" tabindex="0" role="button" '
            f'aria-label="{_e(flow_label(a, b, tot.get(f"{a}-{b}", 0), story))}"'
        )  # noqa: E501
        g.append(
            f'<g{attrs}><path d="{d}" fill="{color}" opacity="{s["opacity"]}"{dash}/>'
            + (f'<path d="{d}" fill="url(#{pid})"/>' if s["hatch"] else "")
            + "</g>"
        )
    for a in order:
        c = view["type_colors"].get(str(a), INK2)
        g.append(
            f'<rect x="{xa}" y="{ly[a]:.1f}" width="{bw}" height="{max(1.0, left[a] * k):.1f}" fill="{c}"/>'
        )
        cy = ly[a] + left[a] * k / 2
        nl = len(_wrap(_type_name(story, a), wrap))
        y0 = cy + 4 - nl * 13 / 2
        g += _type_label(story, a, 0, y0, n=wrap)
        g.append(_text(14, y0 + nl * 13, _f(left[a]), "val"))
    for b in order:
        c = view["type_colors"].get(str(b), INK2)
        g.append(
            f'<rect x="{xb}" y="{ry[b]:.1f}" width="{bw}" height="{max(1.0, right[b] * k):.1f}" fill="{c}"/>'
        )
        cy = ry[b] + right[b] * k / 2
        if narrow:
            g.append(_shape(b, xb + bw + 10, cy, 4.5, c, view))
            g.append(_text(xb + bw + 18, cy + 4, _f(right[b]), "val"))
            continue
        nl = len(_wrap(_type_name(story, b), wrap))
        y0 = cy + 4 - nl * 13 / 2
        g += _type_label(story, b, xb + bw + 6, y0, n=wrap)
        g.append(_text(xb + bw + 20, y0 + nl * 13, _f(right[b]), "val"))
    return _svg(W, H, "".join(g), UI["flows_aria"], "ch-svg alluvial")


def placebo_svg(runs: Sequence[Mapping], W: float = 360) -> str:
    """Облака плацебо (точка — псевдогод), пунктир — 95-й перцентиль, акцентная черта — наблюдение;
    ось общая для всех строк."""
    runs = [r for r in runs if r.get("placebo")]
    if not runs:
        return ""
    lab_h, row, x0, x1 = 30, 50, 10, W - 14
    n_lab, n_sub = int(W / (11 * 0.56)), int(W / (10 * 0.56))  # знаков в строке подписи и пометки
    vmax = max(max(max(r["placebo"]), r.get("observed") or 0, r.get("p95") or 0) for r in runs)
    step = 50 if vmax > 150 else 20 if vmax > 60 else 10
    xmax = math.ceil(vmax * 1.04 / step) * step
    sx = lambda v: x0 + v / xmax * (x1 - x0)  # noqa: E731
    g, y = [], 0.0
    for r in runs:
        labs = _wrap(str(r["label"]), n_lab)
        for j, ln in enumerate(labs):
            g.append(_text(x0, y + 12 + j * 13, ln, "lab"))
        y += 13 * (len(labs) - 1)
        sub = _wrap(str(r["sub"]), n_sub) if r.get("sub") else []
        for j, ln in enumerate(sub):
            g.append(_text(x0, y + 25 + j * 12, ln, "note"))
        y0 = y + lab_h + 12 * max(0, len(sub) - 1)
        dots = "".join(
            f"M{sx(v):.1f} {y0 + 6 + ((i * 0.618034) % 1) * (row - 12):.1f}h0"
            for i, v in enumerate(r["placebo"])
        )  # точка — нулевой отрезок с круглым концом: 200 точек одним путём
        g.append(
            f'<path d="{dots}" stroke="{INK2}" stroke-width="3.6" stroke-linecap="round" opacity="0.55"/>'
        )
        if r.get("p95") is not None:
            px = sx(r["p95"])
            g.append(
                f'<line x1="{px:.1f}" x2="{px:.1f}" y1="{y0}" y2="{y0 + row}" stroke="{INK2}" '
                'stroke-dasharray="4 3" stroke-width="1.2"/>'
            )
        if r.get("observed") is not None:
            ox = sx(r["observed"])
            g.append(
                f'<line x1="{ox:.1f}" x2="{ox:.1f}" y1="{y0 - 3}" y2="{y0 + row + 3}" stroke="{ACCENT}" '
                'stroke-width="3"/>'
            )
            anc, tx = ("end", ox - 5) if ox > (x0 + x1) * 0.6 else ("start", ox + 5)
            g.append(_text(tx, y0 + 10, UI["placebo_obs"].format(n=_f(r["observed"])), "acc", anc))
        y = y0 + row + 12
    for v in range(0, int(xmax) + 1, step):
        g.append(f'<line class="grid" x1="{sx(v):.1f}" x2="{sx(v):.1f}" y1="{y - 8}" y2="{y - 4}"/>')
        g.append(_text(sx(v), y + 8, _f(v), "axis", "middle"))
    g.append(_text(x1, y + 22, UI["placebo_axis"], "axis", "end"))
    lab = "; ".join(
        UI["placebo_aria"].format(label=r["label"], obs=_f(r.get("observed") or 0), p95=_f(r.get("p95") or 0))
        for r in runs
    )
    return _svg(W, y + 28, "".join(g), lab, "ch-svg placebo")


def placebo_runs(story: Mapping, t3: Mapping | None) -> list[dict]:
    """Строки облака: итог по R1 и основной расчёт с пометкой ``unstable_label`` (§4.4)."""
    if not t3:
        return []
    out = []
    fin, main = t3.get("final") or {}, t3.get("main") or {}
    same = fin.get("run") == main.get("run")
    if fin and not same:
        out.append(
            {
                "label": UI["placebo_final"].format(
                    variant=VARIANT_WORDS.get(str(fin.get("run")), str(fin.get("run")))
                ),
                "sub": None,
                **{k: fin.get(k) for k in ("placebo", "observed", "p95", "median")},
            }
        )
    if main:
        lab = t3.get("unstable_label") or ""
        out.append(
            {
                "label": UI["placebo_main"].format(label=lab).rstrip(" —")
                if lab
                else VARIANT_WORDS["main"].capitalize(),
                "sub": None,
                **{k: main.get(k) for k in ("placebo", "observed", "p95", "median")},
            }
        )
    for r in out:  # длинная пометка — второй строкой
        if len(r["label"]) > 56:
            head, _, tail = r["label"].partition(" — ")
            r["label"], r["sub"] = head, tail
    return out


def placebo_summary(t3: Mapping | None) -> str:
    """Итог под графиком плацебо словами (порция 6b): заголовок главы — по самому строгому варианту; при его
    вердикте ``not`` — что число смен не отличается от плацебо; если основной расчёт прошёл, а итог ниже —
    что основной расчёт не устоял к вариантам. Числа — ``checks.t3`` (те же, что на графике)."""
    if not t3:
        return ""
    fin, main = t3.get("final") or {}, t3.get("main") or {}
    out = []
    if fin.get("run") and fin.get("run") != main.get("run"):
        out.append(UI["placebo_sum_head"])
        if t3.get("verdict_final") == "not" and fin.get("observed") is not None:
            out.append(
                UI["placebo_sum_final_not"].format(
                    variant=VARIANT_WORDS.get(str(fin["run"]), str(fin["run"])),
                    obs=_f(fin["observed"]),
                    p95=_f(fin.get("p95") or 0),
                )
            )
        if t3.get("unstable") and main.get("passed") and main.get("observed") is not None:
            out.append(
                UI["placebo_sum_main_up"].format(obs=_f(main["observed"]), p95=_f(main.get("p95") or 0))
            )
    return " ".join(out)


def dynamics_texts(tx: Mapping | None, t3: Mapping | None) -> dict[str, str] | None:
    """Заголовок главы 5, её итог и короткая строка для первого экрана (совет судей 03.10,
    ``site.build.texts.dynamics``): как в отчёте — в основном расчёте смен больше, чем на плацебо, при другом
    правиле связей столько же, по правилу ``robustness.main_text`` смена типа не подтверждена.

    Тексты написаны под один исход: итоговый вердикт T3 — ``not``, основной расчёт прошёл (наблюдение выше
    95-го перцентиля плацебо), итог дал прогон ``tx["final_run"]``. Иначе None — остаётся заголовок словаря
    ``site.headlines``. Числа — ``checks.t3`` (те же, что на графике плацебо)."""
    if not tx or not t3:
        return None
    fin, main = t3.get("final") or {}, t3.get("main") or {}
    ok = (
        t3.get("verdict_final") == "not"
        and t3.get("verdict_main") in ("confirmed", "partial")
        and fin.get("run") == tx.get("final_run")
        and main.get("passed")
        and None not in (main.get("observed"), main.get("p95"), fin.get("observed"), fin.get("p95"))
    )
    if not ok or not float(main["observed"]) > float(main["p95"]):
        return None
    vals = {
        "main_obs": _f(main["observed"]),
        "main_p95": _f(main["p95"]),
        "fin_obs": _f(fin["observed"]),
        "fin_p95": _f(fin["p95"]),
    }
    out = {k: str(tx[k]) for k in ("title", "short", "lead")}
    out |= {k: str(tx[k]) for k in ("sub", "limits_context", "card_status") if tx.get(k)}
    lead = out["lead"].format(**vals).rstrip()
    out["lead"] = lead if lead.endswith((".", "!", "?", "…")) else lead + "."  # абзац — с точкой
    return out


def apply_dynamics(story: dict, dt: Mapping[str, str] | None, old_head: str) -> None:
    """Подставляет тексты ``dynamics_texts`` вместо заголовка T3 словаря: заголовок главы 5, итог главы,
    подзаголовок «Что проверяли» и пункт первого экрана, строка «Где граница» (берёт ``screen0.lead``)."""
    if not dt:
        return
    ch = story["chapters"]["dynamics"]
    ch["title"], ch["verdict"] = dt["title"], dt["lead"]
    old_sub = ch.get("lead")
    if dt.get("sub"):  # порция 6j: подзаголовок «Куда шли смены типа» читался как «смены были»
        ch["lead"] = dt["sub"]
    if dt.get("limits_context") and "limits" in story["chapters"]:  # строка перед дословным исходом T6
        story["chapters"]["limits"]["context"] = dt["limits_context"]
    s0 = story["screen0"]
    if s0.get("lead") == old_head:
        s0["lead"] = dt["short"]
    heads = []
    for hs in s0.get("point_heads") or []:
        hs = [dt["short"] if h == old_head else h for h in hs]
        if dt["short"] in hs:  # порция 6j: рядом с итогом нейтральный «Куда шли смены типа» не нужен
            hs = [h for h in hs if h != old_sub] or hs
        heads.append(hs)
    s0["point_heads"] = heads


def chapter_dynamics(story: Mapping, checks: Mapping, esc: Esc) -> str:
    ch = story["chapters"]["dynamics"]
    view = story["view"]
    texts = ch.get("texts") or []
    text = ""
    if ch.get("lead"):
        text += f'<p class="sub">{esc(ch["lead"])}</p>'
    text += f"<p>{esc(UI['dyn_read'])}</p>"
    if ch.get("verdict"):  # 03.10 (совет судей): итог главы с числами обоих расчётов, как в отчёте
        text += f'<p class="dyn-verdict">{esc(ch["verdict"])}</p>'
    if texts:
        # порция 6l (judge-c5 03.10): у каждого текста исхода — из какого он прогона (рядом два разных «148»)
        runs = ch.get("text_runs") or []
        lab = lambda i: f'<b class="run-lab">{esc(runs[i])}.</b> ' if i < len(runs) else ""  # noqa: E731
        # порция 6l, третий круг: своя строка-разложение перед дословным текстом T2 (text_pre)
        pre = ch.get("text_pre") or []
        body = "".join(
            f"<p>{lab(i)}{esc(pre[i].rstrip('.') + '.')}</p><p>{esc(x)}</p>"
            if i < len(pre) and pre[i]
            else f"<p>{lab(i)}{esc(x)}</p>"
            for i, x in enumerate(texts)
        )
        text += f'<details class="how"><summary>{esc(UI["how"])}</summary>{body}</details>'
    flows = checks.get("flows") or {}
    svg = phone_pair(alluvial_svg(story, flows), alluvial_svg(story, flows, PHONE_W))
    runs = placebo_runs(story, checks.get("t3"))
    psvg = phone_pair(placebo_svg(runs), placebo_svg(runs, PHONE_W))
    if not svg and not psvg:
        return _section("dynamics", ch["title"], text, "", esc)
    st = view.get("flows") or {}
    same_style = st.get("reliable") == st.get("noise")
    key = [f'<li><i class="k-stay"></i>{esc(UI["flows_stay"])}</li>']
    if same_style:
        key.append(
            f'<li><i class="k-{_e(st.get("noise", "pale_hatched"))}"></i>{esc(UI["flows_noise"])}</li>'
        )
    else:
        key.append(f'<li><i class="k-{_e(st.get("reliable"))}"></i>{esc(UI["flows_rel"])}</li>')
        key.append(f'<li><i class="k-{_e(st.get("noise"))}"></i>{esc(UI["flows_noise"])}</li>')
    order = [int(t) for t in view.get("legend_order") or flows.get("types") or []]
    names = "".join(f"<li>{_fig_mark(t, view)}{esc(_type_name(story, t))}</li>" for t in order)
    tot, rel = flows.get("total") or {}, flows.get("reliable") or {}
    chg = sorted(
        (
            (int(a), int(b), int(n))
            for k, n in tot.items()
            for a, b in [k.split("-")]
            if a != b and int(n) > 0
        ),
        key=lambda x: -x[2],
    )
    items = "".join(
        f'<li><button type="button" class="flow-btn" data-a="{a}" data-b="{b}" aria-expanded="false">'
        f"{_fig_mark(a, view)}{esc(_type_name(story, a))} → "
        f"{_fig_mark(b, view)}{esc(_type_name(story, b))}"
        f" · {_f(n)}</button></li>"
        for a, b, n in chg
    )
    show_rel = not same_style
    trows = [
        [esc(_type_name(story, a)), esc(_type_name(story, b)), _f(n)]
        + ([_f(int(rel.get(f"{a}-{b}", 0)))] if show_rel else [])
        for a, b, n in chg
    ]
    head = [UI["flows_col_from"], UI["flows_col_to"], UI["flows_col_n"]] + (
        [UI["flows_col_rel"]] if show_rel else []
    )
    # порция 6l: в таблице, как на графике и в тексте, одно опорное число плацебо — 95-й перцентиль
    prow = [
        [
            esc(r["label"] + (" — " + r["sub"] if r.get("sub") else "")),
            _f(r.get("observed") or 0),
            _f(r.get("p95") or 0),
        ]
        for r in runs
    ]
    fig = (
        '<figure class="ch-fig dyn-fig"><div class="dyn-grid">'
        f'<div class="dyn-flows">{svg}<ul class="type-key">{names}</ul>'
        f'<ul class="flow-key">{"".join(key)}</ul>'
        f'<p class="note">{esc(UI["flows_sizes"])}</p>'
        f'<p class="hint">{esc(UI["flows_hint"])}</p>'
        f'<h3 class="flows-h">{esc(UI["flows_changes"])}</h3>'
        f'<ul class="flow-list" id="flow-list">{items}</ul>'
        '<div class="flow-mo" id="flow-mo" aria-live="polite" hidden></div></div>'
        + (
            f'<div class="dyn-placebo"><h3>{esc(UI["placebo_title"])}</h3>{psvg}'
            f'<p class="note">{esc(UI["placebo_key"])}</p>'
            + (
                f'<p class="placebo-sum">{esc(placebo_summary(checks.get("t3")))}</p>'
                if checks.get("t3")
                else ""
            )
            + "</div>"
            if psvg
            else ""
        )
        + "</div>"
        f'<p class="rel-key">{esc(story.get("reliability_key") or "")}</p>'
        f'<figcaption class="source">{esc(UI["src_sber"])}</figcaption>'
        + _table(head, trows, esc, UI["alt_flows"])
        + (
            _table(
                [UI["placebo_col_run"], UI["placebo_col_obs"], UI["placebo_col_p95"]],
                prow,
                esc,
                UI["alt_placebo"],
            )
            if prow
            else ""
        )
        + "</figure>"
    )
    return _section("dynamics", ch["title"], text, fig, esc, wide=True)


# --- сборка


def chapters_html(
    story: Mapping,
    types: Sequence[Mapping],
    checks: Mapping,
    mo: Mapping[int, Mapping],
    ref: Mapping[str, float],
    esc: Esc,
    passports: str = "",
    detail: str | None = None,
    rival: str = "",
) -> str:
    """Главы 1, 3, 4, 5 в порядке страницы (§2). ``esc`` — экранирование с типографикой ru-text;
    ``passports`` — паспорта типов главы 3 (порция 5b, ``site_findings.passports_html``); ``rival`` — карта-
    соперник T5 (порция 6c, ``site_chapters_tail.rival_maps``)."""
    return "\n".join(
        [
            chapter_basket(story, types, mo, esc),
            chapter_types(story, types, checks, ref, mo, esc, passports, detail, rival),
            chapter_order(story, checks, esc),
            chapter_dynamics(story, checks, esc),
        ]
    )
