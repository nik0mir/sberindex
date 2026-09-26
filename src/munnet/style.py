"""Единый стиль графиков отчёта и лендинга: палитра, подписи, русские числа, макет, сохранение.

Числа по-русски без ``locale`` (в Windows он ненадёжен): десятичная запятая, разряды узким неразрывным
пробелом U+202F для чисел от 10 000, минус U+2212, проценты слитно, «п. п.» и «₽» через U+00A0.
Макет графика (spec_final, В.2): заголовок-вывод слева, подзаголовок, поле графика, строка источника.
Палитра категорий проверена на контраст с белым (WCAG ≥ 3:1) и различимость при протанопии и дейтеранопии.
"""

from __future__ import annotations

import contextlib
import math
import re
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib as mpl
import numpy as np
from cycler import cycler
from matplotlib import patheffects
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.layout_engine import ConstrainedLayoutEngine
from matplotlib.ticker import (
    FixedLocator,
    FuncFormatter,
    LogFormatter,
    NullFormatter,
    NullLocator,
    ScalarFormatter,
)
from matplotlib.transforms import Bbox

from munnet.contracts import CATEGORY_CODES

# --- Категории: порядок, подписи, цвета ----------------------------------------------------------

CATEGORY_ORDER: tuple[str, ...] = CATEGORY_CODES
LABELS: dict[str, str] = {
    "all": "Все категории",
    "food": "Продовольствие",
    "marketplace": "Маркетплейсы",
    "transport": "Транспорт",
    "health": "Здоровье",
    "cafe": "Общественное питание",
    "other": "Прочее",
}
PALETTE: dict[str, str] = {
    "all": "#222222",
    "food": "#009E73",
    "marketplace": "#0072B2",
    "transport": "#B26B00",
    "health": "#CC79A7",
    "cafe": "#882255",
    "other": "#7F7F7F",
}

TEXT = "#222222"  # основной текст
ACCENT = "#222222"  # фокусные точки и линии на графиках без категорий
CONTEXT = "#B8B8B8"  # фон: точки всех МО за подписанными
TEXT2 = "#595959"  # подзаголовок, источник, деления (контраст 7:1)
GRID = "#E6E6E6"
NODATA = "#E0E0E0"  # «нет данных» — заливка со штриховкой HATCH
HATCH = "#9E9E9E"
SEQ_CMAP = "cividis"
DIV_CMAP = "PuOr"

MONTHS_RU: list[str] = ["янв", "фев", "мар", "апр", "май", "июн", "июл", "авг", "сен", "окт", "ноя", "дек"]
THIN, NBSP, MINUS = " ", " ", "−"
EN_DASH = "–"
NA_TEXT = "—"  # пропуск в таблицах и подписях

SOURCE_SBER = "СберИндекс (CC BY-SA 4.0)"
SOURCE_ROSSTAT = "Росстат, БД ПМО в обработке «Если быть точным» (CC BY 4.0)"
SOURCE_FNS = "ФНС, 5-НДФЛ в обработке «Если быть точным»"
SOURCE_SEP = "; "


def join_sources(*sources: str) -> str:
    """Несколько источников в одну строку для ``finish`` (через «; »)."""
    return SOURCE_SEP.join(sources)


# --- Числа по-русски -----------------------------------------------------------------------------

_MIN_GROUPED_DIGITS = 5  # разряды отбиваются только у чисел от 10 000; 2190 пишется слитно
_AUTO_MAX_DECIMALS = 2
_EXACT_TOL = 1e-9


def _is_missing(x: Any) -> bool:
    if x is None:
        return True
    try:
        return bool(np.isnan(x))
    except (TypeError, ValueError):
        return False


def fmt_num(x: float, decimals: int = 0, sign: bool = False) -> str:
    """Число по-русски: 12345.6, 1 → «12 345,6»; 1234 → «1234»; −5 → «−5»; пропуск → «—».

    Разряды — U+202F, только если в целой части не меньше пяти цифр; минус — U+2212; ``sign`` ставит «+»
    перед положительными (ноль после округления остаётся без знака).
    """
    if _is_missing(x):
        return NA_TEXT
    x = float(x)
    if math.isinf(x):
        return ("" if x > 0 else MINUS) + "∞"
    body = f"{abs(x):.{decimals}f}"
    is_zero = float(body) == 0.0
    int_part, _, frac = body.partition(".")
    if len(int_part) >= _MIN_GROUPED_DIGITS:
        head = len(int_part) % 3 or 3
        groups = [int_part[:head]] + [int_part[i : i + 3] for i in range(head, len(int_part), 3)]
        int_part = THIN.join(groups)
    text = int_part + ("," + frac if frac else "")
    if x < 0 and not is_zero:
        return MINUS + text
    if sign and x > 0 and not is_zero:
        return "+" + text
    return text


def fmt_pct(share: float, decimals: int = 1, sign: bool = False) -> str:
    """Доля как процент: 0.151 → «15,1%»; ``sign`` → «+15,1%»."""
    if _is_missing(share):
        return NA_TEXT
    return fmt_num(100.0 * float(share), decimals, sign) + "%"


def fmt_pp(pp: float, decimals: int = 1, sign: bool = True) -> str:
    """Процентные пункты: 4.1 → «+4,1 п. п.» (U+00A0 перед «п.» и внутри)."""
    if _is_missing(pp):
        return NA_TEXT
    return f"{fmt_num(pp, decimals, sign)}{NBSP}п.{NBSP}п."


def fmt_rub(x: float, decimals: int = 0) -> str:
    """Рубли: 26498 → «26 498 ₽»."""
    if _is_missing(x):
        return NA_TEXT
    return f"{fmt_num(x, decimals)}{NBSP}₽"


def fmt_krub(x: float, decimals: int = 1) -> str:
    """Рубли в тысячах: 26498 → «26,5 тыс. ₽» (вход — в рублях)."""
    if _is_missing(x):
        return NA_TEXT
    return f"{fmt_num(float(x) / 1000.0, decimals)}{NBSP}тыс.{NBSP}₽"


def fmt_rho(x: float, decimals: int = 2, sign: bool = False) -> str:
    """Корреляция: −0.717 → «−0,72» (в контрольных числах — ``decimals=3``)."""
    return fmt_num(x, decimals, sign)


_SUPERSCRIPT = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


def fmt_p(p: float, decimals: int = 3) -> str:
    """p-значение: 0.0123 → «0,012»; меньше 0,001 — мантисса и степень: 4.5e−8 → «4,5·10⁻⁸»."""
    if _is_missing(p):
        return NA_TEXT
    p = float(p)
    if p == 0.0:
        return "0"
    if p >= 10.0**-decimals:
        return fmt_num(p, decimals)
    exp = math.floor(math.log10(p))
    mant = p / 10.0**exp
    if round(mant, 1) >= 10.0:  # 9,96·10⁻⁵ округляется до 10,0 — переносим в порядок
        mant, exp = mant / 10.0, exp + 1
    return f"{fmt_num(mant, 1)}·10{str(exp).translate(_SUPERSCRIPT)}"


def auto_decimals(values: Iterable[float], max_decimals: int = _AUTO_MAX_DECIMALS) -> int:
    """Наименьшее число знаков после запятой (не больше ``max_decimals``), при котором все значения точны."""
    vals = [float(v) for v in values if not _is_missing(v) and math.isfinite(float(v))]
    for d in range(max_decimals + 1):
        if all(abs(round(v, d) - v) <= _EXACT_TOL * max(1.0, abs(v)) for v in vals):
            return d
    return max_decimals


def _numeric_suffix(text: str) -> str:
    """Хвост строки после последней цифры: «%», « п. п.», « ₽», « тыс. ₽»."""
    for i in range(len(text) - 1, -1, -1):
        if text[i].isdigit():
            return text[i + 1 :]
    return ""


def fmt_range(a: float, b: float, fmt: Callable[..., str] = fmt_num, **kw: Any) -> str:
    """Диапазон: 9.8, 19.2 → «9,8–19,2»; с ``fmt=fmt_pct``: 0.098, 0.192 → «9,8–19,2%».

    Тире «–» без пробелов, общая единица — один раз в конце. Для ``fmt_num`` без ``decimals`` число
    знаков подбирается так, чтобы оба конца были точны (не больше двух). Если край отрицательный,
    получается «от −5 до 10»: «−5–10» читается плохо.
    """
    if fmt is fmt_num and "decimals" not in kw:
        kw["decimals"] = auto_decimals((a, b))
    sa, sb = fmt(a, **kw), fmt(b, **kw)
    suffix = _numeric_suffix(sb)
    if suffix and _numeric_suffix(sa) == suffix:
        sa = sa[: -len(suffix)]
    if (not _is_missing(a) and a < 0) or (not _is_missing(b) and b < 0):
        return f"от{NBSP}{sa} до{NBSP}{sb}"
    return f"{sa}{EN_DASH}{sb}"


# --- Оси -----------------------------------------------------------------------------------------

_KINDS: dict[str, tuple[Callable[[float], float], Callable[[float, int], str]]] = {
    "num": (lambda v: v, lambda v, d: fmt_num(v, d)),
    "pct": (lambda v: 100.0 * v, lambda v, d: fmt_pct(v, d)),
    "rub": (lambda v: v, lambda v, d: fmt_rub(v, d)),
    "krub": (lambda v: v / 1000.0, lambda v, d: fmt_krub(v, d)),
    "pp": (lambda v: v, lambda v, d: fmt_pp(v, d, sign=True)),
}


class RuFormatter(FuncFormatter):
    """Подписи делений по-русски; без явного ``decimals`` число знаков общее для всех делений оси."""

    def __init__(self, kind: str = "num", decimals: int | None = None) -> None:
        if kind not in _KINDS:
            raise ValueError(f"неизвестный вид подписи {kind!r}; допустимы {sorted(_KINDS)}")
        self.kind = kind
        self.decimals = decimals
        self._auto = 0
        super().__init__(self._format)

    def set_locs(self, locs: Sequence[float]) -> None:
        super().set_locs(locs)
        if self.decimals is None:
            scale = _KINDS[self.kind][0]
            self._auto = auto_decimals(scale(v) for v in locs)

    def _format(self, x: float, pos: int | None = None) -> str:
        d = self.decimals if self.decimals is not None else self._auto
        return _KINDS[self.kind][1](x, d)


def ru_formatter(kind: str = "num", decimals: int | None = None) -> FuncFormatter:
    """Форматтер делений: ``num`` «12 345», ``pct`` (вход — доля) «15%», ``rub`` «26 498 ₽»,
    ``krub`` (вход — рубли) «25 тыс. ₽», ``pp`` «+4 п. п.»."""
    return RuFormatter(kind, decimals)


DEFAULT_RUB_TICKS = (10_000, 15_000, 20_000, 30_000, 50_000, 80_000)
LONG_AXIS_STEP = 3  # у рядов длиннее года подписан каждый третий месяц


def log_rub_axis(ax: Axes, axis: str = "x", ticks: Sequence[float] = DEFAULT_RUB_TICKS) -> None:
    """Логарифмическая ось уровня трат с делениями в рублях: «10 тыс. ₽», «15 тыс. ₽»…"""
    if axis not in ("x", "y"):
        raise ValueError(f"axis — 'x' или 'y', получено {axis!r}")
    getattr(ax, f"set_{axis}scale")("log")
    target = ax.xaxis if axis == "x" else ax.yaxis
    target.set_major_locator(FixedLocator(list(ticks)))
    target.set_major_formatter(ru_formatter("krub"))
    target.set_minor_locator(NullLocator())
    target.set_minor_formatter(NullFormatter())


def month_axis(
    ax: Axes, n_months: int = 12, *, start: int = 1, first_year: int | None = None, step: int | None = None
) -> None:
    """Ось месяцев по-русски: деления ``start…start+n−1`` («янв»…«дек»).

    ``start=1`` — для колонки ``month`` (1…12), ``start=0`` — для ``t`` (0…23). У ряда длиннее года подписан
    каждый ``step``-й месяц (по умолчанию квартал: янв, апр, июл, окт), при ``first_year`` под январями — год.
    """
    step = step or (1 if n_months <= 12 else LONG_AXIS_STEP)
    positions, labels = [], []
    for i in range(0, n_months, step):
        label = MONTHS_RU[i % 12]
        if first_year is not None and n_months > 12 and i % 12 == 0:
            label = f"{label}\n{first_year + i // 12}"
        positions.append(start + i)
        labels.append(label)
    ax.set_xticks(positions, labels)
    ax.xaxis.set_minor_locator(
        FixedLocator([start + i for i in range(n_months)]) if step > 1 else NullLocator()
    )


# --- rcParams и макет ----------------------------------------------------------------------------

TITLE_PT = 13
SUBTITLE_PT = 10
LABEL_PT = 10
TICK_PT = 9
SOURCE_PT = 8
POINT_LABEL_PT = 8

RC: dict[str, Any] = {
    "font.family": "DejaVu Sans",
    "font.size": LABEL_PT,
    "text.color": TEXT,
    "axes.titlesize": TITLE_PT,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.labelsize": LABEL_PT,
    "axes.labelcolor": TEXT,
    "axes.edgecolor": TEXT2,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "axes.axisbelow": True,
    "axes.facecolor": "white",
    "axes.formatter.use_locale": False,
    "axes.unicode_minus": True,
    "axes.prop_cycle": cycler(color=[PALETTE[c] for c in CATEGORY_ORDER]),
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "xtick.labelsize": TICK_PT,
    "ytick.labelsize": TICK_PT,
    "xtick.color": TEXT2,
    "ytick.color": TEXT2,
    "legend.frameon": False,
    "legend.fontsize": TICK_PT,
    "lines.linewidth": 1.8,
    "hatch.color": HATCH,
    "hatch.linewidth": 0.6,
    "image.cmap": SEQ_CMAP,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "svg.fonttype": "none",
    "svg.hashsalt": "munnet",
    "pdf.fonttype": 42,
}

# Размеры фигур, дюймы: при 200 dpi PNG шириной 8" на GitHub ≈ 880 px, кегль 10 pt читается.
SIZES: dict[str, tuple[float, float]] = {
    "full": (8.0, 4.5),
    "half": (4.0, 3.2),
    "map": (8.0, 5.2),
    "tall": (8.0, 6.0),
}

# Поля макета, дюймы.
MARGIN_IN = 0.15  # левый край заголовка и источника
TOP_PAD_IN = 0.12  # от верха фигуры до заголовка
TITLE_GAP_IN = 0.05  # между заголовком и подзаголовком
HEADER_GAP_IN = 0.12  # от подзаголовка до поля графика
BOTTOM_PAD_IN = 0.08  # от низа фигуры до строки источника
FOOTER_GAP_IN = 0.08  # от поля графика до строки источника
LINE_SPACING = 1.25  # высота строки в кеглях


@contextlib.contextmanager
def use() -> Iterator[None]:
    """rcParams проекта на время блока."""
    with mpl.rc_context(RC):
        yield


def new_figure(kind: str = "full", nrows: int = 1, ncols: int = 1, **kw: Any) -> tuple[Figure, Any]:
    """Фигура проекта без pyplot: размер из ``SIZES``, макет constrained, rcParams проекта.

    Возвращает ``(fig, ax)``; при нескольких панелях ``ax`` — массив осей, как у ``subplots``.
    """
    if kind not in SIZES:
        raise ValueError(f"неизвестный вид фигуры {kind!r}; допустимы {sorted(SIZES)}")
    with use():
        fig = Figure(figsize=SIZES[kind], layout="constrained")
        FigureCanvasAgg(fig)
        ax = fig.subplots(nrows, ncols, **kw)
    return fig, ax


def _renderer(fig: Figure):
    if not hasattr(fig.canvas, "get_renderer"):
        FigureCanvasAgg(fig)
    return fig.canvas.get_renderer()


def text_width_in(fig: Figure, text: str, size: float, weight: str = "normal") -> float:
    """Ширина строки текста на фигуре, дюймы (шрифт проекта)."""
    t = fig.text(0, 0, text, fontsize=size, fontweight=weight)
    width = t.get_window_extent(renderer=_renderer(fig)).width / fig.dpi
    t.remove()
    return width


# Склейка при переносе: короткий предлог, союз или частица не остаётся в конце строки, тире и «=» не
# начинают строку. Склеенные слова переносятся вместе, в готовой строке остаётся обычный пробел.
_GLUE = ""
_SHORT_WORDS = re.compile(
    r"(?<![\w-])(в|во|к|ко|с|со|о|об|обо|у|и|а|но|да|на|по|от|до|из|за|не|ни|для|без|при|над|под|про) ",
    re.IGNORECASE,
)


def _glue_short_words(text: str) -> str:
    text = _SHORT_WORDS.sub(lambda m: m.group(1) + _GLUE, text)
    return text.replace(" — ", f"{_GLUE}— ").replace(" = ", f"{_GLUE}={_GLUE}")


def wrap_text(
    fig: Figure, text: str, size: float, weight: str = "normal", width_in: float | None = None
) -> list[str]:
    """Переносит текст по пробелам в ширину фигуры.

    Неразрывные пробелы не рвутся; короткие предлоги и союзы («в», «во», «на», «и»…) переносятся вместе
    со следующим словом, тире и «=» — вместе с предыдущим (правила ru-text). Текст строк совпадает
    с исходным, меняются только места переноса.
    """
    width_in = (fig.get_figwidth() - 2 * MARGIN_IN) if width_in is None else width_in
    lines: list[str] = []
    current = ""
    for word in _glue_short_words(text).split(" "):
        trial = f"{current} {word}" if current else word
        if not current or text_width_in(fig, trial.replace(_GLUE, " "), size, weight) <= width_in:
            current = trial
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return [line.replace(_GLUE, " ") for line in lines]


def _russify_axes(fig: Figure) -> None:
    """Стандартные форматтеры matplotlib (десятичная точка, «1e6») заменяются русскими."""
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            fmt = axis.get_major_formatter()
            if type(fmt) is ScalarFormatter:
                axis.set_major_formatter(ru_formatter("num"))
            elif isinstance(fmt, LogFormatter):
                axis.set_major_formatter(ru_formatter("num"))
                axis.set_minor_formatter(NullFormatter())


def finish(fig: Figure, title: str, subtitle: str, source: str, number: int | None = None) -> None:
    """Оформляет фигуру: заголовок-вывод, подзаголовок, строка источника, русские подписи делений.

    Заголовок и подзаголовок переносятся по ширине фигуры; поле графика сжимается так, чтобы шапка и
    подвал не наезжали на оси. ``source`` — один источник или ``join_sources(...)``; к нему добавляется
    «расчёт munnet» и «Рисунок N». Легенды — только легенды осей (``ax.legend``, в том числе за краем осей):
    их макет учитывает, а ``fig.legend(loc="outside …")`` встаёт у края фигуры и наезжает на источник.
    Подписи точек ``label_points`` раскладываются заново по окончательному месту поля графика.
    """
    _russify_axes(fig)
    width, height = fig.get_figwidth(), fig.get_figheight()
    x = MARGIN_IN / width
    y_in = TOP_PAD_IN
    title_lines = wrap_text(fig, title, TITLE_PT, "bold")
    fig.text(
        x,
        1 - y_in / height,
        "\n".join(title_lines),
        fontsize=TITLE_PT,
        fontweight="bold",
        color=TEXT,
        ha="left",
        va="top",
        linespacing=LINE_SPACING,
    )
    y_in += len(title_lines) * TITLE_PT * LINE_SPACING / 72 + TITLE_GAP_IN
    if subtitle:
        sub_lines = wrap_text(fig, subtitle, SUBTITLE_PT)
        fig.text(
            x,
            1 - y_in / height,
            "\n".join(sub_lines),
            fontsize=SUBTITLE_PT,
            color=TEXT2,
            ha="left",
            va="top",
            linespacing=LINE_SPACING,
        )
        y_in += len(sub_lines) * SUBTITLE_PT * LINE_SPACING / 72
    header_in = y_in + HEADER_GAP_IN

    label = "Источники" if SOURCE_SEP in source else "Источник"
    caption = f"{label}: {source}; расчёт munnet."
    if number is not None:
        caption += f" Рисунок{NBSP}{number}"  # номер не отрывается от слова при переносе
    cap_lines = wrap_text(fig, caption, SOURCE_PT)
    fig.text(
        x,
        BOTTOM_PAD_IN / height,
        "\n".join(cap_lines),
        fontsize=SOURCE_PT,
        color=TEXT2,
        ha="left",
        va="bottom",
        linespacing=LINE_SPACING,
    )
    footer_in = BOTTOM_PAD_IN + len(cap_lines) * SOURCE_PT * LINE_SPACING / 72 + FOOTER_GAP_IN

    bottom, top = footer_in / height, 1 - header_in / height
    engine = fig.get_layout_engine()
    if isinstance(engine, ConstrainedLayoutEngine):
        engine.set(rect=(0.0, bottom, 1.0, top - bottom))
    else:
        fig.subplots_adjust(bottom=max(bottom, fig.subplotpars.bottom), top=min(top, fig.subplotpars.top))
    _replace_point_labels(fig)


# Кандидаты смещения подписи точки, pt: сначала вправо-вверх, затем по кругу и дальше.
_LABEL_OFFSETS = [
    (6, 6),
    (6, -6),
    (-6, 6),
    (-6, -6),
    (0, 10),
    (0, -12),
    (10, 0),
    (-10, 0),
    (14, 14),
    (14, -14),
    (-14, 14),
    (-14, -14),
    (0, 20),
    (0, -22),
]


POINT_MARKER_S = 18  # площадь маркера выделенной точки, pt²
POINT_HALO_PT = 3.0  # полуразмер занятой области вокруг выделенной точки: подпись её не закрывает, pt
_LABEL_GROUPS_ATTR = "_munnet_point_labels"


@dataclass
class _LabelGroup:
    """Подписи одного вызова ``label_points``: точки, аннотации и тексты-препятствия."""

    ax: Axes
    xy: list[tuple[float, float]]
    annotations: list[Any]
    avoid: list[Any] = field(default_factory=list)


def _offset_align(dx: float, dy: float) -> tuple[str, str]:
    return (
        "left" if dx > 0 else "right" if dx < 0 else "center",
        "bottom" if dy > 0 else "top" if dy < 0 else "center",
    )


def _set_offset(ann: Any, dx: float, dy: float) -> None:
    ha, va = _offset_align(dx, dy)
    ann.xyann = (dx, dy)
    ann.set_horizontalalignment(ha)
    ann.set_verticalalignment(va)


def _on_figure(artist: Any) -> bool:
    """Художник на фигуре и видим (убранный ``remove()`` — нет)."""
    return getattr(artist, "figure", None) is not None and artist.get_visible()


def _place_group(group: _LabelGroup, taken: list, renderer: Any) -> None:
    """Раскладывает подписи группы по кандидатам ``_LABEL_OFFSETS``: подпись внутри поля графика и не
    пересекает ``taken`` (уже поставленные подписи), тексты ``avoid`` и выделенные точки всех групп осей."""
    ax = group.ax
    fig = ax.figure
    ax_box = ax.get_window_extent(renderer=renderer)
    halo = POINT_HALO_PT * fig.dpi / 72
    groups = [g for g in getattr(fig, _LABEL_GROUPS_ATTR, []) if g.ax is ax] or [group]
    points = [xy for g in groups for xy in g.xy]
    centers = ax.transData.transform(np.asarray(points, dtype="float64")) if points else np.empty((0, 2))
    markers = [Bbox.from_extents(cx - halo, cy - halo, cx + halo, cy + halo) for cx, cy in centers]
    obstacles = [a.get_window_extent(renderer=renderer) for a in group.avoid if _on_figure(a)]
    for ann in group.annotations:
        if not _on_figure(ann):  # подпись уже убрана с графика
            continue
        chosen = None
        for dx, dy in _LABEL_OFFSETS:
            _set_offset(ann, dx, dy)
            box = ann.get_window_extent(renderer=renderer)
            inside = ax_box.contains(box.x0, box.y0) and ax_box.contains(box.x1, box.y1)
            blocked = any(box.overlaps(b) for b in (*taken, *obstacles, *markers))
            if inside and not blocked:
                chosen = box
                break
        if chosen is None:  # места нет: первое положение
            _set_offset(ann, *_LABEL_OFFSETS[0])
            chosen = ann.get_window_extent(renderer=renderer)
        taken.append(chosen)


def _replace_point_labels(fig: Figure) -> None:
    """Раскладывает подписи ``label_points`` заново после раскладки поля графика (вызывает ``finish``):
    шапка и подвал сжимают поле, и подписи, разведённые на полном поле, иначе сходились бы."""
    groups = getattr(fig, _LABEL_GROUPS_ATTR, [])
    if not groups:
        return
    fig.draw_without_rendering()
    renderer = _renderer(fig)
    taken: dict[int, list] = {}
    for group in groups:
        _place_group(group, taken.setdefault(id(group.ax), []), renderer)


def label_points(
    ax: Axes,
    x: Sequence[float],
    y: Sequence[float],
    labels: Sequence[str],
    *,
    max_labels: int = 7,
    color: str = ACCENT,
    fontsize: float = POINT_LABEL_PT,
    avoid: Sequence[Any] = (),
) -> list:
    """Выделяет точки цветом ``color`` и подписывает их, раздвигая подписи.

    Подпись не пересекает другие подписи, выделенные точки (свои и чужие) и тексты ``avoid`` (подписи
    линий, пояснения в углу поля) и не выходит за поле графика. Подписи не участвуют в раскладке
    constrained (не сжимают поле) и раскладываются заново в ``finish``, когда поле графика встанет на
    окончательное место. Больше ``max_labels`` подписей — ``ValueError``: правило отчёта — не больше 5–7
    подписей МО на графике. Возвращает список аннотаций.
    """
    xs, ys, texts = list(x), list(y), [str(t) for t in labels]
    if not len(xs) == len(ys) == len(texts):
        raise ValueError("x, y и labels должны быть одной длины")
    if len(texts) > max_labels:
        raise ValueError(f"подписей {len(texts)}, а допустимо не больше {max_labels}")
    ax.scatter(xs, ys, s=POINT_MARKER_S, color=color, zorder=3, linewidths=0)
    dx, dy = _LABEL_OFFSETS[0]
    ha, va = _offset_align(dx, dy)
    annotations = [
        ax.annotate(
            text,
            (xi, yi),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha,
            va=va,
            fontsize=fontsize,
            color=TEXT,
            zorder=4,
            path_effects=[patheffects.withStroke(linewidth=2.5, foreground="white")],
            in_layout=False,
        )
        for xi, yi, text in zip(xs, ys, texts, strict=True)
    ]
    fig = ax.figure
    group = _LabelGroup(ax, list(zip(xs, ys, strict=True)), annotations, list(avoid))
    groups = getattr(fig, _LABEL_GROUPS_ATTR, None)
    if groups is None:
        groups = []
        setattr(fig, _LABEL_GROUPS_ATTR, groups)
    groups.append(group)
    renderer = _renderer(fig)
    taken = [a.get_window_extent(renderer=renderer) for g in groups[:-1] if g.ax is ax for a in g.annotations]
    _place_group(group, taken, renderer)
    return annotations


def save(fig: Figure, stem: Path, formats: Sequence[str] = ("png", "svg"), dpi: int = 200) -> list[Path]:
    """Сохраняет фигуру в ``stem.<формат>`` без даты и версии ПО в метаданных.

    Повторная сборка даёт те же байты SVG (``svg.hashsalt``), а PNG — без ложных изменений метаданных.
    """
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    metadata = {"png": {"Software": None}, "svg": {"Date": None}, "pdf": {"CreationDate": None}}
    paths = []
    with use():
        for fmt in formats:
            path = stem.parent / f"{stem.name}.{fmt}"
            fig.savefig(path, format=fmt, dpi=dpi, metadata=metadata.get(fmt))
            paths.append(path)
    return paths
