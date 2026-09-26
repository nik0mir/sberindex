import math

import numpy as np
import pytest
from PIL import Image

from munnet import style
from munnet.style import (
    fmt_krub,
    fmt_num,
    fmt_p,
    fmt_pct,
    fmt_pp,
    fmt_range,
    fmt_rho,
    fmt_rub,
    ru_formatter,
)

THIN, NBSP, MINUS = " ", " ", "−"


@pytest.mark.parametrize(
    ("value", "kwargs", "expected"),
    [
        (1234.5, {"decimals": 1}, "1234,5"),
        (12345, {}, f"12{THIN}345"),
        (-5, {}, f"{MINUS}5"),
        (12345.6, {"decimals": 1}, f"12{THIN}345,6"),
        (-1234567, {}, f"{MINUS}1{THIN}234{THIN}567"),
        (2190, {}, "2190"),
        (0.04, {"decimals": 1, "sign": True}, "0,0"),  # ноль после округления — без знака и без «−0»
        (-0.04, {"decimals": 1}, "0,0"),
        (3.25, {"decimals": 2, "sign": True}, "+3,25"),
        (float("nan"), {}, "—"),
        (None, {}, "—"),
    ],
)
def test_fmt_num(value, kwargs, expected):
    assert fmt_num(value, **kwargs) == expected


def test_units():
    assert fmt_pct(0.125) == "12,5%"
    assert fmt_pct(0.151, sign=True) == "+15,1%"
    assert fmt_pp(4.1) == f"+4,1{NBSP}п.{NBSP}п."
    assert fmt_pp(-2.6) == f"{MINUS}2,6{NBSP}п.{NBSP}п."
    assert fmt_rub(26498) == f"26{THIN}498{NBSP}₽"
    assert fmt_krub(26498) == f"26,5{NBSP}тыс.{NBSP}₽"
    assert fmt_rho(-0.717) == f"{MINUS}0,72"
    assert fmt_rho(0.8654, decimals=3) == "0,865"


def test_p_values():
    assert fmt_p(0.0123) == "0,012"
    assert fmt_p(4.5e-8) == "4,5·10⁻⁸"
    assert fmt_p(9.96e-5) == "1,0·10⁻⁴"  # мантисса 10,0 переносится в порядок


def test_ranges():
    assert fmt_range(9.8, 19.2) == "9,8–19,2"
    assert fmt_range(10, 20) == "10–20"
    assert fmt_range(0.098, 0.192, fmt=fmt_pct) == "9,8–19,2%"
    assert fmt_range(20000, 25000, fmt=lambda v: fmt_krub(v, 0)) == f"20–25{NBSP}тыс.{NBSP}₽"
    assert fmt_range(-5, 10) == f"от{NBSP}{MINUS}5 до{NBSP}10"


def test_ru_formatter_shares_decimals_across_ticks():
    fmt = ru_formatter("pct")
    fmt.set_locs([0.1, 0.15, 0.2])
    assert [fmt(v) for v in (0.1, 0.15, 0.2)] == ["10%", "15%", "20%"]
    fmt = ru_formatter("num")
    fmt.set_locs([0.0, 2.5, 5.0])
    assert [fmt(v) for v in (0.0, 2.5, 5.0)] == ["0,0", "2,5", "5,0"]
    fmt = ru_formatter("krub")
    fmt.set_locs([10000, 15000])
    assert fmt(15000) == f"15{NBSP}тыс.{NBSP}₽"
    with pytest.raises(ValueError):
        ru_formatter("dollars")


def _luminance(hex_color: str) -> float:
    """Относительная яркость по WCAG 2.1."""
    rgb = [int(hex_color[i : i + 2], 16) / 255 for i in (1, 3, 5)]
    lin = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def _contrast(a: str, b: str = "#FFFFFF") -> float:
    la, lb = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def test_palette_contrast_with_white():
    assert set(style.PALETTE) == set(style.LABELS) == set(style.CATEGORY_ORDER)
    for code, color in style.PALETTE.items():
        assert _contrast(color) >= 3.0, code
    assert _contrast(style.TEXT2) >= 4.5  # подзаголовки и источник — текст
    assert len(set(style.PALETTE.values())) == len(style.PALETTE)
    for banned in ("#E69F00", "#999999", "#D55E00"):  # контраст ниже 3:1 или конфликт с транспортом (часть Е)
        assert banned not in style.PALETTE.values() and banned != style.ACCENT


def _line_figure():
    fig, ax = style.new_figure("full")
    x = np.arange(12)
    for code in ("food", "marketplace"):
        ax.plot(x, 0.1 + 0.01 * x, color=style.PALETTE[code])
    ax.set_ylim(0.05, 0.3)
    return fig, ax


def test_finish_russifies_ticks_and_adds_text():
    fig, ax = _line_figure()
    style.finish(
        fig,
        "Доля маркетплейсов выросла вдвое: 9,2 → 19,4%",
        "Медиана по МО, 2023–2024",
        style.SOURCE_SBER,
        11,
    )
    fig.canvas.draw()
    labels = [t.get_text() for t in ax.get_yticklabels() + ax.get_xticklabels() if t.get_text()]
    assert labels and not any("." in s or "-" in s for s in labels)
    texts = [t.get_text() for t in fig.texts]
    assert any(t.startswith("Доля маркетплейсов") for t in texts)
    assert any(f"СберИндекс (CC BY-SA 4.0); расчёт munnet. Рисунок{NBSP}11" in t for t in texts)


def test_finish_wraps_long_title_on_half_figure():
    fig, _ = style.new_figure("half")
    title = "Где тратят меньше, доля маркетплейсов растёт быстрее: +5,5 против +2,6 п. п."
    lines = style.wrap_text(fig, title, style.TITLE_PT, "bold")
    assert len(lines) >= 2 and " ".join(lines) == title


def test_wrap_keeps_short_words_with_next_word():
    fig, _ = style.new_figure("full")
    title = "Свой ритм чаще на Севере: 29% таких МО севернее 60-й параллели, во всей выборке — 10%"
    lines = style.wrap_text(fig, title, style.TITLE_PT, "bold")
    assert len(lines) >= 2 and " ".join(lines) == title
    for line in lines[:-1]:
        last = line.split(" ")[-1].lower()
        assert last not in {"во", "в", "на", "и", "а", "с", "к", "у", "о"}, lines
    assert not any(line.startswith("—") for line in lines[1:])
    # слово «во» у края строки переносится вместе с «всей»
    narrow = style.wrap_text(fig, "параллели, во всей выборке", style.TITLE_PT, "bold", width_in=1.6)
    assert (
        not any(line.endswith(" во") for line in narrow) and " ".join(narrow) == "параллели, во всей выборке"
    )


def test_caption_keeps_figure_number_with_word():
    fig, _ = style.new_figure("half")
    style.finish(
        fig,
        "Заголовок",
        "Подзаголовок",
        style.join_sources(style.SOURCE_SBER, style.SOURCE_ROSSTAT, style.SOURCE_FNS),
        14,
    )
    caption = next(t.get_text() for t in fig.texts if t.get_text().startswith("Источники"))
    assert f"Рисунок{NBSP}14" in caption
    assert not any(line.endswith("Рисунок") for line in caption.splitlines())


def test_two_sources_are_plural():
    fig, _ = _line_figure()
    style.finish(
        fig, "Заголовок", "Подзаголовок", style.join_sources(style.SOURCE_SBER, style.SOURCE_ROSSTAT)
    )
    assert any(t.get_text().startswith("Источники: СберИндекс") for t in fig.texts)


def test_log_rub_axis_and_month_axis():
    fig, ax = style.new_figure("full")
    ax.plot([12_000, 60_000], [1, 2])
    style.log_rub_axis(ax, "x")
    fig.canvas.draw()
    labels = [t.get_text() for t in ax.get_xticklabels() if t.get_text()]
    assert f"20{NBSP}тыс.{NBSP}₽" in labels and f"50{NBSP}тыс.{NBSP}₽" in labels
    fig2, ax2 = style.new_figure("full")
    style.month_axis(ax2, 24, start=0, first_year=2023)
    months = [t.get_text() for t in ax2.get_xticklabels()]
    assert months == ["янв\n2023", "апр", "июл", "окт", "янв\n2024", "апр", "июл", "окт"]
    assert list(ax2.get_xticks()) == [0, 3, 6, 9, 12, 15, 18, 21]
    fig3, ax3 = style.new_figure("half")
    style.month_axis(ax3)
    assert [t.get_text() for t in ax3.get_xticklabels()] == style.MONTHS_RU
    assert list(ax3.get_xticks()) == list(range(1, 13))


def test_label_points_limit_and_no_overlap():
    fig, ax = style.new_figure("full")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    with pytest.raises(ValueError, match="не больше 7"):
        style.label_points(ax, range(8), range(8), [f"МО {i}" for i in range(8)])
    anns = style.label_points(ax, [5] * 5, [5] * 5, ["Арбат", "Хамовники", "Анадырь", "Ямальский", "Певек"])
    renderer = fig.canvas.get_renderer()
    boxes = [a.get_window_extent(renderer=renderer) for a in anns]
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            assert not boxes[i].overlaps(boxes[j])
    assert not any(a.get_in_layout() for a in anns)  # подписи не сжимают поле графика


def _marker_box(ax, x, y, renderer):
    from matplotlib.transforms import Bbox

    cx, cy = ax.transData.transform((x, y))
    half = style.POINT_HALO_PT * ax.figure.dpi / 72
    return Bbox.from_extents(cx - half, cy - half, cx + half, cy + half)


def test_label_points_avoid_other_markers_and_obstacles():
    fig, ax = style.new_figure("full")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    # вторая точка — там, где встала бы подпись первой (вправо-вверх)
    obstacle = ax.text(5.0, 3.0, "пояснение в углу", fontsize=9)
    anns = style.label_points(
        ax,
        [5.0, 5.35, 2.0],
        [5.0, 5.25, 3.05],
        ["Верхнеуслонский", "Свободненский", "Пригород"],
        avoid=[obstacle],
    )
    style.finish(
        fig, "Заголовок", "Подзаголовок, с длинной шапкой, чтобы поле графика сжалось", "СберИндекс", 5
    )
    fig.draw_without_rendering()
    renderer = fig.canvas.get_renderer()
    boxes = [a.get_window_extent(renderer=renderer) for a in anns]
    markers = [_marker_box(ax, x, y, renderer) for x, y in ((5.0, 5.0), (5.35, 5.25), (2.0, 3.05))]
    obstacle_box = obstacle.get_window_extent(renderer=renderer)
    for i, box in enumerate(boxes):
        assert not any(box.overlaps(m) for m in markers), anns[i].get_text()
        assert not box.overlaps(obstacle_box)
        assert all(not box.overlaps(b) for j, b in enumerate(boxes) if j != i)


def test_new_figure_sizes():
    fig, ax = style.new_figure("map")
    assert tuple(fig.get_size_inches()) == style.SIZES["map"]
    with pytest.raises(ValueError):
        style.new_figure("poster")


def test_save_is_reproducible_and_without_dates(tmp_path):
    fig, _ = _line_figure()
    style.finish(fig, "Заголовок", "Подзаголовок", style.SOURCE_SBER, 1)
    first = style.save(fig, tmp_path / "a" / "F01_demo")
    assert [p.suffix for p in first] == [".png", ".svg"] and all(p.exists() for p in first)
    svg = first[1].read_text(encoding="utf-8")
    assert "<dc:date>" not in svg
    assert "Software" not in Image.open(first[0]).info
    second = style.save(fig, tmp_path / "b" / "F01_demo")
    assert first[1].read_bytes() == second[1].read_bytes()
    assert first[0].read_bytes() == second[0].read_bytes()


def test_use_sets_project_rcparams():
    import matplotlib as mpl

    with style.use():
        assert mpl.rcParams["svg.hashsalt"] == "munnet"
        assert mpl.rcParams["axes.formatter.use_locale"] is False
        assert not mpl.rcParams["axes.spines.top"]
    assert math.isclose(style.SIZES["full"][0], 8.0)
