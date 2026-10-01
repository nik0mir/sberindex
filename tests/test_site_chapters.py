"""Главы 1, 3, 4, 5 лендинга (``src/munnet/site_chapters.py``, §3.4–§3.7 ``docs/landing_spec.md``):
форма главы 4 по вердикту T1, стиль потоков главы 5 по T3, облако плацебо без наблюдения внутри,
интерфейс глав без запрещённых слов при любых вердиктах. Без сети и без data/: входы — маленькие словари."""

from __future__ import annotations

import html
import re

import numpy as np
import pandas as pd
import pytest
from fixtures.site.synth_facts import all_combos, combo_levels

from munnet import landing
from munnet import site_chapters as SC
from munnet.config import load_config
from munnet.site_headlines import HeadlineChecker

CFG = load_config("configs/default.yaml")
ESC = html.escape
TYPES4 = [1, 2, 3, 4]
COLORS = {"1": "#501d42", "2": "#b4561d", "3": "#3198c8", "4": "#185d4c"}
SHAPES = {"1": "●", "2": "▲", "3": "■", "4": "◆"}


def story(**view) -> dict:
    v = {
        "type_colors": COLORS,
        "shapes": SHAPES,
        "legend_order": [1, 2, 3, 4],
        "t1_layout": "ladder",
        "line_by_turnover": {"catering": "solid", "retail": "dashed"},
        "flows": {"reliable": "solid_saturated", "noise": "pale_hatched", "stayed": "grey"},
        "arrows": False,
        "unstable_parts": [],
    } | view
    return {
        "view": v,
        "names": {"final": {str(t): f"Тип {t}: пример" for t in TYPES4}},
        "reliability_key": "Сплошная — проверка пройдена",
        "chapters": {
            "basket": {"title": "Что сравниваем"},
            "types": {
                "title": "Заголовок T5",
                "text": "Текст T5.",
                "note": "Описание: …",
                "note_text": "T4.",
            },
            "order": {"title": "Заголовок T1", "text": "Текст T1.", "proxies": "Описание прокси."},
            "dynamics": {
                "title": "Заголовок T3",
                "lead": "Куда шли смены типа 2023 → 2024",
                "texts": ["T3.", "T2."],
            },
        },
    }


def types() -> list[dict]:
    prof = lambda m: [  # noqa: E731
        {"feature": f, "label": f, "median": m, "lo": m - 0.1, "hi": m + 0.1, "q25": m - 0.3, "q75": m + 0.3}
        for f in ("clr_rel_food", "log_wage_rel", "emp_sh_public")
    ]
    return [
        {
            "t": t,
            "color": COLORS[str(t)],
            "size": 10 * t,
            "share_nodes": 0.25,
            "pop_share": 0.1 * t,
            "jaccard": 0.7 if t == 4 else 0.9,
            "unstable": t == 4,
            "unstable_parts": ["clr_rel_food"] if t == 1 else [],
            "profile": prof(0.1 * t),
            "examples": {"typical": [100 + t], "borderline": [], "largest": []},
        }
        for t in TYPES4
    ]


MO = {
    100 + t: {
        "id": 100 + t,
        "n": f"МО {t}",
        "ns": f"МО {t}",
        "r": "Область",
        "t": t,
        "b24": [0.1, -0.2, 0, 0.05, -0.1, 0.02],
    }
    for t in TYPES4
}  # noqa: E501


def checks(line_c: str = "solid", line_r: str = "dashed") -> dict:
    return {
        "t1_order": [2, 1, 3, 4],
        "t1": {
            k: {"med_a": [-0.3, -0.1, 0.2, 1.0], "rho_b": 0.2, "rho_b_ci": [0.15, 0.25], "line": ln}
            for k, ln in (("catering", line_c), ("retail", line_r))
        },
        "t1_rivals": [
            {"label": "деление А", "group": "self_out", "rho_b_catering": 0.3, "rho_b_retail": 0.1},
            {"label": "случайные 1", "group": "random", "rho_b_catering": 0.0, "rho_b_retail": 0.01},
        ],
        "t5": {
            "ami": [
                {"partition": "a", "label": "уровень трат", "ami": 0.3},
                {"partition": "b", "label": "б", "ami": 0.1},
            ]
        },
        "flows": {
            "types": TYPES4,
            "total": {f"{a}-{b}": (50 if a == b else 5) for a in TYPES4 for b in TYPES4},
            "reliable": {f"{a}-{b}": (0 if a == b else 2) for a in TYPES4 for b in TYPES4},
        },
        "t3": {
            "unstable_label": "неустойчиво к правилу рёбер",
            "final": {"run": "variant:graph_basket_cos", "placebo": [5, 60, 90], "observed": 50, "p95": 88.0},
            "main": {"run": "main", "placebo": [1, 2, 3], "observed": 40, "p95": 2.9},
        },
    }


def chapters(st: dict | None = None, ch: dict | None = None) -> str:
    return SC.chapters_html(st or story(), types(), ch or checks(), MO, {"clr_rel_food": 0.0}, ESC)


def section(h: str, cid: str) -> str:
    m = re.search(rf'<section class="chapter[^"]*" id="{cid}".*?</section>', h, re.S)
    assert m, cid
    return m.group(0)


def test_every_chapter_has_title_figure_source_and_table():
    h = chapters()
    for cid, title in (
        ("basket", "Что сравниваем"),
        ("types", "Заголовок T5"),
        ("order", "Заголовок T1"),
        ("dynamics", "Заголовок T3"),
    ):
        sec = section(h, cid)
        assert f'<h2 id="{cid}-title">{title}</h2>' in sec
        assert '<figcaption class="source">' in sec and "СберИндекс (CC BY-SA 4.0)" in sec
        assert '<details class="alt">' in sec  # текстовая альтернатива
        assert all('role="img" aria-label="' in s for s in re.findall(r"<svg[^>]*>", sec))
    assert "{" not in re.sub(r"<style.*?</style>", "", h)  # пустых полей нет


def test_verbatim_texts_under_how_checked():
    """Полные тексты исходов — дословно, под «Как проверяли» (§4.4), заголовок — отдельно."""
    h = chapters()
    for cid, txt in (("types", "Текст T5."), ("order", "Текст T1."), ("dynamics", "T3.")):
        sec = section(h, cid)
        assert re.search(
            rf'<details class="how"><summary>Как проверяли</summary>.*?{re.escape(txt)}', sec, re.S
        )


@pytest.mark.parametrize(
    ("line", "dash", "path"),
    [("solid", False, True), ("dashed", True, True), ("none", False, False)],
)
def test_order_line_by_turnover(line, dash, path):
    """Линия — по каждому обороту отдельно (§3.6): сплошная, пунктир или бледные точки без линии."""
    st = story(line_by_turnover={"catering": line, "retail": line})
    a, _, _ = SC.order_svgs(st, checks(line, line))
    lines = re.findall(r'<path d="M[^"]*" fill="none" stroke="#595959" stroke-width="1.5"[^>]*>', a)
    assert bool(lines) == path
    assert all(("stroke-dasharray" in x) == dash for x in lines)
    assert ('opacity="0.45"' in a) == (not path)


def test_order_columns_by_size_has_no_line():
    """При T1 not — строки по размеру и без линии, даже если оборот прошёл «в целом»."""
    st = story(t1_layout="columns_by_size", legend_order=[3, 1, 4, 2])
    a, _, _ = SC.order_svgs(st, checks("dashed", "dashed"))
    assert 'stroke-width="1.5"' not in a
    labels = re.findall(r'class="lab">([^<]+)<', a)  # порция 6a: названия типов, а не «Тип N»
    assert labels == ["Тип 3: пример", "Тип 1: пример", "Тип 4: пример", "Тип 2: пример"]


def test_order_ladder_bottom_up():
    a, _, _ = SC.order_svgs(story(), checks())
    assert re.findall(r'class="lab">([^<]+)<', a) == [
        "Тип 4: пример",
        "Тип 3: пример",
        "Тип 1: пример",
        "Тип 2: пример",
    ]  # 2 < 1 < 3 < 4 снизу


def test_order_rivals_accent_and_best_label():
    _, b, _ = SC.order_svgs(story(), checks())
    assert b.count('fill="#a3172d"') == 2  # ромб типов — по обороту
    assert "сильнейшее: деление А" in b


def _change_bands(svg: str) -> list[str]:
    return re.findall(r'<g class="flow chg".*?</g>', svg, re.S)


def test_flows_style_confirmed():
    svg = SC.alluvial_svg(story(), checks()["flows"])
    bands = _change_bands(svg)
    assert bands and any('opacity="0.85"' in g for g in bands)  # надёжные насыщенно
    assert any("url(#hflow)" in g for g in bands)  # в пределах шума — штриховка
    assert 'class="flow stay"' in svg


def test_flows_style_not_all_pale():
    st = story(flows={"reliable": "pale_hatched", "noise": "pale_hatched", "stayed": "grey"})
    bands = _change_bands(SC.alluvial_svg(st, checks()["flows"]))
    assert bands and all('opacity="0.4"' in g and "url(#hflow)" in g for g in bands)
    h = SC.chapter_dynamics(st, checks(), ESC)
    assert "Из них обе половины года согласны" not in h  # без различения — без столбца


def test_flows_keyboard_and_list():
    h = SC.chapter_dynamics(story(), checks(), ESC)
    assert 'tabindex="0" role="button"' in h and 'class="flow-btn"' in h
    assert h.count('class="flow-btn"') == 12  # 4 × 4 без диагонали


def test_placebo_rows_final_and_main():
    runs = SC.placebo_runs(story(), checks()["t3"])
    assert len(runs) == 2
    assert "другое правило связей между муниципалитетами" in runs[0]["label"] + (runs[0]["sub"] or "")
    assert runs[0]["label"].startswith(
        "Самый строгий вариант расчёта"
    )  # порция 6b: без «наименьшего вердикта»
    assert "неустойчиво" in runs[1]["label"] + (runs[1]["sub"] or "")
    svg = SC.placebo_svg(runs)
    assert "наблюдение 50" in svg and "наблюдение 40" in svg
    assert svg.count('stroke-dasharray="4 3"') == 2  # 95-й перцентиль у обоих


def test_placebo_summary_by_verdicts():
    """Итог под графиком плацебо (порция 6b): фразы — по вердиктам checks.t3, числа — с графика."""
    t3 = checks()["t3"] | {"verdict_final": "not", "unstable": True}
    t3["main"] = t3["main"] | {"passed": True, "median": 2.0}
    s = SC.placebo_summary(t3)
    assert s.startswith("Заголовок главы — по самому строгому варианту расчёта.")
    assert "«другое правило связей между муниципалитетами» число смен не отличается от плацебо: 50" in s
    assert "88,0" in s and "(40 против медианы 2)" in s and "не устоял" in s
    # итог не «not» — фразы «не отличается» нет; основной расчёт не прошёл — фразы о нём нет
    t3b = t3 | {"verdict_final": "partial", "main": t3["main"] | {"passed": False}}
    s = SC.placebo_summary(t3b)
    assert "не отличается" not in s and "основном расчёте" not in s
    # итог — сам основной расчёт: строки нет
    assert SC.placebo_summary(t3 | {"final": t3["main"]}) == ""
    h = SC.chapter_dynamics(story(), checks() | {"t3": t3}, ESC)
    assert 'class="placebo-sum"' in h


def test_rel_key_placebo_only_with_placebo():
    """Ключ надёжности: фраза о плацебо — только в главе с плацебо (5), в главе 4 — первая фраза ключа."""
    st = story() | {"reliability_key": "Сплошная — пройдена, бледное — нет. Плацебо (псевдогоды) — то и это"}
    assert SC.rel_key(st, placebo=False) == "Сплошная — пройдена, бледное — нет"
    assert SC.rel_key(st) == st["reliability_key"]
    o = SC.chapter_order(st, checks(), ESC)
    assert "Плацебо (псевдогоды)" not in o and "Сплошная — пройдена" in o


def test_t3_run_observed_not_in_cloud():
    """``pair = −1`` — наблюдение: в облако плацебо не входит; p95 и медиана — как в placebo.t3_eval."""
    pl = pd.DataFrame(
        {
            "run": ["main"] * 5 + ["other"],
            "scheme": ["main"] * 4 + ["sensitivity", "main"],
            "pair": [-1, 0, 1, 2, 0, 0],
            "n_reliable": [181, 10, 20, 30, 99, 7],
        }
    )
    r = landing.t3_run(pl, "main", 95)
    assert r["observed"] == 181 and r["placebo"] == [10, 20, 30]
    assert r["p95"] == pytest.approx(np.percentile([10, 20, 30], 95)) and r["median"] == 20
    assert landing.t3_run(pl, "nope", 95) is None


def test_types_panels_shared_row_scale():
    """Малые множества: одна широкая SVG и по панели на тип; строки в одном порядке, группы подписаны."""
    wide, panels = SC.types_svgs(story(), types(), {"clr_rel_food": 0.0})
    assert len(panels) == 4 and 'class="ch-svg sm-wide"' in wide
    for p in panels:
        assert re.findall(r'class="grp">([^<]+)<', p) == [
            "Корзина трат",
            "Экономика места",
            "Занятость, доли разделов",
        ]
    assert "продукты *" in panels[0] and "продукты *" not in panels[1]  # знак неустойчив — только у типа 1


def test_types_cards_flag_unstable_and_link_examples():
    h = section(chapters(), "types")
    assert h.count("неустойчивый тип") == 1
    assert 'href="#mo=101" data-go="101"' in h


def test_basket_steps_one_pane_visible():
    h = section(chapters(), "basket")
    assert h.count('class="step-pane"') == 4 and h.count("hidden>") == 3
    assert 'aria-pressed="true"' in h


def test_ui_has_no_forbidden_words_for_any_verdict():
    """Интерфейс глав проходит тот же линт, что шаблон, при любых вердиктах (§4.3)."""
    hc = HeadlineChecker(CFG.data)
    levels = combo_levels(CFG.data)
    keys = tuple(
        k
        for k in ("T1_ladder_external", "T2_direction", "T3_reliable_placebo", "T7_type_gain", "one_in_ten")
        if k in levels
    )
    texts = SC.ui_strings()
    for combo in all_combos(CFG.data, keys):
        assert hc.text_violations(texts, combo) == [], combo


def test_ui_lint_catches_forbidden_word():
    """Отрицательный контроль: та же проверка ловит «надёжн» при T3 not и «прогноз» всегда."""
    hc = HeadlineChecker(CFG.data)
    combo = next(all_combos(CFG.data)) | {"T3_reliable_placebo": "not"}
    assert hc.text_violations(["надёжные переходы"], combo)
    assert hc.text_violations(["прогноз типа"], combo | {"T3_reliable_placebo": "confirmed"})
