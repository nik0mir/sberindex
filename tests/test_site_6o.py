"""Порция 6o лендинга (``docs/landing_spec.md``, §4.3, журнал 05.10): правки по финальному совету 05.10 —
пересказ исхода T6 названиями типов, короткие h2 глав 4 и 6, лид «Что проверяли», кнопки первого экрана вне
колонки текста и «Пауза / Вращать», строка о цвете над плоской картой, ключ стилей у своего графика.
Синтетика с известным ответом, без data/ и outputs/."""

from __future__ import annotations

from pathlib import Path

from munnet import landing, site_chapters, site_chapters_tail
from munnet.config import load_config

CFG = load_config("configs/default.yaml")
TX = CFG["site"]["build"]["texts"]
TPL = Path("src/munnet/templates")
NAMES = {"1": "Города, меньше транспорта", "2": "Сельские, меньше общепита", "3": "Города, меньше продуктов",
         "4": "Крупные города, меньше продуктов"}  # fmt: skip
FACTS = {"names_final": NAMES, "t6": {"coverage_eps2": 0.08222}}


def test_limits_plain_names_and_scale():
    """Пересказ T6: названия типов вместо номеров, «муниципалитет», «без малого бизнеса», ε² со шкалой и
    неразрывными пробелами; только при исходе not с оговоркой."""
    t = landing.limits_plain(TX, "not", True, [[2, 1], [1, 3]], FACTS)
    assert t.startswith(
        "У муниципалитетов, перешедших из типа «Сельские, меньше общепита» в тип «Города, меньше"
    )
    assert "и из типа «Города, меньше транспорта» в тип «Города, меньше продуктов»" in t
    assert "(без малого бизнеса)" in t and "МСП" not in t and " МО" not in t
    assert "ε² = 0,082 по шкале от 0 до 1" in t
    assert "надёжн" not in t  # корень запрещён сайту при T3 not
    assert landing.limits_plain(TX, "confirmed", True, [[2, 1]], FACTS) is None
    assert landing.limits_plain(TX, "not", False, [[2, 1]], FACTS) is None
    assert landing.limits_plain(TX, "not", True, [[2, 9]], FACTS) is None  # нет названия — дословный текст


def test_limits_chapter_plain_first_verbatim_under_details():
    story = {
        "chapters": {
            "limits": {
                "title": "Ограничения",
                "text": "У МО, надёжно перешедших из типа 2 в тип 1, … (ε² = 0,082).",
                "plain": "Пересказ",
                "verbatim_label": TX["verbatim_label"],
                "context": "Смена типа за год не подтверждена",
                "key": "Как читать: тип 1 — «Города»",
            }
        },
        "view": {},
    }
    h = site_chapters_tail.chapter_limits(story, {}, None, landing._t)
    assert h.index("Пересказ") < h.index("<details") < h.index("надёжно перешедших")
    assert "Как читать" not in h and TX["verbatim_label"] in h


def test_short_titles_and_checked_lead_in_config():
    """Короткие h2 глав 4 и 6 — до ~12 слов, без чисел; лид «Что проверяли» — о порядке работы, без вывода."""
    assert len(TX["comparable_short"].split()) <= 13 and not any(
        ch.isdigit() for ch in TX["comparable_short"]
    )
    lead = TX["checked_lead"]
    assert "{n_nodes}" in lead and "до проверочных расчётов" in lead and "seen_before" in lead
    assert "не подтверждена" not in lead
    assert TX["dynamics"]["sub"].startswith("В основном расчёте смен типа больше, чем бывает случайно")
    assert not TX["hero"]["kicker_note"]  # второй круг: строка о 4 регионах — только в «Ограничениях»
    assert "(configs/default.yaml, раздел interpret.seen_before)" in lead


def test_order_chapter_long_title_first_paragraph():
    story = {"chapters": {"order": {"title": "Коротко", "title_long": "Полное утверждение"}}, "view": {}}
    h = site_chapters.chapter_order(story, {}, landing._t)
    assert '<p class="sub">Полное утверждение.</p>' in h


def test_hero_actions_outside_text_column_and_motion_button():
    """Кнопки вида — вне колонки текста (на 1440 × 900 их не закрывает строка источника); «Пауза / Вращать»
    рядом с ними; строка о цвете — над плоской картой, одна."""
    html = (TPL / "landing.html").read_text(encoding="utf-8")
    text_col = html.split('<div class="hero-text">')[1].split('<div class="hero-stage"')[0]
    head = text_col.split('<div class="h0-actions"')[0]
    assert 'id="search-input"' in head and 'id="h0-types"' not in head
    assert 'id="h0-motion"' in html and "$pause" in html and "$motion_label" in html
    assert html.count('id="flat-note"') == 1
    assert html.index('id="flat-note"') < html.index('id="map-frame"')
    for k in ("pause", "play", "motion_label"):
        assert k in landing.HERO_KEYS and TX["hero"][k]
    js = (TPL / "landing3d.js").read_text(encoding="utf-8")
    assert "let motionOn = !reduce" in js and "swayT += dt" in js
    css = (TPL / "landing.css").read_text(encoding="utf-8")
    assert ".has-3d .h0-actions { position: absolute;" in css


def test_dynamics_style_keys_next_to_own_chart():
    """Ключ стилей потоков — в колонке Санки и с её именем; ключ плацебо начинается с имени графика."""
    ui = site_chapters.UI
    assert ui["placebo_final"].startswith("Самый слабый") and "строг" not in ui["placebo_sum_head"]
    assert ui["placebo_key_head"] == "График плацебо" and ui["flows_key_head"]


def test_no_lowercase_untyped_label_in_templates():
    js = (TPL / "landing.js").read_text(encoding="utf-8")
    assert '"нет типа"' not in js and "<span>Нет типа</span>" not in js


def test_order_short_rivals_from_code():
    """h2 главы 4: лучшие деления без типов — из facts.t1.per, «растёт» — только при монотонности."""
    per = {
        "catering": {"best_rival": "sized:log_wage_rel:+", "mono_a": True},
        "retail": {"best_rival": "sized:log_level_rel:+", "mono_a": True},
    }
    t = landing.order_short(TX, {"per": per})
    assert t == (
        "Оборот Росстата растёт от типа к типу, но при том же размере и доле горожан сильнее упорядочены "
        "группы по зарплате (общепит) и по уровню трат (розница)"
    )
    same = {k: v | {"best_rival": "sized:log_wage_rel:+"} for k, v in per.items()}
    assert landing.order_short(TX, {"per": same}).endswith("группы по зарплате")
    assert landing.order_short(TX, {"per": {"retail": per["retail"] | {"mono_a": False}}}) is None
    assert landing.order_short(TX, {"per": {"retail": {"best_rival": "sized:xyz:+", "mono_a": True}}}) is None


def test_netmap_legend_in_strip_order():
    from munnet import site_netmap

    data = {"ids": [1, 2, 3, 4], "x": site_netmap.encode(__import__("numpy").array([900, 100, 500, 120]))}
    assert site_netmap.strip_order(data, {1: 4, 2: 2, 3: 1, 4: 2}) == [2, 1, 4]


def test_hero_title_short_prepositions_nbsp():
    assert landing._nb_short("по сравнению со своим регионом") == "по сравнению со своим регионом"
