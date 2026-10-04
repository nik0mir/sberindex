"""Порция 6e лендинга (``docs/landing_spec.md``, §4.12): «зачем» на первом экране, текст для плоской карты,
ключ «как читать» главы 7, понятные пометки устойчивости, число главы 6 в обратном примере, строки карточки
о сети корзин и подписи не меньше 12 px. Синтетика с известным ответом, без data/ и outputs/."""

from __future__ import annotations

import re
from pathlib import Path

from test_site_useful import _uf

from munnet import landing, site_findings, site_useful
from munnet.config import load_config

CFG = load_config("configs/default.yaml")
TX = CFG["site"]["build"]["texts"]
TPL = Path("src/munnet/templates")
NAMES = {"1": "Города, меньше транспорта", "2": "Сельские, меньше общепита", "3": "Города, меньше продуктов",
         "4": "Крупные города, меньше продуктов"}  # fmt: skip


def test_limits_key_names_only_types_in_text():
    text = "У МО, перешедших из типа 2 в тип 1 и из типа 1 в тип 3, … различается по типам (ε² = 0,082)."
    key = landing.limits_key(TX["limits_key"], text, NAMES)
    assert key.startswith("Как читать: тип 1 — «Города, меньше транспорта»; тип 2 — «Сельские")
    assert "тип 3 — «Города, меньше продуктов»" in key and "тип 4" not in key
    assert "МО — муниципалитет" in key and "ε²" in key
    # нет «МО» и «ε²» в тексте — нет и их расшифровки; нет номеров типов — строки нет
    assert "МО —" not in landing.limits_key(TX["limits_key"], "Из типа 4 в тип 2.", NAMES)
    assert landing.limits_key(TX["limits_key"], "Ничего о типах.", NAMES) is None


def test_robust_texts_numbers_from_facts_checked_against_variants():
    r1 = {
        "unstable_label": "неустойчиво к правилу рёбер",
        "circularity": "ARI вариантов (0,22 и 0,32) известен, поэтому порог на ARI не ставится.",
    }
    variants = [{"ari": 0.2236}, {"ari": 0.9957}, {"ari": 0.3163}]
    out = landing.robust_texts(TX["robust"], r1, variants)
    assert out["unstable_label"].startswith("неустойчиво: меняется при другом правиле связей")
    assert "(ARI 0,22 и 0,32; 1 — полное совпадение" in out["circularity"]
    # число в тексте этапа 5 не совпало с ARI вариантов — понятного текста нет (остаётся текст этапа 5)
    assert "circularity" not in landing.robust_texts(TX["robust"], r1, [{"ari": 0.5}, {"ari": 0.3163}])
    # нет пометки у этапа 5 — нет и на сайте
    assert landing.robust_texts(TX["robust"], {}, variants) == {}


def test_reverse_example_takes_chapter6_number_by_code():
    ux = TX["useful"]
    t7ex = {"territory_id": 44, "errors": {"A": 0.0439, "B": 0.0583, "C": 0.0488, "D": 0.0496}}
    ex = site_useful.example_texts(ux, _uf(), {}, t7_example=t7ex)
    rev = ex["reverse"]
    assert "0,058 против 0,033 без поправки на регион (в главе 6, с поправкой на регион, — 0,050)" in rev
    assert "ошибаются больше похожих по тратам" in rev
    # пример главы 6 — другой муниципалитет: скобки с его числом нет, пустого поля тоже
    rev2 = site_useful.example_texts(ux, _uf(), {}, t7_example={**t7ex, "territory_id": 7})["reverse"]
    assert "главы 6 —" not in rev2 and "{" not in rev2 and "без поправки на регион." in rev2


def test_card_texts_net_and_similar_caption():
    checks = {"t7": None}
    out = site_findings.card_texts(TX["card"], checks, k_net=5, sim_fields={"n_shown": "5", "n_set": "10"})
    assert out["net_note"].startswith("До 5 муниципалитетов") and "не «похожие по тратам" in out["net_note"]
    assert (
        out["sim_caption"].startswith("Показаны 5 из 10, на которых проверяли точность сверки")
        and "первый — самый похожий" in out["sim_caption"]
        and "км" not in out["sim_caption"]
    )
    assert out["sim_from"].startswith("Муниципалитеты других регионов с самой похожей корзиной трат")
    assert out["net_title"] == "Соседи по сети корзин" and out["arcs_net"] and out["arcs_default"]
    # без числа соседей подписи сети нет (пустого поля на странице не бывает)
    assert "net_note" not in site_findings.card_texts(TX["card"], checks)


def test_hero_why_first_and_flat_lede():
    story = {
        "screen0": {
            "title": "T",
            "coverage": "",
            "hero": {**TX["hero"], "lede_flat": TX["hero"]["lede_flat"]},
        },
        "view": {"type_colors": {"1": "#111"}, "legend_order": [1], "shapes": {"1": "●"}},
        "names": {"final": {"1": "Тип один"}},
    }
    parts = landing.hero_parts(story, "", None)
    head = parts["hero_head"]
    assert head.index('id="hero-why"') < head.index('id="hero-lede"') < head.index('id="hero-lede-flat"')
    assert head.index('id="hero-lede-flat"') < head.index('id="hero-trust"')
    assert 'class="h0-lede lede-3d"' in head and 'class="h0-lede lede-flat"' in head
    assert 'href="#all-mo"' in parts["hero_legend"]
    css = (TPL / "landing.css").read_text(encoding="utf-8")
    # плоская по умолчанию; объёмная — при html.may-3d или html.has-3d; по кнопке «Плоская» — снова плоская
    assert re.search(r"\.lede-3d \{ display: none; \}", css)
    assert "html.has-3d.h0-flat .lede-flat { display: block; }" in css
    html = (TPL / "landing.html").read_text(encoding="utf-8")
    assert 'location.protocol !== "file:"' in html and "may-3d" in html
    js3d = (TPL / "landing3d.js").read_text(encoding="utf-8")
    assert 'classList.remove("has-3d", "may-3d")' in js3d


def test_small_labels_at_least_12px():
    css = (TPL / "landing.css").read_text(encoding="utf-8")
    tail = css[css.index("порция 6e") :]
    for sel in (".h0-bars em", ".mtable td small", ".axis-words", ".card .src"):
        assert re.search(re.escape(sel) + r"[^{]*\{ font-size: 12px; \}", tail), sel
