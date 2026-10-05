"""Порция 6q лендинга (``docs/landing_spec.md``, §4.3, журнал 06.10): «Простыми словами» под подписью
графика «Что устояло» (исход T1), строка о годах над таблицей всех муниципалитетов, ссылка на таблицу сразу
после холста схемы сети. Замороженный блок ``site`` и его хеш не меняются. Синтетика с известным ответом,
без data/ и outputs/."""

from __future__ import annotations

import copy
import html
import re

from munnet import landing, site_chapters, site_chapters_tail, site_netmap
from munnet.config import load_config
from munnet.site_headlines import freeze_hash

CFG = load_config("configs/default.yaml")
TX = CFG["site"]["build"]["texts"]
# как в outputs/interpret/facts.json (t1.per): типы против лучшего деления без типов внутри страт
T1 = {
    "per": {
        "catering": {"best_rival": "sized:log_wage_rel:+", "best_rival_rho": 0.34524608, "rho_b": 0.23719344},
        "retail": {"best_rival": "sized:log_level_rel:+", "best_rival_rho": 0.27705977, "rho_b": 0.17267408},
    }
}
WANT = (
    "в целом порядок типов виден в обороте Росстата. Но среди муниципалитетов похожего размера и с похожей "
    "долей горожан простые деления упорядочивают оборот сильнее: по зарплате — общепит, "
    "по уровню трат — розницу"
)


def test_t1_plain_from_facts():
    assert landing.t1_plain(TX, "partial_overall", T1) == WANT


def test_t1_plain_only_partial_overall():
    for v in ("confirmed", "not", "partial", ""):
        assert landing.t1_plain(TX, v, T1) is None


def test_t1_plain_needs_rival_stronger_and_words():
    t1 = copy.deepcopy(T1)
    t1["per"]["retail"]["best_rival_rho"] = 0.1  # деление не сильнее типов — «слабее» было бы неверно
    assert landing.t1_plain(TX, "partial_overall", t1) is None
    t1 = copy.deepcopy(T1)
    t1["per"]["retail"]["best_rival"] = "sized:unknown_feature:+"
    assert landing.t1_plain(TX, "partial_overall", t1) is None
    t1 = copy.deepcopy(T1)
    del t1["per"]["catering"]["rho_b"]
    assert landing.t1_plain(TX, "partial_overall", t1) is None
    assert landing.t1_plain(TX, "partial_overall", {}) is None
    assert landing.t1_plain({}, "partial_overall", T1) is None


def test_t1_plain_matches_report_wording():
    # соперники — как в report.md §8 «Почему так» и в коротком h2 главы 4; чисел нет (check-ux 06.10)
    assert "по зарплате — общепит" in WANT and "по уровню трат — розницу" in WANT
    assert not re.search(r"\d|ρ", WANT) and "совпада" not in WANT
    assert "по зарплате" in landing.order_short(
        TX, {"per": {k: v | {"mono_a": True} for k, v in T1["per"].items()}}
    )


def test_table_years_note_and_describedby():
    note = TX["dynamics"]["table_years"]
    out = site_chapters_tail.chapter_all_mo(None, html.escape, note)
    assert 'id="mo-years"' in out and "смена типа за год не подтверждена" in out
    assert "Колонки 2023 и 2024 —" not in out  # без повтора абзаца над таблицей (check-ux 06.10)
    assert out.count('aria-describedby="mo-years"') == 2  # только колонки 2023 и 2024
    plain = site_chapters_tail.chapter_all_mo(None, html.escape)
    assert "mo-years" not in plain
    # строка «смену типа между годами показывают колонки» читалась как подтверждённая смена — её нет
    assert "показывают колонки" not in plain


def test_table_years_only_with_dynamics_outcome():
    t3 = {
        "verdict_final": "not",
        "verdict_main": "confirmed",
        "final": {"run": TX["dynamics"]["final_run"], "observed": 33, "p95": 47},
        "main": {"passed": True, "observed": 148, "p95": 236 - 100},
    }
    dt = site_chapters.dynamics_texts(TX["dynamics"], t3)
    assert dt and dt["table_years"] == TX["dynamics"]["table_years"]
    story = {"chapters": {"dynamics": {"title": "x"}}, "screen0": {}}
    site_chapters.apply_dynamics(story, dt, "x")
    assert story["table_years"] == TX["dynamics"]["table_years"]
    t3["verdict_final"] = "confirmed"
    assert site_chapters.dynamics_texts(TX["dynamics"], t3) is None


def test_netmap_link_after_caption_and_source():
    keys = "title lead hint hint_touch aria legend_label panel_empty nojs caption how_label how".split()
    t = dict.fromkeys(keys, "x")
    t["to_table"] = TX["netmap"]["to_table"]
    out = site_netmap.html_block(t, {}, [], {}, {}, html.escape, "src")
    i_link = out.index('href="#all-mo"')
    assert out.index("net-canvas") < out.index("net-cap") < out.index('class="source"') < i_link
    assert "Перейти к таблице всех муниципалитетов" in out


def test_all_mo_section_takes_focus():
    out = site_chapters_tail.chapter_all_mo(None, html.escape)
    assert out.startswith('<section class="chapter wide" id="all-mo" tabindex="-1"')
    assert 'tabindex="-1"' not in site_chapters._section("x", "t", "", "", html.escape)


def test_freeze_hash_unchanged():
    assert freeze_hash(CFG["site"]) == CFG["site"]["freeze"]["sha256"]
    for k in ("t1_plain", "t1_plain_pair"):
        assert k not in CFG["site"]["freeze"]["covers"]
