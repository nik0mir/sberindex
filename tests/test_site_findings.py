"""Порция 5b лендинга (``docs/landing_spec.md``, §4.8): блок «Что устояло», строки карточки о сверке
с соседями по региону, паспорта типов, шрифты из репозитория. Числа — только из facts/checks/типов
(здесь — синтетика с известным ответом), тексты — из ``site.build.texts`` и проходят линт текста сайта."""

from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path

import pytest

from munnet import landing, site_findings
from munnet.config import load_config
from munnet.contracts import QCError

CFG = load_config("configs/default.yaml")
TX = CFG["site"]["build"]["texts"]
ESC = landing._t


def _checks(hi=-0.0032, rival=0.277):
    return {
        "t1": {"retail": {"med_a": [-0.2846, -0.0249, 0.2841, 0.7564], "line": "dashed"}},
        "t1_order": [2, 1, 3, 4],
        "t7": {
            "product": "D",
            "k": 10,
            "n_common": 1542,
            "median_error_abs": {"A": 0.0411, "B": 0.0362, "C": 0.0382, "D": 0.0416},
            "diffs": {"B-D_abs": [-0.0054, [-0.0086, hi]]},
        },
        "r1": {
            "variants": [
                {"variant": "variant:graph_basket_cos", "n": 1776, "same": 833},
                {"variant": "variant:no_level", "n": 1776, "same": 1773},
                {"variant": "variant:nodes_separate", "n": 2016, "same": 1091},
            ]
        },
        "_rival": rival,
    }


def _facts(rival=0.277):
    return {
        "t1": {
            "per": {
                "retail": {
                    "rho_a": 0.5509,
                    "rho_b": 0.1727,
                    "best_rival_rho": rival,
                    "best_rival_label": "уровень трат, по возрастанию",
                    "n_a": 1705,
                }
            }
        }
    }


def _story(t1="partial_overall", t3="not"):
    return {
        "verdicts": {
            "T1_ladder_external": t1,
            "T3_reliable_placebo": t3,
            "T7_utility": "not",
            "one_in_ten": False,
        },
        "screen0": {"lead": "Число смен типа за год не отличается от плацебо"},
        "names": {"final": {"1": "Тип один", "2": "Тип два", "3": "Тип три", "4": "Тип четыре"}},
        "view": {
            "t1_layout": "ladder_place",
            "legend_order": [1, 2, 3, 4],
            "line_by_turnover": {"retail": "dashed"},
            "type_colors": {"1": "#589BA6", "2": "#846D1E", "3": "#574194", "4": "#0F2E56"},
            "shapes": {"1": "●", "2": "▲", "3": "■", "4": "◆"},
        },
    }


def test_numbers_come_from_facts_and_checks():
    """Числа блока — форматированные значения facts.t1 и checks.t7/r1, а не константы шаблона."""
    ft = site_findings.findings_texts(TX["findings"], _story(), _facts(), _checks())
    assert ft is not None
    s = ft["stood"]["text"]
    assert "0,55" in s and "1705" in s and "0,28 против 0,17" in s and "«уровень трат»" in s
    u = ft["use"]["text"]
    assert "0,036 против 0,042" in u and "разность медиан −0,005" in u and "от −0,009 до −0,003" in u
    assert "1542 муниципалитета" in u  # порция 6b: согласование числительного
    assert "10 ориентиров" in ft["use"]["chart"]
    b = ft["border"]["text"]
    assert "46,9% муниципалитетов с типом" in b and "другое правило связей между муниципалитетами" in b
    # порция 6e: после заголовка T3 о плацебо — одна фраза, что такое плацебо (findings.border.placebo)
    assert "от плацебо. Плацебо — те же расчёты" in b and b.rstrip(".").endswith("даёт случайность")
    # другие числа — другой текст (подставляет код)
    c = _checks()
    c["t7"]["median_error_abs"]["B"] = 0.0301
    assert (
        "0,030 против" in site_findings.findings_texts(TX["findings"], _story(), _facts(), c)["use"]["text"]
    )


def test_block_hidden_when_conditions_fail():
    """Тексты написаны под настоящий исход: другой вердикт T1, интервал разности через ноль или соперник
    не сильнее типов — блока нет (а не ослабленный текст)."""
    assert site_findings.findings_texts(TX["findings"], _story(t1="confirmed"), _facts(), _checks()) is None
    assert site_findings.findings_texts(TX["findings"], _story(), _facts(), _checks(hi=0.001)) is None
    assert site_findings.findings_texts(TX["findings"], _story(), _facts(rival=0.1), _checks()) is None
    assert site_findings.t7_numbers(_checks(hi=0.0)) is None


def test_findings_html_and_lint():
    """Три колонки с графиками (SVG с aria-label) и подписью источника; строки проходят линт текста сайта
    при текущих вердиктах, а запрещённое слово ловится."""
    story = _story()
    ft = site_findings.findings_texts(TX["findings"], story, _facts(), _checks())
    h = site_findings.findings_html(ft, story, [2, 1, 3, 4], ESC, "Источник: СберИндекс (CC BY-SA 4.0).")
    assert h.count('class="fd fd-') == 3 and h.count("<svg") == 6 and 'id="findings"' in h
    assert h.count('<svg class="vw fd-svg') == 3 and h.count('<svg class="vp fd-svg') == 3  # и для телефона
    assert "×2,13" in h and "×0,75" in h  # exp(med_a) у типов 4 и 2
    assert h.index("Тип четыре") < h.index("Тип два")  # ступени снизу вверх: тип 4 — верхняя строка
    assert "stroke-dasharray" in h  # линия overall_only — пунктир
    assert "СберИндекс (CC BY-SA 4.0)" in h
    st = CFG.data | {}
    story_lint = {"verdicts": story["verdicts"] | {"T2_direction": "partial", "caveat": True}}
    assert landing.lint_texts(landing.Config(st, Path("x")), story_lint, site_findings.strings(ft)) == []
    bad = site_findings.strings(ft) + ["Соседи — надёжный прогноз"]
    assert landing.lint_texts(landing.Config(st, Path("x")), story_lint, bad)


def test_card_texts_numbers_or_plain():
    """Карточка: при выполненном условии — медианные ошибки из checks.t7, иначе нейтральная строка
    без чисел."""
    c = site_findings.card_texts(TX["card"], _checks())
    assert (
        "0,036 против 0,042" in c["simb_note"]
        and c["sim_note"]
        and c["simb_title"].startswith("С кем сверять")
    )
    c = site_findings.card_texts(TX["card"], _checks(hi=0.01))
    assert not re.search(r"\d", c["simb_note"]) and "sim_note" not in c


def test_rel_text_words():
    """Медиана в логарифмах словами: как подписи островов (exp)."""
    assert site_findings.rel_text(-0.43).startswith("примерно в 1,5 раза меньше")
    assert site_findings.rel_text(math.log(1.17)) == "примерно на 17% больше"
    assert site_findings.rel_text(0.001) == "как в регионе"


def test_passports_from_profile_and_settlement():
    """Паспорт: число МО, «кто обычно» из settlement_shares, пять строк из profile.csv словами, пример —
    кнопка
    с data-go и data-map; порядок — как в types (legend_order)."""
    types = [
        {
            "t": 2,
            "name": "Сельские",
            "color": "#846D1E",
            "size": 473,
            "pop_share": 0.06,
            "who": {"cities": 0.2321, "large_cities": 0.0},
            "unstable_parts": [],
            "profile": [
                {"feature": "clr_rel_cafe", "median": -0.43},
                {"feature": "clr_rel_food", "median": 0.156},
                {"feature": "clr_rel_marketplace", "median": 0.16},
                {"feature": "clr_rel_transport", "median": 0.08},
                {"feature": "log_level_rel", "median": -0.11},
            ],
            "examples": {"typical": [7]},
        },
        {"t": 1, "name": "Города", "color": "#589BA6", "size": 806, "pop_share": 0.21, "profile": [],
         "examples": {}},
    ]  # fmt: skip
    mo = {7: {"ns": "большеберезниковский", "n": "Большеберезниковский район", "r": "Республика Мордовия"}}
    h = site_findings.passports_html(types, _story(), mo, TX["passports"], ESC)
    assert h.count('<article class="pp"') == 2 and h.index("473") < h.index("806")
    assert "23%" in h and "0%" in h and "примерно в 1,5 раза меньше" in h and "примерно на 17% больше" in h
    assert 'data-go="7" data-map="1"' in h and "Большеберезниковский" in h
    assert (
        landing.lint_texts(
            landing.Config(CFG.data, Path("x")),
            {"verdicts": _story()["verdicts"] | {"T2_direction": "partial", "caveat": True}},
            site_findings.passport_strings(types, TX["passports"]),
        )
        == []
    )


def test_fonts_vendored_with_sha256():
    """Шрифты — из templates/vendor/fonts со сверкой sha256 по README; @font-face переписан на vendor/fonts/…;
    внешних адресов (Google Fonts) в шаблоне нет."""
    files = landing.font_files()
    readme = (Path(landing.__file__).parent / "templates" / "vendor" / "README.md").read_text(
        encoding="utf-8"
    )
    woff = [k for k in files if k.endswith(".woff2")]
    assert len(woff) >= 4 and "vendor/fonts/fonts.css" in files
    for k, b in files.items():
        assert hashlib.sha256(b).hexdigest() in readme, k
    css = landing.fonts_css(files)
    assert "url(vendor/fonts/golos-text-cyrillic-wght-normal.woff2)" in css and "url(fonts/" not in css
    tpl = (Path(landing.__file__).parent / "templates" / "landing.html").read_text(encoding="utf-8")
    assert "fonts.googleapis" not in tpl and "fonts.gstatic" not in tpl


def test_fonts_sha_mismatch_is_code3(tmp_path, monkeypatch):
    """Подменённый файл шрифта — код 3 (QCError), как у three.js."""
    import shutil

    src = Path(landing.__file__).parent / "templates" / "vendor"
    dst = tmp_path / "vendor"
    shutil.copytree(src, dst)
    p = next((dst / "fonts").glob("*.woff2"))
    p.write_bytes(p.read_bytes() + b"x")
    monkeypatch.setattr(landing, "_vendor_dir", lambda: dst)
    with pytest.raises(QCError, match="sha256"):
        landing.font_files()
