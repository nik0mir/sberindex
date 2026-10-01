"""Порция 6b лендинга (``docs/landing_spec.md``, §4.10): блок пользы и флаг устойчивости типа из этапа
usefulness. Числа и слова доли — только из facts.json usefulness (здесь — синтетика с известным ответом),
тексты — ``site.build.texts.useful`` и проходят линт текста сайта; вход необязательный (нет файлов —
блока нет),
устаревший вход — код 1; знак разности ошибок у отдельного МО на странице не появляется."""

from __future__ import annotations

import copy
import json
import logging
import os
import re
from pathlib import Path

import pandas as pd
import pytest
from fixtures.site.synth_facts import all_combos, make_facts, site_config
from fixtures.site.synth_inputs import (
    CITY,
    CONTROLS,
    bound_interpret,
    facts_over,
    make_site_inputs,
    page_types,
)

from munnet import landing, site_findings, site_useful
from munnet.config import Config, load_config
from munnet.contracts import MissingInputError, QCError

CFG = load_config("configs/default.yaml")
TX = CFG["site"]["build"]["texts"]
WORDS = CFG["usefulness"]["rule_share"]["words"]
ESC = landing._t
LINT_V = {
    "T1_ladder_external": "partial_overall",
    "T2_direction": "partial",
    "T3_reliable_placebo": "not",
    "T7_utility": "not",
    "T7_type_gain": "neutral",
    "one_in_ten": False,
    "caveat": True,
}


def _uf(d_words="чаще, чем нет", r_words="чаще, чем нет", r_err=0.0353, s5_b=0.0583):
    return {
        "status": "done",
        "label": "описание, не проверка",
        "rule": {
            "n": 1542,
            "reference": "при равной точности было бы 50%",
            "vs_D": {"share": 0.5719, "share_ci": [0.5471, 0.5966], "words": d_words},
            "vs_R": {"share": 0.5253, "share_ci": [0.5007, 0.5497], "words": r_words},
            "median_error_abs": {"B": 0.0362, "C": 0.0382, "D": 0.0416, "R": r_err},
            "gain_median": -0.0053,
        },
        "example": {
            "territory_id": 269,
            "name": "Йошкар-Ола",
            "region": "Республика Марий Эл",
            "own_change": 0.2172,
            "median_B_change": 0.1986,
            "median_D_change": 0.1922,
            "err_B": 0.0154,
            "err_D": 0.0207,
            "d": -0.0053,
            "members_B": [{"territory_id": i} for i in range(10)],
            "members_D": [{"territory_id": i} for i in range(10)],
            "stage5_example": {
                "territory_id": 44,
                "name": "Благовещенский",
                "region": "Республика Башкортостан",
                "err_B": s5_b,
                "err_D": 0.033,
            },
        },
        "type_flag": {
            "n": 1776,
            "stable": 578,
            "per_variant": [
                {"variant": "variant:graph_basket_cos", "n": 1776, "same_share": 0.469},
                {"variant": "variant:no_level", "n": 1776, "same_share": 0.998},
                {"variant": "variant:nodes_separate", "n": 1774, "same_share": 0.573},
            ],
        },
    }


MO = {
    269: {"ns": "Йошкар-Ола", "r": "Республика Марий Эл"},
    44: {"ns": "Благовещенский", "r": "Республика Башкортостан"},
}


def _useful(uf):
    border = TX["findings"]["border"]
    return {
        "use": site_useful.use_texts(TX["useful"], uf, WORDS),
        "example": site_useful.example_texts(TX["useful"], uf, MO),
        "flag": site_useful.flag_summary(border["flag"], border["flag_most"], uf),
        "flag_label": uf["label"],
    }


def _ft(uf):
    from test_site_findings import _checks, _facts, _story

    return site_findings.findings_texts(TX["findings"], _story(), _facts(), _checks(), _useful(uf))


def test_share_and_r_line_from_facts():
    """Доля, интервал, ориентир 50% и слова — из facts.json; оговорка о наборе R — по медианам ошибок."""
    u = site_useful.use_texts(TX["useful"], _uf(), WORDS)
    assert u["share"] == (
        "Соседи по своему региону точнее похожих по тратам чаще, чем нет: в 57,2% случаев из 1542 их ошибка "
        "меньше (95% интервал 54,7–59,7%; при равной точности было бы 50%)"
    )
    assert "не больше соседей (0,035 против 0,036)" in u["r_line"] and "свой регион" in u["r_line"]
    u = site_useful.use_texts(TX["useful"], _uf(r_err=0.04), WORDS)
    assert "ошибаются больше соседей (0,040 против 0,036)" in u["r_line"] and "не больше" not in u["r_line"]
    with pytest.raises(QCError):
        site_useful.use_texts(TX["useful"], _uf(d_words="почти всегда"), WORDS)


def test_edits_by_words_executed_literally():
    """Правки usefulness.rule_share.edits: «чаще, чем нет» — «ориентир лучше» остаётся; «примерно в половине»
    и «реже» — текст меняется; доля B против R не «чаще, чем нет» — совет «со своим регионом»."""
    ft = _ft(_uf())
    assert "ориентир лучше" in ft["use"]["text"] and ft["use"]["title"] == TX["findings"]["use"]["title"]
    assert ft["use"]["share"].startswith("Соседи по своему региону точнее похожих по тратам чаще, чем нет")
    ft = _ft(_uf(d_words=WORDS["about_half"]))
    assert "ориентир лучше" not in ft["use"]["text"] and "примерно в половине случаев" in ft["use"]["text"]
    ft = _ft(_uf(d_words=WORDS["less_often"]))
    assert "в большинстве случаев похожие по тратам" in ft["use"]["text"]
    assert ft["use"]["title"] == TX["findings"]["use"]["title_less_often"]
    ft = _ft(_uf(r_words=WORDS["about_half"]))
    assert ft["use"]["title"] == "Сверяйте со своим регионом"
    # без usefulness — прежний текст колонки, без доли
    from test_site_findings import _checks, _facts, _story

    ft0 = site_findings.findings_texts(TX["findings"], _story(), _facts(), _checks())
    assert "share" not in ft0["use"] and ft0["_example"] is None and "flag" not in ft0["border"]


def test_example_and_reverse_case():
    """Пример по правилу: три значения в процентах со знаком, ошибки; обратный пример этапа 5 — только если
    у него соседи по региону и правда ошибаются больше."""
    ex = site_useful.example_texts(TX["useful"], _uf(), MO)
    assert ex["id"] == 269 and ex["title"] == "Йошкар-Ола, Республика Марий Эл"
    assert "+21,7%" in ex["text"] and "+19,9%" in ex["text"] and "+19,2%" in ex["text"]
    assert "0,015" in ex["text"] and "0,021" in ex["text"]
    assert (
        "Благовещенский, Республика Башкортостан" in ex["reverse"] and "0,058 против 0,033" in ex["reverse"]
    )
    assert "reverse" not in site_useful.example_texts(TX["useful"], _uf(s5_b=0.02), MO)
    svg = site_useful.example_svg(ex)
    assert svg.count("<circle") == 3 and "+21,7%" in svg and 'role="img"' in svg
    phone = site_useful.example_svg(ex, 320)
    assert phone.count('class="axis"') > svg.count('class="axis"')  # подпись оси — в две строки


def test_flag_summary_most_from_lowest_variant():
    """«Чаще всего — от …» — вариант с наименьшей долей «тот же тип»; числа — type_flag."""
    border = TX["findings"]["border"]
    s = site_useful.flag_summary(border["flag"], border["flag_most"], _uf())
    assert s.startswith("Тип устойчив у 578 из 1776 муниципалитетов с типом (32,5%)")
    assert s.endswith("от правила связей между муниципалитетами")
    uf = _uf()
    uf["type_flag"]["per_variant"][2]["same_share"] = 0.3
    assert "районы Москвы и Петербурга" in site_useful.flag_summary(border["flag"], border["flag_most"], uf)


def test_texts_pass_site_lint_and_no_reliable_root():
    """Все строки 6b проходят линт текста сайта при текущих вердиктах; корня «надёжн» нет."""
    u = _useful(_uf())
    texts = site_useful.strings(u["use"], u["example"], u["flag"]) + site_findings.strings(_ft(_uf()))
    assert landing.lint_texts(landing.Config(CFG.data, Path("x")), {"verdicts": LINT_V}, texts) == []
    assert not any("надёжн" in t.lower() for t in texts)
    bad = texts + ["Соседи — надёжный прогноз"]
    assert landing.lint_texts(landing.Config(CFG.data, Path("x")), {"verdicts": LINT_V}, bad)


def test_findings_html_has_share_flag_and_example():
    """Колонка «Что с этим делать» — доля и оговорка; «Где граница» — строка флага с меткой; полоса примера
    с кнопкой карточки; на телефоне — свой вариант графика."""
    from test_site_findings import _story

    h = site_findings.findings_html(_ft(_uf()), _story(), [2, 1, 3, 4], ESC, "Источник: СберИндекс.")
    assert 'class="fd-share"' in h and 'class="fd-flag"' in h and "описание, не проверка" in h
    assert 'id="use-example"' in h and 'data-go="269" data-map="1"' in h
    assert h.count('<svg class="vw fd-svg ex-svg') == 1 and h.count('<svg class="vp fd-svg ex-svg') == 1


def _flags(types: dict[int, int]) -> pd.DataFrame:
    ids = sorted(types)
    return pd.DataFrame(
        {
            "territory_id": ids,
            "type": [types[i] for i in ids],
            "seed_same": [4] * len(ids),
            "seed_runs": [4] * len(ids),
            "variant_same": [3 if i % 2 else 1 for i in ids],
            "variant_runs": [3] * len(ids),
            "flag": ["stable" if i % 2 else "depends" for i in ids],
            "flag_text": ["тип устойчив" if i % 2 else "тип зависит от варианта расчёта" for i in ids],
            "majority": [True] * len(ids),
        }
    )


def test_flag_codes_and_cross_check():
    """Коды флага и слова из mo_flags.csv; сверка с типами и счётом node_seed — расхождение ловится."""
    types = {1: 2, 2: 1, 3: 3}
    f = _flags(types)
    codes, words = site_useful.flag_codes(f)
    assert codes.to_dict() == {1: "s", 2: "d", 3: "s"} and words == {
        "s": "тип устойчив",
        "d": "тип зависит от варианта расчёта",
    }
    nt = pd.Series(types)
    rob = {
        "rob_rule": pd.Series({1: "3/3", 2: "1/3", 3: "3/3"}),
        "rob_seed": pd.Series({1: "4/4", 2: "4/4", 3: "4/4"}),
    }
    assert site_useful.check_flags(f, nt, rob) == []
    assert site_useful.check_flags(f, pd.Series({1: 2, 2: 1, 3: 4}), rob)  # другой тип
    rob_bad = rob | {"rob_rule": pd.Series({1: "2/3", 2: "1/3", 3: "3/3"})}
    assert site_useful.check_flags(f, nt, rob_bad)
    f2 = f.copy()
    f2.loc[1, "flag"] = "stable"  # «устойчив», а вариантов совпало 1 из 3
    assert site_useful.check_flags(f2, nt, {})
    f3 = f.copy()
    f3.loc[0, "flag"], f3.loc[0, "variant_runs"], f3.loc[0, "flag_text"] = (
        "partial_check",
        2,
        "проверен в 2 из 3 вариантов",
    )
    assert site_useful.flag_codes(f3)[0][1] == "p2"


# --- этап site целиком (синтетика) -----------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _no_layout(monkeypatch):
    empty = pd.DataFrame({"territory_id": [], "nx": [], "ny": []})
    monkeypatch.setattr(landing, "similarity", lambda cfg: ({"chosen": None, "preserved": {}}, empty))


def _setup(tmp_path):
    make_site_inputs(tmp_path)
    d = site_config(CFG.data, tmp_path)
    d["site"]["build"]["controls"] = dict(CONTROLS)
    cfg = Config(d, tmp_path / "cfg.yaml")
    combo = next(all_combos(CFG.data)) | {
        "T3_reliable_placebo": "confirmed",
        "T1_ladder_external": "confirmed",
    }
    facts = make_facts(CFG.data, combo, **facts_over())
    facts.pop("synthetic")
    idir = bound_interpret(tmp_path, tmp_path / "outputs", facts)
    return cfg, idir


def _write_useful(out: Path, idir: Path, uf: dict | None = None) -> Path:
    import hashlib

    d = out / "usefulness"
    d.mkdir(parents=True, exist_ok=True)
    uf = copy.deepcopy(uf or _uf())
    uf["inputs_sha256"] = {"facts": hashlib.sha256((idir / "facts.json").read_bytes()).hexdigest()}
    (d / "facts.json").write_text(json.dumps(uf, ensure_ascii=False), encoding="utf-8")
    _flags(page_types()).to_csv(d / "mo_flags.csv", index=False)
    return d


def _mo(tmp_path):
    mo = json.loads((tmp_path / "site" / "data" / "mo.json").read_text(encoding="utf-8"))
    return {i: {k: v[j] for k, v in mo.items()} for j, i in enumerate(mo["id"])}


def test_site_without_usefulness_warns_and_hides(tmp_path, caplog):
    cfg, _ = _setup(tmp_path)
    with caplog.at_level(logging.WARNING):
        story = landing.run(cfg)
    assert any("usefulness" in r.getMessage() for r in caplog.records)
    assert all(r["fl"] is None for r in _mo(tmp_path).values())
    assert "flag_words" not in story["card"]


def test_site_with_usefulness_flags_in_mo_and_card(tmp_path):
    cfg, idir = _setup(tmp_path)
    _write_useful(tmp_path / "outputs", idir)
    story = landing.run(cfg)
    rows = _mo(tmp_path)
    types = page_types()
    for i in types:
        assert rows[i]["fl"] == ("s" if i % 2 else "d"), i
    assert rows[101]["fl"] == rows[CITY]["fl"]  # район столицы — флаг города
    assert story["card"]["flag_words"]["s"] == "тип устойчив"
    # знак разности ошибок у отдельного МО (works, d) на страницу не попадает
    html = (tmp_path / "site" / "index.html").read_text(encoding="utf-8")
    assert '"works"' not in html and "rule_by_mo" not in html


def test_site_usefulness_stale_is_code1(tmp_path):
    cfg, idir = _setup(tmp_path)
    d = _write_useful(tmp_path / "outputs", idir)
    old = (idir / "facts.json").stat().st_mtime - 100
    os.utime(d / "facts.json", (old, old))
    with pytest.raises(MissingInputError, match="старше"):
        landing.run(cfg)
    cfg, idir = _setup(tmp_path)
    d = _write_useful(tmp_path / "outputs", idir)
    uf = json.loads((d / "facts.json").read_text(encoding="utf-8"))
    uf["inputs_sha256"]["facts"] = "0" * 64
    (d / "facts.json").write_text(json.dumps(uf, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(MissingInputError, match="sha256"):
        landing.run(cfg)


def test_site_flag_mismatch_is_code3(tmp_path):
    cfg, idir = _setup(tmp_path)
    d = _write_useful(tmp_path / "outputs", idir)
    f = pd.read_csv(d / "mo_flags.csv")
    f.loc[0, "type"] = 9
    f.to_csv(d / "mo_flags.csv", index=False)
    with pytest.raises(QCError, match="флаг"):
        landing.run(cfg)


def test_demo_has_no_usefulness(tmp_path, caplog):
    from fixtures.site.synth_facts import write_interpret

    cfg, idir = _setup(tmp_path)
    _write_useful(tmp_path / "outputs", idir)
    facts = json.loads((idir / "facts.json").read_text(encoding="utf-8")) | {"synthetic": True}
    with caplog.at_level(logging.WARNING):
        story = landing.run(cfg, demo=write_interpret(tmp_path / "synth", facts))
    assert "flag_words" not in story["card"]
    assert any("обычный режим" in r.getMessage() for r in caplog.records)
    html = (tmp_path / "outputs" / "site_demo" / "index.html").read_text(encoding="utf-8")
    assert not re.search(r'class="fd-share"|id="use-example"', html)


# --- правки текстов по check-ux (порция 6b) -------------------------------------------------------------


def _story(combo_over: dict, numbers: dict | None = None):
    from fixtures.site.synth_facts import combo_levels  # noqa: F401

    combo = next(all_combos(CFG.data)) | combo_over
    facts = make_facts(CFG.data, combo)
    demo = landing.Mode("demo", Path("."), Path("."), "ДЕМО")
    return landing.build_story(CFG, facts, numbers or {}, demo), facts


def test_point2_shows_t3_head_next_to_neutral_question():
    """Пункт 2 «Что проверяли»: при нейтральном подзаголовке T2 рядом — короткий заголовок исхода T3."""
    st, _ = _story({"T3_reliable_placebo": "not"})
    h = CFG["site"]["headlines"]
    p2 = st["screen0"]["point_heads"][1]
    assert p2[0] == h["descriptive"]["dynamics_neutral"] and p2[1] == h["T3_reliable_placebo"]["not"]
    st, _ = _story({"T3_reliable_placebo": "confirmed", "T2_direction": "confirmed"})
    assert all(x != h["descriptive"]["dynamics_neutral"] for x in st["screen0"]["point_heads"][1])


def test_search_none_types_explain_comp_link_from_numbers():
    """Строка пустого поиска — числа разведки; пояснение T5 — только при своём вердикте; связка главы 6 —
    медианы обеих целей из facts.t7."""
    nums = {"n_absent": "8 регионов", "absent": "А, Б", "n_missing_mo": "153 муниципалитета",
            "missing_outside": "Дагестан — 39"}  # fmt: skip
    assert landing.fill_if_all(TX["search_none"], nums).startswith("Нет в данных: 8 регионов целиком — А, Б;")
    assert landing.fill_if_all(TX["search_none"], nums | {"missing_outside": ""}) is None
    t5 = {"max_ami": 0.3129, "max_label": "уровень трат, по возрастанию"}
    s = landing.types_explain(TX["types_explain"]["not_repeats"], t5)
    assert "«уровень трат, по возрастанию» типы совпадают на 0,31" in s
    assert landing.types_explain(TX["types_explain"].get("confirmed"), t5) is None
    t7 = {
        "product": "D",
        "median_error": {"B": 0.0362, "D": 0.0496},
        "median_error_abs": {"B": 0.0362, "D": 0.0416},
    }
    s = landing.comp_link(TX["comp_link"], t7)
    assert "0,036 против 0,042" in s and "0,036 против 0,050" in s
    combo = next(all_combos(CFG.data)) | {"T5_trivial": "not_repeats"}
    facts = make_facts(CFG.data, combo)
    facts["t5"] = dict(facts.get("t5") or {}) | t5
    facts["t7"] = dict(facts["t7"]) | t7
    st = landing.build_story(CFG, facts, {}, landing.Mode("demo", Path("."), Path("."), "ДЕМО"))
    assert (
        "0,31" in st["chapters"]["types"]["explain"] and "0,042" in st["chapters"]["comparable"]["comp_link"]
    )


def test_hero_why_and_stood_title():
    """Строка «зачем» на первом экране; заголовок «Что устояло» — без «оси», ρ расшифрована один раз."""
    st, _ = _story({})
    # порция 6e (check-ux): «зачем» — первой строкой под заголовком; «почему верить» — отдельно:
    # предрегистрация — только о проверках, совет сверки найден в результатах (порция 6d, check-facts)
    hero = st["screen0"]["hero"]
    assert hero["why"].startswith("Зачем:") and "с кем его сравнивать" in hero["why"]
    trust = hero["why_trust"]
    assert (
        trust.startswith("Проверки смысла типов записаны до расчётов") and "найден уже в результатах" in trust
    )
    assert "Высота" not in hero["lede_flat"] and "на этой карте нет" in hero["lede_flat"]
    f = TX["findings"]
    assert f["stood"]["title"] == "Тот же порядок виден в обороте Росстата"
    assert f["stood"]["text"].count("ранговая корреляция Спирмена") == 1 and "вскрыт" not in f["use"]["note"]
    banned = list(CFG["site"]["forbidden_words"]["headlines_always"]) + list(
        CFG["interpret"]["naming"]["banned"]
    )
    for t in (
        f["title"],
        f["stood"]["title"],
        f["border"]["title"],
        f["use"]["title"],
        f["use"]["title_region"],
    ):
        assert not any(landing.norm(w) in landing.norm(t) for w in banned), t
