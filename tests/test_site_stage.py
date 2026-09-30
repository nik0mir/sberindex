"""Этап site на синтетике (``docs/landing_spec.md``, §4.3 и §5): словарь заголовков и заморозка, story.json
на всех сочетаниях вердиктов без нарушений линта, коды выхода 1 и 3, режимы --demo и --dev-blind.
Без сети, без data/ и без outputs/interpret: facts.json — из шаблонов предрегистрации (fixtures/site)."""

from __future__ import annotations

import copy
import json
import os
import re
from pathlib import Path

import pandas as pd
import pytest
from fixtures.site.synth_facts import (
    all_combos,
    combo_levels,
    make_facts,
    site_config,
    write_interpret,
)
from fixtures.site.synth_inputs import (
    CITY,
    CONTROLS,
    INNER,
    UNTYPED,
    bound_interpret,
    facts_over,
    make_site_inputs,
    page_types,
)

from munnet import cli, landing
from munnet.config import Config, load_config
from munnet.contracts import MissingInputError, QCError
from munnet.site_headlines import SLOT, check_headlines, freeze_hash

CFG = load_config("configs/default.yaml")
NUMBERS = {
    "share_cross": "96,9%",
    "share_cross_random": "98,2%",
    "n_rhythm": "311",
    "share_rhythm": "17,5%",
    "share_rhythm_null": "1,5%",
}
DEMO = landing.Mode("demo", Path("."), Path("."), "ДЕМО: синтетические данные")
MAIN6 = (
    "T1_ladder_external",
    "T2_direction",
    "T3_reliable_placebo",
    "T5_trivial",
    "T7_utility",
    "T7_type_gain",
)


def _strings(obj):
    return [s for _, s in landing._strings(obj)]


# --- словарь заголовков: перенос прототипа ------------------------------------------------------------------


def test_headlines_on_config():
    """Текущий конфиг: 0 нарушений заголовков и сочетаний, отрицательный контроль пойман весь, хеш совпал."""
    rep = check_headlines(CFG.data)
    assert rep.missing_levels == []
    assert rep.headline_violations == []
    assert rep.n_combos == 25920
    assert rep.combo_violations == []
    assert len(rep.control) == 29
    assert rep.caught == 29
    assert rep.sha256 == CFG["site"]["freeze"]["sha256"]
    assert rep.ok and rep.problems() == []
    lines = rep.lines()
    assert lines[-3:] == [
        "сочетаний: 25920",
        "нарушений (уникальных): 0",
        f"sha256: {rep.sha256} в конфиге: {rep.sha256}",
    ]
    assert "отрицательный контроль: поймано 29 из 29" in lines


def test_headlines_catch_edits():
    """Правка заголовка ловится дважды: линтом (выпало отрицание) и заморозкой (хеш)."""
    d = copy.deepcopy(CFG.data)
    d["site"]["headlines"]["T3_reliable_placebo"]["not"] = "Число смен типа за год отличается от плацебо"
    rep = check_headlines(d)
    probs = " ".join(rep.problems())
    assert "polarity" in probs
    assert "sha256" in probs
    d["site"]["freeze"]["sha256"] = freeze_hash(d["site"])
    assert any("polarity" in p for p in check_headlines(d).problems())


def test_negative_control_must_catch_all():
    """Пропущенный плохой заголовок контроля — нарушение: проверка, которая не ловит, не проверяет."""
    d = copy.deepcopy(CFG.data)
    good = d["site"]["headlines"]["T3_reliable_placebo"]["not"]
    d["site"]["headlines"]["check"]["negative_control"].append(
        {"test": "T3_reliable_placebo", "key": "not", "h": good}
    )
    rep = check_headlines(d)
    assert rep.caught == 29 and len(rep.control) == 30
    assert any("отрицательный контроль пропустил 1" in p for p in rep.problems())


# --- поля текстов этапа 5 -----------------------------------------------------------------------------------


def test_extract_fields_roundtrip():
    tpl = CFG["interpret"]["tests"]["T3_reliable_placebo"]["outcomes"]["confirmed"]["text"]
    combo = next(all_combos(CFG.data)) | {"T3_reliable_placebo": "confirmed", "one_in_ten": True}
    text = make_facts(CFG.data, combo)["texts"]["T3_reliable_placebo"]
    got = landing.extract_fields(tpl, text)
    assert got["excess"] == "79" and got["excess_ci"] == "74–83" and got["one_in_ten_phrase"].strip()
    assert landing.extract_fields(tpl, text.replace("Сверх плацебо", "Выше плацебо")) is None
    # поле в начале шаблона: текст начинается с прописной (interpret.texts.cap), значение — как подставлено
    tpl7 = CFG["interpret"]["tests"]["T7_utility"]["outcomes"]["partial"]["text"]
    t7 = make_facts(CFG.data, combo | {"T7_utility": "partial", "T7_type_gain": "neutral"})["texts"][
        "T7_utility"
    ]
    assert landing.extract_fields(tpl7, t7)["set_name"] == "сопоставимые территории"


# --- story.json на всех сочетаниях --------------------------------------------------------------------------


def _check_story(story, facts, combo):
    v = combo
    t1 = v["T1_ladder_external"]
    assert landing.lint_story(CFG, story, facts) == []
    assert not [s for s in _strings(story) if SLOT.search(s)]
    view = story["view"]
    assert view["palette"] == ("ordered" if t1 == "confirmed" else "nominal")
    assert view["ladder_words"] == (t1 == "confirmed")
    assert view["t1_layout"] == CFG["site"]["palette_rule"]["t1_chapter_form"]["layout"][t1]
    assert view["show_rival"] == (v["T5_trivial"] != "confirmed")
    t3_ok = v["T3_reliable_placebo"] in ("confirmed", "partial")
    assert view["arrows"] == (v["T2_direction"] in ("confirmed", "partial_capped") and t3_ok)
    assert view["transitions_default_on"] == t3_ok
    assert view["product_set"] == ("A" if v["T7_type_gain"] == "adds" else "D")
    assert view["forecast_words"] is False
    assert view["show_t6_caveat"] == v["caveat"]
    assert (story["screen0"]["t6_caveat"] is not None) == v["caveat"]
    assert view["one_in_ten"] == (v["T3_reliable_placebo"] == "confirmed" and v["one_in_ten"])
    ch = story["chapters"]
    assert all(c["title"] for c in ch.values())
    H = CFG["site"]["headlines"]
    if t3_ok:
        assert ch["limits"]["title"] != H["descriptive"]["limits_neutral"]
    else:
        assert ch["dynamics"]["lead"] == H["descriptive"]["dynamics_neutral"]
        assert ch["limits"]["title"] == H["descriptive"]["limits_neutral"]
        assert story["roles"]["analysts"] is None
    assert story["screen0"]["points"] == [facts["thesis"][k] for k in ("point_1", "point_2", "point_3")]
    assert story["verdicts"]["T2_direction"] == v["T2_direction"]


def test_story_all_verdict_combos():
    """4 × 5 × 3 × 4 × 3 × 3 = 2160 сочетаний T1, T2, T3, T5, T7, type_gain; T6 и флаги — по кругу."""
    L = combo_levels(CFG.data)
    n = 0
    for i, combo in enumerate(all_combos(CFG.data, MAIN6)):
        combo = combo | {
            "T6_bank_coverage": L["T6_bank_coverage"][i % 3],
            "caveat": bool(i % 2),
            "one_in_ten": bool((i // 2) % 2),
        }
        facts = make_facts(CFG.data, combo)
        _check_story(landing.build_story(CFG, facts, NUMBERS, DEMO), facts, combo)
        n += 1
    assert n == 2160


def test_story_t3_t6_flags():
    """Все сочетания T3 × T6 × оговорка × one_in_ten × T2 (флаги, от которых зависят глава 5 и глава 7)."""
    keys = ("T2_direction", "T3_reliable_placebo", "T6_bank_coverage", "caveat", "one_in_ten")
    for combo in all_combos(CFG.data, keys):
        facts = make_facts(CFG.data, combo)
        _check_story(landing.build_story(CFG, facts, NUMBERS, DEMO), facts, combo)


def test_t1_text_cap_used():
    """T1 после потолка по T5 (T1_text) — именно он в тексте и определяет палитру."""
    combo = next(all_combos(CFG.data)) | {"T1_ladder_external": "partial_overall"}
    facts = make_facts(CFG.data, combo)
    facts["verdicts_final"]["T1_ladder_external"] = "confirmed"  # до потолка
    story = landing.build_story(CFG, facts, NUMBERS, DEMO)
    assert story["verdicts"]["T1_ladder_external"] == "partial_overall"
    assert story["view"]["palette"] == "nominal"


# --- run(): коды выхода и режимы ----------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _no_layout(monkeypatch):
    """Раскладка главы 2 требует выходов cluster — в синтетике её заменяет сводка «перестановки нет»."""
    empty = pd.DataFrame({"territory_id": [], "nx": [], "ny": []})
    monkeypatch.setattr(landing, "similarity", lambda cfg: ({"chosen": None, "preserved": {}}, empty))


def _setup(tmp_path, **over):
    make_site_inputs(tmp_path)
    d = site_config(CFG.data, tmp_path)
    d["site"]["build"]["controls"] = dict(CONTROLS)
    cfg = Config(d, tmp_path / "cfg.yaml")
    combo = next(all_combos(CFG.data)) | {
        "T3_reliable_placebo": "confirmed",
        "T1_ladder_external": "confirmed",
    }
    facts = make_facts(CFG.data, combo, **(facts_over() | over))
    return cfg, facts


def _write_cfg(cfg: Config) -> Path:
    import yaml

    p = Path(cfg.path)
    with open(p, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg.data, f, allow_unicode=True, sort_keys=False)
    return p


def test_demo_writes_story_only_to_demo_dir(tmp_path):
    cfg, facts = _setup(tmp_path)
    src = write_interpret(tmp_path / "synth", facts)
    story = landing.run(cfg, demo=src)
    out = tmp_path / "outputs" / "site_demo" / "data" / "story.json"
    assert out.exists()
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved["meta"]["banner"] == CFG["site"]["demo"]["banner"] == story["meta"]["banner"]
    assert saved["chapters"]["similarity"]["title"].startswith("После снятия регионального фона 50,0%")
    assert "25,0%" in saved["roles"]["business"]
    assert not (tmp_path / "site").exists()


def test_code1_no_facts(tmp_path):
    cfg, _ = _setup(tmp_path)
    with pytest.raises(MissingInputError, match="нет"):
        landing.run(cfg)
    assert cli.main(["--config", str(_write_cfg(cfg)), "site"]) == 1


def test_code1_status_not_done(tmp_path):
    cfg, facts = _setup(tmp_path, status="running")
    write_interpret(tmp_path / "outputs", facts)
    with pytest.raises(MissingInputError, match="status"):
        landing.run(cfg)


def test_code1_older_than_labels(tmp_path):
    cfg, facts = _setup(tmp_path)
    d = write_interpret(tmp_path / "outputs", facts)
    lab = tmp_path / "processed" / "cluster_labels.parquet"
    old = lab.stat().st_mtime - 60
    os.utime(d / "facts.json", (old, old))
    with pytest.raises(MissingInputError, match="старше"):
        landing.run(cfg)


def test_code3_blind_outputs(tmp_path):
    cfg, facts = _setup(tmp_path, blind=20260930, label="СЛЕПОЙ ПРОГОН")
    write_interpret(tmp_path / "outputs", facts)
    with pytest.raises(QCError, match="слепые"):
        landing.run(cfg)
    assert cli.main(["--config", str(_write_cfg(cfg)), "site"]) == 3
    assert not (tmp_path / "site").exists()


def test_code3_blind_dir_name(tmp_path):
    """Каталог interpret_blind — слепые выходы, даже если флаг в facts стёрт."""
    cfg, facts = _setup(tmp_path)
    d = write_interpret(tmp_path / "outputs", facts, name="interpret_blind")
    cfg.data["paths"]["outputs"] = str(tmp_path / "outputs")
    with pytest.raises(QCError, match="слепые"):
        landing.load_facts(cfg, landing.Mode("normal", d, tmp_path / "site", None))


def test_code3_synthetic_in_normal_mode(tmp_path):
    """Синтетика в обычном режиме не публикуется."""
    cfg, facts = _setup(tmp_path)
    write_interpret(tmp_path / "outputs", facts)
    with pytest.raises(QCError, match="синтетич"):
        landing.run(cfg)


def test_dev_blind_mode(tmp_path):
    """--dev-blind пишет только в свой каталог с плашкой; не слепые выходы в этом режиме — код 3."""
    cfg, facts = _setup(tmp_path, blind=20260930, label="СЛЕПОЙ ПРОГОН")
    facts.pop("synthetic")
    d = write_interpret(tmp_path / "blind", facts, name="interpret_blind")
    story = landing.run(cfg, dev_blind=d)
    assert story["meta"]["banner"] == "СЛЕПОЙ ПРОГОН" and story["meta"]["mode"] == "dev_blind"
    assert (tmp_path / "outputs" / "site_dev_blind" / "data" / "story.json").exists()
    assert not (tmp_path / "site").exists()
    _, real = _setup(tmp_path)
    d2 = write_interpret(tmp_path / "notblind", real)
    with pytest.raises(QCError, match="не слепой"):
        landing.run(cfg, dev_blind=d2)
    p = _write_cfg(cfg)
    assert cli.main(["--config", str(p), "site", "--dev-blind", str(d)]) == 0


def test_code3_blind_with_marks_stripped(tmp_path):
    """Слепые выходы со стёртыми blind и label в каталоге interpret: размеры типов и sha256 входов те же,
    что у настоящих, — выдаёт поузловая сверка types.csv с cluster_final; код 3, site/ не тронут."""
    cfg, facts = _setup(tmp_path, blind=20260930, label="СЛЕПОЙ ПРОГОН")
    for k in ("synthetic", "blind", "label"):
        facts.pop(k)
    bound_interpret(tmp_path, tmp_path / "outputs", facts, shuffle=True)
    with pytest.raises(QCError, match="слепые или чужие"):
        landing.run(cfg)
    assert cli.main(["--config", str(_write_cfg(cfg)), "site"]) == 3
    assert not (tmp_path / "site").exists()


def test_code3_labels_sha_mismatch(tmp_path):
    """Выходы interpret другого прогона cluster: sha256 меток в facts ≠ файлу — код 3."""
    cfg, facts = _setup(tmp_path)
    facts.pop("synthetic")
    d = bound_interpret(tmp_path, tmp_path / "outputs", facts)
    f = json.loads((d / "facts.json").read_text(encoding="utf-8"))
    f["qc"]["inputs_sha256"]["cluster_final"] = "0" * 64
    (d / "facts.json").write_text(json.dumps(f, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(QCError, match="inputs_sha256.cluster_final"):
        landing.run(cfg)


def test_code1_no_types_csv(tmp_path):
    """Без types.csv сверить типы нельзя — код 1."""
    cfg, facts = _setup(tmp_path)
    facts.pop("synthetic")
    d = bound_interpret(tmp_path, tmp_path / "outputs", facts)
    (d / "types.csv").unlink()
    with pytest.raises(MissingInputError, match="types.csv"):
        landing.run(cfg)


def test_normal_mode_builds_site(tmp_path):
    """Все проверки прошли: страница и данные — в site.build.out, код 0; районы города — в его ячейке."""
    cfg, facts = _setup(tmp_path)
    facts.pop("synthetic")
    bound_interpret(tmp_path, tmp_path / "outputs", facts)
    assert cli.main(["--config", str(_write_cfg(cfg)), "site"]) == 0
    site = tmp_path / "site"
    for name in ("index.html", *(f"data/{n}.json" for n in ("story", "mo", "types", "checks", "hexgrid"))):
        assert (site / name).exists(), name
    mo = json.loads((site / "data" / "mo.json").read_text(encoding="utf-8"))
    rows = {i: {k: v[j] for k, v in mo.items()} for j, i in enumerate(mo["id"])}
    assert len(mo["id"]) == 8 and all(len(v) == 8 for v in mo.values())
    assert {r["role"] for r in rows.values()} == {"territorial", "city", "inner", "untyped"}
    want = page_types()
    for i, t in want.items():
        assert rows[i]["t"] == t
    for i in INNER:  # тип и ячейка района — города
        assert rows[i]["node"] == CITY and rows[i]["t"] == want[CITY]
        assert (rows[i]["hq"], rows[i]["hr"]) == (rows[CITY]["hq"], rows[CITY]["hr"])
    u = rows[UNTYPED]
    assert u["t"] is None and u["node"] is None and "12 из 24" in u["why_null"] and u["hq"] is not None
    cells = {(r["hq"], r["hr"]) for r in rows.values() if r["role"] != "inner"}
    assert len(cells) == 6  # 5 узлов + 1 без типа, каждая ячейка — одна
    assert rows[1]["win"] == "222222" + "1" * 7  # номера окон — после переноса (1 -> 2, 2 -> 1)
    assert (rows[1]["t23"], rows[1]["t24"]) == (2, 1)
    assert rows[1]["st"].count("r") == 7 and rows[1]["rel"] is True
    assert rows[1]["nb"] == [2] and rows[2]["nb"] == [1, 3]
    assert rows[1]["rh"] is not None and rows[2]["rh"] is None  # свой ритм — только надёжный
    html = (site / "index.html").read_text(encoding="utf-8")
    assert (
        '<svg id="hexmap" class="hexmap"' in html
        and 'data-hex-size="' in html
        and "$" not in html.split("<style>")[0]
    )
    assert 'id="data-mo"' in html  # inline_all: работает по file://
    # индекс поиска: регион — номер в списке rl, а не строка у каждого МО (бюджет первого экрана)
    names = json.loads(re.search(r'<script type="application/json" id="names">(.*?)</script>', html, re.S)[1])
    assert set(names) == {"id", "n", "r", "t", "rl"} and len(names["rl"]) == len(set(names["rl"]))
    by_id = dict(zip(names["id"], names["r"], strict=True))
    assert all(names["rl"][by_id[i]] == rows[i]["r"] for i in rows)


def test_code3_controls(tmp_path):
    """Контрольные числа site.build.controls и facts.scope против файлов: расхождение — код 3."""
    cfg, facts = _setup(tmp_path)
    cfg.data["site"]["build"]["controls"]["n_inner"] = 247
    with pytest.raises(QCError, match="n_inner"):
        _demo(tmp_path, facts, cfg)
    cfg, facts = _setup(tmp_path)
    facts["scope"]["n_nodes"] = 1776
    with pytest.raises(QCError, match="n_nodes"):
        _demo(tmp_path, facts, cfg)


def test_code3_type_sizes(tmp_path):
    """Размеры типов facts.ladder против типов узлов (обычный режим): расхождение — код 3."""
    cfg, facts = _setup(tmp_path)
    facts.pop("synthetic")
    facts["ladder"]["sizes_territorial"] = [2, 1, 1, 0]
    bound_interpret(tmp_path, tmp_path / "outputs", facts)
    with pytest.raises(QCError, match="контрольные числа"):
        landing.run(cfg)


def test_node_tables_to_mo(tmp_path):
    """Поузловые выгрузки этапа 5 попадают в mo.json как есть; без них — пустые поля (блок скрывается)."""
    cfg, facts = _setup(tmp_path)
    d = write_interpret(tmp_path / "synth", facts)
    pd.DataFrame(
        {"territory_id": [1, 1], "kind": ["variant", "seed"], "n_same": [2, 5], "n_runs": [3, 5]}
    ).to_csv(d / "node_seed.csv", index=False)
    pd.DataFrame(
        {"territory_id": [1, 2], "second_type": [3, 4], "margin": [0.01, 0.2], "own_nearest": [False, True]}
    ).to_csv(d / "node_margin.csv", index=False)
    pd.DataFrame(
        {"territory_id": [1] * 3, "set": ["D"] * 3, "product": [True] * 3, "rank": [1, 2, 3],
         "other_id": [3, 4, 2], "km": [900.4, 960.6, 60.0]}
    ).to_csv(d / "node_comparable.csv", index=False)  # fmt: skip
    landing.run(cfg, demo=d)
    mo = json.loads((tmp_path / "outputs" / "site_demo" / "data" / "mo.json").read_text(encoding="utf-8"))
    rows = {i: {k: v[j] for k, v in mo.items()} for j, i in enumerate(mo["id"])}
    assert rows[1]["rob_rule"] == "2/3" and rows[1]["rob_seed"] == "5/5" and rows[2]["rob_rule"] is None
    assert rows[1]["second"] == 3 and rows[2]["second"] is None  # граничное — ближе к чужому центру
    assert rows[1]["sim"] == [[3, 900], [4, 961], [2, 60]]
    assert rows[3]["sim"] is None and rows[1]["var"] is None


def _demo(tmp_path, facts, cfg):
    return landing.run(cfg, demo=write_interpret(tmp_path / "synth", facts))


def test_code3_forbidden_word_in_type_name(tmp_path):
    cfg, facts = _setup(tmp_path)
    facts["names_final"]["2"] = "Курортные муниципалитеты"
    with pytest.raises(QCError, match="курорт"):
        _demo(tmp_path, facts, cfg)


def test_code3_forbidden_word_in_site_text(tmp_path):
    """Запрет page_always в тексте, который пишет сайт, ловится и при новом хеше заморозки."""
    cfg, facts = _setup(tmp_path)
    cfg.data["site"]["headlines"]["descriptive"]["explore"] = "Найдите свой муниципалитет: прогноз типа"
    cfg.data["site"]["freeze"]["sha256"] = freeze_hash(cfg.data["site"])
    with pytest.raises(QCError, match="прогноз"):
        _demo(tmp_path, facts, cfg)


def test_code3_by_verdict_word(tmp_path):
    """«ступен» разрешено только при T1 confirmed: в роли, которую пишет сайт, при T1 not — код 3."""
    cfg, facts = _setup(tmp_path)
    story = landing.build_story(cfg, facts, NUMBERS, DEMO)
    story["verdicts"]["T1_ladder_external"] = "not"
    story["roles"]["business_label"] = "описание по ступеням"
    assert any("ступен" in b for b in landing.lint_story(cfg, story, facts))


def test_code3_empty_field(tmp_path):
    cfg, facts = _setup(tmp_path)
    facts["scope"]["n_untyped"] = None
    with pytest.raises(QCError, match="пустое поле"):
        _demo(tmp_path, facts, cfg)


def test_code3_text_not_matching_verdict(tmp_path):
    """Текст этапа 5 не по шаблону своего вердикта — тексты и вердикты разошлись."""
    cfg, facts = _setup(tmp_path)
    facts["verdicts_final"]["T3_reliable_placebo"] = "not"
    with pytest.raises(QCError, match="не совпал с шаблоном"):
        _demo(tmp_path, facts, cfg)


def test_code3_budget(tmp_path):
    cfg, facts = _setup(tmp_path)
    cfg.data["site"]["budget_mb"] = dict(cfg.data["site"]["budget_mb"]) | {"hard": 1e-6}
    with pytest.raises(QCError, match="бюджет"):
        _demo(tmp_path, facts, cfg)


def test_code3_freeze_hash(tmp_path):
    cfg, facts = _setup(tmp_path)
    cfg.data["site"]["decisions"]["fonts"] = "pt_fonts"
    with pytest.raises(QCError, match="sha256"):
        _demo(tmp_path, facts, cfg)


def test_nominal_palette_not_monotone(monkeypatch):
    """Номинальная палитра не подсказывает порядок светлотой ни в каком порядке ступеней из синтетики;
    светлота — одна на проект (style.lightness), а не своя копия."""
    assert not landing.nominal_monotone([2, 1, 3, 4])
    # номинальная, подменённая порядковой гаммой, монотонна; и признак считается через style.lightness
    monkeypatch.setattr(
        landing.style, "TYPE_PALETTE_NOMINAL", landing.style.type_palette("ordered", [2, 1, 3, 4])
    )
    assert landing.nominal_monotone([2, 1, 3, 4])
    monkeypatch.setattr(landing.style, "lightness", lambda c: {"#848BC7": 1, "#606CAF": 3}.get(c, 2))
    assert not landing.nominal_monotone([2, 1, 3, 4])


def test_code3_palette_audit(tmp_path, monkeypatch):
    """Этап site проверяет обе палитры (style.palette_audit) до сборки: провал — код 3, site/ не тронут."""
    cfg, facts = _setup(tmp_path)
    monkeypatch.setattr(
        landing.style, "TYPE_PALETTE_NOMINAL", {1: "#FFEE88", 2: "#B8563B", 3: "#6B7F2E", 4: "#1C5A8C"}
    )
    with pytest.raises(QCError, match="палитр.*nominal.contrast_bg.1"):
        _demo(tmp_path, facts, cfg)
    assert not (tmp_path / "outputs" / "site_demo").exists()


def test_code3_palette_shapes(tmp_path):
    """Фигуры конфига site.palette_rule.type_palette.shapes сверяются с style.TYPE_SHAPES. Ключ заморожен:
    в прогоне этапа его правку раньше ловит sha256, поэтому audit_palettes проверяется напрямую."""
    cfg, facts = _setup(tmp_path)
    cfg.data["site"]["palette_rule"]["type_palette"]["shapes"] = {1: "●", 2: "●", 3: "■", 4: "◆"}
    with pytest.raises(QCError, match="sha256"):
        _demo(tmp_path, facts, cfg)
    with pytest.raises(QCError, match="shapes.config"):
        landing.audit_palettes(cfg, facts)


def test_palette_audit_script(monkeypatch, capsys):
    """scripts/palette_audit.py — ссылка site.palette_rule.type_palette.audit — зовёт style.palette_audit."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("palette_audit_script", "scripts/palette_audit.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.main(["--config", "configs/default.yaml"]) == 0
    assert "nominal.not_monotone" in capsys.readouterr().out
    monkeypatch.setattr(mod.style, "TYPE_PALETTE_NOMINAL", mod.style.type_palette("ordered", [2, 1, 3, 4]))
    assert mod.main(["--config", "configs/default.yaml"]) == 3


# --- порция 3: экран 0 без JS, поля карточки, линт шаблона, T3 по R1 ----------------------------------------


def _html_text(html: str) -> str:
    """Видимый текст страницы без JS: без <script>, <style> и тегов; неразрывный пробел -> пробел."""
    h = re.sub(r"<(script|style)\b.*?</\1>", " ", html, flags=re.S)
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", h).replace(" ", " "))


def _squash(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace(" ", " "))


def test_screen0_readable_without_js(tmp_path):
    """§5 и §7: заголовок, подзаголовок, вводная, три пункта (короткие заголовки и все тексты пункта
    дословно под «Как проверяли»), оговорка, охват, ключ карты и подпись о смещении — в HTML при сборке.
    С порции 5a (§4.7) заголовок по T1 — h2 раздела «Что проверяли» под первым экраном, а не h1 страницы."""
    cfg, facts = _setup(tmp_path)
    facts.pop("synthetic")
    bound_interpret(tmp_path, tmp_path / "outputs", facts)
    story = landing.run(cfg)
    html = (tmp_path / "site" / "index.html").read_text(encoding="utf-8")
    text = _html_text(html)
    s0 = story["screen0"]
    assert '<h2 id="answer-title" class="answer-h">' in html and html.count("<h1") == 1
    assert _squash(s0["title"]) in text and _squash(s0["lead"]) in text
    assert s0["intro"] and _squash(s0["intro"]) in text
    assert html.count('<li class="pt') == 3 and html.count('<details class="how">') >= 3
    for point in s0["points"]:  # все тексты пункта, не только первый
        for t in point:
            assert _squash(t) in text, t[:40]
    for hs in s0["point_heads"]:
        assert hs and all(_squash(h) in text for h in hs)
    for k in ("question", "coverage", "scope"):
        assert _squash(s0[k]) in text, k
    assert "Данные. Данные" not in text and 'id="map-shift"> (медианное смещение' in html
    assert 'class="map-key"' in html and "нет типа" in text
    # заголовок и подзаголовок — по site.headlines.screen0 (T1, затем T3), порядок не меняется
    body = html[html.index("<body>") :]
    assert (
        body.index('id="answer-title"')
        < body.index('id="answer-lead"')
        < body.index('<div class="findings">')
    )
    # порция 5a: первый экран (h1, поиск, карта) — раньше раздела «Что проверяли»; на 375 px поиск —
    # сразу под заголовком и пояснением, до карты
    assert (
        body.index('id="hero-title"')
        < body.index('id="search-input"')
        < body.index('<svg id="hexmap"')
        < body.index('id="answer-title"')
    )


def test_banner_in_html_and_title(tmp_path):
    """Плашка режима — в HTML и в <title> без JS."""
    cfg, facts = _setup(tmp_path, blind=20260930, label="СЛЕПОЙ ПРОГОН")
    facts.pop("synthetic")
    d = write_interpret(tmp_path / "blind", facts, name="interpret_blind")
    landing.run(cfg, dev_blind=d)
    html = (tmp_path / "outputs" / "site_dev_blind" / "index.html").read_text(encoding="utf-8")
    assert '<div class="banner" id="banner" role="status">СЛЕПОЙ ПРОГОН</div>' in html
    assert re.search(r"<title>СЛЕПОЙ ПРОГОН\.", html)


def test_meta_without_build_date(tmp_path):
    """Страница не меняется от дня сборки: в meta нет даты, есть seed и sha256 выводов."""
    cfg, facts = _setup(tmp_path)
    facts.pop("synthetic")
    bound_interpret(tmp_path, tmp_path / "outputs", facts)
    landing.run(cfg)
    html = (tmp_path / "site" / "index.html").read_text(encoding="utf-8")
    meta = json.loads(re.search(r'<script type="application/json" id="meta">(.*?)</script>', html, re.S)[1])
    assert "date" not in meta and meta["seed"] == CFG["seed"] and meta["facts_sha256"]


def test_js_reads_only_mo_fields(tmp_path):
    """Поля МО, которые читает карточка (r.…, nd.…), есть в mo.json — иначе блок молча пропадает."""
    cfg, facts = _setup(tmp_path)
    facts.pop("synthetic")
    bound_interpret(tmp_path, tmp_path / "outputs", facts)
    landing.run(cfg)
    mo = json.loads((tmp_path / "site" / "data" / "mo.json").read_text(encoding="utf-8"))
    js = (Path(landing.__file__).parent / "templates" / "landing.js").read_text(encoding="utf-8")
    card = js[js.index("function statusLine") : js.index('cardBody.addEventListener("click"')]
    used = set(re.findall(r"\b(?:r|nd)\.([a-z][a-z0-9_]*)\b", card))
    missing = sorted(used - set(mo) - {"length"})
    assert not missing, missing
    assert {"wp", "ser", "simb"} <= set(mo)


def test_untyped_cells_are_clickable_in_js():
    """169 ячеек МО без типа попадают в CELLS (карточка объясняет причину), районы столиц — нет."""
    js = (Path(landing.__file__).parent / "templates" / "landing.js").read_text(encoding="utf-8")
    line = next(x for x in js.splitlines() if "CELLS.set(" in x)
    assert 'r.role !== "inner"' in line and "r.id === r.node" not in line


def test_verdict_fields_per_test():
    """Одно имя поля у разных тестов бывает о разном ({best_partition} у T1 и T5): заголовок берёт поле из
    текста своего теста; в общие тексты поле с разными значениями не идёт."""
    combo = next(all_combos(CFG.data)) | {
        "T1_ladder_external": "partial_overall",
        "T5_trivial": "not_repeats",
    }
    facts = make_facts(CFG.data, combo)
    it = CFG["interpret"]["tests"]
    t5 = landing.extract_fields(
        it["T5_trivial"]["outcomes"]["not_repeats"]["text"], facts["texts"]["T5_trivial"]
    )
    t1 = landing.extract_fields(
        it["T1_ladder_external"]["outcomes"]["partial_overall"]["text"], facts["texts"]["T1_ladder_external"]
    )
    if not (t1 and t5 and "best_partition" in t1 and "best_partition" in t5):
        pytest.skip("в шаблонах нет общего поля best_partition")
    other = "«другое деление»"
    facts["texts"]["T1_ladder_external"] = facts["texts"]["T1_ladder_external"].replace(
        t1["best_partition"], other, 1
    )
    facts["thesis"]["point_1"] = [facts["texts"]["T1_ladder_external"], facts["texts"]["T5_trivial"]]
    vd = landing.read_verdicts(CFG, facts)
    assert "best_partition" not in vd.fields
    assert vd.of("T5_trivial")["best_partition"] == t5["best_partition"]
    story = landing.build_story(CFG, facts, NUMBERS, DEMO)
    assert t5["best_partition"] in story["chapters"]["types"]["title"]
    assert landing.lint_story(CFG, story, facts) == []


def test_template_lint_catches_words(monkeypatch):
    """Линт шаблона: строки JS и текст HTML проверяются на page_always и by_verdict (иначе код 3)."""
    texts = landing.template_strings(
        '<p>Прогноз оборота</p><!-- прогноз в комментарии --><b aria-label="ступени лестницы">x</b>',
        'const a = "надёжный переход";\n// предсказание в комментарии\nconst b = `к городской корзине`;',
    )
    assert "Прогноз оборота" in texts and "ступени лестницы" in texts
    assert "надёжный переход" in texts and "к городской корзине" in texts
    assert not any("комментари" in t for t in texts)
    verdicts = {t: "not" for t in ("T1_ladder_external", "T2_direction", "T3_reliable_placebo")}
    verdicts |= {"T5_trivial": "not_repeats", "T6_bank_coverage": "not", "T7_utility": "not"}
    verdicts |= {"T7_type_gain": "neutral", "one_in_ten": False, "caveat": False}
    monkeypatch.setattr(landing, "template_strings", lambda *a: texts)
    bad = " ".join(landing.lint_templates(CFG, {"verdicts": verdicts}, ""))
    assert "прогноз" in bad and "надёжн" in bad and "ступен" in bad


def test_real_templates_pass_lint_all_verdicts():
    """Настоящие landing.html и landing.js проходят линт при всех сочетаниях вердиктов."""
    keys = ("T1_ladder_external", "T2_direction", "T3_reliable_placebo", "T7_type_gain")
    for combo in all_combos(CFG.data, keys):
        story = {"verdicts": combo | {"one_in_ten": False, "caveat": False}}
        assert landing.lint_templates(CFG, story, "") == [], combo


def test_nbsp_display_only():
    nb = " "
    assert landing.nbsp("Траты в регионе и в МО — это а") == f"Траты в{nb}регионе и{nb}в{nb}МО{nb}— это а"
    assert landing.nbsp("Минск-а б") == "Минск-а б"


def test_chapters_in_html_with_story_titles(tmp_path):
    """Главы 1, 3, 4, 5 вписаны при сборке; h2 каждой — заголовок главы из story.json (§7), без JS."""
    cfg, facts = _setup(tmp_path)
    facts.pop("synthetic")
    bound_interpret(tmp_path, tmp_path / "outputs", facts)
    assert cli.main(["--config", str(_write_cfg(cfg)), "site"]) == 0
    html = (tmp_path / "site" / "index.html").read_text(encoding="utf-8")
    story = json.loads((tmp_path / "site" / "data" / "story.json").read_text(encoding="utf-8"))
    for cid, key in (("basket", "basket"), ("types", "types"), ("order", "order"), ("dynamics", "dynamics")):
        m = re.search(rf'<h2 id="{cid}-title">(.*?)</h2>', html, re.S)
        assert m, cid
        assert _squash(_html_text(m[1])) == _squash(story["chapters"][key]["title"]), cid
    assert '<a href="#types">' in html and '<a href="#dynamics">' in html


# --- порция 5a: первый экран — объёмная карта (§4.7) -------------------------------------------------------

CAFE = {1: -0.5, 2: 0.1, 3: 0.7, 4: -0.05, CITY: 1.2}


def _values(cafe):
    """Значения признаков кластеризации (как landing.clustering_values) для синтетики: clr_rel_cafe узла."""
    return pd.DataFrame({"clr_rel_cafe": list(cafe.values()), "log_pop_rel": 0.0}, index=list(cafe))


def _build_with_scene(tmp_path, monkeypatch):
    cfg, facts = _setup(tmp_path)
    facts.pop("synthetic")
    bound_interpret(tmp_path, tmp_path / "outputs", facts)
    monkeypatch.setattr(landing, "clustering_values", lambda cfg: _values(CAFE))
    story = landing.run(cfg)
    site = tmp_path / "site"
    scene = json.loads((site / "data" / "scene.json").read_text(encoding="utf-8"))
    return cfg, story, site, scene


def test_hero_screen0(tmp_path, monkeypatch):
    """h1 — текст участника (site.build.texts.hero.title), числа надзаголовка — из facts.scope, охват —
    descriptive.coverage, легенда — в порядке view.legend_order с числом ячеек и «Без типа»; название проекта
    в <title> и шапке; three.js — из vendor/ через import map (без CDN), sha256 как в README."""
    import hashlib

    _, story, site, _ = _build_with_scene(tmp_path, monkeypatch)
    html = (site / "index.html").read_text(encoding="utf-8")
    text = _html_text(html)
    hx = CFG["site"]["build"]["texts"]["hero"]
    h1 = re.search(r'<h1 id="hero-title">(.*?)</h1>', html, re.S)[1]
    assert _squash(_html_text(h1)).strip() == hx["title"]
    s0 = story["screen0"]
    assert s0["hero"]["kicker"].startswith(f"{CONTROLS['n_nodes']} муниципалитетов в ")
    note = html.split('class="h0-note')[1].split("</p>")[0]
    assert _squash(s0["coverage"]) in _squash(_html_text(note))
    assert re.search(r"<title>Корзина и регион — типы муниципалитетов по тратам жителей</title>", html)
    brand = landing._t("Корзина и регион")  # ru-text: неразрывный пробел после «и»
    assert f'<a class="brand" href="#answer">{brand}' in html and "munnet<span" not in html
    names = story["names"]["final"]
    legend = html.split('<ul class="h0-types">')[1].split("</ul>")[0]
    pos = [legend.index(landing._t(names[str(t)])) for t in story["view"]["legend_order"]]
    assert pos == sorted(pos) and "Без типа" in legend
    assert "Без типа" in text and "×1" in text
    assert '"three": "./vendor/three.module.min.js"' in html and "cdn.jsdelivr" not in html
    readme = (Path(landing.__file__).parent / "templates" / "vendor" / "README.md").read_text(
        encoding="utf-8"
    )
    for name in ("three.module.min.js", "OrbitControls.js", "three-LICENSE.txt"):
        b = (site / "vendor" / name).read_bytes()
        assert hashlib.sha256(b).hexdigest() in readme, name
    assert 'id="data-scene"' in html and 'aria-label="Объёмная карта:' in html


def test_scene_heights_are_exp_clr_cafe(tmp_path, monkeypatch):
    """Высота ячейки = exp(clr_rel_cafe) из значений признаков кластеризации; у МО без типа — пусто."""
    import math

    _, _, _, scene = _build_with_scene(tmp_path, monkeypatch)
    c = scene["cells"]
    h = dict(zip(c["id"], c["h"], strict=True))
    for i, v in CAFE.items():
        assert h[i] == pytest.approx(math.exp(v), abs=1e-3), i
    assert h[UNTYPED] is None
    assert set(c["id"]) == {1, 2, 3, 4, CITY, UNTYPED}  # районы столицы — в ячейке города, отдельных нет
    t = dict(zip(c["id"], c["t"], strict=True))
    assert t[UNTYPED] == 0 and all(t[i] > 0 for i in CAFE)


def test_scene_islands(tmp_path, monkeypatch):
    """Острова: каждая ячейка ровно один раз, ячейки не пересекаются, острова — в порядке легенды (по размеру
    при T1 ≠ confirmed) и затем «Без типа», внутри острова от центра к краю — по убыванию высоты."""
    _, story, _, scene = _build_with_scene(tmp_path, monkeypatch)
    c = scene["cells"]
    isl = scene["islands"]
    present = set(c["t"])
    assert [i["t"] for i in isl] == [int(t) for t in story["view"]["legend_order"] if int(t) in present] + [0]
    assert sum(i["n"] for i in isl) == len(c["id"]) == len(set(c["id"]))
    xs = [i["x"] for i in isl]
    assert xs == sorted(xs)
    pts = list(zip(c["ix"], c["iy"], strict=True))
    step = scene["grid"]["size"] * 3**0.5
    for a in range(len(pts)):
        for b in range(a + 1, len(pts)):
            d = ((pts[a][0] - pts[b][0]) ** 2 + (pts[a][1] - pts[b][1]) ** 2) ** 0.5
            assert d >= step * 0.99
    for s in isl:
        mem = [j for j, t in enumerate(c["t"]) if t == s["t"]]
        dist = [((c["ix"][j] - s["x"]) ** 2 + (c["iy"][j] - s["y"]) ** 2) ** 0.5 for j in mem]
        if s["t"]:
            hs = [c["h"][j] for j in mem]
            by_dist = sorted(range(len(mem)), key=lambda k: dist[k])
            assert by_dist[0] == max(range(len(mem)), key=lambda k: hs[k])
            assert s["note"].startswith("у медианы типа доля кафе ")
        assert max(dist) <= s["rad"]


def test_scene_absent_without_values(tmp_path):
    """Без значений признаков кластеризации (синтетика) объёмной карты нет: первый экран — статичная SVG,
    vendor/ не копируется."""
    cfg, facts = _setup(tmp_path)
    facts.pop("synthetic")
    bound_interpret(tmp_path, tmp_path / "outputs", facts)
    landing.run(cfg)
    site = tmp_path / "site"
    assert not (site / "data" / "scene.json").exists() and not (site / "vendor").exists()
    assert '<svg id="hexmap"' in (site / "index.html").read_text(encoding="utf-8")


def test_code3_forbidden_word_in_hero_text(tmp_path):
    """Тексты первого экрана (site.build.texts.hero) линтуются как текст сайта; h1 — ещё и как заголовок."""
    cfg, facts = _setup(tmp_path)
    cfg.data["site"]["build"]["texts"]["hero"] = dict(cfg["site"]["build"]["texts"]["hero"]) | {
        "lede": "Высота столбика — прогноз доли кафе"
    }
    with pytest.raises(QCError, match="прогноз"):
        _demo(tmp_path, facts, cfg)
    cfg, facts = _setup(tmp_path)
    cfg.data["site"]["build"]["texts"]["hero"] = dict(cfg["site"]["build"]["texts"]["hero"]) | {
        "title": "Жители тратят на кафе по-разному"
    }
    with pytest.raises(QCError, match="запрет в заголовке"):
        _demo(tmp_path, facts, cfg)


def test_code3_by_verdict_word_in_island_note(tmp_path):
    """Подписи островов (scene.json) проходят те же запреты: «надёжн» при T3 not и пустое поле — код 3."""
    cfg, facts = _setup(tmp_path)
    story = landing.build_story(cfg, facts, NUMBERS, DEMO)
    story["verdicts"]["T3_reliable_placebo"] = "not"
    assert landing.lint_texts(cfg, story, ["у медианы типа надёжно больше"])
    assert landing.lint_texts(cfg, story, ["доля {ratio}"])
    assert landing.lint_texts(cfg, story, ["у медианы типа доля кафе на 3% меньше, чем в регионе"]) == []
