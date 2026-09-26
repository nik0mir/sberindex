import json
import sys
import types

import numpy as np
import pandas as pd
import pytest
from synth import make_config, make_processed

import munnet.eda as eda
from munnet import style
from munnet.cli import EXIT_OK, EXIT_QC, main
from munnet.eda.base import EdaCheckError, empty_finding

ALT = "Синтетический график для проверки прогона: доля растёт по месяцам равномерно"


def fake_section(sid: str, *, headline_ok: bool = True, calls: list | None = None) -> types.ModuleType:
    """Фиктивный раздел: один факт, один график, одна таблица, показатель МО; запоминает случайное число."""
    module = types.ModuleType(f"munnet_fake_{sid}")
    module.SECTION_ID = sid
    module.TITLE = f"Фиктивный раздел {sid}"

    def run_section(ctx):
        draw = float(ctx.rng.random())
        if calls is not None:
            calls.append((sid, draw))
        n = len(ctx.data.mo)
        ctx.fact("n_mo", n, "int")
        ctx.fact("draw", draw, "num3")
        fig, ax = style.new_figure("half")
        ax.plot([0, 1], [0, 1])
        number = int(sid[1:])
        title = ctx.headline(f"Раздел {sid}: {n} МО", headline_ok, f"F0{number}: n = {n}")
        ctx.save_figure(
            fig,
            fid=f"F0{number}",
            slug=f"demo_{sid}",
            title=title,
            subtitle="Синтетика",
            alt=ALT,
            data=pd.DataFrame({"x": [0, 1]}),
            check="n > 0",
        )
        ctx.save_table(
            ctx.data.mo[["territory_id", "level_2024"]].head(3),
            tid=f"T0{number}",
            slug=f"t_{sid}",
            title="Таблица",
        )
        ind = ctx.data.mo[["territory_id"]].assign(**{f"ind_{sid}": ctx.data.mo["log_level_2024"]})
        return ctx.finding(
            title=module.TITLE,
            summary_md=f"{{{{{sid}.n_mo}}}} МО",
            indicators=ind,
            indicator_labels={f"ind_{sid}": "Показатель"},
        )

    module.run_section = run_section
    return module


@pytest.fixture
def cfg(tmp_path):
    make_processed(tmp_path)
    return make_config(tmp_path)


@pytest.fixture
def sections(monkeypatch):
    """Разделы e1 и e2 — фиктивные модули; сводка и отчёт по умолчанию отсутствуют."""
    installed = {}

    def install(*mods):
        names = []
        for mod in mods:
            monkeypatch.setitem(sys.modules, mod.__name__, mod)
            names.append(mod.__name__)
            installed[mod.SECTION_ID] = mod
        monkeypatch.setattr(eda, "SECTIONS", tuple(names))
        monkeypatch.setattr(eda, "SECTION_IDS", tuple(m.SECTION_ID for m in mods))

    monkeypatch.setattr(eda, "SYNTHESIS_MODULE", "munnet_fake_missing_synthesis")
    monkeypatch.setattr(eda, "REPORT_MODULE", "munnet_fake_missing_report")
    return install


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_run_writes_shared_outputs(cfg, sections):
    sections(fake_section("e1"), fake_section("e2"))
    eda.run(cfg)
    out = eda.eda_dir(cfg)
    facts = read_json(out / "facts.json")
    assert facts["e1.n_mo"]["text"] == "74" and "e2.draw" in facts
    figures = read_json(out / "figures.json")
    assert [f["fid"] for f in figures] == ["F01", "F02"]
    assert (out / figures[0]["png"]).exists()
    assert [t["tid"] for t in read_json(out / "tables.json")] == ["T01", "T02"]
    ind = pd.read_parquet(out / "indicators.parquet")
    assert list(ind.columns) == ["territory_id", "ind_e1", "ind_e2"] and len(ind) == 74
    assert (out / "sections" / "e1.json").exists() and (out / "sections" / "e2_indicators.parquet").exists()
    text = (out / "facts.json").read_text(encoding="utf-8")
    assert text.startswith('{\n "e1.draw"')  # sort_keys, indent=1, кириллица без \u-экранирования


def test_headline_error_stops_report(cfg, sections, monkeypatch):
    calls = []
    report = types.ModuleType("munnet_fake_report")
    report.write_report = lambda cfg, findings, synthesis: calls.append(len(findings))
    monkeypatch.setitem(sys.modules, "munnet_fake_report", report)
    monkeypatch.setattr(eda, "REPORT_MODULE", "munnet_fake_report")
    sections(fake_section("e1"), fake_section("e2", headline_ok=False))
    with pytest.raises(EdaCheckError, match="F02: n = 74"):
        eda.run(cfg)
    assert (eda.eda_dir(cfg) / "facts.json").exists()  # выходы записаны, отчёт — нет
    assert calls == []
    sections(fake_section("e1"), fake_section("e2"))
    eda.run(cfg)
    assert calls == [2]


def test_only_touches_own_section_files(cfg, sections):
    calls = []
    sections(fake_section("e1", calls=calls), fake_section("e2", calls=calls))
    eda.run(cfg)
    out = eda.eda_dir(cfg)
    shared = {name: (out / name).read_bytes() for name in ("facts.json", "figures.json", "tables.json")}
    e1_before = (out / "sections" / "e1.json").read_bytes()
    eda.run_sections(cfg, ["e2"])
    assert {name: (out / name).read_bytes() for name in shared} == shared
    assert (out / "sections" / "e1.json").read_bytes() == e1_before
    assert calls[1] == calls[2]  # e2 получил тот же генератор, что и в полном прогоне


def test_only_via_cli_and_syn_reassembles(cfg, sections, tmp_path):
    sections(fake_section("e1"), fake_section("e2"))
    config_path = tmp_path / "cfg.yaml"
    import yaml

    config_path.write_text(yaml.safe_dump(cfg.data, allow_unicode=True), encoding="utf-8")
    assert main(["--config", str(config_path), "eda", "--only", "e1"]) == EXIT_OK
    out = eda.eda_dir(cfg)
    assert (out / "sections" / "e1.json").exists() and not (out / "facts.json").exists()
    assert (
        main(["--config", str(config_path), "eda", "--only", "syn"]) == EXIT_OK
    )  # e2 не сохранён — без него
    assert "e1.n_mo" in read_json(out / "facts.json") and "e2.n_mo" not in read_json(out / "facts.json")


def test_only_with_headline_error_exits_3(cfg, sections, tmp_path):
    sections(fake_section("e1", headline_ok=False))
    config_path = tmp_path / "cfg.yaml"
    import yaml

    config_path.write_text(yaml.safe_dump(cfg.data, allow_unicode=True), encoding="utf-8")
    assert main(["--config", str(config_path), "eda", "--only", "e1"]) == EXIT_QC
    assert (eda.eda_dir(cfg) / "sections" / "e1.json").exists()  # итог раздела всё равно сохранён


def test_missing_and_unimplemented_sections_are_skipped(cfg, sections, monkeypatch, caplog):
    broken = fake_section("e2")

    def not_ready(ctx):
        raise NotImplementedError("раздел в работе")

    broken.run_section = not_ready
    sections(fake_section("e1"), broken)
    monkeypatch.setattr(eda, "SECTIONS", ("munnet_fake_e1", "munnet_fake_e2", "munnet_fake_absent"))
    monkeypatch.setattr(eda, "SECTION_IDS", ("e1", "e2", "e3"))
    eda.run(cfg)
    facts = read_json(eda.eda_dir(cfg) / "facts.json")
    assert "e1.n_mo" in facts and not any(k.startswith(("e2.", "e3.")) for k in facts)
    assert "e3" in caplog.text and "e2 не реализован" in caplog.text


def test_synthesis_gets_findings_and_its_own_rng(cfg, sections, monkeypatch):
    seen = {}
    synthesis = types.ModuleType("munnet_fake_synthesis")

    def run_synthesis(ctx, findings):
        seen["sections"] = [f.section for f in findings]
        seen["draw"] = float(ctx.rng.random())
        ctx.fact("top_story", "С5", "str")
        return ctx.finding(title="Сводка", summary_md="")

    synthesis.run_synthesis = run_synthesis
    monkeypatch.setitem(sys.modules, "munnet_fake_synthesis", synthesis)
    monkeypatch.setattr(eda, "SYNTHESIS_MODULE", "munnet_fake_synthesis")
    sections(fake_section("e1"), fake_section("e2"))
    eda.run(cfg)
    assert seen["sections"] == ["e1", "e2"]
    assert seen["draw"] == np.random.default_rng([cfg["seed"], 6]).random()
    assert read_json(eda.eda_dir(cfg) / "facts.json")["syn.top_story"]["text"] == "С5"


def test_merge_indicators_accepts_repeats_and_rejects_conflicts():
    a = empty_finding("e1", "a")
    a.indicators = pd.DataFrame({"territory_id": [1, 2], "x": [1.0, np.nan]})
    b = empty_finding("syn", "b")
    b.indicators = pd.DataFrame({"territory_id": [2, 1], "x": [np.nan, 1.0], "y": [5.0, 6.0]})
    merged = eda.merge_indicators([a, b])
    assert merged.columns.tolist() == ["territory_id", "x", "y"] and merged["y"].tolist() == [6.0, 5.0]
    b.indicators.loc[1, "x"] = 2.0
    with pytest.raises(ValueError, match="расходится"):
        eda.merge_indicators([a, b])


def test_run_without_panel_outputs_is_clear(tmp_path):
    with pytest.raises(FileNotFoundError, match="сначала запустите этап panel"):
        eda.run(make_config(tmp_path))
    with pytest.raises(ValueError, match="неизвестные разделы"):
        eda.run_sections(make_config(tmp_path), ["e7"])
