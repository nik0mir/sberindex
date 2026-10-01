import importlib
import math
import re
import sys
import types
from pathlib import Path

import pytest

from munnet import contracts, style
from munnet.cli import EXIT_MISSING_INPUT, EXIT_NOT_IMPLEMENTED, EXIT_OK, EXIT_QC, STAGES, main
from munnet.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def test_default_config_sources_are_pinned():
    sha256 = re.compile(r"[0-9a-f]{64}")
    for name, source in load_config()["sources"].items():
        if source.get("kind") == "zip_members":
            assert source["base_url"].startswith("https://"), name
            assert source["members"], name
            assert all(sha256.fullmatch(m["sha256"]) for m in source["members"].values()), name
        else:
            assert sha256.fullmatch(source["sha256"]), name
            assert source["urls"], name
            assert all(url.startswith("https://") for url in source["urls"]), name


def test_stage_order():
    assert list(STAGES) == [
        "data",
        "panel",
        "eda",
        "features",
        "network",
        "cluster",
        "evaluate",
        "dynamics",
        "interpret",
        "usefulness",
        "site",
    ]


def test_makefile_stages_match_cli():
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    line = next(s for s in makefile.splitlines() if s.startswith("STAGES ="))
    assert line.split("=", 1)[1].split() == list(STAGES)


def test_every_stage_has_run():
    for module_name, _ in STAGES.values():
        assert callable(importlib.import_module(module_name).run), module_name


def _fake_stage(monkeypatch, run) -> str:
    """Подменяет этап модулем-заглушкой: тест не зависит от того, реализован ли настоящий этап."""
    module = types.ModuleType("munnet_fake_stage")
    module.run = run
    monkeypatch.setitem(sys.modules, "munnet_fake_stage", module)
    monkeypatch.setitem(STAGES, "fake", ("munnet_fake_stage", "подменённый этап"))
    return "fake"


def test_unimplemented_stage_exits_with_code_2(monkeypatch):
    def run(cfg):
        raise NotImplementedError("заглушка")

    assert main([_fake_stage(monkeypatch, run)]) == EXIT_NOT_IMPLEMENTED == 2


def test_qc_error_exits_with_code_3(monkeypatch, caplog):
    def run(cfg):
        raise contracts.QCError("территорий 2189, ожидалось 2190")

    assert main([_fake_stage(monkeypatch, run)]) == EXIT_QC == 3
    assert "территорий 2189" in caplog.text


def test_missing_input_exits_with_code_1(monkeypatch, caplog):
    def run(cfg):
        raise contracts.MissingInputError("сначала запустите этап panel")

    assert main([_fake_stage(monkeypatch, run)]) == EXIT_MISSING_INPUT == 1
    assert "сначала запустите этап panel" in caplog.text


def test_successful_stage_exits_with_code_0(monkeypatch):
    calls = []
    assert main([_fake_stage(monkeypatch, calls.append)]) == EXIT_OK
    assert len(calls) == 1


def test_only_passes_section_ids_to_eda(monkeypatch):
    calls = {}
    module = types.ModuleType("munnet_fake_eda")
    module.run = lambda cfg: calls.setdefault("run", True)
    module.run_sections = lambda cfg, ids: calls.setdefault("ids", ids)
    monkeypatch.setitem(sys.modules, "munnet_fake_eda", module)
    monkeypatch.setitem(STAGES, "eda", ("munnet_fake_eda", "подменённая разведка"))
    assert main(["eda", "--only", "e3,e4", "--only", "syn"]) == EXIT_OK
    assert calls == {"ids": ["e3", "e4", "syn"]}


@pytest.mark.parametrize("argv", [["panel", "--only", "e3"], ["eda", "--only", "e9"], ["eda", "--only", ","]])
def test_only_rejects_bad_usage(argv):
    with pytest.raises(SystemExit):
        main(argv)


def test_stage1_sections_present():
    cfg = load_config()
    for section in ("panel", "context", "eda"):
        assert section in cfg.data, section


def test_category_codes_match_contracts_and_labels():
    panel = load_config()["panel"]
    codes = (*panel["categories"], panel["other"]["code"])
    assert codes == contracts.CATEGORY_CODES == style.CATEGORY_ORDER
    labels = {**panel["categories"], panel["other"]["code"]: panel["other"]["label"]}
    assert labels == style.LABELS


def test_panel_period_matches_contracts():
    cfg = load_config()
    assert tuple(cfg["panel"]["years"]) == contracts.YEARS
    assert cfg["period"]["start"] == f"{contracts.YEARS[0]}-01"
    assert cfg["period"]["end"] == f"{contracts.YEARS[-1]}-12"
    read_years = cfg["context"]["read_years"]
    assert (min(read_years), max(read_years)) == contracts.CONTEXT_YEAR_RANGE


def test_mo_type_and_status_codes_match_contracts():
    panel = load_config()["panel"]
    assert tuple(panel["mo_types"].values()) == contracts.MO_TYPES
    assert tuple(panel["mo_statuses"].values()) == contracts.MO_STATUSES
    assert panel["mo_statuses"][panel["capital_status"]] == "capital"


def test_matrix_weights_sum_to_one():
    weights = load_config()["eda"]["matrix_weights"]
    assert math.isclose(sum(weights.values()), 1.0)
    assert all(w > 0 for w in weights.values())


def test_eda_parameters_are_consistent():
    eda = load_config()["eda"]
    assert set(eda["summer_months"]) <= set(range(1, 13))
    assert 0 < eda["season"]["reliable_r"] < 1
    assert eda["level_rel_min_months"] <= 12
    limits = eda["figure_limits"]
    assert limits["alt_min"] > 0 and limits["title_max"] < limits["subtitle_max"]
    assert eda["figure"]["dpi"] > 0 and set(eda["figure"]["formats"]) <= {"png", "svg", "pdf"}
    for inset in eda["maps"]["insets"].values():
        assert isinstance(inset["region_code"], int) and inset["title"]
    for name, spec in eda["economy_types"].items():
        assert isinstance(spec["rule"], str) and spec["rule"], name


def test_eda_section_params_accept_config():
    """Секции параметров разделов и отчёта в конфиге читаются модулями: неизвестный ключ там — ошибка
    (ValueError или KeyError), и этап eda упал бы на реальных данных."""
    from munnet.eda import report, stories, synthesis
    from munnet.eda import s1_coverage as s1
    from munnet.eda import s2_level as s2
    from munnet.eda import s3_rhythm as s3
    from munnet.eda import s4_basket as s4
    from munnet.eda import s5_place as s5

    cfg = load_config()
    eda = cfg["eda"]
    for section in ("coverage", "level", "basket", "place", "stories", "synthesis", "report_render"):
        assert isinstance(eda[section], dict) and eda[section], section
    assert set(s1.coverage_params(cfg)) == set(s1.COVERAGE_DEFAULTS)
    assert set(s2.params(cfg)) == set(s2.DEFAULTS)
    assert s3.SeasonParams.from_config(cfg).top_n == eda["season"]["top_n"]
    assert set(s4.basket_params(cfg)) == set(s4.BASKET_DEFAULTS)
    assert set(s5.params(cfg)) == set(s5.PLACE_DEFAULTS)
    assert set(eda["stories"]) <= set(stories.params(cfg))
    # пороги балла сигнала в конфиге — ровно те, что используют сюжеты (лишний ключ молча не действует)
    assert set(eda["stories"]["signal_bins"]) == set(stories.DEFAULTS["signal_bins"])
    assert set(stories.DEFAULTS["rejection"]) <= set(eda["rejection"])  # пороги отказа — в одной секции
    assert synthesis.params(cfg)["headline"]["level"] == f"log_level_{eda['reference_year']}"
    assert set(report.params(cfg)) == set(report.DEFAULTS)
    assert all(isinstance(k, int) and isinstance(v, str) and v for k, v in eda["annotations"].items())


def test_context_tolerances():
    context = load_config()["context"]
    assert 0 < context["age"]["sum_tolerance"] < 0.1
    assert 0 < context["unallocated_tolerance"] < 0.1


def test_context_indicator_specs():
    context = load_config()["context"]
    total = context["okved_total"]
    for key in ("payroll", "wage", "retail", "catering_turnover"):
        assert context["indicators"][key]["dims"]["okved2"] == total, key
    for spec in context["indicators"].values():
        assert spec["kind"] in ("stock", "flow", "mean")
        assert not any(
            p.startswith(("Январь-март", "Январь-июнь", "Январь-сентябрь")) for p in spec["periods"]
        )
    assert context["ndfl"]["income"]["indicator_code"] == "Y777000006"
    assert set(context["okved_letters"].values()) == {"A", "B", "E", "H"}
    letters = [x for group in context["okved_groups"].values() for x in group]
    assert len(letters) == len(set(letters))
    assert set(letters) <= set(contracts.OKVED_SECTIONS)
