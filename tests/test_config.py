import importlib
import re

from munnet.cli import STAGES, main
from munnet.config import load_config


def test_default_config_sources_are_pinned():
    cfg = load_config()
    for name, source in cfg["sources"].items():
        assert re.fullmatch(r"[0-9a-f]{64}", source["sha256"]), name
        assert source["urls"], name
        assert all(url.startswith("https://") for url in source["urls"]), name


def test_every_stage_has_run():
    for module_name, _ in STAGES.values():
        assert callable(importlib.import_module(module_name).run), module_name


def test_unimplemented_stage_exits_with_code_2():
    assert main(["features"]) == 2
