import importlib
import re

from munnet.cli import STAGES, main
from munnet.config import load_config


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


def test_every_stage_has_run():
    for module_name, _ in STAGES.values():
        assert callable(importlib.import_module(module_name).run), module_name


def test_unimplemented_stage_exits_with_code_2():
    assert main(["features"]) == 2
