"""Этап interpret целиком на синтетическом конвейере (features → network → cluster → evaluate → dynamics):
прочитан каждый ключ предрегистрации, выходы и отчёт собраны, слепой прогон пишет только в свой каталог
с пометкой, свежесть входов останавливает этап. Без сети и без data/."""

from __future__ import annotations

import collections
import copy
import json
import os
import time
from pathlib import Path

import pandas as pd
import pytest
from synth import make_config, make_processed
from test_cluster_stage import light_cluster
from test_network_stage import light, write_connection

from munnet import clustering, dynamics, features, icvi_stage, interpret, network
from munnet.config import Config
from munnet.contracts import MissingInputError, QCError


def _quiet(fn, cfg):
    """Этапы до interpret на синтетике: утверждения их отчётов написаны под реальные данные (QCError отчёта —
    после записи выходов), поэтому ошибка отчёта здесь не мешает."""
    try:
        fn(cfg)
    except QCError:
        pass


def light_interpret(cfg: Config, root: Path) -> Config:
    d = copy.deepcopy(cfg.data)
    it = d["interpret"]
    fin = pd.read_parquet(Path(d["paths"]["processed"]) / "cluster_final.parquet")
    sizes = collections.Counter(fin["type"].astype(int))
    it["ladder"]["sizes"] = {int(k): int(v) for k, v in sizes.items()}
    it["ladder"]["order"] = sorted(sizes)
    t = it["tests"]
    for key in ("overall", "conditional"):
        t["T1_ladder_external"][key]["permutations"] = 19
    t["T1_ladder_external"]["bootstrap"] = 19
    t["T1_ladder_external"]["regions_up"]["permutations"] = 9
    t["T5_trivial"]["stratified"].update({"permutations": 19, "bootstrap": 19})
    it["strata"]["features"] = {"log_pop_rel": 2, "urban_share_rel": 2}
    t["T3_reliable_placebo"]["pseudo_pairs"] = 2
    t["T2_direction"].update({"bootstrap": 19, "min_pool": 5})
    t["T2_direction"]["place_tree"].update({"permutations": 19, "folds": 3})
    t["T4_basket_vs_place"].update({"bootstrap": 2, "n_init": 3})
    t["T6_bank_coverage"].update({"permutations": 19, "cliff_bootstrap": 19})
    t["T7_utility"].update({"bootstrap": 19, "k": 4, "min_set": 2})
    t["T7_utility"]["sets"]["C"]["draws"] = 3
    t["T7_utility"]["beyond_place"].update({"folds": 3, "repeats": 2, "permutations": 9})
    it["profile"]["bootstrap"] = 9
    it["tree"]["folds"] = 3
    it["tree"]["fca"]["bootstrap"] = 9
    it["controls"]["random"]["draws"] = 3
    it["robustness"]["seeds"] = [42, 142]
    it["robustness"]["seeds_required"] = 1
    it["examples"]["flows"]["min_nodes"] = 1
    it["report"] = str(root / "docs" / "interpretation.md")
    it["report_images"] = str(root / "docs" / "img" / "interpret")
    d["clustering"]["impl"]["workers"] = 1
    return Config(data=d, path=cfg.path)


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    root = tmp_path_factory.mktemp("interpret")
    make_processed(root)
    cfg = make_config(root)
    features.run(cfg)
    write_connection(cfg)
    cfg = light(cfg, root)
    d = copy.deepcopy(cfg.data)
    d["network"]["selection"]["candidates"] = ["basket_dist"]  # итог — сеть корзины, как в данных
    network.run(Config(data=d, path=cfg.path))
    cfg = light_cluster(Config(data=d, path=cfg.path))
    d = copy.deepcopy(cfg.data)
    d["clustering"]["n_clusters"] = [4, 4]
    d["clustering"]["feasible"]["k_range"] = [4, 4]
    d["clustering"]["families"]["attributed"] = ["hybrid"]
    d["clustering"]["families"]["fusion"] = []
    d["clustering"]["final_eligible"] = ["attributed"]
    d["icvi"].update(
        {
            "report": str(root / "docs" / "icvi.md"),
            "report_images": str(root / "docs" / "img"),
            "bootstrap": 3,
        }
    )
    d["dynamics"].update(
        {"report": str(root / "docs" / "dynamics.md"), "report_images": str(root / "docs" / "img")}
    )
    d["dynamics"]["tracking"]["bootstrap"] = 20
    cfg = Config(data=d, path=cfg.path)
    _quiet(clustering.run, cfg)
    _quiet(icvi_stage.run, cfg)
    _quiet(dynamics.run, cfg)
    cfg = light_interpret(cfg, root)
    interpret.run(cfg)
    return cfg, root


def test_stage_reads_every_key_and_writes_outputs(pipeline):
    cfg, root = pipeline  # run() завершился: все листья блока interpret прочитаны (иначе QCError)
    out = Path(cfg["paths"]["outputs"]) / "interpret"
    facts = json.loads((out / "facts.json").read_text(encoding="utf-8"))
    for key in (
        "T1_ladder_external",
        "T2_direction",
        "T3_reliable_placebo",
        "T5_trivial",
        "T6_bank_coverage",
        "T7_utility",
    ):
        assert facts["verdicts_main"][key]
    assert facts["label"] == "" and facts["blind"] is None
    th = facts["thesis"]
    assert th["question"] and len(th["point_1"]) == 2 and len(th["point_2"]) == 2 and th["caveat"]
    report = Path(cfg["interpret"]["report"]).read_text(encoding="utf-8")
    assert "СЛЕПОЙ ПРОГОН" not in report and "## 13. Уточнения реализации" in report
    # пояснения после вскрытия: числа — в facts.json, хеш исходов — от тех же verdicts_final и thesis
    from munnet.interpret import post_unsealing as PU

    pu = facts["post_unsealing"]
    assert pu["outcomes_sha256"] == PU.outcomes_sha256(facts)
    assert {"t3_runs", "t2_relative", "t5_level", "t7_region"} <= set(pu)
    assert "## 14. Журнал после вскрытия" in report and "Пояснение после вскрытия" in report
    for name in ("controls.csv", "t1_runs.csv", "r1_runs.csv", "profile.csv", "examples.csv", "types.csv"):
        assert (out / name).exists()
    # сверка «цепочка окон против прямого сопоставления» опубликована для основного прогона
    cd = facts["qc"]["chain_direct"]["main"]
    assert cd["last_window_differ"] == 0 and "chain_direct_nodes_differ" in cd
    # прогон завершён: метка состояния, sha входов, эволюционный способ пройден заново и повторил dynamics
    assert facts["status"] == "done"
    assert {"cluster_labels", "cluster_final", "dynamics_nodes", "dynamics_labels"} <= set(
        facts["qc"]["inputs_sha256"]
    )
    evo = facts["qc"]["refit_evolutionary"]
    assert len(evo) == 6 and min(evo.values()) >= float(cfg["dynamics"]["impl"]["final_ari_min"])
    assert len(facts["qc"]["refit_other"]) >= 4
    assert "Сверка сравнения" in report
    # пакет слепой проверки названий — без ключа и без хеша ключа; ключ — вне каталога выходов
    pkg = json.loads((out / "naming_test.json").read_text(encoding="utf-8"))
    assert set(pkg) == {"names", "profiles", "package_sha"}
    assert not (out / "naming_test_key.json").exists()
    key_path = Path(cfg["paths"]["interim"]) / "interpret" / "naming_test_key.json"
    assert sorted(json.loads(key_path.read_text(encoding="utf-8")).values()) == sorted(pkg["names"])
    assert "`outputs/interpret/facts.json`" in report
    # охват: без типа — МО без типа ни сами, ни через узел-город
    sc = facts["scope"]
    assert sc["n_partial"] == sc["n_untyped"]
    assert (
        sc["n_partial"]
        == sc["n_partial_series"] - sc["n_partial_typed_via_city"] + sc["n_untyped_full_series"]
    )
    assert f" {sc['n_partial']} МО" in sc["phrase"]
    # поузловые выгрузки для лендинга: описание, не проверка — есть, в вердикты не входят, подписаны в отчёте
    nt = facts["node_tables"]
    assert nt["note"] == "описание, не проверка"
    for name in ("node_comparable.csv", "node_r1.csv", "node_seed.csv", "node_margin.csv", "node_rival.csv"):
        assert name in nt["files"] and (out / name).exists(), name
    assert "описание, не проверка" in report and "node_r1.csv" in report
    r1 = pd.read_csv(out / "node_r1.csv")
    rob = cfg["interpret"]["robustness"]
    expect = {f"variant:{v}" for v in rob["variants"]} | {
        f"seed:{s}" for s in rob["seeds"] if int(s) != int(cfg["seed"])
    }
    assert set(r1["variant"]) == expect
    ty = pd.read_csv(out / "types.csv")
    mg = pd.read_csv(out / "node_margin.csv")
    assert len(mg) == len(ty) and set(mg["territory_id"]) == set(ty["territory_id"])
    cmp_ = pd.read_csv(out / "node_comparable.csv")
    assert set(cmp_["set"]) <= {"A", "D", "B"} and "B" in set(cmp_["set"])
    assert set(cmp_.loc[cmp_["product"], "set"]) == {facts["t7"]["product"]}


def test_blind_run_writes_only_its_own_directory(pipeline):
    cfg, root = pipeline
    report = Path(cfg["interpret"]["report"])
    before = report.stat().st_mtime
    real = json.loads(
        (Path(cfg["paths"]["outputs"]) / "interpret" / "facts.json").read_text(encoding="utf-8")
    )
    interpret.run(cfg, blind=7)
    blind_dir = Path(cfg["paths"]["outputs"]) / "interpret_blind"
    text = (blind_dir / "interpretation_blind.md").read_text(encoding="utf-8")
    assert text.startswith("> **СЛЕПОЙ ПРОГОН.**")
    facts = json.loads((blind_dir / "facts.json").read_text(encoding="utf-8"))
    assert facts["blind"] == 7 and facts["label"] == "СЛЕПОЙ ПРОГОН"
    # сверка «цепочка против прямого» в вариантах и seed идёт на метках в исходном порядке узлов:
    # слепой прогон показывает те же числа, что настоящий (а не следствие перемешивания)
    runs_cd = {r for r in real["qc"]["chain_direct"] if r != "main"}
    assert runs_cd and runs_cd == {r for r in facts["qc"]["chain_direct"] if r != "main"}
    for r in runs_cd:
        assert facts["qc"]["chain_direct"][r] == real["qc"]["chain_direct"][r], r
    # эволюционный способ в слепом прогоне тоже проходится на исходном порядке и повторяет dynamics
    assert facts["qc"]["refit_evolutionary"] == real["qc"]["refit_evolutionary"]
    # пути в слепом отчёте — свои, наблюдаемые переходы помечены как настоящие
    assert "outputs/interpret/" not in text and "`outputs/interpret_blind/facts.json`" in text
    assert "seen_before" in text.split("# ", 1)[0]  # в пометке до заголовка
    assert report.stat().st_mtime == before  # настоящий отчёт не тронут


def test_blind_seed_must_differ_from_seed(pipeline):
    cfg, _ = pipeline
    with pytest.raises(ValueError):
        interpret.run(cfg, blind=int(cfg["seed"]))


def test_freshness_stops_stage(pipeline):
    cfg, _ = pipeline
    labels = Path(cfg["paths"]["processed"]) / "cluster_labels.parquet"
    old = labels.stat().st_mtime
    new = time.time() + 60
    os.utime(labels, (new, new))
    try:
        with pytest.raises(MissingInputError, match="freshness"):
            interpret.run(cfg)
    finally:
        os.utime(labels, (old, old))
    # оборванный прогон помечен: старым выходам рядом верить нельзя
    facts = json.loads(
        (Path(cfg["paths"]["outputs"]) / "interpret" / "facts.json").read_text(encoding="utf-8")
    )
    assert facts["status"] == "failed" and "freshness" in facts["error"]


def test_cache_follows_content_not_mtime_or_workers(pipeline):
    """Перезапуск evaluate (новее время файлов) и другое число процессов кэш тяжёлых шагов не сбрасывают:
    ключ — содержимое входов и код, а не время файлов."""
    cfg, _ = pipeline
    out = Path(cfg["paths"]["outputs"]) / "interpret"
    cache = out / "cache"
    before = {p.name for p in cache.glob("*.pkl")}
    assert before
    new = time.time() + 5
    for p in (Path(cfg["paths"]["outputs"]) / "evaluate").glob("*"):
        if p.is_file():
            os.utime(p, (new, new))
    d = copy.deepcopy(cfg.data)
    d["clustering"]["impl"]["workers"] = 2
    interpret.run(Config(data=d, path=cfg.path))
    assert {p.name for p in cache.glob("*.pkl")} == before
    assert json.loads((out / "facts.json").read_text(encoding="utf-8"))["status"] == "done"


def test_naming_answer_accepted_keeps_all_keys_read_and_stale_is_not_applied(pipeline):
    """Ответ check-ux принят (верно не меньше pass_min): этап доходит до конца (naming.test.fallback прочитан
    и при названиях по правилу), в выходах — названия по правилу; ответ к другому пакету не применяется."""
    cfg, _ = pipeline
    out = Path(cfg["paths"]["outputs"]) / "interpret"
    report = Path(cfg["interpret"]["report"])
    answer = report.with_name(report.stem + "_naming_test.json")
    pkg = json.loads((out / "naming_test.json").read_text(encoding="utf-8"))
    pass_min = int(cfg["interpret"]["naming"]["test"]["pass_min"])
    try:
        answer.write_text(
            json.dumps({"correct": pass_min, "package_sha": pkg["package_sha"]}), encoding="utf-8"
        )
        interpret.run(cfg)
        facts = json.loads((out / "facts.json").read_text(encoding="utf-8"))
        assert facts["naming_test"]["state"] == "accepted"
        assert facts["names_final"] == {t: v["name"] for t, v in facts["names"].items()}
        types = pd.read_csv(out / "types.csv")
        assert {"name", "name_descriptive"} <= set(types.columns)
        answer.write_text(json.dumps({"correct": pass_min, "package_sha": "устаревший"}), encoding="utf-8")
        interpret.run(cfg)
        facts = json.loads((out / "facts.json").read_text(encoding="utf-8"))
        assert facts["naming_test"]["state"] == "stale" and not facts["naming_test"]["done"]
        assert facts["names_final"] == facts["names_descriptive"]
        assert "не применён" in report.read_text(encoding="utf-8")
    finally:
        answer.unlink(missing_ok=True)
