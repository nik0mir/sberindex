"""Пояснения после вскрытия (``munnet.interpret.post_unsealing``): расчёты на синтетике с известным ответом,
сборка фраз не трогает вердикты и тексты исходов, модуль не входит в ключ кэша тяжёлых прогонов."""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from munnet.interpret import post_unsealing as PU
from munnet.interpret import runs as RN


def _facts() -> dict:
    return {
        "verdicts_final": {"T3_reliable_placebo": "not", "T2_direction": "partial"},
        "thesis": {"point_2": ["Текст T3.", "Текст T2."], "question": "Вопрос?"},
        "r1": {"source_run": {"T3_reliable_placebo": "variant:graph_basket_cos", "T2_direction": "main"}},
        "t2": {"main": {"n_up": 8, "n_down": 2}},
        "t6": {"delta": -0.01, "ci": [-0.15, 0.12], "n_movers": 5, "n_stayers": 20},
        "t7": {"diffs": {"D-A": [0.001, [0.0, 0.003]]}, "median_error": {"D": 0.05}},
        "scope": {"city_nodes": [{"name": "Москва", "type": 3}, {"name": "Санкт-Петербург", "type": 3}]},
    }


def _placebo() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for run, obs, pl in (
        ("main", 10, range(20)),  # 20 псевдопар 0…19: больше 10 — у 9 (45%)
        ("variant:graph_basket_cos", 8, [0] * 18 + [9, 9]),  # больше 8 — у 2 из 20 (10%)
        ("seed:142", 30, range(20)),
    ):
        rows += [{"run": run, "scheme": "main", "pair": i, "n_reliable": v} for i, v in enumerate(pl)]
        rows.append({"run": run, "scheme": "main", "pair": -1, "n_reliable": obs})
    r1 = pd.DataFrame(
        {
            "test": ["T3_reliable_placebo"] * 3,
            "run": ["main", "variant:graph_basket_cos", "seed:142"],
            "verdict": ["confirmed", "not", "confirmed"],
        }
    )
    return pd.DataFrame(rows), r1


def test_t3_runs_known_answer():
    pl, r1 = _placebo()
    got = {r["run"]: r for r in PU.t3_runs(pl, r1, 95.0)}
    assert got["main"]["n_reliable"] == 10 and got["main"]["median"] == 9.5
    assert got["main"]["share_above"] == pytest.approx(9 / 20)
    assert got["main"]["p95"] == pytest.approx(np.percentile(np.arange(20), 95))
    assert got["variant:graph_basket_cos"]["share_above"] == pytest.approx(0.1)
    assert got["variant:graph_basket_cos"]["verdict"] == "not"


def test_point2_note_names_run_and_counts_other_runs():
    pl, r1 = _placebo()
    f = _facts()
    pu = {"t3_runs": PU.t3_runs(pl, r1, 95.0)}
    note = PU.point2_note(pu, f)
    assert note.startswith("*Пояснение после вскрытия к пункту 2:*")
    assert "косинус корзин" in note and "8 надёжных" in note and "у 10% псевдопар" in note
    assert "в 1 из 1 остальных прогонов R1" in note  # seed:142 — confirmed; main и источник не считаются
    assert "вверх 8, вниз 2" in note


def test_t2_relative_known_answer():
    d = np.array([-0.1, -0.2, 0.3, 0.0, 0.4, np.nan])
    ta = np.array([2, 2, 2, 1, 2, 2])
    tb = np.array([2, 2, 1, 1, 1, 2])
    rel = np.array([False, False, True, False, True, False])
    r = PU.t2_relative(d, ta, tb, rel, bottom=2, up_to=1)
    assert r["n_stayers"] == 2 and r["median_stayers"] == pytest.approx(-0.15)
    assert r["share_stayers_down"] == 1.0
    assert r["n_movers"] == 2 and r["median_movers"] == pytest.approx(0.35)
    assert r["n_all"] == 5


def test_t5_level_identical_partition_gives_one():
    rng = np.random.default_rng(0)
    ids = np.arange(1, 201)
    step = rng.integers(1, 5, len(ids))
    types = pd.DataFrame({"territory_id": ids, "type": step + 10, "step": step})
    rival = pd.DataFrame({"territory_id": ids, "partition": PU.LEVEL_PARTITION, "best_partition_label": step})
    node_r1 = pd.DataFrame({"territory_id": ids, "variant": "seed:142", "matched_type": step})
    r = PU.t5_level(types, rival, node_r1, 50, 0.95, 42)
    assert r["ami"] == pytest.approx(1.0) and r["same_step_share"] == 1.0
    assert r["ami_runs"]["seed:142"] == pytest.approx(1.0) and r["ci"][0] <= 1.0 + 1e-12


def test_region_eta2_detects_group_effect_only_when_present():
    rng = np.random.default_rng(1)
    g = np.repeat(np.arange(10), 30)
    types = rng.integers(1, 5, len(g))
    strong = PU.region_eta2(g * 1.0 + rng.normal(0, 0.1, len(g)), g, types, 99, 42)
    none = PU.region_eta2(rng.normal(0, 1, len(g)), g, types, 99, 42)
    assert strong["eta2_region"] > 0.9 and strong["p"] == pytest.approx(0.01)
    assert none["eta2_region"] < 0.1 and none["p"] > 0.05


def test_notes_do_not_touch_outcomes_and_sha_is_canonical():
    f = _facts()
    before = copy.deepcopy(f)
    sha = PU.outcomes_sha256(f)
    pl, r1 = _placebo()
    pu = {
        "t3_runs": PU.t3_runs(pl, r1, 95.0),
        "t2_relative": PU.t2_relative(
            np.zeros(3), np.array([2] * 3), np.array([2] * 3), np.zeros(3, bool), 2, 1
        ),
        "t5_level": {
            "ami": 0.3,
            "ci": [0.29, 0.34],
            "n_boot": 10,
            "ami_runs": {},
            "same_step_share": 0.6,
            "n": 3,
        },
        "t7_region": {},
        "outcomes_sha256": sha,
    }
    for fn in (PU.t2_note, PU.t5_note):
        assert fn(pu).startswith("*Пояснение после вскрытия:*")
    for fn in (PU.t6_note, PU.t7_note, PU.city_note):
        assert fn(f).startswith("*Пояснение после вскрытия:*")
    assert "0,003" in PU.t7_note(f) and "6%" in PU.t7_note(f)
    assert "0,12" in PU.t6_note(f)
    PU.point2_note(pu, f)
    PU.t6_total_row(f)
    assert f == before  # фразы только читают facts
    # хеш не зависит от порядка ключей и совпадает у копии
    reordered = {"thesis": dict(reversed(list(f["thesis"].items()))), "verdicts_final": f["verdicts_final"]}
    assert PU.outcomes_sha256(reordered) == sha
    lines = PU.journal_lines(pu)
    assert len(lines) == len(PU.JOURNAL) + 1 and "Исправлений ошибок кода после вскрытия нет" in lines[-1]


def test_post_unsealing_not_in_heavy_cache_key():
    # правка пояснений не должна сбрасывать кэш многочасовых прогонов
    assert "munnet.interpret.post_unsealing" not in RN.heavy_closure()
    assert "munnet.interpret.report" not in RN.heavy_closure()
