"""Этап usefulness (разведка после вскрытия): расчёты на синтетике с известным ответом и сквозной прогон
на синтетических выходах interpret, включая отказы QC. Настоящие выходы не читаются."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from munnet import usefulness as U
from munnet.config import Config, load_config
from munnet.contracts import MissingInputError, QCError

# сквозные тесты usefulness.run на синтетике: справка usefulness_level (входы кластеризации) — заглушка
pytestmark = pytest.mark.usefixtures("stub_usefulness_level")

WORDS = {
    "more_often": "чаще, чем нет",
    "about_half": "примерно в половине случаев",
    "less_often": "реже, чем нет",
}


def test_config_block_matches_implementation():
    cfg = load_config()
    U.check_spec(cfg["usefulness"])
    assert tuple(cfg["interpret"]["windows"]["change_years"]) == U.YEARS


def test_check_spec_rejects_other_rule():
    block = load_config()["usefulness"]
    bad = json.loads(json.dumps(block))
    bad["rule_share"]["against"] = "C"
    with pytest.raises(QCError, match="rule_share.against"):
        U.check_spec(bad)


def test_target_change_is_log1p_difference():
    ctx = pd.DataFrame(
        {
            "territory_id": [1, 1, 2, 2, 3],
            "year": [2023, 2024, 2023, 2024, 2023],
            "retail_pc": [9.0, 19.0, 1.0, 3.0, 5.0],
        }
    )
    y = U.target_change(ctx)
    assert y.loc[1] == pytest.approx(np.log(20) - np.log(10))
    assert y.loc[2] == pytest.approx(np.log(4) - np.log(2))
    assert np.isnan(y.loc[3])


def test_recompute_errors_median_of_known_members():
    y = pd.Series({1: 0.0, 2: 0.1, 3: 0.3, 4: np.nan, 5: 0.5})
    comp = pd.DataFrame(
        {
            "territory_id": [1, 1, 1, 1, 2],
            "set": ["B", "B", "B", "D", "B"],
            "other_id": [2, 3, 4, 5, 1],
        }
    )
    eb = U.recompute_errors(y, comp, np.array([1, 2]), "B")
    # у МО 1 член 4 без цели выпадает: медиана (0,1; 0,3) = 0,2
    assert eb == pytest.approx([0.2, 0.1])
    assert U.recompute_errors(y, comp, np.array([1]), "D") == pytest.approx([0.5])


def test_share_words_by_interval():
    assert U.share_words([0.51, 0.7], WORDS) == "чаще, чем нет"
    assert U.share_words([0.5, 0.7], WORDS) == "примерно в половине случаев"
    assert U.share_words([0.3, 0.49], WORDS) == "реже, чем нет"


def test_pick_example_median_and_tie():
    ids = np.array([10, 4, 7, 2, 9])
    d = np.array([-0.3, 0.1, -0.1, 0.1, 0.5])  # медиана 0,1 — у МО 4 и 2, берётся меньший id
    assert U.pick_example(ids, d) == 2


def test_group_bootstrap_keeps_groups_whole():
    groups = np.array([1, 1, 2, 2, 3, 3])
    vals = np.array([0, 0, 1, 1, 1, 1], dtype=float)
    draws = U.group_bootstrap(groups, {"m": lambda i: vals[i].mean(), "n": lambda i: len(i)}, 300, 0)
    # группы целиком: число МО в выборке всегда 6, доля — из {0, 1/3, 2/3, 1}
    assert set(draws["n"]) == {6.0}
    assert set(np.round(draws["m"] * 3).astype(int)) <= {0, 1, 2, 3}
    assert U.ci(draws["m"], 0.95)[0] <= 2 / 3 <= U.ci(draws["m"], 0.95)[1]


def test_type_flags_levels_and_majority():
    seed = pd.DataFrame(
        {
            "territory_id": [1, 1, 2, 2, 3, 3, 4, 4, 99, 99],
            "kind": ["seed", "variant"] * 5,
            "n_same": [4, 3, 4, 2, 3, 3, 4, 2, 1, 1],
            "n_runs": [4, 3, 4, 3, 4, 3, 4, 2, 1, 1],
        }
    )
    f = U.type_flags(seed, np.array([1, 2, 3, 4]), 3, 4).set_index("territory_id")
    # 4: тот же тип везде, где узел есть, но вариантов 2 из 3 (узел-город) — не «устойчив»
    assert f["flag"].tolist() == ["stable", "depends", "depends", "partial_check"]
    assert f["stable"].tolist() == [True, False, False, False]
    assert f["majority"].tolist() == [True, True, False, True]  # 3: не все повторы seed
    assert U.flag_text(f.loc[4].to_dict() | {"flag": "partial_check"}, 3) == "проверен в 2 из 3 вариантов"
    with pytest.raises(QCError):
        U.type_flags(seed, np.array([1, 5]), 3, 4)
    with pytest.raises(QCError, match="больше"):
        U.type_flags(seed, np.array([1, 2]), 2, 4)


def test_gain_words_by_interval():
    w = {"b_better": "b", "no_difference": "0", "d_better": "d"}
    assert U.gain_words([-0.02, -0.001], w) == "b"
    assert U.gain_words([-0.02, 0.001], w) == "0"
    assert U.gain_words([0.001, 0.02], w) == "d"


def test_region_random_errors_uses_own_group_without_self():
    # группа 1: МО 1–4 с y 0, 1, 2, 3; группа 2: МО 5 (y 100) — в набор R МО 1 не попадает никогда
    y = pd.Series({1: 0.0, 2: 1.0, 3: 2.0, 4: 3.0, 5: 100.0, 6: np.nan})
    g = pd.Series({1: 1, 2: 1, 3: 1, 4: 1, 5: 2, 6: 1})
    er = U.region_random_errors(y, np.array([1, 2]), np.array([3, 3]), g, 50, 0)
    # у МО 1 набор из 3 — все остальные МО группы с целью (2, 3, 4): медиана 2, ошибка 2
    assert er[0] == pytest.approx(2.0)
    # у МО 2 — МО 1, 3, 4: медиана 2, ошибка 1
    assert er[1] == pytest.approx(1.0)
    with pytest.raises(QCError):
        U.region_random_errors(y, np.array([5]), np.array([3]), g, 5, 0)


# --- сквозной прогон на синтетике ---------------------------------------------------------------------------


def _synthetic(tmp: Path, rng: np.random.Generator, n: int = 60, break_errors: bool = False) -> tuple:
    processed, out = tmp / "processed", tmp / "outputs"
    (out / "interpret").mkdir(parents=True)
    processed.mkdir()
    ids = np.arange(1, n + 1)
    groups = (ids - 1) // 10 + 1  # 6 групп регионов по 10 МО
    r23 = rng.uniform(50, 150, n)
    r24 = r23 * np.exp(rng.normal(0.05, 0.05, n) + 0.03 * groups)
    ctx = pd.DataFrame(
        {
            "territory_id": np.r_[ids, ids],
            "year": np.r_[np.full(n, 2023), np.full(n, 2024)],
            "retail_pc": np.r_[r23, r24],
            # новые цели usefulness.by_type_test (проверка — tests/test_usefulness_by_type.py)
            "catering_turnover_pc": np.r_[r23 * 0.2, r24 * 0.2 * np.exp(rng.normal(0, 0.1, n))],
            "ndfl_income_pc": np.r_[r23 * 3, r24 * 3 * np.exp(rng.normal(0, 0.1, n))],
            "shipments_pc": np.r_[r23 * 5, r24 * 5 * np.exp(rng.normal(0, 0.2, n))],
            "ndfl_ok": np.r_[np.ones(n, bool), ids % 13 != 0],
            # население для usefulness.size_posthoc (tests/test_usefulness_size.py); без генератора —
            # прочие числа синтетики те же
            "pop_avg": np.r_[1000.0 * ids, 1010.0 * ids],
        }
    )
    ctx.to_parquet(processed / "context_annual.parquet")
    pd.DataFrame(
        {
            "territory_id": ids,
            "name": [f"МО {i} район" for i in ids],
            "name_short": [f"МО {i}" for i in ids],
            "region_name": [f"Регион {g}" for g in groups],
            "region_group": groups,
            "mo_type": np.where(ids % 3 == 0, "go", np.where(ids % 3 == 1, "mr", "mo")),
        }
    ).to_parquet(processed / "features_nodes.parquet")
    rows = []
    for i, g in zip(ids, groups, strict=True):
        own = [j for j in ids if groups[j - 1] == g and j != i][:5]
        other = [j for j in ids if groups[j - 1] != g][(i % 7) :: 9][:5]
        rows += [(i, "B", False, r + 1, j, 0.0, "km", 10.0 * (r + 1)) for r, j in enumerate(own)]
        rows += [(i, "D", True, r + 1, j, 0.5, "basket_scaled", 500.0 + r) for r, j in enumerate(other)]
    comp = pd.DataFrame(
        rows, columns=["territory_id", "set", "product", "rank", "other_id", "distance", "unit", "km"]
    )
    comp.to_csv(out / "interpret" / "node_comparable.csv", index=False)
    y = U.target_change(ctx)
    eb = U.recompute_errors(y, comp, ids, "B")
    ed = U.recompute_errors(y, comp, ids, "D")
    ec = np.abs(rng.normal(0, 0.05, n))
    err = pd.DataFrame(
        {
            "territory_id": ids,
            "err_abs_B": eb + (1e-6 if break_errors else 0.0),
            "err_abs_C": ec,
            "err_abs_D": ed,
            "common": True,
        }
    )
    err.to_csv(out / "interpret" / "t7_errors.csv", index=False)
    types = pd.DataFrame({"territory_id": ids, "type": (ids % 4) + 1, "name": "x"})
    types.to_csv(out / "interpret" / "types.csv", index=False)
    pd.DataFrame(
        {
            "territory_id": np.repeat(ids, 2),
            "kind": ["seed", "variant"] * n,
            "n_same": np.ravel([[4, 3 if i % 3 else 1] for i in ids]),
            "n_runs": np.ravel([[4, 3] for _ in ids]),
        }
    ).to_csv(out / "interpret" / "node_seed.csv", index=False)
    rob = load_config()["interpret"]["robustness"]
    variants = [f"variant:{v}" for v in rob["variants"]]  # по алфавиту — тот же порядок, что a, b, c
    seeds = [f"seed:{s}" for s in rob["seeds"] if s != 42]
    main_type = (ids % 4) + 1
    pd.DataFrame(
        {
            "territory_id": np.r_[np.repeat(ids, 3), np.repeat(ids, 4)],
            "variant": [*(variants * n), *(seeds * n)],
            "kind": ["variant"] * (3 * n) + ["seed"] * (4 * n),
            "matched_type": np.r_[np.repeat(main_type, 3), np.repeat(main_type, 4)],
            "main_type": np.r_[np.repeat(main_type, 3), np.repeat(main_type, 4)],
            "same": np.r_[np.ravel([[True, bool(i % 3), True] for i in ids]), np.ones(4 * n, bool)],
        }
    ).to_csv(out / "interpret" / "node_r1.csv", index=False)
    facts = {
        "status": "done",
        "blind": None,
        "t7": {
            "product": "D",
            "n_common": n,
            "median_error_abs": {
                "B": float(np.median(eb)),
                "C": float(np.median(ec)),
                "D": float(np.median(ed)),
            },
            "example": {"territory_id": 3},
        },
    }
    (out / "interpret" / "facts.json").write_text(json.dumps(facts), encoding="utf-8")
    data = load_config().data
    data = json.loads(json.dumps(data, default=str))
    data["paths"] = {**data["paths"], "outputs": str(out), "processed": str(processed)}
    data["usefulness"]["rule_share"]["bootstrap"] = 200
    data["usefulness"]["rule_share"]["random_draws"] = 20
    path = tmp / "cfg.yaml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return Config(data=data, path=path), y, eb, ed


def test_run_end_to_end(tmp_path):
    cfg, y, eb, ed = _synthetic(tmp_path, np.random.default_rng(1))
    facts = U.run(cfg)
    out = Path(cfg["paths"]["outputs"]) / "usefulness"
    assert {p.name for p in out.iterdir()} == {
        "facts.json",
        "rule_by_mo.csv",
        "mo_flags.csv",
        "by_type.json",  # проверка usefulness.by_type_test — свои файлы
        "by_type_runs.csv",
        "by_type_by_mo.csv",
        "size_check.json",  # разведка usefulness.size_posthoc — свои файлы
        "size_by_mo.csv",
    }
    r = facts["rule"]
    v = r["vs_D"]
    assert v["works"] == int((eb < ed).sum())
    assert v["share"] == pytest.approx(float((eb < ed).mean()))
    assert v["share_ci"][0] <= v["share"] <= v["share_ci"][1]
    assert v["words"] in WORDS.values()
    assert r["edit"] == cfg["usefulness"]["rule_share"]["edits"][U._word_key(v["words"], WORDS)]
    assert r["gain_median"] == pytest.approx(float(np.median(eb - ed)))
    assert sum(r["delta"][k] for k in ("b_better", "within", "d_better")) == 60
    assert r["n_groups"] == 6
    assert sum(t["n"] for t in r["by_type"]) == 60
    assert set(r) >= {"vs_C", "vs_R", "gain_words", "reference"}
    table = pd.read_csv(out / "rule_by_mo.csv")
    assert (table["err_abs_R"] >= 0).all() and table["err_abs_R"].notna().all()
    ex = facts["example"]
    d = eb - ed
    assert abs(ex["d"] - np.median(d)) == pytest.approx(np.min(np.abs(d - np.median(d))))
    assert ex["own_change"] == pytest.approx(np.expm1(y.loc[ex["territory_id"]]))
    assert 0 < ex["pct_B"] <= 1 and 0 < ex["pct_D"] <= 1
    assert len(ex["members_B"]) == 5 and ex["members_B"][0]["km"] == 10.0
    assert ex["stage5_example"]["territory_id"] == 3 and "err_B" in ex["stage5_example"]
    tf = facts["type_flag"]
    assert tf["n"] == 60 and tf["stable"] == sum(1 for i in range(1, 61) if i % 3)
    assert [p["same_share"] for p in tf["per_variant"]] == pytest.approx([1.0, 40 / 60, 1.0])
    flags = pd.read_csv(out / "mo_flags.csv")
    assert (flags["flag"] == "stable").sum() == tf["stable"]
    assert set(flags["flag_text"]) <= {"тип устойчив", "тип зависит от варианта расчёта"}


def test_run_qc_fails_when_errors_do_not_recompute(tmp_path):
    cfg, *_ = _synthetic(tmp_path, np.random.default_rng(2), break_errors=True)
    with pytest.raises(QCError, match="пересчёт ошибок набора B"):
        U.run(cfg)


def test_run_missing_inputs(tmp_path):
    cfg, *_ = _synthetic(tmp_path, np.random.default_rng(3))
    (Path(cfg["paths"]["outputs"]) / "interpret" / "node_seed.csv").unlink()
    with pytest.raises(MissingInputError):
        U.run(cfg)


# --- сравнение ошибок с допуском (исправление 02.10.2026) ---------------------------------------------------


def test_compare_errors_tie_within_tolerance():
    a = np.array([0.1, 0.1, 0.1, np.nan])
    b = np.array([np.nextafter(0.1, 1.0), 0.1 + 1e-6, 0.1 - 1e-6, 0.2])
    closer, tie = U.compare_errors(a, b, 1e-12)
    assert closer.tolist() == [False, True, False, False]
    assert tie.tolist() == [True, False, False, False]
    # симметрия ничьей; «ближе» — строго за пределом допуска
    assert U.compare_errors(b[:1], a[:1], 1e-12)[1].tolist() == [True]
    assert U.compare_errors(np.array([0.0]), np.array([2e-12]), 1e-12)[0].tolist() == [True]


def test_rule_share_same_sets_different_paths_are_ties(tmp_path):
    """Набор R совпадает с B (в группе региона ровно столько МО с целью, сколько членов у B): ошибка B
    прочитана из CSV, как t7_errors.csv в _read_inputs, ошибка R посчитана в памяти. По смыслу они равны — это
    ничьи, а не победы B (старое точное «<» засчитывало победу при разнице в последнем знаке)."""
    rng = np.random.default_rng(5)
    n_groups, size = 20, 3
    ids = np.arange(1, n_groups * size + 1)
    groups = (ids - 1) // size + 1
    r23 = rng.uniform(50, 150, len(ids))
    r24 = r23 * np.exp(rng.normal(0.05, 0.05, len(ids)))
    ctx = pd.DataFrame(
        {
            "territory_id": np.r_[ids, ids],
            "year": np.r_[np.full(len(ids), 2023), np.full(len(ids), 2024)],
            "retail_pc": np.r_[r23, r24],
        }
    )
    rows = []
    for i, g in zip(ids, groups, strict=True):
        rows += [(i, "B", j) for j in ids[(groups == g) & (ids != i)]]  # B = все остальные МО группы = пул R
        rows += [(i, "D", j) for j in ids[groups != g][(i % 5) :: 11][:3]]
    comp = pd.DataFrame(rows, columns=["territory_id", "set", "other_id"])
    y = U.target_change(ctx)
    mem = pd.DataFrame(
        {
            "territory_id": ids,
            "err_abs_B": U.recompute_errors(y, comp, ids, "B"),
            "err_abs_C": np.abs(rng.normal(0, 0.05, len(ids))),
            "err_abs_D": U.recompute_errors(y, comp, ids, "D"),
            "common": True,
        }
    )
    mem.to_csv(tmp_path / "t7_errors.csv", index=False)
    errors = pd.read_csv(tmp_path / "t7_errors.csv")  # тот же путь, что у настоящего t7_errors.csv
    eb_csv = errors["err_abs_B"].to_numpy()
    # предусловие: чтение CSV меняет последний знак у части ошибок B, и часть из них стала меньше
    assert (eb_csv < mem["err_abs_B"].to_numpy()).any()
    inp = {
        "errors": errors,
        "context": ctx,
        "comparable": comp,
        "nodes": pd.DataFrame({"territory_id": ids, "region_group": groups}),
        "types": pd.DataFrame({"territory_id": ids, "type": (ids % 4) + 1}),
        "facts": {
            "t7": {
                "product": "D",
                "n_common": len(ids),
                "median_error_abs": {s: float(np.median(errors[f"err_abs_{s}"])) for s in ("B", "C", "D")},
            }
        },
    }
    block = json.loads(json.dumps(load_config()["usefulness"]))
    block["rule_share"]["bootstrap"] = 50
    block["rule_share"]["random_draws"] = 5
    res, table = U.rule_share(inp, block, 0)
    assert res["vs_R"]["ties"] == len(ids)
    assert res["vs_R"]["works"] == 0
    assert not table["works_vs_R"].any()
    # против D ничьих нет: счёт тот же, что у точного «<»
    assert res["vs_D"]["ties"] == 0
    assert res["vs_D"]["works"] == int((eb_csv < errors["err_abs_D"].to_numpy()).sum())
