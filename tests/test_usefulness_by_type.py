"""Проверка usefulness.by_type_test: функции на синтетике с известным ответом, сквозной прогон
на синтетических выходах interpret во временной папке и отказы QC. Настоящие выходы и данные не читаются."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from test_usefulness import _synthetic

from munnet import usefulness as U
from munnet import usefulness_by_type as B
from munnet.config import Config, load_config
from munnet.contracts import QCError

# сквозные тесты usefulness.run на синтетике: справка usefulness_level (входы кластеризации) — заглушка
pytestmark = pytest.mark.usefixtures("stub_usefulness_level")

BLOCK = load_config()["usefulness"]["by_type_test"]
HIGH, LOW = [3, 4], [1, 2]


def _copy(x):
    return json.loads(json.dumps(x))


# --- спецификация ------------------------------------------------------------------------------------------


def test_config_block_matches_implementation():
    B.check_spec(BLOCK)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("universe", "min_set"), 4),
        (("statistic", "bootstrap", "n"), 1000),
        (("statistic", "permutation", "center"), "none"),
        (("rule", "levels"), ["not", "reverse", "confirmed"]),
        (("targets", "ndfl", "filter"), None),
        (("universe", "groups", "high"), [4]),
    ],
)
def test_check_spec_rejects_other_rule(path, value):
    bad = _copy(BLOCK)
    node = bad
    for k in path[:-1]:
        node = node[k]
    node[path[-1]] = value
    with pytest.raises(QCError, match="не совпадает"):
        B.check_spec(bad)


def test_check_spec_requires_block():
    with pytest.raises(QCError, match="нет в конфиге"):
        B.check_spec(None)


# --- цели и ошибки -----------------------------------------------------------------------------------------


def test_target_change_log1p_and_filter():
    ctx = pd.DataFrame(
        {
            "territory_id": [1, 1, 2, 2, 3, 4, 4],
            "year": [2023, 2024, 2023, 2024, 2023, 2023, 2024],
            "x": [9.0, 19.0, 1.0, 3.0, 5.0, 0.0, 1.0],
            "ok": [True, True, True, False, True, True, True],
        }
    )
    y = B.target_change(ctx, "x")
    assert y.loc[1] == pytest.approx(np.log(20) - np.log(10))
    assert y.loc[2] == pytest.approx(np.log(4) - np.log(2))
    assert np.isnan(y.loc[3])  # нет 2024 года
    assert y.loc[4] == pytest.approx(np.log(2))  # log1p(0) = 0
    yf = B.target_change(ctx, "x", "ok")
    assert yf.loc[1] == pytest.approx(y.loc[1])
    assert np.isnan(yf.loc[2])  # флаг ложен в 2024 году
    assert np.isnan(yf.loc[3])


def test_target_table_errors_sizes_and_min_set():
    y = pd.Series({1: 0.0, 2: 0.1, 3: 0.3, 4: np.nan, 5: 0.5, 6: 0.7, 7: np.nan})
    comp = pd.DataFrame(
        {
            "territory_id": [1, 1, 1, 1, 1, 2, 2, 7, 7],
            "set": ["B", "B", "B", "D", "D", "B", "D", "B", "D"],
            "other_id": [2, 3, 4, 5, 6, 1, 5, 1, 2],
        }
    )
    tab = B.target_table(y, comp, np.array([1, 2, 7]), min_set=2).set_index("territory_id")
    # МО 1: B — 2, 3 (4 без цели): медиана 0,2; D — 5, 6: медиана 0,6
    assert tab.loc[1, "size_B"] == 2 and tab.loc[1, "size_D"] == 2
    assert tab.loc[1, "err_B"] == pytest.approx(0.2) and tab.loc[1, "err_D"] == pytest.approx(0.6)
    assert bool(tab.loc[1, "in_target"]) and tab.loc[1, "drop_reason"] == ""
    assert tab.loc[2, "drop_reason"] == "min_set" and not tab.loc[2, "in_target"]
    assert tab.loc[7, "drop_reason"] == "no_target"


# --- доли, Δ, страты --------------------------------------------------------------------------------------


def test_share_diff_is_pooled_share_of_group():
    # {3,4}: тип 3 — 2 из 2, тип 4 — 0 из 6 → общая доля 2/8 (а не среднее долей типов 0,5)
    labels = np.array([3, 3, 4, 4, 4, 4, 4, 4, 1, 1, 2, 2])
    works = np.array([1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1], bool)
    mask = np.ones(12, bool)
    d = B.share_diff(B.point_sum, np.isin(labels, HIGH), np.isin(labels, LOW), works, mask)
    assert d["share_hi"] == pytest.approx(2 / 8)
    assert d["share_lo"] == pytest.approx(3 / 4)
    assert d["delta"] == pytest.approx(2 / 8 - 3 / 4)
    # маска выключает МО: без двух последних — {1,2}: 1 из 2
    mask[-2:] = False
    assert B.share_diff(B.point_sum, np.isin(labels, HIGH), np.isin(labels, LOW), works, mask)[
        "share_lo"
    ] == pytest.approx(0.5)


def test_status_delta_harmonic_weights_and_empty_stratum():
    #   ГО: {3,4} 3 МО (works 1,1,0), {1,2} 1 МО (works 0) → разность 2/3, h = 3·1/4 = 0,75
    #   остальные: {3,4} 2 МО (0,0), {1,2} 2 МО (1,0) → разность −0,5, h = 1
    labels = np.array([3, 3, 4, 1, 3, 4, 1, 2])
    works = np.array([1, 1, 0, 0, 0, 0, 1, 0], bool)
    go = np.array([1, 1, 1, 1, 0, 0, 0, 0], bool)
    mask = np.ones(8, bool)
    hi, lo = np.isin(labels, HIGH), np.isin(labels, LOW)
    got = B.status_delta(B.point_sum, hi, lo, works, mask, [go, ~go])
    assert got == pytest.approx((0.75 * (2 / 3) + 1.0 * (-0.5)) / 1.75)
    # в страте ГО нет {1,2} → её вес 0, остаётся только вторая страта
    mask2 = mask.copy()
    mask2[3] = False
    assert B.status_delta(B.point_sum, hi, lo, works, mask2, [go, ~go]) == pytest.approx(-0.5)
    # ни в одной страте нет обеих групп → NaN
    only_hi = np.array([1, 1, 1, 0, 1, 1, 0, 0], bool)
    assert np.isnan(B.status_delta(B.point_sum, hi, lo, works, only_hi, [go, ~go]))


def test_go_delta():
    works = np.array([1, 1, 0, 1, 0, 0], bool)
    is_go = np.array([1, 1, 1, 0, 0, 0], bool)
    assert B.go_delta(B.point_sum, is_go, works, np.ones(6, bool)) == pytest.approx(2 / 3 - 1 / 3)


# --- общий бутстрап и общая перестановка -----------------------------------------------------------------


def test_bootstrap_counts_whole_groups_same_as_usefulness():
    groups = np.array([5, 5, 7, 7, 7, 9, 2, 2])
    works = np.array([1, 0, 1, 1, 0, 1, 0, 0], float)
    gidx, counts = B.bootstrap_counts(groups, 300, 11)
    assert counts.shape == (300, 4) and (counts.sum(axis=1) == 4).all()  # группы целиком, их число — 4
    S = B.boot_sum(gidx, counts)
    share = S(works) / S(np.ones(8))
    # тот же розыгрыш, что у usefulness.group_bootstrap (конкатенация МО выбранных групп)
    ref = U.group_bootstrap(groups, {"m": lambda i: works[i].mean()}, 300, 11)["m"]
    assert share == pytest.approx(ref, abs=1e-15)


def test_permutation_index_within_groups():
    groups = np.array([1, 1, 1, 2, 2, 3])
    labels = np.array([1, 3, 4, 2, 3, 1])
    P = B.permutation_index(groups, 50, 0)
    assert P.shape == (50, 6)
    assert (groups[P] == groups).all()  # метка берётся только из своей группы
    for row in labels[P]:
        for g in (1, 2, 3):
            assert sorted(row[groups == g]) == sorted(labels[groups == g])
    assert np.array_equal(P, B.permutation_index(groups, 50, 0))  # свой default_rng(seed)
    assert len({tuple(r) for r in P}) > 1


def _data(works_by_target: dict, n: int, rng=None) -> dict:
    out = {}
    for t, w in works_by_target.items():
        e_b = np.where(w, 0.01, 0.02)
        out[t] = {
            "mask": np.ones(n, bool),
            "works": np.asarray(w, bool),
            "err_B": e_b,
            "err_D": np.full(n, 0.015),
            "err_R": np.full(n, 0.015),
            "y": np.linspace(-0.1, 0.1, n),
            "size_B": np.full(n, 10),
            "size_D": np.full(n, 10),
        }
    return out


def test_one_permutation_for_all_targets():
    # отгрузка = дополнение общепита на тех же МО: при общих метках Δ_перм(отгрузка) = −Δ_перм(общепит)
    # в каждой перестановке; при раздельных перестановках по целям равенство нарушилось бы
    rng = np.random.default_rng(3)
    n = 40
    groups = np.repeat(np.arange(4), 10)
    labels = rng.integers(1, 5, n)
    w = rng.random(n) < 0.5
    data = _data({"catering": w, "ndfl": w, "shipments": ~w}, n)
    P = B.permutation_index(groups, 200, 1)
    go = np.zeros(n, bool)
    nd = B.null_deltas(data, labels, P, HIGH, LOW, [go, ~go])
    assert nd["delta"]["shipments"] == pytest.approx(-nd["delta"]["catering"])
    assert nd["delta"]["ndfl"] == pytest.approx(nd["delta"]["catering"])
    # проверка по определению для первой перестановки
    lp = labels[P[0]]
    want = w[np.isin(lp, HIGH)].mean() - w[np.isin(lp, LOW)].mean()
    assert nd["delta"]["catering"][0] == pytest.approx(want)


# --- интервалы, p, вердикт ---------------------------------------------------------------------------------


def test_center_adjustment_only_stricter():
    assert B.center_adjusted([-0.10, -0.02], -0.005) == pytest.approx([-0.10, -0.015])  # верх поднят
    assert B.center_adjusted([0.02, 0.10], 0.005) == pytest.approx([0.015, 0.10])  # низ опущен
    assert B.center_adjusted([-0.1, 0.1], 0.0) == pytest.approx([-0.1, 0.1])
    for m0 in (-0.03, -0.001, 0.0, 0.002, 0.04):
        lo, hi = B.center_adjusted([-0.05, 0.03], m0)
        assert lo <= -0.05 and hi >= 0.03  # интервал только шире


def test_perm_p_formula_and_undefined():
    null = np.array([-0.3, -0.1, 0.0, 0.1, 0.2])
    p = B.perm_p(null, -0.1)
    assert p["p_less"] == pytest.approx((2 + 1) / 6)
    assert p["p_greater"] == pytest.approx((4 + 1) / 6)
    p2 = B.perm_p(np.array([-0.3, np.nan, 0.5]), -0.2)
    # неопределённая перестановка считается в пользу нуля в обоих p
    # ≤ −0,2: одна (−0,3) плюс неопределённая; ≥ −0,2: одна (0,5) плюс неопределённая
    assert p2["p_less"] == pytest.approx((2 + 1) / 4) and p2["p_greater"] == pytest.approx((2 + 1) / 4)
    assert p2["n_undefined"] == 1


def test_mde():
    assert B.mde(0.03, -0.004) == pytest.approx(2.8 * 0.03 + 0.004)
    assert B.mde(0.03, 0.004) == pytest.approx(2.8 * 0.03)


@pytest.mark.parametrize(
    ("ci_adj", "pl", "pg", "want"),
    [
        ([-0.15, -0.01], 0.01, 0.99, "confirmed"),
        ([-0.15, -0.01], 0.05, 0.95, "not"),  # p не меньше alpha
        ([-0.15, 0.0], 0.001, 0.999, "not"),  # верхняя граница не ниже нуля
        ([0.01, 0.15], 0.99, 0.01, "reverse"),
        ([0.01, 0.15], 0.99, 0.2, "not"),
        ([0.0, 0.15], 0.99, 0.001, "not"),
        ([float("nan"), float("nan")], 0.001, 0.001, "not"),
    ],
)
def test_verdict_branches(ci_adj, pl, pg, want):
    assert B.verdict(ci_adj, pl, pg, 0.05) == want


def test_noise_flags():
    med = {
        "catering": {"hi": {"B": 0.03, "D": 0.03}, "lo": {"B": 0.046, "D": 0.03}},  # lo: 0,046 > 1,5 · 0,03
        "ndfl": {"hi": {"B": 0.05, "D": 0.10}, "lo": {"B": 0.04, "D": 0.06}},  # hi: 0,10 > 1,5 · 0,06
        "shipments": {"hi": {"B": 0.03, "D": 0.03}, "lo": {"B": 0.045, "D": 0.03}},  # ровно 1,5 — не флаг
    }
    f = B.noise_flags(med, 1.5, 1.5)
    assert f == {"lo": ["catering"], "hi": ["ndfl"]}


def test_lowest_verdict_and_ties():
    levels = ["reverse", "not", "confirmed"]
    assert B.lowest_verdict(["confirmed"] * 8, levels) == 0  # равенство — основной
    v = ["confirmed", "confirmed", "not", "confirmed", "not", "confirmed", "confirmed", "confirmed"]
    assert B.lowest_verdict(v, levels) == 2  # первый наименьший по порядку прогонов
    v = ["not", "not", "not", "not", "not", "not", "reverse", "reverse"]
    assert B.lowest_verdict(v, levels) == 6


def test_run_order_from_interpret_robustness():
    cfg = load_config()
    order = B.run_order(cfg)
    rob = cfg["interpret"]["robustness"]
    assert order[0] == ("main", "main")
    assert [n for n, _ in order[1:4]] == [f"variant:{v}" for v in rob["variants"]]
    assert [n for n, _ in order[4:]] == [f"seed:{s}" for s in rob["seeds"] if s != cfg["seed"]]


# --- один прогон: заложенный эффект ------------------------------------------------------------------------


def _design(groups, labels_n, strata, seed=0, n_boot=2000, n_perm=2000):
    gidx, counts = B.bootstrap_counts(groups, n_boot, seed)
    return {
        "high": HIGH,
        "low": LOW,
        "strata": strata,
        "is_go": strata[0],
        "gidx": gidx,
        "counts": counts,
        "perm": B.permutation_index(groups, n_perm, seed),
        "level": 0.95,
        "alpha": 0.05,
        "noise_lo": 1.5,
        "noise_hi": 1.5,
        "types": [1, 2, 3, 4],
        "tie_tol": 1e-12,
    }


def _planted(p_hi: float, p_lo: float, seed: int = 0, n: int = 1200, n_groups: int = 40):
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(n_groups), n // n_groups)
    labels = rng.integers(1, 5, n)
    go = rng.random(n) < 0.3
    p = np.where(np.isin(labels, HIGH), p_hi, p_lo)
    data = _data({t: rng.random(n) < p for t in ("catering", "ndfl", "shipments", "retail")}, n)
    return data, labels, _design(groups, labels, [go, ~go])


def test_evaluate_run_planted_effect_confirmed():
    data, labels, design = _planted(0.40, 0.70)
    r = B.evaluate_run(data, labels, design)
    assert r["verdict"] == "confirmed"
    manual = np.mean(
        [
            data[t]["works"][np.isin(labels, HIGH)].mean() - data[t]["works"][np.isin(labels, LOW)].mean()
            for t in B.TARGETS
        ]
    )
    assert r["delta"] == pytest.approx(manual)  # среднее по трём целям; розница в него не входит
    assert r["delta_ci_adj"][1] < 0 and r["p_less"] < 0.05
    assert r["delta_ci_adj"][0] <= r["delta_ci"][0] and r["delta_ci_adj"][1] >= r["delta_ci"][1]
    assert r["bootstrap_kept"] + r["bootstrap_dropped"] == 2000
    assert r["mde"] == pytest.approx(2.8 * r["sd_boot"] + max(0.0, -r["null_mean"]))
    assert abs(r["null_mean"]) < 0.01  # метки не связаны с регионом — нуль у нуля
    assert r["status_adds"]  # эффект одинаков в обеих стратах


def test_evaluate_run_planted_effect_reverse_and_none():
    data, labels, design = _planted(0.75, 0.45, seed=1)
    assert B.evaluate_run(data, labels, design)["verdict"] == "reverse"
    data, labels, design = _planted(0.55, 0.55, seed=2)
    r = B.evaluate_run(data, labels, design)
    assert r["verdict"] == "not" and r["delta_ci"][0] < 0 < r["delta_ci"][1]


def test_evaluate_run_drops_bootstrap_samples_with_empty_group():
    # {3,4} — только в одной из 5 групп региона: выборки без неё отбрасываются и считаются
    n = 50
    groups = np.repeat(np.arange(5), 10)
    labels = np.where(groups == 0, 3, 1)
    go = np.zeros(n, bool)
    w = np.arange(n) % 2 == 0
    data = _data({t: w for t in ("catering", "ndfl", "shipments")}, n)
    r = B.evaluate_run(data, labels, _design(groups, labels, [go, ~go], n_boot=500, n_perm=50))
    # вероятность, что группа 0 не попала ни разу, — (4/5)^5 ≈ 0,33
    assert 100 < r["bootstrap_dropped"] < 240
    assert r["bootstrap_kept"] + r["bootstrap_dropped"] == 500


def test_null_center_from_region_composition_makes_verdict_stricter():
    # тип связан с регионом и доля «сработало» — тоже: нуль перестановок внутри регионов не сдвигается,
    # но поправка центра никогда не сужает интервал
    data, labels, design = _planted(0.45, 0.65, seed=4)
    r = B.evaluate_run(data, labels, design)
    lo, hi = r["delta_ci"]
    assert r["delta_ci_adj"] == pytest.approx(B.center_adjusted([lo, hi], r["null_mean"]))


# --- тексты ------------------------------------------------------------------------------------------------


def _run_stub(verdict, **kw):
    base = {
        "verdict": verdict,
        "delta": -0.07,
        "delta_ci_adj": [-0.13, -0.01],
        "p_less": 0.004,
        "p_greater": 0.996,
        "mde": 0.09,
        "delta_status": -0.05,
        "delta_status_ci_adj": [-0.12, 0.02],
        "status_adds": False,
        "delta_go": -0.06,
        "delta_go_ci": [-0.12, -0.004],
        "go_below": False,
        "share_hi": 0.52,
        "share_lo": 0.59,
        "share_hi_more_often": False,
        "noise_lo_targets": [],
        "noise_hi_targets": [],
    }
    return {**base, **kw}


@pytest.mark.parametrize(
    ("kw", "keys", "edit_key"),
    [
        ({"status_adds": True}, ["confirmed", "status_adds", "go_always"], "confirmed.status_adds"),
        ({}, ["confirmed", "status_same", "go_always"], "confirmed.status_same"),
        (
            {"go_below": True, "noise_hi_targets": ["ndfl"]},
            ["confirmed", "status_same", "status_same_go", "noise_caveat_hi", "go_always"],
            "confirmed.status_same",
        ),
        (
            {"status_adds": True, "share_hi_more_often": True},
            ["confirmed", "status_adds", "go_always"],
            "confirmed.share_hi_more_often",
        ),
    ],
)
def test_outcome_texts_confirmed(kw, keys, edit_key):
    t = B.outcome_texts(_run_stub("confirmed", **kw), BLOCK["words"], BLOCK["edits"])
    assert t["word_keys"] == keys and t["edit_key"] == edit_key
    for s in (t["text"], t["edit"]):
        assert "{" not in s and "}" not in s and "п. п.." not in s
    assert "p = 0,004" in t["text"] and "52,0% против 59,0%" in t["text"]
    if "noise_caveat_hi" in keys:
        assert "доход по 5-НДФЛ" in t["text"]


@pytest.mark.parametrize(
    ("verdict", "kw", "keys"),
    [
        ("not", {}, ["not", "go_always"]),
        ("not", {"go_below": True}, ["not", "not_go_below"]),
        ("not", {"noise_lo_targets": ["catering", "shipments"]}, ["not", "noise_caveat_lo", "go_always"]),
        ("reverse", {"noise_lo_targets": ["ndfl"]}, ["reverse", "noise_caveat_lo", "go_always"]),
        ("reverse", {"go_below": True}, ["reverse", "go_always"]),
    ],
)
def test_outcome_texts_not_and_reverse(verdict, kw, keys):
    run = _run_stub(verdict, p_greater=0.003, **kw)
    t = B.outcome_texts(run, BLOCK["words"], BLOCK["edits"])
    assert t["word_keys"] == keys and t["edit_key"] == verdict
    for s in (t["text"], t["edit"]):
        assert "{" not in s and "}" not in s
    if verdict == "not":
        assert "9,0 п. п." in t["text"] and "9,0 п. п." in t["edit"]  # {mde} — по модулю
    if verdict == "reverse":
        assert "p = 0,003" in t["text"]  # при reverse — p_greater
    if "noise_caveat_lo" in keys:
        assert ", ".join(B.TARGET_NAMES[x] for x in kw["noise_lo_targets"]) in t["text"]


def test_fill_rejects_unknown_field_and_leftover_brace():
    with pytest.raises(QCError):
        B.fill("разность {nope}", {"delta": "1"})
    with pytest.raises(QCError):
        B.fill("скобка {{delta}}", {"delta": "1"})
    assert B.fill("разность {delta}.", {"delta": "−1,0 п. п."}) == "разность −1,0 п. п."


def test_check_forbidden_roots():
    B.check_forbidden("Отгрузка: разность −1 п. п.", BLOCK["words"]["forbidden_per_target"])
    for bad in ("Разница Подтверждена", "это видно", "типы различаются"):
        with pytest.raises(QCError, match="запрещённые"):
            B.check_forbidden(bad, BLOCK["words"]["forbidden_per_target"])


# --- сквозной прогон на синтетических выходах interpret ----------------------------------------------------


def test_end_to_end_with_usefulness_run(tmp_path):
    cfg, *_ = _synthetic(tmp_path, np.random.default_rng(1))
    U.run(cfg)
    out = Path(cfg["paths"]["outputs"]) / "usefulness"
    f = json.loads((out / "by_type.json").read_text(encoding="utf-8"))
    names = [n for n, _ in B.run_order(cfg)]
    assert list(f["verdicts"]) == names and set(f["verdicts"].values()) <= {"confirmed", "not", "reverse"}
    levels = BLOCK["rule"]["levels"]
    low = min(names, key=lambda n: (levels.index(f["verdicts"][n]), names.index(n)))
    assert f["main_text"]["run"] == low and f["main_text"]["verdict"] == f["verdicts"][low]
    assert f["main_text"]["text"] == f["runs"][low]["texts"]["text"]
    for r in f["runs"].values():
        for s in (r["texts"]["text"], r["texts"]["edit"], *r["descriptions"].values()):
            assert "{" not in s and "}" not in s
    assert len(f["limitations"]) == len(BLOCK["limitations"])
    assert f["runs"][low]["texts"]["fields"]["mde"] in f["limitations"][6]  # «порог обнаружения {mde}»
    # QC розницы: Δ тем же кодом = по счётчикам rule.by_type этапа
    q = f["qc"]["retail"]
    assert q["abs_diff"] <= 1e-12 and q["n"] == 60
    # ndfl_ok ложен у МО 13, 26, … в 2024 году: они без цели, их соседи по региону с 4 членами — по min_set
    u = f["universe"]["ndfl"]
    assert u["no_target"] == 4 and u["n"] + u["no_target"] + u["min_set"] == 60
    assert sum(u["min_set_by_type"].values()) == u["min_set"]
    runs = pd.read_csv(out / "by_type_runs.csv")
    assert len(runs) == 8 * 4 and set(runs["target"]) == {*B.TARGETS, "retail"}
    by_mo = pd.read_csv(out / "by_type_by_mo.csv")
    assert len(by_mo) == 4 * 60
    m = by_mo["in_target"]
    assert (by_mo.loc[m, "err_R"] >= 0).all() and by_mo.loc[~m, "err_R"].isna().all()


def test_existing_outputs_unchanged(tmp_path, monkeypatch):
    """facts.json, rule_by_mo.csv и mo_flags.csv с проверкой by_type и без неё побайтно одинаковы."""
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    cfg_a, *_ = _synthetic(a, np.random.default_rng(5))
    cfg_b, *_ = _synthetic(b, np.random.default_rng(5))
    U.run(cfg_a)
    monkeypatch.setattr(B, "run", lambda cfg, rule_by_type: None)
    U.run(cfg_b)
    for name in ("facts.json", "rule_by_mo.csv", "mo_flags.csv"):
        pa = Path(cfg_a["paths"]["outputs"]) / "usefulness" / name
        pb = Path(cfg_b["paths"]["outputs"]) / "usefulness" / name
        ta, tb = pa.read_bytes(), pb.read_bytes()
        if name == "facts.json":  # пути входов в sha256 одинаковы по содержимому, сравниваем всё
            assert json.loads(ta) == json.loads(tb)
        else:
            assert ta == tb


def test_qc_retail_fails_on_other_counts(tmp_path):
    cfg, *_ = _synthetic(tmp_path, np.random.default_rng(1))
    U.run(cfg)
    facts_path = Path(cfg["paths"]["outputs"]) / "usefulness" / "facts.json"
    rule = json.loads(facts_path.read_text("utf-8"))["rule"]
    bad = _copy(rule["by_type"])
    bad[0]["works"] += 1
    with pytest.raises(QCError, match="QC розницы"):
        B.run(cfg, bad)
    assert not (Path(cfg["paths"]["outputs"]) / "usefulness" / "by_type.json").exists()  # старый файл удалён


def test_run_labels_qc(tmp_path):
    ids = np.array([1, 2, 3])
    types = pd.DataFrame({"territory_id": ids, "type": [1, 3, 4]})
    cfg = load_config()
    order = B.run_order(cfg)
    rows = [
        {"territory_id": i, "variant": n, "kind": k, "matched_type": t, "main_type": t}
        for n, k in order[1:]
        for i, t in zip(ids, [1, 3, 4], strict=True)
    ]
    r1 = pd.DataFrame(rows)
    lab = B.run_labels(ids, types, r1, order, {1, 2, 3, 4})
    assert list(lab) == [n for n, _ in order] and (lab["main"] == [1, 3, 4]).all()
    with pytest.raises(QCError, match="нет метки"):
        B.run_labels(ids, types, r1.iloc[1:], order, {1, 2, 3, 4})
    bad = r1.copy()
    bad.loc[0, "main_type"] = 2
    with pytest.raises(QCError, match="main_type"):
        B.run_labels(ids, types, bad, order, {1, 2, 3, 4})


def test_config_object_kept(tmp_path):
    # by_type.run берёт пути только из cfg.dir: выходы во временной папке
    cfg, *_ = _synthetic(tmp_path, np.random.default_rng(1))
    assert isinstance(cfg, Config)
    U.run(cfg)
    assert (tmp_path / "outputs" / "usefulness" / "by_type.json").exists()
