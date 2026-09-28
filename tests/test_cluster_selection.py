"""Правило выбора этапа cluster на игрушечных таблицах с известным ответом."""

import numpy as np
import pandas as pd
import pytest

from munnet.clustering import selection as S

DIR = {"quality_features": "max", "quality_graph": "max", "stability": "max", "interpretability": "max"}
CRIT = list(DIR)
ZERO = dict.fromkeys(CRIT, 0.0)


def meta(index, stability=None, simplicity=None, k=None):
    n = len(index)
    return pd.DataFrame(
        {
            "stability": stability if stability is not None else [0.5] * n,
            "simplicity": simplicity if simplicity is not None else [1] * n,
            "k": k if k is not None else [5] * n,
        },
        index=index,
    )


def test_rank01_best_is_one_worst_zero_ties_average():
    r = S.rank01(pd.Series([3.0, 1.0, 2.0, 2.0]))
    assert r.tolist() == [1.0, 0.0, 0.5, 0.5]
    assert S.rank01(pd.Series([3.0, 1.0]), "min").tolist() == [0.0, 1.0]
    assert S.rank01(pd.Series([7.0])).tolist() == [1.0]


def test_quality_rank_is_mean_rank_over_metrics_within_set():
    z = pd.DataFrame(
        {"sw": [1.0, 2.0, 3.0], "ch": [3.0, 2.0, 1.0], "s_dbw": [3.0, 2.0, 1.0]}, index=list("abc")
    )
    q = S.quality_rank(z)
    assert q["a"] == pytest.approx((0 + 1 + 1) / 3)
    assert q["c"] == pytest.approx((1 + 0 + 0) / 3)
    # ранг зависит от набора: без «c» кандидат «b» — худший по sw
    assert S.quality_rank(z.loc[["a", "b"]])["b"] == pytest.approx((1 + 0 + 0) / 3)


def test_pareto_front_with_tolerance():
    crit = pd.DataFrame(
        {
            "quality_features": [0.9, 0.5, 0.89],
            "quality_graph": [0.5, 0.9, 0.5],
            "stability": [0.8, 0.8, 0.7],
            "interpretability": [0.3, 0.3, 0.3],
        },
        index=["a", "b", "c"],
    )
    # без допуска c доминируется a (хуже по двум критериям)
    assert S.pareto_front(crit, DIR, ZERO) == ["a", "b"]
    # с допуском 0,02 по устойчивости c всё равно хуже на 0,1 — доминируется
    tol = {**ZERO, "stability": 0.2, "quality_features": 0.05}
    # с широкими допусками a не лучше c ни по одному критерию сверх допуска — c недоминируем
    assert "c" in S.pareto_front(crit, DIR, tol)


def test_pair_wins_respects_direction_and_tolerance():
    a = pd.Series({"x": 1.0, "y": 0.0})
    b = pd.Series({"x": 0.5, "y": 0.01})
    assert S.pair_wins(a, b, {"x": 0.0, "y": 0.0}) == (1, 1)
    assert S.pair_wins(a, b, {"x": 0.0, "y": 0.02}) == (1, 0)


def test_copeland_scores_and_winner():
    # a обходит b и c; b обходит c
    crit = pd.DataFrame(
        {
            "quality_features": [3, 2, 1],
            "quality_graph": [3, 2, 1],
            "stability": [1, 2, 3],
            "interpretability": [3, 2, 1],
        },
        index=["a", "b", "c"],
        dtype=float,
    )
    s = S.copeland_scores(crit, DIR, ZERO)
    assert s.to_dict() == {"a": 2, "b": 0, "c": -2}
    ch = S.choose(crit, meta(crit.index), DIR, ZERO)
    assert ch.winner == "a" and ch.front == ["a", "b", "c"]


def test_tie_break_chain_stability_then_simplicity_then_k():
    crit = pd.DataFrame({"quality_features": [1.0, 0.0], "quality_graph": [0.0, 1.0]}, index=["a", "b"])
    d = {"quality_features": "max", "quality_graph": "max"}
    z = {"quality_features": 0.0, "quality_graph": 0.0}
    assert S.choose(crit, meta(crit.index, stability=[0.4, 0.6]), d, z).winner == "b"
    assert S.choose(crit, meta(crit.index, simplicity=[2, 1]), d, z).winner == "b"
    assert S.choose(crit, meta(crit.index, k=[4, 7]), d, z).winner == "a"
    ch = S.choose(crit, meta(crit.index, k=[9, 3]), d, z)
    assert ch.winner == "b" and ch.tied == ["a", "b"]


def test_borda_lexicographic_and_copeland_all_can_differ():
    # a: лучший по первому критерию; b: второй везде; c: лучший по двум, худший по двум
    crit = pd.DataFrame(
        {
            "quality_features": [0.9, 0.8, 0.1],
            "quality_graph": [0.1, 0.8, 0.9],
            "stability": [0.1, 0.8, 0.9],
            "interpretability": [0.9, 0.95, 0.1],
        },
        index=["a", "b", "c"],
    )
    m = meta(crit.index, stability=crit["stability"].tolist())
    assert S.choose(crit, m, DIR, ZERO, S.AGG_BORDA).winner == "b"
    lex = S.choose(crit, m, DIR, ZERO, S.AGG_LEX, order=CRIT)
    assert lex.winner == "a"
    lex2 = S.choose(crit, m, DIR, ZERO, S.AGG_LEX, order=["stability", *CRIT[:2], CRIT[3]])
    assert lex2.winner == "c"
    assert len(S.all_orders(CRIT)) == 24


def test_lexicographic_ties_within_tolerance_pass_to_next_criterion():
    crit = pd.DataFrame({"stability": [0.80, 0.79], "interpretability": [0.1, 0.5]}, index=["a", "b"])
    d = {"stability": "max", "interpretability": "max"}
    assert S.lexicographic_set(crit, ["stability", "interpretability"], d, {"stability": 0.0}) == ["a"]
    assert S.lexicographic_set(crit, ["stability", "interpretability"], d, {"stability": 0.02}) == ["b"]


def _cands():
    rows = []
    # метод m1: K=3 лучше по всему; m2: K=4 лучше; m3 недопустим
    for method, k, q, st, it, feas in [
        ("m1", 3, 2.0, 0.9, 0.5, True),
        ("m1", 4, 1.0, 0.8, 0.4, True),
        ("m2", 3, 0.5, 0.6, 0.3, True),
        ("m2", 4, 3.0, 0.7, 0.45, True),
        ("m3", 3, 9.0, 0.99, 0.99, False),
    ]:
        rows.append(
            {
                "cand": f"{method}_k{k}",
                "method": method,
                "k": k,
                "feasible": feas,
                "simplicity": 1,
                "stability": st,
                "interpretability": it,
                "z_sw": q,
                "z_avi": q,
            }
        )
    return pd.DataFrame(rows).set_index("cand")


def test_two_level_picks_k_within_method_then_method():
    c = _cands()
    res = S.two_level(c, ["m1", "m2", "m3"], CRIT, ["z_sw"], ["z_avi"], DIR, ZERO)
    assert res.level1["m1"].winner == "m1_k3"
    assert res.level1["m2"].winner == "m2_k4"
    assert "m3" not in res.level1  # нет допустимых
    # на втором уровне ранги качества пересчитаны среди двух победителей: m2_k4 лучше по качеству (1 против
    # 0),
    # m1_k3 — по устойчивости: по 2 критерия у каждого -> ничья по очкам -> выше устойчивость
    assert res.level2.scores.to_dict() == {"m1_k3": 0, "m2_k4": 0}
    assert res.winner == "m1_k3"


def test_one_step_joint_over_all_pairs():
    c = _cands()
    ch = S.one_step(c, ["m1", "m2"], CRIT, ["z_sw"], ["z_avi"], DIR, ZERO)
    assert set(ch.criteria.index) == {"m1_k3", "m1_k4", "m2_k3", "m2_k4"}
    assert ch.winner in ch.front


def test_empty_set_raises():
    with pytest.raises(ValueError):
        S.choose(pd.DataFrame(columns=CRIT), meta([]), DIR, ZERO)


def test_nan_criterion_counts_as_tie():
    crit = pd.DataFrame({"quality_features": [np.nan, 0.5], "stability": [0.9, 0.1]}, index=["a", "b"])
    d = {"quality_features": "max", "stability": "max"}
    s = S.copeland_scores(crit, d, {"quality_features": 0.0, "stability": 0.0})
    assert s.to_dict() == {"a": 1, "b": -1}


def test_quality_rank_metric_groups_and_nan_policy():
    z = pd.DataFrame(
        {"avi": [1.0, 2.0, 3.0], "mq": [1.0, 2.0, 3.0], "avu": [3.0, 2.0, np.nan]}, index=list("abc")
    )
    # по одной: у «c» AVU не определена и в среднее не входит
    assert S.quality_rank(z)["c"] == pytest.approx(1.0)
    assert S.quality_rank(z)["a"] == pytest.approx((0 + 0 + 1) / 3)
    # AVI и MQ — одна метрика: их ранги усредняются до среднего с AVU
    assert S.quality_rank(z, [("avi", "mq"), "avu"])["a"] == pytest.approx((0 + 1) / 2)
    # неопределённая метрика — худший ранг
    assert S.quality_rank(z, nan_worst=True)["c"] == pytest.approx((1 + 1 + 0) / 3)
    assert S.n_defined(z, ["avi", "mq", "avu"]).tolist() == [3, 3, 2]
    assert S.n_defined(z, [("avi", "mq"), "avu"]).tolist() == [2, 2, 1]


def test_tie_break_with_stability_tolerance_goes_to_smaller_k():
    crit = pd.DataFrame({"quality_features": [1.0, 0.0], "quality_graph": [0.0, 1.0]}, index=["a", "b"])
    d = {"quality_features": "max", "quality_graph": "max"}
    z = {"quality_features": 0.0, "quality_graph": 0.0}
    m = meta(crit.index, stability=[0.901, 0.909], k=[3, 4])
    assert S.choose(crit, m, d, z).winner == "b"  # без допуска — выше устойчивость
    assert S.choose(crit, m, d, z, stab_tol=0.02).winner == "a"  # с допуском — равны, меньше K
