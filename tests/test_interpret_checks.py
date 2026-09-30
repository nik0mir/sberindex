"""Этап interpret на синтетике с известной истиной: у каждой решающей проверки правило умеет оба исхода
(«подтвердилась» и «нет»), контроли, выбор текста исхода, свежесть входов, слепой режим."""

from __future__ import annotations

import os
import time
from collections import Counter
from itertools import product

import numpy as np
import pandas as pd
import pytest

from munnet.config import load_config
from munnet.contracts import MissingInputError, QCError
from munnet.interpret import data as D
from munnet.interpret import describe as DS
from munnet.interpret import external as EX
from munnet.interpret import order as O
from munnet.interpret import partitions as P
from munnet.interpret import placebo as PL
from munnet.interpret import stage as ST
from munnet.interpret import texts as TX
from munnet.interpret.partitions import Partition
from munnet.interpret.spec import Spec

SIZES = [120, 200, 100, 60]  # размеры ступеней снизу вверх (n = 480)


def _spec() -> Spec:
    return Spec(load_config()["interpret"])


@pytest.fixture(scope="module", autouse=True)
def _levels():
    ST._levels(_spec())


# --- Синтетика для T1 и T5 -------------------------------------------------------------------------


def _world(seed: int, signal: tuple[float, float], place_signal: float = 0.0):
    """n узлов, 8 групп региона, две страты-признака; ступени типов 1…4 с размерами SIZES; обороты = сила
    ``signal`` × ступень + ``place_signal`` × признак места + шум."""
    rng = np.random.default_rng(seed)
    n = sum(SIZES)
    ids = np.arange(1, n + 1)
    groups = rng.integers(0, 8, n)
    ranks = np.concatenate([np.full(s, i + 1) for i, s in enumerate(SIZES)])
    rng.shuffle(ranks)
    pop, urb = rng.normal(size=n), rng.normal(size=n)
    place = rng.normal(size=(n, 3))
    q1 = P.rank_quantiles(pop, ids, 2)
    q2 = P.rank_quantiles(urb, ids, 2)
    strata = q1 * 2 + q2
    ind = {}
    for name, s in zip(("catering", "retail"), signal, strict=True):
        ind[name] = s * ranks + place_signal * place[:, 0] + rng.normal(size=n)
    rivals = []
    for j in range(place.shape[1]):
        for sgn, rev in ((1, False), (-1, True)):
            lab = P.sized_groups(sgn * place[:, j], ids, SIZES)
            rivals.append(Partition(f"sized:x{j}:{'-' if rev else '+'}", f"x{j}", lab, "sized", lab))
    rivals.append(Partition("place_only", "K-means", P.kmeans_labels(place, 4, 0, 3), "place_only"))
    return ids, groups, ranks, strata, ind, rivals, place


def _evidence(ids, groups, strata, ind, rivals, n_boot=99, n_perm=99, seed=1) -> O.Evidence:
    rng = np.random.default_rng(seed)
    turn, bb, ba = {}, {}, {}
    for name, v in ind.items():
        t = O.make_turnover(name, 1, v, groups, strata)
        turn[name] = t
        bb[name], ba[name] = O.make_boots(t, n_boot, rng)
    ev = O.Evidence(turnovers=turn, boots_b=bb, boots_a=ba, rivals=[])
    ev.n_perm_t1_a = ev.n_perm_t1_b = ev.n_perm_t5 = n_perm
    O.build_rivals(ev, rivals, {"sized": "sign_of_rho", "place_only": "best_of_24"})
    return ev


def test_t1_confirmed_when_both_turnovers_follow_steps_beyond_place():
    ids, groups, ranks, strata, ind, rivals, _ = _world(0, (1.0, 1.0))
    ev = _evidence(ids, groups, strata, ind, rivals)
    r = O.t1_eval(ranks, ev, np.random.default_rng(2), 0.05, 0.10)
    assert r.verdict == "confirmed"
    assert all(t.overall and t.beyond for t in r.per.values())


def test_t1_not_when_turnovers_are_noise():
    ids, groups, ranks, strata, ind, rivals, _ = _world(1, (0.0, 0.0))
    ev = _evidence(ids, groups, strata, ind, rivals)
    r = O.t1_eval(ranks, ev, np.random.default_rng(2), 0.05, 0.10)
    assert r.verdict == "not"


def test_t1_partial_one_when_only_one_turnover_follows_steps():
    ids, groups, ranks, strata, ind, rivals, _ = _world(2, (1.0, 0.0))
    ev = _evidence(ids, groups, strata, ind, rivals)
    r = O.t1_eval(ranks, ev, np.random.default_rng(2), 0.05, 0.10)
    assert r.verdict == "partial_one"
    assert r.per["catering"].beyond and not r.per["retail"].overall


def test_t1_partial_overall_when_a_place_partition_orders_better():
    """Обороты задаёт признак места, а типы лишь слабо с ним связаны: упорядочены в целом, сверх места —
    нет."""
    ids, groups, ranks, strata, ind, rivals, place = _world(3, (0.0, 0.0), place_signal=2.0)
    ranks_place = P.sized_groups(place[:, 0] + np.random.default_rng(9).normal(0, 1.2, len(ids)), ids, SIZES)
    ev = _evidence(ids, groups, strata, ind, rivals)
    r = O.t1_eval(ranks_place, ev, np.random.default_rng(2), 0.05, 0.10)
    assert r.verdict == "partial_overall"
    assert r.per["catering"].best_rival.startswith("sized:x0")


def test_t1_random_labels_are_not_ordered():
    ids, groups, ranks, strata, ind, rivals, _ = _world(4, (1.0, 1.0))
    ev = _evidence(ids, groups, strata, ind, rivals)
    rng = np.random.default_rng(5)
    verdicts = [O.t1_eval(P.random_labels(SIZES, rng), ev, rng, 0.05, 0.10).verdict for _ in range(5)]
    assert verdicts.count("not") >= 4


def _trivial(ids, place):
    return {
        "dec": Partition("dec", "децили", P.value_quantiles(place[:, 1], 10), "trivial"),
        "x0": Partition("x0", "x0", P.sized_groups(place[:, 0], ids, SIZES), "trivial"),
    }


def test_t5_confirmed_and_not_empty():
    ids, groups, ranks, strata, ind, rivals, place = _world(6, (1.0, 1.0))
    ev = _evidence(ids, groups, strata, ind, rivals)
    r = O.t5_eval(ranks, ev, _trivial(ids, place), np.random.default_rng(3), 0.30, 0.05)
    assert r.verdict == "confirmed" and r.cond1 and r.cond3
    ids, groups, ranks, strata, ind, rivals, place = _world(7, (0.0, 0.0))
    ev = _evidence(ids, groups, strata, ind, rivals)
    r = O.t5_eval(ranks, ev, _trivial(ids, place), np.random.default_rng(3), 0.30, 0.05)
    assert r.verdict == "not_empty"


def test_t5_not_repeats_when_types_copy_a_trivial_partition():
    ids, groups, ranks, strata, ind, rivals, place = _world(8, (0.0, 0.0), place_signal=1.5)
    ev = _evidence(ids, groups, strata, ind, rivals)
    copy_x0 = P.sized_groups(place[:, 0], ids, SIZES)
    r = O.t5_eval(copy_x0, ev, _trivial(ids, place), np.random.default_rng(3), 0.30, 0.05)
    assert not r.cond1 and r.max_partition == "x0" and r.max_ami == pytest.approx(1.0)
    assert r.verdict == "not_repeats"


def test_rival_exclusion_self_out():
    """Справка self_out: без себя в наборе соперников деление по признаку места обгоняет остальных."""
    ids, groups, ranks, strata, ind, rivals, place = _world(9, (0.0, 0.0), place_signal=2.0)
    ev = _evidence(ids, groups, strata, ind, rivals)
    own = P.sized_groups(place[:, 0], ids, SIZES)
    with_self = O.t1_eval(own, ev, np.random.default_rng(1), 0.05, 0.10)
    without = O.t1_eval(own, ev, np.random.default_rng(1), 0.05, 0.10, exclude=["sized:x0:+", "sized:x0:-"])
    assert with_self.verdict == "partial_overall"
    assert without.verdict == "confirmed"


# --- T3 и T2 ---------------------------------------------------------------------------------------


def test_t3_rule_both_outcomes_and_one_in_ten():
    rng = np.random.default_rng(0)
    placebo = {"main": rng.poisson(40, 200), "sensitivity": rng.poisson(40, 200)}
    r = PL.t3_eval({"main": 220, "sensitivity": 210}, placebo, 1776, 95, [0.09, 0.11], "main")
    assert r.verdict == "confirmed" and r.one_in_ten
    r = PL.t3_eval({"main": 42, "sensitivity": 41}, placebo, 1776, 95, [0.09, 0.11], "main")
    assert r.verdict == "not" and not r.one_in_ten
    r = PL.t3_eval({"main": 220, "sensitivity": 41}, placebo, 1776, 95, [0.09, 0.11], "main")
    assert r.verdict == "partial"
    assert r.per["main"].lo <= r.excess <= r.per["main"].hi


def _transitions(n_up: int, n_down: int, rank0: np.ndarray, n: int = 400):
    """Надёжные переходы между соседними ступенями: 0 → 1 (вверх) и 1 → 0 (вниз) в нумерации рангов."""
    lo, hi = int(np.argsort(rank0)[0]), int(np.argsort(rank0)[1])
    t_a = np.full(n, -1)
    t_b = np.full(n, -1)
    rel = np.zeros(n, dtype=bool)
    t_a[:n_up], t_b[:n_up] = lo, hi
    t_a[n_up : n_up + n_down], t_b[n_up : n_up + n_down] = hi, lo
    rel[: n_up + n_down] = True
    return t_a, t_b, rel


def _placebo_mats(k: int, up_share: float, per_pair: int, pairs: int, rank0) -> list[np.ndarray]:
    lo, hi = int(np.argsort(rank0)[0]), int(np.argsort(rank0)[1])
    out = []
    for _ in range(pairs):
        M = np.zeros((k, k), dtype=int)
        M[lo, hi] = round(up_share * per_pair)
        M[hi, lo] = per_pair - M[lo, hi]
        out.append(M)
    return out


def _t2(n_up, n_down, p_up, per_pair=20, pairs=20):
    rank0 = np.array([2, 1, 3, 4])
    ta, tb, rel = _transitions(n_up, n_down, rank0)
    pl = _placebo_mats(4, p_up, per_pair, pairs, rank0)
    adj = [(1, 0), (0, 2), (2, 3)]
    return {
        s: PL.t2_scheme(s, ta, tb, rel, pl, rank0, 0.05, 100, 99, 0.95, np.random.default_rng(0), adj, 95)
        for s in ("main", "sensitivity")
    }


def test_t2_rule_confirmed_down_not_and_undetermined():
    per = _t2(150, 30, 0.5)
    assert PL.t2_verdict(per, 0.9, 0.05)[0] == "confirmed"
    per = _t2(30, 150, 0.5)
    assert PL.t2_verdict(per, 0.9, 0.05)[0] == "down"
    per = _t2(95, 85, 0.5)
    assert PL.t2_verdict(per, 0.9, 0.05)[0] == "not"
    per = _t2(150, 30, 0.5, per_pair=2, pairs=10)  # 20 переходов плацебо < min_pool = 100
    assert all(s.undetermined for s in per.values())
    assert PL.t2_verdict(per, 0.9, 0.05)[0] == "not"


def test_t2_null_is_placebo_share_not_half():
    """Сырая асимметрия 150:30 при плацебо с той же долей «вверх» (0,83) — не «чаще вверх»."""
    per = _t2(150, 30, 150 / 180)
    assert PL.t2_verdict(per, 0.9, 0.05)[0] == "not"


def test_t2_place_cap():
    per = _t2(150, 30, 0.5)
    verdict, key, capped = PL.t2_verdict(per, 0.001, 0.05)
    assert (verdict, key, capped) == ("partial", "partial_capped", True)


def test_place_tree_detects_reclassification_to_place_type():
    rng = np.random.default_rng(0)
    n = 900
    X = rng.normal(size=(n, 2))
    place_type = (X[:, 0] > 0).astype(int) + 2 * (X[:, 1] > 0).astype(int)
    target = place_type.copy()
    flip = rng.random(n) < 0.3
    target[flip] = rng.integers(0, 4, flip.sum())
    t_a = target.copy()
    movers = flip & (target != place_type)
    t_b = t_a.copy()
    t_b[movers] = place_type[movers]  # перешедшие — к типу своего места
    res = PL.place_tree_test(X, target, t_a, t_b, movers, 3, "balanced", 5, 199, 0, np.random.default_rng(1))
    assert res["p"] < 0.05 and res["share_a"] > res["share_null"]
    # перешедшие — случайные узлы исходного типа с случайным назначением: место их не отличает
    movers2 = rng.random(n) < 0.2
    t_b2 = t_a.copy()
    t_b2[movers2] = (t_a[movers2] + 1 + rng.integers(0, 3, movers2.sum())) % 4
    res2 = PL.place_tree_test(
        X, target, t_a, t_b2, movers2, 3, "balanced", 5, 199, 0, np.random.default_rng(1)
    )
    assert res2["p"] > 0.05


def test_pseudo_pairs_cover_calendar_and_are_distinct():
    sets = PL.pseudo_sets(200, np.random.default_rng(0))
    assert len(set(sets)) == 200 and all(len(s) == 6 and list(s) == sorted(s) for s in sets)
    # набор и дополнение — одна псевдопара с переставленными А и Б: зеркальных дублей нет
    classes = {min(s, tuple(sorted(set(range(1, 13)) - set(s)))) for s in sets}
    assert len(classes) == 200
    # ориентация случайна: в А попадает и набор с январём, и его дополнение
    with_jan = sum(1 in s for s in sets)
    assert 60 < with_jan < 140
    # все 462 класса, больше — ошибка; тот же rng — те же наборы
    assert (
        len(
            {
                min(s, tuple(sorted(set(range(1, 13)) - set(s))))
                for s in PL.pseudo_sets(462, np.random.default_rng(1))
            }
        )
        == 462
    )
    with pytest.raises(ValueError):
        PL.pseudo_sets(463, np.random.default_rng(1))
    assert PL.pseudo_sets(5, np.random.default_rng([42, 601])) == PL.pseudo_sets(
        5, np.random.default_rng([42, 601])
    )
    ma_c, mb_c = PL.pseudo_months(tuple(sorted(set(range(1, 13)) - set(sets[0]))))
    a, b = PL.pseudo_months(sets[0])
    assert (ma_c, mb_c) == (b, a)
    assert sorted({t % 12 for t in a}) == list(range(12)) and sorted({t % 12 for t in b}) == list(range(12))
    assert sorted(a + b) == list(range(24))
    h1, h2 = PL.split_halves(a, ((1, 2, 5, 6, 9, 10), (3, 4, 7, 8, 11, 12)))
    assert len(h1) == len(h2) == 6 and sorted({t % 12 + 1 for t in h1}) == [1, 2, 5, 6, 9, 10]


# --- T6 и T7 ---------------------------------------------------------------------------------------


def _t6_world(effect: float, seed: int = 0):
    rng = np.random.default_rng(seed)
    n = 600
    t_a = rng.choice([2, 1], n)
    rel = rng.random(n) < 0.2
    t_b = np.where(rel, np.where(t_a == 2, 1, 3), t_a)
    change = rng.normal(size=n) + effect * rel
    return change, t_a, t_b, rel


def test_t6_rule_both_outcomes():
    ch, ta, tb, rel = _t6_world(1.0)
    r = EX.t6_eval(ch, ta, tb, rel, [(2, 1), (1, 3)], 199, 199, 0.05, 0.95, 0.0, np.random.default_rng(0))
    assert r["verdict"] == "confirmed" and r["delta"] > 0.3
    ch, ta, tb, rel = _t6_world(0.0, seed=1)
    r = EX.t6_eval(ch, ta, tb, rel, [(2, 1), (1, 3)], 199, 199, 0.05, 0.95, 0.0, np.random.default_rng(0))
    assert r["verdict"] == "not"


def _t7_world(type_matters: bool, basket_matters: bool, seed: int = 0):
    rng = np.random.default_rng(seed)
    n = 400
    groups = rng.integers(0, 10, n)
    types = rng.integers(1, 5, n)
    F = rng.normal(size=(n, 3)) + (types[:, None] * 1.5 if type_matters else 0.0)
    y = (F[:, 0] if basket_matters else 0.0) + (types * 2.0 if type_matters else 0.0) + rng.normal(0, 0.3, n)
    xy = rng.normal(size=(n, 2))
    dec = P.value_quantiles(rng.normal(size=n), 10)
    st = EX.t7_sets(F, xy, groups, types, dec, y, y, 10, 5, 5, np.random.default_rng(1))
    return EX.t7_eval(st, 199, 0.95, 0.0, np.random.default_rng(2))


def test_t7_rule_both_outcomes_and_type_gain():
    r = _t7_world(type_matters=True, basket_matters=True)
    assert r["verdict_A"] == "confirmed"
    assert r["type_gain"] in ("adds", "neutral")
    r = _t7_world(type_matters=False, basket_matters=False, seed=3)
    assert r["verdict"] == "not"
    assert r["type_gain"] == "neutral" and r["product"] == "D"


# --- Контроли и тексты ------------------------------------------------------------------------------


def _fake(t1_verdict, t5_verdict):
    per = {
        "catering": O.T1Turnover(
            "catering", 1, 1, 0, (0, 0), 1, 1, [], False, 0, (0, 0), 1, 1, [], False, 0, (0, 0), "", "", 0
        )
    }
    t1 = O.T1Result(verdict=t1_verdict, per=per)
    t5 = O.T5Result(t5_verdict, True, False, {}, {}, "", "", 0.0, {})
    return t1, t5


def test_control_row_expected_levels():
    exp = _spec()["controls"]["expected"]
    ok = ST._control_row("x", "x", "partitions", *_fake("partial_overall", "not_empty"), exp)
    assert ok["t1_ok"] and ok["t5_ok"]
    bad = ST._control_row("x", "x", "partitions", *_fake("partial_one", "not_empty"), exp)
    assert not bad["t1_ok"]
    rnd = ST._control_row("r", "r", "random", *_fake("partial_overall", "not_empty"), exp)
    assert not rnd["t1_ok"]
    bad5 = ST._control_row("x", "x", "partitions", *_fake("not", "partial"), exp)
    assert not bad5["t5_ok"]


def test_outcome_templates_match_code_fields():
    spec = _spec()
    assert TX.validate_outcomes(spec["tests"]) == []


def test_t6_direction_field_is_rejected():
    """T6 не говорит о направлении: поле {direction} в его тексте — расхождение (поправка 30.09.2026)."""
    data = load_config()["interpret"]
    data["tests"]["T6_bank_coverage"]["outcomes"]["not"]["text"] = "Сдвиг {direction} виден."
    problems = TX.validate_outcomes(Spec(data)["tests"])
    assert any("T6_bank_coverage.outcomes.not.text" in p and "direction" in p for p in problems)


def test_flows_text():
    assert TX.flows_text([[2, 1], [1, 3]]) == "из типа 2 в тип 1 и из типа 1 в тип 3"
    assert TX.flows_text([(2, 1)]) == "из типа 2 в тип 1"


def test_amendments_match_config():
    """Каждая поправка до вскрытия: поле есть, текст изменён, а прежний текст дословно записан в комментарии
    «ПОПРАВКА …» конфига (комментарии склеиваются через пробел)."""
    from pathlib import Path

    from munnet.interpret import amendments as AM

    raw = (Path(__file__).resolve().parents[1] / "configs" / "default.yaml").read_text(encoding="utf-8")
    comments = " ".join(line.split("#", 1)[1].strip() for line in raw.splitlines() if "#" in line)
    spec = _spec()
    assert AM.AMENDMENTS
    for a in AM.AMENDMENTS:
        now = AM.current(spec, a.path)
        assert now and now != a.before, a.path
        assert f"ПОПРАВКА {a.date} до вскрытия результатов" in comments
        assert a.before in comments, a.path
    lines = AM.lines(spec)
    assert len(lines) == len(AM.AMENDMENTS) and all("было «" in x and "→ стало «" in x for x in lines)


# sha256 канонического JSON блока ``interpret`` в коммите предрегистрации 90991e1 (ключи приведены к str):
# yaml.safe_load(`git show 90991e1:configs/default.yaml`)["interpret"] -> _block_sha256.
# Посчитано один раз 30.09.2026.
_PREREG_90991E1_SHA256 = "5e4cd3d50d354893800bb6851f6300ffabb7b96cd91075b967224fe61e369d25"


def _block_sha256(block) -> str:
    import hashlib
    import json

    def canon(o):
        if isinstance(o, dict):
            return {str(k): canon(v) for k, v in o.items()}
        if isinstance(o, list):
            return [canon(v) for v in o]
        return o

    blob = json.dumps(canon(block), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _with_before(block, amendments):
    """Копия блока, в которой поля поправок возвращены к прежнему тексту («было»)."""
    import copy

    out = copy.deepcopy(block)
    for a in amendments:
        node = out
        *head, last = a.path.split(".")
        for part in head:
            node = node[part]
        node[last] = a.before
    return out


def test_amendments_complete_vs_preregistration():
    """Список поправок полон: блок ``interpret`` с возвращёнными прежними текстами совпадает с 90991e1.
    Любая правка блока без записи в AMENDMENTS (или правка правила) роняет тест."""
    from munnet.interpret import amendments as AM

    block = load_config()["interpret"]
    assert _block_sha256(_with_before(block, AM.AMENDMENTS)) == _PREREG_90991E1_SHA256
    # самопроверка: без одной поправки или с изменённым порогом хеш не совпадает
    assert _block_sha256(_with_before(block, AM.AMENDMENTS[1:])) != _PREREG_90991E1_SHA256
    tampered = _with_before(block, AM.AMENDMENTS)
    tampered["tests"]["T6_bank_coverage"]["alpha"] = 0.1
    assert _block_sha256(tampered) != _PREREG_90991E1_SHA256


# Слова, которыми текст утверждает направление смен типа или прогноз. На экране 0 (thesis_assembly) их могут
# говорить только тексты T2 и T3 — у них есть проверка направления и числа смен; T6 и T7 их не добавляют.
_CLAIM_MARKERS = (
    "сдвиг",
    "вверх",
    "вниз",
    "чаще",
    "городск",
    "с большей долей",
    "предсказ",
    "предскаж",
    "прогноз",
)


def _fill(template: str) -> str:
    return str(template).format_map(Counter({k: 1 for k in TX.placeholders(template)}))


def _markers(text: str) -> set[str]:
    low = text.lower().replace("ё", "е")
    return {m for m in _CLAIM_MARKERS if m in low}


def test_screen0_t6_t7_add_no_direction_or_forecast_for_any_outcomes():
    """Перебор исходов T2 × T3 × T6 × T7 (и оговорки охвата): тексты T6 и T7 на экране 0 не утверждают
    направления смен типа и прогноза, поэтому не противоречат ни «смен типа не больше плацебо» (T3 not), ни
    «вниз чаще» (T2 down). Прежние тексты (amendments) эта проверка ловит."""
    from munnet.interpret import amendments as AM

    spec = _spec()
    tests = spec["tests"]
    o2, o3 = tests["T2_direction"]["outcomes"], tests["T3_reliable_placebo"]["outcomes"]
    o6, o7 = tests["T6_bank_coverage"]["outcomes"], tests["T7_utility"]["outcomes"]
    movers = tests["T6_bank_coverage"]["movers"]
    words = tests["T2_direction"]["direction_words"]
    med = {"A": 0.04, "B": 0.05, "C": 0.06, "D": 0.045}
    t7_cases = (("confirmed", True), ("partial", True), ("partial", False), ("not", False))
    combos = list(
        product(
            ("confirmed", "partial_capped", "partial", "not", "down"),
            ("confirmed", "partial", "not"),
            ("confirmed", "partial", "not"),
            t7_cases,
            (False, True),
        )
    )
    for k2, v3, v6, (v7, against_b), with_caveat in combos:
        t6 = TX.t6_text({"verdict": v6, "delta": 0.1, "ci": (0.0, 0.2), "p": 0.03}, o6, movers)
        if with_caveat:
            t6 += " " + TX.t6_caveat(0.08, o6)
        r7 = {"verdict": v7, "product": "D", "median_error": med, "against_B": against_b}
        t7 = TX.t7_text(r7, o7)
        screen = " ".join([_fill(o3[v3]["text"]), _fill(o2[k2]["text"]), t7, t6])
        added = _markers(t6) | _markers(t7)
        assert not added, (k2, v3, v6, v7, added)
        assert "{" not in t6 and "{" not in t7
        if v3 == "not":
            assert "сдвиг" not in screen.lower()
        if k2 == "down":
            for w in (words["t1_confirmed"], words["otherwise"]):
                assert str(w) not in screen
    assert len(combos) == 5 * 3 * 3 * 4 * 2
    assert all(_markers(_fill(a.before)) for a in AM.AMENDMENTS)


def _t1_result(verdict_beyond: dict, overall: dict) -> O.T1Result:
    per = {}
    for n in ("catering", "retail"):
        per[n] = O.T1Turnover(
            n,
            100,
            90,
            0.3,
            (0.2, 0.4),
            0.001,
            0.002,
            [1, 2, 3, 4],
            True,
            0.25,
            (0.2, 0.3),
            0.001,
            0.002,
            [0.4, 0.5, 0.6, 0.7],
            True,
            0.1,
            (0.05, 0.15),
            "sized:x",
            "зарплата, по возрастанию",
            0.15,
            overall=overall[n],
            beyond=verdict_beyond[n],
        )
    return O.T1Result(verdict="", per=per)


def test_t1_text_selection_each_verdict():
    out = _spec()["tests"]["T1_ladder_external"]["outcomes"]
    r = _t1_result({"catering": True, "retail": True}, {"catering": True, "retail": True})
    assert TX.t1_text(r, out, "confirmed").startswith("Четыре типа — ступени")
    r = _t1_result({"catering": True, "retail": False}, {"catering": True, "retail": True})
    txt = TX.t1_text(r, out, "partial_one")
    assert "по обороту общепита" in txt and "только в целом" in txt
    r = _t1_result({"catering": True, "retail": False}, {"catering": True, "retail": False})
    assert "не упорядочены и в целом" in TX.t1_text(r, out, "partial_one")
    r = _t1_result({"catering": False, "retail": False}, {"catering": True, "retail": False})
    txt = TX.t1_text(r, out, "partial_overall")
    assert txt.startswith("Типы упорядочены по обороту общепита Росстата в целом")
    assert TX.t1_text(r, out, "not").startswith("Четыре типа корзины трат")
    # «за в целом» у общепита есть — пометки нет
    assert TX.t1_overall_note(r, "partial_overall") == ""


def test_t1_capped_to_partial_overall_without_overall_is_marked(caplog):
    """Буквальное «за сверх места» при |ρ (a)| < rho_min: beyond = True, overall = False. Если
    t1_cap_by_t5 понизил текст до partial_overall, обороты подставляются все, но не молча:
    предупреждение в лог и пометка рядом с текстом; правило не меняется."""
    out = _spec()["tests"]["T1_ladder_external"]["outcomes"]
    r = _t1_result({"catering": True, "retail": True}, {"catering": False, "retail": False})
    with caplog.at_level("WARNING", logger="munnet.interpret.texts"):
        txt = TX.t1_text(r, out, "partial_overall")
    assert txt.startswith("Типы упорядочены по обороту общепита")
    assert any("за в целом" in m for m in caplog.messages)
    note = TX.t1_overall_note(r, "partial_overall")
    assert "«за в целом» не выполнено: |ρ (a)| ниже rho_min" in note
    assert "общепит 0,30" in note
    # при других текстах пометки нет
    assert TX.t1_overall_note(r, "confirmed") == ""


def test_control_row_publishes_strict_verdict_without_affecting_ok():
    exp = _spec()["controls"]["expected"]
    t1, t5 = _fake("partial_overall", "not_empty")
    t1.verdict_strict = "confirmed"  # строгое прочтение выше ожидаемого — на t1_ok не влияет
    row = ST._control_row("x", "x", "partitions", t1, t5, exp)
    assert row["t1_strict"] == "confirmed" and row["t1_ok"]
    tab = pd.DataFrame([{**row, "group": "partitions"}])
    assert ST.controls_info(tab, 1, 0, 0)["passed"]


def test_t3_text_one_in_ten_only_by_excess():
    out = _spec()["tests"]["T3_reliable_placebo"]["outcomes"]
    placebo = {"main": np.full(200, 20.0), "sensitivity": np.full(200, 20.0)}
    r = PL.t3_eval({"main": 200, "sensitivity": 200}, placebo, 1776, 95, [0.09, 0.11], "main")
    assert "каждое десятое" in TX.t3_text(r, out, "main")
    r = PL.t3_eval(
        {"main": 181, "sensitivity": 181},
        {"main": np.full(200, 100.0), "sensitivity": np.full(200, 100.0)},
        1776,
        95,
        [0.09, 0.11],
        "main",
    )
    assert r.verdict == "confirmed" and "каждое десятое" not in TX.t3_text(r, out, "main")


def test_t2_text_partial_other_halves_variants():
    out = _spec()["tests"]["T2_direction"]["outcomes"]
    per = _t2(150, 30, 0.5)
    per["sensitivity"] = _t2(95, 85, 0.5)["sensitivity"]
    v, key, capped = PL.t2_verdict(per, 0.9, 0.05)
    res = PL.T2Result(v, key, per, {"share_a": 0.3, "share_null": 0.2}, capped)
    txt = TX.t2_text(res, out, "вверх", "main", 100)
    assert v == "partial" and txt.endswith("— нет: преобладание направления не доказано.")
    per["sensitivity"] = _t2(150, 30, 0.5, per_pair=2, pairs=10)["sensitivity"]
    v, key, capped = PL.t2_verdict(per, 0.9, 0.05)
    txt = TX.t2_text(PL.T2Result(v, key, per, {}, capped), out, "вверх", "main", 100)
    assert "не определено" in txt


def test_t7_text_capitalised_and_type_gain():
    out = _spec()["tests"]["T7_utility"]["outcomes"]
    r = _t7_world(type_matters=True, basket_matters=True)
    assert TX.t7_text(r, out)[0].isupper()
    assert TX.t7_type_text(r, out)


# --- Лестница, свежесть, слепой режим ---------------------------------------------------------------


def test_ladder_transfer_by_size_and_stop_on_new_sizes():
    lad = _spec()["ladder"]
    types = np.concatenate([np.full(806, 3), np.full(473, 1), np.full(394, 4), np.full(103, 2)])
    transfer, order = D.ladder_transfer(types, lad)
    assert transfer == {3: 1, 1: 2, 4: 3, 2: 4} and order == [2, 1, 3, 4]
    with pytest.raises(QCError):
        D.ladder_transfer(np.concatenate([types, [1]]), lad)


def _fresh_cfg(tmp_path):
    cfg = load_config()
    data = {
        **cfg.data,
        "paths": {**cfg.data["paths"], "processed": str(tmp_path / "p"), "outputs": str(tmp_path / "o")},
    }
    for d in ("p", "o/evaluate", "o/dynamics"):
        (tmp_path / d).mkdir(parents=True)
    return type(cfg)(data=data, path=cfg.path)


def test_freshness_stops_when_outputs_older_than_labels(tmp_path):
    cfg = _fresh_cfg(tmp_path)
    old = time.time() - 3600
    for f in ("o/evaluate/a.csv", "o/dynamics/nodes.csv"):
        (tmp_path / f).write_text("x")
        os.utime(tmp_path / f, (old, old))
    (tmp_path / "p/cluster_labels.parquet").write_text("x")
    with pytest.raises(MissingInputError, match="freshness"):
        D.check_freshness(cfg, Spec(cfg["interpret"]))
    now = time.time() + 10
    for f in ("o/evaluate/a.csv", "o/dynamics/nodes.csv"):
        os.utime(tmp_path / f, (now, now))
    D.check_freshness(cfg, Spec(cfg["interpret"]))


def test_blind_permutation_keeps_sizes_and_is_deterministic():
    ctx = PL.RunCtx("main", None, None, None, "", 0, 4, 0.5, "", None, (), {}, {}, blind=7)
    lab = np.repeat([0, 1, 2, 3], [50, 30, 15, 5])
    a, b = PL.blind(lab, ctx, "x"), PL.blind(lab, ctx, "x")
    assert np.array_equal(a, b) and not np.array_equal(a, lab)
    assert Counter(a.tolist()) == Counter(lab.tolist())
    assert not np.array_equal(PL.blind(lab, ctx, "y"), a)
    ctx0 = PL.RunCtx("main", None, None, None, "", 0, 4, 0.5, "", None, (), {}, {}, blind=None)
    assert PL.blind(lab, ctx0, "x") is lab


def test_apply_blind_shuffles_types_and_dynamics_rows():
    n = 40
    final = np.repeat([1, 2, 3, 4], 10)
    nd = pd.DataFrame(
        {
            "territory_id": np.arange(n),
            "name": "x",
            "region": "r",
            "is_city_node": False,
            "half_type_2023": final,
            "half_type_2024": final[::-1].copy(),
            "reliable": np.arange(n) % 3 == 0,
        }
    )
    data = D.Data(
        cfg=None,
        fin={},
        cp=None,
        main=None,
        final=final.copy(),
        final_raw_numbers={},
        type_jaccard={1: 0.9, 2: 0.9, 3: 0.9, 4: 0.9},
        order=[1, 2, 3, 4],
        nodes_dyn=nd.copy(),
        labels_dyn={("m", "w"): final.copy()},
        context=None,
        territories=None,
        place=None,
        city_ids=[],
    )
    D.apply_blind(data, 11)
    assert Counter(data.final.tolist()) == Counter(final.tolist()) and not np.array_equal(data.final, final)
    pairs_before = set(zip(nd["half_type_2023"], nd["half_type_2024"], nd["reliable"], strict=True))
    pairs_after = set(
        zip(
            data.nodes_dyn["half_type_2023"],
            data.nodes_dyn["half_type_2024"],
            data.nodes_dyn["reliable"],
            strict=True,
        )
    )
    assert pairs_before == pairs_after  # переход целиком переехал к другому узлу
    assert not np.array_equal(data.nodes_dyn["half_type_2023"].to_numpy(), nd["half_type_2023"].to_numpy())


def test_cli_blind_only_with_interpret():
    from munnet import cli

    with pytest.raises(SystemExit):
        cli.main(["--blind", "5", "dynamics"])


# --- Разбиения, названия, формальные понятия --------------------------------------------------------


def test_sized_groups_ties_lower_id_goes_lower():
    v = np.array([1.0, 1.0, 0.0, 2.0])
    ids = np.array([4, 2, 3, 1])
    assert P.sized_groups(v, ids, [1, 2, 1]).tolist() == [2, 2, 1, 3]
    assert P.sized_groups(v, ids, [2, 1, 1]).tolist() == [2, 1, 1, 3]  # на границе id 2 ниже id 4


def test_value_quantiles_merge_duplicate_edges_and_coverage_class():
    v = np.array([0.0] * 50 + list(np.linspace(1, 2, 50)) + [np.nan])
    g = P.value_quantiles(v, 10)
    assert g[-1] == -1 and len(np.unique(g[:-1])) < 10
    frame = pd.DataFrame({"a": [1, np.nan, np.nan], "b": [1, 1, np.nan], "c": [1, 1, np.nan]})
    assert P.coverage_class(frame).tolist() == [0, 1, 2]


def test_federal_districts_cover_all_regions_of_the_data():
    fd = _spec()["federal_districts"].plain()
    codes = [r for regs in fd.values() for r in regs]
    assert len(codes) == len(set(codes)) == 77


def test_naming_rule_words_settlement_and_duplicates():
    nm = _spec()["naming"]
    feats = ["clr_rel_cafe", "clr_rel_food", "clr_rel_marketplace", "log_pop_rel"]
    rows = []
    for t, (cafe, food, mp, pop) in {
        2: (-1.0, 0.8, 0.5, -0.9),
        1: (0.1, -0.1, 0.05, 0.0),
        3: (0.6, -0.4, -0.3, 0.7),
        4: (1.5, -1.0, -0.9, 2.0),
    }.items():
        for f, e in zip(feats, (cafe, food, mp, pop), strict=True):
            rows.append(
                {
                    "type": t,
                    "feature": f,
                    "effect_mad": e,
                    "cliff": 0.5 * np.sign(e) if abs(e) > 0.2 else 0.05,
                }
            )
    prof = pd.DataFrame(rows)
    shares = pd.DataFrame(
        {"large_cities": [0.0, 0.0, 0.1, 0.8], "cities": [0.1, 0.2, 0.6, 0.9], "rural": [0.9, 0.8, 0.4, 0.1]},
        index=pd.Index([2, 1, 3, 4], name="type"),
    )
    names = DS.name_types(prof, feats[:3], ["log_pop_rel"], nm, shares, [2, 1, 3, 4], set())
    assert names[4].name == "Крупные города, больше общепита"
    assert names[2].name.startswith("Сельские, меньше общепита")
    # |δ| < 0,15 у всех частей корзины; доля сельских 0,8 ≥ 0,5 и выше, чем у ступени сверху (0,4)
    assert names[1].name == "Сельские, без выраженных отличий корзины"
    assert "относительно своего региона" in names[4].caption.lower()
    unstable = DS.name_types(
        prof, feats[:3], ["log_pop_rel"], nm, shares, [2, 1, 3, 4], {(4, "clr_rel_cafe")}
    )
    assert unstable[4].parts[0] == "clr_rel_food"
    assert (
        DS.count_words("Крупные города, больше общепита") == 4
        and DS.count_words("торговые и транспортные") == 2
    )


def test_fca_finds_planted_implication():
    rng = np.random.default_rng(0)
    n = 600
    y = rng.integers(1, 4, n)
    X = rng.normal(size=(n, 3))
    X[y == 1, 0] += 3.0  # тип 1 ⇔ высокий признак 0
    res = DS.fca_implications(
        X, y, ["clr_rel_cafe", "log_pop_rel", "clr_rel_food"], [1, 2, 3], 2, 0.8, 0.3, 30, rng
    )
    hit = res.loc[(res["type"] == 1) & (res["size"] == 1)]
    assert len(hit) and hit["stability"].max() >= 0.8


def test_spec_tracks_unread_keys_and_rejects_other_rules():
    sp = Spec({"a": {"b": 1, "c": "x"}, "d": [1, 2]})
    assert set(sp.unread()) == {"interpret.a.b", "interpret.a.c", "interpret.d"}
    sp["a"].expect("c", "x")
    with pytest.raises(ValueError):
        sp["a"].expect("b", 2)
    assert sp.unread() == ["interpret.d"]


def test_outcome_template_with_unknown_field_is_rejected():
    import copy

    data = copy.deepcopy(load_config()["interpret"])
    data["tests"]["T3_reliable_placebo"]["outcomes"]["not"]["text"] += " {новое_поле}"
    problems = TX.validate_outcomes(Spec(data)["tests"])
    assert problems and "новое_поле" in problems[0]


# --- R1: наименьший вердикт, 4-й из 5 seed, прогон-источник; T1 при T5; стоп контролей ------------------


def _r1_runs(t3: dict[str, str], k: int = 4) -> dict:
    """Поддельные прогоны R1: одни и те же метки и корзина, вердикты T1 и T2 — confirmed, T3 — из ``t3``."""
    from types import SimpleNamespace as NS

    rng = np.random.default_rng(0)
    final = np.repeat(np.arange(k), 10)
    run = NS(final=final, basket=rng.normal(size=(len(final), 2)))
    rob = _spec()["robustness"]
    names = ["main", *[f"variant:{v}" for v in rob["variants"]], *[f"tracking:{t}" for t in rob["tracking"]]]
    names += [f"seed:{s}" for s in rob["seeds"] if int(s) != 42]
    runs = {}
    for n in names:
        r = {"run": run, "t2": NS(verdict="confirmed"), "t3": NS(verdict=t3.get(n, "confirmed"))}
        if not n.startswith("tracking:"):
            r["t1"] = NS(verdict="confirmed")  # T1 от способа прослеживания не зависит
        runs[n] = r
    return runs


def _r1(t3: dict[str, str]) -> dict:
    from types import SimpleNamespace as NS

    data = NS(k=4, base=NS(b_names=["clr_rel_food", "clr_rel_cafe"]))
    return ST._r1_summary(_r1_runs(t3), _spec(), data, 42)


def test_r1_fourth_of_five_seeds_sets_the_verdict_and_its_source():
    # seed 42 (основной) и 142, 242 — confirmed, 342 — partial, 442 — not: 4-й по величине — partial
    r = _r1({"seed:342": "partial", "seed:442": "not"})
    assert r["final_verdict"]["T3_reliable_placebo"] == "partial"
    assert r["source_run"]["T3_reliable_placebo"] == "seed:342"
    assert r["summary"]["main_higher"] == {"T3_reliable_placebo": "confirmed"}
    assert r["final_verdict"]["T1_ladder_external"] == "confirmed"
    assert r["source_run"]["T1_ladder_external"] == "main"


def test_r1_lowest_variant_wins_over_seeds():
    r = _r1({"variant:no_level": "not", "seed:442": "partial"})
    assert r["final_verdict"]["T3_reliable_placebo"] == "not"
    assert r["source_run"]["T3_reliable_placebo"] == "variant:no_level"
    table = r["table"]
    stable = table.loc[(table["test"] == "T3_reliable_placebo") & (table["run"] == "устойчив к R1")]
    assert stable["verdict"].tolist() == ["нет"]


def test_r1_tracking_lowers_t3_and_ties_go_to_main():
    r = _r1({"tracking:evolutionary": "partial", "main": "partial"})
    assert r["final_verdict"]["T3_reliable_placebo"] == "partial"
    assert r["source_run"]["T3_reliable_placebo"] == "main"  # при равенстве источник — основной расчёт
    assert r["summary"]["main_higher"] == {}
    r = _r1({})
    assert r["final_verdict"] == {
        "T1_ladder_external": "confirmed",
        "T2_direction": "confirmed",
        "T3_reliable_placebo": "confirmed",
    }
    assert set(r["source_run"].values()) == {"main"}


def test_r1_one_low_seed_is_tolerated_two_are_not():
    assert _r1({"seed:442": "not"})["final_verdict"]["T3_reliable_placebo"] == "confirmed"
    two = _r1({"seed:442": "not", "seed:342": "not"})
    assert two["final_verdict"]["T3_reliable_placebo"] == "not"
    assert two["source_run"]["T3_reliable_placebo"] == "seed:342"


def test_t1_cap_by_t5():
    cap = _spec()["thesis_assembly"]["t1_cap_by_t5"].plain()
    assert ST.cap_t1("confirmed", "not_repeats", cap) == "partial_overall"
    assert ST.cap_t1("partial_one", "not_empty", cap) == "partial_overall"
    assert ST.cap_t1("not", "not_repeats", cap) == "not"  # ниже потолка — без изменений
    assert ST.cap_t1("confirmed", "confirmed", cap) == "confirmed"
    assert ST.cap_t1("confirmed", "partial", cap) == "confirmed"


def _controls_tab(t1_ok: list[bool], rnd_ok: list[bool]) -> pd.DataFrame:
    rows = [
        {"name": f"p{i}", "group": "partitions", "t1_ok": ok, "t5_ok": True} for i, ok in enumerate(t1_ok)
    ]
    rows += [{"name": f"r{i}", "group": "random", "t1_ok": True, "t5_ok": ok} for i, ok in enumerate(rnd_ok)]
    rows += [{"name": "ref", "group": "basket_parts", "t1_ok": np.nan, "t5_ok": np.nan}]  # справка не решает
    return pd.DataFrame(rows)


def test_controls_stop_when_a_control_is_above_expected(tmp_path):
    ok = ST.controls_info(_controls_tab([True, True], [True, False]), 2, 2, 1)
    assert ok["passed"] and ok["n_random_above"] == 1
    ST.stop_on_controls(ok, tmp_path)
    assert not (tmp_path / "controls.json").exists()
    bad = ST.controls_info(_controls_tab([True, False], [True, True]), 2, 2, 1)
    assert not bad["passed"] and bad["bad_partitions"] == ["p1"]
    with pytest.raises(QCError, match="controls"):
        ST.stop_on_controls(bad, tmp_path)
    assert (tmp_path / "controls.json").exists()
    many = ST.controls_info(_controls_tab([True], [False, False]), 1, 2, 1)
    assert not many["passed"]
    with pytest.raises(QCError):
        ST.stop_on_controls(many, tmp_path)


# --- Эволюционный способ на псевдопаре: цепочка шагов, как в наблюдении -----------------------------


def test_evolutionary_chain_forgets_start_like_the_window_chain():
    rng = np.random.default_rng(1)
    Z = np.r_[rng.normal(0.0, 0.1, (50, 1)), rng.normal(10.0, 0.1, (50, 1))]
    C0 = np.array([[3.0], [7.0]])
    lab0 = np.repeat([0, 1], 50)
    means = np.array([[Z[:50].mean()], [Z[50:].mean()]])
    one = PL.evolutionary_chain(Z, C0, lab0, 2, 1, 0.5, 300)[1]
    many = PL.evolutionary_chain(Z, C0, lab0, 2, 12, 0.5, 300)[1]
    assert np.abs(one - means).max() > 1.0  # один шаг: центр — середина между стартом и средним
    assert np.abs(many - means).max() < 1e-2  # 12 шагов: вес старта ~0,5^12


def test_t1_main_verdict_is_literal_strict_is_sensitivity():
    """Главный вердикт — буквальное прочтение (без rho_min для «сверх места»), строгое — рядом."""
    ids, groups, ranks, strata, ind, rivals, _ = _world(0, (1.0, 1.0))
    ev = _evidence(ids, groups, strata, ind, rivals)
    r = O.t1_eval(ranks, ev, np.random.default_rng(2), 0.05, 0.99)  # |ρ (a)| заведомо меньше rho_min
    assert all(abs(t.rho_a) < 0.99 for t in r.per.values())
    assert r.verdict == "confirmed" and all(t.beyond and not t.overall for t in r.per.values())
    assert r.verdict_strict == "not" and not any(t.beyond_strict for t in r.per.values())
    r2 = O.t1_eval(ranks, ev, np.random.default_rng(2), 0.05, 0.10)
    assert r2.verdict == r2.verdict_strict == "confirmed"


def test_t1_verdict_rule():
    assert O.t1_verdict([True, True], [True, True]) == "confirmed"
    assert O.t1_verdict([True, False], [True, False]) == "partial_one"
    assert O.t1_verdict([False, False], [False, True]) == "partial_overall"
    assert O.t1_verdict([False, False], [False, False]) == "not"


def test_coverage_caveat_takes_the_larger_reading():
    assert ST.coverage_eps2_max({"place_year": 0.02, "months_24": 0.07}) == 0.07
    assert ST.coverage_eps2_max({"place_year": 0.08, "months_24": float("nan")}) == 0.08
    assert np.isnan(ST.coverage_eps2_max({"place_year": float("nan")}))


def test_naming_answer_applies_only_to_its_own_package(tmp_path):
    import json

    path = tmp_path / "interpretation_naming_test.json"
    st = ST.naming_answer(path, "abc", 4)
    assert st["state"] == "no_answer" and not st["done"]
    path.write_text(json.dumps({"correct": 4, "package_sha": "abc"}), encoding="utf-8")
    st = ST.naming_answer(path, "abc", 4)
    assert st["state"] == "accepted" and st["done"] and not st["fallback"]
    path.write_text(json.dumps({"correct": 3, "package_sha": "abc"}), encoding="utf-8")
    assert ST.naming_answer(path, "abc", 4)["fallback"]
    path.write_text(json.dumps({"correct": 4, "package_sha": "old"}), encoding="utf-8")
    st = ST.naming_answer(path, "abc", 4)
    assert st["state"] == "stale" and not st["done"]  # ответ к другому пакету не применяется
    path.write_text(json.dumps({"correct": 4}), encoding="utf-8")
    assert ST.naming_answer(path, "abc", 4)["state"] == "stale"
    report = tmp_path / "docs" / "interpretation.md"
    assert (
        ST.naming_result_path(tmp_path, report, None) == tmp_path / "docs" / "interpretation_naming_test.json"
    )
    assert ST.naming_result_path(tmp_path, report, 7).parent.name == "interpret_blind"


def test_chain_direct_check_stops_only_on_the_2024_window(tmp_path):
    import json

    from munnet.dynamics import tracking as T

    spec = Spec({"windows": {"year_b": ["2024-01", "2024-12"]}}, "interpret")
    w24 = "2024-01…2024-12"
    rng = np.random.default_rng(0)
    final = rng.integers(1, 5, 300)
    facts = tmp_path / "facts.json"
    facts.write_text(
        json.dumps({"main_info": {"chain_direct_nodes_differ": 0.0, "chain_direct_windows_same": 13.0}})
    )
    # окно 2024 года уже в нумерации итога (прямое сопоставление даёт те же номера) — проходит
    lab = final.copy()
    lab[:30] = rng.integers(1, 5, 30)
    lab = T.aligned(final - 1, lab - 1, 4) + 1
    res = D.check_chain_direct(facts, final, {(PL.MAIN, w24): lab}, spec, 4)
    assert res["last_window_differ"] == 0
    # цепочка дала другую нумерацию (два типа поменялись номерами) — стоп до тяжёлых расчётов
    swapped = np.where(lab == 1, 2, np.where(lab == 2, 1, lab))
    with pytest.raises(QCError, match="цепочкой"):
        D.check_chain_direct(facts, final, {(PL.MAIN, w24): swapped}, spec, 4)
    # нет числа в facts dynamics — стоп с подсказкой
    facts.write_text(json.dumps({"main_info": {}}))
    with pytest.raises(QCError, match="chain_direct_nodes_differ"):
        D.check_chain_direct(facts, final, {(PL.MAIN, w24): lab}, spec, 4)


def test_cache_key_follows_network_and_impl_config_but_not_interpret_texts():
    import copy

    from munnet.interpret import runs as RN

    cfg = load_config()
    base = RN.config_hash(cfg)
    for path in (("network",), ("clustering", "impl"), ("dynamics", "impl")):
        d = copy.deepcopy(cfg.data)
        node = d
        for k in path[:-1]:
            node = node[k]
        node[path[-1]] = {**node[path[-1]], "_probe": 1}
        assert RN.config_hash(d) != base, path
    d = copy.deepcopy(cfg.data)
    d["interpret"]["report"] = "другой.md"
    assert RN.config_hash(d) == base
    mods = set(RN.HEAVY_MODULES)
    assert {"munnet.clustering.methods", "munnet.clustering.inputs", "munnet.network.graph"} <= mods
    assert len(RN.code_hash()) == 20
    # число процессов результат не меняет — в ключ не входит
    d = copy.deepcopy(cfg.data)
    d["clustering"]["impl"]["workers"] = 13
    assert RN.config_hash(d) == base


def test_code_hash_follows_imports_of_heavy_modules_but_not_report_texts():
    from munnet.interpret import runs as RN

    closure = set(RN.heavy_closure())
    assert set(RN.HEAVY_MODULES) <= closure
    # импорты тяжёлых модулей (в том числе внутри функций) — в ключе; тексты отчёта этапа — нет
    assert {"munnet.interpret.data", "munnet.contracts", "munnet.config"} <= closure
    assert not {"munnet.interpret.report", "munnet.interpret.texts", "munnet.interpret.stage"} & closure


def test_digest_follows_content_not_identity():
    from dataclasses import dataclass
    from types import SimpleNamespace

    import scipy.sparse as sp

    from munnet.interpret import runs as RN

    @dataclass(frozen=True)
    class Box:
        a: np.ndarray
        t: pd.DataFrame
        m: object
        extra: dict

    rng = np.random.default_rng(0)
    a = rng.normal(size=(5, 3))
    t = pd.DataFrame({"id": [1, 2, 3], "name": ["x", "y", "z"]})
    m = sp.random(6, 6, density=0.3, random_state=1, format="csr")
    box = Box(a, t, m, {"b": 1, "a": (1, 2)})
    same = Box(a.copy(), t.copy(), m.copy(), {"a": (1, 2), "b": 1})
    assert RN.digest(box) == RN.digest(same)
    a2 = a.copy()
    a2[0, 0] += 1e-9
    assert RN.digest(Box(a2, t, m, box.extra)) != RN.digest(box)
    t2 = t.copy()
    t2.loc[0, "name"] = "q"
    assert RN.digest(Box(a, t2, m, box.extra)) != RN.digest(box)
    m2 = m.copy()
    m2.data[0] += 1.0
    assert RN.digest(Box(a, t, m2, box.extra)) != RN.digest(box)
    assert RN.digest(SimpleNamespace(x=1)) != RN.digest(SimpleNamespace(x=2))
    assert RN.digest(a.astype(np.float32)) != RN.digest(a)


def test_seed_protocol_must_be_consecutive_from_seed():
    assert ST.check_seed_protocol({"seeds": list(range(42, 52))}, 42) == tuple(range(42, 52))
    for bad in ([43, 44], [42, 44, 45], []):
        with pytest.raises(ValueError, match="протокол seed"):
            ST.check_seed_protocol({"seeds": bad}, 42)


def _chain_case(k=3, n=60, n_roll=5, seed=0):
    from types import SimpleNamespace

    from munnet.dynamics import compute as DC

    rng = np.random.default_rng(seed)
    base = rng.integers(0, k, n)
    fits = {"all24": base.copy()}
    roll = [SimpleNamespace(name=f"w{i}") for i in range(n_roll)]
    for i, q in enumerate(roll):
        lab = base.copy()
        flip = rng.random(n) < 0.15 * (i + 1) / n_roll
        lab[flip] = rng.integers(0, k, int(flip.sum()))
        fits[q.name] = (lab + i) % k  # номера окон сдвинуты — сопоставление их возвращает
    halves_p = {h: None for h in ("odd_2023", "even_2023", "odd_2024", "even_2024")}
    for j, h in enumerate(halves_p):
        fits[h] = (base + j) % k
    track = DC.main_track(fits["all24"], [fits[q.name] for q in roll], {h: fits[h] for h in halves_p}, k)
    return fits, roll, halves_p, track


def test_chain_check_in_blind_run_equals_real():
    """Сверка «цепочка против прямого» в слепом прогоне идёт на метках в исходном порядке узлов: то же
    число, что без перемешивания (раньше в слепом прогоне оно было следствием перемешивания)."""
    from types import SimpleNamespace

    from munnet.dynamics import compute as DC
    from munnet.dynamics import tracking as T
    from munnet.interpret import runs as RN

    k = 3
    fits, roll, halves_p, track = _chain_case(k)
    real = RN.chain_check(
        SimpleNamespace(k=k, blind=None, name="v"), fits, fits["all24"], track, roll, halves_p
    )
    ctx_b = SimpleNamespace(k=k, blind=7, name="v")
    fb = {tag: PL.blind(lab, ctx_b, "fit", tag) for tag, lab in fits.items()}
    tb = DC.main_track(fb["all24"], [fb[q.name] for q in roll], {h: fb[h] for h in halves_p}, k)
    blind = RN.chain_check(ctx_b, fb, fb["all24"], tb, roll, halves_p)
    assert blind == real
    naive = int((tb.rolling[-1] != T.aligned(fb["all24"], fb[roll[-1].name], k)).sum())
    assert naive != real["last_window_differ"]  # без возврата порядка число было бы другим


def test_evo_refit_repeats_dynamics_centers_exactly():
    """Эволюционный способ, пройденный заново тем же кодом, повторяет метки dynamics (ARI 1) и в слепом
    прогоне (метки dynamics перемешаны dynamics_perm)."""
    from types import SimpleNamespace

    from munnet.dynamics import compute as DC
    from munnet.interpret import runs as RN

    k, n = 3, 90
    rng = np.random.default_rng(3)
    centers = rng.normal(scale=4, size=(k, 2))
    truth = rng.integers(0, k, n)
    roll = [SimpleNamespace(name=f"w{i}") for i in range(4)]
    got = {q.name: (None, centers[truth] + rng.normal(size=(n, 2))) for q in roll}
    for per in D.HALF_PERIODS.values():
        got[per] = (None, centers[truth] + rng.normal(size=(n, 2)))
    first = truth.copy()
    z_h = {h: got[per][1] for h, per in D.HALF_PERIODS.items()}
    tr, Cs = DC.evolutionary_track([got[q.name][1] for q in roll], z_h, first, k, 0.5, 100)
    labels = {(PL.MAIN, roll[0].name): first + 1}
    labels[(PL.EVO, roll[0].name)] = tr.rolling[0] + 1
    labels[(PL.EVO, roll[-1].name)] = tr.rolling[-1] + 1
    for h, per in D.HALF_PERIODS.items():
        labels[(PL.EVO, per)] = tr.halves[h] + 1
    ctx = SimpleNamespace(k=k, blind=None, epsilon=0.5, max_iter=100, name="main")
    tr2, Cs2, ari = RN.evo_refit(ctx, labels, got, roll, 0.99)
    assert min(ari.values()) == 1.0 and np.allclose(Cs2[-1], Cs[-1])
    p = D.dynamics_perm(n, 11)
    blinded = {key: v[p] for key, v in labels.items()}
    tr3, Cs3, ari3 = RN.evo_refit(SimpleNamespace(**{**vars(ctx), "blind": 11}), blinded, got, roll, 0.99)
    assert ari3 == ari and np.allclose(Cs3[-1], Cs[-1])


def test_scale_rows_uses_all_nodes_as_clustering_does():
    """Масштаб «как clustering.inputs.scale» — медиана и MAD по всем узлам сети, строки — подвыборка."""
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"a": rng.normal(size=50), "b": rng.exponential(size=50)})
    frame.loc[[0, 1], "a"] = [40.0, 55.0]  # узлы-города: входят в базу масштаба, но не в строки
    rows = np.arange(2, 50)
    Z = P.scale_rows(frame, rows)
    assert np.allclose(Z, P.robust(frame)[rows])
    assert not np.allclose(Z, P.robust(frame.iloc[rows]))  # масштаб подвыборки дал бы другое
    assert np.allclose(P.scale_rows(frame, rows, "zscore"), P.zscore(frame)[rows])
    with pytest.raises(ValueError):
        P.scale_rows(frame, rows, "minmax")


# --- Ветки «частично» ------------------------------------------------------------------------------


def test_t2_partial_when_up_in_one_partition_and_place_cap_keeps_partial():
    up, flat = _t2(150, 30, 0.5), _t2(95, 85, 0.5)
    per = {"main": up["main"], "sensitivity": flat["sensitivity"]}
    assert PL.t2_verdict(per, 0.9, 0.05) == ("partial", "partial", False)
    # place_cap: при partial вердикт не выше partial, текст — обычный partial
    # (partial_capped — только из confirmed)
    assert PL.t2_verdict(per, 0.001, 0.05) == ("partial", "partial", True)


def test_t5_partial_when_only_condition_3_holds():
    """Типы повторяют тривиальное деление (условие 1 нет), но упорядочивают обороты сверх места (3 — да)."""
    ids, groups, ranks, strata, ind, rivals, place = _world(6, (1.0, 1.0))
    ev = _evidence(ids, groups, strata, ind, rivals)
    trivial = {**_trivial(ids, place), "copy": Partition("copy", "копия типов", ranks.copy(), "trivial")}
    r = O.t5_eval(ranks, ev, trivial, np.random.default_rng(3), 0.30, 0.05)
    assert not r.cond1 and r.cond3 and r.max_partition == "copy"
    assert r.verdict == "partial"


def test_t6_partial_when_only_one_of_two_conditions_holds():
    ch, ta, tb, rel = _t6_world(1.0)
    r = EX.t6_eval(ch, ta, tb, rel, [(2, 1), (1, 3)], 199, 199, 0.05, 0.95, 0.9, np.random.default_rng(0))
    assert r["p"] < 0.05 and r["ci"][0] <= 0.9  # p — да, нижняя граница δ выше 0,9 — нет
    assert r["verdict"] == "partial"


def _t7_partial_world(case: str, seed: int = 0):
    """``only_c``: регион задаёт тип, цель — тип + эффект региона (B, соседи своего региона, точнее A; C —
    случайные МО других регионов — хуже A); ``only_b``: цель — только тип, дециль C совпадает с типом
    (C ≈ A), B без типа хуже A."""
    rng = np.random.default_rng(seed)
    n = 400
    groups = rng.integers(0, 10, n)
    types = rng.integers(1, 5, n)
    if case == "only_c":
        types = groups % 4 + 1
    F = rng.normal(size=(n, 3)) * 0.1 + types[:, None] * 1.5
    xy = rng.normal(size=(n, 2))
    if case == "only_c":
        g_eff = rng.normal(0, 1.0, 10)
        y = types * 5.0 + g_eff[groups] + rng.normal(0, 0.3, n)
        dec = P.value_quantiles(rng.normal(size=n), 10)
    else:
        y = types * 2.0 + rng.normal(0, 0.3, n)
        dec = types.copy()
    st = EX.t7_sets(F, xy, groups, types, dec, y, y, 10, 5, 5, np.random.default_rng(1))
    return EX.t7_eval(st, 199, 0.95, 0.0, np.random.default_rng(2))


def test_t7_partial_better_than_only_one_of_b_and_c():
    r = _t7_partial_world("only_c")
    assert r["verdict_A"] == "partial" and r["against_C_A"] and not r["against_B_A"]
    r = _t7_partial_world("only_b")
    assert r["verdict_A"] == "partial" and r["against_B_A"] and not r["against_C_A"]


def test_scope_population_checks_the_phrase_and_report_marks_failure():
    from munnet.interpret import report as R

    ter = pd.DataFrame(
        {"territory_id": [1, 2, 3, 4, 5], "series_status": ["full", "full", "full", "partial", "only_2023"]}
    )
    ctx = pd.DataFrame(
        {
            "territory_id": [1, 1, 2, 3, 4, 5],
            "year": [2023, 2024, 2023, 2023, 2024, 2023],
            "pop_avg": [100.0, 120.0, 50.0, 80.0, 10.0, 20.0],
        }
    )
    r = ST.scope_population(ter, ctx)
    assert (r["pop_median_full"], r["pop_median_partial"], r["phrase_holds"]) == (80.0, 15.0, True)
    assert "не выполнено" not in R._scope_note(r)
    ctx.loc[ctx["territory_id"] == 4, "pop_avg"] = 500.0
    ctx.loc[ctx["territory_id"] == 5, "pop_avg"] = 900.0
    r = ST.scope_population(ter, ctx)
    assert not r["phrase_holds"] and "не выполнено" in R._scope_note(r)
    # группа сравнения — территориальные узлы: внутригородские МО с полным рядом (не типизированы)
    # в неё не входят, их медиана — только справка
    ter = pd.DataFrame(
        {
            "territory_id": [1, 2, 3, 4, 5, 6, 7],
            "series_status": ["full", "full", "full", "partial", "only_2023", "full", "full"],
            "is_inner_city": [False, False, False, False, False, True, True],
        }
    )
    ctx = pd.DataFrame(
        {
            "territory_id": [1, 2, 3, 4, 5, 6, 7],
            "year": [2023] * 7,
            "pop_avg": [100.0, 50.0, 80.0, 60.0, 70.0, 1.0, 2.0],
        }
    )
    r = ST.scope_population(ter, ctx)
    assert (r["pop_median_full"], r["pop_median_partial"], r["pop_median_full_all"]) == (80.0, 65.0, 50.0)
    assert r["phrase_holds"] and r["pop_n_full"] == 3 and r["pop_n_inner_full"] == 2
    note = R._scope_note(r)
    assert "у территориальных узлов сети" in note and "справка" in note
    assert R._t4_note({"text_holds": True}) == ""
    assert "не выполнено" in R._t4_note({"text_holds": False, "ari_basket": 0.1, "ari_place": 0.2})


def test_variant_names_are_checked_against_clustering_variants():
    ST.check_variant_names(["a", "b"], ["a", "b", "c"])
    with pytest.raises(ValueError, match="robustness.variants"):
        ST.check_variant_names(["a", "zzz"], ["a", "b"])


# --- Охват фразы и поузловые выгрузки (описание, не проверка) ---------------------------------------


def test_untyped_counts_city_members_as_typed():
    from munnet.interpret import node_tables as NT
    from munnet.interpret import report as R

    # 1, 2 — свои узлы с типом; 3 — свой узел без типа (неполный ряд); 11, 12 — районы города-узла 900
    # (12 — с неполным рядом, тип — через город); 13 — нет строки в members
    members = pd.DataFrame(
        {
            "territory_id": [1, 2, 3, 11, 12],
            "node_id": pd.array([1, 2, 3, 900, 900], dtype="Int32"),
            "role": ["self", "self", "self", "city_member", "city_member"],
        }
    )
    ids = np.array([1, 2, 3, 11, 12, 13])
    m = NT.untyped_mask(ids, members, np.array([1, 2, 900]))
    assert m.tolist() == [False, False, True, False, False, True]
    assert NT.untyped_mask(ids, None, np.array([1, 2, 900])).tolist() == [
        False,
        False,
        True,
        True,
        True,
        True,
    ]
    # сверка населения — по МО без типа, если они заданы
    ter = pd.DataFrame(
        {
            "territory_id": [1, 2, 3, 12],
            "series_status": ["full", "full", "partial", "partial"],
            "is_inner_city": [False, False, False, True],
        }
    )
    ctx = pd.DataFrame(
        {"territory_id": [1, 2, 3, 12], "year": [2023] * 4, "pop_avg": [100.0, 80.0, 10.0, 500.0]}
    )
    assert ST.scope_population(ter, ctx)["pop_median_partial"] == 255.0
    r = ST.scope_population(ter, ctx, np.array([3]))
    assert r["pop_median_partial"] == 10.0 and r["pop_n_partial"] == 1 and r["phrase_holds"]
    # подпись медианы — по той же группе (МО без типа, n), а не «МО с неполным рядом»
    note = R._scope_note(r)
    assert "медиана населения МО без типа (n = 1) — 10" in note
    assert "МО с неполным рядом —" not in note
    # отклонение от комментария записи названо прямо, с обоими числами
    note = R._scope_note({**r, "n_partial": 1, "n_partial_series": 2, "n_partial_typed_via_city": 1})
    assert "отклонение от комментария записи" in note
    assert "МО с неполным рядом — 2, из них 1 типизировано через узел-город" in note


def test_margin_table_matches_borderline_examples():
    rng = np.random.default_rng(0)
    Z = np.vstack([rng.normal(c, 1.0, size=(30, 2)) for c in (0.0, 3.0, 6.0)])
    types = np.repeat([2, 1, 3], 30)
    ids = np.arange(100, 190)
    order = [2, 1, 3]
    mt = DS.margin_table(Z, types, ids, order).set_index("territory_id")
    ex = DS.typical_examples(Z, types, ids, np.arange(90), np.ones(90, bool), order, 1, 5, True)
    b = ex.loc[ex["kind"] == "borderline"]
    assert len(b) == 15
    for _, r in b.iterrows():
        t = int(ids[int(r["pos"])])
        assert mt.loc[t, "second_type"] == int(r["second_type"])
        assert mt.loc[t, "margin"] == pytest.approx(float(r["distance"]))
    assert (mt["margin"] >= 0).all() and (mt["second_type"] != mt["type"]).sum() == len(mt)


def test_r1_tables_compare_with_main_and_city_node():
    from munnet.interpret import node_tables as NT

    main_ids = np.array([1, 2, 900])
    main_t = np.array([1, 2, 3])
    runs = {
        "variant:a": ("variant", np.array([1, 2, 900]), np.array([1, 1, 3])),
        "variant:nodes": ("variant", np.array([1, 2, 11]), np.array([1, 2, 2])),  # 11 — район города 900
        "seed:142": ("seed", np.array([1, 2, 900]), np.array([2, 2, 3])),
    }
    long, summ = NT.r1_tables(main_ids, main_t, runs, {11: 900})
    assert len(long) == 9
    r11 = long.loc[long["territory_id"] == 11].iloc[0]
    assert r11["main_type"] == 3 and not r11["same"]
    s = summ.set_index(["territory_id", "kind"])
    assert (s.loc[(1, "variant"), "n_same"], s.loc[(1, "variant"), "n_runs"]) == (2, 2)
    assert (s.loc[(2, "variant"), "n_same"], s.loc[(2, "variant"), "n_runs"]) == (1, 2)
    assert (s.loc[(1, "seed"), "n_same"], s.loc[(1, "seed"), "n_runs"]) == (0, 1)
    assert (s.loc[(900, "variant"), "n_runs"], s.loc[(11, "variant"), "n_same"]) == (1, 0)


def test_comparable_and_rival_tables():
    from munnet.interpret import node_tables as NT

    ids = np.array([10, 20, 30])
    F = np.array([[0.0, 0.0], [3.0, 4.0], [0.0, 1.0]])
    xy = np.array([[0.0, 0.0], [3000.0, 4000.0], [0.0, 1000.0]])
    members = {"A": [np.array([2, 1]), np.array([], dtype=int), np.array([0])], "B": [np.array([1]), [], []]}
    t = NT.comparable_table(ids, members, F, xy, "A", ("A", "B"))
    a = t.loc[(t["territory_id"] == 10) & (t["set"] == "A")]
    assert a["other_id"].tolist() == [30, 20] and a["rank"].tolist() == [1, 2]
    assert a["distance"].tolist() == pytest.approx([1.0, 5.0]) and a["product"].all()
    assert a["km"].tolist() == pytest.approx([1.0, 5.0])
    b = t.loc[t["set"] == "B"].iloc[0]
    assert (b["other_id"], b["distance"], b["unit"], b["product"]) == (20, pytest.approx(5.0), "km", False)
    p1 = Partition("region", "регион", np.array([1, 1, 2]), "reference")
    p2 = Partition("wage", "по зарплате", np.array([3, 2, 1]), "sized")
    r = NT.rival_table(ids, [("T5:max_ami", p1), ("T5:best_rival:x", p2), ("T1:best_rival:x", p2)])
    assert len(r) == 6
    w = r.loc[r["partition"] == "wage"]
    assert w["sources"].iloc[0] == "T5:best_rival:x;T1:best_rival:x"
    assert w["best_partition_label"].tolist() == [3, 2, 1]
