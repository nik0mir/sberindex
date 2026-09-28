"""Функции отслеживания типов во времени — на маленьких разбиениях с известным ответом (без data/)."""

from __future__ import annotations

import numpy as np
import pytest

from munnet.dynamics import tracking as T


def test_jaccard_matrix_known_values():
    a = np.array([0, 0, 1, 1])
    b = np.array([0, 1, 1, 1])
    J = T.jaccard_matrix(a, b, 2, 2)
    assert J[0, 0] == pytest.approx(1 / 2)  # {0,1} и {0}
    assert J[1, 1] == pytest.approx(2 / 3)  # {2,3} и {1,2,3}
    assert J[0, 1] == pytest.approx(1 / 4)


def test_match_recovers_permutation():
    rng = np.random.default_rng(0)
    ref = rng.integers(0, 4, 200)
    perm = np.array([2, 0, 3, 1])
    other = perm[ref]
    mapping = T.match(ref, other, 4)
    assert np.array_equal(T.relabel(other, mapping), ref)


def test_match_with_noise_keeps_majority():
    rng = np.random.default_rng(1)
    ref = np.repeat(np.arange(3), 50)
    other = (ref + 1) % 3
    flip = rng.choice(150, 10, replace=False)
    other[flip] = rng.integers(0, 3, 10)
    got = T.relabel(other, T.match(ref, other, 3))
    assert (got == ref).mean() >= 140 / 150


def test_share_changed_is_label_invariant():
    a = np.array([0, 0, 1, 1, 2, 2])
    b = np.array([1, 1, 2, 2, 0, 0])
    assert T.share_changed(a, b, 3) == 0.0
    b2 = np.array([1, 1, 2, 0, 0, 0])
    assert T.share_changed(a, b2, 3) == pytest.approx(1 / 6)


def test_transition_matrix_counts():
    a = np.array([0, 0, 1, 1])
    b = np.array([0, 1, 1, 1])
    M = T.transition_matrix(a, b, 2)
    assert M.tolist() == [[1, 1], [0, 2]]


def _events(a, b, k, tau):
    ev = T.monic_events(a, b, k, k, tau)
    return {(e.kind, e.before, e.after) for e in ev}


def test_monic_survive():
    a = np.array([0] * 10 + [1] * 10)
    b = a.copy()
    ev = _events(a, b, 2, 0.5)
    assert ev == {("survived", (0,), (0,)), ("survived", (1,), (1,))}


def test_monic_split_and_merge():
    a = np.array([0] * 10 + [1] * 10 + [2] * 10)
    # тип 0 делится пополам на новые 0 и 1; типы 1 и 2 сливаются в новый 2
    b = np.array([0] * 5 + [1] * 5 + [2] * 20)
    ev = _events(a, b, 3, 0.5)
    assert ("split", (0,), (0, 1)) in ev
    assert ("merged", (1, 2), (2,)) in ev
    assert not any(k == "survived" for k, _, _ in ev)


def test_monic_emerge_and_disappear():
    a = np.array([0] * 12 + [1] * 12)
    # при высоком пороге ни одна пара не проходит: старые типы исчезли, новые возникли
    b = np.array([0] * 8 + [2] * 4 + [1] * 8 + [2] * 4)
    ev = T.monic_events(a, b, 2, 3, 0.7)
    kinds = sorted(e.kind for e in ev)
    assert "emerged" in kinds and "disappeared" in kinds


def test_monic_threshold_sensitivity():
    a = np.array([0] * 10 + [1] * 10)
    b = np.array([0] * 7 + [1] * 13)  # 0: J = 0,7; 1: J = 10/13
    assert {e.kind for e in T.monic_events(a, b, 2, 2, 0.5)} == {"survived"}
    assert "disappeared" in {e.kind for e in T.monic_events(a, b, 2, 2, 0.75)}


def test_stable_halves():
    # узлы: 0 — надёжный переход 0→1; 1 — половины 2023 разошлись; 2 — без смены; 3 — надёжно без смены
    o23 = np.array([0, 0, 1, 1])
    e23 = np.array([0, 1, 1, 1])
    o24 = np.array([1, 1, 1, 1])
    e24 = np.array([1, 1, 1, 1])
    rel, t23, t24 = T.stable_halves(o23, e23, o24, e24)
    assert rel.tolist() == [True, False, False, False]
    assert t23.tolist() == [0, -1, 1, 1]
    assert t24.tolist() == [1, 1, 1, 1]


def test_paired_bootstrap_detects_change_over_noise():
    rng = np.random.default_rng(2)
    n, k = 600, 3
    y23 = rng.integers(0, k, n)
    y24 = y23.copy()
    move = rng.choice(n, 150, replace=False)  # 25% сменили тип
    y24[move] = (y24[move] + 1) % k
    noise = [(y23, np.where(rng.random(n) < 0.03, (y23 + 1) % k, y23))]
    res = T.paired_bootstrap(y23, y24, noise, k, 200, np.random.default_rng(3))
    assert res.diff == pytest.approx(res.change - res.noise)
    assert res.lo > 0.15 and res.exceeds
    same = T.paired_bootstrap(y23, y23, noise, k, 200, np.random.default_rng(3))
    assert not same.exceeds


def test_paired_bootstrap_deterministic():
    rng = np.random.default_rng(4)
    a, b, c = (rng.integers(0, 3, 100) for _ in range(3))
    r1 = T.paired_bootstrap(a, b, [(a, c)], 3, 50, np.random.default_rng(9))
    r2 = T.paired_bootstrap(a, b, [(a, c)], 3, 50, np.random.default_rng(9))
    assert r1 == r2


def test_procrustes_undoes_rotation():
    rng = np.random.default_rng(5)
    E = rng.normal(size=(100, 3))
    Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    aligned, fit = T.procrustes(E @ Q, E)
    assert np.allclose(aligned, E, atol=1e-10)
    assert fit == pytest.approx(1.0)


def test_nearest_center():
    C = np.array([[0.0, 0.0], [10.0, 0.0]])
    Z = np.array([[1.0, 0.0], [9.0, 1.0], [4.0, 0.0]])
    assert T.nearest(Z, C).tolist() == [0, 1, 0]


def test_evolutionary_kmeans_smooths_toward_history():
    rng = np.random.default_rng(6)
    prev = np.array([[0.0, 0.0], [5.0, 0.0]])
    Z = np.vstack([rng.normal([0, 0], 0.3, (50, 2)), rng.normal([7, 0], 0.3, (50, 2))])
    equal = np.array([50, 50])
    lab0, C0 = T.evolutionary_kmeans(Z, prev, equal, 0.0)
    lab1, C1 = T.evolutionary_kmeans(Z, prev, equal, 0.5)
    assert np.array_equal(lab0, lab1)
    assert C0[1, 0] == pytest.approx(Z[50:, 0].mean(), abs=1e-9)  # cp = 0 — обычный K-means
    # равные размеры (γ = 1/2): прежнее правило cp·c' + (1 − cp)·m
    assert C1[1, 0] == pytest.approx(0.5 * 5.0 + 0.5 * Z[50:, 0].mean(), abs=1e-9)
    lab2, _ = T.evolutionary_kmeans(Z, prev, equal, 0.5)
    assert np.array_equal(lab1, lab2)  # детерминизм


def test_evo_center_matches_paper_weights():
    c_prev, m = np.array([0.0]), np.array([1.0])
    # равные размеры: γ = 1/2, центр = cp·c' + (1 − cp)·m
    assert T.evo_center(c_prev, m, 10, 10, 0.3)[0] == pytest.approx(0.7)
    # неравные: γ = 30 / 40, веса (1 − γ)·cp = 0,075 и γ·(1 − cp) = 0,525, нормированы к сумме 1
    assert T.evo_center(c_prev, m, 30, 10, 0.3)[0] == pytest.approx(0.525 / 0.6)
    # чем больше прошлый кластер, тем сильнее тянет история
    assert T.evo_center(c_prev, m, 10, 30, 0.3)[0] < T.evo_center(c_prev, m, 30, 10, 0.3)[0]
    assert T.evo_center(c_prev, m, 10, 30, 0.0)[0] == pytest.approx(1.0)
    assert T.evo_center(c_prev, m, 10, 30, 1.0)[0] == pytest.approx(0.0)
