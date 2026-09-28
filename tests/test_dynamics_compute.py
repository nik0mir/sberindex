"""Расчёт этапа dynamics на синтетике: способы отслеживания, проверка сюжета, параметры и схема истории."""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from munnet.config import Config, load_config
from munnet.contracts import SchemaError, write_table
from munnet.dynamics import compute as C
from munnet.dynamics import tracking as T
from munnet.dynamics.params import DynParams
from munnet.dynamics.schemas import DYNAMICS_HISTORY


def _windows(rng, n=300, k=3, n_windows=4, move=0.05):
    final = rng.integers(0, k, n)
    labs, cur = [], final.copy()
    for _ in range(n_windows):
        idx = rng.choice(n, int(move * n), replace=False)
        cur = cur.copy()
        cur[idx] = (cur[idx] + 1) % k
        labs.append(cur)
    return final, labs


def test_main_track_ignores_label_permutations():
    rng = np.random.default_rng(0)
    k = 3
    final, labs = _windows(rng, k=k)
    halves = {h: labs[0] if h.endswith("2023") else labs[-1] for h in C.HALVES}
    ref = C.main_track(final, labs, halves, k)
    perms = [rng.permutation(k) for _ in labs]
    shuffled = [p[lab] for p, lab in zip(perms, labs, strict=True)]
    hs = {h: rng.permutation(k)[v] for h, v in halves.items()}
    got = C.main_track(final, shuffled, hs, k)
    for a, b in zip(ref.rolling, got.rolling, strict=True):
        assert np.array_equal(a, b)
    for h in C.HALVES:
        assert np.array_equal(ref.halves[h], got.halves[h])
    assert got.info["chain_direct_windows_same"] == len(labs)


def test_change_vs_noise_and_reliable_matrix():
    rng = np.random.default_rng(1)
    n, k = 400, 3
    y23 = rng.integers(0, k, n)
    y24 = y23.copy()
    y24[:80] = (y24[:80] + 1) % k  # 20% сменили тип — надёжно (половины согласны)
    tr = C.Track(
        "refit_match", [y23, y24], {"odd_2023": y23, "even_2023": y23, "odd_2024": y24, "even_2024": y24}, {}
    )
    ch = C.change_vs_noise(tr, k, 100, np.random.default_rng(2), 0.95)
    assert ch.boot.change == pytest.approx(0.2)
    assert ch.boot.noise == 0.0 and ch.boot.exceeds
    assert ch.reliable.sum() == 80
    assert ch.reliable_matrix.sum() == 80 and np.trace(ch.reliable_matrix) == 0


def test_driver_test_finds_the_moving_part():
    rng = np.random.default_rng(3)
    n = 500
    parts = ("food", "marketplace", "transport", "health", "cafe", "other")
    dB = rng.normal(0, 0.05, (n, 6))
    tr = np.zeros(n, dtype=bool)
    tr[:60] = True
    dB[:60, 1] += 0.4  # у перешедших сильнее всего меняются маркетплейсы
    d = C.driver_test(dB, rng.normal(0, 0.02, n), parts, tr, ~tr, "marketplace", 0.05)
    assert d.significant and d.expectation_met and d.top_part == "marketplace"
    d2 = C.driver_test(dB, rng.normal(0, 0.02, n), parts, tr, ~tr, "cafe", 0.05)
    assert not d2.expectation_met


def test_evolutionary_track_deterministic_and_eps0_is_kmeans_from_history():
    rng = np.random.default_rng(4)
    Z = [np.vstack([rng.normal(c, 0.3, (40, 2)) for c in ((0, 0), (4, 0), (0, 4))]) for _ in range(3)]
    first = np.repeat(np.arange(3), 40)
    halves = {h: Z[0] for h in C.HALVES}
    a, _ = C.evolutionary_track(Z, halves, first, 3, 0.5, 100)
    b, _ = C.evolutionary_track(Z, halves, first, 3, 0.5, 100)
    for x, y in zip(a.rolling, b.rolling, strict=True):
        assert np.array_equal(x, y)
    assert all(T.share_changed(first, lab, 3) == 0 for lab in a.rolling)


def test_params_refuse_unregistered_variant():
    cfg = load_config()
    p = DynParams.from_config(cfg)
    assert p.main == "refit_match" and p.events_jaccard in p.taus and len(p.taus) == 3
    data = copy.deepcopy(cfg.data)
    data["dynamics"]["tracking"]["noise"] = "bootstrap"
    with pytest.raises(ValueError, match="noise"):
        DynParams.from_config(Config(data=data, path=cfg.path))


def test_history_schema(tmp_path):
    df = pd.DataFrame(
        {
            "territory_id": np.array([1, 1], dtype="int32"),
            "window": ["2023-01…2023-12", "2023-02…2024-01"],
            "window_index": np.array([1, 2], dtype="int8"),
            "first_date": ["2023-01", "2023-02"],
            "last_date": ["2023-12", "2024-01"],
            "type": np.array([1, 2], dtype="int16"),
            "type_24m": np.array([1, 1], dtype="int16"),
            "status": pd.Categorical(
                ["reliable", "reliable"], categories=["reliable", "within_noise", "no_change"]
            ),
            "reliable": [True, True],
            "is_city_node": [False, False],
        }
    )
    write_table(df, DYNAMICS_HISTORY, tmp_path / "h.parquet")
    bad = df.assign(reliable=[False, True])
    with pytest.raises(SchemaError):
        write_table(bad, DYNAMICS_HISTORY, tmp_path / "bad.parquet")
