"""Справка usefulness_level (ρ уровня трат рядом с T1 (a)): функции на игрушечных данных с известным ρ,
сверка с t1_runs.csv и место вызова в usefulness.run. Настоящие данные не читаются."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from munnet import usefulness as U
from munnet import usefulness_level as L
from munnet.contracts import QCError


def _boots(n: int, b: int, seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [rng.integers(0, n, n) for _ in range(b)]


def test_spearman_ci_known_values():
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    rho, ci, draws = L.spearman_ci(np.array([10, 20, 30, 40, 50]), y, [np.arange(5)] * 3, 0.95)
    assert rho == pytest.approx(1.0)
    assert ci == pytest.approx((1.0, 1.0))
    assert draws == pytest.approx([1.0, 1.0, 1.0])
    rho, *_ = L.spearman_ci(np.array([5, 4, 3, 2, 1]), y, [np.arange(5)], 0.95)
    assert rho == pytest.approx(-1.0)
    # ρ = 1 − 6·Σd² / (n(n² − 1)): ранги z 2, 1, 4, 3, 5 против 1…5 — Σd² = 4, ρ = 1 − 24/120 = 0,8
    rho, *_ = L.spearman_ci(np.array([2.0, 1.0, 4.0, 3.0, 5.0]), y, [np.arange(5)], 0.95)
    assert rho == pytest.approx(0.8)


def test_spearman_ci_matches_scipy_with_ties():
    rng = np.random.default_rng(1)
    y = rng.normal(size=200)
    z = np.round(y + rng.normal(size=200), 0)  # ступени с ничьими, как у деления на группы
    boots = _boots(200, 50)
    rho, ci, draws = L.spearman_ci(z, y, boots, 0.95)
    assert rho == pytest.approx(spearmanr(z, y).statistic)
    assert draws[7] == pytest.approx(spearmanr(z[boots[7]], y[boots[7]]).statistic)
    assert ci[0] <= rho <= ci[1]


def test_compare_paired_differences():
    rng = np.random.default_rng(2)
    y = rng.normal(size=300)
    level = y + 0.3 * rng.normal(size=300)  # сильная связь
    types = np.digitize(y + 1.5 * rng.normal(size=300), [-1, 0, 1]) + 1  # слабее и ступенями 1…4
    boots = _boots(300, 200, 3)
    res = L.compare({"types": types, "level": level}, y, boots, 0.95)
    assert res["n"] == 300
    d = res["diff_level_minus_types"]
    assert d["point"] == pytest.approx(res["level"]["rho"] - res["types"]["rho"])
    assert d["ci"][0] > 0  # уровень упорядочивает цель сильнее ступеней на всех выборках
    assert "diff_types_minus_types" not in res


def _t1_main(res):
    return pd.DataFrame(
        [
            {
                "run": "main",
                "turnover": "retail",
                "n_a": res["n"],
                "rho_a": res["types"]["rho"],
                "rho_a_lo": res["types"]["ci"][0],
                "rho_a_hi": res["types"]["ci"][1],
            }
        ]
    )


def test_check_against_t1_passes_and_fails():
    rng = np.random.default_rng(4)
    y = rng.normal(size=100)
    res = L.compare({"types": np.digitize(y, [-1, 0, 1]), "level": y}, y, _boots(100, 30), 0.95)
    assert L.check_against_t1({"retail": res}, _t1_main(res))["retail"] == 0.0
    bad = _t1_main(res)
    bad["rho_a"] += 1e-6
    with pytest.raises(QCError, match="расходятся с t1_runs.csv"):
        L.check_against_t1({"retail": res}, bad)
    bad = _t1_main(res)
    bad["n_a"] += 1
    with pytest.raises(QCError, match="n_a"):
        L.check_against_t1({"retail": res}, bad)
    with pytest.raises(QCError, match="нет строки оборота"):
        L.check_against_t1({"catering": res}, _t1_main(res))


def test_usefulness_run_calls_level_last(tmp_path, monkeypatch):
    """Справка вызывается из usefulness.run один раз и после всех прежних блоков: их файлы уже записаны."""
    from pathlib import Path

    from test_usefulness import _synthetic

    cfg, *_ = _synthetic(tmp_path, np.random.default_rng(5))
    out = Path(cfg["paths"]["outputs"]) / "usefulness"
    seen: list[set[str]] = []
    monkeypatch.setattr(L, "run", lambda c: seen.append({p.name for p in out.iterdir()}))
    U.run(cfg)
    assert len(seen) == 1
    assert {"facts.json", "by_type.json", "size_check.json", "size_by_mo.csv"} <= seen[0]
