"""Разведка usefulness.size_posthoc: функции на синтетике с известным ответом, сквозной прогон
на синтетических выходах interpret во временной папке, неизменность прежних выходов этапа. Настоящие данные
не читаются."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from test_usefulness import _synthetic

from munnet import usefulness as U
from munnet import usefulness_by_type as B
from munnet import usefulness_size as Z
from munnet.config import load_config
from munnet.contracts import QCError

BLOCK = load_config()["usefulness"]["size_posthoc"]


def _copy(x):
    return json.loads(json.dumps(x))


# --- спецификация -------------------------------------------------------------------------------------------


def test_config_block_matches_implementation():
    Z.check_spec(BLOCK)
    assert "by_type_test" in load_config()["usefulness"]  # подблок рядом, а не внутри by_type_test
    assert "size_posthoc" not in load_config()["usefulness"]["by_type_test"]


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("quintiles", "n"), 4),
        (("quintiles", "large"), "top2"),
        (("size", "year"), 2024),
        (("statistic", "bootstrap", "n"), 1000),
        (("shared_members_bootstrap", "n"), 500),
        (("statistic", "type_given_size", "permutation", "within"), "region_group"),
        (("status",), "test"),
    ],
)
def test_check_spec_rejects_other_rule(path, value):
    bad = _copy(BLOCK)
    node = bad
    for k in path[:-1]:
        node = node[k]
    node[path[-1]] = value
    with pytest.raises(QCError, match="не совпадает"):
        Z.check_spec(bad)


def test_check_spec_requires_block():
    with pytest.raises(QCError, match="нет в конфиге"):
        Z.check_spec(None)


# --- квинтили и ячейки --------------------------------------------------------------------------------------


def test_quintiles_bounds_and_assignment_like_qcut():
    v = np.arange(1, 11, dtype=float)  # 1…10
    b = Z.quantile_bounds(v, 5)
    assert b == pytest.approx([2.8, 4.6, 6.4, 8.2])
    q = Z.assign_quantile(v, b)
    assert q.tolist() == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
    rng = np.random.default_rng(0)
    w = rng.lognormal(10, 1, 997)
    assert np.array_equal(Z.assign_quantile(w, Z.quantile_bounds(w, 5)), pd.qcut(w, 5, labels=False))
    # на логарифме — то же деление (монотонное преобразование)
    assert np.array_equal(Z.assign_quantile(w, Z.quantile_bounds(w, 5)), pd.qcut(np.log(w), 5, labels=False))
    # значение на границе — в нижний квинтиль; вне базы — по тем же границам; пропуск — −1
    assert Z.assign_quantile(np.array([2.8, 2.80001, 0.0, 100.0, np.nan]), b).tolist() == [0, 1, 0, 4, -1]
    with pytest.raises(QCError, match="нет населения"):
        Z.quantile_bounds(np.array([1.0, np.nan]), 5)


def test_cell_codes_pairs():
    a = np.array([5, 5, 7, 7, 5])
    b = np.array([0, 1, 0, 0, 0])
    c = Z.cell_codes(a, b)
    assert c[0] == c[4] and c[2] == c[3] and len({c[0], c[1], c[2]}) == 3


def test_permutation_within_cells_moves_labels_only_inside_cell():
    rng = np.random.default_rng(1)
    groups = rng.integers(0, 6, 300)
    q = rng.integers(0, 5, 300)
    cell = Z.cell_codes(groups, q)
    perm = B.permutation_index(cell, 200, 42)
    assert (cell[perm] == cell[None, :]).all()  # источник метки — из той же ячейки
    assert all(sorted(row) == list(range(300)) for row in perm[:5])  # перестановка, не выборка
    labels = rng.integers(1, 5, 300)
    for k in np.unique(cell):  # состав меток в каждой ячейке сохраняется
        m = cell == k
        assert sorted(labels[perm[0]][m]) == sorted(labels[m])
    assert (labels[perm] != labels[None, :]).any()  # что-то действительно перемешалось


# --- статистики ---------------------------------------------------------------------------------------------


def _data(masks: dict, works: dict) -> dict:
    return {
        t: {"mask": np.asarray(masks[t]), "works": np.asarray(works[t]) & np.asarray(masks[t])} for t in masks
    }


def test_mean_share_and_delta_known_answer():
    hi = np.array([1, 1, 1, 1, 0, 0, 0, 0], bool)
    lo = ~hi
    masks = {t: np.ones(8, bool) for t in Z.TARGETS}
    masks["ndfl"] = np.array([1, 1, 0, 0, 1, 1, 1, 1], bool)
    works = {
        "catering": np.array([1, 0, 0, 0, 1, 1, 1, 0], bool),  # hi 1/4, lo 3/4
        "ndfl": np.array([1, 1, 1, 1, 1, 0, 0, 0], bool),  # hi 2/2, lo 1/4 (МО 3–4 вне цели)
        "shipments": np.array([1, 1, 0, 0, 1, 1, 0, 0], bool),  # hi 1/2, lo 1/2
    }
    d = _data(masks, works)
    assert Z.mean_share(B.point_sum, hi, d, Z.TARGETS) == pytest.approx((0.25 + 1.0 + 0.5) / 3)
    assert Z.mean_delta(B.point_sum, hi, lo, d, Z.TARGETS) == pytest.approx(((0.25 - 0.75) + 0.75 + 0) / 3)


def test_mean_strat_delta_harmonic_known_answer():
    # одна цель повторена трижды: страта 1 — hi 2 МО (доля 1), lo 2 МО (0): Δ 1, h = 1;
    # страта 2 — hi 1 (0), lo 3 (1):
    # Δ −1, h = 3/4; страта 3 — только lo: h = 0. Итог (1·1 − 0,75)/1,75
    hi = np.array([1, 1, 0, 0, 1, 0, 0, 0, 0], bool)
    lo = np.array([0, 0, 1, 1, 0, 1, 1, 1, 1], bool)
    w = np.array([1, 1, 0, 0, 0, 1, 1, 1, 1], bool)
    s1 = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0], bool)
    s2 = np.array([0, 0, 0, 0, 1, 1, 1, 1, 0], bool)
    s3 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1], bool)
    d = _data({t: np.ones(9, bool) for t in Z.TARGETS}, {t: w for t in Z.TARGETS})
    got = Z.mean_strat_delta(B.point_sum, hi, lo, d, Z.TARGETS, [s1, s2, s3])
    assert got == pytest.approx((1.0 - 0.75) / 1.75)


def test_empty_draws_and_perm_summary():
    gidx = np.array([0, 0, 1, 1])
    counts = np.array([[1.0, 1.0], [2.0, 0.0], [0.0, 2.0]])
    S = B.boot_sum(gidx, counts)
    hi = np.array([1, 0, 0, 0], bool)  # hi только в группе 0
    lo = ~hi
    d = _data({t: np.ones(4, bool) for t in Z.TARGETS}, {t: np.ones(4, bool) for t in Z.TARGETS})
    assert Z.empty_draws(S, hi, lo, d, Z.TARGETS).tolist() == [False, False, True]
    s = Z.perm_summary(np.array([-1.0, 0.0, 1.0, np.nan]), 0.0)
    assert s["p_less"] == pytest.approx((2 + 1 + 1) / 5) and s["null_mean"] == pytest.approx(0.0)


# --- бутстрап с общими членами D ----------------------------------------------------------------------------


def test_weighted_median_rows_equals_repeat_median():
    rng = np.random.default_rng(3)
    for _ in range(200):
        k = rng.integers(1, 9)
        y = np.sort(rng.normal(size=k))
        w = rng.integers(0, 4, k)
        Y = np.r_[y, np.nan, np.nan][None, :]
        W = np.r_[w, 0, 0][None, :]
        got = Z.weighted_median_rows(Y, W)[0]
        if w.sum() == 0:
            assert np.isnan(got)
        else:
            assert got == pytest.approx(np.median(np.repeat(y, w)))


def test_d_member_matrix_sorted_groups_and_padding():
    y = pd.Series({10: 0.3, 11: 0.1, 12: 0.2, 13: np.nan, 20: 0.5})
    comp = pd.DataFrame(
        {
            "territory_id": [1, 1, 1, 1, 2, 2, 1],
            "set": ["D", "D", "D", "D", "D", "B", "B"],
            "other_id": [10, 11, 12, 13, 20, 10, 11],
        }
    )
    member_group = pd.Series({10: 7, 11: 8, 12: 99, 13: 7, 20: 8})
    Y, G = Z.d_member_matrix(y, comp, np.array([1, 2]), member_group, np.array([7, 8]))
    assert Y.shape == (2, 3)
    assert Y[0].tolist() == pytest.approx([0.1, 0.2, 0.3])  # по возрастанию, член без цели выброшен
    assert G[0].tolist() == [1, 2, 0]  # 11 → группа 8 (индекс 1), 12 — вне общего набора (2), 10 → 7 (0)
    assert Y[1, 0] == pytest.approx(0.5) and np.isnan(Y[1, 1:]).all() and G[1].tolist() == [1, 3, 3]


def _naive_shared(prep_raw, groups, counts_row, hi, lo):
    """Эталон как da5.py: циклы, np.repeat и np.median."""
    ug = np.unique(groups)
    pos = {g: i for i, g in enumerate(ug)}
    cnt = counts_row
    ds = []
    for t in Z.TARGETS:
        own, eB, mask, members = prep_raw[t]
        wv = np.zeros(len(own))
        ok = np.zeros(len(own), bool)
        for i in np.flatnonzero(mask & (cnt[[pos[g] for g in groups]] > 0)):
            yv, gv = members[i]
            r = np.array([cnt[pos[g]] if g in pos else 1 for g in gv], dtype=int)
            if r.sum() == 0:
                continue
            eD = abs(own[i] - np.median(np.repeat(yv, r)))
            ok[i] = True
            wv[i] = eB[i] < eD
        wt = cnt[[pos[g] for g in groups]] * ok
        ds.append((wt * wv * hi).sum() / (wt * hi).sum() - (wt * wv * lo).sum() / (wt * lo).sum())
    return np.mean(ds)


def _shared_case(seed=0, n=80, n_groups=8):
    rng = np.random.default_rng(seed)
    ids = np.arange(1, n + 1)
    groups = (ids - 1) % n_groups
    member_group = pd.Series(np.r_[groups, np.full(10, 99)], index=np.r_[ids, np.arange(1001, 1011)])
    hi = rng.random(n) < 0.4
    lo = ~hi
    prep, raw = {}, {}
    for t in Z.TARGETS:
        y = pd.Series(rng.normal(size=n + 10), index=member_group.index)
        rows = []
        for i in ids:
            pool = [j for j in member_group.index if member_group[j] != groups[i - 1]]
            rows += [(i, "D", j) for j in rng.choice(pool, 6, replace=False)]
        comp = pd.DataFrame(rows, columns=["territory_id", "set", "other_id"])
        Y, G = Z.d_member_matrix(y, comp, ids, member_group, np.unique(groups))
        own = y.reindex(ids).to_numpy()
        eB = np.abs(rng.normal(0, 0.8, n))
        mask = rng.random(n) < 0.9
        prep[t] = {"mask": mask, "own": own, "err_B": eB, "Y": Y, "G": G}
        members = {}
        for k, i in enumerate(ids):
            sel = comp.loc[comp["territory_id"] == i, "other_id"].to_numpy()
            members[k] = (y.reindex(sel).to_numpy(), member_group.reindex(sel).to_numpy())
        raw[t] = (own, eB, mask, members)
    return ids, groups, hi, lo, prep, raw


def test_shared_bootstrap_equals_naive_and_resamples_members_with_regions():
    ids, groups, hi, lo, prep, raw = _shared_case()
    gidx, counts = B.bootstrap_counts(groups, 25, 42)
    got = Z.shared_bootstrap(prep, gidx, counts, {"x": (hi, lo)}, Z.TARGETS, tol=1e-12)["x"]
    want = [_naive_shared(raw, groups, counts[b], hi, lo) for b in range(len(counts))]
    assert got == pytest.approx(want, abs=1e-12)
    # при единичных весах — обычная Δ по исходным медианам D
    one = Z.shared_bootstrap(
        prep, gidx, np.ones((1, counts.shape[1])), {"x": (hi, lo)}, Z.TARGETS, tol=1e-12
    )["x"][0]
    plain = []
    for t in Z.TARGETS:
        p = prep[t]
        med = np.array([np.nanmedian(r) for r in p["Y"]])
        w = p["mask"] & (p["err_B"] < np.abs(p["own"] - med))
        plain.append(w[p["mask"] & hi].mean() - w[p["mask"] & lo].mean())
    assert one == pytest.approx(np.mean(plain), abs=1e-12)


def _naive_share(prep_raw, groups, counts_row, member):
    """Эталон доли «B ближе D» группы ``member`` (среднее по целям) при весах групп ``counts_row``."""
    ug = np.unique(groups)
    pos = {g: i for i, g in enumerate(ug)}
    cnt = counts_row
    out = []
    for t in Z.TARGETS:
        own, eB, mask, members = prep_raw[t]
        wv = np.zeros(len(own))
        ok = np.zeros(len(own), bool)
        for i in np.flatnonzero(mask & (cnt[[pos[g] for g in groups]] > 0)):
            yv, gv = members[i]
            r = np.array([cnt[pos[g]] if g in pos else 1 for g in gv], dtype=int)
            if r.sum() == 0:
                continue
            ok[i] = True
            wv[i] = eB[i] < abs(own[i] - np.median(np.repeat(yv, r)))
        wt = cnt[[pos[g] for g in groups]] * ok
        out.append((wt * wv * member).sum() / (wt * member).sum())
    return np.mean(out)


def test_shared_bootstrap_shares_equal_naive_and_keep_deltas():
    # доли с общими членами D (добавлено 02.10): как эталон; разности при этом не меняются
    ids, groups, hi, lo, prep, raw = _shared_case(seed=3)
    gidx, counts = B.bootstrap_counts(groups, 20, 7)
    shares = {"lower": hi, "upper": lo}
    got = Z.shared_bootstrap(prep, gidx, counts, {"x": (hi, lo)}, Z.TARGETS, tol=1e-12, shares=shares)
    plain = Z.shared_bootstrap(prep, gidx, counts, {"x": (hi, lo)}, Z.TARGETS, tol=1e-12)
    assert np.array_equal(got["x"], plain["x"], equal_nan=True)
    for k, member in shares.items():
        want = [_naive_share(raw, groups, counts[b], member) for b in range(len(counts))]
        assert got[k] == pytest.approx(want, abs=1e-12)
    with pytest.raises(QCError, match="совпадают"):
        Z.shared_bootstrap(prep, gidx, counts, {"x": (hi, lo)}, Z.TARGETS, tol=1e-12, shares={"x": hi})


def test_shared_bootstrap_members_follow_their_region():
    # МО 0 (группа 0) с членами D из групп 1 и 2: группа 2 не взята — её член выпадает из медианы D
    prep = {
        t: {
            "mask": np.array([True, True, True]),
            "own": np.array([0.0, 0.0, 0.0]),
            "err_B": np.array([0.15, 1.0, 1.0]),
            "Y": np.array([[0.1, 0.9], [0.1, 0.9], [0.1, 0.9]]),
            "G": np.array([[1, 2], [1, 2], [1, 2]]),
        }
        for t in Z.TARGETS
    }
    gidx = np.array([0, 1, 2])
    hi = np.array([True, False, False])
    lo = ~hi
    # все группы по разу: медиана D 0,5 — у МО 0 err_B 0,15 < 0,5 — B ближе
    a = Z.shared_bootstrap(prep, gidx, np.array([[1.0, 1.0, 1.0]]), {"x": (hi, lo)}, Z.TARGETS, tol=1e-12)[
        "x"
    ][0]
    assert a == pytest.approx(1.0 - 0.0)
    # группа 2 не взята: член 0,9 выпал, медиана D 0,1 — у МО 0 B уже не ближе; МО 2 (группа 2) — вес 0
    b = Z.shared_bootstrap(prep, gidx, np.array([[1.0, 2.0, 0.0]]), {"x": (hi, lo)}, Z.TARGETS, tol=1e-12)[
        "x"
    ][0]
    assert b == pytest.approx(0.0 - 0.0)
    # пустая группа hi — Δ не определена (выборка отбрасывается)
    c = Z.shared_bootstrap(prep, gidx, np.array([[0.0, 1.0, 2.0]]), {"x": (hi, lo)}, Z.TARGETS, tol=1e-12)[
        "x"
    ][0]
    assert np.isnan(c)


# --- сквозной прогон ----------------------------------------------------------------------------------------


def test_end_to_end_with_usefulness_run(tmp_path):
    cfg, *_ = _synthetic(tmp_path, np.random.default_rng(1))
    U.run(cfg)
    out = Path(cfg["paths"]["outputs"]) / "usefulness"
    f = json.loads((out / "size_check.json").read_text(encoding="utf-8"))
    by_type = json.loads((out / "by_type.json").read_text(encoding="utf-8"))
    assert f["kind"] == "posthoc_exploration" and "verdict" not in json.dumps(f)  # ключей вердикта нет
    assert f["n_base"] == 60 and [q["n_base"] for q in f["quintiles"]] == [12] * 5
    # население 1000 × id: границы — квантили 1000…60000
    assert f["quintile_bounds"] == pytest.approx(np.quantile(1000.0 * np.arange(1, 61), [0.2, 0.4, 0.6, 0.8]))
    assert f["large_vs_rest"]["n_large"] == 12
    assert f["qc"]["abs_diff"] <= 1e-12
    assert f["types_main"]["delta"] == pytest.approx(by_type["runs"]["main"]["delta"], abs=1e-12)
    assert f["shared_members_bootstrap"]["n_draws"] == 2000
    lo_, hi_ = f["large_vs_rest"]["delta_ci"]
    assert lo_ <= hi_
    for k in ("type_given_size", "size_given_type"):
        p = f[k]["permutation"]
        assert 0 < p["p_less"] <= 1 and p["n_perm"] == 2000
    # доля по квинтилю = среднее долей по трём целям
    q1 = f["quintiles"][0]
    # четыре нижних квинтиля вместе и интервалы с общими членами D (добавлено 02.10)
    lf = f["lower_four"]
    assert lf["quintiles"] == [1, 2, 3, 4] and lf["n"] == 48
    assert lf["share"] == pytest.approx(np.mean([lf["per_target"][t]["share"] for t in Z.TARGETS]))
    for row in (lf, *f["quintiles"]):
        assert {"share_ci_shared", "sd_boot_shared", "dropped_shared", "share_ci_shared_centered"} <= set(row)
        if row["share"] is None:  # пустая группа у цели в синтетике — доля не определена (null в JSON)
            continue
        shift = row["mean_boot_shared"] - row["share"]
        assert row["shift_shared"] == pytest.approx(shift)
        assert row["share_ci_shared_centered"] == pytest.approx([x - shift for x in row["share_ci_shared"]])
    assert lf["share_ci_shared_centered"][0] <= lf["share"] <= lf["share_ci_shared_centered"][1]
    assert q1["share"] == pytest.approx(np.mean([q1["per_target"][t]["share"] for t in Z.TARGETS]))
    by_mo = pd.read_csv(out / "size_by_mo.csv")
    assert len(by_mo) == 60 and by_mo["in_t7_common"].all()
    assert (by_mo["large"] == (by_mo["quintile"] == 5)).all()
    assert by_mo.sort_values("pop_avg_2023")["quintile"].is_monotonic_increasing
    assert by_mo["pop_avg_2023"].tolist() == pytest.approx((1000.0 * by_mo["territory_id"]).tolist())


def test_nodes_outside_common_get_quintile_by_same_bounds(tmp_path):
    cfg, *_ = _synthetic(tmp_path, np.random.default_rng(1))
    idir = Path(cfg["paths"]["outputs"]) / "interpret"
    e = pd.read_csv(idir / "t7_errors.csv")
    e.loc[e["territory_id"] > 50, "common"] = False  # 10 крупнейших — вне общего набора
    e.to_csv(idir / "t7_errors.csv", index=False)
    res = Z.compute(cfg, Z._read(cfg))
    by_mo = res["by_mo"]
    assert len(by_mo) == 60 and (~by_mo["in_t7_common"]).sum() == 10
    b = res["facts"]["quintile_bounds"]
    assert b == pytest.approx(np.quantile(1000.0 * np.arange(1, 51), [0.2, 0.4, 0.6, 0.8]))
    outside = by_mo.loc[~by_mo["in_t7_common"]]
    # больше верхней границы — верхний квинтиль
    assert (outside["quintile"] == 5).all() and outside["large"].all()
    assert res["facts"]["by_mo"]["n_large_outside_common"] == 10


def test_qc_fails_when_by_type_delta_differs(tmp_path):
    cfg, *_ = _synthetic(tmp_path, np.random.default_rng(1))
    U.run(cfg)
    out = Path(cfg["paths"]["outputs"]) / "usefulness"
    bt = json.loads((out / "by_type.json").read_text(encoding="utf-8"))
    bt["runs"]["main"]["delta"] += 1e-6
    with pytest.raises(QCError, match="by_type_test"):
        Z.run(cfg, bt)
    assert not (out / "size_check.json").exists()  # старый файл удалён до расчёта


def test_missing_population_column_is_input_error(tmp_path):
    cfg, *_ = _synthetic(tmp_path, np.random.default_rng(1))
    p = Path(cfg["paths"]["processed"]) / "context_annual.parquet"
    pd.read_parquet(p).drop(columns=["pop_avg"]).to_parquet(p)
    with pytest.raises(Exception, match="pop_avg"):
        Z.compute(cfg, Z._read(cfg))


def test_existing_outputs_unchanged(tmp_path, monkeypatch):
    """facts.json, rule_by_mo.csv, mo_flags.csv и by_type* с разведкой size_posthoc и без неё побайтно
    одинаковы."""
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    cfg_a, *_ = _synthetic(a, np.random.default_rng(5))
    cfg_b, *_ = _synthetic(b, np.random.default_rng(5))
    U.run(cfg_a)
    monkeypatch.setattr(Z, "run", lambda cfg, by_type_facts=None: None)
    U.run(cfg_b)
    oa = Path(cfg_a["paths"]["outputs"]) / "usefulness"
    ob = Path(cfg_b["paths"]["outputs"]) / "usefulness"
    assert (oa / "size_check.json").exists() and not (ob / "size_check.json").exists()
    names = (
        "facts.json",
        "rule_by_mo.csv",
        "mo_flags.csv",
        "by_type.json",
        "by_type_runs.csv",
        "by_type_by_mo.csv",
    )
    for name in names:  # в facts.json — только sha256 содержимого входов, путей нет: сравнение побайтное
        assert (oa / name).read_bytes() == (ob / name).read_bytes(), name
