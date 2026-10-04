"""Правило usefulness.example_small: функции на синтетике с известным ответом, сквозной прогон
на синтетических выходах interpret и usefulness во временной папке с независимым пересчётом выбора.
Настоящие данные не читаются."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from test_usefulness import _synthetic

from munnet import usefulness as U
from munnet import usefulness_example as E
from munnet.config import load_config
from munnet.contracts import MissingInputError, QCError

pytestmark = pytest.mark.usefixtures("stub_usefulness_level")

BLOCK = load_config()["usefulness"]["example_small"]


def _copy(x):
    return json.loads(json.dumps(x))


# --- спецификация -------------------------------------------------------------------------------------------


def test_config_block_matches_implementation():
    E.check_spec(BLOCK)
    assert BLOCK["recorded"].startswith("2026-10-04")


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("pool", "size"), "all"),
        (("pool", "require"), "any"),
        (("statistic", "score"), "mean_raw_d"),
        (("statistic", "pick"), "best"),
        (("statistic", "seen_target"), "retail"),
        (("words", "any_outcome"), "reselect"),
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
        E.check_spec(bad)


def test_check_spec_missing_block():
    with pytest.raises(QCError, match="подблока нет"):
        E.check_spec(None)


# --- функции правила ----------------------------------------------------------------------------------------


def test_small_mask_boundary_goes_down():
    bounds = np.array([10.0, 20.0, 30.0, 40.0])
    pop = np.array([5.0, 40.0, 40.0001, np.nan, 100.0])
    assert E.small_mask(pop, bounds).tolist() == [True, True, False, False, False]


def test_rank_scores_known_answer():
    # три цели, четыре МО; средние ранги по целям известны
    d = {
        "catering": np.array([-0.3, -0.1, 0.0, 0.2]),  # ранги 1 2 3 4
        "ndfl": np.array([0.5, -0.5, 0.1, 0.1]),  # ранги 4 1 2,5 2,5
        "shipments": np.array([-1.0, 0.0, 1.0, -2.0]),  # ранги 2 3 4 1
    }
    u, s = E.rank_scores(d)
    assert u["catering"] == pytest.approx(np.array([0.5, 1.5, 2.5, 3.5]) / 4)
    assert u["ndfl"] == pytest.approx(np.array([3.5, 0.5, 2.0, 2.0]) / 4)
    # суммы рангов 7; 6; 9,5; 7,5 → s = (сумма − 3 × 0,5) / (3 × 4)
    assert s == pytest.approx((np.array([7.0, 6.0, 9.5, 7.5]) - 1.5) / 12)


def test_rank_scores_scale_free():
    # масштаб цели не влияет: умножение d одной цели на 1000 не меняет s
    rng = np.random.default_rng(0)
    d = {t: rng.normal(size=9) for t in ("a", "b", "c")}
    _, s1 = E.rank_scores(d)
    _, s2 = E.rank_scores({**d, "c": d["c"] * 1000})
    assert s1 == pytest.approx(s2)


def test_rank_scores_rejects_missing():
    with pytest.raises(QCError, match="нет разности"):
        E.rank_scores({"a": np.array([0.1, np.nan])})


def test_pick_closest_to_median_ties_smaller_id():
    ids = np.array([30, 10, 20, 40, 50])
    s = np.array([0.2, 0.6, 0.4, 0.5, 0.5])  # медиана 0,5: у 40 и 50 расстояние 0 — меньший id
    assert E.pick_closest_to_median(ids, s, 1e-12) == 40
    # равенство в пределах допуска: 0,5 + 1e-14 считается равным 0,5
    s2 = np.array([0.2, 0.6, 0.4, 0.5 + 1e-14, 0.5])
    assert E.pick_closest_to_median(np.array([30, 10, 20, 40, 50]), s2, 1e-12) == 40
    # без равенств — ближайший к медиане
    assert E.pick_closest_to_median(np.array([1, 2, 3]), np.array([0.1, 0.35, 0.9]), 1e-12) == 2


# --- сквозной прогон ----------------------------------------------------------------------------------------


def _independent_pick(cfg) -> tuple[int, int]:
    """Выбор заново, без функций модуля: ошибки — groupby-медианы членов с известной целью, ранги — pandas."""
    out, proc = Path(cfg["paths"]["outputs"]), Path(cfg["paths"]["processed"])
    ctx = pd.read_parquet(proc / "context_annual.parquet")
    comp = pd.read_csv(out / "interpret" / "node_comparable.csv")
    err = pd.read_csv(out / "interpret" / "t7_errors.csv")
    size_check = json.loads((out / "usefulness" / "size_check.json").read_text(encoding="utf-8"))
    bound = size_check["quintile_bounds"][-1]
    ids = sorted(err.loc[err["common"], "territory_id"])
    w = ctx.pivot(index="territory_id", columns="year")
    pop = w[("pop_avg", 2023)]
    cols = {"catering": "catering_turnover_pc", "ndfl": "ndfl_income_pc", "shipments": "shipments_pc"}
    d = {}
    for t, c in cols.items():
        y = np.log1p(w[(c, 2024)]) - np.log1p(w[(c, 2023)])
        if t == "ndfl":
            y = y.where(w[("ndfl_ok", 2023)].astype(bool) & w[("ndfl_ok", 2024)].astype(bool))
        e = {}
        for s in ("B", "D"):
            m = comp.loc[comp["set"] == s].assign(y=lambda x, y=y: x["other_id"].map(y)).dropna(subset=["y"])
            g = m.groupby("territory_id")["y"]
            ok = g.size() >= 5
            e[s] = (y - g.median()).abs().where(ok)
        d[t] = (e["B"] - e["D"]).reindex(ids)
    frame = pd.DataFrame(d)
    frame = frame.loc[(pop.reindex(ids) <= bound).to_numpy() & frame.notna().all(axis=1).to_numpy()]
    u = frame.rank(method="average").sub(0.5).div(len(frame))
    s = u.mean(axis=1)
    dist = (s - s.median()).abs()
    best = dist.min()
    return int(min(dist.index[dist <= best + 1e-12])), len(frame)


TIME_KEYS = {
    "seconds"
}  # замеры времени в JSON этапа (size_check.json: seconds и shared_members_bootstrap.seconds)


def _drop_time(x):
    if isinstance(x, dict):
        return {k: _drop_time(v) for k, v in x.items() if k not in TIME_KEYS}
    if isinstance(x, list):
        return [_drop_time(v) for v in x]
    return x


def _hashes(folder: Path) -> dict[str, str]:
    """sha256 каждого файла: JSON — без полей времени (TIME_KEYS), остальное — побайтно."""
    out = {}
    for p in sorted(folder.iterdir()):
        if p.suffix == ".json":
            obj = _drop_time(json.loads(p.read_text(encoding="utf-8")))
            data = json.dumps(obj, ensure_ascii=False, sort_keys=True).encode("utf-8")
        else:
            data = p.read_bytes()
        out[p.name] = hashlib.sha256(data).hexdigest()
    return out


def test_end_to_end_matches_independent_pick(tmp_path):
    cfg, *_ = _synthetic(tmp_path, np.random.default_rng(1))
    U.run(cfg)
    folder = Path(cfg["paths"]["outputs"]) / "usefulness"
    before = _hashes(folder)
    res = E.run(cfg)
    after = _hashes(folder)
    # example_small.json уже записал usefulness.run (последний шаг); повтор даёт те же файлы побайтно
    assert after == before

    tid, n_pool = _independent_pick(cfg)
    assert res["territory_id"] == tid
    assert res["score"]["n_pool"] == n_pool and res["score"]["distance_rank"] == 1
    # пул: только квинтили 1–4 (население 1000 × id, граница — 48 200) и все три цели
    # (доход 5-НДФЛ неизвестен у id % 13 == 0)
    assert res["pool"]["n_small"] == 48
    assert res["pop_avg_2023"] <= res["pool"]["bound"]
    assert res["territory_id"] % 13 != 0
    assert res["qc"]["abs_diff"] <= 1e-12
    pt = res["per_target"]
    assert set(pt) == {"catering", "ndfl", "shipments"}
    for v in pt.values():
        assert v["closer"] in {"B", "D", "tie"} and 0 < v["pct_B_pool"] <= 1 and 0 < v["u"] < 1
    assert res["n_closer"] == sum(v["closer"] == "B" for v in pt.values())
    assert res["retail"]["label"].startswith("цель примера Йошкар-Олы")
    assert len(res["members_B"]) == 5 and res["members_B"][0]["km"] == 10.0
    assert res["region"].startswith("Регион ") and res["flag_text"]
    saved = json.loads((folder / "example_small.json").read_text(encoding="utf-8"))
    assert saved["territory_id"] == tid and saved["kind"] == "illustration"


def test_retail_does_not_affect_pick(tmp_path):
    cfg, *_ = _synthetic(tmp_path, np.random.default_rng(1))
    U.run(cfg)
    first = E.run(cfg)["territory_id"]
    p = Path(cfg["paths"]["processed"]) / "context_annual.parquet"
    ctx = pd.read_parquet(p)
    ctx.loc[ctx["year"] == 2024, "retail_pc"] *= np.exp(np.random.default_rng(9).normal(0, 0.5, 60))
    ctx.to_parquet(p)
    assert E.run(cfg)["territory_id"] == first


def test_qc_fails_when_share_differs_from_size_check(tmp_path):
    cfg, *_ = _synthetic(tmp_path, np.random.default_rng(1))
    U.run(cfg)
    p = Path(cfg["paths"]["outputs"]) / "usefulness" / "size_check.json"
    f = json.loads(p.read_text(encoding="utf-8"))
    f["lower_four"]["share"] += 0.01
    p.write_text(json.dumps(f), encoding="utf-8")
    with pytest.raises(QCError, match="lower_four.share"):
        E.run(cfg)
    assert not (p.parent / "example_small.json").exists()


def test_qc_fails_when_n_differs(tmp_path):
    cfg, *_ = _synthetic(tmp_path, np.random.default_rng(1))
    U.run(cfg)
    p = Path(cfg["paths"]["outputs"]) / "usefulness" / "size_check.json"
    f = json.loads(p.read_text(encoding="utf-8"))
    f["lower_four"]["n"] += 1
    p.write_text(json.dumps(f), encoding="utf-8")
    with pytest.raises(QCError, match="lower_four.n"):
        E.run(cfg)


def test_missing_inputs(tmp_path):
    cfg, *_ = _synthetic(tmp_path, np.random.default_rng(1))
    with pytest.raises(MissingInputError):  # нет size_check.json и mo_flags.csv — usefulness не запускался
        E.run(cfg)


def test_existing_outputs_unchanged_with_and_without_example(tmp_path, monkeypatch):
    """Все прежние выходы usefulness с примером совета и без него одинаковы: JSON — без полей времени,
    остальные файлы — побайтно."""
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    cfg_a, *_ = _synthetic(a, np.random.default_rng(5))
    cfg_b, *_ = _synthetic(b, np.random.default_rng(5))
    U.run(cfg_a)
    monkeypatch.setattr(E, "run", lambda cfg: None)
    U.run(cfg_b)
    ha = _hashes(Path(cfg_a["paths"]["outputs"]) / "usefulness")
    hb = _hashes(Path(cfg_b["paths"]["outputs"]) / "usefulness")
    assert set(ha) - set(hb) == {"example_small.json"}
    assert {k: v for k, v in ha.items() if k != "example_small.json"} == hb


def test_unchanged_check_catches_real_change(tmp_path, monkeypatch):
    """Проверка выше не слепа: если пример изменит прежний выход (facts.json), хеши разойдутся; а одно лишь
    поле времени в size_check.json их не меняет."""
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    cfg_a, *_ = _synthetic(a, np.random.default_rng(5))
    cfg_b, *_ = _synthetic(b, np.random.default_rng(5))
    U.run(cfg_a)
    real_run = E.run

    def run_and_touch(cfg):
        res = real_run(cfg)
        p = Path(cfg["paths"]["outputs"]) / "usefulness" / "facts.json"
        f = json.loads(p.read_text(encoding="utf-8"))
        f["example_small"] = res["territory_id"]
        p.write_text(json.dumps(f, ensure_ascii=False, indent=1), encoding="utf-8")
        return res

    monkeypatch.setattr(E, "run", run_and_touch)
    U.run(cfg_b)
    oa = Path(cfg_a["paths"]["outputs"]) / "usefulness"
    ob = Path(cfg_b["paths"]["outputs"]) / "usefulness"
    ha, hb = _hashes(oa), _hashes(ob)
    assert ha["facts.json"] != hb["facts.json"]
    # поле времени: другое значение seconds не меняет хеш, другое значение данных — меняет
    p = ob / "size_check.json"
    f = json.loads(p.read_text(encoding="utf-8"))
    f["seconds"] += 100.0
    f["shared_members_bootstrap"]["seconds"] += 100.0
    p.write_text(json.dumps(f), encoding="utf-8")
    assert _hashes(ob)["size_check.json"] == ha["size_check.json"]
    f["lower_four"]["share"] += 1e-9
    p.write_text(json.dumps(f), encoding="utf-8")
    assert _hashes(ob)["size_check.json"] != ha["size_check.json"]
