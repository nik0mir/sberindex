"""Сводка разведки (пакет R): сюжеты и матрица решения, показатели МО, «регион или место» (T13, T14, F16)."""

import copy

import numpy as np
import pandas as pd
import pytest
from synth import make_config, make_processed

import munnet.eda as eda
from munnet.config import Config
from munnet.eda import stories, synthesis
from munnet.eda.base import SECTION_IDS, SYNTHESIS_ID, SYNTHESIS_NUMBER, SectionContext, make_fact
from munnet.eda.data import load

# --- Общие данные: разделы E1–E5 на синтетике считаются один раз на модуль ------------------------


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    """Синтетика, итоги пяти разделов и сводка на ней (как в полном прогоне, но без отчёта)."""
    root = tmp_path_factory.mktemp("syn")
    make_processed(root)
    cfg = make_config(root)
    data = load(cfg)
    findings = eda.run_sections(cfg, list(SECTION_IDS), data=data)
    ctx = SectionContext(cfg, data, eda.eda_dir(cfg), SYNTHESIS_ID, eda.section_rng(cfg, SYNTHESIS_NUMBER))
    result = synthesis.run_synthesis(ctx, findings)
    return {"cfg": cfg, "data": data, "findings": findings, "syn": result, "out": eda.eda_dir(cfg)}


@pytest.fixture
def cfg(tmp_path):
    return make_config(tmp_path)


def facts_of(values: dict) -> dict:
    """Словарь фактов из {ключ: (значение, вид)}."""
    return {k: make_fact(k, v, kind) for k, (v, kind) in values.items()}


# Проверенные значения первого прогона на реальных данных (разделы E1–E5, 26.09.2026).
REAL_LIKE = {
    "syn.n_mo": (2190, "int"),
    "e1.n_full": (2016, "int"),
    "e3.reliable_share": (0.156, "pct"),
    "e3.null_reliable_share": (0.015, "pct"),
    "e2.partial_rho_access_within": (0.204, "rho"),
    "e2.n_access_wage": (2035, "int"),
    "e4.growth_half_consistency": (0.353, "rho"),
    "e4.residual_share": (0.236, "pct"),
    "e4.residual_pc1_stability": (0.895, "rho"),
    "e4.n_basket": (2016, "int"),
    "e4.mp_split_half": (0.683, "rho"),
    "e4.rho_mp_pp_level_within": (-0.619, "rho"),
    "e4.mp_step_max_pp": (1.07, "num2"),
    "e5.n_ndfl_usable": (1569, "int"),
    "e5.suburb_ratio": (1.0, "num2"),
    "e5.suburb_mw_p": (0.334, "p"),
}


def matrix_for(values: dict, cfg: Config) -> pd.DataFrame:
    return synthesis.decision_matrix(facts_of(values), cfg).set_index("sid")


# --- Веса и баллы --------------------------------------------------------------------------------


def test_matrix_weights_sum_to_one(cfg):
    w = stories.check_weights(cfg["eda"]["matrix_weights"])
    assert set(w) == set(stories.CRITERIA) and sum(w.values()) == pytest.approx(1.0)
    bad = dict(w, signal=0.5)
    with pytest.raises(ValueError, match="сумма весов"):
        stories.check_weights(bad)
    with pytest.raises(ValueError, match="нужны ключи"):
        stories.check_weights({k: v for k, v in w.items() if k != "signal"})
    with pytest.raises(ValueError, match="отрицательный"):
        stories.check_weights(dict(w, signal=-0.1, practical=0.55))


def test_weighted_score_needs_every_score():
    w = {"a": 0.5, "b": 0.5}
    assert stories.weighted_score({"a": 4, "b": 2}, w) == pytest.approx(3.0)
    assert np.isnan(stories.weighted_score({"a": 4, "b": None}, w))


def test_bin_score_edges():
    bins = [0.1, 0.2, 0.3, 0.4]
    assert [stories.bin_score(v, bins) for v in (0.05, 0.1, 0.25, 0.39, 0.4, 0.9)] == [1, 2, 3, 4, 5, 5]
    assert stories.bin_score(None, bins) is None
    with pytest.raises(ValueError, match="возрастающих"):
        stories.bin_score(0.2, [0.3, 0.2, 0.1, 0.0])
    with pytest.raises(ValueError, match="возрастающих"):
        stories.bin_score(0.2, [0.1, 0.2])


def test_signal_and_coverage_scores_from_facts():
    facts = facts_of({"e2.x": (-0.35, "rho"), "e1.n": (1800, "int"), "syn.n_mo": (2000, "int")})
    score, note = stories.BinScore("e2.x", (0.1, 0.2, 0.3, 0.4), "частный ρ", use_abs=True)(facts)
    assert score == 4 and "|0,35|" in note
    assert stories.BinScore("e9.none", (0.1, 0.2, 0.3, 0.4), "нет")(facts) == (None, "нет факта e9.none")
    bands = (0.6, 0.7, 0.8, 0.9)
    score, note = stories.CoverageScore("e1.n", bands, "МО с полным рядом")(facts)
    assert score == 5 and "1800 из 2000 (90%)" in note  # ровно 0,9 — уже 5
    few = facts_of({"e1.n": (1180, "int"), "syn.n_mo": (2000, "int")})
    assert stories.CoverageScore("e1.n", bands, "МО")(few)[0] == 1  # 59% — меньше нижней границы
    assert stories.CoverageScore("e1.n", bands, "МО", denominator="syn.none")(few)[0] is None


# --- Критерии отказа и роль сюжета ---------------------------------------------------------------


def test_rejection_operations_and_thresholds(cfg):
    prm = stories.params(cfg)
    facts = facts_of(
        {
            "e3.share": (0.2, "pct"),
            "e3.null": (0.05, "pct"),
            "e4.rho": (-0.4, "rho"),
            "e4.step": (1.5, "num2"),
            "e5.p": (0.02, "p"),
        }
    )
    r = stories.Rejection
    check = stories.check_rejection
    ge = check(r("a", "e3.share", ">=", "rejection.s1_reliable_share_min", "доля"), facts, prm, cfg["eda"])
    assert ge.ok is False and "20,0%, нужно ≥ 25,0%" in ge.text
    times = r("b", "e3.share", ">=", "rejection.s1_null_ratio_min", "против нуля", times_fact="e3.null")
    res = check(times, facts, prm, cfg["eda"])
    assert res.ok is True and "3 × 5,0% = 15,0%" in res.text  # 20% ≥ 3 × 5%
    assert check(r("c", "e4.rho", "abs>=", "rejection.s5_rho_within_min", "ρ"), facts, prm, cfg["eda"]).ok
    assert check(r("d", "e4.step", "<=", "step_test_pp", "скачок"), facts, prm, cfg["eda"]).ok  # 1,5 ≤ 2
    assert not check(r("e", "e5.p", "<", "rejection.s6_suburb_p_max", "p"), facts, prm, cfg["eda"]).ok
    missing = check(r("f", "e9.none", ">=", "step_test_pp", "нет"), facts, prm, cfg["eda"])
    assert missing.ok is None and "нет факта e9.none" in missing.text
    with pytest.raises(KeyError, match="нет порога"):
        check(r("g", "e3.share", ">=", "rejection.nope", "нет"), facts, prm, cfg["eda"])
    with pytest.raises(ValueError, match="неизвестная операция"):
        check(r("h", "e3.share", "==", "step_test_pp", "нет"), facts, prm, cfg["eda"])


def result(ok, role=stories.ROLE_SIGNAL):
    return stories.CriterionResult("x", ok, role, "текст")


def test_story_status_rules():
    main, layer, context, na = (
        stories.STATUS_MAIN,
        stories.STATUS_LAYER,
        stories.STATUS_CONTEXT,
        stories.STATUS_NA,
    )
    assert stories.story_status([result(True), result(True)]) == main
    assert stories.story_status([result(False), result(True)]) == layer
    assert stories.story_status([result(False), result(False)]) == context
    # выполнен только критерий данных — сигнала нет, это контекст, а не слой
    assert stories.story_status([result(True, stories.ROLE_DATA), result(False)]) == context
    assert stories.story_status([result(True), result(None)]) == na
    assert stories.rejection_text([result(True)]) == stories.PASSED_TEXT


def test_real_like_facts_give_expected_roles(cfg):
    m = matrix_for(REAL_LIKE, cfg)
    assert m["status"].to_dict() == {
        "С1": "слой",  # 15,6% < 25%, но в 10 раз выше нуля
        "С2": "может быть главным",  # |0,204| ≥ 0,2
        "С3": "контекст",
        "С4": "слой",  # остаток 23,6% < 30%, устойчивость 0,89 ≥ 0,7
        "С5": "может быть главным",
        "С6": "контекст",  # пригороды не отличаются; выполнен только критерий данных
    }
    assert m.loc["С5", "score"] == pytest.approx(4.05) and m.loc["С4", "score"] == pytest.approx(4.0)
    assert m.loc["С1", "signal"] == 2 and m.loc["С6", "coverage"] == 3  # 1569 / 2190 = 72%
    assert m.loc["С2", "criteria"] == stories.PASSED_TEXT
    assert "нужно ≥ 1,20" in m.loc["С6", "criteria"] and "нужно < 0,010" in m.loc["С6", "criteria"]


def test_failed_story_is_never_main_whatever_its_score(cfg):
    values = dict(REAL_LIKE)
    values["e4.mp_split_half"] = (0.5, "rho")  # С5 проходит впритык, сигнал 3 → итог 3,80 < 4,00 у С4
    m = synthesis.decision_matrix(facts_of(values), cfg)
    ranked = stories.ranking(m)
    assert ranked["sid"].iloc[0] == "С4"  # по баллу выше всех — слой
    top, second = stories.pick_top(m)
    assert top["sid"] == "С5" and second["sid"] == "С4" and second["status"] == stories.STATUS_LAYER
    text = stories.verdict_text(m)
    assert text.index("С5") < text.index("С4") and "слой" in text


def test_no_story_passes(cfg):
    values = dict(REAL_LIKE)
    values["e2.partial_rho_access_within"] = (0.05, "rho")
    values["e4.mp_split_half"] = (0.2, "rho")
    m = synthesis.decision_matrix(facts_of(values), cfg)
    top, second = stories.pick_top(m)
    assert top is None and second is not None
    assert stories.verdict_text(m).startswith("Ни один сюжет не прошёл")


def test_missing_section_facts_mean_no_data(cfg):
    values = {k: v for k, v in REAL_LIKE.items() if not k.startswith("e3.")}
    m = matrix_for(values, cfg)
    assert m.loc["С1", "status"] == stories.STATUS_NA and np.isnan(m.loc["С1", "score"])
    assert "нет факта e3.reliable_share" in m.loc["С1", "criteria"]
    top, _ = stories.pick_top(m.reset_index())
    assert top["sid"] == "С5"


def test_stories_contract(cfg):
    stories.validate_stories(stories.STORIES)
    assert [s.sid for s in stories.STORIES] == ["С1", "С2", "С3", "С4", "С5", "С6"]
    prm = stories.params(cfg)
    for s in stories.STORIES:
        for r in s.rejections:  # каждый порог находится в конфиге или в дополнениях DEFAULTS
            assert np.isfinite(stories.resolve_threshold(r.threshold_key, prm, cfg["eda"]))
    broken = copy.deepcopy(stories.STORIES[0].manual)
    broken["practical"] = (7, "")
    bad = stories.Story(
        "С9", "x", "y", broken, stories.STORIES[0].signal, stories.STORIES[0].coverage, (), "s9"
    )
    with pytest.raises(ValueError, match="вне 1–5"):
        stories.validate_stories([bad])


def test_params_reject_unknown_keys(cfg):
    data = copy.deepcopy(cfg.data)
    data["eda"]["stories"] = {"signal_binz": {}}
    with pytest.raises(ValueError, match="eda.stories"):
        stories.params(Config(data, cfg.path))
    data["eda"]["synthesis"] = {"eta2_regionall": 0.5}
    with pytest.raises(ValueError, match="eda.synthesis"):
        synthesis.params(Config(data, cfg.path))
    data["eda"]["stories"] = {"coverage_bands": [0.5, 0.6, 0.7, 0.8]}
    data["eda"]["rejection"]["s1_null_ratio_min"] = 5
    prm = stories.params(Config(data, cfg.path))
    assert prm["coverage_bands"] == [0.5, 0.6, 0.7, 0.8] and prm["rejection"]["s1_null_ratio_min"] == 5
    assert prm["rejection"]["s6_suburb_p_max"] == stories.DEFAULTS["rejection"]["s6_suburb_p_max"]


# --- Показатели МО и «регион или место» ----------------------------------------------------------


def test_story_facts_exist_in_sections(synthetic):
    """Контракт с разделами: каждый факт, на который опираются баллы и критерии, разделы пишут."""
    facts = {k for f in synthetic["findings"] for k in f.facts} | set(synthetic["syn"].facts)
    needed = set()
    for s in stories.STORIES:
        needed |= {s.signal.fact_key, s.coverage.fact_key, s.coverage.denominator}
        needed |= {r.fact_key for r in s.rejections} | {r.times_fact for r in s.rejections if r.times_fact}
    assert needed - facts == set()


def test_collect_indicators_takes_mo_and_sections(synthetic):
    data, findings, cfg = synthetic["data"], synthetic["findings"], synthetic["cfg"]
    ind, labels = synthesis.collect_indicators(data, findings, cfg)
    prm = synthesis.params(cfg)
    assert len(ind) == len(data.mo) and ind["territory_id"].is_unique
    for name in prm["mo_indicators"]:
        assert name in ind.columns and labels[name] != name  # у каждого показателя mo — русская подпись
    expected = np.log(data.mo["spend_to_wage_2023"])
    assert np.allclose(ind["log_spend_to_wage_2023"], expected, equal_nan=True)
    for col in ("own_amplitude", "own_repro_r", "basket_resid_pc1", "log_spend_to_ndfl"):
        assert col in ind.columns and col in labels
    assert synthesis.indicator_sources(ind, findings)["own_amplitude"] == "e3"
    # повтор колонки с теми же значениями допустим, с другими — ошибка
    e3 = next(f for f in findings if f.section == "e3")
    twin = copy.copy(e3)
    twin.indicators = e3.indicators.copy()
    synthesis.collect_indicators(data, [*findings, twin], cfg)
    twin.indicators["own_amplitude"] = twin.indicators["own_amplitude"] + 1
    with pytest.raises(ValueError, match="расходится"):
        synthesis.collect_indicators(data, [*findings, twin], cfg)


def test_region_or_place_separates_regional_from_noise(synthetic):
    data, cfg = synthetic["data"], synthetic["cfg"]
    mo = data.mo
    rng = np.random.default_rng(1)
    ind = pd.DataFrame(
        {
            "territory_id": mo["territory_id"],
            "level": mo["log_level_2024"],
            "noise": rng.normal(size=len(mo)),
            "flag": (rng.random(len(mo)) > 0.5).astype(float),
        }
    )
    t = synthesis.region_or_place(ind, mo, cfg, np.random.default_rng(0)).set_index("indicator")
    assert "flag" not in t.index  # флаг (0/1) не описывается η² и I Морана
    assert t.loc["level", "eta2"] > 0.5 > t.loc["noise", "eta2"] + 0.2
    assert t.loc["level", "moran_i"] > t.loc["noise", "moran_i"] and t.loc["level", "moran_p"] < 0.01
    assert bool(t.loc["level", "regional"]) and not bool(t.loc["noise", "regional"])
    n_inner = int(mo["is_inner_city"].sum())
    assert t.loc["noise", "n_no_inner"] == t.loc["noise", "n"] - n_inner
    assert list(t.index) == ["level", "noise"]  # по убыванию η²
    for k in cfg["eda"]["knn_sensitivity"]:
        assert f"moran_i_k{k}" in t.columns
    again = synthesis.region_or_place(ind, mo, cfg, np.random.default_rng(0)).set_index("indicator")
    pd.testing.assert_frame_equal(t, again)  # один генератор — одинаковые p


def test_classify_and_headline_follow_data(synthetic):
    prm = synthesis.params(synthetic["cfg"])
    base = {"log_level_2024": 0.77, "growth_log": 0.29, "mp_pp_change": 0.41}

    def classify(values):
        t = pd.DataFrame({"indicator": list(values), "eta2": list(values.values())})
        return synthesis.classify(t, prm)

    rp = classify(base)
    assert rp.level_regional and rp.place == ("growth_log", "mp_pp_change")
    title = synthesis.headline_text(rp, prm["headline"]["dynamic"])
    assert "рост и сдвиг к онлайну" in title and "0,77" in title and len(title) <= 90
    rp = classify(dict(base, mp_pp_change=0.56))  # η² сдвига к онлайну ≥ 0,5 — это уже не «место»
    assert rp.place == ("growth_log",)
    assert "онлайн" not in synthesis.headline_text(rp, prm["headline"]["dynamic"])
    rp = classify(dict(base, log_level_2024=0.5))
    assert not rp.level_regional and rp.place == ()


def test_run_synthesis_on_synthetic(synthetic):
    syn, out = synthetic["syn"], synthetic["out"]
    assert syn.section == "syn" and syn.headline_errors == []
    assert [f.fid for f in syn.figures] == ["F16"] and [t.tid for t in syn.tables] == ["T13", "T14"]
    for rec in syn.figures:
        assert (out / rec.png).exists() and (out / rec.data_csv).exists()
    facts = syn.facts
    for k in ("s1", "s2", "s3", "s4", "s5", "s6"):
        assert f"syn.score_{k}" in facts and f"syn.reject_{k}" in facts and f"syn.status_{k}" in facts
    for key in (
        "top_story",
        "second_story",
        "verdict",
        "n_mo",
        "eta2_log_level_2024",
        "moran_log_level_2024",
    ):
        assert f"syn.{key}" in facts
    assert "syn.moran_log_level_2024_no_inner" in facts and facts["syn.n_mo"].value == len(
        synthetic["data"].mo
    )
    t14 = pd.read_csv(out / syn.tables[1].csv)
    assert list(t14["sid"]) == ["С1", "С2", "С3", "С4", "С5", "С6"]
    assert t14["score"].between(1, 5).all()
    # индикаторы сводки — вся таблица показателей МО с подписями
    assert {"log_level_2024", "own_amplitude", "log_spend_to_ndfl"} <= set(syn.indicators.columns)
    assert set(syn.indicators.columns) - {"territory_id"} == set(syn.indicator_labels)
    # на синтетике уровень задан регионом, рост — нет: заголовок F16 проходит проверку
    assert synthetic["syn"].figures[0].title.startswith("Уровень трат")


def test_synthesis_is_reproducible(synthetic):
    cfg, data, findings = synthetic["cfg"], synthetic["data"], synthetic["findings"]
    rng = eda.section_rng(cfg, SYNTHESIS_NUMBER)
    ind, labels = synthesis.collect_indicators(data, findings, cfg)
    t = synthesis.region_or_place(ind, data.mo, cfg, rng, labels=labels)
    first = pd.read_csv(synthetic["out"] / synthetic["syn"].figures[0].data_csv)
    assert np.allclose(first["moran_p"], t["moran_p"]) and list(first["indicator"]) == list(t["indicator"])
