"""Сводка разведки (пакет R): сюжеты и матрица решения, показатели МО, «регион или место» (T13, T14, F16),
проверки сюжетов (без внутригородских территорий, сверх уровня и региона, прототип T15, внешняя проверка)."""

import copy
import logging

import numpy as np
import pandas as pd
import pytest
from synth import make_config, make_processed

import munnet.eda as eda
from munnet.config import Config
from munnet.eda import stats, stories, synthesis
from munnet.eda.base import SECTION_IDS, SYNTHESIS_ID, SYNTHESIS_NUMBER, SectionContext, make_fact
from munnet.eda.data import load

NB = stories.style.NBSP

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


def flat(text: str) -> str:
    return text.replace(NB, " ")


# Проверенные значения прогона на реальных данных (разделы E1–E5 и проверки сводки, 26.09.2026).
REAL_LIKE = {
    "syn.n_mo": (2190, "int"),
    "syn.n_inner": (247, "int"),
    "e1.n_full": (2016, "int"),
    "e3.reliable_share": (0.156, "pct"),
    "e3.null_reliable_share": (0.015, "pct"),
    "e2.partial_rho_access_within": (0.204, "num3"),
    "e2.partial_rho_access_within_no_inner": (0.198, "num3"),
    "e2.n_access_wage": (2035, "int"),
    "e4.growth_half_consistency": (0.353, "rho"),
    "e4.residual_share": (0.236, "pct"),
    "e4.residual_pc1_stability": (0.895, "rho"),
    "e4.n_basket": (2016, "int"),
    "e4.mp_split_half": (0.683, "rho"),
    "syn.mp_split_half_resid": (0.436, "rho"),
    "syn.mp_r2_level_region": (0.576, "pct"),
    "e4.rho_mp_pp_level_within": (-0.619, "rho"),
    "e4.mp_step_max_pp": (1.07, "num2"),
    "e5.n_ndfl_usable": (1607, "int"),
    "e5.suburb_ratio": (0.99, "num2"),
    "e5.suburb_mw_p": (0.492, "p"),
}
# Те же критерии без 247 внутригородских территорий (факты сводки ``stories.NO_INNER``).
NO_INNER_LIKE = {
    "syn.reliable_share_no_inner": (0.174, "pct"),
    "syn.growth_half_consistency_no_inner": (0.39, "rho"),
    "syn.residual_share_no_inner": (0.316, "pct"),
    "syn.residual_pc1_stability_no_inner": (0.889, "rho"),
    "syn.mp_split_half_no_inner": (0.588, "rho"),
    "syn.rho_mp_pp_level_within_no_inner": (-0.619, "rho"),
    "syn.mp_split_half_resid_no_inner": (0.442, "rho"),
}
# С5 проходит все критерии: сигнал сверх уровня и региона не слабее порога.
C5_PASSES = {"syn.mp_split_half_resid": (0.6, "rho")}


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
    assert score == 4 and "−0,35 (балл — по модулю)" in flat(note)  # значение со знаком, балл — по модулю
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
            "e4.rho": (-0.62, "rho"),
            "e4.step": (1.5, "num2"),
            "e5.p": (0.02, "p"),
        }
    )
    r = stories.Rejection
    check = stories.check_rejection
    ge = check(r("a", "e3.share", ">=", "rejection.s1_reliable_share_min", "доля"), facts, prm, cfg["eda"])
    assert ge.ok is False and "20,0%, нужно ≥ 25,0%" in ge.text
    assert ge.short == ge.text  # других оценок нет — короткий текст тот же
    times = r("b", "e3.share", ">=", "rejection.s1_null_ratio_min", "против нуля", times_fact="e3.null")
    res = check(times, facts, prm, cfg["eda"])
    assert res.ok is True and "3 × 5,0% = 15,0%" in res.text  # 20% ≥ 3 × 5%
    within = check(r("c", "e4.rho", "abs>=", "rejection.s5_rho_within_min", "ρ"), facts, prm, cfg["eda"])
    # значение со знаком, как в разделе; «по модулю» — в пороге, вертикальных черт нет
    assert within.ok and "ρ −0,62, по модулю нужно ≥ 0,30" in flat(within.text) and "|" not in within.text
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
        "С2": "контекст",  # без внутригородских 0,198 < 0,2 (по всем МО 0,204 — другая оценка)
        "С3": "контекст",
        "С4": "слой",  # остаток 23,6% < 30%, устойчивость 0,89 ≥ 0,7
        "С5": "слой",  # сверх уровня и региона полугодия согласованы на 0,44 < 0,5
        "С6": "контекст",  # пригороды не отличаются; выполнен только критерий данных
    }
    assert m.loc["С5", "score"] == pytest.approx(3.85) and m.loc["С4", "score"] == pytest.approx(3.5)
    assert m.loc["С4", "signal"] == 3  # доля остатка 23,6% по шкале С1, а не устойчивость 0,89
    assert m.loc["С2", "signal"] == 2 and m.loc["С1", "signal"] == 2
    assert m.loc["С6", "coverage"] == 3  # 1607 / 2190 = 73%
    assert "сверх уровня трат и региона" in flat(m.loc["С5", "criteria"])
    assert (
        "сверх уровня трат и региона" in flat(m.loc["С5", "failed"])
        and "для сравнения" not in m.loc["С5", "failed"]
    )
    assert "нужно ≥ 1,20" in m.loc["С6", "criteria"] and "нужно < 0,010" in m.loc["С6", "criteria"]
    # С5 проходит, если сигнал сверх уровня и региона не слабее порога
    passed = matrix_for({**REAL_LIKE, **C5_PASSES}, cfg)
    assert passed.loc["С5", "status"] == stories.STATUS_MAIN and passed.loc["С5", "criteria"] == "проходит"


def test_other_estimates_mark_criterion_on_the_edge(cfg):
    """Ловушка С2: частный ρ 0,198 без внутригородских при пороге 0,2, а по всем МО и на верхней границе
    интервала бутстрепа — выше порога."""
    prm = stories.params(cfg)
    rej = stories.Rejection(
        "x",
        "e2.r",
        "abs>=",
        "rejection.s2_partial_rho_min",
        "частный ρ",
        alts=(("e2.lo", "нижняя граница"), ("e2.hi", "верхняя"), ("e2.none", "нет такого факта")),
    )
    edge = facts_of({"e2.r": (0.204, "num3"), "e2.lo": (0.153, "num3"), "e2.hi": (0.253, "num3")})
    res = stories.check_rejection(rej, edge, prm, cfg["eda"])
    assert res.ok is True and res.fragile
    assert "(критерий на границе: нижняя граница — 0,153 ✗; верхняя — 0,253 ✓)" in flat(res.text)
    assert "нет такого" not in res.text  # другой оценки без факта просто нет
    assert "границе" not in res.short and res.text.startswith(res.short)
    solid = facts_of({"e2.r": (-0.35, "num3"), "e2.lo": (-0.3, "num3"), "e2.hi": (-0.4, "num3")})
    res = stories.check_rejection(rej, solid, prm, cfg["eda"])
    assert res.ok is True and not res.fragile
    assert "(проверка: нижняя граница — −0,300; верхняя — −0,400; вывод тот же)" in flat(res.text)
    # на реальных числах С2 не проходит без внутригородских, но по всем МО проходит: «на границе»
    values = dict(
        REAL_LIKE,
        **{
            "e2.partial_rho_access_within_lo": (0.153, "num3"),
            "e2.partial_rho_access_within_hi": (0.253, "num3"),
        },
    )
    m = matrix_for(values, cfg)
    assert m.loc["С2", "status"] == stories.STATUS_CONTEXT and bool(m.loc["С2", "fragile"])
    assert m.loc["С2", "role"] == f"{stories.STATUS_CONTEXT} ({stories.FRAGILE_TEXT})"
    assert "по всем МО — 0,204 ✓" in flat(m.loc["С2", "criteria_all"])
    assert m.loc["С2", "criteria_all"].startswith(stories.MARK_FAIL)
    c6 = m.loc["С6", "criteria_all"].split("; ")
    assert c6[0].startswith(stories.MARK_OK) and stories.MARK_FAIL in m.loc["С6", "criteria_all"]


def test_no_inner_estimate_is_added_and_shown(cfg):
    """Оценка без внутригородских территорий добавляется к критерию сама и делает С4 «на границе»."""
    m = matrix_for({**REAL_LIKE, **NO_INNER_LIKE}, cfg)
    c4 = flat(m.loc["С4", "criteria_all"])
    assert "без внутригородских территорий — 31,6% ✓" in c4 and bool(m.loc["С4", "fragile"])
    assert "для сравнения" not in c4  # числа для сравнения — только если есть факт
    with_ref = {**REAL_LIKE, **NO_INNER_LIKE, "syn.resid_part_stability_span": ("0,87–0,93", "str")}
    c4 = flat(matrix_for(with_ref, cfg).loc["С4", "criteria_all"])
    assert "для сравнения: у любой очищенной доли по отдельности — 0,87–0,93" in c4
    assert c4.count("(") == c4.count(")")


def test_matrix_without_inner_city_and_sensitivity(cfg):
    """Ловушка Ж.2: без 247 внутригородских С4 проходит все критерии, а на всех МО — никто."""
    facts = facts_of({**REAL_LIKE, **NO_INNER_LIKE})
    m = synthesis.decision_matrix(facts, cfg)
    other = synthesis.decision_matrix(stories.substitute(facts), cfg)
    assert stories.pick_top(m)[0] is None
    top, _ = stories.pick_top(other)
    assert top["sid"] == "С4" and top["score"] == pytest.approx(3.75)  # сигнал 31,6% → 4
    assert other.set_index("sid").loc["С5", "score"] == pytest.approx(3.6)  # полугодия 0,59 → сигнал 3
    text = flat(stories.sensitivity_text(m, other, 247))
    assert text.startswith("Без 247 внутригородских территорий Москвы и Петербурга все критерии отказа")
    assert "С4 «Состав корзины» (итог 3,75)" in text and "связан с решением по Москве" in text
    assert flat(stories.sensitivity_text(m, m, 247)).endswith("роли сюжетов и лидер по итогу те же.")
    kept = stories.substitute(facts)
    assert kept["e4.residual_share"].key == "e4.residual_share" and kept["e4.residual_share"].value == 0.316


def test_score_splits_into_facts_and_expert_parts(cfg):
    m = matrix_for(REAL_LIKE, cfg)
    w = stories.check_weights(cfg["eda"]["matrix_weights"])
    w_facts = sum(w[c] for c in stories.FACT_CRITERIA)
    assert np.allclose(m["score"], w_facts * m["score_facts"] + (1 - w_facts) * m["score_expert"])
    assert m.loc["С4", "score_facts"] == pytest.approx(3.75)  # сигнал 3, покрытие 5
    assert m.loc["С5", "score_facts"] == pytest.approx(4.375)  # сигнал 4, покрытие 5
    assert stories.facts_leader(m.reset_index())["sid"] == "С5"
    assert np.isnan(stories.part_score({"signal": None, "coverage": 5}, w, stories.FACT_CRITERIA))


def test_lead_margin_says_when_expert_scores_decide(cfg):
    w = stories.check_weights(cfg["eda"]["matrix_weights"])
    # никто не прошёл критерии: лидер — лучший по итогу, разрыв 0,35 больше веса любого балла
    m = synthesis.decision_matrix(facts_of(REAL_LIKE), cfg)
    info = stories.lead(m, w)
    assert (info.top, info.rival) == ("С5", "С4") and info.margin == pytest.approx(0.35) and info.flips == ()
    assert "порядок не меняет" in flat(stories.lead_text(m, w))
    advice = flat(stories.advice_text(m, w, ["Строка прототипа."]))
    assert advice.startswith("Ни у одного сюжета") and "С5 впереди и по одним фактам" in advice
    assert advice.endswith("Строка прототипа.")
    # С4 и С5 проходят, разрыв 0,10 меньше веса балла сети и пользы (0,20): порядок решают экспертные баллы
    close = {**REAL_LIKE, **C5_PASSES, "e4.residual_share": (0.35, "pct")}
    m = synthesis.decision_matrix(facts_of(close), cfg)
    info = stories.lead(m, w)
    assert (info.top, info.rival) == ("С5", "С4") and info.margin == pytest.approx(0.10)
    assert set(info.flips) == {"network_dynamics", "practical"}
    text = flat(stories.lead_text(m, w))
    assert "меняет их порядок: при таком разрыве первое место определяют экспертные баллы" in text
    assert "а не данные" not in text
    assert "Все критерии отказа выполнены у С4 «Состав корзины» и С5" in flat(stories.advice_text(m, w))
    verdict = flat(stories.verdict_text(m))
    assert verdict.startswith("Все критерии отказа прошли 2 сюжета: С5 «Сдвиг к маркетплейсам» (3,85)")
    assert "С4 «Состав корзины» (3,75)" in verdict
    single = m.iloc[:1]  # один сюжет — сравнивать не с кем
    assert stories.lead(single, w) is None and stories.lead_text(single, w) == ""


def test_failed_story_is_never_main_whatever_its_score(cfg):
    values = {**REAL_LIKE, **C5_PASSES}
    values["e4.residual_share"] = (0.45, "pct")  # сигнал С4 — 5, итог 4,00 …
    values["e4.residual_pc1_stability"] = (0.5, "rho")  # … но ось неустойчива: С4 — слой
    m = synthesis.decision_matrix(facts_of(values), cfg)
    ranked = stories.ranking(m)
    assert ranked["sid"].iloc[0] == "С4"  # по баллу выше всех — слой
    top, second = stories.pick_top(m)
    assert top["sid"] == "С5" and second["sid"] == "С4" and second["status"] == stories.STATUS_LAYER
    text = stories.verdict_text(m)
    assert text.index("С5") < text.index("С4") and "слой" in text and "прошёл только" in text


def test_no_story_passes(cfg):
    m = synthesis.decision_matrix(facts_of(REAL_LIKE), cfg)
    top, second = stories.pick_top(m)
    assert top is None and second["sid"] == "С5"
    verdict = flat(stories.verdict_text(m))
    assert verdict.startswith("Ни один сюжет не прошёл")
    assert "выше всех по итогу — С5 «Сдвиг к маркетплейсам» (3,85; слой), у него не выполнено:" in verdict
    assert "сверх уровня трат и региона (ρ остатков МНК) 0,44, нужно ≥ 0,50." in verdict


def test_missing_section_facts_mean_no_data(cfg):
    values = {k: v for k, v in {**REAL_LIKE, **C5_PASSES}.items() if not k.startswith("e3.")}
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
        assert s.main and s.external and s.example is not None and s.user  # проверки и польза есть у всех
    broken = copy.deepcopy(stories.STORIES[0].manual)
    broken["practical"] = (7, "")
    bad = stories.Story(
        "С9", "x", "y", broken, stories.STORIES[0].signal, stories.STORIES[0].coverage, (), "s9"
    )
    with pytest.raises(ValueError, match="вне 1–5"):
        stories.validate_stories([bad])
    rej = stories.Rejection("r", "e1.x", ">=", "step_test_pp", "x", alts=(("e1.y",),))
    odd = stories.Story("С9", "x", "y", stories.STORIES[0].manual, bad.signal, bad.coverage, (rej,), "s9")
    with pytest.raises(ValueError, match="пары"):
        stories.validate_stories([odd])
    sign = stories.Story(
        "С9",
        "x",
        "y",
        stories.STORIES[0].manual,
        bad.signal,
        bad.coverage,
        (),
        "s9",
        external=(stories.External("a", 0, "нет знака"),),
    )
    with pytest.raises(ValueError, match="знак"):
        stories.validate_stories([sign])
    # С4: сила сигнала — доля остатка CLR; у С5 — критерий сигнала сверх уровня и региона; С2 — без ВГТ
    by = {s.sid: s for s in stories.STORIES}
    assert by["С4"].signal.fact_key == "e4.residual_share"
    assert "syn.mp_split_half_resid" in {r.fact_key for r in by["С5"].rejections}
    assert by["С2"].rejections[0].fact_key == "e2.partial_rho_access_within_no_inner"
    assert by["С5"].manual["network_dynamics"][0] == 3


def test_params_reject_unknown_keys(cfg, caplog):
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
    # старый порог s4 (устойчивость очищенной оси) не используется: предупреждение, а не ошибка
    data["eda"]["stories"] = {"signal_bins": {"s4": [0.5, 0.6, 0.7, 0.85], "s9_old": [0.1, 0.2, 0.3, 0.4]}}
    stories._WARNED.clear()
    with caplog.at_level(logging.WARNING, logger="munnet.eda.stories"):
        prm = stories.params(Config(data, cfg.path))
    assert "s9_old" in caplog.text and prm["signal_bins"]["s4_residual"] == [0.1, 0.2, 0.3, 0.4]


# --- Показатели МО и «регион или место» ----------------------------------------------------------


def test_story_facts_exist_in_sections(synthetic):
    """Контракт с разделами: каждый факт, на который опираются баллы и критерии, разделы пишут."""
    facts = {k for f in synthetic["findings"] for k in f.facts} | set(synthetic["syn"].facts)
    needed = set()
    for s in stories.STORIES:
        needed |= {s.signal.fact_key, s.coverage.fact_key, s.coverage.denominator}
        needed |= {r.fact_key for r in s.rejections} | {r.times_fact for r in s.rejections if r.times_fact}
        needed |= {k for r in s.rejections for k, _ in r.ref}
    needed |= set(stories.NO_INNER.values())
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
    region_mean = mo["region_code"].map(rng.normal(size=mo["region_code"].max() + 1).__getitem__)
    ind = pd.DataFrame(
        {
            "territory_id": mo["territory_id"],
            "level": mo["log_level_2024"],
            "noise": rng.normal(size=len(mo)),
            "flag": (rng.random(len(mo)) > 0.5).astype(float),
            "basket_resid_pc1": rng.normal(size=len(mo)) + region_mean,
        }
    )
    t = synthesis.region_or_place(
        ind, mo, cfg, np.random.default_rng(0), reliability={"noise": 0.5, "level": 1.0}
    ).set_index("indicator")
    assert "flag" not in t.index  # флаг (0/1) не описывается η² и I Морана
    assert t.loc["level", "eta2"] > 0.5 > t.loc["noise", "eta2"] + 0.2
    assert t.loc["level", "moran_i"] > t.loc["noise", "moran_i"] and t.loc["level", "moran_p"] < 0.01
    assert bool(t.loc["level", "regional"]) and not bool(t.loc["noise", "regional"])
    n_inner = int(mo["is_inner_city"].sum())
    assert t.loc["noise", "n_no_inner"] == t.loc["noise", "n"] - n_inner
    assert list(t.index) == ["level", "noise", "basket_resid_pc1"]  # по убыванию η², без η² — в конце
    for k in cfg["eda"]["knn_sensitivity"]:
        assert f"moran_i_k{k}" in t.columns
    # очищенный от региона показатель: η² региона и округа не считаются (0 по построению), I Морана — да
    built = t.loc["basket_resid_pc1"]
    assert bool(built["by_construction"]) and np.isnan(built["eta2"]) and np.isnan(built["eta2_fd"])
    assert np.isfinite(built["moran_i"]) and np.isfinite(built["eta2_type"]) and not bool(built["regional"])
    # η² с поправкой на шум: η² / надёжность, не больше 1; η² чистого шума — (k − 1) / (n − 1)
    assert t.loc["noise", "eta2_true"] == pytest.approx(min(t.loc["noise", "eta2"] / 0.5, 1.0))
    k = mo.loc[mo["log_level_2024"].notna(), "region_code"].nunique()
    assert t.loc["level", "eta2_random"] == pytest.approx((k - 1) / (t.loc["level", "n"] - 1))
    for g in synthesis.GROUPINGS:
        if g != "region":
            assert f"eta2_{g}" in t.columns
    again = synthesis.region_or_place(
        ind, mo, cfg, np.random.default_rng(0), reliability={"noise": 0.5, "level": 1.0}
    ).set_index("indicator")
    pd.testing.assert_frame_equal(t, again)  # один генератор — одинаковые p


def test_groupings_and_reliability(synthetic):
    data, cfg = synthetic["data"], synthetic["cfg"]
    g = synthesis.groupings(data.mo, cfg)
    assert list(g.columns) == list(synthesis.GROUPINGS) and len(g) == len(data.mo)
    assert g["size"].dropna().nunique() == synthesis.DEFAULTS["size_groups"]
    assert set(g["urban"].dropna().unique()) <= {0.0, 1.0, 2.0}
    assert g["fd"].notna().all()  # коды субъектов синтетики — настоящие коды справочника
    assert synthesis.spearman_brown(0.683) == pytest.approx(2 * 0.683 / 1.683)
    assert np.isnan(synthesis.spearman_brown(-0.1)) and np.isnan(synthesis.spearman_brown(float("nan")))
    facts = facts_of({"e4.growth_half_consistency": (0.353, "rho"), "e4.mp_split_half": (0.683, "rho")})
    rel = synthesis.reliabilities(data.mo, facts, cfg)
    assert rel["growth_log"] == pytest.approx(synthesis.spearman_brown(0.353))
    assert rel["mp_pp_change"] == pytest.approx(synthesis.spearman_brown(0.683))
    assert 0 < rel["log_level_2024"] <= 1 and "clr_cafe_2024" in rel


def test_classify_and_headline_follow_data(synthetic):
    prm = synthesis.params(synthetic["cfg"])
    base = {"log_level_2024": (0.77, np.nan), "growth_log": (0.29, 0.55), "mp_pp_change": (0.41, 0.50)}

    def classify(values):
        t = pd.DataFrame(
            {
                "indicator": list(values),
                "eta2": [v[0] for v in values.values()],
                "eta2_true": [v[1] for v in values.values()],
            }
        )
        return synthesis.classify(t, prm)

    rp = classify(base)
    assert rp.level_regional and rp.gap_ok and rp.classes == {"growth_log": "half", "mp_pp_change": "half"}
    title = flat(synthesis.headline_text(rp, prm["headline"]["dynamic"]))
    assert title == "Регион задаёт уровень трат (η² 0,77), а рост и прирост доли маркетплейсов — наполовину"
    assert len(title) <= 90 and "онлайн" not in title
    rp = classify(dict(base, mp_pp_change=(0.2, 0.25)))  # с поправкой меньше 0,35 — «меньше половины»
    assert rp.place == ("mp_pp_change",)
    title = flat(synthesis.headline_text(rp, prm["headline"]["dynamic"], max_len=200))
    assert "рост — наполовину; прирост доли маркетплейсов — меньше чем наполовину" in title
    short = flat(synthesis.headline_text(rp, prm["headline"]["dynamic"], max_len=60))
    assert short == "Регион задаёт уровень трат (η² 0,77)"  # не влезло — без перечня показателей
    rp = classify(dict(base, growth_log=(0.29, np.nan)))  # без надёжности — по η² как есть
    assert rp.classes["growth_log"] == "place"
    rp = classify(dict(base, log_level_2024=(0.5, np.nan)))
    assert not rp.level_regional and not rp.gap_ok


def test_story_checks_on_synthetic(synthetic):
    """Проверки сюжетов теми же функциями, что разделы: без внутригородских, сверх уровня, части корзины."""
    data, findings = synthetic["data"], synthetic["findings"]
    halves = synthesis.s4_basket.half_year_growth(data.panel_wide)
    values = synthesis.no_inner_values(data, findings, halves)
    for key in ("residual_share_no_inner", "mp_split_half_no_inner", "reliable_share_no_inner"):
        assert key in values and np.isfinite(values[key])
    mp = synthesis.mp_residual(data.mo, halves)
    assert -1 <= mp["mp_split_half_resid"] <= 1 and 0 <= mp["mp_r2_level"] <= mp["mp_r2_level_region"] <= 1
    parts = synthesis.part_stability(data.mo)
    assert list(parts.index) == list(synthesis.s4_basket.PARTS) and parts.between(-1, 1).all()
    spread = synthesis.share_spread(data.mo)
    assert spread["iqr_pp"].is_monotonic_decreasing and (spread["iqr_pp"] > 0).all()
    pairs = synthesis.mp_pair_signal(data, np.random.default_rng(0), 500)
    assert 0 <= pairs["mp_common_share"] <= 1 and pairs["mp_pair_raw_q50"] > pairs["mp_pair_own_q90"] - 1


def test_prototype_recovers_a_planted_grouping(synthetic):
    """k-means по признаку, который и есть простое деление МО, даёт AMI 1 с ним и ARI 1 на двух замерах."""
    cfg, mo = synthetic["cfg"], synthetic["data"].mo
    groups = synthesis.groupings(mo, cfg).set_axis(mo["territory_id"])
    spend = groups["spend"].dropna()
    x = pd.DataFrame({"x": spend * 10.0 + np.linspace(0, 0.1, len(spend))}, index=spend.index)
    proto = synthesis.ProtoSet("s9", "группа по уровню", x, (x, x.copy()), "дважды")
    t = synthesis.prototype_table([proto], groups, cfg).iloc[0]
    assert t["ami_spend"] == pytest.approx(1.0) and t["ami_max_by"] == synthesis.GROUPINGS["spend"]
    assert t["ari_pair"] == pytest.approx(1.0) and t["k"] == synthesis.DEFAULTS["prototype"]["k"]
    text = flat(synthesis.prototype_text(pd.DataFrame([t]), {"s9": "С9"}))
    assert "типы С9 (группа по уровню трат, AMI 1,00)" in text and "С9 — 1,00" in text


def test_external_check_and_example(synthetic):
    data, findings = synthetic["data"], synthetic["findings"]
    ind, _ = synthesis.collect_indicators(data, findings, synthetic["cfg"])
    table = synthesis.story_table(data, ind)
    ext = pd.DataFrame(index=table.index)
    rng = np.random.default_rng(3)
    ext["good"] = table["mp_pp_change"] * 2 + rng.normal(scale=0.01, size=len(table))
    ext["bad"] = -ext["good"]
    ext["workplace_based"] = False
    story = stories.Story(
        "С9",
        "x",
        "y",
        stories.STORIES[0].manual,
        stories.STORIES[0].signal,
        stories.STORIES[0].coverage,
        (),
        "s9",
        main="mp_pp_change",
        external=(stories.External("good", 1, "хорошая"), stories.External("bad", 1, "обратная")),
        example=stories.Example("mp_pp_change", "id", "pp", ("log_level_2023",), "прирост доли"),
    )
    rows = synthesis.external_checks(story, table, ext, 0.1)
    assert [r["verdict"] for r in rows] == ["знак ожидаемый", "знак обратный"]
    assert rows[0]["rho"] > 0.9 and rows[0]["n"] > 10
    text = flat(synthesis.external_text(rows))
    assert text.startswith("хорошая — ожидается «+», ρ внутри регионов") and "; обратная — " in text
    ex = synthesis.example_mo(story, table, 0.0)
    assert ex is not None and ex["territory_id"] in table.index
    assert "прирост доли" in ex["text"] and "при ожидаемых по уровню трат и региону" in ex["text"]


def test_run_synthesis_on_synthetic(synthetic):
    syn, out = synthetic["syn"], synthetic["out"]
    assert syn.section == "syn" and syn.headline_errors == []
    assert [f.fid for f in syn.figures] == ["F16"] and [t.tid for t in syn.tables] == ["T13", "T14", "T15"]
    for rec in syn.figures:
        assert (out / rec.png).exists() and (out / rec.data_csv).exists()
    facts = syn.facts
    for k in ("s1", "s2", "s3", "s4", "s5", "s6"):
        for key in ("score", "reject", "status", "external", "example", "user", "proto_note", "main_label"):
            assert f"syn.{key}_{k}" in facts
        assert f"syn.score_{k}_no_inner" in facts and f"syn.status_{k}_no_inner" in facts
    for key in (
        "top_story",
        "second_story",
        "verdict",
        "sensitivity",
        "top_story_no_inner",
        "n_mo",
        "eta2_log_level_2024",
        "moran_log_level_2024",
        "eta2_true_growth_log",
        "reliability_mp_pp_change",
        "eta2_random",
        "share_iqr_argmax_gen",
        "mp_pair_own_q90",
        "resid_part_stability_span",
    ):
        assert f"syn.{key}" in facts
    assert facts["syn.weight_signal"].text == "25" and facts["syn.weight_facts"].text == "40"  # % без «,0»
    assert "syn.moran_log_level_2024_no_inner" in facts and facts["syn.n_mo"].value == len(
        synthetic["data"].mo
    )
    t14 = pd.read_csv(out / syn.tables[1].csv)
    assert list(t14["sid"]) == ["С1", "С2", "С3", "С4", "С5", "С6"]
    assert t14["score"].between(1, 5).all() and {"score_no_inner", "status_no_inner", "failed"} <= set(t14)
    # индикаторы сводки — вся таблица показателей МО с подписями
    assert {"log_level_2024", "own_amplitude", "log_spend_to_ndfl"} <= set(syn.indicators.columns)
    assert set(syn.indicators.columns) - {"territory_id"} == set(syn.indicator_labels)
    # на синтетике уровень задан регионом, рост — нет: заголовок F16 проходит проверку
    assert synthetic["syn"].figures[0].title.startswith("Регион задаёт уровень трат")
    # в T13 очищенная ось корзины — «0 по построению», а не находка «самый местный показатель»
    t13 = next(t for t in syn.tables if t.tid == "T13")
    assert synthesis.BUILT_TEXT in t13.markdown and "Корзина без уровня и региона" in t13.markdown
    assert facts["syn.eta2_basket_resid_pc1"].value is None
    t15 = pd.read_csv(out / syn.tables[2].csv)
    assert set(t15["key"]) == {"level", "s1", "s2", "s3", "s4", "s5", "s6"}
    assert t15["ami_max"].between(-0.1, 1.0).all()


def test_definition_checks_show_other_definitions(synthetic):
    """Сверка с ориентирами Б.4: тот же η² и I Морана в другом определении — факты syn.check_*."""
    cfg, data = synthetic["cfg"], synthetic["data"]
    prm = synthesis.params(cfg)
    t = synthesis.definition_checks(data.mo, cfg, np.random.default_rng(0)).set_index("indicator")
    assert set(t.index) == {col for col, _ in prm["definition_checks"].values()}
    assert set(t["checks"]) == set(prm["definition_checks"]) and t["why"].str.len().gt(0).all()
    expected = stats.eta2(data.mo["log_level_2023"].astype(float), data.mo["region_code"])
    assert t.loc["log_level_2023", "eta2"] == pytest.approx(expected)
    facts = synthetic["syn"].facts
    assert "syn.check_moran_log_level_2023" in facts and "syn.check_eta2_sh_cafe_2024" in facts
    assert "syn.check_moran_log_level_2023" in facts["syn.moran_log_level_2024"].note
    # колонки, которой нет в EdaData.mo, сверка просто не делает
    raw = copy.deepcopy(cfg.data)
    raw["eda"]["synthesis"] = {"definition_checks": {"log_level_{y}": ["nope_{p}", "нет колонки"]}}
    empty = synthesis.definition_checks(data.mo, Config(raw, cfg.path), np.random.default_rng(0))
    assert empty.empty


def test_t14_in_report_shows_scores_and_role(synthetic):
    syn, out = synthetic["syn"], synthetic["out"]
    t14 = next(t for t in syn.tables if t.tid == "T14")
    header = flat(t14.markdown.splitlines()[0])
    for label in ("| Сигнал ×0,25 |", "| Сеть ×0,20 |", "По фактам ×0,40", "Экспертный ×0,60", "| Итог |"):
        assert label in header
    assert "Роль" in header and "Критерии отказа" not in header  # критерии целиком — текстом под таблицей
    assert "вес" not in header  # короткие заголовки: таблица читается на ширине страницы
    full = pd.read_csv(out / t14.csv)
    assert {"criteria", "criteria_all", "score_facts", "score_expert", "role", "manual_note"} <= set(
        full.columns
    )
    for k in ("s1", "s5"):
        for key in ("score_facts", "score_expert", "role", "criteria", "gist"):
            assert f"syn.{key}_{k}" in syn.facts
    for key in ("advice", "lead_note", "lead_margin", "facts_leader", "weight_facts", "weight_expert"):
        assert f"syn.{key}" in syn.facts
    # подзаголовок F16 называет наибольшее n среди показателей, а не число МО панели
    f16 = pd.read_csv(out / syn.figures[0].data_csv)
    assert f"n до {int(f16['n'].max())} МО" in flat(syn.figures[0].subtitle)
    assert "Чёрные точки — η² региона" in flat(syn.figures[0].subtitle)


def test_synthesis_is_reproducible(synthetic):
    cfg, data, findings = synthetic["cfg"], synthetic["data"], synthetic["findings"]
    rng = eda.section_rng(cfg, SYNTHESIS_NUMBER)
    ind, labels = synthesis.collect_indicators(data, findings, cfg)
    t = synthesis.region_or_place(ind, data.mo, cfg, rng, labels=labels)
    first = pd.read_csv(synthetic["out"] / synthetic["syn"].figures[0].data_csv)
    assert np.allclose(first["moran_p"], t["moran_p"]) and list(first["indicator"]) == list(t["indicator"])
