"""Сборка отчёта cluster на фактах трёх прогонов (seed 42, 43, 44): проверяемые утверждения CLAIMS,
подстановка всех полей шаблона и типографика.

Факты — копии ``outputs/cluster/report_facts.json`` прогонов с seed конфига 42, 43 и 44 (ворота этапа 3):
при seed 43 победитель среди всех семейств другой (спектральная, а не гибрид), поэтому набор проверяет,
что текст шаблона не опирается на числа одного seed. Таблицы и рисунки здесь не собираются (им нужны
все CSV этапа); их директивы заменяются пустой строкой. После изменения фактов этапа файлы надо обновить
копиями из новых прогонов.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from munnet.clustering import report as R
from munnet.contracts import QCError
from munnet.eda import report as eda_report
from munnet.eda.base import Fact

FIXTURES = Path(__file__).parent / "fixtures" / "cluster_report_facts"
SEEDS = (42, 43, 44)


def load_facts(seed: int) -> dict[str, Fact]:
    raw = json.loads((FIXTURES / f"seed{seed}.json").read_text(encoding="utf-8"))
    return {k: Fact(key=k, value=v["value"], kind="str", text=v["text"]) for k, v in raw.items()}


def render_text(facts: dict[str, Fact]) -> str:
    text = R._NOTE.sub("", R.TEMPLATE.read_text(encoding="utf-8"))
    text = eda_report.fill(text, facts)
    text = R._TABLE.sub("", text)
    text = R._FIGURE.sub("", text)
    return eda_report.nbsp_markdown(eda_report.unwrap_paragraphs(text))


def test_fixtures_differ_in_winner_among_all_families():
    winners = {s: load_facts(s)["cl.all_winner"].value for s in SEEDS}
    assert len(set(winners.values())) > 1, winners
    assert {load_facts(s)["cl.final_method"].value for s in SEEDS} == {"hybrid"}


@pytest.mark.parametrize("seed", SEEDS)
def test_claims_hold(seed):
    R.check_claims(load_facts(seed))


@pytest.mark.parametrize("seed", SEEDS)
def test_template_fills_and_passes_typography(seed):
    md = render_text(load_facts(seed))
    assert "{{" not in md
    assert eda_report.lint_ru(md, R.LINT_ALLOW) == []
    # косметика ворот этапа 3: нет «меняют 0 из них: нет» и повтора фразы про один шаг
    assert "0 из них" not in md
    assert md.count("Один шаг по всем парам") == 0


def test_claims_catch_changed_result():
    facts = load_facts(42)
    facts["cl.final_k"] = replace(facts["cl.final_k"], value=3)
    with pytest.raises(QCError, match="итог — гибрид с K = 4"):
        R.check_claims(facts)


def test_claims_catch_fragile_border_lost():
    """«Итог на этой границе хрупок» держится, только пока цепочки с допуском и без дают разное K."""
    facts = load_facts(42)
    k = facts["cl.grid55_strict_hybrid_k"].value
    facts["cl.grid55_tol_hybrid_k"] = replace(facts["cl.grid55_tol_hybrid_k"], value=k)
    with pytest.raises(QCError, match="итог хрупок"):
        R.check_claims(facts)


def test_changes_text_and_verb():
    assert R._changes_text([], "победителя меняют", "победителя не меняет ни одна") == (
        "победителя не меняет ни одна"
    )
    assert R._changes_text(["а", "б"], "победителя меняют", "—") == "победителя меняют: а; б"
    assert R._verb(1) == "меняет"
    assert R._verb(2) == "меняют"


@pytest.mark.parametrize("seed", SEEDS)
def test_raw_graph_sentence_names_actual_z_winner(seed):
    """Фраза о сравнимых метриках графа называет фактического победителя по z-оценкам, а не «гибрид»."""
    facts = load_facts(seed)
    md = render_text(facts).replace("\u00a0", " ")
    winner = f"«{facts['cl.all_winner_method'].value}» с K = {facts['cl.all_winner_k'].value}"
    assert facts["cl.all_winner"].value == winner.replace("«", "").replace("»", "").replace(" с K", ", K")
    assert f"а не победитель по z-оценкам — {winner}" in md
    assert "а не гибрид" not in md


def test_claims_catch_raw_graph_winner_equal_to_z_winner():
    facts = load_facts(42)
    k = facts["cl.raw_l3_winner_graph_k"].value
    facts["cl.all_winner"] = replace(facts["cl.all_winner"], value=f"Leiden, K = {k}")
    with pytest.raises(QCError, match="а не победитель по z-оценкам"):
        R.check_claims(facts)


def test_claims_catch_noisy_graph_not_a_tie():
    """«KEFRiN не лучше методов по X» держится, только пока все ячейки шумового графа — ничья трёх методов."""
    facts = load_facts(43)
    n = facts["cl.syn_noisy_live"].value
    facts["cl.syn_noisy_x_tie"] = replace(facts["cl.syn_noisy_x_tie"], value=n - 1)
    with pytest.raises(QCError, match="ничья по интервалам"):
        R.check_claims(facts)


@pytest.mark.parametrize("seed", SEEDS)
def test_tau_with_k_named_for_both_z_avi_and_z_mq(seed):
    """Фраза «z AVI и z MQ растут с K» даёт минимум τ с K по каждому ряду, а не только по z AVI."""
    facts = load_facts(seed)
    md = render_text(facts).replace(" ", " ")
    avi, mq = facts["cl.tau_k_avi_min"].text, facts["cl.tau_k_mq_min"].text
    assert f"у z AVI не ниже {avi}, у z MQ — не ниже {mq}" in md
    assert "τ Кендалла с K не ниже" not in md


def test_claims_catch_z_mq_not_growing_with_k():
    facts = load_facts(42)
    facts["cl.tau_k_mq_min"] = replace(facts["cl.tau_k_mq_min"], value=0.5)
    with pytest.raises(QCError, match="z AVI и z MQ растут с K"):
        R.check_claims(facts)


@pytest.mark.parametrize("seed", SEEDS)
def test_noisy_graph_cells_counted_with_none(seed):
    """Ячейки шумового графа: «с победителем» не пишется (там ничьи); остальные — «нет», числом."""
    facts = load_facts(seed)
    n, live, none = (facts[f"cl.syn_noisy_{k}"].value for k in ("n", "live", "none"))
    assert live + none == n
    md = render_text(facts).replace(" ", " ")
    assert "ячеек шумового графа с победителем" not in md
    assert f"в остальных {none} из {n} — «нет»" in md
