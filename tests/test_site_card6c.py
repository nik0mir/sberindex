"""Порция 6c лендинга: полосы «Чем отличается тип» по правилу названия (§3.9 п. 4), строка устойчивости
с вариантами, где тип другой, и адрес ``?mo=<название>`` как поиск. Без сети и без data/."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from munnet import landing
from munnet.config import load_config

CFG = load_config("configs/default.yaml")
TPL = Path(landing.__file__).parent / "templates"
PLACE = [str(f) for f in CFG["interpret"]["tests"]["T2_direction"]["place_tree"]["features"]]


def _prof() -> pd.DataFrame:
    rows = [
        # тип 2: log_pop_rel — наибольшее |отклонение| среди признаков места с |δ| ≥ 0,15
        (2, "clr_rel_cafe", -1.0, -0.99),
        (2, "clr_rel_food", 0.85, 0.78),
        (2, "log_pop_rel", -0.70, -0.68),
        (2, "log_level_rel", -0.90, -0.70),  # не признак места правила (без log_level_rel)
        (2, "emp_sh_primary", 0.95, 0.10),  # отклонение больше, но |δ| ниже порога
        # тип 1: ни один признак места не прошёл порог
        (1, "clr_rel_transport", -0.2, -0.21),
        (1, "emp_sh_primary", 0.41, 0.11),
        (1, "market_access_rel", -0.03, -0.12),
    ]
    return pd.DataFrame(rows, columns=["type", "feature", "effect_mad", "cliff"])


def test_naming_place_follows_naming_rule():
    """Признак места — по правилу interpret.naming: порог |δ Клиффа|, затем наибольшее |отклонение медианы|;
    части корзины и log_level_rel не кандидаты; не прошёл никто — None."""
    cl = float(CFG["interpret"]["naming"]["min_abs_cliff"])
    assert landing.naming_place(_prof(), 2, PLACE, cl) == "log_pop_rel"
    assert landing.naming_place(_prof(), 1, PLACE, cl) is None
    assert landing.naming_place(None, 2, PLACE, cl) is None


def test_why_features_two_parts_and_place():
    """Полосы: части корзины из facts.names (до двух) и признак места по правилу; у типа без прошедших
    признаков полос меньше — недостающие не дорисовываются."""
    facts = {
        "names": {
            "1": {"parts": ["clr_rel_transport"], "place": None},
            "2": {"parts": ["clr_rel_cafe", "clr_rel_food"], "place": None},
        }
    }
    d = SimpleNamespace(cfg=CFG, facts=facts, opt=lambda name: _prof() if name == "profile.csv" else None)
    assert landing.why_features(d) == {
        1: ["clr_rel_transport"],
        2: ["clr_rel_cafe", "clr_rel_food", "log_pop_rel"],
    }
    facts["names"]["2"]["place"] = "emp_sh_industry"  # признак места уже в названии — берётся он
    assert landing.why_features(d)[2][-1] == "emp_sh_industry"


def test_check_var_counts():
    """«Тот же тип в N из M» (node_seed) и типы в вариантах (node_r1) — одно и то же; расхождение —
    находка."""
    mo = pd.DataFrame(
        {"id": [1, 2, 3], "t": [1, 3, np.nan], "var": [[3, 1, 3], [4, 3, None], None],
         "rob_rule": ["1/3", "1/2", None]}
    )  # fmt: skip
    assert landing.check_var_counts(mo) == []
    mo.loc[0, "rob_rule"] = "2/3"
    assert landing.check_var_counts(mo) == ["1: 2/3 против [3, 1, 3]"]


def test_var_words_order_matches_var_column():
    """Названия вариантов — в том же порядке, что столбец var (варианты по алфавиту, как в node_tables)."""
    r1 = pd.DataFrame(
        {"territory_id": [1, 1, 1, 1], "kind": ["variant", "variant", "variant", "seed"],
         "variant": ["variant:nodes_separate", "variant:graph_basket_cos", "variant:no_level", "seed:43"],
         "matched_type": [1, 2, 1, 1]}
    )  # fmt: skip
    d = SimpleNamespace(opt=lambda name: r1 if name == "node_r1.csv" else None)
    assert landing.var_words(d) == [
        "другое правило связей между муниципалитетами",
        "без признака уровня трат",
        "районы Москвы и Петербурга отдельно",
    ]


def test_stability_line_names_other_variants_not_list_in_parens():
    """Строка устойчивости: варианты с другим типом названы (nd.var, CARD.var_words); список вариантов
    в скобках сразу после счёта «тот же тип в N из M» больше не ставится (читался как «где тот же»)."""
    js = (TPL / "landing.js").read_text(encoding="utf-8")
    f = js[js.index("function stabilityLine") : js.index("function renderCard")]
    assert "CARD.var_words" in f and "nd.var" in f and "v !== nd.t" in f
    assert "расчёта${vars ? ` (" not in f and "варианты: ${esc(vars)}" in f


def test_mo_param_like_search():
    """?mo=<название>: тот же поиск, что у поля (search: начало названия, «Кириши» -> «Киришский»); одно
    совпадение — карточка, несколько — открывается поиск с запросом; объёмная карта свой разбор не делает."""
    js = (TPL / "landing.js").read_text(encoding="utf-8")
    f = js[js.index("function openParam") :]
    f = f[: f.index("\n}\n")]
    assert "search(v)" in f and "openSearch(v)" in f and "found.length === 1" in f
    assert "openParam();" in js[js.index("renderExamples();") :]
    js3 = (TPL / "landing3d.js").read_text(encoding="utf-8")
    assert "window.munnet && window.munnet.openParam ? null : moFromParam" in js3
