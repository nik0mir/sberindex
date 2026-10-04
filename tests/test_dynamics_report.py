"""Отчёт dynamics: факты прочтения проверки сюжета после вскрытия и охраняющие их утверждения шаблона."""

from __future__ import annotations

import pandas as pd
import pytest

from munnet.dynamics import report as R
from munnet.eda.base import make_fact

PARTS6 = ["food", "marketplace", "transport", "health", "cafe", "other"]


def _driver(ratios: dict[str, float], pvals: dict[str, float], level=(1.03, 0.38)) -> pd.DataFrame:
    rows = [
        {"feature": f"clr_rel_{q}", "is_part": True, "ratio": ratios[q], "p_value": pvals[q]} for q in PARTS6
    ]
    rows.append({"feature": "log_level_rel", "is_part": False, "ratio": level[0], "p_value": level[1]})
    return pd.DataFrame(rows).set_index("feature")


def _types() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "type": [1, 2, 3],
            "clr_rel_cafe": [-0.4, 0.1, 0.8],
            "clr_rel_marketplace": [0.2, 0.0, -0.3],
            "clr_rel_food": [0.1, 0.0, -0.4],
        }
    ).set_index("type")


RATIOS = {"food": 1.4, "marketplace": 1.25, "transport": 1.18, "health": 1.37, "cafe": 2.1, "other": 1.25}
PVALS = {
    "food": 1e-6,
    "marketplace": 0.001,
    "transport": 0.063,
    "health": 3e-4,
    "cafe": 1e-22,
    "other": 0.012,
}


def test_driver_posthoc_facts_from_table():
    f = R.driver_posthoc(_driver(RATIOS, PVALS), _types(), alpha=0.05)
    assert f["drv_n_parts"] == 6
    assert f["drv_n_sig"] == 5
    assert f["drv_ratio_min"] == pytest.approx(1.18)
    assert f["drv_ratio_max"] == pytest.approx(2.1)
    assert f["drv_ratio_min_part"] == "транспорт"
    assert f["drv_nonsig"] == "транспорт (p = 0,063)"
    assert f["drv_cafe_span"] == pytest.approx(1.2)
    assert f["drv_mp_span"] == pytest.approx(0.5)
    assert f["drv_food_span"] == pytest.approx(0.5)


def test_driver_posthoc_lists_all_nonsignificant_parts():
    pv = dict(PVALS, health=0.2)
    f = R.driver_posthoc(_driver(RATIOS, pv), _types(), alpha=0.05)
    assert f["drv_n_sig"] == 4
    assert f["drv_nonsig"] == "транспорт (p = 0,063), здоровье (p = 0,200)"


def _facts(**over) -> dict:
    vals = {
        "drv_n_sig": (5, "int"),
        "drv_n_parts": (6, "int"),
        "drv_level_p": (0.38, "p"),
        "drv_level_ratio": (1.03, "num2"),
        "drv_ratio_min": (1.18, "num2"),
        "driver_alpha": (0.05, "num2"),
        "drv_p": (0.001, "p"),
        "drv_rank": (5, "int"),
        "drv_cafe_span": (1.2, "num2"),
        "drv_mp_span": (0.5, "num2"),
        "drv_food_span": (0.5, "num2"),
        "drv_top": ("общепит", "str"),
    }
    vals.update(over)
    return {f"dyn.{k}": make_fact(f"dyn.{k}", v, kind) for k, (v, kind) in vals.items()}


POSTHOC_CLAIMS = [k for k in R.CLAIMS if k.startswith("после вскрытия")]


def test_posthoc_claims_hold_on_observed_numbers():
    assert len(POSTHOC_CLAIMS) >= 3
    for text in POSTHOC_CLAIMS:
        assert R.CLAIMS[text](_facts()), text


@pytest.mark.parametrize(
    "over",
    [
        {"drv_n_sig": (3, "int")},  # первая часть выполнилась бы не почти для любой части
        {"drv_level_p": (0.01, "p")},  # уровень трат у перешедших менялся бы сильнее
        {"drv_level_ratio": (1.3, "num2")},  # уровень не меньше любой части корзины
        {"drv_rank": (1, "int")},  # по записанному правилу — не «частично»
        {"drv_cafe_span": (0.4, "num2")},  # по общепиту типы различались бы не сильнее всего
    ],
)
def test_posthoc_claims_break_when_numbers_change(over):
    assert not all(R.CLAIMS[t](_facts(**over)) for t in POSTHOC_CLAIMS)
