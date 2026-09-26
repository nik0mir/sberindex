"""Раздел E2 «Сколько тратят и от чего это зависит»: чистые функции, ловушки и прогон на синтетике."""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pandas as pd
import pytest
from synth import make_config, make_processed, make_section_context

from munnet import maps, style
from munnet.contracts import CATEGORY_CODES, MO_TYPES
from munnet.eda import run_sections, stats
from munnet.eda import s2_level as s2
from munnet.eda.base import PLACEHOLDER

YEAR, BASE = 2024, 2023

# Обязательные факты раздела (spec_final, Б.4 и Д.3) без префикса «e2.».
REQUIRED_FACTS = (
    "level_median_2024",
    "level_mean_2024",
    "level_wmean_2024",
    "level_median_2023",
    "level_wmean_2023",
    "level_p10_2024",
    "level_p90_2024",
    "p90_p10_2024",
    "eta2_region_level",
    "rho_level_wage",
    "rho_level_wage_within",
    "n_level_wage",
    "spend_to_wage_median",
    "rho_level_access",
    "partial_rho_access_within",
    "rho_level_dist_capital_within",
    "n_access_wage",
    "partial_rho_access_within_lo",
    "partial_rho_access_within_hi",
    *(f"median_level_{t}" for t in MO_TYPES),
)
# Проверочные факты раздела сверх обязательных: устойчивость критерия С2 и расщепление связи с доступностью.
EXTRA_FACTS = (
    "partial_rho_access_within_no_inner",
    "n_partial_access_no_inner",
    "partial_rho_threshold",
    "rho_level_access_no_inner",
    "rho_level_access_rest",
    "n_level_access_rest",
    "median_access_inner",
    "median_access_north",
    "median_access_rest",
    "eta2_region_level_no_inner",
)


def make_mo(n: int = 6, **cols) -> pd.DataFrame:
    """Минимальная таблица ``mo`` для чистых функций: все нужные колонки, значения по умолчанию нейтральны."""
    base = {
        "territory_id": np.arange(1, n + 1, dtype="int32"),
        "name": [f"Муниципальное образование {i}" for i in range(1, n + 1)],
        "name_short": [f"МО {i}" for i in range(1, n + 1)],
        "region_name": ["Регион 1"] * n,
        "region_code": np.ones(n, dtype="int16"),
        "mo_type": ["mr"] * n,
        "is_capital": [False] * n,
        "is_inner_city": [False] * n,
        "point_lat": np.full(n, 55.0),
        "level_2023": np.full(n, 20_000.0),
        "level_2024": np.full(n, 23_000.0),
        "level_rel_2024": np.ones(n),
        "pop_2023": np.full(n, 1000.0),
        "pop_2024": np.full(n, 1000.0),
        "weight": np.full(n, 1000.0),
        "wage_2023": np.full(n, 50_000.0),
        "market_access": np.full(n, 500.0),
        "dist_capital_km": np.full(n, 100.0),
    }
    base.update(cols)
    mo = pd.DataFrame(base)
    mo["spend_to_wage_2023"] = mo["level_2023"] / mo["wage_2023"]
    return mo


# --- level_summary -------------------------------------------------------------------------------


def test_level_summary_typical_mo_and_resident():
    mo = make_mo(
        5,
        level_2024=[10.0, 20.0, 30.0, 40.0, np.nan],  # МО с 11 месяцами 2024 года — пропуск уровня
        pop_2024=[1.0, 1.0, 1.0, 7.0, 100.0],
    )
    out = s2.level_summary(mo)
    assert out["n_level_2024"] == 4
    assert out["level_median_2024"] == pytest.approx(25.0)
    assert out["level_mean_2024"] == pytest.approx(25.0)
    assert out["level_wmean_2024"] == pytest.approx((10 + 20 + 30 + 7 * 40) / 10)  # МО без уровня не входит
    assert out["level_p10_2024"] == pytest.approx(np.quantile([10, 20, 30, 40], 0.1))
    assert out["p90_p10_2024"] == pytest.approx(np.quantile([10, 20, 30, 40], 0.9) / 13.0)
    assert out["wmean_to_median_2024"] == pytest.approx(34.0 / 25.0)
    assert out["n_level_weighted_2024"] == 4


def test_level_summary_weight_falls_back_to_common_weight():
    mo = make_mo(2, level_2024=[10.0, 30.0], pop_2024=[np.nan, 1.0], weight=[3.0, 5.0])
    assert s2.level_summary(mo)["level_wmean_2024"] == pytest.approx((3 * 10 + 1 * 30) / 4)
    mo_no_pop = make_mo(2, level_2024=[10.0, 30.0], pop_2024=[np.nan, 1.0], weight=[np.nan, 5.0])
    out = s2.level_summary(mo_no_pop)
    assert out["level_wmean_2024"] == pytest.approx(30.0)  # МО без населения не входит в среднее по жителям
    assert out["n_level_weighted_2024"] == 1


def test_eta2_is_on_log_level_and_without_inner_city():
    levels = [10_000.0, 20_000.0, 40_000.0, 80_000.0, 30_000.0, 60_000.0]
    mo = make_mo(
        6,
        level_2024=levels,
        region_code=np.array([1, 1, 2, 2, 3, 3], dtype="int16"),
        is_inner_city=[False, False, False, False, True, True],
    )
    out = s2.level_summary(mo)
    assert out["eta2_region_level"] == pytest.approx(stats.eta2(np.log(levels), mo["region_code"]))
    assert out["eta2_region_level"] != pytest.approx(stats.eta2(levels, mo["region_code"]))
    assert out["eta2_region_level_no_inner"] == pytest.approx(stats.eta2(np.log(levels[:4]), [1, 1, 2, 2]))


def test_level_summary_by_type_and_spend_to_wage():
    mo = make_mo(
        4,
        mo_type=["mr", "mr", "vgt", "go"],
        level_2024=[20.0, 30.0, 50.0, np.nan],
        level_2023=[10.0, 20.0, 40.0, 30.0],
        wage_2023=[40.0, 40.0, 40.0, 40.0],
    )
    out = s2.level_summary(mo)
    assert out["median_level_mr"] == pytest.approx(25.0)
    assert out["median_level_vgt"] == pytest.approx(50.0)
    assert np.isnan(out["median_level_go"]) and out["n_level_go"] == 0  # уровня нет — медиана не выдумывается
    assert np.isnan(out["median_level_mo"]) and out["n_level_mo"] == 0
    assert out["spend_to_wage_median"] == pytest.approx(np.median([0.25, 0.5, 1.0, 0.75]))


def test_place_groups_north_excludes_inner_city():
    mo = make_mo(
        5,
        level_rel_2024=[2.0, 1.5, 1.3, 0.9, np.nan],
        is_inner_city=[True, False, False, False, False],
        point_lat=[70.0, 65.0, 61.0, 50.0, 70.0],
        market_access=[900.0, 100.0, 300.0, 400.0, np.nan],
    )
    out = s2.place_groups(mo, 60.0)
    assert out["median_rel_inner"] == pytest.approx(2.0)
    assert out["median_rel_north"] == pytest.approx(1.4)  # внутригородская на 70° — не «Север»
    assert out["n_rel_north"] == 2
    assert out["median_rel_rest"] == pytest.approx(0.9)
    assert out["n_rel"] == 4
    assert out["median_access_inner"] == pytest.approx(900.0)
    assert out["median_access_north"] == pytest.approx(200.0)  # пропуск доступности не входит
    assert out["median_access_rest"] == pytest.approx(400.0)


def test_access_rho_rest_excludes_capitals_and_north():
    """Столицы тянут связь вверх, Север — вниз; у остальных МО связи нет — ρ по ним ≈ 0."""
    rng = np.random.default_rng(11)
    n_rest, n_inner, n_north = 400, 120, 60
    level = np.concatenate(
        [
            rng.normal(20_000, 2_000, n_rest),
            rng.normal(50_000, 3_000, n_inner),
            rng.normal(40_000, 3_000, n_north),
        ]
    )
    access = np.concatenate(
        [rng.uniform(200, 500, n_rest), rng.uniform(800, 1000, n_inner), rng.uniform(100, 200, n_north)]
    )
    mo = make_mo(
        len(level),
        level_2024=level,
        level_2023=level,
        market_access=access,
        is_inner_city=[False] * n_rest + [True] * n_inner + [False] * n_north,
        point_lat=np.r_[np.full(n_rest, 55.0), np.full(n_inner, 55.7), np.full(n_north, 66.0)],
    )
    groups = s2.place_groups(mo, 60.0)
    drivers = s2.level_drivers(mo)
    assert drivers["rho_level_access"] > drivers["rho_level_access_no_inner"]
    assert drivers["rho_level_access_no_inner"] < 0 < drivers["rho_level_access"]
    assert abs(groups["rho_level_access_rest"]) < 0.15
    assert groups["n_level_access_rest"] == n_rest
    split = s2.access_split(
        drivers["rho_level_access"], drivers["rho_level_access_no_inner"], groups["rho_level_access_rest"]
    )
    assert split.startswith(": Москва")


def test_access_split_only_when_data_show_it():
    assert s2.access_split(0.19, -0.15, -0.04).startswith(":")
    assert s2.access_split(0.19, 0.25, 0.30) == ""  # без столиц связь сильнее — фраза неверна
    assert s2.access_split(0.19, -0.15, -0.30) == ""  # без Севера не растёт — про Север не говорим
    assert s2.access_split(float("nan"), -0.15, -0.04) == ""


# --- level_drivers: ловушки связей ---------------------------------------------------------------


def _regional(n_regions: int = 20, per_region: int = 60) -> tuple[pd.DataFrame, np.ndarray]:
    """МО в регионах, разнесённых по шкале: сдвиг региона много больше разброса внутри."""
    region = np.repeat(np.arange(1, n_regions + 1), per_region)
    mo = make_mo(len(region), region_code=region.astype("int16"))
    return mo, 3.0 * region.astype(float)


def test_access_through_wage_only_gives_zero_partial_rho():
    """Ловушка С2: доступность рынков связана с тратами только через зарплату — частный ρ ≈ 0."""
    rng = np.random.default_rng(7)
    mo, offset = _regional()
    n = len(mo)
    log_wage = offset + rng.normal(0, 1, n)
    mo["wage_2023"] = np.exp(10 + 0.1 * log_wage)
    mo["level_2024"] = np.exp(9 + 0.1 * (log_wage + rng.normal(0, 0.3, n)))
    mo["market_access"] = log_wage + rng.normal(0, 0.3, n)
    out = s2.level_drivers(mo)
    assert out["rho_level_access"] > 0.8  # в целом связь сильная — повторяет регион и зарплату
    assert abs(out["partial_rho_access_within"]) < 0.1
    assert out["n_access_wage"] == n
    assert out["n_partial_access"] == n


def test_access_beyond_wage_is_found_within_regions():
    rng = np.random.default_rng(8)
    mo, offset = _regional()
    n = len(mo)
    log_wage = offset + rng.normal(0, 1, n)
    access = rng.normal(0, 1, n)
    mo["wage_2023"] = np.exp(10 + 0.1 * log_wage)
    mo["level_2024"] = np.exp(9 + 0.1 * (log_wage + 1.0 * access + rng.normal(0, 0.3, n)))
    mo["market_access"] = access + offset  # доступность тоже выше в «богатых» регионах
    out = s2.level_drivers(mo)
    assert out["partial_rho_access_within"] > 0.5


def test_partial_rho_without_inner_city_drops_their_effect():
    """Проверка устойчивости критерия С2: если связь доступности с тратами сверх зарплаты есть только
    у внутригородских территорий, без них частный ρ ≈ 0."""
    rng = np.random.default_rng(10)
    mo, offset = _regional()
    n = len(mo)
    log_wage = offset + rng.normal(0, 1, n)
    noise = rng.normal(0, 1, n)
    mo["wage_2023"] = np.exp(10 + 0.1 * log_wage)
    mo["level_2024"] = np.exp(9 + 0.1 * (log_wage + noise))
    inner = mo["region_code"].to_numpy() <= 6  # 6 регионов из 20 — «Москва и Петербург»
    mo["is_inner_city"] = inner
    mo["market_access"] = np.where(inner, noise, rng.normal(0, 1, n)) + offset
    out = s2.level_drivers(mo)
    assert out["partial_rho_access_within"] > 0.15
    assert abs(out["partial_rho_access_within_no_inner"]) < 0.1
    assert out["n_partial_access_no_inner"] == int((~inner).sum())


def test_within_region_rho_ignores_pure_region_effect():
    rng = np.random.default_rng(9)
    mo, offset = _regional()
    n = len(mo)
    mo["level_2024"] = np.exp(9 + 0.1 * (offset + rng.normal(0, 1, n)))
    mo["dist_capital_km"] = 10 * offset + rng.normal(0, 1, n)  # общий только регион
    mo["level_2023"] = np.exp(9 + 0.1 * (offset + rng.normal(0, 1, n)))
    mo["wage_2023"] = np.exp(10 + 0.1 * (offset + rng.normal(0, 1, n)))
    out = s2.level_drivers(mo)
    assert out["rho_level_wage"] > 0.9
    assert abs(out["rho_level_wage_within"]) < 0.15
    assert abs(out["rho_level_dist_capital_within"]) < 0.15


def test_wage_rho_without_inner_city():
    mo = make_mo(
        6,
        level_2023=[1.0, 2.0, 3.0, 4.0, 10.0, 0.5],
        wage_2023=[1.0, 2.0, 3.0, 4.0, 0.5, 10.0],  # у внутригородских зарплата по месту работодателя
        is_inner_city=[False, False, False, False, True, True],
    )
    out = s2.level_drivers(mo)
    assert out["rho_level_wage_no_inner"] == pytest.approx(1.0)
    assert out["n_level_wage_no_inner"] == 4
    assert out["rho_level_wage"] < 0.5


def test_access_rho_without_inner_city_can_flip_sign():
    """Столицы тянут общую связь вверх: без внутригородских территорий знак может смениться (Север)."""
    level = [10.0, 20.0, 30.0, 40.0, 50.0, 100.0, 110.0, 120.0]
    access = [50.0, 40.0, 30.0, 20.0, 10.0, 900.0, 950.0, 990.0]
    inner = [False] * 5 + [True] * 3
    mo = make_mo(8, level_2024=level, level_2023=level, market_access=access, is_inner_city=inner)
    out = s2.level_drivers(mo)
    assert out["rho_level_access"] > 0
    assert out["rho_level_access_no_inner"] == pytest.approx(-1.0)
    assert out["rho_level_access_2023"] == pytest.approx(out["rho_level_access"])


# --- F05: линия и подписи ------------------------------------------------------------------------


def test_wage_fit_line_ignores_inner_city_and_labels_skip_them():
    wage = np.array([20_000, 30_000, 45_000, 60_000, 90_000, 120_000, 150_000, 200_000], dtype=float)
    level = 3.0 * wage**0.8
    level[1] *= 1.3  # выше линии
    level[3] *= 0.7  # ниже линии
    inner = np.array([False] * 6 + [True, True])
    level[inner] *= np.array([2.0, 0.4])  # внутригородские — далеко от линии
    mo = make_mo(8, wage_2023=wage, level_2023=level, is_inner_city=inner)
    fit, slope, intercept = s2.wage_fit(mo)
    assert slope == pytest.approx(0.8, abs=0.05)
    assert fit.loc[~fit["is_inner_city"], "resid_log"].abs().median() < 0.01
    labels = s2.residual_labels(fit, 1)
    assert list(labels["territory_id"]) == [2, 4]  # выше линии, затем ниже; внутригородские не подписаны
    assert list(labels["side"]) == ["выше", "ниже"]
    with_inner = s2.residual_labels(fit, 1, include_inner=True)
    assert set(with_inner["territory_id"]) == {7, 8}


def test_wage_fit_drops_missing_and_nonpositive():
    mo = make_mo(4, wage_2023=[np.nan, 0.0, 40_000.0, 60_000.0], level_2023=[1.0, 1.0, 2.0, np.nan])
    fit, slope, _ = s2.wage_fit(mo)
    assert list(fit["territory_id"]) == [3]
    assert np.isnan(slope)


def test_partial_rho_ci_brackets_estimate_and_is_reproducible():
    rng = np.random.default_rng(8)
    mo, offset = _regional()
    n = len(mo)
    log_wage = offset + rng.normal(0, 1, n)
    access = rng.normal(0, 1, n)
    mo["wage_2023"] = np.exp(10 + 0.1 * log_wage)
    mo["level_2024"] = np.exp(9 + 0.1 * (log_wage + 0.3 * access + rng.normal(0, 0.5, n)))
    mo["market_access"] = access + offset
    est = s2.level_drivers(mo)["partial_rho_access_within"]
    lo, hi = s2.partial_rho_ci(mo, np.random.default_rng([0, 2]), 200)
    assert lo < est < hi
    assert hi - lo < 0.3
    assert (lo, hi) == s2.partial_rho_ci(mo, np.random.default_rng([0, 2]), 200)
    assert all(np.isnan(s2.partial_rho_ci(make_mo(2), np.random.default_rng(0), 10)))


@pytest.mark.parametrize(
    ("args", "expected", "absent"),
    [
        ((0.35, 0.30, 0.40, 0.33, 0.2), "уверенно", "границе"),
        ((-0.35, -0.40, -0.30, -0.33, 0.2), "уверенно", "границе"),  # критерий по модулю
        ((0.10, 0.05, 0.15, 0.10, 0.2), "не выполняется", "границе"),
        ((0.204, 0.153, 0.253, 0.198, 0.2), "по другую сторону", "уверенно"),  # реальные данные 2026-09
        ((0.21, 0.15, 0.25, 0.22, 0.2), "на самой границе", "по другую сторону"),
        ((float("nan"), 0.1, 0.2, 0.1, 0.2), "не посчитан", "границе"),
    ],
)
def test_c2_verdict(args, expected, absent):
    text = s2.c2_verdict(*args)
    assert expected in text and absent not in text
    assert not any(ch.isdigit() for ch in text)  # числа текста — только из фактов


def test_column_limits_moves_points_clear_of_label_columns():
    xlim = (27_400.0, 456_000.0)
    above_x, below_x = [40_000.0, 44_000.0], [100_000.0, 190_000.0]
    x0, x1 = s2.column_limits(xlim, above_x, below_x, (0.02, 0.2), (0.98, 0.2), 0.03)

    def frac(x):
        return (np.log(x) - np.log(x0)) / (np.log(x1) - np.log(x0))

    assert x0 < xlim[0]  # слева места не хватало — ось раздвинута влево
    assert frac(min(above_x)) >= 0.02 + 0.2 + 0.03 - 1e-9
    assert frac(max(below_x)) <= 0.98 - 0.2 - 0.03 + 1e-9
    # места хватает или подписей нет — пределы не меняются
    assert s2.column_limits((10.0, 1000.0), [500.0], [20.0], (0.02, 0.1), (0.98, 0.1), 0.02) == pytest.approx(
        (10.0, 1000.0)
    )
    assert s2.column_limits((10.0, 1000.0), [], [], (0.02, 0.1), (0.98, 0.1), 0.02) == pytest.approx(
        (10.0, 1000.0)
    )


def test_column_order_matches_stack_direction():
    pts = pd.DataFrame({"territory_id": [1, 2, 3, 4], "level": [30.0, 50.0, 40.0, 50.0]})
    assert list(s2.column_order(pts, "above")["territory_id"]) == [2, 4, 3, 1]  # сверху вниз; равные — по id
    assert list(s2.column_order(pts, "below")["territory_id"]) == [1, 3, 2, 4]  # снизу вверх


def test_f05_label_columns_do_not_cover_points_or_each_other():
    """Геометрия F05: колонка «выше» — левее своих точек, «ниже» — правее, подписи не пересекаются."""
    from matplotlib.text import Annotation, Text

    rng = np.random.default_rng(3)
    n = 300
    wage = np.exp(rng.uniform(np.log(30_000), np.log(200_000), n))
    wage[:3] = [30_500.0, 31_000.0, 31_500.0]  # выше линии — у самой левой границы зарплат
    wage[3:6] = [190_000.0, 195_000.0, 199_000.0]  # ниже линии — у правой границы
    level = 0.45 * wage * np.exp(rng.normal(0, 0.08, n))
    level[:3] *= 1.8
    level[3:6] *= 0.5
    mo = make_mo(n, wage_2023=wage, level_2023=level)
    fit, slope, intercept = s2.wage_fit(mo)
    labels = s2.residual_labels(fit, 3)
    assert set(labels["territory_id"]) == {1, 2, 3, 4, 5, 6}
    with style.use():
        fig = s2.plot_wage(fit, labels, slope, intercept, s2.DEFAULTS)
        ax = fig.axes[0]
        fig.canvas.draw()  # положения подписей аннотаций считаются при отрисовке
        renderer = fig.canvas.get_renderer()
        anns = [a for a in ax.texts if isinstance(a, Annotation) and a.arrowprops]
        assert len(anns) == 6
        boxes = []
        for a in anns:
            box = Text.get_window_extent(a, renderer=renderer)  # только текст, без выносной линии
            px = ax.transData.transform(a.xy)[0]
            if a.get_horizontalalignment() == "left":  # колонка «выше»
                assert box.x1 < px
            else:
                assert box.x0 > px
            boxes.append(box)
        assert not any(b.overlaps(c) for i, b in enumerate(boxes) for c in boxes[i + 1 :])


@pytest.mark.parametrize(("months", "text"), [(6, "меньше 6 месяцев данных"), (1, "меньше 1 месяца данных")])
def test_na_label(months, text):
    assert s2.na_label(months) == text


@pytest.mark.parametrize(
    ("name", "mo_type", "expected"),
    [
        ("Яльчикский", "mo", "Яльчикский округ"),
        ("Красноярский", "mr", "Красноярский район"),
        ("Анадырь", "go", "Анадырь"),
        ("Арбат", "vgt", "Арбат"),
        ("Город Бердск", "mr", "Город Бердск"),
        ("Нижнеколымский", "go", "Нижнеколымский"),
    ],
)
def test_display_name_adds_type_noun_to_adjectives(name, mo_type, expected):
    assert s2.display_name(name, mo_type) == expected


def test_names_with_regions_groups_regions_without_repeats():
    text = s2.names_with_regions(
        ["Арбат", "Хамовники", "Анадырь", "Раменки"], ["Москва", "Москва", "Чукотский АО", "Москва"]
    )
    assert text == "Арбат, Хамовники и Раменки (Москва), Анадырь (Чукотский АО)"
    assert s2.names_with_regions(["Яльчикский округ"], ["Чувашская Республика"]) == (
        "Яльчикский округ (Чувашская Республика)"
    )


def test_point_label_shortens_region():
    assert s2.point_label("Красноярский район", "Астраханская область") == (
        "Красноярский район" + chr(10) + "Астраханская обл."
    )
    assert s2.short_region("Республика Татарстан") == "Респ. Татарстан"
    assert s2.short_region("Ханты-Мансийский автономный округ — Югра") == "Ханты-Мансийский АО — Югра"


# --- T03: выбросы --------------------------------------------------------------------------------


def test_extremes_order_ties_region_ratio_and_annotations():
    mo = make_mo(
        7,
        level_rel_2024=[3.0, 2.0, 2.0, 1.0, 0.5, 0.4, np.nan],
        region_code=np.array([1, 1, 2, 2, 2, 3, 3], dtype="int16"),
        region_name=["А", "А", "Б", "Б", "Б", "В", "В"],
        level_2024=[60_000.0, 40_000.0, 40_000.0, 20_000.0, np.nan, 8_000.0, 1.0],
    )
    table = s2.extremes(mo, {"3": "пояснение из конфига", 6: "второе"}, 2)
    top = table.loc[table["group"] == s2.GROUP_TOP]
    bottom = table.loc[table["group"] == s2.GROUP_BOTTOM]
    assert list(top["territory_id"]) == [1, 2]  # 2 и 3 равны — раньше меньший territory_id
    assert list(bottom["territory_id"]) == [6, 5]  # снизу — по возрастанию; МО без level_rel нет
    assert list(top["rank"]) == [1, 2]
    row5 = table.set_index("territory_id").loc[5]
    assert row5["rel_region"] == pytest.approx(0.5 / 1.0)  # медиана региона Б — 1,0
    assert np.isnan(row5["level"])  # неполный год: уровня нет, отношение есть
    notes = s2.normalize_annotations({"3": "пояснение из конфига", 6: "второе"})
    assert notes == {3: "пояснение из конфига", 6: "второе"}
    assert table.set_index("territory_id").loc[6, "annotation"] == "второе"
    assert (table["mo_type"] == "муниципальный район").all()


def test_extremes_robust_z_within_type_and_outlier_flag():
    rel = [1.0, 1.1, 0.9, 1.05, 0.95, 5.0, 2.0, 2.2, 1.8, 2.1]
    types = ["mr"] * 6 + ["vgt"] * 4
    mo = make_mo(10, level_rel_2024=rel, mo_type=types)
    table = s2.extremes(mo, None, 5, outlier_z=3.5).set_index("territory_id")
    expected = stats.robust_z(np.log(pd.Series(rel)), groups=pd.Series(types))
    assert table.loc[6, "z_type"] == pytest.approx(expected[5])
    assert bool(table.loc[6, "outlier"])
    assert table.loc[7, "z_type"] == pytest.approx(expected[6])  # внутригородская — среди своих, не выброс
    assert not bool(table.loc[7, "outlier"])


def test_context_hints_threshold_order_flags_and_skip():
    n = 30
    wage = np.full(n, 50_000.0) * np.exp(np.linspace(-0.1, 0.1, n))
    wage[0] = 500_000.0  # очень высокая зарплата
    urban = np.linspace(0.4, 0.6, n)
    urban[0] = 0.05  # мало горожан, |z| меньше, чем у зарплаты
    dist = np.linspace(50, 150, n)
    dist[1] = 0.0  # столица: расстояние 0 — не подсказка
    mo = make_mo(
        n,
        wage_2023=wage,
        urban_share_2023=urban,
        dist_capital_km=dist,
        is_capital=[False, True] + [False] * (n - 2),
    )
    hints = s2.context_hints(mo, 2.0, 1)
    inner = mo.assign(is_inner_city=[True] + [False] * (n - 1))
    assert "зарплата" not in s2.context_hints(inner, 2.0, 3)[1]  # у внутригородских — по месту работы
    assert hints[1].startswith("высокая зарплата (z")
    assert "горожан" not in hints[1]  # только один признак — с наибольшим |z|
    assert hints[2] == "столица региона"
    assert hints[15] == ""
    both = s2.context_hints(mo, 2.0, 3)
    assert "мало горожан" in both[1]
    assert both[1].index("зарплата") < both[1].index("горожан")


def test_display_order_puts_both_ends_into_markdown_head():
    mo = make_mo(30, level_rel_2024=np.linspace(0.5, 2.0, 30))
    table = s2.extremes(mo, None, 10)
    shown = s2.display_order(table, 7)
    head = shown.head(14)
    assert (head["group"] == s2.GROUP_TOP).sum() == 7 and (head["group"] == s2.GROUP_BOTTOM).sum() == 7
    assert len(shown) == 20
    assert list(head.loc[head["group"] == s2.GROUP_TOP, "rank"]) == list(range(1, 8))


# --- T04: типичное МО и типичный житель ----------------------------------------------------------


def _wide(values: dict[int, tuple[float, int]], year: int = YEAR) -> pd.DataFrame:
    """panel_wide: МО -> (v_all, число месяцев года); доли частей постоянны."""
    rows = []
    for tid, (v_all, months) in values.items():
        for m in range(1, months + 1):
            row = {"territory_id": tid, "year": year, "month": m, "v_all": v_all}
            parts = {"food": 0.4, "marketplace": 0.1, "transport": 0.05, "health": 0.05, "cafe": 0.02}
            for c, sh in parts.items():
                row[f"v_{c}"] = v_all * sh
            row["v_other"] = v_all * (1 - sum(parts.values()))
            rows.append(row)
    return pd.DataFrame(rows)


def test_weighting_table_full_year_only_and_weights():
    wide = _wide({1: (10_000, 12), 2: (30_000, 12), 3: (99_000, 11)})
    mo = make_mo(3, pop_2024=[3.0, 1.0, 100.0])
    table = s2.weighting_table(wide, mo)
    assert list(table["category"]) == list(CATEGORY_CODES)
    row = table.set_index("category").loc["all"]
    assert row["n_mo"] == 2  # МО с 11 месяцами не входит
    assert row["median_mo"] == pytest.approx(20_000)
    assert row["mean_mo"] == pytest.approx(20_000)
    assert row["wmean_residents"] == pytest.approx((3 * 10_000 + 30_000) / 4)
    assert row["ratio_wmean_median"] == pytest.approx(15_000 / 20_000)
    cafe = table.set_index("category").loc["cafe"]
    assert cafe["median_mo"] == pytest.approx(0.02 * 20_000)
    assert table.set_index("category").loc["food", "label"] == style.LABELS["food"]


def test_weighting_all_row_matches_level_summary_on_synthetic(eda_data):
    table = s2.weighting_table(eda_data.panel_wide, eda_data.mo).set_index("category")
    summary = s2.level_summary(eda_data.mo)
    assert table.loc["all", "median_mo"] == pytest.approx(summary["level_median_2024"])
    assert table.loc["all", "wmean_residents"] == pytest.approx(summary["level_wmean_2024"])
    assert table.loc["all", "n_mo"] == summary["n_level_2024"]


# --- F04: классы карты ---------------------------------------------------------------------------


def test_map_classes_match_map_bins_and_population():
    values = pd.Series(np.linspace(0.5, 2.0, 21), index=np.arange(101, 122))
    weights = pd.Series(np.arange(1, 22, dtype=float), index=values.index)
    table, cls = s2.map_classes(values, weights, 7)
    edges, labels = maps.quantile_bins(values, 7, s2.times_fmt)
    assert list(table["label"]) == labels
    assert table["n_mo"].sum() == 21
    assert table["pop_share"].sum() == pytest.approx(1.0)
    assert table["population"].iloc[-1] == pytest.approx(weights[cls == 6].sum())
    assert labels[0].endswith("×") and "–" in labels[0]


# --- Мелкие помощники ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("n", "word"),
    [(1, "регион"), (2, "региона"), (5, "регионов"), (11, "регионов"), (21, "регион"), (8, "регионов")],
)
def test_plural(n, word):
    assert s2.plural(n, ("регион", "региона", "регионов")) == word


def test_axis_ticks_tight_limits_with_pad():
    """Ось не тянется до далёкого деления: пределы — края данных с запасом, деления — внутри них."""
    cand = [10_000, 15_000, 20_000, 30_000, 50_000, 70_000, 100_000]
    ticks, lim = s2.axis_ticks(cand, 14_246, 75_367, 1.06)
    assert ticks == [15_000, 20_000, 30_000, 50_000, 70_000]
    assert lim == pytest.approx((14_246 / 1.06, 75_367 * 1.06))
    ticks, lim = s2.axis_ticks([10, 15, 20, 30, 50, 70, 100], 12, 60)  # без запаса — строго края данных
    assert ticks == [15, 20, 30, 50] and lim == (12, 60)


def test_axis_ticks_falls_back_to_nearest_outer_ticks():
    ticks, lim = s2.axis_ticks([10, 20, 50, 100], 22, 45, 1.05)  # внутри запаса кандидатов нет
    assert ticks == [20.0, 50.0]
    assert lim == pytest.approx((20, 50))
    ticks, lim = s2.axis_ticks([10, 20], 50, 60)  # снаружи сверху делений нет — края данных
    assert ticks == [50.0, 60.0] and lim == (50, 60)


def test_typo_keeps_short_words_and_dash_attached():
    text = s2.typo("а в среднем по жителям — 37 тыс. ₽ и к медиане")
    nb = style.NBSP
    assert text == f"а{nb}в{nb}среднем по{nb}жителям{nb}— 37 тыс. ₽ и{nb}к{nb}медиане"
    assert s2.typo("Москва") == "Москва"
    assert s2.typo("На Севере") == f"На{nb}Севере"  # регистр не важен
    assert s2.typo("помощь по-русски") == "помощь по-русски"  # часть слова через дефис не трогаем
    assert s2.typo("сто по") == "сто по"  # «по» в конце строки — нечего привязывать


def test_times_fmt():
    assert s2.times_fmt(1.954) == "1,95×"
    assert s2.times_fmt(float("nan")) == style.NA_TEXT


def test_params_override_from_config(tmp_path):
    cfg = make_config(tmp_path)
    cfg.data["eda"]["level"] = {"map_classes": 5, "checks": {"eta2_min": 0.5}}
    p = s2.params(cfg)
    assert p["map_classes"] == 5
    assert p["checks"]["eta2_min"] == 0.5
    assert p["checks"]["rho_wage_min"] == s2.DEFAULTS["checks"]["rho_wage_min"]
    assert s2.DEFAULTS["map_classes"] == 7  # умолчания не портятся


def test_fact_kinds_and_notes_are_defined():
    keys = [
        *REQUIRED_FACTS,
        *EXTRA_FACTS,
        "n_level_2024",
        "n_level_vgt",
        "median_rel_north",
        "pop_share_top_class",
    ]
    for key in keys:
        assert s2.fact_kind(key)
        assert s2.fact_note(key, YEAR, BASE), key
    assert s2.fact_kind("level_median_2024") == "rub"
    assert s2.fact_kind("n_level_wage") == "int"
    assert s2.fact_kind("rho_level_wage") == "rho"
    assert s2.fact_kind("partial_rho_access_within") == "num3"  # у порога нужны три знака: 0,204 и 0,198
    assert s2.fact_kind("median_access_north") == "int"
    assert s2.fact_note("n_level_wage", YEAR, BASE) != s2.fact_note("n_level_2024", YEAR, BASE)
    with pytest.raises(KeyError):
        s2.fact_kind("unknown_fact")


# --- Раздел целиком на синтетике -----------------------------------------------------------------


@pytest.fixture(scope="module")
def synth_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("e2")
    make_processed(root)
    return root


@pytest.fixture(scope="module")
def eda_data(synth_root):
    from munnet.eda.data import load

    return load(make_config(synth_root))


@pytest.fixture(scope="module")
def finding(synth_root, eda_data):
    ctx = make_section_context(synth_root, "e2", data=eda_data)
    with style.use():
        return s2.run_section(ctx)


def test_run_section_outputs_and_required_facts(finding, synth_root):
    assert finding.section == "e2" and finding.title == s2.TITLE
    assert [f.fid for f in finding.figures] == ["F03", "F04", "F05"]
    assert [f.slug for f in finding.figures] == ["level_distribution", "level_map", "level_vs_wage"]
    assert [t.tid for t in finding.tables] == ["T03", "T04"]
    missing = [k for k in (*REQUIRED_FACTS, *EXTRA_FACTS) if f"e2.{k}" not in finding.facts]
    assert not missing
    out = synth_root / "outputs" / "eda"
    f03 = pd.read_csv(out / finding.figures[0].data_csv)
    assert "is_inner_city" in f03.columns and f03["level"].notna().all()  # выделенные столбцы F03 проверяемы
    for rec in finding.figures:
        assert (out / rec.png).exists() and (out / rec.svg).exists() and (out / rec.data_csv).exists()
        assert len(rec.title) <= 90 and len(rec.alt) >= 40 and rec.check
    for rec in finding.tables:
        assert (out / rec.csv).exists()
    placeholders = {f"{s}.{k}" for s, k in PLACEHOLDER.findall(finding.summary_md)}
    assert placeholders and placeholders <= set(finding.facts)
    bare = PLACEHOLDER.sub("", finding.summary_md)
    assert "{" not in bare and "}" not in bare  # вывод о С2 и фраза о доступности подставлены
    assert "критерия отказа С2" in finding.summary_md
    assert finding.caveats


def test_headlines_pass_and_planted_signal_found(finding):
    """Заложенный сигнал синтетики: сильный эффект региона и связь трат с зарплатой (spec_final, Д.3)."""
    assert finding.headline_errors == []
    assert finding.facts["e2.eta2_region_level"].value > 0.5
    assert finding.facts["e2.rho_level_wage"].value > 0.5
    assert finding.facts["e2.level_wmean_2024"].value > finding.facts["e2.level_median_2024"].value
    assert finding.facts["e2.median_level_vgt"].value > finding.facts["e2.median_level_mr"].value


def test_fact_texts_are_russian(finding):
    median = finding.facts["e2.level_median_2024"]
    assert median.kind == "rub" and median.text.endswith(style.NBSP + "₽") and "." not in median.text
    assert finding.facts["e2.spend_to_wage_median"].text.endswith("%")
    assert finding.facts["e2.top_names"].kind == "str" and "(" in finding.facts["e2.top_names"].value


def test_tables_content(finding, synth_root):
    out = synth_root / "outputs" / "eda"
    t03 = pd.read_csv(out / finding.tables[0].csv)
    assert set(t03["group"]) == {s2.GROUP_TOP, s2.GROUP_BOTTOM}
    assert (t03["group"] == s2.GROUP_TOP).sum() == 10
    assert "Чем выделяется" in finding.tables[0].markdown
    assert finding.tables[0].markdown.count("\n") == 2 * s2.DEFAULTS["md_per_side"] + 1
    t04 = pd.read_csv(out / finding.tables[1].csv)
    assert list(t04["category"]) == list(CATEGORY_CODES)


def test_f05_labels_are_limited_and_not_inner_city(finding, synth_root):
    data = pd.read_csv(synth_root / "outputs" / "eda" / finding.figures[2].data_csv)
    labeled = data.loc[data["labeled"]]
    assert 0 < len(labeled) <= 7
    assert not labeled["is_inner_city"].any()


def test_headlines_fail_when_signal_is_absent(tmp_path, eda_data):
    """Проверки заголовков ловят исчезнувший сигнал: регион перемешан, зарплата — шум."""
    rng = np.random.default_rng(0)
    mo = eda_data.mo.copy()
    mo["region_code"] = rng.permutation(mo["region_code"].to_numpy())
    mo["wage_2023"] = rng.permutation(mo["wage_2023"].to_numpy())
    broken = dataclasses.replace(eda_data, mo=mo)
    ctx = make_section_context(tmp_path, "e2", data=broken)
    with style.use():
        finding = s2.run_section(ctx)
    text = " ".join(finding.headline_errors)
    assert "F04" in text and "F05" in text


def test_map_headline_fails_without_north_group(tmp_path, eda_data):
    """Заголовок F04 называет Север: если МО за 60-й параллелью нет, проверка не проходит (NaN не молчит)."""
    mo = eda_data.mo.assign(point_lat=50.0)
    ctx = make_section_context(tmp_path, "e2", data=dataclasses.replace(eda_data, mo=mo))
    with style.use():
        finding = s2.run_section(ctx)
    assert any("F04" in e for e in finding.headline_errors)
    assert not any("F05" in e for e in finding.headline_errors)


def test_residual_name_facts_match_f05_labels(finding, synth_root):
    data = pd.read_csv(synth_root / "outputs" / "eda" / finding.figures[2].data_csv)
    labeled = data.loc[data["labeled"]]
    above = labeled.loc[labeled["resid_log"] > 0, "name"]
    text = finding.facts["e2.wage_above_names"].value
    assert all(name in text for name in above)
    assert finding.facts["e2.wage_below_names"].kind == "str"


def test_runner_writes_only_section_files(tmp_path):
    make_processed(tmp_path)
    cfg = make_config(tmp_path)
    findings = run_sections(cfg, ["e2"])
    assert [f.section for f in findings] == ["e2"]
    out = tmp_path / "outputs" / "eda"
    saved = json.loads((out / "sections" / "e2.json").read_text(encoding="utf-8"))
    assert saved["headline_errors"] == []
    assert "e2.partial_rho_access_within" in saved["facts"]
    assert not (out / "facts.json").exists()
