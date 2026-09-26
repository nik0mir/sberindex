"""Тесты раздела E4 «Корзина и маркетплейсы» (spec_final, Б.4, Д.3): чистые функции и ``run_section``.

Сигналы, которые раздел обязан находить: строки CLR в сумме 0; ρ прироста доли маркетплейсов с уровнем < 0
на заложенной связи; ρ прироста с начальной долей на раздельных месяцах ≈ 0 на чистом шуме (защита от
регрессии к среднему, а без разделения — ложная отрицательная связь); тест ступеньки ловит заложенный скачок.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from synth import make_eda_data, make_section_context

from munnet import style
from munnet.contracts import CATEGORY_CODES, PARTS
from munnet.eda import s4_basket as s4
from munnet.eda import stats
from munnet.eda.base import PLACEHOLDER, Finding

MANDATORY_FACTS = (
    *(f"share_median_{p}" for p in PARTS),
    *(f"clr_var_{p}" for p in PARTS),
    "clr_var_argmax",
    "pc1_var",
    "rho_pc1_level",
    "r2_clr_level",
    "r2_clr_region",
    "r2_clr_both",
    "residual_share",
    "residual_pc1_stability",
    "mp_share_jan23",
    "mp_share_dec24",
    "mp_resident_jan23",
    "mp_resident_dec24",
    "mp_pp_median",
    "mp_pp_q25",
    "mp_pp_q75",
    "rho_mp_pp_level",
    "rho_mp_pp_level_within",
    "rho_mp_pp_initial",
    "mp_split_half",
    "mp_iqr_2023",
    "mp_iqr_2024",
    "mp_sd_log_2023",
    "mp_sd_log_2024",
    "mp_rub_growth_median",
    "rho_mp_rub_growth_level",
    "rho_mp_rub_incr_level",
    "eta2_mp_pp",
    "mp_step_max_pp",
    "mp_step_month",
    "q1_share_2023",
    "q1_share_2024",
    "q5_share_2023",
    "q5_share_2024",
    "dq1_pp",
    "dq5_pp",
    "growth_median",
    "growth_p10",
    "growth_p90",
    *(f"growth_{c}" for c in CATEGORY_CODES),
    "rho_growth_level",
    "rho_growth_level_within",
    "sd_loglevel_2023",
    "sd_loglevel_2024",
    "growth_half_consistency",
    "n_basket",  # Д.3: покрытие сюжета С4
)


# --- Общие данные ---------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    return make_eda_data(tmp_path_factory.mktemp("e4_data"))


def run(tmp_path: Path, data, **eda_overrides) -> tuple[Finding, object]:
    """``run_section`` на синтетике; ``eda_overrides`` — ключи секции ``eda`` конфига для этого прогона."""
    ctx = make_section_context(tmp_path, "e4", data=data)
    ctx.cfg.data["eda"].update(eda_overrides)
    with style.use():
        return s4.run_section(ctx), ctx


@pytest.fixture(scope="module")
def finding(tmp_path_factory, data):
    root = tmp_path_factory.mktemp("e4_run")
    result, ctx = run(root, data)
    return result, ctx.out_dir


def month_rows(n_mo: int, share: np.ndarray, v_all: float = 20_000.0) -> pd.DataFrame:
    """``panel_wide`` из долей маркетплейсов ``share[i, t]`` (МО × 24 месяца) при постоянных тратах."""
    t = np.arange(24)
    tid = np.repeat(np.arange(1, n_mo + 1), 24)
    wide = pd.DataFrame(
        {
            "territory_id": tid.astype("int32"),
            "year": np.tile(2023 + t // 12, n_mo).astype("int16"),
            "month": np.tile(t % 12 + 1, n_mo).astype("int8"),
            "t": np.tile(t, n_mo).astype("int8"),
        }
    )
    wide["date"] = [f"{y}-{m:02d}" for y, m in zip(wide["year"], wide["month"], strict=True)]
    wide["v_all"] = v_all
    wide["v_marketplace"] = np.round(v_all * share.ravel())
    for c in CATEGORY_CODES:
        if f"v_{c}" not in wide:
            wide[f"v_{c}"] = v_all * 0.1
    return wide


def mo_from_wide(wide: pd.DataFrame, rng: np.random.Generator, n_regions: int = 10) -> pd.DataFrame:
    """Минимальная ``mo`` для проверок маркетплейсов: годовые доли, прирост, уровень, регион."""
    ann = wide.groupby(["territory_id", "year"])[["v_marketplace", "v_all"]].sum()
    share = (ann["v_marketplace"] / ann["v_all"]).unstack("year")
    mo = pd.DataFrame({"territory_id": share.index.astype("int32")})
    mo["sh_marketplace_2023"] = share[2023].to_numpy()
    mo["sh_marketplace_2024"] = share[2024].to_numpy()
    mo["mp_pp_change"] = 100 * (mo["sh_marketplace_2024"] - mo["sh_marketplace_2023"])
    mo["log_level_2023"] = rng.normal(10, 0.3, len(mo))
    mo["region_code"] = rng.integers(0, n_regions, len(mo))
    return mo


def national_mp(values: dict[str, float]) -> pd.DataFrame:
    """``national`` только для маркетплейсов: медиана долей и доля в тратах жителей по месяцам."""
    return pd.DataFrame(
        {
            "date": list(values),
            "category": "marketplace",
            "median_share": list(values.values()),
            "resident_share": list(values.values()),
        }
    )


# --- Корзина: CLR и PCA ---------------------------------------------------------------------------


def test_clr_rows_sum_to_zero_and_components_live_in_clr_space(data):
    sample = s4.basket_sample(data.mo)
    clr = s4.part_frame(sample, "clr", 2024)
    assert np.allclose(clr.sum(axis=1), 0.0, atol=1e-12)
    res = s4.pca_clr(clr, s4.region_controls(sample, 2024))
    resid, _, _ = s4.residualize(clr, s4.region_controls(sample, 2024))
    assert np.allclose(resid.sum(axis=1), 0.0, atol=1e-10)  # очистка не выводит из симплекса
    assert np.allclose(res.loadings.sum(axis=0), 0.0, atol=1e-10)


def test_pca_finds_single_varying_part_and_orients_largest_loading_positive():
    rng = np.random.default_rng(1)
    n = 400
    shares = pd.DataFrame({p: np.full(n, 0.1) for p in PARTS})
    shares["cafe"] = 0.02 * np.exp(rng.normal(0, 0.5, n))
    shares["food"] = 0.45 * np.exp(rng.normal(0, 0.01, n))
    clr = stats.clr(stats.closure(shares))
    res = s4.pca_clr(clr, n_components=3)
    assert res.explained["PC1"] > 0.95
    assert res.loadings["PC1"].abs().idxmax() == "cafe"
    for pc in res.loadings.columns:
        col = res.loadings[pc]
        assert col[col.abs().idxmax()] > 0
    assert np.isclose(res.total_var, clr.var(ddof=1).sum())
    assert res.r2 == 0 and res.residual_share == 1


def test_pca_controls_remove_level_and_region():
    rng = np.random.default_rng(2)
    n = 600
    level = pd.Series(rng.normal(10, 0.3, n))
    region = pd.Series(rng.integers(0, 8, n)).astype("category")
    shift = region.cat.codes.to_numpy() * 0.05
    shares = pd.DataFrame({p: np.full(n, 0.1) for p in PARTS})
    shares["cafe"] = 0.03 * np.exp(2.0 * (level - 10) + shift + rng.normal(0, 0.01, n))
    clr = stats.clr(stats.closure(shares))
    controls = pd.DataFrame({"log_level": level, "region": region})
    clean = s4.pca_clr(clr, controls)
    assert clean.r2 > 0.99 and clean.residual_share < 0.01
    _, r2_level, _ = s4.residualize(clr, controls[["log_level"]])
    _, r2_region, _ = s4.residualize(clr, controls[["region"]])
    assert r2_level > 0.9 > r2_region  # уровень объясняет почти всё, регион — меньше
    assert np.isclose(clean.explained.sum(), 1.0) or clean.explained.sum() < 1.0


def test_design_matrix_turns_region_codes_into_dummies():
    controls = pd.DataFrame({"log_level": [1.0, 2.0, 3.0, 4.0], "region": pd.Categorical([5, 5, 7, 9])})
    x = s4.design_matrix(controls)
    assert x.shape == (4, 1 + 1 + 2)  # свободный член, уровень, дамми без первой из трёх
    assert np.allclose(x[:, 0], 1.0)


def test_residualize_matches_simple_regression():
    rng = np.random.default_rng(3)
    x = rng.normal(0, 1, 200)
    y = pd.DataFrame({"a": 2 * x + rng.normal(0, 1, 200), "b": rng.normal(0, 1, 200)})
    resid, total, by_col = s4.residualize(y, pd.DataFrame({"x": x}))
    slope, intercept = np.polyfit(x, y["a"], 1)
    assert np.allclose(resid["a"], y["a"] - (intercept + slope * x))
    expected_a = 1 - (resid["a"] ** 2).sum() / ((y["a"] - y["a"].mean()) ** 2).sum()
    assert np.isclose(by_col["a"], expected_a)
    ss_res = (resid**2).sum().sum()
    ss_tot = ((y - y.mean()) ** 2).sum().sum()
    assert np.isclose(total, 1 - ss_res / ss_tot)  # доля суммарной дисперсии всех колонок
    assert by_col["a"] > total > by_col["b"]


def basket_mo(n: int, rng: np.random.Generator, *, stable: bool) -> pd.DataFrame:
    """``mo`` для корзины: регион и уровень задают часть долей, а свой фактор МО (доля здоровья) — остаток;
    при ``stable`` фактор общий для 2023 и 2024 годов, иначе в 2024 году он новый."""
    region = rng.integers(0, 6, n)
    level = rng.normal(10, 0.3, n)
    own23 = rng.normal(0, 0.3, n)
    own24 = own23 + rng.normal(0, 0.05, n) if stable else rng.normal(0, 0.3, n)
    mo = pd.DataFrame({"territory_id": np.arange(1, n + 1), "region_code": region})
    for year, own in ((2023, own23), (2024, own24)):
        shares = pd.DataFrame({p: np.full(n, 0.1) for p in PARTS})
        shares["cafe"] = 0.03 * np.exp(1.5 * (level - 10) + 0.1 * region)
        shares["health"] = 0.05 * np.exp(own)
        shares = stats.closure(shares)
        clr = stats.clr(shares)
        for p in PARTS:
            mo[f"sh_{p}_{year}"] = shares[p]
            mo[f"clr_{p}_{year}"] = clr[p]
        mo[f"log_level_{year}"] = level
        mo[f"level_{year}"] = np.exp(level)
    return mo


def test_residual_axis_is_stable_only_when_structure_repeats():
    rng = np.random.default_rng(4)
    stable = s4.basket_analysis(basket_mo(500, rng, stable=True))
    assert stable.stability > 0.9
    assert stable.clean.loadings["PC1"].abs().idxmax() == "health"
    assert stable.axis_cos > 0.9
    assert stable.r2_level > 0.1 and stable.n == 500
    moving = s4.basket_analysis(basket_mo(500, rng, stable=False))
    assert abs(moving.stability) < 0.2


def test_basket_table_has_parts_and_total_row(data):
    res = s4.basket_analysis(data.mo)
    table = s4.basket_table(res)
    assert list(table["code"]) == [*res.summary.sort_values("clr_var", ascending=False).index, "total"]
    total = table.iloc[-1]
    assert np.isclose(total["raw_pc1"], res.raw.explained["PC1"])
    assert np.isclose(total["r2_both"], 1 - res.clean.residual_share)
    assert np.isclose(total["clr_var"], table["clr_var"].iloc[:-1].sum())


def test_basket_summary_sorted_by_median_share(data):
    sample = s4.basket_sample(data.mo)
    summary = s4.basket_summary(s4.part_frame(sample, "sh", 2024), s4.part_frame(sample, "clr", 2024))
    assert summary["median"].is_monotonic_decreasing
    assert set(summary.index) == set(PARTS)
    assert (summary["q25"] <= summary["median"]).all() and (summary["median"] <= summary["q75"]).all()
    assert np.allclose(summary["ratio"], summary["q75"] / summary["q25"]) and (summary["ratio"] >= 1).all()


# --- Маркетплейсы ---------------------------------------------------------------------------------


def test_marketplace_gain_falls_with_level_on_planted_signal(data):
    checks = s4.marketplace_checks(data.panel_wide, data.mo, data.national)
    assert checks["rho_mp_pp_level"] < -0.5
    assert checks["rho_mp_pp_level_within"] < 0
    assert checks["n"] == len(s4.basket_sample(data.mo))
    assert checks["mp_share_dec24"] > checks["mp_share_jan23"]
    assert 0 <= checks["eta2_mp_pp"] <= 1


def test_split_months_protect_from_regression_to_the_mean():
    """Чистый шум: истинный прирост у всех МО одинаков. Без разделения месяцев шум начальной доли даёт
    ложную отрицательную связь («догоняние»), на раздельных месяцах связи нет."""
    rng = np.random.default_rng(5)
    n = 1500
    true_share = rng.uniform(0.10, 0.14, n)[:, None]
    gain = np.where(np.arange(24) >= 12, 0.04, 0.0)[None, :]
    share = np.clip(true_share + gain + rng.normal(0, 0.03, (n, 24)), 0.01, None)
    wide = month_rows(n, share)
    mo = mo_from_wide(wide, rng)
    frame = s4.marketplace_frame(wide, mo)
    naive, _ = stats.spearman(frame["mp_pp_change"], frame["sh_marketplace_2023"])
    split, _ = stats.spearman(frame["gain_odd"], frame["init_odd"])
    split_rev, _ = stats.spearman(frame["gain_even"], frame["init_even"])
    assert naive < -0.3
    assert abs(split) < 0.1 and abs(split_rev) < 0.1


def test_month_share_needs_all_months():
    share = np.full((2, 24), 0.1)
    wide = month_rows(2, share)
    wide = wide.drop(wide.index[(wide["territory_id"] == 2) & (wide["date"] == "2023-03")])
    got = s4.month_share(wide, "marketplace", 2023, s4.ODD_MONTHS)
    assert np.isclose(got[1], 0.1) and np.isnan(got[2])


def test_step_test_catches_planted_jump():
    base = {f"{y}-{m:02d}": 0.10 + 0.04 * (y - 2023) + 0.002 * m for y in (2023, 2024) for m in range(1, 13)}
    smooth, _ = s4.step_test(national_mp(base))
    assert smooth < 1e-9
    jumped = {d: v + (0.03 if d >= "2024-07" else 0.0) for d, v in base.items()}
    size, month = s4.step_test(national_mp(jumped))
    assert np.isclose(size, 3.0)
    assert month == "июнь → июль"
    assert np.isnan(s4.step_test(national_mp({"2023-01": 0.1}))[0])


def test_quintile_gap_groups_and_planted_gradient(data):
    gap = s4.quintile_gap(data.mo, 5)
    assert list(gap["group"]) == [1, 2, 3, 4, 5]
    assert gap["n"].sum() == len(data.mo.dropna(subset=["mp_pp_change", "log_level_2023"]))
    assert (gap["level_max"].iloc[:-1].to_numpy() <= gap["level_min"].iloc[1:].to_numpy()).all()
    assert np.allclose(gap["dpp_exact"], 100 * (gap["share_2024"] - gap["share_2023"]))
    assert (gap["dpp"] - gap["dpp_exact"]).abs().max() <= 0.1 + 1e-9  # из округлённых концов — не дальше 0,1
    assert gap["dpp"].iloc[0] - gap["dpp"].iloc[-1] > 1  # бедные МО прибавляют больше (заложено в синтетике)
    assert "n_inner" in gap and "dpp_mo" in gap


def test_quintile_gap_increment_matches_rounded_ends_and_counts_inner_city():
    """Прирост «гантели» считается из концов, округлённых до десятых процента, как они подписаны рядом
    (13,1% − 10,5% = +2,6, а не +2,5 из точных медиан); медиана приростов МО — отдельная колонка."""
    mo = pd.DataFrame(
        {
            "territory_id": [1, 2, 3, 4],
            "log_level_2023": [9.0, 9.1, 10.0, 10.1],
            "level_2023": np.exp([9.0, 9.1, 10.0, 10.1]),
            "sh_marketplace_2023": [0.11, 0.12, 0.10505, 0.10505],
            "sh_marketplace_2024": [0.17, 0.17, 0.13054, 0.13054],
            "is_inner_city": [False, False, True, False],
        }
    )
    gap = s4.quintile_gap(mo, 2)
    assert list(gap["n_inner"]) == [0, 1]
    top = gap.iloc[-1]
    assert np.isclose(top["dpp_exact"], 2.549) and np.isclose(top["dpp"], 2.6)
    assert style.fmt_pct(top["share_2024"]) == "13,1%" and style.fmt_pct(top["share_2023"]) == "10,5%"
    assert np.isclose(gap.iloc[0]["dpp_mo"], 100 * np.median([0.06, 0.05]))
    assert list(s4.quintile_gap(mo.drop(columns="is_inner_city"), 2)["n_inner"]) == [0, 0]


# --- Рост -------------------------------------------------------------------------------------------


def test_half_year_growth_exact_values():
    rows = []
    for t in range(24):
        year, month = 2023 + t // 12, t % 12 + 1
        v_all = 100.0 if year == 2023 else (110.0 if month <= 6 else 120.0)
        mp = 0.10 if year == 2023 else (0.12 if month <= 6 else 0.15)
        rows.append(
            {"territory_id": 1, "year": year, "month": month, "v_all": v_all, "v_marketplace": v_all * mp}
        )
    wide = pd.DataFrame(rows)
    for c in CATEGORY_CODES:
        if f"v_{c}" not in wide:
            wide[f"v_{c}"] = wide["v_all"] * 0.1
    missing = wide.assign(territory_id=2).drop(index=20)  # у МО 2 нет сентября 2024
    got = s4.half_year_growth(pd.concat([wide, missing], ignore_index=True))
    assert np.isclose(got.loc[1, "growth_h1_all"], 0.10) and np.isclose(got.loc[1, "growth_h2_all"], 0.20)
    assert np.isclose(got.loc[1, "mp_pp_h1"], 2.0) and np.isclose(got.loc[1, "mp_pp_h2"], 5.0)
    assert np.isclose(got.loc[2, "growth_h1_all"], 0.10) and np.isnan(got.loc[2, "growth_h2_all"])


def test_growth_table_and_divergence_on_synth(data):
    table = s4.growth_table(data.panel_wide, data.mo)
    assert list(table["code"]) == list(CATEGORY_CODES)
    full = data.mo.loc[data.mo["series_status"].astype(str) == "full", "growth"]
    assert np.isclose(table.set_index("code").loc["all", "growth_median"], full.median())
    assert (table["growth_p10"] <= table["growth_median"]).all()
    assert table["rub_incr_rho"].between(-1, 1).all()
    rel = s4.growth_vs_level(data.panel_wide, data.mo)
    assert rel["rho_growth_level"] > 0.2  # в синтетике рост выше у богатых МО
    assert rel["n"] == full.notna().sum()
    for key in ("rho_growth_level_split", "rho_growth_level_split_rev", "rho_growth_level_no_inner"):
        assert np.isfinite(rel[key]), key
    no_inner = s4.growth_vs_level(data.panel_wide, data.mo.drop(columns="is_inner_city"))
    assert np.isnan(no_inner["rho_growth_level_no_inner"])


def test_split_growth_level_swaps_months():
    """Раздельные месяцы в обе стороны: уровень по нечётным и рост по чётным — и наоборот."""
    rng = np.random.default_rng(11)
    n = 300
    level = rng.normal(10, 0.3, n)
    t = np.arange(24)
    growth = 0.1 + 0.2 * (level - 10)  # рост выше у богатых — в обоих наборах месяцев
    base = np.exp(level)[:, None] * np.exp(rng.normal(0, 0.02, (n, 24)))
    v_all = base * np.where(t >= 12, 1 + growth[:, None], 1.0)
    wide = month_rows(n, np.full((n, 24), 0.1))
    wide["v_all"] = v_all.ravel()
    ids = pd.Index(np.arange(1, n + 1))
    split = s4.split_growth_level(wide, ids, s4.ODD_MONTHS, s4.EVEN_MONTHS)
    rev = s4.split_growth_level(wide, ids, s4.EVEN_MONTHS, s4.ODD_MONTHS)
    assert split > 0.8 and rev > 0.8


# --- ИПЦ ------------------------------------------------------------------------------------------


def months() -> list[str]:
    return [f"{y}-{m:02d}" for y in (2023, 2024) for m in range(1, 13)]


def test_cpi_index_level_mom_and_errors():
    level = {d: 100.0 + i for i, d in enumerate(months())}
    got = s4.cpi_index({"source": "https://rosstat.gov.ru", "values": level})
    assert np.isclose(got["2024-12"], 123.0)
    mom = s4.cpi_index({"source": "x", "kind": "mom", "values": dict.fromkeys(months(), 101.0)})
    assert np.isclose(mom["2023-01"], 101.0) and np.isclose(mom["2023-02"], 102.01)
    with pytest.raises(ValueError, match="source"):
        s4.cpi_index({"values": level})
    with pytest.raises(ValueError, match="2024-12"):
        s4.cpi_index({"source": "x", "values": {d: v for d, v in level.items() if d != "2024-12"}})
    with pytest.raises(ValueError, match="kind"):
        s4.cpi_index({"source": "x", "kind": "yoy", "values": level})


def test_cpi_ratios_and_real_growth():
    index = pd.Series({d: (100.0 if d < "2024" else 108.0) for d in months()})
    annual, monthly = s4.cpi_ratios(index)
    assert np.isclose(annual, 1.08) and np.allclose(monthly, 1.08)
    assert np.isclose(s4.real_growth(0.15, 1.08), 1.15 / 1.08 - 1)


# --- Параметры, текст, подписи ----------------------------------------------------------------------


def test_basket_params_merge_and_reject_typos(data, tmp_path):
    ctx = make_section_context(tmp_path, "e4", data=data)
    ctx.cfg.data["eda"]["basket"] = {"label_n": 5, "headline": {"pc1_rho_min": 0.7}}
    p = s4.basket_params(ctx.cfg)
    assert p["label_n"] == 5 and p["headline"]["pc1_rho_min"] == 0.7
    assert p["headline"]["growth_rho_min"] == s4.BASKET_DEFAULTS["headline"]["growth_rho_min"]
    assert s4.BASKET_DEFAULTS["label_n"] != 5  # умолчания не испорчены
    ctx.cfg.data["eda"]["basket"] = {"lable_n": 5}
    with pytest.raises(ValueError, match="lable_n"):
        s4.basket_params(ctx.cfg)
    ctx.cfg.data["eda"]["basket"] = {"headline": {"pc1_rho": 0.7}}
    with pytest.raises(ValueError, match="pc1_rho"):
        s4.basket_params(ctx.cfg)
    ctx.cfg.data["eda"]["basket"] = {"level_groups": 7}  # текст не умеет назвать «седьмые части»
    with pytest.raises(ValueError, match="level_groups"):
        s4.basket_params(ctx.cfg)


def test_group_count_changes_the_words_not_only_the_numbers(tmp_path, data):
    result, _ = run(tmp_path, data, basket={"level_groups": 4, "trend_bins": 5})
    f12 = next(f for f in result.figures if f.fid == "F12")
    assert f12.subtitle.startswith("Четверти МО")
    assert f"у{style.NBSP}четверти МО с{style.NBSP}самыми низкими" in result.summary_md
    assert "пятой части" not in result.summary_md
    assert "нижняя четверть" in result.facts["e4.q1_share_2023"].note
    assert result.headline_errors == []


def test_nbsp_binds_single_letter_words_and_dash():
    nb = style.NBSP
    assert s4.nbsp("и в МО с тратами — да") == f"и{nb}в{nb}МО с{nb}тратами{nb}— да"
    assert s4.nbsp("С4 (корзина)") == "С4 (корзина)"


def test_nbsp_binds_percentage_points():
    nb = style.NBSP
    assert s4.nbsp("размах 4,1 → 4,8 п. п.") == f"размах 4,1 → 4,8{nb}п.{nb}п."
    assert s4.nbsp("В п. п. догоняния нет") == f"В{nb}п.{nb}п. догоняния нет"
    assert s4.nbsp("{{e4.mp_iqr_2024}} п. п.)") == f"{{{{e4.mp_iqr_2024}}}}{nb}п.{nb}п.)"


def test_region_short_names():
    assert s4.region_short("Вологодская область") == "Вологодская обл."
    assert s4.region_short("Республика Северная Осетия — Алания") == "Северная Осетия"
    assert s4.region_short("Ханты-Мансийский автономный округ — Югра") == "Ханты-Мансийский АО"
    assert s4.region_short("Москва") == "Москва"
    # республики — устойчивым коротким названием, а не обрубком прилагательного («Чувашская»)
    assert s4.region_short("Чувашская Республика") == "Чувашия"
    assert s4.region_short("Удмуртская Республика") == "Удмуртия"
    assert s4.region_short("Чеченская Республика") == "Чечня"
    assert s4.region_short("Кабардино-Балкарская Республика") == "Кабардино-Балкария"
    assert s4.region_short("Карачаево-Черкесская Республика") == "Карачаево-Черкесия"
    assert s4.region_short("Республика Саха (Якутия)") == "Якутия"
    assert s4.region_short("Республика Тыва") == "Тыва"


def test_part_word_forms_cover_all_parts():
    assert set(s4.PART_FORMS) == set(PARTS)
    assert s4.part_word("cafe", s4.GEN) == "общепита" and s4.part_word("food", s4.DAT) == "продуктам"
    assert s4.part_word("cafe", s4.PREP) == "общепите"


def test_paragraphs_keep_list_items_and_slot_lines():
    template = "**Что видно.** Корзина:\n\n- первый\nпункт\n$robust\n- второй\n\nАбзац\nв две строки"
    assert s4._paragraphs(template) == (
        "**Что видно.** Корзина:\n\n- первый пункт\n$robust\n- второй\n\nАбзац в две строки"
    )


def test_point_names_add_region():
    ter = pd.DataFrame(
        {
            "territory_id": [1, 2],
            "name_short": ["Октябрьский"] * 2,
            "region_name": ["Пермский край", "Республика Коми"],
        }
    )
    assert list(s4.point_names(ter)) == ["Октябрьский (Пермский край)", "Октябрьский (Коми)"]


def test_spread_extremes_skips_points_under_a_neighbour_label():
    x = pd.Series([0.0, 0.01, 0.5, 1.0, 0.99, 0.2])
    y = pd.Series([0.0, 0.01, 0.5, 1.0, 0.99, 0.8])
    score = pd.Series([-10.0, -9.0, 0.0, 10.0, 9.0, 5.0])
    got = s4.spread_extremes(x, y, score, 4, box=(0.1, 0.1))
    assert got[:1] == [0] and 1 not in got  # точка 1 лежит под подписью точки 0
    assert 3 in got and 4 not in got and len(got) == 4


def test_summary_wording_follows_the_data(data, tmp_path):
    _, ctx = run(tmp_path, data)
    p = s4.basket_params(ctx.cfg) | {"step_test_pp": 2.0}
    rej = ctx.cfg["eda"]["rejection"]
    values = {k.split(".", 1)[1]: f.value for k, f in ctx.facts.items()}
    ok = {
        "mp_split_half": 0.7,
        "rho_mp_pp_level_within": -0.6,
        "mp_step_max_pp": 1.0,
        "rho_mp_pp_level": -0.7,
    }
    # В п. п. связи со стартовой долей нет, в логарифмах — догоняние (Б.1, п. 10: обе шкалы сразу)
    scales = values | ok
    scales |= {"rho_mp_pp_initial": 0.01, "rho_mp_pp_initial_rev": 0.05}
    scales |= {"rho_mp_log_initial": -0.5, "rho_mp_log_initial_rev": -0.5, "rho_mp_rub_incr_level": 0.6}
    slots = s4._summary_slots(scales, p, rej, cpi_used=False)
    assert slots["catch_up"].startswith("В п. п. догоняния нет")
    assert "Поэтому в относительном выражении МО с малой долей растут быстрее" in slots["catch_up"]
    assert "в п. п. догоняния по доле нет, в относительном выражении есть" in slots["s5"]
    assert "в рублях разрыв растёт" in slots["s5"]
    # догоняния нет ни в одной шкале
    flat = scales | {"rho_mp_log_initial": 0.02, "rho_mp_log_initial_rev": -0.03}
    assert "ни в п. п., ни в относительном выражении" in s4._summary_slots(flat, p, rej, False)["s5"]
    # два варианта раздельных месяцев не согласны — вывод не делается
    mixed = scales | {"rho_mp_pp_initial_rev": -0.4}
    assert "зависит от выбора месяцев" in s4._summary_slots(mixed, p, rej, False)["catch_up"]
    strong = scales | {
        "rho_mp_pp_initial": -0.5,
        "rho_mp_pp_initial_rev": -0.5,
        "rho_mp_rub_incr_level": -0.3,
    }
    slots = s4._summary_slots(strong, p, rej, cpi_used=False)
    assert "есть догоняние" in slots["catch_up"] and "низкими тратами" in slots["rub"]
    assert "догоняния по доле нет" not in slots["s5"]
    unstable = values | {"residual_pc1_stability": 0.1}
    assert "неустойчива" in s4._summary_slots(unstable, p, rej, cpi_used=False)["s4"]


def test_growth_wording_follows_sign_and_spread(data, tmp_path):
    _, ctx = run(tmp_path, data)
    p = s4.basket_params(ctx.cfg) | {"step_test_pp": 2.0}
    rej = ctx.cfg["eda"]["rejection"]
    values = {k.split(".", 1)[1]: f.value for k, f in ctx.facts.items()}
    diverge = values | {"rho_growth_level": 0.4, "sd_loglevel_2023": 0.30, "sd_loglevel_2024": 0.32}
    assert s4._summary_slots(diverge, p, rej, cpi_used=False)["growth_rel"].startswith(
        "Разрыв в уровнях трат растёт медленно"  # SD выросло на 6,7% — меньше порога slow_spread
    )
    fast = values | {"rho_growth_level": 0.4, "sd_loglevel_2023": 0.30, "sd_loglevel_2024": 0.36}
    assert s4._summary_slots(fast, p, rej, cpi_used=False)["growth_rel"].startswith(
        "Разрыв в уровнях трат растёт:"
    )
    converge = values | {"rho_growth_level": -0.4, "sd_loglevel_2023": 0.32, "sd_loglevel_2024": 0.30}
    text = s4._summary_slots(converge, p, rej, cpi_used=False)["growth_rel"]
    assert text.startswith("Уровни трат сближаются") and "тратили меньше" in text
    mixed = values | {"rho_growth_level": 0.4, "sd_loglevel_2023": 0.32, "sd_loglevel_2024": 0.30}
    text = s4._summary_slots(mixed, p, rej, cpi_used=False)["growth_rel"]
    assert "Разрыв" not in text and "сближаются" not in text and text.startswith("Номинальный рост выше")
    assert "на раздельных месяцах {{e4.rho_growth_level_split}} и {{e4.rho_growth_level_split_rev}}" in text


def test_rub_wording_says_gap_grows_only_when_scales_disagree(data, tmp_path):
    """Формулировка С5 «в рублях разрыв растёт» — только если доля растёт быстрее у бедных МО, а прибавка
    в рублях больше у богатых; причина называется — база выше, как во всех категориях."""
    _, ctx = run(tmp_path, data)
    p = s4.basket_params(ctx.cfg) | {"step_test_pp": 2.0}
    rej = ctx.cfg["eda"]["rejection"]
    values = {k.split(".", 1)[1]: f.value for k, f in ctx.facts.items()}
    base = values | {"mp_split_half": 0.7, "rho_mp_pp_level_within": -0.6, "mp_step_max_pp": 1.0}
    base |= {"rho_mp_rub_growth_level": -0.3, "rub_incr_rho_min": 0.6}
    disagree = s4._summary_slots(
        base | {"rho_mp_pp_level": -0.7, "rho_mp_rub_incr_level": 0.6}, p, rej, False
    )
    assert "разрыв в рублях всё же растёт" in disagree["rub"] and "потому что база выше" in disagree["rub"]
    assert "Так во всех категориях" in disagree["rub"] and "(номинал)" in disagree["rub"]
    assert "в рублях разрыв растёт" in disagree["s5"]
    agree = s4._summary_slots(base | {"rho_mp_pp_level": -0.7, "rho_mp_rub_incr_level": -0.4}, p, rej, False)
    assert "прибавка в рублях больше у МО с низкими тратами" in agree["rub"]
    assert "разрыв растёт" not in agree["s5"]


def test_summary_reads_cleanly_after_filling(finding):
    """После подстановки фактов в тексте нет двойных знаков («на +54%»), двойных точек после «п. п.»,
    незаполненных слотов и неразрывных пробелов, потерянных у однобуквенных слов."""
    result, _ = finding
    text = PLACEHOLDER.sub(lambda m: result.facts[f"{m.group(1)}.{m.group(2)}"].text, result.summary_md)
    for bad in ("на +", "на −", "п..", "..", "$", "{{", " ,", "  "):
        assert bad not in text, bad
    assert " в " not in text and " с " not in text  # после однобуквенных — неразрывный пробел
    for caveat in result.caveats:
        PLACEHOLDER.sub(lambda m: result.facts[f"{m.group(1)}.{m.group(2)}"].text, caveat)  # ключи известны


# --- Сверка с ориентирами Б.4 -----------------------------------------------------------------------


def test_two_year_composition_pools_both_years_and_drops_incomplete():
    share = np.full((3, 24), 0.10)
    share[:, 12:] = 0.20  # во втором году доля вдвое выше
    wide = month_rows(3, share)
    wide.loc[wide["territory_id"] == 2, "v_all"] = 40_000.0  # у МО 2 траты вдвое выше
    wide["v_marketplace"] = np.round(wide["v_all"] * share.ravel())
    wide = wide.drop(wide.index[(wide["territory_id"] == 3) & (wide["date"] == "2024-05")])
    shares, level = s4.two_year_composition(wide, [1, 2, 3])
    assert list(shares.index) == [1, 2]  # у МО 3 нет мая 2024 года
    assert np.allclose(shares["marketplace"], 0.15)  # (0,10 + 0,20) / 2 при равных тратах по месяцам
    assert np.isclose(level[2] - level[1], np.log(2.0))


def test_spearman_demeaned_sees_within_signal_not_group_shift():
    rng = np.random.default_rng(6)
    n = 600
    groups = pd.Series(rng.integers(0, 10, n))
    shift = groups * 5.0  # чистый эффект группы в обеих величинах: в целом ρ высокий
    noise_x, noise_y = rng.normal(0, 1, n), rng.normal(0, 1, n)
    x, y = pd.Series(shift + noise_x), pd.Series(shift + noise_y)
    assert stats.spearman(x, y)[0] > 0.9
    assert abs(s4.spearman_demeaned(x, y, groups)) < 0.1
    related = pd.Series(shift + noise_x + 0.5 * rng.normal(0, 1, n))
    assert s4.spearman_demeaned(x, related, groups) > 0.8


def test_pooled_share_ignores_population():
    wide = pd.DataFrame(
        {
            "date": ["2023-01", "2023-01", "2023-02"],
            "v_all": [10_000.0, 30_000.0, 20_000.0],
            "v_marketplace": [1_000.0, 6_000.0, 5_000.0],
        }
    )
    assert np.isclose(s4.pooled_share(wide, "marketplace", "2023-01"), 7_000 / 40_000)
    assert np.isnan(s4.pooled_share(wide, "marketplace", "2024-12"))


def test_reference_note_formats_value_by_kind():
    ref = {"pc1_var": 0.7775, "rho_mp_pp_level_within": -0.596}
    note = s4.reference_note("pc1_var", "pct", ref)
    assert note.startswith("Ориентир Б.4 посчитан по долям за 2023 и 2024") and note.endswith("77,8%")
    assert s4.reference_note("rho_mp_pp_level_within", "rho", ref).endswith("−0,60")
    assert s4.reference_note("growth_median", "pct", ref) == ""  # считается так же, как ориентир
    assert s4.reference_note("pc1_var", "pct", {}) == ""


def test_reference_variants_explain_benchmarks_in_notes(finding, data):
    result, _ = finding
    ref = s4.reference_variants(data.panel_wide, data.mo)
    assert set(ref) == set(s4.REF_DEFINITIONS)
    for key in s4.REF_DEFINITIONS:
        fact = result.facts[f"e4.{key}"]
        assert "Ориентир Б.4" in fact.note
        assert fact.note.endswith(s4.reference_note(key, fact.kind, ref))
    assert "Ориентир Б.4" not in result.facts["e4.growth_median"].note
    shares = [ref[f"share_median_{p}"] for p in PARTS]
    assert all(0 < v < 1 for v in shares)


def test_rounding_note_shows_exact_and_rounded_difference():
    note = s4.rounding_note({"share_2024": 0.13054, "share_2023": 0.10505})
    assert "13,05% − 10,51% = +2,55 п. п." in note and note.endswith("+2,6 п. п.")


# --- Маркетплейсы: две шкалы ------------------------------------------------------------------------


def test_share_gain_falls_but_ruble_increment_rises_with_level():
    """Сигнал сюжета С5 в двух шкалах: у богатых МО доля растёт медленнее (п. п.), а прибавка в рублях больше
    (траты у них выше) — «в рублях разрыв растёт»."""
    rng = np.random.default_rng(7)
    n = 400
    z = rng.normal(0, 1, n)
    v_all = 20_000.0 * np.exp(0.5 * z)
    gain = 0.05 - 0.01 * z + rng.normal(0, 0.002, n)  # прирост доли меньше у богатых
    share = np.where(np.arange(24) >= 12, 0.12 + gain[:, None], 0.12)
    wide = month_rows(n, share)
    wide["v_all"] = np.repeat(v_all, 24)
    wide["v_marketplace"] = wide["v_all"] * share.ravel()
    mo = mo_from_wide(wide, rng)
    mo["log_level_2023"] = np.log(v_all)
    nat = national_mp({f"{y}-{m:02d}": 0.12 + 0.05 * (y - 2023) for y in (2023, 2024) for m in range(1, 13)})
    checks = s4.marketplace_checks(wide, mo, nat)
    assert checks["rho_mp_pp_level"] < -0.8
    assert checks["rho_mp_rub_incr_level"] > 0.5
    assert checks["rho_mp_rub_growth_level"] < 0  # в процентах быстрее растут бедные
    assert checks["mp_share_dec23"] == pytest.approx(0.12)


def test_constant_pp_gain_is_catch_up_in_logs_on_split_months():
    """Одинаковая прибавка в п. п. при разной стартовой доле: в п. п. догоняния нет (ρ ≈ 0), а в логарифмах
    МО с малой долей растут быстрее (ρ < 0) — и это видно на раздельных месяцах, без регрессии к среднему."""
    rng = np.random.default_rng(12)
    n = 1500
    start = rng.uniform(0.06, 0.18, n)[:, None]
    gain = np.where(np.arange(24) >= 12, 0.04, 0.0)[None, :]
    share = np.clip(start + gain + rng.normal(0, 0.003, (n, 24)), 0.01, None)
    wide = month_rows(n, share)
    mo = mo_from_wide(wide, rng)
    nat = national_mp({f"{y}-{m:02d}": 0.12 + 0.04 * (y - 2023) for y in (2023, 2024) for m in range(1, 13)})
    checks = s4.marketplace_checks(wide, mo, nat)
    assert abs(checks["rho_mp_pp_initial"]) < 0.1 and abs(checks["rho_mp_pp_initial_rev"]) < 0.1
    assert checks["rho_mp_log_initial"] < -0.8 and checks["rho_mp_log_initial_rev"] < -0.8
    assert checks["rho_mp_log_initial_naive"] < -0.8


def test_fixed_denominator_removes_the_denominator_effect():
    """Маркетплейсы растут у всех МО одинаково (+50%), а все траты — быстрее у богатых: прирост доли падает
    с уровнем только из-за знаменателя; при медианном росте знаменателя связи нет."""
    rng = np.random.default_rng(13)
    n = 600
    z = rng.normal(0, 1, n)
    v23 = 20_000.0 * np.exp(0.4 * z)
    growth_all = 0.15 + 0.05 * z + rng.normal(0, 0.005, n)
    t = np.arange(24)
    v_all = np.where(t >= 12, v23[:, None] * (1 + growth_all[:, None]), v23[:, None])
    mp23 = 0.12 * v23 * np.exp(rng.normal(0, 0.05, n))
    v_mp = np.where(t >= 12, mp23[:, None] * 1.5, mp23[:, None])
    wide = month_rows(n, np.full((n, 24), 0.1))
    wide["v_all"], wide["v_marketplace"] = v_all.ravel(), v_mp.ravel()
    mo = mo_from_wide(wide, rng)
    mo["log_level_2023"] = np.log(v23)
    frame = s4.marketplace_frame(wide, mo)
    assert stats.spearman(frame["mp_pp_change"], frame["log_level_2023"])[0] < -0.5
    assert abs(stats.spearman(s4.fixed_denominator_gain(frame), frame["log_level_2023"])[0]) < 0.15
    assert np.isnan(s4.partial_level_within(frame, "mp_pp_change"))  # нет плотности и населения
    frame["log_density_2023"], frame["pop_2023"] = rng.normal(3, 1, n), np.exp(rng.normal(9, 1, n))
    assert s4.partial_level_within(frame, "mp_pp_change") < -0.5
    # траты на маркетплейсах пропорциональны уровню: у верхней пятой части база в exp(0,4 · 2,8) ≈ 3 раза выше
    assert 2.0 < s4.base_ratio(frame, "rub_2023", 5) < 4.5


def test_robust_items_name_trivial_explanations(data, tmp_path):
    """Пункты о знаменателе, районах Москвы и Петербурга и размере МО появляются по данным и называют
    риски."""
    _, ctx = run(tmp_path, data)
    p = s4.basket_params(ctx.cfg) | {"step_test_pp": 2.0}
    values = {k.split(".", 1)[1]: f.value for k, f in ctx.facts.items()}
    v = values | {
        "rho_mp_pp_level_within": -0.6,
        "rho_mp_pp_level_within_fixed": -0.4,
        "rho_growth_level_within": 0.4,
        "q5_n_inner": 237,
        "q5_n": 403,
        "dq1_pp": 5.5,
        "dq5_pp": 2.6,
        "dq1_pp_no_inner": 5.5,
        "dq5_pp_no_inner": 3.4,
        "rho_mp_pp_level_partial": -0.4,
    }
    items, risks = s4._robust_items(v, p)
    assert len(items) == 3 and risks == ["знаменатель", "районы Москвы и Петербурга в верхней группе"]
    assert "разрыв меньше, но остаётся" in items[1] and "остаётся: частный" in items[2]
    none = v | {
        "q5_n_inner": 0,
        "rho_mp_pp_level_partial": float("nan"),
        "rho_mp_pp_level_within_fixed": -0.7,
    }
    items, risks = s4._robust_items(none, p)
    assert len(items) == 1 and items[0].startswith("Знаменатель связь не объясняет") and risks == []


def test_start_phrase_uses_close_only_when_start_gap_is_small():
    gap = pd.DataFrame({"share_2023": [0.117, 0.105], "dpp": [5.5, 2.5]})
    assert s4.start_phrase(gap, ("11,7%", "10,5%")) == "старт близок (11,7% и 10,5%)"
    far = pd.DataFrame({"share_2023": [0.20, 0.10], "dpp": [5.5, 2.5]})
    assert s4.start_phrase(far, ("20,0%", "10,0%")) == "старт (20,0% и 10,0%)"


# --- Подписи точек ------------------------------------------------------------------------------------


def test_label_points_clear_avoids_labels_points_and_obstacles():
    with style.use():
        fig, ax = style.new_figure("full")
        x = np.array([2.0, 2.4, 2.8, 5.0, 8.0])
        y = np.array([2.0, 2.3, 1.8, 5.0, 8.0])
        ax.scatter(x, y)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        # препятствие стоит там, где встала бы первая подпись
        obstacle = ax.text(2.1, 2.1, "подпись линии", fontsize=style.POINT_LABEL_PT)
        texts = ["Первое МО (Регион)", "Второе МО (Регион)", "Третье МО (Регион)", "Четвёртое", "Пятое"]
        anns = s4.label_points_clear(ax, x, y, texts, avoid=[obstacle], max_labels=5)
        assert anns[0].xyann != s4._NEAR_OFFSETS[0]  # первое положение занято препятствием
        renderer = s4._renderer(fig)
        boxes = [s4._text_box(a, renderer) for a in anns]
        blocked = obstacle.get_window_extent(renderer=renderer)
        for i, box in enumerate(boxes):
            assert not box.overlaps(blocked)
            assert not any(box.overlaps(other) for j, other in enumerate(boxes) if j != i)
        with pytest.raises(ValueError, match="не больше"):
            s4.label_points_clear(ax, x, y, texts, max_labels=4)


def test_reserve_layout_puts_axes_where_finish_leaves_them():
    title, subtitle = "Заголовок-вывод с числом 12,3%", "Подзаголовок: что показано, n = 2016"
    positions = []
    with style.use():
        for reserve in (True, False):
            fig, ax = style.new_figure("full")
            ax.plot([0, 1], [0, 1])
            n_texts = len(fig.texts)
            if reserve:
                s4.reserve_layout(fig, title, subtitle, style.SOURCE_SBER, 13)
                assert len(fig.texts) == n_texts  # тексты шапки убраны
                fig.draw_without_rendering()
                positions.append(ax.get_position().bounds)
            else:
                style.finish(fig, title, subtitle, style.SOURCE_SBER, 13)
                fig.draw_without_rendering()
                positions.append(ax.get_position().bounds)
    assert np.allclose(positions[0], positions[1])


# --- Раздел целиком -------------------------------------------------------------------------------


def test_run_section_outputs(finding):
    result, out_dir = finding
    assert isinstance(result, Finding) and result.section == "e4" and result.title == s4.TITLE
    assert [f.fid for f in result.figures] == ["F09", "F10", "F11", "F12", "F13"]
    assert [t.tid for t in result.tables] == ["T08", "T09", "T10"]
    for rec in result.figures:
        for rel in (rec.png, rec.svg, rec.data_csv):
            assert (out_dir / rel).is_file()
        assert rec.check and len(rec.alt) >= 40
    for rec in result.tables:
        assert (out_dir / rec.csv).is_file() and rec.markdown.startswith("|")
    assert result.headline_errors == []
    assert result.caveats and all(isinstance(c, str) for c in result.caveats)


def test_run_section_has_all_mandatory_facts(finding):
    result, _ = finding
    missing = [k for k in MANDATORY_FACTS if f"e4.{k}" not in result.facts]
    assert missing == []
    facts = result.facts
    assert facts["e4.clr_var_argmax"].value == "cafe"
    assert facts["e4.n_basket"].value == facts["e4.n_mp"].value
    assert facts["e4.growth_all"].value == pytest.approx(facts["e4.growth_median"].value)
    assert np.isclose(facts["e4.residual_share"].value, 1 - facts["e4.r2_clr_both"].value)
    assert facts["e4.mp_step_month"].kind == "str"
    assert facts["e4.mp_share_dec23"].kind == "pct"  # декабрь к декабрю — для оговорки о сезонности
    assert facts["e4.residual_pc1_top_load"].value > 0  # знак оси: наибольшая нагрузка положительна


def test_run_section_indicator_and_summary(finding):
    result, _ = finding
    ind = result.indicators
    assert list(ind.columns) == ["territory_id", "basket_resid_pc1"]
    assert ind["territory_id"].is_unique
    assert ind["basket_resid_pc1"].notna().sum() == result.facts["e4.n_basket"].value
    assert "basket_resid_pc1" in result.indicator_labels
    assert "$" not in result.summary_md and "{{e4." in result.summary_md
    assert "  " not in result.summary_md and "..." not in result.summary_md
    assert s4.DROP not in result.summary_md and "\n\n\n" not in result.summary_md
    # «одна мысль — один пункт»: выводы для сюжетов — отдельные пункты, у каждого — вердикт
    assert "**Что это значит для сюжета.**\n\n- **С4" in result.summary_md
    assert result.summary_md.count("\n- **С") == 3
    assert "Маркетплейсы:\n\n- " in result.summary_md


def test_run_section_with_cpi_draws_nominal_real(tmp_path, data):
    values = {d: 100.0 * 1.006 ** (i + 1) for i, d in enumerate(months())}
    result, ctx = run(tmp_path, data, cpi={"source": "https://rosstat.gov.ru/test", "values": values})
    assert [f.fid for f in result.figures][-1] == "F17"
    assert result.headline_errors == []
    real = result.facts["e4.real_growth_median"].value
    nominal = result.facts["e4.growth_median"].value
    annual = result.facts["e4.cpi_annual"].value
    assert np.isclose(real, (1 + nominal) / (1 + annual) - 1)
    t10 = pd.read_csv(ctx.out_dir / result.tables[-1].csv)
    assert "growth_real_median" in t10.columns


def test_headline_check_fails_when_claim_is_false(tmp_path, data):
    result, _ = run(tmp_path, data, basket={"headline": {"pc1_rho_min": 1.01, "mp_annual_ratio_min": 50.0}})
    errors = " ".join(result.headline_errors)
    assert "F10" in errors and "F11" in errors
    assert "F09" not in errors and "F12" not in errors
