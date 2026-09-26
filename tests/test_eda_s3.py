"""Раздел E3 «Годовой ритм»: чистые функции модели, ловушки определения сезонности, прогон на синтетике."""

import copy

import matplotlib as mpl
import numpy as np
import pandas as pd
import pytest
from synth import make_config, make_eda_data, make_section_context

from munnet import style
from munnet.config import Config
from munnet.eda import s3_rhythm as e3
from munnet.eda.base import PLACEHOLDER, Finding

N_T = 24
REQUIRED_FACTS = (
    "common_share",
    "repro_median",
    "share_r06",
    "reliable_n",
    "reliable_share",
    "null_reliable_share",
    "north_share_reliable",
    "north_share_all",
    "peak_dec_share_2023",
    "peak_dec_share_2024",
    "nat_jan",
    "nat_dec",
    "mp_dec",
    "cafe_aug",
    "summer_north",
    "summer_south",
    "cafe_summer_north",
    "cafe_summer_south",
    "rho_summer_lat",
    "rho_summer_nights",
    "pair_raw_q50",
    "pair_own_q90",
    "pair_null_q90",
    "noise_ratio_all",
    "noise_ratio_cafe",
)


def _matrix(values: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(values, index=pd.Index(np.arange(len(values)) + 1, name="territory_id"))


def _trend_series(rng, n=300, noise=0.02, slope_sd=0.01):
    """Ряды «уровень + свой линейный тренд + шум» без всякой сезонности."""
    t = np.arange(N_T)
    a = rng.normal(10, 0.3, n)[:, None]
    b = rng.normal(0.01, slope_sd, n)[:, None]
    return _matrix(a + b * t + rng.normal(0, noise, (n, N_T)))


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    return make_eda_data(tmp_path_factory.mktemp("e3data"))


@pytest.fixture(scope="module")
def params(tmp_path_factory):
    return e3.SeasonParams.from_config(make_config(tmp_path_factory.mktemp("e3cfg")))


@pytest.fixture(scope="module")
def section(tmp_path_factory, data):
    """Один прогон раздела на синтетике: (контекст, итог)."""
    ctx = make_section_context(tmp_path_factory.mktemp("e3run"), "e3", data=data)
    with style.use():
        finding = e3.run_section(ctx)
    return ctx, finding


# --- Модель: тренд, общий и свой ритм ------------------------------------------------------------


def test_detrend_removes_pure_linear_trend():
    t = np.arange(N_T)
    a, b = np.array([10.0, 9.5, 11.0]), np.array([0.01, -0.02, 0.0])
    D, slopes = e3.detrend(_matrix(a[:, None] + b[:, None] * t))
    assert np.allclose(D.to_numpy(), 0.0, atol=1e-10)
    assert np.allclose(slopes.to_numpy(), b)
    assert list(D.index) == [1, 2, 3]


def test_detrend_residuals_orthogonal_to_constant_and_time():
    rng = np.random.default_rng(1)
    D, _ = e3.detrend(_matrix(rng.normal(size=(5, N_T))))
    t = np.arange(N_T)
    assert np.allclose(D.to_numpy().sum(axis=1), 0.0, atol=1e-10)
    assert np.allclose(D.to_numpy() @ t, 0.0, atol=1e-8)


def test_detrend_rejects_missing_months():
    Y = np.ones((2, N_T))
    Y[0, 5] = np.nan
    with pytest.raises(ValueError, match="пропуски"):
        e3.detrend(Y)


def test_own_rhythm_subtracts_monthly_median():
    rng = np.random.default_rng(2)
    D = _matrix(rng.normal(size=(7, N_T)))
    own, g = e3.own_rhythm(D)
    assert np.allclose(own.median(axis=0).to_numpy(), 0.0)
    assert np.allclose(g.to_numpy(), D.median(axis=0).to_numpy())
    assert np.allclose((own + g).to_numpy(), D.to_numpy())


def test_detrend_prevents_spurious_repeatability():
    """Ловушка черновиков: без снятия тренда МО рост внутри года даёт ложную повторяемость профиля."""
    Y = _trend_series(np.random.default_rng(3))
    # Как в черновиках: вычесть среднее своего года и общий профиль месяца, тренд МО не снимать.
    blocks = Y.to_numpy().reshape(len(Y), 2, 12)
    year_demeaned = _matrix((blocks - blocks.mean(axis=2, keepdims=True)).reshape(len(Y), N_T))
    naive_r = e3.reproducibility(e3.own_rhythm(year_demeaned)[0])
    model = e3.fit_rhythm(Y)
    assert naive_r.median() > 0.5
    assert abs(model.r.median()) < 0.1


def test_common_share_near_one_when_all_mo_share_rhythm():
    rng = np.random.default_rng(4)
    t = np.arange(N_T)
    season = np.tile(np.array([-0.15, -0.1, 0, 0, 0.02, 0.03, 0.05, 0.04, 0, 0, 0.01, 0.2]), 2)
    n = 50
    Y = rng.normal(10, 0.3, (n, 1)) + rng.normal(0.01, 0.005, (n, 1)) * t + season
    Y = _matrix(Y + rng.normal(0, 0.002, (n, N_T)))
    model = e3.fit_rhythm(Y)
    assert e3.common_share(model.D, model.O) > 0.95
    assert e3.common_share_by_mo(model.D, model.O).median() > 0.95


def test_common_share_is_low_for_pure_noise():
    Y = _matrix(np.random.default_rng(5).normal(size=(200, N_T)))
    model = e3.fit_rhythm(Y)
    assert e3.common_share(model.D, model.O) < 0.2


def test_reproducibility_known_values():
    rng = np.random.default_rng(6)
    year = rng.normal(size=12)
    own = _matrix(np.vstack([np.r_[year, year], np.r_[year, -year], np.r_[np.zeros(12), year]]))
    r = e3.reproducibility(own)
    assert r.iloc[0] == pytest.approx(1.0)
    assert r.iloc[1] == pytest.approx(-1.0)
    assert np.isnan(r.iloc[2])  # год без разброса — корреляция не определена


def test_profile_and_amplitude():
    year1 = np.linspace(0.0, 0.11, 12)
    year2 = np.linspace(0.0, 0.09, 12)
    own = _matrix(np.r_[year1, year2][None, :])
    prof = e3.own_profile(own)
    assert list(prof.columns) == list(range(1, 13))
    assert np.allclose(prof.to_numpy()[0], (year1 + year2) / 2)
    assert e3.amplitude(own).iloc[0] == pytest.approx(0.1)


def test_is_reliable_needs_both_thresholds():
    r = np.array([0.7, 0.7, 0.5, np.nan])
    A = np.array([0.1, 0.03, 0.1, 0.1])
    assert e3.is_reliable(r, A, 0.6, 0.05).tolist() == [True, False, False, False]


def test_null_share_small_for_repeated_profiles_and_deterministic():
    rng = np.random.default_rng(7)
    n = 400
    profiles = rng.normal(0, 0.05, (n, 12))
    own = _matrix(np.hstack([profiles, profiles]) + rng.normal(0, 0.005, (n, N_T)))
    A = e3.amplitude(own)
    observed = e3.is_reliable(e3.reproducibility(own), A, 0.6, 0.05).mean()
    null_a = e3.null_reliable_share(own, A, np.random.default_rng(11), 20, 0.6, 0.05)
    null_b = e3.null_reliable_share(own, A, np.random.default_rng(11), 20, 0.6, 0.05)
    assert observed > 0.9
    assert 0.0 <= null_a < 0.1
    assert null_a == null_b
    # Обёртка — то же, что нулевое распределение r и правило надёжности с наблюдаемым размахом.
    null_r = e3.null_reproducibility(own, np.random.default_rng(11), 20)
    assert null_r.shape == (20, n)
    assert e3.is_reliable(null_r, A.to_numpy()[None, :], 0.6, 0.05).mean() == pytest.approx(null_a)


def test_null_keeps_observed_amplitude_gate():
    """Нуль переставляет только месяцы 2024 года: МО с размахом ниже порога надёжным не станет."""
    rng = np.random.default_rng(8)
    own = _matrix(rng.normal(0, 0.001, (100, N_T)))
    A = e3.amplitude(own)
    assert (A < 0.05).all()
    assert e3.null_reliable_share(own, A, rng, 5, -1.0, 0.05) == 0.0


# --- Пары МО, профиль страны, летний избыток, шум ------------------------------------------------


def test_pair_correlations_all_pairs_when_few_mo():
    rng = np.random.default_rng(9)
    base = rng.normal(size=N_T)
    X = _matrix(np.vstack([base, base * 2 + 1, rng.normal(size=N_T), rng.normal(size=N_T)]))
    pairs = e3.pair_correlations({"raw": X}, rng, n_pairs=1000)
    assert len(pairs) == 6  # все пары 4 МО
    assert (pairs["territory_a"] != pairs["territory_b"]).all()
    first = pairs.loc[(pairs["territory_a"] == 1) & (pairs["territory_b"] == 2), "raw"].iloc[0]
    assert first == pytest.approx(1.0)
    assert {"raw", "raw_null"} <= set(pairs.columns)


def test_pair_correlations_sampled_pairs_and_null_near_zero():
    rng = np.random.default_rng(10)
    common = rng.normal(size=N_T)
    X = _matrix(common + rng.normal(0, 0.3, (300, N_T)))
    pairs = e3.pair_correlations({"raw": X, "noise": _matrix(rng.normal(size=(300, N_T)))}, rng, 5000)
    assert len(pairs) == 5000
    assert (pairs["territory_a"] != pairs["territory_b"]).all()
    summary = e3.pair_summary(pairs, ["raw", "noise"]).set_index("series")
    assert summary.at["raw", "q50"] > 0.8  # общий сигнал: все пары «похожи»
    assert abs(summary.at["raw", "null_q50"]) < 0.05  # перестановка месяцев его разрушает
    assert abs(summary.at["noise", "q50"]) < 0.05
    assert 0.15 < summary.at["noise", "null_q90"] < 0.4


def test_pair_correlations_rejects_misaligned_rows():
    X = _matrix(np.zeros((3, N_T)))
    with pytest.raises(ValueError, match="не совпадают"):
        e3.pair_correlations({"a": X, "b": X.iloc[:2]}, np.random.default_rng(0), 10)


def _panel(values: dict[int, np.ndarray], missing: dict[int, int] | None = None) -> pd.DataFrame:
    """Мини-панель: у каждой категории те же траты, что у v_all; ``missing`` — выброшенный месяц МО."""
    rows = []
    for tid, v in values.items():
        for t in range(N_T):
            if missing and missing.get(tid) == t:
                continue
            row = {"territory_id": tid, "t": t, "year": 2023 + t // 12, "month": t % 12 + 1}
            row.update({f"v_{c}": float(v[t]) for c in ("all", "food", "marketplace", "transport", "health")})
            row["v_cafe"] = float(v[t])
            rows.append(row)
    wide = pd.DataFrame(rows)
    wide["log_all"] = np.log(wide["v_all"])
    return wide


def test_national_profile_and_december_peak():
    base = np.ones(12)
    base[11] = 2.0  # декабрь вдвое выше остальных
    other = np.ones(12)
    other[6] = 3.0  # пик в июле
    wide = _panel(
        {
            1: np.tile(base, 2) * 100,
            2: np.tile(base, 2) * 200,
            3: np.tile(base, 2) * 50,
            4: np.tile(other, 2) * 10,
            5: np.tile(other, 2) * 10,
        },
        missing={5: 3},  # неполный ряд не входит
    )
    prof = e3.national_profile(wide, ["all"]).set_index("month")
    assert prof["n_mo"].iloc[0] == 4
    assert prof.at[12, "median"] == pytest.approx(2.0 / (13.0 / 12))
    assert e3.peak_month_share(wide).tolist() == [0.75, 0.75]


def test_series_matrix_requires_full_series():
    wide = _panel({1: np.full(N_T, 100.0), 2: np.full(N_T, 50.0)}, missing={2: 7})
    Y = e3.series_matrix(wide, [1], "all")
    assert Y.shape == (1, N_T) and np.allclose(Y.to_numpy(), np.log(100.0))
    with pytest.raises(ValueError, match="не все"):
        e3.series_matrix(wide, [1, 2], "all")


def test_full_ids_selects_only_full(data):
    ids = e3.full_ids(data.mo)
    status = data.mo.set_index("territory_id")["series_status"].astype(str)
    assert (status.loc[ids] == "full").all()
    assert len(ids) == int((status == "full").sum())


def test_seasonal_excess_matches_eda_data(data, params):
    ids = e3.full_ids(data.mo)
    Y = e3.series_matrix(data.panel_wide, ids, "all")
    ours = e3.seasonal_excess(Y, params.summer_months)
    theirs = data.mo.set_index("territory_id").loc[ids, "summer_excess"]
    assert np.allclose(ours.to_numpy(), theirs.to_numpy())


def test_profile_summer_excess_excludes_december():
    prof = pd.DataFrame([[0.0] * 5 + [0.1] * 3 + [0.0] * 3 + [5.0]], columns=range(1, 13))
    assert e3.profile_summer_excess(prof, (6, 7, 8)).iloc[0] == pytest.approx(0.1)


def test_summer_excess_by_year_averages_to_seasonal_excess():
    Y = _matrix(np.random.default_rng(13).normal(0, 0.1, (20, N_T)))
    by_year = e3.summer_excess_by_year(Y, (6, 7, 8))
    assert list(by_year.columns) == [2023, 2024]
    pooled = np.expm1(np.log1p(by_year).mean(axis=1))  # среднее лог-избытков двух лет
    assert np.allclose(pooled.to_numpy(), e3.seasonal_excess(Y, (6, 7, 8)).to_numpy())


def test_december_over_trend_is_smaller_than_ratio_to_year_mean():
    """Ловушка F06: при росте трат внутри года «декабрь к среднему года» завышен; над трендом МО
    остаётся только сам декабрьский подъём."""
    rng = np.random.default_rng(14)
    n, bump, slope = 40, 0.2, 0.03
    t = np.arange(N_T)
    dec = (t % 12 == 11).astype(float)
    Y = rng.normal(10, 0.3, (n, 1)) + slope * t + bump * dec + rng.normal(0, 0.005, (n, N_T))
    wide = _panel({i + 1: np.exp(Y[i]) for i in range(n)})
    f06_dec = e3.national_profile(wide, ["all"]).set_index("month").at[12, "median"] - 1
    over_trend = e3.common_month_excess(e3.fit_rhythm(_matrix(Y)).g, 12)
    # Над трендом остаётся подъём декабря без той его части, что линейный тренд забирает себе.
    bump_left, _ = e3.detrend(_matrix((bump * dec)[None, :]))
    expected = np.expm1(bump_left.to_numpy()[0, [11, 23]].mean())
    assert over_trend == pytest.approx(expected, abs=0.01)
    assert f06_dec > over_trend + 0.1  # рост 3% в месяц добавляет к декабрю ≈ 5,5 месяца роста


def test_common_month_excess_known_values():
    g = np.zeros(N_T)
    g[11], g[23] = 0.1, 0.3
    assert e3.common_month_excess(g, 12) == pytest.approx(np.expm1(0.2))
    assert e3.common_month_excess(g, 1) == pytest.approx(0.0)


def test_month_ratio_by_year_and_annual_growth():
    y23 = np.full(12, 100.0)
    y23[11] = 200.0
    series = np.r_[y23, y23 * 1.1]
    wide = _panel({1: series, 2: series * 2, 3: series * 3})
    ratio = e3.month_ratio_by_year(wide, 12, "marketplace")
    assert ratio.tolist() == pytest.approx([200 / (1300 / 12) - 1] * 2)
    assert e3.annual_growth_median(wide, "marketplace") == pytest.approx(0.1)


def test_peak_shares_by_group():
    peaks = pd.Series([7, 7, 1, 12, 6], index=[1, 2, 3, 4, 5])
    north = pd.Series([True, True, True, False, False], index=peaks.index)
    assert e3.peak_shares(peaks, (6, 7, 8), north) == pytest.approx((2 / 3, 1 / 2))
    share_in, share_out = e3.peak_shares(peaks, (6, 7, 8), pd.Series(True, index=peaks.index))
    assert share_in == pytest.approx(0.6) and np.isnan(share_out)  # пустая группа — NaN


def test_display_and_joined_names():
    frame = pd.DataFrame(
        {
            "name": [
                "Булунский муниципальный район",
                "городской округ Эгвекинот",
                "муниципальный округ Шурышкарский район",
                "городской округ Анадырь",
            ],
            "name_short": ["Булунский", "Эгвекинот", "Шурышкарский", None],
            "mo_type": ["mr", "go", "mo", "go"],
        }
    )
    names = e3.display_names(frame).tolist()
    assert names == ["Булунский район", "Эгвекинот", "Шурышкарский округ", "городской округ Анадырь"]
    assert e3.join_names([]) == style.NA_TEXT
    assert e3.join_names(["A"]) == "A"
    assert e3.join_names(["A", "B", "C"]) == "A, B и C"
    mo = pd.DataFrame({"territory_id": [1, 2], "name": ["x", "y"]})
    ter = pd.DataFrame({"territory_id": [2, 1], "name_short": ["Б", "А"]})
    assert e3.with_short_names(mo, ter)["name_short"].tolist() == ["А", "Б"]
    assert e3.with_short_names(mo, None) is mo


def test_diverging_classes_symmetric_and_open_ends():
    values = pd.Series([-0.06, -0.02, 0.0, 0.03, 0.07, 0.12, 0.3], index=range(10, 17))
    classes = e3.diverging_classes(values, 0.05, 3)
    nb = style.NBSP
    expected = ["ниже −5%", f"от{nb}−5 до{nb}0%", "0–5%", "5–10%", "10% и выше"]
    assert list(classes.colors) == expected  # в легенде — по возрастанию, крайние классы открыты
    assert classes.labels.tolist() == [expected[i] for i in (0, 1, 2, 2, 3, 4, 4)]
    assert list(classes.labels.index) == list(values.index)
    cm = mpl.colormaps[style.DIV_CMAP]
    lo = e3.MAP_MIN_INTENSITY
    mid = lo + (1 - lo) * 0.5
    # Одинаковое расстояние от нуля — зеркальные цвета; ближний к нулю класс — не белый.
    assert classes.colors[expected[1]] == mpl.colors.to_hex(cm(0.5 - 0.5 * lo))
    assert classes.colors[expected[2]] == mpl.colors.to_hex(cm(0.5 + 0.5 * lo))
    assert classes.colors[expected[0]] == mpl.colors.to_hex(cm(0.5 - 0.5 * mid))
    assert classes.colors[expected[3]] == mpl.colors.to_hex(cm(0.5 + 0.5 * mid))
    assert classes.colors[expected[4]] == mpl.colors.to_hex(cm(1.0))


def test_diverging_classes_empty():
    classes = e3.diverging_classes(pd.Series([np.nan]), 0.05, 5)
    assert classes.labels.empty and classes.colors == {}


def _profile_part(median, q25, q75):
    return pd.DataFrame({"month": range(1, 13), "median": median, "q25": q25, "q75": q75})


def test_december_label_above_or_below_band():
    med = np.ones(12)
    med[10], med[11] = 1.0, 1.2
    q75 = med + 0.05
    q75[9] = 1.3  # октябрь: полоса выше декабрьской точки
    y, up = e3.december_label_y(_profile_part(med, med - 0.05, q75))
    assert up and y == pytest.approx(0.3)
    med2 = np.ones(12)
    med2[10], med2[11] = 0.99, 0.97
    q25 = med2 - 0.02
    q25[10] = 0.9
    y2, up2 = e3.december_label_y(_profile_part(med2, q25, med2 + 0.02))
    assert not up2 and y2 == pytest.approx(-0.1)


def test_noise_by_size_finds_noisier_small_mo():
    rng = np.random.default_rng(12)
    n = 200
    pop = pd.Series(np.exp(rng.uniform(7, 13, n)), index=np.arange(n) + 1)
    sd = 0.2 / np.sqrt(pop.to_numpy() / pop.min())  # шум убывает с населением
    own = pd.DataFrame(rng.normal(size=(n, N_T)) * sd[:, None], index=pop.index)
    pop.iloc[0] = np.nan  # МО без населения в группы не входит
    table = e3.noise_by_size({"all": own}, pop, 10)
    assert table["group"].tolist() == list(range(1, 11))
    assert table["n_mo"].sum() == n - 1
    assert (table["pop_min"].diff().dropna() > 0).all()
    assert e3.noise_ratio(table, "all") > 3


def test_robust_sd_rows_is_scaled_mad():
    row = np.array([[1.0, 2.0, 3.0, 4.0, 100.0]])
    assert e3.robust_sd_rows(row).iloc[0] == pytest.approx(1.4826 * 1.0)


def test_size_groups_needs_enough_mo():
    with pytest.raises(ValueError, match="меньше числа групп"):
        e3.size_groups(pd.Series([1.0, 2.0]), 10)


# --- Подсказки, заголовки, текст -----------------------------------------------------------------


def test_hints_name_outlying_context_and_annotations():
    mo = pd.DataFrame(
        {
            "territory_id": [1, 2, 3, 4, 5, 6],
            "point_lat": [50.0, 51.0, 52.0, 53.0, 54.0, 72.0],
            "nights_pc_2023": [1.0, 1.1, 0.9, 1.0, 1.2, np.nan],
        }
    )
    features = {"point_lat": "широта", "nights_pc_2023": "ночёвки на жителя"}
    h = e3.hints(mo, [6, 3], features, 2.0, {3: "гипотеза: вахта"})
    assert h[6].startswith("широта +")
    assert "ночёвки" not in h[6]  # пропуск признака не упоминается
    assert h[3] == "гипотеза: вахта"
    assert e3.hints(mo, [2], features, 2.0)[2] == style.NA_TEXT


def test_hint_z_is_capped_for_near_zero_mad():
    assert e3._z_text(91.4) == "> +10"
    assert e3._z_text(-15.0) == f"< {style.MINUS}10"
    assert e3._z_text(2.345) == "+2,3"


def _values(**over):
    base = {
        "peak_dec_share_2023": 0.968,
        "peak_dec_share_2024": 0.976,
        "mp_dec": 0.53,
        "reliable_share": 0.156,
        "null_reliable_share": 0.014,
        "north_share_reliable": 0.293,
        "north_share_all": 0.096,
    }
    base.update(over)
    return base


def test_headlines_pass_on_reference_numbers():
    heads = e3.headlines(_values(null_reliable_share=0.0149), 60)
    assert all(h.ok for h in heads.values())
    # Один знак после запятой: 1,49% не превращается в «1%», и отношение к нулю не завышается.
    assert heads["F07"].title == (
        f"Свой устойчивый годовой ритм — у{style.NBSP}15,6% МО, на случайных данных — у{style.NBSP}1,5%"
    )
    assert heads["F06"].title.startswith(f"Декабрь — пик трат у{style.NBSP}97–98% МО;")
    assert "севернее 60-й параллели, во всей выборке — 10%" in heads["F08"].title
    assert all(len(h.title) <= 90 for h in heads.values())


@pytest.mark.parametrize(
    ("fid", "over"),
    [
        ("F06", {"peak_dec_share_2024": 0.8}),
        ("F06", {"peak_dec_share_2023": 0.85}),
        ("F06", {"mp_dec": -0.05}),
        ("F07", {"reliable_share": 0.03, "null_reliable_share": 0.014}),
        ("F07", {"reliable_share": 0.0, "null_reliable_share": 0.0}),
        ("F08", {"north_share_reliable": 0.15, "north_share_all": 0.096}),
        ("F08", {"north_share_reliable": float("nan")}),
    ],
)
def test_headline_fails_when_claim_breaks(fid, over):
    head = e3.headlines(_values(**over), 60)[fid]
    assert not head.ok
    assert head.detail.startswith(fid)


def test_story_role_follows_rejection_rule():
    assert "слой" in e3.story_role({"reliable_share": 0.156, "null_reliable_share": 0.014}, 0.25)
    assert "главной осью" in e3.story_role({"reliable_share": 0.3, "null_reliable_share": 0.014}, 0.25)
    assert "слой" in e3.story_role({"reliable_share": 0.3, "null_reliable_share": 0.2}, 0.25)


@pytest.mark.parametrize(
    ("rho", "text"),
    [
        (0.05, "почти нулевая"),
        (-0.099, "почти нулевая"),
        (0.14, "слабая прямая"),
        (-0.2, "слабая обратная"),
        (0.4, "умеренная прямая"),
        (-0.6, "сильная обратная"),
    ],
)
def test_relation_text_by_sign_and_strength(rho, text):
    assert e3.relation_text(rho) == text


def test_relation_text_missing():
    assert e3.relation_text(float("nan")) == f"не{style.NBSP}оценена"


def _text_values(**over):
    """Числа, на которых все утверждения текста верны (порядок величин — как на реальной панели)."""
    base = {
        "nat_dec": 0.211,
        "nat_dec_detrended": 0.127,
        "mp_dec": 0.528,
        "mp_dec_detrended": 0.238,
        "cafe_aug": 0.192,
        "summer_peak_share_north": 0.80,
        "summer_peak_share_south": 0.37,
        "summer_north": 0.09,
        "summer_south": 0.055,
        "cafe_summer_north": 0.28,
        "cafe_summer_south": 0.20,
        "rho": {"lat": (0.14, 2016), "nights": (-0.12, 1260), "mining": (0.40, 321), "agri": (-0.20, 1079)},
        "rho_within": {
            "lat": (0.09, 2016),
            "nights": (-0.06, 1259),
            "mining": (0.04, 311),
            "agri": (-0.10, 1076),
        },
        "rho_within_max": 0.10,
        "pair_raw_q50": 0.91,
        "pair_own_q90": 0.54,
        "pair_null_q90": 0.27,
        "noise_ratio_all": 2.2,
        "noise_ratio_cafe": 4.4,
        "noise_ratio_transport": 3.0,
    }
    base.update(over)
    return base


def test_text_checks_pass_on_reference_numbers():
    checks = e3.text_checks(_text_values())
    assert len(checks) == 8
    assert all(flag for _, flag, _ in checks), [c for c, flag, _ in checks if not flag]


def test_text_checks_allow_missing_southern_group():
    """Без южных МО с устойчивым ритмом (синтетика) утверждение о северных остаётся верным."""
    assert all(flag for _, flag, _ in e3.text_checks(_text_values(summer_peak_share_south=float("nan"))))


@pytest.mark.parametrize(
    ("over", "word"),
    [
        ({"nat_dec_detrended": 0.25}, "декабрьского"),  # над трендом больше, чем к среднему года
        ({"mp_dec_detrended": -0.01}, "декабрьского"),
        ({"cafe_aug": -0.02}, "общепита"),
        ({"summer_peak_share_north": 0.45, "summer_peak_share_south": 0.3}, "северных"),
        ({"summer_peak_share_north": 0.6, "summer_peak_share_south": 0.7}, "северных"),
        ({"summer_south": 0.12}, "летний избыток"),
        (
            {"rho": {"lat": (0.14, 1), "nights": (-0.12, 1), "mining": (0.4, 1), "agri": (0.15, 1)}},
            "аграрного",
        ),
        ({"pair_raw_q50": 0.2}, "похожи"),
        ({"pair_own_q90": 0.25}, "хвост"),
        ({"noise_ratio_cafe": 0.95}, "шумят"),
    ],
)
def test_text_checks_flag_broken_claim(over, word):
    failed = [claim for claim, flag, _ in e3.text_checks(_text_values(**over)) if not flag]
    assert len(failed) == 1 and word in failed[0]


def test_within_words_follow_numbers():
    """Слова о связях внутри регионов (Б.1, п. 9) выбираются по числам, а не пишутся заранее."""
    v = _text_values()
    word, conclusion = e3.within_words(v["rho"], v["rho_within"])
    assert word == "все связи слабеют" and "свойство региона" in conclusion
    # Одна связь внутри регионов сильнее, чем в целом (как ночёвки на синтетике), — вывод другой.
    stronger = {**v["rho_within"], "nights": (-0.32, 1259)}
    word, conclusion = e3.within_words(v["rho"], stronger)
    assert word.startswith("слабеют не") and "одного региона" in conclusion
    # Все слабее, но одна осталась умеренной — тоже не «все связи слабеют».
    moderate = {k: (r * 0.9, n) for k, (r, n) in v["rho"].items()}
    assert e3.within_words(v["rho"], moderate)[0].startswith("слабеют не")


def test_months_text():
    assert e3.months_text([6, 7, 8]) == "июнь–август"
    assert e3.months_text([1, 7]) == "январь, июль"


# --- Параметры ------------------------------------------------------------------------------------


def _cfg_with(tmp_path, **season) -> Config:
    cfg = make_config(tmp_path)
    data = copy.deepcopy(cfg.data)
    data["eda"]["season"].update(season)
    return Config(data=data, path=cfg.path)


def test_params_defaults_and_config_override(tmp_path):
    p = e3.SeasonParams.from_config(make_config(tmp_path))
    assert (p.reliable_r, p.reliable_amplitude, p.null_repeats) == (0.6, 0.05, 20)
    assert p.top_n == e3.SEASON_DEFAULTS["top_n"]
    assert p.summer_months == (6, 7, 8) and p.north_lat == 60.0
    assert e3.SeasonParams.from_config(_cfg_with(tmp_path, top_n=5)).top_n == 5


@pytest.mark.parametrize(
    "season",
    [
        {"noise_categories": ["all", "fuel"]},
        {"noise_groups": 1},
        {"hist_bin": 0.0},
        {"map_step": 0.0},
        {"map_classes_side": 0},
    ],
)
def test_params_reject_bad_config(tmp_path, season):
    with pytest.raises(ValueError, match="eda.season"):
        e3.SeasonParams.from_config(_cfg_with(tmp_path, **season))


# --- Прогон раздела на синтетике -----------------------------------------------------------------


def test_signal_north_group_has_reliable_rhythm(data, params):
    """Заложенный сигнал: у «северной» группы свой летний ритм в оба года, у остальных — чистый шум."""
    res = e3.compute(data, params, np.random.default_rng(0))
    lat = data.mo.set_index("territory_id").loc[res.ids, "point_lat"]
    north = (lat >= 60).to_numpy()
    reliable = res.reliable.to_numpy()
    assert reliable[north].mean() >= 0.5
    assert reliable[~north].mean() <= 0.1
    v = res.values
    assert v["reliable_share"] >= 3 * v["null_reliable_share"]
    assert v["summer_north"] > v["summer_south"]
    assert v["pair_raw_q50"] > v["pair_own_q50"]
    summer = res.table.set_index("territory_id").loc[res.ids[reliable], "summer_own"]
    assert (summer > 0).all()  # надёжный свой ритм северян — летний подъём


def test_compute_is_deterministic(data, params):
    a = e3.compute(data, params, np.random.default_rng([42, 3])).values
    b = e3.compute(data, params, np.random.default_rng([42, 3])).values
    keys = [k for k, v in a.items() if isinstance(v, float)]
    assert keys and all(a[k] == b[k] or (np.isnan(a[k]) and np.isnan(b[k])) for k in keys)


def test_run_section_outputs(section):
    ctx, finding = section
    assert isinstance(finding, Finding) and finding.section == "e3" and finding.title == e3.TITLE
    assert [f.fid for f in finding.figures] == ["F06", "F07", "F08"]
    assert [f.slug for f in finding.figures] == ["national_rhythm", "own_rhythm", "north_rhythm_map"]
    assert [t.tid for t in finding.tables] == ["T05", "T06", "T07"]
    for rec in finding.figures:
        for rel in (rec.png, rec.svg, rec.data_csv):
            assert (ctx.out_dir / rel).is_file()
        assert len(rec.title) <= 90 and len(rec.subtitle) <= 120 and len(rec.alt) >= 40
    for rec in finding.tables:
        assert (ctx.out_dir / rec.csv).is_file() and rec.markdown.startswith("|")


def test_run_section_facts_and_checks(section):
    _, finding = section
    missing = [k for k in REQUIRED_FACTS if f"e3.{k}" not in finding.facts]
    assert not missing
    # И заголовки F06–F08, и утверждения текста верны на синтетике: полный прогон этапа eda на синтетике
    # (тесты сводки и отчёта) не должен падать из-за раздела E3.
    assert finding.headline_errors == []
    used = {f"{s}.{k}" for s, k in PLACEHOLDER.findall(finding.summary_md)}
    assert used and used <= set(finding.facts)
    assert '"' not in finding.summary_md and " - " not in finding.summary_md and "~" not in finding.summary_md
    for fact in finding.facts.values():
        assert fact.note, fact.key


def test_run_section_context_and_trend_facts(section):
    """Связи летнего избытка в целом и внутри регионов (Б.1, п. 9), слова о силе связи, декабрь над
    трендом."""
    _, finding = section
    facts = finding.facts
    for key in e3.RHO_CONTEXT:
        assert f"e3.rho_summer_{key}_within" in facts
        rho = facts[f"e3.rho_summer_{key}"].value
        assert facts[f"e3.summer_{key}_rel"].value == e3.relation_text(np.nan if rho is None else rho)
    assert facts["e3.n_north_inner"].value <= facts["e3.n_north"].value
    assert facts["e3.reliable_north_n"].value <= facts["e3.reliable_n"].value
    # На синтетике траты растут внутри года: над трендом МО декабрь ниже, чем к среднему года.
    assert 0 < facts["e3.nat_dec_detrended"].value < facts["e3.nat_dec"].value
    assert "+54%" in facts["e3.mp_dec"].note  # пояснение расхождения с ориентиром spec_final


def test_run_section_indicators(section, data):
    _, finding = section
    ind = finding.indicators
    assert list(ind.columns) == ["territory_id", "own_amplitude", "own_repro_r", "own_reliable"]
    assert set(finding.indicator_labels) == {"own_amplitude", "own_repro_r", "own_reliable"}
    assert ind["territory_id"].dtype == "int32" and ind["territory_id"].is_unique
    assert len(ind) == len(e3.full_ids(data.mo))
    assert set(ind["own_reliable"].unique()) <= {0.0, 1.0}
    assert int(ind["own_reliable"].sum()) == finding.facts["e3.reliable_n"].value


def test_run_section_tables_content(section):
    ctx, finding = section
    t05 = pd.read_csv(ctx.out_dir / finding.tables[0].csv)
    assert len(t05) == finding.facts["e3.reliable_n"].value  # CSV — все МО с устойчивым ритмом
    md_rows = finding.tables[0].markdown.count("\n") - 1  # без строк заголовка и разделителя
    assert md_rows == min(len(t05), e3.SEASON_DEFAULTS["top_n"])
    assert t05["amplitude_log"].is_monotonic_decreasing
    assert (t05["repro_r"] > 0.6).all() and (t05["amplitude_log"] > 0.05).all()
    t06 = pd.read_csv(ctx.out_dir / finding.tables[1].csv)
    assert t06["series"].tolist() == ["raw", "detrended", "own"]
    t07 = pd.read_csv(ctx.out_dir / finding.tables[2].csv)
    assert {"sd_all", "sd_cafe", "sd_transport"} <= set(t07.columns)


def test_broken_headline_is_recorded(tmp_path, data, monkeypatch):
    """Вывод заголовка перестал быть верным — ошибка в итоге раздела (сборка потом падает с кодом 3)."""
    monkeypatch.setattr(e3, "PEAK_DEC_MIN", 1.01)
    ctx = make_section_context(tmp_path, "e3", data=data)
    with style.use():
        finding = e3.run_section(ctx)
    assert len(finding.headline_errors) == 1
    assert "F06" in finding.headline_errors[0] and "peak_dec_share_2024" in finding.headline_errors[0]
