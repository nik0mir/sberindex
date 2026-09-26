import math

import numpy as np
import pandas as pd
import pytest
from synth import make_config, make_eda_data, make_processed

from munnet.contracts import CATEGORY_CODES, OKVED_GROUPS, PARTS, MissingInputError
from munnet.eda import stats
from munnet.eda.data import CONTEXT_2023, TERRITORY_COLUMNS, build_mo, build_national, load

SUMMER = (6, 7, 8)
# Сезонный множитель месяца: лето +10%, декабрь +20%.
SEASON = {m: (1.1 if m in SUMMER else 1.2 if m == 12 else 1.0) for m in range(1, 13)}
BASE_SHARES = {"food": 0.4, "transport": 0.05, "health": 0.05, "cafe": 0.05}


def wide_rows(tid: int, scale: float, *, mp: dict[int, float], food: float = 0.4, drop: tuple[str, ...] = ()):
    """Ряд МО: v_all = scale × (20 000 в 2023, 22 000 в 2024) × сезон; доли постоянны внутри года."""
    rows = []
    for t in range(24):
        year, month = 2023 + t // 12, t % 12 + 1
        date = f"{year}-{month:02d}"
        if date in drop:
            continue
        v_all = round(scale * (20_000 if year == 2023 else 22_000) * SEASON[month])
        shares = {**BASE_SHARES, "food": food, "marketplace": mp[year]}
        row = {"territory_id": tid, "date": date, "t": t, "year": year, "month": month, "v_all": v_all}
        for c, sh in shares.items():
            row[f"v_{c}"] = round(v_all * sh)
        row["v_other"] = v_all - sum(row[f"v_{c}"] for c in shares)
        rows.append(row)
    return rows


@pytest.fixture
def crafted():
    """Три МО: 1 — полный ряд; 2 — вдвое больше; 3 — вчетверо больше, без декабря 2024 (11 месяцев)."""
    mp = {2023: 0.10, 2024: 0.15}
    rows = (
        wide_rows(1, 1, mp=mp) + wide_rows(2, 2, mp=mp) + wide_rows(3, 4, mp=mp, food=0.3, drop=("2024-12",))
    )
    wide = pd.DataFrame(rows)
    for p in PARTS:
        wide[f"sh_{p}"] = wide[f"v_{p}"] / wide["v_all"]
    wide["log_all"] = np.log(wide["v_all"].astype(float))
    ter = pd.DataFrame({"territory_id": [1, 2, 3]})
    for c in TERRITORY_COLUMNS:
        ter[c] = 0
    ter["series_status"] = ["full", "full", "partial"]
    ter["area_km2"] = [100.0, 50.0, 10.0]
    ctx = pd.DataFrame(
        {
            "territory_id": [1, 1, 2, 2, 3, 3],
            "year": [2023, 2024] * 3,
            "pop_avg": [1000.0, 990.0, np.nan, 500.0, 250.0, 240.0],
            "wage": [50_000.0, 55_000.0, 60_000.0, 66_000.0, 70_000.0, 77_000.0],
            "ndfl_income_pc": [10_000.0, 11_000.0, np.nan, np.nan, 20_000.0, 21_000.0],
            "workplace_based": [False] * 6,
            "ndfl_ok": [True, True, False, False, True, True],
        }
    )
    for c in CONTEXT_2023:
        ctx[c] = 0.5
    return wide, ter, ctx


def mo_of(crafted) -> pd.DataFrame:
    wide, ter, ctx = crafted
    return build_mo(wide, ter, ctx, summer_months=SUMMER, rel_min_months=6).set_index("territory_id")


def test_levels_growth_and_shares_known_values(crafted):
    mo = mo_of(crafted)
    mean_season = sum(SEASON.values()) / 12
    assert mo.loc[1, "level_2023"] == pytest.approx(20_000 * mean_season)
    assert mo.loc[1, "level_2024"] == pytest.approx(22_000 * mean_season)
    assert mo.loc[1, "growth"] == pytest.approx(0.1)
    assert mo.loc[1, "growth_log"] == pytest.approx(math.log(1.1))
    assert mo.loc[1, "log_level_2023"] == pytest.approx(math.log(20_000 * mean_season))
    assert mo.loc[1, "sh_marketplace_2023"] == pytest.approx(0.10)
    assert mo.loc[1, "sh_marketplace_2024"] == pytest.approx(0.15)
    assert mo.loc[1, "mp_pp_change"] == pytest.approx(5.0)
    assert mo.loc[1, "sh_other_2023"] == pytest.approx(1 - 0.4 - 0.1 - 0.05 * 3)
    clr = mo.loc[:, [f"clr_{p}_2023" for p in PARTS]]
    assert clr.sum(axis=1).abs().max() < 1e-12


def test_eleven_months_give_no_annual_level(crafted):
    mo = mo_of(crafted)
    assert mo.loc[3, "level_2023"] == pytest.approx(4 * 20_000 * sum(SEASON.values()) / 12)
    assert np.isnan(mo.loc[3, "level_2024"])
    assert np.isnan(mo.loc[3, "sh_food_2024"]) and np.isnan(mo.loc[3, "mp_pp_change"])
    assert np.isnan(mo.loc[3, "growth"])  # рост — только у полных рядов
    assert np.isnan(mo.loc[3, "summer_excess"]) and np.isnan(mo.loc[3, "dec_peak"])


def test_level_relative_to_monthly_median(crafted):
    mo = mo_of(crafted)
    assert mo.loc[1, "level_rel_2023"] == pytest.approx(0.5)  # медиана месяца — МО 2
    assert mo.loc[3, "level_rel_2023"] == pytest.approx(2.0)
    assert mo.loc[3, "level_rel_2024"] == pytest.approx(2.0)  # 11 месяцев ≥ 6 — уровень есть
    # в декабре 2024 года нет МО 3: медиана — среднее МО 1 и 2 (1,5 уровня МО 1)
    expected = math.exp((11 * math.log(0.5) + math.log(1 / 1.5)) / 12)
    assert mo.loc[1, "level_rel_2024"] == pytest.approx(expected)
    wide, ter, ctx = crafted
    short = build_mo(wide, ter, ctx, summer_months=SUMMER, rel_min_months=12).set_index("territory_id")
    assert np.isnan(short.loc[3, "level_rel_2024"])


def _expected_seasonal(wide: pd.DataFrame, tid: int) -> tuple[float, float]:
    """Летний избыток и декабрьский пик по определению: остатки МНК ln v_all на t, разности по годам."""
    part = wide.loc[wide["territory_id"] == tid].sort_values("t")
    ln = np.log(part["v_all"].to_numpy(dtype=float))
    t = part["t"].to_numpy(dtype=float)
    resid = ln - np.polyval(np.polyfit(t, ln, 1), t)
    month = part["month"].to_numpy()
    summer, dec = np.isin(month, SUMMER), month == 12
    s, d = [], []
    for k in (0, 1):
        y = (t >= 12 * k) & (t < 12 * (k + 1))
        s.append(resid[y & summer].mean() - resid[y & ~summer & ~dec].mean())
        d.append(resid[y & dec].mean() - resid[y & ~dec].mean())
    return math.expm1(np.mean(s)), math.expm1(np.mean(d))


def test_seasonal_indicators(crafted):
    mo = mo_of(crafted)
    summer, dec = _expected_seasonal(crafted[0], 1)
    assert mo.loc[1, "summer_excess"] == pytest.approx(summer)
    assert mo.loc[1, "dec_peak"] == pytest.approx(dec)
    assert mo.loc[2, "summer_excess"] == pytest.approx(summer)  # масштаб не влияет
    # сезонный множитель: лето +10% к остальным месяцам; тренд МО снят, но скачок уровня между годами — нет
    assert 0.05 < summer < 0.1 and 0.1 < dec < 0.2


def test_seasonal_indicators_ignore_mo_trend(crafted):
    """Линейный тренд МО не меняет сезонных показателей: летние месяцы и декабрь в году позже остальных,
    и без снятия тренда показатели росли бы вместе с ростом трат (Б.1, п. 4)."""
    wide, ter, ctx = crafted
    steep = wide.copy()
    factor = np.exp(0.02 * steep["t"])  # +27% в год
    for c in CATEGORY_CODES:
        steep[f"v_{c}"] = steep[f"v_{c}"] * factor
    steep["log_all"] = np.log(steep["v_all"].astype(float))
    base, trended = mo_of(crafted), mo_of((steep, ter, ctx))
    for col in ("summer_excess", "dec_peak"):
        assert trended.loc[1, col] == pytest.approx(base.loc[1, col])
    # сырая формула без тренда завысила бы декабрьский пик примерно на половину годового прироста
    assert trended.loc[1, "dec_peak"] < math.exp(math.log(1.2) - 3 / 11 * math.log(1.1)) - 1


def test_context_ratios_and_weights(crafted):
    mo = mo_of(crafted)
    assert mo.loc[1, "spend_to_wage_2023"] == pytest.approx(mo.loc[1, "level_2023"] / 50_000)
    assert mo.loc[1, "spend_to_ndfl_2023"] == pytest.approx(mo.loc[1, "level_2023"] / 10_000)
    assert np.isnan(mo.loc[2, "spend_to_ndfl_2023"]) and not mo.loc[2, "ndfl_ok"]
    assert mo.loc[2, "weight"] == 500.0 and np.isnan(mo.loc[2, "pop_2023"])  # вес — население 2024 года
    assert mo.loc[1, "log_density_2023"] == pytest.approx(math.log(1000 / 100))
    for g in OKVED_GROUPS:
        assert f"emp_sh_{g}_2023" in mo.columns


def test_national_medians_and_resident_share(crafted):
    wide, _, ctx = crafted
    long = wide.melt(
        id_vars=["territory_id", "date", "t", "year", "month"],
        value_vars=[f"v_{c}" for c in CATEGORY_CODES],
        var_name="category",
    )
    long["category"] = long["category"].str[2:]
    nat = build_national(long, wide, ctx).set_index(["date", "category"])
    row = nat.loc[("2023-01", "food")]
    assert row["n_mo"] == 3
    assert row["median_share"] == pytest.approx(0.4)
    # МО 2 без населения 2023 года не входит; МО 1 и 3: (0,4·1·1000 + 0,3·4·250) / (1·1000 + 4·250) = 0,35
    assert row["resident_share"] == pytest.approx(0.35)
    assert row["median_value"] == pytest.approx(0.4 * 2 * 20_000)
    assert nat.loc[("2024-12", "all"), "n_mo"] == 2
    assert (nat.xs("all", level="category")["resident_share"] == 1).all()


def test_load_on_synthetic_panel(tmp_path):
    data = make_eda_data(tmp_path)
    mo, nat = data.mo, data.national
    assert len(mo) == len(data.territories) and mo["territory_id"].is_unique
    assert mo["territory_id"].dtype == np.int32
    assert len(nat) == 24 * len(CATEGORY_CODES)
    assert not nat.duplicated(["date", "category"]).any()
    assert data.geo_path.exists() and data.controls and "reason" in data.unmatched.columns
    inner = mo.loc[mo["is_inner_city"]]
    assert inner["workplace_based"].all() and inner["spend_to_ndfl_2023"].isna().all()
    # заложенные сигналы синтетики видны в общих показателях
    north = mo["point_lat"] > 60
    assert mo.loc[north, "summer_excess"].median() > mo.loc[~north, "summer_excess"].median()
    assert stats.spearman(mo["mp_pp_change"], mo["log_level_2023"])[0] < -0.3
    assert stats.eta2(mo["log_level_2024"], mo["region_code"]) > 0.5
    assert stats.spearman(mo["level_2023"], mo["wage_2023"])[0] > 0.5
    assert mo.loc[mo["series_status"] != "full", "pop_2023"].max() < mo["pop_2023"].median()


def test_synthetic_noise_grows_in_small_mo(tmp_path):
    """Заложенный сигнал T07: у малых МО помесячный шум доли транспорта и общепита больше."""
    data = make_eda_data(tmp_path)
    wide = data.panel_wide.set_index(["territory_id", "year", "month"])[["sh_transport", "sh_cafe"]]
    diff = (wide.xs(2024, level="year") - wide.xs(2023, level="year")).dropna()  # сезонность сокращается
    sd = diff.groupby(level="territory_id").std()
    pop = data.mo.set_index("territory_id")["weight"].reindex(sd.index)
    small, large = pop <= pop.quantile(0.25), pop >= pop.quantile(0.75)
    for col in sd.columns:
        assert sd.loc[small, col].median() > 1.5 * sd.loc[large, col].median(), col
    unmatched = data.unmatched
    assert set(unmatched["reason"]) == {"not_in_slice"} and unmatched["oktmo"].isna().all()
    succ = data.territories.loc[data.territories["lineage_role"] == "union_successor", "territory_id"]
    assert set(unmatched["territory_id"]) == set(succ) and set(unmatched["year"]) == {2023}
    assert any(v["kind"] == "info" and v["expected"] is None for v in data.controls.values())


def test_load_reads_unmatched_oktmo_as_text(tmp_path):
    make_processed(tmp_path)
    path = tmp_path / "outputs" / "panel" / "unmatched.csv"
    frame = pd.read_csv(path, dtype=str)
    frame.loc[0, "oktmo"] = "01512000"  # код с ведущим нулём
    frame.to_csv(path, index=False, lineterminator="\n")
    data = load(make_config(tmp_path))
    assert data.unmatched.loc[0, "oktmo"] == "01512000"
    assert data.unmatched["reason"].map(type).eq(str).all()


def test_load_requires_panel_outputs(tmp_path):
    cfg = make_config(tmp_path)
    with pytest.raises(FileNotFoundError, match="сначала запустите этап panel"):
        load(cfg)
    make_processed(tmp_path)
    (tmp_path / "outputs" / "panel" / "controls.json").unlink()
    with pytest.raises(MissingInputError, match="controls.json"):
        load(cfg)
