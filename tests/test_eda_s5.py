"""Раздел E5 «Экономика места»: чистые функции, ловушки и прогон на синтетике (tests/synth.py)."""

import copy
import re

import numpy as np
import pandas as pd
import pytest
from synth import make_section_context

from munnet import style
from munnet.config import Config
from munnet.eda import s5_place as s5
from munnet.eda.base import PLACEHOLDER, Finding

NBSP = " "
REQUIRED = (
    "n_ndfl_usable",
    "spend_to_ndfl_median",
    "spend_to_ndfl_p10",
    "spend_to_ndfl_p90",
    "n_recip_gt3",
    "n_recip_lt005",
    "suburb_n",
    "suburb_ratio",
    "suburb_mw_p",
)


@pytest.fixture(scope="module")
def ctx(tmp_path_factory):
    return make_section_context(tmp_path_factory.mktemp("e5"), "e5")


@pytest.fixture(scope="module")
def finding(ctx):
    with style.use():
        return s5.run_section(ctx)


def _mo(n_regions=10, per_region=30, seed=0):
    """Минимальная таблица mo для чистых функций."""
    rng = np.random.default_rng(seed)
    n = n_regions * per_region
    return pd.DataFrame(
        {
            "territory_id": np.arange(1, n + 1, dtype="int32"),
            "region_code": np.repeat(np.arange(n_regions), per_region),
            "x": rng.normal(size=n),
            "y": rng.normal(size=n),
        }
    )


# --- Прогон раздела на синтетике -----------------------------------------------------------------


def test_run_section_returns_own_outputs(finding, ctx):
    assert isinstance(finding, Finding) and finding.section == "e5" and finding.title == s5.TITLE
    assert [f.fid for f in finding.figures] == ["F14", "F15"]
    assert [f.slug for f in finding.figures] == ["economy_signatures", "work_vs_home"]
    assert sorted(t.tid for t in finding.tables) == ["T11", "T12"]
    for rec in finding.figures:
        for rel in (rec.png, rec.svg, rec.data_csv):
            assert (ctx.out_dir / rel).exists(), rel
        assert len(rec.alt) >= 40 and len(rec.title) <= 90 and len(rec.subtitle) <= 120
    for rec in finding.tables:
        assert (ctx.out_dir / rec.csv).exists()


def test_headlines_pass_checks_on_synthetic(finding):
    assert finding.headline_errors == []


def test_required_facts_present(finding, ctx):
    keys = set(finding.facts)
    missing = [k for k in REQUIRED if f"e5.{k}" not in keys]
    assert missing == []
    for t in ctx.cfg["eda"]["economy_types"]:
        assert f"e5.n_type_{t}" in keys
    # Каждая клетка F14 с |ρ| ≥ 0,3 — в фактах; столбец «уровень трат» — целиком, в целом и внутри регионов.
    cells = pd.read_csv(ctx.out_dir / finding.figures[0].data_csv)
    strong = cells.loc[cells["rho_within"].abs() >= s5.PLACE_DEFAULTS["fact_abs_rho"]]
    assert len(strong) > 0
    for r in strong.itertuples():
        assert f"e5.rho_within_{r.row_code}_{r.col_code}" in keys
    for t in s5.PLACE_TRAITS:
        assert f"e5.rho_within_{t.code}_level" in keys and f"e5.rho_all_{t.code}_level" in keys


def test_fact_keys_are_lowercase(finding):
    # Колонки с заглавными буквами (emp_sh_A_2023) не попадают в ключи фактов: там короткие коды.
    assert all(re.fullmatch(r"e5\.[a-z][a-z0-9_]*", k) for k in finding.facts)


def test_summary_uses_only_known_facts(finding):
    refs = {f"{s}.{k}" for s, k in PLACEHOLDER.findall(finding.summary_md)}
    assert refs and refs <= set(finding.facts)
    assert not re.search(r"\d\.\d", PLACEHOLDER.sub("", finding.summary_md))


def test_indicator_is_log_spend_to_ndfl(finding, ctx):
    ind = finding.indicators
    assert list(ind.columns) == ["territory_id", "log_spend_to_ndfl"]
    assert "log_spend_to_ndfl" in finding.indicator_labels
    assert len(ind) == finding.facts["e5.n_ndfl_usable"].value
    assert np.isfinite(ind["log_spend_to_ndfl"]).all()
    mo = ctx.data.mo.set_index("territory_id")
    assert not mo.loc[ind["territory_id"], "workplace_based"].any()
    assert np.allclose(np.exp(ind["log_spend_to_ndfl"]), mo.loc[ind["territory_id"], "spend_to_ndfl_2023"])


def test_signal_suburbs_spend_more_than_they_earn_locally(finding):
    # В синтетике у пригородов столиц получателей дохода по месту работы в 2,5 раза меньше.
    assert finding.facts["e5.suburb_ratio"].value > 1.2
    assert finding.facts["e5.suburb_mw_p"].value < 0.01
    assert finding.facts["e5.suburb_n"].value > 0


def test_signal_agrarian_places_spend_less_within_regions(finding):
    assert finding.facts["e5.rho_within_agri_level"].value < -0.2


def test_f14_excludes_workplace_based(finding, ctx):
    cells = pd.read_csv(ctx.out_dir / finding.figures[0].data_csv)
    mo = ctx.data.mo
    assert cells["n_all"].max() <= int((~mo["workplace_based"]).sum())
    assert "без внутригородских" in finding.figures[0].subtitle


def test_t12_markdown_is_russian(finding):
    md = next(t for t in finding.tables if t.tid == "T12").markdown
    assert "распределение" in md and "высокое" in md and "низкое" in md
    body = "\n".join(md.splitlines()[2:])
    assert not re.search(r"\d\.\d", body)
    n_rows = len(md.splitlines()) - 2
    assert n_rows == len(s5.QUANTILES) + 2 * s5.PLACE_DEFAULTS["edge_n"]


def test_t11_rules_are_words_in_markdown_and_code_in_csv(finding, ctx):
    rec = next(t for t in finding.tables if t.tid == "T11")
    md = rec.markdown
    header = md.splitlines()[0]
    # В отчёте правило словами и без служебных колонок; код правила и фактическое правило — в CSV.
    assert "занятых в добыче не меньше 20%" in md and "emp_sh_B_2023" not in md
    assert header.startswith("| Тип | Правило | МО |") and "Код" not in header
    assert "внутригородских" not in header
    body = re.sub(r"`[^`]*`", "", "\n".join(md.splitlines()[2:]))
    assert not re.search(r"\d\.\d", body)
    csv = pd.read_csv(ctx.out_dir / rec.csv)
    assert "emp_sh_B_2023 >= 0.2" in csv["rule"].tolist()
    # Фактическое правило воспроизводит число МО типа: внутригородские — только в своём типе.
    mo = ctx.data.mo
    for r in csv.loc[csv["type"] != "all"].itertuples():
        assert len(mo.query(r.rule_applied, engine="python")) == r.n, r.type
    assert csv.set_index("type").loc["north", "rule_applied"] == "(point_lat >= 60) and not workplace_based"


@pytest.mark.parametrize(
    ("key", "rule", "words"),
    [
        ("mining", "emp_sh_B_2023 >= 0.2", "занятых в добыче не меньше 20%"),
        ("public_periphery", "emp_sh_public_2023 >= 0.55", "не меньше 55% работников"),
        ("north", "point_lat >=   60", "центр севернее 60° с. ш."),
        ("resort", "nights_pc_2023 >= 2.5", "не меньше 2,5 ночёвок"),
        ("suburb", "dist_capital_km <= 30 and not is_capital and not is_inner_city", "до 30 км по дорогам"),
        ("capital", "is_capital", "административный центр субъекта"),
        ("mining", "emp_sh_B_2023 >= 0.2 and urban_share_2023 < 0.5", None),  # правило не по шаблону — код
        ("unknown_type", "is_capital", None),
    ],
)
def test_rule_words(key, rule, words):
    out = s5.rule_words(key, rule)
    assert out is None if words is None else words in out


# --- Параметры и текст ---------------------------------------------------------------------------


def _cfg_with(ctx, **place):
    data = copy.deepcopy(ctx.cfg.data)
    data["eda"]["place"] = place
    return Config(data=data, path=ctx.cfg.path)


def test_params_default_and_override(ctx):
    assert s5.params(ctx.cfg)["edge_n"] == s5.PLACE_DEFAULTS["edge_n"]
    assert s5.params(_cfg_with(ctx, edge_n=3))["edge_n"] == 3
    with pytest.raises(KeyError, match="неизвестные параметры"):
        s5.params(_cfg_with(ctx, edge_count=3))


def test_typo_nbsp_after_single_letter_words_and_before_dash():
    text = s5.typo("в пригородах и в центре — выше, 10-й и 90-й, {{e5.suburb_ratio}} раза")
    expected = "в~пригородах и~в~центре~— выше, 10-й и~90-й, {{e5.suburb_ratio}} раза".replace("~", NBSP)
    assert text == expected


def test_quantile_key():
    assert [s5.quantile_key(q) for q in s5.QUANTILES] == ["p10", "median", "p90"]


# --- Типы местной экономики (T11) ----------------------------------------------------------------


def test_select_type_counts_and_missing_values():
    mo = pd.DataFrame(
        {"a": [1.0, np.nan, 60.0, 70.0, 20.0], "b": [True, False, True, False, False], "c": [False] * 5}
    )
    assert s5.select_type(mo, "a >= 60").tolist() == [False, False, True, True, False]
    assert s5.select_type(mo, "a <= 50 and not b and not c").sum() == 1  # пропуск в a — «не входит»
    with pytest.raises(ValueError, match="тип mining"):
        s5.select_type(mo, "emp_sh_B_2023 >= 0.2", "mining")


def test_select_type_on_synthetic_config_rules(ctx):
    mo = ctx.data.mo
    assert s5.select_type(mo, "is_inner_city").sum() == 2
    rule = ctx.cfg["eda"]["economy_types"]["suburb"]["rule"]
    chosen = mo.loc[s5.select_type(mo, rule).to_numpy()]
    assert len(chosen) > 0
    assert (chosen["dist_capital_km"] <= 50).all() and not chosen["is_capital"].any()


def _types_mo():
    return pd.DataFrame(
        {
            "territory_id": np.arange(1, 7, dtype="int32"),
            "name": [f"МО {i}" for i in range(1, 7)],
            "region_code": [1, 1, 1, 2, 2, 2],
            "level_rel_2024": [1.0, 2.0, 4.0, 1.0, 1.0, 3.0],
            **{f"sh_{p}_2024": [0.4] * 6 for p in ("food", "other", "cafe", "marketplace")},
            "summer_excess": [0.1] * 6,
            "spend_to_wage_2023": [0.5, 0.5, 9.0, 0.4, 0.4, 0.4],
            "spend_to_ndfl_2023": [2.0] * 6,
            "mp_pp_change": [4.0] * 6,
            "workplace_based": [False, False, True, False, False, False],
            "is_big": [True, True, True, False, False, False],
            "weight": [10.0, 30.0, 20.0, 5.0, 5.0, 5.0],
        }
    )


def test_type_signatures_baseline_counts_and_examples():
    types = {"big": {"rule": "is_big"}, "none": {"rule": "level_rel_2024 > 100", "label": "Пусто"}}
    names = pd.Series({1: "Первое", 2: "Второе", 3: "Третье"})
    out = s5.type_signatures(_types_mo(), types, year=2024, examples_n=2, names=names)
    assert out["type"].tolist() == ["all", "big", "none"]
    assert out["n"].tolist() == [6, 3, 0]
    big = out.set_index("type").loc["big"]
    # Уровень к медиане своего региона: 1/2, 2/2, 4/2 → медиана 1.
    assert big["level_to_region_2024"] == pytest.approx(1.0)
    # Траты / зарплата у МО, где зарплата считается по месту работы (9,0), не учитываются.
    assert big["spend_to_wage_2023"] == pytest.approx(0.5)
    assert big["examples"] == "Второе; Третье"  # по населению: 30, 20
    assert out.set_index("type").loc["none", "label"] == "Пусто"
    assert out.set_index("type").loc["none", "examples"] == ""


def test_type_signatures_examples_by_column_of_rule():
    types = {"big": {"rule": "is_big", "examples_by": "level_rel_2024"}}
    out = s5.type_signatures(_types_mo(), types, year=2024, examples_n=1)
    assert out.loc[1, "examples"] == "МО 3"
    # Правило-порог: примеры — самые выраженные по колонке порога, а не самые людные.
    types = {"rich": {"rule": "level_rel_2024 >= 1.5"}}
    out = s5.type_signatures(_types_mo(), types, year=2024, examples_n=2)
    assert out.loc[1, "examples"] == "МО 3; МО 6"


def test_type_signatures_exclude_workplace_based():
    # МО 3 — внутригородское (занятость по месту работы): в «крупные» не входит, в тип про себя — входит.
    types = {"big": {"rule": "is_big"}, "inner": {"rule": "workplace_based"}}
    out = s5.type_signatures(_types_mo(), types, year=2024, examples_n=3, exclude_workplace_based=True)
    out = out.set_index("type")
    assert out.loc["big", "n"] == 2 and out.loc["big", "n_inner"] == 0
    assert "МО 3" not in out.loc["big", "examples"]
    assert out.loc["inner", "n"] == 1 and out.loc["inner", "n_inner"] == 1
    kept = s5.type_signatures(_types_mo(), types, year=2024, examples_n=3).set_index("type")
    assert kept.loc["big", "n"] == 3 and kept.loc["big", "n_inner"] == 1
    assert s5.targets_workplace_based("is_inner_city") and not s5.targets_workplace_based("is_capital")


@pytest.mark.parametrize(
    ("rule", "column"),
    [
        ("emp_sh_B_2023 >= 0.2", "emp_sh_B_2023"),
        ("nights_pc_2023 >= 5", "nights_pc_2023"),
        ("point_lat > 60", "point_lat"),
        ("dist_capital_km <= 50 and not is_capital", None),
        ("is_capital", None),
        ("unknown_col >= 1", None),
    ],
)
def test_rule_column(rule, column):
    cols = ["emp_sh_B_2023", "nights_pc_2023", "point_lat", "dist_capital_km", "is_capital"]
    assert s5.rule_column(rule, cols) == column


def test_short_names_fill_bare_numbers():
    ter = pd.DataFrame(
        {
            "territory_id": [1, 2, 3],
            "name": ["городской округ Уфа", "муниципальный округ № 78", "район Н"],
            "name_short": ["Уфа", "№ 78", None],
        }
    )
    names = s5.short_names(ter)
    assert names.tolist() == ["Уфа", "МО № 78", "район Н"]


def test_place_facts_for_story_s6(finding):
    facts = finding.facts
    assert -1 <= facts["e5.rho_ndfl_dist_within"].value <= 1
    types = [k for k in facts if k.startswith("e5.ndfl_type_") and not k.endswith(("_min", "_max", "_label"))]
    assert len(types) == 9
    assert facts["e5.ndfl_type_min"].value <= facts["e5.ndfl_type_max"].value
    assert isinstance(facts["e5.ndfl_type_min_label"].value, str)
    assert "внутригородских" in facts["e5.top_nights"].note


# --- Черты места против черт трат (F14) ----------------------------------------------------------


def test_within_region_matrix_does_not_see_pure_region_effect():
    mo = _mo()
    effect = np.repeat(np.linspace(-5, 5, 10), 30)
    mo["x"] = effect + mo["x"]
    mo["y"] = effect + mo["y"]
    m = s5.within_region_matrix(mo, ["x"], ["y"]).iloc[0]
    assert m["rho_all"] > 0.8
    assert abs(m["rho_within"]) < 0.15
    assert m["n_within"] == m["n_all"] == len(mo)


def test_within_region_matrix_sees_place_effect_and_shape():
    mo = _mo()
    mo["z"] = mo["x"] + 0.3 * np.random.default_rng(1).normal(size=len(mo))
    m = s5.within_region_matrix(mo, ["x", "y"], ["z", "x"])
    assert len(m) == 4 and list(m.columns) == ["row", "col", "rho_within", "n_within", "rho_all", "n_all"]
    assert m.set_index(["row", "col"]).loc[("x", "z"), "rho_within"] > 0.8


def test_within_region_matrix_rank_invariant_to_log():
    # ρ Спирмена не зависит от монотонного преобразования: ln зарплаты и расстояния в F14 не нужен.
    mo = _mo()
    mo["x"] = np.exp(mo["x"])
    a = s5.within_region_matrix(mo, ["x"], ["y"]).iloc[0]
    mo["x"] = np.log(mo["x"])
    b = s5.within_region_matrix(mo, ["x"], ["y"]).iloc[0]
    assert a["rho_within"] == pytest.approx(b["rho_within"])


def _cells(rows):
    return pd.DataFrame(rows, columns=["code", "where", "rho"])


def test_signature_headline_both_sides():
    cells = _cells([("urban", "больше горожан", 0.45), ("agri", "больше аграриев", -0.4), ("x", "x", 0.05)])
    head, chosen = s5.signature_headline(cells, min_abs=0.2, max_len=90)
    assert head.ok and set(chosen) == {"pos", "neg"}
    assert "выше" in head.text and "горожан" in head.text and "аграриев" in head.text
    assert "+0,45" in head.text and "−0,40" in head.text
    assert len(head.text) <= 90


def test_signature_headline_falls_back_to_strongest_when_too_long():
    cells = _cells([("urban", "больше горожан", 0.45), ("agri", "больше аграриев", -0.6)])
    head, chosen = s5.signature_headline(cells, min_abs=0.2, max_len=60)
    assert head.ok and "ниже" in head.text and "аграриев" in head.text and "горожан" not in head.text
    assert set(chosen) == {"pos", "neg"}  # обе стороны прошли порог — обе в фактах и тексте
    assert len(head.text) <= 60


def test_signature_headline_one_side_and_none():
    head, chosen = s5.signature_headline(_cells([("a", "a", 0.5), ("b", "b", -0.1)]), min_abs=0.2, max_len=90)
    assert head.ok and set(chosen) == {"pos"} and "выше" in head.text
    head, chosen = s5.signature_headline(
        _cells([("a", "a", 0.15), ("b", "b", -0.1)]), min_abs=0.2, max_len=90
    )
    assert head.ok and chosen == {} and "слабо" in head.text
    head, chosen = s5.signature_headline(_cells([("a", "a", np.nan)]), min_abs=0.2, max_len=90)
    assert not head.ok and chosen == {}


# --- Где зарабатывают и где тратят (F15, T12) ----------------------------------------------------


def _home_mo():
    n = 12
    return pd.DataFrame(
        {
            "territory_id": np.arange(1, n + 1, dtype="int32"),
            "name": [f"МО {i}" for i in range(1, n + 1)],
            "region_name": ["Регион"] * n,
            "dist_capital_km": [10, 20, 30, 40, 0, 100, 200, 300, 400, 500, 5, np.nan],
            "is_capital": [False] * 4 + [True] + [False] * 7,
            "is_inner_city": [False] * 10 + [True, False],
            "spend_to_ndfl_2023": [3.0, 3.0, 3.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.1, np.nan, 1.0],
            "spend_to_wage_2023": [0.4] * 10 + [9.0, 0.4],
            "recipients_to_pop_2023": [0.2, 0.2, 0.2, 0.2, 0.5, 0.5, 0.5, 0.5, 3.0, 3.5, 4.0, 0.04],
            "ndfl_ok": [True] * 10 + [False, True],
            "workplace_based": [False] * 10 + [True, False],
        }
    )


SUBURB = "dist_capital_km <= 50 and not is_capital and not is_inner_city"


def test_work_home_suburbs_and_usable():
    wh = s5.work_home(_home_mo(), suburb_rule=SUBURB, recipients_ok=(0.05, 3.0))
    assert wh["n_usable"] == 11  # внутригородское МО без 5-НДФЛ на жителя не входит
    assert wh["suburb_n"] == 4
    assert wh["suburb_ratio"] == pytest.approx(3.0)
    assert 0 < wh["suburb_mw_p"] < 0.05
    dist = wh["distributions"]
    assert dist.loc["spend_to_ndfl_2023", "n"] == 11
    assert dist.loc["spend_to_ndfl_2023", "median"] == pytest.approx(1.0)
    # Траты / зарплата: МО с зарплатой по месту работы (9,0) исключено.
    assert dist.loc["spend_to_wage_2023", "n"] == 11
    assert dist.loc["spend_to_wage_2023", "p90"] == pytest.approx(0.4)


def test_work_home_recipient_bounds_are_strict():
    wh = s5.work_home(_home_mo(), suburb_rule=SUBURB, recipients_ok=(0.05, 3.0))
    assert wh["n_recip_gt"] == 2  # 3,5 и 4,0; ровно 3,0 — не больше
    assert wh["n_recip_lt"] == 1  # 0,04


def test_work_home_without_suburbs_gives_nan_ratio():
    wh = s5.work_home(_home_mo(), suburb_rule="dist_capital_km < 0", recipients_ok=(0.05, 3.0))
    assert wh["suburb_n"] == 0 and np.isnan(wh["suburb_ratio"]) and np.isnan(wh["suburb_mw_p"])


def test_ndfl_edges_and_table():
    mo = _home_mo()
    wh = s5.work_home(mo, suburb_rule=SUBURB, recipients_ok=(0.05, 3.0))
    hints = pd.Series("", index=mo["territory_id"].to_numpy())
    edges = s5.ndfl_edges(mo, wh["usable"], 2, hints, pd.Series({1: "Короткое"}))
    assert edges["block"].tolist() == ["high", "high", "low", "low"]
    assert edges["spend_to_ndfl_2023"].tolist() == [3.0, 3.0, 0.1, 1.0]
    assert edges.loc[0, "name"] == "Короткое"
    table = s5.work_home_table(wh["distributions"], edges, s5.QUANTILES)
    assert table["block"].tolist()[:3] == ["distribution"] * 3
    assert table["name"].tolist()[:3] == ["10-й перцентиль", "медиана", "90-й перцентиль"]
    assert str(table["territory_id"].dtype) == "Int32" and table["territory_id"].isna().sum() == 3


@pytest.mark.parametrize(
    ("ratio", "p", "verdict"),
    [
        (2.56, 1e-8, "higher"),
        (1.2, 1e-3, "higher"),  # ровно на пороге — ещё «выше»
        (2.56, 0.2, "higher_weak"),  # направление есть, но p Манна — Уитни не меньше α
        (0.5, 1e-8, "lower"),
        (0.5, float("nan"), "lower_weak"),
        (1.05, 1e-8, "similar"),
        (float("nan"), float("nan"), "unknown"),
    ],
)
def test_suburb_verdict(ratio, p, verdict):
    assert s5.suburb_verdict(ratio, p, 1.2, 0.01) == verdict


def test_suburb_headline_variants():
    high = s5._suburb_headline(2.56, 1e-8, 1.2, 0.01)
    assert high.ok and "в 2,6 раза выше" in high.text.replace(NBSP, " ") and "suburb_mw_p <" in high.check
    weak = s5._suburb_headline(2.56, 0.2, 1.2, 0.01)
    assert weak.ok and "ненадёжно" in weak.text
    low = s5._suburb_headline(0.5, 1e-8, 1.2, 0.01)
    assert low.ok and "ниже" in low.text
    same = s5._suburb_headline(1.05, 0.3, 1.2, 0.01, (2.1, 2.0))
    assert same.ok and "не выделяются" in same.text and "2,10 против 2,00" in same.text.replace(NBSP, " ")
    assert not s5._suburb_headline(float("nan"), float("nan"), 1.2, 0.01).ok
    for verdict in s5._SUBURB_TEXT:
        assert len(s5.typo(s5._suburb_text(verdict, 0, 12.345, (12.345, 12.345)))) <= 90
        summary = s5._suburb_text(verdict, 2, 1.0)
        assert verdict == "unknown" or "{{e5.suburb_mw_p}}" in summary
        assert not re.search(r"\d", PLACEHOLDER.sub("", summary))  # числа — только через ключи фактов


# --- Подсказки ------------------------------------------------------------------------------------


def test_context_hints_extremes_annotations_and_limits():
    mo = _mo(n_regions=1, per_region=40)
    mo["a"] = np.linspace(0.0, 1.0, 40)
    mo.loc[0, "a"] = 50.0  # выброс вверх
    mo.loc[1, "a"] = -50.0  # выброс вниз: у черты нет текста для низких значений
    traits = [s5.Trait("a", "a", "A", "", "много A", "")]
    hints = s5.context_hints(mo, traits, z_thr=2.0, max_hints=1, annotations={3: "пояснение"})
    assert hints.loc[1].startswith("много A (z +")
    assert hints.loc[2] == ""
    assert hints.loc[3] == "пояснение"
    assert hints.loc[10] == ""


def test_context_hints_log_scale_and_max_hints():
    mo = _mo(n_regions=1, per_region=40)
    mo["a"] = np.exp(np.linspace(0.0, 1.0, 40))
    mo.loc[0, "a"] = np.exp(40.0)
    mo["b"] = np.linspace(0.0, 1.0, 40)
    mo.loc[0, "b"] = 60.0
    traits = [s5.Trait("a", "a", "A", "", "много A", "", "log"), s5.Trait("b", "b", "B", "", "много B")]
    one = s5.context_hints(mo, traits, z_thr=2.0, max_hints=1)
    both = s5.context_hints(mo, traits, z_thr=2.0, max_hints=2)
    assert one.loc[1].count("(z") == 1 and both.loc[1].count("(z") == 2
    assert "z +" in both.loc[1] and "," not in re.findall(r"z \+(\S+)\)", both.loc[1])[0]  # |z| ≥ 10 — целым


def test_transform_values():
    x = pd.Series([0.0, 1.0, np.e - 1])
    assert s5.transform_values(x, "log1p").tolist() == pytest.approx([0.0, np.log(2), 1.0])
    assert np.isnan(s5.transform_values(x, "log").iloc[0])
    with pytest.raises(ValueError):
        s5.transform_values(x, "sqrt")


# --- Корзина, маскировка регионом и сверка с ориентиром (F14) -----------------------------------


def _long_matrix(cells):
    """Длинная таблица клеток F14: (row_code, row_where, col_code, rho_within, rho_all)."""
    return pd.DataFrame(cells, columns=["row_code", "row_where", "col_code", "rho_within", "rho_all"])


def test_column_leaders_pick_strongest_abs_and_skip():
    m = _long_matrix(
        [
            ("urban", "больше горожан", "cafe", 0.5, 0.4),
            ("density", "плотнее", "cafe", -0.6, -0.1),
            ("wage", "выше зарплата", "cafe", 0.9, 0.8),  # пропускается: тема E2
            ("urban", "больше горожан", "food", np.nan, np.nan),
        ]
    )
    out = s5.column_leaders(m, ["cafe", "food", "none"], skip=["wage"]).set_index("col_code")
    assert out.loc["cafe", "row_code"] == "density" and out.loc["cafe", "rho"] == pytest.approx(-0.6)
    assert out.loc["cafe", "max_abs"] == pytest.approx(0.6)
    assert np.isnan(out.loc["food", "rho"]) and pd.isna(out.loc["food", "where"])
    assert np.isnan(out.loc["none", "max_abs"])


def test_masking_cell_finds_sign_flip():
    cells = pd.DataFrame(
        {
            "code": ["access", "urban", "old"],
            "where": ["выше доступность", "больше горожан", "больше пожилых"],
            "rho_all": [-0.19, 0.54, -0.52],
            "rho_within": [0.30, 0.61, np.nan],
        }
    )
    assert s5.masking_cell(cells)["code"] == "access"
    assert s5.masking_cell(cells.iloc[2:]) is None  # пар без пропусков нет


def test_section_level_rho_uses_hidden_section_as_missing():
    mo = pd.DataFrame({"territory_id": [1, 2, 3, 4, 5], "lvl": [1.0, 2.0, 3.0, 4.0, 5.0]})
    shares = pd.DataFrame(
        {
            "territory_id": [1, 2, 3, 4, 1, 2, 3, 4, 5],
            "year": [2023] * 4 + [2024] * 5,
            "section": ["A"] * 9,
            "share": [0.4, 0.3, 0.2, 0.1, 0.1, 0.2, 0.3, 0.4, 0.5],
        }
    )
    out = s5.section_level_rho(mo, shares, ["A", "O"], "lvl", 2023).set_index("section")
    # Раздел A 2023 года раскрыт у 4 МО (у МО 5 скрыт — пропуск, а не ноль) и убывает с уровнем.
    assert out.loc["A", "n"] == 4 and out.loc["A", "rho"] == pytest.approx(-1.0)
    assert out.loc["O", "n"] == 0 and np.isnan(out.loc["O", "rho"])


# --- Пригороды: сравнение групп и чувствительность к правилу --------------------------------------


def test_compare_groups_ratio_and_empty_group():
    values = pd.Series([3.0, 3.0, 3.0, 1.0, 1.0, 1.0, 9.0])
    usable = pd.Series([True] * 6 + [False])
    group = pd.Series([True, True, True, False, False, False, True])
    out = s5.compare_groups(values, usable, group)
    assert out["n"] == 3 and out["ratio"] == pytest.approx(3.0)  # непригодное МО (9,0) не входит
    empty = s5.compare_groups(values, usable, pd.Series([False] * 7))
    assert empty["n"] == 0 and np.isnan(empty["ratio"]) and np.isnan(empty["p"])


@pytest.mark.parametrize(
    ("rule", "value"),
    [
        (SUBURB, 50.0),
        ("dist_capital_km < 12.5", 12.5),
        ("is_capital", None),
        ("dist_capital_km_x <= 5", None),  # другая колонка с тем же началом имени
    ],
)
def test_rule_threshold(rule, value):
    assert s5.rule_threshold(rule, "dist_capital_km") == value


def test_suburb_variants_replace_threshold_and_add_districts():
    variants = s5.suburb_variants(SUBURB, "dist_capital_km", [25, 100], ["mr", "mo"])
    assert len(variants) == 4
    labels, rules = zip(*variants, strict=True)
    assert rules[0] == SUBURB.replace("50", "25") and rules[2] == SUBURB.replace("50", "100")
    assert rules[1] == f"({rules[0]}) and mo_type in ['mr', 'mo']"
    assert "только районы" in labels[1]
    # Правило без порога расстояния: только оно само.
    assert [r for _, r in s5.suburb_variants("is_capital", "dist_capital_km", [25], [])] == ["is_capital"]


def test_suburb_sensitivity_on_rule_variants():
    types = ["mr", "mr", "go", "go", "go", "mr", "mr", "mr", "mr", "mr", "vgt", "mr"]
    mo = _home_mo().assign(mo_type=types)
    wh = s5.work_home(mo, suburb_rule=SUBURB, recipients_ok=(0.05, 3.0))
    variants = s5.suburb_variants(SUBURB, "dist_capital_km", [25, 50], ["mr", "mo"])
    out = s5.suburb_sensitivity(mo, mo["spend_to_ndfl_2023"], wh["usable"], variants).set_index("variant")
    assert out.loc["до 50 км", "n"] == wh["suburb_n"] == 4
    assert out.loc["до 50 км", "ratio"] == pytest.approx(wh["suburb_ratio"])
    assert out.loc["до 25 км", "n"] == 2  # 10 и 20 км
    assert out.loc["до 50 км, только районы и округа", "n"] == 2  # городские округа (30 и 40 км) выпали


def test_work_home_counts_inner_city_among_high_recipients():
    wh = s5.work_home(_home_mo(), suburb_rule=SUBURB, recipients_ok=(0.05, 3.0))
    assert wh["n_recip_gt"] == 2 and wh["n_recip_gt_inner"] == 1  # 4,0 — у внутригородской территории
    mo = _home_mo()
    mo.loc[9, "is_inner_city"] = True  # 3,5 получателя на жителя — тоже внутригородская
    assert s5.work_home(mo, suburb_rule=SUBURB, recipients_ok=(0.05, 3.0))["n_recip_gt_inner"] == 2


def test_recipient_defects_count_units_flag_and_raw_ratio():
    long = pd.DataFrame(
        {
            "territory_id": [1, 2, 3, 4, 4],
            "year": [2023, 2023, 2023, 2023, 2024],
            "indicator": ["ndfl_recipients"] * 5,
            "dim": ["TOTAL"] * 5,
            "value": [5.0, 30_000.0, 100.0, 20_000.0, 7.0],
            "flag": ["recipients_jump", "", "anomaly;recipients_jump", None, "recipients_jump"],
        }
    )
    annual = pd.DataFrame(
        {
            "territory_id": [1, 2, 3, 4],
            "year": [2023] * 4,
            "pop_avg": [60_000.0, 50_000.0, 40_000.0, 80_000.0],
        }
    )
    out = s5.recipient_defects(long, annual, 2023, 0.05)
    # Флаг у МО 1 и 3 (флаг 2024 года не в счёт); меньше 0,05 на жителя — 5 и 100 получателей.
    assert out == {"n_jump": 2, "n_lt_raw": 2}


# --- Текст «Что видно»: слова по знаку чисел, числа — только ключами фактов ----------------------

# Цифры вне {{ключей}} допустимы только в названиях: 5-НДФЛ, перцентили, номера сюжета, рисунков и таблиц.
ALLOWED_DIGITS = re.compile(r"5-НДФЛ|10-й|90-й|С6|T1[12]|рис\.\s1[45]|этапе\s2|раздел[а-я]*\s2")


def _no_bare_numbers(text):
    return not re.search(r"\d", ALLOWED_DIGITS.sub("", PLACEHOLDER.sub("", text.replace(NBSP, " "))))


def test_summary_has_numbers_only_in_fact_keys(finding):
    assert _no_bare_numbers(finding.summary_md)
    for key in ("e5.lead_cafe_rho", "e5.suburb_sens_max", "e5.n_recip_jump"):
        assert "{{" + key + "}}" in finding.summary_md


def _sig(lead_rows, mask=None, chosen=None):
    lead = pd.DataFrame(lead_rows, columns=["col_code", "row_code", "where", "rho", "max_abs"])
    return {"chosen": chosen or {}, "lead": lead.set_index("col_code"), "mask": mask}


def test_signature_text_compact_when_one_trait_leads_basket():
    rows = [
        ("cafe", "density", "плотнее", 0.62, 0.62),
        ("food", "density", "плотнее", -0.64, 0.64),
        ("mp_pp", "density", "плотнее", -0.52, 0.52),
        ("summer", "dist", "дальше", 0.25, 0.25),
        ("other", "public", "бюджетники", -0.13, 0.13),
    ]
    text = " ".join(s5._signature_text(_sig(rows), 0.2, 0.3)).replace(NBSP, " ")
    assert "Где {{e5.lead_cafe_where}}, доля общепита в корзине выше" in text
    assert "доля продуктов — ниже" in text and "прирост доли маркетплейсов — ниже" in text
    assert "слабо связаны летний избыток и доля «Прочего»" in text
    assert _no_bare_numbers(text)


def test_signature_text_separate_clauses_and_mask_sign_flip():
    rows = [
        ("cafe", "urban", "больше горожан", 0.5, 0.5),
        ("food", "agri", "больше аграриев", 0.4, 0.4),
        ("mp_pp", "old", "больше пожилых", 0.05, 0.05),  # слабее порога — не называется
        ("summer", "dist", "дальше", 0.45, 0.45),  # сильная связь — не «слабо»
        ("other", "public", "бюджетники", -0.13, 0.13),
    ]
    mask = pd.Series({"where": "выше доступность", "rho_all": -0.19, "rho_within": 0.30})
    bullets = s5._signature_text(_sig(rows, mask), 0.2, 0.3)
    assert len(bullets) == 2 and bullets[1].startswith("**Карта регионов маскирует место.**")
    text = " ".join(bullets).replace(NBSP, " ")
    assert "где {{e5.lead_cafe_where}}" in text and "где {{e5.lead_food_where}}" in text
    assert "маркетплейсов" not in text
    assert "слабо связаны доля «Прочего»" in text and "летний" not in text
    assert "без учёта регионов траты ниже" in text and "внутри регионов — выше" in text
    same_sign = pd.Series({"where": "плотнее", "rho_all": 0.19, "rho_within": 0.59})
    assert "маскирует связи места" in " ".join(s5._signature_text(_sig(rows, same_sign), 0.2, 0.3))
    weak_mask = pd.Series({"where": "плотнее", "rho_all": 0.5, "rho_within": 0.55})
    assert "Карта регионов" not in " ".join(s5._signature_text(_sig(rows, weak_mask), 0.2, 0.3))


@pytest.mark.parametrize(
    ("ratios", "verdict", "phrase"),
    [
        ([1.01, 1.14], "similar", "не зависит от правила"),
        ([1.3, 1.5], "higher", "устойчив"),
        ([1.01, 1.3], "similar", "зависит от правила"),
        ([1.1], "similar", ""),  # один вариант — сравнивать не с чем
    ],
)
def test_sensitivity_text(ratios, verdict, phrase):
    text = s5._sensitivity_text(pd.DataFrame({"ratio": ratios}), verdict, 1.2)
    assert (phrase in text) if phrase else text == ""
    assert _no_bare_numbers(text)


# --- Пояснение цветов F15 -------------------------------------------------------------------------


def test_emptiest_corner_avoids_points_and_labels():
    crowded_top_left = np.array([[0.1, 0.95]] * 5)
    assert s5.emptiest_corner(crowded_top_left, np.empty((0, 2))) == ("right", "top")
    both_top = np.array([[0.1, 0.95], [0.9, 0.95]])
    label_bottom_left = np.array([[0.05, 0.05]])
    assert s5.emptiest_corner(both_top, label_bottom_left) == ("right", "bottom")


def test_axes_fraction_on_log_scale():
    frac = s5._axes_fraction(
        pd.Series([1.0, 10.0, 100.0]), pd.Series([0.1, 1.0, 10.0]), (1.0, 100.0), (0.1, 10.0)
    )
    assert frac.tolist() == [pytest.approx([0.0, 0.0]), pytest.approx([0.5, 0.5]), pytest.approx([1.0, 1.0])]


def test_new_facts_present_on_synthetic(finding):
    keys = set(finding.facts)
    for k in (
        "n_recip_gt3_inner",
        "n_recip_lt005_raw",
        "n_recip_jump",
        "n_ndfl_no_level",
        "suburb_ratio_min",
        "suburb_sens_n",
        "suburb_sens_min",
        "suburb_sens_max",
        "mask_where",
        "lead_cafe_rho",
        "max_abs_summer",
        "rho_all_sec_a_level",
        "rho_all_sec_o_level",
        "rho_all_sec_g_level",
    ):
        assert f"e5.{k}" in keys, k
    facts = finding.facts
    lo, hi = facts["e5.suburb_sens_min"].value, facts["e5.suburb_sens_max"].value
    assert lo <= facts["e5.suburb_ratio"].value <= hi
    assert facts["e5.n_recip_lt005_raw"].value >= facts["e5.n_recip_lt005"].value


# --- Доля бюджетников — та же ось, что зарплата: частный ρ и заголовок F14 ----------------------------


def test_partial_within_removes_control_axis():
    # «Бюджетники» — зеркало зарплаты: связь с тратами есть, но при равной зарплате исчезает;
    # «горожане» связаны с тратами и помимо зарплаты.
    mo = _mo(n_regions=10, per_region=60, seed=3)
    rng = np.random.default_rng(4)
    wage = rng.normal(size=len(mo))
    urban = rng.normal(size=len(mo))
    mo = mo.assign(
        wage=wage,
        public=-wage + 0.3 * rng.normal(size=len(mo)),
        urban=urban,
        level=wage + urban + 0.3 * rng.normal(size=len(mo)),
    )
    out = s5.partial_within(mo, ["public", "urban", "wage"], "level", "wage").set_index("row")
    assert abs(out.loc["public", "rho_partial"]) < 0.1 and out.loc["public", "rho_control"] < -0.8
    assert out.loc["urban", "rho_partial"] > 0.5
    assert np.isnan(out.loc["wage", "rho_partial"]) and out.loc["wage", "n_partial"] == 0


def test_split_by_partial_drops_axis_traits_from_headline():
    cells = pd.DataFrame(
        {
            "code": ["urban", "public", "old", "x"],
            "where": ["больше горожан", "больше бюджетников", "больше пожилых", "x"],
            "rho": [0.61, -0.52, -0.50, 0.3],
            "partial": [0.49, -0.08, -0.35, np.nan],  # у «x» частный ρ не посчитан — не отбрасывается
        }
    )
    eligible, axis = s5.split_by_partial(cells, 0.1)
    assert eligible["code"].tolist() == ["urban", "old", "x"]
    assert axis["code"] == "public"
    head, chosen = s5.signature_headline(eligible, min_abs=0.2, max_len=90)
    assert chosen["neg"]["code"] == "old" and "бюджетников" not in head.text
    assert s5.split_by_partial(cells.iloc[[0, 2]], 0.1)[1] is None


def test_f14_has_partial_column_and_facts(finding, ctx):
    cells = pd.read_csv(ctx.out_dir / finding.figures[0].data_csv)
    given = cells["rho_within_given_wage"]
    level = cells["col_code"] == "level"
    assert given[~level].isna().all() and given[level & (cells["row_code"] != "wage")].notna().all()
    facts = finding.facts
    for t in s5.PLACE_TRAITS:
        assert (f"e5.partial_{t.code}_level_wage" in facts) == (t.code != "wage")
    for k in (
        "axis_where",
        "axis_rho",
        "axis_partial",
        "axis_rho_control",
        "sig_pos_partial",
        "mask_rho_all_inner",
    ):
        assert f"e5.{k}" in facts, k
    assert (
        "равной" in finding.figures[0].check
        and "без внутригородских территорий" in finding.figures[0].subtitle
    )


def test_display_names_add_type_to_single_adjectives():
    ter = pd.DataFrame(
        {
            "territory_id": [1, 2, 3, 4, 5],
            "name": [
                "Карагинский муниципальный район",
                "Тенькинский муниципальный округ",
                "городской округ Светлогорский",
                "Таймырский Долгано-Ненецкий муниципальный район",
                "муниципальный округ № 78",
            ],
            "name_short": [
                "Карагинский",
                "Тенькинский",
                "Светлогорский",
                "Таймырский Долгано-Ненецкий",
                "№ 78",
            ],
            "mo_type": ["mr", "mo", "go", "mr", "vgt"],
        }
    )
    assert s5.display_names(ter).tolist() == [
        "Карагинский район",
        "Тенькинский округ",
        "Светлогорский",
        "Таймырский Долгано-Ненецкий",
        "МО № 78",
    ]


def test_one_line_joins_hyphen_breaks():
    assert s5.one_line("Доля\nмаркет-\nплейсов") == "Доля маркетплейсов"
    assert s5.one_line("Занятые в строительстве,\nторговле, транспорте") == (
        "Занятые в строительстве, торговле, транспорте"
    )


# --- С6: столицы, поправка на структуру, Московская область, линия «траты ~ зарплата» --------------


def test_work_home_excludes_unknown_distance_and_counts_capitals():
    wh = s5.work_home(_home_mo(), suburb_rule=SUBURB, recipients_ok=(0.05, 3.0))
    # МО 12 без расстояния пригодно, но ни пригород, ни «остальное»; столица (МО 5) — среди остальных.
    assert wh["n_usable"] == 11 and wh["n_dist_unknown"] == 1 and wh["other_n"] == 6
    assert wh["capital_n"] == 1 and wh["capital_median"] == pytest.approx(1.0)
    assert not wh["eligible"].iloc[11] and wh["usable"].iloc[11]


def test_paired_by_region_separates_capital_gap_from_suburbs():
    # В каждом регионе: столица 1, пригороды 2, прочие 2 — пригороды выше столицы, но и прочие тоже.
    rows = []
    for r in range(4):
        rows += [(r, True, False, 1.0), (r, False, True, 2.0), (r, False, True, 2.0), (r, False, False, 2.0)]
    rows.append((9, False, True, 5.0))  # регион без столицы и прочих — не в сравнении
    d = pd.DataFrame(rows, columns=["region", "capital", "group", "v"])
    out = s5.paired_by_region(d["v"], pd.Series(True, index=d.index), d["region"], d["capital"], d["group"])
    assert out["n_regions"] == 4 and out["group_above"] == 4 and out["other_above"] == 4
    assert out["group_to_capital"] == pytest.approx(2.0) and out["other_to_capital"] == pytest.approx(2.0)
    assert out["group_to_other"] == pytest.approx(1.0)
    empty = s5.paired_by_region(
        d["v"], pd.Series(False, index=d.index), d["region"], d["capital"], d["group"]
    )
    assert empty["n_regions"] == 0 and np.isnan(empty["group_to_capital"])


def test_region_residuals_remove_region_and_controls():
    rng = np.random.default_rng(5)
    region = np.repeat(np.arange(8), 25)
    x = rng.normal(size=region.size)
    y = np.repeat(rng.normal(size=8), 25) + 0.7 * x
    assert np.allclose(s5.region_residuals(y, region, x.reshape(-1, 1)), 0.0, atol=1e-9)
    # Без контролей — только центрирование по региону.
    res = s5.region_residuals(y, region, np.empty((region.size, 0)))
    assert np.allclose(pd.Series(res).groupby(region).mean(), 0.0, atol=1e-12)


def test_adjusted_comparison_sees_effect_beyond_controls():
    rng = np.random.default_rng(6)
    n = 400
    group = pd.Series(rng.random(n) < 0.3)
    # Контроль у группы выше: сырое различие ×exp(0,8 · 0,5) ≈ 1,49, при равном контроле — нет различия.
    mo = pd.DataFrame(
        {"region_code": np.repeat(np.arange(8), n // 8), "ctrl": rng.normal(size=n) + 0.5 * group}
    )
    values = pd.Series(np.exp(0.8 * mo["ctrl"] + 0.05 * rng.normal(size=n)))
    everyone = pd.Series(True, index=mo.index)
    raw = s5.compare_groups(values, everyone, group)
    out = s5.adjusted_comparison(mo, values, everyone, group, ["ctrl"])
    assert raw["ratio"] > 1.3 and out["ratio"] == pytest.approx(1.0, abs=0.03) and out["n_all"] == n
    # Собственный эффект группы ×1,3 сверх контроля виден.
    out2 = s5.adjusted_comparison(mo, values * np.where(group, 1.3, 1.0), everyone, group, ["ctrl"])
    assert out2["ratio"] == pytest.approx(1.3, rel=0.05) and out2["p"] < 1e-6
    none = s5.adjusted_comparison(mo, values, ~everyone, group, ["ctrl"])
    assert none["n_all"] == 0 and np.isnan(none["ratio"])


def test_outer_center_check_uses_straight_distance_to_city():
    mo = pd.DataFrame(
        {
            "territory_id": np.arange(1, 7),
            "region_name": ["Город"] * 2 + ["Область"] * 4,
            "x_aea": [0.0, 2000.0, 10_000.0, 30_000.0, 80_000.0, 120_000.0],
            "y_aea": [0.0] * 6,
            "pop_2023": [1.0, 1.0, 1, 1, 1, 1],
            "spend_to_ndfl_2023": [np.nan, np.nan, 1.0, 1.2, 2.0, 2.2],
        }
    )
    usable = mo["spend_to_ndfl_2023"].notna()
    out = s5.outer_center_check(mo, usable, "Область", "Город", 50)
    # Центр города — 1 км; ближние МО — 9 и 29 км, дальние — 79 и 119 км.
    assert out["n"] == 4 and out["n_near"] == 2
    assert (
        out["near"] == pytest.approx(1.1)
        and out["far"] == pytest.approx(2.1)
        and out["rho"] == pytest.approx(1.0)
    )
    missing = s5.outer_center_check(mo, usable, "Нет такого", "Город", 50)
    assert missing["n"] == 0 and np.isnan(missing["near"])


def test_wage_line_check_finds_group_above_line():
    rng = np.random.default_rng(7)
    n = 300
    group = pd.Series(rng.random(n) < 0.2)
    wage = np.exp(rng.normal(10.5, 0.3, n))
    level = np.exp(0.7 * np.log(wage) + 1.0 + 0.02 * rng.normal(size=n)) * np.where(group, 1.2, 1.0)
    mo = pd.DataFrame({"wage_2023": wage, "level_2023": level, "workplace_based": False})
    out = s5.wage_line_check(mo, pd.Series(True, index=mo.index), group, 30)
    assert out["ratio"] == pytest.approx(1.2, rel=0.03) and out["p"] < 1e-6
    assert out["top_n"] == 30 and out["top_k"] >= 25 and out["base_share"] == pytest.approx(group.mean())


@pytest.mark.parametrize(
    ("ratio", "p", "verdict"),
    [
        (1.3, 1e-5, "strong"),
        (0.7, 1e-5, "strong"),
        (1.15, 1e-5, "weak"),
        (1.3, 0.2, "none"),
        (np.nan, 0.1, "unknown"),
    ],
)
def test_effect_verdict(ratio, p, verdict):
    assert s5.effect_verdict(ratio, p, 1.2, 0.01) == verdict


def test_s6_facts_on_synthetic(finding, ctx):
    facts = finding.facts
    for k in (
        "suburb_ratio_adj",
        "suburb_adj_p",
        "suburb_to_capital_within",
        "other_to_capital_within",
        "suburb_to_other_within",
        "n_regions_paired",
        "capital_median",
        "rho_ratio_recipients",
        "rho_ratio_recipients_within",
        "moscow_obl_near_ratio",
        "moscow_obl_far_ratio",
        "suburb_wage_ratio",
        "suburb_wage_p",
        "n_dist_unknown",
        "suburb_low_names",
        "n_zero_dist",
    ):
        assert f"e5.{k}" in facts, k
    # В синтетике у пригородов получателей в 2,5 раза меньше, а доли пожилых и бюджетников с этим не связаны:
    # поправка на структуру эффект не убирает.
    assert facts["e5.suburb_ratio_adj"].value > 1.5 and facts["e5.suburb_adj_p"].value < 0.01
    # Отношение по рангу обратно числу получателей на жителя.
    assert facts["e5.rho_ratio_recipients"].value < -0.3
    data = pd.read_csv(ctx.out_dir / finding.figures[1].data_csv)
    assert data["is_capital"].any() and not (data["is_capital"] & data["is_suburb"]).any()
    assert "больше 1 — норма" in finding.figures[1].subtitle.replace(NBSP, " ")


def test_summary_is_bulleted_and_names_sample_of_masking(finding):
    text = finding.summary_md.replace(NBSP, " ")
    what, meaning = text.split("**Что это значит для сюжета.**")
    bullets = [line for line in what.splitlines() if line.startswith("- **")]
    assert len(bullets) >= 4
    assert "Крайние МО — гипотезы" in what and "безналичные траты" in what and "по карте" not in text
    if "{{e5.mask_rho_all}}" in text:
        assert "за 2023 год" in text
