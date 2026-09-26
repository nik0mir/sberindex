import json

import numpy as np
import pandas as pd
import pytest
from synth import make_section_context

from munnet import style
from munnet.eda.base import (
    Finding,
    display_name,
    display_names,
    load_finding,
    make_fact,
    save_finding,
    to_markdown,
)

THIN, NBSP, MINUS = " ", " ", "−"
ALT = "Доля маркетплейсов растёт быстрее там, где тратят меньше: разница почти в два раза"


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    from synth import make_eda_data

    return make_eda_data(tmp_path_factory.mktemp("base"))


@pytest.fixture
def ctx(tmp_path, data):
    return make_section_context(tmp_path, "e4", data=data)


@pytest.mark.parametrize(
    ("kind", "value", "text"),
    [
        ("int", 2190, "2190"),
        ("int", np.int64(50521), f"50{THIN}521"),
        ("num1", 2.05, "2,0"),
        ("num2", 0.448, "0,45"),
        ("num3", 0.8654, "0,865"),
        ("pct", 0.156, "15,6%"),
        ("pct_signed", 0.151, "+15,1%"),
        ("pp", 4.1, f"+4,1{NBSP}п.{NBSP}п."),
        ("rub", 26498.4, f"26{THIN}498{NBSP}₽"),
        ("krub", 26498.4, f"26,5{NBSP}тыс.{NBSP}₽"),
        ("rho", -0.717, f"{MINUS}0,72"),
        ("p", 4.5e-8, "4,5·10⁻⁸"),
        ("str", "Белгородская, Брянская", "Белгородская, Брянская"),
    ],
)
def test_fact_formats_by_kind(kind, value, text):
    fact = make_fact("e1.x", value, kind)
    assert fact.text == text


def test_fact_values_are_plain_json():
    assert make_fact("e1.n", np.int32(7), "int").value == 7
    assert type(make_fact("e1.r", np.float32(0.5), "rho").value) is float
    missing = make_fact("e1.m", float("nan"), "pct")
    assert missing.value is None and missing.text == "—"
    with pytest.raises(ValueError, match="неизвестный вид"):
        make_fact("e1.x", 1, "percent")
    with pytest.raises(ValueError, match="требует числа"):
        make_fact("e1.x", "много", "int")


def test_context_fact_prefix_and_duplicates(ctx):
    fact = ctx.fact("mp_share_jan23", 0.092, "pct", note="медиана долей по МО месяца")
    assert fact.key == "e4.mp_share_jan23" and ctx.facts["e4.mp_share_jan23"] is fact
    with pytest.raises(ValueError, match="уже записан"):
        ctx.fact("mp_share_jan23", 0.1, "pct")
    with pytest.raises(ValueError, match="без префикса"):
        ctx.fact("e4.other", 1, "int")


def test_headline_records_failures(ctx):
    assert ctx.headline("Доля выросла вдвое", True, "F11: 9,2 → 19,4%") == "Доля выросла вдвое"
    assert ctx.headline_errors == []
    ctx.headline("Доля выросла вдвое", False, "F11: 9,2 → 12,0%")
    assert len(ctx.headline_errors) == 1 and "F11: 9,2 → 12,0%" in ctx.headline_errors[0]


def _figure():
    fig, ax = style.new_figure("full")
    ax.plot([0, 1, 2], [0.1, 0.15, 0.2], color=style.PALETTE["marketplace"])
    return fig


def test_save_figure_writes_three_files_and_record(ctx):
    data = pd.DataFrame({"x": [0, 1, 2], "y": [0.1, 0.15, 0.2]})
    rec = ctx.save_figure(
        _figure(),
        fid="F11",
        slug="marketplace_trend",
        title="Доля маркетплейсов в типичном МО выросла вдвое",
        subtitle="Медиана долей по МО месяца, 2023–2024",
        alt=ALT,
        data=data,
        check="mp_share_dec24 ≥ 1,8 · mp_share_jan23",
    )
    assert (rec.png, rec.svg, rec.data_csv) == (
        "figures/F11_marketplace_trend.png",
        "figures/F11_marketplace_trend.svg",
        "data/F11_marketplace_trend.csv",
    )
    for rel in (rec.png, rec.svg, rec.data_csv):
        assert (ctx.out_dir / rel).exists()
    csv = (ctx.out_dir / rec.data_csv).read_bytes()
    assert not csv.startswith(b"\xef\xbb\xbf") and b"0.15" in csv  # без BOM, дробная точка
    assert ctx.figures == [rec] and rec.section == "e4"
    assert rec.source == style.SOURCE_SBER  # строка источника рисунка — и в записи для отчёта


@pytest.mark.parametrize(
    ("kw", "message"),
    [
        ({"alt": "Короткий"}, "альт-текст"),
        ({"title": "Очень длинный заголовок " * 5}, "заголовок"),
        ({"fid": "12"}, "номер"),
        ({"slug": "Marketplace-Trend"}, "slug"),
        ({"subtitle": ""}, "подзаголовок"),
    ],
)
def test_save_figure_rejects_bad_records(ctx, kw, message):
    args = {
        "fid": "F12",
        "slug": "marketplace_gap",
        "title": "Заголовок",
        "subtitle": "Подзаголовок",
        "alt": ALT,
        "data": pd.DataFrame({"x": [1]}),
        "check": "dq1_pp − dq5_pp ≥ 1",
    }
    args.update(kw)
    with pytest.raises(ValueError, match=message):
        ctx.save_figure(_figure(), **args)
    assert not (ctx.out_dir / "figures").exists() or not list((ctx.out_dir / "figures").glob("F12*"))


def test_save_table_csv_and_markdown(ctx):
    df = pd.DataFrame({"name": ["Арбат", "Яльчикский"], "level": [91003.0, 12500.4], "share": [0.2, 0.145]})
    rec = ctx.save_table(
        df,
        tid="T03",
        slug="level_extremes",
        title="Верх и низ по уровню",
        md_formats={"share": style.fmt_pct},
        md_labels={"name": "МО", "level": "Уровень, ₽"},
    )
    assert rec.csv == "tables/T03_level_extremes.csv"
    assert pd.read_csv(ctx.out_dir / rec.csv).shape == (2, 3)
    lines = rec.markdown.splitlines()
    assert lines[0] == "| МО | Уровень, ₽ | share |"
    assert lines[1] == "|---|---:|---:|"
    assert lines[2] == f"| Арбат | 91{THIN}003 | 20,0% |"
    assert "14,5%" in lines[3]


def test_to_markdown_defaults():
    md = to_markdown(pd.DataFrame({"a": [True, False], "b": [0.12345, np.nan], "c": ["x|y", None]}))
    rows = md.splitlines()[2:]
    assert rows[0] == "| да | 0,12 | x\\|y |" and rows[1] == "| нет | — | — |"


def test_to_markdown_empty_string_is_dash():
    md = to_markdown(pd.DataFrame({"hint": ["", "много пожилых", "  "]}))
    assert "|  |" not in md and md.splitlines()[2] == "| — |" and md.splitlines()[4] == "| — |"
    assert to_markdown(pd.DataFrame({"x": [1]}), formats={"x": lambda v: ""}).splitlines()[2] == "| — |"


def test_save_table_md_columns_keep_full_csv(ctx):
    df = pd.DataFrame({"name": ["Арбат"], "flag_pop": [1], "share": [0.2]})
    rec = ctx.save_table(
        df, tid="T02", slug="context_coverage", title="Покрытие", md_columns=["share", "name"]
    )
    assert rec.markdown.splitlines()[0] == "| share | name |"
    assert list(pd.read_csv(ctx.out_dir / rec.csv).columns) == ["name", "flag_pop", "share"]
    with pytest.raises(ValueError, match="md_columns"):
        ctx.save_table(df, tid="T04", slug="weighting", title="Веса", md_columns=["nope"])


def test_display_names_add_district_noun():
    assert display_name("Яльчикский", "mo") == "Яльчикский округ"
    assert display_name("Булунский", "mr") == "Булунский район"
    assert display_name("Эгвекинот", "go") == "Эгвекинот"
    assert display_name("Павловский Посад", "go") == "Павловский Посад"
    assert display_name(None, "mr") == "—"
    frame = pd.DataFrame(
        {
            "name": ["Яльчикский муниципальный округ", "городской округ Анадырь"],
            "name_short": ["Яльчикский", None],
            "mo_type": ["mo", "go"],
        },
        index=[5, 7],
    )
    assert display_names(frame).to_dict() == {5: "Яльчикский округ", 7: "городской округ Анадырь"}


def test_finding_checks_labels_and_placeholders(ctx):
    ctx.fact("mp_pp_median", 4.1, "pp")
    ind = pd.DataFrame({"territory_id": [1, 2], "basket_resid_pc1": [0.1, -0.2]})
    with pytest.raises(ValueError, match="нет подписей"):
        ctx.finding(title="Корзина", summary_md="", indicators=ind)
    with pytest.raises(ValueError, match="неизвестные факты"):
        ctx.finding(title="Корзина", summary_md="Прирост {{e4.mp_pp_median}}, {{e4.nope}}")
    finding = ctx.finding(
        title="Корзина и маркетплейсы",
        summary_md="Прирост {{e4.mp_pp_median}}; уровень — {{e2.level_median_2024}}",
        indicators=ind,
        indicator_labels={"basket_resid_pc1": "Очищенная PC1 корзины"},
        caveats=["номинал"],
    )
    assert isinstance(finding, Finding) and finding.facts["e4.mp_pp_median"].text.startswith("+4,1")


def test_finding_round_trip(ctx, tmp_path):
    ctx.fact("n_basket", 1932, "int", note="МО с 12 месяцами обоих лет")
    ctx.headline("Заголовок", False, "F10: ρ = −0,2")
    ind = pd.DataFrame({"territory_id": np.array([3, 1], dtype="int32"), "basket_resid_pc1": [0.5, np.nan]})
    finding = ctx.finding(
        title="Корзина",
        summary_md="{{e4.n_basket}} МО",
        indicators=ind,
        indicator_labels={"basket_resid_pc1": "PC1"},
    )
    path = save_finding(finding, tmp_path / "sections")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["facts"]["e4.n_basket"] == {
        "kind": "int",
        "note": "МО с 12 месяцами обоих лет",
        "text": "1932",
        "value": 1932,
    }
    back = load_finding(path)
    assert back.facts == finding.facts and back.headline_errors == finding.headline_errors
    pd.testing.assert_frame_equal(back.indicators, finding.indicators)


def test_load_finding_accepts_figures_without_source(ctx, tmp_path):
    rec = ctx.save_figure(
        _figure(),
        fid="F13",
        slug="growth_divergence",
        title="Уровни трат расходятся",
        subtitle="Номинал",
        alt=ALT,
        data=pd.DataFrame({"x": [1]}),
        check="rho_growth_level ≥ 0,2",
        source=style.join_sources(style.SOURCE_SBER, style.SOURCE_ROSSTAT),
    )
    assert rec.source.startswith(style.SOURCE_SBER) and style.SOURCE_ROSSTAT in rec.source
    path = save_finding(ctx.finding(title="Рост", summary_md=""), tmp_path / "sections")
    payload = json.loads(path.read_text(encoding="utf-8"))
    for fig in payload["figures"]:
        fig.pop("source")  # JSON до появления поля
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    assert load_finding(path).figures[0].source == ""
