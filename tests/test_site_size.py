"""Порция 6f лендинга (``docs/landing_spec.md``, §4.13): два совета по размеру МО и флаг «крупный» в карточке.

Числа — только из ``size_check.json`` и ``by_type.json`` этапа usefulness (здесь — синтетика с известным
ответом); строка исхода проверки по типам — свой текст ``site.build.texts.size.by_type`` с полями
``runs.main``, а не ``main_text`` (тот смешивает три новых показателя с розницей). Вход необязательный
(нет файлов — советов нет), устаревший — код 1. Плюс правка лицензии 5-НДФЛ."""

from __future__ import annotations

import copy
import json
import logging
import os
import re
from pathlib import Path

import pandas as pd
import pytest
from fixtures.site.synth_inputs import CITY, page_types

from munnet import landing, site_chapters_tail, site_findings, site_size
from munnet.config import load_config
from munnet.contracts import MissingInputError, QCError

CFG = load_config("configs/default.yaml")
TX = CFG["site"]["build"]["texts"]
SX = TX["size"]
NAMES = {"1": "Тип один", "2": "Тип два", "3": "Тип три", "4": "Тип четыре"}
SHARES = [0.5816, 0.5685, 0.5554, 0.5491, 0.5066]
CI = [(0.5459, 0.6231), (0.5341, 0.5998), (0.5134, 0.5911), (0.5058, 0.5940), (0.4601, 0.5425)]
BOUNDS = [11725.0, 18115.3, 28096.2, 53546.3]


def _sc(delta=-0.0526):
    return {
        "status": "done",
        "kind": "posthoc_exploration",
        "n_base": 1542,
        "quintile_bounds": BOUNDS,
        "quintiles": [
            {"quintile": i + 1, "n_base": 308, "share": s, "share_ci": list(c)}
            for i, (s, c) in enumerate(zip(SHARES, CI, strict=True))
        ],
        "crosstab_types_large": {"types_hi_large": 235, "types_hi_rest": 203, "types_lo_large": 74},
        "by_mo": {"n_large": 2},
        "qc": {"by_type_main_delta": delta},
    }


def _bt(verdicts=None):
    v = verdicts or {"main": "confirmed", "variant:a": "confirmed", "seed:1": "confirmed"}
    return {
        "status": "done",
        "groups": {"high": [3, 4], "low": [1, 2]},
        "verdicts": v,
        "main_text": {"text": "В среднем по трём показателям … и по рознице, где гипотеза родилась"},
        "runs": {
            "main": {
                "verdict": v["main"],
                "delta": -0.0526,
                "delta_ci": [-0.0823, -0.0186],
                "delta_ci_adj": [-0.0839, -0.0186],
                "share_hi": 0.5131,
                "share_lo": 0.5657,
            }
        },
    }


def test_size_texts_numbers_from_files():
    """Диапазон нижних четырёх групп, порог и доля верхней — из size_check; исход по типам — из runs.main,
    интервал — delta_ci_adj; слов main_text (розница) на странице нет."""
    st = site_size.size_texts(SX, _bt(), _sc(), NAMES)
    assert st is not None
    small = st["advice_small"].replace("⁠", "")
    assert "55–58%" in small and "до 53,5 тыс. жителей" in small
    assert "(50,7%)" in st["advice_large"] and "от 53,5 тыс." in st["advice_large"]
    bt = st["by_type"]
    assert "во всех 3 прогонах" in bt and "51,3%" in bt and "56,6%" in bt and "5,3 процентного пункта" in bt
    assert "от 1,9 до 8,4" in bt and "76%" in bt and "«Тип три» и «Тип четыре»" in bt
    assert "рознице" not in bt and "{" not in bt
    assert [r["label"] for r in st["rows"]][0] == "до 11,7 тыс. жителей"
    assert [r["large"] for r in st["rows"]] == [False] * 4 + [True]
    assert "похожие лучше" not in " ".join(site_size.strings(st)).lower()


def test_size_texts_none_unless_all_confirmed():
    """Текст написан под «подтверждено во всех прогонах»: иначе советов по размеру нет."""
    assert site_size.size_texts(SX, _bt({"main": "confirmed", "seed:1": "not"}), _sc(), NAMES) is None
    assert site_size.size_texts(SX, _bt({"main": "not"}), _sc(), NAMES) is None


def test_size_svg_bars_from_zero_with_reference():
    """Пять полос от нуля, верхняя группа — акцентом, пунктир 50%, интервалы; вариант для телефона."""
    st = site_size.size_texts(SX, _bt(), _sc(), NAMES)
    for w in (440, 320):
        svg = site_size.size_svg(st, w)
        rects = re.findall(r'<rect x="([\d.]+)"[^>]*fill="(#\w+)"', svg)
        assert len(rects) == 5 and all(float(x) == 0 for x, _ in rects)
        assert [c for _, c in rects].count(site_size.ACCENT) == 1 and rects[-1][1] == site_size.ACCENT
        assert 'stroke-dasharray="3 3"' in svg and svg.count("<path") == 5
        assert "50,7%" in svg and 'role="img"' in svg and "aria-label=" in svg


def test_check_large_and_rows():
    by_mo = pd.DataFrame(
        {"territory_id": [1, 2, 3], "pop_avg_2023": [60000.0, 10000.0, 54000.0], "large": [True, False, True]}
    )
    assert site_size.large_ids(by_mo) == {1, 3}
    assert site_size.check_large(by_mo, _sc()) == []
    bad = by_mo.assign(large=[True, False, False])
    assert site_size.check_large(bad, _sc())
    st = site_size.size_texts(SX, _bt(), _sc(), NAMES)
    site_size.check_rows(st)
    st["rows"][0]["lo"] = 0.9
    with pytest.raises(QCError):
        site_size.check_rows(st)


def test_findings_use_column_with_size():
    """Колонка «Что с этим делать»: два совета, график по размеру, исход по типам; прежнее о рознице —
    под «Где нашли совет»."""
    from test_site_findings import _story
    from test_site_useful import _ft, _uf

    ft = _ft(_uf())
    ft["use"]["size"] = site_size.size_texts(SX, _bt(), _sc(), NAMES)
    h = site_findings.findings_html(ft, _story(), [2, 1, 3, 4], landing._t, "Источник.")
    use = h[h.index('id="fd-use"') :]
    assert use.count('class="fd-adv') == 2 and "size-svg" in use and 'class="fd-bytype"' in use
    assert use.index("size-svg") < use.index('<details class="more fd-more">') < use.index('class="fd-share"')
    assert h.count('<svg class="vw fd-svg size-svg') == 1 and h.count('<svg class="vp fd-svg size-svg') == 1
    texts = site_findings.strings(ft)
    v = {"T1_ladder_external": "partial_overall", "T2_direction": "partial", "T3_reliable_placebo": "not",
         "T7_utility": "not", "T7_type_gain": "neutral", "one_in_ten": False, "caveat": True}  # fmt: skip
    cfg = landing.Config(CFG.data, Path("x"))
    card = site_size.card_note(SX, ft["use"]["size"])
    assert landing.lint_texts(cfg, {"verdicts": v}, texts + [card]) == []
    assert "53,5 тыс." in card


def test_ndfl_license_not_cc_by():
    """5-НДФЛ: на странице набора лицензия не указана — без «CC BY 4.0» (как report.md, приложение Б)."""
    nd = [s for s in site_chapters_tail.SOURCES if "НДФЛ" in s[0]]
    assert len(nd) == 1 and "не указана" in nd[0][2] and "CC BY" not in nd[0][2]
    bd = [s for s in site_chapters_tail.SOURCES if "муниципальных образований" in s[0]]
    assert bd[0][2] == "CC BY 4.0"
    lic = {x["source"]: x["license"] for x in landing.LICENSES}
    assert "не указана" in lic["ФНС, 5-НДФЛ в обработке «Если быть точным»"]


# --- этап site целиком (синтетика) -----------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _no_layout(monkeypatch):
    empty = pd.DataFrame({"territory_id": [], "nx": [], "ny": []})
    monkeypatch.setattr(landing, "similarity", lambda cfg: ({"chosen": None, "preserved": {}}, empty))


def _write_size(d: Path, large: set[int], sc: dict | None = None, bt: dict | None = None) -> None:
    ids = sorted(i for i in page_types() if i != CITY)  # узлов-городов в size_by_mo нет
    pop = [60000.0 if i in large else 20000.0 for i in ids]
    pd.DataFrame(
        {"territory_id": ids, "type": [page_types()[i] for i in ids], "pop_avg_2023": pop,
         "quintile": [5 if i in large else 2 for i in ids], "large": [i in large for i in ids],
         "in_t7_common": True}
    ).to_csv(d / "size_by_mo.csv", index=False)  # fmt: skip
    s = copy.deepcopy(sc or _sc())
    s["by_mo"]["n_large"] = len(large)
    (d / "size_check.json").write_text(json.dumps(s, ensure_ascii=False), encoding="utf-8")
    (d / "by_type.json").write_text(json.dumps(bt or _bt(), ensure_ascii=False), encoding="utf-8")


def _mo(tmp_path):
    mo = json.loads((tmp_path / "site" / "data" / "mo.json").read_text(encoding="utf-8"))
    return {i: {k: v[j] for k, v in mo.items()} for j, i in enumerate(mo["id"])}


def test_site_large_flag_in_mo_and_card(tmp_path):
    from test_site_useful import _setup, _write_useful

    cfg, idir = _setup(tmp_path)
    d = _write_useful(tmp_path / "outputs", idir)
    big = min(i for i in page_types() if i != CITY)
    _write_size(d, {big})
    story = landing.run(cfg)
    rows = _mo(tmp_path)
    assert rows[big]["lg"] == 1
    assert all(r["lg"] is None for i, r in rows.items() if i != big)  # и город, и его районы — без флага
    assert "53,5 тыс." in story["card"]["large_note"]
    html = (tmp_path / "site" / "index.html").read_text(encoding="utf-8")
    assert '"works"' not in html and "rule_by_mo" not in html


def test_site_without_size_files_warns(tmp_path, caplog):
    from test_site_useful import _setup, _write_useful

    cfg, idir = _setup(tmp_path)
    _write_useful(tmp_path / "outputs", idir)
    with caplog.at_level(logging.WARNING):
        story = landing.run(cfg)
    assert any("по размеру" in r.getMessage() for r in caplog.records)
    assert "large_note" not in story["card"]
    assert all(r["lg"] is None for r in _mo(tmp_path).values())


def test_site_size_stale_is_code1(tmp_path):
    from test_site_useful import _setup, _write_useful

    cfg, idir = _setup(tmp_path)
    d = _write_useful(tmp_path / "outputs", idir)
    _write_size(d, set())
    old = (d / "facts.json").stat().st_mtime - 100
    os.utime(d / "size_check.json", (old, old))
    with pytest.raises(MissingInputError, match="старше"):
        landing.run(cfg)
    cfg, idir = _setup(tmp_path)
    d = _write_useful(tmp_path / "outputs", idir)
    _write_size(d, set(), sc=_sc(delta=-0.01))  # посчитано на другом исходе by_type_test
    with pytest.raises(MissingInputError, match="by_type"):
        landing.run(cfg)
