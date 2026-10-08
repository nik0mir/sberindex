"""Главы 6, 7, 9 и подвал лендинга (``src/munnet/site_chapters_tail.py``, §3.8, §3.10, §3.11
``docs/landing_spec.md``): набор B показан всегда, строки по возрастанию ошибки, разности — от набора
продукта, пример сверяется с facts, карты R1 отмечают ровно узлы с другим типом, ограничения без чисел
скрываются, интерфейс без запрещённых слов при любых вердиктах.
Без сети и без data/: входы — маленькие словари."""

from __future__ import annotations

import html
import re
from pathlib import Path

import pandas as pd
import pytest
from fixtures.site.synth_facts import all_combos, combo_levels

from munnet import landing
from munnet import site_chapters_tail as T
from munnet.config import load_config
from munnet.contracts import QCError
from munnet.site_headlines import HeadlineChecker

CFG = load_config("configs/default.yaml")
ESC = html.escape


def story() -> dict:
    return {
        "view": {
            "type_colors": {"1": "#501d42", "2": "#b4561d"},
            "shapes": {"1": "●", "2": "▲"},
            "legend_order": [1, 2],
            "product_set": "D",
            "set_name": "сопоставимые территории",
            "unstable_label": "неустойчиво к правилу рёбер",
        },
        "names": {"final": {"1": "Тип 1", "2": "Тип 2"}},
        "card": {"shifted": "Ячейки сдвинуты", "lines": "Сходство трат"},
        "roles": {
            "ministry": "Министерству: роль.",
            "analysts": None,
            "business": "Бизнесу: роль.",
            "business_label": "описание, не проверка",
        },  # fmt: skip
        "chapters": {
            "comparable": {
                "title": "Заголовок T7",
                "lead": "Заголовок type_gain",
                "text": "Текст T7.",
                "same_period": "Сверка за тот же период",
            },  # fmt: skip
            "limits": {
                "title": "Заголовок главы 7",
                "lead": ["Охват"],
                "text": "Текст T6.",
                "items": {"visitors": "Приезжие не видны", "untyped": "Без типа"},
            },  # fmt: skip
            "method": {
                "title": "Как сделано",
                "command": "uv run --frozen python -m munnet all",
                "prereg": [{"commit": "abc61071df1a", "text": "Код вслепую"}],
                "links": {"repo": "https://github.com/x/y"},
                "numbers": {
                    "n_mo": "10",
                    "n_nodes": "8",
                    "n_edges": "20",
                    "n_types": "2 типа",
                    "method": "гибрид",
                    "n_cand": "3",
                    "n_windows": "13",
                },  # fmt: skip
            },
            "sources": {"title": "Источники и лицензии"},
        },
    }


def checks() -> dict:
    return {
        "t7": {
            "product": "D",
            "k": 10,
            "n_common": 1542,
            "median_error": {"A": 0.048, "B": 0.036, "C": 0.045, "D": 0.050},
            "median_error_abs": {"A": 0.041, "B": 0.036, "C": 0.038, "D": 0.042},
            "diffs": {
                "B-D": [-0.013, [-0.017, -0.010]],
                "B-A": [-0.012, [-0.015, -0.009]],
                "C-D": [-0.005, [-0.007, -0.002]],
                "D-A": [0.001, [0.0, 0.003]],
            },
            "example": {
                "territory_id": 1,
                "errors": {"A": 0.044, "B": 0.058, "C": 0.049, "D": 0.050},
                "members_p": [[2, 900], [3, 1500]],
                "members_b": [[4, 40]],
            },
        },
        "r1": {
            "variants": [
                {"variant": "variant:graph_basket_cos", "n": 4, "same": 2, "ari": 0.22, "diff_ids": [2, 3]},
                {"variant": "variant:no_level", "n": 4, "same": 4, "ari": 1.0, "diff_ids": []},
            ],
            "circularity": "Порог на ARI не ставится.",
        },
    }


MO = {i: {"id": i, "n": f"МО {i}", "ns": f"МО {i}", "r": "Область", "t": 1} for i in range(1, 6)}
GEO = {"w": 100.0, "h": 50.0, "xy": {i: (10.0 * i, 20.0) for i in range(1, 6)}}
METHODS = {
    "metrics": ["sw", "avu"],
    "labels": {"sw": "SW", "avu": "AVU"},
    "better": {"sw": "max", "avu": "min"},
    "rows": [
        {"candidate": "h4", "method": "hybrid", "family": "attributed", "k": 4, "feasible": True,
         "winner": True, "final": True, "v": {"sw": 0.007, "avu": 0.6}, "z": {"sw": 2.2, "avu": -85.6}},
        {"candidate": "l3", "method": "leiden", "family": "graph", "k": 3, "feasible": True,
         "winner": True, "final": False, "v": {"sw": 0.004, "avu": 0.667}, "z": {"sw": 2.6, "avu": None}},
        {"candidate": "w2", "method": "ward", "family": "features", "k": 2, "feasible": False,
         "winner": False, "final": False, "v": {"sw": 0.1, "avu": 1.0}, "z": {"sw": 1.0, "avu": None}},
    ],
}  # fmt: skip
META = {"seed": 42, "sha": "d2daa30dd445", "facts_sha256": "e86de43ec8bd00"}


def page() -> str:
    return T.chapters_html(story(), checks(), MO, GEO, METHODS, META, ESC)


def section(h: str, cid: str) -> str:
    m = re.search(rf'<section class="chapter[^"]*" id="{cid}".*?</section>', h, re.S)
    assert m, cid
    return m.group(0)


def test_every_chapter_has_title_source_and_alt():
    h = page()
    for cid, title in (
        ("comparable", "Заголовок T7"),
        ("limits", "Заголовок главы 7"),
        ("method", "Как сделано"),
    ):
        sec = section(h, cid)
        assert f'<h2 id="{cid}-title">{title}</h2>' in sec
        assert '<figcaption class="source">' in sec and "CC BY-SA 4.0" in sec
        assert all('role="img" aria-label="' in s for s in re.findall(r'<svg class="ch-svg[^>]*>', sec))
    assert '<details class="alt">' in section(h, "comparable") + section(h, "limits")
    assert h.count('id="cells-base"') == 1  # общий слой ячеек — один раз на страницу
    assert "{" not in re.sub(r"<style.*?</style>", "", h)


def test_comparable_rows_sorted_b_always_and_verbatim_text():
    sec = section(page(), "comparable")
    comp = re.search(r'<svg class="vw ch-svg comp dots".*?</svg>', sec, re.S).group(0)
    order = re.findall(r">([ABCD]) · ", comp)
    assert order == ["B", "C", "A", "D"]  # по возрастанию медианной ошибки; B — всегда
    assert re.search(r'<details class="how"><summary>Как проверяли</summary><p>Текст T7\.</p>', sec)
    assert "сопоставимые территории: похожие" in comp  # набор продукта назван {set_name}


def test_diff_rows_follow_product_set():
    rows = T.diff_rows(checks()["t7"])
    assert [(a, b) for a, b, _ in rows] == [("B", "D"), ("C", "D"), ("D", "A")]
    t7a = checks()["t7"] | {"product": "A", "diffs": {"B-A": [-0.01, [-0.02, 0.0]]}}
    assert [(a, b) for a, b, _ in T.diff_rows(t7a)] == [("B", "A")]


def test_example_map_lines_and_card_link():
    sec = section(page(), "comparable")
    ex = re.search(r'<svg class="ch-svg mini-map ex-map".*?</svg>', sec, re.S).group(0)
    assert ex.count("<line ") == 3  # 2 из набора продукта + 1 сосед
    assert '<a class="open-card" href="#mo=1"' in sec


def test_r1_maps_mark_exactly_changed_nodes():
    sec = section(page(), "limits")
    maps = re.findall(r'<svg class="ch-svg mini-map r1-map".*?</svg>', sec, re.S)
    assert len(maps) == 2
    first = re.search(r'<path d="([^"]*)"', maps[0]).group(1)
    assert first.count("h0") == 2  # узлы 2 и 3
    assert re.search(r'<path d=""', maps[1])  # в варианте без уровня тип у всех тот же
    # доля — с одним знаком, как в блоке «Что устояло» (порция 6a)
    assert "тот же тип у 50,0% муниципалитетов с типом (2 из 4)" in sec and "0,22" in sec
    assert '<p class="kicker">Чего данные не показывают</p><h2' in sec


def test_limits_items_drop_rows_without_numbers_and_genitive():
    tpl = {
        "visitors": "Приезжие",
        "untyped": "У {n_untyped_mo} нет типа",
        "workplace": "у {n_recip_gt3_mo} …",
    }
    scope = {"n_untyped": 169, "pop_median_partial": 18397.75, "pop_median_full": 21104}
    out = landing.limits_items(tpl, {"n_recip_gt3": "41"}, scope, None)
    assert out["untyped"] == "У 169 муниципалитетов нет типа"
    assert out["workplace"] == "у 41 муниципалитета …"
    assert "regions" not in out
    assert "workplace" not in landing.limits_items(tpl, {}, scope, None)


def test_methods_table_final_marked_winners_first_and_all_rows():
    sec = section(page(), "method")
    best = re.search(r'<table class="mtable" id="mtable-best".*?</table>', sec, re.S).group(0)
    assert best.count("<tr") == 1 + 2 and 'class="is-final"' in best and "итог" in best
    full = re.search(r'<table class="mtable" id="mtable-all".*?</table>', sec, re.S).group(0)
    assert full.count("<tr") == 1 + 3
    assert "Методы без допустимых кандидатов (1)" in sec
    assert 'data-v="0.007"' in best and "z 2,2" in best
    assert sec.count('class="mstep"') == 5 and "https://github.com/x/y/commit/abc61071df1a" in sec


def test_sources_licenses_and_downloads():
    h = T.sources_html(story(), {"mo.csv": 546_781, "types.csv": 762}, ESC)
    for lic in ("CC BY-SA 4.0", "CC BY 4.0", "ODbL", "MIT"):
        assert lic in h
    assert 'href="data/download/mo.csv"' in h and "534 КБ" in h
    assert "flows.csv" not in h  # нет в выгрузках — нет ссылки


def test_tail_ui_has_no_forbidden_words_for_any_verdict():
    hc = HeadlineChecker(CFG.data)
    levels = combo_levels(CFG.data)
    keys = tuple(
        k
        for k in ("T1_ladder_external", "T2_direction", "T3_reliable_placebo", "T7_type_gain", "one_in_ten")
        if k in levels
    )
    texts = T.ui_strings()
    for combo in all_combos(CFG.data, keys):
        assert hc.text_violations(texts, combo) == [], combo
    # отрицательный контроль: та же проверка ловит «того же типа» без вердикта adds
    combo = next(all_combos(CFG.data, keys)) | {"T7_type_gain": "neutral"}
    assert hc.text_violations(["набор того же типа"], combo)


class _D:
    """Заглушка SiteData: только выгрузки этапа 5 и outputs."""

    def __init__(self, tmp: Path, frames: dict[str, pd.DataFrame], typed: list[int] | None = None):
        self.frames, self.tmp = frames, tmp
        self.cfg = CFG
        ids = typed if typed is not None else [1, 2, 3]
        self.node_type = pd.Series([1] * len(ids), index=ids)  # узлы с типом основного расчёта

    def opt(self, name: str) -> pd.DataFrame | None:
        return self.frames.get(name)

    def outputs(self, rel: str) -> Path:
        return self.tmp / rel


def _t7_frames(err_d: float) -> dict[str, pd.DataFrame]:
    err = pd.DataFrame(
        [{"territory_id": 1, **{f"err_{s}": 0.05 for s in "ABC"}, "err_D": err_d,
          **{f"err_abs_{s}": 0.04 for s in "ABCD"}}]
    )  # fmt: skip
    nc = pd.DataFrame(
        {
            "territory_id": [1, 1, 1],
            "set": ["D", "D", "B"],
            "product": [True, True, False],
            "rank": [1, 2, 1],
            "other_id": [2, 3, 4],
            "km": [900.0, 1500.0, 40.0],
        }
    )
    return {"t7_errors.csv": err, "node_comparable.csv": nc}


def test_t7_example_checked_against_facts(tmp_path):
    t7 = {"product": "D", "example": {"territory_id": 1, "error": 0.0496, "members": [2, 3]}}
    ex = landing.t7_example(_D(tmp_path, _t7_frames(0.0496)), t7)
    assert ex["members_p"] == [[2, 900.0], [3, 1500.0]] and ex["members_b"] == [[4, 40.0]]
    assert ex["errors"]["D"] == pytest.approx(0.0496)
    with pytest.raises(QCError):
        landing.t7_example(_D(tmp_path, _t7_frames(0.07)), t7)
    with pytest.raises(QCError):
        landing.t7_example(
            _D(tmp_path, _t7_frames(0.0496)), t7 | {"example": t7["example"] | {"members": [3]}}
        )


def test_r1_variants_counts_and_ari_from_cluster(tmp_path):
    nr = pd.DataFrame(
        {
            "territory_id": [1, 2, 3, 1, 2, 3, 9],
            "variant": ["variant:a"] * 3 + ["seed:1"] * 3 + ["variant:a"],
            "kind": ["variant"] * 3 + ["seed"] * 3 + ["variant"],
            "same": [True, False, False, True, True, True, False],
        }
    )  # 9 — район столицы: в варианте «районы отдельно» он есть, но типа основного расчёта у него нет
    (tmp_path / "cluster").mkdir()
    pd.DataFrame({"variant": ["a"], "ari_same_candidate_vs_main": [0.25]}).to_csv(
        tmp_path / "cluster" / "variants.csv", index=False
    )
    out = landing.r1_variants(_D(tmp_path, {"node_r1.csv": nr}))
    # порция 6b: доля — среди муниципалитетов с типом основного расчёта (район 9 не входит)
    assert out == [{"variant": "variant:a", "n": 3, "same": 1, "ari": 0.25, "diff_ids": [2, 3]}]


# --- порция 6c: карта-соперник T5 ---------------------------------------------------------------------------


def _rival(**kw) -> dict:
    r = {
        "partition": "sized:log_level_rel:+",
        "label": "уровень трат, по возрастанию",
        "ami": 0.3129,
        "order": "+",
        "groups": {1: 1, 2: 2, 3: 2, 4: 1},
        "types": {1: 1, 2: 1, 3: 2, 4: 2},
    }
    return r | kw


def test_rival_maps_two_maps_on_shared_layout_neutral_colors():
    """Две карты на общем слое ячеек; группы соперника — серые по порядку, не цвета типов; AMI — с запятой;
    таблица «тип × группа» — счёт тех же меток; при show_rival = false карты нет."""
    st = story()
    st["view"]["show_rival"] = True
    h = T.rival_maps(_rival(), st, GEO, ESC)
    assert h.count('<use href="#cells-base"') == 2 and h.count("<svg") == 2
    ramp = T.rival_ramp(2)
    assert all(c in h for c in ramp) and not set(ramp) & {
        c.lower() for c in st["view"]["type_colors"].values()
    }
    assert "0,31" in h and "4 группы" not in h and "2 группы" in h
    assert "1 — наименьшие значения, 2 — наибольшие" in h
    alt = re.search(r'<details class="alt">.*?</details>', h, re.S).group(0)
    assert [re.findall(r"<td>(\d+)</td>", tr) for tr in re.findall(r"<tr>(.*?)</tr>", alt)[1:]] == [
        ["1", "1"],
        ["1", "1"],
    ]
    st["view"]["show_rival"] = False
    assert T.rival_maps(_rival(), st, GEO, ESC) == ""


def test_rival_ramp_monotone_lightness():
    """Последовательная шкала: каждая следующая группа темнее предыдущей (видно и в оттенках серого)."""

    def lum(c: str) -> float:
        r, g, b = (int(c[i : i + 2], 16) for i in (1, 3, 5))
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    for k in (2, 4, 10):
        ls = [lum(c) for c in T.rival_ramp(k)]
        assert all(a > b for a, b in zip(ls, ls[1:], strict=False)), k
    assert lum(T.RIVAL_BASE) > lum(T.rival_ramp(4)[0]) + 30  # ячейки без типа светлее первой группы


def test_rival_partition_must_match_max_ami(tmp_path):
    """Метки соперника берутся из node_rival.csv (T5:max_ami) только для того же деления, что первая строка
    checks.t5.ami; другое деление — код 3; без файла — карты нет."""
    from types import SimpleNamespace

    rv = pd.DataFrame(
        {
            "territory_id": [1, 2, 1],
            "partition": ["sized:log_level_rel:+", "sized:log_level_rel:+", "pop_decile"],
            "partition_label": ["x", "x", "y"],
            "sources": [
                "T5:max_ami;T1:best_rival:retail",
                "T5:max_ami;T1:best_rival:retail",
                "T5:best_rival:retail",
            ],
            "best_partition_label": [2, 1, 7],
        }
    )
    d = SimpleNamespace(opt=lambda name: rv if name == "node_rival.csv" else None)
    mo = pd.DataFrame({"id": [1, 2, 3], "role": ["territorial", "city", "untyped"], "t": [1, 2, None]})
    ck = {"t5": {"ami": [{"partition": "sized:log_level_rel:+", "label": "уровень трат", "ami": 0.31}]}}
    r = landing.rival_partition(d, ck, mo)
    assert (
        r["groups"] == {1: 2, 2: 1} and r["types"] == {1: 1, 2: 2} and r["order"] == "+" and r["ami"] == 0.31
    )
    ck["t5"]["ami"][0]["partition"] = "pop_decile"
    with pytest.raises(QCError, match="карты-соперника"):
        landing.rival_partition(d, ck, mo)
    d = SimpleNamespace(opt=lambda name: None)
    assert landing.rival_partition(d, ck, mo) is None


# --- порция 6c: таблица всех муниципалитетов --------------------------------------------------------------


def test_all_mo_table_skeleton_accessible_and_csv_without_js():
    """Каркас таблицы: caption, scope у заголовков, сортировка кнопками с aria-sort; фильтры скрыты до JS;
    без JS — ссылка на mo.csv с размером; раздел стоит между главой 7 и «Как сделано»."""
    h = T.chapter_all_mo({"mo.csv": 220_000}, ESC)
    assert '<section class="chapter wide" id="all-mo"' in h and "<caption" in h
    assert h.count('<th scope="col"') == 6 and h.count('class="sort"') == 6 and h.count("aria-sort=") == 1
    assert 'id="mo-ctl" hidden' in h and 'id="mo-wrap" hidden' in h and 'id="mo-pager" hidden' in h
    assert 'href="data/download/mo.csv"' in h and "215 КБ" in h and 'class="nojs"' in h
    assert "<tbody></tbody>" in h  # строки рисует JS, не 2192 сразу
    p = T.chapters_html(story(), checks(), MO, GEO, METHODS, META, ESC, {"mo.csv": 1000})
    assert p.index('id="limits"') < p.index('id="all-mo"') < p.index('id="method"')


def test_all_mo_js_pages_and_reads_only_mo_fields():
    """JS таблицы: страница — 50 строк; читает только поля mo.json (иначе столбец молча пустеет); ссылка
    «Таблица» есть в меню шаблона."""
    tpl = Path(landing.__file__).parent / "templates"
    js = (tpl / "landing.js").read_text(encoding="utf-8")
    part = js[js.index("const ALL = ") : js.index("function initAll")]
    assert "size: 50" in part and "slice(a, b)" in part
    used = set(re.findall(r"\b(?:r|nd)\.([a-z][a-z0-9_]*)\b", part))
    cols = set(landing.mo_json(pd.DataFrame({"id": []})))
    assert used and not used - cols, used - cols
    assert '<a href="#all-mo">' in (tpl / "landing.html").read_text(encoding="utf-8")
    assert "rival" not in cols  # карта-соперник собирается при сборке, странице метки не нужны
