"""Порция 6n лендинга (``docs/landing_spec.md``, §4.3): схема «Сеть корзин целиком» в главе «Типы».

Раскладка ``graph_fr`` того же расчёта, что правило ``site.similarity_layout``, показывается при любом
его исходе; правило, замороженные ключи и перестановка карты не меняются. Здесь: поворот без искажений,
k самых похожих, детерминизм данных схемы, доля сохранённых соседей = ``site_layout``, тексты (числа —
кодом), отсутствие осей и подписей порядка, сборка этапа на синтетике.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fixtures.site.synth_facts import all_combos, make_facts, site_config
from fixtures.site.synth_inputs import CITY, CONTROLS, bound_interpret, facts_over, make_site_inputs
from test_site_layout import SITE, _two_cliques

from munnet import landing, site_netmap
from munnet import site_layout as SL
from munnet.config import Config, load_config
from munnet.contracts import QCError

CFG = load_config("configs/default.yaml")
TX = CFG["site"]["build"]["texts"]["netmap"]


def _res(seed: int = 42):
    ids, src, tgt = _two_cliques()
    spec = SL.LayoutSpec.from_config(SITE)
    Z = np.random.default_rng(3).normal(size=(len(ids), 4))
    res = SL.similarity_layout(spec, ids, src, tgt, Z, seed=seed)
    res.info["seed"] = seed
    w = np.linspace(1.0, 0.5, len(src))  # разные веса: порядок соседей определён
    return res, pd.DataFrame({"source": src, "target": tgt, "weight": w})


# --- геометрия и соседи -------------------------------------------------------------------------------------


def test_orient_is_rotation_long_axis_horizontal():
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(200, 2)) * [1.0, 0.2]
    a = np.deg2rad(37)
    c = pts @ np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]).T + [5, -3]
    o = site_netmap.orient(c)
    # расстояния те же (поворот, без отражения и растяжения)
    d0 = np.linalg.norm(c[:, None] - c[None], axis=-1)
    d1 = np.linalg.norm(o[:, None] - o[None], axis=-1)
    np.testing.assert_allclose(d0, d1, atol=1e-9)
    ext = o.max(axis=0) - o.min(axis=0)
    assert ext[0] > 3 * ext[1]  # длинная ось — по горизонтали
    # собственный поворот: ориентация треугольника не меняется
    tri = lambda p: (p[1, 0] - p[0, 0]) * (p[2, 1] - p[0, 1]) - (p[1, 1] - p[0, 1]) * (p[2, 0] - p[0, 0])  # noqa: E731
    assert np.sign(tri(c[:3])) == np.sign(tri(o[:3]))
    # знак — по правилу, а не случайно: повторный расчёт и сдвиг входа дают то же
    np.testing.assert_allclose(site_netmap.orient(c + 100), o, atol=1e-9)


def test_to_grid_common_scale():
    xy, h = site_netmap.to_grid(np.array([[0.0, 0.0], [4.0, 1.0], [2.0, 0.5]]))
    assert xy[:, 0].min() == 0 and xy[:, 0].max() == 1000
    assert h == 250 and xy[:, 1].max() - xy[:, 1].min() == 250  # общий масштаб осей
    assert xy[0, 1] > xy[1, 1]  # y вниз, как на canvas


def test_top_neighbors_by_weight_then_id():
    ids = np.array([10, 20, 30, 40])
    e = pd.DataFrame({"source": [10, 10, 10, 20], "target": [20, 30, 40, 30], "weight": [0.5, 0.9, 0.5, 0.7]})
    nb = site_netmap.top_neighbors(ids, e, 2)
    assert nb.tolist() == [
        [2, 1],
        [2, 0],
        [0, 1],
        [0, -1],
    ]  # 10: 30 (0,9), затем 20 и 40 по 0,5 -> меньший номер
    with pytest.raises(ValueError):
        site_netmap.top_neighbors(ids, pd.DataFrame({"source": [10], "target": [99], "weight": [1.0]}), 2)


def test_encode_roundtrip():
    v = [0, 1, 63, 64, 1000, 1775, 4095, -1]
    s = site_netmap.encode(v)
    assert len(s) == 2 * len(v) and s.endswith("~~")
    assert site_netmap.decode(s) == v
    with pytest.raises(ValueError):
        site_netmap.encode([4096])


# --- данные схемы -------------------------------------------------------------------------------------------


def test_build_deterministic_and_preserved_from_layout():
    r1, e = _res()
    r2, _ = _res()
    a = site_netmap.build(r1, e, 3)
    b = site_netmap.build(r2, e, 3)
    assert landing.dumps(a) == landing.dumps(b)  # побайтно одинаково при том же seed
    assert a["preserved"] == round(r1.preserved["graph_fr"], 6)  # та же доля, что в выборе раскладки
    assert a["random"] == round(r1.random_preserved, 6) and a["min_preserved"] == 0.30
    assert a["chosen"] == r1.chosen and a["seed"] == 42 and a["n_nodes"] == 16
    x, y = site_netmap.decode(a["x"]), site_netmap.decode(a["y"])
    assert len(x) == len(y) == 16 and min(x) == 0 and max(x) == 1000 and max(y) <= a["h"] <= 1000
    nb = np.array(site_netmap.decode(a["nb"])).reshape(16, 3)
    assert (nb >= 0).all()  # в кликах у каждого не меньше трёх соседей
    # соседи — из своей клики (кроме двух узлов моста)
    assert all(set(nb[i]) <= set(range(8)) for i in range(1, 8))
    assert site_netmap.check(a, r1.ids.tolist()) == []
    assert site_netmap.check(a, r1.ids.tolist()[1:])  # узла схемы нет среди МО — ошибка


def test_schema_shown_even_when_rule_says_no():
    """Правило перестановки не прошло (chosen = None) — данные схемы всё равно есть, nxy пустая."""
    ids, src, tgt = _two_cliques()
    spec = SL.LayoutSpec.from_config(SITE)
    spec = SL.LayoutSpec(**{**spec.__dict__, "min_preserved": 1.01})
    r = SL.similarity_layout(spec, ids, src, tgt, np.random.default_rng(0).normal(size=(len(ids), 3)), seed=1)
    e = pd.DataFrame({"source": src, "target": tgt, "weight": 1.0})
    assert r.chosen is None and r.nxy().empty
    d = site_netmap.build(r, e, 3)
    assert d["chosen"] is None and d["min_preserved"] == 1.01 and len(d["ids"]) == 16


# --- тексты и HTML ------------------------------------------------------------------------------------------


def test_texts_numbers_from_data():
    r, e = _res()
    d = site_netmap.build(r, e, 3)
    t = site_netmap.texts(TX, d)
    assert "{" not in "".join(t.values())
    pct = landing.style.fmt_pct(d["preserved"])
    assert pct in t["caption"] and landing.style.fmt_pct(d["random"]) in t["caption"]
    assert "в среднем хотя бы 30% из 3 ближайших" in t["caption"]  # min_pct — из min_preserved
    assert "с 3 самыми похожими" in t["hint"]
    assert site_netmap.texts(None, d) is None and site_netmap.texts(TX, None) is None


def test_real_texts_numbers_and_style():
    d = {"k": 10, "min_preserved": 0.3, "preserved": 0.298874, "random": 0.007827, "n_nodes": 1776,
         "n_edges": 12418, "niter": 1000, "seed": 42}  # fmt: skip
    t = site_netmap.texts(TX, d)
    assert (
        "в среднем хотя бы 30% из 10 ближайших к муниципалитету точек — его соседи по сети корзин; "
        "здесь таких 29,9%, у случайной раскладки было бы 0,8%" in t["caption"]
    )
    assert "отдельных островов на ней нет" in t["lead"] and "обособленных" not in t["lead"]
    assert "для дополнительной иллюстрации, в качестве исключения" in t["caption"]
    assert "с 10 самыми похожими" in t["hint"] and "12 418 связей" in t["how"]
    low = " ".join(t.values()).lower()
    for w in ("отступлен", "нарушен", "отход от правил", "только примета", "лишь", "это только разведка",
              "ступен", "лестниц"):  # fmt: skip
        assert w not in low, w


def test_texts_pass_site_lint_all_t1_verdicts():
    """Тексты блока проходят линт текста сайта при любом исходе (в т. ч. T1 partial_overall, T3 not)."""
    d = {"k": 10, "min_preserved": 0.3, "preserved": 0.3, "random": 0.01, "n_nodes": 1776, "n_edges": 12418,
         "niter": 1000, "seed": 42}  # fmt: skip
    t = site_netmap.texts(TX, d)
    for combo in all_combos(CFG.data):
        story = {"verdicts": combo}
        assert landing.lint_texts(CFG, story, site_netmap.strings(t)) == [], combo


def _html(t):
    types = [{"t": 1, "size": 806}, {"t": 2, "size": 473}, {"t": 3, "size": 394}, {"t": 4, "size": 103}]
    names = {
        "1": "Города, меньше транспорта",
        "2": "Сельские, меньше общепита",
        "3": "Города, меньше продуктов",
        "4": "Крупные города, меньше продуктов",
    }
    return site_netmap.html_block(
        t, {}, types, names, {"1": "●", "2": "▲", "3": "■", "4": "◆"}, landing._t, "Ист."
    )


def test_html_block_no_axes_or_order_marks():
    d = {"k": 10, "min_preserved": 0.3, "preserved": 0.298874, "random": 0.007827, "n_nodes": 1776,
         "n_edges": 12418, "niter": 1000, "seed": 42}  # fmt: skip
    h = _html(site_netmap.texts(TX, d))
    assert 'id="netmap"' in h and 'class="net-canvas" role="img" aria-label="' in h
    assert h.count('class="net-key"') == 4 and 'aria-pressed="false"' in h  # легенда — кнопки с клавиатуры
    assert 'href="#all-mo"' in h  # текстовая альтернатива — таблица всех МО
    assert "1 806" not in h and ">806<" in h
    # на самой схеме и в легенде — ни осей, ни стрелок, ни подписей порядка (T1 не confirmed)
    legend = re.search(r'<div class="net-side">(.*?)</div></div>', h, re.S)[1]
    for w in ("→", "←", "↑", "↓", " от ", "ось", "ступен", "лестниц", "по возрастанию"):
        assert w not in legend, w
    assert "<svg" not in h and "axis" not in h


def test_js_draws_without_axes_and_reads_netmap():
    js = (Path(landing.__file__).parent / "templates" / "landing.js").read_text(encoding="utf-8")
    net = js[js.index("function initNet") : js.index("// --- загрузка")]
    assert "fillText" in net and net.count("fillText") == 1  # единственная подпись — название выбранного МО
    assert "prefers-reduced-motion: reduce" in net and "IntersectionObserver" in net
    assert 'loadData("netmap")' in js and "initNet(netmap)" in js
    for w in ("ступен", "лестниц", "axis"):
        assert w not in net


# --- этап site целиком (синтетика) --------------------------------------------------------------------------


def _setup(tmp_path):
    make_site_inputs(tmp_path)
    d = site_config(CFG.data, tmp_path)
    d["site"]["build"]["controls"] = dict(CONTROLS)
    cfg = Config(d, tmp_path / "cfg.yaml")
    combo = next(all_combos(CFG.data)) | {
        "T3_reliable_placebo": "confirmed",
        "T1_ladder_external": "partial_overall",
    }
    facts = make_facts(CFG.data, combo, **facts_over())
    facts.pop("synthetic")
    bound_interpret(tmp_path, tmp_path / "outputs", facts)
    return cfg


def _fake_similarity(monkeypatch, ids):
    def fake(cfg):
        e = site_netmap.main_edges_weighted(cfg.dir("processed"))
        spec = SL.LayoutSpec.from_config(cfg["site"])
        Z = np.random.default_rng(0).normal(size=(len(ids), 3))
        res = SL.similarity_layout(spec, np.array(ids), e["source"].to_numpy(), e["target"].to_numpy(), Z, 42)
        res.info["seed"] = 42
        empty = pd.DataFrame({"territory_id": [], "nx": [], "ny": []})
        return res.summary(), empty, site_netmap.build(res, e, spec.k)

    monkeypatch.setattr(landing, "similarity", fake)


def test_stage_writes_netmap_and_block(tmp_path, monkeypatch):
    cfg = _setup(tmp_path)
    _fake_similarity(monkeypatch, [1, 2, 3, 4, CITY])
    landing.run(cfg)
    site = tmp_path / "site"
    nm = json.loads((site / "data" / "netmap.json").read_text(encoding="utf-8"))
    assert nm["ids"] == [1, 2, 3, 4, CITY] and nm["method"] == "graph_fr"
    html = (site / "index.html").read_text(encoding="utf-8")
    assert 'id="data-netmap"' in html  # встроено: работает и по file://
    assert 'id="netmap-title"' in html and 'class="net-canvas"' in html
    i_net, i_ami = html.index('id="netmap"'), html.find('class="ami-block"')
    assert i_ami < 0 or i_net < i_ami  # после паспортов, до деления без типов
    assert html.index('id="types"') < i_net < html.index('id="order"')
    # перестановка карты по правилу выключена: nx, ny в mo.json пустые
    mo = json.loads((site / "data" / "mo.json").read_text(encoding="utf-8"))
    assert all(v is None for v in mo["nx"])


def test_stage_twice_same_netmap_bytes(tmp_path, monkeypatch):
    cfg = _setup(tmp_path)
    _fake_similarity(monkeypatch, [1, 2, 3, 4, CITY])
    landing.run(cfg)
    a = (tmp_path / "site" / "data" / "netmap.json").read_bytes()
    landing.run(cfg)
    assert (tmp_path / "site" / "data" / "netmap.json").read_bytes() == a


def test_stage_code3_netmap_node_unknown(tmp_path, monkeypatch):
    cfg = _setup(tmp_path)
    _fake_similarity(monkeypatch, [1, 2, 3, 4, CITY, 777])
    with pytest.raises(QCError, match="схема сети корзин"):
        landing.run(cfg)


def test_stage_without_netmap(tmp_path, monkeypatch):
    """Нет данных схемы (``site.build.netmap: false``) — нет ни блока, ни ссылки на него, ни netmap.json."""
    cfg = _setup(tmp_path)
    monkeypatch.setattr(
        landing,
        "similarity",
        lambda cfg: (
            {"chosen": None, "preserved": {}},
            pd.DataFrame({"territory_id": [], "nx": [], "ny": []}),
            None,
        ),
    )
    landing.run(cfg)
    html = (tmp_path / "site" / "index.html").read_text(encoding="utf-8")
    assert 'id="netmap"' not in html and 'href="#netmap"' not in html
    assert not (tmp_path / "site" / "data" / "netmap.json").exists()


def test_page_stems_catch_word_in_netmap_text():
    """«ступен»/«лестниц» в тексте схемы при T1 partial_overall — код 3 через page_stems."""
    story = {"verdicts": {"T1_ladder_external": "partial_overall"}}
    bad = landing.page_stems(CFG, story, {"index.html": "<p>Типы — ступени одной полосы</p>".encode()})
    assert bad and "ступен" in bad[0]
