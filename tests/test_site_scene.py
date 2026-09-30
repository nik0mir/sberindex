"""Объёмная карта первого экрана (``src/munnet/site_scene.py``, ``docs/landing_spec.md`` §4.7): спираль ячеек,
раскладка островов по типам, высоты и подписи — на маленькой синтетике с известным ответом."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from munnet import site_scene as sc
from munnet.site_hexgrid import PageGrid

NB = chr(0xA0)


def test_spiral_unique_and_compact():
    """Спираль: ячейки не повторяются, первые 1 + 3k(k + 1) заполняют ровно k колец (гекс-расстояние ≤ k)."""
    for n in (1, 2, 7, 19, 20, 806):
        pts = sc.spiral(n)
        assert len(pts) == n and len(set(pts)) == n
        k = sc.rings(n)
        assert all(max(abs(q), abs(r), abs(q + r)) <= k for q, r in pts)
    assert sc.spiral(7)[0] == (0, 0)
    assert {max(abs(q), abs(r), abs(q + r)) for q, r in sc.spiral(7)[1:]} == {1}
    assert [sc.rings(n) for n in (1, 2, 7, 8, 19, 20)] == [0, 1, 1, 2, 2, 3]


def test_ratio_words():
    """Доля кафе словами: от |ln r| ≥ 0,4 — «в N раза», иначе проценты; 1 — «как в регионе»."""
    assert sc.ratio_words(math.exp(0.784)) == f"примерно в{NB}2,2 раза больше, чем в{NB}регионе"
    assert sc.ratio_words(math.exp(-0.431)) == f"примерно в{NB}1,5 раза меньше, чем в{NB}регионе"
    assert sc.ratio_words(math.exp(-0.0318)) == f"примерно на{NB}3% меньше, чем в{NB}регионе"
    assert sc.ratio_words(1.0) == f"как в{NB}регионе"
    assert sc.ratio_words(None) == "нет данных"


def test_cafe_heights_exp_of_clr():
    vals = pd.DataFrame({"clr_rel_cafe": [0.0, np.log(2.0), -np.log(2.0)]}, index=[10, 11, 12])
    h = sc.cafe_heights(vals, [10, 11, 12])
    assert h.to_dict() == pytest.approx({10: 1.0, 11: 2.0, 12: 0.5})
    assert sc.cafe_heights(None, [10]).empty


def test_island_layout_order_no_overlap_heights_from_center():
    """Три типа разного размера и МО без типа: острова слева направо в заданном порядке, 0 — последним;
    ячейки не пересекаются; в центре острова — самая высокая ячейка, к краю высота не растёт по кольцам."""
    rng = np.random.default_rng(0)
    sizes = {2: 30, 1: 12, 3: 5, 0: 4}
    rows, i = [], 0
    for t, n in sizes.items():
        for _ in range(n):
            rows.append({"id": i, "t": t, "h": None if t == 0 else float(rng.uniform(0.3, 3))})
            i += 1
    cells = pd.DataFrame(rows)
    size, gap = 6.0, 34.0
    lay, isl = sc.island_layout(cells, [2, 1, 3], size, gap, cx=500.0, cy=250.0)
    assert [s["t"] for s in isl] == [2, 1, 3, 0]
    assert [s["n"] for s in isl] == [30, 12, 5, 4]
    assert sorted(lay["id"]) == list(range(len(cells)))
    xy = lay[["ix", "iy"]].to_numpy()
    d = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    assert d.min() >= size * math.sqrt(3) * 0.999
    # острова не заходят друг на друга: расстояние между центрами ≥ сумма радиусов + зазор
    for a, b in zip(isl, isl[1:], strict=False):
        assert b["x"] - a["x"] == pytest.approx(a["rad"] + b["rad"] + gap)
    # ряд центрирован по cx
    left = isl[0]["x"] - isl[0]["rad"]
    right = isl[-1]["x"] + isl[-1]["rad"]
    assert (left + right) / 2 == pytest.approx(500.0)
    m = cells.merge(lay, on="id")
    for s in isl[:3]:
        g = m[m["t"] == s["t"]].copy()
        # осевые координаты ячейки относительно центра острова -> номер кольца (гекс-расстояние)
        r = np.round((g["iy"] - s["y"]) / (1.5 * size), 3)
        q = np.round((g["ix"] - s["x"]) / (size * math.sqrt(3)) - r / 2, 3)
        k = np.maximum(np.maximum(q.abs(), r.abs()), (q + r).abs()).round()
        assert g.loc[k.idxmin(), "h"] == g["h"].max()
        # спираль заполняет кольца по порядку: максимум следующего кольца не выше минимума предыдущего
        mx = g.groupby(k)["h"].max().to_numpy()
        mn = g.groupby(k)["h"].min().to_numpy()
        assert len(mx) == sc.rings(s["n"]) + 1
        assert all(mx[j + 1] <= mn[j] + 1e-12 for j in range(len(mx) - 1))


def _mo():
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 9, 101],
            "role": ["territorial", "territorial", "city", "untyped", "inner"],
            "t": [1, 2, 2, np.nan, 2],
            "hq": [0, 1, 2, 3, 2],
            "hr": [0, 0, 0, 0, 0],
        }
    )


TEXTS = {
    "island_note": "у медианы типа доля кафе {ratio}",
    "untyped": "Без типа",
    "untyped_note": "ряд трат неполный",
}


def test_build_scene_contract():
    """scene.json: только ячейки карты (узлы и МО без типа, без районов столиц), высота — exp(clr_rel_cafe),
    подпись острова — медиана типа словами; нет значений — None; узел без значения — ошибка."""
    grid = PageGrid(size=6.0, x0=10.0, y0=10.0, width=100.0, height=40.0)
    vals = pd.DataFrame({"clr_rel_cafe": [0.0, np.log(2.0), np.log(4.0)]}, index=[1, 2, 3])
    sc_ = sc.build_scene(_mo(), grid, vals, [2, 1], {"1": "А", "2": "Б"}, {"1": "#111", "2": "#222"}, TEXTS,
                         {"gap": 34, "dy": 30})  # fmt: skip
    c = sc_["cells"]
    assert c["id"] == [1, 2, 3, 9]
    assert c["h"] == [1.0, 2.0, 4.0, None]
    assert c["t"] == [1, 2, 2, 0]
    assert [s["t"] for s in sc_["islands"]] == [2, 1, 0]
    b = sc_["islands"][0]
    assert b["n"] == 2 and b["h_med"] == pytest.approx(math.sqrt(8), abs=1e-3)  # медиана 2 и 4 — в логарифме
    assert b["note"] == f"у медианы типа доля кафе примерно в{NB}2,8 раза больше, чем в{NB}регионе"
    assert sc_["islands"][-1]["note"] == "ряд трат неполный"
    x, _ = grid.center(np.array([0, 1, 2, 3]), np.zeros(4))
    assert c["x"] == [round(float(v), 1) for v in x]
    assert sc.build_scene(_mo(), grid, None, [2, 1], {}, {}, TEXTS, {"gap": 34, "dy": 30}) is None
    with pytest.raises(ValueError, match="clr_rel_cafe"):
        sc.build_scene(_mo(), grid, vals.drop(index=3), [2, 1], {}, {}, TEXTS, {"gap": 34, "dy": 30})
