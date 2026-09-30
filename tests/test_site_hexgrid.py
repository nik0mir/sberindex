"""Карта равных ячеек (``site_hexgrid``) на синтетике с известным ответом: геометрия шестиугольников,
связность регионов, порядок внутри региона, контуры по сторонам ячеек, пути SVG без накопления ошибки."""

from __future__ import annotations

import re

import numpy as np
import pytest

from munnet import site_hexgrid as H

PARAMS = {
    "cell_km": 10,
    "spare": 0.15,
    "iters": 300,
    "anchor": 0.02,
    "attract": 0.2,
    "max_edge_km": 500,
    "radius_factor": 0.95,
}


def test_axial_roundtrip():
    q, r = np.meshgrid(np.arange(-5, 6), np.arange(-4, 5))
    x, y = H.axial_to_xy(q.ravel(), r.ravel(), 7.0)
    q2, r2 = H.xy_to_axial(x + 0.3, y - 0.2, 7.0)  # сдвиг меньше половины ячейки — та же ячейка
    assert (q2 == q.ravel()).all() and (r2 == r.ravel()).all()


def test_neighbors_are_adjacent():
    """Шесть направлений DIRS — ровно соседи: расстояние между центрами √3·s."""
    for dq, dr in H.DIRS:
        x, y = H.axial_to_xy(np.array([dq]), np.array([dr]), 1.0)
        assert np.hypot(x[0], y[0]) == pytest.approx(np.sqrt(3))


def _clusters():
    """Три региона: плотный (40 точек в квадрате 30 км), средний и редкий (8 точек на 400 км)."""
    rng = np.random.default_rng(1)
    a = rng.uniform(0, 30, (40, 2))
    b = rng.uniform(0, 80, (20, 2)) + [100, 0]
    c = rng.uniform(0, 400, (8, 2)) + [300, 0]
    xy = np.vstack([a, b, c])
    groups = np.array([1] * 40 + [2] * 20 + [3] * 8)
    return xy, groups


def test_build_grid_unique_connected_deterministic():
    xy, groups = _clusters()
    ids = np.arange(len(xy)) + 1
    r1 = H.build_grid(ids, xy[:, 0], xy[:, 1], groups, PARAMS)
    r2 = H.build_grid(ids, xy[:, 0], xy[:, 1], groups, PARAMS)
    cells = set(zip(r1.q.tolist(), r1.r.tolist(), strict=True))
    assert len(cells) == len(ids)  # каждая ячейка — одно МО
    assert r1.stats["groups_split"] == 0  # каждый регион — связное пятно, редкий тоже
    assert (r1.q == r2.q).all() and (r1.r == r2.r).all()  # без случайности
    assert r1.stats["shift_median_km"] >= 0 and r1.stats["shift_max_km"] >= r1.stats["shift_median_km"]


def test_order_within_region_kept():
    """Внутри региона взаимное расположение сохраняется: крайняя западная точка — в западной ячейке,
    крайняя восточная — в восточной (цепочка из 7 точек в одном регионе)."""
    x = np.array([0, 10, 20, 30, 40, 50, 60], dtype=float)
    y = np.zeros_like(x)
    res = H.build_grid(np.arange(7) + 1, x, y, np.ones(7), PARAMS)
    cx, _ = H.axial_to_xy(res.q, res.r, res.s)
    assert np.argmin(cx) == 0 and np.argmax(cx) == 6


def test_components_known():
    q = np.array([0, 1, 5])
    r = np.array([0, 0, 0])
    assert H.components(q, r, np.array(["a", "a", "a"])) == {"a": 2}
    assert H.components(q, r, np.array(["a", "b", "a"])) == {"a": 2, "b": 1}


def test_boundary_edges_counts():
    """Одна ячейка — 6 сторон контура; две соседние разных групп — общая сторона один раз: 11;
    две соседние одной группы — только внешний контур: 10."""
    one = H.boundary_edges(np.array([0]), np.array([0]), np.array([1]), 1.0)
    assert len(one) == 6
    q, r = np.array([0, 1]), np.array([0, 0])
    assert len(H.boundary_edges(q, r, np.array([1, 2]), 1.0)) == 11
    assert len(H.boundary_edges(q, r, np.array([1, 1]), 1.0)) == 10
    lines = H.chain(H.boundary_edges(q, r, np.array([1, 1]), 1.0))
    assert len(lines) == 1 and lines[0][0] == lines[0][-1]  # замкнутый контур одной ломаной


def test_page_axial_matches_template_formula():
    """hq = q + r, hr = −r: центр по формуле шаблона (y = y0 − size·1,5·hr) совпадает с центром модуля."""
    q, r = np.array([0, 3, -2, 4]), np.array([0, 1, 5, -3])
    hq, hr = H.page_axial(q, r)
    g = H.page_grid(hq, hr, 6.0, 10.0)
    px, py = g.center(hq, hr)
    x, y = H.axial_to_xy(q, r, 6.0)
    assert np.allclose(px - x, px[0] - x[0]) and np.allclose(py - y, py[0] - y[0])
    assert px.min() - 6 * np.sqrt(3) / 2 == pytest.approx(10.0) and py.min() - 6 == pytest.approx(10.0)


def test_hex_path_relative_no_drift():
    """Относительные переходы пути SVG восстанавливают абсолютные вершины с точностью округления 0,1
    даже через сотни ячеек (ошибка не копится)."""
    hq = np.arange(300) % 30
    hr = np.arange(300) // 30
    g = H.page_grid(hq, hr, 6.0, 10.0)
    x, y = g.center(hq, hr)
    d = H.hex_path(x, y, 6.0)
    starts = []
    cur = None
    for cmd, a, b in re.findall(r"([Mm])(-?[\d.]+) (-?[\d.]+)", d):
        a, b = float(a), float(b)
        cur = (a, b) if cmd == "M" else (cur[0] + a, cur[1] + b)
        starts.append(cur)
    want = sorted(zip(np.round(y - 6, 1), np.round(x, 1), strict=True))
    got = sorted((round(b, 1), round(a, 1)) for a, b in starts)
    assert len(got) == 300
    assert np.allclose(np.array(got), np.array(want), atol=0.051)


def test_duplicate_ids_rejected():
    with pytest.raises(ValueError, match="повторяются"):
        H.build_grid([1, 1], [0, 1], [0, 1], [1, 1], PARAMS)
