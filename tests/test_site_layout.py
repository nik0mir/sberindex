"""Раскладка «по сходству трат» (site_layout) и палитры типов (style.palette_audit) на синтетике."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from munnet import site_layout as SL
from munnet import style

SITE = {
    "similarity_layout": {
        "nodes": "all_nodes",
        "candidates": {
            "graph_fr": {
                "graph": "network_edges.basket_dist",
                "main_only": True,
                "weights": "none",
                "method": "fruchterman_reingold",
                "niter": 300,
            },
            "hybrid_pca": {"space": "interpret.examples.space", "components": 2},
        },
        "choose": "neighbor_preservation",
        "n_net_neighbors": 3,
        "min_preserved": 0.30,
        "caption": "На схеме сохранено {preserved} соседей по сети; расстояния приблизительны",
    }
}


def _two_cliques(n: int = 8):
    """Две клики по n узлов, соединённые одним ребром; id — не подряд."""
    ids = np.arange(n * 2) * 7 + 100
    src, tgt = [], []
    for base in (0, n):
        for i in range(n):
            for j in range(i + 1, n):
                src.append(ids[base + i])
                tgt.append(ids[base + j])
    src.append(ids[0])
    tgt.append(ids[n])
    return ids, np.array(src), np.array(tgt)


# --- доля сохранённых соседей ------------------------------------------------------------------------


def test_preservation_known_answer():
    # путь 0–1–2–3 на прямой: у каждого узла ближайший на схеме — его сосед по сети
    ids = np.array([10, 11, 12, 13])
    nb = SL.neighbor_lists(ids, np.array([10, 11, 12]), np.array([11, 12, 13]))
    line = np.array([[0.0, 0], [1, 0], [2, 0], [3, 0]])
    assert SL.neighbor_preservation(line, nb, k=1) == pytest.approx(1.0)
    # k = 2: у концов степень 1 → знаменатель min(2, 1) = 1, у середины 2 ближайших = 2 соседа
    assert SL.neighbor_preservation(line, nb, k=2) == pytest.approx(1.0)
    # узел 3 стоит между 1 и 2, ближе к 1: у него ближайший на схеме — не сосед по сети
    moved = np.array([[0.0, 0], [1, 0], [3.5, 0], [2.2, 0]])
    assert SL.preservation_by_node(moved, nb, k=1).tolist() == [1.0, 1.0, 1.0, 0.0]
    # концы пути рядом, середина далеко: ни у кого ближайший не сосед
    bad = np.array([[0.0, 0], [10, 0], [-10, 0], [1, 0]])
    assert SL.preservation_by_node(bad, nb, k=1).tolist() == [0.0, 0.0, 0.0, 0.0]


def test_isolated_nodes_skipped_and_edges_outside_rejected():
    ids = np.array([1, 2, 3])
    nb = SL.neighbor_lists(ids, np.array([1]), np.array([2]))
    per = SL.preservation_by_node(np.array([[0.0, 0], [1, 0], [5, 0]]), nb, k=1)
    assert np.isnan(per[2]) and per[0] == 1.0
    with pytest.raises(ValueError, match="вне списка"):
        SL.neighbor_lists(ids, np.array([1]), np.array([99]))


def test_random_layout_near_k_over_n():
    # кольцо: степень 2 = k, у случайной схемы ожидание доли — k / (n − 1)
    rng = np.random.default_rng(0)
    n, k = 300, 2
    ids = np.arange(n)
    nb = SL.neighbor_lists(ids, ids, np.roll(ids, -1))
    vals = [SL.neighbor_preservation(rng.uniform(size=(n, 2)), nb, k) for _ in range(40)]
    assert np.mean(vals) == pytest.approx(k / (n - 1), abs=0.003)


# --- выбор по правилу --------------------------------------------------------------------------------


def test_choose_rule():
    order = ("graph_fr", "hybrid_pca")
    assert SL.choose({"graph_fr": 0.4, "hybrid_pca": 0.5}, order, 0.3) == "hybrid_pca"
    assert SL.choose({"graph_fr": 0.5, "hybrid_pca": 0.4}, order, 0.3) == "graph_fr"
    assert SL.choose({"graph_fr": 0.4, "hybrid_pca": 0.4}, order, 0.3) == "graph_fr"  # равенство — первый
    assert SL.choose({"graph_fr": 0.29, "hybrid_pca": 0.1}, order, 0.3) is None  # оба ниже порога
    assert SL.choose({"graph_fr": 0.30, "hybrid_pca": 0.1}, order, 0.3) == "graph_fr"  # порог включён


def test_spec_rejects_other_rule():
    SL.LayoutSpec.from_config(SITE)
    bad = copy.deepcopy(SITE)
    bad["similarity_layout"]["choose"] = "stress"
    with pytest.raises(ValueError, match="choose"):
        SL.LayoutSpec.from_config(bad)
    bad = copy.deepcopy(SITE)
    bad["similarity_layout"]["candidates"]["graph_fr"]["weights"] = "weight"
    with pytest.raises(ValueError, match="weights"):
        SL.LayoutSpec.from_config(bad)


# --- кандидаты ---------------------------------------------------------------------------------------


def test_pca_sign_fixed_and_recovers_plane():
    rng = np.random.default_rng(1)
    plane = rng.normal(size=(50, 2)) * [5, 2]
    Z = np.hstack([plane, rng.normal(scale=0.01, size=(50, 3))])
    a, b = SL.layout_pca(Z), SL.layout_pca(Z[:, ::1].copy())
    np.testing.assert_allclose(a, b)
    # PCA восстанавливает плоскость: ближайшие соседи в Z и на схеме почти совпадают
    ids = np.arange(50)
    _, idx = SL.cKDTree(Z).query(Z, k=4)
    nb = [row[1:] for row in idx]
    assert SL.neighbor_preservation(a, nb, 3) > 0.9
    assert ids.size == 50


def test_similarity_layout_two_cliques_deterministic():
    ids, src, tgt = _two_cliques()
    rng = np.random.default_rng(3)
    Z = rng.normal(size=(len(ids), 4))  # шум: PCA не знает сети
    spec = SL.LayoutSpec.from_config(SITE)
    r1 = SL.similarity_layout(spec, ids, src, tgt, Z, seed=42)
    r2 = SL.similarity_layout(spec, ids, src, tgt, Z, seed=42)
    np.testing.assert_allclose(r1.coords["graph_fr"], r2.coords["graph_fr"])
    # раскладка сети держит клики вместе, шумная PCA — нет
    assert r1.preserved["graph_fr"] > 0.9
    assert r1.preserved["graph_fr"] > r1.preserved["hybrid_pca"]
    assert r1.chosen == "graph_fr"
    t = r1.nxy()
    assert t["territory_id"].tolist() == ids.tolist()
    assert t["nx"].between(0, 1000).all() and t["ny"].between(0, 1000).all()
    assert r1.summary()["chosen"] == "graph_fr"


def test_no_layout_when_below_threshold():
    ids, src, tgt = _two_cliques()
    spec = SL.LayoutSpec.from_config(SITE)
    spec = SL.LayoutSpec(**{**spec.__dict__, "min_preserved": 1.01})
    r = SL.similarity_layout(spec, ids, src, tgt, np.random.default_rng(0).normal(size=(len(ids), 3)), seed=1)
    assert r.chosen is None
    assert r.nxy().empty
    assert "перестановки нет" in r.summary()["reason"]


def test_normalize_keeps_aspect():
    p = SL.normalize(np.array([[0.0, 0.0], [2.0, 1.0], [1.0, 0.5]]))
    assert p[:, 0].min() == 0 and p[:, 0].max() == 1000
    assert p[:, 1].max() - p[:, 1].min() == 500  # общий масштаб осей
    assert p[0, 1] > p[1, 1]  # ny вниз, как в SVG


# --- палитры -----------------------------------------------------------------------------------------


def test_delta_e2000_reference():
    # Sharma, Wu, Dalal (2005), табл. 1, пары 1 и 7
    assert style.delta_e2000([50, 2.6772, -79.7751], [50, 0, -82.7485]) == pytest.approx(2.0425, abs=1e-4)
    assert style.delta_e2000([50, 0, 0], [50, -1, 2]) == pytest.approx(2.3669, abs=1e-4)


def test_contrast_and_cvd_basics():
    assert style.contrast("#000000", "#FFFFFF") == pytest.approx(21.0)
    assert style.contrast("#777777", "#FFFFFF") == pytest.approx(4.48, abs=0.01)
    for kind in ("protan", "deutan", "tritan"):  # серый при симуляции не меняется
        np.testing.assert_allclose(style.simulate_cvd("#808080", kind), style.lab("#808080"), atol=1e-3)
    # красный и зелёный одной светлоты при дейтеранопии сближаются сильнее, чем для обычного глаза
    red, green = "#C0392B", "#2E8B57"
    normal = style.delta_e2000(style.lab(red), style.lab(green))
    deutan = style.delta_e2000(style.simulate_cvd(red, "deutan"), style.simulate_cvd(green, "deutan"))
    assert deutan < normal / 2


def test_project_palettes_pass_audit():
    audit = style.palette_audit([2, 1, 3, 4], {1: "●", 2: "▲", 3: "■", 4: "◆"})
    assert audit.ok, audit.failures
    assert audit.exit_code == 0
    names = {c["check"] for c in audit.checks}
    for kind in ("grey", "protan", "deutan", "tritan"):
        assert f"nominal.{kind}" in names and f"ordered.{kind}" in names
    assert "nominal.not_monotone" in names


def test_audit_catches_monotone_nominal(monkeypatch):
    # номинальная, чья светлота растёт по ступеням, — нарушение (код 3)
    order = [2, 1, 3, 4]
    ordered = style.type_palette("ordered", order)
    monkeypatch.setattr(style, "TYPE_PALETTE_NOMINAL", dict(ordered))
    audit = style.palette_audit(order)
    assert audit.exit_code == 3
    assert "nominal.not_monotone" in {c["check"] for c in audit.failures}


def test_audit_catches_light_and_confusable(monkeypatch):
    monkeypatch.setattr(
        style, "TYPE_PALETTE_NOMINAL", {1: "#FFEE88", 2: "#B8563B", 3: "#6B7F2E", 4: "#1C5A8C"}
    )
    fails = {c["check"] for c in style.palette_audit([2, 1, 3, 4]).failures}
    assert "nominal.contrast_bg.1" in fails  # светло-жёлтый на белом
    assert "nominal.deutan" in fails  # кирпичный и оливковый: ΔE2000 ≈ 41, при дейтеранопии ≈ 1,6


def test_audit_catches_wrong_shapes():
    audit = style.palette_audit([2, 1, 3, 4], {1: "●", 2: "●", 3: "■", 4: "◆"})
    assert "shapes.config" in {c["check"] for c in audit.failures}


def test_type_palette_ordered_follows_ladder():
    pal = style.type_palette("ordered", [2, 1, 3, 4])
    assert pal[2] == style.TYPE_PALETTE_ORDERED[0] and pal[4] == style.TYPE_PALETTE_ORDERED[3]
    with pytest.raises(ValueError):
        style.type_palette("rainbow", [1, 2, 3, 4])
