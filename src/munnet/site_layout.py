"""Раскладка «по сходству трат» для лендинга (docs/landing_spec.md §3.3).

Правило — ``site.similarity_layout`` (предрегистрация 30.09.2026).

Два кандидата, оба из тех же данных, из которых построены типы:

- ``graph_fr`` — раскладка Фрухтермана — Рейнгольда основной сети ``basket_dist`` (igraph, без весов,
  ``niter`` итераций; начальные координаты и генератор igraph — от seed конфига);
- ``hybrid_pca`` — первые две главные компоненты стандартизованного пространства итогового гибрида
  (спектральное вложение сети ⊕ признаки места, веса ``hybrid_alpha`` — как у K-means этапа cluster).

Выбор — по заранее записанному правилу ``neighbor_preservation``: для узла со степенью > 0 —
|k ближайших на схеме ∩ соседи в основной сети| / min(k, степень), k = ``n_net_neighbors``; среднее по узлам.
Берётся кандидат с большей долей, при равенстве — первый в ``candidates``; если у обоих доля меньше
``min_preserved``, перестановки нет. Типы в выборе не участвуют: раскраска — только в прототипе PNG.
Координаты для страницы — целые 0…1000 (поля ``nx``, ``ny`` контракта ``mo.json``), ``ny`` — вниз, как в SVG.
"""

from __future__ import annotations

import json
import logging
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from munnet.config import Config

log = logging.getLogger(__name__)

GRAPH_FR = "graph_fr"
HYBRID_PCA = "hybrid_pca"
KNOWN = (GRAPH_FR, HYBRID_PCA)
SCALE = 1000  # координаты страницы: целые 0…SCALE
TIE_TOL = 1e-12  # доли, отличающиеся меньше, считаются равными (правило «при равенстве — первый»)


@dataclass(frozen=True)
class LayoutSpec:
    """Правило ``site.similarity_layout``; другое значение записанного ключа — ошибка."""

    candidates: tuple[str, ...]
    niter: int
    components: int
    k: int
    min_preserved: float
    caption: str

    @classmethod
    def from_config(cls, site: Mapping[str, Any]) -> LayoutSpec:
        sl = site["similarity_layout"]
        c = sl["candidates"]
        names = tuple(str(x) for x in c)
        if set(names) != set(KNOWN):
            raise ValueError(f"site.similarity_layout.candidates: реализовано {KNOWN}, в конфиге {names}")
        fr, pca = c[GRAPH_FR], c[HYBRID_PCA]
        expect = {
            "nodes": (sl.get("nodes"), "all_nodes"),
            "choose": (sl.get("choose"), "neighbor_preservation"),
            "graph_fr.graph": (fr.get("graph"), "network_edges.basket_dist"),
            "graph_fr.main_only": (fr.get("main_only"), True),
            "graph_fr.weights": (fr.get("weights"), "none"),
            "graph_fr.method": (fr.get("method"), "fruchterman_reingold"),
            "hybrid_pca.space": (pca.get("space"), "interpret.examples.space"),
        }
        for key, (got, want) in expect.items():
            if got != want:
                raise ValueError(f"site.similarity_layout.{key}: реализовано {want!r}, в конфиге {got!r}")
        return cls(
            candidates=names,
            niter=int(fr["niter"]),
            components=int(pca["components"]),
            k=int(sl["n_net_neighbors"]),
            min_preserved=float(sl["min_preserved"]),
            caption=str(sl["caption"]),
        )


# --- Соседи и доля сохранённых соседей ------------------------------------------------------------


def neighbor_lists(ids: np.ndarray, source: np.ndarray, target: np.ndarray) -> list[np.ndarray]:
    """Соседи каждого узла неориентированной сети — индексы ``ids``; ребро к узлу вне ``ids`` — ошибка."""
    pos = pd.Index(np.asarray(ids))
    if not pos.is_unique:
        raise ValueError("site_layout: territory_id узлов повторяются")
    s, t = pos.get_indexer(np.asarray(source)), pos.get_indexer(np.asarray(target))
    if (s < 0).any() or (t < 0).any():
        raise ValueError("site_layout: ребро сети ведёт к узлу вне списка узлов")
    keep = s != t
    s, t = s[keep], t[keep]
    a = np.concatenate([s, t])
    b = np.concatenate([t, s])
    order = np.lexsort((b, a))
    a, b = a[order], b[order]
    out: list[np.ndarray] = [np.empty(0, dtype=np.int64) for _ in range(len(pos))]
    if len(a):
        starts = np.flatnonzero(np.r_[True, a[1:] != a[:-1]])
        for st, en in zip(starts, np.r_[starts[1:], len(a)], strict=True):
            out[int(a[st])] = np.unique(b[st:en])
    return out


def preservation_by_node(coords: np.ndarray, neighbors: Sequence[np.ndarray], k: int) -> np.ndarray:
    """Доля сохранённых соседей каждого узла: |k ближайших на схеме ∩ соседи в сети| / min(k, степень);
    у узла без соседей в сети — NaN. Ближайшие — по евклидову расстоянию на схеме, сам узел исключён."""
    coords = np.asarray(coords, dtype=np.float64)
    n = len(coords)
    if len(neighbors) != n:
        raise ValueError("site_layout: число узлов схемы и сети не совпадает")
    kk = min(k + 1, n)
    _, idx = cKDTree(coords).query(coords, k=kk)
    idx = np.asarray(idx).reshape(n, kk)
    out = np.full(n, np.nan)
    for i in range(n):
        deg = len(neighbors[i])
        if deg == 0:
            continue
        near = [j for j in idx[i] if j != i][:k]
        out[i] = len(np.intersect1d(near, neighbors[i])) / min(k, deg)
    return out


def neighbor_preservation(coords: np.ndarray, neighbors: Sequence[np.ndarray], k: int) -> float:
    """Средняя доля сохранённых соседей по узлам со степенью > 0 (правило ``neighbor_preservation``)."""
    per = preservation_by_node(coords, neighbors, k)
    if np.isnan(per).all():
        raise ValueError("site_layout: в сети нет рёбер")
    return float(np.nanmean(per))


# --- Кандидаты ------------------------------------------------------------------------------------


def layout_graph_fr(n: int, neighbors: Sequence[np.ndarray], seed: int, niter: int) -> np.ndarray:
    """Фрухтерман — Рейнгольд без весов и без сетки (точный расчёт сил): начальные координаты — от
    ``numpy`` с ``seed``, генератор igraph на время расчёта — ``random.Random(seed)``."""
    import igraph as ig

    edges = sorted({(min(i, int(j)), max(i, int(j))) for i in range(n) for j in neighbors[i]})
    g = ig.Graph(n=n, edges=edges)
    init = np.random.default_rng(seed).uniform(-np.sqrt(n), np.sqrt(n), size=(n, 2))
    ig.set_random_number_generator(random.Random(seed))
    try:
        lay = g.layout_fruchterman_reingold(niter=niter, seed=init.tolist(), grid=False)
    finally:
        ig.set_random_number_generator(random)
    return np.asarray(lay.coords, dtype=np.float64)


def layout_pca(Z: np.ndarray, components: int = 2) -> np.ndarray:
    """Первые ``components`` главных компонент (SVD центрированной матрицы). Знак компоненты фиксирован:
    наибольшая по модулю нагрузка положительна — повторный расчёт даёт те же координаты."""
    Z = np.asarray(Z, dtype=np.float64)
    Zc = Z - Z.mean(axis=0)
    _, _, vt = np.linalg.svd(Zc, full_matrices=False)
    vt = vt[:components]
    signs = np.sign(vt[np.arange(len(vt)), np.abs(vt).argmax(axis=1)])
    return Zc @ (vt * signs[:, None]).T


def normalize(coords: np.ndarray, scale: int = SCALE) -> np.ndarray:
    """Целые 0…``scale`` с общим масштабом осей (расстояния не искажаются); короткая ось — по центру;
    ``ny`` растёт вниз, как в SVG."""
    c = np.asarray(coords, dtype=np.float64)
    lo, hi = c.min(axis=0), c.max(axis=0)
    span = float((hi - lo).max())
    if span <= 0:
        return np.full(c.shape, scale // 2, dtype=np.int64)
    u = (c - lo) / span * scale
    u += (scale - (hi - lo) / span * scale) / 2
    u[:, 1] = scale - u[:, 1]
    return np.clip(np.rint(u), 0, scale).astype(np.int64)


def choose(preserved: Mapping[str, float], order: Sequence[str], min_preserved: float) -> str | None:
    """Кандидат с наибольшей долей; при равенстве — первый в ``order``; все ниже порога — ``None``."""
    best: str | None = None
    for name in order:
        if best is None or preserved[name] > preserved[best] + TIE_TOL:
            best = name
    if best is None or preserved[best] < min_preserved:
        return None
    return best


# --- Итог ------------------------------------------------------------------------------------------


@dataclass
class LayoutResult:
    ids: np.ndarray
    coords: dict[str, np.ndarray]  # сырые координаты кандидатов
    preserved: dict[str, float]
    per_node: dict[str, np.ndarray]
    random_preserved: float  # та же доля у случайной схемы (для подписи «у случайной схемы ≈ …»)
    chosen: str | None
    spec: LayoutSpec
    n_scored: int  # узлов со степенью > 0
    n_edges: int
    info: dict[str, Any] = field(default_factory=dict)

    def pixels(self, name: str) -> np.ndarray:
        return normalize(self.coords[name])

    def nxy(self) -> pd.DataFrame:
        """``territory_id``, ``nx``, ``ny`` выбранной раскладки; перестановки нет — пустая таблица."""
        cols = {
            "territory_id": pd.Series(dtype="int32"),
            "nx": pd.Series(dtype="int32"),
            "ny": pd.Series(dtype="int32"),
        }
        if self.chosen is None:
            return pd.DataFrame(cols)
        p = self.pixels(self.chosen)
        return pd.DataFrame(
            {
                "territory_id": self.ids.astype("int32"),
                "nx": p[:, 0].astype("int32"),
                "ny": p[:, 1].astype("int32"),
            }
        )

    def summary(self) -> dict[str, Any]:
        reason = (
            f"выбрана {self.chosen}: доля больше и не меньше порога {self.spec.min_preserved}"
            if self.chosen
            else f"перестановки нет: у обоих кандидатов доля меньше порога {self.spec.min_preserved}"
        )
        return {
            "rule": "neighbor_preservation",
            "k": self.spec.k,
            "min_preserved": self.spec.min_preserved,
            "candidates": list(self.spec.candidates),
            "preserved": {k: round(v, 6) for k, v in self.preserved.items()},
            "random_preserved": round(self.random_preserved, 6),
            "chosen": self.chosen,
            "reason": reason,
            "n_nodes": int(len(self.ids)),
            "n_scored": self.n_scored,
            "n_edges": self.n_edges,
            **self.info,
        }


def similarity_layout(
    spec: LayoutSpec, ids: np.ndarray, source: np.ndarray, target: np.ndarray, Z: np.ndarray, seed: int
) -> LayoutResult:
    """Обе раскладки, доли сохранённых соседей и выбор по правилу. ``Z`` — строки в порядке ``ids``."""
    ids = np.asarray(ids)
    if len(Z) != len(ids):
        raise ValueError("site_layout: строк пространства гибрида не столько, сколько узлов")
    nb = neighbor_lists(ids, source, target)
    n = len(ids)
    coords = {
        GRAPH_FR: layout_graph_fr(n, nb, seed, spec.niter),
        HYBRID_PCA: layout_pca(Z, spec.components),
    }
    per = {name: preservation_by_node(coords[name], nb, spec.k) for name in spec.candidates}
    preserved = {name: float(np.nanmean(v)) for name, v in per.items()}
    rnd = np.random.default_rng(seed).uniform(size=(n, 2))
    random_preserved = neighbor_preservation(rnd, nb, spec.k)
    chosen = choose(preserved, spec.candidates, spec.min_preserved)
    n_scored = int(sum(len(x) > 0 for x in nb))
    n_edges = int(sum(len(x) for x in nb) // 2)
    log.info(
        "site: доля сохранённых соседей (k = %d, %d узлов, %d рёбер): %s; случайная схема %.4f; выбор — %s",
        spec.k,
        n_scored,
        n_edges,
        ", ".join(f"{k} {v:.4f}" for k, v in preserved.items()),
        random_preserved,
        chosen or "перестановки нет",
    )
    return LayoutResult(
        ids=ids,
        coords=coords,
        preserved=preserved,
        per_node=per,
        random_preserved=random_preserved,
        chosen=chosen,
        spec=spec,
        n_scored=n_scored,
        n_edges=n_edges,
    )


# --- Входы из выходов этапов network и cluster ------------------------------------------------------


def main_edges(processed: Path) -> pd.DataFrame:
    """Рёбра основной сети ``basket_dist`` (``is_main``) из ``network_edges``."""
    e = pd.read_parquet(
        Path(processed) / "network_edges.parquet",
        columns=["rule", "source", "target", "is_main"],
        filters=[("rule", "==", "basket_dist"), ("is_main", "==", True)],
    )
    if e.empty:
        raise ValueError("site_layout: в network_edges нет основной сети basket_dist")
    return e[["source", "target"]].astype("int64").reset_index(drop=True)


def hybrid_space(cfg: Config) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Стандартизованное пространство итогового гибрида — те же функции и параметры, что у примеров этапа 5
    (``interpret.examples.space: hybrid_input``): спектральное вложение сети ⊕ признаки места с весами
    ``hybrid_alpha`` из ``outputs/cluster/final.json``. Возвращает ``ids``, ``Z`` и параметры."""
    from types import SimpleNamespace

    from munnet.clustering import inputs as CI
    from munnet.clustering import methods as M

    fin = json.loads((cfg.dir("outputs") / "cluster" / "final.json").read_text(encoding="utf-8"))
    g = fin["inputs"]["graph"]
    cp = SimpleNamespace(
        graph_rule=str(g["rule"]),
        graph_sparsify=str(g["sparsify"]),
        graph_k=int(g["k"]),
        attributes=tuple(fin["inputs"]["features"]),
        place_year=int(fin["inputs"]["place_year"]),
    )
    base = CI.load_inputs(cfg, cp).inputs
    k, alpha, best_seed = int(fin["k"]), float(fin["hybrid_alpha"]), int(fin["best_seed"])
    E = M.spectral_embed(base.A, k, best_seed)
    Z = M.block_join([E, base.X], [alpha, 1 - alpha])
    info = {"k": k, "hybrid_alpha": alpha, "best_seed": best_seed, "dims": int(Z.shape[1])}
    return np.asarray(base.ids, dtype=np.int64), Z, info


def build(cfg: Config) -> LayoutResult:
    """Раскладка для этапа site: правило из ``site.similarity_layout``, seed — ``cfg["seed"]``."""
    spec = LayoutSpec.from_config(cfg["site"])
    ids, Z, info = hybrid_space(cfg)
    e = main_edges(cfg.dir("processed"))
    res = similarity_layout(spec, ids, e["source"].to_numpy(), e["target"].to_numpy(), Z, int(cfg["seed"]))
    res.info.update({"hybrid": info, "seed": int(cfg["seed"])})
    return res


# --- Прототип PNG -----------------------------------------------------------------------------------


def plot_candidates(res: LayoutResult, types: Mapping[int, int], stem: Path, dpi: int = 200) -> list[Path]:
    """Прототип: оба кандидата рядом, раскраска и фигуры — по типам (номинальная палитра: вердикт T1
    при сборке прототипа неизвестен). Под каждой схемой — доля сохранённых соседей."""
    from munnet import style

    t = np.array([int(types.get(int(i), 0)) for i in res.ids])
    colors = style.type_palette("nominal", sorted(style.TYPE_PALETTE_NOMINAL))
    with style.use():
        fig, axes = style.new_figure("tall", ncols=2)
        for ax, name in zip(axes, res.spec.candidates, strict=True):
            p = res.pixels(name)
            for typ in sorted(set(t.tolist())):
                m = t == typ
                ax.scatter(
                    p[m, 0],
                    p[m, 1],
                    s=7,
                    marker=style.TYPE_MARKERS.get(typ, "."),
                    linewidths=0,
                    color=colors.get(typ, style.CONTEXT),
                    alpha=0.8,
                    label=f"тип {typ}" if typ else "без типа",
                )
            ax.set_xlim(-10, SCALE + 10)
            ax.set_ylim(SCALE + 10, -10)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            for sp_ in ax.spines.values():
                sp_.set_visible(False)
            ax.set_title(
                f"{name}: сохранено {style.fmt_pct(res.preserved[name])} соседей по сети",
                loc="left",
                fontsize=style.LABEL_PT,
            )
        axes[0].legend(loc="lower left", frameon=False, fontsize=style.TICK_PT, markerscale=2)
        chosen = res.chosen or "перестановки нет"
        style.finish(
            fig,
            f"Раскладка по сходству трат: выбор по правилу — {chosen}",
            f"k = {res.spec.k}; порог {style.fmt_num(res.spec.min_preserved, 2)}; случайная схема — "
            f"{style.fmt_pct(res.random_preserved)}. Прототип: цвет — тип этапа 3, в выборе не участвует",
            style.join_sources(style.SOURCE_SBER, style.SOURCE_ROSSTAT),
        )
        return style.save(fig, stem, dpi=dpi)
