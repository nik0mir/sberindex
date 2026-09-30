"""Карта равных ячеек лендинга (``docs/landing_spec.md``, §3.2): одна ячейка — один узел сети или МО без типа.

Шаги (параметры — ``site.build.hex``):

1. Точки ``x_aea``, ``y_aea`` (Альберс, ``lon_0 = 100``) — в километрах, ось y вниз, как в SVG.
2. Регион — круг площадью «число его ячеек × площадь ячейки» у медианы точек региона. Круги раздвигаются
   без наложений и стягиваются к соседям по триангуляции Делоне (``dorling``): плотный запад раздвигается,
   редкий восток сжимается, карта остаётся сплошной.
3. Ячейки — регионам: сбалансированное назначение ``scipy.optimize.linear_sum_assignment`` по квадрату
   расстояния от центра круга (``region_cells``; оптимум — дискретная диаграмма мощности, регион связен).
4. Внутри региона МО — на его ячейки тем же назначением по квадрату смещения от ``x_aea``, ``y_aea``,
   приведённых к центру и разбросу ячеек региона (``fit_within``): соседство внутри региона сохраняется.

Качество (под картой и в логе): смещение ячейки от исходной точки, км (медиана, 90-й перцентиль, максимум);
число регионов, чьи ячейки распались на части; доля сохранённых 10 ближайших соседей.

Шестиугольники «остриём вверх». Внутри модуля осевые ``q``, ``r`` при оси y вниз: ``x = s·√3·(q + r/2)``,
``y = s·1,5·r``. На странице — соглашение шаблона: ``hq = q + r``, ``hr = −r``,
``x = x0 + size·√3·(hq + hr/2)``, ``y = y0 − size·1,5·hr`` (``page_axial``).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial import Delaunay, cKDTree

log = logging.getLogger(__name__)

SQRT3 = float(np.sqrt(3.0))
# Соседи в осевых координатах (остриё вверх, y вниз): восток, северо-восток, северо-запад, запад, юго-запад,
# юго-восток
DIRS: tuple[tuple[int, int], ...] = ((1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1))


# --- геометрия шестиугольников ------------------------------------------------------------------------------


def axial_to_xy(q: np.ndarray, r: np.ndarray, s: float) -> tuple[np.ndarray, np.ndarray]:
    """Центры ячеек (остриё вверх) в единицах плоскости при стороне ``s``."""
    q = np.asarray(q, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    return s * SQRT3 * (q + r / 2.0), s * 1.5 * r


def xy_to_axial(x: np.ndarray, y: np.ndarray, s: float) -> tuple[np.ndarray, np.ndarray]:
    """Ближайшая ячейка к точкам (кубическое округление)."""
    qf = (SQRT3 / 3.0 * np.asarray(x) - np.asarray(y) / 3.0) / s
    rf = (2.0 / 3.0 * np.asarray(y)) / s
    sf = -qf - rf
    q, r, t = np.round(qf), np.round(rf), np.round(sf)
    dq, dr, dt = np.abs(q - qf), np.abs(r - rf), np.abs(t - sf)
    fix_q = (dq > dr) & (dq > dt)
    fix_r = ~fix_q & (dr > dt)
    q = np.where(fix_q, -r - t, q)
    r = np.where(fix_r, -q - t, r)
    return q.astype(np.int64), r.astype(np.int64)


def corners(s: float) -> np.ndarray:
    """Шесть вершин ячейки относительно центра (остриё вверх, y вниз), по часовой стрелке."""
    ang = np.deg2rad(np.arange(6) * 60.0 - 30.0)
    return np.column_stack([s * np.cos(ang), s * np.sin(ang)])


def _edge_for_dir(s: float) -> list[tuple[int, int]]:
    """Для каждого направления ``DIRS`` — номера двух вершин общей стороны с соседом."""
    c = corners(s)
    out = []
    for dq, dr in DIRS:
        vx, vy = axial_to_xy(np.array([dq]), np.array([dr]), s)
        dots = c @ np.array([vx[0], vy[0]])
        i, j = np.argsort(dots)[-2:]
        out.append((int(min(i, j)), int(max(i, j))))
    return out


# --- назначение ---------------------------------------------------------------------------------------------


def assign(pts: np.ndarray, cells: np.ndarray, s: float) -> np.ndarray:
    """Номер ячейки для каждой точки: минимум суммы квадратов расстояний, каждая ячейка — одна точка."""
    cx, cy = axial_to_xy(cells[:, 0], cells[:, 1], s)
    cost = (pts[:, 0:1] - cx[None, :]) ** 2 + (pts[:, 1:2] - cy[None, :]) ** 2
    if len(cells) < len(pts):
        raise ValueError(f"site_hexgrid: ячеек {len(cells)} меньше точек {len(pts)}")
    rows, cols = linear_sum_assignment(cost)
    out = np.empty(len(pts), dtype=np.int64)
    out[rows] = cols
    return out


# --- проверки -----------------------------------------------------------------------------------------------


def components(q: np.ndarray, r: np.ndarray, groups: np.ndarray) -> dict[Any, int]:
    """Число связных частей ячеек каждой группы (соседство по стороне шестиугольника)."""
    where = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(q, r, strict=True))}
    groups = np.asarray(groups)
    seen = np.zeros(len(q), dtype=bool)
    out: dict[Any, int] = {}
    for i in range(len(q)):
        if seen[i]:
            continue
        g = groups[i]
        out[g] = out.get(g, 0) + 1
        stack = [i]
        seen[i] = True
        while stack:
            j = stack.pop()
            for dq, dr in DIRS:
                k = where.get((int(q[j]) + dq, int(r[j]) + dr))
                if k is not None and not seen[k] and groups[k] == g:
                    seen[k] = True
                    stack.append(k)
    return out


def knn_preserved(a: np.ndarray, b: np.ndarray, k: int = 10) -> float:
    """Средняя доля k ближайших соседей по точкам ``a``, оставшихся среди k ближайших на карте ``b``."""
    k = min(k, len(a) - 1)
    if k < 1:
        return 1.0
    _, ia = cKDTree(a).query(a, k + 1)
    _, ib = cKDTree(b).query(b, k + 1)
    return float(np.mean([len(set(x[1:]) & set(y[1:])) / k for x, y in zip(ia, ib, strict=True)]))


# --- контуры по ячейкам -------------------------------------------------------------------------------------


def boundary_edges(q: np.ndarray, r: np.ndarray, key: np.ndarray, s: float) -> list[tuple[tuple, tuple]]:
    """Стороны ячеек, где сосед другой группы ``key`` или пусто; каждая сторона — один раз."""
    where = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(q, r, strict=True))}
    cx, cy = axial_to_xy(q, r, s)
    c = corners(s)
    edir = _edge_for_dir(s)
    out = []
    for i in range(len(q)):
        for d, (dq, dr) in enumerate(DIRS):
            j = where.get((int(q[i]) + dq, int(r[i]) + dr))
            if j is not None and (key[j] == key[i] or j < i):
                continue  # внутри группы или уже записана со стороны соседа
            a, b = edir[d]
            pa = (round(cx[i] + c[a, 0], 3), round(cy[i] + c[a, 1], 3))
            pb = (round(cx[i] + c[b, 0], 3), round(cy[i] + c[b, 1], 3))
            out.append((pa, pb))
    return out


def chain(edges: Sequence[tuple[tuple, tuple]]) -> list[list[tuple]]:
    """Склейка сторон в ломаные (меньше команд в пути SVG)."""
    adj: dict[tuple, list[int]] = {}
    for i, (a, b) in enumerate(edges):
        adj.setdefault(a, []).append(i)
        adj.setdefault(b, []).append(i)
    used = np.zeros(len(edges), dtype=bool)
    lines = []
    # начинаем с концов ломаных (нечётная степень), потом замкнутые контуры
    starts = [p for p, v in adj.items() if len(v) % 2 == 1] + list(adj)
    for p in starts:
        while any(not used[i] for i in adj[p]):
            line = [p]
            cur = p
            while True:
                nxt = next((i for i in adj[cur] if not used[i]), None)
                if nxt is None:
                    break
                used[nxt] = True
                a, b = edges[nxt]
                cur = b if a == cur else a
                line.append(cur)
            lines.append(line)
    return lines


def svg_path(lines: Sequence[Sequence[tuple]], scale: float, ox: float, oy: float, nd: int = 1) -> str:
    """Ломаные -> атрибут ``d`` в координатах страницы (абсолютные точки, ``nd`` знаков)."""

    def f(v: float) -> str:
        s = f"{v:.{nd}f}".rstrip("0").rstrip(".")
        return "0" if s in ("-0", "") else s

    parts = []
    for line in lines:
        pts = [(f((x - ox) * scale), f((y - oy) * scale)) for x, y in line]
        head = f"M{pts[0][0]} {pts[0][1]}"
        parts.append(head + "L" + " ".join(f"{a} {b}" for a, b in pts[1:]))
    return "".join(parts)


# --- двухуровневая раскладка: регионы -> ячейки -------------------------------------------------------------


def dorling(
    centers: np.ndarray,
    radii: np.ndarray,
    iters: int,
    anchor: float,
    attract: float = 0.0,
    max_edge: float = np.inf,
) -> np.ndarray:
    """Круги регионов без наложений (Дорлинг, 1996): пары, которые перекрываются, раздвигаются; соседи
    по триангуляции Делоне исходных центров (ребро не длиннее ``max_edge``) стягиваются до касания с силой
    ``attract`` — карта остаётся сплошной; каждый круг тянется к исходному центру с силой ``anchor``.
    Последняя четверть итераций — только раздвижка, без притяжений.
    Детерминирована."""
    p0 = np.asarray(centers, dtype=np.float64)
    p = p0.copy()
    rad = np.asarray(radii, dtype=np.float64)
    edges = np.zeros((0, 2), dtype=np.int64)
    if attract and len(p0) >= 3:
        tri = Delaunay(p0)
        e = {tuple(sorted((int(t[a]), int(t[(a + 1) % 3])))) for t in tri.simplices for a in range(3)}
        edges = np.array(sorted(e), dtype=np.int64)
        edges = edges[np.linalg.norm(p0[edges[:, 0]] - p0[edges[:, 1]], axis=1) <= max_edge]
    for it in range(iters):
        settle = it >= 0.75 * iters  # последняя четверть — только раздвижка: наложений не остаётся
        if len(edges) and not settle:
            d = p[edges[:, 1]] - p[edges[:, 0]]
            dist = np.maximum(np.linalg.norm(d, axis=1), 1e-9)
            gap = np.clip(dist - rad[edges[:, 0]] - rad[edges[:, 1]], 0.0, None)
            f = (attract * gap / dist)[:, None] * d / 2.0
            pull = np.zeros_like(p)
            np.add.at(pull, edges[:, 0], f)
            np.add.at(pull, edges[:, 1], -f)
            p = p + pull
        d = p[None, :, :] - p[:, None, :]
        dist = np.maximum(np.linalg.norm(d, axis=2), 1e-9)
        need = rad[:, None] + rad[None, :]
        over = np.clip(need - dist, 0.0, None)
        np.fill_diagonal(over, 0.0)
        # доля перекрытия, которую снимает каждый из пары, — обратно пропорциональна его площади
        w = rad[None, :] ** 2 / (rad[:, None] ** 2 + rad[None, :] ** 2)
        push = -(over * w / dist)[:, :, None] * d
        p = p + 0.5 * push.sum(axis=1) + (0.0 if settle else anchor) * (p0 - p)
    return p


def region_cells(
    centers: np.ndarray, counts: np.ndarray, s: float, spare: float
) -> tuple[np.ndarray, np.ndarray]:
    """Ячейки регионов: сбалансированное назначение (каждому региону — ровно ``counts`` ячеек) по квадрату
    расстояния от центра его круга. Такие области — дискретные диаграммы мощности, выпуклые и связные
    почти всегда. Возвращает ячейки (q, r) и номер региона каждой."""
    cells_set: set[tuple[int, int]] = set()
    for (x, y), c in zip(centers, counts, strict=True):
        # ячейки круга региона с запасом: ближайшие к центру ceil(c · (1 + spare)) из квадрата вокруг него
        m = int(np.ceil(c * (1.0 + spare)))
        rr = int(np.ceil(np.sqrt(m) * 1.2)) + 2
        q0, r0 = xy_to_axial(np.array([x]), np.array([y]), s)
        dq, dr = np.meshgrid(np.arange(-rr, rr + 1), np.arange(-rr, rr + 1))
        q = q0[0] + dq.ravel()
        r = r0[0] + dr.ravel()
        hx, hy = axial_to_xy(q, r, s)
        near = np.argsort((hx - x) ** 2 + (hy - y) ** 2, kind="stable")[:m]
        cells_set |= {(int(q[i]), int(r[i])) for i in near}
    n = int(counts.sum())
    while len(cells_set) < n:  # круги перекрылись: добавить кольцо вокруг уже выбранных ячеек
        cells_set |= {(a + dq, b + dr) for a, b in cells_set for dq, dr in DIRS}
    cells = np.array(sorted(cells_set), dtype=np.int64)
    cx, cy = axial_to_xy(cells[:, 0], cells[:, 1], s)
    slot_region = np.repeat(np.arange(len(counts)), counts)
    c = centers[slot_region]
    # квадрат расстояния без весов: при ограничении «ровно counts ячеек» оптимум — дискретная диаграмма
    # мощности (выпуклые области); деление на радиус дало бы мультипликативные веса и острова
    cost = (cx[None, :] - c[:, 0:1]) ** 2 + (cy[None, :] - c[:, 1:2]) ** 2
    rows, cols = linear_sum_assignment(cost)
    return cells[cols], slot_region[rows]


def fit_within(pts: np.ndarray, cells: np.ndarray, s: float) -> np.ndarray:
    """Точки одного региона -> его ячейки: точки приводятся к центру и разбросу ячеек (сохраняется
    взаимное расположение), затем назначение по квадрату расстояния. Возвращает номер ячейки для точки."""
    cx, cy = axial_to_xy(cells[:, 0], cells[:, 1], s)
    target = np.column_stack([cx, cy])
    if len(pts) == 1:
        return np.array([0])
    src = pts - pts.mean(axis=0)
    sd_s = np.sqrt((src**2).sum(axis=1).mean())
    sd_t = np.sqrt(((target - target.mean(axis=0)) ** 2).sum(axis=1).mean())
    src = src * (sd_t / max(sd_s, 1e-9)) + target.mean(axis=0)
    return assign(src, cells, s)


# --- сборка -------------------------------------------------------------------------------------------------


@dataclass
class HexResult:
    """Назначение ячеек и его качество."""

    ids: np.ndarray
    q: np.ndarray
    r: np.ndarray
    s: float  # сторона ячейки, км
    shift_km: np.ndarray
    groups: np.ndarray
    stats: dict[str, Any] = field(default_factory=dict)

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame({"territory_id": self.ids, "hq": self.q, "hr": self.r, "shift_km": self.shift_km})


def build_grid(
    ids: Sequence[int],
    x_km: Sequence[float],
    y_km: Sequence[float],
    groups: Sequence[Any],
    params: Mapping[str, Any],
) -> HexResult:
    """Назначение ячеек по параметрам ``site.build.hex``. ``y_km`` — вверх (как ``y_aea``); внутри — вниз.

    Два уровня: (1) круги регионов площадью «число МО × площадь ячейки» у медианы точек региона, раздвинутые
    без наложений (``dorling``); (2) ячейки — регионам сбалансированным назначением (``region_cells``),
    внутри региона — МО по взаимному расположению (``fit_within``). Так каждый регион — связное пятно,
    а соседство МО внутри региона сохраняется."""
    ids = np.asarray(ids)
    if len(set(ids.tolist())) != len(ids):
        raise ValueError("site_hexgrid: повторяются territory_id")
    xy = np.column_stack([np.asarray(x_km, dtype=np.float64), -np.asarray(y_km, dtype=np.float64)])
    s = float(params["cell_km"])
    groups = np.asarray(groups)
    codes, uniq = pd.factorize(groups)
    counts = np.bincount(codes)
    cen = np.array([np.median(xy[codes == g], axis=0) for g in range(len(uniq))])
    area = 1.5 * SQRT3 * s * s
    rad = np.sqrt(counts * area / np.pi)
    moved = dorling(
        cen,
        rad * float(params.get("radius_factor", 1.0)),  # < 1: круги чуть перекрываются, регионы смыкаются
        int(params.get("iters", 400)),
        float(params.get("anchor", 0.05)),
        float(params.get("attract", 0.0)),
        float(params.get("max_edge_km", np.inf)),
    )
    cells, reg = region_cells(moved, counts, s, float(params["spare"]))
    q = np.zeros(len(ids), dtype=np.int64)
    r = np.zeros(len(ids), dtype=np.int64)
    for g in range(len(uniq)):
        idx = np.flatnonzero(codes == g)
        cg = cells[reg == g]
        k = fit_within(xy[idx], cg, s)
        q[idx], r[idx] = cg[k, 0], cg[k, 1]
    if len({(a, b) for a, b in zip(q.tolist(), r.tolist(), strict=True)}) != len(ids):
        raise ValueError("site_hexgrid: две точки в одной ячейке")
    cx, cy = axial_to_xy(q, r, s)
    shift = np.hypot(cx - xy[:, 0], cy - xy[:, 1])
    comp = components(q, r, groups)
    split = {g: c for g, c in comp.items() if c > 1}
    stats = {
        "n_cells": int(len(ids)),
        "cell_km": s,
        "spare": float(params["spare"]),
        "shift_median_km": float(np.median(shift)),
        "shift_p90_km": float(np.percentile(shift, 90)),
        "shift_max_km": float(shift.max()),
        "groups": int(len(comp)),
        "groups_split": int(len(split)),
        "split": {str(g): int(c) for g, c in sorted(split.items(), key=lambda t: str(t[0]))},
        "knn10_preserved": knn_preserved(xy, np.column_stack([cx, cy]), 10),
    }
    log.info(
        "site: ячейки %d (сторона %.0f км, запас %.0f%%): смещение медиана %.0f км, 90%% — %.0f, макс. %.0f; "
        "регионов разорвано %d из %d; сохранено ближайших соседей %.2f",
        len(ids),
        s,
        100 * float(params["spare"]),
        stats["shift_median_km"],
        stats["shift_p90_km"],
        stats["shift_max_km"],
        stats["groups_split"],
        stats["groups"],
        stats["knn10_preserved"],
    )
    return HexResult(ids=ids, q=q, r=r, s=s, shift_km=shift, groups=groups, stats=stats)


# --- координаты страницы и SVG ------------------------------------------------------------------------------


def page_axial(q: np.ndarray, r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Осевые модуля (y вниз) -> осевые страницы ``hq``, ``hr`` (``y = y0 − size·1,5·hr``)."""
    return np.asarray(q) + np.asarray(r), -np.asarray(r)


@dataclass(frozen=True)
class PageGrid:
    """Решётка на странице: сторона ячейки ``size`` в единицах SVG, сдвиг ``x0``, ``y0``, размер холста."""

    size: float
    x0: float
    y0: float
    width: float
    height: float

    def center(self, hq: np.ndarray, hr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        hq = np.asarray(hq, dtype=np.float64)
        hr = np.asarray(hr, dtype=np.float64)
        return self.x0 + self.size * SQRT3 * (hq + hr / 2.0), self.y0 - self.size * 1.5 * hr

    def as_dict(self) -> dict[str, float]:
        return {
            "size": self.size,
            "x0": round(self.x0, 2),
            "y0": round(self.y0, 2),
            "width": round(self.width, 1),
            "height": round(self.height, 1),
            "orient": "pointy",
        }


def page_grid(hq: np.ndarray, hr: np.ndarray, size: float, margin: float) -> PageGrid:
    """Сдвиг так, чтобы ячейки с полями ``margin`` начинались у (0, 0)."""
    x = size * SQRT3 * (np.asarray(hq) + np.asarray(hr) / 2.0)
    y = -size * 1.5 * np.asarray(hr)
    w_half, h_half = size * SQRT3 / 2.0, size
    x0 = margin + w_half - float(x.min())
    y0 = margin + h_half - float(y.min())
    width = float(x.max() - x.min()) + 2 * (margin + w_half)
    height = float(y.max() - y.min()) + 2 * (margin + h_half)
    return PageGrid(size=size, x0=x0, y0=y0, width=width, height=height)


def _f(v: float, nd: int = 1) -> str:
    s = f"{v:.{nd}f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def hex_path(xs: Sequence[float], ys: Sequence[float], size: float) -> str:
    """Атрибут ``d`` с шестиугольниками (остриё вверх) в точках ``xs``, ``ys``. Короче абсолютных координат:
    переход к следующей ячейке — относительный ``m`` от начала предыдущей (после ``z`` точка возвращается
    к началу контура); разности берутся между уже округлёнными координатами, поэтому ошибка не копится."""
    w, h = _f(size * SQRT3 / 2.0), _f(size / 2.0)
    v = _f(size)
    body = f"l{w} {h}v{v}l-{w} {h}-{w}-{h}v-{v}z"
    top = np.round(np.asarray(ys, dtype=float) - size, 1)
    pts = sorted(zip(top, np.round(np.asarray(xs, dtype=float), 1), strict=True))
    out = []
    px = py = None
    for y, x in pts:
        if px is None:
            out.append(f"M{_f(x)} {_f(y)}{body}")
        else:
            out.append(f"m{_f(x - px)} {_f(y - py)}{body}")
        px, py = x, y
    return "".join(out)


def borders(hq: np.ndarray, hr: np.ndarray, key: Sequence[Any], grid: PageGrid) -> str:
    """Контур групп ``key`` по сторонам ячеек (соседи другой группы или пусто) — путь SVG страницы."""
    r = -np.asarray(hr)  # обратно к осевым модуля: r = −hr, q = hq − r
    q = np.asarray(hq) - r
    edges = boundary_edges(q, r, np.asarray(key), grid.size)
    lines = chain(edges)
    return svg_path(lines, 1.0, -grid.x0, -grid.y0)
