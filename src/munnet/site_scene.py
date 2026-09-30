"""Объёмная карта первого экрана лендинга (``docs/landing_spec.md``, §4.7): данные сцены ``scene.json``.

Всё, что страница показывает на объёмной карте, считает этот модуль; ``landing3d.js`` только рисует:

- **высота ячейки** — ``exp(clr_rel_cafe)``: доля трат на кафе и рестораны относительно своего региона
  (1 — как в регионе) из тех же значений признаков, что видела кластеризация и что стоят в профиле типов
  (``landing.clustering_values``: корзина B до стандартизации), а не из годовых окон. У МО без типа высоты
  нет —
  плоская серая ячейка;
- **положение на карте** — центр ячейки карты равных ячеек (та же решётка, что у статичной SVG);
- **острова «по типам»** — у каждого типа своя шестиугольная спираль ячеек той же решётки; острова идут слева
  направо в порядке легенды (правило ``order_in_legends``), МО без типа — последним; внутри острова ячейки
  от центра к краю по убыванию высоты (при равенстве — по номеру МО);
- **подписи островов** — название типа, число ячеек и доля кафе у медианы типа словами.

Координаты — в единицах SVG карты (``hexgrid.grid``): x вправо, y вниз (север сверху).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from munnet import style

SQRT3 = math.sqrt(3.0)
CAFE = "clr_rel_cafe"
NB = chr(0xA0)
# числа словами для текстов первого экрана: «один из четырёх типов», «Ниже — четыре типа»
NUM_WORDS = {
    2: ("два", "двух"),
    3: ("три", "трёх"),
    4: ("четыре", "четырёх"),
    5: ("пять", "пяти"),
    6: ("шесть", "шести"),
    7: ("семь", "семи"),
    8: ("восемь", "восьми"),
}
# направления спирали в осевых координатах (q, r) острой ячейки — как обход колец шестиугольника
_DIRS = ((1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1))


def ratio_words(r: float | None, near: float = 0.4) -> str:
    """Доля кафе относительно региона словами (та же формулировка, что у подсказки в ``landing3d.js``):
    |ln r| ≥ ``near`` — «в 1,5 раза больше / меньше», иначе в процентах; 0% — «как в регионе»."""
    if r is None or not np.isfinite(r) or r <= 0:
        return "нет данных"
    if abs(math.log(r)) >= near:
        k = r if r > 1 else 1 / r
        return f"в{NB}{style.fmt_num(k, 1)} раза {'больше' if r > 1 else 'меньше'}, чем в{NB}регионе"
    p = round((r - 1) * 100)
    if p == 0:
        return f"как в{NB}регионе"
    return f"на{NB}{abs(p)}% {'больше' if p > 0 else 'меньше'}, чем в{NB}регионе"


def spiral(n: int) -> list[tuple[int, int]]:
    """Первые ``n`` ячеек шестиугольной спирали (q, r): центр, затем кольца 1, 2, … по обходу."""
    out = [(0, 0)]
    k = 1
    while len(out) < n:
        q, r = -k, k
        for dq, dr in _DIRS:
            for _ in range(k):
                if len(out) >= n:
                    break
                out.append((q, r))
                q, r = q + dq, r + dr
        k += 1
    return out[:n]


def rings(n: int) -> int:
    """Число колец спирали, в которое помещаются ``n`` ячеек (1 + 3k(k + 1) ≥ n)."""
    k = 0
    while 1 + 3 * k * (k + 1) < n:
        k += 1
    return k


def cafe_heights(values: pd.DataFrame | None, node_ids: Sequence[int]) -> pd.Series:
    """``exp(clr_rel_cafe)`` у узлов сети (индекс — номер узла); нет значений — пустая серия."""
    if values is None or CAFE not in values.columns:
        return pd.Series(dtype=float)
    v = values[CAFE].reindex([int(i) for i in node_ids])
    return np.exp(v.astype(float)).dropna()


def island_layout(
    cells: pd.DataFrame, order: Sequence[int], size: float, gap: float, cx: float, cy: float
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Острова по типам: ``cells`` — ``id``, ``t`` (0 — без типа), ``h`` (высота или NaN).

    Возвращает (``id``, ``ix``, ``iy``) каждой ячейки и описания островов (тип, центр, радиус, число ячеек)
    в порядке ``order`` (затем 0). Ряд островов центрирован по ``cx``; у всех островов центр на высоте
    ``cy``."""
    seq = [int(t) for t in order] + [0]
    groups = []
    for t in seq:
        g = cells[cells["t"] == t].copy()
        g["_h"] = g["h"].fillna(-np.inf)
        g = g.sort_values(["_h", "id"], ascending=[False, True])
        groups.append((t, g))
    step = size * SQRT3  # расстояние между центрами соседних ячеек
    rads = [step * (rings(len(g)) + 0.6) for _, g in groups]
    nonempty = [i for i, (_, g) in enumerate(groups) if len(g)]
    total = sum(2 * rads[i] for i in nonempty) + gap * max(len(nonempty) - 1, 0)
    x = cx - total / 2
    pos, islands = [], []
    for i in nonempty:
        t, g = groups[i]
        icx = x + rads[i]
        for tid, (q, r) in zip(g["id"], spiral(len(g)), strict=True):
            pos.append((int(tid), icx + step * (q + r / 2), cy + size * 1.5 * r))
        islands.append({"t": t, "x": icx, "y": cy, "rad": rads[i], "n": int(len(g))})
        x += 2 * rads[i] + gap
    return pd.DataFrame(pos, columns=["id", "ix", "iy"]), islands


def build_scene(
    mo: pd.DataFrame,
    grid: Any,
    values: pd.DataFrame | None,
    order: Sequence[int],
    names: Mapping[str, str],
    colors: Mapping[str, str],
    texts: Mapping[str, str],
    params: Mapping[str, Any],
) -> dict | None:
    """``scene.json``: ячейки (карта и острова, высота, тип) и острова с подписями. Нет значений признаков
    кластеризации (``--demo`` без ``outputs/cluster/final.json``) — None: первый экран остаётся статичной
    картой."""
    cells = mo[mo["role"].isin(["territorial", "city", "untyped"])]
    nodes = cells[cells["role"] != "untyped"]
    h = cafe_heights(values, nodes["id"])
    if h.empty:
        return None
    missing = sorted(set(int(i) for i in nodes["id"]) - set(int(i) for i in h.index))
    if missing:
        raise ValueError(f"site: нет {CAFE} у узлов {missing[:5]} (всего {len(missing)})")
    x, y = grid.center(cells["hq"], cells["hr"])
    df = pd.DataFrame(
        {
            "id": cells["id"].astype(int).to_numpy(),
            "t": cells["t"].fillna(0).astype(int).to_numpy(),
            "x": x,
            "y": y,
        }
    )
    df["h"] = df["id"].map(h)
    lay, islands = island_layout(
        df[["id", "t", "h"]],
        order,
        float(grid.size),
        float(params["gap"]),
        float(grid.width) / 2,
        float(grid.height) / 2 + float(params["dy"]),
    )
    df = df.merge(lay, on="id", how="left", validate="one_to_one")
    for isl in islands:
        t = isl["t"]
        if t:
            med = float(np.exp(np.median(np.log(df.loc[df["t"] == t, "h"].to_numpy(dtype=float)))))
            isl |= {
                "name": names.get(str(t), f"Тип {t}"),
                "color": colors.get(str(t)),
                "h_med": round(med, 3),
                "note": texts["island_note"].replace("{ratio}", ratio_words(med)),
            }
        else:
            isl |= {"name": texts["untyped"], "color": None, "h_med": None, "note": texts["untyped_note"]}
        for k in ("x", "y", "rad"):
            isl[k] = round(float(isl[k]), 1)
    r1 = lambda s: [round(float(v), 1) for v in s]  # noqa: E731
    return {
        "unit": "единицы SVG карты равных ячеек; x вправо, y вниз",
        "height": "exp(clr_rel_cafe) — доля трат на кафе и рестораны относительно своего региона, "
        "1 — как в регионе",
        "grid": grid.as_dict(),
        "order": [int(t) for t in order] + [0],
        "cells": {
            "id": [int(i) for i in df["id"]],
            "t": [int(t) for t in df["t"]],
            "h": [None if pd.isna(v) else round(float(v), 3) for v in df["h"]],
            "x": r1(df["x"]),
            "y": r1(df["y"]),
            "ix": r1(df["ix"]),
            "iy": r1(df["iy"]),
        },
        "islands": islands,
    }


def num_words(n: int) -> tuple[str, str]:
    """Число словами (именительный, родительный); вне словаря — цифрами."""
    return NUM_WORDS.get(int(n), (str(n), str(n)))
