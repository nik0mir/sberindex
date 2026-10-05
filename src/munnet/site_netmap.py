"""Схема «Сеть корзин целиком» в главе «Типы» (порция 6n, решение участника 05.10.2026;
docs/landing_spec.md §4.3).

Раскладка ``graph_fr`` (Фрухтерман — Рейнгольд основной сети ``basket_dist``, ``munnet.site_layout``)
показывается как дополнительная иллюстрация **всегда** — независимо от правила
``site.similarity_layout``. Само правило и замороженные ключи не меняются: интерактивная перестановка карты
по нему по-прежнему включается, только если доля сохранённых соседей не меньше ``min_preserved``. Координаты
и доля берутся из того же ``LayoutResult``, что и выбор раскладки (тот же seed, те же итерации), поэтому числа
на схеме и в ``checks.json`` совпадают.

Здесь только:

- поворот: длинная ось облака — по горизонтали (главные оси координат, знак — по правилу, без типов);
  поворот и общий масштаб осей расстояний не меняют, осей у схемы нет;
- целые координаты: ``x`` — 0…``SCALE``, ``y`` — 0…``h`` в том же масштабе;
- ``k`` самых похожих каждого узла по основной сети (наибольший вес связи, при равенстве — меньший номер) —
  индексы в списке узлов, плоским массивом; координаты и индексы записаны компактно (``encode``);
- тексты (числа — кодом) и HTML блока; рисует страница (canvas в ``landing.js``), ничего не считая.

Оси, стрелки и подписи порядка на схеме — только при T1 ``confirmed`` (``similarity_layout.order_marks_if``);
этот блок их не рисует ни при каком исходе.
"""

from __future__ import annotations

import html
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from munnet import style

SCALE = 1000  # длинная ось схемы на странице: целые 0…SCALE
METHOD = "graph_fr"
# Компактная запись целых 0…4095 двумя знаками (старший, младший) — координаты и соседи занимают вдвое меньше
# места в index.html (схема встроена, как остальные данные; страница декодирует её одной строкой,
# ничего не считая).
# −1 (нет соседа) — «~~».
ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-_"
CODE_MAX = len(ALPHABET) ** 2 - 1

Esc = Callable[[Any], str]


def _fill(tpl: str, vals: Mapping[str, Any]) -> str:
    out = str(tpl)
    for k, v in vals.items():
        out = out.replace("{" + k + "}", str(v))
    return out


def _e(s: Any) -> str:
    return html.escape(str(s), quote=True)


# --- геометрия ----------------------------------------------------------------------------------------------


def orient(coords: np.ndarray) -> np.ndarray:
    """Поворот к главным осям облака: первая — по горизонтали. Знак первой оси — наибольшая по модулю
    компонента её направления положительна (как ``site_layout.layout_pca``); вторая ось — первая, повёрнутая
    на 90° против часовой стрелки (собственный поворот, без отражения). Расстояния не меняются."""
    c = np.asarray(coords, dtype=np.float64)
    cc = c - c.mean(axis=0)
    _, _, vt = np.linalg.svd(cc, full_matrices=False)
    v = vt[0]
    if v[np.abs(v).argmax()] < 0:
        v = -v
    rot = np.array([v, [-v[1], v[0]]])
    return cc @ rot.T


def to_grid(coords: np.ndarray, scale: int = SCALE) -> tuple[np.ndarray, int]:
    """Целые координаты с общим масштабом осей: ``x`` — 0…``scale`` по длинной оси, ``y`` — 0…``h``
    (``y`` растёт вниз, как на canvas). Возвращает (n × 2, h)."""
    c = np.asarray(coords, dtype=np.float64)
    lo, hi = c.min(axis=0), c.max(axis=0)
    span = float(hi[0] - lo[0])
    if span <= 0:
        return np.zeros(c.shape, dtype=np.int64), 0
    u = (c - lo) / span * scale
    u[:, 1] = (hi[1] - lo[1]) / span * scale - u[:, 1]
    h = int(np.ceil((hi[1] - lo[1]) / span * scale))
    return np.clip(np.rint(u), 0, [scale, h]).astype(np.int64), h


# --- соседи -------------------------------------------------------------------------------------------------


def main_edges_weighted(processed: Path) -> pd.DataFrame:
    """Рёбра основной сети ``basket_dist`` (``is_main``) с весами из ``network_edges``."""
    e = pd.read_parquet(
        Path(processed) / "network_edges.parquet",
        columns=["rule", "source", "target", "weight", "is_main"],
        filters=[("rule", "==", "basket_dist"), ("is_main", "==", True)],
    )
    if e.empty:
        raise ValueError("site_netmap: в network_edges нет основной сети basket_dist")
    return e[["source", "target", "weight"]].reset_index(drop=True)


def top_neighbors(ids: np.ndarray, edges: pd.DataFrame, k: int) -> np.ndarray:
    """``k`` самых похожих каждого узла: соседи по сети с наибольшим весом связи (вес убывает с расстоянием
    корзин, поэтому это k ближайших по корзине), при равенстве — меньший номер. Индексы в ``ids``,
    матрица n × k; если соседей меньше k — дополнено −1. Ребро к узлу вне ``ids`` — ошибка."""
    pos = pd.Index(np.asarray(ids, dtype=np.int64))
    s = pos.get_indexer(edges["source"].to_numpy(dtype=np.int64))
    t = pos.get_indexer(edges["target"].to_numpy(dtype=np.int64))
    if (s < 0).any() or (t < 0).any():
        raise ValueError("site_netmap: ребро сети ведёт к узлу вне схемы")
    w = edges["weight"].to_numpy(dtype=np.float64)
    both = pd.DataFrame(
        {
            "a": np.concatenate([s, t]),
            "b": np.concatenate([t, s]),
            "w": np.concatenate([w, w]),
            "id": np.concatenate([pos.to_numpy()[t], pos.to_numpy()[s]]),
        }
    )
    both = both[both["a"] != both["b"]].drop_duplicates(["a", "b"])
    both = both.sort_values(["a", "w", "id"], ascending=[True, False, True], kind="mergesort")
    out = np.full((len(pos), k), -1, dtype=np.int64)
    for a, g in both.groupby("a", sort=True):
        b = g["b"].to_numpy()[:k]
        out[int(a), : len(b)] = b
    return out


# --- компактная запись --------------------------------------------------------------------------------------


def encode(values: Sequence[int]) -> str:
    """Целые 0…4095 (и −1) — по два знака ``ALPHABET`` (−1 — «~~»)."""
    out = []
    for v in values:
        v = int(v)
        if v == -1:
            out.append("~~")
            continue
        if not 0 <= v <= CODE_MAX:
            raise ValueError(f"site_netmap: {v} вне 0…{CODE_MAX}")
        out.append(ALPHABET[v >> 6] + ALPHABET[v & 63])
    return "".join(out)


def decode(s: str) -> list[int]:
    """Обратно к ``encode`` (так же декодирует landing.js)."""
    pos = {c: i for i, c in enumerate(ALPHABET)}
    return [-1 if s[i] == "~" else pos[s[i]] * 64 + pos[s[i + 1]] for i in range(0, len(s), 2)]


# --- данные страницы ----------------------------------------------------------------------------------------


def build(res: Any, edges: pd.DataFrame, k: int) -> dict:
    """``netmap.json``: координаты ``graph_fr`` после поворота, ``k`` самых похожих и числа подписи — доля
    сохранённых соседей ``res.preserved['graph_fr']`` (та же, что в выборе раскладки), случайная схема,
    порог."""
    ids = np.asarray(res.ids, dtype=np.int64)
    xy, h = to_grid(orient(res.coords[METHOD]))
    nb = top_neighbors(ids, edges, k)
    spec = res.spec
    info = dict(getattr(res, "info", {}) or {})
    return {
        "method": METHOD,
        "ids": [int(i) for i in ids],
        "x": encode(xy[:, 0]),
        "y": encode(xy[:, 1]),
        "w": SCALE,
        "h": int(h),
        "k": int(k),
        "nb": encode(nb.ravel()),
        "preserved": round(float(res.preserved[METHOD]), 6),
        "random": round(float(res.random_preserved), 6),
        "min_preserved": float(spec.min_preserved),
        "chosen": res.chosen,
        "n_nodes": int(len(ids)),
        "n_edges": int(res.n_edges),
        "niter": int(spec.niter),
        "seed": int(info.get("seed", 0)),
    }


def check(data: Mapping, mo_ids: Sequence[int]) -> list[str]:
    """Каждый узел схемы — МО страницы с типом (территориальный узел или узел-город); индексы соседей
    в пределах списка узлов."""
    bad = []
    known = {int(i) for i in mo_ids}
    miss = [i for i in data["ids"] if int(i) not in known]
    if miss:
        bad.append(f"узлов схемы нет среди МО с типом: {len(miss)} (например, {miss[:3]})")
    n = len(data["ids"])
    nb = decode(data["nb"])
    if len(nb) != n * int(data["k"]) or any(v < -1 or v >= n for v in nb):
        bad.append("соседи схемы: не n × k или индекс вне списка узлов")
    if len(decode(data["x"])) != n or len(decode(data["y"])) != n:
        bad.append("координаты схемы: длина не равна числу узлов")
    return bad


# --- тексты и HTML ------------------------------------------------------------------------------------------


def texts(tx: Mapping | None, data: Mapping | None) -> dict[str, str] | None:
    """Тексты блока из ``site.build.texts.netmap``; числа — из ``netmap.json``. Нет текстов или данных —
    None."""
    if not tx or not data:
        return None
    k = int(data["k"])
    vals = {
        "k": k,
        "min_n": int(round(float(data["min_preserved"]) * k)),
        "min_pct": style.fmt_pct(float(data["min_preserved"]), 0),
        "preserved": style.fmt_pct(float(data["preserved"])),
        "random": style.fmt_pct(float(data["random"])),
        "n_nodes": style.fmt_num(int(data["n_nodes"])),
        "n_edges": style.fmt_num(int(data["n_edges"])),
        "niter": style.fmt_num(int(data["niter"])),
        "seed": int(data["seed"]),
    }
    return {key: _fill(str(v), vals) for key, v in tx.items() if isinstance(v, str)}


def strings(t: Mapping[str, str] | None) -> list[str]:
    return [str(v) for v in (t or {}).values()]


def strip_order(data: Mapping, type_of: Mapping[int, Any]) -> list[int]:
    """6o (check-ux 05.10): типы в порядке их участков на полосе — по медиане координаты x их точек (на
    телефоне x идёт сверху вниз). Легенда схемы идёт в том же порядке. Нет точек типа — тип в конце."""
    xs = decode(data["x"])
    by: dict[int, list[int]] = {}
    for i, x in zip(data["ids"], xs, strict=True):
        t = type_of.get(int(i))
        if t is not None and t == t:  # без NaN
            by.setdefault(int(t), []).append(int(x))
    return sorted(by, key=lambda t: (sorted(by[t])[len(by[t]) // 2], t))


def html_block(
    t: Mapping[str, str],
    data: Mapping,
    types: Sequence[Mapping],
    names: Mapping[str, str],
    shapes: Mapping[str, str],
    esc: Esc,
    src: str,
) -> str:
    """Блок схемы для главы «Типы»: заголовок-вывод, подводка, легенда-кнопки, холст (рисует ``landing.js``),
    подсказки, подпись, «Как построена схема», источник и ссылка на таблицу (6q). Без осей, стрелок
    и подписей порядка."""
    keys = []
    for ty in types:
        tt = int(ty["t"])
        nm = str(names.get(str(tt)) or f"Тип {tt}")
        keys.append(
            f'<button type="button" class="net-key" data-t="{tt}" aria-pressed="false">'
            f'<i class="fig fig-t{tt}" aria-hidden="true">{_e(shapes.get(str(tt), ""))}</i>'
            f'<span class="nk-name">{esc(nm)}</span> '
            f"<small>{_e(style.fmt_num(int(ty.get('size') or 0)))}</small>"
            "</button>"
        )
    attrs = " ".join(
        f'data-{a}="{_e(t.get(key, ""))}"'
        for a, key in (
            ("open", "tip_open"),
            ("open-touch", "panel_open"),
            ("panel-empty", "panel_empty"),
            ("sel", "selected"),
        )  # fmt: skip
    )
    return (
        f'<div class="net-block" id="netmap" role="group" aria-labelledby="netmap-title" {attrs}>'
        f'<h3 id="netmap-title">{esc(t["title"])}</h3>'
        f'<p class="net-lead">{esc(t["lead"])}.</p>'
        f'<p class="net-hint net-hint-mouse">{esc(t["hint"])}.</p>'
        f'<p class="net-hint net-hint-touch">{esc(t["hint_touch"])}.</p>'
        '<div class="net-stage">'
        f'<canvas class="net-canvas" role="img" aria-label="{_e(t["aria"])}"></canvas>'
        '<div class="net-tip" aria-hidden="true" hidden></div>'
        # легенда и панель касания: на широком экране легенда над схемой, на узком — колонкой справа от полосы
        '<div class="net-side">'
        f'<div class="net-keys" role="group" aria-label="{_e(t["legend_label"])}">{"".join(keys)}</div>'
        f'<div class="net-panel" aria-live="polite"><p class="np-empty">{esc(t["panel_empty"])}</p></div>'
        "</div></div>"
        f'<p class="nojs note">{esc(t["nojs"])}.</p>'
        f'<p class="note net-cap">{esc(t["caption"])}.</p>'
        f'<details class="how net-how"><summary>{esc(t["how_label"])}</summary>'
        f"<p>{esc(t['how'])}.</p></details>"
        f'<p class="source">{esc(src)}</p>'
        # 6q (совет 06.10, check-ux): ссылка на таблицу — после подписи и источника схемы (клавиатура, диктор)
        f'<p class="net-table"><a href="#all-mo">{esc(t["to_table"])}</a></p>'
        "</div>"
    )
