"""Дорожные расстояния между узлами из ``connection.parquet`` СберИндекса.

В источнике ``highway`` записан один раз на пару (x → y), ``railway`` — в обе стороны (2 × 1 319 314 строк):
пары приводятся к виду (min, max) и дедуплицируются до симметризации, иначе железнодорожная пара
считается дважды. Расстояние узла-города (Москва, Петербург) до МО — среднее расстояний его районов
с весами населения (``pop_mean``: ожидаемое расстояние от случайного жителя города; так же с весами
населения собраны и траты узла-города) или до ближайшего района (``min``). Пары районов одного города
исчезают: внутри узла рёбер нет.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

CITY_DISTANCE: tuple[str, ...] = ("pop_mean", "min")


def road_pairs(conn: pd.DataFrame, kind: str = "highway") -> pd.DataFrame:
    """Пары (a, b), a < b, с расстоянием, км: строки вида ``kind``, без петель, дубли направлений
    схлопнуты (расстояние — минимум записей пары; у расходящихся записей — предупреждение в логе)."""
    d = conn.loc[conn["type"].astype(str) == kind, ["territory_id_x", "territory_id_y", "distance"]]
    x = d["territory_id_x"].to_numpy(dtype=np.int64)
    y = d["territory_id_y"].to_numpy(dtype=np.int64)
    out = pd.DataFrame({"a": np.minimum(x, y), "b": np.maximum(x, y), "distance": d["distance"].to_numpy()})
    out = out.loc[out["a"] != out["b"]]
    g = out.groupby(["a", "b"], sort=True)["distance"]
    res = g.min().rename("distance").reset_index()
    spread = (g.max() - g.min()).to_numpy()
    n_dup = len(out) - len(res)
    if n_dup:
        log.info("дороги (%s): %d повторных записей пар схлопнуто (пары в обе стороны)", kind, n_dup)
    if (spread > 1e-6).any():
        log.warning(
            "дороги (%s): у %d пар записи разных направлений расходятся", kind, int((spread > 1e-6).sum())
        )
    return res


def node_distance_matrix(
    pairs: pd.DataFrame,
    node_ids: np.ndarray,
    members: pd.DataFrame,
    weights: pd.Series,
    how: str = "pop_mean",
) -> np.ndarray:
    """Матрица расстояний узлов (n × n, км; пропуск — NaN, диагональ — NaN).

    ``members`` — ``territory_id`` МО -> ``node_id``; ``weights`` — население МО (для ``pop_mean``;
    пропуск веса — медиана весов МО того же узла). Для узла из одного МО формула даёт его расстояние.
    """
    if how not in CITY_DISTANCE:
        raise ValueError(f"network.city_distance = {how!r}: допустимы {list(CITY_DISTANCE)}")
    node_ids = np.asarray(node_ids, dtype=np.int64)
    pos = pd.Series(np.arange(len(node_ids)), index=node_ids)
    m = members.dropna(subset=["node_id"])
    m = m.loc[m["node_id"].astype("int64").isin(node_ids)]
    mo_node = pd.Series(
        m["node_id"].astype("int64").to_numpy(), index=m["territory_id"].astype("int64").to_numpy()
    )
    w = weights.reindex(mo_node.index).astype("float64")
    fill = w.groupby(mo_node.to_numpy()).transform("median")
    w = w.fillna(fill).fillna(1.0)

    p = pairs.loc[pairs["a"].isin(mo_node.index) & pairs["b"].isin(mo_node.index)]
    u = pos.reindex(mo_node.reindex(p["a"].to_numpy()).to_numpy()).to_numpy()
    v = pos.reindex(mo_node.reindex(p["b"].to_numpy()).to_numpy()).to_numpy()
    frame = pd.DataFrame(
        {
            "u": np.minimum(u, v),
            "v": np.maximum(u, v),
            "d": p["distance"].to_numpy(dtype=np.float64),
            "w": w.reindex(p["a"].to_numpy()).to_numpy() * w.reindex(p["b"].to_numpy()).to_numpy(),
        }
    )
    frame = frame.loc[frame["u"] != frame["v"]]  # районы одного узла-города
    if how == "min":
        agg = frame.groupby(["u", "v"])["d"].min()
    else:
        frame["wd"] = frame["w"] * frame["d"]
        s = frame.groupby(["u", "v"])[["wd", "w"]].sum()
        agg = s["wd"] / s["w"]
    n = len(node_ids)
    D = np.full((n, n), np.nan)
    uu = agg.index.get_level_values(0).to_numpy()
    vv = agg.index.get_level_values(1).to_numpy()
    D[uu, vv] = agg.to_numpy()
    D[vv, uu] = agg.to_numpy()
    np.fill_diagonal(D, np.nan)
    return D


def gravity_log(D: np.ndarray, pop: np.ndarray, beta: float, min_km: float) -> np.ndarray:
    """ln S_ij = ln P_i + ln P_j − β·ln max(d_ij, d_min); пропуск расстояния или населения — NaN."""
    lp = np.log(np.asarray(pop, dtype=np.float64))
    with np.errstate(invalid="ignore", divide="ignore"):
        S = lp[:, None] + lp[None, :] - beta * np.log(np.fmax(D, min_km))
    S[~np.isfinite(D)] = np.nan
    np.fill_diagonal(S, np.nan)
    return S.astype(np.float32)
