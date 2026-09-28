"""Входы окон: периоды, сеть G окна и признаки X окна — функциями этапов network и cluster.

- **Периоды** — 13 скользящих 12-месячных окон этапа network (``network.compare.window_list``), нечётные
  и чётные месяцы каждого года (шум) и все 24 месяца (опора: итоговое разбиение этапа cluster).
- **G окна** — правило и kNN итогового разбиения (``final.json`` → ``inputs.graph``): сходство
  ``network.compute.similarity`` на месяцах окна, рёбра ``network.graph.knn_edges``, матрица
  ``clustering.inputs.graph_matrix``. Сверка: для 13 скользящих окон G совпадает с ``network_windows``.
- **X окна** — признаки места ``place_year`` (как на 24 месяцах) и уровень трат окна
  ``clustering.inputs.window_level``; пропуски и масштаб — ``clustering.inputs.build_X``.
- **Пространство гибрида для альтернатив** (``HybridSpace``): спектральное вложение G окна поворачивается
  к вложению 24 месяцев (задача Прокруста: вложение определено с точностью до поворота), X окна
  масштабируется медианой и MAD 24 месяцев; блоки взвешиваются, как в ``clustering.methods.block_join``
  на 24 месяцах. Так центры типов 24 месяцев и центры прошлого окна имеют смысл в любом окне.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from munnet.clustering import inputs as CI
from munnet.clustering import methods as M
from munnet.contracts import NETWORK_WINDOWS, MissingInputError, QCError, read_table
from munnet.dynamics import tracking as T
from munnet.network import compare as NCMP
from munnet.network import compute as NC
from munnet.network import graph as NG
from munnet.network.data import NodeSet
from munnet.network.params import NetworkParams

log = logging.getLogger(__name__)

GRAPH_TOL = 1e-9


@dataclass(frozen=True)
class Period:
    name: str  # «2023-01…2023-12»; у шума — «2023, нечётные»; у опоры — «24 месяца»
    kind: str  # rolling | odd | even | all
    year: int | None
    months: np.ndarray  # номера t месяцев

    def dates(self, ns: NodeSet) -> tuple[str, str]:
        d = ns.months.set_index("t")["date"]
        return str(d[int(self.months[0])]), str(d[int(self.months[-1])])


def periods(ns: NodeSet, netp: NetworkParams, years=(2023, 2024)) -> list[Period]:
    out = [
        Period(name, "rolling", None, months)
        for _, name, months in NCMP.window_list(
            ["rolling"], netp.rolling_months, netp.rolling_step, ns.months
        )
    ]
    for y in years:
        span = np.arange(12 * (y - 2023), 12 * (y - 2023) + 12)
        out.append(Period(f"{y}, нечётные месяцы", "odd", y, span[0::2]))
        out.append(Period(f"{y}, чётные месяцы", "even", y, span[1::2]))
    out.append(Period("24 месяца", "all", None, np.arange(24)))
    return out


def window_graph(ns: NodeSet, netp: NetworkParams, rule: str, k: int, months) -> pd.DataFrame:
    """Рёбра G окна в ключах ``territory_id`` (source < target), вес — S_ij, как в ``network_windows``."""
    spec = netp.rules[rule]
    S = NC.similarity(spec, ns, np.asarray(months)).S
    e = NG.knn_edges(S, k)
    ids = ns.ids
    return pd.DataFrame({"source": ids[e["i"]], "target": ids[e["j"]], "weight": e["weight"].to_numpy()})


def check_rolling(graphs: dict[str, pd.DataFrame], processed: Path, rule: str) -> dict[str, float]:
    """G скользящих окон, собранные здесь, против ``network_windows``: те же рёбра и веса (QCError иначе)."""
    path = processed / "network_windows.parquet"
    if not path.exists():
        raise MissingInputError(f"нет {path}: сначала запустите этап network")
    nw = read_table(path, NETWORK_WINDOWS)
    nw = nw.loc[(nw["rule"] == rule) & (nw["window_kind"].astype(str) == "rolling")]
    worst, missing = 0.0, 0
    for name, e in graphs.items():
        ref = nw.loc[nw["window"] == name, ["source", "target", "weight"]]
        if ref.empty:
            raise QCError(f"dynamics: в network_windows нет окна {name}")
        a = e.set_index(["source", "target"])["weight"].sort_index()
        b = ref.astype({"source": "int64", "target": "int64"}).set_index(["source", "target"])["weight"]
        b = b.sort_index()
        if len(a) != len(b) or not a.index.equals(b.index):
            missing += abs(len(a.index.symmetric_difference(b.index)))
            continue
        worst = max(worst, float(np.abs(a.to_numpy() - b.to_numpy()).max()))
    if missing or worst > GRAPH_TOL:
        raise QCError(
            f"dynamics: сеть окна, построенная функциями network, не совпала с network_windows: "
            f"рёбер вне пересечения {missing}, наибольшая разница веса {worst:.3g}"
        )
    return {"graph_windows_checked": float(len(graphs)), "graph_max_diff": worst}


def window_inputs(base: CI.Inputs, ns: NodeSet, A, months, name: str) -> CI.Inputs:
    """Входы окна: G окна, X — те же признаки места и уровень трат за месяцы окна; корзина окна."""
    raw = base.X_raw.copy()
    if CI.LEVEL in raw.columns:
        raw[CI.LEVEL] = CI.window_level(ns, months)
    X, _, counts, used = CI.build_X(raw, ns.groups)
    return replace(
        base,
        name=name,
        X=X,
        X_raw=raw,
        B=CI.window_clr(ns, months),
        A=A,
        imputed=counts,
        scale_used=used,
    )


# --- Пространство гибрида для альтернатив -------------------------------------------------------


def _norm_block(B: np.ndarray, mean: np.ndarray, tv: float) -> np.ndarray:
    return (B - mean) / np.sqrt(tv if tv > 0 else 1.0)


@dataclass(frozen=True)
class HybridSpace:
    """Опорное пространство гибрида 24 месяцев: Z = [√α·Ẽ, √(1 − α)·X̃], как в ``methods.block_join``."""

    k: int
    alpha: float
    seed: int
    E_ref: np.ndarray  # нормированное вложение 24 месяцев (центрировано, суммарная дисперсия 1)
    x_med: np.ndarray
    x_mad: np.ndarray
    x_mean: np.ndarray
    x_tv: float
    Z_ref: np.ndarray

    @classmethod
    def build(cls, base: CI.Inputs, k: int, alpha: float, seed: int) -> HybridSpace:
        E = M.spectral_embed(base.A, k, seed)
        E_ref = _norm_block(E, E.mean(axis=0), float(E.var(axis=0).sum()))
        filled, _ = CI.impute_region_median(base.X_raw, base.groups)
        v = filled.to_numpy(dtype=np.float64)
        med = np.median(v, axis=0)
        mad = np.median(np.abs(v - med), axis=0) * CI.MAD_SCALE
        sd = v.std(axis=0)
        mad = np.where(mad > 0, mad, np.where(sd > 0, sd, 1.0))
        X = (v - med) / mad
        if not np.allclose(X, base.X, atol=1e-9):
            raise QCError("dynamics: масштаб X опорного пространства не совпал с clustering.inputs.build_X")
        x_mean, x_tv = X.mean(axis=0), float(X.var(axis=0).sum())
        Z = np.hstack([np.sqrt(alpha) * E_ref, np.sqrt(1 - alpha) * _norm_block(X, x_mean, x_tv)])
        Z_check = M.block_join([E, base.X], [alpha, 1 - alpha])
        if not np.allclose(Z, Z_check, atol=1e-9):
            raise QCError("dynamics: опорное пространство гибрида не совпало с methods.block_join")
        return cls(k, alpha, seed, E_ref, med, mad, x_mean, x_tv, Z)

    def embed(self, inp: CI.Inputs) -> tuple[np.ndarray, float]:
        """Z окна в опорном пространстве; второе — доля вложения 24 месяцев, объяснённая повёрнутым
        вложением окна (качество поворота Прокруста)."""
        E = M.spectral_embed(inp.A, self.k, self.seed)
        En = _norm_block(E, E.mean(axis=0), float(E.var(axis=0).sum()))
        Er, fit = T.procrustes(En, self.E_ref)
        filled, _ = CI.impute_region_median(inp.X_raw, inp.groups)
        X = (filled.to_numpy(dtype=np.float64) - self.x_med) / self.x_mad
        Z = np.hstack(
            [np.sqrt(self.alpha) * Er, np.sqrt(1 - self.alpha) * _norm_block(X, self.x_mean, self.x_tv)]
        )
        return Z, fit
