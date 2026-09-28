"""Этап evaluate: индексы качества кластеризации SW, CH, S_Dbw (признаки) и AVI, AVU, MQ (сеть).

Модуль даёт функции индексов, случайный базис (перестановки меток с теми же размерами кластеров) и z-оценку;
этап ``cluster`` импортирует их для выбора метода и числа типов (``clustering.selection`` в конфиге).

Общие правила:

- метка шума ``NOISE = -1`` (HDBSCAN): такие объекты исключаются из всех индексов — строки из X, узлы и их
  рёбра из графа (индекс считается на индуцированном подграфе). Долю шума индекс не прячет: её сообщает
  вызывающий;
- неопределённый случай не роняет расчёт: индекс = NaN и предупреждение ``IcviWarning`` (K < 2, для признаков
  K > n − 1, граф без рёбер, кластер без рёбер для AVI);
- разные K сравниваются только через случайный базис: AVI убывает примерно как 1/K, AVU с ростом K падает
  к «полу», CH растёт с числом объектов (Shalileh, Antonov, Tsyplakova, 2025, с. 558–560); формула базиса
  AVU ≈ (K − 1)/(2K − 3) для кластеров равного размера (при неравных — ниже) — вывод проекта, сверен
  с табл. 3 Doklady до 0,02;
- вырожденный базис (SD ≈ 0: AVU при K = 2 и K = 3 постоянна на любой перестановке) — z = NaN: метрика
  на этом кандидате не информативна.

Формулы и источники — в докстрингах функций. Статус AVI и AVU: первоисточник (Biswas, Biswas, Expert Systems
with Applications, 2017, т. 71, с. 1–17) закрыт; формулы — по двум согласованным пересказам с формулами:
Shalileh S., Antonov E. A., Tsyplakova D. A. Cluster Validity across Attribute and Network Spaces // Doklady
Mathematics. 2025. Vol. 112, No. 3. P. 553–564 (открытый доступ), с. 555, формулы (18)–(21), и Howie и др.,
PVLDB, 2023, т. 16, № 11, с. 3175, формулы (36)–(37).
"""

from __future__ import annotations

import logging
import time
import warnings
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.spatial.distance import cdist

from munnet.config import Config

log = logging.getLogger(__name__)

NOISE: int = -1
"""Метка шума HDBSCAN: такие объекты исключаются из всех индексов."""

SPACE: dict[str, str] = {
    "sw": "features",
    "ch": "features",
    "s_dbw": "features",
    "avi": "graph",
    "avu": "graph",
    "mq": "graph",
}
"""Пространство каждого индекса: признаки X или граф A."""

BETTER: dict[str, str] = {"sw": "max", "ch": "max", "s_dbw": "min", "avi": "max", "avu": "min", "mq": "max"}
"""Направление по первоисточникам; конфиг ``icvi.metrics`` обязан с ним совпадать (проверяет metric_specs)."""

S_DBW_DENSITIES = ("pair", "own")
S_DBW_DENSITY = "pair"
"""Вариант S_Dbw по умолчанию: плотность центров считается по точкам пары кластеров (см. ``s_dbw``)."""

_CHUNK = 32  # перестановок за один матричный шаг случайного базиса: память ~ n × 32K чисел

SD_REL_TOL = 1e-12
"""Порог вырожденного базиса в ``zscore``: SD ≤ SD_REL_TOL · max(1, |mean|) — это шум округления
(у постоянной метрики SD базиса ~1e-16), а не разброс; z из такого SD — случайный знак и ранг."""


class IcviWarning(UserWarning):
    """Индекс не определён на этом разбиении или графе и заменён NaN (или принятым соглашением)."""


@dataclass(frozen=True)
class MetricSpec:
    """Индекс из ``icvi.metrics``: имя, пространство (features | graph) и направление (max | min)."""

    name: str
    space: str
    better: str


def metric_specs(cfg: Config | Mapping[str, Any]) -> dict[str, MetricSpec]:
    """Индексы из ``cfg["icvi"]["metrics"]`` в порядке конфига.

    Имя должно быть известно модулю, а пространство и направление — совпадать с первоисточником
    (``SPACE``, ``BETTER``): иначе z-оценка получит неверный знак.
    """
    specs: dict[str, MetricSpec] = {}
    for name, item in cfg["icvi"]["metrics"].items():
        if name not in SPACE:
            raise ValueError(f"icvi.metrics: неизвестный индекс {name!r}; известны {', '.join(SPACE)}")
        space, better = item.get("space"), item.get("better")
        if space not in ("features", "graph") or space != SPACE[name]:
            raise ValueError(f"icvi.metrics.{name}: space = {space!r}, должно быть {SPACE[name]!r}")
        if better not in ("max", "min") or better != BETTER[name]:
            raise ValueError(f"icvi.metrics.{name}: better = {better!r}, должно быть {BETTER[name]!r}")
        specs[name] = MetricSpec(name=name, space=space, better=better)
    return specs


def options(cfg: Config | Mapping[str, Any]) -> dict[str, Any]:
    """Параметры счёта из ``cfg["icvi"]`` для ``compute``, ``evaluate``, ``baseline_values``
    и ``random_baseline`` (передаются как ``**options(cfg)``): сейчас только ``s_dbw_density``.

    Нет ключа — значение по умолчанию модуля; недопустимое значение — ``ValueError``.
    """
    density = cfg["icvi"].get("s_dbw_density", S_DBW_DENSITY)
    if density not in S_DBW_DENSITIES:
        raise ValueError(f"icvi.s_dbw_density = {density!r}, допустимо {S_DBW_DENSITIES}")
    return {"s_dbw_density": density}


# ---------------------------------------------------------------------------------------------
# Граф из таблицы рёбер


def adjacency(edges: pd.DataFrame, node_ids: Sequence[int], *, weight: str = "weight") -> sp.csr_matrix:
    """Симметричная матрица весов n × n в порядке ``node_ids`` из таблицы рёбер ``network_edges``.

    Колонки ``source``, ``target`` (territory_id) и ``weight``; каждое неориентированное ребро записано
    один раз с ``source < target`` (контракт ``NETWORK_EDGES``). Дубли, петли, обратный порядок,
    неизвестные узлы, отрицательные и пропущенные веса — ошибка: иначе вес ребра молча удвоится
    или пропадёт.
    Узлы без рёбер остаются в матрице пустыми строками.
    """
    ids = pd.Index(np.asarray(node_ids))
    if not ids.is_unique:
        raise ValueError("adjacency: в node_ids есть повторы")
    src = edges["source"].to_numpy(dtype=np.int64)
    tgt = edges["target"].to_numpy(dtype=np.int64)
    w = edges[weight].to_numpy(dtype=np.float64)
    if np.isnan(w).any():
        raise ValueError(f"adjacency: пропуск веса в {int(np.isnan(w).sum())} рёбрах")
    if (w < 0).any():
        raise ValueError(f"adjacency: отрицательный вес в {int((w < 0).sum())} рёбрах")
    if (src == tgt).any():
        raise ValueError(f"adjacency: петли у узлов {sorted(set(src[src == tgt].tolist()))[:5]}")
    if (src > tgt).any():
        raise ValueError(f"adjacency: нарушен порядок source < target в {int((src > tgt).sum())} рёбрах")
    pairs = pd.MultiIndex.from_arrays([src, tgt])
    if pairs.has_duplicates:
        raise ValueError(f"adjacency: дубли рёбер — {int(pairs.duplicated().sum())}")
    i, j = ids.get_indexer(src), ids.get_indexer(tgt)
    if (i < 0).any() or (j < 0).any():
        unknown = sorted(set(src[i < 0].tolist()) | set(tgt[j < 0].tolist()))
        raise ValueError(f"adjacency: узлов {unknown[:5]} нет среди узлов node_ids")
    n = len(ids)
    a = sp.coo_matrix((np.r_[w, w], (np.r_[i, j], np.r_[j, i])), shape=(n, n)).tocsr()
    a.sum_duplicates()
    return a


# ---------------------------------------------------------------------------------------------
# Подготовка входов


def _warn(message: str) -> None:
    warnings.warn(message, IcviWarning, stacklevel=3)


def _codes(labels: Any, n: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Маска не-шума, коды кластеров 0…K−1 для не-шума и K."""
    labels = np.asarray(labels)
    if labels.ndim != 1:
        raise ValueError("метки должны быть одномерным массивом")
    if labels.shape[0] != n:
        raise ValueError(f"длина меток {labels.shape[0]} не равна числу объектов {n}")
    if labels.dtype.kind not in "iuf":  # строки, объекты, логические — не метки кластеров
        raise ValueError(f"метки должны быть целыми числами, получен тип {labels.dtype}")
    if labels.dtype.kind == "f":
        if not np.isfinite(labels).all():
            raise ValueError("метки должны быть целыми числами: есть NaN или бесконечности")
        as_int = labels.astype(np.int64)
        if not np.array_equal(as_int, labels):
            raise ValueError("метки должны быть целыми числами")
        labels = as_int
    bad = (labels < 0) & (labels != NOISE)
    if bad.any():
        raise ValueError(f"отрицательные метки кроме шума {NOISE}: {sorted(set(labels[bad].tolist()))}")
    keep = labels != NOISE
    uniq, codes = np.unique(labels[keep], return_inverse=True)
    return keep, codes.astype(np.intp), len(uniq)


class _Features:
    """Признаки не-шумовых объектов, центрированные: суммы по кластерам без потери точности."""

    def __init__(self, x: np.ndarray):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x[:, None]
        if not np.isfinite(x).all():
            raise ValueError("X: есть пропуски или бесконечности")
        self.x = x - x.mean(axis=0)
        self.x2 = self.x**2
        self.total = float(self.x2.sum())
        self.norm_var = float(np.linalg.norm(self.x2.mean(axis=0)))  # ||σ(X)||, дисперсия с 1/n
        self._dist: np.ndarray | None = None

    @property
    def dist(self) -> np.ndarray:
        """Евклидовы расстояния n × n — один раз на все перестановки."""
        if self._dist is None:
            self._dist = cdist(self.x, self.x)
        return self._dist


class _Graph:
    """Индуцированный подграф не-шумовых узлов в виде троек (i, j, вес) обоих направлений."""

    def __init__(self, a: Any, keep: np.ndarray):
        a = sp.csr_matrix(a, dtype=np.float64)
        n = keep.shape[0]
        if a.shape != (n, n):
            raise ValueError(f"длина меток {n} не равна числу узлов графа {a.shape}")
        if a.nnz and not np.isfinite(a.data).all():
            raise ValueError("A: есть пропуски или бесконечности в весах")
        if a.nnz and (a.data < 0).any():
            raise ValueError("A: отрицательные веса")
        if a.diagonal().any():
            raise ValueError("A: петли (ненулевая диагональ) не допускаются")
        asym = abs(a - a.T)
        if asym.nnz and asym.max() > 1e-12 * max(abs(a).max(), 1.0):
            raise ValueError("A: матрица не симметрична — граф должен быть неориентированным")
        sub = a[keep][:, keep].tocoo()
        self.rows, self.cols, self.w = sub.row.astype(np.intp), sub.col.astype(np.intp), sub.data
        self.two_m = float(self.w.sum())  # 2W: каждое ребро учтено в обе стороны


def _onehot(perms: np.ndarray, k: int) -> sp.csr_matrix:
    """Блочная индикаторная матрица H (b·K × n): строка b·K + k — объекты кластера k в перестановке b."""
    b, n = perms.shape
    rows = (perms + k * np.arange(b)[:, None]).ravel()
    cols = np.tile(np.arange(n), b)
    return sp.csr_matrix((np.ones(b * n), (rows, cols)), shape=(b * k, n))


# ---------------------------------------------------------------------------------------------
# Признаковые индексы: пачка перестановок perms (b × n) с одинаковыми размерами кластеров sizes


def _ch_batch(f: _Features, perms: np.ndarray, sizes: np.ndarray, k: int) -> np.ndarray:
    n = perms.shape[1]
    sums = (_onehot(perms, k) @ f.x).reshape(len(perms), k, -1)
    between = ((sums**2).sum(axis=2) / sizes).sum(axis=1)  # Σ n_k ||μ_k − μ||²
    within = f.total - between  # Σ ||x − μ_k||²
    with np.errstate(divide="ignore", invalid="ignore"):
        ch = between * (n - k) / (within * (k - 1))
    return np.where(within == 0, 1.0, ch)  # как sklearn: при нулевом внутреннем разбросе 1


def _sw_batch(f: _Features, perms: np.ndarray, sizes: np.ndarray, k: int) -> np.ndarray:
    d = f.dist
    b, n = perms.shape
    h = np.zeros((n, b * k))
    h[np.tile(np.arange(n), b), (perms + k * np.arange(b)[:, None]).ravel()] = 1.0
    sums = (d @ h).reshape(n, b, k).transpose(1, 0, 2)  # сумма расстояний до объектов каждого кластера
    own = sizes[perms]
    intra_sum = np.take_along_axis(sums, perms[:, :, None], axis=2)[:, :, 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        a = intra_sum / (own - 1)
        mean_other = sums / sizes
        np.put_along_axis(mean_other, perms[:, :, None], np.inf, axis=2)
        b_ = mean_other.min(axis=2)
        s = (b_ - a) / np.maximum(a, b_)
    s[own == 1] = 0.0  # Rousseeuw, 1987: у единственного объекта кластера s(i) = 0
    return np.nan_to_num(s).mean(axis=1)


def _s_dbw_batch(
    f: _Features, perms: np.ndarray, sizes: np.ndarray, k: int, density: str
) -> tuple[np.ndarray, np.ndarray]:
    """S_Dbw пачки и флаг «отношение плотностей не определено» по перестановкам."""
    b, n = perms.shape
    h = _onehot(perms, k)
    cent = (h @ f.x).reshape(b, k, -1) / sizes[:, None]
    var = np.clip((h @ f.x2).reshape(b, k, -1) / sizes[:, None] - cent**2, 0.0, None)
    norm_sigma = np.linalg.norm(var, axis=2)  # ||σ(v_k)||
    scat = norm_sigma.mean(axis=1) / f.norm_var
    stdev = np.sqrt(norm_sigma.sum(axis=1)) / k
    kk, ll = np.triu_indices(k, 1)
    refs = np.concatenate([cent, (cent[:, kk] + cent[:, ll]) / 2], axis=1)  # центры, затем середины пар
    d2 = (f.x2.sum(axis=1)[None, :, None] + (refs**2).sum(axis=2)[:, None, :]) - 2 * np.einsum(
        "nd,brd->bnr", f.x, refs
    )
    near = (d2 <= (stdev**2)[:, None, None]).astype(np.float64)  # f(x, u) = 1, если d(x, u) ≤ stdev
    onehot = np.zeros((b, n, k))
    np.put_along_axis(onehot, perms[:, :, None], 1.0, axis=2)
    counts = onehot.transpose(0, 2, 1) @ near  # (b, K, R): точки кластера k в окрестности опорной точки r
    mid = k + np.arange(len(kk))
    dens_u = counts[:, kk, mid] + counts[:, ll, mid]
    if density == "pair":
        dens_k = counts[:, kk, kk] + counts[:, ll, kk]
        dens_l = counts[:, kk, ll] + counts[:, ll, ll]
    else:
        dens_k, dens_l = counts[:, kk, kk], counts[:, ll, ll]
    den = np.maximum(dens_k, dens_l)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(den > 0, dens_u / den, np.where(dens_u == 0, 0.0, np.nan))
    undefined = np.isnan(ratio).any(axis=1)
    dens_bw = 2 * ratio.sum(axis=1) / (k * (k - 1))  # симметрия: Σ_{k≠l} = 2 Σ_{k<l}
    return scat + dens_bw, undefined


# ---------------------------------------------------------------------------------------------
# Сетевые индексы через матрицу «кластер × кластер» M = Hᵀ A H


def _cluster_matrix(g: _Graph, perms: np.ndarray, k: int) -> np.ndarray:
    b = perms.shape[0]
    idx = perms[:, g.rows] * k + perms[:, g.cols] + (k * k * np.arange(b))[:, None]
    m = np.bincount(idx.ravel(), weights=np.tile(g.w, b), minlength=b * k * k)
    return m.reshape(b, k, k)


def _avi_from(m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    diag = np.diagonal(m, axis1=1, axis2=2)
    vol = m.sum(axis=2)
    with np.errstate(divide="ignore", invalid="ignore"):
        iso = diag / vol
    empty = (vol == 0).any(axis=1)
    return np.where(empty, np.nan, iso.mean(axis=1)), empty


def _avu_from(m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    k = m.shape[1]
    diag = np.diagonal(m, axis1=1, axis2=2)
    out = m.sum(axis=2) - diag
    den = out[:, :, None] + out[:, None, :] - m
    off = ~np.eye(k, dtype=bool)
    with np.errstate(divide="ignore", invalid="ignore"):
        u = np.where(den > 0, m / den, 0.0)
    u[:, ~off] = 0.0
    zero_den = ((den <= 0) & off).any(axis=(1, 2))
    return u.sum(axis=(1, 2)) / k, zero_den


def _mq_from(m: np.ndarray, two_m: float, resolution: float) -> np.ndarray:
    diag = np.diagonal(m, axis1=1, axis2=2)
    vol = m.sum(axis=2)
    return (diag / two_m - resolution * (vol / two_m) ** 2).sum(axis=1)


# ---------------------------------------------------------------------------------------------
# Общий счёт: одно разбиение — пачка из одной перестановки


@dataclass
class _Prepared:
    keep: np.ndarray
    codes: np.ndarray
    k: int
    sizes: np.ndarray
    features: _Features | None
    graph: _Graph | None


def _prepare(labels: Any, x: Any, a: Any, names: Sequence[str]) -> _Prepared:
    need_x = any(SPACE[n] == "features" for n in names)
    need_a = any(SPACE[n] == "graph" for n in names)
    if need_x and x is None:
        raise ValueError(f"для {[n for n in names if SPACE[n] == 'features']} нужны признаки X")
    if need_a and a is None:
        raise ValueError(f"для {[n for n in names if SPACE[n] == 'graph']} нужен граф A")
    n = np.asarray(labels).shape[0]
    if need_x:
        n_x = np.asarray(x).shape[0]
        if n_x != n:
            raise ValueError(f"длина меток {n} не равна числу строк X {n_x}")
    keep, codes, k = _codes(labels, n)
    features = _Features(np.asarray(x)[keep]) if need_x and keep.any() else None
    graph = _Graph(a, keep) if need_a else None
    return _Prepared(keep, codes, k, np.bincount(codes, minlength=k).astype(np.float64), features, graph)


def _check_names(names: Iterable[str]) -> list[str]:
    names = list(names)
    unknown = [n for n in names if n not in SPACE]
    if unknown:
        raise ValueError(f"неизвестные индексы {unknown}; известны {', '.join(SPACE)}")
    return names


def _degenerate(prep: _Prepared, name: str) -> str | None:
    """Причина, по которой индекс не определён на этом разбиении, или None."""
    n = prep.codes.shape[0]
    if n == 0:
        return "все объекты — шум"
    if prep.k < 2:
        return f"K = {prep.k}: индекс качества разбиения требует хотя бы двух кластеров"
    if SPACE[name] == "features" and name != "s_dbw" and prep.k > n - 1:
        return f"K = {prep.k} при n = {n}: силуэт и CH определены при 2 ≤ K ≤ n − 1"
    if SPACE[name] == "graph" and prep.graph is not None and prep.graph.two_m == 0:
        return "граф без рёбер: сетевые индексы не определены"
    return None


def _batch(
    prep: _Prepared, name: str, perms: np.ndarray, s_dbw_density: str
) -> tuple[np.ndarray, np.ndarray | None, str]:
    """Значения индекса по перестановкам, флаги неопределённости и их пояснение."""
    k, sizes = prep.k, prep.sizes
    if name == "sw":
        return _sw_batch(prep.features, perms, sizes, k), None, ""
    if name == "ch":
        return _ch_batch(prep.features, perms, sizes, k), None, ""
    if name == "s_dbw":
        values, flags = _s_dbw_batch(prep.features, perms, sizes, k, s_dbw_density)
        why = "S_Dbw: у пары кластеров нет точек в окрестности обоих центров, а у середины есть — отношение ∞"
        return np.where(flags, np.nan, values), flags, why
    m = _cluster_matrix(prep.graph, perms, k)
    if name == "avi":
        values, flags = _avi_from(m)
        return values, flags, "AVI: у кластера нет рёбер (объём 0) — изолируемость не определена"
    if name == "avu":
        values, flags = _avu_from(m)
        why = "AVU: знаменатель 0 у пары кластеров без внешних рёбер — объединяемость принята 0"
        return values, flags, why
    return _mq_from(m, prep.graph.two_m, 1.0), None, ""


def _single(prep: _Prepared, name: str, s_dbw_density: str) -> float:
    reason = _degenerate(prep, name)
    if reason is not None:
        _warn(f"{name}: {reason}; значение NaN")
        return float("nan")
    values, flags, why = _batch(prep, name, prep.codes[None, :], s_dbw_density)
    if flags is not None and flags[0]:
        _warn(why)
    return float(values[0])


def _check_density(density: str) -> str:
    if density not in S_DBW_DENSITIES:
        raise ValueError(f"S_Dbw: density = {density!r}, допустимо {S_DBW_DENSITIES}")
    return density


# ---------------------------------------------------------------------------------------------
# Индексы по одному


def silhouette(X: np.ndarray, labels: np.ndarray) -> float:
    """Силуэт (SW), больше — лучше, [−1; 1]. Rousseeuw P. J., Journal of Computational and Applied
    Mathematics, 1987, т. 20 (страница определения не сверена: текст закрыт; счёт сверен со sklearn до 1e-10).

    s(i) = (b(i) − a(i)) / max{a(i), b(i)}, a(i) — среднее евклидово расстояние до своего кластера (без i),
    b(i) — наименьшее среднее расстояние до чужого кластера; у единственного объекта кластера s(i) = 0;
    SW = среднее s(i) по объектам. Определён при 2 ≤ K ≤ n − 1, иначе NaN с предупреждением.
    """
    return _single(_prepare(labels, X, None, ["sw"]), "sw", S_DBW_DENSITY)


def calinski_harabasz(X: np.ndarray, labels: np.ndarray) -> float:
    """Индекс Калинского — Харабаша (CH), больше — лучше. Caliński T., Harabasz J., Communications
    in Statistics, 1974, т. 3, № 1 (страница не сверена; счёт сверен со sklearn до 1e-10).

    CH = [tr B / (K − 1)] / [tr W / (n − K)], tr B = Σ_k n_k ||μ_k − μ||², tr W = Σ_k Σ_{i∈k} ||x_i − μ_k||².
    Растёт с числом объектов n: между выборками разного размера — только CH/n или z-оценка.
    """
    return _single(_prepare(labels, X, None, ["ch"]), "ch", S_DBW_DENSITY)


def s_dbw(X: np.ndarray, labels: np.ndarray, *, density: str = S_DBW_DENSITY) -> float:
    """S_Dbw, меньше — лучше. Halkidi M., Vazirgiannis M., ICDM 2001, с. 187–194 (текст закрыт);
    формулы — в изложении самих авторов: Halkidi, Batistakis, Vazirgiannis. Clustering Validity Checking
    Methods: Part II // SIGMOD Record, 2002, т. 31, № 3, с. 4–5 файла, формулы (11), (14)–(17).

    S_Dbw = Scat + Dens_bw, где
    Scat = (1/K) Σ_k ||σ(v_k)|| / ||σ(X)||, σ — вектор дисперсий по признакам (с 1/n; Doklady, 2025,
    формула (11)), ||·|| — евклидова норма;
    stdev = (1/K) √(Σ_k ||σ(v_k)||);
    Dens_bw = 1/(K(K − 1)) Σ_k Σ_{l≠k} density(u_kl) / max{density(v_k), density(v_l)}, u_kl — середина
    отрезка между центрами v_k и v_l; density(u) = Σ f(x, u), f = 1 при d(x, u) ≤ stdev, иначе 0.

    Вариант (реализации расходятся — фиксируем явно):
    - ``density="pair"`` (по умолчанию) — буквально формула (15) авторов: плотность и середины, и обоих
      центров считается по точкам C_k ∪ C_l («n_ij — число точек кластеров c_i и c_j»);
    - ``density="own"`` — плотность центра только по своему кластеру, как в Shalileh и др., Doklady
      Mathematics, 2025, с. 555, формула (15). Там же в формуле (14) неравенство f записано наоборот
      (f = 0 при d ≤ stdev) — по-видимому, опечатка: у авторов f = 1 внутри окрестности.
    Если у пары max{density(v_k), density(v_l)} = 0, а density(u_kl) > 0, отношение бесконечно — NaN
    с предупреждением; если обе плотности 0, слагаемое 0.

    Вырождение в многомерных признаках (по проверке на X этапа 3 с 10–11 признаками, 28.09.2026; на синтетике
    с тяжёлыми хвостами — тест ``test_random_baseline_counts_undefined_permutations``): stdev = √(Σ‖σ_k‖)/K
    убывает с K и оказывается меньше медианного расстояния до ближайшего соседа, в окрестностях центров
    0–12 точек. Тогда Dens_bw = 0 и S_Dbw = Scat (индекс вырождается в разброс внутри кластеров) или у пары
    отношение ∞ → NaN. Поведение формулы не меняем; в случайном базисе такие перестановки отбрасываются
    и считаются в ``n_undefined`` (см. ``random_baseline``).
    """
    return _single(_prepare(labels, X, None, ["s_dbw"]), "s_dbw", _check_density(density))


def avi(A: sp.spmatrix, labels: np.ndarray) -> float:
    """Средняя изолируемость (AVI), больше — лучше, [0; 1]. Biswas A., Biswas B., Expert Systems with
    Applications, 2017, т. 71, с. 1–17 — текст закрыт; формула по пересказу: Shalileh, Antonov, Tsyplakova,
    Doklady Mathematics, 2025, т. 112, № 3, с. 555, формулы (18)–(19); согласуется с Howie и др., PVLDB, 2023,
    т. 16, № 11, с. 3175, формула (36). Das и др. (arXiv:2212.10797, соавтор — A. Biswas), формулы (19)–(20),
    совместимы с обоими вариантами счёта внутренних рёбер; упорядоченные пары следуют из Doklady (18)
    и Howie (36).

    Isolability_k = Σ_{i,j} a_ij s_ik s_jk / (Σ_{i,j} a_ij s_ik s_jk + Σ_{i,j} a_ij s_ik (1 − s_jk))
    = M_kk / vol_k, где M = Hᵀ A H, vol_k = Σ_l M_kl — сумма взвешенных степеней кластера. Суммы идут
    по всем (i, j), поэтому внутренние рёбра считаются упорядоченными парами (дважды): на двух треугольниках
    с мостом 6/7, а не 3/4. Веса — суммы весов (у Howie и др. — вероятности рёбер p(u, v)).
    AVI = (1/K) Σ_k Isolability_k.
    Случайный базис ≈ 1/K. Кластер без рёбер (объём 0) — NaN с предупреждением.
    """
    return _single(_prepare(labels, None, A, ["avi"]), "avi", S_DBW_DENSITY)


def avu(A: sp.spmatrix, labels: np.ndarray) -> float:
    """Средняя объединяемость (AVU), меньше — лучше, [0; 1]. Biswas, Biswas, 2017 — текст закрыт; формула
    по пересказу: Shalileh и др., Doklady Mathematics, 2025, с. 555, формулы (20)–(21); объединяемость пары
    совпадает с Howie и др., PVLDB, 2023, с. 3175, формула (37).

    Unifiability_kl = M_kl / (out_k + out_l − M_kl), out_k = vol_k − M_kk — вес рёбер из кластера наружу;
    AVU = (1/K) Σ_k Σ_{l≠k} Unifiability_kl (нормировка 1/K — по Doklady; у Howie и др. — «арифметическое
    среднее»; внутри одного K это лишь множитель, z-оценка против случайного базиса от него не зависит).
    Проверка нормировки: в однородной SBM с кластерами равного размера AVU ≈ (K − 1)/(2K − 3) — 0,667;
    0,556; 0,533; 0,524; 0,519 при K = 3, 6, 9, 12, 15; в табл. 3 Doklady — 0,67; 0,56; 0,55; 0,53; 0,52
    (совпадение до 0,02). При неравных размерах случайный базис ниже этой формулы.

    Тождества (на любом графе с внешними рёбрами, при любых весах и размерах):
    - K = 2: out_1 = out_2 = M_12, U_12 = 1 → AVU ≡ 1;
    - K = 3: знаменатель U_kl у всех пар равен M_12 + M_13 + M_23, Σ_{k≠l} U_kl = 2 → AVU ≡ 2/3.
    При K = 2 и 3 AVU постоянна на всех перестановках, SD базиса ~1e-16, и ``zscore`` даёт NaN: на таких
    кандидатах AVU не информативна.

    Как читать: в формуле нет внутренних рёбер кластеров. AVU мала, когда внешние связи кластера разнесены
    по многим кластерам, и велика, когда два кластера отдают друг другу почти всё внешнее — их стоило бы
    слить. Случайные метки разносят внешние рёбра равномерно, поэтому случайный базис AVU близок к её
    минимуму: z-оценка против базиса у AVU обычно около нуля или хуже нуля.
    Знаменатель 0 (оба кластера без внешних рёбер, тогда и числитель 0) — объединяемость 0 с предупреждением:
    предел «смешивания нет», AVU = 0 (Doklady, с. 560).
    """
    return _single(_prepare(labels, None, A, ["avu"]), "avu", S_DBW_DENSITY)


def modularity(A: sp.spmatrix, labels: np.ndarray, *, resolution: float = 1.0) -> float:
    """Модульность (MQ), больше — лучше. Newman M. E. J., Girvan M., Physical Review E, 2004, т. 69, 026113;
    взвешенная — Newman, Physical Review E, 2004, т. 70, 056131; параметр разрешения γ — Reichardt,
    Bornholdt, 2006 (страницы не сверены; счёт сверен с networkx и igraph до 1e-9).

    Q = Σ_k [M_kk / 2W − γ (vol_k / 2W)²], 2W = Σ_ij a_ij — доля веса рёбер внутри кластеров минус её ожидание
    в нулевой модели с теми же (взвешенными) степенями. Граф неориентированный; γ = 1 в выборе этапа 3.
    Сырые значения сравнимы только на одной сети; предел разрешения — Fortunato, Barthélemy, 2007.
    """
    prep = _prepare(labels, None, A, ["mq"])
    reason = _degenerate(prep, "mq")
    if reason is not None:
        _warn(f"mq: {reason}; значение NaN")
        return float("nan")
    m = _cluster_matrix(prep.graph, prep.codes[None, :], prep.k)
    return float(_mq_from(m, prep.graph.two_m, float(resolution))[0])


# ---------------------------------------------------------------------------------------------
# По имени, все сразу, случайный базис


def compute(
    name: str, labels: np.ndarray, *, X: Any = None, A: Any = None, s_dbw_density: str = S_DBW_DENSITY
) -> float:
    """Один индекс по имени из ``SPACE`` (sw, ch, s_dbw — нужен X; avi, avu, mq — нужен A; MQ при γ = 1)."""
    (name,) = _check_names([name])
    return _single(_prepare(labels, X, A, [name]), name, _check_density(s_dbw_density))


def evaluate(
    labels: np.ndarray,
    *,
    X: Any = None,
    A: Any = None,
    names: Iterable[str] | None = None,
    s_dbw_density: str = S_DBW_DENSITY,
) -> dict[str, float]:
    """Выбранные индексы (по умолчанию — все, для которых передано пространство) в порядке ``names``.

    Подготовка (центрирование X, подграф без шума) общая для всех индексов.
    """
    if names is None:
        names = [
            n
            for n in SPACE
            if (SPACE[n] == "features" and X is not None) or (SPACE[n] == "graph" and A is not None)
        ]
        if not names:
            raise ValueError("evaluate: не передано ни X, ни A")
    names = _check_names(names)
    prep = _prepare(labels, X, A, names)
    density = _check_density(s_dbw_density)
    return {n: _single(prep, n, density) for n in names}


def baseline_values(
    labels: np.ndarray,
    *,
    X: Any = None,
    A: Any = None,
    names: Iterable[str] | None = None,
    n_perm: int,
    rng: np.random.Generator,
    s_dbw_density: str = S_DBW_DENSITY,
) -> dict[str, np.ndarray]:
    """Значения индексов на ``n_perm`` перестановках меток (для процентилей и z-оценки).

    Перестановка — ``rng.permutation`` кодов не-шумовых объектов: размеры кластеров те же, шум остаётся
    на местах и исключается. Все индексы считаются на одних и тех же перестановках. Неопределённые значения —
    NaN (о них — одно предупреждение в журнале, а не по штуке на перестановку).
    """
    if names is None:
        names = [
            n
            for n in SPACE
            if (SPACE[n] == "features" and X is not None) or (SPACE[n] == "graph" and A is not None)
        ]
    names = _check_names(names)
    density = _check_density(s_dbw_density)
    prep = _prepare(labels, X, A, names)
    perms = np.stack([rng.permutation(prep.codes) for _ in range(n_perm)]) if n_perm else np.empty((0, 0))
    out: dict[str, np.ndarray] = {}
    for name in names:
        started = time.perf_counter()
        reason = _degenerate(prep, name)
        if reason is not None or n_perm == 0:
            if reason is not None:
                log.warning("случайный базис %s: %s; значения NaN", name, reason)
            out[name] = np.full(n_perm, np.nan)
            continue
        parts = []
        for start in range(0, n_perm, _CHUNK):
            values, _, _ = _batch(prep, name, perms[start : start + _CHUNK], density)
            parts.append(values)
        out[name] = np.concatenate(parts)
        bad = int(np.isnan(out[name]).sum())
        if bad:
            log.warning("случайный базис %s: %d из %d перестановок дали NaN", name, bad, n_perm)
        log.debug(
            "случайный базис %s: %d перестановок за %.2f с", name, n_perm, time.perf_counter() - started
        )
    return out


def random_baseline(
    labels: np.ndarray,
    *,
    X: Any = None,
    A: Any = None,
    names: Iterable[str] | None = None,
    n_perm: int,
    rng: np.random.Generator,
    s_dbw_density: str = S_DBW_DENSITY,
) -> pd.DataFrame:
    """Случайный базис: строка на индекс с колонками ``name``, ``mean``, ``sd`` (ddof = 1), ``n_perm``
    и ``n_undefined``. Перестановки меток с теми же размерами кластеров; шум остаётся на местах
    и исключается.

    ``n_perm`` — число конечных значений, по которым посчитаны ``mean`` и ``sd``; ``n_undefined`` — сколько
    перестановок дали NaN и отброшены (``n_perm + n_undefined`` = запрошенное число). NaN даёт прежде всего
    S_Dbw в многомерных признаках: радиус stdev мал, у пары кластеров в окрестностях центров нет точек,
    а у середины есть — отношение ∞ (см. ``s_dbw``); отбрасывание смещает базис S_Dbw, поэтому число
    отброшенных идёт в отчёт этапа evaluate.

    Нужен, чтобы сравнивать разные K и разные методы: у AVI базис ≈ 1/K, у AVU при равных размерах
    ≈ (K − 1)/(2K − 3) (при неравных — ниже; при K = 2 и 3 базис вырожден — SD ≈ 0), у CH — растёт с n,
    у S_Dbw зависит от K (Shalileh и др., 2025, с. 557–560).
    """
    values = baseline_values(
        labels, X=X, A=A, names=names, n_perm=n_perm, rng=rng, s_dbw_density=s_dbw_density
    )
    rows = []
    for name, v in values.items():
        finite = v[np.isfinite(v)]
        rows.append(
            {
                "name": name,
                "mean": float(finite.mean()) if finite.size else float("nan"),
                "sd": float(finite.std(ddof=1)) if finite.size > 1 else float("nan"),
                "n_perm": int(finite.size),
                "n_undefined": int(v.size - finite.size),
            }
        )
    return pd.DataFrame(rows, columns=["name", "mean", "sd", "n_perm", "n_undefined"])


def zscore(value: float, mean: float, sd: float, better: str) -> float:
    """z-оценка против случайного базиса со знаком «больше — всегда лучше».

    z = (value − mean) / sd при better = "max" и z = (mean − value) / sd при better = "min".
    NaN во входе — NaN. Вырожденный базис — SD ≤ ``SD_REL_TOL`` · max(1, |mean|) (метрика постоянна
    на всех перестановках: например, AVU ≡ 1 при K = 2 и ≡ 2/3 при K = 3) — тоже NaN: метрика на этом
    кандидате не информативна, а z из SD ~1e-16 — шум округления со случайным знаком.
    """
    if better not in ("max", "min"):
        raise ValueError(f"zscore: better = {better!r}, допустимо 'max' или 'min'")
    if not (np.isfinite(value) and np.isfinite(mean) and np.isfinite(sd)):
        return float("nan")
    if sd <= SD_REL_TOL * max(1.0, abs(mean)):
        return float("nan")
    z = (value - mean) / sd
    return float(z if better == "max" else -z)


def run(cfg: Config) -> None:
    raise NotImplementedError("этап evaluate запускается после cluster (PLAN.md, этап 4)")
