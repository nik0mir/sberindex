"""Деления узлов без типов: соперники типов (``place_partitions``), страты (``strata``), тривиальные
деления T5,
отрицательные контроли (``controls``) и случайные метки. Все функции — на массивах территориальных узлов
в порядке ``territory_id``; «ступень» (ранг) — целое 1…4 снизу вверх.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from munnet.clustering import inputs as CI

N_ORDERS_K4 = 24  # 4! порядков групп — «best_of_24»


@dataclass(frozen=True)
class Partition:
    """Деление территориальных узлов: ``labels`` — группы; ``ranks`` — ступень 1…K, если порядок задан."""

    name: str
    label: str  # подпись для текстов («по зарплате, по возрастанию»)
    labels: np.ndarray
    kind: str  # sized | composite | place_only | control | random | reference | types
    ranks: np.ndarray | None = None
    oriented: str = ""  # sign_of_rho | best_of_24 | fixed
    feature: str = ""


# --- Группы заданных размеров ----------------------------------------------------------------------


def order_with_ties(values: np.ndarray, ids: np.ndarray) -> np.ndarray:
    """Порядок узлов по возрастанию значения; при равенстве ниже — меньший ``territory_id``."""
    return np.lexsort((np.asarray(ids), np.asarray(values, dtype=np.float64)))


def sized_groups(values: np.ndarray, ids: np.ndarray, sizes: Sequence[int]) -> np.ndarray:
    """Группы 1…K по возрастанию значения с размерами ``sizes`` снизу вверх (ничья — меньший id ниже)."""
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("sized_groups: пропуски в значениях (значения — после замены пропусков)")
    if int(sum(sizes)) != len(values):
        raise ValueError(f"sized_groups: размеры {list(sizes)} не дают {len(values)} узлов")
    order = order_with_ties(values, ids)
    out = np.empty(len(values), dtype=np.int64)
    start = 0
    for g, size in enumerate(sizes, start=1):
        out[order[start : start + int(size)]] = g
        start += int(size)
    return out


def rank_quantiles(values: np.ndarray, ids: np.ndarray, q: int) -> np.ndarray:
    """Квантильные группы 0…q−1 по рангу (``strata.nodes: territorial_both``): позиция в порядке
    ``order_with_ties`` × q / n, округление вниз; пропуск — −1."""
    values = np.asarray(values, dtype=np.float64)
    out = np.full(len(values), -1, dtype=np.int64)
    ok = np.isfinite(values)
    idx = np.flatnonzero(ok)
    order = order_with_ties(values[ok], np.asarray(ids)[ok])
    pos = np.empty(len(idx), dtype=np.int64)
    pos[order] = np.arange(len(idx))
    out[idx] = pos * q // len(idx)
    return out


def value_quantiles(values: np.ndarray, q: int) -> np.ndarray:
    """Квантильные группы по границам квантилей (совпадающие границы сливаются); пропуск — −1."""
    s = pd.Series(np.asarray(values, dtype=np.float64))
    out = np.full(len(s), -1, dtype=np.int64)
    ok = s.notna().to_numpy()
    out[ok] = pd.qcut(s[ok], q, labels=False, duplicates="drop").to_numpy(dtype=np.int64)
    return out


def rank_sum(frame: pd.DataFrame) -> np.ndarray:
    """Сумма средних рангов столбцов (``composite: rank_pop_urban``)."""
    return frame.rank(method="average").sum(axis=1).to_numpy(dtype=np.float64)


def robust(frame: pd.DataFrame) -> np.ndarray:
    """Масштаб ``clustering.inputs.scale: robust`` — (x − медиана) / (MAD × 1,4826)."""
    X, _ = CI.robust_scale(frame)
    return X


def zscore(frame: pd.DataFrame) -> np.ndarray:
    v = frame.to_numpy(dtype=np.float64)
    sd = v.std(axis=0)
    return (v - v.mean(axis=0)) / np.where(sd > 0, sd, 1.0)


def scale_rows(frame_all: pd.DataFrame, rows: np.ndarray, how: str = "robust") -> np.ndarray:
    """Масштаб «как clustering.inputs.scale» буквально: центр и разброс — по всем строкам ``frame_all``
    (все узлы сети, как их видела кластеризация), результат — строки ``rows`` (например, территориальные
    узлы или МО с известной целью). ``how``: robust — медиана и MAD × 1,4826, zscore — среднее и sd."""
    if how == "robust":
        Z = robust(frame_all)
    elif how == "zscore":
        Z = zscore(frame_all)
    else:
        raise ValueError(f"масштаб {how!r} не реализован")
    return Z[np.asarray(rows)]


def kmeans_labels(Z: np.ndarray, k: int, seed: int, n_init: int) -> np.ndarray:
    from sklearn.cluster import KMeans

    return KMeans(n_clusters=k, n_init=n_init, random_state=seed).fit(Z).labels_.astype(np.int64)


def pca1(Z: np.ndarray) -> np.ndarray:
    """Оценки первой главной компоненты (знак произволен — порядок групп выбирает best_of_24)."""
    C = Z - Z.mean(axis=0)
    _, _, vt = np.linalg.svd(C, full_matrices=False)
    return C @ vt[0]


def density_rel(pop: np.ndarray, area: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """ln(население / площадь) минус медиана группы региона (по узлам со значением); пропуск — NaN."""
    pop = np.asarray(pop, dtype=np.float64)
    area = np.asarray(area, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        v = np.where((pop > 0) & (area > 0), np.log(pop / area), np.nan)
    s = pd.Series(v)
    med = s.groupby(np.asarray(groups)).transform("median").to_numpy()
    return v - med


def log1p_rel(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """ln(1 + x) минус медиана группы региона (``clustering.validation.log1p_rel``)."""
    s = pd.Series(np.log1p(np.asarray(values, dtype=np.float64)))
    med = s.groupby(np.asarray(groups)).transform("median").to_numpy()
    return s.to_numpy() - med


def log1p_abs(values: np.ndarray) -> np.ndarray:
    return np.log1p(np.asarray(values, dtype=np.float64))


# --- Порядок групп -------------------------------------------------------------------------------


def orders(k: int) -> list[tuple[int, ...]]:
    """Все порядки групп 0…k−1 (кортеж: ступень группы g — позиция g в порядке + 1)."""
    return list(itertools.permutations(range(k)))


def ranks_from_order(labels: np.ndarray, order: Sequence[int]) -> np.ndarray:
    """Ступень узла: ``order`` — группы снизу вверх, ступень группы order[i] = i + 1."""
    codes = np.unique(labels)
    step = {int(codes[g]): i + 1 for i, g in enumerate(order)}
    return np.array([step[int(v)] for v in labels], dtype=np.int64)


def random_labels(sizes: Sequence[int], rng: np.random.Generator) -> np.ndarray:
    """Случайные метки 1…K с размерами ``sizes`` (перестановка узлов)."""
    base = np.concatenate([np.full(int(s), g, dtype=np.int64) for g, s in enumerate(sizes, start=1)])
    return base[rng.permutation(len(base))]


def coverage_class(frame: pd.DataFrame, bounds: Sequence[int] = (0, 1, 2)) -> np.ndarray:
    """Класс покрытия — число пропущенных показателей узла: 0 / 1 / 2 и больше (``missing_count``)."""
    miss = frame.isna().sum(axis=1).to_numpy(dtype=np.int64)
    top = int(bounds[-1])
    return np.minimum(miss, top)


def federal_districts(region_codes: np.ndarray, mapping: Mapping[str, Sequence[int]]) -> np.ndarray:
    """Федеральный округ по коду региона (Указ № 849 в редакции 26.06.2023); регион вне списка — ошибка."""
    lookup = {}
    for fd, regs in mapping.items():
        for r in regs:
            if int(r) in lookup:
                raise ValueError(f"federal_districts: регион {r} в двух округах")
            lookup[int(r)] = str(fd)
    missing = sorted({int(r) for r in region_codes} - set(lookup))
    if missing:
        raise ValueError(f"federal_districts: нет округа у регионов {missing}")
    return np.array([lookup[int(r)] for r in region_codes], dtype=object)
