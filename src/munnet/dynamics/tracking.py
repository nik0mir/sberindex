"""Отслеживание типов во времени: сопоставление, переходы, события, шум — чистые функции на метках.

Метки — целые 0…K−1 (номер типа); −1 — «нет согласованного типа» (только в ``stable_halves``).

- ``match`` — венгерский алгоритм (``scipy.optimize.linear_sum_assignment``) по матрице Жаккара узлов:
  тип другого разбиения получает номер того типа опорного, с которым у него наибольший суммарный Жаккар.
- ``share_changed`` — доля узлов, чей тип после сопоставления другой (не зависит от нумерации меток).
- ``monic_events`` — события схемы MONIC (Spiliopoulou и др., 2006) в варианте с порогом Жаккара, как
  у Greene и др. (2010): пары «тип до — тип после» с J ≥ τ образуют двудольный граф; компонента 1 → 1 —
  «сохранился», 1 → ≥2 — «разделился», ≥2 → 1 — «слились», ≥2 → ≥2 — «перестроились», тип без пары
  до — «исчез», после — «возник».
- ``stable_halves`` — надёжный переход узла: тип совпал на нечётных и чётных месяцах и в 2023, и в 2024 году,
  а типы двух лет различаются (предрегистрация ``dynamics.tracking.node_transition``).
- ``paired_bootstrap`` — «изменение сверх шума»: бутстрап узлов, в каждой выборке одни и те же узлы для
  изменения и для шума, сопоставление пересчитывается; изменение сверх шума, если нижняя граница
  интервала разности больше нуля (``dynamics.tracking.change_rule``).
- ``procrustes``, ``nearest``, ``evolutionary_kmeans`` — альтернативы: фиксированные типы и эволюционная
  кластеризация (Chakrabarti, Kumar, Tomkins. Evolutionary clustering // KDD 2006, разд. 4.2: центр
  c_j ← (1 − γ)·cp·c_f(j) + γ·(1 − cp)·среднее кластера, γ = n_j / (n_j + n_f(j)), c_f(j) — ближайший центр
  прошлого шага, n — размеры кластеров; у авторов центр затем нормируется к единичной длине, здесь — веса
  к сумме 1, см. ``evo_center``).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

NONE = -1
EVENT_KINDS: tuple[str, ...] = ("survived", "split", "merged", "reorganized", "emerged", "disappeared")


# --- Сопоставление -------------------------------------------------------------------------------


def contingency(a: np.ndarray, b: np.ndarray, ka: int, kb: int) -> np.ndarray:
    """Число узлов с типом i в ``a`` и j в ``b`` (узлы с −1 не считаются)."""
    a, b = np.asarray(a), np.asarray(b)
    ok = (a >= 0) & (b >= 0)
    return np.bincount(a[ok] * kb + b[ok], minlength=ka * kb).reshape(ka, kb)


def jaccard_from_counts(M: np.ndarray) -> np.ndarray:
    ra = M.sum(axis=1, keepdims=True)
    cb = M.sum(axis=0, keepdims=True)
    union = ra + cb - M
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(union > 0, M / np.where(union > 0, union, 1), 0.0)


def jaccard_matrix(a: np.ndarray, b: np.ndarray, ka: int, kb: int) -> np.ndarray:
    """J[i, j] = |A_i ∩ B_j| / |A_i ∪ B_j| — Жаккар узлов типа i разбиения ``a`` и типа j разбиения ``b``."""
    return jaccard_from_counts(contingency(a, b, ka, kb))


def match(ref: np.ndarray, other: np.ndarray, k: int) -> np.ndarray:
    """Номер опорного типа для каждого типа ``other``: венгерский алгоритм, максимум суммы Жаккара."""
    J = jaccard_matrix(other, ref, k, k)
    rows, cols = linear_sum_assignment(-J)
    mapping = np.arange(k)
    mapping[rows] = cols
    return mapping


def relabel(labels: np.ndarray, mapping: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels)
    out = np.full(len(labels), NONE, dtype=np.int64)
    ok = labels >= 0
    out[ok] = mapping[labels[ok]]
    return out


def aligned(ref: np.ndarray, other: np.ndarray, k: int) -> np.ndarray:
    """``other`` в нумерации типов ``ref``."""
    return relabel(other, match(ref, other, k))


def share_changed(a: np.ndarray, b: np.ndarray, k: int) -> float:
    """Доля узлов, у которых тип в ``b`` (после сопоставления с ``a``) другой."""
    return float(np.mean(aligned(a, b, k) != np.asarray(a)))


def transition_matrix(a: np.ndarray, b: np.ndarray, k: int) -> np.ndarray:
    """Матрица переходов: строки — тип в ``a``, столбцы — тип в ``b`` (метки уже в одной нумерации)."""
    return contingency(a, b, k, k)


# --- События MONIC -------------------------------------------------------------------------------


@dataclass(frozen=True)
class Event:
    kind: str
    before: tuple[int, ...]
    after: tuple[int, ...]
    jaccard: float  # наибольший Жаккар пары внутри события (0 — у «возник» и «исчез»)


def monic_events(a: np.ndarray, b: np.ndarray, ka: int, kb: int, tau: float) -> list[Event]:
    """События между разбиениями ``a`` (до) и ``b`` (после) при пороге Жаккара ``tau``."""
    J = jaccard_matrix(a, b, ka, kb)
    edges = J >= tau
    seen_a, seen_b = set(), set()
    events = []
    for i0 in range(ka):
        if i0 in seen_a:
            continue
        comp_a, comp_b, stack = {i0}, set(), [("a", i0)]
        while stack:
            side, v = stack.pop()
            nbrs = np.flatnonzero(edges[v]) if side == "a" else np.flatnonzero(edges[:, v])
            for u in nbrs.tolist():
                if side == "a" and u not in comp_b:
                    comp_b.add(u)
                    stack.append(("b", u))
                elif side == "b" and u not in comp_a:
                    comp_a.add(u)
                    stack.append(("a", u))
        seen_a |= comp_a
        seen_b |= comp_b
        before, after = tuple(sorted(comp_a)), tuple(sorted(comp_b))
        best = float(J[np.ix_(before, after)].max()) if after else 0.0
        if not after:
            kind = "disappeared"
        elif len(before) == 1 and len(after) == 1:
            kind = "survived"
        elif len(before) == 1:
            kind = "split"
        elif len(after) == 1:
            kind = "merged"
        else:
            kind = "reorganized"
        events.append(Event(kind, before, after, best))
    for j in range(kb):
        if j not in seen_b:
            events.append(Event("emerged", (), (j,), 0.0))
    return events


def event_counts(events: Sequence[Event]) -> dict[str, int]:
    return {k: sum(e.kind == k for e in events) for k in EVENT_KINDS}


# --- Надёжные переходы узлов ---------------------------------------------------------------------


def stable_halves(
    odd_a: np.ndarray, even_a: np.ndarray, odd_b: np.ndarray, even_b: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Надёжный переход и согласованные типы двух лет (−1 — половины года разошлись). Метки всех четырёх
    разбиений — в одной нумерации типов."""
    t_a = np.where(np.asarray(odd_a) == np.asarray(even_a), odd_a, NONE)
    t_b = np.where(np.asarray(odd_b) == np.asarray(even_b), odd_b, NONE)
    reliable = (t_a >= 0) & (t_b >= 0) & (t_a != t_b)
    return reliable, t_a.astype(np.int64), t_b.astype(np.int64)


# --- Изменение сверх шума ------------------------------------------------------------------------


@dataclass(frozen=True)
class BootstrapResult:
    change: float  # доля узлов, сменивших тип между двумя периодами
    noise: float  # доля «смен» между разбиениями одного времени (среднее по парам шума)
    diff: float
    lo: float  # границы интервала разности по перцентилям бутстрапа
    hi: float
    change_lo: float
    change_hi: float
    noise_lo: float
    noise_hi: float
    n_boot: int
    exceeds: bool  # нижняя граница разности больше нуля


def _noise_share(pairs: Sequence[tuple[np.ndarray, np.ndarray]], k: int, idx=None) -> float:
    vals = []
    for x, y in pairs:
        if idx is not None:
            x, y = x[idx], y[idx]
        vals.append(share_changed(x, y, k))
    return float(np.mean(vals))


def paired_bootstrap(
    a: np.ndarray,
    b: np.ndarray,
    noise_pairs: Sequence[tuple[np.ndarray, np.ndarray]],
    k: int,
    n_boot: int,
    rng: np.random.Generator,
    level: float = 0.95,
) -> BootstrapResult:
    """Разность «доля смен a → b» минус «доля смен на шуме» и её интервал бутстрапа узлов с возвращением;
    выборка узлов общая для изменения и шума (парный бутстрап), сопоставление типов — заново в каждой."""
    a, b = np.asarray(a), np.asarray(b)
    pairs = [(np.asarray(x), np.asarray(y)) for x, y in noise_pairs]
    n = len(a)
    change = share_changed(a, b, k)
    noise = _noise_share(pairs, k)
    ch, no = np.empty(n_boot), np.empty(n_boot)
    for r in range(n_boot):
        idx = rng.integers(0, n, n)
        ch[r] = share_changed(a[idx], b[idx], k)
        no[r] = _noise_share(pairs, k, idx)
    d = ch - no
    q = [(1 - level) / 2 * 100, (1 + level) / 2 * 100]
    lo, hi = np.percentile(d, q)
    clo, chi = np.percentile(ch, q)
    nlo, nhi = np.percentile(no, q)
    return BootstrapResult(
        change=change,
        noise=noise,
        diff=change - noise,
        lo=float(lo),
        hi=float(hi),
        change_lo=float(clo),
        change_hi=float(chi),
        noise_lo=float(nlo),
        noise_hi=float(nhi),
        n_boot=n_boot,
        exceeds=bool(lo > 0),
    )


# --- Альтернативы: фиксированные типы и эволюционная кластеризация -------------------------------


def procrustes(E: np.ndarray, ref: np.ndarray) -> tuple[np.ndarray, float]:
    """Поворот ``E`` к ``ref`` (ортогональная задача Прокруста, строки — те же узлы). Второе — доля
    суммы квадратов ``ref``, объяснённая повёрнутым ``E``: 1 − ‖E R − ref‖² / ‖ref‖²."""
    from scipy.linalg import orthogonal_procrustes

    R, _ = orthogonal_procrustes(E, ref)
    out = E @ R
    fit = 1.0 - float(((out - ref) ** 2).sum() / (ref**2).sum())
    return out, fit


def sqdist(Z: np.ndarray, C: np.ndarray) -> np.ndarray:
    return ((Z[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)


def nearest(Z: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Номер ближайшего центра (евклидово расстояние; равенство — меньший номер)."""
    return np.argmin(sqdist(Z, C), axis=1)


def centers(Z: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    return np.vstack([Z[labels == j].mean(axis=0) for j in range(k)])


def evo_center(prev_c: np.ndarray, mean: np.ndarray, n_cur: int, n_prev: int, cp: float) -> np.ndarray:
    """Центр эволюционного K-means (Chakrabarti и др., 2006, разд. 4.2): γ = n_cur / (n_cur + n_prev),
    c = [(1 − γ)·cp·c' + γ·(1 − cp)·m] / [(1 − γ)·cp + γ·(1 − cp)]. Авторы нормируют центр к единичной
    длине (их объекты — единичные векторы); в евклидовом пространстве гибрида аналог — веса с суммой 1.
    При n_cur = n_prev (γ = 1/2) это cp·c' + (1 − cp)·m; cp = 0 — обычное среднее."""
    total = n_cur + n_prev
    gamma = n_cur / total if total > 0 else 0.5
    w_prev, w_cur = (1.0 - gamma) * cp, gamma * (1.0 - cp)
    if w_prev + w_cur <= 0:
        return np.array(prev_c if n_cur == 0 else mean, dtype=np.float64)
    return (w_prev * prev_c + w_cur * mean) / (w_prev + w_cur)


def evolutionary_kmeans(
    Z: np.ndarray,
    prev: np.ndarray,
    prev_sizes: np.ndarray,
    cp: float,
    max_iter: int = 300,
    tol: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray]:
    """Эволюционный K-means одного шага: старт — центры прошлого шага ``prev`` (размеры кластеров
    ``prev_sizes``); проход — узлы к ближайшему центру, затем центр j — ``evo_center`` от ближайшего к нему
    центра прошлого шага f(j) и среднего узлов кластера. Пустой кластер сохраняет центр."""
    C = np.array(prev, dtype=np.float64, copy=True)
    prev_sizes = np.asarray(prev_sizes)
    labels = nearest(Z, C)
    for _ in range(max_iter):
        new = np.empty_like(C)
        for j in range(len(C)):
            mem = labels == j
            if not mem.any():
                new[j] = C[j]
                continue
            f = int(np.argmin(((prev - C[j]) ** 2).sum(axis=1)))
            new[j] = evo_center(prev[f], Z[mem].mean(axis=0), int(mem.sum()), int(prev_sizes[f]), cp)
        new_labels = nearest(Z, new)
        shift = float(np.abs(new - C).max())
        C = new
        if np.array_equal(new_labels, labels) and shift <= tol:
            break
        labels = new_labels
    return labels, C
