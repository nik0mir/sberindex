"""Метод Шалилеха и Миркина для сетей с признаками — KEFRiN (K-means, расширенный на сеть с признаками).

Первоисточник: Shalileh S., Mirkin B. Community partitioning over feature-rich networks using an extended
K-means
method // Entropy. 2022. Vol. 24, no. 5. Art. 626. https://doi.org/10.3390/e24050626 (открытый доступ, CC
BY 4.0).
Реализовано по тексту статьи; код авторов не использовался.

Модель восстановления данных (разд. 2.1–2.2, с. 4–5, формулы 1 и 4): признаки y_iv = Σ_k c_kv s_ik + f_iv,
связи в режиме несуммируемости p_ij = Σ_k λ_kj s_ik + e_ij. Критерий — взвешенная сумма квадратов невязок
(формула 5, с. 5):

    F(S, c, λ) = ρ Σ_{i,v} (y_iv − Σ_k c_kv s_ik)² + ξ Σ_{i,j} (p_ij − Σ_k λ_kj s_ik)²,

при разбиении это (формула 6, с. 6)

    F = Σ_k Σ_{i∈S_k} [ρ ‖y_i − c_k‖² + ξ ‖p_i − λ_k‖²].

Чередующаяся минимизация (с. 6): правило минимального расстояния — узел i в кластер с наименьшим
d(i, k) = ρ‖y_i − c_k‖² + ξ‖p_i − λ_k‖²; центры при заданном разбиении — средние по кластеру
(формула 7): c_kv = Σ_{i∈S_k} y_iv / |S_k|, λ_kj = Σ_{i∈S_k} p_ij / |S_k|. Остановка — разбиение не
изменилось.
Старт (с. 6, K-Means++ с MaxMin): первый центр — строки y_r и p_r случайного узла r, каждый следующий — узел
с наибольшей суммой комбинированных расстояний до уже выбранных центров. Стандартизация (разд. 4.1, с. 11):
признаки — z-оценка (Z), связи — модульностью (M): p_ij − p_i+·p_+j / p_++; веса ρ = ξ = 1 (разд. 2.4, с. 7).
Вариант KEFRiNe — квадрат евклидова расстояния (KEFRiNc и KEFRiNm — косинус и Манхэттен — здесь не нужны).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp


@dataclass(frozen=True)
class KefrinResult:
    labels: np.ndarray
    criterion: float  # F — формула 6
    iterations: int
    converged: bool


def modularity_standardize(P) -> np.ndarray:
    """Стандартизация связей (M), разд. 4.1: p_ij − p_i+·p_+j / p_++ (плотная матрица)."""
    P = P.toarray() if sp.issparse(P) else np.asarray(P, dtype=np.float64)
    P = np.asarray(P, dtype=np.float64)
    r = P.sum(axis=1)
    c = P.sum(axis=0)
    tot = P.sum()
    if tot <= 0:
        return P.copy()
    return P - np.outer(r, c) / tot


class DenseLinks:
    """Матрица связей P как есть (плотная): строки p_i, центры λ_k — средние строк кластера."""

    def __init__(self, P) -> None:
        self.P = np.asarray(P, dtype=np.float64)
        self.n = len(self.P)
        self._norms2 = (self.P**2).sum(axis=1)

    def norms2(self) -> np.ndarray:
        return self._norms2

    def seeds(self, idx) -> np.ndarray:
        return self.P[np.asarray(idx)].copy()

    def centers(self, labels: np.ndarray, K: int, prev: np.ndarray) -> np.ndarray:
        L = prev.copy()
        for k in range(K):
            m = labels == k
            if m.any():
                L[k] = self.P[m].mean(axis=0)
        return L

    def dots(self, L: np.ndarray) -> np.ndarray:
        return self.P @ L.T

    def center_norms2(self, L: np.ndarray) -> np.ndarray:
        return (L**2).sum(axis=1)


class ModularityLinks:
    """P = A − d·dᵀ / T (стандартизация (M) симметричной разреженной A: d — суммы строк, T — сумма всех весов)
    без плотной матрицы: центр кластера λ_k = ā_k − δ_k·d / T (ā_k — средняя строка A, δ_k — средняя степень),
    скалярные произведения — через разреженное A. Числа те же, что у
    ``DenseLinks(modularity_standardize(A))``."""

    def __init__(self, A) -> None:
        self.A = sp.csr_matrix(A, dtype=np.float64)
        self.n = self.A.shape[0]
        self.d = np.asarray(self.A.sum(axis=1)).ravel()
        self.T = float(self.d.sum())
        self.dd = float(self.d @ self.d)
        self.ad = self.A @ self.d  # a_i · d
        a2 = np.asarray(self.A.multiply(self.A).sum(axis=1)).ravel()
        self._norms2 = a2 - 2.0 * self.d * self.ad / self.T + self.d**2 * self.dd / self.T**2

    def norms2(self) -> np.ndarray:
        return self._norms2

    def seeds(self, idx) -> tuple[np.ndarray, np.ndarray]:
        idx = np.asarray(idx)
        return self.A[idx].toarray(), self.d[idx].astype(np.float64)

    def centers(self, labels: np.ndarray, K: int, prev) -> tuple[np.ndarray, np.ndarray]:
        Abar, delta = prev[0].copy(), prev[1].copy()
        for k in range(K):
            m = labels == k
            if m.any():
                Abar[k] = np.asarray(self.A[m].mean(axis=0)).ravel()
                delta[k] = self.d[m].mean()
        return Abar, delta

    def dots(self, L) -> np.ndarray:
        Abar, delta = L
        T = self.T
        a_abar = np.asarray(self.A @ Abar.T)  # n × K
        d_abar = Abar @ self.d  # K
        return (
            a_abar
            - np.outer(self.ad, delta) / T
            - np.outer(self.d, d_abar) / T
            + np.outer(self.d, delta) * self.dd / T**2
        )

    def center_norms2(self, L) -> np.ndarray:
        Abar, delta = L
        return (Abar**2).sum(axis=1) - 2.0 * delta * (Abar @ self.d) / self.T + delta**2 * self.dd / self.T**2


def as_links(P):
    """Разреженная матрица — стандартизация (M) без плотной матрицы; плотная — как есть."""
    return ModularityLinks(P) if sp.issparse(P) else DenseLinks(P)


def _distances(Y, links, C, L, rho: float, xi: float, y2: np.ndarray) -> np.ndarray:
    dy = y2[:, None] - 2.0 * Y @ C.T + (C**2).sum(axis=1)[None, :]
    dp = links.norms2()[:, None] - 2.0 * links.dots(L) + links.center_norms2(L)[None, :]
    return rho * np.maximum(dy, 0.0) + xi * np.maximum(dp, 0.0)


def combined_distances(Y, P, C, L, rho: float, xi: float) -> np.ndarray:
    """d(i, k) = ρ‖y_i − c_k‖² + ξ‖p_i − λ_k‖² для всех узлов и центров (n × K), плотные P и L."""
    Y = np.asarray(Y, dtype=np.float64)
    return _distances(Y, DenseLinks(P), np.asarray(C), np.asarray(L), rho, xi, (Y**2).sum(axis=1))


def criterion(Y, P, labels, rho: float, xi: float) -> float:
    """F по формуле 5: невязки признаков и связей при центрах — средних по кластерам (формула 7)."""
    Y = np.asarray(Y, dtype=np.float64)
    links = P if isinstance(P, DenseLinks | ModularityLinks) else as_links(P)
    K = int(labels.max()) + 1
    C = np.vstack(
        [Y[labels == k].mean(axis=0) if (labels == k).any() else np.zeros(Y.shape[1]) for k in range(K)]
    )
    L = links.centers(labels, K, links.seeds(np.zeros(K, dtype=int)))
    D = _distances(Y, links, C, L, rho, xi, (Y**2).sum(axis=1))
    return float(D[np.arange(len(Y)), labels].sum())


def maxmin_seeds(Y, P, K: int, first: int, rho: float, xi: float) -> np.ndarray:
    """Номера узлов-центров старта: ``first``, затем узел с наибольшей суммой d(i, ·) до выбранных (с. 6)."""
    Y = np.asarray(Y, dtype=np.float64)
    links = P if isinstance(P, DenseLinks | ModularityLinks) else as_links(P)
    y2 = (Y**2).sum(axis=1)
    chosen = [int(first)]
    total = np.zeros(len(Y))
    while len(chosen) < K:
        j = chosen[-1]
        total += _distances(Y, links, Y[[j]], links.seeds([j]), rho, xi, y2)[:, 0]
        cand = total.copy()
        cand[chosen] = -np.inf
        chosen.append(int(np.argmax(cand)))
    return np.asarray(chosen)


def kefrin(Y, P, K: int, first: int, rho: float = 1.0, xi: float = 1.0, max_iter: int = 100) -> KefrinResult:
    """KEFRiNe на стандартизованных признаках ``Y`` (n × V) и связях ``P``: плотная матрица — как есть,
    разреженная — стандартизуется модульностью (M) без плотной матрицы (``ModularityLinks``).

    Пустой кластер (узлы ушли ко всем другим центрам) сохраняет прежний центр: в статье этот случай
    не разобран, а удалять кластер — значит менять K.
    """
    Y = np.asarray(Y, dtype=np.float64)
    links = P if isinstance(P, DenseLinks | ModularityLinks) else as_links(P)
    n = len(Y)
    if not 1 <= K <= n:
        raise ValueError(f"KEFRiN: K = {K} при {n} узлах")
    y2 = (Y**2).sum(axis=1)
    seeds = maxmin_seeds(Y, links, K, first, rho, xi)
    C, L = Y[seeds].copy(), links.seeds(seeds)
    labels = np.full(n, -1)
    converged = False
    it = 0
    while it < max_iter:
        it += 1
        D = _distances(Y, links, C, L, rho, xi, y2)
        new = np.argmin(D, axis=1)
        if np.array_equal(new, labels):
            converged = True
            break
        labels = new
        for k in range(K):
            m = labels == k
            if m.any():
                C[k] = Y[m].mean(axis=0)
        L = links.centers(labels, K, L)
    return KefrinResult(
        labels=labels, criterion=criterion(Y, links, labels, rho, xi), iterations=it, converged=converged
    )
