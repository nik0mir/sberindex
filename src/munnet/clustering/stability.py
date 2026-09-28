"""Устойчивость разбиений и объяснимость типов.

- **Бутстрап узлов** (``clustering.stability``): 80% узлов без возвращения, у каждой выборки свой seed; метод
  с той же настройкой (K или параметр) — на подвыборке, методы по графу — на индуцированном подграфе. Критерий
  выбора — средний ARI разбиения на всех узлах (ограниченного на выборку) с разбиением выборки. По каждому
  кластеру — Жаккар лучшего совпадения (Hennig, 2007): max_j |C ∩ B_j| / |C ∪ B_j| по кластерам выборки,
  среднее по повторам; от 0,75 — устойчив, ниже 0,5 — распадается. Шум HDBSCAN (−1) в ARI — отдельная метка,
  в Жаккаре не участвует.
- **Возмущение рёбер** (вне правила выбора): удалить долю рёбер G, ARI с разбиением на всей сети.
- **Объяснимость** (``clustering.interpretability``): каппа Коэна дерева глубины 3 «17 признаков → тип»
  на стратифицированной перекрёстной проверке.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from munnet.clustering import methods as M
from munnet.clustering import parallel

NOISE = M.NOISE


def subsample(n: int, frac: float, seed: int, rep: int) -> np.ndarray:
    """Номера узлов выборки повтора ``rep`` (отсортированы)."""
    rng = np.random.default_rng([seed, 11, rep])
    return np.sort(rng.choice(n, size=int(round(frac * n)), replace=False))


def task_seed(seed: int, *key: int) -> int:
    return int(np.random.SeedSequence([seed, *key]).generate_state(1)[0] % (2**31 - 1))


def best_match_jaccard(full: np.ndarray, boot: np.ndarray, k_full: int | None = None) -> np.ndarray:
    """Жаккар лучшего совпадения для каждого кластера ``full`` (метки 0…K−1, на тех же узлах, что ``boot``);
    ``k_full`` — K разбиения на всех узлах; кластер, не попавший в выборку, — NaN."""
    if k_full is None:
        k_full = int(full.max()) + 1 if (full != NOISE).any() else 0
    out = np.full(k_full, np.nan)
    bl = [b for b in np.unique(boot) if b != NOISE]
    for c in range(k_full):
        a = full == c
        na = int(a.sum())
        if not na:
            continue
        best = 0.0
        for b in bl:
            m = boot == b
            inter = int((a & m).sum())
            if inter:
                best = max(best, inter / (na + int(m.sum()) - inter))
        out[c] = best
    return out


def boot_task(task) -> list[dict]:
    """Один повтор бутстрапа одного метода по всем его настройкам. ``task`` — (метод, повтор, настройки),
    настройка — (кандидат, K, параметр, метки на всех узлах)."""
    from sklearn.metrics import adjusted_rand_score

    method, rep, settings = task
    st = parallel.STATE
    inp, cp = st["inputs"], st["params"]
    idx = subsample(inp.n, cp.subsample, cp.seed, rep)
    sub = inp.subset(idx)
    seed = task_seed(cp.seed, 13, rep)
    cache: dict = {}
    out = []
    for cand, k, param, full in settings:
        f = M.fit(method, sub, k, param, seed, cp.impl, cache, cp.hybrid_alpha)
        boot = M.canonical(f.labels)
        full_sub = np.asarray(full)[idx]
        out.append(
            {
                "cand": cand,
                "rep": rep,
                "ari": float(adjusted_rand_score(full_sub, boot)),
                "k_boot": M.n_clusters(boot),
                "jaccard": best_match_jaccard(full_sub, boot, M.n_clusters(np.asarray(full))),
            }
        )
    return out


def drop_edges(A: sp.csr_matrix, frac: float, rng: np.random.Generator) -> sp.csr_matrix:
    """Копия G без доли ``frac`` рёбер (неориентированных), выбранных случайно."""
    U = sp.triu(A, k=1).tocoo()
    keep = rng.random(U.nnz) >= frac
    B = sp.coo_matrix((U.data[keep], (U.row[keep], U.col[keep])), shape=A.shape)
    return (B + B.T).tocsr()


def perturb_task(task) -> dict:
    """Повтор возмущения рёбер: (кандидат, метод, K, параметр, повтор, метки на всей сети)."""
    from dataclasses import replace

    from sklearn.metrics import adjusted_rand_score

    cand, method, k, param, rep, full = task
    st = parallel.STATE
    inp, cp = st["inputs"], st["params"]
    frac = float((cp.impl.get("edge_perturbation") or {}).get("drop", 0.1))
    rng = np.random.default_rng([cp.seed, 19, rep])
    pert = replace(inp, A=drop_edges(inp.A, frac, rng))
    f = M.fit(method, pert, k, param, task_seed(cp.seed, 23, rep), cp.impl, {}, cp.hybrid_alpha)
    return {"cand": cand, "rep": rep, "ari": float(adjusted_rand_score(full, M.canonical(f.labels)))}


def tree_kappa(M_: np.ndarray, labels: np.ndarray, depth: int, folds: int, seed: int) -> float:
    """Каппа Коэна меток и предсказаний дерева глубины ``depth`` на стратифицированной ``folds``-кратной
    проверке (шум HDBSCAN — отдельный класс)."""
    import warnings

    from sklearn.metrics import cohen_kappa_score
    from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict
    from sklearn.tree import DecisionTreeClassifier

    labels = np.asarray(labels)
    if len(np.unique(labels)) < 2:
        return float("nan")
    _, counts = np.unique(labels, return_counts=True)
    cv = (
        StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        if counts.min() >= 2
        else KFold(n_splits=folds, shuffle=True, random_state=seed)
    )
    tree = DecisionTreeClassifier(max_depth=depth, random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pred = cross_val_predict(tree, M_, labels, cv=cv)
    return float(cohen_kappa_score(labels, pred))


def tree_rules(M_: np.ndarray, labels: np.ndarray, names, depth: int, seed: int) -> str:
    """Текст дерева на всех узлах (для описания типов)."""
    from sklearn.tree import DecisionTreeClassifier, export_text

    tree = DecisionTreeClassifier(max_depth=depth, random_state=seed).fit(M_, labels)
    return export_text(tree, feature_names=list(names), decimals=2)
