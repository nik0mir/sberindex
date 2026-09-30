"""Синтетика «когда помогает граф» (``clustering.synthetic``): известная истина, сила сигнала в графе
и в признаках задаётся раздельно.

Основной генератор повторяет устройство реальной сети: у каждой группы свой центр в координатах «корзины»
(разнос центров — ``graph_signal`` стандартных отклонений шума), граф — kNN (k = ``knn_k``, объединение
списков) по гауссову ядру расстояния между координатами, как правило ``basket_dist`` этапа network;
признаки X —
``informative`` признаков с центрами групп (разнос ``feature_signal``) и шумовые признаки. Проверка —
посаженный блочный граф ``networkx.stochastic_block_model`` со средней степенью ``sbm_degree`` и долей рёбер
между группами ``sbm_mixing`` (при доле (K − 1)/K граф не несёт сигнала). У блочного графа нет координат
корзины, поэтому базовая линия ``kmeans_joint`` на нём не считается.

Мера — ARI разбиения метода с истиной; итог метода — по тем же правилам, что на данных (``fit_protocol``),
K известно, у Leiden разрешение подбирается под K (``tune_param``; K недостижимо — ближайшее).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from munnet.clustering import methods as M
from munnet.clustering import parallel
from munnet.clustering.inputs import Inputs, graph_matrix, robust_scale
from munnet.clustering.params import PARAM_METHODS, RANDOM_METHODS
from munnet.network import graph as NG
from munnet.network import rules as R


def _labels(n: int, k: int) -> np.ndarray:
    return np.repeat(np.arange(k), int(np.ceil(n / k)))[:n]


def _features(truth, k, p, informative, signal, rng) -> np.ndarray:
    C = np.zeros((k, p))
    C[:, :informative] = rng.normal(0.0, signal, size=(k, informative))
    return C[truth] + rng.normal(size=(len(truth), p))


def _inputs(name: str, X: np.ndarray, B: np.ndarray, A) -> Inputs:
    n = len(X)
    Xs, used = robust_scale(pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])]))
    ids = np.arange(1, n + 1, dtype=np.int64)
    return Inputs(
        name=name,
        ids=ids,
        X=Xs,
        x_names=tuple(f"x{i}" for i in range(X.shape[1])),
        X_raw=pd.DataFrame(X, index=ids),
        B=B,
        b_names=tuple(f"b{i}" for i in range(B.shape[1])),
        A=A,
        groups=np.zeros(n, dtype=np.int64),
        table=pd.DataFrame({"territory_id": ids}),
        scale_used=used,
    )


def make_knn(
    n, k, p, informative, n_basket, graph_signal, feature_signal, knn_k, seed
) -> tuple[Inputs, np.ndarray]:
    """Основной генератор: граф — kNN по координатам «корзины» с центрами групп."""
    rng = np.random.default_rng(seed)
    truth = rng.permutation(_labels(n, k))
    Cb = rng.normal(0.0, graph_signal, size=(k, n_basket))
    B = Cb[truth] + rng.normal(size=(n, n_basket))
    X = _features(truth, k, p, informative, feature_signal, rng)
    D = R.euclid_matrix(B)
    S = R.gaussian_similarity(D, R.median_offdiag(D))
    e = NG.knn_edges(S, knn_k)
    ids = np.arange(1, n + 1)
    A = graph_matrix(pd.DataFrame({"source": ids[e["i"]], "target": ids[e["j"]], "weight": e["weight"]}), ids)
    return _inputs("knn", X, B, A), truth


def make_sbm(n, k, p, informative, feature_signal, mixing, degree, seed) -> tuple[Inputs, np.ndarray]:
    """Проверка: посаженный блочный граф (веса рёбер — 1) и те же признаки."""
    import networkx as nx

    rng = np.random.default_rng(seed)
    sizes = np.bincount(_labels(n, k), minlength=k).tolist()
    nk = sizes[0]
    p_out = mixing * degree / (n - nk)
    p_in = min(1.0, (1.0 - mixing) * degree / (nk - 1))
    probs = [[p_in if a == b else p_out for b in range(k)] for a in range(k)]
    g = nx.stochastic_block_model(sizes, probs, seed=int(rng.integers(2**31 - 1)))
    truth_sorted = np.repeat(np.arange(k), sizes)
    order = rng.permutation(n)  # перемешать номера узлов: истина не совпадает с порядком
    truth = np.empty(n, dtype=np.int64)
    truth[order] = truth_sorted
    ids = np.arange(1, n + 1)
    edges = np.array(list(g.edges()), dtype=np.int64).reshape(-1, 2)
    src, dst = order[edges[:, 0]], order[edges[:, 1]]
    A = graph_matrix(pd.DataFrame({"source": ids[src], "target": ids[dst], "weight": 1.0}), ids)
    X = _features(truth, k, p, informative, feature_signal, rng)
    return _inputs("sbm", X, np.zeros((n, 0)), A), truth


def _run_methods(inp: Inputs, truth: np.ndarray, k: int, methods, seeds, impl, alpha) -> dict[str, float]:
    from sklearn.metrics import adjusted_rand_score

    out = {}
    for m in methods:
        if m == "kmeans_joint" and inp.B.shape[1] == 0:
            out[m] = float("nan")
            continue
        param = None
        if m in PARAM_METHODS:
            chosen, scan = M.tune_param(m, inp, [k], seeds, impl, RANDOM_METHODS)
            if k in chosen:
                param = chosen[k]
            else:  # K недостижимо: ближайшее
                param = min(scan.points, key=lambda v: (abs(scan.points[v] - k), v))
        pf = M.fit_protocol(m, inp, k, param, seeds, impl, RANDOM_METHODS, {}, alpha)
        out[m] = float(adjusted_rand_score(truth, pf.labels))
    return out


def cell_task(task) -> dict:
    """Одна ячейка сетки и один набор данных: ARI всех методов с истиной."""
    design, gs, fs, rep = task
    st = parallel.STATE
    sy, impl, alpha, base_seed = st["synthetic"], st["impl"], st["alpha"], st["seed"]
    k = int(sy["k_true"])
    seed = int(
        np.random.SeedSequence([base_seed, 41, rep, int(gs * 100), int(fs * 100)]).generate_state(1)[0]
    )
    common = dict(n=int(sy["n_nodes"]), k=k, p=int(sy["n_features"]), informative=int(sy["informative"]))
    if design == "knn":
        inp, truth = make_knn(
            **common,
            n_basket=int(sy["n_basket"]),
            graph_signal=gs,
            feature_signal=fs,
            knn_k=int(sy["knn_k"]),
            seed=seed,
        )
    else:
        inp, truth = make_sbm(
            **common, feature_signal=fs, mixing=gs, degree=float(sy["sbm_degree"]), seed=seed
        )
    seeds = tuple(base_seed + i for i in range(int(sy["seeds"])))
    ari = _run_methods(inp, truth, k, tuple(sy["methods"]), seeds, impl, alpha)
    deg = float(inp.A.getnnz(axis=1).mean())
    return {"design": design, "graph_signal": gs, "feature_signal": fs, "rep": rep, "mean_degree": deg, **ari}


def tasks(sy) -> list[tuple]:
    out = []
    for rep in range(int(sy["repeats"])):
        for gs in sy["graph_signal"]:
            for fs in sy["feature_signal"]:
                out.append(("knn", float(gs), float(fs), rep))
        for mix in sy.get("sbm_mixing") or ():
            for fs in sy["feature_signal"]:
                out.append(("sbm", float(mix), float(fs), rep))
    return out


TIE, NONE = "tie", "none"


def summarize(
    raw: pd.DataFrame, methods, families: dict[str, str], min_ari: float = 0.05, ci_level: float = 0.95
) -> pd.DataFrame:
    """Средний ARI по наборам данных в каждой ячейке и его интервал (t-распределение), победитель ячейки
    и выигрыш от графа: лучший метод на (G, X) минус лучший метод только по X.

    Победитель — метод с наибольшим средним, если его интервал не перекрывается с интервалом второго;
    иначе ``tie`` (``winner_set`` — все методы, чей интервал перекрывает интервал лучшего); если лучший
    средний
    ARI ниже ``min_ari`` — ``none``: в ячейке никто не восстанавливает группы."""
    from scipy.stats import t as student

    g = raw.groupby(["design", "graph_signal", "feature_signal"])[list(methods)]
    mean = g.mean()
    sd = g.std()
    n = g.count()
    half = sd / np.sqrt(n) * student.ppf(0.5 + ci_level / 2, np.maximum(n - 1, 1))
    out = (
        mean.join(sd.add_suffix("_sd"))
        .join((mean - half).add_suffix("_lo"))
        .join((mean + half).add_suffix("_hi"))
    )
    out["n_sets"] = n.max(axis=1)
    avail = mean.dropna(axis=1, how="all")
    out["best"] = avail.idxmax(axis=1)
    out["winner_ari"] = avail.max(axis=1)
    winners, sets = [], []
    for idx in avail.index:
        row = avail.loc[idx].dropna().sort_values(ascending=False)
        best = row.index[0]
        lo = float((mean - half).loc[idx, best])
        hi = (mean + half).loc[idx, row.index]
        tied = [m for m in row.index if m == best or float(hi[m]) >= lo]
        if row.iloc[0] < min_ari:
            winners.append(NONE)
            sets.append("")
        elif len(tied) > 1:
            winners.append(TIE)
            sets.append(",".join(tied))
        else:
            winners.append(best)
            sets.append(best)
    out["winner"] = winners
    out["winner_set"] = sets
    feat = [m for m in methods if families.get(m) == "features"]
    graph = [m for m in methods if families.get(m) == "graph"]
    both = [m for m in methods if families.get(m) in ("attributed", "fusion")]
    out["best_features"] = mean[feat].max(axis=1)
    out["best_graph"] = mean[graph].max(axis=1)
    out["best_both"] = mean[both].max(axis=1)
    out["graph_gain"] = out["best_both"] - out["best_features"]
    out["feature_gain"] = out["best_both"] - out["best_graph"]
    return out.reset_index()
