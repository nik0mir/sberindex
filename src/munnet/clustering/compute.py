"""Расчёт кандидатов одного набора входов: сетка «метод × K», допустимость, ICVI против случайного базиса,
устойчивость (бутстрап узлов), объяснимость. Тяжёлые шаги идут задачами в процессах (``parallel``)."""

from __future__ import annotations

import logging
import time
import zlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

from munnet.clustering import methods as M
from munnet.clustering import parallel
from munnet.clustering import stability as ST
from munnet.clustering.inputs import Inputs
from munnet.clustering.params import PARAM_METHODS, RANDOM_METHODS, USES, ClusterParams

log = logging.getLogger(__name__)


def cand_id(method: str, k: int) -> str:
    return f"{method}_k{int(k):02d}"


# --- Задачи для процессов ------------------------------------------------------------------------


def tune_task(method: str):
    st = parallel.STATE
    inp, cp = st["inputs"], st["params"]
    chosen, scan = M.tune_param(method, inp, cp.k_grid, cp.seeds, cp.impl, RANDOM_METHODS)
    return method, chosen, dict(scan.points)


def fit_task(task):
    method, k, param = task
    st = parallel.STATE
    inp, cp = st["inputs"], st["params"]
    pf = M.fit_protocol(method, inp, k, param, cp.seeds, cp.impl, RANDOM_METHODS, {}, cp.hybrid_alpha)
    return method, k, param, pf


def _icvi_block(labels, inp, cp, specs, rng_key: list[int], suffix: str = "", **opts) -> dict:
    """Значения, случайный базис и z-оценки индексов ``specs`` для одного разбиения."""
    from munnet import icvi

    names = list(specs)
    vals = icvi.evaluate(labels, X=inp.X, A=inp.A, names=names, **opts)
    rng = np.random.default_rng([cp.seed, *rng_key])
    base = icvi.random_baseline(
        labels, X=inp.X, A=inp.A, names=names, n_perm=cp.icvi_permutations, rng=rng, **opts
    ).set_index("name")
    row = {}
    for name in names:
        v = float(vals[name])
        mu, sd = float(base.loc[name, "mean"]), float(base.loc[name, "sd"])
        row[f"{name}{suffix}"] = v
        row[f"{name}{suffix}_base_mean"] = mu
        row[f"{name}{suffix}_base_sd"] = sd
        if "n_undefined" in base.columns:
            row[f"{name}{suffix}_base_undefined"] = int(base.loc[name, "n_undefined"])
        row[f"z_{name}{suffix}"] = icvi.zscore(v, mu, sd, str(specs[name]["better"]))
    return row


def icvi_task(task):
    """ICVI кандидата и случайный базис (перестановки меток с теми же размерами кластеров, вариант счёта —
    ``icvi.options``). Для проверок по свойствам индексов (вне предрегистрации) ещё: S_Dbw в варианте own
    и, у разбиений с шумом HDBSCAN, индексы с шумом как отдельным кластером (``*_nc``; без шума — копия)."""
    from munnet import icvi

    cand, labels = task
    st = parallel.STATE
    inp, cp = st["inputs"], st["params"]
    specs = cp.icvi_metrics
    opts = icvi.options({"icvi": cp.icvi_section})
    key = zlib.crc32(cand.encode())
    row = {"cand": cand, **_icvi_block(labels, inp, cp, specs, [17, key], **opts)}
    if "s_dbw" in specs:
        own = _icvi_block(labels, inp, cp, {"s_dbw": specs["s_dbw"]}, [18, key], "_own", s_dbw_density="own")
        row.update(own)
    noise = labels == M.NOISE
    if noise.any():
        nc = labels.copy()
        nc[noise] = labels.max() + 1
        row.update(_icvi_block(nc, inp, cp, specs, [19, key], "_nc", **opts))
    else:
        for name in specs:
            for part in ("", "_base_mean", "_base_sd", "_base_undefined"):
                if f"{name}{part}" in row:
                    row[f"{name}_nc{part}"] = row[f"{name}{part}"]
            row[f"z_{name}_nc"] = row[f"z_{name}"]
    return row


# --- Набор кандидатов ----------------------------------------------------------------------------


@dataclass
class SetResult:
    inputs: Inputs
    cands: pd.DataFrame  # индекс — кандидат
    labels: dict[str, np.ndarray]
    scans: pd.DataFrame  # метод, параметр, K
    boot: pd.DataFrame  # кандидат, повтор, ARI, K выборки
    jaccard: pd.DataFrame  # кандидат, кластер, размер, средний Жаккар лучшего совпадения
    fallback_share: bool
    seconds: dict[str, float]


def feasibility(labels: np.ndarray, cp: ClusterParams, min_share: float) -> dict:
    n = len(labels)
    ok = labels != M.NOISE
    sizes = np.bincount(labels[ok]) if ok.any() else np.array([0])
    k = M.n_clusters(labels)
    noise = float((~ok).mean())
    smin, smax = float(sizes.min() / n), float(sizes.max() / n)
    p = sizes / sizes.sum() if sizes.sum() else sizes
    entropy = float(-(p[p > 0] * np.log(p[p > 0])).sum() / np.log(k)) if k > 1 else float("nan")
    reasons = []
    if not cp.k_range[0] <= k <= cp.k_range[1]:
        reasons.append(f"K = {k} вне {cp.k_range[0]}…{cp.k_range[1]}")
    if smin < min_share:
        reasons.append(f"наименьший тип {smin:.3f} < {min_share}")
    if smax > cp.max_cluster_share:
        reasons.append(f"крупнейший тип {smax:.3f} > {cp.max_cluster_share}")
    if noise > cp.max_noise_share:
        reasons.append(f"шум {noise:.3f} > {cp.max_noise_share}")
    return {
        "k": k,
        "noise_share": noise,
        "min_share": smin,
        "max_share": smax,
        "size_entropy": entropy,
        "feasible": not reasons,
        "infeasible_reason": "; ".join(reasons),
    }


def state_of(inp: Inputs, cp: ClusterParams) -> dict:
    return {"inputs": inp, "params": cp}


def evaluate_set(
    inp: Inputs, cp: ClusterParams, n_boot: int, what: str = "main", pool: parallel.Pool | None = None
) -> SetResult:
    """Все кандидаты набора входов: протокол каждого метода на сетке K, допустимость, ICVI, устойчивость,
    объяснимость. ``pool`` — процессы с входами ``state_of(inp, cp)`` (нет — создаются здесь)."""
    if pool is None:
        with parallel.Pool(state_of(inp, cp), cp.workers) as own:
            return evaluate_set(inp, cp, n_boot, what, own)
    secs: dict[str, float] = {}
    t = time.perf_counter()
    tuned = pool.map(tune_task, [m for m in cp.methods if m in PARAM_METHODS])
    params = {m: chosen for m, chosen, _ in tuned}
    scans = pd.DataFrame(
        [{"method": m, "param": v, "k": kk} for m, _, pts in tuned for v, kk in sorted(pts.items())]
    )
    secs["tune"] = time.perf_counter() - t
    log.info("cluster[%s]: параметры под K подобраны за %.0f с", what, secs["tune"])

    t = time.perf_counter()
    tasks = []
    for m in cp.methods:
        for k in cp.k_grid:
            if m in PARAM_METHODS:
                if k in params.get(m, {}):
                    tasks.append((m, k, float(params[m][k])))
            else:
                tasks.append((m, k, None))
    fits = pool.map(fit_task, tasks)
    secs["fit"] = time.perf_counter() - t
    log.info("cluster[%s]: %d кандидатов за %.0f с", what, len(fits), secs["fit"])

    rows, labels = [], {}
    for m, k, param, pf in fits:
        cid = cand_id(m, k)
        labels[cid] = pf.labels
        rows.append(
            {
                "cand": cid,
                "method": m,
                "family": cp.family_of(m),
                "uses": USES[m],
                "k_target": k,
                "param": np.nan if param is None else float(param),
                "own_score": pf.score,
                "best_seed": np.nan if pf.seed is None else pf.seed,
                "seed_ari": pf.seed_ari,
                "fit_seconds": pf.seconds,
                "bic": pf.extra.get("bic", np.nan),
                "simplicity": cp.simplicity[m],
            }
        )
    cands = pd.DataFrame(rows).set_index("cand")
    fallback = False
    feas = pd.DataFrame({c: feasibility(labels[c], cp, cp.min_cluster_share) for c in cands.index}).T
    if not feas["feasible"].astype(bool).any():
        fallback = True
        log.warning(
            "cluster[%s]: допустимых нет — порог наименьшего типа %.2f", what, cp.min_cluster_share_fallback
        )
        feas = pd.DataFrame(
            {c: feasibility(labels[c], cp, cp.min_cluster_share_fallback) for c in cands.index}
        ).T
    cands = cands.join(feas)
    for col in ("k", "feasible"):
        cands[col] = cands[col].astype(int if col == "k" else bool)
    for col in ("noise_share", "min_share", "max_share", "size_entropy"):
        cands[col] = cands[col].astype("float64")

    t = time.perf_counter()
    ic = pool.map(icvi_task, [(c, labels[c]) for c in cands.index])
    cands = cands.join(pd.DataFrame(ic).set_index("cand"))
    from munnet.clustering.selection import n_defined

    zf = [f"z_{n}" for n in cp.metric_names("features")]
    zg = [f"z_{n}" for n in cp.metric_names("graph")]
    cands["n_metrics_features"] = n_defined(cands, zf).astype(int)
    cands["n_metrics_graph"] = n_defined(cands, zg).astype(int)
    secs["icvi"] = time.perf_counter() - t
    log.info("cluster[%s]: ICVI и случайный базис за %.0f с", what, secs["icvi"])

    t = time.perf_counter()
    btasks = []
    for m in cp.methods:
        settings = [
            (
                c,
                int(cands.loc[c, "k_target"]),
                None if np.isnan(cands.loc[c, "param"]) else float(cands.loc[c, "param"]),
                labels[c],
            )
            for c in cands.index[cands["method"] == m]
        ]
        if settings:
            btasks += [(m, rep, settings) for rep in range(n_boot)]
    bres = [r for rs in pool.map(ST.boot_task, btasks) for r in rs]
    boot = pd.DataFrame([{k: v for k, v in r.items() if k != "jaccard"} for r in bres])
    cands["stability"] = boot.groupby("cand")["ari"].mean()
    cands["stability_sd"] = boot.groupby("cand")["ari"].std()
    jrows = []
    for c in cands.index:
        J = np.vstack([r["jaccard"] for r in bres if r["cand"] == c])
        with np.errstate(all="ignore"):
            mean = np.nanmean(J, axis=0)
        lab = labels[c]
        for cl, v in enumerate(mean):
            jrows.append({"cand": c, "cluster": cl, "size": int((lab == cl).sum()), "jaccard": float(v)})
    jac = pd.DataFrame(jrows)
    cands["jaccard_min"] = jac.groupby("cand")["jaccard"].min()
    cands["jaccard_mean"] = jac.groupby("cand")["jaccard"].mean()
    cands["n_stable"] = jac.assign(s=jac["jaccard"] >= cp.jaccard_stable).groupby("cand")["s"].sum()
    cands["n_dissolved"] = jac.assign(s=jac["jaccard"] < cp.jaccard_dissolved).groupby("cand")["s"].sum()
    secs["bootstrap"] = time.perf_counter() - t
    log.info("cluster[%s]: бутстрап %d повторов за %.0f с", what, n_boot, secs["bootstrap"])

    t = time.perf_counter()
    Mtree = inp.tree_matrix
    cands["interpretability"] = [
        ST.tree_kappa(Mtree, labels[c], cp.tree_depth, cp.cv_folds, cp.seed) for c in cands.index
    ]
    secs["interpretability"] = time.perf_counter() - t
    return SetResult(
        inputs=inp,
        cands=cands,
        labels=labels,
        scans=scans,
        boot=boot,
        jaccard=jac,
        fallback_share=fallback,
        seconds=secs,
    )
