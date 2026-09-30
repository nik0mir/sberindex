"""Анализы, добавленные после предрегистрации (29.09, по замечаниям судьи критерия 3); на выбор не влияют.

- ``threshold_grid`` — сетка порогов допустимости: победитель на обоих уровнях и K гибрида при каждой паре
  «наименьший тип × крупнейший тип», основная цепочка равенств и цепочка с допуском устойчивости;
- ``resolution_limit`` — предел разрешения модульности (Fortunato S., Barthélemy M. Resolution limit in
community
  detection // PNAS. 2007. Vol. 104, no. 1. P. 36–41): сообщество с внутренним весом меньше √(2m) модульность
  может слить с соседним; при разрешении γ порог — √(2m/γ) (толкование для модульности с разрешением);
  там же — связность сообществ (сколько сообществ распадается на части в своём подграфе);
- ``kefrin_task`` — KEFRiN в варианте статьи: признаки как у авторов (z-оценка) и вес сети ξ при ρ = 1.
"""

from __future__ import annotations

import copy
from dataclasses import replace

import numpy as np
import pandas as pd
import scipy.sparse as sp

from munnet.clustering import compute as CO
from munnet.clustering import decide as DE
from munnet.clustering import methods as M
from munnet.clustering import parallel
from munnet.clustering.params import RANDOM_METHODS, ClusterParams


def threshold_grid(res, cp: ClusterParams) -> pd.DataFrame:
    ex = cp.impl.get("extra_checks") or {}
    rows = []
    for smin in ex.get("grid_min_share") or ():
        for smax in ex.get("grid_max_share") or ():
            cp2 = replace(cp, max_cluster_share=float(smax))
            feas = {c: CO.feasibility(res.labels[c], cp2, float(smin))["feasible"] for c in res.cands.index}
            alt = res.cands.copy()
            alt["feasible"] = pd.Series(feas).astype(bool)
            for chain, tol in (("strict", 0.0), ("tolerance", float(cp.tie["stability"]))):
                tl = DE.run_rule(alt, cp2, stab_tol=tol)
                ev = DE.run_rule(alt, cp2, methods=cp.methods, stab_tol=tol)
                hyb = ev.level1.get("hybrid")
                rows.append(
                    {
                        "min_share": float(smin),
                        "max_share": float(smax),
                        "chain": chain,
                        "n_feasible": int(alt["feasible"].sum()),
                        "winner_eligible": tl.winner,
                        "winner_all": ev.winner,
                        "hybrid_k": None if hyb is None else int(alt.loc[hyb.winner, "k"]),
                        "hybrid_feasible_k": ",".join(
                            str(int(k)) for k in alt.loc[(alt["method"] == "hybrid") & alt["feasible"], "k"]
                        ),
                    }
                )
    return pd.DataFrame(rows)


def resolution_limit(res) -> pd.DataFrame:
    """По кандидатам Leiden и Louvain: наименьший внутренний вес сообщества против √(2m/γ) и связность."""
    from scipy.sparse.csgraph import connected_components

    A = res.inputs.A
    m = float(A.sum()) / 2.0
    rows = []
    for c, r in res.cands.loc[res.cands["method"].isin(["leiden", "louvain"])].iterrows():
        lab = res.labels[c]
        gamma = float(r["param"])
        thr = np.sqrt(2 * m / gamma)
        inner, disconnected = [], 0
        for k in np.unique(lab):
            idx = np.flatnonzero(lab == k)
            sub = A[idx][:, idx]
            inner.append(float(sub.sum()) / 2.0)
            if connected_components(sp.csr_matrix(sub), directed=False)[0] > 1:
                disconnected += 1
        inner = np.asarray(inner)
        rows.append(
            {
                "cand": c,
                "method": r["method"],
                "k": int(r["k"]),
                "gamma": gamma,
                "threshold": thr,
                "threshold_gamma1": np.sqrt(2 * m),
                "min_inner_weight": float(inner.min()),
                "n_below": int((inner < thr).sum()),
                "n_disconnected": disconnected,
            }
        )
    return pd.DataFrame(rows)


def kefrin_task(task) -> dict:
    """KEFRiN при признаках ``features`` и весе сети ξ (ρ = 1) с данным K: допустимость и ARI с итогом."""
    from sklearn.metrics import adjusted_rand_score

    features, xi, k, final = task
    st = parallel.STATE
    inp, cp = st["inputs"], st["params"]
    impl = copy.deepcopy(dict(cp.impl))
    impl["shalileh_mirkin"] = {
        **dict(impl.get("shalileh_mirkin") or {}),
        "features": features,
        "rho": 1.0,
        "xi": xi,
    }
    pf = M.fit_protocol("shalileh_mirkin", inp, k, None, cp.seeds, impl, RANDOM_METHODS)
    f = CO.feasibility(pf.labels, cp, cp.min_cluster_share)
    return {
        "features": features,
        "xi_over_rho": float(xi),
        "k": k,
        "feasible": bool(f["feasible"]),
        "min_share": f["min_share"],
        "max_share": f["max_share"],
        "ari_final": float(adjusted_rand_score(final, pf.labels)),
    }


def kefrin_tasks(cp: ClusterParams, final: np.ndarray) -> list[tuple]:
    kc = cp.impl.get("kefrin_curve") or {}
    return [
        (str(f), float(x), int(k), final)
        for f in kc.get("features") or ()
        for x in kc.get("xi_over_rho") or ()
        for k in cp.k_grid
    ]


def kefrin_graph_share(inp, xi: float, features: str) -> float:
    """Доля сети в общем разбросе данных KEFRiN при весе ξ (ρ = 1)."""
    from munnet.clustering import kefrin as KF

    links = KF.ModularityLinks(inp.A)
    mean_row = np.asarray(links.A.mean(axis=0)).ravel()
    lam = (mean_row[None, :], np.array([links.d.mean()]))
    sp_graph = float((links.norms2() - 2 * links.dots(lam)[:, 0] + links.center_norms2(lam)[0]).sum())
    Y = inp.X if features == "inputs" else (inp.X - inp.X.mean(axis=0)) / inp.X.std(axis=0)
    sp_feat = float(((Y - Y.mean(axis=0)) ** 2).sum())
    return xi * sp_graph / (xi * sp_graph + sp_feat)


# --- Шум случайного базиса и seed (ворота этапа 3; добавлено после предрегистрации) -------------------


def icvi_seed_task(task) -> dict:
    """ICVI кандидата с случайным базисом другого seed (разбиение то же)."""
    cand, labels, seed = task
    st = parallel.STATE
    cp = st["params"]
    old = parallel.STATE["params"]
    parallel.STATE["params"] = replace(cp, seed=int(seed))
    try:
        row = CO.icvi_task((cand, labels))
    finally:
        parallel.STATE["params"] = old
    row["seed"] = int(seed)
    return row


def seed_list(cp: ClusterParams) -> list[int]:
    return [cp.seed + i for i in range(int(cp.impl.get("seed_check", 5)))]


def quality_tolerance(pool, res, cp: ClusterParams) -> tuple[dict[str, float], pd.DataFrame]:
    """Допуск ничьей критериев качества по шуму случайного базиса: для каждого seed базиса — критерии качества
    на множестве всех допустимых кандидатов (ранги z-оценок, как в правиле), SD по seed у каждого кандидата,
    медиана по кандидатам. Отступление от предрегистрации (там допуск 0)."""
    from munnet.clustering import selection as S

    feas = [c for c in res.cands.index if bool(res.cands.loc[c, "feasible"])]
    zf, zg = DE.z_cols(cp, "features"), DE.z_cols(cp, "graph")
    tasks = [(c, res.labels[c], s) for s in seed_list(cp)[1:] for c in feas]
    rows = pool.map(icvi_seed_task, tasks)
    z = [res.cands.loc[feas, zf + zg].assign(seed=cp.seed)]
    extra = pd.DataFrame(rows).set_index("cand")
    for s, d in extra.groupby("seed"):
        z.append(d.loc[feas, zf + zg].assign(seed=s))
    crit = []
    for d in z:
        t = S.criteria_table(d, zf, zg, ["quality_features", "quality_graph"])
        crit.append(t.assign(seed=int(d["seed"].iloc[0])))
    crit = pd.concat(crit).rename_axis("cand").reset_index()
    sd = crit.groupby("cand")[["quality_features", "quality_graph"]].std()
    tol = {c: float(sd[c].median()) for c in sd.columns}
    zz = pd.concat(z).rename_axis("cand").reset_index()
    return tol, crit.merge(zz, on=["cand", "seed"])


def with_tolerance(cp: ClusterParams, tol: dict[str, float]) -> ClusterParams:
    return replace(cp, tie={**cp.tie, **tol})


def decide_rows(cands, cp: ClusterParams, cp_tol: ClusterParams, seed: int) -> pd.DataFrame:
    """Победители всех проверок на обоих уровнях по предрегистрированному правилу и с допуском качества."""
    out = []
    for rule, c in (("prereg", cp), ("tolerance", cp_tol)):
        d = DE.decide(cands, c)
        ch = d.checks.assign(rule=rule, seed=seed, kind="check")
        orders = d.orders.rename(columns={"order": "check"}).assign(rule=rule, seed=seed, kind="order")
        out += [ch, orders]
    return pd.concat(out, ignore_index=True)


def seed_frequency(res, cp: ClusterParams, cp_tol: ClusterParams) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Уровни выбора при seed, seed + 1, …: методы, случайный базис и бутстрап — от нового seed, входы те же.
    Ответ: победители по seed и частота «проверка × победитель»; ARI итогового кандидата с основным
    разбиением."""
    from sklearn.metrics import adjusted_rand_score

    parts = [decide_rows(res.cands, cp, cp_tol, cp.seed)]
    ari = []
    for s in seed_list(cp)[1:]:
        cp_s = replace(cp, seed=s, seeds=tuple(s + i for i in range(len(cp.seeds))))
        res_s = CO.evaluate_set(res.inputs, cp_s, cp.bootstrap, f"seed{s}")
        parts.append(
            decide_rows(
                res_s.cands,
                cp_s,
                with_tolerance(cp_s, {k: cp_tol.tie[k] for k in ("quality_features", "quality_graph")}),
                s,
            )
        )
        for c in res.labels:
            if c in res_s.labels:
                ari.append(
                    {"seed": s, "cand": c, "ari": float(adjusted_rand_score(res.labels[c], res_s.labels[c]))}
                )
    runs = pd.concat(parts, ignore_index=True)
    freq = (
        runs.groupby(["rule", "scope", "kind", "check", "winner"], dropna=False)
        .size()
        .rename("n_seeds")
        .reset_index()
    )
    return runs, freq.merge(
        pd.DataFrame(ari).groupby("cand")["ari"].min().rename("min_ari_labels").reset_index(),
        left_on="winner",
        right_on="cand",
        how="left",
    ).drop(columns="cand")
