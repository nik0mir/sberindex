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
