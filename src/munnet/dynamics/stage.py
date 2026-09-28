"""Этап dynamics: как типы локальных экономик меняются во времени (PLAN.md, этап 3; предрегистрация
``dynamics.tracking``).

1. Входы: итог этапа cluster (``outputs/cluster/final.json``, ``cluster_final``), сети окон — функциями этапа
   network (сверка с ``network_windows``), X окна — функциями ``clustering.inputs``.
2. Разбиения 13 скользящих окон, нечётных и чётных месяцев 2023 и 2024 годов и 24 месяцев (сверка
   с ``cluster_final``) — метод и K итога.
3. Три способа отслеживания (основной — перефит и сопоставление), изменение 2023 → 2024 против шума, надёжные
   переходы, события MONIC, проверка сюжета о маркетплейсах, сравнение способов.
4. Выходы: ``data/processed/dynamics_history.parquet``, таблицы и ``facts.json`` в ``outputs/dynamics``,
   рисунки и отчёт ``docs/dynamics.md``.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from munnet.clustering import inputs as CI
from munnet.clustering import methods as M
from munnet.clustering.params import RANDOM_METHODS
from munnet.config import Config
from munnet.contracts import (
    CLUSTER_FINAL,
    NETWORK_WINDOW_NODES,
    PARTS,
    MissingInputError,
    QCError,
    read_table,
    write_table,
)
from munnet.dynamics import compute as C
from munnet.dynamics import tracking as T
from munnet.dynamics import windows as W
from munnet.dynamics.params import APPROACHES, EVOLUTIONARY, FIXED, MAIN, DynParams
from munnet.dynamics.schemas import DYNAMICS_HISTORY, STATUSES
from munnet.eda.base import display_names, write_csv, write_json
from munnet.network.params import NetworkParams

log = logging.getLogger(__name__)

CLUSTER_HINT = "сначала запустите этап cluster: python -m munnet cluster"
CLR_TOL = 1e-9


def _read_final(cfg: Config) -> dict:
    path = cfg.dir("outputs") / "cluster" / "final.json"
    if not path.exists():
        raise MissingInputError(f"нет {path}: {CLUSTER_HINT}")
    fin = json.loads(path.read_text(encoding="utf-8"))
    inp = fin["inputs"]
    if inp.get("impute") != "region_median" or inp.get("scale") != "robust":
        raise ValueError(f"dynamics: X итога собран не как clustering.inputs.build_X: {inp}")
    return fin


def _cluster_params(fin: dict, p: DynParams) -> SimpleNamespace:
    """Параметры входов в виде, который ждёт ``clustering.inputs.load_inputs``."""
    g = fin["inputs"]["graph"]
    if int(fin["inputs"]["place_year"]) != p.place_year:
        raise ValueError(
            f"dynamics: признаки места итога — {fin['inputs']['place_year']} год, "
            f"а предрегистрация требует {p.place_year}"
        )
    return SimpleNamespace(
        graph_rule=str(g["rule"]),
        graph_sparsify=str(g["sparsify"]),
        graph_k=int(g["k"]),
        attributes=tuple(fin["inputs"]["features"]),
        place_year=p.place_year,
    )


def _check_clr(ns, first: W.Period, last: W.Period, processed: Path) -> float:
    """Корзина первого и последнего окна (``window_clr``) = ``network_window_nodes`` (QCError иначе)."""
    wn = read_table(processed / "network_window_nodes.parquet", NETWORK_WINDOW_NODES)
    worst = 0.0
    for per in (first, last):
        ref = wn.loc[(wn["window_kind"].astype(str) == "rolling") & (wn["window"] == per.name)]
        ref = ref.set_index("territory_id").reindex(ns.ids)[[f"clr_rel_{q}" for q in PARTS]].to_numpy()
        worst = max(worst, float(np.abs(CI.window_clr(ns, per.months) - ref).max()))
    if not worst <= CLR_TOL:
        raise QCError(f"dynamics: корзина окна расходится с network_window_nodes на {worst:.3g}")
    return worst


def _types_table(base: CI.Inputs, final: np.ndarray, k: int) -> pd.DataFrame:
    """Типы 24 месяцев: размер и медианы признаков, по которым их узнаёт читатель."""
    rows = []
    B = pd.DataFrame(base.B, columns=list(base.b_names))
    for t in range(k):
        m = final == t
        row = {"type": t + 1, "n": int(m.sum()), "share": float(m.mean())}
        for col in ("clr_rel_cafe", "clr_rel_marketplace", "clr_rel_food"):
            row[col] = float(B.loc[m, col].median())
        for col in ("log_level_rel", "market_access_rel", "log_pop_rel"):
            if col in base.X_raw.columns:
                row[col] = float(base.X_raw.loc[m, col].median())
        rows.append(row)
    return pd.DataFrame(rows)


def run(cfg: Config) -> None:
    t_start = time.perf_counter()
    p = DynParams.from_config(cfg)
    fin = _read_final(cfg)
    cp = _cluster_params(fin, p)
    k, alpha, method = int(fin["k"]), float(fin["hybrid_alpha"]), str(fin["method"])
    seeds = [int(s) for s in fin["seeds"]]
    processed, out = cfg.dir("processed"), cfg.dir("outputs") / "dynamics"
    out.mkdir(parents=True, exist_ok=True)

    main = CI.load_inputs(cfg, cp)
    ns, base = main.ns, main.inputs
    netp = NetworkParams.from_config(cfg)
    final_df = read_table(processed / "cluster_final.parquet", CLUSTER_FINAL)
    final = final_df.set_index("territory_id").reindex(ns.ids)["type"]
    if final.isna().any():
        raise QCError("dynamics: у узлов сети нет итогового типа в cluster_final")
    final = final.to_numpy(dtype=np.int64) - 1

    # --- Сети и входы окон ------------------------------------------------------------------------
    pers = W.periods(ns, netp)
    roll = [q for q in pers if q.kind == "rolling"]
    graphs = {q.name: W.window_graph(ns, netp, cp.graph_rule, cp.graph_k, q.months) for q in pers}
    qc = W.check_rolling({q.name: graphs[q.name] for q in roll}, processed, cp.graph_rule)
    qc["clr_max_diff"] = _check_clr(ns, roll[0], roll[-1], processed)
    qc["level_max_diff"] = main.qc["level_max_diff"]
    if cp.graph_sparsify != "knn":
        raise ValueError("dynamics: сеть окна строится kNN, как network_windows")

    space = W.HybridSpace.build(base, k, alpha, int(fin["best_seed"])) if method == "hybrid" else None
    labels, Z, rows = {}, {}, []
    for q in pers:
        A = CI.graph_matrix(graphs[q.name], ns.ids)
        inp = W.window_inputs(base, ns, A, q.months, q.name)
        pf = M.fit_protocol(method, inp, k, fin["param"], seeds, fin["impl"], RANDOM_METHODS, {}, alpha)
        labels[q.name] = pf.labels
        fit = np.nan
        if space is not None:
            Z[q.name], fit = space.embed(inp)
        first_date, last_date = q.dates(ns)
        rows.append(
            {
                "period": q.name,
                "kind": q.kind,
                "year": q.year,
                "first_date": first_date,
                "last_date": last_date,
                "n_months": len(q.months),
                "n_edges": len(graphs[q.name]),
                "seed_ari": pf.seed_ari,
                "best_seed": pf.seed,
                "procrustes_fit": fit,
                "seconds": pf.seconds,
            }
        )
        log.info("dynamics: %s — K = %d, ARI разных seed %.3f", q.name, k, pf.seed_ari)
    periods_df = pd.DataFrame(rows)
    ari_all = float(adjusted_rand_score(final, labels["24 месяца"]))
    qc["refit_all_ari"] = ari_all
    if ari_all < p.final_ari_min:
        raise QCError(f"dynamics: перефит на 24 месяцах не повторил cluster_final (ARI {ari_all:.3f})")
    if space is None:
        raise ValueError(f"dynamics: альтернативы реализованы для гибрида, итог — {method}")
    C24 = T.centers(space.Z_ref, final, k)
    qc["fixed_ref_agree"] = float(np.mean(T.nearest(space.Z_ref, C24) == final))

    # --- Три способа ------------------------------------------------------------------------------
    halves_names = {
        "odd_2023": "2023, нечётные месяцы",
        "even_2023": "2023, чётные месяцы",
        "odd_2024": "2024, нечётные месяцы",
        "even_2024": "2024, чётные месяцы",
    }
    wnames = [q.name for q in roll]
    Z_roll = [Z[n] for n in wnames]
    Z_halves = {h: Z[n] for h, n in halves_names.items()}
    tracks = {
        MAIN: C.main_track(
            final, [labels[n] for n in wnames], {h: labels[n] for h, n in halves_names.items()}, k
        )
    }
    tracks[FIXED] = C.fixed_track(Z_roll, Z_halves, C24)
    tracks[EVOLUTIONARY], _ = C.evolutionary_track(
        Z_roll, Z_halves, tracks[MAIN].first, k, p.epsilon, p.evo_max_iter
    )
    changes = {
        name: C.change_vs_noise(tr, k, p.bootstrap, np.random.default_rng([p.seed, 71, i]), p.level)
        for i, (name, tr) in enumerate(tracks.items())
    }
    comparison = C.compare_tracks(tracks, changes, k)

    eps_rows = []
    for j, eps in enumerate(p.epsilon_grid):
        tr, _ = C.evolutionary_track(Z_roll, Z_halves, tracks[MAIN].first, k, eps, p.evo_max_iter)
        ch = C.change_vs_noise(tr, k, p.bootstrap, np.random.default_rng([p.seed, 73, j]), p.level)
        cmp = C.compare_tracks(
            {MAIN: tracks[MAIN], EVOLUTIONARY: tr}, {MAIN: changes[MAIN], EVOLUTIONARY: ch}, k
        )
        r = cmp.loc[cmp["approach"] == EVOLUTIONARY].iloc[0].to_dict()
        r["epsilon"] = eps
        eps_rows.append(r)
    eps_df = pd.DataFrame(eps_rows)

    events = pd.concat([C.events_table(tr, k, p.taus, wnames) for tr in tracks.values()], ignore_index=True)
    adjacent = pd.concat([C.adjacent_table(tr, k, wnames) for tr in tracks.values()], ignore_index=True)

    # --- Основной способ: переходы узлов и проверка сюжета -------------------------------------------
    tm, ch = tracks[MAIN], changes[MAIN]
    same = tm.first == tm.last
    dB = CI.window_clr(ns, roll[-1].months) - CI.window_clr(ns, roll[0].months)
    d_level = CI.window_level(ns, roll[-1].months) - CI.window_level(ns, roll[0].months)
    driver = C.driver_test(dB, d_level, PARTS, ch.reliable, same, p.driver_part, p.driver_alpha)
    stable_same = same & (ch.t23 >= 0) & (ch.t23 == ch.t24) & (tm.first == ch.t23)
    driver_alt = C.driver_test(dB, d_level, PARTS, ch.reliable, stable_same, p.driver_part, p.driver_alpha)
    flows = C.flow_profiles(dB, d_level, PARTS, ch.t23, ch.t24, ch.reliable, same)

    status = np.where(ch.reliable, "reliable", np.where(same, "no_change", "within_noise"))
    tab = ns.table.reset_index(drop=True)
    nodes = pd.DataFrame(
        {
            "territory_id": ns.ids,
            "name": display_names(tab).to_numpy(),
            "region": tab["region_name"].astype(str).to_numpy(),
            "is_city_node": tab["is_city_node"].astype(bool).to_numpy(),
            "type_24m": final + 1,
            "type_2023": tm.first + 1,
            "type_2024": tm.last + 1,
            "half_type_2023": np.where(ch.t23 >= 0, ch.t23 + 1, 0),
            "half_type_2024": np.where(ch.t24 >= 0, ch.t24 + 1, 0),
            "status": status,
            "reliable": ch.reliable,
            "fixed_2023": tracks[FIXED].first + 1,
            "fixed_2024": tracks[FIXED].last + 1,
            "evo_2024": tracks[EVOLUTIONARY].last + 1,
            **{f"d_clr_rel_{q}": dB[:, j] for j, q in enumerate(PARTS)},
            "d_log_level_rel": d_level,
        }
    )
    regions = (
        nodes.groupby("region")
        .agg(
            n_nodes=("territory_id", "size"),
            n_reliable=("reliable", "sum"),
            n_changed=("status", lambda s: (s != "no_change").sum()),
        )
        .reset_index()
    )
    regions["share_reliable"] = regions["n_reliable"] / regions["n_nodes"]
    regions = regions.sort_values(["n_reliable", "share_reliable", "region"], ascending=[False, False, True])

    # --- История типов для лендинга ---------------------------------------------------------------
    hist = []
    for i, q in enumerate(roll):
        fd, ld = q.dates(ns)
        hist.append(
            pd.DataFrame(
                {
                    "territory_id": ns.ids.astype("int32"),
                    "window": q.name,
                    "window_index": np.int8(i + 1),
                    "first_date": fd,
                    "last_date": ld,
                    "type": (tm.rolling[i] + 1).astype("int16"),
                    "type_24m": (final + 1).astype("int16"),
                    "status": pd.Categorical(status, categories=STATUSES),
                    "reliable": ch.reliable,
                    "is_city_node": nodes["is_city_node"].to_numpy(),
                }
            )
        )
    history = pd.concat(hist, ignore_index=True)
    write_table(history, DYNAMICS_HISTORY, processed / "dynamics_history.parquet")

    # --- Таблицы и факты --------------------------------------------------------------------------
    write_csv(periods_df, out / "periods.csv")
    write_csv(adjacent, out / "windows.csv")
    write_csv(events, out / "events.csv")
    write_csv(comparison, out / "comparison.csv")
    write_csv(eps_df, out / "evolutionary_epsilon.csv")
    write_csv(driver.table, out / "driver.csv")
    write_csv(driver_alt.table, out / "driver_stable_same.csv")
    write_csv(flows, out / "flows.csv")
    write_csv(nodes, out / "nodes.csv")
    write_csv(regions, out / "regions.csv")
    write_csv(_types_table(base, final, k), out / "types.csv")
    mats = []
    for name, c in changes.items():
        for kind, mat in (
            ("all", c.matrix),
            ("reliable", c.reliable_matrix),
            ("reliable_window", c.window_reliable_matrix),
        ):
            for a in range(k):
                for b in range(k):
                    mats.append(
                        {"approach": name, "kind": kind, "from": a + 1, "to": b + 1, "n": int(mat[a, b])}
                    )
    write_csv(pd.DataFrame(mats), out / "transitions.csv")
    labels_long = pd.concat(
        [
            pd.DataFrame({"approach": name, "period": n, "territory_id": ns.ids, "type": lab + 1})
            for name, tr in tracks.items()
            for n, lab in [
                *zip(wnames, tr.rolling, strict=True),
                *((halves_names[h], tr.halves[h]) for h in C.HALVES),
            ]
        ],
        ignore_index=True,
    )
    write_csv(labels_long, out / "labels.csv")

    facts = {
        "k": k,
        "method": method,
        "alpha": alpha,
        "n_nodes": int(ns.n),
        "n_windows": len(roll),
        "n_seeds": len(seeds),
        "bootstrap": p.bootstrap,
        "level": p.level,
        "tau": p.events_jaccard,
        "taus": list(p.taus),
        "epsilon": p.epsilon,
        "qc": qc,
        "main_info": tm.info,
        "changes": {
            name: {
                **{f: getattr(c.boot, f) for f in c.boot.__dataclass_fields__},
                "n_reliable": int(c.reliable.sum()),
                "noise_2023": c.noise_shares[0],
                "noise_2024": c.noise_shares[1],
                "n_changed": int(c.matrix.sum() - np.trace(c.matrix)),
                "n_reliable_window": int(c.window_reliable_matrix.sum()),
                "n_reliable_window_same": c.n_reliable_window_same,
                "n_reliable_window_other": c.n_reliable_window_other,
                "halves_agree_2023": float(np.mean(c.t23 >= 0)),
                "halves_agree_2024": float(np.mean(c.t24 >= 0)),
            }
            for name, c in changes.items()
        },
        "reliable_same_window": int((ch.reliable & same).sum()),
        "driver": {
            "part": driver.part,
            "p_value": driver.p_value,
            "significant": driver.significant,
            "top_part": driver.top_part,
            "expectation_met": driver.expectation_met,
            "n_transition": driver.n_transition,
            "n_same": driver.n_same,
        },
        "driver_stable_same": {
            "p_value": driver_alt.p_value,
            "top_part": driver_alt.top_part,
            "n_same": driver_alt.n_same,
        },
        "seconds": time.perf_counter() - t_start,
    }
    write_json(facts, out / "facts.json")
    log.info(
        "dynamics: сменили тип 2023 → 2024 %.1f%% узлов, шум %.1f%%, надёжных переходов %d",
        100 * ch.boot.change,
        100 * ch.boot.noise,
        int(ch.reliable.sum()),
    )

    from munnet.dynamics import figures as F
    from munnet.dynamics import report as R

    figs = F.make_figures(cfg, out, k, wnames, tracks, changes, driver, nodes, ns)
    R.write_report(cfg, p, out, figs)
    log.info("dynamics: способы %s, %.0f с", ", ".join(APPROACHES), time.perf_counter() - t_start)
