"""Запуск этапа network: расчёт, выходы ``data/processed/network_*.parquet``, ``outputs/network/`` и отчёт.

Выходы:

- ``network_edges`` — рёбра всех правил при основном разрежении (``is_main``), у выбранного правила
  и альтернатив — ещё при каждом k из ``k_grid``, взаимный kNN и порог;
- ``network_windows`` — рёбра выбранного правила по окнам (кварталы, полугодия, скользящие 12 месяцев);
- ``network_window_nodes`` — корзина узлов относительно группы региона в тех же окнах (атрибуты для этапа 3);
- ``network_nodes`` — степень, k-встречаемость и сообщество зонда Leiden каждого узла по каждому правилу;
- ``outputs/network/`` — таблицы сравнения, ``facts.json`` (все числа отчёта), рисунки, список исключённых
  узлов; отчёт ``docs/network.md`` собирается из шаблона ``templates/network.md``.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from munnet.config import Config
from munnet.contracts import (
    NETWORK_EDGES,
    NETWORK_NODES,
    NETWORK_WINDOW_NODES,
    NETWORK_WINDOWS,
    PARTS,
    QCError,
    coerce,
    write_table,
)
from munnet.eda.base import write_json
from munnet.network import compare as C
from munnet.network import geo
from munnet.network import graph as G
from munnet.network import rules as R
from munnet.network.compute import GeoCache, series_null
from munnet.network.data import NodeSet, load_main, load_mode, window_clr
from munnet.network.params import NetworkParams, RuleSpec

log = logging.getLogger(__name__)

OUTPUT_SUBDIR = "network"
QC_TOL = 1e-9  # корзина окна из трат по месяцам = features_windows (та же формула)
JSON_DIGITS = 10
HIST_BINS = np.linspace(-1.0, 1.0, 81)


def _check_basket(ns: NodeSet, windows: pd.DataFrame) -> float:
    """Сверка: корзина годовых окон из трат по месяцам совпадает с ``features_windows`` (QCError иначе)."""
    worst = 0.0
    cols = [f"clr_rel_{q}" for q in PARTS]
    for year, months in C.YEAR_MONTHS.items():
        ref = windows.loc[windows["window"] == str(year)].set_index("territory_id").reindex(ns.ids)[cols]
        diff = np.abs(window_clr(ns, months) - ref.to_numpy()).max()
        worst = max(worst, float(diff))
    if not worst <= QC_TOL:
        raise QCError(f"корзина окна расходится с features_windows на {worst:.3g} > {QC_TOL}")
    return worst


def _edge_table(
    rule: str, method: str, k: int, edges: pd.DataFrame, res: C.RuleResult, ns: NodeSet, is_main: bool
) -> pd.DataFrame:
    i, j = edges["i"].to_numpy(), edges["j"].to_numpy()
    ids = ns.ids
    src, dst = ids[i], ids[j]
    swap = src > dst
    out = pd.DataFrame(
        {
            "rule": rule,
            "sparsify": method,
            "k": np.int16(k),
            "source": np.where(swap, dst, src).astype("int32"),
            "target": np.where(swap, src, dst).astype("int32"),
            "weight": edges["weight"].to_numpy(dtype=np.float64),
        }
    )
    if res.sim.lag is not None:
        lag = res.sim.lag[i, j].astype("int64")
        out["lag"] = pd.array(np.where(swap, -lag, lag), dtype="Int8")
    else:
        out["lag"] = pd.array([pd.NA] * len(out), dtype="Int8")
    out["q_value"] = res.Q[i, j] if res.Q is not None else np.nan
    out["same_region"] = ns.groups[i] == ns.groups[j]
    info = ns.info
    out["dist_km"] = G.haversine_km(info.lat[i], info.lon[i], info.lat[j], info.lon[j])
    out["is_main"] = bool(is_main)
    return coerce(out, NETWORK_EDGES)


def _node_table(results: dict[str, C.RuleResult], ns: NodeSet) -> pd.DataFrame:
    frames = []
    city = ns.table["is_city_node"].to_numpy(dtype=bool)
    for name, res in results.items():
        frames.append(
            pd.DataFrame(
                {
                    "territory_id": ns.ids.astype("int32"),
                    "rule": name,
                    "degree": G.degree_of(res.edges, ns.n).astype("int32"),
                    "kocc": G.kocc(res.lists, ns.n).astype("int32"),
                    "community": res.probe.membership.astype("int32"),
                    "is_city_node": city,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _signal(ns: NodeSet, p: NetworkParams, rng: np.random.Generator) -> tuple[pd.DataFrame, dict]:
    """Распределения сходства пар «до и после»: корреляция сырых ln трат и своего ритма («Все категории»)
    против циклического сдвига; косинус CLR корзины без поправки на регион и относительно группы региона."""
    iu = np.triu_indices(ns.n, 1)
    months = C.ALL_MONTHS
    corr_raw = R.corr_matrix(ns.raw)[iu]
    own_rule = RuleSpec(name="own_all", kind="rhythm_corr", categories=("all",))
    corr_own = R.corr_matrix(ns.own["all"])[iu]
    null = series_null(own_rule, ns, months, p, rng, None)
    cos_abs = R.cosine_matrix(window_clr(ns, months, relative=False))[iu]
    cos_rel = R.cosine_matrix(window_clr(ns, months))[iu]
    series = {
        "corr_raw": corr_raw,
        "corr_own": corr_own,
        "corr_null": null,
        "cos_abs": cos_abs,
        "cos_rel": cos_rel,
    }
    hist = pd.DataFrame({"bin_left": HIST_BINS[:-1], "bin_right": HIST_BINS[1:]})
    stats = {}
    for name, v in series.items():
        v = v[np.isfinite(v)]
        counts, _ = np.histogram(np.clip(v, -1, 1), bins=HIST_BINS)
        hist[name] = counts / counts.sum()
        stats[name] = {
            "median": float(np.median(v)),
            "q10": float(np.quantile(v, 0.1)),
            "q90": float(np.quantile(v, 0.9)),
            "share_gt_naive": float((v > p.naive_r).mean()),
        }
    return hist, stats


def _rounded(obj):
    if isinstance(obj, dict):
        return {k: _rounded(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_rounded(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        return float(f"{f:.{JSON_DIGITS}g}") if np.isfinite(f) else None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def write_csv(df: pd.DataFrame, path: Path) -> Path:
    """CSV с 6 значащими цифрами: повторный прогон даёт тот же файл (последние знаки сумм BLAS не пишутся)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False, lineterminator="\n", float_format="%.6g")
    tmp.replace(path)
    return path


def _boot_pairs(boot: pd.DataFrame) -> dict:
    """Парный бутстрап надёжности: для каждой пары правил — доля повторов, где первое надёжнее второго, и
    медианная разница."""
    if boot.empty:
        return {}
    wide = boot.pivot(index="replicate", columns="rule", values="reliability")
    out = {}
    for a in wide.columns:
        for b in wide.columns:
            if a < b:
                d = wide[a] - wide[b]
                out[f"{a}__{b}"] = {"share_a_better": float((d > 0).mean()), "median_diff": float(d.median())}
    return out


def _time_bucket(seconds: float) -> str:
    """Время расчёта корзинкой: точные секунды от прогона к прогону разные, отчёт должен повторяться."""
    for limit, text in ((1, "< 1 с"), (10, "1–10 с"), (60, "10–60 с")):
        if seconds < limit:
            return text
    return "> 1 мин"


def run(cfg: Config) -> None:
    """Правила рёбер, сравнение, выбор, динамическая сеть и отчёт."""
    t_start = time.perf_counter()
    p = NetworkParams.from_config(cfg)
    ns, fwin = load_main(cfg, p)
    qc = _check_basket(ns, fwin)
    log.info(
        "network: %d узлов (режим %s), исключено %d; корзина окна = features_windows (%.1g)",
        ns.n,
        ns.mode,
        len(ns.excluded),
        qc,
    )
    raw = Path(cfg["paths"]["raw"])
    conn = pd.read_parquet(
        raw / cfg["panel"]["connection"], columns=["territory_id_x", "territory_id_y", "distance", "type"]
    )
    pairs = geo.road_pairs(conn, "highway")
    rail = geo.road_pairs(conn, "railway")
    road_facts = {
        "highway_rows": int((conn["type"].astype(str) == "highway").sum()),
        "highway_pairs": len(pairs),
        "railway_rows": int((conn["type"].astype(str) == "railway").sum()),
        "railway_pairs": len(rail),
        "zero_distance_pairs": int((pairs["distance"] <= 0).sum()),
        "no_rail_mo": int((~ns.members["territory_id"].isin(set(rail["a"]) | set(rail["b"]))).sum()),
    }
    geo_cache = GeoCache(pairs, p.city_distance)
    D = geo_cache.distance(ns)
    no_road = ns.ids[np.isnan(D).all(axis=1)]
    rng = np.random.default_rng([p.seed, 3])

    road_name = next(r for r, s in p.rules.items() if s.kind == "road")
    results: dict[str, C.RuleResult] = {}
    results[road_name] = C.evaluate_rule(p.rules[road_name], ns, p, geo_cache, None, rng)
    geo_edges = results[road_name].edges
    for name, spec in p.rules.items():
        if name == road_name:
            continue
        log.info("network: правило %s (%s)", name, spec.kind)
        results[name] = C.evaluate_rule(spec, ns, p, geo_cache, geo_edges, rng)
    comparison = pd.DataFrame([r.row for r in results.values()])
    comparison["time_bucket"] = comparison["seconds"].map(_time_bucket)

    log.info("network: чувствительность к k, способы разрежения, сходство правил между собой")
    geo_by_k = {k: G.knn_edges(results[road_name].sim.S, k) for k in p.k_grid}
    grid = pd.DataFrame([row for r in results.values() for row in C.k_grid_rows(r, ns, p, geo_by_k)])
    spars = pd.DataFrame([row for r in results.values() for row in C.sparsify_rows(r, ns, p, geo_cache, rng)])
    pairs_tab = C.pair_rows(results, p, rng)
    boot = C.paired_bootstrap(results, ns, p, rng)
    chosen, orders, sel = C.select(comparison, p)
    by_k = C.select_by_k(grid, p)
    alts = C.alternatives(chosen, comparison, p)
    log.info("network: выбрано правило %s; Парето-фронт %s; альтернативы %s", chosen, sel["front"], alts)

    log.info("network: режимы узлов %s и окна динамики", list(p.modes))
    mode_tab = pd.DataFrame()
    mode_rules = [p.rules[r] for r in [chosen, *alts]]
    for mode in p.modes:
        if mode == ns.mode:
            continue
        other = load_mode(cfg, p, mode)
        mode_tab = pd.concat(
            [mode_tab, pd.DataFrame(C.mode_rows(ns, results, other, mode_rules, p, geo_cache))],
            ignore_index=True,
        )

    win_edges, win_nodes, win_summary, win_noise = C.dynamics(p.rules[chosen], ns, p, geo_cache, rng)
    hist, signal_stats = _signal(ns, p, rng)

    # --- таблицы data/processed ---
    processed = cfg.dir("processed")
    frames = []
    for name, res in results.items():
        frames.append(_edge_table(name, p.method, p.k, res.edges, res, ns, True))
        if name in (chosen, *alts):
            for k in p.k_grid:
                for method in ("knn", "mutual_knn", "threshold"):
                    if k == p.k and method == p.method:
                        continue
                    if method != p.method and k != p.k:
                        continue
                    e = C.sparsify(res.sim.S, method, k, res.Q, p.fdr_q)
                    frames.append(_edge_table(name, method, k, e, res, ns, False))
    write_table(pd.concat(frames, ignore_index=True), NETWORK_EDGES, processed / "network_edges.parquet")
    we = win_edges.copy()
    i, j = we["i"].to_numpy(), we["j"].to_numpy()
    a, b = ns.ids[i], ns.ids[j]
    we = pd.DataFrame(
        {
            "rule": chosen,
            "window_kind": pd.Categorical(win_edges["window_kind"]),
            "window": win_edges["window"].astype(str),
            "n_months": win_edges["n_months"].astype("int8"),
            "source": np.minimum(a, b).astype("int32"),
            "target": np.maximum(a, b).astype("int32"),
            "weight": win_edges["weight"].astype("float64"),
            "is_main_kind": win_edges["window_kind"].astype(str) == p.dyn_main,
        }
    )
    write_table(coerce(we, NETWORK_WINDOWS), NETWORK_WINDOWS, processed / "network_windows.parquet")
    if len(win_nodes):
        wn = win_nodes.astype({"territory_id": "int32", "n_months": "int8"})
        write_table(
            coerce(wn, NETWORK_WINDOW_NODES), NETWORK_WINDOW_NODES, processed / "network_window_nodes.parquet"
        )
    write_table(
        coerce(_node_table(results, ns), NETWORK_NODES), NETWORK_NODES, processed / "network_nodes.parquet"
    )

    # --- outputs/network ---
    out = cfg.dir("outputs") / OUTPUT_SUBDIR
    write_csv(comparison.drop(columns=["seconds"]), out / "comparison.csv")
    write_csv(
        comparison[["rule", "seconds"]], out / "timing.csv"
    )  # время — единственное, что меняется от прогона
    attr_cols = [c for c in comparison.columns if c.startswith("assort_") and c != "assort_region"]
    write_csv(comparison[["rule", *attr_cols]], out / "attributes.csv")
    write_csv(grid, out / "sensitivity_k.csv")
    write_csv(spars, out / "sparsify.csv")
    write_csv(pairs_tab, out / "rule_pairs.csv")
    write_csv(orders, out / "selection_orders.csv")
    write_csv(by_k, out / "selection_by_k.csv")
    write_csv(boot, out / "reliability_bootstrap.csv")
    write_csv(sel["criteria"], out / "selection_criteria.csv")
    write_csv(mode_tab, out / "modes.csv")
    write_csv(win_summary, out / "windows.csv")
    write_csv(win_noise, out / "windows_noise.csv")
    write_csv(hist, out / "signal_hist.csv")
    excluded = ns.excluded.assign(scope="все правила")
    if len(no_road):
        names = ns.table.set_index("territory_id").loc[no_road, "name"].astype(str).to_numpy()
        excluded = pd.concat(
            [
                excluded,
                pd.DataFrame(
                    {
                        "territory_id": no_road.astype("int32"),
                        "name": names,
                        "reason": "нет автодорожных расстояний в connection.parquet",
                        "scope": "правила географии (изолят)",
                    }
                ),
            ],
            ignore_index=True,
        )
    write_csv(excluded, out / "excluded.csv")

    facts = {
        "mode": ns.mode,
        "n_nodes": ns.n,
        "n_excluded": len(ns.excluded),
        "n_no_road": len(no_road),
        "basket_qc_max_diff": qc,
        "road": road_facts,
        "chosen": chosen,
        "alternatives": alts,
        "selection": {k: v for k, v in sel.items() if k != "criteria"},
        "selection_by_k": {str(r["k"]): r["winner"] for _, r in by_k.iterrows()},
        "bootstrap_pairs": _boot_pairs(boot),
        "signal": signal_stats,
        "params": {
            "k": p.k,
            "method": p.method,
            "k_grid": list(p.k_grid),
            "fdr_q": p.fdr_q,
            "shift_repeats": p.shift_repeats,
            "shift_guard": p.shift_guard,
            "bootstrap": p.bootstrap,
            "rewire_repeats": p.rewire_repeats,
            "seeds": len(p.seeds),
            "resolution": p.resolution,
            "mantel_permutations": p.mantel_permutations,
            "naive_r": p.naive_r,
            "rules": {
                n: {
                    "kind": s.kind,
                    "categories": list(s.categories),
                    "max_lag": s.max_lag,
                    "window": s.window,
                    "beta": s.beta,
                }
                for n, s in p.rules.items()
            },
        },
        "sigma": {n: r.sim.sigma for n, r in results.items() if r.sim.sigma is not None},
        "dyn_main": p.dyn_main,
        "city_members": {
            str(int(t)): int(m)
            for t, m in ns.table.loc[ns.table["is_city_node"], ["territory_id", "n_members"]].to_numpy()
        },
    }
    write_json(_rounded(facts), out / "facts.json")

    from munnet.network import figures, report

    figs = figures.make_all(out, comparison, pairs_tab, win_noise, hist, signal_stats, chosen, p)
    if p.report:
        report.write_report(cfg, p, out, figs)
    else:
        log.info("network: network.report пуст — отчёт не собирается")
    log.info(
        "network: выходы — %s/network_*.parquet, %s; %.0f с", processed, out, time.perf_counter() - t_start
    )
