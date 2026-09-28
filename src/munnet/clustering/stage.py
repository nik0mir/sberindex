"""Запуск этапа cluster: расчёт, выбор, проверки, выходы и отчёт.

Выходы:

- ``data/processed/cluster_labels.parquet`` — метки каждого кандидата «метод × K» (основные входы), флаги
  допустимости, победителя метода и итога;
- ``data/processed/cluster_final.parquet`` — итоговый тип узла (1…K) и Жаккар устойчивости его типа;
- ``outputs/cluster/final.json`` — выбранный метод, K, параметр и фиксированные настройки (для этапа
dynamics);
- ``outputs/cluster/*.csv`` — таблица кандидатов со всеми критериями, промежуточные результаты выбора (фронт,
  очки Копленда), проверки чувствительности, варианты входов, устойчивость кластеров, согласие методов,
  профили типов, внешняя проверка, синтетика, кривая гибрида по α, итоговая таблица методов;
- ``outputs/cluster/facts.json`` и ``report_facts.json`` — числа отчёта ``docs/clustering.md``.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from munnet.clustering import analysis as AN
from munnet.clustering import compute as CO
from munnet.clustering import decide as DE
from munnet.clustering import methods as M
from munnet.clustering import parallel
from munnet.clustering import stability as ST
from munnet.clustering import synthetic as SY
from munnet.clustering import validation as VA
from munnet.clustering.inputs import impute_region_median, load_inputs, node_context, variant_inputs
from munnet.clustering.params import RANDOM_METHODS, ClusterParams
from munnet.config import Config
from munnet.contracts import CLUSTER_FINAL, CLUSTER_LABELS, coerce, write_table
from munnet.eda.base import write_json
from munnet.network.graph import ari_on_common

log = logging.getLogger(__name__)

OUTPUT_SUBDIR = "cluster"
JSON_DIGITS = 10
GRAPH_METHODS_FOR_PERTURB = ("leiden", "louvain", "spectral", "shalileh_mirkin", "hybrid")


def write_csv(df: pd.DataFrame, path: Path) -> Path:
    """CSV с 6 значащими цифрами: повторный прогон даёт тот же файл."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False, lineterminator="\n", float_format="%.6g")
    tmp.replace(path)
    return path


def rounded(obj):
    if isinstance(obj, dict):
        return {str(k): rounded(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [rounded(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, float | np.floating):
        f = float(obj)
        return float(f"{f:.{JSON_DIGITS}g}") if np.isfinite(f) else None
    return obj


def by_id(ids: np.ndarray, labels: np.ndarray) -> dict[int, int]:
    return dict(zip(np.asarray(ids).tolist(), np.asarray(labels).tolist(), strict=True))


def connectivity(A: sp.csr_matrix) -> dict:
    from scipy.sparse.csgraph import connected_components

    n_comp, lab = connected_components(A, directed=False)
    sizes = np.bincount(lab)
    deg = A.getnnz(axis=1)
    return {
        "n_components": int(n_comp),
        "giant_share": float(sizes.max() / len(lab)),
        "n_edges": int(A.nnz // 2),
        "mean_degree": float(deg.mean()),
        "weighted": bool(not np.allclose(A.data, 1.0)),
        "weight_min": float(A.data.min()),
        "weight_median": float(np.median(A.data)),
    }


# --- Задачи для процессов ------------------------------------------------------------------------


def validation_task(task) -> pd.DataFrame:
    cand, labels, beyond = task
    st = parallel.STATE
    out = VA.validate(
        labels,
        st["indicators"],
        st["axes"],
        st["X"],
        st["wage"],
        st["groups"],
        st["cfg"],
        st["seed"],
        beyond_permutations=beyond,
    )
    out.insert(0, "cand", cand)
    return out


def alpha_task(task) -> dict:
    from munnet import icvi

    alpha, k = task
    st = parallel.STATE
    inp, cp = st["inputs"], st["params"]
    pf = M.fit_protocol("hybrid", inp, k, None, cp.seeds, cp.impl, RANDOM_METHODS, {}, alpha)
    specs = cp.icvi_metrics
    vals = icvi.evaluate(pf.labels, X=inp.X, A=inp.A, names=list(specs))
    rng = np.random.default_rng([cp.seed, 37, int(round(alpha * 1000))])
    base = icvi.random_baseline(
        pf.labels, X=inp.X, A=inp.A, names=list(specs), n_perm=cp.icvi_permutations, rng=rng
    )
    base = base.set_index("name")
    row = {"alpha": alpha, "k": k, "labels": pf.labels}
    for n, s in specs.items():
        row[f"z_{n}"] = icvi.zscore(
            float(vals[n]), float(base.loc[n, "mean"]), float(base.loc[n, "sd"]), s["better"]
        )
    return row


# --- Части расчёта -------------------------------------------------------------------------------


def variant_rows(cfg, cp, main_data, res, final) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    rows, all_cands, info = [], [], {}
    main_map = by_id(res.inputs.ids, res.labels[final])
    for v in cp.variants:
        t = time.perf_counter()
        vin = variant_inputs(cfg, cp, main_data, v)
        vres = CO.evaluate_set(vin, cp, cp.bootstrap_variants, v.name)
        tl = DE.run_rule(vres.cands, cp)
        w = tl.winner
        same = final if final in vres.labels else None
        rows.append(
            {
                "variant": v.name,
                "n_nodes": vin.n,
                "n_edges": int(vin.A.nnz // 2),
                "n_features": len(vin.x_names),
                "winner": w,
                "method": None if w is None else vres.cands.loc[w, "method"],
                "k": None if w is None else int(vres.cands.loc[w, "k"]),
                "same_winner": w == final,
                "ari_winner_vs_main": np.nan
                if w is None
                else ari_on_common(main_map, by_id(vin.ids, vres.labels[w])),
                "ari_same_candidate_vs_main": np.nan
                if same is None
                else ari_on_common(main_map, by_id(vin.ids, vres.labels[same])),
                "n_common": len(set(main_map) & set(vin.ids.tolist())),
                "seconds": time.perf_counter() - t,
            }
        )
        all_cands.append(vres.cands.assign(variant=v.name).reset_index())
        info[v.name] = {"fallback_share": vres.fallback_share, "graph": connectivity(vin.A)}
        log.info("cluster: вариант %s — победитель %s", v.name, w)
    return pd.DataFrame(rows), pd.concat(all_cands, ignore_index=True) if all_cands else pd.DataFrame(), info


def avu_matrix(A: sp.csr_matrix, labels: np.ndarray) -> pd.DataFrame:
    """Объединяемость пар типов U_kl = M_kl / (out_k + out_l − M_kl) (формула AVU модуля ``icvi``): M = HᵀAH,
    out_k — вес рёбер типа наружу. Большая U_kl — типы k и l отдают друг другу почти всё внешнее."""
    ok = labels != M.NOISE
    k = int(labels[ok].max()) + 1
    H = sp.csr_matrix((np.ones(ok.sum()), (np.flatnonzero(ok), labels[ok])), shape=(len(labels), k))
    Mk = (H.T @ A @ H).toarray()
    out = Mk.sum(axis=1) - np.diag(Mk)
    den = out[:, None] + out[None, :] - Mk
    with np.errstate(divide="ignore", invalid="ignore"):
        U = np.where(den > 0, Mk / den, 0.0)
    np.fill_diagonal(U, np.nan)
    frame = pd.DataFrame(U, index=range(1, k + 1), columns=[f"type_{j}" for j in range(1, k + 1)])
    frame["isolability"] = np.diag(Mk) / Mk.sum(axis=1)
    return frame.rename_axis("type").reset_index()


def kefrin_graph_share(inp) -> float:
    """Доля сети в общем разбросе данных KEFRiN (ρ = ξ = 1): Σ(p − p̄)² / (Σ(p − p̄)² + Σ(y − ȳ)²) при
    стандартизации (M) — насколько граф вообще может влиять на критерий."""
    from munnet.clustering import kefrin as KF

    links = KF.ModularityLinks(inp.A)
    mean_row = np.asarray(links.A.mean(axis=0)).ravel()
    lam = (mean_row[None, :], np.array([links.d.mean()]))
    sp_graph = float((links.norms2() - 2 * links.dots(lam)[:, 0] + links.center_norms2(lam)[0]).sum())
    sp_feat = float(((inp.X - inp.X.mean(axis=0)) ** 2).sum())
    return sp_graph / (sp_graph + sp_feat)


def small_clusters(res, cp: ClusterParams, filled: pd.DataFrame) -> pd.DataFrame:
    """Типы меньше порога допустимости (``feasible.min_cluster_share``) у всех кандидатов: размер, признак X
    с наибольшим по модулю отклонением медианы (в единицах масштаба) и первые узлы с названием и регионом."""
    inp = res.inputs
    n = inp.n
    Xs = pd.DataFrame(inp.X, columns=list(inp.x_names))
    names = inp.table["name"].astype(str) + " (" + inp.table["region_name"].astype(str) + ")"
    rows = []
    for c, lab in res.labels.items():
        for cl in sorted(set(lab.tolist()) - {M.NOISE}):
            m = lab == cl
            if m.sum() / n >= cp.min_cluster_share:
                continue
            med = Xs.loc[m].median()
            top = med.abs().idxmax()
            order = np.argsort(-np.abs(Xs.loc[m, top].to_numpy()))
            rows.append(
                {
                    "cand": c,
                    "method": res.cands.loc[c, "method"],
                    "k": int(res.cands.loc[c, "k"]),
                    "cluster": int(cl),
                    "size": int(m.sum()),
                    "top_feature": top,
                    "top_median_scaled": float(med[top]),
                    "top_median_raw": float(filled.loc[m, top].median()),
                    "examples": "; ".join(names[m].to_numpy()[order][:5]),
                }
            )
    return pd.DataFrame(rows)


def agreement(labels: dict[str, np.ndarray], cands: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

    ari = pd.DataFrame(index=cands, columns=cands, dtype="float64")
    ami = ari.copy()
    for a in cands:
        for b in cands:
            ari.loc[a, b] = adjusted_rand_score(labels[a], labels[b])
            ami.loc[a, b] = adjusted_mutual_info_score(labels[a], labels[b])
    return ari.rename_axis("cand").reset_index(), ami.rename_axis("cand").reset_index()


def profiles(inp, labels: np.ndarray, filled: pd.DataFrame) -> pd.DataFrame:
    """Профиль типа: размер и медианы признаков X (как есть и в единицах масштаба) и корзины."""
    rows = []
    raw = pd.concat([filled.reset_index(drop=True), pd.DataFrame(inp.B, columns=list(inp.b_names))], axis=1)
    scaled = pd.DataFrame(np.hstack([inp.X, inp.B]), columns=[*inp.x_names, *inp.b_names])
    # корзина в «масштабе»: деление на MAD×1,4826 по узлам, как X
    for c in inp.b_names:
        v = scaled[c]
        mad = float(np.median(np.abs(v - v.median())) * 1.4826) or 1.0
        scaled[c] = (v - v.median()) / mad
    for t in sorted(set(labels.tolist())):
        m = labels == t
        for col in raw.columns:
            rows.append(
                {
                    "type": int(t),
                    "size": int(m.sum()),
                    "share": float(m.mean()),
                    "feature": col,
                    "block": "basket" if col in inp.b_names else "X",
                    "median": float(raw.loc[m, col].median()),
                    "median_scaled": float(scaled.loc[m, col].median()),
                }
            )
    return pd.DataFrame(rows)


def alpha_curve(pool, cp: ClusterParams, res, winners: dict, final: str) -> pd.DataFrame:
    """Гибрид при весах графа α из ``protocol.alpha_curve`` на K победителя гибрида: z-оценки ICVI и ARI с
    итогом,
    с K-means (только X) и спектральной (только G) того же K — «когда помогает граф» на данных (в выбор не
    входит)."""
    from sklearn.metrics import adjusted_rand_score

    cands = res.cands
    k_h = int(cands.loc[winners["hybrid"], "k"]) if "hybrid" in winners else int(cands.loc[final, "k"])
    rows = pool.map(alpha_task, [(a, k_h) for a in cp.alpha_curve])
    ref = {}
    for name in ("kmeans", "spectral"):
        c = cand_of(cands, name, k_h)
        if c:
            ref[name] = res.labels[c]
    out = []
    for r in rows:
        lab = r.pop("labels")
        r["ari_final"] = adjusted_rand_score(res.labels[final], lab)
        for name, lab_ref in ref.items():
            r[f"ari_{name}"] = adjusted_rand_score(lab_ref, lab)
        out.append(r)
    return pd.DataFrame(out)


def cand_of(cands: pd.DataFrame, method: str, k: int) -> str | None:
    c = CO.cand_id(method, k)
    return c if c in cands.index else None


def edge_perturbation(pool, cp: ClusterParams, res, winners: dict) -> pd.DataFrame:
    """Удаление доли рёбер G у победителей методов, видящих граф: ARI с разбиением на всей сети."""
    ep = cp.impl.get("edge_perturbation") or {}
    cands = res.cands
    tasks = []
    for m in GRAPH_METHODS_FOR_PERTURB:
        if m in winners:
            c = winners[m]
            param = None if np.isnan(cands.loc[c, "param"]) else float(cands.loc[c, "param"])
            for rep in range(int(ep.get("repeats", 20))):
                tasks.append((c, m, int(cands.loc[c, "k_target"]), param, rep, res.labels[c]))
    return pd.DataFrame(pool.map(ST.perturb_task, tasks))


def extra_selection(cands: pd.DataFrame, res, cp: ClusterParams) -> pd.DataFrame:
    """Проверка вне предрегистрации (на тех же кандидатах): порог наименьшего типа 1% вместо 2%."""
    rows = []
    share = float((cp.impl.get("extra_checks") or {}).get("min_cluster_share", 0.01))
    feas = pd.DataFrame({c: CO.feasibility(res.labels[c], cp, share) for c in cands.index}).T
    alt = cands.copy()
    alt["feasible"] = feas["feasible"].astype(bool)
    tl = DE.run_rule(alt, cp)
    every = DE.run_rule(alt, cp, methods=cp.methods)
    for check, w in (("min_share_01", tl.winner), ("min_share_01_all", every.winner)):
        rows.append(
            {
                "check": check,
                "winner": w,
                "method": None if w is None else cands.loc[w, "method"],
                "k": None if w is None else int(cands.loc[w, "k"]),
                "n_feasible": int(alt["feasible"].sum()),
            }
        )
    return pd.DataFrame(rows)


def clipped_check(cp: ClusterParams, res, final: str):
    """Вне предрегистрации: X с усечёнными хвостами (каждый признак — в пределах своих квантилей
    ``extra_checks.clip_quantiles``), остальное как в вариантах. Показывает, сколько решают хвосты X."""
    from dataclasses import replace

    q = (cp.impl.get("extra_checks") or {}).get("clip_quantiles")
    if not q:
        return None, pd.DataFrame()
    t = time.perf_counter()
    inp = res.inputs
    lo, hi = np.quantile(inp.X, float(q[0]), axis=0), np.quantile(inp.X, float(q[1]), axis=0)
    vin = replace(inp, name="x_clipped", X=np.clip(inp.X, lo, hi))
    vres = CO.evaluate_set(vin, cp, cp.bootstrap_variants, "x_clipped")
    w = DE.run_rule(vres.cands, cp).winner
    main_map = by_id(inp.ids, res.labels[final])
    row = {
        "variant": "x_clipped",
        "n_nodes": vin.n,
        "n_edges": int(vin.A.nnz // 2),
        "n_features": len(vin.x_names),
        "winner": w,
        "method": None if w is None else vres.cands.loc[w, "method"],
        "k": None if w is None else int(vres.cands.loc[w, "k"]),
        "same_winner": w == final,
        "ari_winner_vs_main": np.nan
        if w is None
        else ari_on_common(main_map, by_id(vin.ids, vres.labels[w])),
        "ari_same_candidate_vs_main": ari_on_common(main_map, by_id(vin.ids, vres.labels[final])),
        "n_common": vin.n,
        "seconds": time.perf_counter() - t,
    }
    return row, vres.cands.assign(variant="x_clipped").reset_index()


def external_validation(cfg: Config, cp: ClusterParams, res, filled: pd.DataFrame, full_p: set) -> tuple:
    """Проверки ``clustering.validation`` для итога и всех допустимых кандидатов."""
    inp = res.inputs
    val_cfg = cp.validation
    ctx = node_context(cfg, inp.ids)
    indicators = VA.indicator_frame(ctx, inp.ids, inp.groups, val_cfg["indicators"])
    axes = pd.concat([filled.reset_index(drop=True), pd.DataFrame(inp.B, columns=list(inp.b_names))], axis=1)
    state = {
        "indicators": indicators.reset_index(drop=True),
        "axes": axes,
        "X": inp.X,
        "wage": filled["log_wage_rel"].to_numpy(dtype=np.float64),
        "groups": inp.groups,
        "cfg": val_cfg,
        "seed": cp.seed,
    }
    cands = res.cands
    todo = [c for c in cands.index if bool(cands.loc[c, "feasible"]) or c in full_p]
    B = int(val_cfg["permutations"])
    tasks = [(c, res.labels[c], B if c in full_p else 0) for c in todo]
    parts = parallel.run(validation_task, tasks, state, cp.workers)
    return pd.concat(parts, ignore_index=True), indicators


def run(cfg: Config) -> None:
    """Кандидаты, выбор, проверки, синтетика, внешняя проверка, выходы и отчёт."""
    t_start = time.perf_counter()
    timing: dict[str, float] = {}
    cp = ClusterParams.from_config(cfg)
    out = cfg.dir("outputs") / OUTPUT_SUBDIR
    processed = cfg.dir("processed")
    main_data = load_inputs(cfg, cp)
    inp = main_data.inputs
    ginfo = connectivity(inp.A)
    log.info(
        "cluster: G — %d рёбер, компонент связности %d, веса %s",
        ginfo["n_edges"],
        ginfo["n_components"],
        ginfo["weighted"],
    )

    # 1–2. Кандидаты на основных входах, выбор и чувствительность; на тех же процессах — кривая гибрида
    # по α и возмущение рёбер
    t = time.perf_counter()
    with parallel.Pool(CO.state_of(inp, cp), cp.workers) as pool:
        res = CO.evaluate_set(inp, cp, cp.bootstrap, "main", pool)
        cands = res.cands
        dec = DE.decide(cands, cp)
        final = dec.main.winner
        if final is None:
            raise RuntimeError("cluster: ни у одного допустимого метода нет кандидата")
        winners = {m: ch.winner for m, ch in dec.all_methods.level1.items()}
        timing.update({f"main_{k}": v for k, v in res.seconds.items()})
        timing["main_total"] = time.perf_counter() - t
        t = time.perf_counter()
        alpha_tab = alpha_curve(pool, cp, res, winners, final)
        timing["alpha_curve"] = time.perf_counter() - t
        t = time.perf_counter()
        pert = edge_perturbation(pool, cp, res, winners)
        timing["edge_perturbation"] = time.perf_counter() - t
        t = time.perf_counter()
        kef = pd.DataFrame(pool.map(AN.kefrin_task, AN.kefrin_tasks(cp, res.labels[final])))
        if len(kef):
            shares = {
                (f, x): AN.kefrin_graph_share(inp, x, f)
                for f, x in kef[["features", "xi_over_rho"]].drop_duplicates().itertuples(index=False)
            }
            kef["graph_share"] = [
                shares[(f, x)] for f, x in zip(kef["features"], kef["xi_over_rho"], strict=True)
            ]
        timing["kefrin_curve"] = time.perf_counter() - t
    fm, fk = str(cands.loc[final, "method"]), int(cands.loc[final, "k"])
    log.info("cluster: итог — %s; проверки: %s", final, dec.checks[["check", "winner"]].values.tolist())
    levels = DE.level_table(dec.main, cands)
    levels_all = DE.level_table(dec.all_methods, cands)
    ag_ari, ag_ami = agreement(res.labels, [winners[m] for m in cp.methods if m in winners])
    pert_sum = (
        pert.groupby("cand")["ari"].agg(["mean", "std", "min"]).reset_index() if len(pert) else pd.DataFrame()
    )
    extra = extra_selection(cands, res, cp)
    grid = AN.threshold_grid(res, cp)
    reslim = AN.resolution_limit(res) if cp.impl.get("resolution_limit", True) else pd.DataFrame()

    # 3. Варианты входов (предрегистрация) и проверка с усечёнными хвостами X (вне предрегистрации)
    t = time.perf_counter()
    variants, variant_cands, variant_info = variant_rows(cfg, cp, main_data, res, final)
    clip_row, clip_cands = clipped_check(cp, res, final)
    if clip_row is not None:
        variants = pd.concat([variants, pd.DataFrame([clip_row])], ignore_index=True)
        variant_cands = pd.concat([variant_cands, clip_cands], ignore_index=True)
    timing["variants"] = time.perf_counter() - t

    # 4. Внешняя проверка (после выбора): итог и все допустимые кандидаты; перестановочный p прироста R²
    # сверх атрибутов — у итога и победителей методов (у остальных — только сам прирост)
    t = time.perf_counter()
    filled, _ = impute_region_median(inp.X_raw, inp.groups)
    validation, indicators = external_validation(cfg, cp, res, filled, set(winners.values()) | {final})
    timing["validation"] = time.perf_counter() - t

    # 5. Синтетика
    t = time.perf_counter()
    sy = cp.synthetic
    fam = {m: cp.family_of(m) for m in cp.methods}
    syn_state = {"synthetic": sy, "impl": cp.impl, "alpha": cp.hybrid_alpha, "seed": cp.seed}
    syn_raw = pd.DataFrame(parallel.run(SY.cell_task, SY.tasks(sy), syn_state, cp.workers))
    syn_sum = SY.summarize(syn_raw, list(sy["methods"]), fam)
    timing["synthetic"] = time.perf_counter() - t

    # 8. Ориентир K (iK-means) и профили итога
    ik_min = int(cp.impl.get("ik_means_min_size", 2))
    ik_x, ik_x_sizes = M.ik_means_k(inp.X, ik_min)
    ik_j, ik_j_sizes = M.ik_means_k(M.block_join([inp.B, inp.X], [0.5, 0.5]), ik_min)
    final_labels = res.labels[final]
    prof = profiles(inp, final_labels, filled)
    rules_text = ST.tree_rules(inp.tree_matrix, final_labels, inp.tree_names, cp.tree_depth, cp.seed)

    # --- выходы data/processed ---
    lab_rows = []
    for c, lab in res.labels.items():
        lab_rows.append(
            pd.DataFrame(
                {
                    "candidate": c,
                    "territory_id": inp.ids.astype("int32"),
                    "method": cands.loc[c, "method"],
                    "k": np.int16(cands.loc[c, "k"]),
                    "param": float(cands.loc[c, "param"]),
                    "label": lab.astype("int16"),
                    "feasible": bool(cands.loc[c, "feasible"]),
                    "is_method_winner": c in winners.values(),
                    "is_final": c == final,
                }
            )
        )
    labels_tab = coerce(pd.concat(lab_rows, ignore_index=True), CLUSTER_LABELS)
    write_table(labels_tab, CLUSTER_LABELS, processed / "cluster_labels.parquet")
    jf = res.jaccard.loc[res.jaccard["cand"] == final].set_index("cluster")["jaccard"]
    final_tab = pd.DataFrame(
        {
            "territory_id": inp.ids.astype("int32"),
            "type": (final_labels + 1).astype("int16"),
            "k": np.int16(fk),
            "method": fm,
            "candidate": final,
            "type_jaccard": jf.reindex(final_labels).to_numpy(dtype=np.float64),
            "is_city_node": inp.table["is_city_node"].astype(bool).to_numpy(),
        }
    )
    write_table(coerce(final_tab, CLUSTER_FINAL), CLUSTER_FINAL, processed / "cluster_final.parquet")

    # --- outputs/cluster ---
    cands_out = cands.reset_index()
    write_csv(cands_out.drop(columns=["fit_seconds"]), out / "candidates.csv")
    write_csv(levels, out / "selection_levels.csv")
    write_csv(levels_all, out / "selection_levels_all.csv")
    write_csv(dec.checks, out / "sensitivity.csv")
    write_csv(dec.orders, out / "sensitivity_orders.csv")
    write_csv(extra, out / "extra_checks.csv")
    write_csv(grid, out / "threshold_grid.csv")
    write_csv(reslim, out / "resolution_limit.csv")
    write_csv(kef, out / "kefrin_curve.csv")
    write_csv(dec.icvi, out / "sensitivity_icvi.csv")
    write_csv(dec.edges, out / "k_edges.csv")
    write_csv(avu_matrix(inp.A, final_labels), out / "avu_matrix.csv")
    write_csv(small_clusters(res, cp, filled), out / "small_clusters.csv")
    write_csv(variants.drop(columns=["seconds"]), out / "variants.csv")
    write_csv(variant_cands.drop(columns=["fit_seconds"], errors="ignore"), out / "variant_candidates.csv")
    write_csv(res.scans, out / "param_scans.csv")
    write_csv(res.jaccard, out / "stability_clusters.csv")
    write_csv(res.boot, out / "stability_bootstrap.csv")
    write_csv(ag_ari, out / "agreement_ari.csv")
    write_csv(ag_ami, out / "agreement_ami.csv")
    write_csv(alpha_tab, out / "alpha_curve.csv")
    write_csv(pert, out / "edge_perturbation.csv")
    write_csv(prof, out / "profiles.csv")
    write_csv(validation, out / "validation.csv")
    write_csv(syn_raw, out / "synthetic_raw.csv")
    write_csv(syn_sum, out / "synthetic_summary.csv")
    (out / "tree_rules.txt").write_text(rules_text, encoding="utf-8", newline="\n")
    timing["total_before_report"] = time.perf_counter() - t_start
    timing_tab = pd.concat(
        [
            pd.DataFrame({"step": list(timing), "seconds": list(timing.values())}),
            cands_out[["cand", "fit_seconds"]].rename(columns={"cand": "step", "fit_seconds": "seconds"}),
            variants[["variant", "seconds"]]
            .rename(columns={"variant": "step"})
            .assign(step=lambda d: "variant_" + d["step"]),
        ],
        ignore_index=True,
    )
    write_csv(timing_tab, out / "timing.csv")

    final_param = None if np.isnan(cands.loc[final, "param"]) else float(cands.loc[final, "param"])
    final_json = {
        "candidate": final,
        "method": fm,
        "k": fk,
        "param": final_param,
        "param_name": {"leiden": "resolution", "louvain": "resolution", "hdbscan": "min_cluster_size"}.get(
            fm
        ),
        "best_seed": None if np.isnan(cands.loc[final, "best_seed"]) else int(cands.loc[final, "best_seed"]),
        "seeds": list(cp.seeds),
        "hybrid_alpha": cp.hybrid_alpha,
        "inputs": {
            "graph": {"rule": cp.graph_rule, "sparsify": cp.graph_sparsify, "k": cp.graph_k},
            "features": list(cp.attributes),
            "place_year": cp.place_year,
            "scale": cp.scale,
            "impute": cp.impute,
        },
        "impl": cp.impl,
        "how_to_refit": "munnet.clustering.methods.fit_protocol(method, inputs, k, param, seeds, impl, "
        "RANDOM_METHODS, alpha=hybrid_alpha); входы окна — munnet.clustering.inputs",
    }
    write_json(rounded(final_json), out / "final.json")

    facts = {
        "n_nodes": inp.n,
        "graph": ginfo,
        "qc": main_data.qc,
        "imputed": inp.imputed,
        "scale_used": inp.scale_used,
        "final": final,
        "final_method": fm,
        "final_k": fk,
        "winners": winners,
        "level2_front": dec.main.level2.front,
        "level2_scores": dec.main.level2.scores.to_dict(),
        "level2_tied": dec.main.level2.tied,
        "all_eligible_front": dec.all_methods.level2.front,
        "checks": dec.checks.to_dict(orient="records"),
        "extra_checks": extra.to_dict(orient="records"),
        "icvi_checks_changed": sorted(
            set(dec.icvi.loc[dec.icvi["changed"].astype(bool) & (dec.icvi["level"] == 2), "check"])
        )
        if len(dec.icvi)
        else [],
        "orders_count": dec.orders.groupby("winner").size().to_dict() if len(dec.orders) else {},
        "variants": variants.drop(columns=["seconds"]).to_dict(orient="records"),
        "variant_info": variant_info,
        "fallback_share": res.fallback_share,
        "n_candidates": len(cands),
        "n_feasible": int(cands["feasible"].sum()),
        "kefrin_graph_share": kefrin_graph_share(inp),
        "x_abs_max": {n: float(v) for n, v in zip(inp.x_names, np.abs(inp.X).max(axis=0), strict=True)},
        "ik_means": {"X": ik_x, "X_sizes": ik_x_sizes[:30], "joint": ik_j, "joint_sizes": ik_j_sizes[:30]},
        "edge_perturbation": pert_sum.to_dict(orient="records") if len(pert_sum) else [],
        "indicator_coverage": VA.coverage(indicators),
        "params": {
            "bootstrap": cp.bootstrap,
            "bootstrap_variants": cp.bootstrap_variants,
            "subsample": cp.subsample,
            "seeds": len(cp.seeds),
            "icvi_permutations": cp.icvi_permutations,
            "k_grid": list(cp.k_grid),
            "workers": cp.workers,
        },
    }
    write_json(rounded(facts), out / "facts.json")
    log.info("cluster: расчёт — %.0f с", time.perf_counter() - t_start)

    from munnet.clustering import report

    report.build(cfg, cp, out)
    log.info("cluster: выходы — %s, %s; всего %.0f с", processed, out, time.perf_counter() - t_start)
