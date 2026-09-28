"""Отчёт ``docs/clustering.md``: шаблон ``templates/clustering.md`` + таблицы и числа из ``outputs/cluster``.

Числа текста — только факты ``{{cl.ключ}}`` (``outputs/cluster/report_facts.json``), таблицы — директивы
``<!-- table: имя -->``, рисунки — ``<!-- figure: C01 -->``. Утверждения текста, зависящие от результата
(какой
метод выбран, какие проверки меняют победителя, что показала синтетика), перечислены в ``CLAIMS`` и
проверяются
по числам: если хоть одно перестало быть верным, отчёт не пишется (``QCError``, код 3) — шаблон надо
переписать
под новые числа. Типографика — функции отчёта разведки.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from munnet import style
from munnet.clustering import figures as FG
from munnet.clustering.params import CRITERIA, USES, ClusterParams
from munnet.config import Config
from munnet.contracts import FEATURES_MEMBERS, QCError, read_table
from munnet.eda import report as eda_report
from munnet.eda.base import Fact, make_fact, write_json

log = logging.getLogger(__name__)

TEMPLATE = Path(__file__).resolve().parent / "templates" / "clustering.md"
SECTION = "cl"
_TABLE = re.compile(r"<!--\s*table:\s*([a-z_0-9]+)\s*-->")
_FIGURE = re.compile(r"<!--\s*figure:\s*(C\d{2})\s*-->")
_NOTE = re.compile(r"[ \t]*<!--\s*note\s*:.*?-->[ \t]*\n?", re.S)
LINT_ALLOW = [
    r"CC BY(?:-SA)? \d\.\d",
    r"\b\d{1,2}\.\d{2}\.\d{4}\b",
    r"\$\$.*?\$\$",
    r"\$[^$]+\$",
    r"\bK-means\+\+",
    r"разд\. \d\.\d",  # ссылка на раздел статьи
    r"\b\d{2}\.\d{2}(?= по| после)",  # дата «28.09 по…»
]
CHECK_LABELS: dict[str, str] = {
    "main": "основное правило",
    "borda": "Борда на фронте вместо Копленда",
    "copeland_all": "Копленд по всем допустимым, без фронта",
    "set_icvi_only": "только качество в X и на G",
    "set_plan": "из PLAN.md: качество и устойчивость",
    "tie_x0": "допуски ничьей × 0",
    "tie_x2": "допуски ничьей × 2",
    "joint": "один шаг по всем парам «метод, K»",
    "all_eligible": "все семейства допускаются к итогу",
}
ICVI_LABELS: dict[str, str] = {
    "no_avu": "(а) качество на G без AVU",
    "avi_mq_one": "(б) AVI и MQ — одна метрика",
    "no_s_dbw": "(в) качество в X без S_Dbw",
    "s_dbw_own": "(г) S_Dbw в варианте own",
    "nan_worst": "(д) неопределённая метрика — худший ранг",
    "noise_cluster": "(е) шум HDBSCAN — отдельный кластер",
}
VARIANT_LABELS: dict[str, str] = {
    "graph_basket_cos": "G — сеть «Косинус корзин»",
    "no_level": "X без уровня трат",
    "nodes_separate": "районы Москвы и Петербурга — отдельные узлы",
    "x_clipped": "X с усечёнными хвостами (вне предрегистрации)",
}
INDICATOR_LABELS: dict[str, str] = {
    "ip_per_1000": "ИП на 1000 жителей",
    "orgs_per_1000": "Организации на 1000 жителей",
    "nights_pc": "Ночёвки в КСР на жителя",
}
CHECK_KIND_LABELS: dict[str, str] = {
    "difference": "различие (ε²)",
    "sign": "знак (ρ)",
    "beyond_attributes": "сверх атрибутов (ΔR²)",
}


def _fact(facts: dict, key: str, value, kind: str) -> None:
    facts[f"{SECTION}.{key}"] = make_fact(f"{SECTION}.{key}", value, kind)


def _v(f: Mapping[str, Fact], key: str):
    return f[f"{SECTION}.{key}"].value


def _names(cands: list[str], table: pd.DataFrame) -> str:
    return ", ".join(f"{FG.mlabel(table.loc[c, 'method'])} (K = {int(table.loc[c, 'k'])})" for c in cands)


def _bucket(seconds: float) -> str:
    for limit, text in ((1, "< 1 с"), (10, "1–10 с"), (60, "10–60 с")):
        if seconds < limit:
            return text
    return "> 1 мин"


# --- Входные таблицы -----------------------------------------------------------------------------


class Out:
    """Таблицы ``outputs/cluster`` одним объектом."""

    def __init__(self, out: Path) -> None:
        self.dir = out
        self.js = json.loads((out / "facts.json").read_text(encoding="utf-8"))
        self.final = json.loads((out / "final.json").read_text(encoding="utf-8"))
        read = lambda n: pd.read_csv(out / f"{n}.csv")  # noqa: E731
        self.cands = read("candidates").set_index("cand", drop=False)
        self.levels = read("selection_levels")
        self.levels_all = read("selection_levels_all")
        self.checks = read("sensitivity")
        self.orders = read("sensitivity_orders")
        self.icvi = read("sensitivity_icvi")
        self.edges = read("k_edges")
        self.extra = read("extra_checks")
        self.variants = read("variants")
        self.variant_cands = read("variant_candidates")
        self.scans = read("param_scans")
        self.jac = read("stability_clusters")
        self.ari = read("agreement_ari")
        self.alpha = read("alpha_curve")
        self.pert = read("edge_perturbation")
        self.prof = read("profiles")
        self.val = read("validation")
        self.syn = read("synthetic_summary")
        self.syn_raw = read("synthetic_raw")
        self.avu = read("avu_matrix")
        self.small = read("small_clusters") if (out / "small_clusters.csv").exists() else pd.DataFrame()
        self.timing = read("timing")
        self.rules = (out / "tree_rules.txt").read_text(encoding="utf-8")

    @property
    def winners(self) -> dict[str, str]:
        return dict(self.js["winners"])


def level1_table_all(o: Out) -> pd.DataFrame:
    """Победители уровня K всех методов с рангами качества среди всех методов (уровень 2 без ограничения)."""
    t = o.levels_all.loc[o.levels_all["level"] == 2].copy()
    t = t.rename(columns={f"crit_{c}": c for c in CRITERIA})
    return t[["cand", "method", "k", *CRITERIA, "on_front", "score"]]


# --- Факты ---------------------------------------------------------------------------------------


def build_facts(o: Out, cp: ClusterParams) -> dict[str, Fact]:
    f: dict[str, Fact] = {}
    js, c = o.js, o.cands
    fin = o.final["candidate"]
    _fact(f, "n_nodes", js["n_nodes"], "int")
    g = js["graph"]
    _fact(f, "n_edges", g["n_edges"], "int")
    _fact(f, "mean_degree", g["mean_degree"], "num1")
    _fact(f, "n_components", g["n_components"], "int")
    _fact(f, "weight_min", g["weight_min"], "num2")
    _fact(f, "weight_median", g["weight_median"], "num2")
    for col, n in js["imputed"].items():
        _fact(f, f"imputed_{col}", n, "int")
    _fact(f, "n_x", len(cp.attributes), "int")
    _fact(f, "n_tree", len(cp.attributes) + len(cp.edges), "int")
    _fact(f, "n_candidates", js["n_candidates"], "int")
    _fact(f, "n_feasible", js["n_feasible"], "int")
    _fact(f, "n_methods", len(cp.methods), "int")
    _fact(f, "k_min", min(cp.k_grid), "int")
    _fact(f, "k_max", max(cp.k_grid), "int")
    _fact(f, "n_grid", len(cp.k_grid), "int")
    _fact(f, "bootstrap", cp.bootstrap, "int")
    _fact(f, "bootstrap_variants", cp.bootstrap_variants, "int")
    _fact(f, "subsample", cp.subsample, "pct")
    _fact(f, "seeds", len(cp.seeds), "int")
    _fact(f, "icvi_perms", cp.icvi_permutations, "int")
    _fact(f, "val_perms", int(cp.validation["permutations"]), "int")
    _fact(f, "min_share", cp.min_cluster_share, "pct")
    _fact(f, "min_share_nodes", int(np.ceil(cp.min_cluster_share * js["n_nodes"])), "int")
    _fact(f, "max_share", cp.max_cluster_share, "pct")
    _fact(f, "max_noise", cp.max_noise_share, "pct")
    _fact(f, "alpha", cp.hybrid_alpha, "num2")
    _fact(f, "n_init", int(cp.impl.get("kmeans_n_init", 20)), "int")
    _fact(f, "workers", js["params"]["workers"], "int")
    for crit in ("stability", "interpretability"):
        _fact(f, f"tie_{crit}", cp.tie[crit], "num2")
    _fact(f, "jaccard_stable", cp.jaccard_stable, "num2")
    _fact(f, "jaccard_dissolved", cp.jaccard_dissolved, "num2")
    _fact(f, "tree_depth", cp.tree_depth, "int")
    _fact(f, "cv_folds", cp.cv_folds, "int")
    # итог
    _fact(f, "final_label", FG.mlabel(o.final["method"]), "str")
    _fact(f, "final_method", o.final["method"], "str")
    _fact(f, "final_k", o.final["k"], "int")
    r = c.loc[fin]
    for col, kind in (
        ("stability", "num2"),
        ("jaccard_min", "num2"),
        ("interpretability", "num2"),
        ("seed_ari", "num2"),
        ("z_sw", "num1"),
        ("z_ch", "int"),
        ("z_avi", "int"),
        ("z_mq", "int"),
        ("z_avu", "int"),
        ("z_s_dbw", "num1"),
    ):
        _fact(f, f"final_{col}", float(r[col]), kind)
    sizes = o.jac.loc[o.jac["cand"] == fin].sort_values("cluster")
    for _, s in sizes.iterrows():
        t = int(s["cluster"]) + 1
        _fact(f, f"type{t}_size", int(s["size"]), "int")
        _fact(f, f"type{t}_share", float(s["size"]) / js["n_nodes"], "pct")
        _fact(f, f"type{t}_jaccard", float(s["jaccard"]), "num2")
    # допустимость по методам
    for m in cp.methods:
        sub = c.loc[c["method"] == m]
        _fact(f, f"feas_{m}", int(sub["feasible"].sum()), "int")
        _fact(f, f"cands_{m}", len(sub), "int")
    small = o.small
    if len(small):
        _fact(f, "n_small", len(small), "int")
        _fact(f, "n_small_market", int((small["top_feature"] == "market_access_rel").sum()), "int")
        for m in ("shalileh_mirkin", "kmeans_joint", "kmeans", "hybrid"):
            sm = small.loc[small["method"] == m]
            if len(sm):
                _fact(f, f"small_{m}_min", int(sm["size"].min()), "int")
                _fact(f, f"small_{m}_max", int(sm["size"].max()), "int")
        top = small.loc[small["cand"] == "kmeans_joint_k04"]
        if len(top):
            _fact(f, "suburbs_size", int(top["size"].iloc[0]), "int")
            _fact(f, "suburbs_raw", float(top["top_median_raw"].iloc[0]), "int")
            _fact(f, "suburbs_scaled", float(top["top_median_scaled"].iloc[0]), "int")
            _fact(f, "suburbs_examples", str(top["examples"].iloc[0]).replace("; ", ", "), "str")
    # HDBSCAN и Louvain: какие K достижимы
    sc = o.scans
    h = sc.loc[sc["method"] == "hdbscan"]
    if len(h):
        _fact(f, "hdbscan_kmax", int(h["k"].max()), "int")
        _fact(f, "hdbscan_mcs_min", int(h["param"].min()), "int")
        _fact(f, "hdbscan_mcs_max", int(h["param"].max()), "int")
        zero = h.loc[h["k"] == 0, "param"]
        _fact(f, "hdbscan_zero_from", int(zero.min()) if len(zero) else 0, "int")
    for m in ("leiden", "louvain"):
        got = set(c.loc[c["method"] == m, "k"].astype(int))
        missing = [k for k in cp.k_grid if k not in got]
        _fact(f, f"{m}_missing", ", ".join(str(k) for k in missing) if missing else "нет", "str")
        _fact(f, f"{m}_n_reached", len(got), "int")
    # выбор
    winners = o.winners
    for m, w in winners.items():
        _fact(f, f"win_{m}_k", int(c.loc[w, "k"]), "int")
    l2 = level1_table_all(o).set_index("cand")
    front = [x for x in l2.index if bool(l2.loc[x, "on_front"])]
    _fact(f, "all_front", _names(front, c), "str")
    _fact(f, "all_front_n", len(front), "int")
    top_score = l2["score"].max()
    tied = [x for x in l2.index if l2.loc[x, "score"] == top_score]
    _fact(f, "all_tied", _names(tied, c), "str")
    _fact(f, "all_tied_n", len(tied), "int")
    _fact(f, "all_top_score", int(top_score), "int")
    main_l2 = o.levels.loc[o.levels["level"] == 2]
    _fact(f, "main_l2_n", len(main_l2), "int")
    ch = o.checks
    _fact(f, "n_checks", len(ch) - 1, "int")
    _fact(f, "n_checks_same", int(ch.loc[ch["check"] != "main", "same_as_main"].astype(bool).sum()), "int")
    _fact(f, "orders_same", int((o.orders["winner"] == fin).sum()), "int")
    _fact(f, "orders_total", len(o.orders), "int")
    ic = o.icvi
    l2c = ic.loc[ic["level"] == 2]
    _fact(f, "icvi_n", l2c["check"].nunique(), "int")
    _fact(f, "icvi_l2_changed", int(l2c["changed"].astype(bool).sum()), "int")
    _fact(f, "icvi_l3_changed", int(ic.loc[ic["level"] == 3, "changed"].astype(bool).sum()), "int")
    l1c = ic.loc[(ic["level"] == 1) & ic["changed"].astype(bool)]
    _fact(f, "icvi_l1_changed", len(l1c), "int")
    _fact(
        f,
        "icvi_l1_changes",
        "; ".join(
            f"{ICVI_LABELS[r2['check']]} — {FG.mlabel(r2['method'])}: "
            f"K = {int(c.loc[r2['main_winner'], 'k'])} → {int(r2['k'])}"
            for _, r2 in l1c.iterrows()
        )
        or "нет",
        "str",
    )
    ex = o.extra.set_index("check")
    if "min_share_01" in ex.index:
        _fact(f, "extra01_winner", FG.mlabel(ex.loc["min_share_01", "method"]), "str")
        _fact(f, "extra01_k", int(ex.loc["min_share_01", "k"]), "int")
        _fact(f, "extra01_n", int(ex.loc["min_share_01", "n_feasible"]), "int")
    # варианты
    v = o.variants.set_index("variant")
    for name, row in v.iterrows():
        _fact(f, f"var_{name}_winner", FG.mlabel(row["method"]), "str")
        _fact(f, f"var_{name}_k", int(row["k"]), "int")
        _fact(f, f"var_{name}_ari", float(row["ari_winner_vs_main"]), "num2")
        _fact(f, f"var_{name}_ari_same", float(row["ari_same_candidate_vs_main"]), "num2")
        _fact(f, f"var_{name}_nodes", int(row["n_nodes"]), "int")
    vc = o.variant_cands
    xc = vc.loc[vc["variant"] == "x_clipped"]
    if len(xc):
        for m in cp.eligible_methods:
            _fact(f, f"clip_feas_{m}", int(xc.loc[xc["method"] == m, "feasible"].astype(bool).sum()), "int")
    # кривая гибрида
    a = o.alpha.set_index("alpha")
    half = cp.hybrid_alpha
    for col in ("ari_final", "ari_kmeans", "ari_spectral"):
        if col in a.columns:
            _fact(f, f"alpha_{col}", float(a.loc[half, col]), "num2")
    for al in (0.0, 1.0):
        if al in a.index:
            _fact(f, f"alpha{int(al)}_z_sw", float(a.loc[al, "z_sw"]), "num1")
            _fact(f, f"alpha{int(al)}_z_avi", float(a.loc[al, "z_avi"]), "int")
    _fact(f, "alpha_half_z_sw", float(a.loc[half, "z_sw"]), "num1")
    _fact(f, "alpha_half_z_avi", float(a.loc[half, "z_avi"]), "int")
    # согласие
    A = o.ari.set_index("cand")
    wf = winners
    for m1, m2 in (("hybrid", "spectral"), ("hybrid", "leiden"), ("hybrid", "gmm"), ("spectral", "leiden")):
        if m1 in wf and m2 in wf:
            _fact(f, f"ari_{m1}_{m2}", float(A.loc[wf[m1], wf[m2]]), "num2")
    # возмущение рёбер
    pe = o.pert.groupby("cand")["ari"].mean() if len(o.pert) else pd.Series(dtype=float)
    for m, w in wf.items():
        if w in pe.index:
            _fact(f, f"pert_{m}", float(pe[w]), "num2")
    # внешняя проверка итога
    vf = o.val.loc[o.val["cand"] == fin]
    for _, r2 in vf.iterrows():
        key = f"val_{r2['indicator']}_{r2['check']}" + (f"_{r2['axis']}" if r2["check"] == "sign" else "")
        kind = "num3" if r2["check"] != "sign" else "num2"
        _fact(f, key, float(r2["stat"]), kind)
        _fact(f, f"{key}_null", float(r2["null_mean"]), kind)
        _fact(f, f"{key}_q", float(r2["q"]), "p")
    _fact(f, "val_n_tests", len(vf), "int")
    _fact(f, "val_n_sig", int(vf["significant"].astype(bool).sum()), "int")
    _fact(f, "val_wage_rho_max", float(vf["wage_rho"].abs().max()), "num2")
    _fact(f, "val_nights_n", int(js["indicator_coverage"].get("nights_pc", 0)), "int")
    ba = vf.loc[vf["check"] == "beyond_attributes"].set_index("indicator")
    for ind in ba.index:
        _fact(f, f"val_{ind}_base_r2", float(ba.loc[ind, "base_r2"]), "num2")
    feas_c = o.val["cand"].nunique()
    _fact(f, "val_n_cands", feas_c, "int")
    diff = o.val.loc[o.val["check"] == "difference"]
    _fact(f, "val_diff_all_sig", int(diff.groupby("cand")["significant"].all().sum()), "int")
    # синтетика
    s = o.syn
    knn = s.loc[s["design"] == "knn"]
    fam = {m: cp.family_of(m) for m in cp.methods}
    both = knn.loc[(knn["graph_signal"] > 0) & (knn["feature_signal"] > 0)]
    _fact(f, "syn_n_cells", len(knn), "int")
    _fact(f, "syn_both_cells", len(both), "int")
    won_both = both["winner"].map(lambda m: fam.get(m) in ("attributed", "fusion"))
    _fact(f, "syn_both_won", int(won_both.sum()), "int")
    _fact(f, "syn_joint_won", int((both["winner"] == "kmeans_joint").sum()), "int")
    _fact(f, "syn_repeats", int(cp.synthetic["repeats"]), "int")
    _fact(f, "syn_n", int(cp.synthetic["n_nodes"]), "int")
    _fact(f, "syn_k", int(cp.synthetic["k_true"]), "int")
    _fact(f, "syn_degree", float(o.syn_raw.loc[o.syn_raw["design"] == "knn", "mean_degree"].mean()), "num1")
    strong = knn.loc[(knn["graph_signal"] >= 1) & (knn["feature_signal"] >= 1)]
    for m in ("shalileh_mirkin", "hybrid", "kmeans_joint", "kmeans", "spectral", "leiden"):
        _fact(f, f"syn_strong_{m}", float(strong[m].mean()), "num2")
    no_g = knn.loc[knn["graph_signal"] == 0]
    _fact(f, "syn_nograph_best_feat", float(no_g["best_features"].max()), "num2")
    _fact(f, "syn_nograph_best_both", float(no_g["best_both"].max()), "num2")
    _fact(f, "syn_gain_max", float(knn["graph_gain"].max()), "num2")
    no_f = knn.loc[(knn["feature_signal"] == 0) & (knn["graph_signal"] > 0)]
    _fact(f, "syn_nofeat_feat_max", float(no_f["best_features"].max()), "num2")
    sbm = s.loc[s["design"] == "sbm"]
    if len(sbm):
        mix = sorted(sbm["graph_signal"].unique())
        _fact(f, "sbm_mix_min", mix[0], "num1")
        _fact(f, "sbm_mix_max", mix[-1], "num1")
        low = sbm.loc[sbm["graph_signal"] == mix[0]]
        _fact(f, "sbm_low_leiden", float(low["leiden"].mean()), "num2")
        mid = sbm.loc[np.isclose(sbm["graph_signal"], 0.5)]
        if len(mid):
            _fact(f, "sbm_mid_hybrid", float(mid["hybrid"].mean()), "num2")
            _fact(f, "sbm_mid_leiden", float(mid["leiden"].mean()), "num2")
            _fact(f, "sbm_mid_kefrin", float(mid["shalileh_mirkin"].mean()), "num2")
            _fact(f, "sbm_mid_kmeans", float(mid["kmeans"].mean()), "num2")
        hi = sbm.loc[sbm["graph_signal"] >= 0.7]
        _fact(f, "sbm_high_graph_max", float(hi[["leiden", "spectral"]].max().max()), "num2")
        _fact(f, "sbm_degree", float(cp.synthetic["sbm_degree"]), "int")
    # свойства индексов
    gm = c.loc[c["method"].isin(["leiden", "louvain", "spectral"])]
    tau = kendalltau(gm["z_avi"], gm["z_mq"], nan_policy="omit")[0]
    _fact(f, "tau_avi_mq", float(tau), "num2")
    taus = [kendalltau(d["k"], d["z_avi"])[0] for _, d in gm.groupby("method")]
    _fact(f, "tau_k_avi_min", float(min(taus)), "num2")
    lei = c.loc[c["method"] == "leiden"].set_index("k")["mq"]
    _fact(f, "leiden_mq_k9", float(lei.loc[lei.index >= 9].min()), "num3")
    _fact(f, "leiden_mq_k12", float(lei.loc[lei.index >= 9].max()), "num3")
    k4 = c.loc[c["k"] >= 4, "z_avu"].dropna()
    _fact(f, "avu_neg_share", float((k4 < 0).mean()), "pct")
    _fact(f, "avu_z_min", float(k4.min()), "int")
    _fact(f, "avu_undefined_k3", int(c.loc[c["k"] == 3, "z_avu"].isna().sum()), "int")
    _fact(f, "n_k3", int((c["k"] == 3).sum()), "int")
    _fact(f, "sdbw_undefined", int(c["z_s_dbw"].isna().sum()), "int")
    _fact(f, "sdbw_undefined_kmin", int(c.loc[c["z_s_dbw"].isna(), "k"].min()), "int")
    _fact(f, "sdbw_base_undef_median", float(c["s_dbw_base_undefined"].median()), "int")
    kb = []
    for _, d in c.groupby("method"):
        for name in cp.icvi_metrics:
            z = d.set_index("k")[f"z_{name}"].dropna()
            if len(z) >= 2:
                kb.append({"space": cp.icvi_metrics[name]["space"], "k": int(z.idxmax())})
    kb = pd.DataFrame(kb)
    _fact(f, "kbest_features", float(kb.loc[kb["space"] == "features", "k"].median()), "int")
    _fact(f, "kbest_graph", float(kb.loc[kb["space"] == "graph", "k"].median()), "int")
    e = o.edges.set_index("method")
    on_edge = [m for m in e.index if bool(e.loc[m, "k_on_edge"])]
    _fact(f, "edge_methods", ", ".join(FG.mlabel(m) for m in on_edge) or "нет", "str")
    _fact(f, "edge_n", len(on_edge), "int")
    for m in ("leiden", "louvain"):
        if m in e.index and pd.notna(e.loc[m].get("k_raw_mq")):
            _fact(f, f"{m}_k_raw_mq", int(e.loc[m, "k_raw_mq"]), "int")
    _fact(f, "leiden_stability", float(c.loc[winners["leiden"], "stability"]), "num2")
    lw, hw = c.loc[winners["leiden"]], c.loc[fin]
    _fact(f, "interp_diff", float(lw["interpretability"] - hw["interpretability"]), "num2")
    _fact(f, "kefrin_graph_share", float(js["kefrin_graph_share"]), "pct")
    _fact(f, "market_max_mad", float(js["x_abs_max"]["market_access_rel"]), "int")
    for m in cp.methods:
        sub = c.loc[c["method"] == m, "stability"].dropna()
        if len(sub):
            _fact(f, f"stab_median_{m}", float(sub.median()), "num2")
    zk = c.loc[c["method"] == "shalileh_mirkin", "z_avi"]
    zh = c.loc[c["method"] == "hybrid", "z_avi"]
    _fact(f, "kefrin_zavi_max", float(zk.max()), "int")
    _fact(f, "hybrid_zavi_min", float(zh.min()), "int")
    _fact(f, "hybrid_k3_max_share", float(c.loc["hybrid_k03", "max_share"]), "pct")
    signs = o.val.loc[(o.val["cand"] == fin) & (o.val["check"] == "sign")]
    _fact(f, "val_signs_ok", int(signs["sign_ok"].astype(bool).sum()), "int")
    _fact(f, "val_signs_n", len(signs), "int")
    # K-ориентир и прочее
    _fact(f, "ik_x", js["ik_means"]["X"], "int")
    _fact(f, "ik_joint", js["ik_means"]["joint"], "int")
    U = o.avu.set_index("type")
    Ucols = [x for x in U.columns if x.startswith("type_")]
    Um = U[Ucols].to_numpy(dtype=float)
    i, j = np.unravel_index(np.nanargmax(Um), Um.shape)
    _fact(f, "avu_pair", f"{U.index[i]} и {j + 1}", "str")
    _fact(f, "avu_pair_value", float(Um[i, j]), "num2")
    _fact(f, "iso_min", float(U["isolability"].min()), "num2")
    top_split = re.search(r"\|--- (\w+) ", o.rules)
    _fact(
        f,
        "tree_root",
        FG.FEATURE_LABELS.get(top_split.group(1), top_split.group(1)) if top_split else "—",
        "str",
    )
    pairs = o.dir.parent / "network" / "rule_pairs.csv"
    if pairs.exists():
        rp = pd.read_csv(pairs)
        hit = rp.loc[(rp["rule_a"] == "basket_cos") & (rp["rule_b"] == "basket_dist"), "ari"]
        if len(hit):
            _fact(f, "net_ari_cos_dist", float(hit.iloc[0]), "num2")
    tt = o.timing.set_index("step")["seconds"]
    _fact(f, "time_total_min", float(tt.get("total_before_report", np.nan)) / 60, "int")
    _fact(f, "time_bootstrap_min", float(tt.get("main_bootstrap", np.nan)) / 60, "int")
    _fact(f, "time_variants_min", float(tt.get("variants", np.nan)) / 60, "int")
    return f


# --- Проверяемые утверждения ---------------------------------------------------------------------

CLAIMS: dict[str, Callable[[Mapping[str, Fact]], bool]] = {
    "итог — гибрид с K = 4": lambda f: _v(f, "final_method") == "hybrid" and _v(f, "final_k") == 4,
    "у KEFRiN и базовой линии нет допустимых K; у гибрида — один": lambda f: (
        _v(f, "feas_shalileh_mirkin") == 0 and _v(f, "feas_kmeans_joint") == 0 and _v(f, "feas_hybrid") == 1
    ),
    "мелкие недопустимые типы почти все задаёт доступность рынков": lambda f: (
        _v(f, "n_small_market") >= 0.9 * _v(f, "n_small")
    ),
    "все предрегистрированные проверки сохраняют победителя": lambda f: (
        _v(f, "n_checks_same") == _v(f, "n_checks") and _v(f, "orders_same") == _v(f, "orders_total")
    ),
    "проверки по свойствам индексов итог не меняют": lambda f: (
        _v(f, "icvi_l2_changed") == 0 and _v(f, "icvi_l3_changed") == 0
    ),
    "при всех семействах на фронте ничья гибрида и Leiden, решает устойчивость": lambda f: (
        _v(f, "all_tied_n") == 2 and "Leiden" in _v(f, "all_tied") and "Гибрид" in _v(f, "all_tied")
    ),
    "итог почти совпадает со спектральной по одному графу": lambda f: _v(f, "ari_hybrid_spectral") > 0.9,
    "варианты: без уровня трат — тот же итог; косинус и районы — другой": lambda f: (
        _v(f, "var_no_level_ari") > 0.95
        and _v(f, "var_graph_basket_cos_ari") < 0.5
        and _v(f, "var_nodes_separate_ari") < 0.5
    ),
    "при усечённых хвостах побеждает KEFRiN и разбиение другое": lambda f: (
        _v(f, "var_x_clipped_winner") == FG.mlabel("shalileh_mirkin") and _v(f, "var_x_clipped_ari") < 0.5
    ),
    "все типы итога устойчивы по Хеннигу": lambda f: _v(f, "final_jaccard_min") >= 0.75,
    "внешняя проверка: все проверки итога значимы": lambda f: _v(f, "val_n_sig") == _v(f, "val_n_tests"),
    "показатели не связаны с зарплатой": lambda f: _v(f, "val_wage_rho_max") <= 0.5,
    "в синтетике при сигнале в обоих источниках почти всегда выигрывает метод, видящий оба": lambda f: (
        _v(f, "syn_both_won") >= 0.9 * _v(f, "syn_both_cells")
    ),
    "в синтетике базовая линия сильнее KEFRiN и гибрида при сильных сигналах": lambda f: (
        _v(f, "syn_strong_kmeans_joint") > _v(f, "syn_strong_hybrid") > _v(f, "syn_strong_shalileh_mirkin")
    ),
    "HDBSCAN не дал ни одного K из сетки": lambda f: _v(f, "cands_hdbscan") == 0,
    "признаковые индексы тянут к меньшему K, чем графовые": lambda f: (
        _v(f, "kbest_features") < _v(f, "kbest_graph")
    ),
    "z AVU отрицательна почти у всех разбиений с K ≥ 4": lambda f: _v(f, "avu_neg_share") > 0.9,
    "гибрид устойчивее Leiden к удалению рёбер": lambda f: _v(f, "pert_hybrid") > _v(f, "pert_leiden"),
    "Leiden обходит гибрид по объяснимости сверх допуска": lambda f: (
        _v(f, "interp_diff") > _v(f, "tie_interpretability")
    ),
    "гибрид устойчивее Leiden": lambda f: _v(f, "final_stability") > _v(f, "leiden_stability"),
    "у гибрида при K = 3 крупнейший тип больше половины": lambda f: _v(f, "hybrid_k3_max_share") > 0.5,
    "AVU не определена у всех кандидатов с K = 3": lambda f: _v(f, "avu_undefined_k3") == _v(f, "n_k3"),
    "все ожидаемые знаки подтвердились": lambda f: _v(f, "val_signs_ok") == _v(f, "val_signs_n"),
    "порог 1% не добавляет допустимых кандидатов": lambda f: _v(f, "extra01_n") == _v(f, "n_feasible"),
    "у KEFRiN z AVI ниже, чем у гибрида при любом K": lambda f: (
        _v(f, "kefrin_zavi_max") < _v(f, "hybrid_zavi_min")
    ),
    "типы не повторяют регионы (AMI < 0,1)": lambda f: _v(f, "ami_region") < 0.1,
    "сеть — меньшая часть разброса KEFRiN": lambda f: _v(f, "kefrin_graph_share") < 0.5,
    "шесть проверок по свойствам индексов": lambda f: _v(f, "icvi_n") == 6,
}


def check_claims(facts: Mapping[str, Fact]) -> None:
    broken = []
    for text, fn in CLAIMS.items():
        try:
            ok = bool(fn(facts))
        except (KeyError, TypeError, ValueError):
            ok = False
        if not ok:
            broken.append(text)
    if broken:
        raise QCError("отчёт cluster: перестали быть верными утверждения шаблона: " + "; ".join(broken))


# --- Таблицы -------------------------------------------------------------------------------------


def _n(v, d: int = 2) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return style.NA_TEXT
    return style.fmt_num(float(v), d)


def _md(df: pd.DataFrame, right: int = 1) -> str:
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "|" + "|".join("---" if i < right else "---:" for i in range(len(cols))) + "|",
    ]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(x) for x in r.tolist()) + " |")
    return "\n".join(lines)


def table_comparison(o: Out, cp: ClusterParams) -> str:
    """Метод × критерий: победитель уровня K каждого метода (для HDBSCAN — почему кандидатов нет)."""
    c = o.cands
    pe = o.pert.groupby("cand")["ari"].mean() if len(o.pert) else pd.Series(dtype=float)
    l2 = level1_table_all(o).set_index("cand")
    tt = o.timing.set_index("step")["seconds"]
    rows = []
    for m in cp.methods:
        sub = c.loc[c["method"] == m]
        w = o.winners.get(m)
        if w is None:
            rows.append(
                [FG.mlabel(m), USES[m], f"{int(sub['feasible'].sum())} из {len(sub)}"] + [style.NA_TEXT] * 12
            )
            continue
        r = c.loc[w]
        rows.append(
            [
                FG.mlabel(m),
                USES[m],
                f"{int(sub['feasible'].sum())} из {len(sub)}",
                str(int(r["k"])),
                _n(l2.loc[w, "quality_features"]),
                _n(l2.loc[w, "quality_graph"]),
                _n(r["stability"]),
                _n(r["jaccard_min"]),
                _n(r["interpretability"]),
                _n(r["z_sw"], 1),
                _n(r["z_avi"], 0),
                _n(r["z_avu"], 0),
                _n(r["seed_ari"]),
                _n(pe.get(w, np.nan)),
                _bucket(float(tt.get(w, np.nan))),
            ]
        )
    cols = [
        "Метод",
        "Видит",
        "Допустимых K",
        "Выбран K",
        "Качество в X (ранг)",
        "Качество на G (ранг)",
        "Устойчивость (ARI)",
        "Жаккар, худший тип",
        "Объяснимость (κ)",
        "z SW",
        "z AVI",
        "z AVU",
        "ARI разных seed",
        "ARI при −10% рёбер",
        "Время итога",
    ]
    return _md(pd.DataFrame(rows, columns=cols), right=2)


def table_candidates_eligible(o: Out, cp: ClusterParams) -> str:
    """Все K методов, видящих оба источника: почему недопустимы."""
    c = o.cands
    rows = []
    for m in cp.eligible_methods:
        for _, r in c.loc[c["method"] == m].sort_values("k").iterrows():
            rows.append(
                [
                    FG.mlabel(m),
                    str(int(r["k"])),
                    style.fmt_pct(r["min_share"], 1),
                    style.fmt_pct(r["max_share"], 1),
                    _n(r["stability"]),
                    "да" if bool(r["feasible"]) else "нет",
                ]
            )
    cols = ["Метод", "K", "Наименьший тип", "Крупнейший тип", "Устойчивость", "Допустим"]
    return _md(pd.DataFrame(rows, columns=cols), right=1)


def table_level2_all(o: Out) -> str:
    t = level1_table_all(o)
    rows = []
    for _, r in t.iterrows():
        rows.append(
            [
                f"{FG.mlabel(r['method'])}, K = {int(r['k'])}",
                *[_n(r[c]) for c in CRITERIA],
                "да" if bool(r["on_front"]) else "нет",
                style.NA_TEXT if pd.isna(r["score"]) else style.fmt_num(float(r["score"]), 0, sign=True),
            ]
        )
    cols = ["Победитель метода", *[FG.CRITERION_LABELS[c] for c in CRITERIA], "На фронте", "Очки Копленда"]
    return _md(pd.DataFrame(rows, columns=cols))


def table_sensitivity(o: Out) -> str:
    rows = []
    for _, r in o.checks.iterrows():
        rows.append(
            [
                CHECK_LABELS.get(r["check"], r["check"]),
                f"{FG.mlabel(r['method'])}, K = {int(r['k'])}",
                "да" if bool(r["same_as_main"]) else "**нет**",
            ]
        )
    fin = o.final["candidate"]
    n_same = int((o.orders["winner"] == fin).sum())
    rows.append(
        [
            "24 лексикографических порядка критериев",
            f"{FG.mlabel(o.final['method'])}, K = {o.final['k']} — в {n_same} из {len(o.orders)}",
            "да" if n_same == len(o.orders) else "**нет**",
        ]
    )
    return _md(pd.DataFrame(rows, columns=["Проверка", "Победитель", "Тот же, что в основном правиле"]))


def table_icvi_checks(o: Out, cp: ClusterParams) -> str:
    ic = o.icvi
    c = o.cands
    rows = []
    for name, lab in ICVI_LABELS.items():
        d = ic.loc[ic["check"] == name]
        if d.empty:
            continue
        l1 = d.loc[(d["level"] == 1) & d["changed"].astype(bool)]
        ch1 = (
            "; ".join(
                f"{FG.mlabel(r['method'])}: {int(c.loc[r['main_winner'], 'k'])} → {int(r['k'])}"
                for _, r in l1.iterrows()
            )
            or "нет"
        )
        l2 = d.loc[d["level"] == 2].iloc[0]
        l3 = d.loc[d["level"] == 3].iloc[0]
        rows.append(
            [
                lab,
                ch1,
                f"{FG.mlabel(l2['method'])}, K = {int(l2['k'])}",
                "нет" if not bool(l2["changed"]) else "**да**",
                f"{FG.mlabel(l3['method'])}, K = {int(l3['k'])}",
            ]
        )
    cols = ["Проверка", "Смена K внутри метода", "Итог", "Итог меняется", "Итог при всех семействах"]
    return _md(pd.DataFrame(rows, columns=cols))


def table_variants(o: Out) -> str:
    rows = []
    for _, r in o.variants.iterrows():
        rows.append(
            [
                VARIANT_LABELS.get(r["variant"], r["variant"]),
                style.fmt_num(int(r["n_nodes"])),
                str(int(r["n_features"])),
                f"{FG.mlabel(r['method'])}, K = {int(r['k'])}",
                _n(r["ari_winner_vs_main"]),
                _n(r["ari_same_candidate_vs_main"]),
            ]
        )
    cols = [
        "Входы",
        "Узлов",
        "Признаков X",
        "Победитель",
        "ARI победителя с итогом",
        f"ARI того же метода и K ({FG.mlabel('hybrid')}, K = 4) с итогом",
    ]
    return _md(pd.DataFrame(rows, columns=cols), right=1)


def table_types(o: Out) -> str:
    fin = o.final["candidate"]
    jac = o.jac.loc[o.jac["cand"] == fin].sort_values("cluster")
    U = o.avu.set_index("type")
    n = o.js["n_nodes"]
    rows = []
    for _, r in jac.iterrows():
        t = int(r["cluster"]) + 1
        u = U.loc[t, [x for x in U.columns if x.startswith("type_")]]
        partner = u.astype(float).idxmax()
        rows.append(
            [
                f"Тип {t}",
                style.fmt_num(int(r["size"])),
                style.fmt_pct(r["size"] / n, 1),
                _n(r["jaccard"]),
                _n(U.loc[t, "isolability"]),
                f"с типом {partner.split('_')[1]}: {_n(u.max())}",
            ]
        )
    cols = [
        "Тип",
        "Узлов",
        "Доля",
        "Жаккар лучшего совпадения",
        "Изолируемость на G",
        "Наибольшая объединяемость U_kl",
    ]
    return _md(pd.DataFrame(rows, columns=cols))


def table_profiles(o: Out) -> str:
    p = o.prof.copy()
    p["type"] = p["type"] + 1
    wide = p.pivot(index="feature", columns="type", values="median")
    order = [x for x in FG.FEATURE_LABELS if x in wide.index]
    rows = []
    for feat in order:
        rows.append([FG.FEATURE_LABELS[feat], f"`{feat}`", *[_n(v, 2) for v in wide.loc[feat]]])
    cols = ["Признак", "Колонка", *[f"Тип {t}" for t in wide.columns]]
    return _md(pd.DataFrame(rows, columns=cols), right=2)


def table_validation(o: Out) -> str:
    fin = o.final["candidate"]
    v = o.val.loc[o.val["cand"] == fin]
    rows = []
    for _, r in v.iterrows():
        exp = style.NA_TEXT if pd.isna(r["expected"]) else ("+" if r["expected"] > 0 else "−")
        axis = FG.FEATURE_LABELS.get(r["axis"], "") if isinstance(r["axis"], str) else ""
        rows.append(
            [
                INDICATOR_LABELS.get(r["indicator"], r["indicator"]),
                CHECK_KIND_LABELS[r["check"]] + (f": {axis.lower()}" if axis else ""),
                style.fmt_num(int(r["n"])),
                _n(r["stat"], 3),
                _n(r["null_mean"], 3),
                exp,
                style.fmt_p(float(r["q"])),
                "да" if bool(r["significant"]) else "нет",
            ]
        )
    cols = [
        "Показатель",
        "Проверка",
        "Узлов",
        "Значение",
        "Среднее на перестановках",
        "Ожидали",
        "q (БХ)",
        "Значимо",
    ]
    return _md(pd.DataFrame(rows, columns=cols), right=2)


def table_validation_methods(o: Out, cp: ClusterParams) -> str:
    """Внешняя проверка победителей всех методов: ε² и ΔR² по показателям."""
    rows = []
    for m in cp.methods:
        w = o.winners.get(m)
        if w is None:
            continue
        v = o.val.loc[(o.val["cand"] == w) & (o.val["check"] != "sign")].set_index(["indicator", "check"])
        cells = [f"{FG.mlabel(m)}, K = {int(o.cands.loc[w, 'k'])}"]
        for ind in INDICATOR_LABELS:
            cells.append(
                _n(v.loc[(ind, "difference"), "stat"], 3) if (ind, "difference") in v.index else style.NA_TEXT
            )
        for ind in INDICATOR_LABELS:
            key = (ind, "beyond_attributes")
            cells.append(_n(v.loc[key, "stat"], 3) if key in v.index else style.NA_TEXT)
        rows.append(cells)
    cols = [
        "Разбиение",
        *[f"ε²: {INDICATOR_LABELS[i]}" for i in INDICATOR_LABELS],
        *[f"ΔR²: {INDICATOR_LABELS[i]}" for i in INDICATOR_LABELS],
    ]
    return _md(pd.DataFrame(rows, columns=cols))


def table_synthetic_sbm(o: Out) -> str:
    s = o.syn.loc[o.syn["design"] == "sbm"]
    meths = [m for m in FG.METHOD_LABELS if m in s.columns and s[m].notna().any()]
    rows = []
    for _, r in s.iterrows():
        rows.append(
            [
                _n(r["graph_signal"], 1),
                _n(r["feature_signal"], 1),
                *[_n(r[m]) for m in meths],
                FG.mlabel(r["winner"]),
            ]
        )
    cols = ["Доля рёбер между группами", "Сигнал в X", *[FG.mlabel(m) for m in meths], "Победитель"]
    return _md(pd.DataFrame(rows, columns=cols), right=0)


def table_methods_final(o: Out, cp: ClusterParams, f: Mapping[str, Fact]) -> str:
    """Итоговая таблица «метод → плюсы, минусы, когда применять»: каждая ячейка — со своим числом и
    ссылкой."""
    t = lambda k: f[f"{SECTION}.{k}"].text  # noqa: E731
    c = o.cands
    rows = []
    optim = {
        "kmeans": "сумма квадратов расстояний до центров",
        "ward": "прирост внутригрупповой дисперсии при слиянии",
        "gmm": "правдоподобие смеси гауссиан",
        "hdbscan": "устойчивость кластеров плотности",
        "leiden": "модульность с разрешением γ",
        "louvain": "модульность с разрешением γ",
        "spectral": "разрез графа (нормированный лапласиан) + K-means",
        "shalileh_mirkin": "ρ·Σ(y − c)² + ξ·Σ(p − λ)²",
        "hybrid": "K-means во вложении G ⊕ X",
        "kmeans_joint": "K-means по корзине ⊕ X",
    }
    assume = {
        "kmeans": "сферические кластеры, лёгкие хвосты",
        "ward": "то же, иерархия",
        "gmm": "гауссовы кластеры",
        "hdbscan": "кластеры — сгустки плотности",
        "leiden": "сообщества — плотные части графа",
        "louvain": "то же",
        "spectral": "связный граф, K задан",
        "shalileh_mirkin": "признаки и связи восстанавливаются центрами",
        "hybrid": "вес графа α задан",
        "kmeans_joint": "есть координаты, из которых строится граф",
    }
    param = {
        "hdbscan": "min_cluster_size",
        "leiden": "разрешение γ",
        "louvain": "разрешение γ",
        "shalileh_mirkin": "K; ρ, ξ = 1",
        "hybrid": "K; α",
    }
    small = (
        f"тип пригородов из {t('small_kmeans_min')}–{t('small_kmeans_max')} узлов (табл. 3)"
        if (f"{SECTION}.small_kmeans_min" in f)
        else "мелкие типы"
    )
    pros_cons = {
        "kmeans": (
            f"устойчив и на недопустимых K (медианный ARI бутстрапа {t('stab_median_kmeans')}, "
            f"`candidates.csv`)",
            f"хвост доступности рынков даёт {small}: все K недопустимы",
            f"сигнал только в признаках, хвосты усечены (синтетика: при сигнале в графе 0 лучший метод по X "
            f"— "
            f"ARI {t('syn_nograph_best_feat')}, рис. 1)",
        ),
        "ward": (
            "детерминирован, одно дерево на все K",
            f"те же мелкие типы; устойчивость ниже K-means (медианный ARI {t('stab_median_ward')}, "
            f"`candidates.csv`)",
            "нужна иерархия типов и подтипов",
        ),
        "gmm": (
            f"допустим при K = {t('win_gmm_k')}; разные формы кластеров",
            f"не согласен с методами по графу (ARI с итогом {t('ari_hybrid_gmm')}, рис. 5)",
            "признаки близки к гауссовым, граф не нужен",
        ),
        "hdbscan": (
            "сам находит шум и K",
            f"в X нет сгустков плотности: при min_cluster_size {t('hdbscan_mcs_min')}–{t('hdbscan_mcs_max')} "
            f"K не больше {t('hdbscan_kmax')} (раздел 5)",
            "плотные компактные группы в малой размерности",
        ),
        "leiden": (
            "гарантирует связные сообщества; допустим при всех K (табл. 2)",
            f"устойчивость к удалению рёбер {t('pert_leiden')} против {t('pert_hybrid')} у гибрида (табл. 2)",
            "сигнал только в графе (синтетика SBM: ARI " + t("sbm_low_leiden") + ", табл. 1)",
        ),
        "louvain": (
            "быстрый",
            f"достиг {t('louvain_n_reached')} из {t('n_grid')} значений K (пропуски: {t('louvain_missing')})",
            "быстрый первый взгляд на сообщества",
        ),
        "spectral": (
            f"устойчив к бутстрапу и рёбрам ({t('pert_spectral')}; табл. 2)",
            "видит только граф: признаки места не участвуют",
            "граф связен, сигнал в графе сильнее, чем в признаках",
        ),
        "shalileh_mirkin": (
            "одна целевая функция по X и G; признаки и связи восстанавливаются центрами (раздел 2)",
            f"на X с хвостами все K недопустимы (мелкий тип {t('small_shalileh_mirkin_min')}–"
            f"{t('small_shalileh_mirkin_max')} узлов, табл. 3); в синтетике слабее гибрида "
            f"({t('syn_strong_shalileh_mirkin')} против {t('syn_strong_hybrid')}, рис. 1)",
            "хвосты X усечены: тогда побеждает (раздел 6, «Другие входы»), либо граф без координат (SBM, "
            "табл. 1)",
        ),
        "hybrid": (
            f"устойчив ({t('final_stability')}), все типы устойчивы по Хеннигу (табл. 9)",
            f"при α = {t('alpha')} почти совпадает со спектральной (ARI {t('ari_hybrid_spectral')}, рис. 4); "
            f"зависит от правила рёбер (ARI {t('var_graph_basket_cos_ari_same')} с сетью косинуса, табл. 8)",
            f"граф без исходных координат (SBM: ARI {t('sbm_mid_hybrid')} при доле внешних рёбер 0,5, табл. "
            f"1)",
        ),
        "kmeans_joint": (
            f"лучший в синтетике, где граф построен из координат ({t('syn_joint_won')} из "
            f"{t('syn_both_cells')} "
            "ячеек, рис. 1)",
            f"на данных все K недопустимы (тип пригородов из {t('suburbs_size')} узлов, табл. 3)",
            "координаты, из которых строится граф, доступны, хвосты X усечены",
        ),
    }
    tt = o.timing.set_index("step")["seconds"]
    for m in cp.methods:
        w = o.winners.get(m)
        st = _n(c.loc[w, "stability"]) if w else style.NA_TEXT
        icvi = (
            f"z SW {_n(c.loc[w, 'z_sw'], 1)}, z AVI {_n(c.loc[w, 'z_avi'], 0)}" if w else "нет допустимых K"
        )
        secs = (
            float(tt.get(w, np.nan))
            if w
            else float(np.nanmean([tt.get(x, np.nan) for x in c.index[c["method"] == m]]))
        )
        pro, con, when = pros_cons[m]
        rows.append(
            [
                FG.mlabel(m),
                USES[m],
                optim[m],
                assume[m],
                pro,
                con,
                when,
                param.get(m, "K"),
                st,
                icvi,
                _bucket(secs) if np.isfinite(secs) else style.NA_TEXT,
            ]
        )
    cols = [
        "Метод",
        "Использует",
        "Оптимизирует",
        "Предпосылки",
        "Плюсы",
        "Минусы",
        "Когда применять",
        "Параметр",
        "Устойчивость",
        "ICVI (победитель метода)",
        "Время",
    ]
    return _md(pd.DataFrame(rows, columns=cols), right=len(cols))


def table_small(o: Out) -> str:
    s = o.small
    if s.empty:
        return "Недопустимо мелких типов нет."
    g = s.groupby("method").agg(n=("cluster", "size"), smin=("size", "min"), smax=("size", "max"))
    feat = s.groupby("method")["top_feature"].agg(lambda x: x.value_counts().index[0])
    rows = []
    for m in FG.METHOD_LABELS:
        if m in g.index:
            rows.append(
                [
                    FG.mlabel(m),
                    str(int(g.loc[m, "n"])),
                    str(int(g.loc[m, "smin"]))
                    if g.loc[m, "smin"] == g.loc[m, "smax"]
                    else f"{int(g.loc[m, 'smin'])}–{int(g.loc[m, 'smax'])}",
                    FG.FEATURE_LABELS.get(feat[m], feat[m]),
                ]
            )
    cols = ["Метод", "Мелких типов (по всем K)", "Узлов в мелком типе", "Признак с наибольшим отклонением"]
    return _md(pd.DataFrame(rows, columns=cols))


TABLES: dict[str, Callable] = {
    "comparison": lambda o, cp, f: table_comparison(o, cp),
    "eligible": lambda o, cp, f: table_candidates_eligible(o, cp),
    "level2_all": lambda o, cp, f: table_level2_all(o),
    "sensitivity": lambda o, cp, f: table_sensitivity(o),
    "icvi_checks": lambda o, cp, f: table_icvi_checks(o, cp),
    "variants": lambda o, cp, f: table_variants(o),
    "types": lambda o, cp, f: table_types(o),
    "profiles": lambda o, cp, f: table_profiles(o),
    "validation": lambda o, cp, f: table_validation(o),
    "validation_methods": lambda o, cp, f: table_validation_methods(o, cp),
    "synthetic_sbm": lambda o, cp, f: table_synthetic_sbm(o),
    "methods_final": table_methods_final,
    "small": lambda o, cp, f: table_small(o),
}


# --- Рисунки и сборка ----------------------------------------------------------------------------


def make_figures(cfg: Config, cp: ClusterParams, o: Out) -> list[FG.FigureInfo]:
    fam = {m: cp.family_of(m) for m in cp.methods}
    metrics = list(cp.icvi_metrics)
    figs = [
        FG.fig_synthetic(o.dir, o.syn, fam, int(cp.synthetic["repeats"])),
        FG.fig_methods(o.dir, level1_table_all(o), o.final["method"]),
        FG.fig_by_k(o.dir, o.cands, metrics),
        FG.fig_alpha(o.dir, o.alpha, metrics),
        FG.fig_agreement(o.dir, o.ari, o.cands),
        FG.fig_profiles(o.dir, o.prof, o.final["method"], int(o.final["k"])),
    ]
    processed = Path(cfg["paths"]["processed"])
    members = read_table(processed / "features_members.parquet", FEATURES_MEMBERS)
    final = pd.read_parquet(processed / "cluster_final.parquet")
    figs.append(FG.fig_map(o.dir, cfg, final, members, int(o.final["k"]), o.final["method"], ami_region(cfg)))
    return figs


def ami_region(cfg: Config) -> float:
    """AMI итоговых типов с группой региона: повторяют ли типы карту регионов."""
    from sklearn.metrics import adjusted_mutual_info_score

    from munnet.contracts import FEATURES_NODES

    processed = Path(cfg["paths"]["processed"])
    final = pd.read_parquet(processed / "cluster_final.parquet")
    nodes = read_table(processed / "features_nodes.parquet", FEATURES_NODES).set_index("territory_id")
    grp = nodes.loc[final["territory_id"], "region_group"].to_numpy()
    return float(adjusted_mutual_info_score(grp, final["type"].to_numpy()))


def render(template: str, facts: Mapping[str, Fact], o: Out, cp: ClusterParams, figs, img_rel: str) -> str:
    text = _NOTE.sub("", template)
    text = eda_report.fill(text, facts)
    text = _TABLE.sub(lambda m: TABLES[m.group(1)](o, cp, facts), text)
    by_id = {f.fid: f for f in figs}

    def fig_md(m: re.Match) -> str:
        fg = by_id[m.group(1)]
        n = int(fg.fid[1:])
        return (
            f'<a id="fig-{n}"></a>\n\n![{fg.alt}]({img_rel}/{fg.png.name})\n\n'
            f"*Рисунок {n}. {fg.title}. {fg.subtitle}. Источник: {fg.source}.* "
            f"Данные: `outputs/cluster/figures/{fg.data_csv.name}`."
        )

    text = _FIGURE.sub(fig_md, text)
    text = eda_report.unwrap_paragraphs(text)
    return eda_report.nbsp_markdown(text)


def build(cfg: Config, cp: ClusterParams, out: Path) -> Path | None:
    """Рисунки, факты, проверка утверждений, отчёт ``clustering.report`` и копии PNG в ``report_images``."""
    if not cp.report:
        log.info("cluster: clustering.report пуст — отчёт не собирается")
        return None
    o = Out(out)
    figs = make_figures(cfg, cp, o)
    facts = build_facts(o, cp)
    _fact(facts, "ami_region", ami_region(cfg), "num3")
    write_json(
        {k: {"value": x.value, "text": x.text} for k, x in sorted(facts.items())}, out / "report_facts.json"
    )
    check_claims(facts)
    report = Path(cp.report)
    images = Path(cp.report_images)
    try:
        img_rel = Path(os.path.relpath(images, report.parent)).as_posix()
    except ValueError:
        img_rel = images.as_posix()
    md = render(TEMPLATE.read_text(encoding="utf-8"), facts, o, cp, figs, img_rel)
    problems = eda_report.lint_ru(md, LINT_ALLOW)
    if problems:
        raise QCError("отчёт cluster: типографика: " + "; ".join(problems[:10]))
    report.parent.mkdir(parents=True, exist_ok=True)
    images.mkdir(parents=True, exist_ok=True)
    for fg in figs:
        dest = images / fg.png.name
        if not dest.exists() or dest.read_bytes() != fg.png.read_bytes():
            shutil.copyfile(fg.png, dest)
    old = report.read_text(encoding="utf-8") if report.exists() else None
    if old != md:
        report.write_text(md, encoding="utf-8", newline="\n")
    log.info("cluster: отчёт %s", report)
    return report
