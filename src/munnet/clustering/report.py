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
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from munnet import style
from munnet.clustering import decide as DE
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
    "tie_chain": "цепочка равенств с допуском устойчивости (толкование)",
}
ICVI_LABELS: dict[str, str] = {
    "no_avu": "(а) качество на G без AVU",
    "avi_mq_one": "(б) AVI и MQ — одна метрика",
    "no_s_dbw": "(в) качество в X без S_Dbw",
    "s_dbw_own": "(г) S_Dbw в варианте own",
    "nan_worst": "(д) неопределённая метрика — худший ранг",
    "noise_cluster": "(е) шум HDBSCAN — отдельный кластер",
    "raw_graph": "(з) качество на G: AVI с поправкой на случайность и сырая MQ",
    "raw_graph_avu": "(и) то же и z AVU",
    "raw_sw": "(к) качество в X: сырой SW",
    "raw_all": "(л) сырой SW и AVI с поправкой, сырая MQ",
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


def _changes_text(items: list[str], prefix: str, none_text: str) -> str:
    """«<prefix>: а; б» или ``none_text``, если список пуст (без «меняют 0 из них: нет»)."""
    return f"{prefix}: {'; '.join(items)}" if items else none_text


def _verb(n: int) -> str:
    """«меняет» при одном элементе, «меняют» — при нескольких."""
    return "меняет" if n == 1 else "меняют"


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
        self.grid = read("threshold_grid")
        self.reslim = read("resolution_limit")
        self.kef = read("kefrin_curve")
        self.seed_freq = read("seed_frequency")
        self.seed_runs = read("seed_runs")
        self.tol_checks = read("sensitivity_tolerance")
        self.qtol = read("quality_tolerance")
        self.rules = (out / "tree_rules.txt").read_text(encoding="utf-8")

    @property
    def winners(self) -> dict[str, str]:
        return dict(self.js["winners"])


def level1_table_all(o: Out) -> pd.DataFrame:
    """Победители уровня K всех методов с рангами качества среди всех методов (уровень 2 без ограничения)."""
    t = o.levels_all.loc[o.levels_all["level"] == 2].copy()
    t = t.rename(columns={f"crit_{c}": c for c in CRITERIA})
    return t[["cand", "method", "k", *CRITERIA, "on_front", "score"]]


TOL_GRID_SCALES: tuple[float, ...] = (0.0, 0.5, 1.0, 1.5, 2.0)


def tolerance_sensitivity(o: Out, cp: ClusterParams) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Шум самого допуска качества (отступление от предрегистрации): победитель среди всех семейств при seed
    прогона, если допуск оценить без одного из случайных базисов (``quality_tolerance.csv``), и на сетке
    допусков от 0 до удвоенного по каждой оси. Правило то же, меняется только допуск."""
    qt = o.qtol
    cols = ["quality_features", "quality_graph"]

    def winner(tol: Mapping[str, float]) -> str:
        cpt = replace(cp, tie={**cp.tie, **{k: float(v) for k, v in tol.items()}})
        return DE.run_rule(o.cands, cpt, methods=cp.methods).winner

    jk = []
    for s in sorted(qt["seed"].unique()):
        tol = qt.loc[qt["seed"] != s].groupby("cand")[cols].std().median()
        jk.append({"dropped_seed": int(s), **{k: float(tol[k]) for k in cols}, "winner": winner(tol)})
    full = o.js["quality_tolerance"]
    grid = [
        {
            "scale_features": a,
            "scale_graph": b,
            "winner": winner({cols[0]: a * full[cols[0]], cols[1]: b * full[cols[1]]}),
        }
        for a in TOL_GRID_SCALES
        for b in TOL_GRID_SCALES
    ]
    return pd.DataFrame(jk), pd.DataFrame(grid)


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
        hk = set(small.loc[small["method"] == "hybrid", "k"].astype(int))
        need = {k for k in cp.k_grid if k >= 5}
        _fact(f, "hybrid_small_all_k5", int(need <= hk and not ({3, 4} & hk)), "int")
        mk = small.loc[small["top_feature"] == "market_access_rel", "size"]
        _fact(f, "small_market_size_min", int(mk.min()) if len(mk) else 0, "int")
        _fact(f, "small_market_size_max", int(mk.max()) if len(mk) else 0, "int")
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
    # проверка all_eligible меняет множество кандидатов (все семейства), а не агрегатор: её победитель —
    # факт all_winner, в счёт проверок уровня итога она не входит
    el = ch.loc[(ch["scope"] == "eligible") & ~ch["check"].isin(["main", "all_eligible"])]
    _fact(f, "n_checks", len(el), "int")
    _fact(f, "n_checks_same", int(el["same_as_main"].astype(bool).sum()), "int")
    oe = o.orders.loc[o.orders["scope"] == "eligible"]
    _fact(f, "orders_same", int((oe["winner"] == fin).sum()), "int")
    _fact(f, "orders_total", len(oe), "int")
    al = ch.loc[(ch["scope"] == "all") & (ch["check"] != "main")]
    _fact(f, "all_n_checks", len(al), "int")
    _fact(f, "all_n_same", int(al["same_as_main"].astype(bool).sum()), "int")
    changed = al.loc[~al["same_as_main"].astype(bool)]
    changed_items = [
        f"{CHECK_LABELS[r['check']]} — {FG.mlabel(r['method'])}, K = {int(r['k'])}"
        for _, r in changed.iterrows()
    ]
    _fact(f, "all_changed", "; ".join(changed_items) or "нет", "str")
    _fact(
        f,
        "all_changed_text",
        _changes_text(
            changed_items, f"победителя {_verb(len(changed_items))}", "победителя не меняет ни одна"
        ),
        "str",
    )
    _fact(f, "all_joint_winner", FG.mlabel(al.set_index("check").loc["joint", "method"]), "str")
    _fact(f, "all_joint_k", int(al.set_index("check").loc["joint", "k"]), "int")
    for scope in ("all", "all_level2"):
        od = o.orders.loc[o.orders["scope"] == scope]
        for m in ("hybrid", "leiden", "louvain", "spectral"):
            _fact(f, f"lex_{scope}_{m}", int((od["method"] == m).sum()), "int")
    b2 = ch.loc[(ch["scope"] == "all_level2") & (ch["check"] == "borda")]
    _fact(f, "borda_l2_winner", _cand_label(c, b2["winner"].iloc[0]), "str")
    lw = o.levels_all.loc[o.levels_all["level"] == 2].set_index("cand")
    front = lw.loc[lw["on_front"].astype(bool)]
    ranks = front[[f"crit_{c}" for c in CRITERIA]].rank(ascending=False, method="min").sum(axis=1)
    _fact(f, "borda_hybrid", float(ranks.get(fin, np.nan)), "int")
    _fact(
        f,
        "borda_front",
        "; ".join(f"{_cand_label(c, x)} — {int(v)}" for x, v in ranks.sort_values().items()),
        "str",
    )
    b2_winner = b2["winner"].iloc[0]
    outcome, n_tied, consistent = _borda_outcome(
        ranks, c["stability"], b2_winner, lambda x: _cand_label(c, x)
    )
    _fact(f, "borda_outcome", outcome, "str")
    _fact(f, "borda_n_tied", n_tied, "int")
    _fact(f, "borda_consistent", consistent, "int")
    _fact(f, "all_tradeoff", _tradeoff(lw, fin, c, cp), "str")
    _fact(f, "borda_leiden", float(ranks.get(winners_of(o)["leiden"], np.nan)), "int")
    # устойчивость выбора к seed и допуск качества по шуму базиса
    tolq = js["quality_tolerance"]
    _fact(f, "tol_qf", float(tolq["quality_features"]), "num2")
    _fact(f, "tol_qg", float(tolq["quality_graph"]), "num2")
    sr = o.seed_runs
    seeds = sorted(sr["seed"].unique())
    _fact(f, "n_seeds", len(seeds), "int")
    _fact(f, "seed", int(cp.seed), "int")
    _fact(f, "seed_min", int(min(seeds)), "int")
    _fact(f, "seed_max", int(max(seeds)), "int")
    for rule in ("prereg", "tolerance"):
        for scope in ("eligible", "all"):
            sel = (sr["rule"] == rule) & (sr["scope"] == scope) & (sr["kind"] == "check")
            m = sr.loc[sel & (sr["check"] == "main")]
            _fact(f, f"sf_{rule}_{scope}", _freq_text(m["winner"], o.cands, len(seeds)), "str")
            _fact(f, f"sf_{rule}_{scope}_final", int((m["winner"] == fin).sum()), "int")
        od = sr.loc[(sr["rule"] == rule) & (sr["kind"] == "order") & (sr["scope"] == "all_level2")]
        _fact(f, f"sf_{rule}_lex", _freq_text(od["winner"], o.cands, len(od)), "str")
        m_all = sr.loc[(sr["rule"] == rule) & (sr["scope"] == "all") & (sr["kind"] == "check")]
        n_win = int(m_all.loc[m_all["check"] == "main", "winner"].nunique())
        _fact(f, f"sf_{rule}_all_n", n_win, "int")
        _fact(
            f,
            f"sf_{rule}_all_verdict",
            "победитель меняется от seed" if n_win > 1 else "победитель при этих seed один и тот же",
            "str",
        )
    # разбиения победителей среди всех семейств при разных seed: наименьший ARI с разбиением seed прогона
    sfq = o.seed_freq
    sfm = sfq.loc[
        (sfq["rule"] == "prereg")
        & (sfq["scope"] == "all")
        & (sfq["kind"] == "check")
        & (sfq["check"] == "main")
    ]
    _fact(
        f,
        "sf_all_minari",
        "; ".join(
            f"{_cand_label(c, w)} — {_n(a)}"
            for w, a in zip(sfm["winner"], sfm["min_ari_labels"], strict=True)
        ),
        "str",
    )
    _fact(f, "sf_all_minari_min", float(sfm["min_ari_labels"].min()), "num2")
    _fact(f, "spectral_stability", float(c.loc[winners["spectral"], "stability"]), "num2")
    # шум самого допуска: он оценён по n_seeds базисам; без одного базиса и при допуске от 0 до удвоенного
    jk, grid = tolerance_sensitivity(o, cp)
    for col, key in (("quality_features", "qf"), ("quality_graph", "qg")):
        _fact(f, f"tol_jk_{key}_min", float(jk[col].min()), "num3")
        _fact(f, f"tol_jk_{key}_max", float(jk[col].max()), "num3")
    _fact(f, "tol_jk_n", len(jk), "int")
    _fact(f, "tol_jk_winners", _freq_text(jk["winner"], c, len(jk)), "str")
    _fact(f, "tol_grid_n", len(grid), "int")
    _fact(f, "tol_grid_winners", _freq_text(grid["winner"], c, len(grid)), "str")
    tc = o.tol_checks
    for scope in ("eligible", "all"):
        w = tc.loc[(tc["scope"] == scope) & (tc["check"] == "main"), "winner"].iloc[0]
        _fact(f, f"tol_{scope}_winner", _cand_label(o.cands, w), "str")
    all_main = ch.loc[(ch["scope"] == "all") & (ch["check"] == "main"), "winner"].iloc[0]
    _fact(f, "all_winner", _cand_label(o.cands, all_main), "str")
    _fact(f, "all_winner_method", FG.mlabel(o.cands.loc[all_main, "method"]), "str")
    _fact(f, "all_winner_k", int(o.cands.loc[all_main, "k"]), "int")
    od = o.orders.loc[o.orders["scope"] == "all_level2"]
    _fact(f, "lex_l2_text", _freq_text(od["winner"], o.cands, len(od)), "str")
    # сетка порогов
    g = o.grid
    gs = g.loc[g["chain"] == "strict"]
    _fact(f, "grid_cells", len(gs), "int")
    _fact(f, "grid_same_eligible", int((gs["winner_eligible"] == fin).sum()), "int")
    _fact(f, "grid_same_all", int((gs["winner_all"] == fin).sum()), "int")
    s55 = gs.loc[np.isclose(gs["max_share"], 0.55)]
    _fact(f, "grid55_all_winner", FG.mlabel(o.cands.loc[s55["winner_all"].iloc[0], "method"]), "str")
    _fact(f, "grid55_all_k", int(o.cands.loc[s55["winner_all"].iloc[0], "k"]), "int")
    gt = g.loc[(g["chain"] == "tolerance") & np.isclose(g["max_share"], 0.55)]
    _fact(f, "grid55_tol_hybrid_k", int(gt["hybrid_k"].iloc[0]), "int")
    _fact(
        f,
        "grid50_same_all_share",
        int((gs.loc[np.isclose(gs["max_share"], 0.5), "winner_all"] == fin).all()),
        "int",
    )
    by_max = gs.groupby("max_share")[["winner_eligible", "winner_all"]].nunique()
    _fact(f, "grid_minshare_effect", int((by_max > 1).any(axis=1).sum()), "int")
    _fact(f, "grid_eligible_same", int((gs["winner_eligible"] == fin).all()), "int")
    s55m = gs.loc[np.isclose(gs["max_share"], 0.55) & np.isclose(gs["min_share"], cp.min_cluster_share)]
    _fact(f, "grid55_strict_hybrid_k", int(s55m["hybrid_k"].iloc[0]), "int")
    _fact(f, "grid55_hybrid_feasible", str(s55m["hybrid_feasible_k"].iloc[0]).replace(",", ", "), "str")
    _fact(
        f,
        "all_winner_is_final",
        int(ch.loc[(ch["scope"] == "all") & (ch["check"] == "main"), "winner"].iloc[0] == fin),
        "int",
    )
    _fact(
        f,
        "hybrid_k3_k4_stab_diff",
        float(c.loc["hybrid_k04", "stability"] - c.loc["hybrid_k03", "stability"]),
        "num3",
    )
    # предел разрешения
    rl = o.reslim
    _fact(f, "rl_threshold", float(rl["threshold_gamma1"].iloc[0]), "int")
    _fact(f, "rl_threshold_max", float(rl["threshold"].max()), "int")
    _fact(f, "rl_min_inner", float(rl["min_inner_weight"].min()), "int")
    _fact(f, "rl_n_below", int(rl["n_below"].sum()), "int")
    _fact(f, "rl_disconnected", int(rl["n_disconnected"].sum()), "int")
    _fact(f, "rl_n_cands", len(rl), "int")
    # KEFRiN в варианте статьи
    kf = o.kef
    _fact(
        f,
        "kef_inputs_feasible",
        int(kf.loc[kf["features"] == "inputs", "feasible"].astype(bool).sum()),
        "int",
    )
    z1 = kf.loc[(kf["features"] == "zscore") & np.isclose(kf["xi_over_rho"], 1.0)]
    _fact(f, "kef_z1_feasible", int(z1["feasible"].astype(bool).sum()), "int")
    _fact(f, "kef_z1_minshare_k4", float(z1.loc[z1["k"] == 4, "min_share"].iloc[0]), "pct")
    _fact(f, "kef_z1_graph_share", float(z1["graph_share"].iloc[0]), "pct")
    zf = kf.loc[(kf["features"] == "zscore") & kf["feasible"].astype(bool)]
    _fact(f, "kef_z_feasible", len(zf), "int")
    _fact(
        f,
        "kef_z_feasible_list",
        "; ".join(
            f"ξ/ρ = {style.fmt_num(x, 0)}, K = {int(k)}"
            for x, k in zip(zf["xi_over_rho"], zf["k"], strict=True)
        )
        or "нет",
        "str",
    )
    _fact(f, "kef_ari_max", float(kf["ari_final"].max()), "num2")
    _fact(f, "kef_z_feasible_min_xi", float(zf["xi_over_rho"].min()) if len(zf) else float("nan"), "num2")
    _fact(
        f,
        "kef_z_feasible_graph_share_min",
        float(zf["graph_share"].min()) if len(zf) else float("nan"),
        "pct",
    )
    _fact(f, "kef_z_feasible_ari_max", float(zf["ari_final"].max()) if len(zf) else float("nan"), "num2")
    raw = o.icvi.loc[o.icvi["check"].isin(DE.RAW_CHECKS)]
    r3 = raw.loc[raw["level"] == 3]
    _fact(f, "raw_n", r3["check"].nunique(), "int")
    _fact(f, "raw_l2_changed", int(raw.loc[raw["level"] == 2, "changed"].astype(bool).sum()), "int")
    _fact(f, "raw_l3_changed", int(r3["changed"].astype(bool).sum()), "int")
    r3c = r3.loc[r3["changed"].astype(bool)]
    _fact(
        f,
        "raw_l3_changes",
        "; ".join(
            f"{ICVI_LABELS[r['check']]} — {FG.mlabel(r['method'])}, K = {int(r['k'])}"
            for _, r in r3c.iterrows()
        )
        or "нет",
        "str",
    )
    # метрики графа — (з), (и), (л); сырой SW (к) — отдельно: это качество в X, а не на G
    r3g = r3c.loc[r3c["check"] != "raw_sw"]
    _fact(
        f,
        "raw_l3_changes_graph",
        _changes_text(
            [
                f"{ICVI_LABELS[r['check']]} — {FG.mlabel(r['method'])}, K = {int(r['k'])}"
                for _, r in r3g.iterrows()
            ],
            "сменили победителя",
            "победителя ни одна не меняет",
        ),
        "str",
    )
    r3i = r3.set_index("check")
    _fact(f, "raw_l3_winner_graph", FG.mlabel(r3i.loc["raw_graph", "method"]), "str")
    _fact(f, "raw_l3_winner_graph_k", int(r3i.loc["raw_graph", "k"]), "int")
    _fact(f, "raw_l3_winner_sw", FG.mlabel(r3i.loc["raw_sw", "method"]), "str")
    _fact(
        f,
        "raw_l3_sw_note",
        "это смена победителя"
        if bool(r3i.loc["raw_sw", "changed"])
        else "победитель тот же, что по z-оценкам",
        "str",
    )
    rg = r3.set_index("check").loc["raw_graph", "winner"]
    zw = ch.loc[(ch["scope"] == "all") & (ch["check"] == "main"), "winner"].iloc[0]
    _fact(
        f,
        "raw_graph_verdict",
        f"победитель тот же, что по z-оценкам, — {_cand_quoted(c, zw)}"
        if rg == zw
        else f"побеждает {_cand_quoted(c, rg)}, а не победитель по z-оценкам — {_cand_quoted(c, zw)}: "
        "z-оценки качества на G смещены по K (`docs/icvi.md`)",
        "str",
    )
    rc = DE.with_raw_columns(c)
    lw = winners_of(o)["leiden"]
    for key, cand in (("hybrid", fin), ("leiden", lw)):
        _fact(f, f"avi_adj_{key}", float(rc.loc[cand, "raw_avi_adj"]), "num3")
        _fact(f, f"mq_raw_{key}", float(rc.loc[cand, "raw_mq"]), "num3")
        _fact(f, f"sw_raw_{key}", float(rc.loc[cand, "raw_sw"]), "num3")
    l2a = o.levels_all.loc[o.levels_all["level"] == 2].set_index("cand")
    _fact(
        f,
        "l2_qg_hybrid_minus_leiden",
        float(l2a.loc[fin, "crit_quality_graph"] - l2a.loc[lw, "crit_quality_graph"]),
        "num2",
    )
    ic = o.icvi.loc[~o.icvi["check"].isin(DE.RAW_CHECKS)]
    l2c = ic.loc[ic["level"] == 2]
    _fact(f, "icvi_n", l2c["check"].nunique(), "int")
    _fact(f, "icvi_l2_changed", int(l2c["changed"].astype(bool).sum()), "int")
    _fact(f, "icvi_l3_changed", int(ic.loc[ic["level"] == 3, "changed"].astype(bool).sum()), "int")
    l3c = ic.loc[(ic["level"] == 3) & ic["changed"].astype(bool)]
    _fact(
        f,
        "icvi_l3_text",
        _changes_text(
            [
                f"{ICVI_LABELS[r['check']]} — {FG.mlabel(r['method'])}, K = {int(r['k'])}"
                for _, r in l3c.iterrows()
            ],
            f"победителя {_verb(len(l3c))} {len(l3c)} из них",
            "победителя не меняет ни одна из них",
        ),
        "str",
    )
    _fact(
        f,
        "icvi_l3_changes",
        "; ".join(
            f"{ICVI_LABELS[r['check']]} — {FG.mlabel(r['method'])}, K = {int(r['k'])}"
            for _, r in l3c.iterrows()
        )
        or "нет",
        "str",
    )
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
    # у каких методов сменилось K и сколько K у них на фронте первого уровня (без двух источников)
    l1all = o.levels_all.loc[o.levels_all["level"] == 1]
    front_k = l1all.groupby("method")["on_front"].apply(lambda s: int(s.astype(bool).sum()))
    moved = list(dict.fromkeys(l1c["method"]))
    _fact(
        f,
        "icvi_l1_note",
        (
            "Число K на фронте основного правила у методов со сменой K: "
            + "; ".join(f"{FG.mlabel(m)} — {int(front_k.get(m, 0))}" for m in moved)
            + ". Лишняя или убранная метрика качества переставляет очки Копленда между K фронта."
        )
        if moved
        else "Выбор K внутри методов от набора метрик не зависит.",
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
    best_both = both["best"].map(lambda m: fam.get(m) in ("attributed", "fusion"))
    _fact(f, "syn_both_won", int(best_both.sum()), "int")
    clear = both["winner"].map(lambda m: fam.get(m) in ("attributed", "fusion"))
    _fact(f, "syn_both_clear", int(clear.sum()), "int")
    _fact(f, "syn_both_tie", int((both["winner"] == "tie").sum()), "int")
    _fact(f, "syn_joint_won", int((both["best"] == "kmeans_joint").sum()), "int")
    _fact(f, "syn_joint_clear", int((both["winner"] == "kmeans_joint").sum()), "int")
    _fact(f, "syn_none_cells", int((s["winner"] == "none").sum()), "int")
    _fact(f, "syn_tie_cells", int((s["winner"] == "tie").sum()), "int")
    _fact(f, "syn_all_cells", len(s), "int")
    _fact(f, "syn_clear_cells", int((~s["winner"].isin(["tie", "none"])).sum()), "int")
    _fact(f, "syn_both_hybrid_best", int((both["best"] == "hybrid").sum()), "int")
    c0 = knn.loc[(knn["graph_signal"] == 0) & (knn["feature_signal"] == knn["feature_signal"].max())].iloc[0]
    for m in ("kmeans", "hybrid", "shalileh_mirkin", "kmeans_joint"):
        _fact(f, f"syn_g0_{m}", float(c0[m]), "num2")
    _fact(f, "syn_fmax", float(c0["feature_signal"]), "num1")
    _fact(f, "syn_min_ari", float(cp.synthetic.get("min_ari", 0.05)), "num2")
    _fact(f, "syn_ci", f"{round(100 * float(cp.synthetic.get('ci_level', 0.95)))}%", "str")
    g0 = knn.loc[(knn["graph_signal"] == 0) & (knn["winner"] != "none")]
    g0_sets = g0["winner_set"].fillna("").astype(str).str.split(",")
    _fact(f, "syn_g0_live", len(g0), "int")
    f0 = knn.loc[(knn["feature_signal"] == 0) & (knn["graph_signal"] > 0)]
    f0_sets = f0["winner_set"].fillna("").astype(str).str.split(",")
    _fact(f, "syn_f0_n", len(f0), "int")
    _fact(
        f,
        "syn_f0_both_tie",
        int(f0_sets.map(lambda x: bool({"hybrid", "kmeans_joint"} & set(x))).sum()),
        "int",
    )
    _fact(f, "syn_g0_kefrin_tie", int(g0_sets.map(lambda x: "shalileh_mirkin" in x).sum()), "int")
    # шумовой граф (kNN без сигнала в графе и блочный граф с долей внешних рёбер от 0,7); в ячейках,
    # где лучший средний ARI не ниже порога, — ничья по интервалам K-means, гауссовой смеси и KEFRiN
    noisy_all = s.loc[
        ((s["design"] == "knn") & (s["graph_signal"] == 0))
        | ((s["design"] == "sbm") & (s["graph_signal"] >= 0.7))
    ]
    _fact(f, "syn_noisy_n", len(noisy_all), "int")
    _fact(f, "syn_noisy_none", int((noisy_all["winner"] == "none").sum()), "int")
    noisy = noisy_all.loc[noisy_all["winner"] != "none"]
    noisy_sets = noisy["winner_set"].fillna("").astype(str).str.split(",")
    _fact(f, "syn_noisy_live", len(noisy), "int")
    _fact(
        f,
        "syn_noisy_x_tie",
        int(
            (
                (noisy["winner"] == "tie")
                & noisy_sets.map(lambda x: {"kmeans", "gmm", "shalileh_mirkin"} <= set(x))
            ).sum()
        ),
        "int",
    )
    # сигнал только в графе (kNN): ничья четырёх методов, видящих граф, и у кого лучший средний ARI
    four = {"leiden", "spectral", "hybrid", "kmeans_joint"}
    _fact(
        f,
        "syn_f0_four_tie",
        int(((f0["winner"] == "tie") & f0_sets.map(lambda x: four <= set(x))).sum()),
        "int",
    )
    _fact(
        f,
        "syn_f0_best_text",
        ", ".join(f"{FG.mlabel(m)} — {n}" for m, n in f0["best"].value_counts().items()),
        "str",
    )
    # явные победы методов только по графу во всех ячейках обоих генераторов
    gw = s.loc[s["winner"].isin(["leiden", "louvain", "spectral"])]
    _fact(f, "syn_graph_only_clear_n", len(gw), "int")
    _fact(
        f,
        "syn_graph_only_clear_text",
        "; ".join(
            (
                f"табл. 1а, доля внешних рёбер {_n(r['graph_signal'], 1)}"
                if r["design"] == "sbm"
                else f"табл. 1, сигнал в G {_n(r['graph_signal'], 1)}"
            )
            + f", сигнал в X {_n(r['feature_signal'], 1)} — {FG.mlabel(r['winner'])}"
            + f" (ARI {_n(r[r['winner']])})"
            for _, r in gw.iterrows()
        )
        or "ни в одной ячейке",
        "str",
    )
    # порог «победителя нет»: лучший средний ARI ячеек по обе стороны
    best_ari = s.apply(lambda r: float(r[r["best"]]), axis=1)
    none = s["winner"] == "none"
    _fact(f, "syn_none_best_max", float(best_ari[none].max()) if none.any() else float("nan"), "num3")
    _fact(f, "syn_live_best_min", float(best_ari[~none].min()), "num3")
    _fact(f, "syn_counts", _syn_counts(s), "str")
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
            _fact(f, "sbm_mid_spectral", float(mid["spectral"].mean()), "num2")
            _fact(f, "sbm_mid_n", len(mid), "int")
            _fact(f, "sbm_mid_hybrid_best", int((mid["best"] == "hybrid").sum()), "int")
            _fact(f, "sbm_mid_hybrid_clear", int((mid["winner"] == "hybrid").sum()), "int")
            mid_sets = mid["winner_set"].fillna("").astype(str).str.split(",")
            _fact(
                f,
                "sbm_mid_tie_spectral",
                int(
                    (
                        (mid["winner"] == "tie") & mid_sets.map(lambda x: {"hybrid", "spectral"} <= set(x))
                    ).sum()
                ),
                "int",
            )
        fmax = sbm["feature_signal"] == sbm["feature_signal"].max()
        s7 = sbm.loc[np.isclose(sbm["graph_signal"], 0.7) & fmax]
        if len(s7):
            for m in ("kmeans", "hybrid", "shalileh_mirkin"):
                _fact(f, f"sbm7_{m}", float(s7[m].iloc[0]), "num2")
        hi = sbm.loc[sbm["graph_signal"] >= 0.7]
        _fact(f, "sbm_high_graph_max", float(hi[["leiden", "spectral"]].max().max()), "num2")
        _fact(f, "sbm_degree", float(cp.synthetic["sbm_degree"]), "int")
    # свойства индексов
    gm = c.loc[c["method"].isin(["leiden", "louvain", "spectral"])]
    tau = kendalltau(gm["z_avi"], gm["z_mq"], nan_policy="omit")[0]
    _fact(f, "tau_avi_mq", float(tau), "num2")
    _fact(f, "tau_avi_mq_n", int(gm[["z_avi", "z_mq"]].notna().all(axis=1).sum()), "int")
    # рост с K — у обоих рядов: минимум τ(K, z) по трём графовым методам отдельно для z AVI и z MQ
    for col in ("z_avi", "z_mq"):
        taus = [kendalltau(d["k"], d[col], nan_policy="omit")[0] for _, d in gm.groupby("method")]
        _fact(f, f"tau_k_{col[2:]}_min", float(min(taus)), "num2")
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


def _changes_with_aggregator(f: Mapping[str, Fact]) -> bool:
    """Среди всех семейств при seed прогона победителя меняет хотя бы одна проверка или порядок критериев."""
    return _v(f, "all_n_same") < _v(f, "all_n_checks") or ";" in str(_v(f, "lex_l2_text"))


CLAIMS: dict[str, Callable[[Mapping[str, Fact]], bool]] = {
    # Утверждения текста, которые шаблон пишет словами. Всё, что меняется от seed методов и случайного базиса
    # (победитель среди всех семейств, ничьи на фронте, лексикографические счёты), текст берёт из фактов
    # и формулирует нейтрально; здесь — то, что верно при seed 42, 43 и 44 (ворота этапа 3).
    # итог и допустимость
    "итог — гибрид с K = 4": lambda f: _v(f, "final_method") == "hybrid" and _v(f, "final_k") == 4,
    "у KEFRiN и базовой линии нет допустимых K; у гибрида — один": lambda f: (
        _v(f, "feas_shalileh_mirkin") == 0 and _v(f, "feas_kmeans_joint") == 0 and _v(f, "feas_hybrid") == 1
    ),
    "мелкие недопустимые типы почти все задаёт доступность рынков": lambda f: (
        _v(f, "n_small_market") >= 0.9 * _v(f, "n_small")
    ),
    "у гибрида мелкий тип при каждом K ≥ 5 и ни при K = 3, 4": lambda f: _v(f, "hybrid_small_all_k5") == 1,
    "у гибрида при K = 3 крупнейший тип больше половины": lambda f: _v(f, "hybrid_k3_max_share") > 0.5,
    "порог 1% не добавляет допустимых кандидатов": lambda f: _v(f, "extra01_n") == _v(f, "n_feasible"),
    # чувствительность и seed
    "среди методов с двумя источниками замены агрегатора, критериев, допусков и шага дают тот же итог": (
        lambda f: (
            _v(f, "n_checks_same") == _v(f, "n_checks") and _v(f, "orders_same") == _v(f, "orders_total")
        )
    ),
    "на уровне итога — гибрид K = 4 при всех seed и обоих правилах": lambda f: (
        _v(f, "sf_prereg_eligible_final") == _v(f, "n_seeds")
        and _v(f, "sf_tolerance_eligible_final") == _v(f, "n_seeds")
    ),
    "среди всех семейств победитель зависит от агрегатора и порогов": lambda f: (
        _changes_with_aggregator(f) and _v(f, "grid_same_all") < _v(f, "grid_cells")
    ),
    "сетка порогов: итог тот же во всех ячейках, наименьший тип победителей не меняет": lambda f: (
        _v(f, "grid_eligible_same") == 1 and _v(f, "grid_minshare_effect") == 0
    ),
    "гибрид K = 3 и K = 4 разводит разница устойчивости меньше допуска; при 55% итог хрупок": lambda f: (
        0 < _v(f, "hybrid_k3_k4_stab_diff") < _v(f, "tie_stability")
        and _v(f, "grid55_tol_hybrid_k") != _v(f, "grid55_strict_hybrid_k")
    ),
    "победителей среди всех семейств по seed двое, их разбиения от seed не зависят": lambda f: (
        _v(f, "sf_prereg_all_n") == 2 and _v(f, "sf_all_minari_min") >= 0.999
    ),
    "проверки по свойствам индексов итог не меняют": lambda f: _v(f, "icvi_l2_changed") == 0,
    "шесть проверок по свойствам индексов": lambda f: _v(f, "icvi_n") == 6,
    "z AVI и z MQ растут с K почти монотонно": lambda f: (
        _v(f, "tau_k_avi_min") >= 0.8 and _v(f, "tau_k_mq_min") >= 0.8
    ),
    "сравнимые между K метрики итог не меняют": lambda f: _v(f, "raw_l2_changed") == 0,
    "по сравнимым метрикам графа среди всех семейств — Leiden, а не победитель по z-оценкам; "
    "по z-оценкам на G гибрид выше Leiden": lambda f: (
        _v(f, "raw_l3_winner_graph") == FG.mlabel("leiden")
        and _v(f, "all_winner") != f"{_v(f, 'raw_l3_winner_graph')}, K = {_v(f, 'raw_l3_winner_graph_k')}"
        and _v(f, "avi_adj_leiden") > _v(f, "avi_adj_hybrid")
        and _v(f, "l2_qg_hybrid_minus_leiden") > 0
    ),
    # предел разрешения и KEFRiN
    "ни одно сообщество Leiden и Louvain не упирается в предел разрешения": lambda f: (
        _v(f, "rl_n_below") == 0
    ),
    "все сообщества Leiden и Louvain связны": lambda f: _v(f, "rl_disconnected") == 0,
    "KEFRiN на общем X недопустим при любом весе сети": lambda f: _v(f, "kef_inputs_feasible") == 0,
    "KEFRiN по статье (z, ρ = ξ = 1) недопустим": lambda f: _v(f, "kef_z1_feasible") == 0,
    "KEFRiN допустим только при сети, перевешивающей признаки": lambda f: (
        _v(f, "kef_z_feasible") == 0 or _v(f, "kef_z_feasible_graph_share_min") > 0.5
    ),
    "разбиения KEFRiN далеки от итога (все и допустимые)": lambda f: (
        _v(f, "kef_ari_max") < 0.5 and (_v(f, "kef_z_feasible") == 0 or _v(f, "kef_z_feasible_ari_max") < 0.5)
    ),
    "у KEFRiN z AVI ниже, чем у гибрида при любом K": lambda f: (
        _v(f, "kefrin_zavi_max") < _v(f, "hybrid_zavi_min")
    ),
    "сеть — меньшая часть разброса KEFRiN": lambda f: _v(f, "kefrin_graph_share") < 0.5,
    # гибрид и граф
    "итог почти совпадает со спектральной по одному графу": lambda f: _v(f, "ari_hybrid_spectral") > 0.9,
    "при α = 0,5 гибрид близок к спектральной и далёк от K-means по X": lambda f: (
        _v(f, "alpha_ari_spectral") > 0.9 and _v(f, "alpha_ari_kmeans") < 0.5
    ),
    "методы по графу согласны с итогом, гауссова смесь — нет": lambda f: (
        _v(f, "ari_hybrid_leiden") > 0.5 and _v(f, "ari_hybrid_gmm") < 0.3
    ),
    "Борда на фронте: победитель — среди лучших по сумме мест, ничью решает устойчивость": lambda f: (
        _v(f, "borda_consistent") == 1
    ),
    "гибрид устойчивее Leiden": lambda f: _v(f, "final_stability") > _v(f, "leiden_stability"),
    "гибрид устойчивее Leiden к удалению рёбер": lambda f: _v(f, "pert_hybrid") > _v(f, "pert_leiden"),
    # другие входы
    "без уровня трат — тот же итог; косинус корзин и районы — другое разбиение": lambda f: (
        _v(f, "var_no_level_winner") == FG.mlabel("hybrid")
        and _v(f, "var_no_level_ari") > 0.95
        and _v(f, "var_no_level_ari_same") > 0.9
        and max(_v(f, "var_graph_basket_cos_ari"), _v(f, "var_graph_basket_cos_ari_same")) < 0.5
        and max(_v(f, "var_nodes_separate_ari"), _v(f, "var_nodes_separate_ari_same")) < 0.5
    ),
    "при усечённых хвостах разбиение гибрида почти то же, а победитель другой": lambda f: (
        _v(f, "var_x_clipped_ari_same") > 0.9 and _v(f, "var_x_clipped_ari") < 0.5
    ),
    # типы и внешняя проверка
    "все типы итога устойчивы по Хеннигу": lambda f: _v(f, "final_jaccard_min") >= 0.75,
    "у самой объединяемой пары типов больше половины внешних рёбер ведёт друг в друга": lambda f: (
        _v(f, "avu_pair_value") > 0.5
    ),
    "типы не повторяют регионы (AMI < 0,1)": lambda f: _v(f, "ami_region") < 0.1,
    "внешняя проверка: все проверки итога значимы": lambda f: _v(f, "val_n_sig") == _v(f, "val_n_tests"),
    "показатели не связаны с зарплатой": lambda f: _v(f, "val_wage_rho_max") <= 0.5,
    "все ожидаемые знаки подтвердились": lambda f: _v(f, "val_signs_ok") == _v(f, "val_signs_n"),
    # индексы
    "HDBSCAN не дал ни одного K из сетки": lambda f: _v(f, "cands_hdbscan") == 0,
    "признаковые индексы тянут к меньшему K, чем графовые": lambda f: (
        _v(f, "kbest_features") < _v(f, "kbest_graph")
    ),
    "z AVU отрицательна почти у всех разбиений с K ≥ 4": lambda f: _v(f, "avu_neg_share") > 0.9,
    "AVU не определена у всех кандидатов с K = 3": lambda f: _v(f, "avu_undefined_k3") == _v(f, "n_k3"),
    # синтетика
    "в синтетике при сигнале в обоих источниках лучший средний ARI почти всегда у метода, видящего оба": (
        lambda f: _v(f, "syn_both_won") >= 0.9 * _v(f, "syn_both_cells")
    ),
    "в синтетике при сигнале в обоих источниках чаще всего лучшая — базовая линия, а не гибрид": lambda f: (
        _v(f, "syn_joint_won") > _v(f, "syn_both_hybrid_best")
        and 2 * _v(f, "syn_joint_won") >= _v(f, "syn_both_won")
    ),
    "в синтетике базовая линия сильнее гибрида, гибрид — KEFRiN при сильных сигналах": lambda f: (
        _v(f, "syn_strong_kmeans_joint") > _v(f, "syn_strong_hybrid") > _v(f, "syn_strong_shalileh_mirkin")
    ),
    "гибрид при шумовом графе теряет сигнал признаков": lambda f: (
        _v(f, "syn_g0_kmeans") > _v(f, "syn_g0_hybrid") + 0.2
        and _v(f, "sbm7_shalileh_mirkin") > _v(f, "sbm7_hybrid") + 0.2
    ),
    "KEFRiN при шумовом графе сохраняет признаки, но не лучше методов по X (ничья по интервалам)": lambda f: (
        _v(f, "syn_g0_shalileh_mirkin") > _v(f, "syn_g0_hybrid") + 0.2
        and _v(f, "syn_g0_shalileh_mirkin") > _v(f, "syn_g0_kmeans_joint")
        and _v(f, "syn_noisy_live") > 0
        and _v(f, "syn_noisy_x_tie") == _v(f, "syn_noisy_live")
    ),
    "порог «победителя нет» лежит в разрыве лучших средних ARI": lambda f: (
        _v(f, "syn_none_best_max") < _v(f, "syn_min_ari")
        and _v(f, "syn_live_best_min") > 2 * _v(f, "syn_min_ari")
    ),
    "без сигнала в признаках методы по X бессильны": lambda f: _v(f, "syn_nofeat_feat_max") < 0.1,
    "при сигнале только в графе методы на двух источниках в ничьей с лучшим": lambda f: (
        _v(f, "syn_f0_both_tie") >= 1
    ),
    "блочный граф: Leiden почти точен при малой доле внешних рёбер, при 0,7 и выше граф шумовой": lambda f: (
        _v(f, "sbm_low_leiden") > 0.9 and _v(f, "sbm_high_graph_max") < 0.05
    ),
    "блочный граф 0,5: у гибрида лучший средний ARI больше чем в половине ячеек": lambda f: (
        2 * _v(f, "sbm_mid_hybrid_best") > _v(f, "sbm_mid_n")
    ),
    "блочный граф 0,5: гибрид по среднему выше Leiden, KEFRiN и K-means": lambda f: (
        _v(f, "sbm_mid_hybrid")
        > max(_v(f, "sbm_mid_leiden"), _v(f, "sbm_mid_kefrin"), _v(f, "sbm_mid_kmeans"))
    ),
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


CRIT_LABELS: dict[str, str] = {
    "quality_features": "качеству в X",
    "quality_graph": "качеству на G",
    "stability": "устойчивости",
    "interpretability": "объяснимости",
}


def _tradeoff(lw: pd.DataFrame, fin: str, cands: pd.DataFrame, cp: ClusterParams) -> str:
    """Итог против каждого кандидата фронта (второй уровень среди всех семейств): по каким критериям лучше
    и хуже сверх допуска ничьей."""
    front = [x for x in lw.index if bool(lw.loc[x, "on_front"]) and x != fin]
    head = "" if fin in lw.index and bool(lw.loc[fin, "on_front"]) else "гибрид не на фронте; "
    parts = []
    for x in front:
        better, worse = [], []
        for cr in CRITERIA:
            d = float(lw.loc[fin, f"crit_{cr}"] - lw.loc[x, f"crit_{cr}"])
            if d > cp.tie[cr]:
                better.append(CRIT_LABELS[cr])
            elif d < -cp.tie[cr]:
                worse.append(CRIT_LABELS[cr])
        parts.append(
            f"против «{_cand_label(cands, x)}» гибрид лучше по {', '.join(better) or 'ни одному критерию'}, "
            f"хуже — по {', '.join(worse) or 'ни одному'}"
        )
    return head + ("; ".join(parts) or "других кандидатов на фронте нет")


def _cand_label(cands: pd.DataFrame, cand: str) -> str:
    return f"{FG.mlabel(cands.loc[cand, 'method'])}, K = {int(cands.loc[cand, 'k'])}"


_COUNT_GEN = {2: "двух", 3: "трёх", 4: "четырёх", 5: "пяти"}


def _borda_outcome(ranks: pd.Series, stability: pd.Series, winner, label) -> tuple[str, int, int]:
    """Фраза об исходе Борда на фронте, число равных лучших и согласованность победителя с цепочкой равенств.

    ``ranks`` — сумма мест кандидата (меньше — лучше), ``stability`` — устойчивость кандидатов (первое звено
    цепочки ``selection.tie_break``), ``winner`` — победитель из ``decide``. Если суммы мест лучших равны,
    фраза называет ничью и числа устойчивости; согласованность (1/0) — победитель среди лучших и, при ничьей,
    не менее устойчив, чем остальные равные."""
    top = ranks.min()
    tied = [x for x in ranks.index if ranks[x] == top]
    if len(tied) == 1:
        return f"Её победитель — {label(winner)}", 1, int(winner == tied[0])
    st = stability.reindex(tied).astype(float)
    consistent = int(winner in tied and st[winner] >= st.max())
    by_stab = consistent and all(st[winner] > st[x] for x in tied if x != winner)
    head = (
        f"Ничья: у {_COUNT_GEN.get(len(tied), str(len(tied)))} лучших поровну ({style.fmt_num(top)}), "
        "победителя решает цепочка равенств"
    )
    if by_stab:
        order = [winner, *sorted((x for x in tied if x != winner), key=lambda x: -st[x])]
        vals = "; ".join(f"{label(x)} — {style.fmt_num(st[x], 3)}" for x in order)
        return f"{head} — выше устойчивость ({vals}): {label(winner)}", len(tied), consistent
    return f"{head} (устойчивость, затем простота и K): {label(winner)}", len(tied), consistent


def _cand_quoted(cands: pd.DataFrame, cand: str) -> str:
    """«Leiden» с K = 3 — название метода в кавычках, K отдельно (как в соседних фразах отчёта)."""
    return f"«{FG.mlabel(cands.loc[cand, 'method'])}» с K = {int(cands.loc[cand, 'k'])}"


def _freq_text(winners: pd.Series, cands: pd.DataFrame, total: int) -> str:
    """«Гибрид, K = 4 — 3 из 5; Спектральная, K = 4 — 2 из 5» по убыванию частоты."""
    cnt = winners.value_counts()
    return "; ".join(f"{_cand_label(cands, w)} — {n} из {total}" for w, n in cnt.items())


def _syn_counts(s: pd.DataFrame) -> str:
    """Все ячейки синтетики одинаково: у каждого метода — сколько раз лучший средний ARI, сколько раз явный
    победитель, сколько раз в наборе ничьей; «нет» — отдельно."""
    methods = [m for m in FG.METHOD_LABELS if m in s.columns and m not in ("tie", "none")]
    sets = s["winner_set"].fillna("").astype(str).str.split(",")
    live = s["winner"] != "none"
    parts = []
    for m in methods:
        best = int(((s["best"] == m) & live).sum())
        clear = int((s["winner"] == m).sum())
        tie = int(((s["winner"] == "tie") & sets.map(lambda x, m=m: m in x)).sum())
        if best or clear or tie:
            parts.append((best, clear, tie, m))
    parts.sort(key=lambda p: (-p[0], -p[1], -p[2]))
    return "; ".join(f"{FG.mlabel(m)} — {b}, {c} и {t}" for b, c, t, m in parts)


def table_seed(o: Out) -> str:
    """Частота победителей по seed: проверка × уровень × правило (основное и с допуском качества)."""
    sr = o.seed_runs
    n = sr["seed"].nunique()
    rows = []
    for check in [c for c in CHECK_LABELS if c != "all_eligible"]:
        cells = [CHECK_LABELS[check]]
        for rule in ("prereg", "tolerance"):
            for scope in ("eligible", "all"):
                sel = (sr["rule"] == rule) & (sr["scope"] == scope) & (sr["kind"] == "check")
                m = sr.loc[sel & (sr["check"] == check)]
                cells.append(_freq_text(m["winner"], o.cands, n) if len(m) else style.NA_TEXT)
        rows.append(cells)
    for scope, lab in (
        ("all", "24 порядка на обоих уровнях"),
        ("all_level2", "24 порядка только при выборе метода"),
    ):
        cells = [lab]
        for rule in ("prereg", "tolerance"):
            el = sr.loc[(sr["rule"] == rule) & (sr["kind"] == "order") & (sr["scope"] == "eligible")]
            od = sr.loc[(sr["rule"] == rule) & (sr["kind"] == "order") & (sr["scope"] == scope)]
            cells += [_freq_text(el["winner"], o.cands, len(el)), _freq_text(od["winner"], o.cands, len(od))]
        rows.append(cells)
    cols = [
        "Проверка",
        "Предрегистрация: итог",
        "Предрегистрация: все семейства",
        "С допуском качества: итог",
        "С допуском качества: все семейства",
    ]
    return _md(pd.DataFrame(rows, columns=cols))


def winners_of(o: Out) -> dict[str, str]:
    return o.winners


def table_sensitivity(o: Out) -> str:
    """Проверки на двух уровнях: среди методов с двумя источниками (основное правило) и среди всех
    семейств."""
    ch = o.checks
    rows = []
    cell = lambda r: (  # noqa: E731
        f"{FG.mlabel(r['method'])}, K = {int(r['k'])}" + ("" if bool(r["same_as_main"]) else " **(другой)**")
    )
    for check in [c for c in CHECK_LABELS if c not in ("main", "all_eligible")]:
        el = ch.loc[(ch["scope"] == "eligible") & (ch["check"] == check)]
        al = ch.loc[(ch["scope"] == "all") & (ch["check"] == check)]
        if el.empty and al.empty:
            continue
        rows.append(
            [
                CHECK_LABELS.get(check, check),
                cell(el.iloc[0]) if len(el) else style.NA_TEXT,
                cell(al.iloc[0]) if len(al) else style.NA_TEXT,
            ]
        )
    lex = []
    for scope in ("eligible", "all", "all_level2"):
        od = o.orders.loc[o.orders["scope"] == scope]
        cnt = od.groupby("winner").size().sort_values(ascending=False)
        lex.append(
            "; ".join(
                f"{FG.mlabel(o.cands.loc[w, 'method'])}, K = {int(o.cands.loc[w, 'k'])} — {n}"
                for w, n in cnt.items()
            )
        )
    rows.append(["24 лексикографических порядка (на обоих уровнях)", lex[0], lex[1]])
    rows.append(["24 порядка только при выборе метода (K — по основному правилу)", style.NA_TEXT, lex[2]])
    b2 = ch.loc[(ch["scope"] == "all_level2") & (ch["check"] == "borda")].iloc[0]
    rows.append(["Борда только при выборе метода", style.NA_TEXT, cell(b2)])
    cols = ["Проверка", "Среди методов с двумя источниками (итог)", "Среди всех семейств"]
    return _md(pd.DataFrame(rows, columns=cols))


def table_grid(o: Out) -> str:
    g = o.grid
    c = o.cands
    lab = lambda w: f"{FG.mlabel(c.loc[w, 'method'])}, K = {int(c.loc[w, 'k'])}"  # noqa: E731
    rows = []
    for (smin, smax), d in g.groupby(["min_share", "max_share"]):
        st = d.loc[d["chain"] == "strict"].iloc[0]
        tl = d.loc[d["chain"] == "tolerance"].iloc[0]
        rows.append(
            [
                style.fmt_pct(smin, 1),
                style.fmt_pct(smax, 0),
                str(int(st["n_feasible"])),
                str(st["hybrid_feasible_k"]).replace(",", ", "),
                lab(st["winner_eligible"]),
                lab(st["winner_all"]),
                lab(tl["winner_eligible"]),
                lab(tl["winner_all"]),
            ]
        )
    cols = [
        "Наименьший тип",
        "Крупнейший тип",
        "Допустимых",
        "Допустимые K гибрида",
        "Итог (два источника)",
        "Все семейства",
        "Итог, цепочка с допуском",
        "Все семейства, цепочка с допуском",
    ]
    return _md(pd.DataFrame(rows, columns=cols))


def table_resolution(o: Out) -> str:
    rl = o.reslim
    rows = [
        [
            f"{FG.mlabel(r['method'])}, K = {int(r['k'])}",
            _n(r["gamma"], 2),
            _n(r["threshold"], 0),
            _n(r["min_inner_weight"], 0),
            str(int(r["n_below"])),
            str(int(r["n_disconnected"])),
        ]
        for _, r in rl.iterrows()
    ]
    cols = [
        "Разбиение",
        "γ",
        "Порог √(2m/γ)",
        "Наименьший вес внутри сообщества",
        "Сообществ ниже порога",
        "Несвязных сообществ",
    ]
    return _md(pd.DataFrame(rows, columns=cols))


def table_kefrin(o: Out) -> str:
    kf = o.kef
    rows = []
    for (feat, x), d in kf.groupby(["features", "xi_over_rho"], sort=False):
        feas = d.loc[d["feasible"].astype(bool), "k"].astype(int).tolist()
        k4 = d.loc[d["k"] == 4].iloc[0]
        rows.append(
            [
                "общий X (robust)" if feat == "inputs" else "z-оценка, как у авторов",
                _n(x, 2),
                style.fmt_pct(d["graph_share"].iloc[0], 0),
                ", ".join(map(str, feas)) or "нет",
                style.fmt_pct(k4["min_share"], 1),
                style.fmt_pct(k4["max_share"], 0),
                _n(k4["ari_final"]),
                _n(d["ari_final"].max()),
            ]
        )
    cols = [
        "Признаки",
        "ξ/ρ",
        "Доля сети в разбросе",
        "Допустимые K",
        "K = 4: наименьший тип",
        "K = 4: крупнейший тип",
        "K = 4: ARI с итогом",
        "Наибольший ARI с итогом (K = 3–12)",
    ]
    return _md(pd.DataFrame(rows, columns=cols))


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


def _syn_verdict(r: pd.Series) -> str:
    """Итог ячейки синтетики: метод; «ничья: …» — все, чей интервал перекрывает интервал лучшего; «нет»."""
    if r["winner"] == "none":
        return "нет"
    if r["winner"] == "tie":
        return "ничья: " + ", ".join(FG.mlabel(m) for m in str(r["winner_set"]).split(","))
    return FG.mlabel(r["winner"])


def _syn_table(s: pd.DataFrame, first: str) -> str:
    """Все ячейки одного генератора одинаково: средний ARI каждого метода, лучший с 95% интервалом, итог."""
    meths = [m for m in FG.METHOD_LABELS if m in s.columns and s[m].notna().any()]
    rows = []
    for _, r in s.iterrows():
        b = r["best"]
        best = f"{FG.mlabel(b)} {_n(r[b])} [{_n(r[b + '_lo'])}; {_n(r[b + '_hi'])}]"
        rows.append(
            [
                _n(r["graph_signal"], 1),
                _n(r["feature_signal"], 1),
                *[_n(r[m]) for m in meths],
                best,
                _syn_verdict(r),
            ]
        )
    cols = [
        first,
        "Сигнал в X",
        *[FG.mlabel(m) for m in meths],
        "Лучший средний ARI [95% интервал]",
        "Итог ячейки",
    ]
    return _md(pd.DataFrame(rows, columns=cols), right=0)


def table_synthetic_sbm(o: Out) -> str:
    return _syn_table(o.syn.loc[o.syn["design"] == "sbm"], "Доля рёбер между группами")


def table_synthetic_knn(o: Out) -> str:
    return _syn_table(o.syn.loc[o.syn["design"] == "knn"], "Сигнал в G")


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
            f"сигнал только в признаках, хвосты усечены (синтетика: при сигнале в графе 0 лучший метод "
            f"по X — ARI {t('syn_nograph_best_feat')}, рис. 1, табл. 1)",
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
            f"все сообщества связны на G ({t('rl_disconnected')} несвязных у {t('rl_n_cands')} разбиений, "
            f"раздел 5); допустимых K — {t('feas_leiden')} из {t('cands_leiden')} (табл. 2)",
            f"устойчивость к удалению рёбер {t('pert_leiden')} против {t('pert_hybrid')} у гибрида (табл. 2)",
            f"сигнал только в графе; явная победа метода только по графу в синтетике — "
            f"{t('syn_graph_only_clear_text')}; на kNN-графе без сигнала в X — ничья с гибридом и базовой "
            f"линией ({t('syn_f0_four_tie')} из {t('syn_f0_n')} ячеек, табл. 1)",
        ),
        "louvain": (
            "быстрый",
            f"достиг {t('louvain_n_reached')} из {t('n_grid')} значений K (пропуски: {t('louvain_missing')})",
            "быстрый первый взгляд на сообщества",
        ),
        "spectral": (
            f"устойчив к бутстрапу и рёбрам ({t('pert_spectral')}; табл. 2)",
            "видит только граф: признаки места не участвуют",
            f"граф связен и информативен; в синтетике — в ничьей с гибридом (блочный граф с долей внешних "
            f"рёбер 0,5: {t('sbm_mid_tie_spectral')} из {t('sbm_mid_n')} ячеек, табл. 1а; kNN-граф без "
            f"сигнала в X: {t('syn_f0_four_tie')} из {t('syn_f0_n')}, табл. 1)",
        ),
        "shalileh_mirkin": (
            "одна целевая функция по X и G; признаки и связи восстанавливаются центрами (раздел 2)",
            f"на X с хвостами все K недопустимы (мелкий тип {t('small_shalileh_mirkin_min')}–"
            f"{t('small_shalileh_mirkin_max')} узлов, табл. 3), в варианте статьи тоже (табл. 7а); при "
            f"сильных сигналах в синтетике слабее гибрида ({t('syn_strong_shalileh_mirkin')} против "
            f"{t('syn_strong_hybrid')}, табл. 1)",
            f"граф шумовой, признаки информативны, а нужен метод на двух источниках: сигнал признаков "
            f"сохраняет (граф без сигнала — ARI {t('syn_g0_shalileh_mirkin')} против {t('syn_g0_hybrid')} "
            f"у гибрида и {t('syn_g0_kmeans')} у K-means, табл. 1; SBM с долей внешних рёбер 0,7 — "
            f"{t('sbm7_shalileh_mirkin')} против {t('sbm7_hybrid')} и {t('sbm7_kmeans')}, табл. 1а); "
            f"но методы по X (K-means, гауссова смесь) не хуже: ничья по интервалам в "
            f"{t('syn_noisy_x_tie')} из {t('syn_noisy_live')} ячеек шумового графа, где лучший средний "
            f"ARI не ниже {t('syn_min_ari')}; в остальных {t('syn_noisy_none')} из {t('syn_noisy_n')} "
            f"победителя нет",
        ),
        "hybrid": (
            f"устойчив ({t('final_stability')}), все типы устойчивы по Хеннигу (табл. 9)",
            f"при α = {t('alpha')} почти совпадает со спектральной (ARI {t('ari_hybrid_spectral')}, рис. 4); "
            f"при шумовом графе теряет сигнал признаков: ARI {t('syn_g0_hybrid')} против "
            f"{t('syn_g0_kmeans')} у K-means (табл. 1) и {t('sbm7_hybrid')} против {t('sbm7_kmeans')} "
            f"(SBM 0,7, табл. 1а); "
            f"при координатах графа уступает базовой линии ({t('syn_strong_hybrid')} против "
            f"{t('syn_strong_kmeans_joint')}, табл. 1); зависит от правила рёбер (ARI "
            f"{t('var_graph_basket_cos_ari_same')} с сетью косинуса) и состава узлов "
            f"({t('var_nodes_separate_ari_same')} с районами Москвы и Петербурга, табл. 8)",
            f"граф информативен и дан без координат (SBM с долей внешних рёбер 0,5: средний ARI "
            f"{t('sbm_mid_hybrid')} против {t('sbm_mid_spectral')} у спектральной и {t('sbm_mid_leiden')} "
            f"у Leiden; лучший по среднему в {t('sbm_mid_hybrid_best')} из {t('sbm_mid_n')} ячеек, явно — в "
            f"{t('sbm_mid_hybrid_clear')}, табл. 1а)",
        ),
        "kmeans_joint": (
            f"лучший средний ARI в синтетике, где граф построен из координат ({t('syn_joint_won')} из "
            f"{t('syn_both_cells')} ячеек с сигналом в обоих источниках, явно — {t('syn_joint_clear')}; "
            f"табл. 1)",
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
    "synthetic_knn": lambda o, cp, f: table_synthetic_knn(o),
    "methods_final": table_methods_final,
    "small": lambda o, cp, f: table_small(o),
    "grid": lambda o, cp, f: table_grid(o),
    "seed": lambda o, cp, f: table_seed(o),
    "resolution": lambda o, cp, f: table_resolution(o),
    "kefrin": lambda o, cp, f: table_kefrin(o),
}


# --- Рисунки и сборка ----------------------------------------------------------------------------


def make_figures(cfg: Config, cp: ClusterParams, o: Out) -> list[FG.FigureInfo]:
    fam = {m: cp.family_of(m) for m in cp.methods}
    metrics = list(cp.icvi_metrics)
    figs = [
        FG.fig_synthetic(
            o.dir, o.syn, fam, int(cp.synthetic["repeats"]), float(cp.synthetic.get("min_ari", 0.05))
        ),
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
