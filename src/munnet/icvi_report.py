"""Отчёт ``docs/icvi.md``: шаблон ``templates/icvi.md`` + таблицы и числа из ``outputs/evaluate``.

Числа текста — только факты ``{{icvi.ключ}}`` (``outputs/evaluate/report_facts.json``), таблицы — директивы
``<!-- table: имя -->``, рисунки — ``<!-- figure: I01 -->``. Формулировки, зависящие от данных, собираются
кодом по числам (факты вида «str»); утверждения, записанные в шаблоне прямо, перечислены в ``CLAIMS``
и проверяются по фактам: если хоть одно неверно, отчёт не пишется (``QCError``, код 3).
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

from munnet import icvi, style
from munnet.config import Config
from munnet.contracts import QCError
from munnet.eda import report as eda_report
from munnet.eda.base import Fact, make_fact, write_json
from munnet.icvi_figures import METRIC_LABELS, FigureInfo, make_all, method_label
from munnet.icvi_stage import METRICS, EvalParams

log = logging.getLogger(__name__)

TEMPLATE = Path(__file__).resolve().parent / "templates" / "icvi.md"
SECTION = "icvi"
_TABLE = re.compile(r"<!--\s*table:\s*([a-z_]+)\s*-->")
_FIGURE = re.compile(r"<!--\s*figure:\s*(I\d{2})\s*-->")
_NOTE = re.compile(r"[ \t]*<!--\s*note\s*:.*?-->[ \t]*\n?", re.S)
LINT_ALLOW = [r"CC BY(?:-SA)? \d\.\d", r"\b\d{1,2}\.\d{2}\.\d{4}\b", r"\$\$.*?\$\$", r"\$[^$]+\$"]
FEATURES = ("sw", "ch", "s_dbw")
BETTER_SIGN: dict[str, int] = {m: 1 if b == "max" else -1 for m, b in icvi.BETTER.items()}
GRAPH = ("avi", "avu", "mq")
VALUE_KIND: dict[str, str] = {
    "sw": "num3",
    "ch": "int",
    "s_dbw": "num3",
    "avi": "num3",
    "avu": "num3",
    "mq": "num3",
    "anui": "num3",
}


def _fact(facts: dict, key: str, value, kind: str) -> None:
    facts[f"{SECTION}.{key}"] = make_fact(f"{SECTION}.{key}", value, kind)


def _v(facts: Mapping[str, Fact], key: str) -> float:
    v = facts[f"{SECTION}.{key}"].value
    return float("nan") if v is None else float(v)


def cand_label(method: str, k: int) -> str:
    return f"{method_label(method)}, K = {int(k)}"


def between_share(ch: float, n: int, k: int) -> float:
    """Доля межгрупповой суммы квадратов в общей, B / T, по значению CH (Caliński, Harabasz, 1974).

    CH = [B / (K − 1)] / [W / (n − K)] и T = B + W, поэтому B / W = CH · (K − 1) / (n − K) и
    B / T = CH · (K − 1) / (CH · (K − 1) + n − K) — тождество, без повторного чтения X. Это R² разбиения:
    какая доля разброса признаков приходится на различия средних типов. n — объекты без шума; NaN, если CH
    не определён или n ≤ K.
    """
    if not np.isfinite(ch) or n <= k or k < 2:
        return float("nan")
    b = float(ch) * (k - 1)
    return b / (b + (n - k))


def sd_rel_error(n: int) -> float:
    """Относительная стандартная ошибка выборочного SD по n значениям нормального распределения:
    ≈ 1/√(2(n − 1)) (при n = 200 — 5,0%). Знаменатель z — SD случайного базиса, поэтому крупная z
    (z ≫ 1, ошибка среднего базиса пренебрежимо мала) колеблется относительно на столько же; при тяжёлом
    хвосте базиса — сильнее. NaN при n < 2."""
    return float(1 / np.sqrt(2 * (n - 1))) if n >= 2 else float("nan")


# --- Факты ---------------------------------------------------------------------------------------


def build_facts(out: Path, p: EvalParams, extra: Mapping) -> tuple[dict[str, Fact], dict]:
    """Все числа отчёта из файлов ``outputs/evaluate`` и заголовки рисунков (формулировки по числам)."""
    long = pd.read_csv(out / "icvi_long.csv")
    cands = pd.read_csv(out / "candidates.csv")
    tau_z = pd.read_csv(out / "agreement_z.csv").set_index("metric")
    tau_v = pd.read_csv(out / "agreement_value.csv").set_index("metric")
    tau_k = pd.read_csv(out / "agreement_within_k.csv").set_index("metric")
    tau_k_long = pd.read_csv(out / "agreement_within_k_long.csv")
    kf = pd.read_csv(out / "k_fair.csv")
    kd = pd.read_csv(out / "k_dependence.csv").set_index("metric")
    q = pd.read_csv(out / "space_ranks.csv")
    fu = pd.read_csv(out / "final_unifiability.csv") if (out / "final_unifiability.csv").exists() else None
    facts: dict[str, Fact] = {}
    feas = cands.loc[cands["feasible"].astype(bool)]
    _fact(facts, "n_candidates", len(cands), "int")
    _fact(facts, "n_feasible", len(feas), "int")
    _fact(facts, "n_methods", cands["method"].nunique(), "int")
    for key in ("n_nodes", "n_features", "n_edges"):
        _fact(facts, key, int(extra[key]), "int")
    _fact(facts, "permutations", p.permutations, "int")
    _fact(facts, "bootstrap", p.bootstrap, "int")
    _fact(facts, "ci_level", style.fmt_pct(p.ci_level, 0), "str")
    _fact(facts, "subsample", style.fmt_pct(p.subsample, 0), "str")
    _fact(facts, "ci_scale", float(np.sqrt(p.subsample / (1 - p.subsample))), "num1")
    _fact(facts, "s_dbw_density", p.options["s_dbw_density"], "str")
    rel = sd_rel_error(p.permutations)
    _fact(facts, "z_rel_err", rel, "pct")
    _fact(facts, "z_rel_err_text", style.fmt_pct(rel, 0), "str")

    # итог и победители
    fin = cands.loc[cands["is_final"].astype(bool)]
    has_final = len(fin) == 1
    if not has_final:
        raise QCError(
            f"evaluate: итоговых кандидатов {len(fin)}, нужен ровно один (флаг is_final этапа cluster)"
        )
    _fact(facts, "has_final", int(has_final), "int")
    if has_final:
        f = fin.iloc[0]
        _fact(facts, "final_label", cand_label(f["method"], f["k_eff"]), "str")
        _fact(facts, "final_k", int(f["k_eff"]), "int")
        w = long.loc[long["candidate"] == f["candidate"]].set_index("metric")
        wide_z = long.loc[long["candidate"].isin(feas["candidate"])].pivot(
            index="candidate", columns="metric", values="z"
        )
        for m in (*METRICS, "anui"):
            _fact(facts, f"final_{m}", w.loc[m, "value"], VALUE_KIND[m])
            _fact(facts, f"final_{m}_lo", w.loc[m, "ci_low"], VALUE_KIND[m])
            _fact(facts, f"final_{m}_hi", w.loc[m, "ci_high"], VALUE_KIND[m])
            _fact(facts, f"final_z_{m}", w.loc[m, "z"], "num1")
            if m in METRICS:
                col = wide_z[m].dropna()
                z = w.loc[m, "z"]
                rank = int((col > z).sum()) + 1 if np.isfinite(z) else None
                _fact(facts, f"final_rank_{m}", rank, "int")
                _fact(facts, f"final_rank_n_{m}", len(col), "int")
        # CH и силуэт итога: доля межгрупповой суммы квадратов в общей (R²) по CH, шум исключён
        n_ok = int(round(int(extra["n_nodes"]) * (1 - float(f["noise_share"]))))
        share = between_share(float(w.loc["ch", "value"]), n_ok, int(f["k_eff"]))
        _fact(facts, "final_between_share", share, "pct")
        _fact(facts, "final_between_share_text", style.fmt_pct(share, 0), "str")
        _fact(facts, "final_z_ch_round", w.loc["ch", "z"], "int")
    _fact(facts, "n_winners", int(cands["is_method_winner"].astype(bool).sum()), "int")

    # согласие метрик
    for i, a in enumerate(METRICS):
        for b in METRICS[i + 1 :]:
            _fact(facts, f"tau_{a}_{b}", tau_z.loc[a, b], "num2")
            _fact(facts, f"tauv_{a}_{b}", tau_v.loc[a, b], "num2")

    def mean_tau(t, pairs):
        vals = [t.loc[a, b] for a, b in pairs if np.isfinite(t.loc[a, b])]
        return float(np.mean(vals)) if vals else float("nan")

    pairs_f = [(a, b) for i, a in enumerate(FEATURES) for b in FEATURES[i + 1 :]]
    pairs_g = [(a, b) for i, a in enumerate(GRAPH) for b in GRAPH[i + 1 :]]
    pairs_x = [(a, b) for a in FEATURES for b in GRAPH]
    tf, tg, tx = mean_tau(tau_z, pairs_f), mean_tau(tau_z, pairs_g), mean_tau(tau_z, pairs_x)
    _fact(facts, "tau_within_features", tf, "num2")
    _fact(facts, "tau_within_graph", tg, "num2")
    _fact(facts, "tau_cross", tx, "num2")
    cross = {(a, b): tau_z.loc[a, b] for a, b in pairs_x if np.isfinite(tau_z.loc[a, b])}
    _fact(facts, "tau_cross_max", max(cross.values()) if cross else float("nan"), "num2")
    allp = {(a, b): tau_z.loc[a, b] for i, a in enumerate(METRICS) for b in METRICS[i + 1 :]}
    allp = {k: v for k, v in allp.items() if np.isfinite(v)}
    if allp:
        lo = min(allp, key=allp.get)
        _fact(
            facts,
            "tau_min_pair",
            f"{METRIC_LABELS[lo[0]].split(' ')[0]} и {METRIC_LABELS[lo[1]].split(' ')[0]}",
            "str",
        )
        _fact(facts, "tau_min", allp[lo], "num2")
    else:
        _fact(facts, "tau_min_pair", "—", "str")
        _fact(facts, "tau_min", float("nan"), "num2")

    # согласие внутри одного K: эффект K исключён
    for i, a in enumerate(METRICS):
        for b in METRICS[i + 1 :]:
            _fact(facts, f"tauk_{a}_{b}", tau_k.loc[a, b], "num2")
    _fact(facts, "tauk_n_strata", int(tau_k_long["k"].nunique()) if len(tau_k_long) else 0, "int")
    _fact(facts, "tauk_n_min", int(tau_k_long["n"].min()) if len(tau_k_long) else 0, "int")
    _fact(facts, "tauk_n_max", int(tau_k_long["n"].max()) if len(tau_k_long) else 0, "int")
    _fact(facts, "flip_text", _flip_text(tau_v, tau_z, tau_k, kd), "str")
    _fact(facts, "tauk_text", _tauk_text(tau_z, tau_k), "str")

    # сравнение разных K без смещения
    sel = kf.loc[kf["is_method_winner"].astype(bool) | kf["is_final"].astype(bool)]
    sel = sel.loc[sel["family"].isin(["graph", "attributed"]) | sel["is_final"].astype(bool)]
    for col, key in (("z_avi", "order_z_avi"), ("avi_adj", "order_avi_adj"), ("mq", "order_mq")):
        _fact(facts, key, _order_text(sel, col), "str")
    _fact(facts, "order_z_mq", _order_text(sel, "z_mq"), "str")
    if has_final:
        r = kf.loc[kf["is_final"].astype(bool)].iloc[0]
        _fact(facts, "final_avi_adj", r["avi_adj"], "num3")
        _fact(facts, "final_avi_base", r["avi_base"], "num3")
        _fact(facts, "final_rank_in_k_avi", int(r["rank_in_k_avi"]), "int")
        _fact(facts, "final_rank_in_k_mq", int(r["rank_in_k_mq"]), "int")
        _fact(facts, "final_n_in_k", int(r["n_in_k"]), "int")
    pf = kf.loc[kf["feasible"].astype(bool)]
    _fact(facts, "pct_avi_top", int((pf["pct_avi"] >= 1).sum()), "int")
    _fact(facts, "kfair_text", _kfair_text(sel), "str")
    _fact(facts, "kfair_differs", int(_order_text(sel, "z_avi") != _order_text(sel, "avi_adj")), "int")
    first = [
        str(sel.sort_values(col, ascending=False)["candidate"].iloc[0])
        for col in ("z_avi", "avi_adj", "mq", "z_mq")
        if sel[col].notna().any()
    ]
    fin_c = kf.loc[kf["is_final"].astype(bool), "candidate"]
    _fact(facts, "final_first_any", int(len(fin_c) == 1 and fin_c.iloc[0] in first), "int")
    feas_k = kf.loc[kf["feasible"].astype(bool)].groupby("k_eff").size()
    _fact(facts, "feas_in_k_min", int(feas_k.min()) if len(feas_k) else 0, "int")
    _fact(facts, "feas_in_k_max", int(feas_k.max()) if len(feas_k) else 0, "int")
    sd_avi = long.loc[long["metric"] == "avi"].merge(cands[["candidate", "k_eff"]], on="candidate")
    rho_sd = (
        sd_avi[["baseline_sd", "k_eff"]].corr(method="spearman").iloc[0, 1] if len(sd_avi) >= 3 else np.nan
    )
    _fact(facts, "rho_sd_k_avi", rho_sd, "num2")
    _fact(facts, "flips_consistent", int(_flips_consistent(tau_v, tau_z, tau_k)), "int")

    # AVU: вырожденный базис и знак z
    k_small = feas.loc[feas["k_eff"] <= 3, "candidate"]
    avu = long.loc[long["metric"] == "avu"].merge(cands[["candidate", "k_eff", "feasible"]], on="candidate")
    k3 = avu.loc[avu["k_eff"] == 3, "value"]
    _fact(facts, "n_k3", len(k3), "int")
    _fact(facts, "avu_k3_maxdev", float(np.max(np.abs(k3 - 2 / 3))) if len(k3) else 0.0, "num3")
    _fact(facts, "n_k_small", len(k_small), "int")
    av_f = avu.loc[avu["feasible"].astype(bool)]
    zf = av_f["z"].dropna()
    _fact(facts, "avu_z_nan", int(av_f["z"].isna().sum()), "int")
    _fact(facts, "avu_z_finite", len(zf), "int")
    _fact(facts, "avu_z_neg", int((zf < 0).sum()), "int")
    _fact(facts, "avu_z_neg_share", float((zf < 0).mean()) if len(zf) else float("nan"), "pct")
    _fact(facts, "avu_z_median", float(zf.median()) if len(zf) else float("nan"), "num1")

    # S_Dbw: вырождение
    sd = long.loc[long["metric"] == "s_dbw"].merge(
        cands[["candidate", "feasible", "dens_bw"]], on="candidate"
    )
    sd_f = sd.loc[sd["feasible"].astype(bool)]
    _fact(facts, "sdbw_dens0", int((sd_f["dens_bw"] == 0).sum()), "int")
    _fact(
        facts, "sdbw_dens0_share", float((sd_f["dens_bw"] == 0).mean()) if len(sd_f) else float("nan"), "pct"
    )
    _fact(facts, "sdbw_nan", int(sd_f["value"].isna().sum()), "int")
    _fact(facts, "sdbw_undef_max", int(sd_f["n_undefined"].max()) if len(sd_f) else 0, "int")
    _fact(facts, "sdbw_undef_median", float(sd_f["n_undefined"].median()) if len(sd_f) else 0.0, "int")
    _fact(facts, "stdev_median", float(cands["stdev"].median()), "num2")
    _fact(facts, "nn_median", float(extra["nn_median"]), "num2")
    _fact(facts, "stdev_over_nn", float(cands["stdev"].median() / extra["nn_median"]), "num2")
    cf = cands.loc[cands["feasible"].astype(bool)]
    undefined = cf.loc[cf["candidate"].isin(sd_f.loc[sd_f["value"].isna(), "candidate"])]
    defined = cf.loc[~cf["candidate"].isin(undefined["candidate"])]
    _fact(facts, "sdbw_nan_kmin", int(undefined["k_eff"].min()) if len(undefined) else None, "int")
    _fact(facts, "stdev_median_nan", float(undefined["stdev"].median()) if len(undefined) else None, "num2")
    _fact(facts, "stdev_median_ok", float(defined["stdev"].median()) if len(defined) else None, "num2")
    _fact(facts, "stdev_below_nn", int((cf["stdev"] < extra["nn_median"]).sum()), "int")
    rho_sk = cf[["stdev", "k_eff"]].corr(method="spearman").iloc[0, 1] if len(cf) >= 3 else float("nan")
    _fact(facts, "rho_stdev_k", rho_sk, "num2")

    # зависимость от K
    for m in METRICS:
        _fact(facts, f"rho_value_k_{m}", kd.loc[m, "rho_value_k"], "num2")
        _fact(facts, f"rho_z_k_{m}", kd.loc[m, "rho_z_k"], "num2")

    # свой и чужой ранг
    qf = q.loc[q["feasible"].astype(bool)]
    fam = qf.groupby("family")[["q_features", "q_graph"]].median()
    for fname in fam.index:
        _fact(facts, f"q_{fname}_features", fam.loc[fname, "q_features"], "num2")
        _fact(facts, f"q_{fname}_graph", fam.loc[fname, "q_graph"], "num2")
    plays = (
        "features" in fam.index
        and "graph" in fam.index
        and fam.loc["features", "q_features"] > fam.loc["graph", "q_features"]
        and fam.loc["graph", "q_graph"] > fam.loc["features", "q_graph"]
    )
    _fact(facts, "home_advantage", int(plays), "int")
    rho_fg = (
        qf[["q_features", "q_graph"]].corr(method="spearman").iloc[0, 1] if len(qf) >= 3 else float("nan")
    )
    _fact(facts, "rho_spaces", rho_fg, "num2")

    # итоговое разбиение: U_kl
    if fu is not None and len(fu):
        types = fu["type"].to_numpy()
        mat = fu[[f"type_{t}" for t in types]].to_numpy(dtype=float)
        i, j = np.unravel_index(np.nanargmax(mat), mat.shape)
        _fact(facts, "u_max", mat[i, j], "num2")
        _fact(facts, "u_max_pair", f"{types[i]} и {types[j]}", "str")
        _fact(facts, "u_median", float(np.nanmedian(mat)), "num2")
        _fact(facts, "iso_min", float(fu["isolability"].min()), "num2")
        _fact(facts, "iso_min_type", int(fu.loc[fu["isolability"].idxmin(), "type"]), "int")
        _fact(facts, "iso_max", float(fu["isolability"].max()), "num2")
        _fact(facts, "iso_max_type", int(fu.loc[fu["isolability"].idxmax(), "type"]), "int")

    # формулировки по числам (истинны по построению)
    titles = {
        "better": dict(icvi.BETTER),
        "title_curves": (
            f"Сырые {_names(_k_dependent(facts))} заметно зависят от K (|ρ Спирмена| ≥ 0,5): разные K "
            "по сырым значениям несравнимы"
            if _k_dependent(facts)
            else "Индексы качества по числу типов K: значение, интервал и случайный базис"
        ),
        "title_z": (
            "z-оценки тоже зависят от K: сетевые растут с K, признаковые падают — пространства тянут K "
            "в разные стороны"
            if _z_opposed(facts)
            else "z-оценки индексов по числу типов K: насколько разбиение лучше случайных меток"
        ),
        "title_agreement": (
            "AVI и модульность ранжируют кандидатов согласнее, чем любая пара метрик признаков и сети"
            if np.isfinite(_v(facts, "tau_avi_mq")) and _v(facts, "tau_avi_mq") > _v(facts, "tau_cross_max")
            else "Согласие шести индексов при ранжировании кандидатов"
        ),
        "title_spaces": (
            "Метод выигрывает в пространстве, которое оптимизировал: признаки и связи тянут в разные стороны"
            if plays
            else "Качество кандидатов в признаках и в сети"
        ),
        "title_unif": (
            f"Сильнее всего «сливаются» типы {facts[f'{SECTION}.u_max_pair'].value}: "
            f"объединяемость {style.fmt_num(_v(facts, 'u_max'), 2)}"
            if f"{SECTION}.u_max" in facts
            else "Объединяемость типов итогового разбиения"
        ),
    }
    _fact(facts, "agreement_text", _agreement_text(facts), "str")
    _fact(facts, "spaces_text", _spaces_text(facts, plays), "str")
    _fact(facts, "families_text", _families_text(cands), "str")
    neg, fin_n = int((zf < 0).sum()), len(zf)
    _fact(
        facts,
        "avu_z_neg_text",
        f"у всех {fin_n} допустимых кандидатов с определённой z" if neg == fin_n else f"у {neg} из {fin_n}",
        "str",
    )
    _fact(facts, "k_text", _k_text(facts), "str")
    _fact(facts, "sdbw_text", _sdbw_text(facts), "str")
    _fact(facts, "sdbw_short", _sdbw_short(facts), "str")
    _fact(facts, "final_text", _final_text(facts), "str")
    zm = extra.get("z_match") or {"rel": float("nan"), "nan_mismatch": float("nan"), "n": 0}
    if zm["n"]:
        _fact(facts, "z_match_rel", zm["rel"], "num3")
        _fact(facts, "z_match_nan_mismatch", zm["nan_mismatch"], "int")
    _fact(facts, "z_match_text", _z_match_text(zm), "str")
    return facts, titles


def _agreement_text(facts) -> str:
    tg, tf, tx = _v(facts, "tau_within_graph"), _v(facts, "tau_within_features"), _v(facts, "tau_cross")
    if not all(np.isfinite([tg, tf, tx])):
        return "согласие не определено: слишком мало кандидатов с конечными z"
    if tx < min(tf, tg):
        return "метрики одного пространства согласуются между собой сильнее, чем метрики разных пространств"
    return "метрики разных пространств согласуются не слабее, чем метрики одного пространства"


def _k_dependent(facts) -> list[str]:
    """Индексы, сырые значения которых заметно связаны с K (|ρ Спирмена| ≥ 0,5), по убыванию |ρ|."""
    ms = [m for m in METRICS if abs(_v(facts, f"rho_value_k_{m}")) >= 0.5]
    return sorted(ms, key=lambda m: -abs(_v(facts, f"rho_value_k_{m}")))


def _pair(a: str, b: str) -> str:
    return f"{METRIC_LABELS[a].split(' ')[0]}–{METRIC_LABELS[b].split(' ')[0]}"


def _flip_text(tau_v, tau_z, tau_k, kd) -> str:
    """Пары, у которых τ по сырым значениям и по z разного знака, и почему: связь сырых значений с K."""
    kd = kd.set_index("metric") if "metric" in kd.columns else kd
    parts = []
    for i, a in enumerate(METRICS):
        for b in METRICS[i + 1 :]:
            tv, tz, tk = tau_v.loc[a, b], tau_z.loc[a, b], tau_k.loc[a, b]
            if not all(np.isfinite([tv, tz])) or np.sign(tv) == np.sign(tz) or min(abs(tv), abs(tz)) < 0.2:
                continue
            ra, rb = kd.loc[a, "rho_value_k"], kd.loc[b, "rho_value_k"]
            za, zb = kd.loc[a, "rho_z_k"], kd.loc[b, "rho_z_k"]
            parts.append(
                f"{_pair(a, b)}: τ по сырым значениям {style.fmt_num(tv, 2)}, по z {style.fmt_num(tz, 2)}, "
                f"внутри одного K {style.fmt_num(tk, 2)}; ρ с K у сырых значений (знак «больше — лучше») "
                f"{style.fmt_num(ra, 2)} "
                f"и {style.fmt_num(rb, 2)}, у z — {style.fmt_num(za, 2)} и {style.fmt_num(zb, 2)}"
            )
    return "; ".join(parts) if parts else "знак τ по сырым значениям и по z у всех пар одинаков"


def _flips_consistent(tau_v, tau_z, tau_k) -> bool:
    """У всех пар со сменой знака «сырые → z» знак τ внутри одного K совпадает со знаком по z."""
    for i, a in enumerate(METRICS):
        for b in METRICS[i + 1 :]:
            tv, tz, tk = tau_v.loc[a, b], tau_z.loc[a, b], tau_k.loc[a, b]
            if not all(np.isfinite([tv, tz])) or np.sign(tv) == np.sign(tz) or min(abs(tv), abs(tz)) < 0.2:
                continue
            if not np.isfinite(tk) or np.sign(tk) != np.sign(tz):
                return False
    return True


def _tauk_text(tau_z, tau_k) -> str:
    same = diff = 0
    for i, a in enumerate(METRICS):
        for b in METRICS[i + 1 :]:
            tz, tk = tau_z.loc[a, b], tau_k.loc[a, b]
            if np.isfinite(tz) and np.isfinite(tk) and max(abs(tz), abs(tk)) >= 0.2:
                same += int(np.sign(tz) == np.sign(tk))
                diff += int(np.sign(tz) != np.sign(tk))
    return f"среди пар с |τ| ≥ 0,2 знак τ по z и τ внутри одного K совпадает у {same}, расходится у {diff}"


def _order_text(sel: pd.DataFrame, col: str) -> str:
    d = sel.dropna(subset=[col]).sort_values(col, ascending=False)
    return " > ".join(cand_label(m, k) for m, k in zip(d["method"], d["k_eff"], strict=True))


def _kfair_text(sel: pd.DataFrame) -> str:
    """Меняет ли поправка на случайность порядок итога и победителей сетевых методов."""
    if len(sel) < 2:
        return "сравнивать не с кем"
    by_z = list(sel.sort_values("z_avi", ascending=False)["candidate"])
    by_adj = list(sel.sort_values("avi_adj", ascending=False)["candidate"])
    if by_z == by_adj:
        return "порядок по z AVI и по AVI с поправкой на случайность одинаков"
    return (
        "порядок по z AVI и по AVI с поправкой на случайность различается: z награждает большие K "
        "(разброс случайного базиса сужается с ростом K), поправка на случайность — нет"
    )


def _families_text(cands: pd.DataFrame) -> str:
    """Сколько допустимых кандидатов у каждого семейства и каких методов: семейства неравны."""
    from munnet.icvi_figures import _method_style

    labels, _, fam_labels = _method_style()
    f = cands.loc[cands["feasible"].astype(bool)]
    parts = []
    for fam, g in f.groupby("family"):
        methods = ", ".join(f"{labels.get(m, m)} — {n}" for m, n in g["method"].value_counts().items())
        parts.append(f"{fam_labels.get(fam, fam)} — {len(g)} ({methods})")
    return "; ".join(parts)


def _spaces_text(facts, plays: bool) -> str:
    if plays:
        return (
            "методы по признакам выше в признаковых индексах, методы по графу — в сетевых: каждая метрика "
            "подыгрывает методу, который оптимизировал её пространство"
        )
    return "явного преимущества «своего» пространства у семейств методов нет"


def _names(metrics) -> str:
    return ", ".join(METRIC_LABELS[m].split(" ")[0] for m in metrics)


def _weakened(facts) -> list[str]:
    """Индексы, у которых z-оценка заметно ослабляет связь с K (|ρ| меньше хотя бы на 0,1)."""
    return [m for m in METRICS if abs(_v(facts, f"rho_value_k_{m}")) > abs(_v(facts, f"rho_z_k_{m}")) + 0.1]


def _z_trend(facts, sign: int) -> list[str]:
    return [m for m in METRICS if sign * _v(facts, f"rho_z_k_{m}") >= 0.5]


def _z_opposed(facts) -> bool:
    """z сетевых индексов растут с K, признаковых — падают (|ρ| ≥ 0,5)."""
    up, down = set(_z_trend(facts, 1)), set(_z_trend(facts, -1))
    return bool(up & set(GRAPH)) and bool(down & set(FEATURES)) and not (up & set(FEATURES))


def _k_text(facts) -> str:
    weak = _weakened(facts)
    parts = [
        f"z-оценка ослабляет связь с K у {_names(weak)}" if weak else "z-оценка почти не ослабляет связь с K"
    ]
    up, down = _z_trend(facts, 1), _z_trend(facts, -1)
    if up or down:
        trend = []
        if up:
            trend.append(f"растут с K у {_names(up)}")
        if down:
            trend.append(f"падают у {_names(down)}")
        parts.append("но и сами z не нейтральны к K: они " + ", а ".join(trend))
    if _z_opposed(facts):
        parts.append("поэтому сетевое качество в правиле выбора тянет к большим K, признаковое — к малым")
    return "; ".join(parts)


def _z_match_text(zm) -> str:
    if not zm["n"]:
        return "сверка с z этапа `cluster` не проводилась: его таблицы кандидатов нет в выходах"
    if zm["nan_mismatch"] == 0 and zm["rel"] < 1e-4:
        return (
            "сверка с этапом `cluster`: пар «кандидат × индекс» с определённой z — "
            f"{style.fmt_num(zm['n'])}, все совпадают с точностью до округления в его таблице, "
            "неопределённые z — у тех же кандидатов"
        )
    return (
        f"сверка: z расходятся с z этапа `cluster` (наибольшее относительное расхождение "
        f"{style.fmt_num(zm['rel'], 4)}, несовпадений неопределённости — {int(zm['nan_mismatch'])})"
    )


def _sdbw_short(facts) -> str:
    if _v(facts, "sdbw_nan") == 0:
        return "S_Dbw определён у всех допустимых кандидатов"
    return (
        f"S_Dbw не определён у {facts[f'{SECTION}.sdbw_nan'].text} из {facts[f'{SECTION}.n_feasible'].text} "
        f"допустимых кандидатов, все — с K от {facts[f'{SECTION}.sdbw_nan_kmin'].text} (раздел 5)"
    )


def _final_text(facts) -> str:
    """Где итог хуже случайных меток (z < 0) и разделены ли его типы в признаках (интервал SW и ноль)."""
    neg = [m for m in METRICS if _v(facts, f"final_z_{m}") < 0]
    label = facts[f"{SECTION}.final_label"].text
    parts = [
        f"У итога ({label}) z отрицательна по {_names(neg)} — по этим индексам он хуже случайных меток "
        "тех же размеров"
        if neg
        else f"Итог ({label}) лучше случайных меток по всем шести индексам"
    ]
    lo, hi = _v(facts, "final_sw_lo"), _v(facts, "final_sw_hi")
    if lo <= 0 <= hi and _v(facts, "final_z_avi") > 3 and _v(facts, "final_z_mq") > 3:
        parts.append(
            f"интервал силуэта [{facts[f'{SECTION}.final_sw_lo'].text}; "
            f"{facts[f'{SECTION}.final_sw_hi'].text}] "
            "включает ноль: в признаках типы итога почти не разделены, а в сети изолированы (z AVI "
            f"{facts[f'{SECTION}.final_z_avi'].text}, z MQ {facts[f'{SECTION}.final_z_mq'].text})"
        )
    return "; ".join(parts)


def _sdbw_text(facts) -> str:
    n = _v(facts, "sdbw_nan")
    if n == 0:
        return (
            f"на всех допустимых кандидатах S_Dbw определён; радиус окрестности (медиана "
            f"{facts[f'{SECTION}.stdev_median'].text}) сравним с расстоянием до ближайшего соседа "
            f"({facts[f'{SECTION}.nn_median'].text})"
        )
    return (
        f"S_Dbw не определён у {facts[f'{SECTION}.sdbw_nan'].text} из {facts[f'{SECTION}.n_feasible'].text} "
        f"допустимых кандидатов — все с K от {facts[f'{SECTION}.sdbw_nan_kmin'].text}: радиус окрестности "
        f"убывает с K (медиана {facts[f'{SECTION}.stdev_median_nan'].text} у неопределённых против "
        f"{facts[f'{SECTION}.stdev_median_ok'].text} у остальных при медианном расстоянии до ближайшего "
        f"соседа {facts[f'{SECTION}.nn_median'].text}), и у какой-то пары типов в окрестностях обоих центров "
        "нет точек, а у середины есть — отношение плотностей бесконечно"
    )


# --- Проверяемые утверждения шаблона -------------------------------------------------------------

CLAIMS: dict[str, Callable[[Mapping[str, Fact]], bool]] = {
    "внутри одного K AVI и MQ тоже согласуются (τ > 0,5)": lambda f: _v(f, "tauk_avi_mq") > 0.5,
    "у пар со сменой знака «сырые → z» знак τ внутри одного K совпадает со знаком по z": lambda f: (
        _v(f, "flips_consistent") == 1
    ),
    "разброс случайного базиса AVI сужается с ростом K": lambda f: _v(f, "rho_sd_k_avi") < 0,
    "порядок по z AVI и по AVI с поправкой на случайность различается": lambda f: _v(f, "kfair_differs") == 1,
    "итог не первый по сетевым индексам ни в одном из четырёх порядков": lambda f: (
        _v(f, "final_first_any") == 0
    ),
    "процентиль AVI в базисе у большинства допустимых кандидатов максимален": lambda f: (
        _v(f, "pct_avi_top") > 0.5 * _v(f, "n_feasible")
    ),
    "при K = 3 AVU у всех кандидатов равна 2/3 (тождество)": lambda f: (
        _v(f, "n_k3") == 0 or _v(f, "avu_k3_maxdev") < 1e-9
    ),
    "AVI и MQ согласуются сильнее любой пары метрик из разных пространств": lambda f: (
        _v(f, "tau_avi_mq") > _v(f, "tau_cross_max")
    ),
    "у большинства допустимых кандидатов z AVU меньше нуля (хуже случайных меток)": lambda f: (
        _v(f, "avu_z_neg_share") > 0.5
    ),
    "радиус окрестности S_Dbw убывает с K": lambda f: _v(f, "rho_stdev_k") < 0,
    "CH итога намного выше случайного (z > 3), а интервал силуэта итога включает ноль": lambda f: (
        _v(f, "final_z_ch") > 3 and _v(f, "final_sw_lo") <= 0 <= _v(f, "final_sw_hi")
    ),
    "S_Dbw не определён у части допустимых кандидатов («ломается при больших K»)": lambda f: (
        _v(f, "sdbw_nan") > 0 and _v(f, "stdev_median_nan") < _v(f, "stdev_median_ok")
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
        raise QCError("отчёт evaluate: перестали быть верными утверждения шаблона: " + "; ".join(broken))


def load_facts(out: Path) -> dict[str, Fact]:
    js = json.loads((out / "report_facts.json").read_text(encoding="utf-8"))
    return {k: Fact(key=k, value=v["value"], kind=v["kind"], text=v["text"]) for k, v in js.items()}


# --- Таблицы -------------------------------------------------------------------------------------


def _fmt(v, kind: str) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return style.NA_TEXT
    return make_fact("x", v, kind).text


def _md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "|" + "|".join("---" if i == 0 else "---:" for i in range(len(cols))) + "|",
    ]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(x) for x in r.tolist()) + " |")
    return "\n".join(lines)


def _cell(r) -> str:
    kind = VALUE_KIND[r["metric"]]
    ci = f"[{_fmt(r['ci_low'], kind)}; {_fmt(r['ci_high'], kind)}]"
    return f"{_fmt(r['value'], kind)} {ci}, z {_fmt(r['z'], 'num1')}"


def _winners_table(out: Path, metrics: tuple[str, ...]) -> str:
    long = pd.read_csv(out / "icvi_long.csv")
    cands = pd.read_csv(out / "candidates.csv")
    sel = cands.loc[cands["is_method_winner"].astype(bool) | cands["is_final"].astype(bool)]
    sel = sel.sort_values(["is_final", "family", "method"], ascending=[False, True, True])
    rows = []
    for _, c in sel.iterrows():
        w = long.loc[long["candidate"] == c["candidate"]].set_index("metric")
        name = cand_label(c["method"], c["k_eff"]) + (" — итог" if c["is_final"] else "")
        rows.append([name, *[_cell(w.loc[m].to_dict() | {"metric": m}) for m in metrics]])
    header = ["Кандидат", *[f"{METRIC_LABELS[m]}" for m in metrics]]
    return _md(pd.DataFrame(rows, columns=header))


def table_winners_features(out: Path) -> str:
    return _winners_table(out, FEATURES)


def table_winners_graph(out: Path) -> str:
    return _winners_table(out, (*GRAPH, "anui"))


def table_agreement(out: Path) -> str:
    tau = pd.read_csv(out / "agreement_z.csv").set_index("metric")
    rows = [[METRIC_LABELS[a], *[_fmt(tau.loc[a, b], "num2") for b in METRICS]] for a in METRICS]
    return _md(pd.DataFrame(rows, columns=["τ по z", *[METRIC_LABELS[m].split(" ")[0] for m in METRICS]]))


def table_k(out: Path) -> str:
    kd = pd.read_csv(out / "k_dependence.csv").set_index("metric")
    rows = [
        [METRIC_LABELS[m], _fmt(kd.loc[m, "rho_value_k"], "num2"), _fmt(kd.loc[m, "rho_z_k"], "num2")]
        for m in METRICS
    ]
    return _md(
        pd.DataFrame(
            rows, columns=["Индекс", "ρ Спирмена: значение (знак «больше — лучше») и K", "ρ Спирмена: z и K"]
        )
    )


def table_unif(out: Path) -> str:
    path = out / "final_unifiability.csv"
    if not path.exists():
        return "Итоговое разбиение не отмечено."
    fu = pd.read_csv(path)
    types = fu["type"].tolist()
    rows = []
    for _, r in fu.iterrows():
        rows.append(
            [
                f"Тип {int(r['type'])}",
                _fmt(int(r["size"]), "int"),
                _fmt(r["isolability"], "num2"),
                *[_fmt(r[f"type_{t}"], "num2") for t in types],
            ]
        )
    return _md(pd.DataFrame(rows, columns=["Тип", "МО", "Изолируемость", *[f"U с типом {t}" for t in types]]))


def table_k_fair(out: Path) -> str:
    kf = pd.read_csv(out / "k_fair.csv")
    sel = kf.loc[kf["is_method_winner"].astype(bool) | kf["is_final"].astype(bool)]
    sel = sel.sort_values("avi_adj", ascending=False)
    rows = []
    for _, r in sel.iterrows():
        rows.append(
            [
                cand_label(r["method"], r["k_eff"]) + (" — итог" if r["is_final"] else ""),
                _fmt(r["avi"], "num3"),
                _fmt(r["avi_base"], "num3"),
                _fmt(r["avi_adj"], "num3"),
                _fmt(r["z_avi"], "num1"),
                f"{int(r['rank_in_k_avi'])} из {int(r['n_in_k'])}",
                _fmt(r["mq"], "num3"),
                _fmt(r["z_mq"], "num1"),
                f"{int(r['rank_in_k_mq'])} из {int(r['n_in_k'])}",
                _fmt(r["pct_avi"], "pct"),
            ]
        )
    cols = [
        "Кандидат",
        "AVI",
        "Базис AVI (≈ 1/K)",
        "AVI с поправкой",
        "z AVI",
        "Ранг AVI внутри K",
        "MQ",
        "z MQ",
        "Ранг MQ внутри K",
        "Процентиль AVI в базисе",
    ]
    return _md(pd.DataFrame(rows, columns=cols))


def table_agreement_k(out: Path) -> str:
    tau = pd.read_csv(out / "agreement_within_k.csv").set_index("metric")
    rows = [[METRIC_LABELS[a], *[_fmt(tau.loc[a, b], "num2") for b in METRICS]] for a in METRICS]
    header = ["τ внутри K", *[METRIC_LABELS[m].split(" ")[0] for m in METRICS]]
    return _md(pd.DataFrame(rows, columns=header))


TABLES: dict[str, Callable[[Path], str]] = {
    "k_fair": table_k_fair,
    "agreement_k": table_agreement_k,
    "winners_features": table_winners_features,
    "winners_graph": table_winners_graph,
    "agreement": table_agreement,
    "k": table_k,
    "unif": table_unif,
}


# --- Сборка --------------------------------------------------------------------------------------


def render(template: str, facts: Mapping[str, Fact], out: Path, figs: list[FigureInfo], img_rel: str) -> str:
    text = _NOTE.sub("", template)
    text = eda_report.fill(text, facts)
    text = _TABLE.sub(lambda m: TABLES[m.group(1)](out), text)
    by_id = {f.fid: f for f in figs}

    def fig_md(m: re.Match) -> str:
        f = by_id.get(m.group(1))
        if f is None:
            return ""
        n = int(f.fid[1:])
        # Якорь «fig-N» с тем же N, что в подписи: на него ведут ссылки «рис. N» из текста.
        return (
            f"{eda_report.anchor_tag(eda_report.fig_anchor(f.fid))}"
            f"![{f.alt}]({img_rel}/{f.png.name})\n\n*Рисунок {n}. {f.title}. {f.subtitle}. Источник: "
            f"{style.SOURCE_SBER}.* Данные: `outputs/evaluate/figures/{f.data_csv.name}`."
        )

    text = _FIGURE.sub(fig_md, text)
    text = eda_report.unwrap_paragraphs(text)
    return eda_report.nbsp_markdown(text)


def write_all(cfg: Config, p: EvalParams, out: Path, extra: Mapping) -> Path:
    """Факты, рисунки, проверка утверждений и ``docs/icvi.md`` (отчёт не пишется, если проверка не прошла)."""
    facts, titles = build_facts(out, p, extra)
    write_json(
        {k: {"value": f.value, "kind": f.kind, "text": f.text} for k, f in sorted(facts.items())},
        out / "report_facts.json",
    )
    long = pd.read_csv(out / "icvi_long.csv")
    cands = pd.read_csv(out / "candidates.csv")
    tau_z = pd.read_csv(out / "agreement_z.csv").set_index("metric")
    tau_v = pd.read_csv(out / "agreement_value.csv").set_index("metric")
    q = pd.read_csv(out / "space_ranks.csv")
    fu_path = out / "final_unifiability.csv"
    fu = pd.read_csv(fu_path) if fu_path.exists() else pd.DataFrame()
    figs = make_all(out, long, cands, tau_z, tau_v, fu, q, titles)
    for f in figs:
        _fact(facts, f"fig_{f.fid}_title", f.title, "str")
    check_claims(facts)
    report, images = Path(p.report), Path(p.report_images)
    try:
        img_rel = Path(os.path.relpath(images, report.parent)).as_posix()
    except ValueError:  # разные диски Windows
        img_rel = images.as_posix()
    md = render(TEMPLATE.read_text(encoding="utf-8"), facts, out, figs, img_rel)
    problems = eda_report.lint_ru(md, LINT_ALLOW)
    if problems:
        raise QCError("отчёт evaluate: типографика: " + "; ".join(problems[:10]))
    report.parent.mkdir(parents=True, exist_ok=True)
    images.mkdir(parents=True, exist_ok=True)
    for f in figs:
        shutil.copyfile(f.png, images / f.png.name)
    report.write_text(md, encoding="utf-8", newline="\n")
    log.info("evaluate: отчёт %s", report)
    return report
