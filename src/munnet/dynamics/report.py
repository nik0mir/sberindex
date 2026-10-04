"""Отчёт ``docs/dynamics.md``: шаблон ``templates/dynamics.md`` + числа из ``outputs/dynamics``.

Числа текста — только факты ``{{dyn.ключ}}`` (``outputs/dynamics/report_facts.json``), таблицы — директивы
``<!-- table: имя -->``, рисунки — ``<!-- figure: D01 -->``. Утверждения шаблона, зависящие от результата,
перечислены в ``CLAIMS`` и проверяются по фактам: неверное — ``QCError`` (код 3), отчёт не пишется.
Фразы, у которых по-разному может выйти сам смысл (исход проверки сюжета), код собирает целиком из чисел.
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

from munnet import style
from munnet.config import Config
from munnet.contracts import PARTS, QCError
from munnet.dynamics.figures import APPROACH_LABELS, FEATURE_LABELS, FigureInfo
from munnet.dynamics.params import EVOLUTIONARY, FIXED, MAIN, DynParams
from munnet.eda import report as eda_report
from munnet.eda.base import Fact, make_fact, write_json

log = logging.getLogger(__name__)

TEMPLATE = Path(__file__).resolve().parent / "templates" / "dynamics.md"
SECTION = "dyn"
_TABLE = re.compile(r"<!--\s*table:\s*([a-z_]+)\s*-->")
_FIGURE = re.compile(r"<!--\s*figure:\s*(D\d{2})\s*-->")
_NOTE = re.compile(r"[ \t]*<!--\s*note\s*:.*?-->[ \t]*\n?", re.S)
LINT_ALLOW = [
    r"CC BY(?:-SA)? \d\.\d",
    r"\b\d{1,2}\.\d{2}\.\d{4}\b",
    r"\$\$.*?\$\$",
    r"\$[^$]+\$",
    r"KDD",
    r"разд\.\s\d+\.\d+",
]  # номер раздела статьи
SHORT = {MAIN: "main", FIXED: "fixed", EVOLUTIONARY: "evo"}
PART_GEN: dict[str, str] = {  # «доля трат на …»
    "food": "продовольствие",
    "marketplace": "маркетплейсы",
    "transport": "транспорт",
    "health": "здоровье",
    "cafe": "общепит",
    "other": "прочее",
}


def _fact(facts: dict, key: str, value, kind: str) -> None:
    facts[f"{SECTION}.{key}"] = make_fact(f"{SECTION}.{key}", value, kind)


def _flow_text(M: np.ndarray, top: int = 4) -> str:
    k = M.shape[0]
    flows = sorted(
        ((int(M[a, b]), a, b) for a in range(k) for b in range(k) if a != b and M[a, b] > 0), reverse=True
    )
    return ", ".join(f"из {a + 1} в {b + 1} — {style.fmt_num(n)}" for n, a, b in flows[:top])


def _events_summary(ev: pd.DataFrame, approach: str, kind: str, tau: float) -> dict[str, int]:
    d = ev.loc[(ev["approach"] == approach) & (ev["pair_kind"] == kind) & np.isclose(ev["tau"], tau)]
    return d["event"].value_counts().to_dict()


def _events_text(counts: Mapping[str, int]) -> str:
    names = {
        "survived": "сохранился",
        "split": "разделился",
        "merged": "слились",
        "reorganized": "перестроились",
        "emerged": "возник",
        "disappeared": "исчез",
    }
    parts = [f"«{names[k]}» — {style.fmt_num(v)}" for k, v in counts.items() if v]
    return ", ".join(parts) if parts else "событий нет"


def pct_range(lo: float, hi: float) -> str:
    """Диапазон долей с одним знаком процента: 0,137 и 0,171 -> «13,7–17,1%» (ru-text)."""
    return f"{style.fmt_pct(lo, 1).removesuffix('%')}–{style.fmt_pct(hi, 1)}"


def driver_posthoc(drv: pd.DataFrame, types: pd.DataFrame, alpha: float) -> dict[str, float | int | str]:
    """Числа прочтения проверки сюжета после вскрытия (03.10.2026); предрегистрированный вердикт не меняют.

    Тип окна строится по сети корзин, поэтому первая часть проверки (Манн — Уитни, |изменение| части корзины)
    срабатывает почти для любой части, по которой различаются типы. Факты: сколько из шести частей значимы,
    размах отношений медиан, какие части не значимы, и размах медиан типов по общепиту, маркетплейсам
    и продовольствию (``types.csv``) — по какой части типы различаются сильнее всего.
    """
    parts = drv.loc[drv["is_part"].astype(bool)]
    lo = parts["ratio"].idxmin()
    nonsig = parts.loc[parts["p_value"] >= alpha]
    span = {
        q: float(types[f"clr_rel_{q}"].max() - types[f"clr_rel_{q}"].min())
        for q in ("cafe", "marketplace", "food")
    }
    return {
        "drv_n_parts": len(parts),
        "drv_n_sig": int((parts["p_value"] < alpha).sum()),
        "drv_ratio_min": float(parts["ratio"].min()),
        "drv_ratio_max": float(parts["ratio"].max()),
        "drv_ratio_min_part": PART_GEN[lo.removeprefix("clr_rel_")],
        "drv_nonsig": ", ".join(
            f"{PART_GEN[q.removeprefix('clr_rel_')]} (p = {make_fact('p', v, 'p').text})"
            for q, v in nonsig["p_value"].items()
        ),
        "drv_cafe_span": span["cafe"],
        "drv_mp_span": span["marketplace"],
        "drv_food_span": span["food"],
    }


POSTHOC_KINDS = {
    "drv_n_parts": "int",
    "drv_n_sig": "int",
    "drv_ratio_min": "num2",
    "drv_ratio_max": "num2",
    "drv_ratio_min_part": "str",
    "drv_nonsig": "str",
    "drv_cafe_span": "num2",
    "drv_mp_span": "num2",
    "drv_food_span": "num2",
}


def build_facts(out: Path, p: DynParams, figs: list[FigureInfo]) -> dict[str, Fact]:
    js = json.loads((out / "facts.json").read_text(encoding="utf-8"))
    comp = pd.read_csv(out / "comparison.csv").set_index("approach")
    per = pd.read_csv(out / "periods.csv")
    ev = pd.read_csv(out / "events.csv")
    drv = pd.read_csv(out / "driver.csv").set_index("feature")
    drv_alt = pd.read_csv(out / "driver_stable_same.csv").set_index("feature")
    trans = pd.read_csv(out / "transitions.csv")
    eps = pd.read_csv(out / "evolutionary_epsilon.csv")
    regions = pd.read_csv(out / "regions.csv")
    win = pd.read_csv(out / "windows.csv")
    k = int(js["k"])
    f: dict[str, Fact] = {}
    for key in ("n_nodes", "n_windows", "k", "n_seeds", "bootstrap"):
        _fact(f, key, js[key], "int")
    _fact(f, "n_adjacent", js["n_windows"] - 1, "int")
    _fact(f, "alpha", js["alpha"], "num1")
    _fact(f, "level", js["level"], "pct")
    _fact(f, "tau", js["tau"], "num1")
    taus = sorted(js["taus"])
    _fact(f, "tau_lo", taus[0], "num1")
    _fact(f, "tau_hi", taus[-1], "num1")
    _fact(f, "epsilon", js["epsilon"], "num1")
    _fact(f, "driver_alpha", p.driver_alpha, "num2")
    qc = js["qc"]
    _fact(f, "qc_refit_ari", qc["refit_all_ari"], "num2")
    _fact(f, "qc_fixed_agree", qc["fixed_ref_agree"], "pct")
    _fact(f, "qc_graph_windows", qc["graph_windows_checked"], "int")
    roll = per.loc[per["kind"] == "rolling"]
    _fact(f, "fit_min", float(per.loc[per["kind"] != "all", "procrustes_fit"].min()), "num2")
    _fact(f, "fit_max", float(per.loc[per["kind"] != "all", "procrustes_fit"].max()), "num2")
    _fact(f, "fit_roll_min", float(roll["procrustes_fit"].min()), "num2")
    _fact(f, "seed_ari_min", float(per["seed_ari"].min()), "num2")
    _fact(f, "edges_min", int(roll["n_edges"].min()), "int")
    _fact(f, "edges_max", int(roll["n_edges"].max()), "int")
    for name, s in SHORT.items():
        c, r = js["changes"][name], comp.loc[name]
        _fact(f, f"change_{s}", c["change"], "pct")
        _fact(f, f"n_changed_{s}", c["n_changed"], "int")
        _fact(f, f"noise_{s}", c["noise"], "pct")
        _fact(f, f"noise23_{s}", c["noise_2023"], "pct")
        _fact(f, f"noise24_{s}", c["noise_2024"], "pct")
        _fact(f, f"diff_{s}", 100 * c["diff"], "pp")
        _fact(f, f"lo_{s}", 100 * c["lo"], "num1")
        _fact(f, f"hi_{s}", 100 * c["hi"], "num1")
        _fact(f, f"change_lo_{s}", c["change_lo"], "pct")
        _fact(f, f"change_hi_{s}", c["change_hi"], "pct")
        _fact(f, f"change_ci_{s}", pct_range(c["change_lo"], c["change_hi"]), "str")
        _fact(f, f"exceeds_{s}", int(c["exceeds"]), "int")
        _fact(f, f"n_rel_{s}", c["n_reliable"], "int")
        _fact(f, f"share_rel_{s}", c["n_reliable"] / js["n_nodes"], "pct")
        _fact(f, f"ratio_{s}", c["change"] / c["noise"] if c["noise"] > 0 else np.nan, "num1")
        _fact(f, f"agree23_{s}", c["halves_agree_2023"], "pct")
        _fact(f, f"agree24_{s}", c["halves_agree_2024"], "pct")
        _fact(f, f"adj_mean_{s}", r["adjacent_mean"], "pct")
        _fact(f, f"adj_max_{s}", r["adjacent_max"], "pct")
        _fact(f, f"ari_tr_{s}", r["ari_transitions"], "num2")
        _fact(f, f"jac_changed_{s}", r["jaccard_changed"], "num2")
        _fact(f, f"repro_{s}", r["main_reliable_reproduced"], "pct")
        _fact(f, f"also_rel_{s}", r["main_reliable_also_reliable"], "pct")
        _fact(f, f"ari_last_{s}", r["ari_last_vs_main"], "num2")
        M = trans.loc[(trans["approach"] == name) & (trans["kind"] == "reliable")]
        Mr = np.zeros((k, k), dtype=int)
        Mr[M["from"] - 1, M["to"] - 1] = M["n"]
        _fact(f, f"flows_{s}", _flow_text(Mr), "str")
        if s == "main":
            off = [(Mr[a, b], a, b) for a in range(k) for b in range(k) if a != b]
            n0, a0, b0 = max(off)
            _fact(f, "top_from", a0 + 1, "int")
            _fact(f, "top_to", b0 + 1, "int")
            _fact(f, "top_n", n0, "int")
            _fact(f, "top_share", n0 / max(1, Mr.sum()), "pct")
            flows_sorted = sorted(off, reverse=True)
            _fact(f, "top2_share", sum(x[0] for x in flows_sorted[:2]) / max(1, Mr.sum()), "pct")
    _fact(f, "reliable_same_window", js["reliable_same_window"], "int")
    cm = js["changes"][MAIN]
    _fact(f, "n_rel_window", cm["n_reliable_window"], "int")
    _fact(f, "n_rel_window_other", cm["n_reliable_window_other"], "int")
    info = js["main_info"]
    _fact(f, "chain_same", info["chain_direct_windows_same"], "int")
    # события
    for tau in taus:
        t = style.fmt_num(tau, 1).replace(",", "")
        adj = _events_summary(ev, MAIN, "adjacent", tau)
        fl = _events_summary(ev, MAIN, "first_last", tau)
        nz = _events_summary(ev, MAIN, "noise", tau)
        _fact(f, f"ev_adj_surv_{t}", adj.get("survived", 0), "int")
        _fact(f, f"ev_adj_{t}", _events_text(adj), "str")
        _fact(f, f"ev_fl_{t}", _events_text(fl), "str")
        _fact(f, f"ev_noise_{t}", _events_text(nz), "str")
        _fact(f, f"ev_fl_surv_{t}", fl.get("survived", 0), "int")
        for name, s in SHORT.items():
            _fact(f, f"ev_fl_{s}_{t}", _events_text(_events_summary(ev, name, "first_last", tau)), "str")
    fl05 = ev.loc[
        (ev["approach"] == MAIN) & (ev["pair_kind"] == "first_last") & np.isclose(ev["tau"], p.events_jaccard)
    ]
    surv = fl05.loc[fl05["event"] == "survived", "jaccard"]
    _fact(f, "fl_j_min", float(surv.min()) if len(surv) else np.nan, "num2")
    _fact(f, "fl_j_max", float(surv.max()) if len(surv) else np.nan, "num2")
    _fact(f, "adj_expected", (js["n_windows"] - 1) * k, "int")
    sizes = win.loc[win["approach"] == MAIN, [f"size_{j + 1}" for j in range(k)]].to_numpy()
    rng_rel = (sizes.max(axis=0) - sizes.min(axis=0)) / sizes.mean(axis=0)
    _fact(f, "size_rel_range", float(rng_rel.max()), "pct")
    _fact(f, "size_range_type", int(np.argmax(rng_rel)) + 1, "int")
    # проверка сюжета
    d = js["driver"]
    part = d["part"]
    mp = drv.loc[f"clr_rel_{part}"]
    top = d["top_part"]
    _fact(f, "drv_part", FEATURE_LABELS[f"clr_rel_{part}"].removeprefix("Корзина: "), "str")
    _fact(f, "drv_p", d["p_value"], "p")
    _fact(f, "drv_ratio", mp["ratio"], "num2")
    _fact(f, "drv_auc", mp["auc"], "num2")
    _fact(f, "drv_top", PART_GEN[top], "str")
    _fact(f, "drv_top_ratio", drv.loc[f"clr_rel_{top}", "ratio"], "num2")
    _fact(f, "drv_top_p", drv.loc[f"clr_rel_{top}", "p_value"], "p")
    parts_rank = drv.loc[drv["is_part"].astype(bool)].sort_values("ratio", ascending=False)
    rank = list(parts_rank.index).index(f"clr_rel_{part}") + 1
    _fact(f, "drv_rank", rank, "int")
    _fact(f, "drv_n_tr", d["n_transition"], "int")
    _fact(f, "drv_n_same", d["n_same"], "int")
    _fact(f, "drv_level_ratio", drv.loc["log_level_rel", "ratio"], "num2")
    _fact(f, "drv_level_p", drv.loc["log_level_rel", "p_value"], "p")
    types_tab = pd.read_csv(out / "types.csv").set_index("type")
    for key, value in driver_posthoc(drv, types_tab, p.driver_alpha).items():
        _fact(f, key, value, POSTHOC_KINDS[key])
    if d["significant"] and d["expectation_met"]:
        verdict = "подтвердилась полностью: обе части проверки выполнены"
    elif d["significant"]:
        verdict = (
            "подтвердилась частично: первая часть выполнена, вторая — нет, наибольшее отношение у доли трат "
            f"на {PART_GEN[top]}"
        )
    else:
        verdict = "не подтвердилась: первая часть не выполнена"
    _fact(f, "drv_verdict", verdict, "str")
    a = js["driver_stable_same"]
    _fact(f, "drv_alt_p", a["p_value"], "p")
    _fact(f, "drv_alt_top", PART_GEN[a["top_part"]], "str")
    _fact(f, "drv_alt_n", a["n_same"], "int")
    _fact(f, "drv_alt_ratio", drv_alt.loc[f"clr_rel_{part}", "ratio"], "num2")
    # эволюционная: чувствительность к ε
    _fact(f, "eps_min", float(eps["epsilon"].min()), "num1")
    _fact(f, "eps_max", float(eps["epsilon"].max()), "num1")
    _fact(f, "eps_change_min", float(eps["share_changed"].min()), "pct")
    _fact(f, "eps_change_max", float(eps["share_changed"].max()), "pct")
    _fact(f, "eps_ari_min", float(eps["ari_transitions"].min()), "num2")
    # ось «общепит ↔ маркетплейсы»: порядок типов и направление потоков
    types = pd.read_csv(out / "types.csv").set_index("type")
    cafe, mpl = types["clr_rel_cafe"], types["clr_rel_marketplace"]
    order = list(cafe.sort_values().index)
    _fact(f, "cafe_order", " < ".join(str(int(x)) for x in order), "str")
    rev = list(reversed(order))
    food = types["clr_rel_food"]
    _fact(f, "types_axis", int(list(mpl.sort_values().index) == rev), "int")
    _fact(f, "types_axis_food", int(list(food.sort_values().index) == rev), "int")
    _fact(
        f,
        "types_axis_city",
        int(
            list(types["log_pop_rel"].sort_values().index) == order
            and list(types["market_access_rel"].sort_values().index) == order
        ),
        "int",
    )
    flows = pd.read_csv(out / "flows.csv")
    flows = flows.loc[(flows["flow"] != "без смены") & (flows["n"] >= p.map_flow_min)]
    ok_cafe = ok_mp = 0
    for _, r in flows.iterrows():
        a, b = (int(x) for x in r["flow"].split(" → "))
        ok_cafe += int(np.sign(r["d_cafe"]) == np.sign(cafe[b] - cafe[a]))
        ok_mp += int(np.sign(r["d_marketplace"]) == np.sign(mpl[b] - mpl[a]))
    _fact(f, "n_big_flows", len(flows), "int")
    _fact(f, "n_cafe_consistent", ok_cafe, "int")
    _fact(f, "n_mp_consistent", ok_mp, "int")
    _fact(f, "flow_min", p.map_flow_min, "int")
    # регионы
    top_r = regions.head(p.top_regions)
    _fact(f, "n_regions_rel", int((regions["n_reliable"] > 0).sum()), "int")
    _fact(f, "n_regions", len(regions), "int")
    _fact(f, "top_region", str(top_r.iloc[0]["region"]), "str")
    _fact(f, "top_region_n", int(top_r.iloc[0]["n_reliable"]), "int")
    _fact(
        f, "top_regions_share", float(top_r["n_reliable"].sum() / max(1, regions["n_reliable"].sum())), "pct"
    )
    _fact(f, "top_regions_k", len(top_r), "int")
    _fact(f, "minutes", int(np.ceil(js["seconds"] / 60)), "int")
    for fig in figs:
        _fact(f, f"title_{fig.fid}", fig.title, "str")
    return f


def _v(f: Mapping[str, Fact], key: str) -> float:
    return float(f[f"{SECTION}.{key}"].value)


CLAIMS: dict[str, Callable[[Mapping[str, Fact]], bool]] = {
    "основной способ: изменение 2023 → 2024 сверх шума": lambda f: _v(f, "exceeds_main") == 1,
    "изменение сверх шума при всех трёх способах": lambda f: all(
        _v(f, f"exceeds_{s}") == 1 for s in SHORT.values()
    ),
    "шум двух лет близок (разница меньше 2 п. п.)": lambda f: (
        abs(_v(f, "noise23_main") - _v(f, "noise24_main")) < 0.02
    ),
    "соседние окна: все типы сохранились при основном пороге": lambda f: (
        _v(f, "ev_adj_surv_05") == _v(f, "adj_expected")
    ),
    "2023 → 2024: все типы сохранились при основном пороге": lambda f: _v(f, "ev_fl_surv_05") == _v(f, "k"),
    "перефит 24 месяцев повторяет итог этапа cluster": lambda f: _v(f, "qc_refit_ari") >= 0.99,
    "фиксированные центры на 24 месяцах повторяют итог": lambda f: _v(f, "qc_fixed_agree") == 1.0,
    "цепочка сопоставлений совпала с прямым сопоставлением во всех окнах": lambda f: (
        _v(f, "chain_same") == _v(f, "n_windows")
    ),
    "эволюционная кластеризация почти повторяет основной способ (ARI переходов > 0,8)": lambda f: (
        _v(f, "ari_tr_evo") > 0.8
    ),
    "фиксированные типы согласуются с основным слабее эволюционных": lambda f: (
        _v(f, "ari_tr_fixed") < _v(f, "ari_tr_evo")
    ),
    "эволюционная почти не зависит от ε (разброс доли смен меньше 2 п. п.)": lambda f: (
        _v(f, "eps_change_max") - _v(f, "eps_change_min") < 0.02
    ),
    "соседние окна меняют меньше узлов, чем шум половин": lambda f: (
        _v(f, "adj_mean_main") < _v(f, "noise_main")
    ),
    "два крупнейших надёжных потока — больше половины надёжных переходов": lambda f: (
        _v(f, "top2_share") > 0.5
    ),
    "в половинах года тип совпал у большинства узлов (> 90%)": lambda f: (
        min(_v(f, "agree23_main"), _v(f, "agree24_main")) > 0.9
    ),
    "типы упорядочены по общепиту и в обратном порядке — по маркетплейсам": lambda f: (
        _v(f, "types_axis") == 1
    ),
    "…и по продовольствию": lambda f: _v(f, "types_axis_food") == 1,
    "больше общепита — ближе к городу: население и доступность рынков в том же порядке": lambda f: (
        _v(f, "types_axis_city") == 1
    ),
    "шум — примерно треть изменения (от 25% до 42%)": lambda f: (
        0.25 <= _v(f, "noise_main") / _v(f, "change_main") <= 0.42
    ),
    "при всех порогах все типы сохранились в соседних окнах и между годами": lambda f: all(
        _v(f, f"ev_adj_surv_{t}") == _v(f, "adj_expected") and _v(f, f"ev_fl_surv_{t}") == _v(f, "k")
        for t in ("03", "05", "07")
    ),
    "уровень трат у перешедших не отличается значимо": lambda f: (
        _v(f, "drv_level_p") >= _v(f, "driver_alpha")
    ),
    "фиксированные типы находят меньше смен и надёжных переходов, чем основной": lambda f: (
        _v(f, "n_changed_fixed") < _v(f, "n_changed_main") and _v(f, "n_rel_fixed") < _v(f, "n_rel_main")
    ),
    "эволюционная сглаживает соседние окна сильнее основного": lambda f: (
        _v(f, "adj_mean_evo") < _v(f, "adj_mean_main")
    ),
    "строгое «без смены» даёт тот же вывод": lambda f: (
        _v(f, "drv_alt_p") < _v(f, "driver_alpha") and f["dyn.drv_alt_top"].value == f["dyn.drv_top"].value
    ),
    "каждый крупный поток идёт вдоль оси: общепит и маркетплейсы меняются к профилю нового типа": lambda f: (
        _v(f, "n_cafe_consistent") == _v(f, "n_big_flows") == _v(f, "n_mp_consistent")
        and _v(f, "n_big_flows") > 0
    ),
    "шум — меньше половины изменения при всех способах": lambda f: all(
        _v(f, f"noise_{s}") < 0.5 * _v(f, f"change_{s}") for s in SHORT.values()
    ),
    "надёжные = согласные с окнами + сменившие тип только по половинам (других нет)": lambda f: (
        _v(f, "n_rel_window") + _v(f, "reliable_same_window") == _v(f, "n_rel_main")
        and _v(f, "n_rel_window_other") == 0
    ),
    "уровень трат у перешедших меняется не сильнее (отношение < 1,2)": lambda f: (
        _v(f, "drv_level_ratio") < 1.2
    ),
    # Отступление после вскрытия 03.10.2026 (разделы «Главное за минуту» и 6): охраняют его текст.
    "после вскрытия: по записанному правилу — частично (первая часть выполнена, маркетплейсы не первые)": (
        lambda f: _v(f, "drv_p") < _v(f, "driver_alpha") and _v(f, "drv_rank") > 1
    ),
    "после вскрытия: первая часть выполняется почти для любой части корзины (значимы не меньше 4 из 6)": (
        lambda f: _v(f, "drv_n_sig") >= 4 and _v(f, "drv_n_parts") == 6
    ),
    "после вскрытия: уровень трат у перешедших не значим и меняется слабее любой части корзины": (
        lambda f: (
            _v(f, "drv_level_p") >= _v(f, "driver_alpha")
            and _v(f, "drv_level_ratio") < _v(f, "drv_ratio_min")
        )
    ),
    "после вскрытия: по общепиту типы различаются сильнее маркетплейсов и продовольствия, и он первый": (
        lambda f: (
            _v(f, "drv_cafe_span") > max(_v(f, "drv_mp_span"), _v(f, "drv_food_span"))
            and f["dyn.drv_top"].value == PART_GEN["cafe"]
        )
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
        raise QCError("отчёт dynamics: перестали быть верными утверждения шаблона: " + "; ".join(broken))


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


def table_types(out: Path, p: DynParams) -> str:
    t = pd.read_csv(out / "types.csv")
    df = pd.DataFrame(
        {
            "Тип": [f"Тип {int(x)}" for x in t["type"]],
            "Узлов": [_fmt(x, "int") for x in t["n"]],
            "Доля": [_fmt(x, "pct") for x in t["share"]],
            "Корзина: общепит": [_fmt(x, "num2") for x in t["clr_rel_cafe"]],
            "Корзина: маркетплейсы": [_fmt(x, "num2") for x in t["clr_rel_marketplace"]],
            "Уровень трат": [_fmt(x, "num2") for x in t["log_level_rel"]],
            "Доступность рынков": [_fmt(x, "num1") for x in t["market_access_rel"]],
            "Население": [_fmt(x, "num2") for x in t["log_pop_rel"]],
        }
    )
    return _md(df)


def table_periods(out: Path, p: DynParams) -> str:
    per = pd.read_csv(out / "periods.csv")
    win = pd.read_csv(out / "windows.csv")
    win = win.loc[win["approach"] == MAIN].set_index("window")
    k = len([c for c in win.columns if c.startswith("size_")])
    rows = []
    for _, r in per.iterrows():
        row = {
            "Период": r["period"].replace("…", "–"),
            "Месяцев": _fmt(r["n_months"], "int"),
            "Рёбер G": _fmt(r["n_edges"], "int"),
            "ARI разных seed": _fmt(r["seed_ari"], "num2"),
            "Поворот к 24 месяцам": _fmt(r["procrustes_fit"], "num2"),
        }
        for j in range(k):
            row[f"Тип {j + 1}"] = (
                _fmt(win.loc[r["period"], f"size_{j + 1}"], "int") if r["period"] in win.index else ""
            )
        row["Сменили тип к прошлому окну"] = (
            _fmt(win.loc[r["period"], "share_changed_prev"], "pct") if r["period"] in win.index else ""
        )
        rows.append(row)
    return _md(pd.DataFrame(rows))


def table_change(out: Path, p: DynParams) -> str:
    js = json.loads((out / "facts.json").read_text(encoding="utf-8"))
    rows = []
    for name in (MAIN, FIXED, EVOLUTIONARY):
        c = js["changes"][name]
        rows.append(
            {
                "Способ": APPROACH_LABELS[name],
                "Сменили тип 2023 → 2024": f"{_fmt(c['change'], 'pct')} ({_fmt(c['change_lo'], 'pct')}–"
                f"{_fmt(c['change_hi'], 'pct')})",
                "Шум 2023": _fmt(c["noise_2023"], "pct"),
                "Шум 2024": _fmt(c["noise_2024"], "pct"),
                "Разность": _fmt(100 * c["diff"], "pp"),
                "95% интервал разности": f"{_fmt(100 * c['lo'], 'num1')}…{_fmt(100 * c['hi'], 'num1')}",
                "Сверх шума": "да" if c["exceeds"] else "нет",
                "Надёжных переходов": _fmt(c["n_reliable"], "int"),
            }
        )
    return _md(pd.DataFrame(rows))


def table_matrix(out: Path, p: DynParams) -> str:
    t = pd.read_csv(out / "transitions.csv")
    t = t.loc[t["approach"] == MAIN]
    k = int(t["from"].max())
    allm = t.loc[t["kind"] == "all"].pivot(index="from", columns="to", values="n")
    rel = t.loc[t["kind"] == "reliable_window"].pivot(index="from", columns="to", values="n")
    rows = []
    for a in range(1, k + 1):
        row = {"Тип 2023 \\ тип 2024": f"Тип {a}"}
        for b in range(1, k + 1):
            row[f"Тип {b}"] = (
                _fmt(allm.loc[a, b], "int")
                if a == b
                else f"{_fmt(allm.loc[a, b], 'int')} ({_fmt(rel.loc[a, b], 'int')})"
            )
        rows.append(row)
    return _md(pd.DataFrame(rows))


def table_events(out: Path, p: DynParams) -> str:
    ev = pd.read_csv(out / "events.csv")
    ev = ev.loc[ev["approach"] == MAIN]
    kinds = ["survived", "split", "merged", "reorganized", "emerged", "disappeared"]
    labels = ["сохранился", "разделился", "слились", "перестроились", "возник", "исчез"]
    names = {
        "adjacent": "12 пар соседних окон",
        "first_last": "2023 → 2024",
        "noise": "шум: половины года (2 пары)",
    }
    rows = []
    for tau in sorted(ev["tau"].unique()):
        for pk in ("adjacent", "first_last", "noise"):
            d = ev.loc[np.isclose(ev["tau"], tau) & (ev["pair_kind"] == pk), "event"].value_counts()
            row = {"Порог Жаккара": _fmt(tau, "num1"), "Пары": names[pk]}
            for kk, lab in zip(kinds, labels, strict=True):
                row[lab] = _fmt(int(d.get(kk, 0)), "int")
            rows.append(row)
    return _md(pd.DataFrame(rows))


def table_flows(out: Path, p: DynParams) -> str:
    fl = pd.read_csv(out / "flows.csv")
    nodes = pd.read_csv(out / "nodes.csv")
    rows = []
    dcols = [f"d_clr_rel_{q}" for q in PARTS]
    for _, r in fl.iterrows():
        row = {"Переход": r["flow"], "Узлов": _fmt(r["n"], "int")}
        for q in PARTS:
            row[style.LABELS[q]] = _fmt(r[f"d_{q}"], "num2")
        row["Уровень трат"] = _fmt(r["d_level"], "num2")
        if r["flow"] != "без смены":
            a, b = (int(x) for x in r["flow"].split(" → "))
            m = nodes.loc[
                nodes["reliable"].astype(bool)
                & (nodes["half_type_2023"] == a)
                & (nodes["half_type_2024"] == b)
            ]
            m = m.assign(size=np.sqrt((m[dcols] ** 2).sum(axis=1))).sort_values(
                ["size", "territory_id"], ascending=[False, True]
            )
            ex = [
                f"{eda_report.typograph(str(n))} ({reg})"
                for n, reg in zip(m["name"], m["region"], strict=True)
            ]
            row["Примеры: сильнее всего изменилась корзина"] = "; ".join(ex[: p.examples])
        else:
            row["Примеры: сильнее всего изменилась корзина"] = style.NA_TEXT
        rows.append(row)
    return _md(pd.DataFrame(rows))


def table_regions(out: Path, p: DynParams) -> str:
    r = pd.read_csv(out / "regions.csv").head(p.top_regions)
    df = pd.DataFrame(
        {
            "Регион": r["region"],
            "Узлов": [_fmt(x, "int") for x in r["n_nodes"]],
            "Надёжных переходов": [_fmt(x, "int") for x in r["n_reliable"]],
            "Доля узлов региона": [_fmt(x, "pct") for x in r["share_reliable"]],
            "Сменили тип всего": [_fmt(x, "int") for x in r["n_changed"]],
        }
    )
    return _md(df)


def table_driver(out: Path, p: DynParams) -> str:
    d = pd.read_csv(out / "driver.csv")
    df = pd.DataFrame(
        {
            "Показатель": [FEATURE_LABELS[x] for x in d["feature"]],
            "Медиана модуля Δ: переход": [_fmt(x, "num3") for x in d["median_abs_transition"]],
            "Медиана модуля Δ: без смены": [_fmt(x, "num3") for x in d["median_abs_same"]],
            "Отношение": [_fmt(x, "num2") for x in d["ratio"]],
            "Медиана Δ: переход": [_fmt(x, "num3") for x in d["median_transition"]],
            "AUC": [_fmt(x, "num2") for x in d["auc"]],
            "p (Манн — Уитни, односторонний)": [_fmt(x, "p") for x in d["p_value"]],
        }
    )
    return _md(df)


def table_comparison(out: Path, p: DynParams) -> str:
    c = pd.read_csv(out / "comparison.csv")
    df = pd.DataFrame(
        {
            "Способ": [APPROACH_LABELS[a] for a in c["approach"]],
            "Сменили тип 2023 → 2024": [_fmt(x, "pct") for x in c["share_changed"]],
            "Шум": [_fmt(x, "pct") for x in c["noise"]],
            "Надёжных переходов": [_fmt(x, "int") for x in c["n_reliable"]],
            "Соседние окна: сменили тип, среднее": [_fmt(x, "pct") for x in c["adjacent_mean"]],
            "ARI переходов с основным": [_fmt(x, "num2") for x in c["ari_transitions"]],
            "Жаккар сменивших тип с основным": [_fmt(x, "num2") for x in c["jaccard_changed"]],
            "Надёжные переходы основного: тот же переход": [
                _fmt(x, "pct") for x in c["main_reliable_reproduced"]
            ],
            "…и тоже надёжный": [_fmt(x, "pct") for x in c["main_reliable_also_reliable"]],
        }
    )
    return _md(df)


def table_epsilon(out: Path, p: DynParams) -> str:
    e = pd.read_csv(out / "evolutionary_epsilon.csv")
    df = pd.DataFrame(
        {
            "ε": [_fmt(x, "num1") for x in e["epsilon"]],
            "Сменили тип 2023 → 2024": [_fmt(x, "pct") for x in e["share_changed"]],
            "Шум": [_fmt(x, "pct") for x in e["noise"]],
            "Надёжных переходов": [_fmt(x, "int") for x in e["n_reliable"]],
            "Соседние окна, среднее": [_fmt(x, "pct") for x in e["adjacent_mean"]],
            "ARI переходов с основным": [_fmt(x, "num2") for x in e["ari_transitions"]],
        }
    )
    return _md(df)


TABLES: dict[str, Callable[[Path, DynParams], str]] = {
    "types": table_types,
    "periods": table_periods,
    "change": table_change,
    "matrix": table_matrix,
    "events": table_events,
    "flows": table_flows,
    "regions": table_regions,
    "driver": table_driver,
    "comparison": table_comparison,
    "epsilon": table_epsilon,
}


# --- Сборка --------------------------------------------------------------------------------------


def render(template: str, facts: Mapping[str, Fact], out: Path, p: DynParams, figs, img_rel: str) -> str:
    text = _NOTE.sub("", template)
    text = eda_report.fill(text, facts)
    text = _TABLE.sub(lambda m: TABLES[m.group(1)](out, p), text)
    by_id = {fg.fid: fg for fg in figs}

    def fig_md(m: re.Match) -> str:
        fg = by_id[m.group(1)]
        n = int(fg.fid[1:])
        return (
            f"![{fg.alt}]({img_rel}/{fg.png.name})\n\n*Рисунок {n}. {fg.title}. {fg.subtitle}. Источник: "
            f"{fg.source}.* Данные: `outputs/dynamics/figures/{fg.data_csv.name}`."
        )

    text = _FIGURE.sub(fig_md, text)
    text = eda_report.unwrap_paragraphs(text)
    return eda_report.nbsp_markdown(text)


def write_report(cfg: Config, p: DynParams, out: Path, figs: list[FigureInfo]) -> Path:
    facts = build_facts(out, p, figs)
    write_json(
        {k: {"value": v.value, "text": v.text} for k, v in sorted(facts.items())}, out / "report_facts.json"
    )
    check_claims(facts)
    report, images = Path(p.report), Path(p.report_images)
    try:
        img_rel = Path(os.path.relpath(images, report.parent)).as_posix()
    except ValueError:  # разные диски Windows
        img_rel = images.as_posix()
    md = render(TEMPLATE.read_text(encoding="utf-8"), facts, out, p, figs, img_rel)
    problems = eda_report.lint_ru(md, LINT_ALLOW)
    if problems:
        raise QCError("отчёт dynamics: типографика: " + "; ".join(problems[:10]))
    report.parent.mkdir(parents=True, exist_ok=True)
    images.mkdir(parents=True, exist_ok=True)
    for fg in figs:
        dest = images / fg.png.name
        if not dest.exists() or dest.read_bytes() != fg.png.read_bytes():
            shutil.copyfile(fg.png, dest)
    old = report.read_text(encoding="utf-8") if report.exists() else None
    if old != md:
        report.write_text(md, encoding="utf-8", newline="\n")
    log.info("dynamics: отчёт %s", report)
    return report
