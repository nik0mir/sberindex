"""Отчёт ``docs/network.md``: шаблон ``templates/network.md`` + таблицы и числа из ``outputs/network``.

Числа текста — только факты ``{{net.ключ}}`` (``outputs/network/report_facts.json``), таблицы — директивы
``<!-- table: имя -->``, рисунки — ``<!-- figure: N01 -->``. Утверждения текста, зависящие от результата
(какое правило выбрано, какое надёжнее), перечислены в ``CLAIMS`` и проверяются по числам: если хоть одно
перестало быть верным, отчёт не пишется (``QCError``, код 3) — шаблон надо переписать под новые числа.
Типографика — функции отчёта разведки (неразрывные пробелы, проверка кавычек, тире и десятичной точки).
"""

from __future__ import annotations

import json
import logging
import re
import shutil
from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from munnet import style
from munnet.config import Config
from munnet.contracts import QCError
from munnet.eda import report as eda_report
from munnet.eda.base import Fact, make_fact, write_json
from munnet.network.figures import RULE_LABELS, FigureInfo, label
from munnet.network.params import NetworkParams

log = logging.getLogger(__name__)

TEMPLATE = Path(__file__).resolve().parent / "templates" / "network.md"
SECTION = "net"
_TABLE = re.compile(r"<!--\s*table:\s*([a-z_]+)\s*-->")
_FIGURE = re.compile(r"<!--\s*figure:\s*(N\d{2})\s*-->")
_NOTE = re.compile(r"[ \t]*<!--\s*note\s*:.*?-->[ \t]*\n?", re.S)
LINT_ALLOW = [
    r"CC BY(?:-SA)? \d\.\d",
    r"\b\d{1,2}\.\d{2}\.\d{4}\b",  # дата
    r"\$\$.*?\$\$",  # формулы LaTeX
    r"\$[^$]+\$",
]

# Вид форматирования метрик сравнения (``comparison.csv``).
METRIC_KINDS: dict[str, str] = {
    "n_nodes": "int",
    "n_edges": "int",
    "density": "num3",
    "n_components": "int",
    "giant_share": "pct",
    "isolates": "int",
    "deg_median": "int",
    "deg_max": "int",
    "kocc_skew": "num1",
    "kocc_max": "int",
    "clustering": "num2",
    "assort_region": "num2",
    "within_region": "pct",
    "edge_km_median": "int",
    "attr_consistency": "num2",
    "assort_north": "num2",
    "reliability": "num2",
    "reliability_chance": "num3",
    "rho_years": "num2",
    "boot_jaccard": "num2",
    "geo_jaccard": "num3",
    "geo_difference": "num2",
    "q": "num2",
    "q_null_mean": "num2",
    "q_excess": "num2",
    "q_z": "int",
    "n_comm": "int",
    "ari_seeds": "num2",
    "ami_region": "num2",
    "n_sig_pairs": "int",
    "signal_share": "pct",
    "naive_pairs": "int",
    "naive_null_expected": "int",
    "spearman_jaccard": "num2",
    "sigma": "num2",
}

# Строки таблицы сравнения: метрика -> подпись (порядок = порядок строк).
COMPARISON_ROWS: dict[str, str] = {
    "n_edges": "Рёбер",
    "deg_median": "Степень: медиана",
    "deg_max": "Степень: максимум",
    "kocc_skew": "Асимметрия k-встречаемости (хабы)",
    "clustering": "Коэффициент кластеризации",
    "giant_share": "Доля гигантской компоненты",
    "isolates": "Изолятов",
    "reliability": "Надёжность: Жаккар сетей 2023 и 2024",
    "reliability_chance": "То же при случайных сетях",
    "rho_years": "ρ сходств всех пар 2023 и 2024",
    "boot_jaccard": "Бутстрап месяцев: Жаккар с сетью на всех месяцах",
    "geo_jaccard": "Жаккар с сетью «Дороги»",
    "within_region": "Рёбер внутри группы региона",
    "assort_region": "Ассортативность по группе региона",
    "edge_km_median": "Длина ребра, медиана, км",
    "attr_consistency": "Согласованность с экономикой места",
    "assort_north": "Ассортативность по Северу",
    "q_excess": "Модульность сверх нуля с теми же степенями",
    "n_comm": "Сообществ Leiden",
    "ari_seeds": "ARI разбиений разных seed",
    "ami_region": "AMI сообществ с группой региона",
    "signal_share": "Рёбер kNN, значимых после поправки",
    "time_bucket": "Время расчёта матрицы",
}
ATTR_LABELS: dict[str, str] = {
    "log_wage_rel": "Зарплата относительно региона",
    "urban_share_rel": "Доля горожан относительно региона",
    "age_old_share_rel": "Доля старших относительно региона",
    "log_pop_rel": "Население относительно региона",
    "market_access_rel": "Доступность рынков относительно региона",
    "emp_sh_primary": "Занятость: сельское хозяйство и добыча",
    "emp_sh_industry": "Занятость: промышленность",
    "emp_sh_trade_transport": "Занятость: стройка, торговля, транспорт",
    "emp_sh_market_services": "Занятость: рыночные услуги",
    "emp_sh_public": "Занятость: бюджетный сектор",
    "north": "Север (севернее 60°)",
}


# --- Факты ---------------------------------------------------------------------------------------


def _fact(facts: dict, key: str, value, kind: str) -> None:
    facts[f"{SECTION}.{key}"] = make_fact(f"{SECTION}.{key}", value, kind)


def build_facts(out: Path, p: NetworkParams, figs: list[FigureInfo]) -> dict[str, Fact]:
    """Все числа отчёта из файлов ``outputs/network`` (ключи ``net.*``)."""
    js = json.loads((out / "facts.json").read_text(encoding="utf-8"))
    comp = pd.read_csv(out / "comparison.csv")
    facts: dict[str, Fact] = {}
    for _, r in comp.iterrows():
        for m, kind in METRIC_KINDS.items():
            if m in r and pd.notna(r[m]):
                _fact(facts, f"{m}_{r['rule']}", float(r[m]), kind)
        for col in comp.columns:
            if col.startswith(("deg_9", "kocc_9")) and pd.notna(r[col]):
                _fact(facts, f"{col}_{r['rule']}", int(r[col]), "int")
            if col.startswith("assort_") and col not in METRIC_KINDS and pd.notna(r[col]):
                _fact(facts, f"{col}_{r['rule']}", float(r[col]), "num2")
        _fact(facts, f"hubs_{r['rule']}", str(r["top_hubs"]), "str")
        _fact(facts, f"meaning_{r['rule']}", str(r["meaning"]), "str")
        if "naive_pairs" in r and pd.notna(r["naive_pairs"]) and r["naive_pairs"] > 0:
            share = float(r["naive_null_expected"] / r["naive_pairs"])
            _fact(facts, f"naive_false_share_{r['rule']}", share, "pct")
        if pd.notna(r["reliability_chance"]) and r["reliability_chance"] > 0:
            _fact(facts, f"rel_ratio_{r['rule']}", float(r["reliability"] / r["reliability_chance"]), "int")
        _fact(facts, f"label_{r['rule']}", label(r["rule"]), "str")
        _fact(facts, f"ari3_{r['rule']}", float(r["ari_seeds"]), "num3")
    for tid, m in js.get("city_members", {}).items():
        _fact(facts, f"members_{tid}", m, "int")
    for key in ("n_nodes", "n_excluded", "n_no_road"):
        _fact(facts, key, js[key], "int")
    _fact(facts, "n_nodes_all", js["n_nodes"] + js["n_excluded"], "int")
    for key, v in js["road"].items():
        _fact(facts, f"road_{key}", v, "int")
    _fact(facts, "chosen_label", label(js["chosen"]), "str")
    _fact(facts, "alt_labels", ", ".join(f"«{label(a)}»" for a in js["alternatives"]), "str")
    sel = js["selection"]
    orders = pd.read_csv(out / "selection_orders.csv")
    main = orders.loc[(orders["set"] == "main") & orders["tie"].astype(bool)]
    _fact(facts, "orders_same", int((main["winner"] == js["chosen"]).sum()), "int")
    _fact(facts, "orders_total", len(main), "int")
    _fact(facts, "front", ", ".join(f"«{label(a)}»" for a in sel["front"]), "str")
    sets = pd.read_csv(out / "selection_sets.csv")
    for _, r in sets.iterrows():
        key = f"set_{r['set']}_{'tie' if bool(r['tie']) else 'raw'}"
        _fact(facts, f"{key}_front", _labels(r["front"]), "str")
        _fact(facts, f"{key}_winners", _winners_text(r["winners"]), "str")
        _fact(facts, f"{key}_borda", label(r["borda"]), "str")
    _fact(
        facts,
        "sets_all_basket_tie",
        int(sets.loc[sets["tie"].astype(bool), "all_basket"].astype(bool).all()),
        "int",
    )
    _fact(facts, "sets_borda_same", int((sets["borda"] == js["chosen"]).all()), "int")
    for crit, v in js["tolerance"].items():
        _fact(facts, f"tol_{crit}", v, "num3")
    mod = js["modularity_vs_region"]
    _fact(facts, "mod_rho_ami", mod["rho_ami"], "num2")
    _fact(facts, "mod_rho_within", mod["rho_within"], "num2")
    _fact(facts, "mod_rho_reliability", mod["rho_reliability"], "num2")
    _fact(facts, "mod_n_rules", mod["n_rules"], "int")
    _fact(facts, "mod_max_label", label(mod["max_rule"]), "str")
    boot = pd.read_csv(out / "selection_bootstrap.csv")
    _fact(facts, "n_boot", int(boot["replicate"].nunique()), "int")
    for crit in ("reliability", "attribute_consistency", "modularity"):
        w = boot.pivot(index="replicate", columns="rule", values=crit)
        if {"basket_cos", "basket_dist"} <= set(w.columns):
            _fact(facts, f"boot_n_dist_better_{crit}", int((w["basket_dist"] > w["basket_cos"]).sum()), "int")
    by_k = js.get("selection_by_k", {})
    _fact(facts, "k_same", sum(1 for w in by_k.values() if w == js["chosen"]), "int")
    _fact(facts, "k_total", len(by_k), "int")
    cos_dist = js.get("bootstrap_pairs", {}).get("basket_cos__basket_dist")
    if cos_dist:
        _fact(facts, "boot_dist_better", 1 - cos_dist["share_a_better"], "pct")
        _fact(facts, "boot_dist_minus_cos", -cos_dist["median_diff"], "num3")
    for pair, v in js.get("bootstrap_pairs", {}).items():
        _fact(facts, f"boot_share_{pair}", v["share_a_better"], "pct")
        _fact(facts, f"boot_diff_{pair}", v["median_diff"], "num3")
    for name, st in js["signal"].items():
        for stat, v in st.items():
            _fact(facts, f"sig_{name}_{stat}", v, "pct" if stat.startswith("share") else "num2")
    prm = js["params"]
    for key in (
        "k",
        "fdr_q",
        "shift_repeats",
        "shift_guard",
        "bootstrap",
        "rewire_repeats",
        "seeds",
        "mantel_permutations",
    ):
        _fact(facts, f"p_{key}", prm[key], "num2" if key == "fdr_q" else "int")
    _fact(facts, "p_resolution", prm["resolution"], "num1")
    _fact(facts, "p_naive_r", prm["naive_r"], "num1")
    _fact(facts, "p_k_grid", ", ".join(str(k) for k in prm["k_grid"]), "str")
    rules = prm["rules"]
    lag = next(v for v in rules.values() if v["kind"] == "rhythm_lag")
    dtw = next(v for v in rules.values() if v["kind"] == "rhythm_dtw")
    grav = next(v for v in rules.values() if v["kind"] == "gravity")
    _fact(facts, "p_max_lag", lag["max_lag"], "int")
    _fact(facts, "p_dtw_window", dtw["window"], "int")
    _fact(facts, "p_beta", grav["beta"], "num1")
    _fact(facts, "p_mantel_min_p", 1 / (1 + prm["mantel_permutations"]), "num2")
    # стандартная ошибка z Фишера 1/√(n − 3) при длине ряда 24 и 12 месяцев
    _fact(facts, "se_24", 1 / np.sqrt(24 - 3), "num2")
    _fact(facts, "se_12", 1 / np.sqrt(12 - 3), "num2")
    _fact(facts, "n_rules", len(comp), "int")
    _fact(facts, "n_candidates", len(p.candidates), "int")
    for rule, v in js["sigma"].items():
        _fact(facts, f"sigma_{rule}", v, "num2" if v < 100 else "int")

    grid = pd.read_csv(out / "sensitivity_k.csv")
    for _, r in grid.iterrows():
        for m in (
            "reliability",
            "geo_jaccard",
            "attr_consistency",
            "ami_region",
            "ari_vs_main_k",
            "q_excess",
        ):
            _fact(
                facts,
                f"k{int(r['k'])}_{m}_{r['rule']}",
                float(r[m]),
                "num2" if m != "geo_jaccard" else "num3",
            )
    spars = pd.read_csv(out / "sparsify.csv")
    for _, r in spars.iterrows():
        for m in ("n_edges", "isolates", "giant_share", "reliability", "deg_max", "within_region"):
            kind = (
                "int"
                if m in ("n_edges", "isolates", "deg_max")
                else "pct"
                if m in ("giant_share", "within_region")
                else "num2"
            )
            _fact(facts, f"sp_{r['method']}_{m}_{r['rule']}", float(r[m]), kind)
    pairs = pd.read_csv(out / "rule_pairs.csv")
    for _, r in pairs.iterrows():
        for m in ("jaccard", "rank_corr", "ari"):
            _fact(facts, f"pair_{m}_{r['rule_a']}__{r['rule_b']}", float(r[m]), "num2")
    _fact(facts, "pair_p_max", float(pairs["rank_corr_p"].max()), "num2")
    _fact(
        facts, "pair_n_sig", int((pairs["rank_corr_p"] <= 1 / (1 + prm["mantel_permutations"])).sum()), "int"
    )
    _fact(facts, "pair_n", len(pairs), "int")
    _fact(facts, "pair_n_sig_q", int((pairs["rank_corr_q"] < 0.05).sum()), "int")
    _fact(facts, "pair_q_min", float(pairs["rank_corr_q"].min()), "num3")
    modes = pd.read_csv(out / "modes.csv")
    for _, r in modes.iterrows():
        for col in modes.columns:
            if col in ("mode", "rule") or pd.isna(r[col]):
                continue
            v = float(r[col])
            kind = (
                "pct"
                if col.endswith(("share", "purity", "within_region"))
                else "int"
                if float(v).is_integer() and abs(v) >= 1
                else "num2"
            )
            _fact(facts, f"mode_{r['mode']}_{col}_{r['rule']}", v, kind)
    sep = modes.loc[modes["mode"] == "separate"]
    if len(sep):
        _fact(facts, "mode_inner_share", float(sep["n_inner"].iloc[0] / sep["n_nodes"].iloc[0]), "pct")
    noise = pd.read_csv(out / "windows_noise.csv")
    g = noise.groupby("length")[
        ["noise_jaccard", "change_jaccard", "noise_rank_corr", "change_rank_corr"]
    ].mean()
    for L, r in g.iterrows():
        for m, v in r.items():
            _fact(facts, f"win{int(L)}_{m}", float(v), "num2")
    win = pd.read_csv(out / "windows.csv")
    for _, r in win.iterrows():
        _fact(facts, f"win_{r['window_kind']}_n", int(r["n_windows"]), "int")
        _fact(facts, f"win_{r['window_kind']}_adjacent", float(r["adjacent_jaccard"]), "num2")
    attrs = pd.read_csv(out / "attributes.csv")
    for _, r in attrs.iterrows():
        vals = {
            c[len("assort_") :]: r[c]
            for c in attrs.columns
            if c.startswith("assort_") and c != "assort_north"
        }
        vals = {k: v for k, v in vals.items() if pd.notna(v)}
        if vals:
            best = max(vals, key=vals.get)
            order = sorted(vals, key=vals.get, reverse=True)
            _fact(facts, f"attr_rank_{r['rule']}", ",".join(order), "str")
            _fact(facts, f"attr_best_{r['rule']}", ATTR_LABELS.get(best, best).lower(), "str")
            _fact(facts, f"attr_best_value_{r['rule']}", float(vals[best]), "num2")
    for f in figs:
        _fact(facts, f"fig_{f.fid}_title", f.title, "str")
    return facts


CRITERIA_LABELS: dict[str, str] = {
    "reliability": "надёжность",
    "geo_difference": "отличие от географии",
    "attribute_consistency": "согласованность с местом",
    "simplicity": "простота",
    "modularity": "модульность сверх нуля",
    "probe_stability": "устойчивость зонда (ARI seed)",
}
SET_LABELS: dict[str, str] = {"main": "основной", "plan": "из PLAN.md", "extended": "расширенный"}


def _labels(text: str) -> str:
    """«basket_cos, basket_dist» -> «Косинус корзин», «Расстояние корзин»."""
    return ", ".join(f"«{label(r.strip())}»" for r in str(text).split(",") if r.strip())


def _winners_text(text: str) -> str:
    """«basket_dist: 4; basket_cos = basket_dist: 2» -> подписи правил и число порядков."""
    parts = []
    for item in str(text).split(";"):
        rules, n = item.rsplit(":", 1)
        names = " = ".join(f"«{label(r.strip())}»" for r in rules.split("="))
        parts.append(f"{names} — {int(n)}")
    return "; ".join(parts)


# --- Проверяемые утверждения текста --------------------------------------------------------------


def _v(facts: Mapping[str, Fact], key: str) -> float:
    return float(facts[f"{SECTION}.{key}"].value)


CLAIMS: dict[str, Callable[[Mapping[str, Fact]], bool]] = {
    "выбрано правило «Расстояние корзин»": lambda f: (
        f[f"{SECTION}.chosen_label"].value == RULE_LABELS["basket_dist"]
    ),
    "правила по корзине надёжнее всех правил по ритму": lambda f: (
        min(_v(f, "reliability_basket_dist"), _v(f, "reliability_basket_cos"))
        > max(
            _v(f, "reliability_rhythm_corr"), _v(f, "reliability_rhythm_lag"), _v(f, "reliability_rhythm_dtw")
        )
    ),
    "корзина почти не повторяет дороги (Жаккар < 0,02)": lambda f: _v(f, "geo_jaccard_basket_dist") < 0.02,
    "ритм держит больше рёбер внутри региона, чем корзина": lambda f: (
        _v(f, "within_region_rhythm_corr") > _v(f, "within_region_basket_dist")
    ),
    "у ритма есть хабы, у корзины нет (асимметрия k-встречаемости)": lambda f: (
        _v(f, "kocc_skew_rhythm_corr") > 1 and _v(f, "kocc_skew_basket_dist") < 0.5
    ),
    "гравитация делает Москву хабом": lambda f: (
        _v(f, "deg_90077_geo_gravity") > 10 * _v(f, "deg_median_geo_gravity")
    ),
    "средний ритм по шести частям надёжнее ряда «Все категории»": lambda f: (
        _v(f, "reliability_rhythm_corr") > _v(f, "reliability_rhythm_corr_all")
    ),
    "узлы-города не становятся хабами корзины": lambda f: (
        max(_v(f, "deg_90077_basket_dist"), _v(f, "deg_90078_basket_dist"))
        <= 2 * _v(f, "deg_median_basket_dist")
    ),
    "соседние годовые окна различаются сильнее шума": lambda f: (
        _v(f, "win12_change_jaccard") < _v(f, "win12_noise_jaccard")
    ),
    "в режиме separate районы в корзине реже связаны между собой, чем в ритме и по дорогам": lambda f: (
        _v(f, "mode_separate_inner_same_city_share_basket_dist")
        < min(
            _v(f, "mode_separate_inner_same_city_share_rhythm_corr"),
            _v(f, "mode_separate_inner_same_city_share_geo_road"),
        )
    ),
    "Спирмен меняет рёбра ритма меньше, чем смена года": lambda f: (
        _v(f, "spearman_jaccard_rhythm_corr") > _v(f, "reliability_rhythm_corr")
    ),
    "с допуском ничьей при любом наборе критериев побеждает правило по корзине": lambda f: (
        _v(f, "sets_all_basket_tie") == 1
    ),
    "по сумме рангов (Борда) при всех наборах выбрано расстояние корзин": lambda f: (
        _v(f, "sets_borda_same") == 1
    ),
    "расстояние согласованнее с местом, чем косинус, во всех повторах бутстрапа": lambda f: (
        _v(f, "boot_n_dist_better_attribute_consistency") == _v(f, "n_boot")
        and _v(f, "boot_n_dist_better_reliability") == _v(f, "n_boot")
    ),
    "модульность выше всего у дорожной сети и почти не меняется от вычитания региона": lambda f: (
        f[f"{SECTION}.mod_max_label"].value == RULE_LABELS["geo_road"]
        and abs(_v(f, "q_excess_basket_dist_abs") - _v(f, "q_excess_basket_dist")) < 0.03
        and _v(f, "ami_region_basket_dist_abs") > 10 * _v(f, "ami_region_basket_dist")
    ),
    "вычитание региона у корзины: меньше географии и выше согласованность, но ниже надёжность": lambda f: (
        _v(f, "geo_jaccard_basket_dist_abs") > 2 * _v(f, "geo_jaccard_basket_dist")
        and _v(f, "within_region_basket_dist_abs") > 2 * _v(f, "within_region_basket_dist")
        and _v(f, "attr_consistency_basket_dist") > _v(f, "attr_consistency_basket_dist_abs")
        and _v(f, "reliability_basket_dist_abs") > _v(f, "reliability_basket_dist")
    ),
    "и без вычета региона корзина ближе к географии меньше, чем ритм": lambda f: (
        _v(f, "geo_jaccard_basket_dist_abs") < _v(f, "geo_jaccard_rhythm_corr")
        and _v(f, "ami_region_basket_dist_abs") < _v(f, "ami_region_rhythm_corr")
    ),
    "вычитание региона у ритма убирает географию, но снижает надёжность ниже корзины": lambda f: (
        _v(f, "geo_jaccard_rhythm_corr_rel") < _v(f, "geo_jaccard_rhythm_corr")
        and _v(f, "reliability_rhythm_corr_rel") < _v(f, "reliability_rhythm_corr")
        and _v(f, "reliability_rhythm_corr_rel") < _v(f, "reliability_basket_dist")
    ),
    "поправка Бенджамини — Хохберга не меняет вывод теста Мантела": lambda f: (
        _v(f, "pair_n_sig_q") == _v(f, "pair_n_sig")
    ),
    "косинус и расстояние: надёжность и география в пределах допуска, согласованность за расстоянием, "
    "модульность за косинусом": lambda f: (
        abs(_v(f, "reliability_basket_dist") - _v(f, "reliability_basket_cos")) < _v(f, "tol_reliability")
        and abs(_v(f, "geo_jaccard_basket_dist") - _v(f, "geo_jaccard_basket_cos"))
        < _v(f, "tol_geo_difference")
        and _v(f, "attr_consistency_basket_dist") - _v(f, "attr_consistency_basket_cos")
        > _v(f, "tol_attribute_consistency")
        and _v(f, "q_excess_basket_cos") - _v(f, "q_excess_basket_dist") > _v(f, "tol_modularity")
    ),
    "в основном наборе с допуском фронт — только расстояние корзин": lambda f: (
        f[f"{SECTION}.front"].value == "«" + RULE_LABELS["basket_dist"] + "»"
    ),
    "лаговая корреляция на фронте расширенного набора без допуска — из-за разницы меньше допуска": lambda f: (
        RULE_LABELS["rhythm_lag"] in str(f[f"{SECTION}.set_extended_raw_front"].value)
        and 0 < _v(f, "ari_seeds_rhythm_lag") - _v(f, "ari_seeds_basket_cos") < _v(f, "tol_probe_stability")
    ),
    "без вычета региона корзина надёжнее ритма": lambda f: (
        _v(f, "reliability_basket_dist_abs") > _v(f, "reliability_rhythm_corr")
    ),
    "выбор одинаков при всех k": lambda f: _v(f, "k_same") == _v(f, "k_total"),
    "надёжность растёт с k у всех правил по тратам": lambda f: all(
        _v(f, f"k{a}_reliability_{r}") < _v(f, f"k{b}_reliability_{r}")
        for r in ("basket_cos", "basket_dist", "rhythm_corr", "rhythm_lag", "rhythm_dtw")
        for a, b in ((5, 10), (10, 15), (15, 20))
    ),
    "семейства корзины и ритма почти не пересекаются": lambda f: (
        _v(f, "pair_jaccard_basket_dist__rhythm_corr") < 0.05
        and _v(f, "pair_ari_basket_dist__rhythm_corr") < 0.1
    ),
    "внутри семейства корзины совпадение выше, чем между семействами": lambda f: (
        _v(f, "pair_ari_basket_cos__basket_dist")
        > max(_v(f, "pair_ari_basket_dist__rhythm_corr"), _v(f, "pair_ari_geo_road__basket_dist"))
    ),
    "при любой длине окна соседние окна различаются сильнее шума": lambda f: all(
        _v(f, f"win{L}_change_jaccard") < _v(f, f"win{L}_noise_jaccard") for L in (3, 6, 12)
    ),
    "сеть квартала шумнее годовой": lambda f: _v(f, "win3_noise_jaccard") < _v(f, "win12_noise_jaccard"),
    "у гравитации согласованность отрицательна, у дорог мала": lambda f: (
        _v(f, "attr_consistency_geo_gravity") < 0 and _v(f, "attr_consistency_geo_road") < 0.1
    ),
    "корзина сильнее всего связывает по размеру, затем по зарплате, горожанам и промышленности": lambda f: (
        str(f[f"{SECTION}.attr_rank_basket_dist"].value).split(",")[:4]
        == ["log_pop_rel", "log_wage_rel", "urban_share_rel", "emp_sh_industry"]
    ),
    "порог по r без поправки у одного ряда пропускает много ложных пар": lambda f: (
        _v(f, "naive_null_expected_rhythm_corr_all") > 0.2 * _v(f, "naive_pairs_rhythm_corr_all")
    ),
}


def check_claims(facts: Mapping[str, Fact]) -> None:
    """Все утверждения ``CLAIMS`` верны на фактах; неверное или непроверяемое (нет факта) — ``QCError``."""
    broken = []
    for text, fn in CLAIMS.items():
        try:
            ok = bool(fn(facts))
        except (KeyError, TypeError, ValueError):
            ok = False
        if not ok:
            broken.append(text)
    if broken:
        raise QCError("отчёт network: перестали быть верными утверждения шаблона: " + "; ".join(broken))


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


def table_comparison(out: Path) -> str:
    comp = pd.read_csv(out / "comparison.csv").set_index("rule")
    rows = []
    for m, text in COMPARISON_ROWS.items():
        cells = [text]
        for _, r in comp.iterrows():
            if m == "time_bucket":
                cells.append(str(r["time_bucket"]))
            elif m in ("reliability", "rho_years") and bool(r["static"]):
                cells.append("1 (не меняется)")
            else:
                cells.append(_fmt(r.get(m), METRIC_KINDS.get(m, "num2")))
        rows.append(cells)
    return _md(pd.DataFrame(rows, columns=["Измерение", *[label(r) for r in comp.index]]))


def table_grid(out: Path) -> str:
    g = pd.read_csv(out / "sensitivity_k.csv")
    df = pd.DataFrame(
        {
            "Правило": g["rule"].map(label),
            "k": g["k"].astype(int),
            "Рёбер": g["n_edges"].map(lambda v: _fmt(v, "int")),
            "Надёжность": g["reliability"].map(lambda v: _fmt(v, "num2")),
            "Жаккар с дорогами": g["geo_jaccard"].map(lambda v: _fmt(v, "num3")),
            "Согласованность с местом": g["attr_consistency"].map(lambda v: _fmt(v, "num2")),
            "Асимметрия k-встречаемости": g["kocc_skew"].map(lambda v: _fmt(v, "num1")),
            "Модульность сверх нуля": g["q_excess"].map(lambda v: _fmt(v, "num2")),
            "AMI с регионом": g["ami_region"].map(lambda v: _fmt(v, "num2")),
            "ARI с разбиением при k основном": g["ari_vs_main_k"].map(lambda v: _fmt(v, "num2")),
        }
    )
    return _md(df)


def table_sparsify(out: Path) -> str:
    s = pd.read_csv(out / "sparsify.csv")
    names = {"knn": "kNN (объединение)", "mutual_knn": "взаимный kNN", "threshold": "порог"}
    df = pd.DataFrame(
        {
            "Правило": s["rule"].map(label),
            "Разрежение": s["method"].map(names),
            "Рёбер": s["n_edges"].map(lambda v: _fmt(v, "int")),
            "Гигантская компонента": s["giant_share"].map(lambda v: _fmt(v, "pct")),
            "Изолятов": s["isolates"].map(lambda v: _fmt(v, "int")),
            "Степень: максимум": s["deg_max"].map(lambda v: _fmt(v, "int")),
            "Внутри региона": s["within_region"].map(lambda v: _fmt(v, "pct")),
            "Согласованность с местом": s["attr_consistency"].map(lambda v: _fmt(v, "num2")),
            "Надёжность": s["reliability"].map(lambda v: _fmt(v, "num2")),
        }
    )
    return _md(df)


def table_pairs(out: Path) -> str:
    s = pd.read_csv(out / "rule_pairs.csv")
    df = pd.DataFrame(
        {
            "Правило А": s["rule_a"].map(label),
            "Правило Б": s["rule_b"].map(label),
            "Жаккар рёбер": s["jaccard"].map(lambda v: _fmt(v, "num2")),
            "ρ сходств пар": s["rank_corr"].map(lambda v: _fmt(v, "num2")),
            "p (Мантел)": s["rank_corr_p"].map(lambda v: _fmt(v, "num2")),
            "q (Бенджамини — Хохберг)": s["rank_corr_q"].map(lambda v: _fmt(v, "num3")),
            "ARI сообществ": s["ari"].map(lambda v: _fmt(v, "num2")),
        }
    )
    return _md(df)


def table_selection(out: Path) -> str:
    c = pd.read_csv(out / "selection_criteria.csv")
    df = pd.DataFrame(
        {
            "Правило": c["rule"].map(label),
            "Надёжность": c["reliability"].map(lambda v: _fmt(v, "num2")),
            "Отличие от географии": c["geo_difference"].map(lambda v: _fmt(v, "num3")),
            "Согласованность с местом": c["attribute_consistency"].map(lambda v: _fmt(v, "num2")),
            "Простота (ранг)": (-c["simplicity"]).astype(int),
            "Модульность сверх нуля": c["modularity"].map(lambda v: _fmt(v, "num2")),
            "Устойчивость зонда (ARI seed)": c["probe_stability"].map(lambda v: _fmt(v, "num2")),
        }
    )
    return _md(df)


def table_selection_sets(out: Path) -> str:
    s = pd.read_csv(out / "selection_sets.csv")
    df = pd.DataFrame(
        {
            "Набор": s["set"].map(SET_LABELS),
            "Критерии": s["criteria"].map(
                lambda t: ", ".join(CRITERIA_LABELS[c.strip()] for c in t.split(","))
            ),
            "Допуск ничьей": s["tie"].map(lambda t: "да" if bool(t) else "нет"),
            "Парето-фронт": s["front"].map(_labels),
            "Победители при всех порядках (число порядков)": s["winners"].map(_winners_text),
            "Борда": s["borda"].map(lambda r: f"«{label(r)}»"),
        }
    )
    return _md(df)


def table_tolerance(out: Path) -> str:
    t = pd.read_csv(out / "selection_tolerance.csv")
    df = pd.DataFrame(
        {
            "Критерий": t["criterion"].map(CRITERIA_LABELS),
            "Допуск ничьей (SD по бутстрапу месяцев)": t["tolerance"].map(lambda v: _fmt(v, "num3")),
        }
    )
    return _md(df)


def table_ablation(out: Path) -> str:
    c = pd.read_csv(out / "comparison.csv").set_index("rule")
    pairs = [("basket_dist_abs", "basket_dist"), ("rhythm_corr", "rhythm_corr_rel")]
    spec = [
        ("reliability", "Надёжность: Жаккар сетей 2023 и 2024", "num2"),
        ("geo_jaccard", "Жаккар с сетью «Дороги»", "num3"),
        ("within_region", "Рёбер внутри группы региона", "pct"),
        ("ami_region", "AMI сообществ с группой региона", "num2"),
        ("assort_north", "Ассортативность по Северу", "num2"),
        ("attr_consistency", "Согласованность с экономикой места", "num2"),
        ("q_excess", "Модульность сверх нуля", "num2"),
        ("edge_km_median", "Длина ребра, медиана, км", "int"),
    ]
    cols = [r for pair in pairs for r in pair if r in c.index]
    rows = [[text, *[_fmt(c.loc[r, m], kind) for r in cols]] for m, text, kind in spec]
    return _md(pd.DataFrame(rows, columns=["Измерение", *[label(r) for r in cols]]))


def table_attributes(out: Path) -> str:
    a = pd.read_csv(out / "attributes.csv").set_index("rule")
    cols = [c for c in a.columns if c.startswith("assort_")]
    rows = []
    for c in cols:
        key = c[len("assort_") :]
        rows.append([ATTR_LABELS.get(key, key), *[_fmt(a.loc[r, c], "num2") for r in a.index]])
    return _md(pd.DataFrame(rows, columns=["Признак места", *[label(r) for r in a.index]]))


def table_modes(out: Path) -> str:
    m = pd.read_csv(out / "modes.csv")
    rows = []
    spec = [
        ("n_nodes", "Узлов сети", "int"),
        ("n_inner", "Из них районов Москвы и Петербурга", "int"),
        ("n_edges", "Рёбер", "int"),
        ("inner_same_city_share", "Рёбер районов, ведущих в район того же города", "pct"),
        ("inner_deg_median", "Степень районов, медиана", "int"),
        ("other_deg_median", "Степень остальных узлов, медиана", "int"),
        ("city77_n_comm", "Сообществ, по которым разошлись районы Москвы", "int"),
        ("city77_modal_share", "Доля районов Москвы в самом частом их сообществе", "pct"),
        ("city78_n_comm", "Сообществ, по которым разошлись районы Петербурга", "int"),
        ("city78_modal_share", "Доля районов Петербурга в самом частом их сообществе", "pct"),
        ("jaccard_common", "Жаккар рёбер с основным режимом на общих узлах", "num2"),
        ("ari_common", "ARI сообществ с основным режимом на общих узлах", "num2"),
        ("collapse_deg_90077", "Степень узла-города Москва в основном режиме", "int"),
        ("collapse_deg_90078", "Степень узла-города Петербург в основном режиме", "int"),
    ]
    for col, text, kind in spec:
        if col in m.columns:
            rows.append([text, *[_fmt(v, kind) for v in m[col]]])
    return _md(pd.DataFrame(rows, columns=["Измерение", *[f"{label(r)}" for r in m["rule"]]]))


def table_windows(out: Path) -> str:
    n = pd.read_csv(out / "windows_noise.csv")
    df = pd.DataFrame(
        {
            "Окно, мес.": n["length"].astype(int),
            "Отрезок": n["span"].astype(str).str.replace("…", "–"),
            "Жаккар: нечётные и чётные месяцы (шум)": n["noise_jaccard"].map(lambda v: _fmt(v, "num2")),
            "Жаккар: два соседних окна": n["change_jaccard"].map(lambda v: _fmt(v, "num2")),
            "ρ сходств: шум": n["noise_rank_corr"].map(lambda v: _fmt(v, "num2")),
            "ρ сходств: соседние окна": n["change_rank_corr"].map(lambda v: _fmt(v, "num2")),
        }
    )
    return _md(df)


def table_window_kinds(out: Path) -> str:
    w = pd.read_csv(out / "windows.csv")
    names = {"quarter": "кварталы", "half": "полугодия", "rolling": "скользящие 12 месяцев, шаг 1"}
    df = pd.DataFrame(
        {
            "Окна": w["window_kind"].map(names),
            "Число окон": w["n_windows"].astype(int),
            "Жаккар соседних окон: среднее": w["adjacent_jaccard"].map(lambda v: _fmt(v, "num2")),
            "минимум": w["adjacent_min"].map(lambda v: _fmt(v, "num2")),
            "максимум": w["adjacent_max"].map(lambda v: _fmt(v, "num2")),
        }
    )
    return _md(df)


TABLES: dict[str, Callable[[Path], str]] = {
    "comparison": table_comparison,
    "grid": table_grid,
    "sparsify": table_sparsify,
    "pairs": table_pairs,
    "selection": table_selection,
    "selection_sets": table_selection_sets,
    "tolerance": table_tolerance,
    "ablation": table_ablation,
    "attributes": table_attributes,
    "modes": table_modes,
    "windows": table_windows,
    "window_kinds": table_window_kinds,
}


# --- Сборка --------------------------------------------------------------------------------------


def render(template: str, facts: Mapping[str, Fact], out: Path, figs: list[FigureInfo], img_rel: str) -> str:
    text = _NOTE.sub("", template)
    text = eda_report.fill(text, facts)
    text = _TABLE.sub(lambda m: TABLES[m.group(1)](out), text)
    by_id = {f.fid: f for f in figs}

    def fig_md(m: re.Match) -> str:
        f = by_id[m.group(1)]
        n = int(f.fid[1:])
        return (
            f"![{f.alt}]({img_rel}/{f.png.name})\n\n*Рисунок {n}. {f.title}. {f.subtitle}. Источник: "
            f"{style.SOURCE_SBER}.* Данные: `outputs/network/figures/{f.data_csv.name}`."
        )

    text = _FIGURE.sub(fig_md, text)
    text = eda_report.unwrap_paragraphs(text)
    return eda_report.nbsp_markdown(text)


def write_report(cfg: Config, p: NetworkParams, out: Path, figs: list[FigureInfo]) -> Path:
    """Проверяет утверждения, собирает ``docs/network.md`` и копирует PNG рисунков в ``report_images``."""
    facts = build_facts(out, p, figs)
    write_json(
        {k: {"value": f.value, "text": f.text} for k, f in sorted(facts.items())}, out / "report_facts.json"
    )
    check_claims(facts)
    report = Path(p.report)
    images = Path(p.report_images)
    try:
        img_rel = Path(__import__("os").path.relpath(images, report.parent)).as_posix()
    except ValueError:  # разные диски Windows
        img_rel = images.as_posix()
    md = render(TEMPLATE.read_text(encoding="utf-8"), facts, out, figs, img_rel)
    problems = eda_report.lint_ru(md, LINT_ALLOW)
    if problems:
        raise QCError("отчёт network: типографика: " + "; ".join(problems[:10]))
    report.parent.mkdir(parents=True, exist_ok=True)
    images.mkdir(parents=True, exist_ok=True)
    for f in figs:
        dest = images / f.png.name
        if not dest.exists() or dest.read_bytes() != f.png.read_bytes():
            shutil.copyfile(f.png, dest)
    old = report.read_text(encoding="utf-8") if report.exists() else None
    if old != md:
        report.write_text(md, encoding="utf-8", newline="\n")
    log.info("network: отчёт %s", report)
    return report
