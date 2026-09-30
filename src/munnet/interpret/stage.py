"""Этап interpret (этап 5): проверки тезиса T1–T7 по предрегистрации ``interpret`` (коммит 90991e1; поправки
текстов до вскрытия — ``amendments.py``), фильтр устойчивости R1, профили, названия, примеры МО, описание
правилами и сборка главного вывода.

Порядок: отрицательные контроли (без меток типов; не прошли — стоп, код 3) → T1, T5, T4 на типах → T3 и T2
(псевдопары, способы прослеживания) → R1 (варианты входов и seed) → T6, T7 → профиль, названия, примеры,
дерево → главный вывод по ``thesis_assembly`` → таблицы, ``facts.json`` и отчёт. В конце — проверка, что
прочитан каждый ключ блока.

Слепой прогон (``--blind SEED``): типы и переходы перемешаны по узлам, выходы — в
``outputs/interpret_blind/`` (отчёт и рисунки — там же, не в docs/), крупная пометка «СЛЕПОЙ ПРОГОН».
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from munnet.clustering import inputs as CI
from munnet.clustering import methods as M
from munnet.clustering.params import ClusterParams
from munnet.config import Config
from munnet.contracts import QCError
from munnet.dynamics import tracking as T
from munnet.dynamics import windows as W
from munnet.eda.base import display_names, write_csv, write_json
from munnet.interpret import context as CX
from munnet.interpret import data as D
from munnet.interpret import describe as DS
from munnet.interpret import external as EX
from munnet.interpret import node_tables as NT
from munnet.interpret import order as O
from munnet.interpret import partitions as P
from munnet.interpret import placebo as PL
from munnet.interpret import runs as RN
from munnet.interpret import texts as TX
from munnet.interpret.labels import PROXY_LABELS, feature_label
from munnet.interpret.spec import Spec, resolve
from munnet.network.params import NetworkParams

log = logging.getLogger(__name__)

BLIND_LABEL = "СЛЕПОЙ ПРОГОН"


def _levels(spec: Spec) -> None:
    tests = spec["tests"]
    for name in (
        "T1_ladder_external",
        "T2_direction",
        "T3_reliable_placebo",
        "T5_trivial",
        "T6_bank_coverage",
        "T7_utility",
    ):
        TX.LEVELS[name] = tuple(str(x) for x in tests[name]["rule"]["levels"])


def _check_rules(spec: Spec) -> None:
    """Правила вердиктов реализованы ровно в записанном виде (иначе ValueError, а не тихая подмена)."""
    t = spec["tests"]
    problems = TX.validate_outcomes(t)
    if problems:
        raise ValueError("interpret: тексты исходов и код разошлись: " + "; ".join(problems))
    placeholders = TX.placeholders(str(t["T4_basket_vs_place"]["text"]))
    if not placeholders <= {"ari_basket", "ari_place", "diff", "ci"}:
        raise ValueError(f"T4.text: код не заполняет {sorted(placeholders)}")
    if not TX.placeholders(str(spec["scope"]["phrase"])) <= {"n_nodes", "n_regions", "n_partial"}:
        raise ValueError("scope.phrase: код заполняет n_nodes, n_regions, n_partial")
    stratified = [str(x) for x in t["T5_trivial"]["stratified"]["indicators"]]
    if stratified != [str(x) for x in t["T1_ladder_external"]["indicators"].keys()]:
        raise ValueError("T5.stratified.indicators: реализованы те же обороты, что T1.indicators")
    spec["naming"]["dictionary"].plain()
    spec["federal_districts"].plain()
    spec["naming"]["settlement_words"].plain()
    r1 = t["T1_ladder_external"]["rule"]
    r1.expect("confirmed", "both_beyond")
    r1.expect("partial_one", "one_beyond")
    r1.expect("partial_overall", "any_overall")
    r1.expect("not", "none_overall")
    t1 = t["T1_ladder_external"]
    t1.expect("role", "decide")
    t1.expect("missing", "drop")
    t1["overall"].expect("monotone", "nondecreasing_top_gt_bottom")
    t1["conditional"].expect("statistic", "stratified_spearman")
    t1["conditional"].expect("monotone", "nondecreasing_top_gt_bottom")
    t1["conditional"].expect("beyond_place", True)
    t1.expect("fdr", "bh")
    t1.expect("pairwise", "holm")
    r2 = t["T2_direction"]
    r2.expect("role", "decide")
    r2.expect("year_types", "half_types")
    r2.expect("up", "rank_increase")
    r2.expect("statistic", "up_share")
    r2.expect(None, "T3_placebo")  # ключ «null:» YAML читает как None
    r2.expect("test", "binomial_one_sided")
    r2.expect("p0_undetermined", "neither")
    if [str(x) for x in r2["halves"]] != ["main", "sensitivity"]:
        raise ValueError("T2.halves: реализовано [main, sensitivity]")
    rr = r2["rule"]
    rr.expect("confirmed", "both_halves")
    rr.expect("partial", "one_halves")
    rr.expect("down", "down_any")
    rr.expect("not", "otherwise")
    rr.expect("place_cap", "partial")
    pt = r2["place_tree"]
    pt.expect("target", "type_2023")
    pt.expect("class_weight", "balanced")
    pt.expect("statistic", "movers_to_place_type")
    pt.expect(None, "permute_within_source")  # ключ «null:» YAML читает как None
    r3 = t["T3_reliable_placebo"]
    r3.expect("role", "decide")
    r3.expect("reuse", "munnet.dynamics")
    r3.expect("observed_sensitivity", "recompute")
    r3.expect("excess", "median")
    r3.expect("denominator", "all_nodes")
    rr = r3["rule"]
    rr.expect("confirmed", "both_halves")
    rr.expect("partial", "one_halves")
    rr.expect("not", "no_halves")
    t4 = t["T4_basket_vs_place"]
    t4.expect("role", "describe")
    t4.expect("statistic", "ari_diff")
    t5 = t["T5_trivial"]
    t5.expect("role", "decide")
    t5.expect("region_by_construction", True)
    t5["stratified"].expect("statistic", "kw_epsilon2_within_strata")
    t5["stratified"].expect("beyond_place", True)
    t5["stratified"].expect("fdr", "bh")
    rr = t5["rule"]
    rr.expect("confirmed", "one_and_three")
    rr.expect("partial", "three_not_one")
    rr.expect("not_empty", "one_not_three")
    rr.expect("not_repeats", "neither")
    t6 = t["T6_bank_coverage"]
    t6.expect("role", "decide")
    t6.expect("change", "diff")
    t6.expect("stayers", "same_source_type_unchanged")
    t6.expect("strata", "source_type")
    t6.expect("cliff", "within_source_pairs")
    t6.expect("test", "stratified_permutation_greater")
    t6.expect("per_flow", "describe")
    rr = t6["rule"]
    rr.expect("confirmed", "p_and_cliff")
    rr.expect("partial", "one_of_two")
    rr.expect("not", "none")
    t7 = t["T7_utility"]
    t7.expect("role", "decide")
    t7.expect("scale", "robust")
    t7.expect("other_regions", "region_group")
    t7.expect("type_label", "type_2023")
    t7.expect("common_set", True)
    t7.expect("error", "abs_median")
    t7.expect("example", "median_error_product")
    rr = t7["rule"]
    rr.expect("against_B", "both_targets")
    rr.expect("against_C", "main_target")
    rr.expect("confirmed", "both")
    rr.expect("partial", "one")
    rr.expect("not", "none")
    rr["type_gain"].expect("adds", "lower_gt_0")
    rr.expect("product_set", "by_type_gain")
    sets = t7["sets"]
    sa, sb, sc, sd = sets["A"], sets["B"], sets["C"], sets["D"]
    sa.expect("rule", "nearest_basket")
    sa.expect("same_type", True)
    sa.expect("other_regions", True)
    sb.expect("rule", "nearest_geo")
    sb.expect("own_region", True)
    sc.expect("rule", "random")
    sc.expect("same_pop_decile", True)
    sc.expect("other_regions", True)
    sd.expect("rule", "nearest_basket")
    sd.expect("same_type", False)
    sd.expect("other_regions", True)
    rob = spec["robustness"]
    rob.expect("match", "hungarian_jaccard")
    if [str(x) for x in rob["statements"]] != ["basket_sign", "t1_verdict", "t2_verdict", "t3_verdict"]:
        raise ValueError("robustness.statements: реализованы basket_sign, t1_verdict, t2_verdict, t3_verdict")
    rob.expect("variants_required", "all")
    rob.expect("main_text", "lowest")
    rob.expect("main_run_verdict", "check_section")
    rob.expect("basket_sign_unstable", "limitations")
    pr = spec["profile"]
    pr.expect("center", "median")
    pr.expect("effect_unit", "mad")
    pr.expect("cliff_delta", True)
    if [str(x) for x in pr["weights"]] != ["none", "pop_avg"]:
        raise ValueError("profile.weights: реализовано [none, pop_avg]")
    nm = spec["naming"]
    nm.expect("weights", "none")
    nm.expect("unstable_parts", "skip")
    nm.expect("duplicates", "add_part_2")
    ex = spec["examples"]
    ex.expect("space", "hybrid_input")
    ex.expect("tie", "min_territory_id")
    ex.expect("one_per_region", True)
    ex.expect("exclude_city_nodes", True)
    ex.expect("flag_workplace_based", True)
    ex.expect("show_region", True)
    tr = spec["tree"]
    tr.expect("class_weight", "balanced")
    tr.expect("collapse_same_class", True)
    tr.expect("journalist_rule", "max_type_nodes")
    if [str(x) for x in tr["baselines"]] != ["majority", "balanced_random"]:
        raise ValueError("tree.baselines: реализовано [majority, balanced_random]")
    tr["fca"].expect("binarize", "terciles")
    cs = spec["controls"]
    cs.expect("order", "best_of_24")
    cs.expect("on_fail", "stop")
    cs.expect("publish", True)
    cs["reference"].expect("self_out", True)
    cs["reference"].expect("basket_parts", True)


def _paths(cfg: Config, spec: Spec, blind: int | None) -> tuple[Path, Path, Path]:
    report, images = Path(str(spec["report"])), Path(str(spec["report_images"]))
    if blind is None:
        return cfg.dir("outputs") / "interpret", report, images
    out = cfg.dir("outputs") / "interpret_blind"
    return out, out / "interpretation_blind.md", out / "img"


def _plain(obj: Any) -> Any:
    if is_dataclass(obj):
        return _plain(asdict(obj))
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist() if obj.size <= 64 else f"array[{obj.shape}]"
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


# --- T1/T5 на делении ----------------------------------------------------------------------------------


def _t1(ranks: np.ndarray, ev: O.Evidence, spec: Spec, rng, exclude=()) -> O.T1Result:
    t1 = spec["tests"]["T1_ladder_external"]
    return O.t1_eval(ranks, ev, rng, float(t1["alpha"]), float(t1["overall"]["rho_min"]), exclude)


def _t5(labels: np.ndarray, ev: O.Evidence, cx: CX.Ctx, spec: Spec, rng, exclude=()) -> O.T5Result:
    t5 = spec["tests"]["T5_trivial"]
    return O.t5_eval(
        labels, ev, cx.trivial, rng, float(t5["ami_max"]), float(t5["stratified"]["alpha"]), exclude
    )


def run_controls(cx: CX.Ctx, ev: O.Evidence, spec: Spec, seed: int) -> tuple[pd.DataFrame, dict]:
    """Отрицательные контроли: вердикт не выше ожидаемого; случайные — не больше ``random_max_above`` выше."""
    cs = spec["controls"]
    rng = np.random.default_rng([seed, 311])
    parts = CX.controls_partitions(cx, spec)
    rnd = cs["random"]
    if str(rnd["sizes"]) != "ladder":
        raise ValueError("controls.random.sizes: реализовано ladder")
    exp = cs["expected"]
    rows = []
    # порядок ступеней контроля — по ρ (b) оборота общепита (controls.order: «по ρ (b) общепита»)
    cat = "catering"
    if cat not in ev.turnovers:
        raise ValueError("controls.order: нужен оборот catering в T1.indicators")
    for p in parts:
        ranks, order = O.best_order(p.labels, lambda r: O.rho_b_point(r, ev.turnovers[cat]))
        r1 = _t1(ranks, ev, spec, rng)
        r5 = _t5(p.labels, ev, cx, spec, rng)
        rows.append(_control_row(p.name, p.label, "partitions", r1, r5, exp))
    for i in range(int(rnd["draws"])):
        lab = P.random_labels(cx.sizes, rng)
        r1 = _t1(lab, ev, spec, rng)
        r5 = _t5(lab, ev, cx, spec, rng)
        rows.append(_control_row(f"random_{i + 1}", f"случайные метки {i + 1}", "random", r1, r5, exp))
    tab = pd.DataFrame(rows)
    info = controls_info(tab, len(parts), int(rnd["draws"]), int(cs["random_max_above"]))
    # справки (reference) — описание, в вердикт и в on_fail не входят
    ref_rows = []
    for rv in cx.rivals:
        if rv.kind != "sized" or not rv.name.startswith("sized:") or not rv.name.endswith(":+"):
            continue
        excl = [rv.name, rv.name[:-1] + "-"]
        ranks, _ = O.best_order(rv.labels, lambda r: O.rho_b_point(r, ev.turnovers[cat]))
        r1 = _t1(ranks, ev, spec, rng, exclude=excl)
        ref_rows.append(_control_row(rv.name, rv.label, "self_out", r1, None, None))
    base = cx.data.base
    for j, part in enumerate(base.b_names):
        v = base.B[cx.pos, j]
        lab = P.sized_groups(v, cx.ids, cx.sizes)
        ranks, _ = O.best_order(lab, lambda r: O.rho_b_point(r, ev.turnovers[cat]))
        r1 = _t1(ranks, ev, spec, rng)
        r5 = _t5(lab, ev, cx, spec, rng)
        ref_rows.append(_control_row(part, feature_label(part), "basket_parts", r1, r5, None))
    return pd.concat([tab, pd.DataFrame(ref_rows)], ignore_index=True), info


def controls_info(tab: pd.DataFrame, n_partitions: int, n_random: int, random_max_above: int) -> dict:
    """Итог контролей (``controls.expected``): каждое деление не выше ожидаемого по T1 и T5, случайных выше
    ожидаемого — не больше ``random_max_above``."""
    part = tab.loc[tab["group"] == "partitions"]
    rnd = tab.loc[tab["group"] == "random"]
    bad = part.loc[~(part["t1_ok"].astype(bool) & part["t5_ok"].astype(bool)), "name"].tolist()
    n_above = int((~(rnd["t1_ok"].astype(bool) & rnd["t5_ok"].astype(bool))).sum())
    info = {
        "n_partitions": int(n_partitions),
        "n_random": int(n_random),
        "bad_partitions": bad,
        "n_random_above": n_above,
        "random_max_above": int(random_max_above),
    }
    info["passed"] = not bad and n_above <= int(random_max_above)
    return info


def stop_on_controls(cinfo: dict, out: Path) -> None:
    """``controls.on_fail: stop``: контроли выше ожидаемого — controls.json и стоп (код 3)
    до расчётов на типах."""
    if cinfo["passed"]:
        return
    write_json(_plain(cinfo), out / "controls.json")
    raise QCError(
        "interpret.controls: правило не отличает тип от экономики места — контроли выше ожидаемого: "
        f"{cinfo['bad_partitions']}, случайных выше ожидаемого {cinfo['n_random_above']}"
    )


def _control_row(name, label, group, r1, r5, exp) -> dict:
    row = {"name": name, "label": label, "group": group, "t1": r1.verdict, "t5": r5.verdict if r5 else None}
    # проверка чувствительности (строгое прочтение «за сверх места») — только для публикации:
    # в t1_ok, controls_info и стоп не входит
    row["t1_strict"] = r1.verdict_strict
    for n, t in r1.per.items():
        row[f"rho_b_{n}"] = t.rho_b
        row[f"diff_lo_{n}"] = t.diff_ci[0]
    if r5:
        row["max_ami"] = r5.max_ami
        for n, t in r5.per.items():
            row[f"eps_{n}"] = t.eps
            row[f"eps_diff_lo_{n}"] = t.diff_ci[0]
    if exp is not None:
        key = "random" if group == "random" else "partitions"
        e1 = str(exp["T1_ladder_external"][key])
        e5 = str(exp["T5_trivial"][key])
        row["t1_ok"] = TX.level_rank("T1_ladder_external", r1.verdict) <= TX.level_rank(
            "T1_ladder_external", e1
        )
        row["t5_ok"] = TX.level_rank("T5_trivial", r5.verdict) <= TX.level_rank("T5_trivial", e5)
    return row


# --- Основной расчёт -----------------------------------------------------------------------------------


def run(cfg: Config, blind: int | None = None) -> None:
    t_start = time.perf_counter()
    spec = Spec(cfg["interpret"])
    if int(spec["seed"]) != int(cfg["seed"]):
        raise QCError(f"interpret.seed = {spec['seed']} ≠ seed = {cfg['seed']}")
    seed = int(cfg["seed"])
    if blind is not None and int(blind) == seed:
        raise ValueError("слепой прогон: seed перестановки должен отличаться от общего seed")
    _levels(spec)
    _check_rules(spec)
    out, report_path, img_dir = _paths(cfg, spec, blind)
    out.mkdir(parents=True, exist_ok=True)
    # facts.json — метка состояния каталога: пока прогон идёт или если он оборвался, рядом лежат выходы
    # прошлого прогона и свежий controls.csv; читателям выходов верить им можно только при status = done
    write_status(out, "running", blind)
    try:
        _run(cfg, blind, spec, seed, out, report_path, img_dir, t_start)
    except BaseException as e:
        write_status(out, "failed", blind, error=f"{type(e).__name__}: {e}")
        raise


STATUS_DONE = "done"


def write_status(out: Path, status: str, blind: int | None, **extra: Any) -> None:
    """Метка незавершённого прогона в ``facts.json`` (``status``: running | failed); готовый прогон пишет
    полный ``facts.json`` со ``status = done`` последним шагом."""
    write_json(
        {"status": status, "blind": blind, "time": time.strftime("%Y-%m-%d %H:%M:%S"), **extra},
        out / "facts.json",
    )


def check_seed_protocol(fin: Mapping, seed: int) -> tuple[int, ...]:
    """Протокол гибрида итога — seed … seed + n − 1 (final.json ``seeds``): R1 строит протокол seed s как
    s … s + n − 1, поэтому другой протокол — стоп до расчётов, а не тихое расхождение с записью."""
    seeds = tuple(int(x) for x in fin["seeds"])
    if not seeds or seeds != tuple(range(seed, seed + len(seeds))):
        raise ValueError(
            f"interpret: протокол seed итога {list(seeds)} ≠ {seed}…{seed + len(seeds) - 1}: seed-прогоны R1 "
            "(s … s + n − 1) разошлись бы с записью robustness.seeds"
        )
    return seeds


def _run(  # noqa: C901 — последовательность шагов предрегистрации
    cfg: Config,
    blind: int | None,
    spec: Spec,
    seed: int,
    out: Path,
    report_path: Path,
    img_dir: Path,
    t_start: float,
) -> None:
    data = D.load(cfg, spec, blind)
    check_seed_protocol(data.fin, seed)
    members = load_members(cfg)
    # варианты R1 — сверка с clustering.selection.sensitivity.variants и входы (режим узлов nodes_separate)
    # до контролей и тяжёлых шагов: многочасовой прогон не должен упасть на них после плацебо
    variant_pre = prepare_variants(cfg, data, [str(x) for x in spec["robustness"]["variants"]])
    timing: dict[str, float] = {}

    def tick(name: str, t0: float) -> float:
        timing[name] = time.perf_counter() - t0
        log.info("interpret: %s — %.0f с", name, timing[name])
        return time.perf_counter()

    t0 = time.perf_counter()
    cx = CX.build(data, spec)
    ev = CX.evidence(cx, spec, seed, 0)
    t0 = tick("evidence", t0)

    # --- 1. Отрицательные контроли (без меток типов) --------------------------------------------------
    ctab, cinfo = run_controls(cx, ev, spec, seed)
    write_csv(ctab, out / "controls.csv")
    t0 = tick("controls", t0)
    stop_on_controls(cinfo, out)

    # --- 2. T1, T5, T4 на типах ---------------------------------------------------------------------
    k = data.k
    order = data.order
    types_t = data.final[cx.pos]
    ranks_t = data.rank_of(types_t)
    rng = np.random.default_rng([seed, 401])
    t1_main = _t1(ranks_t, ev, spec, rng)
    t1s = spec["tests"]["T1_ladder_external"]
    pairwise = O.t1_pairwise(ranks_t, ev, int(t1s["bootstrap"]), rng, float(t1s["ci_level"]), k)
    ru = t1s["regions_up"]
    region_up = {}
    for ind in ru["indicators"]:
        vals = cx.density if str(ind) == "density" else cx.indicators[str(ind)]
        region_up[str(ind)] = O.regions_up(
            ranks_t, vals, cx.groups, int(ru["min_steps"]), int(ru["permutations"]), rng
        )
    proxies = _proxies(cx, ranks_t, spec, k)
    t5_main = _t5(types_t, ev, cx, spec, rng)
    comp = spec["tests"]["T5_trivial"]["composition"]
    t5_main.composition = O.composition(
        types_t, cx.urban_group, int(comp["min_types"]), float(comp["min_share"])
    )
    from sklearn.metrics import adjusted_mutual_info_score

    t5_main.region_ami = float(adjusted_mutual_info_score(cx.region_part, types_t))
    t4 = _t4(data, cx, ev, spec, seed, types_t)
    t0 = tick("t1_t5_t4", t0)

    # --- 3. T3 и T2: псевдопары ----------------------------------------------------------------------
    t3s = spec["tests"]["T3_reliable_placebo"]
    schemes = RN.schemes_of(t3s)
    n_pairs = int(t3s["pseudo_pairs"])
    # число процессов — clustering.impl.workers (итог от него не зависит: seed задачи задан её номером)
    workers = ClusterParams.from_config(cfg).workers
    fin = data.fin
    netp = NetworkParams.from_config(cfg)
    k_fin, alpha = int(fin["k"]), float(fin["hybrid_alpha"])
    if k_fin != k:
        raise QCError(f"interpret: K итога {k_fin} ≠ числу типов лестницы {k}")
    seeds_fin = check_seed_protocol(fin, seed)
    dimpl = cfg["dynamics"]["impl"]
    space = W.HybridSpace.build(data.base, k, alpha, int(fin["best_seed"]))
    final0 = data.final - 1  # 0…K−1 в номерах записи
    C24 = T.centers(space.Z_ref, final0, k)
    rob = spec["robustness"]
    tracking_alts = [str(x) for x in rob["tracking"]]
    if set(tracking_alts) != {PL.FIXED, PL.EVO}:
        raise ValueError("robustness.tracking: реализовано [fixed_prototypes, evolutionary]")
    main_ctx = PL.RunCtx(
        name="main",
        ns=data.ns,
        base=data.base,
        netp=netp,
        rule=data.cp.graph_rule,
        knn=data.cp.graph_k,
        k=k,
        alpha=alpha,
        method=str(fin["method"]),
        param=fin["param"],
        seeds=seeds_fin,
        impl=fin["impl"],
        schemes=schemes,
        space=space,
        C24=C24,
        epsilon=float(dimpl["evolutionary"]["epsilon"]),
        max_iter=int(dimpl["evolutionary"]["max_iter"]),
        blind=blind,
    )
    cache = out / "cache"
    # ключ кэша: код тяжёлых модулей и поддеревья конфига сети, входов и реализации (runs.HEAVY_*)
    chash = (RN.code_hash(), RN.config_hash(cfg))
    wins = tuple("…".join(str(x) for x in spec["windows"][w]) for w in ("year_a", "year_b"))
    for w in wins:
        if (PL.MAIN, w) not in data.labels_dyn:
            raise ValueError(f"interpret.windows: окна {w} нет в outputs/dynamics/labels.csv")
    sets_main = PL.pseudo_sets(n_pairs, np.random.default_rng([seed, 601]))
    ari_min = float(dimpl["final_ari_min"])
    # ключ — содержимое входов прогона (контекст, метки, выходы dynamics, наборы месяцев), а не время файлов:
    # перезапуск evaluate или копирование выходов кэш не сбрасывают, другие данные — сбрасывают
    key = RN._key(
        "main",
        blind,
        RN.digest(main_ctx, final0, data.nodes_dyn, data.labels_dyn, sets_main, ari_min, wins),
        chash,
    )
    main_run, alt_runs, qc_main = RN.cached(
        cache,
        key,
        lambda: RN.main_run(
            main_ctx,
            final0,
            data.nodes_dyn,
            data.labels_dyn,
            sets_main,
            workers,
            ari_min,
            wins,
        ),
    )
    t0 = tick("placebo_main", t0)
    rank0 = np.array([order.index(t + 1) + 1 for t in range(k)])  # ступень типа (0…K−1 → 1…K)
    t2s = spec["tests"]["T2_direction"]
    t3_main = _t3(main_run, spec, schemes)
    t2_main = _t2(main_run, spec, rank0, order, seed, data)

    # --- 4. R1: варианты входов и seed ------------------------------------------------------------------
    runs: dict[str, dict] = {"main": {"run": main_run, "t1": t1_main, "t2": t2_main, "t3": t3_main}}
    for name, rr in alt_runs.items():
        runs[f"tracking:{name}"] = {
            "run": rr,
            "t2": _t2(rr, spec, rank0, order, seed, data),
            "t3": _t3(rr, spec, schemes),
        }
    for vname, (vspec, vin, vns) in variant_pre.items():
        t_v = time.perf_counter()
        rule = vspec.graph_rule or data.cp.graph_rule
        vctx = PL.RunCtx(
            name=f"variant:{vname}",
            ns=vns,
            base=vin,
            netp=netp,
            rule=rule,
            knn=data.cp.graph_k,
            k=k,
            alpha=alpha,
            method=str(fin["method"]),
            param=fin["param"],
            seeds=seeds_fin,
            impl=fin["impl"],
            schemes=schemes,
            blind=blind,
        )
        key = RN._key("variant", vname, blind, RN.digest(vctx, final0, data.ids, sets_main), chash)
        vr = RN.cached(
            cache,
            key,
            lambda vctx=vctx: RN.full_run(vctx, final0, data.ids, sets_main, workers, "variant", cache),
        )
        runs[f"variant:{vname}"] = _r1_eval(vr, cx, ev, spec, rank0, order, seed, data)
        log.info("interpret: вариант %s — %.0f с", vname, time.perf_counter() - t_v)
    for s in [int(x) for x in rob["seeds"]]:
        if s == seed:
            continue
        t_s = time.perf_counter()
        seeds_s = tuple(s + i for i in range(len(seeds_fin)))
        sctx = PL.RunCtx(
            name=f"seed:{s}",
            ns=data.ns,
            base=data.base,
            netp=netp,
            rule=data.cp.graph_rule,
            knn=data.cp.graph_k,
            k=k,
            alpha=alpha,
            method=str(fin["method"]),
            param=fin["param"],
            seeds=seeds_s,
            impl=fin["impl"],
            schemes=schemes,
            blind=blind,
        )
        sets_s = PL.pseudo_sets(n_pairs, np.random.default_rng([s, 601]))
        key = RN._key("seed", s, blind, RN.digest(sctx, final0, data.ids, sets_s), chash)
        sr = RN.cached(
            cache,
            key,
            lambda sctx=sctx, sets_s=sets_s: RN.full_run(
                sctx, final0, data.ids, sets_s, workers, "seed", cache
            ),
        )
        ev_s = CX.evidence(cx, spec, s, 0)
        runs[f"seed:{s}"] = _r1_eval(sr, cx, ev_s, spec, rank0, order, s, data)
        log.info("interpret: seed %d — %.0f с", s, time.perf_counter() - t_s)
    t0 = tick("robustness", t0)
    r1 = _r1_summary(runs, spec, data, seed)

    # --- 5. T6, T7 ------------------------------------------------------------------------------------
    t1_final = r1["final_verdict"]["T1_ladder_external"]
    cap = spec["thesis_assembly"]["t1_cap_by_t5"].plain()
    t1_text_verdict = cap_t1(t1_final, t5_main.verdict, cap)
    words = t2s["direction_words"]
    word_confirmed, word_otherwise = str(words["t1_confirmed"]), str(words["otherwise"])
    direction = word_confirmed if t1_text_verdict == "confirmed" else word_otherwise
    t6 = _t6(data, cx, spec, seed, order)
    t7 = _t7(data, cx, spec, seed)
    t0 = tick("t6_t7", t0)

    # --- 6. Профиль, названия, примеры, дерево ---------------------------------------------------------
    prof, names, shares, examples, tree, fca, exceptions, naming_pkg, margin = _describe(
        data, cx, spec, seed, r1
    )
    t0 = tick("describe", t0)
    posthoc = _posthoc(data, cx, spec, types_t)

    # --- 7. Главный вывод -----------------------------------------------------------------------------
    tests = spec["tests"]
    t1_run = runs[r1["source_run"]["T1_ladder_external"]]["t1"]
    t2_run = runs[r1["source_run"]["T2_direction"]]["t2"]
    t3_run = runs[r1["source_run"]["T3_reliable_placebo"]]["t3"]
    main_scheme = "main"
    texts = {
        "T1_ladder_external": TX.t1_text(t1_run, tests["T1_ladder_external"]["outcomes"], t1_text_verdict),
        "T5_trivial": TX.t5_text(t5_main, tests["T5_trivial"]["outcomes"]),
        "T3_reliable_placebo": TX.t3_text(t3_run, tests["T3_reliable_placebo"]["outcomes"], main_scheme),
        "T2_direction": TX.t2_text(
            _with_verdict(t2_run, r1["final_verdict"]["T2_direction"]),
            tests["T2_direction"]["outcomes"],
            direction,
            main_scheme,
            int(t2s["min_pool"]),
        ),
        "T7_utility": TX.t7_text(t7, tests["T7_utility"]["outcomes"])
        + " "
        + TX.t7_type_text(t7, tests["T7_utility"]["outcomes"]),
        "T6_bank_coverage": TX.t6_text(
            t6, tests["T6_bank_coverage"]["outcomes"], tests["T6_bank_coverage"]["movers"]
        )
        + (
            " " + TX.t6_caveat(t6["coverage_eps2"], tests["T6_bank_coverage"]["outcomes"])
            if t6["coverage_eps2"] > float(tests["T6_bank_coverage"]["coverage_caveat"]["epsilon2_max"])
            else ""
        ),
        "T4_basket_vs_place": TX.t4_text(t4, str(tests["T4_basket_vs_place"]["text"])),
        "T1_proxies": TX.t1_proxies(proxies, tests["T1_ladder_external"]["outcomes"]),
    }
    # robustness.main_run_verdict: check_section — основной расчёт выше наименьшего вердикта:
    # его текст и числа —
    # только в разделе проверки с пометкой unstable_label (в главный вывод и в ограничения не идут)
    higher = r1["summary"]["main_higher"]
    if "T1_ladder_external" in higher:
        v1 = cap_t1(t1_main.verdict, t5_main.verdict, cap)
        texts["T1_main"] = TX.t1_text(t1_main, tests["T1_ladder_external"]["outcomes"], v1)
    if "T3_reliable_placebo" in higher:
        texts["T3_main"] = TX.t3_text(t3_main, tests["T3_reliable_placebo"]["outcomes"], main_scheme)
    if "T2_direction" in higher:
        texts["T2_main"] = TX.t2_text(
            t2_main, tests["T2_direction"]["outcomes"], direction, main_scheme, int(t2s["min_pool"])
        )
    # пометка «„за в целом“ не выполнено» рядом с текстом T1, если t1_cap_by_t5 понизил его до
    # partial_overall, а (a) ни у одного оборота не прошло rho_min (правило не меняется)
    t1_notes = {"T1_ladder_external": TX.t1_overall_note(t1_run, t1_text_verdict)}
    if "T1_main" in texts:
        t1_notes["T1_main"] = TX.t1_overall_note(t1_main, cap_t1(t1_main.verdict, t5_main.verdict, cap))
    ta = spec["thesis_assembly"]

    def _point(keys) -> list[str]:
        return [texts[str(x)] + t1_notes.get(str(x), "") for x in keys]

    thesis = {
        "question": str(ta["question"]),
        "point_1": _point(ta["point_1"]),
        "point_2": _point(ta["point_2"]),
        "point_3": _point(ta["point_3"]),
        "caveat": _point(ta["caveat"]),
        "excluded": [str(x) for x in ta["excluded"]],
    }
    edits = _edits(spec, t1_text_verdict, t1_run, t5_main, t3_run, t2_run, t7)

    # --- 8. Выходы ------------------------------------------------------------------------------------
    facts = {
        "blind": blind,
        "label": BLIND_LABEL if blind is not None else "",
        "scope": _scope(data, cx, spec, members),
        "ladder": {"order": order, "transfer": data.final_raw_numbers, "sizes_territorial": cx.sizes},
        "controls": cinfo,
        "verdicts_main": {
            "T1_ladder_external": t1_main.verdict,
            "T2_direction": t2_main.verdict,
            "T3_reliable_placebo": t3_main.verdict,
            "T5_trivial": t5_main.verdict,
            "T6_bank_coverage": t6["verdict"],
            "T7_utility": t7["verdict"],
        },
        "verdicts_final": {
            **r1["final_verdict"],
            "T1_text": t1_text_verdict,
            "T5_trivial": t5_main.verdict,
            "T6_bank_coverage": t6["verdict"],
            "T7_utility": t7["verdict"],
        },
        "r1": r1["summary"],
        "texts": texts,
        "t1_notes": t1_notes,
        "thesis": thesis,
        "edits": edits,
        "t1": _plain(t1_main),
        "t1_regions_up": region_up,
        "t5": _plain(t5_main),
        "t4": {k_: v for k_, v in t4.items() if k_ != "basket_labels"},
        "t3": _plain(
            {s: {k_: v for k_, v in asdict(r).items() if k_ != "placebo"} for s, r in t3_main.per.items()}
        ),
        "t3_verdict": t3_main.verdict,
        "t3_excess": {
            "excess": t3_main.excess,
            "share": t3_main.excess_share,
            "one_in_ten": t3_main.one_in_ten,
        },
        "t2": _plain({s: asdict(r) for s, r in t2_main.per.items()}),
        "t2_place": t2_main.place,
        "t2_describe": _plain(t2_main.describe),
        "t6": _plain({k_: v for k_, v in t6.items() if k_ != "per_flow"}),
        "t7": _plain({k_: v for k_, v in t7.items() if k_ not in ("common_pos", "sets", "node_comparable")}),
        "tree": {
            "balanced_accuracy": tree.balanced_accuracy,
            "accuracy": tree.accuracy,
            "baseline_majority_bacc": tree.baseline_majority_bacc,
            "baseline_majority_acc": tree.baseline_majority_acc,
            "baseline_random_bacc": tree.baseline_random_bacc,
        },
        "names": {int(t): asdict(v) for t, v in names.items()},
        "names_final": {str(t): v for t, v in naming_pkg["final"].items()},
        "names_descriptive": {str(t): v for t, v in naming_pkg["descriptive"].items()},
        "naming_test": naming_pkg["status"],
        "qc": {
            **qc_main,
            "freshness": data.fresh,
            "inputs_sha256": data.inputs_sha,
            "chain_direct": _chain_direct(data, runs),
        },
        "timing": timing,
        "workers": workers,
        "n_pairs": n_pairs,
    }
    write_csv(pd.DataFrame(pairwise), out / "t1_pairwise.csv")
    write_csv(_t1_table(runs), out / "t1_runs.csv")
    write_csv(
        pd.DataFrame([{"proxy": k_, **v} for k_, v in proxies.items()]).drop(columns=["values_text"]),
        out / "t1_proxies.csv",
    )
    write_csv(
        pd.DataFrame(
            [
                {"partition": k_, "label": cx.trivial[k_].label, "ami": v, "ari": t5_main.ari[k_]}
                for k_, v in t5_main.ami.items()
            ]
        ),
        out / "t5_ami.csv",
    )
    write_csv(pd.DataFrame(t5_main.composition["groups"]), out / "t5_composition.csv")
    write_csv(_placebo_table(runs), out / "t3_placebo.csv")
    write_csv(pd.DataFrame(t6["per_flow"]), out / "t6_flows.csv")
    write_csv(_t7_table(t7, cx), out / "t7_errors.csv")
    write_csv(r1["table"], out / "r1_runs.csv")
    write_csv(r1["signs"], out / "r1_basket_signs.csv")
    write_csv(prof, out / "profile.csv")
    write_csv(shares.reset_index(), out / "settlement_shares.csv")
    write_csv(examples, out / "examples.csv")
    write_csv(tree.rules, out / "tree_rules.csv")
    write_csv(fca, out / "fca.csv")
    write_csv(exceptions, out / "exceptions.csv")
    write_csv(posthoc, out / "posthoc.csv")
    write_json(naming_pkg["package"], out / "naming_test.json")  # для check-ux: без ключа
    # ключ ответа — отдельным файлом для того, кто сверяет ответ check-ux (check-ux его не получает);
    # итог сверки — {"correct": n, "package_sha": <из naming_test.json>} в файле naming_test.result_file
    write_json(naming_pkg["key"], naming_key_path(cfg, out))
    types_out = pd.DataFrame(
        {
            "territory_id": data.ids,
            "type": data.final,
            "step": data.rank_of(data.final),
            "name": [naming_pkg["final"][int(t)] for t in data.final],
            "name_descriptive": [naming_pkg["descriptive"][int(t)] for t in data.final],
        }
    )
    write_csv(types_out, out / "types.csv")
    # --- поузловые выгрузки для лендинга: описание, не проверка (в вердикты и главный вывод не входят) ---
    node_files = write_node_tables(out, data, cx, ev, runs, t1_main, t5_main, t7, margin, members, rob, seed)
    facts["node_tables"] = {"note": NT.NOTE, "files": node_files}
    facts["seconds"] = time.perf_counter() - t_start

    from munnet.interpret import report as R

    R.write(cfg, spec, out, report_path, img_dir, facts, blind)
    unread = spec.unread()
    if unread:
        raise QCError("interpret: ключи предрегистрации не прочитаны кодом: " + ", ".join(unread))
    # последним шагом: полный facts.json со status = done — все выходы и отчёт этого прогона уже записаны
    facts["status"] = STATUS_DONE
    write_json(_plain(facts), out / "facts.json")
    log.info(
        "interpret: готово за %.0f с%s",
        time.perf_counter() - t_start,
        f" ({BLIND_LABEL})" if blind is not None else "",
    )


# --- Вспомогательные шаги -----------------------------------------------------------------------------


def _with_verdict(res: PL.T2Result, verdict: str) -> PL.T2Result:
    """Текст T2 по вердикту R1: у прогона-источника вердикт совпадает; ключ текста — по правилу place_cap."""
    if verdict == res.verdict:
        return res
    raise QCError(f"interpret: вердикт T2 прогона-источника {res.verdict} ≠ {verdict}")


def _proxies(cx: CX.Ctx, ranks: np.ndarray, spec: Spec, k: int) -> dict:
    """T1.proxies — описание: доли узлов (статус, столица) или медианы (плотность) по ступеням."""
    from munnet import style
    from munnet.interpret import stats as S

    pr = spec["tests"]["T1_ladder_external"]["proxies"]
    out = {}
    steps = list(range(1, k + 1))
    for name, d in pr.items():
        name = str(name)
        if name == "city_status":
            if str(d["source"]) != "territories.mo_type" or str(d["kind"]) != "binary":
                raise ValueError("T1.proxies.city_status: реализовано territories.mo_type, binary")
            v = (cx.mo_type == str(d["positive"])).astype(float)
            vals = [float(v[ranks == s].mean()) for s in steps]
            text = [style.fmt_pct(x, 0) for x in vals]
        elif name == "capital":
            if str(d["source"]) != "territories.is_capital" or str(d["kind"]) != "binary":
                raise ValueError("T1.proxies.capital: реализовано territories.is_capital, binary")
            v = cx.capital.astype(float)
            vals = [float(v[ranks == s].mean()) for s in steps]
            text = [style.fmt_pct(x, 1) for x in vals]
        elif name == "density":
            vals = S.medians_by(cx.density[np.isfinite(cx.density)], ranks[np.isfinite(cx.density)], steps)
            text = [style.fmt_num(x, 2) for x in vals]
        else:
            raise ValueError(f"T1.proxies.{name}: не реализовано")
        sign = int(d["sign"])
        mono = S.monotone_ok([sign * x for x in vals])
        out[name] = {"label": PROXY_LABELS[name], "values": vals, "values_text": text, "monotone": mono}
    return out


def _t4(data: D.Data, cx: CX.Ctx, ev: O.Evidence, spec: Spec, seed: int, types_t: np.ndarray) -> dict:
    """T4 — описание: ARI итога с разбиением «только корзина» и «только место», разность и интервал."""
    from sklearn.metrics import adjusted_rand_score

    t4 = spec["tests"]["T4_basket_vs_place"]
    bs = t4["basket"]
    if str(bs["method"]) != "spectral_embedding_kmeans" or str(bs["graph"]) != "network_edges.basket_dist":
        raise ValueError("T4.basket: реализовано spectral_embedding_kmeans по network_edges.basket_dist")
    if data.cp.graph_rule != "basket_dist":
        raise ValueError("T4.basket: G итога — не basket_dist")
    dim, kk = int(bs["dim"]), int(bs["k"])
    n_init = int(t4["n_init"])
    pl = t4["place"]
    feats = [str(x) for x in resolve(spec, str(pl["features"]))]
    A = data.base.A
    E = M.spectral_embed(A, dim, seed)
    basket = P.kmeans_labels(E, kk, seed, n_init)
    basket = PL.blind(basket, _blind_ctx(data), "t4-basket")
    place = [r for r in cx.rivals if r.name == "place_only"][0].labels
    fin_t = types_t
    b_t = basket[cx.pos]
    ari_b = float(adjusted_rand_score(fin_t, b_t))
    ari_p = float(adjusted_rand_score(fin_t, place))
    rng = np.random.default_rng([seed, 404])
    frac = float(t4["subsample"])
    n = len(data.ids)
    Zp = cx.scaled(feats, str(pl["scale"]))  # масштаб всей выборки — по всем узлам сети, как в кластеризации
    terr_pos = {int(p): i for i, p in enumerate(cx.pos)}
    diffs = []
    for _ in range(int(t4["bootstrap"])):
        idx = np.sort(rng.choice(n, int(round(frac * n)), replace=False))
        s_seed = int(rng.integers(0, 2**31 - 1))
        Eb = M.spectral_embed(A[idx][:, idx].tocsr(), dim, s_seed)
        bb = P.kmeans_labels(Eb, kk, s_seed, n_init)
        tpos = np.array([terr_pos[int(i)] for i in idx if int(i) in terr_pos])
        keep = np.array([int(i) in terr_pos for i in idx])
        pb = P.kmeans_labels(Zp[tpos], int(pl["k"]), s_seed, n_init)
        f = data.final[idx][keep]
        diffs.append(adjusted_rand_score(f, bb[keep]) - adjusted_rand_score(f, pb))
    from munnet.interpret import stats as S

    ci = S.perc_ci(np.array(diffs), float(t4["ci_level"]))
    desc = [str(x) for x in t4["describe"]]
    res = {"ari_basket": ari_b, "ari_place": ari_p, "diff": ari_b - ari_p, "ci": ci, "basket_labels": basket}
    # сверка утверждения T4.text «итог ближе к разбиению по корзине, чем по месту»:
    # знак ari_basket − ari_place
    res["diff_sign"] = int(np.sign(ari_b - ari_p))
    res["text_holds"] = bool(ari_b > ari_p)
    if not res["text_holds"]:
        log.warning(
            "interpret: T4.text утверждает «итог ближе к корзине, чем к месту», но ARI %.3f ≤ %.3f — "
            "в отчёте рядом с текстом пометка",
            ari_b,
            ari_p,
        )
    if "ari_log_pop_rel_quintiles" in desc:
        q = P.rank_quantiles(cx.raw["log_pop_rel"].to_numpy(dtype=np.float64), cx.ids, 5)
        ok = q >= 0
        res["ari_log_pop_rel_quintiles"] = float(adjusted_rand_score(fin_t[ok], q[ok]))
    if "basket_only_order_by_turnovers" in desc:
        mapping = T.match(data.final - 1, basket, len(data.order))
        bl = T.relabel(basket, mapping) + 1
        r = _t1(data.rank_of(bl[cx.pos]), ev, spec, np.random.default_rng([seed, 405]))
        res["basket_only_t1"] = r.verdict
        res["basket_only_rho_b"] = {n_: t.rho_b for n_, t in r.per.items()}
    return res


def _blind_ctx(data: D.Data) -> PL.RunCtx:
    return PL.RunCtx(
        name="main",
        ns=data.ns,
        base=data.base,
        netp=None,
        rule="",
        knn=0,
        k=data.k,
        alpha=0.0,
        method="",
        param=None,
        seeds=(),
        impl={},
        schemes={},
        blind=data.blind,
    )


def _t3(run: RN.RunOut, spec: Spec, schemes) -> PL.T3Result:
    t3 = spec["tests"]["T3_reliable_placebo"]
    obs = {s: run.observed[s].n for s in schemes}
    pl = {s: np.array([m.sum() for m in run.placebo[s]]) for s in schemes}
    return PL.t3_eval(
        obs, pl, run.n_nodes, float(t3["percentile"]), [float(x) for x in t3["one_in_ten"]], "main"
    )


def _t2(
    run: RN.RunOut, spec: Spec, rank0: np.ndarray, order: list[int], seed: int, data: D.Data
) -> PL.T2Result:
    t2 = spec["tests"]["T2_direction"]
    t3 = spec["tests"]["T3_reliable_placebo"]
    rng = np.random.default_rng([seed, 402, *PL.tag_key(run.name)])
    adj = [(order[i] - 1, order[i + 1] - 1) for i in range(len(order) - 1)]
    per = {}
    for s, obs in run.observed.items():
        per[s] = PL.t2_scheme(
            s,
            obs.t_a,
            obs.t_b,
            obs.reliable,
            run.placebo[s],
            rank0,
            float(t2["alpha"]),
            int(t2["min_pool"]),
            int(t2["bootstrap"]),
            float(t2["ci_level"]),
            rng,
            adj,
            float(t3["percentile"]),
        )
    pt = t2["place_tree"]
    feats = [str(x) for x in pt["features"]]
    names = list(run_names(run, data))
    cols = [names.index(f) for f in feats]
    o = run.observed["main"]
    place = PL.place_tree_test(
        run.X_place[:, cols],
        run.type_2023,
        o.t_a,
        o.t_b,
        o.reliable,
        int(pt["depth"]),
        str(pt["class_weight"]),
        int(pt["folds"]),
        int(pt["permutations"]),
        seed,
        rng,
    )
    verdict, key, capped = PL.t2_verdict(per, place["p"], float(pt["alpha"]))
    res = PL.T2Result(verdict=verdict, text_key=key, per=per, place=place, capped=capped)
    if run.kind == "main":
        # описания D1 и D2 (в вердикт не входят)
        lab = data.labels_dyn
        odd = {y: lab[(PL.MAIN, D.HALF_PERIODS[f"odd_{y}"])] - 1 for y in (2023, 2024)}
        even = {y: lab[(PL.MAIN, D.HALF_PERIODS[f"even_{y}"])] - 1 for y in (2023, 2024)}
        d1 = PL.d1_noise(odd, even, rank0, reverse=False)
        d1r = PL.d1_noise(odd, even, rank0, reverse=True)
        m = per["main"]
        d2 = PL.d2_permutation(o.t_a[o.reliable], o.t_b[o.reliable], rank0, int(t2["bootstrap"]), rng)
        res.describe = {
            "d1_noise_share": d1,
            "d1": m.share_up - d1,
            "d1_reverse_share": d1r,
            "d1_reverse": m.share_up - d1r,
            "d2": d2,
            "texts": t2["describe"].plain(),
            "data": str(t2["data"]),
        }
    else:
        res.describe = {}
    return res


def run_names(run: RN.RunOut, data: D.Data) -> tuple[str, ...]:
    """Имена столбцов X прогона (у варианта без уровня трат — без log_level_rel)."""
    n = run.X_place.shape[1]
    names = tuple(data.base.x_names)
    if n == len(names):
        return names
    return tuple(x for x in names if x != "log_level_rel")


def _r1_eval(
    vr: RN.RunOut, cx: CX.Ctx, ev: O.Evidence, spec: Spec, rank0, order, seed: int, data: D.Data
) -> dict:
    """Утверждения R1 прогона: знаки частей корзины, вердикты T1 (на территориальных узлах), T2, T3."""
    pos = {int(t): i for i, t in enumerate(vr.ids)}
    lab_t = np.array([vr.final[pos[int(t)]] for t in cx.ids]) + 1
    r1 = _t1(data.rank_of(lab_t), ev, spec, np.random.default_rng([seed, 403]))
    t3 = _t3(vr, spec, RN.schemes_of(spec["tests"]["T3_reliable_placebo"]))
    t2 = _t2(vr, spec, rank0, order, seed, data)
    return {"run": vr, "t1": r1, "t2": t2, "t3": t3}


def _chain_direct(data: D.Data, runs: dict) -> dict:
    """Сверка «цепочка окон против прямого сопоставления» по прогонам: основной — ``data.chain`` (при
    расхождении в окне 2024 года этап уже остановлен), варианты и seed — ``qc.chain_info`` их прогона."""
    out = {"main": dict(data.chain)}
    for name, r in runs.items():
        info = getattr(r["run"], "qc", {}).get("chain_info")
        if info is not None:
            out[name] = {k_: float(v) for k_, v in info.items()}
    return out


def cap_t1(t1_verdict: str, t5_verdict: str, cap: Mapping[str, str]) -> str:
    """``thesis_assembly.t1_cap_by_t5``: при вердикте T5 из ``cap`` текст T1 — не выше записанного уровня."""
    if t5_verdict in cap and TX.level_rank("T1_ladder_external", t1_verdict) > TX.level_rank(
        "T1_ladder_external", str(cap[t5_verdict])
    ):
        return str(cap[t5_verdict])
    return t1_verdict


def _signs(labels0: np.ndarray, B: np.ndarray, k: int) -> np.ndarray:
    med = np.median(B, axis=0)
    return np.array(
        [[np.sign(np.median(B[labels0 == t, j]) - med[j]) for j in range(B.shape[1])] for t in range(k)]
    )


def _r1_summary(runs: dict, spec: Spec, data: D.Data, seed: int) -> dict:
    """Фильтр R1: устойчивость знаков и вердиктов; вердикт главного вывода — наименьший среди основного
    расчёта, вариантов, способов прослеживания и 4-го по величине из 5 seed (``main_text: lowest``)."""
    rob = spec["robustness"]
    k = data.k
    variants = [f"variant:{v}" for v in rob["variants"]]
    tracking = [f"tracking:{t}" for t in rob["tracking"]]
    seeds = [int(s) for s in rob["seeds"]]
    seed_runs = ["main" if s == seed else f"seed:{s}" for s in seeds]
    need = int(rob["seeds_required"])
    main = runs["main"]["run"]
    base_signs = _signs(main.final, main.basket, k)
    sign_rows = []
    for name in ["main", *variants, *[r for r in seed_runs if r != "main"]]:
        rr = runs[name]["run"]
        sg = _signs(rr.final, rr.basket, k)
        for t in range(k):
            for j, part in enumerate(data.base.b_names):
                sign_rows.append(
                    {
                        "run": name,
                        "type": t + 1,
                        "part": part,
                        "sign": int(sg[t, j]),
                        "main": int(base_signs[t, j]),
                    }
                )
    signs = pd.DataFrame(sign_rows)
    stable = []
    for (t, part), g in signs.groupby(["type", "part"]):
        same = g.set_index("run")["sign"] == g.set_index("run")["main"]
        ok_var = all(same[v] for v in variants)
        ok_seed = sum(bool(same[r]) for r in seed_runs) >= need
        stable.append({"type": int(t), "part": part, "stable": bool(ok_var and ok_seed)})
    stable = pd.DataFrame(stable)
    rows = []
    final_verdict, source = {}, {}
    for test, key, allowed in (
        ("T1_ladder_external", "t1", ["main", *variants]),
        ("T2_direction", "t2", ["main", *variants, *tracking]),
        ("T3_reliable_placebo", "t3", ["main", *variants, *tracking]),
    ):
        verdicts = {r: runs[r][key].verdict for r in [*allowed, *seed_runs] if key in runs[r]}
        seed_v = sorted((verdicts[r] for r in seed_runs), key=lambda v: -TX.level_rank(test, v))
        fourth = seed_v[need - 1]
        cand = [verdicts[r] for r in allowed] + [fourth]
        low = TX.level_min(test, cand)
        final_verdict[test] = low
        order_runs = ["main", *variants, *tracking, *seed_runs]
        src = next(r for r in order_runs if r in verdicts and verdicts[r] == low)
        source[test] = src
        main_v = verdicts["main"]
        stable_flag = all(
            TX.level_rank(test, verdicts[r]) >= TX.level_rank(test, main_v) for r in allowed
        ) and (
            sum(TX.level_rank(test, verdicts[r]) >= TX.level_rank(test, main_v) for r in seed_runs) >= need
        )
        for r, v in verdicts.items():
            rows.append({"test": test, "run": r, "verdict": v})
        rows.append({"test": test, "run": "итог (main_text: lowest)", "verdict": low})
        rows.append({"test": test, "run": "устойчив к R1", "verdict": "да" if stable_flag else "нет"})
    unstable = {(int(r["type"]), str(r["part"])) for _, r in stable.iterrows() if not r["stable"]}
    summary = {
        "final_verdict": final_verdict,
        "source_run": source,
        "unstable_parts": sorted([f"{t}:{p}" for t, p in unstable]),
        "main_higher": {
            t: runs["main"][k_].verdict
            for t, k_ in (("T1_ladder_external", "t1"), ("T2_direction", "t2"), ("T3_reliable_placebo", "t3"))
            if TX.level_rank(t, runs["main"][k_].verdict) > TX.level_rank(t, final_verdict[t])
        },
        "unstable_label": str(rob["unstable_label"]),
        "circularity": str(rob["circularity"]),
    }
    return {
        "final_verdict": final_verdict,
        "source_run": source,
        "table": pd.DataFrame(rows),
        "signs": signs.merge(stable, on=["type", "part"]),
        "unstable": unstable,
        "summary": summary,
    }


def _t6(data: D.Data, cx: CX.Ctx, spec: Spec, seed: int, order: list[int]) -> dict:
    t6 = spec["tests"]["T6_bank_coverage"]
    ind = t6["indicator"]
    ya, yb = (int(y) for y in spec["windows"]["change_years"])
    groups = data.base.groups
    ids = data.ids
    va = CX.transform(CX.context_values(data, str(ind["source"]), ya, ids), str(ind["transform"]), groups)
    vb = CX.transform(CX.context_values(data, str(ind["source"]), yb, ids), str(ind["transform"]), groups)
    change = np.where(data.terr, vb - va, np.nan)
    nd = data.nodes_dyn
    ta = nd["half_type_2023"].to_numpy(dtype=np.int64)
    tb = nd["half_type_2024"].to_numpy(dtype=np.int64)
    rel = nd["reliable"].astype(bool).to_numpy()
    flows = [(int(a), int(b)) for a, b in t6["movers"]]
    res = EX.t6_eval(
        change,
        ta,
        tb,
        rel,
        flows,
        int(t6["permutations"]),
        int(t6["cliff_bootstrap"]),
        float(t6["alpha"]),
        float(t6["ci_level"]),
        float(t6["rule"]["cliff_lower_min"]),
        np.random.default_rng([seed, 406]),
    )
    # оговорка про охват безнала: ε² по типам для log_level_rel − log_ndfl_rel (узлы с ndfl_ok, place_year)
    py = int(spec["windows"]["place_year"])
    yi = py - 2023
    # уточнение реализации (не в предрегистрации): запись не уточняет окно log_level_rel, поэтому ε² считается
    # по обоим прочтениям — уровень окна place_year и log_level_rel признаков X (24 месяца), — и оговорка
    # дописывается, если хоть одно больше порога (строже любого одного прочтения)
    pl = data.place.loc[data.place["year"] == py].set_index("territory_id").reindex(ids)
    ndfl = pl["log_ndfl_rel"].to_numpy(dtype=np.float64)
    ok = pl["ndfl_ok"].astype(bool).to_numpy() & np.isfinite(ndfl)
    ok &= data.terr
    from munnet.interpret import stats as S

    levels = {
        "place_year": CI.window_level(data.ns, np.arange(12 * yi, 12 * yi + 12)),
        "months_24": CI.window_level(data.ns, np.arange(data.ns.V.shape[1])),
    }
    eps = {}
    for name, level in levels.items():
        m = ok & np.isfinite(level)
        eps[name] = float(S.epsilon2((level - ndfl)[m], data.final[m]))
    res["coverage_eps2_by_level"] = eps
    res["coverage_eps2"] = coverage_eps2_max(eps)
    res["coverage_n"] = int(ok.sum())
    return res


def coverage_eps2_max(eps: Mapping[str, float]) -> float:
    """ε² для решения об оговорке про охват: наибольшее из прочтений (пропуск — если все пропуски)."""
    vals = [float(v) for v in eps.values() if np.isfinite(v)]
    return max(vals) if vals else float("nan")


def _t7(data: D.Data, cx: CX.Ctx, spec: Spec, seed: int) -> dict:
    t7 = spec["tests"]["T7_utility"]
    win = spec["windows"]
    a0, a1 = (str(x) for x in win["year_a"])
    months = data.ns.months
    mm = months.loc[(months["date"] >= a0) & (months["date"] <= a1), "t"].to_numpy()
    basket_feats = [str(x) for x in t7["basket"]]
    B = CI.window_clr(data.ns, mm)
    lvl = CI.window_level(data.ns, mm)
    cols = {name: B[:, j] for j, name in enumerate(data.base.b_names)}
    cols["log_level_rel"] = lvl
    F_all = pd.DataFrame({f: cols[f] for f in basket_feats})  # все узлы сети (база масштаба)
    ya, yb = (int(y) for y in win["change_years"])
    tg, tga = t7["target"], t7["target_abs"]
    groups = data.base.groups
    ids = data.ids

    def target(tspec) -> np.ndarray:
        if str(tspec["change"]) != "diff":
            raise ValueError("T7.target.change: реализовано diff")
        va = CX.transform(
            CX.context_values(data, str(tspec["source"]), ya, ids), str(tspec["transform"]), groups
        )
        vb = CX.transform(
            CX.context_values(data, str(tspec["source"]), yb, ids), str(tspec["transform"]), groups
        )
        return (vb - va)[cx.pos]

    y, y_abs = target(tg), target(tga)
    known = np.isfinite(y) & np.isfinite(y_abs)
    # масштаб «как clustering.inputs.scale» — медиана и MAD по всем узлам сети, строки — МО с известной целью
    F = P.scale_rows(F_all, cx.pos[known], str(t7["scale"]))
    nd = data.nodes_dyn
    types23 = nd["type_2023"].to_numpy(dtype=np.int64)[cx.pos][known]
    dec = cx.trivial["pop_decile"].labels[known]
    sets = t7["sets"]
    rng = np.random.default_rng([seed, 407])
    st = EX.t7_sets(
        F,
        cx.xy[known],
        cx.groups[known],
        types23,
        dec,
        y[known],
        y_abs[known],
        int(t7["k"]),
        int(t7["min_set"]),
        int(sets["C"]["draws"]),
        rng,
    )
    res = EX.t7_eval(st, int(t7["bootstrap"]), float(t7["ci_level"]), float(t7["rule"]["lower_min"]), rng)
    bp = t7["beyond_place"]
    feats = [str(x) for x in resolve(spec, str(bp["features"]))]
    Xp = cx.scaled(feats)[known]  # масштаб — по всем узлам сети, как clustering.inputs.scale
    res["beyond_place"] = EX.beyond_place(
        y[known], Xp, types23, int(bp["folds"]), int(bp["repeats"]), int(bp["permutations"]), seed, rng
    )
    P_ = res["product"]
    common = res["common_pos"]
    ex = EX.median_example(st.err[P_][common], cx.ids[known][common])
    i = int(common[ex])
    kid = cx.ids[known]
    res["example"] = {
        "territory_id": int(kid[i]),
        "error": float(st.err[P_][i]),
        "members": [int(kid[j]) for j in st.members[P_][i]],
    }
    res["n_known"] = int(known.sum())
    res["known_ids"] = kid
    res["sets"] = st
    # поузловая выгрузка для лендинга (описание, не проверка): набор продукта (A или D) и B с расстояниями
    res["node_comparable"] = NT.comparable_table(
        kid, st.members, np.asarray(F, dtype=np.float64), cx.xy[known], str(P_), (str(P_), "B")
    )
    return res


def _describe(data: D.Data, cx: CX.Ctx, spec: Spec, seed: int, r1: dict):
    base = data.base
    pr = spec["profile"]
    feats = [str(x) for g in pr["features"] for x in data.cfg["features"]["space"][str(g)]]
    vals = pd.DataFrame(
        np.hstack([base.B, base.X_raw.to_numpy(dtype=np.float64)]), columns=[*base.b_names, *base.x_names]
    )
    vals = vals[feats].iloc[cx.pos].reset_index(drop=True)
    w = cx.pop23
    types_t = data.final[cx.pos]
    rng = np.random.default_rng([seed, 408])
    prof = DS.profile_table(vals, types_t, w, data.order, int(pr["bootstrap"]), float(pr["ci_level"]), rng)
    nm = spec["naming"]
    pl = (
        data.place.loc[data.place["year"] == int(spec["windows"]["place_year"])]
        .set_index("territory_id")
        .reindex(data.ids)
    )
    mo_all = base.table["mo_type"].astype(str).to_numpy()
    shares = DS.settlement_shares(
        data.final,
        mo_all,
        pl["pop_avg"].to_numpy(dtype=np.float64),
        pl["urban_share"].to_numpy(dtype=np.float64),
        nm["settlement_words"],
        data.order,
    )
    parts = [str(x) for x in nm["dictionary"].keys() if str(x).startswith("clr_rel_")]
    parts = [p for p in data.cfg["features"]["space"]["edges"] if p in parts]
    names = DS.name_types(
        prof, parts, cx.place_feats, nm, shares, data.order, r1["unstable"] if r1 else set()
    )
    banned = [str(b).lower() for b in nm["banned"]]
    for v in names.values():
        low = (v.name + " " + v.caption).lower()
        hit = [b for b in banned if b in low]
        if hit:
            raise QCError(f"interpret.naming: запрещённые слова {hit} в названии типа {v.type}")
    str(nm["assembly"])
    naming_pkg = _naming_test(prof, names, data, cx, spec, seed)
    st = naming_pkg["status"]
    # описательные названия naming.test.fallback считаются всегда и публикуются рядом с названиями по правилу
    descriptive = {int(t): DS.fallback_name(t, v, prof, nm) for t, v in names.items()}
    naming_pkg["descriptive"] = descriptive
    # уточнение реализации (не в предрегистрации): пока нет принятого ответа check-ux,
    # в выходах — описательные названия
    use_rule = bool(st["done"]) and not bool(st["fallback"])
    naming_pkg["final"] = {int(t): (v.name if use_rule else descriptive[int(t)]) for t, v in names.items()}
    # примеры
    exs = spec["examples"]
    E = M.spectral_embed(base.A, data.k, int(data.fin["best_seed"]))
    alpha = float(data.fin["hybrid_alpha"])
    Z = M.block_join([E, base.X], [alpha, 1 - alpha])
    regions = base.table["region_code"].to_numpy()
    eligible = data.terr.copy()
    typical = DS.typical_examples(
        Z,
        data.final,
        data.ids,
        regions,
        eligible,
        data.order,
        int(exs["typical"]),
        int(exs["borderline"]),
        True,
    )
    largest = DS.largest_examples(
        data.final,
        pl["pop_avg"].to_numpy(dtype=np.float64),
        data.ids,
        eligible,
        data.order,
        int(exs["largest"]),
    )
    nd = data.nodes_dyn
    dcols = [c for c in nd.columns if c.startswith("d_")]
    fl = exs["flows"]
    if not bool(fl["one_per_region"]):
        raise ValueError("examples.flows.one_per_region: реализовано true")
    flows = DS.flow_examples(
        nd[dcols].to_numpy(dtype=np.float64),
        nd["half_type_2023"].to_numpy(dtype=np.int64),
        nd["half_type_2024"].to_numpy(dtype=np.int64),
        nd["reliable"].astype(bool).to_numpy(),
        data.ids,
        regions,
        eligible,
        int(fl["min_nodes"]),
        int(fl["per_flow"]),
        True,
    )
    flows["kind"] = "flow"
    allx = pd.concat([typical, largest, flows], ignore_index=True)
    tab = base.table.reset_index(drop=True)
    names_mo = display_names(tab).to_numpy()
    ctx23 = data.context.loc[data.context["year"] == int(spec["windows"]["place_year"])].set_index(
        "territory_id"
    )
    wb = ctx23.reindex(data.ids)["workplace_based"].fillna(False).astype(bool).to_numpy()
    thr = float(exs["unstable_type_jaccard"])
    allx["territory_id"] = data.ids[allx["pos"].to_numpy(dtype=np.int64)]
    allx["name"] = names_mo[allx["pos"].to_numpy(dtype=np.int64)]
    allx["region"] = tab["region_name"].astype(str).to_numpy()[allx["pos"].to_numpy(dtype=np.int64)]
    allx["workplace_based"] = wb[allx["pos"].to_numpy(dtype=np.int64)]
    tcol = allx["type"].where(allx["type"].notna(), allx["source"]).astype(float)
    allx["type_unstable"] = [bool(data.type_jaccard[int(t)] < thr) for t in tcol]
    allx["pop_avg"] = pl["pop_avg"].to_numpy(dtype=np.float64)[allx["pos"].to_numpy(dtype=np.int64)]
    # дерево и формальные понятия
    tr = spec["tree"]
    tfeats = [str(x) for g in tr["features"] for x in data.cfg["features"]["space"][str(g)]]
    Mx = base.tree_matrix
    tnames = list(base.tree_names)
    cols = [tnames.index(f) for f in tfeats]
    filled, _ = CI.impute_region_median(base.X_raw, base.groups)
    inverse = {}
    for f in base.x_names:
        v = filled[f].to_numpy(dtype=np.float64)
        med = float(np.median(v))
        mad = float(np.median(np.abs(v - med)) * CI.MAD_SCALE) or float(np.std(v)) or 1.0
        inverse[f] = (med, mad)
    tree = DS.tree_describe(
        Mx[:, cols],
        data.final,
        [tnames[c] for c in cols],
        inverse,
        int(tr["depth"]),
        str(tr["class_weight"]),
        int(tr["folds"]),
        seed,
        [int(x) for x in tr["rules_per_type"]],
        True,
    )
    fc = tr["fca"]
    fca = DS.fca_implications(
        Mx[:, cols],
        data.final,
        [tnames[c] for c in cols],
        data.order,
        int(fc["max_itemset"]),
        float(fc["precision_min"]),
        float(fc["coverage_min"]),
        int(fc["bootstrap"]),
        np.random.default_rng([seed, 409]),
    )
    fca["in_text"] = fca["stability"] >= float(fc["stable_share"])
    # исключения: силуэт < 0 в пространстве гибрида или ошибка дерева вне фолда
    from sklearn.metrics import silhouette_samples

    ec = exs["exceptions"]
    sil = silhouette_samples(Z, data.final)
    exc = (sil < float(ec["silhouette_below"])) | (tree.oof != data.final)
    rows = []
    for t in data.order:
        m = (data.final == t) & data.terr
        rows.append(
            {
                "type": t,
                "kind": "share",
                "value": float(exc[m].mean()),
                "territory_id": np.nan,
                "name": "",
                "region": "",
            }
        )
        idx = np.flatnonzero(m & exc)
        popv = pl["pop_avg"].to_numpy(dtype=np.float64)[idx]
        top = idx[np.lexsort((data.ids[idx], -np.nan_to_num(popv, nan=-1)))][: int(ec["top"])]
        for i in top:
            rows.append(
                {
                    "type": t,
                    "kind": "largest_exception",
                    "value": float(pl["pop_avg"].to_numpy()[i]),
                    "territory_id": int(data.ids[i]),
                    "name": names_mo[i],
                    "region": str(tab.loc[i, "region_name"]),
                }
            )
    # пограничность каждого узла по тому же правилу (описание, не проверка) — node_margin.csv для лендинга
    margin = DS.margin_table(Z, data.final, data.ids, data.order)
    margin["is_city_node"] = ~data.terr
    return prof, names, shares, allx, tree, fca, pd.DataFrame(rows), naming_pkg, margin


def naming_result_path(cfg_outputs: Path, report: Path, blind: int | None) -> Path:
    """Файл ответа check-ux. Настоящий прогон — рядом с отчётом этапа
    (``docs/interpretation_naming_test.json``, в git: чистый клон берёт тот же ответ); слепой — в своём
    каталоге выходов."""
    if blind is not None:
        return cfg_outputs / "interpret_blind" / "naming_test_result.json"
    return report.with_name(report.stem + "_naming_test.json")


def naming_key_path(cfg: Config, out: Path) -> Path:
    """Ключ слепой проверки названий — вне каталога выходов, который читает check-ux:
    ``paths.interim/<каталог выходов>/naming_test_key.json`` (interim не в git)."""
    d = cfg.dir("interim") / out.name
    d.mkdir(parents=True, exist_ok=True)
    return d / "naming_test_key.json"


def naming_answer(res_path: Path, package_sha: str, pass_min: int) -> dict:
    """Ответ check-ux ``{"correct": n, "package_sha": …}``: принимается, только если ``package_sha`` совпадает
    с пакетом этого прогона. Нет файла — «нет ответа»; другой sha или нет поля — «устарел»: ответ не
    применяется, в выходах описательные названия (старый ответ к новым названиям не подмешивается)."""
    st: dict[str, Any] = {"result_file": str(res_path), "done": False, "fallback": False}
    st["state"] = "no_answer"
    if not res_path.exists():
        return st
    got = json.loads(res_path.read_text(encoding="utf-8"))
    if str(got.get("package_sha", "")) != package_sha:
        log.warning(
            "interpret: ответ check-ux %s устарел (package_sha %s, у пакета %s) — не применяется, "
            "в выходах описательные названия",
            res_path,
            got.get("package_sha"),
            package_sha,
        )
        st["state"] = "stale"
        st["stale_sha"] = str(got.get("package_sha", ""))
        return st
    correct = int(got["correct"])
    st.update({"done": True, "correct": correct, "fallback": correct < pass_min, "state": "accepted"})
    return st


def _naming_test(prof: pd.DataFrame, names: dict, data: D.Data, cx: CX.Ctx, spec: Spec, seed: int) -> dict:
    """Пакет слепой проверки названий для check-ux (``naming.test``): 4 названия и 4 профиля без названий
    и номеров, перемешанные seed. Ответ check-ux — ``naming_answer`` (файл ``naming_result_path``); меньше
    pass_min — все названия заменяются описательными (``fallback``).
    Уточнение реализации: пока принятого ответа нет, названия помечаются «не прошли слепую проверку»."""
    test = spec["naming"]["test"]
    who = str(test["who"])
    pass_min = int(test["pass_min"])
    rng = np.random.default_rng([seed, 410])
    order = list(data.order)
    perm = [order[i] for i in rng.permutation(len(order))]
    mo = cx.mo_type
    types_t = data.final[cx.pos]
    profiles = []
    for j, t in enumerate(perm):
        p = prof.loc[prof["type"] == t]
        m = types_t == t
        profiles.append(
            {
                "profile": f"П{j + 1}",
                "features": {
                    r["feature"]: {"median": r["median"], "q25": r["q25"], "q75": r["q75"]}
                    for _, r in p.iterrows()
                },
                "mo_type_shares": {str(v): float(np.mean(mo[m] == v)) for v in np.unique(mo)},
            }
        )
    name_list = [names[t].name for t in order]
    name_list = [name_list[i] for i in rng.permutation(len(name_list))]
    key = {f"П{j + 1}": names[t].name for j, t in enumerate(perm)}
    package: dict[str, Any] = {"names": name_list, "profiles": profiles}
    # sha пакета (названия и профили), без ключа: при 4! = 24 возможных ключах любой хеш с ключом
    # восстанавливался бы перебором. Ключ — функция пакета (порядок профилей и названия по типам заданы им),
    # поэтому ответ к другому пакету по-прежнему не применяется
    package["package_sha"] = _sha(package)
    res_path = naming_result_path(Path(data.cfg["paths"]["outputs"]), Path(str(spec["report"])), data.blind)
    status = {"who": who, "pass_min": pass_min, **naming_answer(res_path, package["package_sha"], pass_min)}
    return {"package": package, "status": status, "key": key}


def _sha(obj: Any) -> str:
    import hashlib

    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def _posthoc(data: D.Data, cx: CX.Ctx, spec: Spec, types_t: np.ndarray) -> pd.DataFrame:
    """Разведка после регистрации (``posthoc``): ε² Краскела — Уоллиса по типам для отгрузки и инвестиций
    на жителя (log1p_rel), без p-значений; в главный вывод, названия и лендинг не идёт."""
    from munnet.interpret import stats as S

    str(spec["posthoc"])
    rows = []
    c = cx.ctx23
    pop = c["pop_avg"].to_numpy(dtype=np.float64)
    for name, v in (
        ("shipments_pc", c["shipments_pc"].to_numpy(dtype=np.float64)),
        ("invest_pc", c["invest_krub"].to_numpy(dtype=np.float64) * 1000.0 / pop),
    ):
        x = P.log1p_rel(v, cx.groups)
        ok = np.isfinite(x)
        rows.append(
            {
                "indicator": name,
                "n": int(ok.sum()),
                "epsilon2": S.epsilon2(x[ok], types_t[ok]),
                "posthoc": True,
            }
        )
    return pd.DataFrame(rows)


def check_variant_names(names: list[str], known: list[str]) -> None:
    """robustness.variants — только варианты clustering.selection.sensitivity.variants (ValueError иначе)."""
    missing = [v for v in names if v not in set(known)]
    if missing:
        raise ValueError(f"robustness.variants: {missing} нет в clustering.selection.sensitivity.variants")


def prepare_variants(cfg: Config, data: D.Data, names: list[str]) -> dict:
    """Варианты R1 в начале прогона: сверка имён и входы каждого варианта (сеть другого правила, X без
    признака, узлы другого режима) — отсутствующий вход роняет прогон до контролей, а не после плацебо."""
    cpar = ClusterParams.from_config(cfg)
    specs = {v.name: v for v in cpar.variants}
    check_variant_names(names, list(specs))
    out = {}
    for vname in names:
        vs = specs[vname]
        vin = CI.variant_inputs(cfg, data.cp, data.main, vs)
        vns = data.ns
        if vs.nodes_mode and vs.nodes_mode != data.ns.mode:
            vns, _ = CI.load_mode_nodes(cfg, vs.nodes_mode)
        out[vname] = (vs, vin, vns)
    return out


def load_members(cfg: Config) -> pd.DataFrame | None:
    """``features_members`` (МО -> узел сети): охват фразы и сравнение узлов варианта nodes_separate
    с узлами-городами. Нет файла — None (каждое МО своим узлом, с предупреждением)."""
    path = cfg.dir("processed") / "features_members.parquet"
    if not path.exists():
        log.warning("interpret: нет %s — МО без типа считаются без узлов-городов", path)
        return None
    return pd.read_parquet(path, columns=["territory_id", "node_id", "role"])


def write_node_tables(
    out: Path,
    data: D.Data,
    cx: CX.Ctx,
    ev: O.Evidence,
    runs: dict,
    t1: O.T1Result,
    t5: O.T5Result,
    t7: dict,
    margin: pd.DataFrame,
    members: pd.DataFrame | None,
    rob: Spec,
    seed: int,
) -> list[str]:
    """Поузловые выгрузки для лендинга (``docs/landing_spec.md``, §5): только уже посчитанное — метки
    прогонов R1 из ``runs`` (кэш тяжёлых шагов), наборы T7, расстояния до центроидов правила examples,
    деления-соперники. Описание, не проверка: в вердикты, ``thesis_assembly`` и проверки не входят."""
    files = []
    write_csv(t7["node_comparable"], out / "node_comparable.csv")
    files.append("node_comparable.csv")
    r1_runs = {}
    for v in rob["variants"]:
        name = f"variant:{v}"
        r1_runs[name] = ("variant", runs[name]["run"].ids, runs[name]["run"].final + 1)
    for s in [int(x) for x in rob["seeds"]]:
        if s != seed:
            name = f"seed:{s}"
            r1_runs[name] = ("seed", runs[name]["run"].ids, runs[name]["run"].final + 1)
    node_map = {}
    if members is not None:
        mm = members.dropna(subset=["node_id"])
        node_map = dict(zip(mm["territory_id"].astype(int), mm["node_id"].astype(int), strict=True))
    long, summ = NT.r1_tables(data.ids, data.final, r1_runs, node_map)
    write_csv(long, out / "node_r1.csv")
    write_csv(summ, out / "node_seed.csv")
    write_csv(margin, out / "node_margin.csv")
    files += ["node_r1.csv", "node_seed.csv", "node_margin.csv"]
    by_name = {p.name: p for p in ev.rivals}
    rivals = [("T5:max_ami", cx.trivial[t5.max_partition])]
    for turn, r in t5.per.items():
        if r.best_rival in by_name:
            rivals.append((f"T5:best_rival:{turn}", by_name[r.best_rival]))
    for turn, r in t1.per.items():
        if r.best_rival in by_name:
            rivals.append((f"T1:best_rival:{turn}", by_name[r.best_rival]))
    write_csv(NT.rival_table(cx.ids, rivals), out / "node_rival.csv")
    files.append("node_rival.csv")
    return files


def _scope(data: D.Data, cx: CX.Ctx, spec: Spec, members: pd.DataFrame | None) -> dict:
    sc = spec["scope"]
    ter = data.territories
    # «не типизированы» во фразе — МО без типа ни у самих, ни через свой узел-город (features_members):
    # внутригородские МО Москвы и Петербурга с неполным рядом типизированы через город и в счёт не входят
    partial_series = (ter["series_status"].astype(str) != "full").to_numpy()
    untyped = NT.untyped_mask(ter["territory_id"].to_numpy(), members, data.ids)
    n_partial = int(untyped.sum())
    n_regions = int(pd.Series(data.base.table["region_code"]).nunique())
    phrase = TX.fmt(str(sc["phrase"]), n_nodes=len(data.ids), n_regions=n_regions, n_partial=n_partial)
    untyped_ids = ter["territory_id"].to_numpy()[untyped]
    pop = scope_population(ter, data.context, untyped_ids)
    if not pop["phrase_holds"]:
        log.warning(
            "interpret: scope.phrase утверждает «медиана населения у них ниже», но медиана pop_avg МО "
            "с неполным рядом %.0f не ниже, чем у территориальных узлов сети (полный ряд, без "
            "внутригородских МО) %.0f — в отчёте рядом с фразой пометка",
            pop["pop_median_partial"],
            pop["pop_median_full"],
        )
    tab = data.base.table.reset_index(drop=True)
    names = display_names(tab).to_numpy()
    city = [
        {"territory_id": int(data.ids[i]), "name": str(names[i]), "type": int(data.final[i])}
        for i in np.flatnonzero(~data.terr)
    ]
    return {
        "n_nodes": len(data.ids),
        "n_territorial": cx.n,
        "n_regions": n_regions,
        "n_partial": n_partial,  # МО без типа (ни сами, ни через узел-город) — число во фразе
        "n_untyped": n_partial,
        "n_partial_series": int(partial_series.sum()),  # справка: все МО с series_status ≠ full
        # неполный ряд, но тип есть — у узла-города; полный ряд без типа (ожидается 0)
        "n_partial_typed_via_city": int((partial_series & ~untyped).sum()),
        "n_untyped_full_series": int((~partial_series & untyped).sum()),
        "n_no_stratum": int((cx.strata < 0).sum()),  # в T1 (b) и T5 (3) не участвуют (strata.values: raw)
        "city_nodes": city,  # в проверках не участвуют, в таблицах — отдельной строкой (scope.city_nodes)
        "phrase": phrase,
        **pop,
    }


def scope_population(
    territories: pd.DataFrame, context: pd.DataFrame, partial_ids: np.ndarray | None = None
) -> dict:
    """Сверка фразы scope.phrase «медиана населения у них ниже»: медиана pop_avg МО без типа
    (``partial_ids`` — те же МО, что n_partial фразы; не задано — все МО с series_status ≠ full) против
    территориальных узлов сети — МО с полным рядом (series_status = full)
    без внутригородских МО Москвы и Петербурга (is_inner_city): те не типизированы, в сети они свёрнуты
    в узлы-города. Без весов. Для справки публикуется и медиана всех МО с полным рядом (с внутригородскими),
    в сверку она не входит. pop_avg МО — среднее по годам context_annual, где оно есть (у МО только одного
    года — значение этого года); МО без населения не участвуют, их число публикуется."""
    pop = context.groupby("territory_id")["pop_avg"].mean()
    t = territories[["territory_id", "series_status"]].copy()
    t["pop"] = t["territory_id"].map(pop)
    if "is_inner_city" in territories:
        inner = territories["is_inner_city"].astype("boolean").fillna(False).astype(bool).to_numpy()
    else:
        inner = np.zeros(len(t), dtype=bool)
    full_all = (t["series_status"].astype(str) == "full").to_numpy()
    terr = full_all & ~inner
    ok = (t["pop"].notna() & (t["pop"] > 0)).to_numpy()
    part = ~full_all if partial_ids is None else t["territory_id"].isin(partial_ids).to_numpy()
    m_terr = float(t.loc[terr & ok, "pop"].median())
    m_full_all = float(t.loc[full_all & ok, "pop"].median())
    m_part = float(t.loc[part & ok, "pop"].median())
    return {
        "pop_median_full": m_terr,  # группа сравнения — территориальные узлы сети
        "pop_full_group": (
            "территориальные узлы сети (полный ряд, без внутригородских МО Москвы и Петербурга)"
        ),
        "pop_median_full_all": m_full_all,  # справка: все МО с полным рядом, с внутригородскими
        "pop_median_partial": m_part,
        "pop_n_full": int((terr & ok).sum()),
        "pop_n_full_all": int((full_all & ok).sum()),
        "pop_n_inner_full": int((full_all & inner).sum()),
        "pop_n_partial": int((part & ok).sum()),
        "pop_n_missing": int((~ok).sum()),
        "phrase_holds": bool(m_part < m_terr),
    }


def _edits(
    spec: Spec, t1v: str, t1r: O.T1Result, t5: O.T5Result, t3: PL.T3Result, t2: PL.T2Result, t7: dict
) -> list[str]:
    """Правила для отчёта и лендинга (поля edits выбранных исходов) — в главный вывод не входят."""
    tests = spec["tests"]
    out = []
    o1 = tests["T1_ladder_external"]["outcomes"][t1v]
    if "edits" in o1:
        e = str(o1["edits"])
        if "{turnover_beyond}" in e:
            from munnet.interpret.labels import TURNOVER_GEN

            n = [k for k, v in t1r.per.items() if v.beyond] or list(t1r.per)
            e = e.replace("{turnover_beyond}", TURNOVER_GEN[n[0]])
        out.append(e)
    e5 = TX.t5_edits(t5, tests["T5_trivial"]["outcomes"])
    if e5:
        out.append(e5)
    o3 = tests["T3_reliable_placebo"]["outcomes"][t3.verdict]
    if "edits" in o3:
        out.append(str(o3["edits"]))
    o2 = tests["T2_direction"]["outcomes"][t2.text_key]
    if "edits" in o2:
        out.append(str(o2["edits"]))
    o7 = tests["T7_utility"]["outcomes"]
    if "edits" in o7[t7["verdict"]]:
        out.append(str(o7[t7["verdict"]]["edits"]))
    gk = {"adds": "type_adds", "neutral": "type_neutral", "hurts": "type_hurts"}[t7["type_gain"]]
    if "edits" in o7[gk]:
        out.append(str(o7[gk]["edits"]))
    return out


def _t1_table(runs: dict) -> pd.DataFrame:
    rows = []
    for name, r in runs.items():
        if "t1" not in r:
            continue
        for n, t in r["t1"].per.items():
            rows.append(
                {
                    "run": name,
                    "turnover": n,
                    "verdict": r["t1"].verdict,
                    "n_a": t.n_a,
                    "n_b": t.n_b,
                    "rho_a": t.rho_a,
                    "rho_a_lo": t.rho_a_ci[0],
                    "rho_a_hi": t.rho_a_ci[1],
                    "q_a": t.q_a,
                    "mono_a": t.mono_a,
                    "rho_b": t.rho_b,
                    "rho_b_lo": t.rho_b_ci[0],
                    "rho_b_hi": t.rho_b_ci[1],
                    "q_b": t.q_b,
                    "mono_b": t.mono_b,
                    "diff": t.diff_point,
                    "diff_lo": t.diff_ci[0],
                    "diff_hi": t.diff_ci[1],
                    "best_rival": t.best_rival,
                    "best_rival_rho": t.best_rival_rho,
                    "overall": t.overall,
                    "beyond": t.beyond,
                    "beyond_strict": t.beyond_strict,
                    "verdict_strict": r["t1"].verdict_strict,
                    **{f"median_a_{i + 1}": m for i, m in enumerate(t.med_a)},
                    **{f"median_b_{i + 1}": m for i, m in enumerate(t.med_b)},
                }
            )
    return pd.DataFrame(rows)


def _placebo_table(runs: dict) -> pd.DataFrame:
    rows = []
    for name, r in runs.items():
        run = r["run"]
        for s, mats in run.placebo.items():
            for i, m in enumerate(mats):
                rows.append({"run": name, "scheme": s, "pair": i, "n_reliable": int(np.asarray(m).sum())})
            rows.append({"run": name, "scheme": s, "pair": -1, "n_reliable": run.observed[s].n})
    return pd.DataFrame(rows)


def _t7_table(t7: dict, cx: CX.Ctx) -> pd.DataFrame:
    st = t7["sets"]
    ids = t7["known_ids"]
    df = pd.DataFrame({"territory_id": ids})
    for s in "ABCD":
        df[f"err_{s}"] = st.err[s]
        df[f"err_abs_{s}"] = st.err_abs[s]
        df[f"size_{s}"] = st.sizes[s]
    df["common"] = False
    df.loc[t7["common_pos"], "common"] = True
    return df
