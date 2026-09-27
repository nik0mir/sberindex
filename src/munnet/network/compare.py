"""Сравнение правил рёбер одними измерениями, выбор основного правила, режимы узлов и окна динамики.

Измерения каждого правила (сеть — разрежение ``network.sparsify`` на 24 месяцах):

1. паспорт сети (``graph.passport``);
2. надёжность — Жаккар рёбер сетей 2023 и 2024 годов (те же правило и разрежение по месяцам каждого года),
   ρ Спирмена сходств всех пар двух лет, бутстрап месяцев внутри года (корзина и корреляция: у лагов и DTW
   бутстрап ломает порядок месяцев);
3. отличие от географии — Жаккар с сетью ``road``, доля рёбер внутри группы региона;
4. согласованность с атрибутами места — ассортативность Ньюмена по признакам, не входящим в правило;
5. зонд кластеров — Leiden без весов, модульность против сетей с перемешанными рёбрами (степени те же),
   AMI сообществ с группой региона;
6. правила по рядам — значимость пар против циклического сдвига с поправкой Бенджамини — Хохберга;
7. время расчёта матрицы сходства.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from munnet.network import graph as G
from munnet.network.compute import GeoCache, Similarity, qvalues, series_null, similarity, spearman_similarity
from munnet.network.data import NodeSet, window_clr
from munnet.network.params import NetworkParams, RuleSpec

log = logging.getLogger(__name__)

ALL_MONTHS = np.arange(24)
YEAR_MONTHS: dict[int, np.ndarray] = {2023: np.arange(12), 2024: np.arange(12, 24)}
CHANCE_REPEATS = 5  # перестановок узлов для случайного уровня Жаккара
GRID_SEEDS = 3  # seed и перемешиваний зонда в таблице чувствительности к k

# Смысл ребра одной фразой для неспециалиста (по виду правила).
MEANING: dict[str, str] = {
    "basket_cosine": "жители тратят деньги похоже, если сравнивать каждое МО со своим регионом: "
    "одни и те же статьи выше или ниже, чем у соседей по региону",
    "basket_distance": "корзины двух МО почти одинаково отличаются от корзин своих регионов — "
    "и по направлению, и по силе отличия",
    "rhythm_corr": "траты в двух МО поднимаются и опускаются в одни и те же месяцы сверх общего для страны "
    "ритма",
    "rhythm_lag": "траты одного МО повторяют колебания другого, возможно, с задержкой до нескольких месяцев",
    "rhythm_dtw": "у двух МО одинаковая форма колебаний трат, даже если пики сдвинуты на месяц-два",
    "road": "МО близки по дороге",
    "gravity": "между МО сильное «притяжение»: оба крупные и недалеко друг от друга",
}


def sparsify(S: np.ndarray, method: str, k: int, Q: np.ndarray | None, fdr_q: float) -> pd.DataFrame:
    """Рёбра по способу разрежения. ``threshold``: у правил по рядам — пары с q < fdr_q (значимые против
    циклического сдвига), у остальных — самые похожие пары в числе рёбер kNN того же k (порог-квантиль)."""
    if method == "knn":
        return G.knn_edges(S, k)
    if method == "mutual_knn":
        return G.knn_edges(S, k, mutual=True)
    if Q is not None:
        return G.threshold_edges(S, Q < fdr_q)
    return G.threshold_edges(S, G.top_pairs_mask(S, len(G.knn_edges(S, k))))


def chance_jaccard(a: pd.DataFrame, b: pd.DataFrame, n: int, rng: np.random.Generator) -> float:
    """Жаккар двух сетей при случайной перестановке узлов второй (те же размеры и степени)."""
    base = G.edge_set(a)
    vals = []
    for _ in range(CHANCE_REPEATS):
        perm = rng.permutation(n)
        vals.append(G.jaccard(base, G.edge_set(b, perm)))
    return float(np.mean(vals))


@dataclass
class RuleResult:
    spec: RuleSpec
    sim: Similarity
    years: dict[int, Similarity]
    edges: pd.DataFrame
    lists: np.ndarray
    Q: np.ndarray | None
    null: np.ndarray | None
    probe: G.ProbeResult
    row: dict = field(default_factory=dict)


def _q_for(rule: RuleSpec, ns: NodeSet, months, sim: Similarity, p: NetworkParams, rng) -> tuple:
    """q-значения пар и нулевое распределение (только правила по рядам; σ ядра DTW — того же окна)."""
    if not rule.is_series:
        return None, None
    null = series_null(rule, ns, months, p, rng, sim.sigma)
    return qvalues(sim.S, null), null


def evaluate_rule(
    rule: RuleSpec,
    ns: NodeSet,
    p: NetworkParams,
    geo_cache: GeoCache,
    geo_edges: pd.DataFrame | None,
    rng: np.random.Generator,
) -> RuleResult:
    """Все измерения одного правила на основном разрежении (см. описание модуля)."""
    sim = similarity(rule, ns, ALL_MONTHS, geo_cache)
    S = sim.S
    Q, null = _q_for(rule, ns, ALL_MONTHS, sim, p, rng)
    edges = sparsify(S, p.method, p.k, Q, p.fdr_q)
    lists = G.knn_lists(S, p.k)
    info = ns.info
    row: dict = {"rule": rule.name, "kind": rule.kind, "meaning": MEANING[rule.kind]}
    row.update(G.passport(edges, info, lists, p.attr_columns))
    row["assort_north"] = G.assortativity(G.to_igraph(ns.n, edges), ns.attrs["north"].to_numpy())
    row["sigma"] = sim.sigma
    row["seconds"] = sim.seconds

    years = {y: sim if rule.is_geo else similarity(rule, ns, m, geo_cache) for y, m in YEAR_MONTHS.items()}
    q_years = (
        {y: _q_for(rule, ns, m, years[y], p, rng)[0] for y, m in YEAR_MONTHS.items()}
        if p.method == "threshold"
        else {}
    )
    e23, e24 = (sparsify(years[y].S, p.method, p.k, q_years.get(y), p.fdr_q) for y in YEAR_MONTHS)
    row["reliability"] = G.jaccard(G.edge_set(e23), G.edge_set(e24))
    row["reliability_chance"] = chance_jaccard(e23, e24, ns.n, rng)
    row["rho_years"] = G.rank_corr_upper(years[2023].S, years[2024].S) if not rule.is_geo else 1.0
    row["static"] = bool(rule.is_geo)

    # бутстрап месяцев (внутри каждого года): устойчивость сети на всех месяцах к набору месяцев
    if p.bootstrap and (rule.is_basket or rule.kind == "rhythm_corr"):
        main_set = G.edge_set(edges)
        vals = []
        for _ in range(p.bootstrap):
            mb = np.concatenate([rng.choice(m, size=len(m), replace=True) for m in YEAR_MONTHS.values()])
            vals.append(G.jaccard(G.edge_set(G.knn_edges(similarity(rule, ns, mb).S, p.k)), main_set))
        row["boot_jaccard"] = float(np.mean(vals))
    else:
        row["boot_jaccard"] = np.nan

    row["geo_jaccard"] = G.jaccard(G.edge_set(edges), G.edge_set(geo_edges)) if geo_edges is not None else 1.0
    row["geo_difference"] = 1.0 - row["geo_jaccard"]

    pr = G.probe(ns.n, edges, p.seeds, p.resolution, p.rewire_repeats, p.rewire_factor)
    from sklearn.metrics import adjusted_mutual_info_score

    row.update(
        {
            "q": pr.q,
            "q_null_mean": pr.q_null_mean,
            "q_null_sd": pr.q_null_sd,
            "q_z": pr.q_z,
            "q_excess": pr.q_excess,
            "n_comm": pr.n_comm,
            "ari_seeds": pr.ari_seeds,
            "ami_region": float(adjusted_mutual_info_score(ns.groups, pr.membership)),
        }
    )
    if rule.is_series:
        iu = np.triu_indices(ns.n, 1)
        qv = Q[iu]
        row["n_sig_pairs"] = int((qv < p.fdr_q).sum())
        row["signal_share"] = float((Q[edges["i"], edges["j"]] < p.fdr_q).mean()) if len(edges) else np.nan
        if rule.kind in ("rhythm_corr", "rhythm_lag"):  # порог по r имеет смысл только у корреляций
            s_up = S[iu]
            row["naive_pairs"] = int((s_up > p.naive_r).sum())
            row["naive_null_expected"] = float((null > p.naive_r).mean() * len(s_up))
        if rule.kind == "rhythm_corr" and p.spearman_check:
            sp = spearman_similarity(rule, ns, ALL_MONTHS)
            row["spearman_jaccard"] = G.jaccard(
                G.edge_set(G.knn_edges(sp, p.k)), G.edge_set(G.knn_edges(S, p.k))
            )
    deg = G.degree_of(edges, ns.n)
    occ = G.kocc(lists, ns.n)
    for tid, pos in ns.city_pos.items():
        row[f"deg_{tid}"] = int(deg[pos])
        row[f"kocc_{tid}"] = int(occ[pos])
    top = np.argsort(-occ, kind="stable")[:3]
    names = ns.table["name"].astype(str).to_numpy()
    regions = ns.table["region_name"].astype(str).to_numpy()
    row["top_hubs"] = "; ".join(f"{names[t]} ({regions[t]}) — {int(occ[t])}" for t in top)
    return RuleResult(rule, sim, years, edges, lists, Q, null, pr, row)


# --- Чувствительность к k и способу разрежения ----------------------------------------------------


def k_grid_rows(
    res: RuleResult, ns: NodeSet, p: NetworkParams, geo_by_k: dict[int, pd.DataFrame]
) -> list[dict]:
    """Паспорт, надёжность, отличие от географии и зонд при каждом k (основной способ разрежения)."""
    from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

    rows = []
    rule = res.spec
    for k in p.k_grid:
        e = sparsify(res.sim.S, p.method, k, res.Q, p.fdr_q)
        lists = G.knn_lists(res.sim.S, k)
        ps = G.passport(e, ns.info, lists, p.attr_columns)
        e23, e24 = (G.knn_edges(res.years[y].S, k, mutual=p.method == "mutual_knn") for y in YEAR_MONTHS)
        pr = G.probe(ns.n, e, p.seeds[:GRID_SEEDS], p.resolution, GRID_SEEDS, p.rewire_factor)
        rows.append(
            {
                "rule": rule.name,
                "k": k,
                "n_edges": ps["n_edges"],
                "giant_share": ps["giant_share"],
                "deg_max": ps["deg_max"],
                "kocc_skew": ps["kocc_skew"],
                "clustering": ps["clustering"],
                "within_region": ps["within_region"],
                "attr_consistency": ps["attr_consistency"],
                "reliability": G.jaccard(G.edge_set(e23), G.edge_set(e24)),
                "geo_jaccard": G.jaccard(G.edge_set(e), G.edge_set(geo_by_k[k]))
                if not rule.kind == "road"
                else 1.0,
                "q_z": pr.q_z,
                "q_excess": pr.q_excess,
                "n_comm": pr.n_comm,
                "ami_region": float(adjusted_mutual_info_score(ns.groups, pr.membership)),
                "ari_vs_main_k": float(adjusted_rand_score(res.probe.membership, pr.membership)),
            }
        )
    return rows


def sparsify_rows(res: RuleResult, ns: NodeSet, p: NetworkParams, geo_cache: GeoCache, rng) -> list[dict]:
    """Способы разрежения при основном k: kNN, взаимный kNN, порог (значимость или квантиль)."""
    rows = []
    rule = res.spec
    q_years = {y: _q_for(rule, ns, m, res.years[y], p, rng)[0] for y, m in YEAR_MONTHS.items()}
    for method in ("knn", "mutual_knn", "threshold"):
        e = sparsify(res.sim.S, method, p.k, res.Q, p.fdr_q)
        ps = G.passport(e, ns.info, None, p.attr_columns)
        e23, e24 = (sparsify(res.years[y].S, method, p.k, q_years.get(y), p.fdr_q) for y in YEAR_MONTHS)
        rows.append(
            {
                "rule": rule.name,
                "method": method,
                "n_edges": ps["n_edges"],
                "giant_share": ps["giant_share"],
                "isolates": ps["isolates"],
                "deg_median": ps["deg_median"],
                "deg_max": ps["deg_max"],
                "clustering": ps["clustering"],
                "within_region": ps["within_region"],
                "attr_consistency": ps["attr_consistency"],
                "reliability": G.jaccard(G.edge_set(e23), G.edge_set(e24)),
            }
        )
    return rows


# --- Сходство сетей между собой -------------------------------------------------------------------


def pair_rows(results: dict[str, RuleResult], p: NetworkParams, rng) -> pd.DataFrame:
    """Для каждой пары правил: Жаккар рёбер, ρ Спирмена сходств всех пар (перестановочное p, тест
    Мантела) и ARI разбиений зонда."""
    from sklearn.metrics import adjusted_rand_score

    rows = []
    for a, b in itertools.combinations(results, 2):
        ra, rb = results[a], results[b]
        rho, pval = G.mantel_ranks(ra.sim.S, rb.sim.S, p.mantel_permutations, rng)
        rows.append(
            {
                "rule_a": a,
                "rule_b": b,
                "jaccard": G.jaccard(G.edge_set(ra.edges), G.edge_set(rb.edges)),
                "rank_corr": rho,
                "rank_corr_p": pval,
                "ari": float(adjusted_rand_score(ra.probe.membership, rb.probe.membership)),
            }
        )
    return pd.DataFrame(rows)


def paired_bootstrap(results: dict[str, RuleResult], ns: NodeSet, p: NetworkParams, rng) -> pd.DataFrame:
    """Надёжность правил на одних и тех же бутстрап-выборках месяцев (внутри каждого года): парное сравнение
    правил — доля повторов, где одно правило надёжнее другого. Только корзина и корреляция: у лагов и DTW
    повтор месяцев ломает порядок во времени. Бутстрап-Жаккар смещён вниз (повторённые месяцы добавляют
    шум), поэтому сравниваются правила между собой, а не с точечной оценкой."""
    rules = [r for r in results.values() if r.spec.is_basket or r.spec.kind == "rhythm_corr"]
    rows = []
    for b in range(p.bootstrap):
        mb = {y: rng.choice(m, size=len(m), replace=True) for y, m in YEAR_MONTHS.items()}
        for res in rules:
            e = [G.edge_set(G.knn_edges(similarity(res.spec, ns, mb[y]).S, p.k)) for y in YEAR_MONTHS]
            rows.append({"replicate": b, "rule": res.spec.name, "reliability": G.jaccard(e[0], e[1])})
    return pd.DataFrame(rows)


# --- Выбор ---------------------------------------------------------------------------------------


def criteria_table(comparison: pd.DataFrame, p: NetworkParams) -> pd.DataFrame:
    """Критерии выбора кандидатов: больше — лучше (простота — минус ранг)."""
    c = comparison.set_index("rule").loc[list(p.candidates)]
    return pd.DataFrame(
        {
            "reliability": c["reliability"],
            "geo_difference": c["geo_difference"],
            "attribute_consistency": c["attr_consistency"],
            "simplicity": [-p.simplicity[r] for r in c.index],
        },
        index=c.index,
    )


def pareto_front(crit: pd.DataFrame) -> list[str]:
    """Правила, которых никто не доминирует (не хуже по всем критериям и лучше хотя бы по одному)."""
    front = []
    for r in crit.index:
        dominated = any(
            (crit.loc[o] >= crit.loc[r]).all() and (crit.loc[o] > crit.loc[r]).any()
            for o in crit.index
            if o != r
        )
        if not dominated:
            front.append(r)
    return front


def lexicographic(crit: pd.DataFrame, order) -> str:
    """Лучшее правило по критериям в порядке ``order`` (следующий критерий — только при равенстве)."""
    ranked = crit.sort_values(list(order), ascending=False, kind="mergesort")
    return str(ranked.index[0])


def select(comparison: pd.DataFrame, p: NetworkParams) -> tuple[str, pd.DataFrame, dict]:
    """Основное правило: Парето-фронт, на нём — лексикографически по ``priority``; таблица всех порядков."""
    crit = criteria_table(comparison, p)
    front = pareto_front(crit)
    chosen = lexicographic(crit.loc[front], p.priority)
    rows = []
    for order in itertools.permutations(p.priority):
        rows.append(
            {
                "order": " > ".join(order),
                "winner": lexicographic(crit.loc[front], order),
                "declared": order == p.priority,
            }
        )
    orders = pd.DataFrame(rows)
    # сумма рангов (Борда) — ещё одно разумное правило агрегирования
    borda = crit.rank(ascending=False, method="min").sum(axis=1).sort_values(kind="mergesort")
    info = {
        "front": front,
        "chosen": chosen,
        "borda_winner": str(borda.index[0]),
        "orders_same": int((orders["winner"] == chosen).sum()),
        "orders_total": len(orders),
        "criteria": crit.reset_index().rename(columns={"index": "rule"}),
    }
    return chosen, orders, info


def select_by_k(grid: pd.DataFrame, p: NetworkParams) -> pd.DataFrame:
    """Выбор тем же правилом (Парето, затем порядок ``priority``) при каждом k из ``k_grid``."""
    rows = []
    for k, g in grid.groupby("k"):
        g = g.set_index("rule").loc[list(p.candidates)]
        crit = pd.DataFrame(
            {
                "reliability": g["reliability"],
                "geo_difference": 1.0 - g["geo_jaccard"],
                "attribute_consistency": g["attr_consistency"],
                "simplicity": [-p.simplicity[r] for r in g.index],
            },
            index=g.index,
        )
        front = pareto_front(crit)
        rows.append(
            {"k": int(k), "front": ", ".join(front), "winner": lexicographic(crit.loc[front], p.priority)}
        )
    return pd.DataFrame(rows)


def alternatives(chosen: str, comparison: pd.DataFrame, p: NetworkParams) -> list[str]:
    """Альтернативы для проверки устойчивости выводов: лучший кандидат другого семейства (корзина / ритм)
    по тому же порядку критериев и контроль географии ``road``."""
    kinds = comparison.set_index("rule")["kind"]
    family = "basket" if kinds[chosen].startswith("basket") else "rhythm"
    crit = criteria_table(comparison, p)
    other = [r for r in crit.index if not kinds[r].startswith(family)]
    out = [lexicographic(crit.loc[other], p.priority)] if other else []
    road = [r for r in kinds.index if kinds[r] == "road"]
    return out + road[:1]


# --- Режимы узлов ---------------------------------------------------------------------------------


def mode_rows(
    main_ns: NodeSet,
    main: dict[str, RuleResult],
    other: NodeSet,
    rules: list[RuleSpec],
    p: NetworkParams,
    geo_cache: GeoCache,
) -> list[dict]:
    """Как меняется сеть, если районы Москвы и Петербурга — отдельные узлы: доля рёбер между районами
    одного города, степени, сообщества районов, совпадение рёбер и сообществ на общих узлах."""
    rows = []
    t = other.table
    inner = t["is_inner_city"].to_numpy(dtype=bool)
    city_of = np.where(inner, t["region_code"].to_numpy(), -1)
    common = sorted(set(main_ns.ids) & set(other.ids))
    for rule in rules:
        sim = similarity(rule, other, ALL_MONTHS, geo_cache)
        e = sparsify(sim.S, p.method, p.k, None, p.fdr_q)
        ps = G.passport(e, other.info, G.knn_lists(sim.S, p.k), p.attr_columns)
        i, j = e["i"].to_numpy(), e["j"].to_numpy()
        touch = inner[i] | inner[j]
        same_city = inner[i] & inner[j] & (city_of[i] == city_of[j])
        pr = G.probe(other.n, e, p.seeds[:GRID_SEEDS], p.resolution, GRID_SEEDS, p.rewire_factor)
        deg = G.degree_of(e, other.n)
        row = {
            "mode": other.mode,
            "rule": rule.name,
            "n_nodes": other.n,
            "n_inner": int(inner.sum()),
            "n_edges": ps["n_edges"],
            "within_region": ps["within_region"],
            "attr_consistency": ps["attr_consistency"],
            "edges_touching_inner": int(touch.sum()),
            "inner_same_city_share": float(same_city.sum() / touch.sum()) if touch.any() else np.nan,
            "inner_deg_median": float(np.median(deg[inner])) if inner.any() else np.nan,
            "other_deg_median": float(np.median(deg[~inner])),
            "q_z": pr.q_z,
            "q_excess": pr.q_excess,
            "n_comm": pr.n_comm,
        }
        for code in sorted(set(city_of[inner])):
            memb = pr.membership[city_of == code]
            vals, counts = np.unique(memb, return_counts=True)
            row[f"city{code}_modal_share"] = float(counts.max() / counts.sum())
            row[f"city{code}_n_comm"] = int(len(vals))
            # доля узлов модального сообщества района, которые сами — районы того же города
            modal = vals[np.argmax(counts)]
            in_modal = pr.membership == modal
            row[f"city{code}_modal_purity"] = float((city_of[in_modal] == code).mean())
        # совпадение с основным режимом на общих узлах (все, кроме узлов-городов и районов)
        ref = main[rule.name]
        a = G.edge_set(ref.edges, main_ns.ids)
        b = G.edge_set(e, other.ids)
        cs = set(common)
        a = {x for x in a if x[0] in cs and x[1] in cs}
        b = {x for x in b if x[0] in cs and x[1] in cs}
        row["jaccard_common"] = G.jaccard(a, b)
        ma = dict(zip(main_ns.ids.tolist(), ref.probe.membership.tolist(), strict=True))
        mb = dict(zip(other.ids.tolist(), pr.membership.tolist(), strict=True))
        row["ari_common"] = G.ari_on_common({c: ma[c] for c in common}, {c: mb[c] for c in common})
        ref_deg = G.degree_of(ref.edges, main_ns.n)
        for tid, pos in main_ns.city_pos.items():
            row[f"collapse_deg_{tid}"] = int(ref_deg[pos])
        rows.append(row)
    return rows


# --- Окна динамики --------------------------------------------------------------------------------


def window_list(
    kinds, rolling_months: int, rolling_step: int, months: pd.DataFrame
) -> list[tuple[str, str, np.ndarray]]:
    """Окна (вид, имя, месяцы t): кварталы и полугодия календаря и скользящие окна ``rolling_months``
    с шагом ``rolling_step``; имя скользящего окна — «ГГГГ-ММ…ГГГГ-ММ»."""
    dates = months.set_index("t")["date"]
    out = []
    for kind in kinds:
        if kind == "quarter":
            for q in range(8):
                y, qq = 2023 + q // 4, q % 4 + 1
                out.append(("quarter", f"{y}Q{qq}", np.arange(3 * q, 3 * q + 3)))
        elif kind == "half":
            for h in range(4):
                y, hh = 2023 + h // 2, h % 2 + 1
                out.append(("half", f"{y}H{hh}", np.arange(6 * h, 6 * h + 6)))
        elif kind == "rolling":
            for s in range(0, 24 - rolling_months + 1, rolling_step):
                m = np.arange(s, s + rolling_months)
                out.append(("rolling", f"{dates[m[0]]}…{dates[m[-1]]}", m))
    return out


def dynamics(
    rule: RuleSpec, ns: NodeSet, p: NetworkParams, geo_cache: GeoCache, rng: np.random.Generator
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Сети выбранного правила по окнам: рёбра, векторы корзины окон, сводка по видам окон и «шум против
    изменения» (сети на нечётных и чётных месяцах отрезка 2L против двух соседних окон по L месяцев)."""
    windows = window_list(p.dyn_kinds, p.rolling_months, p.rolling_step, ns.months)
    edge_frames, vec_frames, rows = [], [], []
    sets: dict[str, list[set]] = {}
    for kind, name, months in windows:
        if rule.is_series and len(months) < p.min_series_months:
            continue
        S = similarity(rule, ns, months, geo_cache).S
        e = G.knn_edges(S, p.k)
        e = e.assign(window_kind=kind, window=name, n_months=len(months))
        edge_frames.append(e)
        sets.setdefault(kind, []).append(G.edge_set(e))
        if rule.is_basket:
            X = window_clr(ns, months)
            v = pd.DataFrame(
                X,
                columns=[
                    f"clr_rel_{c}" for c in ("food", "marketplace", "transport", "health", "cafe", "other")
                ],
            )
            v.insert(0, "territory_id", ns.ids)
            v.insert(0, "window", name)
            v.insert(0, "window_kind", kind)
            v["n_months"] = len(months)
            v["first_date"] = ns.months.set_index("t")["date"][months[0]]
            v["last_date"] = ns.months.set_index("t")["date"][months[-1]]
            vec_frames.append(v)
    for kind, ss in sets.items():
        adj = [G.jaccard(ss[i], ss[i + 1]) for i in range(len(ss) - 1)]
        rows.append(
            {
                "window_kind": kind,
                "n_windows": len(ss),
                "adjacent_jaccard": float(np.mean(adj)) if adj else np.nan,
                "adjacent_min": float(np.min(adj)) if adj else np.nan,
                "adjacent_max": float(np.max(adj)) if adj else np.nan,
            }
        )
    summary = pd.DataFrame(rows)

    noise_rows = []
    dates = ns.months.set_index("t")["date"]
    for L in p.dyn_lengths:
        if rule.is_series and L < p.min_series_months:
            continue
        for s0 in range(0, 24 - 2 * L + 1, 2 * L):
            span = np.arange(s0, s0 + 2 * L)
            odd, even, first, second = span[0::2], span[1::2], span[:L], span[L:]
            S_o, S_e, S_1, S_2 = (similarity(rule, ns, m, geo_cache).S for m in (odd, even, first, second))
            noise_rows.append(
                {
                    "length": L,
                    "span": f"{dates[span[0]]}…{dates[span[-1]]}",
                    "noise_jaccard": G.jaccard(
                        G.edge_set(G.knn_edges(S_o, p.k)), G.edge_set(G.knn_edges(S_e, p.k))
                    ),
                    "change_jaccard": G.jaccard(
                        G.edge_set(G.knn_edges(S_1, p.k)), G.edge_set(G.knn_edges(S_2, p.k))
                    ),
                    "noise_rank_corr": G.rank_corr_upper(S_o, S_e),
                    "change_rank_corr": G.rank_corr_upper(S_1, S_2),
                }
            )
    edges = pd.concat(edge_frames, ignore_index=True) if edge_frames else pd.DataFrame()
    vecs = pd.concat(vec_frames, ignore_index=True) if vec_frames else pd.DataFrame()
    return edges, vecs, summary, pd.DataFrame(noise_rows)
