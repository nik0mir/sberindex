"""Применение предрегистрированного правила к таблице кандидатов и все проверки чувствительности.

Основной выбор — ``selection.two_level`` по четырём критериям среди ``final_eligible``. Проверки
(``selection.sensitivity``) пересчитывают выбор, меняя одно: агрегатор (Борда, 24 лексикографических порядка,
Копленд по всем допустимым), набор критериев (``sets``), допуски (× ``tie_scale``), шаг (``joint`` — один шаг
по всем парам «метод, K»), состав методов (``all_eligible``). Правило не меняется: смена победителя пишется
в таблицу и объясняется в отчёте.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from munnet.clustering import selection as S
from munnet.clustering.params import CRITERIA, ClusterParams

CHECK_LABELS: dict[str, str] = {
    "main": "основное правило",
    "borda": "Борда на фронте",
    "copeland_all": "Копленд по всем допустимым",
    "lexicographic": "лексикографический порядок",
    "set_icvi_only": "только ICVI",
    "set_plan": "из PLAN.md: ICVI и устойчивость",
    "tie_x0": "допуски × 0",
    "tie_x2": "допуски × 2",
    "joint": "один шаг по всем парам «метод, K»",
    "all_eligible": "все семейства",
}


def z_cols(cp: ClusterParams, space: str) -> list[str]:
    return [f"z_{n}" for n in cp.metric_names(space)]


def run_rule(
    cands: pd.DataFrame,
    cp: ClusterParams,
    methods=None,
    criteria=CRITERIA,
    tol_scale: float = 1.0,
    aggregator: str = S.AGG_COPELAND,
    order=None,
    z_features=None,
    z_graph=None,
    nan_worst: bool = False,
    stab_tol: float = 0.0,
) -> S.TwoLevel:
    tol = {c: cp.tie.get(c, 0.0) * tol_scale for c in criteria}
    return S.two_level(
        cands,
        list(methods if methods is not None else cp.eligible_methods),
        list(criteria),
        z_features if z_features is not None else z_cols(cp, "features"),
        z_graph if z_graph is not None else z_cols(cp, "graph"),
        cp.directions,
        tol,
        aggregator,
        order,
        nan_worst,
        stab_tol,
    )


# Проверки по свойствам индексов (добавлены после предрегистрации 28.09, по свойствам ICVI, найденным при
# независимой проверке модуля до расчётов этапа): что меняется в метриках критериев качества.
ICVI_CHECK_LABELS: dict[str, str] = {
    "no_avu": "качество на G без AVU",
    "avi_mq_one": "AVI и MQ — одна метрика",
    "no_s_dbw": "качество в X без S_Dbw",
    "s_dbw_own": "S_Dbw в варианте own",
    "nan_worst": "неопределённая метрика — худший ранг",
    "noise_cluster": "шум HDBSCAN — отдельный кластер в индексах",
}


def icvi_variants(cp: ClusterParams) -> dict[str, dict]:
    zf, zg = z_cols(cp, "features"), z_cols(cp, "graph")
    out = {
        "no_avu": {"z_graph": [z for z in zg if z != "z_avu"]},
        "avi_mq_one": {"z_graph": [("z_avi", "z_mq"), *[z for z in zg if z not in ("z_avi", "z_mq")]]},
        "no_s_dbw": {"z_features": [z for z in zf if z != "z_s_dbw"]},
        "s_dbw_own": {"z_features": [z if z != "z_s_dbw" else "z_s_dbw_own" for z in zf]},
        "nan_worst": {"nan_worst": True},
        "noise_cluster": {"z_features": [f"{z}_nc" for z in zf], "z_graph": [f"{z}_nc" for z in zg]},
    }
    return out


def icvi_checks(cands: pd.DataFrame, cp: ClusterParams, main: S.TwoLevel, every: S.TwoLevel) -> pd.DataFrame:
    """Для каждой проверки по свойствам индексов: победитель уровня K каждого метода и итог; меняется ли."""
    rows = []
    base1 = {m: ch.winner for m, ch in every.level1.items()}
    for name, kw in icvi_variants(cp).items():
        cols = [
            c
            for v in kw.values()
            if isinstance(v, list)
            for g in v
            for c in ((g,) if isinstance(g, str) else g)
        ]
        if any(c not in cands.columns for c in cols):
            continue
        tl = run_rule(cands, cp, **kw)
        ev = run_rule(cands, cp, methods=cp.methods, **kw)
        for m, ch in ev.level1.items():
            rows.append(
                {
                    "check": name,
                    "level": 1,
                    "method": m,
                    "winner": ch.winner,
                    "k": int(cands.loc[ch.winner, "k"]),
                    "main_winner": base1.get(m),
                    "changed": ch.winner != base1.get(m),
                }
            )
        rows.append(
            {
                "check": name,
                "level": 2,
                "method": None if tl.winner is None else cands.loc[tl.winner, "method"],
                "winner": tl.winner,
                "k": None if tl.winner is None else int(cands.loc[tl.winner, "k"]),
                "main_winner": main.winner,
                "changed": tl.winner != main.winner,
            }
        )
        rows.append(
            {
                "check": name,
                "level": 3,  # все семейства
                "method": None if ev.winner is None else cands.loc[ev.winner, "method"],
                "winner": ev.winner,
                "k": None if ev.winner is None else int(cands.loc[ev.winner, "k"]),
                "main_winner": every.winner,
                "changed": ev.winner != every.winner,
            }
        )
    return pd.DataFrame(rows)


def edge_and_raw(cands: pd.DataFrame, cp: ClusterParams, every: S.TwoLevel) -> pd.DataFrame:
    """Проверка (ж): выбранный K на краю сетки; у Leiden и Louvain — K с наибольшей сырой MQ (справка)."""
    lo, hi = min(cp.k_grid), max(cp.k_grid)
    rows = []
    for m, ch in every.level1.items():
        k = int(cands.loc[ch.winner, "k"])
        row = {"method": m, "winner": ch.winner, "k": k, "k_on_edge": k in (lo, hi)}
        sub = cands.loc[(cands["method"] == m) & cands["feasible"].astype(bool)]
        if m in ("leiden", "louvain") and "mq" in sub.columns and sub["mq"].notna().any():
            row["k_raw_mq"] = int(sub.loc[sub["mq"].idxmax(), "k"])
        rows.append(row)
    return pd.DataFrame(rows)


@dataclass
class Decision:
    main: S.TwoLevel
    all_methods: S.TwoLevel  # уровень 1 по всем методам; уровень 2 — проверка all_eligible
    checks: pd.DataFrame  # проверка -> победитель
    orders: pd.DataFrame  # 24 порядка -> победитель
    icvi: pd.DataFrame  # проверки по свойствам индексов (вне предрегистрации)
    edges: pd.DataFrame  # K на краю сетки, K по сырой MQ


SCOPES: dict[str, str] = {"eligible": "методы с двумя источниками", "all": "все семейства"}


def check_rows(cands: pd.DataFrame, cp: ClusterParams, scope: str) -> tuple[list[dict], list[dict]]:
    """Весь набор проверок чувствительности на одном уровне: ``eligible`` — методы с двумя источниками
    (``final_eligible``, основное правило), ``all`` — все семейства (проверка ``all_eligible``; набор
    проверок на
    этом уровне добавлен после предрегистрации: там есть из чего выбирать). ``tie_chain`` — толкование цепочки
    равенств с допуском устойчивости (добавлено после предрегистрации)."""
    methods = cp.eligible_methods if scope == "eligible" else cp.methods
    rule = lambda **kw: run_rule(cands, cp, methods=methods, **kw).winner  # noqa: E731
    rows = [{"check": "main", "winner": rule()}]
    if "borda" in cp.aggregators:
        rows.append({"check": "borda", "winner": rule(aggregator=S.AGG_BORDA)})
    if "copeland_all" in cp.aggregators:
        rows.append({"check": "copeland_all", "winner": rule(aggregator=S.AGG_COPELAND_ALL)})
    orows = []
    if "lexicographic_all_orders" in cp.aggregators:
        for order in S.all_orders(CRITERIA):
            orows.append({"order": " > ".join(order), "winner": rule(aggregator=S.AGG_LEX, order=order)})
    for name, crit in cp.sets.items():
        rows.append({"check": f"set_{name}", "winner": rule(criteria=crit)})
    for scale in cp.tie_scale:
        rows.append({"check": f"tie_x{scale:g}", "winner": rule(tol_scale=scale)})
    if cp.joint:
        ch = S.one_step(
            cands, methods, CRITERIA, z_cols(cp, "features"), z_cols(cp, "graph"), cp.directions, cp.tie
        )
        rows.append({"check": "joint", "winner": None if ch is None else ch.winner})
    rows.append({"check": "tie_chain", "winner": rule(stab_tol=float(cp.tie["stability"]))})
    if cp.all_eligible and scope == "eligible":
        rows.append({"check": "all_eligible", "winner": run_rule(cands, cp, methods=cp.methods).winner})
    for r in (*rows, *orows):
        r["scope"] = scope
    return rows, orows


def decide(cands: pd.DataFrame, cp: ClusterParams) -> Decision:
    main = run_rule(cands, cp)
    every = run_rule(cands, cp, methods=cp.methods)
    rows, orows = [], []
    for scope, ref in (("eligible", main.winner), ("all", every.winner)):
        r, o = check_rows(cands, cp, scope)
        for x in (*r, *o):
            x["same_as_main"] = x["winner"] == ref
        rows += r
        orows += o
    # толкование «только второй уровень»: K каждого метода — по основному правилу, агрегатор меняется только
    # при выборе метода (так порядки считал судья критерия 3; добавлено после предрегистрации)
    winners = [ch.winner for ch in every.level1.values()]
    sub = cands.loc[winners]
    crit = S.criteria_table(sub, z_cols(cp, "features"), z_cols(cp, "graph"), list(CRITERIA))
    tol = {c: cp.tie.get(c, 0.0) for c in CRITERIA}
    for order in S.all_orders(CRITERIA) if "lexicographic_all_orders" in cp.aggregators else ():
        w = S.choose(crit, sub, cp.directions, tol, S.AGG_LEX, order).winner
        orows.append(
            {
                "order": " > ".join(order),
                "winner": w,
                "scope": "all_level2",
                "same_as_main": w == every.winner,
            }
        )
    w = S.choose(crit, sub, cp.directions, tol, S.AGG_BORDA).winner
    rows.append({"check": "borda", "winner": w, "scope": "all_level2", "same_as_main": w == every.winner})
    checks = pd.DataFrame(rows)
    orders = pd.DataFrame(orows)
    for df in (checks, orders):
        if len(df):
            df["method"] = df["winner"].map(lambda c: None if c is None else cands.loc[c, "method"])
            df["k"] = df["winner"].map(lambda c: None if c is None else int(cands.loc[c, "k"]))
    return Decision(
        main=main,
        all_methods=every,
        checks=checks,
        orders=orders,
        icvi=icvi_checks(cands, cp, main, every),
        edges=edge_and_raw(cands, cp, every),
    )


def level_table(tl: S.TwoLevel, cands: pd.DataFrame) -> pd.DataFrame:
    """Промежуточные результаты: для каждого кандидата уровня 1 и 2 — критерии набора, фронт, очки."""
    rows = []
    for m, ch in tl.level1.items():
        for c in ch.criteria.index:
            rows.append(
                {
                    "level": 1,
                    "set": m,
                    "cand": c,
                    **{f"crit_{k}": v for k, v in ch.criteria.loc[c].items()},
                    "on_front": c in ch.front,
                    "score": ch.scores.get(c, float("nan")),
                    "winner": c == ch.winner,
                }
            )
    if tl.level2 is not None:
        ch = tl.level2
        for c in ch.criteria.index:
            rows.append(
                {
                    "level": 2,
                    "set": "methods",
                    "cand": c,
                    **{f"crit_{k}": v for k, v in ch.criteria.loc[c].items()},
                    "on_front": c in ch.front,
                    "score": ch.scores.get(c, float("nan")),
                    "winner": c == ch.winner,
                }
            )
    out = pd.DataFrame(rows)
    if len(out):
        out["method"] = out["cand"].map(cands["method"])
        out["k"] = out["cand"].map(cands["k"])
    return out
