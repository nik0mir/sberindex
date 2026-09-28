"""Правило выбора ``clustering.selection`` — чистые функции над таблицей критериев (без данных и методов).

Предрегистрация 28.09.2026 (``configs/default.yaml``, блок ``clustering.selection``):

1. **Критерии** (направление ``criteria``): качество в X и на G — средний по трём метрикам ранг z-оценки ICVI
   против случайного базиса среди сравниваемых кандидатов (0 — худший, 1 — лучший); устойчивость — средний
   ARI с бутстрап-разбиениями; объяснимость — каппа неглубокого дерева.
2. **Парето-фронт с допуском** ``tie``: B доминирует A, если A ни по одному критерию не лучше B сверх допуска,
   а B хотя бы по одному лучше A сверх допуска. Фронт — недоминируемые допустимые кандидаты.
3. **Копленд на фронте**: A обходит B, если A лучше B сверх допуска по большему числу критериев, чем B
лучше A;
   очки = победы − поражения.
4. **Равенство очков** — выше устойчивость, затем проще (``simplicity``), затем меньше K (затем имя — только
   чтобы порядок был определён).

Одно правило на двух уровнях: K внутри метода, затем метод среди ``final_eligible`` (каждый со своим K).
Проверки чувствительности (``sensitivity``): агрегаторы Борда (сумма мест на фронте), лексикографический
порядок (все 24 порядка критериев, на фронте с допуском) и Копленд по всем допустимым; наборы критериев
``sets``; допуски × ``tie_scale``; один шаг по всем парам «метод, K» (``joint``); все семейства
(``all_eligible``).
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

AGG_COPELAND = "copeland"
AGG_BORDA = "borda"
AGG_COPELAND_ALL = "copeland_all"
AGG_LEX = "lexicographic"


# --- Критерии ------------------------------------------------------------------------------------


def rank01(values: pd.Series, direction: str = "max") -> pd.Series:
    """Ранг в [0; 1]: 1 — лучший, 0 — худший, равные — средний ранг; один кандидат — 1; пропуск — пропуск."""
    v = values.astype("float64")
    if direction == "min":
        v = -v
    n = int(v.notna().sum())
    if n == 0:
        return v
    if n == 1:
        return v.where(v.isna(), 1.0)
    r = v.rank(method="average")
    return (r - 1.0) / (n - 1.0)


def _groups(metrics) -> list[tuple[str, ...]]:
    """Метрики критерия: имя колонки или кортеж колонок, чьи ранги усредняются в одну метрику."""
    return [(m,) if isinstance(m, str) else tuple(m) for m in metrics]


def quality_rank(z: pd.DataFrame, metrics=None, nan_worst: bool = False) -> pd.Series:
    """Критерий качества: средний по метрикам ранг z-оценки среди строк ``z`` (z уже со знаком: больше —
    лучше). ``metrics`` — колонки или группы колонок (ранги группы усредняются в одну метрику); по умолчанию —
    все колонки по одной. Метрика, не определённая у кандидата, в его среднее не входит (``nan_worst`` —
    проверка: неопределённая метрика получает худший ранг 0)."""
    groups = _groups(metrics if metrics is not None else list(z.columns))
    if not groups:
        return pd.Series(np.nan, index=z.index)
    cols = sorted({c for g in groups for c in g})
    ranks = z[cols].apply(lambda col: rank01(col, "max"))
    if nan_worst:
        ranks = ranks.fillna(0.0)
    per = pd.DataFrame({i: ranks[list(g)].mean(axis=1, skipna=True) for i, g in enumerate(groups)})
    return per.mean(axis=1, skipna=True)


def n_defined(z: pd.DataFrame, metrics) -> pd.Series:
    """Сколько метрик критерия определено у кандидата (вошло в средний ранг)."""
    groups = _groups(metrics)
    return pd.DataFrame({i: z[list(g)].notna().any(axis=1) for i, g in enumerate(groups)}).sum(axis=1)


def criteria_table(
    cands: pd.DataFrame, z_features, z_graph, criteria: Sequence[str], nan_worst: bool = False
) -> pd.DataFrame:
    """Таблица критериев набора ``cands`` (индекс — кандидаты): ранги качества считаются внутри набора;
    ``z_features``, ``z_graph`` — колонки z-оценок или их группы (``quality_rank``)."""
    out = pd.DataFrame(index=cands.index)
    for c in criteria:
        if c in ("quality_features", "quality_graph"):
            metrics = _groups(z_features if c == "quality_features" else z_graph)
            cols = sorted({x for g in metrics for x in g})
            out[c] = quality_rank(cands[cols], metrics, nan_worst)
        else:
            out[c] = cands[c].astype("float64")
    return out


# --- Сравнение пары ------------------------------------------------------------------------------


def _signed(crit: pd.DataFrame, directions: Mapping[str, str]) -> pd.DataFrame:
    """Критерии со знаком: больше — лучше."""
    return pd.DataFrame(
        {c: crit[c] if directions.get(c, "max") == "max" else -crit[c] for c in crit.columns},
        index=crit.index,
    )


def pair_wins(a: pd.Series, b: pd.Series, tol: Mapping[str, float]) -> tuple[int, int]:
    """Сколько критериев A лучше B сверх допуска и сколько B лучше A (критерии со знаком «больше — лучше»);
    пропуск критерия у любого — ничья по нему."""
    wa = wb = 0
    for c in a.index:
        d = a[c] - b[c]
        if not np.isfinite(d):
            continue
        t = float(tol.get(c, 0.0))
        if d > t:
            wa += 1
        elif d < -t:
            wb += 1
    return wa, wb


def pareto_front(crit: pd.DataFrame, directions: Mapping[str, str], tol: Mapping[str, float]) -> list:
    """Недоминируемые кандидаты с допуском (порядок — как в ``crit``). Если допуск сделал всех доминируемыми
    (цикл), фронт — все кандидаты."""
    s = _signed(crit, directions)
    front = []
    for a in s.index:
        dominated = False
        for b in s.index:
            if a == b:
                continue
            wa, wb = pair_wins(s.loc[a], s.loc[b], tol)
            if wa == 0 and wb > 0:
                dominated = True
                break
        if not dominated:
            front.append(a)
    return front or list(s.index)


def copeland_scores(crit: pd.DataFrame, directions: Mapping[str, str], tol: Mapping[str, float]) -> pd.Series:
    """Очки Копленда: победы − поражения в парных сравнениях по числу критериев, выигранных сверх допуска."""
    s = _signed(crit, directions)
    score = pd.Series(0, index=s.index, dtype="int64")
    for a, b in itertools.combinations(s.index, 2):
        wa, wb = pair_wins(s.loc[a], s.loc[b], tol)
        if wa > wb:
            score[a] += 1
            score[b] -= 1
        elif wb > wa:
            score[b] += 1
            score[a] -= 1
    return score


def borda_points(crit: pd.DataFrame, directions: Mapping[str, str]) -> pd.Series:
    """Сумма мест по критериям (1 — лучший; равные — наименьшее место); меньше — лучше. Возвращается со знаком
    минус, чтобы у всех агрегаторов больше было лучше."""
    s = _signed(crit, directions)
    places = s.rank(ascending=False, method="min")
    return -places.sum(axis=1)


def lexicographic_set(
    crit: pd.DataFrame, order: Sequence[str], directions: Mapping[str, str], tol: Mapping[str, float]
) -> list:
    """Кандидаты, лучшие по первому критерию порядка в пределах допуска, среди них — по второму и т. д."""
    s = _signed(crit, directions)
    keep = list(s.index)
    for c in order:
        v = s.loc[keep, c]
        if v.notna().any():
            best = v.max()
            keep = [a for a in keep if not np.isfinite(v[a]) or v[a] >= best - float(tol.get(c, 0.0))]
            keep = [a for a in keep if np.isfinite(v[a])] or keep
        if len(keep) == 1:
            break
    return keep


# --- Разрешение равенств -------------------------------------------------------------------------


def tie_break(cands: Sequence, meta: pd.DataFrame) -> object:
    """Из равных по очкам: выше устойчивость, затем проще, затем меньше K, затем имя кандидата."""

    def key(c):
        r = meta.loc[c]
        stab = float(r["stability"]) if np.isfinite(r["stability"]) else -np.inf
        return (-stab, float(r["simplicity"]), int(r["k"]), str(c))

    return sorted(cands, key=key)[0]


@dataclass(frozen=True)
class Choice:
    winner: object
    front: list
    scores: pd.Series  # очки агрегатора на множестве, где он применялся (больше — лучше)
    criteria: pd.DataFrame  # таблица критериев набора
    aggregator: str
    tied: list = field(default_factory=list)  # равные по очкам до цепочки разрешения


def choose(
    crit: pd.DataFrame,
    meta: pd.DataFrame,
    directions: Mapping[str, str],
    tol: Mapping[str, float],
    aggregator: str = AGG_COPELAND,
    order: Sequence[str] | None = None,
) -> Choice:
    """Победитель набора ``crit`` (строки — допустимые кандидаты, колонки — критерии). ``meta`` — колонки
    ``stability``, ``simplicity``, ``k`` для цепочки равенств. Агрегаторы: ``copeland`` (основной: фронт →
    Копленд), ``borda`` (фронт → сумма мест), ``lexicographic`` (фронт → порядок ``order`` с допуском),
    ``copeland_all`` (Копленд по всем допустимым)."""
    if crit.empty:
        raise ValueError("choose: нет допустимых кандидатов")
    front = pareto_front(crit, directions, tol)
    if aggregator == AGG_COPELAND:
        scores = copeland_scores(crit.loc[front], directions, tol)
    elif aggregator == AGG_COPELAND_ALL:
        scores = copeland_scores(crit, directions, tol)
    elif aggregator == AGG_BORDA:
        scores = borda_points(crit.loc[front], directions)
    elif aggregator == AGG_LEX:
        if order is None:
            raise ValueError("choose: лексикографическому порядку нужен order")
        best = lexicographic_set(crit.loc[front], order, directions, tol)
        scores = pd.Series([1 if a in best else 0 for a in front], index=front, dtype="int64")
    else:
        raise ValueError(f"choose: неизвестный агрегатор {aggregator}")
    top = scores.max()
    tied = [a for a in scores.index if scores[a] == top]
    winner = tie_break(tied, meta)
    return Choice(winner=winner, front=front, scores=scores, criteria=crit, aggregator=aggregator, tied=tied)


# --- Два уровня ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class TwoLevel:
    winner: object  # кандидат-победитель (метод и K)
    level1: dict  # метод -> Choice внутри метода
    level2: Choice | None  # выбор между победителями методов


def two_level(
    cands: pd.DataFrame,
    methods: Sequence[str],
    criteria: Sequence[str],
    z_features: Sequence[str],
    z_graph: Sequence[str],
    directions: Mapping[str, str],
    tol: Mapping[str, float],
    aggregator: str = AGG_COPELAND,
    order: Sequence[str] | None = None,
    nan_worst: bool = False,
) -> TwoLevel:
    """Уровень 1 — K внутри каждого метода из ``methods`` среди его допустимых кандидатов; уровень 2 — метод
    среди победителей уровня 1. ``cands``: колонки ``method``, ``k``, ``feasible``, ``simplicity``,
    ``stability``, ``interpretability`` и z-оценки метрик."""
    crit_cols = list(criteria)
    tol_c = {c: tol.get(c, 0.0) for c in crit_cols}
    level1 = {}
    for m in methods:
        sub = cands.loc[(cands["method"] == m) & cands["feasible"].astype(bool)]
        if sub.empty:
            continue
        crit = criteria_table(sub, z_features, z_graph, crit_cols, nan_worst)
        level1[m] = choose(crit, sub, directions, tol_c, aggregator, order)
    if not level1:
        return TwoLevel(winner=None, level1={}, level2=None)
    winners = [ch.winner for ch in level1.values()]
    sub = cands.loc[winners]
    crit = criteria_table(sub, z_features, z_graph, crit_cols, nan_worst)
    level2 = choose(crit, sub, directions, tol_c, aggregator, order)
    return TwoLevel(winner=level2.winner, level1=level1, level2=level2)


def one_step(
    cands: pd.DataFrame,
    methods: Sequence[str],
    criteria: Sequence[str],
    z_features: Sequence[str],
    z_graph: Sequence[str],
    directions: Mapping[str, str],
    tol: Mapping[str, float],
) -> Choice | None:
    """Проверка ``joint``: одно применение основного правила ко всем допустимым парам «метод, K»."""
    sub = cands.loc[cands["method"].isin(list(methods)) & cands["feasible"].astype(bool)]
    if sub.empty:
        return None
    crit = criteria_table(sub, z_features, z_graph, list(criteria))
    return choose(crit, sub, directions, {c: tol.get(c, 0.0) for c in criteria})


def all_orders(criteria: Sequence[str]) -> list[tuple[str, ...]]:
    return list(itertools.permutations(criteria))
