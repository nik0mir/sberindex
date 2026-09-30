"""T3 и T2: надёжные переходы против псевдогодов без реального времени — функциями этапа dynamics.

Псевдопара (``T3_reliable_placebo.pseudo_pairs``): псевдогод А — случайные 6 календарных месяцев 2023 года
и остальные 6 календарных месяцев 2024-го, псевдогод Б — наоборот; в каждом все 12 календарных месяцев.
Для каждой псевдопары тем же кодом, что у этапа dynamics (``dynamics.windows.window_graph``,
``window_inputs``, ``clustering.methods.fit_protocol``), строятся разбиения псевдогодов и их половин; типы
сопоставляются венгерским алгоритмом по Жаккару (``dynamics.tracking.aligned``), надёжный переход — правило
``stable_halves`` (``dynamics.tracking.stable_halves``). Этап dynamics и его выходы не меняются.

Способы прослеживания R1 (``robustness.tracking``): фиксированные типы — узел половины к ближайшему центру
типа 24 месяцев (``tracking.nearest``); эволюционный — псевдогод Б цепочкой шагов
``tracking.evolutionary_kmeans`` от центров псевдогода А (столько же шагов, сколько между окнами 2023 и 2024
годов в dynamics), половины — один шаг от центров своего псевдогода (``tracking.centers``).
"""

from __future__ import annotations

import itertools
import logging
import zlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.stats import binom

from munnet.clustering import inputs as CI
from munnet.clustering import methods as M
from munnet.clustering import parallel
from munnet.clustering.params import RANDOM_METHODS
from munnet.dynamics import tracking as T
from munnet.dynamics import windows as W
from munnet.interpret import stats as S
from munnet.network.data import NodeSet
from munnet.network.params import NetworkParams

log = logging.getLogger(__name__)

MAIN, FIXED, EVO = "refit_match", "fixed_prototypes", "evolutionary"
N_CAL = 12  # календарных месяцев в году
HALF_SIZE = 6  # «случайные 6 календарных месяцев»
# «95% интервал — n_reliable минус 97,5-й и 2,5-й перцентили плацебо» (T3.excess, записано словами)
EXCESS_LEVEL = 0.95


# --- Контекст прогона и задачи ---------------------------------------------------------------------


@dataclass(frozen=True)
class RunCtx:
    """Всё, что нужно процессу для перефита окна: узлы, входы, правило сети, метод итога и seed."""

    name: str
    ns: NodeSet
    base: CI.Inputs
    netp: NetworkParams
    rule: str
    knn: int
    k: int
    alpha: float
    method: str
    param: Any
    seeds: tuple[int, ...]
    impl: Mapping[str, Any]
    schemes: dict[str, tuple[tuple[int, ...], tuple[int, ...]]]
    space: W.HybridSpace | None = None  # пространство гибрида 24 месяцев (способы R1: fixed, evolutionary)
    C24: np.ndarray | None = None
    epsilon: float = 0.5
    max_iter: int = 300
    blind: int | None = None


def tag_key(*parts: Any) -> list[int]:
    return [zlib.crc32(str(p).encode("utf-8")) for p in parts]


def blind(labels: np.ndarray, ctx: RunCtx, *key: Any) -> np.ndarray:
    """Слепой прогон: метки перемешиваются по узлам (размеры сохраняются), seed — ``--blind`` и ключ
    разбиения;
    в обычном прогоне — как есть."""
    if ctx.blind is None:
        return labels
    rng = np.random.default_rng([int(ctx.blind), *tag_key(ctx.name, *key)])
    return np.asarray(labels)[rng.permutation(len(labels))]


def unblind(labels: np.ndarray, ctx: RunCtx, *key: Any) -> np.ndarray:
    """Обратное к ``blind``: метки в исходном порядке узлов (для сверок, не раскрывающих типов)."""
    if ctx.blind is None:
        return labels
    rng = np.random.default_rng([int(ctx.blind), *tag_key(ctx.name, *key)])
    return np.asarray(labels)[np.argsort(rng.permutation(len(labels)))]


def fit_months(
    ctx: RunCtx, months: Sequence[int], name: str, want_z: bool
) -> tuple[np.ndarray, np.ndarray | None]:
    """Разбиение месяцев ``months`` методом итога (как ``dynamics.stage``): сеть окна, X окна, seed."""
    months = np.asarray(sorted(months))
    e = W.window_graph(ctx.ns, ctx.netp, ctx.rule, ctx.knn, months)
    A = CI.graph_matrix(e, ctx.ns.ids)
    inp = W.window_inputs(ctx.base, ctx.ns, A, months, name)
    pf = M.fit_protocol(ctx.method, inp, ctx.k, ctx.param, ctx.seeds, ctx.impl, RANDOM_METHODS, {}, ctx.alpha)
    Z = ctx.space.embed(inp)[0] if (want_z and ctx.space is not None) else None
    return pf.labels, Z


def fit_task(task: tuple) -> tuple[str, np.ndarray, np.ndarray | None]:
    """Задача процесса: окно (``tag``, месяцы) → метки (после слепой перестановки) и Z для способов R1."""
    tag, months, want_z = task
    ctx: RunCtx = parallel.STATE["ctx"]
    lab, Z = fit_months(ctx, months, tag, want_z)
    return tag, blind(lab, ctx, "fit", tag), Z


# --- Псевдопары ------------------------------------------------------------------------------------


def pseudo_sets(n_pairs: int, rng: np.random.Generator) -> list[tuple[int, ...]]:
    """``n_pairs`` наборов 6 календарных месяцев 1…12 (месяцы 2023 года в псевдогоде А) — различных псевдопар.

    Уточнение реализации (не в предрегистрации): набор S и его дополнение задают одну и ту же псевдопару
    с переставленными псевдогодами А и Б (подгонка зависит только от месяцев), поэтому берутся без повторов
    ``n_pairs`` из C(12, 6) / 2 = 462 классов {S, дополнение S}, а ориентация (какой из двух наборов —
    месяцы 2023 года в А) — случайно от того же ``rng``. Каждый набор по-прежнему равновероятен, как
    «случайные 6 календарных месяцев» записи, но зеркальных дублей среди псевдопар нет.
    """
    full = set(range(1, N_CAL + 1))
    # представитель класса — набор с январём; дополнение — тот же класс
    classes = [s for s in itertools.combinations(range(1, N_CAL + 1), HALF_SIZE) if s[0] == 1]
    if n_pairs > len(classes):
        raise ValueError(f"pseudo_pairs = {n_pairs} больше числа различных псевдопар {len(classes)}")
    pick = rng.choice(len(classes), n_pairs, replace=False)
    flip = rng.random(n_pairs) < 0.5
    out = []
    for i, f in zip(pick, flip, strict=True):
        s = classes[int(i)]
        out.append(tuple(sorted(full - set(s))) if f else s)
    return out


def pseudo_months(S23: Sequence[int]) -> tuple[list[int], list[int]]:
    """Номера t месяцев псевдогодов А и Б (t = 0 — 2023-01, t = 12 — 2024-01)."""
    s = set(int(m) for m in S23)
    a = [m - 1 for m in range(1, N_CAL + 1) if m in s] + [
        N_CAL + m - 1 for m in range(1, N_CAL + 1) if m not in s
    ]
    b = [m - 1 for m in range(1, N_CAL + 1) if m not in s] + [
        N_CAL + m - 1 for m in range(1, N_CAL + 1) if m in s
    ]
    return sorted(a), sorted(b)


def split_halves(
    months: Sequence[int], scheme: tuple[Sequence[int], Sequence[int]]
) -> tuple[list[int], list[int]]:
    """Половины набора месяцев по календарному месяцу: первая — ``scheme[0]``, вторая — ``scheme[1]``."""
    first = {int(m) for m in scheme[0]}
    h1 = [t for t in months if t % N_CAL + 1 in first]
    h2 = [t for t in months if t % N_CAL + 1 not in first]
    return h1, h2


def year_months(year_index: int) -> list[int]:
    return list(range(N_CAL * year_index, N_CAL * year_index + N_CAL))


def reliable_matrix(h: Sequence[np.ndarray], k: int) -> np.ndarray:
    """Матрица надёжных переходов (``stable_halves``) по половинам А1, А2, Б1, Б2 в одной нумерации."""
    rel, ta, tb = T.stable_halves(*h)
    return T.transition_matrix(ta[rel], tb[rel], k)


def evo_steps(ctx: RunCtx) -> int:
    """Число шагов эволюционного способа между окнами 2023 и 2024 годов в наблюдении (dynamics:
    скользящих окон минус одно; при 12-месячных окнах с шагом в месяц — 12)."""
    n = sum(q.kind == "rolling" for q in W.periods(ctx.ns, ctx.netp))
    return max(n - 1, 1)


def evolutionary_chain(
    Z: np.ndarray,
    C0: np.ndarray,
    labels0: np.ndarray,
    k: int,
    steps: int,
    eps: float,
    max_iter: int,
) -> tuple[np.ndarray, np.ndarray]:
    """``steps`` шагов эволюционного K-means на одних данных ``Z`` от центров ``C0`` (размеры — по меткам
    прошлого шага, как ``dynamics.compute.evolutionary_track``): вес стартовых центров убывает как в цепочке
    окон наблюдения."""
    C, lab = np.asarray(C0, dtype=np.float64), np.asarray(labels0)
    for _ in range(int(steps)):
        lab, C = T.evolutionary_kmeans(Z, C, np.bincount(lab, minlength=k), eps, max_iter)
    return lab, C


def pair_task(task: tuple) -> dict:
    """Одна псевдопара: разбиения псевдогодов и их половин (все схемы половин), надёжные переходы по трём
    способам. Возвращает матрицы надёжных переходов «тип А → тип Б» в нумерации итоговых типов прогона."""
    r, S23, final = task
    ctx: RunCtx = parallel.STATE["ctx"]
    k = ctx.k
    track = ctx.space is not None and ctx.C24 is not None
    ma, mb = pseudo_months(S23)
    out: dict[str, Any] = {"r": r, "set": list(S23), MAIN: {}}
    fits = {}
    for nm, months in (("A", ma), ("B", mb)):
        lab, Z = fit_months(ctx, months, f"pseudo{r}{nm}", track)
        fits[nm] = (blind(lab, ctx, "pair", r, nm), Z)
    wa = T.aligned(final, fits["A"][0], k)
    wb = T.aligned(final, fits["B"][0], k)
    if track:
        out[FIXED], out[EVO] = {}, {}
        CA = T.centers(fits["A"][1], wa, k)
        sa = np.bincount(wa, minlength=k)
        # как в наблюдении: псевдогод Б — цепочка шагов от центров А (столько же шагов, сколько в dynamics
        # между окнами 2023 и 2024 годов), а не один шаг: иначе Б сильнее привязан к А и плацебо мягче
        lb_evo, CB = evolutionary_chain(fits["B"][1], CA, wa, k, evo_steps(ctx), ctx.epsilon, ctx.max_iter)
        lb_evo = blind(lb_evo, ctx, "pair-evo", r, "B")
        sb = np.bincount(lb_evo, minlength=k)
    for sname, scheme in ctx.schemes.items():
        halves, zs = [], []
        for nm, months in (("A", ma), ("B", mb)):
            for j, hm in enumerate(split_halves(months, scheme)):
                lab, Z = fit_months(ctx, hm, f"pseudo{r}{nm}{sname}{j}", track)
                lab = blind(lab, ctx, "pair", r, nm, sname, j)
                ref = wa if nm == "A" else wb
                halves.append(T.aligned(ref, lab, k))
                zs.append((nm, Z))
        out[MAIN][sname] = reliable_matrix(halves, k)
        if track:
            fx = [blind(T.nearest(z, ctx.C24), ctx, "pair-fixed", r, sname, i) for i, (_, z) in enumerate(zs)]
            out[FIXED][sname] = reliable_matrix(fx, k)
            ev = []
            for i, (nm, z) in enumerate(zs):
                C0, s0 = (CA, sa) if nm == "A" else (CB, sb)
                lab = T.evolutionary_kmeans(z, C0, s0, ctx.epsilon, ctx.max_iter)[0]
                ev.append(blind(lab, ctx, "pair-evo", r, sname, i))
            out[EVO][sname] = reliable_matrix(ev, k)
    return out


def run_tasks(ctx: RunCtx, tasks: list, fn, workers: int) -> list:
    return parallel.run(fn, tasks, {"ctx": ctx}, workers)


# --- T3 --------------------------------------------------------------------------------------------


@dataclass
class T3Scheme:
    scheme: str
    n_reliable: int
    placebo: np.ndarray
    median: float
    p95: float
    lo: float  # n_reliable − 97,5-й перцентиль
    hi: float  # n_reliable − 2,5-й перцентиль
    passed: bool


@dataclass
class T3Result:
    verdict: str
    per: dict[str, T3Scheme]
    n_nodes: int
    excess: float
    excess_share: float
    one_in_ten: bool


def t3_eval(
    observed: Mapping[str, int],
    placebo: Mapping[str, np.ndarray],
    n_nodes: int,
    percentile: float,
    one_in_ten: Sequence[float],
    main_scheme: str,
) -> T3Result:
    """Правило T3: наблюдаемое число надёжных переходов выше ``percentile``-го перцентиля плацебо того же
    разбиения половин; confirmed — в обоих разбиениях, partial — в одном, not — ни в одном."""
    per = {}
    q_lo, q_hi = (1 - EXCESS_LEVEL) / 2 * 100, (1 + EXCESS_LEVEL) / 2 * 100
    for s, obs in observed.items():
        pl = np.asarray(placebo[s], dtype=np.float64)
        p95 = float(np.percentile(pl, percentile))
        per[s] = T3Scheme(
            scheme=s,
            n_reliable=int(obs),
            placebo=pl,
            median=float(np.median(pl)),
            p95=p95,
            lo=float(obs - np.percentile(pl, q_hi)),
            hi=float(obs - np.percentile(pl, q_lo)),
            passed=bool(obs > p95),
        )
    n_pass = sum(r.passed for r in per.values())
    verdict = "confirmed" if n_pass == len(per) else ("partial" if n_pass >= 1 else "not")
    m = per[main_scheme]
    excess = m.n_reliable - m.median
    share = excess / n_nodes
    return T3Result(
        verdict=verdict,
        per=per,
        n_nodes=n_nodes,
        excess=float(excess),
        excess_share=float(share),
        one_in_ten=bool(one_in_ten[0] <= share <= one_in_ten[1]),
    )


# --- T2 --------------------------------------------------------------------------------------------


def up_down(M_: np.ndarray, rank: np.ndarray) -> tuple[int, int]:
    """Число переходов вверх и вниз по рангу ступени (``up: rank_increase``); диагональ не считается."""
    k = M_.shape[0]
    up = sum(int(M_[a, b]) for a in range(k) for b in range(k) if rank[b] > rank[a])
    down = sum(int(M_[a, b]) for a in range(k) for b in range(k) if rank[b] < rank[a])
    return up, down


@dataclass
class T2Scheme:
    scheme: str
    n_up: int
    n_down: int
    share_up: float
    share_ci: tuple[float, float]
    p0: float  # NaN — не определена (min_pool)
    n_pool: int
    p_up: float
    p_down: float
    up_sig: bool
    down_sig: bool
    undetermined: bool
    net_flow: float
    net_ci: tuple[float, float]
    placebo_abs_n_p95: float
    placebo_n_total: int
    placebo_n_median: float
    z: float
    per_pair: list[dict] = field(default_factory=list)


@dataclass
class T2Result:
    verdict: str  # уровень после place_cap
    text_key: str  # confirmed | partial_capped | partial | not | down
    per: dict[str, T2Scheme]
    place: dict
    capped: bool
    describe: dict = field(default_factory=dict)


def t2_scheme(
    scheme: str,
    t_a: np.ndarray,
    t_b: np.ndarray,
    reliable: np.ndarray,
    placebo: Sequence[np.ndarray],
    rank: np.ndarray,
    alpha: float,
    min_pool: int,
    n_boot: int,
    level: float,
    rng: np.random.Generator,
    adjacent: Sequence[tuple[int, int]],
    pct: float,
) -> T2Scheme:
    """Доля «вверх» u = вверх / (вверх + вниз) по надёжным переходам против p0 плацебо (биномиальный тест)."""
    a, b = np.asarray(t_a)[reliable], np.asarray(t_b)[reliable]
    ra, rb = rank[a], rank[b]
    n_up, n_down = int((rb > ra).sum()), int((rb < ra).sum())
    n = n_up + n_down
    u = n_up / n if n else float("nan")
    pool_up = pool_n = 0
    abs_n, per_pair_counts = [], []
    for M_ in placebo:
        up, down = up_down(M_, rank)
        pool_up += up
        pool_n += up + down
        per_pair_counts.append(int(M_.sum()))
        if up + down:
            abs_n.append(abs(up - down) / (up + down))
    undetermined = pool_n < min_pool
    p0 = float("nan") if undetermined else pool_up / pool_n
    if undetermined or n == 0:
        p_up = p_down = float("nan")
    else:
        p_up = float(binom.sf(n_up - 1, n, p0))
        p_down = float(binom.cdf(n_up, n, p0))
    boots = []
    if n:
        up_mask = (rb > ra).astype(np.float64)
        for _ in range(n_boot):
            idx = rng.integers(0, n, n)
            boots.append(up_mask[idx].mean())
    ci = S.perc_ci(np.array(boots), level) if boots else (float("nan"), float("nan"))
    net = 2 * u - 1 if n else float("nan")
    z = (
        (net - (2 * p0 - 1)) / (2 * np.sqrt(p0 * (1 - p0) / n))
        if (n and not undetermined and 0 < p0 < 1)
        else float("nan")
    )
    per_pair = []
    for lo, hi in adjacent:  # номера типов: lo — нижняя ступень пары
        n_ij = int(((a == lo) & (b == hi)).sum())
        n_ji = int(((a == hi) & (b == lo)).sum())
        pl = []
        for M_ in placebo:
            x, y = int(M_[lo, hi]), int(M_[hi, lo])
            if x + y:
                pl.append(abs(x - y) / (x + y))
        per_pair.append(
            {
                "low": lo,
                "high": hi,
                "n_up": n_ij,
                "n_down": n_ji,
                "net": (n_ij - n_ji) / (n_ij + n_ji) if n_ij + n_ji else float("nan"),
                "placebo_abs_n_p95": float(np.percentile(pl, pct)) if pl else float("nan"),
            }
        )
    return T2Scheme(
        scheme=scheme,
        n_up=n_up,
        n_down=n_down,
        share_up=u,
        share_ci=ci,
        p0=p0,
        n_pool=int(pool_n),
        p_up=p_up,
        p_down=p_down,
        up_sig=bool(not undetermined and p_up < alpha),
        down_sig=bool(not undetermined and p_down < alpha),
        undetermined=bool(undetermined),
        net_flow=net,
        net_ci=(2 * ci[0] - 1, 2 * ci[1] - 1),
        placebo_abs_n_p95=float(np.percentile(abs_n, pct)) if abs_n else float("nan"),
        placebo_n_total=int(sum(per_pair_counts)),
        placebo_n_median=float(np.median(per_pair_counts)) if per_pair_counts else float("nan"),
        z=float(z),
        per_pair=per_pair,
    )


def t2_verdict(per: Mapping[str, T2Scheme], place_p: float, alpha: float) -> tuple[str, str, bool]:
    """Правило T2 (``rule``): confirmed — «чаще вверх» в обоих разбиениях; partial — в одном; down — вверх ни
    в одном, «чаще вниз» хотя бы в одном; not — иначе. ``place_cap``: при p < alpha у place_tree вердикт не
    выше partial (текст partial_capped при обоих разбиениях)."""
    n_up = sum(s.up_sig for s in per.values())
    any_down = any(s.down_sig for s in per.values())
    if n_up == len(per):
        verdict = "confirmed"
    elif n_up >= 1:
        verdict = "partial"
    elif any_down:
        verdict = "down"
    else:
        verdict = "not"
    capped = bool(np.isfinite(place_p) and place_p < alpha and verdict in ("confirmed", "partial"))
    text_key = verdict
    if capped and verdict == "confirmed":
        verdict, text_key = "partial", "partial_capped"
    return verdict, text_key, capped


def place_tree_test(
    X: np.ndarray,
    target: np.ndarray,
    t_a: np.ndarray,
    t_b: np.ndarray,
    reliable: np.ndarray,
    depth: int,
    class_weight: str,
    folds: int,
    n_perm: int,
    seed: int,
    rng: np.random.Generator,
) -> dict:
    """(c) T2: прогноз типа по признакам места (дерево на ``target`` вне фолда) совпадает с типом назначения
    у перешедших чаще, чем у неперешедших того же исходного типа (перестановки статуса внутри исходного типа).

    Неперешедшие исходного типа s — узлы с половинами обоих лет в типе s (как у T6: ``half_type_2023 =
    half_type_2024 = s``; уточнение реализации — определение взято у T6, у place_tree его нет)."""
    import warnings

    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.tree import DecisionTreeClassifier

    tree = DecisionTreeClassifier(max_depth=depth, class_weight=class_weight, random_state=seed)
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pred = cross_val_predict(tree, X, target, cv=cv)
    t_a, t_b = np.asarray(t_a), np.asarray(t_b)
    movers = np.asarray(reliable)
    stayers = (t_a >= 0) & (t_a == t_b)
    status = np.where(movers, t_b, -1)  # −1 — остался
    obs = int(((pred == status) & movers).sum())
    n_move = int(movers.sum())
    pools = [np.flatnonzero((movers | stayers) & (t_a == s)) for s in np.unique(t_a[movers])]
    null = np.empty(n_perm)
    for i in range(n_perm):
        st = status.copy()
        for idx in pools:
            st[idx] = st[idx][rng.permutation(len(idx))]
        null[i] = int(((pred == st) & (st >= 0)).sum())
    return {
        "S": obs,
        "n_movers": n_move,
        "share_a": obs / n_move if n_move else float("nan"),
        "share_null": float(null.mean()) / n_move if n_move else float("nan"),
        "p": S.p_greater(obs, null) if n_move else float("nan"),
        "tree_accuracy": float(np.mean(pred == target)),
    }


def d1_noise(
    odd: Mapping[int, np.ndarray], even: Mapping[int, np.ndarray], rank: np.ndarray, reverse: bool
) -> float:
    """D1 (описание): доля «вверх» среди смен «нечётные → чётные» одного года (``reverse`` — «чётные →
    нечётные»), среднее двух лет."""
    vals = []
    for y in odd:
        a, b = (even[y], odd[y]) if reverse else (odd[y], even[y])
        m = (a >= 0) & (b >= 0) & (a != b)
        if m.any():
            vals.append(float(np.mean(rank[b[m]] > rank[a[m]])))
    return float(np.mean(vals)) if vals else float("nan")


def d2_permutation(t_a: np.ndarray, t_b: np.ndarray, rank: np.ndarray, n_perm: int, rng) -> dict:
    """D2 (описание): типы назначения надёжных переходов переставлены с сохранением маргиналей; доля «вверх»
    на нуле в двух прочтениях — диагональ вне знаменателя и диагональ как «не вверх»."""
    a, b = np.asarray(t_a), np.asarray(t_b)
    ex, inc = [], []
    for _ in range(n_perm):
        bb = b[rng.permutation(len(b))]
        up = rank[bb] > rank[a]
        off = bb != a
        ex.append(up[off].mean() if off.any() else np.nan)
        inc.append(up.mean())
    u = float(np.mean(rank[b] > rank[a])) if len(a) else float("nan")
    return {
        "share_up": u,
        "null_excl_diag": float(np.nanmean(ex)),
        "null_diag_not_up": float(np.mean(inc)),
        "d2_excl_diag": u - float(np.nanmean(ex)),
        "d2_diag_not_up": u - float(np.mean(inc)),
    }
