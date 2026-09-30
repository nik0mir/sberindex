"""T1 (ступени по обороту Росстата), T5 (тривиальные деления) и отрицательные контроли — одно ядро.

``Evidence`` собирается один раз на прогон: выборки двух оборотов (``T1.indicators``) — (a) все
территориальные
узлы с показателем, (b) ещё и со стратой; бутстрап-выборки узлов; 25 соперников ``place_partitions`` с их
ориентацией и значениями ρ (b) и ε² на каждой бутстрап-выборке. Любое деление (типы, контроль, вариант R1)
оценивается функциями ``t1_eval`` и ``t5_eval`` на тех же выборках: «все ρ на одной выборке»
(``conditional.beyond_place``) и «наибольший у соперников — на каждой бутстрап-выборке» (``rival_max``).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import mannwhitneyu, rankdata
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

from munnet.interpret import stats as S
from munnet.interpret.partitions import Partition, orders, ranks_from_order

log = logging.getLogger(__name__)

# «нижняя граница 95% бутстрап-интервала больше 0» — T1.conditional.beyond_place и T5.stratified.beyond_place
# (порог 0 записан словами в предрегистрации, отдельного ключа нет)
BEYOND_MIN = 0.0


@dataclass(frozen=True)
class Turnover:
    """Показатель T1/T5 на территориальных узлах: позиции узлов выборок (a) и (b) и значения."""

    name: str
    sign: int
    a_idx: np.ndarray
    a_y: np.ndarray
    a_groups: np.ndarray
    b_idx: np.ndarray
    b_y: np.ndarray
    b_strata: np.ndarray
    b_yw: np.ndarray  # ранг внутри страты / n_s на полной выборке (b)
    b_ryw: np.ndarray  # ранги b_yw по всей выборке (b)


@dataclass
class Boot:
    """Бутстрап-выборка (b) одного оборота: позиции в выборке (b), ранги внутри страт и их ранги."""

    pos: np.ndarray
    strata: np.ndarray
    yw: np.ndarray
    ryw: np.ndarray


@dataclass
class Evidence:
    turnovers: dict[str, Turnover]
    boots_b: dict[str, list[Boot]]
    boots_a: dict[str, list[np.ndarray]]
    rivals: list[Partition]
    rival_rho: dict[str, np.ndarray] = field(default_factory=dict)  # оборот -> B × R ориентированных ρ (b)
    rival_eps: dict[str, np.ndarray] = field(default_factory=dict)  # оборот -> B × R ε²
    rival_point_rho: dict[str, np.ndarray] = field(default_factory=dict)  # полная выборка (b)
    rival_point_eps: dict[str, np.ndarray] = field(default_factory=dict)
    rival_ranks: dict[str, list[np.ndarray]] = field(
        default_factory=dict
    )  # оборот -> ориентированные ступени
    level: float = 0.95
    n_perm_t1_a: int = 0
    n_perm_t1_b: int = 0
    n_perm_t5: int = 0


def make_turnover(
    name: str,
    sign: int,
    values: np.ndarray,
    groups: np.ndarray,
    strata: np.ndarray,
) -> Turnover:
    """Выборки (a) и (b) показателя (``missing: drop``): (a) — узлы с показателем, (b) — и со стратой."""
    values = np.asarray(values, dtype=np.float64)
    a = np.flatnonzero(np.isfinite(values))
    b = np.flatnonzero(np.isfinite(values) & (np.asarray(strata) >= 0))
    yw = S.within_ranks(values[b], strata[b])
    return Turnover(
        name=name,
        sign=int(sign),
        a_idx=a,
        a_y=values[a],
        a_groups=np.asarray(groups)[a],
        b_idx=b,
        b_y=values[b],
        b_strata=np.asarray(strata)[b],
        b_yw=yw,
        b_ryw=rankdata(yw),
    )


def make_boots(t: Turnover, n_boot: int, rng: np.random.Generator) -> tuple[list[Boot], list[np.ndarray]]:
    """Бутстрап узлов: выборка (b) — ранги внутри страт пересчитываются на выборке; выборка (a) — отдельно."""
    boots_b, boots_a = [], []
    nb, na = len(t.b_idx), len(t.a_idx)
    for _ in range(n_boot):
        pos = rng.integers(0, nb, nb)
        st = t.b_strata[pos]
        yw = S.within_ranks(t.b_y[pos], st)
        boots_b.append(Boot(pos=pos, strata=st, yw=yw, ryw=rankdata(yw)))
        boots_a.append(rng.integers(0, na, na))
    return boots_b, boots_a


# --- Статистики одного деления -------------------------------------------------------------------


def rho_b_point(ranks: np.ndarray, t: Turnover) -> float:
    return S.strat_spearman(ranks[t.b_idx], t.b_yw, t.b_strata, t.b_ryw)


def rho_b_boot(ranks: np.ndarray, t: Turnover, boots: Sequence[Boot]) -> np.ndarray:
    r = ranks[t.b_idx]
    return np.array([S.strat_spearman(r[b.pos], b.yw, b.strata, b.ryw) for b in boots])


def eps_point(labels: np.ndarray, t: Turnover) -> float:
    return S.epsilon2(t.b_yw, labels[t.b_idx], t.b_ryw)


def eps_boot(labels: np.ndarray, t: Turnover, boots: Sequence[Boot]) -> np.ndarray:
    lab = labels[t.b_idx]
    return np.array([S.epsilon2(b.yw, lab[b.pos], b.ryw) for b in boots])


def best_order(
    labels: np.ndarray, score: Callable[[np.ndarray], float]
) -> tuple[np.ndarray, tuple[int, ...]]:
    """Ступени по лучшему из K! порядков групп (``best_of_24``); равенство — первый порядок в
    лексикографическом переборе."""
    k = len(np.unique(labels))
    best, best_s, best_o = None, -np.inf, None
    for o in orders(k):
        r = ranks_from_order(labels, o)
        s = score(r)
        if np.isfinite(s) and s > best_s + 1e-15:
            best, best_s, best_o = r, s, o
    if best is None:
        o = tuple(range(k))
        return ranks_from_order(labels, o), o
    return best, best_o


def build_rivals(ev: Evidence, rivals: Sequence[Partition], orient: Mapping[str, str]) -> None:
    """Ориентация соперников на полной выборке (b) каждого оборота и их ρ (b) и ε² на бутстрап-выборках.

    ``orient``: вид деления -> правило (``place_partitions.orientation``): sign_of_rho — при ρ (b) < 0 порядок
    групп обращается; best_of_24 — лучший из 24 порядков по ρ (b). В бутстрапе порядок не меняется."""
    ev.rivals = list(rivals)
    for name, t in ev.turnovers.items():
        B = len(ev.boots_b[name])
        rho = np.empty((B, len(rivals)))
        eps = np.empty((B, len(rivals)))
        prho, peps, oriented = [], [], []
        for j, p in enumerate(rivals):
            rule = orient[p.kind]
            if rule == "sign_of_rho":
                r = p.ranks
                if rho_b_point(r, t) < 0:
                    r = int(r.max()) + 1 - r
            elif rule == "best_of_24":
                r, _ = best_order(p.labels, lambda x, t=t: rho_b_point(x, t))
            else:
                raise ValueError(f"place_partitions.orientation: {rule!r} не реализовано")
            oriented.append(r)
            prho.append(rho_b_point(r, t))
            peps.append(eps_point(p.labels, t))
            rho[:, j] = rho_b_boot(r, t, ev.boots_b[name])
            eps[:, j] = eps_boot(p.labels, t, ev.boots_b[name])
        ev.rival_rho[name], ev.rival_eps[name] = rho, eps
        ev.rival_point_rho[name], ev.rival_point_eps[name] = np.array(prho), np.array(peps)
        ev.rival_ranks[name] = oriented


# --- T1 --------------------------------------------------------------------------------------------


@dataclass
class T1Turnover:
    name: str
    n_a: int
    n_b: int
    rho_a: float
    rho_a_ci: tuple[float, float]
    p_a: float
    q_a: float
    med_a: list[float]
    mono_a: bool
    rho_b: float
    rho_b_ci: tuple[float, float]
    p_b: float
    q_b: float
    med_b: list[float]
    mono_b: bool
    diff_point: float
    diff_ci: tuple[float, float]
    best_rival: str
    best_rival_label: str
    best_rival_rho: float
    overall: bool = False
    beyond: bool = False
    # чувствительность: «за сверх места» по строгому прочтению — с полным «за в целом» (a),
    # включая |ρ| ≥ rho_min
    beyond_strict: bool = False


@dataclass
class T1Result:
    verdict: str
    per: dict[str, T1Turnover]
    # проверка чувствительности: вердикт по строгому прочтению «за сверх места» (с rho_min; в вывод не идёт)
    verdict_strict: str = ""
    pairwise: list[dict] = field(default_factory=list)


def _allowed(ev: Evidence, exclude: Sequence[str]) -> np.ndarray:
    return np.array([p.name not in set(exclude) for p in ev.rivals])


def t1_eval(
    ranks: np.ndarray,
    ev: Evidence,
    rng: np.random.Generator,
    alpha: float,
    rho_min: float,
    exclude: Sequence[str] = (),
    n_steps: int = 4,
) -> T1Result:
    """Правило T1 на делении со ступенями ``ranks`` (1…n_steps по всем территориальным узлам).

    Оборот «за в целом» — (a): знак, |ρ| ≥ rho_min, q < alpha, монотонность. «За сверх места» — буквально по
    записи «(a) и (b): знак, q < alpha, монотонность, beyond_place»: для (a) — знак, q < alpha, монотонность
    (без rho_min), для (b) — то же и нижняя граница разности с сильнейшим соперником > 0. Строгое прочтение
    (с полным «за в целом», включая rho_min) — ``beyond_strict`` и ``verdict_strict``, проверка
    чувствительности рядом с главным вердиктом.
    """
    steps = list(range(1, n_steps + 1))
    allowed = _allowed(ev, exclude)
    per: dict[str, T1Turnover] = {}
    for name, t in ev.turnovers.items():
        ra = ranks[t.a_idx]
        rho_a = S.spearman(ra, t.a_y)
        null_a = np.array(
            [S.spearman(S.permute_within(ra, t.a_groups, rng), t.a_y) for _ in range(ev.n_perm_t1_a)]
        )
        boot_a = np.array([S.spearman(ra[i], t.a_y[i]) for i in ev.boots_a[name]])
        rb = ranks[t.b_idx]
        rho_b = S.strat_spearman(rb, t.b_yw, t.b_strata, t.b_ryw)
        null_b = np.array(
            [
                S.strat_spearman(S.permute_within(rb, t.b_strata, rng), t.b_yw, t.b_strata, t.b_ryw)
                for _ in range(ev.n_perm_t1_b)
            ]
        )
        boot_b = rho_b_boot(ranks, t, ev.boots_b[name])
        rmax = ev.rival_rho[name][:, allowed].max(axis=1)
        # соперники ориентированы в свою пользу (ρ ≥ 0 на полной выборке), поэтому сравнивается sign × ρ (b);
        # при sign = +1 (все обороты записи) это просто ρ (b)
        diff = t.sign * boot_b - rmax
        point_rivals = np.where(allowed, ev.rival_point_rho[name], -np.inf)
        jb = int(np.argmax(point_rivals))
        per[name] = T1Turnover(
            name=name,
            n_a=len(t.a_idx),
            n_b=len(t.b_idx),
            rho_a=rho_a,
            rho_a_ci=S.perc_ci(boot_a, ev.level),
            p_a=S.p_two_sided(rho_a, null_a),
            q_a=np.nan,
            med_a=S.medians_by(t.a_y, ra, steps),
            # монотонность — по sign × медианы (как у заместителей T1.proxies): при sign = −1 медианы убывают
            mono_a=S.monotone_ok([t.sign * m for m in S.medians_by(t.a_y, ra, steps)]),
            rho_b=rho_b,
            rho_b_ci=S.perc_ci(boot_b, ev.level),
            p_b=S.p_two_sided(rho_b, null_b),
            q_b=np.nan,
            med_b=S.medians_by(t.b_yw, rb, steps),
            mono_b=S.monotone_ok([t.sign * m for m in S.medians_by(t.b_yw, rb, steps)]),
            diff_point=t.sign * rho_b - float(point_rivals.max()),
            diff_ci=S.perc_ci(diff, ev.level),
            best_rival=ev.rivals[jb].name,
            best_rival_label=ev.rivals[jb].label,
            best_rival_rho=float(ev.rival_point_rho[name][jb]),
        )
    names = list(per)
    qa = S.bh_qvalues(np.array([per[n].p_a for n in names]))
    qb = S.bh_qvalues(np.array([per[n].p_b for n in names]))
    for n, a, b in zip(names, qa, qb, strict=True):
        r = per[n]
        r.q_a, r.q_b = float(a), float(b)
        sign = ev.turnovers[n].sign
        r.overall = bool(np.sign(r.rho_a) == sign and abs(r.rho_a) >= rho_min and r.q_a < alpha and r.mono_a)
        cond_b = bool(np.sign(r.rho_b) == sign and r.q_b < alpha and r.mono_b and r.diff_ci[0] > BEYOND_MIN)
        # главный вердикт — буквальное прочтение «(a) и (b): знак, q < alpha, монотонность, beyond_place»:
        # rho_min в перечне для «сверх места» не назван, поэтому для (a) он не требуется
        r.beyond = bool(np.sign(r.rho_a) == sign and r.q_a < alpha and r.mono_a and cond_b)
        # проверка чувствительности (публикуется рядом, в вывод не идёт): строгое прочтение — полное
        # «за в целом» по (a), включая |ρ| ≥ rho_min
        r.beyond_strict = bool(r.overall and cond_b)
    overall = [r.overall for r in per.values()]
    verdict = t1_verdict([r.beyond for r in per.values()], overall)
    strict_v = t1_verdict([r.beyond_strict for r in per.values()], overall)
    return T1Result(verdict=verdict, per=per, verdict_strict=strict_v)


def t1_verdict(beyond: Sequence[bool], overall: Sequence[bool]) -> str:
    """Правило T1: оба оборота «за сверх места» — confirmed, ровно один — partial_one, ни одного, но хотя бы
    один «за в целом» — partial_overall, иначе not."""
    n_beyond, n_overall = sum(bool(b) for b in beyond), sum(bool(o) for o in overall)
    if n_beyond == len(beyond):
        return "confirmed"
    if n_beyond == 1:
        return "partial_one"
    if n_overall >= 1:
        return "partial_overall"
    return "not"


def t1_pairwise(
    ranks: np.ndarray, ev: Evidence, n_boot: int, rng, level: float, n_steps: int = 4
) -> list[dict]:
    """Описание: Манн — Уитни для соседних ступеней (поправка Холма) и бутстрап-интервалы медиан,
    выборка (a)."""
    rows = []
    for name, t in ev.turnovers.items():
        ra = ranks[t.a_idx]
        ps, pairs = [], []
        for s in range(1, n_steps):
            x, y = t.a_y[ra == s + 1], t.a_y[ra == s]
            if len(x) and len(y):
                p = float(mannwhitneyu(x, y, alternative="two-sided").pvalue)
            else:
                p = np.nan
            ps.append(p)
            pairs.append((s, s + 1, float(np.median(y)) if len(y) else np.nan, S.cliff_delta(x, y)))
        adj = S.holm(ps)
        for (lo, hi, _, delta), p, pa in zip(pairs, ps, adj, strict=True):
            rows.append(
                {"turnover": name, "step_low": lo, "step_high": hi, "p": p, "p_holm": pa, "cliff": delta}
            )
        for s in range(1, n_steps + 1):
            v = t.a_y[ra == s]
            if len(v):
                meds = [float(np.median(v[rng.integers(0, len(v), len(v))])) for _ in range(n_boot)]
                lo, hi = S.perc_ci(np.array(meds), level)
            else:
                lo = hi = np.nan
            rows.append(
                {
                    "turnover": name,
                    "step": s,
                    "n": int(len(v)),
                    "median": float(np.median(v)) if len(v) else np.nan,
                    "median_lo": lo,
                    "median_hi": hi,
                }
            )
    return rows


def regions_up(
    ranks: np.ndarray,
    values: np.ndarray,
    groups: np.ndarray,
    min_steps: int,
    n_perm: int,
    rng: np.random.Generator,
) -> dict:
    """Описание: число групп региона с узлами хотя бы ``min_steps`` ступеней, где медиана строго растёт по
    ступеням; нуль — перестановки меток внутри групп региона, односторонний p."""
    ok = np.isfinite(values)
    r, v, g = ranks[ok], values[ok], groups[ok]

    def count(rr: np.ndarray) -> tuple[int, int]:
        n_up = n_elig = 0
        for grp in np.unique(g):
            m = g == grp
            present = np.unique(rr[m])
            if len(present) < min_steps:
                continue
            n_elig += 1
            meds = [np.median(v[m & (rr == s)]) for s in present]
            if np.all(np.diff(meds) > 0):
                n_up += 1
        return n_up, n_elig

    obs, elig = count(r)
    null = np.array([count(S.permute_within(r, g, rng))[0] for _ in range(n_perm)])
    return {
        "n_up": obs,
        "n_eligible": elig,
        "null_mean": float(null.mean()),
        "p": S.p_greater(obs, null),
    }


# --- T5 --------------------------------------------------------------------------------------------


@dataclass
class T5Turnover:
    name: str
    n: int
    eps: float
    eps_ci: tuple[float, float]
    p: float
    q: float
    diff_point: float
    diff_ci: tuple[float, float]
    best_rival: str
    best_rival_label: str
    best_rival_eps: float
    passed: bool = False


@dataclass
class T5Result:
    verdict: str
    cond1: bool
    cond3: bool
    ami: dict[str, float]
    ari: dict[str, float]
    max_partition: str
    max_label: str
    max_ami: float
    per: dict[str, T5Turnover]
    composition: dict = field(default_factory=dict)
    region_ami: float = float("nan")


def t5_eval(
    labels: np.ndarray,
    ev: Evidence,
    trivial: Mapping[str, Partition],
    rng: np.random.Generator,
    ami_max: float,
    alpha: float,
    exclude: Sequence[str] = (),
) -> T5Result:
    """Правило T5: (1) AMI с каждым тривиальным делением меньше ``ami_max``; (3) хотя бы у одного оборота
    q < alpha и нижняя граница «ε² минус наибольший ε² соперников» > 0."""
    ami, ari = {}, {}
    for name, p in trivial.items():
        ami[name] = float(adjusted_mutual_info_score(p.labels, labels))
        ari[name] = float(adjusted_rand_score(p.labels, labels))
    mx = max(ami, key=lambda k: ami[k])
    cond1 = bool(max(ami.values()) < ami_max)
    allowed = _allowed(ev, exclude)
    per: dict[str, T5Turnover] = {}
    for name, t in ev.turnovers.items():
        lab = labels[t.b_idx]
        e = S.epsilon2(t.b_yw, lab, t.b_ryw)
        null = np.array(
            [S.epsilon2(t.b_yw, S.permute_within(lab, t.b_strata, rng), t.b_ryw) for _ in range(ev.n_perm_t5)]
        )
        boot = eps_boot(labels, t, ev.boots_b[name])
        diff = boot - ev.rival_eps[name][:, allowed].max(axis=1)
        point = np.where(allowed, ev.rival_point_eps[name], -np.inf)
        jb = int(np.argmax(point))
        per[name] = T5Turnover(
            name=name,
            n=len(t.b_idx),
            eps=e,
            eps_ci=S.perc_ci(boot, ev.level),
            p=S.p_greater(e, null),
            q=np.nan,
            diff_point=e - float(point.max()),
            diff_ci=S.perc_ci(diff, ev.level),
            best_rival=ev.rivals[jb].name,
            best_rival_label=ev.rivals[jb].label,
            best_rival_eps=float(ev.rival_point_eps[name][jb]),
        )
    names = list(per)
    q = S.bh_qvalues(np.array([per[n].p for n in names]))
    for n, qq in zip(names, q, strict=True):
        per[n].q = float(qq)
        per[n].passed = bool(per[n].q < alpha and per[n].diff_ci[0] > BEYOND_MIN)
    cond3 = any(r.passed for r in per.values())
    verdict = {
        (True, True): "confirmed",
        (False, True): "partial",
        (True, False): "not_empty",
        (False, False): "not_repeats",
    }[(cond1, cond3)]
    return T5Result(
        verdict=verdict,
        cond1=cond1,
        cond3=cond3,
        ami=ami,
        ari=ari,
        max_partition=mx,
        max_label=trivial[mx].label,
        max_ami=ami[mx],
        per=per,
    )


def composition(labels: np.ndarray, urban_group: np.ndarray, min_types: int, min_share: float) -> dict:
    """Описание (редакция 3): в каждой группе доли горожан (кроме пропусков) — число типов с долей не меньше
    ``min_share`` узлов группы; сколько групп выполняют «не меньше ``min_types`` типов»."""
    rows = []
    for g in np.unique(urban_group):
        if g < 0:
            continue
        m = urban_group == g
        _, cnt = np.unique(labels[m], return_counts=True)
        n_types = int((cnt / m.sum() >= min_share).sum())
        rows.append({"group": int(g), "n": int(m.sum()), "n_types": n_types, "ok": n_types >= min_types})
    return {"groups": rows, "n_ok": sum(r["ok"] for r in rows), "n_groups": len(rows)}
