"""Расчёт этапа dynamics на метках: три способа отслеживания, изменение против шума, события, проверка
сюжета и сравнение способов. Входы — массивы; чтение и запись — ``stage``."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.metrics import adjusted_rand_score

from munnet.dynamics import tracking as T
from munnet.dynamics.params import EVOLUTIONARY, FIXED, MAIN

HALVES: tuple[str, ...] = ("odd_2023", "even_2023", "odd_2024", "even_2024")


@dataclass(frozen=True)
class Track:
    """Разбиения одного способа в нумерации итоговых типов 24 месяцев (0…K−1)."""

    approach: str
    rolling: list[np.ndarray]  # 13 окон по порядку
    halves: dict[str, np.ndarray]  # нечётные и чётные месяцы 2023 и 2024 годов
    info: dict[str, float]

    @property
    def first(self) -> np.ndarray:
        return self.rolling[0]

    @property
    def last(self) -> np.ndarray:
        return self.rolling[-1]


# --- Три способа ---------------------------------------------------------------------------------


def main_track(
    final: np.ndarray, rolling: Sequence[np.ndarray], halves: Mapping[str, np.ndarray], k: int
) -> Track:
    """Перефит в каждом окне + сопоставление: первое окно — с итоговыми типами, каждое следующее — с прошлым
    (цепочка); половины года — с окном своего года."""
    chain = [T.aligned(final, rolling[0], k)]
    for lab in rolling[1:]:
        chain.append(T.aligned(chain[-1], lab, k))
    direct = [T.aligned(final, lab, k) for lab in rolling]
    h = {}
    for name in HALVES:
        ref = chain[0] if name.endswith("2023") else chain[-1]
        h[name] = T.aligned(ref, halves[name], k)
    info = {
        "chain_direct_windows_same": float(
            sum(np.array_equal(c, d) for c, d in zip(chain, direct, strict=True))
        ),
        "chain_direct_nodes_differ": float(
            sum(int((c != d).sum()) for c, d in zip(chain, direct, strict=True))
        ),
        "last_equals_direct_first": float(np.array_equal(chain[-1], T.aligned(chain[0], rolling[-1], k))),
    }
    return Track(MAIN, chain, h, info)


def fixed_track(Z_roll: Sequence[np.ndarray], Z_halves: Mapping[str, np.ndarray], C24: np.ndarray) -> Track:
    """Фиксированные типы: узел окна — к ближайшему центру типа 24 месяцев в опорном пространстве."""
    return Track(
        FIXED,
        [T.nearest(z, C24) for z in Z_roll],
        {n: T.nearest(Z_halves[n], C24) for n in HALVES},
        {},
    )


def evolutionary_track(
    Z_roll: Sequence[np.ndarray],
    Z_halves: Mapping[str, np.ndarray],
    first: np.ndarray,
    k: int,
    eps: float,
    max_iter: int,
) -> tuple[Track, list[np.ndarray]]:
    """Эволюционный K-means (Chakrabarti и др., 2006, разд. 4.2; ``eps`` — их cp): первое окно — разбиение
    основного способа (центры — средние по нему), каждое следующее — шаг от центров и размеров кластеров
    прошлого окна. Половины года — один шаг от центров и размеров окна своего года."""
    C = T.centers(Z_roll[0], first, k)
    labs, Cs = [np.asarray(first)], [C]
    for z in Z_roll[1:]:
        sizes = np.bincount(labs[-1], minlength=k)  # размеры кластеров прошлого окна: вес γ статьи
        lab, C = T.evolutionary_kmeans(z, C, sizes, eps, max_iter)
        labs.append(lab)
        Cs.append(C)
    h = {}
    for name in HALVES:
        i = 0 if name.endswith("2023") else -1
        sizes = np.bincount(labs[i], minlength=k)
        h[name] = T.evolutionary_kmeans(Z_halves[name], Cs[i], sizes, eps, max_iter)[0]
    return Track(EVOLUTIONARY, labs, h, {"epsilon": eps}), Cs


# --- Анализ способа ------------------------------------------------------------------------------


@dataclass(frozen=True)
class Change:
    boot: T.BootstrapResult
    reliable: np.ndarray  # bool по узлам
    t23: np.ndarray  # согласованный тип 2023 по половинам (−1 — разошлись)
    t24: np.ndarray
    matrix: np.ndarray  # все переходы первое → последнее окно
    reliable_matrix: np.ndarray  # надёжные переходы в типах половин года (предрегистрированное определение)
    window_reliable_matrix: (
        np.ndarray
    )  # надёжные, у которых переход по окнам тот же: те же оси, что ``matrix``
    n_reliable_window_same: int  # надёжные по половинам, но тип окон 2023 и 2024 годов один
    n_reliable_window_other: int  # надёжные, у которых окна показывают другой переход
    noise_shares: tuple[float, float]


def change_vs_noise(track: Track, k: int, n_boot: int, rng: np.random.Generator, level: float) -> Change:
    h = track.halves
    pairs = [(h["odd_2023"], h["even_2023"]), (h["odd_2024"], h["even_2024"])]
    boot = T.paired_bootstrap(track.first, track.last, pairs, k, n_boot, rng, level)
    rel, t23, t24 = T.stable_halves(h["odd_2023"], h["even_2023"], h["odd_2024"], h["even_2024"])
    same_window = track.first == track.last
    consistent = rel & (t23 == track.first) & (t24 == track.last)
    return Change(
        boot=boot,
        reliable=rel,
        t23=t23,
        t24=t24,
        matrix=T.transition_matrix(track.first, track.last, k),
        reliable_matrix=T.transition_matrix(t23[rel], t24[rel], k),
        window_reliable_matrix=T.transition_matrix(track.first[consistent], track.last[consistent], k),
        n_reliable_window_same=int((rel & same_window).sum()),
        n_reliable_window_other=int((rel & ~same_window & ~consistent).sum()),
        noise_shares=tuple(T.share_changed(x, y, k) for x, y in pairs),
    )


def events_table(track: Track, k: int, taus: Sequence[float], windows: Sequence[str]) -> pd.DataFrame:
    """События MONIC: соседние окна, первое → последнее окно и пары шума (половины одного года)."""
    pairs = [
        (f"{windows[i]} → {windows[i + 1]}", "adjacent", track.rolling[i], track.rolling[i + 1])
        for i in range(len(windows) - 1)
    ]
    pairs.append(("2023 → 2024", "first_last", track.first, track.last))
    for y in (2023, 2024):
        pairs.append(
            (f"{y}: нечётные → чётные", "noise", track.halves[f"odd_{y}"], track.halves[f"even_{y}"])
        )
    rows = []
    for name, kind, a, b in pairs:
        for tau in taus:
            for e in T.monic_events(a, b, k, k, tau):
                rows.append(
                    {
                        "approach": track.approach,
                        "pair": name,
                        "pair_kind": kind,
                        "tau": tau,
                        "event": e.kind,
                        "before": " ".join(str(i + 1) for i in e.before),
                        "after": " ".join(str(i + 1) for i in e.after),
                        "jaccard": e.jaccard,
                    }
                )
    return pd.DataFrame(rows)


def adjacent_table(track: Track, k: int, windows: Sequence[str]) -> pd.DataFrame:
    rows = []
    for i in range(len(windows)):
        lab = track.rolling[i]
        row = {"approach": track.approach, "window": windows[i], "index": i + 1}
        for j in range(k):
            row[f"size_{j + 1}"] = int((lab == j).sum())
        row["share_changed_prev"] = float(np.mean(lab != track.rolling[i - 1])) if i else np.nan
        row["share_changed_first"] = float(np.mean(lab != track.first))
        rows.append(row)
    return pd.DataFrame(rows)


# --- Проверка сюжета -----------------------------------------------------------------------------


@dataclass(frozen=True)
class Driver:
    table: pd.DataFrame  # часть корзины (и уровень) -> медианы |Δ|, отношение, U, p
    part: str
    p_value: float
    significant: bool
    top_part: str  # часть корзины с наибольшим отношением медиан
    expectation_met: bool  # у part наибольшее отношение из шести частей
    n_transition: int
    n_same: int


def driver_test(
    dB: np.ndarray,
    d_level: np.ndarray,
    parts: Sequence[str],
    transition: np.ndarray,
    same: np.ndarray,
    part: str,
    alpha: float,
) -> Driver:
    """|изменение 2023 → 2024| у узлов с надёжным переходом против узлов без смены типа: Манн — Уитни
    (односторонний, «у перешедших больше») по каждой части корзины и по уровню трат; отношение медиан."""
    rows = []
    cols = [(f"clr_rel_{q}", dB[:, j]) for j, q in enumerate(parts)] + [("log_level_rel", d_level)]
    for name, v in cols:
        x, y = np.abs(v[transition]), np.abs(v[same])
        res = mannwhitneyu(x, y, alternative="greater")
        mx, my = float(np.median(x)), float(np.median(y))
        rows.append(
            {
                "feature": name,
                "is_part": name.startswith("clr_rel_"),
                "median_abs_transition": mx,
                "median_abs_same": my,
                "ratio": mx / my if my > 0 else np.nan,
                "median_transition": float(np.median(v[transition])),
                "median_same": float(np.median(v[same])),
                "u": float(res.statistic),
                "auc": float(res.statistic) / (len(x) * len(y)),
                "p_value": float(res.pvalue),
            }
        )
    tab = pd.DataFrame(rows)
    partrows = tab.loc[tab["is_part"]]
    top = str(partrows.loc[partrows["ratio"].idxmax(), "feature"]).removeprefix("clr_rel_")
    p = float(tab.loc[tab["feature"] == f"clr_rel_{part}", "p_value"].iloc[0])
    return Driver(
        table=tab,
        part=part,
        p_value=p,
        significant=p < alpha,
        top_part=top,
        expectation_met=top == part,
        n_transition=int(transition.sum()),
        n_same=int(same.sum()),
    )


def flow_profiles(
    dB: np.ndarray,
    d_level: np.ndarray,
    parts: Sequence[str],
    t23: np.ndarray,
    t24: np.ndarray,
    mask: np.ndarray,
    same: np.ndarray,
) -> pd.DataFrame:
    """Медианы изменения частей корзины и уровня трат 2023 → 2024 по надёжным переходам «тип → тип»
    и у узлов без смены."""
    rows = []
    groups = [
        (f"{a + 1} → {b + 1}", mask & (t23 == a) & (t24 == b))
        for a, b in sorted({(int(a), int(b)) for a, b in zip(t23[mask], t24[mask], strict=True)})
    ]
    groups.append(("без смены", same))
    for name, m in groups:
        row = {"flow": name, "n": int(m.sum())}
        for j, q in enumerate(parts):
            row[f"d_{q}"] = float(np.median(dB[m, j]))
        row["d_level"] = float(np.median(d_level[m]))
        rows.append(row)
    return pd.DataFrame(rows)


# --- Сравнение способов --------------------------------------------------------------------------


def transition_code(a: np.ndarray, b: np.ndarray, k: int) -> np.ndarray:
    return np.asarray(a) * k + np.asarray(b)


def compare_tracks(tracks: Mapping[str, Track], changes: Mapping[str, Change], k: int) -> pd.DataFrame:
    """Согласие способов с основным: ARI меток переходов «тип 2023 × тип 2024», Жаккар множеств сменивших
    тип, доля надёжных переходов основного способа, которые способ повторяет (тот же узел, те же типы)."""
    base, cb = tracks[MAIN], changes[MAIN]
    code0 = transition_code(base.first, base.last, k)
    ch0 = base.first != base.last
    rel0 = cb.reliable
    rows = []
    for name, tr in tracks.items():
        c = changes[name]
        code = transition_code(tr.first, tr.last, k)
        ch = tr.first != tr.last
        same_flow = (tr.first == cb.t23) & (tr.last == cb.t24)
        both_rel = rel0 & c.reliable & (c.t23 == cb.t23) & (c.t24 == cb.t24)
        union = (ch | ch0).sum()
        adj = [float(np.mean(tr.rolling[i] != tr.rolling[i + 1])) for i in range(len(tr.rolling) - 1)]
        rows.append(
            {
                "approach": name,
                "share_changed": c.boot.change,
                "noise": c.boot.noise,
                "diff": c.boot.diff,
                "lo": c.boot.lo,
                "hi": c.boot.hi,
                "exceeds": c.boot.exceeds,
                "n_reliable": int(c.reliable.sum()),
                "adjacent_mean": float(np.mean(adj)),
                "adjacent_max": float(np.max(adj)),
                "ari_transitions": float(adjusted_rand_score(code0, code)),
                "jaccard_changed": float((ch & ch0).sum() / union) if union else np.nan,
                "main_reliable_reproduced": float(same_flow[rel0].mean()) if rel0.any() else np.nan,
                "main_reliable_also_reliable": float(both_rel[rel0].mean()) if rel0.any() else np.nan,
                "ari_first_vs_main": float(adjusted_rand_score(base.first, tr.first)),
                "ari_last_vs_main": float(adjusted_rand_score(base.last, tr.last)),
            }
        )
    return pd.DataFrame(rows)
