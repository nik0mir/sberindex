"""Проверка ``usefulness.by_type_test``: зависит ли польза сверки со своим регионом от типа МО.

Правила записаны в подблоке ``usefulness.by_type_test`` конфига отдельным коммитом 01.10.2026 ДО расчёта
(редакция 3); код написан вслепую — по тексту блока, без взгляда на новые цели. Вызывается из
``munnet.usefulness.run`` после существующих расчётов этапа и пишет только свои файлы:
``<paths.outputs>/usefulness/by_type.json``, ``by_type_runs.csv`` (прогон × цель) и ``by_type_by_mo.csv``
(МО × цель); ``facts.json``, ``rule_by_mo.csv`` и ``mo_flags.csv`` этапа не трогает.

Что считается:

1. **Цели.** Изменение 2023 → 2024 трёх новых показателей (оборот общепита, доход 5-НДФЛ — только при
   ``ndfl_ok`` в оба года, отгрузка) на жителя: ln(1 + x) 2024 минус 2023; рядом — виденная розница той же
   статистикой. Для каждой цели — свои МО общего набора T7: цель известна в оба года, у наборов B (соседи
   по региону) и D (похожие по тратам МО других регионов, ``node_comparable.csv``) не меньше ``min_set``
   членов с известной целью. Ошибка — |y − медиана y членов набора|; рядом — набор R (случайные МО своей
   группы региона, как ``usefulness.region_random_errors``).
2. **Статистика.** «B ближе D» — err_B < err_D (с допуском ``usefulness.rule_share.tie_tol``, 02.10). Δ_t —
   доля в группе типов {3,4} минус доля в {1,2} (общая доля по всем МО группы); решает Δ — среднее Δ_t по трём
   целям. Уточнения: Δ внутри статуса МО (страты «городской округ» / «остальные», гармонические веса) и
   деление «городской округ / остальные».
3. **Вывод.** Один общий бутстрап по группам региона (группы целиком, с возвращением; одна выборка групп
   на все цели и все статистики; выборка с пустой группой {3,4} или {1,2} у какой-либо цели
   отбрасывается) и одна общая перестановка меток типов внутри групп региона (одни и те же метки на все
   цели) — у каждого свой ``default_rng(seed)``; p по формулам блока, интервал с поправкой центра нуля
   (только строже), вердикт confirmed / not / reverse.
4. **Устойчивость.** Вердикт — в 8 прогонах (основной, 3 варианта и 4 повтора seed, группы — по
   ``matched_type`` из ``node_r1.csv``); в главный текст — наименьший, числа — из прогона, давшего его.
5. **QC.** Ключи блока сверяются с записанными (``EXPECTED``); на рознице тем же кодом Δ должна совпасть
   со счётчиками ``rule.by_type`` этапа usefulness; в описаниях отдельных целей нет запрещённых корней;
   в текстах не осталось пустых полей {…}. Не сошлось — ``QCError`` (код 3).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from munnet.config import Config
from munnet.contracts import MissingInputError, QCError
from munnet.style import fmt_p, fmt_pct, fmt_pp, fmt_range
from munnet.usefulness import (  # usefulness импортирует этот модуль внутри run()
    compare_errors,
    region_random_errors,
)

log = logging.getLogger(__name__)

# Ключи подблока by_type_test, которые код исполняет буквально (записаны до расчёта, редакция 3). Другое
# значение — код 3, а не молчаливое исполнение другого правила. Тексты words и edits не сверяются:
# они подставляются как есть.
EXPECTED: dict[tuple[str, ...], Any] = {
    ("targets", "catering", "source"): "context_annual.catering_turnover_pc",
    ("targets", "ndfl", "source"): "context_annual.ndfl_income_pc",
    ("targets", "ndfl", "filter"): "ndfl_ok_both_years",
    ("targets", "shipments", "source"): "context_annual.shipments_pc",
    ("targets", "transform"): "log1p",
    ("targets", "change"): "diff",
    ("targets", "retail_seen"): "show",
    ("universe", "base"): "outputs/interpret/t7_errors.csv",
    ("universe", "target_known"): "both_years",
    ("universe", "members"): "outputs/interpret/node_comparable.csv",
    ("universe", "min_set"): 5,
    ("universe", "R"): {"pool": "region_group_known_target", "size": "known_B_size", "draws": 100},
    ("universe", "types"): "outputs/interpret/types.csv",
    ("universe", "groups"): {"high": [3, 4], "low": [1, 2]},
    ("universe", "status"): {
        "source": "features_nodes.mo_type",
        "strata": {"go": ["go"], "other": ["mo", "mr"]},
    },
    ("statistic", "works"): "strict_less",
    ("statistic", "per_target"): "share_diff",
    ("statistic", "main"): "mean_of_three",
    ("statistic", "within_status"): {"weights": "harmonic"},
    ("statistic", "rival"): {"name": 'mo_type == "go"', "statistic": "share_diff"},
    ("statistic", "share_hi"): "mean_of_three",
    ("statistic", "describe"): [
        "share_by_type",
        "per_target_delta",
        "median_d_diff",
        "vs_R",
        "noise",
        "set_sizes",
    ],
    ("statistic", "noise_flag_lo"): 1.5,
    ("statistic", "noise_flag_hi"): 1.5,
    ("statistic", "bootstrap"): {"n": 2000, "by": "region_group", "empty_group": "drop"},
    ("statistic", "ci_level"): 0.95,
    ("statistic", "permutation"): {
        "within": "region_group",
        "n": 2000,
        "alpha": 0.05,
        "p_less": "(#{Δ_perm ≤ Δ} + 1) / (n + 1)",
        "p_greater": "(#{Δ_perm ≥ Δ} + 1) / (n + 1)",
        "center": "worst_for_hypothesis",
    },
    ("statistic", "mde"): "2,80 × SD бутстрапа Δ + max(0, −m0)",
    ("rule", "levels"): ["reverse", "not", "confirmed"],
    ("rule", "confirmed"): "верхняя граница интервала Δ с поправкой центра < 0 и p_less < alpha",
    ("rule", "reverse"): "нижняя граница интервала Δ с поправкой центра > 0 и p_greater < alpha",
    ("rule", "not"): "иначе",
    ("rule", "robustness", "required"): "lowest_verdict",
    ("rule", "beyond_status"): {"statistic": "within_status", "adds": "upper_lt_0"},
    ("rule", "go_below"): "верхняя граница интервала Δ_go < 0 (без поправки центра)",
    ("rule", "share_hi_more_often"): "нижняя граница 95% интервала share_hi (тот же бутстрап) > 0,5",
    ("words", "forbidden_per_target"): ["подтвержд", "видно", "различа"],
    ("qc", "recompute_retail"): 1.0e-12,
    ("qc", "spec_frozen"): True,
}
TARGETS = ("catering", "ndfl", "shipments")  # решают среднее; порядок — как в блоке
SEEN = "retail"  # виденная цель, источник гипотезы: та же статистика рядом, в вердикт не входит
COLUMNS = {  # targets.<t>.source без префикса «context_annual.»; розница — как usefulness.TARGET_COL
    "catering": "catering_turnover_pc",
    "ndfl": "ndfl_income_pc",
    "shipments": "shipments_pc",
    "retail": "retail_pc",
}
NDFL_FILTER = "ndfl_ok"  # targets.ndfl.filter = ndfl_ok_both_years
TARGET_NAMES = {
    "catering": "оборот общепита",
    "ndfl": "доход по 5-НДФЛ",
    "shipments": "отгрузка",
    "retail": "розничный оборот",
}
SEEN_LABEL = "виденная цель, источник гипотезы"
PP_END = "п. п."  # хвост fmt_pp
MDE_FACTOR = 2.80  # statistic.mde: «2,80 × SD бутстрапа Δ + max(0, −m0)»
YEARS = (2023, 2024)  # interpret.windows.change_years — сверяется в usefulness.run


def _get(block: Mapping, path: tuple[str, ...]) -> Any:
    node: Any = block
    for k in path:
        node = node.get(k) if isinstance(node, Mapping) else None
    return node


def check_spec(block: Mapping | None) -> None:
    """Ключи подблока ``by_type_test`` равны записанным до расчёта (``EXPECTED``); иначе ``QCError``."""
    if not isinstance(block, Mapping):
        raise QCError("usefulness.by_type_test: подблока нет в конфиге, а проверка предрегистрирована")
    bad = [
        f"{'.'.join(p)} = {_get(block, p)!r} (ожидалось {v!r})"
        for p, v in EXPECTED.items()
        if _get(block, p) != v
    ]
    if bad:
        raise QCError(
            "usefulness.by_type_test: блок конфига не совпадает с реализованным правилом: " + "; ".join(bad)
        )


# --- Цели и ошибки ---------------------------------------------------------------------------------------


def target_change(
    context: pd.DataFrame, column: str, filter_col: str | None = None, years: tuple[int, int] = YEARS
) -> pd.Series:
    """Цель: ln(1 + x) года b минус года a, индекс — territory_id. Пропуск в любом году — NaN; с
    ``filter_col`` цель известна, только если флаг истинен в оба года (иначе NaN)."""
    ya, yb = years
    c = context.loc[context["year"].isin([ya, yb])]
    w = c.pivot(index="territory_id", columns="year", values=column).reindex(columns=[ya, yb])
    y = np.log1p(w[yb].astype(float)) - np.log1p(w[ya].astype(float))
    if filter_col is not None:
        f = c.pivot(index="territory_id", columns="year", values=filter_col).reindex(columns=[ya, yb])
        ok = (f[ya].astype("boolean").fillna(False) & f[yb].astype("boolean").fillna(False)).astype(bool)
        y = y.where(ok.reindex(y.index).fillna(False).to_numpy())
    return y.rename("y")


def set_errors(
    y: pd.Series, comparable: pd.DataFrame, ids: np.ndarray, set_name: str
) -> tuple[np.ndarray, np.ndarray]:
    """Ошибка набора у каждого МО ``ids``: |y − медиана y членов с известной целью| и число таких членов.
    Нет ни одного члена с целью или нет цели у самого МО — ошибка NaN."""
    m = comparable.loc[comparable["set"] == set_name, ["territory_id", "other_id"]].copy()
    m["y"] = y.reindex(m["other_id"].to_numpy()).to_numpy()
    m = m.dropna(subset=["y"])
    g = m.groupby("territory_id")["y"]
    med = g.median().reindex(ids).to_numpy(dtype=float)
    size = g.size().reindex(ids).fillna(0).to_numpy(dtype=np.int64)
    own = y.reindex(ids).to_numpy(dtype=float)
    return np.abs(own - med), size


def target_table(y: pd.Series, comparable: pd.DataFrame, ids: np.ndarray, min_set: int) -> pd.DataFrame:
    """МО общего набора × одна цель: y, размеры B и D (члены с целью), ошибки, вход в расчёт цели и причина
    выбытия: ``no_target`` — нет цели у самого МО, ``min_set`` — у B или D меньше ``min_set`` членов
    с целью."""
    eb, nb = set_errors(y, comparable, ids, "B")
    ed, nd = set_errors(y, comparable, ids, "D")
    own = y.reindex(ids).to_numpy(dtype=float)
    known = np.isfinite(own)
    big = (nb >= min_set) & (nd >= min_set)
    reason = np.where(~known, "no_target", np.where(~big, "min_set", ""))
    return pd.DataFrame(
        {
            "territory_id": ids,
            "y": own,
            "size_B": nb,
            "size_D": nd,
            "err_B": eb,
            "err_D": ed,
            "in_target": known & big,
            "drop_reason": reason,
        }
    )


# --- Общий бутстрап и общая перестановка ------------------------------------------------------------------


def bootstrap_counts(groups: np.ndarray, n_boot: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Бутстрап по группам: группы целиком, с возвращением, свой ``default_rng(seed)``. Возвращает индекс
    группы каждого МО (по возрастанию кодов групп) и матрицу ``n_boot × число групп`` — сколько раз группа
    попала в выборку. Розыгрыш тот же, что в ``usefulness.group_bootstrap`` (``integers(0, G, G)``)."""
    uniq, gidx = np.unique(groups, return_inverse=True)
    rng = np.random.default_rng(seed)
    n_g = len(uniq)
    counts = np.empty((n_boot, n_g), dtype=np.float64)
    for b in range(n_boot):
        counts[b] = np.bincount(rng.integers(0, n_g, n_g), minlength=n_g)
    return gidx, counts


def permutation_index(groups: np.ndarray, n_perm: int, seed: int) -> np.ndarray:
    """Перестановки внутри групп: ``n_perm × N`` индексов, ``labels[P[b]]`` — метки b-й перестановки
    (метка МО i берётся у МО ``P[b, i]`` той же группы). Свой ``default_rng(seed)``; одна и та же матрица
    применяется ко всем целям — метки одни на все цели."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    members = [np.flatnonzero(groups == g) for g in uniq]
    out = np.empty((n_perm, len(groups)), dtype=np.int64)
    for b in range(n_perm):
        idx = np.arange(len(groups))
        for m in members:
            idx[m] = rng.permutation(m)
        out[b] = idx
    return out


Summer = Callable[[np.ndarray], np.ndarray]


def point_sum(v: np.ndarray) -> np.ndarray:
    """Сумма индикатора по МО (по последней оси: годится и для матрицы меток перестановок)."""
    return np.asarray(v, dtype=np.float64).sum(axis=-1)


def boot_sum(gidx: np.ndarray, counts: np.ndarray) -> Summer:
    """Сумма индикатора в каждой бутстрап-выборке: Σ_g (сколько раз взята группа g) × (сумма по МО группы)."""
    n_g = counts.shape[1]

    def f(v: np.ndarray) -> np.ndarray:
        return counts @ np.bincount(gidx, weights=np.asarray(v, dtype=np.float64), minlength=n_g)

    return f


def _ratio(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(b > 0, a / np.where(b > 0, b, 1.0), np.nan)


def group_share(S: Summer, member: np.ndarray, works: np.ndarray, mask: np.ndarray) -> tuple:
    """Доля «B ближе D» в группе: сработало / n группы (общая доля по всем МО, без весов). Возвращает
    (доля, n); пустая группа — NaN."""
    m = member & mask
    n = S(m)
    return _ratio(S(m & works), n), n


def share_diff(
    S: Summer, hi: np.ndarray, lo: np.ndarray, works: np.ndarray, mask: np.ndarray
) -> dict[str, np.ndarray]:
    """Δ_t = доля в {3,4} минус доля в {1,2}; ``hi``/``lo`` — индикаторы групп (вектор или матрица меток)."""
    s_hi, n_hi = group_share(S, hi, works, mask)
    s_lo, n_lo = group_share(S, lo, works, mask)
    return {"share_hi": s_hi, "share_lo": s_lo, "delta": s_hi - s_lo, "n_hi": n_hi, "n_lo": n_lo}


def status_delta(
    S: Summer,
    hi: np.ndarray,
    lo: np.ndarray,
    works: np.ndarray,
    mask: np.ndarray,
    strata: Sequence[np.ndarray],
) -> np.ndarray:
    """Δ внутри статуса: Σ_s h_s·(доля{3,4}_s − доля{1,2}_s) / Σ_s h_s, h_s = n_a·n_b / (n_a + n_b);
    у страты без одной из групп h_s = 0; Σ h_s = 0 — NaN."""
    num, den = 0.0, 0.0
    for s in strata:
        d = share_diff(S, hi, lo, works, mask & s)
        n_a, n_b = d["n_hi"], d["n_lo"]
        both = (n_a > 0) & (n_b > 0)
        h = np.where(both, _ratio(n_a * n_b, n_a + n_b), 0.0)
        num = num + np.where(both, h * np.nan_to_num(d["delta"]), 0.0)
        den = den + h
    return _ratio(np.asarray(num, dtype=np.float64), np.asarray(den, dtype=np.float64))


def go_delta(S: Summer, is_go: np.ndarray, works: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Δ_go = доля «B ближе D» у городских округов минус у остальных МО цели."""
    s_go, _ = group_share(S, is_go, works, mask)
    s_ot, _ = group_share(S, ~is_go, works, mask)
    return s_go - s_ot


# --- Интервалы, p, вердикт ----------------------------------------------------------------------------------


def ci(draws: np.ndarray, level: float) -> list[float]:
    """Двусторонний перцентильный интервал по конечным значениям (как ``usefulness.ci``)."""
    a = (1.0 - level) / 2.0
    d = np.asarray(draws, dtype=float)
    d = d[np.isfinite(d)]
    if not len(d):
        return [float("nan"), float("nan")]
    return [float(np.quantile(d, a)), float(np.quantile(d, 1.0 - a))]


def center_adjusted(interval: Sequence[float], m0: float) -> list[float]:
    """Поправка центра нуля, только строже: верхняя = max(U, U − m0), нижняя = min(L, L − m0)."""
    lo, hi = float(interval[0]), float(interval[1])
    return [min(lo, lo - m0), max(hi, hi - m0)]


def perm_p(null: np.ndarray, obs: float) -> dict[str, float]:
    """p_less = (#{Δ_perm ≤ Δ} + 1)/(n + 1), p_greater = (#{Δ_perm ≥ Δ} + 1)/(n + 1). Неопределённая Δ_perm
    (пустая группа после перестановки) считается в пользу нуля в обоих p — строже."""
    null = np.asarray(null, dtype=float)
    bad = ~np.isfinite(null)
    n = len(null)
    le = int(((null <= obs) & ~bad).sum() + bad.sum())
    ge = int(((null >= obs) & ~bad).sum() + bad.sum())
    return {"p_less": (le + 1) / (n + 1), "p_greater": (ge + 1) / (n + 1), "n_undefined": int(bad.sum())}


def mde(sd_boot: float, m0: float) -> float:
    """{mde} = 2,80 × SD бутстрапа Δ + max(0, −m0), по модулю."""
    return abs(MDE_FACTOR * float(sd_boot) + max(0.0, -float(m0)))


def verdict(ci_adj: Sequence[float], p_less: float, p_greater: float, alpha: float) -> str:
    """confirmed: верхняя граница интервала с поправкой < 0 и p_less < alpha; reverse: нижняя > 0
    и p_greater < alpha; иначе not. Неопределённая граница — not."""
    lo, hi = ci_adj
    if np.isfinite(hi) and hi < 0 and p_less < alpha:
        return "confirmed"
    if np.isfinite(lo) and lo > 0 and p_greater < alpha:
        return "reverse"
    return "not"


def lowest_verdict(verdicts: Sequence[str], levels: Sequence[str]) -> int:
    """Номер прогона с наименьшим вердиктом по порядку ``levels``; при равенстве — первый в порядке прогонов
    (основной, затем варианты, затем повторы seed)."""
    rank = [list(levels).index(v) for v in verdicts]
    return int(np.argmin(rank))  # argmin берёт первый из равных


def noise_flags(
    med: Mapping[str, Mapping[str, Mapping[str, float]]], ratio_lo: float, ratio_hi: float
) -> dict[str, list[str]]:
    """Флаги шума по медианам ошибок ``med[цель][группа][набор]`` (группы ``hi``/``lo``, наборы B/D):
    ``lo`` — у {1,2} медиана err_B или err_D больше ratio_lo × медианы у {3,4} хотя бы по одной цели;
    ``hi`` — наоборот. Возвращает перечни целей."""
    out: dict[str, list[str]] = {"lo": [], "hi": []}
    for t, g in med.items():
        if any(g["lo"][s] > ratio_lo * g["hi"][s] for s in ("B", "D")):
            out["lo"].append(t)
        if any(g["hi"][s] > ratio_hi * g["lo"][s] for s in ("B", "D")):
            out["hi"].append(t)
    return out


def fill(template: str, fields: Mapping[str, str]) -> str:
    """Подстановка полей {…}; неизвестное поле или оставшаяся скобка — ``QCError``."""
    try:
        text = str(template).format_map(dict(fields))
    except (KeyError, IndexError, ValueError) as e:
        raise QCError(f"usefulness.by_type_test: в тексте поле без значения ({e}): {template[:80]}…") from e
    if "{" in text or "}" in text:
        raise QCError(f"usefulness.by_type_test: в тексте осталась скобка {{…}}: {text[:80]}…")
    # поле, кончающееся сокращением «п. п.», в конце предложения: точка одна (типографика, смысл тот же)
    return text.replace(PP_END + ".", PP_END)


def check_forbidden(text: str, forbidden: Sequence[str]) -> None:
    """В описании отдельной цели нет запрещённых корней (без учёта регистра); иначе ``QCError``."""
    low = text.lower()
    hit = [w for w in forbidden if w.lower() in low]
    if hit:
        raise QCError(f"usefulness.by_type_test: в описании цели запрещённые корни {hit}: {text[:80]}…")


# --- Один прогон --------------------------------------------------------------------------------------------


def _med(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    return float(np.median(x)) if len(x) else float("nan")


def _mad(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    return float(1.4826 * np.median(np.abs(x - np.median(x)))) if len(x) else float("nan")


def null_deltas(
    data: Mapping[str, Mapping[str, np.ndarray]],
    labels: np.ndarray,
    perm: np.ndarray,
    high: Sequence[int],
    low: Sequence[int],
    strata: Sequence[np.ndarray],
) -> dict[str, dict[str, np.ndarray]]:
    """Нуль перестановок: одна матрица ``perm`` — одни и те же переставленные метки ``labels[perm]`` для всех
    трёх целей; по ним Δ_t и Δ_t внутри статуса каждой цели (по строке на перестановку)."""
    lab_p = labels[perm]
    hi_p, lo_p = np.isin(lab_p, high), np.isin(lab_p, low)
    return {
        "delta": {
            t: share_diff(point_sum, hi_p, lo_p, data[t]["works"], data[t]["mask"])["delta"] for t in TARGETS
        },
        "status": {
            t: status_delta(point_sum, hi_p, lo_p, data[t]["works"], data[t]["mask"], strata) for t in TARGETS
        },
    }


def evaluate_run(
    data: Mapping[str, Mapping[str, np.ndarray]],
    labels: np.ndarray,
    design: Mapping[str, Any],
) -> dict[str, Any]:
    """Все числа одного прогона (основной расчёт, вариант или повтор seed) по меткам типов ``labels``.

    ``data[цель]`` — массивы по МО общего набора: ``mask`` (МО в расчёте цели), ``works`` (err_B < err_D),
    ``err_B``, ``err_D``, ``err_R``, ``y``, ``size_B``, ``size_D``. ``design`` — общее для всех прогонов:
    группы {3,4}/{1,2}, страты статуса, ``is_go``, бутстрап (``gidx``, ``counts``), матрица перестановок
    ``perm``, уровни интервала и alpha, пороги шума, типы для описания."""
    hi_set, lo_set = design["high"], design["low"]
    tol = float(design["tie_tol"])  # usefulness.rule_share.tie_tol — сравнение ошибок, как в rule_share
    hi, lo = np.isin(labels, hi_set), np.isin(labels, lo_set)
    strata, is_go = design["strata"], design["is_go"]
    level, alpha = float(design["level"]), float(design["alpha"])
    S_b = boot_sum(design["gidx"], design["counts"])
    all_t = [*TARGETS, SEEN] if SEEN in data else list(TARGETS)

    point: dict[str, dict] = {}
    boot: dict[str, dict] = {}
    for t in all_t:
        d = data[t]
        point[t] = {
            **share_diff(point_sum, hi, lo, d["works"], d["mask"]),
            "status": status_delta(point_sum, hi, lo, d["works"], d["mask"], strata),
            "go": go_delta(point_sum, is_go, d["works"], d["mask"]),
        }
        boot[t] = {
            **share_diff(S_b, hi, lo, d["works"], d["mask"]),
            "status": status_delta(S_b, hi, lo, d["works"], d["mask"], strata),
            "go": go_delta(S_b, is_go, d["works"], d["mask"]),
        }

    def mean3(src: Mapping[str, dict], key: str) -> np.ndarray:
        return np.mean([np.asarray(src[t][key], dtype=float) for t in TARGETS], axis=0)

    # выборка, где у какой-либо из трёх целей пуста одна из групп, отбрасывается целиком
    drop = np.zeros(design["counts"].shape[0], dtype=bool)
    for t in TARGETS:
        drop |= (boot[t]["n_hi"] == 0) | (boot[t]["n_lo"] == 0)
    keep = ~drop
    b_delta = mean3(boot, "delta")[keep]
    b_status = mean3(boot, "status")[keep]
    b_go = mean3(boot, "go")[keep]
    b_share_hi = mean3(boot, "share_hi")[keep]

    nd = null_deltas(data, labels, design["perm"], hi_set, lo_set, strata)
    null, null_status = nd["delta"], nd["status"]
    n_delta = np.mean([null[t] for t in TARGETS], axis=0)
    n_status = np.mean([null_status[t] for t in TARGETS], axis=0)
    m0 = float(np.nanmean(n_delta)) if np.isfinite(n_delta).any() else float("nan")
    m0_status = float(np.nanmean(n_status)) if np.isfinite(n_status).any() else float("nan")

    delta = float(mean3(point, "delta"))
    delta_status = float(mean3(point, "status"))
    delta_go = float(mean3(point, "go"))
    ci_raw = ci(b_delta, level)
    ci_adj = center_adjusted(ci_raw, m0)
    ci_status_raw = ci(b_status, level)
    ci_status_adj = center_adjusted(ci_status_raw, m0_status)
    ci_go = ci(b_go, level)
    share_hi_ci = ci(b_share_hi, level)
    p = perm_p(n_delta, delta)
    sd_boot = float(np.std(b_delta, ddof=1)) if len(b_delta) > 1 else float("nan")
    v = verdict(ci_adj, p["p_less"], p["p_greater"], alpha)

    # описание по целям (без слов вердикта)
    per_target: dict[str, dict] = {}
    med_err: dict[str, dict] = {}
    for t in all_t:
        d, pt, bt = data[t], point[t], boot[t]
        m = d["mask"]
        grp = {"hi": m & hi, "lo": m & lo}
        by_type = []
        for k in design["types"]:
            mk = labels == k
            s_pt, n_pt = group_share(point_sum, mk, d["works"], m)
            s_bt, _ = group_share(S_b, mk, d["works"], m)
            by_type.append(
                {
                    "type": int(k),
                    "n": int(n_pt),
                    "works": int((mk & m & d["works"]).sum()),
                    "share": float(s_pt),
                    "share_ci": ci(s_bt[keep], level),
                }
            )
        groups: dict[str, dict] = {}
        for gname, gm in grp.items():
            eb, ed, er = d["err_B"][gm], d["err_D"][gm], d["err_R"][gm]
            b_d, tie_bd = compare_errors(eb, ed, tol)
            b_r, tie_br = compare_errors(eb, er, tol)
            r_d, _ = compare_errors(er, ed, tol)
            groups[gname] = {
                "n": int(gm.sum()),
                "works": int((gm & d["works"]).sum()),
                "share": float(np.mean(b_d)) if gm.any() else float("nan"),
                "share_B_closer_than_R": float(np.mean(b_r)) if gm.any() else float("nan"),
                "share_R_closer_than_D": float(np.mean(r_d)) if gm.any() else float("nan"),
                "median_d": _med(eb - ed),
                "mad_change": _mad(d["y"][gm]),
                "median_err_B": _med(eb),
                "median_err_D": _med(ed),
                "ties_B_eq_D": int(tie_bd.sum()),
                "ties_B_eq_R": int(tie_br.sum()),  # добавлено 02.10 вместе с допуском tie_tol
                "median_size_B": _med(d["size_B"][gm].astype(float)),
                "median_size_D": _med(d["size_D"][gm].astype(float)),
            }
        med_err[t] = {
            g: {"B": groups[g]["median_err_B"], "D": groups[g]["median_err_D"]} for g in ("hi", "lo")
        }
        per_target[t] = {
            "n": int(m.sum()),
            "delta": float(pt["delta"]),
            "delta_ci": ci(bt["delta"][keep], level),
            "delta_status": float(pt["status"]),
            "delta_status_ci": ci(bt["status"][keep], level),
            "delta_go": float(pt["go"]),
            "delta_go_ci": ci(bt["go"][keep], level),
            "median_d_diff": groups["hi"]["median_d"] - groups["lo"]["median_d"],
            "null_mean": float(np.nanmean(null[t])) if t in null else None,
            "groups": groups,
            "share_by_type": by_type,
        }
    flags = noise_flags(
        {t: med_err[t] for t in TARGETS}, float(design["noise_lo"]), float(design["noise_hi"])
    )
    return {
        "verdict": v,
        "delta": delta,
        "delta_ci": ci_raw,
        "delta_ci_adj": ci_adj,
        "p_less": p["p_less"],
        "p_greater": p["p_greater"],
        "null_mean": m0,
        "null_sd": float(np.nanstd(n_delta, ddof=1)),
        "null_undefined": p["n_undefined"],
        "sd_boot": sd_boot,
        "mde": mde(sd_boot, m0),
        "delta_status": delta_status,
        "delta_status_ci": ci_status_raw,
        "delta_status_ci_adj": ci_status_adj,
        "null_mean_status": m0_status,
        "status_adds": bool(np.isfinite(ci_status_adj[1]) and ci_status_adj[1] < 0),
        "delta_go": delta_go,
        "delta_go_ci": ci_go,
        "go_below": bool(np.isfinite(ci_go[1]) and ci_go[1] < 0),
        "share_hi": float(mean3(point, "share_hi")),
        "share_lo": float(mean3(point, "share_lo")),
        "share_hi_ci": share_hi_ci,
        "share_hi_more_often": bool(np.isfinite(share_hi_ci[0]) and share_hi_ci[0] > 0.5),
        "noise_lo_targets": flags["lo"],
        "noise_hi_targets": flags["hi"],
        "bootstrap_kept": int(keep.sum()),
        "bootstrap_dropped": int(drop.sum()),
        "bootstrap_status_undefined": int((~np.isfinite(b_status)).sum()),
        "bootstrap_go_undefined": int((~np.isfinite(b_go)).sum()),
        "per_target": per_target,
    }


# --- Тексты -------------------------------------------------------------------------------------------------


def text_fields(run: Mapping[str, Any], outcome: str) -> dict[str, str]:
    """Поля текстов из чисел одного прогона: разности и интервалы — в п. п., доли — в процентах. {p} —
    p_greater при reverse, иначе p_less; {noise_targets} — цели флага lo при not/reverse, hi при confirmed."""
    pp = 100.0

    def rng_pp(iv: Sequence[float]) -> str:
        return fmt_range(pp * iv[0], pp * iv[1], fmt=fmt_pp, decimals=1, sign=False)

    noise = run["noise_hi_targets"] if outcome == "confirmed" else run["noise_lo_targets"]
    return {
        "delta": fmt_pp(pp * run["delta"]),
        "ci": rng_pp(run["delta_ci_adj"]),
        "p": fmt_p(run["p_greater"] if outcome == "reverse" else run["p_less"]),
        "mde": fmt_pp(pp * run["mde"], sign=False),
        "delta_status": fmt_pp(pp * run["delta_status"]),
        "ci_status": rng_pp(run["delta_status_ci_adj"]),
        "delta_go": fmt_pp(pp * run["delta_go"]),
        "ci_go": rng_pp(run["delta_go_ci"]),
        "share_hi": fmt_pct(run["share_hi"]),
        "share_lo": fmt_pct(run["share_lo"]),
        "noise_targets": ", ".join(TARGET_NAMES[t] for t in noise) or "—",
    }


def outcome_texts(run: Mapping[str, Any], words: Mapping[str, Any], edits: Mapping[str, Any]) -> dict:
    """Текст исхода и правки по вердикту прогона: confirmed — основной текст, status_adds или status_same
    (при go_below — и status_same_go), оговорка noise_caveat_hi; not — оговорка noise_caveat_lo и
    not_go_below вместо go_always при go_below; reverse — noise_caveat_lo. go_always — рядом при любом
    исходе, кроме замены not_go_below. Правка при confirmed — share_hi_more_often, если нижняя граница
    share_hi > 0,5, иначе status_adds / status_same."""
    v = run["verdict"]
    f = text_fields(run, v)
    parts = [words[v]]
    keys = [v]
    if v == "confirmed":
        k = "status_adds" if run["status_adds"] else "status_same"
        parts.append(words[k])
        keys.append(k)
        if k == "status_same" and run["go_below"]:
            parts.append(words["status_same_go"])
            keys.append("status_same_go")
        if run["noise_hi_targets"]:
            parts.append(words["noise_caveat_hi"])
            keys.append("noise_caveat_hi")
    elif run["noise_lo_targets"]:
        parts.append(words["noise_caveat_lo"])
        keys.append("noise_caveat_lo")
    if v == "not" and run["go_below"]:
        parts.append(words["not_go_below"])
        keys.append("not_go_below")
    else:
        parts.append(words["go_always"])
        keys.append("go_always")
    if v == "confirmed":
        ek = "share_hi_more_often" if run["share_hi_more_often"] else keys[1]
        edit_key, edit = f"confirmed.{ek}", edits["confirmed"][ek]
    else:
        edit_key, edit = v, edits[v]
    return {
        "verdict": v,
        "word_keys": keys,
        "text": " ".join(fill(p, f) for p in parts),
        "edit_key": edit_key,
        "edit": fill(edit, f),
        "fields": f,
    }


def target_description(
    t: str, pt: Mapping[str, Any], high: Sequence[int], low: Sequence[int], words: Mapping[str, Any]
) -> str:
    """Описание одной цели — без слов вердикта (запрещённые корни проверяет ``check_forbidden``)."""
    g = pt["groups"]
    hi_s, lo_s = f"{high[0]}–{high[-1]}", f"{low[0]}–{low[-1]}"
    iv = pt["delta_ci"]
    text = (
        f"{TARGET_NAMES[t][:1].upper()}{TARGET_NAMES[t][1:]}"
        + (f" ({SEEN_LABEL})" if t == SEEN else "")
        + f": у МО типов {hi_s} соседи по региону ближе похожих по тратам МО других регионов"
        f" в {fmt_pct(g['hi']['share'])} случаев ({g['hi']['n']} МО), у типов {lo_s} — в"
        f" {fmt_pct(g['lo']['share'])} ({g['lo']['n']} МО); разность {fmt_pp(100.0 * pt['delta'])},"
        f" 95% интервал {fmt_range(100.0 * iv[0], 100.0 * iv[1], fmt=fmt_pp, decimals=1, sign=False)}"
    )
    caveat = words.get(f"target_caveat_{t}")
    return text + (f"; {caveat}" if caveat else "")


# --- Сборка ------------------------------------------------------------------------------------------


def run_order(cfg: Config) -> list[tuple[str, str]]:
    """Прогоны в порядке «основной, варианты (interpret.robustness.variants), повторы seed»: (имя, kind)."""
    rob = cfg["interpret"]["robustness"]
    seed = int(cfg["seed"])
    variants = [f"variant:{v}" for v in rob["variants"]]
    seeds = [f"seed:{int(s)}" for s in rob["seeds"] if int(s) != seed]
    if len(variants) != 3 or len(seeds) != 4:
        raise QCError("usefulness.by_type_test: ожидались 3 варианта и 4 повтора seed (interpret.robustness)")
    return [("main", "main"), *[(v, "variant") for v in variants], *[(s, "seed") for s in seeds]]


def run_labels(
    ids: np.ndarray,
    types: pd.DataFrame,
    r1: pd.DataFrame,
    order: Sequence[tuple[str, str]],
    allowed: set[int],
) -> dict[str, np.ndarray]:
    """Метки типов МО общего набора в каждом прогоне: основной — ``types.csv``, остальные — ``matched_type``
    из ``node_r1.csv``. Нет МО в прогоне, метка вне групп или ``main_type`` ≠ ``types.csv`` — ``QCError``."""
    main = types.set_index("territory_id")["type"].reindex(ids)
    if main.isna().any() or not set(main.astype(int)) <= allowed:
        raise QCError("usefulness.by_type_test: у части МО общего набора нет типа из групп {3,4}/{1,2}")
    out = {"main": main.to_numpy(dtype=np.int64)}
    for name, kind in order:
        if kind == "main":
            continue
        g = r1.loc[(r1["variant"] == name) & (r1["kind"] == kind)].set_index("territory_id")
        if g.index.duplicated().any():
            raise QCError(f"usefulness.by_type_test: node_r1.csv — повторы узлов в прогоне {name}")
        lab = g["matched_type"].reindex(ids)
        if lab.isna().any() or not set(lab.astype(int)) <= allowed:
            raise QCError(f"usefulness.by_type_test: node_r1.csv — в прогоне {name} нет метки у части МО")
        mt = g["main_type"].reindex(ids).to_numpy(dtype=np.int64)
        if not np.array_equal(mt, out["main"]):
            raise QCError(f"usefulness.by_type_test: node_r1.csv main_type ≠ types.csv в прогоне {name}")
        out[name] = lab.to_numpy(dtype=np.int64)
    return out


def qc_retail(run_main: Mapping[str, Any], by_type_counts: Sequence[Mapping[str, Any]], n_base: int,
              high: Sequence[int], low: Sequence[int], tol: float) -> dict:  # fmt: skip
    """Δ розницы тем же кодом на всех МО общего набора = (w3 + w4)/(n3 + n4) − (w1 + w2)/(n1 + n2) по
    счётчикам ``rule.by_type`` этапа usefulness; иначе ``QCError``."""
    c = {int(r["type"]): r for r in by_type_counts}
    if not set(high) | set(low) <= set(c):
        raise QCError("usefulness.by_type_test: в rule.by_type нет части типов групп")
    exact = sum(c[k]["works"] for k in high) / sum(c[k]["n"] for k in high) - sum(
        c[k]["works"] for k in low
    ) / sum(c[k]["n"] for k in low)
    pt = run_main["per_target"][SEEN]
    got = float(pt["delta"])
    if pt["n"] != n_base or not abs(got - exact) <= tol:
        raise QCError(
            f"usefulness.by_type_test: QC розницы — Δ {got!r} на {pt['n']} МО, по счётчикам {exact!r} "
            f"на {n_base} МО (допуск {tol})"
        )
    return {"delta": got, "exact_from_counts": exact, "abs_diff": abs(got - exact), "n": pt["n"], "tol": tol}


def _read(cfg: Config) -> dict[str, Any]:
    out = cfg.dir("outputs") / "interpret"
    processed = cfg.dir("processed")
    paths = {
        "t7_errors": out / "t7_errors.csv",
        "node_comparable": out / "node_comparable.csv",
        "types": out / "types.csv",
        "node_r1": out / "node_r1.csv",
        "context": processed / "context_annual.parquet",
        "nodes": processed / "features_nodes.parquet",
    }
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise MissingInputError(f"usefulness.by_type_test: нет входов {missing} — сначала этап interpret")
    cols = ["territory_id", "year", *dict.fromkeys(COLUMNS.values()), NDFL_FILTER]
    return {
        "errors": pd.read_csv(paths["t7_errors"]),
        "comparable": pd.read_csv(paths["node_comparable"]),
        "types": pd.read_csv(paths["types"]),
        "r1": pd.read_csv(paths["node_r1"]),
        "context": pd.read_parquet(paths["context"], columns=cols),
        "nodes": pd.read_parquet(paths["nodes"], columns=["territory_id", "region_group", "mo_type"]),
    }


def compute(cfg: Config, inp: Mapping[str, Any], rule_by_type: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Весь расчёт блока на прочитанных входах ``inp``; ``rule_by_type`` — счётчики ``rule.by_type`` этапа
    usefulness (для QC розницы). Возвращает ``{"facts", "runs", "by_mo"}``."""
    block = cfg["usefulness"]["by_type_test"]
    check_spec(block)
    seed = int(cfg["seed"])
    uni, st = block["universe"], block["statistic"]
    high, low = [int(k) for k in uni["groups"]["high"]], [int(k) for k in uni["groups"]["low"]]
    order = run_order(cfg)

    e = inp["errors"]
    ids = np.sort(e.loc[e["common"].astype(bool), "territory_id"].to_numpy(dtype=np.int64))
    nodes = inp["nodes"].set_index("territory_id")
    group_of = nodes["region_group"]
    groups = group_of.reindex(ids).to_numpy()
    if pd.isna(groups).any():
        raise QCError("usefulness.by_type_test: у части МО общего набора нет группы региона")
    groups = groups.astype(np.int64)
    mo_type = nodes["mo_type"].astype(str).reindex(ids).to_numpy()
    strata_def = uni["status"]["strata"]
    strata = [np.isin(mo_type, list(v)) for v in strata_def.values()]
    covered = np.sum(strata, axis=0)
    if (covered != 1).any():
        raise QCError("usefulness.by_type_test: у части МО статус вне страт (или в двух стратах)")
    is_go = mo_type == "go"  # statistic.rival.name = mo_type == "go"

    min_set = int(uni["min_set"])
    tie_tol = float(cfg["usefulness"]["rule_share"]["tie_tol"])  # одно сравнение ошибок на весь этап
    r_cfg = uni["R"]
    data: dict[str, dict[str, np.ndarray]] = {}
    tables = []
    for t in (*TARGETS, SEEN):
        y = target_change(inp["context"], COLUMNS[t], NDFL_FILTER if t == "ndfl" else None)
        tab = target_table(y, inp["comparable"], ids, min_set)
        m = tab["in_target"].to_numpy()
        err_r = np.full(len(ids), np.nan)
        if m.any():
            err_r[m] = region_random_errors(
                y, ids[m], tab["size_B"].to_numpy()[m], group_of, int(r_cfg["draws"]), seed
            )
        tab["err_R"] = err_r
        eb, ed = tab["err_B"].to_numpy(), tab["err_D"].to_numpy()
        works = m & compare_errors(np.nan_to_num(eb, nan=np.inf), np.nan_to_num(ed, nan=np.inf), tie_tol)[0]
        tab["works"] = works
        tab.insert(1, "target", t)
        tables.append(tab)
        data[t] = {
            "mask": m,
            "works": works,
            "err_B": eb,
            "err_D": ed,
            "err_R": err_r,
            "y": tab["y"].to_numpy(),
            "size_B": tab["size_B"].to_numpy(),
            "size_D": tab["size_D"].to_numpy(),
        }

    allowed = set(high) | set(low)
    labels = run_labels(ids, inp["types"], inp["r1"], order, allowed)
    gidx, counts = bootstrap_counts(groups, int(st["bootstrap"]["n"]), seed)
    perm = permutation_index(groups, int(st["permutation"]["n"]), seed)
    design = {
        "high": high,
        "low": low,
        "strata": strata,
        "is_go": is_go,
        "gidx": gidx,
        "counts": counts,
        "perm": perm,
        "level": float(st["ci_level"]),
        "alpha": float(st["permutation"]["alpha"]),
        "noise_lo": float(st["noise_flag_lo"]),
        "noise_hi": float(st["noise_flag_hi"]),
        "types": sorted(allowed),
        "tie_tol": tie_tol,
    }
    runs = {name: evaluate_run(data, labels[name], design) for name, _ in order}

    retail_qc = qc_retail(
        runs["main"], rule_by_type, len(ids), high, low, float(block["qc"]["recompute_retail"])
    )

    words, edits = block["words"], block["edits"]
    forbidden = list(words["forbidden_per_target"])
    for r in runs.values():
        r["descriptions"] = {}
        for t, pt in r["per_target"].items():
            text = target_description(t, pt, high, low, words)
            check_forbidden(text, forbidden)
            r["descriptions"][t] = text
        r["texts"] = outcome_texts(r, words, edits)

    names = [n for n, _ in order]
    levels = list(block["rule"]["levels"])
    k = lowest_verdict([runs[n]["verdict"] for n in names], levels)
    low_name = names[k]
    unstable = levels.index(runs["main"]["verdict"]) > levels.index(runs[low_name]["verdict"])
    main_edits = [{"key": runs[low_name]["texts"]["edit_key"], "text": runs[low_name]["texts"]["edit"]}]
    if unstable:
        main_edits.append(
            {"key": "unstable", "text": fill(edits["unstable"], runs[low_name]["texts"]["fields"])}
        )

    by_mo = pd.concat(tables, ignore_index=True)
    main_type = pd.Series(labels["main"], index=ids)
    universe = {}
    for t in (*TARGETS, SEEN):
        reason = by_mo.loc[by_mo["target"] == t].set_index("territory_id")["drop_reason"]
        universe[t] = {
            "n_base": len(ids),
            "n": int(data[t]["mask"].sum()),
            **{r: int((reason == r).sum()) for r in ("no_target", "min_set")},
            **{
                f"{r}_by_type": {
                    int(k): int(((reason == r) & (main_type == k)).sum()) for k in sorted(allowed)
                }
                for r in ("no_target", "min_set")
            },
            "n_by_type": {int(k): int(((reason == "") & (main_type == k)).sum()) for k in sorted(allowed)},
        }

    facts = {
        "status": "done",
        "note": "проверка usefulness.by_type_test: правила записаны до расчёта (f744563), код — вслепую",
        "question": str(block["question"]),
        "targets": list(TARGETS),
        "seen_target": SEEN,
        "groups": {"high": high, "low": low},
        "n_base": len(ids),
        "n_groups": int(len(np.unique(groups))),
        "universe": universe,
        "design": {
            "bootstrap": {"n": int(counts.shape[0]), "by": "region_group", "seed": seed},
            "permutation": {"n": int(perm.shape[0]), "within": "region_group", "seed": seed},
            "R": {"draws": int(r_cfg["draws"]), "seed": seed},
            "ci_level": design["level"],
            "alpha": design["alpha"],
            "run_order": names,
        },
        "verdicts": {n: runs[n]["verdict"] for n in names},
        "main_text": {
            "run": low_name,
            "verdict": runs[low_name]["verdict"],
            "text": runs[low_name]["texts"]["text"],
            "word_keys": runs[low_name]["texts"]["word_keys"],
            "edits": main_edits,
            "unstable": bool(unstable),
        },
        "check_section": (
            {
                "run": "main",
                "verdict": runs["main"]["verdict"],
                "text": runs["main"]["texts"]["text"],
                "unstable": str(words["unstable"]),
            }
            if unstable
            else None
        ),
        "limitations": [fill(t, runs[low_name]["texts"]["fields"]) for t in block["limitations"]],
        "negative_control": {
            n: {"null_mean": runs[n]["null_mean"], "null_sd": runs[n]["null_sd"]} for n in names
        },
        "qc": {"retail": retail_qc, "spec_frozen": True, "forbidden_per_target": forbidden},
        "runs": runs,
    }
    rows = []
    for n in names:
        r = runs[n]
        for t, pt in r["per_target"].items():
            g = pt["groups"]
            rows.append(
                {
                    "run": n,
                    "target": t,
                    "in_decision": t in TARGETS,
                    "n": pt["n"],
                    "n_hi": g["hi"]["n"],
                    "n_lo": g["lo"]["n"],
                    "share_hi": g["hi"]["share"],
                    "share_lo": g["lo"]["share"],
                    "delta": pt["delta"],
                    "delta_lo": pt["delta_ci"][0],
                    "delta_hi": pt["delta_ci"][1],
                    "delta_status": pt["delta_status"],
                    "delta_status_lo": pt["delta_status_ci"][0],
                    "delta_status_hi": pt["delta_status_ci"][1],
                    "delta_go": pt["delta_go"],
                    "delta_go_lo": pt["delta_go_ci"][0],
                    "delta_go_hi": pt["delta_go_ci"][1],
                    "median_d_diff": pt["median_d_diff"],
                    "median_err_B_hi": g["hi"]["median_err_B"],
                    "median_err_B_lo": g["lo"]["median_err_B"],
                    "median_err_D_hi": g["hi"]["median_err_D"],
                    "median_err_D_lo": g["lo"]["median_err_D"],
                    "verdict_of_run": r["verdict"],
                }
            )
    return {"facts": facts, "runs": pd.DataFrame(rows), "by_mo": by_mo}


def run(cfg: Config, rule_by_type: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Расчёт и запись выходов блока; старые выходы блока удаляются до расчёта (не останутся при сбое)."""
    out = cfg.dir("outputs") / "usefulness"
    out.mkdir(parents=True, exist_ok=True)
    files = {
        "facts": out / "by_type.json",
        "runs": out / "by_type_runs.csv",
        "by_mo": out / "by_type_by_mo.csv",
    }
    for p in files.values():
        p.unlink(missing_ok=True)
    res = compute(cfg, _read(cfg), rule_by_type)
    res["runs"].to_csv(files["runs"], index=False)
    res["by_mo"].to_csv(files["by_mo"], index=False)
    _write_json(files["facts"], res["facts"])
    f = res["facts"]
    log.info(
        "usefulness.by_type_test: вердикты по прогонам %s; в главный текст — %s (%s)",
        f["verdicts"], f["main_text"]["verdict"], f["main_text"]["run"],
    )  # fmt: skip
    return f


def _jsonable(x: Any) -> Any:
    if isinstance(x, Mapping):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return _jsonable(x.item()) if x.ndim == 0 else [_jsonable(v) for v in x.tolist()]
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating | float):
        v = float(x)
        return v if np.isfinite(v) else None
    return x


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(_jsonable(obj), ensure_ascii=False, indent=1), encoding="utf-8")
