"""Сюжеты-кандидаты этапа 1 и матрица решения (spec_final, Б.5).

Шесть сюжетов: С1–С3 из ``PLAN.md``, С4–С6 подсказаны данными. У каждого:

- **экспертные баллы** 1–5 (сеть и динамика, практическая ценность, объяснимость, риски данных) с обоснованием
  в одну строку — их задаёт автор, участник вправе изменить;
- **балл силы сигнала и балл покрытия** — считаются из фактов разведки по порогам;
- **критерии отказа** — сюжет может стать главным, только если выполнены все; иначе он «слой» (хотя бы один
  критерий силы сигнала выполнен) или «контекст» (не выполнен ни один), независимо от балла.

Итог = Σ вес × балл, веса — ``eda.matrix_weights`` (выведены из критериев жюри, в сумме 1). Матрица —
подсказка: выбор сюжета делает участник.

Пороги критериев — ``eda.rejection`` (и ``eda.step_test_pp``); пороги баллов — ``eda.stories`` поверх
``DEFAULTS``. Функции чистые: вход — словарь фактов ``{«e3.reliable_share»: Fact}`` и конфиг.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd

from munnet import style
from munnet.eda.base import Fact, make_fact

# Критерии оценки сюжета: ключ ``eda.matrix_weights`` -> подпись в матрице.
CRITERIA: dict[str, str] = {
    "signal": "Сила сигнала",
    "network_dynamics": "Сеть и динамика",
    "practical": "Практическая ценность",
    "coverage": "Покрытие",
    "explainability": "Объяснимость",
    "data_risk": "Риски данных (5 — рисков мало)",
}
MANUAL_CRITERIA: tuple[str, ...] = ("network_dynamics", "practical", "explainability", "data_risk")
# Короткие подписи критериев для обоснований в одну строку.
CRITERIA_SHORT: dict[str, str] = {
    "signal": "сигнал",
    "network_dynamics": "сеть и динамика",
    "practical": "практическая ценность",
    "coverage": "покрытие",
    "explainability": "объяснимость",
    "data_risk": "риски данных",
}
SCORE_MIN, SCORE_MAX = 1, 5
# Допуск суммы весов матрицы: веса из YAML — десятичные дроби.
WEIGHTS_TOL = 1e-6

STATUS_MAIN = "может быть главным"
STATUS_LAYER = "слой"
STATUS_CONTEXT = "контекст"
STATUS_NA = "нет данных"
PASSED_TEXT = "проходит"
NO_STORY = "нет"

ROLE_SIGNAL = "signal"  # критерий силы сигнала: выполнен — у сюжета есть на что опереться хотя бы как у слоя
ROLE_DATA = "data"  # критерий достаточности или чистоты данных: сам по себе сюжет слоем не делает

# Факт с числом МО панели — знаменатель покрытия (пишет сводка).
N_MO_FACT = "syn.n_mo"

# Параметры по умолчанию; ``eda.stories`` конфига переопределяет их ключ за ключом, отсутствующие пороги
# ``eda.rejection`` берутся из ``rejection`` (spec_final, Б.5).
DEFAULTS: dict[str, Any] = {
    # Балл силы сигнала: значение < bins[0] → 1, < bins[1] → 2, < bins[2] → 3, < bins[3] → 4, иначе 5.
    "signal_bins": {
        "s1": [0.1, 0.2, 0.3, 0.4],  # доля МО с устойчивым своим ритмом
        "s2": [0.1, 0.2, 0.3, 0.4],  # |частный ρ| уровня с доступностью рынков
        "s3": [0.3, 0.45, 0.6, 0.75],  # ρ роста по полугодиям
        "s4": [0.5, 0.6, 0.7, 0.85],  # устойчивость очищенной PC1 корзины
        "s5": [0.3, 0.45, 0.6, 0.75],  # ρ прироста доли маркетплейсов по полугодиям (шкала С3)
        "s6": [1.1, 1.25, 1.5, 2.0],  # траты к доходу 5-НДФЛ: пригороды к остальным
    },
    # Балл покрытия по доле МО панели, пригодных для сюжета: < bands[0] → 1 … ≥ bands[3] → 5.
    "coverage_bands": [0.6, 0.7, 0.8, 0.9],
    # Пороги критериев отказа, которых нет в части Г конфига (``eda.rejection`` их переопределяет).
    "rejection": {
        "s1_null_ratio_min": 3.0,  # С1: доля надёжных МО ≥ 3 × доля на перестановочном нуле
        "s6_suburb_p_max": 0.01,  # С6: отличие пригородов надёжно при p Манна — Уитни меньше
    },
}

_OPS: dict[str, tuple[Callable[[float, float], bool], str]] = {
    ">=": (lambda v, t: v >= t, "≥"),
    "<=": (lambda v, t: v <= t, "≤"),
    "<": (lambda v, t: v < t, "<"),
    "abs>=": (lambda v, t: abs(v) >= t, "≥"),
}


# --- Параметры -----------------------------------------------------------------------------------


def params(cfg: Any | None = None) -> dict[str, Any]:
    """Параметры сюжетов: ``DEFAULTS``, поверх — ``eda.stories`` конфига; неизвестный ключ — ``ValueError``.

    В ``rejection`` результата — пороги критериев отказа: ``DEFAULTS["rejection"]``, поверх —
    ``eda.rejection`` конфига (секция части Г).
    """
    out = copy.deepcopy(DEFAULTS)
    if cfg is None:
        return out
    eda = cfg["eda"]
    user = eda.get("stories") or {}
    unknown = sorted(set(user) - set(DEFAULTS))
    if unknown:
        raise ValueError(f"eda.stories: неизвестные ключи {unknown}; допустимы {sorted(DEFAULTS)}")
    for key, value in user.items():
        if isinstance(out[key], dict):
            out[key].update(value)
        else:
            out[key] = value
    out["rejection"].update(eda.get("rejection") or {})
    return out


def check_weights(weights: Mapping[str, float]) -> dict[str, float]:
    """Веса матрицы: ровно критерии ``CRITERIA``, неотрицательные, в сумме 1; иначе ``ValueError``."""
    w = {k: float(v) for k, v in weights.items()}
    if set(w) != set(CRITERIA):
        raise ValueError(f"eda.matrix_weights: нужны ключи {sorted(CRITERIA)}, получены {sorted(w)}")
    if any(v < 0 for v in w.values()):
        raise ValueError(f"eda.matrix_weights: отрицательный вес {w}")
    total = sum(w.values())
    if abs(total - 1.0) > WEIGHTS_TOL:
        raise ValueError(f"eda.matrix_weights: сумма весов {total}, а должна быть 1")
    return w


# --- Баллы из фактов -----------------------------------------------------------------------------


def _value(facts: Mapping[str, Fact], key: str) -> float | None:
    """Числовое значение факта или None (нет факта, пропуск, строка)."""
    fact = facts.get(key)
    if fact is None or fact.value is None or isinstance(fact.value, str):
        return None
    value = float(fact.value)
    return value if math.isfinite(value) else None


def _fmt_like(facts: Mapping[str, Fact], key: str, value: float) -> str:
    """Число в формате факта ``key`` (проценты — как проценты, ρ — как ρ); без факта — два знака."""
    fact = facts.get(key)
    kind = fact.kind if fact is not None and fact.kind != "str" else "num2"
    return make_fact(key, value, kind).text


def bin_score(value: float | None, bins: Sequence[float]) -> int | None:
    """Балл 1–5 по возрастающим порогам: ``value < bins[0]`` → 1, …, ``value ≥ bins[-1]`` → 5; None → None."""
    edges = [float(b) for b in bins]
    if len(edges) != SCORE_MAX - SCORE_MIN or edges != sorted(edges):
        raise ValueError(f"пороги балла {bins}: нужно {SCORE_MAX - SCORE_MIN} возрастающих чисел")
    if value is None:
        return None
    return SCORE_MIN + sum(value >= e for e in edges)


@dataclass(frozen=True)
class BinScore:
    """Балл силы сигнала 1–5 из одного факта по порогам ``bins`` (по модулю при ``use_abs``)."""

    fact_key: str
    bins: tuple[float, ...]
    label: str
    use_abs: bool = False

    def __call__(self, facts: Mapping[str, Fact]) -> tuple[int | None, str]:
        value = _value(facts, self.fact_key)
        if value is None:
            return None, f"нет факта {self.fact_key}"
        shown = abs(value) if self.use_abs else value
        score = bin_score(shown, self.bins)
        prefix = "|" if self.use_abs else ""
        text = f"{self.label} {prefix}{_fmt_like(facts, self.fact_key, shown)}{prefix}"
        return score, style_nbsp(text)


@dataclass(frozen=True)
class CoverageScore:
    """Балл покрытия 1–5: доля МО панели (факт ``denominator``), пригодных для сюжета (факт ``fact_key``)."""

    fact_key: str
    bands: tuple[float, ...]
    label: str
    denominator: str = N_MO_FACT

    def __call__(self, facts: Mapping[str, Fact]) -> tuple[int | None, str]:
        n, total = _value(facts, self.fact_key), _value(facts, self.denominator)
        if n is None or total is None or total <= 0:
            missing = self.fact_key if n is None else self.denominator
            return None, f"нет факта {missing}"
        share = n / total
        text = f"{self.label} — {style.fmt_num(n)} из {style.fmt_num(total)} ({style.fmt_pct(share, 0)})"
        return bin_score(share, self.bands), style_nbsp(text)


# --- Критерии отказа -----------------------------------------------------------------------------


@dataclass(frozen=True)
class Rejection:
    """Критерий отказа: ``факт op порог`` (при ``times_fact`` — порог × значение этого факта).

    ``threshold_key`` — путь порога в секции ``eda`` конфига через точку: «rejection.s2_partial_rho_min»,
    «step_test_pp». ``role`` — ``signal`` (сила сигнала) или ``data`` (достаточность и чистота данных).
    """

    name: str
    fact_key: str
    op: str  # ">=", "<=", "<", "abs>="
    threshold_key: str
    text: str
    times_fact: str = ""
    role: str = ROLE_SIGNAL


@dataclass(frozen=True)
class CriterionResult:
    """Итог проверки критерия: выполнен (True/False) или не проверяется (None — нет факта)."""

    name: str
    ok: bool | None
    role: str
    text: str  # «доля МО с устойчивым своим ритмом 15,6%, нужно ≥ 25,0%»


def resolve_threshold(path: str, prm: Mapping[str, Any], eda: Mapping[str, Any] | None) -> float:
    """Порог по пути «rejection.<ключ>» (из ``params``) или «<ключ>» секции ``eda``; нет — ``KeyError``."""
    head, _, tail = path.partition(".")
    if head == "rejection" and tail:
        if tail not in prm["rejection"]:
            raise KeyError(f"нет порога eda.rejection.{tail}")
        return float(prm["rejection"][tail])
    node: Any = eda or {}
    for part in path.split("."):
        if not isinstance(node, Mapping) or part not in node:
            raise KeyError(f"нет порога eda.{path}")
        node = node[part]
    return float(node)


def check_rejection(
    rej: Rejection, facts: Mapping[str, Fact], prm: Mapping[str, Any], eda: Mapping[str, Any] | None
) -> CriterionResult:
    """Проверяет один критерий отказа по фактам; пропуск факта — ``ok=None``."""
    if rej.op not in _OPS:
        raise ValueError(f"критерий {rej.name}: неизвестная операция {rej.op!r}; допустимы {sorted(_OPS)}")
    compare, sym = _OPS[rej.op]
    threshold = resolve_threshold(rej.threshold_key, prm, eda)
    value = _value(facts, rej.fact_key)
    scale = _value(facts, rej.times_fact) if rej.times_fact else 1.0
    if value is None or scale is None:
        missing = rej.fact_key if value is None else rej.times_fact
        return CriterionResult(rej.name, None, rej.role, style_nbsp(f"{rej.text}: нет факта {missing}"))
    limit = threshold * scale
    shown = abs(value) if rej.op.startswith("abs") else value
    bar = "|" if rej.op.startswith("abs") else ""
    value_text = f"{bar}{_fmt_like(facts, rej.fact_key, shown)}{bar}"
    if rej.times_fact:
        limit_text = (
            f"{style.fmt_num(threshold, 1 if threshold % 1 else 0)} × "
            f"{_fmt_like(facts, rej.times_fact, scale)} = {_fmt_like(facts, rej.fact_key, limit)}"
        )
    else:
        limit_text = _fmt_like(facts, rej.fact_key, limit)
    ok = bool(compare(value, limit))
    text = f"{rej.text} {value_text}, нужно {sym} {limit_text}"
    return CriterionResult(rej.name, ok, rej.role, style_nbsp(text))


def story_status(results: Sequence[CriterionResult]) -> str:
    """Роль сюжета по критериям отказа (независимо от балла).

    Все выполнены — «может быть главным»; хоть один не проверяется (нет факта) — «нет данных»; иначе «слой»,
    если выполнен хотя бы один критерий силы сигнала, и «контекст», если ни одного.
    """
    if any(r.ok is None for r in results):
        return STATUS_NA
    if all(r.ok for r in results):
        return STATUS_MAIN
    if any(r.ok and r.role == ROLE_SIGNAL for r in results):
        return STATUS_LAYER
    return STATUS_CONTEXT


def rejection_text(results: Sequence[CriterionResult]) -> str:
    """«проходит» или перечень невыполненных и непроверенных критериев через «; »."""
    failed = [r.text for r in results if not r.ok]
    return PASSED_TEXT if not failed else "; ".join(failed)


# --- Сюжеты --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Story:
    """Сюжет-кандидат: код для ключей фактов (``key``), номер «С1», название, суть, баллы, критерии отказа."""

    sid: str
    title: str
    gist: str
    manual: dict[str, tuple[int, str]]  # network_dynamics, practical, explainability, data_risk
    signal: Callable[[Mapping[str, Fact]], tuple[int | None, str]]
    coverage: Callable[[Mapping[str, Fact]], tuple[int | None, str]]
    rejections: tuple[Rejection, ...]
    key: str = ""  # латинский код для ключей фактов: «s1»


def build_stories(prm: Mapping[str, Any] | None = None) -> tuple[Story, ...]:
    """Шесть сюжетов Б.5 с порогами баллов из ``prm`` (``params(cfg)``; None — ``DEFAULTS``)."""
    prm = params(None) if prm is None else prm
    bins = {k: tuple(float(x) for x in v) for k, v in prm["signal_bins"].items()}
    bands = tuple(float(x) for x in prm["coverage_bands"])

    def cover(fact_key: str, label: str) -> CoverageScore:
        return CoverageScore(fact_key, bands, label)

    full = "МО с полным рядом за 24 месяца"
    return (
        Story(
            sid="С1",
            key="s1",
            title="Годовой ритм",
            gist="типы МО по форме года: северное лето, новогодний пик, ровный ритм",
            manual={
                "network_dynamics": (4, "корреляции и DTW своего ритма, скользящие годовые окна"),
                "practical": (3, "завоз, сезонная нагрузка, северные надбавки"),
                "explainability": (4, "«северяне тратят летом»"),
                "data_risk": (2, "два цикла; курорты юга выпали; неясно, где совершена трата"),
            },
            signal=BinScore("e3.reliable_share", bins["s1"], "доля МО с устойчивым своим ритмом"),
            coverage=cover("e1.n_full", full),
            rejections=(
                Rejection(
                    "s1_share",
                    "e3.reliable_share",
                    ">=",
                    "rejection.s1_reliable_share_min",
                    "доля МО с устойчивым своим ритмом",
                ),
                Rejection(
                    "s1_null",
                    "e3.reliable_share",
                    ">=",
                    "rejection.s1_null_ratio_min",
                    "та же доля против перестановочного нуля",
                    times_fact="e3.null_reliable_share",
                ),
            ),
        ),
        Story(
            sid="С2",
            key="s2",
            title="Ядра и периферия",
            gist="центры и периферия по уровню трат, доступности рынков и дорогам",
            manual={
                "network_dynamics": (3, "дорожные расстояния и гравитация естественны, но статичны"),
                "practical": (4, "размещение услуг и транспорт"),
                "explainability": (4, "знакомая рамка «центр — периферия»"),
                "data_risk": (3, "уровень трат сводится к зарплате и региону"),
            },
            signal=BinScore(
                "e2.partial_rho_access_within",
                bins["s2"],
                "частный ρ уровня с доступностью рынков сверх зарплаты внутри регионов",
                use_abs=True,
            ),
            coverage=cover("e2.n_access_wage", "МО с доступностью рынков и зарплатой"),
            rejections=(
                Rejection(
                    "s2_partial",
                    "e2.partial_rho_access_within",
                    "abs>=",
                    "rejection.s2_partial_rho_min",
                    "частный ρ уровня с доступностью рынков сверх зарплаты внутри регионов",
                ),
            ),
        ),
        Story(
            sid="С3",
            key="s3",
            title="Устойчивость потребления",
            gist="кто наращивает траты и кто проседает",
            manual={
                "network_dynamics": (2, "корреляции роста на двух годах"),
                "practical": (3, "мониторинг потребления"),
                "explainability": (3, "«кто растёт» понятно, но рост номинальный"),
                "data_risk": (1, "номинал, два года, шум полугодий"),
            },
            signal=BinScore("e4.growth_half_consistency", bins["s3"], "согласованность роста по полугодиям"),
            coverage=cover("e1.n_full", full),
            rejections=(
                Rejection(
                    "s3_half",
                    "e4.growth_half_consistency",
                    ">=",
                    "rejection.s3_half_consistency_min",
                    "согласованность роста по полугодиям (ρ)",
                ),
            ),
        ),
        Story(
            sid="С4",
            key="s4",
            title="Состав корзины",
            gist="типы по составу трат, очищенному от уровня и региона",
            manual={
                "network_dynamics": (4, "косинус векторов расходов, помесячные сети"),
                "practical": (3, "торговля и сервисы"),
                "explainability": (3, "очищенные оси труднее назвать"),
                "data_risk": (3, "доля общепита зависит от сетевых кафе"),
            },
            signal=BinScore(
                "e4.residual_pc1_stability", bins["s4"], "устойчивость очищенной PC1 корзины 2023 → 2024"
            ),
            coverage=cover("e4.n_basket", "МО с 12 месяцами обоих лет"),
            rejections=(
                Rejection(
                    "s4_residual",
                    "e4.residual_share",
                    ">=",
                    "rejection.s4_residual_share_min",
                    "доля дисперсии CLR после очистки от уровня и региона",
                ),
                Rejection(
                    "s4_stability",
                    "e4.residual_pc1_stability",
                    ">=",
                    "rejection.s4_residual_stability_min",
                    "устойчивость очищенной PC1 (ρ 2023 и 2024)",
                ),
            ),
        ),
        Story(
            sid="С5",
            key="s5",
            title="Сдвиг к маркетплейсам",
            gist="где и как быстро траты уходят в онлайн",
            manual={
                "network_dynamics": (4, "лаговые корреляции доли — «кто опережает», помесячная динамика"),
                "practical": (4, "ритейл, логистика, пункты выдачи, местный бизнес"),
                "explainability": (4, "«онлайн растёт быстрее там, где тратят меньше»"),
                "data_risk": (3, "только относительный сигнал; возможна переклассификация продавцов"),
            },
            signal=BinScore(
                "e4.mp_split_half", bins["s5"], "согласованность прироста доли маркетплейсов по полугодиям"
            ),
            coverage=cover("e1.n_full", full),
            rejections=(
                Rejection(
                    "s5_half",
                    "e4.mp_split_half",
                    ">=",
                    "rejection.s5_split_half_min",
                    "согласованность прироста доли по полугодиям (ρ)",
                ),
                Rejection(
                    "s5_within",
                    "e4.rho_mp_pp_level_within",
                    "abs>=",
                    "rejection.s5_rho_within_min",
                    "ρ прироста доли с уровнем трат внутри регионов",
                ),
                Rejection(
                    "s5_step",
                    "e4.mp_step_max_pp",
                    "<=",
                    "step_test_pp",
                    "наибольший скачок годового прироста доли между соседними месяцами, п. п.",
                    role=ROLE_DATA,
                ),
            ),
        ),
        Story(
            sid="С6",
            key="s6",
            title="Где зарабатывают и где тратят",
            gist="пригороды, центры занятости и вахта по балансу доходов и трат",
            manual={
                "network_dynamics": (3, "гравитация «работа — жительство» по дорогам"),
                "practical": (4, "агломерации, транспорт, налоги по месту жительства"),
                "explainability": (4, "«живут в пригороде, работают в центре»"),
                "data_risk": (
                    2,
                    "5-НДФЛ по работодателю, аномальные отношения получателей к жителям, без МСП",
                ),
            },
            signal=BinScore("e5.suburb_ratio", bins["s6"], "траты к доходу 5-НДФЛ: пригороды к остальным"),
            coverage=cover("e5.n_ndfl_usable", "МО с пригодным доходом 5-НДФЛ"),
            rejections=(
                Rejection(
                    "s6_usable",
                    "e5.n_ndfl_usable",
                    ">=",
                    "rejection.s6_usable_min",
                    "МО с пригодным доходом 5-НДФЛ",
                    role=ROLE_DATA,
                ),
                Rejection(
                    "s6_ratio",
                    "e5.suburb_ratio",
                    ">=",
                    "rejection.s6_suburb_ratio_min",
                    "траты к доходу 5-НДФЛ: пригороды к остальным",
                ),
                Rejection(
                    "s6_p",
                    "e5.suburb_mw_p",
                    "<",
                    "rejection.s6_suburb_p_max",
                    "p теста Манна — Уитни для пригородов",
                ),
            ),
        ),
    )


STORIES: tuple[Story, ...] = build_stories()


def validate_stories(stories: Sequence[Story]) -> None:
    """Проверяет сюжеты: уникальные номера и коды, экспертные баллы 1–5 с обоснованием по всем четырём
    критериям, известные операции критериев; иначе ``ValueError``."""
    problems = []
    for field_name in ("sid", "key"):
        values = [getattr(s, field_name) for s in stories]
        if len(set(values)) != len(values) or not all(values):
            problems.append(f"{field_name} сюжетов пусты или повторяются: {values}")
    for s in stories:
        if set(s.manual) != set(MANUAL_CRITERIA):
            problems.append(
                f"{s.sid}: экспертные баллы по {sorted(s.manual)}, нужны {sorted(MANUAL_CRITERIA)}"
            )
        for crit, (score, why) in s.manual.items():
            if not SCORE_MIN <= int(score) <= SCORE_MAX or not why.strip():
                problems.append(f"{s.sid}.{crit}: балл {score} вне 1–5 или нет обоснования")
        for r in s.rejections:
            if r.op not in _OPS or r.role not in (ROLE_SIGNAL, ROLE_DATA):
                problems.append(f"{s.sid}.{r.name}: операция {r.op!r} или роль {r.role!r} неизвестны")
    if problems:
        raise ValueError("; ".join(problems))


# --- Матрица решения -----------------------------------------------------------------------------


def weighted_score(scores: Mapping[str, int | None], weights: Mapping[str, float]) -> float:
    """Σ вес × балл по критериям ``weights``; хоть один балл неизвестен — NaN."""
    if any(scores.get(k) is None for k in weights):
        return float("nan")
    return float(sum(float(weights[k]) * float(scores[k]) for k in weights))


MATRIX_COLUMNS: tuple[str, ...] = (
    "sid",
    "key",
    "title",
    "gist",
    *CRITERIA,
    "score",
    "status",
    "criteria",
    "signal_note",
    "coverage_note",
    "manual_note",
)


def evaluate(
    stories: Sequence[Story],
    facts: Mapping[str, Fact],
    weights: Mapping[str, float],
    prm: Mapping[str, Any],
    eda: Mapping[str, Any] | None,
) -> pd.DataFrame:
    """Матрица решения: строка на сюжет — баллы по шести критериям, итог, роль и критерии отказа.

    Колонки ``MATRIX_COLUMNS``; сюжеты — в порядке ``stories`` (С1…С6), итог округлять не нужно.
    """
    validate_stories(stories)
    w = check_weights(weights)
    rows = []
    for s in stories:
        sig, sig_note = s.signal(facts)
        cov, cov_note = s.coverage(facts)
        scores: dict[str, int | None] = {"signal": sig, "coverage": cov}
        scores.update({k: int(v[0]) for k, v in s.manual.items()})
        results = [check_rejection(r, facts, prm, eda) for r in s.rejections]
        manual_note = "; ".join(f"{CRITERIA_SHORT[k]} {s.manual[k][0]}: {s.manual[k][1]}" for k in s.manual)
        rows.append(
            {
                "sid": s.sid,
                "key": s.key,
                "title": s.title,
                "gist": s.gist,
                **{k: scores[k] for k in CRITERIA},
                "score": weighted_score(scores, w),
                "status": story_status(results),
                "criteria": rejection_text(results),
                "signal_note": sig_note,
                "coverage_note": cov_note,
                "manual_note": style_nbsp(manual_note),
            }
        )
    out = pd.DataFrame(rows, columns=list(MATRIX_COLUMNS))
    for k in CRITERIA:
        out[k] = out[k].astype("Int64")
    return out


def ranking(matrix: pd.DataFrame) -> pd.DataFrame:
    """Сюжеты по убыванию итога; при равенстве — по силе сигнала, затем по порядку С1…С6.

    Сюжеты без итога (нет данных) — в конце.
    """
    order = matrix.assign(
        _score=matrix["score"].fillna(-math.inf),
        _signal=matrix["signal"].astype("float64").fillna(-math.inf),
        _pos=range(len(matrix)),
    )
    order = order.sort_values(["_score", "_signal", "_pos"], ascending=[False, False, True])
    return order.drop(columns=["_score", "_signal", "_pos"])


def pick_top(matrix: pd.DataFrame) -> tuple[pd.Series | None, pd.Series | None]:
    """Главный кандидат и следующий по баллу.

    Главный — лучший по итогу среди сюжетов со статусом «может быть главным» (сюжет с невыполненным
    критерием главным не становится при любом балле); следующий — лучший из остальных с посчитанным
    итогом (главный или слой — это связка «главный сюжет + слой»). Нет кандидата — None.
    """
    ranked = ranking(matrix)
    ranked = ranked.loc[ranked["score"].notna()]
    mains = ranked.loc[ranked["status"] == STATUS_MAIN]
    top = mains.iloc[0] if len(mains) else None
    rest = ranked if top is None else ranked.loc[ranked["sid"] != top["sid"]]
    second = rest.iloc[0] if len(rest) else None
    return top, second


def verdict_text(matrix: pd.DataFrame) -> str:
    """Одна фраза-подсказка по матрице (выбор — за участником)."""
    top, second = pick_top(matrix)

    def name(row: pd.Series, with_status: bool = False) -> str:
        inner = style.fmt_num(row["score"], 2) + (f"; {row['status']}" if with_status else "")
        return f"{row['sid']} «{row['title']}» ({inner})"

    if top is None:
        text = "Ни один сюжет не прошёл все критерии отказа: главный сюжет по этим данным не выбирается"
        if second is not None:
            text += f"; выше всех по баллу — {name(second, True)}"
        return style_nbsp(text + ".")
    text = f"Выше всех по баллу среди сюжетов, прошедших критерии отказа, — {name(top)}"
    if second is not None:
        text += f"; следующий по баллу — {name(second, True)}"
    others = matrix.loc[matrix["sid"] != top["sid"], "status"]
    if len(others) and others.isin([STATUS_LAYER, STATUS_CONTEXT]).all():
        text += "; остальные сюжеты — слой или контекст"
    return style_nbsp(text + ".")


# --- Типографика ---------------------------------------------------------------------------------

_SHORT_WORDS = ("в", "с", "к", "о", "у", "и", "а", "я", "В", "С", "К", "О", "У", "И", "А", "Я")


def style_nbsp(text: str) -> str:
    """Неразрывный пробел после однобуквенных предлогов и союзов и перед тире (``ru-text``)."""
    words = text.split(" ")
    out = []
    for i, word in enumerate(words):
        out.append(word)
        if i < len(words) - 1:
            bare = word.lstrip("(«„")
            out.append(style.NBSP if bare in _SHORT_WORDS else " ")
    joined = "".join(out)
    return joined.replace(" — ", f"{style.NBSP}— ")
