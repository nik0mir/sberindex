"""Сюжеты-кандидаты этапа 1 и матрица решения (spec_final, Б.5).

Шесть сюжетов: С1–С3 из ``PLAN.md``, С4–С6 подсказаны данными. У каждого:

- **экспертные баллы** 1–5 (сеть и динамика, практическая ценность, объяснимость, риски данных) с обоснованием
  в одну строку — их задаёт автор, участник вправе изменить;
- **балл силы сигнала и балл покрытия** — считаются из фактов разведки по порогам;
- **критерии отказа** — сюжет может стать главным, только если выполнены все; иначе он «слой» (хотя бы один
  критерий силы сигнала выполнен) или «контекст» (не выполнен ни один), независимо от балла;
- **внешняя проверка и пример МО** — переменные контекста вне признаков сюжета с ожидаемым знаком связи
  (гипотезы записаны до этапа 2), МО с наибольшим отклонением главного показателя от того, что дают уровень
  трат и регион, и кому сюжет полезен; считает сводка (``synthesis``).

Итог = Σ вес × балл, веса — ``eda.matrix_weights`` (выведены из критериев жюри, в сумме 1). Итог
раскладывается на две части в той же шкале 1–5: «по фактам» (сила сигнала и покрытие) и «экспертную» (четыре
балла автора), чтобы было видно, сколько в порядке сюжетов от данных, а сколько от оценок автора.

Честность подсказки: у критерия отказа могут быть другие оценки той же величины (границы интервала бутстрепа,
другое определение «внутри регионов», выборка без 247 внутригородских территорий Москвы и Петербурга); если
хоть одна из них даёт другой ответ, критерий «на границе». Матрица считается второй раз без внутригородских
территорий (``NO_INNER``): так видно, зависит ли выбор сюжета от решения по Москве и Петербургу. Разрыв между
лидером и следующим сюжетом сравнивается с весом одного экспертного балла: если его сдвиг на 1 меняет порядок,
подсказка об этом говорит. Матрица — подсказка: выбор сюжета делает участник.

Отступления от Б.5 (причины — в отчёте о проверке сюжетов, 26.09.2026):

- С2 оценивается без внутригородских территорий: зарплата Росстата у них описывает работодателей района,
  а не жителей, поэтому контроль «сверх зарплаты» там не работает; оценка по всем МО — другая оценка;
- сила сигнала С4 — доля дисперсии CLR, оставшаяся после очистки от уровня и региона (шкала С1): устойчивость
  очищенной оси 2023 → 2024 такая же, как у любой очищенной доли (0,87–0,93), и силу структуры не различает;
- у С5 добавлен критерий силы сигнала «согласованность полугодий сверх уровня трат и региона»: прирост доли
  маркетплейсов во многом повторяет уровень трат, а С2 и С4 проверяются именно сверх уровня и региона.

Пороги критериев — ``eda.rejection`` (и ``eda.step_test_pp``); пороги баллов — ``eda.stories`` поверх
``DEFAULTS``. Функции чистые: вход — словарь фактов ``{«e3.reliable_share»: Fact}`` и конфиг.
"""

from __future__ import annotations

import copy
import dataclasses
import logging
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd

from munnet import style
from munnet.eda.base import FACT_KINDS, Fact, make_fact

log = logging.getLogger(__name__)

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
FACT_CRITERIA: tuple[str, ...] = ("signal", "coverage")  # баллы, посчитанные из фактов разведки
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
FRAGILE_TEXT = "критерий на границе"  # добавка к роли, если другая оценка критерия даёт другой ответ
MARK_OK, MARK_FAIL, MARK_NA = "✓", "✗", "?"  # отметки критериев в полном перечне

ROLE_SIGNAL = "signal"  # критерий силы сигнала: выполнен — у сюжета есть на что опереться хотя бы как у слоя
ROLE_DATA = "data"  # критерий достаточности или чистоты данных: сам по себе сюжет слоем не делает

# Факт с числом МО панели — знаменатель покрытия (пишет сводка).
N_MO_FACT = "syn.n_mo"
# Факт с числом внутригородских территорий Москвы и Петербурга (пишет сводка).
N_INNER_FACT = "syn.n_inner"

# Та же величина без 247 внутригородских территорий Москвы и Петербурга: факт основной оценки -> факт без них
# (считает сводка теми же функциями, что и разделы). Такая оценка добавляется к другим оценкам критерия, и по
# ней же матрица считается второй раз (``substitute``). Критерий С2 уже считается без внутригородских.
NO_INNER_LABEL = "без внутригородских территорий"
NO_INNER: dict[str, str] = {
    "e3.reliable_share": "syn.reliable_share_no_inner",
    "e4.growth_half_consistency": "syn.growth_half_consistency_no_inner",
    "e4.residual_share": "syn.residual_share_no_inner",
    "e4.residual_pc1_stability": "syn.residual_pc1_stability_no_inner",
    "e4.mp_split_half": "syn.mp_split_half_no_inner",
    "e4.rho_mp_pp_level_within": "syn.rho_mp_pp_level_within_no_inner",
    "syn.mp_split_half_resid": "syn.mp_split_half_resid_no_inner",
}

# Та же величина на узлах сети (``nodes.mode`` конфига, по умолчанию Москва и Петербург — два узла-города):
# факт основной оценки -> факт сводки, посчитанный теми же функциями разделов на узлах
# (``synthesis.node_values``). В режиме узлов матрица T14 считается по этим фактам, а оценка «247
# внутригородских территорий отдельными узлами» (факты разделов) становится другой оценкой критерия и второй
# матрицей. С6 на узлах не пересчитывается: внутригородские территории и узлы-города не входят в выборку
# 5-НДФЛ (``ndfl_ok`` = false), меняется только знаменатель покрытия.
NODES_LABEL = "на узлах сети"
SEPARATE_LABEL = "247 внутригородских территорий отдельными узлами"
NODES: dict[str, str] = {
    N_MO_FACT: "syn.n_nodes",
    "e1.n_full": "syn.n_full_nodes",
    "e3.reliable_share": "syn.reliable_share_nodes",
    "e3.share_r06": "syn.share_r06_nodes",
    "e3.null_reliable_share": "syn.null_reliable_share_nodes",
    "e2.partial_rho_access_within_no_inner": "syn.partial_rho_access_nodes",
    "e2.n_access_wage": "syn.n_access_wage_nodes",
    "e4.growth_half_consistency": "syn.growth_half_consistency_nodes",
    "e4.residual_share": "syn.residual_share_nodes",
    "e4.residual_pc1_stability": "syn.residual_pc1_stability_nodes",
    "e4.n_basket": "syn.n_basket_nodes",
    "e4.mp_split_half": "syn.mp_split_half_nodes",
    "e4.rho_mp_pp_level_within": "syn.rho_mp_pp_level_within_nodes",
    "e4.mp_step_max_pp": "syn.mp_step_max_pp_nodes",
    "syn.mp_split_half_resid": "syn.mp_split_half_resid_nodes",
    "syn.mp_r2_level_region": "syn.mp_r2_level_region_nodes",
    "syn.resid_part_stability_span": "syn.resid_part_stability_span_nodes",
    "e5.n_ndfl_usable": "syn.n_ndfl_usable_nodes",
}
# Ключи основной оценки, у которых оценка разделов включает 247 внутригородских территорий (С2 в разделе уже
# без них): для них оценка разделов — другая оценка критерия с подписью ``SEPARATE_LABEL``.
SEPARATE_PREFIX = "separate:"


def separate_alternatives(facts: Mapping[str, Fact]) -> tuple[dict[str, Fact], dict[str, str]]:
    """Копии фактов разделов под ключами ``separate:<ключ>`` и карта «ключ -> копия» для ``other_estimates``.

    Только для ключей ``NODES``, чья оценка в разделах включает внутригородские территории (без С2 и без
    знаменателей покрытия).
    """
    skip = {N_MO_FACT, "e2.partial_rho_access_within_no_inner", "e2.n_access_wage", "e1.n_full"}
    skip |= {"e4.n_basket", "e5.n_ndfl_usable"}
    copies, mapping = {}, {}
    for key in NODES:
        if key in skip or key not in facts:
            continue
        alias = SEPARATE_PREFIX + key
        copies[alias] = dataclasses.replace(facts[key], key=alias)
        mapping[key] = alias
    return copies, mapping


# Параметры по умолчанию; ``eda.stories`` конфига переопределяет их ключ за ключом, отсутствующие пороги
# ``eda.rejection`` берутся из ``rejection`` (spec_final, Б.5).
DEFAULTS: dict[str, Any] = {
    # Балл силы сигнала: значение < bins[0] → 1, < bins[1] → 2, < bins[2] → 3, < bins[3] → 4, иначе 5.
    "signal_bins": {
        "s1": [0.1, 0.2, 0.3, 0.4],  # доля МО с устойчивым своим ритмом
        "s2": [0.1, 0.2, 0.3, 0.4],  # |частный ρ| уровня с доступностью рынков
        "s3": [0.3, 0.45, 0.6, 0.75],  # ρ роста по полугодиям
        # доля дисперсии CLR после очистки от уровня и региона (шкала С1; отступление от Б.5, где балл
        # давала устойчивость очищенной PC1 — она одинакова у любой очищенной доли)
        "s4_residual": [0.1, 0.2, 0.3, 0.4],
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
_TRANSFORMS: tuple[str, ...] = ("id", "exp", "expm1")  # как показать значение примера МО
_WARNED: set[tuple[str, ...]] = set()  # о лишних порогах балла сигнала предупреждаем один раз за процесс


# --- Параметры -----------------------------------------------------------------------------------


def params(cfg: Any | None = None) -> dict[str, Any]:
    """Параметры сюжетов: ``DEFAULTS``, поверх — ``eda.stories`` конфига; неизвестный ключ — ``ValueError``.

    В ``rejection`` результата — пороги критериев отказа: ``DEFAULTS["rejection"]``, поверх —
    ``eda.rejection`` конфига (секция части Г). Пороги балла сигнала, которыми не пользуется ни один сюжет
    (например, ``s4`` из прежней шкалы С4), остаются в результате, но попадают в лог предупреждением.
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
    stale = sorted(set(out["signal_bins"]) - set(DEFAULTS["signal_bins"]))
    if stale and tuple(stale) not in _WARNED:
        _WARNED.add(tuple(stale))
        log.warning(
            "eda.stories.signal_bins: пороги %s не использует ни один сюжет (сила сигнала С4 — доля остатка "
            "CLR, ключ s4_residual)",
            stale,
        )
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
    """Балл силы сигнала 1–5 из одного факта по порогам ``bins`` (по модулю при ``use_abs``).

    В обосновании значение показано со знаком, как в разделе; балл при ``use_abs`` — по модулю.
    """

    fact_key: str
    bins: tuple[float, ...]
    label: str
    use_abs: bool = False

    def __call__(self, facts: Mapping[str, Fact]) -> tuple[int | None, str]:
        value = _value(facts, self.fact_key)
        if value is None:
            return None, f"нет факта {self.fact_key}"
        score = bin_score(abs(value) if self.use_abs else value, self.bins)
        text = f"{self.label} {_fmt_like(facts, self.fact_key, value)}"
        if self.use_abs and value < 0:
            text += " (балл — по модулю)"
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
    ``alts`` — другие оценки той же величины: (ключ факта, подпись), например границы интервала бутстрепа или
    другое определение «внутри регионов»; оценка без внутригородских территорий (``NO_INNER``) добавляется
    сама. Другие оценки не меняют ответ критерия, но показывают, устойчив ли он. ``ref`` — числа для сравнения
    (ключ факта, подпись): что даёт та же мера у заведомо неинформативной величины.
    """

    name: str
    fact_key: str
    op: str  # ">=", "<=", "<", "abs>="
    threshold_key: str
    text: str
    times_fact: str = ""
    role: str = ROLE_SIGNAL
    alts: tuple[tuple[str, str], ...] = ()
    ref: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class CriterionResult:
    """Итог проверки критерия: выполнен (True/False) или не проверяется (None — нет факта).

    ``fragile`` — хотя бы одна другая оценка той же величины (``Rejection.alts``) даёт другой ответ.
    """

    name: str
    ok: bool | None
    role: str
    text: str  # «доля МО с устойчивым своим ритмом 15,6%, нужно ≥ 25,0% (проверка: …)»
    fragile: bool = False
    short: str = ""  # то же без других оценок и чисел для сравнения


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


AltMaps = Sequence[tuple[Mapping[str, str], str]]
DEFAULT_ALTS: tuple[tuple[Mapping[str, str], str], ...] = ((NO_INNER, NO_INNER_LABEL),)


def other_estimates(rej: Rejection, alt_maps: AltMaps | None = None) -> list[tuple[str, str]]:
    """Другие оценки критерия: ``rej.alts`` и оценки из карт ``alt_maps`` (ключ основной оценки -> факт другой
    оценки, подпись); по умолчанию — оценка без внутригородских территорий (``NO_INNER``)."""
    alts = list(rej.alts)
    for mapping, label in DEFAULT_ALTS if alt_maps is None else alt_maps:
        extra = mapping.get(rej.fact_key)
        if extra and extra not in {key for key, _ in alts}:
            alts.append((extra, label))
    return alts


def check_rejection(
    rej: Rejection,
    facts: Mapping[str, Fact],
    prm: Mapping[str, Any],
    eda: Mapping[str, Any] | None,
    alt_maps: AltMaps | None = None,
) -> CriterionResult:
    """Проверяет один критерий отказа по фактам; пропуск факта — ``ok=None``.

    Значение показывается со знаком, как в разделе; у критерия «по модулю» это сказано в пороге. Другие
    оценки (``other_estimates``, пропущенные факты не учитываются) сравниваются с тем же порогом: если хоть
    одна даёт другой ответ, критерий «на границе» (``fragile``), и текст называет каждую с отметкой ✓ или ✗;
    иначе — «проверка: …; вывод тот же». Числа для сравнения (``rej.ref``) идут в конце.
    """
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
    need = "по модулю нужно" if rej.op.startswith("abs") else "нужно"
    if rej.times_fact:
        limit_text = (
            f"{style.fmt_num(threshold, 1 if threshold % 1 else 0)} × "
            f"{_fmt_like(facts, rej.times_fact, scale)} = {_fmt_like(facts, rej.fact_key, limit)}"
        )
    else:
        limit_text = _fmt_like(facts, rej.fact_key, limit)
    ok = bool(compare(value, limit))
    text = f"{rej.text} {_fmt_like(facts, rej.fact_key, value)}, {need} {sym} {limit_text}"
    short = style_nbsp(text)
    found = [
        (key, label, v)
        for key, label in other_estimates(rej, alt_maps)
        if (v := _value(facts, key)) is not None
    ]
    fragile = any(bool(compare(v, limit)) != ok for _, _, v in found)
    notes = []
    if found and fragile:
        listed = "; ".join(
            f"{label} — {_fmt_like(facts, key, v)} {MARK_OK if compare(v, limit) else MARK_FAIL}"
            for key, label, v in found
        )
        notes.append(f"{FRAGILE_TEXT}: {listed}")
    elif found:
        listed = "; ".join(f"{label} — {_fmt_like(facts, key, v)}" for key, label, v in found)
        notes.append(f"проверка: {listed}; вывод тот же")
    refs = [
        (label, facts[key].text) for key, label in rej.ref if key in facts and facts[key].value is not None
    ]
    if refs:
        notes.append("для сравнения: " + "; ".join(f"{label} — {t}" for label, t in refs))
    if notes:
        text += " (" + "; ".join(notes) + ")"
    return CriterionResult(rej.name, ok, rej.role, style_nbsp(text), fragile, short)


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


def criteria_text(results: Sequence[CriterionResult]) -> str:
    """Все критерии сюжета с отметкой: «✓» выполнен, «✗» не выполнен, «?» нет факта; через «; »."""
    marks = {True: MARK_OK, False: MARK_FAIL, None: MARK_NA}
    return "; ".join(f"{marks[r.ok]}{style.NBSP}{r.text}" for r in results)


def role_text(status: str, results: Sequence[CriterionResult]) -> str:
    """Роль для таблицы: статус и «критерий на границе», если хоть один критерий неустойчив."""
    return f"{status} ({FRAGILE_TEXT})" if any(r.fragile for r in results) else status


# --- Сюжеты --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class External:
    """Внешняя проверка: переменная контекста вне признаков сюжета и ожидаемый знак её связи с главным
    показателем внутри регионов (+1 или −1). Гипотеза записана до этапа 2; колонка — из таблицы внешних
    переменных сводки (``synthesis.external_frame``)."""

    column: str
    sign: int
    label: str


@dataclass(frozen=True)
class Example:
    """Пример МО: наибольшее отклонение показателя ``column`` от того, что дают контроли ``controls``
    и регион (МНК). ``transform`` — как показать значение (id, exp, expm1), ``kind`` — вид факта для текста,
    ``what`` — что за показатель («прирост доли маркетплейсов»)."""

    column: str
    transform: str
    kind: str
    controls: tuple[str, ...]
    what: str


@dataclass(frozen=True)
class Story:
    """Сюжет-кандидат: код для ключей фактов (``key``), номер «С1», название, суть, баллы, критерии отказа.

    ``main`` — главный показатель сюжета для внешней проверки (колонка показателей МО сводки), ``external`` —
    внешние переменные с ожидаемым знаком, ``example`` — как искать пример МО, ``user`` — кому сюжет полезен
    и для какого решения.
    """

    sid: str
    title: str
    gist: str
    manual: dict[str, tuple[int, str]]  # network_dynamics, practical, explainability, data_risk
    signal: Callable[[Mapping[str, Fact]], tuple[int | None, str]]
    coverage: Callable[[Mapping[str, Fact]], tuple[int | None, str]]
    rejections: tuple[Rejection, ...]
    key: str = ""  # латинский код для ключей фактов: «s1»
    main: str = ""
    external: tuple[External, ...] = ()
    example: Example | None = None
    user: str = ""


# Кто стоит в знаменателе покрытия и в подписях критериев: МО панели или узлы сети (``nodes.mode``).
UNIT_MO, UNIT_NODES = "mo", "nodes"
UNIT_WORDS: dict[str, dict[str, str]] = {
    UNIT_MO: {"nom": "МО", "gen": "МО"},
    UNIT_NODES: {"nom": "узлы сети", "gen": "узлов сети"},
}


def build_stories(prm: Mapping[str, Any] | None = None, unit: str = UNIT_MO) -> tuple[Story, ...]:
    """Шесть сюжетов Б.5 с порогами баллов из ``prm`` (``params(cfg)``; None — ``DEFAULTS``).

    ``unit`` — чьи доли и счётчики в подписях: ``mo`` (МО панели, факты разделов) или ``nodes`` (узлы сети,
    факты ``NODES``): у матрицы на узлах знаменатель — узлы, и называть их «МО» нельзя.
    """
    if unit not in UNIT_WORDS:
        raise ValueError(f"unit = {unit!r}: допустимы {sorted(UNIT_WORDS)}")
    nom, gen = UNIT_WORDS[unit]["nom"], UNIT_WORDS[unit]["gen"]
    if unit == UNIT_NODES:
        s2_where = "внутри групп регионов на узлах сети (зарплата Москвы — выброс, не входит)"
    else:
        s2_where = "внутри регионов без внутригородских территорий (у них зарплата — по месту работы)"
    prm = params(None) if prm is None else prm
    bins = {k: tuple(float(x) for x in v) for k, v in prm["signal_bins"].items()}
    bands = tuple(float(x) for x in prm["coverage_bands"])

    def cover(fact_key: str, label: str) -> CoverageScore:
        return CoverageScore(fact_key, bands, label)

    full = f"{nom} с полным рядом за 24 месяца"
    level23 = ("log_level_2023",)
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
            signal=BinScore("e3.reliable_share", bins["s1"], f"доля {gen} с устойчивым своим ритмом"),
            coverage=cover("e1.n_full", full),
            rejections=(
                Rejection(
                    "s1_share",
                    "e3.reliable_share",
                    ">=",
                    "rejection.s1_reliable_share_min",
                    f"доля {gen} с устойчивым своим ритмом",
                    alts=(("e3.share_r06", "без условия на размах"),),
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
            main="summer_excess",
            external=(
                External("emp_sh_B_2023", 1, "доля занятых в добыче"),
                External("point_lat", 1, "широта"),
            ),
            example=Example("summer_excess", "id", "pct_signed", level23, "летний избыток трат"),
            user="власти северных регионов и торговля — завоз и сезонная нагрузка на сервисы",
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
                "e2.partial_rho_access_within_no_inner",
                bins["s2"],
                f"частный ρ уровня с доступностью рынков сверх зарплаты {s2_where}",
                use_abs=True,
            ),
            coverage=cover("e2.n_access_wage", f"{nom} с доступностью рынков и зарплатой"),
            rejections=(
                Rejection(
                    "s2_partial",
                    "e2.partial_rho_access_within_no_inner",
                    "abs>=",
                    "rejection.s2_partial_rho_min",
                    f"частный ρ уровня с доступностью рынков сверх зарплаты {s2_where}",
                    alts=(
                        ("e2.partial_rho_access_within", "по всем МО"),
                        ("e2.partial_rho_access_within_lo", "нижняя граница интервала бутстрепа по всем МО"),
                        ("e2.partial_rho_access_within_hi", "верхняя"),
                        ("e2.partial_rho_access_within_localrank", "ранги внутри регионов, все МО"),
                        ("e2.partial_rho_access_within_size", "при равном размере МО, все МО"),
                    ),
                ),
            ),
            main="log_level_2024",
            external=(External("retail_pc", 1, "оборот розничной торговли на жителя"),),
            example=Example("log_level_2024", "exp", "rub", ("log_wage_2023",), "траты жителя в месяц"),
            user="власти субъекта — где размещать услуги и транспорт для периферии",
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
            main="growth_log",
            external=(External("payroll_growth", 1, "рост фонда оплаты труда 2024 к 2023"),),
            example=Example("growth_log", "expm1", "pct_signed", level23, "номинальный рост трат"),
            user="мониторинг потребления — какие МО проседают по тратам",
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
                "e4.residual_share",
                bins["s4_residual"],
                "доля дисперсии CLR после очистки от уровня и региона",
            ),
            coverage=cover("e4.n_basket", f"{nom} с 12 месяцами обоих лет"),
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
                    ref=(("syn.resid_part_stability_span", "у любой очищенной доли по отдельности"),),
                ),
            ),
            main="basket_resid_pc1",
            external=(External("catering_turnover_pc", 1, "оборот общепита на жителя"),),
            example=Example("sh_cafe_2024", "id", "pct", ("log_level_2024",), "доля общепита в тратах"),
            user="торговые сети и сервисы — формат магазинов и кафе в МО",
        ),
        Story(
            sid="С5",
            key="s5",
            title="Сдвиг к маркетплейсам",
            gist="где и как быстро растёт доля маркетплейсов в тратах",
            manual={
                "network_dynamics": (
                    3,
                    "помесячные ряды доли у всех МО почти одинаковы (общий рост и ритм), свой остаток "
                    "слабый, опережения по лаговым корреляциям нет; рёбра — по сходству годовых признаков",
                ),
                "practical": (4, "ритейл, логистика, пункты выдачи, местный бизнес"),
                "explainability": (4, "«доля маркетплейсов растёт быстрее там, где тратят меньше»"),
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
                    "s5_resid",
                    "syn.mp_split_half_resid",
                    ">=",
                    "rejection.s5_split_half_min",
                    "согласованность прироста доли по полугодиям сверх уровня трат и региона "
                    "(ρ остатков МНК)",
                    ref=(
                        ("syn.mp_r2_level_region", "доля разброса прироста, которую дают уровень и регион"),
                    ),
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
                    "наибольший скачок годового прироста доли между соседними месяцами (п. п.)",
                    role=ROLE_DATA,
                ),
            ),
            main="mp_pp_change",
            external=(
                External("market_access", -1, "индекс доступности рынков"),
                External("ip_per_1000", -1, "индивидуальные предприниматели на 1000 жителей"),
            ),
            example=Example("mp_pp_change", "id", "pp", level23, "прирост доли маркетплейсов"),
            user="маркетплейсы и логистика — где открывать пункты выдачи; местная торговля — где быстрее "
            "теряет долю",
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
                    "5-НДФЛ по работодателю, аномальные отношения получателей к жителям, Росстат без малого "
                    "бизнеса",
                ),
            },
            signal=BinScore("e5.suburb_ratio", bins["s6"], "траты к доходу 5-НДФЛ: пригороды к остальным"),
            coverage=cover("e5.n_ndfl_usable", f"{nom} с пригодным доходом 5-НДФЛ"),
            rejections=(
                Rejection(
                    "s6_usable",
                    "e5.n_ndfl_usable",
                    ">=",
                    "rejection.s6_usable_min",
                    f"{nom} с пригодным доходом 5-НДФЛ",
                    role=ROLE_DATA,
                ),
                Rejection(
                    "s6_ratio",
                    "e5.suburb_ratio",
                    ">=",
                    "rejection.s6_suburb_ratio_min",
                    "траты к доходу 5-НДФЛ: пригороды к остальным",
                    alts=(
                        ("e5.suburb_sens_min", "наименьшее в вариантах правила пригорода"),
                        ("e5.suburb_sens_max", "наибольшее"),
                    ),
                ),
                Rejection(
                    "s6_p",
                    "e5.suburb_mw_p",
                    "<",
                    "rejection.s6_suburb_p_max",
                    "p теста Манна — Уитни для пригородов",
                ),
            ),
            main="log_spend_to_ndfl",
            external=(
                External(
                    "employees_to_working_age",
                    -1,
                    "работники крупных и средних организаций на жителя трудоспособного возраста",
                ),
            ),
            example=Example("log_spend_to_ndfl", "exp", "num2", (), "траты к доходу 5-НДФЛ"),
            user="власти агломераций — транспорт и налоги по месту жительства",
        ),
    )


STORIES: tuple[Story, ...] = build_stories()


def validate_stories(stories: Sequence[Story]) -> None:
    """Проверяет сюжеты: уникальные номера и коды, экспертные баллы 1–5 с обоснованием по всем четырём
    критериям, известные операции и роли критериев, другие оценки — пары «ключ факта, подпись», знак внешней
    проверки ±1, известные преобразование и вид факта примера МО; иначе ``ValueError``."""
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
            for pairs, what in ((r.alts, "другие оценки"), (r.ref, "числа для сравнения")):
                if any(len(a) != 2 or not all(str(x).strip() for x in a) for a in pairs):
                    problems.append(f"{s.sid}.{r.name}: {what} — пары «ключ факта, подпись»")
        for e in s.external:
            if e.sign not in (-1, 1) or not e.column or not e.label:
                problems.append(f"{s.sid}: внешняя проверка {e} — нужны колонка, подпись и знак ±1")
        ex = s.example
        if ex is not None and (ex.transform not in _TRANSFORMS or ex.kind not in FACT_KINDS or not ex.column):
            problems.append(f"{s.sid}: пример МО {ex} — неизвестное преобразование или вид факта")
    if problems:
        raise ValueError("; ".join(problems))


# --- Матрица решения -----------------------------------------------------------------------------


def weighted_score(scores: Mapping[str, int | None], weights: Mapping[str, float]) -> float:
    """Σ вес × балл по критериям ``weights``; хоть один балл неизвестен — NaN."""
    if any(scores.get(k) is None for k in weights):
        return float("nan")
    return float(sum(float(weights[k]) * float(scores[k]) for k in weights))


def part_score(scores: Mapping[str, int | None], weights: Mapping[str, float], keys: Sequence[str]) -> float:
    """Часть итога в той же шкале 1–5: Σ вес × балл по ``keys`` / Σ весов ``keys``.

    Итог = доля весов «по фактам» × балл по фактам + доля экспертных весов × экспертный балл. Хоть один балл
    неизвестен или веса части нулевые — NaN.
    """
    total = sum(float(weights[k]) for k in keys)
    if total <= 0 or any(scores.get(k) is None for k in keys):
        return float("nan")
    return float(sum(float(weights[k]) * float(scores[k]) for k in keys) / total)


MATRIX_COLUMNS: tuple[str, ...] = (
    "sid",
    "key",
    "title",
    "gist",
    *CRITERIA,
    "score_facts",
    "score_expert",
    "score",
    "status",
    "fragile",
    "role",
    "criteria",
    "criteria_all",
    "failed",
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
    alt_maps: AltMaps | None = None,
) -> pd.DataFrame:
    """Матрица решения: строка на сюжет — баллы по шести критериям, части итога и итог, роль, критерии отказа.

    Колонки ``MATRIX_COLUMNS``: ``score_facts`` и ``score_expert`` — части итога в шкале 1–5 (сила сигнала
    и покрытие; четыре экспертных балла), ``status`` — роль по критериям отказа, ``fragile`` — хоть один
    критерий на границе, ``role`` — роль с этой пометкой, ``criteria`` — «проходит» или невыполненные
    критерии, ``criteria_all`` — все критерии с отметками, ``failed`` — невыполненные критерии коротко (без
    других оценок). Сюжеты — в порядке ``stories`` (С1…С6).
    """
    validate_stories(stories)
    w = check_weights(weights)
    rows = []
    for s in stories:
        sig, sig_note = s.signal(facts)
        cov, cov_note = s.coverage(facts)
        scores: dict[str, int | None] = {"signal": sig, "coverage": cov}
        scores.update({k: int(v[0]) for k, v in s.manual.items()})
        results = [check_rejection(r, facts, prm, eda, alt_maps) for r in s.rejections]
        status = story_status(results)
        manual_note = "; ".join(f"{CRITERIA_SHORT[k]} {s.manual[k][0]}: {s.manual[k][1]}" for k in s.manual)
        rows.append(
            {
                "sid": s.sid,
                "key": s.key,
                "title": s.title,
                "gist": s.gist,
                **{k: scores[k] for k in CRITERIA},
                "score_facts": part_score(scores, w, FACT_CRITERIA),
                "score_expert": part_score(scores, w, MANUAL_CRITERIA),
                "score": weighted_score(scores, w),
                "status": status,
                "fragile": any(r.fragile for r in results),
                "role": role_text(status, results),
                "criteria": rejection_text(results),
                "criteria_all": criteria_text(results),
                "failed": "; ".join(r.short for r in results if r.ok is False),
                "signal_note": sig_note,
                "coverage_note": cov_note,
                "manual_note": style_nbsp(manual_note),
            }
        )
    out = pd.DataFrame(rows, columns=list(MATRIX_COLUMNS))
    for k in CRITERIA:
        out[k] = out[k].astype("Int64")
    return out


def substitute(facts: Mapping[str, Fact], mapping: Mapping[str, str] = NO_INNER) -> dict[str, Fact]:
    """Факты, где основная оценка заменена другой (по умолчанию — без внутригородских территорий).

    Ключ факта остаётся прежним, значение и текст — другой оценки; нет другой оценки — факт как был.
    """
    out = dict(facts)
    for main, other in mapping.items():
        if other in facts and facts[other].value is not None:
            out[main] = dataclasses.replace(facts[other], key=main)
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


def leader(matrix: pd.DataFrame) -> pd.Series | None:
    """Главный кандидат (``pick_top``), а если ни один сюжет не прошёл критерии — лучший по итогу."""
    top, second = pick_top(matrix)
    return top if top is not None else second


def _name(row: pd.Series, score_col: str = "score", with_status: bool = False, total: bool = False) -> str:
    """«С5 «Сдвиг к маркетплейсам» (3,85; слой, критерий на границе)» — номер, название, балл и роль;
    при ``total`` балл подписан: «(итог 3,75)»."""
    inner = ("итог " if total else "") + style.fmt_num(row[score_col], 2)
    if with_status:
        inner += f"; {row['status']}" + (f", {FRAGILE_TEXT}" if bool(row.get("fragile", False)) else "")
    return f"{row['sid']} «{row['title']}» ({inner})"


def _join(names: Sequence[str]) -> str:
    names = list(names)
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " и " + names[-1]


def _stories_word(n: int) -> str:
    """«2 сюжета», «5 сюжетов»."""
    if n % 10 == 1 and n % 100 != 11:
        return f"{n} сюжет"
    if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
        return f"{n} сюжета"
    return f"{n} сюжетов"


def verdict_text(matrix: pd.DataFrame) -> str:
    """Одна-две фразы подсказки по матрице (выбор — за участником).

    Никто не прошёл критерии отказа — так и сказано, и назван лучший по итогу с его ролью; прошли несколько —
    названы все по убыванию итога; прошёл один — назван он и следующий по итогу.
    """
    top, second = pick_top(matrix)
    if top is None:
        text = "Ни один сюжет не прошёл все критерии отказа: главного сюжета по этим данным нет"
        if second is not None:
            text += f"; выше всех по итогу — {_name(second, with_status=True)}"
            if str(second.get("failed", "")):
                text += f", у него не выполнено: {second['failed']}"
        return style_nbsp(text + ".")
    ranked = ranking(matrix)
    mains = ranked.loc[(ranked["status"] == STATUS_MAIN) & ranked["score"].notna()]
    if len(mains) > 1:
        names = [
            _name(r)
            if not bool(r["fragile"])
            else f"{r['sid']} «{r['title']}» ({style.fmt_num(r['score'], 2)}; {FRAGILE_TEXT})"
            for _, r in mains.iterrows()
        ]
        text = f"Все критерии отказа прошли {_stories_word(len(mains))}: {_join(names)}"
        rest = ranked.loc[(ranked["status"] != STATUS_MAIN) & ranked["score"].notna()]
        if len(rest) and float(rest.iloc[0]["score"]) > float(mains.iloc[-1]["score"]):
            text += f"; выше некоторых из них по итогу — {_name(rest.iloc[0], with_status=True)}"
        return style_nbsp(text + ".")
    text = f"Все критерии отказа прошёл только {_name(top)}"
    if second is not None:
        text += f"; следующий по итогу — {_name(second, with_status=True)}"
    return style_nbsp(text + ".")


def sensitivity_text(
    matrix: pd.DataFrame, other: pd.DataFrame, n_inner: int | None, where: str | None = None
) -> str:
    """Меняет ли выбор сюжета другое решение по Москве и Петербургу: ``other`` — матрица при этом решении.

    ``where`` — начало фразы о другом решении («С 247 внутригородскими территориями отдельными узлами»); по
    умолчанию — исключение внутригородских территорий («Без 247 внутригородских территорий Москвы и
    Петербурга»). Сравниваются сюжеты, прошедшие все критерии отказа, и лучший по итогу. Ничего не
    меняется — одна фраза об этом; иначе — кто начинает или перестаёт проходить критерии и кто выходит вперёд.
    """
    if where is None:
        where = "Без " + (f"{n_inner} " if n_inner else "") + "внутригородских территорий Москвы и Петербурга"

    def mains(m: pd.DataFrame) -> list[str]:
        r = ranking(m)
        return list(r.loc[(r["status"] == STATUS_MAIN) & r["score"].notna(), "sid"])

    a, b = mains(matrix), mains(other)
    lead_a, lead_b = leader(matrix), leader(other)
    sid_a = None if lead_a is None else lead_a["sid"]
    sid_b = None if lead_b is None else lead_b["sid"]
    if a == b and sid_a == sid_b:
        return style_nbsp(f"{where} роли сюжетов и лидер по итогу те же.")
    rows = other.set_index("sid", drop=False)
    parts = []
    gained = [s for s in b if s not in a]
    lost = [s for s in a if s not in b]
    if gained:
        verb = "проходит" if len(gained) == 1 else "проходят"
        parts.append(f"все критерии отказа {verb} {_join([_name(rows.loc[s], total=True) for s in gained])}")
    if lost:
        verb = "перестаёт" if len(lost) == 1 else "перестают"
        parts.append(f"{verb} проходить критерии {_join(lost)}")
    if lead_b is not None and sid_b != sid_a and sid_b not in gained:
        parts.append(f"выше всех по итогу — {_name(lead_b, with_status=True)}")
    if not parts and lead_b is not None:
        parts.append(f"выше всех по итогу — {_name(lead_b, with_status=True)}")
    text = f"{where} " + "; ".join(parts)
    return style_nbsp(text + ": выбор сюжета связан с решением по Москве и Петербургу.")


@dataclass(frozen=True)
class Lead:
    """Главный кандидат и ближайший по итогу соперник: разрыв итогов и экспертные критерии, сдвиг одного
    балла в которых на 1 меняет их порядок (вес критерия больше разрыва и балл есть куда сдвинуть)."""

    top: str
    rival: str
    margin: float  # итог главного кандидата − итог соперника
    flips: tuple[str, ...]


def lead(matrix: pd.DataFrame, weights: Mapping[str, float]) -> Lead | None:
    """Разрыв лидера с лучшим по итогу из остальных сюжетов; нет пары — None.

    Лидер — главный кандидат (``pick_top``); если ни один сюжет не прошёл критерии отказа — лучший по итогу.
    """
    top, rival = pick_top(matrix)
    if top is None:
        ranked = ranking(matrix)
        ranked = ranked.loc[ranked["score"].notna()]
        if len(ranked) < 2:
            return None
        top, rival = ranked.iloc[0], ranked.iloc[1]
    if rival is None:
        return None
    margin = float(top["score"] - rival["score"])
    flips = tuple(
        k
        for k in MANUAL_CRITERIA
        if float(weights[k]) > margin and (int(rival[k]) < SCORE_MAX or int(top[k]) > SCORE_MIN)
    )
    return Lead(str(top["sid"]), str(rival["sid"]), margin, flips)


def lead_text(matrix: pd.DataFrame, weights: Mapping[str, float]) -> str:
    """Насколько устойчив порядок лидера и соперника к экспертным баллам; нет пары — пусто."""
    info = lead(matrix, weights)
    if info is None:
        return ""
    names = f"{info.top} и {info.rival}"
    gap = style.fmt_num(info.margin, 2)
    if info.margin <= 0:
        rival = matrix.loc[matrix["sid"] == info.rival, "status"]
        if len(rival) and rival.iloc[0] == STATUS_MAIN:
            return style_nbsp(f"Итоги {names} равны: какой из них главный, решает участник.")
        return style_nbsp(f"По итогу {info.rival} не ниже {info.top}, но не прошёл все критерии отказа.")
    if info.flips:
        w = sorted(float(weights[k]) for k in info.flips)
        span = (
            style.fmt_num(w[0], 2) if w[0] == w[-1] else f"{style.fmt_num(w[0], 2)}–{style.fmt_num(w[-1], 2)}"
        )
        return style_nbsp(
            f"Разрыв между {names} — {gap} балла, а сдвиг одного экспертного балла на 1 (вес в итоге — "
            f"{span}) меняет их порядок: при таком разрыве первое место определяют экспертные баллы."
        )
    return style_nbsp(
        f"Разрыв между {names} — {gap} балла: сдвиг любого экспертного балла на 1 порядок не меняет."
    )


def facts_leader(matrix: pd.DataFrame) -> pd.Series | None:
    """Сюжет с наибольшим баллом «по фактам» (при равенстве — по итогу, затем по порядку С1…С6)."""
    d = matrix.loc[matrix["score_facts"].notna()]
    if d.empty:
        return None
    d = d.assign(_pos=range(len(d))).sort_values(
        ["score_facts", "score", "_pos"], ascending=[False, False, True]
    )
    return d.iloc[0]


def advice_text(matrix: pd.DataFrame, weights: Mapping[str, float], extra: Sequence[str] = ()) -> str:
    """Подсказка к матрице: у кого выполнены все критерии (с пометкой «на границе»), устойчив ли порядок
    лидеров к экспертным баллам, кто впереди по одним фактам; ``extra`` — фразы сводки (чувствительность
    к внутригородским территориям, прототип типологии). Выбор — за участником."""
    parts = []
    mains = matrix.loc[matrix["status"] == STATUS_MAIN]
    if len(mains):
        names = [
            f"{r.sid} «{r.title}»" + (f" ({FRAGILE_TEXT})" if r.fragile else "") for r in mains.itertuples()
        ]
        parts.append(f"Все критерии отказа выполнены у {_join(names)}.")
    else:
        parts.append("Ни у одного сюжета не выполнены все критерии отказа.")
    if text := lead_text(matrix, weights):
        parts.append(text)
    top = leader(matrix)
    best = facts_leader(matrix)
    if best is not None and top is not None and best["sid"] != top["sid"]:
        parts.append(
            f"По одним фактам (сила сигнала и покрытие) впереди {_name(best, 'score_facts')} против "
            f"{style.fmt_num(top['score_facts'], 2)} у {top['sid']}, а {top['sid']} выходит вперёд за счёт "
            f"экспертных баллов ({style.fmt_num(top['score_expert'], 2)} против "
            f"{style.fmt_num(best['score_expert'], 2)})."
        )
    elif best is not None and top is not None:
        parts.append(f"{top['sid']} впереди и по одним фактам (сила сигнала и покрытие).")
    parts.extend(t for t in extra if t)
    return style_nbsp(" ".join(parts))


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
