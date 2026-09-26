"""Сводка разведки (spec_final, Б.4 «Сводка», Б.5, Б.6): показатели МО, «регион или место», матрица решения.

1. ``collect_indicators`` — показатели МО в одну таблицу: колонки ``EdaData.mo`` (уровень, рост, CLR долей,
   прирост доли маркетплейсов, летний избыток, декабрьский пик, ln трат к зарплате) и ``Finding.indicators``
   разделов (E3 — свой ритм, E4 — очищенная ось корзины, E5 — ln трат к доходу 5-НДФЛ). Таблица становится
   ``outputs/eda/indicators.parquet``.
2. ``region_or_place`` (T13, F16) — для каждого показателя η² региона (доля разброса между регионами)
   и I Морана (сходство с ближайшими соседями, kNN, перестановочный тест), с внутригородскими
   территориями Москвы и Петербурга и без них. Показатель с η² выше порога — «региональный»: для этапа 2
   его лучше брать относительно региона.
3. ``decision_matrix`` (T14) — матрица решения по сюжетам ``stories.STORIES``: баллы, итог, роль и
   критерии отказа, посчитанные из фактов разделов.

Параметры — ``eda.synthesis`` конфига поверх ``DEFAULTS``; случайность (перестановки I Морана) — только
``ctx.rng``. Числа текста — факты ``syn.*``.
"""

from __future__ import annotations

import copy
import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from munnet import style
from munnet.config import Config
from munnet.contracts import YEARS
from munnet.eda import stats, stories
from munnet.eda.base import SYNTHESIS_ID, Fact, Finding, SectionContext
from munnet.eda.data import CONTEXT_YEAR, EdaData

log = logging.getLogger(__name__)

SECTION_ID = SYNTHESIS_ID  # «syn»: префикс фактов сводки
TITLE = "Регион или место"
MO_SOURCE = "mo"  # источник показателя «общая таблица МО» (EdaData.mo)

# Параметры по умолчанию; ``eda.synthesis`` конфига переопределяет их ключ за ключом.
DEFAULTS: dict[str, Any] = {
    # Показатели МО из EdaData.mo: имя -> [колонка, преобразование id | log]; {y} — eda.reference_year,
    # {c} — год контекста (2023). Состав — spec_final, Б.4 «Сводка».
    "mo_indicators": {
        "log_level_{y}": ["log_level_{y}", "id"],
        "growth_log": ["growth_log", "id"],
        "clr_food_{y}": ["clr_food_{y}", "id"],
        "clr_marketplace_{y}": ["clr_marketplace_{y}", "id"],
        "clr_transport_{y}": ["clr_transport_{y}", "id"],
        "clr_health_{y}": ["clr_health_{y}", "id"],
        "clr_cafe_{y}": ["clr_cafe_{y}", "id"],
        "clr_other_{y}": ["clr_other_{y}", "id"],
        "mp_pp_change": ["mp_pp_change", "id"],
        "summer_excess": ["summer_excess", "id"],
        "dec_peak": ["dec_peak", "id"],
        "log_spend_to_wage_{c}": ["spend_to_wage_{c}", "log"],
    },
    # Б.6 (4): η² выше — показатель почти целиком региональный, для этапа 2 брать относительно региона.
    "eta2_regional": 0.6,
    # «Больше свойство места»: η² ниже — разброс внутри регионов больше, чем между ними.
    "place_eta2_max": 0.5,
    "headline": {  # проверка заголовка F16 (spec_final, В.5)
        "level": "log_level_{y}",  # показатель «свойство региона»
        # показатели «больше свойство места» -> слово в заголовке F16
        "dynamic": {"growth_log": "рост", "mp_pp_change": "сдвиг к онлайну"},
        "gap_min": 0.3,  # η² уровня минус наибольший η² динамики — не меньше
    },
}

# Подписи показателей МО: полная (таблицы, indicators.parquet) и короткая (ось F16).
MO_LABELS: dict[str, tuple[str, str]] = {
    "log_level_{y}": ("Уровень трат {y}, ln ₽ на жителя в месяц", "Уровень трат (ln)"),
    "growth_log": ("Рост трат {y} к {p}, ln, номинал", "Номинальный рост трат"),
    "clr_food_{y}": ("CLR доли продовольствия, {y}", "Доля продовольствия (CLR)"),
    "clr_marketplace_{y}": ("CLR доли маркетплейсов, {y}", "Доля маркетплейсов (CLR)"),
    "clr_transport_{y}": ("CLR доли транспорта, {y}", "Доля транспорта (CLR)"),
    "clr_health_{y}": ("CLR доли здоровья, {y}", "Доля здоровья (CLR)"),
    "clr_cafe_{y}": ("CLR доли общественного питания, {y}", "Доля общепита (CLR)"),
    "clr_other_{y}": ("CLR доли «Прочего», {y}", "Доля «Прочего» (CLR)"),
    "mp_pp_change": ("Прирост доли маркетплейсов {p} → {y}, п. п.", "Прирост доли маркетплейсов"),
    "summer_excess": ("Летний избыток трат, среднее двух лет", "Летний избыток трат"),
    "dec_peak": ("Декабрьский пик трат, среднее двух лет", "Декабрьский пик трат"),
    "log_spend_to_wage_{c}": ("Траты к средней зарплате, {c}, ln", "Траты к зарплате (ln)"),
}
# Короткие подписи показателей разделов для оси F16 (полные — из Finding.indicator_labels).
SECTION_SHORT_LABELS: dict[str, str] = {
    "own_amplitude": "Размах своего ритма",
    "own_repro_r": "Повторяемость своего ритма",
    "own_reliable": "Устойчивый свой ритм",
    "basket_resid_pc1": "Очищенная ось корзины",
    "log_spend_to_ndfl": "Траты к доходу 5-НДФЛ (ln)",
}
_TRANSFORMS = {
    "id": lambda s: s.astype("float64"),
    "log": lambda s: np.log(s.astype("float64").where(s > 0)),
}
_FACT_KEY = re.compile(r"[a-z][a-z0-9_]*")
BINARY_LEVELS = 2  # у флага (0/1) не больше двух значений: η² и I Морана его не описывают
NB = style.NBSP


# --- Параметры -----------------------------------------------------------------------------------


def params(cfg: Config | None = None) -> dict[str, Any]:
    """Параметры сводки: ``DEFAULTS``, поверх — ``eda.synthesis``; ``{y}``, ``{p}`` и ``{c}`` подставлены.

    Неизвестный ключ в ``eda.synthesis`` — ``ValueError``: опечатка не должна молча оставлять значение
    по умолчанию. Без конфига — ``DEFAULTS`` и последний год панели.
    """
    out = copy.deepcopy(DEFAULTS)
    user = (cfg["eda"].get("synthesis") or {}) if cfg is not None else {}
    unknown = sorted(set(user) - set(DEFAULTS))
    if unknown:
        raise ValueError(f"eda.synthesis: неизвестные ключи {unknown}; допустимы {sorted(DEFAULTS)}")
    for key, value in user.items():
        if key == "headline":
            out[key].update(value)
        else:
            out[key] = value
    years = _years(cfg)
    out["mo_indicators"] = {
        _sub(name, years): [_sub(col, years), tr] for name, (col, tr) in out["mo_indicators"].items()
    }
    out["headline"]["level"] = _sub(out["headline"]["level"], years)
    out["headline"]["dynamic"] = {_sub(k, years): v for k, v in out["headline"]["dynamic"].items()}
    return out


def _years(cfg: Config | None) -> dict[str, int]:
    """Годы подстановки: {y} — год отчёта (``eda.reference_year``), {p} — предыдущий, {c} — контекста."""
    y = int(cfg["eda"]["reference_year"]) if cfg is not None else YEARS[-1]
    return {"y": y, "p": y - 1, "c": CONTEXT_YEAR}


def _sub(text: str, years: Mapping[str, int]) -> str:
    return text.format(**years)


# --- Показатели МО -------------------------------------------------------------------------------


def mo_labels(cfg: Config | None = None) -> tuple[dict[str, str], dict[str, str]]:
    """Полные и короткие подписи показателей ``EdaData.mo`` с подставленными годами."""
    years = _years(cfg)
    full = {_sub(k, years): _sub(v[0], years) for k, v in MO_LABELS.items()}
    short = {_sub(k, years): _sub(v[1], years) for k, v in MO_LABELS.items()}
    return full, short


def collect_indicators(
    data: EdaData, findings: Sequence[Finding], cfg: Config | None = None
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Показатели МО в одну таблицу: строка на МО панели (``territory_id``), колонки — показатели.

    Сначала колонки ``EdaData.mo`` из ``eda.synthesis.mo_indicators`` (преобразование ``log`` — ln
    положительных значений), затем ``Finding.indicators`` разделов в их порядке. Одно имя из двух источников
    с разными значениями — ``ValueError``. Возвращает таблицу и подписи ``колонка -> по-русски``.
    """
    prm = params(cfg)
    full, _ = mo_labels(cfg)
    mo = data.mo
    out = pd.DataFrame({"territory_id": mo["territory_id"].astype("int32").to_numpy()})
    labels: dict[str, str] = {}
    for name, (col, transform) in prm["mo_indicators"].items():
        if col not in mo.columns:
            raise KeyError(f"eda.synthesis.mo_indicators: в EdaData.mo нет колонки {col}")
        if transform not in _TRANSFORMS:
            raise ValueError(f"показатель {name}: неизвестное преобразование {transform!r}")
        values = _TRANSFORMS[transform](mo[col])
        out[name] = values.where(np.isfinite(values)).to_numpy()
        labels[name] = full.get(name, name)
    for f in findings:
        ind = f.indicators
        if ind is None or ind.empty or len(ind.columns) <= 1:
            continue
        cols = [c for c in ind.columns if c != "territory_id"]
        part = ind[["territory_id", *cols]].copy()
        part["territory_id"] = part["territory_id"].astype("int32")
        merged = out[["territory_id"]].merge(part, on="territory_id", how="left")
        for c in cols:
            if c in out.columns:
                a, b = out[c].astype("float64"), merged[c].astype("float64")
                if not np.allclose(a, b, equal_nan=True):
                    raise ValueError(f"показатель {c} раздела {f.section} расходится с уже собранным")
                continue
            out[c] = merged[c].to_numpy()
            labels[c] = f.indicator_labels.get(c, c)
    return out, labels


def indicator_sources(ind: pd.DataFrame, findings: Sequence[Finding]) -> dict[str, str]:
    """Откуда показатель: идентификатор раздела или «mo» (общая таблица МО)."""
    src = {c: MO_SOURCE for c in ind.columns if c != "territory_id"}
    for f in findings:
        if f.indicators is None:
            continue
        for c in f.indicators.columns:
            if c != "territory_id" and c in src and src[c] == MO_SOURCE:
                src[c] = f.section
    return src


def is_binary(values: pd.Series) -> bool:
    """Флаг: не больше двух разных значений (η² и I Морана для него не считаются)."""
    return values.dropna().nunique() <= BINARY_LEVELS


# --- Регион или место (T13) ----------------------------------------------------------------------


def region_or_place(
    ind: pd.DataFrame,
    mo: pd.DataFrame,
    cfg: Config,
    rng: np.random.Generator,
    *,
    labels: Mapping[str, str] | None = None,
    sources: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """T13: η² региона и I Морана каждого показателя МО, со всеми МО и без внутригородских территорий.

    η² — доля межрегиональной суммы квадратов (``stats.eta2``, группы — ``region_code``); I Морана — на весах
    k ближайших соседей по точкам МО в проекции Альберса (``x_aea``, ``y_aea``) с ``eda.permutations``
    перестановками: основной k — ``eda.knn_main`` (I, p, z), чувствительность — ``eda.knn_sensitivity``
    (только I). Пропуски отбрасываются до весов. Флаги (≤ 2 значений) пропускаются. Порядок строк — по
    убыванию η²; ``regional`` — η² выше ``eda.synthesis.eta2_regional``.
    """
    prm = params(cfg)
    eda = cfg["eda"]
    k_main = int(eda["knn_main"])
    k_sens = [int(k) for k in eda["knn_sensitivity"]]
    perms = int(eda["permutations"])
    labels = dict(labels or {})
    sources = dict(sources or {})
    geo = mo[["territory_id", "region_code", "x_aea", "y_aea", "is_inner_city"]].copy()
    geo["territory_id"] = geo["territory_id"].astype("int32")
    table = ind.merge(geo, on="territory_id", how="left", validate="one_to_one")
    xy = table[["x_aea", "y_aea"]]
    region = table["region_code"]
    outer = ~table["is_inner_city"].astype("boolean").fillna(False).to_numpy(dtype=bool)
    rows = []
    for col in [c for c in ind.columns if c != "territory_id"]:
        y = table[col].astype("float64")
        if is_binary(y):
            log.info("Сводка: %s — флаг, η² и I Морана не считаются", col)
            continue
        main = stats.morans_i(y, xy, k_main, perms, rng)
        row: dict[str, Any] = {
            "indicator": col,
            "label": labels.get(col, col),
            "source": sources.get(col, MO_SOURCE),
            "n": int(y.notna().sum()),
            "eta2": stats.eta2(y, region),
            "moran_i": main.I,
            "moran_p": main.p,
            "moran_z": main.z,
        }
        for k in k_sens:
            row[f"moran_i_k{k}"] = stats.morans_i(y, xy, k, perms, rng).I
        y_out = y.where(outer)
        no_inner = stats.morans_i(y_out, xy, k_main, perms, rng)
        row.update(
            {
                "n_no_inner": int(y_out.notna().sum()),
                "eta2_no_inner": stats.eta2(y_out, region),
                "moran_i_no_inner": no_inner.I,
                "moran_p_no_inner": no_inner.p,
            }
        )
        rows.append(row)
    cols = [
        "indicator",
        "label",
        "source",
        "n",
        "eta2",
        "moran_i",
        "moran_p",
        "moran_z",
        *[f"moran_i_k{k}" for k in k_sens],
        "n_no_inner",
        "eta2_no_inner",
        "moran_i_no_inner",
        "moran_p_no_inner",
    ]
    out = pd.DataFrame(rows, columns=cols)
    out["regional"] = out["eta2"] > float(prm["eta2_regional"])
    out = out.sort_values("eta2", ascending=False, na_position="last", kind="mergesort")
    return out.reset_index(drop=True)


def inner_shift(t13: pd.DataFrame, short: Mapping[str, str]) -> tuple[str, float, float]:
    """Показатель, чей η² сильнее всего меняется без внутригородских: (подпись, η² с ними, η² без них).

    Нет ни одного показателя с обоими η² — (прочерк, NaN, NaN).
    """
    d = t13.dropna(subset=["eta2", "eta2_no_inner"])
    if d.empty:
        return style.NA_TEXT, float("nan"), float("nan")
    delta = d["eta2_no_inner"] - d["eta2"]
    row = d.loc[delta.abs().idxmax()]
    name = _lower_first(short.get(row["indicator"], row["label"]))
    return name, float(row["eta2"]), float(row["eta2_no_inner"])


def _lower_first(text: str) -> str:
    """Первая буква строчная, если это не аббревиатура («CLR», «ln»)."""
    if len(text) > 1 and text[0].isupper() and not text[1].isupper():
        return text[0].lower() + text[1:]
    return text


def _join_labels(names: Sequence[str]) -> str:
    """Перечень через запятую с «и» перед последним."""
    names = list(names)
    if not names:
        return style.NA_TEXT
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + f" и{NB}" + names[-1]


# --- Матрица решения (T14) -----------------------------------------------------------------------


def decision_matrix(facts: Mapping[str, Fact], cfg: Config) -> pd.DataFrame:
    """T14: матрица решения по сюжетам Б.5 из фактов разделов и сводки (колонки ``stories.MATRIX_COLUMNS``).

    Веса — ``eda.matrix_weights`` (сумма 1, иначе ``ValueError``); пороги критериев отказа — ``eda.rejection``
    и ``eda.step_test_pp``; пороги баллов — ``eda.stories``. Знаменатель покрытия — факт ``syn.n_mo``.
    """
    prm = stories.params(cfg)
    return stories.evaluate(stories.build_stories(prm), facts, cfg["eda"]["matrix_weights"], prm, cfg["eda"])


# --- График F16 ----------------------------------------------------------------------------------

X_PAD = 0.05  # поле F16 слева от наименьшего значения, в единицах η²
Y_PAD = 0.7  # поле F16 сверху и снизу от крайних строк, в долях шага строк


def plot_region_or_place(t13: pd.DataFrame, short: Mapping[str, str], threshold: float):
    """F16: η² региона (точки) и I Морана (кружки) по показателям, сверху — самые региональные.

    Серии названы в подзаголовке («точки — η², кружки — I Морана»), легенды нет; пунктир — порог η²,
    правее которого показатель для этапа 2 лучше брать относительно региона.
    """
    d = t13.dropna(subset=["eta2"]).iloc[::-1].reset_index(drop=True)
    fig, ax = style.new_figure("tall")
    y = np.arange(len(d))
    lo = d[["eta2", "moran_i"]].min(axis=1)
    hi = d[["eta2", "moran_i"]].max(axis=1)
    ax.hlines(y, lo, hi, color=style.CONTEXT, linewidth=1.2, zorder=1)
    ax.scatter(d["eta2"], y, s=40, color=style.ACCENT, zorder=3, linewidths=0)
    ax.scatter(d["moran_i"], y, s=40, facecolors="white", edgecolors=style.ACCENT, linewidths=1.3, zorder=3)
    ax.axvline(threshold, color=style.TEXT2, linewidth=0.8, linestyle="--", zorder=0)
    ax.set_yticks(y, [short.get(i, lab) for i, lab in zip(d["indicator"], d["label"], strict=True)])
    ax.tick_params(axis="y", length=0)
    values = pd.concat([d["eta2"], d["moran_i"]]).dropna()
    left = min(0.0, float(values.min())) - X_PAD if len(values) else 0.0
    ax.set_xlim(left, 1.0)
    ax.set_ylim(-Y_PAD, len(d) - 1 + Y_PAD)
    ax.grid(True, axis="x")
    ax.grid(False, axis="y")
    ax.set_xlabel("η² региона и I Морана")
    return fig


# --- Сводка --------------------------------------------------------------------------------------


def _facts_of(findings: Sequence[Finding], ctx: SectionContext) -> dict[str, Fact]:
    facts: dict[str, Fact] = {}
    for f in findings:
        facts.update(f.facts)
    facts.update(ctx.facts)
    return facts


def _indicator_facts(ctx: SectionContext, t13: pd.DataFrame) -> None:
    """Факты T13: η², I Морана и p по каждому показателю, с внутригородскими территориями и без них."""
    for row in t13.itertuples(index=False):
        key = str(row.indicator)
        if not _FACT_KEY.fullmatch(key):
            log.warning("Сводка: имя показателя %s не годится для ключа факта — факты не записаны", key)
            continue
        note = f"{row.label}; n = {row.n}"
        ctx.fact(f"eta2_{key}", row.eta2, "num2", f"η² региона: {note}")
        ctx.fact(f"eta2_{key}_no_inner", row.eta2_no_inner, "num2", f"η² без внутригородских: {note}")
        ctx.fact(f"moran_{key}", row.moran_i, "num2", f"I Морана, основной k: {note}")
        ctx.fact(f"moran_{key}_p", row.moran_p, "p", f"перестановочное p I Морана: {note}")
        ctx.fact(
            f"moran_{key}_no_inner", row.moran_i_no_inner, "num2", f"I Морана без внутригородских: {note}"
        )


def _story_facts(ctx: SectionContext, matrix: pd.DataFrame, weights: Mapping[str, float]) -> None:
    """Факты матрицы: веса, итог, роль, критерии отказа и обоснования баллов по каждому сюжету."""
    for crit, w in weights.items():
        ctx.fact(
            f"weight_{crit}", float(w), "pct", f"вес критерия «{stories.CRITERIA[crit]}» (eda.matrix_weights)"
        )
    for row in matrix.itertuples(index=False):
        k = row.key
        ctx.fact(f"title_{k}", f"{row.sid} «{row.title}»", "str", row.gist)
        ctx.fact(f"score_{k}", row.score, "num2", f"{row.sid}: Σ вес × балл (1–5)")
        ctx.fact(f"status_{k}", row.status, "str", f"{row.sid}: роль по критериям отказа")
        ctx.fact(f"reject_{k}", row.criteria, "str", f"{row.sid}: «проходит» или невыполненные критерии")
        sig = None if pd.isna(row.signal) else int(row.signal)
        cov = None if pd.isna(row.coverage) else int(row.coverage)
        ctx.fact(f"signal_{k}", sig, "int", f"{row.sid}: балл силы сигнала")
        ctx.fact(f"signal_note_{k}", row.signal_note, "str", f"{row.sid}: из чего балл силы сигнала")
        ctx.fact(f"coverage_{k}", cov, "int", f"{row.sid}: балл покрытия")
        ctx.fact(f"coverage_note_{k}", row.coverage_note, "str", f"{row.sid}: из чего балл покрытия")
        ctx.fact(f"manual_note_{k}", row.manual_note, "str", f"{row.sid}: экспертные баллы (stories.py)")
    top, second = stories.pick_top(matrix)
    ctx.fact(
        "n_main", int((matrix["status"] == stories.STATUS_MAIN).sum()), "int", "сюжетов, прошедших критерии"
    )
    for name, row in (("top", top), ("second", second)):
        note = "лучший по итогу среди прошедших критерии отказа" if name == "top" else "следующий по итогу"
        ctx.fact(f"{name}_story", stories.NO_STORY if row is None else row["sid"], "str", note)
        title = style.NA_TEXT if row is None else f"{row['sid']} «{row['title']}»"
        ctx.fact(f"{name}_story_title", title, "str", note)
        ctx.fact(f"{name}_score", None if row is None else row["score"], "num2", note)
        ctx.fact(f"{name}_status", style.NA_TEXT if row is None else row["status"], "str", note)
    ctx.fact("verdict", stories.verdict_text(matrix), "str", "подсказка по матрице; выбор — за участником")


@dataclass(frozen=True)
class RegionPlace:
    """Что показывает T13 для заголовка F16 и текста: уровень — регион, какие показатели динамики — место."""

    level: str  # ключ показателя уровня
    level_eta2: float
    level_regional: bool  # η² уровня не ниже eda.synthesis.eta2_regional
    dynamic: dict[str, float]  # η² показателей динамики (NaN — не посчитан)
    place: tuple[str, ...]  # показатели динамики «больше свойство места», в порядке конфига


def classify(t13: pd.DataFrame, prm: Mapping[str, Any]) -> RegionPlace:
    """Уровень — «свойство региона», если η² ≥ ``eta2_regional``; показатель динамики — «больше свойство
    места», если его η² < ``place_eta2_max`` и η² уровня выше его хотя бы на ``headline.gap_min``."""
    h = prm["headline"]
    eta = t13.set_index("indicator")["eta2"]
    level = float(eta.get(h["level"], np.nan))
    dynamic = {k: float(eta.get(k, np.nan)) for k in h["dynamic"]}
    gap, place_max = float(h["gap_min"]), float(prm["place_eta2_max"])
    place = tuple(
        k
        for k, v in dynamic.items()
        if np.isfinite(v) and np.isfinite(level) and v < place_max and level - v >= gap
    )
    regional = bool(np.isfinite(level) and level >= float(prm["eta2_regional"]))
    return RegionPlace(h["level"], level, regional, dynamic, place)


def headline_text(rp: RegionPlace, words: Mapping[str, str]) -> str:
    """Заголовок F16: «Уровень трат — свойство региона (η² 0,77), рост и сдвиг к онлайну — больше свойство
    места»; в перечень попадают только показатели «места» (все показатели динамики, если таких нет)."""
    keys = rp.place or tuple(rp.dynamic)
    names = _join_labels([words.get(k, k) for k in keys])
    eta = style.fmt_num(rp.level_eta2, 2)
    return stories.style_nbsp(
        f"Уровень трат — свойство региона (η²{NB}{eta}), {names} — больше свойство места"
    )


def _headline(ctx: SectionContext, prm: Mapping[str, Any], rp: RegionPlace) -> tuple[str, str]:
    """Заголовок F16 с проверкой: уровень — регион, хотя бы один показатель динамики — место."""
    h = prm["headline"]
    ok = rp.level_regional and bool(rp.place)
    dyn_text = ", ".join(f"eta2_{k} = {style.fmt_num(v, 2)}" for k, v in rp.dynamic.items())
    gap = style.fmt_num(float(h["gap_min"]), 1)
    place_max = style.fmt_num(float(prm["place_eta2_max"]), 1)
    regional = style.fmt_num(float(prm["eta2_regional"]), 1)
    detail = (
        f"F16: eta2_{rp.level} = {style.fmt_num(rp.level_eta2, 2)} (нужно ≥ {regional}), {dyn_text}; хотя бы "
        f"у одного показателя динамики η² < {place_max} и ниже η² уровня на {gap} и больше"
    )
    check = (
        f"eta2_{rp.level} ≥ {regional}; у показателей из заголовка eta2 < {place_max} "
        f"и eta2_{rp.level} − eta2 ≥ {gap}"
    )
    return ctx.headline(headline_text(rp, h["dynamic"]), ok, detail), check


def figure_alt(t13: pd.DataFrame, prm: Mapping[str, Any], short: Mapping[str, str]) -> str:
    """Альт-текст F16 из чисел T13: η² уровня и показателей динамики, число региональных показателей."""
    eta = t13.set_index("indicator")["eta2"]
    keys = [prm["headline"]["level"], *prm["headline"]["dynamic"]]
    parts = [f"{_lower_first(short.get(k, k))} — {style.fmt_num(eta[k], 2)}" for k in keys if k in eta.index]
    n_reg = int(t13["regional"].sum())
    thr = style.fmt_num(float(prm["eta2_regional"]), 1)
    return (
        f"Точки η² региона и кружки I Морана по {len(t13)} показателям МО. η²: " + "; ".join(parts) + ". "
        f"Выше {thr} — {n_reg} из {len(t13)} показателей: их лучше брать относительно региона"
    )


def summary_md(rp: RegionPlace, short: Mapping[str, str]) -> str:
    """«Что видно» и «Что это значит для сюжета» раздела «Регион или место»: числа — только ``{{syn.ключ}}``.

    Утверждения выбираются по ``rp`` (те же, что проверяет заголовок F16): «уровень почти целиком
    региональный» — только при η² уровня не ниже порога, «больше зависят от места» — только о показателях
    ``rp.place``.
    """
    lvl = rp.level

    def f(key: str) -> str:
        return "{{" + f"{SECTION_ID}.{key}" + "}}"

    seen = [
        f"Для каждого показателя МО посчитаны η² региона (доля разброса между регионами в{NB}общем разбросе) "
        f"и{NB}I Морана (насколько МО похожи на{NB}{f('knn_main')} ближайших соседей: чем больше, тем "
        f"сильнее соседи похожи друг на{NB}друга), таблица T13 и{NB}рис.{NB}16."
    ]
    if np.isfinite(rp.level_eta2):
        lead = "Уровень трат почти целиком региональный" if rp.level_regional else "У уровня трат"
        seen.append(f"{lead}: η² {f('eta2_' + lvl)}, I Морана {f('moran_' + lvl)}.")
    if rp.place:
        items = [
            f"{_lower_first(short.get(k, k))}{NB}— η² {f('eta2_' + k)}, I Морана {f('moran_' + k)}"
            for k in rp.place
        ]
        verb = "зависит" if len(items) == 1 else "зависят"
        seen.append(f"Больше {verb} от места, чем от региона: {'; '.join(items)}.")
    seen.append(
        f"Выше порога η²{NB}>{NB}{f('eta2_regional_min')}{NB}— {f('n_regional')} из{NB}{f('n_indicators')} "
        f"показателей: {f('regional_indicators')}."
    )
    inner = f"Без {f('n_inner')} внутригородских территорий Москвы и{NB}Петербурга"
    if np.isfinite(rp.level_eta2):
        inner += (
            f" η² уровня трат меняется с{NB}{f('eta2_' + lvl)} до{NB}{f('eta2_' + lvl + '_no_inner')}, "
            f"I Морана{NB}— с{NB}{f('moran_' + lvl)} до{NB}{f('moran_' + lvl + '_no_inner')};"
        )
    inner += (
        f" сильнее всего меняется η² показателя «{f('inner_max_shift')}»: с{NB}{f('inner_max_shift_from')} "
        f"до{NB}{f('inner_max_shift_to')}."
    )
    seen.append(inner)
    means = []
    if rp.level_regional:
        means.append("Типы МО по уровню трат повторят карту регионов.")
    means.append(
        "Для этапа 2 показатели выше порога η² лучше брать относительно региона (отношение к медиане региона "
        "или ранг внутри региона), остальные можно брать как есть."
    )
    means.append(
        f"Москва и{NB}Петербург дают {f('n_inner')} мелких соседних узлов, которые могут стать отдельным "
        f"«кластером столиц». Оставить их отдельными узлами, свернуть в{NB}два узла с{NB}весами населения "
        f"или исключить с{NB}оговоркой{NB}— решается до этапа 2 по T13."
    )
    means.append(f"Матрица решения (T14){NB}— подсказка, выбор сюжета за участником. {f('verdict')}")
    return "**Что видно.** " + " ".join(seen) + "\n\n**Что это значит для сюжета.** " + " ".join(means)


CAVEATS: list[str] = [
    f"η² и{NB}I Морана описывают связь показателя с{NB}географией, а{NB}не{NB}причину.",
    f"I Морана считается по{NB}ближайшим соседям по{NB}точкам внутри полигонов МО, а{NB}не{NB}по{NB}дорогам; "
    f"внутригородские территории мелкие и{NB}соседствуют друг с{NB}другом, поэтому с{NB}ними I выше.",
    f"Экспертные баллы матрицы задал автор (`src/munnet/eda/stories.py`), баллы сигнала и{NB}покрытия "
    f"посчитаны из фактов; участник вправе изменить и{NB}то, и{NB}другое.",
]

T13_LABELS: dict[str, str] = {
    "indicator": "Код",
    "label": "Показатель",
    "source": "Раздел",
    "n": "МО",
    "eta2": "η² региона",
    "moran_i": "I Морана",
    "moran_p": "p",
    "eta2_no_inner": "η² без внутригородских",
    "moran_i_no_inner": "I без внутригородских",
    "regional": "Брать относительно региона",
}
# Колонки T13 только в CSV данных F16 (в таблице отчёта их нет, чтобы она читалась на ширине страницы).
T13_CSV_ONLY: tuple[str, ...] = ("moran_z", "n_no_inner", "moran_p_no_inner")
T14_COLUMNS: tuple[str, ...] = ("sid", "title", *stories.CRITERIA, "score", "status", "criteria")
T14_LABELS: dict[str, str] = {
    "sid": "№",
    "title": "Сюжет",
    **stories.CRITERIA,
    "data_risk": "Риски данных (5 — мало)",
    "score": "Итог",
    "status": "Роль",
    "criteria": "Критерии отказа",
}


def run_synthesis(ctx: SectionContext, findings: list[Finding]) -> Finding:
    """Сводка: показатели МО, T13 и F16 «регион или место», T14 — матрица решения, факты ``syn.*``."""
    cfg, data = ctx.cfg, ctx.data
    prm = params(cfg)
    eda = cfg["eda"]
    _, short = mo_labels(cfg)
    short = {**short, **SECTION_SHORT_LABELS}

    ind, labels = collect_indicators(data, findings, cfg)
    sources = indicator_sources(ind, findings)
    t13 = region_or_place(ind, data.mo, cfg, ctx.rng, labels=labels, sources=sources)
    log.info("Сводка: %d показателей МО, %d в T13", len(ind.columns) - 1, len(t13))

    # Факты: выборка и параметры
    mo = data.mo
    ctx.fact("n_mo", len(mo), "int", "МО панели (знаменатель покрытия сюжетов)")
    ctx.fact(
        "n_inner", int(mo["is_inner_city"].sum()), "int", "внутригородских территорий Москвы и Петербурга"
    )
    ctx.fact("n_indicators", len(t13), "int", "показателей МО в T13 (без флагов)")
    ctx.fact("knn_main", int(eda["knn_main"]), "int", "соседей в весах I Морана (eda.knn_main)")
    ctx.fact("permutations", int(eda["permutations"]), "int", "перестановок в тесте I Морана")
    thr = float(prm["eta2_regional"])
    ctx.fact("eta2_regional_min", thr, "num1", "порог η²: показатель брать относительно региона")
    ctx.fact(
        "n_workplace_based", int(mo["workplace_based"].sum()), "int", "МО с показателями по месту работы"
    )
    ctx.fact(
        "n_ndfl_ok", int(mo["ndfl_ok"].sum()), "int", "МО с пригодным доходом 5-НДФЛ (флаг ndfl_ok, 2023)"
    )
    _indicator_facts(ctx, t13)
    regional = t13.loc[t13["regional"]]
    names = [
        _lower_first(short.get(i, lab))
        for i, lab in zip(regional["indicator"], regional["label"], strict=True)
    ]
    ctx.fact("n_regional", len(regional), "int", f"показателей с η² > {thr}")
    ctx.fact("regional_indicators", _join_labels(names), "str", f"показатели с η² > {thr}, по убыванию η²")
    shift_name, shift_from, shift_to = inner_shift(t13, short)
    note = "показатель, чей η² сильнее всего меняется без внутригородских территорий"
    ctx.fact("inner_max_shift", shift_name, "str", note)
    ctx.fact("inner_max_shift_from", shift_from, "num2", f"{note}: η² со всеми МО")
    ctx.fact("inner_max_shift_to", shift_to, "num2", f"{note}: η² без внутригородских")

    # Матрица решения
    matrix = decision_matrix(_facts_of(findings, ctx), cfg)
    _story_facts(ctx, matrix, stories.check_weights(eda["matrix_weights"]))

    # F16
    rp = classify(t13, prm)
    title, check = _headline(ctx, prm, rp)
    n_mo = style.fmt_num(len(mo))
    ctx.save_figure(
        plot_region_or_place(t13, short, thr),
        fid="F16",
        slug="region_or_place",
        title=title,
        subtitle=(
            f"Точки — η² региона, кружки — I Морана (k{NB}={NB}{int(eda['knn_main'])}); n до {n_mo} МО; "
            f"пунктир — η²{NB}{style.fmt_num(thr, 1)}: правее — брать относительно региона"
        ),
        alt=figure_alt(t13, prm, short),
        data=t13,
        check=check,
        # среди показателей — траты к зарплате Росстата и к доходу 5-НДФЛ
        source=style.join_sources(style.SOURCE_SBER, style.SOURCE_ROSSTAT, style.SOURCE_FNS),
    )

    # T13, T14
    k_sens = [int(k) for k in eda["knn_sensitivity"]]
    t13_cols = [c for c in t13.columns if c not in T13_CSV_ONLY]
    md_formats = {
        c: (lambda x: style.fmt_num(x, 2)) for c in ("eta2", "moran_i", "eta2_no_inner", "moran_i_no_inner")
    }
    md_formats["moran_p"] = style.fmt_p
    md_formats.update({f"moran_i_k{k}": (lambda x: style.fmt_num(x, 2)) for k in k_sens})
    t13_labels = {**T13_LABELS, **{f"moran_i_k{k}": f"I Морана, k{NB}={NB}{k}" for k in k_sens}}
    ctx.save_table(
        t13[t13_cols],
        tid="T13",
        slug="region_or_place",
        title="Регион или место: η² региона и I Морана показателей МО, с Москвой и Петербургом и без них",
        md_rows=len(t13),
        md_formats=md_formats,
        md_labels=t13_labels,
    )
    t14 = matrix[list(T14_COLUMNS)]
    ctx.save_table(
        t14,
        tid="T14",
        slug="decision_matrix",
        title="Матрица решения: баллы 1–5, итог Σ вес × балл, роль по критериям отказа",
        md_rows=len(t14),
        md_formats={"score": lambda x: style.fmt_num(x, 2)},
        md_labels=T14_LABELS,
    )

    return ctx.finding(
        title=TITLE,
        summary_md=summary_md(rp, short),
        indicators=ind,
        indicator_labels=labels,
        caveats=CAVEATS,
    )
