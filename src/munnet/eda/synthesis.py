"""Сводка разведки (spec_final, Б.4 «Сводка», Б.5, Б.6): показатели МО, «регион или место», матрица решения.

1. ``collect_indicators`` — показатели МО в одну таблицу: колонки ``EdaData.mo`` (уровень, рост, CLR долей,
   прирост доли маркетплейсов, летний избыток, декабрьский пик, ln трат к зарплате) и ``Finding.indicators``
   разделов (E3 — свой ритм, E4 — очищенная ось корзины, E5 — ln трат к доходу 5-НДФЛ). Таблица становится
   ``outputs/eda/indicators.parquet``.
2. ``region_or_place`` (T13, F16) — для каждого показателя η² региона (доля разброса между регионами)
   и I Морана (сходство с ближайшими соседями, kNN, перестановочный тест), с внутригородскими
   территориями Москвы и Петербурга и без них; η² других простых делений (федеральный округ, тип МО, размер,
   доля горожан) и η² с поправкой на шум (η² / надёжность показателя). Показатель с η² выше порога —
   «региональный»: для этапа 2 его лучше брать относительно региона. У показателя, очищенного от региона
   (очищенная ось корзины), η² равен нулю по построению и не считается.
3. ``decision_matrix`` (T14) — матрица решения по сюжетам ``stories.STORIES``: баллы, итог, роль и
   критерии отказа, посчитанные из фактов разделов; та же матрица без внутригородских территорий.
4. Проверки сюжетов: значения критериев без внутригородских территорий и согласованность прироста доли
   маркетплейсов сверх уровня и региона (теми же функциями, что и разделы E3 и E4), устойчивость очищенных
   долей корзины по отдельности, сигнал помесячных рядов доли маркетплейсов для рёбер, разброс долей
   в п. п., внешняя проверка и пример МО каждого сюжета, быстрый прототип типов (T15, k-means).

Параметры — ``eda.synthesis`` конфига поверх ``DEFAULTS``; случайность (перестановки I Морана, пары МО) —
только ``ctx.rng``, k-means — ``seed`` конфига. Числа текста — факты ``syn.*``.
"""

from __future__ import annotations

import copy
import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd

from munnet import style
from munnet.config import Config
from munnet.contracts import YEARS
from munnet.eda import s3_rhythm, s4_basket, stats, stories
from munnet.eda.base import SYNTHESIS_ID, Fact, Finding, SectionContext, display_names, make_fact, to_markdown
from munnet.eda.data import CONTEXT_YEAR, EdaData

log = logging.getLogger(__name__)

SECTION_ID = SYNTHESIS_ID  # «syn»: префикс фактов сводки
TITLE = "Регион или место"
MO_SOURCE = "mo"  # источник показателя «общая таблица МО» (EdaData.mo)
FULL_STATUS = "full"

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
    # «Около половины»: η² с поправкой на шум в пределах place_eta2_max ± noise_tol; ниже — «меньше
    # половины» (больше свойство места), выше — «почти целиком» регион.
    "place_eta2_max": 0.5,
    "noise_tol": 0.15,
    "headline": {  # проверка заголовка F16 (spec_final, В.5)
        "level": "log_level_{y}",  # показатель «свойство региона»
        # показатели динамики, о которых говорит заголовок F16 (слова заголовка — TITLE_WORDS в коде)
        "dynamic": {"growth_log": "рост", "mp_pp_change": "прирост доли маркетплейсов"},
        "gap_min": 0.3,  # η² уровня минус η² показателя динамики (без поправки на шум) — не меньше
    },
    # Сверка определений: η² и I Морана того же показателя в другом определении (колонка EdaData.mo, почему
    # сравниваем). Ориентиры Б.4 (I Морана уровня 0,82, общепита 0,74; η² CLR маркетплейсов 0,66) посчитаны
    # за другой год или по сырой доле; факты syn.check_* показывают, что расхождение — от определения.
    "definition_checks": {
        "log_level_{y}": ["log_level_{p}", "тот же показатель за {p} год"],
        "clr_cafe_{y}": ["sh_cafe_{y}", "сырая доля общепита вместо CLR (Б.1, п. 3: в связях — только CLR)"],
        "clr_marketplace_{y}": ["clr_marketplace_{p}", "тот же показатель за {p} год"],
    },
    # Показатели, очищенные от региона (остаток МНК на дамми регионов): η² региона у них 0 по построению.
    "region_residualized": ["basket_resid_pc1"],
    # Простые деления МО для η² и прототипа: группы по населению и по уровню трат, границы доли горожан
    # (сельские — меньше первой границы, смешанные, городские — не меньше второй).
    "size_groups": 10,
    "level_groups": 5,
    "urban_cuts": [0.001, 0.5],
    # Быстрый прототип типов (T15): k-means по признакам сюжета, k и число запусков.
    "prototype": {"k": 5, "n_init": 20},
    # Внешняя проверка: |ρ| меньше — «связи почти нет».
    "external_weak": 0.1,
    # Пример МО ищется среди МО с населением не меньше этой квантили (малые МО шумят).
    "example_min_pop_quantile": 0.25,
}

# Подписи показателей МО: полная (таблицы, indicators.parquet) и короткая (ось F16, текст).
MO_LABELS: dict[str, tuple[str, str]] = {
    "log_level_{y}": ("Уровень трат {y}, ln ₽ на жителя в месяц", "Уровень трат"),
    "growth_log": ("Рост трат {y} к {p}, ln, номинал", "Номинальный рост трат"),
    "clr_food_{y}": ("CLR доли продовольствия, {y}", "Доля продовольствия"),
    "clr_marketplace_{y}": ("CLR доли маркетплейсов, {y}", "Доля маркетплейсов"),
    "clr_transport_{y}": ("CLR доли транспорта, {y}", "Доля транспорта"),
    "clr_health_{y}": ("CLR доли здоровья, {y}", "Доля здоровья"),
    "clr_cafe_{y}": ("CLR доли общественного питания, {y}", "Доля общепита"),
    "clr_other_{y}": ("CLR доли «Прочего», {y}", "Доля «Прочего»"),
    "mp_pp_change": ("Прирост доли маркетплейсов {p} → {y}, п. п.", "Прирост доли маркетплейсов"),
    "summer_excess": ("Летний избыток трат, среднее двух лет", "Летний избыток трат"),
    "dec_peak": ("Декабрьский пик трат, среднее двух лет", "Декабрьский пик трат"),
    "log_spend_to_wage_{c}": ("Траты к средней зарплате, {c}, ln", "Траты к зарплате"),
}
# Короткие подписи показателей разделов для оси F16 (полные — из Finding.indicator_labels).
SECTION_SHORT_LABELS: dict[str, str] = {
    "own_amplitude": "Размах своего ритма",
    "own_repro_r": "Повторяемость своего ритма",
    "own_reliable": "Устойчивый свой ритм",
    "basket_resid_pc1": "Корзина без уровня и региона",
    "log_spend_to_ndfl": "Траты к доходу 5-НДФЛ",
}
# Части корзины в родительном падеже: «в п. п. сильнее всего различается доля продовольствия».
PART_GENITIVE: dict[str, str] = {
    "food": "продовольствия",
    "marketplace": "маркетплейсов",
    "transport": "транспорта",
    "health": "здоровья",
    "cafe": "общепита",
    "other": "«Прочего»",
}
# Слова заголовка F16 для показателей динамики (именительный падеж).
TITLE_WORDS: dict[str, str] = {"growth_log": "рост", "mp_pp_change": "прирост доли маркетплейсов"}
# Классы «сколько воспроизводимого разброса даёт регион» по η² с поправкой на шум.
CLASS_PLACE, CLASS_HALF, CLASS_REGION = "place", "half", "region"
CLASS_WORDS: dict[str, str] = {
    CLASS_PLACE: "меньше чем наполовину",
    CLASS_HALF: "наполовину",
    CLASS_REGION: "почти целиком",
}
CLASS_SHARE: dict[str, str] = {
    CLASS_PLACE: "меньше половины",
    CLASS_HALF: "около половины",
    CLASS_REGION: "почти весь",
}
# Простые деления МО: колонка ``groupings`` -> подпись.
GROUPINGS: dict[str, str] = {
    "region": "регион",
    "fd": "федеральный округ",
    "type": "тип МО",
    "size": "группа по населению",
    "spend": "группа по уровню трат",
    "urban": "доля горожан",
}
# Те же деления в винительном падеже: «место означает размер МО и долю горожан».
GROUPINGS_ACC: dict[str, str] = {
    "region": "регион",
    "fd": "федеральный округ",
    "type": "тип МО",
    "size": "размер МО",
    "spend": "уровень трат",
    "urban": "долю горожан",
}
# Федеральные округа по коду субъекта справочника СберИндекса (region_code).
FEDERAL_DISTRICTS: dict[str, tuple[int, ...]] = {
    "Центральный": (31, 32, 33, 36, 37, 40, 44, 46, 48, 50, 57, 62, 67, 68, 69, 71, 76, 77),
    "Северо-Западный": (10, 11, 29, 35, 39, 47, 51, 53, 60, 78, 83),
    "Южный": (1, 8, 23, 30, 34, 61, 91, 92),
    "Северо-Кавказский": (5, 6, 7, 9, 15, 20, 26),
    "Приволжский": (2, 12, 13, 16, 18, 21, 43, 52, 56, 58, 59, 63, 64, 73),
    "Уральский": (45, 66, 72, 74, 86, 89),
    "Сибирский": (4, 17, 19, 22, 24, 38, 42, 54, 55, 70),
    "Дальневосточный": (3, 14, 25, 27, 28, 41, 49, 65, 75, 79, 87),
}
FD_BY_REGION: dict[int, str] = {code: fd for fd, codes in FEDERAL_DISTRICTS.items() for code in codes}

_TRANSFORMS = {
    "id": lambda s: s.astype("float64"),
    "log": lambda s: np.log(s.astype("float64").where(s > 0)),
}
_FACT_KEY = re.compile(r"[a-z][a-z0-9_]*")
BINARY_LEVELS = 2  # у флага (0/1) не больше двух значений: η² и I Морана его не описывают
NB = style.NBSP
PCT = 100


# --- Параметры -----------------------------------------------------------------------------------


def params(cfg: Config | None = None) -> dict[str, Any]:
    """Параметры сводки: ``DEFAULTS``, поверх — ``eda.synthesis``; ``{y}``, ``{p}`` и ``{c}`` подставлены.

    Неизвестный ключ в ``eda.synthesis`` — ``ValueError``: опечатка не должна молча оставлять значение
    по умолчанию. Вложенные словари (``headline``, ``prototype``) сливаются ключ за ключом. Без конфига —
    ``DEFAULTS`` и последний год панели.
    """
    out = copy.deepcopy(DEFAULTS)
    user = (cfg["eda"].get("synthesis") or {}) if cfg is not None else {}
    unknown = sorted(set(user) - set(DEFAULTS))
    if unknown:
        raise ValueError(f"eda.synthesis: неизвестные ключи {unknown}; допустимы {sorted(DEFAULTS)}")
    for key, value in user.items():
        if key in ("headline", "prototype"):
            out[key].update(value)
        else:
            out[key] = value
    years = _years(cfg)
    out["mo_indicators"] = {
        _sub(name, years): [_sub(col, years), tr] for name, (col, tr) in out["mo_indicators"].items()
    }
    out["headline"]["level"] = _sub(out["headline"]["level"], years)
    out["headline"]["dynamic"] = {_sub(k, years): v for k, v in out["headline"]["dynamic"].items()}
    out["definition_checks"] = {
        _sub(k, years): [_sub(col, years), _sub(why, years)]
        for k, (col, why) in out["definition_checks"].items()
    }
    return out


def _years(cfg: Config | None) -> dict[str, int]:
    """Годы подстановки: {y} — год отчёта (``eda.reference_year``), {p} — предыдущий, {c} — контекста."""
    y = int(cfg["eda"]["reference_year"]) if cfg is not None else YEARS[-1]
    return {"y": y, "p": y - 1, "c": CONTEXT_YEAR}


def _sub(text: str, years: Mapping[str, int]) -> str:
    return text.format(**years)


def _inner(frame: pd.DataFrame) -> pd.Series:
    """Флаг внутригородской территории (пропуск — нет)."""
    return frame["is_inner_city"].astype("boolean").fillna(False).astype(bool)


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


# --- Простые деления МО --------------------------------------------------------------------------


def groupings(mo: pd.DataFrame, cfg: Config | None = None) -> pd.DataFrame:
    """Простые деления МО (колонки ``GROUPINGS``) в порядке строк ``mo``: регион, федеральный округ, тип МО,
    группа по населению (``size_groups`` равных по числу МО), группа по уровню трат 2023 года
    (``level_groups``), класс по доле горожан 2023 года (``urban_cuts``). Пропуск — NaN."""
    prm = params(cfg)
    y0 = _years(cfg)["p"]
    out = pd.DataFrame(index=mo.index)
    out["region"] = mo["region_code"].astype("Int64")
    out["fd"] = mo["region_code"].map(FD_BY_REGION)
    out["type"] = mo["mo_type"].astype("object").where(mo["mo_type"].notna())
    out["size"] = _qgroups(mo["weight"], int(prm["size_groups"]))
    out["spend"] = _qgroups(mo[f"log_level_{y0}"], int(prm["level_groups"]))
    lo, hi = (float(x) for x in prm["urban_cuts"])
    urban = mo["urban_share_2023"].astype("float64")
    out["urban"] = np.select([urban < lo, urban < hi, urban >= hi], [0.0, 1.0, 2.0], default=np.nan)
    return out


def _qgroups(values: pd.Series, n: int) -> pd.Series:
    """Группы равной численности по значению (ранги — чтобы одинаковые значения не ломали границы)."""
    v = values.astype("float64")
    ranks = v.rank(method="first")
    if ranks.notna().sum() < n:
        return pd.Series(np.nan, index=values.index)
    return pd.qcut(ranks, n, labels=False).astype("float64")


# --- Регион или место (T13) ----------------------------------------------------------------------


def spearman_brown(r: float) -> float:
    """Надёжность целого по согласованности двух половин: 2r / (1 + r); r ≤ 0 или пропуск — NaN."""
    if r is None or not np.isfinite(r) or r <= 0:
        return float("nan")
    return float(2 * r / (1 + r))


def reliabilities(mo: pd.DataFrame, facts: Mapping[str, Fact], cfg: Config | None = None) -> dict[str, float]:
    """Надёжность показателей МО — насколько повторяется порядок МО в двух независимых замерах.

    Рост и прирост доли маркетплейсов — согласованность полугодий (факты E4 ``growth_half_consistency``
    и ``mp_split_half``) с поправкой Спирмена — Брауна на длину года; уровень и CLR долей — ρ соседних лет
    (``log_level_{p}`` и ``_{y}``). У остальных показателей надёжность не оценивается.
    """
    y = _years(cfg)
    out: dict[str, float] = {}
    for key, fact in (("growth_log", "e4.growth_half_consistency"), ("mp_pp_change", "e4.mp_split_half")):
        f = facts.get(fact)
        if f is not None and isinstance(f.value, (int, float)):
            out[key] = spearman_brown(float(f.value))
    for stem in ("log_level", *(f"clr_{p}" for p in s4_basket.PARTS)):
        a, b = f"{stem}_{y['p']}", f"{stem}_{y['y']}"
        if a in mo.columns and b in mo.columns:
            out[b] = float(stats.spearman(mo[a], mo[b])[0])
    return out


def region_or_place(
    ind: pd.DataFrame,
    mo: pd.DataFrame,
    cfg: Config,
    rng: np.random.Generator,
    *,
    labels: Mapping[str, str] | None = None,
    sources: Mapping[str, str] | None = None,
    reliability: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """T13: η² региона и I Морана каждого показателя МО, со всеми МО и без внутригородских территорий.

    η² — доля межрегиональной суммы квадратов (``stats.eta2``, группы — ``region_code``); I Морана — на весах
    k ближайших соседей по точкам МО в проекции Альберса (``x_aea``, ``y_aea``) с ``eda.permutations``
    перестановками: основной k — ``eda.knn_main`` (I, p, z), чувствительность — ``eda.knn_sensitivity``
    (только I). Пропуски отбрасываются до весов. Флаги (≤ 2 значений) пропускаются. Ещё колонки: η² других
    делений (``groupings``), η² чистого шума при том же числе регионов ((k − 1)/(n − 1)), надёжность
    (``reliability``) и η² с поправкой на шум ``eta2_true`` = min(η² / надёжность, 1). У показателей
    ``region_residualized`` η² региона и округа — NaN (``by_construction``: 0 по построению). Порядок строк —
    по убыванию η²; ``regional`` — η² выше ``eda.synthesis.eta2_regional``.
    """
    prm = params(cfg)
    eda = cfg["eda"]
    k_main = int(eda["knn_main"])
    k_sens = [int(k) for k in eda["knn_sensitivity"]]
    perms = int(eda["permutations"])
    labels = dict(labels or {})
    sources = dict(sources or {})
    reliability = dict(reliability or {})
    residualized = set(prm["region_residualized"])
    geo = mo[["territory_id", "region_code", "x_aea", "y_aea", "is_inner_city"]].copy()
    geo["territory_id"] = geo["territory_id"].astype("int32")
    groups = groupings(mo, cfg)
    groups["territory_id"] = mo["territory_id"].astype("int32").to_numpy()
    table = ind.merge(geo, on="territory_id", how="left", validate="one_to_one")
    table = table.merge(groups, on="territory_id", how="left", validate="one_to_one")
    xy = table[["x_aea", "y_aea"]]
    region = table["region_code"]
    outer = ~_inner(table).to_numpy()
    rows = []
    for col in [c for c in ind.columns if c != "territory_id"]:
        y = table[col].astype("float64")
        if is_binary(y):
            log.info("Сводка: %s — флаг, η² и I Морана не считаются", col)
            continue
        main = stats.morans_i(y, xy, k_main, perms, rng)
        by_construction = col in residualized
        n = int(y.notna().sum())
        k_regions = int(region[y.notna()].nunique())
        row: dict[str, Any] = {
            "indicator": col,
            "label": labels.get(col, col),
            "source": sources.get(col, MO_SOURCE),
            "n": n,
            "eta2": np.nan if by_construction else stats.eta2(y, region),
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
                "eta2_no_inner": np.nan if by_construction else stats.eta2(y_out, region),
                "moran_i_no_inner": no_inner.I,
                "moran_p_no_inner": no_inner.p,
            }
        )
        for g in GROUPINGS:
            if g == "region":
                continue
            row[f"eta2_{g}"] = np.nan if by_construction and g == "fd" else stats.eta2(y, table[g])
        row["eta2_random"] = (k_regions - 1) / (n - 1) if n > 1 else np.nan
        rel = reliability.get(col, np.nan)
        row["reliability"] = rel
        row["eta2_true"] = min(row["eta2"] / rel, 1.0) if np.isfinite(rel) and rel > 0 else np.nan
        row["by_construction"] = by_construction
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
        *[f"eta2_{g}" for g in GROUPINGS if g != "region"],
        "eta2_random",
        "reliability",
        "eta2_true",
        "by_construction",
    ]
    out = pd.DataFrame(rows, columns=cols)
    out["regional"] = (out["eta2"] > float(prm["eta2_regional"])).fillna(False).astype(bool)
    out = out.sort_values("eta2", ascending=False, na_position="last", kind="mergesort")
    return out.reset_index(drop=True)


def definition_checks(mo: pd.DataFrame, cfg: Config, rng: np.random.Generator) -> pd.DataFrame:
    """Сверка определений: η² и I Морана (как в T13) для колонок ``eda.synthesis.definition_checks``.

    Строка на колонку-сверку; к колонкам ``region_or_place`` добавлены ``checks`` (какой показатель T13
    сверяется) и ``why`` (чем отличается определение). Колонки, которых нет в ``mo``, пропускаются.
    """
    prm = params(cfg)
    checks = {col: (main, why) for main, (col, why) in prm["definition_checks"].items() if col in mo.columns}
    skipped = sorted(set(c for c, _ in prm["definition_checks"].values()) - set(checks))
    if skipped:
        log.warning("Сводка: для сверки определений нет колонок %s в EdaData.mo", skipped)
    ind = pd.DataFrame({"territory_id": mo["territory_id"].astype("int32").to_numpy()})
    for col in checks:
        ind[col] = mo[col].astype("float64").to_numpy()
    out = region_or_place(ind, mo, cfg, rng)
    out["checks"] = out["indicator"].map({c: m for c, (m, _) in checks.items()})
    out["why"] = out["indicator"].map({c: w for c, (_, w) in checks.items()})
    return out


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


# --- Проверки сюжетов ----------------------------------------------------------------------------

_SOFT_ERRORS = (ValueError, KeyError, np.linalg.LinAlgError)


def no_inner_values(
    data: EdaData, findings: Sequence[Finding], halves: pd.DataFrame | None = None
) -> dict[str, float]:
    """Значения критериев отказа без внутригородских территорий — теми же функциями, что в разделах.

    Ключи — хвосты фактов ``stories.NO_INNER`` («residual_share_no_inner»…), кроме прироста доли сверх уровня
    (его считает ``mp_residual``). Не посчиталось (мало МО, нет итога раздела) — ключа нет, в лог
    предупреждение.
    """
    mo, wide = data.mo, data.panel_wide
    outer = mo.loc[~_inner(mo)]
    halves = s4_basket.half_year_growth(wide) if halves is None else halves
    out: dict[str, float] = {}
    try:
        b = s4_basket.basket_analysis(outer)
        out["residual_share_no_inner"] = b.clean.residual_share
        out["residual_pc1_stability_no_inner"] = b.stability
    except _SOFT_ERRORS as e:
        log.warning("Сводка: корзина без внутригородских не посчитана: %s", e)
    try:
        m = s4_basket.marketplace_checks(wide, outer, data.national, halves)
        out["mp_split_half_no_inner"] = float(m["mp_split_half"])
        out["rho_mp_pp_level_within_no_inner"] = float(m["rho_mp_pp_level_within"])
    except _SOFT_ERRORS as e:
        log.warning("Сводка: маркетплейсы без внутригородских не посчитаны: %s", e)
    try:
        g = s4_basket.growth_table(wide, outer, halves).set_index("code")
        out["growth_half_consistency_no_inner"] = float(g.loc["all", "half_consistency"])
    except _SOFT_ERRORS as e:
        log.warning("Сводка: рост без внутригородских не посчитан: %s", e)
    e3 = next((f for f in findings if f.section == "e3"), None)
    if e3 is not None and "own_reliable" in e3.indicators.columns:
        rel = mo[["territory_id", "series_status", "is_inner_city"]].merge(
            e3.indicators[["territory_id", "own_reliable"]], on="territory_id", how="left"
        )
        keep = (rel["series_status"].astype(str) == FULL_STATUS) & ~_inner(rel)
        out["reliable_share_no_inner"] = float(rel.loc[keep, "own_reliable"].astype("float64").mean())
    return {k: v for k, v in out.items() if np.isfinite(v)}


def _region_category(region: pd.Series) -> pd.Series:
    """Регион категорией (для дамми в ``s4_basket.residualize``): код региона иначе войдёт как число."""
    return region.astype("int64").astype(str).astype("category")


def mp_residual(mo: pd.DataFrame, halves: pd.DataFrame) -> dict[str, float]:
    """Прирост доли маркетплейсов сверх уровня трат и региона (критерий С5 ``s5_resid``).

    Выборка — как у ``e4.mp_split_half``: МО с долями обоих лет, уровнем 2023 года и обоими полугодиями.
    ``mp_split_half_resid`` — ρ полугодий после МНК каждого на лог-уровень 2023 года и дамми регионов (и то же
    без внутригородских — ``…_no_inner``); ``mp_r2_level`` и ``mp_r2_level_region`` — доля дисперсии годового
    прироста, объяснённая уровнем и уровнем с регионом.
    """
    y0 = YEARS[0]
    f = mo.dropna(subset=["mp_pp_change", f"log_level_{y0}"]).set_index("territory_id")
    h = halves.reindex(f.index)
    d = pd.DataFrame(
        {
            "h1": h["mp_pp_h1"],
            "h2": h["mp_pp_h2"],
            "y": f["mp_pp_change"],
            "log_level": f[f"log_level_{y0}"],
            "region": f["region_code"],
            "inner": _inner(f),
        }
    ).dropna(subset=["h1", "h2", "y", "log_level", "region"])
    out: dict[str, float] = {"mp_resid_n": float(len(d))}

    def ctrl(frame: pd.DataFrame, region: bool = True) -> pd.DataFrame:
        c = pd.DataFrame({"log_level": frame["log_level"]}, index=frame.index)
        if region:
            c["region"] = _region_category(frame["region"])
        return c

    try:
        res, _, _ = s4_basket.residualize(d[["h1", "h2"]], ctrl(d))
        out["mp_split_half_resid"] = float(stats.spearman(res["h1"], res["h2"])[0])
        _, r2_both, _ = s4_basket.residualize(d[["y"]], ctrl(d))
        _, r2_level, _ = s4_basket.residualize(d[["y"]], ctrl(d, region=False))
        out["mp_r2_level_region"], out["mp_r2_level"] = float(r2_both), float(r2_level)
        o = d.loc[~d["inner"]]
        res_o, _, _ = s4_basket.residualize(o[["h1", "h2"]], ctrl(o))
        out["mp_split_half_resid_no_inner"] = float(stats.spearman(res_o["h1"], res_o["h2"])[0])
    except _SOFT_ERRORS as e:
        log.warning("Сводка: прирост доли маркетплейсов сверх уровня не посчитан: %s", e)
    return {k: v for k, v in out.items() if np.isfinite(v)}


def part_stability(mo: pd.DataFrame) -> pd.Series:
    """Устойчивость очищенных долей корзины по отдельности: ρ остатков CLR части 2023 и 2024 годов.

    Очистка — как у E4 (``s4_basket.basket_analysis``): МНК CLR года на лог-уровень своего года и дамми
    регионов, выборка ``basket_sample``. Показывает, какую устойчивость даёт любая очищенная ось.
    """
    y0, y1 = YEARS
    sample = s4_basket.basket_sample(mo)
    r1, _, _ = s4_basket.residualize(
        s4_basket.part_frame(sample, "clr", y1), s4_basket.region_controls(sample, y1)
    )
    r0, _, _ = s4_basket.residualize(
        s4_basket.part_frame(sample, "clr", y0), s4_basket.region_controls(sample, y0)
    )
    return pd.Series({p: float(stats.spearman(r1[p], r0[p])[0]) for p in r1.columns}, name="stability")


def share_spread(mo: pd.DataFrame) -> pd.DataFrame:
    """Разброс долей корзины 2024 года: медиана, квартили, отношение квартилей, дисперсия CLR (как F09 у E4)
    и межквартильный размах в п. п. ``iqr_pp``; сортировка — по убыванию размаха в п. п."""
    y1 = YEARS[-1]
    sample = s4_basket.basket_sample(mo)
    summary = s4_basket.basket_summary(
        s4_basket.part_frame(sample, "sh", y1), s4_basket.part_frame(sample, "clr", y1)
    )
    summary["iqr_pp"] = PCT * (summary["q75"] - summary["q25"])
    return summary.sort_values("iqr_pp", ascending=False)


def mp_pair_signal(data: EdaData, rng: np.random.Generator, n_pairs: int) -> dict[str, float]:
    """Сигнал помесячных рядов доли маркетплейсов для рёбер сети (модель E3 на ln доли).

    Ряды МО с полным рядом: ln доли маркетплейсов → снять линейный тренд МО → вычесть общий ритм месяца
    (``s3_rhythm.detrend``, ``own_rhythm``). Корреляции случайных пар МО: сырые ряды (медиана), свой остаток
    (90-й перцентиль), он же на перемешанных месяцах и со сдвигом на месяц (МО a на месяц раньше МО b).
    ``mp_common_share`` — доля общего ритма после снятия тренда.
    """
    ids = s3_rhythm.full_ids(data.mo)
    wide = data.panel_wide
    share = s3_rhythm.series_matrix(wide, ids, "marketplace") - s3_rhythm.series_matrix(wide, ids, "all")
    detrended, _ = s3_rhythm.detrend(share)
    own, _ = s3_rhythm.own_rhythm(detrended)
    pairs = s3_rhythm.pair_correlations({"raw": share, "own": own}, rng, n_pairs)
    pos = pd.Series(np.arange(len(own)), index=own.index)
    a, b = pos.loc[pairs["territory_a"]].to_numpy(), pos.loc[pairs["territory_b"]].to_numpy()
    v = own.to_numpy()
    early, late = _z_rows(v[:, :-1]), _z_rows(v[:, 1:])
    lag = (early[a] * late[b]).mean(axis=1)
    return {
        "mp_pair_raw_q50": float(np.nanquantile(pairs["raw"], 0.5)),
        "mp_pair_own_q90": float(np.nanquantile(pairs["own"], 0.9)),
        "mp_pair_null_q90": float(np.nanquantile(pairs["own_null"], 0.9)),
        "mp_pair_lag_q90": float(np.nanquantile(lag, 0.9)),
        "mp_common_share": s3_rhythm.common_share(detrended, own),
    }


def _z_rows(v: np.ndarray) -> np.ndarray:
    v = v - v.mean(axis=1, keepdims=True)
    sd = v.std(axis=1, keepdims=True)
    return np.divide(v, sd, out=np.full_like(v, np.nan), where=sd > 0)


# --- Внешняя проверка и пример МО ----------------------------------------------------------------

# Внешние переменные контекста: колонка -> (колонка context_annual, год: «c» — год контекста,
# «y» — год отчёта).
EXTERNAL_CONTEXT: dict[str, tuple[str, str]] = {
    "retail_pc": ("retail_pc", "c"),
    "catering_turnover_pc": ("catering_turnover_pc", "c"),
    "employees_to_working_age": ("employees_to_working_age", "c"),
    "ip_per_1000": ("ip_per_1000", "y"),  # число ИП есть только на 1 апреля года отчёта
}
EXTERNAL_MO: tuple[str, ...] = ("emp_sh_B_2023", "point_lat", "market_access")


def external_frame(data: EdaData, cfg: Config | None = None) -> pd.DataFrame:
    """Внешние переменные сюжетов (индекс — ``territory_id``): показатели Росстата из ``context_annual``
    (``EXTERNAL_CONTEXT``), рост фонда оплаты труда года отчёта к году контекста (``payroll_growth``)
    и колонки ``EdaData.mo`` (``EXTERNAL_MO``), плюс флаг ``workplace_based``."""
    years = _years(cfg)
    ca = data.context_annual
    mo = data.mo.set_index("territory_id")
    out = pd.DataFrame(index=mo.index)
    for name, (col, year) in EXTERNAL_CONTEXT.items():
        part = ca.loc[ca["year"] == years[year]].set_index("territory_id")
        out[name] = part[col].astype("float64").reindex(out.index) if col in part else np.nan
    pay = {
        y: ca.loc[ca["year"] == years[y]].set_index("territory_id")["payroll_krub"].astype("float64")
        for y in ("c", "y")
    }
    growth = pay["y"].reindex(out.index) / pay["c"].reindex(out.index) - 1.0
    out["payroll_growth"] = growth.where(np.isfinite(growth))
    for col in EXTERNAL_MO:
        out[col] = mo[col].astype("float64") if col in mo else np.nan
    out["workplace_based"] = mo["workplace_based"].astype("boolean").fillna(False).astype(bool)
    return out


def story_table(data: EdaData, ind: pd.DataFrame) -> pd.DataFrame:
    """МО с колонками ``EdaData.mo``, показателями сводки и производными для примеров МО
    (``log_wage_2023``), названием для текста (``display``); индекс — ``territory_id``."""
    mo = data.mo.set_index("territory_id")
    extra = [c for c in ind.columns if c != "territory_id" and c not in mo.columns]
    table = mo.join(ind.set_index("territory_id")[extra])
    table["log_wage_2023"] = np.log(table["wage_2023"].astype("float64").where(table["wage_2023"] > 0))
    ter = data.territories.set_index("territory_id")
    names = table[["name", "mo_type"]].copy()
    if "name_short" in ter.columns:
        names["name_short"] = ter["name_short"].reindex(table.index)
    table["display"] = display_names(names)
    return table


def external_checks(story: stories.Story, table: pd.DataFrame, ext: pd.DataFrame, weak: float) -> list[dict]:
    """Внешняя проверка сюжета: ρ внутри регионов главного показателя с каждой внешней переменной (МО,
    где показатели по месту работы, не берутся). Строка: колонка, подпись, ожидаемый знак, ρ, n, вывод."""
    if not story.main or story.main not in table.columns:
        return []
    keep = ~ext["workplace_based"].reindex(table.index).fillna(False).astype(bool)
    rows = []
    for e in story.external:
        if e.column not in ext.columns:
            continue
        x = ext[e.column].reindex(table.index)
        rho, n = stats.spearman_within(table.loc[keep, story.main], x[keep], table.loc[keep, "region_code"])
        if not np.isfinite(rho):
            verdict = "не посчитано"
        elif abs(rho) < weak:
            verdict = "связи почти нет"
        else:
            strength = "слабая связь, " if abs(rho) < STRONG_RHO else ""
            verdict = strength + ("знак ожидаемый" if np.sign(rho) == e.sign else "знак обратный")
        rows.append(
            {"column": e.column, "label": e.label, "sign": e.sign, "rho": rho, "n": n, "verdict": verdict}
        )
    return rows


def external_text(rows: Sequence[Mapping[str, Any]]) -> str:
    """«оборот общепита на жителя — ожидается «+», ρ внутри регионов 0,10 (n = 1241), слабая связь, знак
    ожидаемый»; несколько переменных — через «; »."""
    if not rows:
        return style.NA_TEXT
    parts = []
    for r in rows:
        sign = "+" if r["sign"] > 0 else "−"
        parts.append(
            f"{r['label']} — ожидается «{sign}», ρ внутри регионов {style.fmt_rho(r['rho'])} "
            f"(n{NB}={NB}{style.fmt_num(r['n'])}), {r['verdict']}"
        )
    return stories.style_nbsp("; ".join(parts))


# Контроли примера МО в дательном падеже: «ожидаемых по уровню трат и региону».
_CONTROL_WORDS: dict[str, str] = {
    "log_level_2023": "уровню трат",
    "log_level_2024": "уровню трат",
    "log_wage_2023": "зарплате",
}
STRONG_RHO = 0.3  # |ρ| внешней проверки меньше — «слабая связь»
STRONG_SHARE = 0.5  # простое деление названо в тексте, если его η² не меньше половины η² региона


def _shown(value: float, transform: str) -> float:
    if transform == "exp":
        return float(np.exp(value))
    if transform == "expm1":
        return float(np.expm1(value))
    return float(value)


def example_mo(story: stories.Story, table: pd.DataFrame, min_pop_quantile: float) -> dict | None:
    """Пример МО сюжета: наибольшее по модулю отклонение показателя ``story.example.column`` от МНК на
    контроли примера и дамми регионов среди МО с населением не меньше квантили ``min_pop_quantile``.

    Если среди контролей зарплата, МО с показателями по месту работы не берутся. Возвращает название МО,
    регион, значение и ожидаемое по контролям (оба — в шкале показа ``transform``) и текст для отчёта.
    """
    ex = story.example
    if ex is None or ex.column not in table.columns or any(c not in table.columns for c in ex.controls):
        return None
    d = table.dropna(subset=[ex.column, *ex.controls, "region_code"])
    if "log_wage_2023" in ex.controls:
        d = d.loc[~d["workplace_based"].astype("boolean").fillna(False).astype(bool)]
    if len(d) < 10:
        return None
    ctrl = pd.DataFrame({c: d[c].astype("float64") for c in ex.controls}, index=d.index)
    ctrl["region"] = _region_category(d["region_code"])
    try:
        resid, _, _ = s4_basket.residualize(d[[ex.column]].astype("float64"), ctrl)
    except _SOFT_ERRORS as e:
        log.warning("Сводка: пример МО для %s не посчитан: %s", story.sid, e)
        return None
    r = resid[ex.column]
    weight = d["weight"].astype("float64")
    eligible = weight >= weight.quantile(float(min_pop_quantile))
    if not eligible.any():
        return None
    idx = r[eligible].abs().idxmax()
    value = float(d.loc[idx, ex.column])
    shown, expected = _shown(value, ex.transform), _shown(value - float(r[idx]), ex.transform)
    fmt = make_fact("x", shown, ex.kind).text, make_fact("x", expected, ex.kind).text
    words = [*dict.fromkeys(_CONTROL_WORDS.get(c, c) for c in ex.controls), "региону"]
    name = f"{d.loc[idx, 'display']} ({d.loc[idx, 'region_name']})"
    text = f"{name}: {ex.what} {fmt[0]} при ожидаемых по {' и '.join(words)} {fmt[1]}"
    return {"territory_id": int(idx), "name": name, "value": shown, "expected": expected, "text": text}


# --- Быстрый прототип типов (T15) ----------------------------------------------------------------


@dataclass(frozen=True)
class ProtoSet:
    """Признаки быстрого прототипа сюжета: основной набор и, если есть, два независимых замера тех же
    признаков (полугодия или годы) для шумового базиса «типы во времени»."""

    key: str  # код сюжета («s4») или «level» — уровень трат для сравнения
    features: str  # какие признаки, по-русски
    main: pd.DataFrame
    pair: tuple[pd.DataFrame, pd.DataFrame] | None = None
    pair_label: str = ""


def _cleaned_clr(sample: pd.DataFrame, year: int) -> pd.DataFrame:
    res, _, _ = s4_basket.residualize(
        s4_basket.part_frame(sample, "clr", year), s4_basket.region_controls(sample, year)
    )
    return res


def prototype_sets(data: EdaData, table: pd.DataFrame, halves: pd.DataFrame) -> list[ProtoSet]:
    """Признаки быстрого прототипа по сюжетам на МО с полным рядом; «level» — один уровень трат (эталон
    «типов, которые повторяют уровень»)."""
    y0, y1 = YEARS
    full = table.loc[table["series_status"].astype(str) == FULL_STATUS]
    log_dist = np.log1p(full["dist_capital_km"].astype("float64"))
    access = full["market_access"].astype("float64")
    out = [
        ProtoSet(
            "level",
            "уровень трат",
            full[[f"log_level_{y1}"]],
            (full[[f"log_level_{y0}"]], full[[f"log_level_{y1}"]].set_axis([f"log_level_{y0}"], axis=1)),
            f"{y0} и {y1} годы",
        )
    ]
    s1_cols = [c for c in ("summer_excess", "dec_peak", "own_amplitude") if c in full.columns]
    if s1_cols:
        out.append(ProtoSet("s1", "летний избыток, декабрьский пик, размах своего ритма", full[s1_cols]))

    def s2(year: int) -> pd.DataFrame:
        return pd.DataFrame({"level": full[f"log_level_{year}"], "access": access, "dist": log_dist})

    out.append(
        ProtoSet(
            "s2",
            "уровень трат, доступность рынков, расстояние до столицы",
            s2(y1),
            (s2(y0), s2(y1)),
            f"{y0} и {y1} годы",
        )
    )
    h = halves.reindex(full.index)
    out.append(
        ProtoSet(
            "s3",
            "номинальный рост трат",
            full[["growth_log"]],
            (
                np.log1p(h[["growth_h1_all"]]).set_axis(["g"], axis=1),
                np.log1p(h[["growth_h2_all"]]).set_axis(["g"], axis=1),
            ),
            "январь–июнь и июль–декабрь",
        )
    )
    sample = s4_basket.basket_sample(full.reset_index())
    if len(sample) > 10:
        c1 = _cleaned_clr(sample, y1)
        out.append(
            ProtoSet(
                "s4",
                "шесть долей CLR без уровня и региона",
                c1,
                (_cleaned_clr(sample, y0), c1),
                f"{y0} и {y1} годы",
            )
        )
    wide = data.panel_wide
    start = {
        tag: s4_basket.month_share(wide, s4_basket.MP, y0, months).reindex(full.index)
        for tag, months in (("h1", s4_basket.FIRST_HALF), ("h2", s4_basket.SECOND_HALF))
    }
    out.append(
        ProtoSet(
            "s5",
            "прирост доли маркетплейсов и её начальный уровень",
            full[["mp_pp_change", f"sh_{s4_basket.MP}_{y0}"]],
            (
                pd.DataFrame({"pp": h["mp_pp_h1"], "start": start["h1"]}),
                pd.DataFrame({"pp": h["mp_pp_h2"], "start": start["h2"]}),
            ),
            "январь–июнь и июль–декабрь",
        )
    )
    if "log_spend_to_ndfl" in full.columns:
        out.append(
            ProtoSet(
                "s6",
                "траты к доходу 5-НДФЛ, расстояние до столицы",
                pd.DataFrame({"ratio": full["log_spend_to_ndfl"], "dist": log_dist}),
            )
        )
    return out


def _kmeans(x: pd.DataFrame, k: int, n_init: int, seed: int) -> pd.Series:
    """Метки k-means на стандартизованных признаках (строки без пропусков)."""
    from sklearn.cluster import KMeans

    x = x.dropna()
    z = (x - x.mean()) / x.std(ddof=0).replace(0, 1.0)
    labels = KMeans(n_clusters=k, n_init=n_init, random_state=seed).fit_predict(z.to_numpy())
    return pd.Series(labels, index=x.index)


def prototype_table(sets: Sequence[ProtoSet], groups: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """T15: быстрый прототип типов. Для каждого набора признаков — k-means (``prototype.k``), AMI его типов
    с простыми делениями МО (``groups`` — ``groupings``, индекс ``territory_id``) и наибольший из них; ARI
    типов двух независимых замеров (полугодий или лет) — насколько типы повторяются на новых данных."""
    from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

    prm = params(cfg)
    k, n_init, seed = int(prm["prototype"]["k"]), int(prm["prototype"]["n_init"]), int(cfg["seed"])
    rows = []
    for s in sets:
        lab = _kmeans(s.main, k, n_init, seed) if len(s.main.dropna()) > k else pd.Series(dtype="int64")
        row: dict[str, Any] = {"key": s.key, "features": s.features, "n": len(lab), "k": k}
        for g in GROUPINGS:
            ref = groups[g].reindex(lab.index)
            ok = ref.notna()
            row[f"ami_{g}"] = (
                float(adjusted_mutual_info_score(ref[ok].astype(str), lab[ok])) if ok.sum() > k else np.nan
            )
        ami = pd.Series({g: row[f"ami_{g}"] for g in GROUPINGS}).dropna()
        row["ami_max"] = float(ami.max()) if len(ami) else np.nan
        row["ami_max_by"] = GROUPINGS[ami.idxmax()] if len(ami) else style.NA_TEXT
        row["ari_pair"], row["pair_label"] = np.nan, s.pair_label or style.NA_TEXT
        if s.pair is not None:
            both = s.pair[0].join(s.pair[1], lsuffix="_a", rsuffix="_b").dropna()
            if len(both) > k:
                a = _kmeans(s.pair[0].loc[both.index], k, n_init, seed)
                b = _kmeans(s.pair[1].loc[both.index], k, n_init, seed)
                row["ari_pair"] = float(adjusted_rand_score(a, b.loc[a.index]))
        rows.append(row)
    return pd.DataFrame(rows)


def prototype_text(t15: pd.DataFrame, sids: Mapping[str, str]) -> str:
    """Строка подсказки по T15: чьи типы сильнее всего повторяют простое деление, чьи слабее, эталон уровня
    и согласие типов двух замеров."""
    d = t15.loc[t15["key"].isin(list(sids)) & t15["ami_max"].notna()]
    if d.empty:
        return ""
    k = int(t15["k"].iloc[0])
    hi, lo = d.loc[d["ami_max"].idxmax()], d.loc[d["ami_max"].idxmin()]
    text = (
        f"Быстрый прототип (k-means, k{NB}={NB}{k}, таблица T15): сильнее всего простое деление повторяют "
        f"типы {sids[hi['key']]} ({hi['ami_max_by']}, AMI {style.fmt_num(hi['ami_max'], 2)}), слабее всего — "
        f"типы {sids[lo['key']]} (AMI не больше {style.fmt_num(lo['ami_max'], 2)})"
    )
    ref = t15.loc[t15["key"] == "level"]
    if len(ref) and np.isfinite(ref["ami_max"].iloc[0]):
        text += (
            f"; для сравнения, типы по{NB}одному уровню трат совпадают с{NB}группами по{NB}уровню "
            f"на{NB}AMI {style.fmt_num(ref['ami_max'].iloc[0], 2)}"
        )
    pairs = d.loc[d["ari_pair"].notna()]
    if len(pairs):
        listed = ", ".join(f"{sids[r.key]} — {style.fmt_num(r.ari_pair, 2)}" for r in pairs.itertuples())
        text += f". Типы двух независимых замеров совпадают (ARI): {listed}"
    return stories.style_nbsp(text + ".")


# --- График F16 ----------------------------------------------------------------------------------

X_PAD = 0.05  # поле F16 слева от наименьшего значения, в единицах η²
Y_PAD = 0.7  # поле F16 сверху и снизу от крайних строк, в долях шага строк
DIAMOND_MIN_GAP = 0.03  # ромб η² с поправкой на шум рисуется, если он отличается от η² хотя бы на столько


def plot_region_or_place(t13: pd.DataFrame, short: Mapping[str, str], threshold: float):
    """F16: η² региона (чёрные точки), η² с поправкой на шум (серые ромбы) и I Морана (белые кружки) по
    показателям, сверху — самые региональные.

    Серии названы в подзаголовке, легенды нет; пунктир — порог η², правее которого показатель для этапа 2
    лучше брать относительно региона (подписан на поле). У показателей, очищенных от региона, η² не рисуется —
    на его месте подпись «η² = 0 по построению».
    """
    d = t13.dropna(subset=["eta2", "moran_i"], how="all").iloc[::-1].reset_index(drop=True)
    fig, ax = style.new_figure("tall")
    y = np.arange(len(d))
    cols = ["eta2", "moran_i", "eta2_true"]
    lo, hi = d[cols].min(axis=1), d[cols].max(axis=1)
    ax.hlines(y, lo, hi, color=style.CONTEXT, linewidth=1.2, zorder=1)
    # кольцо I Морана крупнее и ниже точки η²: при близких значениях точка видна внутри кольца
    ax.scatter(d["moran_i"], y, s=70, facecolors="white", edgecolors=style.ACCENT, linewidths=1.3, zorder=3)
    ax.scatter(d["eta2"], y, s=34, color=style.ACCENT, zorder=4, linewidths=0)
    diamond = d["eta2_true"].notna() & ((d["eta2_true"] - d["eta2"]).abs() >= DIAMOND_MIN_GAP)
    ax.scatter(
        d.loc[diamond, "eta2_true"],
        y[diamond.to_numpy()],
        s=42,
        marker="D",
        color=style.TEXT2,
        zorder=4,
        linewidths=0,
    )
    built = d["by_construction"].fillna(False).astype(bool).to_numpy()
    for yi, moran in zip(y[built], d.loc[built, "moran_i"], strict=True):
        ax.text(
            max(float(moran) if np.isfinite(moran) else 0.0, 0.0) + 0.03,
            yi,
            "η² = 0 по построению",
            va="center",
            ha="left",
            fontsize=style.POINT_LABEL_PT,
            color=style.TEXT2,
        )
    ax.axvline(threshold, color=style.TEXT2, linewidth=0.8, linestyle="--", zorder=0)
    ax.text(
        threshold + 0.01,
        0.5,
        "правее — брать\nотносительно региона",
        va="center",
        ha="left",
        fontsize=style.POINT_LABEL_PT,
        color=style.TEXT2,
    )
    ax.set_yticks(y, [short.get(i, lab) for i, lab in zip(d["indicator"], d["label"], strict=True)])
    ax.tick_params(axis="y", length=0)
    values = pd.concat([d[c] for c in cols]).dropna()
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


def _indicator_facts(ctx: SectionContext, t13: pd.DataFrame, checks: pd.DataFrame | None = None) -> None:
    """Факты T13: η², I Морана и p по каждому показателю, с внутригородскими территориями и без них; η² других
    делений, надёжность и η² с поправкой на шум.

    Сверки определений (``definition_checks``) — факты ``syn.check_*``; в пояснениях фактов сверяемого
    показателя названо, где искать сверку. У показателя, очищенного от региона, η² — пропуск
    (по построению 0).
    """
    see: dict[str, str] = {}
    for row in (checks if checks is not None else pd.DataFrame()).itertuples(index=False):
        col = str(row.indicator)
        if not _FACT_KEY.fullmatch(col):
            continue
        note = f"сверка к {row.checks}: {row.why}; n = {row.n}"
        ctx.fact(f"check_eta2_{col}", row.eta2, "num2", f"η² региона, {note}")
        ctx.fact(f"check_moran_{col}", row.moran_i, "num2", f"I Морана, {note}")
        ctx.fact(
            f"check_moran_{col}_no_inner",
            row.moran_i_no_inner,
            "num2",
            f"I Морана без внутригородских, {note}",
        )
        see[str(row.checks)] = (
            f"; другое определение ({row.why}) — факты syn.check_eta2_{col}, syn.check_moran_{col}"
        )
    for row in t13.itertuples(index=False):
        key = str(row.indicator)
        if not _FACT_KEY.fullmatch(key):
            log.warning("Сводка: имя показателя %s не годится для ключа факта — факты не записаны", key)
            continue
        note = f"{row.label}; n = {row.n}{see.get(key, '')}"
        built = "; показатель очищен от региона — η² равен 0 по построению" if row.by_construction else ""
        ctx.fact(f"eta2_{key}", row.eta2, "num2", f"η² региона: {note}{built}")
        ctx.fact(f"eta2_{key}_no_inner", row.eta2_no_inner, "num2", f"η² без внутригородских: {note}{built}")
        ctx.fact(f"moran_{key}", row.moran_i, "num2", f"I Морана, основной k: {note}")
        ctx.fact(f"moran_{key}_p", row.moran_p, "p", f"перестановочное p I Морана: {note}")
        ctx.fact(
            f"moran_{key}_no_inner", row.moran_i_no_inner, "num2", f"I Морана без внутригородских: {note}"
        )
        for g, label in GROUPINGS.items():
            if g != "region":
                ctx.fact(
                    f"eta2_{g}_{key}", getattr(row, f"eta2_{g}"), "num2", f"η² деления «{label}»: {note}"
                )
        ctx.fact(
            f"reliability_{key}",
            row.reliability,
            "num2",
            f"надёжность (полугодия по Спирмену — Брауну или ρ соседних лет): {note}",
        )
        ctx.fact(f"eta2_true_{key}", row.eta2_true, "num2", f"η² с поправкой на шум, η² / надёжность: {note}")


def _pct_int(share: float) -> int:
    return int(round(PCT * float(share)))


def _story_facts(
    ctx: SectionContext,
    matrix: pd.DataFrame,
    weights: Mapping[str, float],
    other: pd.DataFrame | None = None,
    extra_advice: Sequence[str] = (),
) -> None:
    """Факты матрицы: веса (в процентах итога), итог, роль, критерии отказа, обоснования баллов по каждому
    сюжету; то же без внутригородских территорий (``other``) и фраза о чувствительности."""
    for crit, w in weights.items():
        ctx.fact(
            f"weight_{crit}",
            _pct_int(w),
            "int",
            f"вес критерия «{stories.CRITERIA[crit]}», % итога (eda.matrix_weights)",
        )
    w_facts = sum(float(weights[c]) for c in stories.FACT_CRITERIA)
    ctx.fact(
        "weight_facts", _pct_int(w_facts), "int", "% итога, посчитанный из фактов: сила сигнала и покрытие"
    )
    ctx.fact("weight_expert", _pct_int(1.0 - w_facts), "int", "% итога из экспертных баллов (stories.py)")
    for row in matrix.itertuples(index=False):
        k = row.key
        ctx.fact(f"title_{k}", f"{row.sid} «{row.title}»", "str", row.gist)
        ctx.fact(f"gist_{k}", row.gist, "str", f"{row.sid}: суть сюжета одной фразой")
        ctx.fact(f"score_{k}", row.score, "num2", f"{row.sid}: Σ вес × балл (1–5)")
        ctx.fact(f"score_facts_{k}", row.score_facts, "num2", f"{row.sid}: балл по фактам (сигнал, покрытие)")
        ctx.fact(
            f"score_expert_{k}", row.score_expert, "num2", f"{row.sid}: экспертный балл (четыре критерия)"
        )
        ctx.fact(f"status_{k}", row.status, "str", f"{row.sid}: роль по критериям отказа")
        ctx.fact(f"role_{k}", row.role, "str", f"{row.sid}: роль и пометка «критерий на границе»")
        ctx.fact(f"reject_{k}", row.criteria, "str", f"{row.sid}: «проходит» или невыполненные критерии")
        ctx.fact(f"criteria_{k}", row.criteria_all, "str", f"{row.sid}: все критерии отказа с отметками")
        sig = None if pd.isna(row.signal) else int(row.signal)
        cov = None if pd.isna(row.coverage) else int(row.coverage)
        ctx.fact(f"signal_{k}", sig, "int", f"{row.sid}: балл силы сигнала")
        ctx.fact(f"signal_note_{k}", row.signal_note, "str", f"{row.sid}: из чего балл силы сигнала")
        ctx.fact(f"coverage_{k}", cov, "int", f"{row.sid}: балл покрытия")
        ctx.fact(f"coverage_note_{k}", row.coverage_note, "str", f"{row.sid}: из чего балл покрытия")
        ctx.fact(f"manual_note_{k}", row.manual_note, "str", f"{row.sid}: экспертные баллы (stories.py)")
    if other is not None:
        for row in other.itertuples(index=False):
            note = (
                f"{row.sid} без внутригородских территорий (критерии и сигнал — без них, остальное — то же)"
            )
            ctx.fact(f"score_{row.key}_no_inner", row.score, "num2", f"{note}: итог")
            ctx.fact(f"status_{row.key}_no_inner", row.status, "str", f"{note}: роль")
        best = stories.leader(other)
        ctx.fact(
            "top_story_no_inner",
            stories.NO_STORY if best is None else best["sid"],
            "str",
            "без внутригородских: главный кандидат или, если критерии не прошёл никто, лучший по итогу",
        )
        n_inner = ctx.facts.get(f"{SECTION_ID}.n_inner")
        ctx.fact(
            "sensitivity",
            stories.sensitivity_text(matrix, other, None if n_inner is None else int(n_inner.value)),
            "str",
            "меняет ли выбор сюжета исключение внутригородских территорий Москвы и Петербурга",
        )
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
    info = stories.lead(matrix, weights)
    ctx.fact(
        "lead_margin",
        None if info is None else info.margin,
        "num2",
        "итог лидера минус итог следующего по баллу",
    )
    ctx.fact(
        "lead_flips",
        None if info is None else len(info.flips),
        "int",
        "экспертных критериев, сдвиг балла в которых на 1 меняет порядок двух первых сюжетов",
    )
    ctx.fact(
        "lead_note",
        stories.lead_text(matrix, weights),
        "str",
        "разрыв двух первых сюжетов и меняет ли его один экспертный балл",
    )
    best = stories.facts_leader(matrix)
    ctx.fact(
        "facts_leader",
        stories.NO_STORY if best is None else f"{best['sid']} «{best['title']}»",
        "str",
        "сюжет с наибольшим баллом по фактам (сила сигнала и покрытие)",
    )
    ctx.fact(
        "advice",
        stories.advice_text(matrix, weights, extra_advice),
        "str",
        "подсказка: критерии, устойчивость порядка к экспертным баллам, лидер по фактам, прототип",
    )


@dataclass(frozen=True)
class RegionPlace:
    """Что показывает T13 для заголовка F16 и текста: уровень — регион, сколько воспроизводимого разброса
    показателей динамики даёт регион."""

    level: str  # ключ показателя уровня
    level_eta2: float
    level_regional: bool  # η² уровня не ниже eda.synthesis.eta2_regional
    dynamic: dict[str, float]  # η² показателей динамики без поправки (NaN — не посчитан)
    corrected: dict[str, float]  # η² с поправкой на шум (NaN — надёжность не оценена)
    classes: dict[str, str]  # показатель -> CLASS_PLACE | CLASS_HALF | CLASS_REGION
    gap_ok: bool  # η² уровня выше η² хотя бы одного показателя динамики на headline.gap_min и больше

    @property
    def place(self) -> tuple[str, ...]:
        """Показатели динамики, у которых регион даёт меньше половины воспроизводимого разброса."""
        return tuple(k for k, c in self.classes.items() if c == CLASS_PLACE)


def noise_class(value: float, prm: Mapping[str, Any]) -> str:
    """Класс по η² (с поправкой на шум, если есть): меньше ``place_eta2_max − noise_tol`` — «место», больше
    ``place_eta2_max + noise_tol`` — «регион», между — «наполовину»."""
    mid, tol = float(prm["place_eta2_max"]), float(prm["noise_tol"])
    if value < mid - tol:
        return CLASS_PLACE
    if value > mid + tol:
        return CLASS_REGION
    return CLASS_HALF


def classify(t13: pd.DataFrame, prm: Mapping[str, Any]) -> RegionPlace:
    """Уровень — «свойство региона», если η² ≥ ``eta2_regional``; показатели динамики — по классам
    ``noise_class`` их η² с поправкой на шум (без оценки надёжности — по η² как есть)."""
    h = prm["headline"]
    t = t13.set_index("indicator")
    eta = t["eta2"]
    true = t["eta2_true"] if "eta2_true" in t else pd.Series(np.nan, index=t.index)
    level = float(eta.get(h["level"], np.nan))
    dynamic = {k: float(eta.get(k, np.nan)) for k in h["dynamic"]}
    corrected = {k: float(true.get(k, np.nan)) for k in h["dynamic"]}
    classes = {}
    for k, raw in dynamic.items():
        v = corrected[k] if np.isfinite(corrected[k]) else raw
        if np.isfinite(v):
            classes[k] = noise_class(v, prm)
    gap = float(h["gap_min"])
    gap_ok = bool(
        np.isfinite(level)
        and any(np.isfinite(v) and level - v >= gap for k, v in dynamic.items() if k in classes)
    )
    regional = bool(np.isfinite(level) and level >= float(prm["eta2_regional"]))
    return RegionPlace(h["level"], level, regional, dynamic, corrected, classes, gap_ok)


def headline_text(rp: RegionPlace, words: Mapping[str, str], max_len: int = 90) -> str:
    """Заголовок F16: «Регион задаёт уровень трат (η² 0,77), а рост и прирост доли маркетплейсов —
    наполовину».

    Показатели динамики сгруппированы по классам ``noise_class``; слова — ``TITLE_WORDS``, для других
    показателей — ``words`` (``headline.dynamic``). Длиннее ``max_len`` — короче: без перечня показателей.
    """
    eta = style.fmt_num(rp.level_eta2, 2)
    head = (
        f"Регион задаёт уровень трат (η²{NB}{eta})"
        if rp.level_regional
        else (f"Регион задаёт уровень трат лишь отчасти (η²{NB}{eta})")
    )
    phrases = []
    for cls in (CLASS_HALF, CLASS_PLACE, CLASS_REGION):
        names = [TITLE_WORDS.get(k, words.get(k, k)) for k, c in rp.classes.items() if c == cls]
        if names:
            phrases.append(f"{' и '.join(names)} — {CLASS_WORDS[cls]}")
    candidates = [f"{head}, а " + "; ".join(phrases)] if phrases else []
    if len({c for c in rp.classes.values()}) == 1:
        candidates.append(f"{head}, а динамику трат — {CLASS_WORDS[next(iter(rp.classes.values()))]}")
    candidates.append(head)
    for text in candidates:
        text = stories.style_nbsp(text)
        if len(text) <= max_len:
            return text
    return candidates[-1]


def _headline(ctx: SectionContext, prm: Mapping[str, Any], rp: RegionPlace) -> tuple[str, str]:
    """Заголовок F16 с проверкой: уровень — регион, η² уровня выше η² динамики на ``gap_min`` (без поправки
    на шум, как в В.5), у каждого показателя динамики посчитан класс."""
    h = prm["headline"]
    ok = rp.level_regional and rp.gap_ok
    dyn_text = ", ".join(
        f"eta2_{k} = {style.fmt_num(v, 2)} (с поправкой {style.fmt_num(rp.corrected.get(k, np.nan), 2)})"
        for k, v in rp.dynamic.items()
    )
    gap = style.fmt_num(float(h["gap_min"]), 1)
    regional = style.fmt_num(float(prm["eta2_regional"]), 1)
    detail = (
        f"F16: eta2_{rp.level} = {style.fmt_num(rp.level_eta2, 2)} (нужно ≥ {regional}), {dyn_text}; "
        f"η² уровня "
        f"выше η² хотя бы одного показателя динамики на {gap} и больше"
    )
    check = f"eta2_{rp.level} ≥ {regional}; eta2_{rp.level} − eta2 динамики ≥ {gap}; классы по eta2_true"
    title_max = int(ctx.cfg["eda"]["figure_limits"]["title_max"])
    return ctx.headline(headline_text(rp, h["dynamic"], title_max), ok, detail), check


def figure_alt(t13: pd.DataFrame, prm: Mapping[str, Any], short: Mapping[str, str]) -> str:
    """Альт-текст F16 из чисел T13: η² уровня и показателей динамики (с поправкой на шум), число
    региональных показателей."""
    t = t13.set_index("indicator")
    keys = [prm["headline"]["level"], *prm["headline"]["dynamic"]]
    parts = []
    for k in keys:
        if k not in t.index:
            continue
        item = f"{_lower_first(short.get(k, k))} — {style.fmt_num(t.loc[k, 'eta2'], 2)}"
        if np.isfinite(t.loc[k, "eta2_true"]) and k != keys[0]:
            item += f" (с поправкой на шум {style.fmt_num(t.loc[k, 'eta2_true'], 2)})"
        parts.append(item)
    n = int(t13["eta2"].notna().sum())
    n_reg = int(t13["regional"].sum())
    thr = style.fmt_num(float(prm["eta2_regional"]), 1)
    return (
        f"Чёрные точки — η² региона, серые ромбы — η² с поправкой на шум, белые кружки — I Морана по "
        f"{len(t13)} показателям МО. η²: " + "; ".join(parts) + ". "
        f"Выше {thr} — {n_reg} из {n} показателей: их лучше брать относительно региона"
    )


def summary_md(rp: RegionPlace, short: Mapping[str, str], t13: pd.DataFrame) -> str:
    """«Что видно» и «Что это значит для сюжета» раздела «Регион или место»: числа — только ``{{syn.ключ}}``.

    Утверждения выбираются по ``rp`` и T13 (те же, что проверяет заголовок F16): «уровень почти целиком
    региональный» — только при η² уровня не ниже порога; доля региона в воспроизводимом разбросе динамики —
    по классам ``noise_class``; «место — это город или село, крупное или малое МО» — только если η² какого-то
    из этих делений у показателя динамики не меньше половины η² региона.
    """
    lvl = rp.level
    t = t13.set_index("indicator")

    def f(key: str) -> str:
        return "{{" + f"{SECTION_ID}.{key}" + "}}"

    def name(k: str) -> str:
        return _lower_first(short.get(k, k))

    seen = [
        f"- **Как считали.** Для каждого показателя МО посчитаны η² региона{NB}— доля разброса между "
        f"регионами в{NB}общем разбросе{NB}— и{NB}I Морана: насколько значение МО похоже на{NB}значения "
        f"{f('knn_main')} ближайших соседей, чем больше, тем сильнее сходство (таблица T13, рис.{NB}16). "
        f"На{NB}чистом шуме при{NB}{f('n_regions')} регионах η² около {f('eta2_random')}."
    ]
    if np.isfinite(rp.level_eta2):
        lead = "Уровень трат почти целиком региональный" if rp.level_regional else "У уровня трат"
        seen.append(f"- **{lead}:** η² {f('eta2_' + lvl)}, I Морана {f('moran_' + lvl)}.")
    dyn = [k for k in rp.classes if k in t.index]
    if dyn:
        raw = " и ".join(f"{f('eta2_' + k)} у{NB}показателя «{name(k)}»" for k in dyn)
        text = f"- **Динамика трат.** η² ниже: {raw}."
        rel = [k for k in dyn if np.isfinite(rp.corrected.get(k, np.nan))]
        if rel:
            listed = " и ".join(f"{f('reliability_' + k)}" for k in rel)
            true = " и ".join(f"{f('eta2_true_' + k)}" for k in rel)
            classes = {rp.classes[k] for k in rel}
            share = CLASS_SHARE[classes.pop()] if len(classes) == 1 else "разную долю"
            text += (
                f" Но и{NB}шума в{NB}них больше: надёжность по{NB}полугодиям{NB}— {listed}. С{NB}поправкой "
                f"на{NB}шум (η² / надёжность) на{NB}регион приходится {true} воспроизводимого разброса, "
                f"то есть {share}."
            )
        seen.append(text)
        other, named = [], []
        for k in dyn:
            eta = t.loc[k, "eta2"]
            if not np.isfinite(eta):
                continue
            close = [
                g
                for g in GROUPINGS
                if g not in ("region", "fd") and t.loc[k, f"eta2_{g}"] >= STRONG_SHARE * eta
            ]
            close.sort(key=lambda g: -t.loc[k, f"eta2_{g}"])
            if close:
                items = ", ".join(f"{GROUPINGS[g]}{NB}— {f(f'eta2_{g}_{k}')}" for g in close)
                other.append(f"у{NB}показателя «{name(k)}»: {items} (регион{NB}— {f('eta2_' + k)})")
                named += [g for g in close if g not in named]
        if other:
            seen.append(
                f"- **Кроме региона.** Сопоставимую долю разброса дают и{NB}простые деления МО: "
                + "; ".join(other)
                + f". «Место» здесь во{NB}многом означает {_join_labels([GROUPINGS_ACC[g] for g in named])}."
            )
    seen.append(
        f"- **Порог.** η² больше {f('eta2_regional_min')}{NB}— у{NB}{f('n_regional')} из{NB}"
        f"{f('n_indicators')} показателей: {f('regional_indicators')}."
    )
    built = [k for k in t.index if bool(t.loc[k, "by_construction"])]
    if built:
        items = ", ".join(f"«{name(k)}» (I Морана {f('moran_' + k)})" for k in built)
        seen.append(
            f"- **Показатели без региона.** У{NB}показателя {items} η² равен нулю по{NB}построению: регион "
            f"из{NB}него уже вычтен, поэтому в{NB}счёт показателей он не входит; I Морана показывает, "
            f"что соседи похожи и{NB}после вычета."
        )
    inner = f"- **Москва и{NB}Петербург.** Без {f('n_inner')} внутригородских территорий"
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
        means.append("- Типы МО по уровню трат повторят карту регионов.")
    means.append(
        "- Для этапа 2 показатели выше порога η² лучше брать относительно региона (отношение к медиане "
        "региона или ранг внутри региона), остальные можно брать как есть."
    )
    if dyn:
        means.append(
            "- У показателей динамики низкий η² во многом от шума: их лучше считать по году, а не "
            "по полугодию, и сравнивать с надёжностью, прежде чем называть «свойством места»."
        )
    means.append(
        f"- Москва и{NB}Петербург дают {f('n_inner')} мелких соседних узлов, которые могут стать отдельным "
        f"«кластером столиц». Оставить их отдельными узлами, свернуть в{NB}два узла с{NB}весами населения "
        f"или исключить с{NB}оговоркой{NB}— решается до этапа 2 по{NB}T13; от этого зависит "
        f"и{NB}выбор сюжета (раздел «Какой сюжет выбрать»)."
    )
    means.append(
        f"- Сюжет, чьи признаки почти целиком региональные, даст типы, похожие на{NB}карту регионов: это "
        f"проверяет быстрый прототип типов (таблица T15)."
    )
    return (
        "**Что видно.**\n\n" + "\n".join(seen) + "\n\n**Что это значит для сюжета.**\n\n" + "\n".join(means)
    )


CAVEATS: list[str] = [
    f"η² и{NB}I Морана описывают связь показателя с{NB}географией, а{NB}не{NB}причину.",
    f"I Морана считается по{NB}ближайшим соседям по{NB}точкам внутри полигонов МО, а{NB}не{NB}по{NB}дорогам; "
    f"внутригородские территории мелкие и{NB}соседствуют друг с{NB}другом, поэтому с{NB}ними I выше.",
    f"Поправка на шум грубая: надёжность оценена по{NB}двум полугодиям или соседним годам, "
    f"а{NB}η² / надёжность считает, что шум между регионами не различается.",
    f"Экспертные баллы матрицы задал автор (`src/munnet/eda/stories.py`), баллы сигнала и{NB}покрытия "
    f"посчитаны из фактов; участник вправе изменить и{NB}то, и{NB}другое.",
    f"Быстрый прототип типов (T15){NB}— один запуск k-means без выбора числа типов и{NB}признаков: "
    f"он показывает, что типы повторяют, а{NB}не{NB}какими они будут на{NB}этапе 3.",
]

T13_LABELS: dict[str, str] = {
    "indicator": "Код",
    "label": "Показатель",
    "source": "Раздел",
    "n": "МО",
    "eta2": "η² региона",
    "reliability": "Надёжность",
    "eta2_true": "η² без шума",
    "eta2_fd": "η² округа",
    "eta2_type": "η² типа МО",
    "eta2_size": "η² размера",
    "eta2_urban": "η² доли горожан",
    "moran_i": "I Морана",
    "moran_p": "p",
    "eta2_no_inner": "η² без внутригородских",
    "moran_i_no_inner": "I без внутригородских",
    "regional": "Брать относительно региона",
}
# Колонки T13 в отчёте (в CSV — все): остальные только в CSV, чтобы таблица читалась на ширине страницы.
T13_MD_COLUMNS: tuple[str, ...] = (
    "label",
    "n",
    "eta2",
    "reliability",
    "eta2_true",
    "eta2_fd",
    "eta2_type",
    "eta2_size",
    "eta2_urban",
    "moran_i",
    "eta2_no_inner",
    "moran_i_no_inner",
    "regional",
)
BUILT_TEXT = "— (0 по построению)"
# Колонки T14 в отчёте (в CSV — вся матрица с критериями и обоснованиями): критерии отказа целиком идут
# текстом под таблицей, чтобы таблица читалась на ширине страницы.
T14_COLUMNS: tuple[str, ...] = (
    "sid",
    "title",
    *stories.FACT_CRITERIA,
    *stories.MANUAL_CRITERIA,
    "score_facts",
    "score_expert",
    "score",
    "role",
)
T14_LABELS: dict[str, str] = {
    "sid": "№",
    "title": "Сюжет",
    "signal": "Сигнал",
    "coverage": "Покрытие",
    "network_dynamics": "Сеть",
    "practical": "Ценность",
    "explainability": "Объяснимость",
    "data_risk": "Риски",
    "score_facts": "По фактам",
    "score_expert": "Экспертный",
    "score": "Итог",
    "role": "Роль",
}
T15_COLUMNS: tuple[str, ...] = (
    "key",
    "features",
    "n",
    "k",
    *(f"ami_{g}" for g in GROUPINGS),
    "ami_max",
    "ami_max_by",
    "ari_pair",
    "pair_label",
)
T15_LABELS: dict[str, str] = {
    "story": "Сюжет",
    "features": "Признаки",
    "n": "МО",
    **{f"ami_{g}": f"AMI: {label}" for g, label in GROUPINGS.items()},
    "ami_max": "Наибольший AMI",
    "ami_max_by": "С каким делением",
    "ari_pair": "ARI двух замеров",
    "pair_label": "Замеры",
}


def t14_labels(weights: Mapping[str, float]) -> dict[str, str]:
    """Заголовки T14 с весами: «Сигнал ×0,25», «По фактам ×0,40»."""
    w = stories.check_weights(weights)
    w_facts = sum(w[c] for c in stories.FACT_CRITERIA)
    extra = {**w, "score_facts": w_facts, "score_expert": 1.0 - w_facts}
    return {k: f"{v}{NB}×{style.fmt_num(extra[k], 2)}" if k in extra else v for k, v in T14_LABELS.items()}


def _t13_markdown(t13: pd.DataFrame, short: Mapping[str, str] | None = None) -> str:
    """Markdown T13: колонки ``T13_MD_COLUMNS``, подписи — короткие (``short``); η² показателя, очищенного
    от региона, — «— (0 по построению)»."""

    def num(x: Any) -> str:
        return style.fmt_num(x, 2)

    shown = t13.copy()
    if short:
        shown["label"] = [
            short.get(i, lab) for i, lab in zip(shown["indicator"], shown["label"], strict=True)
        ]
    for c in ("eta2", "eta2_no_inner", "eta2_fd", "eta2_true"):
        shown[c] = [
            BUILT_TEXT if built and c in ("eta2", "eta2_fd") else (style.NA_TEXT if built else num(v))
            for v, built in zip(shown[c], shown["by_construction"], strict=True)
        ]
    formats = {
        c: num for c in ("reliability", "eta2_type", "eta2_size", "eta2_urban", "moran_i", "moran_i_no_inner")
    }
    return to_markdown(shown[list(T13_MD_COLUMNS)], formats, T13_LABELS)


def _check_facts(
    ctx: SectionContext, values: Mapping[str, float], notes: Mapping[str, tuple[str, str]]
) -> None:
    """Факты проверок сюжетов: ключ -> (вид, пояснение); нет значения — пропуск."""
    for key, (kind, note) in notes.items():
        ctx.fact(key, values.get(key), kind, note)


NO_INNER_NOTE = "без 247 внутригородских территорий Москвы и Петербурга, та же функция раздела"
CHECK_NOTES: dict[str, tuple[str, str]] = {
    "reliable_share_no_inner": ("pct", f"доля МО с устойчивым своим ритмом (E3), {NO_INNER_NOTE}"),
    "growth_half_consistency_no_inner": ("rho", f"ρ роста по полугодиям (E4), {NO_INNER_NOTE}"),
    "residual_share_no_inner": (
        "pct",
        f"доля дисперсии CLR после очистки от уровня и региона (E4), {NO_INNER_NOTE}",
    ),
    "residual_pc1_stability_no_inner": (
        "rho",
        f"устойчивость очищенной PC1 2023 → 2024 (E4), {NO_INNER_NOTE}",
    ),
    "mp_split_half_no_inner": ("rho", f"ρ прироста доли маркетплейсов по полугодиям (E4), {NO_INNER_NOTE}"),
    "rho_mp_pp_level_within_no_inner": (
        "rho",
        f"ρ прироста доли маркетплейсов с уровнем трат внутри регионов (E4), {NO_INNER_NOTE}",
    ),
    "mp_split_half_resid": (
        "rho",
        "ρ прироста доли маркетплейсов за январь–июнь и июль–декабрь после МНК каждого на лог-уровень "
        "2023 года и дамми регионов (выборка e4.mp_split_half)",
    ),
    "mp_split_half_resid_no_inner": ("rho", f"то же, {NO_INNER_NOTE}"),
    "mp_r2_level": (
        "pct",
        "доля дисперсии годового прироста доли маркетплейсов, объяснённая лог-уровнем 2023 года",
    ),
    "mp_r2_level_region": (
        "pct",
        "доля дисперсии годового прироста доли маркетплейсов, объяснённая лог-уровнем 2023 года и регионом",
    ),
    "mp_pair_raw_q50": ("num2", "медиана корреляции сырых помесячных рядов ln доли маркетплейсов пар МО"),
    "mp_pair_own_q90": ("num2", "90-й перцентиль корреляции своего остатка доли маркетплейсов пар МО"),
    "mp_pair_null_q90": ("num2", "то же на перемешанных месяцах"),
    "mp_pair_lag_q90": ("num2", "то же со сдвигом на месяц: МО a на месяц раньше МО b"),
    "mp_common_share": ("pct", "доля общего ритма в ряду ln доли маркетплейсов после снятия тренда МО"),
}


def run_synthesis(ctx: SectionContext, findings: list[Finding]) -> Finding:
    """Сводка: показатели МО, T13 и F16 «регион или место», T14 — матрица решения, T15 — прототип типов,
    проверки сюжетов, факты ``syn.*``."""
    cfg, data = ctx.cfg, ctx.data
    prm = params(cfg)
    eda = cfg["eda"]
    _, short = mo_labels(cfg)
    short = {**short, **SECTION_SHORT_LABELS}
    section_facts = _facts_of(findings, ctx)

    ind, labels = collect_indicators(data, findings, cfg)
    sources = indicator_sources(ind, findings)
    rel = reliabilities(data.mo, section_facts, cfg)
    t13 = region_or_place(ind, data.mo, cfg, ctx.rng, labels=labels, sources=sources, reliability=rel)
    log.info("Сводка: %d показателей МО, %d в T13", len(ind.columns) - 1, len(t13))
    checks = definition_checks(data.mo, cfg, ctx.rng)  # после T13: p-значения T13 от сверки не зависят

    # Факты: выборка и параметры
    mo = data.mo
    ctx.fact("n_mo", len(mo), "int", "МО панели (знаменатель покрытия сюжетов)")
    ctx.fact("n_inner", int(_inner(mo).sum()), "int", "внутригородских территорий Москвы и Петербурга")
    counted = t13.loc[t13["eta2"].notna()]
    ctx.fact(
        "n_indicators", len(counted), "int", "показателей МО с η² в T13 (без флагов и очищенных от региона)"
    )
    ctx.fact("knn_main", int(eda["knn_main"]), "int", "соседей в весах I Морана (eda.knn_main)")
    ctx.fact("permutations", int(eda["permutations"]), "int", "перестановок в тесте I Морана")
    thr = float(prm["eta2_regional"])
    ctx.fact("eta2_regional_min", thr, "num1", "порог η²: показатель брать относительно региона")
    ctx.fact("n_regions", int(mo["region_code"].nunique()), "int", "регионов панели")
    at_level = t13.loc[t13["indicator"] == prm["headline"]["level"], "eta2_random"]
    ctx.fact(
        "eta2_random",
        float(at_level.iloc[0]) if len(at_level) else float(t13["eta2_random"].median()),
        "num2",
        "η² чистого шума при том же числе регионов: (k − 1) / (n − 1), n и k — у показателя уровня трат",
    )
    ctx.fact(
        "n_workplace_based", int(mo["workplace_based"].sum()), "int", "МО с показателями по месту работы"
    )
    ctx.fact(
        "n_ndfl_ok", int(mo["ndfl_ok"].sum()), "int", "МО с пригодным доходом 5-НДФЛ (флаг ndfl_ok, 2023)"
    )
    _indicator_facts(ctx, t13, checks)
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

    # Проверки сюжетов: без внутригородских, прирост доли сверх уровня, рёбра по доле, устойчивость частей
    halves = s4_basket.half_year_growth(data.panel_wide)
    values = {**no_inner_values(data, findings, halves), **mp_residual(mo, halves)}
    try:
        values.update(mp_pair_signal(data, ctx.rng, int(eda["pair_sample"])))
    except _SOFT_ERRORS as e:
        log.warning("Сводка: ряды доли маркетплейсов не посчитаны: %s", e)
    _check_facts(ctx, values, CHECK_NOTES)
    try:
        parts = part_stability(mo)
    except _SOFT_ERRORS as e:
        log.warning("Сводка: устойчивость очищенных долей не посчитана: %s", e)
        parts = pd.Series(dtype="float64")
    span = (
        style.fmt_range(parts.min(), parts.max(), style.fmt_num, decimals=2) if parts.notna().any() else None
    )
    part_note = "ρ 2023 → 2024 остатка CLR каждой доли после очистки от уровня и региона (как у E4)"
    ctx.fact("resid_part_stability_span", span, "str", f"{part_note}: от наименьшего до наибольшего")
    ctx.fact(
        "resid_part_stability_min", parts.min() if len(parts) else None, "rho", f"{part_note}: наименьший"
    )
    ctx.fact(
        "resid_part_stability_max", parts.max() if len(parts) else None, "rho", f"{part_note}: наибольший"
    )
    try:
        spread = share_spread(mo)
    except _SOFT_ERRORS as e:
        log.warning("Сводка: разброс долей не посчитан: %s", e)
        spread = pd.DataFrame(columns=["label", "iqr_pp"])
    for part in s4_basket.PARTS:
        label = style.LABELS.get(part, part)
        value = spread["iqr_pp"].get(part) if len(spread) else None
        ctx.fact(f"share_iqr_pp_{part}", value, "num1", f"межквартильный размах доли «{label}», 2024, п. п.")
    top_part = spread.index[0] if len(spread) else None
    note = "часть корзины с наибольшим межквартильным размахом доли в п. п."
    ctx.fact("share_iqr_argmax", top_part, "str", note)
    ctx.fact(
        "share_iqr_argmax_label",
        None if top_part is None else _lower_first(style.LABELS.get(top_part, top_part)),
        "str",
        f"{note}, по-русски",
    )
    ctx.fact(
        "share_iqr_argmax_gen",
        None if top_part is None else PART_GENITIVE.get(top_part, top_part),
        "str",
        f"{note}, в родительном падеже",
    )
    ctx.fact(
        "share_iqr_pp_max",
        None if top_part is None else spread["iqr_pp"].iloc[0],
        "num1",
        "наибольший межквартильный размах доли, п. п.",
    )

    # Внешняя проверка, пример МО, прототип
    stories_all = stories.build_stories(stories.params(cfg))
    table = story_table(data, ind)
    ext = external_frame(data, cfg)
    sids = {s.key: s.sid for s in stories_all}
    for s in stories_all:
        rows = external_checks(s, table, ext, float(prm["external_weak"]))
        main = _lower_first(short.get(s.main, s.main)) if s.main else None
        ctx.fact(f"main_label_{s.key}", main, "str", f"{s.sid}: главный показатель для внешней проверки")
        ctx.fact(
            f"external_{s.key}", external_text(rows), "str", f"{s.sid}: внешняя проверка, ρ внутри регионов"
        )
        ex = example_mo(s, table, float(prm["example_min_pop_quantile"]))
        ctx.fact(
            f"example_{s.key}",
            None if ex is None else stories.style_nbsp(ex["text"]),
            "str",
            f"{s.sid}: пример МО — наибольшее отклонение от того, что дают контроли и регион",
        )
        ctx.fact(f"user_{s.key}", s.user or None, "str", f"{s.sid}: кому полезен и для какого решения")
    try:
        t15 = prototype_table(
            prototype_sets(data, table, halves), groupings(mo, cfg).set_axis(mo["territory_id"]), cfg
        )
    except _SOFT_ERRORS as e:
        log.warning("Сводка: прототип типов не посчитан: %s", e)
        t15 = pd.DataFrame(columns=T15_COLUMNS)
    rows15 = {row.key: row for row in t15.itertuples(index=False)}
    for key in ["level", *sids]:
        row = rows15.get(key)
        who = sids.get(key, "уровень трат")
        ami = None if row is None else row.ami_max
        ctx.fact(f"proto_ami_max_{key}", ami, "num2", f"прототип {who}: наибольший AMI с простым делением")
        by = None if row is None else row.ami_max_by
        ctx.fact(f"proto_ami_max_by_{key}", by, "str", f"прототип {who}: с каким делением")
        ari = None if row is None else row.ari_pair
        ctx.fact(f"proto_ari_{key}", ari, "num2", f"прототип {who}: ARI типов двух замеров")
        if key not in sids:
            continue
        note = None
        if row is not None and np.isfinite(row.ami_max):
            note = f"наибольшее сходство с делением «{row.ami_max_by}» (AMI {style.fmt_num(row.ami_max, 2)})"
            if np.isfinite(row.ari_pair):
                pair = style.fmt_num(row.ari_pair, 2)
                note += f"; типы двух замеров ({row.pair_label}) совпадают на{NB}ARI {pair}"
            note = stories.style_nbsp(note)
        ctx.fact(f"proto_note_{key}", note, "str", f"{who}: быстрый прототип типов")
    k_proto = int(prm["prototype"]["k"])
    ctx.fact("proto_k", k_proto, "int", "типов в быстром прототипе (k-means)")
    proto_line = prototype_text(t15, sids) if len(t15) else ""

    # Матрица решения: со всеми МО и без внутригородских
    facts_now = _facts_of(findings, ctx)
    matrix = decision_matrix(facts_now, cfg)
    other = decision_matrix(stories.substitute(facts_now), cfg)
    _story_facts(ctx, matrix, stories.check_weights(eda["matrix_weights"]), other, [proto_line])
    matrix["score_no_inner"] = other["score"].to_numpy()
    matrix["status_no_inner"] = other["status"].to_numpy()

    # F16
    rp = classify(t13, prm)
    title, check = _headline(ctx, prm, rp)
    n_max = style.fmt_num(int(t13["n"].max())) if len(t13) else style.NA_TEXT
    ctx.save_figure(
        plot_region_or_place(t13, short, thr),
        fid="F16",
        slug="region_or_place",
        title=title,
        subtitle=(
            f"Чёрные точки — η² региона, серые ромбы — η² с поправкой на шум, белые — I Морана "
            f"(k{NB}={NB}{int(eda['knn_main'])}); n{NB}до{NB}{n_max} МО"
        ),
        alt=figure_alt(t13, prm, short),
        data=t13,
        check=check,
        # среди показателей — траты к зарплате Росстата и к доходу 5-НДФЛ
        source=style.join_sources(style.SOURCE_SBER, style.SOURCE_ROSSTAT, style.SOURCE_FNS),
    )

    # T13, T14, T15
    rec13 = ctx.save_table(
        t13,
        tid="T13",
        slug="region_or_place",
        title=(
            "Регион или место: η² региона и других делений, надёжность и η² с поправкой на шум, I Морана; "
            "с Москвой и Петербургом и без них"
        ),
        md_rows=len(t13),
    )
    ctx.tables[ctx.tables.index(rec13)] = replace(rec13, markdown=_t13_markdown(t13, short))
    score_fmt = {c: (lambda x: style.fmt_num(x, 2)) for c in ("score", "score_facts", "score_expert")}
    rec = ctx.save_table(
        matrix,
        tid="T14",
        slug="decision_matrix",
        title=(
            "Матрица решения: баллы 1–5 с весами (×), итог Σ вес × балл и его части, роль по критериям "
            "отказа; риски: 5 — рисков мало"
        ),
        md_rows=len(matrix),
    )
    # В отчёте — только баллы и роль (CSV — вся матрица): SectionContext.save_table пишет Markdown тех же
    # колонок, что и CSV, поэтому Markdown записи заменяется.
    md = to_markdown(matrix[list(T14_COLUMNS)], score_fmt, t14_labels(eda["matrix_weights"]))
    ctx.tables[ctx.tables.index(rec)] = replace(rec, markdown=md)
    t15 = t15.assign(story=[sids.get(k, "уровень трат, для сравнения") for k in t15["key"]])
    num2 = {
        c: (lambda x: style.fmt_num(x, 2)) for c in [*(f"ami_{g}" for g in GROUPINGS), "ami_max", "ari_pair"]
    }
    ctx.save_table(
        t15,
        tid="T15",
        slug="prototype",
        title=(
            f"Быстрый прототип типов: k-means (k{NB}={NB}{k_proto}) по признакам сюжета у МО с полным рядом; "
            "AMI типов с простыми делениями МО и ARI типов двух независимых замеров"
        ),
        md_rows=len(t15),
        md_formats=num2,
        md_labels=T15_LABELS,
        md_columns=["story", "features", "n", *(f"ami_{g}" for g in GROUPINGS), "ari_pair", "pair_label"],
    )

    return ctx.finding(
        title=TITLE,
        summary_md=summary_md(rp, short, t13),
        indicators=ind,
        indicator_labels=labels,
        caveats=CAVEATS,
    )
