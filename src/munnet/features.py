"""Этап features (этап 2 плана): признаки узлов сети по окнам времени и ряды для рёбер.

Вход — выходы этапа panel (``data/processed``), узлы — ``munnet.nodes`` по ``nodes.mode`` (по умолчанию
Москва и Петербург — два узла-города). Узел сети — узел с рядом не короче ``features.min_months`` месяцев;
остальные остаются в ``features_nodes`` с ``is_node`` = false и причиной ``drop_reason`` (никто не выпадает
молча). Выходы — ``data/processed/features_*.parquet`` по контрактам ``munnet.contracts``:

- ``features_nodes`` — справочник узлов: регион, группа региона, узел-город, число МО, статус ряда, узел
  или нет;
- ``features_members`` — каждое МО панели -> узел (``self``, ``city_member``, ``excluded``);
- ``features_windows`` — корзина и уровень по окнам (год, полугодие, квартал; ``features.windows``): доли
  шести частей от «Все категории», CLR, CLR относительно группы региона (``clr_rel_*``: CLR минус центр
  группы), уровень трат и его логарифм относительно группы (``log_level_rel``), прирост доли маркетплейсов
  к тому же окну прошлого года в п. п. (``mp_pp_yoy``). Окно есть у узла, только если в нём все месяцы окна;
- ``features_rhythm`` — годовой ритм (определение E3, ``s3_rhythm``): летний избыток и декабрьский пик по ряду
  без тренда узла, свой ритм (после тренда и общего ритма месяца): летний избыток, размах, повторяемость
  2023 ~ 2024, надёжность (``eda.season``), размах, сжатый на повторяемость, месяц пика; флаг Севера;
- ``features_place`` — экономика места по годам контекста: доли занятости по укрупнённым разделам ОКВЭД2,
  зарплата, доля горожан, доля старше трудоспособного возраста, население, доступность рынков, расстояние
  до столицы региона, Север, доход 5-НДФЛ на жителя (только ``ndfl_ok``); у части — значения относительно
  группы региона (``*_rel``, центр — ``features.place_center``);
- ``features_basket_monthly`` — помесячные CLR-векторы корзины (и относительно группы региона) для рёбер;
- ``features_rhythm_monthly`` — ряды ln трат по категориям: без тренда узла и свой ритм (для корреляций,
  лаговых корреляций и DTW следующего этапа).

«Относительно региона» — всегда относительно группы региона узла (``region_group``): город вместе со своей
областью, иначе у узла-города остаток тождественно нулевой. Доли корзины — композиция: сравнение только
в CLR; доля части в месяце меньше ``features.clr_floor`` заменяется мультипликативно (остальные доли
сжимаются так, чтобы сумма осталась 1), счётчик — ``n_replaced``.

В ``outputs/features/`` — ``checks.json`` (числа решения о сюжете, воспроизведённые на узлах; разброс
по seed), таблица признаков ``feature_table.csv`` (формула, покрытие, надёжность 2023 ~ 2024, η² группы
региона, наибольший |ρ| с другими признаками) и таблицы типов ``types_*.csv``.
"""

from __future__ import annotations

import logging
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from munnet import features_report, nodes
from munnet.config import Config
from munnet.contracts import (
    CATEGORY_CODES,
    FEATURES_BASKET_MONTHLY,
    FEATURES_MEMBERS,
    FEATURES_NODES,
    FEATURES_PLACE,
    FEATURES_RHYTHM,
    FEATURES_RHYTHM_MONTHLY,
    FEATURES_WINDOWS,
    N_MONTHS,
    OKVED_GROUPS,
    PARTS,
    WINDOW_KINDS,
    YEARS,
    coerce,
    write_table,
)
from munnet.eda import s3_rhythm, stats
from munnet.eda.base import write_json

log = logging.getLogger(__name__)

OUTPUT_SUBDIR = "features"
MP = "marketplace"
PP = 100.0
CENTERS: tuple[str, ...] = ("mean", "median")
# Окна: вид -> {суффикс: месяцы}.
WINDOW_MONTHS: dict[str, dict[str, tuple[int, ...]]] = {
    "year": {"": tuple(range(1, 13))},
    "half": {"H1": tuple(range(1, 7)), "H2": tuple(range(7, 13))},
    "quarter": {f"Q{q}": tuple(range(3 * q - 2, 3 * q + 1)) for q in range(1, 5)},
}
# Признаки места, у которых есть версия относительно группы региона: колонка -> (исходная, преобразование).
PLACE_RELATIVE: dict[str, tuple[str, str]] = {
    "log_wage_rel": ("wage", "log"),
    "urban_share_rel": ("urban_share", "id"),
    "age_old_share_rel": ("age_old_share", "id"),
    "log_pop_rel": ("pop_avg", "log"),
    "market_access_rel": ("market_access", "id"),
    "log_ndfl_rel": ("ndfl_income_pc", "log"),
}


# --- Параметры -----------------------------------------------------------------------------------


@dataclass(frozen=True)
class FeatureParams:
    """Параметры этапа: ``features`` конфига плюс общие определения разведки (``eda.season``, лето, Север)."""

    min_months: int
    windows: tuple[str, ...]
    clr_floor: float
    region_center: str
    place_center: str
    context_years: tuple[int, ...]
    rhythm_categories: tuple[str, ...]
    summer_months: tuple[int, ...]
    north_lat: float
    reliable_r: float
    reliable_amplitude: float
    null_repeats: int
    checks: Mapping[str, Any]
    space: Mapping[str, Any]
    seed: int

    @classmethod
    def from_config(cls, cfg: Config) -> FeatureParams:
        f, eda = cfg["features"], cfg["eda"]
        season = eda["season"]
        p = cls(
            min_months=int(f["min_months"]),
            windows=tuple(str(w) for w in f["windows"]),
            clr_floor=float(f["clr_floor"]),
            region_center=str(f["region_center"]),
            place_center=str(f["place_center"]),
            context_years=tuple(int(y) for y in f["context_years"]),
            rhythm_categories=tuple(str(c) for c in f["rhythm_categories"]),
            summer_months=tuple(int(m) for m in eda["summer_months"]),
            north_lat=float(eda["north_lat"]),
            reliable_r=float(season["reliable_r"]),
            reliable_amplitude=float(season["reliable_amplitude"]),
            null_repeats=int(season["null_repeats"]),
            checks=dict(f.get("checks") or {}),
            space=dict(f.get("space") or {}),
            seed=int(cfg["seed"]),
        )
        problems = []
        if not 1 <= p.min_months <= N_MONTHS:
            problems.append(f"min_months = {p.min_months}: нужно 1…{N_MONTHS}")
        unknown = sorted(set(p.windows) - set(WINDOW_KINDS))
        if unknown or not p.windows:
            problems.append(f"windows: неизвестные окна {unknown}; допустимы {list(WINDOW_KINDS)}")
        if not 0 <= p.clr_floor < 1 / len(PARTS):
            problems.append(f"clr_floor = {p.clr_floor}: нужно 0 ≤ порог < 1/{len(PARTS)}")
        for name in ("region_center", "place_center"):
            if getattr(p, name) not in CENTERS:
                problems.append(f"{name} = {getattr(p, name)!r}: допустимы {list(CENTERS)}")
        bad_years = sorted(set(p.context_years) - set(YEARS))
        if bad_years:
            problems.append(f"context_years: годы {bad_years} вне панели {list(YEARS)}")
        bad_cats = sorted(set(p.rhythm_categories) - set(CATEGORY_CODES))
        if bad_cats:
            problems.append(f"rhythm_categories: {bad_cats}; допустимы {list(CATEGORY_CODES)}")
        if problems:
            raise ValueError("features: " + "; ".join(problems))
        return p


# --- Композиции и центр группы -------------------------------------------------------------------


def replace_small(shares: pd.DataFrame, floor: float) -> tuple[pd.DataFrame, pd.Series]:
    """Мультипликативная замена почти нулей: доля ``< floor`` становится ``floor``, остальные доли строки
    умножаются на ``(1 − k·floor) / Σ остальных`` (k — число заменённых), сумма строки остаётся 1.

    Возвращает замкнутые доли и число заменённых частей в строке. ``floor`` = 0 — без замены.
    """
    closed = stats.closure(shares.astype("float64"))
    small = closed < floor
    k = small.sum(axis=1)
    if floor <= 0 or not small.any().any():
        return closed, k.astype("int8")
    rest = closed.where(~small).sum(axis=1)
    scale = (1.0 - k * floor) / rest
    out = closed.mul(scale, axis=0).where(~small, floor)
    return out, k.astype("int8")


def clr_frame(shares: pd.DataFrame, floor: float) -> tuple[pd.DataFrame, pd.Series]:
    """CLR долей (колонки — части) после мультипликативной замены почти нулей; строки CLR в сумме 0."""
    replaced, k = replace_small(shares, floor)
    return stats.clr(replaced), k


def group_center(values: pd.DataFrame, groups: pd.Series, how: str) -> pd.DataFrame:
    """Центр группы для каждой строки: среднее или медиана значений строк той же группы (без пропусков)."""
    g = values.groupby(groups.to_numpy())
    center = g.transform("mean") if how == "mean" else g.transform("median")
    center.index = values.index
    return center


def relative_clr(clr: pd.DataFrame, groups: pd.Series, how: str) -> pd.DataFrame:
    """CLR относительно группы региона: CLR минус центр группы. Центр ``mean`` — среднее CLR (центр композиций
    по Эйчисону), строка остаётся в сумме 0; ``median`` — медиана по координатам, после чего строка
    центрируется повторно (иначе сумма не 0)."""
    rel = clr - group_center(clr, groups, how)
    return rel.sub(rel.mean(axis=1), axis=0) if how == "median" else rel


def relative_values(
    values: pd.Series, groups: pd.Series, how: str, use: pd.Series | None = None
) -> pd.Series:
    """Значение минус центр своей группы; центр — только по строкам ``use`` (например, без выбросов)."""
    v = values.astype("float64")
    basis = v.where(use.astype(bool)) if use is not None else v
    center = group_center(basis.to_frame("v"), groups, how)["v"]
    return v - center


# --- Узлы ----------------------------------------------------------------------------------------


def select_nodes(node_table: pd.DataFrame, min_months: int) -> pd.DataFrame:
    """Справочник узлов с флагом ``is_node`` (ряд не короче ``min_months``) и причиной для остальных."""
    out = node_table.copy()
    out["is_node"] = out["n_months"].astype(int) >= min_months
    reason = "ряд " + out["n_months"].astype(int).astype(str) + f" мес. < features.min_months = {min_months}"
    out["drop_reason"] = reason.where(~out["is_node"]).astype("str")
    out.loc[out["is_node"], "drop_reason"] = None
    return out


# --- Окна корзины --------------------------------------------------------------------------------


def window_list(kinds: Sequence[str]) -> list[tuple[str, str, int, tuple[int, ...]]]:
    """Окна (имя, вид, год, месяцы) в порядке видов ``WINDOW_KINDS`` и лет панели."""
    out = []
    for kind in WINDOW_KINDS:
        if kind not in kinds:
            continue
        for year in YEARS:
            for suffix, months in WINDOW_MONTHS[kind].items():
                out.append((f"{year}{suffix}", kind, year, months))
    return out


def window_sums(wide: pd.DataFrame, year: int, months: Sequence[int]) -> pd.DataFrame:
    """Суммы трат частей и «Все категории» по окну у узлов со всеми месяцами окна; ``n`` — число месяцев."""
    part = wide.loc[(wide["year"] == year) & wide["month"].isin(list(months))]
    g = part.groupby("territory_id")
    sums = g[[f"v_{c}" for c in CATEGORY_CODES]].sum()
    sums["n"] = g.size()
    return sums.loc[sums["n"] == len(months)]


def window_features(wide: pd.DataFrame, groups: pd.Series, p: FeatureParams) -> pd.DataFrame:
    """``features_windows``: уровень, доли, CLR и CLR относительно группы по каждому окну; прирост доли
    маркетплейсов к тому же окну прошлого года (п. п.). ``groups`` — группа региона по ``territory_id``."""
    frames = []
    for name, kind, year, months in window_list(p.windows):
        sums = window_sums(wide, year, months)
        if sums.empty:
            continue
        shares = pd.DataFrame({q: sums[f"v_{q}"] / sums["v_all"] for q in PARTS}, index=sums.index)
        clr, _ = clr_frame(shares, p.clr_floor)
        grp = groups.reindex(sums.index)
        rel = relative_clr(clr, grp, p.region_center)
        level = sums["v_all"] / sums["n"]
        log_level = np.log(level)
        out = pd.DataFrame(
            {
                "territory_id": sums.index.astype("int32"),
                "window": name,
                "window_kind": kind,
                "year": year,
                "n_months": sums["n"].astype("int8").to_numpy(),
                "region_group": grp.to_numpy(),
                "level": level.to_numpy(),
                "log_level": log_level.to_numpy(),
                "log_level_rel": relative_values(log_level, grp, p.region_center).to_numpy(),
            }
        )
        for q in PARTS:
            out[f"sh_{q}"] = shares[q].to_numpy()
        for q in PARTS:
            out[f"clr_{q}"] = clr[q].to_numpy()
        for q in PARTS:
            out[f"clr_rel_{q}"] = rel[q].to_numpy()
        frames.append(out)
    table = pd.concat(frames, ignore_index=True)
    prev = table.assign(year=table["year"] + 1)
    prev["window"] = prev["year"].astype(str) + prev["window"].str.slice(4)
    table = table.merge(
        prev[["territory_id", "window", f"sh_{MP}"]].rename(columns={f"sh_{MP}": "_mp_prev"}),
        on=["territory_id", "window"],
        how="left",
        validate="one_to_one",
    )
    table["mp_pp_yoy"] = PP * (table[f"sh_{MP}"] - table["_mp_prev"])
    table = table.drop(columns="_mp_prev")
    table["window_kind"] = pd.Categorical(table["window_kind"], categories=list(WINDOW_KINDS), ordered=True)
    return coerce(table, FEATURES_WINDOWS)


# --- Ритм ----------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RhythmResult:
    features: pd.DataFrame
    monthly: pd.DataFrame
    common_share: float
    null_reliable_share: float


def rhythm_features(
    wide: pd.DataFrame, node_table: pd.DataFrame, p: FeatureParams, rng: np.random.Generator
) -> RhythmResult:
    """Ритм узлов с полным рядом (24 месяца) по определению E3 (``s3_rhythm.fit_rhythm``).

    ``summer_excess``, ``dec_peak`` — по ряду без тренда узла (как ``EdaData.mo``); ``own_*`` — свой ритм.
    Общий ритм месяца — медиана по узлам с полным рядом, поэтому он свой у каждого режима узлов.
    ``null_reliable_share`` — доля «надёжных» на перестановочном нуле (месяцы 2024 года внутри узла).
    """
    counts = wide.groupby("territory_id").size()
    ids = pd.Index(np.sort(counts.index[counts == N_MONTHS].to_numpy()), name="territory_id")
    models = {c: s3_rhythm.fit_rhythm(s3_rhythm.series_matrix(wide, ids, c)) for c in p.rhythm_categories}
    main = models["all"] if "all" in models else s3_rhythm.fit_rhythm(s3_rhythm.series_matrix(wide, ids))
    detr_prof = s3_rhythm.own_profile(main.D)
    rest = [m for m in detr_prof.columns if m != s3_rhythm.DECEMBER]
    dec_peak = np.expm1(detr_prof[s3_rhythm.DECEMBER] - detr_prof[rest].mean(axis=1))
    reliable = s3_rhythm.is_reliable(main.r, main.A, p.reliable_r, p.reliable_amplitude)
    lat = node_table.set_index("territory_id")["point_lat"].reindex(ids).astype("float64")
    feats = pd.DataFrame(
        {
            "territory_id": ids.astype("int32"),
            "summer_excess": s3_rhythm.seasonal_excess(main.D, p.summer_months).to_numpy(),
            "dec_peak": dec_peak.to_numpy(),
            "own_summer": np.expm1(s3_rhythm.profile_summer_excess(main.profile, p.summer_months)).to_numpy(),
            "own_amplitude": main.A.to_numpy(),
            "own_r": main.r.to_numpy(),
            "own_reliable": np.asarray(reliable, dtype=bool),
            "own_amplitude_shrunk": s3_rhythm.shrunk_amplitude(main.A, main.r).to_numpy(),
            "own_peak_month": main.profile.idxmax(axis=1).astype("int8").to_numpy(),
            "north": (lat >= p.north_lat).to_numpy(),
        }
    )
    null = s3_rhythm.null_reliable_share(
        main.O, main.A, rng, p.null_repeats, p.reliable_r, p.reliable_amplitude
    )
    months = wide.drop_duplicates("t").set_index("t")[["date", "year", "month"]].sort_index()
    parts = []
    for cat, m in models.items():
        frame = pd.DataFrame(
            {
                "territory_id": np.repeat(ids.to_numpy(), N_MONTHS).astype("int32"),
                "t": np.tile(np.arange(N_MONTHS), len(ids)).astype("int8"),
                "log_value": m.Y.to_numpy().ravel(),
                "detrended": m.D.to_numpy().ravel(),
                "own": m.O.to_numpy().ravel(),
            }
        )
        frame["category"] = cat
        parts.append(frame)
    monthly = pd.concat(parts, ignore_index=True).join(months, on="t")
    monthly["category"] = pd.Categorical(monthly["category"], categories=list(CATEGORY_CODES), ordered=True)
    return RhythmResult(
        features=coerce(feats, FEATURES_RHYTHM),
        monthly=coerce(monthly, FEATURES_RHYTHM_MONTHLY),
        common_share=s3_rhythm.common_share(main.D, main.O),
        null_reliable_share=float(null),
    )


# --- Экономика места -----------------------------------------------------------------------------


def place_features(
    context: pd.DataFrame, node_table: pd.DataFrame, ids: pd.Index, p: FeatureParams
) -> pd.DataFrame:
    """``features_place``: экономика места узлов ``ids`` по годам ``context_years``.

    Доли занятости — по укрупнённым разделам ОКВЭД2 и «не раскрыто» (из контекста узлов); зарплата
    с флагом выброса; доход 5-НДФЛ на жителя — только при ``ndfl_ok``. Версии «относительно группы региона»
    (``*_rel``) — значение минус центр группы (``place_center``); выбросы в центр не входят.
    """
    ter = node_table.set_index("territory_id").reindex(ids)
    frames = []
    for year in p.context_years:
        c = context.loc[context["year"] == year].set_index("territory_id").reindex(ids)
        out = pd.DataFrame(index=ids)
        out["year"] = year
        out["region_group"] = ter["region_group"]
        for g in (*OKVED_GROUPS, "unallocated"):
            out[f"emp_sh_{g}"] = c[f"emp_sh_{g}"].astype("float64")
        out["wage"] = c["wage"].astype("float64")
        out["wage_outlier"] = c["wage_outlier"].astype("boolean").fillna(False).astype(bool)
        out["urban_share"] = c["urban_share"].astype("float64")
        out["age_old_share"] = c["age_old_share"].astype("float64")
        out["pop_avg"] = c["pop_avg"].astype("float64")
        out["log_pop"] = np.log(out["pop_avg"])
        out["market_access"] = ter["market_access"].astype("float64")
        out["dist_capital_km"] = ter["dist_capital_km"].astype("float64")
        out["north"] = ter["point_lat"].astype("float64") >= p.north_lat
        out["ndfl_ok"] = c["ndfl_ok"].astype("boolean").fillna(False).astype(bool)
        out["ndfl_income_pc"] = c["ndfl_income_pc"].astype("float64").where(out["ndfl_ok"])
        grp = out["region_group"]
        for col, (src, tr) in PLACE_RELATIVE.items():
            v = out[src].astype("float64")
            v = np.log(v.where(v > 0)) if tr == "log" else v
            use = ~out["wage_outlier"] if src == "wage" else None
            out[col] = relative_values(v, grp, p.place_center, use)
        frames.append(out.reset_index())
    return coerce(pd.concat(frames, ignore_index=True), FEATURES_PLACE)


# --- Помесячная корзина --------------------------------------------------------------------------


def basket_monthly(wide: pd.DataFrame, groups: pd.Series, p: FeatureParams) -> pd.DataFrame:
    """``features_basket_monthly``: CLR шести долей узла в месяце (после замены почти нулей) и CLR
    относительно группы региона того же месяца (центр — по узлам группы с этим месяцем)."""
    shares = wide[[f"sh_{q}" for q in PARTS]].set_axis(list(PARTS), axis=1)
    clr, k = clr_frame(shares, p.clr_floor)
    grp = wide["territory_id"].map(groups)
    key = grp.astype(str) + "|" + wide["date"].astype(str)
    rel = relative_clr(clr, key, p.region_center)
    out = wide[["territory_id", "date", "t", "year", "month"]].copy()
    out["region_group"] = grp.to_numpy()
    for q in PARTS:
        out[f"clr_{q}"] = clr[q].to_numpy()
    for q in PARTS:
        out[f"clr_rel_{q}"] = rel[q].to_numpy()
    out["n_replaced"] = k.to_numpy()
    n_rep = int((k > 0).sum())
    log.info("корзина по месяцам: мультипликативная замена долей < %g у %d узло-месяцев", p.clr_floor, n_rep)
    return coerce(out, FEATURES_BASKET_MONTHLY)


# --- Проверки решения о сюжете (checks.json) -----------------------------------------------------


def period_clr(wide: pd.DataFrame, mask: pd.Series, ids: pd.Index, floor: float) -> pd.DataFrame:
    """CLR долей по суммам трат периода (``mask`` — строки панели), индекс — ``ids``."""
    d = wide.loc[mask].groupby("territory_id")[[f"v_{q}" for q in PARTS]].sum().reindex(ids)
    clr, _ = clr_frame(d.set_axis(list(PARTS), axis=1), floor)
    return clr


def _kmeans(x: pd.DataFrame, k: int, n_init: int, seed: int):
    from sklearn.cluster import KMeans

    return KMeans(n_clusters=k, n_init=n_init, random_state=seed).fit(x.to_numpy())


def _tree_kappa(labels: pd.Series, attrs: pd.DataFrame, depth: int, folds: int, seed: int) -> float:
    """Каппа Коэна меток и предсказания неглубокого дерева по признакам места (кросс-валидация)."""
    from sklearn.metrics import cohen_kappa_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.tree import DecisionTreeClassifier

    x = attrs.reindex(labels.index).astype("float64")
    x = x.fillna(x.median())  # как в черновой проверке 26.09: пропуск — медиана признака
    tree = DecisionTreeClassifier(max_depth=depth, random_state=seed)
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    pred = cross_val_predict(tree, x.to_numpy(), labels.to_numpy(), cv=cv)
    return float(cohen_kappa_score(labels.to_numpy(), pred))


def story_checks(
    wide: pd.DataFrame,
    ids: pd.Index,
    groups: pd.Series,
    regions: pd.Series,
    place: pd.DataFrame,
    log_level: pd.DataFrame,
    p: FeatureParams,
    seed: int,
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    """Числа решения о сюжете (PLAN.md, этап 1) на узлах ``ids`` с полным рядом, как в черновой проверке.

    1. k-means (``checks.k``) по CLR 2024 года относительно группы региона: AMI типов с группой региона
       и с субъектом; ARI типов 2023 и 2024 годов (k-means по каждому году отдельно); каппа неглубокого
       дерева «экономика места 2023 года -> тип» (кросс-валидация). То же для корзины, очищенной от уровня
       и региона (С4), — для сравнения.
    2. Динамика: k-means на объединённых векторах 2023 и 2024 годов (относительно группы региона в каждом
       периоде), метки всех периодов — по его центрам. Доля узлов, сменивших тип 2023 -> 2024, против шума —
       смены типа между нечётными и чётными месяцами одного года и между одинаковыми полугодиями.
    3. Общая шкала: векторы минус средний CLR группы за оба года (общий сдвиг к маркетплейсам не вычитается).
       Опустевший тип — тип, потерявший наибольшую долю своих узлов 2023 года; в черновике 26.09 это «офлайн»-
       тип (маркетплейсы и общепит ниже отсчёта, продовольствие, транспорт и здоровье выше): его центр
       записан в ``checks.json``, чтобы название проверялось по числам, а не по номеру типа.
    """
    from sklearn.metrics import adjusted_mutual_info_score as ami
    from sklearn.metrics import adjusted_rand_score as ari

    c = p.checks
    k, n_init = int(c.get("k", 5)), int(c.get("n_init", 20))
    grp = groups.reindex(ids)
    reg = regions.reindex(ids)
    y = wide["year"]
    m = wide["month"]
    masks = {
        "2023": y == 2023,
        "2024": y == 2024,
        "2023_odd": (y == 2023) & (m % 2 == 1),
        "2023_even": (y == 2023) & (m % 2 == 0),
        "2024_odd": (y == 2024) & (m % 2 == 1),
        "2024_even": (y == 2024) & (m % 2 == 0),
        "2023H1": (y == 2023) & (m <= 6),
        "2023H2": (y == 2023) & (m > 6),
        "2024H1": (y == 2024) & (m <= 6),
        "2024H2": (y == 2024) & (m > 6),
    }
    clr = {name: period_clr(wide, mask, ids, p.clr_floor) for name, mask in masks.items()}
    rel = {name: relative_clr(v, grp, "mean") for name, v in clr.items()}

    out: dict[str, Any] = {"n_nodes": len(ids), "k": k, "seed": seed}
    km24 = _kmeans(rel["2024"], k, n_init, seed)
    lab24 = pd.Series(km24.labels_, index=ids)
    lab23 = pd.Series(_kmeans(rel["2023"], k, n_init, seed).labels_, index=ids)
    out["ami_region"] = float(ami(grp.astype(str), lab24))
    out["ami_region_code"] = float(ami(reg.astype(str), lab24))
    out["ari_years"] = float(ari(lab23, lab24))
    attrs = place.loc[place["year"] == int(c.get("tree_year", YEARS[0]))].set_index("territory_id")
    cols = [col for col in c.get("tree_features", []) if col in attrs.columns]
    tree_x = attrs[cols].astype("float64")
    depth, folds = int(c.get("tree_depth", 3)), int(c.get("cv_folds", 5))
    out["tree_kappa"] = _tree_kappa(lab24, tree_x, depth, folds, seed)
    # С4 для сравнения: CLR 2024 года, очищенный МНК от ln уровня 2024 года и группы региона
    from munnet.eda.s4_basket import residualize

    ctrl = pd.DataFrame(
        {"log_level": log_level["2024"].reindex(ids), "region": grp.astype(str).astype("category")}
    )
    clean, _, _ = residualize(clr["2024"], ctrl)
    lab_clean = pd.Series(_kmeans(clean, k, n_init, seed).labels_, index=ids)
    out["ami_region_clean"] = float(ami(grp.astype(str), lab_clean))
    out["tree_kappa_clean"] = _tree_kappa(lab_clean, tree_x, depth, folds, seed)

    tables: dict[str, pd.DataFrame] = {}
    for mode, frames in (
        ("relative", rel),
        (
            "fixed",
            {n: v - group_center((clr["2023"] + clr["2024"]) / 2, grp, "mean") for n, v in clr.items()},
        ),
    ):
        pooled = pd.concat([frames["2023"], frames["2024"]])
        km = _kmeans(pooled, k, n_init, seed)
        lab = {n: pd.Series(km.predict(v.to_numpy()), index=ids) for n, v in frames.items()}
        centers = pd.DataFrame(km.cluster_centers_, columns=list(PARTS))
        centers["n_2023"] = lab["2023"].value_counts().reindex(range(k), fill_value=0).to_numpy()
        centers["n_2024"] = lab["2024"].value_counts().reindex(range(k), fill_value=0).to_numpy()
        centers.index.name = "type"
        tables[f"types_{mode}"] = centers.reset_index()
        moved = float((lab["2023"] != lab["2024"]).mean())
        noise = [float((lab[f"{yy}_odd"] != lab[f"{yy}_even"]).mean()) for yy in ("2023", "2024")]
        halves = [float((lab[f"2023{h}"] != lab[f"2024{h}"]).mean()) for h in ("H1", "H2")]
        if mode == "relative":
            out["moved_share"] = moved
            out["noise_share_2023"], out["noise_share_2024"] = noise
            out["noise_share"] = float(np.mean(noise))
            out["moved_share_halves"] = float(np.mean(halves))
            ct = pd.crosstab(lab["2023"], lab["2024"]).reindex(index=range(k), columns=range(k), fill_value=0)
            ct.index.name = "type_2023"
            tables["types_transitions"] = ct.reset_index()
        else:
            loss = (centers["n_2023"] - centers["n_2024"]) / centers["n_2023"].where(centers["n_2023"] > 0)
            offline = int(loss.idxmax())
            out["offline_type"] = offline
            out["offline_2023"] = int(centers.loc[offline, "n_2023"])
            out["offline_2024"] = int(centers.loc[offline, "n_2024"])
            for q in PARTS:
                out[f"offline_center_{q}"] = float(centers.loc[offline, q])
            # «офлайн»: маркетплейсы и общепит ниже отсчёта — то, что название утверждает
            out["offline_is_offline"] = bool(
                centers.loc[offline, MP] < 0 and centers.loc[offline, "cafe"] < 0
            )
            out["moved_share_fixed"] = moved
            out["noise_share_fixed"] = float(np.mean(noise))
    return out, tables


# --- Таблица признаков ---------------------------------------------------------------------------

COLLINEAR_RHO = 0.8  # |ρ| Спирмена больше — пара признаков почти дублирует друг друга (умолчание space)


@dataclass(frozen=True)
class FeatureDoc:
    """Описание признака для таблицы признаков и ``docs/features.md``."""

    table: str  # таблица data/processed
    group: str  # группа признаков (ключ ``FEATURE_GROUPS``)
    label: str  # подпись для людей
    formula: str  # формула словами
    norm: str  # нормировка: CLR, относительно группы региона, логарифм…
    captures: str  # какую экономическую разницу ловит


FEATURE_GROUPS: dict[str, str] = {
    "basket": "Корзина",
    "level": "Уровень трат",
    "marketplace": "Маркетплейсы",
    "rhythm": "Годовой ритм",
    "place": "Экономика места",
}
_REL = "относительно группы региона"
_PART_LABELS = {
    "food": "продовольствие",
    "marketplace": "маркетплейсы",
    "transport": "транспорт",
    "health": "здоровье",
    "cafe": "общепит",
    "other": "«Прочее»",
}
_GROUP_LABELS = {
    "primary": "сельское хозяйство и добыча",
    "industry": "обработка и энергетика",
    "trade_transport": "стройка, торговля, транспорт",
    "market_services": "рыночные услуги",
    "public": "госуправление, образование, здравоохранение",
}
FEATURE_DOCS: dict[str, FeatureDoc] = {
    **{
        f"clr_rel_{q}": FeatureDoc(
            "features_windows",
            "basket",
            f"Корзина: {_PART_LABELS[q]}",
            f"CLR доли «{_PART_LABELS[q]}» за год минус среднее CLR группы региона",
            f"CLR, {_REL}",
            "на что жители тратят больше или меньше, чем соседи по региону",
        )
        for q in PARTS
    },
    "log_level_rel": FeatureDoc(
        "features_windows",
        "level",
        "Уровень трат",
        "ln средних трат жителя в месяц минус центр группы региона",
        f"логарифм, {_REL}",
        "насколько жители тратят больше или меньше соседей по региону",
    ),
    "sh_marketplace": FeatureDoc(
        "features_windows",
        "marketplace",
        "Доля маркетплейсов",
        "траты на маркетплейсы / «Все категории» за год",
        "доля без нормировки",
        "насколько покупки ушли в онлайн",
    ),
    "mp_pp_yoy": FeatureDoc(
        "features_windows",
        "marketplace",
        "Прирост доли маркетплейсов",
        "доля маркетплейсов года минус доля того же окна годом раньше, п. п.",
        "разность долей, п. п.",
        "как быстро покупки уходят в онлайн",
    ),
    "summer_excess": FeatureDoc(
        "features_rhythm",
        "rhythm",
        "Летний избыток трат",
        "лето к прочим месяцам без декабря, ряд без тренда узла",
        "логарифм, без тренда узла",
        "сезонность: завоз, отпуска, северное лето",
    ),
    "own_summer": FeatureDoc(
        "features_rhythm",
        "rhythm",
        "Летний избыток своего ритма",
        "то же после вычета общего ритма месяца (сдвиг на одно число для всех узлов)",
        "логарифм, без тренда узла и общего ритма",
        "лето сильнее или слабее, чем у типичного узла",
    ),
    "dec_peak": FeatureDoc(
        "features_rhythm",
        "rhythm",
        "Декабрьский пик",
        "декабрь к январю–ноябрю, ряд без тренда узла",
        "логарифм, без тренда узла",
        "новогодний пик трат",
    ),
    "own_amplitude_shrunk": FeatureDoc(
        "features_rhythm",
        "rhythm",
        "Сила своего ритма",
        "размах своего годового профиля × max(r, 0), r — повторяемость 2023 ~ 2024",
        "логарифм, сжатие на повторяемость",
        "есть ли у узла устойчивый свой годовой ритм",
    ),
    **{
        f"emp_sh_{g}": FeatureDoc(
            "features_place",
            "place",
            f"Занятость: {_GROUP_LABELS[g]}",
            "работники разделов ОКВЭД2 группы / все работники крупных и средних организаций",
            "доля",
            "чем зарабатывает место",
        )
        for g in OKVED_GROUPS
    },
    "log_wage_rel": FeatureDoc(
        "features_place",
        "place",
        "Зарплата",
        "ln средней зарплаты минус медиана группы региона (без выбросов)",
        f"логарифм, {_REL}",
        "заработки на месте",
    ),
    "urban_share": FeatureDoc(
        "features_place",
        "place",
        "Доля горожан",
        "городское население / всё население",
        "доля",
        "урбанизация",
    ),
    "urban_share_rel": FeatureDoc(
        "features_place",
        "place",
        "Доля горожан к региону",
        "доля горожан минус медиана группы региона",
        f"доля, {_REL}",
        "урбанизация сверх своего региона",
    ),
    "age_old_share": FeatureDoc(
        "features_place",
        "place",
        "Доля старше трудоспособного возраста",
        "старше трудоспособного возраста / всё население",
        "доля",
        "старение",
    ),
    "age_old_share_rel": FeatureDoc(
        "features_place",
        "place",
        "Доля старших к региону",
        "доля старше трудоспособного возраста минус медиана группы региона",
        f"доля, {_REL}",
        "старение сверх своего региона",
    ),
    "log_pop": FeatureDoc(
        "features_place", "place", "Население", "ln среднегодового населения", "логарифм", "размер"
    ),
    "log_pop_rel": FeatureDoc(
        "features_place",
        "place",
        "Население к региону",
        "ln населения минус медиана группы региона",
        f"логарифм, {_REL}",
        "крупнее или мельче соседей по региону",
    ),
    "market_access": FeatureDoc(
        "features_place",
        "place",
        "Доступность рынков",
        "индекс доступности рынков СберИндекса",
        "индекс 0–1000",
        "доступ к рынкам",
    ),
    "market_access_rel": FeatureDoc(
        "features_place",
        "place",
        "Доступность рынков к региону",
        "индекс минус медиана группы региона",
        f"индекс, {_REL}",
        "центр или периферия своего региона",
    ),
    "log_ndfl_rel": FeatureDoc(
        "features_place",
        "place",
        "Доход 5-НДФЛ",
        "ln дохода 5-НДФЛ на жителя минус медиана группы региона (только ndfl_ok)",
        f"логарифм, {_REL}",
        "доходы по месту работы",
    ),
}
# Версии одного показателя (относительно региона и как есть): их связь не считается дублированием.
SAME_VARIABLE: tuple[frozenset[str], ...] = (
    frozenset({"urban_share", "urban_share_rel"}),
    frozenset({"age_old_share", "age_old_share_rel"}),
    frozenset({"log_pop", "log_pop_rel"}),
    frozenset({"market_access", "market_access_rel"}),
)
ROLES: tuple[str, ...] = ("edges", "attributes", "layer", "dynamics")
ROLE_TEXT: dict[str, str] = {
    "edges": "рёбра",
    "attributes": "атрибут",
    "layer": "слой (ритм)",
    "dynamics": "динамика",
    "outside": "вне пространства",
}


def rhythm_by_year(monthly: pd.DataFrame, summer_months: Sequence[int]) -> dict[str, pd.DataFrame]:
    """Летний избыток и декабрьский пик каждого года отдельно (ряд «Все категории» без тренда узла): для
    надёжности признаков ритма 2023 ~ 2024. Колонки — годы панели, индекс — ``territory_id``."""
    d = monthly.loc[monthly["category"].astype(str) == "all"]
    D = d.pivot(index="territory_id", columns="t", values="detrended").sort_index(axis=1)
    summer = s3_rhythm.summer_excess_by_year(D, summer_months)
    blocks = D.to_numpy().reshape(len(D), len(YEARS), 12)
    dec = np.expm1(blocks[:, :, 11] - blocks[:, :, :11].mean(axis=2))
    return {"summer_excess": summer, "dec_peak": pd.DataFrame(dec, index=D.index, columns=list(YEARS))}


def _same_variable(x: str, y: str) -> bool:
    return any({x, y} <= pair for pair in SAME_VARIABLE)


def _verdict(row: pd.Series, space: Mapping[str, Any], eta2_max: float) -> tuple[bool, str]:
    """Проверка места признака в пространстве: (выполнено, почему) по правилам ``features.space``."""
    rel_min = float(space.get("min_reliability", 0.5))
    rho_max = float(space.get("max_abs_rho", COLLINEAR_RHO))
    dup = float(space.get("duplicate_rho", 0.999))
    rel, rho, eta2 = row["reliability"], row["max_abs_rho_role"], row["eta2_region_group"]
    role = row["role"]
    if role == "outside":
        reasons = []
        if row["max_abs_rho"] >= dup:
            reasons.append(f"ранги совпадают с {row['max_abs_rho_with']}")
        if np.isfinite(rel) and rel < rel_min:
            reasons.append("ненадёжен")
        if np.isfinite(eta2) and eta2 > eta2_max:
            reasons.append("почти целиком региональный")
        return True, "; ".join(reasons) or "для интерпретации"
    problems = []
    if np.isfinite(rel) and rel < rel_min:
        problems.append("надёжность ниже порога")
    if role == "attributes" and np.isfinite(rho) and rho > rho_max:
        problems.append(f"дублирует {row['max_abs_rho_role_with']}")
    if role == "attributes" and np.isfinite(eta2) and eta2 > eta2_max:
        problems.append("почти целиком региональный")
    if role == "layer" and np.isfinite(eta2) and eta2 > eta2_max and not problems:
        return True, "региональный по природе: это слой, а не атрибут типов"
    if problems:
        return False, "; ".join(problems)
    return True, "повторяемость встроена в признак" if not np.isfinite(rel) else "правила выполнены"


def feature_table(
    windows: pd.DataFrame,
    rhythm: pd.DataFrame,
    place: pd.DataFrame,
    n_nodes: int,
    by_year_rhythm: Mapping[str, pd.DataFrame] | None = None,
    coords: pd.DataFrame | None = None,
    space: Mapping[str, Any] | None = None,
    moran: Mapping[str, Any] | None = None,
    eta2_max: float = 0.6,
) -> pd.DataFrame:
    """Таблица признаков ``FEATURE_DOCS``: формула, нормировка, покрытие, надёжность (ρ Спирмена двух
    независимых замеров: 2023 ~ 2024, а у прироста доли маркетплейсов — январь–июнь ~ июль–декабрь 2024 года),
    η² группы региона, I Морана (k ближайших узлов по ``coords``, перестановочный тест; ``moran``: ``k``,
    ``permutations``, ``rng``), наибольший |ρ| с другими признаками (2024 год, контекст 2023 года; версии
    одного показателя не считаются), роль в пространстве ``space`` и проверка правил роли."""
    y0, y1 = YEARS
    y = windows.loc[windows["window_kind"].astype(str) == "year"]
    by_year = {yr: y.loc[y["year"] == yr].set_index("territory_id") for yr in YEARS}
    halves = windows.loc[windows["window_kind"].astype(str) == "half"].set_index(["window", "territory_id"])
    pl = {
        yr: place.loc[place["year"] == yr].set_index("territory_id")
        for yr in YEARS
        if yr in set(place["year"])
    }
    rh = rhythm.set_index("territory_id")
    ry = dict(by_year_rhythm or {})
    space = dict(space or {})
    role_of = {f: r for r in ROLES for f in space.get(r, [])}
    unknown = sorted(set(role_of) - set(FEATURE_DOCS))
    if unknown:
        raise ValueError(f"features.space: неизвестные признаки {unknown}; допустимы {sorted(FEATURE_DOCS)}")
    wide = pd.DataFrame(index=by_year[y1].index)
    rows = []
    for name, doc in FEATURE_DOCS.items():
        basis = f"{y0} ~ {y1}"
        if doc.table == "features_windows":
            cur = by_year[y1][name]
            a, b = by_year[y0][name], cur
            if a.isna().all() and f"{y1}H1" in halves.index.get_level_values(0):
                a, b = halves.loc[f"{y1}H1", name], halves.loc[f"{y1}H2", name]
                basis = f"{y1}: январь–июнь ~ июль–декабрь"
            rel = stats.spearman(a, b.reindex(a.index))[0]
            grp = by_year[y1]["region_group"]
        elif doc.table == "features_rhythm":
            cur = rh[name].astype("float64")
            per_year = ry.get(name)
            rel = stats.spearman(per_year[y0], per_year[y1])[0] if per_year is not None else float("nan")
            basis = basis if per_year is not None else "—"
            grp = by_year[y1]["region_group"].reindex(rh.index)
        else:
            base = pl.get(y0)
            cur = base[name].astype("float64")
            other = pl.get(y1)
            rel = (
                stats.spearman(cur, other[name].reindex(cur.index))[0] if other is not None else float("nan")
            )
            grp = base["region_group"]
        wide[name] = cur.reindex(wide.index).astype("float64")
        row = {
            "feature": name,
            "group": doc.group,
            "label": doc.label,
            "table": doc.table,
            "formula": doc.formula,
            "norm": doc.norm,
            "captures": doc.captures,
            "role": role_of.get(name, "outside"),
            "coverage": int(cur.notna().sum()),
            "coverage_share": float(cur.notna().sum() / n_nodes) if n_nodes else float("nan"),
            "reliability": rel,
            "reliability_basis": basis,
            "eta2_region_group": stats.eta2(cur.astype("float64"), grp.reindex(cur.index)),
            "moran_i": float("nan"),
            "moran_p": float("nan"),
        }
        if coords is not None and moran is not None:
            xy = coords.reindex(cur.index)
            m = stats.morans_i(cur, xy, int(moran["k"]), int(moran["permutations"]), moran["rng"])
            row["moran_i"], row["moran_p"] = m.I, m.p
        rows.append(row)
    out = pd.DataFrame(rows).set_index("feature")
    corr = wide.rank().corr().abs()
    for name in out.index:
        others = corr[name].drop(name)
        others = others[[not _same_variable(name, o) for o in others.index]]
        if others.notna().any():
            out.loc[name, "max_abs_rho"] = float(others.max())
            out.loc[name, "max_abs_rho_with"] = str(others.idxmax())
        role = out.loc[name, "role"]
        same_role = [o for o in others.index if out.loc[o, "role"] == role and role != "outside"]
        if same_role and others[same_role].notna().any():
            out.loc[name, "max_abs_rho_role"] = float(others[same_role].max())
            out.loc[name, "max_abs_rho_role_with"] = str(others[same_role].idxmax())
        edges = [o for o in others.index if out.loc[o, "role"] == "edges"]
        if role == "attributes" and edges:
            out.loc[name, "max_abs_rho_edges"] = float(others[edges].max())
            out.loc[name, "max_abs_rho_edges_with"] = str(others[edges].idxmax())
    for col in ("max_abs_rho", "max_abs_rho_role", "max_abs_rho_edges"):
        if col not in out.columns:
            out[col] = np.nan
    for col in ("max_abs_rho_with", "max_abs_rho_role_with", "max_abs_rho_edges_with"):
        if col not in out.columns:
            out[col] = None
    out["collinear_08"] = out["max_abs_rho"] > float(space.get("max_abs_rho", COLLINEAR_RHO))
    checks = [_verdict(r, space, eta2_max) for _, r in out.iterrows()]
    out["check_ok"] = [c[0] for c in checks]
    out["check_note"] = [c[1] for c in checks]
    return out.reset_index()


# --- Запуск --------------------------------------------------------------------------------------


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        df.to_csv(tmp, index=False, lineterminator="\n", float_format="%.6g")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


# Числа checks.json, у которых разброс по seed не нужен: параметры и номера типов.
NO_RANGE: tuple[str, ...] = ("k", "n_nodes", "seed", "offline_type", "offline_is_offline")


JSON_DIGITS = 10  # значащих цифр в checks.json: центры k-means расходятся в 17-м знаке (многопоточные суммы)


def _rounded(obj: Any) -> Any:
    """Числа JSON с ``JSON_DIGITS`` значащими цифрами: повторный прогон даёт тот же файл побайтно."""
    if isinstance(obj, dict):
        return {k: _rounded(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_rounded(v) for v in obj]
    if isinstance(obj, float) and np.isfinite(obj):
        return float(f"{obj:.{JSON_DIGITS}g}")
    return obj


def _range(values: Sequence[dict[str, Any]], key: str) -> list[float] | None:
    v = [float(x[key]) for x in values if key in x and x[key] is not None and np.isfinite(float(x[key]))]
    return [min(v), max(v)] if v else None


@dataclass(frozen=True)
class StoryInputs:
    """Входы ``story_checks``: узлы сети, узлы с полным рядом, группы и субъекты, траты, признаки места,
    ln уровня по годам. Одни и те же у этапа features и у сводки разведки (``decision_numbers``)."""

    table: pd.DataFrame
    ids: pd.Index
    full: pd.Index
    groups: pd.Series
    regions: pd.Series
    wide: pd.DataFrame
    windows: pd.DataFrame
    place: pd.DataFrame
    log_level: pd.DataFrame


def story_inputs(nd: nodes.NodeData, p: FeatureParams) -> StoryInputs:
    """Узлы сети (``min_months``), их траты, окна корзины и признаки места — вход проверок сюжета."""
    table = select_nodes(nd.nodes, p.min_months)
    ids = pd.Index(np.sort(table.loc[table["is_node"], "territory_id"].to_numpy()), name="territory_id")
    groups = table.set_index("territory_id")["region_group"].astype("int16")
    wide = nd.panel_wide.loc[nd.panel_wide["territory_id"].isin(ids)].reset_index(drop=True)
    counts = wide.groupby("territory_id").size()
    full = pd.Index(np.sort(counts.index[counts == N_MONTHS].to_numpy()), name="territory_id")
    windows = window_features(wide, groups, p)
    y = windows.loc[windows["window_kind"].astype(str) == "year"]
    return StoryInputs(
        table=table,
        ids=ids,
        full=full,
        groups=groups,
        regions=table.set_index("territory_id")["region_code"],
        wide=wide,
        windows=windows,
        place=place_features(nd.context_annual, table, ids, p),
        log_level=y.pivot(index="territory_id", columns="window", values="log_level"),
    )


def decision_numbers(nd: nodes.NodeData, cfg: Config) -> dict[str, Any]:
    """Числа решения о сюжете (``story_checks``, seed конфига) — для сводки разведки: те же функции и входы,
    что у этапа features, поэтому числа совпадают с ``outputs/features/checks.json`` (раздел ``story``)."""
    p = FeatureParams.from_config(cfg)
    s = story_inputs(nd, p)
    values, _ = story_checks(s.wide, s.full, s.groups, s.regions, s.place, s.log_level, p, p.seed)
    return values


def run(cfg: Config) -> None:
    """Признаки узлов: ``data/processed/features_*.parquet`` и ``outputs/features/``."""
    p = FeatureParams.from_config(cfg)
    nd = nodes.load_node_data(cfg)
    s = story_inputs(nd, p)
    table, ids, groups, wide, windows, place = s.table, s.ids, s.groups, s.wide, s.windows, s.place
    log.info(
        "features: режим узлов %s, узлов сети %d из %d (ряд не короче %d мес.)",
        nd.mode,
        len(ids),
        len(table),
        p.min_months,
    )
    singles = groups.reindex(ids).value_counts()
    if (singles == 1).any():
        log.warning(
            "features: в %d группах региона один узел — признаки «относительно региона» у них 0",
            int((singles == 1).sum()),
        )

    rng = np.random.default_rng([p.seed, 2])  # этап 2 плана: свой поток случайности
    rhythm = rhythm_features(wide, table, p, rng)
    monthly = basket_monthly(wide, groups, p)

    processed = cfg.dir("processed")
    write_table(table, FEATURES_NODES, processed / "features_nodes.parquet")
    write_table(nd.members, FEATURES_MEMBERS, processed / "features_members.parquet")
    write_table(windows, FEATURES_WINDOWS, processed / "features_windows.parquet")
    write_table(rhythm.features, FEATURES_RHYTHM, processed / "features_rhythm.parquet")
    write_table(place, FEATURES_PLACE, processed / "features_place.parquet")
    write_table(monthly, FEATURES_BASKET_MONTHLY, processed / "features_basket_monthly.parquet")
    write_table(rhythm.monthly, FEATURES_RHYTHM_MONTHLY, processed / "features_rhythm_monthly.parquet")

    out_dir = cfg.dir("outputs") / OUTPUT_SUBDIR
    full, regions, log_level = s.full, s.regions, s.log_level
    if not full.equals(pd.Index(rhythm.features["territory_id"], name="territory_id")):
        raise ValueError("features: узлы с полным рядом у ритма и у проверок сюжета разошлись")
    n_seeds = int(p.checks.get("seeds", 1))
    runs, tables = [], {}
    for i in range(max(n_seeds, 1)):
        vals, tabs = story_checks(wide, full, groups, regions, place, log_level, p, p.seed + i)
        runs.append(vals)
        if i == 0:
            tables = tabs
    main = runs[0]
    reference = {k: float(v) for k, v in (p.checks.get("reference") or {}).items()}
    draft = {k: float(v) for k, v in (p.checks.get("draft_26_09") or {}).items()}
    tol = float(p.checks.get("reference_tol", 0.01))

    def diff(ref: Mapping[str, float]) -> dict[str, float]:
        return {k: float(main[k]) - v for k, v in ref.items() if main.get(k) is not None}

    checks = {
        "node_mode": nd.mode,
        "node_mode_text": nodes.MODE_TEXT[nd.mode],
        "region_group_text": nodes.group_text(nd),
        "n_nodes_all": len(table),
        "n_nodes": len(ids),
        "n_city_nodes": len(nd.city_ids),
        "min_months": p.min_months,
        "rhythm": {
            "n_full": len(full),
            "reliable_share": float(rhythm.features["own_reliable"].mean()),
            "null_reliable_share": rhythm.null_reliable_share,
            "common_share": rhythm.common_share,
        },
        "clr_replaced_node_months": int((monthly["n_replaced"] > 0).sum()),
        "story": main,
        "story_seeds": [p.seed + i for i in range(len(runs))],
        "story_seed_range": {
            k: _range(runs, k) for k, v in main.items() if isinstance(v, (int, float)) and k not in NO_RANGE
        },
        "reference": reference,
        "reference_tol": tol,
        "diff_vs_reference": diff(reference),
        "reference_ok": {k: abs(d) <= tol * max(1.0, abs(reference[k])) for k, d in diff(reference).items()},
        "draft_26_09": {
            "note": (
                "черновые скрипты 26.09 вне конвейера: 1774 МО без внутригородских территорий, регион — свой "
                "субъект, seed 0; на них опиралось решение о сюжете (PLAN.md, этап 1); не контрольные числа"
            ),
            "values": draft,
            "diff": diff(draft),
        },
    }
    write_json(_rounded(checks), out_dir / "checks.json")
    for name, t in tables.items():
        _write_csv(t, out_dir / f"{name}.csv")
    coords = table.set_index("territory_id")[["x_aea", "y_aea"]]
    moran = {
        "k": int(cfg["eda"]["knn_main"]),
        "permutations": int(cfg["eda"]["permutations"]),
        "rng": np.random.default_rng([p.seed, 3]),
    }
    eta2_max = float(cfg["eda"]["synthesis"]["eta2_regional"])
    ft = feature_table(
        windows,
        rhythm.features,
        place,
        len(ids),
        rhythm_by_year(rhythm.monthly, p.summer_months),
        coords,
        p.space,
        moran,
        eta2_max,
    )
    _write_csv(ft, out_dir / "feature_table.csv")
    features_report.write_report(cfg, p, s, ft, checks, rhythm, moran, eta2_max)
    for k, d in sorted(checks["diff_vs_reference"].items()):
        ok = checks["reference_ok"][k]
        (log.info if ok else log.warning)(
            "features: %s = %.4g (сверка: %.4g, разница %+.4g%s; черновик 26.09: %s)",
            k,
            main[k],
            reference[k],
            d,
            "" if ok else ", больше допуска",
            draft.get(k, "—"),
        )
    log.info("features: выходы — %s/features_*.parquet, %s", processed, out_dir)
