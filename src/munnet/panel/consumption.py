"""Панель трат «МО × месяц × категория»: чтение, седьмая категория «Прочее», широкая форма, качество рядов.

Траты СберИндекса — оценка средних безналичных потребительских расходов жителя МО в текущем месяце,
₽, номинал. «Все категории» включает пять категорий и иные, поэтому остаток «Прочее» = «Все категории» −
сумма пяти всегда больше нуля, а шесть частей образуют замкнутую композицию (spec_final, А.3, А.4).
Панель никого не выбрасывает: порог числа месяцев применяют следующие этапы.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from munnet.contracts import CATEGORY_CODES, PARTS, YEARS, QCError

SOURCE_COLUMNS = ["date", "territory_id", "category", "value"]  # без служебного __index_level_0__ (Л23)
TOTAL = CATEGORY_CODES[0]  # «Все категории»
OTHER = CATEGORY_CODES[-1]  # остаток «Прочее»
SOURCE_PARTS = PARTS[:-1]  # пять категорий источника
MONTHS_PER_YEAR = 12
MAX_EXAMPLES = 10


def month_list(years: Sequence[int]) -> list[str]:
    """Месяцы панели строкой «ГГГГ-ММ» по порядку: номер в списке — ``t``."""
    return [f"{y}-{m:02d}" for y in years for m in range(1, MONTHS_PER_YEAR + 1)]


def add_month_columns(df: pd.DataFrame, years: Sequence[int] = YEARS) -> pd.DataFrame:
    """Добавляет ``t`` (int8, 0 — первый месяц панели), ``year`` (int16), ``month`` (int8) по ``date``.

    Месяц вне годов панели — ``ValueError``.
    """
    index = {d: i for i, d in enumerate(month_list(years))}
    t = df["date"].map(index)
    if t.isna().any():
        bad = sorted(df.loc[t.isna(), "date"].unique())[:MAX_EXAMPLES]
        raise ValueError(f"месяцы вне годов панели {list(years)}: {bad}")
    out = df.copy()
    out["t"] = t.astype("int8")
    out["year"] = out["date"].str.slice(0, 4).astype("int16")
    out["month"] = out["date"].str.slice(5, 7).astype("int8")
    return out


def read_consumption(path: Path, categories: dict[str, str]) -> pd.DataFrame:
    """Читает траты с явным списком колонок и переводит названия категорий в коды.

    ``categories`` — код → название источника (``panel.categories``). Неизвестная категория — ``ValueError``.
    Выход: ``territory_id`` (int32), ``date`` (строка «ГГГГ-ММ»), ``category`` (код), ``value`` (int32).
    """
    df = pd.read_parquet(path, columns=SOURCE_COLUMNS)
    to_code = {name: code for code, name in categories.items()}
    unknown = sorted(set(df["category"].dropna()) - set(to_code))
    if unknown:
        raise ValueError(f"{path}: неизвестные категории {unknown}; известны {sorted(to_code)}")
    out = pd.DataFrame(
        {
            "territory_id": df["territory_id"].astype("int32"),
            "date": df["date"].astype("str"),
            "category": df["category"].map(to_code).astype("str"),
            "value": df["value"].astype("int32"),
        }
    )
    return out


def _examples(frame: pd.DataFrame, cols: list[str]) -> str:
    rows = frame[cols].head(MAX_EXAMPLES).itertuples(index=False)
    more = f" и ещё {len(frame) - MAX_EXAMPLES}" if len(frame) > MAX_EXAMPLES else ""
    return "; ".join(str(tuple(r)) for r in rows) + more


def add_other(long: pd.DataFrame) -> pd.DataFrame:
    """Добавляет седьмую категорию ``other`` = ``all`` − сумма пяти категорий источника.

    Проверки до расчёта (``QCError`` с перечнем МО и месяцев): дубли ключа, неполный набор категорий в
    МО-месяце, остаток ≤ 0. Выход — длинная таблица из семи категорий: ``territory_id, date, category``
    (упорядоченная категория ``CATEGORY_CODES``), ``value`` (int32), ``is_derived`` (True только у ``other``).
    """
    key = ["territory_id", "date", "category"]
    dup = long.duplicated(subset=key, keep=False)
    if dup.any():
        raise QCError(f"траты: {int(dup.sum())} строк с повторяющимся ключом: {_examples(long[dup], key)}")
    wide = long.pivot(index=["territory_id", "date"], columns="category", values="value")
    need = [TOTAL, *SOURCE_PARTS]
    absent = [c for c in need if c not in wide.columns]
    if absent:
        raise QCError(f"траты: нет категорий {absent}")
    incomplete = wide[need].isna().any(axis=1)
    if incomplete.any():
        bad = wide[incomplete].reset_index()
        raise QCError(
            f"траты: в {int(incomplete.sum())} МО-месяцах не все категории: "
            f"{_examples(bad, ['territory_id', 'date'])}"
        )
    vals = wide[need].astype("int64")
    other = vals[TOTAL] - vals[list(SOURCE_PARTS)].sum(axis=1)
    bad = other <= 0
    if bad.any():
        frame = other[bad].rename("other").reset_index()
        raise QCError(
            f"траты: «Прочее» ≤ 0 в {int(bad.sum())} МО-месяцах (сумма пяти категорий не меньше «Все "
            f"категории»): {_examples(frame, ['territory_id', 'date', 'other'])}"
        )
    extra = other.astype("int32").rename("value").reset_index()
    extra["category"] = OTHER
    base = long[key + ["value"]]
    out = pd.concat([base, extra[key + ["value"]]], ignore_index=True)
    out["category"] = pd.Categorical(out["category"], categories=list(CATEGORY_CODES), ordered=True)
    out["value"] = out["value"].astype("int32")
    out["is_derived"] = (out["category"] == OTHER).to_numpy()
    return out.sort_values(key, kind="mergesort").reset_index(drop=True)


def to_wide(long: pd.DataFrame, years: Sequence[int] = YEARS) -> pd.DataFrame:
    """Широкая форма «МО × месяц»: ``v_<код>`` (int32, ₽), ``sh_<часть>`` (доля от ``v_all``), ``log_all``.

    Шесть долей в сумме дают 1 (части в сумме равны ``v_all`` точно, в целых). Только наблюдённые месяцы.
    """
    w = long.pivot(index=["territory_id", "date"], columns="category", values="value")
    w.columns = [str(c) for c in w.columns]
    w = w.reset_index()
    out = add_month_columns(w[["territory_id", "date"]], years)
    for code in CATEGORY_CODES:
        out[f"v_{code}"] = w[code].astype("int32").to_numpy()
    total = out[f"v_{TOTAL}"].astype("float64")
    for part in PARTS:
        out[f"sh_{part}"] = out[f"v_{part}"].astype("float64") / total
    out["log_all"] = np.log(total)
    out["territory_id"] = out["territory_id"].astype("int32")
    return out.sort_values(["territory_id", "date"], kind="mergesort").reset_index(drop=True)


def _runs_of_zeros(row: np.ndarray) -> int:
    """Самая длинная серия нулей в одномерном массиве 0/1."""
    best = cur = 0
    for x in row:
        cur = cur + 1 if x == 0 else 0
        best = max(best, cur)
    return best


def series_quality(long: pd.DataFrame, years: Sequence[int] = YEARS) -> pd.DataFrame:
    """Качество ряда трат каждого МО (А.4, ``territories``).

    ``n_months``, ``n_<год>`` — месяцев с данными; ``first_date``, ``last_date`` — границы ряда;
    ``coverage_pattern`` — строка из 0/1 по месяцам панели; ``series_status``: ``full`` (все месяцы),
    ``only_<год>`` (ровно календарный год и ничего больше), ``partial`` (прочее); ``has_internal_gap`` —
    пропуск между первым и последним месяцем; ``longest_gap`` — самый длинный такой пропуск.
    """
    months = month_list(years)
    obs = long[["territory_id", "date"]].drop_duplicates()
    ids = np.sort(obs["territory_id"].astype("int32").unique())
    mat = np.zeros((len(ids), len(months)), dtype=np.int8)
    row = pd.Index(ids).get_indexer(obs["territory_id"])
    col = pd.Index(months).get_indexer(obs["date"])
    if (col < 0).any():
        raise ValueError(
            f"месяцы вне годов панели: {sorted(obs.loc[col < 0, 'date'].unique())[:MAX_EXAMPLES]}"
        )
    mat[row, col] = 1

    n = mat.sum(axis=1)
    first = mat.argmax(axis=1)
    last = len(months) - 1 - mat[:, ::-1].argmax(axis=1)
    out = pd.DataFrame({"territory_id": ids.astype("int32"), "n_months": n.astype("int8")})
    per_year = {}
    for i, y in enumerate(years):
        block = mat[:, i * MONTHS_PER_YEAR : (i + 1) * MONTHS_PER_YEAR].sum(axis=1)
        per_year[y] = block
        out[f"n_{y}"] = block.astype("int8")
    out["first_date"] = np.asarray(months, dtype=object)[first]
    out["last_date"] = np.asarray(months, dtype=object)[last]
    out["first_date"] = out["first_date"].astype("str")
    out["last_date"] = out["last_date"].astype("str")
    out["coverage_pattern"] = ["".join("1" if x else "0" for x in r) for r in mat]
    out["coverage_pattern"] = out["coverage_pattern"].astype("str")

    status = np.full(len(ids), "partial", dtype=object)
    status[n == len(months)] = "full"
    for y in years:
        others = n - per_year[y]
        status[(per_year[y] == MONTHS_PER_YEAR) & (others == 0)] = f"only_{y}"
    out["series_status"] = status
    out["series_status"] = out["series_status"].astype("str")
    inner_gaps = [_runs_of_zeros(mat[i, first[i] : last[i] + 1]) for i in range(len(ids))]
    out["longest_gap"] = np.asarray(inner_gaps, dtype=np.int8)
    out["has_internal_gap"] = out["longest_gap"] > 0
    return out[
        [
            "territory_id",
            "n_months",
            *[f"n_{y}" for y in years],
            "first_date",
            "last_date",
            "coverage_pattern",
            "series_status",
            "has_internal_gap",
            "longest_gap",
        ]
    ]


def build_panel(
    raw: pd.DataFrame, years: Sequence[int] = YEARS
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Из прочитанных трат — ``panel_long`` (7 категорий), ``panel_wide`` и качество рядов МО."""
    long = add_other(raw)
    panel_long = add_month_columns(long, years)
    panel_long = panel_long[["territory_id", "date", "t", "year", "month", "category", "value", "is_derived"]]
    panel_wide = to_wide(long, years)
    quality = series_quality(long, years)
    return panel_long, panel_wide, quality
