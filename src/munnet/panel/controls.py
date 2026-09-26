"""Контрольные числа этапа panel и отчёты о покрытии (spec_final, А.11).

Жёсткие числа — инварианты закреплённых sha256 входов: расхождение означает ошибку кода и роняет этап
(``QCError``, код выхода 3). Мягкие — покрытие контекста: отклонение больше ``panel.controls.tolerance``
даёт предупреждение. Всё посчитанное, чего нет в конфиге, пишется как информационное (без проверки).
``controls.json`` пишется до проверки, чтобы расхождение можно было разобрать.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from munnet.config import Config
from munnet.contracts import QCError
from munnet.panel.bdmo import TOTAL_DIM

log = logging.getLogger(__name__)

MLN = 1e6
JSON_DIGITS = 6  # знаков после запятой у дробных чисел в controls.json
UNION_HISTORY = "Объединение"  # метка oktmo_history БД ПМО: код изменился при объединении МО

Tables = dict[str, pd.DataFrame]


# --- Жёсткие числа ------------------------------------------------------------------------------


def _hard_fixed() -> dict[str, Callable[[Tables], float]]:
    return {
        "consumption_rows": lambda t: len(t["consumption"]),
        "territories": lambda t: len(t["territories"]),
        "months": lambda t: t["panel_wide"]["date"].nunique(),
        "mo_months": lambda t: len(t["panel_wide"]),
        "panel_long_rows": lambda t: len(t["panel_long"]),
        "dup_keys": lambda t: int(t["consumption"].duplicated(["territory_id", "date", "category"]).sum()),
        "other_nonpositive": lambda t: int((t["panel_wide"]["v_other"] <= 0).sum()),
        "exactly_12_months": lambda t: int((t["territories"]["n_months"] == 12).sum()),
        "internal_gap": lambda t: int(t["territories"]["has_internal_gap"].sum()),
        "dict_versions": lambda t: len(t["dictionary"]),
        "dict_territories": lambda t: t["dictionary"]["territory_id"].nunique(),
        "inner_city": lambda t: int(t["territories"]["is_inner_city"].sum()),
        "regions": lambda t: t["territories"]["region_code"].nunique(),
        "polygons_matched": lambda t: int(
            t["territories"]["territory_id"].isin(t["geo"]["territory_id"]).sum()
        ),
        "market_access": lambda t: int(t["territories"]["market_access"].notna().sum()),
    }


def _hard_pattern(name: str, t: Tables) -> float | None:
    """Числа с годом или статусом в имени: ``series_<статус>``, ``in_slice_<год>``."""
    if m := re.fullmatch(r"series_(\w+)", name):
        return int((t["territories"]["series_status"].astype(str) == m.group(1)).sum())
    if m := re.fullmatch(r"in_slice_(\d{4})", name):
        col = f"oktmo_{m.group(1)}"
        return int(t["territories"][col].notna().sum()) if col in t["territories"] else None
    return None


# --- Мягкие и информационные числа --------------------------------------------------------------


def _coverage(long: pd.DataFrame, indicator: str, year: int) -> int:
    """Сколько МО панели имеют значение показателя за год: по итогу ``TOTAL``, без итога — по разрезам."""
    sel = long[(long["indicator"].astype(str) == indicator) & (long["year"] == year)]
    if (sel["dim"] == TOTAL_DIM).any():
        sel = sel[sel["dim"] == TOTAL_DIM]
    return int(sel["territory_id"].nunique())


def _soft_value(name: str, t: Tables) -> float | None:
    long, ann = t["context_long"], t["context_annual"]
    pop = long[(long["indicator"].astype(str) == "population") & (long["dim"] == TOTAL_DIM)]
    if m := re.fullmatch(r"pop_jan1_(\d{4})", name):
        return int(pop.loc[pop["year"] == int(m.group(1)), "territory_id"].nunique())
    if m := re.fullmatch(r"pop_avg_both_(\d{4})", name):
        a = ann[ann["year"] == int(m.group(1))]
        return int((a["pop_avg_method"].astype(str) == "mean_jan1").sum())
    if m := re.fullmatch(r"pop_carried_(\d{4})", name):
        return int(((pop["year"] == int(m.group(1))) & (pop["method"].astype(str) == "carried")).sum())
    if m := re.fullmatch(r"pop_fallback_period_(\d{4})", name):
        flag = pop["flag"].fillna("").str.contains("fallback_period")
        return int(((pop["year"] == int(m.group(1))) & flag).sum())
    if m := re.fullmatch(r"pop_sum_(\d{4})_mln", name):
        return round(float(ann.loc[ann["year"] == int(m.group(1)), "pop_jan1"].sum()) / MLN, JSON_DIGITS)
    if m := re.fullmatch(r"(\w+)_(\d{4})", name):
        indicator, year = m.group(1), int(m.group(2))
        if indicator in set(long["indicator"].astype(str)):
            return _coverage(long, indicator, year)
    return None


def _stable_union_values(t: Tables) -> int:
    """Значения, найденные мостом ``stable`` у строк источника с «Объединение» в ``oktmo_history``.

    Ручная проверка запрета моста для объединений (А.6): число показывается в таблице T02 разведки.
    """
    long = t["context_long"]
    st = long[long["method"].astype(str) == "stable"]
    rows = pd.concat(
        [t.get("bdmo_rows", pd.DataFrame()), t.get("ndfl_rows", pd.DataFrame())], ignore_index=True
    )
    if st.empty or rows.empty:
        return 0
    rows = rows[rows["oktmo_history"].fillna("").str.contains(UNION_HISTORY, regex=False)]
    keys = rows[["indicator", "oktmo", "year", "dim"]].astype({"indicator": "str"})
    hit = st.assign(indicator=st["indicator"].astype(str)).merge(
        keys,
        left_on=["indicator", "oktmo_used", "year", "dim"],
        right_on=["indicator", "oktmo", "year", "dim"],
    )
    return len(hit)


def _info(t: Tables, cfg: Config) -> dict[str, float]:
    """Информационные числа: покрытие всех показателей по годам, методы, мост ``stable``."""
    long = t["context_long"]
    out: dict[str, float] = {}
    for (indicator, year), _ in long.groupby([long["indicator"].astype(str), "year"]):
        out[f"{indicator}_{int(year)}"] = _coverage(long, indicator, int(year))
    methods = long[long["dim"].isin([TOTAL_DIM]) | (long["indicator"].astype(str) == "age")]
    methods = methods.drop_duplicates(["territory_id", "year", "indicator"])
    for (indicator, year, method), g in methods.groupby(
        [methods["indicator"].astype(str), "year", methods["method"].astype(str)]
    ):
        out[f"method_{indicator}_{int(year)}_{method}"] = int(len(g))
    out["stable_union_values"] = _stable_union_values(t)
    return out


def compute_controls(tables: Tables, cfg: Config, info: dict | None = None) -> dict:
    """Все контрольные числа: жёсткие и мягкие из ``panel.controls`` и информационные.

    ``tables`` — ``consumption`` (прочитанные траты), ``panel_long``, ``panel_wide``, ``territories``,
    ``dictionary``, ``geo``, ``context_long``, ``context_annual``, ``bdmo_rows``, ``ndfl_rows``. ``info`` —
    дополнительные числа этапа (например, число пар моста). Возвращает ``{имя: значение}``; число, которое
    не удалось посчитать, — ``None``.
    """
    ctl = cfg["panel"]["controls"]
    fixed = _hard_fixed()
    actual: dict[str, float | None] = {}
    for name in ctl["hard"]:
        actual[name] = fixed[name](tables) if name in fixed else _hard_pattern(name, tables)
    for name in ctl["soft"]:
        actual[name] = _soft_value(name, tables)
    for name, value in {**_info(tables, cfg), **(info or {})}.items():
        actual.setdefault(name, value)
    return {k: (int(v) if isinstance(v, bool) else v) for k, v in actual.items()}


def _json_value(v):
    """Число для JSON: целые — int, дробные — округлённый float, пропуск — None."""
    if v is None:
        return None
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return round(float(v), JSON_DIGITS)
    return v


def atomic_write(path: Path, write: Callable[[str], None]) -> None:
    """Пишет файл через временный рядом и ``os.replace``: старый файл не портится при сбое.

    ``write`` получает путь временного файла и пишет в него.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        write(tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def _write_json(obj: dict, path: Path) -> None:
    def write(tmp: str) -> None:
        with open(tmp, "w", encoding="utf-8", newline="\n") as f:
            json.dump(obj, f, ensure_ascii=False, sort_keys=True, indent=1)
            f.write("\n")

    atomic_write(path, write)


def check_controls(actual: dict, cfg: Config, out_path: Path) -> list[str]:
    """Сверяет числа с конфигом, пишет ``controls.json`` и только потом проверяет.

    Формат записи: ``{имя: {expected, actual, kind: hard | soft | info, ok}}``. Жёсткое расхождение (или
    число, которое не посчиталось) → ``QCError`` с перечнем; мягкое — предупреждение в логе. Возвращает
    список предупреждений по мягким числам.
    """
    ctl = cfg["panel"]["controls"]
    tol = float(ctl["tolerance"])
    report, errors, warnings = {}, [], []
    for name, expected in ctl["hard"].items():
        got = actual.get(name)
        ok = got is not None and float(got) == float(expected)
        report[name] = {"expected": expected, "actual": _json_value(got), "kind": "hard", "ok": ok}
        if not ok:
            errors.append(f"{name}: ожидалось {expected}, получено {got}")
    for name, expected in ctl["soft"].items():
        got = actual.get(name)
        ok = got is not None and abs(float(got) - float(expected)) <= tol * abs(float(expected))
        report[name] = {"expected": expected, "actual": _json_value(got), "kind": "soft", "ok": ok}
        if not ok:
            warnings.append(f"{name}: ожидалось {expected} (±{tol:.0%}), получено {got}")
    for name, got in actual.items():
        if name not in report:
            report[name] = {"expected": None, "actual": _json_value(got), "kind": "info", "ok": True}
    _write_json(report, Path(out_path))
    for w in warnings:
        log.warning("мягкое контрольное число: %s", w)
    if errors:
        raise QCError(
            f"не сошлись {len(errors)} жёстких контрольных чисел этапа panel:\n- " + "\n- ".join(errors)
        )
    return warnings


# --- Отчёты о покрытии --------------------------------------------------------------------------


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    atomic_write(path, lambda tmp: df.to_csv(tmp, index=False, lineterminator="\n", encoding="utf-8"))


def write_coverage_reports(tables: Tables, out_dir: Path) -> None:
    """CSV покрытия: по месяцам, по регионам, контекст «показатель × год × метод», потери с причинами.

    ``coverage_by_month.csv`` — МО с тратами в месяц; ``coverage_by_region.csv`` — МО панели и статусы
    рядов по субъектам; ``context_coverage.csv`` — сколько МО панели получили значение и каким методом;
    ``unmatched.csv`` — МО панели без значения показателя за год и причина.
    """
    out_dir = Path(out_dir)
    wide, ter, long = tables["panel_wide"], tables["territories"], tables["context_long"]

    by_month = wide.groupby("date", sort=True).agg(n_mo=("territory_id", "nunique")).reset_index()
    _write_csv(by_month, out_dir / "coverage_by_month.csv")

    status = pd.crosstab(ter["region_code"], ter["series_status"].astype(str))
    status.columns = [f"n_{c}" for c in status.columns]
    region = (
        ter.groupby("region_code", sort=True)
        .agg(
            region_name=("region_name", "first"),
            n_mo=("territory_id", "size"),
            n_inner_city=("is_inner_city", "sum"),
            n_mo_months=("n_months", "sum"),
        )
        .join(status)
        .fillna(0)
        .reset_index()
    )
    for col in region.columns:
        if col.startswith("n_"):
            region[col] = region[col].astype("int64")
    _write_csv(region, out_dir / "coverage_by_region.csv")

    n_panel = len(ter)
    sel = long[(long["dim"] == TOTAL_DIM) | (long["indicator"].astype(str) == "age")]
    sel = sel.drop_duplicates(["territory_id", "year", "indicator"])
    cov = (
        sel.groupby([sel["indicator"].astype(str), "year", sel["method"].astype(str)], sort=True)
        .size()
        .rename("n_mo")
        .reset_index()
    )
    total = cov.groupby(["indicator", "year"])["n_mo"].transform("sum")
    cov["n_mo_indicator"] = total
    cov["share_of_panel"] = (total / n_panel).round(JSON_DIGITS)
    _write_csv(cov, out_dir / "context_coverage.csv")

    _write_csv(tables["unmatched"], out_dir / "unmatched.csv")
