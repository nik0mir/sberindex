"""Маленький мир для тестов узлов сети и признаков: известные ответы свёртки Москвы и Петербурга.

МО 1 и 2 — Московская область (50), МО 3 — Карелия (10, севернее 60°), МО 4 и 5 — Тверская область (69),
районы 11 и 12 — Москва (77), район 21 — Петербург (78); Ленинградской области (47) в мире нет, поэтому
Петербург остаётся в своей группе региона. У района 12 нет марта 2023 года, у МО 5 — всего 2024 года.

Траты района: части = база × (4, 1, 1, 1, 1, 2) для (food, marketplace, transport, health, cafe, other),
«Все категории» = сумма частей; у каждого МО свой множитель месяца, чтобы ряды не совпадали.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from munnet.contracts import (
    CATEGORY_CODES,
    CITY_CONTEXT,
    CONTEXT_ANNUAL,
    N_MONTHS,
    OKVED_SECTIONS,
    OKVED_SHARES,
    PANEL_WIDE,
    PARTS,
    TERRITORIES,
    YEARS,
    coerce,
    validate,
)

PART_WEIGHTS = {"food": 4, "marketplace": 1, "transport": 1, "health": 1, "cafe": 1, "other": 2}
# territory_id -> (регион, название региона, тип, внутригородская, база трат, население, широта)
MO = {
    1: (50, "Московская область", "mr", False, 150, 20_000, 55.9),
    2: (50, "Московская область", "go", False, 200, 60_000, 55.6),
    3: (10, "Республика Карелия", "mr", False, 180, 15_000, 63.0),
    4: (69, "Тверская область", "mr", False, 120, 10_000, 57.0),
    5: (69, "Тверская область", "go", False, 160, 40_000, 56.9),
    11: (77, "Москва", "vgt", True, 100, 1_000, 55.75),
    12: (77, "Москва", "vgt", True, 300, 3_000, 55.70),
    21: (78, "Санкт-Петербург", "vgt", True, 250, 2_000, 59.94),
}
MISSING = {12: ["2023-03"], 5: [f"2023-{m:02d}" for m in range(1, 13)]}
CITY_RECIPIENTS = 5_600.0  # получатели 5-НДФЛ Москвы из строки «Субъект РФ»
# Доли разделов ОКВЭД2 в работниках: у районов Москвы — с нераскрытым остатком; группы — как в конфиге.
SECTION_SHARES = {11: {"A": 0.5}, 12: {"A": 0.2, "C": 0.8}}
DEFAULT_SECTIONS = {"A": 0.1, "G": 0.9}
GROUPS = {
    "primary": "AB",
    "industry": "CDE",
    "trade_transport": "FGH",
    "market_services": "IJKLMNRST",
    "public": "OPQ",
}


def months() -> list[str]:
    return [f"{y}-{m:02d}" for y in YEARS for m in range(1, 13)]


def territories() -> pd.DataFrame:
    rows = []
    for tid, (reg, reg_name, mo_type, inner, _, _, lat) in MO.items():
        present = [d for d in months() if d not in MISSING.get(tid, [])]
        pattern = "".join("1" if d in present else "0" for d in months())
        n23 = sum(d.startswith("2023") for d in present)
        status = "full" if len(present) == N_MONTHS else ("only_2024" if n23 == 0 else "partial")
        rows.append(
            {
                "territory_id": tid,
                "name": f"МО {tid}",
                "name_short": f"МО {tid}",
                "region_code": reg,
                "region_name": reg_name,
                "mo_type": mo_type,
                "mo_status": None,
                "is_capital": False,
                "is_inner_city": inner,
                "oktmo_2023": f"{reg:02d}{tid:06d}",
                "oktmo_2024": f"{reg:02d}{tid:06d}",
                "shape": 1,
                "center_name": None,
                "center_lat": np.nan,
                "center_lon": np.nan,
                "point_lat": lat,
                "point_lon": 37.6 if reg in (50, 77) else 30.0 + tid,
                "x_aea": 1000.0 * tid,
                "y_aea": 2000.0 * tid,
                "area_km2": 10.0 * tid,
                "market_access": 400.0 + tid,
                "dist_capital_km": np.nan if inner else 5.0 * tid,
                "n_months": len(present),
                "n_2023": n23,
                "n_2024": len(present) - n23,
                "first_date": present[0],
                "last_date": present[-1],
                "coverage_pattern": pattern,
                "series_status": status,
                "has_internal_gap": tid == 12,
                "longest_gap": 1 if tid == 12 else 0,
                "lineage_role": "none",
                "lineage_change_id": None,
                "successor_id": pd.NA,
            }
        )
    return validate(coerce(pd.DataFrame(rows), TERRITORIES), TERRITORIES)


def panel_wide() -> pd.DataFrame:
    rows = []
    for tid, spec in MO.items():
        base = spec[4]
        for t, d in enumerate(months()):
            if d in MISSING.get(tid, []):
                continue
            factor = 1.0 + 0.01 * ((t * tid) % 7)
            parts = {p: int(round(base * w * factor)) for p, w in PART_WEIGHTS.items()}
            row = {"territory_id": tid, "date": d, "t": t, "year": int(d[:4]), "month": int(d[5:])}
            row.update({f"v_{p}": v for p, v in parts.items()})
            row["v_all"] = sum(parts.values())
            rows.append(row)
    df = pd.DataFrame(rows)
    for p in PARTS:
        df[f"sh_{p}"] = df[f"v_{p}"] / df["v_all"]
    df["log_all"] = np.log(df["v_all"].astype("float64"))
    df = df.astype({"territory_id": "int32", "t": "int8", "year": "int16", "month": "int8"})
    for c in CATEGORY_CODES:
        df[f"v_{c}"] = df[f"v_{c}"].astype("int32")
    return validate(df, PANEL_WIDE)


def context_annual() -> pd.DataFrame:
    """Контекст: у районов Москвы работники 100 и 300, зарплата 50 000 и 100 000, розница только у 11."""
    rows = []
    for tid, (_, _, _, inner, _, pop, _) in MO.items():
        for year in YEARS:
            r = {c.name: np.nan for c in CONTEXT_ANNUAL.columns}
            r.update(territory_id=tid, year=year, pop_jan1=float(pop), pop_avg=float(pop))
            r.update(pop_jan1_method="direct", pop_avg_method="mean_jan1", urban_share=1.0 if inner else 0.5)
            r.update(age_young_share=0.2, age_working_share=0.6, age_old_share=0.2)
            employees = {11: 100.0, 12: 300.0}.get(tid, pop / 5)
            wage = {11: 50_000.0, 12: 100_000.0}.get(tid, 40_000.0)
            r.update(employees=employees, wage=wage, payroll_krub=employees * wage * 12 / 1000)
            sh = SECTION_SHARES.get(tid, DEFAULT_SECTIONS)
            for sec in OKVED_SECTIONS:
                r[f"emp_sh_{sec}"] = sh.get(sec, np.nan)
            for group, letters in GROUPS.items():
                r[f"emp_sh_{group}"] = sum(sh.get(x, 0.0) for x in letters)
            r.update(emp_sh_unallocated=1.0 - sum(sh.values()), emp_n_disclosed=len(sh))
            r.update(retail_krub=10.0 if tid != 12 else np.nan)
            r.update(ndfl_income_rub={11: 1e6, 12: 3e6}.get(tid, pop * 30_000.0), ndfl_recipients=pop * 0.8)
            r.update(workplace_based=bool(inner), ndfl_ok=not inner)
            r["recipients_to_pop"] = r["ndfl_recipients"] / pop
            if not inner:
                r["ndfl_income_pc"] = r["ndfl_income_rub"] / pop / 12
                r["retail_pc"] = r["retail_krub"] * 1000 / pop / 12
                r["payroll_pc"] = r["payroll_krub"] * 1000 / pop / 12
            rows.append(r)
    df = pd.DataFrame(rows)
    df["emp_sh_renormalized"] = False
    df["ndfl_ok_neighbour"] = False
    return validate(coerce(df, CONTEXT_ANNUAL), CONTEXT_ANNUAL)


def okved_shares() -> pd.DataFrame:
    """Район 11: A — 50 из 100, остальное не раскрыто; район 12: A — 60, C — 240 из 300; прочие: A, G."""
    shares = SECTION_SHARES
    rows = []
    ann = context_annual().set_index(["territory_id", "year"])["employees"]
    for tid in MO:
        for year in YEARS:
            total = float(ann.loc[(tid, year)])
            sh = shares.get(tid, DEFAULT_SECTIONS)
            for s in OKVED_SECTIONS:
                share = sh.get(s)
                rows.append((tid, year, s, None if share is None else share * total, share))
            un = 1.0 - sum(sh.values())
            rows.append((tid, year, "unallocated", un * total, un))
    df = pd.DataFrame(rows, columns=["territory_id", "year", "section", "employees", "share"])
    df = df.astype({"territory_id": "int32", "year": "int16", "employees": "float64", "share": "float64"})
    return validate(df, OKVED_SHARES)


def city_context() -> pd.DataFrame:
    rows = [(77, y, "ndfl_recipients", CITY_RECIPIENTS) for y in YEARS]
    df = pd.DataFrame(rows, columns=["region_code", "year", "indicator", "value"])
    df["oktmo"] = "45000000"
    df["source_code"] = "Y777000028"
    df["report_type"] = "Свод"
    df["flag"] = pd.Series([None] * len(df), dtype="str")
    return validate(df.astype({"region_code": "int16", "year": "int16"}), CITY_CONTEXT)


def node_config(mode: str = "collapse") -> dict:
    """Секция ``nodes`` конфига для маленького мира."""
    return {
        "mode": mode,
        "city_id_base": 90000,
        "min_pop_coverage": 1.0,
        "cities": {
            77: {"name": "Москва", "oktmo": "45000000", "region_group": 50, "outliers": ["wage", "ndfl"]},
            78: {"name": "Санкт-Петербург", "oktmo": "40000000", "region_group": 47, "outliers": []},
        },
    }
