"""5-НДФЛ ФНС по муниципалитетам («Если быть точным»): доход и число получателей (spec_final, А.8).

Файл — 34 млн строк, поэтому читается только ``pyarrow.dataset`` с фильтром до чтения (Л27). Доход —
``Y777000006`` «Сумма дохода, полученная физическими лицами» со ставкой «Всего», а не ``Y777000030``: тот на
уровне МО почти везде 0 (Л1). Только ``tax_authority`` = «Всего»: строки инспекций других регионов дублируют
ключ (Л2). 5-НДФЛ считается по месту налогового агента (работодателя), а не жительства (Л3).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds

from munnet.panel.bdmo import KEY, OUT_COLUMNS, TOTAL_DIM, join_flags, source_codes, stable_pairs
from munnet.panel.territory import normalize_oktmo

log = logging.getLogger(__name__)

READ_COLUMNS = [
    "indicator_code",
    "indicator_unit",
    "object_oktmo",
    "oktmo_stable",
    "oktmo_history",
    "year",
    "indicator_period",
    "indicator_value",
]
NA_CODES = ("CD", "ND", "UD")  # служебные коды «Если быть точным» по умолчанию (context.na_codes, Л18)
INDICATORS = ("income", "recipients")  # ключи context.ndfl -> ndfl_<ключ>


def _filter(spec: dict, name: str):
    """Фильтр до чтения: верхний уровень МО, годы, итог по инспекциям и поля показателя ``spec[name]``."""
    base = (
        (ds.field("object_level") == spec["object_level"])
        & ds.field("year").isin([int(y) for y in spec["years"]])
        & (ds.field("tax_authority") == spec["tax_authority"])
    )
    for col, val in spec[name].items():
        base = base & (ds.field(col) == val)
    return base


def _rows(table: pd.DataFrame, na_codes: Sequence[str]) -> pd.DataFrame:
    """Строки 5-НДФЛ в формате ``bdmo.select``: ноль и пропуск удаляются до дедупликации."""
    df = table.copy()
    for col in ("object_oktmo", "oktmo_stable", "oktmo_history"):
        df[col] = df[col].where(~df[col].isin(list(na_codes)))
    df["oktmo"] = normalize_oktmo(df["object_oktmo"])
    df["oktmo_stable"] = normalize_oktmo(df["oktmo_stable"])
    df["year"] = df["year"].astype("int16")
    df["value"] = df["indicator_value"].astype("float64")
    df["dim"] = TOTAL_DIM
    df = df[df["oktmo"].notna() & df["value"].notna() & (df["value"] > 0)]
    df = df.assign(_order=range(len(df))).sort_values([*KEY, "_order"], kind="mergesort")
    n_values = df.groupby(KEY)["value"].transform("nunique")
    conflict = pd.Series("", index=df.index).where(n_values <= 1, "dup_conflict")
    df["flag"] = join_flags(conflict)
    out = df.drop_duplicates(subset=KEY, keep="first").copy()
    out["period_used"] = out["indicator_period"].astype("str")
    out["unit"] = out["indicator_unit"].astype("str")
    out["source_code"] = out["indicator_code"].astype("str")
    out["mun_type"] = pd.Series(pd.NA, index=out.index, dtype="str")
    out["mun_type_oktmo"] = pd.Series(pd.NA, index=out.index, dtype="str")
    return out[OUT_COLUMNS].sort_values(KEY, kind="mergesort").reset_index(drop=True)


def _codes(dset: ds.Dataset, spec: dict, na_codes: Sequence[str]) -> pd.DataFrame:
    """``bdmo.source_codes`` всех строк МО верхнего уровня файла: все годы и все показатели.

    Старый код МО, сменившего тип и название, встречается только в строках прошлых лет: если брать пары
    лишь из прочитанных строк двух показателей за нужные годы, шаг ``stable`` не находит его новый код.
    """
    flt = ds.field("object_level") == spec["object_level"]
    table = dset.to_table(columns=["object_oktmo", "oktmo_stable", "year"], filter=flt)
    na = pa.array(list(na_codes), type=pa.string())
    cols = [
        pc.if_else(pc.is_in(table[n], na), pa.scalar(None, pa.string()), table[n])
        for n in ("object_oktmo", "oktmo_stable")
    ]
    table = pa.table([*cols, table["year"]], names=["object_oktmo", "oktmo_stable", "year"])
    return source_codes(table, code="object_oktmo")


def read_ndfl(path: Path, spec: dict, na_codes: Sequence[str] = NA_CODES) -> dict[str, pd.DataFrame]:
    """Читает доход и получателей 5-НДФЛ по ``context.ndfl`` и пары моста ``stable`` этого файла.

    Возвращает ``{"ndfl_income": строки, "ndfl_recipients": строки, "pairs": пары, "present": коды}``;
    ``present`` — все коды файла по годам (``oktmo, year``) для ``match_rows``; строки — в формате
    ``bdmo.select`` (``dim`` = ``TOTAL``). Значение ≤ 0 или пустое удаляется до дедупликации (ноль дохода при
    ненулевых получателях — пропуск, а не ноль); дубли ключа с разными значениями — первая строка и флаг
    ``dup_conflict``. Пары (``oktmo``, ``oktmo_stable``) — из всех строк МО верхнего уровня файла за все годы
    и по всем показателям (А.6: «для 5-НДФЛ — его собственные пары»). ``na_codes`` — служебные коды, которые
    читаются как пропуск (``context.na_codes``).
    """
    dset = ds.dataset(path)
    out = {}
    for name in INDICATORS:
        frame = dset.to_table(columns=READ_COLUMNS, filter=_filter(spec, name)).to_pandas()
        out[f"ndfl_{name}"] = _rows(frame, na_codes)
    codes = _codes(dset, spec, na_codes)
    out["pairs"] = stable_pairs([codes])
    out["present"] = codes[["oktmo", "year"]].drop_duplicates().reset_index(drop=True)
    for name in INDICATORS:
        log.info("5-НДФЛ %s: %d строк МО-год", name, len(out[f"ndfl_{name}"]))
    log.info("5-НДФЛ: %d пар моста oktmo_stable", len(out["pairs"]))
    return out


CITY_READ_COLUMNS = [*READ_COLUMNS, "report_type"]


def read_city_rows(
    path: Path, spec: dict, cities: dict[int, str], na_codes: Sequence[str] = NA_CODES
) -> pd.DataFrame:
    """Доход и получатели 5-НДФЛ городов федерального значения целиком (узлы-города ``munnet.nodes``).

    ``cities`` — код субъекта -> ОКТМО города (``nodes.cities``). Строки — уровень и отчёт
    ``spec["city_rows"]`` («Субъект РФ», «Свод»): отчёт «МО» с тем же ключом у Москвы — другой отчёт (единицы
    получателей), его брать нельзя. Фильтры показателей и инспекции — те же, что у ``read_ndfl``; ноль
    и пропуск удаляются, дубль ключа с разными значениями — первая строка и флаг ``dup_conflict``. Город без
    строки «Свод» в результат не попадает (у Петербурга её нет). Колонки — схема ``contracts.CITY_CONTEXT``.
    """
    city = spec["city_rows"]
    by_oktmo = {str(v): int(k) for k, v in cities.items()}
    dset = ds.dataset(path)
    parts = []
    for name in INDICATORS:
        flt = (
            (ds.field("object_level") == city["object_level"])
            & (ds.field("report_type") == city["report_type"])
            & ds.field("object_oktmo").isin(list(by_oktmo))
            & ds.field("year").isin([int(y) for y in spec["years"]])
            & (ds.field("tax_authority") == spec["tax_authority"])
        )
        for col, val in spec[name].items():
            flt = flt & (ds.field(col) == val)
        frame = dset.to_table(columns=CITY_READ_COLUMNS, filter=flt).to_pandas()
        rows = _rows(frame, na_codes)
        rows["report_type"] = city["report_type"]
        parts.append(rows.assign(indicator=f"ndfl_{name}"))
    out = pd.concat(parts, ignore_index=True)
    out["region_code"] = out["oktmo"].map(by_oktmo).astype("int16")
    out["year"] = out["year"].astype("int16")
    out = out[["region_code", "year", "indicator", "value", "oktmo", "source_code", "report_type", "flag"]]
    log.info("5-НДФЛ, города целиком: %d строк (%s)", len(out), ", ".join(sorted(set(out["oktmo"]))))
    return out.sort_values(["region_code", "year", "indicator"], kind="mergesort").reset_index(drop=True)
