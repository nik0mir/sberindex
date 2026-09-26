"""Чтение БД ПМО Росстата («Если быть точным») и отбор строк показателя (spec_final, А.7).

Порядок обработки строк одного показателя (Л11): фильтр разрезов → выбор периода по приоритету →
дедупликация → проверка уникальности. Годовое значение — только «Январь-декабрь» или запасной период из
конфига; кварталы и одномесячные периоды не берутся никогда (Л12, Л13). ОКТМО читается строкой (ведущий
ноль не теряется), служебные коды ``CD``, ``ND``, ``UD`` → пропуск (Л18).
"""

from __future__ import annotations

import gzip
import logging
import re
from collections.abc import Iterable, Sequence
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pcsv

from munnet.contracts import OKVED_SECTIONS
from munnet.panel.territory import normalize_oktmo

log = logging.getLogger(__name__)

DELIMITER = ";"
TOTAL_DIM = "TOTAL"  # разрез «итог» во всех выходах
TOTAL_LABEL = "Всего"  # итог в возрастных и прочих разрезах БД ПМО
SECTION_RE = re.compile(r"^Раздел\s+(\S)\b")  # «Раздел C Обрабатывающие производства»
ANOMALY_COMMENT = "Аномальное значение показателя"  # комментарий Росстата -> флаг anomaly
AGE_COLUMN = "vozr"  # колонка возраста в Y48112014
KEY = ["oktmo", "year", "dim"]
# Описательные колонки, которые не нужны ни для отбора, ни для аудита: не читаем (память и время, Л27).
SKIP_COLUMNS = (
    "indicator_section_code",
    "indicator_section",
    "indicator_name",
    "mun_district",
    "municipality",
)
OUT_COLUMNS = [
    "oktmo",
    "oktmo_stable",
    "year",
    "dim",
    "value",
    "period_used",
    "flag",
    "unit",
    "source_code",
    "oktmo_history",
    "mun_type",
    "mun_type_oktmo",
]


def join_flags(*parts: pd.Series) -> pd.Series:
    """Склеивает флаги через «;» без пустых и повторов, по алфавиту; пусто — NA. Индекс серий общий."""
    joined = parts[0].fillna("").astype("str")
    for part in parts[1:]:
        joined = joined + ";" + part.fillna("").astype("str")
    norm = {u: ";".join(sorted({f for f in u.split(";") if f})) for u in joined.unique()}
    out = joined.map(norm)
    return out.where(out != "").astype("str")


def _read_level(path: Path, level: str, na_codes: Sequence[str]) -> pa.Table:
    """Все строки файла уровня ``level`` за все годы: колонки строкой, ``na_codes`` и пустое → пропуск."""
    with gzip.open(path, "rt", encoding="utf-8") as f:
        header = f.readline().strip().split(DELIMITER)
    table = pcsv.read_csv(
        path,
        parse_options=pcsv.ParseOptions(delimiter=DELIMITER),
        convert_options=pcsv.ConvertOptions(
            column_types={c: pa.string() for c in header},
            include_columns=[c for c in header if c not in SKIP_COLUMNS],
            strings_can_be_null=True,
            null_values=[""],
        ),
    )
    table = table.filter(pc.equal(table["mun_level"], level))
    na = pa.array(list(na_codes), type=pa.string())
    cols = [
        pc.if_else(pc.is_in(table[n], na), pa.scalar(None, pa.string()), table[n]) for n in table.column_names
    ]
    return pa.table(cols, names=table.column_names)


def source_codes(table: pa.Table, code: str = "oktmo") -> pd.DataFrame:
    """Все коды источника по годам: уникальные (``oktmo``, ``oktmo_stable``, ``year``), коды нормализованы.

    ``code`` — колонка кода в ``table`` (``oktmo`` в БД ПМО, ``object_oktmo`` в 5-НДФЛ). Служебные коды уже
    заменены пропуском. Из этой таблицы берутся пары моста ``stable`` (``stable_pairs``) и множество кодов,
    которые источник знает в каждом году (``crosswalk.match_rows``, флаг ``union_merged``).
    """
    cols = [code, "oktmo_stable", "year"]
    codes = table.select(cols).group_by(cols).aggregate([]).to_pandas()
    codes = codes[codes[code].notna() & codes["year"].notna()]
    out = pd.DataFrame(
        {
            "oktmo": normalize_oktmo(codes[code]),
            "oktmo_stable": normalize_oktmo(codes["oktmo_stable"]),
            "year": codes["year"].astype("int16"),
        }
    )
    return out.drop_duplicates().sort_values(["oktmo", "year"], kind="mergesort").reset_index(drop=True)


def _to_frame(table: pa.Table, years: Sequence[int]) -> pd.DataFrame:
    year_strs = [str(int(y)) for y in years]
    df = table.filter(pc.is_in(table["year"], pa.array(year_strs))).to_pandas()
    df["value"] = pd.to_numeric(df["indicator_value"], errors="raise").astype("float64")
    df["year"] = df["year"].astype("int16")
    df["oktmo"] = normalize_oktmo(df["oktmo"])
    df["oktmo_stable"] = normalize_oktmo(df["oktmo_stable"])
    return df


def read_bdmo(path: Path, level: str, na_codes: Sequence[str], years: Sequence[int]) -> pd.DataFrame:
    """Читает файл БД ПМО: все колонки строкой, только ``mun_level == level`` и годы ``years``.

    Заголовок берётся из первой строки gzip-файла, чтобы задать всем колонкам строковый тип (быстрый
    ``pyarrow.csv`` вместо ``pd.read_csv``, Л27); описательные колонки ``SKIP_COLUMNS`` не читаются. Пустые
    строки и ``na_codes`` → пропуск во всех полях. Добавляются ``value`` (float64 из ``indicator_value``)
    и ``year`` (int16); ``oktmo`` и ``oktmo_stable`` нормализуются к 8 цифрам.
    """
    return _to_frame(_read_level(path, level, na_codes), years)


def read_bdmo_with_codes(
    path: Path, level: str, na_codes: Sequence[str], years: Sequence[int]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Как ``read_bdmo`` плюс ``source_codes`` строк уровня ``level`` за все годы файла.

    Коды собираются до фильтра лет (А.6: пары моста — «из всех прочитанных файлов»): код, который
    встречается только в строках прошлых лет, тоже ведёт к своему стабильному коду.
    """
    table = _read_level(path, level, na_codes)
    return _to_frame(table, years), source_codes(table)


def okved_section(label: str, total: str, letters: dict[str, str]) -> str | None:
    """Разрез ОКВЭД2 → ``TOTAL``, буква раздела латиницей или ``None`` (не раздел: подвиды торговли и т. п.).

    Итог — только точная строка ``total`` («Всего по обследуемым…»): «Всего по отдельным…» — не итог (Л15).
    Кириллические буквы разделов (А, В, Е, Н) переводятся в латиницу по ``letters`` (Л16).
    """
    if not isinstance(label, str):
        return None
    if label == total:
        return TOTAL_DIM
    m = SECTION_RE.match(label)
    if not m:
        return None
    letter = letters.get(m.group(1), m.group(1))
    return letter if letter in OKVED_SECTIONS else None


def _dims(df: pd.DataFrame, spec: dict, cfg_context: dict) -> pd.Series:
    """Разрез каждой строки: ``TOTAL``, буква ОКВЭД2 или код группы; ``None`` — строка не нужна."""
    by = spec.get("by")
    if by is None:
        return pd.Series(TOTAL_DIM, index=df.index, dtype="str")
    if by == "okved2":
        total, letters = cfg_context["okved_total"], cfg_context["okved_letters"]
        cache = {lab: okved_section(lab, total, letters) for lab in df["okved2"].dropna().unique()}
        dim = df["okved2"].map(cache)
    else:
        dim = df[by].map(spec["by_values"])
    keep = spec.get("keep")
    if keep is not None:
        dim = dim.where(dim.isin(keep))
    return dim.astype("str")


def select(df: pd.DataFrame, spec: dict, cfg_context: dict) -> pd.DataFrame:
    """Строки одного показателя: ``oktmo, oktmo_stable, year, dim, value, period_used, flag`` и аудит.

    ``spec`` — запись ``context.indicators`` (``dims``, ``by``, ``keep``, ``periods``, ``years``); для
    разрезов не из ОКВЭД2 — ``by`` (колонка) и ``by_values`` (метка → код). Порядок (А.7, п. 4): фильтр
    разрезов и годов, пустые значения долой → период по приоритету ``periods`` (прочие периоды не берутся;
    не первый по списку — флаг ``fallback_period``) → дедупликация по (oktmo, year, dim): одинаковые
    значения схлопываются, приоритетнее строка с ``mun_type == mun_type_oktmo``, если значения всё равно
    разные — первая, флаг ``dup_conflict`` → проверка уникальности ключа. Комментарий «Аномальное значение
    показателя» — флаг ``anomaly``, значение сохраняется.
    """
    rows = df[df["year"].isin([int(y) for y in spec["years"]])]
    for col, val in (spec.get("dims") or {}).items():
        rows = rows[rows[col] == val]
    rows = rows.assign(dim=_dims(rows, spec, cfg_context))
    rows = rows[rows["dim"].notna() & rows["value"].notna() & rows["oktmo"].notna()]
    periods = list(spec["periods"])
    rank = rows["indicator_period"].map({p: i for i, p in enumerate(periods)})
    rows = rows.assign(_rank=rank)[rank.notna()]
    rows = rows.assign(_order=range(len(rows)))
    best_rank = rows.groupby(KEY)["_rank"].transform("min")
    rows = rows[rows["_rank"] == best_rank]

    same_type = rows["mun_type"].fillna("") == rows["mun_type_oktmo"].fillna("#")
    rows = rows.assign(_pri=(~same_type).astype("int8"))
    best_pri = rows.groupby(KEY)["_pri"].transform("min")
    rows = rows[rows["_pri"] == best_pri].sort_values([*KEY, "_order"], kind="mergesort")
    n_values = rows.groupby(KEY)["value"].transform("nunique")
    rows = rows.assign(_conflict=n_values > 1)
    out = rows.drop_duplicates(subset=KEY, keep="first").copy()

    fallback = pd.Series("", index=out.index).where(out["_rank"] == 0, "fallback_period")
    conflict = pd.Series("", index=out.index).where(~out["_conflict"], "dup_conflict")
    comment = out["comment"].fillna("") if "comment" in out else pd.Series("", index=out.index)
    anomaly = pd.Series("", index=out.index).where(
        ~comment.str.contains(ANOMALY_COMMENT, regex=False), "anomaly"
    )
    out["flag"] = join_flags(fallback, conflict, anomaly)
    out["period_used"] = out["indicator_period"].astype("str")
    out["unit"] = out["indicator_unit"].astype("str")
    out["source_code"] = out["indicator_code"].astype("str")
    for col in ("oktmo_history", "mun_type", "mun_type_oktmo"):
        if col not in out:
            out[col] = pd.Series(pd.NA, index=out.index, dtype="str")
    out = out[OUT_COLUMNS].sort_values(KEY, kind="mergesort").reset_index(drop=True)
    dup = out.duplicated(subset=KEY, keep=False)
    if dup.any():  # проверка после дедупликации: сюда код попасть не должен
        raise AssertionError(f"после дедупликации остались дубли ключа {KEY}: {out[dup].head()}")
    return out


def age_spec(cfg_context: dict, year: int) -> dict:
    """Спецификация ``select`` для возрастных групп года из ``context.age`` (три группы и итог)."""
    age = cfg_context["age"]
    by_values = {label: code for code, label in age["groups"].items()}
    by_values[TOTAL_LABEL] = TOTAL_DIM
    return {
        "dims": dict(age["dims"]),
        "by": AGE_COLUMN,
        "by_values": by_values,
        "periods": [age["period"]],
        "years": [int(year)],
    }


def check_age(rows: pd.DataFrame, groups: Iterable[str], tolerance: float) -> pd.DataFrame:
    """Возрастные группы: сумма трёх групп против итога «Всего» в пределах ``tolerance`` (Л19).

    Строкам групп, где сумма расходится с итогом больше чем на ``tolerance`` или итога нет, добавляется
    флаг ``age_sum_mismatch``. Возвращает только строки групп.
    """
    groups = list(groups)
    part = rows[rows["dim"].isin(groups)].copy()
    total = rows[rows["dim"] == TOTAL_DIM].set_index(["oktmo", "year"])["value"]
    sums = part.groupby(["oktmo", "year"])["value"].transform("sum")
    idx = pd.MultiIndex.from_frame(part[["oktmo", "year"]])
    ref = total.reindex(idx).to_numpy()
    bad = pd.Series(~(abs(sums.to_numpy() - ref) <= tolerance * ref), index=part.index)
    mark = pd.Series("", index=part.index).where(~bad, "age_sum_mismatch")
    part["flag"] = join_flags(part["flag"], mark)
    return part.reset_index(drop=True)


def stable_pairs(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """Пары (``oktmo``, ``oktmo_stable``) из всех кодов источника: мост шага ``stable`` (А.6).

    Код, у которого в источнике два разных стабильных кода, исключается и пишется в лог.
    """
    parts = [f[["oktmo", "oktmo_stable"]].dropna().drop_duplicates() for f in frames]
    if not parts:
        return pd.DataFrame({"oktmo": pd.Series(dtype="str"), "oktmo_stable": pd.Series(dtype="str")})
    pairs = pd.concat(parts, ignore_index=True).drop_duplicates()
    multi = pairs["oktmo"].duplicated(keep=False)
    if multi.any():
        log.warning(
            "мост oktmo_stable: у %d кодов несколько стабильных, исключены: %s",
            pairs.loc[multi, "oktmo"].nunique(),
            sorted(pairs.loc[multi, "oktmo"].unique())[:10],
        )
        pairs = pairs[~multi]
    return pairs.sort_values(["oktmo", "oktmo_stable"]).reset_index(drop=True)
