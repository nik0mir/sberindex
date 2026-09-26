"""Соединение строк Росстата и ФНС с МО справочника по ОКТМО: три шага по срезу на год (spec_final, А.6).

Для каждого года Y берётся срез справочника S (все МО среза, не только панель) и множество его «прямых»
кодов D. МО получает строки одного кода источника — все разрезы этого кода:

1. ``direct``  — строки с кодом МО в срезе;
2. ``version`` — код другой версии того же МО, если он не прямой код другого МО среза. Если таких версий
   со строками несколько, берётся ближайшая к Y по интервалу действия (при равенстве — более поздняя), а не
   первая по году, как в А.6: так при сдвиге «на 1 января Y» под кодом Y − 1 (Л4) берётся код Y − 1, а не
   код давней версии;
3. ``stable``  — мост через ``oktmo_stable`` источника: стабильный код, который достаётся двум и более МО
   среза, запрещён (объединение, Л9); строки с прямым кодом другого МО не забираются; один кандидат или
   одинаковые значения → берём; иначе строка, чей код сам входит в стабильные коды МО, иначе ``multi_row``.

Прямой код важнее моста, значения частей после объединений не суммируются (Л10). Граничный случай
объединения: источник уже записал объединённое МО под кодом одной из частей, а коды другой части за этот
год в источнике не встречаются совсем (5-НДФЛ 2023 года: Павлово-Посадский округ под кодом Павловского
Посада, Электрогорска нет). Значение остаётся у части с прямым кодом, но получает флаг ``union_merged``:
оно уже за объединённое МО и на жителя части не делится. Проверке нужен ``present`` — все коды источника
по годам; без него шаг не выполняется.

У МО без значения — причина: ``region_no_rows`` (у субъекта нет строк за год), ``union_banned`` (мост
запрещён объединением или значение поглощено объединённым МО), ``multi_row``, ``no_code`` (ни один код МО
и мост не нашли строк).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
import pandas as pd

from munnet.panel.bdmo import join_flags
from munnet.panel.territory import dictionary_slice, region_prefix

log = logging.getLogger(__name__)

REASONS = ("region_no_rows", "union_banned", "multi_row", "no_code")
UNION_MERGED = "union_merged"  # флаг значения, которое источник уже дал за объединённое МО
BASE_COLUMNS = [
    "territory_id",
    "year",
    "dim",
    "value",
    "oktmo_used",
    "method",
    "flag",
    "period_used",
    "reason",
]


def _version_distance(year_from: pd.Series, year_to: pd.Series, year: int) -> pd.Series:
    """Сколько лет от года ``year`` до интервала версии ``[year_from; year_to)``: 0, если год внутри."""
    last = year_to.astype("int64") - 1
    first = year_from.astype("int64")
    before = (first - year).clip(lower=0)
    after = (year - last).clip(lower=0)
    return before + after


def _signature(rows: pd.DataFrame) -> dict[str, tuple]:
    """Код → отсортированный набор (разрез, значение): чтобы понять, различаются ли кандидаты моста."""
    sig = {}
    for code, g in rows.groupby("oktmo", sort=True):
        sig[code] = tuple(sorted(zip(g["dim"].astype(str), g["value"].astype(float), strict=True)))
    return sig


def _merged_parts(
    assigned: pd.DataFrame,
    slice_st: pd.DataFrame,
    banned: set,
    stable_of: pd.Series,
    codes_of: pd.Series,
    present_y: set | None,
) -> tuple[set, set]:
    """Части объединения, которые источник уже записал под кодом другой части (флаг ``union_merged``).

    МО ``i`` с прямым кодом (или кодом другой версии), чей стабильный код запрещён, и другое МО ``j`` среза
    с тем же стабильным кодом без значения, ни один код которого за год в источнике не встречается
    (``present_y``): значение ``i`` уже за объединённое МО. Возвращает (МО ``i``, МО ``j``).
    """
    if present_y is None or not banned or assigned.empty:
        return set(), set()
    have = set(assigned["territory_id"])
    st = assigned["oktmo_used"].map(stable_of)
    hit = assigned[st.isin(banned) & assigned["method"].isin(["direct", "version"])]
    merged, absorbed = set(), set()
    for tid, code in zip(hit["territory_id"], hit["oktmo_used"], strict=True):
        others = set(slice_st.loc[slice_st["stable"] == stable_of[code], "territory_id"]) - {tid}
        gone = {j for j in others if j not in have and not (codes_of.get(j, set()) & present_y)}
        if gone:
            merged.add(tid)
            absorbed |= gone
    return merged, absorbed


def _match_year(
    rows_y: pd.DataFrame,
    sl: pd.DataFrame,
    versions: pd.DataFrame,
    mo_stables: pd.DataFrame,
    year: int,
    stable_of: pd.Series,
    present_y: set | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Один год: назначения (``territory_id, oktmo_used, method, merged``) и потери (``reason``)."""
    direct_codes = set(sl["oktmo8"])
    codes_with_rows = set(rows_y["oktmo"])

    # 1. Прямой код среза.
    direct = sl[sl["oktmo8"].isin(codes_with_rows)]
    assigned = pd.DataFrame(
        {"territory_id": direct["territory_id"], "oktmo_used": direct["oktmo8"], "method": "direct"}
    )
    rest = sl[~sl["territory_id"].isin(assigned["territory_id"])]

    # 2. Код другой версии того же МО, если он не прямой код другого МО среза.
    other = versions[
        versions["territory_id"].isin(rest["territory_id"]) & versions["oktmo8"].isin(codes_with_rows)
    ]
    blocked = other["oktmo8"].isin(direct_codes)
    if blocked.any():
        log.info(
            "год %d: шаг version заблокирован у %d кодов (прямые коды других МО)", year, int(blocked.sum())
        )
    other = other[~blocked].assign(_dist=lambda d: _version_distance(d["year_from"], d["year_to"], year))
    other = other.sort_values(["territory_id", "_dist", "year_from"], ascending=[True, True, False])
    version = other.drop_duplicates("territory_id")
    assigned = pd.concat(
        [
            assigned,
            pd.DataFrame(
                {
                    "territory_id": version["territory_id"],
                    "oktmo_used": version["oktmo8"],
                    "method": "version",
                }
            ),
        ],
        ignore_index=True,
    )
    rest = rest[~rest["territory_id"].isin(assigned["territory_id"])]

    # 3. Мост через oktmo_stable источника с запретом объединений.
    slice_st = mo_stables[mo_stables["territory_id"].isin(sl["territory_id"])]
    owners = slice_st.groupby("stable")["territory_id"].nunique()
    banned = set(owners[owners >= 2].index)
    bridge = rows_y.loc[
        ~rows_y["oktmo"].isin(direct_codes) & rows_y["oktmo_stable"].notna(), ["oktmo", "oktmo_stable"]
    ].drop_duplicates()
    my_st = slice_st[slice_st["territory_id"].isin(rest["territory_id"])]
    cand_all = my_st.merge(bridge, left_on="stable", right_on="oktmo_stable", how="inner")
    cand = cand_all[~cand_all["stable"].isin(banned)]
    banned_hit = set(cand_all.loc[cand_all["stable"].isin(banned), "territory_id"]) - set(
        cand["territory_id"]
    )

    stable_rows, multi = [], []
    for tid, g in cand.groupby("territory_id", sort=True):
        codes = sorted(g["oktmo"].unique())
        if len(codes) > 1:
            sig = _signature(rows_y[rows_y["oktmo"].isin(codes)])
            if len(set(sig.values())) > 1:
                own = sorted(set(codes) & set(my_st.loc[my_st["territory_id"] == tid, "stable"]))
                if len(own) != 1:
                    multi.append(tid)
                    continue
                codes = own
        stable_rows.append((tid, codes[0]))
    if stable_rows:
        st = pd.DataFrame(stable_rows, columns=["territory_id", "oktmo_used"]).assign(method="stable")
        assigned = pd.concat([assigned, st], ignore_index=True)
    rest = rest[~rest["territory_id"].isin(assigned["territory_id"])]

    # 4. Объединение под кодом одной из частей: значение остаётся, но с флагом union_merged.
    codes_of = versions.groupby("territory_id")["oktmo8"].agg(set)
    merged, absorbed = _merged_parts(assigned, slice_st, banned, stable_of, codes_of, present_y)
    if merged:
        log.info("год %d: значение уже за объединённое МО (union_merged) у МО %s", year, sorted(merged))
    assigned["merged"] = assigned["territory_id"].isin(merged)

    shared = assigned["oktmo_used"].duplicated(keep=False)
    if shared.any():
        log.warning(
            "год %d: один код источника назначен нескольким МО: %s",
            year,
            assigned[shared].sort_values("oktmo_used").head(10).to_dict("records"),
        )

    # Причины потерь.
    prefixes = set(region_prefix(rows_y["oktmo"]))
    reason = pd.Series("no_code", index=rest.index, dtype="str")
    reason[rest["territory_id"].isin(banned_hit | absorbed)] = "union_banned"
    reason[rest["territory_id"].isin(multi)] = "multi_row"
    reason[~region_prefix(rest["oktmo8"]).isin(prefixes)] = "region_no_rows"
    lost = pd.DataFrame({"territory_id": rest["territory_id"], "reason": reason})
    return assigned, lost


def match_rows(
    rows: pd.DataFrame,
    dic: pd.DataFrame,
    pairs: pd.DataFrame,
    years: Sequence[int],
    last_year: int,
    *,
    present: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Сопоставляет строки показателя с МО справочника за каждый год ``years`` (А.6).

    ``rows`` — выход ``bdmo.select`` или ``ndfl.read_ndfl`` (``oktmo, oktmo_stable, year, dim, value,
    period_used, flag`` и колонки аудита); ``dic`` — все версии справочника (``read_dictionary``);
    ``pairs`` — пары (``oktmo``, ``oktmo_stable``) источника; ``last_year`` — последний год справочника;
    ``present`` — все коды источника по годам (``oktmo, year``, любые показатели и разрезы): по нему
    видно, что часть объединения за год в источнике уже не встречается (флаг ``union_merged``); без него
    этот шаг не выполняется.

    Выход — все МО срезов: у сопоставленных по строке на разрез (``territory_id, year, dim, value,
    oktmo_used, method, flag, period_used`` и колонки аудита из ``rows``), у остальных одна строка с
    ``value`` = NA и причиной ``reason``.
    """
    extras = [c for c in rows.columns if c not in {"oktmo", "year", "dim", "value", "period_used", "flag"}]
    versions = dic[["territory_id", "oktmo8", "year_from", "year_to"]].dropna(subset=["oktmo8"])
    stable_of = pairs.drop_duplicates("oktmo").set_index("oktmo")["oktmo_stable"]
    mo_stables = (
        versions.assign(stable=versions["oktmo8"].map(stable_of).astype("str"))
        .dropna(subset=["stable"])[["territory_id", "stable"]]
        .drop_duplicates()
    )
    present_by_year = None
    if present is not None:
        present_by_year = present.groupby(present["year"].astype("int64"))["oktmo"].agg(set).to_dict()
    out = []
    for year in years:
        year = int(year)
        sl = dictionary_slice(dic, year, last_year)[["territory_id", "oktmo8"]]
        rows_y = rows[rows["year"] == year]
        present_y = None if present_by_year is None else present_by_year.get(year, set())
        assigned, lost = _match_year(rows_y, sl, versions, mo_stables, year, stable_of, present_y)
        got = assigned.merge(rows_y, left_on="oktmo_used", right_on="oktmo", how="inner")
        mark = pd.Series("", index=got.index).where(~got["merged"].astype(bool), UNION_MERGED)
        got["flag"] = join_flags(got["flag"], mark)
        got = got.drop(columns="merged").assign(reason=pd.Series(pd.NA, index=got.index, dtype="str"))
        lost = lost.assign(year=year, method=pd.Series(pd.NA, index=lost.index, dtype="str"))
        out.extend([got, lost])
    res = pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=BASE_COLUMNS + extras)
    for col in BASE_COLUMNS + extras:
        if col not in res:
            res[col] = pd.NA
    res = res[BASE_COLUMNS + extras].copy()
    res["territory_id"] = res["territory_id"].astype("int32")
    res["year"] = res["year"].astype("int16")
    res["value"] = res["value"].astype("float64")
    for col in ("dim", "oktmo_used", "method", "flag", "period_used", "reason"):
        res[col] = res[col].astype("str")
    order = np.lexsort(
        (res["dim"].fillna("").to_numpy(), res["territory_id"].to_numpy(), res["year"].to_numpy())
    )
    return res.iloc[order].reset_index(drop=True)
