"""Годовой контекст МО панели из БД ПМО Росстата и 5-НДФЛ ФНС (spec_final, А.4, А.7–А.10).

Каждое значение сопоставляется с МО по алгоритму ``crosswalk.match_rows`` и хранится в ``context_long`` с
происхождением (код строки, метод, период, флаги). ``context_annual`` — МО × год одной строкой: население,
структура занятости, обороты и показатели «на жителя» в ₽ в месяц, чтобы стоять на одной оси с тратами.
Траты СберИндекса уже «на жителя» и на население не делятся; годовые потоки Росстата и ФНС делятся на
среднегодовое население и на 12 (А.9). Показатели по месту работы (занятость, ФОТ, обороты, 5-НДФЛ) у
внутригородских территорий Москвы и Петербурга на жителя не считаются (Л3).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from munnet.config import Config
from munnet.contracts import OKVED_SECTIONS
from munnet.panel.bdmo import (
    TOTAL_DIM,
    age_spec,
    check_age,
    join_flags,
    read_bdmo_with_codes,
    select,
    stable_pairs,
)
from munnet.panel.crosswalk import UNION_MERGED, match_rows
from munnet.panel.ndfl import read_ndfl
from munnet.panel.territory import dictionary_slice

log = logging.getLogger(__name__)

KRUB_TO_RUB = 1000.0  # тыс. ₽ → ₽
MONTHS_PER_YEAR = 12.0  # годовой поток → ₽ в месяц
PER_THOUSAND = 1000.0  # «на 1000 жителей»
POPULATION = "population"  # показатель населения «на 1 января» в context.indicators
URBAN, RURAL = "population_urban", "population_rural"
AGE = "age"
NDFL_INCOME, NDFL_RECIPIENTS = "ndfl_income", "ndfl_recipients"
EMPLOYEES, SHIPMENTS = "employees", "shipments"
MINING_SECTION = "B"  # добыча полезных ископаемых: доля в отгрузке
UNALLOCATED = "unallocated"
NOT_IN_SLICE = "not_in_slice"  # причина потери: МО нет в срезе справочника этого года
NONPOSITIVE = "nonpositive"  # причина потери: население ≤ 0 и перенести нечего
# Допуски по умолчанию, если в конфиге нет ключей (spec_final, А.7, п. 6 и А.4, emp_sh_unallocated).
AGE_SUM_TOLERANCE = 0.01  # context.age.sum_tolerance: сумма трёх групп против «Всего»
UNALLOCATED_TOLERANCE = 0.01  # context.unallocated_tolerance: сумма раскрытых долей занятости выше 1
SHARE_EPS = 1e-9  # перебор суммы долей меньше этого — шум округления, а не перебор источника
# Дефект 5-НДФЛ: в строке получателей «Всего» единицы при нормальном доходе (у ~30 МО в 2023 году). Скачок
# получателей год к году не меньше чем в RECIPIENTS_JUMP раз при доходе, изменившемся не больше чем
# в INCOME_STABLE раз, помечает меньшее значение (context.ndfl.recipients_jump, context.ndfl.income_stable).
RECIPIENTS_JUMP = 10.0
INCOME_STABLE = (0.5, 2.0)
RECIPIENTS_JUMP_FLAG = "recipients_jump"
# Значения с этими флагами остаются в context_long для аудита, но не входят в context_annual.
EXCLUDED_FLAGS = (UNION_MERGED, RECIPIENTS_JUMP_FLAG)

# Показатель context.indicators (итог) -> колонка context_annual.
ANNUAL_TOTALS = {
    "employees": "employees",
    "payroll": "payroll_krub",
    "wage": "wage",
    "retail": "retail_krub",
    "catering_turnover": "catering_turnover_krub",
    "shipments": "shipments_krub",
    "orgs": "orgs",
    "ip": "ip",
    "nights": "nights",
    "beds": "beds",
    "invest": "invest_krub",
    NDFL_INCOME: "ndfl_income_rub",
    NDFL_RECIPIENTS: "ndfl_recipients",
}
# Потоки в тыс. ₽ в год -> «на жителя» в ₽ в месяц (NA при workplace_based).
PER_CAPITA_KRUB = {
    "payroll_pc": "payroll_krub",
    "retail_pc": "retail_krub",
    "catering_turnover_pc": "catering_turnover_krub",
    "shipments_pc": "shipments_krub",
}
LONG_COLUMNS = [
    "territory_id",
    "year",
    "indicator",
    "dim",
    "value",
    "unit",
    "source_code",
    "period_used",
    "oktmo_used",
    "method",
    "flag",
]
ROWS_COLUMNS = [
    "indicator",
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


@dataclass
class ContextResult:
    """Все выходы контекста: таблицы ``data/processed``, потери и строки источников для аудита."""

    context_long: pd.DataFrame
    context_annual: pd.DataFrame
    okved_shares: pd.DataFrame
    unmatched: pd.DataFrame
    bdmo_rows: pd.DataFrame
    ndfl_rows: pd.DataFrame
    info: dict


# --- Население ----------------------------------------------------------------------------------


def population(
    matched: pd.DataFrame, dic: pd.DataFrame, last_year: int, carry_years: Sequence[int] | None = None
) -> pd.DataFrame:
    """Население «на 1 января» после правил А.9: ≤ 0 → пропуск, пропуск → перенос прошлого года.

    ``matched`` — выход ``match_rows`` для населения (``dim`` = ``TOTAL``, все МО срезов). Значение ≤ 0
    становится пропуском с флагом ``zero_replaced`` (Краснинский район, Л7). Перенос (``method = carried``):
    только в годы ``carry_years`` (по умолчанию — все годы, кроме первого), только с предыдущего года и
    только наблюдённого (не перенесённого) значения, только для МО, которые есть в срезе справочника этого
    года (Рязанская область 2024, Л6). Выход — строки МО срезов: сопоставленные и оставшиеся без значения
    (``value`` = NA, ``reason``).
    """
    pop = matched[matched["dim"].isna() | (matched["dim"] == TOTAL_DIM)].copy()
    zero = pop["value"].notna() & (pop["value"] <= 0)
    pop["flag"] = join_flags(pop["flag"], pd.Series("", index=pop.index).where(~zero, "zero_replaced"))
    pop.loc[zero, "value"] = np.nan
    pop.loc[zero, "reason"] = NONPOSITIVE
    pop["dim"] = TOTAL_DIM
    years = sorted(int(y) for y in pop["year"].unique())
    if carry_years is None:
        carry_years = years[1:]
    for y in sorted(int(v) for v in carry_years):
        in_slice = set(dictionary_slice(dic, y, last_year)["territory_id"])
        prev = pop[(pop["year"] == y - 1) & pop["value"].notna() & (pop["method"] != "carried")]
        prev = prev.drop_duplicates("territory_id").set_index("territory_id")
        cur = pop["year"] == y
        have = set(pop.loc[cur & pop["value"].notna(), "territory_id"])
        need = sorted(t for t in in_slice if t not in have and t in prev.index)
        if not need:
            continue
        src = prev.loc[need].reset_index()
        old = pop[cur & pop["territory_id"].isin(need)].set_index("territory_id")
        was_zero = old["flag"].fillna("").str.contains("zero_replaced").reindex(need).fillna(False).to_numpy()
        carried = src.assign(
            year=np.int16(y),
            method="carried",
            reason=pd.Series(pd.NA, index=src.index, dtype="str"),
            flag=join_flags(src["flag"], pd.Series(np.where(was_zero, "zero_replaced", ""), index=src.index)),
        )
        pop = pd.concat(
            [pop[~(cur & pop["territory_id"].isin(need))], carried[pop.columns]], ignore_index=True
        )
        log.info("население %d: перенесено с %d года у %d МО", y, y - 1, len(need))
    pop["year"] = pop["year"].astype("int16")
    pop["territory_id"] = pop["territory_id"].astype("int32")
    return pop.sort_values(["year", "territory_id"], kind="mergesort").reset_index(drop=True)


# --- Отбор и сопоставление ----------------------------------------------------------------------


def _bdmo_path(cfg: Config, file: str) -> Path:
    return Path(cfg["paths"]["raw"]) / cfg["context"]["bdmo_dir"] / f"{file}.csv.gz"


def _select_all(cfg: Config) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    """Читает каждый файл БД ПМО один раз, отбирает строки всех его показателей и собирает коды источника.

    Возвращает строки показателей, пары моста (``oktmo``, ``oktmo_stable``) и все коды БД ПМО по годам
    (``oktmo, year``) — из всех строк верхнего уровня всех файлов за все годы (А.6).
    """
    cx = cfg["context"]
    by_file: dict[str, list[tuple[str, dict]]] = {}
    for name, spec in cx["indicators"].items():
        by_file.setdefault(spec["file"], []).append((name, spec))
    for year, file in cx["age"]["files"].items():
        by_file.setdefault(file, []).append((AGE, age_spec(cx, int(year))))
    selected: dict[str, list[pd.DataFrame]] = {}
    code_frames = []
    for file, specs in by_file.items():
        path = _bdmo_path(cfg, file)
        frame, codes = read_bdmo_with_codes(path, cx["level"], cx["na_codes"], cx["read_years"])
        code_frames.append(codes)
        for name, spec in specs:
            rows = select(frame, spec, cx)
            if name == AGE:
                tol = cx["age"].get("sum_tolerance", AGE_SUM_TOLERANCE)
                rows = check_age(rows, cx["age"]["groups"].keys(), tol)
            selected.setdefault(name, []).append(rows)
        log.info("БД ПМО %s: %d строк верхнего уровня", file, len(frame))
        del frame
    rows = {name: pd.concat(parts, ignore_index=True) for name, parts in selected.items()}
    codes = pd.concat(code_frames, ignore_index=True)
    present = codes[["oktmo", "year"]].drop_duplicates().reset_index(drop=True)
    return rows, stable_pairs(code_frames), present


def _indicator_years(cfg: Config, name: str) -> list[int]:
    cx = cfg["context"]
    if name == AGE:
        return sorted(int(y) for y in cx["age"]["files"])
    if name in (NDFL_INCOME, NDFL_RECIPIENTS):
        return sorted(int(y) for y in cx["ndfl"]["years"])
    return sorted(int(y) for y in cx["indicators"][name]["years"])


def _to_long(matched: pd.DataFrame, name: str, panel_ids: pd.Index) -> pd.DataFrame:
    got = matched[matched["value"].notna() & matched["territory_id"].isin(panel_ids)]
    out = got[[c for c in LONG_COLUMNS if c != "indicator"]].copy()
    out.insert(2, "indicator", name)
    return out


def flag_recipient_jumps(
    income: pd.DataFrame,
    recipients: pd.DataFrame,
    jump: float = RECIPIENTS_JUMP,
    income_band: Sequence[float] = INCOME_STABLE,
) -> pd.DataFrame:
    """Флаг ``recipients_jump`` у получателей 5-НДФЛ с дефектом источника: единицы вместо тысяч.

    ``income``, ``recipients`` — выходы ``match_rows`` (``territory_id, year, dim, value, flag``). Кандидат —
    пара соседних лет одного МО, где получателей больше или меньше не меньше чем в ``jump`` раз, а доход
    изменился в пределах ``income_band`` раз. Помечается меньшее значение получателей, если доход на
    получателя в его году дальше от медианы года (в логарифмах), чем в году с большим значением: так
    выглядит дефект (Тосненский район, 2023 год: 5 получателей при доходе 2,86·10¹⁰ ₽, в 2024 году —
    66 705). Обратный случай — получателей в сотни тысяч больше при почти том же доходе (налоговый агент
    с выплатами жителям всей страны, например в Кольцове в 2024 году) — не дефект единиц и не помечается.
    Возвращает копию ``recipients`` с дополненной колонкой ``flag``.
    """
    key = ["territory_id", "year"]

    def total(df: pd.DataFrame) -> pd.Series:
        sel = df[df["value"].notna() & (df["dim"] == TOTAL_DIM)]
        return sel.drop_duplicates(key).set_index(key)["value"]

    both = pd.DataFrame({"rec": total(recipients), "inc": total(income)}).dropna().reset_index()
    both = both.sort_values(key, kind="mergesort").reset_index(drop=True)
    per = both["inc"] / both["rec"]
    both["dev"] = np.log(per / per.groupby(both["year"]).transform("median")).abs()
    prev = both.groupby("territory_id")[["year", "rec", "inc", "dev"]].shift(1)
    q = both["inc"] / prev["inc"]
    r = both["rec"] / prev["rec"]
    lo, hi = income_band
    stable = (both["year"] - prev["year"] == 1) & (q >= lo) & (q <= hi)
    up = stable & (r >= jump) & (prev["dev"] > both["dev"])  # меньшее и аномальное — прошлый год
    down = stable & (r <= 1.0 / jump) & (both["dev"] > prev["dev"])  # меньшее и аномальное — этот год
    bad = pd.concat(
        [
            pd.DataFrame({"territory_id": both.loc[up, "territory_id"], "year": prev.loc[up, "year"]}),
            both.loc[down, key],
        ]
    )
    out = recipients.copy()
    bad_keys = set(zip(bad["territory_id"].astype(int), bad["year"].astype(int), strict=True))
    if not bad_keys:
        return out
    keys = zip(out["territory_id"].astype(int), out["year"].astype(int), strict=True)
    hit = pd.Series([k in bad_keys for k in keys], index=out.index) & out["value"].notna()
    mark = pd.Series("", index=out.index).where(~hit, RECIPIENTS_JUMP_FLAG)
    out["flag"] = join_flags(out["flag"], mark)
    log.info(
        "5-НДФЛ: получатели с дефектом источника (%s) у %d МО-лет срезов",
        RECIPIENTS_JUMP_FLAG,
        int(hit.sum()),
    )
    return out


def _has_flag(flag: pd.Series, names: Sequence[str]) -> pd.Series:
    """Есть ли в строке флагов через «;» хотя бы один из ``names``."""
    pattern = r"(?:^|;)(?:" + "|".join(names) + r")(?:;|$)"
    return flag.fillna("").astype(str).str.contains(pattern, regex=True).astype(bool)


# --- Годовая таблица ----------------------------------------------------------------------------


def _pivot(long: pd.DataFrame, indicator: str, dim: str = TOTAL_DIM) -> pd.Series:
    sel = long[(long["indicator"] == indicator) & (long["dim"] == dim)]
    return sel.set_index(["territory_id", "year"])["value"]


def _employment(
    long: pd.DataFrame, cfg_context: dict, index: pd.MultiIndex, tol: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Доли разделов ОКВЭД2 в численности работников, группы разделов и «не раскрыто» (Л17).

    Доля раздела — работники раздела / итог; скрытый раздел — NA. «Не раскрыто» = 1 − сумма раскрытых
    долей. Если раскрытые доли в сумме больше 1 (на любую величину), они нормируются на сумму, «не
    раскрыто» = 0 и ставится ``emp_sh_renormalized`` (кроме перебора меньше ``SHARE_EPS`` — шума
    округления): композиция замкнута, доли групп не выходят за 1.
    Это отступление от А.4, где при переборе до 1% «не раскрыто» обрезается до 0, а доли не трогаются;
    перебор больше ``tol`` — ещё и предупреждение в логе. Группы — суммы раскрытых долей разделов по
    ``context.okved_groups``.
    Возвращает колонки ``context_annual``, работников по разделам и итог (NA, если итога нет или он ≤ 0).
    """
    emp = long[long["indicator"] == EMPLOYEES]
    wide = emp.pivot_table(index=["territory_id", "year"], columns="dim", values="value", aggfunc="first")
    wide = wide.reindex(index)
    total = wide[TOTAL_DIM] if TOTAL_DIM in wide else pd.Series(np.nan, index=index)
    total = total.where(total > 0)
    sections = pd.DataFrame({s: wide[s] if s in wide else np.nan for s in OKVED_SECTIONS}, index=index)
    shares = sections.div(total, axis=0)
    disclosed = shares.notna().sum(axis=1)
    ssum = shares.sum(axis=1, min_count=1).fillna(0.0)
    above = total.notna() & (ssum > 1.0)
    renormalized = above & (ssum > 1.0 + SHARE_EPS)
    over = above & (ssum > 1.0 + tol)
    shares.loc[above] = shares.loc[above].div(ssum[above], axis=0)
    unalloc = (1.0 - shares.sum(axis=1, min_count=1).fillna(0.0)).clip(lower=0.0)
    unalloc = unalloc.where(total.notna())
    out = pd.DataFrame(index=index)
    for s in OKVED_SECTIONS:
        out[f"emp_sh_{s}"] = shares[s]
    for group, letters in cfg_context["okved_groups"].items():
        out[f"emp_sh_{group}"] = shares[list(letters)].sum(axis=1, min_count=0).where(total.notna())
    out["emp_sh_unallocated"] = unalloc
    out["emp_n_disclosed"] = pd.array(disclosed.where(total.notna()), dtype="Int8")
    out["emp_sh_renormalized"] = renormalized.astype(bool)
    if over.any():
        log.warning(
            "занятость: сумма раскрытых долей больше 1 + %.2f у %d МО-лет, нормировано", tol, int(over.sum())
        )
    return out, sections.where(total.notna(), np.nan), total


def _okved_shares(emp: pd.DataFrame, sections: pd.DataFrame, total: pd.Series) -> pd.DataFrame:
    """Длинная таблица долей: МО × год × раздел (A…U и ``unallocated``) для МО-лет с итогом.

    ``employees`` — работники раздела из источника, ``share`` — доля из ``_employment``: если раскрытые
    разделы в сумме больше итога, доли нормированы на их сумму (``emp_sh_renormalized``), и тогда
    ``share`` ≠ ``employees`` / итог (на реальных данных — до 3·10⁻⁴).
    """
    ok = total.notna()
    parts = []
    for s in OKVED_SECTIONS:
        parts.append(
            pd.DataFrame(
                {"section": s, "employees": sections.loc[ok, s], "share": emp.loc[ok, f"emp_sh_{s}"]},
            )
        )
    un_share = emp.loc[ok, "emp_sh_unallocated"]
    parts.append(pd.DataFrame({"section": UNALLOCATED, "employees": total[ok] * un_share, "share": un_share}))
    out = pd.concat(parts).reset_index()
    out["territory_id"] = out["territory_id"].astype("int32")
    out["year"] = out["year"].astype("int16")
    out["section"] = out["section"].astype("str")
    out["employees"] = out["employees"].astype("float64")
    out["share"] = out["share"].astype("float64")
    return out.sort_values(["territory_id", "year", "section"], kind="mergesort").reset_index(drop=True)


def build_annual(
    long: pd.DataFrame, territories: pd.DataFrame, cfg: Config
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``context_annual`` (МО панели × годы панели) и ``okved_shares`` из ``context_long`` (А.4).

    Значения с флагами ``EXCLUDED_FLAGS`` (``union_merged`` — уже за объединённое МО, ``recipients_jump`` —
    дефект получателей 5-НДФЛ) в годовую таблицу не входят. Доход на получателя пуст, если получателей
    меньше нижней границы ``context.ndfl.recipients_ratio_ok`` от числа жителей: так выглядит и тот же
    дефект источника, а не только учёт по месту работы (Л3).
    """
    cx = cfg["context"]
    if "flag" in long:
        long = long[~_has_flag(long["flag"], EXCLUDED_FLAGS)]
    years = [int(y) for y in cfg["panel"]["years"]]
    ids = territories["territory_id"].astype("int32").to_numpy()
    index = pd.MultiIndex.from_product([np.sort(ids), years], names=["territory_id", "year"])
    ann = pd.DataFrame(index=index)

    pop = long[(long["indicator"] == POPULATION) & (long["dim"] == TOTAL_DIM)].set_index(
        ["territory_id", "year"]
    )
    ann["pop_jan1"] = pop["value"].reindex(index)
    ann["pop_jan1_method"] = pop["method"].reindex(index).astype("str")
    observed = pop[pop["method"] != "carried"]["value"]
    nxt = pd.MultiIndex.from_arrays([index.get_level_values(0), index.get_level_values(1) + 1])
    pop_next = pd.Series(observed.reindex(nxt).to_numpy(), index=index)
    both = ann["pop_jan1"].notna() & pop_next.notna()
    ann["pop_avg"] = ann["pop_jan1"].where(~both, (ann["pop_jan1"] + pop_next) / 2.0)
    ann["pop_avg_method"] = pd.Series(np.where(both, "mean_jan1", "jan1_only"), index=index).where(
        ann["pop_jan1"].notna()
    )

    total = ann["pop_jan1"]
    urban = _pivot(long, URBAN).reindex(index)
    rural = _pivot(long, RURAL).reindex(index)
    share = (urban / total).where(urban.notna(), 1.0 - rural / total)
    bad = share.notna() & ((share < 0) | (share > 1))
    if bad.any():
        log.warning("доля горожан вне [0; 1] у %d МО-лет — пропуск", int(bad.sum()))
    ann["urban_share"] = share.where(~bad)

    groups = list(cx["age"]["groups"])
    age = pd.DataFrame({g: _pivot(long, AGE, g).reindex(index) for g in groups})
    age_sum = age.sum(axis=1, min_count=len(groups))
    for g in groups:
        ann[f"age_{g}_share"] = age[g] / age_sum

    for name, col in ANNUAL_TOTALS.items():
        ann[col] = _pivot(long, name).reindex(index)

    emp, sections, emp_total = _employment(
        long, cx, index, cx.get("unallocated_tolerance", UNALLOCATED_TOLERANCE)
    )
    mining = _pivot(long, SHIPMENTS, MINING_SECTION).reindex(index)
    ann["shipments_mining_share"] = mining / ann["shipments_krub"].where(ann["shipments_krub"] > 0)

    inner = territories.set_index("territory_id")["is_inner_city"].astype(bool)
    workplace = pd.Series(inner.reindex(index.get_level_values(0)).to_numpy(), index=index)
    avg = ann["pop_avg"]
    for pc_col, src in PER_CAPITA_KRUB.items():
        ann[pc_col] = (ann[src] * KRUB_TO_RUB / avg / MONTHS_PER_YEAR).where(~workplace)
    ann["recipients_to_pop"] = ann["ndfl_recipients"] / avg
    lo, hi = cx["ndfl"]["recipients_ratio_ok"]
    ndfl_ok = (ann["recipients_to_pop"] >= lo) & (ann["recipients_to_pop"] <= hi) & ~workplace
    ann["ndfl_ok"] = ndfl_ok.fillna(False).astype(bool)
    ann["ndfl_income_pc"] = (ann["ndfl_income_rub"] / avg / MONTHS_PER_YEAR).where(ann["ndfl_ok"])
    few = (ann["recipients_to_pop"] < lo).fillna(False).astype(bool)  # единицы получателей
    per_recipient = ann["ndfl_income_rub"] / ann["ndfl_recipients"] / MONTHS_PER_YEAR
    ann["ndfl_income_per_recipient"] = per_recipient.where(~few)
    working = ann["pop_jan1"] * ann["age_working_share"]
    ann["employees_to_working_age"] = (ann["employees"] / working).where(~workplace)
    for col in ("orgs", "ip", "beds"):
        ann[f"{col}_per_1000"] = ann[col] / avg * PER_THOUSAND
    ann["nights_pc"] = ann["nights"] / avg
    ann["workplace_based"] = workplace.astype(bool)
    ann = ann.join(emp)

    out = ann.reset_index()
    out["territory_id"] = out["territory_id"].astype("int32")
    out["year"] = out["year"].astype("int16")
    shares = _okved_shares(emp, sections, emp_total)
    return out, shares


# --- Потери -------------------------------------------------------------------------------------


def _unmatched(
    matched: dict[str, pd.DataFrame], territories: pd.DataFrame, dic: pd.DataFrame, cfg: Config
) -> pd.DataFrame:
    """МО панели без значения показателя за год — с причиной (``outputs/panel/unmatched.csv``).

    Годы — годы показателя, начиная с первого года панели (население — и год после последнего, он нужен
    для среднегодового). Причины — из ``match_rows``, плюс ``not_in_slice`` (МО нет в срезе справочника
    года: предшественники объединений в 2024 году, преемники в 2023-м), ``nonpositive`` и флаги
    ``EXCLUDED_FLAGS`` (значение есть в ``context_long``, но не входит в ``context_annual``).
    """
    last_year = int(cfg["panel"]["dictionary_last_year"])
    first = min(int(y) for y in cfg["panel"]["years"])
    ids = territories["territory_id"].astype("int32")
    info = territories.set_index("territory_id")[["name", "region_code", "region_name"]]
    out = []
    for name, m in matched.items():
        for year in _indicator_years(cfg, name):
            if year < first:
                continue
            my = m[m["year"] == year]
            excluded = my["value"].notna() & _has_flag(my["flag"], EXCLUDED_FLAGS)
            have = set(my.loc[my["value"].notna() & ~excluded, "territory_id"])
            reason = (
                my[my["value"].isna()].drop_duplicates("territory_id").set_index("territory_id")["reason"]
            )
            for flag in EXCLUDED_FLAGS:
                hit = my.loc[excluded & _has_flag(my["flag"], [flag]), "territory_id"].unique()
                hit = [t for t in hit if t not in reason.index and t not in have]
                reason = pd.concat([reason, pd.Series(flag, index=pd.Index(hit, dtype="int32"))])
            sl = dictionary_slice(dic, year, last_year).set_index("territory_id")["oktmo8"]
            missing = [t for t in ids if t not in have]
            if not missing:
                continue
            r = pd.Series(missing, dtype="int32")
            frame = pd.DataFrame(
                {
                    "territory_id": r,
                    "year": year,
                    "indicator": name,
                    "reason": r.map(reason).where(r.isin(sl.index), NOT_IN_SLICE).fillna("no_code"),
                    "oktmo": r.map(sl),
                }
            )
            out.append(frame)
    if not out:
        cols = ["territory_id", "year", "indicator", "reason", "oktmo", "name", "region_code", "region_name"]
        return pd.DataFrame(columns=cols)
    res = pd.concat(out, ignore_index=True)
    res = res.join(info, on="territory_id")
    res["year"] = res["year"].astype("int16")
    return res.sort_values(["indicator", "year", "territory_id"], kind="mergesort").reset_index(drop=True)


# --- Сборка -------------------------------------------------------------------------------------


def build_context_full(cfg: Config, dic: pd.DataFrame, territories: pd.DataFrame) -> ContextResult:
    """Весь контекст: отбор строк, сопоставление, население, годовая таблица, доли ОКВЭД2 и потери."""
    cx = cfg["context"]
    last_year = int(cfg["panel"]["dictionary_last_year"])
    panel_years = [int(y) for y in cfg["panel"]["years"]]
    panel_ids = pd.Index(territories["territory_id"].astype("int32"))

    rows, pairs, present = _select_all(cfg)
    ndfl = read_ndfl(Path(cfg["paths"]["raw"]) / cx["ndfl"]["file"], cx["ndfl"], cx["na_codes"])

    matched: dict[str, pd.DataFrame] = {}
    for name, r in rows.items():
        matched[name] = match_rows(r, dic, pairs, _indicator_years(cfg, name), last_year, present=present)
    matched[POPULATION] = population(matched[POPULATION], dic, last_year, carry_years=panel_years)
    for name in (NDFL_INCOME, NDFL_RECIPIENTS):
        years = _indicator_years(cfg, name)
        matched[name] = match_rows(ndfl[name], dic, ndfl["pairs"], years, last_year, present=ndfl["present"])
    matched[NDFL_RECIPIENTS] = flag_recipient_jumps(
        matched[NDFL_INCOME],
        matched[NDFL_RECIPIENTS],
        cx["ndfl"].get("recipients_jump", RECIPIENTS_JUMP),
        cx["ndfl"].get("income_stable", INCOME_STABLE),
    )

    long = pd.concat([_to_long(m, name, panel_ids) for name, m in matched.items()], ignore_index=True)
    long["territory_id"] = long["territory_id"].astype("int32")
    long["year"] = long["year"].astype("int16")
    long["indicator"] = pd.Categorical(long["indicator"], categories=list(matched))
    for col in ("dim", "unit", "source_code", "period_used", "oktmo_used", "method", "flag"):
        long[col] = long[col].astype("str")
    long = long.sort_values(["territory_id", "year", "indicator", "dim"], kind="mergesort").reset_index(
        drop=True
    )

    annual, shares = build_annual(long, territories, cfg)
    unmatched = _unmatched(matched, territories, dic, cfg)

    bdmo_rows = pd.concat([r.assign(indicator=name) for name, r in rows.items()], ignore_index=True)[
        ROWS_COLUMNS
    ]
    ndfl_rows = pd.concat(
        [ndfl[name].assign(indicator=name) for name in (NDFL_INCOME, NDFL_RECIPIENTS)], ignore_index=True
    )[ROWS_COLUMNS]
    sort = ["indicator", "oktmo", "year", "dim"]
    info = {"bdmo_stable_pairs": len(pairs), "ndfl_stable_pairs": len(ndfl["pairs"])}
    for flag in EXCLUDED_FLAGS:  # значения МО панели в context_long, не вошедшие в context_annual
        info[f"values_{flag}"] = int(_has_flag(long["flag"], [flag]).sum())
    return ContextResult(
        context_long=long,
        context_annual=annual,
        okved_shares=shares,
        unmatched=unmatched,
        bdmo_rows=bdmo_rows.sort_values(sort, kind="mergesort").reset_index(drop=True),
        ndfl_rows=ndfl_rows.sort_values(sort, kind="mergesort").reset_index(drop=True),
        info=info,
    )


def build_context(
    cfg: Config, dic: pd.DataFrame, territories: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """``context_long``, ``context_annual``, ``okved_shares``, ``unmatched`` (контракт spec_final, Д.2)."""
    res = build_context_full(cfg, dic, territories)
    return res.context_long, res.context_annual, res.okved_shares, res.unmatched
