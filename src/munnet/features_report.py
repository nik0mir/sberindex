"""Отчёт ``docs/features.md``: пространство признаков узлов и его обоснование (этап features).

Шаблон — ``templates/features.md``; числа текста — только факты ``{{feat.ключ}}`` (``outputs/features/
report_facts.json``), таблица признаков — директива ``<!-- table: features -->`` из ``feature_table.csv``.
Выводы, зависящие от данных (какие пары дублируют друг друга, какие признаки ненадёжны, выше ли каппа
корзины относительно региона), собираются из чисел, а не пишутся в шаблоне: при других данных меняется
и текст. Типографика — функции отчёта разведки (неразрывные пробелы, проверка кавычек, тире и точки).
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from munnet import style
from munnet.config import Config
from munnet.contracts import PARTS, YEARS, QCError
from munnet.eda import report as eda_report
from munnet.eda import stats
from munnet.eda.base import Fact, make_fact, write_json

if TYPE_CHECKING:
    from munnet.features import FeatureParams, RhythmResult, StoryInputs

log = logging.getLogger(__name__)

TEMPLATE = Path(__file__).resolve().parent / "templates" / "features.md"
SECTION = "feat"
FACTS_FILE = "report_facts.json"
_TABLE = re.compile(r"<!--\s*table:\s*([a-z_]+)\s*-->")
_NOTE = re.compile(r"[ \t]*<!--\s*note\s*:.*?-->[ \t]*\n?", re.S)
LINT_ALLOW = [r"CC BY(?:-SA)? \d\.\d", r"\b\d{1,2}\.\d{2}\.\d{4}\b"]
NB = style.NBSP


def _fact(facts: dict[str, Fact], key: str, value: Any, kind: str, note: str = "") -> None:
    facts[f"{SECTION}.{key}"] = make_fact(f"{SECTION}.{key}", value, kind, note)


def _code(name: str) -> str:
    return f"`{name}`"


def _join(items: Sequence[str]) -> str:
    items = list(items)
    if not items:
        return ""
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + f" и{NB}" + items[-1]


def _range(values: pd.Series, digits: int = 2) -> str:
    v = values.dropna()
    if v.empty:
        return style.NA_TEXT
    lo, hi = float(v.min()), float(v.max())
    if round(lo, digits) == round(hi, digits):
        return style.fmt_num(lo, digits)
    return style.fmt_range(lo, hi, style.fmt_num, decimals=digits)


# --- Значения признаков ------------------------------------------------------------------------


def feature_values(s: StoryInputs, rhythm: pd.DataFrame) -> pd.DataFrame:
    """Значения признаков ``FEATURE_DOCS`` в одной таблице узлов: окна — последний год, место — первый год
    контекста, ритм — как есть (как в ``features.feature_table``)."""
    from munnet.features import FEATURE_DOCS

    y0, y1 = YEARS
    w = s.windows
    win = w.loc[(w["window_kind"].astype(str) == "year") & (w["year"] == y1)].set_index("territory_id")
    place = s.place.loc[s.place["year"] == y0].set_index("territory_id")
    rh = rhythm.set_index("territory_id")
    out = pd.DataFrame(index=win.index)
    for name, doc in FEATURE_DOCS.items():
        src = {"features_windows": win, "features_place": place, "features_rhythm": rh}[doc.table]
        out[name] = src[name].reindex(out.index).astype("float64")
    return out


def pairs_above(values: pd.DataFrame, names: Sequence[str], threshold: float) -> list[tuple[str, str, float]]:
    """Пары признаков ``names`` с |ρ Спирмена| > ``threshold`` (версии одного показателя не считаются), по
    убыванию |ρ|."""
    from munnet.features import _same_variable

    corr = values[list(names)].rank().corr()
    out = []
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            r = float(corr.loc[a, b])
            if np.isfinite(r) and abs(r) > threshold and not _same_variable(a, b):
                out.append((a, b, r))
    return sorted(out, key=lambda x: -abs(x[2]))


# --- Факты -------------------------------------------------------------------------------------


def build_facts(
    cfg: Config,
    p: FeatureParams,
    s: StoryInputs,
    ft: pd.DataFrame,
    checks: Mapping[str, Any],
    rhythm: RhythmResult,
    moran: Mapping[str, Any],
    eta2_max: float,
) -> dict[str, Fact]:
    """Все числа и собранные из чисел фразы отчёта (``feat.*``)."""
    from munnet.features import FEATURE_DOCS

    facts: dict[str, Fact] = {}
    y0, y1 = YEARS
    table = ft.set_index("feature")
    space = p.space
    rel_min = float(space.get("min_reliability", 0.5))
    rho_max = float(space.get("max_abs_rho", 0.8))
    dup = float(space.get("duplicate_rho", 0.999))

    # Узлы
    n_nodes, n_all = len(s.ids), len(s.table)
    _fact(facts, "n_nodes", n_nodes, "int", "узлов сети (ряд не короче features.min_months)")
    _fact(facts, "n_nodes_all", n_all, "int", "узлов в режиме nodes.mode")
    _fact(facts, "n_dropped", n_all - n_nodes, "int", "узлов вне сети (короткий ряд)")
    _fact(facts, "n_city_nodes", checks["n_city_nodes"], "int", "узлов-городов")
    _fact(facts, "min_months", p.min_months, "int", "features.min_months")
    _fact(facts, "node_mode_text", checks["node_mode_text"], "str", "режим узлов словами")
    _fact(
        facts, "region_group_text", checks["region_group_text"], "str", "как считается «относительно региона»"
    )
    _fact(facts, "n_groups", int(s.groups.reindex(s.ids).nunique()), "int", "групп региона среди узлов сети")
    _fact(facts, "knn", int(moran["k"]), "int", "соседей в весах I Морана (eda.knn_main)")
    _fact(facts, "permutations", int(moran["permutations"]), "int", "перестановок в тесте I Морана")
    _fact(facts, "eta2_max", eta2_max, "num1", "η² выше — показатель почти целиком региональный")
    _fact(facts, "rel_min", rel_min, "num1", "features.space.min_reliability")
    _fact(facts, "rho_max", rho_max, "num1", "features.space.max_abs_rho")

    # Почему относительно региона: η² сырых величин на узлах (та же мера, что T13 разведки)
    w = s.windows
    last = w.loc[(w["window_kind"].astype(str) == "year") & (w["year"] == y1)]
    grp = last["region_group"]
    raw = {"level": last["log_level"], **{q: last[f"clr_{q}"] for q in PARTS}}
    eta_raw = {k: stats.eta2(v.astype("float64"), grp) for k, v in raw.items()}
    for k, v in eta_raw.items():
        _fact(facts, f"eta2_raw_{k}", v, "num2", f"η² группы региона: {k} без поправки на регион, {y1}, узлы")
    _fact(facts, "eta2_raw_min", min(eta_raw.values()), "num2", "наименьший η² уровня и шести CLR как есть")
    _fact(facts, "eta2_raw_max", max(eta_raw.values()), "num2", "наибольший")
    n, k = int(grp.notna().sum()), int(grp.nunique())
    _fact(
        facts,
        "eta2_random",
        (k - 1) / (n - 1),
        "num2",
        "η² чистого шума при том же числе групп: (k − 1)/(n − 1)",
    )
    rel_cols = ["log_level_rel", *(f"clr_rel_{q}" for q in PARTS)]
    rel_eta = max(stats.eta2(last[c].astype("float64"), grp) for c in rel_cols)
    _fact(
        facts, "eta2_rel_max", rel_eta, "num2", "наибольший η² уровня и корзины относительно группы региона"
    )
    story = checks["story"]
    for key, kind in (
        ("tree_kappa", "num3"),
        ("tree_kappa_clean", "num3"),
        ("ami_region", "num3"),
        ("moved_share", "pct"),
        ("noise_share", "pct"),
        ("k", "int"),
    ):
        _fact(facts, key, story.get(key), kind, f"checks.json, story.{key}")
    better = story.get("tree_kappa", np.nan) > story.get("tree_kappa_clean", np.nan)
    _fact(facts, "kappa_word", "сильнее" if better else "не сильнее", "str", "сравнение двух каппа")

    # Признаки по одному
    for name in FEATURE_DOCS:
        r = table.loc[name]
        _fact(facts, f"{name}_rel", r["reliability"], "num2", f"надёжность {name} ({r['reliability_basis']})")
        _fact(facts, f"{name}_eta2", r["eta2_region_group"], "num2", f"η² группы региона {name}")
        _fact(facts, f"{name}_moran", r["moran_i"], "num2", f"I Морана {name}, k = {moran['k']}")
        _fact(facts, f"{name}_cov", int(r["coverage"]), "int", f"узлов со значением {name}")

    basket = table.loc[[f"clr_rel_{q}" for q in PARTS]]
    _fact(facts, "basket_rel_range", _range(basket["reliability"]), "str", "надёжность шести CLR")
    _fact(facts, "basket_moran_range", _range(basket["moran_i"]), "str", "I Морана шести CLR")
    emp = table.loc[
        [f"emp_sh_{g}" for g in ("primary", "industry", "trade_transport", "market_services", "public")]
    ]
    _fact(facts, "emp_rel_range", _range(emp["reliability"]), "str", "надёжность долей занятости")
    _fact(facts, "emp_eta2_range", _range(emp["eta2_region_group"]), "str", "η² долей занятости")
    place_attr = table.loc[(table["group"] == "place") & (table["role"] == "attributes"), "moran_i"]
    _fact(facts, "place_moran_range", _range(place_attr), "str", "I Морана признаков места — атрибутов")

    values = feature_values(s, rhythm.features)
    names = list(FEATURE_DOCS)
    # Ранговые дубли: одна из пары — вне пространства
    dups = [(a, b, r) for a, b, r in pairs_above(values, names, dup)]
    roles = table["role"].to_dict()

    def pair_text(a: str, b: str, r: float) -> str:
        return f"{_code(a)} и{NB}{_code(b)} (ρ{NB}={NB}{style.fmt_rho(r)})"

    dup_texts = []
    for a, b, r in dups:
        kept = [x for x in (a, b) if roles[x] != "outside"]
        tail = f": в пространстве {_code(kept[0])}" if len(kept) == 1 else ""
        dup_texts.append(pair_text(a, b, r) + tail)
    _fact(facts, "duplicates", "; ".join(dup_texts) or "нет", "str", f"пары с |ρ| > {dup}")
    coll = [x for x in pairs_above(values, names, rho_max) if abs(x[2]) <= dup]
    _fact(
        facts,
        "collinear",
        "; ".join(pair_text(a, b, r) for a, b, r in coll) or "нет",
        "str",
        f"пары с |ρ| > {rho_max}, кроме ранговых дублей",
    )
    unrel = table.loc[table["reliability"] < rel_min].sort_values("reliability")
    _fact(
        facts,
        "unreliable",
        _join([f"{_code(n)} ({style.fmt_num(v, 2)})" for n, v in unrel["reliability"].items()]) or "нет",
        "str",
        f"признаки с надёжностью ниже {rel_min}",
    )
    # Роли и проверка выбора
    for role in ("edges", "attributes", "layer", "dynamics"):
        part = table.loc[table["role"] == role]
        _fact(facts, f"n_{role}", len(part), "int", f"признаков роли {role}")
        _fact(
            facts, f"n_{role}_ok", int(part["check_ok"].sum()), "int", f"из них прошли проверку роли {role}"
        )
        bad = part.loc[~part["check_ok"].astype(bool)]
        _fact(
            facts,
            f"{role}_failed",
            "; ".join(f"{_code(n)} — {t}" for n, t in bad["check_note"].items()) or "нет",
            "str",
            f"признаки роли {role}, не прошедшие проверку",
        )
    attrs = table.loc[table["role"] == "attributes"]
    attr_pairs = pairs_above(values, list(attrs.index), -1.0)
    top = attr_pairs[0] if attr_pairs else None
    _fact(
        facts, "attr_rho_max", None if top is None else abs(top[2]), "num2", "наибольший |ρ| внутри атрибутов"
    )
    _fact(facts, "attr_rho_pair", "—" if top is None else pair_text(*top), "str", "эта пара")
    edge_rho = attrs["max_abs_rho_edges"].astype("float64")
    who = edge_rho.idxmax() if edge_rho.notna().any() else None
    _fact(
        facts,
        "edge_rho_max",
        None if who is None else edge_rho[who],
        "num2",
        "наибольший |ρ| атрибута с рёбрами",
    )
    _fact(
        facts,
        "edge_rho_pair",
        "—" if who is None else f"{_code(who)} и{NB}{_code(str(attrs.loc[who, 'max_abs_rho_edges_with']))}",
        "str",
        "атрибут и признак рёбер с наибольшим |ρ|",
    )
    _fact(facts, "edges_list", _join([_code(n) for n in table.index[table["role"] == "edges"]]), "str", "")
    _fact(facts, "attributes_list", _join([_code(n) for n in attrs.index]), "str", "")
    _fact(
        facts, "dynamics_list", _join([_code(n) for n in table.index[table["role"] == "dynamics"]]), "str", ""
    )
    _fact(facts, "layer_list", _join([_code(n) for n in table.index[table["role"] == "layer"]]), "str", "")
    _fact(
        facts, "own_summer_eta2_group", table.loc["own_summer", "eta2_region_group"], "num2", "η² своего лета"
    )
    wn = pairs_above(values, ["log_wage_rel", "log_ndfl_rel"], -1.0)
    _fact(facts, "wage_ndfl_rho", wn[0][2] if wn else None, "rho", "ρ зарплаты и дохода 5-НДФЛ к региону")

    # Ритм
    rf = rhythm.features
    _fact(facts, "reliable_share", float(rf["own_reliable"].mean()), "pct", "узлов с надёжным своим ритмом")
    north = rf["north"].astype(bool)
    _fact(facts, "reliable_north", float(rf.loc[north, "own_reliable"].mean()), "pct", "то же на Севере")
    _fact(facts, "reliable_rest", float(rf.loc[~north, "own_reliable"].mean()), "pct", "то же вне Севера")
    _fact(facts, "null_share", rhythm.null_reliable_share, "pct", "то же на перестановочном нуле")
    _fact(facts, "n_north", int(north.sum()), "int", "узлов севернее eda.north_lat")
    return facts


# --- Утверждения шаблона ------------------------------------------------------------------------


def check_claims(ft: pd.DataFrame, checks: Mapping[str, Any], p: FeatureParams, eta2_max: float) -> list[str]:
    """Утверждения шаблона, зависящие от данных; возвращает неверные (пусто — всё верно).

    Шаблон говорит: корзина относительно региона связана с экономикой места сильнее очищенной (каппа), типы
    меняются сверх шума, уровень трат почти целиком региональный, ранговый дубль — летний избыток трат и свой
    летний избыток (в пространстве — свой), сильно связанные пары — только внутри корзины, декабрьский пик
    ненадёжен, доступность рынков и доля старших как есть — региональны, с рёбрами сильнее всего связан
    уровень трат.
    """
    t = ft.set_index("feature")
    story = checks["story"]
    space = p.space
    rel_min = float(space.get("min_reliability", 0.5))
    rho_max = float(space.get("max_abs_rho", 0.8))
    dup = float(space.get("duplicate_rho", 0.999))
    bad = []

    def need(ok: bool, text: str) -> None:
        if not ok:
            bad.append(text)

    need(story.get("tree_kappa", 0) > story.get("tree_kappa_clean", 1), "каппа корзины к региону > очищенной")
    need(story.get("moved_share", 0) > story.get("noise_share", 1), "смена типа > шума")
    duplicates = {
        frozenset((a, str(t.loc[a, "max_abs_rho_with"])))
        for a in t.index
        if np.isfinite(t.loc[a, "max_abs_rho"]) and t.loc[a, "max_abs_rho"] > dup
    }
    need(duplicates == {frozenset({"summer_excess", "own_summer"})}, "ранговый дубль — только летний избыток")
    need(
        t.loc["own_summer", "role"] != "outside" and t.loc["summer_excess", "role"] == "outside",
        "в пространстве — свой летний избыток",
    )
    strong = t.loc[(t["max_abs_rho"] > rho_max) & (t["max_abs_rho"] <= dup)]
    need(bool((strong["group"] == "basket").all()), "сильно связанные пары — только внутри корзины")
    need(t.loc["dec_peak", "reliability"] < rel_min, "декабрьский пик ненадёжен")
    last = t.loc[[f"clr_rel_{q}" for q in PARTS]]
    need(bool((last["eta2_region_group"] < 1e-6).all()), "η² корзины относительно региона — ноль")
    need(t.loc["own_summer", "eta2_region_group"] > eta2_max, "свой летний избыток — региональный (Север)")
    for name in ("market_access", "age_old_share"):
        need(t.loc[name, "eta2_region_group"] > eta2_max, f"{name} как есть — региональный")
    attrs = t.loc[t["role"] == "attributes", "max_abs_rho_edges"].astype("float64")
    need(attrs.notna().any() and attrs.idxmax() == "log_level_rel", "с рёбрами сильнее всего связан уровень")
    return bad


# --- Таблица -----------------------------------------------------------------------------------


def _num(v: Any, digits: int = 2) -> str:
    return (
        style.NA_TEXT
        if v is None or (isinstance(v, float) and not np.isfinite(v))
        else style.fmt_num(v, digits)
    )


def table_features(ft: pd.DataFrame) -> str:
    """Markdown таблицы признаков: группа, признак, нормировка, роль, покрытие, надёжность, η², I Морана,
    наибольший |ρ| с другим признаком, проверка роли."""
    from munnet.features import FEATURE_GROUPS, ROLE_TEXT

    head = (
        "| Группа | Признак | Нормировка | Роль | Узлов | Надёжность | η² группы региона | I Морана | "
        "Наибольший \\|ρ\\| (с чем) | Проверка |\n|---|---|---|---|---:|---:|---:|---:|---|---|"
    )
    rows = [head]
    for r in ft.itertuples(index=False):
        rho = (
            style.NA_TEXT
            if not np.isfinite(r.max_abs_rho)
            else f"{style.fmt_num(r.max_abs_rho, 2)} ({_code(str(r.max_abs_rho_with))})"
        )
        mark = "✓" if bool(r.check_ok) else "✗"
        rows.append(
            f"| {FEATURE_GROUPS[r.group]} | {r.label} ({_code(r.feature)}) | {r.norm} | "
            f"{ROLE_TEXT[r.role]} | "
            f"{style.fmt_num(r.coverage)} | {_num(r.reliability)} | {_num(r.eta2_region_group)} | "
            f"{_num(r.moran_i)} | {rho} | {mark} {r.check_note} |"
        )
    return "\n".join(rows)


# --- Сборка ------------------------------------------------------------------------------------


def render(template: str, facts: Mapping[str, Fact], ft: pd.DataFrame) -> str:
    text = _NOTE.sub("", template)
    text = eda_report.fill(text, facts)
    tables = {"features": lambda: table_features(ft)}
    text = _TABLE.sub(lambda m: tables[m.group(1)](), text)
    text = eda_report.unwrap_paragraphs(text)
    return eda_report.nbsp_markdown(text)


def write_report(
    cfg: Config,
    p: FeatureParams,
    s: StoryInputs,
    ft: pd.DataFrame,
    checks: Mapping[str, Any],
    rhythm: RhythmResult,
    moran: Mapping[str, Any],
    eta2_max: float,
) -> Path:
    """Собирает ``docs/features.md`` (``features.report``) и ``outputs/features/report_facts.json``."""
    facts = build_facts(cfg, p, s, ft, checks, rhythm, moran, eta2_max)
    out = cfg.dir("outputs") / "features"
    write_json(
        {k: {"value": f.value, "text": f.text, "note": f.note} for k, f in sorted(facts.items())},
        out / FACTS_FILE,
    )
    wrong = check_claims(ft, checks, p, eta2_max)
    if wrong:
        msg = "отчёт features: утверждения шаблона неверны на этих данных: " + "; ".join(wrong)
        if cfg["features"].get("report_strict", True):
            raise QCError(msg + " (перепишите шаблон src/munnet/templates/features.md)")
        log.warning("%s — отчёт собран без проверки (features.report_strict = false)", msg)
    md = render(TEMPLATE.read_text(encoding="utf-8"), facts, ft)
    problems = eda_report.lint_ru(md, LINT_ALLOW)
    if problems:
        raise QCError("отчёт features: типографика: " + "; ".join(problems[:10]))
    report = Path(cfg["features"]["report"])
    report.parent.mkdir(parents=True, exist_ok=True)
    old = report.read_text(encoding="utf-8") if report.exists() else None
    if old != md:
        report.write_text(md, encoding="utf-8", newline="\n")
    log.info("features: отчёт %s", report)
    return report
