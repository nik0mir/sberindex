"""Отчёт ``docs/interpretation.md`` (слепой прогон — ``outputs/interpret_blind/interpretation_blind.md``):
шаблон ``templates/interpretation.md``, поля ``{{имя}}`` из ``facts.json`` и таблицы
``<!-- table: имя -->`` из CSV этапа. Тексты исходов и главный вывод собраны кодом этапа по правилам
записи; здесь только вёрстка.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from munnet import style
from munnet.config import Config
from munnet.contracts import QCError
from munnet.eda import report as eda_report
from munnet.eda.base import to_markdown
from munnet.eda.s1_coverage import plural
from munnet.interpret.amendments import lines as amendment_lines
from munnet.interpret.clarifications import CLARIFICATIONS
from munnet.interpret.labels import SCHEME_LABELS, SET_NAMES, TURNOVER_NOM, feature_label
from munnet.interpret.spec import Spec

log = logging.getLogger(__name__)

TEMPLATE = Path(__file__).resolve().parent / "templates" / "interpretation.md"
_FIELD = re.compile(r"\{\{([a-z0-9_]+)\}\}")
_TABLE = re.compile(r"<!--\s*table:\s*([a-z0-9_]+)\s*-->")
_FIGURE = re.compile(r"<!--\s*figure:\s*(I\d{2})\s*-->")
LINT_ALLOW = [r"CC BY(?:-SA)? \d\.\d", r"\b\d{1,2}\.\d{2}\.\d{4}\b", r"«[^»]*\.[^»]*»"]
VERDICT_RU = {
    "confirmed": "подтвердилась",
    "partial_one": "частично (один оборот сверх места)",
    "partial_overall": "частично (только в целом)",
    "partial": "частично",
    "partial_capped": "частично (часть переходов — к типу места)",
    "not": "не подтвердилась",
    "down": "значимо вниз",
    "not_empty": "не повторяют, но внешним оборотом не подтверждено",
    "not_repeats": "повторяют тривиальное деление",
}
F2 = lambda v: style.fmt_num(v, 2)  # noqa: E731
F3 = lambda v: style.fmt_num(v, 3)  # noqa: E731


def _v(x: str) -> str:
    return VERDICT_RU.get(str(x), str(x))


def _ci(lo: float, hi: float, f=F2) -> str:
    return style.fmt_range(lo, hi, f) if np.isfinite(lo) and np.isfinite(hi) else style.NA_TEXT


def _md(df: pd.DataFrame) -> str:
    return to_markdown(df) if len(df) else "*Нет строк.*"


# --- Таблицы -------------------------------------------------------------------------------------------


def t_verdicts(out: Path, f: Mapping) -> str:
    rows = []
    vm, vf = f["verdicts_main"], f["verdicts_final"]
    for t, name in (
        ("T1_ladder_external", "T1: ступени по обороту Росстата"),
        ("T5_trivial", "T5: не тривиальное деление"),
        ("T3_reliable_placebo", "T3: надёжные переходы против плацебо"),
        ("T2_direction", "T2: направление переходов"),
        ("T6_bank_coverage", "T6: перешедшие МО в обороте Росстата"),
        ("T7_utility", "T7: польза подбора"),
    ):
        final = vf["T1_text"] if t == "T1_ladder_external" else vf[t]
        rows.append(
            {"Проверка": name, "Основной расчёт": _v(vm[t]), "В главном выводе (после R1)": _v(final)}
        )
    return _md(pd.DataFrame(rows))


def t_controls(out: Path, f: Mapping) -> str:
    c = pd.read_csv(out / "controls.csv")
    c = c.loc[c["group"].isin(["partitions", "random"])]
    df = pd.DataFrame(
        {
            "Деление": c["label"],
            "T1": [_v(x) for x in c["t1"]],
            "T5": [_v(x) for x in c["t5"]],
            "Не выше ожидаемого": [bool(a) and bool(b) for a, b in zip(c["t1_ok"], c["t5_ok"], strict=True)],
        }
    )
    return _md(df)


def t_controls_reference(out: Path, f: Mapping) -> str:
    c = pd.read_csv(out / "controls.csv")
    c = c.loc[c["group"].isin(["self_out", "basket_parts"])]
    df = pd.DataFrame(
        {
            "Деление": c["label"],
            "Справка": ["без себя в наборе" if g == "self_out" else "одна часть корзины" for g in c["group"]],
            "T1": [_v(x) for x in c["t1"]],
            "T5": [_v(x) if isinstance(x, str) else style.NA_TEXT for x in c["t5"]],
        }
    )
    return _md(df)


def t_t1(out: Path, f: Mapping) -> str:
    d = pd.read_csv(out / "t1_runs.csv")
    d = d.loc[d["run"] == "main"]
    df = pd.DataFrame(
        {
            "Оборот": [TURNOVER_NOM[x] for x in d["turnover"]],
            "Узлов (a) / (b)": [
                f"{style.fmt_num(a)} / {style.fmt_num(b)}" for a, b in zip(d["n_a"], d["n_b"], strict=True)
            ],
            "ρ (a) [95%]": [
                f"{F2(r)} [{_ci(lo, hi)}]"
                for r, lo, hi in zip(d["rho_a"], d["rho_a_lo"], d["rho_a_hi"], strict=True)
            ],
            "q (a)": [style.fmt_p(x) for x in d["q_a"]],
            "Монотонно (a)": d["mono_a"].astype(bool),
            "ρ (b) [95%]": [
                f"{F2(r)} [{_ci(lo, hi)}]"
                for r, lo, hi in zip(d["rho_b"], d["rho_b_lo"], d["rho_b_hi"], strict=True)
            ],
            "q (b)": [style.fmt_p(x) for x in d["q_b"]],
            "Монотонно (b)": d["mono_b"].astype(bool),
            "Разность с сильнейшим [95%]": [
                f"{F2(r)} [{_ci(lo, hi)}]"
                for r, lo, hi in zip(d["diff"], d["diff_lo"], d["diff_hi"], strict=True)
            ],
            "«За в целом»": d["overall"].astype(bool),
            "«За сверх места»": d["beyond"].astype(bool),
            "«За сверх места», строго (с |ρ (a)| ≥ rho_min)": d["beyond_strict"].astype(bool),
        }
    )
    return _md(df)


def t_t1_regions(out: Path, f: Mapping) -> str:
    rows = []
    for ind, r in f["t1_regions_up"].items():
        rows.append(
            {
                "Показатель": TURNOVER_NOM.get(ind, "плотность населения"),
                "Групп региона, где медиана растёт по ступеням": f"{r['n_up']} из {r['n_eligible']}",
                "На нуле (среднее)": F2(r["null_mean"]),
                "p (односторонний)": style.fmt_p(r["p"]),
            }
        )
    return _md(pd.DataFrame(rows))


def t_t5_ami(out: Path, f: Mapping) -> str:
    d = pd.read_csv(out / "t5_ami.csv").sort_values("ami", ascending=False).head(10)
    return _md(
        pd.DataFrame(
            {"Деление": d["label"], "AMI": [F3(x) for x in d["ami"]], "ARI": [F3(x) for x in d["ari"]]}
        )
    )


def t_t5_eps(out: Path, f: Mapping) -> str:
    rows = []
    for n, r in f["t5"]["per"].items():
        rows.append(
            {
                "Оборот": TURNOVER_NOM[n],
                "Узлов": style.fmt_num(r["n"]),
                "ε² типов [95%]": f"{F3(r['eps'])} [{_ci(*r['eps_ci'], F3)}]",
                "q": style.fmt_p(r["q"]),
                "Сильнейшее деление без типов": r["best_rival_label"],
                "Его ε²": F3(r["best_rival_eps"]),
                "Разность [95%]": f"{F3(r['diff_point'])} [{_ci(*r['diff_ci'], F3)}]",
            }
        )
    return _md(pd.DataFrame(rows))


def t_t3(out: Path, f: Mapping) -> str:
    rows = []
    for s, r in f["t3"].items():
        rows.append(
            {
                "Половины": SCHEME_LABELS[s],
                "Надёжных переходов": style.fmt_num(r["n_reliable"]),
                "Плацебо: медиана": style.fmt_num(r["median"], 0),
                "Плацебо: 95-й перцентиль": style.fmt_num(r["p95"], 1),
                "Избыток, 95% интервал": _ci(r["lo"], r["hi"], lambda v: style.fmt_num(v, 0)),
                "Выше перцентиля": bool(r["passed"]),
            }
        )
    return _md(pd.DataFrame(rows))


def t_t2(out: Path, f: Mapping) -> str:
    rows = []
    for s, r in f["t2"].items():
        rows.append(
            {
                "Половины": SCHEME_LABELS[s],
                "Вверх / вниз": f"{r['n_up']} / {r['n_down']}",
                "Доля «вверх» [95%]": f"{F2(r['share_up'])} [{_ci(*r['share_ci'])}]",
                "p0 плацебо": F2(r["p0"]) if not r["undetermined"] else f"не определена ({r['n_pool']})",
                "p «чаще вверх»": style.fmt_p(r["p_up"]),
                "p «чаще вниз»": style.fmt_p(r["p_down"]),
                "N; 95-й перцентиль |N| плацебо": f"{F2(r['net_flow'])}; {F2(r['placebo_abs_n_p95'])}",
            }
        )
    return _md(pd.DataFrame(rows))


def t_r1(out: Path, f: Mapping) -> str:
    d = pd.read_csv(out / "r1_runs.csv")
    wide = d.pivot_table(index="run", columns="test", values="verdict", aggfunc="first")
    wide = wide.rename(
        columns={"T1_ladder_external": "T1", "T2_direction": "T2", "T3_reliable_placebo": "T3"}
    )
    wide = wide.map(lambda x: _v(x) if isinstance(x, str) else style.NA_TEXT)
    return _md(wide.reset_index().rename(columns={"run": "Прогон"}))


def t_t6(out: Path, f: Mapping) -> str:
    d = pd.read_csv(out / "t6_flows.csv")
    if not len(d):
        return _md(d)
    df = pd.DataFrame(
        {
            "Поток": [f"{a} → {b}" for a, b in zip(d["source"], d["dest"], strict=True)],
            "Перешли / остались": [f"{a} / {b}" for a, b in zip(d["n_movers"], d["n_stayers"], strict=True)],
            "Дельта Клиффа": [F2(x) for x in d["cliff"]],
            "Медиана Δ: перешли": [F3(x) for x in d["median_movers"]],
            "Медиана Δ: остались": [F3(x) for x in d["median_stayers"]],
        }
    )
    return _md(df)


def t_t7(out: Path, f: Mapping) -> str:
    t7 = f["t7"]
    rows = []
    for s in "ABCD":
        rows.append(
            {
                "Набор": SET_NAMES[s] if s in SET_NAMES else s,
                "Медианная ошибка": F3(t7["median_error"][s]),
                "Без относительности": F3(t7["median_error_abs"][s]),
            }
        )
    d = t7["diffs"]
    rows2 = [
        {"Разность": k.replace("_abs", " (без относительности)"), "Точка": F3(v[0]), "95%": _ci(*v[1], F3)}
        for k, v in d.items()
    ]
    return _md(pd.DataFrame(rows)) + "\n\n" + _md(pd.DataFrame(rows2))


def t_names(out: Path, f: Mapping) -> str:
    rows = []
    for t, v in f["names"].items():
        rows.append(
            {
                "Тип": t,
                "Название в выходах": f["names_final"][str(t)] if "names_final" in f else v["name"],
                "Кандидат по правилу": v["name"],
                "Описательное": f.get("names_descriptive", {}).get(str(t), style.NA_TEXT),
                "Подпись": v["caption"],
                "Части корзины": ", ".join(feature_label(p) for p in v["parts"]) or style.NA_TEXT,
            }
        )
    return _md(pd.DataFrame(rows))


def t_profile(out: Path, f: Mapping) -> str:
    p = pd.read_csv(out / "profile.csv")
    p = p.loc[p["type"] > 0]
    order, types = list(dict.fromkeys(p["feature"])), list(dict.fromkeys(p["type"]))
    wide = p.pivot_table(index="feature", columns="type", values="effect_mad", aggfunc="first").loc[
        order, types
    ]
    cl = p.pivot_table(index="feature", columns="type", values="cliff", aggfunc="first").loc[order, types]
    rows = []
    for feat in wide.index:
        row = {"Признак": feature_label(feat)}
        for t in wide.columns:
            row[f"Тип {t}: MAD / δ"] = f"{F2(wide.loc[feat, t])} / {F2(cl.loc[feat, t])}"
        rows.append(row)
    return _md(pd.DataFrame(rows))


def t_examples(out: Path, f: Mapping) -> str:
    e = pd.read_csv(out / "examples.csv")
    e = e.loc[e["kind"].isin(["typical", "borderline", "largest"])]
    kind = {"typical": "типичное", "borderline": "пограничное", "largest": "крупнейшее"}
    df = pd.DataFrame(
        {
            "Тип": e["type"].astype(int),
            "Вид": [kind[k] for k in e["kind"]],
            "МО": e["name"],
            "Регион": e["region"],
            "Ранг / второй тип": [
                (
                    style.fmt_num(r)
                    if k == "typical"
                    else (f"тип {int(s)}" if k == "borderline" else style.NA_TEXT)
                )
                for k, r, s in zip(e["kind"], e["rank"], e["second_type"], strict=True)
            ],
            "Пометки": [
                "; ".join(x for x, ok in (("по месту работы", w), ("тип неустойчив", u)) if ok)
                or style.NA_TEXT
                for w, u in zip(e["workplace_based"], e["type_unstable"], strict=True)
            ],
        }
    )
    return _md(df)


def t_flows(out: Path, f: Mapping) -> str:
    e = pd.read_csv(out / "examples.csv")
    e = e.loc[e["kind"] == "flow"]
    if not len(e):
        return "*Потоков надёжных переходов не меньше порога нет.*"
    df = pd.DataFrame(
        {
            "Переход": [f"{int(a)} → {int(b)}" for a, b in zip(e["source"], e["dest"], strict=True)],
            "Переходов в потоке": e["n_flow"].astype(int),
            "МО": e["name"],
            "Регион": e["region"],
        }
    )
    return _md(df)


def t_tree(out: Path, f: Mapping) -> str:
    d = pd.read_csv(out / "tree_rules.csv")
    df = pd.DataFrame(
        {
            "Тип": d["type"],
            "Правило": d["rule"],
            "Точность": [style.fmt_pct(x, 0) for x in d["precision"]],
            "Покрытие": [style.fmt_pct(x, 0) for x in d["coverage"]],
            "Для журналиста": d["journalist"].astype(bool),
        }
    )
    return _md(df)


def t_fca(out: Path, f: Mapping) -> str:
    d = pd.read_csv(out / "fca.csv")
    d = d.loc[d["in_text"].astype(bool)] if len(d) else d
    if not len(d):
        return "*Устойчивых импликаций с заданными точностью и покрытием нет.*"
    d = d.sort_values(["type", "coverage"], ascending=[True, False]).groupby("type").head(3)
    df = pd.DataFrame(
        {
            "Тип": d["type"],
            "Набор ⇒ тип": d["itemset"],
            "Точность": [style.fmt_pct(x, 0) for x in d["precision"]],
            "Покрытие": [style.fmt_pct(x, 0) for x in d["coverage"]],
            "Устойчивость": [style.fmt_pct(x, 0) for x in d["stability"]],
        }
    )
    return _md(df)


def t_exceptions(out: Path, f: Mapping) -> str:
    d = pd.read_csv(out / "exceptions.csv")
    sh = d.loc[d["kind"] == "share"]
    return _md(
        pd.DataFrame({"Тип": sh["type"], "Доля исключений": [style.fmt_pct(x, 1) for x in sh["value"]]})
    )


def t_posthoc(out: Path, f: Mapping) -> str:
    d = pd.read_csv(out / "posthoc.csv")
    return _md(
        pd.DataFrame(
            {"Показатель": d["indicator"], "Узлов": d["n"], "ε² по типам": [F3(x) for x in d["epsilon2"]]}
        )
    )


def t_t2_pairs(out: Path, f: Mapping) -> str:
    rows = []
    for s_, r in f["t2"].items():
        for pp in r.get("per_pair", []):
            rows.append(
                {
                    "Половины": SCHEME_LABELS[s_],
                    "Пара ступеней (типы)": f"{int(pp['low']) + 1}–{int(pp['high']) + 1}",
                    "Вверх / вниз": f"{pp['n_up']} / {pp['n_down']}",
                    "N пары": F2(pp["net"]),
                    "95-й перцентиль |N| пары на плацебо": F2(pp["placebo_abs_n_p95"]),
                }
            )
    return _md(pd.DataFrame(rows))


def t_weighted(out: Path, f: Mapping) -> str:
    """Профиль: «типичное МО» (медиана без весов) и «типичный житель» (медиана с весами населения) рядом."""
    p = pd.read_csv(out / "profile.csv")
    order, types = list(dict.fromkeys(p["feature"])), [t for t in dict.fromkeys(p["type"]) if t > 0]
    rows = []
    for feat in order:
        row = {"Признак": feature_label(feat)}
        for t in [0, *types]:
            r = p.loc[(p["feature"] == feat) & (p["type"] == t)].iloc[0]
            head = "Все" if t == 0 else f"Тип {t}"
            row[f"{head}: МО / житель"] = f"{F2(r['median'])} / {F2(r['wmedian'])}"
        rows.append(row)
    return _md(pd.DataFrame(rows))


def t_city(out: Path, f: Mapping) -> str:
    rows = f["scope"].get("city_nodes", [])
    if not rows:
        return ""
    return _md(pd.DataFrame({"Узел-город": [r["name"] for r in rows], "Тип": [r["type"] for r in rows]}))


TABLES: dict[str, Callable[[Path, Mapping], str]] = {
    "t2_pairs": t_t2_pairs,
    "weighted": t_weighted,
    "city": t_city,
    "verdicts": t_verdicts,
    "controls": t_controls,
    "controls_reference": t_controls_reference,
    "t1": t_t1,
    "t1_regions": t_t1_regions,
    "t5_ami": t_t5_ami,
    "t5_eps": t_t5_eps,
    "t3": t_t3,
    "t2": t_t2,
    "r1": t_r1,
    "t6": t_t6,
    "t7": t_t7,
    "names": t_names,
    "profile": t_profile,
    "examples": t_examples,
    "flows": t_flows,
    "tree": t_tree,
    "fca": t_fca,
    "exceptions": t_exceptions,
    "posthoc": t_posthoc,
}


# --- Рисунки -------------------------------------------------------------------------------------------


def figures(out: Path, f: Mapping) -> dict[str, tuple[Path, str]]:
    figs: dict[str, tuple[Path, str]] = {}
    src = style.join_sources(style.SOURCE_SBER, style.SOURCE_ROSSTAT)
    # I01 — профиль: отклонение медианы в MAD × 1,4826
    p = pd.read_csv(out / "profile.csv")
    p = p.loc[p["type"] > 0]
    order, types = list(dict.fromkeys(p["feature"])), list(dict.fromkeys(p["type"]))
    wide = p.pivot_table(index="feature", columns="type", values="effect_mad", aggfunc="first").loc[
        order, types
    ]
    fig, ax = style.new_figure("tall")
    M = wide.to_numpy(dtype=float)
    lim = max(0.5, float(np.nanmax(np.abs(M)))) if M.size else 1.0
    ax.imshow(M, cmap=style.DIV_CMAP, vmin=-lim, vmax=lim, aspect="auto")
    ax.set_yticks(range(len(wide.index)), [feature_label(x) for x in wide.index])
    ax.set_xticks(range(len(wide.columns)), [f"Тип {t}" for t in wide.columns])
    ax.grid(False)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, style.fmt_num(M[i, j], 1), ha="center", va="center", fontsize=7)
    style.finish(
        fig,
        "Профили типов",
        "Медиана признака в типе минус медиана всех территориальных узлов, в единицах MAD × 1,4826",
        src,
    )
    figs["I01"] = (_save(fig, out, "I01_profile"), "Тепловая карта отклонений медиан признаков по типам")
    # I02 — T3: плацебо против наблюдения
    d = pd.read_csv(out / "t3_placebo.csv")
    d = d.loc[d["run"] == "main"]
    fig, axes = style.new_figure("full", 1, 2)
    for ax, s in zip(np.atleast_1d(axes), ("main", "sensitivity"), strict=False):
        pl = d.loc[(d["scheme"] == s) & (d["pair"] >= 0), "n_reliable"]
        ob = d.loc[(d["scheme"] == s) & (d["pair"] < 0), "n_reliable"]
        ax.hist(pl, bins=20, color=style.CONTEXT)
        if len(ob):
            ax.axvline(float(ob.iloc[0]), color=style.ACCENT)
        ax.set_title(SCHEME_LABELS[s], fontsize=9)
        ax.set_xlabel("Надёжных переходов")
    style.finish(
        fig,
        "Надёжные переходы: наблюдение и псевдогоды",
        "Гистограмма — псевдопары без реального времени; линия — наблюдаемое число",
        style.SOURCE_SBER,
    )
    figs["I02"] = (
        _save(fig, out, "I02_placebo"),
        "Гистограммы числа надёжных переходов на псевдогодах и наблюдаемое число",
    )
    # I03 — T7: медианные ошибки наборов
    t7 = f["t7"]
    fig, ax = style.new_figure("full")
    keys = list("ABCD")
    vals = [t7["median_error"][k] for k in keys]
    ax.barh(range(4), vals, color=style.CONTEXT)
    ax.set_yticks(range(4), [SET_NAMES[k] for k in keys])
    ax.set_xlabel("Медианная ошибка изменения розничного оборота 2023 → 2024")
    style.finish(
        fig,
        "Ошибка подбора сопоставимых территорий",
        "Медиана по МО |y − медиана y набора|; y — изменение ln(1 + розница на жителя) относительно региона",
        src,
    )
    figs["I03"] = (_save(fig, out, "I03_t7"), "Столбцы медианных ошибок четырёх наборов МО")
    return figs


def _save(fig, out: Path, stem: str) -> Path:
    paths = style.save(fig, out / "figures" / stem, formats=("png",))
    return paths[0]


# --- Сборка --------------------------------------------------------------------------------------------


def _t2_describe(t2d: Mapping, d2: Mapping, place: Mapping) -> str:
    nan = float("nan")
    return (
        "D1 — доля «вверх» минус доля «вверх» у смен «нечётные → чётные» одного года: "
        f"{F2(t2d.get('d1', nan))}, в обратной ориентации {F2(t2d.get('d1_reverse', nan))}; "
        f"D2 — доля «вверх» минус нуль перестановок назначений: {F2(d2.get('d2_excl_diag', nan))} "
        f"(диагональ вне знаменателя), {F2(d2.get('d2_diag_not_up', nan))} (диагональ как «не вверх»); "
        "прогноз типа по месту совпадает с типом назначения у "
        f"{style.fmt_pct(place.get('share_a', nan), 1)} перешедших против "
        f"{style.fmt_pct(place.get('share_null', nan), 1)} на нуле "
        f"(p = {style.fmt_p(place.get('p', nan))}). Данные: {t2d.get('data', '')}."
    )


def _t1_strict_note(t1: Mapping) -> str:
    """Проверка чувствительности: вердикт основного расчёта T1 по строгому прочтению «за сверх места».

    Главный вердикт — буквальное прочтение записи («(a) и (b): знак, q < alpha, монотонность, beyond_place»).
    """
    st = t1.get("verdict_strict")
    if not st:
        return ""
    same = "совпадает с главным" if st == t1.get("verdict") else "расходится с главным"
    return (
        " Проверка чувствительности: главный вердикт — по буквальному прочтению оборота «за сверх места» "
        "(знак, q < alpha, монотонность и beyond_place для версий (a) и (b), без порога |ρ| версии (a)); "
        f"по строгому прочтению (с порогом |ρ| версии (a)) вердикт основного расчёта — «{_v(st)}», {same}; "
        "в главный вывод идёт буквальное прочтение; обоснование — раздел 13 «Уточнения реализации»."
    )


def _scope_note(sc: Mapping) -> str:
    """Пометка рядом с фразой охвата, если «медиана населения у них ниже» по данным не выполнено."""
    if "phrase_holds" not in sc:
        return ""
    # подпись — ровно та группа, по которой считается медиана: МО без типа (те же, что n_partial фразы)
    n_pop = f" (n = {style.fmt_num(sc['pop_n_partial'], 0)})" if "pop_n_partial" in sc else ""
    med = (
        f"медиана населения МО без типа{n_pop} — {style.fmt_num(sc['pop_median_partial'], 0)}, "
        + (
            "у территориальных узлов сети (полный ряд, без внутригородских МО Москвы и Петербурга)"
            if "pop_full_group" in sc
            else "с полным"
        )
        + f" — {style.fmt_num(sc['pop_median_full'], 0)}"
    )
    if "pop_median_full_all" in sc:
        med += (
            " (справка: у всех МО с полным рядом, включая внутригородские МО Москвы и Петербурга, — "
            f"{style.fmt_num(sc['pop_median_full_all'], 0)})"
        )
    dev = _scope_deviation(sc)
    if sc["phrase_holds"]:
        return f"; {med}{dev}"
    return f"; **пометка: утверждение о населении не выполнено** — {med}{dev}"


def _scope_deviation(sc: Mapping) -> str:
    """Отклонение от комментария записи (interpret.scope.phrase): там n_partial — все МО с неполным рядом,
    код считает МО без типа. Оба числа — рядом с фразой, чтобы читатель видел разницу."""
    keys = ("n_partial", "n_partial_series", "n_partial_typed_via_city")
    if not all(k in sc for k in keys) or sc["n_partial_series"] == sc["n_partial"]:
        return ""
    via = int(sc["n_partial_typed_via_city"])
    return (
        "; отклонение от комментария записи: там число во фразе — все МО с неполным рядом, здесь — МО без "
        f"типа; МО с неполным рядом — {style.fmt_num(sc['n_partial_series'], 0)}, из них {via} "
        f"{plural(via, 'типизировано', 'типизированы', 'типизированы')} через узел-город "
        "(внутригородские МО Москвы и Петербурга; раздел 13 «Уточнения реализации»)"
    )


def _t4_note(t4: Mapping) -> str:
    """Пометка рядом с текстом T4, если «итог ближе к корзине, чем к месту» по точечным ARI не выполнено."""
    if t4.get("text_holds", True):
        return ""
    return (
        " **Пометка: утверждение «итог ближе к разбиению по корзине, чем по месту» не выполнено** — "
        f"ARI с корзиной {F2(t4['ari_basket'])} не больше ARI с местом {F2(t4['ari_place'])}."
    )


def _coverage_note(t6: Mapping) -> str:
    """Описание оговорки про охват: ε² по обоим прочтениям уровня трат (решает наибольшее)."""
    by = t6.get("coverage_eps2_by_level", {})
    if not by:
        return ""
    return (
        " Оговорка про охват (описание, без p-значений): ε² разности уровня трат и дохода 5-НДФЛ по типам — "
        f"{F3(by.get('place_year', np.nan))} по уровню года дохода и {F3(by.get('months_24', np.nan))} "
        "по уровню 24 месяцев; решает наибольшее."
    )


def _main_note(t: Mapping, key: str, f: Mapping) -> str:
    """Основной расчёт выше итогового вердикта (``main_run_verdict: check_section``): текст с пометкой."""
    if key not in t:
        return ""
    return "\n\n" + f"*Основной расчёт — {f['r1']['unstable_label']}:* {t[key]}"


def _transfer_text(f: Mapping) -> str:
    tr = {int(k): int(v) for k, v in f["ladder"]["transfer"].items()}
    if all(k == v for k, v in tr.items()):
        return "номера типов cluster_final совпали с номерами записи (размеры те же), перенос не понадобился"
    pairs = ", ".join(f"{k} → {v}" for k, v in sorted(tr.items()))
    return f"номера типов cluster_final перенесены на номера записи по размеру: {pairs}"


def _chain_note(f: Mapping, blind: int | None = None) -> str:
    """Сверка «цепочка окон против прямого сопоставления» (qc.chain_direct): наблюдение T2/T3 сопоставлено
    с итогом цепочкой окон, плацебо — напрямую."""
    cd = f.get("qc", {}).get("chain_direct", {})
    if not cd:
        return ""
    bad = sorted(r for r, v in cd.items() if float(v.get("last_window_differ", 0)) > 0)
    head = (
        "Сверка сравнения: наблюдаемые типы 2024 года этап dynamics сопоставляет с итогом цепочкой окон, "
        "а псевдогоды плацебо — напрямую."
    )
    if bad:
        note = (
            f"{head} В окне 2024 года оба пути расходятся в прогонах {', '.join(bad)} — там сравнение "
            "с плацебо неравноценно (`facts.json`: `qc.chain_direct`)."
        )
        if blind is not None:
            note += (
                " В слепом прогоне сверка идёт на метках, возвращённых в исходный порядок узлов, поэтому "
                "число узлов — свойство данных, как в настоящем прогоне, а не следствие перемешивания."
            )
        return note
    return (
        f"{head} В окне 2024 года оба пути дают одни и те же номера у всех узлов во всех прогонах "
        "(`facts.json`: `qc.chain_direct`), так что сравнение равноценно."
    )


def fields(spec: Spec, f: Mapping, blind: int | None, out_dir: str = "outputs/interpret") -> dict[str, str]:
    tests = spec["tests"]
    th = f["thesis"]
    t = f["texts"]
    banner = ""
    if blind is not None:
        banner = (
            f"> **{f['label']}.** Типы и переходы перемешаны по узлам (seed {blind}): "
            "числа ниже не описывают "
            "настоящие типы и служат только отладке кода. Исключение — всё, что считается по наблюдаемым "
            "переходам основного расчёта: перестановка строк узлов (переход целиком достаётся другому "
            "узлу) их не меняет, это настоящие числа. Это число надёжных переходов T3, «вверх» и «вниз» "
            "T2 (известны до записи, seen_before), а также пары соседних ступеней n_ij против n_ji, "
            "бутстрап-интервал доли «вверх» и описания D1 (половины года тех же узлов) и D2 (нуль "
            "перестановок назначений): их в seen_before нет, при отладке их не читать. Плацебо рядом "
            "с ними построено на перемешанных метках, поэтому сравнение наблюдения с плацебо в этом отчёте "
            "ничего не говорит о тезисе.\n\n"
        )
    t2d = f["t2_describe"]
    d2 = t2d.get("d2", {})
    t7 = f["t7"]
    ex = t7.get("example", {})
    comp = f["t5"]["composition"]
    names_rule = str(spec["naming"]["assembly"])
    nt = f["naming_test"]
    if nt.get("done"):
        naming_status = f"{nt['who']}: верно {nt['correct']} из 4, порог {nt['pass_min']}"
    elif nt.get("state") == "stale":
        naming_status = (
            f"{nt['who']}: записанный ответ относится к другому пакету названий и профилей (package_sha "
            "не совпал) и не применён; в выходах описательные названия, нужна новая проверка"
        )
    else:
        naming_status = f"{nt['who']}: ещё не проведена, в выходах описательные названия"
    seen = [str(x) for x in spec["seen_before"]]
    r1 = f["r1"]
    unstable = r1["unstable_parts"]
    higher = r1["main_higher"]
    exp = spec["controls"]["expected"].plain()
    return {
        "banner": banner,
        "out_dir": out_dir,
        "t3_chain": _chain_note(f, blind),
        "question": th["question"],
        "point_1": " ".join(th["point_1"]),
        "point_2": " ".join(th["point_2"]),
        "point_3": " ".join(th["point_3"]),
        "caveat": " ".join(th["caveat"]),
        "scope": f["scope"]["phrase"] + _scope_note(f["scope"]),
        "edits": "\n".join(f"- {e}" for e in f["edits"]) or "- Правок нет.",
        "seen_before": "\n".join(f"- {s}" for s in seen),
        "controls_expected": (
            f"T1 — не выше «{_v(exp['T1_ladder_external']['partitions'])}» у делений и "
            f"«{_v(exp['T1_ladder_external']['random'])}» у случайных; "
            f"T5 — не выше «{_v(exp['T5_trivial']['partitions'])}»; случайных выше ожидаемого — не больше "
            f"{f['controls']['random_max_above']} из {f['controls']['n_random']}"
        ),
        "controls_verdict": "контроли не выше ожидаемого — правило отличает тип от экономики места"
        if f["controls"]["passed"]
        else "контроли выше ожидаемого",
        "controls_seen": str(spec["controls"]["seen"]),
        "t1_question": str(tests["T1_ladder_external"]["question"]),
        "ladder_transfer": _transfer_text(f),
        "n_no_stratum": str(f["scope"].get("n_no_stratum", "—")),
        "t7_dropped": str(f["t7"].get("n_dropped", "—")),
        "t7_common": str(f["t7"].get("n_common", "—")),
        "t1_verdict_main": _v(f["verdicts_main"]["T1_ladder_external"]),
        "t1_verdict_text": _v(f["verdicts_final"]["T1_text"]),
        "t1_text": t["T1_ladder_external"]
        + f.get("t1_notes", {}).get("T1_ladder_external", "")
        + _main_note(t, "T1_main", f)
        + (f.get("t1_notes", {}).get("T1_main", "") if "T1_main" in t else "")
        + _t1_strict_note(f["t1"]),
        "t1_proxies": t["T1_proxies"],
        "t1_circularity": str(tests["T1_ladder_external"]["circularity"]),
        "t5_question": str(tests["T5_trivial"]["question"]),
        "t5_verdict": _v(f["verdicts_main"]["T5_trivial"]),
        "t5_text": t["T5_trivial"],
        "t5_comp_ok": str(comp["n_ok"]),
        "t5_comp_n": str(comp["n_groups"]),
        "t5_comp_types": str(spec["tests"]["T5_trivial"]["composition"]["min_types"]),
        "t5_region_ami": F3(f["t5"]["region_ami"]),
        "t5_circularity": str(tests["T5_trivial"]["circularity"]),
        "t4_text": " ".join(
            [
                str(tests["T4_basket_vs_place"]["question"]),
                t["T4_basket_vs_place"] + _t4_note(f["t4"]),
                str(tests["T4_basket_vs_place"]["circularity"]),
            ]
        ),
        "t4_ari_pop": F2(f["t4"].get("ari_log_pop_rel_quintiles", np.nan)),
        "t4_basket_t1": _v(f["t4"].get("basket_only_t1", "—")),
        "t3_question": str(tests["T3_reliable_placebo"]["question"])
        + " "
        + str(tests["T3_reliable_placebo"]["circularity"]),
        "t3_verdict_main": _v(f["verdicts_main"]["T3_reliable_placebo"]),
        "t3_verdict_text": _v(f["verdicts_final"]["T3_reliable_placebo"]),
        "t3_text": t["T3_reliable_placebo"] + _main_note(t, "T3_main", f),
        "t2_question": str(tests["T2_direction"]["question"])
        + " "
        + str(tests["T2_direction"]["circularity"]),
        "t2_verdict_main": _v(f["verdicts_main"]["T2_direction"]),
        "t2_verdict_text": _v(f["verdicts_final"]["T2_direction"]),
        "t2_text": t["T2_direction"] + _main_note(t, "T2_main", f),
        "t2_describe": _t2_describe(t2d, d2, f["t2_place"]),
        "r1_unstable": (
            "Части корзины, у которых знак отклонения типа не прошёл R1 (в ограничения, не в названия): "
            + ", ".join(f"тип {u.split(':')[0]} — {feature_label(u.split(':')[1]).lower()}" for u in unstable)
            + "."
            if unstable
            else "Знаки отклонения всех частей корзины у всех типов прошли R1."
        ),
        "r1_main_higher": (
            "Основной расчёт выше итогового вердикта ("
            + r1["unstable_label"]
            + "): "
            + ", ".join(f"{k.split('_')[0]} — {_v(v)}" for k, v in higher.items())
            + ". "
            + r1["circularity"]
            if higher
            else "Основной расчёт не выше итогового вердикта ни в одной проверке. " + r1["circularity"]
        ),
        "t6_question": str(tests["T6_bank_coverage"]["question"])
        + " "
        + str(tests["T6_bank_coverage"]["circularity"]),
        "t6_verdict": _v(f["verdicts_main"]["T6_bank_coverage"]),
        "t6_text": t["T6_bank_coverage"] + _coverage_note(f["t6"]),
        "t7_question": str(tests["T7_utility"]["question"]) + " " + str(tests["T7_utility"]["circularity"]),
        "t7_verdict": _v(f["verdicts_main"]["T7_utility"]),
        "t7_text": t["T7_utility"],
        "t7_example": (
            "Пример для лендинга — МО с медианной ошибкой набора продукта: "
            f"territory_id {ex.get('territory_id')} "
            f"(ошибка {F3(ex.get('error', np.nan))}), членов набора {len(ex.get('members', []))}; "
            f"сверх 10 признаков места метки типа дают ΔR² {F3(t7['beyond_place']['delta_r2'])} "
            f"(p = {style.fmt_p(t7['beyond_place']['p'])})."
        ),
        "naming_rule": names_rule,
        "naming_test": naming_status,
        "tree_depth": str(spec["tree"]["depth"]),
        "tree_bacc": F2(f["tree"]["balanced_accuracy"]),
        "tree_major": F2(f["tree"]["baseline_majority_bacc"]),
        "tree_random": F2(f["tree"]["baseline_random_bacc"]),
        "posthoc": str(spec["posthoc"]),
        "clarifications": "\n".join(f"- {c}" for c in CLARIFICATIONS),
        "amendments": "\n".join(amendment_lines(spec)),
        "sources": "\n".join(
            [
                f"- Все файлы — в `{out_dir}/`. Вердикты, тексты, главный вывод: `{out_dir}/facts.json` "
                "(ключи `verdicts_*`, `texts`, `thesis`).",
                "- T1: `t1_runs.csv`, `t1_pairwise.csv`, `t1_proxies.csv`; T5: `t5_ami.csv`, "
                "`t5_composition.csv`; контроли: `controls.csv`.",
                "- T3 и T2: `t3_placebo.csv`, `facts.json` (`t3`, `t2`, `t2_describe`); "
                "R1: `r1_runs.csv`, `r1_basket_signs.csv`.",
                "- T6: `t6_flows.csv`; T7: `t7_errors.csv`; профиль и названия: `profile.csv`, "
                "`settlement_shares.csv`, "
                "`types.csv`; примеры: `examples.csv`; правила: `tree_rules.csv`, `fca.csv`, "
                "`exceptions.csv`; "
                "разведка после регистрации: `posthoc.csv`; пакет слепой проверки названий: "
                "`naming_test.json`.",
                "- Поузловые выгрузки для лендинга — описание, не проверка (в вердикты и главный вывод "
                "не входят): `node_comparable.csv` (наборы T7), `node_r1.csv` и `node_seed.csv` (тип узла "
                "в прогонах R1), `node_margin.csv` (пограничность), `node_rival.csv` (деления-соперники).",
                f"- Источники данных: {style.SOURCE_SBER}; {style.SOURCE_ROSSTAT}; {style.SOURCE_FNS}.",
            ]
        ),
    }


def write(
    cfg: Config, spec: Spec, out: Path, report: Path, images: Path, facts: Mapping, blind: int | None
) -> Path:
    figs = figures(out, facts)
    out_dir = f"outputs/{out.name}"
    fl = fields(spec, facts, blind, out_dir)
    try:
        img_rel = Path(os.path.relpath(images, report.parent)).as_posix()
    except ValueError:
        img_rel = images.as_posix()
    text = TEMPLATE.read_text(encoding="utf-8")
    missing = sorted(set(_FIELD.findall(text)) - set(fl))
    if missing:
        raise KeyError(f"отчёт interpret: нет полей {missing}")
    text = _FIELD.sub(lambda m: fl[m.group(1)], text)
    text = _TABLE.sub(lambda m: TABLES[m.group(1)](out, facts), text)

    def fig_md(m: re.Match) -> str:
        path, alt = figs[m.group(1)]
        n = int(m.group(1)[1:])
        return f"![{alt}]({img_rel}/{path.name})\n\n*Рисунок {n}.* Данные: `{out_dir}/figures/`."

    text = _FIGURE.sub(fig_md, text)
    text = eda_report.unwrap_paragraphs(text)
    text = eda_report.nbsp_markdown(text)
    problems = eda_report.lint_ru(text, LINT_ALLOW)
    if problems:
        log.warning("отчёт interpret: типографика — %s", "; ".join(problems[:10]))
    report.parent.mkdir(parents=True, exist_ok=True)
    images.mkdir(parents=True, exist_ok=True)
    for path, _ in figs.values():
        shutil.copyfile(path, images / path.name)
    report.write_text(text, encoding="utf-8", newline="\n")
    log.info("interpret: отчёт %s", report)
    if blind is None and "СЛЕПОЙ" in text:
        raise QCError("отчёт interpret: пометка слепого прогона в настоящем отчёте")
    return report
