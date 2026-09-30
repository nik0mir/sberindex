"""Тексты исходов: выбор по вердикту и подстановка чисел — механически, по полям ``outcomes`` предрегистрации.

Слова текстов берутся только из блока ``interpret``; код подставляет числа и подписи (``labels``). Главный
вывод — ``thesis_assembly``: вопрос и три пункта из полей ``text`` исходов, без новых слов.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from munnet import style
from munnet.interpret.labels import ONE_IN_TEN_PHRASE, SCHEME_LABELS, SET_NAMES, TURNOVER_GEN, TURNOVER_NOM

log = logging.getLogger(__name__)

LEVELS: dict[
    str, tuple[str, ...]
] = {}  # тест -> порядок вердиктов снизу вверх (rule.levels), заполняет stage


def fmt(template: str, **fields: Any) -> str:
    """Подстановка полей ``{имя}``; неизвестное поле — KeyError (шаблон и код разошлись)."""
    return str(template).format_map(_Strict(fields))


class _Strict(dict):
    def __missing__(self, key: str) -> str:
        raise KeyError(f"поле текста {{{key}}} не заполнено")


def cap(text: str) -> str:
    return text[:1].upper() + text[1:] if text else text


def rho(x: float) -> str:
    return style.fmt_rho(x, 2)


def per_turnover(values: Mapping[str, float], names: Sequence[str], f=rho) -> str:
    """«общепит 0,xx, розница 0,yy» (комментарий к outcomes T1)."""
    return ", ".join(f"{TURNOVER_NOM[n]} {f(values[n])}" for n in names)


def joined_gen(names: Sequence[str]) -> str:
    g = [TURNOVER_GEN[n] for n in names]
    return g[0] if len(g) == 1 else " и ".join([", ".join(g[:-1]), g[-1]])


def best_label(labels: Mapping[str, str], names: Sequence[str]) -> str:
    """Подпись сильнейшего соперника; у разных оборотов разные — через «», «» с оборотом в скобках
    (уточнение реализации: у одного поля {best_partition} в тексте с двумя оборотами)."""
    uniq = list(dict.fromkeys(labels[n] for n in names))
    if len(uniq) == 1:
        return uniq[0]
    return "», «".join(f"{labels[n]} ({TURNOVER_NOM[n]})" for n in names)


def ci_text(ci: tuple[float, float], f=None) -> str:
    f = f or (lambda v: style.fmt_num(v, 2))
    lo, hi = ci
    if lo < 0 or hi < 0:
        return f"от {f(lo)} до {f(hi)}"
    return f"{f(lo)}–{f(hi)}"


def level_min(test: str, verdicts: Sequence[str]) -> str:
    order = LEVELS[test]
    return min(verdicts, key=order.index)


def level_rank(test: str, v: str) -> int:
    return LEVELS[test].index(v)


# --- T1 --------------------------------------------------------------------------------------------


def t1_text(res, out: Mapping, verdict: str | None = None) -> str:
    verdict = verdict or res.verdict
    names = list(res.per)
    o = out[verdict]
    cond = {n: res.per[n].rho_b for n in names}
    best = {n: res.per[n].best_rival_label for n in names}
    best_rho = {n: res.per[n].best_rival_rho for n in names}
    if verdict == "confirmed":
        return fmt(
            o["text"],
            rho_cond=per_turnover(cond, names),
            best_partition=best_label(best, names),
            rho_best=per_turnover(best_rho, names),
        )
    if verdict == "partial_one":
        beyond = [n for n in names if res.per[n].beyond] or names[:1]
        other = [n for n in names if n not in beyond][0]
        if res.per[other].overall:
            clause = fmt(
                o["other_overall"],
                turnover_other=TURNOVER_GEN[other],
                best_partition_other=best[other],
                rho_best_other=rho(best_rho[other]),
            )
        else:
            clause = fmt(o["other_none"], turnover_other=TURNOVER_GEN[other])
        b = beyond[0]
        return fmt(
            o["text"],
            turnover_beyond=TURNOVER_GEN[b],
            rho_cond=rho(cond[b]),
            best_partition=best[b],
            rho_best=rho(best_rho[b]),
            other_clause=clause,
        )
    if verdict == "partial_overall":
        ov = [n for n in names if res.per[n].overall]
        if not ov:
            # текст понижен до partial_overall (thesis_assembly.t1_cap_by_t5), а «за в целом» не выполнено
            # ни у одного оборота: правило не меняем, подставляем все обороты, но не молча — предупреждение
            # в лог и пометка рядом с текстом (t1_overall_note)
            log.warning(
                "interpret: текст T1 «%s», но «за в целом» не выполнено ни у одного оборота (|ρ (a)| ниже "
                "rho_min) — подставлены все обороты, рядом с текстом пометка",
                verdict,
            )
            ov = names
        return fmt(
            o["text"],
            turnovers_overall=joined_gen(ov),
            rho_overall=per_turnover({n: res.per[n].rho_a for n in ov}, ov),
            rho_cond=per_turnover(cond, ov),
            best_partition=best_label(best, ov),
            rho_best=per_turnover(best_rho, ov),
        )
    return fmt(o["text"])


def t1_overall_note(res, verdict: str | None = None) -> str:
    """Пометка рядом с текстом T1: текст «partial_overall» утверждает «за в целом», а оно не выполнено
    ни у одного оборота (бывает, когда «за сверх места» по буквальному прочтению выполнено без порога
    |ρ (a)| ≥ rho_min, а thesis_assembly.t1_cap_by_t5 понизил текст). Иначе — пустая строка."""
    verdict = verdict or res.verdict
    names = list(res.per)
    if verdict != "partial_overall" or any(res.per[n].overall for n in names):
        return ""
    low = [n for n in names if res.per[n].beyond]
    vals = f" (ρ (a): {per_turnover({n: res.per[n].rho_a for n in low}, low)})" if low else ""
    return (
        " **Пометка: «за в целом» не выполнено: |ρ (a)| ниже rho_min**"
        f"{vals}; текст выбран правилом thesis_assembly.t1_cap_by_t5, обороты в нём подставлены все, "
        "правило не менялось."
    )


def t1_proxies(proxies: Mapping[str, dict], out: Mapping) -> str:
    items = []
    for d in proxies.values():
        items.append(
            fmt(
                out["proxies"]["proxy_item"],
                proxy=d["label"],
                values=" / ".join(d["values_text"]),
                monotone="да" if d["monotone"] else "нет",
            )
        )
    return fmt(out["proxies"]["text"], proxy_summary="; ".join(items))


# --- T5 --------------------------------------------------------------------------------------------


def t5_text(res, out: Mapping) -> str:
    o = out[res.verdict]
    names = list(res.per)
    passed = [n for n in names if res.per[n].passed]
    use = passed if res.verdict in ("confirmed", "partial") else names
    eps = {n: res.per[n].eps for n in names}
    best = {n: res.per[n].best_rival_label for n in names}
    best_eps = {n: res.per[n].best_rival_eps for n in names}
    fields = {
        "max_partition": res.max_label,
        "max_ami": style.fmt_num(res.max_ami, 2),
        "turnovers": joined_gen(use),
        "eps2": per_turnover(eps, use, lambda v: style.fmt_num(v, 3)),
        "best_partition": best_label(best, use),
        "eps2_best": per_turnover(best_eps, use, lambda v: style.fmt_num(v, 3)),
    }
    return fmt(o["text"], **fields)


def t5_edits(res, out: Mapping) -> str | None:
    o = out[res.verdict]
    return fmt(o["edits"], max_partition=res.max_label) if "edits" in o else None


# --- T3 --------------------------------------------------------------------------------------------


def t3_text(res, out: Mapping, main_scheme: str) -> str:
    o = out[res.verdict]
    m = res.per[main_scheme]
    passed = [s for s, r in res.per.items() if r.passed]
    return fmt(
        o["text"],
        n_reliable=style.fmt_num(m.n_reliable),
        share=style.fmt_pct(m.n_reliable / res.n_nodes, 1),
        placebo_median=style.fmt_num(m.median, 0),
        p95=style.fmt_num(m.p95, 0),
        excess=style.fmt_num(res.excess, 0),
        excess_share=style.fmt_pct(res.excess_share, 1),
        excess_ci=ci_text((m.lo, m.hi), lambda v: style.fmt_num(v, 0)),
        one_in_ten_phrase=ONE_IN_TEN_PHRASE if res.one_in_ten else "",
        passed_halves=", ".join(SCHEME_LABELS[s] for s in passed) or "—",
    )


# --- T2 --------------------------------------------------------------------------------------------


def _p0_text(s, out: Mapping, min_pool: int) -> str:
    if s.undetermined:
        return fmt(
            out["not"]["p0_undetermined"], n_pool=style.fmt_num(s.n_pool), min_pool=style.fmt_num(min_pool)
        )
    return style.fmt_num(s.p0, 2)


def t2_text(res, out: Mapping, direction: str, main_scheme: str, min_pool: int) -> str:
    key = res.text_key
    o = out[key]
    per = res.per
    m = per[main_scheme]
    if key in ("confirmed", "partial_capped", "not"):
        text = fmt(
            o["text"],
            direction=direction,
            n_up=style.fmt_num(m.n_up),
            n_down=style.fmt_num(m.n_down),
            share_up=style.fmt_num(m.share_up, 2),
            p0=_p0_text(m, out, min_pool),
        )
    elif key == "partial":
        passed = [s for s, r in per.items() if r.up_sig][0]
        other = [s for s in per if s != passed][0]
        so = per[other]
        if so.undetermined:
            oh = fmt(
                o["other_halves"]["undetermined"],
                n_pool=style.fmt_num(so.n_pool),
                min_pool=style.fmt_num(min_pool),
            )
        elif so.down_sig:
            oh = fmt(o["other_halves"]["down"])
        else:
            oh = fmt(o["other_halves"]["not"])
        sp = per[passed]
        text = fmt(
            o["text"],
            direction=direction,
            passed_halves=SCHEME_LABELS[passed],
            share_up=style.fmt_num(sp.share_up, 2),
            p0=style.fmt_num(sp.p0, 2),
            other_halves=oh,
        )
    else:  # down
        s = m if m.down_sig else [r for r in per.values() if r.down_sig][0]
        text = fmt(o["text"], share_up=style.fmt_num(s.share_up, 2), p0=style.fmt_num(s.p0, 2))
    if res.capped:
        pl = res.place
        text = (
            text
            + " "
            + fmt(
                out["place_override"]["text"],
                share_a=style.fmt_pct(pl["share_a"], 1),
                share_null=style.fmt_pct(pl["share_null"], 1),
            )
        )
    return text


# --- T6, T7, T4 ------------------------------------------------------------------------------------


def flows_text(flows: Sequence[Sequence[int]]) -> str:
    """Потоки T6 (``movers``) словами: [[2, 1], [1, 3]] -> «из типа 2 в тип 1 и из типа 1 в тип 3»."""
    parts = [f"из типа {int(a)} в тип {int(b)}" for a, b in flows]
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " и " + parts[-1]


def t6_text(res: Mapping, out: Mapping, flows: Sequence[Sequence[int]]) -> str:
    """Текст исхода T6 — только о МО потоков ``movers`` против оставшихся того же исходного типа.

    Слов о направлении (``{direction}``) здесь нет: текст стоит на экране 0 при любом исходе T2 и T3
    (поправка до вскрытия 30.09.2026, ``amendments.py``)."""
    o = out[res["verdict"]]
    fields = {"flows": flows_text(flows)}
    if "{delta}" in o["text"]:
        fields["delta"] = style.fmt_num(res["delta"], 2)
    if "{ci}" in o["text"]:
        fields["ci"] = ci_text(res["ci"])
    if "{p}" in o["text"]:
        fields["p"] = style.fmt_p(res["p"])
    return fmt(o["text"], **fields)


def t6_caveat(eps2: float, out: Mapping) -> str:
    return fmt(out["caveat"]["text"], eps2=style.fmt_num(eps2, 3))


def t7_text(res: Mapping, out: Mapping) -> str:
    P = res["product"]
    o = out[res["verdict"]]
    med = res["median_error"]
    f3 = lambda v: style.fmt_num(v, 3)  # noqa: E731
    if res["verdict"] == "confirmed":
        text = fmt(o["text"], set_name=SET_NAMES[P], err_P=f3(med[P]), err_B=f3(med["B"]), err_C=f3(med["C"]))
    elif res["verdict"] == "partial":
        passed = SET_NAMES["B"] if res["against_B"] else SET_NAMES["C"]
        failed = SET_NAMES["C"] if res["against_B"] else SET_NAMES["B"]
        text = fmt(o["text"], set_name=SET_NAMES[P], passed_set=passed, failed_set=failed)
    else:
        text = fmt(o["text"])
    return cap(text)


def t7_type_text(res: Mapping, out: Mapping) -> str:
    key = {"adds": "type_adds", "neutral": "type_neutral", "hurts": "type_hurts"}[res["type_gain"]]
    point, ci = res["diffs"]["D-A"]
    return fmt(
        out[key]["text"],
        d_minus_a=style.fmt_num(point, 3, sign=True),
        ci_d=ci_text(ci, lambda v: style.fmt_num(v, 3)),
    )


def t4_text(res: Mapping, tpl: str) -> str:
    return fmt(
        tpl,
        ari_basket=style.fmt_num(res["ari_basket"], 2),
        ari_place=style.fmt_num(res["ari_place"], 2),
        diff=style.fmt_num(res["diff"], 2, sign=True),
        ci=ci_text(res["ci"]),
    )


def finite(x: float) -> bool:
    return bool(np.isfinite(x))


# --- Сверка шаблонов исходов с кодом --------------------------------------------------------------

_T1 = {"rho_cond", "best_partition", "rho_best"}
_T2 = {"direction", "n_up", "n_down", "share_up", "p0"}
_T5 = {"max_partition", "max_ami", "turnovers", "eps2", "best_partition", "eps2_best"}
_T7_TYPE = {"d_minus_a", "ci_d"}
ALLOWED_FIELDS: dict[str, dict[str, set[str]]] = {
    "T1_ladder_external": {
        "confirmed.text": _T1,
        "confirmed.edits": set(),
        "partial_one.text": _T1 | {"turnover_beyond", "other_clause"},
        "partial_one.other_overall": {"turnover_other", "best_partition_other", "rho_best_other"},
        "partial_one.other_none": {"turnover_other"},
        "partial_one.edits": {"turnover_beyond"},
        "partial_overall.text": _T1 | {"turnovers_overall", "rho_overall"},
        "partial_overall.edits": set(),
        "not.text": set(),
        "not.edits": set(),
        "proxies.text": {"proxy_summary"},
        "proxies.proxy_item": {"proxy", "values", "monotone"},
    },
    "T2_direction": {
        "confirmed.text": _T2,
        "partial_capped.text": _T2,
        "partial.text": {"direction", "passed_halves", "share_up", "p0", "other_halves"},
        "partial.other_halves.not": set(),
        "partial.other_halves.down": set(),
        "partial.other_halves.undetermined": {"n_pool", "min_pool"},
        "not.text": _T2 - {"direction"},
        "not.p0_undetermined": {"n_pool", "min_pool"},
        "down.text": {"share_up", "p0"},
        "down.edits": set(),
        "place_override.text": {"share_a", "share_null"},
    },
    "T3_reliable_placebo": {
        "confirmed.text": {
            "n_reliable",
            "share",
            "placebo_median",
            "p95",
            "excess",
            "excess_share",
            "excess_ci",
            "one_in_ten_phrase",
        },
        "partial.text": {"n_reliable", "passed_halves", "excess", "excess_ci", "placebo_median"},
        "partial.edits": set(),
        "not.text": {"n_reliable", "placebo_median", "excess", "excess_ci"},
        "not.edits": set(),
    },
    "T5_trivial": {
        "confirmed.text": _T5,
        "partial.text": _T5,
        "not_empty.text": _T5,
        "not_repeats.text": _T5,
        "not_repeats.edits": {"max_partition"},
    },
    "T6_bank_coverage": {  # без {direction}: поправка до вскрытия 30.09.2026 (amendments.py)
        "confirmed.text": {"flows", "delta", "ci"},
        "partial.text": {"flows", "p", "delta", "ci"},
        "not.text": {"flows"},
        "caveat.text": {"eps2"},
    },
    "T7_utility": {
        "confirmed.text": {"set_name", "err_P", "err_B", "err_C"},
        "partial.text": {"set_name", "passed_set", "failed_set"},
        "partial.edits": set(),
        "not.text": set(),
        "not.edits": set(),
        "type_adds.text": _T7_TYPE,
        "type_neutral.text": _T7_TYPE,
        "type_neutral.edits": set(),
        "type_hurts.text": _T7_TYPE,
        "type_hurts.edits": set(),
    },
}


def placeholders(template: str) -> set[str]:
    import string

    return {f for _, f, _, _ in string.Formatter().parse(str(template)) if f}


def _walk(node, prefix: str = ""):
    for key, value in node.items():
        path = f"{prefix}{key}"
        if hasattr(value, "items"):
            yield from _walk(value, path + ".")
        else:
            yield path, value


def validate_outcomes(tests) -> list[str]:
    """Каждое поле ``outcomes`` каждого решающего теста: путь реализован, и все его поля ``{…}`` код
    заполняет.
    Возвращает список расхождений (пустой — шаблоны и код согласованы)."""
    problems = []
    for test, allowed in ALLOWED_FIELDS.items():
        seen = set()
        for path, value in _walk(tests[test]["outcomes"]):
            seen.add(path)
            if path not in allowed:
                problems.append(f"{test}.outcomes.{path}: поле не реализовано")
                continue
            extra = placeholders(value) - allowed[path]
            if extra:
                problems.append(f"{test}.outcomes.{path}: код не заполняет {sorted(extra)}")
        missing = set(allowed) - seen
        if missing:
            problems.append(f"{test}.outcomes: нет полей {sorted(missing)}")
    return problems
