"""Порция 6f лендинга (``docs/landing_spec.md``, §4.13): два совета по размеру муниципалитета.

Источники — этап usefulness: заранее записанная проверка ``usefulness.by_type_test`` (``by_type.json``)
и разведка после её вскрытия ``usefulness.size_posthoc`` (``size_check.json``, ``size_by_mo.csv``).
Этап site ничего не пересчитывает: доли по квинтилям населения, интервалы, границы групп, исход проверки
по типам и флаг «крупный» — из этих файлов.

Вход необязательный, как в порции 6b: нет файлов, режим ``--demo`` / ``--dev-blind`` или нет выходов 6b
(``facts.json`` usefulness) — советов по размеру и строки «крупный» в карточке нет, в логе предупреждение.
Файлы старше ``outputs/usefulness/facts.json`` или посчитаны на другом исходе ``by_type_test``
(``size_check.qc.by_type_main_delta`` ≠ разности основного прогона ``by_type.json``) — код 1
(устаревший вход).

Тексты — ``site.build.texts.size`` (линтуются как текст сайта). Знак разности ошибок у отдельного МО
по-прежнему не показывается: флаг «крупный» — только про население.
"""

from __future__ import annotations

import html
import logging
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from munnet import style
from munnet.contracts import MissingInputError, QCError

log = logging.getLogger(__name__)

INK = "#1d1d1d"
INK2 = "#595959"
ACCENT = "#a3172d"
# полосы групп 1–4: palette_audit.py — контраст к бумаге #fcfbf8 3,52 : 1, ΔL* к акценту 20,8 (у #595959 —
# 2,6: в оттенках серого акцент с ним сливался)
BAR = "#7f8790"
FILES = ("by_type.json", "size_check.json", "size_by_mo.csv")


def _fill(tpl: str, vals: Mapping[str, Any]) -> str:
    out = str(tpl)
    for k, v in vals.items():
        out = out.replace("{" + k + "}", str(v))
    return out


def _e(s: Any) -> str:
    return html.escape(str(s if s is not None else ""), quote=True)


def load(cfg: Any, mode: Any, useful_in: Mapping | None) -> dict | None:
    """Выходы ``by_type_test`` и ``size_posthoc`` или None (советов по размеру нет).

    Код 1 (``MissingInputError``): файл старше ``facts.json`` usefulness или ``size_check.json`` посчитан
    на другом исходе ``by_type_test``."""
    import json

    if mode.name != "normal":
        log.warning(
            "site (%s): советы по размеру муниципалитета не показываются — нужен обычный режим", mode.name
        )
        return None
    d = cfg.dir("outputs") / "usefulness"
    missing = [f for f in FILES if not (d / f).exists()]
    if missing:
        log.warning(
            "site: нет %s в %s — советы по размеру муниципалитета не показываются "
            "(usefulness.by_type_test, usefulness.size_posthoc)",
            ", ".join(missing),
            d,
        )
        return None
    if useful_in is None:
        log.warning(
            "site: нет выходов usefulness порции 6b — советы по размеру муниципалитета не показываются"
        )
        return None
    fp = d / "facts.json"
    for f in FILES:
        if fp.exists() and (d / f).stat().st_mtime < fp.stat().st_mtime:
            raise MissingInputError(
                f"site: {d / f} старше {fp} — перезапустите usefulness.by_type_test и size_posthoc"
            )
    with open(d / "by_type.json", encoding="utf-8") as f:
        bt = json.load(f)
    with open(d / "size_check.json", encoding="utf-8") as f:
        sc = json.load(f)
    if bt.get("status") != "done" or sc.get("status") != "done":
        log.warning("site: by_type.json или size_check.json не в статусе done — советов по размеру нет")
        return None
    main = (bt.get("runs") or {}).get("main") or {}
    want = (sc.get("qc") or {}).get("by_type_main_delta")
    if want is None or main.get("delta") is None or abs(float(want) - float(main["delta"])) > 1e-12:
        raise MissingInputError(
            "site: size_check.json посчитан на другом исходе by_type_test (qc.by_type_main_delta) — "
            "перезапустите usefulness.size_posthoc"
        )
    q = ((cfg["usefulness"].get("size_posthoc") or {}).get("quintiles")) or {}
    if int(q.get("n") or 0) != 5 or q.get("large") != "top":
        log.warning(
            "site: usefulness.size_posthoc.quintiles не «5 групп, крупные — верхняя» — советов по размеру нет"
        )
        return None
    by_mo = pd.read_csv(d / "size_by_mo.csv")
    return {"by_type": bt, "size": sc, "by_mo": by_mo, "dir": Path(d)}


def large_ids(by_mo: pd.DataFrame) -> set[int]:
    """Узлы с флагом ``large`` (верхний квинтиль населения 2023 года)."""
    lg = by_mo["large"].astype(str).str.lower().isin(["true", "1"])
    return {int(i) for i in by_mo.loc[lg, "territory_id"]}


def check_large(by_mo: pd.DataFrame, sc: Mapping) -> list[str]:
    """Флаг «крупный» согласован с границей верхнего квинтиля и счётом ``by_mo.n_large`` (иначе код 3)."""
    bad = []
    bound = float(sc["quintile_bounds"][-1])
    lg = by_mo["territory_id"].astype(int).isin(large_ids(by_mo))
    pop = by_mo["pop_avg_2023"].astype(float)
    if (pop[lg] < bound).any() or (pop[~lg] >= bound).any():
        bad.append("size_by_mo.csv: флаг large не совпадает с границей верхнего квинтиля")
    n = (sc.get("by_mo") or {}).get("n_large")
    if n is not None and int(n) != int(lg.sum()):
        bad.append(f"size_by_mo.csv: {int(lg.sum())} крупных против by_mo.n_large = {n}")
    return bad


def _thousands(x: float) -> str:
    """53 546 → «53,5 тыс.» (граница группы населения)."""
    return f"{style.fmt_num(float(x) / 1000.0, 1)} тыс."


def group_labels(bounds: Sequence[float]) -> list[str]:
    """Подписи пяти групп населения по границам квинтилей: «до 11,7 тыс. жителей», «11,7–18,1 тыс.», …"""
    b = [_thousands(x).replace(" тыс.", "") for x in bounds]
    out = [f"до {b[0]} тыс. жителей"]
    out += [f"{b[i]}–{b[i + 1]} тыс." for i in range(len(b) - 1)]
    out.append(f"от {b[-1]} тыс.")
    return out


def mo_count(n: int) -> str:
    from munnet.site_useful import mo_count as _mc

    return _mc(n)


def size_texts(
    tx: Mapping, bt: Mapping, sc: Mapping, names: Mapping[str, str], k: int = 10
) -> dict[str, Any] | None:
    """Два совета, подписи графика, строка исхода проверки по типам. None — нет текстов или исход основного
    прогона не «confirmed» (тексты написаны под этот исход). ``k`` — размер набора «свой регион» (ближайшие
    муниципалитеты того же региона, ``interpret.tests.T7_utility.k``)."""
    if not tx:
        return None
    qs = sorted(sc["quintiles"], key=lambda q: int(q["quintile"]))
    if len(qs) != 5:
        return None
    main = bt["runs"]["main"]
    if main.get("verdict") != "confirmed":
        log.warning(
            "site: by_type_test основного прогона — %r, советы по размеру написаны под confirmed",
            main.get("verdict"),
        )
        return None
    big = qs[4]
    bound = float(sc["quintile_bounds"][-1])
    # порция 6g: нижние четыре группы — одной долей (lower_four) с интервалом при общих соседях в наборах
    # (share_ci_shared_centered), верхняя — с обычным интервалом квинтиля (тот же, что на графике); диапазоны
    # не переносятся по тире (WORD JOINER по обе стороны)
    lf = sc.get("lower_four") or {}
    if lf.get("share") is None or not lf.get("share_ci_shared_centered"):
        log.warning("site: в size_check.json нет lower_four — советов по размеру нет (size_posthoc)")
        return None
    wj = "⁠"
    slo, shi = (float(x) for x in lf["share_ci_shared_centered"])
    # тот же способ, что у совета небольшим: бутстрап с общими членами D вокруг точечной доли (есть всегда
    # вместе с lower_four); обычный интервал — только на графике
    blo, bhi = (float(x) for x in (big.get("share_ci_shared_centered") or big["share_ci"]))
    vals = {
        "small_share": style.fmt_pct(lf["share"], 1),
        "small_lo": style.fmt_num(100 * slo, 1) + wj,
        "small_hi": wj + style.fmt_pct(shi, 1),
        "threshold": _thousands(bound),
        "share": style.fmt_pct(big["share"], 1),
        "large_lo": style.fmt_num(100 * blo, 1) + wj,
        "large_hi": wj + style.fmt_pct(bhi, 1),
        "k": style.fmt_num(k),
    }
    # оговорка о пороге «самые крупные»: интервал разности «верхняя группа − остальные» при общих соседях
    lv = ((sc.get("shared_members_bootstrap") or {}).get("large_vs_rest") or {}).get("delta_ci")
    caveat = ""
    if lv:
        dlo, dhi = (100 * float(x) for x in lv)
        key = "large_caveat" if dhi >= 0 else "large_caveat_below"
        if tx.get(key):
            caveat = _fill(
                tx[key],
                {
                    "d_lo": style.fmt_num(dlo, 1, sign=True),
                    "d_hi": style.fmt_num(dhi, 1, sign=True),
                    "small_share": vals["small_share"],  # порция 6j: «56,3% против 50% при равной точности»
                },
            )
    labels = group_labels(sc["quintile_bounds"])
    rows = [
        {
            "label": labels[i],
            "share": float(q["share"]),
            "lo": float(q["share_ci"][0]),
            "hi": float(q["share_ci"][1]),
            "text": style.fmt_pct(q["share"], 1),
            "large": i == 4,
        }
        for i, q in enumerate(qs)
    ]
    verdicts = list((bt.get("verdicts") or {}).values())
    hi_types = [str(t) for t in (bt.get("groups") or {}).get("high") or []]
    ct = sc.get("crosstab_types_large") or {}
    n_large = int(ct.get("types_hi_large", 0)) + int(ct.get("types_lo_large", 0))
    tnames = " и ".join(f"«{names.get(t, 'Тип ' + t)}»" for t in hi_types)
    if not verdicts or any(v != "confirmed" for v in verdicts) or not n_large:
        # в вывод идёт наименьший вердикт из всех прогонов (usefulness.by_type_test): текст — под «во всех»
        log.warning("site: by_type_test подтверждена не во всех прогонах — советов по размеру нет")
        return None
    # свой текст (site.build.texts.size.by_type), а не by_type.json main_text: числа — только runs.main
    # (среднее по трём новым показателям), розница в них не входит
    ci = sorted(abs(100 * float(x)) for x in main["delta_ci_adj"])
    by_type = _fill(
        tx["by_type"],
        {
            "n": style.fmt_num(len(verdicts)),
            "types": tnames,
            "share_hi": style.fmt_pct(main["share_hi"], 1),
            "share_lo": style.fmt_pct(main["share_lo"], 1),
            "delta": style.fmt_num(abs(100 * float(main["delta"])), 1),
            "ci_lo": style.fmt_num(ci[0], 1),
            "ci_hi": style.fmt_num(ci[1], 1),
            "large_hi": style.fmt_pct(int(ct.get("types_hi_large", 0)) / n_large, 0),
        },
    )
    # 03.10 (совет судей): крупные МО и типы 3–4 не разводятся — число из crosstab_types_large, без слов
    # «разницу объясняет размер»
    types_size = _fill(
        tx["types_size"],
        {"n_hi_large": style.fmt_num(int(ct.get("types_hi_large", 0))), "n_large": style.fmt_num(n_large)},
    )
    # порция 6m (совет 04.10): тип при равном размере — перестановка типов внутри групп «регион × пятая часть»
    pl = ((sc.get("type_given_size") or {}).get("permutation") or {}).get("p_less")
    type_perm = ""
    if pl is not None and tx.get("type_perm"):
        type_perm = _fill(tx["type_perm"], {"p": style.fmt_num(float(pl), 2)})
    return {
        "advice_small": _fill(tx["advice_small"], vals),
        "advice_large": _fill(tx["advice_large"], vals),
        "large_caveat": caveat,
        "lead_small": tx.get("lead_small", ""),
        "lead_large": tx.get("lead_large", ""),
        "chart": _fill(
            tx["chart"], {"n_mo": mo_count(int(sc.get("n_base") or sum(int(q["n_base"]) for q in qs)))}
        ),
        "types_size": types_size,
        "type_perm": type_perm,
        "by_type": by_type,
        "more": tx["more"],
        "aria": _fill(tx["aria"], {"vals": "; ".join(f"{r['label']} — {r['text']}" for r in rows)}),
        "ref": tx["ref"],
        "rows": rows,
        "threshold": vals["threshold"],
        "small_share": vals["small_share"],
        "share": vals["share"],
        "k": vals["k"],
        "large_pop": bound,
    }


def card_notes(tx: Mapping, st: Mapping | None, n_shown: int = 5) -> dict[str, Any]:
    """Порция 6j (check-ux): строки карточки «С кем сверять» — как совет по размеру. ``simb_note`` — для МО
    меньше порога, ``simb_note_large`` — для крупных (порог ``large_pop``, флаг ``lg`` в ``mo.json``);
    прежние строки о рознице под списком похожих (``sim_note``) и строка «крупный» (``large_note``)
    снимаются — их смысл теперь в строке над списками. Нет текстов — пустой словарь (строки карточки
    как в порции 6d)."""
    if not st or not tx.get("card_small") or not tx.get("card_large"):
        return {}
    f = {k: st[k] for k in ("threshold", "small_share", "share", "k")} | {"n_shown": style.fmt_num(n_shown)}
    return {
        "simb_note": _fill(tx["card_small"], f),
        "simb_note_large": _fill(tx["card_large"], f),
        # порция 6l (check-ux 04.10): у крупных заголовок блока — про оба ориентира
        "simb_title_large": str(tx.get("card_title_large") or ""),
        "sim_note": "",
        "large_note": "",
        "large_pop": float(st["large_pop"]),
    }


def example_note(tx: Mapping, by_mo: pd.DataFrame, ex_id: int, name: str) -> str | None:
    """Порция 6l (check-ux 03.10): пример пользы (``usefulness.pick_example``) выбран на рознице, где размер
    ещё не учитывали. Если это крупный муниципалитет (флаг ``large`` в ``size_by_mo.csv``), а совет над ним —
    небольшим и средним, строка ``size.ex_large`` говорит об этом: население 2023 года и совет для крупных.
    Пример не крупный, нет шаблона или узла в базе — None (строки нет)."""
    tpl = tx.get("ex_large")
    row = by_mo[by_mo["territory_id"].astype(int) == int(ex_id)]
    if not tpl or row.empty or int(ex_id) not in large_ids(by_mo):
        return None
    return _fill(str(tpl), {"name": name, "pop": _thousands(float(row["pop_avg_2023"].iloc[0]))})


def strings(st: Mapping | None, card: str = "") -> list[str]:
    """Все показываемые строки 6f — для линта (``landing.lint_texts``)."""
    out = [card]
    if st:
        keys = ("advice_small", "advice_large", "large_caveat", "lead_small", "lead_large", "chart",
                "types_size", "type_perm", "by_type", "more", "aria", "ref")  # fmt: skip
        out += [str(st[k]) for k in keys] + [r["label"] for r in st["rows"]]
    return [s for s in out if s]


# --- график ---------------------------------------------------------------------------------------------


def _text(x: float, y: float, s: str, cls: str = "", anchor: str = "start") -> str:
    c = f' class="{cls}"' if cls else ""
    a = f' text-anchor="{anchor}"' if anchor != "start" else ""
    return f'<text x="{x:.1f}" y="{y:.1f}"{c}{a}>{_e(_nb(s))}</text>'


def _nb(s: str) -> str:
    """Типографика ru-text в подписях SVG (порция 6j): неразрывный пробел после однобуквенных слов и перед
    тире — та же функция, что для HTML (``landing.nbsp``)."""
    from munnet.landing import nbsp

    return nbsp(str(s)) if s else s


def size_svg(st: Mapping, W: float = 440) -> str:
    """Пять полос «доля случаев, где свой регион точнее» по группам населения: от нуля, общая шкала без
    подложки (подложка до края подсказала бы потолок шкалы), пунктир 50% («при равной точности»),
    95% интервал — тонкая линия с засечками; подпись группы — над полосой, значение — сразу за интервалом;
    самая крупная группа — акцентом (о ней второй совет)."""
    rows = st["rows"]
    row, top = 36, 18
    x0, x1 = 0.0, W - 58
    hi_all = max(r["hi"] for r in rows)
    dom = max(0.7, math.ceil(hi_all * 10) / 10)
    sx = lambda v: x0 + v / dom * (x1 - x0)  # noqa: E731
    H = top + row * len(rows) + 22
    g = []
    x50 = sx(0.5)
    for i, r in enumerate(rows):
        y = top + i * row
        col = ACCENT if r["large"] else BAR
        g.append(_text(0, y + 11, r["label"], "lab th" if r["large"] else "lab"))
        g.append(
            f'<rect x="{x0:.1f}" y="{y + 16}" width="{sx(r["share"]) - x0:.1f}" height="10" fill="{col}"/>'
        )
        yc = y + 21
        a, b = sx(r["lo"]), sx(r["hi"])
        g.append(
            f'<path d="M{a:.1f},{yc - 6}V{yc + 6}M{a:.1f},{yc}H{b:.1f}M{b:.1f},{yc - 6}V{yc + 6}" '
            f'stroke="{INK}" stroke-width="1" fill="none"/>'
        )
        g.append(_text(max(a, b) + 6, y + 26, r["text"], "val"))  # значение — сразу за интервалом
    yb = top + row * len(rows)
    g.append(
        f'<line x1="{x50:.1f}" x2="{x50:.1f}" y1="{top - 4}" y2="{yb + 2}" stroke="{INK}" '
        f'stroke-width="1" stroke-dasharray="3 3"/>'
    )
    g.append(_text(x50, top - 8, st["ref"], "axis", "middle"))
    # порция 6l (check-ux): деление 60% — столбики доходят до 58%
    for t in (0.0, 0.25, 0.5, 0.6):
        g.append(_text(sx(t), yb + 16, style.fmt_pct(t, 0), "axis", "start" if t == 0 else "middle"))
    return (
        f'<svg class="fd-svg size-svg" viewBox="0 0 {W:g} {H:g}" role="img" aria-label="{_e(st["aria"])}" '
        f'xmlns="http://www.w3.org/2000/svg">{"".join(g)}</svg>'
    )


def check_rows(st: Mapping) -> None:
    """Доли и интервалы — в [0, 1], интервал содержит долю (иначе код 3)."""
    for r in st["rows"]:
        if not (0 <= r["lo"] <= r["share"] <= r["hi"] <= 1):
            raise QCError(f"site: size_check.json: доля {r['share']} вне интервала {r['lo']}–{r['hi']}")
