"""Порция 5b лендинга (``docs/landing_spec.md``, §4.8): блок «Что устояло и чем полезно» под первым экраном,
строки карточки МО о сверке с соседями по региону и паспорта типов главы 3.

Страница ничего не считает: числа берутся из ``facts.json`` этапа 5 и выгрузок сайта (``checks.json``,
``types.json``), слова — из ``site.build.texts`` (линтуются как текст сайта). Тексты блока написаны после
вскрытия под настоящие вердикты, поэтому блок собирается, только если вердикты — из ``findings.requires``
и выполнены числовые условия (порядок у соперника сильнее, интервал разности ошибок не пересекает ноль);
иначе его нет.
"""

from __future__ import annotations

import html
import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from munnet import style
from munnet.site_chapters import VARIANT_WORDS as _VW

Esc = Callable[[str], str]
NB = " "
INK = "#1d1d1d"
INK2 = "#595959"
GREY = "#8d96a1"
RULE = "#d6d3cc"
# варианты расчёта R1 (node_r1.csv) словами — один словарь с главой 7 (site_chapters.VARIANT_WORDS)
VARIANT_WORDS = {k: v for k, v in _VW.items() if k.startswith("variant:")}
PASSPORT_ROWS = ("clr_rel_cafe", "clr_rel_food", "clr_rel_marketplace", "clr_rel_transport", "log_level_rel")


def _e(s: Any) -> str:
    return html.escape(str(s if s is not None else ""), quote=True)


def _f(x: float, d: int = 0) -> str:
    return style.fmt_num(x, d)


def _fill(tpl: str, vals: Mapping[str, str]) -> str:
    out = tpl
    for k, v in vals.items():
        out = out.replace("{" + k + "}", v)
    return out


def _dot(s: str) -> str:
    s = s.rstrip()
    return s if not s or s[-1] in ".!?…" else s + "."


def _svg(w: float, h: float, body: str, label: str, cls: str) -> str:
    return (
        f'<svg class="{cls}" viewBox="0 0 {w:g} {h:g}" role="img" aria-label="{_e(label)}" '
        f'xmlns="http://www.w3.org/2000/svg">{body}</svg>'
    )


def _text(x: float, y: float, s: str, cls: str = "", anchor: str = "start") -> str:
    c = f' class="{cls}"' if cls else ""
    a = f' text-anchor="{anchor}"' if anchor != "start" else ""
    return f'<text x="{x:.1f}" y="{y:.1f}"{c}{a}>{_e(_nb(s))}</text>'


def _nb(s: str) -> str:
    """Типографика ru-text в подписях SVG (порция 6j): неразрывный пробел после однобуквенных слов и перед
    тире — та же функция, что для HTML (``landing.nbsp``)."""
    from munnet.landing import nbsp

    return nbsp(str(s)) if s else s


def rel_text(v: float | None) -> str:
    """Медиана в логарифмах относительно региона словами: |v| ≥ 0,4 — «примерно в 1,5 раза больше»,
    иначе проценты («примерно на 17% меньше»), 0% — «как в регионе» (как подписи островов первого экрана).
    «Примерно»: exp CLR — приблизительное отношение долей (доля нормирована на всю корзину),
    как в отчёте, §8."""
    if v is None or not math.isfinite(v):
        return "—"
    r = math.exp(v)
    if abs(v) >= 0.4:
        k = r if v > 0 else 1 / r
        return f"примерно в{NB}{_f(k, 1)} раза {'больше' if v > 0 else 'меньше'}"
    pct = round((r - 1) * 100)
    if pct == 0:
        return "как в регионе"
    return f"примерно на{NB}{abs(pct)}% {'больше' if pct > 0 else 'меньше'}"


# --- числа блока ----------------------------------------------------------------------------------


def t7_numbers(checks: Mapping) -> dict[str, Any] | None:
    """Сверка соседей по своему региону (B) с набором продукта T7 (A или D) без поправки на регион:
    медианные ошибки ``t7.median_error_abs`` и разность ``t7.diffs["B-<набор>_abs"]`` с 95% интервалом.
    Утверждение «соседи лучше» — только если верхняя граница интервала меньше нуля; иначе None."""
    t7 = checks.get("t7") or {}
    p = t7.get("product")
    err = t7.get("median_error_abs") or {}
    dd = (t7.get("diffs") or {}).get(f"B-{p}_abs")
    if not p or err.get("B") is None or err.get(p) is None or not dd or len(dd) != 2 or len(dd[1]) != 2:
        return None
    diff, (lo, hi) = dd
    if hi is None or not hi < 0:
        return None
    return {
        "err_b": _f(err["B"], 3),
        "err_p": _f(err[p], 3),
        "diff": _f(diff, 3),
        "lo": _f(lo, 3),
        "hi": _f(hi, 3),
        "n": _f(t7.get("n_common") or 0),
        "n_int": int(t7.get("n_common") or 0),
        "k": int(t7.get("k") or 10),
        "raw": {"B": float(err["B"]), "P": float(err[p]), "product": p},
    }


def t1_numbers(facts: Mapping, checks: Mapping) -> dict[str, Any] | None:
    """Розница T1: ρ «в целом» (a), ρ внутри страт у типов (b) и у лучшего деления без типов. Утверждение
    «деление без типов упорядочено сильнее» — только если его ρ больше ρ (b) типов; иначе None."""
    r = ((facts.get("t1") or {}).get("per") or {}).get("retail") or {}
    c = (checks.get("t1") or {}).get("retail") or {}
    need = ("rho_a", "rho_b", "best_rival_rho", "best_rival_label", "n_a")
    if any(r.get(k) is None for k in need) or not c.get("med_a"):
        return None
    if not float(r["best_rival_rho"]) > float(r["rho_b"]):
        return None
    return {
        "rho_a": _f(r["rho_a"], 2),
        "rho_b": _f(r["rho_b"], 2),
        "rival_rho": _f(r["best_rival_rho"], 2),
        "rival": str(r["best_rival_label"]).split(",")[0].strip(),
        "n_a": _f(r["n_a"]),
        "med_a": list(c["med_a"]),
        "line": c.get("line", "none"),
    }


def r1_shares(checks: Mapping) -> list[dict]:
    """Доли «тот же тип» по вариантам расчёта R1 (``node_r1.csv`` через ``checks.r1.variants``),
    по возрастанию."""
    out = []
    for v in (checks.get("r1") or {}).get("variants") or []:
        if v.get("n"):
            out.append(
                {
                    "variant": v["variant"],
                    "what": VARIANT_WORDS.get(v["variant"], v["variant"]),
                    "share": v["same"] / v["n"],
                    "same": int(v["same"]),
                    "n": int(v["n"]),
                }
            )
    return sorted(out, key=lambda x: x["share"])


def findings_texts(
    tx: Mapping,
    story: Mapping,
    facts: Mapping,
    checks: Mapping,
    useful: Mapping | None = None,
    level: Mapping | None = None,
    icvi: Mapping | None = None,
) -> dict[str, Any] | None:
    """Тексты блока с подставленными числами; None — блок не показывается (вердикты не те, что в
    ``requires``, или числовое условие не выполнено). ``useful`` (порция 6b, ``site_useful``) — доля случаев,
    оговорка о наборе R, пример и строка флага; по словам доли текст колонки «Что с этим делать» меняется
    по заранее записанным правкам ``usefulness.rule_share.edits``."""
    if not tx:
        return None
    v = story.get("verdicts") or {}
    for test, levels in (tx.get("requires") or {}).items():
        if v.get(test) not in levels:
            return None
    t1 = t1_numbers(facts, checks)
    t7 = t7_numbers(checks)
    shares = r1_shares(checks)
    t5 = facts.get("t5") or {}
    if t1 is None or t7 is None or not shares or t5.get("max_ami") is None or not t5.get("max_label"):
        return None  # порция 6j: строка «Где граница» говорит AMI ближайшего деления — без него блока нет
    low = shares[0]  # вариант с наименьшей долей «тот же тип»; все варианты — на графике
    variants = _fill(tx["border"]["variant"], {"what": low["what"], "pct": style.fmt_pct(low["share"], 1)})
    t3_head = _dot(str((story.get("screen0") or {}).get("lead") or ""))
    u = tx["use"]
    ut = (useful or {}).get("use")
    # правки по словам доли B против D (usefulness.rule_share.edits): more_often — текст как был
    d_key = (ut or {}).get("d_key", "more_often")
    text_key = {"about_half": "text_about_half", "less_often": "text_less_often"}.get(d_key, "text")
    title = u["title"]
    if d_key == "less_often":
        title = u["title_less_often"]
    elif ut and ut.get("r_key") != "more_often":  # edits.vs_R_not_more_often
        title = u["title_region"]
    nums = {k: t7[k] for k in ("err_b", "err_p", "diff", "lo", "hi")} | {"n_mo": _mo_count(t7["n_int"])}
    use = {
        **{k: u[k] for k in ("label", "note", "bar_b", "bar_p")},
        "title": title,
        "chart": _fill(u["chart"], {"k": _f(t7["k"])}),
        "text": _fill(u[text_key], nums),
    }
    if ut:
        use |= {k: ut.get(k, "") for k in ("share", "r_line", "rules", "r_edit")}
    if (useful or {}).get("size"):  # порция 6f: два совета по размеру МО (site_size)
        use["size"] = useful["size"]
    # 6o (решение участника 05.10): для какого решения совет — пример, где оба ориентира далеко
    why = ((useful or {}).get("example_small") or {}).get("why")
    if why:
        use["why"] = why
    border_flag = (useful or {}).get("flag")
    return {
        "title": tx["title"],
        "intro": tx.get("intro", ""),
        "stood": {
            **{k: tx["stood"][k] for k in ("label", "title", "chart", "chart_note")},
            "text": _fill(
                tx["stood"]["text"], {k: t1[k] for k in ("rho_a", "rho_b", "rival_rho", "rival", "n_a")}
            ),
        }
        # порция 6l: ρ уровня трат с оборотом Росстата рядом с ρ типов (stood.level, level_rho.json)
        | level_texts(tx["stood"], level, facts),
        "border": {
            **{k: tx["border"][k] for k in ("label", "title", "chart")},
            "text": _fill(
                tx["border"]["text"],
                {"variants": variants, "t3_head": t3_head}
                # порция 6j: «почти ничего не добавляют» → ближайшее деление и AMI из facts.t5
                | {
                    "ami": _f(float(t5["max_ami"]), 2),
                    "ami_label": str(t5["max_label"]),
                },
            )
            # порция 6e (check-ux): что такое плацебо — одной фразой после заголовка T3, если он о плацебо
            + (
                " " + _dot(str(tx["border"]["placebo"]))
                if tx["border"].get("placebo") and "плацебо" in t3_head.lower()
                else ""
            ),
            # порция 6m (совет 04.10): разделение типов в сети корзин и в признаках (report_facts)
            # 6o (совет 05.10): плюс точность правила дерева этапа interpret (facts.tree.accuracy)
            **sep_texts(tx["border"].get("sep"), icvi, (facts.get("tree") or {}).get("accuracy")),
            **(
                {"flag": border_flag, "flag_label": (useful or {}).get("flag_label", "")}
                if border_flag
                else {}
            ),
        },
        "use": use,
        "_t1": t1,
        "_t7": t7,
        "_shares": shares,
        "_example": (useful or {}).get("example"),
        "_example_small": (useful or {}).get("example_small"),  # порция 6m: пример под советом
    }


def sep_texts(tpl: str | None, icvi: Mapping | None, tree_acc: float | None = None) -> dict[str, str]:
    """Порция 6m (совет 04.10): «по корзине трат типы идут друг за другом вдоль одной оси…, по уровню трат
    и экономике места чётких границ нет» — доля силы связей внутри типа (AVI итога и её случайный базис)
    и силуэт итога из ``outputs/evaluate/report_facts.json`` (``icvi.final_avi``, ``icvi.final_avi_base``,
    ``icvi.final_sw``); 6o — точность правила дерева на перекрёстной проверке ``{tree_acc}``
    (``facts.tree.accuracy`` этапа interpret). Нет шаблона или чисел — пусто."""
    keys = ("icvi.final_avi", "icvi.final_avi_base", "icvi.final_sw")
    if not tpl or not icvi or any((icvi.get(k) or {}).get("value") is None for k in keys):
        return {}
    if "{tree_acc}" in str(tpl) and tree_acc is None:
        return {}
    val = lambda k: float(icvi[k]["value"])  # noqa: E731
    sw = icvi["icvi.final_sw"].get("text") or _f(val("icvi.final_sw"), 3)
    return {
        "sep": _fill(
            str(tpl),
            {
                "avi": style.fmt_pct(val("icvi.final_avi"), 0),
                "base": style.fmt_pct(val("icvi.final_avi_base"), 0),
                "sw": str(sw),
                "k": _f(int(icvi.get("knn_k") or 10)),
                "tree_acc": style.fmt_pct(float(tree_acc), 0) if tree_acc is not None else "",
            },
        )
    }


def level_texts(stx: Mapping, lv: Mapping | None, facts: Mapping) -> dict[str, str]:
    """Порция 6l: строка «уровень трат связан с розницей сильнее типов» (``stood.level``) и, если по общепиту
    интервал разности захватывает ноль, — ``stood.level_same``; метка — поле ``note`` файла до двоеточия.
    Строка только при 95% интервале разности «уровень − типы» по рознице выше нуля; иначе пусто (текст написан
    под этот исход). ρ типов по рознице в файле не совпал с ``facts.t1`` — код 3."""
    from munnet.contracts import QCError

    if not lv or not stx.get("level"):
        return {}
    t = lv.get("turnovers") or {}
    r, c = t.get("retail") or {}, t.get("catering") or {}
    rho_a = (((facts.get("t1") or {}).get("per") or {}).get("retail") or {}).get("rho_a")
    if rho_a is None or not r.get("types") or not r.get("level") or not r.get("diff_level_minus_types"):
        return {}
    if abs(float(r["types"]["rho"]) - float(rho_a)) > 5e-4:
        raise QCError(f"site: level_rho.json: ρ типов по рознице {r['types']['rho']} ≠ facts.t1 {rho_a}")
    if not float(r["diff_level_minus_types"]["ci"][0]) > 0:
        return {}
    nums = {"level_r": _f(r["level"]["rho"], 2), "types_r": _f(r["types"]["rho"], 2)}
    text = _fill(str(stx["level"]), nums)
    dc = (c.get("diff_level_minus_types") or {}).get("ci")
    if stx.get("level_same") and dc and float(dc[0]) <= 0 <= float(dc[1]):
        text = (
            _dot(text)
            + " "
            + _fill(
                str(stx["level_same"]),
                {"level_c": _f(c["level"]["rho"], 2), "types_c": _f(c["types"]["rho"], 2)},
            )
        )
    label = str(lv.get("note") or "").split(":")[0].strip()
    return {"level": text, "level_label": label}


def _mo_count(n: int) -> str:
    """«1542 муниципалитета» (порция 6b: было «1542 муниципалитетов»)."""
    return f"{_f(n)} {_plural(n)}"


def strings(ft: Mapping | None) -> list[str]:
    """Все показываемые строки блока — для линта (``landing.lint_texts``)."""
    if not ft:
        return []
    out = [ft["title"], ft["intro"]]
    for part in ("stood", "border", "use"):
        out += [str(x) for k, x in ft[part].items() if isinstance(x, str)]
    from munnet import site_size

    out += site_size.strings(ft["use"].get("size"))
    return out


def titles(ft: Mapping | None) -> list[str]:
    """Заголовки блока и колонок (h2, h3) — для запретов в заголовках (``headlines_always``)."""
    if not ft:
        return []
    return [ft["title"]] + [str(ft[p]["title"]) for p in ("stood", "border", "use")]


# --- графики блока --------------------------------------------------------------------------------


def _shape(t: int, x: float, y: float, r: float, color: str, shapes: Mapping) -> str:
    s = shapes.get(str(t), "●")
    if s == "▲":
        return (
            f'<polygon points="{x:.1f},{y - r * 1.15:.1f} {x + r * 1.1:.1f},{y + r * 0.8:.1f} '
            f'{x - r * 1.1:.1f},{y + r * 0.8:.1f}" fill="{color}" stroke="#fff" stroke-width="1"/>'
        )
    if s == "■":
        return (
            f'<rect x="{x - r * 0.85:.1f}" y="{y - r * 0.85:.1f}" '
            f'width="{r * 1.7:.1f}" height="{r * 1.7:.1f}" '
            f'fill="{color}" stroke="#fff" stroke-width="1"/>'
        )
    if s == "◆":
        q = r * 1.2
        return (
            f'<polygon points="{x:.1f},{y - q:.1f} {x + q:.1f},{y:.1f} '
            f'{x:.1f},{y + q:.1f} {x - q:.1f},{y:.1f}" '
            f'fill="{color}" stroke="#fff" stroke-width="1"/>'
        )
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="{color}" stroke="#fff" stroke-width="1"/>'


LOG_AXIS = "логарифмическая шкала: ×2 — вдвое больше, чем в регионе"
LOG_AXIS_SHORT = "логарифмическая шкала"  # телефон: холст 320


def retail_svg(story: Mapping, order: Sequence[int], t1: Mapping, title: str, W: float = 440) -> str:
    """Медианы розничного оборота на жителя по типам к своему региону (лог-шкала, подписи «×0,75»):
    строки — как в главе 4 при ``t1_layout`` (снизу вверх по ``ladder.order``), линия по ``line_by_turnover``
    (пунктир — прошла только проверка «в целом»)."""
    view = story["view"]
    names = story["names"]["final"]
    layout = view.get("t1_layout", "ladder")
    rows_top = (
        list(reversed(order)) if layout != "columns_by_size" else [int(x) for x in view["legend_order"]]
    )
    med = dict(zip(order, t1["med_a"], strict=False))
    stacked = W < 400  # телефон (порция 6a): название типа — над своей точкой, шкала во всю ширину
    lab_w, row, top = (0, 40, 6) if stacked else (184, 26, 6)
    x0, x1 = (8 if stacked else lab_w + 12), W - 58
    dy = 10 if stacked else 0  # точка ниже подписи
    lo, hi = math.log(0.5), math.log(2.5)
    vals = [v for v in med.values() if v is not None]
    lo, hi = min([lo, *vals]) - 0.05, max([hi, *vals]) + 0.05
    sx = lambda v: x0 + (v - lo) / (hi - lo) * (x1 - x0)  # noqa: E731
    H = top + row * len(rows_top) + 40  # порция 6m: ещё строка — подпись логарифмической шкалы
    g = []
    for m, s in ((0.5, "×0,5"), (1.0, "×1"), (2.0, "×2")):
        x = sx(math.log(m))
        stroke = f'stroke="{INK2}" stroke-dasharray="2 2"' if m == 1 else f'stroke="{RULE}"'
        g.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{top}" y2="{top + row * len(rows_top)}" {stroke}/>')
        g.append(_text(x, top + row * len(rows_top) + 15, s, "axis", "middle"))
    # порция 6m (совет 04.10, judge-c6): подпись шкалы прямо у оси
    ya = top + row * len(rows_top) + 33
    g.append(_text(x0, ya, LOG_AXIS_SHORT, "axis") if stacked else _text(x1, ya, LOG_AXIS, "axis", "end"))
    pts = []
    holes = []  # порция 6i: на телефоне подпись типа над точкой — линия медиан под подписью прерывается
    for j, t in enumerate(rows_top):
        y = top + j * row + row / 2
        nm = str(names.get(str(t)) or f"Тип {t}")
        g.append(_text(0, y + 4 - dy, nm, "lab"))
        lw = len(nm) * 12 * 0.58 + 6  # оценка ширины подписи (≈ 0,58 кегля на знак) с запасом
        holes.append(f'<rect x="-2" y="{y + 4 - dy - 12:.1f}" width="{lw:.1f}" height="16" fill="#000"/>')
        if med.get(t) is not None:
            pts.append((sx(med[t]), y + dy, t))
    line = (view.get("line_by_turnover") or {}).get("retail") or t1.get("line", "none")
    if line in ("solid", "dashed") and layout != "columns_by_size" and len(pts) > 1:
        d = "M" + "L".join(f"{x:.1f},{y:.1f}" for x, y, _ in pts)
        dash = ' stroke-dasharray="4 3"' if line == "dashed" else ""
        mask = ""
        if stacked:
            mid = f"fd-retail-mask-{int(W)}"
            g.insert(
                0,
                f'<defs><mask id="{mid}" maskUnits="userSpaceOnUse" x="0" y="0" width="{W:g}" height="{H:g}">'
                f'<rect x="0" y="0" width="{W:g}" height="{H:g}" fill="#fff"/>{"".join(holes)}</mask></defs>',
            )
            mask = f' mask="url(#{mid})"'
        g.append(f'<path d="{d}" fill="none" stroke="{INK2}" stroke-width="1.4"{dash}{mask}/>')
    for x, y, t in pts:
        g.append(_shape(t, x, y, 5.5, view["type_colors"].get(str(t), INK2), view["shapes"]))
        g.append(_text(W, y + 4, "×" + _f(math.exp(med[t]), 2), "val", "end"))  # значения — столбцом справа
    return _svg(W, H, "".join(g), title, "fd-svg")


def shares_svg(shares: Sequence[Mapping], title: str, W: float = 440) -> str:
    """Доли «тот же тип» по вариантам расчёта: полосы от нуля до 100%, подписи прямо у полос."""
    row, top = 34, 2
    x1 = W - 56
    g = []
    for i, s in enumerate(shares):
        y = top + i * row
        g.append(_text(0, y + 11, s["what"], "lab"))
        g.append(f'<rect x="0" y="{y + 16}" width="{x1:.1f}" height="9" fill="#eeebe4"/>')
        g.append(f'<rect x="0" y="{y + 16}" width="{s["share"] * x1:.1f}" height="9" fill="{INK2}"/>')
        g.append(_text(x1 + 6, y + 25, style.fmt_pct(s["share"], 1), "val"))
    H = top + row * len(shares) + 2
    return _svg(W, H, "".join(g), title, "fd-svg")


def errors_svg(t7: Mapping, bar_b: str, bar_p: str, title: str, W: float = 440) -> str:
    """Медианные ошибки сверки: соседи по региону и набор продукта T7, полосы от нуля, подписи у полос."""
    row, top = 34, 2
    x1 = W - 60
    mx = max(t7["raw"]["B"], t7["raw"]["P"]) * 1.05
    g = []
    for i, (lab, v, s, col) in enumerate(
        ((bar_b, t7["raw"]["B"], t7["err_b"], INK), (bar_p, t7["raw"]["P"], t7["err_p"], "#b8bec7"))
    ):
        y = top + i * row
        g.append(_text(0, y + 11, lab, "lab"))
        g.append(f'<rect x="0" y="{y + 16}" width="{v / mx * x1:.1f}" height="9" fill="{col}"/>')
        g.append(_text(v / mx * x1 + 6, y + 25, s, "val"))
    H = top + row * 2 + 2
    return _svg(W, H, "".join(g), title, "fd-svg")


# телефон (порция 6a): холст 320 единиц — подпись в 11,5 единиц на 343 px ≈ 12,3 px
PHONE_W = 320


def _pair(build: Callable[[float], str]) -> str:
    from munnet.site_chapters import phone_pair

    return phone_pair(build(440), build(PHONE_W))


def findings_html(
    ft: Mapping | None, story: Mapping, order: Sequence[int], esc: Esc, src: str, net_link: str = ""
) -> str:
    """Раздел «Что устояло и чем полезно» (три колонки; на телефоне — одна). Нет текстов — пусто.
    ``net_link`` — ссылка из «Где граница» на схему сети корзин в главе «Типы» (порция 6n)."""
    if not ft:
        return ""
    cols = []
    for key, fig in (
        ("stood", _pair(lambda w: retail_svg(story, order, ft["_t1"], ft["stood"]["chart"], w))),
        ("border", _pair(lambda w: shares_svg(ft["_shares"], ft["border"]["chart"], w))),
        (
            "use",
            _pair(
                lambda w: errors_svg(ft["_t7"], ft["use"]["bar_b"], ft["use"]["bar_p"], ft["use"]["chart"], w)
            ),
        ),
    ):
        p = ft[key]
        extra = ""
        if key == "stood":
            extra = f'<p class="fd-cap">{esc(_dot(p["chart"]))} {esc(_dot(p["chart_note"]))}</p>'
            if p.get("plain") and p.get("plain_label"):  # 6q (совет 06.10): пересказ исхода T1 под подписью
                extra += f'<p class="fd-plain"><b>{esc(p["plain_label"])}:</b> {esc(_dot(p["plain"]))}</p>'
        elif key == "border":
            extra = f'<p class="fd-cap">{esc(_dot(p["chart"]))}</p>'
            if p.get("flag"):  # порция 6b: флаг устойчивости типа (usefulness.type_flag)
                extra += (
                    f'<p class="fd-flag">{esc(_dot(p["flag"]))} '
                    f'<span class="label-note">{esc(p.get("flag_label") or "")}</span></p>'
                )
        else:
            note = _dot(p["note"]) + (" " + _dot(p["rules"]) if p.get("rules") else "")
            extra = f'<p class="fd-cap">{esc(_dot(p["chart"]))}</p><p class="fd-note">{esc(note)}</p>'
        body = f"<p>{esc(_dot(p['text']))}</p>"
        if key == "border" and p.get("sep"):  # порция 6m: разделение типов в сети корзин и в признаках
            body += f'<p class="fd-sep">{esc(_dot(p["sep"]))}</p>'
            if net_link:
                body += f'<p class="fd-netlink"><a href="#netmap">{esc(net_link)}</a></p>'
        if key == "stood" and p.get("level"):  # порция 6l: ρ уровня трат рядом с ρ типов, с меткой
            lab = f' <span class="label-note">{esc(p["level_label"])}</span>' if p.get("level_label") else ""
            body += f'<p class="fd-level">{esc(_dot(p["level"]))}{lab}</p>'
        if key == "use" and p.get("share"):  # порция 6b: доля случаев и оговорка о своём регионе
            body += f'<p class="fd-share">{esc(_dot(p["share"]))}</p><p>{esc(_dot(p["r_line"]))}</p>'
            if p.get("r_edit"):  # порция 6g: откуда заголовок «со своим регионом» (edits.vs_R_not_more_often)
                body += f'<p class="fd-note fd-redit">{esc(_dot(p["r_edit"]))}</p>'
        if key == "use" and p.get("why") and not p.get("size"):  # 6o: без советов по размеру — под текстом
            body += f'<p class="fd-why">{esc(_dot(p["why"]))}</p>'
        if key == "use" and p.get(
            "size"
        ):  # порция 6f: два совета по размеру; розница — под «Где нашли совет»
            old = ""
            if ft.get("_example_small") and ft.get("_example"):  # порция 6m: прежний пример — к рознице
                from munnet import site_useful as _su
                from munnet.site_chapters import phone_pair as _pp

                old = _su.example_html(ft["_example"], esc, _pp, PHONE_W, dom_id="use-example-retail")
            cols.append(_use_size_html(p, body, fig, extra + old, esc))
            continue
        cols.append(
            f'<div class="fd fd-{key}" id="fd-{key}"><p class="fd-lab">{esc(p["label"])}</p>'
            f"<h3>{esc(p['title'])}</h3>"
            f"{body}<figure>{fig}{extra}</figure></div>"
        )
    from munnet import site_useful
    from munnet.site_chapters import phone_pair

    # порция 6m: под советом — пример небольшого МО (usefulness.example_small); прежний пример (розница,
    # Йошкар-Ола) — в раскрывающемся «Где нашли совет»
    small = site_useful.example_small_html(ft.get("_example_small"), esc)
    example = small or site_useful.example_html(ft.get("_example"), esc, phone_pair, PHONE_W)
    return (
        '<section class="findings5" id="findings" aria-labelledby="findings-title">'
        f'<h2 id="findings-title">{esc(ft["title"])}</h2>'
        + (f'<p class="fd-intro">{esc(_dot(ft["intro"]))}</p>' if ft.get("intro") else "")
        + f'<div class="fd-grid">{"".join(cols)}</div>{example}<p class="source">{esc(src)}</p></section>'
    )


def _use_size_html(p: Mapping, body: str, fig: str, extra: str, esc: Esc) -> str:
    """Колонка «Что с этим делать» с советами по размеру (порция 6f): два совета, график по группам населения
    с подписью и пометкой «после вскрытия», строка исхода заранее записанной проверки по типам; прежний текст
    о рознице, доля случаев, оговорка о случайных МО региона и график ошибок — в раскрывающемся блоке."""
    from munnet import site_size

    st = p["size"]
    # 6o, второй круг (check-ux 05.10): «Зачем сверять» — сразу после совета небольшим и средним МО
    why = f'<p class="fd-why">{esc(_dot(p["why"]))}</p>' if p.get("why") else ""
    advice = "".join(
        f'<p class="fd-adv fd-adv-{k}"><b>{esc(st["lead_" + k])}</b> {esc(_dot(st["advice_" + k]))}</p>'
        + (why if k == "small" else "")
        for k in ("small", "large")
    )
    if st.get(
        "large_caveat"
    ):  # выигрыш небольшой, граница выбрана после первых результатов, интервал разности
        advice += f'<p class="fd-note fd-adv-note">{esc(_dot(st["large_caveat"]))}</p>'
    chart = _pair(lambda w: site_size.size_svg(st, w))
    # 03.10 (совет судей): исход проверки по типам и то, что крупные МО — почти те же типы 3–4, одним абзацем
    bytype = _dot(st["by_type"]) + (" " + _dot(st["types_size"]) if st.get("types_size") else "")
    bytype += " " + _dot(st["type_perm"]) if st.get("type_perm") else ""  # порция 6m: p перестановки
    return (
        f'<div class="fd fd-use" id="fd-use"><p class="fd-lab">{esc(p["label"])}</p>'
        f"<h3>{esc(p['title'])}</h3>{advice}"
        f'<figure>{chart}<p class="fd-cap">{esc(_dot(st["chart"]))}</p></figure>'
        f'<p class="fd-bytype">{esc(bytype)}</p>'
        f'<details class="more fd-more"><summary>{esc(st["more"])}</summary>'
        f"{body}<figure>{fig}{extra}</figure></details></div>"
    )


def card_texts(
    tx: Mapping | None, checks: Mapping, k_net: int | None = None, sim_fields: Mapping[str, str] | None = None
) -> dict[str, str]:
    """Строки карточки о сверке: с числами ``t7`` (если условие ``t7_numbers`` выполнено) или без них.
    Порция 6e: откуда список похожих, подпись списка (``sim_caption`` — поля ``n_shown``, ``n_set``),
    переключатель «Соседи по сети корзин» (``k_net`` — ``site.build.n_net_shown``) и ключ дуг."""
    if not tx:
        return {}
    t7 = t7_numbers(checks)
    keys = ("simb_title", "sim_title", "stability", "sim_from", "net_title", "net_toggle", "arcs_default",
            "arcs_net", "type_note", "basket_scale")  # fmt: skip
    out = {k: str(tx[k]) for k in keys if tx.get(k)}
    if tx.get("net_note") and k_net:
        out["net_note"] = _fill(tx["net_note"], {"k": _f(k_net)})
    if tx.get("sim_caption") and sim_fields and all(sim_fields.get(k) for k in ("n_shown", "n_set")):
        out["sim_caption"] = _fill(tx["sim_caption"], sim_fields)
    if t7 is not None:
        out["simb_note"] = _fill(tx["simb_note"], {"err_b": t7["err_b"], "err_p": t7["err_p"]})
        out["sim_note"] = str(tx["sim_note"])
    else:
        out["simb_note"] = str(tx.get("simb_plain") or "")
    return out


# --- паспорта типов (глава 3) ----------------------------------------------------------------------


def passport_rows(ty: Mapping, rows: Mapping[str, str]) -> list[tuple[str, str, float | None]]:
    """Пять строк паспорта: (признак, подпись, медиана типа из ``profile.csv``)."""
    med = {p["feature"]: p.get("median") for p in ty.get("profile") or []}
    return [(f, rows[f], med.get(f)) for f in PASSPORT_ROWS if f in rows]


def example_flag(r: Mapping, flag_words: Mapping[str, str] | None) -> str:
    """Слова флага «тип зависит от варианта расчёта» у примера (поле ``fl`` в ``mo.json`` = ``d``);
    у устойчивого типа и без флага — пусто."""
    return str((flag_words or {}).get("d") or "") if r.get("fl") == "d" else ""


def middle_type(types: Sequence[Mapping]) -> int | None:
    """Тип с медианой доли кафе и ресторанов ближе всего к своему региону
    (|медиана ``clr_rel_cafe``| — наименьшая)."""
    best = None
    for ty in types:
        med = {p["feature"]: p.get("median") for p in ty.get("profile") or []}
        v = med.get("clr_rel_cafe")
        if v is not None and math.isfinite(v) and (best is None or abs(v) < best[0]):
            best = (abs(v), int(ty["t"]))
    return best[1] if best else None


def _pct_rel(v: float) -> int:
    """|exp(v) − 1| в процентах, округлено, как в ``rel_text``."""
    return abs(round((math.exp(v) - 1) * 100))


def middle_note(ty: Mapping, tx: Mapping, middle_t: int | None) -> str:
    """Сноска к середине оси (совет судей 03.10, ``passports.middle``): только у типа ``middle_type`` и только
    если медиана каждой части корзины в паспорте отличается от региона не больше чем на ``middle_max_pct``;
    часть корзины из названия типа (после запятой) — её медиана в процентах. Иначе пусто."""
    tpl = tx.get("middle")
    if not tpl or middle_t is None or int(ty["t"]) != middle_t:
        return ""
    rows = [
        (f, lab, v)
        for f, lab, v in passport_rows(ty, tx["rows"])
        if f.startswith("clr_rel_") and v is not None
    ]
    if not rows:
        return ""
    mx = max(_pct_rel(v) for _, _, v in rows)
    if mx > int(tx.get("middle_max_pct", 5)):
        return ""
    name = str(ty.get("name") or "")
    part = name.split(",", 1)[1].strip() if "," in name else ""
    hit = [(lab, v) for _, lab, v in rows if part and norm_stem(lab) in part.lower()]
    if not hit:
        return ""
    return _fill(
        tpl,
        {"max_pct": f"{mx}%", "part": part[:1].upper() + part[1:], "part_pct": f"{_pct_rel(hit[0][1])}%"},
    )


def norm_stem(label: str) -> str:
    """Основа подписи части корзины для поиска в названии типа: первые 6 букв без регистра."""
    return label.lower().replace("ё", "е")[:6]


def passports_html(
    types: Sequence[Mapping],
    story: Mapping,
    mo: Mapping[int, Mapping],
    tx: Mapping | None,
    esc: Esc,
    flag_words: Mapping[str, str] | None = None,
) -> str:
    """Паспорта типов: крупное число МО цветом типа, значок и название, «кто обычно», пять строк относительно
    региона (полоса от нуля на общей шкале и словами), пример — кнопка, открывающая карточку и карту; рядом
    с примером — флаг «тип зависит от варианта расчёта» (``flag_words`` — ``story.card.flag_words``);
    у середины оси — сноска ``middle_note``."""
    if not tx or not types:
        return ""
    middle_t = middle_type(types)
    view = story["view"]
    rows_cfg = tx["rows"]
    vals = [abs(v) for ty in types for _, _, v in passport_rows(ty, rows_cfg) if v is not None]
    lim = max([0.1, *vals]) * 1.05
    cards = []
    for ty in types:
        t = int(ty["t"])
        col = ty.get("color") or view["type_colors"].get(str(t), INK2)
        unst = set(ty.get("unstable_parts") or [])
        lis = []
        for i, (f, lab, v) in enumerate(passport_rows(ty, rows_cfg)):
            if v is None:
                continue
            w = abs(v) / lim * 50
            side = f"left:50%;width:{w:.1f}%" if v >= 0 else f"right:50%;width:{w:.1f}%"
            bar = col if i == 0 else GREY
            star = "*" if f in unst else ""
            lis.append(
                f'<li{" class=main" if i == 0 else ""}><span class="pp-lab">{esc(lab)}{star}</span>'
                f'<span class="pp-track"><i style="{side};background:{_e(bar)}"></i></span>'
                f'<span class="pp-val">{esc(rel_text(v))}</span></li>'
            )
        who = ""
        if ty.get("who"):
            w = ty["who"]
            who = _fill(
                tx["who"],
                {"cities": style.fmt_pct(w["cities"], 0), "large": style.fmt_pct(w["large_cities"], 0)},
            )
        n = int(ty.get("size") or 0)
        size = _fill(
            tx["size"],
            {"mo_word": _plural(n), "pop": style.fmt_pct(ty.get("pop_share") or 0, 0)},
        )
        ex = ""
        typical = ((ty.get("examples") or {}).get("typical") or [])[:1]
        for i in typical:
            r = mo.get(int(i))
            if r:
                nm = str(r.get("ns") or r.get("n") or "")
                nm = nm[:1].upper() + nm[1:]
                # 03.10 (совет судей): тип примера зависит от варианта расчёта — флаг usefulness.type_flag
                fl = example_flag(r, flag_words)
                ex = (
                    f'<p class="pp-ex">{esc(tx["example"])}: '
                    f'<button type="button" data-go="{int(i)}" data-map="1">'
                    f"{esc(nm)}</button>, {esc(r.get('r', ''))}"
                    + (f'<span class="pp-flag">{esc(fl[:1].upper() + fl[1:])}</span>' if fl else "")
                    + "</p>"
                )
        mid = middle_note(ty, tx, middle_t)
        shape = _e(view["shapes"].get(str(t), ""))
        cards.append(
            f'<article class="pp" style="--c:{_e(col)}"><p class="pp-num">{_f(n)}</p>'
            f'<p class="pp-size">{esc(size)}</p>'
            f'<h3><i class="fig fig-t{t}" aria-hidden="true">{shape}</i>'
            f"{esc(ty.get('name') or f'Тип {t}')}</h3>"
            + (f'<p class="pp-who">{esc(_dot(who))}</p>' if who else "")
            + f'<ul class="pp-rows">{"".join(lis)}</ul>'
            + (f'<p class="pp-mid">{esc(_dot(mid))}</p>' if mid else "")
            + f"{ex}</article>"
        )
    # порция 6b (check-ux): как собрано название и почему продукты и маркетплейсы у медиан почти равны
    naming = f'<p class="pp-intro pp-naming">{esc(_dot(tx["naming"]))}</p>' if tx.get("naming") else ""
    foot = f'<p class="pp-foot">{esc(_dot(tx["footnote"]))}</p>' if tx.get("footnote") else ""
    return (
        f'<p class="pp-intro">{esc(_dot(tx["intro"]))}</p>{naming}'
        f'<div class="passports">{"".join(cards)}</div>{foot}'
    )


def _plural(n: int) -> str:
    a, b = abs(n) % 100, abs(n) % 10
    if 10 < a < 20:
        return "муниципалитетов"
    if b == 1:
        return "муниципалитет"
    if 1 < b < 5:
        return "муниципалитета"
    return "муниципалитетов"


def passport_strings(types: Sequence[Mapping], tx: Mapping | None) -> list[str]:
    """Строки паспортов (подписи и слова «в 1,5 раза больше») — для линта."""
    if not tx:
        return []
    out = [tx["intro"], tx["example"], tx.get("detail", ""), tx.get("naming", ""), tx.get("footnote", "")]
    out += list(tx["rows"].values())
    mt = middle_type(types)
    out += [middle_note(ty, tx, mt) for ty in types]
    for ty in types:
        out += [rel_text(v) for _, _, v in passport_rows(ty, tx["rows"])]
        n = int(ty.get("size") or 0)
        out.append(
            _fill(tx["size"], {"mo_word": _plural(n), "pop": style.fmt_pct(ty.get("pop_share") or 0, 0)})
        )
        if ty.get("who"):
            w = ty["who"]
            out.append(
                _fill(
                    tx["who"],
                    {"cities": style.fmt_pct(w["cities"], 0), "large": style.fmt_pct(w["large_cities"], 0)},
                )
            )
    return out
