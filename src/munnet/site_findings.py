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
    return f'<text x="{x:.1f}" y="{y:.1f}"{c}{a}>{_e(s)}</text>'


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
    tx: Mapping, story: Mapping, facts: Mapping, checks: Mapping, useful: Mapping | None = None
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
    if t1 is None or t7 is None or not shares:
        return None
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
        use |= {"share": ut["share"], "r_line": ut["r_line"], "rules": ut["rules"]}
    border_flag = (useful or {}).get("flag")
    return {
        "title": tx["title"],
        "intro": tx.get("intro", ""),
        "stood": {
            **{k: tx["stood"][k] for k in ("label", "title", "chart", "chart_note")},
            "text": _fill(
                tx["stood"]["text"], {k: t1[k] for k in ("rho_a", "rho_b", "rival_rho", "rival", "n_a")}
            ),
        },
        "border": {
            **{k: tx["border"][k] for k in ("label", "title", "chart")},
            "text": _fill(tx["border"]["text"], {"variants": variants, "t3_head": t3_head})
            # порция 6e (check-ux): что такое плацебо — одной фразой после заголовка T3, если он о плацебо
            + (
                " " + _dot(str(tx["border"]["placebo"]))
                if tx["border"].get("placebo") and "плацебо" in t3_head.lower()
                else ""
            ),
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
    }


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
    H = top + row * len(rows_top) + 22
    g = []
    for m, s in ((0.5, "×0,5"), (1.0, "×1"), (2.0, "×2")):
        x = sx(math.log(m))
        stroke = f'stroke="{INK2}" stroke-dasharray="2 2"' if m == 1 else f'stroke="{RULE}"'
        g.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{top}" y2="{top + row * len(rows_top)}" {stroke}/>')
        g.append(_text(x, top + row * len(rows_top) + 15, s, "axis", "middle"))
    pts = []
    for j, t in enumerate(rows_top):
        y = top + j * row + row / 2
        g.append(_text(0, y + 4 - dy, str(names.get(str(t)) or f"Тип {t}"), "lab"))
        if med.get(t) is not None:
            pts.append((sx(med[t]), y + dy, t))
    line = (view.get("line_by_turnover") or {}).get("retail") or t1.get("line", "none")
    if line in ("solid", "dashed") and layout != "columns_by_size" and len(pts) > 1:
        d = "M" + "L".join(f"{x:.1f},{y:.1f}" for x, y, _ in pts)
        dash = ' stroke-dasharray="4 3"' if line == "dashed" else ""
        g.append(f'<path d="{d}" fill="none" stroke="{INK2}" stroke-width="1.4"{dash}/>')
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


def findings_html(ft: Mapping | None, story: Mapping, order: Sequence[int], esc: Esc, src: str) -> str:
    """Раздел «Что устояло и чем полезно» (три колонки; на телефоне — одна). Нет текстов — пусто."""
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
        if key == "use" and p.get("share"):  # порция 6b: доля случаев и оговорка о своём регионе
            body += f'<p class="fd-share">{esc(_dot(p["share"]))}</p><p>{esc(_dot(p["r_line"]))}</p>'
        cols.append(
            f'<div class="fd fd-{key}" id="fd-{key}"><p class="fd-lab">{esc(p["label"])}</p>'
            f"<h3>{esc(p['title'])}</h3>"
            f"{body}<figure>{fig}{extra}</figure></div>"
        )
    from munnet import site_useful
    from munnet.site_chapters import phone_pair

    example = site_useful.example_html(ft.get("_example"), esc, phone_pair, PHONE_W)
    return (
        '<section class="findings5" id="findings" aria-labelledby="findings-title">'
        f'<h2 id="findings-title">{esc(ft["title"])}</h2>'
        + (f'<p class="fd-intro">{esc(_dot(ft["intro"]))}</p>' if ft.get("intro") else "")
        + f'<div class="fd-grid">{"".join(cols)}</div>{example}<p class="source">{esc(src)}</p></section>'
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
            "arcs_net")  # fmt: skip
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


def passports_html(
    types: Sequence[Mapping], story: Mapping, mo: Mapping[int, Mapping], tx: Mapping | None, esc: Esc
) -> str:
    """Паспорта типов: крупное число МО цветом типа, значок и название, «кто обычно», пять строк относительно
    региона (полоса от нуля на общей шкале и словами), пример — кнопка, открывающая карточку и карту."""
    if not tx or not types:
        return ""
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
                ex = (
                    f'<p class="pp-ex">{esc(tx["example"])}: '
                    f'<button type="button" data-go="{int(i)}" data-map="1">'
                    f"{esc(nm)}</button>, {esc(r.get('r', ''))}</p>"
                )
        shape = _e(view["shapes"].get(str(t), ""))
        cards.append(
            f'<article class="pp" style="--c:{_e(col)}"><p class="pp-num">{_f(n)}</p>'
            f'<p class="pp-size">{esc(size)}</p>'
            f'<h3><i class="fig fig-t{t}" aria-hidden="true">{shape}</i>'
            f"{esc(ty.get('name') or f'Тип {t}')}</h3>"
            + (f'<p class="pp-who">{esc(_dot(who))}</p>' if who else "")
            + f'<ul class="pp-rows">{"".join(lis)}</ul>{ex}</article>'
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
