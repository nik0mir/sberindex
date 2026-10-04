"""Порция 6b лендинга (``docs/landing_spec.md``, §4.10): блок пользы и флаг устойчивости типа МО из этапа
``usefulness`` (разведка после вскрытия; правила — блок ``usefulness`` конфига, записан до расчёта).

Этап site ничего не пересчитывает: доля случаев, интервалы, слова доли, медианные ошибки, пример и флаги —
из ``<paths.outputs>/usefulness/`` (``facts.json``, ``mo_flags.csv``). Вход необязательный: нет файлов
(или режим ``--demo`` / ``--dev-blind``) — блока пользы и флага нет, в логе предупреждение. Выходы usefulness
старше ``outputs/interpret/facts.json`` или посчитаны на другом его содержимом — код 1 (устаревший вход).

Тексты — ``site.build.texts.useful`` и ``findings.border.flag`` (линтуются как текст сайта). Знак разности
ошибок у отдельного МО не показывается (``usefulness.rule_share.per_mo: not_shown``).
"""

from __future__ import annotations

import hashlib
import html
import json
import logging
import math
import re
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from munnet import style
from munnet.contracts import MissingInputError, QCError

log = logging.getLogger(__name__)

Esc = Callable[[Any], str]
INK = "#1d1d1d"
INK2 = "#595959"
GREY = "#8d96a1"
RULE = "#d6d3cc"
ACCENT = "#a3172d"
# коды флага для mo.json и scene.json (слова — из mo_flags.csv, flag_text)
FLAG_CODE = {"stable": "s", "depends": "d"}


def _fill(tpl: str, vals: Mapping[str, str]) -> str:
    out = tpl
    for k, v in vals.items():
        out = out.replace("{" + k + "}", str(v))
    return out


def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load(cfg: Any, mode: Any) -> dict | None:
    """Выходы этапа usefulness или None (блок пользы и флаг не показываются).

    Код 1 (``MissingInputError``): ``facts.json`` usefulness старше ``facts.json`` этапа interpret или его
    ``inputs_sha256.facts`` не совпадает с содержимым ``facts.json`` interpret — вход устарел."""
    if mode.name != "normal":
        log.warning(
            "site (%s): блок пользы и флаг устойчивости типа не показываются — нужен обычный режим", mode.name
        )
        return None
    d = cfg.dir("outputs") / "usefulness"
    fp, flags_p = d / "facts.json", d / "mo_flags.csv"
    if not fp.exists() or not flags_p.exists():
        log.warning(
            "site: нет %s или %s — блок пользы и флаг устойчивости типа не показываются (этап usefulness)",
            fp,
            flags_p.name,
        )
        return None
    ip = mode.interpret_dir / "facts.json"
    if ip.exists() and fp.stat().st_mtime < ip.stat().st_mtime:
        raise MissingInputError(f"site: {fp} старше {ip} — перезапустите этап usefulness")
    with open(fp, encoding="utf-8") as f:
        uf = json.load(f)
    if uf.get("status") != "done":
        log.warning("site: %s: status = %r — блок пользы не показывается", fp, uf.get("status"))
        return None
    want = (uf.get("inputs_sha256") or {}).get("facts")
    if ip.exists() and want and want != _sha256(ip):
        raise MissingInputError(
            f"site: {fp} посчитан на другом {ip.name} этапа interpret (sha256) — "
            "перезапустите этап usefulness"
        )
    flags = pd.read_csv(flags_p)
    return {"facts": uf, "flags": flags}


def flag_codes(flags: pd.DataFrame) -> tuple[pd.Series, dict[str, str]]:
    """Код флага по узлу (``s`` — устойчив, ``d`` — зависит, ``p<N>`` — проверен в N вариантах) и слова кода
    из ``flag_text``. У одного кода разные слова — код 3."""
    code = flags["flag"].map(FLAG_CODE)
    part = flags["flag"] == "partial_check"
    code[part] = "p" + flags.loc[part, "variant_runs"].astype(int).astype(str)
    if code.isna().any():
        bad = sorted(set(flags.loc[code.isna(), "flag"].astype(str)))
        raise QCError(f"site: mo_flags.csv: неизвестный флаг {bad}")
    words: dict[str, str] = {}
    for c, t in zip(code, flags["flag_text"].astype(str), strict=True):
        if words.setdefault(c, t) != t:
            raise QCError(f"site: mo_flags.csv: у флага {c} разные слова «{words[c]}» и «{t}»")
    return pd.Series(code.to_numpy(), index=flags["territory_id"].astype(int)), words


def check_flags(flags: pd.DataFrame, node_type: pd.Series, rob: Mapping[str, pd.Series]) -> list[str]:
    """Сверка флагов с тем, что показывает карточка (код 3): тип узла — как в ``types.csv``; счёт вариантов
    и повторов — тот же, что в ``node_seed.csv`` (``rob_rule``, ``rob_seed``); «устойчив» — ровно когда тип
    тот же во всех вариантах и повторах."""
    bad = []
    f = flags.set_index(flags["territory_id"].astype(int))
    t = node_type.reindex(f.index)
    if t.isna().any() or (t.astype(int) != f["type"].astype(int)).any():
        bad.append("mo_flags.csv: тип узла не совпадает с types.csv")
    if set(f.index) != set(node_type.index):
        bad.append(f"mo_flags.csv: {len(f)} узлов против {len(node_type)} узлов с типом")
    for col, same, runs in (
        ("rob_rule", "variant_same", "variant_runs"),
        ("rob_seed", "seed_same", "seed_runs"),
    ):
        s = rob.get(col)
        if s is None:
            continue
        want = f[same].astype(int).astype(str) + "/" + f[runs].astype(int).astype(str)
        got = s.reindex(f.index)
        n = int((got != want).sum())
        if n:
            bad.append(f"mo_flags.csv: счёт {col} не совпал с node_seed.csv у {n} узлов")
    full = (f["variant_same"] == f["variant_runs"]) & (f["seed_same"] == f["seed_runs"])
    stable = f["flag"] == "stable"
    if (stable & ~full).any():
        bad.append("mo_flags.csv: «тип устойчив» у узла с другим типом в каком-то прогоне")
    return bad


# --- тексты ---------------------------------------------------------------------------------------------


def _mo_word(n: int, gen: bool = False) -> str:
    a, b = abs(n) % 100, abs(n) % 10
    if gen:  # «из 1776 муниципалитетов», «из 21 муниципалитета»
        return "муниципалитета" if (b == 1 and a != 11) else "муниципалитетов"
    if 10 < a < 20 or b == 0 or b > 4:
        return "муниципалитетов"
    return "муниципалитет" if b == 1 else "муниципалитета"


def mo_count(n: int, gen: bool = False) -> str:
    """«1542 муниципалитета», «1776 муниципалитетов»; ``gen`` — после «из»."""
    return f"{style.fmt_num(n)} {_mo_word(n, gen)}"


def words_key(words: str, cfg_words: Mapping[str, str]) -> str:
    """Ключ слов доли (``more_often`` / ``about_half`` / ``less_often``) по словам из facts.json."""
    for k, w in cfg_words.items():
        if w == words:
            return k
    raise QCError(f"site: слова доли «{words}» не из usefulness.rule_share.words")


def use_texts(tx: Mapping, uf: Mapping, cfg_words: Mapping[str, str]) -> dict[str, Any]:
    """Строки колонки «Что с этим делать» из facts.json usefulness: доля случаев (B против D) с интервалом
    и ориентиром 50%, оговорка о наборе R по медианным ошибкам, правка по словам доли (``edits``)."""
    r = uf["rule"]
    vd, vr = r["vs_D"], r["vs_R"]
    lo, hi = vd["share_ci"]
    med = r["median_error_abs"]
    share = _fill(
        tx["share"],
        {
            "words": vd["words"],
            "share": style.fmt_pct(vd["share"], 1),
            "n": style.fmt_num(r["n"]),
            "lo": style.fmt_num(100 * lo, 1),
            "hi": style.fmt_pct(hi, 1),
            "reference": r["reference"],
        },
    )
    rk = "r_not_worse" if float(med["R"]) <= float(med["B"]) else "r_worse"
    r_line = _fill(tx[rk], {"err_r": style.fmt_num(med["R"], 3), "err_b": style.fmt_num(med["B"], 3)})
    r_key = words_key(vr["words"], cfg_words)
    # порция 6g: заранее записанная правка edits.vs_R_not_more_often сработала — одна фраза, откуда заголовок
    # «со своим регионом» (доля B против R, интервал и ничьи — из facts.json)
    r_edit = ""
    if r_key != "more_often" and tx.get("r_edit"):
        rlo, rhi = vr["share_ci"]
        ties = int(vr.get("ties") or 0)
        r_edit = _fill(
            tx["r_edit"],
            {
                "words": vr["words"],
                "share": style.fmt_pct(vr["share"], 1),
                "lo": style.fmt_num(100 * rlo, 1),
                "hi": style.fmt_pct(rhi, 1),
                # ничьи (наборы совпали) — отдельной вставкой; нет ничьих — вставки нет
                "ties": _fill(tx.get("r_edit_ties", ""), {"n": style.fmt_num(ties)}) if ties else "",
            },
        )
    return {
        "share": share,
        "r_line": r_line,
        "r_edit": r_edit,
        "rules": tx["rules"],
        "d_key": words_key(vd["words"], cfg_words),
        "r_key": r_key,
    }


# хвост названия по виду МО (mo.json, поле «k»): «Благовещенский» → «Благовещенский район», как в отчёте
KIND_SUFFIX = {"муниципальный район": " район", "муниципальный округ": " муниципальный округ"}


def display_name(r: Mapping, fallback: str = "") -> str:
    """Короткое название МО с заглавной буквы и хвостом вида МО для районов и муниципальных округов."""
    name = str(r.get("ns") or r.get("n") or fallback or "")
    name = name[:1].upper() + name[1:]
    suffix = KIND_SUFFIX.get(str(r.get("k") or ""), "")
    return name if not suffix or name.endswith(suffix.strip()) else name + suffix


def example_texts(
    tx: Mapping, uf: Mapping, mo: Mapping[int, Mapping], k: int = 10, t7_example: Mapping | None = None
) -> dict[str, Any] | None:
    """Пример по правилу (``example``) и обратный пример этапа 5 (``example.stage5_example``). Порция 6e:
    ``t7_example`` — пример главы 6 (``checks.t7.example``: ошибки наборов из ``t7_errors.csv``); если это
    тот же муниципалитет, поле ``{err_d_rel}`` обратного примера — ошибка набора D с поправкой на регион,
    как в таблице главы 6; иначе скобка с этим полем не показывается."""
    ex = uf.get("example") or {}
    if not ex or ex.get("own_change") is None:
        return None
    tid = int(ex["territory_id"])
    r = mo.get(tid) or {}
    name = display_name(r, str(ex.get("name") or ""))
    region = str(r.get("r") or ex.get("region") or "")
    kb, kd = len(ex.get("members_B") or []) or k, len(ex.get("members_D") or []) or k
    pct = lambda v: style.fmt_pct(v, 1, sign=True)  # noqa: E731
    vals = {
        "name": name,
        "region": region,
        "own": pct(ex["own_change"]),
        "b": pct(ex["median_B_change"]),
        "d": pct(ex["median_D_change"]),
        "k_b": style.fmt_num(kb),
        "k_d": style.fmt_num(kd),
        "err_b": style.fmt_num(ex["err_B"], 3),
        "err_d": style.fmt_num(ex["err_D"], 3),
    }
    out: dict[str, Any] = {
        "id": tid,
        "name": name,
        "label": tx["ex_label"],
        "title": _fill(tx["ex_title"], vals),
        "text": _fill(tx["ex_text"], vals),
        "rule": tx["ex_rule"],
        "open": tx["ex_open"],
        "aria": _fill(tx["ex_aria"], vals),
        "axis": tx["ex_axis"],
        "rows": [
            (_fill(tx["ex_own"], vals), float(ex["own_change"]), vals["own"], "own"),
            (_fill(tx["ex_b"], {"k": vals["k_b"]}), float(ex["median_B_change"]), vals["b"], "b"),
            (_fill(tx["ex_d"], {"k": vals["k_d"]}), float(ex["median_D_change"]), vals["d"], "d"),
        ],
    }
    s5 = ex.get("stage5_example") or {}
    # обратный случай — только если у примера этапа 5 соседи по региону и правда ошибаются больше
    if s5 and s5.get("err_B") is not None and float(s5["err_B"]) > float(s5["err_D"]):
        r5 = mo.get(int(s5["territory_id"])) or {}
        rel = None
        if t7_example and int(t7_example.get("territory_id") or -1) == int(s5["territory_id"]):
            rel = (t7_example.get("errors") or {}).get("D")
        tpl = str(tx["ex_reverse"])
        if rel is None:  # нет числа главы 6 — без скобки с ним
            tpl = re.sub(r"\s*\([^()]*\{err_d_rel\}[^()]*\)", "", tpl)
        out["reverse"] = _fill(
            tpl,
            {
                "err_d_rel": style.fmt_num(rel, 3) if rel is not None else "",
                "name": display_name(r5, str(s5.get("name") or "")),
                "region": str(r5.get("r") or s5.get("region") or ""),
                "err_b": style.fmt_num(s5["err_B"], 3),
                "err_d": style.fmt_num(s5["err_D"], 3),
                "words": uf["rule"]["vs_D"]["words"],
            },
        )
        out["reverse_id"] = int(s5["territory_id"])
    return out


def flag_summary(tpl: str, most: Mapping[str, str], uf: Mapping) -> str | None:
    """Строка «тип устойчив у N из M…»: числа — ``type_flag``, «чаще всего» — вариант с наименьшей долей
    «тот же тип» (``type_flag.per_variant``)."""
    tf = uf.get("type_flag") or {}
    pv = [v for v in tf.get("per_variant") or [] if v.get("same_share") is not None]
    if not tf.get("n") or not pv:
        return None
    low = min(pv, key=lambda v: v["same_share"])
    if low["variant"] not in most:
        return None
    n = int(tf["n"])
    return _fill(
        tpl,
        {
            "stable": style.fmt_num(tf["stable"]),
            "n_mo": mo_count(n, gen=True),
            "pct": style.fmt_pct(tf["stable"] / n, 1),
            "most": most[low["variant"]],
        },
    )


def strings(ut: Mapping | None, ex: Mapping | None, flag: str | None) -> list[str]:
    """Все показываемые строки 6b — для линта (``landing.lint_texts``)."""
    out = [flag or ""]
    if ut:
        out += [ut["share"], ut["r_line"], ut["rules"], ut.get("r_edit") or ""]
    if ex:
        out += [ex["label"], ex["title"], ex["text"], ex["rule"], ex["open"], ex["aria"], ex["axis"]]
        out += [r[0] for r in ex["rows"]] + [ex.get("reverse") or "", ex.get("size_note") or ""]
    return [s for s in out if s]


# --- график примера ---------------------------------------------------------------------------------------


def _e(s: Any) -> str:
    return html.escape(str(s if s is not None else ""), quote=True)


def _text(x: float, y: float, s: str, cls: str = "", anchor: str = "start") -> str:
    c = f' class="{cls}"' if cls else ""
    a = f' text-anchor="{anchor}"' if anchor != "start" else ""
    return f'<text x="{x:.1f}" y="{y:.1f}"{c}{a}>{_e(_nb(s))}</text>'


def _nb(s: str) -> str:
    """Типографика ru-text в подписях SVG (порция 6j): неразрывный пробел после однобуквенных слов и перед
    тире — та же функция, что для HTML (``landing.nbsp``)."""
    from munnet.landing import nbsp

    return nbsp(str(s)) if s else s


def example_svg(ex: Mapping, W: float = 440) -> str:
    """Три точки на одной шкале в процентах: сам муниципалитет (акцент, вертикаль через все строки)
    и медианы двух наборов; отрезок от точки набора до вертикали — ошибка сверки. Подписи — прямо у точек:
    название строки над точкой, значение — справа столбцом. Шкала — от минимума до максимума с запасом
    (точечный график, не полосы)."""
    rows = ex["rows"]
    vals = [v for _, v, _, _ in rows]
    lo, hi = min(vals), max(vals)
    span = max(hi - lo, 0.02)
    lo, hi = lo - span * 0.35, hi + span * 0.35
    row, top = 34, 4
    x0, x1 = 6, W - 64
    sx = lambda v: x0 + (v - lo) / (hi - lo) * (x1 - x0)  # noqa: E731
    H = top + row * len(rows) + 36
    own = next(v for _, v, _, k in rows if k == "own")
    g = [
        f'<line x1="{sx(own):.1f}" x2="{sx(own):.1f}" y1="{top + 14}" y2="{top + row * len(rows)}" '
        f'stroke="{ACCENT}" stroke-width="1" stroke-dasharray="3 3"/>'
    ]
    for j, (lab, v, s, kind) in enumerate(rows):
        y = top + j * row + 22
        g.append(_text(x0, y - 10, lab, "lab th" if kind == "own" else "lab"))
        if kind != "own":
            g.append(
                f'<line x1="{sx(v):.1f}" x2="{sx(own):.1f}" y1="{y + 2:.1f}" y2="{y + 2:.1f}" '
                f'stroke="{GREY}" stroke-width="2"/>'
            )
        col = ACCENT if kind == "own" else INK if kind == "b" else GREY
        g.append(
            f'<circle cx="{sx(v):.1f}" cy="{y + 2:.1f}" r="5" fill="{col}" stroke="#fff" stroke-width="1"/>'
        )
        g.append(_text(W, y + 6, s, "val", "end"))
    # ось: три-четыре круглых значения в процентах
    step = 0.01 if (hi - lo) < 0.06 else 0.02 if (hi - lo) < 0.12 else 0.05
    t = math.ceil(lo / step) * step
    ya = top + row * len(rows) + 4
    g.append(f'<line x1="{x0}" x2="{x1}" y1="{ya}" y2="{ya}" stroke="{RULE}"/>')
    while t <= hi + 1e-12:
        x = sx(t)
        g.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{ya}" y2="{ya + 4}" stroke="{INK2}"/>')
        g.append(_text(x, ya + 15, style.fmt_pct(t, 0), "axis", "middle"))
        t += step
    # подпись оси — от левого края; не помещается в строку (телефон) — две строки по словам
    cap, words, lines = "", str(ex["axis"]).split(), []
    for w in words:
        if cap and len(cap) + 1 + len(w) > int((W - x0) / (11.5 * 0.56)):
            lines.append(cap)
            cap = w
        else:
            cap = f"{cap} {w}".strip()
    lines.append(cap)
    for i, ln in enumerate(lines):
        g.append(_text(x0, ya + 31 + 13 * i, ln, "axis"))
    H += 13 * (len(lines) - 1)
    return (
        f'<svg class="fd-svg ex-svg" viewBox="0 0 {W:g} {H + 6:g}" role="img" aria-label="{_e(ex["aria"])}" '
        f'xmlns="http://www.w3.org/2000/svg">{"".join(g)}</svg>'
    )


def example_html(
    ex: Mapping | None, esc: Esc, pair: Callable[[str, str], str], phone_w: float, dom_id: str = "use-example"
) -> str:
    """Полоса «Пример по правилу» под тремя колонками блока: текст и кнопка слева, график справа."""
    if not ex:
        return ""
    fig = pair(example_svg(ex), example_svg(ex, phone_w))
    rev = f'<p class="fd-rev">{esc(_dot(ex["reverse"]))}</p>' if ex.get("reverse") else ""
    return (
        f'<div class="fd-ex" id="{dom_id}">'
        f'<div class="fd-ex-text"><p class="fd-lab">{esc(ex["label"])}</p><h3>{esc(ex["title"])}</h3>'
        f"<p>{esc(_dot(ex['text']))}</p>"
        f'<p class="fd-note">{esc(_dot(ex["rule"]))}</p>'
        # порция 6l: пример — крупное МО, совет над ним — небольшим и средним (site_size.example_note)
        + (f'<p class="fd-note fd-exsize">{esc(_dot(ex["size_note"]))}</p>' if ex.get("size_note") else "")
        + f'<p><button type="button" class="open-card" data-go="{int(ex["id"])}" data-map="1">'
        f"{esc(ex['open'])}</button></p></div>"
        f"<figure>{fig}</figure>{rev}</div>"
    )


def _dot(s: str) -> str:
    s = s.rstrip()
    return s if not s or s[-1] in ".!?…" else s + "."


# --- порция 6m, второй круг: пример совета небольшим и средним МО (usefulness.example_small) ---

EXAMPLE_SMALL_TARGETS = ("shipments", "ndfl", "catering")  # порядок строк таблицы примера


def load_example_small(cfg: Any, mode: Any) -> dict | None:
    """``outputs/usefulness/example_small.json`` (правило ``usefulness.example_small`` записано до выбора,
    28a2079) или None: нет файла, не обычный режим, ``status`` не ``done`` — примера нет, остаётся прежний.
    Файл старше ``facts.json`` этапа interpret — код 1; сверка ``qc`` не сошлась — код 3."""
    if mode.name != "normal":
        return None
    p = cfg.dir("outputs") / "usefulness" / "example_small.json"
    if not p.exists():
        log.warning("site: нет %s — пример совета небольшим и средним МО не показывается", p)
        return None
    ip = mode.interpret_dir / "facts.json"
    if ip.exists() and p.stat().st_mtime < ip.stat().st_mtime:
        raise MissingInputError(f"site: {p} старше {ip} — перезапустите этап usefulness")
    with open(p, encoding="utf-8") as f:
        ex = json.load(f)
    if ex.get("status") != "done":
        log.warning("site: %s: status = %r — пример не показывается", p, ex.get("status"))
        return None
    qc = ex.get("qc") or {}
    if float(qc.get("abs_diff", 1.0)) > 1e-12:
        raise QCError(f"site: {p}: доля «соседи ближе» у МО пула не совпала с size_check.json (qc.abs_diff)")
    return ex


def example_small_texts(
    tx: Mapping, ex: Mapping | None, mo: Mapping[int, Mapping], words: Mapping, flag_words: Mapping[str, str]
) -> dict[str, Any] | None:
    """Тексты примера: название с регионом, население и тип с флагом устойчивости, таблица по трём показателям
    совета (своё изменение, медианы соседей по региону и похожих по тратам МО других регионов, кто ближе —
    поле ``closer`` выхода), списки ориентиров с регионами, правило выбора и «вывода об отдельном МО нет»
    (``usefulness.example_small.words``). Нет примера или текстов — None."""
    if not ex or not tx:
        return None
    pt = ex.get("per_target") or {}
    if any(t not in pt for t in EXAMPLE_SMALL_TARGETS):
        return None
    tid = int(ex["territory_id"])
    r = mo.get(tid) or {}
    name = str(r.get("n") or ex.get("name") or "")
    region = str(r.get("r") or ex.get("region") or "")
    pct = lambda v: style.fmt_pct(float(v), 0, sign=True)  # noqa: E731
    flag = str(ex.get("flag_text") or "")
    flag = str(flag_words.get("s") or flag) if flag == "тип устойчив" else flag
    sub = _fill(
        tx["sub"],
        {
            "pop": f"{style.fmt_num(float(ex['pop_avg_2023']) / 1000.0, 1)} тыс.",
            "type": str(ex.get("type_name") or ""),
        },
    ) + (f" ({flag})" if flag else "")
    rows = []
    far = float(tx.get("far_err") or 0)
    for t in EXAMPLE_SMALL_TARGETS:
        d = pt[t]
        # порция 6m, п. 14: медианы ориентиров совпали при целых процентах — все три числа строки с десятыми
        dec = 1 if pct(d["median_B_change"]) == pct(d["median_D_change"]) else 0
        fmt = lambda v, n=dec: style.fmt_pct(float(v), n, sign=True)  # noqa: E731
        c = str(d.get("closer"))
        both_far = far > 0 and min(float(d.get("err_B", 0)), float(d.get("err_D", 0))) > far
        closer = (tx.get("closer_far") or {}).get(c) if both_far else None
        rows.append(
            [
                str(tx["targets"][t]),
                fmt(d["own_change"]),
                fmt(d["median_B_change"]),
                fmt(d["median_D_change"]),
                str(closer or tx["closer"].get(c, "")),
            ]
        )

    def names_b() -> str:
        full = [(mo.get(int(m["territory_id"])) or {}).get("n") or m["name"] for m in ex["members_B"]]
        return ", ".join(str(x) for x in full)

    def names_d() -> str:
        return ", ".join(f"{m['name']} ({m['region']})" for m in ex["members_D"])

    pool = ex.get("pool") or {}
    n_pool = int(pool.get("n_pool") or 0)
    gen = "муниципалитета" if n_pool % 10 == 1 and n_pool % 100 != 11 else "муниципалитетов"  # «среди 481 …»
    rule = _fill(
        str(tx.get("rule") or words.get("rule") or ""),
        {
            "n_pool": style.fmt_num(n_pool),
            "mo_gen": gen,
            "bound": style.fmt_num(float(pool.get("bound") or 0) / 1000.0, 1),
        },
    )
    rt = ex.get("retail") or {}
    retail = ""
    if tx.get("retail") and rt.get("own_change") is not None:  # розница в выбор не входила — строкой
        retail = _fill(
            tx["retail"],
            {
                "own": pct(rt["own_change"]),
                "b": pct(rt["median_B_change"]),
                "d": pct(rt["median_D_change"]),
                "closer": str(tx["closer"].get(str(rt.get("closer")), "")),
            },
        )
    return {
        "id": tid,
        "label": tx["label"],
        "title": f"{name}, {region}",
        "sub": sub,
        "intro": tx["intro"],
        "head": list(tx["head"]),
        "rows": rows,
        "members_b": _fill(tx["members_b"], {"region": region, "names": names_b()}),
        "members_d": _fill(tx["members_d"], {"names": names_d()}),
        "rule": rule,
        "retail": retail,
        "no_claim": str(tx.get("no_claim") or words.get("no_claim") or ""),
        "open": tx["open"],
        "caption": str(tx.get("caption") or ""),
    }


def example_small_strings(es: Mapping | None) -> list[str]:
    """Строки примера — для линта."""
    if not es:
        return []
    out = [es["label"], es["title"], es["sub"], es["intro"], es["members_b"], es["members_d"], es["open"]]
    out += [es.get("retail") or "", es.get("rule") or "", es.get("no_claim") or "", es.get("caption") or ""]
    out += list(es["head"]) + [c for r in es["rows"] for c in r]
    return [s for s in out if s]


def example_small_html(es: Mapping | None, esc: Esc) -> str:
    """Полоса «Пример по правилу» под тремя колонками: таблица по трём показателям совета (текстом — графику
    с разными масштабами изменений таблица понятнее), списки ориентиров, правило выбора мелко."""
    if not es:
        return ""
    th = "".join(f'<th scope="col">{esc(h)}</th>' for h in es["head"])
    trs = "".join(
        f'<tr><th scope="row">{esc(r[0])}</th>' + "".join(f"<td>{esc(c)}</td>" for c in r[1:]) + "</tr>"
        for r in es["rows"]
    )
    return (
        '<div class="fd-ex fd-ex-small" id="use-example">'
        f'<div class="fd-ex-text"><p class="fd-lab">{esc(es["label"])}</p><h3>{esc(es["title"])}</h3>'
        f'<p class="fd-ex-sub">{esc(_dot(es["sub"]))}</p><p>{esc(_dot(es["intro"]))}</p>'
        f'<div class="tbl-wrap"><table class="ex-tbl"><thead><tr>{th}</tr></thead>'
        f"<tbody>{trs}</tbody></table></div>"
        + (f'<p class="fd-cap">{esc(_dot(es["caption"]))}</p>' if es.get("caption") else "")
        + (f'<p class="fd-ex-retail">{esc(_dot(es["retail"]))}</p>' if es.get("retail") else "")
        + f'<p class="fd-note">{esc(_dot(es["members_b"]))}</p>'
        f'<p class="fd-note">{esc(_dot(es["members_d"]))}</p>'
        f'<p class="fd-note">{esc(_dot(es["rule"]))} {esc(_dot(es["no_claim"]))}</p>'
        f'<p><button type="button" class="open-card" data-go="{int(es["id"])}" data-map="1">'
        f"{esc(es['open'])}</button></p></div></div>"
    )
