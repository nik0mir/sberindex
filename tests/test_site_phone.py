"""Порция 6a лендинга (``docs/landing_spec.md``, §4.9): графики для телефона, названия типов в графиках,
плоская карта первого экрана. Подписи SVG на экране 375 px (колонка 343 px) — не мельче 12 px: у графика
либо вариант для телефона (``vp``; широкий ``vw`` на телефоне скрыт), либо достаточно крупный кегль.
Кегль берётся из ``templates/landing.css`` (правила ``.ch-svg …``, ``.fd-svg …``, ``.mini-map …``,
на телефоне — с медиазапросами до 700 px), ширина холста — из ``viewBox``. Без сети и без data/."""

from __future__ import annotations

import re
from pathlib import Path

import test_site_chapters as TC
import test_site_chapters_tail as TT
import test_site_findings as TF

from munnet import landing, site_findings
from munnet import site_chapters as SC

CSS = (Path(landing.__file__).parent / "templates" / "landing.css").read_text(encoding="utf-8")
JS3D = (Path(landing.__file__).parent / "templates" / "landing3d.js").read_text(encoding="utf-8")
HTML_T = (Path(landing.__file__).parent / "templates" / "landing.html").read_text(encoding="utf-8")
COL = 343.0  # колонка текста на экране 375 px (поля 16 px)
SVG_ROOTS = ("ch-svg", "fd-svg", "mini-map")


def _rules(css: str, phone: bool) -> list[tuple[str, float]]:
    """Пары (селектор, кегль px) в порядке файла; на телефоне — и правила медиазапросов max-width ≤ 700 px."""
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    out: list[tuple[str, float]] = []
    pos = 0
    for m in re.finditer(r"@media([^{]+)\{((?:[^{}]*\{[^{}]*\})*)\s*\}", css):
        out += _plain(css[pos : m.start()])
        q = m.group(1)
        mw = re.search(r"max-width:\s*(\d+)px", q)
        if phone and mw and int(mw.group(1)) >= 375 and "min-width" not in q:
            out += _plain(m.group(2))
        pos = m.end()
    return out + _plain(css[pos:])


def _plain(block: str) -> list[tuple[str, float]]:
    out = []
    for sel, body in re.findall(r"([^{}]+)\{([^{}]*)\}", block):
        fs = re.search(r"(?:^|;)\s*font-size:\s*([\d.]+)px", body)
        if fs:
            out += [(s.strip(), float(fs.group(1))) for s in sel.split(",")]
    return out


def _font(svg_cls: set[str], text_cls: set[str], rules: list[tuple[str, float]]) -> float:
    """Кегль подписи по правилам вида «.root text» и «.root .cls» (последнее подходящее — как в каскаде;
    правило с классом подписи сильнее «text»)."""
    size, best = 0.0, -1
    for sel, px in rules:
        m = re.fullmatch(r"\.([\w-]+)((?:\.[\w-]+)*) (text|\.[\w-]+)", sel)
        if not m or m.group(1) not in svg_cls:
            continue
        extra = set(filter(None, m.group(2).split(".")))
        if not extra <= svg_cls:
            continue
        spec = 1 if m.group(3) == "text" else 2
        if m.group(3) != "text" and m.group(3)[1:] not in text_cls:
            continue
        if spec >= best:
            size, best = px, spec
    return size


def phone_report(html: str) -> list[tuple[str, float]]:
    """(класс графика, наименьшая подпись в px на 343 px) для каждой SVG, видимой на телефоне."""
    rules = _rules(CSS, phone=True)
    out = []
    for attrs, body in re.findall(r"<svg([^>]*)>(.*?)</svg>", html, re.S):
        cm = re.search(r'class="([^"]*)"', attrs)
        cls = set(cm.group(1).split()) if cm else set()
        if not cls & set(SVG_ROOTS) or "vw" in cls or "sm-wide" in cls:
            continue
        vb = re.search(r'viewBox="[\d.\-]+ [\d.\-]+ ([\d.]+) [\d.]+"', attrs)
        if not vb:
            continue
        sizes = []
        for ta in re.findall(r"<text([^>]*)>", body):
            tm = re.search(r'class="([^"]*)"', ta)
            sizes.append(_font(cls, set(tm.group(1).split()) if tm else set(), rules))
        if sizes:
            out.append((cm.group(1), min(sizes) * COL / float(vb.group(1))))
    return out


def _all_charts() -> str:
    st = TC.story()
    parts = [TC.chapters(st)]
    parts.append(TT.page())
    fstory = TF._story()
    ft = site_findings.findings_texts(TF.TX["findings"], fstory, TF._facts(), TF._checks())
    parts.append(site_findings.findings_html(ft, fstory, [2, 1, 3, 4], TF.ESC, "Источник"))
    return "\n".join(parts)


def test_every_chart_readable_at_343px():
    """У каждого графика глав и блока «Что устояло» на телефоне подписи ≥ 12 px (вариант vp или кегль)."""
    rep = phone_report(_all_charts())
    kinds = {c.replace("vp ", "") for c, _ in rep}
    for want in ("fd-svg", "ch-svg basket", "ch-svg ami", "ch-svg order-a", "ch-svg order-b",
                 "ch-svg alluvial", "ch-svg placebo", "ch-svg comp dots", "ch-svg diff dots",
                 "ch-svg ex-err dots", "ch-svg sm-panel", "ch-svg mini-map ex-map"):  # fmt: skip
        assert want in kinds, want  # проверка не пустая: все виды графиков найдены
    small = [(c, round(px, 1)) for c, px in rep if px < SC.PHONE_MIN_PX]
    assert not small, small


def test_phone_check_catches_small_text():
    """Отрицательный контроль: широкий вариант (без vp) на телефоне был бы мельче 12 px."""
    svg = SC.basket_svg([0.1, -0.2, 0.0, 0.3, 0.1, -0.1], ["а"] * 6, set(), "x")
    assert phone_report(svg) and phone_report(svg)[0][1] < SC.PHONE_MIN_PX


def test_phone_variants_paired_and_hidden_by_css():
    """Каждому широкому варианту (vw) — пара для телефона (vp); CSS прячет один из них по ширине экрана."""
    h = _all_charts()
    assert h.count('<svg class="vw ') == h.count('<svg class="vp ') > 0
    assert re.search(r"svg\.vp\s*\{\s*display:\s*none", CSS)
    assert re.search(r"@media \(max-width: 700px\) \{\s*svg\.vw \{ display: none; \}", CSS)
    ids = re.findall(r'<pattern id="([^"]+)"', h)
    assert len(ids) == len(set(ids))  # штриховки широкого варианта и телефонного — разные id


def test_type_names_in_order_and_dynamics():
    """Главы 4 и 5: строки, блоки и кнопки потоков подписаны названием типа с фигурой, а не «Тип N»."""
    st = TC.story()
    h = TC.chapters(st)
    for cid in ("order", "dynamics"):
        sec = TC.section(h, cid)
        # «Тип N» без названия (в синтетике названия — «Тип N: пример»)
        assert not re.search(r"Тип \d\b(?!: пример)", sec), cid
        assert "Тип 1: пример" in sec
    dyn = TC.section(h, "dynamics")
    btn = re.search(r'<button type="button" class="flow-btn".*?</button>', dyn, re.S).group(0)
    assert 'class="fig fig-t' in btn and "пример" in btn


def test_basket_unstable_note_on_own_line():
    """«Знак у типа неустойчив» — своей строкой под полосой, не на строке подписи части."""
    svg = SC.basket_svg([0.1, -0.2, 0.0, 0.3, 0.1, -0.1], [f"часть {i}" for i in range(6)], {3}, "x")
    ys_lab = {float(y) for y in re.findall(r'<text x="[\d.]+" y="([\d.]+)" class="lab"', svg)}
    y_note = float(re.search(r'<text x="[\d.]+" y="([\d.]+)" class="note"', svg).group(1))
    assert all(abs(y_note - y) >= 11 for y in ys_lab)


def test_flat_toggle_and_canvas_label():
    """Переключатель «Объёмная / Плоская»; aria-label холста не обещает карту «ниже», числа согласованы."""
    assert 'id="h0-flat"' in HTML_T and "data-flat=" in HTML_T and "data-solid=" in HTML_T
    assert "h0-flat" in JS3D and "setFlat" in JS3D
    hx = TF.CFG["site"]["build"]["texts"]["hero"]
    assert "ниже" not in hx["canvas_label"] and "{n_untyped}" in hx["canvas_label"]
    assert "{n_cells}" not in hx["canvas_label"]
    for k in ("to_flat", "to_3d", "flat_note", "picked_note"):
        assert hx.get(k), k
