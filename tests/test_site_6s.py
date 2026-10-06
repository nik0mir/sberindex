"""Порция 6s лендинга (``docs/landing_spec.md``, §4.3, журнал 07.10): первый экран на окне 901–1299 px —
объёмная карта подбирается по проекции между колонкой текста и правым краем холста, при «Разложить по типам»
колонка текста скрыта и острова во всю ширину; на 901–1240 px нет лишних сдвигов подсказки и легенды (кнопок
вида две, они в одном ряду); строка над h1 не заходит под кнопки; на 901–1199 px в легенде только типы.
Проверяются шаблоны, без data/ и outputs/ и без чисел раскладки."""

from __future__ import annotations

import re
from pathlib import Path

TPL = Path("src/munnet/templates")
JS = (TPL / "landing3d.js").read_text(encoding="utf-8")
CSS = (TPL / "landing.css").read_text(encoding="utf-8")
BAND = "(min-width: 901px) and (max-width: 1299px)"


def _media_blocks(css: str, query: str) -> list[str]:
    """Тела всех блоков ``@media <query> { … }`` (вложенные скобки правил учитываются)."""
    out = []
    for m in re.finditer(re.escape(f"@media {query}") + r"\s*\{", css):
        depth, i = 1, m.end()
        while depth:
            depth += {"{": 1, "}": -1}.get(css[i], 0)
            i += 1
        out.append(css[m.end() : i - 1])
    return out


def test_mid_band_view_fitted_by_projection_in_js():
    """Вид карты и островов на 901–1299 px подбирается по проекции ячеек, а не одной формулой для всех."""
    assert f'matchMedia("{BAND}")' in JS
    assert "fitBetween(VIEWS.map," in JS and "fitBetween(VIEWS.islands," in JS
    # узкий экран — по той же границе, что CSS, а не по ширине холста без полосы прокрутки
    assert 'narrow = matchMedia("(max-width: 900px)").matches' in JS and "narrow = w < 900" not in JS
    # подобранный вид дальше прежнего — отдаление колесом и дальняя плоскость не меньше него
    assert "controls.maxDistance = Math.max(5600," in JS and "camera.far = Math.max(8000," in JS


def test_islands_hide_text_column_in_same_band():
    """Острова во всю ширину на тех же ширинах, где колонка текста скрыта: граница CSS = граница JS."""
    blocks = _media_blocks(CSS, BAND)
    assert any(".has-3d.h0-islands .hero-text { visibility: hidden; }" in b for b in blocks)


def test_no_leftover_offsets_for_wrapped_buttons():
    """На 901–1240 px подсказка и легенда по типам — на местах широкого экрана (кнопок вида две, ряд один)."""
    blocks = _media_blocks(CSS, "(min-width: 901px) and (max-width: 1240px)")
    assert blocks, "блок 901–1240 px есть (перенос кнопок, если не поместятся)"
    body = "\n".join(blocks)
    assert ".h0-howto" not in body and ".h0-islands .h0-legend" not in body
    assert ".has-3d .h0-actions { max-width: 380px; flex-wrap: wrap; }" in body
    assert ".h0-kicker" in body  # строка над h1 переносится раньше кнопок вида


def test_legend_types_only_where_it_would_cover_text():
    """На 901–1199 px ключ высоты скрыт только у карты (у островов он единственная легенда)."""
    body = "\n".join(_media_blocks(CSS, "(min-width: 901px) and (max-width: 1199px)"))
    assert ".has-3d:not(.h0-islands) .h0-legend .h0-hkey { display: none; }" in body
