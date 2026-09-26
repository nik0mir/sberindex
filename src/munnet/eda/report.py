"""Отчёт разведки ``docs/eda.md`` (spec_final, В.6, В.7): шаблон + итоги разделов + факты → Markdown.

Текст — шаблон ``templates/eda.md``, числа — только факты (``{{e3.reliable_share}}`` → «15,6%»). Директивы
шаблона (HTML-комментарии, на GitHub не видны):

- ``<!-- section: e2 -->`` — блок раздела: рисунки с подписями, «Что видно / что это значит», таблицы,
  оговорки; рисунки и таблицы, поставленные в шаблоне отдельно, в блок раздела не входят;
- ``<!-- figure: F16 -->``, ``<!-- table: T14 -->`` — один рисунок или одна таблица в этом месте;
- ``<!-- lead: F16 -->`` — заголовок-вывод рисунка жирным (он защищён проверкой) как подводка раздела;
- ``<!-- sources -->`` — версии исходных данных из ``sources`` конфига;
- ``<!-- if e3 e4 -->…<!-- endif -->`` — текст только при итогах всех названных разделов (без вложенности);
- ``<!-- note: … -->`` — заметка для редактора шаблона, в отчёт не попадает.

Проверки (В.7): неизвестный ключ факта — ``KeyError``; незаполненное поле или директива — ``ValueError``;
``lint_ru`` (прямые кавычки, дефис вместо тире, десятичная точка, три точки, двойные пробелы) не пуст —
``EdaCheckError``, отчёт не пишется; PNG в ``docs/img/eda`` больше ``eda.figure.docs_max_kb`` — сначала
сжимается палитрой, не помогло — ``ValueError``. Повторная сборка на тех же данных даёт тот же файл.
"""

from __future__ import annotations

import copy
import io
import os
import re
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from munnet.config import Config
from munnet.eda.base import EDA_SUBDIR, PLACEHOLDER, EdaCheckError, Fact, FigureRecord, Finding, TableRecord

TEMPLATE = Path(__file__).resolve().parent / "templates" / "eda.md"
KB = 1024

# Параметры по умолчанию; ``eda.report_render`` конфига переопределяет их ключ за ключом.
DEFAULTS: dict[str, Any] = {
    # Что lint_ru не считает ошибкой (регулярные выражения): номер лицензии и дата через точку.
    "lint_allow": [r"CC BY(?:-SA)? \d\.\d", r"\b\d{1,2}\.\d{2}\.\d{4}\b"],
    "strict_lint": True,  # замечания lint_ru останавливают сборку (EdaCheckError), отчёт не пишется
    "shrink_oversize": True,  # PNG больше лимита — пересохранить с палитрой (без сглаживания цветов)
    "png_colors": 256,  # цветов палитры при сжатии
}

_DIRECTIVE = re.compile(r"<!--\s*(section|figure|table|lead|sources)\b\s*:?\s*([^>]*?)\s*-->")
_IF_OPEN = re.compile(r"<!--\s*if\s+([^>]*?)\s*-->")
# Перевод строки после открывающего и закрывающего тегов входит в блок: соседние условные пункты списка
# остаются одним плотным списком, а удалённый блок не оставляет пустых строк.
_IF_BLOCK = re.compile(r"<!--\s*if\s+([^>]*?)\s*-->\n?(.*?)<!--\s*endif\s*-->\n?", re.S)
_NOTE = re.compile(r"[ \t]*<!--\s*note\s*:.*?-->[ \t]*\n?", re.S)
_ANY_COMMENT = re.compile(r"<!--")
_LEFTOVER_FIELD = re.compile(r"\{\{|\}\}")
_SECTION_ID = re.compile(r"[a-z]+\d*")

# Названия исходных данных для таблицы версий (ключи — ``sources`` конфига).
SOURCE_NAMES: dict[str, str] = {
    "hackathon": "СберИндекс: траты по МО, индекс доступности рынков, связи между МО",
    "borders": "СберИндекс: справочник и границы МО",
    "bdmo": "Росстат, БД ПМО в обработке «Если быть точным»",
    "ndfl": "ФНС, 5-НДФЛ в обработке «Если быть точным»",
}
SHA_SHOWN = 12  # знаков sha256 в таблице версий (полный хеш — в конфиге)
_RELEASE = re.compile(r"[^/]+_v\d{8}")  # выгрузка «Если быть точным» в адресе: data_bdmo_118_v20250918


def params(cfg: Config | None = None) -> dict[str, Any]:
    """Параметры отчёта: ``DEFAULTS``, поверх — ``eda.report_render``; неизвестный ключ — ``ValueError``."""
    out = copy.deepcopy(DEFAULTS)
    user = (cfg["eda"].get("report_render") or {}) if cfg is not None else {}
    unknown = sorted(set(user) - set(DEFAULTS))
    if unknown:
        raise ValueError(f"eda.report_render: неизвестные ключи {unknown}; допустимы {sorted(DEFAULTS)}")
    out.update(user)
    return out


# --- Поля и условные блоки -----------------------------------------------------------------------


def fill(template: str, facts: Mapping[str, Fact]) -> str:
    """Подставляет ``{{раздел.ключ}}`` → ``Fact.text``.

    Неизвестные ключи — ``KeyError`` со списком всех таких ключей; оставшиеся ``{{`` или ``}}`` (опечатка
    в поле) — ``ValueError``.
    """
    unknown = sorted({f"{s}.{k}" for s, k in PLACEHOLDER.findall(template) if f"{s}.{k}" not in facts})
    if unknown:
        raise KeyError(f"в отчёте неизвестные факты: {', '.join(unknown)}")
    out = PLACEHOLDER.sub(lambda m: facts[f"{m.group(1)}.{m.group(2)}"].text, template)
    for n, line in enumerate(out.splitlines(), 1):
        if _LEFTOVER_FIELD.search(line):
            raise ValueError(f"строка {n}: незаполненное поле: {line.strip()[:120]}")
    return out


def apply_conditions(text: str, present: Iterable[str]) -> str:
    """Оставляет ``<!-- if e3 e4 -->…<!-- endif -->``, только если итоги всех названных разделов есть.

    Вложенные условия и ``if`` без ``endif`` — ``ValueError``.
    """
    have = set(present)

    def repl(m: re.Match) -> str:
        ids = m.group(1).split()
        if not ids or not all(_SECTION_ID.fullmatch(i) for i in ids):
            raise ValueError(f"условие шаблона: неверный список разделов {m.group(1)!r}")
        if _IF_OPEN.search(m.group(2)):
            raise ValueError(f"вложенное условие в блоке «if {m.group(1)}»: не поддерживается")
        return m.group(2) if set(ids) <= have else ""

    out = _IF_BLOCK.sub(repl, text)
    if _IF_OPEN.search(out) or re.search(r"<!--\s*endif\s*-->", out):
        raise ValueError("в шаблоне «if» без «endif» или «endif» без «if»")
    return out


# --- Типографика и проверка текста ---------------------------------------------------------------

_INLINE_CODE = re.compile(r"`[^`]*`")
_LINK_TARGET = re.compile(r"\]\([^)]*\)")
_URL = re.compile(r"https?://\S+")
_PATHLIKE = re.compile(r"\S*[/\\]\S*")
_COMMENT = re.compile(r"<!--.*?-->")
_LIST_MARKER = re.compile(r"^\s*(?:[-*+]|\d+\.)\s+")
MASK = "▢"  # замена кода, ссылок и путей при проверке: не пробел, чтобы не рождать «двойной пробел»
_EMPTY_CELL = re.compile(r"\|\s{2,}\|")

LINT_RULES: tuple[tuple[re.Pattern, str], ...] = (
    (re.compile(r'"'), "прямая кавычка вместо «»"),
    (re.compile(r" - "), "дефис вместо тире «—»"),
    (re.compile(r"\d\.\d"), "десятичная точка вместо запятой"),
    (re.compile(r"\.\.\."), "три точки вместо «…»"),
    (re.compile(r"\S {2,}\S"), "двойной пробел"),
)


def _mask(line: str, allow: Sequence[re.Pattern]) -> str:
    """Строка без того, что не проверяется: код, ссылки, адреса, пути, комментарии, разрешённые образцы."""
    for pattern in (_COMMENT, _INLINE_CODE, _LINK_TARGET, _URL, *allow, _PATHLIKE):
        line = pattern.sub(MASK, line)
    previous = None
    while previous != line:  # пустая ячейка таблицы «|  |» — не двойной пробел; соседние ячейки перекрываются
        previous = line
        line = _EMPTY_CELL.sub(f"|{MASK}|", line)
    return line


def lint_ru(md: str, allow: Sequence[str] | None = None) -> list[str]:
    """Замечания к типографике Markdown (``ru-text``): прямые кавычки, дефис в роли тире, десятичная точка,
    «...», двойные пробелы. Код (блоки и `встроенный`), адреса ссылок, пути и образцы ``allow`` (по умолчанию
    — ``DEFAULTS["lint_allow"]``: лицензии «CC BY 4.0», даты) не проверяются. Пустой список — замечаний нет.
    """
    allowed = [re.compile(p) for p in (DEFAULTS["lint_allow"] if allow is None else allow)]
    problems = []
    fenced = False
    for n, raw in enumerate(md.splitlines(), 1):
        if raw.lstrip().startswith("```"):
            fenced = not fenced
            continue
        if fenced:
            continue
        line = _LIST_MARKER.sub("", _mask(raw, allowed), count=1)
        for pattern, what in LINT_RULES:
            m = pattern.search(line)
            if m:
                start = max(0, m.start() - 30)
                problems.append(f"строка {n}: {what}: «…{line[start : m.end() + 30].strip()}…»")
    return problems


_EMPTY_QUOTES = re.compile(r'""')
_QUOTED = re.compile(r'"([^"\n]+)"')


def typograph(text: str) -> str:
    """Типографика строк из данных (названия МО, подписи): ``""`` убирается, ``"…"`` → «…», ``...`` → «…».

    Код в обратных кавычках не трогается.
    """
    parts = re.split(r"(`[^`]*`)", text)
    for i in range(0, len(parts), 2):
        s = _EMPTY_QUOTES.sub("", parts[i])
        s = _QUOTED.sub(r"«\1»", s)
        parts[i] = s.replace("...", "…")
    return "".join(parts)


NBSP = " "
# Однобуквенные предлоги и союзы: после них — неразрывный пробел (ru-text), чтобы строка не кончалась на «в».
_ONE_LETTER = re.compile(r"(?<![\w-])([ВвСсКкОоУуИиАаЯя]) (?=\S)")
_BEFORE_DASH = re.compile(r"(?<=\S) — ")
_ABBREV = re.compile(r"\b(рис\.|см\.|т\.|п\.) (?=[\w\d{])")
# Начало строки Markdown, которое не продолжает абзац предыдущей строки.
_BLOCK_START = re.compile(r"^\s*(?:[-*+] |\d+\. |#|\||>|```|<!--)")


def unwrap_paragraphs(text: str) -> str:
    """Склеивает перенесённые строки абзацев и пунктов списка шаблона в одну строку.

    В шаблоне строки переносятся для удобства правки, а неразрывный пробел через перенос строки не
    поставить; Markdown всё равно склеивает такие строки. Код, заголовки, таблицы, цитаты, комментарии и
    начала пунктов списка не склеиваются.
    """
    out: list[str] = []
    fenced = False
    for line in text.split("\n"):
        if line.lstrip().startswith("```"):
            fenced = not fenced
            out.append(line)
            continue
        prev = out[-1] if out else ""
        joinable = (
            not fenced
            and line.strip()
            and not _BLOCK_START.match(line)
            and prev.strip()
            and not re.match(r"^\s*(?:#|\||```|<!--)", prev)
            and "-->" not in prev
        )
        if joinable:
            out[-1] = prev.rstrip() + " " + line.strip()
        else:
            out.append(line)
    return "\n".join(out)


def nbsp_markdown(text: str) -> str:
    """Неразрывные пробелы в тексте шаблона: после однобуквенных предлогов и союзов, перед тире, в «рис. 4»,
    «т. е.», «п. п.». Код, таблицы и комментарии не трогаются."""
    out = []
    fenced = False
    for line in text.split("\n"):
        if line.lstrip().startswith("```"):
            fenced = not fenced
        if fenced or line.lstrip().startswith(("|", "<!--", "```")):
            out.append(line)
            continue
        parts = re.split(r"(`[^`]*`)", line)
        for i in range(0, len(parts), 2):
            s = parts[i]
            previous = None
            while previous != s:  # «и в МО»: два однобуквенных подряд
                previous = s
                s = _ONE_LETTER.sub(lambda m: m.group(1) + NBSP, s)
            s = _BEFORE_DASH.sub(NBSP + "— ", s)
            parts[i] = _ABBREV.sub(lambda m: m.group(1) + NBSP, s)
        out.append("".join(parts))
    return "\n".join(out)


# --- Картинки ------------------------------------------------------------------------------------


def _quantize_png(data: bytes, colors: int) -> bytes:
    """PNG с палитрой ``colors`` цветов (медианное сечение, без сглаживания): графики и карты сжимаются в 3–4
    раза почти без видимой разницы; результат детерминирован."""
    from PIL import Image

    with Image.open(io.BytesIO(data)) as im:
        pal = im.convert("RGB").quantize(
            colors=int(colors), method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE
        )
        buf = io.BytesIO()
        pal.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


_FIGURE_FILE = re.compile(r"F\d{2}_[a-z0-9_]+\.png")


def copy_images(
    figures: Sequence[FigureRecord],
    src_root: Path,
    dest: Path,
    max_kb: int,
    *,
    shrink: bool = True,
    colors: int = 256,
) -> list[Path]:
    """Копирует PNG графиков отчёта в ``dest`` (``docs/img/eda``), каждый не больше ``max_kb`` КБ.

    Файл больше лимита при ``shrink`` пересохраняется с палитрой ``colors`` цветов; всё равно больше —
    ``ValueError``, и тогда ничего не копируется. Неизменившиеся файлы не перезаписываются; PNG графиков
    (``Fxx_*.png``), которых в отчёте больше нет, из ``dest`` удаляются. Возвращает пути копий.
    """
    limit = int(max_kb) * KB
    ready: list[tuple[Path, bytes]] = []
    problems = []
    for rec in figures:
        if not rec.png:
            continue
        src = Path(src_root) / rec.png
        if not src.exists():
            raise FileNotFoundError(f"{rec.fid}: нет файла {src}")
        data = src.read_bytes()
        if len(data) > limit and shrink:
            data = _quantize_png(data, colors)
        if len(data) > limit:
            problems.append(f"{rec.fid} {src.name}: {len(data) // KB} КБ, лимит {max_kb} КБ")
        ready.append((Path(dest) / src.name, data))
    if problems:
        raise ValueError("PNG больше лимита eda.figure.docs_max_kb: " + "; ".join(problems))
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for path, data in ready:
        if not path.exists() or path.read_bytes() != data:
            _atomic_write_bytes(path, data)
    keep = {p.name for p, _ in ready}
    for old in dest.glob("*.png"):
        if _FIGURE_FILE.fullmatch(old.name) and old.name not in keep:
            old.unlink()
    return [p for p, _ in ready]


# --- Блоки отчёта --------------------------------------------------------------------------------


def _strip_end(text: str) -> str:
    return text.rstrip().rstrip(".;:")


def _code_path(path: Path | str) -> str:
    """Путь для текста: относительно рабочего каталога (корня репозитория), через «/», в обратных кавычках."""
    p = Path(path)
    try:
        p = p.resolve().relative_to(Path.cwd().resolve())
    except ValueError:
        pass
    return f"`{p.as_posix()}`"


class _Layout:
    """Пути отчёта: куда смотрят картинки из Markdown и где лежат выходы разведки."""

    def __init__(self, cfg: Config):
        self.report = Path(cfg["eda"]["report"])
        self.images = Path(cfg["eda"]["report_images"])
        self.outputs = Path(cfg["paths"]["outputs"]) / EDA_SUBDIR
        rel = os.path.relpath(self.images.resolve(), self.report.resolve().parent)
        self.image_prefix = Path(rel).as_posix()


def render_figure(rec: FigureRecord, layout: _Layout) -> str:
    """Рисунок: картинка с альт-текстом, подпись «Рисунок N. <подзаголовок>», путь к CSV данных."""
    alt = typograph(rec.alt).replace("[", "(").replace("]", ")")
    image = f"{layout.image_prefix}/{Path(rec.png).name}"
    number = int(rec.fid[1:])
    caption = f"Рисунок {number}. {typograph(_strip_end(rec.subtitle))}."
    source = getattr(rec, "source", "")
    if source:
        caption += f" Источник: {typograph(_strip_end(source))}."
    data = _code_path(layout.outputs / rec.data_csv)
    return f"![{alt}]({image})\n\n*{caption}* Данные: {data}."


def render_table(rec: TableRecord, layout: _Layout) -> str:
    """Таблица: номер и название, Markdown первых строк, путь к CSV целиком."""
    title = typograph(_strip_end(rec.title))
    csv = _code_path(layout.outputs / rec.csv)
    return f"**Таблица {rec.tid}.** {title}.\n\n{typograph(rec.markdown)}\n\nЦеликом: {csv}."


def render_section(finding: Finding, layout: _Layout, placed: set[str]) -> str:
    """Блок раздела: рисунки, «Что видно / что это значит», таблицы, оговорки; без поставленных отдельно."""
    figures = sorted((r for r in finding.figures if r.fid not in placed), key=lambda r: r.fid)
    tables = sorted((r for r in finding.tables if r.tid not in placed), key=lambda r: r.tid)
    blocks = [render_figure(r, layout) for r in figures]
    if finding.summary_md.strip():
        blocks.append(finding.summary_md.strip())
    blocks += [render_table(r, layout) for r in tables]
    if finding.caveats:
        items = "\n".join(f"- {typograph(c.strip())}" for c in finding.caveats)
        blocks.append(f"**Оговорки.**\n\n{items}")
    return "\n\n".join(blocks)


def missing_section(sid: str) -> str:
    """Заглушка вместо раздела без сохранённого итога."""
    return (
        f"_Раздел не собран: нет итога раздела {sid}. Запустите "
        f"`python -m munnet eda --only {sid}`, затем `python -m munnet eda --only syn`._"
    )


def sources_md(cfg: Config) -> str:
    """Таблица версий исходных данных из ``sources`` конфига: файл или выгрузка и начало sha256."""
    rows = ["| Данные | Файл или выгрузка | sha256 |", "|---|---|---|"]
    for key, src in cfg["sources"].items():
        name = SOURCE_NAMES.get(key, key)
        if "members" in src:
            found = _RELEASE.search(str(src.get("base_url", "")))
            release = found.group(0) if found else key
            rows.append(f"| {name} | `{release}` | у каждого файла свой, см. `configs/default.yaml` |")
        else:
            sha = str(src.get("sha256", ""))[:SHA_SHOWN]
            rows.append(f"| {name} | `{src.get('file', '')}` | `{sha}…` |")
    return "\n".join(rows)


def _index(findings: Sequence[Finding]) -> tuple[dict[str, FigureRecord], dict[str, TableRecord]]:
    figures, tables = {}, {}
    for f in findings:
        for r in f.figures:
            figures[r.fid] = r
        for t in f.tables:
            tables[t.tid] = t
    return figures, tables


def render_report(
    cfg: Config, findings: list[Finding], synthesis: Finding, template: str | None = None
) -> str:
    """Markdown отчёта: условия → директивы → поля фактов. Шаблон по умолчанию — ``templates/eda.md``.

    Раздел без итога получает заглушку; рисунок или таблица, которых нет ни в одном итоге, — ``ValueError``
    (директиву для необязательного раздела ставят внутри ``<!-- if … -->``).
    """
    text = TEMPLATE.read_text(encoding="utf-8") if template is None else template
    everything = [*findings, synthesis]
    present = {f.section for f in everything}
    by_section = {f.section: f for f in everything}
    figures, tables = _index(everything)
    layout = _Layout(cfg)

    text = nbsp_markdown(unwrap_paragraphs(_NOTE.sub("", apply_conditions(text, present))))
    placed = {
        arg.strip() for kind, arg in _DIRECTIVE.findall(text) if kind in ("figure", "table") and arg.strip()
    }

    def repl(m: re.Match) -> str:
        kind, arg = m.group(1), m.group(2).strip()
        if kind == "sources":
            return sources_md(cfg)
        if kind == "section":
            return (
                render_section(by_section[arg], layout, placed) if arg in by_section else missing_section(arg)
            )
        if kind in ("figure", "lead"):
            if arg not in figures:
                raise ValueError(f"в шаблоне рисунок {arg}, а в итогах разделов его нет")
            if kind == "lead":
                return f"**{typograph(_strip_end(figures[arg].title))}.**"
            return render_figure(figures[arg], layout)
        if arg not in tables:
            raise ValueError(f"в шаблоне таблица {arg}, а в итогах разделов её нет")
        return render_table(tables[arg], layout)

    text = _DIRECTIVE.sub(repl, text)
    if _ANY_COMMENT.search(text):
        line = next(ln for ln in text.splitlines() if "<!--" in ln)
        raise ValueError(f"в шаблоне неизвестная директива: {line.strip()[:120]}")
    facts: dict[str, Fact] = {}
    for f in everything:
        facts.update(f.facts)
    text = fill(text, facts)
    text = "\n".join(line.rstrip() for line in text.splitlines())
    text = re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"
    return text


def write_report(cfg: Config, findings: list[Finding], synthesis: Finding) -> Path:
    """Собирает ``docs/eda.md`` и копирует его картинки в ``docs/img/eda``.

    Порядок: Markdown → ``lint_ru`` (замечания при ``strict_lint`` — ``EdaCheckError``) → картинки (больше
    лимита — ``ValueError``) → запись отчёта через временный файл, только если текст изменился. При любой
    ошибке прежний отчёт остаётся на месте.
    """
    prm = params(cfg)
    md = render_report(cfg, findings, synthesis)
    problems = lint_ru(md, prm["lint_allow"])
    if problems and prm["strict_lint"]:
        raise EdaCheckError(
            f"{len(problems)} замечаний к типографике отчёта, отчёт не записан:\n- " + "\n- ".join(problems)
        )
    layout = _Layout(cfg)
    figures = [r for f in [*findings, synthesis] for r in f.figures]
    copy_images(
        figures,
        layout.outputs,
        layout.images,
        int(cfg["eda"]["figure"]["docs_max_kb"]),
        shrink=bool(prm["shrink_oversize"]),
        colors=int(prm["png_colors"]),
    )
    path = layout.report
    data = md.encode("utf-8")
    if not path.exists() or path.read_bytes() != data:
        _atomic_write_bytes(path, data)
    return path
