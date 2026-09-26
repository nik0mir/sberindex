"""Контракт разделов разведки: факты, записи графиков и таблиц, итог раздела, контекст раздела.

Раздел — модуль с ``SECTION_ID``, ``TITLE`` и ``run_section(ctx) -> Finding``. Через ``SectionContext`` он
регистрирует факты (каждое число текста — в ``facts.json``), сохраняет графики (PNG, SVG, CSV данных) и
таблицы (CSV и Markdown для отчёта), отмечает заголовки, переставшие быть верными. Итог раздела
сохраняется отдельно (``outputs/eda/sections/<id>.json`` и ``<id>_indicators.parquet``), поэтому разделы
можно пересчитывать по одному, не трогая чужие результаты.
"""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, get_args

import numpy as np
import pandas as pd

from munnet import style
from munnet.config import Config
from munnet.contracts import QCError

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from munnet.eda.data import EdaData

FactKind = Literal["int", "num1", "num2", "num3", "pct", "pct_signed", "pp", "rub", "krub", "rho", "p", "str"]
FACT_KINDS: tuple[str, ...] = get_args(FactKind)

# Модули разделов в порядке отчёта и их идентификаторы (префиксы ключей фактов).
SECTIONS: tuple[str, ...] = (
    "munnet.eda.s1_coverage",
    "munnet.eda.s2_level",
    "munnet.eda.s3_rhythm",
    "munnet.eda.s4_basket",
    "munnet.eda.s5_place",
)
SECTION_IDS: tuple[str, ...] = ("e1", "e2", "e3", "e4", "e5")
SYNTHESIS_ID = "syn"
SYNTHESIS_MODULE = "munnet.eda.synthesis"
REPORT_MODULE = "munnet.eda.report"
# Номер «раздела» сводки для генератора случайных чисел: default_rng([seed, номер]).
SYNTHESIS_NUMBER = len(SECTIONS) + 1

EDA_SUBDIR = "eda"
SECTIONS_SUBDIR = "sections"
FIGURES_SUBDIR, DATA_SUBDIR, TABLES_SUBDIR = "figures", "data", "tables"

_FID = re.compile(r"F\d{2}")
_TID = re.compile(r"T\d{2}")
_SLUG = re.compile(r"[a-z0-9]+(_[a-z0-9]+)*")
_KEY = re.compile(r"[a-z][a-z0-9_]*")
PLACEHOLDER = re.compile(r"\{\{\s*([a-z0-9_]+)\.([A-Za-z0-9_]+)\s*\}\}")


class EdaCheckError(QCError):
    """Заголовок графика перестал быть верным на текущих данных (код выхода 3)."""


@dataclass(frozen=True)
class Fact:
    """Число или строка для текста отчёта: значение, вид форматирования, готовый текст по-русски."""

    key: str  # «e3.reliable_share»
    value: float | int | str | None
    kind: FactKind
    text: str  # «15,6%»
    note: str = ""  # определение и выборка


@dataclass(frozen=True)
class FigureRecord:
    """Сохранённый график: пути — относительно ``outputs/eda``."""

    fid: str
    slug: str
    section: str
    title: str
    subtitle: str
    alt: str
    png: str
    svg: str
    data_csv: str
    check: str  # формулировка проверки заголовка
    source: str = ""  # источник данных, как в строке «Источник» на рисунке (пусто в старых JSON)


@dataclass(frozen=True)
class TableRecord:
    """Сохранённая таблица: CSV полностью, Markdown — первые строки для отчёта."""

    tid: str
    slug: str
    section: str
    title: str
    csv: str
    markdown: str


@dataclass
class Finding:
    """Итог раздела разведки."""

    section: str
    title: str
    facts: dict[str, Fact]
    figures: list[FigureRecord]
    tables: list[TableRecord]
    indicators: pd.DataFrame  # territory_id + показатели МО для сводки; может быть пустым
    indicator_labels: dict[str, str]  # колонка -> русская подпись
    summary_md: str  # «Что видно» и «Что это значит» с {{ключами}} фактов
    caveats: list[str] = field(default_factory=list)
    headline_errors: list[str] = field(default_factory=list)


# --- Факты ---------------------------------------------------------------------------------------

_FORMATTERS: dict[str, Callable[[Any], str]] = {
    "int": lambda v: style.fmt_num(v, 0),
    "num1": lambda v: style.fmt_num(v, 1),
    "num2": lambda v: style.fmt_num(v, 2),
    "num3": lambda v: style.fmt_num(v, 3),
    "pct": lambda v: style.fmt_pct(v, 1),
    "pct_signed": lambda v: style.fmt_pct(v, 1, sign=True),
    "pp": lambda v: style.fmt_pp(v, 1),
    "rub": lambda v: style.fmt_rub(v, 0),
    "krub": lambda v: style.fmt_krub(v, 1),
    "rho": lambda v: style.fmt_rho(v),
    "p": lambda v: style.fmt_p(v),
    "str": lambda v: style.NA_TEXT if v is None else str(v),
}


def _plain(value: Any) -> float | int | str | None:
    """Значение факта в типах JSON: numpy -> Python, NaN и бесконечность -> None."""
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    return str(value)


def make_fact(key: str, value: Any, kind: FactKind, note: str = "") -> Fact:
    """Факт с текстом по виду ``kind``: «int» 2190, «pct» 15,6%, «pp» +4,1 п. п., «rho» −0,72…"""
    if kind not in FACT_KINDS:
        raise ValueError(f"факт {key}: неизвестный вид {kind!r}; допустимы {list(FACT_KINDS)}")
    plain = _plain(value)
    if kind == "str":
        plain = None if plain is None else str(value)
    elif isinstance(plain, str):
        raise ValueError(f"факт {key}: вид {kind} требует числа, получено {value!r}")
    return Fact(key=key, value=plain, kind=kind, text=_FORMATTERS[kind](plain), note=note)


# --- Названия МО ---------------------------------------------------------------------------------

# Типы МО, к названию-прилагательному которых добавляется существительное: «Яльчикский» -> «Яльчикский округ».
MO_TYPE_NOUNS: dict[str, str] = {"mr": "район", "mo": "округ"}
ADJECTIVE_ENDINGS: tuple[str, ...] = ("ий", "ый", "ой")


def display_name(name: Any, mo_type: Any = "") -> str:
    """Название МО для таблиц, подписей и текста: к одному слову-прилагательному района (``mr``) или
    округа (``mo``) добавляется «район» или «округ»; остальные названия — как есть, пропуск — «—».

    «Булунский» (mr) -> «Булунский район», «Яльчикский» (mo) -> «Яльчикский округ», «Эгвекинот» (go) —
    без изменений.
    """
    if name is None or (not isinstance(name, str) and pd.isna(name)):
        return style.NA_TEXT
    text = str(name)
    noun = MO_TYPE_NOUNS.get(str(mo_type))
    if noun and " " not in text and text.endswith(ADJECTIVE_ENDINGS):
        return f"{text} {noun}"
    return text


def display_names(frame: pd.DataFrame) -> pd.Series:
    """``display_name`` для строк таблицы: ``name_short`` (пропуск — ``name``) и ``mo_type``, если есть.

    Индекс — как у ``frame``; так разделы подписывают МО одинаково.
    """
    base = frame["name"].astype("object")
    if "name_short" in frame.columns:
        short = frame["name_short"].astype("object")
        base = short.where(short.notna(), base)
    types = frame["mo_type"].astype(str) if "mo_type" in frame.columns else pd.Series("", index=frame.index)
    names = [display_name(n, t) for n, t in zip(base, types, strict=True)]
    return pd.Series(names, index=frame.index, dtype="object", name="name")


# --- Контекст раздела ----------------------------------------------------------------------------


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def write_csv(df: pd.DataFrame, path: Path) -> Path:
    """CSV для машин: UTF-8 без BOM, запятая, дробная точка, без индекса; запись атомарная."""
    _atomic_write_text(Path(path), df.to_csv(index=False, lineterminator="\n"))
    return Path(path)


def write_parquet(df: pd.DataFrame, path: Path) -> Path:
    """Parquet (pyarrow, snappy, без индекса) через временный файл рядом и ``os.replace``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        df.to_parquet(tmp, engine="pyarrow", compression="snappy", index=False)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return path


def write_json(obj: Any, path: Path) -> Path:
    """JSON проекта: UTF-8, ``ensure_ascii=False``, ``sort_keys=True``, ``indent=1``; запись атомарная."""
    _atomic_write_text(Path(path), json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=1) + "\n")
    return Path(path)


# Числа по модулю от 100 в Markdown по умолчанию — без дробной части, меньше — два знака.
MD_BIG_NUMBER = 100


def _close(fig: Figure) -> None:
    """Закрывает фигуру pyplot; фигуры ``style.new_figure`` pyplot не держит — им закрытие не нужно."""
    if getattr(fig.canvas, "manager", None) is not None:
        import matplotlib.pyplot as plt

        plt.close(fig)


def _md_cell(value: Any, formatter: Callable[[Any], str] | None) -> str:
    """Ячейка Markdown; пустая строка — «—»: иначе «|  |» даёт двойной пробел, который ловит lint_ru."""
    if formatter is not None:
        text = formatter(value)
    elif value is None or (not isinstance(value, str) and pd.isna(value)):
        text = style.NA_TEXT
    elif isinstance(value, (bool, np.bool_)):
        text = "да" if value else "нет"
    elif isinstance(value, (int, np.integer)):
        text = style.fmt_num(value)
    elif isinstance(value, (float, np.floating)):
        text = style.fmt_num(value, 0 if abs(value) >= MD_BIG_NUMBER else 2)
    else:
        text = str(value)
    text = " ".join(str(text).replace("|", "\\|").split("\n")).strip()
    return text or style.NA_TEXT


def to_markdown(
    df: pd.DataFrame,
    formats: dict[str, Callable[[Any], str]] | None = None,
    labels: dict[str, str] | None = None,
) -> str:
    """Таблица Markdown по-русски: числа через ``style``, числовые колонки выровнены вправо."""
    formats = formats or {}
    labels = labels or {}
    header = [labels.get(str(c), str(c)).replace("|", "\\|") for c in df.columns]
    numeric = [
        pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c]) for c in df.columns
    ]
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---:" if n else "---" for n in numeric) + "|"]
    for row in df.itertuples(index=False):
        cells = [_md_cell(v, formats.get(str(c))) for c, v in zip(df.columns, row, strict=True)]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


class SectionContext:
    """Всё, что нужно разделу: конфиг, данные, каталог выходов, генератор случайных чисел, регистрация фактов,
    графиков и таблиц."""

    def __init__(self, cfg: Config, data: EdaData, out_dir: Path, section: str, rng: np.random.Generator):
        self.cfg = cfg
        self.data = data
        self.out_dir = Path(out_dir)
        self.section = section
        self.rng = rng
        self.facts: dict[str, Fact] = {}
        self.figures: list[FigureRecord] = []
        self.tables: list[TableRecord] = []
        self.headline_errors: list[str] = []
        limits = cfg["eda"]["figure_limits"]
        self.title_max, self.subtitle_max, self.alt_min = (
            limits["title_max"],
            limits["subtitle_max"],
            limits["alt_min"],
        )
        fig_cfg = cfg["eda"]["figure"]
        self.dpi, self.formats = int(fig_cfg["dpi"]), tuple(fig_cfg["formats"])

    # Факты и заголовки

    def fact(self, key: str, value: Any, kind: FactKind, note: str = "") -> Fact:
        """Регистрирует факт ``<раздел>.<key>`` (ключ — без префикса); повтор ключа — ошибка."""
        if not _KEY.fullmatch(key):
            raise ValueError(
                f"ключ факта {key!r}: латиница в нижнем регистре, цифры и «_», без префикса раздела"
            )
        full = f"{self.section}.{key}"
        if full in self.facts:
            raise ValueError(f"факт {full} уже записан")
        fact = make_fact(full, value, kind, note)
        self.facts[full] = fact
        return fact

    def headline(self, text: str, ok: bool, detail: str) -> str:
        """Возвращает заголовок; если проверка ``ok`` не выполнена — запоминает ошибку с ``detail``
        (имя графика и текущие числа), сборка отчёта тогда падает с кодом 3."""
        if not bool(ok):
            self.headline_errors.append(
                f"{self.section}: заголовок «{text}» неверен на текущих данных — {detail}"
            )
        return text

    # Графики и таблицы

    def _rel(self, path: Path) -> str:
        return path.relative_to(self.out_dir).as_posix()

    def save_figure(
        self,
        fig: Figure,
        *,
        fid: str,
        slug: str,
        title: str,
        subtitle: str,
        alt: str,
        data: pd.DataFrame,
        check: str,
        source: str = style.SOURCE_SBER,
    ) -> FigureRecord:
        """Оформляет график (``style.finish``), пишет ``figures/Fxx_slug.png|svg`` и ``data/Fxx_slug.csv``.

        Проверяет номер, slug, длины заголовка и подзаголовка, альт-текст (нарушение — ``ValueError``);
        фигура закрывается в любом случае.
        """
        try:
            problems = []
            if not _FID.fullmatch(fid):
                problems.append(f"номер {fid!r} не вида F01")
            if not _SLUG.fullmatch(slug):
                problems.append(f"slug {slug!r}: латиница в нижнем регистре, цифры и «_»")
            if not title or len(title) > self.title_max:
                problems.append(f"заголовок {len(title)} знаков, допустимо 1–{self.title_max}")
            if not subtitle or len(subtitle) > self.subtitle_max:
                problems.append(f"подзаголовок {len(subtitle)} знаков, допустимо 1–{self.subtitle_max}")
            if len(alt) < self.alt_min:
                problems.append(f"альт-текст {len(alt)} знаков, нужно не меньше {self.alt_min}")
            if not check:
                problems.append("не задана проверка заголовка")
            if any(f.fid == fid for f in self.figures):
                problems.append(f"{fid} уже сохранён в этом разделе")
            if problems:
                raise ValueError(f"{self.section} {fid} {slug}: " + "; ".join(problems))
            stem = f"{fid}_{slug}"
            style.finish(fig, title, subtitle, source, number=int(fid[1:]))
            paths = style.save(fig, self.out_dir / FIGURES_SUBDIR / stem, formats=self.formats, dpi=self.dpi)
            by_ext = {p.suffix.lstrip("."): p for p in paths}
            data_path = write_csv(pd.DataFrame(data), self.out_dir / DATA_SUBDIR / f"{stem}.csv")
            record = FigureRecord(
                fid=fid,
                slug=slug,
                section=self.section,
                title=title,
                subtitle=subtitle,
                alt=alt,
                png=self._rel(by_ext["png"]) if "png" in by_ext else "",
                svg=self._rel(by_ext["svg"]) if "svg" in by_ext else "",
                data_csv=self._rel(data_path),
                check=check,
                source=source,
            )
            self.figures.append(record)
            return record
        finally:
            _close(fig)

    def save_table(
        self,
        df: pd.DataFrame,
        *,
        tid: str,
        slug: str,
        title: str,
        md_rows: int = 15,
        md_formats: dict[str, Callable[[Any], str]] | None = None,
        md_labels: dict[str, str] | None = None,
        md_columns: Sequence[str] | None = None,
    ) -> TableRecord:
        """Пишет ``tables/Txx_slug.csv`` целиком и Markdown первых ``md_rows`` строк для отчёта.

        ``md_formats`` — форматтер значения по колонке (например, ``style.fmt_pct``), ``md_labels`` —
        русские заголовки колонок только для Markdown (CSV остаётся с машинными именами), ``md_columns`` —
        колонки Markdown в нужном порядке (CSV пишется со всеми колонками; None — все).
        """
        problems = []
        if not _TID.fullmatch(tid):
            problems.append(f"номер {tid!r} не вида T01")
        if not _SLUG.fullmatch(slug):
            problems.append(f"slug {slug!r}: латиница в нижнем регистре, цифры и «_»")
        if any(t.tid == tid for t in self.tables):
            problems.append(f"{tid} уже сохранена в этом разделе")
        unknown = [c for c in (md_columns or ()) if c not in df.columns]
        if unknown:
            problems.append(f"md_columns: нет колонок {unknown}")
        if problems:
            raise ValueError(f"{self.section} {tid} {slug}: " + "; ".join(problems))
        path = write_csv(df, self.out_dir / TABLES_SUBDIR / f"{tid}_{slug}.csv")
        shown = df if md_columns is None else df[list(md_columns)]
        md = to_markdown(shown.head(md_rows), md_formats, md_labels)
        record = TableRecord(
            tid=tid, slug=slug, section=self.section, title=title, csv=self._rel(path), markdown=md
        )
        self.tables.append(record)
        return record

    def finding(
        self,
        *,
        title: str,
        summary_md: str,
        indicators: pd.DataFrame | None = None,
        indicator_labels: dict[str, str] | None = None,
        caveats: Sequence[str] = (),
    ) -> Finding:
        """Собирает итог раздела. Проверяет: у каждого показателя МО есть подпись, ``territory_id`` уникален,
        все ``{{<раздел>.ключ}}`` своего раздела в ``summary_md`` есть среди фактов."""
        labels = dict(indicator_labels or {})
        ind = (
            pd.DataFrame(columns=["territory_id"])
            if indicators is None
            else indicators.reset_index(drop=True)
        )
        problems = []
        if "territory_id" not in ind.columns:
            problems.append("в indicators нет territory_id")
        elif ind["territory_id"].duplicated().any():
            problems.append("в indicators повторяется territory_id")
        unlabeled = [c for c in ind.columns if c != "territory_id" and c not in labels]
        if unlabeled:
            problems.append(f"нет подписей показателей {unlabeled}")
        unknown = sorted(
            f"{s}.{k}"
            for s, k in PLACEHOLDER.findall(summary_md)
            if s == self.section and f"{s}.{k}" not in self.facts
        )
        if unknown:
            problems.append(f"в summary_md неизвестные факты {unknown}")
        if problems:
            raise ValueError(f"{self.section}: " + "; ".join(problems))
        return Finding(
            section=self.section,
            title=title,
            facts=dict(self.facts),
            figures=list(self.figures),
            tables=list(self.tables),
            indicators=ind,
            indicator_labels=labels,
            summary_md=summary_md,
            caveats=list(caveats),
            headline_errors=list(self.headline_errors),
        )


# --- Сохранение итога раздела --------------------------------------------------------------------


def facts_to_json(facts: dict[str, Fact]) -> dict[str, dict[str, Any]]:
    """Факты для ``facts.json``: ключ -> {value, kind, text, note}."""
    return {k: {"value": f.value, "kind": f.kind, "text": f.text, "note": f.note} for k, f in facts.items()}


def save_finding(finding: Finding, sections_dir: Path) -> Path:
    """Пишет ``<id>.json`` (всё, кроме показателей МО) и ``<id>_indicators.parquet``; возвращает путь JSON."""
    sections_dir = Path(sections_dir)
    ind_path = write_parquet(finding.indicators, sections_dir / f"{finding.section}_indicators.parquet")
    payload = {
        "section": finding.section,
        "title": finding.title,
        "facts": facts_to_json(finding.facts),
        "figures": [asdict(f) for f in finding.figures],
        "tables": [asdict(t) for t in finding.tables],
        "indicators": ind_path.name,
        "indicator_labels": finding.indicator_labels,
        "summary_md": finding.summary_md,
        "caveats": finding.caveats,
        "headline_errors": finding.headline_errors,
    }
    return write_json(payload, sections_dir / f"{finding.section}.json")


def load_finding(path: Path) -> Finding:
    """Восстанавливает ``Finding`` из файлов ``save_finding``."""
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    facts = {k: Fact(key=k, **v) for k, v in payload["facts"].items()}
    indicators = pd.read_parquet(path.parent / payload["indicators"])
    return Finding(
        section=payload["section"],
        title=payload["title"],
        facts=facts,
        figures=[FigureRecord(**f) for f in payload["figures"]],
        tables=[TableRecord(**t) for t in payload["tables"]],
        indicators=indicators,
        indicator_labels=payload["indicator_labels"],
        summary_md=payload["summary_md"],
        caveats=payload["caveats"],
        headline_errors=payload["headline_errors"],
    )


def empty_finding(section: str, title: str) -> Finding:
    """Пустой итог (раздел или сводка ещё не реализованы)."""
    return Finding(section, title, {}, [], [], pd.DataFrame(columns=["territory_id"]), {}, "", [], [])
