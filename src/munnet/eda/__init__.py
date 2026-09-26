"""Этап eda: разведочный анализ панели — графики, таблицы, ``outputs/eda/facts.json`` и отчёт ``docs/eda.md``.

Порядок полного прогона (``python -m munnet eda``): ``data.load`` → разделы E1–E5 (каждый внутри
``style.use()`` со своим генератором ``default_rng([seed, номер раздела])``) → сводка → общие выходы
(``facts.json``, ``figures.json``, ``tables.json``, ``indicators.parquet``) → если какой-то заголовок
перестал быть верным — ``EdaCheckError`` (код 3) и отчёт не пишется, иначе ``report.write_report``.

Один или несколько разделов отдельно (``python -m munnet eda --only e3`` или ``run_sections(cfg, ["e3"])``)
пишут только свои файлы: графики и таблицы своих номеров и ``outputs/eda/sections/<id>.json``. Общие выходы
и отчёт при этом не меняются, поэтому разделы можно дорабатывать параллельно. ``--only syn`` собирает сводку,
общие выходы и отчёт из сохранённых итогов разделов (раздел без сохранённого итога пропускается).
"""

from __future__ import annotations

import importlib
import logging
import time
from collections.abc import Iterable, Sequence
from dataclasses import asdict
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd

from munnet import style
from munnet.config import Config
from munnet.eda.base import (
    EDA_SUBDIR,
    REPORT_MODULE,
    SECTION_IDS,
    SECTIONS,
    SECTIONS_SUBDIR,
    SYNTHESIS_ID,
    SYNTHESIS_MODULE,
    SYNTHESIS_NUMBER,
    EdaCheckError,
    Finding,
    SectionContext,
    empty_finding,
    facts_to_json,
    load_finding,
    save_finding,
    write_json,
    write_parquet,
)
from munnet.eda.data import EdaData, load

log = logging.getLogger(__name__)

SYNTHESIS_TITLE = "Сводка"


def eda_dir(cfg: Config) -> Path:
    """Каталог выходов разведки ``outputs/eda``."""
    path = cfg.dir("outputs") / EDA_SUBDIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def sections_dir(cfg: Config) -> Path:
    """Каталог итогов разделов ``outputs/eda/sections``."""
    return eda_dir(cfg) / SECTIONS_SUBDIR


def section_rng(cfg: Config, number: int) -> np.random.Generator:
    """Генератор раздела ``default_rng([seed, номер])``: результат не зависит от порядка и состава прогона."""
    return np.random.default_rng([int(cfg["seed"]), int(number)])


def _import_optional(module_name: str) -> ModuleType | None:
    """Модуль или None, если его ещё нет (ошибки импорта внутри модуля не глотаются)."""
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as e:
        if e.name == module_name:
            return None
        raise


def _section_module(sid: str) -> ModuleType | None:
    module_name = SECTIONS[SECTION_IDS.index(sid)]
    module = _import_optional(module_name)
    if module is None:
        log.warning("Раздел %s (%s) ещё не написан — пропускаю", sid, module_name)
        return None
    for attr in ("SECTION_ID", "TITLE", "run_section"):
        if not hasattr(module, attr):
            raise AttributeError(f"{module_name}: нет {attr} — контракт раздела, см. munnet.eda.base")
    if module.SECTION_ID != sid:
        raise ValueError(f"{module_name}: SECTION_ID = {module.SECTION_ID!r}, ожидался {sid!r}")
    return module


def run_section(cfg: Config, data: EdaData, sid: str) -> Finding | None:
    """Считает один раздел и сохраняет его итог; None — раздел ещё не написан или не реализован."""
    module = _section_module(sid)
    if module is None:
        return None
    number = SECTION_IDS.index(sid) + 1
    ctx = SectionContext(cfg, data, eda_dir(cfg), sid, section_rng(cfg, number))
    start = time.perf_counter()
    try:
        with style.use():
            finding = module.run_section(ctx)
    except NotImplementedError as e:
        log.warning("Раздел %s не реализован: %s", sid, e)
        return None
    if not isinstance(finding, Finding) or finding.section != sid:
        raise TypeError(f"раздел {sid}: run_section должен вернуть Finding с section = {sid!r}")
    save_finding(finding, sections_dir(cfg))
    log.info("Раздел %s «%s»: %.1f с", sid, finding.title, time.perf_counter() - start)
    return finding


def _check_ids(ids: Iterable[str]) -> list[str]:
    ids = list(dict.fromkeys(i.strip() for i in ids if i.strip()))
    allowed = (*SECTION_IDS, SYNTHESIS_ID)
    unknown = [i for i in ids if i not in allowed]
    if unknown or not ids:
        raise ValueError(f"неизвестные разделы {unknown}; допустимы {list(allowed)}")
    return ids


def _headline_errors(findings: Iterable[Finding]) -> list[str]:
    return [e for f in findings for e in f.headline_errors]


def run_sections(cfg: Config, ids: Sequence[str], data: EdaData | None = None) -> list[Finding]:
    """Считает только разделы ``ids`` (например, ``["e3"]``) и пишет только их файлы.

    Общие выходы и отчёт не трогаются. С ``"syn"`` в ``ids`` после разделов собирается сводка, общие выходы и
    отчёт; итоги остальных разделов берутся из ``outputs/eda/sections``. Ошибки заголовков — ``EdaCheckError``
    после записи файлов.
    """
    ids = _check_ids(ids)
    data = load(cfg) if data is None else data
    findings = [f for sid in ids if sid != SYNTHESIS_ID and (f := run_section(cfg, data, sid)) is not None]
    if SYNTHESIS_ID in ids:
        fresh = {f.section: f for f in findings}
        ordered = []
        for sid in SECTION_IDS:
            if sid in fresh:
                ordered.append(fresh[sid])
            elif (path := sections_dir(cfg) / f"{sid}.json").exists():
                ordered.append(load_finding(path))
            else:
                log.warning("Нет сохранённого итога раздела %s — сводка без него", sid)
        assemble(cfg, data, ordered)
    else:
        errors = _headline_errors(findings)
        if errors:
            raise EdaCheckError(f"{len(errors)} заголовков неверны:\n- " + "\n- ".join(errors))
    return findings


def run_synthesis(cfg: Config, data: EdaData, findings: list[Finding]) -> Finding:
    """Сводка (пакет R) по итогам разделов; пустой итог, если модуля сводки ещё нет."""
    module = _import_optional(SYNTHESIS_MODULE)
    if module is None or not hasattr(module, "run_synthesis"):
        log.warning("Сводка (%s) ещё не написана — пропускаю", SYNTHESIS_MODULE)
        return empty_finding(SYNTHESIS_ID, SYNTHESIS_TITLE)
    ctx = SectionContext(cfg, data, eda_dir(cfg), SYNTHESIS_ID, section_rng(cfg, SYNTHESIS_NUMBER))
    try:
        with style.use():
            synthesis = module.run_synthesis(ctx, findings)
    except NotImplementedError as e:
        log.warning("Сводка не реализована: %s", e)
        return empty_finding(SYNTHESIS_ID, SYNTHESIS_TITLE)
    save_finding(synthesis, sections_dir(cfg))
    return synthesis


def merge_indicators(findings: Iterable[Finding]) -> pd.DataFrame:
    """Показатели МО всех итогов в одну таблицу по ``territory_id``.

    Колонка, уже взятая из более раннего итога, повторно не добавляется, но должна совпадать (иначе
    ``ValueError``): сводка может вернуть и собранную таблицу целиком.
    """
    merged: pd.DataFrame | None = None
    for f in findings:
        ind = f.indicators
        if ind is None or ind.empty or len(ind.columns) <= 1:
            continue
        if merged is None:
            merged = ind.copy()
            continue
        new = [c for c in ind.columns if c != "territory_id" and c not in merged.columns]
        repeated = [c for c in ind.columns if c != "territory_id" and c in merged.columns]
        if repeated:
            both = merged[["territory_id", *repeated]].merge(
                ind[["territory_id", *repeated]], on="territory_id"
            )
            for c in repeated:
                a, b = both[f"{c}_x"], both[f"{c}_y"]
                same = (a == b) | (a.isna() & b.isna())
                if pd.api.types.is_float_dtype(a) and pd.api.types.is_float_dtype(b):
                    same |= np.isclose(a.astype(float), b.astype(float), equal_nan=True)
                if not same.all():
                    raise ValueError(f"показатель {c} из {f.section} расходится с уже собранным")
        merged = merged.merge(ind[["territory_id", *new]], on="territory_id", how="outer")
    if merged is None:
        return pd.DataFrame({"territory_id": pd.Series(dtype="int32")})
    return merged.sort_values("territory_id").reset_index(drop=True)


def write_outputs(cfg: Config, findings: list[Finding]) -> dict[str, Path]:
    """Общие выходы: ``facts.json``, ``figures.json``, ``tables.json``, ``indicators.parquet``."""
    out = eda_dir(cfg)
    facts: dict = {}
    for f in findings:
        facts.update(facts_to_json(f.facts))
    figures = sorted((asdict(r) for f in findings for r in f.figures), key=lambda r: r["fid"])
    tables = sorted((asdict(r) for f in findings for r in f.tables), key=lambda r: r["tid"])
    paths = {
        "facts": write_json(facts, out / "facts.json"),
        "figures": write_json(figures, out / "figures.json"),
        "tables": write_json(tables, out / "tables.json"),
    }
    paths["indicators"] = write_parquet(merge_indicators(findings), out / "indicators.parquet")
    return paths


def assemble(cfg: Config, data: EdaData, findings: list[Finding]) -> Finding:
    """Сводка → общие выходы → проверка заголовков → отчёт. Возвращает итог сводки."""
    synthesis = run_synthesis(cfg, data, findings)
    everything = [*findings, synthesis]
    write_outputs(cfg, everything)
    errors = _headline_errors(everything)
    if errors:
        raise EdaCheckError(f"{len(errors)} заголовков неверны, отчёт не записан:\n- " + "\n- ".join(errors))
    report = _import_optional(REPORT_MODULE)
    if report is not None and hasattr(report, "write_report"):
        path = report.write_report(cfg, findings, synthesis)
        log.info("Отчёт: %s", path)
    else:
        log.warning("Отчёт (%s) ещё не написан — пропускаю", REPORT_MODULE)
    return synthesis


def run(cfg: Config) -> None:
    """Полный прогон разведки: все разделы, сводка, общие выходы, отчёт."""
    data = load(cfg)
    findings = [f for sid in SECTION_IDS if (f := run_section(cfg, data, sid)) is not None]
    assemble(cfg, data, findings)
