"""Этап panel: панель «МО × месяц × категория», справочник территорий и годовой контекст Росстата и ФНС.

Запуск: ``python -m munnet panel``. Входы — ``data/raw`` (траты, доступность рынков и связи СберИндекса,
справочник границ МО, БД ПМО, 5-НДФЛ); выходы (spec_final, А.2):

- ``data/processed``: ``panel_long``, ``panel_wide``, ``territories``, ``territories_geo`` (GeoParquet),
  ``context_long``, ``context_annual``, ``okved_shares`` — по схемам ``munnet.contracts``;
- ``data/interim``: ``bdmo_rows``, ``ndfl_rows`` — отобранные строки источников до соединения (аудит);
- ``outputs/panel``: ``controls.json`` и CSV покрытия и потерь.

Этап идемпотентен: одинаковые входы дают побайтно одинаковые таблицы (сортировка по ключу, без дат).
Жёсткое контрольное число не сошлось — ``QCError`` (код выхода 3).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd

from munnet.config import Config
from munnet.contracts import (
    CONTEXT_ANNUAL,
    CONTEXT_LONG,
    OKVED_SHARES,
    PANEL_LONG,
    PANEL_WIDE,
    TERRITORIES,
    MissingInputError,
    coerce,
    write_table,
)
from munnet.panel.consumption import build_panel, read_consumption
from munnet.panel.context import build_context_full
from munnet.panel.controls import atomic_write, check_controls, compute_controls, write_coverage_reports
from munnet.panel.territory import build_territories, read_dictionary, read_polygons

log = logging.getLogger(__name__)

OUTPUT_SUBDIR = "panel"  # outputs/panel


def input_paths(cfg: Config) -> list[Path]:
    """Все входы этапа в ``data/raw``: траты, доступность, связи, справочник, полигоны, БД ПМО, 5-НДФЛ."""
    raw = Path(cfg["paths"]["raw"])
    p, cx = cfg["panel"], cfg["context"]
    files = [p[k] for k in ("consumption", "market_access", "connection", "dictionary", "polygons")]
    bdmo = {spec["file"] for spec in cx["indicators"].values()} | set(cx["age"]["files"].values())
    files += [f"{cx['bdmo_dir']}/{name}.csv.gz" for name in sorted(bdmo)]
    files.append(cx["ndfl"]["file"])
    return [raw / f for f in files]


def _require_inputs(cfg: Config) -> None:
    missing = [str(path) for path in input_paths(cfg) if not path.exists()]
    if missing:
        raise MissingInputError(
            f"нет входов этапа panel ({len(missing)}): {', '.join(missing)}; "
            "сначала запустите этап data: python -m munnet data"
        )


def run(cfg: Config) -> None:
    """Собирает панель, справочник территорий и контекст, пишет выходы и проверяет контрольные числа."""
    start = time.perf_counter()
    _require_inputs(cfg)
    p = cfg["panel"]
    raw = Path(cfg["paths"]["raw"])
    years = [int(y) for y in p["years"]]

    consumption = read_consumption(raw / p["consumption"], p["categories"])
    panel_long, panel_wide, quality = build_panel(consumption, years)
    log.info("панель: %d МО, %d МО-месяцев, %d строк", len(quality), len(panel_wide), len(panel_long))

    dic = read_dictionary(raw / p["dictionary"])
    polygons = read_polygons(raw / p["polygons"])
    market_access = pd.read_parquet(raw / p["market_access"], columns=["territory_id", "market_access"])
    territories, geo = build_territories(dic, polygons, quality, market_access, raw / p["connection"], cfg)
    log.info("справочник: %d МО панели, %d полигонов", len(territories), len(geo))

    ctx = build_context_full(cfg, dic, territories)
    log.info("контекст: %d значений, %d потерь", len(ctx.context_long), len(ctx.unmatched))

    processed = cfg.dir("processed")
    interim = cfg.dir("interim")
    out_dir = cfg.dir("outputs") / OUTPUT_SUBDIR
    for df, schema, name in (
        (panel_long, PANEL_LONG, "panel_long"),
        (panel_wide, PANEL_WIDE, "panel_wide"),
        (territories, TERRITORIES, "territories"),
        (ctx.context_long, CONTEXT_LONG, "context_long"),
        (ctx.context_annual, CONTEXT_ANNUAL, "context_annual"),
        (ctx.okved_shares, OKVED_SHARES, "okved_shares"),
    ):
        write_table(coerce(df, schema), schema, processed / f"{name}.parquet")
    atomic_write(
        processed / "territories_geo.parquet", lambda t: geo.to_parquet(t, index=False, compression="snappy")
    )
    for name, rows in (("bdmo_rows", ctx.bdmo_rows), ("ndfl_rows", ctx.ndfl_rows)):
        atomic_write(
            interim / f"{name}.parquet",
            lambda t, rows=rows: rows.to_parquet(t, compression="snappy", index=False),
        )

    tables = {
        "consumption": consumption,
        "panel_long": panel_long,
        "panel_wide": panel_wide,
        "territories": territories,
        "dictionary": dic,
        "geo": geo,
        "context_long": ctx.context_long,
        "context_annual": ctx.context_annual,
        "unmatched": ctx.unmatched,
        "bdmo_rows": ctx.bdmo_rows,
        "ndfl_rows": ctx.ndfl_rows,
    }
    actual = compute_controls(tables, cfg, info=ctx.info)
    write_coverage_reports(tables, out_dir)
    warnings = check_controls(actual, cfg, out_dir / "controls.json")
    log.info(
        "этап panel готов за %.1f с; мягких расхождений: %d; выходы: %s, %s",
        time.perf_counter() - start,
        len(warnings),
        processed,
        out_dir,
    )
