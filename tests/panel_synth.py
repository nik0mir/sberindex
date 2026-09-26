"""Синтетическое мини-дерево ``data/raw`` для тестов этапа panel (spec_final, Д.2).

Мир из 23 МО в пяти субъектах воспроизводит эталонные случаи части А на малых числах:

- ОКТМО с ведущим нулём, хвост ``000`` в справочнике, 26 версий МО с полуоткрытыми интервалами;
- сдвиг «на 1 января» под кодом прошлого года (МО 3, как Магадан, Л4) — шаг ``version``;
- объединение 2024 года ``union_22``: 4 + 5 → 6, код 4 переходит к 6 (как 46759000, Л8), стабильный код
  частей общий (Л9); в 5-НДФЛ 2023 года объединённый округ уже записан под кодом 4, а кода 5 нет —
  флаг ``union_merged``; давнее объединение ``union_19`` без предшественника в панели;
- объединение 2024 года ``union_23`` (как Сасовский): 21 (в панели) + 22 (вне панели) → 23; в Рязанской
  области строк за 2024 год нет, переносить преемнику нечего;
- Саратов (МО 7): прямой код и часть под другим кодом с тем же стабильным (Л10);
- преобразованное в 2024 году МО 8: новый код есть только в источнике — шаг ``stable``;
- МО 9 в 2025 году: два кандидата моста с разными значениями — ``multi_row``;
- Рязанская область (МО 10 и центр 20 вне панели): в 2024–2025 годах строк нет — перенос (Л6);
- Краснинский район (МО 11): ноль вместо населения (Л7); Кемерово (МО 12): запасной период (Л5);
  в 5-НДФЛ старый код МО 11 встречается только в строках 2016 года, новый — с 2023-го (мост ``stable``);
- МО 2 помечено центром субъекта, как Кировский район Калужской области: центр — городской округ 1;
- районы Москвы 13, 14 без координат центра (внутригородские, по месту работы, Л3);
- Чукотка: полигон через 180° (МО 15) и сосед западнее (МО 16), U-образный полигон МО 2;
- МО 17 и 18 вне панели: старый код 17 стал прямым кодом 18 — шаг ``version`` заблокирован.

``make_raw_tree(root)`` пишет xlsx, gpkg, parquet трат, доступности и связей, gzip-CSV БД ПМО и parquet
5-НДФЛ и возвращает конфиг проекта с путями внутри ``root`` и контрольными числами этого мира.
"""

from __future__ import annotations

import copy
import gzip
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from munnet.config import DEFAULT_CONFIG, Config, load_config

TOP = "Муниципальное образование верхнего уровня"
LOW = "Муниципальное образование нижнего уровня"
TOTAL_OKVED = "Всего по обследуемым видам экономической деятельности"
OTHER_TOTAL_OKVED = "Всего по отдельным видам экономической деятельности"
JAN1, YEAR_VALUE, JAN_DEC = "На 1 января", "Значение показателя за год", "Январь-декабрь"
MR, GO, MO_, VGT = (
    "муниципальный район",
    "городской округ",
    "муниципальный округ",
    "внутригородская территория города федерального значения",
)
CAPITAL = "административный_центр_субъекта"
BDMO_MR = "Муниципальный район"

# Субъекты: код справочника -> (название, префикс ОКТМО). Коды справочника и ОКТМО разные (Л20).
REGIONS = {
    10: ("Первая область", "41"),
    20: ("Вторая область", "42"),
    62: ("Рязанская область", "61"),
    77: ("Москва", "45"),
    87: ("Чукотский автономный округ", "77"),
}

# Версии справочника: territory_id, region_code, oktmo, year_from, year_to, короткое имя, тип, статус,
# change_id_from, change_id_to, shape.
VERSIONS = [
    (1, 10, "41-701-000-000", 2018, 9999, "Первый", GO, CAPITAL, None, None, 1),
    (2, 10, "41-610-000-000", 2018, 9999, "Второй", MR, CAPITAL, None, None, 1),  # как Кировский район
    (3, 10, "41-620-000-000", 2018, 2023, "Сдвиговый", MR, None, None, None, 1),
    (3, 10, "41-520-000-000", 2023, 9999, "Сдвиговый", MO_, None, None, None, 1),
    (4, 10, "41-630-000-000", 2018, 2024, "Павловский Посад", GO, None, None, "union_22", 1),
    (5, 10, "41-631-000-000", 2018, 2024, "Электрогорск", GO, None, None, "union_22", 1),
    (6, 10, "41-630-000-000", 2024, 9999, "Павлово-Посадский", GO, None, "union_22", None, 1),
    (7, 20, "42-701-000-000", 2023, 9999, "Саратов", GO, CAPITAL, "union_19", None, 1),
    (19, 20, "42-701-000-000", 2018, 2023, "Саратов старый", GO, CAPITAL, None, "union_19", 1),
    (8, 20, "42-650-000-000", 2018, 2024, "Преобразованный", MR, None, None, None, 1),
    (8, 20, "42-550-000-000", 2024, 9999, "Преобразованный", MO_, None, None, None, 1),
    (9, 20, "42-660-000-000", 2018, 9999, "Неоднозначный", MR, None, None, None, 2),
    (10, 62, "61-710-000-000", 2018, 9999, "Рязанский", MR, None, None, None, 1),
    (20, 62, "61-701-000-000", 2018, 9999, "Город Третий", GO, CAPITAL, None, None, 1),
    (11, 20, "42-630-000-000", 2018, 9999, "Краснинский", MR, None, None, None, 1),
    (12, 20, "42-640-000-000", 2018, 9999, "Кемеровский", MR, None, None, None, 1),
    (13, 77, "45-374-000-000", 2018, 9999, "Арбат", VGT, None, None, None, 1),
    (14, 77, "45-382-000-000", 2018, 9999, "Тверской", VGT, None, None, None, 1),
    (15, 87, "77-605-000-000", 2018, 9999, "Анадырский", MR, None, None, None, 1),
    (16, 87, "77-610-000-000", 2018, 9999, "Билибинский", MR, None, None, None, 3),
    (17, 20, "42-670-000-000", 2018, 2022, "Блокированный", MR, None, None, None, 1),
    (17, 20, "42-671-000-000", 2022, 9999, "Блокированный", MO_, None, None, None, 1),
    (18, 20, "42-670-000-000", 2022, 9999, "Наследник кода", GO, None, None, None, 1),
    # Как Сасовский (union_23): район 21 в панели, город 22 без трат, преемник 23 — с 2024 года.
    (21, 62, "61-620-000-000", 2018, 2024, "Сасовский район", MR, None, None, "union_23", 1),
    (22, 62, "61-720-000-000", 2018, 2024, "Город Сасово", GO, None, None, "union_23", 1),
    (23, 62, "61-625-000-000", 2024, 9999, "Сасовский округ", MO_, None, "union_23", None, 1),
]
PANEL_IDS = (*range(1, 17), 21, 23)  # МО с тратами; 17–20 и 22 — только в справочнике
MONTHS = [f"{y}-{m:02d}" for y in (2023, 2024) for m in range(1, 13)]
# Покрытие рядов: МО -> список месяцев с данными (по умолчанию все 24).
COVERAGE = {
    4: MONTHS[:12],  # only_2023: предшественник объединения
    5: MONTHS[:12],
    6: MONTHS[14:],  # partial: преемник с 2024-03
    12: MONTHS[:3],  # partial: только январь–март 2023
    15: MONTHS[12:],  # only_2024
    16: MONTHS[:4] + MONTHS[7:],  # partial: разрыв 2023-05…2023-07
    21: MONTHS[:12],  # only_2023: предшественник union_23
    23: MONTHS[14:],  # partial: преемник union_23 с 2024-03
}
CATEGORY_NAMES = {
    "all": "Все категории",
    "food": "Продовольствие",
    "marketplace": "Маркетплейсы",
    "transport": "Транспорт",
    "health": "Здоровье",
    "cafe": "Общественное питание",
}

# Контрольные числа этого мира (заменяют числа реальных данных в конфиге).
N_MO_MONTHS = sum(len(COVERAGE.get(t, MONTHS)) for t in PANEL_IDS)
EXPECTED_HARD = {
    "consumption_rows": N_MO_MONTHS * 6,
    "territories": 18,
    "months": 24,
    "mo_months": N_MO_MONTHS,
    "panel_long_rows": N_MO_MONTHS * 7,
    "dup_keys": 0,
    "other_nonpositive": 0,
    "series_full": 10,
    "series_only_2023": 3,
    "series_only_2024": 1,
    "series_partial": 4,
    "exactly_12_months": 4,
    "internal_gap": 1,
    "dict_versions": len(VERSIONS),
    "dict_territories": 23,
    "in_slice_2023": 16,
    "in_slice_2024": 15,
    "inner_city": 2,
    "regions": 5,
    "polygons_matched": 18,
    "market_access": 15,
}
EXPECTED_SOFT = {
    "pop_jan1_2023": 16,
    "pop_jan1_2024": 14,
    "pop_jan1_2025": 12,
    "pop_avg_both_2024": 12,
    "pop_carried_2024": 1,
    "pop_fallback_period_2023": 1,
    "ndfl_income_2023": 5,  # 1, 8, 13, мост stable у 11 и значение с флагом union_merged у 4
    "ndfl_recipients_2023": 3,
}


# --- Справочник и полигоны ----------------------------------------------------------------------


def dictionary_frame() -> pd.DataFrame:
    """Версии МО в формате ``t_dict_municipal_districts.xlsx`` (ОКТМО с дефисами, 18 полей)."""
    rows = []
    for tid, rc, oktmo, yf, yt, short, mtype, status, ch_from, ch_to, shape in VERSIONS:
        name_region = REGIONS[rc][0]
        inner = mtype == VGT
        rows.append(
            {
                "municipal_district_name_short": short,
                "oktmo": oktmo,
                "municipal_district_name": f"{short} ({mtype})",
                "municipal_district_type": mtype,
                "municipal_district_status": status,
                "shape": shape,
                "shape_linked_oktmo": None,
                "municipal_district_center": None if inner else f"г {short}",
                "source_rosstat": "synthetic.csv",
                "year_from": yf,
                "year_to": yt,
                "territory_id": tid,
                "change_id_from": ch_from,
                "change_id_to": ch_to,
                "region_code": rc,
                "region_name": name_region,
                "municipal_district_center_lat": np.nan if inner else 50.0 + tid / 10,
                "municipal_district_center_lon": np.nan if inner else 40.0 + tid / 10,
            }
        )
    return pd.DataFrame(rows)


def u_shape(x0: float, y0: float, size: float) -> shapely.Polygon:
    """U-образный полигон: центроид лежит в вырезе, то есть снаружи."""
    s = size / 3
    pts = [(0, 0), (3, 0), (3, 3), (2, 3), (2, 1), (1, 1), (1, 3), (0, 3)]
    return shapely.Polygon([(x0 + a * s, y0 + b * s) for a, b in pts])


def _box(lon: float, lat: float, d: float = 0.3) -> shapely.Polygon:
    return shapely.box(lon, lat, lon + d, lat + d)


def polygons_gdf() -> gpd.GeoDataFrame:
    """Полигоны всех 23 МО в EPSG:4326; ``territory_id`` — строка, как в gpkg (Л21)."""
    geoms = {
        1: _box(38.0, 55.0),
        2: u_shape(38.4, 55.0, 0.6),
        3: _box(39.1, 55.0),
        4: _box(38.0, 55.4),
        5: _box(38.4, 55.8, 0.1),
        6: _box(38.0, 55.4, 0.5),
        7: _box(46.0, 51.5),
        8: _box(46.4, 51.5),
        9: _box(46.8, 51.5),
        10: _box(40.0, 54.5),
        11: _box(47.2, 51.5),
        12: _box(47.6, 51.5),
        13: _box(37.60, 55.74, 0.03),
        14: _box(37.64, 55.74, 0.03),
        15: shapely.MultiPolygon([_box(179.0, 64.0, 1.0), _box(-180.0, 64.0, 1.0)]),
        16: _box(176.0, 64.0, 1.0),
        17: _box(48.0, 51.5),
        18: _box(48.4, 51.5),
        19: _box(46.0, 51.9),
        20: _box(39.6, 54.5),
        21: _box(41.0, 54.3),
        22: _box(41.3, 54.3, 0.1),
        23: _box(41.0, 54.3, 0.4),
    }
    tids = sorted(geoms)
    years = {tid: (yf, yt) for tid, _, _, yf, yt, *_ in VERSIONS}
    return gpd.GeoDataFrame(
        {
            "osm_ref": [f"ref{t}" for t in tids],
            "osm_vers": ["1" for _ in tids],
            "territory_id": [str(t) for t in tids],
            "year_from": [years[t][0] for t in tids],
            "year_to": [years[t][1] for t in tids],
        },
        geometry=[geoms[t] for t in tids],
        crs="EPSG:4326",
    )


# --- Траты, доступность, связи ------------------------------------------------------------------


def consumption_frame(seed: int = 0) -> pd.DataFrame:
    """Траты в формате ``consumption.parquet``: шесть категорий, «Прочее» > 0, непоследовательный индекс."""
    rng = np.random.default_rng(seed)
    rows = []
    for tid in PANEL_IDS:
        for date in COVERAGE.get(tid, MONTHS):
            parts = {c: int(rng.integers(500, 5000)) for c in CATEGORY_NAMES if c != "all"}
            total = sum(parts.values()) + int(rng.integers(1000, 8000))
            for code, value in {"all": total, **parts}.items():
                rows.append((date, tid, CATEGORY_NAMES[code], value))
    df = pd.DataFrame(rows, columns=["date", "territory_id", "category", "value"])
    df["territory_id"] = df["territory_id"].astype("int16")
    df["value"] = df["value"].astype("int32")
    df.index = np.arange(len(df)) * 3 + 7  # parquet сохранит его как __index_level_0__ (Л23)
    return df


def market_access_frame() -> pd.DataFrame:
    """Индекс доступности рынков: у МО 16 нет значения, МО 17 вне панели."""
    tids = [*range(1, 16), 17]
    return pd.DataFrame(
        {
            "territory_id": np.array(tids, dtype="int32"),
            "market_access": np.linspace(110.3, 1000.0, len(tids)),
        }
    )


def connection_frame() -> pd.DataFrame:
    """Связи центров МО: каждая пара записана один раз; есть и железная дорога (её не берём)."""
    rows = [
        (1, 2, 30.0, "highway"),
        (3, 1, 50.0, "highway"),  # центр субъекта во втором столбце: искать в обе стороны
        (1, 4, 45.0, "highway"),
        (5, 1, 55.0, "highway"),
        (1, 6, 44.0, "highway"),
        (1, 2, 99.0, "railway"),
        (7, 8, 40.0, "highway"),
        (9, 7, 60.0, "highway"),
        (7, 11, 70.0, "highway"),
        (12, 7, 80.0, "highway"),
        (8, 9, 15.0, "highway"),  # пара без центра: не нужна
        (10, 20, 25.0, "highway"),  # центр Рязанской области вне панели
        (20, 21, 190.0, "highway"),
        (23, 20, 185.0, "highway"),
    ]
    df = pd.DataFrame(rows, columns=["territory_id_x", "territory_id_y", "distance", "type"])
    df["territory_id_x"] = df["territory_id_x"].astype("int64")
    df["territory_id_y"] = df["territory_id_y"].astype("int64")
    return df


# --- БД ПМО -------------------------------------------------------------------------------------

BDMO_BASE_HEAD = ["indicator_section_code", "indicator_section", "indicator_code", "indicator_name"]
BDMO_BASE_TAIL = [
    "region_id",
    "region_name",
    "mun_level",
    "mun_district",
    "municipality",
    "oktmo",
    "mun_type",
    "mun_type_oktmo",
    "oktmo_stable",
    "oktmo_history",
    "oktmo_year_from",
    "oktmo_year_to",
    "year",
    "indicator_value",
    "indicator_unit",
    "indicator_period",
    "comment",
]
BDMO_DIMS = {
    "Y48112027": ["mest"],
    "Y48112014_2023": ["grup_2", "vozr"],
    "Y48112014_2024": ["grup_2", "vozr"],
    "Y48423005": ["okved2"],
    "Y48423006": ["okved2"],
    "Y48423007": ["okved2"],
    "Y48401003": ["okved2"],
    "Y48401006": ["okved2"],
    "Y48401011": ["okved2"],
    "Y49010002": [],
    "Y49010003": [],
    "Y47000004": ["vozr"],
    "Y48060002": [],
    "Y48109001": ["istinv", "okved2"],
}


def brow(
    oktmo: str,
    year: int,
    value,
    period: str,
    *,
    stable: str | None = None,
    unit: str = "Человек",
    level: str = TOP,
    mun_type: str = BDMO_MR,
    mun_type_oktmo: str | None = None,
    history: str = "Без изменений",
    comment: str = "CD",
    **dims,
) -> dict:
    """Строка БД ПМО: все поля строкой, пустое значение — пустая строка."""
    return {
        "indicator_section_code": "31",
        "indicator_section": "Раздел",
        "indicator_name": "Показатель",
        "region_id": "01",
        "region_name": "Регион БД ПМО",
        "mun_level": level,
        "mun_district": "МО",
        "municipality": "МО",
        "oktmo": oktmo,
        "mun_type": mun_type,
        "mun_type_oktmo": mun_type_oktmo if mun_type_oktmo is not None else BDMO_MR,
        "oktmo_stable": stable if stable is not None else oktmo,
        "oktmo_history": history,
        "oktmo_year_from": "2010",
        "oktmo_year_to": "2025",
        "year": str(year),
        "indicator_value": "" if value is None else str(value),
        "indicator_unit": unit,
        "indicator_period": period,
        "comment": comment,
        **dims,
    }


def write_bdmo(path: Path, code: str, rows: list[dict]) -> Path:
    """gzip-CSV БД ПМО с разделителем «;» и заголовком в первой строке."""
    dims = BDMO_DIMS.get(
        Path(path).name.split(".")[0], sorted({k for r in rows for k in r} - set(BDMO_BASE_TAIL))
    )
    cols = [*BDMO_BASE_HEAD, *dims, *BDMO_BASE_TAIL]
    frame = pd.DataFrame(rows, columns=[c for c in cols if c != "indicator_code"])
    frame.insert(2, "indicator_code", code.split("_")[0])
    frame = frame[cols].fillna("")
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as f:
        frame.to_csv(f, sep=";", index=False, lineterminator="\n")
    return path


# Население «Все население» на 1 января по МО и годам (None — строки нет); особые случаи — ниже.
POP = {
    1: 300000,
    2: 50000,
    3: 20000,
    4: 91858,
    5: 29930,
    6: 122210,
    7: 937364,
    8: 15000,
    9: 12000,
    10: 25000,
    11: 11882,
    12: 40000,
    13: 35000,
    14: 80000,
    15: 23000,
    16: 8000,
    18: 5000,
    19: 900000,
    20: 500000,
    21: 30000,
    22: 26000,
}


def population_rows() -> list[dict]:
    """Строки Y48112027 с эталонными случаями А.6, А.9 (коды — без дефисов, как в БД ПМО)."""
    all_pop = {"mest": "Все население"}
    r: list[dict] = []

    def add(oktmo, year, value, period=JAN1, **kw):
        r.append(brow(oktmo, year, value, period, **all_pop, **kw))

    plain = {1: "41701000", 2: "41610000", 9: "42660000", 13: "45374000", 14: "45382000", 15: "77605000"}
    plain |= {16: "77610000", 18: "42670000", 12: "42640000"}
    for year in (2022, 2023, 2024, 2025):
        for tid, code in plain.items():
            if tid == 9 and year == 2025:
                continue  # 2025: только два кандидата моста (multi_row)
            if tid == 12 and year == 2023:
                continue  # Кемерово 2023: только «Значение показателя за год»
            if tid == 13 and year == 2023:
                continue  # дубль с конфликтом — ниже
            add(code, year, POP[tid] + (year - 2022) * 10)
    # МО 3: «на 1 января 2023» под кодом 2022 года (Л4), дальше — под новым.
    add("41620000", 2022, POP[3], stable="41520000")
    add("41620000", 2023, POP[3] + 10, stable="41520000")
    add("41520000", 2024, POP[3] + 20)
    add("41520000", 2025, POP[3] + 30)
    # Объединение union_22: части с общим стабильным кодом (Л9); в 2024 году часть пишется отдельно (Л10).
    for year in (2022, 2023):
        add("41630000", year, POP[4] if year == 2023 else POP[4] - 100, mun_type="Городской округ")
        add(
            "41631000",
            year,
            POP[5] if year == 2023 else POP[5] - 100,
            stable="41630000",
            mun_type="Городской округ",
        )
    add("41630000", 2024, POP[6], mun_type="Городской округ")
    add("41630000", 2025, POP[6] + 590, mun_type="Городской округ")
    add("41631000", 2024, POP[5], stable="41630000", mun_type="Городской округ")
    # Саратов: до 2023 года код у старого МО 19, с 2023-го — у 7; в 2024 году часть под своим кодом.
    add("42701000", 2022, POP[19])
    add("42701000", 2023, 942315)
    add("42701000", 2024, POP[7])
    add("42701000", 2025, 941080)
    add("42702000", 2024, 23861, stable="42701000")
    # МО 8: в 2024–2025 годах только новый код, которого нет в справочнике — мост stable.
    add("42650000", 2022, POP[8])
    add("42650000", 2023, POP[8] + 10)
    add("42560000", 2024, POP[8] + 20, stable="42650000")
    add("42560000", 2025, POP[8] + 30, stable="42650000")
    # МО 9, 2025: два кандидата моста с разными значениями.
    add("42661000", 2025, 100, stable="42660000")
    add("42662000", 2025, 200, stable="42660000")
    # Рязанская область: строки только за 2022 и 2023 годы (Л6); у преемника 23 переносить нечего.
    for year in (2022, 2023):
        add("61710000", year, POP[10])
        add("61701000", year, POP[20])
        add("61620000", year, POP[21])
        add("61720000", year, POP[22])
    # Краснинский: ноль на 1 января 2023 года (Л7).
    add("42630000", 2022, POP[11])
    add("42630000", 2023, 0)
    add("42630000", 2024, 13181)
    add("42630000", 2025, 12995)
    # Кемерово, 2023: нет «На 1 января», есть «Значение показателя за год» (Л5).
    add("42640000", 2023, 40100, period=YEAR_VALUE)
    # Москва, 2023: два одинаково приоритетных значения — dup_conflict; дубль МО 2 с чужим mun_type.
    add("45374000", 2023, 35000)
    add("45374000", 2023, 35500)
    add("41610000", 2023, 99999, mun_type="Внутригородской район")
    # Нижний уровень и служебный код не мешают.
    add("41610000", 2023, 777, level=LOW)
    add("77610000", 2024, POP[16] + 20, stable="CD")  # тот же, что строка МО 16, но стабильный код — CD
    # Городское и сельское население.
    r.append(brow("41701000", 2023, 270000, JAN1, mest="Городское население"))
    r.append(brow("41701000", 2023, 30000, JAN1, mest="Сельское население"))
    r.append(brow("41610000", 2023, 20000, JAN1, mest="Сельское население"))
    r.append(brow("41610000", 2023, "", JAN1, mest="Городское население"))
    r.append(brow("41701000", 2023, 5, JAN1, mest="UD"))
    return r


def employees_rows() -> list[dict]:
    """Y48423005: итог, два «итога» (Л15), кириллица разделов (Л16), нарастающие итоги (Л12)."""
    r = []
    sec = {
        "Раздел А Сельское хозяйство": 100,  # кириллическая А
        "Раздел В Добыча полезных ископаемых": 50,  # кириллическая В
        "Раздел C Обрабатывающие производства": 200,
        "Раздел Е Водоснабжение": 30,  # кириллическая Е
        "Раздел Н Транспортировка и хранение": 70,  # кириллическая Н
        "Раздел O Государственное управление": 150,
        "Раздел P Образование": 200,
        "Раздел Q Здравоохранение": 100,
    }
    r.append(brow("41701000", 2023, 1000, JAN_DEC, okved2=TOTAL_OKVED))
    r.append(brow("41701000", 2023, 900, JAN_DEC, okved2=OTHER_TOTAL_OKVED))
    r.append(brow("41701000", 2023, 800, "Январь-сентябрь", okved2=TOTAL_OKVED))
    r.append(brow("41701000", 2023, 5, JAN_DEC, okved2="Торговля розничная в неспециализированных магазинах"))
    for label, v in sec.items():
        r.append(brow("41701000", 2023, v, JAN_DEC, okved2=label))
    # Сумма раскрытых больше итога на 10% -> нормировка.
    r.append(brow("41610000", 2023, 100, JAN_DEC, okved2=TOTAL_OKVED))
    r.append(brow("41610000", 2023, 60, JAN_DEC, okved2="Раздел А Сельское хозяйство"))
    r.append(brow("41610000", 2023, 50, JAN_DEC, okved2="Раздел C Обрабатывающие производства"))
    # Больше итога на 0,5% -> «не раскрыто» обрезается до 0.
    r.append(brow("41620000", 2023, 1000, JAN_DEC, okved2=TOTAL_OKVED, stable="41520000"))
    r.append(brow("41620000", 2023, 1005, JAN_DEC, okved2="Раздел P Образование", stable="41520000"))
    r.append(brow("45374000", 2023, 90000, JAN_DEC, okved2=TOTAL_OKVED))
    return r


def age_rows(year: int) -> list[dict]:
    """Y48112014: три группы трудоспособности и «Всего»; интервалы через U+2012 с конфликтом (Л19, Л11)."""
    g = {"Моложе трудоспособного возраста": 60000, "Трудоспособный возраст": 170000}
    g["Старше трудоспособного возраста"] = 70000
    r = [brow("41701000", year, 300000, JAN1, grup_2="Всего", vozr="Всего")]
    for label, v in g.items():
        r.append(brow("41701000", year, v, JAN1, grup_2="Всего", vozr=label))
        r.append(brow("41701000", year, v // 2, JAN1, grup_2="Мужчины", vozr=label))
    r.append(brow("41701000", year, 15000, JAN1, grup_2="Всего", vozr="0‒4"))
    r.append(brow("41701000", year, 16000, JAN1, grup_2="Всего", vozr="0‒4"))  # конфликт вне отбора
    # МО 2: сумма групп на 5% меньше итога -> флаг age_sum_mismatch.
    r.append(brow("41610000", year, 50000, JAN1, grup_2="Всего", vozr="Всего"))
    for label, v in {"Моложе трудоспособного возраста": 10000, "Трудоспособный возраст": 30000}.items():
        r.append(brow("41610000", year, v, JAN1, grup_2="Всего", vozr=label))
    r.append(brow("41610000", year, 7500, JAN1, grup_2="Всего", vozr="Старше трудоспособного возраста"))
    return r


def bdmo_tables() -> dict[str, list[dict]]:
    """Строки всех файлов БД ПМО из ``context.indicators`` и ``context.age``."""
    krub = {"unit": "Тысяча рублей"}
    total = {"okved2": TOTAL_OKVED}
    return {
        "Y48112027": population_rows(),
        "Y48112014_2023": age_rows(2023),
        "Y48112014_2024": age_rows(2024),
        "Y48423005": employees_rows(),
        "Y48423006": [
            brow("41701000", 2023, 1200, JAN_DEC, **krub, **total),
            brow("45374000", 2023, 5_000_000, JAN_DEC, **krub, **total),
        ],
        "Y48423007": [
            brow("41701000", 2023, 100000, JAN_DEC, unit="Рубль", **total),
            brow(
                "41610000",
                2023,
                45000,
                JAN_DEC,
                unit="Рубль",
                comment="Аномальное значение показателя",
                **total,
            ),
            brow("41610000", 2023, 40000, "Январь-март", unit="Рубль", **total),
        ],
        "Y48401003": [
            brow("41701000", 2023, 2400, JAN_DEC, **krub, **total),
            brow("41701000", 2023, 999, JAN_DEC, **krub, okved2=OTHER_TOTAL_OKVED),
            brow("42640000", 2023, 800, YEAR_VALUE, **krub, **total),
            brow("41610000", 2023, 700, "Январь-сентябрь", **krub, **total),
            brow("41610000", 2023, 90, "Декабрь", **krub, **total),
            brow("45374000", 2023, 10_000_000, JAN_DEC, **krub, **total),
        ],
        "Y48401006": [brow("41701000", 2023, 120, JAN_DEC, **krub, **total)],
        "Y48401011": [
            brow("41701000", 2023, 5000, JAN_DEC, **krub, **total),
            brow("41701000", 2023, 1000, JAN_DEC, **krub, okved2="Раздел В Добыча полезных ископаемых"),
            brow("41701000", 2023, 3000, JAN_DEC, **krub, okved2="Раздел C Обрабатывающие производства"),
        ],
        "Y49010002": [
            brow("41701000", 2024, 3000, "На 1 июля", unit="Единица"),
            brow("41610000", 2024, 400, "На 1 января", unit="Единица"),
        ],
        "Y49010003": [brow("41701000", 2024, 6000, "На 1 апреля")],
        "Y47000004": [brow("41701000", 2023, 150000, YEAR_VALUE, unit="Единица", vozr="Всего")],
        "Y48060002": [brow("41701000", 2023, 900, YEAR_VALUE, unit="Единица")],
        "Y48109001": [brow("41701000", 2023, 70000, YEAR_VALUE, **krub, istinv="Всего", **total)],
    }


# --- 5-НДФЛ -------------------------------------------------------------------------------------


def ndfl_frame() -> pd.DataFrame:
    """Строки 5-НДФЛ: доход Y777000006 «Всего», получатели Y777000028; мешающие строки (Л1, Л2)."""
    rows = []

    def add(
        code,
        oktmo,
        year,
        value,
        *,
        tax_rate="CD",
        individual="Всего",
        income="Всего",
        authority="Всего",
        level=TOP,
        stable=None,
        history="Без изменений",
    ):
        rows.append(
            {
                "indicator_name": "показатель",
                "indicator_unit": "Рубль" if code != "Y777000028" else "Человек",
                "indicator_code": code,
                "individual_type": individual if code == "Y777000028" else "CD",
                "income": income if code == "Y777000028" else "CD",
                "deduction": "CD",
                "tax_rate": tax_rate,
                "object_name": "МО",
                "object_level": level,
                "object_oktmo": oktmo,
                "oktmo_stable": stable or oktmo,
                "oktmo_history": history,
                "oktmo_year_from": "2010",
                "oktmo_year_to": "2025",
                "year": year,
                "indicator_period": YEAR_VALUE,
                "indicator_value": value,
                "null_value_reason": "NN",
                "report_region_id": "01",
                "report_region_name": "Регион",
                "report_date": pd.Timestamp(f"{year + 1}-06-01").date(),
                "report_type": "МО",
                "tax_authority": authority,
                "index_form": "5-НДФЛ",
                "source": "6-НДФЛ",
            }
        )

    add("Y777000006", "41701000", 2023, 1.2e9, tax_rate="Всего")
    add("Y777000006", "41701000", 2023, 1.0e9, tax_rate="13%")
    add("Y777000030", "41701000", 2023, 0.0)
    add("Y777000006", "41701000", 2023, 5.0e5, tax_rate="Всего", authority="УФНС по другому региону")
    add("Y777000006", "41610000", 2023, 0.0, tax_rate="Всего")  # ноль дохода -> пропуск
    add("Y777000006", "45374000", 2023, 5.0e9, tax_rate="Всего")
    add("Y777000006", "41701000", 2023, 3.0e9, tax_rate="Всего", level=LOW)
    add("Y777000006", "42650000", 2023, 4.0e8, tax_rate="Всего")
    add("Y777000028", "41701000", 2023, 120000.0)
    add("Y777000028", "41701000", 2023, 99.0, authority="УФНС по другому региону")
    add("Y777000028", "41701000", 2023, 1000.0, income="По коду дохода 2000")
    add("Y777000028", "41610000", 2023, 500.0)
    add("Y777000028", "45374000", 2023, 350000.0)
    add("Y777000028", "42560000", 2024, 5000.0, stable="42650000")
    # union_22: в 2023 году объединённый округ уже под кодом части 4, кода части 5 за 2023 год нет.
    add("Y777000006", "41630000", 2022, 1.5e9, tax_rate="Всего")
    add("Y777000006", "41631000", 2022, 3.0e8, tax_rate="Всего", stable="41630000")
    add("Y777000006", "41630000", 2023, 2.0e9, tax_rate="Всего", history="Объединение")
    # МО 11 сменило тип и код: пара «старый код → стабильный» есть только в строках 2016 года другого
    # показателя, а за 2023 год — строка только под новым кодом, которого нет в справочнике.
    add("Y777000030", "42630000", 2016, 0.0, stable="42635000")
    add("Y777000006", "42635000", 2023, 6.0e8, tax_rate="Всего", history="Изменен тип и название")
    df = pd.DataFrame(rows)
    df["year"] = df["year"].astype("int64")
    return df


# --- Сборка дерева ------------------------------------------------------------------------------


def make_config(root: Path) -> Config:
    """Конфиг проекта с путями внутри ``root`` и контрольными числами синтетического мира."""
    base = load_config(DEFAULT_CONFIG)
    data = copy.deepcopy(base.data)
    root = Path(root)
    data["paths"] = {
        "raw": str(root / "data" / "raw"),
        "interim": str(root / "data" / "interim"),
        "processed": str(root / "data" / "processed"),
        "outputs": str(root / "outputs"),
    }
    data["panel"]["controls"]["hard"] = dict(EXPECTED_HARD)
    data["panel"]["controls"]["soft"] = dict(EXPECTED_SOFT)
    return Config(data=data, path=DEFAULT_CONFIG)


def make_raw_tree(root: Path, seed: int = 0) -> Config:
    """Пишет мини-дерево ``root/data/raw`` и возвращает конфиг для него."""
    cfg = make_config(root)
    raw = Path(cfg["paths"]["raw"])
    p, cx = cfg["panel"], cfg["context"]
    for rel in (p["consumption"], p["dictionary"], p["polygons"], cx["ndfl"]["file"]):
        (raw / rel).parent.mkdir(parents=True, exist_ok=True)
    consumption_frame(seed).to_parquet(raw / p["consumption"])
    market_access_frame().to_parquet(raw / p["market_access"], index=False)
    connection_frame().to_parquet(raw / p["connection"], index=False)
    dictionary_frame().to_excel(raw / p["dictionary"], index=False)
    polygons_gdf().to_file(raw / p["polygons"], driver="GPKG", engine="pyogrio")
    for code, rows in bdmo_tables().items():
        write_bdmo(raw / cx["bdmo_dir"] / f"{code}.csv.gz", code, rows)
    ndfl_frame().to_parquet(raw / cx["ndfl"]["file"], index=False)
    return cfg
