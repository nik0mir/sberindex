"""Справочник территорий панели: версии МО, срезы по годам, полигоны, точки, объединения, расстояния.

Ключ всех выходов — ``territory_id`` (int32). ОКТМО нужен только как мост к Росстату и ФНС и хранится
строкой ровно из 8 цифр (spec_final, А.3). Алгоритм — spec_final, А.5; ловушки — Л20–Л22, Л28.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import shapely
from pyproj import CRS

from munnet.config import Config

log = logging.getLogger(__name__)

OKTMO_DIGITS = 8  # ОКТМО МО верхнего уровня — первые 8 цифр кода (А.3)
OKTMO_RE = r"\d{8}(?:000)?"  # 8 цифр в БД ПМО и 5-НДФЛ, 11 с хвостом 000 в справочнике
WGS84 = "EPSG:4326"
M2_PER_KM2 = 1e6
UNION_PREFIX = "union_"  # объединения в change_id_from / change_id_to справочника
INNER_CITY_CODE = "vgt"  # код внутригородской территории в panel.mo_types
CITY_CODE = "go"  # код городского округа в panel.mo_types: при нескольких центрах субъекта берём его
ROAD_TYPE = "highway"  # тип связи в connection.parquet для расстояния до столицы региона (А.5, п. 7)
MAX_EXAMPLES = 10

# Колонки справочника (xlsx), которые читаются строкой: ОКТМО не теряет ведущий ноль.
DICT_STR_COLUMNS = {"oktmo": str, "shape_linked_oktmo": str}


# --- ОКТМО и справочник ------------------------------------------------------------------------


def normalize_oktmo(s: pd.Series) -> pd.Series:
    """Приводит ОКТМО к строке ровно из 8 цифр: убирает всё, кроме цифр, и берёт первые 8.

    Пропуск остаётся пропуском. Допустимы только 8 цифр (БД ПМО, 5-НДФЛ) и 11 цифр с хвостом ``000``
    (справочник, ``79-701-000-000``). Иначе — ошибка, а не дополнение нулями: так ловится потерянный
    ведущий ноль (``01512000`` → ``1512000``, ``01-512-000-000`` → ``1-512-000-000``: без проверки длины
    последний молча стал бы чужим кодом ``15120000``). Все плохие коды перечисляются в одном ``ValueError``.
    """
    raw = s.astype("str")
    present = raw.notna()
    digits = raw.str.replace(r"\D", "", regex=True)
    bad = present & ~digits.fillna("").str.fullmatch(OKTMO_RE)
    if bad.any():
        examples = ", ".join(repr(v) for v in raw[bad].unique()[:MAX_EXAMPLES])
        raise ValueError(
            f"ОКТМО: {int(bad.sum())} кодов не приводятся к {OKTMO_DIGITS} цифрам "
            f"(нужно 8 цифр или 11 с хвостом 000): {examples}"
        )
    return digits.str.slice(0, OKTMO_DIGITS).where(present).astype("str")


def read_dictionary(path: Path) -> pd.DataFrame:
    """Читает версии МО справочника границ (xlsx): ОКТМО строкой, ``oktmo8``, типы ключей (А.5, п. 1).

    ``territory_id`` → int32, ``region_code`` → int16, ``year_from`` и ``year_to`` → int16 (9999 — действует
    сейчас). ``oktmo8`` — нормализованный код (ошибка, если код не приводится к 8 цифрам).
    """
    dic = pd.read_excel(path, dtype=DICT_STR_COLUMNS)
    dic["territory_id"] = dic["territory_id"].astype("int32")
    dic["region_code"] = dic["region_code"].astype("int16")
    dic["year_from"] = dic["year_from"].astype("int16")
    dic["year_to"] = dic["year_to"].astype("int16")
    dic["shape"] = dic["shape"].astype("int8")
    for col in dic.columns:
        if dic[col].dtype == object:
            dic[col] = dic[col].astype("str")
    dic["oktmo8"] = normalize_oktmo(dic["oktmo"])
    return dic.sort_values(["territory_id", "year_from"], kind="mergesort").reset_index(drop=True)


def dictionary_slice(dic: pd.DataFrame, year: int, last_year: int) -> pd.DataFrame:
    """Срез справочника на год: версии с ``year_from ≤ Y′ < year_to``, где ``Y′ = min(year, last_year)``.

    Интервал полуоткрытый. Справочник знает преобразования только по ``last_year``, поэтому для более
    поздних лет берётся срез ``last_year``. В срезе у каждого МО одна версия.
    """
    y = min(int(year), int(last_year))
    out = dic[(dic["year_from"] <= y) & (dic["year_to"] > y)]
    dup = out["territory_id"].duplicated(keep=False)
    if dup.any():
        raise ValueError(f"срез {y}: у МО несколько версий: {sorted(out.loc[dup, 'territory_id'].unique())}")
    return out


def versions_in_years(dic: pd.DataFrame, years: Sequence[int], last_year: int) -> pd.DataFrame:
    """Версии, действующие хотя бы в одном из годов (по срезам ``dictionary_slice``), без повторов."""
    parts = [dictionary_slice(dic, y, last_year) for y in years]
    out = pd.concat(parts).drop_duplicates(subset=["territory_id", "year_from"])
    return out.sort_values(["territory_id", "year_from"], kind="mergesort")


def latest_versions(
    dic: pd.DataFrame, ids: Iterable[int], years: Sequence[int], last_year: int
) -> pd.DataFrame:
    """Для каждого МО — последняя версия из действующих в ``years`` (А.5, п. 3).

    У предшественников объединений 2024 года это версия 2023 года, у преемников — 2024-го. МО, которого
    нет ни в одном срезе годов, получает свою последнюю версию вообще (с предупреждением в логе).
    """
    ids = pd.Index(pd.unique(np.asarray(list(ids), dtype="int32")))
    active = versions_in_years(dic, years, last_year)
    active = active[active["territory_id"].isin(ids)]
    out = active.groupby("territory_id", sort=True).tail(1)
    missing = ids.difference(out["territory_id"])
    if len(missing):
        log.warning("МО без версии в годах %s, берём последнюю версию: %s", list(years), list(missing))
        extra = dic[dic["territory_id"].isin(missing)].groupby("territory_id").tail(1)
        out = pd.concat([out, extra])
    return out.sort_values("territory_id").reset_index(drop=True)


# --- Полигоны и точки ---------------------------------------------------------------------------


def read_polygons(path: Path) -> gpd.GeoDataFrame:
    """Читает полигоны МО (gpkg): ``territory_id`` строкой → int32 (Л21), EPSG:4326, по одному на МО."""
    gdf = gpd.read_file(path, engine="pyogrio")
    if gdf.crs is None or not CRS.from_user_input(gdf.crs).equals(CRS.from_user_input(WGS84)):
        raise ValueError(f"{path}: ожидалась система координат {WGS84}, получена {gdf.crs}")
    tid = pd.to_numeric(gdf["territory_id"], errors="raise")
    gdf["territory_id"] = tid.astype("int32")
    dup = gdf["territory_id"].duplicated(keep=False)
    if dup.any():
        raise ValueError(f"{path}: у МО несколько полигонов: {sorted(gdf.loc[dup, 'territory_id'].unique())}")
    gdf["year_from"] = gdf["year_from"].astype("int16")
    gdf["year_to"] = gdf["year_to"].astype("int16")
    gdf = gdf[["territory_id", "year_from", "year_to", "geometry"]]
    return gdf.sort_values("territory_id").reset_index(drop=True)


def _to_crs(gdf: gpd.GeoDataFrame, crs: str) -> gpd.GeoDataFrame:
    if gdf.crs is not None and CRS.from_user_input(gdf.crs).equals(CRS.from_user_input(crs)):
        return gdf
    return gdf.to_crs(crs)


def _as_multipolygon(geom):
    if geom is None or geom.is_empty:
        return geom
    if geom.geom_type == "Polygon":
        return shapely.MultiPolygon([geom])
    if geom.geom_type == "MultiPolygon":
        return geom
    polys = [g for g in getattr(geom, "geoms", []) if g.geom_type in ("Polygon", "MultiPolygon")]
    parts = [p for g in polys for p in (g.geoms if g.geom_type == "MultiPolygon" else [g])]
    return shapely.MultiPolygon(parts)


def simplify_tolerance(area_m2: np.ndarray, simplify_m: float, simplify_rel: float) -> np.ndarray:
    """Допуск упрощения каждого полигона, м: ``min(simplify_m, simplify_rel · √площади)`` (А.4)."""
    return np.minimum(float(simplify_m), float(simplify_rel) * np.sqrt(np.asarray(area_m2, dtype="float64")))


def project_and_simplify(
    gdf: gpd.GeoDataFrame, crs: str, simplify_m: float, simplify_rel: float
) -> gpd.GeoDataFrame:
    """Проецирует полигоны в ``crs`` и упрощает каждый с допуском ``min(simplify_m, simplify_rel·√площади)``.

    Проекция Альберса с ``lon_0 = 100`` сшивает полигоны Чукотки у антимеридиана (Л28). Адаптивный допуск
    не даёт схлопнуться маленьким МО (районы Москвы ≈ 8 км²). ``preserve_topology=True``; результат —
    MultiPolygon. Уже спроецированный слой повторно не проецируется.
    """
    out = _to_crs(gdf, crs).copy()
    tolerance = simplify_tolerance(shapely.area(out.geometry.values), simplify_m, simplify_rel)
    simple = shapely.simplify(out.geometry.values, tolerance, preserve_topology=True)
    out["geometry"] = [_as_multipolygon(g) for g in simple]
    return out


def representative_points(gdf_proj: gpd.GeoDataFrame, crs: str) -> pd.DataFrame:
    """Точка МО — ``representative_point()`` полигона в проекции (всегда внутри МО, в отличие от центроида).

    Возвращает ``territory_id, x_aea, y_aea`` (м, проекция ``crs``) и ``point_lat, point_lon`` (WGS84).
    """
    proj = _to_crs(gdf_proj, crs)
    pts = proj.geometry.representative_point()
    wgs = gpd.GeoSeries(pts.values, crs=proj.crs).to_crs(WGS84)
    return pd.DataFrame(
        {
            "territory_id": proj["territory_id"].to_numpy(),
            "x_aea": pts.x.to_numpy(dtype="float64"),
            "y_aea": pts.y.to_numpy(dtype="float64"),
            "point_lat": wgs.y.to_numpy(dtype="float64"),
            "point_lon": wgs.x.to_numpy(dtype="float64"),
        }
    )


# --- Объединения и расстояния -------------------------------------------------------------------


def lineage(dic: pd.DataFrame, panel_ids) -> pd.DataFrame:
    """Роли МО панели в объединениях, которые видны в панели (А.5, п. 6).

    Событие ``union_*`` учитывается, если хотя бы один предшественник (версия с ``change_id_to``) есть в
    панели: так остаются объединения 2024 года (``union_22``, ``union_23``), а давние объединения, чьих
    предшественников в тратах нет, не метятся. Предшественник получает ``union_predecessor`` и
    ``successor_id`` (МО с тем же ``change_id_from``), преемник — ``union_successor``. Переносы и
    разделения не трогаем: ``territory_id`` тот же. Ряды не склеиваем.
    """
    ids = pd.Index(pd.unique(np.asarray(list(panel_ids), dtype="int32")))
    to_ = dic["change_id_to"].fillna("").str.startswith(UNION_PREFIX)
    from_ = dic["change_id_from"].fillna("").str.startswith(UNION_PREFIX)
    pred = dic.loc[to_, ["territory_id", "change_id_to", "year_to"]]
    succ = dic.loc[from_, ["territory_id", "change_id_from", "year_from"]]
    events = set(pred.loc[pred["territory_id"].isin(ids), "change_id_to"])
    successor_of = succ[succ["change_id_from"].isin(events)].drop_duplicates("change_id_from")
    successor_of = successor_of.set_index("change_id_from")["territory_id"]

    out = pd.DataFrame({"territory_id": ids.astype("int32")})
    out["lineage_role"] = "none"
    out["lineage_change_id"] = pd.Series(pd.NA, index=out.index, dtype="str")
    out["successor_id"] = pd.array([pd.NA] * len(out), dtype="Int32")

    p = pred[pred["territory_id"].isin(ids) & pred["change_id_to"].isin(events)]
    p = p.sort_values("year_to").drop_duplicates("territory_id", keep="last").set_index("territory_id")
    s = succ[succ["territory_id"].isin(ids) & succ["change_id_from"].isin(events)]
    s = s.sort_values("year_from").drop_duplicates("territory_id", keep="last").set_index("territory_id")
    is_s = out["territory_id"].isin(s.index)
    out.loc[is_s, "lineage_role"] = "union_successor"
    out.loc[is_s, "lineage_change_id"] = out.loc[is_s, "territory_id"].map(s["change_id_from"])
    is_p = out["territory_id"].isin(p.index)  # предшественник важнее: его ряд обрывается
    out.loc[is_p, "lineage_role"] = "union_predecessor"
    change = out.loc[is_p, "territory_id"].map(p["change_id_to"])
    out.loc[is_p, "lineage_change_id"] = change
    out.loc[is_p, "successor_id"] = change.map(successor_of).astype("Int32").to_numpy()
    out["lineage_role"] = out["lineage_role"].astype("str")
    return out


def regional_centers(versions: pd.DataFrame, p: dict) -> pd.DataFrame:
    """Административные центры субъектов для расстояния и ``is_capital``: ``region_code, territory_id``.

    ``versions`` — версии справочника, действующие в годах панели (МО вне панели тоже: центр субъекта
    нужен и тогда, когда трат по нему нет). Центр — версия со статусом ``panel.capital_status``. Если
    у субъекта таких МО несколько (в справочнике так помечены и районы, чей центр — город-столица или
    соседний город), берётся городской округ; если его нет — все помеченные.
    """
    status = _map_or_fail(versions["municipal_district_status"], p["mo_statuses"], "статус МО")
    mo_type = _map_or_fail(versions["municipal_district_type"], p["mo_types"], "тип МО")
    cap = versions.loc[status == p["mo_statuses"][p["capital_status"]], ["region_code", "territory_id"]]
    cap = cap.assign(is_city=(mo_type[cap.index] == CITY_CODE).to_numpy()).drop_duplicates()
    has_city = cap.groupby("region_code")["is_city"].transform("any")
    return cap.loc[cap["is_city"] | ~has_city, ["region_code", "territory_id"]].reset_index(drop=True)


def distance_to_capital(
    connection_path: Path, capitals: pd.DataFrame, panel: pd.DataFrame, *, road_type: str = ROAD_TYPE
) -> pd.Series:
    """Автодорожное расстояние (км) от МО до административного центра своего субъекта (А.5, п. 7).

    ``capitals`` — ``region_code, territory_id`` центров; ``panel`` — ``territory_id, region_code``.
    Пара в ``connection.parquet`` записана один раз, поэтому ищется в обе стороны; читается только
    ``type = road_type`` и только пары с центром (``pyarrow.dataset`` с фильтром). У самого центра 0;
    если центров у субъекта несколько — ближайший; нет центра или дороги — NA. Серия выровнена по ``panel``.
    """
    cap = capitals[["region_code", "territory_id"]].drop_duplicates()
    cap_ids = sorted(int(x) for x in cap["territory_id"].unique())
    dset = ds.dataset(connection_path)
    flt = (ds.field("type") == road_type) & (
        ds.field("territory_id_x").isin(cap_ids) | ds.field("territory_id_y").isin(cap_ids)
    )
    con = dset.to_table(columns=["territory_id_x", "territory_id_y", "distance"], filter=flt).to_pandas()
    con["territory_id_x"] = con["territory_id_x"].astype("int32")
    con["territory_id_y"] = con["territory_id_y"].astype("int32")
    fwd = con.rename(columns={"territory_id_x": "cap", "territory_id_y": "mo"})
    back = con.rename(columns={"territory_id_y": "cap", "territory_id_x": "mo"})
    pairs = pd.concat([fwd, back], ignore_index=True)
    pairs = pairs[pairs["cap"].isin(cap_ids)]
    cap_region = cap.rename(columns={"territory_id": "cap"})
    pairs = pairs.merge(cap_region, on="cap", how="inner")
    self_rows = cap_region.rename(columns={"cap": "mo"}).assign(cap=lambda d: d["mo"], distance=0.0)
    pairs = pd.concat([pairs, self_rows[pairs.columns]], ignore_index=True)
    best = pairs.groupby(["mo", "region_code"], as_index=False)["distance"].min()
    best = best.astype({"mo": "int32", "region_code": "int16"})
    left = pd.DataFrame(
        {
            "mo": panel["territory_id"].astype("int32").to_numpy(),
            "region_code": panel["region_code"].astype("int16").to_numpy(),
        }
    )
    dist = left.merge(best, on=["mo", "region_code"], how="left")["distance"].to_numpy(dtype="float64")
    return pd.Series(dist, index=panel.index, name="dist_capital_km")


# --- Сборка справочника -------------------------------------------------------------------------


def _map_or_fail(values: pd.Series, mapping: dict[str, str], what: str) -> pd.Series:
    present = values.notna()
    unknown = sorted(set(values[present]) - set(mapping))
    if unknown:
        raise ValueError(f"{what}: нет в конфиге значений {unknown}")
    return values.map(mapping).where(present)


def build_territories(
    dic: pd.DataFrame,
    polygons: gpd.GeoDataFrame,
    quality: pd.DataFrame,
    market_access: pd.DataFrame,
    connection_path: Path,
    cfg: Config,
) -> tuple[pd.DataFrame, gpd.GeoDataFrame]:
    """Справочник МО панели (``territories``) и упрощённые полигоны всех МО (``territories_geo``).

    ``quality`` — выход ``consumption.series_quality`` (МО панели и качество ряда), ``market_access`` —
    ``territory_id, market_access``. Названия, тип, статус и регион — из последней версии, действующей в
    годах панели; ``oktmo_<год>`` — из срезов; точка — ``representative_point()`` полигона; площадь — в
    проекции Альберса. Центр субъекта (``regional_centers``) — МО справочника со статусом центра в годах
    панели, в том числе центр, которого нет в панели (отступление от А.5, п. 7: там — только МО панели);
    если так помечено несколько МО субъекта, центр — городской округ. По тем же центрам — ``is_capital``:
    у района, помеченного в справочнике центром, хотя центр субъекта — городской округ (Кировский район
    Калужской области, Биробиджанский район), ``is_capital`` = false и ``mo_status`` пуст. Ошибка, если
    у МО панели нет версии или полигона.
    """
    p = cfg["panel"]
    years = [int(y) for y in p["years"]]
    last_year = int(p["dictionary_last_year"])
    ids = quality["territory_id"].astype("int32")

    unknown = sorted(set(ids) - set(dic["territory_id"]))
    if unknown:
        raise ValueError(f"МО панели нет в справочнике: {unknown[:MAX_EXAMPLES]} (всего {len(unknown)})")
    ver = latest_versions(dic, ids, years, last_year)
    ter = pd.DataFrame(
        {
            "territory_id": ver["territory_id"].astype("int32"),
            "name": ver["municipal_district_name"].astype("str"),
            "name_short": ver["municipal_district_name_short"].astype("str"),
            "region_code": ver["region_code"].astype("int16"),
            "region_name": ver["region_name"].astype("str"),
            "mo_type": _map_or_fail(ver["municipal_district_type"], p["mo_types"], "тип МО"),
            "mo_status": _map_or_fail(ver["municipal_district_status"], p["mo_statuses"], "статус МО"),
            "shape": ver["shape"].astype("int8"),
            "center_name": ver["municipal_district_center"].astype("str"),
            "center_lat": ver["municipal_district_center_lat"].astype("float64"),
            "center_lon": ver["municipal_district_center_lon"].astype("float64"),
        }
    )
    # Столица субъекта — тот же центр, что для расстояния (городской округ важнее района с тем же статусом).
    capitals = regional_centers(versions_in_years(dic, years, last_year), p)
    capital_code = p["mo_statuses"][p["capital_status"]]
    marked = (ter["mo_status"] == capital_code).fillna(False).astype(bool)
    ter["is_capital"] = marked & ter["territory_id"].isin(capitals["territory_id"])
    demoted = marked & ~ter["is_capital"]
    if demoted.any():
        log.info(
            "статус центра субъекта снят у %d МО (в субъекте центр — городской округ): %s",
            int(demoted.sum()),
            ter.loc[demoted, ["territory_id", "name"]].to_dict("records"),
        )
        ter["mo_status"] = ter["mo_status"].where(~demoted)
    inner = (ter["mo_type"] == INNER_CITY_CODE) & ter["region_name"].isin(p["inner_city_regions"])
    ter["is_inner_city"] = inner.astype(bool)
    for y in years:
        sl = dictionary_slice(dic, y, last_year).set_index("territory_id")["oktmo8"]
        ter[f"oktmo_{y}"] = ter["territory_id"].map(sl).astype("str")

    # Полигоны: проекция, точка внутри МО, площадь; упрощение — только для карты.
    missing = sorted(set(ids) - set(polygons["territory_id"]))
    if missing:
        raise ValueError(f"у МО панели нет полигона: {missing[:MAX_EXAMPLES]} (всего {len(missing)})")
    proj = _to_crs(polygons, p["crs"])
    pts = representative_points(proj, p["crs"]).set_index("territory_id")
    area = pd.Series(shapely.area(proj.geometry.values) / M2_PER_KM2, index=proj["territory_id"].to_numpy())
    for col in ("point_lat", "point_lon", "x_aea", "y_aea"):
        ter[col] = ter["territory_id"].map(pts[col]).astype("float64")
    ter["area_km2"] = ter["territory_id"].map(area).astype("float64")

    ma = market_access[["territory_id", "market_access"]].copy()
    ma["territory_id"] = ma["territory_id"].astype("int32")
    ter["market_access"] = (
        ter["territory_id"].map(ma.set_index("territory_id")["market_access"]).astype("float64")
    )

    ter["dist_capital_km"] = distance_to_capital(connection_path, capitals, ter)

    q = quality.copy()
    q["territory_id"] = q["territory_id"].astype("int32")
    ter = ter.merge(q, on="territory_id", how="left", validate="one_to_one")
    ter = ter.merge(lineage(dic, ids), on="territory_id", how="left", validate="one_to_one")

    geo = project_and_simplify(proj, p["crs"], p["simplify_m"], p["simplify_rel"])
    last = dic.groupby("territory_id").tail(1).set_index("territory_id")  # последняя версия МО
    geo["region_code"] = geo["territory_id"].map(last["region_code"]).astype("int16")
    geo["region_name"] = geo["territory_id"].map(last["region_name"]).astype("str")
    geo["in_panel"] = geo["territory_id"].isin(ids).astype(bool)
    cols = ["territory_id", "year_from", "year_to", "region_code", "in_panel", "region_name", "geometry"]
    geo = gpd.GeoDataFrame(geo[cols], geometry="geometry", crs=proj.crs).sort_values("territory_id")
    ter = ter.sort_values("territory_id").reset_index(drop=True)
    return ter, geo.reset_index(drop=True)


def region_prefix(oktmo8: pd.Series) -> pd.Series:
    """Код субъекта в ОКТМО — первые две цифры (по ним ищем, есть ли у региона строки источника)."""
    return oktmo8.str.slice(0, 2)
