"""Справочник территорий: ОКТМО, срезы, полигоны, точки, упрощение, объединения, расстояния (А.5)."""

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import shapely
from panel_synth import (
    connection_frame,
    dictionary_frame,
    make_config,
    polygons_gdf,
    u_shape,
)
from pyproj import Transformer

from munnet.panel.territory import (
    dictionary_slice,
    distance_to_capital,
    lineage,
    normalize_oktmo,
    project_and_simplify,
    read_dictionary,
    read_polygons,
    regional_centers,
    representative_points,
    simplify_tolerance,
)

CRS = make_config(".")["panel"]["crs"]


@pytest.fixture(scope="module")
def dic(tmp_path_factory):
    path = tmp_path_factory.mktemp("dict") / "dict.xlsx"
    dictionary_frame().to_excel(path, index=False)
    return read_dictionary(path)


# --- ОКТМО --------------------------------------------------------------------------------------


def test_normalize_oktmo_keeps_leading_zero_and_na():
    out = normalize_oktmo(pd.Series(["01-512-000-000", "01512000", None, "79-701-000-000"]))
    assert out.iloc[0] == "01512000"
    assert out.iloc[1] == "01512000"
    assert pd.isna(out.iloc[2])
    assert out.iloc[3] == "79701000"


def test_normalize_oktmo_rejects_seven_digits_and_nonzero_tail():
    with pytest.raises(ValueError) as err:
        normalize_oktmo(pd.Series(["1512000", "79-701-000-001", "79701000"]))
    msg = str(err.value)
    assert "2 кодов" in msg and "1512000" in msg and "79-701-000-001" in msg


@pytest.mark.parametrize("code", ["1-512-000-000", "1512000000", "151200000"])
def test_normalize_oktmo_rejects_lost_leading_zero_with_zero_tail(code):
    # Без проверки длины код справочника без ведущего нуля молча стал бы чужим кодом 15120000.
    with pytest.raises(ValueError, match=code):
        normalize_oktmo(pd.Series(["01-512-000-000", code]))


def test_read_dictionary_types_and_oktmo8(dic):
    assert dic["territory_id"].dtype == np.int32
    assert dic["region_code"].dtype == np.int16
    assert dic["oktmo8"].str.fullmatch(r"\d{8}").all()
    assert set(dic.loc[dic["territory_id"] == 3, "oktmo8"]) == {"41620000", "41520000"}


# --- Срезы --------------------------------------------------------------------------------------


def test_slice_is_half_open(dic):
    s23 = dictionary_slice(dic, 2023, 2024)
    three = s23[s23["territory_id"] == 3]
    assert list(three["oktmo8"]) == ["41520000"]  # версия 2018–2023 в 2023 году уже не действует
    assert 6 not in set(s23["territory_id"]) and {4, 5} <= set(s23["territory_id"])
    s24 = dictionary_slice(dic, 2024, 2024)
    assert 6 in set(s24["territory_id"]) and not ({4, 5} & set(s24["territory_id"]))


def test_slice_after_last_year_uses_last_year(dic):
    pd.testing.assert_frame_equal(dictionary_slice(dic, 2025, 2024), dictionary_slice(dic, 2024, 2024))


def test_same_code_belongs_to_different_mo_in_different_years(dic):
    code = "41630000"  # как 46759000: у 1471 до 2024 года, у 3014 с 2024-го (Л8)
    s23 = dictionary_slice(dic, 2023, 2024)
    s24 = dictionary_slice(dic, 2024, 2024)
    assert s23.loc[s23["oktmo8"] == code, "territory_id"].tolist() == [4]
    assert s24.loc[s24["oktmo8"] == code, "territory_id"].tolist() == [6]


# --- Полигоны и точки ---------------------------------------------------------------------------


def test_read_polygons_casts_string_territory_id(tmp_path):
    path = tmp_path / "poly.gpkg"
    polygons_gdf().to_file(path, driver="GPKG", engine="pyogrio")
    gdf = read_polygons(path)
    assert gdf["territory_id"].dtype == np.int32
    assert sorted(gdf["territory_id"]) == list(range(1, 24))
    assert str(gdf.crs).upper().endswith("4326")


def test_point_inside_u_shape_while_centroid_outside():
    poly = u_shape(38.4, 55.0, 0.6)
    gdf = gpd.GeoDataFrame({"territory_id": np.array([2], dtype="int32")}, geometry=[poly], crs="EPSG:4326")
    proj = gdf.to_crs(CRS)
    assert not proj.geometry.iloc[0].contains(proj.geometry.iloc[0].centroid)  # центроид в вырезе
    pts = representative_points(proj, CRS)
    assert proj.geometry.iloc[0].contains(shapely.Point(pts["x_aea"].iloc[0], pts["y_aea"].iloc[0]))
    assert poly.contains(shapely.Point(pts["point_lon"].iloc[0], pts["point_lat"].iloc[0]))


def test_polygon_across_antimeridian_is_compact_and_east_of_neighbour():
    gdf = polygons_gdf()
    gdf["territory_id"] = gdf["territory_id"].astype("int32")
    sub = gdf[gdf["territory_id"].isin([15, 16])]
    proj = project_and_simplify(sub, CRS, 2000, 0.05).set_index("territory_id")
    minx, miny, maxx, maxy = proj.geometry.loc[15].bounds
    assert max(maxx - minx, maxy - miny) < 200_000  # 2° долготы на 64° с. ш. ≈ 97 км, а не полмира
    assert proj.geometry.loc[15].geom_type == "MultiPolygon"
    # «Восточнее» в конической проекции — по местному направлению на восток у соседа.
    to_aea = Transformer.from_crs("EPSG:4326", CRS, always_xy=True)
    e0, e1 = np.array(to_aea.transform(176.5, 64.5)), np.array(to_aea.transform(176.6, 64.5))
    east = (e1 - e0) / np.linalg.norm(e1 - e0)
    c15 = np.array(proj.geometry.loc[15].centroid.coords[0])
    c16 = np.array(proj.geometry.loc[16].centroid.coords[0])
    step = c15 - c16
    assert step @ east > 100_000  # ≈ 3,5° долготы восточнее, ≈ 170 км
    assert np.linalg.norm(step) < 300_000


def test_simplify_tolerance_is_capped_by_size():
    assert simplify_tolerance(np.array([9e6]), 2000, 0.05)[0] == pytest.approx(150.0)  # район Москвы
    assert simplify_tolerance(np.array([1e10]), 2000, 0.05)[0] == 2000.0  # крупный район


def _wavy_square(side: float) -> shapely.Polygon:
    t = np.linspace(0, side, 21)
    ring = (
        [(x, 0.0) for x in t]
        + [(side, y + 7.0 * np.sin(y)) for y in t[1:]]
        + [(x, side) for x in t[::-1][1:]]
        + [(0.0, y) for y in t[::-1][1:]]
    )
    return shapely.Polygon(ring)


@pytest.mark.parametrize(
    "poly",
    [
        # Г-образный МО 3 × 3 км: при постоянном допуске 2 км площадь выросла бы на 20%.
        shapely.Polygon([(0, 0), (3000, 0), (3000, 1000), (1000, 1000), (1000, 3000), (0, 3000)]),
        _wavy_square(2000.0),  # квадрат 2 × 2 км при допуске 2 км схлопнулся бы в треугольник (0,5 площади)
    ],
    ids=["L_3km", "square_2km"],
)
def test_adaptive_simplification_keeps_small_polygons(poly):
    gdf = gpd.GeoDataFrame({"territory_id": np.array([1], dtype="int32")}, geometry=[poly], crs=CRS)
    out = project_and_simplify(gdf, CRS, 2000, 0.05)
    assert out.geometry.iloc[0].area == pytest.approx(poly.area, rel=0.05)
    assert out.geometry.iloc[0].is_valid


# --- Объединения и расстояния -------------------------------------------------------------------


def test_lineage_marks_only_unions_seen_in_panel(dic):
    lin = lineage(dic, range(1, 17)).set_index("territory_id")
    assert lin.loc[4, "lineage_role"] == "union_predecessor" and lin.loc[4, "successor_id"] == 6
    assert lin.loc[5, "lineage_role"] == "union_predecessor" and lin.loc[5, "lineage_change_id"] == "union_22"
    assert lin.loc[6, "lineage_role"] == "union_successor" and pd.isna(lin.loc[6, "successor_id"])
    # union_19: предшественника (МО 19) в панели нет — давнее объединение не метится.
    assert lin.loc[7, "lineage_role"] == "none"
    assert (lin.drop([4, 5, 6])["lineage_role"] == "none").all()


def test_lineage_successor_with_predecessor_outside_panel(dic):
    # Как Сасовский (union_23): район 21 в панели, город 22 без трат, преемник 23 появляется в 2024 году.
    lin = lineage(dic, [*range(1, 17), 21, 23]).set_index("territory_id")
    assert lin.loc[21, ["lineage_role", "lineage_change_id", "successor_id"]].tolist() == [
        "union_predecessor",
        "union_23",
        23,
    ]
    assert lin.loc[23, "lineage_role"] == "union_successor" and pd.isna(lin.loc[23, "successor_id"])
    assert 22 not in lin.index


def test_distance_to_capital_both_directions(tmp_path):
    path = tmp_path / "connection.parquet"
    connection_frame().to_parquet(path, index=False)
    capitals = pd.DataFrame({"region_code": [10, 20, 62], "territory_id": [1, 7, 20]})
    panel = pd.DataFrame(
        {"territory_id": [1, 2, 3, 8, 9, 10, 13], "region_code": [10, 10, 10, 20, 20, 62, 77]},
        index=list("abcdefg"),
    )
    dist = distance_to_capital(path, capitals, panel)
    assert list(dist.index) == list("abcdefg")
    assert dist["a"] == 0.0  # сам центр
    assert dist["b"] == 30.0  # автодорога, а не железная (99 км)
    assert dist["c"] == 50.0  # пара записана как (3, 1): ищется в обе стороны
    assert dist["d"] == 40.0 and dist["e"] == 60.0
    assert dist["f"] == 25.0  # центр субъекта вне панели
    assert np.isnan(dist["g"])  # у субъекта нет центра


def test_regional_centers_prefer_city_and_include_non_panel(dic):
    cfg = make_config(".")
    extra = pd.DataFrame(
        {
            "territory_id": np.array([30, 31], dtype="int32"),
            "region_code": np.array([40, 40], dtype="int16"),
            "municipal_district_status": ["административный_центр_субъекта"] * 2,
            "municipal_district_type": ["городской округ", "муниципальный район"],  # как Калуга и Кировский
        }
    )
    versions = pd.concat([dic, extra], ignore_index=True)
    centers = regional_centers(versions, cfg["panel"]).set_index("region_code")["territory_id"]
    assert centers.loc[40] == 30  # из двух помеченных центров — городской округ
    assert centers.loc[62] == 20  # центр Рязанской области вне панели тоже годится
    assert set(centers.loc[[20]]) == {7, 19}  # в справочнике оба Саратова — городские округа
