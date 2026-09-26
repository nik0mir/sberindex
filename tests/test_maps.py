import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import shapely
from matplotlib.patches import Patch
from synth import make_config, make_processed

from munnet import maps, style
from munnet.config import load_config
from munnet.contracts import MissingInputError

NBSP = " "


@pytest.fixture(scope="module")
def geo(tmp_path_factory):
    root = tmp_path_factory.mktemp("maps")
    processed = make_processed(root)
    return maps.load_geometry(processed)


@pytest.fixture(scope="module")
def cfg(tmp_path_factory):
    return make_config(tmp_path_factory.mktemp("cfg"))


def test_antimeridian_polygon_is_compact_and_east_after_projection():
    crs = load_config()["panel"]["crs"]
    chukotka = shapely.MultiPolygon(
        [shapely.box(179.0, 65.0, 180.0, 66.0), shapely.box(-180.0, 65.0, -179.0, 66.0)]
    )
    neighbour = shapely.box(176.0, 65.0, 177.0, 66.0)
    proj = gpd.GeoSeries([chukotka, neighbour], crs="EPSG:4326").to_crs(crs)
    x0, _, x1, _ = proj.iloc[0].bounds
    assert x1 - x0 < 150_000  # два куска у 180° сшиты, а не разнесены через всю карту
    assert proj.iloc[0].centroid.x > proj.iloc[1].centroid.x  # и лежат восточнее полигона на 176°
    assert proj.iloc[0].distance(proj.iloc[1]) < 150_000  # рядом с ним, а не на другом краю карты


def test_quantile_bins_labels_are_russian():
    values = pd.Series(np.linspace(15_000.0, 45_000.0, 101))
    edges, labels = maps.quantile_bins(values, 5, lambda v: style.fmt_krub(v, 0))
    assert len(edges) == 6 and len(labels) == 5
    assert all("–" in lab and "." not in lab.replace("тыс.", "") for lab in labels)
    assert labels[0] == f"15–21{NBSP}тыс.{NBSP}₽"
    _, pct = maps.quantile_bins(pd.Series([0.101, 0.152, 0.2, 0.25]), 2, lambda v: style.fmt_pct(v, 1))
    assert pct[0].endswith("%") and "," in pct[0] and pct[0].count("%") == 1


def test_classify_puts_maximum_in_last_class():
    edges = np.array([0.0, 1.0, 2.0, 3.0])
    cls = maps.classify(pd.Series([0.0, 0.5, 1.0, 2.9, 3.0, np.nan]), edges)
    assert cls.tolist()[:5] == [0, 0, 1, 2, 2] and pd.isna(cls.iloc[5])


def test_map_frame_follows_unions(geo):
    y23, y24 = maps.map_frame(geo, 2023), maps.map_frame(geo, 2024)
    union = geo.loc[(geo["year_from"] == 2024) | (geo["year_to"] == 2024), "territory_id"].tolist()
    assert len(union) == 2
    assert set(union) & set(y23["territory_id"]) != set(union) & set(y24["territory_id"])
    assert len(y23) == len(y24)


def test_region_borders_one_line_per_region_and_cached(geo):
    frame = maps.map_frame(geo, 2024)
    borders = maps.region_borders(frame)
    assert len(borders) == frame["region_code"].nunique()
    assert all(g.geom_type in ("LineString", "MultiLineString") for g in borders)
    assert maps.region_borders(frame) is borders


def test_add_inset_keeps_main_limits(geo, cfg):
    frame = maps.map_frame(geo, 2024)
    values = pd.Series(1.0, index=frame.loc[frame["in_panel"], "territory_id"])
    fig, ax = style.new_figure("map")
    edges, labels = maps.quantile_bins(pd.Series([0.0, 1.0, 2.0]), 2, style.fmt_num)
    maps.choropleth(ax, frame, values, bins=(edges, labels))
    ax.set_xlim(*frame.total_bounds[[0, 2]])
    ax.set_ylim(*frame.total_bounds[[1, 3]])
    before = (ax.get_xlim(), ax.get_ylim())
    moscow = maps.insets_from_config(cfg)[0]
    inset = maps.add_inset(fig, ax, frame, values, moscow, (0.0, 0.6, 0.2, 0.3), bins=(edges, labels))
    assert (ax.get_xlim(), ax.get_ylim()) == before
    assert inset.get_title(loc="left") == moscow.title == "Москва"
    x0, y0, x1, y1 = maps.inset_window(frame, moscow)
    assert x1 - x0 == y1 - y0 == moscow.extent_km * 1000  # окно — квадрат из конфига
    assert inset.get_xlim() == (x0, x1)


def test_inset_window_without_extent_covers_region(geo):
    frame = maps.map_frame(geo, 2024)
    spb = maps.Inset("spb", 78, "Санкт-Петербург")
    x0, y0, x1, y1 = maps.inset_window(frame, spb)
    bx0, by0, bx1, by1 = frame.loc[frame["region_code"] == 78].total_bounds
    assert x0 < bx0 and y0 < by0 and x1 > bx1 and y1 > by1
    with pytest.raises(ValueError, match="нет МО региона"):
        maps.inset_window(frame, maps.Inset("x", 99, "Нет такого"))


def test_missing_regions_hatched_with_legend_entry(geo, cfg):
    frame = maps.map_frame(geo, 2024)
    panel_ids = frame.loc[frame["in_panel"], "territory_id"]
    values = pd.Series(np.linspace(0.8, 1.6, len(panel_ids)), index=panel_ids.to_numpy())
    values.iloc[0] = np.nan  # МО панели без значения — отдельный серый пункт
    fig, ax = maps.russia_map(
        values,
        geo,
        2024,
        k=5,
        fmt=lambda v: style.fmt_num(v, 2) + "×",
        legend_title="Уровень к медиане",
        insets=maps.insets_from_config(cfg),
        absent_note="1 регион без данных",
    )
    legend = [t.get_text() for t in ax.get_legend().get_texts()]
    assert maps.NO_SBER_LABEL in legend and "нет значения" in legend
    assert len(legend) == 5 + 2
    assert maps.absent_regions(frame) == [31]
    assert [t.get_text() for t in ax.texts] == ["1 регион без данных"]  # одна выноска, а не подпись на регион
    assert len(ax.child_axes) == 2  # врезки Москвы и Петербурга


def test_hatch_patch_and_kinds(geo):
    patch = maps.hatch_missing(None, gpd.GeoDataFrame(geometry=[], crs=geo.crs))
    assert isinstance(patch, Patch) and patch.get_hatch() == "////"
    frame = maps.map_frame(geo, 2023)
    ids = frame.loc[frame["in_panel"], "territory_id"].to_numpy()
    values = pd.Series(np.linspace(-1, 1, len(ids)), index=ids)
    fig, ax = maps.russia_map(values, geo, 2023, kind="diverging", center=0.0, legend_title="Отклонение")
    assert ax.child_axes  # полоса непрерывной шкалы
    assert maps.NO_SBER_LABEL in [t.get_text() for t in ax.get_legend().get_texts()]
    status = pd.Series(np.where(np.arange(len(ids)) % 2, "полный ряд", "неполный ряд"), index=ids)
    colors = {"полный ряд": style.PALETTE["marketplace"], "неполный ряд": style.PALETTE["transport"]}
    fig, ax = maps.russia_map(status, geo, 2023, kind="categorical", cmap=colors)
    assert [t.get_text() for t in ax.get_legend().get_texts()][:2] == ["полный ряд", "неполный ряд"]
    with pytest.raises(ValueError):
        maps.russia_map(status, geo, 2023, kind="categorical", cmap="viridis")


def test_load_geometry_accepts_file_path(tmp_path):
    processed = make_processed(tmp_path)
    by_dir = maps.load_geometry(processed)
    by_file = maps.load_geometry(processed / maps.GEO_FILE)
    assert len(by_dir) == len(by_file) and by_file["territory_id"].dtype == np.int32


def test_load_geometry_requires_panel(tmp_path):
    with pytest.raises(MissingInputError, match="сначала запустите этап panel"):
        maps.load_geometry(tmp_path)
