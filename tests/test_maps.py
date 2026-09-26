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


def test_row_major_order_reads_by_rows():
    assert maps.row_major_order(9, 5) == [0, 5, 1, 6, 2, 7, 3, 8, 4]
    assert maps.row_major_order(4, 5) == [0, 1, 2, 3]
    assert sorted(maps.row_major_order(7, 3)) == list(range(7))


def _legend_rows(fig, legend) -> list[list[str]]:
    """Подписи легенды, прочитанные по строкам: группировка по высоте, внутри строки — слева направо."""
    fig.draw_without_rendering()
    renderer = fig.canvas.get_renderer()
    items = []
    for t in legend.get_texts():
        box = t.get_window_extent(renderer=renderer)
        items.append((round(box.y0), box.x0, t.get_text()))
    rows: dict[int, list] = {}
    for y, x, text in items:
        rows.setdefault(y, []).append((x, text))
    return [[text for _, text in sorted(row)] for _, row in sorted(rows.items(), reverse=True)]


def test_class_legend_reads_in_ascending_order_by_rows(geo):
    frame = maps.map_frame(geo, 2024)
    ids = frame.loc[frame["in_panel"], "territory_id"].to_numpy()
    values = pd.Series(np.linspace(0.5, 3.0, len(ids)), index=ids)
    values.iloc[0] = np.nan
    fig, ax = maps.russia_map(values, geo, 2024, k=7, fmt=lambda v: style.fmt_num(v, 2) + "×")
    rows = _legend_rows(fig, ax.get_legend())
    flat = [t for row in rows for t in row]
    classes = [t for t in flat if t.endswith("×")]
    lows = [float(t.split("–")[0].replace(",", ".")) for t in classes]
    assert len(rows) >= 2 and lows == sorted(lows)  # по строкам — по возрастанию
    assert flat[-2:] == ["нет значения", maps.NO_SBER_LABEL]  # пункты «нет данных» — в конце


def test_legend_columns_fit_figure_width():
    fig, _ = style.new_figure("map")
    long = [f"очень длинная подпись класса номер {i}" for i in range(8)]
    ncols = maps.legend_ncols(fig, long)
    assert 1 <= ncols < 5
    assert maps.legend_width_in(fig, long, ncols) <= fig.get_figwidth() - 2 * style.MARGIN_IN
    assert maps.legend_ncols(fig, ["1–2", "2–3", "3–4"]) == 3


def test_diverging_norm_is_symmetric_around_center():
    norm = maps.diverging_norm(pd.Series([-0.06, 0.0, 0.1, 0.28]), center=0.0)
    assert (norm.vmin, norm.vcenter, norm.vmax) == (pytest.approx(-0.28), 0.0, pytest.approx(0.28))
    assert norm(-0.06) - 0.5 == pytest.approx(-(norm(0.06) - 0.5))  # равные отклонения — равная яркость


def test_colorbar_and_na_legend_below_map(geo):
    frame = maps.map_frame(geo, 2023)
    ids = frame.loc[frame["in_panel"], "territory_id"].to_numpy()
    values = pd.Series(np.linspace(-0.06, 0.28, len(ids)), index=ids)
    values.iloc[0] = np.nan
    fig, ax = maps.russia_map(
        values,
        geo,
        2023,
        kind="diverging",
        center=0.0,
        legend_title="Отклонение от медианы",
        na_label="нет значения за 2023 год",
        absent_note="Регионы без данных СберИндекса: 8",
    )
    style.finish(fig, "Заголовок", "Подзаголовок", style.SOURCE_SBER, 8)
    fig.draw_without_rendering()
    renderer = fig.canvas.get_renderer()
    map_box = ax.get_window_extent(renderer=renderer)
    cbar_ax = ax.child_axes[0]
    cbar_box = cbar_ax.get_tightbbox(renderer)
    legend_box = ax.get_legend().get_window_extent(renderer=renderer)
    note_box = ax.texts[0].get_window_extent(renderer=renderer)
    assert cbar_box.y1 <= map_box.y0 + 1 and legend_box.y1 <= map_box.y0 + 1  # под картой
    assert not legend_box.overlaps(cbar_box) and not note_box.overlaps(cbar_box)
    assert fig.bbox.contains(legend_box.x1, legend_box.y0)


def test_inset_connectors_from_lower_corners(geo, cfg):
    frame = maps.map_frame(geo, 2024)
    values = pd.Series(1.0, index=frame.loc[frame["in_panel"], "territory_id"])
    fig, ax = maps.russia_map(values, geo, 2024, kind="continuous", insets=maps.insets_from_config(cfg))
    indicators = [c for c in ax.get_children() if type(c).__name__ == "InsetIndicator"]
    assert len(indicators) == 2
    below = 0
    for ind, which, rect in zip(indicators, maps.insets_from_config(cfg), maps.INSET_RECTS, strict=True):
        _, _, _, top = maps.inset_window(frame, which)
        if ax.transLimits.transform((0.0, top))[1] < rect[1]:  # рамка на карте ниже врезки
            below += 1
            visible = [bool(c.get_visible()) for c in ind.connectors]
            assert visible == [True, False, True, False]  # нижний левый и нижний правый углы врезки
    assert below >= 1  # на синтетике рамка Москвы ниже врезок, Петербург — у верхнего края


def test_load_geometry_accepts_file_path(tmp_path):
    processed = make_processed(tmp_path)
    by_dir = maps.load_geometry(processed)
    by_file = maps.load_geometry(processed / maps.GEO_FILE)
    assert len(by_dir) == len(by_file) and by_file["territory_id"].dtype == np.int32


def test_load_geometry_requires_panel(tmp_path):
    with pytest.raises(MissingInputError, match="сначала запустите этап panel"):
        maps.load_geometry(tmp_path)
