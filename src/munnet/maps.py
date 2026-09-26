"""Картограммы МО в равновеликой конической проекции Альберса (spec_final, В.4).

Полигоны читаются из ``territories_geo.parquet`` этапа panel (уже в проекции ``panel.crs``: площади Севера
не раздуваются, Чукотка не рвётся на 180°). МО без данных СберИндекса — серые со штриховкой и отдельным
пунктом легенды, а не пустота, похожая на ноль. Москва и Петербург — во врезках с линиями к рамке на
основной карте: их 247 внутригородских территорий на общей карте не видны. Границы регионов — серая линия
поверх МО.

Раскладка ``russia_map`` подобрана по пустым местам карты России в этой проекции: врезки — в левом верхнем
углу (над Калининградом, левее Кольского полуострова), выноска о выпавших регионах — в левом нижнем углу,
непрерывная шкала — правее неё (под южной границей, где Казахстан); легенда классов — под картой: внутри
карты ей не хватает места.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import geopandas as gpd
import matplotlib as mpl
import numpy as np
import pandas as pd
import shapely
from matplotlib import patheffects
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Colormap, Normalize, TwoSlopeNorm
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

from munnet import style
from munnet.config import Config
from munnet.contracts import GEO_COLUMNS, MissingInputError, SchemaError

GEO_FILE = "territories_geo.parquet"
NO_SBER_LABEL = "нет данных СберИндекса"
NA_LABEL = "нет значения"

MO_EDGE_COLOR = "white"
MO_EDGE_WIDTH = 0.1  # pt: МО без заметной обводки
REGION_EDGE_COLOR = "#8C8C8C"
REGION_EDGE_WIDTH = 0.4  # pt: границы регионов
INSET_EDGE_COLOR = style.TEXT2
INSET_EDGE_WIDTH = 0.6
MAP_PAD = 0.02  # поле вокруг карты, доля размаха
INSET_PAD = 0.08  # поле вокруг региона во врезке без заданного окна, доля размаха
LEGEND_PT = 8
NOTE_PT = 7
# Врезки в долях осей карты (x0, y0, ширина, высота): левый верхний угол — над Калининградом и левее Кольского
# полуострова у карты России пусто.
INSET_RECTS: tuple[tuple[float, float, float, float], ...] = (
    (0.0, 0.79, 0.12, 0.21),
    (0.125, 0.79, 0.11, 0.21),
)
LEGEND_COLS = 5  # легенда классов под картой: не больше пяти колонок
COLORBAR_RECT = (0.2, 0.03, 0.2, 0.025)  # непрерывная шкала: под южной границей (Казахстан), доли осей
COLORBAR_LEGEND_ANCHOR = (0.19, 0.09)  # пункты «нет данных» над полосой шкалы
NOTE_XY = (0.0, 0.0)  # выноска о выпавших регионах: левый нижний угол осей


@dataclass(frozen=True)
class Inset:
    """Врезка: регион по ``region_code`` справочника, подпись и окно (сторона квадрата, км).

    ``extent_km`` задаёт квадратное окно вокруг медианы точек МО региона (плотное ядро города);
    None — весь регион.
    """

    key: str
    region_code: int
    title: str
    extent_km: float | None = None


def insets_from_config(cfg: Config) -> tuple[Inset, ...]:
    """Врезки из ``eda.maps.insets`` конфига в порядке записи."""
    return tuple(
        Inset(
            key=key,
            region_code=int(spec["region_code"]),
            title=str(spec["title"]),
            extent_km=None if spec.get("extent_km") is None else float(spec["extent_km"]),
        )
        for key, spec in cfg["eda"]["maps"]["insets"].items()
    )


def load_geometry(processed_dir: Path) -> gpd.GeoDataFrame:
    """Полигоны всех МО справочника из ``territories_geo.parquet`` (проекция Альберса); id — int32.

    Принимает каталог ``data/processed`` или сам файл (``EdaData.geo_path``).
    """
    path = Path(processed_dir)
    path = path if path.suffix == ".parquet" else path / GEO_FILE
    if not path.exists():
        raise MissingInputError(f"нет {path}: сначала запустите этап panel: python -m munnet panel")
    geo = gpd.read_parquet(path)
    missing = [c for c in GEO_COLUMNS if c not in geo.columns]
    if missing:
        raise SchemaError(f"{GEO_FILE}: нет колонок {missing}")
    geo["territory_id"] = geo["territory_id"].astype("int32")
    return geo


def map_frame(geo: gpd.GeoDataFrame, year: int) -> gpd.GeoDataFrame:
    """Полигоны МО, действующих в году: ``year_from ≤ year < year_to`` (9999 — действует сейчас)."""
    mask = (geo["year_from"] <= year) & (geo["year_to"] > year)
    return geo.loc[mask].reset_index(drop=True)


_BORDER_CACHE: dict[tuple[Any, ...], gpd.GeoSeries] = {}


def _exterior_lines(geom: Any) -> Any:
    """Только внешние контуры: дырки от зазоров между упрощёнными полигонами МО не рисуются."""
    polys = list(geom.geoms) if hasattr(geom, "geoms") else [geom]
    rings = [
        shapely.LineString(p.exterior.coords) for p in polys if p.geom_type == "Polygon" and not p.is_empty
    ]
    return shapely.MultiLineString(rings) if len(rings) != 1 else rings[0]


def region_borders(frame: gpd.GeoDataFrame) -> gpd.GeoSeries:
    """Границы регионов: растворение полигонов МО по ``region_code``; результат кэшируется по составу МО."""
    ids = np.sort(frame["territory_id"].to_numpy(dtype=np.int64))
    key = (len(frame), hashlib.sha1(ids.tobytes()).hexdigest(), str(frame.crs))
    if key not in _BORDER_CACHE:
        valid = frame[["region_code", "geometry"]].copy()
        valid["geometry"] = shapely.make_valid(valid.geometry.values)
        dissolved = valid.dissolve(by="region_code")
        lines = [_exterior_lines(g) for g in dissolved.geometry]
        _BORDER_CACHE[key] = gpd.GeoSeries(lines, index=dissolved.index, crs=frame.crs)
    return _BORDER_CACHE[key]


def quantile_bins(values: pd.Series, k: int, fmt: Callable[[float], str]) -> tuple[np.ndarray, list[str]]:
    """Квантильные классы: границы (k + 1, совпавшие схлопываются) и подписи «20–25 тыс. ₽».

    ``fmt`` форматирует одно число (например, ``lambda v: style.fmt_krub(v, 0)``); общая единица
    остаётся только у правого края.
    """
    v = pd.Series(values, dtype="float64").dropna()
    if v.empty:
        raise ValueError("quantile_bins: нет значений")
    edges = np.unique(np.quantile(v.to_numpy(), np.linspace(0.0, 1.0, k + 1)))
    if len(edges) == 1:
        edges = np.array([edges[0], edges[0]])
    labels = [style.fmt_range(edges[i], edges[i + 1], fmt=fmt) for i in range(len(edges) - 1)]
    return edges, labels


def classify(values: pd.Series, edges: np.ndarray) -> pd.Series:
    """Номер класса 0…k−1: ``edges[i] ≤ v < edges[i + 1]``, максимум — в последнем классе; пропуск — NA."""
    v = pd.Series(values, dtype="float64")
    idx = np.searchsorted(np.asarray(edges)[1:-1], v.to_numpy(), side="right")
    return pd.Series(idx, index=v.index, dtype="Int64").where(v.notna())


def _class_colors(cmap: str | Colormap, k: int) -> list[str]:
    cm = mpl.colormaps[cmap] if isinstance(cmap, str) else cmap
    return [mpl.colors.to_hex(cm(x)) for x in (np.linspace(0.0, 1.0, k) if k > 1 else [0.5])]


def choropleth(
    ax: Axes,
    frame: gpd.GeoDataFrame,
    values: pd.Series,
    *,
    bins: tuple[np.ndarray, Sequence[str]] | None = None,
    cmap: str | Colormap = style.SEQ_CMAP,
    norm: Normalize | None = None,
    categories: Mapping[Any, str] | None = None,
) -> list[Patch]:
    """Заливает МО ``frame`` по ``values`` (серия с индексом ``territory_id``); МО без значения не рисуются.

    Три режима: ``categories`` — значение → цвет (легенда в порядке словаря); ``bins`` — границы и подписи
    классов из ``quantile_bins``; иначе непрерывная шкала ``cmap`` + ``norm``. Возвращает элементы легенды
    (для непрерывной шкалы — пустой список: шкалу рисует ``russia_map``).
    """
    vals = frame["territory_id"].map(values)
    shown = frame.loc[vals.notna()]
    vals = vals.loc[shown.index]
    handles: list[Patch] = []
    if categories is not None:
        unknown = sorted(set(vals.astype(str)) - {str(c) for c in categories})
        if unknown:
            raise ValueError(f"choropleth: нет цвета для значений {unknown}")
        lookup = {str(k): c for k, c in categories.items()}
        colors = [lookup[str(v)] for v in vals]
        handles = [Patch(facecolor=c, label=str(k)) for k, c in categories.items()]
    elif bins is not None:
        edges, labels = bins
        palette = _class_colors(cmap, len(labels))
        colors = [palette[int(i)] for i in classify(vals, edges)]
        handles = [Patch(facecolor=c, label=lab) for c, lab in zip(palette, labels, strict=True)]
    else:
        cm = mpl.colormaps[cmap] if isinstance(cmap, str) else cmap
        norm = norm or Normalize(vmin=float(vals.min()), vmax=float(vals.max()))
        colors = [mpl.colors.to_hex(cm(norm(float(v)))) for v in vals]
    if not shown.empty:
        shown.plot(ax=ax, color=colors, edgecolor=MO_EDGE_COLOR, linewidth=MO_EDGE_WIDTH)
    return handles


def hatch_missing(ax: Axes | None, frame_missing: gpd.GeoDataFrame, label: str = NO_SBER_LABEL) -> Patch:
    """МО без данных: заливка ``NODATA`` со штриховкой ``HATCH``; возвращает пункт легенды."""
    if ax is not None and not frame_missing.empty:
        frame_missing.plot(ax=ax, facecolor=style.NODATA, edgecolor=style.HATCH, hatch="////", linewidth=0)
    return Patch(facecolor=style.NODATA, edgecolor=style.HATCH, hatch="////", label=label, linewidth=0)


def plain_missing(ax: Axes | None, frame_missing: gpd.GeoDataFrame, label: str = NA_LABEL) -> Patch:
    """МО панели без значения показателя (например, мало месяцев): заливка ``NODATA`` без штриховки."""
    if ax is not None and not frame_missing.empty:
        frame_missing.plot(ax=ax, color=style.NODATA, edgecolor=MO_EDGE_COLOR, linewidth=MO_EDGE_WIDTH)
    return Patch(facecolor=style.NODATA, label=label)


def _padded_bounds(frame: gpd.GeoDataFrame, pad: float) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = frame.total_bounds
    dx, dy = (x1 - x0) * pad, (y1 - y0) * pad
    return x0 - dx, y0 - dy, x1 + dx, y1 + dy


def _draw_layers(
    ax: Axes, frame: gpd.GeoDataFrame, values: pd.Series, *, na_label: str = NA_LABEL, **kw: Any
) -> list[Patch]:
    """Слои карты: значения, МО панели без значения, МО без данных СберИндекса, границы регионов."""
    vals = frame["territory_id"].map(values)
    in_panel = frame["in_panel"].astype(bool)
    handles = choropleth(ax, frame, values, **kw)
    panel_na = frame.loc[vals.isna() & in_panel]
    absent = frame.loc[~in_panel]
    if not panel_na.empty:
        handles.append(plain_missing(ax, panel_na, na_label))
    if not absent.empty:
        handles.append(hatch_missing(ax, absent))
    region_borders(frame).plot(ax=ax, color=REGION_EDGE_COLOR, linewidth=REGION_EDGE_WIDTH)
    return handles


def inset_window(frame: gpd.GeoDataFrame, which: Inset) -> tuple[float, float, float, float]:
    """Окно врезки (x0, y0, x1, y1) в координатах карты: квадрат ``extent_km`` вокруг медианы точек МО
    региона или весь регион с полем."""
    part = frame.loc[frame["region_code"].astype(int) == int(which.region_code)]
    if part.empty:
        raise ValueError(f"врезка {which.key}: нет МО региона {which.region_code} в кадре карты")
    if which.extent_km is None:
        return _padded_bounds(part, INSET_PAD)
    points = part.geometry.representative_point()
    cx, cy = float(np.median(points.x)), float(np.median(points.y))
    half = which.extent_km * 1000 / 2
    return cx - half, cy - half, cx + half, cy + half


def add_inset(
    fig: Figure,
    ax: Axes,
    frame: gpd.GeoDataFrame,
    values: pd.Series,
    which: Inset,
    rect: tuple[float, float, float, float],
    *,
    na_label: str = NA_LABEL,
    **kw: Any,
) -> Axes:
    """Врезка региона ``which`` в прямоугольнике ``rect`` (доли осей ``ax``).

    Во врезке — те же слои и цвета, что на основной карте (``kw`` передаются в ``choropleth``); на основной
    карте окно врезки обведено и соединено с ней линиями. Пределы основных осей не меняются.
    """
    x0, y0, x1, y1 = inset_window(frame, which)
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    inset = ax.inset_axes(rect)
    visible = frame.cx[x0:x1, y0:y1]
    _draw_layers(inset, visible, values, na_label=na_label, **kw)
    inset.set_xlim(x0, x1)
    inset.set_ylim(y0, y1)
    inset.set_aspect("equal", adjustable="box", anchor="NW")
    inset.set_xticks([])
    inset.set_yticks([])
    inset.grid(False)
    for spine in inset.spines.values():
        spine.set_visible(True)
        spine.set_color(INSET_EDGE_COLOR)
        spine.set_linewidth(INSET_EDGE_WIDTH)
    inset.set_title(which.title, fontsize=LEGEND_PT, fontweight="normal", color=style.TEXT, loc="left", pad=2)
    ax.indicate_inset(
        (x0, y0, x1 - x0, y1 - y0), inset, edgecolor=INSET_EDGE_COLOR, linewidth=INSET_EDGE_WIDTH, alpha=1.0
    )
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    return inset


def absent_regions(frame: gpd.GeoDataFrame) -> list[int]:
    """Коды регионов кадра, где ни одного МО панели."""
    has_panel = frame.groupby("region_code")["in_panel"].any()
    return sorted(int(c) for c in has_panel.index[~has_panel])


def annotate_absent(
    ax: Axes, frame: gpd.GeoDataFrame, text: str, xytext: tuple[float, float] = NOTE_XY
) -> Any:
    """Одна выноска о выпавших регионах: текст в пустом углу карты (доли осей) и стрелка к ним.

    Подписи каждого региона на карте России налезают друг на друга: восемь выпавших регионов лежат рядом
    на юго-западе. Возвращает аннотацию или None, если выпавших регионов нет.
    """
    codes = absent_regions(frame)
    if not codes:
        return None
    part = frame.loc[frame["region_code"].astype(int).isin(codes)]
    point = shapely.union_all(shapely.make_valid(part.geometry.values)).representative_point()
    return ax.annotate(
        text,
        (point.x, point.y),
        xytext=xytext,
        textcoords="axes fraction",
        fontsize=NOTE_PT,
        color=style.TEXT,
        ha="left",
        va="bottom",
        arrowprops={"arrowstyle": "-", "color": style.TEXT2, "linewidth": 0.6, "shrinkA": 2, "shrinkB": 0},
        path_effects=[patheffects.withStroke(linewidth=2, foreground="white")],
    )


def _fmt_ticks(fmt: Callable[[float], str]) -> FuncFormatter:
    return FuncFormatter(lambda v, pos=None: fmt(v))


_LEGEND_STYLE: dict[str, Any] = {
    "fontsize": LEGEND_PT,
    "title_fontsize": LEGEND_PT,
    "alignment": "left",
    "handlelength": 1.2,
    "handleheight": 1.0,
    "columnspacing": 1.2,
}


def _class_legend(ax: Axes, handles: list[Patch], title: str) -> None:
    """Легенда классов под картой, слева, в несколько колонок: легенда осей за их нижним краем, макет
    constrained оставляет ей место и не пускает на строку источника."""
    ax.legend(
        handles=handles,
        title=title or None,
        loc="upper left",
        bbox_to_anchor=(0.0, 0.0),
        ncols=min(len(handles), LEGEND_COLS),
        borderaxespad=0.0,
        **_LEGEND_STYLE,
    )


def russia_map(
    values: pd.Series,
    geo: gpd.GeoDataFrame,
    year: int,
    *,
    kind: str = "quantile",
    k: int = 7,
    cmap: str | Colormap | Mapping[Any, str] | None = None,
    legend_title: str = "",
    labels: dict[int, str] | None = None,
    fmt: Callable[[float], str] = style.fmt_num,
    center: float | None = None,
    insets: Sequence[Inset] = (),
    na_label: str = NA_LABEL,
    absent_note: str | None = None,
) -> tuple[Figure, Axes]:
    """Карта России: МО года ``year`` по ``values`` (индекс — ``territory_id``).

    ``kind``: ``quantile`` — ``k`` квантильных классов с подписями ``fmt`` (``labels`` заменяет подпись класса
    по номеру); ``diverging`` — шкала ``DIV_CMAP`` с центром ``center`` (по умолчанию медиана);
    ``continuous`` — непрерывная ``SEQ_CMAP``; ``categorical`` — ``cmap`` задаёт значение → цвет.
    МО вне панели — штриховка «нет данных СберИндекса», МО панели без значения — серые ``na_label``.
    ``insets`` — врезки (``insets_from_config``); ``absent_note`` — текст выноски о выпавших регионах.
    Легенда — ``ax.get_legend()``: классы — под картой, у непрерывной шкалы — пункты «нет данных» над ней.
    """
    frame = map_frame(geo, year)
    fig, ax = style.new_figure("map")
    values = pd.Series(values) if kind == "categorical" else pd.Series(values, dtype="float64")
    kw: dict[str, Any] = {}
    scale: ScalarMappable | None = None
    if kind == "quantile":
        edges, bin_labels = quantile_bins(values, k, fmt)
        if labels:
            bin_labels = [labels.get(i, lab) for i, lab in enumerate(bin_labels)]
        kw = {"bins": (edges, bin_labels), "cmap": cmap or style.SEQ_CMAP}
    elif kind == "categorical":
        if not isinstance(cmap, Mapping):
            raise ValueError("categorical: cmap — словарь «значение → цвет»")
        kw = {"categories": cmap}
    elif kind in ("diverging", "continuous"):
        v = values.dropna()
        if kind == "diverging":
            mid = float(v.median()) if center is None else float(center)
            eps = np.finfo(float).eps * max(1.0, abs(mid))
            norm: Normalize = TwoSlopeNorm(
                vcenter=mid, vmin=min(float(v.min()), mid - eps), vmax=max(float(v.max()), mid + eps)
            )
            cm = cmap or style.DIV_CMAP
        else:
            norm, cm = Normalize(vmin=float(v.min()), vmax=float(v.max())), cmap or style.SEQ_CMAP
        kw = {"cmap": cm, "norm": norm}
        scale = ScalarMappable(norm=norm, cmap=cm)
    else:
        raise ValueError(f"неизвестный вид карты {kind!r}")
    handles = _draw_layers(ax, frame, values, na_label=na_label, **kw)
    x0, y0, x1, y1 = _padded_bounds(frame, MAP_PAD)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_axis_off()
    if scale is not None:
        cax = ax.inset_axes(COLORBAR_RECT)
        cbar = fig.colorbar(scale, cax=cax, orientation="horizontal")
        cbar.outline.set_visible(False)
        cbar.ax.tick_params(labelsize=LEGEND_PT, length=2)
        if (
            kind == "diverging"
        ):  # края и центр: у двухсторонней нормы равномерные деления вводят в заблуждение
            cbar.set_ticks([norm.vmin, norm.vcenter, norm.vmax])
        cbar.ax.xaxis.set_major_formatter(
            style.RuFormatter("num") if fmt is style.fmt_num else _fmt_ticks(fmt)
        )
        if legend_title:
            cbar.ax.set_title(legend_title, fontsize=LEGEND_PT, fontweight="normal", loc="left", pad=3)
        if handles:
            ax.legend(
                handles=handles,
                loc="lower left",
                bbox_to_anchor=COLORBAR_LEGEND_ANCHOR,
                borderaxespad=0.0,
                **_LEGEND_STYLE,
            )
    elif handles:
        _class_legend(ax, handles, legend_title)
    for inset, rect in zip(insets, INSET_RECTS, strict=False):
        add_inset(fig, ax, frame, values, inset, rect, na_label=na_label, **kw)
    if absent_note:
        annotate_absent(ax, frame, absent_note)
    return fig, ax
