"""Рисунки этапа dynamics (D01–D05): заголовок — вывод, который проверяется числами при сборке
(у D03 — описание рисунка).

D01 — изменение 2023 → 2024 против шума по трём способам; D02 — размеры типов по 13 окнам; D03 — матрица
переходов (все и надёжные); D04 — проверка сюжета: какие части корзины сильнее меняются у перешедших;
D05 — карта надёжных переходов.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from munnet import style
from munnet.clustering.figures import TYPE_COLORS
from munnet.contracts import PARTS
from munnet.dynamics.params import EVOLUTIONARY, FIXED, MAIN
from munnet.eda.base import write_csv

FIG_SUBDIR = "figures"
SRC_BOTH = style.join_sources(style.SOURCE_SBER, style.SOURCE_ROSSTAT)
APPROACH_LABELS: dict[str, str] = {
    MAIN: "Перефит и сопоставление (основной)",
    FIXED: "Фиксированные типы",
    EVOLUTIONARY: "Эволюционная кластеризация",
}
FEATURE_LABELS: dict[str, str] = {
    **{f"clr_rel_{q}": f"Корзина: {style.LABELS[q].lower()}" for q in PARTS},
    "log_level_rel": "Уровень трат",
}
PART_GENITIVE: dict[str, str] = {  # «доля …»
    "food": "продовольствия",
    "marketplace": "маркетплейсов",
    "transport": "транспорта",
    "health": "здоровья",
    "cafe": "общепита",
    "other": "прочего",
}
STATUS_LABELS: dict[str, str] = {
    "within_noise": "смена в пределах шума",
    "no_change": "без смены",
}


@dataclass(frozen=True)
class FigureInfo:
    fid: str
    slug: str
    title: str
    subtitle: str
    alt: str
    png: Path
    data_csv: Path
    source: str = SRC_BOTH


def _save(fig, out: Path, fid: str, slug: str, data: pd.DataFrame, **info) -> FigureInfo:
    stem = out / FIG_SUBDIR / f"{fid}_{slug}"
    style.save(fig, stem, formats=("png",))
    csv = out / FIG_SUBDIR / f"{fid}_{slug}.csv"
    write_csv(data, csv)
    return FigureInfo(fid=fid, slug=slug, png=stem.with_suffix(".png"), data_csv=csv, **info)


def pct(x: float) -> str:
    return style.fmt_pct(x, 1)


def fig_change(out: Path, changes, n_boot: int) -> FigureInfo:
    rows = []
    for name in (MAIN, FIXED, EVOLUTIONARY):
        b = changes[name].boot
        rows.append(
            {"approach": name, "what": "change", "value": b.change, "lo": b.change_lo, "hi": b.change_hi}
        )
        rows.append({"approach": name, "what": "noise", "value": b.noise, "lo": b.noise_lo, "hi": b.noise_hi})
    data = pd.DataFrame(rows)
    fig, ax = style.new_figure("full")
    y = np.arange(3)[::-1]
    for what, color, lab, dy in (
        ("change", style.PALETTE["marketplace"], "сменили тип 2023 → 2024", 0.12),
        ("noise", style.CONTEXT, "«смены» внутри одного года (шум)", -0.12),
    ):
        d = data.loc[data["what"] == what]
        ax.errorbar(
            d["value"] * 100,
            y + dy,
            xerr=[(d["value"] - d["lo"]) * 100, (d["hi"] - d["value"]) * 100],
            fmt="o",
            color=color if what == "change" else style.TEXT2,
            ecolor=color if what == "change" else style.TEXT2,
            capsize=3,
            label=lab,
        )
    ax.set_yticks(y, [APPROACH_LABELS[a] for a in (MAIN, FIXED, EVOLUTIONARY)])
    ax.set_xlim(0, None)
    ax.set_xlabel("Доля узлов, %")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.35, -0.16), ncols=2)
    exceeds_all = all(changes[a].boot.exceeds for a in (MAIN, FIXED, EVOLUTIONARY))
    m = changes[MAIN].boot
    title = (
        f"Типы сменили {pct(m.change)} узлов — больше шума ({pct(m.noise)}) при любом способе отслеживания"
        if exceeds_all
        else f"Типы сменили {pct(m.change)} узлов при шуме {pct(m.noise)}"
    )
    subtitle = (
        f"Доля узлов сети, сменивших тип между окнами 2023 и 2024 годов, и доля «смен» между разбиениями "
        f"нечётных и чётных месяцев одного года (среднее двух лет); отрезки — 95% интервал бутстрапа узлов, "
        f"{n_boot} повторов"
    )
    return _save(
        fig_finish(fig, title, subtitle),
        out,
        "D01",
        "change_noise",
        data,
        title=title,
        subtitle=subtitle,
        alt="Точки с интервалами: доля сменивших тип против шума по трём способам",
    )


def fig_finish(fig, title: str, subtitle: str, source: str = SRC_BOTH):
    style.finish(fig, title, subtitle, source)
    return fig


def fig_sizes(out: Path, windows, track, k: int) -> FigureInfo:
    sizes = np.array([[int((lab == j).sum()) for j in range(k)] for lab in track.rolling])
    data = pd.DataFrame(sizes, columns=[f"type_{j + 1}" for j in range(k)])
    data.insert(0, "window", windows)
    fig, ax = style.new_figure("full")
    x = np.arange(len(windows))
    for j in range(k):
        ax.plot(x, sizes[:, j], marker="o", ms=3, color=TYPE_COLORS[j], label=f"Тип {j + 1}")
    ticks = [0, len(windows) // 2, len(windows) - 1]
    ax.set_xticks(ticks, [windows[i].replace("…", "–") for i in ticks])
    ax.set_ylabel("Узлов в типе")
    ax.set_ylim(0, float(sizes.max()) * 1.3)
    ax.legend(loc="upper right", ncols=k)
    big = {int(np.argmax(s)) for s in sizes}
    title = (
        f"Тип {big.pop() + 1} остаётся крупнейшим во всех {len(windows)} окнах"
        if len(big) == 1
        else f"Размеры типов по {len(windows)} скользящим окнам"
    )
    subtitle = (
        "Основной способ: в каждом 12-месячном окне заново тот же метод и K, типы соседних окон сопоставлены "
        "венгерским алгоритмом по Жаккару узлов; окно подписано первым и последним месяцем"
    )
    return _save(
        fig_finish(fig, title, subtitle),
        out,
        "D02",
        "sizes",
        data,
        title=title,
        subtitle=subtitle,
        alt="Линии: число узлов в каждом типе по 13 окнам",
    )


def fig_transitions(out: Path, change, k: int) -> FigureInfo:
    M_all, M_rel = change.matrix, change.window_reliable_matrix  # обе части — в типах 12-месячных окон
    rows = [
        {"from": a + 1, "to": b + 1, "all": int(M_all[a, b]), "reliable": int(M_rel[a, b])}
        for a in range(k)
        for b in range(k)
    ]
    data = pd.DataFrame(rows)
    fig, ax = style.new_figure("half")
    fig.set_size_inches(6.6, 4.8)
    show = M_rel.astype(float).copy()
    np.fill_diagonal(show, np.nan)
    show[show == 0] = np.nan  # нулевые клетки — белые: цвет только у переходов, которые есть
    vmax = max(1.0, float(np.nanmax(show)))
    ax.imshow(np.ma.masked_invalid(show), cmap="Blues", vmin=-0.15 * vmax, vmax=vmax)  # светлое — мало
    for a in range(k):
        for b in range(k):
            if a == b:
                txt, col = f"остались\n{style.fmt_num(M_all[a, b])}", style.TEXT2
            else:
                txt = f"{style.fmt_num(M_rel[a, b])}\nиз {style.fmt_num(M_all[a, b])}"
                col = "white" if M_rel[a, b] > 0.5 * max(1, np.nanmax(show)) else style.TEXT
            ax.text(b, a, txt, ha="center", va="center", fontsize=8, color=col)
    ax.set_xticks(range(k), [f"Тип {j + 1}" for j in range(k)])
    ax.set_yticks(range(k), [f"Тип {j + 1}" for j in range(k)])
    ax.set_xlabel("Тип в 2024 году")
    ax.set_ylabel("Тип в 2023 году")
    ax.grid(False)
    # заголовок — описание, без «главного» потока: смена типа за год не подтверждена (T3, совет 06.10.2026)
    title = "Надёжные смены типа между окнами 2023 и 2024 годов"
    subtitle = (
        "Типы — по 12-месячным окнам 2023 и 2024 годов; в клетке — надёжные переходы (тип совпал на нечётных "
        "и чётных месяцах каждого года, переход тот же) из всех, на диагонали — сохранившие тип окна; ещё "
        f"{change.n_reliable_window_same} узлов надёжно сменили тип по половинам лет, но не по окнам"
    )
    return _save(
        fig_finish(fig, title, subtitle),
        out,
        "D03",
        "transitions",
        data,
        title=title,
        subtitle=subtitle,
        alt="Матрица 4 на 4: число надёжных и всех переходов между типами",
    )


def fig_driver(out: Path, driver) -> FigureInfo:
    t = driver.table.copy()
    t["label"] = t["feature"].map(FEATURE_LABELS)
    t = t.iloc[::-1].reset_index(drop=True)
    fig, ax = style.new_figure("full")
    colors = [
        style.PALETTE["marketplace"] if f == f"clr_rel_{driver.part}" else style.CONTEXT for f in t["feature"]
    ]
    ax.barh(np.arange(len(t)), t["ratio"], color=colors)
    ax.axvline(1.0, color=style.TEXT2, lw=0.8)
    for i, r in t.iterrows():
        ax.text(r["ratio"] + 0.03, i, style.fmt_num(r["ratio"], 2), va="center", fontsize=8, color=style.TEXT)
    ax.set_yticks(np.arange(len(t)), t["label"])
    ax.set_xlabel("Медиана |изменения| у перешедших / у сохранивших тип")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    top, part = PART_GENITIVE[driver.top_part], PART_GENITIVE[driver.part]
    if driver.expectation_met:
        title = f"У перешедших узлов сильнее всего изменилась доля {part}"
    elif driver.significant:
        title = f"Доля {part} у перешедших узлов меняется сильнее, но сильнее всего — доля {top}"
    else:
        title = f"Доля {part} у перешедших узлов меняется не сильнее; сильнее всего — доля {top}"
    subtitle = (
        f"Изменение 2023 → 2024 (корзина — CLR относительно группы региона, уровень — логарифм относительно "
        f"региона); узлы с надёжным переходом ({style.fmt_num(driver.n_transition)}) против узлов без смены "
        f"типа ({style.fmt_num(driver.n_same)})"
    )
    return _save(
        fig_finish(fig, title, subtitle, style.SOURCE_SBER),
        out,
        "D04",
        "driver",
        driver.table,
        title=title,
        subtitle=subtitle,
        alt="Столбцы: отношение медиан изменения по шести частям корзины и уровню трат",
        source=style.SOURCE_SBER,
    )


def flow_categories(nodes: pd.DataFrame, min_n: int) -> pd.Series:
    """Категория узла для карты: надёжный переход «a → b» (крупные — свои цвета), прочие надёжные, шум,
    без смены."""
    flow = nodes["half_type_2023"].astype(str) + " → " + nodes["half_type_2024"].astype(str)
    rel = nodes["reliable"].astype(bool)
    counts = flow[rel].value_counts()
    big = set(counts[counts >= min_n].index)
    cat = np.where(
        rel,
        np.where(flow.isin(big), "Тип " + flow.str.replace(" → ", " → тип "), "другие надёжные переходы"),
        nodes["status"].map(STATUS_LABELS),
    )
    return pd.Series(cat, index=nodes.index)


def fig_map(out: Path, cfg, nodes: pd.DataFrame, ns, min_n: int) -> FigureInfo:
    from munnet import maps

    geo = maps.load_geometry(Path(cfg["paths"]["processed"]))
    cat = flow_categories(nodes, min_n)
    node_cat = pd.Series(cat.to_numpy(), index=nodes["territory_id"].to_numpy())
    mem = ns.members.loc[ns.members["role"].astype(str) != "excluded", ["territory_id", "node_id"]]
    mo = mem.assign(category=mem["node_id"].map(node_cat)).dropna(subset=["category"])
    values = pd.Series(mo["category"].to_numpy(), index=mo["territory_id"].astype("int32").to_numpy())
    flows = sorted(c for c in cat.unique() if c.startswith("Тип "))
    palette = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]
    colors = {c: palette[i % len(palette)] for i, c in enumerate(flows)}
    colors["другие надёжные переходы"] = "#882255"
    colors["смена в пределах шума"] = "#BDBDBD"
    colors["без смены"] = "#F2F2F2"
    colors = {c: v for c, v in colors.items() if c in set(cat)}
    rel = nodes["reliable"].astype(bool)
    n_reg, n_all = nodes.loc[rel, "region"].nunique(), nodes["region"].nunique()
    title = (
        f"Надёжно сменили тип {pct(rel.mean())} узлов сети — в {style.fmt_num(n_reg)} регионах "
        f"из {style.fmt_num(n_all)}"
    )
    subtitle = (
        "Статус перехода узла сети у каждого МО; надёжный переход — тип совпал на нечётных и чётных месяцах "
        "каждого года, а типы двух лет разные; районы Москвы и Петербурга — статус своего узла-города; "
        "серым — МО вне сети"
    )
    fig, _ = maps.russia_map(
        values,
        geo,
        2023,
        kind="categorical",
        cmap=colors,
        legend_title="2023 → 2024",
        insets=maps.insets_from_config(cfg),
        na_label="вне сети",
    )
    return _save(
        fig_finish(fig, title, subtitle),
        out,
        "D05",
        "map",
        mo.rename(columns={"node_id": "node"}),
        title=title,
        subtitle=subtitle,
        alt="Карта России: МО окрашены по переходу типа своего узла",
    )


def make_figures(cfg, out: Path, k, windows, tracks, changes, driver, nodes, ns) -> list[FigureInfo]:
    from munnet.dynamics.params import DynParams

    p = DynParams.from_config(cfg)
    return [
        fig_change(out, changes, p.bootstrap),
        fig_sizes(out, windows, tracks[MAIN], k),
        fig_transitions(out, changes[MAIN], k),
        fig_driver(out, driver),
        fig_map(out, cfg, nodes, ns, p.map_flow_min),
    ]
