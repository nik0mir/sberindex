"""Рисунки этапа evaluate: индексы по K, z по K, согласие метрик, объединяемость итоговых типов,
«признаки против связей».

Каждый рисунок: заголовок-вывод по числам (если вывод неверен — нейтральная формулировка), подзаголовок,
строка источника (``style.finish``), PNG и SVG в ``outputs/evaluate/figures/`` и CSV с данными рядом.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

from munnet import style
from munnet.icvi_stage import METRICS, write_csv

FIG_SUBDIR = "figures"
METRIC_LABELS: dict[str, str] = {
    "sw": "SW (силуэт)",
    "ch": "CH",
    "s_dbw": "S_Dbw",
    "avi": "AVI",
    "avu": "AVU",
    "mq": "MQ (модульность)",
    "anui": "ANUI",
}
BETTER_TEXT: dict[str, str] = {"max": "больше — лучше", "min": "меньше — лучше"}
FAMILY_MARKERS: dict[str, str] = {
    "features": "o",
    "graph": "s",
    "attributed": "D",
    "fusion": "^",
    "other": "x",
}


def _method_style():
    """Подписи и цвета методов — те же, что в отчёте этапа cluster."""
    from munnet.clustering.figures import FAMILY_LABELS, METHOD_COLORS, METHOD_LABELS

    return METHOD_LABELS, METHOD_COLORS, FAMILY_LABELS


def method_label(method: str) -> str:
    return _method_style()[0].get(method, method)


@dataclass(frozen=True)
class FigureInfo:
    fid: str
    slug: str
    title: str
    subtitle: str
    alt: str
    png: Path
    data_csv: Path
    check: str


def _save(fig, out: Path, fid: str, slug: str, data: pd.DataFrame, **info) -> FigureInfo:
    stem = out / FIG_SUBDIR / f"{fid}_{slug}"
    style.save(fig, stem)
    csv = write_csv(data, out / FIG_SUBDIR / f"{fid}_{slug}.csv")
    return FigureInfo(fid=fid, slug=slug, png=stem.with_suffix(".png"), data_csv=csv, **info)


def _curves(
    out: Path, long: pd.DataFrame, cands: pd.DataFrame, col: str, fid: str, slug: str, facts
) -> FigureInfo:
    labels, colors, _ = _method_style()
    c = cands.set_index("candidate")
    d = long.loc[long["metric"].isin(METRICS)].merge(
        cands[["candidate", "family", "k_eff", "is_method_winner", "is_final", "feasible"]], on="candidate"
    )
    with style.use():
        fig, first = style.new_figure("tall")
        first.remove()
        gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.2])
        axes = np.array([[fig.add_subplot(gs[r, c]) for c in range(3)] for r in range(2)])
        legend_ax = fig.add_subplot(gs[2, :])
        legend_ax.set_axis_off()
        for ax, name in zip(axes.ravel(), METRICS, strict=True):
            m = d.loc[d["metric"] == name]
            if col == "value":
                band = m.groupby("k_eff")["baseline_mean"].agg(["min", "max"])
                ax.fill_between(
                    band.index, band["min"], band["max"], color=style.GRID, lw=0, label="случайный базис"
                )
            else:
                ax.axhline(0, color=style.TEXT2, lw=0.8)
            for method, g in m.groupby("method", sort=False):
                g = g.sort_values("k_eff")
                color = colors.get(method, style.ACCENT)
                fam = c.loc[g["candidate"].iloc[0], "family"]
                ax.plot(
                    g["k_eff"],
                    g[col],
                    color=color,
                    lw=1.0,
                    marker=FAMILY_MARKERS.get(fam, "o"),
                    ms=2.5,
                    label=labels.get(method, method),
                )
                if col == "value":
                    ax.vlines(g["k_eff"], g["ci_low"], g["ci_high"], color=color, lw=0.6, alpha=0.6)
                w = g.loc[g["is_method_winner"].astype(bool)]
                ax.scatter(w["k_eff"], w[col], s=36, facecolor="white", edgecolor=color, lw=1.4, zorder=4)
                f = g.loc[g["is_final"].astype(bool)]
                ax.scatter(f["k_eff"], f[col], s=90, marker="*", color=style.ACCENT, zorder=5)
            short = METRIC_LABELS[name].split(" ")[0]
            ax.set_title(f"{short}: {BETTER_TEXT[facts['better'][name]]}", fontsize=8)
            ax.set_xlabel("K")
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        handles, names = axes[1, 1].get_legend_handles_labels()
        legend_ax.legend(handles, names, fontsize=6, ncol=5, loc="center", frameon=False)
        if col == "value":
            title = facts["title_curves"]
            subtitle = (
                "Значение индекса на всех узлах и 95% интервал по подвыборкам узлов; серая полоса — "
                "случайный базис при тех же размерах кластеров; кружок — K, выбранный методом, звезда — итог"
            )
        else:
            title = facts["title_z"]
            subtitle = (
                "z-оценка против случайного базиса, знак «больше — лучше»; разные K сравнивают по z вместе "
                "с поправкой на случайность и рангом внутри K; кружок — K, выбранный методом, звезда — итог"
            )
        style.finish(fig, title, subtitle, style.SOURCE_SBER)
    data = d[
        [
            "candidate",
            "method",
            "family",
            "k_eff",
            "metric",
            "value",
            "ci_low",
            "ci_high",
            "baseline_mean",
            "z",
            "is_method_winner",
            "is_final",
        ]
    ]
    alt = "Шесть панелей, по одной на индекс; по оси X — число типов K, линии — методы; " + (
        "серая полоса — случайный базис" if col == "value" else "нулевая линия — уровень случайных меток"
    )
    return _save(
        fig,
        out,
        fid,
        slug,
        data,
        title=title,
        subtitle=subtitle,
        alt=alt,
        check="заголовок выбран по фактам icvi (title_curves, title_z)",
    )


def fig_curves(out, long, cands, facts) -> FigureInfo:
    """I01: сырые индексы по K с базисом и интервалами."""
    return _curves(out, long, cands, "value", "I01", "by_k", facts)


def fig_z(out, long, cands, facts) -> FigureInfo:
    """I02: z-оценки по K."""
    return _curves(out, long, cands, "z", "I02", "z_by_k", facts)


def fig_agreement(out: Path, tau_z: pd.DataFrame, tau_v: pd.DataFrame, facts) -> FigureInfo:
    """I03: τ Кендалла ранжирований кандидатов по z и по сырым значениям."""
    names = [METRIC_LABELS[m].split(" ")[0] for m in METRICS]
    with style.use():
        fig, axes = style.new_figure("full", 1, 2)
        for ax, tau, head in zip(axes, (tau_z, tau_v), ("по z-оценкам", "по сырым значениям"), strict=True):
            mat = tau.loc[list(METRICS), list(METRICS)].to_numpy(dtype=float)
            ax.imshow(mat, cmap=style.DIV_CMAP, vmin=-1, vmax=1)
            for i in range(len(METRICS)):
                for j in range(len(METRICS)):
                    v = mat[i, j]
                    txt = style.NA_TEXT if not np.isfinite(v) else style.fmt_num(v, 2)
                    ax.text(
                        j,
                        i,
                        txt,
                        ha="center",
                        va="center",
                        fontsize=6.5,
                        color="white" if np.isfinite(v) and abs(v) > 0.6 else style.TEXT,
                    )
            ax.set_xticks(range(len(METRICS)), names, fontsize=7)
            ax.set_yticks(range(len(METRICS)), names, fontsize=7)
            ax.set_title(head, fontsize=8)
            ax.grid(False)
        title = facts["title_agreement"]
        subtitle = (
            "τ Кендалла между ранжированиями допустимых кандидатов «метод × K» (лучше — выше); "
            "1 — метрики упорядочивают кандидатов одинаково, −1 — наоборот"
        )
        style.finish(fig, title, subtitle, style.SOURCE_SBER)
    data = (
        pd.concat([tau_z.assign(basis="z"), tau_v.assign(basis="value")]).rename_axis("metric").reset_index()
    )
    alt = "Две тепловые карты 6 × 6: согласие метрик по z-оценкам и по сырым значениям"
    return _save(
        fig,
        out,
        "I03",
        "agreement",
        data,
        title=title,
        subtitle=subtitle,
        alt=alt,
        check="заголовок выбран по τ AVI–MQ и среднему τ между пространствами",
    )


def fig_unifiability(out: Path, fu: pd.DataFrame, facts) -> FigureInfo:
    """I04: матрица объединяемости U_kl и изолируемость типов итогового разбиения."""
    types = fu["type"].to_numpy()
    mat = fu[[f"type_{t}" for t in types]].to_numpy(dtype=float)
    with style.use():
        fig, axes = style.new_figure("full", 1, 2, width_ratios=[1.3, 1])
        ax = axes[0]
        ax.imshow(np.where(np.isfinite(mat), mat, np.nan), cmap=style.SEQ_CMAP, vmin=0, vmax=1)
        for i in range(len(types)):
            for j in range(len(types)):
                if np.isfinite(mat[i, j]):
                    ax.text(
                        j,
                        i,
                        style.fmt_num(mat[i, j], 2),
                        ha="center",
                        va="center",
                        fontsize=6,
                        color="white" if mat[i, j] < 0.5 else style.TEXT,
                    )
        ax.set_xticks(range(len(types)), [str(t) for t in types], fontsize=7)
        ax.set_yticks(range(len(types)), [str(t) for t in types], fontsize=7)
        ax.set_xlabel("Тип")
        ax.set_ylabel("Тип")
        ax.set_title("Объединяемость пары U_kl (меньше — лучше)", fontsize=8)
        ax.grid(False)
        ax = axes[1]
        ax.barh([str(t) for t in types], fu["isolability"], color=style.PALETTE["marketplace"])
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.set_xlabel("Изолируемость типа (больше — лучше)")
        ax.set_ylabel("Тип")
        title = facts["title_unif"]
        subtitle = (
            "U_kl — доля внешних связей пары типов, которые они отдают друг другу; изолируемость — "
            "доля связей типа внутри него (Biswas, Biswas, 2017, в пересказе Shalileh и др., 2025)"
        )
        style.finish(fig, title, subtitle, style.SOURCE_SBER)
    alt = "Тепловая карта объединяемости пар типов итогового разбиения и столбцы изолируемости каждого типа"
    return _save(
        fig,
        out,
        "I04",
        "unifiability",
        fu,
        title=title,
        subtitle=subtitle,
        alt=alt,
        check="заголовок называет пару типов с наибольшей U_kl",
    )


def fig_spaces(out: Path, q: pd.DataFrame, facts) -> FigureInfo:
    """I05: качество в признаках против качества в сети (средний перцентильный ранг z)."""
    labels, colors, fam_labels = _method_style()
    with style.use():
        fig, ax = style.new_figure("full")
        for fam, g in q.groupby("family"):
            ax.scatter(
                g["q_features"],
                g["q_graph"],
                s=16,
                marker=FAMILY_MARKERS.get(fam, "o"),
                color=[colors.get(m, style.ACCENT) for m in g["method"]],
                alpha=np.where(g["feasible"].astype(bool), 0.9, 0.3),
                label=fam_labels.get(fam, fam),
            )
        ax.plot([0, 1], [1, 0], color=style.GRID, lw=0.8)
        sel = q.loc[q["is_method_winner"].astype(bool) | q["is_final"].astype(bool)].copy()
        sel["score"] = sel["q_features"] + sel["q_graph"] + 10 * sel["is_final"].astype(bool)
        sel = sel.sort_values("score", ascending=False).head(6).dropna(subset=["q_features", "q_graph"])
        style.label_points(
            ax,
            sel["q_features"],
            sel["q_graph"],
            [f"{labels.get(m, m)}, K = {k}" for m, k in zip(sel["method"], sel["k"], strict=True)],
            max_labels=7,
        )
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(-0.03, 1.03)
        ax.set_xlabel("Качество в признаках X (ранг z по SW, CH, S_Dbw)")
        ax.set_ylabel("Качество в сети G (AVI, AVU, MQ)")
        ax.legend(fontsize=7, loc="lower left")
        title = facts["title_spaces"]
        subtitle = (
            "Кандидаты «метод × K», ранг среди допустимых (0 — худший, 1 — лучший), бледные — недопустимые; "
            "подписаны итог и лучшие победители методов"
        )
        style.finish(fig, title, subtitle, style.SOURCE_SBER)
    alt = "Точечная диаграмма: по оси X — качество кандидата в признаках, по оси Y — в сети, цвет — метод"
    return _save(
        fig,
        out,
        "I05",
        "spaces",
        q,
        title=title,
        subtitle=subtitle,
        alt=alt,
        check="заголовок выбран по среднему рангу семейств в своём и чужом пространстве",
    )


def make_all(out: Path, long, cands, tau_z, tau_v, fu, q, facts) -> list[FigureInfo]:
    figs = [
        fig_curves(out, long, cands, facts),
        fig_z(out, long, cands, facts),
        fig_agreement(out, tau_z, tau_v, facts),
    ]
    if len(fu):
        figs.append(fig_unifiability(out, fu, facts))
    figs.append(fig_spaces(out, q, facts))
    return figs
