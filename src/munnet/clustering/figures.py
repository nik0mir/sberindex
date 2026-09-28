"""Рисунки этапа cluster: синтетика «когда помогает граф», метод × критерий, критерии и ICVI по K, кривая
гибрида по α, согласие методов, профили и карта итоговых типов.

Каждый рисунок: заголовок-вывод, подзаголовок с числами, строка источника (``style.finish``), PNG и SVG
в ``outputs/cluster/figures/`` и CSV с данными рисунка рядом. Заголовок собирается по числам: если вывод
перестал быть верным, выбирается нейтральная формулировка (``check`` в записи рисунка говорит, что проверено).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from munnet import style

FIG_SUBDIR = "figures"

METHOD_LABELS: dict[str, str] = {
    "kmeans": "K-means",
    "ward": "Уорд",
    "gmm": "Гауссова смесь",
    "hdbscan": "HDBSCAN",
    "leiden": "Leiden",
    "louvain": "Louvain",
    "spectral": "Спектральная",
    "shalileh_mirkin": "KEFRiN",
    "hybrid": "Гибрид",
    "kmeans_joint": "K-means корзина ⊕ X",
}
FAMILY_LABELS: dict[str, str] = {
    "features": "по признакам X",
    "graph": "по графу G",
    "attributed": "на сети с признаками",
    "fusion": "базовая линия",
}
METHOD_COLORS: dict[str, str] = {
    "kmeans": "#0072B2",
    "ward": "#56B4E9",
    "gmm": "#2B5C8A",
    "hdbscan": "#8FB9D9",
    "leiden": "#D55E00",
    "louvain": "#E69F00",
    "spectral": "#B26B00",
    "shalileh_mirkin": "#882255",
    "hybrid": "#CC79A7",
    "kmeans_joint": "#595959",
}
# Типы итоговой типологии: палитра Окабе — Ито без жёлтого (плохо читается на белом) и чёрного.
TYPE_COLORS: tuple[str, ...] = (
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#882255",
    "#595959",
    "#44AA99",
    "#999933",
    "#AA4499",
    "#332288",
)
FEATURE_LABELS: dict[str, str] = {
    "log_level_rel": "Уровень трат",
    "log_wage_rel": "Зарплата",
    "urban_share_rel": "Доля горожан",
    "age_old_share_rel": "Доля старших",
    "log_pop_rel": "Население",
    "market_access_rel": "Доступность рынков",
    "emp_sh_primary": "Занятость: сельск. хоз. и добыча",
    "emp_sh_industry": "Занятость: промышленность",
    "emp_sh_trade_transport": "Занятость: стройка, торговля, транспорт",
    "emp_sh_market_services": "Занятость: рыночные услуги",
    "emp_sh_public": "Занятость: бюджетный сектор",
    "clr_rel_food": "Корзина: продовольствие",
    "clr_rel_marketplace": "Корзина: маркетплейсы",
    "clr_rel_transport": "Корзина: транспорт",
    "clr_rel_health": "Корзина: здоровье",
    "clr_rel_cafe": "Корзина: общепит",
    "clr_rel_other": "Корзина: прочее",
}
METRIC_LABELS: dict[str, str] = {
    "sw": "SW (силуэт)",
    "ch": "CH (Калински — Харабаш)",
    "s_dbw": "S_Dbw",
    "avi": "AVI",
    "avu": "AVU",
    "mq": "MQ (модульность)",
}
CRITERION_LABELS: dict[str, str] = {
    "quality_features": "Качество в X",
    "quality_graph": "Качество на G",
    "stability": "Устойчивость",
    "interpretability": "Объяснимость",
}


def mlabel(method: str) -> str:
    return METHOD_LABELS.get(method, method)


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
    source: str = style.SOURCE_SBER


def _save(fig, out: Path, fid: str, slug: str, data: pd.DataFrame, **info) -> FigureInfo:
    from munnet.clustering.stage import write_csv

    stem = out / FIG_SUBDIR / f"{fid}_{slug}"
    style.save(fig, stem)
    csv = out / FIG_SUBDIR / f"{fid}_{slug}.csv"
    write_csv(data, csv)
    return FigureInfo(fid=fid, slug=slug, png=stem.with_suffix(".png"), data_csv=csv, **info)


SRC_BOTH = style.join_sources(style.SOURCE_SBER, style.SOURCE_ROSSTAT)
SRC_SYNTH = "синтетические данные (генератор этапа)"


# --- C01: синтетика ------------------------------------------------------------------------------


def fig_synthetic(out: Path, summary: pd.DataFrame, families: dict[str, str], repeats: int) -> FigureInfo:
    """C01: какой метод выигрывает при каком сочетании сигнала в графе и в признаках (основной генератор)."""
    s = summary.loc[summary["design"] == "knn"].copy()
    gs = sorted(s["graph_signal"].unique())
    fs = sorted(s["feature_signal"].unique())
    gain = s.pivot(index="graph_signal", columns="feature_signal", values="graph_gain").reindex(
        index=gs, columns=fs
    )
    win = s.pivot(index="graph_signal", columns="feature_signal", values="winner").reindex(
        index=gs, columns=fs
    )
    wari = s.pivot(index="graph_signal", columns="feature_signal", values="winner_ari").reindex(
        index=gs, columns=fs
    )
    both_top = s["winner"].map(lambda m: families.get(m) in ("attributed", "fusion"))
    mixed = s.loc[(s["graph_signal"] > 0) & (s["feature_signal"] > 0)]
    share_both = float(both_top.loc[mixed.index].mean()) if len(mixed) else float("nan")
    title = (
        "Когда сигнал есть и в графе, и в признаках, выигрывают методы, видящие оба источника"
        if share_both >= 0.5
        else "Какой метод выигрывает при разной силе сигнала в графе и в признаках"
    )
    subtitle = (
        f"Средний ARI с посаженным разбиением, наборов данных на ячейку: {repeats};"
        " слева — победитель ячейки, справа — выигрыш лучшего метода на (G, X) над лучшим "
        f"методом только по X; сигнал есть в обоих источниках — в {style.fmt_pct(share_both, 0)} таких ячеек "
        "побеждает метод, видящий оба"
    )
    with style.use():
        fig, axes = style.new_figure("tall", 1, 2)
        ax = axes[0]
        codes = {m: i for i, m in enumerate(METHOD_COLORS)}
        img = np.vectorize(lambda m: codes.get(m, -1))(win.to_numpy())
        from matplotlib.colors import ListedColormap

        cmap = ListedColormap(list(METHOD_COLORS.values()))
        ax.imshow(img, cmap=cmap, vmin=0, vmax=len(codes) - 1, origin="lower", aspect="auto")
        for i in range(len(gs)):
            for j in range(len(fs)):
                ax.text(
                    j,
                    i,
                    f"{mlabel(win.iloc[i, j])}\n{style.fmt_num(wari.iloc[i, j], 2)}",
                    ha="center",
                    va="center",
                    fontsize=style.POINT_LABEL_PT - 1,
                    color="white",
                )
        ax.set_title("Победитель (ARI)", fontsize=style.LABEL_PT)
        ax = axes[1]
        lim = max(0.05, float(np.nanmax(np.abs(gain.to_numpy()))))
        ax.imshow(gain.to_numpy(), cmap=style.DIV_CMAP, vmin=-lim, vmax=lim, origin="lower", aspect="auto")
        for i in range(len(gs)):
            for j in range(len(fs)):
                ax.text(
                    j,
                    i,
                    style.fmt_num(gain.iloc[i, j], 2, sign=True),
                    ha="center",
                    va="center",
                    fontsize=style.POINT_LABEL_PT,
                )
        ax.set_title("Выигрыш от графа, ΔARI", fontsize=style.LABEL_PT)
        for a in axes:
            a.set_xticks(range(len(fs)), [style.fmt_num(v, 1) for v in fs])
            a.set_yticks(range(len(gs)), [style.fmt_num(v, 1) for v in gs])
            a.set_xlabel("Сигнал в признаках (разнос центров, SD шума)")
        axes[0].set_ylabel("Сигнал в графе (разнос центров корзины, SD шума)")
        style.finish(fig, title, subtitle, SRC_SYNTH)
    alt = (
        "Две тепловые карты: победитель по ячейкам сетки «сигнал в графе × сигнал в признаках» и выигрыш "
        "от графа"
    )
    data = s[
        [
            "graph_signal",
            "feature_signal",
            "winner",
            "winner_ari",
            "best_features",
            "best_graph",
            "best_both",
            "graph_gain",
        ]
    ]
    return _save(
        fig,
        out,
        "C01",
        "synthetic",
        data,
        title=title,
        subtitle=subtitle,
        alt=alt,
        check="в половине и более ячеек с сигналом в обоих источниках побеждает метод на (G, X)",
        source=SRC_SYNTH,
    )


# --- C02: метод × критерий -----------------------------------------------------------------------


def fig_methods(out: Path, table: pd.DataFrame, final_method: str) -> FigureInfo:
    """C02: победители уровня K всех методов по четырём критериям (ранги качества — среди всех методов)."""
    t = table.copy()
    cols = list(CRITERION_LABELS)
    title = "Методы по четырём критериям: ни один не лучший сразу во всём"
    best = {c: t.loc[t[c].idxmax(), "method"] for c in cols if t[c].notna().any()}
    if len(set(best.values())) == 1:
        title = f"{mlabel(next(iter(best.values())))} — лучший по всем четырём критериям"
    subtitle = (
        "Победитель уровня K каждого метода; качество — средний ранг z-оценок ICVI среди победителей "
        "методов с допустимым K "
        "(1 — лучший), устойчивость — средний ARI с бутстрап-разбиениями, объяснимость — каппа дерева "
        "глубины 3; "
        "серым — метод без допустимого K"
    )
    with style.use():
        fig, ax = style.new_figure("tall")
        M = t[cols].to_numpy(dtype=float)
        ax.imshow(M, cmap=style.SEQ_CMAP, vmin=0, vmax=1, aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M[i, j]
                ax.text(
                    j,
                    i,
                    style.NA_TEXT if not np.isfinite(v) else style.fmt_num(v, 2),
                    ha="center",
                    va="center",
                    fontsize=style.POINT_LABEL_PT,
                    color="white" if v < 0.55 else style.TEXT,
                )
        ylabels = [
            f"{mlabel(m)}, K = {int(k)}" + (" ★" if m == final_method else "")
            for m, k in zip(t["method"], t["k"], strict=True)
        ]
        ax.set_yticks(range(len(t)), ylabels)
        ax.set_xticks(range(len(cols)), [CRITERION_LABELS[c] for c in cols])
        ax.tick_params(axis="x", labeltop=True, labelbottom=False)
        style.finish(fig, title, subtitle, SRC_BOTH)
    alt = "Тепловая карта: строки — методы с выбранным K, столбцы — четыре критерия выбора"
    return _save(
        fig,
        out,
        "C02",
        "methods",
        t,
        title=title,
        subtitle=subtitle,
        alt=alt,
        check="заголовок «ни один не лучший во всём» — если лучшие по критериям методы различаются",
        source=SRC_BOTH,
    )


# --- C03: критерии и ICVI по K --------------------------------------------------------------------


def fig_by_k(out: Path, cands: pd.DataFrame, metrics: list[str]) -> FigureInfo:
    """C03: z-оценки шести ICVI по K у каждого метода: признаковые и графовые индексы тянут K в разные
    стороны."""
    c = cands.copy()
    rows = []
    for m, sub in c.groupby("method"):
        for name in metrics:
            z = sub.set_index("k")[f"z_{name}"].dropna()
            if len(z) >= 2:
                rows.append({"method": m, "metric": name, "k_best": int(z.idxmax())})
    kb = pd.DataFrame(rows)
    graph_m = [x for x in metrics if x in ("avi", "avu", "mq")]
    feat_m = [x for x in metrics if x not in graph_m]
    kf = kb.loc[kb["metric"].isin(feat_m), "k_best"].median() if len(kb) else np.nan
    kg = kb.loc[kb["metric"].isin(graph_m), "k_best"].median() if len(kb) else np.nan
    title = (
        "Индексы в признаках тянут к малому числу типов, индексы на графе — к большому"
        if np.isfinite(kf) and np.isfinite(kg) and kf < kg
        else "z-оценки шести индексов качества по числу типов"
    )
    subtitle = (
        "z-оценка против случайного базиса (перестановки меток с теми же размерами), больше — лучше; медиана "
        f"лучшего K по методам: у SW, CH, S_Dbw — {style.fmt_num(kf, 0)}, у AVI, AVU, MQ — "
        f"{style.fmt_num(kg, 0)}; "
        "полые точки — недопустимые разбиения"
    )
    with style.use():
        fig, axes = style.new_figure("tall", 2, 3, sharex=True)
        for ax, name in zip(axes.ravel(), metrics, strict=False):
            for m, sub in c.groupby("method", sort=False):
                sub = sub.sort_values("k")
                col = METHOD_COLORS.get(m, style.TEXT2)
                ax.plot(sub["k"], sub[f"z_{name}"], color=col, linewidth=1.0, label=mlabel(m))
                feas = sub["feasible"].astype(bool)
                ax.scatter(sub.loc[feas, "k"], sub.loc[feas, f"z_{name}"], s=10, color=col, zorder=3)
                ax.scatter(
                    sub.loc[~feas, "k"],
                    sub.loc[~feas, f"z_{name}"],
                    s=10,
                    facecolor="white",
                    edgecolor=col,
                    zorder=3,
                )
            ax.set_title(METRIC_LABELS.get(name, name), fontsize=style.LABEL_PT)
            ax.axhline(0, color=style.GRID, linewidth=0.8)
        for ax in axes[-1]:
            ax.set_xlabel("K")
        axes[0][0].legend(
            loc="upper left",
            bbox_to_anchor=(0.0, -1.55),
            ncol=5,
            fontsize=style.POINT_LABEL_PT,
            frameon=False,
        )
        style.finish(fig, title, subtitle, SRC_BOTH)
    alt = "Шесть панелей: z-оценки SW, CH, S_Dbw, AVI, AVU, MQ по K для каждого метода"
    data = c[["cand", "method", "k", "feasible", *[f"z_{n}" for n in metrics], *metrics]]
    return _save(
        fig,
        out,
        "C03",
        "icvi_by_k",
        data,
        title=title,
        subtitle=subtitle,
        alt=alt,
        check="медиана лучшего K у признаковых индексов меньше, чем у графовых",
        source=SRC_BOTH,
    )


# --- C04: кривая гибрида по α ---------------------------------------------------------------------


def fig_alpha(out: Path, alpha: pd.DataFrame, metrics: list[str]) -> FigureInfo:
    """C04: гибрид при весе графа α от 0 (только X) до 1 (только G)."""
    a = alpha.sort_values("alpha").copy()
    feat = [m for m in metrics if m in ("sw", "ch", "s_dbw")]
    graph = [m for m in metrics if m in ("avi", "avu", "mq")]
    title = "С ростом веса графа качество на графе растёт, а в признаках падает"
    zf0, zf1 = a[f"z_{feat[0]}"].iloc[0], a[f"z_{feat[0]}"].iloc[-1]
    zg0, zg1 = a[f"z_{graph[0]}"].iloc[0], a[f"z_{graph[0]}"].iloc[-1]
    if not (zg1 > zg0 and zf1 < zf0):
        title = "Качество гибрида в двух пространствах при разном весе графа"
    k = int(a["k"].iloc[0])
    subtitle = (
        f"Гибрид «спектральное вложение G ⊕ X», K = {k}; α = 0 — только признаки, α = 1 — только граф; "
        "z-оценки ICVI против случайного базиса и ARI с итоговой типологией"
    )
    with style.use():
        fig, axes = style.new_figure("full", 1, 2)
        ax = axes[0]
        for name in metrics:
            ls = "-" if name in feat else "--"
            ax.plot(
                a["alpha"], a[f"z_{name}"], ls, marker="o", markersize=3, label=METRIC_LABELS.get(name, name)
            )
        ax.set_yscale("symlog", linthresh=10)
        ax.set_xlabel("Вес графа α")
        ax.set_ylabel("z-оценка (симметричный логарифм)")
        ax.legend(fontsize=style.POINT_LABEL_PT, loc="best")
        ax = axes[1]
        for col, lab in (
            ("ari_final", "с итоговой типологией"),
            ("ari_kmeans", "с K-means (только X)"),
            ("ari_spectral", "со спектральной (только G)"),
        ):
            if col in a.columns:
                ax.plot(a["alpha"], a[col], marker="o", markersize=3, label=lab)
        ax.set_xlabel("Вес графа α")
        ax.set_ylabel("ARI")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=style.POINT_LABEL_PT, loc="best")
        style.finish(fig, title, subtitle, SRC_BOTH)
    alt = (
        "Слева — z-оценки шести индексов по весу графа, справа — ARI гибрида с итогом и с методами одного "
        "источника"
    )
    return _save(
        fig,
        out,
        "C04",
        "alpha",
        a,
        title=title,
        subtitle=subtitle,
        alt=alt,
        check="z первого графового индекса при α = 1 выше, чем при α = 0, а признакового — ниже",
        source=SRC_BOTH,
    )


# --- C05: согласие методов ------------------------------------------------------------------------


def fig_agreement(out: Path, ari: pd.DataFrame, cands: pd.DataFrame) -> FigureInfo:
    """C05: ARI между победителями методов."""
    A = ari.set_index("cand")
    names = list(A.index)
    methods = [cands.loc[c, "method"] for c in names]
    fam = [cands.loc[c, "family"] for c in names]
    M = A.to_numpy(dtype=float)
    off = ~np.eye(len(names), dtype=bool)
    same_f = np.array([[fam[i] == fam[j] for j in range(len(names))] for i in range(len(names))])
    within = float(np.nanmean(M[off & same_f])) if (off & same_f).any() else float("nan")
    between = float(np.nanmean(M[off & ~same_f])) if (off & ~same_f).any() else float("nan")
    title = (
        "Методы одного семейства согласны между собой сильнее, чем с методами другого"
        if within > between
        else "Согласие разбиений разных методов"
    )
    subtitle = (
        f"ARI между победителями уровня K; среднее внутри семейства — {style.fmt_num(within, 2)}, между "
        f"семействами — {style.fmt_num(between, 2)}"
    )
    with style.use():
        fig, ax = style.new_figure("tall")
        ax.imshow(M, cmap=style.SEQ_CMAP, vmin=0, vmax=1)
        for i in range(len(names)):
            for j in range(len(names)):
                ax.text(
                    j,
                    i,
                    style.fmt_num(M[i, j], 2),
                    ha="center",
                    va="center",
                    fontsize=style.POINT_LABEL_PT - 1,
                    color="white" if M[i, j] < 0.55 else style.TEXT,
                )
        lab = [f"{mlabel(m)} ({int(cands.loc[c, 'k'])})" for m, c in zip(methods, names, strict=True)]
        ax.set_xticks(range(len(names)), lab, rotation=60, ha="right")
        ax.set_yticks(range(len(names)), lab)
        style.finish(fig, title, subtitle, SRC_BOTH)
    alt = "Матрица ARI между победителями методов с допустимым K"
    data = A.reset_index()
    return _save(
        fig,
        out,
        "C05",
        "agreement",
        data,
        title=title,
        subtitle=subtitle,
        alt=alt,
        check="средний ARI внутри семейства больше, чем между семействами",
        source=SRC_BOTH,
    )


# --- C06: профили итоговых типов ------------------------------------------------------------------


def fig_profiles(out: Path, prof: pd.DataFrame, method: str, k: int) -> FigureInfo:
    """C06: медианы признаков X и корзины по итоговым типам (в единицах MAD, 0 — медиана всех узлов)."""
    p = prof.copy()
    p["type"] = p["type"] + 1
    wide = p.pivot(index="type", columns="feature", values="median_scaled")
    order = [f for f in FEATURE_LABELS if f in wide.columns]
    wide = wide[order]
    sizes = p.drop_duplicates("type").set_index("type")["size"]
    title = f"Профили {k} типов: чем каждый отличается от типичной территории"
    subtitle = (
        f"{mlabel(method)}, K = {k}; медиана признака в типе минус медиана всех узлов, в единицах MAD × "
        f"1,4826; "
        "корзина и уровень трат — относительно своего региона; названия типам даст интерпретация"
    )
    with style.use():
        fig, ax = style.new_figure("tall")
        M = wide.to_numpy(dtype=float)
        lim = max(0.5, float(np.nanmax(np.abs(M))))
        ax.imshow(M, cmap=style.DIV_CMAP, vmin=-lim, vmax=lim, aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(
                    j,
                    i,
                    style.fmt_num(M[i, j], 1),
                    ha="center",
                    va="center",
                    fontsize=style.POINT_LABEL_PT - 1,
                )
        ax.set_yticks(range(len(wide)), [f"Тип {t} ({style.fmt_num(int(sizes[t]))})" for t in wide.index])
        ax.set_xticks(range(len(order)), [FEATURE_LABELS[f] for f in order], rotation=60, ha="right")
        style.finish(fig, title, subtitle, SRC_BOTH)
    alt = (
        "Тепловая карта: строки — типы с числом узлов, столбцы — 17 признаков; цвет — отклонение медианы типа"
    )
    return _save(
        fig,
        out,
        "C06",
        "profiles",
        wide.reset_index(),
        title=title,
        subtitle=subtitle,
        alt=alt,
        check="заголовок без чисел, зависящих от данных, кроме K",
        source=SRC_BOTH,
    )


# --- C07: карта итоговых типов --------------------------------------------------------------------


def fig_map(
    out: Path, cfg, final: pd.DataFrame, members: pd.DataFrame, k: int, method: str, ami_region: float
) -> FigureInfo:
    """C07: итоговые типы на карте: тип узла у каждого МО (районы Москвы и Петербурга — тип своего
    узла-города)."""
    from munnet import maps

    geo = maps.load_geometry(Path(cfg["paths"]["processed"]))
    node_type = final.set_index("territory_id")["type"]
    mem = members.loc[members["role"].astype(str) != "excluded", ["territory_id", "node_id"]]
    mo = mem.assign(type=mem["node_id"].map(node_type)).dropna(subset=["type"])
    values = pd.Series(
        [f"Тип {int(t)}" for t in mo["type"]], index=mo["territory_id"].astype("int32").to_numpy()
    )
    colors = {f"Тип {t}": TYPE_COLORS[(t - 1) % len(TYPE_COLORS)] for t in range(1, k + 1)}
    title = (
        f"Итоговые типы (K = {k}) на карте: типы не повторяют регионы"
        if ami_region < 0.1
        else f"Итоговые типы (K = {k}) на карте"
    )
    subtitle = (
        f"{mlabel(method)}, K = {k}; тип узла сети у каждого МО; районы Москвы и Петербурга — тип своего "
        f"узла-города; серым — МО вне сети (ряд короче 24 месяцев); AMI типов с группой региона — "
        f"{style.fmt_num(ami_region, 3)}"
    )
    fig, _ = maps.russia_map(
        values,
        geo,
        2023,
        kind="categorical",
        cmap=colors,
        legend_title="Тип",
        insets=maps.insets_from_config(cfg),
        na_label="вне сети",
    )
    style.finish(fig, title, subtitle, SRC_BOTH)
    alt = "Карта России: МО окрашены по типу своего узла сети"
    data = mo.rename(columns={"node_id": "node"})
    return _save(
        fig,
        out,
        "C07",
        "map",
        data,
        title=title,
        subtitle=subtitle,
        alt=alt,
        check="«не повторяют регионы» — AMI типов с группой региона меньше 0,1 (проверка в отчёте)",
        source=SRC_BOTH,
    )
