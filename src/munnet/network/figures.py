"""Рисунки этапа network: «до и после», надёжность против географии, совпадение разбиений, окна динамики.

Каждый рисунок: заголовок-вывод, подзаголовок с числами, строка источника (``style.finish``), PNG и SVG
в ``outputs/network/figures/`` и CSV с данными рисунка рядом. Заголовок собирается по числам: если вывод
перестал быть верным, выбирается нейтральная формулировка (``check`` в записи рисунка говорит, что проверено).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from munnet import style
from munnet.network.params import NetworkParams

FIG_SUBDIR = "figures"

RULE_LABELS: dict[str, str] = {
    "basket_cos": "Косинус корзин",
    "basket_dist": "Расстояние корзин",
    "rhythm_corr": "Корреляция ритма",
    "rhythm_corr_all": "Корреляция ритма (только итог)",
    "rhythm_lag": "Лаговая корреляция",
    "rhythm_dtw": "DTW ритма",
    "geo_road": "Дороги",
    "geo_gravity": "Гравитация",
    "basket_dist_abs": "Расстояние корзин без вычета региона",
    "rhythm_corr_rel": "Корреляция ритма сверх региона",
}
# Вертикальный сдвиг подписи точки на рис. 2 (пункты): соседние точки подписываются выше и ниже.
LABEL_DY: dict[str, float] = {"basket_cos": -12, "rhythm_corr_all": -12, "rhythm_dtw": 12}
FAMILY_COLOR: dict[str, str] = {
    "basket": style.PALETTE["marketplace"],
    "rhythm": style.PALETTE["transport"],
    "geo": style.TEXT2,
}


def label(rule: str) -> str:
    return RULE_LABELS.get(rule, rule)


def family(kind: str) -> str:
    return "basket" if kind.startswith("basket") else "rhythm" if kind.startswith("rhythm") else "geo"


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
    csv = out / FIG_SUBDIR / f"{fid}_{slug}.csv"
    from munnet.network.stage import write_csv

    write_csv(data, csv)
    return FigureInfo(fid=fid, slug=slug, png=stem.with_suffix(".png"), data_csv=csv, **info)


def fig_signal(out: Path, hist: pd.DataFrame, stats: dict, n: int) -> FigureInfo:
    """N01: сходство пар «до и после»: ряды без обработки и свой ритм; корзина без поправки и относительно
    региона."""
    raw, own, cos_abs, cos_rel = (stats[k]["median"] for k in ("corr_raw", "corr_own", "cos_abs", "cos_rel"))
    ok = raw > 0.5 and abs(own) < 0.1 and cos_abs > 0.5 and abs(cos_rel) < 0.1
    title = (
        "Без обработки почти все пары МО похожи; после снятия общего фона и региона сходство становится "
        "избирательным"
        if ok
        else "Распределение сходства пар до и после снятия общего фона и региона"
    )
    subtitle = (
        f"Медиана по парам: корреляция ln трат {style.fmt_num(raw, 2)} → своего ритма "
        f"{style.fmt_num(own, 2)}; "
        f"косинус корзин {style.fmt_num(cos_abs, 2)} → относительно региона {style.fmt_num(cos_rel, 2)}; "
        f"{style.fmt_num(n)} узлов, 2023–2024"
    )
    with style.use():
        fig, axes = style.new_figure("full", 1, 2, sharey=False)
        ax = axes[0]
        edges = np.append(hist["bin_left"].to_numpy(), hist["bin_right"].iloc[-1])
        ax.stairs(
            hist["corr_raw"], edges, color=style.CONTEXT, fill=True, alpha=0.7, label="ln трат без обработки"
        )
        ax.stairs(hist["corr_own"], edges, color=style.ACCENT, linewidth=1.6, label="свой ритм")
        ax.stairs(
            hist["corr_null"],
            edges,
            color=style.TEXT2,
            linewidth=1.0,
            linestyle="--",
            label="свой ритм, ряды сдвинуты",
        )
        ax.set_xlabel("Корреляция пары МО по 24 месяцам")
        ax.set_ylabel("Доля пар")
        ax.legend(loc="upper left")
        ax = axes[1]
        ax.stairs(hist["cos_abs"], edges, color=style.CONTEXT, fill=True, alpha=0.7, label="CLR корзины")
        ax.stairs(
            hist["cos_rel"],
            edges,
            color=style.PALETTE["marketplace"],
            linewidth=1.6,
            label="CLR минус центр региона",
        )
        ax.set_xlabel("Косинус корзин пары МО, 2023–2024")
        ax.legend(loc="upper left")
        for a in axes:
            a.yaxis.set_major_formatter(style.ru_formatter("pct", 0))
            a.set_xlim(-1, 1)
        style.finish(fig, title, subtitle, style.SOURCE_SBER)
    alt = (
        f"Две гистограммы: корреляции сырых рядов сосредоточены у {style.fmt_num(raw, 2)}, своего ритма — "
        "около нуля, как на сдвинутых рядах, но с более длинным правым хвостом; косинусы корзин без поправки "
        f"у {style.fmt_num(cos_abs, 2)}, относительно региона — от −1 до 1"
    )
    return _save(
        fig,
        out,
        "N01",
        "signal",
        hist,
        title=title,
        subtitle=subtitle,
        alt=alt,
        check="медиана сырых > 0,5, своего ритма и корзин относительно региона по модулю < 0,1",
    )


def fig_pareto(out: Path, comparison: pd.DataFrame, chosen: str, p: NetworkParams) -> FigureInfo:
    """N02: надёжность (Жаккар сетей 2023 и 2024 годов) против доли рёбер, общих с дорожной сетью."""
    c = comparison.loc[~comparison["static"].astype(bool)].copy()
    c["family"] = c["kind"].map(family)
    abl = c["ablation"].astype(bool) if "ablation" in c.columns else pd.Series(False, index=c.index)
    cand = c.loc[c["rule"].isin(p.candidates)]
    best = cand.set_index("rule").loc[chosen]
    others = cand.loc[cand["rule"] != chosen]
    top = bool(
        (best["reliability"] >= others["reliability"]).all()
        and (best["geo_jaccard"] <= others["geo_jaccard"]).all()
    )
    title = (
        f"{label(chosen)} относительно региона — самая надёжная из сетей-кандидатов "
        "и почти не повторяет дороги"
        if top
        else "Надёжность сетей и совпадение с дорожной сетью"
    )
    road = comparison.loc[comparison["kind"] == "road"].iloc[0]
    lo, hi = float(c["attr_consistency"].min()), float(c["attr_consistency"].max())
    subtitle = (
        f"Жаккар рёбер kNN (k = {p.k}) сетей 2023 и 2024 годов против доли рёбер, общих с сетью "
        f"«{label(road['rule'])}»; случайный уровень Жаккара ≈ "
        f"{style.fmt_num(c['reliability_chance'].mean(), 3)}; размер точки — согласованность "
        f"с экономикой места (от {style.fmt_num(lo, 2)} до {style.fmt_num(hi, 2)}); пустые круги — абляции"
    )
    # сдвиг подписи по вертикали (пункты) для соседних точек, иначе подписи наезжают друг на друга
    dy_label = LABEL_DY
    with style.use():
        fig, ax = style.new_figure("full")
        for idx, r in c.iterrows():
            col = FAMILY_COLOR[r["family"]]
            size = 60 + 540 * (float(r["attr_consistency"]) - lo) / (hi - lo if hi > lo else 1.0)
            hollow = bool(abl.loc[idx])
            ax.scatter(
                r["geo_jaccard"],
                r["reliability"],
                s=size,
                color="white" if hollow else col,
                alpha=1.0 if hollow else 0.85,
                edgecolor=col if hollow else style.ACCENT if r["rule"] == chosen else "white",
                linewidth=1.6,
                zorder=3,
            )
            ax.annotate(
                label(r["rule"]),
                (r["geo_jaccard"], r["reliability"]),
                # подпись справа от круга (радиус в пунктах — √площади / 2)
                xytext=(4 + np.sqrt(size) / 2, dy_label.get(r["rule"], 2)),
                textcoords="offset points",
                fontsize=style.POINT_LABEL_PT,
                color=style.TEXT,
                va="center",
            )
        ax.set_xlabel(f"Доля рёбер, общих с сетью «{label(road['rule'])}» (Жаккар)")
        ax.set_ylabel("Надёжность: Жаккар сетей 2023 и 2024")
        ax.set_xlim(-0.005, max(0.12, float(c["geo_jaccard"].max()) * 1.3))
        top_y = max(0.2, float(c["reliability"].max()) * 1.25)
        ax.set_ylim(0, top_y)
        ax.set_yticks(np.arange(0, top_y + 1e-9, 0.05))
        style.finish(fig, title, subtitle, style.join_sources(style.SOURCE_SBER, style.SOURCE_ROSSTAT))
    data = c[["rule", "kind", "reliability", "reliability_chance", "geo_jaccard", "attr_consistency"]]
    alt = (
        "Точечный график: правила по корзине вверху слева (надёжнее и не повторяют дороги), правила по ритму "
        "ниже и правее"
    )
    return _save(
        fig,
        out,
        "N02",
        "pareto",
        data,
        title=title,
        subtitle=subtitle,
        alt=alt,
        check="у выбранного правила надёжность не ниже, а доля общих с дорогами рёбер не выше, чем у других "
        "кандидатов (абляции не в счёт)",
    )


def fig_ari(out: Path, pairs: pd.DataFrame, comparison: pd.DataFrame) -> FigureInfo:
    """N03: ARI разбиений зонда Leiden между правилами (выше диагонали) и Жаккар рёбер (ниже)."""
    rules = comparison["rule"].tolist()
    n = len(rules)
    M = np.full((n, n), np.nan)
    pos = {r: i for i, r in enumerate(rules)}
    for _, r in pairs.iterrows():
        i, j = pos[r["rule_a"]], pos[r["rule_b"]]
        a, b = min(i, j), max(i, j)
        M[a, b] = r["ari"]
        M[b, a] = r["jaccard"]
    fam = comparison.set_index("rule")["kind"].map(family)
    fa, fb = pairs["rule_a"].map(fam), pairs["rule_b"].map(fam)
    basket_other = pairs.loc[(fa == "basket") ^ (fb == "basket")]
    max_cross = float(basket_other["ari"].max()) if len(basket_other) else float("nan")
    title = (
        "Сообщества сети по корзине почти не совпадают с сообществами по ритму и по дорогам"
        if max_cross < 0.2
        else "Совпадение сообществ сетей разных правил"
    )
    subtitle = (
        "Выше диагонали — ARI разбиений Leiden (модульность, γ = 1), ниже — Жаккар рёбер kNN; "
        f"наибольший ARI правила по корзине с правилом другого семейства {style.fmt_num(max_cross, 2)}"
    )
    with style.use():
        fig, ax = style.new_figure("tall")
        im = ax.imshow(M, cmap=style.SEQ_CMAP, vmin=0, vmax=1)
        ax.set_xticks(range(n), [label(r) for r in rules], rotation=40, ha="right")
        ax.set_yticks(range(n), [label(r) for r in rules])
        ax.grid(False)
        for i in range(n):
            for j in range(n):
                if np.isfinite(M[i, j]):
                    ax.text(
                        j,
                        i,
                        style.fmt_num(M[i, j], 2),
                        ha="center",
                        va="center",
                        fontsize=style.POINT_LABEL_PT,
                        color="white" if M[i, j] < 0.5 else style.TEXT,
                    )
        del im  # числа подписаны в клетках: шкала цвета не нужна
        style.finish(fig, title, subtitle, style.SOURCE_SBER)
    data = pd.DataFrame(M, index=rules, columns=rules).reset_index().rename(columns={"index": "rule"})
    alt = (
        "Тепловая карта: высокие значения только внутри семейств правил (корзина с корзиной, ритм с ритмом, "
        "дороги с гравитацией)"
    )
    return _save(
        fig,
        out,
        "N03",
        "ari",
        data,
        title=title,
        subtitle=subtitle,
        alt=alt,
        check="наибольший ARI правила по корзине с правилами по ритму и географии < 0,2",
    )


def fig_windows(out: Path, noise: pd.DataFrame, chosen: str, k: int) -> FigureInfo:
    """N04: Жаккар сетей на нечётных и чётных месяцах одного отрезка (шум) против двух соседних окон."""
    g = noise.groupby("length")[["noise_jaccard", "change_jaccard"]].mean()
    ok = bool((g["change_jaccard"] < g["noise_jaccard"]).all())
    title = (
        "Сети соседних окон различаются сильнее, чем сети одного времени: связи между МО меняются"
        if ok
        else "Сети соседних окон и шум месяцев"
    )
    subtitle = (
        f"{label(chosen)}, kNN k = {k}; шум — сети на нечётных и чётных месяцах одного отрезка, изменение — "
        "сети двух соседних окон того же отрезка; среднее по отрезкам"
    )
    with style.use():
        fig, ax = style.new_figure("full")
        xs = np.arange(len(g))
        w = 0.36
        ax.bar(xs - w / 2, g["noise_jaccard"], width=w, color=style.CONTEXT, label="одно время (шум месяцев)")
        ax.bar(
            xs + w / 2,
            g["change_jaccard"],
            width=w,
            color=style.PALETTE["marketplace"],
            label="соседние окна",
        )
        for x, (a, b) in zip(xs, g.to_numpy(), strict=True):
            ax.text(
                x - w / 2, a, style.fmt_num(a, 2), ha="center", va="bottom", fontsize=style.POINT_LABEL_PT
            )
            ax.text(
                x + w / 2, b, style.fmt_num(b, 2), ha="center", va="bottom", fontsize=style.POINT_LABEL_PT
            )
        ax.set_xticks(xs, [f"{int(L)} мес." for L in g.index])
        ax.set_xlabel("Длина окна")
        ax.set_ylabel("Жаккар рёбер")
        ax.legend(loc="upper left")
        style.finish(fig, title, subtitle, style.SOURCE_SBER)
    alt = "Столбцы: при любой длине окна общих рёбер у соседних окон меньше, чем у двух сетей одного времени"
    return _save(
        fig,
        out,
        "N04",
        "windows",
        g.reset_index(),
        title=title,
        subtitle=subtitle,
        alt=alt,
        check="при каждой длине окна Жаккар соседних окон ниже Жаккара нечётных и чётных месяцев",
    )


def make_all(out: Path, comparison, pairs, noise, hist, stats, chosen, p: NetworkParams) -> list[FigureInfo]:
    return [
        fig_signal(out, hist, stats, int(comparison["n_nodes"].iloc[0])),
        fig_pareto(out, comparison, chosen, p),
        fig_ari(out, pairs, comparison),
        fig_windows(out, noise, chosen, p.k),
    ]
