"""Этап evaluate: индексы качества для всех кандидатов этапа cluster — значения, интервалы, базис, согласие.

Входы:

- ``data/processed/cluster_labels.parquet`` (схема ``CLUSTER_LABELS``) — метки каждого кандидата «метод × K»
  на узлах основного режима, флаги допустимости, победителя метода и итога; −1 — шум HDBSCAN;
- X и G — строго те же, что у этапа cluster: ``munnet.clustering.inputs.load_inputs`` с параметрами
  ``clustering.inputs`` (X — 11 признаков после пропусков и робастного масштаба, G — сеть корзины
  ``basket_dist``, kNN, k = 10, веса S_ij).

Для каждого кандидата и каждого из шести индексов (``icvi.metrics``) и ANUI (справочно):

- значение на всех узлах (шум исключён — строки X, узлы и их рёбра G);
- случайный базис: ``icvi.baseline_permutations`` перестановок меток с теми же размерами кластеров, seed —
  тот же рецепт, что у этапа cluster (``[seed, 17, crc32(кандидат)]``), поэтому z совпадают с его z;
  z со знаком «больше — лучше», ``n_undefined`` — перестановки с NaN (вырождение S_Dbw);
- 95% интервал по подвыборкам узлов без возвращения (те же выборки, что ``clustering.stability.subsample``):
  отклонения значений подвыборок от их среднего умножаются на √(m/(n − m)) и откладываются от значения
  на всех узлах (``subsample_interval``). Метки кандидата не пересчитываются: интервал говорит, насколько
  индекс зависит от состава узлов при данном разбиении, а не насколько устойчив метод.

Анализ: кривые по K, τ Кендалла ранжирований кандидатов (по z и по сырым значениям), зависимость от K,
«пространство метода» (ранг в признаках против ранга в сети), разбор вырождений, матрица объединяемости
итогового разбиения.

Выходы — ``outputs/evaluate/``: ``icvi_long.parquet`` и ``icvi_long.csv``, ``candidates.csv``,
``agreement_z.csv``, ``agreement_value.csv``, ``k_dependence.csv``, ``space_ranks.csv``,
``final_unifiability.csv``, ``report_facts.json``, ``figures/``; отчёт ``icvi.report`` (``docs/icvi.md``).
"""

from __future__ import annotations

import logging
import math
import os
import time
import warnings
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import kendalltau, spearmanr

from munnet import icvi
from munnet.config import Config
from munnet.contracts import (
    CANDIDATE_PATTERN,
    CLUSTER_LABELS,
    Col,
    MissingInputError,
    QCError,
    TableSchema,
    read_table,
    validate,
    write_table,
)

log = logging.getLogger(__name__)

OUTPUT_SUBDIR = "evaluate"
METRICS: tuple[str, ...] = tuple(icvi.SPACE)
EXTRA: tuple[str, ...] = ("anui",)  # справочно: в правило выбора не входит
SPACE_OF: dict[str, str] = {**icvi.SPACE, "anui": "graph"}
BETTER_OF: dict[str, str] = {**icvi.BETTER, "anui": "max"}
CLUSTER_HINT = "сначала запустите этап cluster: python -m munnet cluster"

ICVI_LONG = TableSchema(
    name="icvi_long",
    columns=(
        Col("candidate", "string", pattern=CANDIDATE_PATTERN),
        Col("method", "string"),
        Col("k", "int16", range=(0, 999)),
        Col("space", "string", values=("features", "graph")),
        Col("metric", "string", values=(*METRICS, *EXTRA)),
        Col("value", "float64", nullable=True),
        Col("ci_low", "float64", nullable=True),
        Col("ci_high", "float64", nullable=True),
        Col("baseline_mean", "float64", nullable=True),
        Col("baseline_sd", "float64", nullable=True),
        Col("z", "float64", nullable=True),
        Col("n_undefined", "int32", range=(0, 1e9)),
        Col("percentile", "float64", nullable=True, range=(0.0, 1.0)),
    ),
    key=("candidate", "metric"),
)


@dataclass(frozen=True)
class EvalParams:
    """Параметры этапа: секция ``icvi`` (+ доля подвыборки и семейства методов из ``clustering``)."""

    specs: dict[str, icvi.MetricSpec]
    options: dict[str, Any]
    permutations: int
    bootstrap: int
    ci_level: float
    subsample: float
    seed: int
    workers: int
    families: dict[str, str]  # метод -> семейство
    report: str
    report_images: str

    @classmethod
    def from_config(cls, cfg: Config) -> EvalParams:
        ic, cl = cfg["icvi"], cfg["clustering"]
        workers = int((cl.get("impl") or {}).get("workers", 0))
        if workers <= 0:
            workers = os.cpu_count() or 1
        level = float(ic.get("ci_level", 0.95))
        if not 0 < level < 1:
            raise ValueError(f"icvi.ci_level = {level}: нужно (0; 1)")
        return cls(
            specs=icvi.metric_specs(cfg),
            options=icvi.options(cfg),
            permutations=int(ic["baseline_permutations"]),
            bootstrap=int(ic.get("bootstrap", 100)),
            ci_level=level,
            subsample=float(cl["stability"]["subsample"]),
            seed=int(cfg["seed"]),
            workers=workers,
            families={str(m): str(fam) for fam, ms in cl["families"].items() for m in ms},
            report=str(ic.get("report", "docs/icvi.md")),
            report_images=str(ic.get("report_images", "docs/img/icvi")),
        )


# --- Входы ---------------------------------------------------------------------------------------


def load_xa(cfg: Config) -> tuple[np.ndarray, np.ndarray, sp.csr_matrix]:
    """Узлы, X и G — функциями этапа cluster (``clustering.inputs.load_inputs``), без своей сборки."""
    from munnet.clustering.inputs import load_inputs
    from munnet.clustering.params import ClusterParams

    inp = load_inputs(cfg, ClusterParams.from_config(cfg)).inputs
    return inp.ids, inp.X, inp.A


def read_labels(cfg: Config) -> pd.DataFrame:
    path = Path(cfg["paths"]["processed"]) / "cluster_labels.parquet"
    if not path.exists():
        raise MissingInputError(f"нет {path}: {CLUSTER_HINT}")
    return read_table(path, CLUSTER_LABELS)


def label_matrix(tab: pd.DataFrame, ids: np.ndarray) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Кандидаты (по одной строке) и их метки в порядке ``ids``; узлы меток должны совпасть с узлами X и G."""
    want = set(np.asarray(ids).tolist())
    labels: dict[str, np.ndarray] = {}
    for cand, g in tab.groupby("candidate", sort=True):
        have = set(g["territory_id"].astype(int).tolist())
        if have != want:
            raise QCError(
                f"evaluate: у кандидата {cand} {len(have)} узлов, у входов cluster {len(want)}; "
                f"различаются {len(have ^ want)} узлов — метки и входы от разных прогонов"
            )
        labels[str(cand)] = g.set_index("territory_id").loc[ids, "label"].to_numpy(dtype=np.int64)
    cols = ["candidate", "method", "k", "param", "feasible", "is_method_winner", "is_final"]
    cands = tab[cols].drop_duplicates("candidate").sort_values("candidate").reset_index(drop=True)
    return cands, labels


# --- Интервал по подвыборкам ---------------------------------------------------------------------


def subsample_interval(
    value: float, boot: np.ndarray, *, m: int, n: int, level: float
) -> tuple[float, float]:
    """Интервал для значения на всех n узлах по значениям на подвыборках m узлов без возвращения.

    У подвыборки без возвращения доли f = m/n разброс статистики меньше выборочного: для среднего
    Var(θ_m − θ_n) ≈ σ²(1/m − 1/n), а Var(θ_n) ≈ σ²/n, отношение — m/(n − m). Поэтому отклонения
    подвыборок от их среднего умножаются на c = √(m/(n − m)) (при f = 0,8 c = 2) и откладываются
    от значения на всех узлах: [θ_n + c·q_{α/2}, θ_n + c·q_{1−α/2}] (идея масштабирования подвыборок —
    Politis, Romano, Wolf, Subsampling, 1999; для индексов разбиения это приближение). Центрирование
    по среднему подвыборок убирает сдвиг из-за меньшего числа узлов. Меньше половины конечных значений —
    NaN.
    """
    b = np.asarray(boot, dtype=np.float64)
    b = b[np.isfinite(b)]
    if not np.isfinite(value) or b.size < max(2, len(boot) / 2) or m >= n:
        return float("nan"), float("nan")
    c = math.sqrt(m / (n - m))
    dev = b - b.mean()
    alpha = 1 - level
    return float(value + c * np.quantile(dev, alpha / 2)), float(value + c * np.quantile(dev, 1 - alpha / 2))


# --- Расчёт одного кандидата ---------------------------------------------------------------------


def candidate_task(task) -> dict[str, Any]:
    """Значения, базис, подвыборки и слагаемые S_Dbw одного кандидата. ``task`` — (кандидат, метки)."""
    from munnet.clustering import parallel
    from munnet.clustering import stability as ST

    cand, labels = task
    st = parallel.STATE
    x, a, p = st["X"], st["A"], st["params"]
    names = list(METRICS)
    n = labels.shape[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", icvi.IcviWarning)
        values = icvi.evaluate(labels, X=x, A=a, names=names, **p.options)
        rng = np.random.default_rng([p.seed, 17, zlib.crc32(cand.encode())])
        base = icvi.baseline_values(
            labels, X=x, A=a, names=names, n_perm=p.permutations, rng=rng, **p.options
        )
        boot = {name: np.full(p.bootstrap, np.nan) for name in names}
        sizes = np.empty(p.bootstrap, dtype=np.int64)
        for rep in range(p.bootstrap):
            idx = ST.subsample(n, p.subsample, p.seed, rep)
            lab = labels[idx]
            sizes[rep] = len(idx)
            vals = icvi.evaluate(lab, X=x[idx], A=a[idx][:, idx], names=names, **p.options)
            k_sub = len(np.unique(lab[lab != icvi.NOISE]))
            n_sub, n_all = int((lab != icvi.NOISE).sum()), int((labels != icvi.NOISE).sum())
            k_all = len(np.unique(labels[labels != icvi.NOISE]))
            for name in names:
                v = vals[name]
                if name == "ch" and np.isfinite(v) and n_sub > k_sub:
                    v *= (n_all - k_all) / (n_sub - k_sub)  # CH ∝ (n − K) при том же отношении B/W
                boot[name][rep] = v
        parts = icvi.s_dbw_parts(x, labels, density=p.options["s_dbw_density"])
    return {
        "candidate": cand,
        "values": values,
        "base": base,
        "boot": boot,
        "m": int(sizes.max()),
        "parts": parts,
    }


def percentile_in_baseline(value: float, base: np.ndarray, better: str) -> float:
    """Доля перестановок базиса, которые кандидат обходит в сторону «лучше» (ничьи — пополам); NaN, если
    значение или все перестановки не определены. 1 — лучше всех случайных меток тех же размеров."""
    b = np.asarray(base, dtype=np.float64)
    b = b[np.isfinite(b)]
    if not np.isfinite(value) or b.size == 0:
        return float("nan")
    worse = b < value if better == "max" else b > value
    return float((worse.sum() + 0.5 * (b == value).sum()) / b.size)


def _stats(v: np.ndarray) -> tuple[float, float, int]:
    fin = v[np.isfinite(v)]
    mean = float(fin.mean()) if fin.size else float("nan")
    sd = float(fin.std(ddof=1)) if fin.size > 1 else float("nan")
    return mean, sd, int(v.size - fin.size)


@dataclass
class Result:
    long: pd.DataFrame
    cands: pd.DataFrame
    final_u: pd.DataFrame
    labels: dict[str, np.ndarray]


def evaluate_candidates(
    ids: np.ndarray, x: np.ndarray, a: sp.csr_matrix, tab: pd.DataFrame, p: EvalParams
) -> Result:
    """Длинная таблица индексов и таблица кандидатов для меток ``tab`` на входах (ids, X, A)."""
    from munnet.clustering import parallel

    cands, labels = label_matrix(tab, ids)
    tasks = [(c, labels[c]) for c in cands["candidate"]]
    t0 = time.perf_counter()
    out = parallel.run(candidate_task, tasks, {"X": x, "A": a, "params": p}, p.workers)
    log.info("evaluate: %d кандидатов за %.1f с", len(out), time.perf_counter() - t0)
    n = len(ids)
    rows, extra = [], []
    meta = cands.set_index("candidate")
    for r in out:
        cand = r["candidate"]
        values, base, boot = dict(r["values"]), dict(r["base"]), dict(r["boot"])
        values["anui"] = icvi.anui(values["avi"], values["avu"])
        base["anui"] = base["avi"] / (1 + base["avi"] * base["avu"])
        boot["anui"] = boot["avi"] / (1 + boot["avi"] * boot["avu"])
        for name in (*METRICS, *EXTRA):
            mean, sd, undefined = _stats(base[name])
            lo, hi = subsample_interval(values[name], boot[name], m=r["m"], n=n, level=p.ci_level)
            rows.append(
                {
                    "candidate": cand,
                    "method": meta.loc[cand, "method"],
                    "k": int(meta.loc[cand, "k"]),
                    "space": SPACE_OF[name],
                    "metric": name,
                    "value": float(values[name]),
                    "ci_low": lo,
                    "ci_high": hi,
                    "baseline_mean": mean,
                    "baseline_sd": sd,
                    "z": icvi.zscore(values[name], mean, sd, BETTER_OF[name]),
                    "n_undefined": undefined,
                    "percentile": percentile_in_baseline(values[name], base[name], BETTER_OF[name]),
                }
            )
        lab = labels[cand]
        ok = lab != icvi.NOISE
        extra.append(
            {
                "candidate": cand,
                "k_eff": len(np.unique(lab[ok])),
                "noise_share": float(1 - ok.mean()),
                "scat": r["parts"]["scat"],
                "dens_bw": r["parts"]["dens_bw"],
                "stdev": r["parts"]["stdev"],
                "s_dbw_undefined": bool(r["parts"]["undefined"]),
            }
        )
    long = pd.DataFrame(rows, columns=list(ICVI_LONG.names))
    long["k"] = long["k"].astype("int16")
    long["n_undefined"] = long["n_undefined"].astype("int32")
    validate(long, ICVI_LONG)
    cands = cands.merge(pd.DataFrame(extra), on="candidate")
    cands.insert(2, "family", cands["method"].map(p.families).fillna("other"))
    final = cands.loc[cands["is_final"].astype(bool), "candidate"]
    final_u = final_unifiability(a, labels[final.iloc[0]]) if len(final) else pd.DataFrame()
    return Result(long=long, cands=cands, final_u=final_u, labels=labels)


def final_unifiability(a: sp.csr_matrix, labels: np.ndarray) -> pd.DataFrame:
    """Объединяемость пар типов итогового разбиения, изолируемость и размер типа; тип = метка + 1."""
    u, iso, uniq = icvi.unifiability(a, labels)
    types = (uniq + 1).astype(int)
    frame = pd.DataFrame(u, columns=[f"type_{t}" for t in types])
    frame.insert(0, "type", types)
    frame["isolability"] = iso
    frame["size"] = [int((labels == c).sum()) for c in uniq]
    frame["max_u"] = np.nanmax(u, axis=1) if len(uniq) > 1 else np.nan
    frame["max_u_with"] = [int(types[np.nanargmax(row)]) if np.isfinite(row).any() else 0 for row in u]
    return frame


# --- Анализ --------------------------------------------------------------------------------------


def _wide(long: pd.DataFrame, cands: pd.DataFrame, value: str, feasible_only: bool) -> pd.DataFrame:
    pool = cands.loc[cands["feasible"].astype(bool), "candidate"] if feasible_only else cands["candidate"]
    w = long.loc[long["candidate"].isin(pool)].pivot(index="candidate", columns="metric", values=value)
    if value == "value":  # «больше — лучше» у всех столбцов
        for name in w.columns:
            if BETTER_OF.get(name) == "min":
                w[name] = -w[name]
    return w


def agreement(
    long: pd.DataFrame,
    cands: pd.DataFrame,
    *,
    value: str = "z",
    metrics: list[str] | None = None,
    feasible_only: bool = True,
) -> pd.DataFrame:
    """τ Кендалла (τ_b) ранжирований кандидатов по парам метрик, «лучше — выше»; попарно по кандидатам,
    где обе метрики конечны (NaN z при вырожденном базисе исключается). Диагональ — 1."""
    metrics = list(METRICS) if metrics is None else metrics
    w = _wide(long, cands, value, feasible_only)
    tau = pd.DataFrame(np.eye(len(metrics)), index=metrics, columns=metrics)
    for i, a_ in enumerate(metrics):
        for b_ in metrics[i + 1 :]:
            both = w[[a_, b_]].dropna()
            t = kendalltau(both[a_], both[b_]).statistic if len(both) >= 3 else float("nan")
            tau.loc[a_, b_] = tau.loc[b_, a_] = t
    return tau


def agreement_within_k(
    long: pd.DataFrame, cands: pd.DataFrame, *, value: str = "value", min_n: int = 5
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """τ Кендалла внутри одного K (по всем кандидатам с этим K, «лучше — выше»): эффект K исключён.

    Внутри K допустимых кандидатов мало (2–5), поэтому берутся все кандидаты этого K — допустимость касается
    размеров типов, а не того, как метрики упорядочивают разбиения. Слой K участвует, если у пары метрик
    в нём не меньше ``min_n`` кандидатов с конечными значениями. Итог — среднее τ по слоям, взвешенное числом
    кандидатов (τ стратифицированный по K). Возвращает матрицу и длинную таблицу по слоям."""
    w = _wide(long, cands, value, feasible_only=False).reindex(columns=list(METRICS))
    k = cands.set_index("candidate")["k_eff"].reindex(w.index)
    rows = []
    for kk, idx in w.groupby(k).groups.items():
        g = w.loc[idx]
        for i, a_ in enumerate(METRICS):
            for b_ in METRICS[i + 1 :]:
                d = g[[a_, b_]].dropna()
                if len(d) >= min_n and d[a_].nunique() > 1 and d[b_].nunique() > 1:
                    t = kendalltau(d[a_], d[b_]).statistic
                    rows.append({"k": int(kk), "a": a_, "b": b_, "tau": t, "n": len(d)})
    by_k = pd.DataFrame(rows, columns=["k", "a", "b", "tau", "n"])
    tau = pd.DataFrame(np.eye(len(METRICS)), index=list(METRICS), columns=list(METRICS))
    for (a_, b_), g in by_k.groupby(["a", "b"]):
        v = float(np.average(g["tau"], weights=g["n"])) if len(g) else float("nan")
        tau.loc[a_, b_] = tau.loc[b_, a_] = v
    for i, a_ in enumerate(METRICS):
        for b_ in METRICS[i + 1 :]:
            if by_k.loc[(by_k["a"] == a_) & (by_k["b"] == b_)].empty:
                tau.loc[a_, b_] = tau.loc[b_, a_] = float("nan")
    return tau, by_k


def k_fair(long: pd.DataFrame, cands: pd.DataFrame) -> pd.DataFrame:
    """Сравнение разных K без смещения, дополняющее z: AVI с поправкой на случайность
    (AVI − E) / (1 − E), где E — среднее AVI случайного базиса (≈ 1/K), — по образцу поправки ARI
    (Hubert, Arabie, 1985); MQ уже отсчитана от нулевой модели (среднее базиса ≈ 0); процентиль в базисе;
    ранг кандидата среди всех кандидатов того же K (1 — лучший)."""
    v = long.pivot(index="candidate", columns="metric", values="value")
    base = long.pivot(index="candidate", columns="metric", values="baseline_mean")
    z = long.pivot(index="candidate", columns="metric", values="z")
    pct = long.pivot(index="candidate", columns="metric", values="percentile")
    out = cands[["candidate", "method", "family", "k_eff", "feasible", "is_method_winner", "is_final"]].copy()
    out = out.set_index("candidate")
    out["avi"] = v["avi"]
    out["avi_base"] = base["avi"]
    with np.errstate(divide="ignore", invalid="ignore"):
        out["avi_adj"] = (v["avi"] - base["avi"]) / (1 - base["avi"])
    out["mq"] = v["mq"]
    out["mq_base"] = base["mq"]
    out["z_avi"] = z["avi"]
    out["z_mq"] = z["mq"]
    out["pct_avi"] = pct["avi"]
    out["pct_mq"] = pct["mq"]
    out["n_in_k"] = out.groupby("k_eff")["avi"].transform("size")
    for col in ("avi", "avi_adj", "mq"):
        out[f"rank_in_k_{col}"] = out.groupby("k_eff")[col].rank(ascending=False, method="min")
    return out.reset_index()


def k_dependence(long: pd.DataFrame, cands: pd.DataFrame) -> pd.DataFrame:
    """ρ Спирмена «значение — K» и «z — K» по допустимым кандидатам (значение со знаком «больше — лучше»)."""
    k = cands.set_index("candidate")["k_eff"]
    rows = []
    for value, col in (("value", "rho_value_k"), ("z", "rho_z_k")):
        w = _wide(long, cands, value, True)
        for name in METRICS:
            s = w[name].dropna()
            rho = spearmanr(s, k.loc[s.index]).statistic if s.size >= 3 and s.nunique() > 1 else float("nan")
            rows.append({"metric": name, "col": col, "rho": rho})
    d = pd.DataFrame(rows).pivot(index="metric", columns="col", values="rho").reindex(list(METRICS))
    return d.reset_index()[["metric", "rho_value_k", "rho_z_k"]]


def space_ranks(long: pd.DataFrame, cands: pd.DataFrame) -> pd.DataFrame:
    """Качество кандидата в X и в G: средний перцентильный ранг z среди допустимых (0 — худший, 1 — лучший),
    как ``quality_features`` и ``quality_graph`` правила выбора этапа 3; недопустимые — ранг среди допустимых
    по вставке (для графика)."""
    w_all = _wide(long, cands, "z", False)
    pool = set(cands.loc[cands["feasible"].astype(bool), "candidate"])
    out = pd.DataFrame(index=w_all.index)
    for space, names in (("features", ("sw", "ch", "s_dbw")), ("graph", ("avi", "avu", "mq"))):
        ranks = []
        for name in names:
            ref = w_all.loc[w_all.index.isin(pool), name].dropna().to_numpy()
            col = w_all[name]
            if ref.size < 2:
                ranks.append(pd.Series(np.nan, index=col.index))
                continue
            r = col.map(
                lambda v, ref=ref: (
                    np.nan
                    if not np.isfinite(v)
                    else (np.sum(ref < v) + 0.5 * np.sum(ref == v) - 0.5) / (ref.size - 1)
                )
            )
            ranks.append(r.clip(0, 1))
        out[f"q_{space}"] = pd.concat(ranks, axis=1).mean(axis=1, skipna=True)
    out = out.reset_index().rename(columns={"index": "candidate"})
    return cands[["candidate", "method", "family", "k", "feasible", "is_method_winner", "is_final"]].merge(
        out, on="candidate"
    )


def nn_distance_median(x: np.ndarray) -> float:
    """Медиана расстояния до ближайшего соседа в X — масштаб, с которым сравнивается радиус stdev S_Dbw."""
    from sklearn.neighbors import NearestNeighbors

    d, _ = NearestNeighbors(n_neighbors=2).fit(x).kneighbors(x)
    return float(np.median(d[:, 1]))


def z_match_cluster(path: Path, long: pd.DataFrame) -> dict[str, float]:
    """Сверка с этапом cluster: z тех же кандидатов из его ``candidates.csv`` (колонки ``z_<индекс>``).

    Наибольшее относительное расхождение |Δz| / max(1, |z|) и число несовпадений «NaN или нет»; файла нет —
    NaN (сверка не проводилась)."""
    if not path.exists():
        return {"rel": float("nan"), "nan_mismatch": float("nan"), "n": 0}
    c = pd.read_csv(path)
    key = "cand" if "cand" in c.columns else "candidate"
    c = c.set_index(key)
    w = long.pivot(index="candidate", columns="metric", values="z")
    rel, mism, n = 0.0, 0, 0
    for m in METRICS:
        if f"z_{m}" not in c.columns:
            continue
        a = c[f"z_{m}"].reindex(w.index)
        b = w[m]
        both = a.notna() & b.notna()
        mism += int((a.isna() != b.isna()).sum())
        n += int(both.sum())
        if both.any():
            rel = max(rel, float((np.abs(a[both] - b[both]) / np.maximum(1.0, np.abs(b[both]))).max()))
    return {"rel": rel, "nan_mismatch": float(mism), "n": n}


# --- Запуск --------------------------------------------------------------------------------------


def write_csv(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, float_format="%.10g", lineterminator="\n")
    return path


def run(cfg: Config) -> None:
    from munnet import icvi_report

    p = EvalParams.from_config(cfg)
    tab = read_labels(cfg)
    ids, x, a = load_xa(cfg)
    res = evaluate_candidates(ids, x, a, tab, p)
    out = Path(cfg.dir("outputs")) / OUTPUT_SUBDIR
    out.mkdir(parents=True, exist_ok=True)
    write_table(res.long, ICVI_LONG, out / "icvi_long.parquet")
    write_csv(res.long, out / "icvi_long.csv")
    write_csv(res.cands, out / "candidates.csv")
    tau_z = agreement(res.long, res.cands, value="z")
    tau_v = agreement(res.long, res.cands, value="value")
    write_csv(tau_z.rename_axis("metric").reset_index(), out / "agreement_z.csv")
    write_csv(tau_v.rename_axis("metric").reset_index(), out / "agreement_value.csv")
    write_csv(k_dependence(res.long, res.cands), out / "k_dependence.csv")
    tau_k, tau_k_long = agreement_within_k(res.long, res.cands)
    write_csv(tau_k.rename_axis("metric").reset_index(), out / "agreement_within_k.csv")
    write_csv(tau_k_long, out / "agreement_within_k_long.csv")
    write_csv(k_fair(res.long, res.cands), out / "k_fair.csv")
    write_csv(space_ranks(res.long, res.cands), out / "space_ranks.csv")
    if len(res.final_u):
        write_csv(res.final_u, out / "final_unifiability.csv")
    extra = {"n_nodes": len(ids), "n_features": int(x.shape[1]), "n_edges": int(a.nnz // 2)}
    extra["nn_median"] = nn_distance_median(x)
    extra["z_match"] = z_match_cluster(Path(cfg["paths"]["outputs"]) / "cluster" / "candidates.csv", res.long)
    icvi_report.write_all(cfg, p, out, extra)
    log.info("evaluate: выходы %s", out)
