"""Параметры этапа cluster из секции ``clustering`` конфига (и ``icvi`` — метрики и случайный базис).

Блок ``clustering`` до ключа ``impl`` — предрегистрация 28.09.2026 (коммит 2ee1363): состав методов, сетка K,
допустимость, правило выбора, чувствительность, устойчивость, объяснимость, внешняя проверка. Ключи ``impl``,
``synthetic``, ``report`` добавлены этапом 3 и в предрегистрацию не входят.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from munnet.config import Config

FAMILIES: tuple[str, ...] = ("features", "graph", "attributed", "fusion")
METHODS: tuple[str, ...] = (
    "kmeans",
    "ward",
    "gmm",
    "hdbscan",
    "leiden",
    "louvain",
    "spectral",
    "shalileh_mirkin",
    "hybrid",
    "kmeans_joint",
)
# Методы, у которых K — результат, а подбирается параметр (разрешение γ или min_cluster_size).
PARAM_METHODS: tuple[str, ...] = ("leiden", "louvain", "hdbscan")
# Методы со случайностью: итог — лучший из protocol.seeds запусков по собственному критерию.
RANDOM_METHODS: tuple[str, ...] = (
    "kmeans",
    "gmm",
    "leiden",
    "louvain",
    "spectral",
    "shalileh_mirkin",
    "hybrid",
    "kmeans_joint",
)
# Что видит метод: X — признаки, G — граф.
USES: dict[str, str] = {
    "kmeans": "X",
    "ward": "X",
    "gmm": "X",
    "hdbscan": "X",
    "leiden": "G",
    "louvain": "G",
    "spectral": "G",
    "shalileh_mirkin": "X и G",
    "hybrid": "X и G",
    "kmeans_joint": "X и координаты G",
}
CRITERIA: tuple[str, ...] = ("quality_features", "quality_graph", "stability", "interpretability")
QUALITY_SPACE: dict[str, str] = {"quality_features": "features", "quality_graph": "graph"}
AGGREGATORS: tuple[str, ...] = ("borda", "lexicographic_all_orders", "copeland_all")
VARIANT_KEYS: tuple[str, ...] = ("graph_rule", "drop_features", "nodes_mode")


@dataclass(frozen=True)
class Variant:
    name: str
    graph_rule: str | None = None
    drop_features: tuple[str, ...] = ()
    nodes_mode: str | None = None


@dataclass(frozen=True)
class ClusterParams:
    # входы
    graph_rule: str
    graph_sparsify: str
    graph_k: int
    features_key: str
    place_year: int
    scale: str
    impute: str
    attributes: tuple[str, ...]
    edges: tuple[str, ...]
    # методы и протокол
    k_grid: tuple[int, ...]
    families: Mapping[str, tuple[str, ...]]
    final_eligible: tuple[str, ...]
    seeds: tuple[int, ...]
    hybrid_alpha: float
    alpha_curve: tuple[float, ...]
    # допустимость
    k_range: tuple[int, int]
    min_cluster_share: float
    min_cluster_share_fallback: float
    max_cluster_share: float
    max_noise_share: float
    # выбор
    directions: Mapping[str, str]
    tie: Mapping[str, float]
    rule: str
    simplicity: Mapping[str, float]
    aggregators: tuple[str, ...]
    sets: Mapping[str, tuple[str, ...]]
    tie_scale: tuple[float, ...]
    joint: bool
    all_eligible: bool
    variants: tuple[Variant, ...]
    # устойчивость, объяснимость
    bootstrap: int
    bootstrap_variants: int
    subsample: float
    jaccard_stable: float
    jaccard_dissolved: float
    tree_depth: int
    cv_folds: int
    tree_inputs: tuple[str, ...]
    # внешняя проверка
    validation: Mapping[str, Any]
    # ICVI
    icvi_metrics: Mapping[str, Mapping[str, str]]
    icvi_permutations: int
    icvi_section: Mapping[str, Any]
    # реализация (не предрегистрация)
    impl: Mapping[str, Any]
    synthetic: Mapping[str, Any]
    report: str
    report_images: str
    seed: int
    workers: int = field(default=1)

    @property
    def methods(self) -> tuple[str, ...]:
        return tuple(m for fam in FAMILIES for m in self.families.get(fam, ()))

    def family_of(self, method: str) -> str:
        for fam, ms in self.families.items():
            if method in ms:
                return fam
        raise KeyError(method)

    @property
    def eligible_methods(self) -> tuple[str, ...]:
        return tuple(m for fam in self.final_eligible for m in self.families.get(fam, ()))

    def metric_names(self, space: str) -> tuple[str, ...]:
        return tuple(n for n, s in self.icvi_metrics.items() if s["space"] == space)

    @classmethod
    def from_config(cls, cfg: Config) -> ClusterParams:
        c: Mapping[str, Any] = cfg["clustering"]
        inp, proto, feas, sel = c["inputs"], c["protocol"], c["feasible"], c["selection"]
        stab, interp = c["stability"], c["interpretability"]
        space = cfg["features"]["space"]
        feats = str(inp["features"])
        k_lo, k_hi = (int(x) for x in c["n_clusters"])
        sens = sel.get("sensitivity") or {}
        variants = []
        for name, spec in (sens.get("variants") or {}).items():
            spec = spec or {}
            variants.append(
                Variant(
                    name=str(name),
                    graph_rule=spec.get("graph_rule"),
                    drop_features=tuple(str(x) for x in spec.get("drop_features") or ()),
                    nodes_mode=spec.get("nodes_mode"),
                )
            )
        impl = dict(c.get("impl") or {})
        workers = int(impl.get("workers", 0))
        if workers <= 0:
            workers = os.cpu_count() or 1
        icvi = cfg.data.get("icvi") or {}
        p = cls(
            graph_rule=str(inp["graph"]["rule"]),
            graph_sparsify=str(inp["graph"]["sparsify"]),
            graph_k=int(inp["graph"]["k"]),
            features_key=feats,
            place_year=int(inp["place_year"]),
            scale=str(inp["scale"]),
            impute=str(inp["impute"]),
            attributes=tuple(str(x) for x in space[feats]),
            edges=tuple(str(x) for x in space["edges"]),
            k_grid=tuple(range(k_lo, k_hi + 1)),
            families={str(k): tuple(str(m) for m in v) for k, v in c["families"].items()},
            final_eligible=tuple(str(x) for x in c["final_eligible"]),
            seeds=tuple(int(cfg["seed"]) + i for i in range(int(proto["seeds"]))),
            hybrid_alpha=float(proto["hybrid_alpha"]),
            alpha_curve=tuple(float(a) for a in proto["alpha_curve"]),
            k_range=(int(feas["k_range"][0]), int(feas["k_range"][1])),
            min_cluster_share=float(feas["min_cluster_share"]),
            min_cluster_share_fallback=0.01,
            max_cluster_share=float(feas["max_cluster_share"]),
            max_noise_share=float(feas["max_noise_share"]),
            directions={str(k): str(v) for k, v in sel["criteria"].items()},
            tie={str(k): float(v) for k, v in sel["tie"].items()},
            rule=str(sel["rule"]),
            simplicity={str(k): float(v) for k, v in sel["simplicity"].items()},
            aggregators=tuple(str(a) for a in sens.get("aggregators") or ()),
            sets={str(k): tuple(str(x) for x in v) for k, v in (sens.get("sets") or {}).items()},
            tie_scale=tuple(float(x) for x in sens.get("tie_scale") or ()),
            joint=bool(sens.get("joint", False)),
            all_eligible=bool(sens.get("all_eligible", False)),
            variants=tuple(variants),
            bootstrap=int(stab["bootstrap"]),
            bootstrap_variants=int(stab["bootstrap_variants"]),
            subsample=float(stab["subsample"]),
            jaccard_stable=float(stab["jaccard_stable"]),
            jaccard_dissolved=float(stab["jaccard_dissolved"]),
            tree_depth=int(interp["tree_depth"]),
            cv_folds=int(interp["cv_folds"]),
            tree_inputs=tuple(str(x) for x in interp["inputs"]),
            validation=dict(c["validation"]),
            icvi_metrics={str(k): dict(v) for k, v in (icvi.get("metrics") or {}).items()},
            icvi_permutations=int(icvi.get("baseline_permutations", 200)),
            icvi_section=dict(icvi),
            impl=impl,
            synthetic=dict(c.get("synthetic") or {}),
            report=str(c.get("report", "")),
            report_images=str(c.get("report_images", "docs/img/cluster")),
            seed=int(cfg["seed"]),
            workers=workers,
        )
        p.check()
        return p

    def check(self) -> None:
        problems = []
        unknown = sorted(set(self.methods) - set(METHODS))
        if unknown:
            problems.append(f"families: неизвестные методы {unknown}; допустимы {list(METHODS)}")
        bad_fam = sorted(set(self.final_eligible) - set(self.families))
        if bad_fam:
            problems.append(f"final_eligible: нет семейств {bad_fam}")
        if sorted(self.directions) != sorted(CRITERIA):
            problems.append(f"selection.criteria: нужны ровно {list(CRITERIA)}")
        if self.rule != "pareto_copeland":
            problems.append(f"selection.rule = {self.rule!r}: реализовано только pareto_copeland")
        no_simple = sorted(set(self.methods) - set(self.simplicity))
        if no_simple:
            problems.append(f"selection.simplicity: нет ранга у {no_simple}")
        for name, crit in self.sets.items():
            bad = sorted(set(crit) - set(CRITERIA))
            if bad or not crit:
                problems.append(f"sensitivity.sets.{name}: {bad}")
        bad_agg = sorted(set(self.aggregators) - set(AGGREGATORS))
        if bad_agg:
            problems.append(f"sensitivity.aggregators: {bad_agg}; допустимы {list(AGGREGATORS)}")
        if self.scale != "robust" or self.impute != "region_median":
            problems.append("inputs: реализованы scale = robust и impute = region_median")
        if str(self.impl.get("undefined_metric", "skip")) != "skip":
            problems.append("impl.undefined_metric: реализовано skip («NaN → худший ранг» — проверка)")
        if not 0 < self.subsample < 1:
            problems.append(f"stability.subsample = {self.subsample}: нужно (0; 1)")
        spaces = {s.get("space") for s in self.icvi_metrics.values()}
        if not {"features", "graph"} <= spaces:
            problems.append("icvi.metrics: нужны метрики обоих пространств (features, graph)")
        if problems:
            raise ValueError("clustering: " + "; ".join(problems))
