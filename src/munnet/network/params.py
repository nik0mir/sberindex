"""Параметры этапа network из секции ``network`` конфига (и ``dynamics`` для скользящих окон)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from munnet.config import Config
from munnet.contracts import CATEGORY_CODES, N_MONTHS, NODE_MODES

RULE_KINDS: tuple[str, ...] = (
    "basket_cosine",
    "basket_distance",
    "rhythm_corr",
    "rhythm_lag",
    "rhythm_dtw",
    "road",
    "gravity",
)
BASKET_KINDS: tuple[str, ...] = ("basket_cosine", "basket_distance")
SERIES_KINDS: tuple[str, ...] = ("rhythm_corr", "rhythm_lag", "rhythm_dtw")
GEO_KINDS: tuple[str, ...] = ("road", "gravity")
SPARSIFY: tuple[str, ...] = ("knn", "mutual_knn", "threshold")
CRITERIA: tuple[str, ...] = ("reliability", "geo_difference", "attribute_consistency", "simplicity")
WINDOW_KINDS_NET: tuple[str, ...] = ("quarter", "half", "rolling")


@dataclass(frozen=True)
class RuleSpec:
    name: str
    kind: str
    categories: tuple[str, ...]
    max_lag: int = 0
    window: int = 0
    beta: float = 2.0
    min_km: float = 1.0

    @property
    def is_series(self) -> bool:
        return self.kind in SERIES_KINDS

    @property
    def is_basket(self) -> bool:
        return self.kind in BASKET_KINDS

    @property
    def is_geo(self) -> bool:
        return self.kind in GEO_KINDS

    @property
    def reach(self) -> int:
        """Досягаемость правила во времени (для нулевой модели): лаг L, окно DTW w, у корреляции 0."""
        return self.max_lag if self.kind == "rhythm_lag" else self.window if self.kind == "rhythm_dtw" else 0


@dataclass(frozen=True)
class NetworkParams:
    rules: dict[str, RuleSpec]
    city_distance: str
    spearman_check: bool
    method: str
    k: int
    k_grid: tuple[int, ...]
    fdr_q: float
    shift_repeats: int
    shift_guard: int
    rewire_repeats: int
    rewire_factor: int
    mantel_permutations: int
    naive_r: float
    bootstrap: int
    resolution: float
    seeds: tuple[int, ...]
    attr_year: int
    attr_columns: tuple[str, ...]
    attr_relative: tuple[str, ...]
    priority: tuple[str, ...]
    candidates: tuple[str, ...]
    simplicity: Mapping[str, float]
    modes: tuple[str, ...]
    dyn_kinds: tuple[str, ...]
    dyn_main: str
    dyn_lengths: tuple[int, ...]
    min_series_months: int
    rolling_months: int
    rolling_step: int
    report: str
    report_images: str
    seed: int

    @classmethod
    def from_config(cls, cfg: Config) -> NetworkParams:
        net: Mapping[str, Any] = cfg["network"]
        default_cats = tuple(str(c) for c in net.get("rhythm_categories") or ("all",))
        problems: list[str] = []
        rules: dict[str, RuleSpec] = {}
        for name, spec in (net.get("rules") or {}).items():
            kind = str(spec.get("kind"))
            if kind not in RULE_KINDS:
                problems.append(f"rules.{name}.kind = {kind!r}: допустимы {list(RULE_KINDS)}")
                continue
            cats = tuple(str(c) for c in spec.get("categories") or default_cats)
            bad = sorted(set(cats) - set(CATEGORY_CODES))
            if bad:
                problems.append(f"rules.{name}.categories: {bad}; допустимы {list(CATEGORY_CODES)}")
            rules[str(name)] = RuleSpec(
                name=str(name),
                kind=kind,
                categories=cats,
                max_lag=int(spec.get("max_lag", 0)),
                window=int(spec.get("window", 0)),
                beta=float(spec.get("beta", 2.0)),
                min_km=float(spec.get("min_km", 1.0)),
            )
        if not any(r.kind == "road" for r in rules.values()):
            problems.append("rules: нужна контрольная сеть вида road (отличие от географии)")
        sp, null, probe = net["sparsify"], net["nulls"], net["probe"]
        attrs, sel, dyn = net["attributes"], net["selection"], net["dynamics"]
        top_dyn = cfg.data.get("dynamics") or {}
        p = cls(
            rules=rules,
            city_distance=str(net.get("city_distance", "pop_mean")),
            spearman_check=bool(net.get("spearman_check", True)),
            method=str(sp["method"]),
            k=int(sp["k"]),
            k_grid=tuple(int(x) for x in sp["k_grid"]),
            fdr_q=float(sp["fdr_q"]),
            shift_repeats=int(null["shift_repeats"]),
            shift_guard=int(null["shift_guard"]),
            rewire_repeats=int(null["rewire_repeats"]),
            rewire_factor=int(null["rewire_factor"]),
            mantel_permutations=int(null["mantel_permutations"]),
            naive_r=float(null["naive_r"]),
            bootstrap=int(net["bootstrap"]),
            resolution=float(probe["resolution"]),
            seeds=tuple(int(cfg["seed"]) + i for i in range(int(probe["seeds"]))),
            attr_year=int(attrs["year"]),
            attr_columns=tuple(str(c) for c in attrs["columns"]),
            attr_relative=tuple(str(c) for c in attrs.get("relative") or ()),
            priority=tuple(str(c) for c in sel["priority"]),
            candidates=tuple(str(c) for c in sel["candidates"]),
            simplicity={str(k): float(v) for k, v in (sel.get("simplicity") or {}).items()},
            modes=tuple(str(m) for m in net.get("modes") or ()),
            dyn_kinds=tuple(str(k) for k in dyn["kinds"]),
            dyn_main=str(dyn["main_kind"]),
            dyn_lengths=tuple(int(x) for x in dyn["lengths"]),
            min_series_months=int(dyn.get("min_series_months", 12)),
            rolling_months=int(top_dyn.get("window_months", 12)),
            rolling_step=int(top_dyn.get("step_months", 1)),
            report=str(net.get("report", "docs/network.md")),
            report_images=str(net.get("report_images", "docs/img/network")),
            seed=int(cfg["seed"]),
        )
        if p.method not in SPARSIFY:
            problems.append(f"sparsify.method = {p.method!r}: допустимы {list(SPARSIFY)}")
        if p.k not in p.k_grid:
            problems.append(f"sparsify.k = {p.k} должно входить в k_grid {list(p.k_grid)}")
        if sorted(p.priority) != sorted(CRITERIA):
            problems.append(f"selection.priority: нужны ровно {list(CRITERIA)}")
        unknown = sorted(set(p.candidates) - set(rules))
        if unknown or not p.candidates:
            problems.append(f"selection.candidates: нет правил {unknown}")
        no_simple = sorted(set(p.candidates) - set(p.simplicity))
        if no_simple:
            problems.append(f"selection.simplicity: нет ранга у {no_simple}")
        bad_modes = sorted(set(p.modes) - set(NODE_MODES))
        if bad_modes:
            problems.append(f"modes: {bad_modes}; допустимы {list(NODE_MODES)}")
        bad_kinds = sorted(set(p.dyn_kinds) - set(WINDOW_KINDS_NET))
        if bad_kinds or p.dyn_main not in p.dyn_kinds:
            problems.append(f"dynamics.kinds {list(p.dyn_kinds)}, main_kind {p.dyn_main!r}")
        if not 1 <= p.rolling_months <= N_MONTHS or p.rolling_step < 1:
            problems.append("dynamics.window_months / step_months вне периода панели")
        if not 0 < p.fdr_q < 1:
            problems.append(f"sparsify.fdr_q = {p.fdr_q}: нужно (0; 1)")
        if problems:
            raise ValueError("network: " + "; ".join(problems))
        return p
