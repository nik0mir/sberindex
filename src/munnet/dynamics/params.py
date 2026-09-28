"""Параметры этапа dynamics: предрегистрация ``dynamics.tracking`` (не меняется) и реализация
``dynamics.impl``."""

from __future__ import annotations

from dataclasses import dataclass

from munnet.config import Config

MAIN = "refit_match"
FIXED = "fixed_prototypes"
EVOLUTIONARY = "evolutionary"
APPROACHES: tuple[str, ...] = (MAIN, FIXED, EVOLUTIONARY)


@dataclass(frozen=True)
class DynParams:
    seed: int
    windows: str
    place_year: int
    main: str
    alternatives: tuple[str, ...]
    events_jaccard: float
    events_sensitivity: tuple[float, ...]
    change: str
    noise: str
    change_rule: str
    bootstrap: int
    node_transition: str
    driver: str
    level: float
    driver_alpha: float
    final_ari_min: float
    epsilon: float
    epsilon_grid: tuple[float, ...]
    evo_max_iter: int
    map_flow_min: int
    top_regions: int
    examples: int
    report: str
    report_images: str

    @property
    def taus(self) -> tuple[float, ...]:
        return tuple(sorted({self.events_jaccard, *self.events_sensitivity}))

    @property
    def driver_part(self) -> str:
        return self.driver.removeprefix("clr_rel_")

    @classmethod
    def from_config(cls, cfg: Config) -> DynParams:
        d = cfg["dynamics"]
        tr, im = d["tracking"], d["impl"]
        evo = im["evolutionary"]
        if str(im.get("noise_mean", "years")) != "years":  # реализовано только среднее двух лет
            raise ValueError("dynamics.impl.noise_mean: реализовано years")
        p = cls(
            seed=int(cfg["seed"]),
            windows=str(tr["windows"]),
            place_year=int(tr["place_year"]),
            main=str(tr["main"]),
            alternatives=tuple(str(a) for a in tr["alternatives"]),
            events_jaccard=float(tr["events_jaccard"]),
            events_sensitivity=tuple(float(x) for x in im["events_sensitivity"]),
            change=str(tr["change"]),
            noise=str(tr["noise"]),
            change_rule=str(tr["change_rule"]),
            bootstrap=int(tr["bootstrap"]),
            node_transition=str(tr["node_transition"]),
            driver=str(tr["driver"]),
            level=float(im["level"]),
            driver_alpha=float(im["driver_alpha"]),
            final_ari_min=float(im["final_ari_min"]),
            epsilon=float(evo["epsilon"]),
            epsilon_grid=tuple(float(x) for x in evo["epsilon_grid"]),
            evo_max_iter=int(evo["max_iter"]),
            map_flow_min=int(im["map_flow_min"]),
            top_regions=int(im["top_regions"]),
            examples=int(im["examples"]),
            report=str(d["report"]),
            report_images=str(d["report_images"]),
        )
        p.check()
        return p

    def check(self) -> None:
        """Реализованы ровно предрегистрированные варианты; иное — ошибка конфига, а не тихая подмена."""
        expected = {
            "windows": (self.windows, "rolling"),
            "main": (self.main, MAIN),
            "change": (self.change, "first_last"),
            "noise": (self.noise, "odd_even"),
            "change_rule": (self.change_rule, "paired_bootstrap"),
            "node_transition": (self.node_transition, "stable_halves"),
        }
        bad = [f"{k}={v!r} (реализовано {e!r})" for k, (v, e) in expected.items() if v != e]
        if set(self.alternatives) != {FIXED, EVOLUTIONARY}:
            bad.append(f"alternatives={list(self.alternatives)}")
        if not self.driver.startswith("clr_rel_"):
            bad.append(f"driver={self.driver!r}")
        if not 0 <= self.epsilon <= 1 or not all(0 <= e <= 1 for e in self.epsilon_grid):
            bad.append("impl.evolutionary.epsilon вне [0, 1]")
        if self.bootstrap < 1:
            bad.append("bootstrap < 1")
        if bad:
            raise ValueError("dynamics: " + "; ".join(bad))
