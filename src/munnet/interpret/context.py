"""Подготовка этапа interpret: массивы территориальных узлов, показатели Росстата, страты, соперники типов
(``place_partitions``), тривиальные деления T5 и отрицательные контроли — всё без меток типов, кроме размеров
групп (``sized`` — размеры типов среди территориальных узлов по ``ladder.order``)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from munnet.clustering import inputs as CI
from munnet.interpret import order as O
from munnet.interpret import partitions as P
from munnet.interpret.data import Data
from munnet.interpret.labels import PARTITION_LABELS, feature_label, sized_label
from munnet.interpret.partitions import Partition
from munnet.interpret.spec import Spec, resolve

log = logging.getLogger(__name__)


@dataclass
class Ctx:
    data: Data
    spec: Spec
    pos: np.ndarray  # позиции территориальных узлов в порядке узлов сети
    ids: np.ndarray
    groups: np.ndarray
    region: np.ndarray  # код субъекта
    mo_type: np.ndarray
    capital: np.ndarray
    area: np.ndarray
    xy: np.ndarray
    filled: pd.DataFrame  # X, как его видела кластеризация (после замены пропусков, до масштаба)
    filled_all: (
        pd.DataFrame
    )  # то же по всем узлам сети (строки — узлы сети): база масштаба clustering.inputs.scale
    raw: pd.DataFrame  # X без замены пропусков
    ctx23: pd.DataFrame  # context_annual года place_year по территориальным узлам
    strata: np.ndarray
    sizes: list[int]  # размеры типов среди территориальных узлов по ladder.order
    indicators: dict[str, np.ndarray] = field(default_factory=dict)  # T1: оборот -> значения
    signs: dict[str, int] = field(default_factory=dict)
    rivals: list[Partition] = field(default_factory=list)
    trivial: dict[str, Partition] = field(default_factory=dict)
    region_part: np.ndarray | None = None
    urban_group: np.ndarray | None = None
    place_feats: list[str] = field(default_factory=list)
    attributes: list[str] = field(default_factory=list)
    density: np.ndarray | None = None
    pop23: np.ndarray | None = None

    @property
    def n(self) -> int:
        return len(self.pos)

    def scaled(self, feats: list[str], how: str = "robust") -> np.ndarray:
        """Признаки X территориальных узлов в масштабе clustering.inputs.scale: медиана и MAD × 1,4826
        (или среднее и sd при ``how = zscore``) — по всем узлам сети, как в кластеризации."""
        return P.scale_rows(self.filled_all[feats].reset_index(drop=True), self.pos, how)


def context_values(data: Data, source: str, year: int, ids: np.ndarray) -> np.ndarray:
    table, col = source.split(".", 1)
    if table != "context_annual":
        raise ValueError(f"источник {source!r}: реализован context_annual.<показатель>")
    c = data.context.loc[data.context["year"] == int(year)].set_index("territory_id")
    return c.reindex(ids)[col].to_numpy(dtype=np.float64)


def transform(values: np.ndarray, how: str, groups: np.ndarray) -> np.ndarray:
    if how == "log1p_rel":
        return P.log1p_rel(values, groups)
    if how == "log1p":
        return P.log1p_abs(values)
    raise ValueError(f"преобразование {how!r} не реализовано")


def build(data: Data, spec: Spec) -> Ctx:
    base = data.base
    terr = data.terr
    pos = np.flatnonzero(terr)
    tab = base.table.iloc[pos].reset_index(drop=True)
    ids = base.ids[pos]
    filled_all, _ = CI.impute_region_median(base.X_raw, base.groups)
    pp = spec["place_partitions"]
    pp.expect("nodes", "territorial")
    pp.expect("values", "clustering_x")
    place_year = int(spec["windows"]["place_year"])
    c = data.context.loc[data.context["year"] == place_year].set_index("territory_id").reindex(ids)
    order = data.order
    ft = data.final[pos]
    sizes = [int((ft == t).sum()) for t in order]
    ctx = Ctx(
        data=data,
        spec=spec,
        pos=pos,
        ids=ids,
        groups=base.groups[pos],
        region=tab["region_code"].to_numpy(dtype=np.int64),
        mo_type=tab["mo_type"].astype(str).to_numpy(),
        capital=tab["is_capital"].astype(bool).to_numpy(),
        area=tab["area_km2"].to_numpy(dtype=np.float64),
        xy=tab[["x_aea", "y_aea"]].to_numpy(dtype=np.float64),
        filled=filled_all.iloc[pos].reset_index(drop=True),
        filled_all=filled_all.reset_index(drop=True),
        raw=base.X_raw.iloc[pos].reset_index(drop=True),
        ctx23=c.reset_index(),
        strata=np.full(len(pos), -1),
        sizes=sizes,
    )
    ctx.attributes = [str(x) for x in data.cfg["features"]["space"][str(pp["sized"]["features"])]]
    ctx.place_feats = [str(x) for x in resolve(spec, "interpret.tests.T2_direction.place_tree.features")]
    ctx.pop23 = c["pop_avg"].to_numpy(dtype=np.float64)

    # --- страты (interpret.strata) ---------------------------------------------------------------
    st = spec["strata"]
    st.expect("values", "raw")
    st.expect("nodes", "territorial_both")
    st.expect("within_rank", "average_over_n")
    feats = st["features"].plain()
    both = np.ones(len(pos), dtype=bool)
    for f in feats:
        both &= np.isfinite(ctx.raw[f].to_numpy(dtype=np.float64))
    code = np.zeros(len(pos), dtype=np.int64)
    mult = 1
    for f, q in reversed(list(feats.items())):
        v = np.where(both, ctx.raw[f].to_numpy(dtype=np.float64), np.nan)
        g = P.rank_quantiles(v, ids, int(q))
        code = code + mult * np.where(g >= 0, g, 0)
        mult *= int(q)
    ctx.strata = np.where(both, code, -1)

    # --- показатели T1 -----------------------------------------------------------------------------
    t1 = spec["tests"]["T1_ladder_external"]
    forbidden = {str(x) for x in t1["forbidden"]}
    for name, ind in t1["indicators"].items():
        col = str(ind["source"]).split(".", 1)[1]
        if col in forbidden:
            raise ValueError(f"T1.indicators.{name}: {col} в списке forbidden")
        v = context_values(data, str(ind["source"]), int(ind["year"]), ids)
        ctx.indicators[str(name)] = transform(v, str(ind["transform"]), ctx.groups)
        ctx.signs[str(name)] = int(ind["sign"])

    # --- соперники (place_partitions) -----------------------------------------------------------------
    sized = pp["sized"]
    rivals: list[Partition] = []
    for f in ctx.attributes:
        v = ctx.filled[f].to_numpy(dtype=np.float64)
        for sgn, rev in ((1, False), (-1, True)):
            lab = P.sized_groups(sgn * v, ids, sizes)
            rivals.append(
                Partition(
                    f"sized:{f}:{'-' if rev else '+'}", sized_label(f, rev), lab, "sized", lab, feature=f
                )
            )
    for comp in sized["composite"]:
        if str(comp) != "rank_pop_urban":
            raise ValueError(f"place_partitions.sized.composite: {comp!r} не реализовано")
        rs = P.rank_sum(ctx.filled[["log_pop_rel", "urban_share_rel"]])
        for sgn, rev in ((1, False), (-1, True)):
            lab = P.sized_groups(sgn * rs, ids, sizes)
            rivals.append(
                Partition(
                    f"composite:{comp}:{'-' if rev else '+'}",
                    f"{PARTITION_LABELS[str(comp)]}, {'по убыванию' if rev else 'по возрастанию'}",
                    lab,
                    "sized",
                    lab,
                    feature=str(comp),
                )
            )
    po = place_only_partition(ctx, spec, seed_offset=0, scale="robust")
    rivals.append(po)
    ctx.rivals = rivals
    pp.expect("rival_max", "per_bootstrap")
    pp.expect("best_partition", "point")

    # --- тривиальные деления T5 ------------------------------------------------------------------------
    t5 = spec["tests"]["T5_trivial"]
    trivial: dict[str, Partition] = {}
    for name, d in t5["partitions"].items():
        name = str(name)
        if name == "pop_decile":
            v = context_values(data, str(d["source"]), int(d["year"]), ids)
            lab = P.value_quantiles(v, int(d["groups"]))
        elif name == "urban_group":
            v = context_values(data, str(d["source"]), int(d["year"]), ids)
            lab = P.value_quantiles(v, int(d["groups"]))
            if str(d["missing"]) != "own_group":
                raise ValueError("T5.partitions.urban_group.missing: реализовано own_group")
            ctx.urban_group = lab.copy()
        elif name == "mo_type":
            if str(d["source"]) != "territories.mo_type":
                raise ValueError("T5.partitions.mo_type: реализовано territories.mo_type")
            lab = pd.factorize(ctx.mo_type)[0]
        elif name == "federal_district":
            mapping = resolve(spec, str(d["source"])).plain()
            lab = pd.factorize(P.federal_districts(ctx.region, mapping))[0]
        elif name == "coverage_class":
            if str(d["classes"]) != "missing_count":
                raise ValueError("T5.coverage_class.classes: реализовано missing_count")
            cc = (
                data.context.loc[data.context["year"] == int(d["year"])]
                .set_index("territory_id")
                .reindex(ids)
            )
            # уточнение реализации: границы классов 0 / 1 / 2 и больше записаны в комментарии, ключа нет
            lab = P.coverage_class(cc[[str(x) for x in d["columns"]]])
        elif name == "place_partitions":
            if str(d["source"]) != "interpret.place_partitions.sized":
                raise ValueError("T5.partitions.place_partitions: реализовано place_partitions.sized")
            for r in rivals:
                if r.kind == "sized":
                    trivial[r.name] = r
            continue
        elif name == "place_only":
            if str(d["source"]) != "interpret.place_partitions.place_only":
                raise ValueError("T5.partitions.place_only: реализовано place_partitions.place_only")
            trivial["place_only"] = po
            continue
        else:
            raise ValueError(f"T5.partitions.{name}: не реализовано")
        trivial[name] = Partition(name, PARTITION_LABELS.get(name, name), np.asarray(lab), "trivial")
    ctx.trivial = trivial
    ctx.region_part = pd.factorize(ctx.groups)[0]
    t1p = t1["proxies"]
    dens = t1p["density"]
    if str(dens["source"]) != "pop_avg/area_km2" or str(dens["transform"]) != "log_rel":
        raise ValueError("T1.proxies.density: реализовано pop_avg/area_km2, log_rel")
    pop_d = context_values(data, "context_annual.pop_avg", int(dens["year"]), ids)
    ctx.density = P.density_rel(pop_d, ctx.area, ctx.groups)
    return ctx


def place_only_partition(ctx: Ctx, spec: Spec, seed_offset: int, scale: str) -> Partition:
    """K-means по 10 признакам места (``T4_basket_vs_place.place``) на территориальных узлах."""
    pp = spec["place_partitions"]
    ref = str(pp["place_only"])
    pl = resolve(spec, ref)
    if str(pl["method"]) != "kmeans":
        raise ValueError(f"{ref}.method: реализовано kmeans")
    feats = [str(x) for x in resolve(spec, str(pl["features"]))]
    base_scale = str(pl["scale"])
    use = base_scale if scale == "robust" else scale
    # масштаб — по всем узлам сети, как clustering.inputs.scale; K-means — на территориальных узлах
    Z = ctx.scaled(feats, use)
    n_init = int(resolve(spec, "interpret.tests.T4_basket_vs_place.n_init"))
    seed = int(spec["seed"]) + int(seed_offset)
    lab = P.kmeans_labels(Z, int(pl["k"]), seed, n_init)
    name = "place_only" if (seed_offset == 0 and use == base_scale) else f"place_only[{use},+{seed_offset}]"
    return Partition(name, PARTITION_LABELS["place_only"], lab, "place_only")


def evidence(ctx: Ctx, spec: Spec, seed: int, tag: int) -> O.Evidence:
    """Выборки T1/T5 и соперники на бутстрап-выборках узлов (seed прогона)."""
    t1 = spec["tests"]["T1_ladder_external"]
    t5s = spec["tests"]["T5_trivial"]["stratified"]
    n_boot = int(t1["bootstrap"])
    if int(t5s["bootstrap"]) != n_boot:
        raise ValueError(
            "T1.bootstrap и T5.stratified.bootstrap различаются: реализованы одни выборки на оба"
        )
    rng = np.random.default_rng([seed, 501, tag])
    turn, bb, ba = {}, {}, {}
    for name, v in ctx.indicators.items():
        t = O.make_turnover(name, ctx.signs[name], v, ctx.groups, ctx.strata)
        turn[name] = t
        bb[name], ba[name] = O.make_boots(t, n_boot, rng)
    ev = O.Evidence(turnovers=turn, boots_b=bb, boots_a=ba, rivals=[])
    ev.level = float(t1["ci_level"])
    ev.n_perm_t1_a = int(t1["overall"]["permutations"])
    ev.n_perm_t1_b = int(t1["conditional"]["permutations"])
    ev.n_perm_t5 = int(t5s["permutations"])
    orient = spec["place_partitions"]["orientation"]
    O.build_rivals(ev, ctx.rivals, {"sized": str(orient["sized"]), "place_only": str(orient["place_only"])})
    return ev


def controls_partitions(ctx: Ctx, spec: Spec) -> list[Partition]:
    """Отрицательные контроли (``controls.partitions``): деления без типов вне набора соперников."""
    cs = spec["controls"]
    cs.expect("sizes", "ladder")
    out = []
    for name, d in cs["partitions"].items():
        name = str(name)
        if name == "composites":
            for comp in d:
                feats = [str(x) for x in comp]
                v = ctx.scaled(feats).sum(axis=1)  # масштаб clustering.inputs.scale — по всем узлам сети
                lab = P.sized_groups(v, ctx.ids, ctx.sizes)
                out.append(
                    Partition(
                        "composite:" + "+".join(feats),
                        "сумма: " + ", ".join(feature_label(f).lower() for f in feats),
                        lab,
                        "control",
                    )
                )
            continue
        if "base" in d:
            if str(d["base"]) != "place_only":
                raise ValueError(f"controls.{name}.base: реализовано place_only")
            off = int(d["seed_offset"]) if "seed_offset" in d else 0
            scale = str(d["scale"]) if "scale" in d else "robust"
            p = place_only_partition(ctx, spec, off, scale)
            out.append(
                Partition(name, f"{PARTITION_LABELS['place_only']} ({name})", p.labels, "control_kmeans")
            )
            continue
        if name == "density_rel":
            if (
                str(d["missing"]) != "zero"
                or str(d["transform"]) != "log_rel"
                or str(d["source"]) != "pop_avg/area_km2"
            ):
                raise ValueError(
                    "controls.density_rel: реализовано pop_avg/area_km2, log_rel и missing: zero"
                )
            pop = context_values(ctx.data, "context_annual.pop_avg", int(d["year"]), ctx.ids)
            v = P.density_rel(pop, ctx.area, ctx.groups)
            v = np.where(np.isfinite(v), v, 0.0)
            out.append(
                Partition(name, "плотность населения", P.sized_groups(v, ctx.ids, ctx.sizes), "control")
            )
            continue
        if name == "place_pc1":
            if str(d["scale"]) != "robust" or str(d["method"]) != "pca1":
                raise ValueError("controls.place_pc1: реализовано robust и pca1")
            feats = [str(x) for x in resolve(spec, str(d["features"]))]
            v = P.pca1(ctx.scaled(feats))  # масштаб — по всем узлам сети, PCA — на территориальных узлах
            out.append(
                Partition(
                    name, "первая главная компонента места", P.sized_groups(v, ctx.ids, ctx.sizes), "control"
                )
            )
            continue
        raise ValueError(f"controls.partitions.{name}: не реализовано")
    return out
