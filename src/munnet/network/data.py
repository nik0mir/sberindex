"""Узлы сети и их данные: траты частей корзины по месяцам, свой ритм, экономика места, население.

Основной режим узлов читается из выходов этапа features (``features_*.parquet``). Другие режимы
(``network.modes``, например ``separate`` — 247 районов Москвы и Петербурга отдельными узлами) собираются
теми же функциями ``munnet.nodes`` и ``munnet.features`` в памяти: так сравнение режимов не зависит
от того, с каким режимом запускался этап features.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from munnet import features, nodes
from munnet.config import Config
from munnet.contracts import (
    CONTEXT_ANNUAL,
    FEATURES_MEMBERS,
    FEATURES_NODES,
    FEATURES_PLACE,
    FEATURES_RHYTHM_MONTHLY,
    FEATURES_WINDOWS,
    N_MONTHS,
    PARTS,
    MissingInputError,
    read_table,
)
from munnet.network.graph import NodeInfo
from munnet.network.params import NetworkParams

log = logging.getLogger(__name__)

FEATURES_HINT = "сначала запустите этапы panel и features: python -m munnet panel features"


@dataclass(frozen=True)
class NodeSet:
    """Узлы сети в порядке номеров 0…n−1 (по возрастанию ``territory_id``) и их данные."""

    mode: str
    table: pd.DataFrame  # строки features_nodes узлов сети
    V: np.ndarray  # n × 24 × 6: траты шести частей корзины в месяц, ₽ на жителя
    own: dict[str, np.ndarray]  # категория -> n × 24: свой ритм
    raw: np.ndarray  # n × 24: ln трат «Все категории» без обработки (для сравнения «до и после»)
    attrs: pd.DataFrame  # атрибуты места (индекс 0…n−1)
    pop: np.ndarray  # население (среднее по годам контекста)
    members: pd.DataFrame  # МО -> узел
    mo_pop: pd.Series  # население МО панели (веса расстояния узла-города)
    excluded: pd.DataFrame  # territory_id, name, reason — узлы, не вошедшие в сеть
    months: pd.DataFrame  # t -> date, year, month
    clr_floor: float
    region_center: str

    @property
    def ids(self) -> np.ndarray:
        return self.table["territory_id"].to_numpy(dtype=np.int64)

    @property
    def n(self) -> int:
        return len(self.table)

    @property
    def groups(self) -> np.ndarray:
        return self.table["region_group"].to_numpy(dtype=np.int64)

    @property
    def info(self) -> NodeInfo:
        return NodeInfo(
            ids=self.ids,
            region=self.groups,
            lat=self.table["point_lat"].to_numpy(dtype=np.float64),
            lon=self.table["point_lon"].to_numpy(dtype=np.float64),
            attrs=self.attrs,
        )

    @property
    def city_pos(self) -> dict[int, int]:
        """territory_id узла-города -> номер узла."""
        t = self.table
        return {
            int(tid): int(i)
            for i, tid in zip(t.index, t["territory_id"], strict=True)
            if t.loc[i, "is_city_node"]
        }

    def series(self, categories: tuple[str, ...], months: list[int] | np.ndarray) -> list[np.ndarray]:
        """Свой ритм выбранных категорий, столбцы — месяцы ``months`` (номера t)."""
        return [self.own[c][:, months] for c in categories]


# --- Корзина за окно -----------------------------------------------------------------------------


def window_clr(ns: NodeSet, months: list[int] | np.ndarray, relative: bool = True) -> np.ndarray:
    """Корзина узлов за окно ``months`` (номера t; повторы — для бутстрапа): суммы трат частей → доли →
    CLR (мультипликативная замена долей < ``features.clr_floor``) → минус центр группы региона (как
    ``features.window_features``: окна года, полугодия и квартала совпадают с ``features_windows``)."""
    sums = ns.V[:, np.asarray(months), :].sum(axis=1)
    shares = pd.DataFrame(sums / sums.sum(axis=1, keepdims=True), index=ns.ids, columns=list(PARTS))
    clr, _ = features.clr_frame(shares, ns.clr_floor)
    if not relative:
        return clr.to_numpy()
    grp = pd.Series(ns.groups, index=ns.ids)
    return features.relative_clr(clr, grp, ns.region_center).to_numpy()


# --- Сборка --------------------------------------------------------------------------------------


def _cube(monthly: pd.DataFrame, ids: np.ndarray, value: str, categories) -> dict[str, np.ndarray]:
    out = {}
    for c in categories:
        d = monthly.loc[monthly["category"].astype(str) == c]
        m = d.pivot(index="territory_id", columns="t", values=value).reindex(
            index=ids, columns=range(N_MONTHS)
        )
        if m.isna().any().any():
            raise ValueError(f"network: у узлов пропуски ряда {value} категории {c}")
        out[c] = m.to_numpy(dtype=np.float64)
    return out


def check_columns(p: NetworkParams) -> list[str]:
    """Проверочные колонки атрибутов: признаки ``attributes.check`` и их остатки на ``check_residual_on``."""
    return [c for col in p.check_columns for c in (col, f"{col}_resid" if p.check_residual_on else None) if c]


def _attributes(place: pd.DataFrame, ids: np.ndarray, p: NetworkParams, center: str) -> pd.DataFrame:
    pl = place.loc[place["year"] == p.attr_year].set_index("territory_id").reindex(ids)
    out = pd.DataFrame(index=ids)
    for col in p.attr_columns:
        if col not in pl.columns:
            raise ValueError(f"network.attributes: нет признака {col} в features_place")
        v = pl[col].astype("float64")
        if col in p.attr_relative:
            v = features.relative_values(v, pl["region_group"], center)
        out[col] = v
    for col in p.check_columns:  # проверочные признаки вне пространства атрибутов (как есть)
        if col not in pl.columns:
            raise ValueError(f"network.attributes.check: нет признака {col} в features_place")
        out[col] = pl[col].astype("float64")
        base = p.check_residual_on
        if base:  # то же сверх атрибута base: остаток МНК на узлах, где есть оба признака
            y, x = out[col], out[base] if base in out.columns else pl[base].astype("float64")
            ok = y.notna() & x.notna()
            slope, icpt = np.polyfit(x[ok], y[ok], 1)
            out[f"{col}_resid"] = (y - (icpt + slope * x)).where(ok)
    out["north"] = pl["north"].astype("float64")
    return out.reset_index(drop=True)


def build_nodeset(
    mode: str,
    node_table: pd.DataFrame,
    members: pd.DataFrame,
    monthly: pd.DataFrame,
    place: pd.DataFrame,
    context_annual: pd.DataFrame,
    p: NetworkParams,
    fp: features.FeatureParams,
) -> NodeSet:
    """Узлы сети — ``is_node`` таблицы узлов с полным рядом своего ритма; остальные — в ``excluded``."""
    have = set(monthly["territory_id"].unique())
    table = node_table.loc[node_table["is_node"].astype(bool) & node_table["territory_id"].isin(have)]
    table = table.sort_values("territory_id").reset_index(drop=True)
    ids = table["territory_id"].to_numpy(dtype=np.int64)
    rest = node_table.loc[~node_table["territory_id"].isin(ids)]
    reason = (
        rest["drop_reason"]
        .astype("object")
        .where(rest["drop_reason"].notna(), "нет полного ряда своего ритма")
    )
    excluded = pd.DataFrame(
        {
            "territory_id": rest["territory_id"].astype("int32"),
            "name": rest["name"].astype(str),
            "reason": reason,
        }
    ).reset_index(drop=True)
    vals = _cube(monthly, ids, "log_value", PARTS)
    V = np.stack([np.exp(vals[c]) for c in PARTS], axis=2)
    cats = sorted({c for r in p.rules.values() for c in r.categories} | {"all"})
    own = _cube(monthly, ids, "own", cats)
    raw = _cube(monthly, ids, "log_value", ["all"])["all"]
    pl = place.loc[place["territory_id"].isin(ids)]
    pop = pl.groupby("territory_id")["pop_avg"].mean().reindex(ids).to_numpy(dtype=np.float64)
    mo_pop = context_annual.groupby("territory_id")["pop_avg"].mean()
    months = monthly.drop_duplicates("t").set_index("t")[["date", "year", "month"]].sort_index().reset_index()
    return NodeSet(
        mode=mode,
        table=table,
        V=V,
        own=own,
        raw=raw,
        attrs=_attributes(place, ids, p, fp.place_center),
        pop=pop,
        members=members[["territory_id", "node_id", "role"]].copy(),
        mo_pop=mo_pop,
        excluded=excluded,
        months=months,
        clr_floor=fp.clr_floor,
        region_center=fp.region_center,
    )


def _read(processed: Path, name: str, schema):
    path = processed / f"{name}.parquet"
    if not path.exists():
        raise MissingInputError(f"нет {path}: {FEATURES_HINT}")
    return read_table(path, schema)


def load_main(cfg: Config, p: NetworkParams) -> tuple[NodeSet, pd.DataFrame]:
    """Узлы основного режима из выходов features; второй результат — ``features_windows`` (сверка корзины)."""
    processed = Path(cfg["paths"]["processed"])
    fp = features.FeatureParams.from_config(cfg)
    ns = build_nodeset(
        str(cfg["nodes"]["mode"]),
        _read(processed, "features_nodes", FEATURES_NODES),
        _read(processed, "features_members", FEATURES_MEMBERS),
        _read(processed, "features_rhythm_monthly", FEATURES_RHYTHM_MONTHLY),
        _read(processed, "features_place", FEATURES_PLACE),
        _read(processed, "context_annual", CONTEXT_ANNUAL),
        p,
        fp,
    )
    return ns, _read(processed, "features_windows", FEATURES_WINDOWS)


def load_mode(cfg: Config, p: NetworkParams, mode: str) -> NodeSet:
    """Узлы другого режима ``nodes.mode``: те же функции узлов и признаков, что у этапов features."""
    data = copy.deepcopy(cfg.data)
    data["nodes"]["mode"] = mode
    cfg_m = Config(data=data, path=cfg.path)
    fp = features.FeatureParams.from_config(cfg_m)
    fp = features.FeatureParams(
        **{
            **fp.__dict__,
            "rhythm_categories": tuple(
                sorted({c for r in p.rules.values() for c in r.categories} | {"all", *PARTS})
            ),
        }
    )
    nd = nodes.load_node_data(cfg_m)
    table = features.select_nodes(nd.nodes, fp.min_months)
    ids = pd.Index(np.sort(table.loc[table["is_node"], "territory_id"].to_numpy()), name="territory_id")
    wide = nd.panel_wide.loc[nd.panel_wide["territory_id"].isin(ids)].reset_index(drop=True)
    rng = np.random.default_rng([fp.seed, 2])
    rhythm = features.rhythm_features(wide, table, fp, rng)
    place = features.place_features(nd.context_annual, table, ids, fp)
    ctx = read_table(Path(cfg["paths"]["processed"]) / "context_annual.parquet", CONTEXT_ANNUAL)
    log.info("network: режим узлов %s — %d узлов сети", mode, len(ids))
    return build_nodeset(mode, table, nd.members, rhythm.monthly, place, ctx, p, fp)
