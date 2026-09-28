"""Входы кластеризации — одни для всех методов: узлы, признаки X, координаты корзины и граф G.

- **Узлы** — узлы сети основного режима (``features_nodes.is_node``), порядок — по ``territory_id``.
- **X** — ``features.space.attributes``: уровень трат ``log_level_rel`` за 24 месяца (сумма трат за период,
  логарифм, минус центр группы региона — формула ``features.window_features``, окно — весь период, как корзина
  рёбер) и признаки места за ``inputs.place_year`` (``features_place``). Пропуск — медиана группы региона
  (``impute: region_median``), затем ``(x − медиана) / (MAD × 1,4826)`` по узлам (``scale: robust``).
- **Корзина** — шесть ``clr_rel_*`` за 24 месяца (``network.data.window_clr``): те же координаты, из которых
  построен G; нужны базовой линии ``kmeans_joint`` и дереву объяснимости.
- **G** — рёбра ``network_edges`` правила ``inputs.graph`` с весом ``weight`` = S_ij; матрица симметрична,
  диагональ нулевая.

Варианты чувствительности (``selection.sensitivity.variants``) собираются теми же функциями: другая сеть
из ``network_edges``, X без признака, режим узлов ``separate`` — в памяти функциями этапов features и network
(как ``network.data.load_mode``), основные выходы ``data/processed`` не трогаются.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from munnet import features, nodes
from munnet.config import Config
from munnet.contracts import (
    CONTEXT_ANNUAL,
    FEATURES_PLACE,
    FEATURES_WINDOWS,
    N_MONTHS,
    NETWORK_EDGES,
    PARTS,
    MissingInputError,
    QCError,
    read_table,
)
from munnet.network import graph as NG
from munnet.network import rules as R
from munnet.network.data import NodeSet, build_nodeset, load_main, window_clr
from munnet.network.params import NetworkParams

log = logging.getLogger(__name__)

MAD_SCALE = 1.4826
LEVEL = "log_level_rel"
QC_TOL = 1e-9
NETWORK_HINT = "сначала запустите этапы panel features network: python -m munnet panel features network"


@dataclass(frozen=True)
class Inputs:
    """Атрибутированная сеть (G, X) на n узлах; строки всех матриц — узлы в порядке ``ids``."""

    name: str
    ids: np.ndarray  # territory_id, int64
    X: np.ndarray  # n × p: признаки после пропусков и масштаба
    x_names: tuple[str, ...]
    X_raw: pd.DataFrame  # признаки как есть (с пропусками), индекс — territory_id
    B: np.ndarray  # n × 6: корзина относительно группы региона за 24 месяца
    b_names: tuple[str, ...]
    A: sp.csr_matrix  # n × n: веса рёбер G, симметрична, диагональ 0
    groups: np.ndarray  # группа региона
    table: pd.DataFrame  # строки features_nodes узлов (название, регион, узел-город)
    imputed: dict[str, int] = field(default_factory=dict)  # признак -> число заменённых пропусков
    scale_used: dict[str, str] = field(default_factory=dict)  # признак -> «mad» или «sd» (MAD = 0)

    @property
    def n(self) -> int:
        return len(self.ids)

    @property
    def tree_matrix(self) -> np.ndarray:
        """17 признаков дерева объяснимости: корзина ⊕ X."""
        return np.hstack([self.B, self.X])

    @property
    def tree_names(self) -> tuple[str, ...]:
        return (*self.b_names, *self.x_names)

    def subset(self, idx: np.ndarray) -> Inputs:
        """Подвыборка узлов (бутстрап): X и корзина — строки, G — индуцированный подграф."""
        idx = np.asarray(idx)
        return replace(
            self,
            ids=self.ids[idx],
            X=self.X[idx],
            X_raw=self.X_raw.iloc[idx],
            B=self.B[idx],
            A=self.A[idx][:, idx].tocsr(),
            groups=self.groups[idx],
            table=self.table.iloc[idx].reset_index(drop=True),
        )


# --- Признаки ------------------------------------------------------------------------------------


def window_level(ns: NodeSet, months) -> np.ndarray:
    """``log_level_rel`` за окно ``months``: траты «Все категории» (сумма шести частей) за окно / число
    месяцев → логарифм → минус центр группы региона (``features.relative_values``, центр
    ``features.region_center``) — как ``features.window_features`` для года, полугодия и квартала."""
    months = np.asarray(months)
    level = ns.V[:, months, :].sum(axis=(1, 2)) / len(months)
    v = pd.Series(np.log(level), index=ns.ids)
    grp = pd.Series(ns.groups, index=ns.ids)
    return features.relative_values(v, grp, ns.region_center).to_numpy()


def check_level(ns: NodeSet, windows: pd.DataFrame) -> float:
    """Сверка: уровень годовых окон по той же функции совпадает с ``features_windows`` (QCError иначе)."""
    worst = 0.0
    for year in (2023, 2024):
        months = [t for t in range(N_MONTHS) if 2023 + t // 12 == year]
        ref = windows.loc[windows["window"] == str(year)].set_index("territory_id").reindex(ns.ids)[LEVEL]
        worst = max(worst, float(np.abs(window_level(ns, months) - ref.to_numpy()).max()))
    if not worst <= QC_TOL:
        raise QCError(f"cluster: уровень трат окна расходится с features_windows на {worst:.3g}")
    return worst


def impute_region_median(frame: pd.DataFrame, groups: np.ndarray) -> tuple[pd.DataFrame, dict[str, int]]:
    """Пропуск — медиана признака в своей группе региона; в группе нет значений — медиана по всем узлам."""
    out = frame.copy()
    counts = {}
    g = pd.Series(groups, index=frame.index)
    for col in frame.columns:
        v = frame[col].astype("float64")
        miss = v.isna()
        counts[col] = int(miss.sum())
        if not miss.any():
            continue
        med = v.groupby(g).transform("median")
        out[col] = v.fillna(med).fillna(float(v.median()))
    return out, counts


def robust_scale(frame: pd.DataFrame) -> tuple[np.ndarray, dict[str, str]]:
    """(x − медиана) / (MAD × 1,4826); у признака с MAD = 0 — стандартное отклонение (записывается)."""
    cols = []
    used = {}
    for col in frame.columns:
        v = frame[col].to_numpy(dtype=np.float64)
        med = np.median(v)
        mad = np.median(np.abs(v - med)) * MAD_SCALE
        if mad > 0:
            used[col] = "mad"
        else:
            mad = float(np.std(v)) or 1.0
            used[col] = "sd"
        cols.append((v - med) / mad)
    return np.column_stack(cols), used


def build_X(
    raw: pd.DataFrame, groups: np.ndarray
) -> tuple[np.ndarray, pd.DataFrame, dict[str, int], dict[str, str]]:
    filled, counts = impute_region_median(raw, groups)
    X, used = robust_scale(filled)
    return X, filled, counts, used


# --- Граф ----------------------------------------------------------------------------------------


def graph_matrix(edges: pd.DataFrame, ids: np.ndarray, weight: str = "weight") -> sp.csr_matrix:
    """Симметричная разреженная матрица весов по рёбрам ``source``–``target`` (territory_id); рёбра к узлам
    вне ``ids`` отбрасываются."""
    pos = pd.Series(np.arange(len(ids)), index=ids)
    s = edges["source"].map(pos)
    t = edges["target"].map(pos)
    ok = s.notna() & t.notna()
    i, j = s[ok].to_numpy(dtype=np.int64), t[ok].to_numpy(dtype=np.int64)
    w = edges.loc[ok, weight].to_numpy(dtype=np.float64)
    n = len(ids)
    A = sp.coo_matrix((np.r_[w, w], (np.r_[i, j], np.r_[j, i])), shape=(n, n)).tocsr()
    A.setdiag(0)
    A.eliminate_zeros()
    return A


def rule_graph(ns: NodeSet, rule: str, p: NetworkParams, k: int) -> sp.csr_matrix:
    """G правила корзины на всех месяцах, пересчитанный функциями этапа network (для режима узлов, которого
    нет в ``network_edges``)."""
    spec = p.rules[rule]
    X = window_clr(ns, np.arange(N_MONTHS), relative=spec.relative)
    if spec.kind == "basket_distance":
        D = R.euclid_matrix(X)
        S = R.gaussian_similarity(D, R.median_offdiag(D))
    elif spec.kind == "basket_cosine":
        S = R.cosine_matrix(X)
    else:
        raise ValueError(f"cluster: правило {rule} ({spec.kind}) не по корзине")
    e = NG.knn_edges(S, k)
    frame = pd.DataFrame({"source": ns.ids[e["i"]], "target": ns.ids[e["j"]], "weight": e["weight"]})
    return graph_matrix(frame, ns.ids)


def read_edges(processed: Path, rule: str, sparsify: str, k: int) -> pd.DataFrame:
    path = processed / "network_edges.parquet"
    if not path.exists():
        raise MissingInputError(f"нет {path}: {NETWORK_HINT}")
    e = read_table(path, NETWORK_EDGES)
    sel = e.loc[(e["rule"] == rule) & (e["sparsify"].astype(str) == sparsify) & (e["k"] == k)]
    if sel.empty:
        raise MissingInputError(f"в {path} нет рёбер {rule}/{sparsify}/k={k}: {NETWORK_HINT}")
    return sel


# --- Сборка --------------------------------------------------------------------------------------


def _assemble(
    name: str, ns: NodeSet, place: pd.DataFrame, A: sp.csr_matrix, attributes, place_year: int
) -> Inputs:
    ids = ns.ids
    pl = place.loc[place["year"] == place_year].set_index("territory_id").reindex(ids)
    raw = pd.DataFrame(index=pd.Index(ids, name="territory_id"))
    for col in attributes:
        if col == LEVEL:
            raw[col] = window_level(ns, np.arange(N_MONTHS))
        elif col in pl.columns:
            raw[col] = pl[col].astype("float64").to_numpy()
        else:
            raise ValueError(f"cluster: признака {col} нет в features_place")
    X, _, counts, used = build_X(raw, ns.groups)
    B = window_clr(ns, np.arange(N_MONTHS))
    return Inputs(
        name=name,
        ids=ids.astype(np.int64),
        X=X,
        x_names=tuple(attributes),
        X_raw=raw,
        B=B,
        b_names=tuple(f"clr_rel_{q}" for q in PARTS),
        A=A,
        groups=ns.groups.astype(np.int64),
        table=ns.table.reset_index(drop=True),
        imputed=counts,
        scale_used=used,
    )


@dataclass(frozen=True)
class MainData:
    inputs: Inputs
    ns: NodeSet
    place: pd.DataFrame
    qc: dict[str, float]


def load_inputs(cfg: Config, cp) -> MainData:
    """Входы основного режима из выходов этапов features и network; сверки: уровень трат годовых окон
    равен ``features_windows``, G из ``network_edges`` совпадает с пересчитанным функциями этапа network."""
    processed = Path(cfg["paths"]["processed"])
    netp = NetworkParams.from_config(cfg)
    ns, fwin = load_main(cfg, netp)
    qc = {
        "level_max_diff": check_level(
            ns, read_table(processed / "features_windows.parquet", FEATURES_WINDOWS)
        )
    }
    _ = fwin
    place = read_table(processed / "features_place.parquet", FEATURES_PLACE)
    edges = read_edges(processed, cp.graph_rule, cp.graph_sparsify, cp.graph_k)
    A = graph_matrix(edges, ns.ids)
    if cp.graph_sparsify == "knn":
        A2 = rule_graph(ns, cp.graph_rule, netp, cp.graph_k)
        diff = abs(A - A2)
        qc["graph_max_diff"] = float(diff.max()) if diff.nnz else 0.0
        qc["graph_nnz_equal"] = float(A.nnz == A2.nnz)
        if not (qc["graph_max_diff"] <= 1e-6 and A.nnz == A2.nnz):
            raise QCError(
                f"cluster: G из network_edges не совпал с пересчитанным ({qc['graph_max_diff']:.3g})"
            )
    inputs = _assemble("main", ns, place, A, cp.attributes, cp.place_year)
    log.info(
        "cluster: %d узлов, %d рёбер G, признаков X %d; пропуски заменены: %s",
        inputs.n,
        A.nnz // 2,
        len(cp.attributes),
        {k: v for k, v in inputs.imputed.items() if v},
    )
    return MainData(inputs=inputs, ns=ns, place=place, qc=qc)


def load_mode_nodes(cfg: Config, mode: str) -> tuple[NodeSet, pd.DataFrame]:
    """Узлы и признаки места другого режима ``nodes.mode`` в памяти — теми же функциями, что этапы features
    и network (``network.data.load_mode``); отличие — возвращается и ``features_place`` режима."""
    netp = NetworkParams.from_config(cfg)
    data = copy.deepcopy(cfg.data)
    data["nodes"]["mode"] = mode
    cfg_m = Config(data=data, path=cfg.path)
    fp = features.FeatureParams.from_config(cfg_m)
    cats = tuple(sorted({c for r in netp.rules.values() for c in r.categories} | {"all", *PARTS}))
    fp = features.FeatureParams(**{**fp.__dict__, "rhythm_categories": cats})
    nd = nodes.load_node_data(cfg_m)
    table = features.select_nodes(nd.nodes, fp.min_months)
    ids = pd.Index(np.sort(table.loc[table["is_node"], "territory_id"].to_numpy()), name="territory_id")
    wide = nd.panel_wide.loc[nd.panel_wide["territory_id"].isin(ids)].reset_index(drop=True)
    rng = np.random.default_rng([fp.seed, 2])
    rhythm = features.rhythm_features(wide, table, fp, rng)
    place = features.place_features(nd.context_annual, table, ids, fp)
    ctx = read_table(Path(cfg["paths"]["processed"]) / "context_annual.parquet", CONTEXT_ANNUAL)
    ns = build_nodeset(mode, table, nd.members, rhythm.monthly, place, ctx, netp, fp)
    log.info("cluster: режим узлов %s — %d узлов", mode, ns.n)
    return ns, place


def variant_inputs(cfg: Config, cp, main: MainData, variant) -> Inputs:
    """Входы варианта чувствительности: сеть другого правила, X без признаков или другой режим узлов."""
    processed = Path(cfg["paths"]["processed"])
    attributes = tuple(a for a in cp.attributes if a not in variant.drop_features)
    ns, place = main.ns, main.place
    rule = variant.graph_rule or cp.graph_rule
    if variant.nodes_mode and variant.nodes_mode != ns.mode:
        ns, place = load_mode_nodes(cfg, variant.nodes_mode)
        A = rule_graph(ns, rule, NetworkParams.from_config(cfg), cp.graph_k)
    elif variant.graph_rule:
        A = graph_matrix(read_edges(processed, rule, cp.graph_sparsify, cp.graph_k), ns.ids)
    else:
        A = main.inputs.A
    return _assemble(variant.name, ns, place, A, attributes, cp.place_year)


def node_context(cfg: Config, ids: np.ndarray) -> pd.DataFrame:
    """Годовой контекст узлов (узлы-города — по правилам ``munnet.nodes``, как у этапа features)."""
    nd = nodes.load_node_data(cfg)
    ctx = nd.context_annual
    return ctx.loc[ctx["territory_id"].isin(ids)].copy()
