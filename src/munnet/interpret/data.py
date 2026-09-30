"""Входы этапа interpret: итог cluster, выходы dynamics, признаки узлов, контекст Росстата — и слепой режим.

- **Свежесть** (``freshness: strict``): любой файл верхнего уровня ``outputs/evaluate`` и ``outputs/dynamics``
  старше ``data/processed/cluster_labels.parquet`` — стоп (код 1, как записано в предрегистрации).
- **Лестница** (``ladder``): номера типов ``cluster_final`` переносятся на номера записи по размеру;
  другой набор размеров — стоп (код 3). Все номера типов этапа — номера записи; ступень — позиция
  в ``ladder.order``.
- **Слепой прогон** (``--blind SEED``): типы узлов заменены случайной перестановкой по узлам (те же размеры),
  строки узлов dynamics перемешаны по узлам (переход целиком достаётся другому узлу); seed — не общий.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from munnet.clustering import inputs as CI
from munnet.config import Config
from munnet.contracts import (
    CLUSTER_FINAL,
    CONTEXT_ANNUAL,
    MissingInputError,
    QCError,
    read_table,
)
from munnet.interpret.placebo import tag_key
from munnet.interpret.spec import Spec

log = logging.getLogger(__name__)

HINT = "запустите этапы по порядку: python -m munnet cluster evaluate dynamics"
HALF_PERIODS = {
    "odd_2023": "2023, нечётные месяцы",
    "even_2023": "2023, чётные месяцы",
    "odd_2024": "2024, нечётные месяцы",
    "even_2024": "2024, чётные месяцы",
}


def check_freshness(cfg: Config, spec: Spec) -> dict:
    """``freshness: strict`` — выходы evaluate и dynamics не старше меток cluster, иначе MissingInputError."""
    spec.expect("freshness", "strict")
    labels = Path(cfg["paths"]["processed"]) / "cluster_labels.parquet"
    if not labels.exists():
        raise MissingInputError(f"нет {labels}: {HINT}")
    t_lab = labels.stat().st_mtime
    info = {"cluster_labels": t_lab}
    stale = []
    for stage in ("evaluate", "dynamics"):
        d = Path(cfg["paths"]["outputs"]) / stage
        files = [p for p in d.glob("*") if p.is_file()] if d.exists() else []
        if not files:
            raise MissingInputError(f"нет выходов {d}: {HINT}")
        oldest = min(p.stat().st_mtime for p in files)
        info[stage] = oldest
        if oldest < t_lab:
            stale.append(f"{stage} ({min(files, key=lambda p: p.stat().st_mtime).name})")
    if stale:
        raise MissingInputError(
            "interpret.freshness: выходы старше data/processed/cluster_labels.parquet — "
            + ", ".join(stale)
            + f"; {HINT}"
        )
    return info


def inputs_sha(cfg: Config) -> dict[str, str]:
    """sha256 файлов, от которых зависят типы и переходы этапа: метки и итог cluster, выходы dynamics.
    Публикуются в facts.json (qc.inputs_sha256), чтобы читатель выходов сверял их по содержимому, а не по
    времени изменения."""
    import hashlib

    out = Path(cfg["paths"]["outputs"])
    processed = Path(cfg["paths"]["processed"])
    files = {
        "cluster_labels": processed / "cluster_labels.parquet",
        "cluster_final": processed / "cluster_final.parquet",
        "cluster_final_json": out / "cluster" / "final.json",
        "dynamics_nodes": out / "dynamics" / "nodes.csv",
        "dynamics_labels": out / "dynamics" / "labels.csv",
        "dynamics_facts": out / "dynamics" / "facts.json",
    }
    return {k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in files.items() if p.exists()}


def ladder_transfer(types: np.ndarray, spec: Spec) -> tuple[dict[int, int], list[int]]:
    """Перенос номеров типов на номера записи по размеру (``ladder.sizes``); другие размеры — QCError."""
    order = [int(x) for x in spec["order"]]
    sizes = {int(k): int(v) for k, v in spec["sizes"].plain().items()}
    if sorted(order) != sorted(sizes):
        raise ValueError("interpret.ladder: order и sizes — разные номера типов")
    if len(set(sizes.values())) != len(sizes):
        raise ValueError("interpret.ladder.sizes: размеры совпадают — перенос по размеру неоднозначен")
    have = Counter(int(t) for t in types)
    if sorted(have.values()) != sorted(sizes.values()):
        raise QCError(
            f"interpret.ladder: размеры типов cluster_final {dict(sorted(have.items()))} "
            "не совпадают с записью "
            f"{sizes}: нужна новая запись предрегистрации"
        )
    by_size = {v: k for k, v in sizes.items()}
    return {t: by_size[n] for t, n in have.items()}, order


@dataclass
class Data:
    cfg: Config
    fin: dict
    cp: SimpleNamespace
    main: CI.MainData
    final: np.ndarray  # номера записи 1…K по узлам сети (после переноса и слепой перестановки)
    final_raw_numbers: dict[int, int]  # перенос: номер cluster_final -> номер записи
    type_jaccard: dict[int, float]  # номер записи -> Жаккар по Хеннигу
    order: list[int]  # номера типов снизу вверх
    nodes_dyn: pd.DataFrame  # outputs/dynamics/nodes.csv в порядке узлов сети, номера записи
    labels_dyn: dict[tuple[str, str], np.ndarray]  # (способ, период) -> номера записи
    context: pd.DataFrame
    territories: pd.DataFrame
    place: pd.DataFrame  # features_place (все годы)
    city_ids: list[int]
    blind: int | None = None
    fresh: dict = field(default_factory=dict)
    inputs_sha: dict = field(default_factory=dict)  # sha256 файлов-входов (для сверки читателями выходов)
    chain: dict = field(default_factory=dict)  # сверка «цепочка окон против прямого сопоставления» (T2/T3)

    @property
    def ns(self):
        return self.main.ns

    @property
    def base(self) -> CI.Inputs:
        return self.main.inputs

    @property
    def ids(self) -> np.ndarray:
        return self.base.ids

    @property
    def k(self) -> int:
        return len(self.order)

    @property
    def terr(self) -> np.ndarray:
        return ~np.isin(self.ids, self.city_ids)

    def rank_of(self, types: np.ndarray) -> np.ndarray:
        """Ступень 1…K по номеру типа (номера записи)."""
        pos = {t: i + 1 for i, t in enumerate(self.order)}
        return np.array([pos[int(t)] for t in types], dtype=np.int64)


def _perm(n: int, blind: int, *key) -> np.ndarray:
    return np.random.default_rng([int(blind), *tag_key(*key)]).permutation(n)


def load(cfg: Config, spec: Spec, blind: int | None) -> Data:
    fresh = check_freshness(cfg, spec)
    out = Path(cfg["paths"]["outputs"])
    processed = Path(cfg["paths"]["processed"])
    fin_path = out / "cluster" / "final.json"
    if not fin_path.exists():
        raise MissingInputError(f"нет {fin_path}: {HINT}")
    fin = json.loads(fin_path.read_text(encoding="utf-8"))
    g = fin["inputs"]["graph"]
    place_year = int(spec["windows"]["place_year"])
    if (
        int(fin["inputs"]["place_year"]) != place_year
        or int(cfg["dynamics"]["tracking"]["place_year"]) != place_year
    ):
        raise ValueError("interpret.windows.place_year: не совпадает с итогом cluster или dynamics.tracking")
    cp = SimpleNamespace(
        graph_rule=str(g["rule"]),
        graph_sparsify=str(g["sparsify"]),
        graph_k=int(g["k"]),
        attributes=tuple(fin["inputs"]["features"]),
        place_year=place_year,
    )
    main = CI.load_inputs(cfg, cp)
    ids = main.inputs.ids
    fdf = read_table(processed / "cluster_final.parquet", CLUSTER_FINAL).set_index("territory_id")
    raw = fdf.reindex(ids)["type"]
    if raw.isna().any():
        raise QCError("interpret: у узлов сети нет итогового типа в cluster_final")
    raw = raw.to_numpy(dtype=np.int64)
    transfer, order = ladder_transfer(raw, spec["ladder"])
    final = np.array([transfer[int(t)] for t in raw], dtype=np.int64)
    tj = fdf.reindex(ids)["type_jaccard"].to_numpy(dtype=np.float64)
    type_jaccard = {int(transfer[int(t)]): float(tj[raw == t][0]) for t in np.unique(raw)}

    scope = spec["scope"]
    city_ids = [int(x) for x in scope["city_nodes"]]
    table_city = set(
        main.inputs.table.loc[main.inputs.table["is_city_node"].astype(bool), "territory_id"].astype(int)
    )
    if set(city_ids) != table_city:
        raise ValueError(f"interpret.scope.city_nodes {city_ids} ≠ узлы-города сети {sorted(table_city)}")

    nodes_path = out / "dynamics" / "nodes.csv"
    lab_path = out / "dynamics" / "labels.csv"
    for pth in (nodes_path, lab_path):
        if not pth.exists():
            raise MissingInputError(f"нет {pth}: {HINT}")
    nd = pd.read_csv(nodes_path).set_index("territory_id").reindex(ids)
    if nd["type_24m"].isna().any():
        raise QCError("interpret: узлы dynamics не совпадают с узлами сети")
    if not np.array_equal(nd["type_24m"].to_numpy(dtype=np.int64), raw):
        raise QCError(
            "interpret: типы 24 месяцев в outputs/dynamics/nodes.csv ≠ cluster_final (перезапустите dynamics)"
        )
    type_cols = [
        "type_24m",
        "type_2023",
        "type_2024",
        "half_type_2023",
        "half_type_2024",
        "fixed_2023",
        "fixed_2024",
        "evo_2024",
    ]
    for c in type_cols:
        v = nd[c].to_numpy(dtype=np.int64)
        nd[c] = [transfer[int(x)] if x > 0 else 0 for x in v]
    lab = pd.read_csv(lab_path)
    labels_dyn = {}
    for (appr, per), d in lab.groupby(["approach", "period"]):
        v = d.set_index("territory_id").reindex(ids)["type"].to_numpy(dtype=np.int64)
        labels_dyn[(str(appr), str(per))] = np.array([transfer[int(x)] for x in v], dtype=np.int64)

    chain = check_chain_direct(out / "dynamics" / "facts.json", final, labels_dyn, spec, len(order))
    context = read_table(processed / "context_annual.parquet", CONTEXT_ANNUAL)
    territories = pd.read_parquet(processed / "territories.parquet")
    data = Data(
        cfg=cfg,
        fin=fin,
        cp=cp,
        main=main,
        final=final,
        final_raw_numbers=transfer,
        type_jaccard=type_jaccard,
        order=order,
        nodes_dyn=nd.reset_index(),
        labels_dyn=labels_dyn,
        context=context,
        territories=territories,
        place=main.place,
        city_ids=city_ids,
        blind=blind,
        fresh=fresh,
        inputs_sha=inputs_sha(cfg),
        chain=chain,
    )
    if blind is not None:
        apply_blind(data, int(blind))
    return data


def check_chain_direct(facts_path: Path, final: np.ndarray, labels_dyn: dict, spec: Spec, k: int) -> dict:
    """Сверка сверх записи для T2/T3: наблюдаемые типы 2024 года этап dynamics сопоставляет с итогом цепочкой
    окон, а псевдогоды плацебо — напрямую. Сравнение равноценно, только если для окна 2024 года оба пути
    дают одни и те же номера типов у всех узлов. Считается до слепой перестановки (номера типов не
    раскрываются: это число узлов, а не состав типов).

    - ``last_window_differ`` — узлов окна 2024 года, у которых номер по цепочке ≠ номеру при прямом
      сопоставлении с итогом; больше 0 — стоп (QCError) до тяжёлых расчётов;
    - ``chain_direct_nodes_differ`` и ``chain_direct_windows_same`` — из ``outputs/dynamics/facts.json``
      (все 13 окон; промежуточные окна на наблюдаемые переходы не влияют) — публикуются, при > 0 —
      предупреждение.
    """
    from munnet.dynamics import tracking as T
    from munnet.interpret.placebo import MAIN

    if not facts_path.exists():
        raise MissingInputError(f"нет {facts_path}: {HINT}")
    info = json.loads(facts_path.read_text(encoding="utf-8")).get("main_info", {})
    if "chain_direct_nodes_differ" not in info:
        raise QCError(
            f"interpret: в {facts_path} нет main_info.chain_direct_nodes_differ — перезапустите dynamics"
        )
    w24 = "…".join(str(x) for x in spec["windows"]["year_b"])
    if (MAIN, w24) not in labels_dyn:
        raise ValueError(f"interpret.windows: окна {w24} нет в outputs/dynamics/labels.csv")
    chain24 = np.asarray(labels_dyn[(MAIN, w24)], dtype=np.int64) - 1
    direct24 = T.aligned(np.asarray(final, dtype=np.int64) - 1, chain24, k)
    res = {
        "chain_direct_nodes_differ": float(info["chain_direct_nodes_differ"]),
        "chain_direct_windows_same": float(info.get("chain_direct_windows_same", np.nan)),
        "last_window_differ": int((direct24 != chain24).sum()),
    }
    if res["last_window_differ"] > 0:
        raise QCError(
            "interpret: окно 2024 года в outputs/dynamics/labels.csv сопоставлено цепочкой окон иначе, чем "
            f"напрямую с итогом ({res['last_window_differ']} узлов): наблюдение T2/T3 и плацебо (прямое "
            "сопоставление) неравноценны — нужно решение участника до расчётов"
        )
    if res["chain_direct_nodes_differ"] > 0:
        log.warning(
            "interpret: в промежуточных окнах dynamics цепочка и прямое сопоставление расходятся "
            "(%d узло-окон); "
            "окно 2024 года совпадает, наблюдение T2/T3 от этого не зависит",
            int(res["chain_direct_nodes_differ"]),
        )
    return res


def dynamics_perm(n: int, blind: int) -> np.ndarray:
    """Перестановка узлов слепого прогона для выходов dynamics (строки узлов и метки окон)."""
    return _perm(n, blind, "dynamics")


def apply_blind(data: Data, seed: int) -> None:
    """Слепой прогон: типы — перестановка по узлам (размеры те же); строки узлов dynamics и метки окон
    dynamics — одна перестановка по узлам (переход целиком достаётся другому узлу)."""
    n = len(data.final)
    data.final = data.final[_perm(n, seed, "final")]
    p = dynamics_perm(n, seed)
    nd = data.nodes_dyn
    cols = [c for c in nd.columns if c not in ("territory_id", "name", "region", "is_city_node")]
    for c in cols:
        nd[c] = nd[c].to_numpy()[p]
    data.labels_dyn = {k: v[p] for k, v in data.labels_dyn.items()}
    # устойчивость типов (Жаккар по Хеннигу) — тоже перемешана между номерами типов
    types = list(data.type_jaccard)
    tj = [data.type_jaccard[t] for t in types]
    q = _perm(len(types), seed, "type_jaccard")
    data.type_jaccard = {t: tj[int(q[i])] for i, t in enumerate(types)}
    log.warning("interpret: СЛЕПОЙ ПРОГОН — типы и переходы перемешаны по узлам (seed %d)", seed)
