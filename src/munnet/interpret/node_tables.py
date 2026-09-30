"""Поузловые выгрузки этапа 5 для лендинга (``docs/landing_spec.md``, §5) и охват фразы ``scope.phrase``.

Всё здесь — описание уже посчитанного, а не проверка: в вердикты, ``thesis_assembly`` и проверки T1–T7
не входит, в отчёте упоминается с подписью «описание, не проверка». Метки вариантов и seed R1 берутся
из прогонов, которые этап уже вычислил (и закэшировал), — модуль ничего не пересчитывает и в
``runs.heavy_closure`` не входит.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from munnet.interpret.partitions import Partition

NOTE = "описание, не проверка"


# --- Охват ------------------------------------------------------------------------------------------


def node_of(territory_ids: np.ndarray, members: pd.DataFrame | None) -> np.ndarray:
    """Узел сети каждого МО по ``features_members`` (``self`` — само МО, ``city_member`` — узел-город);
    МО без строки в ``members`` — своим узлом."""
    ids = np.asarray(territory_ids).astype(np.int64)
    if members is None or members.empty:
        return ids
    m = members.dropna(subset=["node_id"]).set_index("territory_id")["node_id"].astype(np.int64)
    return pd.Series(ids).map(m).fillna(pd.Series(ids)).astype(np.int64).to_numpy()


def untyped_mask(
    territory_ids: np.ndarray, members: pd.DataFrame | None, typed_nodes: np.ndarray
) -> np.ndarray:
    """МО без типа — у которых нет типа ни у самих, ни через свой узел-город (узел МО не среди узлов
    с типом)."""
    return ~np.isin(node_of(territory_ids, members), np.asarray(typed_nodes).astype(np.int64))


# --- T7: сопоставимые территории ----------------------------------------------------------------------


def comparable_table(
    ids: np.ndarray,
    members: Mapping[str, Sequence[np.ndarray]],
    F: np.ndarray,
    xy: np.ndarray,
    product: str,
    sets: Sequence[str] = ("A", "D", "B"),
) -> pd.DataFrame:
    """Наборы T7 каждого МО (строки ``ids`` — МО с известной целью, как в ``external.t7_sets``): член набора,
    его ранг и расстояние. A и D — евклидово расстояние в масштабированной корзине T7 (``T7.basket``,
    ``T7.scale``), B — километры по ``x_aea``, ``y_aea``; ``km`` — километры по карте у любого набора.
    ``product`` — набор продукта по правилу T7 (A или D)."""
    rows = []
    for s in sets:
        is_b = s == "B"
        for i, mem in enumerate(members[s]):
            mem = np.asarray(mem, dtype=np.int64)
            if not len(mem):
                continue
            km = np.sqrt(((xy[mem] - xy[i]) ** 2).sum(axis=1)) / 1000.0
            d = km if is_b else np.sqrt(((F[mem] - F[i]) ** 2).sum(axis=1))
            for r, (j, dj, kj) in enumerate(zip(mem, d, km, strict=True), start=1):
                rows.append(
                    {
                        "territory_id": int(ids[i]),
                        "set": s,
                        "product": s == product,
                        "rank": r,
                        "other_id": int(ids[j]),
                        "distance": float(dj),
                        "unit": "km" if is_b else "basket_scaled",
                        "km": float(kj),  # расстояние по карте для подписи «сопоставимых» (любой набор)
                    }
                )
    cols = ["territory_id", "set", "product", "rank", "other_id", "distance", "unit", "km"]
    return pd.DataFrame(rows, columns=cols)


# --- R1: метки вариантов и seed ------------------------------------------------------------------------


def r1_tables(
    main_ids: np.ndarray,
    main_types: np.ndarray,
    runs: Mapping[str, tuple[str, np.ndarray, np.ndarray]],
    node_of_id: Mapping[int, int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``node_r1`` — тип узла в каждом прогоне R1 после сопоставления с итогом (``runs``: имя -> (вид variant
    или seed, id узлов прогона, метки 1…K в нумерации итога)) рядом с основным типом; узел прогона, которого
    нет среди узлов итога (внутригородское МО варианта nodes_separate), сравнивается с типом своего
    узла-города (``node_of_id``). ``node_seed`` — по узлу и виду прогона: сколько прогонов дали основной
    тип (n_same) из скольких, где узел есть (n_runs)."""
    main = {int(t): int(v) for t, v in zip(main_ids, main_types, strict=True)}
    node_of_id = node_of_id or {}
    rows = []
    for name, (kind, ids, labels) in runs.items():
        for t, lab in zip(ids, labels, strict=True):
            t = int(t)
            ref = main.get(t, main.get(int(node_of_id.get(t, t))))
            rows.append(
                {
                    "territory_id": t,
                    "variant": name,
                    "kind": kind,
                    "matched_type": int(lab),
                    "main_type": ref,
                    "same": None if ref is None else bool(int(lab) == ref),
                }
            )
    cols = ["territory_id", "variant", "kind", "matched_type", "main_type", "same"]
    long = pd.DataFrame(rows, columns=cols)
    long["main_type"] = long["main_type"].astype("Int64")
    long["same"] = long["same"].astype("boolean")
    ok = long.loc[long["same"].notna()]
    summ = (
        ok.groupby(["territory_id", "kind"], sort=True)["same"]
        .agg(n_same=lambda s: int(s.sum()), n_runs="size")
        .reset_index()
    )
    return long, summ


# --- Соперники T5 и T1 ---------------------------------------------------------------------------------


def rival_table(ids: np.ndarray, rivals: Sequence[tuple[str, Partition]]) -> pd.DataFrame:
    """Группа каждого территориального узла в делениях-соперниках: ``rivals`` — (источник, деление), например
    («T5:max_ami», …) или («T1:best_rival:<оборот>», …); одно деление из нескольких источников — одна
    колонка ``sources`` через «;»."""
    by_name: dict[str, tuple[Partition, list[str]]] = {}
    for src, p in rivals:
        if p.name not in by_name:
            by_name[p.name] = (p, [])
        if src not in by_name[p.name][1]:
            by_name[p.name][1].append(src)
    frames = []
    for name, (p, srcs) in by_name.items():
        frames.append(
            pd.DataFrame(
                {
                    "territory_id": np.asarray(ids).astype(np.int64),
                    "partition": name,
                    "partition_label": p.label,
                    "sources": ";".join(srcs),
                    "best_partition_label": np.asarray(p.labels),
                }
            )
        )
    cols = ["territory_id", "partition", "partition_label", "sources", "best_partition_label"]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=cols)
