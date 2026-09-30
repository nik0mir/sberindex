"""Маленький полный мир для сборки лендинга (этап site) без data/: 4 территориальных узла в 2 регионах,
узел-город с 2 районами, 1 МО без типа. Дополняет ``synth_facts.make_inputs`` (те же узлы 1–4, та же доля
межрегиональных рёбер 50% и доля своего ритма 25%) всем, что читают ``mo.json``, ``types.json``,
``checks.json`` и карта ячеек. Известный ответ: 2192 → здесь 8 строк mo.json (7 МО + город),
6 ячеек (5 узлов + 1 без типа), районы города — в ячейке города."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from fixtures.site.synth_facts import FINAL_IDS, FINAL_RAW, TRANSFER, make_inputs, write_bound_interpret

CITY = 90077
INNER = [101, 102]
UNTYPED = 5
CITY_RAW_TYPE = 3
PARTS = ("food", "marketplace", "transport", "health", "cafe", "other")
CONTROLS = {"n_nodes": 5, "n_untyped": 1, "n_inner": 2}
# точки в километрах Альберса: регион 1 — запад, регион 2 — восток, город — рядом с регионом 1
XY = {1: (0, 0), 2: (60, 20), 3: (900, 0), 4: (960, -40), UNTYPED: (930, 60), CITY: (30, 80)}
REGION = {1: 1, 2: 1, 3: 2, 4: 2, UNTYPED: 2, CITY: 77, 101: 77, 102: 77}
REGION_NAME = {1: "Регион Один", 2: "Регион Два", 77: "Москва"}


def page_types() -> dict[int, int]:
    """Тип страницы каждого узла (после переноса номеров)."""
    out = {i: TRANSFER[t] for i, t in zip(FINAL_IDS, FINAL_RAW, strict=True)}
    out[CITY] = TRANSFER[CITY_RAW_TYPE]
    return out


def facts_over() -> dict:
    """Поля facts.json, согласованные с этим миром (контрольные числа и размеры типов)."""
    return {
        "scope": {"n_nodes": 5, "n_regions": 3, "n_untyped": 1, "city_nodes": []},
        "ladder": {"order": [2, 1, 3, 4], "sizes_territorial": [1, 1, 1, 1]},
    }


def _old(p: Path) -> None:
    t = p.stat().st_mtime - 3600
    os.utime(p, (t, t))


def make_site_inputs(root: Path) -> None:
    make_inputs(root)
    proc = Path(root) / "processed"
    ids = [*FINAL_IDS, UNTYPED, *INNER]
    kind = {1: "go", 2: "mr", 3: "mr", 4: "mo", UNTYPED: "mr", 101: "vgt", 102: "vgt"}
    terr = pd.DataFrame(
        {
            "territory_id": pd.array(ids, dtype="int32"),
            "name": [f"МО {i}" for i in ids],
            "name_short": [f"М{i}" for i in ids],
            "region_code": [REGION[i] for i in ids],
            "region_name": [REGION_NAME[REGION[i]] for i in ids],
            "mo_type": [kind[i] for i in ids],
            "is_inner_city": [i in INNER for i in ids],
        }
    )
    terr.to_parquet(proc / "territories.parquet")
    nid = [*FINAL_IDS, UNTYPED, CITY]
    nodes = pd.DataFrame(
        {
            "territory_id": pd.array(nid, dtype="int32"),
            "name": [f"МО {i}" if i != CITY else "Москва" for i in nid],
            "name_short": [f"М{i}" if i != CITY else "Москва" for i in nid],
            "region_code": [REGION[i] for i in nid],
            "region_name": [REGION_NAME[REGION[i]] for i in nid],
            "mo_type": ["go", "mr", "mr", "mo", "mr", "city"],
            "is_city_node": [i == CITY for i in nid],
            "n_members": [1, 1, 1, 1, 1, len(INNER)],
            "x_aea": [XY[i][0] * 1000.0 for i in nid],
            "y_aea": [XY[i][1] * 1000.0 for i in nid],
            "n_months": [24, 24, 24, 24, 12, 24],
            "has_internal_gap": [False] * 5 + [False],
            "is_node": [i != UNTYPED for i in nid],
        }
    )
    nodes.to_parquet(proc / "features_nodes.parquet")
    mem_ids = [*FINAL_IDS, UNTYPED, *INNER]
    pd.DataFrame(
        {
            "territory_id": pd.array(mem_ids, dtype="int32"),
            "node_id": pd.array([*FINAL_IDS, None, CITY, CITY], dtype="Int32"),
            "role": [*(["self"] * 5), "city_member", "city_member"],
            "region_code": [REGION[i] for i in mem_ids],
        }
    ).to_parquet(proc / "features_members.parquet")
    node_ids = [*FINAL_IDS, CITY]
    raw = [*FINAL_RAW, CITY_RAW_TYPE]
    pd.DataFrame(
        {
            "territory_id": pd.array(node_ids, dtype="int32"),
            "type": raw,
            "type_jaccard": [0.9, 0.7, 0.95, 0.8, 0.95],
            "is_city_node": [i == CITY for i in node_ids],
        }
    ).to_parquet(proc / "cluster_final.parquet")
    _old(proc / "cluster_final.parquet")
    rows = []
    for i, t in zip(node_ids, raw, strict=True):
        for w in range(1, 14):
            change = i == 1 and w >= 7  # узел 1 надёжно сменил тип с 7-го окна
            rows.append(
                {
                    "territory_id": i,
                    "window_index": w,
                    "type": 2 if change else t,
                    "status": "reliable" if change else "no_change",
                }
            )
    pd.DataFrame(rows).to_parquet(proc / "dynamics_history.parquet")
    dyn = Path(root) / "outputs" / "dynamics"
    dyn.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "territory_id": node_ids,
            "type_2023": raw,
            "type_2024": [2 if i == 1 else t for i, t in zip(node_ids, raw, strict=True)],
            "reliable": [i == 1 for i in node_ids],
        }
    ).to_csv(dyn / "nodes.csv", index=False)
    ca = pd.DataFrame(
        [
            {
                "territory_id": i,
                "year": y,
                "pop_avg": 1000.0 * (i % 97 + 1) + y - 2023,
                "workplace_based": i in INNER,
            }
            for i in ids
            for y in (2023, 2024)
        ]
    )
    ca.to_parquet(proc / "context_annual.parquet")
    pd.DataFrame(
        [{"territory_id": i, "year": y, "pop_avg": 5e6} for i in node_ids for y in (2023, 2024) if i == CITY]
        + [{"territory_id": i, "year": y, "pop_avg": 2000.0} for i in FINAL_IDS for y in (2023, 2024)]
    ).to_parquet(proc / "features_place.parquet")
    rng = np.random.default_rng(0)
    fw = pd.DataFrame(
        [
            {"territory_id": i, "window": w, **{f"clr_rel_{p}": float(rng.normal(0, 0.2)) for p in PARTS}}
            for i in node_ids
            for w in ("2023", "2024", "2023H1")
        ]
    )
    fw.to_parquet(proc / "features_windows.parquet")
    pd.DataFrame(
        [
            {"territory_id": i, "category": c, "month": m, "own": 0.01 * m * (i == 1)}
            for i in node_ids
            for c in ("all", "food")
            for m in range(1, 13)
            for _ in range(2)
        ]
    ).to_parquet(proc / "features_rhythm_monthly.parquet")
    pd.DataFrame(
        {
            "rule": ["basket_dist", "basket_dist", "geo_road"],
            "source": [1, 2, 1],
            "target": [2, 3, 3],
            "weight": [0.9, 0.8, 1.0],
            "is_main": [True, True, True],
            "same_region": [True, False, True],
            "dist_km": [60.0, 840.0, 900.0],
        }
    ).to_parquet(proc / "network_edges.parquet")


def bound_interpret(root: Path, out_root: Path, facts, shuffle: bool = False) -> Path:
    """``write_bound_interpret`` плюс строка узла-города в ``types.csv`` (его тип не перемешивается)."""
    d = write_bound_interpret(root, out_root, facts, shuffle=shuffle)
    t = pd.read_csv(d / "types.csv")
    t = pd.concat([t, pd.DataFrame({"territory_id": [CITY], "type": [TRANSFER[CITY_RAW_TYPE]]})])
    t.to_csv(d / "types.csv", index=False)
    return d
