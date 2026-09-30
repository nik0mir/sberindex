"""Синтетический ``facts.json`` этапа interpret для тестов этапа site (``docs/landing_spec.md``, §4.3).

Поля — как в выходах этапа 5 (ключи слепого прогона 30.09.2026): ``status``, ``blind``, ``label``,
``verdicts_final``, ``texts``, ``t1_notes``, ``thesis``, ``edits``, ``t3_excess``, ``t6``, ``t7``,
``names_final``, ``names_descriptive``, ``naming_test``, ``r1``, ``scope``, ``ladder``, ``qc``. Тексты исходов
собираются из шаблонов предрегистрации ``interpret.tests.<тест>.outcomes`` с выдуманными значениями полей —
так же, как их собирает этап 5 (``interpret.texts``); этап site извлекает поля обратно по тем же шаблонам.
Сочетание вердиктов — словарь с ключами ``site_headlines.HeadlineChecker.combo_levels``.
Ни одного числа из настоящих результатов здесь нет; пометка ``"synthetic": true``.
"""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
import os
import re
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from munnet.site_headlines import HeadlineChecker

ONE_IN_TEN_PHRASE = " — примерно каждое десятое МО"  # как munnet.interpret.labels.ONE_IN_TEN_PHRASE
SET_NAMES = {
    "A": "сопоставимые территории того же типа",
    "D": "сопоставимые территории",
    "B": "соседи по своему региону",
    "C": "случайные МО того же размера",
}
# Выдуманные значения полей: только чтобы тексты были похожи на настоящие
FAKE: dict[str, str] = {
    "rho_cond": "общепит 0,41, розница 0,35",
    "best_partition": "децили населения, по возрастанию",
    "rho_best": "0,20",
    "turnover_beyond": "общепита",
    "turnover_other": "розницы",
    "best_partition_other": "статус МО",
    "rho_best_other": "0,18",
    "turnovers_overall": "общепита и розницы",
    "rho_overall": "общепит 0,30, розница 0,28",
    "n_up": "40",
    "n_down": "21",
    "share_up": "0,66",
    "p0": "0,50",
    "passed_halves": "«нечётные и чётные месяцы»",
    "n_pool": "3",
    "min_pool": "10",
    "share_a": "31%",
    "share_null": "25%",
    "n_reliable": "181",
    "share": "10,2%",
    "placebo_median": "102",
    "p95": "107",
    "excess": "79",
    "excess_share": "4,4%",
    "excess_ci": "74–83",
    "max_ami": "0,05",
    "max_partition": "федеральный округ",
    "turnovers": "общепита",
    "eps2": "0,110",
    "eps2_best": "0,050",
    "flows": "из типа 2 в тип 1 и из типа 1 в тип 3",
    "delta": "0,21",
    "ci": "0,05–0,37",
    "p": "0,030",
    "err_P": "0,041",
    "err_B": "0,052",
    "err_C": "0,060",
    "d_minus_a": "0,004",
    "ci_d": "от −0,002 до 0,010",
    "ari_basket": "0,61",
    "ari_place": "0,12",
    "diff": "0,49",
}
SLOT = re.compile(r"\{([a-z_0-9A-Z]+)\}")


def fill(tpl: str, values: Mapping[str, str]) -> str:
    return SLOT.sub(lambda m: values[m.group(1)], tpl)


def cap(s: str) -> str:
    return s[:1].upper() + s[1:]


def combo_levels(cfg: Mapping) -> dict[str, list]:
    return HeadlineChecker(cfg).combo_levels()


def all_combos(cfg: Mapping, keys: tuple[str, ...] | None = None) -> Iterator[dict]:
    """Все сочетания уровней (или только по ``keys``; остальное — первым уровнем)."""
    L = combo_levels(cfg)
    names = list(L) if keys is None else list(keys)
    base = {k: v[0] for k, v in L.items()}
    for combo in itertools.product(*(L[k] for k in names)):
        yield base | dict(zip(names, combo, strict=True))


def make_facts(cfg: Mapping, combo: Mapping[str, Any], **over: Any) -> dict:
    """facts.json этапа interpret для сочетания ``combo`` (ключи — как в ``combo_levels``)."""
    it = cfg["interpret"]["tests"]
    v = dict(combo)
    t1, t2, t3, t5, t6, t7 = (
        v["T1_ladder_external"],
        v["T2_direction"],
        v["T3_reliable_placebo"],
        v["T5_trivial"],
        v["T6_bank_coverage"],
        v["T7_utility"],
    )
    dw = it["T2_direction"]["direction_words"]
    product = "A" if v["T7_type_gain"] == "adds" else "D"
    vals = dict(FAKE)
    vals["direction"] = dw["t1_confirmed"] if t1 == "confirmed" else dw["otherwise"]
    vals["set_name"] = SET_NAMES[product]
    vals["passed_set"], vals["failed_set"] = SET_NAMES["B"], SET_NAMES["C"]
    vals["one_in_ten_phrase"] = ONE_IN_TEN_PHRASE if (t3 == "confirmed" and v["one_in_ten"]) else ""
    o1 = it["T1_ladder_external"]["outcomes"][t1]
    if t1 == "partial_one":
        vals["other_clause"] = fill(o1["other_overall"], vals)
    o2 = it["T2_direction"]["outcomes"]
    vals["other_halves"] = o2["partial"]["other_halves"]["not"]
    texts = {
        "T1_ladder_external": cap(fill(o1["text"], vals)),
        "T1_proxies": "Описание: доля городских округов по ступеням снизу вверх — 20% / 25% / 30% / 35%.",
        "T2_direction": cap(fill(o2[t2]["text"], vals))
        + (" " + fill(o2["place_override"]["text"], vals) if t2 == "partial_capped" else ""),
        "T3_reliable_placebo": cap(fill(it["T3_reliable_placebo"]["outcomes"][t3]["text"], vals)),
        "T5_trivial": cap(fill(it["T5_trivial"]["outcomes"][t5]["text"], vals)),
        "T6_bank_coverage": cap(fill(it["T6_bank_coverage"]["outcomes"][t6]["text"], vals))
        + (" " + fill(it["T6_bank_coverage"]["outcomes"]["caveat"]["text"], vals) if v["caveat"] else ""),
        "T7_utility": cap(fill(it["T7_utility"]["outcomes"][t7]["text"], vals))
        + " "
        + fill(it["T7_utility"]["outcomes"]["type_" + v["T7_type_gain"]]["text"], vals),
        "T4_basket_vs_place": fill(it["T4_basket_vs_place"]["text"], vals),
    }
    ta = cfg["interpret"]["thesis_assembly"]
    eps_max = float(it["T6_bank_coverage"]["coverage_caveat"]["epsilon2_max"])
    names = {str(t): f"Тип {t}: синтетическое название" for t in (1, 2, 3, 4)}
    facts = {
        "status": "done",
        "blind": None,
        "label": "",
        "synthetic": True,
        "verdicts_final": {
            "T1_ladder_external": t1,
            "T1_text": t1,
            "T2_direction": "partial" if t2 == "partial_capped" else t2,
            "T3_reliable_placebo": t3,
            "T5_trivial": t5,
            "T6_bank_coverage": t6,
            "T7_utility": t7,
        },
        "texts": texts,
        "t1_notes": {"T1_ladder_external": ""},
        "thesis": {
            "question": ta["question"],
            "point_1": [texts[k] for k in ta["point_1"]],
            "point_2": [texts[k] for k in ta["point_2"]],
            "point_3": [texts[k] for k in ta["point_3"]],
            "caveat": [texts[k] for k in ta["caveat"]],
            "excluded": list(ta["excluded"]),
        },
        "edits": [],
        "t3_excess": {"excess": 79.0, "share": 0.044, "one_in_ten": bool(v["one_in_ten"])},
        "t6": {"verdict": t6, "coverage_eps2": eps_max * (2 if v["caveat"] else 0.5)},
        "t7": {"verdict": t7, "type_gain": v["T7_type_gain"], "product": product},
        "names_final": names,
        "names_descriptive": dict(names),
        "naming_test": {"done": False, "fallback": False, "state": "no_answer"},
        "r1": {"unstable_label": "неустойчиво к правилу рёбер", "unstable_parts": []},
        "scope": {"n_nodes": 1776, "n_regions": 73, "n_untyped": 169},
        "ladder": {"order": [2, 1, 3, 4], "sizes_territorial": [472, 806, 394, 102]},
        "qc": {"freshness": {}},
    }
    facts.update(over)
    return facts


def write_interpret(root: Path, facts: Mapping, name: str = "interpret") -> Path:
    """Каталог выходов interpret с одним facts.json."""
    d = Path(root) / name
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "facts.json", "w", encoding="utf-8") as f:
        json.dump(facts, f, ensure_ascii=False)
    return d


# Итоговые типы cluster_final маленьких входов и перенос номеров на номера записи (как facts.ladder.transfer)
FINAL_IDS = [1, 2, 3, 4]
FINAL_RAW = [1, 2, 3, 4]
TRANSFER = {1: 2, 2: 1, 3: 3, 4: 4}


def write_bound_interpret(
    root: Path, out_root: Path, facts: Mapping, name: str = "interpret", shuffle: bool = False
) -> Path:
    """Каталог interpret, привязанный к меткам маленьких входов: sha256 cluster_labels и cluster_final
    в ``qc.inputs_sha256``, ``ladder.transfer`` и ``types.csv``. ``shuffle`` — типы переставлены по узлам
    с теми же размерами, как в слепом прогоне (пометки blind и label при этом можно стереть)."""
    proc = Path(root) / "processed"
    f = copy.deepcopy(dict(facts))
    f["qc"] = {
        **dict(f.get("qc") or {}),
        "inputs_sha256": {
            k: hashlib.sha256((proc / f"{k}.parquet").read_bytes()).hexdigest()
            for k in ("cluster_labels", "cluster_final")
        },
    }
    f["ladder"] = {**dict(f["ladder"]), "transfer": {str(k): v for k, v in TRANSFER.items()}}
    d = write_interpret(out_root, f, name=name)
    types = [TRANSFER[t] for t in FINAL_RAW]
    if shuffle:
        types = types[1:] + types[:1]  # циклический сдвиг: размеры те же, каждый узел — чужой тип
    pd.DataFrame({"territory_id": FINAL_IDS, "type": types}).to_csv(d / "types.csv", index=False)
    return d


def make_inputs(root: Path) -> None:
    """Маленькие входы site_numbers и метки cluster: 4 узла в 2 регионах, 2 ребра основной сети."""
    proc = Path(root) / "processed"
    proc.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "rule": ["basket_dist", "basket_dist", "geo_road"],
            "is_main": [True, True, True],
            "same_region": [True, False, True],
        }
    ).to_parquet(proc / "network_edges.parquet")
    pd.DataFrame({"territory_id": [1, 2, 3, 4], "rule": "basket_dist"}).to_parquet(
        proc / "network_nodes.parquet"
    )
    pd.DataFrame({"territory_id": [1, 2, 3, 4], "region_code": ["01", "01", "02", "02"]}).to_parquet(
        proc / "features_nodes.parquet"
    )
    pd.DataFrame({"territory_id": [1, 2, 3, 4], "own_reliable": [True, False, False, False]}).to_parquet(
        proc / "features_rhythm.parquet"
    )
    pd.DataFrame({"territory_id": [1], "label": [1]}).to_parquet(proc / "cluster_labels.parquet")
    pd.DataFrame({"territory_id": FINAL_IDS, "type": FINAL_RAW}).to_parquet(proc / "cluster_final.parquet")
    for name in ("cluster_labels", "cluster_final"):
        old = (proc / f"{name}.parquet").stat().st_mtime - 3600
        os.utime(proc / f"{name}.parquet", (old, old))
    eda = Path(root) / "outputs" / "eda"
    eda.mkdir(parents=True, exist_ok=True)
    with open(eda / "facts.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "syn.reliable_share_nodes": {"value": 0.25, "text": "25,0%"},
                "syn.null_reliable_share_nodes": {"value": 0.015, "text": "1,5%"},
            },
            f,
            ensure_ascii=False,
        )


def site_config(cfg: Mapping, root: Path) -> dict:
    """Копия конфига с путями во временный каталог (site/ тоже туда)."""
    d = copy.deepcopy(dict(cfg))
    d["paths"] = dict(d["paths"])
    for k in ("interim", "processed", "outputs"):
        d["paths"][k] = str(Path(root) / k)
    d["site"] = copy.deepcopy(d["site"])
    d["site"]["build"]["out"] = str(Path(root) / "site")
    return d
