"""Справка ``level_rho``: упорядочивает ли сам уровень трат обороты Росстата сильнее четырёх типов.

ПОСЧИТАНО ПОСЛЕ ПРОВЕРОК, ПРАВИЛО ЗАРАНЕЕ НЕ ЗАПИСАНО (04.10.2026). Повод — замечание судьи критерия 5
и devils-advocate (03.10): ρ уровня трат с розницей Росстата ≈ 0,63 выше ρ типов (0,55). Вердиктов нет,
это описание рядом с T1 (a); вердикты и тексты этапа 5, прежние выходы usefulness не меняются.

Всё по определениям T1 ``rho_a`` основного прогона interpret (``munnet.interpret``, код не копируется):
узлы и цели — ``context.build`` (территориальные узлы, оборот общепита и розницы Росстата на жителя
2023 года, ``log1p_rel``, пропуск — узел выпадает), выборка (a) и бутстрап узлов — ``order.make_turnover``
и ``order.make_boots`` с тем же генератором, что ``context.evidence`` основного прогона
(``default_rng([seed, 501, 0])``, обороты в порядке записи), число повторов и уровень интервала —
``interpret.tests.T1_ladder_external.bootstrap`` и ``ci_level``. Сравниваются на одних выборках:

- **types** — ступени четырёх типов (``Data.rank_of``), как в T1; сверяются с ``t1_runs.csv`` (run = main):
  ρ, интервал и n должны совпасть (иначе код 3) — это доказательство, что узлы, цель и бутстрап те же;
- **level** — непрерывный признак ``log_level_rel`` (уровень трат относительно группы региона, окно признаков
  типов) в том виде, в каком его видела кластеризация (``Ctx.filled``);
- **rival** — существующее деление-соперник ``sized:log_level_rel:+`` (четыре группы по уровню трат
  с размерами типов, ``context.build``), ориентация — ``order.build_rivals``
  (``place_partitions.orientation.sized``).

Рядом — разности «level − types» и «rival − types» на тех же бутстрап-выборках (парные интервалы).
Выход — ``<paths.outputs>/usefulness/level_rho.json``; вызывается из ``munnet.usefulness.run`` последним.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from munnet.config import Config
from munnet.contracts import MissingInputError, QCError

log = logging.getLogger(__name__)

FEATURE = "log_level_rel"  # признак типов «уровень трат относительно группы региона»
RIVAL = "sized:log_level_rel:+"  # деление-соперник T1/T5 по этому признаку (interpret.place_partitions.sized)
EVIDENCE_TAG = 0  # context.evidence(cx, spec, seed, 0) — выборки основного прогона T1
# допуск сверки ρ и интервала типов с t1_runs.csv: тот же код и те же выборки — совпадение точное
QC_TOL = 1e-12
NOTE = (
    "посчитано после проверок, правило заранее не записано: справка рядом с T1 (a), вердиктов нет; "
    "узлы, цели, бутстрап и деление-соперник — из кода T1 основного прогона interpret"
)


def spearman_ci(
    z: np.ndarray, y: np.ndarray, boots: Sequence[np.ndarray], level: float
) -> tuple[float, tuple[float, float], np.ndarray]:
    """ρ Спирмена ``z`` и ``y``, его значения на бутстрап-выборках (индексы ``boots``) и перцентильный
    интервал — как ``rho_a`` и ``rho_a_ci`` в ``order.t1_eval``."""
    from munnet.interpret import stats as S

    z, y = np.asarray(z, dtype=np.float64), np.asarray(y, dtype=np.float64)
    rho = float(S.spearman(z, y))
    draws = np.array([S.spearman(z[i], y[i]) for i in boots], dtype=np.float64)
    return rho, S.perc_ci(draws, level), draws


def compare(
    scores: Mapping[str, np.ndarray],
    y: np.ndarray,
    boots: Sequence[np.ndarray],
    level: float,
    base: str = "types",
) -> dict[str, Any]:
    """ρ каждой оценки узлов из ``scores`` с целью ``y`` на одних бутстрап-выборках и парные разности
    «оценка − ``base``» (точка и перцентильный интервал)."""
    from munnet.interpret import stats as S

    res: dict[str, Any] = {"n": int(len(y))}
    draws = {}
    for name, z in scores.items():
        rho, ci, d = spearman_ci(z, y, boots, level)
        res[name] = {"rho": rho, "ci": [float(ci[0]), float(ci[1])]}
        draws[name] = d
    for name in scores:
        if name == base:
            continue
        diff = draws[name] - draws[base]
        lo, hi = S.perc_ci(diff, level)
        res[f"diff_{name}_minus_{base}"] = {
            "point": res[name]["rho"] - res[base]["rho"],
            "ci": [float(lo), float(hi)],
        }
    return res


def check_against_t1(res: Mapping[str, Any], t1_main: pd.DataFrame, tol: float = QC_TOL) -> dict[str, float]:
    """Сверка ρ, интервала и n типов с ``t1_runs.csv`` основного прогона; расхождение — QCError (код 3)."""
    dev = {}
    for turnover, r in res.items():
        row = t1_main.loc[t1_main["turnover"] == turnover]
        if len(row) != 1:
            raise QCError(f"usefulness.level_rho: в t1_runs.csv (run = main) нет строки оборота {turnover!r}")
        row = row.iloc[0]
        if int(row["n_a"]) != int(r["n"]):
            raise QCError(f"usefulness.level_rho: n {r['n']} ≠ t1_runs.csv n_a {row['n_a']} ({turnover})")
        got = [r["types"]["rho"], *r["types"]["ci"]]
        ref = [float(row["rho_a"]), float(row["rho_a_lo"]), float(row["rho_a_hi"])]
        d = float(np.max(np.abs(np.array(got) - np.array(ref))))
        if not d <= tol:
            raise QCError(
                f"usefulness.level_rho: ρ типов и интервал ({turnover}) расходятся с t1_runs.csv "
                f"на {d:.3g} — узлы, цель или бутстрап не те, что у T1"
            )
        dev[turnover] = d
    return dev


def _inputs(cfg: Config) -> dict[str, Any]:
    """Узлы, цели, ступени и бутстрап-выборки (a) — кодом interpret, как в основном прогоне T1."""
    from munnet.interpret import context as CX
    from munnet.interpret import data as D
    from munnet.interpret import order as O
    from munnet.interpret.spec import Spec

    spec = Spec(cfg["interpret"])
    seed = int(cfg["seed"])
    if int(spec["seed"]) != seed:
        raise QCError(f"usefulness.level_rho: interpret.seed = {spec['seed']} ≠ seed = {seed}")
    data = D.load(cfg, spec, None)
    cx = CX.build(data, spec)
    t1 = spec["tests"]["T1_ladder_external"]
    n_boot, level = int(t1["bootstrap"]), float(t1["ci_level"])
    # тот же генератор и порядок вызовов, что в context.evidence: make_boots для каждого оборота
    # (выборки b и a вперемешку)
    rng = np.random.default_rng([seed, 501, EVIDENCE_TAG])
    turn, boots_b, boots_a = {}, {}, {}
    for name, v in cx.indicators.items():
        t = O.make_turnover(name, cx.signs[name], v, cx.groups, cx.strata)
        turn[name] = t
        boots_b[name], boots_a[name] = O.make_boots(t, n_boot, rng)
    rivals = [p for p in cx.rivals if p.name == RIVAL]
    if len(rivals) != 1:
        raise QCError(f"usefulness.level_rho: деления {RIVAL!r} нет среди соперников interpret")
    ev = O.Evidence(turnovers=turn, boots_b=boots_b, boots_a=boots_a, rivals=[])
    orient = spec["place_partitions"]["orientation"]
    O.build_rivals(ev, rivals, {"sized": str(orient["sized"]), "place_only": str(orient["place_only"])})
    if FEATURE not in cx.filled.columns:
        raise QCError(f"usefulness.level_rho: признака {FEATURE!r} нет во входах кластеризации")
    return {
        "turnovers": turn,
        "boots_a": boots_a,
        "ranks_types": data.rank_of(data.final[cx.pos]),
        "level": cx.filled[FEATURE].to_numpy(dtype=np.float64),
        "level_missing_raw": int(np.isnan(cx.raw[FEATURE].to_numpy(dtype=np.float64)).sum()),
        "rival": rivals[0],
        "rival_ranks": {name: ev.rival_ranks[name][0] for name in turn},
        "rival_rho_b": {name: float(ev.rival_point_rho[name][0]) for name in turn},
        "n_boot": n_boot,
        "level_ci": level,
        "n_nodes": len(cx.pos),
    }


def run(cfg: Config) -> dict[str, Any]:
    """Расчёт и запись ``level_rho.json``; прежний файл блока удаляется до расчёта."""
    out = cfg.dir("outputs") / "usefulness"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "level_rho.json"
    path.unlink(missing_ok=True)
    t1_path = Path(cfg["paths"]["outputs"]) / "interpret" / "t1_runs.csv"
    if not t1_path.exists():
        raise MissingInputError(f"usefulness.level_rho: нет {t1_path} — сначала этап interpret")
    t1 = pd.read_csv(t1_path)
    t1_main = t1.loc[t1["run"] == "main"]
    inp = _inputs(cfg)
    res: dict[str, Any] = {}
    for name, t in inp["turnovers"].items():
        a = t.a_idx
        scores = {
            "types": inp["ranks_types"][a],
            "level": inp["level"][a],
            "rival": inp["rival_ranks"][name][a],
        }
        res[name] = compare(scores, t.a_y, inp["boots_a"][name], inp["level_ci"])
    dev = check_against_t1(res, t1_main)
    rv = inp["rival"]
    facts = {
        "status": "done",
        "note": NOTE,
        "feature": FEATURE,
        "rival": RIVAL,
        "rival_label": rv.label,
        # ориентация соперника по T1 (sign_of_rho на ρ (b)): False — порядок групп «по возрастанию уровня»
        "rival_flipped": {
            name: bool(not np.array_equal(inp["rival_ranks"][name], rv.ranks)) for name in inp["turnovers"]
        },
        "rival_rho_b": inp["rival_rho_b"],
        "n_nodes": int(inp["n_nodes"]),
        "level_missing_raw": inp["level_missing_raw"],
        "bootstrap": inp["n_boot"],
        "ci_level": inp["level_ci"],
        "turnovers": res,
        "qc": {"t1_runs_max_dev": dev, "tol": QC_TOL},
        "inputs_sha256": {"t1_runs": hashlib.sha256(t1_path.read_bytes()).hexdigest()},
    }
    path.write_text(json.dumps(facts, ensure_ascii=False, indent=1), encoding="utf-8")
    for name, r in res.items():
        log.info(
            "usefulness.level_rho (после проверок, правило заранее не записано): %s, n %d — ρ типов %.3f, "
            "уровня трат %.3f, четырёх групп по уровню %.3f",
            name, r["n"], r["types"]["rho"], r["level"]["rho"], r["rival"]["rho"],
        )  # fmt: skip
    return facts
