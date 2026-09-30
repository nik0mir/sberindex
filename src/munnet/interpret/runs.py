"""Прогоны T2 и T3 (и фильтр R1): основной, варианты входов, seed и способы прослеживания.

Основной прогон берёт наблюдаемые переходы из выходов этапа dynamics (``nodes.csv``, ``labels.csv``) и
перефитит только половины чувствительности и псевдопары. Варианты входов (``robustness.variants``) и seed
(``robustness.seeds``) повторяют этап dynamics целиком теми же функциями (``dynamics.compute.main_track``):
24 месяца → сопоставление с итогом (венгерский алгоритм по Жаккару, ``robustness.match``), 13 скользящих окон,
половины, псевдопары. Результат прогона — матрицы надёжных переходов наблюдения и плацебо и типы 2023 года.
"""

from __future__ import annotations

import hashlib
import logging
import pickle
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from munnet.dynamics import compute as DC
from munnet.dynamics import tracking as T
from munnet.dynamics import windows as W
from munnet.interpret import placebo as PL

log = logging.getLogger(__name__)


@dataclass
class Observed:
    """Наблюдаемые типы половин (−1 — половины разошлись) и надёжные переходы одного разбиения половин."""

    t_a: np.ndarray
    t_b: np.ndarray
    reliable: np.ndarray

    @property
    def n(self) -> int:
        return int(self.reliable.sum())


@dataclass
class RunOut:
    name: str
    kind: str  # main | variant | seed | tracking
    n_nodes: int
    ids: np.ndarray
    final: np.ndarray  # 0…K−1 в нумерации итога (номера записи − 1)
    type_2023: np.ndarray
    observed: dict[str, Observed]  # схема половин -> наблюдение
    placebo: dict[str, list[np.ndarray]]  # схема половин -> матрицы псевдопар
    X_place: np.ndarray | None = None
    basket: np.ndarray | None = None  # корзина 24 месяцев прогона (знаки R1)
    seconds: float = 0.0
    qc: dict = field(default_factory=dict)
    sets: list = field(default_factory=list)


def schemes_of(spec_t3) -> dict[str, tuple[tuple[int, ...], tuple[int, ...]]]:
    """Разбиения половин: основное ``odd_even`` и чувствительность ``halves_sensitivity``."""
    spec_t3.expect("halves", "odd_even")
    sens = [tuple(int(m) for m in h) for h in spec_t3["halves_sensitivity"]]
    if sorted(sens[0] + sens[1]) != list(range(1, 13)):
        raise ValueError("T3.halves_sensitivity: половины должны покрыть 12 календарных месяцев")
    return {"main": (tuple(range(1, 13, 2)), tuple(range(2, 13, 2))), "sensitivity": (sens[0], sens[1])}


def _key(*parts: Any) -> str:
    h = hashlib.sha256()
    for p in parts:
        if isinstance(p, np.ndarray):
            h.update(p.tobytes())
        else:
            h.update(repr(p).encode("utf-8"))
    return h.hexdigest()[:20]


def cached(cache_dir: Path, key: str, fn):
    """Кэш тяжёлого прогона: ключ — входы прогона и исходный код псевдопар (смена кода — новый расчёт)."""
    path = cache_dir / f"{key}.pkl"
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    res = fn()
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(res, f)
    tmp.replace(path)
    return res


# Модули, от которых зависят тяжёлые прогоны (псевдопары, варианты, seed): сеть окна, входы и методы
# кластеризации, прослеживание dynamics. Смена любого — новый расчёт, а не устаревший кэш.
HEAVY_MODULES: tuple[str, ...] = (
    "munnet.interpret.placebo",
    "munnet.interpret.runs",
    "munnet.dynamics.compute",
    "munnet.dynamics.tracking",
    "munnet.dynamics.windows",
    "munnet.dynamics.params",
    "munnet.clustering.methods",
    "munnet.clustering.inputs",
    "munnet.clustering.kefrin",
    "munnet.clustering.parallel",
    "munnet.clustering.params",
    "munnet.network.compare",
    "munnet.network.compute",
    "munnet.network.graph",
    "munnet.network.data",
    "munnet.network.params",
    "munnet.network.rules",
    "munnet.features",
    "munnet.nodes",
)

# Поддеревья конфига, от которых зависят тяжёлые прогоны (параметры сети, входов, реализации методов
# и прослеживания, варианты чувствительности R1).
HEAVY_CONFIG: tuple[tuple[str, ...], ...] = (
    ("seed",),
    ("features",),
    ("network",),
    ("clustering", "inputs"),
    ("clustering", "impl"),
    ("clustering", "selection", "sensitivity"),
    ("dynamics", "impl"),
    ("dynamics", "tracking"),
)


def _munnet_imports(name: str) -> set[str]:
    """Модули ``munnet.*``, которые импортирует модуль ``name`` (в том числе внутри функций)."""
    import ast
    import importlib.util

    spec = importlib.util.find_spec(name)
    if spec is None or spec.origin is None:
        raise ModuleNotFoundError(name)
    tree = ast.parse(Path(spec.origin).read_text(encoding="utf-8"))
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out |= {a.name for a in node.names if a.name.split(".")[0] == "munnet"}
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                raise ValueError(f"{name}: относительный импорт — ключ кэша его не прослеживает")
            mod = node.module or ""
            if mod.split(".")[0] != "munnet":
                continue
            # «from пакет import модуль» — зависимость от модуля; сам пакет (его __init__) — только если
            # из него берут имя, а не модуль (ленивые импорты в функциях __init__ зависимостью не считаются)
            for a in node.names:
                sub = f"{mod}.{a.name}"
                try:
                    is_mod = importlib.util.find_spec(sub) is not None
                except ModuleNotFoundError:
                    is_mod = False
                out.add(sub if is_mod else mod)
    return out


def heavy_closure() -> list[str]:
    """``HEAVY_MODULES`` и все модули ``munnet``, которые они импортируют прямо или через другие модули."""
    seen: set[str] = set()
    todo = list(HEAVY_MODULES)
    while todo:
        name = todo.pop()
        if name in seen:
            continue
        seen.add(name)
        todo.extend(_munnet_imports(name) - seen)
    return sorted(seen)


def code_hash() -> str:
    """Хеш исходного кода ``heavy_closure()``: модули тяжёлых прогонов и всё, что они импортируют из munnet
    (правка текстов отчёта кэш не сбрасывает, правка любого модуля расчёта — сбрасывает)."""
    import importlib.util

    texts = []
    for name in heavy_closure():
        origin = importlib.util.find_spec(name).origin
        texts.append((name, Path(str(origin)).read_text(encoding="utf-8")))
    return _key(*texts)


# Ключи конфига, от которых результат не зависит (число процессов проверено тестом cluster
# test_workers_do_not_change_result): в ключ кэша не входят — смена числа процессов не сбрасывает расчёт.
CONFIG_IGNORED: tuple[tuple[str, ...], ...] = (("clustering", "impl", "workers"),)


def config_hash(cfg) -> str:
    """Хеш поддеревьев конфига ``HEAVY_CONFIG`` без ``CONFIG_IGNORED`` (нет ключа — ``None``, это тоже часть
    ключа)."""
    import copy

    parts = []
    for path in HEAVY_CONFIG:
        node: Any = cfg.data if hasattr(cfg, "data") else cfg
        for k in path:
            node = node.get(k) if isinstance(node, dict) else None
        node = copy.deepcopy(node)
        for ign in CONFIG_IGNORED:
            if ign[: len(path)] == path and len(ign) > len(path) and isinstance(node, dict):
                sub = node
                for k in ign[len(path) : -1]:
                    sub = sub.get(k) if isinstance(sub, dict) else None
                if isinstance(sub, dict):
                    sub.pop(ign[-1], None)
        parts.append((path, node))
    return _key(*parts)


def _feed(h, obj: Any) -> None:
    """Содержимое объекта в хеш: массивы — байтами, таблицы — хешем строк, классы данных — по полям."""
    import dataclasses

    import pandas as pd
    import scipy.sparse as sp

    if obj is None or isinstance(obj, (bool, int, float, complex, str, bytes, np.generic, Path)):
        h.update(f"{type(obj).__name__}:{obj!r};".encode())
    elif isinstance(obj, np.ndarray):
        if obj.dtype.hasobject:
            h.update(b"ndarray-object:" + pickle.dumps(obj.tolist(), protocol=4))
        else:
            h.update(f"ndarray:{obj.dtype.str}:{obj.shape};".encode())
            h.update(np.ascontiguousarray(obj).tobytes())
    elif sp.issparse(obj):
        m = sp.csr_matrix(obj, copy=True)
        m.sum_duplicates()
        m.sort_indices()
        h.update(f"sparse:{m.shape};".encode())
        for a in (m.data, m.indices, m.indptr):
            _feed(h, a)
    elif isinstance(obj, (pd.DataFrame, pd.Series)):
        h.update(f"{type(obj).__name__}:{obj.shape};".encode())
        cols = list(obj.columns) if isinstance(obj, pd.DataFrame) else [obj.name]
        _feed(h, [str(c) for c in cols])
        _feed(h, [str(d) for d in (obj.dtypes if isinstance(obj, pd.DataFrame) else [obj.dtype])])
        try:
            _feed(h, pd.util.hash_pandas_object(obj, index=True).to_numpy())
        except TypeError:  # непрехешируемые ячейки (списки) — через pickle
            h.update(pickle.dumps(obj, protocol=4))
    elif isinstance(obj, dict) or (hasattr(obj, "items") and hasattr(obj, "keys")):
        items = sorted(obj.items(), key=lambda kv: repr(kv[0]))
        h.update(f"map:{len(items)};".encode())
        for k, v in items:
            _feed(h, k)
            _feed(h, v)
    elif isinstance(obj, (list, tuple, set, frozenset)):
        seq = sorted(obj, key=repr) if isinstance(obj, (set, frozenset)) else obj
        h.update(f"{type(obj).__name__}:{len(seq)};".encode())
        for v in seq:
            _feed(h, v)
    elif dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        h.update(f"dataclass:{type(obj).__qualname__};".encode())
        for f in dataclasses.fields(obj):
            _feed(h, f.name)
            _feed(h, getattr(obj, f.name))
    elif hasattr(obj, "__dict__"):
        h.update(f"object:{type(obj).__qualname__};".encode())
        _feed(h, vars(obj))
    else:
        h.update(b"pickle:" + pickle.dumps(obj, protocol=4))


def digest(*objs: Any) -> str:
    """Хеш содержимого входов тяжёлого прогона (контекст прогона, метки, выходы dynamics, наборы месяцев):
    ключ кэша следует за данными, а не за временем изменения файлов."""
    h = hashlib.sha256()
    for o in objs:
        _feed(h, o)
    return h.hexdigest()[:20]


def pair_tasks(sets: list[tuple[int, ...]], final: np.ndarray) -> list[tuple]:
    return [(r, s, final) for r, s in enumerate(sets)]


def collect_pairs(res: list[dict], approach: str, schemes) -> dict[str, list[np.ndarray]]:
    return {s: [np.asarray(r[approach][s]) for r in res] for s in schemes}


def observed_from(halves: dict[str, np.ndarray]) -> Observed:
    rel, ta, tb = T.stable_halves(
        halves["odd_2023"], halves["even_2023"], halves["odd_2024"], halves["even_2024"]
    )
    return Observed(ta, tb, rel)


def sens_halves_tasks(schemes) -> list[tuple]:
    """Половины 2023 и 2024 годов по разбиению чувствительности (наблюдение пересчитывается на тех же
    половинах)."""
    out = []
    for yi, y in enumerate((2023, 2024)):
        for j, hm in enumerate(PL.split_halves(PL.year_months(yi), schemes["sensitivity"])):
            out.append((f"sens_{y}_{j}", tuple(hm)))
    return out


def full_run(
    ctx: PL.RunCtx,
    final_ref: np.ndarray,
    ref_ids: np.ndarray,
    sets: list[tuple[int, ...]],
    workers: int,
    kind: str,
    cache_dir: Path,
) -> RunOut:
    """Вариант или seed: этап dynamics целиком тем же кодом и псевдопары."""
    t0 = time.perf_counter()
    ns = ctx.ns
    k = ctx.k
    pers = W.periods(ns, ctx.netp)
    roll = [q for q in pers if q.kind == "rolling"]
    halves_p = {
        ("odd" if q.kind == "odd" else "even") + f"_{q.year}": q for q in pers if q.kind in ("odd", "even")
    }
    tasks = [("all24", tuple(range(24)), False)]
    tasks += [(q.name, tuple(int(m) for m in q.months), False) for q in roll]
    tasks += [(h, tuple(int(m) for m in q.months), False) for h, q in halves_p.items()]
    tasks += [(tag, months, False) for tag, months in sens_halves_tasks(ctx.schemes)]
    fits = {tag: lab for tag, lab, _ in PL.run_tasks(ctx, tasks, PL.fit_task, workers)}
    # 24 месяца → нумерация итога по общим узлам (robustness.match: hungarian_jaccard)
    pos_ref = {int(t): i for i, t in enumerate(ref_ids)}
    common = np.array([int(t) in pos_ref for t in ns.ids])
    ref_c = final_ref[[pos_ref[int(t)] for t in ns.ids[common]]]
    mapping = T.match(ref_c, fits["all24"][common], k)
    final_v = T.relabel(fits["all24"], mapping)
    track = DC.main_track(final_v, [fits[q.name] for q in roll], {h: fits[h] for h in halves_p}, k)
    obs = {"main": observed_from(track.halves)}
    sens = {}
    for yi, y in enumerate((2023, 2024)):
        ref = track.rolling[0] if yi == 0 else track.rolling[-1]
        for j in range(2):
            sens[f"{y}_{j}"] = T.aligned(ref, fits[f"sens_{y}_{j}"], k)
    obs["sensitivity"] = observed_from(
        {
            "odd_2023": sens["2023_0"],
            "even_2023": sens["2023_1"],
            "odd_2024": sens["2024_0"],
            "even_2024": sens["2024_1"],
        }
    )
    # как в основном прогоне (data.check_chain_direct): наблюдение 2024 года — цепочкой окон, плацебо —
    # напрямую с итогом прогона; расхождение в последнем окне публикуется (стоп здесь уронил бы прогон
    # на полпути)
    chain_info = chain_check(ctx, fits, final_v, track, roll, halves_p)
    last_differ = int(chain_info["last_window_differ"])
    if last_differ > 0:
        log.warning(
            "interpret: %s — окно 2024 года по цепочке и напрямую расходится у %d узлов: наблюдение T2/T3 "
            "и плацебо неравноценны (qc.chain_direct в facts.json)",
            ctx.name,
            last_differ,
        )
    pres = PL.run_tasks(ctx, pair_tasks(sets, final_v), PL.pair_task, workers)
    placebo = collect_pairs(pres, PL.MAIN, ctx.schemes)
    place_cols = [ctx.base.x_names.index(f) for f in ctx.base.x_names if f != "log_level_rel"]
    return RunOut(
        name=ctx.name,
        kind=kind,
        n_nodes=int(ns.n),
        ids=ns.ids.copy(),
        final=final_v,
        type_2023=track.rolling[0],
        observed=obs,
        placebo=placebo,
        X_place=ctx.base.X,
        basket=ctx.base.B,
        seconds=time.perf_counter() - t0,
        qc={"place_cols": place_cols, "chain_info": chain_info},
    )


def chain_check(ctx: PL.RunCtx, fits: dict, final_v: np.ndarray, track, roll: list, halves_p: dict) -> dict:
    """Сверка «цепочка окон против прямого сопоставления» прогона, повторяющего dynamics: число узлов окна
    2024 года, у которых номер по цепочке ≠ номеру при прямом сопоставлении с итогом прогона, и сводка
    ``main_track`` по всем окнам.

    В слепом прогоне метки каждого окна перемешаны отдельно, и цепочка по ним ничего не говорит о данных,
    поэтому сверка идёт на метках, возвращённых в исходный порядок узлов: это число узлов, оно не зависит
    от номеров типов и вердиктов не раскрывает (так же устроена ``refit_ari``) — слепой прогон заранее
    показывает, что будет в настоящем."""
    k = ctx.k
    if ctx.blind is not None:
        fits = {tag: PL.unblind(lab, ctx, "fit", tag) for tag, lab in fits.items()}
        final_v = fits["all24"]
        track = DC.main_track(final_v, [fits[q.name] for q in roll], {h: fits[h] for h in halves_p}, k)
    last = int((track.rolling[-1] != T.aligned(final_v, fits[roll[-1].name], k)).sum())
    return {**{k_: float(v) for k_, v in track.info.items()}, "last_window_differ": float(last)}


def refit_ari(ctx: PL.RunCtx, labels_dyn: dict, refit: dict[str, np.ndarray]) -> dict[str, float]:
    """Сверка сверх записи: ARI перефита окон с метками outputs/dynamics/labels.csv. В слепом прогоне обе
    стороны возвращаются в исходный порядок узлов (ARI не зависит от номеров типов и вердиктов не раскрывает),
    поэтому сверка работает и в слепом режиме — до настоящего прогона."""
    from sklearn.metrics import adjusted_rand_score

    from munnet.interpret.data import dynamics_perm

    qc = {}
    for w, lab in refit.items():
        ref = np.asarray(labels_dyn[(PL.MAIN, w)]) - 1
        if ctx.blind is not None:
            ref = ref[np.argsort(dynamics_perm(len(ref), int(ctx.blind)))]
            lab = PL.unblind(lab, ctx, "fit", w)
        qc[f"refit_ari_{w}"] = float(adjusted_rand_score(ref, lab))
    return qc


def _dyn_true(ctx: PL.RunCtx, v: np.ndarray) -> np.ndarray:
    """Метки outputs/dynamics в исходном порядке узлов (в слепом прогоне они перемешаны ``dynamics_perm``)."""
    if ctx.blind is None:
        return np.asarray(v)
    from munnet.interpret.data import dynamics_perm

    return np.asarray(v)[np.argsort(dynamics_perm(len(v), int(ctx.blind)))]


def evo_refit(ctx: PL.RunCtx, labels_dyn: dict, got: dict, roll: list, ari_min: float):
    """Эволюционный способ dynamics, пройденный заново тем же кодом на перефите окон: от первого окна
    основного способа (labels.csv) через все скользящие окна. Возвращает путь, центры окон и ARI с метками
    способа в labels.csv (окна 2023 и 2024 годов и половины). Меньше ``ari_min`` — предупреждение и
    публикация (qc.refit_evolutionary), а не стоп: способ — одна из альтернатив R1, основной расчёт от него
    не зависит. В слепом прогоне путь проходится на метках в исходном порядке узлов (ARI не раскрывает
    вердиктов), а метки половин чувствительности перемешиваются потом, как все метки слепого прогона."""
    from sklearn.metrics import adjusted_rand_score

    from munnet.interpret.data import HALF_PERIODS

    k = ctx.k
    first = _dyn_true(ctx, labels_dyn[(PL.MAIN, roll[0].name)] - 1)
    z_roll = [got[q.name][1] for q in roll]
    z_halves = {h: got[per][1] for h, per in HALF_PERIODS.items()}
    tr, Cs = DC.evolutionary_track(z_roll, z_halves, first, k, ctx.epsilon, ctx.max_iter)
    ari = {}
    for i in (0, -1):
        ref = _dyn_true(ctx, labels_dyn[(PL.EVO, roll[i].name)] - 1)
        ari[roll[i].name] = float(adjusted_rand_score(ref, tr.rolling[i]))
    for h, per in HALF_PERIODS.items():
        ari[per] = float(adjusted_rand_score(_dyn_true(ctx, labels_dyn[(PL.EVO, per)] - 1), tr.halves[h]))
    if min(ari.values()) < ari_min:
        log.warning(
            "interpret: эволюционный способ, пройденный заново, не повторил outputs/dynamics/labels.csv "
            "(ARI %s < %.2f): центры половин чувствительности этого способа приближённые "
            "(qc.refit_evolutionary)",
            {k_: round(v, 3) for k_, v in ari.items()},
            ari_min,
        )
    return tr, Cs, ari


def main_run(
    ctx: PL.RunCtx,
    final: np.ndarray,
    nodes_dyn,
    labels_dyn: dict,
    sets: list[tuple[int, ...]],
    workers: int,
    final_ari_min: float,
    windows: tuple[str, str],
) -> tuple[RunOut, dict[str, RunOut], dict]:
    """Основной прогон: наблюдение — выходы dynamics; перефит — половины чувствительности, окна 2023
    и 2024 годов
    (сверка с labels.csv и центры для способов R1) и псевдопары со способами прослеживания."""
    t0 = time.perf_counter()
    k = ctx.k
    w23, w24 = windows  # interpret.windows.year_a и year_b — окна type_2023 и type_2024 этапа dynamics
    from munnet.interpret.data import HALF_PERIODS

    pers = W.periods(ctx.ns, ctx.netp)
    roll = [q for q in pers if q.kind == "rolling"]
    if (roll[0].name, roll[-1].name) != (w23, w24) or (
        tuple(int(m) for m in roll[0].months),
        tuple(int(m) for m in roll[-1].months),
    ) != (tuple(range(12)), tuple(range(12, 24))):
        raise ValueError(
            f"interpret.windows: {w23} и {w24} должны быть первым и последним скользящим окном dynamics "
            f"(12 месяцев 2023 и 2024 годов), а они {roll[0].name} и {roll[-1].name}"
        )
    tasks = [(w23, tuple(range(12)), True), (w24, tuple(range(12, 24)), True)]
    if ctx.space is not None:
        # уточнение реализации (clarifications, R1 evolutionary): все скользящие окна и половины года —
        # чтобы пройти эволюционный способ тем же кодом dynamics и взять его центры окон 2023 и 2024 годов
        tasks += [(q.name, tuple(int(m) for m in q.months), True) for q in roll[1:-1]]
        tasks += [(q.name, tuple(int(m) for m in q.months), True) for q in pers if q.kind in ("odd", "even")]
    tasks += [(tag, months, True) for tag, months in sens_halves_tasks(ctx.schemes)]
    got = {tag: (lab, Z) for tag, lab, Z in PL.run_tasks(ctx, tasks, PL.fit_task, workers)}
    qc: dict[str, Any] = refit_ari(ctx, labels_dyn, {w: got[w][0] for w in (w23, w24)})
    if min(qc.values()) < final_ari_min:
        from munnet.contracts import QCError

        raise QCError(
            f"interpret: перефит окон 2023 и 2024 годов не повторил outputs/dynamics/labels.csv: {qc}"
        )
    if ctx.space is not None:
        # описание сверх стоп-правила: перефит остальных окон основного способа против labels.csv
        other = [q.name for q in roll[1:-1]] + list(HALF_PERIODS.values())
        qc["refit_other"] = refit_ari(ctx, labels_dyn, {w: got[w][0] for w in other})

    def half_types(appr: str) -> dict[str, np.ndarray]:
        from munnet.interpret.data import HALF_PERIODS

        return {h: labels_dyn[(appr, per)] - 1 for h, per in HALF_PERIODS.items()}

    nd = nodes_dyn
    ha = nd["half_type_2023"].to_numpy(dtype=np.int64) - 1
    hb = nd["half_type_2024"].to_numpy(dtype=np.int64) - 1
    rel = nd["reliable"].astype(bool).to_numpy()
    out: dict[str, RunOut] = {}
    obs_main = {"main": Observed(ha, hb, rel)}
    wa = labels_dyn[(PL.MAIN, w23)] - 1
    wb = labels_dyn[(PL.MAIN, w24)] - 1
    s = {
        tag: T.aligned(wa if "2023" in tag else wb, got[tag][0], k)
        for tag, _ in sens_halves_tasks(ctx.schemes)
    }
    obs_main["sensitivity"] = observed_from(
        {
            "odd_2023": s["sens_2023_0"],
            "even_2023": s["sens_2023_1"],
            "odd_2024": s["sens_2024_0"],
            "even_2024": s["sens_2024_1"],
        }
    )
    pres = PL.run_tasks(ctx, pair_tasks(sets, final), PL.pair_task, workers)
    main = RunOut(
        name="main",
        kind="main",
        n_nodes=int(ctx.ns.n),
        ids=ctx.ns.ids.copy(),
        final=final,
        type_2023=nd["type_2023"].to_numpy(dtype=np.int64) - 1,
        observed=obs_main,
        placebo=collect_pairs(pres, PL.MAIN, ctx.schemes),
        X_place=ctx.base.X,
        basket=ctx.base.B,
        qc=qc,
    )
    # способы прослеживания R1 (robustness.tracking)
    if ctx.space is not None:
        fixed_h = half_types(PL.FIXED)
        obs_f = {"main": observed_from(fixed_h)}
        zs = {tag: got[tag][1] for tag, _ in sens_halves_tasks(ctx.schemes)}
        fs = {tag: PL.blind(T.nearest(z, ctx.C24), ctx, "obs-fixed", tag) for tag, z in zs.items()}
        obs_f["sensitivity"] = observed_from(
            {
                "odd_2023": fs["sens_2023_0"],
                "even_2023": fs["sens_2023_1"],
                "odd_2024": fs["sens_2024_0"],
                "even_2024": fs["sens_2024_1"],
            }
        )
        out[PL.FIXED] = replace(
            main,
            name=PL.FIXED,
            kind="tracking",
            type_2023=nd["fixed_2023"].to_numpy(dtype=np.int64) - 1,
            observed=obs_f,
            placebo=collect_pairs(pres, PL.FIXED, ctx.schemes),
        )
        evo_h = half_types(PL.EVO)
        obs_e = {"main": observed_from(evo_h)}
        # центры окон 2023 и 2024 годов эволюционного способа этап dynamics не сохраняет: способ проходится
        # заново тем же кодом (dynamics.compute.evolutionary_track) от первого окна основного способа
        evo_tr, Cs, qc["refit_evolutionary"] = evo_refit(ctx, labels_dyn, got, roll, final_ari_min)
        es = {}
        for y, i in ((2023, 0), (2024, -1)):
            Cw = Cs[i]
            sw = np.bincount(evo_tr.rolling[i], minlength=k)
            for j in range(2):
                tag = f"sens_{y}_{j}"
                lab = T.evolutionary_kmeans(zs[tag], Cw, sw, ctx.epsilon, ctx.max_iter)[0]
                es[tag] = PL.blind(lab, ctx, "obs-evo", tag)
        obs_e["sensitivity"] = observed_from(
            {
                "odd_2023": es["sens_2023_0"],
                "even_2023": es["sens_2023_1"],
                "odd_2024": es["sens_2024_0"],
                "even_2024": es["sens_2024_1"],
            }
        )
        out[PL.EVO] = replace(
            main,
            name=PL.EVO,
            kind="tracking",
            type_2023=labels_dyn[(PL.EVO, w23)] - 1,
            observed=obs_e,
            placebo=collect_pairs(pres, PL.EVO, ctx.schemes),
        )
    main.seconds = time.perf_counter() - t0
    return main, out, qc
