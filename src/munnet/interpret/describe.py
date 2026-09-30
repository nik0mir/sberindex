"""Описания типов (не проверки): профиль, названия по правилу, примеры МО, дерево и формальные понятия, T4.

Профиль — медиана и межквартильный размах признака в типе против всех узлов, отклонение медианы в единицах
MAD × 1,4826, дельта Клиффа «тип против остальных» с бутстрап-интервалом; рядом — взвешенные населением
медианы («типичный житель»). p-значений по признакам кластеризации нет (Gao, Bien, Witten, 2022).
"""

from __future__ import annotations

import itertools
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from munnet.interpret import stats as S

# --- Профиль ---------------------------------------------------------------------------------------


def profile_table(
    values: pd.DataFrame,
    types: np.ndarray,
    weights: np.ndarray,
    type_order: Sequence[int],
    n_boot: int,
    level: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Строки «тип × признак» и «все × признак»: медиана, квартили, взвешенные медиана и квартили, отклонение
    медианы в MAD × 1,4826 всех узлов, дельта Клиффа «тип против остальных» с интервалом бутстрапа узлов,
    интервал медианы типа."""
    rows = []
    types = np.asarray(types)
    n = len(types)
    boots = [rng.integers(0, n, n) for _ in range(n_boot)]
    for col in values.columns:
        v = values[col].to_numpy(dtype=np.float64)
        ok = np.isfinite(v)
        med_all = float(np.median(v[ok]))
        mad_all = S.mad_scale(v[ok])
        rows.append(
            {
                "type": 0,
                "feature": col,
                "n": int(ok.sum()),
                "median": med_all,
                "q25": float(np.percentile(v[ok], 25)),
                "q75": float(np.percentile(v[ok], 75)),
                "wmedian": S.weighted_quantile(v, weights, 0.5),
                "wq25": S.weighted_quantile(v, weights, 0.25),
                "wq75": S.weighted_quantile(v, weights, 0.75),
                "effect_mad": 0.0,
                "cliff": np.nan,
                "cliff_lo": np.nan,
                "cliff_hi": np.nan,
                "median_lo": np.nan,
                "median_hi": np.nan,
            }
        )
        for t in type_order:
            m = (types == t) & ok
            rest = (types != t) & ok
            d = S.cliff_delta(v[m], v[rest])
            bd, bm = [], []
            for b in boots:
                vb, tb = v[b], types[b]
                okb = np.isfinite(vb)
                x, y = vb[okb & (tb == t)], vb[okb & (tb != t)]
                bd.append(S.cliff_delta(x, y))
                bm.append(float(np.median(x)) if len(x) else np.nan)
            lo, hi = S.perc_ci(np.array(bd), level)
            mlo, mhi = S.perc_ci(np.array(bm), level)
            med = float(np.median(v[m])) if m.any() else np.nan
            rows.append(
                {
                    "type": int(t),
                    "feature": col,
                    "n": int(m.sum()),
                    "median": med,
                    "q25": float(np.percentile(v[m], 25)) if m.any() else np.nan,
                    "q75": float(np.percentile(v[m], 75)) if m.any() else np.nan,
                    "wmedian": S.weighted_quantile(v[m], weights[m], 0.5),
                    "wq25": S.weighted_quantile(v[m], weights[m], 0.25),
                    "wq75": S.weighted_quantile(v[m], weights[m], 0.75),
                    "effect_mad": (med - med_all) / mad_all if mad_all > 0 else np.nan,
                    "cliff": d,
                    "cliff_lo": lo,
                    "cliff_hi": hi,
                    "median_lo": mlo,
                    "median_hi": mhi,
                }
            )
    return pd.DataFrame(rows)


# --- Названия --------------------------------------------------------------------------------------

# Служебные слова не считаются в длине названия (``naming.words``: «предлоги и союзы не считаются»).
FUNCTION_WORDS = frozenset(
    {"и", "в", "во", "к", "ко", "от", "с", "со", "по", "на", "а", "или", "без", "для", "у"}
)


def count_words(text: str) -> int:
    words = [w.strip(",.;:«»") for w in text.replace("—", " ").split()]
    return sum(1 for w in words if w and w.lower() not in FUNCTION_WORDS and not w.endswith(":"))


def settlement_shares(
    types: np.ndarray,
    mo_type: np.ndarray,
    pop: np.ndarray,
    urban: np.ndarray,
    rules: Mapping[str, Mapping],
    type_order: Sequence[int],
) -> pd.DataFrame:
    """Доли узлов типа, выполняющих условие слова о поселении (``naming.settlement_words``).

    large_cities — городской округ (узел-город засчитывается как go) с населением не меньше pop_min;
    cities — абсолютная доля горожан не меньше urban_share_min; rural — меньше urban_share_below.
    Доли — по узлам
    со значением; пропуски доли горожан — отдельный столбец."""
    rows = []
    for t in type_order:
        m = types == t
        row = {"type": int(t), "n": int(m.sum()), "urban_missing": int((m & ~np.isfinite(urban)).sum())}
        for word, r in rules.items():
            if word == "priority":
                continue
            if "mo_type" in r:
                go = np.isin(mo_type, [str(r["mo_type"]), "city"])
                cond = go & (pop >= float(r["pop_min"]))
                base = m
            elif "urban_share_min" in r:
                cond = urban >= float(r["urban_share_min"])
                base = m & np.isfinite(urban)
            elif "urban_share_below" in r:
                cond = urban < float(r["urban_share_below"])
                base = m & np.isfinite(urban)
            else:
                raise ValueError(f"naming.settlement_words.{word}: неизвестное условие {dict(r)}")
            row[word] = float((cond & base).sum() / base.sum()) if base.any() else np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index("type")


@dataclass
class TypeName:
    type: int
    name: str
    caption: str
    parts: list[str]
    place: str | None
    settlement: str | None
    words: int
    fallback: bool = False


# Слова, записанные в предрегистрации только в комментариях блока naming (ключей для них нет):
# «не нашлось ни одной части корзины — «без выраженных отличий корзины»» и слова о поселении.
NO_PARTS_TEXT = "без выраженных отличий корзины"
SETTLEMENT_LABELS: dict[str, str] = {
    "large_cities": "крупные города",
    "cities": "города",
    "rural": "сельские",
}


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:] if s else s


def _word(dic: Mapping, prof_t: pd.DataFrame, f: str) -> str:
    return str(dic[f]["pos" if prof_t.loc[f, "effect_mad"] > 0 else "neg"])


def name_types(
    prof: pd.DataFrame,
    parts: Sequence[str],
    place_features: Sequence[str],
    naming: Mapping,
    shares: pd.DataFrame,
    ladder: Sequence[int],
    unstable: set[tuple[int, str]],
) -> dict[int, TypeName]:
    """Название по правилу ``naming``: части корзины с наибольшим |отклонением медианы| (|δ| не меньше
    min_abs_cliff, знак устойчив в R1 — иначе следующая) и признак места; слово о поселении заменяет признак
    места; больше ``words[1]`` слов — признак места опускается; одинаковые названия — плюс часть корзины 2,
    совпадение осталось — ``test.fallback``."""
    n_parts = int(naming["basket_parts"])
    n_place = int(naming["place_features"])
    min_cliff = float(naming["min_abs_cliff"])
    _, wmax = (int(x) for x in naming["words"])
    dic = naming["dictionary"]
    rules = naming["settlement_words"]
    priority = [str(w) for w in rules["priority"]]
    qualifier = str(naming["qualifier"])
    out: dict[int, TypeName] = {}
    for pos, t in enumerate(ladder):
        p = prof.loc[prof["type"] == t].set_index("feature")

        def pick(feats: Sequence[str], count: int, skip: set[str], p=p) -> list[str]:
            cand = [f for f in feats if f in p.index and f not in skip]
            cand = [f for f in cand if np.isfinite(p.loc[f, "cliff"]) and abs(p.loc[f, "cliff"]) >= min_cliff]
            cand.sort(key=lambda f: (-abs(p.loc[f, "effect_mad"]), f))
            return cand[:count]

        chosen = pick(parts, n_parts, {f for (tt, f) in unstable if tt == t})
        words = [_word(dic, p, f) for f in chosen]
        place = pick(place_features, n_place, set())
        place_word = _word(dic, p, place[0]) if place else None
        settle = None
        for word in priority:
            r = rules[word]
            share = shares.loc[t, word]
            if not (np.isfinite(share) and share >= float(r["share_min"])):
                continue
            nb = pos - 1 if str(r["compare_with"]) == "lower" else pos + 1
            if 0 <= nb < len(ladder) and share > shares.loc[ladder[nb], word]:
                settle = SETTLEMENT_LABELS[word]
                break
        lead = settle or place_word
        part1 = words[0] if words else NO_PARTS_TEXT
        name = f"{lead}, {part1}" if lead else part1
        if count_words(name) > wmax:
            lead = settle
            name = f"{lead}, {part1}" if lead else part1
        caption = "; ".join([*words[1:], qualifier])
        out[t] = TypeName(
            type=int(t),
            name=_cap(name),
            caption=_cap(caption),
            parts=list(chosen),
            place=place[0] if (place and lead == place_word and not settle) else None,
            settlement=settle,
            words=count_words(name),
        )
    dup = [v.name for v in out.values()]
    for t, v in out.items():
        if dup.count(v.name) > 1 and len(v.parts) > 1:
            p = prof.loc[prof["type"] == t].set_index("feature")
            v.name = f"{v.name}, {_word(dic, p, v.parts[1])}"
    dup = [v.name for v in out.values()]
    for t, v in out.items():
        if dup.count(v.name) > 1:
            v.name = fallback_name(t, v, prof, naming)
            v.fallback = True
    return out


def fallback_name(t: int, v: TypeName, prof: pd.DataFrame, naming: Mapping) -> str:
    """``naming.test.fallback`` — «Тип {n}: {часть корзины 1}, {часть корзины 2}». Уточнение реализации
    (не в предрегистрации): частей, прошедших правило названия, меньше двух — пустое место опускается."""
    p = prof.loc[prof["type"] == t].set_index("feature")
    ws = [_word(naming["dictionary"], p, f) for f in v.parts[:2]]
    tpl = str(naming["test"]["fallback"]).replace("{n}", str(t))
    if len(ws) < 2:
        tpl = tpl.replace(", {часть корзины 2}", "")
        ws = ws or [NO_PARTS_TEXT]
    out = tpl.replace("{часть корзины 1}", ws[0])
    return out.replace("{часть корзины 2}", ws[1]) if len(ws) > 1 else out


# --- Примеры ---------------------------------------------------------------------------------------


def centroid_distances(Z: np.ndarray, types: np.ndarray, type_order: Sequence[int]) -> np.ndarray:
    """Расстояния узлов до центроидов типов (столбцы — в порядке ``type_order``), как в правиле examples."""
    C = np.vstack([Z[types == t].mean(axis=0) for t in type_order])
    return np.sqrt(((Z[:, None, :] - C[None, :, :]) ** 2).sum(axis=2))


def margin_table(
    Z: np.ndarray, types: np.ndarray, ids: np.ndarray, type_order: Sequence[int]
) -> pd.DataFrame:
    """Пограничность каждого узла по правилу examples (описание, не проверка): ``margin`` — разность
    расстояний до двух ближайших центроидов, ``second_type`` — ближайший из двух, кроме своего типа
    (если свой центроид не ближайший, ``own_nearest`` = False, и второй тип — ближайший центроид)."""
    D = centroid_distances(Z, types, type_order)
    order = np.argsort(D, axis=1, kind="stable")[:, :2]
    srt = np.take_along_axis(D, order, axis=1)
    own = np.array([list(type_order).index(int(t)) for t in types])
    other = np.where(order[:, 0] == own, order[:, 1], order[:, 0])
    return pd.DataFrame(
        {
            "territory_id": np.asarray(ids).astype(np.int64),
            "type": np.asarray(types).astype(np.int64),
            "second_type": np.asarray(type_order)[other].astype(np.int64),
            "margin": srt[:, 1] - srt[:, 0],
            "own_nearest": order[:, 0] == own,
        }
    )


def typical_examples(
    Z: np.ndarray,
    types: np.ndarray,
    ids: np.ndarray,
    regions: np.ndarray,
    eligible: np.ndarray,
    type_order: Sequence[int],
    n_typical: int,
    n_border: int,
    one_per_region: bool,
) -> pd.DataFrame:
    """Типичные (ближайшие к центроиду типа, с рангом расстояния) и пограничные (наименьшая разность
    расстояний
    до двух ближайших центроидов, со вторым типом) узлы; равенство — меньший territory_id."""
    D = centroid_distances(Z, types, type_order)
    rows = []
    for j, t in enumerate(type_order):
        m = np.flatnonzero(types == t)
        d_own = D[m, j]
        rank = np.empty(len(m), dtype=np.int64)
        rank[np.lexsort((ids[m], d_own))] = np.arange(1, len(m) + 1)
        used: set = set()
        for i in m[np.lexsort((ids[m], d_own))]:
            if not eligible[i] or (one_per_region and regions[i] in used):
                continue
            used.add(regions[i])
            rows.append(
                {
                    "type": int(t),
                    "kind": "typical",
                    "pos": int(i),
                    "distance": float(D[i, j]),
                    "rank": int(rank[np.flatnonzero(m == i)[0]]),
                    "second_type": np.nan,
                }
            )
            if sum(r["type"] == t and r["kind"] == "typical" for r in rows) >= n_typical:
                break
        srt = np.sort(D[m], axis=1)
        gap = srt[:, 1] - srt[:, 0]
        used = set()
        for idx in np.lexsort((ids[m], gap)):
            i = m[idx]
            if not eligible[i] or (one_per_region and regions[i] in used):
                continue
            used.add(regions[i])
            near = np.argsort(D[i], kind="stable")[:2]
            other = type_order[int(near[1] if near[0] == j else near[0])]
            rows.append(
                {
                    "type": int(t),
                    "kind": "borderline",
                    "pos": int(i),
                    "distance": float(gap[idx]),
                    "rank": np.nan,
                    "second_type": int(other),
                }
            )
            if sum(r["type"] == t and r["kind"] == "borderline" for r in rows) >= n_border:
                break
    return pd.DataFrame(rows)


def largest_examples(
    types: np.ndarray,
    pop: np.ndarray,
    ids: np.ndarray,
    eligible: np.ndarray,
    type_order: Sequence[int],
    n: int,
) -> pd.DataFrame:
    rows = []
    for t in type_order:
        m = np.flatnonzero((types == t) & eligible & np.isfinite(pop))
        order = m[np.lexsort((ids[m], -pop[m]))][:n]
        rows.extend({"type": int(t), "kind": "largest", "pos": int(i)} for i in order)
    return pd.DataFrame(rows)


def flow_examples(
    dvec: np.ndarray,
    t_a: np.ndarray,
    t_b: np.ndarray,
    reliable: np.ndarray,
    ids: np.ndarray,
    regions: np.ndarray,
    eligible: np.ndarray,
    min_nodes: int,
    per_flow: int,
    one_per_region: bool,
) -> pd.DataFrame:
    """Примеры надёжных переходов: потоки не меньше ``min_nodes``; ближайшие к медианному вектору
    изменения."""
    rows = []
    flows = sorted({(int(a), int(b)) for a, b in zip(t_a[reliable], t_b[reliable], strict=True)})
    for a, b in flows:
        m = np.flatnonzero(reliable & (t_a == a) & (t_b == b))
        if len(m) < min_nodes:
            continue
        med = np.median(dvec[m], axis=0)
        d = np.sqrt(((dvec[m] - med) ** 2).sum(axis=1))
        used: set = set()
        k = 0
        for idx in np.lexsort((ids[m], d)):
            i = m[idx]
            if not eligible[i] or (one_per_region and regions[i] in used):
                continue
            used.add(regions[i])
            rows.append(
                {"source": a, "dest": b, "n_flow": int(len(m)), "pos": int(i), "distance": float(d[idx])}
            )
            k += 1
            if k >= per_flow:
                break
    return pd.DataFrame(rows)


# --- Дерево и формальные понятия ------------------------------------------------------------------


@dataclass
class TreeResult:
    balanced_accuracy: float
    accuracy: float
    baseline_majority_bacc: float
    baseline_majority_acc: float
    baseline_random_bacc: float
    rules: pd.DataFrame
    oof: np.ndarray


def _collapse(tree, node: int, cls_of) -> tuple[bool, int]:
    """(лист ли после свёртки, класс листа). Разбиение, обе ветви которого ведут в один класс,
    сворачивается."""
    left, right = tree.children_left[node], tree.children_right[node]
    if left == -1:
        return True, cls_of(node)
    ll, lc = _collapse(tree, left, cls_of)
    rl, rc = _collapse(tree, right, cls_of)
    if ll and rl and lc == rc:
        return True, lc
    return False, -1


def tree_describe(
    Mx: np.ndarray,
    y: np.ndarray,
    names: Sequence[str],
    inverse: Mapping[str, tuple[float, float]],
    depth: int,
    class_weight: str,
    folds: int,
    seed: int,
    rules_per_type: Sequence[int],
    collapse: bool,
) -> TreeResult:
    """Дерево глубины ``depth`` (веса классов ``class_weight``): сбалансированная точность вне фолда, базовые
    линии, правила листьев «если …, то тип» с покрытием и точностью (пороги — в исходных единицах)."""
    from sklearn.metrics import accuracy_score, balanced_accuracy_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.tree import DecisionTreeClassifier

    tree = DecisionTreeClassifier(max_depth=depth, class_weight=class_weight, random_state=seed)
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        oof = cross_val_predict(tree, Mx, y, cv=cv)
        tree.fit(Mx, y)
    classes = np.unique(y)
    maj = classes[np.argmax([np.sum(y == c) for c in classes])]
    tr = tree.tree_
    leaf_of = tree.apply(Mx)
    counts: dict[int, np.ndarray] = {}

    def node_counts(node: int) -> np.ndarray:
        if node not in counts:
            below = _descendants(tr, node)
            m = np.isin(leaf_of, below)
            counts[node] = np.array([np.sum(y[m] == c) for c in classes])
        return counts[node]

    def cls_of(node: int) -> int:
        return int(classes[int(np.argmax(tr.value[node][0]))])

    leaves: list[tuple[int, list[tuple[str, str, float]]]] = []

    def walk(node: int, conds: list) -> None:
        is_leaf, _ = _collapse(tr, node, cls_of) if collapse else (tr.children_left[node] == -1, 0)
        if is_leaf:
            leaves.append((node, conds))
            return
        f = names[tr.feature[node]]
        thr = float(tr.threshold[node])
        walk(tr.children_left[node], [*conds, (f, "≤", thr)])
        walk(tr.children_right[node], [*conds, (f, ">", thr)])

    walk(0, [])
    rows = []
    totals = {c: int(np.sum(y == c)) for c in classes}
    for node, conds in leaves:
        cnt = node_counts(node)
        n_leaf = int(cnt.sum())
        cls = int(classes[int(np.argmax(cnt))]) if n_leaf else cls_of(node)
        n_cls = int(cnt[list(classes).index(cls)])
        rows.append(
            {
                "type": cls,
                "rule": _rule_text(conds, inverse),
                "n_leaf": n_leaf,
                "n_type": n_cls,
                "precision": n_cls / n_leaf if n_leaf else np.nan,
                "coverage": n_cls / totals[cls] if totals[cls] else np.nan,
            }
        )
    rules = pd.DataFrame(rows).sort_values(["type", "n_type", "precision"], ascending=[True, False, False])
    rules["rank_in_type"] = rules.groupby("type").cumcount() + 1
    rules = rules.loc[rules["rank_in_type"] <= int(max(rules_per_type))].reset_index(drop=True)
    rules["journalist"] = rules["rank_in_type"] == 1
    return TreeResult(
        balanced_accuracy=float(balanced_accuracy_score(y, oof)),
        accuracy=float(accuracy_score(y, oof)),
        baseline_majority_bacc=float(balanced_accuracy_score(y, np.full(len(y), maj))),
        baseline_majority_acc=float(np.mean(y == maj)),
        baseline_random_bacc=1.0 / len(classes),
        rules=rules,
        oof=oof,
    )


def _descendants(tr, node: int) -> list[int]:
    out, stack = [], [node]
    while stack:
        v = stack.pop()
        if tr.children_left[v] == -1:
            out.append(v)
        else:
            stack.extend([tr.children_left[v], tr.children_right[v]])
    return out


def _rule_text(conds: Sequence[tuple[str, str, float]], inverse: Mapping[str, tuple[float, float]]) -> str:
    """Условия пути, по признаку — самое узкое (≤ — наименьший порог, > — наибольший), в исходных единицах."""
    from munnet.interpret.labels import feature_label

    lo: dict[str, float] = {}
    hi: dict[str, float] = {}
    for f, op, thr in conds:
        med, mad = inverse.get(f, (0.0, 1.0))
        raw = med + mad * thr
        if op == "≤":
            hi[f] = min(hi.get(f, np.inf), raw)
        else:
            lo[f] = max(lo.get(f, -np.inf), raw)
    parts = []
    for f in dict.fromkeys([c[0] for c in conds]):
        lab = feature_label(f)
        if f in lo and f in hi:
            parts.append(f"{lab} от {S_fmt(lo[f])} до {S_fmt(hi[f])}")
        elif f in lo:
            parts.append(f"{lab} > {S_fmt(lo[f])}")
        else:
            parts.append(f"{lab} ≤ {S_fmt(hi[f])}")
    return " и ".join(parts) if parts else "все узлы"


def S_fmt(x: float) -> str:  # noqa: N802 — короткое имя форматтера порога
    from munnet import style

    return style.fmt_num(x, 2)


def fca_implications(
    Mx: np.ndarray,
    y: np.ndarray,
    names: Sequence[str],
    type_order: Sequence[int],
    max_itemset: int,
    precision_min: float,
    coverage_min: float,
    n_boot: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Формальные понятия (своя реализация на numpy): бинаризация по терцилям узлов, замкнутые наборы до
    ``max_itemset`` признаков, импликации «набор ⇒ тип» с точностью и покрытием не ниже порогов;
    устойчивость —
    доля бутстрап-выборок узлов, где импликация сохраняет оба порога (бинаризация фиксирована)."""
    n, p = Mx.shape
    levels = ("низкий", "средний", "высокий")
    items, cols = [], []
    for j in range(p):
        q1, q2 = np.percentile(Mx[:, j], [100 / 3, 200 / 3])
        v = Mx[:, j]
        for lev, mask in zip(levels, (v <= q1, (v > q1) & (v <= q2), v > q2), strict=True):
            items.append((j, lev))
            cols.append(mask)
    I = np.column_stack(cols)  # noqa: E741 — матрица объектов × признаков контекста
    by_feat = {j: [i for i, (jj, _) in enumerate(items) if jj == j] for j in range(p)}
    found = []
    for size in range(1, max_itemset + 1):
        for feats in itertools.combinations(range(p), size):
            for combo in itertools.product(*[by_feat[f] for f in feats]):
                ext = np.logical_and.reduce([I[:, c] for c in combo])
                n_ext = int(ext.sum())
                if not n_ext:
                    continue
                closure = set(np.flatnonzero(I[ext].all(axis=0)).tolist())
                if closure != set(combo):
                    continue
                for t in type_order:
                    tt = y == t
                    hit = int((ext & tt).sum())
                    prec, cov = hit / n_ext, hit / int(tt.sum())
                    if prec >= precision_min and cov >= coverage_min:
                        found.append((combo, t, ext, prec, cov, n_ext))
    rows = []
    if found:
        E = np.vstack([f[2] for f in found]).astype(np.float64)  # импликации × узлы
        Tm = np.vstack([(y == f[1]) for f in found]).astype(np.float64)
        ok_counts = np.zeros(len(found))
        for _ in range(n_boot):
            w = np.bincount(rng.integers(0, n, n), minlength=n).astype(np.float64)
            ext_w = E @ w
            hit_w = (E * Tm) @ w
            type_w = Tm @ w
            prec = np.divide(hit_w, ext_w, out=np.zeros_like(hit_w), where=ext_w > 0)
            cov = np.divide(hit_w, type_w, out=np.zeros_like(hit_w), where=type_w > 0)
            ok_counts += (prec >= precision_min) & (cov >= coverage_min)
        from munnet.interpret.labels import feature_label

        for (combo, t, _, prec, cov, n_ext), okc in zip(found, ok_counts, strict=True):
            text = " ∧ ".join(f"{feature_label(names[items[c][0]]).lower()} — {items[c][1]}" for c in combo)
            rows.append(
                {
                    "type": int(t),
                    "itemset": text,
                    "size": len(combo),
                    "extent": n_ext,
                    "precision": prec,
                    "coverage": cov,
                    "stability": okc / n_boot,
                }
            )
    cols_out = ["type", "itemset", "size", "extent", "precision", "coverage", "stability"]
    return pd.DataFrame(rows, columns=cols_out)


# --- T4 --------------------------------------------------------------------------------------------


def ari_on(a: np.ndarray, b: np.ndarray) -> float:
    return float(adjusted_rand_score(a, b))
