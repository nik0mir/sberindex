"""Внешняя проверка типов (``clustering.validation``) — НЕ критерий выбора: считается после выбора.

Показатели не входят ни в рёбра, ни в атрибуты (ИП и организации на 1000 жителей, ночёвки в коллективных
средствах размещения на жителя); преобразование ``log1p_rel``: ln(1 + x) минус медиана группы региона.
Нулевая модель всех проверок — перестановки меток типов внутри групп региона (состав групп сохраняется).

1. **Различие**: ε² Краскела — Уоллиса показателя по типам (ε² = H / (n − 1)) против перестановок.
2. **Знаки**: ρ Спирмена по узлам между показателем и средним по типу узла значением оси профиля;
   p — односторонне в ожидаемую сторону (``expected``).
3. **Сверх атрибутов**: прирост R² на перекрёстной проверке от добавления меток типа (индикаторы) к линейной
   модели рангов показателя на 11 признаках X; p — по перестановкам меток.

Поправка Бенджамини — Хохберга — по всем проверкам блока одного разбиения. Показатель, у которого |ρ| с
``log_wage_rel`` выше ``wage_link_max``, помечается «связан с зарплатой».
"""

from __future__ import annotations

import zlib
from collections.abc import Mapping

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from munnet.network.rules import bh_qvalues


def log1p_rel(values: pd.Series, groups: np.ndarray) -> pd.Series:
    """ln(1 + x) минус медиана своей группы региона (пропуск остаётся пропуском)."""
    v = np.log1p(values.astype("float64"))
    med = v.groupby(np.asarray(groups)).transform("median")
    return v - med.to_numpy()


def permute_within(labels: np.ndarray, groups: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Метки, переставленные внутри каждой группы: узлы группы получают метки той же группы в случайном
    порядке (сортировка по группе и случайному ключу)."""
    labels = np.asarray(labels)
    groups = np.asarray(groups)
    by_group = np.argsort(groups, kind="stable")
    shuffled = np.lexsort((rng.random(len(labels)), groups))
    out = np.empty_like(labels)
    out[by_group] = labels[shuffled]
    return out


def _codes(labels: np.ndarray) -> tuple[np.ndarray, int]:
    _, inv = np.unique(labels, return_inverse=True)
    return inv, int(inv.max()) + 1


def epsilon2(y: np.ndarray, labels: np.ndarray, ranks: np.ndarray | None = None) -> float:
    """ε² Краскела — Уоллиса: H / (n − 1), H — по средним рангам (без поправки на связи)."""
    r = rankdata(y) if ranks is None else ranks
    n = len(r)
    inv, k = _codes(labels)
    cnt = np.bincount(inv, minlength=k)
    sums = np.bincount(inv, weights=r, minlength=k)
    ok = cnt > 0
    H = 12.0 / (n * (n + 1)) * float((sums[ok] ** 2 / cnt[ok]).sum()) - 3.0 * (n + 1)
    return H / (n - 1)


def type_mean(axis: np.ndarray, labels: np.ndarray) -> np.ndarray:
    inv, k = _codes(labels)
    cnt = np.bincount(inv, minlength=k)
    return (np.bincount(inv, weights=axis, minlength=k) / cnt)[inv]


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = rankdata(a), rankdata(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def type_mean_ranks(axis: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Средние ранги по узлам значения «среднее оси по типу узла» — как ``rankdata(type_mean(axis, labels))``,
    но через K уровней (быстро для перестановок)."""
    inv, k = _codes(labels)
    cnt = np.bincount(inv, minlength=k).astype(np.float64)
    vals = np.bincount(inv, weights=axis, minlength=k) / np.where(cnt > 0, cnt, 1.0)
    uniq, uinv = np.unique(vals, return_inverse=True)
    ucnt = np.bincount(uinv, weights=cnt)
    below = np.cumsum(ucnt) - ucnt
    return (below + (ucnt + 1.0) / 2.0)[uinv][inv]


def _pearson_rank(ry: np.ndarray, rx: np.ndarray) -> float:
    a, b = ry - ry.mean(), rx - rx.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else float("nan")


def _design(labels: np.ndarray) -> np.ndarray:
    inv, k = _codes(labels)
    D = np.zeros((len(labels), k))
    D[np.arange(len(labels)), inv] = 1.0
    return D[:, 1:]  # первый тип — базовый


class BeyondAttributes:
    """Прирост R² на перекрёстной проверке от индикаторов типа сверх линейной модели на X.

    Разбиения на части фиксируются один раз (``repeats`` × ``folds``); базовая модель [1, X] считается
    один раз на часть, прирост от индикаторов — через остатки (теорема Фриша — Во): β_D — МНК остатков Y
    на остатки D после [1, X] на обучающей части, прогноз полной модели = прогноз базовой + (D − [1, X]Γ)β_D.
    Поэтому перестановочный нуль (тысяча разбиений на типы) считается за секунды."""

    def __init__(self, y: np.ndarray, X: np.ndarray, n_folds: int, repeats: int, seed: int) -> None:
        self.Y = rankdata(y) / len(y)
        rng = np.random.default_rng([seed, 29])
        self.parts = []
        for _ in range(repeats):
            for tr, te in _folds(len(y), n_folds, rng):
                A_tr = np.column_stack([np.ones(len(tr)), X[tr]])
                A_te = np.column_stack([np.ones(len(te)), X[te]])
                G = np.linalg.pinv(A_tr.T @ A_tr)  # (AᵀA)⁻¹
                beta = G @ (A_tr.T @ self.Y[tr])
                self.parts.append(
                    {
                        "tr": tr,
                        "te": te,
                        "A_tr": A_tr,
                        "A_te": A_te,
                        "G": G,
                        "res_tr": self.Y[tr] - A_tr @ beta,
                        "pred_te": A_te @ beta,
                        "sst": float(((self.Y[te] - self.Y[tr].mean()) ** 2).sum()),
                    }
                )
        self.repeats = repeats
        self.n_folds = n_folds

    def _r2(self, D: np.ndarray | None) -> float:
        """Средний по повторам R² (сумма квадратов ошибок по частям повтора)."""
        out = []
        for r in range(self.repeats):
            sse = sst = 0.0
            for part in self.parts[r * self.n_folds : (r + 1) * self.n_folds]:
                pred = part["pred_te"]
                if D is not None and D.shape[1]:
                    Dtr = D[part["tr"]]
                    C = part["A_tr"].T @ Dtr  # (p + 1) × (K − 1)
                    GC = part["G"] @ C
                    M = Dtr.T @ Dtr - C.T @ GC  # D̃ᵀD̃ = DᵀD − (AᵀD)ᵀ(AᵀA)⁻¹(AᵀD)
                    b = Dtr.T @ part["res_tr"]  # D̃ᵀỸ = DᵀỸ: остатки ортогональны [1, X]
                    try:
                        bd = np.linalg.solve(M, b)
                    except np.linalg.LinAlgError:
                        bd = np.linalg.lstsq(M, b, rcond=None)[0]
                    pred = pred + D[part["te"]] @ bd - part["A_te"] @ (GC @ bd)
                sse += float(((self.Y[part["te"]] - pred) ** 2).sum())
                sst += part["sst"]
            out.append(1.0 - sse / sst)
        return float(np.mean(out))

    def base_r2(self) -> float:
        return self._r2(None)

    def gain(self, labels: np.ndarray) -> float:
        return self._r2(_design(labels)) - self.base_r2()


def _folds(n: int, n_folds: int, rng: np.random.Generator) -> list[tuple[np.ndarray, np.ndarray]]:
    perm = rng.permutation(n)
    parts = np.array_split(perm, n_folds)
    return [(np.sort(np.concatenate(parts[:i] + parts[i + 1 :])), np.sort(parts[i])) for i in range(n_folds)]


def validate(
    labels: np.ndarray,
    indicators: pd.DataFrame,
    axes: pd.DataFrame,
    X: np.ndarray,
    wage: np.ndarray,
    groups: np.ndarray,
    cfg: Mapping,
    seed: int,
    permutations: int | None = None,
    beyond_permutations: int | None = None,
) -> pd.DataFrame:
    """Все проверки одного разбиения. ``indicators`` — показатели после ``log1p_rel`` (строки — узлы),
    ``axes`` — оси профиля (строки — узлы). Строка ответа — проверка: вид, показатель, ось, статистика,
    ожидаемый знак, p, q (Бенджамини — Хохберг по блоку), число узлов, связь с зарплатой."""
    B = int(permutations or cfg["permutations"])
    Bb = int(beyond_permutations if beyond_permutations is not None else B)
    expected: Mapping[str, Mapping[str, int]] = cfg.get("expected") or {}
    ba = cfg.get("beyond_attributes") or {}
    wmax = float(cfg.get("wage_link_max", 0.5))
    rows = []
    for name in indicators.columns:
        y_all = indicators[name].to_numpy(dtype=np.float64)
        ok = np.isfinite(y_all)
        y, lab, grp = y_all[ok], np.asarray(labels)[ok], np.asarray(groups)[ok]
        n = int(ok.sum())
        wage_rho = spearman(y, wage[ok]) if n > 2 else float("nan")
        linked = bool(np.isfinite(wage_rho) and abs(wage_rho) > wmax)
        base = {"indicator": name, "n": n, "wage_rho": wage_rho, "wage_linked": linked}
        rng = np.random.default_rng([seed, 31, zlib.crc32(name.encode())])
        perms = [permute_within(lab, grp, rng) for _ in range(B)]
        ry = rankdata(y)
        # 1) различие
        e2 = epsilon2(y, lab, ry)
        null = np.array([epsilon2(y, pl, ry) for pl in perms])
        rows.append(
            {
                **base,
                "check": "difference",
                "axis": "",
                "stat": e2,
                "expected": np.nan,
                "null_mean": float(null.mean()),
                "p": (1 + int((null >= e2).sum())) / (1 + B),
            }
        )
        # 2) знаки
        for ax, sign in (expected.get(name) or {}).items():
            a = axes[ax].to_numpy(dtype=np.float64)[ok]
            rho = _pearson_rank(ry, type_mean_ranks(a, lab))
            null = np.array([_pearson_rank(ry, type_mean_ranks(a, pl)) for pl in perms])
            s = float(sign)
            rows.append(
                {
                    **base,
                    "check": "sign",
                    "axis": ax,
                    "stat": rho,
                    "expected": s,
                    "null_mean": float(np.nanmean(null)),
                    "p": (1 + int((s * null >= s * rho).sum())) / (1 + B),
                }
            )
        # 3) сверх атрибутов
        ba_model = BeyondAttributes(y, X[ok], int(ba.get("folds", 5)), int(ba.get("repeats", 20)), seed)
        base_r2 = ba_model.base_r2()
        gain = ba_model._r2(_design(lab)) - base_r2
        p_b = null_mean = np.nan
        if Bb > 0:
            nulls = np.array([ba_model._r2(_design(pl)) - base_r2 for pl in perms[:Bb]])
            p_b = (1 + int((nulls >= gain).sum())) / (1 + len(nulls))
            null_mean = float(nulls.mean())
        rows.append(
            {
                **base,
                "check": "beyond_attributes",
                "axis": "",
                "stat": gain,
                "expected": np.nan,
                "null_mean": null_mean,
                "p": p_b,
                "base_r2": base_r2,
            }
        )
    out = pd.DataFrame(rows)
    pv = out["p"].to_numpy(dtype=np.float64)
    q = np.full(len(pv), np.nan)
    fin = np.isfinite(pv)
    if fin.any():
        q[fin] = bh_qvalues(pv[fin])
    out["q"] = q
    out["significant"] = out["q"] <= float(cfg.get("alpha", 0.05))
    out["sign_ok"] = np.where(out["check"] == "sign", np.sign(out["stat"]) == out["expected"], np.nan)
    return out


def indicator_frame(
    ctx: pd.DataFrame, ids: np.ndarray, groups: np.ndarray, spec: Mapping[str, int]
) -> pd.DataFrame:
    """Показатели ``spec`` (показатель -> год) узлов ``ids`` после ``log1p_rel``."""
    out = pd.DataFrame(index=pd.Index(ids, name="territory_id"))
    for name, year in spec.items():
        v = ctx.loc[ctx["year"] == int(year)].set_index("territory_id")[name].reindex(ids).astype("float64")
        out[name] = log1p_rel(v, groups).to_numpy()
    return out


def coverage(ind: pd.DataFrame) -> dict[str, int]:
    return {c: int(ind[c].notna().sum()) for c in ind.columns}
