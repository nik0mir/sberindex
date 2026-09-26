"""Раздел разведки E4 «Корзина и маркетплейсы» (spec_final, Б.4, Д.3; вопросы Q12–Q17).

Что проверяет раздел:

- чем различаются корзины МО и не сводится ли состав трат к уровню трат и региону: доли шести частей,
  CLR, PCA на сырых CLR и на CLR, очищенных МНК от лог-уровня и дамми регионов, устойчивость очищенной
  главной оси между 2023 и 2024 годами (F09, F10, T08);
- как быстро растёт доля маркетплейсов и где быстрее — в двух шкалах сразу (п. п. и рубли), с защитой от
  регрессии к среднему (начальная доля по нечётным месяцам, прирост по чётным), согласованностью полугодий
  и тестом ступеньки на смену границ категорий (F11, F12, T09);
- расходятся ли уровни трат и насколько согласован рост МО по полугодиям (F13, T10).

Рост везде номинальный; реальный — только при заданном ``eda.cpi`` (F17 и колонка T10). «Типичное МО» —
медиана по МО без весов, «доля в тратах жителей» — ``national.resident_share`` (веса — население).
Где ориентир спецификации (Б.4) посчитан другим определением, чем закреплено в Б.1, Б.2 и В.5, значение
по определению ориентира пишется в ``note`` факта (``reference_variants``): расхождение объяснено числом.
Чистые функции — вверху модуля и тестируются отдельно, рисование — в ``_fig_*``, сборка — ``run_section``.
Раздел читает только ``ctx.data`` и параметры конфига (``eda.basket`` поверх ``BASKET_DEFAULTS``).
"""

from __future__ import annotations

import copy
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from string import Template
from typing import Any

import numpy as np
import pandas as pd
from matplotlib import patheffects
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.text import Text
from matplotlib.transforms import Bbox

from munnet import style
from munnet.contracts import CATEGORY_CODES, PARTS, YEARS
from munnet.eda import stats
from munnet.eda.base import Fact, FactKind, Finding, SectionContext, make_fact

SECTION_ID = "e4"
TITLE = "Корзина и маркетплейсы"

MP = "marketplace"
Y0, Y1 = YEARS
MONTHS: tuple[int, ...] = tuple(range(1, 13))
FIRST_HALF: tuple[int, ...] = MONTHS[:6]  # январь–июнь
SECOND_HALF: tuple[int, ...] = MONTHS[6:]  # июль–декабрь
ODD_MONTHS: tuple[int, ...] = MONTHS[0::2]  # январь, март, …, ноябрь — начальная доля или уровень
EVEN_MONTHS: tuple[int, ...] = MONTHS[1::2]  # февраль, апрель, …, декабрь — прирост
PP = 100.0  # доля -> процентные пункты
MONTHS_FULL_RU: tuple[str, ...] = (
    "январь",
    "февраль",
    "март",
    "апрель",
    "май",
    "июнь",
    "июль",
    "август",
    "сентябрь",
    "октябрь",
    "ноябрь",
    "декабрь",
)
# Деления логарифмической оси уровня трат, ₽ в месяц: берутся те, что попадают в диапазон данных.
RUB_TICKS: tuple[int, ...] = (5_000, 10_000, 15_000, 20_000, 30_000, 50_000, 80_000, 120_000)
PC_COLUMNS_PREFIX = "PC"
CLR_VAR_DECIMALS = 3  # дисперсия CLR — три знака, как в факте ``clr_var_<часть>`` (вид num3)

# Параметры раздела. Источник истины — секция ``eda.basket`` конфига; здесь — значения по умолчанию на время,
# пока секции нет (каждый ключ секции переопределяет ключ отсюда, неизвестный ключ — ошибка).
BASKET_DEFAULTS: dict[str, Any] = {
    "n_components": 3,  # главных компонент CLR в T08 (PC1–PC3)
    "level_groups": 5,  # пятые части МО по уровню трат 2023 года (F12)
    "label_n": 6,  # подписей МО на точечных графиках F10 и F13 (В.2: не больше 5–7)
    "label_box": [0.35, 0.12],  # размер подписи в долях размаха осей: подписанные точки не ближе
    "trend_bins": 10,  # линия тренда на F10 и F13 — медиана по десятым частям уровня трат
    "display_quantiles": [0.0, 1.0],  # ось роста на F13; при [0,005; 0,995] хвосты рисуются у края
    "weak_rho": 0.1,  # |ρ| меньше — в тексте «связи почти нет» (как «незначимые» клетки F14)
    "headline": {  # проверки заголовков (spec_final, В.5)
        "pc1_rho_min": 0.6,  # F10: |ρ PC1 с лог-уровнем| не меньше
        "mp_growth_ratio_min": 1.8,  # F11: доля декабря 2024 / доля января 2023 не меньше («вдвое»)
        "quintile_gap_pp_min": 1.0,  # F12: прирост нижней пятой части минус верхней, п. п., не меньше
        "growth_rho_min": 0.2,  # F13: ρ роста с уровнем в целом и внутри регионов не меньше
    },
}

# Группы МО равной численности (F12 — по уровню трат, линии тренда F10 и F13 — по десятым частям):
# формы, которые нужны текстам, — им. мн., род. ед., дат. мн., им. ед. Другое число групп текст назвать
# не умеет, поэтому ``basket_params`` его не пропускает.
GROUP_WORDS: dict[int, tuple[str, str, str, str]] = {
    4: ("четверти", "четверти", "четвертям", "четверть"),
    5: ("пятые части", "пятой части", "пятым частям", "пятая часть"),
    10: ("десятые части", "десятой части", "десятым частям", "десятая часть"),
}

_SINGLE_LETTER = re.compile(r"(^|[\s(«„])([ВвСсКкИиАаОоУуЯя]) ")
# Сокращение названий регионов в подписях точек (порядок важен: сначала хвост «— Алания», «— Югра»).
_REGION_RULES: tuple[tuple[str, str], ...] = (
    (r"\s+—.*$", ""),
    (r"^Республика\s+", ""),
    (r"\s+Республика$", ""),
    (r"автономный округ", "АО"),
    (r"автономная область", "авт. обл."),
    (r"область", "обл."),
)


# --- Параметры и текст ---------------------------------------------------------------------------


def basket_params(cfg: Any) -> dict[str, Any]:
    """Параметры раздела: ``BASKET_DEFAULTS``, поверх — ``eda.basket`` конфига (вложенные словари сливаются).

    Неизвестный ключ в ``eda.basket`` — ``ValueError``: опечатка не должна молча оставлять значение
    по умолчанию.
    """
    out = copy.deepcopy(BASKET_DEFAULTS)
    user = cfg["eda"].get("basket") or {}
    unknown = sorted(set(user) - set(BASKET_DEFAULTS))
    if unknown:
        raise ValueError(f"eda.basket: неизвестные ключи {unknown}; допустимы {sorted(BASKET_DEFAULTS)}")
    for key, value in user.items():
        if isinstance(value, Mapping) and isinstance(out[key], dict):
            bad = sorted(set(value) - set(out[key]))
            if bad:
                raise ValueError(f"eda.basket.{key}: неизвестные ключи {bad}")
            out[key].update(value)
        else:
            out[key] = value
    for key in ("level_groups", "trend_bins"):
        if int(out[key]) not in GROUP_WORDS:
            raise ValueError(f"eda.basket.{key} = {out[key]}: допустимы {sorted(GROUP_WORDS)}")
    return out


def nbsp(text: str) -> str:
    """Типографика ru-text: неразрывный пробел после однобуквенных слов (в, с, к, и, а, о, у, я)
    и перед тире."""
    previous = None
    while previous != text:  # «и в МО»: два однобуквенных подряд
        previous = text
        text = _SINGLE_LETTER.sub(lambda m: f"{m.group(1)}{m.group(2)}{style.NBSP}", text)
    return text.replace(" — ", f"{style.NBSP}— ")


# --- Корзина: доли, CLR, PCA ---------------------------------------------------------------------


def basket_sample(mo: pd.DataFrame) -> pd.DataFrame:
    """МО с 12 месяцами в обоих годах: доли, CLR и уровень известны за 2023 и 2024 годы.

    Индекс — ``territory_id``.
    Это выборка сюжета С4 (факт ``n_basket``): на ней считаются F09, T08 и устойчивость очищенной оси.
    """
    cols = [f"{k}_{p}_{y}" for k in ("sh", "clr") for p in PARTS for y in YEARS]
    cols += [f"log_level_{y}" for y in YEARS]
    return mo.dropna(subset=cols).set_index("territory_id")


def part_frame(frame: pd.DataFrame, prefix: str, year: int) -> pd.DataFrame:
    """Колонки ``<prefix>_<часть>_<год>`` как таблица с колонками-частями (``PARTS``)."""
    return frame[[f"{prefix}_{p}_{year}" for p in PARTS]].set_axis(list(PARTS), axis=1)


def basket_summary(shares: pd.DataFrame, clr: pd.DataFrame) -> pd.DataFrame:
    """По частям корзины: медиана и квартили доли по МО, дисперсия CLR; сортировка по медиане доли (убывание).

    Индекс — код части, колонки: ``label``, ``median``, ``q25``, ``q75``, ``clr_var`` (дисперсия с ddof = 1).
    """
    out = pd.DataFrame(
        {
            "label": [style.LABELS[p] for p in shares.columns],
            "median": shares.median(),
            "q25": shares.quantile(0.25),
            "q75": shares.quantile(0.75),
            "clr_var": clr[shares.columns].var(ddof=1),
        },
        index=shares.columns,
    )
    return out.sort_values("median", ascending=False)


def design_matrix(controls: pd.DataFrame) -> np.ndarray:
    """Матрица регрессоров МНК: свободный член, числовые колонки как есть, категориальные — дамми без первой.

    Категориальными считаются колонки типов category, object, string и bool (регион передавайте категорией:
    целочисленный код региона иначе войдёт как число).
    """
    blocks = [np.ones((len(controls), 1))]
    for col in controls.columns:
        s = controls[col]
        if (
            isinstance(s.dtype, pd.CategoricalDtype)
            or pd.api.types.is_object_dtype(s)
            or pd.api.types.is_string_dtype(s)
            or pd.api.types.is_bool_dtype(s)
        ):
            dummies = pd.get_dummies(s.astype("category"), drop_first=True, dtype="float64")
            blocks.append(dummies.to_numpy())
        else:
            blocks.append(s.to_numpy(dtype="float64")[:, None])
    return np.column_stack(blocks)


def residualize(y: pd.DataFrame, controls: pd.DataFrame) -> tuple[pd.DataFrame, float, pd.Series]:
    """Остатки МНК каждой колонки ``y`` на ``controls`` (со свободным членом) и доля объяснённой дисперсии.

    Возвращает (остатки, R² суммарной дисперсии всех колонок, R² по колонкам). Строки ``y`` и ``controls`` —
    в одном порядке и без пропусков. Для CLR строки остатков в сумме дают 0, как и строки CLR.
    """
    if len(y) != len(controls):
        raise ValueError("y и controls разной длины")
    values = y.to_numpy(dtype="float64")
    x = design_matrix(controls.reset_index(drop=True))
    beta, *_ = np.linalg.lstsq(x, values, rcond=None)
    resid = values - x @ beta
    centered = values - values.mean(axis=0)
    ss_tot = (centered**2).sum(axis=0)
    ss_res = (resid**2).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        by_col = pd.Series(1.0 - ss_res / ss_tot, index=y.columns)
    total = float(1.0 - ss_res.sum() / ss_tot.sum()) if ss_tot.sum() > 0 else float("nan")
    return pd.DataFrame(resid, index=y.index, columns=y.columns), total, by_col


@dataclass(frozen=True)
class PcaResult:
    """PCA матрицы CLR (сырой или очищенной от контролей).

    ``explained`` — доли дисперсии анализируемой матрицы (у очищенной — от дисперсии остатков); ``r2`` —
    доля суммарной дисперсии CLR, объяснённая контролями (0 без контролей), ``residual_share`` = 1 − ``r2``.
    Знак компоненты условен: наибольшая по модулю нагрузка положительна.
    """

    scores: pd.DataFrame  # индекс — строки без пропусков; колонки PC1…PCk
    loadings: pd.DataFrame  # индекс — части; колонки PC1…PCk
    explained: pd.Series  # PC1…PCk -> доля дисперсии
    total_var: float  # суммарная дисперсия CLR до очистки (сумма дисперсий колонок, ddof = 1)
    r2: float
    r2_by_part: pd.Series  # R² контролей по частям (NaN без контролей)
    center: pd.Series  # средние колонок анализируемой матрицы — для проекции других лет

    @property
    def residual_share(self) -> float:
        """Доля дисперсии CLR, оставшаяся после очистки от контролей."""
        return 1.0 - self.r2

    def project(self, matrix: pd.DataFrame) -> pd.DataFrame:
        """Счета строк ``matrix`` (те же колонки-части) на осях этого PCA."""
        centered = matrix[self.loadings.index] - self.center
        return pd.DataFrame(
            centered.to_numpy() @ self.loadings.to_numpy(), index=matrix.index, columns=self.loadings.columns
        )


def pca_clr(clr: pd.DataFrame, controls: pd.DataFrame | None = None, n_components: int = 3) -> PcaResult:
    """PCA на CLR (SVD центрированной матрицы); при ``controls`` — на остатках МНК CLR на контроли.

    Строки с пропуском в CLR или контролях отбрасываются (счета — только по оставшимся строкам).
    """
    frame = clr.copy()
    if controls is not None:
        controls = controls.loc[clr.index]
        ok = frame.notna().all(axis=1) & controls.notna().all(axis=1)
        frame, controls = frame.loc[ok], controls.loc[ok]
    else:
        frame = frame.dropna()
    if len(frame) <= n_components:
        raise ValueError(f"PCA: строк {len(frame)}, нужно больше {n_components}")
    total_var = float(frame.var(ddof=1).sum())
    if controls is None:
        matrix, r2, r2_by = frame, 0.0, pd.Series(np.nan, index=frame.columns)
    else:
        matrix, r2, r2_by = residualize(frame, controls)
    center = matrix.mean()
    centered = (matrix - center).to_numpy()
    _, sing, vt = np.linalg.svd(centered, full_matrices=False)
    var = sing**2 / (len(matrix) - 1)
    k = min(n_components, vt.shape[0])
    vt = vt[:k]
    signs = np.sign(vt[np.arange(k), np.abs(vt).argmax(axis=1)])
    vt = vt * signs[:, None]
    names = [f"{PC_COLUMNS_PREFIX}{i + 1}" for i in range(k)]
    total = var.sum()
    explained = pd.Series(var[:k] / total if total > 0 else np.nan, index=names)
    return PcaResult(
        scores=pd.DataFrame(centered @ vt.T, index=matrix.index, columns=names),
        loadings=pd.DataFrame(vt.T, index=matrix.columns, columns=names),
        explained=explained,
        total_var=total_var,
        r2=float(r2),
        r2_by_part=r2_by,
        center=center,
    )


def region_controls(
    frame: pd.DataFrame, year: int, *, level: bool = True, region: bool = True
) -> pd.DataFrame:
    """Контроли очистки CLR года: ``log_level_<год>`` и регион категорией (для дамми)."""
    cols: dict[str, pd.Series] = {}
    if level:
        cols["log_level"] = frame[f"log_level_{year}"].astype("float64")
    if region:
        cols["region"] = frame["region_code"].astype("category")
    return pd.DataFrame(cols, index=frame.index)


@dataclass(frozen=True)
class BasketResult:
    """Всё о корзине для T08, F09, F10: сводка частей, сырой и очищенный PCA, R², устойчивость."""

    summary: pd.DataFrame
    raw: PcaResult
    clean: PcaResult
    r2_level: float
    r2_region: float
    r2_level_by_part: pd.Series
    r2_region_by_part: pd.Series
    stability: float  # ρ счёта очищенной PC1 2024 года и счёта 2023 года на той же оси
    axis_cos: float  # |cos| нагрузок очищенной PC1, посчитанной по 2023 и по 2024 году отдельно
    rho_pc1_level: float  # ρ Спирмена счёта PC1 сырых CLR 2024 года с log_level_2024
    level: pd.Series  # level_2024 выборки (₽) — ось X рисунка F10
    n: int


PC1 = f"{PC_COLUMNS_PREFIX}1"


def basket_analysis(mo: pd.DataFrame, n_components: int = 3) -> BasketResult:
    """Корзина 2024 года на выборке ``basket_sample``: сводка частей, PCA сырых и очищенных CLR, R²,
    устойчивость.

    Очистка — МНК каждой CLR на лог-уровень своего года и дамми регионов. Устойчивость — ρ Спирмена счёта
    очищенной PC1 2024 года и счёта 2023 года на той же оси (остатки 2023 года — от своей регрессии).
    """
    sample = basket_sample(mo)
    sh24, clr24 = part_frame(sample, "sh", Y1), part_frame(sample, "clr", Y1)
    clr23 = part_frame(sample, "clr", Y0)
    raw = pca_clr(clr24, n_components=n_components)
    clean = pca_clr(clr24, region_controls(sample, Y1), n_components=n_components)
    _, r2_level, r2_level_by = residualize(clr24, region_controls(sample, Y1, region=False))
    _, r2_region, r2_region_by = residualize(clr24, region_controls(sample, Y1, level=False))
    resid23, _, _ = residualize(clr23, region_controls(sample, Y0))
    score23 = clean.project(resid23)[PC1]
    stability, _ = stats.spearman(clean.scores[PC1], score23.loc[clean.scores.index])
    clean23 = pca_clr(clr23, region_controls(sample, Y0), n_components=1)
    axis_cos = float(abs(clean.loadings[PC1] @ clean23.loadings[PC1]))
    rho_pc1, _ = stats.spearman(raw.scores[PC1], sample.loc[raw.scores.index, f"log_level_{Y1}"])
    return BasketResult(
        summary=basket_summary(sh24, clr24),
        raw=raw,
        clean=clean,
        r2_level=r2_level,
        r2_region=r2_region,
        r2_level_by_part=r2_level_by,
        r2_region_by_part=r2_region_by,
        stability=float(stability),
        axis_cos=axis_cos,
        rho_pc1_level=float(rho_pc1),
        level=sample.loc[raw.scores.index, f"level_{Y1}"],
        n=len(sample),
    )


def basket_table(res: BasketResult) -> pd.DataFrame:
    """T08: строка на часть корзины (медиана доли, дисперсия CLR, нагрузки PC сырых CLR, R² уровня, региона
    и обоих, нагрузка очищенной PC1) и итоговая строка ``total``: те же колонки как доли дисперсии."""
    parts = list(res.summary.sort_values("clr_var", ascending=False).index)
    pcs = list(res.raw.loadings.columns)
    rows = []
    for p in parts:
        row = {
            "code": p,
            "label": style.LABELS[p],
            "median_share_2024": res.summary.loc[p, "median"],
            "clr_var": res.summary.loc[p, "clr_var"],
        }
        row.update({f"raw_{pc.lower()}": res.raw.loadings.loc[p, pc] for pc in pcs})
        row.update(
            {
                "r2_level": res.r2_level_by_part[p],
                "r2_region": res.r2_region_by_part[p],
                "r2_both": res.clean.r2_by_part[p],
                "clean_pc1": res.clean.loadings.loc[p, PC1],
            }
        )
        rows.append(row)
    total = {
        "code": "total",
        "label": "Итого: сумма дисперсий CLR, дальше — доли дисперсии",
        "median_share_2024": np.nan,
        "clr_var": res.raw.total_var,
    }
    total.update({f"raw_{pc.lower()}": res.raw.explained[pc] for pc in pcs})
    total.update(
        {
            "r2_level": res.r2_level,
            "r2_region": res.r2_region,
            "r2_both": res.clean.r2,
            "clean_pc1": res.clean.explained[PC1],
        }
    )
    rows.append(total)
    return pd.DataFrame(rows)


# --- Маркетплейсы --------------------------------------------------------------------------------


def month_sums(
    panel_wide: pd.DataFrame, columns: Sequence[str], year: int, months: Sequence[int]
) -> pd.DataFrame:
    """Суммы колонок ``columns`` по МО за месяцы ``months`` года; NaN, если у МО есть не все эти месяцы."""
    w = panel_wide.loc[(panel_wide["year"] == year) & panel_wide["month"].isin(list(months))]
    g = w.groupby("territory_id")
    sums = g[list(columns)].sum().astype("float64")
    return sums.where(g.size() == len(months), axis=0)


def month_share(panel_wide: pd.DataFrame, part: str, year: int, months: Sequence[int]) -> pd.Series:
    """Доля части за месяцы года: Σ ``v_<часть>`` / Σ ``v_all`` (как годовая доля ``mo``);
    NaN при неполноте."""
    s = month_sums(panel_wide, [f"v_{part}", "v_all"], year, months)
    return s[f"v_{part}"] / s["v_all"]


def annual_levels(panel_wide: pd.DataFrame) -> pd.DataFrame:
    """Среднемесячные траты по категориям за полный год: колонки ``<категория>_<год>``;
    NaN, если месяцев меньше 12."""
    cols = [f"v_{c}" for c in CATEGORY_CODES]
    out = {}
    for y in YEARS:
        sums = month_sums(panel_wide, cols, y, MONTHS) / len(MONTHS)
        for c in CATEGORY_CODES:
            out[f"{c}_{y}"] = sums[f"v_{c}"]
    return pd.DataFrame(out)


def half_year_growth(panel_wide: pd.DataFrame) -> pd.DataFrame:
    """Рост по полугодиям (номинал): ``growth_h1_<кат.>`` = Σ января–июня 2024 / Σ января–июня 2023 − 1,
    ``growth_h2_<кат.>`` — то же за июль–декабрь; ``mp_pp_h1``, ``mp_pp_h2`` — прирост доли маркетплейсов,
    п. п.

    Индекс — ``territory_id``; NaN, если в каком-то из четырёх полугодий нет всех шести месяцев.
    """
    cols = [f"v_{c}" for c in CATEGORY_CODES]
    out = {}
    for tag, months in (("h1", FIRST_HALF), ("h2", SECOND_HALF)):
        a, b = month_sums(panel_wide, cols, Y0, months), month_sums(panel_wide, cols, Y1, months)
        a, b = a.align(b, join="outer")
        for c in CATEGORY_CODES:
            out[f"growth_{tag}_{c}"] = b[f"v_{c}"] / a[f"v_{c}"] - 1.0
        out[f"mp_pp_{tag}"] = PP * (b[f"v_{MP}"] / b["v_all"] - a[f"v_{MP}"] / a["v_all"])
    return pd.DataFrame(out)


def mp_series(national: pd.DataFrame, column: str = "median_share") -> pd.Series:
    """Колонка ``national`` для маркетплейсов по месяцам (индекс — ``date``)."""
    s = national.loc[national["category"].astype(str) == MP].set_index("date")[column]
    return s.astype("float64").sort_index()


def yoy_pp(national: pd.DataFrame) -> pd.Series:
    """Годовой прирост медианной доли маркетплейсов по календарным месяцам, п. п.:
    доля(2024, m) − доля(2023, m)."""
    s = mp_series(national)
    vals = [PP * (s.get(f"{Y1}-{m:02d}", np.nan) - s.get(f"{Y0}-{m:02d}", np.nan)) for m in MONTHS]
    return pd.Series(vals, index=pd.Index(MONTHS, name="month"), dtype="float64")


def step_test(national: pd.DataFrame) -> tuple[float, str]:
    """Тест ступеньки: максимум |Δ_{m+1} − Δ_m| годового прироста медианной доли маркетплейсов, п. п., и пара
    месяцев, где он случился («октябрь → ноябрь»). Резкий скачок — повод проверить границы категорий
    (переклассификацию продавцов). Не видны: плавный сдвиг и ступенька ровно в январе 2024 года (она сдвигает
    Δ всех месяцев одинаково)."""
    jumps = yoy_pp(national).diff().abs().iloc[1:]
    if jumps.isna().all():
        return float("nan"), style.NA_TEXT
    month = int(jumps.idxmax())
    return float(jumps.max()), f"{MONTHS_FULL_RU[month - 2]} → {MONTHS_FULL_RU[month - 1]}"


def quintile_gap(mo: pd.DataFrame, groups: int = 5) -> pd.DataFrame:
    """Доля маркетплейсов 2023 → 2024 по группам МО равной численности по ``log_level_2023``
    (группа 1 — тратят меньше всех).

    Только МО с долями обоих лет (12 месяцев в каждом году). Колонки: ``group``, ``n``, ``level_min``,
    ``level_max`` (₽, уровень 2023 года), ``share_2023``, ``share_2024`` (медианы долей), ``dpp`` — разность
    медиан, п. п. (её и показывают «гантели»).
    """
    cols = ["log_level_2023", f"level_{Y0}", f"sh_{MP}_{Y0}", f"sh_{MP}_{Y1}"]
    frame = mo.dropna(subset=cols)
    group = pd.qcut(frame["log_level_2023"], groups, labels=False) + 1
    g = frame.groupby(group)
    out = pd.DataFrame(
        {
            "n": g.size(),
            "level_min": g[f"level_{Y0}"].min(),
            "level_max": g[f"level_{Y0}"].max(),
            "share_2023": g[f"sh_{MP}_{Y0}"].median(),
            "share_2024": g[f"sh_{MP}_{Y1}"].median(),
        }
    )
    out["dpp"] = PP * (out["share_2024"] - out["share_2023"])
    out.index.name = "group"
    return out.reset_index()


def marketplace_frame(panel_wide: pd.DataFrame, mo: pd.DataFrame) -> pd.DataFrame:
    """МО с долями маркетплейсов обоих лет: прирост, уровень, регион, траты на маркетплейсах в рублях,
    начальные доли и приросты на раздельных месяцах; индекс — ``territory_id``."""
    cols = ["mp_pp_change", "log_level_2023", f"sh_{MP}_{Y0}", f"sh_{MP}_{Y1}"]
    frame = mo.dropna(subset=cols).set_index("territory_id")
    frame = frame[[*cols, "region_code"]].copy()
    ids = frame.index
    levels = annual_levels(panel_wide).reindex(ids)
    frame["rub_2023"], frame["rub_2024"] = levels[f"{MP}_{Y0}"], levels[f"{MP}_{Y1}"]
    frame["rub_growth"] = frame["rub_2024"] / frame["rub_2023"] - 1.0
    frame["rub_incr"] = frame["rub_2024"] - frame["rub_2023"]
    for tag, init_m, gain_m in (("odd", ODD_MONTHS, EVEN_MONTHS), ("even", EVEN_MONTHS, ODD_MONTHS)):
        frame[f"init_{tag}"] = month_share(panel_wide, MP, Y0, init_m).reindex(ids)
        gain = month_share(panel_wide, MP, Y1, gain_m) - month_share(panel_wide, MP, Y0, gain_m)
        frame[f"gain_{tag}"] = PP * gain.reindex(ids)
    return frame


def _iqr(x: pd.Series) -> float:
    return float(x.quantile(0.75) - x.quantile(0.25))


def marketplace_checks(
    panel_wide: pd.DataFrame, mo: pd.DataFrame, national: pd.DataFrame, halves: pd.DataFrame | None = None
) -> dict[str, float | int | str]:
    """Проверки сюжета С5 (T09): связь прироста доли с уровнем в целом и внутри регионов, с начальной долей на
    раздельных месяцах, согласованность полугодий, разброс долей в двух шкалах, рубли, η² региона, ступенька.

    Прирост — ``mo.mp_pp_change`` (п. п., годовые доли); уровень — ``log_level_2023``. Начальная доля по
    нечётным месяцам 2023 года, прирост — по чётным (``rho_mp_pp_initial``); наоборот — ``…_rev``; без
    разделения (смещено регрессией к среднему) — ``…_naive``.
    """
    f = marketplace_frame(panel_wide, mo)
    halves = half_year_growth(panel_wide) if halves is None else halves
    h = halves.reindex(f.index)
    rho = lambda a, b: stats.spearman(a, b)[0]  # noqa: E731 — короткая запись ниже
    mp = mp_series(national)
    res = mp_series(national, "resident_share")
    step_max, step_month = step_test(national)
    return {
        "n": int(len(f)),
        "mp_pp_median": float(f["mp_pp_change"].median()),
        "mp_pp_q25": float(f["mp_pp_change"].quantile(0.25)),
        "mp_pp_q75": float(f["mp_pp_change"].quantile(0.75)),
        "rho_mp_pp_level": rho(f["mp_pp_change"], f["log_level_2023"]),
        "rho_mp_pp_level_within": stats.spearman_within(
            f["mp_pp_change"], f["log_level_2023"], f["region_code"]
        )[0],
        "eta2_mp_pp": stats.eta2(f["mp_pp_change"], f["region_code"]),
        "rho_mp_pp_initial": rho(f["gain_odd"], f["init_odd"]),
        "rho_mp_pp_initial_rev": rho(f["gain_even"], f["init_even"]),
        "rho_mp_pp_initial_naive": rho(f["mp_pp_change"], f[f"sh_{MP}_{Y0}"]),
        "mp_split_half": rho(h["mp_pp_h1"], h["mp_pp_h2"]),
        "mp_iqr_2023": PP * _iqr(f[f"sh_{MP}_{Y0}"]),
        "mp_iqr_2024": PP * _iqr(f[f"sh_{MP}_{Y1}"]),
        "mp_sd_log_2023": float(np.log(f[f"sh_{MP}_{Y0}"]).std(ddof=1)),
        "mp_sd_log_2024": float(np.log(f[f"sh_{MP}_{Y1}"]).std(ddof=1)),
        "mp_share_2023": float(f[f"sh_{MP}_{Y0}"].median()),
        "mp_share_2024": float(f[f"sh_{MP}_{Y1}"].median()),
        "mp_rub_growth_median": float(f["rub_growth"].median()),
        "rho_mp_rub_growth_level": rho(f["rub_growth"], f["log_level_2023"]),
        "rho_mp_rub_incr_level": rho(f["rub_incr"], f["log_level_2023"]),
        "mp_share_jan23": float(mp.get(f"{Y0}-01", np.nan)),
        "mp_share_dec23": float(mp.get(f"{Y0}-12", np.nan)),
        "mp_share_dec24": float(mp.get(f"{Y1}-12", np.nan)),
        "mp_resident_jan23": float(res.get(f"{Y0}-01", np.nan)),
        "mp_resident_dec24": float(res.get(f"{Y1}-12", np.nan)),
        "mp_step_max_pp": step_max,
        "mp_step_month": step_month,
    }


# --- Рост и расхождение уровней ------------------------------------------------------------------


def growth_table(
    panel_wide: pd.DataFrame, mo: pd.DataFrame, halves: pd.DataFrame | None = None
) -> pd.DataFrame:
    """T10: номинальный рост 2024/2023 по категориям у МО с полным рядом.

    Колонки: ``code``, ``label``, ``growth_median``, ``growth_p10``, ``growth_p90`` (рост среднемесячных трат
    категории), ``half_consistency`` (ρ роста января–июня и июля–декабря), ``sd_log_2023``, ``sd_log_2024``
    (SD ln среднемесячных трат категории — разброс уровней в лог-шкале), ``n``.
    """
    ids = mo.loc[mo["series_status"].astype(str) == "full", "territory_id"]
    levels = annual_levels(panel_wide).reindex(ids)
    halves = (half_year_growth(panel_wide) if halves is None else halves).reindex(ids)
    rows = []
    for c in CATEGORY_CODES:
        a, b = levels[f"{c}_{Y0}"], levels[f"{c}_{Y1}"]
        growth = (b / a - 1.0).dropna()
        rows.append(
            {
                "code": c,
                "label": style.LABELS[c],
                "growth_median": float(growth.median()),
                "growth_p10": float(growth.quantile(0.1)),
                "growth_p90": float(growth.quantile(0.9)),
                "half_consistency": stats.spearman(halves[f"growth_h1_{c}"], halves[f"growth_h2_{c}"])[0],
                "sd_log_2023": float(np.log(a).std(ddof=1)),
                "sd_log_2024": float(np.log(b).std(ddof=1)),
                "n": int(len(growth)),
            }
        )
    return pd.DataFrame(rows)


def growth_vs_level(panel_wide: pd.DataFrame, mo: pd.DataFrame) -> dict[str, float | int]:
    """ρ номинального роста (``mo.growth``) с ``log_level_2023``: в целом, внутри регионов и на раздельных
    месяцах (уровень — ln средних трат нечётных месяцев 2023 года, рост — по чётным месяцам), чтобы общий
    шум уровня 2023 года в обеих величинах не создавал ложной связи (регрессия к среднему)."""
    f = mo.dropna(subset=["growth", "log_level_2023"]).set_index("territory_id")
    odd = month_sums(panel_wide, ["v_all"], Y0, ODD_MONTHS)["v_all"] / len(ODD_MONTHS)
    even0 = month_sums(panel_wide, ["v_all"], Y0, EVEN_MONTHS)["v_all"]
    even1 = month_sums(panel_wide, ["v_all"], Y1, EVEN_MONTHS)["v_all"]
    split_level = np.log(odd).reindex(f.index)
    split_growth = (even1 / even0 - 1.0).reindex(f.index)
    rho, n = stats.spearman(f["growth"], f["log_level_2023"])
    return {
        "rho_growth_level": rho,
        "rho_growth_level_within": stats.spearman_within(f["growth"], f["log_level_2023"], f["region_code"])[
            0
        ],
        "rho_growth_level_split": stats.spearman(split_growth, split_level)[0],
        "n": int(n),
    }


# --- ИПЦ (только при eda.cpi) --------------------------------------------------------------------


def cpi_index(cpi: Mapping[str, Any]) -> pd.Series:
    """Уровень цен по месяцам ``ГГГГ-ММ`` из ``eda.cpi``.

    Формат: ``{source: ссылка, base: "2022-12", values: {"2023-01": …}, kind: level | mom}``. ``kind: level``
    (по умолчанию) — ``values`` уже уровень цен к базовому месяцу; ``kind: mom`` — индекс к предыдущему месяцу
    в процентах (как публикует Росстат), уровень считается цепочкой от базы = 100. Нужны все месяцы 2023–2024
    годов и ссылка на источник, иначе ``ValueError``.
    """
    if not str(cpi.get("source") or "").strip():
        raise ValueError("eda.cpi: нужна ссылка на источник (source)")
    kind = cpi.get("kind", "level")
    values = pd.Series({str(k): float(v) for k, v in dict(cpi.get("values") or {}).items()}).sort_index()
    need = [f"{y}-{m:02d}" for y in YEARS for m in MONTHS]
    missing = [d for d in need if d not in values.index]
    if missing:
        raise ValueError(f"eda.cpi: нет месяцев {missing}")
    if kind == "level":
        return values
    if kind == "mom":
        return (values / PP).cumprod() * PP
    raise ValueError(f"eda.cpi.kind: {kind!r}; допустимы level и mom")


def cpi_ratios(index: pd.Series) -> tuple[float, pd.Series]:
    """Годовой рост цен (среднее 2024 / среднее 2023) и помесячный (месяц 2024 / тот же месяц 2023)."""
    by_year = {y: index[[f"{y}-{m:02d}" for m in MONTHS]].to_numpy() for y in YEARS}
    annual = float(by_year[Y1].mean() / by_year[Y0].mean())
    monthly = pd.Series(by_year[Y1] / by_year[Y0], index=pd.Index(MONTHS, name="month"))
    return annual, monthly


def yoy_by_month(panel_wide: pd.DataFrame, ids: Sequence[int]) -> pd.Series:
    """Медиана по МО номинального годового роста трат по месяцам: v_all(2024, m) / v_all(2023, m) − 1."""
    w = panel_wide.loc[panel_wide["territory_id"].isin(list(ids))]
    piv = w.pivot_table(index="territory_id", columns=["year", "month"], values="v_all", aggfunc="first")
    out = [
        float((piv[(Y1, m)] / piv[(Y0, m)] - 1.0).median()) if (Y0, m) in piv and (Y1, m) in piv else np.nan
        for m in MONTHS
    ]
    return pd.Series(out, index=pd.Index(MONTHS, name="month"))


def real_growth(nominal: float | pd.Series, price_ratio: float | pd.Series) -> float | pd.Series:
    """Реальный рост: (1 + номинальный) / (рост цен) − 1."""
    return (1.0 + nominal) / price_ratio - 1.0


# --- Сверка с ориентирами спецификации -------------------------------------------------------------
# Часть ориентиров Б.4 посчитана не теми определениями, что закреплены в Б.1, Б.2 и В.5: корзина — по долям
# двух лет вместе, «внутри регионов» — на значениях, центрированных по региону, «доля в сумме трат» — без
# весов населения. Раздел считает по закреплённым определениям, а значения по определениям ориентиров
# пишет в ``note`` своих фактов: расхождение с ориентиром объяснено числом, а не словами.

REF_DEFINITIONS: dict[str, str] = {
    **dict.fromkeys(
        [
            *(f"share_median_{p}" for p in PARTS),
            *(f"clr_var_{p}" for p in PARTS),
            "pc1_var",
            "r2_clr_level",
            "r2_clr_region",
        ],
        f"по долям за {Y0} и {Y1} годы вместе",
    ),
    "rho_pc1_level": f"по долям и уровню трат за {Y0} и {Y1} годы вместе (знак оси условен)",
    "rho_mp_pp_level_within": "на значениях, центрированных по среднему региона, а не на рангах",
    "rho_growth_level_within": "на значениях (ln роста и лог-уровень), центрированных по среднему региона",
    "rho_mp_rub_incr_level": f"с лог-уровнем трат за {Y0} и {Y1} годы вместе",
    "mp_resident_jan23": "как доля в сумме трат МО без весов населения",
    "mp_resident_dec24": "как доля в сумме трат МО без весов населения",
}


def two_year_composition(panel_wide: pd.DataFrame, ids: Sequence[int]) -> tuple[pd.DataFrame, pd.Series]:
    """Доли шести частей за оба года вместе (Σ 24 месяцев ``v_<часть>`` / Σ ``v_all``) и ln среднемесячных
    трат за два года; только МО из ``ids`` с 12 месяцами в каждом году. Индекс — ``territory_id``."""
    cols = [*(f"v_{p}" for p in PARTS), "v_all"]
    sums = sum(month_sums(panel_wide, cols, y, MONTHS) for y in YEARS)  # NaN, если год неполный
    sums = sums.reindex(list(ids)).dropna()
    shares = pd.DataFrame({p: sums[f"v_{p}"] / sums["v_all"] for p in PARTS})
    return shares, np.log(sums["v_all"] / (len(YEARS) * len(MONTHS)))


def spearman_demeaned(x: pd.Series, y: pd.Series, groups: pd.Series) -> float:
    """ρ Спирмена значений, центрированных по среднему своей группы (региона); ряды выравниваются по индексу.

    Второй способ «внутри регионов». Основной способ раздела — ранги, центрированные по региону
    (``stats.spearman_within``, Б.1, п. 9): он не зависит от шкалы величины и выбросов внутри региона.
    """
    frame = pd.concat({"x": x, "y": y, "g": groups}, axis=1).dropna()
    centered = frame[["x", "y"]] - frame.groupby("g", observed=True)[["x", "y"]].transform("mean")
    return stats.spearman(centered["x"], centered["y"])[0]


def pooled_share(panel_wide: pd.DataFrame, part: str, date: str) -> float:
    """Доля части в сумме трат всех МО месяца без весов населения: Σ ``v_<часть>`` / Σ ``v_all``.

    Это не доля в тратах жителей: ``v`` — траты на жителя, и сумма без весов даёт маленькому МО тот же вес,
    что и большому. Доля в тратах жителей — ``national.resident_share`` (веса — население, Б.2).
    """
    month = panel_wide.loc[panel_wide["date"].astype(str) == date]
    total = float(month["v_all"].sum())
    return float(month[f"v_{part}"].sum()) / total if total > 0 else float("nan")


def reference_variants(panel_wide: pd.DataFrame, mo: pd.DataFrame) -> dict[str, float]:
    """Ключевые факты раздела по определениям, которыми посчитаны ориентиры Б.4 (ключи — ``REF_DEFINITIONS``).

    Корзина — доли за два года вместе на выборке ``basket_sample``, уровень — ln среднемесячных трат за два
    года; «внутри регионов» — ``spearman_demeaned``; прибавка трат на маркетплейсах в рублях — против
    лог-уровня за два года; доля маркетплейсов в январе 2023 и декабре 2024 — ``pooled_share``.
    """
    sample = basket_sample(mo)
    shares, level2 = two_year_composition(panel_wide, sample.index)
    clr = stats.clr(shares)
    raw = pca_clr(clr, n_components=1)
    region = pd.DataFrame({"region": sample.loc[clr.index, "region_code"].astype("category")})
    _, r2_level, _ = residualize(clr, level2.to_frame("log_level"))
    _, r2_region, _ = residualize(clr, region)
    out: dict[str, float] = {f"share_median_{p}": float(shares[p].median()) for p in PARTS}
    out.update({f"clr_var_{p}": float(clr[p].var(ddof=1)) for p in PARTS})
    out["pc1_var"] = float(raw.explained[PC1])
    out["rho_pc1_level"] = stats.spearman(raw.scores[PC1], level2.loc[raw.scores.index])[0]
    out["r2_clr_level"], out["r2_clr_region"] = float(r2_level), float(r2_region)
    mp = marketplace_frame(panel_wide, mo)
    out["rho_mp_pp_level_within"] = spearman_demeaned(
        mp["mp_pp_change"], mp["log_level_2023"], mp["region_code"]
    )
    mp_level2 = two_year_composition(panel_wide, mp.index)[1]
    out["rho_mp_rub_incr_level"] = stats.spearman(mp["rub_incr"], mp_level2.reindex(mp.index))[0]
    full = mo.loc[mo["series_status"].astype(str) == "full"].set_index("territory_id")
    out["rho_growth_level_within"] = spearman_demeaned(
        full["growth_log"], full["log_level_2023"], full["region_code"]
    )
    out["mp_resident_jan23"] = pooled_share(panel_wide, MP, f"{Y0}-01")
    out["mp_resident_dec24"] = pooled_share(panel_wide, MP, f"{Y1}-12")
    return out


# --- Рисунки -------------------------------------------------------------------------------------


def region_short(region_name: str) -> str:
    """Короткое название региона для подписи точки: «Вологодская обл.», «Тыва», «Ямало-Ненецкий АО»."""
    text = str(region_name)
    for pattern, repl in _REGION_RULES:
        text = re.sub(pattern, repl, text)
    return text.strip()


def point_names(territories: pd.DataFrame) -> pd.Series:
    """Подписи МО на точечных графиках: «Белозерский (Вологодская обл.)» — короткие названия МО повторяются
    в разных регионах (десять «Октябрьских»), поэтому к названию добавлен регион."""
    ter = territories.set_index("territory_id")
    names = (ter["name_short"] if "name_short" in ter else ter["name"]).astype(str)
    if "region_name" not in ter:
        return names
    return names + " (" + ter["region_name"].map(region_short) + ")"


def _rub_ticks(values: pd.Series) -> list[int]:
    lo, hi = float(np.nanmin(values)), float(np.nanmax(values))
    ticks = [t for t in RUB_TICKS if lo <= t <= hi]
    return ticks or [int(round(np.nanmedian(values), -3))]


def _binned_median(x: pd.Series, y: pd.Series, bins: int) -> pd.DataFrame:
    """Медианы x и y по группам равной численности по x — робастная линия тренда."""
    frame = pd.DataFrame({"x": x, "y": y}).dropna()
    k = min(bins, max(1, len(frame) // 2))
    frame["bin"] = pd.qcut(frame["x"].rank(method="first"), k, labels=False)
    return frame.groupby("bin")[["x", "y"]].median()


def spread_extremes(x: pd.Series, y: pd.Series, score: pd.Series, n: int, box: Sequence[float]) -> list:
    """Точки для подписей: по ``n // 2`` с наибольшим и наименьшим ``score``, чтобы подписи не закрывали
    соседние подписанные точки.

    Координаты нормируются на размах x и y (0…1); точка-кандидат отбрасывается, если уже выбранная точка
    лежит в прямоугольнике ``box`` = (ширина, высота) вокруг неё — примерно размер подписи. Индексы — как
    у входных рядов; кандидаты с пропуском пропускаются.
    """
    frame = pd.DataFrame({"x": x, "y": y, "s": score}).dropna()
    if frame.empty:
        return []
    for col in ("x", "y"):
        span = frame[col].max() - frame[col].min()
        frame[col] = (frame[col] - frame[col].min()) / span if span > 0 else 0.0
    width, height = (float(v) for v in box)
    half = max(1, n // 2)
    chosen: list = []
    for ascending in (True, False):
        taken = 0
        for idx in frame.sort_values("s", ascending=ascending).index:
            if taken == half:
                break
            px, py = frame.at[idx, "x"], frame.at[idx, "y"]
            clash = any(
                abs(px - frame.at[c, "x"]) < width and abs(py - frame.at[c, "y"]) < height for c in chosen
            )
            if idx not in chosen and not clash:
                chosen.append(idx)
                taken += 1
    return chosen


# Смещения подписи от точки, pt: сначала рядом, затем дальше — с выноской (тонкая линия к точке).
_NEAR_OFFSETS: tuple[tuple[int, int], ...] = (
    (6, 6),
    (6, -6),
    (-6, 6),
    (-6, -6),
    (0, 10),
    (0, -12),
    (10, 0),
    (-10, 0),
)
_FAR_OFFSETS: tuple[tuple[int, int], ...] = (
    (18, 18),
    (18, -18),
    (-18, 18),
    (-18, -18),
    (0, 26),
    (0, -28),
    (32, 6),
    (-32, 6),
    (32, -6),
    (-32, -6),
)
_POINT_HALO_PT = 4.0  # выделенная точка — препятствие для чужих подписей: квадрат ±4 pt вокруг неё
# Подпись должна помещаться в поле графика с запасом (доля ширины и высоты поля): после ``style.finish``
# заголовок и подзаголовок уменьшают поле, и точки сдвигаются относительно подписей постоянного размера.
_INSIDE_MARGIN = 0.04
_PT_PER_INCH = 72.0


def _renderer(fig: Any) -> Any:
    """Рендерер фигуры для измерения подписей (Agg, если у холста его нет)."""
    if not hasattr(fig.canvas, "get_renderer"):
        FigureCanvasAgg(fig)
    return fig.canvas.get_renderer()


def label_points_clear(
    ax: Any,
    x: Sequence[float],
    y: Sequence[float],
    labels: Sequence[str],
    *,
    avoid: Sequence[Any] = (),
    max_labels: int = 7,
) -> list:
    """Выделяет точки и подписывает их так, чтобы подпись не закрывала другие подписи, другие выделенные
    точки и тексты ``avoid`` (подписи линий, пояснения в углу поля) и не выходила за поле графика.

    Отличие от ``style.label_points``: препятствия — не только уже поставленные подписи. Дальние положения
    подписи соединяются с точкой тонкой выноской. Если места нет, подпись встаёт в первое положение.
    Больше ``max_labels`` подписей — ``ValueError`` (правило В.2: не больше 5–7 подписей МО). Поле графика
    к этому моменту должно стоять на окончательном месте: ``_save`` вызывает ``reserve_layout`` до подписей.
    """
    xs, ys, texts = list(x), list(y), [str(t) for t in labels]
    if not len(xs) == len(ys) == len(texts):
        raise ValueError("x, y и labels должны быть одной длины")
    if len(texts) > max_labels:
        raise ValueError(f"подписей {len(texts)}, а допустимо не больше {max_labels}")
    ax.scatter(xs, ys, s=18, color=style.ACCENT, zorder=3, linewidths=0)
    fig = ax.figure
    renderer = _renderer(fig)
    fig.draw_without_rendering()  # раскладка constrained: поле графика — там, где оно будет на рисунке
    full = ax.get_window_extent(renderer=renderer)
    mx, my = _INSIDE_MARGIN * full.width, _INSIDE_MARGIN * full.height
    ax_box = Bbox.from_extents(full.x0 + mx, full.y0 + my, full.x1 - mx, full.y1 - my)
    halo = _POINT_HALO_PT * fig.dpi / _PT_PER_INCH
    centers = ax.transData.transform(np.column_stack([xs, ys])) if xs else np.empty((0, 2))
    points = [Bbox.from_extents(cx - halo, cy - halo, cx + halo, cy + halo) for cx, cy in centers]
    taken = [a.get_window_extent(renderer=renderer) for a in avoid]
    out = []
    for i, (xi, yi, text) in enumerate(zip(xs, ys, texts, strict=True)):
        obstacles = [*taken, *(b for j, b in enumerate(points) if j != i)]
        placed = None
        for dx, dy in (*_NEAR_OFFSETS, *_FAR_OFFSETS):
            ann = _point_label(ax, xi, yi, text, dx, dy, leader=(dx, dy) in _FAR_OFFSETS)
            box = _text_box(ann, renderer)
            inside = ax_box.contains(box.x0, box.y0) and ax_box.contains(box.x1, box.y1)
            if inside and not any(box.overlaps(b) for b in obstacles):
                placed = (ann, box)
                break
            ann.remove()
        if placed is None:
            dx, dy = _NEAR_OFFSETS[0]
            ann = _point_label(ax, xi, yi, text, dx, dy, leader=False)
            placed = (ann, _text_box(ann, renderer))
        out.append(placed[0])
        taken.append(placed[1])
    return out


def _text_box(ann: Any, renderer: Any) -> Bbox:
    """Рамка текста подписи без выноски: положение подписи пересчитывается до измерения."""
    ann.update_positions(renderer)
    return Text.get_window_extent(ann, renderer=renderer)


def _point_label(ax: Any, x: float, y: float, text: str, dx: int, dy: int, *, leader: bool) -> Any:
    """Подпись точки со смещением (dx, dy) pt; выноска — тонкая серая линия к точке."""
    arrow = {"arrowstyle": "-", "color": style.TEXT2, "linewidth": 0.5, "shrinkA": 0, "shrinkB": 3}
    return ax.annotate(
        text,
        (x, y),
        xytext=(dx, dy),
        textcoords="offset points",
        ha="left" if dx > 0 else "right" if dx < 0 else "center",
        va="bottom" if dy > 0 else "top" if dy < 0 else "center",
        fontsize=style.POINT_LABEL_PT,
        color=style.TEXT,
        zorder=4,
        path_effects=[patheffects.withStroke(linewidth=2.5, foreground="white")],
        arrowprops=arrow if leader else None,
        in_layout=False,  # подписи не сжимают поле графика при раскладке constrained
    )


def _trend_label(ax: Any, trend: pd.DataFrame, bins: int) -> Any:
    """Подпись медианной линии у её правого конца; возвращает текст — препятствие для подписей точек."""
    last = trend.iloc[-1]
    return ax.annotate(
        f"медиана по {GROUP_WORDS[bins][2]}",
        (np.exp(last["x"]), last["y"]),
        xytext=(-4, 10),
        textcoords="offset points",
        ha="right",
        fontsize=style.POINT_LABEL_PT,
        color=style.ACCENT,
        zorder=4,
        path_effects=[patheffects.withStroke(linewidth=2.5, foreground="white")],
    )


def _fig_basket(res: BasketResult):
    """F09: медиана и межквартильный размах доли каждой части (слева), дисперсия CLR (справа,
    столбцы от нуля)."""
    summary = res.summary
    fig, (ax, ax_var) = style.new_figure("full", 1, 2, sharey=True, gridspec_kw={"width_ratios": [3, 1]})
    y = np.arange(len(summary))[::-1]
    for yi, (part, row) in zip(y, summary.iterrows(), strict=True):
        color = style.PALETTE[part]
        ax.hlines(yi, row["q25"], row["q75"], color=color, linewidth=6, alpha=0.35)
        ax.scatter([row["median"]], [yi], s=46, color=color, zorder=3)
        ax.annotate(
            style.fmt_pct(row["median"]),
            (row["q75"], yi),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=style.POINT_LABEL_PT,
            color=style.TEXT,
        )
    ax.set_yticks(y, list(summary["label"]))
    ax.set_xlim(0, float(summary["q75"].max()) * 1.15)
    ax.xaxis.set_major_formatter(style.ru_formatter("pct"))
    ax.set_xlabel("Доля в тратах 2024 г.: точка — медиана по МО, полоса — половина МО")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    top = summary["clr_var"].idxmax()
    ax_var.barh(y, summary["clr_var"], color=[style.PALETTE[p] for p in summary.index], height=0.6)
    for yi, (part, row) in zip(y, summary.iterrows(), strict=True):
        ax_var.annotate(
            style.fmt_num(row["clr_var"], CLR_VAR_DECIMALS),
            (row["clr_var"], yi),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize=style.POINT_LABEL_PT,
            fontweight="bold" if part == top else "normal",
            color=style.TEXT,
        )
    ax_var.set_xlim(0, float(summary["clr_var"].max()) * 1.4)
    ax_var.set_xlabel("Дисперсия CLR")
    ax_var.tick_params(axis="y", length=0)
    ax_var.grid(axis="y", visible=False)
    ax_var.set_xticks([])
    ax_var.spines["bottom"].set_visible(False)
    return fig


def _fig_basket_vs_level(
    level: pd.Series, pc1: pd.Series, names: pd.Series, p: Mapping[str, Any], top_label: str
) -> tuple[Any, Callable[[], list]]:
    """F10: PC1 сырых CLR против уровня трат 2024 года (лог-ось в рублях), медианная линия и подписи.

    Возвращает фигуру и отложенный шаг подписей точек (его выполняет ``_save`` после раскладки шапки).

    ``top_label`` — часть с наибольшей нагрузкой PC1: она положительна по соглашению о знаке, поэтому вверху
    оси — МО, где доля этой части выше; это написано в углу поля графика."""
    fig, ax = style.new_figure("full")
    ax.scatter(level, pc1, s=7, color=style.CONTEXT, alpha=0.8, linewidths=0, zorder=2)
    trend = _binned_median(np.log(level), pc1, int(p["trend_bins"]))
    ax.plot(np.exp(trend["x"]), trend["y"], color=style.ACCENT, linewidth=1.6, zorder=3)
    style.log_rub_axis(ax, "x", ticks=_rub_ticks(level))
    ax.set_xlabel("Траты на жителя в месяц, 2024 г. (шкала логарифмическая)")
    ax.set_ylabel("PC1 сырых CLR")
    hint = ax.text(
        0.01,
        0.98,
        f"выше — больше доля «{top_label}»",
        transform=ax.transAxes,
        va="top",
        fontsize=style.POINT_LABEL_PT,
        color=style.TEXT2,
    )
    x = np.log(level)
    slope, intercept = np.polyfit(x.to_numpy(), pc1.to_numpy(), 1)
    resid = pc1 - (intercept + slope * x)
    n = int(p["label_n"])
    idx = spread_extremes(x, pc1, resid, n, p["label_box"])
    avoid = [hint, _trend_label(ax, trend, int(p["trend_bins"]))]
    texts = names.reindex(idx).fillna("").tolist()
    return fig, lambda: label_points_clear(ax, level.loc[idx], pc1.loc[idx], texts, avoid=avoid, max_labels=n)


def _fig_mp_trend(trend: pd.DataFrame):
    """F11: медиана долей маркетплейсов по МО месяца с полосой межквартильного размаха и доля
    в тратах жителей."""
    fig, ax = style.new_figure("full")
    color = style.PALETTE[MP]
    t = trend["t"].to_numpy()
    ax.fill_between(t, trend["q25"], trend["q75"], color=color, alpha=0.15, linewidth=0)
    ax.plot(t, trend["median_share"], color=color, linewidth=2.2)
    ax.plot(t, trend["resident_share"], color=style.ACCENT, linewidth=1.4, linestyle="--")
    style.month_axis(ax, len(t), start=int(t.min()), first_year=Y0)
    ax.yaxis.set_major_formatter(style.ru_formatter("pct"))
    ax.set_ylim(0, float(trend["q75"].max()) * 1.12)
    ax.set_xlim(t.min() - 0.5, t.max() + 6)
    end = trend.iloc[-1]
    for value, text, col, dy in (
        (end["median_share"], f"типичное МО {style.fmt_pct(end['median_share'])}", color, 6),
        (end["resident_share"], f"все жители {style.fmt_pct(end['resident_share'])}", style.ACCENT, -10),
    ):
        ax.annotate(
            text,
            (t.max(), value),
            xytext=(6, dy),
            textcoords="offset points",
            va="center",
            fontsize=style.POINT_LABEL_PT,
            color=col,
        )
    ax.annotate(
        "половина МО",
        (t.max(), end["q75"]),
        xytext=(6, 4),
        textcoords="offset points",
        va="bottom",
        fontsize=style.POINT_LABEL_PT,
        color=style.TEXT2,
    )
    return fig


def _level_range_labels(gap: pd.DataFrame) -> list[str]:
    """Подписи групп по уровню трат 2023 года: «до 18,5 тыс. ₽», «18,5–21,0 тыс. ₽», «от 29,0 тыс. ₽»."""
    edges = [(gap["level_max"].iloc[i] + gap["level_min"].iloc[i + 1]) / 2 for i in range(len(gap) - 1)]
    labels = []
    for i in range(len(gap)):
        if i == 0:
            labels.append(f"до {style.fmt_krub(edges[0])}")
        elif i == len(gap) - 1:
            labels.append(f"от {style.fmt_krub(edges[-1])}")
        else:
            labels.append(style.fmt_range(edges[i - 1], edges[i], fmt=style.fmt_krub))
    return [nbsp(x) for x in labels]


def start_phrase(gap: pd.DataFrame, texts: tuple[str, str]) -> str:
    """Подзаголовок F12 о стартовых долях крайних групп: «старт близок (11,7% и 10,5%)», если стартовые доли
    нижней и верхней пятых частей различаются меньше, чем их приросты (в п. п.), иначе «старт: 11,7% и 10,5%».

    Так слово «близок» защищено данными: разрыв в приросте не объясняется разным стартом. ``texts`` — готовые
    строки долей (тексты фактов ``q1_share_2023`` и ``q5_share_2023``).
    """
    first, last = gap.iloc[0], gap.iloc[-1]
    start_gap = abs(PP * (first[f"share_{Y0}"] - last[f"share_{Y0}"]))
    growth_gap = abs(first["dpp"] - last["dpp"])
    word = "старт близок" if start_gap < growth_gap else "старт"
    return f"{word} ({texts[0]} и {texts[1]})"


def _fig_mp_gap(gap: pd.DataFrame):
    """F12: «гантели» доли маркетплейсов 2023 → 2024 по группам уровня трат, подпись прироста в п. п."""
    fig, ax = style.new_figure("full")
    color = style.PALETTE[MP]
    y = np.arange(len(gap))[::-1]  # первая группа (тратят меньше) — сверху
    for yi, row in zip(y, gap.itertuples(index=False), strict=True):
        ax.plot([row.share_2023, row.share_2024], [yi, yi], color=style.CONTEXT, linewidth=2.5, zorder=1)
        ax.scatter([row.share_2023], [yi], s=60, facecolor="white", edgecolor=color, linewidth=1.6, zorder=3)
        ax.scatter([row.share_2024], [yi], s=60, color=color, zorder=3)
        ax.annotate(
            style.fmt_pct(row.share_2023),
            (row.share_2023, yi),
            xytext=(-8, 0),
            textcoords="offset points",
            ha="right",
            va="center",
            fontsize=style.POINT_LABEL_PT,
            color=style.TEXT2,
        )
        ax.annotate(
            f"{style.fmt_pct(row.share_2024)} ({style.fmt_pp(row.dpp)})",
            (row.share_2024, yi),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            fontsize=style.POINT_LABEL_PT,
            color=style.TEXT,
            fontweight="bold" if yi in (y.max(), y.min()) else "normal",
        )
    first = gap.iloc[0]
    for x, text in ((first["share_2023"], str(Y0)), (first["share_2024"], str(Y1))):
        ax.annotate(
            text,
            (x, y.max()),
            xytext=(0, 12),
            textcoords="offset points",
            ha="center",
            fontsize=style.POINT_LABEL_PT,
            color=style.TEXT2,
        )
    ax.set_yticks(y, _level_range_labels(gap))
    lo = float(gap[["share_2023", "share_2024"]].min().min())
    hi = float(gap[["share_2023", "share_2024"]].max().max())
    pad = (hi - lo) * 0.35
    ax.set_xlim(max(0.0, lo - pad), hi + pad * 1.4)
    ax.set_ylim(-0.6, len(gap) - 0.2)
    ax.xaxis.set_major_formatter(style.ru_formatter("pct"))
    ax.set_xlabel("Медиана доли маркетплейсов в тратах МО")
    ax.set_ylabel("Траты на жителя, 2023 г.")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    return fig


def _fig_growth(
    frame: pd.DataFrame, names: pd.Series, p: Mapping[str, Any]
) -> tuple[Any, Callable[[], list]]:
    """F13: номинальный рост против уровня 2023 года; медианная линия; крайние точки — у края оси
    с подписью. Возвращает фигуру и отложенный шаг подписей точек (его выполняет ``_save``)."""
    fig, ax = style.new_figure("full")
    level, growth = frame["level_2023"], frame["growth"]
    lo, hi = (float(v) for v in growth.quantile(list(p["display_quantiles"])))
    span = hi - lo
    lo, hi = lo - 0.05 * span, hi + 0.05 * span
    shown = growth.clip(lo, hi)
    inside = growth.between(lo, hi)
    ax.scatter(level[inside], shown[inside], s=7, color=style.CONTEXT, alpha=0.8, linewidths=0, zorder=2)
    for mask, marker in ((growth > hi, "^"), (growth < lo, "v")):
        if mask.any():
            ax.scatter(level[mask], shown[mask], s=16, marker=marker, color=style.CONTEXT, zorder=2)
    trend = _binned_median(np.log(level), growth, int(p["trend_bins"]))
    ax.plot(np.exp(trend["x"]), trend["y"], color=style.ACCENT, linewidth=1.6, zorder=3)
    style.log_rub_axis(ax, "x", ticks=_rub_ticks(level))
    ax.set_ylim(lo - 0.04 * span, hi + 0.04 * span)
    ax.yaxis.set_major_formatter(style.ru_formatter("pct"))
    ax.set_xlabel("Траты на жителя в месяц, 2023 г. (шкала логарифмическая)")
    ax.set_ylabel("Рост трат 2024/2023, номинал")
    median = float(growth.median())
    ax.axhline(median, color=style.TEXT2, linewidth=0.8, linestyle=":", zorder=1)
    median_label = ax.annotate(
        f"медиана всех МО {style.fmt_pct(median, 1, sign=True)}",
        (float(level.min()), median),
        xytext=(0, 3),
        textcoords="offset points",
        va="bottom",
        fontsize=style.POINT_LABEL_PT,
        color=style.TEXT2,
        zorder=4,
        path_effects=[patheffects.withStroke(linewidth=2.5, foreground="white")],
    )
    n = int(p["label_n"])
    idx = spread_extremes(np.log(level), shown, growth, n, p["label_box"])
    labels = [f"{names.get(i, '')}: {style.fmt_pct(growth[i], 0, sign=True)}" for i in idx]
    avoid = [median_label, _trend_label(ax, trend, int(p["trend_bins"]))]
    return fig, lambda: label_points_clear(
        ax, level.loc[idx], shown.loc[idx], labels, avoid=avoid, max_labels=n
    )


def _fig_nominal_real(frame: pd.DataFrame):
    """F17: годовой прирост трат по месяцам, номинал и реал (только при ``eda.cpi``)."""
    fig, ax = style.new_figure("full")
    m = frame["month"].to_numpy()
    ax.plot(m, frame["nominal"], color=style.ACCENT, linewidth=2.0)
    ax.plot(m, frame["real"], color=style.ACCENT, linewidth=1.4, linestyle="--")
    ax.axhline(0, color=style.TEXT2, linewidth=0.8)
    style.month_axis(ax, len(m), start=int(m.min()))
    ax.yaxis.set_major_formatter(style.ru_formatter("pct"))
    ax.set_xlim(m.min() - 0.5, m.max() + 2.2)
    for col, text in (("nominal", "номинал"), ("real", "реал")):
        ax.annotate(
            text,
            (m.max(), frame[col].iloc[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=style.POINT_LABEL_PT,
            color=style.ACCENT,
        )
    ax.set_ylabel("Медиана годового прироста трат по МО")
    return fig


# --- Раздел --------------------------------------------------------------------------------------


def _weak(rho: float, threshold: float) -> bool:
    return bool(np.isfinite(rho) and abs(rho) < threshold)


# Текст раздела: {{e4.ключ}} — числа из фактов (подставляет отчёт), $слот — формулировка, выбранная по данным.
# Одиночные переводы строк склеиваются в пробел, пустая строка — граница абзаца.
SUMMARY_TEMPLATE = """
**Что видно.** Корзина типичного МО в 2024 году (медианы долей по {{e4.n_basket}} МО с полными годами):
продовольствие {{e4.share_median_food}}, «Прочее» {{e4.share_median_other}}, маркетплейсы
{{e4.share_median_marketplace}}, транспорт {{e4.share_median_transport}}, здоровье {{e4.share_median_health}},
общепит {{e4.share_median_cafe}}. Сильнее всего МО различаются долей общепита: дисперсия CLR (логарифма доли
относительно среднего геометрического всех шести долей) у общепита {{e4.clr_var_cafe}}, у продовольствия
{{e4.clr_var_food}}. Первая главная компонента CLR несёт {{e4.pc1_var}} дисперсии, сильнее всего нагружена на
{{e4.pc1_top_part}} ({{e4.pc1_top_load}}) и $pc1_link: ρ = {{e4.rho_pc1_level}}. Уровень трат объясняет
{{e4.r2_clr_level}} дисперсии CLR, регион {{e4.r2_clr_region}}, оба вместе {{e4.r2_clr_both}}; после
очистки от них остаётся {{e4.residual_share}}. $res_axis: на неё приходится {{e4.residual_pc1_var}}
дисперсии остатка, $stability.

Медианная доля маркетплейсов выросла с {{e4.mp_share_jan23}} в январе 2023 года до {{e4.mp_share_dec24}}
в декабре 2024-го, а декабрь к декабрю — с {{e4.mp_share_dec23}}. Доля в тратах всех жителей (траты МО,
взвешенные населением) выросла с {{e4.mp_resident_jan23}} до {{e4.mp_resident_dec24}}. Медиана годовых долей
{{e4.mp_share_2023}} в 2023 году и {{e4.mp_share_2024}} в 2024-м, типичный прирост за год {{e4.mp_pp_median}}
(у половины МО от {{e4.mp_pp_q25}} до {{e4.mp_pp_q75}}). В процентных пунктах доля растёт $pp_dir: ρ прироста
с уровнем трат {{e4.rho_mp_pp_level}}, внутри регионов {{e4.rho_mp_pp_level_within}}. У $group МО
с самыми низкими тратами прирост {{e4.dq1_pp}}, с самыми высокими {{e4.dq5_pp}} при стартовых долях
{{e4.q1_share_2023}} и {{e4.q5_share_2023}}. При этом $catch_up (ρ = {{e4.rho_mp_pp_initial}}
на раздельных месяцах и {{e4.rho_mp_pp_initial_naive}} без разделения). $rub
Разброс долей в процентных пунктах $iqr_dir (межквартильный размах {{e4.mp_iqr_2023}} → {{e4.mp_iqr_2024}}
п. п.)$joint в логарифмах $sd_dir (SD ln доли {{e4.mp_sd_log_2023}} → {{e4.mp_sd_log_2024}}). Прирост доли
за январь–июнь и за июль–декабрь $split (ρ = {{e4.mp_split_half}}); η² региона для прироста
{{e4.eta2_mp_pp}}. $step

Номинальный рост трат 2024/2023 в типичном МО {{e4.growth_median}} (10-й и 90-й перцентили {{e4.growth_p10}}
и {{e4.growth_p90}}); по категориям: маркетплейсы {{e4.growth_marketplace}}, общепит {{e4.growth_cafe}},
продовольствие {{e4.growth_food}}, здоровье {{e4.growth_health}}, транспорт {{e4.growth_transport}}, «Прочее»
{{e4.growth_other}}. $real $growth_rel $half

**Что это значит для сюжета.** С4 (состав корзины): $s4_lead, поэтому признаками узлов годятся CLR,
очищенные от лог-уровня и региона (показатель сводки — счёт очищенной PC1); $s4. С5 (сдвиг к маркетплейсам):
$s5. С3 (устойчивость потребления): $s3.
"""


def _ge(value: Any, threshold: float) -> bool:
    """value ≥ threshold; пропуск (None, NaN) — False."""
    return value is not None and bool(np.isfinite(value)) and value >= threshold


def _grew(before: Any, after: Any) -> bool:
    """Выросла ли величина; пропуск — False."""
    return before is not None and after is not None and after > before


def _direction(before: Any, after: Any, up: str, down: str) -> str:
    """Слово направления изменения: ``up``, если выросло, иначе ``down``."""
    return up if _grew(before, after) else down


def _sign(rho: Any, weak: float) -> int:
    """Знак связи для выбора слов: +1, −1 или 0 (|ρ| < ``weak`` или пропуск)."""
    if rho is None or not np.isfinite(rho) or abs(rho) < weak:
        return 0
    return 1 if rho > 0 else -1


def _by_sign(rho: Any, weak: float, pos: str, neg: str, zero: str) -> str:
    """Формулировка по знаку связи: ``pos`` при ρ > 0, ``neg`` при ρ < 0, ``zero`` при слабой связи."""
    return {1: pos, -1: neg, 0: zero}[_sign(rho, weak)]


def _paragraphs(template: str) -> str:
    """Склеивает строки шаблона: одиночный перевод строки — пробел, пустая строка — граница абзаца."""
    blocks = [
        " ".join(line.strip() for line in b.strip().splitlines()) for b in template.strip().split("\n\n")
    ]
    return "\n\n".join(blocks)


def _basket_slots(v: Mapping[str, Any], p: Mapping[str, Any], rej: Mapping[str, float]) -> dict[str, str]:
    """Слоты текста про корзину: связь PC1 с уровнем, ось остатка, её устойчивость, вывод для С4."""
    ph = "{{" + SECTION_ID + ".%s}}"
    rho_pc1 = v["rho_pc1_level"]
    pc1_close = rho_pc1 is not None and _ge(abs(rho_pc1), p["headline"]["pc1_rho_min"])
    same_top = str(v["residual_pc1_top"]).lower() == str(v["pc1_top_part"]).lower()
    load = ph % "residual_pc1_top_load"
    res_axis = (
        f"Главная ось остатка {'снова ' if same_top else ''}держится на части «{ph % 'residual_pc1_top'}» "
        f"(нагрузка {load})"
    )
    stable = _ge(v["residual_pc1_stability"], rej["s4_residual_stability_min"])
    enough = _ge(v["residual_share"], rej["s4_residual_share_min"])
    stab = ph % "residual_pc1_stability"
    if stable and enough:
        s4 = (
            "очищенной структуры остаётся достаточно, и её главная ось устойчива между годами: "
            "критерии отказа выполнены"
        )
    elif stable:
        s4 = (
            "главная ось очищенной структуры устойчива между годами, но после очистки различий остаётся "
            "мало: критерий отказа по доле остатка не выполнен, корзина годится как слой"
        )
    else:
        s4 = "очищенная структура неустойчива между годами: критерий отказа по устойчивости не выполнен"
    if str(v["residual_pc1_top"]).lower() == style.LABELS["cafe"].lower():
        s4 += "; ось остатка держится на общепите, а его доля, вероятно, зависит от присутствия сетевых кафе"
    return {
        "pc1_link": "почти повторяет уровень трат" if pc1_close else "связана с уровнем трат слабее",
        "res_axis": res_axis,
        "stability": (
            f"её счёт в 2023 и 2024 годах согласован (ρ = {stab})"
            if stable
            else f"но её счёт в 2023 и 2024 годах согласован слабо (ρ = {stab})"
        ),
        "s4_lead": (
            "уровень трат и регион объясняют лишь часть различий корзин"
            if enough
            else "сырая корзина во многом повторяет уровень трат и регион"
        ),
        "s4": s4,
    }


def _marketplace_slots(
    v: Mapping[str, Any], p: Mapping[str, Any], rej: Mapping[str, float]
) -> dict[str, str]:
    """Слоты текста про маркетплейсы: направление прироста, догоняние, рубли, разброс, ступенька, вывод
    для С5."""
    ph = "{{" + SECTION_ID + ".%s}}"
    weak = p["weak_rho"]
    pp_sign = _sign(v["rho_mp_pp_level"], weak)
    incr_sign = _sign(v["rho_mp_rub_incr_level"], weak)
    no_catch_up = _sign(v["rho_mp_pp_initial"], weak) == 0
    differs = pp_sign != 0 and incr_sign != 0 and pp_sign != incr_sign
    pct_dir = _by_sign(
        v["rho_mp_rub_growth_level"],
        weak,
        "выше у МО с высоким уровнем трат",
        "выше у МО с низким уровнем трат",
        "почти не связан с уровнем трат",
    )
    incr_dir = _by_sign(
        v["rho_mp_rub_incr_level"],
        weak,
        "больше у МО с высоким уровнем трат",
        "больше у МО с низким уровнем трат",
        "почти не связана с уровнем трат",
    )
    rub = (
        f"{'В рублях картина другая' if differs else 'В рублях'}: рост трат на маркетплейсах в медиане "
        f"{ph % 'mp_rub_growth_median'}, в процентах он {pct_dir} (ρ с уровнем трат "
        f"{ph % 'rho_mp_rub_growth_level'}), {'но' if differs else 'а'} прибавка в рублях {incr_dir} "
        f"(ρ = {ph % 'rho_mp_rub_incr_level'})."
    )
    iqr_dir = _direction(v["mp_iqr_2023"], v["mp_iqr_2024"], "растёт", "сжимается")
    sd_dir = _direction(v["mp_sd_log_2023"], v["mp_sd_log_2024"], "растёт", "сжимается")
    split_ok = _ge(v["mp_split_half"], rej["s5_split_half_min"])
    step_ok = v["mp_step_max_pp"] is not None and v["mp_step_max_pp"] <= p["step_test_pp"]
    step_nums = f"{ph % 'mp_step_max_pp'} п. п. ({ph % 'mp_step_month'})"
    step = (
        "Скачка, похожего на смену границ категорий, нет: наибольшее изменение годового прироста доли между "
        f"соседними месяцами {step_nums}."
        if step_ok
        else f"Есть скачок годового прироста доли между соседними месяцами, {step_nums}: нужно проверить "
        "границы категорий."
    )
    within = v["rho_mp_pp_level_within"]
    within_ok = within is not None and bool(np.isfinite(within)) and abs(within) >= rej["s5_rho_within_min"]
    pp_where = _by_sign(
        v["rho_mp_pp_level"],
        weak,
        "быстрее там, где тратят больше",
        "быстрее там, где тратят меньше",
        "почти одинаково при любом уровне трат",
    )
    if split_ok and within_ok and step_ok:
        wording = [f"доля онлайна растёт {pp_where}"]
        if no_catch_up:
            wording.append("догоняния по доле нет")
        if differs and incr_sign > 0:
            wording.append("в рублях разрыв растёт")
        s5 = (
            "критерии отказа выполнены; сигнал устойчивый, но относительный: признаки узла — прирост доли "
            f"в п. п. и в рублях вместе; формулировка — «{', '.join(wording)}»"
        )
    else:
        s5 = "не все критерии отказа выполнены (полугодия, связь внутри регионов, ступенька), скорее слой"
    return {
        "group": GROUP_WORDS[int(p["level_groups"])][1],
        "pp_dir": pp_where,
        "catch_up": _by_sign(
            v["rho_mp_pp_initial"],
            weak,
            "прирост выше там, где доля и так была выше",
            "там, где доля была ниже, она растёт быстрее: есть догоняние по доле",
            "догоняния по доле нет: прирост почти не связан с начальной долей",
        ),
        "rub": rub,
        "iqr_dir": iqr_dir,
        "sd_dir": sd_dir,
        "joint": ", а" if iqr_dir != sd_dir else " и",
        "split": "согласован" if split_ok else "согласован слабо",
        "step": step,
        "s5": s5,
    }


def _growth_slots(
    v: Mapping[str, Any], p: Mapping[str, Any], rej: Mapping[str, float], cpi_used: bool
) -> dict[str, str]:
    """Слоты текста про рост: реал, расхождение уровней, согласованность полугодий, вывод для С3."""
    ph = "{{" + SECTION_ID + ".%s}}"
    sign = _sign(v["rho_growth_level"], p["weak_rho"])
    sd_up = _grew(v["sd_loglevel_2023"], v["sd_loglevel_2024"])
    relation = _by_sign(
        v["rho_growth_level"],
        p["weak_rho"],
        "рост выше там, где тратили больше",
        "рост выше там, где тратили меньше",
        "рост почти не связан с уровнем трат",
    )
    if sign > 0 and sd_up:
        lead = "Уровни трат расходятся: "
    elif sign < 0 and not sd_up:
        lead = "Уровни трат сближаются: "
    else:
        lead = ""
    text = (
        f"{relation} (ρ = {ph % 'rho_growth_level'}, внутри регионов {ph % 'rho_growth_level_within'}, "
        f"на раздельных месяцах {ph % 'rho_growth_level_split'}), {'и' if lead else 'а'} разброс "
        f"лог-уровня (SD) {'вырос' if sd_up else 'снизился'} с {ph % 'sd_loglevel_2023'} до "
        f"{ph % 'sd_loglevel_2024'}."
    )
    s3_ok = _ge(v["growth_half_consistency"], rej["s3_half_consistency_min"])
    half = ph % "growth_half_consistency"
    return {
        "real": (
            f"С поправкой на цены (ИПЦ, {ph % 'cpi_source'}) медианный рост {ph % 'real_growth_median'}."
            if cpi_used
            else "Реальный рост не считается: ИПЦ в конфиге не задан, МО сравниваются только между собой."
        ),
        "growth_rel": lead + text if lead else text[:1].upper() + text[1:],
        "half": (
            f"Рост отдельных МО за январь–июнь и за июль–декабрь согласован: ρ = {half}."
            if s3_ok
            else f"Но рост отдельных МО за январь–июнь и за июль–декабрь согласован слабо: ρ = {half}."
        ),
        "s3": (
            "рост по полугодиям согласован, критерий отказа выполнен"
            if s3_ok
            else "рост МО по полугодиям согласован слабо, критерий отказа не выполнен: рост годится только "
            "как контекст"
        ),
    }


def _summary_slots(
    v: Mapping[str, Any], p: Mapping[str, Any], rej: Mapping[str, float], cpi_used: bool
) -> dict[str, str]:
    """Формулировки для ``SUMMARY_TEMPLATE``, выбранные по знакам фактов и порогам ``eda.rejection``."""
    return {
        **_basket_slots(v, p, rej),
        **_marketplace_slots(v, p, rej),
        **_growth_slots(v, p, rej, cpi_used),
    }


def _summary_md(f: dict[str, Fact], p: Mapping[str, Any], rej: Mapping[str, float], cpi_used: bool) -> str:
    """«Что видно» и «Что это значит» (``SUMMARY_TEMPLATE``) с ``{{e4.ключ}}`` вместо чисел.

    Утверждения из заголовков F09–F13 защищены их проверками; остальные формулировки (направления,
    «согласован», «догоняния нет», выполнен ли критерий отказа) выбираются по знакам фактов и порогам,
    поэтому текст остаётся верным при любых данных.
    """
    values = {k.split(".", 1)[1]: fact.value for k, fact in f.items()}
    text = Template(_paragraphs(SUMMARY_TEMPLATE)).substitute(_summary_slots(values, p, rej, cpi_used))
    return nbsp(text)


CAVEATS: tuple[str, ...] = (
    "Рост — номинальный, в текущих ценах: общая инфляция в нём не снята. МО сравниваются по относительному "
    "росту и ранговым связям, в которых общий рост цен сокращается.",
    "Траты привязаны к жителям МО: траты приезжих в курортных МО не видны, а траты жителей вне своего МО "
    "(поездки, онлайн) по смыслу входят, но модель привязки СберИндекс не раскрывает.",
    "«Прочее» — остаток «Все категории» минус пять категорий: в нём иные траты; по описанию набора авиа- и "
    "ж/д билеты не входят в «Транспорт» и, вероятно, попадают сюда.",
    "Январь 2023 → декабрь 2024 сравнивает разные месяцы года: сезонность доли маркетплейсов в этом "
    "отношении не снята. Сопоставимы по сезону декабрь к декабрю ({{e4.mp_share_dec23}} → "
    "{{e4.mp_share_dec24}}) и годовые медианы ({{e4.mp_share_2023}} → {{e4.mp_share_2024}}).",
    "Сдвиг к маркетплейсам может частично быть переклассификацией продавцов: тест ступеньки ловит только "
    "резкий скачок, плавную переклассификацию он не видит.",
    "Знак главной компоненты условен (наибольшая по модулю нагрузка положительна): ρ PC1 с уровнем "
    "читается по модулю.",
    "«Внутри регионов» — ранги всей выборки, центрированные по региону (Б.1, п. 9). На самих значениях, "
    "центрированных по региону, связи немного слабее; оба числа есть в пояснениях фактов (facts.json).",
)


@dataclass(frozen=True)
class CpiInfo:
    """ИПЦ из ``eda.cpi``: годовой рост цен (отношение средних), помесячный (месяц 2024 / месяц 2023),
    источник."""

    annual: float
    monthly: pd.Series
    source: str


def cpi_info(cfg: Any) -> CpiInfo | None:
    """``CpiInfo`` из конфига или None, если ``eda.cpi`` не задан (тогда рост только номинальный)."""
    cpi = cfg["eda"].get("cpi")
    if not cpi:
        return None
    annual, monthly = cpi_ratios(cpi_index(cpi))
    return CpiInfo(annual=annual, monthly=monthly, source=str(cpi["source"]))


def _fv(ctx: SectionContext, key: str) -> Fact:
    """Уже записанный факт раздела по ключу без префикса."""
    return ctx.facts[f"{SECTION_ID}.{key}"]


FactWriter = Callable[..., Fact]


def reference_note(key: str, kind: FactKind, ref: Mapping[str, float]) -> str:
    """Пояснение к расхождению с ориентиром Б.4: каким определением посчитан ориентир и что это определение
    даёт на тех же данных; пустая строка, если ключ считается так же, как ориентир."""
    if key not in ref or key not in REF_DEFINITIONS:
        return ""
    text = make_fact(f"{SECTION_ID}.{key}", ref[key], kind).text
    return f"Ориентир Б.4 посчитан {REF_DEFINITIONS[key]}; тем же способом здесь {text}"


def fact_writer(ctx: SectionContext, ref: Mapping[str, float]) -> FactWriter:
    """``ctx.fact``, который дописывает в ``note`` пояснение ``reference_note`` для ключей из ``ref``."""

    def write(key: str, value: Any, kind: FactKind, note: str = "") -> Fact:
        extra = reference_note(key, kind, ref)
        return ctx.fact(key, value, kind, f"{note}. {extra}" if extra else note)

    return write


def _save(ctx: SectionContext, fig: Any, *, place: Callable[[], Any] | None = None, **kw: Any) -> None:
    """``ctx.save_figure`` с типографикой ru-text в заголовке, подзаголовке и альт-тексте.

    ``place`` — отложенные подписи точек: они ставятся после того, как поле графика заняло окончательное
    место (шапка и подвал рисунка уменьшают его), иначе подписи, разведённые на полном поле, сходятся.
    """
    for key in ("title", "subtitle", "alt"):
        kw[key] = nbsp(kw[key])
    if place is not None:
        reserve_layout(
            fig, kw["title"], kw["subtitle"], kw.get("source", style.SOURCE_SBER), int(kw["fid"][1:])
        )
        place()
    ctx.save_figure(fig, **kw)


def reserve_layout(fig: Any, title: str, subtitle: str, source: str, number: int) -> None:
    """Раскладка поля графика как на готовом рисунке, но без текстов шапки и подвала.

    ``style.finish`` пишет заголовок, подзаголовок и источник и сжимает поле графика; здесь он вызывается с
    теми же строками, после чего его тексты убираются: поле остаётся на окончательном месте, а
    ``ctx.save_figure`` вызовет ``style.finish`` ещё раз и поставит тексты.
    """
    before = len(fig.texts)
    style.finish(fig, title, subtitle, source, number)
    for text in list(fig.texts[before:]):
        text.remove()


def _basket_facts(fact: FactWriter, basket: BasketResult) -> None:
    """Факты корзины: доли, дисперсии CLR, PCA сырых и очищенных CLR, R², устойчивость."""
    summ = basket.summary
    sample_note = f"МО с 12 месяцами в {Y0} и {Y1} годах; доли {Y1} года, медиана по МО без весов"
    fact("n_basket", basket.n, "int", "МО с 12 месяцами в обоих годах — выборка сюжета С4")
    for part in PARTS:
        fact(f"share_median_{part}", summ.loc[part, "median"], "pct", sample_note)
    for part in PARTS:
        fact(
            f"clr_var_{part}",
            summ.loc[part, "clr_var"],
            "num3",
            f"дисперсия CLR доли {Y1} года по МО, ddof = 1",
        )
    argmax = str(summ["clr_var"].idxmax())
    fact("clr_var_argmax", argmax, "str", "код части с наибольшей дисперсией CLR")
    fact("clr_var_argmax_label", style.LABELS[argmax].lower(), "str", "та же часть по-русски")
    fact("pc1_var", basket.raw.explained[PC1], "pct", f"доля дисперсии сырых CLR {Y1} года у PC1")
    fact(
        "rho_pc1_level",
        basket.rho_pc1_level,
        "rho",
        f"ρ Спирмена счёта PC1 сырых CLR {Y1} года с log_level_{Y1}; знак оси условен: наибольшая по модулю "
        "нагрузка положительна",
    )
    top = str(basket.raw.loadings[PC1].abs().idxmax())
    fact(
        "pc1_top_part",
        style.LABELS[top].lower(),
        "str",
        "часть с наибольшей по модулю нагрузкой PC1 сырых CLR",
    )
    fact("pc1_top_load", basket.raw.loadings.loc[top, PC1], "num2", "нагрузка этой части на PC1 сырых CLR")
    r2_note = f"R² МНК всех CLR {Y1} года (доля суммарной дисперсии)"
    fact("r2_clr_level", basket.r2_level, "pct", f"{r2_note} на log_level_{Y1}")
    fact("r2_clr_region", basket.r2_region, "pct", f"{r2_note} на дамми регионов")
    fact("r2_clr_both", basket.clean.r2, "pct", f"{r2_note} на лог-уровень и дамми регионов")
    fact(
        "residual_share",
        basket.clean.residual_share,
        "pct",
        "1 − r2_clr_both: доля дисперсии CLR после очистки",
    )
    fact("residual_pc1_var", basket.clean.explained[PC1], "pct", "доля дисперсии остатков у очищенной PC1")
    clean_top = str(basket.clean.loadings[PC1].abs().idxmax())
    fact(
        "residual_pc1_top",
        style.LABELS[clean_top],
        "str",
        "часть с наибольшей по модулю нагрузкой очищенной PC1",
    )
    fact(
        "residual_pc1_top_load",
        basket.clean.loadings.loc[clean_top, PC1],
        "num2",
        "нагрузка этой части на очищенную PC1 (знак оси условен: наибольшая по модулю нагрузка положительна)",
    )
    fact(
        "residual_pc1_stability",
        basket.stability,
        "rho",
        f"ρ счёта очищенной PC1 {Y1} года и счёта {Y0} года на той же оси "
        f"(остатки {Y0} — от своей регрессии)",
    )
    fact(
        "residual_axis_cos",
        basket.axis_cos,
        "num2",
        "|cos| нагрузок очищенной PC1, посчитанной по каждому году",
    )


def _marketplace_facts(fact: FactWriter, mpc: Mapping[str, Any], gap: pd.DataFrame) -> None:
    """Факты маркетплейсов: динамика долей, связь прироста с уровнем и начальной долей, рубли, ступенька."""
    words = GROUP_WORDS[len(gap)]
    n = int(mpc["n"])
    note = f"МО с долями обоих лет (n = {n}); прирост — годовая доля {Y1} минус годовая доля {Y0}"
    fact("n_mp", n, "int", "МО с 12 месяцами в обоих годах — выборка проверок С5")
    fact(
        "mp_share_jan23",
        mpc["mp_share_jan23"],
        "pct",
        "медиана долей маркетплейсов по МО месяца, январь 2023",
    )
    fact(
        "mp_share_dec23",
        mpc["mp_share_dec23"],
        "pct",
        "медиана долей маркетплейсов по МО месяца, декабрь 2023: декабрь к декабрю — без сезонности",
    )
    fact(
        "mp_share_dec24",
        mpc["mp_share_dec24"],
        "pct",
        "медиана долей маркетплейсов по МО месяца, декабрь 2024",
    )
    jan = mpc["mp_share_jan23"]
    ratio = mpc["mp_share_dec24"] / jan if jan and np.isfinite(jan) else float("nan")
    fact(
        "mp_share_ratio",
        ratio,
        "num1",
        "декабрь 2024 / январь 2023, медианы долей; сравнивает разные месяцы года, сезонность доли не снята",
    )
    resident = "Σ v_marketplace × pop_avg / Σ v_all × pop_avg по МО месяца (national.resident_share, Б.2)"
    fact(
        "mp_resident_jan23",
        mpc["mp_resident_jan23"],
        "pct",
        f"доля в тратах всех жителей, январь 2023: {resident}",
    )
    fact(
        "mp_resident_dec24",
        mpc["mp_resident_dec24"],
        "pct",
        f"доля в тратах всех жителей, декабрь 2024: {resident}",
    )
    fact("mp_share_2023", mpc["mp_share_2023"], "pct", f"медиана годовых долей {Y0}; {note}")
    fact("mp_share_2024", mpc["mp_share_2024"], "pct", f"медиана годовых долей {Y1}; {note}")
    for key in ("mp_pp_median", "mp_pp_q25", "mp_pp_q75"):
        fact(key, mpc[key], "pp", note)
    fact("rho_mp_pp_level", mpc["rho_mp_pp_level"], "rho", f"ρ Спирмена mp_pp_change с log_level_{Y0}")
    fact(
        "rho_mp_pp_level_within",
        mpc["rho_mp_pp_level_within"],
        "rho",
        "то же внутри регионов: ранги всей выборки, центрированные по региону (stats.spearman_within)",
    )
    fact(
        "rho_mp_pp_initial",
        mpc["rho_mp_pp_initial"],
        "rho",
        f"ρ прироста доли по чётным месяцам с начальной долей по нечётным месяцам {Y0}: без регрессии "
        f"к среднему. Обратное разделение (начало — чётные месяцы) даёт "
        f"{make_fact('rev', mpc['rho_mp_pp_initial_rev'], 'rho').text}; ориентир Б.4 — диапазон двух "
        "вариантов разделения",
    )
    fact(
        "rho_mp_pp_initial_rev",
        mpc["rho_mp_pp_initial_rev"],
        "rho",
        "то же: начало — чётные месяцы, прирост — нечётные",
    )
    fact(
        "rho_mp_pp_initial_naive",
        mpc["rho_mp_pp_initial_naive"],
        "rho",
        "без разделения месяцев: смещено регрессией к среднему",
    )
    fact(
        "mp_split_half",
        mpc["mp_split_half"],
        "rho",
        "ρ прироста доли за январь–июнь и за июль–декабрь (год к году)",
    )
    fact("mp_iqr_2023", mpc["mp_iqr_2023"], "num1", f"межквартильный размах годовой доли {Y0}, п. п.")
    fact("mp_iqr_2024", mpc["mp_iqr_2024"], "num1", f"межквартильный размах годовой доли {Y1}, п. п.")
    fact("mp_sd_log_2023", mpc["mp_sd_log_2023"], "num3", f"SD ln годовой доли {Y0}")
    fact("mp_sd_log_2024", mpc["mp_sd_log_2024"], "num3", f"SD ln годовой доли {Y1}")
    fact(
        "mp_rub_growth_median",
        mpc["mp_rub_growth_median"],
        "pct_signed",
        "медианный рост трат на маркетплейсах в ₽, номинал",
    )
    fact(
        "rho_mp_rub_growth_level", mpc["rho_mp_rub_growth_level"], "rho", f"ρ роста в рублях с log_level_{Y0}"
    )
    fact(
        "rho_mp_rub_incr_level",
        mpc["rho_mp_rub_incr_level"],
        "rho",
        f"ρ прибавки в ₽ в месяц с log_level_{Y0}",
    )
    fact("eta2_mp_pp", mpc["eta2_mp_pp"], "num2", "η² региона для mp_pp_change")
    fact(
        "mp_step_max_pp",
        mpc["mp_step_max_pp"],
        "num2",
        "тест ступеньки: max |Δ(m+1) − Δ(m)| годового прироста медианной доли, п. п.",
    )
    fact("mp_step_month", mpc["mp_step_month"], "str", "пара соседних месяцев с наибольшим скачком")
    first, last = gap.iloc[0], gap.iloc[-1]
    gap_note = f"медиана доли в {words[1]} МО по log_level_{Y0}"
    for tag, row, where in (("q1", first, "нижняя"), ("q5", last, "верхняя")):
        for y in YEARS:
            fact(
                f"{tag}_share_{y}",
                row[f"share_{y}"],
                "pct",
                f"{where} {words[3]} по уровню, {y}; {gap_note}",
            )
    for key, row, where in (("dq1_pp", first, "нижней"), ("dq5_pp", last, "верхней")):
        fact(
            key,
            row["dpp"],
            "pp",
            f"разность медиан долей {Y1} и {Y0} в {where} {words[1]}; {rounding_note(row)}",
        )


def rounding_note(row: Mapping[str, float]) -> str:
    """Почему разность медиан долей может разойтись на 0,1 п. п. с разностью долей, написанных рядом.

    Разность считается по точным медианам, а доли в тексте округлены до десятых процента: 13,05% − 10,51%
    = +2,55 п. п. пишется «+2,5», хотя из «13,1%» и «10,5%» выходит «+2,6».
    """
    a, b = PP * row[f"share_{Y1}"], PP * row[f"share_{Y0}"]
    exact = style.fmt_pp(a - b, 2)
    rounded = style.fmt_pp(round(a, 1) - round(b, 1), 1)
    return (
        f"точно {style.fmt_num(a, 2)}% − {style.fmt_num(b, 2)}% = {exact}; из долей, округлённых до десятых, "
        f"{rounded}"
    )


def _growth_facts(
    fact: FactWriter, mo: pd.DataFrame, gt: pd.DataFrame, gvl: Mapping[str, Any], cpi: CpiInfo | None
) -> None:
    """Факты роста: медиана и перцентили, по категориям, связь с уровнем, разброс уровней, полугодия, ИПЦ."""
    growth = mo.loc[mo["series_status"].astype(str) == "full", "growth"].dropna()
    note = f"рост среднемесячных трат {Y1}/{Y0} − 1, номинал; МО с полным рядом (n = {len(growth)})"
    fact("growth_median", growth.median(), "pct_signed", note)
    fact("growth_p10", growth.quantile(0.1), "pct_signed", note)
    fact("growth_p90", growth.quantile(0.9), "pct_signed", note)
    for row in gt.itertuples(index=False):
        fact(f"growth_{row.code}", row.growth_median, "pct_signed", f"медиана по МО: {row.label}, номинал")
    fact("rho_growth_level", gvl["rho_growth_level"], "rho", f"ρ growth с log_level_{Y0}")
    fact(
        "rho_growth_level_within",
        gvl["rho_growth_level_within"],
        "rho",
        "то же внутри регионов (stats.spearman_within)",
    )
    fact(
        "rho_growth_level_split",
        gvl["rho_growth_level_split"],
        "rho",
        f"уровень — нечётные месяцы {Y0}, рост — чётные: без общего шума уровня {Y0}",
    )
    total = gt.set_index("code").loc["all"]
    fact(
        "sd_loglevel_2023", total["sd_log_2023"], "num3", f"SD ln среднемесячных трат {Y0}, МО с полным рядом"
    )
    fact(
        "sd_loglevel_2024", total["sd_log_2024"], "num3", f"SD ln среднемесячных трат {Y1}, МО с полным рядом"
    )
    fact(
        "growth_half_consistency",
        total["half_consistency"],
        "rho",
        "ρ роста января–июня и июля–декабря (год к году)",
    )
    if cpi is not None:
        fact("cpi_annual", cpi.annual - 1.0, "pct_signed", f"рост цен: среднее ИПЦ {Y1} / среднее {Y0} − 1")
        fact(
            "real_growth_median",
            real_growth(float(growth.median()), cpi.annual),
            "pct_signed",
            "медианный номинальный рост, делённый на годовой рост цен",
        )
        fact("cpi_source", cpi.source, "str", "источник ИПЦ из eda.cpi")


def _basket_figures(
    ctx: SectionContext, basket: BasketResult, names: pd.Series, p: Mapping[str, Any]
) -> None:
    """F09 и F10."""
    hl, summ = p["headline"], basket.summary
    argmax = str(summ["clr_var"].idxmax())
    food = style.fmt_pct(summ.loc["food", "median"])
    n = style.fmt_num(basket.n)
    _save(
        ctx,
        _fig_basket(basket),
        fid="F09",
        slug="basket",
        title=ctx.headline(
            f"Продовольствие — {food} трат, но сильнее всего МО различаются долей общепита",
            argmax == "cafe",
            f"F09 basket: наибольшая дисперсия CLR у {argmax!r}",
        ),
        subtitle=(
            f"Доли шести частей трат {Y1} г.: медиана и половина МО; справа — дисперсия CLR "
            f"(разброс логарифма доли); n = {n}"
        ),
        alt=(
            f"В типичном МО продовольствие занимает {food} трат, а сильнее всего МО различаются долей "
            f"общепита: дисперсия CLR {_fv(ctx, 'clr_var_cafe').text} при {_fv(ctx, 'clr_var_food').text} "
            "у продовольствия"
        ),
        data=summ.reset_index(names="part"),
        check="clr_var_argmax == 'cafe'",
    )
    rho = basket.rho_pc1_level
    rho_text = _fv(ctx, "rho_pc1_level").text
    top = str(basket.raw.loadings[PC1].abs().idxmax())
    pc1 = basket.raw.scores[PC1]
    fig10, place10 = _fig_basket_vs_level(basket.level, pc1, names, p, style.LABELS[top])
    _save(
        ctx,
        fig10,
        place=place10,
        fid="F10",
        slug="basket_vs_level",
        title=ctx.headline(
            f"Главная ось состава корзины почти повторяет уровень трат: ρ = {rho_text}",
            bool(np.isfinite(rho) and abs(rho) >= hl["pc1_rho_min"]),
            f"F10 basket_vs_level: |ρ| = {style.fmt_rho(abs(rho))}, порог {hl['pc1_rho_min']}",
        ),
        subtitle=(
            f"PC1 сырых CLR {Y1} г.: {_fv(ctx, 'pc1_var').text} дисперсии, нагрузка «{style.LABELS[top]}» "
            f"{style.fmt_num(basket.raw.loadings.loc[top, PC1], 2)}; n = {n}"
        ),
        alt=(
            "Счёт первой главной компоненты долей трат почти повторяет уровень трат МО: "
            f"ρ = {rho_text}, главная "
            f"нагрузка — {style.LABELS[top].lower()}"
        ),
        data=pd.DataFrame(
            {
                "territory_id": pc1.index,
                "name": names.reindex(pc1.index).to_numpy(),
                f"level_{Y1}": basket.level.to_numpy(),
                "pc1": pc1.to_numpy(),
            }
        ),
        check=f"abs(rho_pc1_level) >= {hl['pc1_rho_min']}",
    )


def _marketplace_figures(
    ctx: SectionContext,
    mpc: Mapping[str, Any],
    gap: pd.DataFrame,
    trend: pd.DataFrame,
    p: Mapping[str, Any],
    rej: Mapping[str, float],
) -> None:
    """F11 и F12."""
    hl = p["headline"]
    jan, dec = _fv(ctx, "mp_share_jan23").text, _fv(ctx, "mp_share_dec24").text
    ratio = _fv(ctx, "mp_share_ratio").value
    _save(
        ctx,
        _fig_mp_trend(trend),
        fid="F11",
        slug="marketplace_trend",
        title=ctx.headline(
            "Доля маркетплейсов в типичном МО за два года выросла вдвое: "
            f"{jan}{style.NBSP}→{style.NBSP}{dec}",
            ratio is not None and ratio >= hl["mp_growth_ratio_min"],
            f"F11 marketplace_trend: декабрь 2024 / январь 2023 = {style.fmt_num(ratio, 2)}, "
            f"порог {hl['mp_growth_ratio_min']}",
        ),
        subtitle=(
            "Типичное МО — медиана долей по МО месяца, полоса — половина МО; пунктир — доля в тратах всех "
            "жителей (веса — население)"
        ),
        alt=(
            f"Медианная доля маркетплейсов в тратах МО выросла с {jan} в январе 2023 года до {dec} "
            "в декабре 2024 года"
        ),
        data=trend,
        check=f"mp_share_dec24 >= {hl['mp_growth_ratio_min']} * mp_share_jan23",
    )
    first, last = gap.iloc[0], gap.iloc[-1]
    diff = float(first["dpp"] - last["dpp"])
    within = mpc["rho_mp_pp_level_within"]
    ok = diff >= hl["quintile_gap_pp_min"] and within <= -rej["s5_rho_within_min"]
    dq1 = style.fmt_num(first["dpp"], 1, sign=True)
    dq5 = style.fmt_num(last["dpp"], 1, sign=True)
    start = start_phrase(gap, (_fv(ctx, "q1_share_2023").text, _fv(ctx, "q5_share_2023").text))
    rub = _by_sign(
        mpc["rho_mp_rub_incr_level"],
        p["weak_rho"],
        "в рублях прибавка выше у МО с высокими тратами",
        "в рублях прибавка выше у МО с низкими тратами",
        "прибавка в рублях почти не связана с тратами",
    )
    _save(
        ctx,
        _fig_mp_gap(gap),
        fid="F12",
        slug="marketplace_gap",
        title=ctx.headline(
            f"Где тратят меньше, доля маркетплейсов растёт быстрее: {dq1} против "
            f"{dq5}{style.NBSP}п.{style.NBSP}п.",
            ok,
            f"F12 marketplace_gap: dq1 − dq5 = {style.fmt_num(diff, 2)} п. п. "
            f"(порог {hl['quintile_gap_pp_min']}), "
            f"ρ внутри регионов {style.fmt_rho(within)} (порог −{rej['s5_rho_within_min']})",
        ),
        subtitle=(
            f"{GROUP_WORDS[len(gap)][0].capitalize()} МО по тратам {Y0} г.; {start}; {rub}; "
            f"n = {style.fmt_num(mpc['n'])}"
        ),
        alt=(
            f"Прирост доли маркетплейсов за год у МО с низкими тратами {_fv(ctx, 'dq1_pp').text}, у МО "
            f"с высокими {_fv(ctx, 'dq5_pp').text} при стартовых долях {_fv(ctx, 'q1_share_2023').text} и "
            f"{_fv(ctx, 'q5_share_2023').text}"
        ),
        data=gap,
        check=(
            f"dq1_pp - dq5_pp >= {hl['quintile_gap_pp_min']} and "
            f"rho_mp_pp_level_within <= -{rej['s5_rho_within_min']}"
        ),
    )


def _growth_figures(
    ctx: SectionContext,
    mo: pd.DataFrame,
    gvl: Mapping[str, Any],
    wide: pd.DataFrame,
    names: pd.Series,
    p: Mapping[str, Any],
    cpi: CpiInfo | None,
) -> None:
    """F13 и (при ИПЦ) F17."""
    hl = p["headline"]
    full = mo.loc[mo["series_status"].astype(str) == "full"]
    frame = full.set_index("territory_id")[[f"level_{Y0}", "growth", "region_code"]].dropna()
    rho, within, split = (
        gvl["rho_growth_level"],
        gvl["rho_growth_level_within"],
        gvl["rho_growth_level_split"],
    )
    ok = bool(np.isfinite(rho) and rho >= hl["growth_rho_min"] and within >= hl["growth_rho_min"])
    fig13, place13 = _fig_growth(frame, names, p)
    _save(
        ctx,
        fig13,
        place=place13,
        fid="F13",
        slug="growth_divergence",
        title=ctx.headline(
            "Уровни трат расходятся: где тратили больше, там выше и номинальный рост "
            f"(ρ = {style.fmt_rho(rho, sign=True)})",
            ok,
            f"F13 growth_divergence: ρ = {style.fmt_rho(rho)}, внутри регионов {style.fmt_rho(within)}, "
            f"порог {hl['growth_rho_min']}",
        ),
        subtitle=(
            f"Рост {Y1}/{Y0}, номинал; ρ внутри регионов {style.fmt_rho(within, sign=True)}, на раздельных "
            f"месяцах {style.fmt_rho(split, sign=True)}; n = {style.fmt_num(len(frame))}"
        ),
        alt=(
            f"Номинальный рост трат выше в МО, где траты в {Y0} году были выше: "
            f"ρ = {_fv(ctx, 'rho_growth_level').text}, "
            f"внутри регионов {_fv(ctx, 'rho_growth_level_within').text}"
        ),
        data=frame.assign(name=names.reindex(frame.index)).reset_index(),
        check=(
            f"rho_growth_level >= {hl['growth_rho_min']} and "
            f"rho_growth_level_within >= {hl['growth_rho_min']}"
        ),
    )
    if cpi is None:
        return
    nominal = yoy_by_month(wide, full["territory_id"])
    data = pd.DataFrame(
        {"month": list(MONTHS), "nominal": nominal.to_numpy(), "price_ratio": cpi.monthly.to_numpy()}
    )
    data["real"] = real_growth(data["nominal"], data["price_ratio"])
    nom, real = _fv(ctx, "growth_median"), _fv(ctx, "real_growth_median")
    real_value = float(real.value) if real.value is not None else float("nan")
    real_text = (
        f"реально — на {style.fmt_pct(real_value, 0)}"
        if real_value >= 0
        else f"реально снизились на {style.fmt_pct(-real_value, 0)}"
    )
    _save(
        ctx,
        _fig_nominal_real(data),
        fid="F17",
        slug="nominal_real",
        title=ctx.headline(
            f"Номинально траты выросли на {style.fmt_pct(nom.value, 0)}, {real_text}",
            nom.value is not None and nom.value > 0,
            f"F17 nominal_real: медианный номинальный рост {nom.text}, а заголовок говорит о росте",
        ),
        subtitle=f"Медиана по МО годового прироста трат по месяцам {Y1}/{Y0}; реал — с поправкой на ИПЦ",
        alt=f"Номинальный рост трат {Y1}/{Y0} в медиане {nom.text}, после поправки на рост цен — {real.text}",
        data=data,
        check="eda.cpi задан and growth_median > 0",
        source=style.join_sources(style.SOURCE_SBER, f"Росстат, ИПЦ ({cpi.source})"),
    )


def _num(decimals: int) -> Any:
    return lambda x: style.fmt_num(x, decimals)


def _tables(
    ctx: SectionContext, basket: BasketResult, mpc: Mapping[str, Any], gt: pd.DataFrame, step_pp: float
) -> None:
    """T08, T09, T10."""
    t08 = basket_table(basket)
    loads = [c for c in t08.columns if c.startswith(("raw_", "r2_", "clean_"))]
    ctx.save_table(
        t08,
        tid="T08",
        slug="basket_pca",
        title=(
            "Состав корзины: дисперсия CLR, нагрузки главных компонент сырых CLR, R² уровня и региона, "
            "нагрузка очищенной PC1; в последней строке — доли дисперсии"
        ),
        md_formats={
            "median_share_2024": style.fmt_pct,
            "clr_var": _num(CLR_VAR_DECIMALS),
            **dict.fromkeys(loads, _num(2)),
        },
        md_labels={
            "code": "Код",
            "label": "Часть трат",
            "median_share_2024": f"Медиана доли {Y1}",
            "clr_var": "Дисперсия CLR",
            **{f"raw_{pc.lower()}": pc for pc in basket.raw.loadings.columns},
            "r2_level": "R² уровень",
            "r2_region": "R² регион",
            "r2_both": "R² оба",
            "clean_pc1": "PC1 очищенных",
        },
    )
    ctx.save_table(
        _checks_table(ctx.facts, mpc, int(mpc["n"]), step_pp),
        tid="T09",
        slug="marketplace_checks",
        title="Проверки сдвига к маркетплейсам: связь с уровнем, догоняние, устойчивость, рубли, ступенька",
        md_formats={"value": _num(3)},
        md_labels={"label": "Проверка", "text": "Значение", "n": "МО", "key": "Факт", "value": "Число"},
    )
    formats = {c: (lambda x: style.fmt_pct(x, 1, sign=True)) for c in gt.columns if c.startswith("growth_")}
    formats.update({"half_consistency": style.fmt_rho, "sd_log_2023": _num(3), "sd_log_2024": _num(3)})
    ctx.save_table(
        gt,
        tid="T10",
        slug="growth_noise",
        title=f"Рост трат {Y1}/{Y0} по категориям (номинал), согласованность полугодий и разброс уровней",
        md_formats=formats,
        md_labels={
            "code": "Код",
            "label": "Категория",
            "growth_median": "Рост, медиана",
            "growth_p10": "10-й перцентиль",
            "growth_p90": "90-й перцентиль",
            "half_consistency": "ρ полугодий",
            "sd_log_2023": f"SD ln {Y0}",
            "sd_log_2024": f"SD ln {Y1}",
            "n": "МО",
            "growth_real_median": "Реальный рост",
        },
    )


def run_section(ctx: SectionContext) -> Finding:
    """Раздел E4: F09–F13 (F17 при ИПЦ), T08–T10, факты ``e4.*``, показатель МО ``basket_resid_pc1``."""
    cfg = ctx.cfg
    p = basket_params(cfg)
    p["step_test_pp"] = float(cfg["eda"]["step_test_pp"])
    rej = cfg["eda"]["rejection"]
    mo, wide, national = ctx.data.mo, ctx.data.panel_wide, ctx.data.national
    cpi = cpi_info(cfg)

    basket = basket_analysis(mo, n_components=int(p["n_components"]))
    halves = half_year_growth(wide)
    mpc = marketplace_checks(wide, mo, national, halves)
    gap = quintile_gap(mo, int(p["level_groups"]))
    gt = growth_table(wide, mo, halves)
    if cpi is not None:
        gt["growth_real_median"] = real_growth(gt["growth_median"], cpi.annual)
    gvl = growth_vs_level(wide, mo)

    fact = fact_writer(ctx, reference_variants(wide, mo))
    _basket_facts(fact, basket)
    _marketplace_facts(fact, mpc, gap)
    _growth_facts(fact, mo, gt, gvl, cpi)
    names = point_names(ctx.data.territories)
    _basket_figures(ctx, basket, names, p)
    _marketplace_figures(ctx, mpc, gap, _mp_trend_frame(wide, national), p, rej)
    _growth_figures(ctx, mo, gvl, wide, names, p, cpi)
    _tables(ctx, basket, mpc, gt, p["step_test_pp"])

    indicators = pd.DataFrame({"territory_id": mo["territory_id"].astype("int32")})
    indicators["basket_resid_pc1"] = (
        indicators["territory_id"].map(basket.clean.scores[PC1]).astype("float64")
    )
    return ctx.finding(
        title=TITLE,
        summary_md=_summary_md(ctx.facts, p, rej, cpi is not None),
        indicators=indicators,
        indicator_labels={
            "basket_resid_pc1": f"Корзина без уровня и региона: счёт очищенной PC1 CLR долей {Y1} г."
        },
        caveats=[nbsp(c) for c in CAVEATS],
    )


def _mp_trend_frame(wide: pd.DataFrame, national: pd.DataFrame) -> pd.DataFrame:
    """Данные F11: по месяцам медиана и квартили долей маркетплейсов по МО, доля в тратах жителей, число МО и
    годовой прирост медианной доли (п. п., только месяцы 2024 года — вход теста ступеньки)."""
    nat = national.loc[national["category"].astype(str) == MP].set_index("date")
    q = wide.groupby("date")[f"sh_{MP}"].quantile([0.25, 0.75]).unstack()
    t = wide.groupby("date")["t"].first()
    out = pd.DataFrame(
        {
            "t": t.astype(int),
            "median_share": nat["median_share"],
            "q25": q[0.25],
            "q75": q[0.75],
            "resident_share": nat["resident_share"],
            "n_mo": nat["n_mo"],
        }
    ).sort_values("t")
    yoy = yoy_pp(national)
    out["yoy_pp"] = [yoy[int(d[5:])] if d.startswith(str(Y1)) else np.nan for d in out.index.astype(str)]
    out.index.name = "date"
    return out.reset_index()


def _checks_table(facts: dict[str, Fact], mpc: Mapping[str, Any], n: int, step_pp: float) -> pd.DataFrame:
    """T09: строка на проверку сюжета С5 — ключ факта, подпись, число, готовый текст, число МО."""
    rows = [
        ("mp_pp_median", "Прирост доли за год, медиана"),
        ("rho_mp_pp_level", f"ρ прироста доли с уровнем трат {Y0}"),
        ("rho_mp_pp_level_within", "то же внутри регионов"),
        ("eta2_mp_pp", "η² региона для прироста доли"),
        ("rho_mp_pp_initial", "ρ прироста с начальной долей: начало — нечётные месяцы, прирост — чётные"),
        ("rho_mp_pp_initial_rev", "то же: начало — чётные месяцы, прирост — нечётные"),
        ("rho_mp_pp_initial_naive", "то же без разделения месяцев (смещено регрессией к среднему)"),
        ("mp_split_half", "ρ прироста за январь–июнь и за июль–декабрь"),
        ("mp_iqr_2024", f"Межквартильный размах доли, п. п.: {Y0} → {Y1}"),
        ("mp_sd_log_2024", f"SD ln доли: {Y0} → {Y1}"),
        ("mp_rub_growth_median", "Рост трат на маркетплейсах в рублях, медиана"),
        ("rho_mp_rub_growth_level", "ρ роста в рублях с уровнем трат"),
        ("rho_mp_rub_incr_level", "ρ прибавки в рублях с уровнем трат"),
        ("mp_step_max_pp", "Тест ступеньки: наибольший скачок годового прироста, п. п."),
    ]
    out = []
    for key, label in rows:
        f = facts[f"{SECTION_ID}.{key}"]
        text = f.text
        if key == "mp_iqr_2024":
            text = f"{facts[f'{SECTION_ID}.mp_iqr_2023'].text} → {f.text}"
        elif key == "mp_sd_log_2024":
            text = f"{facts[f'{SECTION_ID}.mp_sd_log_2023'].text} → {f.text}"
        elif key == "mp_step_max_pp":
            flag = (
                "проверить границы категорий"
                if not (f.value is not None and f.value <= step_pp)
                else "скачка нет"
            )
            text = f"{f.text} ({mpc['mp_step_month']}): {flag}"
        elif key == "mp_pp_median":
            q25, q75 = facts[f"{SECTION_ID}.mp_pp_q25"].text, facts[f"{SECTION_ID}.mp_pp_q75"].text
            text = f"{f.text} (половина МО — от {q25} до {q75})"
        out.append(
            {"label": label, "text": nbsp(text), "n": n, "key": f"{SECTION_ID}.{key}", "value": f.value}
        )
    return pd.DataFrame(out)
