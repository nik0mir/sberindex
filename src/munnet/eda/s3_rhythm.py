"""Раздел E3 разведки: годовой ритм трат (spec_final, Б.4 E3; графики F06–F08, таблицы T05–T07).

Одно определение сезонности (Б.1, п. 4) — только МО с полным рядом (24 месяца). Для МО i и месяца
t = 0…23, y = ln траты (все категории или одна категория):

1. тренд МО — МНК ``y_it = a_i + b_i·t``, остаток ``d_it``;
2. общий ритм ``g_t`` — медиана ``d_it`` по МО, свой для каждого из 24 календарных месяцев;
3. свой ритм ``o_it = d_it − g_t``; повторяемость ``r_i`` — корреляция профилей 2023 и 2024 годов;
4. свой профиль ``s_i,m`` — среднее двух лет; размах ``A_i = max s − min s`` (лог-пункты);
5. надёжный свой ритм: ``r_i > eda.season.reliable_r`` и ``A_i > eda.season.reliable_amplitude``;
6. нуль: месяцы 2024 года переставлены внутри МО (``eda.season.null_repeats`` раз), размах — наблюдаемый;
7. доля общего ритма ``1 − Σ o² / Σ d²`` по всем МО.

Без снятия тренда МО повторяемость завышена: рост внутри года одинаково наклоняет оба годовых профиля.
STL и X-13 на двух циклах ненадёжны и не используются.

Размах своего профиля у типичного МО близок к размаху шума и больше у малых МО, поэтому критерий
надёжности держится на ``r``; в сводку (``Finding.indicators``) идёт размах, сжатый на повторяемость:
``A_i · max(r_i, 0)``. Проверка устойчивости: тренд МО со сдвигом уровня на рубеже 2023 и 2024 годов —
такой сдвиг после снятия линейного тренда даёт в оба года одинаковую «пилу» с пиком в январе.

Траты в наборе — средние безналичные расходы жителей МО: траты приезжих в курортном МО не видны, поэтому
курортный тип по этим данным не выделяется; связь летнего избытка с туризмом смотрим по ночёвкам Росстата.

Север — МО, у которых точка внутри полигона (``point_lat``, решение spec_final, часть Е) севернее
``eda.north_lat``, без внутригородских территорий (``is_inner_city``) — так же, как в разделах E2 и E5.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib.ticker import FixedLocator, FuncFormatter

from munnet import maps, style
from munnet.config import Config
from munnet.contracts import CATEGORY_CODES, N_MONTHS, YEARS
from munnet.eda import stats
from munnet.eda.base import Finding, SectionContext

if TYPE_CHECKING:
    from munnet.eda.data import EdaData

log = logging.getLogger(__name__)

SECTION_ID = "e3"
TITLE = "Годовой ритм"

MONTHS_IN_YEAR = 12
JANUARY, AUGUST, DECEMBER = 1, 8, 12
# Месяцы «пилы» от сдвига уровня на рубеже лет: пик своего профиля в январе–марте (проверка устойчивости).
WINTER_MONTHS: tuple[int, ...] = (1, 2, 3)
FULL_STATUS = "full"
# Средний размах 12 независимых нормальных величин в единицах их SD (константа d2 контрольных карт, n = 12):
# размах своего профиля из чистого шума ≈ D2_12 · σ / √2, где σ — шум одного года, а профиль — среднее двух.
D2_12 = 3.258
# Квантили корреляций пар МО в T06 (ключи фактов pair_*_q50 и pair_*_q90).
PAIR_QUANTILES: tuple[float, ...] = (0.5, 0.9)
# Категории, без которых нет обязательных фактов (nat_*, mp_dec, cafe_aug, cafe_summer_*).
REQUIRED_CATEGORIES: tuple[str, ...] = ("all", "marketplace", "cafe")
SERIES_KINDS: tuple[str, ...] = ("raw", "detrended", "own")
NB = style.NBSP

# Параметры раздела, которых ещё нет в ``eda.season`` конфига: значения из spec_final (Б.4 E3, В.5).
# Конфиг имеет приоритет: ключ ``eda.season.<имя>`` заменяет значение по умолчанию.
SEASON_DEFAULTS: dict[str, Any] = {
    "top_n": 15,  # T05: надёжных МО с наибольшим размахом
    "examples": 4,  # регионов и МО-примеров в строковых фактах для текста
    "profile_categories": ["all", "food", "marketplace", "transport", "health", "cafe"],  # F06: 6 панелей
    "noise_categories": ["all", "cafe", "transport"],  # T07
    "noise_groups": 10,  # T07: децили населения
    "hint_z": 2.0,  # подсказка к МО: признаки контекста с |робастный z| > 2 (Б.1, п. 8)
    "hist_bin": 0.05,  # F07: ширина столбца гистограммы r
    "map_step": 0.05,  # F08: шаг классов летнего избытка своего профиля (0,05 ≈ 5%)
    "map_classes_side": 4,  # F08: не больше 4 классов по каждую сторону нуля (соседние цвета различимы)
}

# Признаки контекста для подсказки в T05: колонка ``EdaData.mo`` -> короткая подпись.
HINT_FEATURES: dict[str, str] = {
    "point_lat": "широта",
    "nights_pc_2023": "ночёвки на жителя",
    "emp_sh_B_2023": "доля добычи",
    "emp_sh_A_2023": "доля сельского хозяйства",
    "urban_share_2023": "доля горожан",
    "log_density_2023": "плотность",
    "age_old_share_2023": "доля пожилых",
}
# Колонки ``EdaData.mo`` в таблице своего ритма (T05, данные F08).
CONTEXT_COLUMNS: tuple[str, ...] = ("point_lat", "nights_pc_2023", "emp_sh_B_2023", "emp_sh_A_2023")
# ρ Спирмена летнего избытка с контекстом (Б.4 E3): ключ факта rho_summer_<ключ> -> колонка ``EdaData.mo``.
RHO_CONTEXT: dict[str, str] = {
    "lat": "point_lat",
    "nights": "nights_pc_2023",
    "mining": "emp_sh_B_2023",
    "agri": "emp_sh_A_2023",
}

# Проверки заголовков (В.5): при каком соотношении чисел вывод заголовка верен.
PEAK_DEC_MIN = 0.9  # F06: «декабрь — пик» — у ≥ 90% МО
RELIABLE_VS_NULL = 3.0  # F07: надёжных МО не меньше чем втрое больше, чем на перемешанных месяцах
NORTH_RATIO_MIN = 2.0  # F08: «чаще на Севере» — доля МО с устойчивым ритмом на Севере вдвое выше прочих
# Текст: «северный слой почти не зависит от сдвига уровня» — со сдвигом на Севере остаётся ≥ 80% прежней доли.
STEP_KEEP_MIN = 0.8
# Сила связи в словах текста (ρ Спирмена по модулю): < 0,1 — почти нулевая (как незначимые клетки F14,
# Б.4 E5), < 0,3 — слабая, < 0,5 — умеренная, иначе сильная.
WEAK_RHO = 0.1
MODERATE_RHO = 0.3
STRONG_RHO = 0.5

# Долготы западной и восточной точек России (Калининград, Чукотка через 180°) — линия параллели на F08.
RUSSIA_LON_RANGE = (19.0, 191.0)
PARALLEL_STEP_DEG = 0.5
BAND_ALPHA = 0.18  # F06: прозрачность межквартильной полосы
Y_HEADROOM = 1.3  # F07: запас над самым высоким столбцом для подписей
R_TICK = 0.2  # F07: шаг делений оси r
LABEL_GAP_PT = 5  # F06: отступ подписи декабря от линии и полосы, pt
LABEL_SPAN_MONTHS = 2  # F06: подпись декабря по ширине занимает ≈ 2 месяца левее конца линии
LABEL_SPAN_LONG = 7  # F06: подпись «дек +53%, над трендом +24%» — ≈ 7 месяцев
PROFILE_PAD = 0.08  # F06: запас общей шкалы сверху и снизу для подписей, доля размаха
MAP_MIN_INTENSITY = 0.4  # F08: ближний к нулю класс — не белый (отличим от серых МО без ритма)
HINT_Z_CAP = 10.0  # подсказка: |z| больше 10 пишется как «> +10» (у признаков с почти нулевым MAD)

CATEGORY_TEXT = {
    "all": "все категории",
    "food": "продовольствие",
    "marketplace": "маркетплейсы",
    "transport": "транспорт",
    "health": "здоровье",
    "cafe": "общепит",
    "other": "прочее",
}
# Родительный падеж: «декабрь выше во всех категориях, кроме общепита» (альт-текст F06).
CATEGORY_GENITIVE = {
    "all": "трат в целом",
    "food": "продовольствия",
    "marketplace": "маркетплейсов",
    "transport": "транспорта",
    "health": "здоровья",
    "cafe": "общепита",
    "other": "прочего",
}
SERIES_LABELS = {"raw": "Логарифм трат без обработки", "detrended": "Без тренда МО", "own": "Свой ритм"}
NULL_TEXT = "перемешанные месяцы 2024 года"  # одно название нуля во всех подписях для людей
MONTHS_FULL = [
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
]
MONTHS_PREPOSITIONAL = [
    "январе",
    "феврале",
    "марте",
    "апреле",
    "мае",
    "июне",
    "июле",
    "августе",
    "сентябре",
    "октябре",
    "ноябре",
    "декабре",
]


# --- Параметры -----------------------------------------------------------------------------------


@dataclass(frozen=True)
class SeasonParams:
    """Параметры раздела из конфига (``eda.season``, ``eda.pair_sample``, ``eda.north_lat``…)."""

    reliable_r: float
    reliable_amplitude: float
    null_repeats: int
    pair_sample: int
    north_lat: float
    summer_months: tuple[int, ...]
    reference_year: int
    top_n: int
    examples: int
    profile_categories: tuple[str, ...]
    noise_categories: tuple[str, ...]
    noise_groups: int
    hint_z: float
    hist_bin: float
    map_step: float
    map_classes_side: int
    s1_reliable_min: float

    @classmethod
    def from_config(cls, cfg: Config) -> SeasonParams:
        """Собирает параметры: ``eda.season`` поверх ``SEASON_DEFAULTS``; ошибки конфига — ``ValueError``."""
        eda = cfg["eda"]
        season = {**SEASON_DEFAULTS, **eda["season"]}
        params = cls(
            reliable_r=float(season["reliable_r"]),
            reliable_amplitude=float(season["reliable_amplitude"]),
            null_repeats=int(season["null_repeats"]),
            pair_sample=int(eda["pair_sample"]),
            north_lat=float(eda["north_lat"]),
            summer_months=tuple(int(m) for m in eda["summer_months"]),
            reference_year=int(eda["reference_year"]),
            top_n=int(season["top_n"]),
            examples=int(season["examples"]),
            profile_categories=tuple(season["profile_categories"]),
            noise_categories=tuple(season["noise_categories"]),
            noise_groups=int(season["noise_groups"]),
            hint_z=float(season["hint_z"]),
            hist_bin=float(season["hist_bin"]),
            map_step=float(season["map_step"]),
            map_classes_side=int(season["map_classes_side"]),
            s1_reliable_min=float(eda["rejection"]["s1_reliable_share_min"]),
        )
        problems = []
        unknown = sorted(set(params.profile_categories + params.noise_categories) - set(CATEGORY_CODES))
        if unknown:
            problems.append(f"неизвестные категории {unknown}; допустимы {list(CATEGORY_CODES)}")
        if not params.summer_months or any(not 1 <= m < DECEMBER for m in params.summer_months):
            problems.append(f"eda.summer_months {list(params.summer_months)}: месяцы 1…11")
        if params.null_repeats < 1 or params.pair_sample < 1 or params.noise_groups < 2 or params.top_n < 1:
            problems.append("null_repeats, pair_sample, top_n — не меньше 1, noise_groups — не меньше 2")
        if not 0 < params.hist_bin < 1:
            problems.append(f"hist_bin = {params.hist_bin}: ширина столбца в (0; 1)")
        if not 0 < params.map_step < 1 or params.map_classes_side < 1:
            problems.append("map_step — в (0; 1), map_classes_side — не меньше 1")
        if problems:
            raise ValueError("eda.season: " + "; ".join(problems))
        return params


# --- Матрицы рядов -------------------------------------------------------------------------------


def full_ids(mo: pd.DataFrame) -> pd.Index:
    """``territory_id`` МО с полным рядом (``series_status == full``), по возрастанию."""
    ids = mo.loc[mo["series_status"].astype(str) == FULL_STATUS, "territory_id"]
    return pd.Index(np.sort(ids.to_numpy()), name="territory_id")


def series_matrix(panel_wide: pd.DataFrame, ids: Sequence[int], category: str = "all") -> pd.DataFrame:
    """ln трат категории: строки — МО ``ids``, колонки — ``t`` 0…23 по порядку.

    У МО из ``ids`` должны быть все 24 месяца; пропуск — ``ValueError`` (модель E3 — только полные ряды).
    """
    col = f"v_{category}"
    part = panel_wide.loc[panel_wide["territory_id"].isin(list(ids)), ["territory_id", "t", col]]
    wide = part.pivot(index="territory_id", columns="t", values=col).astype("float64")
    wide = wide.reindex(index=pd.Index(ids, name="territory_id"), columns=range(N_MONTHS))
    if wide.isna().any().any():
        bad = wide.index[wide.isna().any(axis=1)].tolist()
        raise ValueError(f"series_matrix: у МО {bad[:5]} не все {N_MONTHS} месяцев")
    if (wide <= 0).any().any():
        raise ValueError(f"series_matrix: траты {category} должны быть больше нуля")
    wide.columns.name = "t"
    return np.log(wide)


def _frame(X: Any) -> pd.DataFrame:
    frame = X if isinstance(X, pd.DataFrame) else pd.DataFrame(np.asarray(X, dtype="float64"))
    return frame.astype("float64")


def _year_blocks(X: Any, months: int = MONTHS_IN_YEAR) -> np.ndarray:
    """Матрица n × (k·12) -> массив n × k × 12 (годовые блоки)."""
    v = _frame(X).to_numpy()
    if v.shape[1] % months:
        raise ValueError(f"число месяцев {v.shape[1]} не кратно {months}")
    return v.reshape(v.shape[0], v.shape[1] // months, months)


def _row_corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Корреляция Пирсона строк ``a`` и ``b`` попарно; у строки без разброса — NaN."""
    a = a - a.mean(axis=-1, keepdims=True)
    b = b - b.mean(axis=-1, keepdims=True)
    den = np.sqrt((a * a).sum(axis=-1) * (b * b).sum(axis=-1))
    safe = np.where(den > 0, den, 1.0)
    return np.where(den > 0, (a * b).sum(axis=-1) / safe, np.nan)


# --- Модель ритма (чистые функции) ---------------------------------------------------------------


def detrend(Y: Any, step_at: int | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Снимает линейный тренд каждого МО: МНК ``y_it = a_i + b_i·t`` по всем месяцам строки.

    ``Y`` — матрица «МО × месяц» (колонки по порядку времени, t = 0, 1, …). Возвращает остатки ``D``
    (та же форма, в каждой строке ортогональны константе и t) и наклоны ``b_i`` (лог-пункты в месяц).
    ``step_at`` — проверка устойчивости: в тренд добавляется сдвиг уровня ``c_i·[t ≥ step_at]`` (например,
    12 — рубеж 2023 и 2024 годов). Пропуски недопустимы.
    """
    frame = _frame(Y)
    v = frame.to_numpy()
    if np.isnan(v).any():
        raise ValueError("detrend: в рядах есть пропуски")
    t = np.arange(v.shape[1], dtype="float64")
    columns = [np.ones_like(t), t]
    if step_at is not None:
        if not 0 < int(step_at) < v.shape[1]:
            raise ValueError(f"detrend: step_at = {step_at} вне 1…{v.shape[1] - 1}")
        columns.append((t >= int(step_at)).astype("float64"))
    X = np.column_stack(columns)
    beta, *_ = np.linalg.lstsq(X, v.T, rcond=None)
    D = v - (X @ beta).T
    return (
        pd.DataFrame(D, index=frame.index, columns=frame.columns),
        pd.Series(beta[1], index=frame.index, name="slope"),
    )


def own_rhythm(D: Any) -> tuple[pd.DataFrame, pd.Series]:
    """Свой ритм: из остатка после тренда вычитается общий ритм ``g_t`` — медиана по МО каждого месяца.

    Возвращает (``O``, ``g``): у ``O`` медиана каждого столбца равна нулю.
    """
    frame = _frame(D)
    g = frame.median(axis=0).rename("common")
    return frame - g, g


def reproducibility(own: Any) -> pd.Series:
    """Повторяемость своего ритма ``r_i``: корреляция профиля 2023 года с профилем 2024 года.

    ``own`` — свой ритм ``O``, 24 месяца (два годовых блока). У ряда без разброса в каком-то году — NaN.
    """
    frame = _frame(own)
    blocks = _year_blocks(frame)
    if blocks.shape[1] != len(YEARS):
        raise ValueError(f"reproducibility: нужно {len(YEARS)} года, получено {blocks.shape[1]}")
    return pd.Series(_row_corr(blocks[:, 0], blocks[:, 1]), index=frame.index, name="r")


def own_profile(own: Any) -> pd.DataFrame:
    """Свой профиль ``s_i,m`` — среднее годовых блоков своего ритма ``O`` (или любой матрицы «МО × месяц»);
    колонки — месяцы 1…12."""
    frame = _frame(own)
    prof = _year_blocks(frame).mean(axis=1)
    return pd.DataFrame(prof, index=frame.index, columns=pd.RangeIndex(1, MONTHS_IN_YEAR + 1, name="month"))


def amplitude(own: Any) -> pd.Series:
    """Размах своего профиля ``A_i = max_m s − min_m s`` (лог-пункты: 0,05 ≈ 5%)."""
    prof = own_profile(own)
    return (prof.max(axis=1) - prof.min(axis=1)).rename("amplitude")


def noise_amplitude(own: Any) -> pd.Series:
    """Размах своего профиля, который дал бы один шум МО (лог-пункты): ``D2_12 · σ_i / √2``.

    ``σ_i`` — шум одного года по неповторяющейся части своего ритма: SD по месяцам ``(o_2023 − o_2024) / √2``
    (повторяющийся профиль в разности сокращается). Профиль — среднее двух лет, поэтому его шум — σ / √2,
    а средний размах 12 таких значений — ``D2_12`` их SD. Если ритм от года к году меняется, σ завышена.
    """
    frame = _frame(own)
    blocks = _year_blocks(frame)
    if blocks.shape[1] != len(YEARS):
        raise ValueError(f"noise_amplitude: нужно {len(YEARS)} года, получено {blocks.shape[1]}")
    sigma = ((blocks[:, 0] - blocks[:, 1]) / np.sqrt(2)).std(axis=1, ddof=1)
    return pd.Series(D2_12 * sigma / np.sqrt(2), index=frame.index, name="noise_amplitude")


def shrunk_amplitude(A: Any, r: Any) -> pd.Series:
    """Размах, сжатый на повторяемость: ``A_i · max(r_i, 0)``; пропуск ``r`` — пропуск.

    Показатель МО для сводки: у МО без повторяющегося профиля (r ≤ 0) — ноль, поэтому размах шума малых МО
    не выдаётся за сильный ритм.
    """
    a = pd.Series(A, dtype="float64")
    rr = pd.Series(r, dtype="float64").reindex(a.index)
    return (a * rr.clip(lower=0)).rename("shrunk_amplitude")


def common_share(D: Any, own: Any) -> float:
    """Доля общего ритма ``1 − Σ o² / Σ d²`` по всем МО и месяцам (после снятия тренда МО)."""
    d, o = _frame(D).to_numpy(), _frame(own).to_numpy()
    return float(1.0 - (o**2).sum() / (d**2).sum())


def common_share_by_mo(D: Any, own: Any) -> pd.Series:
    """Доля общего ритма в каждом МО: ``1 − Σ_t o² / Σ_t d²``."""
    d, o = _frame(D), _frame(own)
    ss_d, ss_o = (d.to_numpy() ** 2).sum(axis=1), (o.to_numpy() ** 2).sum(axis=1)
    share = 1.0 - np.divide(ss_o, ss_d, out=np.full_like(ss_d, np.nan), where=ss_d > 0)
    return pd.Series(share, index=d.index, name="common_share")


def is_reliable(r: Any, A: Any, r_thr: float, a_thr: float) -> np.ndarray:
    """Надёжный свой ритм: ``r > r_thr`` и ``A > a_thr``; пропуск — не надёжен.

    Формы согласуются по правилам numpy: ``r`` может быть матрицей «повтор × МО», ``A`` — строкой МО.
    """
    r_v, a_v = np.asarray(r, dtype="float64"), np.asarray(A, dtype="float64")
    with np.errstate(invalid="ignore"):
        return (r_v > r_thr) & (a_v > a_thr)


def null_reproducibility(own: Any, rng: np.random.Generator, repeats: int) -> np.ndarray:
    """Нулевое распределение ``r``: месяцы 2024 года переставлены внутри каждого МО, ``repeats`` раз.

    Возвращает матрицу «повтор × МО». Перестановка разрушает соответствие месяцев двух лет и сохраняет
    распределение значений каждого года.
    """
    blocks = _year_blocks(own)
    first, second = blocks[:, 0], blocks[:, 1]
    out = np.empty((int(repeats), blocks.shape[0]))
    for k in range(int(repeats)):
        out[k] = _row_corr(first, rng.permuted(second, axis=1))
    return out


def null_reliable_share(
    own: Any, A: Any, rng: np.random.Generator, repeats: int, r_thr: float, a_thr: float
) -> float:
    """Доля «надёжных» МО на нуле: ``r`` — по перестановке месяцев 2024 года, размах ``A`` — наблюдаемый.

    Среднее по ``repeats`` перестановкам; сравнивается с наблюдаемой долей надёжных МО.
    """
    null_r = null_reproducibility(own, rng, repeats)
    return float(is_reliable(null_r, np.asarray(A, dtype="float64")[None, :], r_thr, a_thr).mean())


def profile_summer_excess(profile: pd.DataFrame, summer_months: Sequence[int]) -> pd.Series:
    """Летний избыток профиля: среднее летних месяцев минус среднее остальных без декабря (лог-пункты)."""
    summer = set(int(m) for m in summer_months)
    in_summer = [m for m in profile.columns if m in summer]
    rest = [m for m in profile.columns if m not in summer and m != DECEMBER]
    return (profile[in_summer].mean(axis=1) - profile[rest].mean(axis=1)).rename("summer")


def seasonal_excess(Y: Any, summer_months: Sequence[int]) -> pd.Series:
    """Летний избыток трат МО по формуле ``EdaData.mo.summer_excess`` (Б.2) для любого ряда ln трат.

    Для каждого года — среднее ln за летние месяцы минус среднее ln за остальные месяцы без декабря;
    среднее двух лет; exp − 1. Для ln ``v_all`` без тренда МО (``detrend``, остатки ``D`` модели) совпадает
    с ``mo.summer_excess``. Разности линейны, поэтому среднее по годам равно разности на профиле,
    усреднённом по годам (``own_profile``).
    """
    diff = profile_summer_excess(own_profile(Y), summer_months)
    return np.expm1(diff).rename("summer_excess")


def summer_excess_by_year(Y: Any, summer_months: Sequence[int]) -> pd.DataFrame:
    """Летний избыток каждого года отдельно (та же формула, что ``seasonal_excess``, без усреднения по годам).

    Колонки — годы ``YEARS``; нужен, чтобы проверить, держится ли связь с контекстом в оба года.
    """
    frame = _frame(Y)
    blocks = _year_blocks(frame)
    months = pd.RangeIndex(1, MONTHS_IN_YEAR + 1, name="month")
    out = {
        year: np.expm1(
            profile_summer_excess(
                pd.DataFrame(blocks[:, k], index=frame.index, columns=months), summer_months
            )
        )
        for k, year in enumerate(YEARS)
    }
    return pd.DataFrame(out, index=frame.index)


def common_month_excess(g: Any, month: int) -> float:
    """Общий ритм календарного месяца над трендом МО: exp(среднего ``g_t`` этого месяца по годам) − 1.

    ``g`` — общий ритм модели (медиана остатков после тренда МО, 24 месяца). В отличие от профиля F06
    («месяц / среднее своего года») здесь рост трат внутри года уже снят.
    """
    v = np.asarray(g, dtype="float64").reshape(-1, MONTHS_IN_YEAR)
    return float(np.expm1(v[:, int(month) - 1].mean()))


def peak_shares(peak_month: pd.Series, months: Sequence[int], mask: pd.Series) -> tuple[float, float]:
    """Доля МО с пиком своего профиля в ``months`` при ``mask`` и при ``~mask`` (пустая группа — NaN)."""
    in_months = pd.Series(peak_month).astype(int).isin([int(m) for m in months])
    m = pd.Series(mask).reindex(in_months.index).fillna(False).astype(bool)
    part_in, part_out = in_months[m], in_months[~m]
    return (
        float(part_in.mean()) if len(part_in) else float("nan"),
        float(part_out.mean()) if len(part_out) else float("nan"),
    )


@dataclass(frozen=True)
class RhythmModel:
    """Модель E3 для одной категории: ряды, остатки, общий и свой ритм, повторяемость, профиль, размах."""

    Y: pd.DataFrame
    D: pd.DataFrame
    slopes: pd.Series
    O: pd.DataFrame  # noqa: E741 — обозначение из спецификации
    g: pd.Series
    r: pd.Series
    profile: pd.DataFrame
    A: pd.Series


def fit_rhythm(Y: Any, step_at: int | None = None) -> RhythmModel:
    """Модель E3 целиком: тренд МО → общий ритм → свой ритм → повторяемость, профиль, размах.

    ``step_at`` — тренд МО со сдвигом уровня (проверка устойчивости, ``detrend``); по умолчанию — без него.
    """
    D, slopes = detrend(Y, step_at)
    O, g = own_rhythm(D)  # noqa: E741
    prof = own_profile(O)
    return RhythmModel(
        Y=_frame(Y),
        D=D,
        slopes=slopes,
        O=O,
        g=g,
        r=reproducibility(O),
        profile=prof,
        A=(prof.max(axis=1) - prof.min(axis=1)).rename("amplitude"),
    )


# --- Общий ритм страны (F06) ---------------------------------------------------------------------


def _complete_ids(panel_wide: pd.DataFrame) -> pd.Index:
    counts = panel_wide.groupby("territory_id").size()
    return pd.Index(np.sort(counts.index[counts == N_MONTHS].to_numpy()), name="territory_id")


def _ratio_blocks(panel_wide: pd.DataFrame, ids: pd.Index, category: str) -> np.ndarray:
    """Траты категории к среднему своего года: массив МО × год × месяц."""
    col = f"v_{category}"
    part = panel_wide.loc[panel_wide["territory_id"].isin(ids), ["territory_id", "t", col]]
    v = part.pivot(index="territory_id", columns="t", values=col).reindex(index=ids, columns=range(N_MONTHS))
    blocks = _year_blocks(v.astype("float64"))
    return blocks / blocks.mean(axis=2, keepdims=True)


def national_profile(
    panel_wide: pd.DataFrame, categories: Sequence[str] = tuple(SEASON_DEFAULTS["profile_categories"])
) -> pd.DataFrame:
    """Общий годовой ритм страны: медиана по МО отношения «месяц / среднее своего года» и квартили.

    МО — только с полными 24 месяцами; отношение сначала усредняется по двум годам в каждом МО, затем
    берутся медиана, 25-й и 75-й перцентили по МО. Колонки: ``category``, ``month`` 1…12, ``median``,
    ``q25``, ``q75``, ``n_mo``.
    """
    ids = _complete_ids(panel_wide)
    if ids.empty:
        raise ValueError("national_profile: нет МО с полным рядом")
    rows = []
    for c in categories:
        avg = _ratio_blocks(panel_wide, ids, c).mean(axis=1)
        q25, med, q75 = np.quantile(avg, [0.25, 0.5, 0.75], axis=0)
        rows.extend((c, m + 1, med[m], q25[m], q75[m], len(ids)) for m in range(MONTHS_IN_YEAR))
    return pd.DataFrame(rows, columns=["category", "month", "median", "q25", "q75", "n_mo"])


def month_ratio_by_year(panel_wide: pd.DataFrame, month: int, category: str = "all") -> pd.Series:
    """Медиана по МО (полные ряды) отношения «месяц / среднее своего года» − 1, отдельно по годам."""
    ids = _complete_ids(panel_wide)
    med = np.median(_ratio_blocks(panel_wide, ids, category), axis=0)
    return pd.Series(med[:, int(month) - 1] - 1, index=pd.Index(YEARS, name="year"), name="ratio")


def annual_growth_median(panel_wide: pd.DataFrame, category: str = "all") -> float:
    """Медиана по МО (полные ряды) номинального роста годовой суммы трат категории: Σ 2024 / Σ 2023 − 1.

    Нужна только для пояснения к ``mp_dec``: ориентир «+54%» spec_final — это рост маркетплейсов за год,
    а не декабрь к среднему года.
    """
    ids = _complete_ids(panel_wide)
    col = f"v_{category}"
    part = panel_wide.loc[panel_wide["territory_id"].isin(ids), ["territory_id", "t", col]]
    v = part.pivot(index="territory_id", columns="t", values=col).reindex(index=ids, columns=range(N_MONTHS))
    sums = _year_blocks(v.astype("float64")).sum(axis=2)
    return float(np.median(sums[:, -1] / sums[:, 0]) - 1)


def peak_month_share(panel_wide: pd.DataFrame, month: int = DECEMBER, category: str = "all") -> pd.Series:
    """Доля МО (полные ряды), у которых пик трат года приходится на ``month``, по годам."""
    ids = _complete_ids(panel_wide)
    peak = _ratio_blocks(panel_wide, ids, category).argmax(axis=2) + 1
    return pd.Series((peak == month).mean(axis=0), index=pd.Index(YEARS, name="year"), name="peak_share")


# --- Сигнал для рёбер (T06) ----------------------------------------------------------------------


def _row_z(v: np.ndarray) -> np.ndarray:
    v = v - v.mean(axis=1, keepdims=True)
    sd = v.std(axis=1, keepdims=True)
    return np.divide(v, sd, out=np.full_like(v, np.nan), where=sd > 0)


def pair_correlations(X: Mapping[str, Any], rng: np.random.Generator, n_pairs: int) -> pd.DataFrame:
    """Корреляции рядов пар МО для нескольких видов рядов и те же корреляции на нуле.

    ``X`` — словарь «вид ряда -> матрица МО × месяц» с одинаковыми строками (например, сырые ln-ряды,
    ряды без тренда, свой ритм). Пары: все, если их не больше ``n_pairs``, иначе ``n_pairs`` случайных пар
    разных МО (одни и те же для всех видов). Нуль — месяцы переставлены внутри каждого МО (одна
    перестановка на вид ряда). Колонки: ``territory_a``, ``territory_b``, ``<вид>``, ``<вид>_null``.
    """
    frames = {name: _frame(m) for name, m in X.items()}
    if not frames:
        raise ValueError("pair_correlations: нет рядов")
    index = next(iter(frames.values())).index
    for name, f in frames.items():
        if not f.index.equals(index):
            raise ValueError(f"pair_correlations: строки ряда {name} не совпадают с остальными")
    n = len(index)
    if n < 2:
        raise ValueError("pair_correlations: нужно хотя бы два МО")
    if n_pairs >= n * (n - 1) // 2:
        a, b = np.triu_indices(n, k=1)
    else:
        a = rng.integers(0, n, size=int(n_pairs))
        b = (a + rng.integers(1, n, size=int(n_pairs))) % n
    ids = index.to_numpy()
    out = pd.DataFrame({"territory_a": ids[a], "territory_b": ids[b]})
    for name, f in frames.items():
        z = _row_z(f.to_numpy())
        out[name] = (z[a] * z[b]).mean(axis=1)
        zn = _row_z(rng.permuted(f.to_numpy(), axis=1))
        out[f"{name}_null"] = (zn[a] * zn[b]).mean(axis=1)
    return out


def pair_summary(pairs: pd.DataFrame, kinds: Sequence[str]) -> pd.DataFrame:
    """Квантили ``PAIR_QUANTILES`` корреляций пар по видам рядов и на нуле (строка на вид ряда)."""
    rows = []
    for kind in kinds:
        row: dict[str, Any] = {"series": kind, "label": SERIES_LABELS.get(kind, kind), "n_pairs": len(pairs)}
        for q in PAIR_QUANTILES:
            tag = f"q{round(100 * q)}"
            row[tag] = float(np.nanquantile(pairs[kind], q))
            row[f"null_{tag}"] = float(np.nanquantile(pairs[f"{kind}_null"], q))
        rows.append(row)
    return pd.DataFrame(rows)


# --- Шум малых МО (T07) --------------------------------------------------------------------------


def robust_sd_rows(own: Any) -> pd.Series:
    """Робастное SD каждой строки: 1,4826 × MAD значений ряда (лог-пункты)."""
    frame = _frame(own)
    v = frame.to_numpy()
    med = np.median(v, axis=1, keepdims=True)
    return pd.Series(stats.MAD_SCALE * np.median(np.abs(v - med), axis=1), index=frame.index, name="sd")


def size_groups(pop: pd.Series, n_groups: int) -> pd.Series:
    """Номер группы 1…``n_groups`` по населению (1 — самые малые МО, равные по числу МО); пропуск — NA."""
    pop = pd.Series(pop, dtype="float64")
    known = pop.dropna()
    if len(known) < n_groups:
        raise ValueError(f"size_groups: МО с населением {len(known)} меньше числа групп {n_groups}")
    groups = pd.qcut(known.rank(method="first"), n_groups, labels=False) + 1
    return groups.reindex(pop.index).astype("Int64").rename("group")


def noise_by_size(own: Mapping[str, Any], pop: pd.Series, n_groups: int) -> pd.DataFrame:
    """Шум своего ритма по группам населения: медиана по МО робастного SD ``o_it`` для каждой категории.

    ``own`` — словарь «категория -> O (МО × месяц)» с индексом ``territory_id``; ``pop`` — население с тем
    же индексом (МО без населения не входят). Строка на группу: ``group`` (1 — самые малые), ``n_mo``,
    ``pop_min``, ``pop_max``, ``sd_<категория>``.
    """
    sds = pd.DataFrame({c: robust_sd_rows(o) for c, o in own.items()})
    pop = pd.Series(pop, dtype="float64").reindex(sds.index)
    frame = sds.assign(group=size_groups(pop, n_groups), pop=pop).dropna(subset=["group"])
    g = frame.groupby("group")
    out = pd.DataFrame({"n_mo": g.size(), "pop_min": g["pop"].min(), "pop_max": g["pop"].max()})
    for c in own:
        out[f"sd_{c}"] = g[c].median()
    out = out.reset_index()
    out["group"] = out["group"].astype(int)
    out["n_mo"] = out["n_mo"].astype(int)
    return out


def noise_ratio(table: pd.DataFrame, category: str) -> float:
    """Во сколько раз шум своего ритма у самых малых МО больше, чем у самых крупных (группа 1 / последняя)."""
    first = float(table.loc[table["group"] == table["group"].min(), f"sd_{category}"].iloc[0])
    last = float(table.loc[table["group"] == table["group"].max(), f"sd_{category}"].iloc[0])
    return first / last if last > 0 else float("nan")


# --- Кто с надёжным ритмом (T05, факты) ----------------------------------------------------------

# Существительное к короткому названию-прилагательному (как в разделе E2): «Булунский» -> «Булунский район».
TYPE_NOUNS: dict[str, str] = {"mr": "район", "mo": "округ"}
ADJECTIVE_ENDINGS: tuple[str, ...] = ("ий", "ый", "ой")


def with_short_names(mo: pd.DataFrame, territories: pd.DataFrame | None) -> pd.DataFrame:
    """``mo`` с колонкой ``name_short`` из справочника территорий, если её в ``mo`` ещё нет."""
    if "name_short" in mo.columns or territories is None or "name_short" not in territories.columns:
        return mo
    names = territories.drop_duplicates("territory_id").set_index("territory_id")["name_short"]
    return mo.assign(name_short=mo["territory_id"].map(names))


def join_names(names: Sequence[str]) -> str:
    """Перечень через запятую с «и» перед последним: «A, B и C»; пустой — прочерк."""
    items = [str(n) for n in names]
    if not items:
        return style.NA_TEXT
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " и " + items[-1]


def display_names(frame: pd.DataFrame) -> pd.Series:
    """Название МО для таблиц и текста: ``name_short`` (пропуск — ``name``); к одному слову-прилагательному
    района или округа (``mo_type`` ``mr`` или ``mo``) добавляется «район» или «округ».

    «Булунский» (mr) -> «Булунский район», «Эгвекинот» (go) -> «Эгвекинот». Индекс — как у ``frame``.
    """
    full = frame["name"].astype("string")
    base = frame["name_short"].astype("string").fillna(full) if "name_short" in frame.columns else full
    types = frame["mo_type"].astype(str) if "mo_type" in frame.columns else pd.Series("", index=frame.index)
    out = []
    for name, kind in zip(base.fillna(style.NA_TEXT), types, strict=True):
        noun = TYPE_NOUNS.get(kind)
        single_adjective = " " not in name and name.endswith(ADJECTIVE_ENDINGS)
        out.append(f"{name} {noun}" if noun and single_adjective else str(name))
    return pd.Series(out, index=frame.index, dtype="object", name="name")


def _z_text(z: float) -> str:
    """Робастный z для подсказки: «+3,1»; больше ``HINT_Z_CAP`` по модулю — «> +10» или «< −10»."""
    if abs(z) > HINT_Z_CAP:
        return f"{'>' if z > 0 else '<'} {style.fmt_num(np.sign(z) * HINT_Z_CAP, 0, sign=True)}"
    return style.fmt_num(z, 1, sign=True)


def hints(
    mo: pd.DataFrame,
    ids: Sequence[int],
    features: Mapping[str, str],
    z_thr: float,
    annotations: Mapping[Any, str] | None = None,
) -> pd.Series:
    """Подсказка к МО: признаки контекста с |робастный z| > ``z_thr`` («широта +3,1») и пояснение из
    ``eda.annotations``. z считается по всем строкам ``mo``; пропуск признака не упоминается."""
    base = mo.set_index("territory_id")
    notes = {int(k): str(v) for k, v in (annotations or {}).items()}
    z = pd.DataFrame({c: stats.robust_z(base[c]) for c in features if c in base.columns}, index=base.index)
    out = {}
    for tid in ids:
        parts = [
            f"{label} {_z_text(z.at[tid, c])}"
            for c, label in features.items()
            if c in z.columns and pd.notna(z.at[tid, c]) and abs(z.at[tid, c]) > z_thr
        ]
        if int(tid) in notes:
            parts.append(notes[int(tid)])
        out[tid] = "; ".join(parts) if parts else style.NA_TEXT
    return pd.Series(out, name="hint", dtype="object")


def rhythm_table(
    model: RhythmModel, reliable: pd.Series, mo: pd.DataFrame, summer_months: Sequence[int]
) -> pd.DataFrame:
    """Свой ритм по МО модели: повторяемость, размах, пик и летний избыток своего профиля, надёжность и
    контекст (для T05, данных F08 и показателей сводки). Строка на МО, колонка ``territory_id``.

    ``name`` — короткое название для людей (``display_names``; нужны ``name`` и, если есть, ``name_short``
    и ``mo_type`` в ``mo``)."""
    base = mo.set_index("territory_id").reindex(model.r.index)
    out = pd.DataFrame(index=model.r.index)
    out["name"] = display_names(base)
    out["region_name"] = base["region_name"]
    out["amplitude_log"] = model.A
    out["peak_month"] = model.profile.idxmax(axis=1).astype(int)
    out["summer_own"] = np.expm1(profile_summer_excess(model.profile, summer_months))
    out["repro_r"] = model.r
    out["reliable"] = reliable.reindex(out.index).fillna(False).astype(bool)
    for c in CONTEXT_COLUMNS:
        out[c] = base[c].astype("float64") if c in base.columns else np.nan
    return out.reset_index()


def top_counts(values: pd.Series, n: int) -> str:
    """«Регион A — 27, регион B — 18…»: ``n`` самых частых значений с числом (при равенстве — по алфавиту)."""
    counts = pd.Series(values).astype(str).value_counts()
    counts = counts.sort_index().sort_values(ascending=False, kind="mergesort").head(n)
    return ", ".join(f"{name} — {cnt}" for name, cnt in counts.items()) if len(counts) else style.NA_TEXT


def group_median(values: pd.Series, mask: pd.Series) -> tuple[float, float]:
    """Медианы ``values`` при ``mask`` и при ``~mask`` (пропуски отброшены)."""
    v = pd.Series(values, dtype="float64")
    m = pd.Series(mask).reindex(v.index).fillna(False).astype(bool)
    return float(v[m].median()), float(v[~m].median())


def _share(flag: pd.Series, mask: pd.Series) -> float:
    """Доля ``flag`` среди МО с ``mask``; пустая группа — NaN."""
    f = pd.Series(flag).astype(bool)
    m = pd.Series(mask).reindex(f.index).fillna(False).astype(bool)
    return float(f[m].mean()) if m.any() else float("nan")


# --- Заголовки с проверкой (В.5) -----------------------------------------------------------------


@dataclass(frozen=True)
class Headline:
    """Заголовок-вывод графика, выполнена ли его проверка, формулировка проверки и текущие числа."""

    fid: str
    title: str
    ok: bool
    check: str
    detail: str


def headlines(v: Mapping[str, float], north_lat: float) -> dict[str, Headline]:
    """Заголовки F06–F08 из чисел раздела; если вывод перестал быть верным, ``ok`` = False.

    Нужны ключи ``peak_dec_share_<год>``, ``nat_dec``, ``reliable_share``, ``null_reliable_share``,
    ``reliable_share_north``, ``reliable_share_rest``; пропуск (NaN) — проверка не выполнена.
    """
    peaks = [v[f"peak_dec_share_{y}"] for y in YEARS]
    lo, hi, nat_dec = min(peaks), max(peaks), v["nat_dec"]
    rel, null = v["reliable_share"], v["null_reliable_share"]
    rel_north, rel_rest = v["reliable_share_north"], v["reliable_share_rest"]
    pct, nb = style.fmt_pct, style.NBSP
    peak_text = pct(lo, 0) if pct(lo, 0) == pct(hi, 0) else style.fmt_range(lo, hi, fmt=pct, decimals=0)
    peak_detail = ", ".join(f"peak_dec_share_{y} = {p:.3f}" for y, p in zip(YEARS, peaks, strict=True))
    return {
        "F06": Headline(
            "F06",
            f"Декабрь — пик трат у{nb}{peak_text} МО: в{nb}типичном МО {pct(nat_dec, 0, sign=True)} "
            "к среднему месяцу года",
            bool(lo >= PEAK_DEC_MIN and nat_dec > 0),
            f"peak_dec_share_<год> ≥ {style.fmt_num(PEAK_DEC_MIN, 1)} в оба года и nat_dec > 0",
            f"F06: {peak_detail}, nat_dec = {nat_dec:.3f}",
        ),
        # Один знак после запятой: при целых 1,49% читалось бы как «1%», а отношение к нулю — как 16 раз.
        "F07": Headline(
            "F07",
            f"Свой устойчивый годовой ритм — у{nb}{pct(rel, 1)} МО, на перемешанных месяцах — "
            f"у{nb}{pct(null, 1)}",
            bool(rel > 0 and rel >= RELIABLE_VS_NULL * null),
            f"reliable_share ≥ {style.fmt_num(RELIABLE_VS_NULL)} · null_reliable_share",
            f"F07: reliable_share = {rel:.4f}, null_reliable_share = {null:.4f}",
        ),
        "F08": Headline(
            "F08",
            f"Свой устойчивый ритм — у{nb}{pct(rel_north, 0)} МО севернее {style.fmt_num(north_lat)}-й "
            f"параллели и{nb}у{nb}{pct(rel_rest, 0)} остальных",
            bool(rel_north > 0 and rel_north >= NORTH_RATIO_MIN * rel_rest),
            f"reliable_share_north ≥ {style.fmt_num(NORTH_RATIO_MIN)} · reliable_share_rest",
            f"F08: reliable_share_north = {rel_north:.4f}, reliable_share_rest = {rel_rest:.4f}",
        ),
    }


# --- Графики -------------------------------------------------------------------------------------


def _pct_formatter(decimals: int = 0, sign: bool = True) -> FuncFormatter:
    return FuncFormatter(lambda v, pos=None: style.fmt_pct(v, decimals, sign=sign))


def december_label_y(part: pd.DataFrame, span: int = LABEL_SPAN_MONTHS) -> tuple[float, bool]:
    """Где подписать декабрь на панели F06: над полосой последних месяцев или под ней.

    ``part`` — профиль одной категории (``month``, ``median``, ``q25``, ``q75`` — отношения к среднему года).
    Если к декабрю линия растёт, подпись встаёт над верхним краем полосы месяцев ``12 − span … 12``, иначе —
    под нижним: так текст шириной в ``span`` месяцев не ложится на линию и полосу соседних месяцев.
    Возвращает (y как отклонение от среднего года, True — подпись сверху).
    """
    by = part.set_index("month")
    up = float(by.at[DECEMBER, "median"]) >= float(by.at[DECEMBER - 1, "median"])
    last = by.loc[by.index >= DECEMBER - span]
    return float(last["q75"].max() if up else last["q25"].min()) - 1, up


def december_label(dec: float, over_trend: float | None = None) -> str:
    """Подпись декабря на панели F06: «дек +21%»; с ``over_trend`` — «дек +53%, над трендом +24%»."""
    text = f"дек {style.fmt_pct(dec, 0, sign=True)}"
    if over_trend is not None and np.isfinite(over_trend):
        text += f", над трендом {style.fmt_pct(over_trend, 0, sign=True)}"
    return text


def plot_national_rhythm(
    profile: pd.DataFrame, categories: Sequence[str], over_trend: Mapping[str, float] | None = None
):
    """F06: малые множители с общей шкалой — медиана «месяц / среднее года» − 1 и межквартильная полоса.

    ``over_trend`` — категории, у которых к подписи декабря добавляется декабрь над трендом МО (там, где рост
    внутри года заметно завышает декабрь, — у маркетплейсов).
    """
    over_trend = dict(over_trend or {})
    ncols = (len(categories) + 1) // 2
    fig, axes = style.new_figure("tall", 2, ncols, sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    months = np.arange(1, MONTHS_IN_YEAR + 1)
    for ax, c in zip(axes, categories, strict=False):
        part = profile.loc[profile["category"] == c].sort_values("month")
        color = style.PALETTE[c]
        ax.fill_between(months, part["q25"] - 1, part["q75"] - 1, color=color, alpha=BAND_ALPHA, linewidth=0)
        ax.plot(months, part["median"] - 1, color=color)
        ax.axhline(0, color=style.TEXT2, linewidth=0.6)
        dec = float(part.set_index("month").at[DECEMBER, "median"]) - 1
        label = december_label(dec, over_trend.get(c))
        span = LABEL_SPAN_MONTHS if c not in over_trend else LABEL_SPAN_LONG
        y, up = december_label_y(part, span)
        ax.annotate(
            label,
            (DECEMBER, y),
            xytext=(0, LABEL_GAP_PT if up else -LABEL_GAP_PT),
            textcoords="offset points",
            ha="right",
            va="bottom" if up else "top",
            fontsize=style.POINT_LABEL_PT,
            color=style.TEXT,
        )
        ax.set_title(style.LABELS[c], fontsize=style.LABEL_PT, loc="left")
        style.month_axis(ax, MONTHS_IN_YEAR, step=3)
        ax.yaxis.set_major_formatter(_pct_formatter())
    lo, hi = axes[0].get_ylim()  # общая шкала: место для подписей над полосой и под ней
    pad = (hi - lo) * PROFILE_PAD
    axes[0].set_ylim(lo - pad, hi + pad)
    for ax in axes[len(categories) :]:
        ax.set_visible(False)
    return fig


def hist_table(r: pd.Series, null_r: np.ndarray, bin_width: float) -> pd.DataFrame:
    """Гистограммы F07: доля МО в каждом столбце ``r`` и та же доля на нуле (все перестановки вместе)."""
    edges = np.arange(-1.0, 1.0 + bin_width / 2, bin_width)
    obs = pd.Series(r, dtype="float64").dropna().to_numpy()
    nul = np.asarray(null_r, dtype="float64").ravel()
    nul = nul[~np.isnan(nul)]
    return pd.DataFrame(
        {
            "bin_left": edges[:-1],
            "bin_right": edges[1:],
            "share_mo": np.histogram(obs, bins=edges)[0] / max(len(obs), 1),
            "share_null": np.histogram(nul, bins=edges)[0] / max(len(nul), 1),
        }
    )


def plot_own_rhythm(
    hist: pd.DataFrame, r_thr: float, share_above: float, a_thr: float, share_reliable: float
):
    """F07: столбцы — доля МО по повторяемости r (выше порога — акцентом), контур — нулевое распределение.

    Подпись у порога называет обе доли всех МО: правее порога r (``share_above``) и правее порога при размахе
    больше ``a_thr`` (``share_reliable``, число заголовка), чтобы 16,8% у порога не спорили с 15,6%
    в заголовке.
    """
    fig, ax = style.new_figure("full")
    left, right = hist["bin_left"].to_numpy(), hist["bin_right"].to_numpy()
    edges = np.append(left, right[-1])
    width = right - left
    colors = [style.ACCENT if center > r_thr else style.CONTEXT for center in left + width / 2]
    ax.bar(left, hist["share_mo"], width=width, align="edge", color=colors, linewidth=0)
    ax.stairs(hist["share_null"].to_numpy(), edges, color=style.TEXT2, linewidth=1.4)
    ax.axvline(r_thr, color=style.TEXT, linewidth=0.8, linestyle="--")
    top = float(max(hist["share_mo"].max(), hist["share_null"].max(), 1e-9))
    ax.set_ylim(0, top * Y_HEADROOM)
    ax.set_xlim(edges[0], edges[-1])
    ax.text(
        r_thr + width[0] / 2,
        top * (Y_HEADROOM - 0.05),
        f"r > {style.fmt_num(r_thr, 1)}: {style.fmt_pct(share_above, 1)} МО\n"
        f"и размах > {style.fmt_pct(a_thr, 0)}: {style.fmt_pct(share_reliable, 1)}",
        fontsize=style.POINT_LABEL_PT,
        color=style.TEXT,
        ha="left",
        va="top",
    )
    peak = int(np.argmax(hist["share_null"].to_numpy()))
    ax.annotate(
        f"контур — {NULL_TEXT}",
        ((left[peak] + right[peak]) / 2, float(hist["share_null"].iloc[peak])),
        xytext=(-12, 14),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=style.POINT_LABEL_PT,
        color=style.TEXT2,
        arrowprops={"arrowstyle": "-", "color": style.TEXT2, "linewidth": 0.6},
    )
    ax.set_xlabel("Повторяемость своего профиля 2023 и 2024 годов, r")
    ax.set_ylabel("Доля МО")
    ax.xaxis.set_major_locator(FixedLocator(np.round(np.arange(-1.0, 1.0 + R_TICK / 2, R_TICK), 1)))
    ax.xaxis.set_major_formatter(style.ru_formatter("num", 1))
    ax.yaxis.set_major_formatter(_pct_formatter(0, sign=False))
    return fig


@dataclass(frozen=True)
class MapClasses:
    """Классы картограммы: подпись класса каждого МО и цвет каждой подписи в порядке легенды."""

    labels: pd.Series
    colors: dict[str, str]


def diverging_classes(values: pd.Series, step: float, side: int, cmap: str = style.DIV_CMAP) -> MapClasses:
    """Классы расходящейся шкалы с нулём на границе классов: ``[k·step; (k+1)·step)``.

    По каждую сторону нуля не больше ``side`` классов; крайний отрицательный класс открыт («ниже −5%»),
    крайний положительный — тоже («20% и выше»). Цвет зависит только от удалённости класса от нуля и
    одинаков по обе стороны: «от −5 до 0%» и «0–5%» одинаково бледны, но не белые (``MAP_MIN_INTENSITY``),
    самый дальний класс — край шкалы ``cmap``. Так −6% не выглядит так же ярко, как +28%, а легенда
    называет классы словами, а не делениями непрерывной шкалы. В легенде только непустые классы, по
    возрастанию.
    """
    v = pd.Series(values, dtype="float64").dropna()
    k = np.floor(np.round(v.to_numpy() / step, 9)).astype(int).clip(-side, side - 1)
    present = sorted(set(k.tolist()))
    if not present:
        return MapClasses(pd.Series(dtype="object"), {})
    dist = {c: (c + 1 if c >= 0 else -c) for c in present}
    far = max(dist.values())
    cm = mpl.colormaps[cmap]
    pct = style.fmt_pct
    labels, colors = {}, {}
    for c in present:
        lo, hi = c * step, (c + 1) * step
        if c < 0 and c == present[0]:
            text = f"ниже {pct(hi, 0)}"
        elif c > 0 and c == present[-1]:
            text = f"{pct(lo, 0)} и выше"
        else:
            text = style.fmt_range(lo, hi, fmt=pct, decimals=0)
        t = MAP_MIN_INTENSITY + (1 - MAP_MIN_INTENSITY) * (dist[c] - 1) / max(far - 1, 1)
        labels[c] = text
        colors[text] = mpl.colors.to_hex(cm(0.5 + 0.5 * t if c >= 0 else 0.5 - 0.5 * t))
    return MapClasses(pd.Series([labels[c] for c in k], index=v.index, dtype="object"), colors)


REGION_FORMS = ("регион", "региона", "регионов")


def plural(n: int, forms: tuple[str, str, str]) -> str:
    """Форма слова после числа: 1 регион, 2 региона, 5 регионов, 21 регион, 11 регионов."""
    n = abs(int(n))
    if n % 10 == 1 and n % 100 != 11:
        return forms[0]
    if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
        return forms[1]
    return forms[2]


def parallel_xy(lat: float, crs: str) -> tuple[np.ndarray, np.ndarray]:
    """Параллель ``lat`` поперёк России в проекции карты (x, y в метрах)."""
    from pyproj import Transformer

    lons = np.arange(RUSSIA_LON_RANGE[0], RUSSIA_LON_RANGE[1] + PARALLEL_STEP_DEG / 2, PARALLEL_STEP_DEG)
    to_map = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    x, y = to_map.transform(lons, np.full_like(lons, lat))
    return np.asarray(x), np.asarray(y)


def draw_parallel(ax, lat: float, crs: str) -> None:
    """Пунктир параллели с подписью «60° с. ш.» у восточного края видимой части; пределы осей не меняются."""
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    x, y = parallel_xy(lat, crs)
    ax.plot(x, y, color=style.TEXT2, linewidth=0.7, linestyle=(0, (4, 3)), zorder=5)
    inside = (x >= min(xlim)) & (x <= max(xlim)) & (y >= min(ylim)) & (y <= max(ylim))
    if inside.any():
        i = int(np.flatnonzero(inside)[-1])
        ax.annotate(
            f"{style.fmt_num(lat)}°{style.NBSP}с.{style.NBSP}ш.",
            (x[i], y[i]),
            xytext=(-2, 3),
            textcoords="offset points",
            ha="right",
            va="bottom",
            fontsize=maps.NOTE_PT,
            color=style.TEXT2,
        )
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


def plot_rhythm_map(classes: MapClasses, geo, params: SeasonParams, cfg: Config):
    """F08: МО с устойчивым своим ритмом по классам летнего избытка своего профиля (``diverging_classes``,
    легенда классов под картой), остальные МО панели — серые, МО без данных — штриховка; врезки и пунктир
    параллели."""
    n_absent = len(maps.absent_regions(maps.map_frame(geo, params.reference_year)))
    fig, ax = maps.russia_map(
        classes.labels,
        geo,
        params.reference_year,
        kind="categorical",
        cmap=classes.colors,
        legend_title="Летний избыток своего профиля (сверх общего ритма страны)",
        insets=maps.insets_from_config(cfg),
        na_label="нет устойчивого ритма",  # сюда же МО с неполным рядом: у них ритм не оценить
        absent_note=f"{n_absent} {plural(n_absent, REGION_FORMS)} без данных" if n_absent else None,
    )
    draw_parallel(ax, params.north_lat, cfg["panel"]["crs"])
    return fig


# --- Текст раздела -------------------------------------------------------------------------------


def months_text(months: Sequence[int], prepositional: bool = False) -> str:
    """«июнь–август» для подряд идущих месяцев, иначе перечисление через запятую.

    ``prepositional`` — предложный падеж для «в …»: «январе–марте»."""
    names = MONTHS_PREPOSITIONAL if prepositional else MONTHS_FULL
    ms = sorted(int(m) for m in months)
    if len(ms) > 1 and ms == list(range(ms[0], ms[-1] + 1)):
        return f"{names[ms[0] - 1]}{style.EN_DASH}{names[ms[-1] - 1]}"
    return ", ".join(names[m - 1] for m in ms)


def _except_text(exceptions: Sequence[str], dev: Mapping[str, float]) -> str:
    """«, кроме общепита (−3%)» — категории-исключения в родительном падеже с отклонением; пусто — «»."""
    if not exceptions:
        return ""
    items = [f"{CATEGORY_GENITIVE.get(c, c)} ({style.fmt_pct(dev[c], 0, sign=True)})" for c in exceptions]
    return ", кроме " + join_names(items)


def national_alt(profile: pd.DataFrame, categories: Sequence[str], v: Mapping[str, Any]) -> str:
    """Альт-текст F06 из данных: в каких категориях январь ниже среднего года, а декабрь выше, и исключения.

    ``profile`` — ``national_profile`` (медиана «месяц / среднее года»); ``v`` — числа раздела (``nat_dec``,
    ``nat_jan``, ``mp_dec``, ``mp_dec_detrended``).
    """
    dev = profile.set_index(["category", "month"])["median"] - 1
    jan = {c: float(dev[(c, JANUARY)]) for c in categories}
    dec = {c: float(dev[(c, DECEMBER)]) for c in categories}
    jan_exc = [c for c in categories if not jan[c] < 0]
    dec_exc = [c for c in categories if not dec[c] > 0]
    pct = style.fmt_pct
    return (
        f"Январь ниже среднего года во всех категориях{_except_text(jan_exc, jan)}, декабрь выше во всех"
        f"{_except_text(dec_exc, dec)}: все категории {pct(v['nat_dec'], 0, sign=True)} в декабре "
        f"и {pct(v['nat_jan'], 0, sign=True)} в январе, маркетплейсы в декабре "
        f"{pct(v['mp_dec'], 0, sign=True)} (над трендом МО {pct(v['mp_dec_detrended'], 0, sign=True)})"
    )


def _nbsp(text: str) -> str:
    """«~» в тексте раздела -> неразрывный пробел (после однобуквенных предлогов и союзов, В.6)."""
    return text.replace("~", style.NBSP)


# Текст раздела для отчёта: числа — только {{e3.ключ}} из facts.json, «~» — неразрывный пробел.
SUMMARY_PARAGRAPHS: tuple[str, ...] = (
    "**Что видно.** Годовой ритм трат в~основном общий для страны. После снятия линейного тренда каждого МО "
    "общий ритм календарного месяца объясняет {{e3.common_share}} оставшегося разброса (медиана по~МО "
    "{{e3.common_share_mo_median}}). В~типичном МО траты отклоняются от~среднего своего года "
    "на~{{e3.nat_jan}} в~январе и~на~{{e3.nat_dec}} в~декабре; декабрь оказывается месяцем с~самыми "
    "большими тратами у~{{e3.peak_dec_share_2023}} МО в~2023 году и~у~{{e3.peak_dec_share_2024}} в~2024-м. "
    "Часть декабрьского превышения даёт рост трат внутри года: над собственным линейным трендом МО декабрь "
    "отклоняется на~{{e3.nat_dec_detrended}}, у~маркетплейсов~— на~{{e3.mp_dec_detrended}} при "
    "{{e3.mp_dec}} к~среднему года. У~общепита есть свой летний подъём: в~августе траты отклоняются "
    "от~среднего года на~{{e3.cafe_aug}}.",
    "**Свой ритм.** Устойчивым своим ритмом назван профиль года, который остаётся после вычета общего ритма "
    "и~повторяется в~оба года: корреляция r профилей 2023 и~2024 годов больше {{e3.reliable_r}}, размах "
    "больше {{e3.reliable_amplitude}}. Такой ритм есть у~{{e3.reliable_n}} МО из~{{e3.n_full}} "
    "({{e3.reliable_share}}); на~данных с~перемешанными месяцами 2024 года его находят "
    "у~{{e3.null_reliable_share}}. Сигнал не~случаен, но~у~большинства МО своего ритма нет (медиана r "
    "{{e3.repro_median}}). Критерий держится в~основном на~повторяемости r, а~не~на~размахе: у~типичного МО "
    "размах своего профиля {{e3.amp_median}} при размахе чистого шума {{e3.noise_amp_median}}, связь размаха "
    "с~населением {{e3.amp_pop_rel}} (ρ Спирмена {{e3.rho_amp_pop}}~— у~малых МО размах раздувает шум), "
    "и~порог размаха отсекает лишь {{e3.amp_cut_n}} МО с~r > {{e3.reliable_r}}. Поэтому в~сводку "
    "(таблица T13) идёт размах, умноженный на~r (при r ≤ 0~— ноль): его связь с~населением "
    "{{e3.shrunk_pop_rel}} ({{e3.rho_amp_shrunk_pop}}).",
    "**Где свой ритм.** На~Севере (севернее {{e3.north_lat}}-й параллели, без внутригородских территорий "
    "Петербурга) устойчивый свой ритм есть у~{{e3.reliable_share_north}} МО, у~остальных~— "
    "у~{{e3.reliable_share_rest}}. Среди МО с~устойчивым ритмом северных {{e3.north_share_reliable}}, "
    "во~всей выборке~— {{e3.north_share_all}}. Больше всего МО с~устойчивым ритмом в~регионах: "
    "{{e3.reliable_top_regions}}. У~северных МО с~устойчивым ритмом он летний: пик своего профиля летом "
    "({{e3.summer_months}}) у~{{e3.summer_peak_share_north}}, у~остальных только "
    "у~{{e3.summer_peak_share_south}}. Самые сильные профили: {{e3.top_names}} (таблица T05).",
    "Летний избыток трат над трендом МО ({{e3.summer_months}} к~остальным месяцам без декабря) у~типичного "
    "северного МО "
    "{{e3.summer_north}}, у~остальных {{e3.summer_south}}; у~общепита {{e3.cafe_summer_north}} против "
    "{{e3.cafe_summer_south}}. Связь летнего избытка с~широтой {{e3.summer_lat_rel}} (ρ Спирмена "
    "{{e3.rho_summer_lat}}), с~долей добычи в~занятости {{e3.summer_mining_rel}} ({{e3.rho_summer_mining}}), "
    "с~долей сельского хозяйства {{e3.summer_agri_rel}} ({{e3.rho_summer_agri}}), с~ночёвками в~гостиницах "
    "на~жителя {{e3.summer_nights_rel}} ({{e3.rho_summer_nights}}). Внутри регионов {{e3.within_word}} "
    "(ρ по~модулю не~больше {{e3.rho_summer_within_max}}): {{e3.within_conclusion}}. «Аграрного» летнего "
    "слоя в~тратах не~видно. "
    "Траты приезжих в~курортных МО в~этих данных не~видны; северный летний подъём, вероятно, отражает траты "
    "самих жителей в~сезон отпусков (гипотеза).",
    "**Что это значит для сюжета.** Годовой ритм (сюжет С1) {{e3.story_role}}: устойчивый свой ритм есть "
    "у~{{e3.reliable_share}} МО при пороге критерия отказа {{e3.s1_reliable_min}}. На~Севере такой ритм есть "
    "у~{{e3.reliable_share_north}} МО~— {{e3.north_role}}. Для этапа 2 корреляции "
    "и~DTW рядов нужно считать по~своему ритму. У~рядов логарифма трат без обработки медиана корреляции "
    "случайной пары МО "
    "{{e3.pair_raw_q50}}: общий рост и~декабрь делают все МО «похожими». У~своего ритма 90-й перцентиль "
    "корреляции {{e3.pair_own_q90}} при {{e3.pair_null_q90}} на~перемешанных месяцах (таблица T06). "
    "Помесячные ряды малых МО шумят: в~самой малой из~{{e3.noise_groups}} равных групп МО по~населению шум "
    "своего ритма в~{{e3.noise_ratio_all}} раза больше, чем в~самой крупной, у~общепита "
    "в~{{e3.noise_ratio_cafe}} раза (у~транспорта отношение {{e3.noise_ratio_transport}}, таблица T07). "
    "Для малых МО по~общепиту и~транспорту надёжнее годовые доли или сжатие к~региону.",
)
SUMMARY_MD = _nbsp("\n\n".join(SUMMARY_PARAGRAPHS) + "\n")

CAVEATS: tuple[str, ...] = tuple(
    _nbsp(text)
    for text in (
        "Два годовых цикла: повторяемость своего профиля опирается на~две точки на~месяц, поэтому долю МО "
        "с~устойчивым ритмом сравниваем с~той~же долей на~перемешанных месяцах 2024 года; стандартные методы "
        "выделения сезонности (STL, X-13) на~двух годах ненадёжны.",
        "Профиль рисунка 6 («месяц / среднее своего года») включает рост трат внутри года, поэтому декабрь "
        "в~нём выше, чем над трендом МО (у~маркетплейсов {{e3.mp_dec}} против {{e3.mp_dec_detrended}}); "
        "своему ритму этот рост не~мешает.",
        "Сдвиг уровня ряда на~рубеже 2023 и~2024 годов после снятия линейного тренда даёт в~оба года "
        "одинаковую «пилу» с~пиком в~январе. Если добавить такой сдвиг в~тренд МО, устойчивый ритм остаётся "
        "у~{{e3.reliable_share_step}} МО (на~перемешанных месяцах~— {{e3.null_reliable_share_step}}). "
        "Из~{{e3.winter_peak_n}} МО вне Севера с~устойчивым ритмом и~пиком своего профиля "
        f"в~{months_text(WINTER_MONTHS, prepositional=True)} его теряют "
        "{{e3.winter_step_lost}}: их «ритм» может быть сдвигом уровня, а~не~годовым циклом. Северный слой "
        "от~этого почти не~зависит: устойчивый ритм у~{{e3.reliable_share_north}} северных МО без сдвига "
        "и~у~{{e3.reliable_share_north_step}} со~сдвигом.",
        "Траты привязаны к~жителям МО: траты приезжих в~курортном МО не~видны, и~курортный тип по~этим "
        "данным не~выделить; траты жителей вне своего МО (поездки, онлайн) по~смыслу входят, но~как модель "
        "СберИндекса их привязывает, не~раскрыто. Объяснение северного лета отпусками остаётся гипотезой.",
        "Север~— МО севернее {{e3.north_lat}}° с.~ш. по~точке внутри полигона, без внутригородских "
        "территорий Петербурга (севернее этой широты их {{e3.n_north_inner}}), как в~разделах 2 и~5.",
        "Курортные регионы юга (Краснодарский край, Крым, Севастополь) выпали из~данных целиком; выводы "
        "сделаны о~МО с~полным рядом за~24 месяца.",
        "Суммы номинальные; линейный тренд МО снимается до~расчёта ритма, поэтому общая инфляция и~рост МО "
        "на~свой ритм не~влияют.",
    )
)

INDICATOR_LABELS = {
    "own_amplitude": (
        "Размах своего годового профиля × повторяемость r (при r ≤ 0 — ноль), разность логарифмов"
    ),
    "own_repro_r": "Повторяемость своего профиля 2023 и 2024 годов, r",
    "own_reliable": "Устойчивый свой ритм (1 — да)",
}


# --- Раздел --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RhythmResults:
    """Все вычисления раздела до рисования: модели, нуль, профиль страны, пары, шум, таблица МО.

    ``top`` — все МО с устойчивым своим ритмом по убыванию размаха, с подсказками (T05: в отчёте первые
    ``top_n``, в CSV — все); ``shrunk`` — размах, сжатый на повторяемость (показатель для сводки);
    ``values`` — числа для фактов и проверок текста.
    """

    params: SeasonParams
    ids: pd.Index
    models: dict[str, RhythmModel]
    reliable: pd.Series
    shrunk: pd.Series
    null_r: np.ndarray
    profile: pd.DataFrame
    peaks: pd.Series
    pairs: pd.DataFrame
    pair_tab: pd.DataFrame
    noise: pd.DataFrame
    table: pd.DataFrame
    top: pd.DataFrame
    values: dict[str, Any]


def compute(
    data: EdaData,
    params: SeasonParams,
    rng: np.random.Generator,
    annotations: Mapping[Any, str] | None = None,
) -> RhythmResults:
    """Все вычисления раздела на ``EdaData``; случайность — только ``rng`` (сначала нуль, потом пары).

    ``annotations`` — ручные пояснения к МО (``eda.annotations``) для подсказок T05.
    """
    wide = data.panel_wide
    mo = with_short_names(data.mo, data.territories)
    ids = full_ids(mo)
    if len(ids) < params.noise_groups:
        raise ValueError(f"E3: МО с полным рядом {len(ids)} — слишком мало для модели")
    base = mo.set_index("territory_id").reindex(ids)
    needed = dict.fromkeys(("all", "cafe", "marketplace", *params.noise_categories))
    models = {c: fit_rhythm(series_matrix(wide, ids, c)) for c in needed}
    main = models["all"]
    reliable = pd.Series(is_reliable(main.r, main.A, params.reliable_r, params.reliable_amplitude), index=ids)
    null_r = null_reproducibility(main.O, rng, params.null_repeats)
    null_by_repeat = is_reliable(
        null_r, main.A.to_numpy()[None, :], params.reliable_r, params.reliable_amplitude
    ).mean(axis=1)
    pairs = pair_correlations({"raw": main.Y, "detrended": main.D, "own": main.O}, rng, params.pair_sample)
    pair_tab = pair_summary(pairs, SERIES_KINDS).set_index("series")
    # Проверка устойчивости: тренд МО со сдвигом уровня на рубеже лет (нуль — после пар, чтобы не менять их).
    step = fit_rhythm(main.Y, step_at=MONTHS_IN_YEAR)
    reliable_step = pd.Series(
        is_reliable(step.r, step.A, params.reliable_r, params.reliable_amplitude), index=ids
    )
    null_step = null_reliable_share(
        step.O, step.A, rng, params.null_repeats, params.reliable_r, params.reliable_amplitude
    )

    profile_cats = tuple(dict.fromkeys((*params.profile_categories, *REQUIRED_CATEGORIES)))
    profile = national_profile(wide, profile_cats)
    nat = profile.set_index(["category", "month"])["median"] - 1
    peaks = peak_month_share(wide, DECEMBER)
    noise = noise_by_size(
        {c: models[c].O for c in params.noise_categories}, base["weight"], params.noise_groups
    )

    # Север — как в разделах E2 и E5: широта точки ≥ north_lat, без внутригородских территорий.
    inner = base["is_inner_city"].astype("boolean").fillna(False).astype(bool)
    above = base["point_lat"].astype("float64") >= params.north_lat
    north = above & ~inner
    # Летний избыток — по ряду без тренда МО, как mo.summer_excess (EdaData): лето в году позже остальных
    # месяцев, и без снятия тренда избыток рос бы вместе с ростом трат МО.
    summer_all = seasonal_excess(main.D, params.summer_months)
    s_north, s_south = group_median(summer_all, north)
    c_north, c_south = group_median(seasonal_excess(models["cafe"].D, params.summer_months), north)
    by_year = summer_excess_by_year(main.D, params.summer_months)
    rho, rho_within, rho_years = {}, {}, {}
    for key, col in RHO_CONTEXT.items():
        x = base[col].astype("float64")
        rho[key] = stats.spearman(base["summer_excess"], x)
        rho_within[key] = stats.spearman_within(base["summer_excess"], x, base["region_name"])
        rho_years[key] = {y: stats.spearman(by_year[y], x)[0] for y in YEARS}
    table = rhythm_table(main, reliable, mo, params.summer_months)
    table["north"] = table["territory_id"].map(north).fillna(False).astype(bool)
    top = table.loc[table["reliable"]].sort_values("amplitude_log", ascending=False, kind="mergesort").copy()
    top["hint"] = (
        hints(
            mo.loc[mo["territory_id"].isin(ids)],
            top["territory_id"],
            HINT_FEATURES,
            params.hint_z,
            annotations,
        )
        .reindex(top["territory_id"])
        .to_numpy()
    )
    rel_mask = reliable.to_numpy()
    peak_north, peak_south = peak_shares(
        table.set_index("territory_id").loc[rel_mask, "peak_month"], params.summer_months, north[rel_mask]
    )
    # Пик своего профиля в январе–марте у МО вне Севера с устойчивым ритмом: кто теряет его со сдвигом уровня.
    winter = main.profile.idxmax(axis=1).astype(int).isin(WINTER_MONTHS) & reliable & ~north
    weight = base["weight"].astype("float64")
    shrunk = shrunk_amplitude(main.A, main.r)
    rho_amp = stats.spearman(main.A, weight)[0]
    rho_shrunk = stats.spearman(shrunk, weight)[0]

    values: dict[str, Any] = {
        "n_full": len(ids),
        "common_share": common_share(main.D, main.O),
        "common_share_mo_median": float(common_share_by_mo(main.D, main.O).median()),
        "repro_median": float(main.r.median()),
        "share_r06": float((main.r > params.reliable_r).mean()),
        "reliable_n": int(reliable.sum()),
        "reliable_share": float(reliable.mean()),
        "null_reliable_share": float(null_by_repeat.mean()),
        "null_se": float(null_by_repeat.std(ddof=1) / np.sqrt(len(null_by_repeat)))
        if len(null_by_repeat) > 1
        else float("nan"),
        "north_share_reliable": float(north[rel_mask].mean()) if rel_mask.any() else float("nan"),
        "north_share_all": float(north.mean()),
        "reliable_north_n": int((north & reliable).sum()),
        "reliable_share_north": _share(reliable, north),
        "reliable_share_rest": _share(reliable, ~north),
        "summer_peak_share_north": peak_north,
        "summer_peak_share_south": peak_south,
        "n_north_inner": int((above & inner).sum()),
        "amp_cut_n": int(((main.r > params.reliable_r) & ~reliable).sum()),
        "amp_median": float(np.expm1(main.A.median())),
        "noise_amp_median": float(np.expm1(noise_amplitude(main.O).median())),
        "rho_amp_pop": rho_amp,
        "rho_amp_pop_reliable": stats.spearman(main.A[reliable], weight[reliable])[0],
        "rho_amp_shrunk_pop": rho_shrunk,
        "reliable_share_step": float(reliable_step.mean()),
        "reliable_share_north_step": _share(reliable_step, north),
        "reliable_share_rest_step": _share(reliable_step, ~north),
        "null_reliable_share_step": null_step,
        "winter_peak_n": int(winter.sum()),
        "winter_step_lost": int((winter & ~reliable_step).sum()),
        "reliable_top_regions": top_counts(base.loc[rel_mask, "region_name"], params.examples),
        "top_names": join_names(top["name"].head(params.examples).astype(str).tolist()),
        "nat_jan": float(nat.loc[("all", JANUARY)]),
        "nat_dec": float(nat.loc[("all", DECEMBER)]),
        "nat_dec_detrended": common_month_excess(main.g, DECEMBER),
        "mp_dec": float(nat.loc[("marketplace", DECEMBER)]),
        "mp_dec_detrended": common_month_excess(models["marketplace"].g, DECEMBER),
        "mp_dec_years": month_ratio_by_year(wide, DECEMBER, "marketplace").to_dict(),
        "mp_growth_median": annual_growth_median(wide, "marketplace"),
        "cafe_aug": float(nat.loc[("cafe", AUGUST)]),
        "summer_north": s_north,
        "summer_south": s_south,
        "cafe_summer_north": c_north,
        "cafe_summer_south": c_south,
        "rho": rho,
        "rho_within": rho_within,
        "rho_years": rho_years,
        "rho_within_max": float(np.nanmax([abs(r) for r, _ in rho_within.values()]))
        if any(np.isfinite(r) for r, _ in rho_within.values())
        else float("nan"),
        "n_north": int(north.sum()),
    }
    for y in YEARS:
        values[f"peak_dec_share_{y}"] = float(peaks.loc[y])
    for kind in SERIES_KINDS:
        for tag in ("q50", "q90"):
            values[f"pair_{kind}_{tag}"] = float(pair_tab.at[kind, tag])
    values["pair_null_q50"] = float(pair_tab.at["own", "null_q50"])
    values["pair_null_q90"] = float(pair_tab.at["own", "null_q90"])
    for c in params.noise_categories:
        values[f"noise_ratio_{c}"] = noise_ratio(noise, c)
    return RhythmResults(
        params=params,
        ids=ids,
        models=models,
        reliable=reliable,
        shrunk=shrunk,
        null_r=null_r,
        profile=profile,
        peaks=peaks,
        pairs=pairs,
        pair_tab=pair_tab,
        noise=noise,
        table=table,
        top=top,
        values=values,
    )


def relation_text(rho: float) -> str:
    """Сила и знак связи словами для текста: «почти нулевая» (|ρ| < 0,1), «слабая прямая» (< 0,3),
    «умеренная обратная» (< 0,5), «сильная прямая» (иначе); пропуск — «не оценена»."""
    if not np.isfinite(rho):
        return _nbsp("не~оценена")
    size = abs(rho)
    if size < WEAK_RHO:
        return "почти нулевая"
    strength = "слабая" if size < MODERATE_RHO else "умеренная" if size < STRONG_RHO else "сильная"
    return f"{strength} {'прямая' if rho > 0 else 'обратная'}"


def within_words(
    rho: Mapping[str, tuple[float, int]], within: Mapping[str, tuple[float, int]]
) -> tuple[str, str]:
    """Что происходит со связями летнего избытка внутри регионов — словами по числам (Б.1, п. 9).

    Если внутри регионов каждая связь слабее, чем в целом, и все они слабые (|ρ| < 0,3), летний подъём —
    скорее свойство региона; иначе часть связи держится и между МО одного региона. Возвращает (что со
    связями, вывод).
    """
    pairs = [(rho[k][0], within[k][0]) for k in within]
    weaker = all(np.isfinite(w) and abs(w) < abs(r) for r, w in pairs)
    small = all(np.isfinite(w) and abs(w) < MODERATE_RHO for _, w in pairs)
    if pairs and weaker and small:
        return "все связи слабеют", "летний подъём скорее свойство региона, чем отдельного МО"
    return _nbsp("слабеют не~все связи"), _nbsp(
        "часть различий летнего подъёма держится и~между МО одного региона"
    )


def amplitude_size_text(rho: float) -> str:
    """Связь размаха своего профиля с населением словами для названия T05 (по знаку и силе ρ)."""
    num = style.fmt_rho(rho)
    if not np.isfinite(rho):
        return "Связь размаха с населением не оценена"
    if rho <= -WEAK_RHO:
        return f"Размах больше у малых МО (ρ Спирмена с населением среди них {num}): часть его — шум"
    if rho >= WEAK_RHO:
        return f"Размах больше у крупных МО (ρ Спирмена с населением среди них {num})"
    return f"Размах почти не связан с населением (ρ Спирмена среди них {num})"


def story_role(v: Mapping[str, Any], s1_min: float) -> str:
    """Роль сюжета С1 по его критериям отказа (Б.5): главная ось или слой."""
    share, null = v["reliable_share"], v["null_reliable_share"]
    passes = share >= s1_min and share >= RELIABLE_VS_NULL * null
    return _nbsp("может стать главной осью типологии" if passes else "годится как слой, а~не~как главная ось")


def north_role(v: Mapping[str, Any], s1_min: float) -> str:
    """Что доля МО с устойчивым ритмом на Севере значит для С1 — словами по числам (порог критерия отказа)."""
    north, share = v["reliable_share_north"], v["reliable_share"]
    if not north >= s1_min:
        return _nbsp("и~это тоже ниже порога")
    if share >= s1_min:
        return _nbsp("это тоже выше порога")
    return _nbsp("это выше порога: С1~— сильный слой для Севера, а~не~для всей страны")


def text_checks(v: Mapping[str, Any]) -> list[tuple[str, bool, str]]:
    """Утверждения текста раздела, которые не следуют из чисел напрямую: (утверждение, верно ли, числа).

    Ошибка любой проверки роняет сборку отчёта так же, как неверный заголовок графика. Слова, которые
    зависят от знака или силы связи (``relation_text``, ``within_words``, ``story_role``), выбираются
    по числам и проверки не требуют.
    """
    north, south = v["summer_peak_share_north"], v["summer_peak_share_south"]
    return [
        (
            "часть декабрьского превышения — рост внутри года, но над трендом декабрь всё равно выше",
            bool(0 < v["nat_dec_detrended"] < v["nat_dec"] and 0 < v["mp_dec_detrended"] < v["mp_dec"]),
            f"nat_dec_detrended = {v['nat_dec_detrended']:.3f}, nat_dec = {v['nat_dec']:.3f}, "
            f"mp_dec_detrended = {v['mp_dec_detrended']:.3f}, mp_dec = {v['mp_dec']:.3f}",
        ),
        (
            "у общепита летний подъём: август выше среднего года",
            bool(v["cafe_aug"] > 0),
            f"cafe_aug = {v['cafe_aug']:.3f}",
        ),
        (
            # Без южных МО с устойчивым ритмом (доля — NaN) сравнивать не с чем: утверждение о северных верно.
            "у большинства северных МО с устойчивым ритмом пик своего профиля летом, у остальных реже",
            bool(north > 0.5 and not south >= north),
            f"summer_peak_share_north = {north:.3f}, summer_peak_share_south = {south:.3f}",
        ),
        (
            "летний избыток у северных МО больше, чем у остальных (все категории и общепит)",
            bool(v["summer_north"] > v["summer_south"] and v["cafe_summer_north"] > v["cafe_summer_south"]),
            f"summer_north = {v['summer_north']:.3f}, summer_south = {v['summer_south']:.3f}, "
            f"cafe_summer_north = {v['cafe_summer_north']:.3f}, "
            f"cafe_summer_south = {v['cafe_summer_south']:.3f}",
        ),
        (
            "«аграрного» летнего слоя нет: с долей сельского хозяйства связь не прямая",
            bool(v["rho"]["agri"][0] < WEAK_RHO),
            f"rho_summer_agri = {v['rho']['agri'][0]:.3f}",
        ),
        (
            "критерий держится в основном на r: порог размаха отсекает меньше половины МО с r выше порога",
            bool(v["amp_cut_n"] < 0.5 * v["share_r06"] * v["n_full"]),
            f"amp_cut_n = {v['amp_cut_n']}, share_r06 · n_full = {v['share_r06'] * v['n_full']:.1f}",
        ),
        (
            "северный слой почти не зависит от сдвига уровня на рубеже лет",
            bool(v["reliable_share_north_step"] >= STEP_KEEP_MIN * v["reliable_share_north"]),
            f"reliable_share_north_step = {v['reliable_share_north_step']:.3f}, "
            f"reliable_share_north = {v['reliable_share_north']:.3f}",
        ),
        (
            "ряды логарифма трат без обработки у всех МО «похожи» сильнее, чем свой ритм на перемешанных "
            "месяцах",
            bool(v["pair_raw_q50"] > v["pair_null_q90"]),
            f"pair_raw_q50 = {v['pair_raw_q50']:.3f}, pair_null_q90 = {v['pair_null_q90']:.3f}",
        ),
        (
            "у своего ритма хвост корреляций пар выше, чем на перемешанных месяцах",
            bool(v["pair_own_q90"] > v["pair_null_q90"]),
            f"pair_own_q90 = {v['pair_own_q90']:.3f}, pair_null_q90 = {v['pair_null_q90']:.3f}",
        ),
        (
            "помесячные ряды малых МО шумят сильнее, чем у крупных (все категории и общепит)",
            bool(v["noise_ratio_all"] > 1 and v["noise_ratio_cafe"] > 1),
            f"noise_ratio_all = {v['noise_ratio_all']:.3f}, noise_ratio_cafe = {v['noise_ratio_cafe']:.3f}",
        ),
    ]


def fact_rows(res: RhythmResults) -> list[tuple[str, Any, str, str]]:
    """Все числа раздела для ``facts.json``: (ключ без префикса, значение, вид, определение и выборка).

    Пояснение называет определение и выборку, а там, где число расходится с ориентиром spec_final (Б.4)
    больше чем на 2%, — чем именно ориентир посчитан иначе (проверено сверкой на панели).
    """
    p, v = res.params, res.values
    pct, rho_t = style.fmt_pct, style.fmt_rho
    lat = style.fmt_num(p.north_lat)
    north = (
        f"севернее {lat}° с. ш. (eda.north_lat; точка внутри полигона МО) без внутригородских территорий, "
        "как в разделах E2 и E5"
    )
    rest = "остальных МО с полным рядом (южнее, а также все внутригородские территории)"
    prof = "медиана по МО отношения «месяц / среднее своего года» − 1, среднее двух лет"
    detrended = "exp(общего ритма g декабря, среднее двух лет) − 1"
    summer = (
        "медиана по МО с полным рядом летнего избытка mo.summer_excess (формула Б.2). Ориентиры spec_final "
        "(+10,7 и +5,7%, общепит +32,7 и +20,1%) — избыток медианного профиля группы, Север — по центру МО "
        "из справочника (центра нет у внутригородских территорий), тренд МО не снят; на панели так они "
        "воспроизводятся точно. Здесь типичное МО — медиана по МО, и летний избыток считается над линейным "
        "трендом МО, как mo.summer_excess: лето в году позже остальных месяцев, и без снятия тренда избыток "
        "рос бы вместе с ростом трат МО"
    )
    null_note = (
        f"перестановка месяцев 2024 года внутри МО, размах наблюдаемый; среднее по {p.null_repeats} "
        f"перестановкам, стандартная ошибка {pct(v['null_se'], 2)}. Ориентир 1,4% (другой генератор, "
        "20 перестановок) в пределах двух ошибок"
    )
    step = (
        "тренд МО со сдвигом уровня на рубеже 2023 и 2024 годов (y = a + b·t + c·[t ≥ 12]), дальше — та же "
        "модель и те же пороги"
    )
    amp = "размах своего профиля (разность логарифмов самого высокого и самого низкого месяца)"
    mp_years = ", ".join(f"{y} — {pct(x, 1, sign=True)}" for y, x in v["mp_dec_years"].items())
    rows: list[tuple[str, Any, str, str]] = [
        ("n_full", v["n_full"], "int", "МО с полным рядом 24 месяца — выборка модели E3"),
        ("common_share", v["common_share"], "pct", "1 − Σ o² / Σ d² по всем МО после снятия тренда МО"),
        ("common_share_mo_median", v["common_share_mo_median"], "pct", "медиана по МО доли общего ритма"),
        ("repro_median", v["repro_median"], "rho", "медиана по МО корреляции r профилей 2023 и 2024 годов"),
        (
            "share_r06",
            v["share_r06"],
            "pct",
            f"доля МО с r > {style.fmt_num(p.reliable_r, 1)} при любом размахе",
        ),
        ("reliable_n", v["reliable_n"], "int", "МО с устойчивым своим ритмом: r и размах выше порогов"),
        (
            "reliable_share",
            v["reliable_share"],
            "pct",
            "доля МО с устойчивым своим ритмом, МО с полным рядом",
        ),
        (
            "null_reliable_share",
            v["null_reliable_share"],
            "pct",
            f"то же на перемешанных месяцах (нуль): {null_note}",
        ),
        ("reliable_r", p.reliable_r, "num1", "порог повторяемости r (eda.season.reliable_r)"),
        (
            "reliable_amplitude",
            pct(p.reliable_amplitude, 0),
            "str",
            f"порог размаха своего профиля, разность логарифмов (eda.season.reliable_amplitude = "
            f"{p.reliable_amplitude}); в тексте — в процентах без знаков после запятой",
        ),
        (
            "amp_cut_n",
            v["amp_cut_n"],
            "int",
            f"МО с r > {style.fmt_num(p.reliable_r, 1)}, которых отсекает только порог размаха: критерий "
            "держится в основном на r. На реальных данных порог относительно шума МО (размах больше размаха "
            "чистого шума, см. noise_amp_median) не отсекает ни одного МО с r > 0,6: при такой повторяемости "
            "размах выше шума",
        ),
        (
            "amp_median",
            v["amp_median"],
            "pct",
            f"медиана по МО: {amp}; exp(размаха) − 1",
        ),
        (
            "noise_amp_median",
            v["noise_amp_median"],
            "pct",
            f"медиана по МО размаха своего профиля из чистого шума: d2(12) · σ / √2, d2 = {D2_12}, σ — SD по "
            "месяцам неповторяющейся части (o 2023 − o 2024) / √2; exp − 1. По SD всего своего ритма "
            "(с повторяющейся частью) шум завышен",
        ),
        (
            "rho_amp_pop",
            v["rho_amp_pop"],
            "rho",
            f"ρ Спирмена: {amp} с населением (mo.weight), все МО с полным рядом; у малых МО размах "
            "раздувает шум",
        ),
        (
            "rho_amp_pop_reliable",
            v["rho_amp_pop_reliable"],
            "rho",
            "то же среди МО с устойчивым своим ритмом (T05: самые сильные профили — у малых северных МО)",
        ),
        (
            "rho_amp_shrunk_pop",
            v["rho_amp_shrunk_pop"],
            "rho",
            "ρ Спирмена размаха, сжатого на повторяемость (размах × max(r, 0)), с населением — показатель "
            "own_amplitude для сводки (T13)",
        ),
        ("amp_pop_rel", relation_text(v["rho_amp_pop"]), "str", "сила и знак rho_amp_pop"),
        ("shrunk_pop_rel", relation_text(v["rho_amp_shrunk_pop"]), "str", "сила и знак rho_amp_shrunk_pop"),
        ("north_lat", p.north_lat, "int", "граница Севера, градусы северной широты (eda.north_lat)"),
        ("n_north", v["n_north"], "int", f"МО с полным рядом {north}"),
        (
            "n_north_inner",
            v["n_north_inner"],
            "int",
            f"внутригородские территории с полным рядом, чья точка внутри полигона севернее {lat}° (районы "
            "Петербурга); в Север не входят",
        ),
        (
            "north_share_reliable",
            v["north_share_reliable"],
            "pct",
            f"доля МО {north} среди МО с устойчивым своим ритмом",
        ),
        (
            "north_share_all",
            v["north_share_all"],
            "pct",
            f"доля МО {north} среди МО с полным рядом; ориентир 8,4% посчитан по центру МО из справочника",
        ),
        ("reliable_north_n", v["reliable_north_n"], "int", f"МО с устойчивым своим ритмом {north}"),
        (
            "reliable_share_north",
            v["reliable_share_north"],
            "pct",
            f"доля МО с устойчивым своим ритмом среди МО {north}",
        ),
        ("reliable_share_rest", v["reliable_share_rest"], "pct", f"то же среди {rest}"),
        ("north_role", north_role(v, p.s1_reliable_min), "str", "reliable_share_north и порог критерия С1"),
        (
            "reliable_share_step",
            v["reliable_share_step"],
            "pct",
            f"доля МО с устойчивым своим ритмом, проверка устойчивости: {step}",
        ),
        (
            "reliable_share_north_step",
            v["reliable_share_north_step"],
            "pct",
            f"то же среди МО {north}",
        ),
        ("reliable_share_rest_step", v["reliable_share_rest_step"], "pct", f"то же среди {rest}"),
        (
            "null_reliable_share_step",
            v["null_reliable_share_step"],
            "pct",
            f"доля «надёжных» на перемешанных месяцах при тренде со сдвигом уровня; {p.null_repeats} "
            "перестановок",
        ),
        (
            "winter_peak_n",
            v["winter_peak_n"],
            "int",
            "МО вне Севера с устойчивым своим ритмом и пиком своего профиля "
            f"в {months_text(WINTER_MONTHS, prepositional=True)}",
        ),
        (
            "winter_step_lost",
            v["winter_step_lost"],
            "int",
            f"из них теряют устойчивый ритм при тренде со сдвигом уровня ({step})",
        ),
        (
            "summer_peak_share_north",
            v["summer_peak_share_north"],
            "pct",
            f"доля МО с пиком своего профиля летом ({months_text(p.summer_months)}) среди МО с устойчивым "
            f"ритмом {north}",
        ),
        (
            "summer_peak_share_south",
            v["summer_peak_share_south"],
            "pct",
            "то же среди остальных МО с устойчивым ритмом",
        ),
        (
            "reliable_top_regions",
            v["reliable_top_regions"],
            "str",
            "регионы с наибольшим числом МО с устойчивым своим ритмом",
        ),
        (
            "top_names",
            v["top_names"],
            "str",
            "МО с наибольшим размахом устойчивого своего профиля (T05, в CSV — все такие МО); в ориентире "
            "spec_final (Ямальский, Булунский, Эгвекинот, Тигильский) — примеры без меры силы, места — в T05",
        ),
        *(
            (
                f"peak_dec_share_{y}",
                v[f"peak_dec_share_{y}"],
                "pct",
                f"доля МО с пиком трат {y} года в декабре",
            )
            for y in YEARS
        ),
        ("nat_jan", v["nat_jan"], "pct_signed", f"январь, все категории: {prof}"),
        ("nat_dec", v["nat_dec"], "pct_signed", f"декабрь, все категории: {prof}"),
        (
            "nat_dec_detrended",
            v["nat_dec_detrended"],
            "pct_signed",
            f"декабрь над линейным трендом МО, все категории: {detrended}",
        ),
        (
            "mp_dec",
            v["mp_dec"],
            "pct_signed",
            f"декабрь, маркетплейсы: {prof}; по годам: {mp_years}. Ориентир spec_final «+54%» — не декабрь, "
            f"а медиана номинального роста трат на маркетплейсах 2024/2023 (на панели "
            f"{pct(v['mp_growth_median'], 1, sign=True)}); в том же определении январь и декабрь всех "
            "категорий совпадают с ориентирами",
        ),
        (
            "mp_dec_detrended",
            v["mp_dec_detrended"],
            "pct_signed",
            f"декабрь над линейным трендом МО, маркетплейсы: {detrended}",
        ),
        ("cafe_aug", v["cafe_aug"], "pct_signed", f"август, общественное питание: {prof}"),
        ("summer_months", months_text(p.summer_months), "str", "летние месяцы (eda.summer_months)"),
        (
            "s1_reliable_min",
            pct(p.s1_reliable_min, 0),
            "str",
            f"порог критерия отказа С1 (eda.rejection.s1_reliable_share_min = {p.s1_reliable_min})",
        ),
        ("story_role", story_role(v, p.s1_reliable_min), "str", "роль С1 по критериям отказа Б.5"),
        ("summer_north", v["summer_north"], "pct_signed", f"{summer}; все категории, МО {north}"),
        ("summer_south", v["summer_south"], "pct_signed", f"{summer}; все категории, {rest}"),
        ("cafe_summer_north", v["cafe_summer_north"], "pct_signed", f"{summer}; общепит, МО {north}"),
        ("cafe_summer_south", v["cafe_summer_south"], "pct_signed", f"{summer}; общепит, {rest}"),
        ("noise_groups", p.noise_groups, "int", "групп МО равной численности по населению в T07"),
    ]
    rho_texts = {
        "lat": "широтой (точка внутри полигона)",
        "nights": "ночёвками в гостиницах на жителя 2023 года (Росстат)",
        "mining": "долей добычи в занятости 2023 года (Росстат, без малого бизнеса)",
        "agri": "долей сельского хозяйства в занятости 2023 года (Росстат, без малого бизнеса)",
    }
    extra = {
        "nights": " Ориентир −0,20 посчитан по 2024 году (лето к среднему года с декабрём, траты к медиане "
        "страны); связь неустойчива между годами"
    }
    for key, text in rho_texts.items():
        rho, n = v["rho"][key]
        years = ", ".join(f"{y} — {rho_t(x)}" for y, x in v["rho_years"][key].items())
        note = f"ρ Спирмена mo.summer_excess с {text}; n = {n}; по годам: {years}.{extra.get(key, '')}"
        rows.append((f"rho_summer_{key}", rho, "rho", note))
        rho_w, n_w = v["rho_within"][key]
        rows.append(
            (
                f"rho_summer_{key}_within",
                rho_w,
                "rho",
                f"то же внутри регионов (ранги, центрированные по региону, stats.spearman_within); n = {n_w}",
            )
        )
        rows.append((f"summer_{key}_rel", relation_text(rho), "str", "сила и знак rho_summer_" + key))
    word, conclusion = within_words(v["rho"], v["rho_within"])
    note = "по числам rho_summer_*_within: все слабее, чем в целом, и |ρ| < 0,3 — «все связи слабеют»"
    rows.append(("within_word", word, "str", note))
    rows.append(("within_conclusion", conclusion, "str", f"вывод к within_word; {note}"))
    rows.append(
        (
            "rho_summer_within_max",
            v["rho_within_max"],
            "rho",
            "наибольший по модулю ρ летнего избытка с контекстом внутри регионов (четыре признака)",
        )
    )
    n_pairs = len(res.pairs)
    for kind in SERIES_KINDS:
        for tag, word in (("q50", "медиана"), ("q90", "90-й перцентиль")):
            label = SERIES_LABELS[kind]
            note = f"{word} корреляций рядов пар МО: {label[:1].lower()}{label[1:]}; пар {n_pairs}"
            rows.append((f"pair_{kind}_{tag}", v[f"pair_{kind}_{tag}"], "rho", note))
    rows.append(
        (
            "pair_null_q50",
            v["pair_null_q50"],
            "rho",
            "медиана корреляций своего ритма пар МО на перемешанных месяцах (месяцы переставлены внутри МО)",
        )
    )
    rows.append(("pair_null_q90", v["pair_null_q90"], "rho", "90-й перцентиль того же"))
    for c in p.noise_categories:
        note = (
            f"разброс своего ритма ({CATEGORY_TEXT[c]}, 1,4826 × MAD по месяцам): медиана в самой малой "
            f"группе МО по населению / в самой крупной; групп {p.noise_groups}"
        )
        rows.append((f"noise_ratio_{c}", v[f"noise_ratio_{c}"], "num1", note))
    return rows


def run_section(ctx: SectionContext) -> Finding:
    """Раздел E3: модель ритма, факты, F06–F08, T05–T07 и показатели МО для сводки."""
    p = SeasonParams.from_config(ctx.cfg)
    res = compute(ctx.data, p, ctx.rng, ctx.cfg["eda"].get("annotations"))
    v = res.values
    for key, value, kind, note in fact_rows(res):
        ctx.fact(key, value, kind, note)
    for claim, ok, detail in text_checks(v):
        ctx.headline(f"текст раздела: {claim}", ok, detail)
    heads = headlines(v, p.north_lat)
    main = res.models["all"]
    log.info(
        "E3: устойчивый свой ритм у %d МО из %d (%.1f%%, на нуле %.1f%%), доля общего ритма %.3f",
        v["reliable_n"],
        v["n_full"],
        100 * v["reliable_share"],
        100 * v["null_reliable_share"],
        v["common_share"],
    )
    n_full = style.fmt_num(v["n_full"])

    # F06: общий ритм страны.
    h = heads["F06"]
    shown = res.profile.loc[res.profile["category"].isin(p.profile_categories)].reset_index(drop=True)
    ctx.save_figure(
        plot_national_rhythm(shown, p.profile_categories, {"marketplace": v["mp_dec_detrended"]}),
        fid="F06",
        slug="national_rhythm",
        title=ctx.headline(h.title, h.ok, h.detail),
        subtitle=f"Траты месяца к среднему своего года, рост внутри года не снят: медиана и средняя половина "
        f"МО; 2023–2024, n{NB}={NB}{n_full}",
        alt=national_alt(shown, p.profile_categories, v),
        data=shown,
        check=h.check,
    )

    # F07: повторяемость своего ритма против нуля.
    h = heads["F07"]
    hist = hist_table(main.r, res.null_r, p.hist_bin)
    ctx.save_figure(
        plot_own_rhythm(hist, p.reliable_r, v["share_r06"], p.reliable_amplitude, v["reliable_share"]),
        fid="F07",
        slug="own_rhythm",
        title=ctx.headline(h.title, h.ok, h.detail),
        subtitle=f"После снятия тренда МО и общего ритма; надёжно: r > {style.fmt_num(p.reliable_r, 1)} "
        f"и размах > {style.fmt_pct(p.reliable_amplitude, 0)}; n{NB}={NB}{n_full}",
        alt=f"Повторяемость своего профиля у большинства МО близка к нулю, как на перемешанных месяцах; "
        f"устойчивый свой ритм — у {style.fmt_pct(v['reliable_share'], 1)} МО против "
        f"{style.fmt_pct(v['null_reliable_share'], 1)} на перемешанных месяцах 2024 года",
        data=hist,
        check=h.check,
    )

    # F08: карта МО с устойчивым своим ритмом.
    h = heads["F08"]
    table = res.table
    classes = diverging_classes(
        table.loc[table["reliable"]].set_index("territory_id")["summer_own"], p.map_step, p.map_classes_side
    )
    map_data = table[
        [
            "territory_id",
            "name",
            "region_name",
            "point_lat",
            "reliable",
            "repro_r",
            "amplitude_log",
            "summer_own",
            "north",
        ]
    ].assign(map_class=table["territory_id"].map(classes.labels))
    ctx.save_figure(
        plot_rhythm_map(classes, maps.load_geometry(ctx.data.geo_path), p, ctx.cfg),
        fid="F08",
        slug="north_rhythm_map",
        title=ctx.headline(h.title, h.ok, h.detail),
        subtitle=f"Летний избыток своего профиля ({months_text(p.summer_months)}) у МО с устойчивым ритмом; "
        f"пунктир — {style.fmt_num(p.north_lat)}° с. ш.; Север без районов Петербурга",
        alt=f"МО с устойчивым своим годовым ритмом сосредоточены на Севере, и летом траты у них выше общего "
        f"ритма: {style.fmt_pct(v['north_share_reliable'], 0)} таких МО севернее "
        f"{style.fmt_num(p.north_lat)}-й параллели при {style.fmt_pct(v['north_share_all'], 0)} "
        "среди всех МО с полным рядом",
        data=map_data,
        check=h.check,
    )

    # T05: МО с самым сильным устойчивым своим ритмом.
    top = res.top
    top_cols = [
        "territory_id",
        "name",
        "region_name",
        "amplitude_log",
        "peak_month",
        "summer_own",
        "repro_r",
        *CONTEXT_COLUMNS,
        "hint",
    ]
    ctx.save_table(
        top[top_cols].reset_index(drop=True),
        tid="T05",
        slug="rhythm_top",
        title=f"МО с самым сильным устойчивым своим ритмом: первые {p.top_n} из {v['reliable_n']} по размаху "
        f"своего профиля. {amplitude_size_text(v['rho_amp_pop_reliable'])}",
        md_rows=p.top_n,
        md_formats={
            "amplitude_log": lambda a: style.fmt_pct(np.expm1(a), 0),
            "peak_month": lambda m: MONTHS_FULL[int(m) - 1],
            "summer_own": lambda s: style.fmt_pct(s, 0, sign=True),
            "repro_r": style.fmt_rho,
            "point_lat": lambda x: style.fmt_num(x, 1),
            "nights_pc_2023": lambda x: style.fmt_num(x, 1),
            "emp_sh_B_2023": lambda x: style.fmt_pct(x, 0),
            "emp_sh_A_2023": lambda x: style.fmt_pct(x, 0),
        },
        md_labels={
            "territory_id": "id",
            "name": "МО",
            "region_name": "Регион",
            "amplitude_log": "Размах",
            "peak_month": "Пик",
            "summer_own": "Летний избыток",
            "repro_r": "r",
            "point_lat": "Широта",
            "nights_pc_2023": "Ночёвки на жителя",
            "emp_sh_B_2023": "Добыча",
            "emp_sh_A_2023": "Сельское хозяйство",
            "hint": f"Подсказка (|z| > {style.fmt_num(p.hint_z)})",
        },
    )

    # T06: сигнал для рёбер сети.
    t06 = res.pair_tab.reset_index()[["series", "label", "n_pairs", "q50", "q90", "null_q50", "null_q90"]]
    ctx.save_table(
        t06,
        tid="T06",
        slug="edge_signal",
        title=f"Сигнал для рёбер (общий ритм — {style.fmt_pct(v['common_share'], 0)} разброса после тренда): "
        "корреляции и DTW — по своему ритму, иначе все МО похожи",
        md_formats={k: style.fmt_rho for k in ("q50", "q90", "null_q50", "null_q90")},
        md_labels={
            "series": "Код",
            "label": "Ряд",
            "n_pairs": "Пар МО",
            "q50": "Медиана ρ",
            "q90": "90-й перцентиль ρ",
            "null_q50": "Медиана, месяцы перемешаны",
            "null_q90": "90-й перцентиль, месяцы перемешаны",
        },
    )

    # T07: шум своего ритма по группам населения.
    md_noise: dict[str, Any] = {f"sd_{c}": (lambda x: style.fmt_pct(x, 1)) for c in p.noise_categories}
    md_noise.update({"pop_min": style.fmt_num, "pop_max": style.fmt_num})
    ctx.save_table(
        res.noise,
        tid="T07",
        slug="noise_by_size",
        title="Шум своего ритма по группам МО по населению (1 — самые малые): разброс 1,4826 × MAD, "
        "в процентах",
        md_formats=md_noise,
        md_labels={
            "group": "Группа",
            "n_mo": "МО",
            "pop_min": "Население от",
            "pop_max": "Население до",
            **{f"sd_{c}": f"Шум: {CATEGORY_TEXT[c]}" for c in p.noise_categories},
        },
    )

    indicators = pd.DataFrame(
        {
            "territory_id": res.ids.to_numpy().astype("int32"),
            "own_amplitude": res.shrunk.reindex(res.ids).to_numpy(),
            "own_repro_r": main.r.to_numpy(),
            "own_reliable": res.reliable.to_numpy().astype("float64"),
        }
    )
    return ctx.finding(
        title=TITLE,
        summary_md=SUMMARY_MD,
        indicators=indicators,
        indicator_labels=INDICATOR_LABELS,
        caveats=CAVEATS,
    )
