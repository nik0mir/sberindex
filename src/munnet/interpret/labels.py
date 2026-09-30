"""Подписи этапа interpret: признаки, обороты, разбиения половин, наборы T7, соперники.

Подписи признаков — как в отчёте этапа 3 (``munnet.clustering.figures.FEATURE_LABELS``). Остальное — слова для
подстановки в поля текстов исходов (``{turnover_beyond}``, ``{set_name}`` …): в предрегистрации они записаны
в самих текстах или в комментариях к ключам, отдельных ключей для них нет (уточнение реализации).
"""

from __future__ import annotations

from munnet.clustering.figures import FEATURE_LABELS

# Обороты T1/T5: именительный падеж («общепит 0,xx, розница 0,yy» — комментарий к outcomes T1)
# и родительный («по обороту общепита»).
TURNOVER_NOM: dict[str, str] = {"catering": "общепит", "retail": "розница"}
TURNOVER_GEN: dict[str, str] = {"catering": "общепита", "retail": "розницы"}
# Разбиения половин T2/T3: основное (odd_even) и чувствительность (halves_sensitivity).
SCHEME_LABELS: dict[str, str] = {
    "main": "«нечётные и чётные месяцы»",
    "sensitivity": "«месяцы парами»",
}
# Наборы T7 (комментарий к product_set и тексты исходов).
SET_NAMES: dict[str, str] = {
    "A": "сопоставимые территории того же типа",
    "D": "сопоставимые территории",
    "B": "соседи по своему региону",
    "C": "случайные МО того же размера",
}
# «Каждое десятое МО» — вставка {one_in_ten_phrase} (комментарий к T3.one_in_ten).
ONE_IN_TEN_PHRASE = " — примерно каждое десятое МО"
PARTITION_LABELS: dict[str, str] = {
    "pop_decile": "децили населения",
    "urban_group": "децили доли горожан",
    "mo_type": "статус МО",
    "federal_district": "федеральный округ",
    "coverage_class": "покрытие данных",
    "place_only": "K-means по признакам места",
    "rank_pop_urban": "сумма рангов населения и доли горожан",
    "region": "группа региона",
}
PROXY_LABELS: dict[str, str] = {
    "city_status": "доля городских округов",
    "capital": "доля региональных столиц",
    "density": "медиана плотности относительно региона",
}


def feature_label(name: str) -> str:
    return FEATURE_LABELS.get(name, name)


def sized_label(feature: str, reverse: bool) -> str:
    base = PARTITION_LABELS.get(feature) or feature_label(feature).lower()
    return f"{base}, {'по убыванию' if reverse else 'по возрастанию'}"
