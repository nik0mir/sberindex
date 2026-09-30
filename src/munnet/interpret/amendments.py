"""Поправки текстов предрегистрации ``interpret`` до вскрытия результатов этапа 5.

Каждая поправка — путь к полю внутри блока ``interpret``, прежний текст («было»), дата и причина. «Стало»
не записывается здесь, а берётся из конфига при сборке отчёта, поэтому список «было → стало» механический.
Поправка только сужает утверждение до проверенного: правила, пороги и статистики не меняются. В конфиге
рядом с полем — комментарий «ПОПРАВКА <дата> до вскрытия результатов» с тем же прежним текстом
(сверяет тест ``test_amendments_match_config``).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Amendment:
    date: str
    path: str  # путь внутри блока interpret, через точку
    before: str
    reason: str


_T6_REASON = (
    "текст T6 стоит на экране 0 при любом исходе (thesis_assembly.caveat) и утверждал «сдвиг {direction}» "
    "как общий вывод: при T3 not выходило «смен типа не больше плацебо… Сдвиг виден», при T2 down — «сдвиг "
    "к городской корзине» рядом с «вниз чаще». T6 сравнивает только МО потоков movers с оставшимися того же "
    "исходного типа; новый текст говорит только это (devils-advocate, 30.09.2026)"
)
_T7_REASON = (
    "цель T7 — изменение розничного оборота за тот же период 2023 → 2024, что и у самого МО: это сверка, "
    "а не опережающий прогноз; «предсказывают», «как прогноз», «предсказывает» утверждали больше "
    "проверенного (devils-advocate, 30.09.2026)"
)

AMENDMENTS: tuple[Amendment, ...] = (
    Amendment(
        "30.09.2026",
        "tests.T6_bank_coverage.outcomes.confirmed.text",
        "Сдвиг {direction} виден и вне данных банка: оборот общепита Росстата на жителя у перешедших МО рос "
        "быстрее, чем у оставшихся того же исходного типа (дельта Клиффа {delta}, интервал {ci}).",
        _T6_REASON,
    ),
    Amendment(
        "30.09.2026",
        "tests.T6_bank_coverage.outcomes.partial.text",
        "В обороте общепита Росстата сдвиг {direction} виден слабо (p = {p}, дельта Клиффа {delta}, интервал "
        "{ci}); распространение безналичной оплаты как объяснение не исключено.",
        _T6_REASON,
    ),
    Amendment(
        "30.09.2026",
        "tests.T6_bank_coverage.outcomes.not.text",
        "Сдвиг {direction} виден в безналичных тратах; в обороте общепита Росстата (без МСП) — нет. "
        "Распространение безналичной оплаты как объяснение не исключено.",
        _T6_REASON,
    ),
    Amendment(
        "30.09.2026",
        "tests.T7_utility.outcomes.confirmed.text",
        "{set_name} из других регионов предсказывают изменение розничного оборота 2023 → 2024 точнее, чем "
        "соседи по своему региону и случайные МО того же размера (медианная ошибка {err_P} против {err_B} "
        "и {err_C}).",
        _T7_REASON,
    ),
    Amendment(
        "30.09.2026",
        "tests.T7_utility.outcomes.partial.text",
        "{set_name} точнее, чем {passed_set}, но не точнее, чем {failed_set}: как ориентир для сравнения "
        "— да, как прогноз — нет.",
        _T7_REASON,
    ),
    Amendment(
        "30.09.2026",
        "tests.T7_utility.outcomes.not.text",
        "Подбор сопоставимых территорий — описательный инструмент: изменение розничного оборота он "
        "предсказывает не лучше соседей по региону и случайных МО того же размера.",
        _T7_REASON,
    ),
)


def current(block: Mapping[str, Any], path: str) -> str:
    """Текущее значение поля ``path`` внутри блока ``interpret`` («стало»)."""
    node: Any = block
    for part in path.split("."):
        node = node[part]
    return str(node)


def lines(block: Mapping[str, Any]) -> list[str]:
    """Строки отчёта «было → стало» — по одной на поправку."""
    return [
        f"- {a.date}, `interpret.{a.path}`: было «{a.before}» → стало «{current(block, a.path)}»; "
        f"правило не менялось. Почему: {a.reason}."
        for a in AMENDMENTS
    ]
