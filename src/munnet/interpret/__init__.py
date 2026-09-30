"""Этап interpret (этап 5 плана): проверки тезиса T1–T7, фильтр R1, профили, названия и примеры типов.

Правила — предрегистрация ``interpret`` в ``configs/default.yaml`` (коммит 90991e1; тексты исходов T6 и T7
уточнены до вскрытия, правила не менялись — ``amendments.py``); отчёт — ``docs/interpretation.md``.
Слепой прогон для отладки — ``python -m munnet interpret --blind SEED``.
"""

from munnet.config import Config


def run(cfg: Config, blind: int | None = None) -> None:
    from munnet.interpret.stage import run as _run

    _run(cfg, blind=blind)
