"""Этап dynamics: типы локальных экономик по скользящим окнам, переходы, события и шум (PLAN.md, этап 3).

Порядок и правила — предрегистрация ``dynamics.tracking`` в ``configs/default.yaml``;
отчёт — ``docs/dynamics.md``.
"""

from munnet.config import Config


def run(cfg: Config) -> None:
    from munnet.dynamics.stage import run as _run

    _run(cfg)
