"""Общие фикстуры тестов."""

from __future__ import annotations

import pytest


@pytest.fixture
def stub_usefulness_level(monkeypatch):
    """Заглушка справки ``munnet.usefulness_level`` для сквозных тестов ``usefulness.run`` на синтетике.

    Справка читает узлы, цели и бутстрап кодом interpret (``interpret.data.load``, входы кластеризации), а на
    синтетических выходах interpret их нет. Подключается явно: ``pytestmark =
    pytest.mark.usefixtures("stub_usefulness_level")`` в тестовых файлах usefulness. Функции справки
    проверяет ``tests/test_usefulness_level.py`` на игрушечных данных, сверку с T1 — сама справка на
    настоящих данных (код 3 при расхождении с ``t1_runs.csv``)."""
    from munnet import usefulness_level

    monkeypatch.setattr(usefulness_level, "run", lambda cfg: None)
