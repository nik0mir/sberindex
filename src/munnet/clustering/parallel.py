"""Параллельный запуск задач этапа: процессы получают общие входы один раз (инициализатор), задача несёт
свой seed. Итог не зависит от числа процессов: порядок ответов — порядок задач, случайность — от номера
задачи.
При ``workers`` = 1 всё считается в текущем процессе (тесты, отладка)."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor
from typing import Any

log = logging.getLogger(__name__)

STATE: dict[str, Any] = {}


def init_state(state: dict[str, Any], limit_threads: bool = True) -> None:
    """Инициализатор процесса: общие входы задач (входы кластеризации, параметры реализации); в дочернем
    процессе — один поток BLAS (иначе процессы × потоки переподписывают ядра)."""
    import warnings

    warnings.filterwarnings("ignore")
    STATE.clear()
    STATE.update(state)
    if limit_threads:
        from threadpoolctl import threadpool_limits

        STATE["_threads"] = threadpool_limits(limits=1)


class Pool:
    """Процессы с общими входами на несколько шагов: ``with Pool(state, workers) as p: p.map(fn, tasks)``."""

    def __init__(self, state: dict[str, Any], workers: int) -> None:
        self.state = state
        self.workers = int(workers)
        self._ex: ProcessPoolExecutor | None = None
        self._saved: dict[str, Any] = {}

    def __enter__(self) -> Pool:
        if self.workers > 1:
            self._ex = ProcessPoolExecutor(
                max_workers=self.workers, initializer=init_state, initargs=(self.state,)
            )
        else:
            self._saved = dict(STATE)
            init_state(self.state, limit_threads=False)
        return self

    def __exit__(self, *exc) -> None:
        if self._ex is not None:
            self._ex.shutdown()
        else:
            STATE.clear()
            STATE.update(self._saved)

    def map(self, fn: Callable[[Any], Any], tasks: Sequence[Any] | Iterable[Any]) -> list:
        tasks = list(tasks)
        if not tasks:
            return []
        if self._ex is None:
            return [fn(t) for t in tasks]
        return list(self._ex.map(fn, tasks, chunksize=max(1, len(tasks) // (8 * self.workers))))


def run(
    fn: Callable[[Any], Any], tasks: Sequence[Any] | Iterable[Any], state: dict[str, Any], workers: int
) -> list:
    """``[fn(t) for t in tasks]`` в ``workers`` процессах; ``state`` — в ``STATE`` каждого процесса."""
    tasks = list(tasks)
    if not tasks:
        return []
    with Pool(state, workers if len(tasks) > 1 else 1) as pool:
        return pool.map(fn, tasks)
