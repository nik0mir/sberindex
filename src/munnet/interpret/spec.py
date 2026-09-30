"""Чтение предрегистрации ``interpret`` (configs/default.yaml, коммит 90991e1; поправки текстов
до вскрытия — ``amendments.py``) с учётом прочитанных ключей.

Этап реализует блок буквально: каждый порог, число повторов, список признаков и текст исхода берётся из блока.
``Spec`` записывает путь каждого прочитанного ключа; в конце этапа все листья блока должны быть прочитаны
(``unread`` пуст), иначе этап останавливается: ключ, который код не прочитал, код и не реализовал.
Варианты, которых код не умеет (другое значение ``statistic``, ``test`` и т. п.), — ``ValueError``, а не тихая
подмена: ``expect``.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any


class Spec:
    """Узел блока ``interpret``: словарь, доступ к которому записывается в общий журнал ``seen``."""

    def __init__(
        self, data: Mapping[Any, Any], path: str = "interpret", seen: set[str] | None = None
    ) -> None:
        self._data = data
        self._path = path
        self._seen: set[str] = set() if seen is None else seen

    # --- доступ ---------------------------------------------------------------------------------

    def _child(self, key: Any) -> str:
        return f"{self._path}.{key}"

    def __getitem__(self, key: Any) -> Any:
        if key not in self._data:
            raise KeyError(f"{self._child(key)}: ключа нет в предрегистрации")
        value = self._data[key]
        path = self._child(key)
        self._seen.add(path)
        if isinstance(value, Mapping):
            return Spec(value, path, self._seen)
        return value

    def __contains__(self, key: Any) -> bool:
        return key in self._data

    def keys(self) -> list[Any]:
        return list(self._data.keys())

    def items(self) -> Iterator[tuple[Any, Any]]:
        for key in self._data:
            yield key, self[key]

    def __len__(self) -> int:
        return len(self._data)

    @property
    def path(self) -> str:
        return self._path

    def plain(self) -> dict[Any, Any]:
        """Словарь целиком (все листья под узлом считаются прочитанными): для текстов и публикации."""
        for leaf in self._leaves(self._data, self._path):
            self._seen.add(leaf)
        return _plain(self._data)

    # --- проверки -------------------------------------------------------------------------------

    def expect(self, key: Any, implemented: Any) -> Any:
        """Значение ключа, равное реализованному варианту; иное — ``ValueError`` (правило не подменяется)."""
        value = self[key]
        ok = value == implemented if not isinstance(implemented, tuple) else value in implemented
        if not ok:
            raise ValueError(f"{self._child(key)} = {value!r}: реализовано {implemented!r}")
        return value

    @staticmethod
    def _leaves(data: Any, path: str) -> list[str]:
        if isinstance(data, Mapping) and data:
            out = []
            for k, v in data.items():
                out.extend(Spec._leaves(v, f"{path}.{k}"))
            return out
        return [path]

    def leaves(self) -> list[str]:
        return self._leaves(self._data, self._path)

    def unread(self) -> list[str]:
        """Листья блока, которых код не прочитал."""
        return [leaf for leaf in self.leaves() if leaf not in self._seen]


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_plain(v) for v in value]
    return value


def resolve(spec_root: Spec, ref: str) -> Spec | Any:
    """Ссылка вида ``interpret.tests.T2_direction.place_tree.features`` внутри блока — её значение."""
    parts = ref.split(".")
    if parts[0] != "interpret":
        raise ValueError(f"ссылка {ref!r} вне блока interpret")
    node: Any = spec_root
    for p in parts[1:]:
        node = node[p]
    return node
