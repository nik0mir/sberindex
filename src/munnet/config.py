"""Чтение конфигурации из YAML."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG = Path("configs/default.yaml")


@dataclass(frozen=True)
class Config:
    data: dict[str, Any]
    path: Path

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def dir(self, name: str) -> Path:
        """Каталог из секции paths; создаётся, если его ещё нет."""
        path = Path(self.data["paths"][name])
        path.mkdir(parents=True, exist_ok=True)
        return path


def load_config(path: str | Path = DEFAULT_CONFIG) -> Config:
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        return Config(data=yaml.safe_load(f), path=path)
