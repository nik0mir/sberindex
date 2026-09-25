"""Командная строка: python -m munnet [--config путь] этап [этап ...]."""

import argparse
import importlib
import logging
from pathlib import Path

from munnet.config import DEFAULT_CONFIG, load_config

log = logging.getLogger("munnet")

# Этапы в порядке выполнения: имя -> (модуль с функцией run(cfg), описание).
STAGES = {
    "data": ("munnet.data", "скачать и распаковать исходные данные"),
    "features": ("munnet.features", "собрать панель «МО × месяц × категория» и признаки узлов"),
    "network": ("munnet.network", "построить рёбра по правилам из конфига"),
    "cluster": ("munnet.clustering", "кластеризовать узлы сети разными методами"),
    "evaluate": ("munnet.icvi", "посчитать ICVI: SW, CH, S_Dbw, AVI, AVU, MQ"),
    "dynamics": ("munnet.dynamics", "проследить изменения кластеров во времени"),
    "site": ("munnet.landing", "собрать данные для интерактивного лендинга"),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="munnet",
        description="Пайплайн проекта. Этапы: "
        + "; ".join(f"{name} — {desc}" for name, (_, desc) in STAGES.items())
        + "; all — все по порядку.",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="YAML с параметрами")
    parser.add_argument("stages", nargs="+", choices=[*STAGES, "all"], metavar="этап")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config)
    names = list(STAGES) if "all" in args.stages else args.stages
    for name in names:
        module_name, desc = STAGES[name]
        log.info("Этап %s: %s", name, desc)
        try:
            importlib.import_module(module_name).run(cfg)
        except NotImplementedError as e:
            log.error("Этап %s: %s", name, e)
            return 2
    return 0
