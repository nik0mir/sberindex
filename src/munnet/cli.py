"""Командная строка: python -m munnet [--config путь] этап [этап ...] [--only e3,e4].

Коды выхода: 0 — успех; 1 — нет входов этапа (не запускался предыдущий этап); 2 — этап не реализован
(``NotImplementedError``); 3 — не прошла проверка (``QCError``: жёсткое контрольное число или заголовок
графика, переставший быть верным). Сообщение называет, что именно не сошлось.
"""

import argparse
import importlib
import logging
from pathlib import Path

from munnet.config import DEFAULT_CONFIG, load_config
from munnet.contracts import MissingInputError, QCError

log = logging.getLogger("munnet")

# Этапы в порядке выполнения: имя -> (модуль с функцией run(cfg), описание).
STAGES = {
    "data": ("munnet.data", "скачать и распаковать исходные данные"),
    "panel": ("munnet.panel", "собрать панель «МО × месяц × категория», справочник территорий и контекст"),
    "eda": ("munnet.eda", "разведочный анализ: графики, таблицы и отчёт docs/eda.md"),
    "features": ("munnet.features", "признаки узлов для сети (этап 2)"),
    "network": ("munnet.network", "построить рёбра по правилам из конфига"),
    "cluster": ("munnet.clustering", "кластеризовать узлы сети разными методами"),
    "evaluate": ("munnet.icvi", "посчитать ICVI: SW, CH, S_Dbw, AVI, AVU, MQ"),
    "dynamics": ("munnet.dynamics", "проследить изменения кластеров во времени"),
    "interpret": ("munnet.interpret", "проверки тезиса, профили, названия и примеры типов (этап 5)"),
    "site": ("munnet.landing", "собрать данные для интерактивного лендинга"),
}

# Этап, для которого действует --only: разделы разведки считаются по одному.
ONLY_STAGE = "eda"
# Этап, для которого действует --blind: слепой прогон с перемешанными метками типов (отладка).
BLIND_STAGE = "interpret"

EXIT_OK, EXIT_MISSING_INPUT, EXIT_NOT_IMPLEMENTED, EXIT_QC = 0, 1, 2, 3


def _parse_only(
    parser: argparse.ArgumentParser, values: list[str] | None, stages: list[str]
) -> list[str] | None:
    if not values:
        return None
    if ONLY_STAGE not in stages:
        parser.error(f"--only действует только вместе с этапом {ONLY_STAGE}")
    from munnet.eda.base import SECTION_IDS, SYNTHESIS_ID

    ids = [i.strip() for v in values for i in v.split(",") if i.strip()]
    allowed = (*SECTION_IDS, SYNTHESIS_ID)
    unknown = [i for i in ids if i not in allowed]
    if unknown or not ids:
        parser.error(f"--only: неизвестные разделы {unknown}; допустимы {', '.join(allowed)}")
    return ids


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="munnet",
        description="Пайплайн проекта. Этапы: "
        + "; ".join(f"{name} — {desc}" for name, (_, desc) in STAGES.items())
        + "; all — все по порядку.",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="YAML с параметрами")
    parser.add_argument(
        "--only",
        action="append",
        metavar="РАЗДЕЛЫ",
        help="для этапа eda: только эти разделы через запятую (e1…e5; syn — сводка и отчёт из сохранённых "
        "итогов разделов); общие выходы разведки пишет только полный прогон или syn",
    )
    parser.add_argument(
        "--blind",
        type=int,
        metavar="SEED",
        help="для этапа interpret: слепой прогон — типы и переходы перемешаны по узлам с этим seed "
        "(не общим); "
        "выходы — outputs/interpret_blind, отчёт с пометкой «СЛЕПОЙ ПРОГОН»",
    )
    parser.add_argument("stages", nargs="+", choices=[*STAGES, "all"], metavar="этап")
    args = parser.parse_args(argv)
    names = list(STAGES) if "all" in args.stages else args.stages
    only = _parse_only(parser, args.only, names)
    if args.blind is not None and BLIND_STAGE not in names:
        parser.error(f"--blind действует только вместе с этапом {BLIND_STAGE}")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config)
    for name in names:
        module_name, desc = STAGES[name]
        log.info("Этап %s: %s", name, desc)
        try:
            module = importlib.import_module(module_name)
            if name == ONLY_STAGE and only:
                module.run_sections(cfg, only)
            elif name == BLIND_STAGE and args.blind is not None:
                module.run(cfg, blind=args.blind)
            else:
                module.run(cfg)
        except NotImplementedError as e:
            log.error("Этап %s: %s", name, e)
            return EXIT_NOT_IMPLEMENTED
        except QCError as e:
            log.error("Этап %s: не прошла проверка: %s", name, e)
            return EXIT_QC
        except MissingInputError as e:
            log.error("Этап %s: %s", name, e)
            return EXIT_MISSING_INPUT
    return EXIT_OK
