"""Проверка палитр типов до вёрстки — ссылка ``site.palette_rule.type_palette.audit`` в конфиге.

Тонкая обёртка над ``munnet.style.palette_audit``: порядок ступеней — ``interpret.ladder.order``, фигуры —
``site.palette_rule.type_palette.shapes``. Этап site делает ту же проверку сам (порядок — из facts.json
этапа 5); этот скрипт — для правки палитр в ``style.py`` без прогона этапов.

Запуск: ``PYTHONIOENCODING=utf-8 uv run --frozen python scripts/palette_audit.py``, ключи ``--config <yaml>``
и ``--json``. Код выхода: 0 — все проверки пройдены, 3 — есть нарушение (как у этапов).
"""

from __future__ import annotations

import argparse
import json
import sys

from munnet import style
from munnet.config import load_config


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Проверка палитр типов (контраст, серый, дальтонизм, фигуры)")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--json", action="store_true", help="итог одним JSON вместо строк")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    order = [int(t) for t in cfg["interpret"]["ladder"]["order"]]
    shapes = cfg["site"]["palette_rule"]["type_palette"]["shapes"]
    audit = style.palette_audit(order, shapes)
    if args.json:
        sys.stdout.write(json.dumps(audit.to_dict(), ensure_ascii=False, indent=1) + "\n")
    else:
        for c in audit.checks:
            mark = "ok  " if c["passed"] else "FAIL"
            sys.stdout.write(f"{mark} {c['check']}: {c['value']} (порог {c['threshold']}) — {c['detail']}\n")
        sys.stdout.write(f"итог: {len(audit.checks) - len(audit.failures)} из {len(audit.checks)} пройдены\n")
    return audit.exit_code


if __name__ == "__main__":
    sys.exit(main())
