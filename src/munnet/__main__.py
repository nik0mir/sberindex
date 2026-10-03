"""Точка входа ``python -m munnet``: передаёт аргументы командной строке ``munnet.cli.main``."""

import sys

from munnet.cli import main

sys.exit(main())
