"""Проверка хука readonly_guard: разрешённые и запрещённые команды проверяющих агентов.

Запуск из корня репозитория: PYTHONIOENCODING=utf-8 uv run --frozen python .claude/hooks/test_guard.py —
ожидается «failures: 0». Хук вызывается тем же интерпретатором, CLAUDE_PROJECT_DIR — корень репозитория.
"""

import json
import os
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
GUARD = HERE / "readonly_guard.py"
ROOT = HERE.parents[1]  # <репозиторий>/.claude/hooks → <репозиторий>
env = dict(os.environ, CLAUDE_PROJECT_DIR=str(ROOT), PYTHONIOENCODING="utf-8")
ALLOW = [
    "git status --short",
    'git -C "$SRC" log --oneline | head',
    "git rev-parse --short HEAD",
    'git clone --quiet --no-hardlinks "$SRC" "$T/clean"',
    'PYTHONIOENCODING=utf-8 uv run --frozen python -m munnet --config "$T/judge.yaml" features network',
    "uv run --frozen python -m munnet --help",
    "PYTHONIOENCODING=utf-8 uv run --frozen python - <<'EOF'\nimport pandas as pd\nx = 1 > 0\npd.DataFrame().to_csv('a')\nprint(x >= 1)\nEOF",  # noqa: E501
    "uv run --frozen pytest -q -p no:cacheprovider",
    r'grep -rnE "=\s*[0-9]+" src/munnet > /dev/null',
    "ls site/ 2>&1 | head",
    'echo x > "$T/a.txt"',
    "cat report/report.md > $T/out.md",
    'T=$(mktemp -d); command -v cygpath >/dev/null && T=$(cygpath -m "$T"); echo $T',
    "uv run --frozen ruff check . && uv run --frozen ruff format --check .",
    'python -c "print(1 > 0)"',
    "grep -rn munnet src | head",
    'mkdir -p "$T/eda" && cp outputs/x.png "$T/eda/"',
    'cd "$T" && echo hi > note.txt && rm -f note.txt',
    "uv export --frozen --no-emit-project --no-dev | tail -n +3 | diff - requirements.txt",
    "PYTHONIOENCODING=utf-8 uv run --frozen python .claude/skills/scientific-visualization/scripts/palette_audit.py --help",  # noqa: E501
    'find src -name "*.py" -newer PLAN.md 2>/dev/null',
    "echo done >&2",
    "cat a.txt &> /dev/null",
    'cd "$T/clean" && uv sync --frozen && uv run --frozen pytest -q -p no:cacheprovider',
    'uv sync --frozen --project "$T/clean"',
    'python -m venv "$T/venv" && "$T/venv/Scripts/python" -m pip install -r requirements.txt',
    'tar --exclude=./.venv --exclude=./.git --exclude=./data -cf - . | tar -xf - -C "$T"',
    "HTTPS_PROXY=http://127.0.0.1:9 HTTP_PROXY=http://127.0.0.1:9 uv run --frozen pytest -q -p no:cacheprovider",  # noqa: E501
    # Этап 1 (panel, eda) — только с временным конфигом из раздела 8 скила munnet-agents.
    'PYTHONIOENCODING=utf-8 uv run --frozen python -m munnet --config "$T/check.yaml" panel eda',
    "git status --short   # после прогона — то же, что до него",
    'diff "$SRC/docs/eda.md" "$T/copy/docs/eda.md"',
    'cp docs/eda.md "$T/eda_before.md"',
    "grep -c . outputs/panel/controls.json",
]
BLOCK = [
    "git commit -m x",
    "git add .",
    "git push origin main",
    "git checkout -- src/munnet/features.py",
    "git stash",
    "git reset --hard",
    "uv run --frozen python -m munnet features",
    "uv run --frozen python -m munnet --config configs/default.yaml features",
    'PYTHONIOENCODING=utf-8 uv run --frozen python -m munnet --config "$T/j.yaml" all',
    'uv run --frozen python -m munnet --config "$T/j.yaml" data',
    "echo x > report/report.md",
    "cat a >> README.md",
    "rm -rf outputs",
    "rm data/interim/x.parquet",
    "sed -i 's/a/b/' src/munnet/features.py",
    "uv add statsmodels",
    "uv lock",
    "pip install playwright",
    "uv run --frozen ruff check --fix .",
    "uv run --frozen ruff format .",
    "echo x | tee report/notes.md",
    'cp "$T/x.py" src/munnet/x.py',
    'rm -rf "$SRC/outputs"',
    "echo x > /c/Users/1/Desktop/sberindex/docs/x.md",
    "uv run munnet --config configs/default.yaml network",
    "uv sync --frozen",
    "python -m pip install playwright",
    "uv lock --upgrade",
    "uv run --frozen python -m munnet panel eda",
    "uv run --frozen python -m munnet --config configs/default.yaml eda",
    "echo x > docs/eda.md",
    "rm -rf docs/img/eda",
    'cp "$T/docs/eda.md" docs/eda.md',
    'cp "$T/x.png" docs/img/eda/F01_coverage_map.png',
    "sed -i 's/a/b/' .claude/agents/judge-c1-methodology.md",
    "git restore docs/eda.md",
]


def run() -> int:
    """Прогоняет все команды через хук, печатает по строке на команду; возвращает число несовпадений."""
    bad = 0
    for expect, cases in ((0, ALLOW), (2, BLOCK)):
        for c in cases:
            payload = json.dumps({"tool_name": "Bash", "tool_input": {"command": c}})
            p = subprocess.run(
                [sys.executable, str(GUARD)],
                input=payload,
                capture_output=True,
                text=True,
                env=env,
                encoding="utf-8",
            )
            ok = p.returncode == expect
            bad += not ok
            print(("OK  " if ok else "FAIL"), expect, p.returncode, repr(c[:70]), p.stderr.strip()[:110])
    print("failures:", bad)
    return bad


def test_readonly_guard() -> None:
    """Тот же прогон под pytest: uv run --frozen pytest .claude/hooks/test_guard.py -p no:cacheprovider -q."""
    assert run() == 0


if __name__ == "__main__":
    sys.exit(1 if run() else 0)
