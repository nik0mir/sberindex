"""Проверка агентов проекта: фронтматтер, «когда не звать», инструменты, скилы, строки шкалы жюри, хук,
ссылки, типографика скила, инструкции и агентов.

Запуск из корня репозитория (после любой правки `.claude/agents`, скила munnet-agents или хука):

    PYTHONIOENCODING=utf-8 uv run --frozen python .claude/tools/validate_agents.py

Код выхода 0 — «проблем нет», 1 — список проблем в выводе. Предупреждения код выхода не меняют.
"""

from __future__ import annotations

import pathlib
import re
import sys

import yaml

REPO = pathlib.Path(__file__).resolve().parents[2]
AGENTS = REPO / ".claude" / "agents"
SKILLS = REPO / ".claude" / "skills"
SKILL = SKILLS / "munnet-agents" / "SKILL.md"
GUIDE = REPO / "docs" / "agents.md"
HOOK = REPO / ".claude" / "hooks" / "readonly_guard.py"
CLI = REPO / "src" / "munnet" / "cli.py"
HOOK_COMMAND = (
    'uv run --frozen --project "$CLAUDE_PROJECT_DIR" python '
    '"$CLAUDE_PROJECT_DIR/.claude/hooks/readonly_guard.py"'
)

FIELDS = {
    "name", "description", "tools", "disallowedTools", "model", "permissionMode", "maxTurns", "memory",
    "skills", "mcpServers", "hooks", "isolation", "effort", "context", "agent", "color",
}  # fmt: skip
BASE_TOOLS = {
    "Read", "Grep", "Glob", "Bash", "Edit", "Write", "WebSearch", "WebFetch", "NotebookEdit", "Agent",
}  # fmt: skip
BROWSER = {
    "preview_start", "preview_stop", "navigate", "computer", "read_page", "get_page_text", "find",
    "resize_window", "read_console_messages", "read_network_requests", "javascript_tool", "browser_batch",
}  # fmt: skip
# Файлы, которые появятся по плану этапов (этап 1 — пакеты F, P, E1–E5, R; дальше — отчёт, лендинг, журналы):
# ссылаться на них можно заранее.
PLANNED = (
    "src/munnet/panel/", "src/munnet/eda/", "src/munnet/contracts.py", "src/munnet/style.py",
    "src/munnet/maps.py", "docs/eda.md", "docs/img/eda/", "docs/hypotheses.md", "docs/selfcheck.md",
    "report/report.md", "site/index.html", "site/data",
)  # fmt: skip
# Следы временной папки сессии и старого места инструкции — в установленных файлах их быть не должно.
TEMP_MARKERS = ("AppData", "scratchpad", "wf1/", "wf2/", "criteria.md", ".claude/agents/README.md")
# Устаревшие формулировки, которые противоречат разделу 5 скила.
VALUE_FIXED = "запрет «о туристах» снят: смысл value установлен (скил, раздел 5, п. 15)"
STALE = {
    r"пока не решён вопрос о смысле": VALUE_FIXED,
    r"туристы без ответа": VALUE_FIXED,
    r"медиан\w* 0,40": "0,40 — сезонность, завышенная трендом МО; верное число — скил, раздел 5, п. 5",
}
NBSP = str.maketrans({"\u00a0": " ", "\u202f": " "})  # неразрывные пробелы → обычные
TYPO = (
    ("прямые кавычки", r'"'),
    ("дефис вместо тире", r"\S\s-\s"),
    (
        "десятичная точка",
        # номера пунктов («п. 6.1», «разд. 5.2»), лицензии, версии и даты вида 04.10 — не десятичные дроби
        r"(?<!п\. )(?<!разд\. )(?<!, )(?<!BY )(?<!BY-SA )(?<![\w.:])(?!\d\d\.\d\d\b)\d+\.\d+(?![\w.])(?!\.)",
    ),
    ("три точки", r"\.\.\."),
)
PATH_RX = re.compile(r"`((?:src|tests|configs|docs|report|site|\.claude)/[^`\s*<>…]+)`")


def split_front(path: pathlib.Path) -> tuple[dict, str]:
    text = path.read_text(encoding="utf-8")
    m = re.match(r"^---\n(.*?)\n---\n(.*)$", text, re.S)
    if not m:
        raise ValueError("нет фронтматтера")
    return yaml.safe_load(m.group(1)), m.group(2)


def scale_rows() -> tuple[dict[int, str], dict[int, str]]:
    """Строки шкалы из раздела 6 скила: как их цитируют судьи (без колонки веса) и веса."""
    rows, weights = {}, {}
    for line in SKILL.read_text(encoding="utf-8").split("\n"):
        m = re.match(r"^\| (\d) \| (.*) \|$", line)
        if not m:
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 7:  # №, критерий, вес, четыре уровня
            n = int(cells[0])
            rows[n] = "| " + " | ".join(cells[:2] + cells[3:]) + " |"
            weights[n] = cells[2]
    return rows, weights


def check_paths(name: str, body: str, problems: list[str]) -> None:
    for p in PATH_RX.findall(body):
        if "{" in p or p.startswith(PLANNED):
            continue
        if not (REPO / p).exists():
            problems.append(f"{name}: нет пути {p}")


def check_typography(name: str, body: str, problems: list[str]) -> None:
    prose = re.sub(r"```.*?```", "", body, flags=re.S)
    prose = re.sub(r"`[^`]*`", "", prose).translate(NBSP)
    for label, rx in TYPO:
        hits = [line.strip()[:70] for line in prose.split("\n") if re.search(rx, line)]
        if hits:
            problems.append(f"{name}: {label}: {hits[:3]}")


def check_text(name: str, text: str, problems: list[str]) -> None:
    for marker in TEMP_MARKERS:
        if marker in text:
            problems.append(f"{name}: ссылка на временную папку или старый путь ({marker})")
    flat = text.translate(NBSP)
    for rx, why in STALE.items():
        if re.search(rx, flat):
            problems.append(f"{name}: устаревшая формулировка «{rx}» — {why}")


def main() -> int:
    problems: list[str] = []
    warnings: list[str] = []
    skills = {p.parent.name for p in SKILLS.glob("*/SKILL.md")}
    rows, weights = scale_rows()
    if sorted(rows) != [1, 2, 3, 4, 5, 6]:
        problems.append(f"SKILL.md: в разделе 6 не шесть строк шкалы ({sorted(rows)})")
    if not HOOK.exists():
        problems.append(f"нет файла хука {HOOK.relative_to(REPO)}")

    report = []
    files = sorted(AGENTS.glob("*.md"))
    for f in files:
        try:
            fm, body = split_front(f)
        except (ValueError, yaml.YAMLError) as e:
            problems.append(f"{f.name}: фронтматтер не читается: {e}")
            continue
        if fm.get("name") != f.stem:
            problems.append(f"{f.name}: name ≠ имя файла")
        # Основная сессия выбирает агента по описанию: в нём должно быть сказано, когда его не звать.
        if "Не зови" not in str(fm.get("description", "")).translate(NBSP):
            problems.append(f"{f.name}: в description нет «Не зови …» (когда агент не нужен)")
        extra = set(fm) - FIELDS
        if extra:
            problems.append(f"{f.name}: неизвестные поля {sorted(extra)}")
        for t in (x.strip() for x in re.findall(r"Agent\([^)]*\)|[^,\s][^,]*", fm.get("tools", ""))):
            if t.startswith("Agent("):
                continue
            if t.startswith("mcp__Claude_Browser__"):
                if t.split("__")[-1] not in BROWSER:
                    problems.append(f"{f.name}: неизвестный браузерный инструмент {t}")
            elif t not in BASE_TOOLS:
                problems.append(f"{f.name}: неизвестный инструмент {t}")
        for s in fm.get("skills", []):
            if s not in skills:
                problems.append(f"{f.name}: скил {s} не найден в .claude/skills")

        if f.stem.startswith(("judge-", "check-", "devils-")):
            if "Write" in fm["tools"] or "Edit" in fm["tools"]:
                problems.append(f"{f.name}: у проверяющего есть запись")
            for t in ("Write", "Edit"):
                if t not in fm.get("disallowedTools", ""):
                    problems.append(f"{f.name}: {t} не в disallowedTools")
            if not fm.get("maxTurns") or not 30 <= fm["maxTurns"] <= 60:
                problems.append(f"{f.name}: maxTurns вне 30–60")
            if fm.get("model") == "haiku" or "memory" in fm:
                problems.append(f"{f.name}: haiku или memory у проверяющего")
            try:
                cmds = [h["command"] for h in fm["hooks"]["PreToolUse"][0]["hooks"]]
                matcher = fm["hooks"]["PreToolUse"][0]["matcher"]
            except (KeyError, IndexError, TypeError):
                cmds, matcher = [], None
            if matcher != "Bash" or HOOK_COMMAND not in cmds:
                problems.append(
                    f"{f.name}: нет хука readonly_guard на Bash с командой из validate_agents.HOOK_COMMAND"
                )

        mj = re.match(r"judge-c(\d)", f.stem)
        if mj:
            n = int(mj.group(1))
            if rows.get(n) and rows[n] not in body:
                problems.append(f"{f.name}: строка шкалы критерия {n} не совпадает с разделом 6 скила")
            flat = body.translate(NBSP)
            for w in (
                "плохо — 0",
                "удовлетворительно — 1",
                "хорошо — 2",
                "отлично — 3",
                "Формат ответа",
                "запуском",
            ):
                if w not in flat:
                    problems.append(f"{f.name}: нет «{w}»")
        if f.stem == "judge-council":
            for n, w in weights.items():
                if not re.search(rf"^\| {n} \| [^|]+ \| {re.escape(w)} \|", body, re.M):
                    problems.append(f"{f.name}: вес критерия {n} не равен {w} из скила")

        check_text(f.name, f.read_text(encoding="utf-8"), problems)
        check_paths(f.name, body, problems)
        check_typography(f.name, body, problems)
        report.append(
            (f.stem, fm.get("model"), fm.get("maxTurns"), len(fm["description"].split()), len(body.split()))
        )

    for doc in (SKILL, GUIDE):
        if not doc.exists():
            problems.append(f"нет файла {doc.relative_to(REPO)}")
            continue
        text = doc.read_text(encoding="utf-8")
        name = str(doc.relative_to(REPO))
        check_text(name, text, problems)
        check_paths(name, text, problems)
        check_typography(name, re.sub(r"^---\n.*?\n---\n", "", text, flags=re.S), problems)
    if "## 12. Правила анализа" not in SKILL.read_text(encoding="utf-8"):
        problems.append("SKILL.md: нет раздела 12 «Правила анализа» (на него ссылаются агенты)")
    try:
        split_front(SKILL)
    except (ValueError, yaml.YAMLError) as e:
        problems.append(f"SKILL.md: фронтматтер не читается: {e}")

    # Порядок этапов в скиле и в cli.py (пока пакет F не добавил panel и eda, это предупреждение).
    order = re.search(r"порядок: `([a-z ]+)`", SKILL.read_text(encoding="utf-8"))
    stages = (
        re.findall(r'^\s+"([a-z]+)": \("munnet\.', CLI.read_text(encoding="utf-8"), re.M)
        if CLI.exists()
        else []
    )
    if order and stages and order.group(1).split() != stages:
        warnings.append(
            f"порядок этапов в скиле ({order.group(1)}) ≠ STAGES в src/munnet/cli.py ({' '.join(stages)})"
        )

    for name, model, turns, dw, bw in report:
        print(f"{name:26} {model!s:7} maxTurns={turns!s:4} description={dw:3} слов  промпт={bw:4} слов")
    print("агентов:", len(report))
    for w in warnings:
        print("предупреждение:", w)
    print("ПРОБЛЕМЫ:" if problems else "проблем нет", *problems, sep="\n  ")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
