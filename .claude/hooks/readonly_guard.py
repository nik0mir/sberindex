"""Хук PreToolUse для проверяющих агентов (judge-*, check-*, devils-advocate) из .claude/agents.

Не даёт через Bash менять репозиторий: git-команды с записью, запуск этапов munnet с конфигом по умолчанию
или этапов data и all, перенаправления и rm/mv/cp/sed -i/tee в пути репозитория, установку пакетов,
ruff --fix. Всё, что пишет во временную папку ($T, $TMP, /tmp, AppData/Local/Temp), разрешено.

Протокол Claude Code: JSON вызова приходит на stdin; код выхода 2 блокирует вызов, а текст из stderr
возвращается агенту как причина. Любая внутренняя ошибка хука вызов не блокирует (код 0).
Только стандартная библиотека: хук запускается до каждого вызова Bash и должен быть быстрым.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import sys

# Переменные, которые указывают в сам репозиторий. Любую другую переменную ($T, $TMP и т. п.) хук считает
# временной папкой: по договорённости из скила munnet-agents проверяющие пишут только в T=$(mktemp -d).
REPO_VARS = {"SRC", "PWD", "REPO", "ROOT", "CLAUDE_PROJECT_DIR", "HOME_REPO"}
TEMP_MARKERS = ("/tmp/", "/dev/", "appdata/local/temp", "/var/folders/", "/private/var/", "%temp%")

GIT_READONLY = {
    "status",
    "log",
    "diff",
    "show",
    "rev-parse",
    "ls-files",
    "ls-tree",
    "blame",
    "grep",
    "describe",
    "cat-file",
    "shortlog",
    "clone",
    "--version",
    "help",
}
MUTATING_CMDS = {"rm", "rmdir", "mv", "cp", "touch", "mkdir", "tee", "truncate", "ln", "chmod", "install"}
DEST_ONLY = {"mv", "cp", "ln", "install"}  # проверяется только последний аргумент (куда пишем)
UV_MUTATING = {"add", "remove", "lock", "sync", "pip", "tool", "venv", "init"}


def norm(path: str) -> str:
    """Приводит путь Windows, MSYS и POSIX к виду c:/users/... в нижнем регистре."""
    p = path.strip().strip("\"'").replace("\\", "/").lower()
    m = re.match(r"^/([a-z])/(.*)$", p)
    if m:
        p = f"{m.group(1)}:/{m.group(2)}"
    return p.rstrip("/")


def strip_heredocs(cmd: str) -> str:
    """Убирает тела heredoc: код Python внутри них хук не разбирает."""
    lines = cmd.split("\n")
    out, delim = [], None
    for line in lines:
        if delim is not None:
            if line.strip() == delim:
                delim = None
            continue
        m = re.search(r"<<-?\s*['\"]?([A-Za-z_][A-Za-z0-9_]*)['\"]?", line)
        if m:
            delim = m.group(1)
            line = line[: m.start()]
        out.append(line)
    return "\n".join(out)


def split_segments(cmd: str) -> list[str]:
    """Режет команду на простые команды по ;, &&, ||, | и переводам строки (вне кавычек)."""
    segs, buf, quote, i = [], [], None, 0
    while i < len(cmd):
        ch = cmd[i]
        if quote:
            buf.append(ch)
            if ch == quote:
                quote = None
        elif ch in "\"'":
            quote = ch
            buf.append(ch)
        elif cmd.startswith("&&", i) or cmd.startswith("||", i):
            segs.append("".join(buf))
            buf = []
            i += 1
        elif ch in ";|\n" and not cmd.startswith(">|", i - 1):
            segs.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
        i += 1
    segs.append("".join(buf))
    return [s.strip() for s in segs if s.strip()]


def tokens(seg: str) -> list[str]:
    try:
        return shlex.split(seg, posix=True)
    except ValueError:
        return seg.split()


def target_state(path: str, project: str, cwd_safe: bool) -> str:
    """Возвращает 'safe', 'repo' или 'other' для пути, в который пишут."""
    raw = path.strip().strip("\"'")
    if not raw or raw.startswith("&"):
        return "safe"
    m = re.match(r"^\$\{?([A-Za-z_][A-Za-z0-9_]*)", raw)
    if m:
        name = m.group(1)
        if name in REPO_VARS:
            return "repo"
        return "safe"  # $T, $TMP и прочие переменные считаем временными
    if raw.startswith("$("):
        return "safe"
    p = norm(raw)
    if any(mark in p + "/" for mark in TEMP_MARKERS) or p in ("/dev/null", "nul"):
        return "safe"
    is_abs = bool(re.match(r"^[a-z]:/", p)) or p.startswith("/") or p.startswith("~")
    if is_abs:
        return "repo" if project and (p == project or p.startswith(project + "/")) else "other"
    return "safe" if cwd_safe else "repo"


def check_git(tok: list[str]) -> str | None:
    i = 1
    while i < len(tok) and tok[i].startswith("-"):
        if tok[i] in ("-C", "-c", "--git-dir", "--work-tree"):
            i += 1
        i += 1
    sub = tok[i] if i < len(tok) else "--version"
    if sub not in GIT_READONLY:
        return f"git {sub} меняет репозиторий; можно только status, log, diff, show, ls-files и т. п."
    return None


def check_munnet(seg: str, tok: list[str]) -> str | None:
    if "--help" in tok or "-h" in tok:
        return None
    cfg = None
    for j, t in enumerate(tok):
        if t == "--config" and j + 1 < len(tok):
            cfg = tok[j + 1]
        elif t.startswith("--config="):
            cfg = t.split("=", 1)[1]
    if cfg is None:
        return "этап munnet без --config пишет в data/, outputs/ и docs/eda.md: нужен временный конфиг"
    c = norm(cfg)
    if c.startswith("configs/") or "/configs/" in c:
        return "конфиг из configs/ пишет в репозиторий: нужен временный конфиг (см. «Перезапуск без следов»)"
    after = tok[tok.index("munnet") + 1 :] if "munnet" in tok else tok
    if "data" in after or "all" in after:
        return "этапы data и all проверяющим запрещены: data/raw только читается"
    return None


def is_munnet_run(tok: list[str]) -> bool:
    """Запуск пакета: python -m munnet …, munnet …, uv run [опции] munnet …"""
    if any(tok[j] == "-m" and tok[j + 1] == "munnet" for j in range(len(tok) - 1)):
        return True
    if os.path.basename(tok[0]).removesuffix(".exe") == "munnet":
        return True
    if tok[0] == "uv" and "run" in tok:
        rest = [t for t in tok[tok.index("run") + 1 :] if not t.startswith("-")]
        return bool(rest) and rest[0] == "munnet"
    return False


REDIRECT_QUOTED = re.compile(r"(?<![<>&0-9])(?:[0-9]|&)?>>?\|?\s*(\"[^\"]*\"|'[^']*')")
REDIRECT_BARE = re.compile(r"(?<![<>&0-9])(?:[0-9]|&)?>>?\|?\s*([^\s;|&<>\"']+)")


def check(cmd: str, project: str) -> str | None:
    body = strip_heredocs(cmd)
    cwd_safe = False
    for seg in split_segments(body):
        tok = tokens(seg)
        # Префиксы вида VAR=value перед командой.
        while tok and re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", tok[0]):
            tok = tok[1:]
        if not tok:
            continue
        name = os.path.basename(tok[0])
        if name == "cd":
            dest = tok[1] if len(tok) > 1 else "~"
            cwd_safe = target_state(dest, project, False) == "safe" or cwd_safe and not dest.startswith("/")
            continue
        # Перенаправления в файлы: цель в кавычках ищем в исходной строке, без кавычек — в строке,
        # где содержимое кавычек стёрто (иначе «>» внутри python -c "…" дал бы ложное срабатывание).
        unquoted = re.sub(r"\"[^\"]*\"|'[^']*'", '""', seg)
        targets = [m.group(1) for m in REDIRECT_QUOTED.finditer(seg)]
        targets += [m.group(1) for m in REDIRECT_BARE.finditer(unquoted)]
        for t in targets:
            if target_state(t, project, cwd_safe) == "repo":
                return f"перенаправление вывода в файл репозитория ({t}): пиши только во временную папку"
        if name == "git":
            reason = check_git(tok)
            if reason:
                return reason
            if "clone" in tok:
                dest = [t for t in tok[tok.index("clone") + 1 :] if not t.startswith("-")]
                if len(dest) >= 2 and target_state(dest[-1], project, cwd_safe) == "repo":
                    return "git clone внутрь репозитория запрещён: клонируй во временную папку"
            continue
        if is_munnet_run(tok):
            reason = check_munnet(seg, tok)
            if reason:
                return reason
        if name == "uv" and len(tok) > 1 and tok[1] in UV_MUTATING:
            # uv sync и uv venv разрешены только во временной копии (check-repro): после cd в неё
            # или с --project/--directory, указывающим во временную папку.
            where = [tok[j + 1] for j in range(len(tok) - 1) if tok[j] in ("--project", "--directory")]
            in_temp = cwd_safe or any(target_state(w, project, False) == "safe" for w in where)
            if not (tok[1] in ("sync", "venv") and in_temp):
                return f"uv {tok[1]} меняет окружение или uv.lock репозитория"
        is_pip = name.removesuffix(".exe") in ("pip", "pip3") or (
            name.startswith("python") and "pip" in tok and "install" in tok
        )
        if is_pip and "install" in tok:
            # pip из временного venv ($T/venv/…) или после cd во временную копию — можно.
            if not (cwd_safe or target_state(tok[0], project, False) == "safe" and tok[0].startswith("$")):
                return "установка пакетов в окружение репозитория запрещена"
        if "ruff" in tok:
            k = tok.index("ruff")
            rest = tok[k + 1 :]
            if "--fix" in rest or (rest[:1] == ["format"] and "--check" not in rest and "--diff" not in rest):
                return "ruff --fix и ruff format без --check правят файлы"
        if name == "sed" and any(t == "-i" or t.startswith("-i") for t in tok[1:]):
            files = [t for t in tok[1:] if not t.startswith("-")][1:]
            if any(target_state(f, project, cwd_safe) == "repo" for f in files):
                return "sed -i по файлам репозитория запрещён"
        if name in MUTATING_CMDS:
            args = [t for t in tok[1:] if not t.startswith("-")]
            if name in DEST_ONLY:
                args = args[-1:]
            for a in args:
                if target_state(a, project, cwd_safe) == "repo":
                    return f"{name} {a}: запись в репозиторий запрещена, работай во временной папке"
    return None


def main() -> int:
    try:
        # В Windows без этого кириллица в причине блокировки превращается в кракозябры.
        sys.stderr.reconfigure(encoding="utf-8")
        data = json.load(sys.stdin)
        if data.get("tool_name") != "Bash":
            return 0
        cmd = (data.get("tool_input") or {}).get("command") or ""
        project = norm(os.environ.get("CLAUDE_PROJECT_DIR") or data.get("cwd") or os.getcwd())
        reason = check(cmd, project)
    except Exception:  # хук не должен ломать работу агента
        return 0
    if reason:
        print(
            f"Заблокировано хуком readonly_guard: {reason}. Проверяющий агент не меняет репозиторий.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
