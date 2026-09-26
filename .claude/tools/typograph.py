"""Типографика ru-text для Markdown: только вне фронтматтера, блоков кода и `инлайн-кода`. Идемпотентна.

Запуск из корня репозитория (аргументы — пути или шаблоны glob относительно текущей папки):

    PYTHONIOENCODING=utf-8 uv run --frozen python .claude/tools/typograph.py ".claude/agents/*.md"
"""

import pathlib
import re
import sys

NB, NNB = "\u00a0", "\u202f"  # неразрывный и узкий неразрывный пробелы
UNITS = r"(₽|МО|км|млн|тыс\.|МБ|ГБ|px|минут|часов|строк|слов|команд|агентов|судей|месяц\w*|лет|дней)"


def fix(t: str) -> str:
    for _ in range(2):  # «и в» — два однобуквенных подряд
        t = re.sub(r"(?<![\w\-])([вксоуиаяВКСОУИАЯ]) (?=\S)", lambda m: m.group(1) + NB, t)
    t = re.sub(r"(?<=[^\s|]) — ", NB + "— ", t)
    t = re.sub(r"(?<=\d) (?=\d{3}(?!\d))", NNB, t)
    t = re.sub(r"(?<=\d) (?=" + UNITS + r")", NB, t)
    t = re.sub(r"\b(т\.|п\.) (?=(д\.|е\.|п\.))", lambda m: m.group(1) + NB, t)
    t = re.sub(r"(?<=\bп\.) (?=\d)", NB, t)
    t = re.sub(r"(?<=№) ", NB, t)
    return t.replace("...", "…")


def process(path: pathlib.Path) -> int:
    s = path.read_text(encoding="utf-8")
    head, body = "", s
    m = re.match(r"^(---\n.*?\n---\n)(.*)$", s, re.S)
    if m:
        head, body = m.group(1), m.group(2)
    out, fence = [], False
    for line in body.split("\n"):
        if line.strip().startswith("```"):
            fence = not fence
            out.append(line)
            continue
        if fence or re.match(r"^\| \d \| ", line):  # строки шкалы жюри — дословно, как в Положении
            out.append(line)
            continue
        parts = re.split(r"(`[^`]*`)", line)
        out.append(
            "".join(p if p.startswith("`") and p.endswith("`") and len(p) > 1 else fix(p) for p in parts)
        )
    new = head + "\n".join(out)
    if new != s:
        path.write_text(new, encoding="utf-8")
    return sum(a != b for a, b in zip(s.split("\n"), new.split("\n"), strict=False))


for arg in sys.argv[1:]:
    for p in sorted(pathlib.Path().glob(arg)):
        print(f"{p}: изменено строк {process(p)}")
