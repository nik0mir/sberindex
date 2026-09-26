"""Отчёт разведки (пакет R): поля фактов, условия и директивы шаблона, типографика, картинки, docs/eda.md."""

import copy
import io
import math
import re
from dataclasses import replace

import numpy as np
import pytest
from PIL import Image
from synth import make_config, make_processed

import munnet.eda as eda
from munnet.cli import EXIT_OK, main
from munnet.eda import report
from munnet.eda.base import SECTION_IDS, EdaCheckError, FigureRecord, make_fact
from munnet.eda.data import load

NB = " "

# Разделы отчёта в порядке spec_final, В.6.
HEADINGS = [
    "# Разведка данных: траты жителей",
    "## Главное за минуту",
    "## Какой сюжет выбрать",
    "## 1. Что есть в данных",
    "## 2. Сколько тратят",
    "## 3. Годовой ритм",
    "## 4. Корзина и маркетплейсы",
    "## 5. Экономика места",
    "## 6. Регион или место",
    "## 7. Что это значит для этапа 2",
    "## Ограничения и ловушки данных",
    "## Как пересобрать и проверить",
    "## Словарь",
]


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """Синтетика: разделы E1–E5 и сводка (как ``--only e1…e5``, затем сводка без записи отчёта)."""
    root = tmp_path_factory.mktemp("report")
    make_processed(root)
    cfg = make_config(root)
    data = load(cfg)
    findings = eda.run_sections(cfg, list(SECTION_IDS), data=data)
    syn = eda.run_synthesis(cfg, data, findings)
    return {"cfg": cfg, "data": data, "findings": findings, "syn": syn, "root": root}


def facts_of(values: dict) -> dict:
    return {k: make_fact(k, v, kind) for k, (v, kind) in values.items()}


# --- Поля фактов и условия -----------------------------------------------------------------------


def test_fill_replaces_fields_and_rejects_unknown():
    facts = facts_of({"e3.reliable_share": (0.156, "pct"), "syn.n_mo": (2190, "int")})
    assert report.fill("у {{e3.reliable_share}} из {{ syn.n_mo }} МО", facts) == "у 15,6% из 2190 МО"
    with pytest.raises(KeyError, match="e3.unknown, e9.x"):
        report.fill("{{e3.unknown}} и {{e9.x}}", facts)
    with pytest.raises(ValueError, match="незаполненное поле"):
        report.fill("{{e3.reliable_share}} и {{ опечатка }}", facts)


def test_conditions_keep_or_drop_blocks():
    text = "A\n<!-- if e1 -->\n- один\n<!-- endif -->\n<!-- if e1 e3 -->\n- два\n<!-- endif -->\nB\n"
    assert report.apply_conditions(text, {"e1", "e3"}) == "A\n- один\n- два\nB\n"  # плотный список
    assert report.apply_conditions(text, {"e1"}) == "A\n- один\nB\n"
    assert report.apply_conditions(text, set()) == "A\nB\n"
    with pytest.raises(ValueError, match="вложенное"):
        report.apply_conditions("<!-- if e1 --><!-- if e2 -->x<!-- endif --><!-- endif -->", {"e1"})
    with pytest.raises(ValueError, match="без «endif»"):
        report.apply_conditions("<!-- if e1 -->x", {"e1"})
    with pytest.raises(ValueError, match="неверный список"):
        report.apply_conditions("<!-- if E1 -->x<!-- endif -->", {"e1"})


def test_unwrap_and_nbsp_in_template_text():
    text = (
        "Абзац в\nдве строки — с тире.\n- пункт и\n  продолжение\n- второй пункт\n\n```\nкод в\nблоке\n```\n"
    )
    out = report.nbsp_markdown(report.unwrap_paragraphs(text))
    lines = out.split("\n")
    assert lines[0] == f"Абзац в{NB}две строки{NB}— с{NB}тире."
    assert lines[1] == f"- пункт и{NB}продолжение" and lines[2] == "- второй пункт"
    assert "код в\nблоке" in out  # код не склеивается и не меняется
    assert report.nbsp_markdown("см. рис. 4 и т. е.") == f"см.{NB}рис.{NB}4 и{NB}т.{NB}е."


def test_nbsp_in_table_cells_but_not_code_or_empty_cells():
    row = "| в МО и — город | — | `a и b` |"
    assert report.nbsp_markdown(row) == f"| в{NB}МО и{NB}— город | — | `a и b` |"
    assert report.nbsp_markdown("|---|---:|") == "|---|---:|"
    # перед кодом тоже: «в `файле`», а внутри кода ничего не меняется
    assert report.nbsp_markdown("лежит в `a и b` — тут") == f"лежит в{NB}`a и b`{NB}— тут"


# --- Типографика ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("line", "what"),
    [
        ('Посёлок "Новый"', "прямая кавычка"),
        ("траты - это расходы", "дефис вместо тире"),
        ("доля 12.5 процента", "десятичная точка"),
        ("и так далее...", "три точки"),
        ("два  пробела", "двойной пробел"),
    ],
)
def test_lint_ru_catches(line, what):
    problems = report.lint_ru(f"# Заголовок\n\n{line}\n")
    assert len(problems) == 1 and problems[0].startswith("строка 3:") and what in problems[0]


def test_lint_ru_ignores_code_paths_links_and_allowed():
    md = "\n".join(
        [
            "```bash",
            'python -m munnet "eda" - 1.5 ...  x',
            "```",
            'Команда `uv run "x" - 2.5...` и файл docs/img/eda/F01_map.png, CSV outputs/eda/F01_v1.2.csv.',
            "- пункт списка; ссылка [описание](https://sberindex.ru/a.b-c/d.1.2) и https://x.org/v1.5",
            "1. нумерованный пункт",
            "Данные скачаны 25.09.2026; лицензии CC BY-SA 4.0 и CC BY 4.0.",
            "| a | b |  | c |",
            "<!-- комментарий 1.5 - x -->",
        ]
    )
    assert report.lint_ru(md) == []
    assert report.lint_ru("CC BY 4.0", allow=[]) != []  # без разрешённых образцов точка ловится


def test_typograph_data_strings():
    assert report.typograph('поселение Мосрентген""') == "поселение Мосрентген"
    assert report.typograph('посёлок "Новый" и ...') == "посёлок «Новый» и …"
    assert report.typograph('код `a "b"...`') == 'код `a "b"...`'


# --- Перекрёстные ссылки и числа -----------------------------------------------------------------


def test_link_refs_links_figures_and_tables():
    text = "\n".join(
        [
            f"Текст (рис. 4, 5) и рис.{NB}11 и 12, таблица T06 и T13, рис. 99.",
            "`рис. 4` и [рис. 4](#fig-4) не трогаются; файл `T06_x.csv`, путь outputs/T06",
            "## Заголовок рис. 4",
            "| T06 | рис. 4 |",
            "![альт рис. 4](img.png)",
            f"{report.anchor_tag('tab-t06')}**Таблица T06.** Подпись",
        ]
    )
    out = report.link_refs(text, ["F04", "F05", "F11", "F12"], ["T06"]).split("\n")
    assert out[0] == (
        f"Текст ([рис. 4](#fig-4), [5](#fig-5)) и [рис.{NB}11](#fig-11) и [12](#fig-12), "
        "[таблица T06](#tab-t06) и T13, рис. 99."
    )
    assert (
        out[1:] == text.split("\n")[1:]
    )  # код, готовые ссылки, пути, заголовки, таблицы, картинки — как были


def test_check_links_finds_missing_images_and_anchors(tmp_path):
    (tmp_path / "img").mkdir()
    (tmp_path / "img" / "a.png").write_bytes(b"png")
    md = "\n".join(
        [
            "# Отчёт",
            f"## 2. Сколько{NB}тратят",
            f"{report.anchor_tag('fig-1')}![a](img/a.png)",
            "![b](img/b.png)",
            "[рис. 1](#fig-1), [рис. 2](#fig-2), [раздел](#2-сколько-тратят)",
        ]
    )
    assert report.check_links(md, tmp_path) == [
        "нет файла картинки img/b.png",
        "ссылка на несуществующий якорь #fig-2",
    ]


def test_stray_numbers_flags_numbers_not_from_facts():
    facts = facts_of({"e1.n": (2190, "int"), "e3.s": (0.156, "pct"), "e2.rub": (26561, "rub")})
    md = "\n".join(
        [
            "Траты есть по 2190 МО, доля 15,6%, 26 561 ₽ в 2024 году; 90-й перцентиль, баллы 1–5.",
            "Ещё 42 МО и 0,5 угадывания.",
            "| 777 | таблица |",
            "```",
            "999",
            "```",
            "[рис. 4](#fig-4), код `T 55`, путь outputs/x/7.csv",
            "*Рисунок 3. n = 2048.*",
        ]
    )
    assert report.stray_numbers(md, facts) == [
        "строка 2: число 0,5 не из фактов",
        "строка 2: число 42 не из фактов",
    ]
    assert (
        report.stray_numbers("AUC 0,73 (0,5 — не лучше угадывания)", facts_of({"e1.a": (0.73, "num2")})) == []
    )


# --- Картинки ------------------------------------------------------------------------------------


def png_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr.astype("uint8")).save(buf, format="PNG")
    return buf.getvalue()


def record(fid: str, png: str) -> FigureRecord:
    return FigureRecord(fid, "x", "e1", "t", "s", "альт", png, "", "data/x.csv", "проверка")


def test_copy_images_limit_shrink_and_cleanup(tmp_path):
    rng = np.random.default_rng(0)
    src = tmp_path / "src"
    (src / "figures").mkdir(parents=True)
    small = np.full((40, 60, 3), 200)
    gradient = np.linspace(0, 255, 800)[None, :, None] * np.ones((500, 1, 3))
    smooth = gradient + rng.integers(-3, 4, size=(500, 800, 3))  # тысячи цветов, но плавно
    noise = rng.integers(0, 256, size=(500, 800, 3))  # не сжать ничем
    for name, arr in (("F01_small", small), ("F02_smooth", smooth), ("F03_noise", noise)):
        (src / "figures" / f"{name}.png").write_bytes(png_bytes(np.clip(arr, 0, 255)))
    raw = (src / "figures" / "F02_smooth.png").stat().st_size
    shrunk = len(report._quantize_png((src / "figures" / "F02_smooth.png").read_bytes(), 256))
    limit_kb = math.ceil(shrunk / 1024) + 1
    assert raw > limit_kb * 1024  # без сжатия файл не проходит
    dest = tmp_path / "docs" / "img"
    dest.mkdir(parents=True)
    (dest / "F99_old.png").write_bytes(b"old")
    (dest / "readme.txt").write_text("не картинка графика", encoding="utf-8")
    ok = [record("F01", "figures/F01_small.png"), record("F02", "figures/F02_smooth.png")]

    with pytest.raises(ValueError, match="F02"):
        report.copy_images(ok, src, dest, limit_kb, shrink=False)
    assert sorted(p.name for p in dest.iterdir()) == ["F99_old.png", "readme.txt"]  # ничего не скопировано
    with pytest.raises(ValueError, match="F03"):
        report.copy_images([*ok, record("F03", "figures/F03_noise.png")], src, dest, limit_kb)

    paths = report.copy_images(ok, src, dest, limit_kb)
    assert [p.name for p in paths] == ["F01_small.png", "F02_smooth.png"]
    assert all(p.stat().st_size <= limit_kb * 1024 for p in paths)
    assert (dest / "F01_small.png").read_bytes() == (src / "figures" / "F01_small.png").read_bytes()
    assert not (dest / "F99_old.png").exists() and (dest / "readme.txt").exists()
    before = {p.name: p.stat().st_mtime_ns for p in paths}
    report.copy_images(ok, src, dest, limit_kb)
    assert {p.name: p.stat().st_mtime_ns for p in paths} == before  # неизменившиеся не перезаписаны


# --- Сборка отчёта -------------------------------------------------------------------------------


def test_template_has_all_sections_in_order():
    text = report.TEMPLATE.read_text(encoding="utf-8")
    positions = [text.find(h) for h in HEADINGS]
    assert all(p >= 0 for p in positions) and positions == sorted(positions)
    for sid in (*SECTION_IDS, "syn"):
        assert f"<!-- section: {sid} -->" in text


def test_render_report_on_synthetic(built):
    md = report.render_report(built["cfg"], built["findings"], built["syn"])
    flat = md.replace(NB, " ")  # в заголовках тоже неразрывные пробелы после «в», «и»
    lines = flat.split("\n")
    starts = [next(i for i, ln in enumerate(lines) if ln.startswith(h)) for h in HEADINGS]
    assert starts == sorted(starts)
    assert "{{" not in md and "<!--" not in md
    assert report.lint_ru(md) == []
    figures = [r for f in [*built["findings"], built["syn"]] for r in f.figures]
    for r in figures:  # каждый рисунок — ровно один раз, путь относительно docs/
        assert md.count(f"](img/eda/{r.fid}_{r.slug}.png)") == 1
    pos = {key: flat.index(key) for key in ("## Какой сюжет", "**Таблица T14.**", "## 1. Что есть")}
    assert pos["## Какой сюжет"] < pos["**Таблица T14.**"] < pos["## 1. Что есть"]
    assert flat.index("**Таблица T00.**") > flat.index("## Как пересобрать и проверить")
    assert md.count("**Таблица T00.**") == 1  # поставлена отдельно — в блоке раздела e1 её нет
    assert md.count("**Таблица T14.**") == 1 and md.count("**Таблица T13.**") == 1
    assert built["syn"].facts["syn.verdict"].text in md
    tables = [t for f in [*built["findings"], built["syn"]] for t in f.tables]
    for r in figures:  # у каждого рисунка и таблицы — якорь, на который ведут «рис. N» и «таблица TNN»
        assert md.count(report.anchor_tag(report.fig_anchor(r.fid))) == 1
    for t in tables:
        assert md.count(report.anchor_tag(report.tab_anchor(t.tid))) == 1
    assert "](#fig-1)" in md and "](#tab-t14)" in md
    facts = {k: v for f in [*built["findings"], built["syn"]] for k, v in f.facts.items()}
    assert report.stray_numbers(md, facts) == []  # каждое число текста — из фактов


def test_main_points_are_short_with_numbers_and_links(built):
    """В.6: «Главное за минуту» — 5–7 пунктов, в каждом число и ссылка на рисунок или таблицу."""
    md = report.render_report(built["cfg"], built["findings"], built["syn"])
    block = md.split("## Главное за минуту", 1)[1].split("\n## ", 1)[0]
    points = [ln for ln in block.strip().split("\n") if ln.startswith("- ")]
    assert 5 <= len(points) <= 7 and len(points) == len(block.strip().split("\n"))  # только пункты
    for p in points:
        assert re.search(r"\d", p) and re.search(r"\]\(#(?:fig|tab)-", p), p
        assert p.startswith("- **")  # вывод жирным в начале — читается за секунды


def test_nbsp_reaches_section_text_captions_and_tables(built):
    """Неразрывные пробелы ставятся по готовому Markdown: в итогах разделов, оговорках и ячейках таблиц,
    а не только в тексте шаблона."""
    findings = [copy.copy(f) for f in built["findings"]]
    e2 = next(f for f in findings if f.section == "e2")
    e2.summary_md = e2.summary_md + " Центры и периферия — проверка."
    e2.caveats = [*e2.caveats, "Траты в МО с тестом."]
    table = e2.tables[0]
    e2.tables = [replace(table, markdown=table.markdown + "\n| в строке | и ещё |"), *e2.tables[1:]]
    md = report.render_report(built["cfg"], findings, built["syn"])
    assert f"Центры и{NB}периферия{NB}— проверка." in md
    assert f"Траты в{NB}МО с{NB}тестом." in md
    assert f"| в{NB}строке | и{NB}ещё |" in md
    assert report.lint_ru(md) == []


def test_title_directive_inserts_headline_as_plain_text(built):
    cfg, findings, syn = built["cfg"], built["findings"], built["syn"]
    title = syn.figures[0].title
    md = report.render_report(cfg, findings, syn, template="**Раздел. <!-- title: F16 -->.**\n")
    assert md.replace(NB, " ").strip() == f"**Раздел. {title.replace(NB, ' ').rstrip('.')}.**"
    assert "**" + title not in md  # не жирная подводка, а часть фразы


def test_missing_section_gets_stub_not_error(built):
    findings = [f for f in built["findings"] if f.section != "e3"]
    md = report.render_report(built["cfg"], findings, built["syn"])
    assert "Раздел не собран: нет итога раздела e3" in md
    assert "{{" not in md and "**Таблица T05.**" not in md and "F06_national_rhythm" not in md
    assert report.lint_ru(md) == []


def test_render_report_rejects_unknown_items(built):
    cfg, findings, syn = built["cfg"], built["findings"], built["syn"]
    with pytest.raises(ValueError, match="рисунок F42"):
        report.render_report(cfg, findings, syn, template="<!-- figure: F42 -->")
    with pytest.raises(ValueError, match="таблица T42"):
        report.render_report(cfg, findings, syn, template="<!-- table: T42 -->")
    with pytest.raises(ValueError, match="неизвестная директива"):
        report.render_report(cfg, findings, syn, template="<!-- figur: F01 -->")
    with pytest.raises(KeyError, match="e1.nope"):
        report.render_report(cfg, findings, syn, template="{{e1.nope}}")


def test_write_report_is_idempotent_and_guarded(built):
    cfg, findings, syn = built["cfg"], built["findings"], built["syn"]
    path = report.write_report(cfg, findings, syn)
    first = path.read_bytes()
    images = sorted(p.name for p in (built["root"] / "docs" / "img" / "eda").iterdir())
    n_figures = sum(len(f.figures) for f in [*findings, syn])
    assert len(images) == n_figures and all(re.fullmatch(r"F\d{2}_[a-z0-9_]+\.png", n) for n in images)
    limit = cfg["eda"]["figure"]["docs_max_kb"] * 1024
    assert all((built["root"] / "docs" / "img" / "eda" / n).stat().st_size <= limit for n in images)
    assert report.write_report(cfg, findings, syn).read_bytes() == first  # повторная сборка — тот же файл
    assert report.check_links(first.decode("utf-8"), path.parent) == []  # все картинки и якоря на месте

    broken = copy.copy(findings[0])
    broken.summary_md = findings[0].summary_md + ' Посёлок "Новый".'
    with pytest.raises(EdaCheckError, match="прямая кавычка"):
        report.write_report(cfg, [broken, *findings[1:]], syn)
    assert path.read_bytes() == first  # при замечаниях отчёт не перезаписан
    broken.summary_md = findings[0].summary_md + " См. [раздел](#нет-такого)."
    with pytest.raises(EdaCheckError, match="нет-такого"):
        report.write_report(cfg, [broken, *findings[1:]], syn)
    assert path.read_bytes() == first


def test_full_run_through_cli(tmp_path):
    """Полный прогон этапа eda на синтетике командой: разделы → сводка → общие выходы → docs/eda.md."""
    import yaml

    make_processed(tmp_path)
    cfg = make_config(tmp_path)
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text(yaml.safe_dump(cfg.data, allow_unicode=True), encoding="utf-8")
    assert main(["--config", str(config_path), "eda"]) == EXIT_OK
    md = (tmp_path / "docs" / "eda.md").read_text(encoding="utf-8")
    assert md.startswith("# Разведка данных: траты жителей 74 муниципалитетов")
    facts = (eda.eda_dir(cfg) / "facts.json").read_text(encoding="utf-8")
    assert '"syn.top_story"' in facts and '"e3.reliable_share"' in facts
