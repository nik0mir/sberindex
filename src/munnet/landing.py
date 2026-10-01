"""Этап site: данные и тексты интерактивного лендинга (``docs/landing_spec.md``, §4–5).

Порядок: входы и их свежесть (код 1) → слепые выходы (код 3) → проверка словаря заголовков и заморозки
(``site_headlines``, код 3) → обе палитры типов ``style.palette_audit`` (код 3) → вердикты и значения полей
из ``facts.json`` этапа interpret → ``story.json`` (тексты слотов по §4, флаги ``view``) → линт текста,
который пишет сам сайт (код 3) → бюджет (код 3) → запись.

Страница ничего не считает: вердикты, тексты и числа берутся из файлов этапов. Значения полей ``{…}`` в
заголовках — те, что подставил этап 5: они извлекаются из его текстов исходов по шаблонам предрегистрации
(``interpret.tests.<тест>.outcomes``), а не пересчитываются.

Режимы:
- обычный — ``outputs/interpret``, пишет в ``site.build.out``; слепые выходы — код 3, в том числе со стёртыми
  пометками: тип каждого узла в ``types.csv`` сверяется с ``cluster_final`` (``check_bound_to_labels``);
- ``dev_blind`` — каталог слепого прогона ``interpret_blind`` для разработки: пишет только в
  ``site.build.dev_blind_out``, страница помечена «СЛЕПОЙ ПРОГОН»; не слепые выходы здесь — код 3;
- ``demo`` — синтетический ``facts.json`` (``"synthetic": true``): пишет в ``site.demo.out`` с плашкой
  ``site.demo.banner``.

Данные страницы (§5): ``mo.json`` (2190 МО и 2 узла-города колонками), ``types.json``, ``checks.json``,
``methods.json``, ``hexgrid.json`` (карта равных ячеек — ``site_hexgrid``), статичная SVG экрана 0, выгрузки
``data/download/*.csv`` и ``index.html`` по шаблону ``src/munnet/templates/landing.html``. Контрольные числа
(``site.build.controls``: 1776 узлов, 169 МО без типа, 247 районов столиц; ``facts.scope``; размеры типов
``facts.ladder``) сверяются с файлами — расхождение код 3. Выгрузки этапа 5, которых нет (``--demo``), дают
пустые поля: блок скрывается, landing.py ничего не пересчитывает. Пока не собираются ``geo.json`` (режим
«Площадь») и PDF (план §10, пп. 9 и 11).
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from munnet import (
    site_chapters,
    site_chapters_tail,
    site_findings,
    site_hexgrid,
    site_scene,
    site_useful,
    style,
)
from munnet.config import Config
from munnet.contracts import MissingInputError, QCError
from munnet.site_headlines import SLOT, HeadlineChecker, check_headlines, freeze_hash, norm

log = logging.getLogger(__name__)

BLIND_LABEL = "СЛЕПОЙ ПРОГОН"  # как interpret.stage.BLIND_LABEL (не импортируется: тянет весь этап)
BLIND_DIR = "interpret_blind"
# Наборы T7 (как munnet.interpret.labels.SET_NAMES; сверяются с полями текстов этапа 5 при сборке)
SET_NAMES = {
    "A": "сопоставимые территории того же типа",
    "D": "сопоставимые территории",
    "B": "соседи по своему региону",
    "C": "случайные МО того же размера",
}
TESTS = (
    "T1_ladder_external",
    "T2_direction",
    "T3_reliable_placebo",
    "T5_trivial",
    "T6_bank_coverage",
    "T7_utility",
)
LICENSES = [
    {"source": "СберИндекс", "license": "CC BY-SA 4.0"},
    {"source": "Росстат, БД ПМО в обработке «Если быть точным»", "license": "CC BY 4.0"},
    {"source": "ФНС, 5-НДФЛ в обработке «Если быть точным»", "license": "CC BY 4.0"},
]


# --- режимы и входы -----------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Mode:
    """Откуда читать выходы interpret, куда писать и какую плашку ставить."""

    name: str  # normal | dev_blind | demo
    interpret_dir: Path
    out: Path
    banner: str | None


def _outputs_path(cfg: Config, value: str) -> Path:
    """``outputs/<x>`` из конфига — внутри ``paths.outputs`` (работает и с временным конфигом)."""
    p = Path(value)
    if p.parts and p.parts[0] == "outputs":
        return cfg.dir("outputs").joinpath(*p.parts[1:])
    return p


def resolve_mode(cfg: Config, dev_blind: str | Path | None = None, demo: str | Path | None = None) -> Mode:
    site = cfg["site"]
    if dev_blind is not None and demo is not None:
        raise ValueError("site: --dev-blind и --demo вместе не используются")
    if dev_blind is not None:
        return Mode(
            "dev_blind", Path(dev_blind), _outputs_path(cfg, site["build"]["dev_blind_out"]), BLIND_LABEL
        )
    if demo is not None:
        return Mode("demo", Path(demo), _outputs_path(cfg, site["demo"]["out"]), str(site["demo"]["banner"]))
    return Mode("normal", cfg.dir("outputs") / "interpret", Path(site["build"]["out"]), None)


def load_facts(cfg: Config, mode: Mode) -> dict:
    """``facts.json`` этапа interpret с проверками кода 1 (нет, не done, старше меток) и кода 3 (слепой)."""
    path = mode.interpret_dir / "facts.json"
    if not path.exists():
        raise MissingInputError(f"site: нет {path} — сначала этап interpret")
    with open(path, encoding="utf-8") as f:
        facts = json.load(f)
    if facts.get("status") != "done":
        raise MissingInputError(f"site: {path}: status = {facts.get('status')!r}, а не 'done'")
    if mode.name != "demo":
        _check_fresh(cfg, mode.interpret_dir)
    blind = (
        bool(facts.get("blind")) or facts.get("label") == BLIND_LABEL or mode.interpret_dir.name == BLIND_DIR
    )
    if mode.name == "dev_blind" and not blind:
        raise QCError(
            f"site --dev-blind: {path} — не слепой прогон (facts.blind пуст); режим только для слепых"
        )
    if mode.name != "dev_blind" and blind:
        raise QCError(
            f"site: слепые выходы ({path}: blind = {facts.get('blind')!r}) — страница не публикуется"
        )
    if mode.name == "demo" and not facts.get("synthetic"):
        raise QCError(f"site --demo: {path} без пометки synthetic — режим только для синтетики")
    if mode.name != "demo" and facts.get("synthetic"):
        raise QCError(f"site: {path} — синтетический facts.json; он собирается только с --demo")
    if mode.name == "normal":
        check_bound_to_labels(cfg, mode.interpret_dir, facts)
    return facts


def check_bound_to_labels(cfg: Config, interpret_dir: Path, facts: Mapping) -> None:
    """Код 3: выходы interpret посчитаны не на текущих метках cluster или типы в них перемешаны.

    Защита от слепых выходов, у которых стёрты пометки (``blind``, ``label``, имя каталога): слепой прогон
    переставляет типы по узлам с теми же размерами и читает те же файлы, поэтому ни размеры типов,
    ни sha256 входов его не выдают. Выдаёт поузловое сравнение: тип каждого узла в ``types.csv`` должен
    равняться типу ``cluster_final.parquet`` после переноса номеров ``facts.ladder.transfer``.
    sha256 ``cluster_labels`` и ``cluster_final`` из ``facts.qc.inputs_sha256`` сверяются с файлами
    (выходы другого прогона cluster — по содержимому, а не по времени изменения)."""
    processed = Path(cfg["paths"]["processed"])
    files = {k: processed / f"{k}.parquet" for k in ("cluster_labels", "cluster_final")}
    types_csv = interpret_dir / "types.csv"
    missing = [str(p) for p in [*files.values(), types_csv] if not p.exists()]
    if missing:
        raise MissingInputError(f"site: нет {missing} — этапы cluster и interpret")
    sha = (facts.get("qc") or {}).get("inputs_sha256") or {}
    for k, p in files.items():
        have = hashlib.sha256(p.read_bytes()).hexdigest()
        if sha.get(k) != have:
            raise QCError(
                f"site: facts.qc.inputs_sha256.{k} = {sha.get(k)!r} ≠ sha256 {p.name} {have[:12]}… — "
                "выходы interpret посчитаны не на текущих метках; перезапустите interpret"
            )
    transfer = {int(k): int(v) for k, v in ((facts.get("ladder") or {}).get("transfer") or {}).items()}
    fin = pd.read_parquet(files["cluster_final"], columns=["territory_id", "type"])
    if not transfer or set(transfer) != set(fin["type"].astype(int)):
        raise QCError(
            f"site: facts.ladder.transfer {transfer} не покрывает типы cluster_final "
            f"{sorted(set(fin['type'].astype(int)))}"
        )
    want = pd.Series(
        [transfer[int(t)] for t in fin["type"]], index=fin["territory_id"].astype(int), name="want"
    )
    got = pd.read_csv(types_csv, usecols=["territory_id", "type"]).set_index("territory_id")["type"]
    got.index = got.index.astype(int)
    both = pd.concat([want, got.rename("got")], axis=1)
    bad = both[both["want"].isna() | both["got"].isna() | (both["want"] != both["got"])]
    if len(bad):
        raise QCError(
            f"site: тип {len(bad)} из {len(both)} узлов в {types_csv.name} ≠ cluster_final после переноса "
            f"номеров (например, узел {int(bad.index[0])}) — слепые или чужие выходы; страница не публикуется"
        )


def _check_fresh(cfg: Config, interpret_dir: Path) -> None:
    """Код 1: выходы interpret старше ``cluster_labels`` (типы пересчитаны после проверок этапа 5)."""
    labels = Path(cfg["paths"]["processed"]) / "cluster_labels.parquet"
    if not labels.exists():
        raise MissingInputError(f"site: нет {labels} — сначала этап cluster")
    t_lab = labels.stat().st_mtime
    stale = sorted(p.name for p in interpret_dir.glob("*") if p.is_file() and p.stat().st_mtime < t_lab)
    if stale:
        raise MissingInputError(
            f"site: выходы {interpret_dir} старше {labels.name}: {', '.join(stale)} — перезапустите interpret"
        )


def site_numbers(cfg: Config) -> dict[str, str]:
    """Числа для полей текстов, которые пишет сам сайт (глава 2 и роль «Бизнесу»), — из файлов этапов.

    ``share_cross`` — доля рёбер основной сети корзин между регионами; ``share_cross_random`` —
    1 − Σ n_r (n_r − 1) / (N (N − 1)) по регионам узлов сети (комментарий к ``descriptive.similarity``);
    ``n_rhythm`` — узлы с надёжным своим ритмом, доли — ``outputs/eda/facts.json`` (сверяются с n_rhythm)."""
    processed = Path(cfg["paths"]["processed"])
    # правило основной сети — то же, что укладывает раскладка главы 2 (network_edges.<правило>)
    rule = str(cfg["site"]["similarity_layout"]["candidates"]["graph_fr"]["graph"]).split(".", 1)[1]
    need = {n: processed / f"{n}.parquet" for n in ("network_edges", "network_nodes", "features_nodes")}
    need["features_rhythm"] = processed / "features_rhythm.parquet"
    eda_facts = cfg.dir("outputs") / "eda" / "facts.json"
    missing = [str(p) for p in [*need.values(), eda_facts] if not p.exists()]
    if missing:
        raise MissingInputError(f"site: нет входов {missing} — этапы eda, features, network")
    edges = pd.read_parquet(need["network_edges"], columns=["rule", "is_main", "same_region"])
    main = edges[(edges["rule"] == rule) & edges["is_main"]]
    nodes = pd.read_parquet(need["network_nodes"], columns=["territory_id", "rule"])
    ids = nodes.loc[nodes["rule"] == rule, "territory_id"]
    reg = pd.read_parquet(need["features_nodes"], columns=["territory_id", "region_code"])
    n_r = reg[reg["territory_id"].isin(ids)]["region_code"].value_counts()
    big_n = int(n_r.sum())
    rhythm = pd.read_parquet(need["features_rhythm"], columns=["territory_id", "own_reliable"])
    with open(eda_facts, encoding="utf-8") as f:
        ef = json.load(f)
    share = ef["syn.reliable_share_nodes"]
    n_rhythm = int(rhythm["own_reliable"].sum())
    if abs(n_rhythm / len(rhythm) - float(share["value"])) > 1e-9:
        raise QCError(
            f"site: свой ритм надёжен у {n_rhythm} из {len(rhythm)} узлов ≠ syn.reliable_share_nodes "
            f"{share['text']}"
        )
    # «почему регионов меньше 77»: регионы с тратами без единого узла сети (у всех МО ряд неполный) —
    # из файлов; регионы без трат — из разведки (e1.absent_regions); число сверяется с e1
    fn = pd.read_parquet(need["features_nodes"], columns=["region_code", "region_name", "is_node"])
    node_regs = set(fn.loc[fn["is_node"], "region_code"])
    inc = sorted(
        {str(r) for c, r in zip(fn["region_code"], fn["region_name"], strict=True) if c not in node_regs}
    )
    n_inc_eda = (ef.get("e1.n_all_incomplete_regions") or {}).get("value")
    if n_inc_eda is not None and int(n_inc_eda) != len(inc):
        raise QCError(f"site: регионов без узлов {len(inc)} ≠ e1.n_all_incomplete_regions {n_inc_eda}")
    absent = ef.get("e1.n_absent_regions") or {}
    n_abs = int(absent.get("value") or 0)
    n_out = int((ef.get("e1.n_missing_outside") or {}).get("value") or 0)
    out_txt = str((ef.get("e1.missing_outside_regions") or {}).get("text") or "")
    regions = (
        {
            "n_incomplete": f"{len(inc)} {plural_ru(len(inc), 'регионе', 'регионах', 'регионах')}",
            "incomplete": ", ".join(inc),
            "n_absent": f"{n_abs} {plural_ru(n_abs, 'региона', 'регионов', 'регионов')}",
            "absent": str((ef.get("e1.absent_regions") or {}).get("text") or ""),
        }
        | (
            {
                "n_missing_mo": f"{n_out} "
                + plural_ru(n_out, "муниципалитет", "муниципалитета", "муниципалитетов"),
                "missing_outside": out_txt,
            }
            if n_out and out_txt
            else {}
        )
        if inc and absent.get("value")
        else {}
    )
    # глава 7 (site.build.texts.limits): рост трат (номинал) и НДФЛ по месту работы — тексты разведки как есть
    eda_txt = {
        k: str((ef.get(src) or {}).get("text") or "")
        for k, src in (
            ("growth_median", "e4.growth_median"),
            ("growth_p10", "e4.growth_p10"),
            ("growth_p90", "e4.growth_p90"),
            ("n_recip_gt3", "e5.n_recip_gt3"),
        )
    }
    return (
        regions
        | eda_txt
        | {
            "share_cross": style.fmt_pct(1 - float(main["same_region"].mean())),
            "share_cross_random": style.fmt_pct(1 - float((n_r * (n_r - 1)).sum()) / (big_n * (big_n - 1))),
            "n_rhythm": style.fmt_num(n_rhythm),
            "share_rhythm": str(share["text"]),
            "share_rhythm_null": str(ef["syn.null_reliable_share_nodes"]["text"]),
        }
    )


# --- вердикты и поля текстов этапа 5 ------------------------------------------------------------------------


@lru_cache(maxsize=256)
def template_regex(template: str) -> re.Pattern[str]:
    """Шаблон текста исхода -> регулярное выражение с группой на каждое поле (повтор поля — то же значение).

    После шаблона допустимо продолжение через пробел (T7 + type_gain, T2 + place_override, T6 + оговорка)."""
    out, seen = [], set()
    for part in re.split(r"(\{[A-Za-z_0-9]+\})", template):
        m = re.fullmatch(r"\{([A-Za-z_0-9]+)\}", part)
        if m:
            name = m.group(1)
            out.append(f"(?P={name})" if name in seen else f"(?P<{name}>.*?)")
            seen.add(name)
        else:
            out.append(re.escape(part))
    return re.compile("".join(out) + r"(?:\s.*)?\Z", re.S)


def _lower_first(s: str) -> str:
    return s[:1].lower() + s[1:]


def cap(s: str) -> str:
    """Прописная в начале (как ``interpret.texts.cap``: поле в начале текста или заголовка)."""
    return s[:1].upper() + s[1:] if s else s


def plural_ru(n: int, one: str, few: str, many: str) -> str:
    """Форма слова после числа по правилу русского числительного."""
    a, b = abs(int(n)) % 100, abs(int(n)) % 10
    if 10 < a < 20:
        return many
    return one if b == 1 else few if 1 < b < 5 else many


_NBSP_WORD = re.compile(r"(?<![\w\u00ad-])([вксоуиаяВКСОУИАЯ]) ")


def nbsp(s: str) -> str:
    """Типографика ru-text для показа: неразрывный пробел после однобуквенных слов и перед тире.
    Только для отображения: в story.json тексты этапа 5 остаются побуквенно как в facts.json."""
    if not s:
        return s
    s = _NBSP_WORD.sub(NBSP_REPL, str(s))
    return s.replace(" — ", NBSP + "— ")


NBSP = chr(0xA0)
NBSP_REPL = "\\1" + NBSP


def extract_fields(template: str, text: str) -> dict[str, str] | None:
    """Поля, которые этап 5 подставил в ``template``, чтобы получить ``text``; None — не тот шаблон."""
    m = template_regex(_lower_first(template)).match(_lower_first(text))
    return None if m is None else m.groupdict()


@dataclass
class Verdicts:
    """Ключи исходов для словаря заголовков и флаги сочетания (как в переборе ``site_headlines``)."""

    keys: dict[str, Any]  # тест -> ключ заголовка; плюс T7_type_gain, one_in_ten, caveat
    fields: dict[str, str] = field(default_factory=dict)  # поле -> значение, одинаковое во всех текстах
    by_test: dict[str, dict[str, str]] = field(default_factory=dict)  # тест -> поля его текста

    def of(self, test: str) -> dict[str, str]:
        """Поля заголовка теста: из текста его исхода (правило headlines.check.slots: subset_of_text)."""
        return self.fields | self.by_test.get(test, {})

    def __getitem__(self, k: str) -> Any:
        return self.keys[k]


def read_verdicts(cfg: Config, facts: Mapping) -> Verdicts:
    """Вердикты ``facts.verdicts_final`` -> ключи текстов исходов; поля — из ``facts.texts`` по шаблонам.

    T1 — ``T1_text`` (после потолка ``thesis_assembly.t1_cap_by_t5``: именно он сказан в тексте); T2 —
    ``partial_capped``, если текст этапа 5 написан по этому шаблону. Текст, не совпавший с шаблоном своего
    вердикта, — код 3: тексты и вердикты разошлись."""
    it = cfg["interpret"]["tests"]
    vf = facts["verdicts_final"]
    texts = facts["texts"]
    keys: dict[str, Any] = {}
    fields: dict[str, str] = {}
    by_test: dict[str, dict[str, str]] = {}
    clash: set[str] = set()
    for test in TESTS:
        v = vf.get("T1_text", vf[test]) if test == "T1_ladder_external" else vf[test]
        cands = ["partial_capped", "partial"] if (test == "T2_direction" and v == "partial") else [v]
        for key in cands:
            got = extract_fields(it[test]["outcomes"][key]["text"], texts[test])
            if got is not None:
                keys[test] = key
                by_test[test] = dict(got)
                for name, val in got.items():
                    # одно имя поля у разных тестов бывает о разном ({best_partition} у T1 — соперник по ρ,
                    # у T5 — по ε²): такое поле — только в заголовке своего теста, в общие тексты не идёт
                    if name in fields and fields[name] != val:
                        clash.add(name)
                    fields.setdefault(name, val)
                break
        else:
            raise QCError(
                f"site: текст {test} не совпал с шаблоном исхода {cands} (вердикт {v}): {texts[test]!r}"
            )
    for name in clash:
        fields.pop(name, None)
    t7 = facts["t7"]
    keys["T7_type_gain"] = t7["type_gain"]
    set_name = SET_NAMES[t7["product"]]
    if fields.setdefault("set_name", set_name) != set_name:
        raise QCError(f"site: {{set_name}} в тексте T7 {fields['set_name']!r} ≠ набор продукта {set_name!r}")
    keys["one_in_ten"] = keys["T3_reliable_placebo"] == "confirmed" and bool(fields.get("one_in_ten_phrase"))
    t6 = it["T6_bank_coverage"]
    caveat = float(facts["t6"]["coverage_eps2"]) > float(t6["coverage_caveat"]["epsilon2_max"])
    cav_tpl = t6["outcomes"]["caveat"]["text"]
    cav_in_text = re.search(_caveat_regex(cav_tpl), texts["T6_bank_coverage"])
    if caveat != bool(cav_in_text):
        raise QCError(
            f"site: оговорка T6 (ε² {facts['t6']['coverage_eps2']}) и текст T6 этапа 5 не согласованы"
        )
    keys["caveat"] = caveat
    return Verdicts(keys, fields, by_test)


def _caveat_regex(tpl: str) -> str:
    """Оговорка T6 внутри текста: шаблон без якорей."""
    return "".join(
        "(.*?)" if re.fullmatch(r"\{[A-Za-z_0-9]+\}", p) else re.escape(p)
        for p in re.split(r"(\{[A-Za-z_0-9]+\})", tpl)
    )


def fill(template: str, values: Mapping[str, str]) -> str:
    """Подстановка полей; поле без значения (нет, None или "") остаётся ``{…}`` и ловится линтом (код 3)."""
    return SLOT.sub(lambda m: str(values.get(m.group(0)[1:-1]) or m.group(0)), template)


# --- story.json ---------------------------------------------------------------------------------------------


def _headline(H: Mapping, test: str, key: str, vd: Verdicts) -> str:
    """Короткий заголовок исхода; поля — из текста исхода того же теста (``Verdicts.of``)."""
    base = "T7_utility" if test == "T7_type_gain" else test
    return cap(fill(H[test][key], vd.of(base)))


def build_story(cfg: Config, facts: Mapping, numbers: Mapping[str, str], mode: Mode) -> dict:
    """Тексты слотов (§4.1), флаги ``view`` (§4.3) и метаданные; только выбор по вердиктам, без расчётов."""
    site, it = cfg["site"], cfg["interpret"]
    H, R, A, P = site["headlines"], site["roles"], site["always"], site["palette_rule"]
    vd = read_verdicts(cfg, facts)
    v, f = vd.keys, vd.fields
    t1, t2, t3, t5, t6, t7 = (v[t] for t in TESTS)
    th = facts["thesis"]
    texts = facts["texts"]
    scope = facts["scope"]
    ch, s0 = H["chapters"], H["screen0"]
    ok_if = lambda cond: all(v[t] in lv for t, lv in cond.items())  # noqa: E731

    # экран 0
    scope_vals = {
        k: (style.fmt_num(scope[k]) if scope.get(k) is not None else "") for k in _slots(A["scope_reader"])
    }
    screen0 = {
        "title": _headline(H, s0["title"], v[s0["title"]], vd),
        "lead": _headline(H, s0["lead"], v[s0["lead"]], vd),
        "question": th["question"],
        "points": [list(th["point_1"]), list(th["point_2"]), list(th["point_3"])],
        "caveat": list(th["caveat"]),
        "t6_caveat": H["T6_caveat"]["caveat"] if v["caveat"] else None,
        "scope": fill(A["scope_reader"], scope_vals),
        "coverage": H["descriptive"]["coverage"],
    }
    dyn_lead = (
        _headline(H, "T2_direction", t2, vd)
        if ok_if(ch["dynamics"]["lead_if"])
        else H["descriptive"]["dynamics_neutral"]
    )
    limits_title = (
        _headline(H, "T6_bank_coverage", t6, vd)
        if ok_if(ch["limits"]["title_if"])
        else H["descriptive"]["limits_neutral"]
    )
    # после вскрытия (site.build.texts, §4.4): короткие заголовки пунктов — те же заголовки site.headlines,
    # что у глав (по тем же правилам показа); полные тексты пунктов — дословно под «Как проверяли»
    heads_by_test = {
        "T1_ladder_external": _headline(H, "T1_ladder_external", t1, vd),
        "T3_reliable_placebo": _headline(H, "T3_reliable_placebo", t3, vd),
        "T5_trivial": _headline(H, "T5_trivial", t5, vd),
        "T2_direction": dyn_lead,
        "T7_utility": _headline(H, "T7_utility", t7, vd),
        "T6_bank_coverage": limits_title,
    }
    on_top = {s0["title"], s0["lead"]}  # уже стоят заголовком и подзаголовком экрана 0
    ta = it["thesis_assembly"]
    heads = []
    for k in ("point_1", "point_2", "point_3"):
        hs = [heads_by_test[t] for t in ta[k] if t not in on_top] or [heads_by_test[ta[k][0]]]
        # порция 6b (check-ux): если в пункте остался только нейтральный вопрос (подзаголовок T2 без
        # вердикта), рядом с ним — короткий заголовок исхода T3 (тот же, что подзаголовком выше)
        if hs == [H["descriptive"]["dynamics_neutral"]] and "T3_reliable_placebo" in ta[k]:
            hs = hs + [heads_by_test["T3_reliable_placebo"]]
        if "T7_utility" in ta[k]:
            hs.append(_headline(H, "T7_type_gain", v["T7_type_gain"], vd))
        heads.append(hs)
    TX = site["build"].get("texts") or {}
    n_types = len(facts["ladder"]["order"])
    screen0 |= {
        "intro": fill(
            TX["intro"],
            {
                "n_nodes": style.fmt_num(scope["n_nodes"]),
                "n_types": f"{n_types} {plural_ru(n_types, 'тип', 'типа', 'типов')}",
            },
        )
        if TX.get("intro") and scope.get("n_nodes") is not None
        else None,
        "point_heads": heads,
        "caveat_head": H["T6_caveat"]["caveat"] if v["caveat"] else limits_title,
        "labels": {k: TX.get(k) for k in ("question_label", "how_checked", "caveat_label") if TX.get(k)},
        "gloss": dict(TX.get("gloss") or {}),
        "regions_note": regions_note(TX.get("regions_note"), numbers),
        "search_none": fill_if_all(TX.get("search_none"), numbers),
        "hero": hero_texts(cfg, TX.get("hero") or {}, scope, n_types),
    }
    flows = P["t3_flows"][t3]
    n_set = int(it["tests"]["T7_utility"]["k"])
    chapters = {
        "basket": {"title": H["descriptive"]["basket"]},
        "similarity": {
            "title": fill(H["descriptive"]["similarity"], numbers),
            "construction": A["honesty"]["construction"],
            "lines": A["honesty"]["lines"],
            "layout": A["honesty"]["layout"],
            "layout_caption": None,  # TODO: {preserved} — из раскладки (site_layout, следующая порция)
        },
        "types": {
            "title": _headline(H, "T5_trivial", t5, vd),
            "note": H["T4_basket_vs_place"]["describe"],
            "text": texts["T5_trivial"],
            "note_text": texts["T4_basket_vs_place"],
            "explain": types_explain((TX.get("types_explain") or {}).get(t5), facts.get("t5") or {}),
        },
        "order": {
            "title": _headline(H, "T1_ladder_external", t1, vd),
            "text": texts["T1_ladder_external"] + facts.get("t1_notes", {}).get("T1_ladder_external", ""),
            "proxies": texts.get("T1_proxies", ""),
        },
        "dynamics": {
            "title": _headline(H, "T3_reliable_placebo", t3, vd),
            "lead": dyn_lead,
            "texts": [texts["T3_reliable_placebo"], texts["T2_direction"]],
            "layer_name": flows["layer_name"],
        },
        "comparable": {
            "title": _headline(H, "T7_utility", t7, vd),
            "lead": _headline(H, "T7_type_gain", v["T7_type_gain"], vd),
            "text": texts["T7_utility"],
            "same_period": A["honesty"]["t7_same_period"],
            "comp_link": comp_link(TX.get("comp_link"), facts.get("t7") or {}),
            "similar_caption": fill(
                site["similar_caption"], {"n_shown": str(site["n_similar_shown"]), "n_set": str(n_set)}
            ),
        },
        "limits": {
            "title": limits_title,
            "lead": [H["descriptive"]["coverage"]] + ([H["T6_caveat"]["caveat"]] if v["caveat"] else []),
            "text": texts["T6_bank_coverage"],
            "items": limits_items(TX.get("limits") or {}, numbers, scope, screen0.get("regions_note")),
        },
        "explore": {"title": H["descriptive"]["explore"], "nodata": A["honesty"]["nodata"]},
        "method": {
            "title": H["descriptive"]["method"],
            "command": site["build"].get("command"),
            "prereg": [dict(p) for p in site["build"].get("prereg") or []],
            "links": {"repo": site["build"].get("repo_url")},
            "numbers": {},  # заполняет run() из выгрузок (method_numbers)
        },
        "sources": {"title": H["descriptive"]["sources"], "licenses": LICENSES},
    }
    an = R["analysts"]
    roles = {
        "ministry": fill(R["ministry"][t7], f),
        "analysts": fill(an[t3], f) if (ok_if(an["show_if"]) and an[t3]) else None,
        "business": fill(R["business"]["text"], numbers),
        "business_label": R["business"]["label"],
        "examples": None,  # TODO: реальные МО по правилам roles.*.example (следующая порция)
    }
    view = build_view(cfg, facts, vd)
    return {
        "meta": {
            "mode": mode.name,
            "banner": mode.banner,
            "seed": cfg["seed"],
            "site_freeze_sha256": freeze_hash(site),
            "facts_sha256": _sha(facts),
            "licenses": LICENSES,
        },
        "verdicts": {t: v[t] for t in (*TESTS, "T7_type_gain", "one_in_ten", "caveat")},
        "screen0": screen0,
        "chapters": chapters,
        "roles": roles,
        "names": {
            "final": dict(facts["names_final"]),
            "descriptive": dict(facts["names_descriptive"]),
            "caption": "Относительно своего региона",
        },
        "reliability_key": P["reliability_grammar"]["key"],
        "stability_label": A["mo_stability"]["label"],
        "card": {  # строки карточки из site.build.texts (линтуются как текст сайта)
            "no_comparable": TX.get("no_comparable"),
            "shifted": TX.get("shifted"),
            "lines": A["honesty"]["lines"],
            # варианты R1 в node_r1 / node_seed (kind = variant): у узлов-городов нет nodes_separate
            "variants": {"3": ", ".join(R1_WORDS.values()), "2": ", ".join(list(R1_WORDS.values())[:2])},
        },
        "view": view,
    }


def hero_texts(cfg: Config, tx: Mapping[str, str], scope: Mapping, n_types: int) -> dict[str, str] | None:
    """Тексты первого экрана (``site.build.texts.hero``, порция 5a): поля — числа из ``facts.scope``,
    число типов словами и годы периода из конфига. Шаблон подписи острова ``island_note`` (поле ``{ratio}``)
    в story не идёт: готовые подписи — в ``scene.json``, их линтует ``run`` (``lint_texts``).
    Нет блока — None (прежний экран 0)."""
    if not tx:
        return None
    nom, gen = site_scene.num_words(n_types)
    y0, y1 = (str(cfg["period"][k])[:4] for k in ("start", "end"))
    n_nodes = int(scope.get("n_nodes") or 0)
    n_reg = int(scope.get("n_regions") or 0)
    n_cells = n_nodes + int(scope.get("n_untyped") or 0)
    vals = {
        "n_nodes": style.fmt_num(n_nodes) if n_nodes else "",
        "mo_word": plural_ru(n_nodes, "муниципалитет", "муниципалитета", "муниципалитетов"),
        "n_regions": style.fmt_num(n_reg) if n_reg else "",
        "region_word": plural_ru(n_reg, "регионе", "регионах", "регионах"),
        "years": f"{y0}–{y1} годов" if y0 != y1 else f"{y0} года",
        "n_types_gen": gen,
        "n_types": f"{nom} {plural_ru(n_types, 'тип', 'типа', 'типов')}",
        "n_cells": style.fmt_num(n_cells) if n_cells else "",
        "n_untyped": style.fmt_num(int(scope.get("n_untyped") or 0)),
    }
    return {k: fill(str(v), vals) for k, v in tx.items() if k != "island_note"}


def fill_if_all(tpl: str | None, vals: Mapping[str, str]) -> str | None:
    """Шаблон с полями из ``vals``; нет хотя бы одного значения — None (строка не показывается)."""
    if not tpl or not all(vals.get(k) for k in _slots(tpl)):
        return None
    return fill(tpl, {k: vals[k] for k in _slots(tpl)})


def types_explain(tpl: str | None, t5: Mapping) -> str | None:
    """Пояснение под заголовком главы 3 (порция 6b): AMI с самым близким делением — ``facts.t5``."""
    if not tpl or t5.get("max_ami") is None or not t5.get("max_label"):
        return None
    return fill(tpl, {"ami": style.fmt_num(float(t5["max_ami"]), 2), "max_label": str(t5["max_label"])})


def comp_link(tpl: str | None, t7: Mapping) -> str | None:
    """Связка главы 6 с блоком «Что с этим делать» (порция 6b): медианные ошибки соседей по региону (B)
    и набора продукта в обеих целях — ``facts.t7``."""
    p = t7.get("product")
    rel, ab = t7.get("median_error") or {}, t7.get("median_error_abs") or {}
    if not tpl or not p or any(x.get(k) is None for x in (rel, ab) for k in ("B", p)):
        return None
    f3 = lambda v: style.fmt_num(float(v), 3)  # noqa: E731
    return fill(tpl, {"b_abs": f3(ab["B"]), "p_abs": f3(ab[p]), "b_rel": f3(rel["B"]), "p_rel": f3(rel[p])})


def regions_note(tpl: str | None, numbers: Mapping[str, str]) -> str | None:
    """Строка «почему регионов меньше»: числа — из ``site_numbers`` (нет чисел — строки нет)."""
    keys = ("n_incomplete", "incomplete", "n_absent", "absent")
    if not tpl or not all(numbers.get(k) for k in keys):
        return None
    return fill(tpl, {k: numbers[k] for k in keys})


def limits_items(
    tpl: Mapping[str, str], numbers: Mapping[str, str], scope: Mapping, regions: str | None
) -> dict[str, str]:
    """Ограничения главы 7 (``site.build.texts.limits``) в порядке страницы; числа — из ``site_numbers``
    и ``facts.scope`` этапа 5. Строка, у которой нет хотя бы одного числа, не показывается."""
    num = lambda k: style.fmt_num(round(float(scope[k]))) if scope.get(k) is not None else ""  # noqa: E731
    vals = dict(numbers) | {
        "n_untyped": num("n_untyped"),
        "pop_partial": num("pop_median_partial"),
        "pop_full": num("pop_median_full"),
    }
    # «у 41 муниципалитета», «у 169 муниципалитетов» — родительный падеж после «у»
    for k in ("n_untyped", "n_recip_gt3"):
        n = int(re.sub(r"\D", "", str(vals.get(k) or "")) or -1)
        vals[k + "_mo"] = (
            f"{vals[k]} {plural_ru(n, 'муниципалитета', 'муниципалитетов', 'муниципалитетов')}"
            if n >= 0
            else ""
        )
    out: dict[str, str] = {}
    for k in ("visitors", "regions", "untyped", "nominal", "workplace"):
        t = regions if k == "regions" else tpl.get(k)
        if not t or not all(vals.get(s) for s in _slots(t)):
            continue
        out[k] = fill(t, vals)
    return out


def _slots(s: str) -> list[str]:
    return [m[1:-1] for m in SLOT.findall(s)]


def _sha(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def build_view(cfg: Config, facts: Mapping, vd: Verdicts) -> dict:
    """Флаги страницы по вердиктам (§4.3): палитра, форма главы 4, стрелки, потоки, соперник, набор T7."""
    site = cfg["site"]
    P = site["palette_rule"]
    v = vd.keys
    t1, t2, t3, t5 = v["T1_ladder_external"], v["T2_direction"], v["T3_reliable_placebo"], v["T5_trivial"]
    order = [int(x) for x in facts["ladder"]["order"]]
    sizes = [int(x) for x in facts["ladder"]["sizes_territorial"]]
    ordered = all(v[t] == lv for t, lv in P["type_palette"]["ordered_if"].items())
    colors = palette_colors(ordered, order)
    by_size = [t for _, t in sorted(zip(sizes, order, strict=True), key=lambda p: -p[0])]
    arrows = t2 in P["t2_arrows"] and t3 in ("confirmed", "partial")
    flows = P["t3_flows"][t3]
    t7 = facts["t7"]
    return {
        "palette": "ordered" if ordered else "nominal",
        "type_colors": colors,
        "shapes": {str(k): s for k, s in P["type_palette"]["shapes"].items()},
        "legend_order": order
        if P["order_in_legends"]["T1_confirmed"] == "ladder" and t1 == "confirmed"
        else by_size,
        "ladder_words": t1 == "confirmed",
        "t1_layout": P["t1_chapter_form"]["layout"][t1],
        "order_marks": all(v[t] in lv for t, lv in site["similarity_layout"]["order_marks_if"].items()),
        "dir_words": arrows,
        "arrows": arrows,
        "dynamics_lead": "T2"
        if all(v[t] in lv for t, lv in _chap(site, "dynamics")["lead_if"].items())
        else "neutral",
        "limits_title": "T6"
        if all(v[t] in lv for t, lv in _chap(site, "limits")["title_if"].items())
        else "neutral",
        "flows": {k: flows[k] for k in ("reliable", "noise", "stayed")},
        "transitions_default_on": flows["layer_default"] in (True, "on"),  # YAML читает on как true
        "show_rival": t5 != "confirmed",
        "product_set": t7["product"],
        "set_name": SET_NAMES[t7["product"]],
        "forecast_words": False,
        "one_in_ten": bool(v["one_in_ten"]),
        "show_t6_caveat": bool(v["caveat"]),
        "reliability": dict(P["reliability_grammar"]) | {"key": None},
        "unstable_label": facts.get("r1", {}).get("unstable_label"),
        "unstable_parts": list(facts.get("r1", {}).get("unstable_parts", [])),
        "line_by_turnover": None,  # TODO: по facts.t1 каждого оборота (checks.json, следующая порция)
    }


def _chap(site: Mapping, name: str) -> Mapping:
    return site["headlines"]["chapters"][name]


def palette_colors(ordered: bool, order: list[int]) -> dict[str, str]:
    """Цвета типов: порядковая — ступени светлоты по ``ladder.order``; иначе номинальная по номеру типа."""
    if ordered:
        return {str(t): style.TYPE_PALETTE_ORDERED[i] for i, t in enumerate(order)}
    return {str(t): style.TYPE_PALETTE_NOMINAL[t] for t in order}


def nominal_monotone(order: list[int]) -> bool:
    """Светлоты номинальной палитры по ``ladder.order`` монотонны — подсказывали бы непроверенный порядок.
    Светлота и признак монотонности — те же функции, что в ``style.palette_audit`` (одна копия)."""
    return style.monotone([style.lightness(style.TYPE_PALETTE_NOMINAL[t]) for t in order])


def audit_palettes(cfg: Config, facts: Mapping) -> style.PaletteAudit:
    """``style.palette_audit`` обеих палитр (site.palette_rule.type_palette.audit) до вёрстки: порядок
    ступеней — ``facts.ladder.order`` (после переноса номеров по размерам, как в story.json), фигуры —
    из конфига. Провал любой проверки — код 3."""
    tp = cfg["site"]["palette_rule"]["type_palette"]
    audit = style.palette_audit([int(x) for x in facts["ladder"]["order"]], tp["shapes"])
    if audit.exit_code:
        raise QCError(
            "site: палитры типов не прошли проверку: "
            + "; ".join(f"{c['check']} = {c['value']} (порог {c['threshold']})" for c in audit.failures[:10])
        )
    log.info("site: палитры типов — %d проверок пройдены", len(audit.checks))
    return audit


# --- линт ---------------------------------------------------------------------------------------------------


def _strings(obj: Any, path: str = "") -> Iterable[tuple[str, str]]:
    if isinstance(obj, str):
        yield path, obj
    elif isinstance(obj, Mapping):
        for k, x in obj.items():
            yield from _strings(x, f"{path}.{k}" if path else str(k))
    elif isinstance(obj, list | tuple):
        for i, x in enumerate(obj):
            yield from _strings(x, f"{path}[{i}]")


# Части story.json, которые пишет сам сайт (линтуются); тексты этапа 5 — дословно и только сверяются
SITE_AUTHORED = ("screen0.title", "screen0.lead", "screen0.t6_caveat", "screen0.scope", "screen0.coverage")
VERBATIM_PREFIXES = ("screen0.question", "screen0.points", "screen0.caveat", "names.")
VERBATIM_KEYS = ("text", "texts", "note_text", "proxies")


def _is_verbatim(path: str) -> bool:
    if path.startswith(VERBATIM_PREFIXES):
        return True
    last = re.sub(r"\[\d+\]$", "", path).rsplit(".", 1)[-1]
    return path.startswith("chapters.") and last in VERBATIM_KEYS


def lint_story(cfg: Config, story: Mapping, facts: Mapping) -> list[str]:
    """Нарушения текста страницы (код 3): запреты ``forbidden_words`` по текущим вердиктам, запреты
    в заголовках и названиях типов, пустые поля ``{…}``, главы без заголовка, расхождение дословных текстов
    с ``facts.json``, монотонная номинальная палитра."""
    site = cfg["site"]
    fw = site["forbidden_words"]
    hc = HeadlineChecker(cfg.data)
    v = dict(story["verdicts"])
    bad: list[str] = []
    authored: list[tuple[str, str]] = []
    for path, s in _strings(
        {
            k: story[k]
            for k in ("screen0", "chapters", "roles", "names", "reliability_key", "card")
            if k in story
        }
    ):
        if SLOT.search(s):
            bad.append(f"пустое поле {SLOT.findall(s)} в {path}: «{s}»")
        if not _is_verbatim(path):
            authored.append((path, s))
    for path, s in _strings(story["meta"]):
        if SLOT.search(s):
            bad.append(f"пустое поле {SLOT.findall(s)} в {path}")
    for kind, word, text in hc.text_violations([s for _, s in authored], v):
        bad.append(f"{kind}: «{word}» в «{text}»")
    s0h = story["screen0"]
    headlines = [s0h["title"], s0h["lead"]] + [
        c[k] for c in story["chapters"].values() for k in ("title", "lead") if isinstance(c.get(k), str)
    ]
    headlines += [h for hs in s0h.get("point_heads") or [] for h in hs] + (
        [s0h["caveat_head"]] if s0h.get("caveat_head") else []
    )
    if (s0h.get("hero") or {}).get("title"):  # h1 первого экрана (порция 5a) — тоже заголовок
        headlines.append(s0h["hero"]["title"])
    banned_h = list(fw["headlines_always"]) + list(cfg["interpret"]["naming"]["banned"])
    for h in headlines:
        bad += [f"запрет в заголовке: «{w}» в «{h}»" for w in banned_h if norm(w) in norm(h)]
    for kind in ("final", "descriptive"):
        for t, name in story["names"][kind].items():
            bad += [
                f"запрет в названии типа {t}: «{w}» в «{name}»"
                for w in list(cfg["interpret"]["naming"]["banned"]) + list(fw["page_always"])
                if norm(w) in norm(name)
            ]
    for name, ch in story["chapters"].items():
        if not (isinstance(ch.get("title"), str) and ch["title"].strip()):
            bad.append(f"глава {name} без заголовка")
    # тексты этапа 5 — побуквенно из facts.json
    th, tx = facts["thesis"], facts["texts"]
    s0 = story["screen0"]
    if s0["question"] != th["question"] or s0["points"] != [th["point_1"], th["point_2"], th["point_3"]]:
        bad.append("экран 0: вопрос или пункты не совпадают с facts.thesis")
    if s0["caveat"] != th["caveat"]:
        bad.append("экран 0: оговорка не совпадает с facts.thesis.caveat")
    for test, got in (
        ("T5_trivial", story["chapters"]["types"]["text"]),
        ("T7_utility", story["chapters"]["comparable"]["text"]),
        ("T6_bank_coverage", story["chapters"]["limits"]["text"]),
    ):
        if got != tx[test]:
            bad.append(f"текст {test} не совпадает с facts.texts")
    if (
        site["palette_rule"]["type_palette"].get("nominal_not_monotone")
        and story["view"]["palette"] == "nominal"
    ):
        if nominal_monotone([int(x) for x in facts["ladder"]["order"]]):
            bad.append("номинальная палитра монотонна по светлоте в порядке ladder.order")
    return bad


_CYR = re.compile(r"[А-Яа-яЁё]")
_JS_LIT = re.compile(r'"(?:[^"\\\n]|\\.)*"|\'(?:[^\'\\\n]|\\.)*\'|`(?:[^`\\]|\\.)*`')


def template_strings(html: str, js: str, svg: str = "") -> list[str]:
    """Тексты интерфейса, которые пишет сам шаблон: текст и подписи (aria-label, placeholder, title, alt)
    в ``landing.html`` без комментариев и подстановок, строковые литералы ``landing.js`` с кириллицей
    (без комментариев), текст статичной SVG."""
    h = re.sub(r"<!--.*?-->", " ", html, flags=re.S)
    h = re.sub(r"<(style|script)\b.*?</\1>", " ", h, flags=re.S)
    out = [m.strip() for m in re.findall(r">([^<>]+)<", h) if _CYR.search(m)]
    out += [
        v for v in re.findall(r'(?:aria-label|placeholder|title|alt)="([^"]*)"', h + svg) if _CYR.search(v)
    ]
    code = re.sub(r"(?m)^\s*//.*$", " ", js)
    code = re.sub(r"(?<=[;{}),\s])//[^\n\"'`]*$", " ", code, flags=re.M)
    out += [m[1:-1] for m in _JS_LIT.findall(code) if _CYR.search(m)]
    out += [m.strip() for m in re.findall(r">([^<>]+)<", svg) if _CYR.search(m)]
    return out


def lint_templates(cfg: Config, story: Mapping, svg: str) -> list[str]:
    """Линт шаблона (§4.1, forbidden_words): интерфейс, aria-label, подписи и SVG — те же запреты
    ``page_always`` и ``by_verdict``, что у текста сайта, при текущих вердиктах. Нарушение — код 3."""
    tdir = Path(__file__).parent / "templates"
    read = lambda n: (tdir / n).read_text(encoding="utf-8") if (tdir / n).exists() else ""  # noqa: E731
    hc = HeadlineChecker(cfg.data)
    texts = (
        template_strings(read("landing.html"), read("landing.js") + "\n" + read("landing3d.js"), svg)
        + site_chapters.ui_strings()
        + site_chapters_tail.ui_strings()
    )
    return [
        f"{kind}: «{w}» в «{t[:80]}»" for kind, w, t in hc.text_violations(texts, dict(story["verdicts"]))
    ]


def lint_texts(cfg: Config, story: Mapping, texts: Iterable[str]) -> list[str]:
    """Линт готовых строк, которых нет в story.json (подписи островов ``scene.json``): те же запреты
    ``page_always`` и ``by_verdict`` при текущих вердиктах, пустые поля ``{…}``. Нарушение — код 3."""
    texts = [t for t in texts if t]
    hc = HeadlineChecker(cfg.data)
    bad = [f"пустое поле {SLOT.findall(t)}: «{t}»" for t in texts if SLOT.search(t)]
    return bad + [
        f"{kind}: «{w}» в «{t[:80]}»" for kind, w, t in hc.text_violations(texts, dict(story["verdicts"]))
    ]


def screen_examples(types: list[dict], story: Mapping, n: int = 4) -> list[int]:
    """Кнопки-примеры экрана 0: первый типичный пример каждого типа (``examples.csv``) в порядке легенды;
    по одному на каждый из четырёх типов (решение участника 30.09, было три — тип 4 оставался без примера)."""
    out = []
    for r in types:
        ex = (r.get("examples") or {}).get("typical") or []
        if ex:
            out.append(int(ex[0]))
    return out[:n]


# --- бюджет и запись ----------------------------------------------------------------------------------------


def dumps(obj: Any) -> bytes:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


# --- данные страницы: входы ---------------------------------------------------------------------------------

PARTS = ("food", "marketplace", "transport", "health", "cafe", "other")
PART_LABELS = {
    "food": "продукты",
    "marketplace": "маркетплейсы",
    "transport": "транспорт",
    "health": "здоровье",
    "cafe": "кафе и рестораны",
    "other": "прочее",
}
KIND_RU = {
    "mr": "муниципальный район",
    "mo": "муниципальный округ",
    "go": "городской округ",
    "vgt": "внутригородская территория",
    "city": "город федерального значения",
}
OKRUG_RU = {
    "central": "Центральный",
    "northwestern": "Северо-Западный",
    "southern": "Южный",
    "north_caucasian": "Северо-Кавказский",
    "volga": "Приволжский",
    "ural": "Уральский",
    "siberian": "Сибирский",
    "far_eastern": "Дальневосточный",
}
STATUS_CODE = {"no_change": "n", "reliable": "r", "within_noise": "w"}
# варианты R1 (interpret.robustness) словами для карточки; у узлов-городов — первые два (без третьего)
R1_WORDS = {  # один словарь с главами (порция 6b: «другое правило связей между муниципалитетами»)
    k: site_chapters.VARIANT_WORDS[k]
    for k in ("variant:graph_basket_cos", "variant:no_level", "variant:nodes_separate")
}
N_WINDOWS = 13
ICVI_BETTER = {
    "sw": "max",
    "ch": "max",
    "s_dbw": "min",
    "avi": "max",
    "avu": "min",
    "mq": "max",
    "anui": "max",
}
ICVI_LABELS = {
    "sw": "SW",
    "ch": "CH",
    "s_dbw": "S_Dbw",
    "avi": "AVI",
    "avu": "AVU",
    "mq": "MQ",
    "anui": "ANUI",
}


def _num(x: Any, nd: int = 2) -> float | int | None:
    """Число для JSON: пропуск -> None, округление до ``nd`` знаков (целые остаются целыми)."""
    if x is None or (isinstance(x, float) and not np.isfinite(x)) or pd.isna(x):
        return None
    if isinstance(x, (int, np.integer)) and not isinstance(x, bool):
        return int(x)
    v = round(float(x), nd)
    return int(v) if nd == 0 else v


@dataclass
class SiteData:
    """Входы этапа site, прочитанные один раз. Номера типов — как на странице (после ``ladder.transfer``)."""

    cfg: Config
    mode: Mode
    facts: Mapping
    terr: pd.DataFrame  # territories (2190 МО)
    nodes: pd.DataFrame  # features_nodes (1945: узлы сети и МО без типа), индекс — territory_id
    members: pd.DataFrame  # features_members (2190: МО -> узел)
    transfer: dict[int, int]  # номер cluster_final -> номер страницы
    node_type: pd.Series  # узел -> тип страницы
    type_source: str

    def opt(self, name: str) -> pd.DataFrame | None:
        """Выгрузка этапа interpret (``outputs/interpret/<name>``); нет файла — None (блок скрывается)."""
        p = self.mode.interpret_dir / name
        return pd.read_csv(p) if p.exists() else None

    def processed(self, name: str, **kw: Any) -> pd.DataFrame:
        p = Path(self.cfg["paths"]["processed"]) / f"{name}.parquet"
        if not p.exists():
            raise MissingInputError(f"site: нет {p} — этапы panel, features, network, cluster, dynamics")
        return pd.read_parquet(p, **kw)

    def outputs(self, rel: str) -> Path:
        return self.cfg.dir("outputs") / rel

    def tmap(self, s: pd.Series) -> pd.Series:
        """Номера cluster (окна, dynamics) -> номера страницы."""
        return s.map(lambda v: self.transfer.get(int(v)) if pd.notna(v) else None)


def load_site_data(cfg: Config, mode: Mode, facts: Mapping) -> SiteData:
    """Справочник, узлы, состав узлов и тип каждого узла. Тип — из ``types.csv`` этапа interpret (в обычном
    режиме он уже сверен с ``cluster_final`` поузлово); без него (``--demo``) — ``cluster_final`` через
    ``facts.ladder.transfer`` (нет переноса — номера как есть)."""
    proc = Path(cfg["paths"]["processed"])
    need = [
        proc / f"{n}.parquet" for n in ("territories", "features_nodes", "features_members", "cluster_final")
    ]
    missing = [str(p) for p in need if not p.exists()]
    if missing:
        raise MissingInputError(f"site: нет {missing} — этапы panel, features, cluster")
    terr = pd.read_parquet(need[0])
    nodes = pd.read_parquet(need[1]).set_index("territory_id", drop=False)
    members = pd.read_parquet(need[2])
    final = pd.read_parquet(need[3], columns=["territory_id", "type"])
    tr = (facts.get("ladder") or {}).get("transfer") or {}
    transfer = {int(k): int(v) for k, v in tr.items()} or {int(t): int(t) for t in final["type"].unique()}
    types_csv = mode.interpret_dir / "types.csv"
    if types_csv.exists():
        t = pd.read_csv(types_csv, usecols=["territory_id", "type"])
        node_type = pd.Series(t["type"].astype(int).to_numpy(), index=t["territory_id"].astype(int))
        source = "types.csv"
    else:
        node_type = pd.Series(
            [transfer[int(v)] for v in final["type"]], index=final["territory_id"].astype(int)
        )
        source = "cluster_final + ladder.transfer"
    return SiteData(cfg, mode, facts, terr, nodes, members, transfer, node_type.sort_index(), source)


def check_controls(d: SiteData) -> list[str]:
    """Код 3: контрольные числа ``site.build.controls`` и ``facts.scope`` против файлов, размеры типов
    ``facts.ladder.sizes_territorial`` против типов узлов (в ``--demo`` размеры синтетические — пропуск)."""
    ctl = d.cfg["site"]["build"]["controls"]
    scope = d.facts.get("scope") or {}
    nodes = d.nodes
    got = {
        "n_nodes": int(nodes["is_node"].sum()),
        "n_untyped": int((~nodes["is_node"]).sum()),
        "n_inner": int((d.members["role"] == "city_member").sum()),
    }
    bad = []
    for k, v in got.items():
        if int(ctl[k]) != v:
            bad.append(f"{k}: в файлах {v}, в site.build.controls {ctl[k]}")
        if k in scope and scope[k] is not None and int(scope[k]) != v:
            bad.append(f"{k}: в файлах {v}, в facts.scope {scope[k]}")
    n_members = int(nodes.loc[nodes["is_city_node"], "n_members"].sum())
    if n_members != got["n_inner"]:
        bad.append(f"n_inner: районов в узлах-городах {n_members} ≠ city_member {got['n_inner']}")
    if len(d.node_type) != got["n_nodes"] or set(d.node_type.index) != set(nodes.index[nodes["is_node"]]):
        bad.append(f"типы есть у {len(d.node_type)} узлов ({d.type_source}), узлов сети {got['n_nodes']}")
    n_regions = int(nodes.loc[nodes["is_node"], "region_code"].nunique())
    if scope.get("n_regions") is not None and int(scope["n_regions"]) != n_regions:
        bad.append(f"n_regions: в файлах {n_regions}, в facts.scope {scope['n_regions']}")
    lad = d.facts.get("ladder") or {}
    if d.mode.name == "demo":
        log.warning("site --demo: размеры типов в facts синтетические — сверка с файлами пропущена")
    elif lad.get("order") and lad.get("sizes_territorial"):
        terr_nodes = nodes.index[nodes["is_node"] & ~nodes["is_city_node"]]
        counts = d.node_type.reindex(terr_nodes).value_counts()
        for t, n in zip(lad["order"], lad["sizes_territorial"], strict=True):
            if int(counts.get(int(t), 0)) != int(n):
                bad.append(f"тип {t}: территориальных узлов {int(counts.get(int(t), 0))}, в facts.ladder {n}")
    for c in scope.get("city_nodes") or []:
        have = d.node_type.get(int(c["territory_id"]))
        if have is not None and int(have) != int(c["type"]):
            bad.append(f"узел-город {c['name']}: тип {have}, в facts.scope {c['type']}")
    return bad


# --- карта ячеек --------------------------------------------------------------------------------------------


def okrug_of(cfg: Config) -> dict[int, str]:
    """Регион -> федеральный округ (``interpret.federal_districts``)."""
    out = {}
    for k, regs in cfg["interpret"]["federal_districts"].items():
        for r in regs:
            out[int(r)] = OKRUG_RU.get(k, k)
    return out


@dataclass
class HexMap:
    """Ячейки страницы: ``hq``, ``hr`` каждого из 1945 объектов, решётка и качество раскладки."""

    cells: pd.DataFrame  # territory_id, hq, hr, shift_km, region_code
    grid: site_hexgrid.PageGrid
    stats: dict[str, Any]


def build_hexmap(d: SiteData) -> HexMap:
    hp = d.cfg["site"]["build"]["hex"]
    n = d.nodes
    res = site_hexgrid.build_grid(
        n["territory_id"].to_numpy(),
        n["x_aea"].to_numpy() / 1000.0,
        n["y_aea"].to_numpy() / 1000.0,
        n["region_code"].to_numpy(),
        hp,
    )
    hq, hr = site_hexgrid.page_axial(res.q, res.r)
    grid = site_hexgrid.page_grid(hq, hr, float(hp["size"]), float(hp["margin"]))
    cells = pd.DataFrame(
        {
            "territory_id": res.ids.astype(int),
            "hq": hq.astype(int),
            "hr": hr.astype(int),
            "shift_km": res.shift_km,
            "region_code": n["region_code"].to_numpy().astype(int),
        }
    )
    if res.stats["groups_split"]:
        raise QCError(f"site: ячейки регионов разорваны: {res.stats['split']} — подберите site.build.hex")
    return HexMap(cells, grid, res.stats)


def build_hexgrid_json(d: SiteData, hm: HexMap, types_by_cell: pd.Series) -> dict:
    """``hexgrid.json``: решётка, контуры регионов и округов по сторонам ячеек, подписи, смещения."""
    c = hm.cells
    ok = okrug_of(d.cfg)
    region_names = d.nodes.set_index("territory_id")["region_name"]
    okrug = c["region_code"].map(lambda r: ok.get(int(r), "—"))
    x, y = hm.grid.center(c["hq"], c["hr"])
    regions = []
    for code, g in c.assign(x=x, y=y).groupby("region_code"):
        regions.append(
            {
                "code": int(code),
                "name": str(region_names.loc[g["territory_id"].iloc[0]]),
                "okrug": ok.get(int(code)),
                "n": int(len(g)),
                "x": round(float(g["x"].median()), 1),
                "y": round(float(g["y"].median()), 1),
            }
        )
    okrugs = [
        {"name": name, "x": round(float(g["x"].median()), 1), "y": round(float(g["y"].median()), 1)}
        for name, g in c.assign(x=x, y=y, okrug=okrug.to_numpy()).groupby("okrug")
    ]
    return {
        "grid": hm.grid.as_dict(),
        "formula": "x = x0 + size·√3·(hq + hr/2), y = y0 − size·1,5·hr",
        "cell_km": hm.stats["cell_km"],
        "n_cells": hm.stats["n_cells"],
        "shift_km": {
            "median": int(round(hm.stats["shift_median_km"])),
            "p90": int(round(hm.stats["shift_p90_km"])),
            "max": int(round(hm.stats["shift_max_km"])),
        },
        "regions_split": hm.stats["groups_split"],
        "knn10_preserved": round(hm.stats["knn10_preserved"], 3),
        "paths": {
            "regions": site_hexgrid.borders(c["hq"], c["hr"], c["region_code"].to_numpy(), hm.grid),
            "okrugs": site_hexgrid.borders(c["hq"], c["hr"], okrug.to_numpy(), hm.grid),
        },
        "regions": regions,
        "okrugs": okrugs,
        "cities": city_labels(d, hm),
    }


def city_labels(d: SiteData, hm: HexMap) -> list[dict]:
    """Подписи крупнейших по ``pop_avg`` городов (узлы-города и городские округа): чтобы узнавалась карта."""
    n = int(d.cfg["site"]["build"]["hex"]["n_city_labels"])
    pop = node_pop(d)
    nodes = d.nodes
    cand = nodes[nodes["is_node"] & (nodes["is_city_node"] | (nodes["mo_type"].astype(str) == "go"))]
    top = pop.reindex(cand.index).dropna().sort_values(ascending=False).head(n)
    cells = hm.cells.set_index("territory_id")
    x, y = hm.grid.center(cells.loc[top.index, "hq"], cells.loc[top.index, "hr"])
    centers = d.terr.set_index("territory_id")["center_name"].to_dict() if "center_name" in d.terr else {}
    return [
        {
            "id": int(i),
            "name": city_name(nodes.loc[i], centers.get(int(i))),
            "x": round(float(a), 1),
            "y": round(float(b), 1),
        }
        for i, a, b in zip(top.index, x, y, strict=True)
    ]


def city_name(node: pd.Series, center: Any) -> str:
    """Подпись города: у городского округа — его центр без «г» («г Челябинск» -> «Челябинск»), иначе
    короткое имя. «Челябинский» (прилагательное из названия округа) на карте не читается как город."""
    c = str(center or "")
    if str(node.get("mo_type")) == "go" and c.startswith("г "):
        return c[2:].strip()
    return str(node["name_short"])


def node_pop(d: SiteData) -> pd.Series:
    """Население: ``pop_avg`` 2024 года, иначе 2023-го (``context_annual``; города — ``features_place``)."""
    ca = d.processed("context_annual", columns=["territory_id", "year", "pop_avg"])
    pl = d.processed("features_place", columns=["territory_id", "year", "pop_avg"])
    both = pd.concat([ca, pl[pl["territory_id"].isin(d.nodes.index[d.nodes["is_city_node"]])]])
    both = both.dropna(subset=["pop_avg"]).sort_values(["territory_id", "year"])
    return both.groupby("territory_id")["pop_avg"].last()


def first_screen_svg(d: SiteData, hm: HexMap, hexgrid: Mapping, mo: pd.DataFrame, story: Mapping) -> str:
    """Статичная SVG карты ячеек экрана 0 (работает без JS): ячейки по типам (цвет и подпись), без типа —
    штриховка, контуры регионов и округов, подписи городов и округов, выноски примеров типов."""
    g = hm.grid
    view = story["view"]
    colors = view["type_colors"]
    names = story["names"]["final"]
    cells = mo[mo["role"].isin(["territorial", "city", "untyped"])]
    parts = [
        # класс hexmap — по нему landing.css стилизует линии эго-сети, выделение и подписи
        '<svg id="hexmap" class="hexmap" xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {g.width:.0f} {g.height:.0f}" '
        f'role="img" aria-label="{_esc(map_summary(d, mo, story))}" data-hex-size="{g.size}" '
        f'data-hex-x0="{g.x0:.2f}" data-hex-y0="{g.y0:.2f}" data-hex-orient="pointy">',
        f"<title>{_esc(map_summary(d, mo, story))}</title>",
        '<defs><pattern id="hatch0" width="3" height="3" patternUnits="userSpaceOnUse" '
        f'patternTransform="rotate(45)"><rect width="3" height="3" fill="{style.NODATA}"/>'
        f'<line x1="0" y1="0" x2="0" y2="3" stroke="{style.HATCH}" stroke-width="1.2"/></pattern></defs>',
    ]
    order = [int(t) for t in view["legend_order"]]
    for t in [*order, 0]:
        sub = cells[cells["t"].fillna(0).astype(int) == t]
        if sub.empty:
            continue
        x, y = g.center(sub["hq"], sub["hr"])
        fill = "url(#hatch0)" if t == 0 else colors[str(t)]
        label = "нет типа" if t == 0 else names.get(str(t), f"Тип {t}")
        parts.append(
            f'<path class="cells-t{t}" fill="{fill}" stroke="#fff" stroke-width="0.6" '
            f'aria-label="{_esc(label)}: {len(sub)}" d="{site_hexgrid.hex_path(x, y, g.size)}"/>'
        )
    paths = hexgrid["paths"]
    parts.append(
        f'<path class="regions" fill="none" stroke="#595959" stroke-width="0.8" stroke-linejoin="round" '
        f'd="{paths["regions"]}"/>'
        f'<path class="okrugs" fill="none" stroke="#111" stroke-width="1.8" stroke-linejoin="round" '
        f'd="{paths["okrugs"]}"/>'
    )
    lab = [
        '<g class="labels" font-size="11" paint-order="stroke" stroke="#fff" stroke-width="3" fill="#222">'
    ]
    taken: list[tuple[float, float, float, float]] = []
    for c in hexgrid["cities"]:
        x, y = c["x"] + g.size, c["y"] + 4
        taken.append((x, y - LINE_H, x + CHAR_W * len(c["name"]), y))
        lab.append(f'<text x="{x:.1f}" y="{y:.1f}">{_esc(c["name"])}</text>')
    for o in hexgrid["okrugs"]:
        # подпись округа — в первом свободном месте у медианы его ячеек; нет места — без подписи
        half = CHAR_W * 1.1 * len(o["name"]) / 2
        for dy in (0, LINE_H + 2, -(LINE_H + 2), 2 * (LINE_H + 2)):
            box = (o["x"] - half, o["y"] + dy - LINE_H, o["x"] + half, o["y"] + dy)
            if _free(box, taken, g.width, g.height):
                taken.append(box)
                lab.append(
                    f'<text class="okrug" x="{o["x"]:.1f}" y="{o["y"] + dy:.1f}" text-anchor="middle" '
                    f'font-size="10" fill="#595959" letter-spacing="0.08em">{_esc(o["name"].upper())}</text>'
                )
                break
    lab.append("</g>")
    parts += lab
    parts += callouts(d, hm, mo, story, taken)
    parts.append("</svg>")
    return "".join(parts)


CHAR_W = 6.6  # ширина знака подписи 11 px с запасом, оценка для раскладки подписей
LINE_H = 13.0


def _free(box: tuple[float, float, float, float], taken: list, w: float, h: float) -> bool:
    x0, y0, x1, y1 = box
    if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
        return False
    return all(x1 < a or x0 > c or y1 < b or y0 > e for a, b, c, e in taken)


def callouts(d: SiteData, hm: HexMap, mo: pd.DataFrame, story: Mapping, taken: list) -> list[str]:
    """Выноски у типичного примера каждого типа (``examples.csv``, kind = typical, ранг 1): две строки
    «Название (регион)» и «— тип». Место — первое свободное из восьми вокруг ячейки (без наложений на другие
    подписи, в пределах холста). Без выгрузки примеров (``--demo``) выносок нет — остаётся ключ под картой."""
    ex = d.opt("examples.csv")
    if ex is None:
        return []
    ex = ex[(ex["kind"] == "typical") & (ex["rank"] == 1)].sort_values("type")
    rows = mo.set_index("id")
    g = hm.grid
    out = [
        '<g class="callout" font-size="11" paint-order="stroke" stroke="#fff" stroke-width="3" fill="#222">'
    ]
    offsets = [(1, -1), (1, 1), (-1, -1), (-1, 1), (1, -2.5), (-1, -2.5), (1, 2.5), (-1, 2.5)]
    for _, e in ex.iterrows():
        tid = int(e["territory_id"])
        if tid not in rows.index or pd.isna(rows.loc[tid, "hq"]):
            continue
        r = rows.loc[tid]
        x, y = g.center([r["hq"]], [r["hr"]])
        cx, cy = float(x[0]), float(y[0])
        shape = str(story["view"]["shapes"].get(str(int(e["type"])), ""))
        lines = [
            f"{shape} {cap(str(r['ns']))}".strip()
        ]  # порция 6b: было «Название (регион) — тип» в две строки
        w = CHAR_W * max(len(t) for t in lines)
        h = LINE_H * len(lines)
        cands = []
        for sx, sy in offsets:
            tx, ty = cx + sx * 16, cy + sy * 16
            box = (
                tx if sx > 0 else tx - w,
                ty - h if sy < 0 else ty,
                tx + w if sx > 0 else tx,
                ty if sy < 0 else ty + h,
            )
            cands.append((sx, sy, tx, ty, box))
        # первое свободное место; нет свободного — первое в пределах холста (наложение лучше обрезки)
        pick = next((c for c in cands if _free(c[4], taken, g.width, g.height)), None) or next(
            (c for c in cands if _free(c[4], [], g.width, g.height)), cands[0]
        )
        sx, sy, tx, ty, box = pick
        taken.append(box)
        anchor = "start" if sx > 0 else "end"
        y_first = box[1] + LINE_H - 3
        tspans = "".join(
            f'<tspan x="{tx:.1f}" y="{y_first + i * LINE_H:.1f}">{_esc(t)}</tspan>'
            for i, t in enumerate(lines)
        )
        out.append(
            f'<line x1="{cx:.1f}" y1="{cy:.1f}" x2="{tx:.1f}" y2="{ty:.1f}" stroke="#222" '
            f'stroke-width="0.8"/>'
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{g.size * 1.1:.1f}" fill="none" stroke="#222" '
            f'stroke-width="1.4"/><text text-anchor="{anchor}">{tspans}</text>'
        )
    out.append("</g>")
    return out


def map_summary(d: SiteData, mo: pd.DataFrame, story: Mapping) -> str:
    """Итоговая строка карты для aria-label (числа — из файлов)."""
    names = story["names"]["final"]
    nodes = mo[mo["role"].isin(["territorial", "city"])]
    counts = nodes["t"].value_counts()
    types = "; ".join(
        f"{names.get(str(t), f'Тип {t}')} — {style.fmt_num(int(counts.get(int(t), 0)))}"
        for t in story["view"]["legend_order"]
    )
    n_reg = int(d.nodes.loc[d.nodes["is_node"], "region_code"].nunique())
    n_untyped = int((mo["role"] == "untyped").sum())
    return (
        f"Карта равных ячеек: {style.fmt_num(len(nodes))} муниципалитетов в {n_reg} регионах, "
        f"Москва и Петербург — по одной ячейке. Типы: {types}. "
        f"Ещё {style.fmt_num(n_untyped)} без типа — штриховка"
    )


def _esc(s: str) -> str:
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


# --- mo.json ------------------------------------------------------------------------------------------------


def build_mo_frame(d: SiteData, hm: HexMap, nxy: pd.DataFrame, values: pd.DataFrame | None) -> pd.DataFrame:
    """Строка на каждое из 2190 МО и на 2 узла-города (``role`` = city): все поля контракта §5 ``mo.json``."""
    terr = d.terr.set_index("territory_id", drop=False).sort_index()
    nodes = d.nodes
    mem = d.members.set_index("territory_id")
    city_ids = [int(i) for i in nodes.index[nodes["is_city_node"]]]
    rows = []
    for tid, t in terr.iterrows():
        role = str(mem.loc[tid, "role"])
        node = mem.loc[tid, "node_id"]
        if role == "city_member":
            kind = "inner"
        elif tid in nodes.index and bool(nodes.loc[tid, "is_node"]):
            kind = "territorial"
        else:
            kind = "untyped"
        rows.append(
            {
                "id": int(tid),
                "n": t["name"],
                "ns": t["name_short"],
                "r": t["region_name"],
                "k": KIND_RU.get(str(t["mo_type"]), str(t["mo_type"])),
                "role": kind,
                "node": int(node) if kind != "untyped" and pd.notna(node) else None,
                "region_code": int(t["region_code"]),
            }
        )
    for cid in city_ids:
        c = nodes.loc[cid]
        rows.append(
            {
                "id": cid,
                "n": c["name"],
                "ns": c["name_short"] if pd.notna(c["name_short"]) else c["name"],
                "r": c["region_name"],
                "k": KIND_RU["city"],
                "role": "city",
                "node": cid,
                "region_code": int(c["region_code"]),
            }
        )
    mo = pd.DataFrame(rows).set_index("id", drop=False).rename_axis(None)
    node = mo["node"]
    # МО без типа: причина словами (ряд неполный — число месяцев из 24 и пропуски внутри ряда)
    un = mo["role"] == "untyped"
    src = nodes.reindex(mo.index[un])
    mo["why_null"] = None
    mo.loc[un, "why_null"] = [
        f"ряд трат неполный — есть {int(m)} из 24 месяцев" + (", с пропусками" if g else "")
        for m, g in zip(src["n_months"], src["has_internal_gap"], strict=True)
    ]
    by_node = lambda s: node.map(s)  # noqa: E731 — значение узла для МО (район столицы — значение города)
    mo["t"] = by_node(d.node_type)
    dyn = d.outputs("dynamics/nodes.csv")
    if not dyn.exists():
        raise MissingInputError(f"site: нет {dyn} — этап dynamics")
    dn = pd.read_csv(dyn).set_index("territory_id")
    mo["t23"] = by_node(d.tmap(dn["type_2023"]))
    mo["t24"] = by_node(d.tmap(dn["type_2024"]))
    mo["rel"] = by_node(dn["reliable"].astype(bool))
    hist = d.processed("dynamics_history", columns=["territory_id", "window_index", "type", "status"])
    hist = hist.sort_values(["territory_id", "window_index"])
    win = hist.groupby("territory_id")["type"].apply(
        lambda s: "".join(str(d.transfer.get(int(v), int(v))) for v in s)
    )
    st = hist.groupby("territory_id")["status"].apply(lambda s: "".join(STATUS_CODE[str(v)] for v in s))
    if not (win.str.len() == N_WINDOWS).all():
        raise QCError(f"site: в dynamics_history не у всех узлов {N_WINDOWS} окон")
    mo["win"], mo["st"] = by_node(win), by_node(st)
    # ячейка: у района столицы — ячейка города
    cells = hm.cells.set_index("territory_id")
    own = mo["role"].isin(["territorial", "city", "untyped"])
    key = mo["id"].where(own, mo["node"])
    mo["hq"] = key.map(cells["hq"])
    mo["hr"] = key.map(cells["hr"])
    if mo.loc[own | (mo["role"] == "inner"), ["hq", "hr"]].isna().any().any():
        raise QCError("site: не у всех МО есть ячейка")
    lay = nxy.set_index("territory_id") if len(nxy) else None
    mo["nx"] = by_node(lay["nx"]) if lay is not None else None
    mo["ny"] = by_node(lay["ny"]) if lay is not None else None
    pop = node_pop(d)
    mo["pop"] = mo["id"].map(pop).round()
    ca = d.processed("context_annual", columns=["territory_id", "year", "workplace_based"])
    mo["wp"] = mo["id"].map(ca.sort_values("year").groupby("territory_id")["workplace_based"].last())
    if "series_status" in terr.columns:
        ser = terr["series_status"].astype(str)
        mo["ser"] = mo["id"].map(ser.where(ser != "full"))  # только неполный ряд (пометка в карточке)
    fw = d.processed("features_windows", columns=["territory_id", "window", *[f"clr_rel_{p}" for p in PARTS]])
    for year in ("2023", "2024"):
        w = fw[fw["window"] == year].set_index("territory_id")[[f"clr_rel_{p}" for p in PARTS]]
        b = pd.Series([[_num(v) for v in row] for row in w.to_numpy()], index=w.index, dtype=object)
        # своё окно — у самого МО (есть у узлов); у района столицы — корзина города
        mo["b" + year[2:]] = mo["id"].where(mo["id"].isin(b.index), mo["node"]).map(b)
    # «почему этот тип»: значения признаков названия типа у узла
    feats = why_features(d)
    if values is not None and feats:
        mo["why"] = [
            [
                _num(values.at[n, f], 3) if n in values.index and f in values.columns else None
                for f in feats[t]
            ]
            if pd.notna(n) and pd.notna(t) and int(t) in feats
            else None
            for n, t in zip(mo["node"], mo["t"], strict=True)
        ]
    else:
        mo["why"] = None
    rh = d.processed("features_rhythm", columns=["territory_id", "own_reliable"]).set_index("territory_id")
    rm = d.processed(
        "features_rhythm_monthly",
        columns=["territory_id", "category", "month", "own"],
        filters=[("category", "==", "all")],
    )
    prof = rm.groupby(["territory_id", "month"])["own"].mean().unstack("month")
    reliable = rh.index[rh["own_reliable"].astype(bool)]
    prof = prof.reindex(reliable)
    rh_list = pd.Series(
        [[_num(v, 3) for v in row] for row in prof.to_numpy()], index=prof.index, dtype=object
    )
    mo["rh"] = by_node(rh_list)
    mo["nb"] = by_node(main_neighbors(d))
    extra = node_tables(d)
    for col, s in extra.items():
        mo[col] = by_node(s)
    return mo.drop(columns=["region_code"]).sort_values(["role", "id"], key=_role_key)


def _role_key(s: pd.Series) -> pd.Series:
    if s.name == "role":
        return s.map({"territorial": 0, "city": 1, "inner": 2, "untyped": 3})
    return s


def main_neighbors(d: SiteData) -> pd.Series:
    """До 10 соседей узла в основной сети корзин (``basket_dist``, ``is_main``) по убыванию веса."""
    k = int(d.cfg["site"]["similarity_layout"]["n_net_neighbors"])
    e = d.processed(
        "network_edges",
        columns=["rule", "source", "target", "weight", "is_main"],
        filters=[("rule", "==", "basket_dist"), ("is_main", "==", True)],
    )
    both = pd.concat(
        [
            e[["source", "target", "weight"]],
            e.rename(columns={"source": "target", "target": "source"})[["source", "target", "weight"]],
        ]
    ).sort_values(["source", "weight", "target"], ascending=[True, False, True])
    return both.groupby("source")["target"].apply(lambda s: [int(x) for x in s.head(k)])


def node_tables(d: SiteData) -> dict[str, pd.Series]:
    """Поузловые выгрузки этапа 5 (§5): сопоставимые (набор продукта), устойчивость, граничные, R1, соперник.
    Нет файла — поле пустое у всех (блок скрывается); ничего не пересчитывается."""
    out: dict[str, pd.Series] = {}
    k = int(d.cfg["site"]["n_similar_shown"])
    comp = d.opt("node_comparable.csv")
    if comp is not None:
        c = comp[comp["product"].astype(bool) & (comp["rank"] <= k)].sort_values(["territory_id", "rank"])
        pairs = lambda g: [[int(i), int(round(km))] for i, km in zip(g["other_id"], g["km"], strict=True)]  # noqa: E731
        out["sim"] = c.groupby("territory_id").apply(pairs, include_groups=False)
        b = comp[(comp["set"] == "B") & (comp["rank"] <= k)].sort_values(["territory_id", "rank"])
        if len(b):
            out["simb"] = b.groupby("territory_id").apply(pairs, include_groups=False)
    seed = d.opt("node_seed.csv")
    if seed is not None:
        for kind, col in (("variant", "rob_rule"), ("seed", "rob_seed")):
            s = seed[seed["kind"] == kind].set_index("territory_id")
            out[col] = s["n_same"].astype(int).astype(str) + "/" + s["n_runs"].astype(int).astype(str)
    mg = d.opt("node_margin.csv")
    if mg is not None:
        m = mg.set_index("territory_id")
        # граничное — узел ближе к центру другого типа, чем к своему (own_nearest = False): второй тип
        out["second"] = m["second_type"].where(~m["own_nearest"].astype(bool))
    r1 = d.opt("node_r1.csv")
    if r1 is not None:
        v = r1[r1["kind"] == "variant"]
        variants = sorted(v["variant"].unique())
        piv = v.pivot_table(index="territory_id", columns="variant", values="matched_type", aggfunc="first")
        piv = piv.reindex(columns=variants)
        out["var"] = pd.Series(
            [[_num(x, 0) for x in row] for row in piv.to_numpy()], index=piv.index, dtype=object
        )
    rv = d.opt("node_rival.csv")
    if rv is not None:
        best = rv[rv["sources"].astype(str).str.contains("T5:max_ami", regex=False)]
        out["rival"] = best.set_index("territory_id")["best_partition_label"]
    return out


def check_r1_shares(r1: Mapping, uf: Mapping) -> list[str]:
    """Доли «тот же тип» по вариантам (``checks.r1.variants`` — среди муниципалитетов с типом) против
    ``usefulness.type_flag.per_variant``: одно определение на странице и в отчёте (порция 6b)."""
    want = {v["variant"]: v for v in ((uf.get("type_flag") or {}).get("per_variant") or [])}
    bad = []
    for v in r1.get("variants") or []:
        w = want.get(v["variant"])
        if w is None:
            continue
        if int(w["n"]) != int(v["n"]) or abs(float(w["same_share"]) - v["same"] / v["n"]) > 1e-9:
            bad.append(f"{v['variant']}: {v['same']} из {v['n']} против {w['same_share']:.4f} из {w['n']}")
    return bad


def useful_texts(
    cfg: Config, tx: Mapping, useful_in: Mapping | None, mo_rows: Mapping[int, Mapping]
) -> dict | None:
    """Строки 6b из выходов usefulness (``site_useful``): доля случаев, оговорка о наборе R, пример, флаг."""
    if useful_in is None or not tx.get("useful"):
        return None
    uf = useful_in["facts"]
    ux = tx["useful"]
    border = (tx.get("findings") or {}).get("border") or {}
    words = cfg["usefulness"]["rule_share"]["words"]
    return {
        "use": site_useful.use_texts(ux, uf, words),
        "example": site_useful.example_texts(ux, uf, mo_rows),
        "flag": site_useful.flag_summary(border.get("flag", ""), border.get("flag_most") or {}, uf)
        if border.get("flag")
        else None,
        "flag_label": str(uf.get("label") or cfg["usefulness"]["type_flag"]["label"]),
    }


def mo_json(mo: pd.DataFrame) -> dict:
    """Колонки ``mo.json`` (массивы одинаковой длины)."""
    cols = [
        "id", "n", "ns", "r", "k", "role", "node", "why_null", "t", "t23", "t24", "win", "st", "rel",
        "hq", "hr", "nx", "ny", "pop", "wp", "ser", "b23", "b24", "why", "rh", "nb", "sim", "simb",
        "rob_rule", "rob_seed", "fl",
        "second", "var", "rival",
    ]  # fmt: skip
    ints = {"id", "node", "t", "t23", "t24", "hq", "hr", "nx", "ny", "pop", "second", "rival"}
    out: dict[str, list] = {}
    for c in cols:
        s = mo[c] if c in mo.columns else pd.Series([None] * len(mo), index=mo.index)
        if c in ints:
            out[c] = [None if pd.isna(v) else int(v) for v in s]
        elif c in ("rel", "wp"):
            out[c] = [None if pd.isna(v) else bool(v) for v in s]
        else:
            out[c] = [None if (not isinstance(v, list) and pd.isna(v)) else v for v in s]
    return out


# --- types.json, checks.json, methods.json ------------------------------------------------------------------


def clustering_values(cfg: Config) -> pd.DataFrame | None:
    """Признаки узлов так, как их видела кластеризация (корзина B и признаки места X до стандартизации) — для
    полос «почему этот тип». Нет ``outputs/cluster/final.json`` — None."""
    fin_p = cfg.dir("outputs") / "cluster" / "final.json"
    if not fin_p.exists():
        return None
    from types import SimpleNamespace

    from munnet.clustering import inputs as CI

    fin = json.loads(fin_p.read_text(encoding="utf-8"))
    g = fin["inputs"]["graph"]
    cp = SimpleNamespace(
        graph_rule=str(g["rule"]),
        graph_sparsify=str(g["sparsify"]),
        graph_k=int(g["k"]),
        attributes=tuple(fin["inputs"]["features"]),
        place_year=int(fin["inputs"]["place_year"]),
    )
    base = CI.load_inputs(cfg, cp).inputs
    vals = pd.DataFrame(
        np.hstack([base.B, base.X_raw.to_numpy(dtype=np.float64)]),
        columns=[*base.b_names, *base.x_names],
        index=np.asarray(base.ids, dtype=np.int64),
    )
    return vals


def why_features(d: SiteData) -> dict[int, list[str]]:
    """Признаки полос «почему»: части корзины и признак места из названия типа (``facts.names``)."""
    out = {}
    for t, v in (d.facts.get("names") or {}).items():
        if not isinstance(v, Mapping):
            continue
        feats = [str(p) for p in v.get("parts") or []][:2]
        if v.get("place"):
            feats.append(str(v["place"]))
        if feats:
            out[int(t)] = feats
    return out


def build_types(d: SiteData, mo: pd.DataFrame, story: Mapping, values: pd.DataFrame | None) -> list[dict]:
    """Четыре записи типов (§5): названия, правило, размер, профиль, корзина, полосы «почему», примеры."""
    from munnet.clustering.figures import FEATURE_LABELS

    view = story["view"]
    names = story["names"]
    nodes = mo[mo["role"].isin(["territorial", "city"])]
    pop = nodes["pop"].fillna(0)
    prof = d.opt("profile.csv")
    rules = d.opt("tree_rules.csv")
    ex = d.opt("examples.csv")
    ss = d.opt("settlement_shares.csv")  # «кто обычно» для паспортов главы 3 (порция 5b)
    who_by = (
        ss.set_index("type")
        if ss is not None and {"type", "cities", "large_cities"} <= set(ss.columns)
        else None
    )
    jac = pd.read_parquet(
        Path(d.cfg["paths"]["processed"]) / "cluster_final.parquet", columns=["type", "type_jaccard"]
    ).drop_duplicates("type")
    jac_by = {
        d.transfer.get(int(t), int(t)): float(j)
        for t, j in zip(jac["type"], jac["type_jaccard"], strict=True)
    }
    min_j = float(d.cfg["interpret"]["examples"]["unstable_type_jaccard"])  # «неустойчивый тип» ниже порога
    feats = why_features(d)
    unstable = list(view.get("unstable_parts") or [])
    out = []
    qs = np.linspace(0, 1, 21)
    for t in [int(x) for x in view["legend_order"]]:
        sel = nodes[nodes["t"] == t]
        rec: dict[str, Any] = {
            "t": t,
            "name": names["final"].get(str(t)),
            "name_descr": names["descriptive"].get(str(t)),
            "caption": names.get("caption"),
            "fig": view["shapes"].get(str(t)),
            "color": view["type_colors"].get(str(t)),
            "size": int(len(sel)),
            "size_territorial": int((sel["role"] == "territorial").sum()),
            "share_nodes": _num(len(sel) / max(len(nodes), 1), 4),
            "pop_share": _num(pop[sel.index].sum() / max(pop.sum(), 1), 4),
            "jaccard": _num(jac_by.get(t), 3),
            "unstable": (jac_by.get(t, 1.0) < min_j) if t in jac_by else None,
            "unstable_parts": [p.split(":", 1)[1] for p in unstable if p.startswith(f"{t}:")],
            "rule_text": None,
            "profile": [],
            "basket": None,
            "why": None,
            "examples": {"typical": [], "borderline": [], "largest": []},
            "who": None,
        }
        if who_by is not None and t in who_by.index:
            w = who_by.loc[t]
            rec["who"] = {"cities": _num(w["cities"], 4), "large_cities": _num(w["large_cities"], 4)}
        if rules is not None:
            r = rules[(rules["type"] == t) & rules["journalist"].astype(bool)]
            rec["rule_text"] = str(r["rule"].iloc[0]) if len(r) else None
        if prof is not None:
            p = prof[prof["type"] == t]
            rec["profile"] = [
                {
                    "feature": str(f),
                    "label": FEATURE_LABELS.get(str(f), str(f)),
                    "median": _num(m, 4),
                    "lo": _num(lo, 4),
                    "hi": _num(hi, 4),
                    "q25": _num(a, 4),
                    "q75": _num(b, 4),
                }
                for f, m, lo, hi, a, b in zip(
                    p["feature"], p["median"], p["median_lo"], p["median_hi"], p["q25"], p["q75"], strict=True
                )
            ]
            pb = p.set_index("feature")
            rec["basket"] = {
                k: [_num(pb.at[f"clr_rel_{x}", c], 4) if f"clr_rel_{x}" in pb.index else None for x in PARTS]
                for k, c in (("med", "median"), ("q25", "q25"), ("q75", "q75"))
            }
        if values is not None and t in feats:
            vt = values.reindex([int(i) for i in sel["id"]])
            rec["why"] = [
                {
                    "feature": f,
                    "label": FEATURE_LABELS.get(f, f),
                    "kind": "basket" if f.startswith("clr_rel_") else "place",
                    "q": [_num(v, 3) for v in np.nanquantile(values[f].to_numpy(dtype=float), qs)],
                    "med": _num(np.nanmedian(vt[f].to_numpy(dtype=float)), 3),
                    "lo": _num(np.nanquantile(vt[f].to_numpy(dtype=float), 0.25), 3),
                    "hi": _num(np.nanquantile(vt[f].to_numpy(dtype=float), 0.75), 3),
                }
                for f in feats[t]
                if f in values.columns
            ]
        if ex is not None:
            e = ex[ex["type"] == t]
            for kind in ("typical", "borderline", "largest"):
                ek = e[e["kind"] == kind].sort_values(["rank", "territory_id"], na_position="last")
                rec["examples"][kind] = [int(i) for i in ek["territory_id"]]
        out.append(rec)
    return out


def profile_reference(d: SiteData) -> dict[str, float]:
    """Медианы признаков по всем узлам (``profile.csv``, строки ``type = 0``) — вертикаль главы 3."""
    prof = d.opt("profile.csv")
    if prof is None:
        return {}
    p = prof[prof["type"] == 0]
    return {str(k): _num(v, 4) for k, v in zip(p["feature"], p["median"], strict=True) if pd.notna(v)}


def build_checks(d: SiteData, mo: pd.DataFrame, layout: Mapping, hm: HexMap) -> dict:
    """Проверки для глав 4–7 (§5): T1, T3, T5, T7, R1, потоки 4 × 4, сеть, раскладка и карта ячеек.
    Всё — из выгрузок этапов; нет выгрузки — None."""
    f = d.facts
    out: dict[str, Any] = {}
    t1 = (f.get("t1") or {}).get("per")
    out["t1"] = None
    if t1:
        out["t1"] = {
            name: {
                "rho_a": _num(v.get("rho_a"), 3),
                "rho_a_ci": [_num(x, 3) for x in v.get("rho_a_ci") or []],
                "rho_b": _num(v.get("rho_b"), 3),
                "rho_b_ci": [_num(x, 3) for x in v.get("rho_b_ci") or []],
                "overall": v.get("overall"),
                "beyond": v.get("beyond"),
                "line": "solid" if v.get("beyond") else ("dashed" if v.get("overall") else "none"),
                "med_a": [_num(x, 4) for x in v.get("med_a") or []],
                "med_b": [_num(x, 4) for x in v.get("med_b") or []],
                "best_rival": v.get("best_rival_label"),
                "best_rival_rho": _num(v.get("best_rival_rho"), 3),
            }
            for name, v in t1.items()
        }
    ctr = d.opt("controls.csv")
    out["t1_rivals"] = (
        None
        if ctr is None
        else [
            {
                "name": r["name"],
                "label": r["label"],
                "group": r["group"],
                "rho_b_catering": _num(r.get("rho_b_catering"), 3),
                "rho_b_retail": _num(r.get("rho_b_retail"), 3),
            }
            for r in ctr.to_dict("records")
        ]
    )
    out["t1_order"] = [int(x) for x in (f.get("ladder") or {}).get("order") or []]  # порядок med_a в t1
    out["t3"] = t3_checks(d)
    ami = d.opt("t5_ami.csv")
    out["t5"] = (
        None
        if ami is None
        else {
            "ami": [
                {"partition": p, "label": lab, "ami": _num(a, 4)}
                for p, lab, a in sorted(
                    zip(ami["partition"], ami["label"], ami["ami"], strict=True), key=lambda x: -x[2]
                )
            ],
        }
    )
    t7 = f.get("t7") or {}
    out["t7"] = (
        {
            "product": t7.get("product"),
            "median_error": {k: _num(v, 4) for k, v in (t7.get("median_error") or {}).items()},
            "median_error_abs": {k: _num(v, 4) for k, v in (t7.get("median_error_abs") or {}).items()},
            "diffs": {
                k: [_num(v[0], 4), [_num(x, 4) for x in v[1]]] for k, v in (t7.get("diffs") or {}).items()
            },
            "example": t7_example(d, t7),
            "k": int(d.cfg["interpret"]["tests"]["T7_utility"]["k"]),
            **{k: t7.get(k) for k in ("n_common", "n_known", "n_dropped")},
        }
        if t7.get("median_error")
        else None
    )
    r1 = d.opt("r1_runs.csv")
    out["r1"] = {
        "runs": None if r1 is None else r1.to_dict("records"),
        "unstable_label": (f.get("r1") or {}).get("unstable_label"),
        "circularity": (f.get("r1") or {}).get("circularity"),
        "variants": r1_variants(d),
    }
    out["flows"] = flows_matrix(d, mo)
    out["network"] = network_numbers(d)
    out["layout"] = dict(layout)
    out["hex"] = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in hm.stats.items()}
    return out


def t7_example(d: SiteData, t7: Mapping) -> dict | None:
    """Пример главы 6 (``facts.t7.example`` — МО с медианной ошибкой набора продукта): его ошибки по четырём
    наборам (``t7_errors.csv``) и оба набора с км по прямой (``node_comparable.csv``: продукт и B)."""
    ex = t7.get("example")
    if not ex:
        return None
    tid = int(ex["territory_id"])
    out: dict[str, Any] = {"territory_id": tid, "error": _num(ex.get("error"), 4)}
    err = d.opt("t7_errors.csv")
    if err is not None and (err["territory_id"] == tid).any():
        row = err[err["territory_id"] == tid].iloc[0]
        for kind, suf in (("errors", ""), ("errors_abs", "_abs")):
            cols = {s: f"err{suf}_{s}" for s in "ABCD"}
            out[kind] = {s: _num(row[c], 4) for s, c in cols.items() if c in row and pd.notna(row[c])}
        prod = t7.get("product")
        if prod and out["errors"].get(prod) is not None and ex.get("error") is not None:
            if abs(float(row[f"err_{prod}"]) - float(ex["error"])) > 1e-9:  # тот же МО, та же ошибка
                raise QCError(f"site: t7_errors.csv не сходится с facts.t7.example у {tid}")
    nc = d.opt("node_comparable.csv")
    if nc is not None:
        me = nc[nc["territory_id"] == tid].sort_values("rank")
        for kind, sel in (("members_p", me["product"].astype(bool)), ("members_b", me["set"] == "B")):
            pairs = zip(me.loc[sel, "other_id"], me.loc[sel, "km"], strict=True)
            out[kind] = [[int(o), _num(km, 0)] for o, km in pairs]
        if ex.get("members") and [o for o, _ in out["members_p"]] != [int(x) for x in ex["members"]]:
            raise QCError(f"site: node_comparable.csv не сходится с facts.t7.example.members у {tid}")
    return out


def r1_variants(d: SiteData) -> list[dict] | None:
    """Карты главы 7: по варианту R1 — узлы, где тип другой (``node_r1.csv``, ``same`` — как у этапа 5),
    число совпавших и ARI того же кандидата с основным расчётом (``outputs/cluster/variants.csv``)."""
    nr = d.opt("node_r1.csv")
    if nr is None:
        return None
    vp = d.outputs("cluster/variants.csv")
    ari = pd.read_csv(vp).set_index("variant")["ari_same_candidate_vs_main"].to_dict() if vp.exists() else {}
    out = []
    typed = set(int(i) for i in d.node_type.index)
    for v, g in nr[nr["kind"] == "variant"].groupby("variant", sort=True):
        # порция 6b: знаменатель — муниципалитеты с типом основного расчёта (у варианта «районы отдельно» —
        # без 242 районов столиц), как report.md §8 и usefulness.type_flag.per_variant
        g = g[g["territory_id"].astype(int).isin(typed)]
        same = g["same"].astype(bool)
        a = ari.get(str(v).split(":", 1)[-1])
        out.append(
            {
                "variant": str(v),
                "n": int(len(g)),
                "same": int(same.sum()),
                "ari": _num(a, 4) if a is not None else None,
                "diff_ids": sorted(int(x) for x in g.loc[~same, "territory_id"]),
            }
        )
    return out


def t3_checks(d: SiteData) -> dict | None:
    """T3 для главы 5 по ``robustness.main_run_verdict: check_section``: итоговый вердикт — по R1 (прогон,
    давший наименьший вердикт, ``facts.r1.source_run``; его числа — те, что этап 5 сказал в тексте T3),
    рядом — основной расчёт (``facts.t3.main``) с пометкой ``unstable_label``, если он выше итогового.
    Облака плацебо обоих прогонов — из ``t3_placebo.csv`` (scheme = main); ничего не пересчитывается."""
    f = d.facts
    pl = d.opt("t3_placebo.csv")
    main = (f.get("t3") or {}).get("main") or {}
    if pl is None or not main:
        return None
    test = "T3_reliable_placebo"
    r1 = f.get("r1") or {}
    src = (r1.get("source_run") or {}).get(test, "main")
    v_final = (f.get("verdicts_final") or {}).get(test)
    v_main = (f.get("verdicts_main") or {}).get(test, f.get("t3_verdict"))
    said = read_verdicts(d.cfg, f).by_test.get(test, {})
    unstable = v_main != v_final
    pct = float(d.cfg["interpret"]["tests"][test].get("percentile", 95))
    runs = {r: t3_run(pl, r, pct) for r in dict.fromkeys([src, "main"])}
    m = runs["main"]
    if m and main:  # сверка с facts.t3.main: та же формула, те же числа, иначе код 3
        bad = [
            k
            for k, v in (
                ("observed", main.get("n_reliable")),
                ("p95", main.get("p95")),
                ("median", main.get("median")),
            )
            if v is not None and (m[k] is None or abs(float(m[k]) - float(v)) > 1e-6)
        ]
        if bad:
            raise QCError(f"site: t3_placebo.csv не сходится с facts.t3.main: {bad}")
    fin = runs.get(src) or {}
    return {
        "verdict_final": v_final,
        "verdict_main": v_main,
        "unstable": unstable,
        "unstable_label": r1.get("unstable_label") if unstable else None,
        "percentile": pct,
        "final": {"run": src, **fin, "text_fields": said},  # text_fields — числа из текста T3 (дословно)
        "main": {"run": "main", **(m or {}), "passed": main.get("passed")},
    }


def t3_run(pl: pd.DataFrame, run: str, pct: float) -> dict | None:
    """Облако одного прогона (разбиение половин ``main``): ``pair = −1`` — наблюдение,
    ``pair ≥ 0`` — псевдогоды.
    Перцентиль и медиана — ``np.percentile`` и ``np.median``, как в ``interpret.placebo.t3_eval``."""
    g = pl[(pl["run"] == run) & (pl["scheme"] == "main")]
    if g.empty:
        return None
    cloud = g.loc[g["pair"] >= 0, "n_reliable"].to_numpy(dtype=np.float64)
    obs = g.loc[g["pair"] == -1, "n_reliable"]
    return {
        "placebo": [int(x) for x in cloud],
        "observed": int(obs.iloc[0]) if len(obs) else None,
        "p95": _num(float(np.percentile(cloud, pct)), 2) if len(cloud) else None,
        "median": _num(float(np.median(cloud)), 2) if len(cloud) else None,
    }


def flows_matrix(d: SiteData, mo: pd.DataFrame) -> dict:
    """Потоки 4 × 4 «тип окна 2023 → тип окна 2024» (``outputs/dynamics/nodes.csv``): всего и надёжных."""
    nodes = mo[mo["role"].isin(["territorial", "city"])]
    order = sorted({int(t) for t in nodes["t"].dropna()})
    tot = {f"{a}-{b}": 0 for a in order for b in order}
    rel = dict(tot)
    for a, b, r in zip(nodes["t23"], nodes["t24"], nodes["rel"], strict=True):
        if pd.isna(a) or pd.isna(b):
            continue
        k = f"{int(a)}-{int(b)}"
        tot[k] = tot.get(k, 0) + 1
        rel[k] = rel.get(k, 0) + int(bool(r) and a != b)
    return {"types": order, "total": tot, "reliable": rel}


def network_numbers(d: SiteData) -> dict:
    """Доля рёбер основной сети корзин внутри своего региона и медиана их длины, км."""
    e = d.processed(
        "network_edges",
        columns=["rule", "is_main", "same_region", "dist_km"],
        filters=[("rule", "==", "basket_dist"), ("is_main", "==", True)],
    )
    return {
        "n_edges": int(len(e)),
        "share_same_region": _num(float(e["same_region"].mean()) if len(e) else None, 4),
        "median_km": _num(float(e["dist_km"].median()) if len(e) else None, 0),
    }


def build_methods(d: SiteData) -> dict | None:
    """Таблица «метод × метрика» (``outputs/evaluate``): значения, z к случайному базису, допустимость."""
    cand_p, long_p = d.outputs("evaluate/candidates.csv"), d.outputs("evaluate/icvi_long.csv")
    if not (cand_p.exists() and long_p.exists()):
        log.warning("site: нет outputs/evaluate — methods.json пустой")
        return None
    cand = pd.read_csv(cand_p)
    lg = pd.read_csv(long_p)
    val = lg.pivot_table(index="candidate", columns="metric", values="value", aggfunc="first")
    z = lg.pivot_table(index="candidate", columns="metric", values="z", aggfunc="first")
    metrics = [m for m in ICVI_BETTER if m in val.columns]
    rows = []
    for r in cand.to_dict("records"):
        c = r["candidate"]
        rows.append(
            {
                "candidate": c,
                "method": r["method"],
                "family": r["family"],
                "k": int(r["k"]),
                "feasible": bool(r["feasible"]),
                "winner": bool(r["is_method_winner"]),
                "final": bool(r["is_final"]),
                "v": {m: _num(val.at[c, m], 4) if c in val.index else None for m in metrics},
                "z": {m: _num(z.at[c, m], 2) if c in z.index else None for m in metrics},
            }
        )
    return {
        "metrics": metrics,
        "labels": {m: ICVI_LABELS[m] for m in metrics},
        "better": {m: ICVI_BETTER[m] for m in metrics},
        "rows": rows,
        "source": "outputs/evaluate/candidates.csv, icvi_long.csv",
    }


def similarity(cfg: Config) -> tuple[dict, pd.DataFrame]:
    """Раскладка «по сходству трат» по правилу ``site.similarity_layout`` (``site_layout``): сводка
    и координаты ``nx``, ``ny`` выбранной раскладки (перестановки нет — пустая таблица)."""
    from munnet import site_layout

    if not (cfg.dir("outputs") / "cluster" / "final.json").exists():
        raise MissingInputError("site: нет outputs/cluster/final.json — этап cluster (раскладка главы 2)")
    res = site_layout.build(cfg)
    summary = res.summary()
    if res.chosen:
        summary["caption"] = fill(
            str(cfg["site"]["similarity_layout"]["caption"]),
            {"preserved": style.fmt_pct(res.preserved[res.chosen])},
        )
    return summary, res.nxy()


# --- выгрузки CSV и HTML ------------------------------------------------------------------------------------


def download_csvs(mo: pd.DataFrame, types: list[dict], checks: Mapping, story: Mapping) -> dict[str, bytes]:
    """``data/download/*.csv`` (CC BY-SA 4.0): все МО, типы, потоки; UTF-8 с BOM — открывается в Excel."""
    names = story["names"]["final"]
    tname = lambda t: None if pd.isna(t) else names.get(str(int(t)))  # noqa: E731
    m = pd.DataFrame(
        {
            "territory_id": mo["id"],
            "name": mo["n"],
            "region": mo["r"],
            "kind": mo["k"],
            "role": mo["role"],
            "node_id": mo["node"],
            "type": mo["t"],
            "type_name": mo["t"].map(tname),
            "type_2023": mo["t23"],
            "type_2024": mo["t24"],
            "reliable_change": mo["rel"],
            "why_no_type": mo["why_null"],
            "pop_avg": mo["pop"],
        }
    )
    ty = pd.DataFrame(
        [
            {
                "type": r["t"],
                "name": r["name"],
                "rule": r["rule_text"],
                "nodes": r["size"],
                "share_nodes": r["share_nodes"],
                "pop_share": r["pop_share"],
            }
            for r in types
        ]
    )
    fl = checks["flows"]
    flows = pd.DataFrame(
        [
            {"from_type": int(a), "to_type": int(b), "n": fl["total"][k], "n_reliable": fl["reliable"][k]}
            for k in fl["total"]
            for a, b in [k.split("-")]
        ]
    )
    for col in ("node_id", "type", "type_2023", "type_2024", "pop_avg"):
        m[col] = pd.array(m[col].round() if col == "pop_avg" else m[col], dtype="Int64")
    enc = lambda df: df.to_csv(index=False).encode("utf-8-sig")  # noqa: E731
    readme = (
        "Выгрузки лендинга munnet. Данные: СберИндекс (CC BY-SA 4.0), Росстат и ФНС в обработке "
        "«Если быть точным» "
        "(CC BY 4.0). Выгрузки распространяются на условиях CC BY-SA 4.0 с указанием источников.\n"
        "mo.csv — все муниципалитеты и два узла-города (role = city); тип района Москвы или Петербурга — "
        "тип города.\n"
        "types.csv — типы; flows.csv — смены типа между окнами 2023 и 2024 годов.\n"
    )
    return {
        "data/download/mo.csv": enc(m),
        "data/download/types.csv": enc(ty),
        "data/download/flows.csv": enc(flows),
        "data/download/README.txt": readme.encode("utf-8"),
    }


FALLBACK_TEMPLATE = """<!doctype html>
<html lang="$lang"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>$title</title><style>$css</style></head><body>
<main id="answer"><div class="map-frame">$first_screen_svg</div></main>
<script type="application/json" id="story">$story_json</script>
<script type="application/json" id="names">$names_json</script>
<script type="application/json" id="meta">$meta_json</script>
$data_inline
<script type="module">$js</script>
</body></html>
"""


def _t(s: Any) -> str:
    """Текст для HTML: экранирование и типографика ru-text (только показ)."""
    return nbsp(_esc(s)) if s else ""


def _dot(s: str) -> str:
    return s if s.rstrip().endswith((".", "!", "?", "…")) else s + "."


def _glosses(gloss: Mapping[str, str], texts: Iterable[str]) -> list[str]:
    """Пояснения терминов, которые встречаются в текстах (подстрока без регистра, ё = е)."""
    hay = norm(" ".join(texts))
    return [g for k, g in gloss.items() if norm(k) in hay]


def fig_html(t: int | None, shapes: Mapping[str, str]) -> str:
    if t is None:
        return '<i class="fig fig-t0" aria-hidden="true"></i>'
    return f'<i class="fig fig-t{int(t)}" aria-hidden="true">{_esc(shapes.get(str(int(t)), ""))}</i>'


def screen0_html(
    story: Mapping, meta: Mapping, hexgrid: Mapping | None, mo: pd.DataFrame | None
) -> dict[str, str]:
    """Экран 0 — вписан в HTML при сборке (§5, §7: без JS читаются вопрос, заголовок, пункты, оговорка, охват,
    ключ карты, примеры и подпись о смещении ячеек); JS только добавляет поведение. Тексты этапа 5 —
    дословно (типографика ru-text — только при показе), короткие заголовки — site.headlines."""
    s0 = story["screen0"]
    lab = s0.get("labels") or {}
    gloss = s0.get("gloss") or {}
    how = lab.get("how_checked", "Как проверяли")
    shapes = story["view"]["shapes"]
    names = story["names"]["final"]
    out: dict[str, str] = {}
    banner = meta.get("banner")
    out["banner_html"] = (
        f'<div class="banner" id="banner" role="status">{_t(banner)}</div>'
        if banner
        else '<div class="banner" id="banner" role="status" hidden></div>'
    )
    head = []
    if s0.get("intro"):
        head.append(f'<p class="intro" id="answer-intro">{_t(_dot(s0["intro"]))}</p>')
    # порция 5a: h1 страницы — заголовок первого экрана (site.build.texts.hero.title); заголовок по T1
    # остаётся    # заголовком раздела «Что проверяли» (h2) и главы 4
    tag = "h2" if s0.get("hero") else "h1"
    head.append(f'<{tag} id="answer-title" class="answer-h">{_t(s0["title"])}</{tag}>')
    head.append(f'<p class="lead" id="answer-lead">{_t(s0["lead"])}</p>')
    head += [f'<p class="gloss">{_t(_dot(g))}</p>' for g in _glosses(gloss, [s0["title"], s0["lead"]])]
    out["answer_head"] = "\n".join(head)

    items = []
    for i, (hs, texts) in enumerate(zip(s0.get("point_heads") or [], s0["points"], strict=False)):
        body = []
        if i == 0 and s0.get("question"):
            q = lab.get("question_label", "Вопрос")
            body.append(f'<p class="q"><b>{_t(q)}.</b> {_t(s0["question"])}</p>')
        body += [f"<p>{_t(x)}</p>" for x in texts]
        body += [f'<p class="gloss">{_t(_dot(g))}</p>' for g in _glosses(gloss, texts)]
        heads = "".join(f'<span class="pt-h">{_t(_dot(h))}</span> ' for h in hs).strip()
        items.append(
            f'<li class="pt{" extra" if i else ""}"><p class="pt-head">{heads}</p>'
            f'<details class="how"><summary>{_t(how)}</summary>{"".join(body)}</details></li>'
        )
    notes = []
    cav = list(s0.get("caveat") or [])
    if s0.get("caveat_head") or cav:
        body = "".join(f"<p>{_t(x)}</p>" for x in cav) + "".join(
            f'<p class="gloss">{_t(_dot(g))}</p>' for g in _glosses(gloss, cav)
        )
        notes.append(
            f'<div class="caveat"><p><b>{_t(lab.get("caveat_label", "Оговорка"))}.</b> '
            f"{_t(_dot(s0.get('caveat_head') or ''))}</p>"
            + (f'<details class="how"><summary>{_t(how)}</summary>{body}</details>' if body else "")
            + "</div>"
        )
    if s0.get("coverage"):
        notes.append(f'<p class="coverage">{_t(_dot(s0["coverage"]))}</p>')
    scope = " ".join(_dot(x) for x in (s0.get("scope"), s0.get("regions_note")) if x)
    notes.append(f'<p class="scope" id="answer-scope">{_t(scope)}</p>')
    out["answer_findings"] = (
        '<div class="findings">'
        f'<ol class="points" id="answer-points" aria-label="Три вывода">{"".join(items)}</ol>'
        f'<div class="notes">{"".join(notes)}</div></div>'
    )
    sh = (hexgrid or {}).get("shift_km") or {}
    out["map_shift"] = (
        f" (медианное смещение ячейки — {style.fmt_num(sh['median'])} км, "
        f"наибольшее — {style.fmt_num(sh['max'])} км)"
        if sh
        else ""
    )
    key, ex = [], []
    if mo is not None:
        nodes = mo[mo["role"].isin(["territorial", "city"])]
        counts = nodes["t"].value_counts()
        for t in story["view"]["legend_order"]:
            nm = names.get(str(t), f"Тип {t}")
            key.append(
                f"<li>{fig_html(int(t), shapes)}<span>{_t(nm)} · "
                f"{style.fmt_num(int(counts.get(int(t), 0)))}</span></li>"
            )
        n0 = int((mo["role"] == "untyped").sum())
        key.append(f"<li>{fig_html(None, shapes)}<span>нет типа · {style.fmt_num(n0)}</span></li>")
        rows = mo.set_index("id")
        for i in s0.get("examples") or []:
            if i in rows.index:
                r = rows.loc[i]
                nm = cap(str(r["ns"] if isinstance(r["ns"], str) else r["n"]))
                ex.append(
                    f'<a class="ex" href="#mo={int(i)}" data-go="{int(i)}">'
                    f"{fig_html(r['t'] if pd.notna(r['t']) else None, shapes)}"
                    f"<span>{_t(nm)}<br><small>{_t(r['r'])}</small></span></a>"
                )
    out["map_key"] = "".join(key)
    out["examples_html"] = "".join(ex)
    bits = [f"seed {meta['seed']}"] if meta.get("seed") is not None else []
    if meta.get("sha"):
        bits.append(f"код — коммит {meta['sha']}")
    if meta.get("facts_sha256"):
        bits.append(f"выводы — facts.json sha256 {str(meta['facts_sha256'])[:12]}")
    out["meta_line"] = _t("Воспроизводимость: " + " · ".join(bits)) if bits else ""
    out |= hero_parts(story, out["map_shift"], mo)
    return out


HERO_KEYS = (
    "brand", "brand_sub", "to_types", "to_map", "howto", "howto_touch", "more", "reset", "zoom_in",
    "zoom_out",
    "canvas_label", "checked_label", "to_flat", "to_3d", "flat_note", "picked_note",
)  # fmt: skip


def hero_parts(story: Mapping, map_shift: str, mo: pd.DataFrame | None) -> dict[str, str]:
    """Первый экран (порция 5a): вписанные при сборке надзаголовок, h1, пояснение, легенда типов в порядке
    ``view.legend_order`` с числом ячеек, ключ высоты и подпись с охватом (``descriptive.coverage`` —
    всегда)."""
    s0 = story["screen0"]
    hx = s0.get("hero") or {}
    out = {k: _t(hx.get(k, "")) for k in HERO_KEYS}
    out["hero_head"] = (
        f'<p class="h0-kicker" id="hero-kicker">{_t(hx.get("kicker", ""))}</p>'
        f'<h1 id="hero-title">{_t(hx.get("title") or s0["title"])}</h1>'
        f'<p class="h0-lede" id="hero-lede">{_t(hx.get("lede", ""))}</p>'
        + (f'<p class="h0-why" id="hero-why">{_t(_dot(hx["why"]))}</p>' if hx.get("why") else "")
    )
    colors = story["view"]["type_colors"]
    names = story["names"]["final"]
    leg = []
    if mo is not None:
        counts = mo[mo["role"].isin(["territorial", "city"])]["t"].value_counts()
        for t in story["view"]["legend_order"]:
            leg.append(
                f'<li><i class="hx" style="--c:{_esc(colors.get(str(t), ""))}" aria-hidden="true"></i>'
                f"<span>{_t(names.get(str(t), f'Тип {t}'))}</span>"
                f"<b>{style.fmt_num(int(counts.get(int(t), 0)))}</b></li>"
            )
        n0 = int((mo["role"] == "untyped").sum())
        leg.append(
            f'<li><i class="hx hx0" aria-hidden="true"></i><span>{_t(hx.get("untyped", "Без типа"))}</span>'
            f"<b>{style.fmt_num(n0)}</b></li>"
        )
    bars = "".join(
        f'<span style="height:{10 * k}px"><em>×{style.fmt_num(k / 2, 0 if k % 2 == 0 else 1)}</em></span>'
        for k in (1, 2, 4)
    )
    out["hero_legend"] = (
        f'<ul class="h0-types">{"".join(leg)}</ul>'
        f'<div class="h0-hkey"><div class="h0-bars" aria-hidden="true">{bars}</div>'
        f"<p>{_t(_dot(hx.get('height_key', '')))}</p></div>"
    )
    note = _dot(hx.get("cells_note", "") + (map_shift or ""))
    out["hero_note"] = " ".join(x for x in (_t(note), _t(_dot(s0.get("coverage") or ""))) if x)
    return out


def _json_script(obj: Any) -> str:
    """JSON для ``<script type="application/json">``: «</» экранируется, чтобы не закрыть тег."""
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")


def git_sha() -> str | None:
    """Коммит кода страницы; если код или конфиг изменены после коммита — с пометкой «+изменения»
    (страница собрана не из того, что лежит в коммите)."""
    import subprocess

    try:
        run = lambda *a: subprocess.run(  # noqa: E731
            ["git", *a], capture_output=True, text=True, timeout=10, check=True
        ).stdout.strip()
        sha = run("rev-parse", "--short=12", "HEAD") or None
        dirty = run("status", "--porcelain", "--untracked-files=no", "--", "src", "configs")
        return f"{sha}+изменения" if (sha and dirty) else sha
    except (OSError, subprocess.SubprocessError):
        return None


def render_html(
    story: Mapping,
    names: Mapping,
    meta: Mapping,
    svg: str,
    data: Mapping[str, Any],
    inline: bool,
    parts: Mapping[str, str] | None = None,
    fonts: str = "",
) -> tuple[bytes, int]:
    """``index.html`` из ``src/munnet/templates/landing.html`` (``string.Template``) со встроенными CSS и JS.
    Возвращает байты страницы и размер встроенных данных (для бюджета первого экрана)."""
    from string import Template

    tdir = Path(__file__).parent / "templates"
    html_p = tdir / "landing.html"
    tpl = html_p.read_text(encoding="utf-8") if html_p.exists() else FALLBACK_TEMPLATE
    tpl = nbsp(tpl)  # ru-text и для статичного текста шаблона (подстановки $… и теги не затрагиваются)
    css = (tdir / "landing.css").read_text(encoding="utf-8") if (tdir / "landing.css").exists() else ""
    css = (
        (fonts + "\n" + css) if fonts else css
    )  # @font-face из vendor/fonts (порция 5b), без внешних запросов
    js = (tdir / "landing.js").read_text(encoding="utf-8") if (tdir / "landing.js").exists() else ""
    js3d = (tdir / "landing3d.js").read_text(encoding="utf-8") if (tdir / "landing3d.js").exists() else ""
    blocks = (
        "\n".join(
            f'<script type="application/json" id="data-{k}">{_json_script(v)}</script>'
            for k, v in data.items()
            if v is not None
        )
        if inline
        else ""
    )
    hx = story["screen0"].get("hero") or {}
    title = (
        f"{hx['brand']} — {hx.get('brand_sub', '')}".rstrip(" —")
        if hx.get("brand")
        else f"{story['screen0']['title']} — munnet"
    )
    if meta.get("banner"):
        title = f"{meta['banner']}. {title}"
    parts = dict(parts or {})
    for k in (
        "banner_html",
        "answer_head",
        "answer_findings",
        "map_shift",
        "map_key",
        "examples_html",
        "meta_line",
        "chapters_html",
        "sources_html",
        "findings_html",
        "hero_head",
        "hero_legend",
        "hero_note",
        *HERO_KEYS,
    ):
        parts.setdefault(k, "")
    page = Template(tpl).substitute(
        **parts,
        title=_esc(title),
        lang="ru",
        css=css,
        js=js,
        js3d=js3d,
        story_json=_json_script(story),
        names_json=_json_script(names),
        first_screen_svg=svg,
        data_inline=blocks,
        meta_json=_json_script(meta),
    )
    return page.encode("utf-8"), len(blocks.encode("utf-8"))


def check_budget(payloads: Mapping[str, bytes], budget: Mapping[str, float], inline_bytes: int) -> list[str]:
    """Вес страницы — ``index.html`` плюс данные, которые она не встроила (выгрузки CSV не загружаются);
    первый экран — ``index.html`` без встроенных блоков данных. Больше ``budget_mb.hard`` — код 3; мягкие
    пределы — предупреждение в логе."""
    mb = 2**20
    html = len(payloads.get("index.html", b""))
    first = (html - inline_bytes) / mb
    # сторонние модули (vendor/*.js, порция 5a) грузит страница — они входят в её вес
    # и шрифты (порция 5b): считаются все подмножества woff2, хотя браузер грузит только нужные
    vendor = (
        sum(len(b) for n, b in payloads.items() if n.startswith("vendor/") and n.endswith((".js", ".woff2")))
        / mb
    )
    if inline_bytes:
        page = html / mb + vendor
    else:
        page = (
            sum(len(b) for n, b in payloads.items() if n == "index.html" or n.endswith(".json")) / mb + vendor
        )
    log.info("site: страница %.2f МБ (из них vendor %.2f МБ), первый экран %.2f МБ", page, vendor, first)
    if page > float(budget["total"]):
        log.warning("site: страница %.2f МБ больше budget_mb.total %.2f", page, float(budget["total"]))
    if first > float(budget["first_screen"]):
        log.warning(
            "site: первый экран %.2f МБ больше budget_mb.first_screen %.2f", first, budget["first_screen"]
        )
    if page > float(budget["hard"]):
        return [f"страница {page:.2f} МБ больше budget_mb.hard {budget['hard']} МБ"]
    return []


VENDOR = ("three.module.min.js", "OrbitControls.js", "three-LICENSE.txt", "README.md")
# шрифты (порция 5b): Golos Text и Unbounded, SIL OFL 1.1, файлы Fontsource 5.3.0 — templates/vendor/fonts
FONTS_DIR = "fonts"
FONTS_CSS = "fonts/fonts.css"


def _vendor_dir() -> Path:
    return Path(__file__).parent / "templates" / "vendor"


def _checked(name: str, readme: str) -> bytes:
    """Байты файла ``templates/vendor/<name>`` со сверкой sha256 по ``templates/vendor/README.md`` (код 3)."""
    b = (_vendor_dir() / name).read_bytes()
    want = re.search(rf"`{re.escape(name)}`.*?`([0-9a-f]{{64}})`", readme)
    if not want or hashlib.sha256(b).hexdigest() != want[1]:
        raise QCError(f"site: vendor/{name}: sha256 не совпал с templates/vendor/README.md")
    return b


def font_files() -> dict[str, bytes]:
    """Шрифты страницы (woff2, fonts.css, тексты OFL) — в ``vendor/fonts/`` рядом с ``index.html``; sha256
    каждого файла сверяются с ``templates/vendor/README.md`` (расхождение — код 3). Внешних запросов нет."""
    vdir = _vendor_dir()
    readme = (vdir / "README.md").read_text(encoding="utf-8")
    out = {}
    for f in sorted((vdir / FONTS_DIR).iterdir()):
        if f.is_file():
            name = f"{FONTS_DIR}/{f.name}"
            out[f"vendor/{name}"] = _checked(name, readme)
    if f"vendor/{FONTS_CSS}" not in out:
        raise QCError(f"site: нет templates/vendor/{FONTS_CSS}")
    return out


def fonts_css(files: Mapping[str, bytes]) -> str:
    """``@font-face`` для встраивания в ``<style>`` страницы: пути ``url(fonts/…)`` из ``fonts.css`` (они
    относительно самого файла) переписываются на ``vendor/fonts/…`` — относительно ``index.html``."""
    css = files[f"vendor/{FONTS_CSS}"].decode("utf-8")
    return re.sub(r"url\((['\"]?)fonts/", r"url(\g<1>vendor/fonts/", css)


def vendor_files() -> dict[str, bytes]:
    """three.js 0.170.0 (MIT) из ``templates/vendor`` — в ``vendor/`` рядом с ``index.html`` (import map
    страницы, без CDN); sha256 сверяются с ``templates/vendor/README.md`` — расхождение (файл заменён) —
    код 3."""
    vdir = _vendor_dir()
    readme = (vdir / "README.md").read_text(encoding="utf-8")
    out = {}
    for name in VENDOR:
        out[f"vendor/{name}"] = (vdir / name).read_bytes() if name == "README.md" else _checked(name, readme)
    return out


def build_scene(
    cfg: Config, story: dict, mo: pd.DataFrame, hm: HexMap, values: pd.DataFrame | None
) -> dict | None:
    """``scene.json`` объёмной карты (``site_scene``) и линт подписей островов; нет значений признаков
    кластеризации — None (первый экран — статичная карта)."""
    hx = (cfg["site"]["build"].get("texts") or {}).get("hero") or {}
    sp = cfg["site"]["build"].get("scene")
    if not hx or not sp:
        return None
    try:
        scene = site_scene.build_scene(
            mo,
            hm.grid,
            values,
            [int(t) for t in story["view"]["legend_order"]],
            story["names"]["final"],
            story["view"]["type_colors"],
            hx,
            sp,
        )
    except ValueError as e:
        raise QCError(str(e)) from e
    if scene is None:
        log.warning("site: нет значений признаков кластеризации — объёмной карты не будет, только статичная")
        return None
    scene["unit_h"] = float(sp["unit"])
    bad = lint_texts(cfg, story, [x for i in scene["islands"] for x in (i["name"], i["note"])])
    if bad:
        raise QCError("site: линт подписей островов: " + "; ".join(bad[:10]))
    return scene


def cells_geo(hm: HexMap, mo: pd.DataFrame) -> dict:
    """Центры ячеек карты экрана 0 (узлы и МО без типа) — для маленьких карт глав 6 и 7."""
    c = mo[mo["role"].isin(["territorial", "city", "untyped"]) & mo["hq"].notna()]
    x, y = hm.grid.center(c["hq"], c["hr"])
    return {
        "w": round(hm.grid.width, 1),
        "h": round(hm.grid.height, 1),
        "xy": {int(i): (float(a), float(b)) for i, a, b in zip(c["id"], x, y, strict=True)},
    }


def method_numbers(story: Mapping, mo: pd.DataFrame, checks: Mapping, methods: Mapping | None) -> dict:
    """Числа пяти шагов главы 9 — из выгрузок страницы (ничего не пересчитывается)."""
    final = next((r for r in (methods or {}).get("rows") or [] if r.get("final")), None)
    n_types = len(story["view"].get("legend_order") or [])
    nodes = mo["role"].isin(["territorial", "city"])
    return {
        "n_mo": style.fmt_num(int((mo["role"] != "city").sum())),  # строки узлов-городов — не МО
        "n_nodes": style.fmt_num(int(nodes.sum())),
        "n_edges": style.fmt_num((checks.get("network") or {}).get("n_edges") or 0),
        "n_types": f"{n_types} {plural_ru(n_types, 'тип', 'типа', 'типов')}",
        "method": site_chapters_tail.METHOD_WORDS.get(final["method"], final["method"]) if final else "",
        "n_cand": style.fmt_num(len((methods or {}).get("rows") or [])),
        "n_windows": style.fmt_num(N_WINDOWS),
    }


def rows_of(cols: Mapping[str, list]) -> list[dict]:
    """Колонки ``mo.json`` -> записи (как ``rowsOf`` в landing.js)."""
    keys = list(cols)
    n = len(cols[keys[0]]) if keys else 0
    return [{k: cols[k][i] for k in keys} for i in range(n)]


def names_index(mo: pd.DataFrame) -> dict:
    """Индекс поиска (встроен в страницу): id, короткое название (полное — в ``mo.json``), регион, тип.
    Регион — номер в списке ``rl`` (77 названий вместо 2192 строк: так первый экран укладывается в бюджет)."""
    regions = sorted({str(r) for r in mo["r"]})
    pos = {r: i for i, r in enumerate(regions)}
    return {
        "id": [int(i) for i in mo["id"]],
        "n": [s if isinstance(s, str) else n for s, n in zip(mo["ns"], mo["n"], strict=True)],
        "r": [pos[str(r)] for r in mo["r"]],
        "t": [None if pd.isna(t) else int(t) for t in mo["t"]],
        "rl": regions,
    }


# --- запуск -------------------------------------------------------------------------------------------------


def run(cfg: Config, dev_blind: str | Path | None = None, demo: str | Path | None = None) -> dict:
    """Собрать лендинг. Коды выхода — ``site.exit_codes``; возвращает story.json (для тестов)."""
    mode = resolve_mode(cfg, dev_blind, demo)
    facts = load_facts(cfg, mode)
    numbers = site_numbers(cfg)
    rep = check_headlines(cfg.data)
    if not rep.ok:
        raise QCError("site: словарь заголовков: " + "; ".join(rep.problems()[:10]))
    audit_palettes(cfg, facts)
    story = build_story(cfg, facts, numbers, mode)
    bad = lint_story(cfg, story, facts)
    if bad:
        raise QCError("site: линт текста: " + "; ".join(bad[:10]))
    d = load_site_data(cfg, mode, facts)
    bad = check_controls(d)
    if bad:
        raise QCError("site: контрольные числа: " + "; ".join(bad[:10]))
    # порция 6b: выходы этапа usefulness — необязательный вход (нет — блока пользы и флага нет;
    # устарел — код 1)
    useful_in = site_useful.load(cfg, mode)
    layout, nxy = similarity(cfg)
    story["chapters"]["similarity"]["layout_caption"] = layout.get("caption")
    story["view"]["similarity_layout"] = layout.get("chosen")
    hm = build_hexmap(d)
    values = clustering_values(cfg)
    mo = build_mo_frame(d, hm, nxy, values)
    flag_codes: dict[int, str] = {}
    if useful_in is not None:
        codes, words = site_useful.flag_codes(useful_in["flags"])
        nodes_mo = mo[mo["role"].isin(["territorial", "city"])].set_index("id")
        rob = {c: nodes_mo[c] for c in ("rob_rule", "rob_seed") if c in nodes_mo.columns}
        bad = site_useful.check_flags(useful_in["flags"], d.node_type, rob)
        if bad:
            raise QCError("site: флаг устойчивости типа: " + "; ".join(bad[:5]))
        flag_codes = {int(k): str(v) for k, v in codes.items()}
        mo["fl"] = mo["node"].map(lambda n: flag_codes.get(int(n)) if pd.notna(n) else None)
        story["card"]["flag_words"] = words
    types = build_types(d, mo, story, values)
    checks = build_checks(d, mo, layout, hm)
    if useful_in is not None:  # те же доли «тот же тип», что у usefulness.type_flag.per_variant (иначе код 3)
        bad = check_r1_shares(checks.get("r1") or {}, useful_in["facts"])
        if bad:
            raise QCError("site: доли «тот же тип» не совпали с usefulness: " + "; ".join(bad))
    if checks["t1"]:
        story["view"]["line_by_turnover"] = {k: v["line"] for k, v in checks["t1"].items()}
    hexgrid = build_hexgrid_json(d, hm, mo.set_index("id")["t"])
    scene = build_scene(cfg, story, mo, hm, values)
    if scene is not None and flag_codes:  # подсказка объёмной карты: флаг после типа (порция 6b)
        scene["cells"]["f"] = [flag_codes.get(int(i)) for i in scene["cells"]["id"]]
    methods = build_methods(d)
    story["meta"]["pending"] = ["geo.json", "munnet_landing.pdf"]
    story["meta"]["type_source"] = d.type_source
    meta = {
        "seed": cfg["seed"],
        "sha": git_sha(),
        "mode": mode.name,
        "banner": mode.banner,
        "licenses": LICENSES,
        "site_freeze_sha256": story["meta"]["site_freeze_sha256"],
        "facts_sha256": story["meta"]["facts_sha256"],
    }
    data = {
        "mo": mo_json(mo),
        "types": {"types": types, "parts": list(PARTS), "part_labels": PART_LABELS},
        "checks": checks,
        "methods": methods,
        "hexgrid": hexgrid,
        "scene": scene,
    }
    story["screen0"]["examples"] = screen_examples(types, story)
    data["types"]["reference"] = profile_reference(d)
    svg = first_screen_svg(d, hm, hexgrid, mo, story)
    mo_rows = {int(r["id"]): r for r in rows_of(data["mo"])}
    # порция 5b: блок «Что устояло», строки карточки о сверке, паспорта типов (тексты — site.build.texts)
    tx = cfg["site"]["build"].get("texts") or {}
    useful = useful_texts(cfg, tx, useful_in, mo_rows)
    ft = site_findings.findings_texts(tx.get("findings") or {}, story, facts, checks, useful)
    if useful_in is not None and ft is None:
        log.warning("site: блок «Что устояло» не собран — доля случаев и пример пользы не показаны")
    if tx.get("findings") and ft is None:
        log.warning(
            "site: блок «Что устояло» не показан — вердикты или числовые условия не те, "
            "под которые он написан"
        )
    story["card"].update(site_findings.card_texts(tx.get("card"), checks))
    ptx = tx.get("passports")
    bad = lint_texts(
        cfg,
        story,
        site_findings.strings(ft)
        + [str(v) for v in story["card"].values() if isinstance(v, str)]
        + list((story["card"].get("flag_words") or {}).values())
        + site_findings.passport_strings(types, ptx)
        + site_useful.strings(
            (useful or {}).get("use"), (useful or {}).get("example"), (useful or {}).get("flag")
        ),
    )
    banned_h = list(cfg["site"]["forbidden_words"]["headlines_always"]) + list(
        cfg["interpret"]["naming"]["banned"]
    )
    bad += [f"запрет в заголовке: «{w}» в «{h}»" for h in site_findings.titles(ft) for w in banned_h
            if norm(w) in norm(h)]  # fmt: skip
    if bad:
        raise QCError("site: линт текстов порции 5b: " + "; ".join(bad[:10]))
    passports = site_findings.passports_html(types, story, mo_rows, ptx, _t)
    csvs = download_csvs(mo, types, checks, story)
    story["chapters"]["method"]["numbers"] = method_numbers(story, mo, checks, methods)
    geo = cells_geo(hm, mo)
    chapters = "\n".join(
        [
            site_chapters.chapters_html(
                story,
                types,
                checks,
                mo_rows,
                data["types"]["reference"],
                _t,
                passports,
                (ptx or {}).get("detail"),
            ),
            site_chapters_tail.chapters_html(story, checks, mo_rows, geo, methods, meta, _t),
        ]
    )
    sources = site_chapters_tail.sources_html(
        story, {k.rsplit("/", 1)[-1]: len(b) for k, b in csvs.items()}, _t
    )
    bad = lint_templates(cfg, story, svg)
    if bad:
        raise QCError("site: линт шаблона: " + "; ".join(bad[:10]))
    inline = bool(cfg["site"]["build"].get("inline_all", True))
    order = [int(x) for x in checks.get("t1_order") or facts["ladder"]["order"]]
    findings = site_findings.findings_html(ft, story, order, _t, site_chapters.UI["src_rosstat"])
    parts = screen0_html(story, meta, hexgrid, mo) | {
        "chapters_html": chapters,
        "sources_html": sources,
        "findings_html": findings,
    }
    fonts = font_files()
    html, inline_bytes = render_html(story, names_index(mo), meta, svg, data, inline, parts, fonts_css(fonts))
    payloads: dict[str, bytes] = {"index.html": html, "data/story.json": dumps(story)}
    for k, v in data.items():
        if v is not None:
            payloads[f"data/{k}.json"] = dumps(v)
    payloads.update(csvs)
    payloads.update(fonts)
    if scene is not None:
        payloads.update(vendor_files())
    over = check_budget(payloads, cfg["site"]["budget_mb"], inline_bytes)
    if over:
        raise QCError("site: бюджет: " + "; ".join(over))
    mode.out.mkdir(parents=True, exist_ok=True)
    for name, b in payloads.items():
        p = mode.out / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b)
    log.info(
        "site (%s): %d файлов в %s (%s); строк mo.json %d; ячеек %d%s",
        mode.name,
        len(payloads),
        mode.out,
        ", ".join(
            f"{n} {len(b) / 1024:.0f} КБ" for n, b in payloads.items() if not n.startswith("data/download")
        ),
        len(mo),
        hm.stats["n_cells"],
        f"; плашка «{mode.banner}»" if mode.banner else "",
    )
    return story
