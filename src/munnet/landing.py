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

TODO (следующая порция, план §10, п. 5): ``mo.json``, ``types.json``, ``checks.json``, ``methods.json``,
``hexgrid.json``, ``geo.json``, выгрузки CSV, ``index.html`` и PDF; контрольные числа (1776, 169, 247,
размеры типов) против файлов (поузловая сверка типов с ``cluster_final`` и sha256 меток — уже
в ``check_bound_to_labels``); подпись раскладки ``{preserved}`` и реальные МО ролей. Пока их нет, обычный
режим после всех проверок завершается кодом 2 и ничего не пишет в ``site/``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from munnet import style
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
# Выходы, которые собирает следующая порция (контракт §5); сейчас — заглушки
PENDING_OUTPUTS = ("mo.json", "types.json", "checks.json", "methods.json", "hexgrid.json", "index.html")
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
    return {
        "share_cross": style.fmt_pct(1 - float(main["same_region"].mean())),
        "share_cross_random": style.fmt_pct(1 - float((n_r * (n_r - 1)).sum()) / (big_n * (big_n - 1))),
        "n_rhythm": style.fmt_num(n_rhythm),
        "share_rhythm": str(share["text"]),
        "share_rhythm_null": str(ef["syn.null_reliable_share_nodes"]["text"]),
    }


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


def extract_fields(template: str, text: str) -> dict[str, str] | None:
    """Поля, которые этап 5 подставил в ``template``, чтобы получить ``text``; None — не тот шаблон."""
    m = template_regex(_lower_first(template)).match(_lower_first(text))
    return None if m is None else m.groupdict()


@dataclass
class Verdicts:
    """Ключи исходов для словаря заголовков и флаги сочетания (как в переборе ``site_headlines``)."""

    keys: dict[str, Any]  # тест -> ключ заголовка; плюс T7_type_gain, one_in_ten, caveat
    fields: dict[str, str] = field(default_factory=dict)  # поле -> значение из текстов этапа 5

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
    for test in TESTS:
        v = vf.get("T1_text", vf[test]) if test == "T1_ladder_external" else vf[test]
        cands = ["partial_capped", "partial"] if (test == "T2_direction" and v == "partial") else [v]
        for key in cands:
            got = extract_fields(it[test]["outcomes"][key]["text"], texts[test])
            if got is not None:
                keys[test] = key
                for name, val in got.items():
                    if name in fields and fields[name] != val:
                        raise QCError(
                            f"site: поле {{{name}}} в текстах этапа 5 разное: {fields[name]!r} и {val!r}"
                        )
                    fields[name] = val
                break
        else:
            raise QCError(
                f"site: текст {test} не совпал с шаблоном исхода {cands} (вердикт {v}): {texts[test]!r}"
            )
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
    return Verdicts(keys, fields)


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


def _headline(H: Mapping, test: str, key: str, fields: Mapping[str, str]) -> str:
    return cap(fill(H[test][key], fields))


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
        "title": _headline(H, s0["title"], v[s0["title"]], f),
        "lead": _headline(H, s0["lead"], v[s0["lead"]], f),
        "question": th["question"],
        "points": [list(th["point_1"]), list(th["point_2"]), list(th["point_3"])],
        "caveat": list(th["caveat"]),
        "t6_caveat": H["T6_caveat"]["caveat"] if v["caveat"] else None,
        "scope": fill(A["scope_reader"], scope_vals),
        "coverage": H["descriptive"]["coverage"],
    }
    dyn_lead = (
        _headline(H, "T2_direction", t2, f)
        if ok_if(ch["dynamics"]["lead_if"])
        else H["descriptive"]["dynamics_neutral"]
    )
    limits_title = (
        _headline(H, "T6_bank_coverage", t6, f)
        if ok_if(ch["limits"]["title_if"])
        else H["descriptive"]["limits_neutral"]
    )
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
            "title": _headline(H, "T5_trivial", t5, f),
            "note": H["T4_basket_vs_place"]["describe"],
            "text": texts["T5_trivial"],
            "note_text": texts["T4_basket_vs_place"],
        },
        "order": {
            "title": _headline(H, "T1_ladder_external", t1, f),
            "text": texts["T1_ladder_external"] + facts.get("t1_notes", {}).get("T1_ladder_external", ""),
            "proxies": texts.get("T1_proxies", ""),
        },
        "dynamics": {
            "title": _headline(H, "T3_reliable_placebo", t3, f),
            "lead": dyn_lead,
            "texts": [texts["T3_reliable_placebo"], texts["T2_direction"]],
            "layer_name": flows["layer_name"],
        },
        "comparable": {
            "title": _headline(H, "T7_utility", t7, f),
            "lead": _headline(H, "T7_type_gain", v["T7_type_gain"], f),
            "text": texts["T7_utility"],
            "same_period": A["honesty"]["t7_same_period"],
            "similar_caption": fill(
                site["similar_caption"], {"n_shown": str(site["n_similar_shown"]), "n_set": str(n_set)}
            ),
        },
        "limits": {
            "title": limits_title,
            "lead": [H["descriptive"]["coverage"]] + ([H["T6_caveat"]["caveat"]] if v["caveat"] else []),
            "text": texts["T6_bank_coverage"],
        },
        "explore": {"title": H["descriptive"]["explore"], "nodata": A["honesty"]["nodata"]},
        "method": {"title": H["descriptive"]["method"]},
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
            "pending": list(PENDING_OUTPUTS),
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
        "view": view,
    }


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
        {k: story[k] for k in ("screen0", "chapters", "roles", "names", "reliability_key")}
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
    headlines = [story["screen0"]["title"], story["screen0"]["lead"]] + [
        c[k] for c in story["chapters"].values() for k in ("title", "lead") if isinstance(c.get(k), str)
    ]
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


# --- бюджет и запись ----------------------------------------------------------------------------------------


def dumps(obj: Any) -> bytes:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def check_budget(payloads: Mapping[str, bytes], budget: Mapping[str, float]) -> list[str]:
    """Жёсткий предел ``budget_mb.hard`` на всю страницу — код 3; мягкие пределы — предупреждение в логе."""
    mb = sum(len(b) for b in payloads.values()) / 2**20
    if mb > float(budget["total"]):
        log.warning("site: страница %.2f МБ больше budget_mb.total %.2f", mb, float(budget["total"]))
    first = sum(len(b) for n, b in payloads.items() if n in ("index.html", "data/story.json")) / 2**20
    if first > float(budget["first_screen"]):
        log.warning("site: первый экран %.2f МБ больше budget_mb.first_screen", first)
    if mb > float(budget["hard"]):
        return [f"страница {mb:.2f} МБ больше budget_mb.hard {budget['hard']} МБ"]
    return []


# Заглушки выходов следующей порции (контракт §5): каждая вернёт байты файла в site/data/
Builder = Callable[[Config, Mapping], bytes | None]


def build_mo(cfg: Config, facts: Mapping) -> bytes | None:
    """TODO: mo.json — 2190 строк колонками (§5): тип, окна, узел, соседи, корзина, устойчивость."""
    return None


def build_types(cfg: Config, facts: Mapping) -> bytes | None:
    """TODO: types.json — 4 записи: названия, правило, размер, профиль, примеры, фигуры."""
    return None


def build_checks(cfg: Config, facts: Mapping) -> bytes | None:
    """TODO: checks.json — T1, T3, T5, T7, R1, потоки 4 × 4 из outputs/interpret и outputs/dynamics."""
    return None


def build_methods(cfg: Config, facts: Mapping) -> bytes | None:
    """TODO: methods.json — методы × SW, CH, S_Dbw, AVI, AVU, MQ из outputs/evaluate."""
    return None


def build_hexgrid(cfg: Config, facts: Mapping) -> bytes | None:
    """TODO: hexgrid.json — ячейки, контуры регионов, подписи (site.build.hex)."""
    return None


BUILDERS: dict[str, Builder] = {
    "data/mo.json": build_mo,
    "data/types.json": build_types,
    "data/checks.json": build_checks,
    "data/methods.json": build_methods,
    "data/hexgrid.json": build_hexgrid,
}


def run(cfg: Config, dev_blind: str | Path | None = None, demo: str | Path | None = None) -> dict:
    """Собрать данные лендинга. Коды выхода — ``site.exit_codes``; возвращает story.json (для тестов)."""
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
    payloads: dict[str, bytes] = {"data/story.json": dumps(story)}
    pending = []
    for name, fn in BUILDERS.items():
        b = fn(cfg, facts)
        if b is None:
            pending.append(name)
        else:
            payloads[name] = b
    over = check_budget(payloads, cfg["site"]["budget_mb"])
    if over:
        raise QCError("site: бюджет: " + "; ".join(over))
    if mode.name == "normal" and pending:
        raise NotImplementedError(f"site: ещё не собраны {pending} (план §10, п. 5); site/ не тронут")
    mode.out.mkdir(parents=True, exist_ok=True)
    for name, b in payloads.items():
        p = mode.out / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b)
    log.info(
        "site (%s): story.json записан в %s; заглушки: %s%s",
        mode.name,
        mode.out,
        ", ".join(pending) or "нет",
        f"; плашка «{mode.banner}»" if mode.banner else "",
    )
    return story
