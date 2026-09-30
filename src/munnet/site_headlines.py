"""Проверка кодом предрегистрированных заголовков лендинга (``site.headlines.check``) и заморозки ``site``.

Перенос прототипа редакции 4 (30.09.2026, после devils-advocate, круг 6) без изменения логики: те же правила
в том же порядке — поля ``{…}`` и их порядок, цифры, объект сравнения, глоссарий, основы слов, короткие слова,
полярность, кванторы, обязательные оговорки ``must_keep``, запреты ``forbidden_words``; затем отрицательный
контроль (плохие заголовки должны быть пойманы все), перебор всех сочетаний вердиктов и sha256 замороженных
ключей. Правила — ``docs/landing_spec.md``, §4.3, пороги и словари — ``configs/default.yaml``, блок ``site``.
Проверка словарная: смысл целиком она не видит, поэтому словарь ещё читают check-ux и devils-advocate.

Запуск отдельно (вывод — как у прототипа): ``python -m munnet.site_headlines configs/default.yaml --hash``.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import re
import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

SLOT = re.compile(r"\{[a-z_0-9]+\}")
WORD = re.compile(r"[а-яa-z]+")
NUM = re.compile(r"\d+(?:[.,]\d+)?%?|%")
TOKQ = re.compile(r"\{[a-z_0-9]+\}|[а-яa-z]+")
# ключи site.headlines, которые не являются заголовками исходов проверок
SKIP = ("check", "screen0", "chapters", "descriptive")
# тесты, по уровням которых перебираются сочетания вердиктов (порядок — как в прототипе)
COMBO_TESTS = (
    "T1_ladder_external",
    "T2_direction",
    "T3_reliable_placebo",
    "T5_trivial",
    "T6_bank_coverage",
    "T7_utility",
)
TYPE_GAIN_LEVELS = ["adds", "neutral", "hurts"]
# подстановка {set_name} при переборе сочетаний (как в прототипе; значения — labels.SET_NAMES A и D)
SET_NAME_ADDS = "сопоставимые территории того же типа"
SET_NAME_OTHER = "сопоставимые территории"


def norm(s: str) -> str:
    """Без регистра, ё = е."""
    return s.lower().replace("ё", "е")


def tokens(s: str) -> list[str]:
    return WORD.findall(norm(SLOT.sub(" ", s)))


def freeze_blob(site: Mapping[str, Any]) -> str:
    """Канонический JSON ключей ``site.freeze.covers`` (так же, как при записи хеша)."""
    return json.dumps(
        {k: site[k] for k in site["freeze"]["covers"]},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def freeze_hash(site: Mapping[str, Any]) -> str:
    """sha256 замороженных ключей блока ``site``; сверяется с ``site.freeze.sha256``."""
    return hashlib.sha256(freeze_blob(site).encode("utf-8")).hexdigest()


Violation = tuple[str, Any]


@dataclass
class HeadlineChecker:
    """Линт заголовков по словарям конфига. ``cfg`` — весь конфиг (нужны блоки ``site`` и ``interpret``)."""

    cfg: Mapping[str, Any]
    site: Mapping[str, Any] = field(init=False)
    it: Mapping[str, Any] = field(init=False)
    H: Mapping[str, Any] = field(init=False)
    chk: Mapping[str, Any] = field(init=False)
    fw: Mapping[str, Any] = field(init=False)
    _hits_cache: dict[str, list] = field(init=False, repr=False, default_factory=dict)

    def __post_init__(self) -> None:
        self.site = self.cfg["site"]
        self.it = self.cfg["interpret"]["tests"]
        self.H = self.site["headlines"]
        self.chk = self.H["check"]
        self.fw = self.site["forbidden_words"]

    # --- текст исхода, с которым сверяется заголовок ----------------------------------------------------

    def main_text(self, test: str, key: str) -> tuple[str, str]:
        """(основной текст исхода, дополнительные части) из ``interpret.tests``."""
        it = self.it
        if test == "T6_caveat":
            return it["T6_bank_coverage"]["outcomes"]["caveat"]["text"], ""
        if test == "T7_type_gain":
            return it["T7_utility"]["outcomes"]["type_" + key]["text"], ""
        if test == "T4_basket_vs_place":
            return it[test]["text"], ""
        o = it[test]["outcomes"][key]
        return o["text"], " ".join(o.get(k, "") for k in ("other_overall", "other_none"))

    def stem(self, w: str) -> str:
        st = self.chk["stems"]
        return w[: max(st["min_stem"], len(w) - st["cut"])]

    def seq(self, s: str, skip: set[str]) -> list[str]:
        """Последовательность значимых слов и полей {…} (без регистра, ё = е), без skip."""
        return [w for w in TOKQ.findall(norm(s)) if w not in skip]

    def same(self, text_tok: str, head_tok: str) -> bool:
        """Поле — то же поле; слово — слово текста начинается с основы слова заголовка."""
        if head_tok.startswith("{"):
            return text_tok == head_tok
        return not text_tok.startswith("{") and text_tok.startswith(self.stem(head_tok))

    def polar(self, s: str) -> list[tuple[str, bool]]:
        """[(слово, отрицается ли)] — отрицается, если перед ним стоит negator."""
        ts, neg = tokens(s), set(self.chk["polarity"]["negators"])
        return [(w, i > 0 and ts[i - 1] in neg) for i, w in enumerate(ts)]

    # --- линт одного заголовка --------------------------------------------------------------------------

    def lint(self, test: str, key: str, h: str) -> list[Violation]:
        """Нарушения заголовка ``h`` исхода ``test.key``; пустой список — заголовок проходит."""
        chk, fw = self.chk, self.fw
        bad: list[Violation] = []
        t, extra = self.main_text(test, key)
        full = t + " " + extra
        hs, ts = set(SLOT.findall(h)), set(SLOT.findall(full))
        if not hs <= ts:
            bad.append(("slots", hs - ts))
        # порядок полей заголовка — подпоследовательность полей текста ({n_up}/{n_down} наоборот — нарушение)
        if chk.get("slot_order") == "same_as_text":
            it_ = iter(SLOT.findall(full))
            if not all(x in it_ for x in SLOT.findall(h)):
                bad.append(("slot_order", SLOT.findall(h)))
        hn, tn = norm(SLOT.sub(" ", h)), norm(SLOT.sub(" ", full))
        # цифры и «%» вне полей — только та же последовательность, что в тексте исхода («на 50% точнее» — нет)
        if chk.get("digits") == "slots_or_text":
            for d in NUM.findall(SLOT.sub(" ", h)):
                if d not in SLOT.sub(" ", full):
                    bad.append(("digits", d))
        # объект сравнения: первое значимое слово или поле после сравнительной степени — после неё и в тексте
        co = chk.get("comparative_object")
        if co:
            skip = set(co["skip"]) | set(chk["short_words"]["stop"]) | set(chk["polarity"]["negators"])
            hq, tq = self.seq(h, skip), self.seq(full, skip)
            for i, w in enumerate(hq):
                if w not in co["words"] or i + 1 >= len(hq):
                    continue
                nxt = hq[i + 1]
                pos = [j for j, x in enumerate(tq) if x == w]
                hit = [j for j, x in enumerate(tq) if self.same(x, nxt)]
                if pos and hit and not any(k > j for j in pos for k in hit):
                    bad.append(("comparative_object", f"{w} {nxt}"))
        tmain = norm(SLOT.sub(" ", t))
        # глоссарий: фраза для читателя засчитывается, если в тексте есть её исходная основа; затем удаляется
        for plain, src in chk["glossary"].items():
            if norm(plain) in hn:
                if not any(norm(x) in tn for x in src):
                    bad.append(("glossary", plain))
                hn = hn.replace(norm(plain), " ")
        for w in WORD.findall(hn):
            if len(w) < chk["stems"]["min_len"]:
                continue
            if self.stem(w) not in tn and not any(norm(a) in w for a in chk["allow"]):
                bad.append(("stem", w))
        # короткие слова (короче stems.min_len) вне закрытого стоп-списка: до 3 букв — то же слово в тексте,
        # 4 буквы — слово текста с теми же первыми тремя буквами; отрицатели проверяет polarity
        sw = chk["short_words"]
        ttok = tokens(tn)
        for w in WORD.findall(hn):
            if len(w) >= chk["stems"]["min_len"] or w in sw["stop"] or w in chk["polarity"]["negators"]:
                continue
            if not any(
                x == w or (len(w) == sw["prefix_from"] and x.startswith(w[: len(w) - 1])) for x in ttok
            ):
                bad.append(("short", w))
        # полярность
        pol = chk["polarity"]
        hp = list(self.polar(hn))
        tp = self.polar(full)
        for w, n in hp:
            if w in pol["negators"] or (
                len(w) < chk["stems"]["min_len"] and not (n and pol["any_length_if_negated"])
            ):
                continue
            s = self.stem(w)
            occ = [tn_ for tw, tn_ in tp if s in tw]
            if n and True not in occ:
                bad.append(("polarity: отрицание не из текста", w))
            if not n and occ and False not in occ:
                bad.append(("polarity: в тексте только с отрицанием", w))
        for sa in pol["standalone"]:
            if re.search(rf"(^|[^а-я]){sa}([^а-я]|$)", hn) and not re.search(
                rf"(^|[^а-я]){sa}([^а-я]|$)", tn
            ):
                bad.append(("polarity: standalone", sa))
        vn_text = [v for v in pol["verdict_negations"] if norm(v) in tmain]
        vn_head = [v for v in pol["verdict_negations"] if norm(v) in norm(h)]
        if vn_text and not vn_head:
            bad.append(("polarity: отрицание исхода выпало", vn_text))
        for v in vn_head:
            if norm(v) not in tn:
                bad.append(("polarity: отрицание не из текста", v))
        # кванторы: пара «квантор + следующее слово» — рядом и в тексте
        ht, tt = tokens(hn), tokens(tn)
        tpairs = {(a, self.stem(b)) for a, b in zip(tt, tt[1:], strict=False)}
        for i, w in enumerate(ht):
            for q in chk["quantifiers"]:
                if w.startswith(q):
                    nxt = ht[i + 1] if i + 1 < len(ht) else ""
                    if not any(a.startswith(q) and (self.stem(nxt) == b or b in nxt) for a, b in tpairs):
                        bad.append(("quantifier", f"{w} {nxt}"))
        # must_keep
        for item in chk["must_keep"].get(test, {}).get(key, []):
            if not any(norm(alt) in norm(h) for alt in item.split("|")):
                bad.append(("must_keep", item))
        for f in fw["unless_in_text"]:
            if norm(f) in hn and norm(f) not in tn:
                bad.append(("unless_in_text", f))
        for f in fw["headlines_always"] + self.cfg["interpret"]["naming"]["banned"] + fw["page_always"]:
            if norm(f) in norm(h):
                bad.append(("always", f))
        return bad

    # --- все заголовки исходов, отрицательный контроль ------------------------------------------------

    def headline_items(self) -> Iterable[tuple[str, str, str]]:
        """(тест, исход, заголовок) всех заголовков исходов словаря."""
        for test, d in self.H.items():
            if test in SKIP:
                continue
            for key, h in d.items():
                yield test, key, h

    def missing_levels(self) -> list[tuple[str, set[str]]]:
        """Уровни ``interpret.tests.<тест>.rule.levels`` без заголовка (в прототипе — assert)."""
        out = []
        for test, d in self.H.items():
            if test in SKIP:
                continue
            levels = self.it.get(test, {}).get("rule", {}).get("levels")
            if levels and not set(levels) <= set(d):
                out.append((test, set(levels) - set(d)))
        return out

    def negative_control(self) -> list[tuple[Mapping[str, str], list[Violation]]]:
        """Плохие заголовки ``check.negative_control`` и нарушения, которыми они пойманы."""
        return [(nc, self.lint(nc["test"], nc["key"], nc["h"])) for nc in self.chk["negative_control"]]

    # --- сочетания вердиктов ---------------------------------------------------------------------------

    def combo_levels(self) -> dict[str, list]:
        """Уровни перебора: шесть тестов, type_gain, partial_capped у T2, флаги one_in_ten и оговорки T6."""
        L: dict[str, list] = {t: list(self.it[t]["rule"]["levels"]) for t in COMBO_TESTS}
        L["T7_type_gain"] = list(TYPE_GAIN_LEVELS)
        L["T2_direction"] = L["T2_direction"] + ["partial_capped"]
        L["one_in_ten"] = [True, False]
        L["caveat"] = [True, False]
        return L

    def shown_texts(self, v: Mapping[str, Any]) -> list[str]:
        """Тексты, которые сайт пишет сам при сочетании вердиктов ``v`` (заголовки, роли, подписи)."""
        H, site = self.H, self.site
        R, A = site["roles"], site["always"]
        shown = [
            H[t][v[t]]
            for t in ("T1_ladder_external", "T3_reliable_placebo", "T5_trivial", "T7_utility", "T7_type_gain")
        ]
        dyn = H["chapters"]["dynamics"]
        if all(v[t] in lv for t, lv in dyn["lead_if"].items()):
            shown.append(H["T2_direction"][v["T2_direction"]])
        else:
            shown.append(H["descriptive"]["dynamics_neutral"])
        lim = H["chapters"]["limits"]
        if all(v[t] in lv for t, lv in lim["title_if"].items()):
            shown.append(H["T6_bank_coverage"][v["T6_bank_coverage"]])
        else:
            shown.append(H["descriptive"]["limits_neutral"])
        if v["caveat"]:
            shown.append(H["T6_caveat"]["caveat"])
        shown += [H["T4_basket_vs_place"]["describe"], *H["descriptive"].values()]
        shown += [R["ministry"][v["T7_utility"]], R["business"]["text"], R["business"]["label"]]
        an = R["analysts"]
        if all(v[t] in lv for t, lv in an["show_if"].items()) and an[v["T3_reliable_placebo"]]:
            shown.append(an[v["T3_reliable_placebo"]])
        shown += [
            A["scope_reader"],
            *A["honesty"].values(),
            site["palette_rule"]["reliability_grammar"]["key"],
            site["similarity_layout"]["caption"],
            site["similar_caption"],
        ]
        return shown

    def text_violations(
        self, texts: Iterable[str | tuple[str, str]], v: Mapping[str, Any]
    ) -> list[tuple[str, str, str]]:
        """Запреты ``page_always`` и ``by_verdict`` (все условия строки одновременно) в текстах сайта.

        Элемент ``texts`` — строка или пара (как показать в отчёте, что проверять) — например, заголовок
        до и после подстановки полей."""
        out = []
        for item in texts:
            shown, checked = item if isinstance(item, tuple) else (item, item)
            for kind, f, cond in self._hits(checked):
                if kind == "combo":
                    ok = all(
                        (v["one_in_ten"] is lv) if t == "one_in_ten" else v.get(t) in lv
                        for t, lv in cond.items()
                    )
                    if f == "того же типа" and "ограничение «того же типа»" in norm(checked):
                        ok = True
                    if ok:
                        continue
                out.append((kind, f, shown))
        return out

    def _hits(self, checked: str) -> list[tuple[str, str, Any]]:
        """Запретные подстроки текста (без вердиктов; кэш: перебор сочетаний повторяет те же тексты)."""
        cache = self._hits_cache
        if checked not in cache:
            fw, hn = self.fw, norm(checked)
            cache[checked] = [("page_always", f, None) for f in fw["page_always"] if norm(f) in hn] + [
                ("combo", f, cond) for f, cond in fw["by_verdict"].items() if norm(f) in hn
            ]
        return cache[checked]

    def combos(self) -> tuple[int, list[tuple[str, str, str]]]:
        """Перебор всех сочетаний: (число сочетаний, уникальные нарушения в порядке появления)."""
        L = self.combo_levels()
        dw = self.it["T2_direction"]["direction_words"]
        n = 0
        seen: list[tuple[str, str, str]] = []
        for combo in itertools.product(*L.values()):
            v = dict(zip(L, combo, strict=True))
            n += 1
            direction = dw["t1_confirmed"] if v["T1_ladder_external"] == "confirmed" else dw["otherwise"]
            set_name = SET_NAME_ADDS if v["T7_type_gain"] == "adds" else SET_NAME_OTHER
            # в отчёт идёт исходный заголовок (до подстановки), как в прототипе
            pairs = [
                (h, h.replace("{direction}", direction).replace("{set_name}", set_name))
                for h in self.shown_texts(v)
            ]
            for b in self.text_violations(pairs, v):
                if b not in seen:
                    seen.append(b)
        return n, seen


@dataclass
class HeadlineReport:
    """Итог полной проверки словаря: то, что этап site и тест требуют (иначе код 3)."""

    rows: list[tuple[str, str, str, list[Violation]]]
    missing_levels: list[tuple[str, set[str]]]
    control: list[tuple[Mapping[str, str], list[Violation]]]
    n_combos: int
    combo_violations: list[tuple[str, str, str]]
    sha256: str
    sha256_expected: str

    @property
    def headline_violations(self) -> list[tuple[str, str, Violation]]:
        return [(t, k, b) for t, k, _, bs in self.rows for b in bs]

    @property
    def caught(self) -> int:
        return sum(bool(b) for _, b in self.control)

    @property
    def ok(self) -> bool:
        return not self.problems()

    def problems(self) -> list[str]:
        """Словесные причины кода 3; пустой список — словарь проходит."""
        out = [f"нет заголовка для уровней {t}: {sorted(m)}" for t, m in self.missing_levels]
        out += [f"заголовок {t}.{k}: {b}" for t, k, b in self.headline_violations]
        missed = [f"{nc['test']}.{nc['key']}: {nc['h']}" for nc, b in self.control if not b]
        if missed:
            out.append(f"отрицательный контроль пропустил {len(missed)} из {len(self.control)}: {missed}")
        out += [f"сочетания вердиктов: {b[0]} «{b[1]}» в «{b[2]}»" for b in self.combo_violations]
        if self.sha256 != self.sha256_expected:
            out.append(
                f"sha256 замороженных ключей site {self.sha256} ≠ freeze.sha256 {self.sha256_expected}"
            )
        return out

    def lines(self, with_hash: bool = True) -> list[str]:
        """Строки отчёта в том же виде, что печатал прототип (для сверки вывода)."""
        out = [
            f"{t:20} {k:16} {len(h.split()):2} сл. {'OK ' if not b else 'ПЛОХО'} {h}"
            for t, k, h, b in self.rows
        ]
        for nc, b in self.control:
            out.append(
                f"  контроль {'пойман' if b else 'ПРОПУЩЕН'}: {nc['test']}.{nc['key']}: {nc['h']} -> {b[:3]}"
            )
        out.append(f"отрицательный контроль: поймано {self.caught} из {len(self.control)}")
        out.append(f"сочетаний: {self.n_combos}")
        # в прототипе нарушения заголовков и сочетаний шли в один список; уникальные — по первым трём полям
        seen: list[tuple] = []
        for t, k, b in self.headline_violations:
            key = (t, k, b[0])
            if key not in seen:
                seen.append(key)
        for b in self.combo_violations:
            if b not in seen:
                seen.append(b)
        out += [f"НАРУШЕНИЕ: {b}" for b in seen]
        out.append(f"нарушений (уникальных): {len(seen)}")
        if with_hash:
            out.append(f"sha256: {self.sha256} в конфиге: {self.sha256_expected}")
        return out


def check_headlines(cfg: Mapping[str, Any]) -> HeadlineReport:
    """Полная проверка словаря ``site.headlines`` и заморозки на конфиге ``cfg`` (словарь YAML)."""
    hc = HeadlineChecker(cfg)
    rows = [(t, k, h, hc.lint(t, k, h)) for t, k, h in hc.headline_items()]
    n, combo_bad = hc.combos()
    return HeadlineReport(
        rows=rows,
        missing_levels=hc.missing_levels(),
        control=hc.negative_control(),
        n_combos=n,
        combo_violations=combo_bad,
        sha256=freeze_hash(cfg["site"]),
        sha256_expected=str(cfg["site"]["freeze"]["sha256"]),
    )


def main(argv: list[str] | None = None) -> int:
    """``python -m munnet.site_headlines <конфиг> [--hash]``: отчёт как у прототипа; 3 — есть нарушения."""
    argv = sys.argv[1:] if argv is None else argv
    path = Path(argv[0]) if argv else Path("configs/default.yaml")
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    rep = check_headlines(cfg)
    sys.stdout.write("\n".join(rep.lines(with_hash="--hash" in argv)) + "\n")
    return 0 if rep.ok else 3


if __name__ == "__main__":
    raise SystemExit(main())
