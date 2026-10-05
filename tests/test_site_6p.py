"""Порция 6p лендинга (``docs/landing_spec.md``, §4.3, журнал 05.10): строка «Простыми словами»
под замороженным исходом T5 ``not_repeats`` в пункте «Что проверяли». Строку собирает код из ``facts.t5``;
сам исход, его ярлыки и хеш заморозки блока ``site`` не меняются. Синтетика
с известным ответом, без data/ и outputs/."""

from __future__ import annotations

import copy

from munnet import landing
from munnet.config import load_config
from munnet.site_headlines import freeze_hash

CFG = load_config("configs/default.yaml")
TX = CFG["site"]["build"]["texts"]
# как в outputs/interpret/facts.json (t5): соперники по обороту — из проверки T5 (ε² внутри страт)
T5 = {
    "verdict": "not_repeats",
    "max_partition": "sized:log_level_rel:+",
    "max_label": "уровень трат, по возрастанию",
    "max_ami": 0.313,
    "per": {
        "catering": {"best_rival": "sized:log_wage_rel:+"},
        "retail": {"best_rival": "sized:emp_sh_trade_transport:-"},
    },
}
WANT = (
    "проверяли, говорят ли типы о муниципалитете больше, чем простое деление по одному признаку. "
    "Ближе всего типы к делению по уровню трат. Среди муниципалитетов похожего размера и с похожей долей "
    "горожан оборот общепита и розницы Росстата различается между типами не сильнее, чем при простых "
    "делениях, например по зарплате"
)


def test_plain_from_facts():
    s = landing.t5_plain(TX, "not_repeats", T5)
    assert s == WANT
    # условие страт и «не сильнее» — как в замороженном исходе; соперник T5 для розницы не назван
    assert "похожего размера и с похожей долей горожан" in s and "не сильнее" in s
    assert "занятых" not in s


def test_plain_follows_facts():
    """Деление и пример — из facts.t5 (max_partition, соперник общепита); соперник розницы не влияет."""
    t5 = copy.deepcopy(T5)
    t5["max_partition"] = "sized:log_wage_rel:+"
    t5["per"]["catering"]["best_rival"] = "sized:emp_sh_trade_transport:-"
    t5["per"]["retail"]["best_rival"] = "sized:log_level_rel:+"
    s = landing.t5_plain(TX, "not_repeats", t5)
    assert "Ближе всего типы к делению по зарплате." in s
    assert s.endswith("например по доле занятых в строительстве, торговле и транспорте")
    t5["per"]["catering"]["best_rival"] = "kmeans:place"  # нет слова для соперника — без примера
    assert landing.t5_plain(TX, "not_repeats", t5).endswith("не сильнее, чем при простых делениях")
    t5 = copy.deepcopy(T5)
    del t5["per"]["retail"]
    assert "оборот общепита Росстата" in landing.t5_plain(TX, "not_repeats", t5)


def test_plain_hidden_on_other_verdict_or_unknown_key():
    for v in ("confirmed", "partial", "not_empty"):
        assert landing.t5_plain(TX, v, T5) is None
    t5 = copy.deepcopy(T5)
    t5["max_partition"] = "sized:log_pop_rel:+"
    assert landing.t5_plain(TX, "not_repeats", t5) is None
    t5 = copy.deepcopy(T5)
    t5["per"]["other"] = {"best_rival": "sized:log_wage_rel:+"}  # оборота нет в словаре
    assert landing.t5_plain(TX, "not_repeats", t5) is None
    assert landing.t5_plain(TX, "not_repeats", {}) is None
    assert landing.t5_plain({}, "not_repeats", T5) is None


def test_plain_line_visible_under_point_head_not_in_details():
    """Строка — под видимым заголовком пункта, до «Как проверяли»; в раскрывающемся блоке её нет."""
    heads = ["Типы упорядочены…", "Типы по составу близки к делению «X»"]
    plain = landing.point_plain_html([None, WANT], TX["t5_plain_label"])
    body = landing.point_texts_html(["Текст T1.", "Дословный текст T5."], [], [])
    h = landing.point_li_html(heads, plain, "Как проверяли", body, extra=False)
    i_h, i_p, i_d = h.index("близки к делению «X»"), h.index('<p class="plain">'), h.index("<details")
    assert i_h < i_p < i_d
    assert h.count('class="plain"') == 1 and "<b>Простыми словами:</b> проверяли, говорят ли типы" in h
    assert h.index("Дословный текст T5") > i_d
    assert landing.point_plain_html([None, WANT], None) == "" and landing.point_plain_html([None], "М") == ""


def test_frozen_t5_text_and_hash_unchanged():
    it = CFG["interpret"]["tests"]["T5_trivial"]["outcomes"]["not_repeats"]["text"]
    assert it == (
        "Типы по составу близки к делению «{max_partition}» (AMI {max_ami}) и внутри страт размера "
        "и доли горожан по обороту Росстата различаются не сильнее, чем деление «{best_partition}» "
        "(ε² {eps2} против {eps2_best})."
    )
    assert CFG["site"]["headlines"]["T5_trivial"]["not_repeats"] == (
        "Типы по составу близки к делению «{max_partition}» и по обороту Росстата различаются не сильнее, "
        "чем «{best_partition}»"
    )
    assert freeze_hash(CFG["site"]) == CFG["site"]["freeze"]["sha256"]
    assert "t5_plain" not in CFG["site"]["freeze"]["covers"]
