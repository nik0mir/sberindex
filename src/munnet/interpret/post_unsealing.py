"""Пояснения после вскрытия результатов этапа 5 и «Журнал после вскрытия».

Правило участника: после вскрытия предрегистрированные правила, вердикты и тексты исходов не меняются.
Допускаются (1) исправление доказанной ошибки кода с записью «после вскрытия» и показом обоих результатов,
(2) пояснения рядом с текстом исхода с пометкой «пояснение после вскрытия». Всё — в журнале ``JOURNAL``.

Модуль только считает описательные числа для пояснений и собирает их фразы: в ``verdicts_*``, ``texts``,
``thesis``, ``edits`` и проверки T1–T7 ничего не пишет. Сверка — ``outcomes_sha256``: хеш ``verdicts_final``
и ``thesis`` в этом прогоне против записанного до пояснений (``OUTCOMES_SHA256_BEFORE``). Модуль не входит
в ``runs.HEAVY_MODULES`` и тяжёлые прогоны не импортирует: правка пояснений кэш не сбрасывает.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_mutual_info_score

from munnet import style

NOTE = "Пояснение после вскрытия"
POSTHOC_NOTE = "разведка после вскрытия, в выводы не идёт"
# sha256 канонического JSON {"verdicts_final", "thesis"} прогона 30.09.2026 до пояснений
# (outputs/interpret/facts.json, копия сделана до правок; команда — outcomes_sha256 ниже)
OUTCOMES_SHA256_BEFORE = "b53344518402ba86915959e9566277a4cc34eaee354ff7a269d289214328a83c"
# подписи прогонов R1 (r1_runs.csv: run) для пояснений
RUN_LABELS: dict[str, str] = {
    "main": "основной расчёт",
    "variant:graph_basket_cos": "правило рёбер „косинус корзин“",
    "variant:no_level": "без уровня трат в признаках",
    "variant:nodes_separate": "внутригородские МО отдельными узлами",
    "tracking:fixed_prototypes": "прослеживание по фиксированным типам",
    "tracking:evolutionary": "эволюционное прослеживание",
}
LEVEL_PARTITION = "sized:log_level_rel:+"  # деление «уровень трат, по возрастанию» (T5, наибольший AMI)
F2 = lambda v: style.fmt_num(v, 2)  # noqa: E731
F3 = lambda v: style.fmt_num(v, 3)  # noqa: E731


@dataclass(frozen=True)
class Entry:
    date: str
    what: str
    kind: str  # «пояснение» | «исправление ошибки кода» | «разведка»
    before: str
    after: str
    why: str


JOURNAL: tuple[Entry, ...] = (
    Entry(
        "30.09.2026",
        "главный вывод, пункт 2: какой прогон дал вердикт T3 и откуда числа T2",
        "пояснение",
        "текст пункта 2 без указания прогона",
        "тот же текст и пояснение рядом (прогон R1 с вердиктом T3, число прогонов R1 выше 95-го перцентиля, "
        "«вверх» и «вниз» T2 основного расчёта)",
        "в пункте 2 рядом стоят «надёжных переходов 148» (T3, вариант правила рёбер) и «вверх 148» (T2, "
        "основной расчёт) — совпадение чисел читалось как одно и то же",
    ),
    Entry(
        "30.09.2026",
        "T2: почему общий сдвиг всех МО вверх невозможен",
        "пояснение",
        "текст исхода T2",
        "тот же текст и пояснение с медианой изменения доли общепита у оставшихся в нижнем типе",
        "корзина считается относительно среднего своего региона; без этого «чаще вверх» читалось как сдвиг "
        "всей совокупности",
    ),
    Entry(
        "30.09.2026",
        "T5: насколько типы повторяют деление по уровню трат",
        "пояснение",
        "текст исхода T5",
        "тот же текст и AMI с бутстрап-интервалом, AMI в прогонах R1, доля МО с той же ступенью",
        "исход «повторяют тривиальное деление» без меры близости не говорит, насколько",
    ),
    Entry(
        "30.09.2026",
        "T6: интервал дельты Клиффа в таблице и смысл «нет»",
        "пояснение",
        "таблица потоков без интервала",
        "строка «все потоки» с интервалом и пояснение: не обнаружено, исключено только отличие "
        "больше границы",
        "«не подтвердилась» читалось как «отличия нет»",
    ),
    Entry(
        "30.09.2026",
        "T7: сколько добавляет тип; η² региона для изменения розницы",
        "пояснение и разведка",
        "текст исхода T7",
        "тот же текст и верхняя граница вклада типа; η² группы региона против перестановок — в разведке",
        "«тип ничего не добавляет» без верхней границы не говорит, насколько мало",
    ),
    Entry(
        "30.09.2026",
        "узлы Москвы и Петербурга в типе 3",
        "пояснение",
        "таблица узлов-городов без пояснения",
        "таблица и одна фраза о корзине относительно своей группы регионов",
        "читатель ждёт столицы в верхнем типе",
    ),
)


# --- Сверка неизменности исходов ---------------------------------------------------------------------


def outcomes_sha256(facts: Mapping) -> str:
    """sha256 канонического JSON ``verdicts_final`` и ``thesis`` (ключи отсортированы, UTF-8)."""
    obj = {"verdicts_final": facts["verdicts_final"], "thesis": facts["thesis"]}
    js = json.dumps(json.loads(json.dumps(obj, ensure_ascii=False)), ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(js.encode("utf-8")).hexdigest()


# --- Расчёты -------------------------------------------------------------------------------------------


def t3_runs(placebo: pd.DataFrame, r1: pd.DataFrame, percentile: float) -> list[dict]:
    """T3 по прогонам R1 из ``t3_placebo.csv`` (pair = −1 — наблюдение) и ``r1_runs.csv``: наблюдение,
    медиана и перцентиль плацебо, доля псевдопар с числом переходов больше наблюдаемого, вердикт прогона."""
    verdicts = r1.loc[r1["test"] == "T3_reliable_placebo"].set_index("run")["verdict"]
    rows = []
    for (run, scheme), g in placebo.groupby(["run", "scheme"], sort=False):
        obs = int(g.loc[g["pair"] == -1, "n_reliable"].iloc[0])
        pl = g.loc[g["pair"] >= 0, "n_reliable"].to_numpy(dtype=np.float64)
        rows.append(
            {
                "run": str(run),
                "scheme": str(scheme),
                "n_reliable": obs,
                "median": float(np.median(pl)),
                "p95": float(np.percentile(pl, percentile)),
                "share_above": float(np.mean(pl > obs)),
                "verdict": str(verdicts.get(run, "")),
            }
        )
    return rows


def t2_relative(
    d_cafe: np.ndarray, t_a: np.ndarray, t_b: np.ndarray, reliable: np.ndarray, bottom: int, up_to: int
) -> dict:
    """Изменение доли общепита в корзине относительно среднего своего региона (CLR, окно ``year_b`` минус
    окно ``year_a``) у оставшихся в нижнем типе (определение T6: половины обоих лет в типе, не надёжный
    переход), у надёжно перешедших из него на ступень выше и среднее по всем узлам сети (0 по построению)."""
    d = np.asarray(d_cafe, dtype=np.float64)
    rel = np.asarray(reliable, dtype=bool)
    st = ~rel & (t_a == bottom) & (t_b == bottom) & np.isfinite(d)
    mv = rel & (t_a == bottom) & (t_b == up_to) & np.isfinite(d)
    return {
        "bottom": int(bottom),
        "up_to": int(up_to),
        "n_stayers": int(st.sum()),
        "median_stayers": float(np.median(d[st])) if st.any() else float("nan"),
        "share_stayers_down": float(np.mean(d[st] < 0)) if st.any() else float("nan"),
        "n_movers": int(mv.sum()),
        "median_movers": float(np.median(d[mv])) if mv.any() else float("nan"),
        "mean_all": float(np.nanmean(d)),
        "n_all": int(np.isfinite(d).sum()),
    }


def t5_level(
    types: pd.DataFrame, rival: pd.DataFrame, node_r1: pd.DataFrame, n_boot: int, level: float, seed: int
) -> dict:
    """Близость типов к делению «уровень трат, по возрастанию»: AMI с перцентильным бутстрап-интервалом
    (узлы с возвращением), AMI меток прогонов R1 (``node_r1.csv``) с тем же делением основного расчёта,
    доля узлов, у которых ступень типа совпадает с группой деления (группа g — ступень g по размеру)."""
    lv = rival.loc[rival["partition"] == LEVEL_PARTITION, ["territory_id", "best_partition_label"]]
    m = types[["territory_id", "type", "step"]].merge(lv, on="territory_id", how="inner", validate="1:1")
    a = m["type"].to_numpy()
    b = m["best_partition_label"].to_numpy()
    ami = float(adjusted_mutual_info_score(b, a))
    rng = np.random.default_rng([seed, 931])
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, len(a), len(a))
        boots[i] = adjusted_mutual_info_score(b[idx], a[idx])
    lo, hi = np.percentile(boots, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    runs = {}
    for run, g in node_r1.groupby("variant", sort=True):
        j = g[["territory_id", "matched_type"]].merge(lv, on="territory_id", how="inner", validate="1:1")
        runs[str(run)] = float(adjusted_mutual_info_score(j["best_partition_label"], j["matched_type"]))
    return {
        "partition": LEVEL_PARTITION,
        "n": int(len(m)),
        "ami": ami,
        "ci": [float(lo), float(hi)],
        "n_boot": int(n_boot),
        "ami_runs": runs,
        "same_step_share": float(np.mean(m["step"].to_numpy() == b)),
    }


def eta2(y: np.ndarray, groups: np.ndarray) -> float:
    """η² однофакторного разложения: межгрупповая сумма квадратов / общая."""
    y = np.asarray(y, dtype=np.float64)
    g = pd.Series(y).groupby(np.asarray(groups))
    ss_b = float((g.count() * (g.mean() - y.mean()) ** 2).sum())
    ss_t = float(((y - y.mean()) ** 2).sum())
    return ss_b / ss_t if ss_t > 0 else float("nan")


def region_eta2(y: np.ndarray, groups: np.ndarray, types: np.ndarray, n_perm: int, seed: int) -> dict:
    """η² группы региона для изменения розничного оборота против перестановок меток региона; рядом η² типа."""
    y = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(y)
    y, groups, types = y[ok], np.asarray(groups)[ok], np.asarray(types)[ok]
    obs = eta2(y, groups)
    rng = np.random.default_rng([seed, 932])
    null = np.array([eta2(y, rng.permutation(groups)) for _ in range(n_perm)])
    return {
        "n": int(len(y)),
        "n_groups": int(len(np.unique(groups))),
        "eta2_region": obs,
        "null_median": float(np.median(null)),
        "null_p95": float(np.percentile(null, 95)),
        "p": float((1 + np.sum(null >= obs - 1e-12)) / (1 + n_perm)),
        "eta2_type": eta2(y, types),
        "n_perm": int(n_perm),
    }


# --- Фразы ---------------------------------------------------------------------------------------------


def _run_label(run: str) -> str:
    if run.startswith("seed:"):
        return f"seed {run.split(':', 1)[1]}"
    return RUN_LABELS.get(run, run)


def point2_note(pu: Mapping, f: Mapping) -> str:
    """Пункт 2: какой прогон дал вердикт T3 в главном выводе и откуда «вверх» и «вниз» T2."""
    rows = [r for r in pu["t3_runs"] if r["scheme"] == "main"]
    by = {r["run"]: r for r in rows}
    src3 = f["r1"]["source_run"]["T3_reliable_placebo"]
    src2 = f["r1"]["source_run"]["T2_direction"]
    main = by.get("main")
    other = [r for r in rows if r["run"] not in ("main", src3)]
    n_ok = sum(r["verdict"] == "confirmed" for r in other)
    parts = []
    if src3 != "main" and src3 in by:
        s = by[src3]
        parts.append(
            f"вердикт T3 в главном выводе дал прогон R1 «{_run_label(src3)}»: {s['n_reliable']} надёжных "
            f"переходов против медианы плацебо ≈ {style.fmt_num(s['median'], 0)}, у "
            f"{style.fmt_pct(s['share_above'], 0)} псевдопар их больше — неотличимо от плацебо"
        )
        if main is not None:
            parts.append(
                f"в основном расчёте и в {n_ok} из {len(other)} остальных прогонов R1 число надёжных "
                f"переходов выше 95-го перцентиля плацебо (основной расчёт: {main['n_reliable']} против "
                f"{style.fmt_num(main['p95'], 0)})"
            )
    t2 = f["t2"]["main"]
    t2_run = "основного расчёта" if src2 == "main" else f"прогона «{_run_label(src2)}»"
    parts.append(
        f"числа T2 — из {t2_run}: надёжных переходов {t2['n_up'] + t2['n_down']}, из них вверх "
        f"{t2['n_up']}, вниз {t2['n_down']}"
        + (
            f"; совпадение «{t2['n_up']}» у T3 и T2 случайно — это разные прогоны и разные величины"
            if src3 != src2 and src3 in by and by[src3]["n_reliable"] == t2["n_up"]
            else ""
        )
    )
    return f"*{NOTE} к пункту 2:* " + "; ".join(parts) + "."


def t2_note(pu: Mapping) -> str:
    r = pu["t2_relative"]
    return (
        f"*{NOTE}:* корзина считается относительно среднего своего региона, поэтому общий сдвиг всех МО "
        "вверх невозможен по построению: среднее изменение доли общепита (CLR относительно региона, 2024 "
        f"минус 2023) по всем {r['n_all']} узлам — {F3(r['mean_all'])}. Переходы вверх идут рядом с тем, "
        f"что МО, оставшиеся в нижнем типе (тип {r['bottom']}), отошли ниже: медиана изменения у "
        f"{r['n_stayers']} оставшихся — {F3(r['median_stayers'])} "
        f"(снизилась у {style.fmt_pct(r['share_stayers_down'], 0)}), "
        f"у {r['n_movers']} перешедших из типа {r['bottom']} в тип {r['up_to']} — "
        f"{F3(r['median_movers'])}. Это расхождение МО внутри регионов, а не рост доли общепита у всех."
    )


def t5_note(pu: Mapping) -> str:
    r = pu["t5_level"]
    runs = ", ".join(f"{_run_label(k)} — {F2(v)}" for k, v in r["ami_runs"].items())
    return (
        f"*{NOTE}:* AMI типов с делением «уровень трат, по возрастанию» — {F2(r['ami'])} (95% бутстрап-"
        f"интервал {style.fmt_range(r['ci'][0], r['ci'][1], F2)}, {r['n_boot']} выборок узлов; 0 — "
        f"случайное совпадение, 1 — полное); в прогонах R1: {runs}. Ступень типа совпадает с группой "
        f"деления по уровню трат у {style.fmt_pct(r['same_step_share'], 0)} из {r['n']} узлов. Исход T5 "
        "по правилу записи не меняется; числа показывают меру близости."
    )


def t6_note(f: Mapping) -> str:
    t6 = f["t6"]
    lo, hi = t6["ci"]
    return (
        f"*{NOTE}:* «не подтвердилась» здесь значит «не обнаружено», а не «отличия нет»: 95% интервал "
        f"дельты Клиффа — {style.fmt_range(lo, hi, F2)}, поэтому исключено только отличие перешедших "
        f"от оставшихся сильнее его границ (дельта больше ≈ {F2(hi)} в пользу перешедших или меньше "
        f"≈ {F2(lo)}); более слабое отличие данные не исключают."
    )


def t7_note(f: Mapping) -> str:
    t7 = f["t7"]
    pt, (lo, hi) = t7["diffs"]["D-A"]
    med_d = t7["median_error"]["D"]
    return (
        f"*{NOTE}:* тип добавляет к точности подбора не больше {F3(hi)} медианной ошибки (верхняя граница "
        f"95% интервала D − A, точка {F3(pt)}) при медианной ошибке набора без учёта типа {F3(med_d)}, "
        f"то есть не больше {style.fmt_pct(hi / med_d if med_d > 0 else float('nan'), 0)} её."
    )


def city_note(f: Mapping) -> str:
    rows = f["scope"].get("city_nodes", [])
    if not rows:
        return ""
    types = sorted({int(r["type"]) for r in rows})
    where = f"в типе {types[0]}" if len(types) == 1 else "в типах " + ", ".join(map(str, types))
    names = ", ".join(r["name"] for r in rows)
    return (
        f"*{NOTE}:* узлы-города ({names}) — {where}, потому что их корзина считается относительно "
        "своей группы регионов (Москва — вместе с Московской областью, Петербург — с Ленинградской): тип "
        "говорит, чем город отличается от своей области, а не от крупных центров других регионов."
    )


def t6_total_row(f: Mapping) -> dict:
    t6 = f["t6"]
    return {
        "Поток": "все потоки (взвешенно, правило T6)",
        "Перешли / остались": f"{t6['n_movers']} / {t6['n_stayers']}",
        "Дельта Клиффа": F2(t6["delta"]),
        "95% интервал": style.fmt_range(t6["ci"][0], t6["ci"][1], F2),
    }


def region_table(pu: Mapping) -> pd.DataFrame:
    rows = []
    for key, label in (
        ("retail_rel", "относительно региона (цель T7)"),
        ("retail_abs", "без относительности"),
    ):
        r = pu["t7_region"].get(key)
        if not r:
            continue
        rows.append(
            {
                "Изменение розницы 2023 → 2024": label,
                "МО": r["n"],
                "η² группы региона": F3(r["eta2_region"]),
                "Перестановки: медиана; 95-й перцентиль": f"{F3(r['null_median'])}; {F3(r['null_p95'])}",
                "p": style.fmt_p(r["p"]),
                "η² типа": F3(r["eta2_type"]),
            }
        )
    return pd.DataFrame(rows)


def region_note(pu: Mapping) -> str:
    """Подпись к разведке η²: чем различаются две строки (арифметика определения цели T7)."""
    r = pu["t7_region"]
    if "retail_rel" not in r or "retail_abs" not in r:
        return ""
    return (
        "Цель T7 — разность относительных значений: из изменения ln(1 + оборот) каждого МО вычитается "
        "изменение медианы его группы региона, общее для всех МО группы, поэтому η² группы региона у неё "
        f"другое, чем без относительности ({F2(r['retail_rel']['eta2_region'])} против "
        f"{F2(r['retail_abs']['eta2_region'])}). Соседи по своему региону (набор B) делят этот общий "
        "сдвиг по построению."
    )


def journal_lines(pu: Mapping) -> list[str]:
    out = [
        f"- {e.date}, {e.what} ({e.kind}): было — {e.before}; стало — {e.after}. Почему: {e.why}."
        for e in JOURNAL
    ]
    same = pu.get("outcomes_sha256") == OUTCOMES_SHA256_BEFORE
    out.append(
        "- Исправлений ошибок кода после вскрытия нет. Правила, вердикты и тексты исходов не менялись: "
        f"sha256 `verdicts_final` и `thesis` до пояснений — `{OUTCOMES_SHA256_BEFORE[:16]}…`, "
        f"в этом прогоне — `{str(pu.get('outcomes_sha256', ''))[:16]}…`, "
        + ("совпадают." if same else "**не совпадают** (прогон на других данных или изменились исходы).")
    )
    return out
