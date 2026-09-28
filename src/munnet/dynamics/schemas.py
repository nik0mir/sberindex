"""Схема выхода этапа dynamics для лендинга: история типа каждого узла по скользящим окнам."""

from __future__ import annotations

from munnet.contracts import DATE_PATTERN, POSITIVE_INT, Check, Col, TableSchema

# Статус узла за 2023 → 2024 (один на все окна): надёжный переход (stable_halves), смена типа в пределах
# шума (типы окон 2023 и 2024 годов различаются, но половины года не согласны) или без смены.
STATUSES: tuple[str, ...] = ("reliable", "within_noise", "no_change")


def _reliable_consistent(df):
    return df["reliable"] == (df["status"].astype(str) == "reliable")


DYNAMICS_HISTORY = TableSchema(
    name="dynamics_history",
    columns=(
        Col("territory_id", "int32", range=POSITIVE_INT),
        Col("window", "string"),
        Col("window_index", "int8", range=(1, 24)),
        Col("first_date", "string", pattern=DATE_PATTERN),
        Col("last_date", "string", pattern=DATE_PATTERN),
        Col("type", "int16", range=(1, 999)),
        Col("type_24m", "int16", range=(1, 999)),
        Col("status", "category", values=STATUSES),
        Col("reliable", "bool"),
        Col("is_city_node", "bool"),
    ),
    key=("territory_id", "window"),
    checks=(Check("флаг reliable совпадает со статусом", _reliable_consistent),),
)
