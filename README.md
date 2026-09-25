# Типы локальных экономик России

Проект для [онлайн-конкурса СберИндекса 2026](https://sber.ru/sberindex/konkurs_sberindex), направление «Кластеризация».
Строим динамическую атрибутированную сеть муниципальных образований (МО): узлы — МО с экономическими
признаками, рёбра — экономическая близость, сеть пересчитывается по месяцам. Кластеры сети трактуем как типы
локальных экономик и прослеживаем, как они меняются во времени.

Статус: каркас репозитория. Реализован этап загрузки данных, остальные этапы идут по [плану](PLAN.md).

## Быстрый старт

Нужны Python 3.12, [uv](https://docs.astral.sh/uv/) и системная библиотека libarchive
(для RAR-архива со справочником границ): в Ubuntu/Debian — `apt install libarchive13`, в macOS —
`brew install libarchive` и `export LIBARCHIVE=$(brew --prefix libarchive)/lib/libarchive.dylib`.
В Windows проект запускается через WSL.

```bash
make setup   # окружение строго по uv.lock
make all     # все этапы с параметрами из configs/default.yaml
```

Без uv и make:

```bash
python3.12 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
pip install --no-deps -e .
python -m munnet all
```

Команды запускаются из корня репозитория: относительные пути в конфиге считаются от него.

## Этапы

| Этап | Команда | Что делает | Статус |
|---|---|---|---|
| data | `make data` | Скачивает архивы с первого доступного зеркала, сверяет sha256, распаковывает в `data/raw/` | готов |
| features | `make features` | Панель «МО × месяц × категория», признаки узлов | план, этапы 1–2 |
| network | `make network` | Рёбра по нескольким правилам, разрежение | план, этап 2 |
| cluster | `make cluster` | Кластеризация по признакам, по сети и на атрибутированной сети | план, этап 3 |
| evaluate | `make evaluate` | ICVI: SW, CH, S_Dbw, AVI, AVU, MQ | план, этап 4 |
| dynamics | `make dynamics` | Кластеры по скользящим окнам, переходы МО между типами | план, этап 3 |
| site | `make site` | Данные для интерактивного лендинга | план, этап 6 |

Список этапов и параметры командной строки: `python -m munnet --help`.

## Конфигурация

Все гиперпараметры лежат в [`configs/default.yaml`](configs/default.yaml): источники данных с контрольными
суммами, правила рёбер, методы кластеризации, набор ICVI, окна для динамики, `seed`. Чтобы поменять параметры,
скопируйте файл и передайте его явно:

```bash
python -m munnet --config configs/my.yaml cluster evaluate
make CONFIG=configs/my.yaml all
```

## Структура

```
configs/           гиперпараметры (YAML)
src/munnet/        пакет: один модуль на этап, cli.py запускает их по порядку
tests/             тесты: make test
report/            методологический отчёт
site/              интерактивный лендинг
docs/materials.md  конспект условий конкурса и описание исходных данных
certs/             временно: сертификаты для сайтов Сбера, см. certs/README.md
data/, outputs/    создаются при запуске, в git не хранятся
```

## Данные и лицензии

Исходные данные скачиваются этапом `data`, подробности — в [docs/materials.md](docs/materials.md).
Все наборы опубликованы Лабораторией СберИндекс по лицензии
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/deed.ru), производные данные проекта
распространяются на тех же условиях.

- Потребительские безналичные расходы на уровне муниципальных образований по категориям трат. СберИндекс.
- Индекс доступности рынков на уровне муниципальных образований. СберИндекс.
- Автодорожные и железнодорожные связи между муниципальными образованиями. СберИндекс.

  Данные доступны по адресу
  https://sberindex.ru/ru/research/data-sense-opisanie-nabora-dannikh-khakatona-sberindeksa-po-munitsipalnim-dannim
  (данные скачаны 25.09.2026).
- Данные о границах и преобразованиях муниципальных образований. СберИндекс. Данные доступны по адресу
  https://sberindex.ru/ru/research/dataset-borders-and-changes-of-municipalities (данные скачаны 25.09.2026).

## Разработка

```bash
make test   # pytest
make lint   # ruff
make requirements   # обновить requirements.txt после изменения зависимостей
```

## Лицензия

Код проекта распространяется по лицензии [MIT](LICENSE). Исходные и производные данные — по
CC BY-SA 4.0 (см. раздел «Данные и лицензии»). Навыки в `.claude/skills/` — под собственными лицензиями,
перечень в [.claude/skills/THIRD_PARTY.md](.claude/skills/THIRD_PARTY.md).
