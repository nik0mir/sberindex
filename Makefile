# Запуск пайплайна: make all, отдельный этап: make data, другой конфиг: make CONFIG=configs/my.yaml cluster
CONFIG ?= configs/default.yaml
RUN = uv run --frozen python -m munnet --config $(CONFIG)
STAGES = data features network cluster evaluate dynamics site

.PHONY: setup all $(STAGES) test lint requirements clean

# Окружение строго по uv.lock
setup:
	uv sync --frozen

all:
	$(RUN) all

$(STAGES):
	$(RUN) $@

test:
	uv run --frozen pytest

lint:
	uv run --frozen ruff check .
	uv run --frozen ruff format --check .

# requirements.txt для установки через pip, собирается из uv.lock
requirements:
	uv export --frozen --no-emit-project --no-dev --output-file requirements.txt

# Удаляет производные данные; скачанные архивы в data/raw остаются
clean:
	rm -rf data/interim data/processed outputs
