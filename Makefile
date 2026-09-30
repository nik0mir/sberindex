# Запуск пайплайна: make all, отдельный этап: make data, другой конфиг: make CONFIG=configs/my.yaml cluster
# Этап 1: make panel eda; один раздел разведки: make eda ONLY=e3 (несколько: ONLY=e3,e4; сводка и отчёт: ONLY=syn)
# Windows без make: uv run --frozen python -m munnet panel eda (раздел: ... -m munnet eda --only e3)
# Этап 2: make features (узлы сети и признаки узлов; режим узлов — nodes.mode в конфиге)
CONFIG ?= configs/default.yaml
ONLY ?=
RUN = uv run --frozen python -m munnet --config $(CONFIG)
STAGES = data panel eda features network cluster evaluate dynamics interpret site

.PHONY: setup all $(STAGES) test lint requirements clean

# Окружение строго по uv.lock
setup:
	uv sync --frozen

all:
	$(RUN) all

$(STAGES):
	$(RUN) $@ $(if $(and $(ONLY),$(filter eda,$@)),--only $(ONLY))

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
