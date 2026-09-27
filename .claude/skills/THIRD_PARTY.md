# Сторонние скилы

Файлы лицензий лежат рядом и обязательны при распространении копий. В результаты работы (код решения, отчёт, лендинг)
эти скилы ничего не добавляют.

## K-Dense-AI/scientific-agent-skills — MIT (`LICENSE-scientific-agent-skills.md`)

Коммит `49c6e97775eaa18ba791bebe23162a70ae601c18` (21.09.2026):
aeon, geopandas, networkx, peer-review, scientific-visualization, scientific-writing, scikit-learn, shap,
statistical-analysis, statsmodels, umap-learn.

Изменения: удалены разделы «Citing Scientific Agent Skills» и поле `skill-author`; в scientific-writing и
peer-review удалены требования раскрывать использование ИИ-инструментов.

## obra/superpowers — MIT (`LICENSE-superpowers.md`)

Коммит `8ca22dba9a94f28898bbce59f2537ff4d87c747d` (25.09.2026):
brainstorming, writing-plans, executing-plans, subagent-driven-development, dispatching-parallel-agents,
verification-before-completion, systematic-debugging, test-driven-development, requesting-code-review,
receiving-code-review, finishing-a-development-branch, using-git-worktrees.

Изменения: из brainstorming удалён браузерный «visual companion» (скрипты сервера с внешним логотипом);
убраны префиксы `superpowers:` в ссылках на скилы; пути `docs/superpowers/…` → `docs/…`,
рабочая папка `.superpowers/sdd` → `.work/sdd`; убрана ссылка на не включённый скил using-superpowers.

## anthropics/skills — Apache 2.0 (`LICENSE.txt` в папке каждого скила)

Коммит `33375500bcea98d610eb30ce10ac4e59b89c390d` (24.09.2026), без изменений: frontend-design, webapp-testing.

## aref-vc/tufte-claude-skill — MIT (`tufte/LICENSE`)

Коммит `a145acf0c158f822d70ccdc2590164abb83b39cb`: tufte. Взяты только файлы, на которые ссылается SKILL.md
(без README и картинок из него). Добавлена строка: русский текст оформляется по ru-text.

## danielrosehill/Claude-Data-Visualisation-And-Publishing-Plugin — MIT (`data-storytelling/LICENSE`)

Коммит `752a981bcdb7f47f66b2156d429b20f77759dc4a`: data-storytelling. Убраны ссылки на не включённые скилы плагина.

## nextlevelbuilder/ui-ux-pro-max-skill — MIT (`ui-ux-pro-max/LICENSE`)

Коммит `09170eec67eefd46a7ae85de61b40c194020f997` (27.09.2026, версия 2.13.0): ui-ux-pro-max. Взяты SKILL.md,
`references/`, `data/` и скрипты поиска `core.py`, `design_system.py`, `reasoning_contract.py`, `search.py`;
не взяты тесты и `validate_data.py` (инструмент авторов для обновления базы). Пути команд `${CLAUDE_PLUGIN_ROOT}/…`
заменены на путь от корня репозитория, добавлены строки про запуск в Windows и про русский текст по ru-text.

## geserdugarov/shared-skill-ru-text — MIT (`ru-text/LICENSE`)

Коммит `bbab21fff36b3167ae8b549e353965cb62295715`: ru-text. Убраны строки с благодарностями и ссылками
на `sources.md` (сам файл не включён), метаданные чужой платформы, конфиги агентов других платформ и
отсутствующая команда плагина.
