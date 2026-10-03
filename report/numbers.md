# Реестр чисел методологического отчёта

Откуда взято каждое число `report/report.md`. Нужен для сверки (check-facts): число текста должно совпадать
с источником.

**Редакция 02.10.2026 (этап 7).** Отчёт сокращён: подробности ушли в отчёты этапов `docs/*.md` и в новые приложения
В (воспроизводимость) и Г (польза сверки). Реестр полный, строки не удалялись. Колонка «Раздел» — где число стоит
в новой редакции (В и Г — приложения); «в report.md нет — `docs/…`» — числа в отчёте больше нет, источник прежний,
а само число осталось в отчёте этапа; «часть чисел — только …» — в отчёте осталась часть чисел строки.

Как читать колонку «Источник»:

- ключ вида `cl.final_k` — поле `text` этого ключа в файле фактов этапа: `e1.`…`e5.` и `syn.` —
  `outputs/eda/facts.json`; `feat.` — `outputs/features/report_facts.json`; `net.` —
  `outputs/network/report_facts.json`; `cl.` — `outputs/cluster/report_facts.json`; `icvi.` —
  `outputs/evaluate/report_facts.json`; `dyn.` — `outputs/dynamics/report_facts.json`;
- `int.<ключ>` — значение ключа в `outputs/interpret/facts.json` (этап `interpret`). Числа §1 и §8 сверены 30.09.2026 по выходам полного перерасчёта этапа (завершён 30.09.2026 14:41, тот же код и конфиг): от утреннего прогона `facts.json` отличается только названиями типов после слепой проверки (`names_final`, `naming_test`), новым блоком пояснений `post_unsealing` и временем (`seconds`, `timing`); все остальные числа совпали побайтно. 02.10.2026 этап пересчитан целиком (кэш сброшен правкой палитры сайта): все 24 CSV
  и `naming_test.json` совпали с прогоном 30.09 побайтно, в `facts.json` отличаются только `seconds` и `timing`
  (раздел 10). `csv:outputs/interpret/…` — CSV того же этапа;
- `use.<ключ>` — значение ключа в `outputs/usefulness/facts.json` (этап `usefulness`, разведка после вскрытия; первый прогон 01.10.2026 10:23:55 +0300 — время записи файлов, после коммитов `82636b3` и `cf9cd49` в 10:22:51). Повторный прогон 01.10.2026 в отдельную папку дал те же sha256 трёх файлов. После пересчёта `interpret` 02.10.2026
  этап повторён: пять CSV и `by_type.json` — побайтно, в `facts.json` отличается только `inputs_sha256`. `csv:outputs/usefulness/…` — CSV того же этапа;
- `bt.<ключ>` — значение ключа в `outputs/usefulness/by_type.json` (проверка `usefulness.by_type_test`: правила — `f744563`, код — `183d683`, прогон 01.10.2026), `bt.main` — ключи `runs.main`; `size.<ключ>` — значение ключа в `outputs/usefulness/size_check.json` (разведка после вскрытия `usefulness.size_posthoc`, коммит `8f73a3a`). Файлы на диске — от прогона этапа 02.10.2026 00:56–00:57 тем же кодом `usefulness_by_type.py` (`git diff 183d683 8f73a3a -- src/munnet/usefulness_by_type.py` пуст); повтор 02.10.2026 в отдельную папку дал те же sha256 (раздел 10);
- `controls:<ключ>` — поле `actual` в `outputs/panel/controls.json`;
- `yaml:<путь>` — значение в `configs/default.yaml`;
- `csv:<файл> [строка; колонка]` — значение в CSV;
- `git:<коммит>` — `git show <коммит>`;
- `замер` — измерение 28.09.2026 при подготовке отчёта (команда — в разделе 10 отчёта).

Числа, которые повторяют уже внесённое (например, 1776 в разных разделах), внесены один раз — при первом
появлении. Номера разделов, рисунков, таблиц и формул статей не вносятся; годы в ссылках на литературу —
по приложению Б отчёта (проверены по DOI).

## Раздел 1. Вопрос и главный вывод

| Раздел | Число в тексте | Источник | Что это |
|---|---|---|---|
| 1, 8, Г | ρ 0,55; 0,37 | `int.t1.per.retail.rho_a` (0,551), `int.t1.per.catering.rho_a` (0,374) | ρ Спирмена порядка типов с оборотом розницы и общепита на жителя относительно региона, без страт |
| 8 | ε² 0,152 и 0,194; 0,002 | `cl.val_ip_per_1000_difference`, `cl.val_orgs_per_1000_difference`; `cl.val_ip_per_1000_difference_null`, `cl.val_orgs_per_1000_difference_null` (0,002) | внешняя проверка этапа 3 |
| 1 | 0,35 и 0,28 против 0,24 и 0,17 | `int.t1.per.catering.best_rival_rho` (0,345), `int.t1.per.retail.best_rival_rho` (0,277), `int.t1.per.catering.rho_b` (0,237), `int.t1.per.retail.rho_b` (0,173) | ρ внутри страт размера и доли горожан: лучшее деление без типов и типы |
| 1 | AMI 0,31; порог 0,30 | `int.t5.max_ami` (0,313, деление `sized:log_level_rel:+`); `yaml:interpret.tests.T5_trivial.ami_max` | T5 |
| 1 | 181; 21; 47 | `int.t3.main.n_reliable`, `int.t3.main.median`, `int.t3.main.p95` (47,05) | надёжные переходы и плацебо, основной расчёт |
| 1, 8 | 8 из 9 | `csv:outputs/interpret/r1_runs.csv` [test = T3_reliable_placebo; runs кроме main — 9 строк, из них confirmed 8, not — variant:graph_basket_cos] | прогоны устойчивости R1 |
| 1 | 148; 33; 0,82; 0,49 | `int.t2.main.n_up`, `int.t2.main.n_down`, `int.t2.main.share_up` (0,818), `int.t2.main.p0` (0,488) | направление переходов |
| 8, Г | 0,036; 0,050; 0,042; 0,038; −0,005 (от −0,009 до −0,003) | `int.t7.median_error.B` (0,0362), `int.t7.median_error.D` (0,0496); без поправки на регион — `int.t7.median_error_abs.D` (0,0416), `int.t7.median_error_abs.C` (0,0382), `int.t7.diffs.B-D_abs` (−0,0054; интервал −0,0086…−0,0032) | медианная ошибка: соседи по региону, похожие по корзине без типа; без поправки на регион — похожие по корзине, случайные того же размера; разность «соседи минус похожие по корзине» без поправки |
| 1 | у 93% узлов | `int.tree.accuracy` (0,927) | правило дерева восстанавливает тип |
| 1, 8, Г | 46,9%; 57,3% узлов с типом; ρ от 0,46 до 0,55 | `use.type_flag.per_variant` [graph_basket_cos 0,4690 × 1776; nodes_separate 0,5727 × 1774] (счёт — `csv:outputs/interpret/node_r1.csv`, kind = variant, узлы с типом: 833 и 1016); `csv:outputs/interpret/t1_runs.csv` [turnover = retail; rho_a: наименьший 0,461 — variant:nodes_separate, наибольший 0,551 — main и seed-прогоны] | тип отдельного МО при другом правиле рёбер и с районами столиц отдельно; порядок типов по рознице во всех прогонах R1 |
| 1, 8 | `90991e1`, 29.09.2026; `abc6107` | `git:90991e1` (2026-09-29 23:41 +0300), `git:abc6107` | предрегистрация этапа 5; код, написанный вслепую |
| 1, 8, Г | 882 из 1542; 57,2% (95% интервал 54,7–59,7%); «чаще, чем нет»; 50% | `use.rule.vs_D.works` (882), `use.rule.n` (1542), `use.rule.vs_D.share` (0,5720), `use.rule.vs_D.share_ci` (0,5472; 0,5966), `use.rule.vs_D.words`; `use.rule.reference` | доля случаев, где соседи по региону ближе похожих по корзине МО других регионов, цель без поправки на регион |
| 1, 8, Г | 0,035 против 0,036; «примерно в половине случаев» — 48,4% (45,9–50,8%); было 52,5% | `use.rule.median_error_abs.R` (0,0353), `use.rule.median_error_abs.B` (0,0362); `use.rule.vs_R.share` (0,4838), `use.rule.vs_R.share_ci` (0,4588; 0,5080), `use.rule.vs_R.words`; «до» — `yaml:usefulness.rule_share.tie_tol` (комментарий «Было (точное «<»): B ближе R в 810 из 1542, равенств 11») и `git:766f891` (сообщение); 52,5% и интервал — прежняя строка этого реестра (`git show 766f891^:report/numbers.md`: `use.rule.vs_R.share` 0,5253, `.share_ci` 0,5007; 0,5497) | против случайных МО своего региона (набор R) |
| 1, 8, 9 | 32,5% | `use.type_flag.stable_share` (0,3255) | флаг «тип устойчив» |
| — (история правки) | (было 47–54%) | — | правка 01.10, порция 6d: вместо «47–54%» — 46,9% и 57,3%, те же числа и знаменатель (узлы с типом), что в разделах 8 и 9 и на сайте; 54,1% (1091 из 2016, вместе с районами столиц) остаётся только в разделе 9 как второй знаменатель |
| 1, 8, 9, В | `82636b3`; `cf9cd49` | `git:82636b3`, `git:cf9cd49` (оба 2026-10-01 10:22:51 +0300) | предрегистрация блока `usefulness`; код и тесты |
| 1, Г | примерно на 13% | арифметика: 1 − `use.rule.median_error_abs.B` / `use.rule.median_error_abs.D` = 1 − 0,036222 / 0,041634 = 0,1300 (те же значения — `int.t7.median_error_abs`) | на сколько медианная ошибка соседей по региону меньше, чем у похожих по корзине МО других регионов, цель без поправки на регион; отношение медиан, а не медиана отношений |
| 1 (0,5–0,6 п. п.), Г (0,005) | 0,005; около 0,5–0,6 процентного пункта (п. п.) | разность медиан `int.t7.diffs.B-D_abs` (−0,00541). Перевод: ошибка — разность логарифмов δ = ln((1 + g₁)/(1 + g₂)), поэтому g₁ − g₂ = (e^δ − 1)·(1 + g₂) ≈ 0,0054·(1 + g₂): 0,54 п. п. при g₂ = 0 и 0,64 п. п. при медианном росте розницы 1542 МО 18,9% (команда: медиана log1p(retail_pc) 2024 − 2023 из `data/processed/context_annual.parquet` по `territory_id` файла `outputs/usefulness/rule_by_mo.csv` → 0,1728, exp − 1 = 0,189; в тексте отчёта это число не приводится) | выигрыш в понятных единицах; разность медиан, а не медиана поэлементной разности (−0,0053) |
| 1 | два совета по размеру; «примерно четыре пятых»; «пятая часть по населению» | решение участника 02.10.2026 (`yaml:usefulness.size_posthoc`, шапка блока); `yaml:usefulness.size_posthoc.quintiles` (`n: 5`, `large: top`) | четыре нижних квинтиля и верхний |
| 1, 8, Г | 53,5 тыс. жителей | `size.quintile_bounds[3]` (53 546,3); `size.large_vs_rest.pop_from` | нижняя граница верхнего квинтиля, население 2023 года (`context_annual.pop_avg`) |
| 1, 8, Г | 56,3% (95% интервал 54,2–58,4%; с поправкой на общих похожих МО — 53,5–59,3%) | `size.lower_four.share` (0,5629), `.share_ci` (0,5415; 0,5840), `.share_ci_shared_centered` (0,5351; 0,5930) | четыре нижних квинтиля вместе, доля «B ближе D», среднее по трём показателям; разведка после вскрытия (замечание внутренней проверки 02.10); до 02.10 здесь было «55–58% в каждой из четырёх групп» |
| 1, 8, Г | 50,7% (интервал 46,0–54,2%; с поправкой на общих похожих МО — 45,5–55,9%) | `size.quintiles[4].share` (0,5066), `.share_ci` (0,4601; 0,5425), `.share_ci_shared_centered` (0,4550; 0,5589) | верхний квинтиль; верхняя граница 0,54247 → 54,2% (не 54,3) |
| 1 | «в основном городские округа» | команда: `size_by_mo.csv` (in_t7_common, quintile = 5) × `data/processed/features_nodes.parquet` (mo_type) → go 194, mr 88, mo 27 из 309; числа в отчёт не вносятся | утверждение без числа |
| 1, 8 | «подтвердилась во всех 8 прогонах»; −5,3 п. п. | `bt.verdicts` (8 × confirmed), `bt.main.delta` (−0,05261) | проверка `usefulness.by_type_test` |
| 1, Г | «не видно, чтобы тип при равном размере что-то добавлял»: −3,2 п. п., интервал от −7,2 до +1,6; p 0,55 | `size.type_given_size.delta` (−0,03209), `.delta_ci` (−0,07235; 0,01620), `.permutation.p_less` (0,5522) | разведка после вскрытия; до 02.10 — «тип ничего не добавляет» |
| Г (Г.4, «Ограничения») | «рядом с ним чаще сельские районы с другой экономикой» — наше объяснение | `yaml:usefulness.by_type_test.hypothesis` («механизм не проверяется»); решение участника 02.10 | не проверено, числа нет |

С 02.10.2026 раздел 7 повторяет из чисел раздела 1 только 181; таблица «шести проверок» раздела 1 — 0,35 и 0,28
против 0,24 и 0,17; AMI 0,31; 181; 47; 8 из 9; 148; 33; 0,82; 0,49.

## Шапка и раздел 2. Данные

| Раздел | Число в тексте | Источник | Что это |
|---|---|---|---|
| шапка | 10 методов | `cl.n_methods` | методов кластеризации, включая базовую линию |
| шапка, 4, 7 | 13 окон | `dyn.n_windows` | скользящих окон |
| шапка, 4 | 12 месяцев | `yaml:dynamics.window_months` | длина окна |
| 2 | 2190 | `e1.n_mo` | МО панели |
| 2 | 77 | `e1.n_regions` | регионов с тратами |
| 2, 8 | 1776 | `feat.n_nodes` | узлов сети |
| 2 | 2023-01…2024-12 | `yaml:period` | период данных |
| 2 | 8 регионов | `e1.n_absent_regions` | регионов без данных; список — `e1.absent_regions` |
| в report.md нет — `docs/eda.md` | 153 | `e1.n_missing_outside` | МО без трат в остальных регионах |
| в report.md нет — `docs/eda.md` | 303 126 | `controls:consumption_rows` | строк исходной таблицы трат |
| 2 | 27,1% | `e4.share_median_other` | медиана доли «Прочего», 2024 год |
| 2 | 24 месяца | `yaml:features.min_months` | полный ряд |
| 2 | 2016 | `e1.n_full` | МО с полным рядом |
| в report.md нет — `docs/eda.md` | 92,1% | `e1.full_share` | их доля |
| 2 | 17 698 | `e1.pop_median_incomplete` | медиана населения неполных |
| 2 | 23 841 | `e1.pop_median_full` | медиана населения полных |
| 2 | 4,5·10⁻⁸ | `e1.missing_mw_p` | p теста Манна — Уитни |
| 2 | 247 | `e1.n_inner_city` | внутригородских территорий |
| в report.md нет — `docs/eda.md` | 11,3% | `e1.inner_city_node_share` | их доля среди МО |
| в report.md нет — `docs/eda.md` | 15,5% | `e1.pop_share_inner_city` | их доля населения |
| в report.md нет — `docs/eda.md` | 1945 | `feat.n_nodes_all` | узлов в режиме `collapse` |
| в report.md нет — `docs/eda.md` | 169 | `feat.n_dropped` | узлов вне сети |
| 2 (Бурятия и др. — в report.md нет, `docs/eda.md`) | 73 региона из 77; Бурятия 22 МО, Дагестан 12, Чечня 5, Ингушетия 2 | `int.scope.n_regions` (73; сайт сверяет с узлами сети, `landing.py`, проверка n_regions); `e1.n_all_incomplete_regions` (4), `e1.all_incomplete_regions` | регионы, где есть узлы сети; регионы, где ряд неполный у всех МО (порция 6d: почему на сайте 73, а в отчёте 77) |
| 2 | +15,1% | `e4.growth_median` | медианный номинальный рост трат |
| 2 | 21 | `e1.controls_hard_n` | жёстких контрольных чисел; не сошлось — `e1.controls_hard_failed` = 0 |
| в report.md нет — `docs/eda.md` | 32 | `e1.controls_soft_n` | мягких; вне допуска — `e1.controls_soft_warnings` = 0 |
| 2 | 24 (`features.min_months: 24`) | `yaml:features.min_months` | параметр |

## Раздел 3. Признаки узлов

| Раздел | Число в тексте | Источник | Что это |
|---|---|---|---|
| 3 | 1986 (Aitchison) | литература | год издания |
| 3, 8 | 0,29 | `feat.eta2_raw_min` | наименьший η² как есть |
| 3 | 0,62 | `feat.eta2_raw_max` | наибольший η² как есть |
| в report.md нет — `docs/features.md` | 0,04 | `feat.eta2_random` | η² шума |
| 3 | 0,00 | `feat.eta2_rel_max` | наибольший η² после поправки |
| 3 | 6 (рёбра) | `feat.n_edges` | признаков рёбер; прошли — `feat.n_edges_ok` |
| 3 | 11 (атрибуты) | `feat.n_attributes` | атрибутов; прошли — `feat.n_attributes_ok` |
| 3 | 2 (слой) | `feat.n_layer` | признаков слоя; прошли — `feat.n_layer_ok` |
| 3 | 2 (динамика) | `feat.n_dynamics` | признаков динамики |
| 3 | пять укрупнённых разделов ОКВЭД2 | `yaml:features.space.attributes` | пять признаков `emp_sh_*` |
| 3 | 0,5 | `feat.rel_min` | порог надёжности; `yaml:features.space.min_reliability` |
| в report.md нет — `docs/features.md` | 0,6 | `feat.eta2_max` | порог η² атрибута |
| 3 | 0,8 | `feat.rho_max` | порог \|ρ\|; `yaml:features.space.max_abs_rho` |
| в report.md нет — `docs/features.md` | 0,86–0,96 | `feat.basket_rel_range` | надёжность корзины |
| в report.md нет — `docs/features.md` | 0,25 | `feat.dec_peak_rel` | надёжность декабрьского пика |
| 3 | 0.001 | `yaml:features.clr_floor` | порог замены малых долей |
| 3 | 0,121 | `feat.tree_kappa_clean` | каппа дерева, очищенная корзина |
| 3 | 0,315 | `feat.tree_kappa` | каппа дерева, корзина относительно региона |
| в report.md нет — `docs/features.md` | 0,005 | `feat.ami_region` | AMI типов с регионом |
| в report.md нет — `docs/features.md` | 0,73 | `feat.edge_rho_max` | \|ρ\| уровня трат и доли общепита (`feat.edge_rho_pair`) |

## Раздел 4. Рёбра сети

| Раздел | Число в тексте | Источник | Что это |
|---|---|---|---|
| 4 | 0,91 | `net.sig_corr_raw_median` | медиана корреляции сырых рядов |
| 4 | 0,97 | `net.sig_cos_abs_median` | медиана косинуса корзин без поправки |
| 4 | −0,01 | `net.sig_corr_own_median` | то же для своего ритма |
| 4 | 0,01 | `net.sig_cos_rel_median` | то же для корзин относительно региона |
| 4 | 0,57 | `net.sigma_basket_dist` | σ ядра |
| 4 | k = 10 | `yaml:network.sparsify.k` | соседей в kNN |
| 4 | пять правил | `net.n_candidates` | кандидатов |
| в report.md нет — `docs/network.md` | 3 месяца | `yaml:network.rules.rhythm_lag.max_lag` | наибольший лаг |
| в report.md нет — `docs/network.md` | 2 месяца | `yaml:network.rules.rhythm_dtw.window` | окно DTW |
| 4, 8 | 0,15; 0,14; 0,08 | `net.reliability_basket_dist`, `net.reliability_basket_cos`, `net.reliability_rhythm_corr` | надёжность |
| 4 | 1 (дороги) | `net.reliability_geo_road` | 1,00 |
| 4 | 0,004; 0,004; 0,005; 0,003 | `net.reliability_chance_basket_dist`, `…_basket_cos`, `…_rhythm_corr`, `…_geo_road` | случайный уровень |
| 4 | 0,009; 0,009; 0,084; 1,000 | `net.geo_jaccard_basket_dist`, `…_basket_cos`, `…_rhythm_corr`, `…_geo_road` | доля рёбер, общих с дорогами |
| 4 | 3,1%; 3,0%; 25,7%; 76,7% | `net.within_region_basket_dist`, `…_basket_cos`, `…_rhythm_corr`, `…_geo_road` | рёбер внутри группы региона |
| в report.md нет — `docs/network.md` | 10 признаков | `yaml:network.attributes.columns` | признаков согласованности |
| 4, 8 | 0,27; 0,25; 0,22; 0,07 | `net.attr_consistency_basket_dist`, `…_basket_cos`, `…_rhythm_corr`, `…_geo_road` | согласованность с местом |
| 1, 4, 8 | 0,1; 0,2; 2,4; 0,3 | `net.kocc_skew_basket_dist`, `…_basket_cos`, `…_rhythm_corr`, `…_geo_road` | асимметрия k-встречаемости |
| 4 | 25; 23; 91; 25 | `net.deg_max_basket_dist`, `…_basket_cos`, `…_rhythm_corr`, `…_geo_road` | наибольшая степень |
| 8 (часть чисел — только `docs/network.md`) | 0,05; 0,03; 0,53; 0,87 | `net.assort_north_basket_dist`, `…_basket_cos`, `…_rhythm_corr`, `…_geo_road` | ассортативность по Северу |
| 4 | 10 сетей | `net.n_rules` | сетей в полной таблице |
| 4 | 24 порядка | `csv:outputs/network/selection_sets.csv [set = main; winners = basket_dist: 24]` | порядков основного набора |
| 4 («k от 5 до 20») | k = 5, 10, 15 и 20 | `csv:outputs/network/selection_by_k.csv [winner = basket_dist при всех k]` | чувствительность к k |
| в report.md нет — `docs/network.md` | три набора критериев | `yaml:network.selection` | `priority`, `sets.plan`, `sets.extended`; Борда — `selection_sets.csv [borda]` |
| 4, 9 | `cc6f265` | `git:cc6f265` | запись PLAN.md 25.09.2026 |
| 4 | 20 из 20 | `net.boot_n_dist_better_reliability`, `net.n_boot` | парный бутстрап месяцев (в отчёте с 03.10.2026) |
| 4 | 0,016 | `net.tol_reliability` | допуск ничьей по надёжности; правило — `yaml:network.selection.tie` |
| 4 | «добавлен после первой проверки» | `yaml:network.selection.tie` (комментарий «добавлен 27.09 после замечания судьи»); `docs/network.md`, раздел 5 | допуск ничьей не предрегистрирован |
| 4 | 25.09.2026 | `git log -1 --format=%ad cc6f265` → 2026-09-25 | дата записи набора PLAN.md |
| 4 | по 3 из 6 порядков (с допуском); 4 из 6 (без допуска) | `net.set_plan_tie_winners` («Косинус корзин» — 3; «Расстояние корзин» — 3), `net.set_plan_raw_winners` («Расстояние корзин» — 4; «Косинус корзин» — 2); 6 = 3! порядков трёх критериев `yaml:network.selection.sets.plan` | набор PLAN.md |
| 4 | 0,22 | `net.reliability_basket_dist_abs` | надёжность без вычета региона |
| в report.md нет — `docs/network.md` | 0,28 | `net.check_log_ndfl_rel_basket_dist` | ассортативность по доходу 5-НДФЛ |
| в report.md нет — `docs/network.md` | 0,011 | `net.check_log_ndfl_rel_resid_basket_dist` | то же сверх зарплаты |
| в report.md нет — `docs/network.md` | ±0,010 | `net.check_log_ndfl_rel_resid_null_sd_basket_dist` | случайный уровень |
| шапка, 4, 7 | 13 окон | `net.win_rolling_n` | скользящих окон |
| в report.md нет — `docs/network.md` | 0,51 | `net.win12_noise_jaccard` | Жаккар двух сетей одного времени |
| 4, 8 | 0,15 (сети 2023 и 2024 годов) | `net.win12_change_jaccard` | Жаккар сетей двух лет |
| 4 | 0,73 | `net.ami_region_geo_road` | AMI сообществ дорог с регионом |
| 4 | 0,32 | `net.pair_ari_basket_cos__basket_dist` | ARI сообществ косинуса и расстояния |
| 4 | `k_grid: [5, 10, 15, 20]` | `yaml:network.sparsify.k_grid` | параметр |

## Раздел 5. Методы кластеризации

Выходы этапа `cluster` — прогон 30.09.2026 с `seed: 42` (`outputs/cluster/report_facts.json`, `docs/clustering.md`).

| Раздел | Число в тексте | Источник | Что это |
|---|---|---|---|
| 5 | K = 3, …, 12 | `cl.k_min`, `cl.k_max` | сетка K |
| 1, 5 | `2ee1363`, 28.09.2026 | `git:2ee1363` | коммит предрегистрации |
| в report.md нет — `docs/clustering.md` | 12 418 | `cl.n_edges` | рёбер G |
| в report.md нет — `docs/clustering.md` | 0,11 | `cl.weight_min` | наименьший вес ребра |
| в report.md нет — `docs/clustering.md` | одна компонента | `cl.n_components` | 1 |
| 5 | 11 атрибутов | `cl.n_x` | признаков X |
| в report.md нет — `docs/clustering.md` | 17 разбиений; 360; от 152 до 334 | `cl.rl_n_cands`, `cl.rl_min_inner`, `cl.rl_threshold`, `cl.rl_threshold_max` | предел разрешения модульности |
| 5 | ρ = ξ = 1 | `yaml:clustering.impl.shalileh_mirkin.rho`, `…xi` | веса KEFRiN |
| 5 | α = 0,5 | `cl.alpha`; `yaml:clustering.protocol.hybrid_alpha` | вес графа в гибриде |
| 5 (2% и 50%; 36 и 10% — `docs/clustering.md`) | 2%; 36; 50%; 10% | `cl.min_share`, `cl.min_share_nodes`, `cl.max_share`, `cl.max_noise` | пороги допустимости |
| 6 | 200 разметок | `cl.icvi_perms` | случайный базис |
| 5, 8 | 100 подвыборок; 80% | `cl.bootstrap`, `cl.subsample` | бутстрап устойчивости |
| 5 (глубина 3; остальное — `docs/clustering.md`) | глубина 3; 17 признаков; 5-кратная | `cl.tree_depth`, `cl.n_tree`, `cl.cv_folds` | дерево объяснимости |
| в report.md нет — `docs/clustering.md` | 0,02; 0 | `cl.tie_stability`, `cl.tie_interpretability`; `yaml:clustering.selection.tie` | допуски ничьей |
| в report.md нет — `docs/clustering.md` | 600 узлов; 4 группы; шести координатам | `cl.syn_n`, `cl.syn_k`, `yaml:clustering.synthetic.n_basket` | синтетика |
| в report.md нет — `docs/clustering.md` | от 0,3 до 0,9 | `cl.sbm_mix_min`, `cl.sbm_mix_max` | доля рёбер между группами в блочном графе |
| 5 | 45 ячеек; 20 наборов | `cl.syn_all_cells`, `cl.syn_repeats`; `yaml:clustering.synthetic.repeats` | |
| 5 (95%; 0,05 — `docs/clustering.md`) | 95%; 0,05 | `cl.syn_ci`, `cl.syn_min_ari`; `yaml:clustering.synthetic.ci_level`, `…min_ari` | правило ячейки |
| в report.md нет — `docs/clustering.md` | 9; 30; 6 | `cl.syn_clear_cells`, `cl.syn_tie_cells`, `cl.syn_none_cells` | явный победитель, ничья, «нет» |
| в report.md нет — `docs/clustering.md` | 16 из 16; явно — 6 | `cl.syn_both_won`, `cl.syn_both_cells`, `cl.syn_both_clear` | сигнал в обоих источниках |
| в report.md нет — `docs/clustering.md` | во всех 16 — базовая линия | `docs/clustering.md`, раздел 4, второй пункт («16 из 16»); `cl.syn_both_hybrid_best` = 0 | |
| 5 (часть чисел — только `docs/clustering.md`) | 0,91; 0,83; 0,44 | `cl.syn_strong_kmeans_joint`, `cl.syn_strong_hybrid`, `cl.syn_strong_shalileh_mirkin` | сильные сигналы |
| в report.md нет — `docs/clustering.md` | во всех 4 ячейках | `cl.syn_f0_four_tie`, `cl.syn_f0_n` | ничья на kNN без сигнала в X |
| в report.md нет — `docs/clustering.md` | одна ячейка; 0,3; 0,98 | `cl.syn_graph_only_clear_n`, `cl.syn_graph_only_clear_text`, `cl.sbm_low_leiden` | явная победа метода только по графу |
| 5 (5 из 5 и «в 1»; 0,43, 0,34, 0,17 и «в 3» — `docs/clustering.md`) | 5 из 5; 0,43; 0,34; 0,17; в 1; в 3 | `cl.sbm_mid_hybrid_best`, `cl.sbm_mid_n`, `cl.sbm_mid_hybrid`, `cl.sbm_mid_spectral`, `cl.sbm_mid_leiden`, `cl.sbm_mid_hybrid_clear`, `cl.sbm_mid_tie_spectral` | блочный граф, доля 0,5 |
| 5 (0,03 и 0,46; 2,0 и 0,39 — `docs/clustering.md`) | 2,0; 0,03; 0,46; 0,39 | `cl.syn_fmax`, `cl.syn_g0_hybrid`, `cl.syn_g0_kmeans`, `cl.syn_g0_shalileh_mirkin` | шумовой граф kNN |
| в report.md нет — `docs/clustering.md` | 0,7; 0,04; 0,43; 0,40 | `cl.sbm7_hybrid`, `cl.sbm7_kmeans`, `cl.sbm7_shalileh_mirkin` | блочный граф, доля 0,7, сигнал в X 2,0 |
| в report.md нет — `docs/clustering.md` | во всех 9 ячейках; 9 из 9; 0,05 | `cl.syn_noisy_x_tie`, `cl.syn_noisy_live`, `cl.syn_min_ari` | шумовой граф (kNN без сигнала в графе и блочный граф с долей от 0,7), ячейки не «нет»: ничья K-means, гауссовой смеси и KEFRiN; то же — табл. 13 |
| 9 | 28.09; `512cc67` | `git log --format='%h %ad' --date=short -S sbm_mixing -- configs/default.yaml` → `512cc67 2026-09-28`; предрегистрация `2ee1363` синтетику прямо исключает (комментарий блока `clustering` в `git show 2ee1363 -- configs/default.yaml`) | синтетика целиком не предрегистрирована; то же — `docs/clustering.md`, раздел 10 |
| 9 | 29.09; было 5 наборов | `docs/clustering.md`, раздел 10, пункт «Синтетика изменена 29 сентября» | |
| в report.md нет — `docs/clustering.md` | 0,040; 0,149 | `cl.syn_none_best_max`, `cl.syn_live_best_min` | разрыв у порога 0,05 |
| 5 | 87; 29 | `cl.n_candidates`, `cl.n_feasible` | кандидатов, допустимых |
| 5 | 10 значений K | `cl.cands_shalileh_mirkin`, `cl.cands_kmeans_joint`; допустимых — `cl.feas_shalileh_mirkin` = 0, `cl.feas_kmeans_joint` = 0 | |
| в report.md нет — `docs/clustering.md` | 105 из 108 | `cl.n_small_market`, `cl.n_small` | мелкие типы по доступности рынков |
| в report.md нет — `docs/clustering.md` | 16 районов; 26 MAD | `cl.suburbs_size`, `cl.suburbs_scaled` | мелкий тип гибрида |
| в report.md нет — `docs/clustering.md` | 1,2%; ξ/ρ от 16 | `cl.kef_z1_minshare_k4`, `cl.kef_z_feasible_min_xi` | KEFRiN в варианте статьи |
| в report.md нет — `docs/clustering.md` | 51,5% | `cl.hybrid_k3_max_share` | крупнейший тип гибрида при K = 3 |
| 5, 8 | K = 4; 806, 473, 394, 103 | `cl.final_k`, `cl.type1_size` … `cl.type4_size` | итог |
| 5 (часть чисел — только `docs/clustering.md`) | 0,91; 0,90; 0,75 | `cl.final_stability`, `cl.final_jaccard_min`, `cl.jaccard_stable` | устойчивость итога |
| в report.md нет — `docs/clustering.md` | 10 seed; ARI 1,00 | `cl.seeds`, `cl.final_seed_ari` | разбиения гибрида K = 4 по seed метода |
| 5 | seed 42–46; 5 из 5 (оба правила) | `cl.seed_min`, `cl.seed_max`, `cl.n_seeds`, `cl.sf_prereg_eligible`, `cl.sf_tolerance_eligible` | итог по seed |
| в report.md нет — `docs/clustering.md` | фронт из трёх: Leiden K = 3, Louvain K = 6, гибрид | `cl.all_front`, `cl.all_front_n` | все семейства, seed 42 |
| в report.md нет — `docs/clustering.md` | +1 | `cl.all_top_score`, `cl.all_tied` | очки Копленда |
| в report.md нет — `docs/clustering.md` | 0,91 против 0,62 | `cl.final_stability`, `cl.leiden_stability` | равенство очков решает устойчивость |
| в report.md нет — `docs/clustering.md` | Борда на фронте: 7 и 7 (Leiden K = 3, гибрид), 10 (Louvain K = 6); 0,909 против 0,615 | `cl.borda_front`, `cl.borda_leiden`, `cl.borda_hybrid`, `cl.borda_n_tied` = 2, `cl.borda_outcome`, `cl.borda_consistent` = 1 | ничья, решает устойчивость; сумма мест пересчитана вручную по `csv:outputs/cluster/selection_levels_all.csv [level = 2; on_front = True]` |
| в report.md нет — `docs/clustering.md` | seed конфига 43 и 44: поровну у трёх (Leiden, спектральная, гибрид), выбран гибрид | `cl.borda_outcome` в `tests/fixtures/cluster_report_facts/seed43.json` (по 10; 0,912 / 0,902 / 0,628) и `seed44.json` (по 9; 0,910 / 0,897 / 0,623) | фикстуры — `report_facts.json` прогонов 43 и 44; сами прогоны вне репозитория |
| 5 | 4 из 5; спектральная K = 4 — 1 | `cl.sf_prereg_all` | все семейства, seed 42–46, предрегистрация |
| в report.md нет — `docs/clustering.md` | наименьший ARI 1,00 | `cl.sf_all_minari`, `cl.sf_all_minari_min` | разбиения победителей по seed |
| в report.md нет — `docs/clustering.md` | спектральная не на фронте при seed 42: качество в X 0,50 против 0,25; на G 0,50 и 0,50; устойчивость 0,909 и 0,901; объяснимость 0,894 и 0,905; допуск 0,02 | `csv:outputs/cluster/selection_levels_all.csv [level = 2; cand = hybrid_k04, spectral_k04; crit_quality_features, crit_quality_graph, crit_stability, crit_interpretability]`; `cl.tie_stability`, `cl.tie_interpretability` | гибрид доминирует спектральную только по качеству в X |
| в report.md нет — `docs/clustering.md` | 0,91 против 0,90 | `cl.final_stability`, `cl.spectral_stability` | |
| в report.md нет — `docs/clustering.md` | 7 из 8 | `cl.all_n_same`, `cl.all_n_checks` | проверки чувствительности, все семейства |
| в report.md нет — `docs/clustering.md` | один шаг — спектральная K = 4 | `cl.all_changed`, `cl.all_joint_winner`, `cl.all_joint_k` | |
| в report.md нет — `docs/clustering.md` | 24 порядка; 12; 6; 6 | `cl.orders_total`, `cl.lex_l2_text` | порядки при выборе метода |
| 5 | Leiden K = 3; 0,927; 0,884; 0,606; 0,590 | `cl.raw_l3_winner_graph`, `cl.raw_l3_winner_graph_k`, `cl.avi_adj_leiden`, `cl.avi_adj_hybrid`, `cl.mq_raw_leiden`, `cl.mq_raw_hybrid` | метрики графа, сравнимые между K |
| 5 (55% и спектральная K = 3; 0,008 и K = 4 / K = 3 — `docs/clustering.md`) | 55%; спектральная K = 3; 0,008; 0,02; K = 4 / K = 3 | `cl.grid55_all_winner`, `cl.grid55_all_k`, `cl.hybrid_k3_k4_stab_diff`, `cl.tie_stability`, `cl.grid55_strict_hybrid_k`, `cl.grid55_tol_hybrid_k` | порог крупнейшего типа |
| в report.md нет — `docs/clustering.md` | 1%–3% не меняет | `cl.grid_minshare_effect` = 0; `csv:outputs/cluster/threshold_grid.csv` | порог наименьшего типа |
| в report.md нет — `docs/clustering.md` | K = 3 не участвовал во внешней проверке | `csv:outputs/cluster/validation.csv [cand: hybrid_k03 нет]` | |
| в report.md нет — `docs/clustering.md` | iK-means: 16 по X, 20 по корзине ⊕ X; порог 2 узла; граница сетки 12 | `cl.ik_x`, `cl.ik_joint`; `yaml:clustering.impl.ik_means_min_size`; `yaml:clustering.n_clusters` | ориентир K, в выбор не входит |
| в report.md нет — `docs/clustering.md` | 5 базисам; 0,03; 0,04 | `yaml:clustering.impl.seed_check`; `cl.tol_qf`, `cl.tol_qg` | допуск по шуму базиса |
| в report.md нет — `docs/clustering.md`, таблица 6б | таблица «правило × уровень» | `cl.final_label`, `cl.final_k`, `cl.all_winner`, `cl.tol_eligible_winner`, `cl.tol_all_winner`, `cl.sf_prereg_eligible`, `cl.sf_tolerance_eligible`, `cl.sf_prereg_all`, `cl.sf_tolerance_all` | |
| в report.md нет — `docs/clustering.md` | 0,031–0,033; 0,030–0,039 | `cl.tol_jk_qf_min`, `cl.tol_jk_qf_max`, `cl.tol_jk_qg_min`, `cl.tol_jk_qg_max` | допуск без одного базиса |
| 5 | 5 из 5 | `cl.tol_jk_winners`, `cl.tol_jk_n` | |
| в report.md нет — `docs/clustering.md` | 25 сочетаний; 25 из 25 | `cl.tol_grid_n`, `cl.tol_grid_winners` | сетка допусков |
| в report.md нет — `docs/clustering.md` | 0,032, 0,031 и 0,035; 0,043, 0,042 и 0,037 | поле `value` ключей `cl.tol_qf`, `cl.tol_qg` в `report_facts.json` трёх прогонов 30.09: seed 42 — репозиторий (0,03194; 0,04256), seed 43 и 44 — временные папки (0,03150; 0,04188 и 0,03529; 0,03694) | наблюдение по трём прогонам, одной командой не воспроизводится; те же числа — в `docs/clustering.md`, раздел 10 (опечатка 0,032 для seed 43 исправлена 30.09 в шаблоне `src/munnet/clustering/templates/clustering.md`) |
| в report.md нет — `docs/clustering.md` | seed 44 с допуском: спектральная в прогонах 42 и 43, гибрид в 44 | `csv:outputs/cluster/seed_runs.csv [check = main; scope = all; rule = tolerance; seed = 44] = spectral_k04`; то же в `seed_runs.csv` прогона 43; в прогоне 44 — `hybrid_k04` | прогоны 43 и 44 — вне репозитория |
| в report.md нет — `docs/clustering.md` | 23 из 276 (seed 44–46); 28 из 368 (seed 44–47); 0 из 368; 0 из 368, 276 и 368 | сравнение `seed_runs.csv` трёх прогонов 30.09 по ключу (`check`, `scope`, `rule`, `seed`, `kind`), число строк с разным `winner`: `rule = tolerance` — пары прогонов 42–44, 43–44, 42–43; `rule = prereg` — 42–43, 42–44, 43–44. Все расхождения — на уровне всех семейств (`scope` = `all`, `all_level2` и проверка `all_eligible`) | наблюдение по трём прогонам, одной командой не воспроизводится; прогоны 43 и 44 — вне репозитория |
| в report.md нет — `docs/clustering.md` | гибрид с K = 4 при seed 42 | `cl.all_winner`, `cl.seed` | победитель по z-оценкам среди всех семейств |
| 5 | 0,94 | `cl.ari_hybrid_spectral` | ARI гибрида и спектральной |
| 5 | 0,139; 0,183; 0,026; 0,003; 0,023; 0,014 | `csv:outputs/cluster/validation.csv [cand = spectral_k04]`: `check = difference`, `stat` — ИП 0,139123, организации 0,183152, ночёвки 0,0263739; `check = beyond_attributes`, `stat` — ИП 0,00318078, организации 0,0229629, ночёвки 0,014112; то же — `docs/clustering.md`, таблица 12 | внешняя проверка спектральной K = 4: ε² и прирост R² («почему гибрид, а не спектральная», с 03.10.2026) |
| 5 | пороги 2% и 50% записаны до расчётов | `git show 2ee1363 -- configs/default.yaml`: `feasible.min_cluster_share: 0.02`, `max_cluster_share: 0.5` | пороги допустимости в предрегистрации |
| 5 | 47; 6; 12 | `cl.imputed_urban_share_rel`, `cl.imputed_age_old_share_rel`, `cl.imputed_market_access_rel`; то же — `outputs/cluster/facts.json`, ключ `imputed`; `docs/clustering.md`, раздел 1 | пропуски X, заполненные медианой группы региона (подготовка X, с 03.10.2026) |
| 5 | 1,4826 | `munnet.clustering.inputs:MAD_SCALE`; `yaml:clustering.inputs.scale` (комментарий «(x − медиана) / (MAD × 1,4826)»); у всех 11 признаков масштаб — MAD (`outputs/cluster/facts.json`, ключ `scale_used`) | масштаб X |
| 5 | `place_center: median`; `region_center: mean`; `scale: robust`; `impute: region_median` | `yaml:features.place_center`, `yaml:features.region_center`, `yaml:clustering.inputs` | центр группы региона и подготовка X |
| 5 (0,22; 0,28 и K = 3 — `docs/clustering.md`) | 0,22; 0,28; K = 3 | `cl.var_graph_basket_cos_ari_same`, `cl.var_graph_basket_cos_ari`, `cl.var_graph_basket_cos_k` | сеть косинуса |
| в report.md нет — `docs/clustering.md` | 0,32; 0,40; K = 4 | `cl.var_nodes_separate_ari_same`, `cl.var_nodes_separate_ari`, `cl.var_nodes_separate_k` | районы отдельными узлами |
| в report.md нет — `docs/clustering.md` | 1,00 | `cl.var_no_level_ari_same` | без уровня трат |
| в report.md нет — `docs/clustering.md` | KEFRiN, K = 4; 0,15; 0,97 | `cl.var_x_clipped_winner`, `cl.var_x_clipped_k`, `cl.var_x_clipped_ari`, `cl.var_x_clipped_ari_same` | усечённые хвосты |
| в report.md нет — `docs/clustering.md` | 1000 перестановок | `cl.val_perms` | внешняя проверка |
| в report.md нет — `docs/clustering.md` | 10 проверок | `cl.val_n_sig`, `cl.val_n_tests` | значимы все |
| 5, 8 (0,024 — в разделе 8 только `docs/clustering.md`) | 0,152; 0,194; 0,024 | `cl.val_ip_per_1000_difference`, `cl.val_orgs_per_1000_difference`, `cl.val_nights_pc_difference` | ε² |
| 8 (0,002; 0,003 — `docs/clustering.md`) | 0,002–0,003 | `cl.val_ip_per_1000_difference_null` (0,002), `cl.val_orgs_per_1000_difference_null` (0,002), `cl.val_nights_pc_difference_null` (0,003) | ε² на перестановках |
| в report.md нет — `docs/clustering.md` | 4 из 4 | `cl.val_signs_ok`, `cl.val_signs_n` | знаки |
| 5, 8 | от 0,006 до 0,026; в разделе 5 — 0,006, 0,026 и 0,010 | `cl.val_ip_per_1000_beyond_attributes` (0,006), `cl.val_orgs_per_1000_beyond_attributes` (0,026), `cl.val_nights_pc_beyond_attributes` (0,010) | прирост R² |
| в report.md нет — `docs/clustering.md` | 0,85 | `cl.stab_median_kmeans` | медианная устойчивость K-means |
| в report.md нет — `docs/clustering.md` | 2–34 | `cl.small_kmeans_min`, `cl.small_kmeans_max` | мелкий тип K-means |
| в report.md нет — `docs/clustering.md` | 0,48 | `cl.syn_nograph_best_feat` | лучший метод по X без сигнала в графе |
| 5 | 0,53 | `cl.stab_median_ward` | медианная устойчивость Уорда |
| 5 | K = 3; 0,09 | `cl.win_gmm_k`, `cl.ari_hybrid_gmm` | гауссова смесь |
| в report.md нет — `docs/clustering.md` | от 5 до 400; 2 кластера | `cl.hdbscan_mcs_min`, `cl.hdbscan_mcs_max`, `cl.hdbscan_kmax` | HDBSCAN |
| в report.md нет — `docs/clustering.md` | 10 разбиений Leiden без несвязных | `cl.feas_leiden`, `cl.rl_disconnected`; `csv:outputs/cluster/resolution_limit.csv` | |
| 5 | 10% рёбер; 0,63; 0,98 | `yaml:clustering.impl.edge_perturbation.drop`, `cl.pert_leiden`, `cl.pert_hybrid` | удаление рёбер |
| 5 | 7 из 10 | `cl.louvain_n_reached`, `cl.n_grid` | |
| в report.md нет — `docs/clustering.md` | 0,90; 0,97 | `cl.spectral_stability`, `cl.pert_spectral` | спектральная |
| в report.md нет — `docs/clustering.md` | 1–32 | `cl.small_shalileh_mirkin_min`, `cl.small_shalileh_mirkin_max` | мелкий тип KEFRiN |
| в report.md нет — `docs/clustering.md` | 16 узлов (K = 4) | `csv:outputs/cluster/small_clusters.csv [cand = kmeans_joint_k04; size = 16]` | мелкий тип базовой линии |
| 5 (часть ключей — прил. В, `docs/clustering.md`) | `seeds: 10`, `hybrid_alpha: 0.5`, `n_clusters: [3, 12]`, `repeats: 20`, `ci_level: 0.95`, `min_ari: 0.05`, `seed_check: 5`, `tie_chain_tolerance: 0`, `quality_tolerance: baseline_sd` | `yaml:clustering.protocol`, `yaml:clustering.n_clusters`, `yaml:clustering.synthetic`, `yaml:clustering.impl` | параметры |

## Раздел 6. ICVI

| Раздел | Число в тексте | Источник | Что это |
|---|---|---|---|
| 6 | 87 | `icvi.n_candidates` | кандидатов |
| 6 | 200 | `icvi.permutations` | перестановок базиса |
| 6 | z = 178 | `icvi.final_z_avi` (178,0) | z AVI итога |
| в report.md нет — `docs/icvi.md` | 100 подвыборок | `icvi.bootstrap` | |
| в report.md нет — `docs/icvi.md` | 80% | `icvi.subsample` | |
| в report.md нет — `docs/icvi.md` | 2,0 | `icvi.ci_scale` | множитель отклонений |
| в report.md нет — `docs/icvi.md` | 95% | `icvi.ci_level` | |
| 6 (18 ячеек: итог, спектральная, Leiden; остальные — `docs/icvi.md`, раздел 3) | таблица «кандидат × индекс» (30 ячеек) | `csv:outputs/evaluate/icvi_long.csv [candidate ∈ {hybrid_k04, spectral_k04, leiden_k03, louvain_k06, gmm_k03}; value, ci_low, ci_high, z]` | итог: `icvi.final_sw`, `icvi.final_sw_lo`, `icvi.final_sw_hi`, `icvi.final_z_sw` и т. д. для ch, s_dbw, avi, avu, mq |
| 6 | в среднем по типам 91,3% | `icvi.final_avi` (0,913) | AVI итога в процентах; AVI — невзвешенное среднее изолируемостей типов (`munnet.icvi:avi`), поэтому «в среднем по типам» (с 03.10.2026) |
| 6 | z −6,5; −85,6 | `icvi.final_z_s_dbw` (−6,4956), `icvi.final_z_avu` (−85,571) | итог хуже случайных меток по S_Dbw и AVU; те же числа — в таблице раздела 6 и `docs/icvi.md`, раздел 3 |
| 6 | 178,0 | `icvi.final_z_avi` | |
| 6 | 2/3; 9 кандидатов | `icvi.avu_k3_maxdev` (0,000), `icvi.n_k3` | AVU при K = 3 |
| 6 | 26 | `icvi.avu_z_neg`, `icvi.avu_z_finite` | z AVU < 0 |
| в report.md нет — `docs/icvi.md` | −52,0 | `icvi.avu_z_median` | |
| 6 | 0,60 | `icvi.u_max` (пара `icvi.u_max_pair`) | U₁₂ |
| в report.md нет — `docs/icvi.md` | 0,16 | `icvi.u_median` | |
| в report.md нет — `docs/icvi.md` | 0,90 | `icvi.tau_avi_mq` | τ по z |
| в report.md нет — `docs/icvi.md` | 0,84 | `icvi.tauk_avi_mq` | τ внутри K |
| в report.md нет — `docs/icvi.md` | 8 из 29; K от 9 | `icvi.sdbw_nan`, `icvi.n_feasible`, `icvi.sdbw_nan_kmin` | S_Dbw не определён |
| 6 (часть чисел — только `docs/icvi.md`) | 0,96; 0,95; −0,92; −0,60 | `icvi.rho_z_k_avi`, `icvi.rho_z_k_mq`, `icvi.rho_z_k_ch`, `icvi.rho_z_k_sw` | ρ z с K |
| в report.md нет — `docs/icvi.md` | 0,927 | `cl.avi_adj_leiden` | AVI с поправкой, Leiden |
| в report.md нет — `docs/icvi.md` | 0,884 | `icvi.final_avi_adj` | итог |
| в report.md нет — `docs/icvi.md` | 0,249 | `icvi.final_avi_base` | E |
| в report.md нет — `docs/icvi.md` | Louvain, K = 6 первым по z AVI | `icvi.order_z_avi` | |
| в report.md нет — `docs/icvi.md` | Leiden с K = 3 при метриках с поправкой (выбор) | `cl.raw_l3_winner_graph`, `cl.raw_l3_winner_graph_k`; итог не меняется — `cl.raw_l2_changed` = 0 | фраза о пересчёте выбора |
| 6, 8 | −0,26 | `icvi.tau_cross` | τ между пространствами |
| в report.md нет — `docs/icvi.md` | −0,81 | `icvi.tau_min` (пара `icvi.tau_min_pair`) | CH и MQ |
| 6, 8 | `baseline_permutations: 200`, `s_dbw_density: pair`, `bootstrap: 100`, `ci_level: 0.95` | `yaml:icvi` | параметры |

## Раздел 7. Динамика

| Раздел | Число в тексте | Источник | Что это |
|---|---|---|---|
| 7 | 15,3% | `dyn.change_main` | сменили тип, основной способ |
| 7 | K = 4 | `dyn.k` | |
| 9 | 2023 год (признаки места) | `yaml:dynamics.tracking.place_year` | |
| 7, 8 | 1000 выборок | `dyn.bootstrap` | |
| 1, 7, 8 | 95% | `dyn.level` | |
| в report.md нет — `docs/dynamics.md` | 13,7–17,1% | `dyn.change_lo_main`, `dyn.change_hi_main` | интервал |
| 7 | 5,5% | `dyn.noise_main` | шум |
| 7 | +9,8 п. п. | `dyn.diff_main` | |
| 7 | 8,3…11,5 | `dyn.lo_main`, `dyn.hi_main` | |
| 7 | 181 | `dyn.n_rel_main` | надёжных переходов |
| 7 | 14,1%; 5,6%; +8,4 п. п.; 6,9…10,0; 161 | `dyn.change_fixed`, `dyn.noise_fixed`, `dyn.diff_fixed`, `dyn.lo_fixed`, `dyn.hi_fixed`, `dyn.n_rel_fixed` | фиксированные типы |
| 7 | 15,3%; 5,3%; +10,0 п. п.; 8,3…11,7; 184 | `dyn.change_evo`, `dyn.noise_evo`, `dyn.diff_evo`, `dyn.lo_evo`, `dyn.hi_evo`, `dyn.n_rel_evo` | эволюционная кластеризация |
| в report.md нет — `docs/dynamics.md` | 0,99 | `dyn.ari_tr_evo` | ARI переходов |
| в report.md нет — `docs/dynamics.md` | 0,91 | `dyn.ari_tr_fixed` | |
| 7 | 10,2% | `dyn.share_rel_main` | доля надёжных переходов |
| в report.md нет — `docs/dynamics.md` | 55 из 73 | `dyn.n_regions_rel`, `dyn.n_regions` | регионы с переходами |
| в report.md нет — `docs/dynamics.md` | 74; 55; 26; 18 | `dyn.flows_main` | потоки |
| в report.md нет — `docs/dynamics.md` | 0,3; 0,5; 0,7 | `dyn.tau_lo`, `dyn.tau`, `dyn.tau_hi` | пороги MONIC |
| 7 | 12 пар; все 4 типа сохранились | `dyn.n_adjacent`, `dyn.ev_adj_surv_05` (48 = 12 × 4), `dyn.ev_fl_surv_05` (4) | события; пороги 0,3 и 0,7 — `dyn.ev_*_03`, `dyn.ev_*_07` |
| в report.md нет — `docs/dynamics.md` | 2 < 1 < 3 < 4 | `dyn.cafe_order` | порядок типов по общепиту |
| в report.md нет — `docs/dynamics.md` | α = 0,05 | `dyn.driver_alpha` | |
| 7 | 1,25 | `dyn.drv_ratio` | |
| 7 | p = 0,001 | `dyn.drv_p` | |
| в report.md нет — `docs/dynamics.md` | AUC 0,57 | `dyn.drv_auc` | |
| 7 | 5-е | `dyn.drv_rank` | |
| 7 | 2,11 | `dyn.drv_top_ratio` (часть — `dyn.drv_top`) | |
| в report.md нет — `docs/dynamics.md` | 3,3·10⁻²² | `dyn.drv_top_p` | |
| в report.md нет — `docs/dynamics.md` | 1399 | `dyn.drv_alt_n` | строгое «без смены» |
| в report.md нет — `docs/dynamics.md` | 6,7·10⁻⁴ | `dyn.drv_alt_p` | |
| в report.md нет — `docs/dynamics.md` | 1,27 | `dyn.drv_alt_ratio` | |
| в report.md нет — `docs/dynamics.md` | 1,03 | `dyn.drv_level_ratio` | |
| в report.md нет — `docs/dynamics.md` | 0,379 | `dyn.drv_level_p` | |
| в report.md нет — `docs/dynamics.md` | 11 месяцев из 12 | `yaml:dynamics.window_months`, `yaml:dynamics.step_months` | пересечение соседних окон |
| в report.md нет — `docs/dynamics.md` | 0,99 (seed) | `dyn.seed_ari_min` | |
| 7, 8 | `events_jaccard: 0.5`, `bootstrap: 1000`, `window_months: 12`, `step_months: 1` | `yaml:dynamics` | параметры |

## Раздел 8. Типы локальных экономик

| Раздел | Число в тексте | Источник | Что это |
|---|---|---|---|
| 8, Г | 473, 806, 394, 103 | `csv:outputs/interpret/settlement_shares.csv` [n] (с узлами-городами; в `profile.csv` у типа 3 — 392 территориальных узла) | узлов в типе |
| 1, 8 | названия четырёх типов | `int.names.<тип>.name` | правило названий |
| 1, 8 | медианы CLR: общепит −0,43 / −0,03 / +0,41 / +0,78; продовольствие +0,16 / +0,03 / −0,15 / −0,38; маркетплейсы +0,16 / +0,03 / −0,15 / −0,37; транспорт +0,08 / −0,02 / −0,03 / +0,07; уровень трат −0,11 / −0,04 / +0,13 / +0,34 | `csv:outputs/interpret/profile.csv` [type; feature = clr_rel_cafe, clr_rel_food, clr_rel_marketplace, clr_rel_transport, log_level_rel; median] | профиль, порядок типов 2, 1, 3, 4 — `int.ladder.order` |
| 8, 9 | 1774 | `int.scope.n_territorial` | территориальные узлы профиля |
| 8 (77% — в report.md нет, `docs/interpretation.md`) | 23% / 52% / 75% / 74%; 0% / 3% / 14% / 60%; 77% | `csv:outputs/interpret/settlement_shares.csv` [cities; large_cities; rural у типа 2 = 0,768] | состав по типу поселения; условия — `yaml:interpret.naming.settlement_words` |
| 8 | Большеберезниковский район, Верховский район, Любинский район, Ярославль | `csv:outputs/interpret/examples.csv` [kind = typical; rank = 1] | типичные МО |
| 8 | e^0,41 ≈ 1,5; −0,43 → 1,5; +0,34 → 40% | exp(0,410) = 1,51; exp(0,431) = 1,54; exp(0,338) = 1,40 из `profile.csv` | чтение CLR и уровня трат |
| 8 | не больше 0,03 | `profile.csv` [type = 1; clr_rel_*; median]: наибольший модуль 0,032 (clr_rel_food) | тип 1 близок к региону |
| 8 | 4 из 4; порог 4 из 4 | `int.naming_test.correct` (4), `int.naming_test.state` (accepted), `int.naming_test.pass_min` (4); ответ проверяющего — `docs/interpretation_naming_test.json`; порог — `yaml:interpret.naming.test.pass_min` | слепая проверка названий |
| 8 (часть чисел — только `docs/interpretation.md`) | 92,7%; 92,6%; 45,4% | `int.tree.accuracy`, `int.tree.balanced_accuracy`, `int.tree.baseline_majority_acc` | дерево «корзина → тип» |
| 8 | −0,27; 0,21; −0,26; −0,34; 0,59; 96% / 97%, 96% / 94%, 92% / 80%, 92% / 92%; 11% | `csv:outputs/interpret/tree_rules.csv` [rule; precision; coverage] | правила отнесения |
| 8 (часть чисел — только `docs/interpretation.md`) | 27 867 ₽; 11 692 ₽; 3 330 ₽; 1 278 ₽ | `data/processed/panel_long.parquet`, territory_id = 44, среднее `value` за 24 месяца по category = all, food, marketplace, cafe | пример МО, траты |
| 8 (часть чисел — только `docs/interpretation.md`) | 43,3%; 9,4%; 4,6%; 31,4% | `data/processed/features_windows.parquet`, territory_id = 44, window = 2023: sh_food, sh_marketplace, sh_cafe, sh_other | пример МО, корзина |
| 1, 8, Г (часть чисел — только `docs/interpretation.md`) | +0,29 / +0,31; −0,06 / −0,07; +0,05 / +0,02; +0,19 / +0,27 | то же, window = 2023 и 2024: clr_rel_cafe, clr_rel_food, clr_rel_marketplace, log_level_rel | пример МО относительно региона |
| 8 | тип 3 у territory_id 44 | `csv:outputs/interpret/types.csv` [territory_id = 44; type] | пример МО, тип |
| 1, 8, Г | 0,050; 0,058; 1542; 10 МО | `int.t7.example.error` (0,0496), `csv:outputs/interpret/t7_errors.csv` [territory_id = 44; err_B = 0,0583], `int.t7.n_common`, `yaml:interpret.tests.T7_utility.k` | пример МО, сверка |
| 8, Г | 0,058 против 0,033 (пример этапа 5 без поправки на регион); на сайте 0,058 против 0,050 | `use.example.stage5_example.err_B` (0,0583), `use.example.stage5_example.err_D` (0,0330); сайт — `outputs/site_5a/index.html`, глава 6, таблица «Ошибки наборов таблицей», колонка «У муниципалитета примера»: B 0,0583, D 0,0496 (= `int.t7.example.error`) | Благовещенский район, две цели |
| 1, 8, 9, Г (часть чисел — только `docs/interpretation.md`) | T1: 0,52–0,59, 1705; 0,32–0,42, 1241; 0,17, 0,24; 0,28, 0,35; −0,10 (от −0,16 до −0,05); −0,11 (от −0,17 до −0,04) | `int.t1.per.retail.rho_a_ci`, `n_a`; `int.t1.per.catering.rho_a_ci`, `n_a`; `rho_b`; `best_rival_rho`; `int.t1.per.retail.diff_point`, `diff_ci`; `int.t1.per.catering.diff_point`, `diff_ci` | T1 |
| 1 (часть чисел — только `docs/interpretation.md`) | T5: 0,28; 0,12; −0,003; не выше 0,10 | `int.t5.ami.sized:log_pop_rel:+` (0,283), `int.t5.ami.sized:log_wage_rel:+` (0,123), `int.t5.region_ami` (−0,0032); наибольший AMI делений `sized:emp_sh_*` — 0,0985 (`emp_sh_industry:+`) | T5 |
| 8 (часть чисел — только `docs/interpretation.md`) | 0,024; 0,002–0,003; 0,006–0,026 | `cl.val_nights_pc_difference`; как в разделе 5 | внешняя проверка этапа 3 |
| 1, 8 | T3: 10,2%; 148; ≈ 34; от −103 до 143; 236; 11%; 8 из 9 | `int.texts.T3_main`; `int.texts.T3_reliable_placebo` (вариант `graph_basket_cos`, `int.r1.source_run.T3_reliable_placebo`); `int.post_unsealing.t3_runs` [run = variant:graph_basket_cos, scheme = main: p95 = 236,05, share_above = 0,11]; `csv:outputs/interpret/r1_runs.csv` (как в разделе 1) | T3 |
| в report.md нет — `docs/interpretation.md` | T2: 0,76–0,87; 40,9%; 19,1% | `int.t2.main.share_ci`, `int.t2_place.share_a` (0,409), `int.t2_place.share_null` (0,191) | T2 |
| 8, 9, Г (часть чисел — только `docs/interpretation.md`) | T6: 81; 586; −0,015; от −0,15 до 0,12; p = 0,60; ε² 0,082 | `int.t6.n_movers`, `int.t6.n_stayers`, `int.t6.delta`, `int.t6.ci`, `int.t6.p` (0,598), `int.t6.coverage_eps2` | T6; δ Клиффа — `src/munnet/interpret/external.py`, функция T6 |
| 8, Г (часть чисел — только `docs/interpretation.md`) | T7: 0,045; 0,048; 0,050; −0,012 (от −0,015 до −0,009); −0,005 (от −0,008 до −0,003); +0,001 (от 0,000 до 0,003) | `int.t7.median_error.C`, `.A`, `.D`; `int.t7.diffs.B-A`, `int.t7.diffs.B-A_abs`, `int.t7.diffs.D-A` | T7 |
| 8 | K = 4 | `cl.final_k` | число типов |

Пояснение к T3 («тяжёлый хвост плацебо» на сети «косинус корзин») опирается только на числа `int.post_unsealing.t3_runs` (95-й перцентиль 236 против 47 в основном расчёте, доля псевдопар выше реальности 0,11). Объяснение причины хвоста («два почти равных решения с K = 4») из сообщения коммита `ea65acd` в текст не вошло: в журнале после вскрытия и в коде его нет, отдельного расчёта в репозитории тоже.

## Раздел 8 и приложение Г, «Чем полезно» и «Флаг устойчивости типа» (этап `usefulness`)

| Раздел | Число в тексте | Источник | Что это |
|---|---|---|---|
| 1, 8, 9, В | `82636b3`; `cf9cd49`; 01.10.2026 | `git:82636b3`, `git:cf9cd49`; время записи `outputs/usefulness/*` — 2026-10-01 10:23:55 +0300 | правила до расчёта, код, первый прогон |
| 1, 8, Г | n = 1542; 10 МО в наборах; 100 розыгрышей; 58 групп; 2000 выборок | `use.rule.n`, `yaml:interpret.tests.T7_utility.k`, `use.rule.random_draws`, `use.rule.n_groups`, `use.rule.bootstrap.n` | |
| 1, 8 | 50% (пороги слов) | `yaml:usefulness.rule_share.words` (комментарии), `munnet.usefulness:share_words` | |
| 1, 8, Г | табл.: 882 / 894 / 746 из 1542 | `use.rule.vs_D.works`, `use.rule.vs_C.works`, `use.rule.vs_R.works` (746) | соседи ближе; R — после исправления 766f891 |
| 1, 8, Г | табл.: 57,2% (54,7–59,7%); 58,0% (55,4–60,4%); 52,5% (50,1–55,0%) | `use.rule.vs_D.share`, `.share_ci`; `use.rule.vs_C.share` (0,5798), `.share_ci` (0,5541; 0,6045); `use.rule.vs_R.share` (0,4838), `.share_ci` (0,4588; 0,5080) | доли и интервалы |
| 8 | табл.: «чаще, чем нет» дважды; «примерно в половине случаев» | `use.rule.vs_D.words`, `use.rule.vs_C.words`, `use.rule.vs_R.words` | слова по правилу |
| 8, Г | табл.: 0,042; 0,038; 0,035; 0,036 | `use.rule.median_error_abs.D` (0,0416), `.C` (0,0382), `.R` (0,0353), `.B` (0,0362) | медианные ошибки, цель без поправки |
| Г | 75 МО | `use.rule.vs_R.ties` (75); команда: `rule_by_mo.csv`, \|err_abs_B − err_abs_R\| ≤ 1e-12 → 75 | равенства с набором R; механизм — команда 02.10 по группам региона: у 19 из 75 МО в группе ровно столько МО с известной целью, сколько в B (R всегда = B), у 56 — группы 7–66 МО (медиана 13), где медиана ошибок по 100 розыгрышам равна ошибке B |
| Г | допуск 0,01; около 1% | `use.rule.delta.value`; `yaml:usefulness.rule_share.delta` (комментарий «0,01 ≈ 1% изменения оборота») | |
| Г | 691; 381; 470 | `use.rule.delta.b_better`, `use.rule.delta.within`, `use.rule.delta.d_better` | разбор с допуском |
| Г | −0,0053 (−0,0083…−0,0035); «у типичного МО соседи по региону ближе» | `use.rule.gain_median`, `use.rule.gain_ci`, `use.rule.gain_words` | медиана поэлементной разности |
| Г | −0,005 (разность медиан) | `int.t7.diffs.B-D_abs` (как в разделе 1) | другая величина, не смешивается с медианой разности |
| 1, 8, Г | 48,4% (45,9–50,8%) в пункте «Почти весь выигрыш…»; 45,9–50,8% в «Почему так» | `use.rule.vs_R.share`, `.share_ci`; `use.rule.random_draws` (100) | против R; доля зависит и от розыгрышей набора R — без нового числа |
| 8 | правка `edits.more_often` остаётся; `vs_R_not_more_often` сработала | `use.rule.edit`; `use.rule.edit_R`; `use.rule.vs_R.words` («примерно в половине случаев») | до 766f891 правка не срабатывала, и совет держался на решении участника 01.10 (журнал `docs/landing_spec.md` §4.3, порция 6d) |
| 1, 8, Г | исправление ошибки кода: табл. «до / после» — 810 и 746 из 1542; 11 и 75 равенств; 52,5% (50,1–55,0%) и 48,4% (45,9–50,8%); «чаще, чем нет» и «примерно в половине случаев» | «после» — `use.rule.vs_R.works`, `.ties`, `.share`, `.share_ci`, `.words`; «до» — `yaml:usefulness.rule_share.tie_tol` (комментарий «Было (точное «<»): B ближе R в 810 из 1542, равенств 11») и `git:766f891` (сообщение); 52,5% и интервал — прежняя строка этого реестра (`git show 766f891^:report/numbers.md`: `use.rule.vs_R.share` 0,5253, `.share_ci` 0,5007; 0,5497) | блок «Исправление ошибки кода после внутренней проверки» |
| 8 | 64 из 75 равенств; до $1{,}1 \cdot 10^{-16}$ | `git:766f891` (сообщение); арифметика: 810 − 746 = 64, 75 − 11 = 64 | сколько равенств засчитывались победой B; размер расхождения — по сообщению коммита, независимо не проверено |
| 8 (ε), Г (ε и 2,5·10⁻⁸) | $\varepsilon = 10^{-12}$; самая малая настоящая разность $2{,}5 \cdot 10^{-8}$ | `use.rule.tie_tol` (1e-12), `yaml:usefulness.rule_share.tie_tol`; команда: `rule_by_mo.csv`, наименьшая \|err_abs_B − err_abs_D\| > 1e-12 → 2,47e-8 (C — 9,7e-6, R — 3,0e-6) | допуск равенства |
| 1, 8, 9, Г | `766f891`; 02.10.2026 | `git:766f891` (2026-10-02 02:03:14 +0300) | коммит исправления |
| 1, 8, Г | по типам: 60,8% (55,9–65,1%; 411); 59,0% (55,2–62,9%; 693); 50,3% (44,4–56,8%; 344); 53,2% (44,6–61,3%; 94); слова | `use.rule.by_type` [type = 2, 1, 3, 4: share, share_ci, n, words] | без сравнения между типами |
| Г | Йошкар-Ола, Республика Марий Эл, тип 4 «Крупные города, меньше продуктов» | `use.example.name`, `.region`, `.type`, `.type_name` | пример по правилу `median_paired_diff` |
| Г | 21,7%; 19,9%; 19,2% | `use.example.own_change` (0,2172), `.median_B_change` (0,1986), `.median_D_change` (0,1922) | изменение оборота, exp − 1 |
| Г | 10 соседей и км: 17, 31, 34, 45, 60, 64, 64, 68, 75, 82 | `use.example.members_B` [name, km: 16,9; 30,6; 34,1; 44,7; 60,0; 64,4; 64,5; 68,2; 74,9; 82,0] | |
| Г | 10 похожих; от 232 до 5863 км | `use.example.members_D` [name; km: наименьший 232,4 — Кстовский, наибольший 5863,4 — Находкинский] | |
| Г | 0,015; 0,021; 0,005 | `use.example.err_B` (0,0154), `.err_D` (0,0207), `.d` (−0,0053) | ошибки примера |
| Г | 23-й и 28-й процентили | `use.example.pct_B` (0,2283), `.pct_D` (0,2776) | доля МО с ошибкой не больше, чем у примера |
| 8, Г | 0,058 против 0,033; 67-й и 42-й процентили | `use.example.stage5_example.err_B` (0,0583), `.err_D` (0,0330), `.pct_B` (0,6744), `.pct_D` (0,4163) | Благовещенский район, цель без поправки |
| Г | Йошкар-Ола по цели относительно региона: 0,013 против 0,015 | `csv:outputs/interpret/t7_errors.csv`, territory_id 269: `err_D` 0,01251, `err_B` 0,01542 | пример по правилу, другая цель |
| Г | 3 варианта; 4 повтора; seed 142, 242, 342, 442 | `yaml:usefulness.type_flag.variants`, `.seeds`; `yaml:interpret.robustness.variants`, `yaml:interpret.robustness.seeds` (42 — основной расчёт) | |
| Г | флаг «проверен в N из 3» — ни у кого; Москва и Петербург | `use.type_flag.partial_check` (0); `csv:outputs/usefulness/mo_flags.csv` [territory_id 90077, 90078: variant_runs = 2, variant_same = 1] | |
| 1, 8, 9, В, Г | 578 из 1776 (32,5%); 1198 | `use.type_flag.stable`, `use.type_flag.n`, `use.type_flag.stable_share` (0,3255), `use.type_flag.depends` | флаг |
| 8, 9, В, Г (часть чисел — только прил. Г) | табл.: 100%; 99,8% (1773 из 1776); 57,3% (1016 из 1774); 46,9% (833 из 1776) | `use.type_flag.seeds_only_share` (1,0); `use.type_flag.per_variant` [no_level 0,9983 × 1776; nodes_separate 0,5727 × 1774; graph_basket_cos 0,4690 × 1776]; счёт — `csv:outputs/interpret/node_r1.csv` [kind = variant; same; узлы с типом] | разбивка |
| 8, В, Г | 71,5% (1270 из 1776) | `use.type_flag.majority`, `use.type_flag.majority_share` (0,7151) | мягкое правило |
| Г (часть чисел — только прил. Г) | 54,1% (1091 из 2016); 242 района | `csv:outputs/interpret/node_r1.csv` [variant = nodes_separate: 2016 строк, same 1091]; 242 = 2016 − 1774: районы Москвы (144) и Петербурга (98), которых нет в основном расчёте (`data/processed/territories.parquet`, region_name) | второй знаменатель раздела 9 (сайт с порции 6b — 57,3%) |
| 8, Г | по типам: 62,6% (296 из 473); 7,8% (63 из 806); 30,5% (120 из 394); 96,1% (99 из 103) | `use.type_flag.by_type` [type = 2, 1, 3, 4: stable, n, stable_share] | |

## Раздел 8 и приложение Г, «Зависит ли польза от типа МО» (проверка `usefulness.by_type_test` и разведка `usefulness.size_posthoc`)

| Раздел | Число в тексте | Источник | Что это |
|---|---|---|---|
| 1, 8, В | `f744563`; `183d683`; 01.10.2026; `8f73a3a` | `git:f744563` (2026-10-01 22:38:02 +0300), `git:183d683` (23:17:47), `git:8f73a3a` (2026-10-02 01:09:51) | правила до расчёта, код вслепую, разведка после вскрытия |
| 8 | 59–61% у типов 1 и 2 (розница) | `use.rule.by_type` [type 1 — 0,5902; type 2 — 0,6083] | как в «Чем полезно» |
| 8, Г | 58 групп; 2000 выборок; 2000 перестановок; p < 0,05; 8 прогонов (3 варианта, 4 повтора) | `bt.n_groups`, `bt.design.bootstrap.n`, `bt.design.permutation.n`, `bt.design.alpha`, `bt.design.run_order` | правило вердикта — `yaml:usefulness.by_type_test.rule` |
| 8 | 51,3%; 56,6%; −5,3 п. п. (от −8,4 до −1,9); p = 0,002 | `bt.main.share_hi` (0,5131), `bt.main.share_lo` (0,5657), `bt.main.delta` (−0,05261), `bt.main.delta_ci_adj` (−0,08391; −0,01856), `bt.main.p_less` (0,0020) | интервал с поправкой центра, как в `bt.main_text.fields.ci` |
| Г | −6,1 п. п. (от −9,4 до −2,3) | `bt.main.delta_status` (−0,06060), `bt.main.delta_status_ci_adj` (−0,09427; −0,02341) | внутри статуса МО |
| Г | −0,4 п. п. (от −4,2 до 2,8) | `bt.main.delta_go` (−0,00413), `bt.main.delta_go_ci` (−0,04225; 0,02843) | деление «городской округ / остальные» |
| Г | −5,7; −5,2; −5,8 | `bt.runs.variant:graph_basket_cos.delta` (−0,05655), `bt.runs.variant:no_level.delta` (−0,05219), `bt.runs.variant:nodes_separate.delta` (−0,05833) | варианты расчёта типов |
| Г | повторы seed — те же типы и числа; «по сути четыре» | `bt.runs.seed:142…442` — все поля равны `bt.runs.main`; `yaml:usefulness.by_type_test.power.lowest_verdict` («повторы seed дают те же метки») | 1 основной + 3 варианта |
| Г | 4,6 п. п.; 2,80; около 80% | `bt.main.mde` (0,04562); `yaml:usefulness.by_type_test.statistic.mde`; 80% — `yaml:usefulness.by_type_test.words.not` («с вероятностью около 80%») | порог обнаружения |
| Г | общепит −7,0 (от −12,8 до −0,3; 828); доход −2,0 (от −7,1 до 3,6; 1231); отгрузка −6,8 (от −12,2 до −1,4; 1541) | `bt.main.per_target.<цель>.delta`, `.delta_ci`, `.n`; тексты — `bt.main.descriptions` | описание, без вывода по цели |
| Г | розница −8,8 п. п. (от −14,9 до −2,3) | `bt.main.per_target.retail.delta` (−0,08779), `.delta_ci` (−0,14925; −0,02297) | виденная цель |
| 1, 8, Г | табл.: границы 11,7; 18,1; 28,1; 53,5 тыс. | `size.quintile_bounds` (11 725; 18 115,3; 28 096,2; 53 546,3) | квинтили по 1542 МО |
| 8, Г | табл.: 309 / 308 / 308 / 308 / 309 МО | `size.quintiles[*].n_base` | |
| 8, Г | табл.: 5,5%; 7,8%; 14,6%; 38,0%; 76,1% | `size.quintiles[*].share_types_hi` (0,0550; 0,0779; 0,1461; 0,3799; 0,7605) | доля МО типов 3–4 в квинтиле |
| 1, 8, Г | табл.: 58,2% (54,6–62,3%); 56,8% (53,4–60,0%); 55,5% (51,3–59,1%); 54,9% (50,6–59,4%); 50,7% (46,0–54,2%) | `size.quintiles[*].share`, `.share_ci` | среднее по трём показателям; описание |
| 1, 8, Г | табл., колонка с общими членами D: 53,3–63,4%; 52,3–61,7%; 50,4–60,9%; 50,3–60,2%; 45,5–55,9% | `size.quintiles[*].share_ci_shared_centered` (0,5329; 0,6344 / 0,5234; 0,6171 / 0,5044; 0,6086 / 0,5030; 0,6021 / 0,4550; 0,5589) | интервал, сдвинутый к точечной доле |
| 1, 8, Г | табл., строка «до 53,5 вместе»: 1233; 16,5%; 56,3% (54,2–58,4%); 53,5–59,3% | `size.lower_four.n` (1233), `.share`, `.share_ci`, `.share_ci_shared_centered`; 16,5% — арифметика `size.crosstab_types_large`: types_hi_rest / (types_hi_rest + types_lo_rest) = 203 / 1233 = 0,1646 | четыре нижних квинтиля вместе |
| 1, 8, Г | сдвиг вверх: на 2,7 п. п. (59,0% против 56,3%) | `size.lower_four.shift_shared` (0,0269), `.mean_boot_shared` (0,5898), `.share` (0,5629); `yaml:usefulness.size_posthoc.shared_members_bootstrap.shares` (комментарий: сдвиг увидели 02.10, поправка добавлена после) | |
| Г | без сдвига — 56,2–62,0% | `size.lower_four.share_ci_shared` (0,5620; 0,6199) | ограничения подраздела |
| 1, 8, Г | 50,7% против 56,3%; −5,6 п. п. (от −10,4 до −1,8); p 0,0015 | `size.large_vs_rest.share_hi` (0,5066), `.share_lo` (0,5629), `.delta` (−0,05628), `.delta_ci` (−0,10369; −0,01795), `.permutation.p_less` (0,0015) | крупнейшие против остальных |
| 1, Г | тип при равном размере: −3,2 п. п. (от −7,2 до +1,6); «столько же даёт перестановка»; p 0,55 | `size.type_given_size.delta` (−0,03209), `.delta_ci` (−0,07235; 0,01620), `.permutation.null_mean` (−0,03443), `.permutation.p_less` (0,5522) | внутри квинтилей |
| Г | размер при равном типе: −3,6 п. п.; p 0,045; от −9,2 до +0,9 | `size.size_given_type.delta` (−0,03575), `.permutation.p_less` (0,0450), `.delta_ci` (−0,09245; 0,00901) | внутри групп типов |
| Г | с общими членами: θ от −10,1 до −0,7; крупные от −10,4 до +0,1 | `size.shared_members_bootstrap.types_main.delta_ci` (−0,10108; −0,00696), `.large_vs_rest.delta_ci` (−0,10431; 0,00105) | бутстрап с пересыпкой членов D |
| Г | 675 МО; типы −3,6 (от −7,6 до +0,8); крупные −4,9 (от −9,2 до −0,8) | `size.all_three_targets.n`, `.types.delta` (−0,03559), `.types.delta_ci` (−0,07570; 0,00849), `.large_vs_rest.delta` (−0,04930), `.large_vs_rest.delta_ci` (−0,09191; −0,00806) | МО со всеми тремя показателями |
| 1, 8, Г | «Как это меняет совет»: 56,3%, 54,2–58,4%, 53,5–59,3%; 50,7%, 46,0–54,2%, 45,5–55,9% | `size.lower_four.*`, `size.quintiles[4].*` (как выше) | до 02.10 здесь было «нижняя граница выше 50% в каждом из четырёх нижних квинтилей» — снято (решение главной сессии, вариант «б») |
| 8, 9, Г | правка `edits.confirmed.status_adds` не применяется | `bt.main_text.edit_key`, `bt.main_text.edit`; решение участника 02.10.2026 | отступление после вскрытия; запись для лендинга — `docs/landing_spec.md` §4 (делает expert-viz; на 02.10 01:20 записи ещё нет) |
| 8, Г | 438 МО типов 3–4; 309 крупных; общих 235 | `size.crosstab_types_large` (types_hi_large 235 + types_hi_rest 203 = 438; types_hi_large 235 + types_lo_large 74 = 309) | кого касалась бы правка по типу и кого — совет по размеру |
| 9, Г | верхние 28,4% МО в черновике | `yaml:usefulness.size_posthoc.seen_before[0]` | порог не независим от увиденного |
| 8, Г | 5 (min_set); 2000 (бутстрап с общими членами) | `yaml:usefulness.by_type_test.universe.min_set`; `yaml:usefulness.size_posthoc.shared_members_bootstrap.n` | параметры |

## Раздел 9. Ограничения

Повторяют числа разделов 2–7, кроме:

| Раздел | Число в тексте | Источник | Что это |
|---|---|---|---|
| в report.md нет — разделы 2–8 / `docs/*.md` | 5 регионах | `e1.n_gap_regions_2024` | нет зарплаты за 2024 год |
| 9 | 0,007 | `icvi.final_sw` | силуэт итога |
| в report.md нет — разделы 2–8 / `docs/*.md` | 143 | `cl.hybrid_zavi_min` | наименьшая z AVI гибрида |
| 5 (4 из 5 и 1; 3 из 5 и 2 из 5 — `docs/clustering.md`, таблица 6б) | 4 из 5, 1; 3 из 5, 2 из 5 | `cl.sf_prereg_all`, `cl.sf_tolerance_all` | выбор среди всех семейств по seed |
| 9 (29.09) | 5 → 20; 29.09 | `yaml:clustering.synthetic.repeats` (комментарий «было 5»); `docs/clustering.md`, раздел 10 | синтетика не предрегистрирована |
| 9 (ε² 0,082); AMI 0,31 и «не выше 0,10» — 1, 8; 0,024, 77% и 52% — в report.md нет, `docs/interpretation.md` | AMI 0,31; не выше 0,10; ε² 0,082; ε² 0,024; 77%; 52% | как в разделе 8 | ограничения смысла типов |
| 8, 9 (часть чисел — только разделы 2–8 / `docs/*.md`) | 1705; 1241 из 1774 | `int.t1.per.retail.n_a`, `int.t1.per.catering.n_a`, `int.scope.n_territorial` | покрытие оборотов Росстата |
| в report.md нет — разделы 2–8 / `docs/*.md` | 1115; 1569 | `int.t6.n_nodes`, `int.t7.n_known` | оба года оборота общепита; оба года розницы |
| 8 (46,9%, 99,8%, «все»); 54,1% (1091 из 2016) — в report.md нет, `outputs/interpret/node_r1.csv` | 46,9% (833 из 1776); 54,1% (1091 из 2016); 99,8%; все; ρ от 0,46 до 0,55 | `csv:outputs/interpret/node_r1.csv` [same: graph_basket_cos 833/1776, nodes_separate 1091/2016, no_level 1773/1776 = 0,998, seed:142…442 — 1776/1776]; `csv:outputs/interpret/t1_runs.csv` (как в разделе 1) | тип отдельного МО в прогонах R1; порядок по рознице |
| 8, 9 | 57,3% среди 1774 | `use.type_flag.per_variant` [nodes_separate] | как в разделе 8 |
| 1, 8, 9, В, Г | 32,5% (578 из 1776); у типа 1 — 7,8% | `use.type_flag.stable_share`, `use.type_flag.stable`; `use.type_flag.by_type` [type = 1] | флаг устойчивости |
| 1, 8 (в разделе 9 с 02.10 — без чисел) | 57,2%; около 0,5–0,6 п. п.; примерно на 13%; 48,4% (было 52,5%) | `use.rule.vs_D.share`; `use.rule.vs_R.share`; как в разделе 1 | совет «сверяйте изменения со своим регионом» (до 02.10 здесь было и «у типов 3 и 4 — примерно в половине случаев») |
| 1, 8, Г (в разделе 9 с 02.10 — без чисел) | 50,7% (46,0–54,2%; 45,5–55,9%); до +0,1 п. п.; −3,2 п. п. (от −7,2 до +1,6) | `size.quintiles[4].share`, `.share_ci`, `.share_ci_shared_centered`; `size.shared_members_bootstrap.large_vs_rest.delta_ci[1]` (0,00105); `size.type_given_size.delta`, `.delta_ci` | совет самым крупным МО |

## Раздел 10 и приложение В. Воспроизводимость

| Раздел | Число в тексте | Источник | Что это |
|---|---|---|---|
| 10 | 3.12; 3.11–3.13 | `pyproject.toml [requires-python = ">=3.11,<3.14"]`; `uv run --frozen python --version` → 3.12.14 | версия Python |
| 10, Г | `seed: 42` | `yaml:seed` | |
| 10, В | 263 МБ; 38 МБ; около 55 МБ | замер 02.10.2026: `du -sh data/processed` → 38M; `du -scb` по папкам `outputs/*` без `outputs/site*` → 55 464 725 байт (README — «около 55 МБ»); `data/raw` — замер 28.09 | место на диске; папки сборок лендинга `outputs/site_*` — не выходы этапов |
| В | 68 с | замер: `panel` во временной папке | |
| В | 42 с | `git:aacfaf6` (README той версии: «весь этап `eda` — 42 секунды», замер 26.09.2026) | раньше источником был нынешний README — круговая ссылка |
| В | 16 с | замер: `features` во временной папке | |
| В | 431 с (около 7 мин) | замер: `network` во временной папке | |
| 10, В, README | 69 мин; 4 мин; 13 мин; 6 процессов | `cl.time_total_min`, `cl.time_bootstrap_min`, `cl.time_variants_min`, `cl.workers` | прогон 30.09 с seed 42; однократный замер (с 03.10.2026 подписан так) |
| В | 104 с | замер: `evaluate` во временной папке | |
| В | 33 с | `outputs/dynamics/facts.json`, ключ `seconds` (33,3) | |
| 10, В | 16 ГБ | `yaml:clustering.impl.workers` (комментарий) | |
| 10, В | sha256 трёх файлов | `yaml:sources.hackathon.sha256`, `yaml:sources.borders.sha256`, `yaml:sources.ndfl.sha256` | |
| В | 14 файлов | `yaml:sources.bdmo.members` | число файлов БД ПМО |
| 8, 10, В, Г | 30.09.2026; seed 43 и 44; 1776 узлах; ARI 1,00 | сравнение `data/processed/cluster_final.parquet` с `cluster_final.parquet` прогонов с `seed: 43` и `seed: 44` во временных папках: `sklearn.metrics.adjusted_rand_score` = 1.0, доля совпавших меток 1.0 на 1776 общих узлах | проверено при подготовке отчёта 30.09 |
| В | спектральная с K = 4 при seed 43 | `cl.all_winner` в `report_facts.json` прогона с `seed: 43` | |
| в report.md нет — заменено строкой «7886 с» (прогон 02.10.2026); прил. В | 7241 с, около 2 ч; 6 процессов; 101 мин | `int.seconds` (7240,5), `int.workers` (6), `int.timing.robustness` (6057,8 с) | время этапа `interpret`, полный перерасчёт 30.09.2026 (утренний прогон — 7374 с) |
| В | 41 с по журналу этапа; 28 с; около 7 с; 1,2 МБ | замер 02.10.2026: `usefulness` с временным конфигом (выходы во временную папку, копия `outputs/interpret`): журнал 01:14:34,3 → 01:15:15,4; проверка по типам — 01:14:41,9 → 01:15:09,5; разведка по размеру — `size.seconds`: 7,0 в репозитории (после повтора 02.10.2026 21:36, строка «`usefulness` после пересчёта» ниже), 5,9 и 5,8 — прогон и повтор 02.10.2026 ночью, 6,3 — прогон с нуля 03.10.2026; до 03.10.2026 в тексте было «6 с»; `du -sh` выходов → 1,2M | время и место этапа; полное время процесса с запуском `uv` не замерено |
| 10, В | побайтно (sha256) | повторный прогон 01.10.2026: sha256 `facts.json` 892c5209…, `rule_by_mo.csv` 660936db…, `mo_flags.csv` 1a56bb12… совпали с файлами `outputs/usefulness/` | воспроизведение этапа |
| В | 02.10.2026: семь файлов побайтно; `size_check.json` — только `seconds` | повтор `usefulness` 02.10.2026 во временную папку: `sha256sum -c` — OK для `by_type.json` (24924f93…), `by_type_by_mo.csv` (c7bfffbc…), `by_type_runs.csv` (2d2cb3d1…), `facts.json` (892c5209…), `mo_flags.csv` (1a56bb12…), `rule_by_mo.csv` (660936db…), `size_by_mo.csv` (93828183…); `size_check.json` (eb5f5d7f…) отличается только полем `seconds` (5,9 → 5,8; поэлементное сравнение JSON) | воспроизведение проверки по типам и разведки по размеру |
| 10, В | 7886 с (2 ч 11 мин); 110 мин — прогоны устойчивости | `int.seconds` (7885,8), `int.timing.robustness` (6627,5 с = 110,5 мин); журнал прогона (вне репозитория): «interpret: готово за 7887 с» | время `interpret`, полный пересчёт 02.10.2026 (заменяет 7241 с и 101 мин прогона 30.09) |
| 10, В | все 24 CSV и `facts.json` совпали побайтно, кроме полей времени | команда 02.10.2026: `cmp` 24 файлов `outputs/interpret/*.csv` и `naming_test.json` с копией выходов прогона 30.09, снятой до пересчёта, — совпали все; поэлементное сравнение `facts.json` — отличаются только `seconds` (7240,5 → 7885,8) и 7 полей `timing` | пересчёт `interpret` 02.10.2026 |
| 10, В | `usefulness` после пересчёта: пять CSV и `by_type.json` побайтно; `facts.json` — только отпечаток входа; `size_check.json` — только время | прогон 02.10.2026 21:36 (время записи файлов); сравнение с копией до прогона: `cmp` и поэлементно JSON — `inputs_sha256.facts` (560b3e79… → cc4f54aa…), `seconds` (6,3 → 7,0), `shared_members_bootstrap.seconds` (2,0 → 2,2); `ls outputs/usefulness/*.csv` → 5 | `seconds` 7,0 — источник «около 7 с» в таблице времени приложения В; в прогоне с нуля 03.10.2026 — 6,3 |
| 10, В, README | около 3,5 ч (сумма однократных замеров); «от 3 до 3,5 ч» — между прогоном 03.10.2026 и этой суммой | арифметика по таблице времени приложения В: 68 + 42 + 16 + 431 + 4170 (`cluster`, `outputs/cluster/timing.csv`, `total_before_report`) + 104 + 33 + 7886 + 41 = 12 791 с ≈ 3 ч 33 мин | полный расчёт без скачивания |
| 10, В, README | 03.10.2026; `5a2ea8b`; код 0; 3 ч 1 мин 35 с (3 ч 2 мин); Windows 10 | журнал прогона с нуля (вне репозитория, `run.log` в рабочей папке прогона): первая строка `Sat Oct  3 13:11:07`, последние — `EXIT=0` и `Sat Oct  3 16:12:42`; чистая копия — `git clone` коммита `5a2ea8b`, `uv sync --frozen`; копия конфига отличается от `configs/default.yaml` одной строкой (`paths.raw`) | прогон с нуля в чистой копии; этап `data` не запускался |
| В, README | 57 с; 52 с; 9 с; 6 мин 53 с; 47 мин 23 с; 1 мин 20 с; 45 с; 2 ч 2 мин 15 с; 41 с; 12 с | журнал прогона 03.10.2026: разность времени строк «Этап <имя>:» соседних этапов — panel 13:11:15, eda 13:12:12, features 13:13:04, network 13:13:13, cluster 13:20:06, evaluate 14:07:29, dynamics 14:08:49, interpret 14:09:34, usefulness 16:11:49, site 16:12:30, конец 16:12:42 | колонка «Прогон с нуля 03.10.2026» таблицы времени |
| 10, README | `cluster` 47 мин 23 с; `interpret` 2 ч 2 мин | журнал: «cluster: выходы … всего 2843 с»; `seconds` в `outputs/interpret/facts.json` чистой копии — 7333,8 с (2 ч 2 мин 14 с; по строкам журнала — 2 ч 2 мин 15 с) | |
| В | бутстрап 3 мин; пересчёт на других входах 9 мин | `cl.time_bootstrap_min` (3,04), `cl.time_variants_min` (8,96) в `outputs/cluster/report_facts.json` чистой копии (`cl.time_total_min` — 47,1) | `cluster` 03.10.2026 |
| 10, В, README | все таблицы, json, parquet и рисунки побайтно, кроме замеров времени и производных отпечатков | сравнение sha256 всех файлов `outputs/`, `data/interim/`, `data/processed/`, `docs/img/**` и сайта чистой копии с основным репозиторием (сайт — с `outputs/site_5a`); при расхождении — поэлементный разбор json, csv, parquet с допуском 1e-9 (скрипт и итог — вне репозитория). Отличаются только `seconds`, `timing`, `cl.time_*`, `qc.freshness`, `inputs_sha256`, `facts_sha256` и строка коммита сборки сайта; `docs/clustering.md` — три числа времени | прогон с нуля 03.10.2026 |

## Приложение А

| Раздел | Число в тексте | Источник | Что это |
|---|---|---|---|
| А | n = 1776 | `feat.n_nodes` | |
| А | 11 признаков | `cl.n_x` | |
| А | k = 10 | `yaml:network.sparsify.k` | |

## Приложение Б

| Раздел | Число в тексте | Источник | Что это |
|---|---|---|---|
| Б | 25.09.2026 | `ls -l --time-style=full-iso data/raw/*/`: `hackathonlicence.zip` — 2026-09-25 23:46 +0300, `t_dict_municipal.rar` — 23:46, файлы `bdmo/` и `ndfl/` — 23:47 | дата скачивания; манифеста загрузки нет, загрузчик пишет файл через `.part` и переименовывает (`src/munnet/data.py:66`) |
| Б | тома, номера, страницы, DOI | проверка библиографии по Crossref и страницам издателей 28.09.2026 (заметка участника `literature-verified`) | библиография |
