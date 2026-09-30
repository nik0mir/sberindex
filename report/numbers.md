# Реестр чисел методологического отчёта

Откуда взято каждое число `report/report.md`. Нужен для сверки (check-facts): число текста должно совпадать
с источником.

Как читать колонку «Источник»:

- ключ вида `cl.final_k` — поле `text` этого ключа в файле фактов этапа: `e1.`…`e5.` и `syn.` —
  `outputs/eda/facts.json`; `feat.` — `outputs/features/report_facts.json`; `net.` —
  `outputs/network/report_facts.json`; `cl.` — `outputs/cluster/report_facts.json`; `icvi.` —
  `outputs/evaluate/report_facts.json`; `dyn.` — `outputs/dynamics/report_facts.json`;
- `controls:<ключ>` — поле `actual` в `outputs/panel/controls.json`;
- `yaml:<путь>` — значение в `configs/default.yaml`;
- `csv:<файл> [строка; колонка]` — значение в CSV;
- `git:<коммит>` — `git show <коммит>`;
- `замер` — измерение 28.09.2026 при подготовке отчёта (команда — в разделе 10 отчёта).

Числа, которые повторяют уже внесённое (например, 1776 в разных разделах), внесены один раз — при первом
появлении. Номера разделов, рисунков, таблиц и формул статей не вносятся; годы в ссылках на литературу —
по приложению Б отчёта (проверены по DOI).

## Шапка и раздел 2. Данные

| Раздел | Число в тексте | Источник | Что это |
|---|---|---|---|
| шапка | 10 методов | `cl.n_methods` | методов кластеризации, включая базовую линию |
| шапка | 13 окон | `dyn.n_windows` | скользящих окон |
| шапка | 12 месяцев | `yaml:dynamics.window_months` | длина окна |
| 2 | 2190 | `e1.n_mo` | МО панели |
| 2 | 77 | `e1.n_regions` | регионов с тратами |
| 2 | 1776 | `feat.n_nodes` | узлов сети |
| 2 | 2023-01…2024-12 | `yaml:period` | период данных |
| 2 | 8 регионов | `e1.n_absent_regions` | регионов без данных; список — `e1.absent_regions` |
| 2 | 153 | `e1.n_missing_outside` | МО без трат в остальных регионах |
| 2 | 303 126 | `controls:consumption_rows` | строк исходной таблицы трат |
| 2 | 27,1% | `e4.share_median_other` | медиана доли «Прочего», 2024 год |
| 2 | 24 месяца | `yaml:features.min_months` | полный ряд |
| 2 | 2016 | `e1.n_full` | МО с полным рядом |
| 2 | 92,1% | `e1.full_share` | их доля |
| 2 | 17 698 | `e1.pop_median_incomplete` | медиана населения неполных |
| 2 | 23 841 | `e1.pop_median_full` | медиана населения полных |
| 2 | 4,5·10⁻⁸ | `e1.missing_mw_p` | p теста Манна — Уитни |
| 2 | 247 | `e1.n_inner_city` | внутригородских территорий |
| 2 | 11,3% | `e1.inner_city_node_share` | их доля среди МО |
| 2 | 15,5% | `e1.pop_share_inner_city` | их доля населения |
| 2 | 1945 | `feat.n_nodes_all` | узлов в режиме `collapse` |
| 2 | 169 | `feat.n_dropped` | узлов вне сети |
| 2 | +15,1% | `e4.growth_median` | медианный номинальный рост трат |
| 2 | 21 | `e1.controls_hard_n` | жёстких контрольных чисел; не сошлось — `e1.controls_hard_failed` = 0 |
| 2 | 32 | `e1.controls_soft_n` | мягких; вне допуска — `e1.controls_soft_warnings` = 0 |
| 2 | 24 (`features.min_months: 24`) | `yaml:features.min_months` | параметр |

## Раздел 3. Признаки узлов

| Раздел | Число в тексте | Источник | Что это |
|---|---|---|---|
| 3 | 1986 (Aitchison) | литература | год издания |
| 3 | 0,29 | `feat.eta2_raw_min` | наименьший η² как есть |
| 3 | 0,62 | `feat.eta2_raw_max` | наибольший η² как есть |
| 3 | 0,04 | `feat.eta2_random` | η² шума |
| 3 | 0,00 | `feat.eta2_rel_max` | наибольший η² после поправки |
| 3 | 6 (рёбра) | `feat.n_edges` | признаков рёбер; прошли — `feat.n_edges_ok` |
| 3 | 11 (атрибуты) | `feat.n_attributes` | атрибутов; прошли — `feat.n_attributes_ok` |
| 3 | 2 (слой) | `feat.n_layer` | признаков слоя; прошли — `feat.n_layer_ok` |
| 3 | 2 (динамика) | `feat.n_dynamics` | признаков динамики |
| 3 | пять укрупнённых разделов ОКВЭД2 | `yaml:features.space.attributes` | пять признаков `emp_sh_*` |
| 3 | 0,5 | `feat.rel_min` | порог надёжности; `yaml:features.space.min_reliability` |
| 3 | 0,6 | `feat.eta2_max` | порог η² атрибута |
| 3 | 0,8 | `feat.rho_max` | порог \|ρ\|; `yaml:features.space.max_abs_rho` |
| 3 | 0,86–0,96 | `feat.basket_rel_range` | надёжность корзины |
| 3 | 0,25 | `feat.dec_peak_rel` | надёжность декабрьского пика |
| 3 | 0.001 | `yaml:features.clr_floor` | порог замены малых долей |
| 3 | 0,121 | `feat.tree_kappa_clean` | каппа дерева, очищенная корзина |
| 3 | 0,315 | `feat.tree_kappa` | каппа дерева, корзина относительно региона |
| 3 | 0,005 | `feat.ami_region` | AMI типов с регионом |
| 3 | 0,73 | `feat.edge_rho_max` | \|ρ\| уровня трат и доли общепита (`feat.edge_rho_pair`) |

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
| 4 | 3 месяца | `yaml:network.rules.rhythm_lag.max_lag` | наибольший лаг |
| 4 | 2 месяца | `yaml:network.rules.rhythm_dtw.window` | окно DTW |
| 4 | 0,15; 0,14; 0,08 | `net.reliability_basket_dist`, `net.reliability_basket_cos`, `net.reliability_rhythm_corr` | надёжность |
| 4 | 1 (дороги) | `net.reliability_geo_road` | 1,00 |
| 4 | 0,004; 0,004; 0,005; 0,003 | `net.reliability_chance_basket_dist`, `…_basket_cos`, `…_rhythm_corr`, `…_geo_road` | случайный уровень |
| 4 | 0,009; 0,009; 0,084; 1,000 | `net.geo_jaccard_basket_dist`, `…_basket_cos`, `…_rhythm_corr`, `…_geo_road` | доля рёбер, общих с дорогами |
| 4 | 3,1%; 3,0%; 25,7%; 76,7% | `net.within_region_basket_dist`, `…_basket_cos`, `…_rhythm_corr`, `…_geo_road` | рёбер внутри группы региона |
| 4 | 10 признаков | `yaml:network.attributes.columns` | признаков согласованности |
| 4 | 0,27; 0,25; 0,22; 0,07 | `net.attr_consistency_basket_dist`, `…_basket_cos`, `…_rhythm_corr`, `…_geo_road` | согласованность с местом |
| 4 | 0,1; 0,2; 2,4; 0,3 | `net.kocc_skew_basket_dist`, `…_basket_cos`, `…_rhythm_corr`, `…_geo_road` | асимметрия k-встречаемости |
| 4 | 25; 23; 91; 25 | `net.deg_max_basket_dist`, `…_basket_cos`, `…_rhythm_corr`, `…_geo_road` | наибольшая степень |
| 4 | 0,05; 0,03; 0,53; 0,87 | `net.assort_north_basket_dist`, `…_basket_cos`, `…_rhythm_corr`, `…_geo_road` | ассортативность по Северу |
| 4 | 10 сетей | `net.n_rules` | сетей в полной таблице |
| 4 | 24 порядка | `csv:outputs/network/selection_sets.csv [set = main; winners = basket_dist: 24]` | порядков основного набора |
| 4 | k = 5, 10, 15 и 20 | `csv:outputs/network/selection_by_k.csv [winner = basket_dist при всех k]` | чувствительность к k |
| 4 | три набора критериев | `yaml:network.selection` | `priority`, `sets.plan`, `sets.extended`; Борда — `selection_sets.csv [borda]` |
| 4 | `cc6f265` | `git:cc6f265` | запись PLAN.md 25.09.2026 |
| 4 | 20 из 20 | `net.boot_n_dist_better_reliability`, `net.n_boot` | парный бутстрап месяцев |
| 4 | 0,22 | `net.reliability_basket_dist_abs` | надёжность без вычета региона |
| 4 | 0,28 | `net.check_log_ndfl_rel_basket_dist` | ассортативность по доходу 5-НДФЛ |
| 4 | 0,011 | `net.check_log_ndfl_rel_resid_basket_dist` | то же сверх зарплаты |
| 4 | ±0,010 | `net.check_log_ndfl_rel_resid_null_sd_basket_dist` | случайный уровень |
| 4 | 13 окон | `net.win_rolling_n` | скользящих окон |
| 4 | 0,51 | `net.win12_noise_jaccard` | Жаккар двух сетей одного времени |
| 4 | 0,15 (сети 2023 и 2024 годов) | `net.win12_change_jaccard` | Жаккар сетей двух лет |
| 4 | 0,73 | `net.ami_region_geo_road` | AMI сообществ дорог с регионом |
| 4 | 0,32 | `net.pair_ari_basket_cos__basket_dist` | ARI сообществ косинуса и расстояния |
| 4 | `k_grid: [5, 10, 15, 20]` | `yaml:network.sparsify.k_grid` | параметр |

## Раздел 5. Методы кластеризации

Выходы этапа `cluster` — прогон 30.09.2026 с `seed: 42` (`outputs/cluster/report_facts.json`, `docs/clustering.md`).

| Раздел | Число в тексте | Источник | Что это |
|---|---|---|---|
| 5 | K = 3, …, 12 | `cl.k_min`, `cl.k_max` | сетка K |
| 5 | `2ee1363`, 28.09.2026 | `git:2ee1363` | коммит предрегистрации |
| 5 | 12 418 | `cl.n_edges` | рёбер G |
| 5 | 0,11 | `cl.weight_min` | наименьший вес ребра |
| 5 | одна компонента | `cl.n_components` | 1 |
| 5 | 11 атрибутов | `cl.n_x` | признаков X |
| 5 | 17 разбиений; 360; от 152 до 334 | `cl.rl_n_cands`, `cl.rl_min_inner`, `cl.rl_threshold`, `cl.rl_threshold_max` | предел разрешения модульности |
| 5 | ρ = ξ = 1 | `yaml:clustering.impl.shalileh_mirkin.rho`, `…xi` | веса KEFRiN |
| 5 | α = 0,5 | `cl.alpha`; `yaml:clustering.protocol.hybrid_alpha` | вес графа в гибриде |
| 5 | 2%; 36; 50%; 10% | `cl.min_share`, `cl.min_share_nodes`, `cl.max_share`, `cl.max_noise` | пороги допустимости |
| 5 | 200 разметок | `cl.icvi_perms` | случайный базис |
| 5 | 100 подвыборок; 80% | `cl.bootstrap`, `cl.subsample` | бутстрап устойчивости |
| 5 | глубина 3; 17 признаков; 5-кратная | `cl.tree_depth`, `cl.n_tree`, `cl.cv_folds` | дерево объяснимости |
| 5 | 0,02; 0 | `cl.tie_stability`, `cl.tie_interpretability`; `yaml:clustering.selection.tie` | допуски ничьей |
| 5 | 600 узлов; 4 группы; шести координатам | `cl.syn_n`, `cl.syn_k`, `yaml:clustering.synthetic.n_basket` | синтетика |
| 5 | от 0,3 до 0,9 | `cl.sbm_mix_min`, `cl.sbm_mix_max` | доля рёбер между группами в блочном графе |
| 5 | 45 ячеек; 20 наборов | `cl.syn_all_cells`, `cl.syn_repeats`; `yaml:clustering.synthetic.repeats` | |
| 5 | 95%; 0,05 | `cl.syn_ci`, `cl.syn_min_ari`; `yaml:clustering.synthetic.ci_level`, `…min_ari` | правило ячейки |
| 5 | 9; 30; 6 | `cl.syn_clear_cells`, `cl.syn_tie_cells`, `cl.syn_none_cells` | явный победитель, ничья, «нет» |
| 5 | 16 из 16; явно — 6 | `cl.syn_both_won`, `cl.syn_both_cells`, `cl.syn_both_clear` | сигнал в обоих источниках |
| 5 | во всех 16 — базовая линия | `docs/clustering.md`, раздел 4, второй пункт («16 из 16»); `cl.syn_both_hybrid_best` = 0 | |
| 5 | 0,91; 0,83; 0,44 | `cl.syn_strong_kmeans_joint`, `cl.syn_strong_hybrid`, `cl.syn_strong_shalileh_mirkin` | сильные сигналы |
| 5 | во всех 4 ячейках | `cl.syn_f0_four_tie`, `cl.syn_f0_n` | ничья на kNN без сигнала в X |
| 5 | одна ячейка; 0,3; 0,98 | `cl.syn_graph_only_clear_n`, `cl.syn_graph_only_clear_text`, `cl.sbm_low_leiden` | явная победа метода только по графу |
| 5 | 5 из 5; 0,43; 0,34; 0,17; в 1; в 3 | `cl.sbm_mid_hybrid_best`, `cl.sbm_mid_n`, `cl.sbm_mid_hybrid`, `cl.sbm_mid_spectral`, `cl.sbm_mid_leiden`, `cl.sbm_mid_hybrid_clear`, `cl.sbm_mid_tie_spectral` | блочный граф, доля 0,5 |
| 5 | 2,0; 0,03; 0,46; 0,39 | `cl.syn_fmax`, `cl.syn_g0_hybrid`, `cl.syn_g0_kmeans`, `cl.syn_g0_shalileh_mirkin` | шумовой граф kNN |
| 5 | 0,7; 0,04; 0,43; 0,40 | `cl.sbm7_hybrid`, `cl.sbm7_kmeans`, `cl.sbm7_shalileh_mirkin` | блочный граф, доля 0,7, сигнал в X 2,0 |
| 5 | во всех 9 ячейках; 9 из 9; 0,05 | `cl.syn_noisy_x_tie`, `cl.syn_noisy_live`, `cl.syn_min_ari` | шумовой граф (kNN без сигнала в графе и блочный граф с долей от 0,7), ячейки не «нет»: ничья K-means, гауссовой смеси и KEFRiN; то же — табл. 13 |
| 5 | 28.09; `512cc67` | `git log --format='%h %ad' --date=short -S sbm_mixing -- configs/default.yaml` → `512cc67 2026-09-28`; предрегистрация `2ee1363` синтетику прямо исключает (комментарий блока `clustering` в `git show 2ee1363 -- configs/default.yaml`) | синтетика целиком не предрегистрирована; то же — `docs/clustering.md`, раздел 10 |
| 5 | 29.09; было 5 наборов | `docs/clustering.md`, раздел 10, пункт «Синтетика изменена 29 сентября» | |
| 5 | 0,040; 0,149 | `cl.syn_none_best_max`, `cl.syn_live_best_min` | разрыв у порога 0,05 |
| 5 | 87; 29 | `cl.n_candidates`, `cl.n_feasible` | кандидатов, допустимых |
| 5 | 10 значений K | `cl.cands_shalileh_mirkin`, `cl.cands_kmeans_joint`; допустимых — `cl.feas_shalileh_mirkin` = 0, `cl.feas_kmeans_joint` = 0 | |
| 5 | 105 из 108 | `cl.n_small_market`, `cl.n_small` | мелкие типы по доступности рынков |
| 5 | 16 районов; 26 MAD | `cl.suburbs_size`, `cl.suburbs_scaled` | мелкий тип гибрида |
| 5 | 1,2%; ξ/ρ от 16 | `cl.kef_z1_minshare_k4`, `cl.kef_z_feasible_min_xi` | KEFRiN в варианте статьи |
| 5 | 51,5% | `cl.hybrid_k3_max_share` | крупнейший тип гибрида при K = 3 |
| 5 | K = 4; 806, 473, 394, 103 | `cl.final_k`, `cl.type1_size` … `cl.type4_size` | итог |
| 5 | 0,91; 0,90; 0,75 | `cl.final_stability`, `cl.final_jaccard_min`, `cl.jaccard_stable` | устойчивость итога |
| 5 | 10 seed; ARI 1,00 | `cl.seeds`, `cl.final_seed_ari` | разбиения гибрида K = 4 по seed метода |
| 5 | seed 42–46; 5 из 5 (оба правила) | `cl.seed_min`, `cl.seed_max`, `cl.n_seeds`, `cl.sf_prereg_eligible`, `cl.sf_tolerance_eligible` | итог по seed |
| 5 | фронт из трёх: Leiden K = 3, Louvain K = 6, гибрид | `cl.all_front`, `cl.all_front_n` | все семейства, seed 42 |
| 5 | +1 | `cl.all_top_score`, `cl.all_tied` | очки Копленда |
| 5 | 0,91 против 0,62 | `cl.final_stability`, `cl.leiden_stability` | равенство очков решает устойчивость |
| 5 | Борда на фронте: 7 и 7 (Leiden K = 3, гибрид), 10 (Louvain K = 6); 0,909 против 0,615 | `cl.borda_front`, `cl.borda_leiden`, `cl.borda_hybrid`, `cl.borda_n_tied` = 2, `cl.borda_outcome`, `cl.borda_consistent` = 1 | ничья, решает устойчивость; сумма мест пересчитана вручную по `csv:outputs/cluster/selection_levels_all.csv [level = 2; on_front = True]` |
| 5 | seed конфига 43 и 44: поровну у трёх (Leiden, спектральная, гибрид), выбран гибрид | `cl.borda_outcome` в `tests/fixtures/cluster_report_facts/seed43.json` (по 10; 0,912 / 0,902 / 0,628) и `seed44.json` (по 9; 0,910 / 0,897 / 0,623) | фикстуры — `report_facts.json` прогонов 43 и 44; сами прогоны вне репозитория |
| 5 | 4 из 5; спектральная K = 4 — 1 | `cl.sf_prereg_all` | все семейства, seed 42–46, предрегистрация |
| 5 | наименьший ARI 1,00 | `cl.sf_all_minari`, `cl.sf_all_minari_min` | разбиения победителей по seed |
| 5 | спектральная не на фронте при seed 42: качество в X 0,50 против 0,25; на G 0,50 и 0,50; устойчивость 0,909 и 0,901; объяснимость 0,894 и 0,905; допуск 0,02 | `csv:outputs/cluster/selection_levels_all.csv [level = 2; cand = hybrid_k04, spectral_k04; crit_quality_features, crit_quality_graph, crit_stability, crit_interpretability]`; `cl.tie_stability`, `cl.tie_interpretability` | гибрид доминирует спектральную только по качеству в X |
| 5 | 0,91 против 0,90 | `cl.final_stability`, `cl.spectral_stability` | |
| 5 | 7 из 8 | `cl.all_n_same`, `cl.all_n_checks` | проверки чувствительности, все семейства |
| 5 | один шаг — спектральная K = 4 | `cl.all_changed`, `cl.all_joint_winner`, `cl.all_joint_k` | |
| 5 | 24 порядка; 12; 6; 6 | `cl.orders_total`, `cl.lex_l2_text` | порядки при выборе метода |
| 5 | Leiden K = 3; 0,927; 0,884; 0,606; 0,590 | `cl.raw_l3_winner_graph`, `cl.raw_l3_winner_graph_k`, `cl.avi_adj_leiden`, `cl.avi_adj_hybrid`, `cl.mq_raw_leiden`, `cl.mq_raw_hybrid` | метрики графа, сравнимые между K |
| 5 | 55%; спектральная K = 3; 0,008; 0,02; K = 4 / K = 3 | `cl.grid55_all_winner`, `cl.grid55_all_k`, `cl.hybrid_k3_k4_stab_diff`, `cl.tie_stability`, `cl.grid55_strict_hybrid_k`, `cl.grid55_tol_hybrid_k` | порог крупнейшего типа |
| 5 | 1%–3% не меняет | `cl.grid_minshare_effect` = 0; `csv:outputs/cluster/threshold_grid.csv` | порог наименьшего типа |
| 5 | K = 3 не участвовал во внешней проверке | `csv:outputs/cluster/validation.csv [cand: hybrid_k03 нет]` | |
| 5 | iK-means: 16 по X, 20 по корзине ⊕ X; порог 2 узла; граница сетки 12 | `cl.ik_x`, `cl.ik_joint`; `yaml:clustering.impl.ik_means_min_size`; `yaml:clustering.n_clusters` | ориентир K, в выбор не входит |
| 5 | 5 базисам; 0,03; 0,04 | `yaml:clustering.impl.seed_check`; `cl.tol_qf`, `cl.tol_qg` | допуск по шуму базиса |
| 5 | таблица «правило × уровень» | `cl.final_label`, `cl.final_k`, `cl.all_winner`, `cl.tol_eligible_winner`, `cl.tol_all_winner`, `cl.sf_prereg_eligible`, `cl.sf_tolerance_eligible`, `cl.sf_prereg_all`, `cl.sf_tolerance_all` | |
| 5 | 0,031–0,033; 0,030–0,039 | `cl.tol_jk_qf_min`, `cl.tol_jk_qf_max`, `cl.tol_jk_qg_min`, `cl.tol_jk_qg_max` | допуск без одного базиса |
| 5 | 5 из 5 | `cl.tol_jk_winners`, `cl.tol_jk_n` | |
| 5 | 25 сочетаний; 25 из 25 | `cl.tol_grid_n`, `cl.tol_grid_winners` | сетка допусков |
| 5 | 0,032, 0,031 и 0,035; 0,043, 0,042 и 0,037 | поле `value` ключей `cl.tol_qf`, `cl.tol_qg` в `report_facts.json` трёх прогонов 30.09: seed 42 — репозиторий (0,03194; 0,04256), seed 43 и 44 — временные папки (0,03150; 0,04188 и 0,03529; 0,03694) | наблюдение по трём прогонам, одной командой не воспроизводится; те же числа — в `docs/clustering.md`, раздел 10 (опечатка 0,032 для seed 43 исправлена 30.09 в шаблоне `src/munnet/clustering/templates/clustering.md`) |
| 5 | seed 44 с допуском: спектральная в прогонах 42 и 43, гибрид в 44 | `csv:outputs/cluster/seed_runs.csv [check = main; scope = all; rule = tolerance; seed = 44] = spectral_k04`; то же в `seed_runs.csv` прогона 43; в прогоне 44 — `hybrid_k04` | прогоны 43 и 44 — вне репозитория |
| 5 | 23 из 276 (seed 44–46); 28 из 368 (seed 44–47); 0 из 368; 0 из 368, 276 и 368 | сравнение `seed_runs.csv` трёх прогонов 30.09 по ключу (`check`, `scope`, `rule`, `seed`, `kind`), число строк с разным `winner`: `rule = tolerance` — пары прогонов 42–44, 43–44, 42–43; `rule = prereg` — 42–43, 42–44, 43–44. Все расхождения — на уровне всех семейств (`scope` = `all`, `all_level2` и проверка `all_eligible`) | наблюдение по трём прогонам, одной командой не воспроизводится; прогоны 43 и 44 — вне репозитория |
| 5 | гибрид с K = 4 при seed 42 | `cl.all_winner`, `cl.seed` | победитель по z-оценкам среди всех семейств |
| 5 | 0,94 | `cl.ari_hybrid_spectral` | ARI гибрида и спектральной |
| 5 | 0,22; 0,28; K = 3 | `cl.var_graph_basket_cos_ari_same`, `cl.var_graph_basket_cos_ari`, `cl.var_graph_basket_cos_k` | сеть косинуса |
| 5 | 0,32; 0,40; K = 4 | `cl.var_nodes_separate_ari_same`, `cl.var_nodes_separate_ari`, `cl.var_nodes_separate_k` | районы отдельными узлами |
| 5 | 1,00 | `cl.var_no_level_ari_same` | без уровня трат |
| 5 | KEFRiN, K = 4; 0,15; 0,97 | `cl.var_x_clipped_winner`, `cl.var_x_clipped_k`, `cl.var_x_clipped_ari`, `cl.var_x_clipped_ari_same` | усечённые хвосты |
| 5 | 1000 перестановок | `cl.val_perms` | внешняя проверка |
| 5 | 10 проверок | `cl.val_n_sig`, `cl.val_n_tests` | значимы все |
| 5 | 0,152; 0,194; 0,024 | `cl.val_ip_per_1000_difference`, `cl.val_orgs_per_1000_difference`, `cl.val_nights_pc_difference` | ε² |
| 5 | 0,002–0,003 | `cl.val_ip_per_1000_difference_null` (0,002), `cl.val_orgs_per_1000_difference_null` (0,002), `cl.val_nights_pc_difference_null` (0,003) | ε² на перестановках |
| 5 | 4 из 4 | `cl.val_signs_ok`, `cl.val_signs_n` | знаки |
| 5 | от 0,006 до 0,026 | `cl.val_ip_per_1000_beyond_attributes` (0,006), `cl.val_orgs_per_1000_beyond_attributes` (0,026), `cl.val_nights_pc_beyond_attributes` (0,010) | прирост R² |
| 5, табл. | 0,85 | `cl.stab_median_kmeans` | медианная устойчивость K-means |
| 5, табл. | 2–34 | `cl.small_kmeans_min`, `cl.small_kmeans_max` | мелкий тип K-means |
| 5, табл. | 0,48 | `cl.syn_nograph_best_feat` | лучший метод по X без сигнала в графе |
| 5, табл. | 0,53 | `cl.stab_median_ward` | медианная устойчивость Уорда |
| 5, табл. | K = 3; 0,09 | `cl.win_gmm_k`, `cl.ari_hybrid_gmm` | гауссова смесь |
| 5, табл. | от 5 до 400; 2 кластера | `cl.hdbscan_mcs_min`, `cl.hdbscan_mcs_max`, `cl.hdbscan_kmax` | HDBSCAN |
| 5, табл. | 10 разбиений Leiden без несвязных | `cl.feas_leiden`, `cl.rl_disconnected`; `csv:outputs/cluster/resolution_limit.csv` | |
| 5, табл. | 10% рёбер; 0,63; 0,98 | `yaml:clustering.impl.edge_perturbation.drop`, `cl.pert_leiden`, `cl.pert_hybrid` | удаление рёбер |
| 5, табл. | 7 из 10 | `cl.louvain_n_reached`, `cl.n_grid` | |
| 5, табл. | 0,90; 0,97 | `cl.spectral_stability`, `cl.pert_spectral` | спектральная |
| 5, табл. | 1–32 | `cl.small_shalileh_mirkin_min`, `cl.small_shalileh_mirkin_max` | мелкий тип KEFRiN |
| 5, табл. | 16 узлов (K = 4) | `csv:outputs/cluster/small_clusters.csv [cand = kmeans_joint_k04; size = 16]` | мелкий тип базовой линии |
| 5 | `seeds: 10`, `hybrid_alpha: 0.5`, `n_clusters: [3, 12]`, `repeats: 20`, `ci_level: 0.95`, `min_ari: 0.05`, `seed_check: 5`, `tie_chain_tolerance: 0`, `quality_tolerance: baseline_sd` | `yaml:clustering.protocol`, `yaml:clustering.n_clusters`, `yaml:clustering.synthetic`, `yaml:clustering.impl` | параметры |

## Раздел 6. ICVI

| Раздел | Число в тексте | Источник | Что это |
|---|---|---|---|
| 6 | 87 | `icvi.n_candidates` | кандидатов |
| 6 | 200 | `icvi.permutations` | перестановок базиса |
| 6 | z = 178 | `icvi.final_z_avi` (178,0) | z AVI итога |
| 6 | 100 подвыборок | `icvi.bootstrap` | |
| 6 | 80% | `icvi.subsample` | |
| 6 | 2,0 | `icvi.ci_scale` | множитель отклонений |
| 6 | 95% | `icvi.ci_level` | |
| 6 | таблица «кандидат × индекс» (30 ячеек) | `csv:outputs/evaluate/icvi_long.csv [candidate ∈ {hybrid_k04, spectral_k04, leiden_k03, louvain_k06, gmm_k03}; value, ci_low, ci_high, z]` | итог: `icvi.final_sw`, `icvi.final_sw_lo`, `icvi.final_sw_hi`, `icvi.final_z_sw` и т. д. для ch, s_dbw, avi, avu, mq |
| 6 | 91,3% | `icvi.final_avi` (0,913) | AVI итога в процентах |
| 6 | 178,0 | `icvi.final_z_avi` | |
| 6 | 2/3; 9 кандидатов | `icvi.avu_k3_maxdev` (0,000), `icvi.n_k3` | AVU при K = 3 |
| 6 | 26 | `icvi.avu_z_neg`, `icvi.avu_z_finite` | z AVU < 0 |
| 6 | −52,0 | `icvi.avu_z_median` | |
| 6 | 0,60 | `icvi.u_max` (пара `icvi.u_max_pair`) | U₁₂ |
| 6 | 0,16 | `icvi.u_median` | |
| 6 | 0,90 | `icvi.tau_avi_mq` | τ по z |
| 6 | 0,84 | `icvi.tauk_avi_mq` | τ внутри K |
| 6 | 8 из 29; K от 9 | `icvi.sdbw_nan`, `icvi.n_feasible`, `icvi.sdbw_nan_kmin` | S_Dbw не определён |
| 6 | 0,96; 0,95; −0,92; −0,60 | `icvi.rho_z_k_avi`, `icvi.rho_z_k_mq`, `icvi.rho_z_k_ch`, `icvi.rho_z_k_sw` | ρ z с K |
| 6 | 0,927 | `cl.avi_adj_leiden` | AVI с поправкой, Leiden |
| 6 | 0,884 | `icvi.final_avi_adj` | итог |
| 6 | 0,249 | `icvi.final_avi_base` | E |
| 6 | Louvain, K = 6 первым по z AVI | `icvi.order_z_avi` | |
| 6 | Leiden с K = 3 при метриках с поправкой (выбор) | `cl.raw_l3_winner_graph`, `cl.raw_l3_winner_graph_k`; итог не меняется — `cl.raw_l2_changed` = 0 | фраза о пересчёте выбора |
| 6 | −0,26 | `icvi.tau_cross` | τ между пространствами |
| 6 | −0,81 | `icvi.tau_min` (пара `icvi.tau_min_pair`) | CH и MQ |
| 6 | `baseline_permutations: 200`, `s_dbw_density: pair`, `bootstrap: 100`, `ci_level: 0.95` | `yaml:icvi` | параметры |

## Раздел 7. Динамика

| Раздел | Число в тексте | Источник | Что это |
|---|---|---|---|
| 7 | 15,3% | `dyn.change_main` | сменили тип, основной способ |
| 7 | 2,8 раза | `dyn.ratio_main` | изменение / шум |
| 7 | K = 4 | `dyn.k` | |
| 7 | 2023 год (признаки места) | `yaml:dynamics.tracking.place_year` | |
| 7 | 1000 выборок | `dyn.bootstrap` | |
| 7 | 95% | `dyn.level` | |
| 7 | 13,7–17,1% | `dyn.change_lo_main`, `dyn.change_hi_main` | интервал |
| 7 | 5,5% | `dyn.noise_main` | шум |
| 7 | +9,8 п. п. | `dyn.diff_main` | |
| 7 | 8,3…11,5 | `dyn.lo_main`, `dyn.hi_main` | |
| 7 | 181 | `dyn.n_rel_main` | надёжных переходов |
| 7 | 14,1%; 5,6%; +8,4 п. п.; 6,9…10,0; 161 | `dyn.change_fixed`, `dyn.noise_fixed`, `dyn.diff_fixed`, `dyn.lo_fixed`, `dyn.hi_fixed`, `dyn.n_rel_fixed` | фиксированные типы |
| 7 | 15,3%; 5,3%; +10,0 п. п.; 8,3…11,7; 184 | `dyn.change_evo`, `dyn.noise_evo`, `dyn.diff_evo`, `dyn.lo_evo`, `dyn.hi_evo`, `dyn.n_rel_evo` | эволюционная кластеризация |
| 7 | 0,99 | `dyn.ari_tr_evo` | ARI переходов |
| 7 | 0,91 | `dyn.ari_tr_fixed` | |
| 7 | 10,2% | `dyn.share_rel_main` | доля надёжных переходов |
| 7 | 55 из 73 | `dyn.n_regions_rel`, `dyn.n_regions` | регионы с переходами |
| 7 | 74; 55; 26; 18 | `dyn.flows_main` | потоки |
| 7 | 0,3; 0,5; 0,7 | `dyn.tau_lo`, `dyn.tau`, `dyn.tau_hi` | пороги MONIC |
| 7 | 12 пар; все 4 типа сохранились | `dyn.n_adjacent`, `dyn.ev_adj_surv_05` (48 = 12 × 4), `dyn.ev_fl_surv_05` (4) | события; пороги 0,3 и 0,7 — `dyn.ev_*_03`, `dyn.ev_*_07` |
| 7 | 2 < 1 < 3 < 4 | `dyn.cafe_order` | порядок типов по общепиту |
| 7 | α = 0,05 | `dyn.driver_alpha` | |
| 7 | 1,25 | `dyn.drv_ratio` | |
| 7 | p = 0,001 | `dyn.drv_p` | |
| 7 | AUC 0,57 | `dyn.drv_auc` | |
| 7 | 5-е | `dyn.drv_rank` | |
| 7 | 2,11 | `dyn.drv_top_ratio` (часть — `dyn.drv_top`) | |
| 7 | 3,3·10⁻²² | `dyn.drv_top_p` | |
| 7 | 1399 | `dyn.drv_alt_n` | строгое «без смены» |
| 7 | 6,7·10⁻⁴ | `dyn.drv_alt_p` | |
| 7 | 1,27 | `dyn.drv_alt_ratio` | |
| 7 | 1,03 | `dyn.drv_level_ratio` | |
| 7 | 0,379 | `dyn.drv_level_p` | |
| 7 | 11 месяцев из 12 | `yaml:dynamics.window_months`, `yaml:dynamics.step_months` | пересечение соседних окон |
| 7 | 0,99 (seed) | `dyn.seed_ari_min` | |
| 7 | `events_jaccard: 0.5`, `bootstrap: 1000`, `window_months: 12`, `step_months: 1` | `yaml:dynamics` | параметры |

## Раздел 9. Ограничения

Повторяют числа разделов 2–7, кроме:

| Раздел | Число в тексте | Источник | Что это |
|---|---|---|---|
| 9 | 5 регионах | `e1.n_gap_regions_2024` | нет зарплаты за 2024 год |
| 9 | 0,007 | `icvi.final_sw` | силуэт итога |
| 9 | 143 | `cl.hybrid_zavi_min` | наименьшая z AVI гибрида |
| 9 | 4 из 5, 1; 3 из 5, 2 из 5 | `cl.sf_prereg_all`, `cl.sf_tolerance_all` | выбор среди всех семейств по seed |
| 9 | 5 → 20; 29.09 | `yaml:clustering.synthetic.repeats` (комментарий «было 5»); `docs/clustering.md`, раздел 10 | синтетика не предрегистрирована |

## Раздел 10. Воспроизводимость

| Раздел | Число в тексте | Источник | Что это |
|---|---|---|---|
| 10 | 3.12; 3.11–3.13 | `pyproject.toml [requires-python = ">=3.11,<3.14"]`; `uv run --frozen python --version` → 3.12.14 | версия Python |
| 10 | `seed: 42` | `yaml:seed` | |
| 10 | 263 МБ; 38 МБ; 42 МБ | замер: `du -sh data/raw data/processed outputs` | место на диске |
| 10 | 68 с | замер: `panel` во временной папке | |
| 10 | 42 с | README, раздел «Этап 1» | замер 26.09 |
| 10 | 16 с | замер: `features` во временной папке | |
| 10 | 431 с (около 7 мин) | замер: `network` во временной папке | |
| 10 | 69 мин; 4 мин; 13 мин; 6 процессов | `cl.time_total_min`, `cl.time_bootstrap_min`, `cl.time_variants_min`, `cl.workers` | прогон 30.09 с seed 42 |
| 10 | 104 с | замер: `evaluate` во временной папке | |
| 10 | не больше 1 мин | `dyn.minutes` | |
| 10 | 16 ГБ | `yaml:clustering.impl.workers` (комментарий) | |
| 10 | sha256 трёх файлов | `yaml:sources.hackathon.sha256`, `yaml:sources.borders.sha256`, `yaml:sources.ndfl.sha256` | |
| 10 | 14 файлов | `yaml:sources.bdmo.members` | число файлов БД ПМО |
| 10 | 30.09.2026; seed 43 и 44; 1776 узлах; ARI 1,00 | сравнение `data/processed/cluster_final.parquet` с `cluster_final.parquet` прогонов с `seed: 43` и `seed: 44` во временных папках: `sklearn.metrics.adjusted_rand_score` = 1.0, доля совпавших меток 1.0 на 1776 общих узлах | проверено при подготовке отчёта 30.09 |
| 10 | спектральная с K = 4 при seed 43 | `cl.all_winner` в `report_facts.json` прогона с `seed: 43` | |

## Приложение А

| Раздел | Число в тексте | Источник | Что это |
|---|---|---|---|
| А | n = 1776 | `feat.n_nodes` | |
| А | 11 признаков | `cl.n_x` | |
| А | k = 10 | `yaml:network.sparsify.k` | |

## Приложение Б

| Раздел | Число в тексте | Источник | Что это |
|---|---|---|---|
| Б | 25.09.2026 | `ls -l --time-style=full-iso data/raw/*/`: `hackathonlicence.zip` — 2026-09-25 23:46 +0300, `t_dict_municipal.rar` — 23:46, файлы `bdmo/` и `ndfl/` — 23:47 | дата скачивания; манифеста загрузки нет, загрузчик пишет файл через `.part` и переименовывает (`src/munnet/data.py:66`) |
| Б | тома, номера, страницы, DOI | проверка библиографии по Crossref и страницам издателей 28.09.2026 (заметка участника `literature-verified`) | библиография |
