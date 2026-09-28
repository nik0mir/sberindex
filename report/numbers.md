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

Результаты сравнения среди всех семейств (фронт, Копленд, Борда, порядки критериев, пороги 55–60%) из отчёта
убраны до правок этапа cluster и в реестр не входят. Числа, которые повторяют уже внесённое (например, 1776 в разных разделах), внесены один раз — при первом
появлении. Номера разделов, рисунков, таблиц и формул статей не вносятся.

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

| Раздел | Число в тексте | Источник | Что это |
|---|---|---|---|
| 5 | K = 3, …, 12 | `cl.k_min`, `cl.k_max` | сетка K |
| 5 | `2ee1363`, 28.09.2026 | `git:2ee1363` | коммит предрегистрации |
| 5 | 12 418 | `cl.n_edges` | рёбер G |
| 5 | 0,11 | `cl.weight_min` | наименьший вес ребра |
| 5 | одна компонента | `cl.n_components` | 1 |
| 5 | 11 атрибутов | `cl.n_x` | признаков X |
| 5 | ρ = ξ = 1 | `yaml:clustering.impl.shalileh_mirkin.rho`, `…xi` | веса KEFRiN |
| 5 | α = 0,5 | `cl.alpha`; `yaml:clustering.protocol.hybrid_alpha` | вес графа в гибриде |
| 5 | 2% | `cl.min_share` | наименьший тип |
| 5 | 36 | `cl.min_share_nodes` | узлов в 2% |
| 5 | 50% | `cl.max_share` | крупнейший тип |
| 5 | 10% | `cl.max_noise` | шум HDBSCAN |
| 5 | 200 разметок | `cl.icvi_perms` | случайный базис |
| 5 | 100 подвыборок | `cl.bootstrap` | бутстрап устойчивости |
| 5 | 80% | `cl.subsample` | доля узлов в подвыборке |
| 5 | глубина 3 | `cl.tree_depth` | дерево объяснимости |
| 5 | 17 признаков | `cl.n_tree` | входы дерева |
| 5 | 5-кратная | `cl.cv_folds` | перекрёстная проверка |
| 5 | 0,02; 0 | `cl.tie_stability`, `cl.tie_interpretability`; `yaml:clustering.selection.tie` | допуски ничьей |
| 5 | 20 наборов вместо 5 | `yaml:clustering.synthetic.repeats` | изменено этапом cluster после выходов 28.09 (незакоммиченная правка): числа синтетики ниже — из прогона с 5 наборами и будут пересчитаны |
| 5 | 600 узлов | `cl.syn_n` | синтетика |
| 5 | 4 группы | `cl.syn_k` | синтетика |
| 5 | шести координатам | `yaml:clustering.synthetic.n_basket` | 6 |
| 5 | 16 ячеек | `cl.syn_both_won`, `cl.syn_both_cells` | 16 из 16 |
| 5 | 0,91 | `cl.syn_strong_kmeans_joint` | базовая линия, сильные сигналы |
| 5 | 0,84 | `cl.syn_strong_hybrid` | гибрид |
| 5 | 0,42 (KEFRiN, синтетика) | `cl.syn_strong_shalileh_mirkin` | KEFRiN |
| 5 | половина рёбер между группами | `yaml:clustering.synthetic.sbm_mixing` | доля 0,5 |
| 5 | 0,42 (гибрид, блочный граф) | `cl.sbm_mid_hybrid` | |
| 5 | 0,14 | `cl.sbm_mid_leiden` | |
| 5 | 0,19 | `cl.sbm_mid_kefrin` | |
| 5 | 0,3 | `cl.sbm_mix_min` | доля внешних рёбер |
| 5 | 0,98 | `cl.sbm_low_leiden` | Leiden при доле 0,3 |
| 5 | 0,00; 0,55; 0,59; сигнал в признаках 2, в графе 0 | `csv:outputs/cluster/synthetic_summary.csv [design = knn; graph_signal = 0; feature_signal = 2; hybrid, shalileh_mirkin, kmeans]` | пометка «ждёт правок этапа cluster» и строка «Гибрид» в таблице методов |
| 5 | 87 | `cl.n_candidates` | кандидатов |
| 5 | 29 | `cl.n_feasible` | допустимых |
| 5 | 10 значений K | `cl.cands_shalileh_mirkin`, `cl.cands_kmeans_joint`; допустимых — `cl.feas_shalileh_mirkin` = 0, `cl.feas_kmeans_joint` = 0 | |
| 5 | 105 из 108 | `cl.n_small_market`, `cl.n_small` | мелких типов, выделенных по доступности рынков |
| 5 | 16 районов (мелкий тип гибрида) | `cl.suburbs_size` | |
| 5 | 26 MAD | `cl.suburbs_scaled` | |
| 5 | K = 4 | `cl.final_k` | итог |
| 5 | 51,5% | `cl.hybrid_k3_max_share` | крупнейший тип гибрида при K = 3 |
| 5 | 806, 473, 394, 103 | `cl.type1_size` … `cl.type4_size` | размеры типов |
| 5 | 0,91 | `cl.final_stability` | устойчивость итога |
| 5 | 0,90 | `cl.final_jaccard_min` | худший Жаккар типа |
| 5 | 0,75 | `cl.jaccard_stable` | порог Хеннига |
| 5 | 10 разных seed; ARI 1,00 | `cl.seeds`, `cl.final_seed_ari` | совпадение разбиений гибрида K = 4 по seed |
| 5 | 0,94 | `cl.ari_hybrid_spectral` | ARI гибрида и спектральной |
| 5 | 0,22 | `cl.var_graph_basket_cos_ari_same` | сеть косинуса |
| 5 | 0,32 | `cl.var_nodes_separate_ari_same` | районы отдельными узлами |
| 5 | 1,00 | `cl.var_no_level_ari_same` | без уровня трат |
| 5 | KEFRiN, K = 4 | `cl.var_x_clipped_winner`, `cl.var_x_clipped_k` | усечённые хвосты |
| 5 | 0,15 | `cl.var_x_clipped_ari` | |
| 5 | 0,97 | `cl.var_x_clipped_ari_same` | |
| 5 | 1000 перестановок | `cl.val_perms` | внешняя проверка |
| 5 | 10 проверок | `cl.val_n_sig`, `cl.val_n_tests` | значимы все |
| 5 | 0,152; 0,194; 0,024 | `cl.val_ip_per_1000_difference`, `cl.val_orgs_per_1000_difference`, `cl.val_nights_pc_difference` | ε² |
| 5 | 0,002–0,003 | `cl.val_ip_per_1000_difference_null` (0,002), `cl.val_orgs_per_1000_difference_null` (0,002), `cl.val_nights_pc_difference_null` (0,003) | ε² на перестановках |
| 5 | 4 из 4 | `cl.val_signs_ok`, `cl.val_signs_n` | знаки |
| 5 | от 0,006 до 0,026 | `cl.val_ip_per_1000_beyond_attributes` (0,006), `cl.val_orgs_per_1000_beyond_attributes` (0,026), `cl.val_nights_pc_beyond_attributes` (0,010) | прирост R² |
| 5 | 2–34 | `cl.small_kmeans_min`, `cl.small_kmeans_max` | мелкий тип K-means |
| 5 | 0,53 | `cl.stab_median_ward` | медианная устойчивость Уорда |
| 5 | K = 3 (гауссова смесь) | `cl.win_gmm_k` | |
| 5 | 0,09 | `cl.ari_hybrid_gmm` | |
| 5 | от 5 до 400; 2 кластера | `cl.hdbscan_mcs_min`, `cl.hdbscan_mcs_max`, `cl.hdbscan_kmax` | HDBSCAN |
| 5 | 10 K (Leiden) | `cl.feas_leiden` | допустим при всех K |
| 5 | 10% рёбер | `yaml:clustering.impl.edge_perturbation.drop` | 0.1 |
| 5 | 0,63 | `cl.pert_leiden` | |
| 5 | 7 из 10 | `cl.louvain_n_reached` | |
| 5 | 0,90 (спектральная) | `csv:outputs/cluster/candidates.csv [spectral_k04; stability = 0,901]` | |
| 5 | 0,97 (спектральная) | `cl.pert_spectral` | |
| 5 | 1–32 | `cl.small_shalileh_mirkin_min`, `cl.small_shalileh_mirkin_max` | мелкий тип KEFRiN |
| 5 | 2–33 | `cl.small_kmeans_joint_min`, `cl.small_kmeans_joint_max` | мелкий тип базовой линии |
| 5 | `seeds: 10`, `hybrid_alpha: 0.5`, `n_clusters: [3, 12]` | `yaml:clustering.protocol`, `yaml:clustering.n_clusters` | параметры |

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
| 10 | 17 мин; 3 мин; 8 мин; 6 процессов | `cl.time_total_min`, `cl.time_bootstrap_min`, `cl.time_variants_min`, `cl.workers` | |
| 10 | 104 с | замер: `evaluate` во временной папке | |
| 10 | не больше 1 мин | `dyn.minutes` | |
| 10 | 16 ГБ | `yaml:clustering.impl.workers` (комментарий) | |
| 10 | sha256 трёх файлов | `yaml:sources.hackathon.sha256`, `yaml:sources.borders.sha256`, `yaml:sources.ndfl.sha256` | |
| 10 | 14 файлов | `yaml:sources.bdmo.members` | число файлов БД ПМО |

## Приложение А

| Раздел | Число в тексте | Источник | Что это |
|---|---|---|---|
| А | n = 1776 | `feat.n_nodes` | |
| А | 11 признаков | `cl.n_x` | |
| А | k = 10 | `yaml:network.sparsify.k` | |
