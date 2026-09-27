# Figure and table cross-reference audit

The audit expands all `\input` files, checks that each label has a meaningful prose reference before its source environment, and then verifies the rendered PDF reading order. Final counts are seven figures and fifteen tables.

| ID | Label | Caption | Source location | First textual reference | Reference exists | Reference precedes float source | Rendered page | Rendered order status | Action taken |
|---|---|---|---|---|---|---|---|---|---|
| Fig. 1 | `fig:system` | Heterogeneous heavy-duty platoon on a long descent | `sections/01_introduction.tex:19` | `sections/01_introduction.tex:2` | YES | YES | 3 | PASS (ref p. 1) | Added conflict-oriented reference |
| Table I | `tab:positioning` | Scope relative to closest literature | `sections/02_related_work.tex:12` | `sections/02_related_work.tex:10` | YES | YES | 3 | PASS (ref p. 2) | Added positioning explanation |
| Table II | `tab:parameters` | Vehicle parameters required per class | `tables/parameter_table.tex:1` | `sections/03_problem_formulation.tex:47` | YES | YES | 3 | PASS (same-page reference first) | Added paired provenance/symbol reference |
| Table III | `tab:nomenclature` | Core symbols | `tables/nomenclature.tex:1` | `sections/03_problem_formulation.tex:47` | YES | YES | 3 | PASS (same-page reference first) | Added paired provenance/symbol reference |
| Fig. 2 | `fig:coupled` | Coupled dual-brake/thermal dynamics | `sections/03_problem_formulation.tex:52` | `sections/03_problem_formulation.tex:51` | YES | YES | 4 | PASS (ref p. 3) | Added coupling explanation |
| Table IV | `tab:predictive_symbols` | Predictive and learning symbols | `tables/predictive_symbols.tex:1` | `sections/05_method.tex:45` | YES | YES | 6 | PASS (ref p. 5) | Moved first reference ahead of top float |
| Table V | `tab:constraints` | Projection constraints and guarantee status | `tables/constraint_table.tex:1` | `sections/05_method.tex:45` | YES | YES | 6 | PASS (ref p. 5) | Moved first reference ahead of top float |
| Fig. 3 | `fig:architecture` | Certificate-aware CTDE architecture | `sections/05_method.tex:114` | `sections/05_method.tex:112` | YES | YES | 8 | PASS (ref p. 7) | Added execution/CTDE summary |
| Fig. 4 | `fig:gradient` | Regularity-gated KKT path and fallback | `sections/05_method.tex:121` | `sections/05_method.tex:97` | YES | YES | 8 | PASS (ref p. 7) | Added regularity/fallback semantics |
| Table VI | `tab:final-parameters` | Frozen literature-calibrated parameters | `generated/final/tables/table_final_parameters.tex:1` | `sections/07_experiments.tex:3` | YES | YES | 10 | PASS (ref p. 9) | Retained provenance reference |
| Table VII | `tab:formal-protocol` | Frozen formal learning protocol | `sections/07_experiments.tex:14` | `sections/07_experiments.tex:12` | YES | YES | 10 | PASS (ref p. 9) | Bottom placement preserves order |
| Fig. 5 | `fig:model-feasibility` | M1 physical-feasibility evidence | `sections/08_results_and_limitations.tex:8` | `sections/08_results_and_limitations.tex:6` | YES | YES | 10 | PASS (same-page reference first) | Kept reserve/count distinction; bottom placement |
| Table VIII | `tab:final-model-evidence` | Selected frozen model-based evidence | `generated/final/tables/table_final_model_evidence.tex:1` | `sections/08_results_and_limitations.tex:15` | YES | YES | 11 | PASS (ref p. 10) | Moved input after explanatory prose |
| Table IX | `tab:phase2m-controller` | Matched controller comparison | `generated/tables/table_m_b_controller_comparison.tex:1` | `sections/08_results_and_limitations.tex:15` | YES | YES | 11 | PASS (ref p. 10) | Moved input after explanatory prose |
| Table X | `tab:phase2m-predictive` | Predictive-certificate and horizon metrics | `generated/tables/table_m_c_predictive_certificate.tex:1` | `sections/08_results_and_limitations.tex:21` | YES | YES | 11 | PASS (ref p. 10) | Retained horizon interpretation |
| Table XI | `tab:phase2m-robustness` | Robustness/design-sensitivity results | `generated/tables/table_m_d_robustness.tex:1` | `sections/08_results_and_limitations.tex:27` | YES | YES | 11 | PASS (ref p. 10) | Retained limitation-oriented reference |
| Fig. 6 | `fig:communication` | M7 communication evidence | `sections/08_results_and_limitations.tex:33` | `sections/08_results_and_limitations.tex:31` | YES | YES | 13 | PASS (ref p. 11) | Explicitly rejects instantaneous-global interpretation |
| Table XII | `tab:phase2m-communication` | Communication/bottleneck diagnostics | `generated/tables/table_m_e_communication.tex:1` | `sections/08_results_and_limitations.tex:31` | YES | YES | 12 | PASS (ref p. 11) | Paired with Fig. 6 and delayed semantics |
| Table XIII | `tab:phase2m-runtime` | Component-wise controller runtime | `generated/tables/table_m_f_runtime.tex:1` | `sections/08_results_and_limitations.tex:44` | YES | YES | 12 | PASS (ref p. 11) | Added component-runtime reference and P99 definition |
| Fig. 7 | `fig:learning-diagnostics` | Formal regularity/fallback diagnostics | `sections/08_results_and_limitations.tex:53` | `sections/08_results_and_limitations.tex:51` | YES | YES | 13 | PASS (ref p. 12) | Added fallback-outcome explanation |
| Table XIV | `tab:final-learning-evidence` | Ten-paired-seed learning evidence | `generated/final/tables/table_final_learning_evidence.tex:1` | `sections/08_results_and_limitations.tex:60` | YES | YES | 13 | PASS (ref p. 12) | Moved input after mechanistic-equality explanation |
| Table XV | `tab:external-baseline` | Deterministic matched heavy-duty comparison | `sections/08_results_and_limitations.tex:67` | `sections/08_results_and_limitations.tex:65` | YES | YES | 14 | PASS (ref p. 12) | Moved input after domain-transfer caveat |

Automated source audit: `missing_references=[]`, `references_after_source=[]`.

Rendered-order audit: **PASS — every first meaningful reference precedes the corresponding figure/table in normal two-column reading order.**
