# sweettv-recsys

Midterm exam, LLM4Rec (HSE): Kaggle "sweet.tv — TV program recommender".
Author: Panasenkova Polina (Kaggle: polinapa1).

## Data
Put the Kaggle files into `data/` (not in git):
`dataset11-30.csv`, `export_arh_11-20-final.csv`, `export_arh_21-30-final.csv`,
`export_arh_31-42-final.csv`, `submission.csv`.

## Environment
Python 3.12.7, pandas 2.2.2, numpy 1.26.4, scipy 1.13.1, pyarrow,
implicit 0.7.3, lightgbm 4.7.0.

## Run order
1. `python src/explore.py` – basic counts of the log and schedule
2. `python src/join.py` – first per-view join (kept for history, replaced by airings.py)
3. `python src/airings.py` – view log -> (user, airing) screen time, `data/airings_11_30.parquet`
4. `python src/evaluate_v2.py` – validation labels (screen time >= 0.8), popularity and own-history baselines
5. `python src/submit.py` – v1 popularity (made before the official label rule); `python src/submit_v2.py` – v2 own history
6. `python src/calibrate.py` – old threshold check (not used for the final labels)
7. `python src/models.py` – ALS and content-based on validation
8. `python src/hybrid.py`, `python src/submit_hybrid.py` – history + ALS fill (v3, v4)
9. `python src/ltr.py`, `python src/submit_ltr.py` – LightGBM ranker (v5)
10. `python src/ltr_v2.py`, `python src/submit_ltr_v2.py` – ranker experiments A-D, v6 = B + D
11. `python src/robust_v7.py` – v6 bagged over seeds 42-46 (v7, final) and bootstrap checks
12. `python src/fix_report.py` – weekly activity diagnostics and reprinted bootstrap rows
13. `python src/two_users.py` – lists for the two users in the report

Validation: fit on weeks 11-20, evaluate on weeks 21-30 (MAP@5).

## Submissions (outputs/)
| file | validation MAP@5 | Kaggle private |
|---|---|---|
| submission_v1_popularity.csv | 0.2537* | 0.27809 |
| submission_v2_own_history.csv | 0.3604 | 0.38517 |
| submission_v3_hybrid.csv | 0.3631 | 0.38383 |
| submission_v5_ltr.csv | 0.3903 | 0.41889 |
| submission_v6_ltr_features.csv | 0.4022 | 0.41452 |
| **submission_v7_robust.csv** (final) | 0.4048 | 0.41958 |

*v1 was built with the earlier 300 s rule (src/submit.py); 0.2537 is popularity under the official rule.

Note: the v3 lists in `outputs/two_users.md` are wrong for both users (they repeat ALS);
the report uses the v3 lists from the submission files. `submission_v4_history_als.csv`
was built but not submitted.
