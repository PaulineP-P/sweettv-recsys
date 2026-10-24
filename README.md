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
2. `python src/airings.py` – view log -> (user, airing) screen time, `data/airings_11_30.parquet`
3. `python src/evaluate_v2.py` – validation labels (screen time >= 0.8), popularity and own-history baselines
4. `python src/models.py` – ALS and content-based on validation
5. `python src/hybrid.py` – history + ALS fill variants
6. `python src/ltr.py` – LightGBM ranker (v5)
7. `python src/ltr_v2.py` – ranker experiments A-D (v6 = B + D)
8. `python src/submit_ltr.py` – test submissions of the ranker (v5, v6, v7)
9. `python src/two_users.py` – lists for the two users in the report

Validation: fit on weeks 11-20, evaluate on weeks 21-30 (MAP@5).

## Submissions (outputs/)
| file | validation MAP@5 | Kaggle private |
|---|---|---|
| submission_v1_popularity.csv | 0.2537 | 0.27809 |
| submission_v2_own_history.csv | 0.3604 | 0.38517 |
| submission_v3_hybrid.csv | 0.3631 | 0.38383 |
| submission_v5_ltr.csv | 0.3903 | 0.41889 |
| submission_v6_ltr_features.csv | 0.4022 | 0.41452 |
| **submission_v7_robust.csv** (final) | 0.4048 | 0.41958 |

Note: the v3 lists in `outputs/two_users.md` are wrong (they repeat ALS);
the report uses the lists from the submission files.
