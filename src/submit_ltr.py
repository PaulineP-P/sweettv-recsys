"""Build the LTR test submission (v5), the validation recipe shifted forward.

Validation run (src/ltr.py):
  fit       features weeks 11-15, labels weeks 16-20
  predict   features weeks 16-20, target weeks 21-30

Test run (this script), same features, same hyperparameters, seed 42:
  fit       features weeks 21-25, labels weeks 26-30
  predict   features weeks 26-30, target weeks 31-42

The ALS and the genre profiles of the fit run are built on weeks 11-25, the
ones of the prediction run on weeks 11-30.  The only thing read from the
31-42 weeks is the schedule (candidate filter, schedule counts).

Candidates per user: the shows with at least one screen_time >= 0.8 airing
from week 11 to week 30, plus the 30 most popular shows, restricted to the
shows airing in weeks 31-42.  Users with fewer than 5 candidates are padded
with the popularity top 5.

Run: python src/submit_ltr.py
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import time
from pathlib import Path

import lightgbm
import pandas as pd

from evaluate_v2 import (SCHEDULE_TEST, SCHEDULE_VALIDATION, SUBMISSION,
                          TEST_WEEKS, TOP_K, candidate_pool, load_airings,
                          qualifying, rank_shows, show_popularity)
from ltr import (FEATURES, LGB_PARAMS, Scorers, build_dataset, build_labels,
                 fit_ranker, predict_topk, schedule_table)
from models import SCHEDULE_TRAIN, TieBreak, load_schedule, user_show_counts_full

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
V3_CSV = OUTPUTS / "submission_v3_hybrid.csv"
OUT_CSV = OUTPUTS / "submission_v5_ltr.csv"

HISTORY_START_WEEK = 11
TOP_POPULAR_CANDIDATES = 30

FIT_FEATURE_WEEKS = (21, 25)
FIT_LABEL_WEEKS = (26, 30)
TEST_FEATURE_WEEKS = (26, 30)
TARGET_WEEKS = TEST_WEEKS


def section(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def write_submission(template, preds, pool, out_csv):
    """Write the recommendations and check the file against the template."""
    out = template[["user_id"]].copy()
    out["tv_show_id"] = [" ".join(str(s) for s in preds[u])
                         for u in template["user_id"]]

    assert list(out.columns) == list(template.columns), "header mismatch"
    assert len(out) == len(template), "row count mismatch"
    assert out["user_id"].tolist() == template["user_id"].tolist(), \
        "row order / user_id text mismatch"

    for user, row in zip(out["user_id"], out["tv_show_id"]):
        shows = [int(s) for s in row.split()]
        assert len(shows) == TOP_K, f"{user}: got {len(shows)} shows"
        assert len(set(shows)) == TOP_K, f"{user}: duplicate shows"
        assert set(shows) <= pool, f"{user}: show outside the pool"

    out.to_csv(out_csv, index=False, lineterminator="\n")
    with open(out_csv, "rb") as fh:
        blob = fh.read()
    if blob.endswith(b"\n"):
        with open(out_csv, "wb") as fh:
            fh.write(blob[:-1])
    return out


def main():
    started = time.perf_counter()

    section("ENVIRONMENT")
    print(f"lightgbm version : {lightgbm.__version__}")
    print(f"fit              : features {FIT_FEATURE_WEEKS}, "
          f"labels {FIT_LABEL_WEEKS}")
    print(f"predict          : features {TEST_FEATURE_WEEKS}, "
          f"target {TARGET_WEEKS}")
    print(f"LightGBM parameters: {LGB_PARAMS}")

    section("LOADING")
    air = load_airings()
    template = pd.read_csv(SUBMISSION, dtype=str)
    template["user_id"] = template["user_id"].astype(str)
    submission_users = template["user_id"].tolist()
    meta = load_schedule([SCHEDULE_TRAIN, SCHEDULE_VALIDATION, SCHEDULE_TEST])
    categories = sorted({v for v in meta["tv_show_category"].dropna().unique()
                         if isinstance(v, str)} | {"unknown"})
    test_pool = candidate_pool(TARGET_WEEKS, SCHEDULE_TEST)
    print(f"airing rows       : {len(air)}")
    print(f"submission users  : {len(submission_users)}")
    print(f"shows with metadata: {len(meta)}, categories: {len(categories)}")
    print(f"target pool       : {len(test_pool)} labelled shows airing in weeks "
          f"{TARGET_WEEKS[0]}-{TARGET_WEEKS[1]}")

    # ----------------------------------------------------------------------- #
    # fit setup: features 21-25, labels 26-30
    # ----------------------------------------------------------------------- #

    section(f"FIT SETUP (features {FIT_FEATURE_WEEKS}, labels {FIT_LABEL_WEEKS})")
    t0 = time.perf_counter()
    fit_target_shows = set(schedule_table([SCHEDULE_VALIDATION], FIT_LABEL_WEEKS))
    fit_sched_feature = schedule_table([SCHEDULE_VALIDATION], FIT_FEATURE_WEEKS)
    fit_sched_target = schedule_table([SCHEDULE_VALIDATION], FIT_LABEL_WEEKS)
    fit_labels, fit_truth = build_labels(air, FIT_LABEL_WEEKS)

    fit_hist = qualifying(air, (HISTORY_START_WEEK, FIT_FEATURE_WEEKS[1]))
    fit_counts = user_show_counts_full(fit_hist)
    fit_pop = show_popularity(fit_hist)
    fit_pop_top = rank_shows(fit_pop, k=TOP_POPULAR_CANDIDATES)
    fit_scorers = Scorers(fit_counts, fit_target_shows, meta)

    fit_users = sorted(fit_truth)
    fit_data = build_dataset(air, meta, FIT_FEATURE_WEEKS, FIT_LABEL_WEEKS,
                             fit_sched_feature, fit_sched_target, fit_target_shows,
                             fit_pop_top, fit_users, fit_scorers, fit_labels,
                             categories)
    print(f"users with a label window truth : {len(fit_users)}")
    print(f"candidate rows                   : {len(fit_data)} "
          f"({fit_data['user_id'].nunique()} users, "
          f"{fit_data.groupby('user_id').size().mean():.1f} per user)")
    print(f"positive rows (label > 0)        : {(fit_data['label'] > 0).sum()}")
    print(f"build time                       : {time.perf_counter() - t0:.2f}s")

    section("TRAINING THE RANKER")
    train_groups = fit_data[fit_data.groupby("user_id")["label"]
                            .transform("sum") > 0]["user_id"].nunique()
    t0 = time.perf_counter()
    ranker = fit_ranker(fit_data, FEATURES)
    print(f"training groups (users) : {train_groups}")
    print(f"fit time                : {time.perf_counter() - t0:.2f}s")

    # ----------------------------------------------------------------------- #
    # prediction setup: features 26-30, target 31-42
    # ----------------------------------------------------------------------- #

    section(f"PREDICTION SETUP (features {TEST_FEATURE_WEEKS}, "
            f"target {TARGET_WEEKS})")
    t0 = time.perf_counter()
    test_sched_feature = schedule_table([SCHEDULE_VALIDATION], TEST_FEATURE_WEEKS)
    test_sched_target = schedule_table([SCHEDULE_TEST], TARGET_WEEKS)

    q_hist = qualifying(air, (HISTORY_START_WEEK, TEST_FEATURE_WEEKS[1]))
    test_counts = user_show_counts_full(q_hist)
    test_pop = show_popularity(q_hist)
    test_pop_top = rank_shows(test_pop, k=TOP_POPULAR_CANDIDATES)
    test_scorers = Scorers(test_counts, test_pool, meta)

    test_data = build_dataset(air, meta, TEST_FEATURE_WEEKS, TARGET_WEEKS,
                              test_sched_feature, test_sched_target, test_pool,
                              test_pop_top, submission_users, test_scorers, {},
                              categories)
    tie = TieBreak(q_hist, test_pool)
    popularity_top5 = rank_shows(test_pop[test_pop["tv_show_id"].isin(test_pool)])
    print(f"candidate rows                   : {len(test_data)} "
          f"({test_data['user_id'].nunique()} users)")
    print(f"popularity top-5 (weeks 11-30)   : {popularity_top5}")
    print(f"build time                       : {time.perf_counter() - t0:.2f}s")
    print("\nthe ALS and the genre profiles of this run use weeks 11-30;")
    print("weeks 31-42 contribute the schedule only (pool, schedule counts).")

    # ----------------------------------------------------------------------- #
    # prediction
    # ----------------------------------------------------------------------- #

    section("PREDICTION")
    t0 = time.perf_counter()
    preds, test_data = predict_topk(ranker, test_data, FEATURES,
                                    submission_users, tie, popularity_top5)
    print(f"predict time            : {time.perf_counter() - t0:.2f}s")

    sizes = test_data.groupby("user_id").size()
    short = [u for u in submission_users if sizes.get(u, 0) < TOP_K]
    padded_slots = sum(max(TOP_K - sizes.get(u, 0), 0) for u in submission_users)
    print(f"candidates per user     : min {sizes.reindex(submission_users).fillna(0).min():.0f} "
          f"median {sizes.median():.0f} mean {sizes.mean():.1f} "
          f"max {sizes.max()}")
    print(f"users with fewer than {TOP_K} candidates (padded): {len(short)} / "
          f"{len(submission_users)}")
    print(f"slots filled by popularity padding: {padded_slots}")

    # ----------------------------------------------------------------------- #
    # write and check
    # ----------------------------------------------------------------------- #

    section(f"WRITING outputs/{OUT_CSV.name}")
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    out = write_submission(template, preds, test_pool, OUT_CSV)
    print(f"rows                 : {len(out)}")
    print(f"header               : {', '.join(out.columns)}")
    print(f"row order/user text  : identical to data/submission.csv")
    print(f"shows per row        : {TOP_K}, distinct, inside the target pool")
    with open(OUT_CSV, "rb") as fh:
        blob = fh.read()
    print(f"trailing newline     : {blob.endswith(b'\n')}")

    if V3_CSV.exists():
        v3 = pd.read_csv(V3_CSV, dtype=str).set_index("user_id")["tv_show_id"]
        mine = out.set_index("user_id")["tv_show_id"]
        shared = [u for u in mine.index if u in v3.index]
        n_diff = sum(1 for u in shared if mine[u] != v3[u])
        print(f"rows differing from {V3_CSV.name}: {n_diff} / {len(shared)}")

    section("HEAD OF THE FILE")
    with open(OUT_CSV, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if i >= 6:
                break
            print(line.rstrip("\n"))

    section("DONE")
    print(f"total runtime: {time.perf_counter() - started:.2f}s")


if __name__ == "__main__":
    main()
