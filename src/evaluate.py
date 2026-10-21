"""Baseline evaluation of the popularity recommender against the joined interactions.

Task rules
----------
watched        : a (user, show) pair counts as watched when overlap_sec >= T
train          : weeks 11-20
validation     : weeks 21-30
truth(u)       : {tv_show_id | u watched it in validation}
candidate pool : labelled shows (tv_show_id != 0) airing in the target weeks
                 (21-30 for validation, 31-42 for test)
metric         : MAP@5, with
                 AP@5 = sum over ranks k of P@k * rel(k) / min(|truth|, 5)
                 averaged over the evaluated users
population     : only the 1,260 users in data/submission.csv that have at least
                 one validation show are scored (the rest have no truth)

Every user_id is handled as a string everywhere (submission.csv, interactions
parquet) so that ids join correctly.

Run: python src/evaluate.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

INTERACTIONS = DATA / "interactions_11_30.parquet"
SUBMISSION = DATA / "submission.csv"

# Schedule file holding the labelled airings of each split.
SCHEDULE_BY_SPLIT = {
    "train": DATA / "export_arh_11-20-final.csv",
    "validation": DATA / "export_arh_21-30-final.csv",
    "test": DATA / "export_arh_31-42-final.csv",
}

TRAIN_WEEKS = (11, 20)
VALID_WEEKS = (21, 30)
TEST_WEEKS = (31, 42)
TOP_K = 5
THRESHOLDS = (60, 300, 900)


def section(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


# --------------------------------------------------------------------------- #
# metric
# --------------------------------------------------------------------------- #

def ap_at_k(truth, predictions, k=TOP_K):
    """AP@k = sum over ranks k of P@k * rel(k) / min(|truth|, k).

    rel(k) is 1 when the item at rank k is in the truth set, P@k is the
    precision at k.  Returns NaN when there is no truth to score against.
    """
    if not truth:
        return float("nan")
    hits = 0
    score = 0.0
    for rank, item in enumerate(list(predictions)[:k], start=1):
        if item in truth:
            hits += 1
            score += hits / rank
    return score / min(len(truth), k)


def map_at_k(truths, predictions, k=TOP_K):
    """Mean AP@k over users.  `truths` and `predictions` are dicts user -> value."""
    aps = [ap_at_k(truths[u], predictions[u], k) for u in truths]
    aps = [a for a in aps if not np.isnan(a)]
    if not aps:
        return float("nan")
    return float(np.mean(aps))


def unit_test_map_at_5():
    """Two hand-checkable users.

    user A: truth {10, 20}, predictions [10, 30, 20, 40, 50]
        k=1 hit -> P@1 = 1/1, k=3 hit -> P@3 = 2/3, other ranks contribute 0
        AP@5 = (1 + 2/3) / min(2, 5) = (5/3) / 2 = 5/6
    user B: truth {60}, predictions [70, 80, 90, 60, 100]
        only k=4 is a hit -> P@4 = 1/4
        AP@5 = (1/4) / min(1, 5) = 1/4
    MAP@5 = (5/6 + 1/4) / 2 = 13/24
    """
    truths = {
        "userA": {10, 20},
        "userB": {60},
    }
    predictions = {
        "userA": [10, 30, 20, 40, 50],
        "userB": [70, 80, 90, 60, 100],
    }
    expected_ap = {"userA": 5 / 6, "userB": 1 / 4}
    expected_map = 13 / 24

    section("UNIT TEST: map_at_5")
    computed = {}
    for user in ("userA", "userB"):
        got = ap_at_k(truths[user], predictions[user], TOP_K)
        computed[user] = got
        print(f"  {user}: truth={sorted(truths[user])} preds={predictions[user]}")
        print(f"    expected AP@5 = {expected_ap[user]:.10f}  "
              f"computed = {got:.10f}  "
              f"{'OK' if abs(got - expected_ap[user]) < 1e-12 else 'FAIL'}")
    got_map = map_at_k(truths, predictions, TOP_K)
    print(f"  expected MAP@5 = {expected_map:.10f}  computed = {got_map:.10f}  "
          f"{'OK' if abs(got_map - expected_map) < 1e-12 else 'FAIL'}")
    assert abs(computed["userA"] - 5 / 6) < 1e-12
    assert abs(computed["userB"] - 1 / 4) < 1e-12
    assert abs(got_map - 13 / 24) < 1e-12
    print("  unit test PASSED")


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #

def load_interactions():
    """Joined interactions with user_id kept as string."""
    inter = pq.read_table(INTERACTIONS).to_pandas()
    inter["user_id"] = inter["user_id"].astype(str)
    inter["tv_show_id"] = inter["tv_show_id"].astype("int64")
    inter["week"] = inter["week"].astype("int64")
    inter["overlap_sec"] = inter["overlap_sec"].astype("int64")
    return inter


def load_submission_users():
    """The 1,260 submission users, user_id read as string."""
    sub = pd.read_csv(SUBMISSION, dtype=str)
    sub["user_id"] = sub["user_id"].astype(str)
    return sub["user_id"].tolist(), sub


def candidate_pool(split):
    """Distinct labelled tv_show_id that air in the weeks of `split`."""
    path = SCHEDULE_BY_SPLIT[split]
    ids = set()
    for chunk in pd.read_csv(path, chunksize=200_000, usecols=["tv_show_id"],
                              dtype=str, keep_default_na=False):
        chunk_ids = chunk["tv_show_id"].str.strip()
        ids.update(chunk_ids[chunk_ids.ne("0") & chunk_ids.ne("")].unique())
    # the schedule stores ids as strings, the interactions as int64
    return {int(i) for i in ids}


def top_popular(train_pairs, pool, k=TOP_K):
    """Most-watched shows by distinct train users, restricted to the pool.

    Ties are broken by ascending tv_show_id so the ranking is deterministic.
    """
    counts = train_pairs.groupby("tv_show_id")["user_id"].nunique()
    counts = counts[counts.index.isin(pool)]
    ranked = counts.rename("users").reset_index()
    ranked = ranked.sort_values(by=["users", "tv_show_id"], ascending=[False, True])
    return ranked.head(k)["tv_show_id"].tolist()


# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #

def evaluate(inter, submission_users, pool, threshold, split="validation"):
    lo, hi = {"train": TRAIN_WEEKS, "validation": VALID_WEEKS,
              "test": TEST_WEEKS}[split]

    train = inter[(inter["week"] >= TRAIN_WEEKS[0])
                  & (inter["week"] <= TRAIN_WEEKS[1])
                  & (inter["overlap_sec"] >= threshold)]
    val = inter[(inter["week"] >= lo) & (inter["week"] <= hi)
                & (inter["overlap_sec"] >= threshold)]

    truth_all = val.groupby("user_id")["tv_show_id"].agg(set)
    eval_users = [u for u in submission_users if u in truth_all.index]
    truths = {u: truth_all[u] for u in eval_users}

    predictions = top_popular(train, pool, TOP_K)

    reachable = []
    for u in eval_users:
        reachable.append(len(truths[u] & pool) / len(truths[u]))

    return {
        "threshold": threshold,
        "n_eval_users": len(eval_users),
        "n_submission_users": len(submission_users),
        "map_at_5": map_at_k(truths, {u: predictions for u in eval_users}, TOP_K),
        "top5": predictions,
        "mean_truth_size": float(np.mean([len(truths[u]) for u in eval_users])) if eval_users else 0.0,
        "mean_truth_reachable": float(np.mean(reachable)) if reachable else 0.0,
        "n_train_pairs": len(train),
        "n_val_pairs": len(val),
        "n_pool": len(pool),
        "n_popular_in_pool": len(predictions),
    }


def main():
    unit_test_map_at_5()

    section("LOADING")
    inter = load_interactions()
    submission_users, sub = load_submission_users()
    print(f"interactions rows : {len(inter)}")
    print(f"submission users   : {len(submission_users)} (distinct "
          f"{sub['user_id'].nunique()})")
    print(f"user_id dtype      : interactions={inter['user_id'].dtype}, "
          f"submission={sub['user_id'].dtype}")
    overlap = set(sub["user_id"]) & set(inter["user_id"])
    print(f"submission users present in interactions: {len(overlap)}")

    pool = candidate_pool("validation")
    print(f"candidate pool (labelled shows airing in weeks {VALID_WEEKS[0]}-"
          f"{VALID_WEEKS[1]}): {len(pool)}")

    section("VALIDATION MAP@5 - popularity baseline")
    results = []
    for threshold in THRESHOLDS:
        res = evaluate(inter, submission_users, pool, threshold)
        results.append(res)
        print(f"\nT = {threshold} s")
        print(f"  users evaluated (submission users with >=1 validation show): "
              f"{res['n_eval_users']} / {res['n_submission_users']}")
        print(f"  MAP@5                       : {res['map_at_5']:.6f}")
        print(f"  mean |truth| per user       : {res['mean_truth_size']:.2f}")
        print(f"  mean share of truth in pool : {res['mean_truth_reachable']:.2%}")
        print(f"  train pairs / val pairs     : {res['n_train_pairs']} / {res['n_val_pairs']}")
        print(f"  top-5 (by distinct train users): {res['top5']}")

    section("SUMMARY")
    print(f"{'T (s)':>8} {'eval users':>12} {'MAP@5':>10} {'|truth|':>9} {'truth in pool':>14}")
    for res in results:
        print(f"{res['threshold']:>8} {res['n_eval_users']:>12} {res['map_at_5']:>10.6f} "
              f"{res['mean_truth_size']:>9.2f} {res['mean_truth_reachable']:>14.2%}")


if __name__ == "__main__":
    main()
