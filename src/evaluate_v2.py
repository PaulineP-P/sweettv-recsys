"""Validation labels and baselines under the organisers' screen-time rule.

Rule (competition Data tab):
  drop tv_show_id = 0
  screen time = seconds the user watched that airing / airing duration
  keep only airings with screen time >= 0.8
  for each user count how many such airings each show has
  the answer is the 5 most frequent shows, descending

Tie-break (used everywhere, for the truth as well as for both baselines):
  equal counts are broken by the more recent last qualifying airing first,
  then by the smaller tv_show_id.

This script:
  1. builds the validation labels for weeks 21-30: truth = the user's top 5
     shows by count of >= 0.8 airings; only users with at least 5 such shows
     can be scored
  2. re-runs the popularity baseline: popularity = number of training users
     (weeks 11-20) with at least one >= 0.8 airing of the show, restricted to
     the validation pool, same top 5 for everyone
  3. adds an "own history" baseline: the user's own top 5 shows by count of
     >= 0.8 airings in weeks 11-20, restricted to the validation pool and
     padded with the popularity list when a user has fewer than 5

The helpers here are reused by src/submit_v2.py so that the validation and the
test submission rank shows identically.

Run: python src/evaluate_v2.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from evaluate import ap_at_k, map_at_k

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

AIRINGS = DATA / "airings_11_30.parquet"
SUBMISSION = DATA / "submission.csv"
SCHEDULE_VALIDATION = DATA / "export_arh_21-30-final.csv"
SCHEDULE_TEST = DATA / "export_arh_31-42-final.csv"

TRAIN_WEEKS = (11, 20)
VALID_WEEKS = (21, 30)
TEST_WEEKS = (31, 42)
TOP_K = 5
SCREEN_TIME_MIN = 0.8


def section(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


# --------------------------------------------------------------------------- #
# shared helpers
# --------------------------------------------------------------------------- #

def load_airings():
    air = pq.read_table(AIRINGS).to_pandas()
    air["user_id"] = air["user_id"].astype(str)
    air["tv_show_id"] = air["tv_show_id"].astype("int64")
    air["week"] = air["week"].astype("int64")
    air["screen_time"] = air["screen_time"].astype("float64")
    return air


def candidate_pool(weeks, schedule_path):
    """Labelled shows airing in the given ISO weeks (by airing start_time)."""
    ids = set()
    for chunk in pd.read_csv(schedule_path, chunksize=200_000, dtype=str,
                             keep_default_na=False,
                             usecols=["tv_show_id", "start_time"]):
        week = pd.to_datetime(chunk["start_time"], format="%d.%m.%Y %H:%M:%S",
                              errors="coerce").dt.isocalendar().week.to_numpy()
        sel = chunk["tv_show_id"].str.strip()
        keep = (week >= weeks[0]) & (week <= weeks[1]) & sel.ne("0") & sel.ne("")
        ids.update(sel[keep].unique())
    return {int(i) for i in ids}


def qualifying(air, weeks):
    """Rows with screen_time >= 0.8 inside the given weeks."""
    mask = ((air["week"] >= weeks[0]) & (air["week"] <= weeks[1])
            & (air["screen_time"] >= SCREEN_TIME_MIN))
    return air[mask]


def user_show_counts(q):
    """Per (user, show): number of qualifying airings and the last one."""
    return (q.groupby(["user_id", "tv_show_id"])
             .agg(count=("airing_start", "size"),
                  last_qual=("airing_start", "max"))
             .reset_index())


def show_popularity(q):
    """Per show: distinct qualifying users and the last qualifying airing."""
    return (q.groupby("tv_show_id")
             .agg(count=("user_id", "nunique"),
                  last_qual=("airing_start", "max"))
             .reset_index())


def rank_shows(df, k=TOP_K):
    """Order shows by count desc, then more recent last qualifying airing,
    then smaller tv_show_id.  `df` needs tv_show_id, count, last_qual."""
    if df.empty:
        return []
    ordered = df.sort_values(by=["count", "last_qual", "tv_show_id"],
                              ascending=[False, False, True], kind="stable")
    return list(ordered["tv_show_id"].head(k))


def pad_with_popularity(ranked, popularity_top5, k=TOP_K):
    """Top up a user's list with the global popularity list, no duplicates."""
    out = list(ranked)
    for show in popularity_top5:
        if len(out) >= k:
            break
        if show not in out:
            out.append(show)
    return out[:k]


# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #

def main():
    section("LOADING")
    air = load_airings()
    sub = pd.read_csv(SUBMISSION, dtype=str)
    submission_users = sub["user_id"].astype(str).tolist()
    pool = candidate_pool(VALID_WEEKS, SCHEDULE_VALIDATION)
    print(f"airing rows       : {len(air)}")
    print(f"submission users  : {len(submission_users)}")
    print(f"validation pool   : {len(pool)} labelled shows airing in weeks "
          f"{VALID_WEEKS[0]}-{VALID_WEEKS[1]}")

    section("VALIDATION LABELS (screen_time >= 0.8, weeks 21-30)")
    q_val = qualifying(air, VALID_WEEKS)
    val_counts = user_show_counts(q_val)
    print(f"qualifying validation airings : {len(q_val)}")

    truth, n_qualifying_shows = {}, {}
    for user, grp in val_counts.groupby("user_id", sort=False):
        n_qualifying_shows[user] = len(grp)
        if len(grp) >= TOP_K:
            truth[user] = rank_shows(grp)
    eval_users = [u for u in submission_users if u in truth]
    print(f"users with >= 1 qualifying show  : {len(n_qualifying_shows)}")
    print(f"users with >= 5 qualifying shows : {len(truth)}")
    print(f"--> of the {len(submission_users)} submission users, "
          f"{len(eval_users)} qualify")
    print(f"    not qualifying: {len(submission_users) - len(eval_users)}")

    sizes = pd.Series([n_qualifying_shows[u] for u in eval_users])
    print(f"\nqualifying shows per evaluated user: min {sizes.min()} "
          f"median {sizes.median()} mean {sizes.mean():.1f} max {sizes.max()}")
    reach = np.mean([len(set(truth[u]) & pool) / TOP_K for u in eval_users])
    print(f"mean share of truth inside the validation pool: {reach:.2%}")

    section("TRAIN SIDE (screen_time >= 0.8, weeks 11-20)")
    q_train = qualifying(air, TRAIN_WEEKS)
    print(f"qualifying training airings: {len(q_train)}")

    pop = show_popularity(q_train)
    pop = pop[pop["tv_show_id"].isin(pool)]
    popularity_top5 = rank_shows(pop)
    print(f"popularity top-5 (distinct train users): {popularity_top5}")
    print(pop.sort_values(by=["count", "last_qual", "tv_show_id"],
                          ascending=[False, False, True]).head(TOP_K).to_string(index=False))

    section("BASELINES")
    truths = {u: set(truth[u]) for u in eval_users}

    pop_preds = {u: popularity_top5 for u in eval_users}
    pop_map = map_at_k(truths, pop_preds, TOP_K)

    own_counts = user_show_counts(q_train)
    own_map, n_padded = {}, 0
    for user, grp in own_counts.groupby("user_id", sort=False):
        ranked = rank_shows(grp[grp["tv_show_id"].isin(pool)])
        if len(ranked) < TOP_K:
            n_padded += 1
            ranked = pad_with_popularity(ranked, popularity_top5)
        own_map[user] = ranked
    own_preds = {u: own_map.get(u, popularity_top5) for u in eval_users}
    own_map_score = map_at_k(truths, own_preds, TOP_K)

    print(f"users needing popularity padding: {n_padded} / {len(eval_users)}")
    print(f"\n{'baseline':<20} {'MAP@5':>10}")
    print(f"{'popularity':<20} {pop_map:>10.6f}")
    print(f"{'own history':<20} {own_map_score:>10.6f}")

    section("SPOT CHECK")
    u = eval_users[0]
    print(f"user {u}")
    print(f"  truth (top {TOP_K} by count): {truth[u]}")
    print(f"  popularity : {pop_preds[u]}")
    print(f"  own history: {own_preds[u]}")
    print(f"  popularity hits : {len(set(pop_preds[u]) & set(truth[u]))}/{TOP_K}  "
          f"AP@5 = {ap_at_k(truths[u], pop_preds[u], TOP_K):.6f}")
    print(f"  own history hits: {len(set(own_preds[u]) & set(truth[u]))}/{TOP_K}  "
          f"AP@5 = {ap_at_k(truths[u], own_preds[u], TOP_K):.6f}")


if __name__ == "__main__":
    main()
