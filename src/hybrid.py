"""Hybrid recommender: expected number of viewings in the target window.

Same split, pool, users, labels, recommendation tie-break and both truth
tie-breaks as src/models.py:

  * train = weeks 11-20, validation = weeks 21-30
  * pool  = labelled shows airing in weeks 21-30
  * users = the 1150 submission users with >= 5 qualifying validation shows
  * labels = airings with screen_time >= 0.8, counted per (user, show)

The score of a show for a user is built from three pieces:

  weight(a)     0.5 ** ((last_train_week - week(a)) / H) for every qualifying
                airing a of the show, so that recent airings count more
  rate(u,s)     sum of weight(a) over the user's qualifying airings of s in
                the training weeks, divided by the number of airings of s in
                the training weeks (taken from the schedule, so the denominator
                counts every airing of the show, watched or not)
  expected(u,s) rate(u,s) * the number of airings of s in the target weeks

The number of target airings comes from the target-window schedule only; the
viewing data of the target window is never touched.

Ranking is by expected desc, then the usual tie-break (the user's own more
recent last qualifying airing desc, then the smaller tv_show_id).  A user with
fewer than 5 shows of score > 0 is topped up with the popularity list, or with
the ALS ranking (factors 32) in front of it.

H is tuned on the validation split only: H in {2, 4, 8, inf}.  The ablation
"recency count" is the recency-weighted count on its own, without the airing
factor, so the two differ only in the score function.  The final hybrid is the
best of the four variants (score function x H), also selected on the
validation split.

Run: python src/hybrid.py
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate import ap_at_k, map_at_k
from evaluate_v2 import (SCHEDULE_VALIDATION, SUBMISSION, TOP_K, TRAIN_WEEKS,
                          VALID_WEEKS, candidate_pool, load_airings,
                          pad_with_popularity, qualifying, rank_shows, show_popularity)
from models import (ALS_FACTORS, SCHEDULE_TRAIN, TieBreak, build_truth, load_schedule,
                    method_als, method_content, method_own_history,
                    method_popularity, rank_by_score, user_show_counts_full)

H_VALUES = (2.0, 4.0, 8.0, float("inf"))
ALS_FACTORS_FOR_HYBRID = 32
ALS_DEPTH = 20
FINAL_H = 8.0
BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_SEED = 42


def section(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


# --------------------------------------------------------------------------- #
# schedule side
# --------------------------------------------------------------------------- #

def schedule_airing_counts(path, weeks):
    """{show: (number of airings, last airing)} inside the given ISO weeks.

    Only the schedule is read here, never the viewing data.
    """
    counts = {}
    for chunk in pd.read_csv(path, usecols=["tv_show_id", "start_time"], dtype=str,
                             keep_default_na=False, chunksize=200_000):
        show = chunk["tv_show_id"].str.strip()
        keep = show.ne("0") & show.ne("")
        chunk = chunk[keep]
        show = show[keep]
        if chunk.empty:
            continue
        start = pd.to_datetime(chunk["start_time"], format="%d.%m.%Y %H:%M:%S",
                               errors="coerce")
        week = start.dt.isocalendar().week.to_numpy()
        inside = (week >= weeks[0]) & (week <= weeks[1])
        if not inside.any():
            continue
        frame = pd.DataFrame({"s": show[inside].to_numpy(),
                              "t": start[inside].to_numpy()})
        for s, grp in frame.groupby("s"):
            n, last = len(grp), grp["t"].max()
            if s in counts:
                counts[s] = (counts[s][0] + n, max(counts[s][1], last))
            else:
                counts[s] = (n, last)
    return counts


def weighted_counts(q_train, H, last_train_week):
    """Per (user, show): the recency-weighted number of qualifying airings."""
    weeks = q_train["week"].to_numpy(dtype=np.float64)
    if np.isinf(H):
        weights = np.ones(len(weeks))
    else:
        weights = np.power(0.5, (last_train_week - weeks) / H)
    frame = pd.DataFrame({"user_id": q_train["user_id"].to_numpy(),
                          "tv_show_id": q_train["tv_show_id"].to_numpy(),
                          "w": weights})
    return (frame.groupby(["user_id", "tv_show_id"], as_index=False)["w"]
                .sum())


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #

def build_scores(weighted, n_train_air, n_target_air, pool, use_airing_factor):
    """Per (user, show): the recency-weighted count and the expected value.

    The rows are restricted to the shows of the pool, because only those can
    be recommended.  `use_airing_factor` switches between the plain recency
    weighted count and rate * target airings.
    """
    df = weighted[weighted["tv_show_id"].isin(pool)].copy()
    df["n_train"] = df["tv_show_id"].map(n_train_air)
    df["n_target"] = df["tv_show_id"].map(n_target_air)
    assert df["n_train"].notna().all(), "show without training airings"
    assert (df["n_train"] > 0).all(), "show without training airings"
    assert df["n_target"].notna().all(), "show outside the target window"

    df["rate"] = df["w"] / df["n_train"]
    if use_airing_factor:
        df["score"] = df["rate"] * df["n_target"]
    else:
        df["score"] = df["w"]
    return df[["user_id", "tv_show_id", "score", "rate"]]


def method_scored(scores, tie, popularity_top5, users, als_preds=None):
    """Rank by score desc; top up with the ALS list, then with popularity.

    The tie-break is the user's own last qualifying airing of the show, so a
    user with tied scores gets exactly the same order as the own-history
    baseline.
    """
    by_user = {u: grp for u, grp in scores.groupby("user_id", sort=False)}
    preds = {}
    for user in users:
        grp = by_user.get(user)
        ranked = []
        if grp is not None and len(grp):
            show_ids = grp["tv_show_id"].to_numpy()
            ranked = rank_by_score(grp["score"].to_numpy(), show_ids,
                                   tie.for_ids(user, show_ids), k=TOP_K)
        if len(ranked) < TOP_K and als_preds is not None:
            for show in als_preds.get(user, []):
                if len(ranked) >= TOP_K:
                    break
                if show not in ranked:
                    ranked.append(show)
        if len(ranked) < TOP_K:
            ranked = pad_with_popularity(ranked, popularity_top5)
        preds[user] = ranked
    return preds


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main():
    started = time.perf_counter()

    section("ENVIRONMENT")
    print(f"python  : {platform.python_version()}")
    print(f"numpy   : {np.__version__}")
    print(f"pandas  : {pd.__version__}")
    print(f"train weeks {TRAIN_WEEKS} -> validation weeks {VALID_WEEKS}, top-K = {TOP_K}")
    print(f"H candidates: {list(H_VALUES)}")

    section("LOADING")
    air = load_airings()
    sub = pd.read_csv(SUBMISSION, dtype=str)
    submission_users = sub["user_id"].astype(str).tolist()
    pool = candidate_pool(VALID_WEEKS, SCHEDULE_VALIDATION)
    print(f"validation pool: {len(pool)} labelled shows airing in weeks "
          f"{VALID_WEEKS[0]}-{VALID_WEEKS[1]}")

    meta = load_schedule([SCHEDULE_TRAIN, SCHEDULE_VALIDATION])

    section("LABELS AND TRUTH")
    q_train = qualifying(air, TRAIN_WEEKS)
    q_val = qualifying(air, VALID_WEEKS)
    train_counts = user_show_counts_full(q_train)
    val_counts = user_show_counts_full(q_val)
    truth_recency = build_truth(val_counts, "recency")
    truth_first = build_truth(val_counts, "first-occurrence")
    users = [u for u in submission_users if u in truth_recency]
    truths = {"recency": {u: truth_recency[u] for u in users},
              "first-occurrence": {u: truth_first[u] for u in users}}
    print(f"evaluated users: {len(users)}")

    section("SCHEDULE SIDE (airings, no viewing data)")
    n_train_air = {int(s): n for s, (n, _) in
                   schedule_airing_counts(SCHEDULE_TRAIN, TRAIN_WEEKS).items()}
    n_target_air = {int(s): n for s, (n, _) in
                    schedule_airing_counts(SCHEDULE_VALIDATION, VALID_WEEKS).items()}
    print(f"shows airing in the training weeks : {len(n_train_air)}")
    print(f"shows airing in the target weeks   : {len(n_target_air)}")
    print(f"pool shows with target airings     : {len(set(pool) & set(n_target_air))}")

    pop = show_popularity(q_train)
    pop = pop[pop["tv_show_id"].isin(pool)]
    popularity_top5 = rank_shows(pop)
    tie = TieBreak(q_train, pool)
    print(f"popularity top-5: {popularity_top5}")

    results = {}

    def run(name, fn):
        t0 = time.perf_counter()
        preds = fn()
        elapsed = time.perf_counter() - t0
        results[name] = (preds, elapsed)
        print(f"  {name:<34s} {elapsed:8.2f}s")
        return preds

    section("BASELINES")
    preds_pop = run("popularity",
                    lambda: method_popularity(pool, pop, popularity_top5, users))
    preds_own = run("own history",
                    lambda: method_own_history(pool, train_counts, popularity_top5, users)[0])
    preds_als = run(f"ALS (factors={ALS_FACTORS_FOR_HYBRID})",
                    lambda: method_als(train_counts, pool, tie, popularity_top5, users,
                                       ALS_FACTORS_FOR_HYBRID, depth=ALS_DEPTH)[0])
    preds_content = run("content-based",
                        lambda: method_content(train_counts, pool, meta, tie,
                                               popularity_top5, users)[0])

    # ----------------------------------------------------------------------- #
    # tuning
    # ----------------------------------------------------------------------- #

    section("TUNING H (validation only)")
    last_train_week = TRAIN_WEEKS[1]
    tuning = {}
    for H in H_VALUES:
        t0 = time.perf_counter()
        weighted = weighted_counts(q_train, H, last_train_week)
        count_scores = build_scores(weighted, n_train_air, n_target_air, pool, False)
        exp_scores = build_scores(weighted, n_train_air, n_target_air, pool, True)

        preds_count = method_scored(count_scores, tie, popularity_top5, users)
        preds_exp = method_scored(exp_scores, tie, popularity_top5, users)
        preds_count_als = method_scored(count_scores, tie, popularity_top5, users,
                                        als_preds=preds_als)
        preds_exp_als = method_scored(exp_scores, tie, popularity_top5, users,
                                      als_preds=preds_als)

        tuning[H] = {
            "count": preds_count,
            "expected": preds_exp,
            "count_als": preds_count_als,
            "expected_als": preds_exp_als,
        }
        elapsed = time.perf_counter() - t0
        print(f"  H={str(H):>4s}  "
              + "   ".join(f"{name} {map_at_k(truths['recency'], preds, TOP_K):.6f}"
                           for name, preds in
                           (("recency count", preds_count), ("expected", preds_exp),
                            ("count+ALS", preds_count_als), ("expected+ALS", preds_exp_als)))
              + f"   ({elapsed:.2f}s)")

    def best(key):
        return max(H_VALUES, key=lambda h: map_at_k(truths["recency"],
                                                    tuning[h][key], TOP_K))

    best_count_h = best("count")
    best_exp_h = best("expected")
    print(f"\nbest H for the recency count : {best_count_h}")
    print(f"best H for expected           : {best_exp_h}")

    # the recency count with H = inf is the plain count, so it must reproduce
    # the own-history baseline exactly: same candidates, same tie-break
    inf_count = tuning[float("inf")]["count"]
    n_same = sum(1 for u in users if list(inf_count[u]) == list(preds_own[u]))
    print(f"recency count (H=inf) reproduces own history for {n_same} / {len(users)} users")
    assert n_same == len(users), "tie-break differs from the own-history baseline"

    # rebuild the chosen variants so that their runtime is measured the
    # same way as the baselines
    t0 = time.perf_counter()
    weighted_count = weighted_counts(q_train, best_count_h, last_train_week)
    count_scores = build_scores(weighted_count, n_train_air, n_target_air, pool, False)
    weighted_exp = weighted_counts(q_train, best_exp_h, last_train_week)
    exp_scores = build_scores(weighted_exp, n_train_air, n_target_air, pool, True)
    t_build = time.perf_counter() - t0

    t0 = time.perf_counter()
    preds_count = method_scored(count_scores, tie, popularity_top5, users)
    t_count = time.perf_counter() - t0

    t0 = time.perf_counter()
    preds_exp = method_scored(exp_scores, tie, popularity_top5, users)
    t_exp = time.perf_counter() - t0

    # the best variant over score function and H, for reference
    variants = [("count", best_count_h), ("expected", best_exp_h),
                ("count_als", best("count_als")), ("expected_als", best("expected_als"))]
    best_key, best_h = max(
        variants, key=lambda kv: map_at_k(truths["recency"], tuning[kv[1]][kv[0]], TOP_K))
    print(f"best variant on the validation split: {best_key} (H={best_h})")

    # the strategy that is submitted: the recency-weighted history at H=8,
    # the missing slots filled with the ALS list and then with popularity
    t0 = time.perf_counter()
    weighted_final = weighted_counts(q_train, FINAL_H, last_train_week)
    count_final = build_scores(weighted_final, n_train_air, n_target_air, pool, False)
    t_build_final = time.perf_counter() - t0

    t0 = time.perf_counter()
    preds_final = method_scored(count_final, tie, popularity_top5, users, als_preds=preds_als)
    t_final = time.perf_counter() - t0
    preds_final_pop = method_scored(count_final, tie, popularity_top5, users)

    section("RESULTS")
    rows = [
        ("popularity", preds_pop, results["popularity"][1]),
        ("own history", preds_own, results["own history"][1]),
        (f"ALS (factors={ALS_FACTORS_FOR_HYBRID})", preds_als,
         results[f"ALS (factors={ALS_FACTORS_FOR_HYBRID})"][1]),
        ("content-based", preds_content, results["content-based"][1]),
        (f"recency count (H={best_count_h})", preds_count, t_count),
        (f"expected (H={best_exp_h})", preds_exp, t_exp),
        (f"FINAL (H={FINAL_H}, ALS fill)", preds_final, t_final),
    ]

    header = (f"{'method':<30s} {'MAP@5 recency':>15s} {'MAP@5 first-occ':>16s} "
              f"{'hits@5':>8s} {'runtime':>9s}")
    print(header)
    print("-" * len(header))
    for name, preds, runtime in rows:
        s_rec = map_at_k(truths["recency"], preds, TOP_K)
        s_first = map_at_k(truths["first-occurrence"], preds, TOP_K)
        hits = np.mean([len(set(preds[u]) & set(truths["recency"][u])) for u in users])
        print(f"{name:<30s} {s_rec:>15.6f} {s_first:>16.6f} {hits:>8.3f} {runtime:>8.2f}s")
    print(f"\n(score construction for the chosen H took {t_build:.2f}s)")

    # ----------------------------------------------------------------------- #
    # final strategy
    # ----------------------------------------------------------------------- #

    section(f"FINAL STRATEGY (recency count H={FINAL_H}, ALS fill, then popularity)")

    n_own = count_final.groupby("user_id").size()
    short = [u for u in users if n_own.get(u, 0) < TOP_K]
    rest = [u for u in users if u not in set(short)]

    def subset(truth_dict, users_subset):
        return {u: truth_dict[u] for u in users_subset}

    print(f"users with fewer than {TOP_K} shows in their recency-weighted history "
          f"(H={FINAL_H}): {len(short)} / {len(users)}")

    print("\nMAP@5 of the whole evaluated set")
    header = (f"{'method':<30s} {'MAP@5 recency':>15s} {'MAP@5 first-occ':>16s} "
              f"{'hits@5':>8s}")
    print(header)
    print("-" * len(header))

    def row(name, preds, user_subset=None):
        user_subset = users if user_subset is None else user_subset
        s_rec = map_at_k(subset(truths["recency"], user_subset),
                         subset(preds, user_subset), TOP_K)
        s_first = map_at_k(subset(truths["first-occurrence"], user_subset),
                           subset(preds, user_subset), TOP_K)
        hits = np.mean([len(set(preds[u]) & set(truths["recency"][u]))
                        for u in user_subset]) if user_subset else float("nan")
        print(f"{name:<30s} {s_rec:>15.6f} {s_first:>16.6f} {hits:>8.3f}")

    row("final hybrid (ALS fill)", preds_final)
    row("same, popularity fill", preds_final_pop)

    print(f"\nMAP@5 of the {len(short)} users that needed a fill")
    print(header)
    print("-" * len(header))
    row("final hybrid (ALS fill)", preds_final, short)
    row("same, popularity fill", preds_final_pop, short)

    if rest:
        print(f"\nMAP@5 of the {len(rest)} users that did not need a fill")
        print(header)
        print("-" * len(header))
        row("final hybrid (ALS fill)", preds_final, rest)
        row("same, popularity fill", preds_final_pop, rest)

    # where do the filled slots come from?
    own_shows = {u: set(count_final[count_final["user_id"] == u]["tv_show_id"])
                 for u in short}
    n_filled = 0
    n_from_als = 0
    for u in short:
        filled = [s for s in preds_final[u] if s not in own_shows[u]]
        n_filled += len(filled)
        n_from_als += sum(1 for s in filled if s in set(preds_als.get(u, [])))
    print(f"\nfilled slots for those users: {n_filled}, "
          f"of which {n_from_als} come from the ALS list")

    # ----------------------------------------------------------------------- #
    # bootstrap
    # ----------------------------------------------------------------------- #

    section("BOOTSTRAP: recency count H=8 vs H=inf (both with the ALS fill)")

    preds_h8 = tuning[FINAL_H]["count_als"]
    preds_hinf = tuning[float("inf")]["count_als"]
    n_diff_rows = sum(1 for u in users if preds_h8[u] != preds_hinf[u])

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n_users = len(users)
    resamples = rng.integers(0, n_users, size=(BOOTSTRAP_RESAMPLES, n_users))

    for truth_name in ("recency", "first-occurrence"):
        ap_h8 = np.array([ap_at_k(truths[truth_name][u], preds_h8[u], TOP_K)
                          for u in users])
        ap_hinf = np.array([ap_at_k(truths[truth_name][u], preds_hinf[u], TOP_K)
                            for u in users])
        observed = ap_h8.mean() - ap_hinf.mean()
        draws = ap_h8[resamples].mean(axis=1) - ap_hinf[resamples].mean(axis=1)
        lo, hi = np.percentile(draws, [2.5, 97.5])
        print(f"\ntruth: {truth_name}")
        print(f"  MAP@5 H=8   : {ap_h8.mean():.6f}")
        print(f"  MAP@5 H=inf : {ap_hinf.mean():.6f}")
        print(f"  observed difference (H=8 - H=inf): {observed:+.6f}")
        print(f"  bootstrap mean difference        : {draws.mean():+.6f}")
        print(f"  95% percentile interval          : [{lo:+.6f}, {hi:+.6f}]")
        print(f"  95% interval, 8 decimals         : [{lo:+.8f}, {hi:+.8f}]")
        print(f"  resamples with a positive difference: {(draws > 0).mean():.3f}")

    print(f"\nusers whose ranking differs between H=8 and H=inf: {n_diff_rows} / {n_users}")
    print(f"({BOOTSTRAP_RESAMPLES} resamples of {n_users} users, seed 42)")

    section("SPOT CHECK")
    u = users[0]
    print(f"user {u}")
    print(f"  truth recency          : {truths['recency'][u]}")
    print(f"  own history            : {preds_own[u]}")
    print(f"  recency count (H={best_count_h})   : {preds_count[u]}")
    print(f"  expected (H={best_exp_h})         : {preds_exp[u]}")
    print(f"  final hybrid           : {preds_final[u]}")

    section("DONE")
    print(f"final strategy: recency count H={FINAL_H}, ALS fill, then popularity")
    print(f"total runtime: {time.perf_counter() - started:.2f}s")


if __name__ == "__main__":
    main()
