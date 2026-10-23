"""Build the hybrid test submissions (v3 and v4).

Strategy (validated in src/hybrid.py):

  * the user's own shows are ranked by the recency-weighted number of
    qualifying airings, weight(a) = 0.5 ** ((30 - week(a)) / H)
  * a user with at least 5 such shows gets that ranking
  * a user with fewer gets the missing slots from the ALS ranking
    (factors 32, fitted on weeks 11-30) and then from the popularity list
  * ties are broken by the user's own more recent last qualifying airing,
    then by the smaller tv_show_id

Everything is trained on weeks 11-30 and the candidate pool is the set of
labelled shows airing in weeks 31-42.  The two files differ only in H:

  v3  H = 8    recency weighted history          submission_v3_hybrid.csv
  v4  H = inf  plain count of qualifying airings  submission_v4_history_als.csv

The output keeps the exact header, row order and user_id text of
data/submission.csv (read with dtype=str, written back unchanged).

Run: python src/submit_hybrid.py
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from pathlib import Path

import pandas as pd

from evaluate_v2 import (SCHEDULE_TEST, SUBMISSION, TEST_WEEKS, TOP_K,
                         candidate_pool, load_airings, qualifying,
                         show_popularity, rank_shows)
from hybrid import (ALS_DEPTH, ALS_FACTORS_FOR_HYBRID, method_scored,
                    weighted_counts)
from models import TieBreak, method_als, user_show_counts_full

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"

V2_CSV = OUTPUTS / "submission_v2_own_history.csv"

VARIANTS = [
    (8.0, "submission_v3_hybrid.csv", "v3: recency weighted history, H=8"),
    (float("inf"), "submission_v4_history_als.csv", "v4: plain count, H=inf"),
]

ALL_WEEKS = (11, 30)
LAST_TRAIN_WEEK = 30


def section(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def write_submission(template, preds, pool, out_csv):
    """Write the recommendations and check the file against the template."""
    out = template[["user_id"]].copy()
    out["tv_show_id"] = [" ".join(str(s) for s in preds[u]) for u in template["user_id"]]

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
    section("LOADING")
    air = load_airings()
    template = pd.read_csv(SUBMISSION, dtype=str)
    template["user_id"] = template["user_id"].astype(str)
    submission_users = template["user_id"].tolist()
    pool = candidate_pool(TEST_WEEKS, SCHEDULE_TEST)
    print(f"airing rows       : {len(air)}")
    print(f"submission users  : {len(submission_users)}")
    print(f"candidate pool    : {len(pool)} labelled shows airing in weeks "
          f"{TEST_WEEKS[0]}-{TEST_WEEKS[1]}")

    section(f"TRAINING SIDE (screen_time >= 0.8, weeks {ALL_WEEKS[0]}-{ALL_WEEKS[1]})")
    q = qualifying(air, ALL_WEEKS)
    counts = user_show_counts_full(q)
    print(f"qualifying airings : {len(q)}")
    print(f"(user, show) pairs  : {len(counts)}")

    pop = show_popularity(q)
    pop = pop[pop["tv_show_id"].isin(pool)]
    popularity_top5 = rank_shows(pop)
    print(f"popularity top-5 (distinct users, weeks {ALL_WEEKS[0]}-{ALL_WEEKS[1]}): "
          f"{popularity_top5}")

    section(f"ALS (factors {ALS_FACTORS_FOR_HYBRID}, weeks {ALL_WEEKS[0]}-{ALL_WEEKS[1]})")
    tie = TieBreak(q, pool)
    als_preds, als_info = method_als(counts, pool, tie, popularity_top5,
                                      submission_users, ALS_FACTORS_FOR_HYBRID,
                                      depth=ALS_DEPTH)
    print(f"matrix {als_info['n_users']} users x {als_info['n_items']} shows, "
          f"{als_info['nnz']} cells")
    print(f"pool shows with training data: {als_info['pool_known']} "
          f"(unknown: {als_info['pool_unknown']})")
    print(f"users falling back to popularity: {als_info['fallback']} / "
          f"{len(submission_users)}")

    users_with_any = set(counts["user_id"].unique())
    print(f"users with no qualifying airing at all in weeks "
          f"{ALL_WEEKS[0]}-{ALL_WEEKS[1]}: "
          f"{sum(1 for u in submission_users if u not in users_with_any)}")

    v2 = None
    if V2_CSV.exists():
        v2 = pd.read_csv(V2_CSV, dtype=str).set_index("user_id")["tv_show_id"]

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    for H, name, label in VARIANTS:
        section(f"{label} -> outputs/{name}")

        weighted = weighted_counts(q, H, LAST_TRAIN_WEEK)
        scores = weighted[weighted["tv_show_id"].isin(pool)].copy()
        scores["score"] = scores["w"]
        scores = scores[["user_id", "tv_show_id", "score"]]
        preds = method_scored(scores, tie, popularity_top5, submission_users,
                              als_preds=als_preds)

        n_own = scores.groupby("user_id").size()
        own_of = {u: set(scores[scores["user_id"] == u]["tv_show_id"])
                  for u in n_own.index}
        short = [u for u in submission_users if n_own.get(u, 0) < TOP_K]
        n_als_users = 0
        n_als_slots = 0
        for user in short:
            als_set = set(als_preds.get(user, []))
            filled = [s for s in preds[user] if s not in own_of.get(user, set())]
            from_als = [s for s in filled if s in als_set]
            n_als_slots += len(from_als)
            if from_als:
                n_als_users += 1

        out = write_submission(template, preds, pool, OUTPUTS / name)
        print(f"users with fewer than {TOP_K} shows in their recency-weighted history: "
              f"{len(short)} / {len(submission_users)}")
        print(f"users that got at least one show from the ALS list: {n_als_users} / "
              f"{len(submission_users)}")
        print(f"shows taken from the ALS list: {n_als_slots}")
        print(f"wrote {OUTPUTS / name}  ({len(out)} rows)")

        if v2 is not None:
            mine = out.set_index("user_id")["tv_show_id"]
            shared = [u for u in mine.index if u in v2.index]
            n_diff = sum(1 for u in shared if mine[u] != v2[u])
            print(f"rows differing from {V2_CSV.name}: {n_diff} / {len(shared)}")

    section("HEAD OF outputs/submission_v4_history_als.csv")
    with open(OUTPUTS / "submission_v4_history_als.csv", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if i >= 6:
                break
            print(line.rstrip("\n"))


if __name__ == "__main__":
    main()
