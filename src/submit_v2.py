"""Build the second test submission: the "own history" baseline (v2).

For every user in data/submission.csv:
  * take their shows with screen_time >= 0.8 airings in weeks 11-30
  * rank them by count of qualifying airings, ties broken by the more recent
    last qualifying airing and then by the smaller tv_show_id
  * keep only shows that air in weeks 31-42 (the candidate pool)
  * take the top 5; if fewer than 5 survive, pad with the global popularity
    list (weeks 11-30, same definition), never repeating a show

The output keeps the exact header, row order and user_id text of
data/submission.csv (read with dtype=str, written back unchanged).

Run: python src/submit_v2.py
"""

from pathlib import Path

import pandas as pd

from evaluate_v2 import (SCHEDULE_TEST, SUBMISSION, TEST_WEEKS, TOP_K,
                         candidate_pool, load_airings, pad_with_popularity,
                         qualifying, rank_shows, show_popularity, user_show_counts)

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"

OUT_CSV = OUTPUTS / "submission_v2_own_history.csv"

ALL_WEEKS = (11, 30)


def section(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


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

    section("TRAINING SIDE (screen_time >= 0.8, weeks 11-30)")
    q = qualifying(air, ALL_WEEKS)
    print(f"qualifying airings : {len(q)}")

    counts = user_show_counts(q)
    pop = show_popularity(q)
    pop = pop[pop["tv_show_id"].isin(pool)]
    popularity_top5 = rank_shows(pop)
    print(f"popularity top-5 (distinct users, weeks 11-30): {popularity_top5}")
    print(pop.sort_values(by=["count", "last_qual", "tv_show_id"],
                          ascending=[False, False, True]).head(TOP_K).to_string(index=False))

    section("PER-USER RECOMMENDATIONS")
    by_user = {user: grp for user, grp in counts.groupby("user_id", sort=False)}
    n_padded = 0
    n_fully_padded = 0
    own_counts = []
    recs = {}
    for user in submission_users:
        grp = by_user.get(user)
        ranked = []
        if grp is not None:
            ranked = rank_shows(grp[grp["tv_show_id"].isin(pool)])
        own_counts.append(len(ranked))
        if len(ranked) < TOP_K:
            n_padded += 1
            if not ranked:
                n_fully_padded += 1
            ranked = pad_with_popularity(ranked, popularity_top5)
        assert len(ranked) == TOP_K, f"{user}: got {len(ranked)} shows"
        assert len(set(ranked)) == TOP_K, f"{user}: duplicate shows"
        recs[user] = ranked

    series = pd.Series(own_counts)
    print(f"own-history shows kept (in pool): min {series.min()} "
          f"median {series.median()} mean {series.mean():.1f} max {series.max()}")
    print(f"users needing popularity padding : {n_padded} / {len(submission_users)}")
    print(f"users needing a full 5-show pad  : {n_fully_padded}")

    section("WRITING SUBMISSION")
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    out = template[["user_id"]].copy()
    out["tv_show_id"] = [ " ".join(str(s) for s in recs[u]) for u in template["user_id"] ]

    assert list(out.columns) == list(template.columns), "header mismatch"
    assert len(out) == len(template), "row count mismatch"
    assert out["user_id"].tolist() == template["user_id"].tolist(), \
        "row order / user_id text mismatch"

    out.to_csv(OUT_CSV, index=False, lineterminator="\n")
    with open(OUT_CSV, "rb") as fh:
        blob = fh.read()
    if blob.endswith(b"\n"):
        with open(OUT_CSV, "wb") as fh:
            fh.write(blob[:-1])

    print(f"wrote {OUT_CSV}")
    print(f"rows: {len(out)}  (template rows: {len(template)})")

    section("HEAD OF outputs/submission_v2_own_history.csv")
    with open(OUT_CSV, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if i >= 6:
                break
            print(line.rstrip("\n"))


if __name__ == "__main__":
    main()
