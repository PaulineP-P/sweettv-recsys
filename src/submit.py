"""Build the test submission with the popularity baseline (v1).

Method
------
labels        : watched = overlap_sec >= 300 s (fixed for all methods)
popularity    : distinct users who watched each show in weeks 11-30
candidate pool: labelled shows (tv_show_id != 0) airing in weeks 31-42
                 (export_arh_31-42-final.csv)
recommendation: the top 5 pool shows by distinct-watcher count, the same for
                 every user; ties broken by ascending tv_show_id

The output keeps the exact header, row order and user_id text of
data/submission.csv (read with dtype=str, written back unchanged).

Run: python src/submit.py
"""

from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from evaluate import (TRAIN_WEEKS, TOP_K, candidate_pool, load_interactions,
                      top_popular)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"

SUBMISSION_TEMPLATE = DATA / "submission.csv"
SCHEDULE_TEST = DATA / "export_arh_31-42-final.csv"
INTERACTIONS = DATA / "interactions_11_30.parquet"

OUT_CSV = OUTPUTS / "submission_v1_popularity.csv"

THRESHOLD = 300
TEST_WEEKS = (31, 42)


def section(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def show_details(show_ids):
    """Modal title / category per show id, taken from the weeks 31-42 schedule."""
    wanted = set(show_ids)
    titles = {sid: {} for sid in wanted}
    cats = {sid: {} for sid in wanted}
    for chunk in pd.read_csv(SCHEDULE_TEST, chunksize=200_000, dtype=str,
                             keep_default_na=False,
                             usecols=["tv_show_id", "tv_show_title", "tv_show_category"]):
        ids = pd.to_numeric(chunk["tv_show_id"].str.strip(), errors="coerce")
        hit = chunk[ids.isin(wanted)]
        if hit.empty:
            continue
        for sid, title, category in zip(hit["tv_show_id"], hit["tv_show_title"],
                                        hit["tv_show_category"]):
            titles[int(sid)][title] = titles[int(sid)].get(title, 0) + 1
            cats[int(sid)][category] = cats[int(sid)].get(category, 0) + 1

    def modal(counter):
        if not counter:
            return ""
        return max(counter.items(), key=lambda kv: (kv[1], -len(kv[0])))[0]

    return {sid: (modal(titles[sid]), modal(cats[sid])) for sid in wanted}


def main():
    section("POPULARITY (weeks 11-30, overlap_sec >= 300)")
    inter = load_interactions()
    used = inter[(inter["week"] >= TRAIN_WEEKS[0]) & (inter["week"] <= 30)
                 & (inter["overlap_sec"] >= THRESHOLD)]
    print(f"labelled interactions used: {len(used)} "
          f"(weeks {TRAIN_WEEKS[0]}-30, overlap_sec >= {THRESHOLD})")
    print(f"distinct users: {used['user_id'].nunique()}")
    print(f"distinct shows: {used['tv_show_id'].nunique()}")

    section("CANDIDATE POOL (labelled shows airing in weeks 31-42)")
    pool = candidate_pool("test")
    print(f"pool size: {len(pool)}")

    top5 = top_popular(used, pool, TOP_K)
    counts = used.groupby("tv_show_id")["user_id"].nunique()
    details = show_details(top5)

    section("TOP 5 RECOMMENDED SHOWS")
    for rank, sid in enumerate(top5, start=1):
        title, category = details[sid]
        print(f"  {rank}. tv_show_id={sid:<10} watchers(weeks 11-30)={counts[sid]:<6} "
              f"category={category!r}")
        print(f"       title: {title!r}")

    section("WRITING SUBMISSION")
    OUTPUTS.mkdir(parents=True, exist_ok=True)

    template = pd.read_csv(SUBMISSION_TEMPLATE, dtype=str)
    template["user_id"] = template["user_id"].astype(str)
    submission = template[["user_id"]].copy()
    submission["tv_show_id"] = " ".join(str(sid) for sid in top5)

    assert list(submission.columns) == list(template.columns), "header mismatch"
    assert len(submission) == len(template), "row count mismatch"
    assert submission["user_id"].tolist() == template["user_id"].tolist(), \
        "row order / user_id text mismatch"

    # match the template byte format: \n endings, no trailing newline
    submission.to_csv(OUT_CSV, index=False, lineterminator="\n")
    with open(OUT_CSV, "rb") as fh:
        blob = fh.read()
    if blob.endswith(b"\n"):
        with open(OUT_CSV, "wb") as fh:
            fh.write(blob[:-1])

    print(f"wrote {OUT_CSV}")
    print(f"rows: {len(submission)}  (template rows: {len(template)})")

    section("HEAD OF outputs/submission_v1_popularity.csv")
    with open(OUT_CSV, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if i >= 6:
                break
            print(line.rstrip("\n"))


if __name__ == "__main__":
    main()
