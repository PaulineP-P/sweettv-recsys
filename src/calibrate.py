"""Calibrate the "watched" label definition against the Kaggle score.

The popularity submission scored private 0.27809 / public 0.28234 on Kaggle,
while our own validation MAP@5 for the same T=300 labels is ~0.73.  That gap
means our label definition is much looser than the organisers'.

This script sweeps candidate label definitions and reports the popularity
baseline MAP@5 on validation for each, so the definition can be chosen to
match the Kaggle score.  Everything else is unchanged:

  train      : weeks 11-20
  validation : weeks 21-30
  pool       : labelled shows airing in weeks 21-30
  users      : the 1,260 submission users with >= 1 validation show
  baseline   : top 5 shows by distinct train users, same list for every user

Run: python src/calibrate.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

from evaluate import (TOP_K, TRAIN_WEEKS, VALID_WEEKS, candidate_pool,
                      load_interactions, load_submission_users, map_at_k,
                      top_popular)

ROOT = Path(__file__).resolve().parents[1]

KAGGLE_PRIVATE = 0.27809
KAGGLE_PUBLIC = 0.28234

# A label definition is a predicate on the interaction rows.
LABEL_DEFS = [
    ("overlap_sec >= 300", lambda d: d["overlap_sec"] >= 300),
    ("overlap_sec >= 900", lambda d: d["overlap_sec"] >= 900),
    ("overlap_sec >= 1800", lambda d: d["overlap_sec"] >= 1800),
    ("overlap_sec >= 3600", lambda d: d["overlap_sec"] >= 3600),
    ("overlap_frac >= 0.3", lambda d: d["overlap_frac"] >= 0.3),
    ("overlap_frac >= 0.5", lambda d: d["overlap_frac"] >= 0.5),
    ("overlap_frac >= 0.8", lambda d: d["overlap_frac"] >= 0.8),
    ("overlap_sec >= 300 AND overlap_frac >= 0.5",
     lambda d: (d["overlap_sec"] >= 300) & (d["overlap_frac"] >= 0.5)),
]


def section(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def run_definition(inter, submission_users, pool, name, predicate):
    """Popularity baseline MAP@5 on validation for one label definition."""
    mask = predicate(inter)

    train = inter[(inter["week"] >= TRAIN_WEEKS[0]) & (inter["week"] <= TRAIN_WEEKS[1]) & mask]
    val = inter[(inter["week"] >= VALID_WEEKS[0]) & (inter["week"] <= VALID_WEEKS[1]) & mask]

    truth_all = val.groupby("user_id")["tv_show_id"].agg(set)
    eval_users = [u for u in submission_users if u in truth_all.index]
    truths = {u: truth_all[u] for u in eval_users}

    top5 = top_popular(train, pool, TOP_K)
    predictions = {u: top5 for u in eval_users}
    score = map_at_k(truths, predictions, TOP_K)

    mean_truth = float(np.mean([len(truths[u]) for u in eval_users])) if eval_users else 0.0
    reach = (float(np.mean([len(truths[u] & pool) / len(truths[u]) for u in eval_users]))
             if eval_users else 0.0)

    return {
        "name": name,
        "map": score,
        "mean_truth": mean_truth,
        "n_users": len(eval_users),
        "n_train": len(train),
        "n_val": len(val),
        "top5": top5,
        "reach": reach,
        "gap": abs(score - KAGGLE_PRIVATE),
    }


def main():
    section("LOADING")
    inter = load_interactions()
    submission_users, _ = load_submission_users()
    pool = candidate_pool("validation")
    print(f"interactions rows : {len(inter)}")
    print(f"submission users   : {len(submission_users)}")
    print(f"candidate pool    : {len(pool)} labelled shows airing in weeks "
          f"{VALID_WEEKS[0]}-{VALID_WEEKS[1]}")
    print(f"kaggle reference  : private {KAGGLE_PRIVATE} / public {KAGGLE_PUBLIC} "
          f"(popularity, weeks 11-30 labels, weeks 31-42 target)")

    section("LABEL DEFINITION SWEEP")
    rows = []
    for name, predicate in LABEL_DEFS:
        res = run_definition(inter, submission_users, pool, name, predicate)
        rows.append(res)

    print(f"{'label definition':<44} {'MAP@5':>9} {'|truth|':>9} {'users':>6} "
          f"{'val pairs':>10} {'|gap to kaggle|':>15}")
    for res in rows:
        print(f"{res['name']:<44} {res['map']:>9.6f} {res['mean_truth']:>9.2f} "
              f"{res['n_users']:>6} {res['n_val']:>10} {res['gap']:>15.6f}")

    best = min(rows, key=lambda r: r["gap"])
    section("CLOSEST TO THE KAGGLE SCORE")
    print(f"  {best['name']}")
    print(f"  validation MAP@5   : {best['map']:.6f}   "
          f"(kaggle private {KAGGLE_PRIVATE}, gap {best['gap']:.6f})")
    print(f"  mean |truth|/user  : {best['mean_truth']:.2f}")
    print(f"  users evaluated    : {best['n_users']} / {len(submission_users)}")
    print(f"  train pairs / val  : {best['n_train']} / {best['n_val']}")
    print(f"  top-5              : {best['top5']}")

    section("TOP-5 PER DEFINITION")
    for res in rows:
        print(f"  {res['name']:<44} {res['top5']}")

    section("FINE SWEEP: overlap_sec THRESHOLD (validation)")
    print(f"{'T (s)':>7} {'MAP@5':>9} {'|truth|':>9} {'users':>6} {'gap to kaggle':>14}")
    fine = []
    for threshold in (1800, 2100, 2400, 2700, 3000, 3300, 3600, 4200, 4800, 5400):
        res = run_definition(inter, submission_users, pool,
                             f"overlap_sec >= {threshold}",
                             lambda d, t=threshold: d["overlap_sec"] >= t)
        fine.append(res)
        print(f"{threshold:>7} {res['map']:>9.6f} {res['mean_truth']:>9.2f} "
              f"{res['n_users']:>6} {res['gap']:>14.6f}")
    best_fine = min(fine, key=lambda r: r["gap"])
    print(f"\n  best fine threshold: overlap_sec >= "
          f"{best_fine['name'].split('>= ')[1]} -> MAP@5 {best_fine['map']:.6f} "
          f"(gap {best_fine['gap']:.6f})")

    section("PREVIEW: weeks 11-30 popularity on the weeks 31-42 pool")
    test_pool = candidate_pool("test")
    print(f"  weeks 31-42 pool: {len(test_pool)} labelled shows")
    print(f"  {'label definition':<44} top-5 for the next submission")
    for name, predicate in LABEL_DEFS:
        mask = predicate(inter)
        used = inter[(inter["week"] >= TRAIN_WEEKS[0]) & (inter["week"] <= 30) & mask]
        top5 = top_popular(used, test_pool, TOP_K)
        print(f"  {name:<44} {top5}")


if __name__ == "__main__":
    main()
