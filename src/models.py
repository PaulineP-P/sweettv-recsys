"""Compare four recommenders on one validation split.

Split, pool, users, labels and the recommendation tie-break are identical to
src/evaluate_v2.py:

  * train       = weeks 11-20, validation = weeks 21-30
  * pool        = labelled shows (tv_show_id != 0) airing in weeks 21-30
  * users       = the submission users with >= 5 qualifying validation shows
  * labels      = airings with screen_time >= 0.8, counted per (user, show)
  * recommendation tie-break = primary key desc, then the more recent last
    qualifying airing desc, then the smaller tv_show_id; the qualifying
    airing is the user's own airing of that show, and for a show the user
    has not watched the last qualifying airing of the show in the training
    weeks is used instead

The four methods:

  popularity    the same global top-5 list for everybody
  own history   the user's own most watched shows, padded with popularity
  ALS           implicit ALS factorisation of the user x show count matrix
  content       cosine between a genre profile of the user and each pool show

Two models are fitted on the training weeks only (weeks 11-20):

  ALS      user x show matrix of the number of >= 0.8 airings, with the
           confidence 1 + 10 * log1p(count) written straight into the matrix
           (implicit reads matrix values as the confidence itself, and treats
           an absent cell as confidence 1).  Factors 32 / 64 / 128,
           regularization 0.05, iterations 20, random_state 42.
  content  every show is a one-hot vector over tv_show_category and the three
           genre fields; a user profile is the count-weighted average of the
           vectors of the user's shows; the score is the cosine between that
           profile and a pool show.

Both may recommend shows the user already watched, and both fall back to the
popularity list for users without any history.

Every method is scored against the truth built with both tie-breaks:

  recency          count desc, last qualifying airing desc, tv_show_id asc
  first-occurrence count desc, first qualifying airing desc, tv_show_id asc

Run: python src/models.py
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import platform
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import scipy.sparse as sp

import implicit
from implicit.als import AlternatingLeastSquares

from evaluate import map_at_k
from evaluate_v2 import (DATA, SCHEDULE_VALIDATION, SUBMISSION, TOP_K, TRAIN_WEEKS,
                          VALID_WEEKS, candidate_pool, load_airings,
                          pad_with_popularity, qualifying, rank_shows,
                          show_popularity, user_show_counts)

SCHEDULE_TRAIN = DATA / "export_arh_11-20-final.csv"

GENRE_FIELDS = ["tv_show_category", "tv_show_genre_1", "tv_show_genre_2", "tv_show_genre_3"]

ALS_FACTORS = (32, 64, 128)
ALS_REGULARIZATION = 0.05
ALS_ITERATIONS = 20
ALS_RANDOM_STATE = 42
ALS_CONFIDENCE_SCALE = 10.0

EPOCH = pd.Timestamp("1970-01-01")


def section(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


# --------------------------------------------------------------------------- #
# labels and truth
# --------------------------------------------------------------------------- #

def user_show_counts_full(q):
    """Per (user, show): number of qualifying airings, first and last one."""
    return (q.groupby(["user_id", "tv_show_id"])
             .agg(count=("airing_start", "size"),
                  first_qual=("airing_start", "min"),
                  last_qual=("airing_start", "max"))
             .reset_index())


def build_truth(counts, tiebreak):
    """{user: [top 5 shows]} for the users with at least 5 qualifying shows.

    recency          -> count desc, last qualifying airing desc, id asc
    first-occurrence -> count desc, first qualifying airing desc, id asc
    """
    if tiebreak == "recency":
        by, asc = ["count", "last_qual", "tv_show_id"], [False, False, True]
    elif tiebreak == "first-occurrence":
        by, asc = ["count", "first_qual", "tv_show_id"], [False, False, True]
    else:
        raise ValueError(f"unknown tiebreak {tiebreak!r}")

    truth = {}
    for user, grp in counts.groupby("user_id", sort=False):
        if len(grp) < TOP_K:
            continue
        ordered = grp.sort_values(by=by, ascending=asc, kind="stable")
        truth[user] = list(ordered["tv_show_id"].head(TOP_K))
    return truth


# --------------------------------------------------------------------------- #
# show metadata
# --------------------------------------------------------------------------- #

def load_schedule(sources):
    """Read schedule files and collect per-show metadata.

    `sources` is a list of paths; every row of every file contributes.  For
    each genre field the show's most common non-empty value is kept.

    Returns a DataFrame with one row per show and one column per genre field.
    """
    counters = {}

    for path in sources:
        for chunk in pd.read_csv(path, usecols=["tv_show_id"] + GENRE_FIELDS,
                                 dtype=str, keep_default_na=False, chunksize=200_000):
            show = chunk["tv_show_id"].str.strip()
            keep = show.ne("0") & show.ne("")
            chunk = chunk[keep].reset_index(drop=True)
            show = show[keep].reset_index(drop=True)
            if chunk.empty:
                continue

            for field in GENRE_FIELDS:
                values = chunk[field].str.strip()
                sel = values.ne("")
                if not sel.any():
                    continue
                pairs = pd.DataFrame({"s": show[sel].to_numpy(), "v": values[sel].to_numpy()})
                for s, grp in pairs.groupby("s"):
                    counter = counters.setdefault((s, field), Counter())
                    counter.update(grp["v"].to_numpy().tolist())

    meta_rows = {}
    for (s, field), counter in counters.items():
        # most common value; on a tie the alphabetically smallest one
        best = min(counter.items(), key=lambda kv: (-kv[1], kv[0]))
        meta_rows.setdefault(s, {})[field] = best[0]

    meta = pd.DataFrame.from_dict(meta_rows, orient="index")
    meta.index = meta.index.astype("int64")
    meta.index.name = "tv_show_id"
    return meta


def show_vectors(meta, shows):
    """One-hot matrix (len(shows), total genre dimensions) for the given shows.

    Every field contributes its own one-hot block; a show with no value for a
    field contributes an all-zero block.
    """
    vocab = {}
    for field in GENRE_FIELDS:
        values = sorted({v for v in meta[field].dropna().unique()})
        vocab[field] = {v: i for i, v in enumerate(values)}

    offsets, width = {}, 0
    for field in GENRE_FIELDS:
        offsets[field] = width
        width += len(vocab[field])

    vectors = np.zeros((len(shows), width), dtype=np.float64)
    for row, show in enumerate(shows):
        for field in GENRE_FIELDS:
            value = meta.at[show, field] if show in meta.index else None
            if value is None or (isinstance(value, float) and np.isnan(value)):
                continue
            idx = vocab[field].get(value)
            if idx is not None:
                vectors[row, offsets[field] + idx] = 1.0
    return vectors, vocab, offsets


# --------------------------------------------------------------------------- #
# ranking helpers
# --------------------------------------------------------------------------- #

def rank_by_score(scores, show_ids, tie_dates, k=TOP_K):
    """Order shows by score desc, then more recent tie date desc, then id asc.

    `tie_dates` must be a datetime64 array aligned with `show_ids`.
    """
    if len(show_ids) == 0:
        return []
    scores = np.asarray(scores, dtype=np.float64)
    dates = np.asarray(tie_dates, dtype="datetime64[ns]").astype("int64")
    ids = np.asarray(show_ids, dtype=np.int64)
    order = np.lexsort((ids, -dates, -scores))
    return [int(show_ids[i]) for i in order[:k]]


class TieBreak:
    """The tie-break date of a (user, show) pair.

    The user's own last qualifying airing of the show is used when the user
    has one, otherwise the show's last qualifying airing in the training
    weeks.  This is the same rule that the own-history baseline uses, so the
    model based methods break their ties exactly like it does.
    """

    def __init__(self, q_train, pool):
        self.pool_ids = sorted(pool)
        self.pool_pos = {s: i for i, s in enumerate(self.pool_ids)}

        last_train = q_train.groupby("tv_show_id")["airing_start"].max().to_dict()
        self.global_dates = {show: last_train.get(show, EPOCH) for show in pool}
        self.pool_dates = np.array([self.global_dates[s] for s in self.pool_ids],
                                   dtype="datetime64[ns]")

        pairs = (q_train.groupby(["user_id", "tv_show_id"])["airing_start"]
                       .max().reset_index())
        self.own_dates = {}
        for user, grp in pairs.groupby("user_id", sort=False):
            self.own_dates[user] = {int(s): d for s, d in
                                    zip(grp["tv_show_id"], grp["airing_start"])}

    def for_pool(self, user):
        """Date vector aligned with self.pool_ids."""
        dates = self.pool_dates.copy()
        for show, date in self.own_dates.get(user, {}).items():
            i = self.pool_pos.get(show)
            if i is not None:
                dates[i] = np.datetime64(pd.Timestamp(date))
        return dates

    def for_ids(self, user, show_ids):
        """Date vector aligned with the given shows."""
        own = self.own_dates.get(user, {})
        out = []
        for show in show_ids:
            date = own.get(int(show))
            if date is None:
                date = self.global_dates.get(int(show), EPOCH)
            out.append(np.datetime64(pd.Timestamp(date)))
        return np.array(out, dtype="datetime64[ns]")


# --------------------------------------------------------------------------- #
# methods
# --------------------------------------------------------------------------- #

def method_popularity(pool, pop, popularity_top5, users):
    """The same global top-5 list for every user."""
    return {u: list(popularity_top5) for u in users}


def method_own_history(pool, own_counts, popularity_top5, users):
    """The user's own most watched shows, padded with the popularity list."""
    by_user = {u: g for u, g in own_counts.groupby("user_id", sort=False)}
    preds, n_padded = {}, 0
    for user in users:
        grp = by_user.get(user)
        ranked = []
        if grp is not None:
            ranked = rank_shows(grp[grp["tv_show_id"].isin(pool)])
        if len(ranked) < TOP_K:
            n_padded += 1
            ranked = pad_with_popularity(ranked, popularity_top5)
        preds[user] = ranked
    return preds, n_padded


def build_als_matrix(counts, pool):
    """user x show matrix whose cells hold the confidence 1 + 10 * log1p(count).

    The columns cover every show with a training interaction and, so that the
    model can score them, every show in the pool.
    """
    users = sorted(counts["user_id"].unique())
    items = sorted(set(counts["tv_show_id"].unique()) | set(pool))
    uidx = {u: i for i, u in enumerate(users)}
    iidx = {s: i for i, s in enumerate(items)}

    rows = counts["user_id"].map(uidx).to_numpy()
    cols = counts["tv_show_id"].map(iidx).to_numpy()
    values = 1.0 + ALS_CONFIDENCE_SCALE * np.log1p(counts["count"].to_numpy(dtype=np.float64))

    matrix = sp.csr_matrix((values.astype(np.float32), (rows, cols)),
                            shape=(len(users), len(items)), dtype=np.float32)
    return matrix, uidx, iidx


def method_als(counts, pool, tie, popularity_top5, users, factors, depth=TOP_K):
    """Implicit ALS over the user x show count matrix, scored against the pool.

    `depth` controls how many ranked shows are returned per user; the list is
    always padded with popularity up to TOP_K.
    """
    matrix, uidx, iidx = build_als_matrix(counts, pool)

    model = AlternatingLeastSquares(factors=factors, regularization=ALS_REGULARIZATION,
                                    iterations=ALS_ITERATIONS, alpha=1.0,
                                    random_state=ALS_RANDOM_STATE)
    model.fit(matrix, show_progress=False)

    pool_ids = sorted(pool)
    pool_idx = np.array([iidx[s] for s in pool_ids])
    pool_arr = np.array(pool_ids, dtype=np.int64)

    known = set(counts["tv_show_id"].unique())
    n_known = len(set(pool) & known)

    scored_users = [u for u in users if u in uidx]
    preds, n_fallback = {}, 0
    if scored_users:
        U = model.user_factors[[uidx[u] for u in scored_users]]
        S = U @ model.item_factors.T
        for row, user in enumerate(scored_users):
            scores = S[row, pool_idx]
            ranked = rank_by_score(scores, pool_arr, tie.for_pool(user), k=depth)
            if len(ranked) < TOP_K:
                ranked = pad_with_popularity(ranked, popularity_top5)
            preds[user] = ranked

    for user in users:
        if user not in preds:
            n_fallback += 1
            preds[user] = list(popularity_top5)

    info = {"n_users": matrix.shape[0], "n_items": matrix.shape[1], "nnz": matrix.nnz,
            "pool_known": n_known, "pool_unknown": len(pool) - n_known,
            "fallback": n_fallback, "model": model}
    return preds, info


def method_content(counts, pool, meta, tie, popularity_top5, users):
    """Cosine between a genre profile of the user and each pool show."""
    shows = sorted(set(counts["tv_show_id"].unique()) | set(pool))
    vectors, _, _ = show_vectors(meta, shows)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    assert (norms > 0).all(), "shows without any genre information"
    unit = vectors / norms

    sidx = {s: i for i, s in enumerate(shows)}
    users_all = sorted(counts["user_id"].unique())
    uidx = {u: i for i, u in enumerate(users_all)}

    matrix = sp.csr_matrix(
        (counts["count"].to_numpy(dtype=np.float64),
         (counts["user_id"].map(uidx).to_numpy(),
          counts["tv_show_id"].map(sidx).to_numpy())),
        shape=(len(users_all), len(shows)))

    weights = np.asarray(matrix.sum(axis=1)).ravel()
    profiles = matrix @ vectors
    profiles = profiles / np.where(weights > 0, weights, 1.0)[:, None]
    pnorm = np.linalg.norm(profiles, axis=1, keepdims=True)
    profiles = profiles / np.where(pnorm > 0, pnorm, 1.0)

    pool_ids = sorted(pool)
    pool_idx = np.array([sidx[s] for s in pool_ids])
    pool_arr = np.array(pool_ids, dtype=np.int64)
    pool_unit = unit[pool_idx]

    preds, n_fallback = {}, 0
    for user in users:
        if user not in uidx or weights[uidx[user]] == 0:
            n_fallback += 1
            preds[user] = list(popularity_top5)
            continue
        scores = profiles[uidx[user]] @ pool_unit.T
        ranked = rank_by_score(scores, pool_arr, tie.for_pool(user))
        if len(ranked) < TOP_K:
            ranked = pad_with_popularity(ranked, popularity_top5)
        preds[user] = ranked

    info = {"n_shows": len(shows), "n_dim": vectors.shape[1],
            "n_users": len(users_all), "fallback": n_fallback}
    return preds, info


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main():
    started = time.perf_counter()

    section("ENVIRONMENT")
    print(f"python  : {platform.python_version()}  ({sys.executable})")
    print(f"numpy   : {np.__version__}")
    print(f"scipy   : {scipy.__version__}")
    print(f"pandas  : {pd.__version__}")
    print(f"implicit: {implicit.__version__}")
    print(f"train weeks {TRAIN_WEEKS} -> validation weeks {VALID_WEEKS}, top-K = {TOP_K}")

    section("LOADING")
    air = load_airings()
    sub = pd.read_csv(SUBMISSION, dtype=str)
    submission_users = sub["user_id"].astype(str).tolist()
    pool = candidate_pool(VALID_WEEKS, SCHEDULE_VALIDATION)
    print(f"airing rows      : {len(air)}")
    print(f"submission users : {len(submission_users)}")
    print(f"validation pool  : {len(pool)} labelled shows airing in weeks "
          f"{VALID_WEEKS[0]}-{VALID_WEEKS[1]}")

    meta = load_schedule([SCHEDULE_TRAIN, SCHEDULE_VALIDATION])
    print(f"shows with metadata: {len(meta)}")

    section("LABELS (screen_time >= 0.8)")
    q_train = qualifying(air, TRAIN_WEEKS)
    q_val = qualifying(air, VALID_WEEKS)
    train_counts = user_show_counts_full(q_train)
    val_counts = user_show_counts_full(q_val)

    # the counts must agree with the validated helper in evaluate_v2
    reference = user_show_counts(q_train)
    assert len(reference) == len(train_counts)
    merged = reference.merge(train_counts, on=["user_id", "tv_show_id"], suffixes=("_ref", "_new"))
    assert (merged["count_ref"] == merged["count_new"]).all()
    assert (merged["last_qual_ref"] == merged["last_qual_new"]).all()
    print("train counts agree with evaluate_v2.user_show_counts")

    truth_recency = build_truth(val_counts, "recency")
    truth_first = build_truth(val_counts, "first-occurrence")
    users = [u for u in submission_users if u in truth_recency]
    print(f"evaluated users  : {len(users)} "
          f"(submission users with >= {TOP_K} qualifying validation shows)")

    diff = sum(1 for u in users if set(truth_recency[u]) != set(truth_first[u]))
    print(f"users whose truth set differs between the two tie-breaks: {diff}")

    section("TRAIN SIDE (weeks 11-20)")
    pop = show_popularity(q_train)
    pop = pop[pop["tv_show_id"].isin(pool)]
    popularity_top5 = rank_shows(pop)
    print(f"popularity top-5: {popularity_top5}")

    # tie-break dates for the model based methods: the user's own last
    # qualifying airing of the show, otherwise the show's last qualifying
    # airing in the training weeks
    tie = TieBreak(q_train, pool)
    n_from_pool = sum(1 for s in pool
                      if s not in q_train["tv_show_id"].unique())
    print(f"tie-break dates from the training weeks: {len(pool) - n_from_pool} shows, "
          f"without any qualifying airing: {n_from_pool} shows")

    results = {}

    def run(name, fn):
        t0 = time.perf_counter()
        preds, info = fn()
        elapsed = time.perf_counter() - t0
        results[name] = (preds, info, elapsed)
        print(f"  {name:<24s} {elapsed:8.2f}s")
        return preds, info, elapsed

    section("METHODS")
    preds_pop, _, t_pop = run("popularity",
                              lambda: (method_popularity(pool, pop, popularity_top5, users), {}))
    preds_own, n_padded, t_own = run(
        "own history",
        lambda: method_own_history(pool, train_counts, popularity_top5, users))
    print(f"    users needing popularity padding: {n_padded}")

    als_results = {}
    for factors in ALS_FACTORS:
        preds_als, info, t_als = run(
            f"ALS (factors={factors})",
            lambda f=factors: method_als(train_counts, pool, tie, popularity_top5, users, f))
        als_results[factors] = (preds_als, info, t_als)
        print(f"    matrix {info['n_users']} users x {info['n_items']} shows, "
              f"{info['nnz']} cells, pool shows with training data: "
              f"{info['pool_known']} (unknown: {info['pool_unknown']}), "
              f"users falling back to popularity: {info['fallback']}")

    preds_content, c_info, t_content = run(
        "content-based",
        lambda: method_content(train_counts, pool, meta, tie, popularity_top5, users))
    print(f"    {c_info['n_shows']} shows, {c_info['n_dim']} genre dimensions, "
          f"{c_info['n_users']} profiles, "
          f"users falling back to popularity: {c_info['fallback']}")

    section("DIAGNOSTICS")
    known_train = set(train_counts["tv_show_id"].unique())
    pool_known = pool & known_train
    reach_pool = np.mean([len(set(truth_recency[u]) & pool) / TOP_K for u in users])
    reach_known = np.mean([len(set(truth_recency[u]) & pool_known) / TOP_K for u in users])
    print(f"truth shows inside the pool                 : {reach_pool:.2%}")
    print(f"truth shows inside the ALS-scorable subset : {reach_known:.2%}  "
          f"({len(pool_known)} of {len(pool)} pool shows have training interactions)")
    print(f"ceiling of any method limited to that subset: {reach_known:.6f}")
    print("\nmean hits in the top 5 (recency truth):")
    for name, (preds, _, _) in results.items():
        hits = np.mean([len(set(preds[u]) & set(truth_recency[u])) for u in users])
        print(f"  {name:<24s} {hits:5.3f} / {TOP_K}")

    section("RESULTS")
    truths = {tb: {u: truth[u] for u in users} for tb, truth in
              (("recency", truth_recency), ("first-occurrence", truth_first))}
    scores = {}
    for name, (preds, _, _) in results.items():
        scores[name] = {tb: map_at_k(t, preds, TOP_K) for tb, t in truths.items()}
    for factors, (preds, _, _) in als_results.items():
        scores[f"ALS (factors={factors})"] = {
            tb: map_at_k(t, preds, TOP_K) for tb, t in truths.items()}

    header = f"{'method':<24s} {'MAP@5 recency':>15s} {'MAP@5 first-occ':>16s} {'runtime':>9s}"
    print(header)
    print("-" * len(header))

    def row(name, preds, elapsed):
        s = scores[name]
        print(f"{name:<24s} {s['recency']:>15.6f} {s['first-occurrence']:>16.6f} "
              f"{elapsed:>8.2f}s")

    row("popularity", preds_pop, t_pop)
    row("own history", preds_own, t_own)
    best_factors = max(ALS_FACTORS, key=lambda f: scores[f"ALS (factors={f})"]["recency"])
    row(f"ALS (factors={best_factors})", als_results[best_factors][0],
        als_results[best_factors][2])
    row("content-based", preds_content, t_content)

    print("\nall ALS factor settings:")
    print(header)
    print("-" * len(header))
    for factors in ALS_FACTORS:
        row(f"ALS (factors={factors})", als_results[factors][0], als_results[factors][2])

    section("SPOT CHECK")
    u = users[0]
    print(f"user {u}")
    print(f"  truth recency          : {truth_recency[u]}")
    print(f"  truth first-occurrence : {truth_first[u]}")
    print(f"  popularity             : {preds_pop[u]}")
    print(f"  own history            : {preds_own[u]}")
    print(f"  ALS (factors={best_factors})       : {als_results[best_factors][0][u]}")
    print(f"  content-based          : {preds_content[u]}")

    section("DONE")
    print(f"total runtime: {time.perf_counter() - started:.2f}s")


if __name__ == "__main__":
    main()
