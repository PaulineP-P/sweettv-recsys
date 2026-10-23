"""Learning-to-rank hybrid: candidate generation and re-ranking.

Strict split rule of the exam:

  * every model is fitted on the training weeks only
  * the validation weeks are used for selection only

So the ranker is never fitted on the weeks it is evaluated on.  It is fitted
twice over, on two different windows:

  fit        features from weeks 11-15, labels from weeks 16-20
  validation features from weeks 16-20, labels from weeks 21-30

Nothing in the validation run is fitted on weeks 21-30: the ALS, the genre
profiles, the popularity and the candidate lists are all built from weeks
11-20.  The only thing that comes from weeks 21-30 is the schedule (the
number of airings of a show in the target window, a feature and a candidate
filter) and the labels themselves.

Candidates per user:
  * the shows with at least one screen_time >= 0.8 airing from week 11 up to
    the end of the feature window
  * plus the 30 most popular shows
  * only the shows that air in the label window (schedule only)

Features per (user, show):
  count_window        qualifying airings in the 5-week feature window
  count_last2         qualifying airings in the last 2 weeks of it
  count_since11       qualifying airings from week 11 to the window end
  weeks_since_last    weeks between the window end and the last qualifying airing
  mean_screen_time    mean screen time over the airings of the feature window
  user_show_share     share of the user's qualifying airings that are this show
  schedule_window     schedule airings of the show in the feature window
  schedule_target     schedule airings of the show in the label window
  popularity          distinct qualifying users of the show since week 11
  als_score           ALS score, fitted on weeks 11..window end
  content_score       genre cosine, fitted on weeks 11..window end
  show_category       the show category, as a categorical

Label: the relevance of the show in the user's true top 5 of the label window
(5 for the first show, 4 for the second, ... 0 for everything else).  The top 5
follows the official rule: count of qualifying airings descending, then the
more recent last qualifying airing, then the smaller tv_show_id.

Run: python src/ltr.py
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import platform
import sys
import time
from pathlib import Path

import lightgbm
import numpy as np
import pandas as pd
import scipy.sparse as sp
from implicit.als import AlternatingLeastSquares
from lightgbm import LGBMRanker

from evaluate import ap_at_k, map_at_k
from evaluate_v2 import (SCHEDULE_VALIDATION, SUBMISSION, TOP_K, TRAIN_WEEKS,
                          VALID_WEEKS, candidate_pool, load_airings,
                          pad_with_popularity, qualifying, rank_shows, show_popularity)
from hybrid import (ALS_DEPTH, ALS_FACTORS_FOR_HYBRID, build_scores, method_scored,
                    schedule_airing_counts, weighted_counts)
from models import (ALS_FACTORS, ALS_ITERATIONS, ALS_RANDOM_STATE,
                    ALS_REGULARIZATION, SCHEDULE_TRAIN, TieBreak, build_truth,
                    load_schedule, method_als, method_content, method_own_history,
                    method_popularity, rank_by_score, show_vectors,
                    user_show_counts_full)

HISTORY_START_WEEK = 11
TOP_POPULAR_CANDIDATES = 30

FIT_FEATURE_WEEKS = (11, 15)
FIT_LABEL_WEEKS = (16, 20)
VAL_FEATURE_WEEKS = (16, 20)
VAL_LABEL_WEEKS = (21, 30)

FEATURES = ["count_window", "count_last2", "count_since11", "weeks_since_last",
            "mean_screen_time", "user_show_share", "schedule_window",
            "schedule_target", "popularity", "als_score", "content_score",
            "show_category"]
CATEGORICAL = ["show_category"]

LGB_PARAMS = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "n_estimators": 400,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 20,
    "subsample": 0.9,
    "subsample_freq": 1,
    "colsample_bytree": 0.9,
    "random_state": 42,
    "n_jobs": 1,
    "verbose": -1,
}

BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_SEED = 42

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
VAL_PREDICTIONS_CSV = OUTPUTS / "val_ltr_predictions.csv"
VAL_TRUTH_CSV = OUTPUTS / "val_truth.csv"

# feature ablations: name -> features
ABLATIONS = {
    "all features": FEATURES,
    "no user_show_share": [f for f in FEATURES if f != "user_show_share"],
    "no popularity": [f for f in FEATURES if f != "popularity"],
    "no ALS + content": [f for f in FEATURES
                          if f not in ("als_score", "content_score")],
    "no schedule": [f for f in FEATURES
                    if f not in ("schedule_window", "schedule_target")],
    "history counts only": ["count_window", "count_last2", "count_since11",
                            "weeks_since_last", "mean_screen_time",
                            "user_show_share"],
    "no history counts": [f for f in FEATURES
                          if f not in ("count_window", "count_last2",
                                      "count_since11", "weeks_since_last",
                                      "mean_screen_time", "user_show_share")],
}


def section(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def write_rankings(path, users, rankings):
    """user_id + the shows in rank order, space separated, one row per user."""
    frame = pd.DataFrame({
        "user_id": [str(u) for u in users],
        "tv_show_id": [" ".join(str(s) for s in rankings[u]) for u in users],
    })
    frame.to_csv(path, index=False, lineterminator="\n")
    with open(path, "rb") as fh:
        blob = fh.read()
    if blob.endswith(b"\n"):
        with open(path, "wb") as fh:
            fh.write(blob[:-1])


def read_rankings(path):
    """{user: [shows]} from a file written by write_rankings."""
    frame = pd.read_csv(path, dtype=str)
    return {u: [int(s) for s in row.split()]
            for u, row in zip(frame["user_id"], frame["tv_show_id"])}


# --------------------------------------------------------------------------- #
# schedule
# --------------------------------------------------------------------------- #

def schedule_table(paths, weeks):
    """{show: (number of airings, last airing)} inside the given ISO weeks."""
    out = {}
    for path in paths:
        for show, (n, last) in schedule_airing_counts(path, weeks).items():
            key = int(show)
            if key in out:
                out[key] = (out[key][0] + n, max(out[key][1], last))
            else:
                out[key] = (n, last)
    return out


# --------------------------------------------------------------------------- #
# labels
# --------------------------------------------------------------------------- #

def build_labels(air, label_weeks):
    """{user: {show: relevance}} and {user: [shows]} of the true top 5.

    The top 5 follows the official rule: count of qualifying airings
    descending, then the more recent last qualifying airing, then the smaller
    tv_show_id.  The relevance is 5 for the first show, 4 for the second and
    so on.
    """
    q = qualifying(air, label_weeks)
    counts = (q.groupby(["user_id", "tv_show_id"])
                .agg(count=("screen_time", "size"),
                     last_qual=("airing_start", "max"))
                .reset_index())

    labels, truth = {}, {}
    for user, grp in counts.groupby("user_id", sort=False):
        ordered = grp.sort_values(by=["count", "last_qual", "tv_show_id"],
                                  ascending=[False, False, True], kind="stable")
        top = ordered.head(TOP_K)
        shows = [int(s) for s in top["tv_show_id"]]
        truth[user] = shows
        labels[user] = {show: TOP_K - i for i, show in enumerate(shows)}
    return labels, truth


# --------------------------------------------------------------------------- #
# ALS and content scores
# --------------------------------------------------------------------------- #

class Scorers:
    """ALS and genre-cosine scores, both fitted on weeks 11..window end."""

    def __init__(self, counts, pool, meta, factors=ALS_FACTORS_FOR_HYBRID):
        shows = sorted(set(counts["tv_show_id"].unique()) | set(pool))
        users = sorted(counts["user_id"].unique())
        self.sidx = {s: i for i, s in enumerate(shows)}
        self.pidx = {u: i for i, u in enumerate(users)}

        matrix, uidx, iidx = self._als_matrix(counts, shows, users)
        model = AlternatingLeastSquares(factors=factors,
                                        regularization=ALS_REGULARIZATION,
                                        iterations=ALS_ITERATIONS, alpha=1.0,
                                        random_state=ALS_RANDOM_STATE)
        model.fit(matrix, show_progress=False)
        self.model = model
        self.uidx, self.iidx = uidx, iidx

        vectors, _, _ = show_vectors(meta, shows)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        self.unit = vectors / np.where(norms > 0, norms, 1.0)

        weights = np.asarray(matrix.sum(axis=1)).ravel()
        profiles = matrix @ vectors
        profiles = profiles / np.where(weights > 0, weights, 1.0)[:, None]
        pnorm = np.linalg.norm(profiles, axis=1, keepdims=True)
        self.profiles = profiles / np.where(pnorm > 0, pnorm, 1.0)

    @staticmethod
    def _als_matrix(counts, shows, users):
        uidx = {u: i for i, u in enumerate(users)}
        iidx = {s: i for i, s in enumerate(shows)}
        rows = counts["user_id"].map(uidx).to_numpy()
        cols = counts["tv_show_id"].map(iidx).to_numpy()
        values = 1.0 + 10.0 * np.log1p(counts["count"].to_numpy(dtype=np.float64))
        matrix = sp.csr_matrix((values.astype(np.float32), (rows, cols)),
                                shape=(len(users), len(shows)), dtype=np.float32)
        return matrix, uidx, iidx

    def _dot(self, user_ids, show_ids, left, right, left_idx, right_idx):
        li = pd.Series(user_ids).map(left_idx).to_numpy(dtype=np.float64)
        ri = pd.Series(show_ids).map(right_idx).to_numpy(dtype=np.float64)
        ok = ~np.isnan(li) & ~np.isnan(ri)
        out = np.zeros(len(li))
        out[ok] = np.einsum("ij,ij->i", left[li[ok].astype(int)],
                            right[ri[ok].astype(int)])
        return out

    def als_score(self, user_ids, show_ids):
        return self._dot(user_ids, show_ids, self.model.user_factors,
                         self.model.item_factors, self.uidx, self.iidx)

    def content_score(self, user_ids, show_ids):
        return self._dot(user_ids, show_ids, self.profiles, self.unit,
                         self.pidx, self.sidx)


# --------------------------------------------------------------------------- #
# dataset
# --------------------------------------------------------------------------- #

def build_dataset(air, meta, feature_weeks, label_weeks, sched_feature, sched_target,
                  target_shows, popularity_top, users, scorers, labels, categories):
    """One row per (user, show) candidate, with the features and the label.

    `users` are the users the dataset is built for; `labels` maps a user to
    the graded relevance of its shows in the label window.
    """
    w0, w1 = feature_weeks

    hist = air[(air["week"] >= HISTORY_START_WEEK) & (air["week"] <= w1)]
    q_hist = qualifying(air, (HISTORY_START_WEEK, w1))
    q_feat = qualifying(air, feature_weeks)
    all_feat = air[(air["week"] >= w0) & (air["week"] <= w1)]

    agg = (q_hist.groupby(["user_id", "tv_show_id"])
                  .agg(count_since11=("screen_time", "size"),
                       last_week=("week", "max"))
                  .reset_index())
    win = (q_feat.groupby(["user_id", "tv_show_id"])
                   .agg(count_window=("screen_time", "size"))
                   .reset_index())
    last2 = (q_feat[q_feat["week"] >= w1 - 1]
             .groupby(["user_id", "tv_show_id"]).size()
             .rename("count_last2").reset_index())
    mean_st = (all_feat.groupby(["user_id", "tv_show_id"])["screen_time"].mean()
                    .rename("mean_screen_time").reset_index())

    # candidates: the user's own shows plus the popular ones
    own = agg[["user_id", "tv_show_id"]]
    popular = [s for s in popularity_top if s in target_shows]
    left = pd.DataFrame({"user_id": users})
    left["_k"] = 1
    right = pd.DataFrame({"tv_show_id": popular})
    right["_k"] = 1
    cross = left.merge(right, on="_k").drop(columns="_k")
    pairs = pd.concat([own, cross], ignore_index=True).drop_duplicates(
        subset=["user_id", "tv_show_id"])
    pairs = pairs[pairs["user_id"].isin(set(users))]
    pairs = pairs[pairs["tv_show_id"].isin(target_shows)].reset_index(drop=True)

    out = pairs.merge(agg, on=["user_id", "tv_show_id"], how="left")
    out = out.merge(win, on=["user_id", "tv_show_id"], how="left")
    out = out.merge(last2, on=["user_id", "tv_show_id"], how="left")
    out = out.merge(mean_st, on=["user_id", "tv_show_id"], how="left")
    for col in ("count_since11", "last_week", "count_window", "count_last2",
                "mean_screen_time"):
        out[col] = out[col].fillna(0)

    # weeks since the last qualifying airing of the show
    out["weeks_since_last"] = np.where(out["last_week"] > 0,
                                       w1 - out["last_week"], 99)
    out = out.drop(columns=["last_week"])

    # share of the user's qualifying airings that are this show
    total = q_hist.groupby("user_id").size().rename("user_qual_total")
    out = out.merge(total, left_on="user_id", right_index=True, how="left")
    out["user_qual_total"] = out["user_qual_total"].fillna(0)
    out["user_show_share"] = np.where(out["user_qual_total"] > 0,
                                      out["count_since11"] / out["user_qual_total"], 0.0)

    # schedule side
    out["schedule_window"] = out["tv_show_id"].map(
        lambda s: sched_feature.get(int(s), (0, None))[0])
    out["schedule_target"] = out["tv_show_id"].map(
        lambda s: sched_target.get(int(s), (0, None))[0])

    # popularity since week 11
    pop = (q_hist.groupby("tv_show_id")["user_id"].nunique()
           .rename("popularity").reset_index())
    out = out.merge(pop, on="tv_show_id", how="left")
    out["popularity"] = out["popularity"].fillna(0)

    # model scores
    out["als_score"] = scorers.als_score(out["user_id"].to_numpy(),
                                          out["tv_show_id"].to_numpy())
    out["content_score"] = scorers.content_score(out["user_id"].to_numpy(),
                                                  out["tv_show_id"].to_numpy())

    # show category
    out["show_category"] = out["tv_show_id"].map(
        lambda s: meta.at[int(s), "tv_show_category"]
        if int(s) in meta.index and isinstance(meta.at[int(s), "tv_show_category"], str)
        else "unknown")
    out["show_category"] = pd.Categorical(out["show_category"], categories=categories)

    # label
    out["label"] = 0
    for user, grp in out.groupby("user_id", sort=False):
        relevance = labels.get(user, {})
        if relevance:
            out.loc[grp.index, "label"] = [relevance.get(int(s), 0)
                                           for s in grp["tv_show_id"]]

    return out.sort_values(["user_id", "tv_show_id"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# training and prediction
# --------------------------------------------------------------------------- #

def fit_ranker(train_data, features, categorical=None):
    """Fit the ranker on the training window.

    Only the users with at least one positive label take part in a group.
    """
    categorical = [c for c in (categorical or CATEGORICAL) if c in features]
    keep = train_data[train_data.groupby("user_id")["label"].transform("sum") > 0]
    keep = keep.sort_values(["user_id", "tv_show_id"]).reset_index(drop=True)
    groups = keep.groupby("user_id", sort=False).size().to_numpy()
    model = LGBMRanker(**LGB_PARAMS)
    model.fit(keep[features], keep["label"].to_numpy(), group=groups,
              categorical_feature=categorical)
    return model


def predict_topk(model, val_data, features, users, tie, popularity_top5,
                 categorical=None):
    """The top 5 of every user, ranked by the ranker score."""
    categorical = [c for c in (categorical or CATEGORICAL) if c in features]
    data = val_data.sort_values(["user_id", "tv_show_id"]).reset_index(drop=True)
    data["ranker_score"] = model.predict(data[features],
                                         categorical_feature=categorical)
    by_user = {u: grp for u, grp in data.groupby("user_id", sort=False)}
    preds = {}
    for user in users:
        grp = by_user.get(user)
        ranked = []
        if grp is not None and len(grp):
            show_ids = grp["tv_show_id"].to_numpy()
            ranked = rank_by_score(grp["ranker_score"].to_numpy(), show_ids,
                                   tie.for_ids(user, show_ids), k=TOP_K)
        preds[user] = pad_with_popularity(ranked, popularity_top5)
    return preds, data


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main():
    started = time.perf_counter()

    section("ENVIRONMENT")
    print(f"python  : {platform.python_version()}  ({sys.executable})")
    print(f"numpy   : {np.__version__}")
    print(f"pandas  : {pd.__version__}")
    print(f"lightgbm: {lightgbm.__version__}")
    print(f"fit      : features {FIT_FEATURE_WEEKS}, labels {FIT_LABEL_WEEKS}")
    print(f"validate : features {VAL_FEATURE_WEEKS}, labels {VAL_LABEL_WEEKS}")
    print(f"LightGBM parameters: {LGB_PARAMS}")

    section("LOADING")
    air = load_airings()
    sub = pd.read_csv(SUBMISSION, dtype=str)
    submission_users = sub["user_id"].astype(str).tolist()
    meta = load_schedule([SCHEDULE_TRAIN, SCHEDULE_VALIDATION])
    categories = sorted({v for v in meta["tv_show_category"].dropna().unique()
                         if isinstance(v, str)} | {"unknown"})
    print(f"airing rows       : {len(air)}")
    print(f"submission users  : {len(submission_users)}")
    print(f"shows with metadata: {len(meta)}, categories: {len(categories)}")

    # ----------------------------------------------------------------------- #
    # the truth of the validation window
    # ----------------------------------------------------------------------- #

    q_train = qualifying(air, TRAIN_WEEKS)
    q_val = qualifying(air, VALID_WEEKS)
    val_counts = user_show_counts_full(q_val)
    truth_recency = build_truth(val_counts, "recency")
    truth_first = build_truth(val_counts, "first-occurrence")
    users = [u for u in submission_users if u in truth_recency]
    truths = {"recency": {u: truth_recency[u] for u in users},
              "first-occurrence": {u: truth_first[u] for u in users}}
    pool = candidate_pool(VALID_WEEKS, SCHEDULE_VALIDATION)
    print(f"evaluated users   : {len(users)} "
          f"(submission users with >= {TOP_K} qualifying validation shows)")
    print(f"validation pool   : {len(pool)} shows airing in weeks {VALID_WEEKS[0]}-{VALID_WEEKS[1]}")

    # ----------------------------------------------------------------------- #
    # the fit setup: features 11-15, labels 16-20
    # ----------------------------------------------------------------------- #

    section("FIT SETUP (features 11-15, labels 16-20)")
    t0 = time.perf_counter()
    fit_target_shows = set(schedule_table([SCHEDULE_TRAIN], FIT_LABEL_WEEKS))
    fit_sched_feature = schedule_table([SCHEDULE_TRAIN], FIT_FEATURE_WEEKS)
    fit_sched_target = schedule_table([SCHEDULE_TRAIN], FIT_LABEL_WEEKS)
    fit_labels, fit_truth = build_labels(air, FIT_LABEL_WEEKS)

    fit_hist = qualifying(air, (HISTORY_START_WEEK, FIT_FEATURE_WEEKS[1]))
    fit_counts = user_show_counts_full(fit_hist)
    fit_pop = show_popularity(fit_hist)
    fit_pop_top = rank_shows(fit_pop, k=TOP_POPULAR_CANDIDATES)
    fit_scorers = Scorers(fit_counts, fit_target_shows, meta)

    fit_users = sorted(fit_truth)
    fit_data = build_dataset(air, meta, FIT_FEATURE_WEEKS, FIT_LABEL_WEEKS,
                              fit_sched_feature, fit_sched_target, fit_target_shows,
                              fit_pop_top, fit_users, fit_scorers, fit_labels, categories)
    t_fit_data = time.perf_counter() - t0
    print(f"users with a label window truth : {len(fit_users)}")
    print(f"popular shows (candidates)       : {fit_pop_top}")
    print(f"candidate rows                   : {len(fit_data)} "
          f"({fit_data['user_id'].nunique()} users, "
          f"{fit_data.groupby('user_id').size().mean():.1f} per user)")
    print(f"positive rows (label > 0)        : {(fit_data['label'] > 0).sum()}")
    print(f"build time                       : {t_fit_data:.2f}s")

    # ----------------------------------------------------------------------- #
    # the validation setup: features 16-20, labels 21-30
    # ----------------------------------------------------------------------- #

    section("VALIDATION SETUP (features 16-20, labels 21-30)")
    t0 = time.perf_counter()
    val_sched_feature = schedule_table([SCHEDULE_TRAIN], VAL_FEATURE_WEEKS)
    val_sched_target = schedule_table([SCHEDULE_VALIDATION], VAL_LABEL_WEEKS)
    val_labels, val_truth = build_labels(air, VAL_LABEL_WEEKS)

    val_counts_hist = user_show_counts_full(q_train)
    val_pop = show_popularity(q_train)
    val_pop_top = rank_shows(val_pop, k=TOP_POPULAR_CANDIDATES)
    val_scorers = Scorers(val_counts_hist, pool, meta)

    val_data = build_dataset(air, meta, VAL_FEATURE_WEEKS, VAL_LABEL_WEEKS,
                             val_sched_feature, val_sched_target, pool,
                             val_pop_top, users, val_scorers, val_labels, categories)
    t_val_data = time.perf_counter() - t0
    print(f"popular shows (candidates)       : {val_pop_top}")
    print(f"candidate rows                   : {len(val_data)} "
          f"({val_data['user_id'].nunique()} users, "
          f"{val_data.groupby('user_id').size().mean():.1f} per user)")
    print(f"build time                       : {t_val_data:.2f}s")
    print("\nnothing above was fitted on weeks 21-30: the ALS, the genre profiles,")
    print("the popularity and the candidates are all built from weeks 11-20.")

    # how much of the truth is reachable inside the candidate set?
    reachable, ceiling = [], []
    for u in users:
        truth_shows = set(val_truth.get(u, []))
        cand = set(val_data[val_data["user_id"] == u]["tv_show_id"])
        found = len(truth_shows & cand)
        reachable.append(found / max(len(truth_shows), 1))
        ceiling.append(found / min(len(truth_shows), TOP_K))
    print(f"\ncandidate recall of the truth top 5 : {np.mean(reachable):.4f}")
    print(f"ceiling MAP@5 of the candidate set   : {np.mean(ceiling):.6f}")

    # ----------------------------------------------------------------------- #
    # training
    # ----------------------------------------------------------------------- #

    section("TRAINING THE RANKER")
    keep = fit_data[fit_data.groupby("user_id")["label"].transform("sum") > 0]
    keep = keep.sort_values(["user_id", "tv_show_id"]).reset_index(drop=True)
    groups = keep.groupby("user_id", sort=False).size().to_numpy()
    y = keep["label"].to_numpy()
    print(f"training groups (users) : {len(groups)}")
    print(f"training rows           : {len(keep)}")
    print(f"label distribution       : {dict(pd.Series(y).value_counts().sort_index())}")

    t0 = time.perf_counter()
    ranker = fit_ranker(fit_data, FEATURES)
    t_fit = time.perf_counter() - t0
    print(f"fit time                : {t_fit:.2f}s")

    # ----------------------------------------------------------------------- #
    # prediction
    # ----------------------------------------------------------------------- #

    section("PREDICTION")
    tie = TieBreak(q_train, pool)
    popularity_top5 = rank_shows(val_pop[val_pop["tv_show_id"].isin(pool)])

    t0 = time.perf_counter()
    preds_ranker, val_data = predict_topk(ranker, val_data, FEATURES, users,
                                          tie, popularity_top5)
    t_predict = time.perf_counter() - t0
    by_user = {u: grp for u, grp in val_data.groupby("user_id", sort=False)}
    print(f"predict time            : {t_predict:.2f}s")
    sizes = val_data.groupby("user_id").size()
    print(f"candidates per user     : min {sizes.min()} median {sizes.median():.0f} "
          f"mean {sizes.mean():.1f} max {sizes.max()}")
    print(f"users padded with popularity: "
          f"{sum(1 for u in users if len(by_user.get(u, [])) < TOP_K)}")

    # ----------------------------------------------------------------------- #
    # baselines on the same split
    # ----------------------------------------------------------------------- #

    section("BASELINES (same split: features 11-20, labels 21-30)")
    preds_pop = method_popularity(pool, val_pop, popularity_top5, users)
    preds_own, _ = method_own_history(pool, val_counts_hist, popularity_top5, users)
    preds_als, als_info = method_als(val_counts_hist, pool, tie, popularity_top5,
                                      users, ALS_FACTORS_FOR_HYBRID, depth=ALS_DEPTH)
    preds_content, _ = method_content(val_counts_hist, pool, meta, tie,
                                       popularity_top5, users)

    n_train_air = {int(s): n for s, (n, _) in
                   schedule_airing_counts(SCHEDULE_TRAIN, TRAIN_WEEKS).items()}
    n_target_air = {int(s): n for s, (n, _) in
                    schedule_airing_counts(SCHEDULE_VALIDATION, VALID_WEEKS).items()}
    weighted = weighted_counts(q_train, 8.0, TRAIN_WEEKS[1])
    count_scores = build_scores(weighted, n_train_air, n_target_air, pool, False)
    preds_v3 = method_scored(count_scores, tie, popularity_top5, users,
                             als_preds=preds_als)

    # ----------------------------------------------------------------------- #
    # results
    # ----------------------------------------------------------------------- #

    section("RESULTS")
    rows = [
        ("popularity", preds_pop),
        ("own history", preds_own),
        (f"recency count H=8 + ALS (v3)", preds_v3),
        (f"ALS (factors={ALS_FACTORS_FOR_HYBRID})", preds_als),
        ("content-based", preds_content),
        ("LTR ranker", preds_ranker),
    ]
    header = (f"{'method':<32s} {'MAP@5 recency':>15s} {'MAP@5 first-occ':>16s} "
              f"{'hits@5':>8s}")
    print(header)
    print("-" * len(header))
    for name, preds in rows:
        s_rec = map_at_k(truths["recency"], preds, TOP_K)
        s_first = map_at_k(truths["first-occurrence"], preds, TOP_K)
        hits = np.mean([len(set(preds[u][:TOP_K]) & set(truths["recency"][u]))
                        for u in users])
        print(f"{name:<32s} {s_rec:>15.6f} {s_first:>16.6f} {hits:>8.3f}")

    section("WHERE THE HITS LAND (recency truth, first 5 only)")
    for name, preds in (("own history", preds_own),
                        ("ALS", preds_als),
                        ("LTR ranker", preds_ranker)):
        positions, p1 = [], 0
        for u in users:
            truth = set(truths["recency"][u])
            at = [rank for rank, show in enumerate(preds[u][:TOP_K], start=1)
                  if show in truth]
            if at:
                positions.extend(at)
                p1 += at[0] == 1
        print(f"  {name:<12s} P@1 hit rate {p1 / len(users):.3f}  "
              f"hits per user {len(positions) / len(users):.3f}  "
              f"mean rank of a hit {np.mean(positions):.2f}")

    # ----------------------------------------------------------------------- #
    # how solid is the gain over the v3 hybrid?
    # ----------------------------------------------------------------------- #

    section("PAIRED BOOTSTRAP: LTR ranker against the other methods")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n_users = len(users)
    resamples = rng.integers(0, n_users, size=(BOOTSTRAP_RESAMPLES, n_users))
    opponents = [("v3 (H=8 + ALS)", preds_v3), ("own history", preds_own),
                 ("popularity", preds_pop)]
    for truth_name in ("recency", "first-occurrence"):
        truth = truths[truth_name]
        ap_ltr = np.array([ap_at_k(truth[u], preds_ranker[u], TOP_K)
                           for u in users])
        print(f"\ntruth: {truth_name}")
        for opp_name, opp_preds in opponents:
            ap_opp = np.array([ap_at_k(truth[u], opp_preds[u], TOP_K)
                               for u in users])
            observed = ap_ltr.mean() - ap_opp.mean()
            draws = ap_ltr[resamples].mean(axis=1) - ap_opp[resamples].mean(axis=1)
            lo, hi = np.percentile(draws, [2.5, 97.5])
            better = int((ap_ltr > ap_opp).sum())
            worse = int((ap_ltr < ap_opp).sum())
            print(f"  vs {opp_name:<16s} difference {observed:+.6f}  "
                  f"95% CI [{lo:+.6f}, {hi:+.6f}]  "
                  f"P(better) {(draws > 0).mean():.3f}  "
                  f"users better/worse {better}/{worse}")
    print(f"\n({BOOTSTRAP_RESAMPLES} resamples of {n_users} users, seed {BOOTSTRAP_SEED})")

    # ----------------------------------------------------------------------- #
    # which features carry the gain?
    # ----------------------------------------------------------------------- #

    section("FEATURE ABLATION (same split, same seed)")
    header = (f"{'variant':<22s} {'MAP@5 recency':>15s} {'MAP@5 first-occ':>16s} "
              f"{'hits@5':>8s} {'delta recency':>14s}")
    print(header)
    print("-" * len(header))
    full_rec = map_at_k(truths["recency"], preds_ranker, TOP_K)
    for name, features in ABLATIONS.items():
        t_abl = time.perf_counter()
        model = fit_ranker(fit_data, features)
        preds, _ = predict_topk(model, val_data, features, users, tie,
                                popularity_top5)
        s_rec = map_at_k(truths["recency"], preds, TOP_K)
        s_first = map_at_k(truths["first-occurrence"], preds, TOP_K)
        hits = np.mean([len(set(preds[u][:TOP_K]) & set(truths["recency"][u]))
                        for u in users])
        print(f"{name:<22s} {s_rec:>15.6f} {s_first:>16.6f} {hits:>8.3f} "
              f"{s_rec - full_rec:>+14.6f}")
        if name == "all features":
            print(f"{'':<22s} (each ablation is refitted and re-scored, "
                  f"{time.perf_counter() - t_abl:.1f}s for this row)")

    # ----------------------------------------------------------------------- #
    # save the validation ranking and the truth so MAP@5 can be recomputed
    # ----------------------------------------------------------------------- #

    section("VALIDATION FILES")
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    write_rankings(VAL_PREDICTIONS_CSV, users, preds_ranker)
    write_rankings(VAL_TRUTH_CSV, users, truths["recency"])

    back_pred = read_rankings(VAL_PREDICTIONS_CSV)
    back_truth = read_rankings(VAL_TRUTH_CSV)
    assert list(back_pred) == list(users), "prediction user order mismatch"
    assert list(back_truth) == list(users), "truth user order mismatch"
    assert all(len(v) == TOP_K for v in back_pred.values()), "prediction row length"
    assert all(len(v) == TOP_K for v in back_truth.values()), "truth row length"
    recomputed = map_at_k({u: set(back_truth[u]) for u in back_truth}, back_pred, TOP_K)
    print(f"wrote {VAL_PREDICTIONS_CSV}  ({len(back_pred)} users)")
    print(f"wrote {VAL_TRUTH_CSV}  ({len(back_truth)} users)")
    print(f"truth tie-break in the file : recency")
    print(f"MAP@5 recomputed from the two files : {recomputed:.6f}")
    print(f"MAP@5 in memory                     : {full_rec:.6f}")

    section("FEATURE IMPORTANCES (gain)")
    gains = ranker.booster_.feature_importance(importance_type="gain")
    order = np.argsort(-gains)
    total = gains.sum()
    for i in order:
        print(f"  {FEATURES[i]:<20s} {gains[i]:>12.1f}  {gains[i] / total:>7.2%}")

    section("RUNTIME")
    print(f"fit setup build     : {t_fit_data:.2f}s")
    print(f"validation build    : {t_val_data:.2f}s")
    print(f"ranker fit          : {t_fit:.2f}s")
    print(f"ranker predict      : {t_predict:.2f}s")
    print(f"total               : {time.perf_counter() - started:.2f}s")
    print(f"lightgbm version    : {lightgbm.__version__}")

    section("SPOT CHECK")
    u = users[0]
    print(f"user {u}")
    print(f"  truth recency    : {truths['recency'][u]}")
    print(f"  own history      : {preds_own[u]}")
    print(f"  v3 (H=8 + ALS)   : {preds_v3[u]}")
    print(f"  LTR ranker       : {preds_ranker[u]}")
    shown = val_data[val_data["user_id"] == u].sort_values("ranker_score", ascending=False)
    print(f"  top candidates by the ranker:")
    for _, row in shown.head(8).iterrows():
        print(f"    {int(row['tv_show_id']):>10}  score {row['ranker_score']:9.4f}  "
              f"label {int(row['label'])}  cnt_window {row['count_window']:.0f}  "
              f"cnt_since11 {row['count_since11']:.0f}  pop {row['popularity']:.0f}")


if __name__ == "__main__":
    main()
