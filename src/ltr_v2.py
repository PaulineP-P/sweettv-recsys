"""LTR v2: four improvements, each measured on the validation window.

Strict rules, unchanged:
  * fit only on weeks up to 20
  * validate on weeks 21-30
  * nothing is ever fitted on weeks 21-30

Baseline (v5): the ranker of src/ltr.py -- features 11-15, labels 16-20,
validate features 16-20, labels 21-30, 12 features, own history + top-30
popular candidates, LightGBM lambdarank, num_leaves 31, 400 trees,
learning_rate 0.05, seed 42.

Changes, each compared with the baseline on the validation window:

  A  wider candidates: ALS top-30, content top-30, popular top-100 and the
     top-20 target-window shows on the user's 3 most-watched channels, on top
     of the own history.  The training side uses the same recipe with the fit
     window.
  B  six new features:
     channel_watch_share  share of the user's watch time on the show's channel
     hour_overlap         overlap of the show's target-window hours with the
                          hours the user usually watches
     distinct_weeks       distinct weeks with a >= 0.8 airing
     partial_views        airings with 0.3 <= screen_time < 0.8
     trend                qualifying count in the last 2 weeks / whole window
     target_ratio         target-window airings / feature-window airings
  C  more training groups: the windows 11-15 -> 16-20, 11-14 -> 15-19 and
     11-13 -> 14-18 are stacked, each with its own ALS, content model,
     schedule tables and candidate lists (all ending by week 20).
  D  a small grid: num_leaves in {31, 63} x n_estimators in {300, 600},
     learning_rate 0.05, seed 42, picked on the validation window.

A change is kept only when it raises MAP@5 on the recency truth.  The final
recipe combines the kept changes and uses the best grid setting.

Run: python src/ltr_v2.py
"""

import gc
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import platform
import sys
import time
from pathlib import Path

import lightgbm
import numpy as np
import pandas as pd

from evaluate import ap_at_k, map_at_k
from evaluate_v2 import (SCHEDULE_VALIDATION, SUBMISSION, TOP_K, TRAIN_WEEKS,
                          VALID_WEEKS, candidate_pool, load_airings,
                          pad_with_popularity, qualifying, rank_shows,
                          show_popularity)
from ltr import FEATURES as FEATURES_BASE
from ltr import (LGB_PARAMS, Scorers, build_labels, read_rankings,
                 write_rankings)
from models import SCHEDULE_TRAIN, TieBreak, load_schedule, rank_by_score, \
    user_show_counts_full

OUTPUTS = Path(__file__).resolve().parents[1] / "outputs"
VAL_V5_CSV = OUTPUTS / "val_ltr_predictions.csv"
VAL_V6_CSV = OUTPUTS / "val_ltr_v6_predictions.csv"

HISTORY_START_WEEK = 11
POPULAR_BASE = 30
POPULAR_WIDE = 100
TOP_ALS = 30
TOP_CONTENT = 30
TOP_CHANNEL = 20
N_CHANNELS = 3

FEATURES_EXTRA = ["channel_watch_share", "hour_overlap", "distinct_weeks",
                  "partial_views", "trend", "target_ratio"]
CATEGORICAL = ["show_category"]

GRID = [(31, 300), (31, 600), (63, 300), (63, 600)]

BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_SEED = 42

VAL_FEATURE_WEEKS = (16, 20)
VAL_LABEL_WEEKS = (21, 30)
WINDOWS_STACK = [(11, 15, 16, 20), (11, 14, 15, 19), (11, 13, 14, 18)]


def section(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


# --------------------------------------------------------------------------- #
# schedule frames: (show, channel, week, hour) for every airing
# --------------------------------------------------------------------------- #

def load_schedule_frame(path):
    frame = pd.read_csv(path, usecols=["channel_id", "tv_show_id", "start_time"],
                        dtype=str, keep_default_na=False)
    show = frame["tv_show_id"].str.strip()
    keep = show.ne("0") & show.ne("")
    frame, show = frame[keep], show[keep]
    start = pd.to_datetime(frame["start_time"], format="%d.%m.%Y %H:%M:%S",
                           errors="coerce")
    ok = start.notna()
    return pd.DataFrame({
        "show": show[ok].astype("int64").to_numpy(),
        "channel": pd.to_numeric(frame["channel_id"][ok], errors="coerce")
                     .fillna(-1).astype("int32").to_numpy(),
        "week": start[ok].dt.isocalendar().week.astype("int16").to_numpy(),
        "hour": start[ok].dt.hour.astype("int8").to_numpy(),
    })


def sched_counts(frame, weeks):
    inside = (frame["week"] >= weeks[0]) & (frame["week"] <= weeks[1])
    return frame[inside].groupby("show").size().to_dict()


def sched_hour_fractions(frame, weeks, shows):
    inside = (frame["week"] >= weeks[0]) & (frame["week"] <= weeks[1])
    sub = frame[inside]
    sub = sub[sub["show"].isin(shows)]
    counts = sub.groupby("show").size()
    out = {int(s): np.zeros(24) for s in shows}
    for (show, hour), n in sub.groupby(["show", "hour"]).size().items():
        out[int(show)][int(hour)] = n
    for show, arr in out.items():
        total = arr.sum()
        if total:
            out[show] = arr / total
    return out


def sched_channel_lists(frame, weeks):
    """{channel: [(show, number of target-window airings), ...]}, best first."""
    inside = (frame["week"] >= weeks[0]) & (frame["week"] <= weeks[1])
    sub = frame[inside]
    out = {}
    for (channel, show), n in sub.groupby(["channel", "show"]).size().items():
        out.setdefault(int(channel), []).append((int(show), int(n)))
    for channel in out:
        out[channel].sort(key=lambda item: (-item[1], item[0]))
    return out


def show_channel_map(*frames):
    both = pd.concat([f[["show", "channel"]] for f in frames], ignore_index=True)
    counts = both.groupby(["show", "channel"]).size().reset_index(name="n")
    counts = counts.sort_values(["show", "n", "channel"],
                                ascending=[True, False, True])
    return counts.drop_duplicates("show").set_index("show")["channel"].to_dict()


# --------------------------------------------------------------------------- #
# user side: hours and channels
# --------------------------------------------------------------------------- #

def user_hour_distribution(air, weeks, users):
    """{user: 24 shares of the airings that start in that hour}."""
    user_set = set(users)
    sub = air[(air["week"] >= weeks[0]) & (air["week"] <= weeks[1])
              & air["user_id"].isin(user_set)]
    frame = pd.DataFrame({"user_id": sub["user_id"].to_numpy(),
                          "hour": sub["airing_start"].dt.hour.to_numpy()})
    counts = frame.groupby(["user_id", "hour"]).size()
    out = {}
    for user, grp in counts.groupby(level=0, sort=False):
        arr = np.zeros(24)
        arr[grp.index.get_level_values(1).to_numpy()] = grp.to_numpy()
        out[user] = arr / arr.sum()
    return out


def user_channel_time(air, show_channel, weeks, users):
    """{user: {channel: watch time}} and {user: total watch time}.

    Watch time is the sum of the screen time of the user's airings.
    """
    user_set = set(users)
    sub = air[(air["week"] >= weeks[0]) & (air["week"] <= weeks[1])
              & air["user_id"].isin(user_set)]
    frame = pd.DataFrame({"user_id": sub["user_id"].to_numpy(),
                          "show": sub["tv_show_id"].to_numpy(),
                          "watch": sub["screen_time"].to_numpy()})
    frame["channel"] = frame["show"].map(show_channel)
    frame = frame[frame["channel"].notna()]
    frame["channel"] = frame["channel"].astype("int32")
    by_channel, totals = {}, {}
    for (user, channel), watch in frame.groupby(
            ["user_id", "channel"])["watch"].sum().items():
        by_channel.setdefault(user, {})[int(channel)] = float(watch)
        totals[user] = totals.get(user, 0.0) + float(watch)
    return by_channel, totals


def top_channels(by_channel, totals, user, k=N_CHANNELS):
    channels = by_channel.get(user)
    if not channels or totals.get(user, 0.0) <= 0:
        return []
    ranked = sorted(channels.items(), key=lambda item: (-item[1], item[0]))
    return [channel for channel, _ in ranked[:k]]


# --------------------------------------------------------------------------- #
# ALS and content candidates
# --------------------------------------------------------------------------- #

def topk_scores(scorers, users, pool, k, kind):
    """Top k shows of the pool per user, deterministic (score desc, id asc)."""
    pool_ids = np.array(sorted(s for s in pool if s in scorers.sidx),
                        dtype=np.int64)
    if len(pool_ids) == 0:
        return {}
    item_idx = np.array([scorers.sidx[s] for s in pool_ids])
    if kind == "als":
        user_factors = scorers.model.user_factors
        user_index = scorers.uidx
        item_factors = scorers.model.item_factors[item_idx]
    else:
        user_factors = scorers.profiles
        user_index = scorers.pidx
        item_factors = scorers.unit[item_idx]

    known = [(u, user_index[u]) for u in users if u in user_index]
    out = {}
    chunk = 512
    for start in range(0, len(known), chunk):
        batch = known[start:start + chunk]
        scores = user_factors[[i for _, i in batch]] @ item_factors.T
        for row, (user, _) in enumerate(batch):
            order = np.lexsort((pool_ids, -scores[row]))[:k]
            out[user] = [int(pool_ids[i]) for i in order]
    return out


# --------------------------------------------------------------------------- #
# window specification
# --------------------------------------------------------------------------- #

class Window:
    """One feature/label window and the schedule data that belongs to it."""

    def __init__(self, feat_weeks, label_weeks, feat_frame, target_frame, pool):
        self.feat_weeks = feat_weeks
        self.label_weeks = label_weeks
        self.history_end = feat_weeks[1]
        self.target_counts = sched_counts(target_frame, label_weeks)
        self.target_shows = set(self.target_counts)
        self.pool = pool if pool is not None else self.target_shows
        self.feat_counts = sched_counts(feat_frame, feat_weeks)
        self.hour_fractions = sched_hour_fractions(target_frame, label_weeks,
                                                   self.target_shows)
        self.channel_lists = sched_channel_lists(target_frame, label_weeks)


# --------------------------------------------------------------------------- #
# dataset
# --------------------------------------------------------------------------- #

def build_dataset(air, meta, win, cfg, users, scorers, labels, categories,
                  user_hours, user_channels, user_watch, show_channel):
    """One row per (user, show) candidate, with the features and the label."""
    w0, w1 = win.feat_weeks
    q_hist = qualifying(air, (HISTORY_START_WEEK, w1))
    q_feat = qualifying(air, win.feat_weeks)
    all_feat = air[(air["week"] >= w0) & (air["week"] <= w1)]

    agg = (q_hist.groupby(["user_id", "tv_show_id"])
                  .agg(count_since11=("screen_time", "size"),
                       last_week=("week", "max"))
                  .reset_index())
    win_counts = (q_feat.groupby(["user_id", "tv_show_id"])
                        .agg(count_window=("screen_time", "size"),
                             distinct_weeks=("week", "nunique"))
                        .reset_index())
    last2 = (q_feat[q_feat["week"] >= w1 - 1]
             .groupby(["user_id", "tv_show_id"]).size()
             .rename("count_last2").reset_index())
    all_stats = (all_feat.groupby(["user_id", "tv_show_id"])["screen_time"]
                 .agg(mean_screen_time="mean").reset_index())
    partial = (all_feat[(all_feat["screen_time"] >= 0.3)
                        & (all_feat["screen_time"] < 0.8)]
               .groupby(["user_id", "tv_show_id"]).size()
               .rename("partial_views").reset_index())

    # candidates: own history + popular + the extra sources
    hist_pop = show_popularity(q_hist)
    popular = [s for s in rank_shows(hist_pop, k=cfg["popular"])
               if s in win.pool]
    frames = [agg[["user_id", "tv_show_id"]]]
    left = pd.DataFrame({"user_id": users})
    left["_k"] = 1
    if popular:
        right = pd.DataFrame({"tv_show_id": popular})
        right["_k"] = 1
        frames.append(left.merge(right, on="_k")[["user_id", "tv_show_id"]])
    if cfg["extra_candidates"]:
        extras = {
            "als": topk_scores(scorers, users, win.pool, TOP_ALS, "als"),
            "content": topk_scores(scorers, users, win.pool, TOP_CONTENT,
                                   "content"),
        }
        channel_candidates = {}
        for user in users:
            counts = {}
            for channel in top_channels(user_channels, user_watch, user):
                for show, n in win.channel_lists.get(channel, []):
                    if show in win.pool:
                        counts[show] = max(counts.get(show, 0), n)
            ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
            channel_candidates[user] = [show for show, _ in ranked[:TOP_CHANNEL]]
        extras["channel"] = channel_candidates
        for source in extras.values():
            rows = [(user, show) for user, shows in source.items() for show in shows]
            if rows:
                frames.append(pd.DataFrame(rows, columns=["user_id", "tv_show_id"]))

    pairs = pd.concat(frames, ignore_index=True)
    pairs = pairs[pairs["user_id"].isin(set(users))]
    pairs = pairs[pairs["tv_show_id"].isin(win.pool)]
    pairs = pairs.drop_duplicates(subset=["user_id", "tv_show_id"]).reset_index(drop=True)

    out = pairs.merge(agg, on=["user_id", "tv_show_id"], how="left")
    out = out.merge(win_counts, on=["user_id", "tv_show_id"], how="left")
    out = out.merge(last2, on=["user_id", "tv_show_id"], how="left")
    out = out.merge(all_stats, on=["user_id", "tv_show_id"], how="left")
    out = out.merge(partial, on=["user_id", "tv_show_id"], how="left")
    for col in ("count_since11", "last_week", "count_window", "count_last2",
                "mean_screen_time", "distinct_weeks", "partial_views"):
        out[col] = out[col].fillna(0)

    out["weeks_since_last"] = np.where(out["last_week"] > 0,
                                       w1 - out["last_week"], 99)
    out = out.drop(columns=["last_week"])

    total = q_hist.groupby("user_id").size().rename("user_qual_total")
    out = out.merge(total, left_on="user_id", right_index=True, how="left")
    out["user_qual_total"] = out["user_qual_total"].fillna(0)
    out["user_show_share"] = np.where(out["user_qual_total"] > 0,
                                      out["count_since11"] / out["user_qual_total"], 0.0)

    out["schedule_window"] = out["tv_show_id"].map(
        lambda s: win.feat_counts.get(int(s), 0))
    out["schedule_target"] = out["tv_show_id"].map(
        lambda s: win.target_counts.get(int(s), 0))
    out["target_ratio"] = out["schedule_target"] / out["schedule_window"].clip(lower=1)

    pop = (q_hist.groupby("tv_show_id")["user_id"].nunique()
           .rename("popularity").reset_index())
    out = out.merge(pop, on="tv_show_id", how="left")
    out["popularity"] = out["popularity"].fillna(0)

    out["als_score"] = scorers.als_score(out["user_id"].to_numpy(),
                                         out["tv_show_id"].to_numpy())
    out["content_score"] = scorers.content_score(out["user_id"].to_numpy(),
                                                 out["tv_show_id"].to_numpy())

    if cfg["extra_features"]:
        out["channel"] = out["tv_show_id"].map(show_channel)
        channel_share = {}
        for user, channels in user_channels.items():
            denom = user_watch.get(user, 0.0)
            if denom <= 0:
                continue
            for channel, watch in channels.items():
                channel_share[(user, channel)] = watch / denom
        out["channel_watch_share"] = [
            channel_share.get((user, channel), 0.0)
            for user, channel in zip(out["user_id"], out["channel"])]
        out = out.drop(columns=["channel"])

        hour_overlap = []
        for user, show in zip(out["user_id"], out["tv_show_id"]):
            user_row = user_hours.get(user)
            show_row = win.hour_fractions.get(int(show))
            if user_row is None or show_row is None:
                hour_overlap.append(0.0)
            else:
                hour_overlap.append(float(np.dot(user_row, show_row)))
        out["hour_overlap"] = hour_overlap
        out["trend"] = np.where(out["count_window"] > 0,
                                out["count_last2"] / out["count_window"], 0.0)

    out["show_category"] = out["tv_show_id"].map(
        lambda s: meta.at[int(s), "tv_show_category"]
        if int(s) in meta.index and isinstance(meta.at[int(s), "tv_show_category"], str)
        else "unknown")
    out["show_category"] = pd.Categorical(out["show_category"], categories=categories)

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

def fit_ranker(train_data, features, params):
    keep = train_data[train_data.groupby(["window", "user_id"])["label"]
                      .transform("sum") > 0]
    keep = keep.sort_values(["window", "user_id", "tv_show_id"]).reset_index(drop=True)
    groups = keep.groupby(["window", "user_id"], sort=False).size().to_numpy()
    model = lightgbm.LGBMRanker(**params)
    model.fit(keep[features], keep["label"].to_numpy(), group=groups,
              categorical_feature=[c for c in CATEGORICAL if c in features])
    return model


def predict_topk(model, data, features, users, tie, popularity_top5):
    data = data.copy()
    data["ranker_score"] = model.predict(
        data[features], categorical_feature=[c for c in CATEGORICAL if c in features])
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
    return preds


def score(preds, users, truths):
    recency = map_at_k(truths["recency"], preds, TOP_K)
    first = map_at_k(truths["first-occurrence"], preds, TOP_K)
    hits = float(np.mean([len(set(preds[u][:TOP_K]) & set(truths["recency"][u]))
                          for u in users]))
    ap_recency = np.array([ap_at_k(truths["recency"][u], preds[u], TOP_K)
                           for u in users])
    return {"recency": recency, "first": first, "hits": hits, "ap": ap_recency}


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main():
    started = time.perf_counter()

    section("ENVIRONMENT")
    print(f"python   : {platform.python_version()}  ({sys.executable})")
    print(f"lightgbm : {lightgbm.__version__}")
    print(f"fit      : weeks <= 20 only, windows {WINDOWS_STACK}")
    print(f"validate : features {VAL_FEATURE_WEEKS}, labels {VAL_LABEL_WEEKS}")
    print(f"baseline params: {LGB_PARAMS}")
    print(f"grid           : {GRID}")

    section("LOADING")
    t0 = time.perf_counter()
    air = load_airings()
    template = pd.read_csv(SUBMISSION, dtype=str)
    template["user_id"] = template["user_id"].astype(str)
    submission_users = template["user_id"].tolist()
    meta = load_schedule([SCHEDULE_TRAIN, SCHEDULE_VALIDATION])
    categories = sorted({v for v in meta["tv_show_category"].dropna().unique()
                         if isinstance(v, str)} | {"unknown"})
    train_frame = load_schedule_frame(SCHEDULE_TRAIN)
    val_frame = load_schedule_frame(SCHEDULE_VALIDATION)
    show_channel = show_channel_map(train_frame, val_frame)
    print(f"airing rows       : {len(air)}")
    print(f"submission users  : {len(submission_users)}")
    print(f"schedule rows     : train {len(train_frame)}, validation {len(val_frame)}")
    print(f"shows with a channel: {len(show_channel)} (of {len(meta)} with metadata)")
    print(f"load time         : {time.perf_counter() - t0:.2f}s")

    t0 = time.perf_counter()
    val_counts = user_show_counts_full(qualifying(air, VALID_WEEKS))
    val_labels, truth_recency = build_labels(air, VAL_LABEL_WEEKS)
    truth_first = {}
    for user, grp in val_counts.groupby("user_id", sort=False):
        if len(grp) < TOP_K:
            continue
        ordered = grp.sort_values(by=["count", "first_qual", "tv_show_id"],
                                  ascending=[False, False, True], kind="stable")
        truth_first[user] = [int(s) for s in ordered["tv_show_id"].head(TOP_K)]
    users = [u for u in submission_users
             if u in truth_recency and len(truth_recency[u]) >= TOP_K]
    truths = {"recency": {u: truth_recency[u] for u in users},
              "first-occurrence": {u: truth_first[u] for u in users}}
    val_pool = candidate_pool(VALID_WEEKS, SCHEDULE_VALIDATION)
    print(f"evaluated users   : {len(users)}")
    print(f"validation pool   : {len(val_pool)} shows")
    print(f"truth build time  : {time.perf_counter() - t0:.2f}s")

    section("WINDOWS")
    val_window = Window(VAL_FEATURE_WEEKS, VAL_LABEL_WEEKS, train_frame, val_frame,
                        val_pool)
    train_windows = [Window((f0, f1), (l0, l1), train_frame, train_frame, None)
                     for f0, f1, l0, l1 in WINDOWS_STACK]
    print(f"target shows (validation)      : {len(val_window.target_shows)}")
    for window in train_windows:
        print(f"fit window features {window.feat_weeks} labels {window.label_weeks}: "
              f"{len(window.target_shows)} target shows")

    section("SHARED INPUTS")
    t0 = time.perf_counter()
    q_train = qualifying(air, TRAIN_WEEKS)
    train_counts = user_show_counts_full(q_train)
    val_pop = show_popularity(q_train)
    popularity_top5 = rank_shows(val_pop[val_pop["tv_show_id"].isin(val_pool)])
    tie = TieBreak(q_train, val_pool)
    val_scorers = Scorers(train_counts, val_window.target_shows, meta)
    print(f"popularity top-5 (weeks 11-20): {popularity_top5}")
    print(f"validation ALS/content (weeks 11-20): {time.perf_counter() - t0:.2f}s")

    # ----------------------------------------------------------------------- #
    # cached dataset builders
    # ----------------------------------------------------------------------- #

    train_cache, val_cache = {}, {}

    def train_data(cfg):
        key = (cfg["candidates"], cfg["features"], cfg["windows"])
        if key not in train_cache:
            t = time.perf_counter()
            chosen = train_windows if cfg["windows"] else train_windows[:1]
            frames = []
            for i, window in enumerate(chosen):
                counts = user_show_counts_full(
                    qualifying(air, (HISTORY_START_WEEK, window.history_end)))
                scorers = Scorers(counts, window.target_shows, meta)
                labels, _ = build_labels(air, window.label_weeks)
                window_users = sorted(labels)
                hours = user_hour_distribution(
                    air, (HISTORY_START_WEEK, window.history_end), window_users)
                channels, watch = user_channel_time(
                    air, show_channel, (HISTORY_START_WEEK, window.history_end),
                    window_users)
                frame = build_dataset(
                    air, meta, window, cfg, window_users, scorers, labels,
                    categories, hours, channels, watch, show_channel)
                frame["window"] = i
                frames.append(frame)
                print(f"    window {i} features {window.feat_weeks} labels "
                      f"{window.label_weeks}: {len(frame)} rows "
                      f"({frame['user_id'].nunique()} users)")
                del scorers, hours, channels, watch
                gc.collect()
            data = pd.concat(frames, ignore_index=True)
            del frames
            gc.collect()
            train_cache[key] = data
            print(f"  train data {key}: {len(data)} rows in "
                  f"{time.perf_counter() - t:.1f}s")
        return train_cache[key]

    def validation_data(cfg):
        key = (cfg["candidates"], cfg["features"])
        if key not in val_cache:
            t = time.perf_counter()
            hours = user_hour_distribution(
                air, (HISTORY_START_WEEK, val_window.history_end), users)
            channels, watch = user_channel_time(
                air, show_channel, (HISTORY_START_WEEK, val_window.history_end),
                users)
            data = build_dataset(
                air, meta, val_window, cfg, users, val_scorers, val_labels,
                categories, hours, channels, watch, show_channel)
            val_cache[key] = data
            print(f"  validation data {key}: {len(data)} rows "
                  f"({data['user_id'].nunique()} users) in "
                  f"{time.perf_counter() - t:.1f}s")
        return val_cache[key]

    def recall(cfg):
        data = validation_data(cfg)
        by_user = {u: set(g["tv_show_id"]) for u, g in
                   data.groupby("user_id", sort=False)}
        return float(np.mean([len(set(truths["recency"][u]) & by_user.get(u, set()))
                              / TOP_K for u in users]))

    def run(cfg, params, features):
        t = time.perf_counter()
        model = fit_ranker(train_data(cfg), features, params)
        preds = predict_topk(model, validation_data(cfg), features, users, tie,
                             popularity_top5)
        result = score(preds, users, truths)
        result["recall"] = recall(cfg)
        result["params"] = params
        result["runtime"] = time.perf_counter() - t
        result["model"] = model
        result["preds"] = preds
        return result

    base_cfg = {"candidates": "base", "features": "base", "windows": False,
                "popular": POPULAR_BASE, "extra_candidates": False,
                "extra_features": False}
    wide_cfg = dict(base_cfg, candidates="wide", popular=POPULAR_WIDE,
                    extra_candidates=True)
    feat_cfg = dict(base_cfg, features="extra", extra_features=True)
    stack_cfg = dict(base_cfg, windows=True)
    base_features = list(FEATURES_BASE)
    extra_features = list(FEATURES_BASE) + FEATURES_EXTRA

    # ----------------------------------------------------------------------- #
    # experiments: baseline, A, B, C
    # ----------------------------------------------------------------------- #

    section("EXPERIMENT: baseline v5 ranker")
    baseline = run(base_cfg, dict(LGB_PARAMS), base_features)
    print(f"MAP@5 recency {baseline['recency']:.6f}, first-occ {baseline['first']:.6f}, "
          f"recall {baseline['recall']:.4f}, {baseline['runtime']:.1f}s")

    section("EXPERIMENT: A wider candidates")
    res_a = run(wide_cfg, dict(LGB_PARAMS), base_features)
    print(f"MAP@5 recency {res_a['recency']:.6f} "
          f"({res_a['recency'] - baseline['recency']:+.6f}), recall "
          f"{res_a['recall']:.4f} ({res_a['recall'] - baseline['recall']:+.4f}), "
          f"{res_a['runtime']:.1f}s")

    section("EXPERIMENT: B six new features")
    res_b = run(feat_cfg, dict(LGB_PARAMS), extra_features)
    print(f"MAP@5 recency {res_b['recency']:.6f} "
          f"({res_b['recency'] - baseline['recency']:+.6f}), recall "
          f"{res_b['recall']:.4f}, {res_b['runtime']:.1f}s")

    section("EXPERIMENT: C stacked training windows")
    res_c = run(stack_cfg, dict(LGB_PARAMS), base_features)
    print(f"MAP@5 recency {res_c['recency']:.6f} "
          f"({res_c['recency'] - baseline['recency']:+.6f}), recall "
          f"{res_c['recall']:.4f}, {res_c['runtime']:.1f}s")

    kept = []
    for name, res in (("A", res_a), ("B", res_b), ("C", res_c)):
        if res["recency"] > baseline["recency"]:
            kept.append(name)
            print(f"KEEP {name}: {res['recency'] - baseline['recency']:+.6f} MAP@5")
        else:
            print(f"drop {name}: {res['recency'] - baseline['recency']:+.6f} MAP@5")

    # ----------------------------------------------------------------------- #
    # the combination (caches are pruned to it before it is built)
    # ----------------------------------------------------------------------- #

    combo_cfg = dict(base_cfg)
    combo_features = base_features
    if "A" in kept:
        combo_cfg.update(candidates="wide", popular=POPULAR_WIDE,
                         extra_candidates=True)
    if "B" in kept:
        combo_cfg.update(features="extra", extra_features=True)
        combo_features = extra_features
    if "C" in kept:
        combo_cfg.update(windows=True)

    combo_key = (combo_cfg["candidates"], combo_cfg["features"], combo_cfg["windows"])
    train_cache = {k: v for k, v in train_cache.items() if k == combo_key}
    val_cache = {k: v for k, v in val_cache.items()
                 if k == combo_key[:2]}
    gc.collect()

    section("EXPERIMENT: all kept changes together")
    print(f"recipe: candidates={combo_cfg['candidates']}, "
          f"features={combo_cfg['features']}, windows={combo_cfg['windows']}")
    res_combo = run(combo_cfg, dict(LGB_PARAMS), combo_features)
    print(f"MAP@5 recency {res_combo['recency']:.6f} "
          f"({res_combo['recency'] - baseline['recency']:+.6f}), first-occ "
          f"{res_combo['first']:.6f}, recall {res_combo['recall']:.4f}, "
          f"{res_combo['runtime']:.1f}s")

    # ----------------------------------------------------------------------- #
    # D: small grid on the kept recipe
    # ----------------------------------------------------------------------- #

    section("EXPERIMENT: D small LightGBM grid (on the kept recipe)")
    grid_rows = []
    grid_total = 0.0
    grid_data = train_data(combo_cfg)
    val_data = validation_data(combo_cfg)
    for leaves, n_estimators in GRID:
        params = dict(LGB_PARAMS, num_leaves=leaves, n_estimators=n_estimators)
        t = time.perf_counter()
        model = fit_ranker(grid_data, combo_features, params)
        preds = predict_topk(model, val_data, combo_features, users, tie,
                             popularity_top5)
        result = score(preds, users, truths)
        result["recall"] = res_combo["recall"]
        result["params"] = params
        result["runtime"] = time.perf_counter() - t
        result["model"] = model
        result["preds"] = preds
        grid_total += result["runtime"]
        grid_rows.append(((leaves, n_estimators), result))
        print(f"  num_leaves {leaves:>2}, n_estimators {n_estimators:>3}: "
              f"MAP@5 recency {result['recency']:.6f}, first-occ "
              f"{result['first']:.6f}  ({result['runtime']:.1f}s)")
    best_key, best_grid = max(grid_rows, key=lambda item: item[1]["recency"])
    best_grid["runtime"] = grid_total
    print(f"best grid setting: num_leaves {best_key[0]}, n_estimators {best_key[1]} "
          f"({best_grid['recency']:.6f}), grid total {grid_total:.1f}s")

    tuning_helps = best_grid["recency"] > res_combo["recency"]
    print(f"D {'kept' if tuning_helps else 'dropped'}: "
          f"{best_grid['recency'] - res_combo['recency']:+.6f} vs the kept recipe")
    final = best_grid if tuning_helps else res_combo

    # ----------------------------------------------------------------------- #
    # one table
    # ----------------------------------------------------------------------- #

    section("RESULTS (validation weeks 21-30, %d users)" % len(users))
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n_users = len(users)
    resamples = rng.integers(0, n_users, size=(BOOTSTRAP_RESAMPLES, n_users))
    ap_base = baseline["ap"]

    rows = [("v5 baseline ranker", baseline),
            ("+A wider candidates", res_a),
            ("+B six new features", res_b),
            ("+C stacked windows", res_c),
            ("+D grid best", best_grid),
            ("all kept together", final)]

    header = (f"{'variant':<24s} {'MAP@5 recency':>13s} {'MAP@5 first':>12s} "
              f"{'recall':>7s} {'vs v5':>10s} {'95% CI vs v5':>24s} {'runtime':>8s}")
    print(header)
    print("-" * len(header))
    for name, res in rows:
        diff = res["recency"] - baseline["recency"]
        if res is baseline:
            ci = ""
        else:
            draws = res["ap"][resamples].mean(axis=1) - ap_base[resamples].mean(axis=1)
            lo, hi = np.percentile(draws, [2.5, 97.5])
            ci = f"[{lo:+.6f}, {hi:+.6f}]"
        print(f"{name:<24s} {res['recency']:>13.6f} {res['first']:>12.6f} "
              f"{res['recall']:>7.4f} {diff:>+10.6f} {ci:>24s} "
              f"{res['runtime']:>7.1f}s")
    print(f"\npaired bootstrap: {BOOTSTRAP_RESAMPLES} resamples of {n_users} users, "
          f"seed {BOOTSTRAP_SEED}")
    print(f"kept: {', '.join(kept) if kept else 'none'}"
          f"{', D' if tuning_helps else ''}")
    print(f"final: candidates={combo_cfg['candidates']}, "
          f"features={combo_cfg['features']}, windows={combo_cfg['windows']}, "
          f"num_leaves={final['params']['num_leaves']}, "
          f"n_estimators={final['params']['n_estimators']}")
    print(f"candidate recall of the truth: baseline {baseline['recall']:.4f}, "
          f"final {final['recall']:.4f}")

    section("VALIDATION PREDICTIONS OF THE FINAL MODEL")
    write_rankings(VAL_V6_CSV, users, final["preds"])
    back = read_rankings(VAL_V6_CSV)
    assert list(back) == list(users), "user order mismatch"
    assert all(len(v) == TOP_K for v in back.values()), "row length"
    recomputed = map_at_k({u: set(truths["recency"][u]) for u in users}, back, TOP_K)
    print(f"wrote {VAL_V6_CSV} ({len(back)} users)")
    print(f"MAP@5 recomputed from the file: {recomputed:.6f} "
          f"(in memory {final['recency']:.6f})")

    v5 = read_rankings(VAL_V5_CSV)
    ap_v6 = np.array([ap_at_k(truths["recency"][u], back[u], TOP_K) for u in users])
    ap_v5 = np.array([ap_at_k(truths["recency"][u], v5[u], TOP_K) for u in users])
    observed = ap_v6.mean() - ap_v5.mean()
    draws = ap_v6[resamples].mean(axis=1) - ap_v5[resamples].mean(axis=1)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    print(f"B+D vs v5 (from the saved files): {observed:+.6f}, "
          f"95% CI [{lo:+.6f}, {hi:+.6f}]")
    print(f"users better/worse: {int((ap_v6 > ap_v5).sum())} / "
          f"{int((ap_v6 < ap_v5).sum())}")

    section("FINAL MODEL FEATURE IMPORTANCES (gain)")
    gains = final["model"].booster_.feature_importance(importance_type="gain")
    total = gains.sum()
    for i in np.argsort(-gains):
        print(f"  {combo_features[i]:<20s} {gains[i]:>12.1f}  {gains[i] / total:>7.2%}")
    print(f"total runtime: {time.perf_counter() - started:.2f}s")


if __name__ == "__main__":
    main()
