import gc
import os
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd

from evaluate import ap_at_k, map_at_k
from evaluate_v2 import (SUBMISSION, TOP_K, TRAIN_WEEKS, VALID_WEEKS,
                         candidate_pool, load_airings, pad_with_popularity,
                         qualifying, rank_shows, show_popularity)
from ltr import (FEATURES as FEATURES_BASE, LGB_PARAMS, Scorers, build_labels,
                 read_rankings)
from ltr_v2 import (FEATURES_EXTRA, HISTORY_START_WEEK, POPULAR_BASE, Window,
                    build_dataset, fit_ranker, load_schedule_frame,
                    predict_topk, show_channel_map, user_channel_time,
                    user_hour_distribution)
from models import SCHEDULE_TRAIN, SCHEDULE_VALIDATION, TieBreak, load_schedule, user_show_counts_full, rank_by_score

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
VAL_V5_CSV = OUTPUTS / "val_ltr_predictions.csv"
VAL_V6_CSV = OUTPUTS / "val_ltr_v6_predictions.csv"
VAL_V7_CSV = OUTPUTS / "val_v7_predictions.csv"

def section(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")

def load_rankings_dict(path):
    frame = pd.read_csv(path, dtype=str)
    return {u: [int(s) for s in r.split()] for u, r in
            zip(frame["user_id"], frame["tv_show_id"])}

def bootstrap_ci_vs_base(ap_base, ap_other, users, n=1000, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(users), size=(n, len(users)))
    draws = ap_other[idx].mean(1) - ap_base[idx].mean(1)
    return float(draws.mean()), float(np.percentile(draws, [2.5, 97.5])[0]), float(np.percentile(draws, [2.5, 97.5])[1]), \
           int((ap_other > ap_base).sum()), int((ap_other < ap_base).sum())

def main():
    started = time.perf_counter()
    section("(1) CORRECTED per-week active users")
    air = load_airings()
    # correct: nunique(user_id) of users with >=1 screen_time >= 0.8 in that week
    q = qualifying(air, (11, 30))
    active = q.groupby("week")["user_id"].nunique().sort_index()
    total_distinct = air["user_id"].nunique()
    print(f"Total distinct users in airings_11_30: {total_distinct}")
    print("Active users per week (>=1 qualifying airing, screen_time>=0.8):")
    for w in range(11, 31):
        v = int(active.get(w, 0))
        print(f"  week {w:2d}: {v:5d}")
        assert v <= total_distinct, f"week {w} has {v} > {total_distinct}"
    print("All weeks <= total distinct users: OK")

    print("\nComparing label windows:")
    for name, weeks in (("16-20", (16, 20)), ("26-30", (26, 30))):
        qq = qualifying(air, weeks)
        l, t = build_labels(air, weeks)
        print(f"  {name}: qual events {len(qq):6d}, active users {qq['user_id'].nunique():5d}, "
              f"label users {len(t):5d}, pos slots {sum(len(x) for x in t.values()):6d} (avg {sum(len(x) for x in t.values())/max(1,len(t)):.2f})")

    section("RELOAD VALIDATION SETUP")
    template = pd.read_csv(SUBMISSION, dtype=str)
    subscription_users = template["user_id"].astype(str).tolist()
    meta = load_schedule([SCHEDULE_TRAIN, SCHEDULE_VALIDATION])
    categories = sorted({v for v in meta["tv_show_category"].dropna().unique()
                         if isinstance(v, str)} | {"unknown"})

    val_counts_full = user_show_counts_full(qualifying(air, VALID_WEEKS))
    val_labels, truth_recency = build_labels(air, VALID_WEEKS)
    truth_first = {}
    for user, grp in val_counts_full.groupby("user_id", sort=False):
        if len(grp) < TOP_K:
            continue
        o = grp.sort_values(["count", "first_qual", "tv_show_id"],
                            ascending=[False, False, True], kind="stable")
        truth_first[user] = [int(s) for s in o["tv_show_id"].head(TOP_K)]
    users = [u for u in subscription_users
             if u in truth_recency and len(truth_recency[u]) >= TOP_K]
    truths = {"recency": {u: truth_recency[u] for u in users},
              "first-occurrence": {u: truth_first[u] for u in users}}

    val_pool = candidate_pool(VALID_WEEKS, SCHEDULE_VALIDATION)
    q_train = qualifying(air, TRAIN_WEEKS)
    val_pop = show_popularity(q_train)
    popularity_top5 = rank_shows(val_pop[val_pop["tv_show_id"].isin(val_pool)])
    tie = TieBreak(q_train, val_pool)

    train_frame = load_schedule_frame(SCHEDULE_TRAIN)
    val_frame = load_schedule_frame(SCHEDULE_VALIDATION)
    show_channel = show_channel_map(train_frame, val_frame)
    val_window = Window((16, 20), (21, 30), train_frame, val_frame, val_pool)
    val_scorers = Scorers(user_show_counts_full(qualifying(air, TRAIN_WEEKS)),
                          val_window.target_shows, meta)
    val_labels_map, _ = build_labels(air, (21, 30))
    val_hours = user_hour_distribution(air, (TRAIN_WEEKS[0], val_window.history_end), users)
    val_channels, val_watch = user_channel_time(air, show_channel,
                                                (TRAIN_WEEKS[0], val_window.history_end),
                                                users)

    preds_v5 = load_rankings_dict(VAL_V5_CSV)
    preds_v6 = load_rankings_dict(VAL_V6_CSV)
    preds_v7 = load_rankings_dict(VAL_V7_CSV)
    ap_v5 = np.array([ap_at_k(truths["recency"][u], preds_v5[u], TOP_K) for u in users])
    ap_v6_arr = np.array([ap_at_k(truths["recency"][u], preds_v6[u], TOP_K) for u in users])
    ap_v7_arr = np.array([ap_at_k(truths["recency"][u], preds_v7[u], TOP_K) for u in users])

    section("(2) CLEAN comparisons vs v6")
    base_cfg_bag = {"candidates": "base", "features": "extra", "windows": False,
                    "popular": POPULAR_BASE, "extra_candidates": False,
                    "extra_features": True}
    features_v6 = list(FEATURES_BASE) + FEATURES_EXTRA
    params_v6 = dict(LGB_PARAMS, num_leaves=31, n_estimators=300,
                     learning_rate=0.05, random_state=42)

    # need scores for ensembles - rebuild quickly
    fit_window_bag = Window((11, 15), (16, 20), train_frame, train_frame, None)
    fit_users_bag = sorted(build_labels(air, (16, 20))[0])
    fit_hist_bag = qualifying(air, (HISTORY_START_WEEK, fit_window_bag.history_end))
    fit_counts_bag = user_show_counts_full(fit_hist_bag)
    fit_scorers_bag = Scorers(fit_counts_bag, fit_window_bag.target_shows, meta)
    fit_labels_bag, _ = build_labels(air, (16, 20))
    fit_hours_bag = user_hour_distribution(air, (HISTORY_START_WEEK, fit_window_bag.history_end), fit_users_bag)
    fit_channels_bag, fit_watch_bag = user_channel_time(air, show_channel, (HISTORY_START_WEEK, fit_window_bag.history_end), fit_users_bag)
    fit_data_bag = build_dataset(air, meta, fit_window_bag, base_cfg_bag, fit_users_bag, fit_scorers_bag, fit_labels_bag,
                                 categories, fit_hours_bag, fit_channels_bag, fit_watch_bag, show_channel)
    fit_data_bag["window"] = 0
    val_data_bag = build_dataset(air, meta, val_window, base_cfg_bag, users, val_scorers, val_labels_map,
                                 categories, val_hours, val_channels, val_watch, show_channel)

    # v5
    params_v5 = dict(LGB_PARAMS)
    fit_window_v5 = Window((11, 15), (16, 20), train_frame, train_frame, None)
    fit_users_v5 = sorted(build_labels(air, (16, 20))[0])
    fit_hist_v5 = qualifying(air, (HISTORY_START_WEEK, fit_window_v5.history_end))
    fit_counts_v5 = user_show_counts_full(fit_hist_v5)
    fit_scorers_v5 = Scorers(fit_counts_v5, fit_window_v5.target_shows, meta)
    fit_labels_v5, _ = build_labels(air, (16, 20))
    fit_hours_v5 = user_hour_distribution(air, (HISTORY_START_WEEK, fit_window_v5.history_end), fit_users_v5)
    fit_channels_v5, fit_watch_v5 = user_channel_time(air, show_channel, (HISTORY_START_WEEK, fit_window_v5.history_end), fit_users_v5)
    base_cfg_v5 = {"candidates": "base", "features": "base", "windows": False, "popular": POPULAR_BASE, "extra_candidates": False, "extra_features": False}
    fit_data_v5 = build_dataset(air, meta, fit_window_v5, base_cfg_v5, fit_users_v5, fit_scorers_v5, fit_labels_v5,
                                categories, fit_hours_v5, fit_channels_v5, fit_watch_v5, show_channel)
    fit_data_v5["window"] = 0
    model_v5 = fit_ranker(fit_data_v5, list(FEATURES_BASE), params_v5)
    val_data_v5s = build_dataset(air, meta, val_window, base_cfg_v5, users, val_scorers, val_labels_map,
                                 categories, val_hours, val_channels, val_watch, show_channel)
    s5 = val_data_v5s.copy()
    s5["ranker_score"] = model_v5.predict(s5[list(FEATURES_BASE)], categorical_feature=[c for c in ["show_category"] if c in FEATURES_BASE])

    model_v6 = fit_ranker(fit_data_bag, features_v6, params_v6)
    s6 = val_data_bag.copy()
    s6["ranker_score"] = model_v6.predict(s6[features_v6], categorical_feature=[c for c in ["show_category"] if c in features_v6])

    # bagged preds
    scores_bag = []
    for sseed in range(42, 47):
        params = dict(LGB_PARAMS, num_leaves=31, n_estimators=300, learning_rate=0.05, random_state=sseed)
        model = fit_ranker(fit_data_bag, features_v6, params)
        data = val_data_bag.copy()
        data["ranker_score"] = model.predict(data[features_v6], categorical_feature=[c for c in ["show_category"] if c in features_v6])
        scores_bag.append(data[["user_id", "tv_show_id", "ranker_score"]].copy())
        del model
        gc.collect()
    all_scores = pd.concat(scores_bag, ignore_index=True)
    avg = all_scores.groupby(["user_id", "tv_show_id"])["ranker_score"].mean().reset_index()
    by_user_bag = {u: grp for u, grp in avg.groupby("user_id", sort=False)}
    preds_bag = {}
    for user in users:
        grp = by_user_bag.get(user)
        ranked = []
        if grp is not None and len(grp):
            show_ids = grp["tv_show_id"].to_numpy()
            ranked = rank_by_score(grp["ranker_score"].to_numpy(), show_ids, tie.for_ids(user, show_ids), k=TOP_K)
        preds_bag[user] = pad_with_popularity(ranked, popularity_top5)
    ap_bag = np.array([ap_at_k(truths["recency"][u], preds_bag[u], TOP_K) for u in users])
    obs_bag, lo_bag, hi_bag, b_b, w_b = bootstrap_ci_vs_base(ap_v6_arr, ap_bag, users)
    print(f"v6 bagged (42-46): MAP {ap_bag.mean():.6f} vs v6 {obs_bag:+.6f}, 95% CI [{lo_bag:+.6f}, {hi_bag:+.6f}], better/worse {b_b}/{w_b}")

    # v5+v6
    s5m = s5[["user_id", "tv_show_id", "ranker_score"]].rename(columns={"ranker_score": "s5"})
    s6m = s6[["user_id", "tv_show_id", "ranker_score"]].rename(columns={"ranker_score": "s6"})
    merged = s5m.merge(s6m, on=["user_id", "tv_show_id"])
    merged["ravg"] = 0.5 * merged["s5"] + 0.5 * merged["s6"]
    by_user_ra = {u: grp for u, grp in merged.groupby("user_id", sort=False)}
    preds_ra = {}
    for user in users:
        grp = by_user_ra.get(user)
        ranked = []
        if grp is not None and len(grp):
            show_ids = grp["tv_show_id"].to_numpy()
            ranked = rank_by_score(grp["ravg"].to_numpy(), show_ids, tie.for_ids(user, show_ids), k=TOP_K)
        preds_ra[user] = pad_with_popularity(ranked, popularity_top5)
    ap_ra = np.array([ap_at_k(truths["recency"][u], preds_ra[u], TOP_K) for u in users])
    obs_ra, lo_ra, hi_ra, b_ra, w_ra = bootstrap_ci_vs_base(ap_v6_arr, ap_ra, users)
    print(f"rank-avg v5+v6:   MAP {ap_ra.mean():.6f} vs v6 {obs_ra:+.6f}, 95% CI [{lo_ra:+.6f}, {hi_ra:+.6f}], better/worse {b_ra}/{w_ra}")

    # v5+v6+own
    qh = qualifying(air, TRAIN_WEEKS)
    own_counts = user_show_counts_full(qh)
    own_by = {u: {int(s): float(c) for s, c in zip(g["tv_show_id"], g["count"])} for u, g in own_counts.groupby("user_id", sort=False)}
    rows = []
    for _, r in merged.iterrows():
        u = r["user_id"]
        c = int(r["tv_show_id"])
        sown = own_by.get(u, {}).get(c, 0.0)
        rows.append((u, c, r["s5"], r["s6"], sown))
    mm = pd.DataFrame(rows, columns=["user_id", "tv_show_id", "s5", "s6", "sown"])
    mm["ravg3"] = (mm["s5"] + mm["s6"] + mm["sown"]) / 3.0
    by_user_ra3 = {u: grp for u, grp in mm.groupby("user_id", sort=False)}
    preds_ra3 = {}
    for user in users:
        grp = by_user_ra3.get(user)
        ranked = []
        if grp is not None and len(grp):
            show_ids = grp["tv_show_id"].to_numpy()
            ranked = rank_by_score(grp["ravg3"].to_numpy(), show_ids, tie.for_ids(user, show_ids), k=TOP_K)
        preds_ra3[user] = pad_with_popularity(ranked, popularity_top5)
    ap_ra3 = np.array([ap_at_k(truths["recency"][u], preds_ra3[u], TOP_K) for u in users])
    obs_ra3, lo_ra3, hi_ra3, b_ra3, w_ra3 = bootstrap_ci_vs_base(ap_v6_arr, ap_ra3, users)
    print(f"rank-avg v5+v6+own: MAP {ap_ra3.mean():.6f} vs v6 {obs_ra3:+.6f}, 95% CI [{lo_ra3:+.6f}, {hi_ra3:+.6f}], better/worse {b_ra3}/{w_ra3}")

    print(f"\nRuntime: {time.perf_counter()-started:.1f}s")

if __name__ == "__main__":
    main()
