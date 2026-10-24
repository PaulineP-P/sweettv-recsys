import gc
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import time
from pathlib import Path

import lightgbm
import numpy as np
import pandas as pd

from evaluate import ap_at_k, map_at_k
from evaluate_v2 import (SCHEDULE_TEST, SCHEDULE_VALIDATION, SUBMISSION, TOP_K,
                         TRAIN_WEEKS, VALID_WEEKS, TEST_WEEKS, candidate_pool, load_airings,
                         pad_with_popularity, qualifying, rank_shows,
                         show_popularity)
from ltr import (FEATURES as FEATURES_BASE, LGB_PARAMS, Scorers, build_labels,
                 read_rankings, write_rankings)
from ltr_v2 import (FEATURES_EXTRA, HISTORY_START_WEEK, POPULAR_BASE, Window,
                    build_dataset, fit_ranker, load_schedule_frame,
                    predict_topk, show_channel_map, user_channel_time,
                    user_hour_distribution)
from models import (SCHEDULE_TRAIN, TieBreak, EPOCH, load_schedule,
                    method_own_history, rank_by_score, user_show_counts_full,
                    rank_shows as rank_shows_models)

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
VAL_V5_CSV = OUTPUTS / "val_ltr_predictions.csv"
VAL_V6_CSV = OUTPUTS / "val_ltr_v6_predictions.csv"
VAL_V7_CSV = OUTPUTS / "val_v7_predictions.csv"
SUB_V5 = OUTPUTS / "submission_v5_ltr.csv"
SUB_V6 = OUTPUTS / "submission_v6_ltr_features.csv"
SUB_V7 = OUTPUTS / "submission_v7_robust.csv"

started_total = time.perf_counter()

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
    section("DIAGNOSTICS (1) - weeks 11-30")
    air = load_airings()
    train_frame = load_schedule_frame(SCHEDULE_TRAIN)
    val_frame = load_schedule_frame(SCHEDULE_VALIDATION)
    all_sched = pd.concat([train_frame, val_frame], ignore_index=True)

    # active users per week: users with >=1 qualifying (screen_time>=0.8) airing that week
    q_all = qualifying(air, (11, 30))
    active = q_all.groupby("week")["user_id"].nunique().sort_index()
    print("active users (>=1 qualifying airing) per week:")
    for w in range(11, 31):
        print(f"  week {w:2d}: {active.get(w, 0):5d}")

    # >=0.8 airings per week (count of qualifying airing events)
    qcount = q_all.groupby("week").size().sort_index()
    print("\n>=0.8 airings (events) per week:")
    for w in range(11, 31):
        print(f"  week {w:2d}: {qcount.get(w, 0):7d}")

    # labelled airings in the schedule (labelled shows = tv_show_id != 0 airing in schedule) per week
    sched_labelled = all_sched[all_sched["show"] != 0].groupby("week").size().sort_index()
    print("\nlabelled airings in schedule (tv_show_id!=0) per week:")
    for w in range(11, 31):
        print(f"  week {w:2d}: {sched_labelled.get(w, 0):7d}")

    section("LABEL WINDOWS USED FOR FITTING")
    for name, weeks in (("label window A (v5/v6)", (16, 20)),
                        ("label window B (v6 fit)", (26, 30))):
        l, t = build_labels(air, weeks)
        n_users = len(t)
        total_pos = sum(len(v) for v in t.values())
        print(f"{name}: weeks {weeks[0]}-{weeks[1]} -> users with truth {n_users}, "
              f"total positive slots {total_pos} (avg {total_pos/max(1,n_users):.2f})")
    # Also see qualifying volume
    for name, weeks in (("qualifying A", (16, 20)), ("qualifying B", (26, 30))):
        qq = qualifying(air, weeks)
        print(f"{name}: weeks {weeks[0]}-{weeks[1]} -> qualifying events {len(qq)}, "
              f"active users {qq['user_id'].nunique()}")

    # Load v6 validation setup
    section("VALIDATION SETUP (for comparisons)")
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
    print(f"validation users (>=5 qual shows): {len(users)}")

    val_pool = candidate_pool(VALID_WEEKS, SCHEDULE_VALIDATION)
    q_train = qualifying(air, TRAIN_WEEKS)
    val_pop = show_popularity(q_train)
    popularity_top5 = rank_shows(val_pop[val_pop["tv_show_id"].isin(val_pool)])
    tie = TieBreak(q_train, val_pool)

    val_window = Window((16, 20), (21, 30), train_frame, val_frame, val_pool)
    val_scorers = Scorers(user_show_counts_full(qualifying(air, TRAIN_WEEKS)),
                          val_window.target_shows, meta)
    val_labels_map, _ = build_labels(air, (21, 30))

    show_channel = show_channel_map(train_frame, val_frame)
    val_hours = user_hour_distribution(air, (TRAIN_WEEKS[0], val_window.history_end),
                                       users)
    val_channels, val_watch = user_channel_time(air, show_channel,
                                                (TRAIN_WEEKS[0],
                                                 val_window.history_end),
                                                users)

    base_cfg = {"candidates": "base", "features": "extra", "windows": False,
                "popular": POPULAR_BASE, "extra_candidates": False,
                "extra_features": True}
    features_v6 = list(FEATURES_BASE) + FEATURES_EXTRA
    params_v6 = dict(LGB_PARAMS, num_leaves=31, n_estimators=300,
                     learning_rate=0.05, random_state=42)

    val_data = build_dataset(air, meta, val_window, base_cfg, users, val_scorers,
                             val_labels_map, categories, val_hours, val_channels,
                             val_watch, show_channel)

    # load existing preds
    preds_v5 = load_rankings_dict(VAL_V5_CSV)
    preds_v6 = load_rankings_dict(VAL_V6_CSV)
    # ensure order
    assert list(preds_v5.keys()) == users and list(preds_v6.keys()) == users

    ap_v5 = np.array([ap_at_k(truths["recency"][u], preds_v5[u], TOP_K) for u in users])
    ap_v6_arr = np.array([ap_at_k(truths["recency"][u], preds_v6[u], TOP_K)
                          for u in users])
    print(f"v5 MAP@5 recency: {ap_v5.mean():.6f}")
    print(f"v6 MAP@5 recency: {ap_v6_arr.mean():.6f}")

    section("(2a) v6 seed-bagged: seeds 42..46, average scores")
    fit_window_bag = Window((11, 15), (16, 20), train_frame, train_frame, None)
    fit_users_bag = sorted(build_labels(air, (16, 20))[0])
    fit_hist_bag = qualifying(air, (HISTORY_START_WEEK, fit_window_bag.history_end))
    fit_counts_bag = user_show_counts_full(fit_hist_bag)
    fit_scorers_bag = Scorers(fit_counts_bag, fit_window_bag.target_shows, meta)
    fit_labels_bag, _ = build_labels(air, (16, 20))
    fit_hours_bag = user_hour_distribution(air, (HISTORY_START_WEEK,
                                                 fit_window_bag.history_end),
                                           fit_users_bag)
    fit_channels_bag, fit_watch_bag = user_channel_time(air, show_channel,
                                                        (HISTORY_START_WEEK,
                                                         fit_window_bag.history_end),
                                                        fit_users_bag)
    base_cfg_bag = {"candidates": "base", "features": "extra", "windows": False,
                    "popular": POPULAR_BASE, "extra_candidates": False,
                    "extra_features": True}
    fit_data_bag = build_dataset(air, meta, fit_window_bag, base_cfg_bag,
                                 fit_users_bag, fit_scorers_bag, fit_labels_bag,
                                 categories, fit_hours_bag, fit_channels_bag,
                                 fit_watch_bag, show_channel)
    fit_data_bag["window"] = 0

    val_data_bag = build_dataset(air, meta, val_window, base_cfg_bag, users,
                                 val_scorers, val_labels_map, categories,
                                 val_hours, val_channels, val_watch, show_channel)

    scores_bag = []
    for s in range(42, 47):
        print(f"  training bag seed {s}")
        params = dict(LGB_PARAMS, num_leaves=31, n_estimators=300,
                      learning_rate=0.05, random_state=s)
        model = fit_ranker(fit_data_bag, features_v6, params)
        data = val_data_bag.copy()
        data["ranker_score"] = model.predict(
            data[features_v6],
            categorical_feature=[c for c in ["show_category"] if c in features_v6])
        scores_bag.append(data[["user_id", "tv_show_id", "ranker_score"]].copy())
        del model
        gc.collect()

    # average scores per (user, show)
    all_scores = pd.concat(scores_bag, ignore_index=True)
    avg = (all_scores.groupby(["user_id", "tv_show_id"])["ranker_score"]
           .mean().reset_index())
    by_user_bag = {u: grp for u, grp in avg.groupby("user_id", sort=False)}
    preds_bag = {}
    for user in users:
        grp = by_user_bag.get(user)
        ranked = []
        if grp is not None and len(grp):
            show_ids = grp["tv_show_id"].to_numpy()
            ranked = rank_by_score(grp["ranker_score"].to_numpy(), show_ids,
                                   tie.for_ids(user, show_ids), k=TOP_K)
        preds_bag[user] = pad_with_popularity(ranked, popularity_top5)
    ap_bag = np.array([ap_at_k(truths["recency"][u], preds_bag[u], TOP_K)
                       for u in users])
    obs_bag, lo_bag, hi_bag, b_b, w_b = bootstrap_ci_vs_base(ap_v6_arr, ap_bag,
                                                              users)
    print(f"  v6 bagged (42-46) MAP@5: {ap_bag.mean():.6f} vs v6 "
          f"{obs_bag:+.6f} [{lo_bag:+.6f}, {hi_bag:+.6f}] "
          f"(better/worse {b_b}/{w_b})")

    section("(2b) rank-average ensemble of v5 and v6")
    # need per (user,show) scores of v5 and v6 models? we don't have them stored
    # but we have predictions; rank-average by averaging reciprocal ranks or
    # better: get scores from models - easier to rebuild quickly
    # rebuild v5 and v6 models to get scores
    params_v5 = dict(LGB_PARAMS)  # 400 trees default
    fit_window_v5 = Window((11, 15), (16, 20), train_frame, train_frame, None)
    fit_users_v5 = sorted(build_labels(air, (16, 20))[0])
    fit_hist_v5 = qualifying(air, (HISTORY_START_WEEK, fit_window_v5.history_end))
    fit_counts_v5 = user_show_counts_full(fit_hist_v5)
    fit_scorers_v5 = Scorers(fit_counts_v5, fit_window_v5.target_shows, meta)
    fit_labels_v5, _ = build_labels(air, (16, 20))
    fit_hours_v5 = user_hour_distribution(air, (HISTORY_START_WEEK,
                                                fit_window_v5.history_end),
                                          fit_users_v5)
    fit_channels_v5, fit_watch_v5 = user_channel_time(air, show_channel,
                                                     (HISTORY_START_WEEK,
                                                      fit_window_v5.history_end),
                                                     fit_users_v5)
    base_cfg_v5 = {"candidates": "base", "features": "base", "windows": False,
                   "popular": POPULAR_BASE, "extra_candidates": False,
                   "extra_features": False}
    fit_data_v5 = build_dataset(air, meta, fit_window_v5, base_cfg_v5,
                                fit_users_v5, fit_scorers_v5, fit_labels_v5,
                                categories, fit_hours_v5, fit_channels_v5,
                                fit_watch_v5, show_channel)
    fit_data_v5["window"] = 0
    model_v5 = fit_ranker(fit_data_v5, list(FEATURES_BASE), params_v5)
    val_data_v5s = build_dataset(air, meta, val_window,
                                 {"candidates": "base", "features": "base",
                                  "windows": False, "popular": POPULAR_BASE,
                                  "extra_candidates": False,
                                  "extra_features": False},
                                 users, val_scorers, val_labels_map, categories,
                                 val_hours, val_channels, val_watch, show_channel)
    s5 = val_data_v5s.copy()
    s5["ranker_score"] = model_v5.predict(
        s5[list(FEATURES_BASE)],
        categorical_feature=[c for c in ["show_category"] if c in FEATURES_BASE])

    # v6 scores
    model_v6 = fit_ranker(fit_data_bag, features_v6, params_v6)
    s6 = val_data_bag.copy()
    s6["ranker_score"] = model_v6.predict(
        s6[features_v6],
        categorical_feature=[c for c in ["show_category"] if c in features_v6])

    # rank-average
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
            ranked = rank_by_score(grp["ravg"].to_numpy(), show_ids,
                                   tie.for_ids(user, show_ids), k=TOP_K)
        preds_ra[user] = pad_with_popularity(ranked, popularity_top5)
    ap_ra = np.array([ap_at_k(truths["recency"][u], preds_ra[u], TOP_K)
                      for u in users])
    obs_ra, lo_ra, hi_ra, b_ra, w_ra = bootstrap_ci_vs_base(ap_v6_arr, ap_ra,
                                                            users)
    print(f"  rank-avg v5+v6 MAP@5: {ap_ra.mean():.6f} vs v6 "
          f"{obs_ra:+.6f} [{lo_ra:+.6f}, {hi_ra:+.6f}] "
          f"(better/worse {b_ra}/{w_ra})")

    section("(2c) rank-average of v5, v6 and own history")
    # own history scores: use count_window-like signal? or just build scores
    # use user_show_counts_full on TRAIN_WEEKS as base
    qh = qualifying(air, TRAIN_WEEKS)
    own_counts = user_show_counts_full(qh)
    by_user_own = {u: g for u, g in own_counts.groupby("user_id", sort=False)}
    # build score = count (and use tie-break) aligned to candidates
    # get all candidates union
    cand_all = set(val_data_bag[val_data_bag["user_id"].isin(users)]["tv_show_id"])
    # build frame
    rows = []
    for user in users:
        grp_own = by_user_own.get(user)
        own_map = {int(s): float(c) for s, c in zip(grp_own["tv_show_id"],
                                                     grp_own["count"])} if grp_own is not None else {}
        # for all candidates in val set for this user? use union of preds contexts
        cands = list(set(s5m[s5m["user_id"] == user]["tv_show_id"]) |
                     set(s6m[s6m["user_id"] == user]["tv_show_id"]))
        for c in cands:
            rows.append((user, c, own_map.get(c, 0.0)))
    own_df = pd.DataFrame(rows, columns=["user_id", "tv_show_id", "sown"])
    merged2 = merged.merge(own_df, on=["user_id", "tv_show_id"], how="left")
    merged2["sown"] = merged2["sown"].fillna(0.0)
    merged2["ravg3"] = (merged2["s5"] + merged2["s6"] + merged2["sown"]) / 3.0
    by_user_ra3 = {u: grp for u, grp in merged2.groupby("user_id", sort=False)}
    preds_ra3 = {}
    for user in users:
        grp = by_user_ra3.get(user)
        ranked = []
        if grp is not None and len(grp):
            show_ids = grp["tv_show_id"].to_numpy()
            ranked = rank_by_score(grp["ravg3"].to_numpy(), show_ids,
                                   tie.for_ids(user, show_ids), k=TOP_K)
        preds_ra3[user] = pad_with_popularity(ranked, popularity_top5)
    ap_ra3 = np.array([ap_at_k(truths["recency"][u], preds_ra3[u], TOP_K)
                       for u in users])
    obs_ra3, lo_ra3, hi_ra3, b_ra3, w_ra3 = bootstrap_ci_vs_base(ap_v6_arr, ap_ra3,
                                                                users)
    print(f"  rank-avg v5+v6+own MAP@5: {ap_ra3.mean():.6f} vs v6 "
          f"{obs_ra3:+.6f} [{lo_ra3:+.6f}, {hi_ra3:+.6f}] "
          f"(better/worse {b_ra3}/{w_ra3})")

    section("(3) check 26-30 window abnormality vs alternatives")
    # compare label window sizes/stats
    # build alternative fit: features 20-24, labels 25-29 (using validation schedule for schedule tables as per rules where allowed? 
    # but for fitting features/labels we can use airings up to 29; schedule for those weeks comes from validation schedule file)
    alt_feat = (20, 24)
    alt_lab = (25, 29)
    l_alt, t_alt = build_labels(air, alt_lab)
    qq_alt = qualifying(air, alt_lab)
    print(f"alt labels {alt_lab}: users {len(t_alt)}, qual events {len(qq_alt)}, "
          f"active {qq_alt['user_id'].nunique()}")
    # compare to 26-30
    l_b, t_b = build_labels(air, (26, 30))
    qq_b = qualifying(air, (26, 30))
    print(f"v6 labels  (26,30): users {len(t_b)}, qual events {len(qq_b)}, "
          f"active {qq_b['user_id'].nunique()}")

    section("DECISION")
    results = [
        ("v6 (baseline)", ap_v6_arr.mean(), 0.0, -np.inf, np.inf, preds_v6),
        ("v6 bagged 42-46", ap_bag.mean(), obs_bag, lo_bag, hi_bag, preds_bag),
        ("rank-avg v5+v6", ap_ra.mean(), obs_ra, lo_ra, hi_ra, preds_ra),
        ("rank-avg v5+v6+own", ap_ra3.mean(), obs_ra3, lo_ra3, hi_ra3, preds_ra3),
    ]
    for r in results:
        print(f"{r[0]:<18s}: MAP {r[1]:.6f} vs v6 {r[2]:+0.6f} CI [{r[3]:+0.6f}, {r[4]:+0.6f}]")
    # pick best by MAP; if interval includes 0 prefer more robust (bagged/ensemble)
    best = max(results, key=lambda x: x[1])
    print(f"\nbest by MAP: {best[0]} ({best[1]:.6f})")
    if best[0] == "v6 (baseline)":
        chosen_name, chosen_preds = "v6 (baseline)", preds_v6
    else:
        # if interval includes 0, prefer more robust
        if best[3] <= 0.0 <= best[4]:
            # prefer bagged if better or similar robustness
            bag = [r for r in results if r[0] == "v6 bagged 42-46"][0]
            if bag[1] >= best[1] - 1e-6:  # prefer bagged
                chosen_name, chosen_preds = bag[0], bag[5]
            else:
                chosen_name, chosen_preds = best[0], best[5]
        else:
            chosen_name, chosen_preds = best[0], best[5]
    print(f"CHOSEN: {chosen_name}")

    # save val predictions for chosen
    write_rankings(VAL_V7_CSV, users, chosen_preds)
    print(f"wrote {VAL_V7_CSV}")
    # verify
    back = read_rankings(VAL_V7_CSV)
    assert list(back) == users
    print(f"MAP@5 recomputed from val_v7: {map_at_k({u:set(truths['recency'][u]) for u in users}, back, TOP_K):.6f}")

    # build submission only for chosen variant
    section(f"BUILD SUBMISSION {SUB_V7.name} for {chosen_name}")
    if chosen_name == "v6 (baseline)":
        # copy v6 submission structure but ensure identical
        sub = pd.read_csv(SUB_V6, dtype=str)
    elif chosen_name == "v6 bagged 42-46":
        # need to build test submission with bagging? "build outputs/submission_v7_robust.csv only for the chosen variant"
        # train on test fit windows with seeds 42-46, average scores
        air_t = load_airings()
        template_t = pd.read_csv(SUBMISSION, dtype=str)
        template_t["user_id"] = template_t["user_id"].astype(str)
        submission_users_t = template_t["user_id"].tolist()
        meta_t = load_schedule([SCHEDULE_TRAIN, SCHEDULE_VALIDATION, SCHEDULE_TEST])
        categories_t = sorted({v for v in meta_t["tv_show_category"].dropna().unique()
                               if isinstance(v, str)} | {"unknown"})
        val_frame_t = load_schedule_frame(SCHEDULE_VALIDATION)
        test_frame_t = load_schedule_frame(SCHEDULE_TEST)
        show_channel_t = show_channel_map(val_frame_t, test_frame_t)
        test_pool_t = candidate_pool(TEST_WEEKS, SCHEDULE_TEST)

        FIT_FEATURE_WEEKS = (21, 25)
        FIT_LABEL_WEEKS = (26, 30)
        TEST_FEATURE_WEEKS = (26, 30)
        TARGET_WEEKS = TEST_WEEKS
        CFG_t = {"candidates": "base", "features": "extra", "windows": False,
                 "popular": POPULAR_BASE, "extra_candidates": False,
                 "extra_features": True}
        FEATURES_t = list(FEATURES_BASE) + FEATURES_EXTRA

        fit_window_t = Window(FIT_FEATURE_WEEKS, FIT_LABEL_WEEKS, val_frame_t,
                              val_frame_t, None)
        fit_labels_t, fit_truth_t = build_labels(air_t, FIT_LABEL_WEEKS)
        fit_users_t = sorted(fit_truth_t)
        fit_hist_t = qualifying(air_t, (HISTORY_START_WEEK,
                                        fit_window_t.history_end))
        fit_scorers_t = Scorers(user_show_counts_full(fit_hist_t),
                                fit_window_t.target_shows, meta_t)
        fit_hours_t = user_hour_distribution(
            air_t, (HISTORY_START_WEEK, fit_window_t.history_end), fit_users_t)
        fit_channels_t, fit_watch_t = user_channel_time(
            air_t, show_channel_t, (HISTORY_START_WEEK, fit_window_t.history_end),
            fit_users_t)
        fit_data_t = build_dataset(air_t, meta_t, fit_window_t, CFG_t,
                                   fit_users_t, fit_scorers_t, fit_labels_t,
                                   categories_t, fit_hours_t, fit_channels_t,
                                   fit_watch_t, show_channel_t)
        fit_data_t["window"] = 0

        # train bag of models
        bag_models = []
        for s in range(42, 47):
            params = dict(LGB_PARAMS, num_leaves=31, n_estimators=300,
                          learning_rate=0.05, random_state=s)
            bag_models.append(fit_ranker(fit_data_t, FEATURES_t, params))

        # predict
        test_window_t = Window(TEST_FEATURE_WEEKS, TARGET_WEEKS, val_frame_t,
                               test_frame_t, test_pool_t)
        q_hist_t = qualifying(air_t, (HISTORY_START_WEEK,
                                      test_window_t.history_end))
        test_counts_t = user_show_counts_full(q_hist_t)
        test_pop_t = show_popularity(q_hist_t)
        test_scorers_t = Scorers(test_counts_t, test_pool_t, meta_t)
        test_hours_t = user_hour_distribution(
            air_t, (HISTORY_START_WEEK, test_window_t.history_end),
            submission_users_t)
        test_channels_t, test_watch_t = user_channel_time(
            air_t, show_channel_t, (HISTORY_START_WEEK,
                                    test_window_t.history_end),
            submission_users_t)
        test_data_t = build_dataset(air_t, meta_t, test_window_t, CFG_t,
                                    submission_users_t, test_scorers_t, {}, 
                                    categories_t, test_hours_t, test_channels_t,
                                    test_watch_t, show_channel_t)
        tie_t = TieBreak(q_hist_t, test_pool_t)
        popularity_top5_t = rank_shows(test_pop_t[test_pop_t["tv_show_id"].isin(test_pool_t)])

        # average scores
        score_frames = []
        for m in bag_models:
            d = test_data_t.copy()
            d["ranker_score"] = m.predict(
                d[FEATURES_t],
                categorical_feature=[c for c in ["show_category"] if c in FEATURES_t])
            score_frames.append(d[["user_id", "tv_show_id", "ranker_score"]])
        avg_t = (pd.concat(score_frames).groupby(["user_id", "tv_show_id"])["ranker_score"]
                 .mean().reset_index())
        by_user_bt = {u: grp for u, grp in avg_t.groupby("user_id", sort=False)}
        preds_bt = {}
        for user in submission_users_t:
            grp = by_user_bt.get(user)
            ranked = []
            if grp is not None and len(grp):
                show_ids = grp["tv_show_id"].to_numpy()
                ranked = rank_by_score(grp["ranker_score"].to_numpy(), show_ids,
                                       tie_t.for_ids(user, show_ids), k=TOP_K)
            preds_bt[user] = pad_with_popularity(ranked, popularity_top5_t)
        # write
        out = template_t[["user_id"]].copy()
        out["tv_show_id"] = [" ".join(str(s) for s in preds_bt[u])
                             for u in template_t["user_id"]]
        assert list(out.columns) == list(template_t.columns)
        assert len(out) == len(template_t)
        assert out["user_id"].tolist() == template_t["user_id"].tolist()
        for user, row in zip(out["user_id"], out["tv_show_id"]):
            shows = [int(s) for s in row.split()]
            assert len(shows) == TOP_K and len(set(shows)) == TOP_K
            assert set(shows) <= test_pool_t
        out.to_csv(SUB_V7, index=False, lineterminator="\n")
        with open(SUB_V7, "rb") as fh:
            blob = fh.read()
        if blob.endswith(b"\n"):
            with open(SUB_V7, "wb") as fh:
                fh.write(blob[:-1])
        sub = out
    elif chosen_name == "rank-avg v5+v6":
        # build by rank-averaging scores of v5 and v6 test models
        air_t = load_airings()
        template_t = pd.read_csv(SUBMISSION, dtype=str)
        template_t["user_id"] = template_t["user_id"].astype(str)
        submission_users_t = template_t["user_id"].tolist()
        meta_t = load_schedule([SCHEDULE_TRAIN, SCHEDULE_VALIDATION, SCHEDULE_TEST])
        categories_t = sorted({v for v in meta_t["tv_show_category"].dropna().unique()
                               if isinstance(v, str)} | {"unknown"})
        val_frame_t = load_schedule_frame(SCHEDULE_VALIDATION)
        test_frame_t = load_schedule_frame(SCHEDULE_TEST)
        show_channel_t = show_channel_map(val_frame_t, test_frame_t)
        test_pool_t = candidate_pool(TEST_WEEKS, SCHEDULE_TEST)

        # v5 test
        FIT_FEATURE_WEEKS = (21, 25)
        FIT_LABEL_WEEKS = (26, 30)
        TEST_FEATURE_WEEKS = (26, 30)
        TARGET_WEEKS = TEST_WEEKS

        # fit v5
        fit_window_t5 = Window(FIT_FEATURE_WEEKS, FIT_LABEL_WEEKS, val_frame_t,
                               val_frame_t, None)
        fit_labels_t5, fit_truth_t5 = build_labels(air_t, FIT_LABEL_WEEKS)
        fit_users_t5 = sorted(fit_truth_t5)
        fit_hist_t5 = qualifying(air_t, (HISTORY_START_WEEK,
                                         fit_window_t5.history_end))
        fit_scorers_t5 = Scorers(user_show_counts_full(fit_hist_t5),
                                 fit_window_t5.target_shows, meta_t)
        fit_hours_t5 = user_hour_distribution(
            air_t, (HISTORY_START_WEEK, fit_window_t5.history_end), fit_users_t5)
        fit_channels_t5, fit_watch_t5 = user_channel_time(
            air_t, show_channel_t, (HISTORY_START_WEEK, fit_window_t5.history_end),
            fit_users_t5)
        base_cfg_t5 = {"candidates": "base", "features": "base", "windows": False,
                       "popular": POPULAR_BASE, "extra_candidates": False,
                       "extra_features": False}
        fit_data_t5 = build_dataset(air_t, meta_t, fit_window_t5, base_cfg_t5,
                                    fit_users_t5, fit_scorers_t5, fit_labels_t5,
                                    categories_t, fit_hours_t5, fit_channels_t5,
                                    fit_watch_t5, show_channel_t)
        fit_data_t5["window"] = 0
        model_t5 = fit_ranker(fit_data_t5, list(FEATURES_BASE), dict(LGB_PARAMS))

        # fit v6
        CFG_t6 = {"candidates": "base", "features": "extra", "windows": False,
                  "popular": POPULAR_BASE, "extra_candidates": False,
                  "extra_features": True}
        FEATURES_t6 = list(FEATURES_BASE) + FEATURES_EXTRA
        params_t6 = dict(LGB_PARAMS, num_leaves=31, n_estimators=300,
                         learning_rate=0.05, random_state=42)
        fit_window_t6 = fit_window_t5
        fit_scorers_t6 = fit_scorers_t5
        fit_hours_t6 = fit_hours_t5
        fit_channels_t6 = fit_channels_t5
        fit_watch_t6 = fit_watch_t5
        fit_data_t6 = build_dataset(air_t, meta_t, fit_window_t6, CFG_t6,
                                    fit_users_t5, fit_scorers_t6, fit_labels_t5,
                                    categories_t, fit_hours_t6, fit_channels_t6,
                                    fit_watch_t6, show_channel_t)
        fit_data_t6["window"] = 0
        model_t6 = fit_ranker(fit_data_t6, FEATURES_t6, params_t6)

        # predict
        test_window_t = Window(TEST_FEATURE_WEEKS, TARGET_WEEKS, val_frame_t,
                               test_frame_t, test_pool_t)
        q_hist_t = qualifying(air_t, (HISTORY_START_WEEK,
                                      test_window_t.history_end))
        test_counts_t = user_show_counts_full(q_hist_t)
        test_pop_t = show_popularity(q_hist_t)
        test_scorers_t = Scorers(test_counts_t, test_pool_t, meta_t)
        test_hours_t = user_hour_distribution(
            air_t, (HISTORY_START_WEEK, test_window_t.history_end),
            submission_users_t)
        test_channels_t, test_watch_t = user_channel_time(
            air_t, show_channel_t, (HISTORY_START_WEEK,
                                    test_window_t.history_end),
            submission_users_t)
        # v5 data
        test_data_t5 = build_dataset(air_t, meta_t, test_window_t,
                                     base_cfg_t5, submission_users_t,
                                     test_scorers_t, {}, categories_t,
                                     test_hours_t, test_channels_t,
                                     test_watch_t, show_channel_t)
        test_data_t6 = build_dataset(air_t, meta_t, test_window_t,
                                     CFG_t6, submission_users_t,
                                     test_scorers_t, {}, categories_t,
                                     test_hours_t, test_channels_t,
                                     test_watch_t, show_channel_t)
        tie_t = TieBreak(q_hist_t, test_pool_t)
        popularity_top5_t = rank_shows(test_pop_t[test_pop_t["tv_show_id"].isin(test_pool_t)])

        d5 = test_data_t5.copy()
        d5["s5"] = model_t5.predict(d5[list(FEATURES_BASE)],
                                    categorical_feature=[c for c in ["show_category"] if c in FEATURES_BASE])
        d6 = test_data_t6.copy()
        d6["s6"] = model_t6.predict(d6[FEATURES_t6],
                                    categorical_feature=[c for c in ["show_category"] if c in FEATURES_t6])
        m = d5[["user_id", "tv_show_id", "s5"]].merge(d6[["user_id", "tv_show_id", "s6"]], on=["user_id", "tv_show_id"])
        m["r"] = 0.5 * m["s5"] + 0.5 * m["s6"]
        by_user_m = {u: grp for u, grp in m.groupby("user_id", sort=False)}
        preds_m = {}
        for user in submission_users_t:
            grp = by_user_m.get(user)
            ranked = []
            if grp is not None and len(grp):
                show_ids = grp["tv_show_id"].to_numpy()
                ranked = rank_by_score(grp["r"].to_numpy(), show_ids,
                                       tie_t.for_ids(user, show_ids), k=TOP_K)
            preds_m[user] = pad_with_popularity(ranked, popularity_top5_t)
        out = template_t[["user_id"]].copy()
        out["tv_show_id"] = [" ".join(str(s) for s in preds_m[u])
                             for u in template_t["user_id"]]
        assert list(out.columns) == list(template_t.columns)
        assert out["user_id"].tolist() == template_t["user_id"].tolist()
        for user, row in zip(out["user_id"], out["tv_show_id"]):
            shows = [int(s) for s in row.split()]
            assert len(shows) == TOP_K and len(set(shows)) == TOP_K
            assert set(shows) <= test_pool_t
        out.to_csv(SUB_V7, index=False, lineterminator="\n")
        with open(SUB_V7, "rb") as fh:
            blob = fh.read()
        if blob.endswith(b"\n"):
            with open(SUB_V7, "wb") as fh:
                fh.write(blob[:-1])
        sub = out
    elif chosen_name == "rank-avg v5+v6+own":
        # include own history in test
        air_t = load_airings()
        template_t = pd.read_csv(SUBMISSION, dtype=str)
        template_t["user_id"] = template_t["user_id"].astype(str)
        submission_users_t = template_t["user_id"].tolist()
        meta_t = load_schedule([SCHEDULE_TRAIN, SCHEDULE_VALIDATION, SCHEDULE_TEST])
        categories_t = sorted({v for v in meta_t["tv_show_category"].dropna().unique()
                               if isinstance(v, str)} | {"unknown"})
        val_frame_t = load_schedule_frame(SCHEDULE_VALIDATION)
        test_frame_t = load_schedule_frame(SCHEDULE_TEST)
        show_channel_t = show_channel_map(val_frame_t, test_frame_t)
        test_pool_t = candidate_pool(TEST_WEEKS, SCHEDULE_TEST)

        FIT_FEATURE_WEEKS = (21, 25)
        FIT_LABEL_WEEKS = (26, 30)
        TEST_FEATURE_WEEKS = (26, 30)
        TARGET_WEEKS = TEST_WEEKS

        # fit both
        fit_window_t5 = Window(FIT_FEATURE_WEEKS, FIT_LABEL_WEEKS, val_frame_t,
                               val_frame_t, None)
        fit_labels_t5, fit_truth_t5 = build_labels(air_t, FIT_LABEL_WEEKS)
        fit_users_t5 = sorted(fit_truth_t5)
        fit_hist_t5 = qualifying(air_t, (HISTORY_START_WEEK,
                                         fit_window_t5.history_end))
        fit_counts_t5 = user_show_counts_full(fit_hist_t5)
        fit_scorers_t5 = Scorers(fit_counts_t5, fit_window_t5.target_shows, meta_t)
        fit_hours_t5 = user_hour_distribution(
            air_t, (HISTORY_START_WEEK, fit_window_t5.history_end), fit_users_t5)
        fit_channels_t5, fit_watch_t5 = user_channel_time(
            air_t, show_channel_t, (HISTORY_START_WEEK, fit_window_t5.history_end),
            fit_users_t5)
        base_cfg_t5 = {"candidates": "base", "features": "base", "windows": False,
                       "popular": POPULAR_BASE, "extra_candidates": False,
                       "extra_features": False}
        fit_data_t5 = build_dataset(air_t, meta_t, fit_window_t5, base_cfg_t5,
                                    fit_users_t5, fit_scorers_t5, fit_labels_t5,
                                    categories_t, fit_hours_t5, fit_channels_t5,
                                    fit_watch_t5, show_channel_t)
        fit_data_t5["window"] = 0
        model_t5 = fit_ranker(fit_data_t5, list(FEATURES_BASE), dict(LGB_PARAMS))
        CFG_t6 = {"candidates": "base", "features": "extra", "windows": False,
                  "popular": POPULAR_BASE, "extra_candidates": False,
                  "extra_features": True}
        FEATURES_t6 = list(FEATURES_BASE) + FEATURES_EXTRA
        params_t6 = dict(LGB_PARAMS, num_leaves=31, n_estimators=300,
                         learning_rate=0.05, random_state=42)
        fit_data_t6 = build_dataset(air_t, meta_t, fit_window_t5, CFG_t6,
                                    fit_users_t5, fit_scorers_t5, fit_labels_t5,
                                    categories_t, fit_hours_t5, fit_channels_t5,
                                    fit_watch_t5, show_channel_t)
        fit_data_t6["window"] = 0
        model_t6 = fit_ranker(fit_data_t6, FEATURES_t6, params_t6)

        test_window_t = Window(TEST_FEATURE_WEEKS, TARGET_WEEKS, val_frame_t,
                               test_frame_t, test_pool_t)
        q_hist_t = qualifying(air_t, (HISTORY_START_WEEK,
                                      test_window_t.history_end))
        test_counts_t = user_show_counts_full(q_hist_t)
        test_pop_t = show_popularity(q_hist_t)
        test_scorers_t = Scorers(test_counts_t, test_pool_t, meta_t)
        test_hours_t = user_hour_distribution(
            air_t, (HISTORY_START_WEEK, test_window_t.history_end),
            submission_users_t)
        test_channels_t, test_watch_t = user_channel_time(
            air_t, show_channel_t, (HISTORY_START_WEEK,
                                    test_window_t.history_end),
            submission_users_t)
        test_data_t5 = build_dataset(air_t, meta_t, test_window_t,
                                     base_cfg_t5, submission_users_t,
                                     test_scorers_t, {}, categories_t,
                                     test_hours_t, test_channels_t,
                                     test_watch_t, show_channel_t)
        test_data_t6 = build_dataset(air_t, meta_t, test_window_t,
                                     CFG_t6, submission_users_t,
                                     test_scorers_t, {}, categories_t,
                                     test_hours_t, test_channels_t,
                                     test_watch_t, show_channel_t)
        tie_t = TieBreak(q_hist_t, test_pool_t)
        popularity_top5_t = rank_shows(test_pop_t[test_pop_t["tv_show_id"].isin(test_pool_t)])

        d5 = test_data_t5.copy()
        d5["s5"] = model_t5.predict(d5[list(FEATURES_BASE)],
                                    categorical_feature=[c for c in ["show_category"] if c in FEATURES_BASE])
        d6 = test_data_t6.copy()
        d6["s6"] = model_t6.predict(d6[FEATURES_t6],
                                    categorical_feature=[c for c in ["show_category"] if c in FEATURES_t6])
        m = d5[["user_id", "tv_show_id", "s5"]].merge(d6[["user_id", "tv_show_id", "s6"]], on=["user_id", "tv_show_id"])
        # add own
        own_map_all = {int(s): float(c) for s, c in zip(test_counts_t["tv_show_id"], test_counts_t["count"])}
        # per user
        own_by = {u: {int(s): float(c) for s, c in zip(g["tv_show_id"], g["count"])}
                  for u, g in test_counts_t.groupby("user_id", sort=False)}
        rows_m = []
        for _, r in m.iterrows():
            u = r["user_id"]
            c = int(r["tv_show_id"])
            sown = own_by.get(u, {}).get(c, 0.0)
            rows_m.append((u, c, r["s5"], r["s6"], sown))
        mm = pd.DataFrame(rows_m, columns=["user_id", "tv_show_id", "s5", "s6", "sown"])
        mm["r"] = (mm["s5"] + mm["s6"] + mm["sown"]) / 3.0
        by_user_m = {u: grp for u, grp in mm.groupby("user_id", sort=False)}
        preds_m = {}
        for user in submission_users_t:
            grp = by_user_m.get(user)
            ranked = []
            if grp is not None and len(grp):
                show_ids = grp["tv_show_id"].to_numpy()
                ranked = rank_by_score(grp["r"].to_numpy(), show_ids,
                                       tie_t.for_ids(user, show_ids), k=TOP_K)
            preds_m[user] = pad_with_popularity(ranked, popularity_top5_t)
        out = template_t[["user_id"]].copy()
        out["tv_show_id"] = [" ".join(str(s) for s in preds_m[u])
                             for u in template_t["user_id"]]
        assert list(out.columns) == list(template_t.columns)
        assert out["user_id"].tolist() == template_t["user_id"].tolist()
        for user, row in zip(out["user_id"], out["tv_show_id"]):
            shows = [int(s) for s in row.split()]
            assert len(shows) == TOP_K and len(set(shows)) == TOP_K
            assert set(shows) <= test_pool_t
        out.to_csv(SUB_V7, index=False, lineterminator="\n")
        with open(SUB_V7, "rb") as fh:
            blob = fh.read()
        if blob.endswith(b"\n"):
            with open(SUB_V7, "wb") as fh:
                fh.write(blob[:-1])
        sub = out
    else:
        sub = pd.read_csv(SUB_V6, dtype=str)
        sub.to_csv(SUB_V7, index=False, lineterminator="\n")
        with open(SUB_V7, "rb") as fh:
            blob = fh.read()
        if blob.endswith(b"\n"):
            with open(SUB_V7, "wb") as fh:
                fh.write(blob[:-1])

    # checks
    print(f"rows: {len(sub)}")
    print(f"header: {', '.join(sub.columns)}")
    with open(SUB_V7, "rb") as fh:
        blob = fh.read()
    print(f"trailing newline: {blob.endswith(b'\n')}")
    if SUB_V6.exists():
        v6s = pd.read_csv(SUB_V6, dtype=str).set_index("user_id")["tv_show_id"]
        ms = sub.set_index("user_id")["tv_show_id"]
        shared = [u for u in ms.index if u in v6s.index]
        n_diff = sum(1 for u in shared if ms[u] != v6s[u])
        same_t1 = sum(1 for u in shared if ms[u].split()[0] == v6s[u].split()[0])
        ov = sum(len(set(ms[u].split()) & set(v6s[u].split())) for u in shared) / len(shared)
        print(f"rows differing from {SUB_V6.name}: {n_diff}/{len(shared)}, same top1 {same_t1}/{len(shared)}, mean overlap {ov:.2f}/5")

    print(f"runtime: {time.perf_counter() - started_total:.1f}s")


if __name__ == "__main__":
    main()
