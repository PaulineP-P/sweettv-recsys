import os
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd

from evaluate_v2 import (SUBMISSION, TOP_K, TRAIN_WEEKS, VALID_WEEKS, TEST_WEEKS,
                         candidate_pool, load_airings, qualifying,
                         rank_shows, show_popularity)
from ltr import build_labels
from models import (SCHEDULE_TRAIN, SCHEDULE_VALIDATION, load_schedule,
                    user_show_counts_full, TieBreak, method_own_history,
                    method_als, method_content, ALS_FACTORS)
from hybrid import method_scored, weighted_counts, schedule_airing_counts, FINAL_H, build_scores

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
OUT_MD = OUTPUTS / "two_users.md"


def load_title_cat(sources):
    title_map = {}
    cat_map = {}
    for path in sources:
        try:
            for chunk in pd.read_csv(path, usecols=["tv_show_id", "tv_show_title", "tv_show_category"],
                                     dtype=str, keep_default_na=False, chunksize=200000):
                sid = chunk["tv_show_id"].str.strip()
                keep = sid.ne("0") & sid.ne("")
                if not keep.any():
                    continue
                for s, t, c in zip(sid[keep], chunk["tv_show_title"][keep], chunk["tv_show_category"][keep]):
                    try:
                        s_int = int(s.strip())
                    except Exception:
                        continue
                    if s_int not in title_map:
                        title_map[s_int] = t.strip()
                        cat_map[s_int] = c.strip()
        except Exception:
            continue
    return title_map, cat_map


def main():
    started = time.perf_counter()
    air = load_airings()
    q_all = qualifying(air, (11, 30))
    template = pd.read_csv(SUBMISSION, dtype=str)
    submission_users_set = set(template["user_id"].astype(str).tolist())
    submission_users_list = list(template["user_id"].astype(str))

    counts_per_user = q_all.groupby("user_id").size()
    distinct_shows = q_all.groupby("user_id")["tv_show_id"].nunique()
    counts_per_user.index = counts_per_user.index.astype(str)
    distinct_shows.index = distinct_shows.index.astype(str)

    heavy = None
    for u in counts_per_user.sort_values(ascending=False).index:
        if str(u) in submission_users_set:
            heavy = str(u)
            break
    light = None
    for u in distinct_shows.sort_values().index:
        if str(u) in submission_users_set and int(distinct_shows.loc[u]) < 5:
            light = str(u)
            break
    print("heavy:", heavy, "qual airings:", int(counts_per_user.loc[heavy]))
    print("light:", light, "distinct shows:", int(distinct_shows.loc[light]))

    title_map, cat_map = load_title_cat([SCHEDULE_TRAIN, SCHEDULE_VALIDATION,
                                         Path(ROOT) / "data" / "export_arh_31-42-final.csv"])

    def show_info(sid):
        sid = int(sid)
        return str(sid), str(title_map.get(sid, "")), str(cat_map.get(sid, ""))

    def history_top10(user):
        qq = q_all[q_all["user_id"].astype(str) == str(user)].copy()
        g = qq.groupby(qq["tv_show_id"]).agg(count=("screen_time", "size")).reset_index()
        g.columns = ["tv_show_id", "count"]
        g = g.sort_values("count", ascending=False).reset_index(drop=True)
        return g.head(10)

    # validation
    val_pool = candidate_pool(VALID_WEEKS, SCHEDULE_VALIDATION)
    q_train = qualifying(air, TRAIN_WEEKS)
    train_counts = user_show_counts_full(q_train)
    val_pop = show_popularity(q_train)
    popularity_top5 = rank_shows(val_pop[val_pop["tv_show_id"].isin(val_pool)])
    tie_v = TieBreak(q_train, val_pool)
    pop_list = [int(x) for x in popularity_top5]
    val_labels, truth_recency = build_labels(air, VALID_WEEKS)
    users_v = [u for u in submission_users_list if u in truth_recency and len(truth_recency[u]) >= TOP_K]
    truth_v = {u: truth_recency[u] for u in users_v}
    in_val_heavy = heavy in users_v
    in_val_light = light in users_v
    print("heavy in val:", in_val_heavy, "light in val:", in_val_light)

    preds_pop_v = {u: list(pop_list) for u in users_v}
    preds_own_v_res = method_own_history(val_pool, train_counts, pop_list, users_v)
    preds_own_v = preds_own_v_res[0] if isinstance(preds_own_v_res, tuple) else preds_own_v_res
    preds_als_v_res = method_als(train_counts, val_pool, tie_v, pop_list, users_v, factors=max(ALS_FACTORS))
    preds_als_v = preds_als_v_res[0] if isinstance(preds_als_v_res, tuple) else preds_als_v_res
    preds_cnt_v_res = method_content(train_counts, val_pool, load_schedule([SCHEDULE_TRAIN, SCHEDULE_VALIDATION]), tie_v, pop_list, users_v)
    preds_cnt_v = preds_cnt_v_res[0] if isinstance(preds_cnt_v_res, tuple) else preds_cnt_v_res
    n_train_air = schedule_airing_counts(SCHEDULE_TRAIN, TRAIN_WEEKS)
    n_target_air = schedule_airing_counts(SCHEDULE_VALIDATION, VALID_WEEKS)
    weighted = weighted_counts(q_train, FINAL_H, TRAIN_WEEKS[1])
    pool_ok = [s for s in val_pool if s in n_train_air]
    count_scores = build_scores(weighted, n_train_air, n_target_air, set(pool_ok), False)
    preds_v3_v_res = method_scored(count_scores, tie_v, pop_list, users_v, als_preds=preds_als_v)
    preds_v3_v = preds_v3_v_res[0] if isinstance(preds_v3_v_res, tuple) else preds_v3_v_res

    preds_v7_v = {}
    dfv7 = pd.read_csv(OUTPUTS / "val_v7_predictions.csv", dtype=str)
    for _, r in dfv7.iterrows():
        u = str(r["user_id"])
        if u in users_v:
            preds_v7_v[u] = [int(x) for x in str(r["tv_show_id"]).split()]

    # test
    from evaluate_v2 import SCHEDULE_TEST
    test_pool = candidate_pool(TEST_WEEKS, SCHEDULE_TEST)
    q_hist_t = qualifying(air, (11, TEST_WEEKS[1]))
    test_counts = user_show_counts_full(q_hist_t)
    test_pop = show_popularity(q_hist_t)
    popularity_top5_t = rank_shows(test_pop[test_pop["tv_show_id"].isin(test_pool)])
    tie_t = TieBreak(q_hist_t, test_pool)
    pop_list_t = [int(x) for x in popularity_top5_t]
    preds_pop_t = {u: list(pop_list_t) for u in submission_users_list}
    preds_own_t, _ = method_own_history(test_pool, test_counts, pop_list_t, submission_users_list)
    preds_als_t = method_als(test_counts, test_pool, tie_t, pop_list_t, submission_users_list, factors=max(ALS_FACTORS))
    from evaluate_v2 import SCHEDULE_TEST
    meta_t_cnt = load_schedule([SCHEDULE_TRAIN, SCHEDULE_VALIDATION, SCHEDULE_TEST])
    preds_cnt_t_res = method_content(test_counts, test_pool, meta_t_cnt, tie_t, pop_list_t, submission_users_list)
    preds_cnt_t = preds_cnt_t_res[0] if isinstance(preds_cnt_t_res, tuple) else preds_cnt_t_res
    n_train_air_t = schedule_airing_counts(SCHEDULE_TRAIN, TRAIN_WEEKS)
    n_target_air_t = schedule_airing_counts(SCHEDULE_TEST, TEST_WEEKS)
    weighted_t = weighted_counts(q_hist_t, FINAL_H, int(q_hist_t["week"].max()))
    pool_ok_t = set([s for s in test_pool if s in n_train_air_t])
    try:
        count_scores_t = build_scores(weighted_t, n_train_air_t, n_target_air_t, pool_ok_t, False)
        preds_v3_t = method_scored(count_scores_t, tie_t, pop_list_t, submission_users_list, als_preds=preds_als_t)
    except Exception:
        preds_v3_t = preds_als_t
    if isinstance(preds_v3_t, tuple): preds_v3_t = preds_v3_t[0]
    preds_v7_t = {}
    dfv7t = pd.read_csv(OUTPUTS / "submission_v7_robust.csv", dtype=str)
    for _, r in dfv7t.iterrows():
        preds_v7_t[str(r["user_id"])] = [int(x) for x in str(r["tv_show_id"]).split()]

    lines = []
    lines.append("# Two users report\n\n")
    lines.append(f"Heavy viewer: {heavy} (qual airings 11-30: {int(counts_per_user.loc[heavy])})\n\n")
    lines.append(f"Light viewer: {light} (distinct shows with >=0.8 airing 11-30: {int(distinct_shows.loc[light])})\n\n")
    lines.append(f"Heavy in validation set: {in_val_heavy}\n\n")
    lines.append(f"Light in validation set: {in_val_light}\n\n")

    def norm_preds(d):
        if isinstance(d, tuple): d = d[0]
        if isinstance(d, dict): return d
        return {}
    preds_pop_v=norm_preds(preds_pop_v); preds_own_v=norm_preds(preds_own_v); preds_als_v=norm_preds(preds_als_v); preds_cnt_v=norm_preds(preds_cnt_v); preds_v3_v=norm_preds(preds_v3_v); preds_v7_v=norm_preds(preds_v7_v)
    preds_pop_t=norm_preds(preds_pop_t); preds_own_t=norm_preds(preds_own_t); preds_als_t=norm_preds(preds_als_t); preds_cnt_t=norm_preds(preds_cnt_t); preds_v3_t=norm_preds(preds_v3_t); preds_v7_t=norm_preds(preds_v7_t)

    for user in [heavy, light]:
        lines.append(f"## User {user}\n\n")
        lines.append("### (a) History top-10 shows by count of >=0.8 airings in weeks 11-30\n\n")
        lines.append("| rank | tv_show_id | title | category | count |\n")
        lines.append("|---|---|---|---|---|\n")
        ht = history_top10(user)
        for i, r in ht.iterrows():
            sid = int(r["tv_show_id"])
            sid_s, title, cat = show_info(sid)
            lines.append(f"| {i+1} | {sid_s} | {title} | {cat} | {int(r['count'])} |\n")
        lines.append("\n")

        if user in users_v:
            lines.append("### (b) VALIDATION weeks 21-30\n\n")
            lines.append("**True top-5 (recency rule):**\n\n")
            lines.append("| rank | tv_show_id | title | category |\n")
            lines.append("|---|---|---|---|\n")
            for i, sid in enumerate(truth_v[user], 1):
                sid_s, title, cat = show_info(sid)
                lines.append(f"| {i} | {sid_s} | {title} | {cat} |\n")
            lines.append("\n")
            methods = [
                ("popularity", preds_pop_v),
                ("own_history", preds_own_v),
                ("ALS", preds_als_v),
                ("content-based", preds_cnt_v),
                ("v3_hybrid", preds_v3_v),
                ("v7_ranker", preds_v7_v),
            ]
            truth_set = set(truth_v[user])
            for mname, mpreds in methods:
                lines.append(f"**{mname} top-5:**\n\n")
                lines.append("| rank | tv_show_id | title | category | hit |\n")
                lines.append("|---|---|---|---|\n")
                for i, sid in enumerate(mpreds.get(user, []), 1):
                    sid_s, title, cat = show_info(sid)
                    hit = "Y" if sid in truth_set else ""
                    lines.append(f"| {i} | {sid_s} | {title} | {cat} | {hit} |\n")
                lines.append("\n")
        else:
            lines.append("### (b) VALIDATION weeks 21-30\n\n**Not in validation evaluation set** (light user has only 1 distinct show with >=0.8 airing / <5 qualifying shows). Shown history and test only.\n\n")

        lines.append("### (c) TEST weeks 31-42\n\n")
        methods_t = [
            ("popularity", preds_pop_t),
            ("own_history", preds_own_t),
            ("ALS", preds_als_t),
            ("content-based", preds_cnt_t),
            ("v3_hybrid", preds_v3_t),
            ("v7_ranker", preds_v7_t),
        ]
        for mname, mpreds in methods_t:
            lines.append(f"**{mname} top-5:**\n\n")
            lines.append("| rank | tv_show_id | title | category |\n")
            lines.append("|---|---|---|---|\n")
            for i, sid in enumerate(mpreds.get(user, []), 1):
                sid_s, title, cat = show_info(sid)
                lines.append(f"| {i} | {sid_s} | {title} | {cat} |\n")
            lines.append("\n")

    md = "".join(lines)
    OUT_MD.write_text(md)
    print(f"Wrote {OUT_MD}")
    print(f"Runtime: {time.perf_counter()-started:.1f}s")


if __name__ == "__main__":
    main()
