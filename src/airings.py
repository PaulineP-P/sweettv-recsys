"""Join the view log to the schedule, aggregated per (user, airing).

The organisers' rule (from the competition Data tab) is:

  drop tv_show_id = 0
  for each user and each airing, screen time = seconds the user watched that
  airing / airing duration
  keep only airings with screen time >= 0.8

"Seconds the user watched that airing" is the sum over every view session that
overlaps the airing, so the join is aggregated on the airing key
(channel_id + airing start_time) BEFORE dividing by the airing duration.  A
user who watches one 2-hour programme in three sessions gets one row with the
total, not three rows with partial screen times.

Writes data/airings_11_30.parquet with:
    user_id (string), tv_show_id (int64), week (int16), airing_start
    (timestamp), screen_time (float64)

Streaming design:
  * the labelled schedule is loaded once into per-channel numpy arrays;
  * the view log is processed in chunks, each chunk matched by binary search;
  * only (user_code, airing_idx, overlap_sec) triples are kept in memory;
  * the aggregation and the parquet write happen once, at the end.

Run: python src/airings.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

VIEW_LOG = DATA / "dataset11-30.csv"
OUT_PARQUET = DATA / "airings_11_30.parquet"

SCHEDULE_FILES = [
    DATA / "export_arh_11-20-final.csv",
    DATA / "export_arh_21-30-final.csv",
]

CHUNK = 500_000
NS = 1_000_000_000

SCHEMA = pa.schema([
    ("user_id", pa.string()),
    ("tv_show_id", pa.int64()),
    ("week", pa.int16()),
    ("airing_start", pa.timestamp("ns")),
    ("screen_time", pa.float64()),
])


def section(title):
    print(f"\n{'=' * 74}\n{title}\n{'=' * 74}")


def load_schedule():
    """Labelled airings of weeks 11-30, with a global airing index."""
    frames = []
    for path in SCHEDULE_FILES:
        for chunk in pd.read_csv(path, chunksize=CHUNK, dtype=str, keep_default_na=False):
            chunk = chunk.copy()
            chunk["start_time"] = pd.to_datetime(
                chunk["start_time"], format="%d.%m.%Y %H:%M:%S", errors="coerce")
            chunk["duration"] = pd.to_numeric(chunk["duration"], errors="coerce")
            frames.append(chunk[["channel_id", "start_time", "duration", "tv_show_id"]])

    sched = pd.concat(frames, ignore_index=True)
    n_all = len(sched)

    ids = sched["tv_show_id"].str.strip()
    keep = (ids != "0") & (ids != "") & sched["start_time"].notna() \
        & sched["duration"].notna() & (sched["duration"] > 0)
    sched = sched[keep].copy()
    sched["tv_show_id"] = ids[keep].astype("int64")

    # the airing key is (channel_id, airing start_time); it must be unique
    dup = int(sched.duplicated(subset=["channel_id", "start_time"]).sum())
    assert dup == 0, f"{dup} duplicate airing keys"
    sched = sched.sort_values(["channel_id", "start_time"]).reset_index(drop=True)
    sched["airing_idx"] = np.arange(len(sched), dtype="int64")
    sched["week"] = sched["start_time"].dt.isocalendar().week.to_numpy(dtype="int16")

    print(f"schedule rows read           : {n_all}")
    print(f"labelled airings kept        : {len(sched)}")
    print(f"channels with airings        : {sched['channel_id'].nunique()}")
    print(f"week range of airings        : {sched['week'].min()} - {sched['week'].max()}")
    dropped_w10 = int((sched["week"] < 11).sum())
    if dropped_w10:
        print(f"airings outside weeks 11-30  : {dropped_w10} (week "
              f"{sched.loc[sched['week'] < 11, 'week'].min()})")
    return sched


def build_index(sched):
    """channel -> (starts, ends, airing_idx, durations) sorted by start."""
    index = {}
    for channel, grp in sched.groupby("channel_id", sort=False):
        index[channel] = (
            grp["start_time"].to_numpy(dtype="datetime64[ns]").astype("int64"),
            grp["start_time"].to_numpy(dtype="datetime64[ns]").astype("int64")
            + grp["duration"].to_numpy(dtype="int64") * NS,
            grp["airing_idx"].to_numpy(dtype="int64"),
            grp["duration"].to_numpy(dtype="int64"),
        )
    return index


def match_chunk(chunk, index, user_codes):
    """(user_code, airing_idx, overlap_sec) triples for one chunk of views."""
    start = pd.to_datetime(chunk["start_time"], errors="coerce")
    stop = pd.to_datetime(chunk["stop_time"], errors="coerce")
    start_ok = start.notna().to_numpy()
    stop_ok = stop.notna().to_numpy()
    vs = start.to_numpy(dtype="datetime64[ns]").astype("int64")
    ve = stop.to_numpy(dtype="datetime64[ns]").astype("int64")

    valid = start_ok & stop_ok & (ve > vs)
    if not valid.any():
        return None

    users = chunk["user_id"].to_numpy()
    codes, uniques = pd.factorize(chunk["vsetv_id"].to_numpy(), sort=False)
    order = np.argsort(codes, kind="stable")
    sorted_codes = codes[order]
    bounds = np.searchsorted(sorted_codes, np.arange(len(uniques) + 1))

    out_user, out_airing, out_ov, out_lo, out_hi = [], [], [], [], []

    for code, channel in enumerate(uniques):
        entry = index.get(channel)
        if entry is None:
            continue
        pos = order[bounds[code]:bounds[code + 1]]
        pos = pos[valid[pos]]
        if len(pos) == 0:
            continue

        s, e, airing_idx, _ = entry
        runmax = np.maximum.accumulate(e)

        lb = np.searchsorted(runmax, vs[pos], side="right")
        hi = np.searchsorted(s, ve[pos], side="left")
        counts = np.clip(hi - lb, 0, None)

        nz = counts > 0
        if not nz.any():
            continue
        grp = np.flatnonzero(nz)
        c = counts[nz]
        v_local = np.repeat(grp, c)
        offs = np.arange(c.sum()) - np.repeat(np.cumsum(c) - c, c)
        p_local = lb[v_local] + offs

        v_ids = pos[v_local]
        lo = np.maximum(vs[v_ids], s[p_local])
        hi = np.minimum(ve[v_ids], e[p_local])
        ov = hi - lo
        keep = (s[p_local] < ve[v_ids]) & (e[p_local] > vs[v_ids]) & (ov > 0)
        if not keep.any():
            continue

        out_user.append(users[v_ids[keep]])
        out_airing.append(airing_idx[p_local[keep]])
        out_ov.append(ov[keep])
        out_lo.append(lo[keep])
        out_hi.append(hi[keep])

    if not out_user:
        return None
    return (np.concatenate(out_user), np.concatenate(out_airing),
            np.concatenate(out_ov), np.concatenate(out_lo), np.concatenate(out_hi))


def main():
    section("LOADING SCHEDULE")
    sched = load_schedule()
    index = build_index(sched)

    section("JOINING VIEW LOG -> AIRINGS")
    user_codes = {}
    user_list, airing_list, ov_list = [], [], []
    n_views = 0

    for chunk in pd.read_csv(VIEW_LOG, chunksize=CHUNK, dtype=str):
        n_views += len(chunk)
        res = match_chunk(chunk, index, user_codes)
        if res is None:
            continue
        users, airing_idx, ov, _lo, _hi = res
        user_list.append(users)
        airing_list.append(airing_idx)
        ov_list.append(ov)

    users = np.concatenate(user_list)
    airing_idx = np.concatenate(airing_list)
    ov_ns = np.concatenate(ov_list)
    print(f"views processed            : {n_views}")
    print(f"raw (view, airing) matches : {len(users)}")

    section("AGGREGATING PER (USER, AIRING)")
    pairs = pd.DataFrame({"user_id": users, "airing_idx": airing_idx,
                          "watched_sec": ov_ns // NS})
    agg = pairs.groupby(["user_id", "airing_idx"], as_index=False, sort=False)["watched_sec"].sum()
    print(f"(user, airing) pairs       : {len(agg)}")

    meta = sched[["airing_idx", "tv_show_id", "week", "start_time", "duration"]]
    out = agg.merge(meta, on="airing_idx", how="left")
    out["screen_time"] = out["watched_sec"] / out["duration"]
    out = out.rename(columns={"start_time": "airing_start"})
    out = out[["user_id", "tv_show_id", "week", "airing_start", "screen_time"]]
    out["user_id"] = out["user_id"].astype(str)
    out["tv_show_id"] = out["tv_show_id"].astype("int64")
    out["week"] = out["week"].astype("int16")
    out["airing_start"] = out["airing_start"].astype("datetime64[ns]")
    out["screen_time"] = out["screen_time"].astype("float64")

    section("RESULTS")
    st = out["screen_time"]
    print(f"rows written               : {len(out)}")
    print(f"distinct users             : {out['user_id'].nunique()}")
    print(f"distinct shows             : {out['tv_show_id'].nunique()}")
    print(f"screen_time min/median/max : {st.min():.4f} / {st.median():.4f} / {st.max():.4f}")
    print(f"screen_time > 1 (double counted): {int((st > 1).sum())} "
          f"({(st > 1).mean():.4%})")
    for thr in (0.5, 0.8, 0.9, 1.0):
        print(f"  screen_time >= {thr}: {int((st >= thr).sum())} "
              f"({(st >= thr).mean():.2%})")

    pq.write_table(pa.Table.from_pandas(out, schema=SCHEMA), OUT_PARQUET)
    print(f"\nwrote {OUT_PARQUET}")


if __name__ == "__main__":
    main()
