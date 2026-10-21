"""Join the view log to the TV schedule by channel + airing-interval overlap.

A view (user_id, vsetv_id, [start_time, stop_time]) is matched to a schedule row
when the channel agrees (vsetv_id == channel_id) and the two intervals overlap:
the programme is on air during [start_time, start_time + duration].

Only labelled programmes (tv_show_id != 0) are kept. For every match we record
the overlap in seconds and as a fraction of the programme's full duration.

Streaming design:
  * the labelled schedule (~237k rows) is loaded once into per-channel numpy
    arrays (sorted by start_time);
  * the view log is processed in chunks, matched with binary search per channel;
  * matches are appended to a parquet writer in batches, so nothing large is
    ever held in memory.

Run: python src/join.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

VIEW_LOG = DATA / "dataset11-30.csv"
OUT_PARQUET = DATA / "interactions_11_30.parquet"

# Weeks 11-20 and 21-30 cover the view-log window (2020-03-09 .. 2020-07-26).
# The weeks 31-42 schedule is not used: no view starts after 2020-07-26 23:58.
SCHEDULE_FILES = [
    DATA / "export_arh_11-20-final.csv",
    DATA / "export_arh_21-30-final.csv",
]

CHUNK = 500_000
FLUSH_EVERY = 1_000_000
NS = 1_000_000_000

SCHEMA = pa.schema([
    ("user_id", pa.string()),
    ("tv_show_id", pa.int64()),
    ("week", pa.int16()),
    ("overlap_sec", pa.int64()),
    ("overlap_frac", pa.float64()),
])


def section(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def build_schedule_index():
    """Per-channel arrays of labelled airings, sorted by start time."""
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

    # keep only labelled programmes with a usable airing interval
    ids = sched["tv_show_id"].str.strip()
    sched = sched[(ids != "0") & (ids != "") & sched["start_time"].notna()
                  & sched["duration"].notna() & (sched["duration"] > 0)].copy()
    sched["tv_show_id"] = ids[sched.index].astype("int64")

    print(f"schedule rows read              : {n_all}")
    print(f"labelled rows kept (id != 0)    : {len(sched)}")

    index = {}
    for channel, grp in sched.groupby("channel_id", sort=False):
        grp = grp.sort_values("start_time")
        starts = grp["start_time"].to_numpy(dtype="datetime64[ns]").astype("int64")
        durations = grp["duration"].to_numpy(dtype="int64")
        index[channel] = (
            starts,
            starts + durations * NS,
            grp["tv_show_id"].to_numpy(dtype="int64"),
            durations,
        )
    print(f"channels with labelled airings  : {len(index)}")
    return index


def match_chunk(chunk, index):
    """Return (view_row, show_id, overlap_ns, programme_duration) for one chunk."""
    start = pd.to_datetime(chunk["start_time"], errors="coerce")
    stop = pd.to_datetime(chunk["stop_time"], errors="coerce")

    start_ok = start.notna().to_numpy()
    stop_ok = stop.notna().to_numpy()
    vs = start.to_numpy(dtype="datetime64[ns]").astype("int64")
    ve = stop.to_numpy(dtype="datetime64[ns]").astype("int64")

    # NaT is encoded as int64 min, so validity must come from the notna masks
    valid = start_ok & stop_ok & (ve > vs)
    if not valid.any():
        return None

    codes, uniques = pd.factorize(chunk["vsetv_id"].to_numpy(), sort=False)
    order = np.argsort(codes, kind="stable")
    sorted_codes = codes[order]
    bounds = np.searchsorted(sorted_codes, np.arange(len(uniques) + 1))

    out_rows, out_ids, out_ov, out_dur = [], [], [], []

    for code, channel in enumerate(uniques):
        entry = index.get(channel)
        if entry is None:
            continue
        pos = order[bounds[code]:bounds[code + 1]]
        pos = pos[valid[pos]]
        if len(pos) == 0:
            continue

        s, e, show_ids, durations = entry
        runmax = np.maximum.accumulate(e)

        # candidate window per view: programmes starting before the view ends and
        # not already finished when the view starts
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
        p_ids = p_local
        ov_start = np.maximum(vs[v_ids], s[p_ids])
        ov_end = np.minimum(ve[v_ids], e[p_ids])
        ov = ov_end - ov_start

        keep = (s[p_ids] < ve[v_ids]) & (e[p_ids] > vs[v_ids]) & (ov > 0)
        if not keep.any():
            continue

        out_rows.append(v_ids[keep])
        out_ids.append(show_ids[p_ids[keep]])
        out_ov.append(ov[keep])
        out_dur.append(durations[p_ids[keep]])

    if not out_rows:
        return None
    return (np.concatenate(out_rows), np.concatenate(out_ids),
            np.concatenate(out_ov), np.concatenate(out_dur))


def main():
    section("LOADING SCHEDULE")
    index = build_schedule_index()

    section("JOINING VIEW LOG -> SCHEDULE")
    writer = pq.ParquetWriter(OUT_PARQUET, SCHEMA, compression="snappy")

    n_views = 0
    n_views_matched = 0
    n_invalid = 0
    n_matches = 0
    all_ov_sec = []
    all_ov_frac = []
    all_user = []
    all_show = []
    batch = []

    for chunk in pd.read_csv(VIEW_LOG, chunksize=CHUNK, dtype=str):
        n_views += len(chunk)
        res = match_chunk(chunk, index)
        if res is None:
            continue

        view_rows, show_ids, ov_ns, durations = res
        n_matches += len(view_rows)
        n_views_matched += len(np.unique(view_rows))

        ov_sec = ov_ns // NS
        ov_frac = ov_sec / durations

        week = pd.to_datetime(chunk["start_time"], errors="coerce").dt.isocalendar().week
        week = week.to_numpy(dtype="float64")[view_rows]
        week = np.nan_to_num(week, nan=0).astype("int16")

        frame = pd.DataFrame({
            "user_id": chunk["user_id"].to_numpy()[view_rows],
            "tv_show_id": show_ids,
            "week": week,
            "overlap_sec": ov_sec,
            "overlap_frac": ov_frac,
        })
        all_ov_sec.append(ov_sec)
        all_ov_frac.append(ov_frac)
        all_user.append(frame["user_id"].to_numpy())
        all_show.append(show_ids)
        batch.append(frame)

        if sum(len(b) for b in batch) >= FLUSH_EVERY:
            writer.write_table(pa.Table.from_pandas(pd.concat(batch, ignore_index=True),
                                                    schema=SCHEMA))
            batch = []

    if batch:
        writer.write_table(pa.Table.from_pandas(pd.concat(batch, ignore_index=True),
                                                schema=SCHEMA))
    writer.close()

    section("RESULTS")
    ov_sec = np.concatenate(all_ov_sec)
    ov_frac = np.concatenate(all_ov_frac)
    users = np.concatenate(all_user)
    shows = np.concatenate(all_show)

    print(f"views processed                 : {n_views}")
    print(f"views with >=1 labelled match   : {n_views_matched} "
          f"({n_views_matched / n_views:.2%})")
    print(f"matches written                 : {n_matches}  -> {OUT_PARQUET}")
    print(f"distinct users in matches       : {len(np.unique(users))}")
    print(f"distinct tv_show_id in matches  : {len(np.unique(shows))}")
    print(f"mean matches per matched view   : {n_matches / max(n_views_matched, 1):.2f}")

    def dist(label, values):
        q = np.percentile(values, [10, 25, 50, 75, 90])
        print(f"\n{label} distribution (n={len(values)}):")
        for name, val in zip(["p10", "p25", "median", "p75", "p90"], q):
            print(f"  {name:<7}: {val:12.2f}")
        print(f"  min    : {values.min():12.2f}")
        print(f"  max    : {values.max():12.2f}")
        print(f"  mean   : {values.mean():12.2f}")

    dist("overlap_sec", ov_sec.astype("float64"))
    dist("overlap_frac", ov_frac)


if __name__ == "__main__":
    main()
