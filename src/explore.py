"""Exploratory summary of the files in data/.

Streams the large CSVs in chunks so we never hold a full file in memory.
Run: python src/explore.py
"""

from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"
CHUNK = 200_000

VIEW_LOG = DATA / "dataset11-30.csv"
SUBMISSION = DATA / "submission.csv"
SCHEDULE_FILES = {
    "weeks 11-20": DATA / "export_arh_11-20-final.csv",
    "weeks 21-30": DATA / "export_arh_21-30-final.csv",
    "weeks 31-42": DATA / "export_arh_31-42-final.csv",
}


def section(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def count_rows(path):
    """Row count (excluding header) by streaming the file in chunks."""
    total = 0
    for chunk in pd.read_csv(path, chunksize=CHUNK, dtype=str, keep_default_na=False):
        total += len(chunk)
    return total


def explore_view_log():
    section("VIEW LOG: data/dataset11-30.csv")
    header = pd.read_csv(VIEW_LOG, nrows=0)
    print("columns:", list(header.columns))
    print("dtypes:", dict(header.dtypes.astype(str)))

    n = count_rows(VIEW_LOG)
    print("rows:", n)

    # Date range + duration stats, streamed.
    start_min = start_max = stop_min = stop_max = None
    dur_min = dur_max = None
    durations = []
    null_counts = dict.fromkeys(header.columns, 0)
    users = set()

    for chunk in pd.read_csv(VIEW_LOG, chunksize=CHUNK):
        start = pd.to_datetime(chunk["start_time"], errors="coerce")
        stop = pd.to_datetime(chunk["stop_time"], errors="coerce")
        dur = pd.to_numeric(chunk["duraton"], errors="coerce")

        start_min = start.min() if start_min is None else min(start_min, start.min())
        start_max = start.max() if start_max is None else max(start_max, start.max())
        stop_min = stop.min() if stop_min is None else min(stop_min, stop.min())
        stop_max = stop.max() if stop_max is None else max(stop_max, stop.max())

        dur_min = dur.min() if dur_min is None else min(dur_min, dur.min())
        dur_max = dur.max() if dur_max is None else max(dur_max, dur.max())
        durations.append(dur.dropna().to_numpy())

        for col in header.columns:
            null_counts[col] += int(chunk[col].isna().sum())
        users.update(chunk["user_id"].unique().tolist())

    durations = pd.concat([pd.Series(d) for d in durations])

    print("\n-- date range (start_time):", start_min, "->", start_max)
    print("-- date range (stop_time) :", stop_min, "->", stop_max)
    print("-- distinct users in view log:", len(users))
    print("-- null counts:", null_counts)

    print("\n-- duraton (seconds) distribution --")
    print(f"min    : {durations.min():.0f}")
    print(f"median : {durations.median():.0f}")
    print(f"p90    : {durations.quantile(0.90):.0f}")
    print(f"max    : {durations.max():.0f}")
    print(f"mean   : {durations.mean():.1f}")
    print(f"zeros  : {(durations == 0).sum()}")
    print(f"negatives: {(durations < 0).sum()}")
    print("quantiles:", {f"p{p}": round(durations.quantile(p / 100), 1)
                         for p in (1, 5, 10, 25, 50, 75, 90, 95, 99)})
    return durations


def explore_schedule(label, path):
    section(f"SCHEDULE: {path.name}  ({label})")
    header = pd.read_csv(path, nrows=0)
    print("columns:", list(header.columns))
    print("dtypes:", dict(header.dtypes.astype(str)))

    n = count_rows(path)
    print("rows:", n)

    zero_rows = 0
    blank_id_rows = 0
    total_rows = 0
    start_min = start_max = None
    id_counter = {}

    for chunk in pd.read_csv(path, chunksize=CHUNK, dtype={"tv_show_id": str},
                             keep_default_na=False):
        total_rows += len(chunk)
        ids = chunk["tv_show_id"].astype(str).str.strip()
        zero_rows += int((ids == "0").sum())
        blank_id_rows += int((ids == "").sum())

        starts = pd.to_datetime(chunk["start_time"], format="%d.%m.%Y %H:%M:%S",
                                errors="coerce")
        start_min = starts.min() if start_min is None else min(start_min, starts.min())
        start_max = starts.max() if start_max is None else max(start_max, starts.max())

        for value, count in ids.value_counts().items():
            id_counter[value] = id_counter.get(value, 0) + int(count)

    print("start_time range:", start_min, "->", start_max)
    print(f"rows with tv_show_id == 0 : {zero_rows} "
          f"({zero_rows / total_rows:.2%} of {total_rows})")
    print(f"rows with blank tv_show_id: {blank_id_rows}")
    print(f"distinct tv_show_id values: {len(id_counter)}")
    print("top 5 tv_show_id by row count:",
          sorted(id_counter.items(), key=lambda kv: -kv[1])[:5])
    return zero_rows


def explore_submission():
    section("SUBMISSION: data/submission.csv")
    header = pd.read_csv(SUBMISSION, nrows=0)
    print("columns:", list(header.columns))
    print("dtypes:", dict(header.dtypes.astype(str)))

    sub = pd.read_csv(SUBMISSION, dtype=str)
    print("rows:", len(sub))
    print("distinct user_id:", sub["user_id"].nunique())
    print("duplicate user_id rows:", int(sub["user_id"].duplicated().sum()))
    print("null user_id:", int(sub["user_id"].isna().sum()))

    # tv_show_id column holds space-separated lists of 5 show ids.
    lists = sub["tv_show_id"].fillna("").astype(str).str.split()
    lengths = lists.str.len()
    print("predictions per user -> min:", lengths.min(), "median:",
          lengths.median(), "max:", lengths.max())
    print("users whose list is all zeros:",
          int(lists.apply(lambda xs: bool(xs) and all(v == "0" for v in xs)).sum()))
    print("sample rows:")
    print(sub.head(5).to_string(index=False))


if __name__ == "__main__":
    explore_view_log()
    for label, path in SCHEDULE_FILES.items():
        explore_schedule(label, path)
    explore_submission()
