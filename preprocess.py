"""Preprocessing and graph construction: raw GeoLife -> one Stop-Move graph per trajectory.

Step 1  Labeled trajectories (8,690 candidates from 60 users)
        - GeoLife modes -> five classes (taxi -> car, subway -> train).
        - Same-label intervals of a user that overlap or touch are merged; time ranges claimed by
          two different labels are removed; a point belongs to an interval if start <= t < end.
        - An interval is kept if it has >= 20 points, >= 60 s and >= 50 m after duplicate
          timestamps and jumps faster than 80 m/s are removed.
        -> data/processed/trajectories.csv, points.parquet

Step 2  Graphs for the two evaluation settings
        random  label-dependent cohort: mode-specific speed/acceleration caps of Dabiri and Heaslip
                applied iteratively, validity criteria checked again -> 8,640 trajectories;
                segmentation "merge_short"
        uid     label-independent cohort: no caps, all 8,690 trajectories;
                segmentation "merge_short_split_long"
        -> data/processed/graphs_<setting>.pt (list of torch_geometric Data) and .csv (metadata)

Usage:  python preprocess.py [--geolife "PATH/Geolife Trajectories 1.3/Data"] [--overwrite]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

# ----------------------------------------------------------------------------- constants
REPO = Path(__file__).resolve().parent
RAW_GEOLIFE = REPO / "data" / "raw" / "Geolife Trajectories 1.3" / "Data"   # holds 000/, 001/, ..., 181/
PROCESSED = REPO / "data" / "processed"

MODE_MAP = {"walk": "walk", "bike": "bike", "bus": "bus", "car": "car", "driving": "car",
            "taxi": "car", "train": "train", "subway": "train"}
CLASSES = ["walk", "bike", "bus", "car", "train"]
C2I = {c: i for i, c in enumerate(CLASSES)}

MIN_POINTS, MIN_DURATION_SEC, MIN_DISTANCE_M = 20, 60.0, 50.0      # validity criteria
MAX_DT_SEC = 600.0              # pairs further apart: distance, speed and derived kinematics set to 0
CANDIDATE_MAX_SPEED_MPS = 80.0  # jump removal for candidate selection (up to 5 passes)
DABIRI_LIMITS = {               # (max speed m/s, max |acceleration| m/s^2), label-dependent cohort
    "walk": (7.0, 3.0), "bike": (12.0, 3.0), "bus": (34.0, 2.0), "car": (50.0, 10.0), "train": (34.0, 3.0),
}
STOP_SPEED_MPS = 0.5            # 1.8 km/h
MIN_SEG_POINTS, MIN_SEG_DURATION_SEC = 3, 5.0        # merge_short
MAX_SEG_DURATION_SEC, MAX_SEG_DISTANCE_M = 180.0, 500.0  # split_long

SETTINGS = {  # setting -> (apply mode-specific caps, segmentation rule)
    "random": (True, "merge_short"),
    "uid": (False, "merge_short_split_long"),
}

NODE_FEATURES = [
    "duration_sec", "distance_m", "displacement_m", "straightness",
    "mean_speed", "p50_speed", "p90_speed", "max_speed", "speed_std",
    "moving_mean_speed", "moving_p90_speed", "moving_max_speed",
    "stop_ratio", "stop_duration_sum", "stop_duration_max", "is_stop_segment", "num_stop_points",
    "acc_mean", "acc_std", "acc_abs_p95", "jerk_mean", "jerk_std",
    "bearing_rate_mean", "bearing_rate_std", "segment_point_count",
]
BINARY_FEATURES = ["is_stop_segment"]   # not standardized


# ----------------------------------------------------------------------------- GeoLife reading
def read_labels(path: Path, uid: str) -> pd.DataFrame:
    lab = pd.read_csv(path, sep="\t")
    lab.columns = [str(c).strip() for c in lab.columns]
    lab = lab.rename(columns={"Start Time": "start_time", "End Time": "end_time", "Transportation Mode": "raw_mode"})
    lab["start_time"] = pd.to_datetime(lab["start_time"], errors="coerce")
    lab["end_time"] = pd.to_datetime(lab["end_time"], errors="coerce")
    lab["label"] = lab["raw_mode"].astype(str).str.strip().str.lower().map(MODE_MAP)
    lab["uid"] = uid
    ok = lab["label"].isin(CLASSES) & (lab["end_time"] > lab["start_time"])
    return lab.loc[ok, ["uid", "label", "start_time", "end_time"]].reset_index(drop=True)


def _merge_same_label(lab: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (uid, label), g in lab.groupby(["uid", "label"]):
        g = g.sort_values(["start_time", "end_time"])
        cur_s = cur_e = None
        for s, e in zip(g["start_time"], g["end_time"]):
            if cur_s is None:
                cur_s, cur_e = s, e
            elif s <= cur_e:
                cur_e = max(cur_e, e)
            else:
                rows.append((uid, label, cur_s, cur_e)); cur_s, cur_e = s, e
        rows.append((uid, label, cur_s, cur_e))
    return pd.DataFrame(rows, columns=["uid", "label", "start_time", "end_time"])


def canonical_intervals(lab: pd.DataFrame) -> pd.DataFrame:
    """Non-overlapping single-label intervals with cross-label ambiguity removed."""
    merged = _merge_same_label(lab)
    out, next_id = [], 0
    for uid, g in merged.groupby("uid"):
        cuts = sorted(set(g["start_time"]).union(g["end_time"]))
        pieces = []
        for a, b in zip(cuts[:-1], cuts[1:]):
            active = g[(g["start_time"] < b) & (g["end_time"] > a)]["label"].unique()
            if len(active) == 1:
                pieces.append([active[0], a, b])
        if not pieces:
            continue
        cur = pieces[0]
        for p in pieces[1:]:
            if p[0] == cur[0] and p[1] == cur[2]:
                cur[2] = p[2]
            else:
                out.append((f"{uid}_canon_{next_id:07d}", uid, *cur)); next_id += 1; cur = p
        out.append((f"{uid}_canon_{next_id:07d}", uid, *cur)); next_id += 1
    return pd.DataFrame(out, columns=["traj_id", "uid", "label", "start_time", "end_time"])


def read_plt(path: Path) -> pd.DataFrame:
    cols = ["lat", "lon", "zero", "altitude", "days", "date", "time"]
    df = pd.read_csv(path, skiprows=6, names=cols, usecols=cols, low_memory=False)
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df["datetime"] = pd.to_datetime(df["date"].astype(str) + " " + df["time"].astype(str), errors="coerce")
    df = df.dropna(subset=["lat", "lon", "datetime"])
    return df[["datetime", "lat", "lon"]]


def read_user_points(user_dir: Path) -> pd.DataFrame:
    """All .plt points of one user, sorted by time, exact (time, lat, lon) duplicates removed.
    source_point_index is the position after this sort and breaks timestamp ties later."""
    frames = [read_plt(p) for p in sorted((user_dir / "Trajectory").glob("*.plt"))]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame(columns=["datetime", "lat", "lon", "source_point_index"])
    pts = pd.concat(frames, ignore_index=True)
    pts = pts.sort_values("datetime").drop_duplicates(subset=["datetime", "lat", "lon"]).reset_index(drop=True)
    pts["source_point_index"] = np.arange(len(pts), dtype=np.int64)
    return pts


def assign_points(intervals: pd.DataFrame, root: Path) -> pd.DataFrame:
    """Points of every interval (half-open window), one row per point."""
    parts = []
    for uid, ints in intervals.groupby("uid"):
        pts = read_user_points(root / uid)
        t = pts["datetime"]
        for r in ints.itertuples(index=False):
            sel = pts.loc[(t >= r.start_time) & (t < r.end_time)].copy()
            if sel.empty:
                continue
            sel["traj_id"], sel["uid"], sel["label"] = r.traj_id, uid, r.label
            parts.append(sel)
    out = pd.concat(parts, ignore_index=True)
    return out.sort_values(["uid", "datetime", "source_point_index"]).reset_index(drop=True)


# ----------------------------------------------------------------------------- kinematics
R_EARTH_M = 6371000.0


def _f(x):
    return np.nan_to_num(np.asarray(x, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)


def haversine_m(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = (np.radians(_f(v)) for v in (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return _f(R_EARTH_M * 2.0 * np.arctan2(np.sqrt(a), np.sqrt(np.maximum(1.0 - a, 0.0))))


def bearing_deg(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = (np.radians(_f(v)) for v in (lat1, lon1, lat2, lon2))
    dlon = lon2 - lon1
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return _f((np.degrees(np.arctan2(y, x)) + 360.0) % 360.0)


def pair_motion(df: pd.DataFrame):
    """Interval, distance and speed between each point and the previous one.
    Pairs with dt <= 0 or dt > MAX_DT_SEC get zero interval, distance and speed."""
    n = len(df)
    lat, lon = df["lat"].to_numpy(np.float64), df["lon"].to_numpy(np.float64)
    raw_dt = df["datetime"].diff().dt.total_seconds().fillna(0.0).to_numpy(np.float64)
    valid = (raw_dt > 0) & (raw_dt <= MAX_DT_SEC)
    dist = np.zeros(n)
    if n > 1:
        dist[1:] = haversine_m(lat[:-1], lon[:-1], lat[1:], lon[1:])
    dist[~valid] = 0.0
    dt = np.where(valid, raw_dt, 0.0)
    speed = np.zeros(n)
    np.divide(dist, dt, out=speed, where=dt > 0)
    return dt, dist, speed, valid


def dedup_time(points: pd.DataFrame) -> pd.DataFrame:
    """Sort by time and keep the first point of each duplicated timestamp."""
    return (points.sort_values(["datetime", "source_point_index"], kind="mergesort")
            .drop_duplicates(subset=["datetime"]).reset_index(drop=True))


def remove_candidate_jumps(df: pd.DataFrame, max_iter: int = 5) -> pd.DataFrame:
    """Class-agnostic removal of > 80 m/s jumps, used only for candidate selection."""
    cur = df.sort_values("datetime").drop_duplicates(subset=["datetime"]).reset_index(drop=True)
    for _ in range(max_iter):
        if len(cur) < 2:
            break
        _, _, speed, _ = pair_motion(cur)
        bad = speed > CANDIDATE_MAX_SPEED_MPS
        if not bad.any():
            break
        cur = cur.loc[~bad].reset_index(drop=True)
    return cur


def dabiri_filter(df: pd.DataFrame, label: str, max_iter: int = 10000) -> tuple[pd.DataFrame, int]:
    """Iteratively drop the destination point of pairs above the mode-specific caps."""
    vmax, amax = DABIRI_LIMITS[label]
    cur, removed = df.copy(), 0
    for _ in range(max_iter):
        if len(cur) < 2:
            break
        dt, _, speed, _ = pair_motion(cur)
        acc = np.zeros(len(cur))
        np.divide(np.diff(speed, prepend=speed[0]), dt, out=acc, where=dt > 0)
        bad = (speed > vmax) | (np.abs(acc) > amax)
        if not bad.any():
            break
        removed += int(bad.sum())
        cur = cur.loc[~bad].reset_index(drop=True)
    return cur, removed


def point_features(df: pd.DataFrame) -> pd.DataFrame:
    """Speed, acceleration, jerk, bearing rate, stop flag and running stop duration per point."""
    df = df.reset_index(drop=True).copy()
    dt, dist, speed, valid = pair_motion(df)
    acc = np.zeros(len(df)); np.divide(np.diff(speed, prepend=speed[0]), dt, out=acc, where=dt > 0)
    jerk = np.zeros(len(df)); np.divide(np.diff(acc, prepend=acc[0]), dt, out=jerk, where=dt > 0)
    lat, lon = df["lat"].to_numpy(np.float64), df["lon"].to_numpy(np.float64)
    bearing = np.zeros(len(df))
    if len(df) > 1:
        bearing[1:] = bearing_deg(lat[:-1], lon[:-1], lat[1:], lon[1:])
    bearing[dt <= 0] = 0.0
    bdiff = ((bearing - np.roll(bearing, 1) + 180.0) % 360.0) - 180.0
    bdiff[0] = 0.0
    brate = np.zeros(len(df)); np.divide(bdiff, dt, out=brate, where=dt > 0)
    stop = (speed < STOP_SPEED_MPS).astype(np.int64)
    run = np.zeros(len(df)); acc_t = 0.0
    for i, s in enumerate(stop):
        acc_t = acc_t + dt[i] if s else 0.0
        run[i] = acc_t
    f32 = lambda v: np.nan_to_num(np.asarray(v, np.float32), nan=0.0, posinf=0.0, neginf=0.0)  # noqa: E731
    df["delta_t"], df["distance_m"], df["speed_mps"] = f32(dt), f32(dist), f32(speed)
    df["acc_mps2"], df["jerk_mps3"], df["bearing_rate_deg_s"] = f32(acc), f32(jerk), f32(brate)
    df["stop_flag"], df["stop_duration_sec"] = stop, f32(run)
    return df


def is_valid(df: pd.DataFrame) -> bool:
    """>= 20 points, >= 60 s and >= 50 m cumulative distance."""
    if df.empty:
        return False
    dur = (df["datetime"].iloc[-1] - df["datetime"].iloc[0]).total_seconds()
    return len(df) >= MIN_POINTS and dur >= MIN_DURATION_SEC and float(df["distance_m"].sum()) >= MIN_DISTANCE_M


# ----------------------------------------------------------------------------- Stop-Move graph
def segments(df: pd.DataFrame, rule: str) -> list[tuple[int, int]]:
    """Inclusive (start, end) point index ranges of the behavioral segments.

    Consecutive points with the same stop flag form a segment. Then:
    merge_short             a segment with < 3 points or < 5 s is merged into its preceding
                            segment (the first one into the next)
    merge_short_split_long  moving segments are cut every 180 s or 500 m (each piece >= 3 points),
                            then merge_short
    """
    flags = df["stop_flag"].to_numpy(int)
    n = len(flags)
    cut = np.flatnonzero(flags[1:] != flags[:-1]) + 1
    segs = [(int(a), int(b) - 1) for a, b in zip(np.r_[0, cut], np.r_[cut, n])]

    sec = (df["datetime"] - df["datetime"].iloc[0]).dt.total_seconds().to_numpy(np.float64)
    cum = np.r_[0.0, np.cumsum(df["distance_m"].to_numpy(np.float64))]
    dur = lambda a, b: max(sec[b] - sec[a], 0.0) if b > a else 0.0      # noqa: E731
    dist = lambda a, b: max(cum[b + 1] - cum[a], 0.0) if b >= a else 0.0  # noqa: E731

    if rule == "merge_short_split_long":
        split = []
        for a, b in segs:
            if flags[a:b + 1].mean() >= 0.5:          # stop segment: keep
                split.append((a, b)); continue
            s = a
            while s <= b:
                if b - s + 1 <= MIN_SEG_POINTS:
                    split.append((s, b)); break
                e = s
                while e < b:
                    if e - s + 1 >= MIN_SEG_POINTS and (dur(s, e) >= MAX_SEG_DURATION_SEC
                                                        or dist(s, e) >= MAX_SEG_DISTANCE_M):
                        break
                    e += 1
                split.append((s, e)); s = e + 1
        segs = split

    short = lambda a, b: (b - a + 1) < MIN_SEG_POINTS or dur(a, b) < MIN_SEG_DURATION_SEC  # noqa: E731
    merged: list[list[int]] = []
    for a, b in segs:
        if merged and (short(*merged[-1]) or short(a, b)):
            merged[-1][1] = b
        else:
            merged.append([a, b])
    if len(merged) > 1 and short(*merged[-1]):
        last = merged.pop(); merged[-1][1] = last[1]
    return [tuple(m) for m in merged] or [(0, n - 1)]


def _mean(v):
    v = v[np.isfinite(v)]; return float(v.mean()) if v.size else 0.0


def _std(v):
    v = v[np.isfinite(v)]; return float(v.std()) if v.size else 0.0      # population SD


def _pct(v, q):
    v = v[np.isfinite(v)]; return float(np.percentile(v, q)) if v.size else 0.0


def _sum32(v) -> float:
    """float32 sum in blocks of 8,192 values, block sums added in float32."""
    v = np.asarray(v, dtype=np.float32)
    total = np.float32(0.0)
    for i in range(0, len(v), 8192):
        total = np.float32(total + v[i:i + 8192].sum(dtype=np.float32))
    return float(total)


def segment_features(s: pd.DataFrame) -> list[float]:
    """25 node features of one segment, in the order of NODE_FEATURES."""
    dur = max((s["datetime"].iloc[-1] - s["datetime"].iloc[0]).total_seconds(), 0.0)
    dist = max(_sum32(s["distance_m"]), 0.0)
    disp = float(haversine_m(s["lat"].iloc[0], s["lon"].iloc[0], s["lat"].iloc[-1], s["lon"].iloc[-1]))
    v = s["speed_mps"].to_numpy(np.float64)
    stop = s["stop_flag"].to_numpy(np.float64)
    vm = v[stop == 0]
    acc = s["acc_mps2"].to_numpy(np.float64)
    jerk = s["jerk_mps3"].to_numpy(np.float64)
    br = s["bearing_rate_deg_s"].to_numpy(np.float64)
    stop_ratio = _mean(stop)
    feats = [
        dur, dist, disp, disp / dist if dist > 0 else 0.0,
        _mean(v), _pct(v, 50), _pct(v, 90), _pct(v, 100), _std(v),
        _mean(vm), _pct(vm, 90), _pct(vm, 100),
        stop_ratio, _sum32(s.loc[s["stop_flag"] == 1, "delta_t"]), float(s["stop_duration_sec"].max()),
        1.0 if stop_ratio >= 0.5 else 0.0, float(stop.sum()),
        _mean(acc), _std(acc), _pct(np.abs(acc), 95), _mean(jerk), _std(jerk), _mean(br), _std(br),
        float(len(s)),
    ]
    return np.nan_to_num(np.asarray(feats, np.float32), nan=0.0, posinf=0.0, neginf=0.0).tolist()


def build_graph(df: pd.DataFrame, rule: str, y: int) -> Data:
    """One trajectory -> one graph: nodes = segments, bidirectional edges between neighbors.
    Self-loops are added inside the GATv2 layers, not stored here."""
    segs = segments(df, rule)
    x = torch.tensor([segment_features(df.iloc[a:b + 1]) for a, b in segs], dtype=torch.float32)
    k = len(segs)
    pairs = [p for i in range(k - 1) for p in ((i, i + 1), (i + 1, i))]
    edge_index = torch.tensor(pairs, dtype=torch.long).reshape(-1, 2).T.contiguous()
    return Data(x=x, edge_index=edge_index, y=torch.tensor([y]))


# ----------------------------------------------------------------------------- steps
def extract_trajectories(root: Path):
    labels = pd.concat([read_labels(d / "labels.txt", d.name) for d in sorted(root.iterdir())
                        if (d / "labels.txt").exists()], ignore_index=True)
    intervals = canonical_intervals(labels)
    print(f"label users {labels.uid.nunique()}, labeled intervals {len(intervals)}", flush=True)
    points = assign_points(intervals, root)

    keep = []
    for tid, g in points.groupby("traj_id", sort=True):
        g = remove_candidate_jumps(g.sort_values(["datetime", "source_point_index"]))
        _, dist, _, _ = pair_motion(g)
        if is_valid(g.assign(distance_m=dist)):
            keep.append(tid)
    traj = intervals[intervals.traj_id.isin(keep)].sort_values("traj_id").reset_index(drop=True)
    pts = points[points.traj_id.isin(keep)].reset_index(drop=True)
    traj.to_csv(PROCESSED / "trajectories.csv", index=False)
    pts.to_parquet(PROCESSED / "points.parquet", index=False)
    print(f"trajectories {len(traj)} from {traj.uid.nunique()} users; points {len(pts):,}")
    print(traj.label.value_counts().reindex(CLASSES).to_string(), flush=True)


def build_graphs(setting: str):
    caps, rule = SETTINGS[setting]
    traj = pd.read_csv(PROCESSED / "trajectories.csv", dtype={"uid": str})
    pts = pd.read_parquet(PROCESSED / "points.parquet")
    groups = dict(tuple(pts.groupby("traj_id", sort=False)))
    graphs, rows, t0 = [], [], time.time()
    for r in traj.itertuples(index=False):
        g = dedup_time(groups[r.traj_id])
        removed = 0
        if caps:
            g, removed = dabiri_filter(g, r.label)
        g = point_features(g)
        if caps and not is_valid(g):
            continue
        data = build_graph(g, rule, C2I[r.label])
        data.graph_id = len(graphs)
        data.traj_id = r.traj_id
        graphs.append(data)
        rows.append(dict(graph_id=data.graph_id, traj_id=r.traj_id, uid=r.uid, label=r.label,
                         num_points=len(g), points_removed_by_caps=removed, num_nodes=int(data.num_nodes),
                         num_edges=int(data.edge_index.shape[1]),
                         duration_sec=(g.datetime.iloc[-1] - g.datetime.iloc[0]).total_seconds(),
                         distance_m=float(g.distance_m.sum())))
        if len(graphs) % 1000 == 0:
            print(f"  {setting}: {len(graphs)} graphs, {time.time() - t0:.0f} s", flush=True)
    torch.save(graphs, PROCESSED / f"graphs_{setting}.pt")
    meta = pd.DataFrame(rows)
    meta.to_csv(PROCESSED / f"graphs_{setting}.csv", index=False)
    print(f"graphs_{setting}: {len(graphs)} graphs, {meta.uid.nunique()} users, {meta.num_points.sum():,} points, "
          f"{meta.num_nodes.sum():,} nodes ({meta.num_points.sum() / meta.num_nodes.sum():.1f} points per node)",
          flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--geolife", type=Path, default=RAW_GEOLIFE, help="GeoLife 'Data' folder (000 ... 181)")
    ap.add_argument("--overwrite", action="store_true", help="redo steps whose output already exists")
    args = ap.parse_args()
    PROCESSED.mkdir(parents=True, exist_ok=True)

    if args.overwrite or not (PROCESSED / "trajectories.csv").exists():
        if not args.geolife.exists():
            sys.exit(f"GeoLife folder not found: {args.geolife}\nSee README.")
        extract_trajectories(args.geolife)
    for setting in SETTINGS:
        if args.overwrite or not (PROCESSED / f"graphs_{setting}.pt").exists():
            build_graphs(setting)


if __name__ == "__main__":
    main()
