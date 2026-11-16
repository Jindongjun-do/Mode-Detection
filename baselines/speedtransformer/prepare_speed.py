"""Speed sequences (km/h) of the 8,690 label-independent trajectories for SpeedTransformer.

Same points as the proposed model (duplicate timestamps removed, first point kept, no caps).
Speed = haversine distance / time difference between consecutive points, in km/h as in the
public SpeedTransformer preprocessing; the first point of each trajectory has no speed and is dropped.

Output: outputs/baselines/speedtransformer/speed_sequences_8690.csv (time_diff, speed, distance, traj_id, label, uid)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as C  # noqa: E402

R = 6_371_000.0


def haversine_m(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2.0 * R * np.arctan2(np.sqrt(a), np.sqrt(np.maximum(0.0, 1.0 - a)))


def main():
    meta = C.load_meta()
    points = C.load_points(meta)
    rows = []
    for r in meta.itertuples(index=False):
        p = points[r.traj_id]
        dt = p["datetime"].diff().dt.total_seconds().to_numpy()[1:]
        assert (dt > 0).all()
        lat, lon = p["lat"].to_numpy(float), p["lon"].to_numpy(float)
        d = haversine_m(lat[:-1], lon[:-1], lat[1:], lon[1:])
        rows.append(pd.DataFrame({"time_diff": dt, "speed": d / dt * 3.6, "distance": d,
                                  "traj_id": r.traj_id, "label": r.label, "uid": r.uid}))
    out = pd.concat(rows, ignore_index=True)
    assert np.isfinite(out[["time_diff", "speed", "distance"]].to_numpy()).all()
    assert out.traj_id.nunique() == 8690 and out.uid.nunique() == 60
    path = C.OUTPUTS / "speedtransformer" / "speed_sequences_8690.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    print(f"{len(out):,} speed values, {out.traj_id.nunique()} trajectories -> {path}")


if __name__ == "__main__":
    main()
