"""Data and folds shared by the three comparison models (user-disjoint protocol, Section III-G).

All comparison models use the label-independent cohort: the same 8,690 trajectories of 60 users
as the proposed model (data/processed/trajectories.csv from preprocess.py), no mode-specific caps,
duplicate timestamps removed (first point kept), and the user folds in splits/uid_folds.csv.
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
PROCESSED = REPO / "data" / "processed"
SPLITS = REPO / "splits"
OUTPUTS = REPO / "outputs" / "baselines"
CLASSES = ["walk", "bike", "bus", "car", "train"]
SEEDS = [1042, 2042, 3042]


def load_meta() -> pd.DataFrame:
    meta = pd.read_csv(PROCESSED / "trajectories.csv", dtype={"uid": str}).sort_values("traj_id").reset_index(drop=True)
    meta["uid"] = meta["uid"].str.zfill(3)
    meta["y"] = meta["label"].map({c: i for i, c in enumerate(CLASSES)})
    assert len(meta) == 8690 and meta["uid"].nunique() == 60, "run preprocess.py first"
    return meta


def load_points(meta: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """traj_id -> points (datetime, lat, lon), sorted, first point of duplicated timestamps kept."""
    pts = pd.read_parquet(PROCESSED / "points.parquet", columns=["traj_id", "datetime", "lat", "lon", "source_point_index"])
    pts = pts[pts.traj_id.isin(set(meta.traj_id))]
    pts = pts.sort_values(["traj_id", "datetime", "source_point_index"], kind="mergesort")
    pts = pts.drop_duplicates(["traj_id", "datetime"], keep="first")
    return {k: g[["datetime", "lat", "lon"]].reset_index(drop=True) for k, g in pts.groupby("traj_id", sort=False)}


def load_folds(meta: pd.DataFrame) -> list[dict]:
    users = set(meta.uid)
    folds = []
    for r in pd.read_csv(SPLITS / "uid_folds.csv", dtype=str).itertuples(index=False):
        te = sorted(u.strip().zfill(3) for u in r.test_uid_list.split(","))
        va = sorted(u.strip().zfill(3) for u in r.validation_uid_list.split(","))
        tr = sorted(users - set(te) - set(va))
        assert (len(tr), len(va), len(te)) == (36, 9, 15)
        folds.append(dict(fold=int(r.fold), train=tr, validation=va, test=te))
    return folds


def set_seed(seed: int):
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def save_predictions(out: Path, meta: pd.DataFrame, test_rows: np.ndarray, proba: np.ndarray, fold: int, seed: int):
    """Append trajectory probabilities in the format read by train_eval.py --evaluate."""
    df = pd.DataFrame(proba, columns=[f"p_{c}" for c in CLASSES])
    df.insert(0, "traj_id", meta.traj_id.to_numpy()[test_rows])
    df.insert(1, "uid", meta.uid.to_numpy()[test_rows])
    df.insert(2, "true_label", meta.label.to_numpy()[test_rows])
    df.insert(3, "fold", fold); df.insert(4, "seed", seed)
    path = out / "predictions_by_seed.csv"
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


def done_pairs(out: Path) -> set:
    path = out / "predictions_by_seed.csv"
    if not path.exists():
        return set()
    d = pd.read_csv(path, usecols=["fold", "seed"]).drop_duplicates()
    return set(zip(d.fold, d.seed))
