"""Dabiri and Heaslip [4] CNN under the user-disjoint 4-fold protocol.

Re-implements the authors' public Keras code (Instance_Creation.py, Keras_Data_Creation.py,
CNN-Keras.py; github.com/sinadabiri/Transport-Mode-GPS-CNN, commit 4a0fede) in PyTorch.

Usage (from the repository root):
  python baselines/cnn/dabiri_cnn.py build     # instance cache -> outputs/baselines/cnn/instances.npz
  python baselines/cnn/dabiri_cnn.py train     # 4 folds x 3 seeds -> predictions_by_seed.csv
  python train_eval.py --evaluate outputs/baselines/cnn
"""
from __future__ import annotations

import json
import math
import sys
import time
from math import factorial
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as C  # noqa: E402

OUT = C.OUTPUTS / "cnn"
CACHE = OUT / "instances.npz"
THRESHOLD, MIN_TRIP = 200, 20 * 60        # 200 points per instance; new instance after a gap > 20 min
EPOCHS, BATCH, LR = 100, 64, 1e-3         # original: 62 epochs; here the common maximum of 100


# ------------------------------------------------------------------ instance creation (official rules)
def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    """Official smoothing function (np.mat / np.int replaced for current NumPy)."""
    half = (window_size - 1) // 2
    b = np.array([[k ** i for i in range(order + 1)] for k in range(-half, half + 1)], dtype=float)
    m = np.linalg.pinv(b)[deriv] * rate ** deriv * factorial(deriv)
    first = y[0] - np.abs(y[1:half + 1][::-1] - y[0])
    last = y[-1] + np.abs(y[-half - 1:-1][::-1] - y[-1])
    return np.convolve(m[::-1], np.concatenate((first, y, last)), mode="valid")


def official_bearing(lat0, lon0, lat1, lon1):
    """Bearing expression exactly as written in Instance_Creation.py."""
    y = math.sin(math.radians(lon1) - math.radians(lon0)) * math.radians(math.cos(lat1))
    x = math.radians(math.cos(lat0)) * math.radians(math.sin(lat1)) - \
        math.radians(math.sin(lat0)) * math.radians(math.cos(lat1)) * math.radians(math.cos(lon1) - math.radians(lon0))
    return (math.atan2(y, x) * 180. / math.pi + 360) % 360


def distance_m(a, b):
    """Geodesic distance (geopy replacement of the removed vincenty); haversine for the few
    GeoLife points whose latitude is outside [-90, 90]."""
    from geopy.distance import geodesic
    if abs(a[0]) <= 90 and abs(b[0]) <= 90:
        return geodesic(a, b).meters
    p1, p2 = math.radians(a[0]), math.radians(b[0])
    h = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(b[1] - a[1]) / 2) ** 2
    return 2 * 6371000.0 * math.asin(min(1.0, math.sqrt(max(h, 0.0))))


def instances(df: pd.DataFrame) -> list[np.ndarray]:
    t = df["datetime"].values.astype("datetime64[ms]").astype(np.int64) / 1000.0
    lat, lon, n = df["lat"].to_numpy(), df["lon"].to_numpy(), len(df)
    delta = np.append(np.diff(t), 3.0); delta[delta == 0] = 0.1
    groups, i = [], 0
    while i <= n - 1:
        idx = []
        while i <= n - 1 and len(idx) < THRESHOLD:
            idx.append(i); i += 1
            if delta[i - 1] > MIN_TRIP:
                break
        if len(idx) >= 10:                     # instances with < 10 points are dropped
            groups.append(np.asarray(idx))
    out = []
    for idx in groups:
        la, lo, tt = lat[idx], lon[idx], t[idx]
        rd, dt, sp, br = [], [], [], []
        for k in range(len(idx) - 1):
            d = distance_m((la[k], lo[k]), (la[k + 1], lo[k + 1]))
            rd.append(d); dt.append((tt[k + 1] - tt[k]) + 1)          # +1 s as in the official code
            sp.append(d / dt[-1]); br.append(official_bearing(la[k], lo[k], la[k + 1], lo[k + 1]))
        if sum(rd) == 0:
            continue
        sp.append(sp[-1]); br.append(br[-1])
        acc = [(sp[k + 1] - sp[k]) / dt[k] for k in range(len(sp) - 1)]; acc.append(acc[-1])
        jerk = [(acc[k + 1] - acc[k]) / dt[k] for k in range(len(acc) - 1)]; jerk.append(jerk[-1])
        brate = [abs(br[k + 1] - br[k]) for k in range(len(br) - 1)]; brate.append(brate[-1])
        x = np.zeros((4, THRESHOLD), np.float32)                         # zero padding, no scaling
        for c, v in enumerate(savitzky_golay(np.asarray(v, float), 9, 3) for v in (sp, acc, jerk, brate)):
            x[c, :len(v)] = v
        out.append(x)
    return out


def build(meta):
    points = C.load_points(meta)
    X, traj, t0 = [], [], time.time()
    for j, tid in enumerate(meta.traj_id):
        for x in instances(points[tid]):
            X.append(x); traj.append(j)
        if j % 1000 == 0:
            print(f"[instances] {j}/{len(meta)} {time.time() - t0:.0f}s", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, X=np.stack(X), traj=np.asarray(traj))
    print(f"{len(X)} instances from {len(set(traj))} trajectories")


# ------------------------------------------------------------------ model and training
class DabiriCNN(nn.Module):
    """Keras Conv2D(kernel (1, 3), 'same') on a 1 x 200 x 4 input equals Conv1d(kernel 3) over 200 points."""

    def __init__(self, length=THRESHOLD):
        super().__init__()
        conv = lambda i, o: nn.Conv1d(i, o, 3, padding=1)  # noqa: E731
        self.f = nn.Sequential(conv(4, 32), nn.ReLU(), conv(32, 32), nn.ReLU(), nn.MaxPool1d(2),
                               conv(32, 64), nn.ReLU(), conv(64, 64), nn.ReLU(), nn.MaxPool1d(2),
                               conv(64, 128), nn.ReLU(), conv(128, 128), nn.ReLU(), nn.MaxPool1d(2),
                               nn.Dropout(0.5), nn.Flatten())
        flat = 128 * (length // 8)
        self.c = nn.Sequential(nn.Linear(flat, flat // 4), nn.ReLU(), nn.Dropout(0.5), nn.Linear(flat // 4, 5))
        for m in self.modules():                                         # Keras default initialization
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.c(self.f(x))


def predict(model, X):
    model.eval()
    with torch.no_grad():
        return np.concatenate([torch.softmax(model(X[b:b + 4096]), 1).cpu().numpy() for b in range(0, len(X), 4096)])


def train(meta):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    z = np.load(CACHE)
    DX, DTRAJ = z["X"].astype(np.float32), z["traj"]
    users, Y = meta.uid.to_numpy(), meta.y.to_numpy()
    done = C.done_pairs(OUT)
    for seed in C.SEEDS:
        for f in C.load_folds(meta):
            if (f["fold"], seed) in done:
                continue
            tr, va, te = (np.isin(users[DTRAJ], f[k]) for k in ("train", "validation", "test"))
            C.set_seed(seed)
            model = DabiriCNN().to(dev)
            opt = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-8)
            Xtr, ytr = torch.from_numpy(DX[tr]).to(dev), torch.from_numpy(Y[DTRAJ[tr]]).to(dev)
            Xva, yva = torch.from_numpy(DX[va]).to(dev), Y[DTRAJ[va]]
            g = torch.Generator().manual_seed(seed)
            best, best_ep, state, t0 = -1, 0, None, time.time()
            for ep in range(1, EPOCHS + 1):
                model.train(); perm = torch.randperm(len(Xtr), generator=g).to(dev)
                for b in range(0, len(perm), BATCH):
                    ix = perm[b:b + BATCH]
                    opt.zero_grad(); nn.functional.cross_entropy(model(Xtr[ix]), ytr[ix]).backward(); opt.step()
                acc = (predict(model, Xva).argmax(1) == yva).mean()   # checkpoint: validation-user accuracy
                if acc > best:
                    best, best_ep, state = acc, ep, {k: v.detach().clone() for k, v in model.state_dict().items()}
            model.load_state_dict(state)
            test_rows = np.flatnonzero(np.isin(users, f["test"]))
            p_inst = predict(model, torch.from_numpy(DX[te]).to(dev))
            P = pd.DataFrame(p_inst).assign(t=DTRAJ[te]).groupby("t").mean().reindex(test_rows).to_numpy()
            C.save_predictions(OUT, meta, test_rows, P, f["fold"], seed)   # trajectory = mean of instances
            json.dump({"fold": f["fold"], "seed": seed, "best_epoch": best_ep, "val_instance_acc": float(best)},
                      open(OUT / f"run_f{f['fold']}_s{seed}.json", "w"))
            print(f"fold {f['fold']} seed {seed}: best epoch {best_ep}, {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    meta = C.load_meta()
    {"build": build, "train": train}[sys.argv[1]](meta)
