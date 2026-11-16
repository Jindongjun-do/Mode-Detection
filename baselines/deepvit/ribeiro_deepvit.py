"""Ribeiro et al. [6] DeepInsight + DeepViT under the user-disjoint 4-fold protocol.

Imports the authors' public code (github.com/RRibeiro8/transport-mode-identification-memoria,
commit 3df4293) from baselines/deepvit/official/ (clone it there, see README):
features_generator.FeaturesGenerator.extractFeatures (399 features per trajectory),
pyDeepInsight.ImageTransformer + Norm2Scaler, archs.vit.DeepViTModel.

Usage (from the repository root):
  python baselines/deepvit/ribeiro_deepvit.py build   # features -> outputs/baselines/deepvit/features_8690.csv
  python baselines/deepvit/ribeiro_deepvit.py train   # 4 folds x 3 seeds -> predictions_by_seed.csv
  python train_eval.py --evaluate outputs/baselines/deepvit
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.signal

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import common as C  # noqa: E402

OFFICIAL = Path(os.environ.get("RIBEIRO_REPO", HERE / "official"))
sys.path.insert(0, str(OFFICIAL))
OUT = C.OUTPUTS / "deepvit"
FEAT = OUT / "features_8690.csv"
EPOCHS = 100                                # original: 300 epochs; here the common maximum of 100

# TSFEL 0.1.5 (pinned by the official requirements) calls scipy.signal.ricker / cwt,
# which recent SciPy removed. The original SciPy implementations are restored.
if not hasattr(scipy.signal, "ricker"):
    def ricker(points, a):
        A = 2 / (np.sqrt(3 * a) * (np.pi ** 0.25))
        vec = np.arange(0, points) - (points - 1.0) / 2
        return A * (1 - vec ** 2 / a ** 2) * np.exp(-vec ** 2 / (2 * a ** 2))
    scipy.signal.ricker = ricker
if not hasattr(scipy.signal, "cwt"):
    def cwt(data, wavelet, widths, dtype=None, **kwargs):
        out = np.empty((len(widths), len(data)), dtype=np.float64)
        for i, w in enumerate(widths):
            N = np.min([10 * w, len(data)])
            out[i] = scipy.signal.convolve(data, np.conj(wavelet(N, w, **kwargs)[::-1]), mode="same")
        return out
    scipy.signal.cwt = cwt


def _load(name, rel):
    """Import one pyDeepInsight module by path (its __init__ also imports grad-cam / torchcam)."""
    spec = importlib.util.spec_from_file_location(name, OFFICIAL / rel)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------ features (official definitions)
def earth_rad(lat):
    a, b = 6378137, 6356752
    lat = np.radians(lat)
    return np.sqrt(((a ** 2 * np.cos(lat)) ** 2 + (b ** 2 * np.sin(lat)) ** 2) / ((a * np.cos(lat)) ** 2 + (b * np.sin(lat)) ** 2))


def basic_features(df: pd.DataFrame, traj_id: int, label: str) -> pd.DataFrame:
    """Vectorised FeaturesGenerator.extractBasicFeatures for one trajectory (same formulas)."""
    lat, lon = df["lat"].to_numpy(float), df["lon"].to_numpy(float)
    t = df["datetime"].values.astype("datetime64[ms]").astype(np.int64) / 1000.0
    lo1, la1, lo2, la2 = map(np.radians, [lon[:-1], lat[:-1], lon[1:], lat[1:]])
    a = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    dist = earth_rad(lat[1:]) * 2 * np.arcsin(np.sqrt(a))
    bearing = np.degrees(np.arctan2(np.cos(la2) * np.sin(lo2 - lo1),
                                    np.cos(la1) * np.sin(la2) - np.sin(la1) * np.cos(la2) * np.cos(lo2 - lo1)))
    dur = np.diff(t)
    speed = np.where(dur > 0, dist / np.where(dur > 0, dur, 1), 0.0)
    stops = np.cumsum(speed < 0.6)
    stop_ratio = np.where(stops > 0, stops / np.arange(1, len(speed) + 1), 0.0)
    sp, be, du = np.r_[0.0, speed], np.r_[0.0, bearing], np.r_[0.0, dur]
    acc, jerk, br, brr = (np.zeros_like(sp) for _ in range(4))
    for i in range(1, len(sp)):
        if du[i] > 0:
            acc[i] = (sp[i] - sp[i - 1]) / du[i]; jerk[i] = (acc[i] - acc[i - 1]) / du[i]
            br[i] = (be[i] - be[i - 1]) / du[i]; brr[i] = (br[i] - br[i - 1]) / du[i]
    return pd.DataFrame({"traj_id": traj_id, "lat": lat[1:], "long": lon[1:], "haversine": dist, "duration": dur,
                         "speed": speed, "acceleration": acc[1:], "jerk": jerk[1:], "bearing": bearing,
                         "b_rate": br[1:], "br_rate": brr[1:], "stop_ratio": stop_ratio, "label": label})


def _patch_tsfel():
    """Compatibility with current NumPy / pandas only; numerical definitions unchanged."""
    import functools
    import inspect
    import tsfel.feature_extraction.features as TF
    for name, fn in inspect.getmembers(TF, inspect.isfunction):
        if fn.__module__ != TF.__name__ or getattr(fn, "_patched", False):
            continue

        def wrap(f):
            @functools.wraps(f)
            def g(signal, *a, **k):
                return f(np.asarray(signal, dtype=float), *a, **k)
            g._patched = True
            return g
        setattr(TF, name, wrap(fn))
    TF.autocorr = lambda s: float(np.correlate(np.asarray(s, float), np.asarray(s, float))[0])


def _extract_keep_invalid(gen):
    """Official extractFeatures, but rows with NaN/inf are kept (the original drops them)."""
    orig = pd.DataFrame.isnull
    try:
        pd.DataFrame.isnull = lambda self: pd.DataFrame(False, index=self.index, columns=self.columns)
        gen.extractFeatures()
    finally:
        pd.DataFrame.isnull = orig
    return gen.dataset


def _features_chunk(items):
    import warnings
    warnings.simplefilter("ignore")
    _patch_tsfel()
    from features_generator import FeaturesGenerator
    rows = []
    for j, df, lab in items:
        sub = FeaturesGenerator(pd.DataFrame()); sub.features = basic_features(df, j, lab)
        feats = _extract_keep_invalid(sub); feats["row"] = j
        rows.append(feats)
    return pd.concat(rows, ignore_index=True)


def build(meta):
    points = C.load_points(meta)
    items = [(j, points[t], lab) for j, (t, lab) in enumerate(zip(meta.traj_id, meta.label))]
    chunks = [items[i:i + 100] for i in range(0, len(items), 100)]
    OUT.mkdir(parents=True, exist_ok=True)
    parts, t0 = [], time.time()
    with Pool(int(os.environ.get("NPROC", 2))) as pool:
        for k, part in enumerate(pool.imap(_features_chunk, chunks)):
            parts.append(part)
            print(f"[features] {min((k + 1) * 100, len(items))}/{len(items)} {time.time() - t0:.0f}s", flush=True)
    df = pd.concat(parts, ignore_index=True).sort_values("row").reset_index(drop=True)
    assert len(df) == len(meta)
    df.to_csv(FEAT, index=False)


# ------------------------------------------------------------------ training
def train(meta):
    import torch
    import torch.nn as nn
    from sklearn.manifold import TSNE
    ImageTransformer = _load("pdi_image_transformer", "pyDeepInsight/image_transformer.py").ImageTransformer
    Norm2Scaler = _load("pdi_utils", "pyDeepInsight/utils.py").Norm2Scaler
    from archs.vit import DeepViTModel                 # dim 64, depth 12, heads 8, patch 32
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    feats = pd.read_csv(FEAT)
    assert (feats["row"].to_numpy() == np.arange(len(meta))).all()
    RX = feats.drop(columns=["label", "row"]).to_numpy(float)
    finite = np.isfinite(RX).all(1)
    RX = np.where(np.isfinite(RX), RX, 0.0)           # evaluation rows keep every trajectory
    users, Y = meta.uid.to_numpy(), meta.y.to_numpy()
    done = C.done_pairs(OUT)
    for seed in C.SEEDS:
        for f in C.load_folds(meta):
            if (f["fold"], seed) in done:
                continue
            tr = np.isin(users, f["train"]) & finite      # training rows with NaN/inf dropped, as in the original
            va, te = np.isin(users, f["validation"]), np.isin(users, f["test"])
            C.set_seed(seed); t0 = time.time()
            reducer = TSNE(n_components=2, perplexity=50, metric="cosine", init="random", learning_rate="auto",
                           random_state=seed, n_jobs=-1)
            it = ImageTransformer(feature_extractor=reducer, pixels=(64, 64), discretization="assignment")
            it.fit(RX[tr])                                  # t-SNE map from training users only
            ln = Norm2Scaler(); Ztr = ln.fit_transform(RX[tr])
            img = lambda Z: torch.from_numpy(np.transpose(it.transform(Z), (0, 3, 1, 2)).astype(np.float32)).to(dev)  # noqa: E731
            Itr, Iva, Ite = img(Ztr), img(ln.transform(RX[va])), img(ln.transform(RX[te]))
            ytr, yva = torch.from_numpy(Y[tr]).to(dev), Y[va]
            model = DeepViTModel().to(dev)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
            crit = nn.CrossEntropyLoss()
            g = torch.Generator().manual_seed(seed)

            def proba(I):
                model.eval()
                with torch.no_grad():
                    return torch.cat([torch.softmax(model(I[b:b + 512]), 1) for b in range(0, len(I), 512)]).cpu().numpy()

            best, best_ep, state = -1, 0, None
            for ep in range(1, EPOCHS + 1):
                model.train(); perm = torch.randperm(len(Itr), generator=g).to(dev)
                for b in range(0, len(perm), 128):
                    ix = perm[b:b + 128]
                    opt.zero_grad(); crit(model(Itr[ix]), ytr[ix]).backward(); opt.step()
                acc = (proba(Iva).argmax(1) == yva).mean()    # checkpoint: validation-user accuracy
                if acc > best:
                    best, best_ep, state = acc, ep, {k: v.detach().clone() for k, v in model.state_dict().items()}
            model.load_state_dict(state)
            C.save_predictions(OUT, meta, np.flatnonzero(te), proba(Ite), f["fold"], seed)
            json.dump({"fold": f["fold"], "seed": seed, "best_epoch": best_ep, "val_acc": float(best)},
                      open(OUT / f"run_f{f['fold']}_s{seed}.json", "w"))
            print(f"fold {f['fold']} seed {seed}: best epoch {best_ep}, {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    meta = C.load_meta()
    {"build": build, "train": train}[sys.argv[1]](meta)
