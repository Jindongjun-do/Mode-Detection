"""Training and evaluation of the three-layer GATv2 on the Stop-Move graphs.

Settings (graphs from preprocess.py)
  random  trajectory-level 70/15/15 split in splits/random_split.csv (8,640 trajectories)
  uid     user-disjoint 4-fold CV in splits/uid_folds.csv: 36 train / 9 validation / 15 test users

Training, per fold and seed
  z-scores and class weights from the training data only; class-weighted cross-entropy;
  AdamW lr 0.001, weight decay 0.0001, batch 64, dropout 0.2, gradient clipping 5.0;
  at most 100 epochs; checkpoint = highest validation macro-F1 (validation loss breaks ties);
  stop after 15 epochs without improvement; seeds 1042, 2042, 3042.

Evaluation
  The probabilities of the three seeds are averaged. For uid, the 8,690 out-of-fold predictions
  of the four folds are pooled.

Output: outputs/<setting>/
  fold<k>_seed<s>.pt         best checkpoint (state_dict) and standardization statistics
  predictions_by_seed.csv    test probabilities of every trajectory for every seed
  train_log.csv              best epoch, stop epoch, time per fold and seed
  ensemble_predictions.csv   mean of the seed probabilities and the predicted class
  metrics_ensemble.csv       accuracy, macro-F1, per-class precision / recall / F1
  metrics_by_seed.csv        the same for each single seed, plus mean and SD
  metrics_by_fold.csv        ensemble accuracy and macro-F1 per fold
  confusion_matrix.csv       counts, rows = true mode, columns = predicted mode
Finished (fold, seed) pairs are skipped when the script is run again.

Usage:  python train_eval.py --setting random
        python train_eval.py --setting uid
        python train_eval.py --evaluate outputs/baselines/cnn      # evaluation only
"""
from __future__ import annotations

import argparse
import copy
import os
import random
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv, global_add_pool, global_max_pool, global_mean_pool

from preprocess import BINARY_FEATURES, C2I, CLASSES, NODE_FEATURES, PROCESSED, REPO

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
warnings.filterwarnings("ignore", message=".*scatter\\(reduce='max'\\).*")

SPLITS = REPO / "splits"
OUTPUTS = REPO / "outputs"
SEEDS = [1042, 2042, 3042]
TRAIN = dict(lr=1e-3, weight_decay=1e-4, batch_size=64, dropout=0.2, max_epochs=100,
             patience=15, grad_clip=5.0, layers=3, hidden=128, heads=4)
PCOLS = [f"p_{c}" for c in CLASSES]


# ----------------------------------------------------------------------------- model
class GATv2Net(nn.Module):
    """Input projection, three GATv2 layers (GELU -> residual -> LayerNorm -> dropout),
    mean/max/sum readout and an MLP classifier; 161,797 parameters for 25 inputs."""

    def __init__(self, in_dim: int = 25, layers: int = 3, hidden: int = 128, heads: int = 4,
                 dropout: float = 0.2, n_classes: int = 5):
        super().__init__()
        self.inp = nn.Linear(in_dim, hidden)
        self.convs = nn.ModuleList([GATv2Conv(hidden, hidden // heads, heads=heads, concat=True,
                                              dropout=dropout, add_self_loops=True) for _ in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.Linear(hidden * 3, 128), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(128, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, n_classes))

    def forward(self, d):
        x = self.drop(F.gelu(self.inp(d.x)))
        for conv, norm in zip(self.convs, self.norms):
            x = self.drop(norm(F.gelu(conv(x, d.edge_index)) + x))
        g = torch.cat([global_mean_pool(x, d.batch), global_max_pool(x, d.batch), global_add_pool(x, d.batch)], 1)
        return self.head(g)


# ----------------------------------------------------------------------------- training
def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def make_folds(meta: pd.DataFrame, setting: str) -> list[dict]:
    if setting == "random":
        s = pd.read_csv(SPLITS / "random_split.csv").set_index("traj_id")["split"]
        part = meta["traj_id"].map(s)
        assert part.notna().all(), "graphs and splits/random_split.csv do not match"
        return [dict(fold=0, **{k: np.flatnonzero(part.eq(k).to_numpy()) for k in ("train", "validation", "test")})]
    folds, users = [], set(meta["uid"])
    for r in pd.read_csv(SPLITS / "uid_folds.csv", dtype=str).itertuples(index=False):
        te = {u.strip().zfill(3) for u in r.test_uid_list.split(",")}
        va = {u.strip().zfill(3) for u in r.validation_uid_list.split(",")}
        tr = users - te - va
        assert len(te) == 15 and len(va) == 9 and len(tr) == 36
        folds.append(dict(fold=int(r.fold), train=np.flatnonzero(meta.uid.isin(tr).to_numpy()),
                          validation=np.flatnonzero(meta.uid.isin(va).to_numpy()),
                          test=np.flatnonzero(meta.uid.isin(te).to_numpy())))
    assert len(np.unique(np.concatenate([f["test"] for f in folds]))) == len(meta)
    return folds


def standardize(graphs, train_idx):
    cont = [i for i, n in enumerate(NODE_FEATURES) if n not in BINARY_FEATURES]
    xs = torch.cat([graphs[i].x for i in train_idx])
    mu, sd = xs[:, cont].mean(0), xs[:, cont].std(0, unbiased=False)
    sd = torch.where(sd == 0, torch.ones_like(sd), sd)
    out = []
    for g in graphs:
        x = g.x.float().clone(); x[:, cont] = (x[:, cont] - mu) / sd
        out.append(Data(x=x, edge_index=g.edge_index, y=g.y.view(-1)[:1].long()))
    return out, {"mean": mu, "std": sd, "columns": cont}


@torch.no_grad()
def infer(model, loader, device, crit=None):
    model.eval(); P, Y, loss, n = [], [], 0.0, 0
    for b in loader:
        b = b.to(device); logits = model(b)
        if crit is not None:
            loss += float(crit(logits, b.y)) * b.num_graphs; n += b.num_graphs
        P.append(torch.softmax(logits, 1).cpu().numpy()); Y.append(b.y.cpu().numpy())
    return np.vstack(P), np.concatenate(Y), loss / max(n, 1)


def train_one(data, f, y, seed, device, T):
    w = torch.tensor(len(f["train"]) / (5 * np.bincount(y[f["train"]], minlength=5)), dtype=torch.float32, device=device)
    va = DataLoader([data[i] for i in f["validation"]], batch_size=T["batch_size"])
    set_seed(seed)
    tr = DataLoader([data[i] for i in f["train"]], batch_size=T["batch_size"], shuffle=True)
    model = GATv2Net(len(NODE_FEATURES), T["layers"], T["hidden"], T["heads"], T["dropout"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=T["lr"], weight_decay=T["weight_decay"])
    crit = nn.CrossEntropyLoss(weight=w)        # weighted mean normalized by the sum of weights
    best, state, best_ep, stale = (-np.inf, np.inf), None, 0, 0
    for ep in range(1, T["max_epochs"] + 1):
        model.train()
        for b in tr:
            b = b.to(device); opt.zero_grad(set_to_none=True)
            crit(model(b), b.y).backward()
            nn.utils.clip_grad_norm_(model.parameters(), T["grad_clip"]); opt.step()
        vp, vy, vl = infer(model, va, device, crit)
        vf = f1_score(vy, vp.argmax(1), average="macro", labels=range(5), zero_division=0)
        if vf > best[0] + 1e-12 or (abs(vf - best[0]) <= 1e-12 and vl < best[1]):
            best, state, best_ep, stale = (vf, vl), copy.deepcopy(model.state_dict()), ep, 0
        else:
            stale += 1
        if stale >= T["patience"]:
            break
    model.load_state_dict(state)
    return model, dict(best_epoch=best_ep, stop_epoch=ep, best_val_macro_f1=100 * best[0])


def train(setting: str, out: Path, seeds: list[int]):
    T = dict(TRAIN)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    graphs = torch.load(PROCESSED / f"graphs_{setting}.pt", weights_only=False)
    meta = pd.read_csv(PROCESSED / f"graphs_{setting}.csv", dtype={"uid": str})
    y = meta["label"].map(C2I).to_numpy()
    out.mkdir(parents=True, exist_ok=True)
    pred_path, log_path = out / "predictions_by_seed.csv", out / "train_log.csv"
    done = set()
    if pred_path.exists():
        d = pd.read_csv(pred_path, usecols=["fold", "seed"]).drop_duplicates()
        done = set(zip(d.fold, d.seed))

    for f in make_folds(meta, setting):
        if setting == "uid":
            assert not set(meta.uid.iloc[f["train"]]) & set(meta.uid.iloc[f["test"]])
        data, stats = standardize(graphs, f["train"])
        te = DataLoader([data[i] for i in f["test"]], batch_size=T["batch_size"])
        for seed in seeds:
            if (f["fold"], seed) in done:
                print(f"skip fold {f['fold']} seed {seed} (done)"); continue
            t0 = time.time()
            model, info = train_one(data, f, y, seed, device, T)
            p, ty, _ = infer(model, te, device)
            assert np.array_equal(ty, y[f["test"]])
            torch.save({"state_dict": model.state_dict(), "standardization": stats, "config": T},
                       out / f"fold{f['fold']}_seed{seed}.pt")
            df = pd.DataFrame(p, columns=PCOLS)
            df.insert(0, "traj_id", meta.traj_id.iloc[f["test"]].to_numpy())
            df.insert(1, "uid", meta.uid.iloc[f["test"]].to_numpy())
            df.insert(2, "true_label", meta.label.iloc[f["test"]].to_numpy())
            df.insert(3, "fold", f["fold"]); df.insert(4, "seed", seed)
            df.to_csv(pred_path, mode="a", header=not pred_path.exists(), index=False)
            acc = 100 * (p.argmax(1) == ty).mean()
            row = dict(fold=f["fold"], seed=seed, **info, test_accuracy=acc, seconds=round(time.time() - t0, 1))
            pd.DataFrame([row]).to_csv(log_path, mode="a", header=not log_path.exists(), index=False)
            print(f"fold {f['fold']} seed {seed}: test acc {acc:.2f}%, best epoch {info['best_epoch']}, "
                  f"stopped {info['stop_epoch']}, {row['seconds']} s", flush=True)


# ----------------------------------------------------------------------------- evaluation
def summarize(y: np.ndarray, pred: np.ndarray) -> dict:
    p, r, f, n = precision_recall_fscore_support(y, pred, labels=range(5), zero_division=0)
    out = {"n": int(len(y)), "accuracy": 100 * accuracy_score(y, pred),
           "macro_f1": 100 * f1_score(y, pred, labels=range(5), average="macro", zero_division=0)}
    for i, c in enumerate(CLASSES):
        out.update({f"{c}_precision": 100 * p[i], f"{c}_recall": 100 * r[i], f"{c}_f1": 100 * f[i], f"{c}_n": int(n[i])})
    return out


def evaluate(out: Path):
    """Ensemble the seeds in out/predictions_by_seed.csv and write the metric files."""
    pr = pd.read_csv(out / "predictions_by_seed.csv")
    pr["y"] = pr.true_label.map(C2I)
    seeds = sorted(pr.seed.unique())

    by_seed = pd.DataFrame([{"seed": s, **summarize(g.y.to_numpy(), g[PCOLS].to_numpy().argmax(1))}
                            for s, g in pr.groupby("seed")])
    agg = by_seed.drop(columns="seed").agg(["mean", "std"])          # SD with ddof = 1
    agg.insert(0, "seed", agg.index)
    pd.concat([by_seed, agg], ignore_index=True).to_csv(out / "metrics_by_seed.csv", index=False)

    ens = pr.groupby(["traj_id", "uid", "true_label", "fold", "y"], as_index=False)[PCOLS].mean()
    assert (pr.groupby("traj_id").seed.nunique() == len(seeds)).all(), "some trajectories are missing seeds"
    ens["pred_label"] = [CLASSES[i] for i in ens[PCOLS].to_numpy().argmax(1)]
    ens.drop(columns="y").to_csv(out / "ensemble_predictions.csv", index=False)
    yp = ens[PCOLS].to_numpy().argmax(1)
    m = summarize(ens.y.to_numpy(), yp)
    pd.DataFrame([m]).to_csv(out / "metrics_ensemble.csv", index=False)
    pd.DataFrame([{"fold": k, **{x: summarize(g.y.to_numpy(), g[PCOLS].to_numpy().argmax(1))[x]
                                 for x in ("n", "accuracy", "macro_f1")}} for k, g in ens.groupby("fold")]
                 ).to_csv(out / "metrics_by_fold.csv", index=False)
    cm = pd.DataFrame(confusion_matrix(ens.y.to_numpy(), yp, labels=range(5)), index=CLASSES, columns=CLASSES)
    cm.to_csv(out / "confusion_matrix.csv")

    print(f"{out}: {m['n']} trajectories, {len(seeds)}-seed ensemble")
    print(f"  accuracy {m['accuracy']:.2f}%  macro-F1 {m['macro_f1']:.2f}%")
    print(f"  single seed: accuracy {agg.loc['mean', 'accuracy']:.2f} ± {agg.loc['std', 'accuracy']:.2f}, "
          f"macro-F1 {agg.loc['mean', 'macro_f1']:.2f} ± {agg.loc['std', 'macro_f1']:.2f}")
    print(cm.to_string())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--setting", choices=["random", "uid"], help="train and evaluate")
    g.add_argument("--evaluate", type=Path, metavar="DIR", help="only evaluate DIR/predictions_by_seed.csv")
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    args = ap.parse_args()
    if args.evaluate is not None:
        evaluate(args.evaluate)
        return
    out = OUTPUTS / args.setting
    train(args.setting, out, args.seeds)
    evaluate(out)


if __name__ == "__main__":
    main()
