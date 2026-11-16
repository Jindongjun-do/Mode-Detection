"""SpeedTransformer [7] under the user-disjoint 4-fold protocol.

Speed windows of 200 values, stride 50, zero padding with a key-padding mask; official
TrajectoryModel (4 Transformer layers, d_model 128, 8 heads, 4 KV heads, dropout 0.1),
AdamW lr 2e-4, weight decay 1e-4, gradient clipping 1.0, batch 512, mixed precision.
Speed is standardized with training-user statistics. Checkpoint = lowest validation loss,
at most 50 epochs with patience 7. Trajectory probability = mean of its window probabilities.

Usage (from the repository root, CUDA required):
  python baselines/speedtransformer/prepare_speed.py
  python baselines/speedtransformer/run_uid4fold.py
  python train_eval.py --evaluate outputs/baselines/speedtransformer
"""

from __future__ import annotations

import copy
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DATA_PATH = REPO / "outputs" / "baselines" / "speedtransformer" / "speed_sequences_8690.csv"   # from prepare_speed.py
FOLD_PATH = REPO / "splits" / "uid_folds.csv"
# Official SpeedTransformer code (github.com/othmaneechc/SpeedTransformer, commit 2c0e7ac),
# cloned into baselines/speedtransformer/official; TrajectoryModel is imported from models/transformer.
SPEEDTRANSFORMER_DIR = HERE / "official" / "models" / "transformer"
OUT_DIR = REPO / "outputs" / "baselines" / "speedtransformer"

CLASS_ORDER = ["walk", "bike", "bus", "car", "train"]
CLASS_TO_IDX = {label: idx for idx, label in enumerate(CLASS_ORDER)}
SEEDS = [1042, 2042, 3042]
WINDOW_SIZE = 200
STRIDE = 50
BATCH_SIZE = 512
MAX_EPOCHS = 50
PATIENCE = 7
D_MODEL = 128
NHEAD = 8
NUM_LAYERS = 4
DROPOUT = 0.1
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP = 1.0
KV_HEADS = 4
USE_AMP = True

sys.path.insert(0, str(SPEEDTRANSFORMER_DIR))
from model_utils import TrajectoryModel  # noqa: E402


class WindowDataset(Dataset):
    def __init__(self, x: np.ndarray, masks: np.ndarray, y: np.ndarray):
        self.x = torch.from_numpy(x.astype(np.float32, copy=False)).unsqueeze(-1)
        self.masks = torch.from_numpy(masks.astype(bool, copy=False))
        self.y = torch.from_numpy(y.astype(np.int64, copy=False))

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.x[idx], self.masks[idx], self.y[idx]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def parse_uid_list(value: str) -> list[str]:
    return [part.strip().zfill(3) for part in str(value).split(",") if part.strip()]


def load_folds(all_users: set[str]) -> list[dict]:
    frame = pd.read_csv(FOLD_PATH, dtype=str)
    folds = []
    test_union: set[str] = set()
    for row in frame.to_dict("records"):
        test_users = set(parse_uid_list(row["test_uid_list"]))
        val_users = set(parse_uid_list(row["validation_uid_list"]))
        train_users = all_users - test_users - val_users
        if len(train_users) != 36 or len(val_users) != 9 or len(test_users) != 15:
            raise AssertionError(f"Fold {row['fold']} 사용자 수 오류")
        if train_users & val_users or train_users & test_users or val_users & test_users:
            raise AssertionError(f"Fold {row['fold']} 사용자 중복")
        test_union |= test_users
        folds.append({
            "fold": int(row["fold"]),
            "train": sorted(train_users),
            "validation": sorted(val_users),
            "test": sorted(test_users),
        })
    if test_union != all_users:
        raise AssertionError("4개 폴드 평가 사용자 합집합이 전체 60명과 다릅니다.")
    return folds


def make_windows(
    trajectories: dict[str, dict],
    trajectory_ids: list[str],
    scaler: StandardScaler,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    xs, masks, ys, owners = [], [], [], []
    for traj_id in trajectory_ids:
        record = trajectories[traj_id]
        sequence = scaler.transform(record["speed"].reshape(-1, 1)).reshape(-1).astype(np.float32)
        label_idx = CLASS_TO_IDX[record["label"]]
        for start in range(0, len(sequence), STRIDE):
            end = start + WINDOW_SIZE
            part = sequence[start:end]
            actual = len(part)
            padded = np.zeros(WINDOW_SIZE, dtype=np.float32)
            padded[:actual] = part
            mask = np.zeros(WINDOW_SIZE, dtype=bool)
            mask[actual:] = True
            xs.append(padded)
            masks.append(mask)
            ys.append(label_idx)
            owners.append(traj_id)
            if end >= len(sequence):
                break
    return np.stack(xs), np.stack(masks), np.asarray(ys), owners


@torch.no_grad()
def predict_probabilities(model: TrajectoryModel, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    model.model.eval()
    probs, labels = [], []
    for sequences, masks, y in loader:
        sequences = sequences.to(model.device, non_blocking=True)
        masks = masks.to(model.device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=model.scaler is not None):
            logits = model.model(sequences, None, src_key_padding_mask=masks)
        probs.append(F.softmax(logits, dim=1).cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(probs), np.concatenate(labels)


def aggregate_by_trajectory(
    window_probs: np.ndarray,
    window_labels: np.ndarray,
    owners: list[str],
    trajectories: dict[str, dict],
) -> pd.DataFrame:
    grouped: dict[str, list[np.ndarray]] = defaultdict(list)
    grouped_labels: dict[str, set[int]] = defaultdict(set)
    for prob, label, owner in zip(window_probs, window_labels, owners):
        grouped[owner].append(prob)
        grouped_labels[owner].add(int(label))
    rows = []
    for traj_id in sorted(grouped):
        if len(grouped_labels[traj_id]) != 1:
            raise AssertionError(f"창 레이블 불일치: {traj_id}")
        mean_prob = np.mean(np.stack(grouped[traj_id]), axis=0)
        row = {
            "traj_id": traj_id,
            "uid": trajectories[traj_id]["uid"],
            "true_label": trajectories[traj_id]["label"],
            "window_count": len(grouped[traj_id]),
        }
        row.update({f"p_{label}": float(mean_prob[idx]) for idx, label in enumerate(CLASS_ORDER)})
        rows.append(row)
    return pd.DataFrame(rows)


def metric_dict(y_true: list[str], probabilities: np.ndarray) -> dict:
    pred_idx = probabilities.argmax(axis=1)
    y_pred = [CLASS_ORDER[idx] for idx in pred_idx]
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, labels=CLASS_ORDER, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, labels=CLASS_ORDER, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=CLASS_ORDER, average="macro", zero_division=0)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred, labels=CLASS_ORDER)),
        "predictions": y_pred,
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA를 사용할 수 없어 학습을 중단합니다.")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Python:", sys.version)
    print("PyTorch:", torch.__version__)
    print("CUDA runtime:", torch.version.cuda)
    print("GPU:", torch.cuda.get_device_name(0))

    data = pd.read_csv(DATA_PATH, dtype={"traj_id": str, "label": str, "uid": str})
    data["uid"] = data["uid"].str.zfill(3)
    trajectories = {}
    for traj_id, group in data.groupby("traj_id", sort=True, observed=True):
        labels = group["label"].unique()
        users = group["uid"].unique()
        if len(labels) != 1 or len(users) != 1:
            raise AssertionError(f"궤적 레이블/사용자 불일치: {traj_id}")
        trajectories[traj_id] = {
            "speed": group["speed"].to_numpy(dtype=np.float32),
            "label": labels[0],
            "uid": users[0],
        }
    if len(trajectories) != 8_690:
        raise AssertionError(f"궤적 수 불일치: {len(trajectories)}")
    all_users = {record["uid"] for record in trajectories.values()}
    folds = load_folds(all_users)
    (OUT_DIR / "fold_assignment.json").write_text(
        json.dumps(folds, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    fold_seed_rows = []
    all_prediction_frames = []
    for fold_spec in folds:
        fold = fold_spec["fold"]
        ids = {
            split: sorted([tid for tid, rec in trajectories.items() if rec["uid"] in set(fold_spec[split])])
            for split in ["train", "validation", "test"]
        }
        if set(ids["train"]) & set(ids["validation"]) or set(ids["train"]) & set(ids["test"]):
            raise AssertionError(f"Fold {fold}: 궤적 중복")

        train_values = np.concatenate([trajectories[tid]["speed"] for tid in ids["train"]]).reshape(-1, 1)
        scaler = StandardScaler().fit(train_values)
        arrays = {
            split: make_windows(trajectories, ids[split], scaler)
            for split in ["train", "validation", "test"]
        }
        print(
            f"FOLD {fold}: users={len(fold_spec['train'])}/{len(fold_spec['validation'])}/{len(fold_spec['test'])}, "
            f"traj={len(ids['train'])}/{len(ids['validation'])}/{len(ids['test'])}, "
            f"windows={len(arrays['train'][2])}/{len(arrays['validation'][2])}/{len(arrays['test'][2])}",
            flush=True,
        )

        for seed in SEEDS:
            run_dir = OUT_DIR / f"fold_{fold}" / f"seed_{seed}"
            cached_predictions = run_dir / "test_trajectory_predictions.csv"
            cached_metrics = run_dir / "run_metrics.json"
            if cached_predictions.exists() and cached_metrics.exists():
                traj_predictions = pd.read_csv(
                    cached_predictions, dtype={"traj_id": str, "uid": str, "true_label": str}
                )
                row = json.loads(cached_metrics.read_text(encoding="utf-8"))
                if len(traj_predictions) != len(ids["test"]):
                    raise AssertionError(f"Fold {fold}, seed {seed}: 캐시 평가 수 불일치")
                all_prediction_frames.append(traj_predictions)
                fold_seed_rows.append(row)
                print(
                    f"  seed={seed}: cached accuracy={row['accuracy']:.4f}, "
                    f"macro_f1={row['macro_f1']:.4f}",
                    flush=True,
                )
                continue

            run_start = time.perf_counter()
            set_seed(seed)
            loaders = {}
            for split in ["train", "validation", "test"]:
                x, masks, y, _ = arrays[split]
                generator = torch.Generator().manual_seed(seed)
                loaders[split] = DataLoader(
                    WindowDataset(x, masks, y),
                    batch_size=BATCH_SIZE if split != "test" else BATCH_SIZE * 2,
                    shuffle=(split == "train"),
                    num_workers=0,
                    pin_memory=True,
                    generator=generator if split == "train" else None,
                )

            encoder = SimpleNamespace(classes_=np.asarray(CLASS_ORDER, dtype=object))
            model = TrajectoryModel(feature_columns=["speed"], label_encoder=encoder, use_amp=USE_AMP)
            model.prepare_model(
                window_size=WINDOW_SIZE,
                d_model=D_MODEL,
                nhead=NHEAD,
                num_layers=NUM_LAYERS,
                dropout=DROPOUT,
                lr=LEARNING_RATE,
                weight_decay=WEIGHT_DECAY,
                kv_heads=KV_HEADS,
            )

            best_val = float("inf")
            best_epoch = 0
            best_state = None
            epochs_no_improve = 0
            history = []
            torch.cuda.reset_peak_memory_stats()
            for epoch in range(1, MAX_EPOCHS + 1):
                train_loss, train_acc = model.train_one_epoch(loaders["train"], gradient_clip=GRADIENT_CLIP)
                val_loss, val_acc = model.evaluate(loaders["validation"])
                history.append({
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_accuracy": train_acc,
                    "validation_loss": val_loss,
                    "validation_accuracy": val_acc,
                })
                if val_loss < best_val:
                    best_val = val_loss
                    best_epoch = epoch
                    best_state = copy.deepcopy(model.model.state_dict())
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= PATIENCE:
                        break
            if best_state is None:
                raise RuntimeError("최적 체크포인트가 생성되지 않았습니다.")
            model.model.load_state_dict(best_state)

            window_probs, window_labels = predict_probabilities(model, loaders["test"])
            _, _, _, test_owners = arrays["test"]
            traj_predictions = aggregate_by_trajectory(
                window_probs, window_labels, test_owners, trajectories
            )
            traj_predictions.insert(0, "fold", fold)
            traj_predictions.insert(1, "seed", seed)
            probability_cols = [f"p_{label}" for label in CLASS_ORDER]
            metrics = metric_dict(
                traj_predictions["true_label"].tolist(),
                traj_predictions[probability_cols].to_numpy(),
            )
            traj_predictions["pred_label"] = metrics.pop("predictions")
            all_prediction_frames.append(traj_predictions)
            elapsed = time.perf_counter() - run_start
            row = {
                "fold": fold,
                "seed": seed,
                "best_epoch": best_epoch,
                "stop_epoch": history[-1]["epoch"],
                "best_validation_loss": best_val,
                "test_trajectories": len(traj_predictions),
                "test_windows": len(window_labels),
                "training_seconds": elapsed,
                "gpu_peak_memory_mb": torch.cuda.max_memory_allocated() / 1024**2,
                **metrics,
            }
            fold_seed_rows.append(row)
            run_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(history).to_csv(run_dir / "training_history.csv", index=False)
            traj_predictions.to_csv(run_dir / "test_trajectory_predictions.csv", index=False)
            cached_metrics.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
            torch.save(best_state, run_dir / "best_model.pth")
            print(
                f"  seed={seed}: best={best_epoch}, stop={history[-1]['epoch']}, "
                f"accuracy={row['accuracy']:.4f}, macro_f1={row['macro_f1']:.4f}, "
                f"time={elapsed:.1f}s",
                flush=True,
            )
            del model, loaders, best_state
            torch.cuda.empty_cache()

    predictions = pd.concat(all_prediction_frames, ignore_index=True)
    predictions.to_csv(OUT_DIR / "fold_seed_trajectory_predictions.csv", index=False)
    predictions[["traj_id", "uid", "true_label", "fold", "seed"] + [f"p_{c}" for c in CLASS_ORDER]].to_csv(
        OUT_DIR / "predictions_by_seed.csv", index=False)   # read by train_eval.py --evaluate
    fold_seed_metrics = pd.DataFrame(fold_seed_rows)
    fold_seed_metrics.to_csv(OUT_DIR / "fold_seed_metrics.csv", index=False)

    prob_cols = [f"p_{label}" for label in CLASS_ORDER]
    ensemble = (
        predictions.groupby(["fold", "traj_id", "uid", "true_label"], as_index=False, observed=True)[prob_cols]
        .mean()
        .sort_values(["fold", "traj_id"])
    )
    if ensemble["traj_id"].nunique() != 8_690 or len(ensemble) != 8_690:
        raise AssertionError("OOF 앙상블에서 각 궤적이 정확히 한 번 평가되지 않았습니다.")
    ensemble_metrics = metric_dict(ensemble["true_label"].tolist(), ensemble[prob_cols].to_numpy())
    ensemble["pred_label"] = ensemble_metrics.pop("predictions")
    ensemble.to_csv(OUT_DIR / "oof_ensemble_predictions.csv", index=False)

    cm = confusion_matrix(ensemble["true_label"], ensemble["pred_label"], labels=CLASS_ORDER)
    pd.DataFrame(cm, index=CLASS_ORDER, columns=CLASS_ORDER).to_csv(OUT_DIR / "confusion_matrix_raw.csv")
    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    pd.DataFrame(cm_norm, index=CLASS_ORDER, columns=CLASS_ORDER).to_csv(OUT_DIR / "confusion_matrix_row_normalized.csv")
    report = classification_report(
        ensemble["true_label"], ensemble["pred_label"], labels=CLASS_ORDER,
        output_dict=True, zero_division=0,
    )
    pd.DataFrame(report).T.to_csv(OUT_DIR / "classification_report.csv")

    seed_oof_rows = []
    for seed in SEEDS:
        frame = predictions[predictions["seed"] == seed].sort_values(["fold", "traj_id"])
        values = metric_dict(frame["true_label"].tolist(), frame[prob_cols].to_numpy())
        values.pop("predictions")
        seed_oof_rows.append({"seed": seed, **values})
    seed_oof = pd.DataFrame(seed_oof_rows)
    seed_oof.to_csv(OUT_DIR / "seed_oof_metrics.csv", index=False)

    summary = {
        "design": {
            "cohort": "8,690 GeoLife trajectories; 60 users; 5 classes",
            "split": "same 4-fold UID-disjoint allocation as proposed GATv2; 36 train, 9 validation, 15 test users",
            "seeds": SEEDS,
            "prediction_unit": "complete source trajectory",
            "window_probability_aggregation": "arithmetic mean within trajectory, then arithmetic mean across seeds",
            "model": "SpeedTransformer architecture from public repository",
            "window_size": WINDOW_SIZE,
            "stride": STRIDE,
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "checkpoint_metric": "validation loss",
            "feature": "speed in km/h",
        },
        "oof_three_seed_probability_ensemble": ensemble_metrics,
        "seed_oof_mean": {
            key: float(seed_oof[key].mean())
            for key in ["accuracy", "macro_precision", "macro_recall", "macro_f1", "cohen_kappa"]
        },
        "seed_oof_std": {
            key: float(seed_oof[key].std(ddof=1))
            for key in ["accuracy", "macro_precision", "macro_recall", "macro_f1", "cohen_kappa"]
        },
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("FINAL SUMMARY")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
