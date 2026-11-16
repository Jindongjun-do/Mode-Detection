# Variable-Size Stop-Move Segment Representation of Complete GPS Trajectories for User-Independent Transportation Mode Detection

## Data

GeoLife GPS Trajectories 1.3 (Microsoft Research): https://www.microsoft.com/en-us/download/details.aspx?id=52367

Unzip it so that the user folders are at `data/raw/Geolife Trajectories 1.3/Data/000 ... 181`, or pass
the path with `--geolife`.

## Overview

Code for the paper by Dongjun Jin, Eunjeong Ko, Suji Kim, and Jisup Shim (IEEE Access, under review).

Each complete single-mode GPS trajectory becomes one graph. Each node is a consecutive stationary or
moving behavioral segment with 25 features, and edges connect temporally adjacent segments. A
three-layer GATv2 classifies the trajectory as walk, bike, bus, car or train.

| Evaluation | Trajectories | Accuracy | Macro-F1 |
|---|---|---|---|
| Trajectory-level random split (test set) | 8,640 (1,296 test) | 94.37% | 92.97% |
| User-disjoint 4-fold CV (out-of-fold) | 8,690 | 85.60% | 82.92% |

## Repository layout

```
preprocess.py       GeoLife -> labeled trajectories -> Stop-Move graphs
train_eval.py       GATv2 model, training, seed ensemble and metrics
splits/             random_split.csv (8,640 trajectories) and uid_folds.csv (4 user folds)
baselines/          comparison models under the same user folds
  cnn/              CNN of Dabiri and Heaslip [4]
  deepvit/          DeepViT of Ribeiro et al. [6]
  speedtransformer/ SpeedTransformer of Zhang et al. [7]
```

Each `baselines/<model>/README.md` gives the original repository, the commit used, what was changed,
and the run order.

## Running

```bash
pip install -r requirements.txt     # PyTorch Geometric: see https://pytorch-geometric.readthedocs.io

python preprocess.py                # trajectories and graphs for both settings
python train_eval.py --setting random
python train_eval.py --setting uid
```

Tested with Python 3.12, PyTorch 2.10 (CUDA 12.6), PyTorch Geometric 2.8 and an NVIDIA RTX 4090.

| Step | Output | Time |
|---|---|---|
| `preprocess.py` | `data/processed/`: 8,690 trajectories; `graphs_random` (8,640 graphs, 158,044 nodes); `graphs_uid` (8,690 graphs, 279,696 nodes) | about 20 min |
| `train_eval.py --setting random` | `outputs/random/` | about 8 min |
| `train_eval.py --setting uid` | `outputs/uid/` | about 20 min |

`train_eval.py` writes the ensemble accuracy and macro-F1, per-class scores, per-seed and per-fold
scores, and the confusion matrix.

| Paper item | Setting |
|---|---|
| Tables III–IV, Fig. 4 | `random` |
| Tables V–IX, Figs. 5–6 | `uid` |

## Method summary

**Labeled trajectories**
- GeoLife modes are mapped to five classes: taxi → car and subway → train.
- Same-label intervals of a user that overlap or touch are merged, and time ranges claimed by two
  different labels are removed. A point belongs to an interval if it is in [start, end).
- An interval is kept if it has at least 20 points, lasts at least 60 s and covers at least 50 m,
  after duplicate timestamps and jumps faster than 80 m/s are removed. This gives 8,690 trajectories
  from 60 users.

**Kinematics**
- Haversine distance (R = 6,371 km); the first point of each duplicated timestamp is kept.
- A pair of consecutive points more than 600 s apart is a logging gap: its distance, speed and the
  derived acceleration, jerk and bearing rate are set to 0.
- A point is stationary if its speed is below 0.5 m/s (1.8 km/h).

**Settings**

| Setting | Cohort | Segmentation |
|---|---|---|
| `random` | Mode-specific speed and acceleration caps of [4] applied repeatedly, validity criteria checked again: 8,640 trajectories | `merge_short` |
| `uid` | No caps: all 8,690 trajectories | `merge_short_split_long` |

A new segment starts whenever the stop flag changes. Then:
- `merge_short`: a segment with fewer than 3 points or shorter than 5 s is merged into the preceding
  segment.
- `merge_short_split_long`: moving segments are first cut into pieces of at least 3 points each time
  a piece reaches 180 s or 500 m; then `merge_short`.

The stop-segment indicator is 1 when at least half of a segment's points are stationary.

**Node features (25)**: duration, cumulative distance, displacement, straightness; mean, median,
90th percentile, maximum and SD of speed; mean, 90th percentile and maximum of moving-point speed;
stationary-point ratio, total stop time, maximum consecutive stop time, stop-segment indicator,
number of stationary points; mean, SD and 95th percentile of absolute acceleration; mean and SD of
jerk; mean and SD of bearing rate; number of GPS points. Continuous features are z-scored with
training-fold statistics.

**Training**
- Class-weighted cross-entropy; AdamW, learning rate 0.001, weight decay 0.0001.
- Batch size 64, dropout 0.2, gradient clipping 5.0.
- At most 100 epochs. The checkpoint with the highest validation macro-F1 is kept (validation loss
  breaks ties), and training stops after 15 epochs without improvement.
- Seeds 1042, 2042 and 3042; the final prediction averages the three seeds' probabilities.

## Citation

```
D. Jin, E. Ko, S. Kim, and J. Shim, "Variable-size stop-move segment representation of complete GPS
trajectories for user-independent transportation mode detection," IEEE Access, under review.
```
