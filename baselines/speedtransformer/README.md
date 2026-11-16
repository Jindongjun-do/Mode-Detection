# SpeedTransformer of Zhang et al. [7]

Y. Zhang, O. Echchabi, T. Feng, W. Zhang, H.-K. Liao, and C. Chang, "Detecting transportation mode
using dense smartphone GPS trajectories and Transformer models," *Int. J. Geogr. Inf. Sci.*, 2026,
doi:10.1080/13658816.2026.2640213.

- Original repository: https://github.com/othmaneechc/SpeedTransformer
- Commit used: `2c0e7ac` (reference [30] of the paper)
- The original code is not copied into this repository; clone it as shown below.

`run_uid4fold.py` uses `TrajectoryModel` from `models/transformer/model_utils.py` of the original code
without changes, together with its `prepare_model`, `train_one_epoch` and `evaluate` methods.

## What is kept from the original code

- Input: the speed sequence in km/h.
- Windows of 200 values with stride 50. The last window is zero-padded and masked.
- Model: 4 Transformer layers, d_model 128, 8 heads, 4 key/value heads, dropout 0.1 (728,494 parameters).
- Optimizer: AdamW with learning rate 0.0002 and weight decay 0.0001.
- Training: gradient clipping 1.0, batch size 512, and mixed precision.
- Speed is standardized with `StandardScaler`.

## What was changed for the common user-disjoint protocol

| Item | Original code | Here |
|---|---|---|
| Data | GeoLife windows from the original preprocessing | Speed of the same 8,690 trajectories as the proposed model (no caps, duplicate timestamps removed) |
| Data split | Random 70/15/15 split of trajectories | Same user folds as the proposed model (`splits/uid_folds.csv`) |
| Scaler | Fitted on the training data | Fitted on the training users of each fold |
| Prediction unit | Window | Trajectory: mean of its window probabilities |
| Checkpoint | Lowest validation loss | Lowest validation loss on the validation users; at most 50 epochs with patience 7 (every run stopped by epoch 18) |
| Seeds | One run | 1042, 2042 and 3042; probabilities averaged |

## Run order

Run every command from the repository root, after `preprocess.py`. CUDA is required.

```bash
git clone https://github.com/othmaneechc/SpeedTransformer baselines/speedtransformer/official
git -C baselines/speedtransformer/official checkout 2c0e7ac
python baselines/speedtransformer/prepare_speed.py   # speed sequences of the 8,690 trajectories
python baselines/speedtransformer/run_uid4fold.py    # 4 folds × 3 seeds
python train_eval.py --evaluate outputs/baselines/speedtransformer
```

The output goes to `outputs/baselines/speedtransformer/`.

Paper result (Tables V–VI) for the three-seed ensemble on the 8,690 out-of-fold trajectories:
accuracy 81.19%, macro-F1 76.41%.
