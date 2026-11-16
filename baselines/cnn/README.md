# CNN of Dabiri and Heaslip [4]

S. Dabiri and K. Heaslip, "Inferring transportation modes from GPS trajectories using a convolutional
neural network," *Transp. Res. Part C*, vol. 86, pp. 360–371, 2018, doi:10.1016/j.trc.2017.11.021.

- Original repository: https://github.com/sinadabiri/Transport-Mode-GPS-CNN
- Commit used: `4a0fede` (reference [28] of the paper)
- License of the original code: MIT

The original code is written in Keras. `dabiri_cnn.py` reimplements `Instance_Creation.py`,
`Keras_Data_Creation.py` and `CNN-Keras.py` in PyTorch. It does not import the original files, so
there is nothing to clone. The original repository is needed only to compare against.

## What is kept from the original code

- Instances of up to 200 consecutive GPS points. A new instance starts after a gap of more than
  20 min. Instances with fewer than 10 points, or with zero total distance, are dropped.
- Four channels computed with the original formulas: speed, acceleration, jerk and bearing rate.
  This includes the +1 s added to every time interval and the original bearing expression.
- Savitzky–Golay smoothing (window 9, order 3) using the original function.
- Zero padding to 200 points and no feature scaling.
- The same network:
  - Conv(32, 32)–Pool–Conv(64, 64)–Pool–Conv(128, 128)–Pool
  - Dropout 0.5, then a dense layer of a quarter of the flattened size, then Dropout 0.5 and softmax
  - 2,660,869 parameters
- Adam with learning rate 0.001, batch size 64, and Keras default (Glorot uniform) initialization.

## What was changed for the common user-disjoint protocol

| Item | Original code | Here |
|---|---|---|
| Outlier removal | Mode-specific speed, acceleration and jerk caps (uses the true label) | None |
| Data split | Random split of instances | Same user folds as the proposed model (`splits/uid_folds.csv`): 36 training, 9 validation and 15 test users |
| Epochs | 62 | 100 |
| Checkpoint | Best accuracy on the test data | Best accuracy on the validation users |
| Prediction unit | Instance | Trajectory: mean of the trajectory's instance probabilities |
| Seeds | One run | 1042, 2042 and 3042; the three seeds' probabilities are averaged |
| Distance | `vincenty` (removed from geopy) | `geopy.distance.geodesic`; haversine for the few GeoLife points with latitude outside ±90° |
| Layer type | Keras `Conv2D` with kernel (1, 3) on a 1 × 200 input | `Conv1d` with kernel 3 (mathematically the same) |

## Run order

Run every command from the repository root, after `preprocess.py`.

```bash
pip install -r baselines/requirements.txt
python baselines/cnn/dabiri_cnn.py build     # instance cache (28,753 instances), CPU, about 1 h
python baselines/cnn/dabiri_cnn.py train     # 4 folds × 3 seeds, GPU
python train_eval.py --evaluate outputs/baselines/cnn
```

The output goes to `outputs/baselines/cnn/`. Fold and seed pairs that are already finished are skipped
when `train` is run again.

Paper result (Tables V–VI) for the three-seed ensemble on the 8,690 out-of-fold trajectories:
accuracy 80.79%, macro-F1 76.20%.
