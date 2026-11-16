# DeepInsight + DeepViT of Ribeiro et al. [6]

R. Ribeiro, A. Trifan, and A. J. R. Neves, "A deep learning approach for transportation mode
identification using a transformation of GPS trajectory data features into an image representation,"
*Int. J. Data Sci. Anal.*, vol. 20, no. 2, pp. 1023–1032, 2025, doi:10.1007/s41060-024-00510-3.

- Original repository: https://github.com/RRibeiro8/transport-mode-identification-memoria
- Commit used: `3df4293` (reference [29] of the paper)
- License of the original code: GPL-3.0. It is not copied into this repository; clone it as shown below.

`ribeiro_deepvit.py` imports these parts of the original code directly:
- `features_generator.FeaturesGenerator.extractFeatures`
- `pyDeepInsight/image_transformer.py`
- `pyDeepInsight/utils.py` (`Norm2Scaler`)
- `archs/vit.py` (`DeepViTModel`)

## What is kept from the original code

- 399 features per trajectory: 41 TSFEL features for each of 7 point signals, plus 112 descriptive
  statistics. They are computed by the original `extractFeatures`.
- The DeepInsight image: t-SNE (perplexity 50, cosine) with `assignment` discretization, giving a
  64 × 64 RGB image, followed by `Norm2Scaler`.
- `DeepViTModel` with dim 64, depth 12, heads 8 and patch 32 (2,574,789 parameters).
- AdamW with learning rate 0.0001, batch size 128, and unweighted cross-entropy.
- The point-level basic features (`extractBasicFeatures`) use the same formulas. They are vectorized
  because the original row-by-row loop is too slow for 4.8 M points.

## What was changed for the common user-disjoint protocol

| Item | Original code | Here |
|---|---|---|
| Outlier removal | Mode-specific speed cap | None |
| Trajectory unit | Re-split every 20 min | The 8,690 complete trajectories; one image per trajectory |
| t-SNE map and `Norm2Scaler` | Fitted on all trajectories before the split | Fitted on the training users of each fold only |
| Data split | Random | Same user folds as the proposed model (`splits/uid_folds.csv`) |
| Epochs | 300 | 100 |
| Checkpoint | Best accuracy on the test data | Best accuracy on the validation users |
| Rows with NaN/inf features | Dropped | Dropped from training. Validation and test rows are kept with NaN/inf set to 0, so that every trajectory gets a prediction. |
| Seeds | One run | 1042, 2042 and 3042; probabilities averaged |
| TSFEL 0.1.5 | Old NumPy/SciPy | Wrapped for current NumPy and SciPy, and `scipy.signal.ricker`/`cwt` restored; the numerical definitions are unchanged |

Known issue: for 83 trajectories, the model outputs NaN probabilities. `np.argmax` maps them to the first
class (walk), and the reported results keep that behavior.

## Run order

Run every command from the repository root, after `preprocess.py`.

```bash
git clone https://github.com/RRibeiro8/transport-mode-identification-memoria baselines/deepvit/official
git -C baselines/deepvit/official checkout 3df4293
pip install -r baselines/requirements.txt
python baselines/deepvit/ribeiro_deepvit.py build   # 399 features per trajectory, CPU (NPROC workers)
python baselines/deepvit/ribeiro_deepvit.py train   # 4 folds × 3 seeds, GPU
python train_eval.py --evaluate outputs/baselines/deepvit
```

The output goes to `outputs/baselines/deepvit/`. If the original code is somewhere else, set
`RIBEIRO_REPO=<path>`.

Paper result (Tables V–VI) for the three-seed ensemble on the 8,690 out-of-fold trajectories:
accuracy 83.08%, macro-F1 78.16%.
