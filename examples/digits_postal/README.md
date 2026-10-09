# Example: postal digit reader

A sorting machine needs to read handwritten zip-code digits from 8x8 scans.
This example tunes a classifier for that job under two hard deployment limits:
the model must fit on the sorting terminal (**under 5 MB**) and a single
configuration must not burn more than **30 seconds** of training time on the
shared workstation.

Configuration: [`config.yaml`](config.yaml) (also shipped as
`configs/digits_hgb.yaml`). Full output: [`report.md`](report.md).

## How to reproduce

```bash
autotuneml optimize --config configs/digits_hgb.yaml
autotuneml train     --config configs/digits_hgb.yaml --from-experiment digits_postal
autotuneml evaluate  digits_postal
autotuneml report    digits_postal
```

## Data

The scikit-learn digits set: 1797 labeled 8x8 images, 64 pixel features, 10
classes. Split once, stratified, seed 21: **1437 training rows, 360 test rows**.
The test rows were never touched during the search; they were read exactly once,
after the parameters were frozen.

## Search

30 trials, 5-fold cross-validation, NSGA-II over three objectives: validation
accuracy, training time, model size. Early stopping was on: the winning trial ran
**126 of its 353 allowed boosting rounds**.

|  | CV accuracy | train time | model size |
|---|---|---|---|
| baseline (defaults) | 0.9701 | 8.32 s | 2.32 MB |
| selected (trial 3) | 0.9715 | 3.82 s | 1.23 MB |

The Pareto front held two trials. Trial 15 is the interesting alternative:
0.9680 accuracy at 3.18 s and 0.61 MB — half the size for a third of a point of
accuracy. Trial 3 won on the configured weights (1.0 / 0.3 / 0.2) with all
constraints satisfied.

## Final model

Refit on all 1437 training rows, scored once on the 360 held-out rows:

| metric | value |
|---|---|
| accuracy | 0.9694 (349/360) |
| f1 | 0.9694 |
| log loss | 0.0680 |

Weakest digits on the test set: **8** (recall 0.914, confused with 1 and 5) and
**3** (recall 0.919). The full per-class table and confusion matrix are in
[`report.md`](report.md).

## Decision

Ship trial 3. It clears the 5 MB edge budget with 4x headroom, trains in under
4 seconds, and its test accuracy (0.9694) matches its cross-validation estimate
(0.9715) within half a point — no sign the search overfit the validation folds.
If the terminal gets tighter on storage, trial 15 is the documented fallback.

## Hardware note

This run happened on a CPU-only workstation (10-core Apple Silicon, 16 GB RAM),
so `device: auto` resolved to `cpu` — recorded in the experiment's `meta.json`
under `hardware`. On a CUDA machine the same configuration hands `xgboost` and
`lightgbm` the GPU automatically; `hist_gradient_boosting` itself is CPU-only.
Requesting `device: cuda` where no GPU exists fails fast instead of silently
training on the CPU.
