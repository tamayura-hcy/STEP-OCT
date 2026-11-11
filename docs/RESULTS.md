# Results

Accuracy (%) and macro one-vs-rest AUC (%), mean ± sample standard deviation over five
seeds. Source: the paper, Tables I and II.

## Main comparison

| Method | A→B Acc | A→C Acc | B→C Acc | A→B AUC | A→C AUC | B→C AUC |
|---|---|---|---|---|---|---|
| Frozen source ResNet-50 | 57.06±2.50 | 70.68±0.56 | 61.76±7.39 | 85.93±1.53 | 86.86±0.86 | 86.12±3.36 |
| **STEP** | **92.92±1.80** | **90.65±0.34** | **90.39±2.35** | **97.00±0.90** | **97.89±0.25** | **97.19±0.77** |

Eight reference adapters (BN-adapt, T3A, SAR, SAR², COME, TENT, Mint, DeYO) are compared
in the paper on the same backbone, the same splits and the same single contrast-corrected
input; STEP is highest in every column, at 5.6 frozen-forward equivalents per image.

## Stage ablation

| Stage | Configuration | A→B | A→C | B→C |
|---|---|---|---|---|
| L0 | Frozen source, raw images | 57.06 | 70.68 | 61.76 |
| L1 | Cohort normalisation | 68.76 | 71.00 | 76.45 |
| L2 | Four-view initialization | 82.79 | 75.84 | 81.43 |
| L3 | One projection update | 86.91 | 82.04 | 85.20 |
| L4 | Fixed rank, 30 updates | 90.41 | 84.91 | 86.38 |
| L5 | Progressive rank, 30 updates | **92.92** | **90.65** | **90.39** |

## Reproduction notes

* The five seeds are 42, 123, 777, 2024 and 3407; the only stochastic parts of STEP are
  the minibatch order in the buffer sweep and the two fixed noise banks.
* `scripts/evaluate.py` writes `results/summary.json` with one record per (direction,
  seed) containing accuracy, macro AUC, per-class recall and the runtime.
* Numbers in this file were produced with one RTX 4060 Ti (16 GB); step 2 of the quick
  start takes a few seconds per direction on the 918-image cohort B split.
