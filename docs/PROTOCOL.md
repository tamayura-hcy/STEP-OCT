# Protocol

What is shared, what was selected during development, and what never enters an
adaptation update.

## Budget and setting

* One frozen source model per direction: ResNet-50, three-class head of two 1024-unit
  ReLU layers, trained on the source cohort and then discarded as a training set.
* Transductive: the whole unlabeled target cohort is available at once and receives the
  predictions. At most **five image-equivalent forward passes per image** — one for the
  buffer re-estimation and four for the views.
* No backward pass, no optimiser, no learning rate, no stopping rule.
* No target label and no source image enters any adaptation update. Labels are read
  only to score the result.

## Shared configuration

One configuration is used for all three directions (`configs/default.yaml`):

| Item | Value |
|---|---|
| Input | shorter side 292, centre crop 256×256, intensities in [0, 1] |
| Batch for buffer re-estimation | 64, label-blind shuffled |
| BN momentum / eps | 0.1 / 1e-6 |
| Corrected views | `Q(3, tau_i)` with `tau_i = 0.30 × mean_c median_spat(c_3)`, and `Q(5, 0.24)` |
| Noise views | `clip(x + 0.007 η)`, η from two fixed banks of 64 tensors, indexed by image index mod 64 |
| Subspace | thin SVD of the standardised descriptor matrix, `D = 80` |
| Rank schedule | 16, 23, 30, 37, 44, 52, 59, 66, 73, 80 |
| Updates per rank | 3 |
| Seeds | 42, 123, 777, 2024, 3407 |

## What was selected during development

Two design choices were fixed before the final runs, using labelled evaluations on
these same cohorts, and are declared in the paper:

1. **The peak-position rule** (`k = 1`: the single position with the largest maximum
   class response supplies the descriptor value, and the logits at the peak position
   supply the initial label). Alternatives `k = 4`, `k = 8` and class-wise maxima were
   measured and are reported in the ablation.
2. **The rank schedule** — the start rank, the end rank, the number of stages and the
   number of updates per stage — chosen from one-at-a-time sweeps around the shared
   configuration.

Everything else (the gain rule, the two window widths, the noise level, the subspace
dimension, the view composition) is either read off the target cohort or fixed by
construction.

## Scoring

Accuracy and macro one-vs-rest AUC in percent, mean and sample standard deviation over
the five seeds. Every baseline follows its own test-time procedure on the same target
split, with the same frozen backbone and the same target images, so the rows of the
comparison share everything except the adaptation.
