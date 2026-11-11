# STEP — Staged Target-Label Estimation by Projection

[English](README.md) | [中文](README.zh-CN.md) | [日本語](README.ja.md)

Source-free, gradient-free adaptation of a frozen OCT classifier to a new scanner.

**Paper:** *STEP: Staged Target-Label Estimation by Projection for Source-Free
Cross-Device OCT Classification*, submitted to IEEE Transactions on Instrumentation
and Measurement.
**Code:** https://github.com/tamayura-hcy/STEP-OCT

---

## What this repository contains

A B-scan is not a photograph of tissue: it is the output of a measurement chain, and
two scanners are two transfer characteristics of the same measurand. A classifier
trained on one device therefore reads a different instrument, not a different retina.
STEP repairs that difference in two places and nothing else:

1. **On the images.** A local contrast equaliser whose strength is read off each
   image, followed by a re-estimation of the encoder's normalisation buffers on the
   unlabeled target cohort.
2. **On the decision.** One provisional label per image, read at the spatial peak of
   the stage-4 map, then refined inside a projection subspace whose rank is raised on
   a fixed schedule from 16 to 80.

No source image, no target label, no gradient, no optimiser, and no constant fitted to
the cohort being classified. Five image-equivalent forward passes per image.

| | A→B | A→C | B→C |
|---|---|---|---|
| Frozen source ResNet-50 | 57.06 | 70.68 | 61.76 |
| **STEP** | **92.92** | **90.65** | **90.39** |

(accuracy %, mean over five seeds; see [`docs/RESULTS.md`](docs/RESULTS.md))

## Install

```bash
git clone https://github.com/tamayura-hcy/STEP-OCT.git
cd STEP-OCT
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Python ≥ 3.9, PyTorch ≥ 2.0. A single 12 GB GPU is enough; CPU works for a small cohort.

## Data

The three public cohorts are **not** redistributed here. `docs/DATA.md` lists the
sources, the licences and the folder layout expected by the scripts:

```
data/
  BOE/   {train,val,test}/{AMD,DME,NORMAL}/*.jpeg     # 2014_BOE_Srinivasan (A)
  TMI/   {train,val,test}/{AMD,DME,NORMAL}/*.jpeg     # Macular Dataset–Heidelberg (B)
  CELL/  {train,val,test}/{AMD,DME,NORMAL}/*.jpeg     # OCT2017 / ZhangLabData (C)
```

Every image is resized on the shorter side to 292 and centre-cropped to 256x256 by
the loader; nothing else is done to the pixels.

## Quick start

```bash
# 1. train the frozen source model for cohort A
python scripts/train_source.py --data data/BOE/train --val data/BOE/val \
    --out runs/source_BOE.pth --epochs 30

# 2. adapt it to cohort B with no labels and no gradients
python scripts/run_step.py --target data/TMI/test --ckpt runs/source_BOE.pth \
    --out runs/AB_step42.json --seed 42

# 3. all three directions, five seeds, one table
python scripts/evaluate.py --root data --ckpt-dir runs --out results/summary.json
```

Step 2 prints accuracy, macro one-vs-rest AUC and per-class recall.

## Repository layout

```
step/                 the method, in six short modules
  equalise.py         Eq. (1): local contrast equaliser, cohort-measured gain
  views.py            the four views: two corrected, two noise realisations
  bn.py               buffer re-estimation on the unlabeled cohort
  extract.py          Eq. (2): peak descriptor H and initial labels T0
  readout.py          Eq. (3): progressive-rank refinement in one subspace
  pipeline.py         run_step(): the whole thing on one target cohort
  model.py, data.py   ResNet-50 + head, ImageFolder loader
scripts/              train_source.py, run_step.py, evaluate.py
configs/default.yaml  every constant of the shared configuration
docs/                 DATA.md, PROTOCOL.md, RESULTS.md
```

## Configuration

All numbers in `configs/default.yaml` are the ones reported in the paper and are
shared by all three directions: 256x256 crop, batch 64, BN momentum 0.1, eps = 1e-6,
tau = 0.30 x (channel mean of the median local contrast) for the 3x3 view and 0.24 for
the fixed 5x5 view, noise sigma = 0.007 from two fixed banks of 64 tensors, D = 80,
ranks 16 → 80 in ten steps of three updates.

The peak-position rule and the rank schedule were selected during development on
labelled evaluations of these same cohorts; `docs/PROTOCOL.md` records exactly what was
selected on what, and no label enters an adaptation update.

## Citation

```bibtex
@article{han2026step,
  title={STEP: Staged Target-Label Estimation by Projection for Source-Free Cross-Device OCT Classification},
  author={Han, Chengyou and Luo, Yuemei and Liu, Weide and Cheng, Jun},
  journal={IEEE Transactions on Instrumentation and Measurement},
  year={2026}
}
```

## License and data

Code released under the MIT License (see `LICENSE`). The three cohorts keep their own
licences and are not redistributed; please obtain them from the original providers.

## Contact

Corresponding author: Yuemei Luo — luoyuemei@nuist.edu.cn
