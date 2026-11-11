#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run STEP on one unlabeled target cohort.

    python scripts/run_step.py --target data/TMI/test --ckpt runs/source_BOE.pth \
        --out runs/AB_step42.json --seed 42 --device cuda

Prints the accuracy, the macro one-vs-rest AUC and the per-class recall.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from step.model import build_model          # noqa: E402
from step.pipeline import run_step          # noqa: E402
from step.readout import RANKS              # noqa: E402


def macro_auc(y, scores, k=3):
    from sklearn.metrics import roc_auc_score
    aucs = []
    for c in range(k):
        try:
            aucs.append(roc_auc_score((y == c).astype(int), scores[:, c]))
        except ValueError:
            pass
    return float(np.mean(aucs)) * 100.0 if aucs else float("nan")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target", required=True, help="ImageFolder of the unlabeled target cohort")
    p.add_argument("--ckpt", required=True, help="frozen source checkpoint")
    p.add_argument("--out", default="", help="where to write the JSON report")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--rounds", type=int, default=3, help="updates at each rank")
    p.add_argument("--ranks", default=",".join(str(r) for r in RANKS))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()

    model = build_model(a.ckpt).to(a.device)
    ranks = [int(r) for r in a.ranks.split(",") if r.strip()]

    out = run_step(model, model.layer3, model.layer4, model.head, a.target,
                   batch_size=a.batch_size, ranks=ranks, rounds=a.rounds,
                   device=a.device, seed=a.seed, workers=a.workers)

    recall = [float((out["pred"][out["y"] == c] == c).mean() * 100.0)
              for c in range(len(out["classes"])) if (out["y"] == c).any()]
    report = dict(acc=out["acc"], auc=macro_auc(out["y"], out["scores"]),
                  recall=recall, classes=out["classes"],
                  ranks=ranks, rounds=a.rounds, seed=a.seed, target=a.target,
                  ckpt=a.ckpt)
    print(json.dumps(report, indent=2))
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)


if __name__ == "__main__":
    main()
