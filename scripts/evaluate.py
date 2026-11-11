#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run all three transfer directions and summarise them.

    python scripts/evaluate.py --root data --ckpt-dir runs --out results/summary.json

Layout expected under --root:

    data/
      BOE/   {train,val,test}  class folders AMD DME NORMAL
      TMI/   {train,val,test}
      CELL/  {train,val,test}

Directions: A->B, A->C, B->C, with A = BOE, B = TMI, C = CELL.  The source model
for A->B is trained on A and the one for B->C on B; the target split is always
the `test` folder of the second cohort.
"""
import argparse
import itertools
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from step.model import build_model     # noqa: E402
from step.pipeline import run_step     # noqa: E402

DIRECTIONS = {"AB": ("BOE", "TMI"), "AC": ("BOE", "CELL"), "BC": ("TMI", "CELL")}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data")
    p.add_argument("--ckpt-dir", default="runs")
    p.add_argument("--out", default="results/summary.json")
    p.add_argument("--seeds", default="42,123,777,2024,3407")
    p.add_argument("--device", default="cuda")
    a = p.parse_args()

    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    import torch
    device = a.device if torch.cuda.is_available() else "cpu"

    rows = []
    for name, (src, tgt) in DIRECTIONS.items():
        ckpt = os.path.join(a.ckpt_dir, "source_%s.pth" % src)
        if not os.path.exists(ckpt):
            print("[skip] missing", ckpt)
            continue
        for seed in seeds:
            model = build_model(ckpt).to(device)
            out = run_step(model, model.layer3, model.layer4, model.head,
                           os.path.join(a.root, tgt, "test"), device=device, seed=seed)
            rows.append(dict(direction=name, source=src, target=tgt, seed=seed,
                             acc=out["acc"]))
            print("%s  seed %-5d  acc %.2f" % (name, seed, out["acc"]), flush=True)

    if rows:
        import statistics
        summary = {}
        for name in DIRECTIONS:
            acc = [r["acc"] for r in rows if r["direction"] == name]
            if acc:
                summary[name] = dict(mean=round(statistics.mean(acc), 2),
                                     sd=round(statistics.stdev(acc), 2) if len(acc) > 1 else 0.0,
                                     n=len(acc))
        print(json.dumps(summary, indent=2))
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump(dict(rows=rows, summary=summary), fh, indent=2)


if __name__ == "__main__":
    main()
