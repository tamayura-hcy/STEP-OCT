#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Train the frozen source model used by every transfer direction.

    python scripts/train_source.py --data data/BOE/train --val data/BOE/val \
        --out runs/source_BOE.pth --epochs 30

The checkpoint is a plain dict {"state_dict": ...} so that `build_model` can read
it back.  Evaluation (accuracy / AUC) is done on the held-out split of the same
cohort; nothing about the target cohort is used here.
"""
import argparse
import os
import sys

import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from step.data import make_loader     # noqa: E402
from step.model import build_model    # noqa: E402


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y, _ in loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.numel()
    return 100.0 * correct / max(total, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--val", default="")
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()

    device = a.device
    train_loader, _ = make_loader(a.data, a.batch_size, shuffle=True, train=True,
                                  workers=a.workers)
    val_loader = None
    if a.val:
        val_loader, _ = make_loader(a.val, a.batch_size, workers=a.workers)

    model = build_model().to(device)
    for q in model.parameters():
        q.requires_grad_(True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)

    best = -1.0
    for ep in range(a.epochs):
        model.train()
        run, seen = 0.0, 0
        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            loss = crit(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            run += loss.item() * y.numel()
            seen += y.numel()
        sched.step()
        msg = "epoch %02d  loss %.4f" % (ep + 1, run / max(seen, 1))
        if val_loader is not None:
            acc = evaluate(model, val_loader, device)
            msg += "  val %.2f%%" % acc
            if acc > best:
                best = acc
                model.eval()
                for q in model.parameters():
                    q.requires_grad_(False)
                os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
                torch.save({"state_dict": model.state_dict()}, a.out)
                for q in model.parameters():
                    q.requires_grad_(True)
        print(msg, flush=True)

    if val_loader is None:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        torch.save({"state_dict": model.state_dict()}, a.out)
    print("saved", a.out)


if __name__ == "__main__":
    main()
