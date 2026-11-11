# -*- coding: utf-8 -*-
"""End-to-end STEP run on one unlabeled target cohort."""
import numpy as np
import torch

from .bn import reestimate_bn
from .data import make_loader
from .extract import extract
from .readout import RANKS, refine, subspace
from .views import make_noise_banks


@torch.no_grad()
def run_step(model, stage3, stage4, head, target_dir, batch_size=64,
             ranks=RANKS, rounds=3, device="cpu", seed=0, workers=4):
    """Adapt `model` to `target_dir` and return predictions and diagnostics.

    Five image-equivalent forward passes per image in total: one for the
    buffer re-estimation and four for the views.  No label, no gradient and no
    source image enters the procedure.
    """
    loader, ds = make_loader(target_dir, batch_size=batch_size, shuffle=True,
                             workers=workers)
    reestimate_bn(model, loader, momentum=0.1, eps=1e-6, device=device)

    # a second, label-blind pass in evaluation mode for the four views
    eval_loader, _ = make_loader(target_dir, batch_size=batch_size, shuffle=False,
                                 workers=workers)
    banks = make_noise_banks(seed=seed)
    H, T0, L, y = extract(model, eval_loader, stage3, stage4, head, banks, device)

    U = subspace(H.numpy())
    scores, _ = refine(T0.numpy(), U, ranks, rounds)
    pred = scores.argmax(1)

    out = dict(pred=pred, scores=scores, y=y.numpy(), H=H.numpy(), T0=T0.numpy(),
               classes=ds.classes)
    out["acc"] = float((pred == out["y"]).mean() * 100.0)
    return out
