# -*- coding: utf-8 -*-
"""Re-estimating the encoder's normalisation buffers on the target cohort.

Each label-blind shuffled minibatch is passed once through the encoder with its
batch-norm layers in training mode, which updates the running means and
variances while the affine parameters stay fixed.  The buffers carry over from
one minibatch to the next, so at the end of the sweep every activation is in the
target cohort's units.  This pass produces no descriptor and is discarded.
"""
import torch
from torch import nn


def _bn_modules(model):
    return [m for m in model.modules() if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d))]


@torch.no_grad()
def reestimate_bn(model, loader, momentum=0.1, eps=1e-6, device="cpu"):
    """Update every BN buffer of `model` on the unlabeled `loader`."""
    bns = _bn_modules(model)
    for m in bns:
        m.momentum = momentum
        m.eps = eps
        m.reset_running_stats()
        m.track_running_stats = True

    was_training = model.training
    model.train()
    for m in bns:
        m.train()
    for x, *_ in loader:
        model(x.to(device))
    if not was_training:
        model.eval()
    return model


@torch.no_grad()
def frozen_forward(model, x):
    """Evaluation-mode forward, weights and buffers untouched."""
    was = model.training
    model.eval()
    out = model(x)
    if was:
        model.train()
    return out
