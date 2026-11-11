# -*- coding: utf-8 -*-
"""The four views the frozen encoder sees (Sec. III-B of the paper).

Two views carry the contrast correction, with two different window widths, and
two are the raw image plus a fixed Gaussian realisation taken from one of two
banks.  Exactly half of what the encoder sees is therefore corrected and half is
not, so a single correction is never the only evidence the readout has.
"""
import torch

from .equalise import cohort_tau, equalise

# (window, tau rule) for the two corrected views.  The first tau is measured per
# image on a 3x3 contrast; the second is the fixed 0.24 multiple on a 5x5 window.
VIEW_SPEC = ((3, "cohort", 0.30), (5, "fixed", 0.24))
NOISE_SIGMA = 0.007


def make_noise_banks(n_banks=2, size=64, seed=0):
    """Two fixed banks of Gaussian tensors, indexed by image index modulo `size`."""
    banks = []
    for b in range(n_banks):
        g = torch.Generator().manual_seed(seed + 1000 * (b + 1))
        banks.append([torch.empty(1, 3, size, size).normal_(generator=g) for _ in range(size)])
    return banks


def _noise_view(x, bank, index):
    rows = []
    for i, img in enumerate(x):
        eta = bank[int(index) % len(bank)]
        eta = eta[:, : img.shape[0]].to(device=x.device, dtype=x.dtype)
        rows.append(torch.clamp(img + NOISE_SIGMA * eta, 0.0, 1.0))
    return torch.stack(rows)


def four_views(x, banks, index):
    """Return the four views of the batch `x` (B, C, H, W) as a list of tensors."""
    v1 = equalise(x, cohort_tau(x, VIEW_SPEC[0][0]), w=VIEW_SPEC[0][0])
    v2 = equalise(x, VIEW_SPEC[1][2], w=VIEW_SPEC[1][0])
    v3 = _noise_view(x, banks[0], index)
    v4 = _noise_view(x, banks[1], index)
    return [v1, v2, v3, v4]
