# -*- coding: utf-8 -*-
"""Local contrast equalisation (Eq. 1 of the paper).

    c_w      = sqrt(max(m_w - mu_w^2, 0)) / (mu_w + eps)
    a_{w,tau}= clip(tau / (c_w + eps), 0.2, 5)
    Q_{w,tau}(x) = clip(mu_w + a_{w,tau} * (x - mu_w), 0, 1)

`mu_w` and `m_w` are the channel-wise window means of `x` and `x^2` on a w x w
grid.  The gain is applied to each pixel's deviation from its own local mean, so
the operator acts on residuals and not on the image as a whole.  `tau` is never
fitted: it is a fixed multiple of a contrast measured on the target cohort.
"""
import torch
import torch.nn.functional as F


def window_moments(x, w):
    """Channel-wise window mean of x and x^2 (stride 1, zero padding included)."""
    k = torch.ones(1, 1, w, w, dtype=x.dtype, device=x.device) / (w * w)
    c = x.shape[1]
    mu = F.conv2d(x, k.expand(c, 1, w, w), padding=w // 2, groups=c)
    m2 = F.conv2d(x * x, k.expand(c, 1, w, w), padding=w // 2, groups=c)
    return mu, m2


def local_contrast(x, w=3, eps=1e-6):
    """Local coefficient of variation c_w of every pixel."""
    mu, m2 = window_moments(x, w)
    return torch.sqrt(torch.clamp(m2 - mu * mu, min=0.0)) / (mu + eps)


def cohort_tau(x, w=3, eps=1e-6, multiplier=0.30):
    """tau_i = multiplier * channel-average of the per-image spatial median of c_w.

    Measured on the unlabeled target cohort, one number per image, so the
    operator moves with the cohort instead of being tuned on it.
    """
    c = local_contrast(x, w, eps)
    med = c.flatten(2).median(dim=2).values.mean(dim=1)
    return multiplier * med


def equalise(x, tau, w=3, eps=1e-6, lo=0.2, hi=5.0):
    """Apply Q_{w,tau} to a batch.  `tau` is a scalar or a (B,) tensor."""
    mu, _ = window_moments(x, w)
    c = local_contrast(x, w, eps)
    if not torch.is_tensor(tau):
        tau = torch.full((x.shape[0],), float(tau), dtype=x.dtype, device=x.device)
    gain = torch.clamp(tau.view(-1, 1, 1, 1) / (c + eps), lo, hi)
    return torch.clamp(mu + gain * (x - mu), 0.0, 1.0)
