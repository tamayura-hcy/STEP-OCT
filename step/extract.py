# -*- coding: utf-8 -*-
"""Turning the frozen network into the two matrices STEP needs (Eq. 2).

    h_ik  = 1/4 sum_v max_p F3_{iv}[k, p]        descriptor -> H  (n x 1024)
    L_ip  = 1/4 sum_v g(F4_{iv}[:, p])           logits     -> T0 (n x 3)

Both are read off the same four views in the same forward passes, so row i of H
and row i of T0 describe the same image.  The initial label is the one-hot
encoding of the argmax of the logits at the peak position
p_i* = argmax_p max_c L_ip,c.
"""
import torch
import torch.nn.functional as F

from .views import four_views


class StageTaps:
    """Forward hooks that expose the stage-3 map and the (unpooled) stage-4 map."""

    def __init__(self, model, stage3, stage4):
        self.f3 = None
        self.f4 = None
        self._hooks = [
            stage3.register_forward_hook(lambda m, i, o: setattr(self, "f3", o)),
            stage4.register_forward_hook(lambda m, i, o: setattr(self, "f4", o)),
        ]

    def close(self):
        for h in self._hooks:
            h.remove()


@torch.no_grad()
def _position_logits(head, f4):
    """Apply the frozen head at every one of the 64 stage-4 positions."""
    b, c, h, w = f4.shape
    flat = f4.flatten(2).transpose(1, 2).reshape(b * h * w, c)
    out = head(flat)
    return out.reshape(b, h * w, -1)


@torch.no_grad()
def extract(model, loader, stage3, stage4, head, banks, device="cpu"):
    """Run the frozen model over `loader`.

    Returns
    -------
    H  : (n, 1024)  descriptor matrix
    T0 : (n, 3)     one-hot initial labels
    L  : (n, 64, 3) per-position logits averaged over the four views
    y  : (n,)       ground truth, for scoring only
    """
    taps = StageTaps(model, stage3, stage4)
    per_view_H = [[] for _ in range(4)]
    per_view_L = [[] for _ in range(4)]
    ys = []
    model.eval()
    try:
        for batch in loader:
            x, y = batch[0].to(device), batch[1]
            index = batch[2] if len(batch) > 2 else torch.arange(len(x))
            for v, xv in enumerate(four_views(x, banks, index)):
                model(xv)
                f3, f4 = taps.f3, taps.f4               # (B,1024,16,16) / (B,2048,8,8)
                per_view_H[v].append(f3.flatten(2).max(dim=2).values.double())
                per_view_L[v].append(_position_logits(head, f4).double())
            ys.append(y)
    finally:
        taps.close()

    H = sum(torch.cat(per_view_H[v]) for v in range(4)) / 4.0
    L = sum(torch.cat(per_view_L[v]) for v in range(4)) / 4.0
    y = torch.cat(ys)

    peak_pos = L.max(dim=2).values.argmax(dim=1)
    t0 = L[torch.arange(len(L)), peak_pos].argmax(dim=1)
    T0 = F.one_hot(t0, num_classes=L.shape[2]).double()
    return H, T0, L, y
