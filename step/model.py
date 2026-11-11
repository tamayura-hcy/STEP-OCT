# -*- coding: utf-8 -*-
"""Backbone + head used throughout the paper.

ResNet-50 encoder, stage-3 = `layer3` (1024 x 16 x 16) and stage-4 = `layer4`
(2048 x 8 x 8) before global pooling, with a two-1024-unit ReLU head on top.
Everything here is loaded from a checkpoint and then frozen: STEP never updates
a weight, only the batch-norm buffers and the target label matrix.
"""
import torch
from torch import nn


class OCTNet(nn.Module):
    def __init__(self, num_classes=3, width=1024):
        super().__init__()
        from torchvision.models import resnet50

        body = resnet50(weights=None)
        self.stem = nn.Sequential(body.conv1, body.bn1, body.relu, body.maxpool,
                                  body.layer1, body.layer2)
        self.layer3 = body.layer3
        self.layer4 = body.layer4
        self.head = nn.Sequential(
            nn.Linear(2048, width), nn.ReLU(inplace=True),
            nn.Linear(width, width), nn.ReLU(inplace=True),
            nn.Linear(width, num_classes),
        )

    def forward(self, x):
        f = self.layer4(self.layer3(self.stem(x)))
        f = torch.nn.functional.adaptive_avg_pool2d(f, 1).flatten(1)
        return self.head(f)


def build_model(ckpt=None, num_classes=3, map_location="cpu"):
    model = OCTNet(num_classes=num_classes)
    if ckpt:
        state = torch.load(ckpt, map_location=map_location)
        state = state.get("state_dict", state.get("model", state))
        state = {k.replace("module.", ""): v for k, v in state.items()}
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            print("[warn] checkpoint keys  missing=%d unexpected=%d"
                  % (len(missing), len(unexpected)))
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    return model
