# -*- coding: utf-8 -*-
"""Dataset helper: an ImageFolder that also returns the row index.

Images are resized on the shorter side to 292 and centre-cropped to 256 x 256,
and intensities are scaled to [0, 1] by `ToTensor`, exactly as in the paper.
"""
import os

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder

MEAN = (0.5, 0.5, 0.5)
STD = (0.5, 0.5, 0.5)


def build_transform(train=False):
    steps = [transforms.Grayscale(num_output_channels=3),
             transforms.Resize(292),
             transforms.CenterCrop(256)]
    if train:
        steps += [transforms.RandomHorizontalFlip()]
    steps += [transforms.ToTensor(),
              transforms.Normalize(MEAN, STD)]
    return transforms.Compose(steps)


class IndexedFolder(Dataset):
    """ImageFolder wrapper whose __getitem__ also returns the sample index."""

    def __init__(self, root, train=False):
        if not os.path.isdir(root):
            raise FileNotFoundError(root)
        self.base = ImageFolder(root, transform=build_transform(train))
        self.classes = self.base.classes
        self.samples = self.base.samples

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        x, y = self.base[i]
        return x, y, i


def make_loader(root, batch_size=64, shuffle=False, train=False, workers=4):
    ds = IndexedFolder(root, train=train)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=workers, pin_memory=torch.cuda.is_available()), ds
