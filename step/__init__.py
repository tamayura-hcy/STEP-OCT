# -*- coding: utf-8 -*-
"""STEP - Staged Target-Label Estimation by Projection.

Source-free, gradient-free adaptation of a frozen OCT classifier to a new
scanner.  See README.md for the three commands that reproduce a direction.
"""
from .bn import reestimate_bn
from .equalise import cohort_tau, equalise, local_contrast
from .extract import extract
from .readout import RANKS, predict, refine, subspace
from .views import four_views, make_noise_banks

__all__ = [
    "reestimate_bn",
    "cohort_tau",
    "equalise",
    "local_contrast",
    "extract",
    "RANKS",
    "predict",
    "refine",
    "subspace",
    "four_views",
    "make_noise_banks",
]
__version__ = "1.0.0"
