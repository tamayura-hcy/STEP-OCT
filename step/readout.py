# -*- coding: utf-8 -*-
"""Progressive-rank label refinement inside one subspace (Eq. 3).

    S^(t) = U_{d_t} (U_{d_t}^T T^(t-1)),     T^(t) = onehot(argmax S^(t))

One SVD of the standardised descriptor matrix gives every U_d as a prefix, so the
cohort is described once and only the rank changes.  Because P_d' P_d = P_d for
d <= d', composing the projections alone would change nothing; the path matters
only because the labels are reassigned to hard values in between.
"""
import numpy as np

RANKS = (16, 23, 30, 37, 44, 52, 59, 66, 73, 80)
D_MAX = 80


def standardise(H):
    H = np.asarray(H, dtype=np.float64)
    Z = (H - H.mean(0)) / (H.std(0) + 1e-8)
    return Z - Z.mean(0)


def subspace(H, d_max=D_MAX):
    U, _, _ = np.linalg.svd(standardise(H), full_matrices=False)
    return U[:, : min(d_max, U.shape[1])]


def refine(T0, U, ranks=RANKS, rounds=3):
    """Run `rounds` projection+reassignment updates at every rank in `ranks`."""
    T = np.asarray(T0, dtype=np.float64).copy()
    S = T
    for d in ranks:
        Ud = U[:, : min(d, U.shape[1])]
        for _ in range(rounds):
            S = Ud @ (Ud.T @ T)
            T = np.eye(T.shape[1])[S.argmax(1)]
    return S, T


def predict(H, T0, ranks=RANKS, rounds=3):
    U = subspace(H)
    S, _ = refine(T0, U, ranks, rounds)
    return S.argmax(1)
