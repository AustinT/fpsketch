"""Guarantee (0): the vectorized splitmix64 rewrite still does the same
statistical job as the original one-hash-per-element ``hashlib.blake2b``
implementation this package shipped with first.
Agreement is checked distributionally over many seeds -- mean dot product and
squared norms -- never bitwise, since the hash function itself changed by design.
"""

from __future__ import annotations

import numpy as np
from _reference import encode_sparse_hashlib

from fpsketch import encode_sparse

M = 256
NUM_BLOCKS = 4
SEEDS = range(60)

# Two count fingerprints with a known overlap: features 20..29 (10 of them)
# are shared, at counts 2 vs 3, so the true unary dot product (S_min) is
# 10 * min(2, 3) = 20 -- a nonzero anchor to measure agreement against,
# unlike a random/uncorrelated pair whose true dot product can land near
# zero and make relative tolerances meaningless.
_FPS = [
    {i: 2 for i in range(0, 30)},
    {i: 3 for i in range(20, 50)},
]
_TRUE_DOT = 20.0


def test_matches_hashlib_reference_distributionally():
    new_dots, ref_dots = [], []
    new_sq, ref_sq = [], []
    for seed in SEEDS:
        new = encode_sparse(_FPS, dim=M, num_blocks=NUM_BLOCKS, seed=seed)
        ref = encode_sparse_hashlib(_FPS, m=M, num_blocks=NUM_BLOCKS, seed=seed)

        new_dots.append(float(new[0] @ new[1]))
        ref_dots.append(float(ref[0] @ ref[1]))
        new_sq.append(float((new**2).sum()))
        ref_sq.append(float((ref**2).sum()))

    mean_new_dot, mean_ref_dot = np.mean(new_dots), np.mean(ref_dots)
    assert abs(mean_new_dot - _TRUE_DOT) < 0.1 * _TRUE_DOT
    assert abs(mean_ref_dot - _TRUE_DOT) < 0.1 * _TRUE_DOT
    assert abs(mean_new_dot - mean_ref_dot) < 0.1 * _TRUE_DOT

    mean_new_sq, mean_ref_sq = np.mean(new_sq), np.mean(ref_sq)
    assert abs(mean_new_sq - mean_ref_sq) < 0.1 * mean_ref_sq
