"""Guarantee (1): the sketch preserves inner products, the fundamental CountSketch
guarantee this package's approximation of T_MM rests on. Binary (count<=1)
fingerprints are the simplest case: T_DP(sketch(x), sketch(x')) targets the exact
set-intersection dot product, so ground truth is trivial to compute by hand.
"""

from __future__ import annotations

import numpy as np
from _reference import t_dp, t_mm_dict

from fpsketch import encode_sparse


def test_binary_dot_product_unbiased_over_seeds():
    # Two sets of 30 features each, overlapping in exactly 10 -- known dot product.
    a_features = range(0, 30)
    b_features = range(20, 50)
    fps = [{i: 1 for i in a_features}, {i: 1 for i in b_features}]
    true_overlap = len(set(a_features) & set(b_features))
    assert true_overlap == 10

    dots = []
    for seed in range(200):
        out = encode_sparse(fps, dim=64, seed=seed)
        dots.append(float(out[0] @ out[1]))

    mean_dot = np.mean(dots)
    assert abs(mean_dot - true_overlap) < 0.06 * true_overlap


def test_binary_pairwise_mae():
    rng = np.random.default_rng(0)
    d, n, nnz = 2000, 50, 30
    fps = [{int(i): 1 for i in rng.choice(d, size=nnz, replace=False)} for _ in range(n)]

    sketch = encode_sparse(fps, dim=4096, num_blocks=4, seed=0)
    k_hat = t_dp(sketch)

    k_true = np.array([[t_mm_dict(a, b) for b in fps] for a in fps])

    iu = np.triu_indices(n, k=1)
    mae = np.mean(np.abs(k_hat[iu] - k_true[iu]))
    assert mae < 0.01
