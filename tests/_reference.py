"""Reference math used by tests only. Not imported by the package itself.

No test_ prefix -- pytest does not collect this file as a test module.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from hashlib import blake2b

import numpy as np

EPS = 1e-12


def t_mm_dict(a: Mapping[int, int], b: Mapping[int, int]) -> float:
    """Exact min-max Tanimoto (T_MM) between two sparse count dicts."""
    keys = set(a) | set(b)
    s_min = sum(min(a.get(k, 0), b.get(k, 0)) for k in keys)
    s_max = sum(max(a.get(k, 0), b.get(k, 0)) for k in keys)
    return s_min / s_max if s_max else 0.0


def t_dp(X: np.ndarray) -> np.ndarray:
    """Pairwise dot-product Tanimoto (T_DP) for all rows of a dense array."""
    G = X @ X.T
    sq = np.einsum("ij,ij->i", X, X)
    return G / np.maximum(sq[:, None] + sq[None, :] - G, EPS)


def _hash_tuple_blake2b(t: tuple) -> int:
    return int.from_bytes(blake2b(repr(t).encode()).digest(), "big")


def encode_sparse_hashlib(
    fps: Sequence[Mapping[int, int]],
    m: int,
    num_blocks: int = 4,
    seed: int = 0,
) -> np.ndarray:
    """The original one-hash-per-element ``hashlib.blake2b`` implementation of
    ``encode_sparse``, kept only as a trusted reference. The vectorized
    splitmix64 implementation uses a different hash function by design (see
    ``PLAN-vectorized-sketch.md``), so this is compared against distributionally
    (mean dot product, squared norms), never bitwise.
    """
    if num_blocks > m:
        raise ValueError("Must have m >= num_blocks")

    standard_block_length = m // num_blocks
    out = np.zeros((len(fps), m))
    for row, fp in enumerate(fps):
        for block_idx in range(num_blocks):
            block_start = block_idx * standard_block_length
            block_length = standard_block_length
            if block_idx == num_blocks - 1:
                block_length += m % num_blocks
            for feature_idx, feature_count in fp.items():
                for count in range(feature_count):
                    fp_key = (seed, block_idx, feature_idx, count)
                    col = _hash_tuple_blake2b(("col", *fp_key)) % block_length
                    sign = 2 * (_hash_tuple_blake2b(("sign", *fp_key)) % 2) - 1
                    out[row, block_start + col] += sign / (num_blocks**0.5)

    return out
