"""Sparse count-fingerprint sketching.

Hashes each molecule's thermometer-encoded count fingerprint into a fixed-width
dense vector such that ``encode_sparse(...) @ encode_sparse(...).T`` approximates
``T_MM`` (min-max Tanimoto similarity) as a plain dot product.

This is a block-disjoint sparse Johnson-Lindenstrauss sketch (`num_blocks`
independent CountSketches, each writing into its own disjoint slice of the output,
averaged for lower-variance dot products): each unary "rung" of a feature's count
gets its own hashed column and sign, so the thermometer decomposition happens
implicitly inside the hash key rather than as a separate materialized step.

This implementation prioritizes correctness and auditability over speed -- it
hashes one count-rung at a time with `hashlib.blake2b`. A vectorized rewrite is a
known follow-up, not done here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from hashlib import blake2b

import numpy as np


def _hash_tuple(t: tuple) -> int:
    data = repr(t).encode()
    hash_digest = blake2b(data).digest()
    return int.from_bytes(hash_digest, "big")


def encode_sparse(
    fps: Sequence[Mapping[int, int]],
    m: int,
    num_blocks: int = 4,
    seed: int | str | bytes = 0,
) -> np.ndarray:
    """Sketch sparse count fingerprints into a dense ``(len(fps), m)`` array.

    Args:
        fps: one ``{feature_id: count}`` mapping per molecule (e.g. RDKit's
            ``GetSparseCountFingerprint(mol).GetNonzeroElements()``).
        m: output sketch width.
        num_blocks: number of disjoint sparse-JL blocks to split ``m`` into.
            More blocks trade a small amount of dot-product accuracy for
            better tail concentration; the default of 4 is a reasonable
            general-purpose choice.
        seed: hash seed; two sketches are only dot-product-comparable if built
            with the same seed.

    Returns:
        A ``(len(fps), m)`` float64 array. Its rows' pairwise dot products
        approximate ``T_DP``, which equals ``T_MM`` on the original counts.
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

                    col_tuple = ("col", *fp_key)
                    col = _hash_tuple(col_tuple) % block_length

                    sign_tuple = ("sign", *fp_key)
                    sign_01 = _hash_tuple(sign_tuple) % 2
                    sign = 2 * sign_01 - 1

                    out[row, block_start + col] += sign / (num_blocks**0.5)

    return out
