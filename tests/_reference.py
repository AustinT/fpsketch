"""Reference math used by tests only. Not imported by the package itself.

No test_ prefix -- pytest does not collect this file as a test module.
"""

from __future__ import annotations

from collections.abc import Mapping

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
