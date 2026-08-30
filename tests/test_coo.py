"""Tests for encode_coo: the direct COO/vectorized-input entry point.

encode_coo shares its core pipeline with encode_sparse (see sketching.py's
_encode_core), so these tests focus on the parts specific to the COO path:
duck-typed input handling, shape-from-.shape (not from the max row seen), and
that it agrees with encode_sparse on an equivalent dict-of-counts input.
"""

from __future__ import annotations

import numpy as np
import pytest

from fpsketch import encode_coo, encode_sparse


class _FakeCoo:
    """Minimal duck-typed COO stand-in -- exercises the contract without a
    scipy dependency."""

    def __init__(self, row, col, data, shape):
        self.row = np.asarray(row)
        self.col = np.asarray(col)
        self.data = np.asarray(data)
        self.shape = shape


def _coo_from_fps(fps, n_features):
    rows = [r for r, fp in enumerate(fps) for _ in fp]
    cols = [feat for fp in fps for feat in fp]
    data = [count for fp in fps for count in fp.values()]
    return _FakeCoo(rows, cols, data, shape=(len(fps), n_features))


def test_encode_coo_matches_encode_sparse_on_equivalent_input():
    fps = [{1: 1, 2: 2, 3: 3}, {4: 4, 5: 5, 6: 6}]
    via_sparse = encode_sparse(fps, dim=256, num_blocks=4, seed=0)
    via_coo = encode_coo(_coo_from_fps(fps, n_features=10), dim=256, num_blocks=4, seed=0)
    assert np.array_equal(via_sparse, via_coo)


def test_encode_coo_shape_comes_from_coo_shape_not_max_row():
    # Row 3 has no entries at all -- shape[0] must still drive n_rows.
    coo = _FakeCoo(row=[0, 0], col=[1, 2], data=[1, 1], shape=(4, 10))
    out = encode_coo(coo, dim=32, seed=0)
    assert out.shape == (4, 32)
    assert np.array_equal(out[3], np.zeros(32))


def test_encode_coo_drops_zero_and_negative_counts():
    coo = _FakeCoo(row=[0, 0, 0], col=[1, 2, 3], data=[0, -3, 2], shape=(1, 10))
    out = encode_coo(coo, dim=32, seed=0)
    equivalent = encode_sparse([{3: 2}], dim=32, seed=0)
    assert np.array_equal(out, equivalent)


def test_encode_coo_rejects_negative_feature_ids():
    coo = _FakeCoo(row=[0], col=[-1], data=[1], shape=(1, 10))
    with pytest.raises(ValueError):
        encode_coo(coo, dim=32, seed=0)


def test_encode_coo_rejects_non_coo_input():
    with pytest.raises(TypeError):
        encode_coo({"not": "a coo array"}, dim=32, seed=0)


def test_encode_coo_matches_encode_sparse_on_scipy_coo_array():
    scipy_sparse = pytest.importorskip("scipy.sparse")

    fps = [{1: 1, 2: 2, 3: 3}, {4: 4, 5: 5, 6: 6}]
    via_sparse = encode_sparse(fps, dim=256, num_blocks=4, seed=0)

    rows = [r for r, fp in enumerate(fps) for _ in fp]
    cols = [feat for fp in fps for feat in fp]
    data = [count for fp in fps for count in fp.values()]
    coo = scipy_sparse.coo_array((data, (rows, cols)), shape=(len(fps), 10))

    via_coo = encode_coo(coo, dim=256, num_blocks=4, seed=0)
    assert np.array_equal(via_sparse, via_coo)
