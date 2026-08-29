import sys

import numpy as np
import pytest

from fpsketch import encode_sparse


def test_shape_and_l1_norm_matches_count_mass():
    fps = [
        {1: 1, 2: 2, 3: 3},
        {4: 4, 5: 5, 6: 6},
    ]
    m = 128
    out = encode_sparse(fps, m=m, seed=0)
    assert out.shape == (len(fps), m)

    sq_norms = (out**2).sum(axis=1)
    assert np.allclose(sq_norms, [sum(fp.values()) for fp in fps], rtol=0.1)


def test_deterministic_given_same_seed():
    fps = [{1: 1, 2: 2, 3: 3}]
    a = encode_sparse(fps, m=64, seed=42)
    b = encode_sparse(fps, m=64, seed=42)
    assert np.array_equal(a, b)


def test_different_seeds_give_different_output():
    fps = [{1: 1, 2: 2, 3: 3}]
    a = encode_sparse(fps, m=64, seed=0)
    b = encode_sparse(fps, m=64, seed=1)
    assert not np.array_equal(a, b)


def test_num_blocks_exposed_and_respected():
    fps = [{1: 1, 2: 2, 3: 3}]
    a = encode_sparse(fps, m=64, num_blocks=1, seed=0)
    b = encode_sparse(fps, m=64, num_blocks=8, seed=0)
    assert a.shape == b.shape == (1, 64)
    assert not np.array_equal(a, b)


def test_num_blocks_greater_than_m_raises():
    with pytest.raises(ValueError):
        encode_sparse([{1: 1}], m=4, num_blocks=8, seed=0)


def test_empty_fingerprint_gives_zero_row():
    out = encode_sparse([{}], m=16, seed=0)
    assert out.shape == (1, 16)
    assert np.array_equal(out, np.zeros((1, 16)))


def test_encode_mols_missing_rdkit_raises_helpful_import_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "rdkit", None)
    monkeypatch.setitem(sys.modules, "rdkit.Chem", None)
    monkeypatch.setitem(sys.modules, "rdkit.Chem.rdFingerprintGenerator", None)

    from fpsketch import encode_mols

    with pytest.raises(ImportError, match="fpsketch\\[chem\\]"):
        encode_mols([object()], m=16)
