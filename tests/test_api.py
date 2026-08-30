import sys

import numpy as np
import pytest

from fpsketch import encode_sparse


def test_shape_and_l1_norm_matches_count_mass():
    fps = [
        {1: 1, 2: 2, 3: 3},
        {4: 4, 5: 5, 6: 6},
    ]
    m = 512
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


def test_output_dtype_is_float64():
    out = encode_sparse([{1: 1, 2: 2}], m=16, seed=0)
    assert out.dtype == np.float64


def test_non_divisor_num_blocks_gives_correct_shape_and_norms():
    # m=1000, num_blocks=3 -> block lengths 333/333/334. Catches any
    # accidental power-of-two or exact-divisibility assumption in the
    # block-boundary arithmetic (see PLAN-vectorized-sketch.md Step D).
    fps = [{1: 1, 2: 2, 3: 3}, {4: 4, 5: 5, 6: 6}]
    out = encode_sparse(fps, m=1000, num_blocks=3, seed=0)
    assert out.shape == (2, 1000)

    sq_norms = (out**2).sum(axis=1)
    assert np.allclose(sq_norms, [sum(fp.values()) for fp in fps], rtol=0.1)


def test_large_feature_ids_supported():
    # Ids near 2**32 and 2**63 -- there is no bit-packing in this
    # implementation, so nothing should cap out (unlike the reference branch
    # noted in PLAN-vectorized-sketch.md, which packed level bits into the id
    # and so capped ids at 2**56).
    fps = [{2**32 + 5: 2, 2**63 + 7: 3}]
    out = encode_sparse(fps, m=64, seed=0)
    assert out.shape == (1, 64)
    assert np.isfinite(out).all()
    # Loose sanity bound, not a variance check (nnz=5 is too small a sample
    # for that) -- this just confirms the huge ids contributed real mass
    # rather than silently collapsing to zero or blowing up.
    assert 1.0 < (out**2).sum() < 15.0


def test_encode_sparse_matches_pinned_golden_output():
    # A byte-for-byte regression pin (invariant 7 in PLAN-vectorized-sketch.md:
    # same seed -> identical output, across processes and platforms). num_blocks=4
    # keeps 1/sqrt(num_blocks) == 0.5 exactly representable, so exact equality
    # is meaningful here rather than a source of float flakiness. If this ever
    # legitimately needs to change (e.g. HASH_VERSION bump), regenerate it
    # deliberately -- don't just update it to make a failure go away.
    fps = [{1: 1, 2: 2, 3: 3}, {4: 1}]
    out = encode_sparse(fps, m=12, num_blocks=4, seed=42)
    expected = np.array(
        [
            [0.5, -0.5, 0.0, -0.5, 1.0, 0.5, 0.0, 1.0, 0.0, 1.0, 0.5, 0.5],
            [-0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5, 0.0, -0.5, 0.0],
        ]
    )
    assert np.array_equal(out, expected)


def test_encode_mols_missing_rdkit_raises_helpful_import_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "rdkit", None)
    monkeypatch.setitem(sys.modules, "rdkit.Chem", None)
    monkeypatch.setitem(sys.modules, "rdkit.Chem.rdFingerprintGenerator", None)

    from fpsketch import encode_mols

    with pytest.raises(ImportError, match="fpsketch\\[chem\\]"):
        encode_mols([object()], m=16)
