import numpy as np
import pytest

from fpsketch._splitmix import GAMMA, _splitmix64_hash
from fpsketch.sketching import _subkeys, _unary_triples


def test_subkeys_returns_distinct_uint64_values():
    keys = _subkeys(0, 4)
    assert keys.dtype == np.uint64
    assert len(keys) == 4
    assert len(set(keys.tolist())) == 4


def test_subkeys_large_n_does_not_raise():
    # num_blocks=8 needs 2 * 8 = 16 subkeys; blake2b would raise here.
    keys = _subkeys(0, 16)
    assert keys.dtype == np.uint64
    assert len(keys) == 16


def test_subkeys_different_seeds_share_no_values():
    a = set(_subkeys(0, 8).tolist())
    b = set(_subkeys(1, 8).tolist())
    assert a.isdisjoint(b)


def test_subkeys_negative_int_seed_works():
    keys = _subkeys(-1, 2)
    assert keys.dtype == np.uint64
    assert len(keys) == 2


def test_subkeys_deterministic():
    a = _subkeys(0, 4)
    b = _subkeys(0, 4)
    assert np.array_equal(a, b)


# Pinned splitmix64 vectors, verified against an independent pure-Python
# implementation. mix(0) == 0xE220A8397B1DCDAF also matches the well-known
# splitmix64 first output for state 0, so it double-checks against the
# literature too.
@pytest.mark.parametrize(
    "x, expected",
    [
        (0, 16294208416658607535),
        (1, 10451216379200822465),
        (2, 10905525725756348110),
        (2**64 - 1, 16490336266968443936),
    ],
)
def test_splitmix64_hash_matches_pinned_vectors(x, expected):
    result = _splitmix64_hash(np.array([x], dtype=np.uint64))
    assert result[0] == np.uint64(expected)


def test_splitmix64_hash_dtype_is_uint64():
    result = _splitmix64_hash(np.array([0, 1, 2], dtype=np.uint64))
    assert result.dtype == np.uint64


def test_unary_triples_empty_input_gives_empty_arrays():
    rows, ids, levels = _unary_triples([])
    assert len(rows) == len(ids) == len(levels) == 0
    assert rows.dtype == np.int64
    assert ids.dtype == np.uint64
    assert levels.dtype == np.uint64


def test_unary_triples_all_empty_fingerprints_gives_empty_arrays():
    rows, ids, levels = _unary_triples([{}, {}])
    assert len(rows) == len(ids) == len(levels) == 0


def test_unary_triples_drops_zero_and_negative_counts():
    rows, ids, levels = _unary_triples([{1: 0, 2: -3, 3: 2}])
    assert rows.tolist() == [0, 0]
    assert ids.tolist() == [3, 3]
    assert levels.tolist() == [0, 1]


def test_unary_triples_dtype_and_shape():
    rows, ids, levels = _unary_triples([{7: 2, 9: 1}, {}, {4: 3}])
    assert rows.dtype == np.int64
    assert ids.dtype == np.uint64
    assert levels.dtype == np.uint64
    assert len(rows) == len(ids) == len(levels) == 6


def test_hash_quality_buckets_and_signs_well_mixed():
    """Smoke test for the block hashing pipeline (splitmix64 -> mod
    block_length for the bucket, splitmix64 -> top bit for the sign).
    PLAN-vectorized-sketch.md flags a silent float64 promotion in the mixer
    as the most likely bug in this rewrite; that would badly skew both the
    bucket histogram and the sign distribution well before any accuracy test
    would notice, so this checks both directly against ~200k synthetic keys.
    """
    n = 200_000
    block_length = 4096
    ids = np.arange(n, dtype=np.uint64)
    levels = np.zeros(n, dtype=np.uint64)

    with np.errstate(over="ignore"):
        fp_keys = _splitmix64_hash(ids) + levels * GAMMA

    col_key, sign_key = _subkeys(seed=0, n=2)
    col_hash = _splitmix64_hash(fp_keys ^ col_key)
    sign_hash = _splitmix64_hash(fp_keys ^ sign_key)

    bucket = (col_hash % np.uint64(block_length)).astype(np.int64)
    sign = 2.0 * ((sign_hash >> np.uint64(63)) & np.uint64(1)).astype(np.float64) - 1.0

    counts = np.bincount(bucket, minlength=block_length)
    expected = n / block_length
    chi2_per_dof = np.sum((counts - expected) ** 2 / expected) / (block_length - 1)
    assert 0.8 < chi2_per_dof < 1.2

    assert abs(sign.mean()) < 0.02

    corr = np.corrcoef(sign, bucket % 2)[0, 1]
    assert abs(corr) < 0.02
