"""Sparse count-fingerprint sketching.

Hashes each molecule's unary-encoded count fingerprint into a fixed-width
dense vector such that ``encode_sparse(...) @ encode_sparse(...).T`` approximates
``T_MM`` (min-max Tanimoto similarity) as a plain dot product.

This is a block-disjoint sparse Johnson-Lindenstrauss sketch (`num_blocks`
independent CountSketches, each writing into its own disjoint slice of the output,
averaged for lower-variance dot products): each unary level of a feature's count
gets its own hashed column and sign, so the unary decomposition happens
implicitly inside the hash key rather than as a separate materialized step.

Two entry points share this pipeline: ``encode_sparse`` takes one
``{feature_id: count}`` dict per molecule, while ``encode_coo`` takes an
already-vectorized COO-format sparse array (e.g. ``scipy.sparse.coo_array``,
or anything exposing ``.row``/``.col``/``.data``/``.shape``) directly, skipping
the per-molecule dict traversal for callers who already have their counts in
that form.

Hashing is vectorized: `hashlib` is used only once per call, in `_subkeys`, to
turn the user's seed into a handful of subkeys. Per-element hashing is done with
a pure-numpy splitmix64 mixer instead of `hashlib` per element.
"""

from __future__ import annotations

import hashlib
import struct
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from ._splitmix import GAMMA as SPLITMIX64_GAMMA
from ._splitmix import _splitmix64_hash

U64 = np.uint64

_EMPTY_COO = (
    np.empty(0, dtype=np.int64),
    np.empty(0, dtype=U64),
    np.empty(0, dtype=np.int64),
)
_EMPTY_UNARY = (
    np.empty(0, dtype=np.int64),
    np.empty(0, dtype=U64),
    np.empty(0, dtype=U64),
)


def _coo_from_fps(
    fps: Sequence[Mapping[int, int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flatten sparse count fingerprints into parallel COO ``(row, id, count)`` arrays.

    This is the same ``(row, col, data)`` shape as a COO sparse matrix of
    molecules x features, just not wrapped in a sparse-array class. Callers who
    already have that shape (e.g. a ``scipy.sparse.coo_array``) can skip this
    function entirely and go straight to ``encode_coo``.

    Args:
        fps: one ``{feature_id: count}`` mapping per molecule. Feature ids must be
            non-negative; entries with a count of zero or less are dropped.

    Returns:
        Three parallel 1-D arrays, each of length ``nnz`` (the number of
        surviving, positive-count entries):

        - ``rows`` (int64): which molecule the entry came from, an index into ``fps``.
        - ``ids`` (uint64): which feature the entry came from.
        - ``counts`` (int64): the (positive) count for that ``(row, id)`` pair.

    Example:
        >>> rows, ids, counts = _coo_from_fps([{7: 2, 9: 1}, {}, {4: 3}])
        >>> rows
        array([0, 0, 2])
        >>> ids
        array([7, 9, 4], dtype=uint64)
        >>> counts
        array([2, 1, 3])
    """
    row_chunks, feat_chunks, count_chunks = [], [], []
    for row, fp in enumerate(fps):
        if not fp:
            continue
        feat_chunks.append(np.fromiter(fp.keys(), dtype=U64, count=len(fp)))
        count_chunks.append(np.fromiter(fp.values(), dtype=np.int64, count=len(fp)))
        row_chunks.append(np.full(len(fp), row, dtype=np.int64))

    if not feat_chunks:
        return _EMPTY_COO

    ids = np.concatenate(feat_chunks)
    counts = np.concatenate(count_chunks)
    rows = np.concatenate(row_chunks)

    keep = counts > 0
    return rows[keep], ids[keep], counts[keep]


def _unary_expand(
    rows: np.ndarray, ids: np.ndarray, counts: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Expand COO ``(row, id, count)`` triples into flat unary ``(row, id, level)`` triples.

    The unary encoding represents a count ``c`` by the indicators
    ``1[c > 0], 1[c > 1], ..., 1[c > c - 1]`` -- i.e. ``c`` levels, numbered
    ``0`` to ``c - 1``. This function materializes one entry per level, but
    only as three flat index arrays: the unary expansion's own (enormous)
    dimension is never allocated, and the arrays are exactly the input a
    vectorized CountSketch needs.

    Args:
        rows, ids, counts: parallel COO arrays as returned by ``_coo_from_fps``
            (or built directly from a caller's sparse array in ``encode_coo``).
            ``counts`` must already be strictly positive -- callers are
            responsible for dropping zero/negative entries before this point.

    Returns:
        Three parallel 1-D arrays, each of length ``nnz`` (the total count mass,
        ``counts.sum()``):

        - ``out_rows`` (int64): which molecule the level came from.
        - ``out_ids`` (uint64): which feature the level came from.
        - ``levels`` (uint64): which level this is within its feature, ``0 .. count - 1``.

    Example:
        >>> rows, ids, levels = _unary_expand(
        ...     np.array([0, 0, 2]), np.array([7, 9, 4], dtype=np.uint64), np.array([2, 1, 3])
        ... )
        >>> rows
        array([0, 0, 0, 2, 2, 2])
        >>> ids
        array([7, 7, 9, 4, 4, 4], dtype=uint64)
        >>> levels
        array([0, 1, 0, 0, 1, 2], dtype=uint64)
    """
    if not counts.size:
        return _EMPTY_UNARY

    # One entry per level: feature f with count c is repeated c times.
    out_rows = np.repeat(rows, counts)
    out_ids = np.repeat(ids, counts)

    # Levels are a "ragged arange": 0..c-1 restarting at every feature. Globally
    # that is arange(nnz) minus the position at which each entry's run began.
    run_starts = np.repeat(np.cumsum(counts) - counts, counts)
    levels = np.arange(len(out_rows), dtype=np.int64) - run_starts

    return out_rows, out_ids, levels.astype(U64)


def _subkeys(seed: int, n: int) -> np.ndarray:
    """Derive ``n`` independent 64 bit ints from a single seed.

    Done by hashing the seed interpreting the hash digest as a stream of ints.
    The shake_256 hash is used because it can produce an arbitrarily long digest.

    Args:
        seed: the user's seed (int)
        n: number of uint64 subkeys to derive.

    Returns:
        A ``(n,)`` uint64 array.

    Example:
        >>> keys = _subkeys(0, 4)
        >>> keys.dtype
        dtype('uint64')
        >>> len(keys) == len(set(keys.tolist()))
        True
    """
    seed_as_bytes = struct.pack("<q", seed)
    digest = hashlib.shake_256(seed_as_bytes).digest(8 * n)
    return np.frombuffer(digest, dtype="<u8").copy()


def encode_sparse(
    fps: Sequence[Mapping[int, int]],
    dim: int,
    num_blocks: int = 4,
    seed: int = 0,
    scale: bool = True,
) -> np.ndarray:
    """Sketch sparse count fingerprints into a dense ``(len(fps), dim)`` array.

    Args:
        fps: one ``{feature_id: count}`` mapping per molecule (e.g. RDKit's
            ``GetSparseCountFingerprint(mol).GetNonzeroElements()``).
        dim: output sketch width.
        num_blocks: number of disjoint sparse-JL blocks to split ``dim`` into.
            More blocks trade a small amount of dot-product accuracy for
            better tail concentration; the default of 4 is a reasonable
            general-purpose choice.
        seed: hash seed; two sketches are only dot-product-comparable if built
            with the same seed.
        scale: if True (default), divide the output by ``sqrt(num_blocks)`` so
            that ``sketch @ sketch.T`` and ``(sketch**2).sum(axis=1)`` directly
            approximate the true (unnormalized) dot product / count mass of the
            original unary encoding. This factor cancels out of the ``T_DP``
            ratio (``s.s' / (||s||^2 + ||s'||^2 - s.s')``), so callers who only
            ever compute Tanimoto similarity through that ratio can safely set
            this to False and skip the extra pass over the output array; anyone
            comparing raw dot products/norms, including across different
            ``num_blocks`` settings, should leave it True.

    Returns:
        A ``(len(fps), dim)`` float64 array. Its rows' pairwise dot products
        approximate ``T_DP``, which equals ``T_MM`` on the original counts.
    """
    rows, ids, counts = _coo_from_fps(fps)
    return _encode_core(rows, ids, counts, len(fps), dim, num_blocks, seed, scale)


def encode_coo(
    coo: Any,
    dim: int,
    num_blocks: int = 4,
    seed: int = 0,
    scale: bool = True,
) -> np.ndarray:
    """Sketch an already-vectorized COO sparse count array into ``(n_rows, dim)``.

    This is the same computation as ``encode_sparse``, but for callers who
    already have their molecules x features counts in COO form (e.g. a
    ``scipy.sparse.coo_array``) and want to skip the per-molecule dict
    traversal ``encode_sparse`` does internally to get there.

    Args:
        coo: a COO-format sparse array-like object exposing ``.row`` (molecule
            index per entry), ``.col`` (feature id per entry), ``.data`` (count
            per entry), and ``.shape`` (``(n_rows, n_features)``) -- e.g. a
            ``scipy.sparse.coo_array``. Feature ids (``.col``) must be
            non-negative; entries with a count of zero or less are dropped.
        dim: output sketch width.
        num_blocks: see ``encode_sparse``.
        seed: see ``encode_sparse``.
        scale: see ``encode_sparse``.

    Returns:
        A ``(coo.shape[0], dim)`` float64 array, matching ``encode_sparse``'s
        output for the equivalent dict-of-counts input.
    """
    for attr in ("row", "col", "data", "shape"):
        if not hasattr(coo, attr):
            raise TypeError(
                "encode_coo expects a COO-format sparse array-like object "
                "(e.g. scipy.sparse.coo_array) exposing .row, .col, .data, and "
                f".shape; got {type(coo).__name__!r}, missing '{attr}'."
            )

    n_rows = coo.shape[0]
    rows = np.asarray(coo.row, dtype=np.int64)
    ids = np.asarray(coo.col)
    counts = np.asarray(coo.data, dtype=np.int64)

    if ids.size and ids.min() < 0:
        raise ValueError("encode_coo: column indices (feature ids) must be non-negative.")
    ids = ids.astype(U64)

    keep = counts > 0
    rows, ids, counts = rows[keep], ids[keep], counts[keep]

    return _encode_core(rows, ids, counts, n_rows, dim, num_blocks, seed, scale)


def _encode_core(
    rows: np.ndarray,
    ids: np.ndarray,
    counts: np.ndarray,
    n_rows: int,
    dim: int,
    num_blocks: int,
    seed: int,
    scale: bool,
) -> np.ndarray:
    """Shared sketch pipeline for ``encode_sparse`` and ``encode_coo``.

    NOTE: for speed this algorithm implements custom hashing in numpy instead
    of using python's standard library (e.g. hashlib). The motivation for each
    step is explained inline in the comments.

    Args:
        rows, ids, counts: parallel COO arrays (molecule index, feature id,
            positive count), as returned by ``_coo_from_fps`` or built directly
            in ``encode_coo``.
        n_rows: number of molecules (the output's first dimension).
        dim, num_blocks, seed, scale: see ``encode_sparse``.
    """
    if num_blocks > dim:
        raise ValueError("Must have dim >= num_blocks")

    # Step 0: Transform data to vectorized format.
    # These are 1D arrays; shape is nnz in the dataset.
    # Use to initialize the output array
    rows, feature_ids, count_levels = _unary_expand(rows, ids, counts)
    out = np.zeros((n_rows, dim))

    # Hash step 1: create a single 64 bit hash of the
    # (feature_ids, count_levels) tuples. We call it "fp_keys".
    # It gets combined with another hash of the (seed, function)
    # tuple later.
    #
    # Step 1a: hash feature_ids only with splitmix. There is no
    # guarantee that feature_ids are uniformly distributed across
    # 64 bits (they could be a hand constructed fingerprint with ids 0, 1, 2, etc).
    # We apply splitmix once to hopefully disperse them uniformly
    fp_keys = _splitmix64_hash(feature_ids)  # maps feature_ids

    # Step 1b: add an offset from count_levels. The scaling constant
    # SPLITMIX64_GAMMA is a canonical shift constant from the splitmix
    # algorithm, designed to cycle through all 2^64 ints before repeating,
    # so identical fp_key values with different levels will get mapped
    # to different bits (unless counts are > 2^64, which is not actually supported).
    # We ignore overflow errors because overflow is intended (technically it should
    # be addition mod 2^64, but overflow handles this for us).
    #
    # After this step, we consider the (feature_id, count_level) pairs to
    # be jointly hashed and no longer refer to these features individually.
    with np.errstate(over="ignore"):
        fp_keys = fp_keys + count_levels * SPLITMIX64_GAMMA

    # Hash step 2: get two 64 bit ints for each block: one for the column hash
    # and one for the sign hash. We call the _subkeys method for this, which
    # generates these pseudo-randomly from the input seed. Call it "seed_block_keys"
    # for maximum clarity.
    seed_block_keys = _subkeys(seed=seed, n=2 * num_blocks)

    # Main loop over blocks: we will now separately fill in each block
    standard_block_length = dim // num_blocks  # round down, last block will get remainder
    for block_idx in range(num_blocks):
        # Get block start and end indices
        block_start = block_idx * standard_block_length
        this_block_length = standard_block_length
        if block_idx == num_blocks - 1:
            this_block_length += dim % num_blocks  # add remainder to last block

        # Hash step 3: XOR fp_keys with the appropriate seed_block_keys.
        # The function f(a) = a XOR b (for a constant b) is bijective,
        # so distinct fp_keys are guaranteed to map to distinct outputs.
        # We call these "pre*" because we will hash again in step 4
        this_block_col_key = seed_block_keys[2 * block_idx]
        this_block_sign_key = seed_block_keys[2 * block_idx + 1]
        pre_col_fp_hash = fp_keys ^ this_block_col_key
        pre_sign_fp_hash = fp_keys ^ this_block_sign_key

        # Hash step 4: although we could use the pre-hashes directly
        # as the hash, they are arguably not independent enough between
        # blocks. For example, we will only use one bit of the sign hash,
        # and if that bit happens to match between 2 block keys then *all*
        # the signs will match between those two blocks. To avoid this,
        # we re-hash the pre-hashes with splitmix, so differences cascade
        # more uniformly to all the bits.
        col_hash = _splitmix64_hash(pre_col_fp_hash)
        sign_hash = _splitmix64_hash(pre_sign_fp_hash)

        # Pick a col: col_hash mod block length.
        # intp dtype is meant for array indexing.
        # NOTE: col is relative to the block start- we apply this offset later
        col = (col_hash % U64(this_block_length)).astype(np.intp)

        # Pick a sign: take the first bit. We do this by bit-shifting by 63
        # and AND-ing with 1 (just to be safe).
        # 0 means "-1" and 1 means 1- we shift this in the next step
        sign01 = (sign_hash >> U64(63)) & U64(1)
        sign = 2 * sign01.astype(out.dtype) - 1

        # We will now add entries to out using np.bincount. However, this
        # accepts a 1D output, so we need to flatten and unflatten
        fill_indices = rows * this_block_length + col  # indices to fill in flattened array
        fill_values = np.bincount(
            fill_indices,  # we are counting every time each index is hit
            weights=sign,  # by default it adds 1, we override to add either +1 or -1
            minlength=n_rows * this_block_length,  # ensure full output dimension
        )
        out[:, block_start : block_start + this_block_length] = fill_values.reshape(
            n_rows, this_block_length
        )

    # Before returning, scale out by the number of blocks (see the `scale` docstring
    # on encode_sparse: this cancels out of the T_DP ratio, but keeps raw dot
    # products/norms comparable across different num_blocks settings).
    if scale:
        out = out / np.sqrt(num_blocks)
    return out
