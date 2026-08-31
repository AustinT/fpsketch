"""RDKit molecule entry point. Requires the ``chem`` extra (``pip install fpsketch[chem]``)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from .sketching import encode_sparse


def encode_mols(
    mols: Sequence[Any],
    dim: int = 2048,
    generator: Any | None = None,
    num_blocks: int = 4,
    seed: int = 0,
    scale: bool = True,
) -> np.ndarray:
    """Sketch RDKit molecules into a dense ``(len(mols), dim)`` array.

    Args:
        mols: RDKit ``ROMol`` objects.
        dim: output sketch width.
        generator: an ``rdFingerprintGenerator`` generator (Morgan, RDKitFP,
            AtomPair, TopologicalTorsion, ...). Defaults to
            ``GetMorganGenerator(radius=2)`` if not given. Any generator
            exposing ``GetSparseCountFingerprint`` works.
        num_blocks: see ``encode_sparse``.
        seed: hash seed; two sketches are only dot-product-comparable if built
            with the same seed.
        scale: see ``encode_sparse``.
    """
    try:
        from rdkit.Chem import rdFingerprintGenerator
    except ImportError as e:
        raise ImportError(
            "encode_mols requires rdkit. Install it with `pip install fpsketch[chem]`."
        ) from e

    if generator is None:
        generator = rdFingerprintGenerator.GetMorganGenerator(radius=2)

    fps = [generator.GetSparseCountFingerprint(mol).GetNonzeroElements() for mol in mols]
    return encode_sparse(fps, dim=dim, num_blocks=num_blocks, seed=seed, scale=scale)
