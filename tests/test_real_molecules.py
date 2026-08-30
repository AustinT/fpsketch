"""Guarantee (2): on real molecules, sketch-derived T_DP matches T_MM as computed
by RDKit's own native Tanimoto similarity on count fingerprints.

RDKit's DataStructs.BulkTanimotoSimilarity on count fingerprints uses the
min-max convention (T_MM), not cosine/T_DP -- verified independently in the
research repo backing this package (max diff < 1e-12 vs. a from-scratch T_MM).
"""

from __future__ import annotations

import numpy as np
import pytest

rdkit = pytest.importorskip("rdkit")

from _reference import t_dp  # noqa: E402
from rdkit import Chem, DataStructs  # noqa: E402
from rdkit.Chem import rdFingerprintGenerator  # noqa: E402

from fpsketch import encode_mols, encode_sparse  # noqa: E402

N_MOLS = 200
M = 2048
SEED = 0


@pytest.fixture(scope="module")
def mols(zinc_smiles):
    parsed = (Chem.MolFromSmiles(s) for s in zinc_smiles)
    valid = [m for m in parsed if m is not None]
    return valid[:N_MOLS]


def test_encode_sparse_matches_rdkit_tanimoto(mols):
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2)
    fps = [generator.GetSparseCountFingerprint(m) for m in mols]

    k_true = np.array([DataStructs.BulkTanimotoSimilarity(fp, fps) for fp in fps])

    dicts = [fp.GetNonzeroElements() for fp in fps]
    sketch = encode_sparse(dicts, dim=M, seed=SEED)
    k_hat = t_dp(sketch)

    iu = np.triu_indices(len(mols), k=1)

    # MAE
    mae = np.mean(np.abs(k_hat[iu] - k_true[iu]))
    assert mae < 0.015

    # corr
    corr = np.corrcoef(k_hat[iu], k_true[iu])[0, 1]
    assert corr > 0.95


def test_encode_mols_matches_encode_sparse(mols):
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2)
    dicts = [generator.GetSparseCountFingerprint(m).GetNonzeroElements() for m in mols]

    via_encode_sparse = encode_sparse(dicts, dim=M, seed=SEED)
    via_encode_mols = encode_mols(mols, dim=M, seed=SEED)

    assert np.array_equal(via_encode_sparse, via_encode_mols)


def test_encode_mols_accepts_custom_generator(mols):
    subset = mols[:20]
    ap_generator = rdFingerprintGenerator.GetAtomPairGenerator()

    out_default = encode_mols(subset, dim=256, seed=SEED)
    out_atompair = encode_mols(subset, generator=ap_generator, dim=256, seed=SEED)

    assert out_default.shape == out_atompair.shape == (20, 256)
    assert not np.array_equal(out_default, out_atompair)
