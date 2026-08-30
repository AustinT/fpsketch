"""Nearest-neighbor search over a handful of molecules using fpsketch, compared
against RDKit's native Tanimoto similarity.

Run with:
    uv run --extra chem python examples/similarity_search.py
"""

from __future__ import annotations

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from fpsketch import encode_mols

SMILES = [
    "CC(=O)Oc1ccccc1C(=O)O",  # aspirin
    "CC(=O)Nc1ccc(O)cc1",  # paracetamol
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",  # ibuprofen
    "c1ccc2c(c1)ccc1ccccc12",  # anthracene
    "c1ccc2ccccc2c1",  # naphthalene
    "CCO",  # ethanol
    "CCCCO",  # butanol
    "OCC(O)CO",  # glycerol
    "c1ccccc1",  # benzene
    "Cc1ccccc1",  # toluene
    "c1ccc(cc1)c1ccccc1",  # biphenyl
    "CC(C)(C)c1ccccc1",  # tert-butylbenzene
    "CN1CCC[C@H]1c1cccnc1",  # nicotine
    "Cn1cnc2c1c(=O)n(C)c(=O)n2C",  # caffeine
    "OC(=O)c1ccccc1O",  # salicylic acid
]

QUERY_IDX = 0  # aspirin
TOP_K = 5


def t_dp(X: np.ndarray) -> np.ndarray:
    G = X @ X.T
    sq = np.einsum("ij,ij->i", X, X)
    return G / np.maximum(sq[:, None] + sq[None, :] - G, 1e-12)


def main() -> None:
    mols = [Chem.MolFromSmiles(s) for s in SMILES]

    sketch = encode_mols(mols, dim=2048, seed=0)
    k_hat = t_dp(sketch)

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2)
    fps = [generator.GetSparseCountFingerprint(m) for m in mols]
    k_true = np.array([DataStructs.BulkTanimotoSimilarity(fp, fps) for fp in fps])

    query_sims_hat = k_hat[QUERY_IDX]
    query_sims_true = k_true[QUERY_IDX]

    ranked = sorted(
        (i for i in range(len(mols)) if i != QUERY_IDX),
        key=lambda i: -query_sims_hat[i],
    )[:TOP_K]

    print(f"query: {SMILES[QUERY_IDX]}")
    print(f"{'SMILES':<40} {'fpsketch T_DP':>15} {'RDKit T_MM':>12}")
    for i in ranked:
        print(f"{SMILES[i]:<40} {query_sims_hat[i]:>15.4f} {query_sims_true[i]:>12.4f}")


if __name__ == "__main__":
    main()
