"""fpsketch: compress molecular count fingerprints to a fixed low dimension
while preserving Tanimoto (T_MM) similarity as a plain dot product.
"""

from .mols import encode_mols
from .sketching import encode_coo, encode_sparse

__all__ = [
    "encode_sparse",
    "encode_coo",
    "encode_mols",
]

__version__ = "0.1.0"
