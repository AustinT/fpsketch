"""fpsketch: compress molecular count fingerprints to a fixed low dimension
while preserving Tanimoto (T_MM) similarity as a plain dot product.
"""

from importlib.metadata import PackageNotFoundError, version

from .mols import encode_mols
from .sketching import encode_coo, encode_sparse

__all__ = [
    "encode_sparse",
    "encode_coo",
    "encode_mols",
]

try:
    __version__ = version("fpsketch")
except PackageNotFoundError:  # pragma: no cover - not installed, e.g. running from source
    __version__ = "unknown"
