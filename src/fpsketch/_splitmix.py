"""A numpy implementation of splitmix"""

import numpy as np

U64 = np.uint64

# Fixed constants for splitmix
GAMMA = U64(0x9E3779B97F4A7C15)
MIX_A = U64(0xBF58476D1CE4E5B9)
MIX_B = U64(0x94D049BB133111EB)


def _splitmix64_hash(x: np.ndarray) -> np.ndarray:
    """
    Splitmix64 function- an 64 bit rng/hash which is simple to implement,
    invertible, and is considered fairly pseudo-random.

    It is not strong enough for cryptography but doesn't need to be:
    this is just used to encode fingerprints.
    """

    # Ignore int overflows- this is an intended part of the algorithm
    with np.errstate(over="ignore"):
        z = x + GAMMA
        z = (z ^ (z >> U64(30))) * MIX_A
        z = (z ^ (z >> U64(27))) * MIX_B
        z = (z ^ (z >> U64(31)))
    return z
