"""One-off script that generated zinc_sample.smiles. Not a test; not run by pytest.

Reproduces the exact sample committed as zinc_sample.smiles:

    python tests/data/generate_sample.py

Source: ../../../zinc250k.smiles (250,456 lines, relative to this file), sampled
without replacement via random.Random(0).sample(...). If the source file has
moved, pass its path explicitly with --source.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

DEFAULT_N = 1000
DEFAULT_SEED = 0
DEFAULT_OUT = Path(__file__).parent / "zinc_sample.smiles"
DEFAULT_SOURCE = Path(__file__).resolve().parents[3] / "zinc250k.smiles"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    lines = [s.strip() for s in args.source.read_text().splitlines() if s.strip()]
    sample = random.Random(args.seed).sample(lines, args.n)
    args.out.write_text("\n".join(sample) + "\n")
    print(f"wrote {len(sample)} SMILES from {len(lines)}-line {args.source} to {args.out}")


if __name__ == "__main__":
    main()
