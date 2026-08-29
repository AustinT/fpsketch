from pathlib import Path

import pytest

ZINC_SAMPLE = Path(__file__).parent / "data" / "zinc_sample.smiles"


@pytest.fixture(scope="module")
def zinc_smiles() -> list[str]:
    return [s.strip() for s in ZINC_SAMPLE.read_text().splitlines() if s.strip()]
