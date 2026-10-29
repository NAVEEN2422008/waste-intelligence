import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import generate_data  # noqa: E402


@pytest.fixture(scope="session")
def small_data():
    """Small SIMULATED dataset so tests run fast (8 zones x 300 days)."""
    return generate_data.generate_dataset(n_zones=8, n_days=300, seed=1, test_days=60)
