"""Pytest configuration and fixtures."""

import pytest
import numpy as np
from quantum_protein_folding.data.loaders import load_hp_sequence
from quantum_protein_folding.data.preprocess import map_to_lattice


@pytest.fixture
def simple_sequence():
    """Fixture providing a simple HP sequence."""
    return load_hp_sequence("HPHPH")


@pytest.fixture
def simple_encoding(simple_sequence):
    """Fixture providing a simple lattice encoding."""
    return map_to_lattice(
        simple_sequence,
        lattice_dim=2,
        encoding_type='turn_direction'
    )


@pytest.fixture
def random_seed():
    """Fixture for reproducible random tests."""
    np.random.seed(42)
    return 42
