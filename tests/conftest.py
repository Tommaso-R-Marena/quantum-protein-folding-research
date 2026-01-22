"""Pytest configuration and fixtures."""

import pytest
import numpy as np

from quantum_protein_folding.data.loaders import load_hp_sequence
from quantum_protein_folding.data.preprocess import map_to_lattice


@pytest.fixture
def simple_hp_sequence():
    """Simple HP sequence for testing."""
    return load_hp_sequence("HPHH")


@pytest.fixture
def medium_hp_sequence():
    """Medium HP sequence."""
    return load_hp_sequence("HPHPPHHPHH")


@pytest.fixture
def simple_lattice_encoding(simple_hp_sequence):
    """Simple lattice encoding."""
    return map_to_lattice(
        simple_hp_sequence,
        lattice_dim=2,
        encoding_type='turn_direction'
    )


@pytest.fixture
def simple_conformation_2d():
    """Simple 2D conformation."""
    return np.array([
        [0, 0],
        [1, 0],
        [1, 1],
        [0, 1]
    ])


@pytest.fixture
def random_seed():
    """Set random seed for reproducibility."""
    np.random.seed(42)
    return 42
