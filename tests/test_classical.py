"""Tests for classical algorithms."""

import pytest
import numpy as np

from quantum_protein_folding.data.loaders import load_hp_sequence
from quantum_protein_folding.data.preprocess import map_to_lattice
from quantum_protein_folding.classical import (
    simulated_annealing_fold,
    exact_enumeration_fold,
    compute_energy,
)


class TestSimulatedAnnealing:
    """Test simulated annealing."""
    
    def test_sa_basic(self):
        """Test basic SA functionality."""
        sequence = load_hp_sequence("HPHP")
        encoding = map_to_lattice(sequence, lattice_dim=2)
        
        result = simulated_annealing_fold(
            encoding,
            max_iterations=100,
            seed=42
        )
        
        assert result.conformation.shape == (4, 2)
        assert isinstance(result.energy, float)
        assert result.method == 'simulated_annealing'
    
    def test_sa_reproducible(self):
        """Test that SA is reproducible with same seed."""
        sequence = load_hp_sequence("HPH")
        encoding = map_to_lattice(sequence, lattice_dim=2)
        
        result1 = simulated_annealing_fold(encoding, max_iterations=50, seed=42)
        result2 = simulated_annealing_fold(encoding, max_iterations=50, seed=42)
        
        assert np.allclose(result1.conformation, result2.conformation)
        assert np.isclose(result1.energy, result2.energy)


class TestExactEnumeration:
    """Test exact enumeration."""
    
    def test_exact_small_sequence(self):
        """Test exact enumeration on small sequence."""
        sequence = load_hp_sequence("HPH")
        encoding = map_to_lattice(sequence, lattice_dim=2)
        
        result = exact_enumeration_fold(encoding, max_conformations=1000)
        
        assert result.conformation.shape == (3, 2)
        assert result.method == 'exact_enumeration'
    
    def test_exact_too_large(self):
        """Test that exact enumeration raises error for large sequences."""
        sequence = load_hp_sequence("H" * 15)  # Too large
        encoding = map_to_lattice(sequence, lattice_dim=2)
        
        with pytest.raises(ValueError):
            exact_enumeration_fold(encoding)


class TestEnergyComputation:
    """Test energy calculations."""
    
    def test_compute_energy(self):
        """Test energy computation."""
        sequence = load_hp_sequence("HPHP")
        
        # Create simple conformation
        conformation = np.array([
            [0, 0],
            [1, 0],
            [2, 0],
            [2, 1]
        ])
        
        energy = compute_energy(conformation, sequence, lattice_dim=2)
        
        assert isinstance(energy, float)
