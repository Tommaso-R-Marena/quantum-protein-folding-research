"""Tests for classical baseline algorithms."""

import pytest
import numpy as np

from quantum_protein_folding.data.loaders import load_hp_sequence
from quantum_protein_folding.data.preprocess import map_to_lattice
from quantum_protein_folding.classical import (
    compute_energy,
    simulated_annealing_fold,
    exact_enumeration_fold,
)


class TestEnergyComputation:
    """Test energy calculation."""
    
    def test_linear_conformation_energy(self):
        """Test energy of linear conformation."""
        seq = load_hp_sequence("HHH")
        conformation = np.array([[0, 0], [1, 0], [2, 0]])  # Linear
        
        energy = compute_energy(conformation, seq, lattice_dim=2)
        
        # Linear has no contacts, so energy should be small (only bias)
        assert energy < 10.0  # No large penalties
    
    def test_invalid_conformation_penalty(self):
        """Invalid conformation should have high energy."""
        seq = load_hp_sequence("HHH")
        
        # Overlapping residues
        bad_conf = np.array([[0, 0], [0, 0], [1, 0]])
        
        energy = compute_energy(bad_conf, seq, lattice_dim=2, constraint_weight=10.0)
        
        # Should have large penalty
        assert energy > 100.0


class TestSimulatedAnnealing:
    """Test simulated annealing."""
    
    def test_sa_runs(self):
        """Test SA completes successfully."""
        seq = load_hp_sequence("HPH")
        encoding = map_to_lattice(seq, lattice_dim=2)
        
        result = simulated_annealing_fold(
            encoding,
            max_iterations=100,
            seed=42
        )
        
        assert result.conformation is not None
        assert result.energy is not None
        assert result.method == 'simulated_annealing'
    
    def test_sa_reproducible(self):
        """SA with same seed should be reproducible."""
        seq = load_hp_sequence("HPH")
        encoding = map_to_lattice(seq, lattice_dim=2)
        
        result1 = simulated_annealing_fold(encoding, max_iterations=50, seed=42)
        result2 = simulated_annealing_fold(encoding, max_iterations=50, seed=42)
        
        # Should get same result
        assert np.allclose(result1.conformation, result2.conformation)
        assert abs(result1.energy - result2.energy) < 1e-10


class TestExactEnumeration:
    """Test exact enumeration."""
    
    def test_exact_small_sequence(self):
        """Test exact enumeration on small sequence."""
        seq = load_hp_sequence("HPH")
        encoding = map_to_lattice(seq, lattice_dim=2)
        
        result = exact_enumeration_fold(encoding, max_conformations=1000)
        
        assert result.conformation is not None
        assert result.energy is not None
        assert result.method == 'exact_enumeration'
    
    def test_exact_too_large(self):
        """Exact enumeration should fail for large sequences."""
        seq = load_hp_sequence("H" * 15)  # Too long
        encoding = map_to_lattice(seq, lattice_dim=2)
        
        with pytest.raises(ValueError):
            exact_enumeration_fold(encoding)
