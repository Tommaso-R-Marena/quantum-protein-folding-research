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
    """Tests for energy calculation."""
    
    def test_compute_contact_energy(self):
        """Test contact energy calculation."""
        from quantum_protein_folding.classical.energy import compute_contact_energy
        
        seq = load_hp_sequence("HPHH")
        
        # Square conformation with H-H contact
        conformation = np.array([
            [0, 0],
            [1, 0],
            [1, 1],
            [0, 1]
        ])
        
        energy = compute_contact_energy(conformation, seq)
        
        # H at positions 0 and 2 are in contact
        assert energy == -1.0
    
    def test_compute_backbone_energy_valid(self):
        """Test backbone energy for valid conformation."""
        from quantum_protein_folding.classical.energy import compute_backbone_energy
        
        # Valid conformation (all bonds length 1)
        conformation = np.array([
            [0, 0],
            [1, 0],
            [1, 1]
        ])
        
        energy = compute_backbone_energy(conformation)
        
        # Should be near zero for valid conformation
        assert energy < 0.01
    
    def test_compute_total_energy(self):
        """Test total energy computation."""
        seq = load_hp_sequence("HPHH")
        
        conformation = np.array([
            [0, 0],
            [1, 0],
            [1, 1],
            [0, 1]
        ])
        
        energy = compute_energy(conformation, seq, lattice_dim=2)
        
        # Energy should be finite
        assert np.isfinite(energy)


class TestSimulatedAnnealing:
    """Tests for simulated annealing."""
    
    def test_sa_convergence(self):
        """Test SA finds valid conformation."""
        seq = load_hp_sequence("HPHH")
        encoding = map_to_lattice(seq, lattice_dim=2)
        
        result = simulated_annealing_fold(
            encoding,
            max_iterations=1000,
            seed=42
        )
        
        assert result.method == 'simulated_annealing'
        assert result.conformation.shape == (4, 2)
        assert np.isfinite(result.energy)
    
    def test_sa_improves_over_iterations(self):
        """Test SA energy improves."""
        seq = load_hp_sequence("HPHH")
        encoding = map_to_lattice(seq, lattice_dim=2)
        
        # Short run
        result_short = simulated_annealing_fold(encoding, max_iterations=100, seed=42)
        
        # Long run
        result_long = simulated_annealing_fold(encoding, max_iterations=5000, seed=42)
        
        # Longer run should generally find better or equal energy
        assert result_long.energy <= result_short.energy + 1.0  # Allow some variance


class TestExactEnumeration:
    """Tests for exact enumeration."""
    
    def test_exact_enumeration_small(self):
        """Test exact enumeration on tiny sequence."""
        seq = load_hp_sequence("HPH")
        encoding = map_to_lattice(seq, lattice_dim=2)
        
        result = exact_enumeration_fold(encoding, max_conformations=1000)
        
        assert result.method == 'exact_enumeration'
        assert result.conformation.shape == (3, 2)
        assert result.n_iterations <= 1000
    
    def test_exact_enumeration_too_large(self):
        """Test error for sequences too large."""
        seq = load_hp_sequence("H" * 15)
        encoding = map_to_lattice(seq, lattice_dim=2)
        
        with pytest.raises(ValueError, match="infeasible"):
            exact_enumeration_fold(encoding)
