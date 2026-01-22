"""Tests for VQE solver."""

import pytest
import numpy as np

from quantum_protein_folding.models import VQEFoldingModel
from quantum_protein_folding.data.loaders import load_hp_sequence


class TestVQESolver:
    """Test VQE implementation."""
    
    def test_vqe_initialization(self):
        """Test VQE model initialization."""
        model = VQEFoldingModel(
            sequence="HPHP",
            lattice_dim=2,
            ansatz_depth=2
        )
        
        assert model.solver.n_params > 0
        assert model.encoding.n_qubits > 0
    
    def test_vqe_run_small(self):
        """Test VQE on small sequence."""
        model = VQEFoldingModel(
            sequence="HP",
            lattice_dim=2,
            ansatz_depth=1,
            optimizer='COBYLA'
        )
        
        result = model.run(maxiter=10)
        
        assert result.optimal_value is not None
        assert len(result.optimal_params) == model.solver.n_params
        assert len(result.convergence_history) > 0
    
    def test_vqe_convergence(self):
        """Test that VQE converges (energy decreases)."""
        model = VQEFoldingModel(
            sequence="HPH",
            lattice_dim=2,
            ansatz_depth=2
        )
        
        result = model.run(maxiter=20)
        
        # Final energy should be lower than initial
        initial_energy = result.convergence_history[0]
        final_energy = result.convergence_history[-1]
        
        assert final_energy <= initial_energy
    
    def test_vqe_bitstring_length(self):
        """Test that bitstring has correct length."""
        model = VQEFoldingModel(
            sequence="HPHP",
            lattice_dim=2
        )
        
        result = model.run(maxiter=5)
        
        assert len(result.optimal_bitstring) == model.encoding.n_qubits
