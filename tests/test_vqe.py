"""Tests for VQE solver."""

import pytest
import numpy as np
from qiskit.quantum_info import SparsePauliOp

from quantum_protein_folding.quantum.vqe import VQESolver
from quantum_protein_folding.models.vqe_model import VQEFoldingModel


class TestVQESolver:
    """Test VQE implementation."""
    
    def test_vqe_initialization(self):
        """Test VQE solver initialization."""
        # Simple Hamiltonian
        H = SparsePauliOp.from_list([('ZZ', 1.0), ('XX', -1.0)])
        
        solver = VQESolver(
            hamiltonian=H,
            n_qubits=2,
            ansatz_depth=1
        )
        
        assert solver.n_qubits == 2
        assert solver.n_params > 0
    
    @pytest.mark.slow
    def test_vqe_simple_problem(self):
        """Test VQE on simple problem (H = Z)."""
        # Ground state of Z is |0> with energy -1
        H = SparsePauliOp.from_list([('Z', 1.0)])
        
        solver = VQESolver(
            hamiltonian=H,
            n_qubits=1,
            ansatz_depth=1,
            optimizer='COBYLA'
        )
        
        result = solver.run(maxiter=50)
        
        # Should find ground state energy ~ -1
        assert result.optimal_value < 0  # Negative
        assert result.optimal_value > -1.5  # Not too negative
        assert len(result.convergence_history) > 0


class TestVQEFoldingModel:
    """Test high-level VQE folding API."""
    
    def test_model_initialization(self):
        """Test model creation."""
        model = VQEFoldingModel(
            sequence="HPH",
            lattice_dim=2,
            ansatz_depth=1,
            shots=512
        )
        
        assert model.sequence.length == 3
        assert model.encoding.n_qubits > 0
    
    @pytest.mark.slow
    def test_full_folding_pipeline(self):
        """Test complete folding workflow."""
        model = VQEFoldingModel(
            sequence="HPH",
            lattice_dim=2,
            ansatz_depth=1
        )
        
        result = model.run(maxiter=20)  # Quick test
        
        assert result.optimal_value is not None
        assert result.optimal_bitstring is not None
        assert len(result.convergence_history) > 0
        
        # Decode conformation
        conf = model.decode_conformation(result.optimal_bitstring)
        assert conf.shape == (3, 2)  # 3 residues, 2D
