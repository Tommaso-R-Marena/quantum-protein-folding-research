"""Tests for QAOA solver."""

import pytest
import numpy as np
from qiskit.quantum_info import SparsePauliOp

from quantum_protein_folding.quantum.qaoa import QAOASolver
from quantum_protein_folding.models.qaoa_model import QAOAFoldingModel


class TestQAOASolver:
    """Test QAOA implementation."""
    
    def test_qaoa_initialization(self):
        """Test QAOA initialization."""
        H = SparsePauliOp.from_list([('ZZ', 1.0), ('Z', -0.5)])
        
        solver = QAOASolver(
            hamiltonian=H,
            p_layers=1
        )
        
        assert solver.n_qubits == 2
        assert solver.p_layers == 1
        assert solver.n_params == 2  # 1 beta + 1 gamma
    
    def test_qaoa_circuit_structure(self):
        """Test QAOA circuit has correct structure."""
        H = SparsePauliOp.from_list([('Z', 1.0)])
        
        solver = QAOASolver(hamiltonian=H, p_layers=2)
        
        # Should have 2*p parameters (beta and gamma for each layer)
        assert solver.n_params == 4
    
    @pytest.mark.slow
    def test_qaoa_maxcut(self):
        """Test QAOA on simple MaxCut problem."""
        # Simple 2-qubit MaxCut: H = ZZ (want opposite states)
        H = SparsePauliOp.from_list([('ZZ', 1.0)])
        
        solver = QAOASolver(
            hamiltonian=H,
            p_layers=1,
            optimizer='COBYLA'
        )
        
        result = solver.run(maxiter=30)
        
        # Should find solution (|01> or |10>)
        assert result.optimal_bitstring in ['01', '10']


class TestQAOAFoldingModel:
    """Test QAOA folding model."""
    
    def test_model_creation(self):
        """Test QAOA model initialization."""
        model = QAOAFoldingModel(
            sequence="HP",
            p_layers=1,
            lattice_dim=2
        )
        
        assert model.sequence.length == 2
        assert model.solver.p_layers == 1
    
    @pytest.mark.slow
    def test_qaoa_folding(self):
        """Test QAOA folding workflow."""
        model = QAOAFoldingModel(
            sequence="HP",
            p_layers=1,
            lattice_dim=2
        )
        
        result = model.run(maxiter=20)
        
        assert result.optimal_value is not None
        assert result.optimal_bitstring is not None
        assert len(result.solution_distribution) > 0
