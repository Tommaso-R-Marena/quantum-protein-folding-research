"""Tests for quantum algorithms (VQE, QAOA)."""

import pytest
import numpy as np
from qiskit.quantum_info import SparsePauliOp

from quantum_protein_folding.quantum.vqe import VQESolver
from quantum_protein_folding.quantum.qaoa import QAOASolver
from quantum_protein_folding.quantum.circuit_builder import (
    build_hardware_efficient_ansatz,
    build_qaoa_circuit,
)
from quantum_protein_folding.quantum.optimizer import VariationalOptimizer


class TestCircuitBuilder:
    """Tests for quantum circuit construction."""
    
    def test_hardware_efficient_ansatz(self):
        """Test hardware-efficient ansatz creation."""
        n_qubits = 4
        depth = 2
        
        circuit, params = build_hardware_efficient_ansatz(n_qubits, depth)
        
        assert circuit.num_qubits == n_qubits
        assert len(params) > 0
        assert circuit.num_parameters == len(params)
    
    def test_qaoa_circuit(self):
        """Test QAOA circuit construction."""
        # Simple Hamiltonian
        hamiltonian = SparsePauliOp.from_list([('ZZ', 1.0), ('Z', -0.5)])
        p_layers = 2
        
        circuit, params = build_qaoa_circuit(hamiltonian, p_layers)
        
        assert circuit.num_qubits == hamiltonian.num_qubits
        assert len(params) == 2 * p_layers  # beta and gamma for each layer


class TestOptimizer:
    """Tests for classical optimizer."""
    
    def test_optimizer_convergence(self):
        """Test optimizer finds minimum."""
        # Simple quadratic function
        def objective(x):
            return (x[0] - 3) ** 2 + (x[1] + 2) ** 2
        
        optimizer = VariationalOptimizer(method='COBYLA', maxiter=100)
        
        initial_params = np.array([0.0, 0.0])
        result = optimizer.optimize(objective, initial_params)
        
        assert result.success
        assert np.allclose(result.optimal_params, [3.0, -2.0], atol=0.1)
        assert result.optimal_value < 0.01


class TestVQESolver:
    """Tests for VQE solver."""
    
    def test_vqe_initialization(self):
        """Test VQE solver initialization."""
        hamiltonian = SparsePauliOp.from_list([('ZZ', 1.0), ('XX', -1.0)])
        
        solver = VQESolver(
            hamiltonian=hamiltonian,
            n_qubits=2,
            ansatz_depth=2,
            optimizer='COBYLA'
        )
        
        assert solver.n_qubits == 2
        assert solver.hamiltonian.num_qubits == 2
        assert solver.n_params > 0
    
    @pytest.mark.slow
    def test_vqe_simple_problem(self):
        """Test VQE on simple problem (slow test)."""
        # Simple 2-qubit Hamiltonian: H = Z0 + Z1 (ground state |11>)
        hamiltonian = SparsePauliOp.from_list([('IZ', 1.0), ('ZI', 1.0)])
        
        solver = VQESolver(
            hamiltonian=hamiltonian,
            n_qubits=2,
            ansatz_depth=1,
            optimizer='COBYLA',
            shots=1024
        )
        
        result = solver.run(maxiter=50)
        
        # Ground state energy should be -2 (both qubits in |1>)
        assert result.optimal_value < -1.5  # Allow some tolerance
        assert len(result.convergence_history) > 0


class TestQAOASolver:
    """Tests for QAOA solver."""
    
    def test_qaoa_initialization(self):
        """Test QAOA initialization."""
        hamiltonian = SparsePauliOp.from_list([('ZZ', 1.0)])
        
        solver = QAOASolver(
            hamiltonian=hamiltonian,
            p_layers=2,
            optimizer='COBYLA'
        )
        
        assert solver.n_qubits == 2
        assert solver.p_layers == 2
        assert solver.n_params == 4  # 2 betas + 2 gammas
    
    @pytest.mark.slow
    def test_qaoa_max_cut(self):
        """Test QAOA on MaxCut problem."""
        # MaxCut on 2 nodes: minimize -ZZ (solution: |01> or |10>)
        hamiltonian = SparsePauliOp.from_list([('ZZ', -1.0)])
        
        solver = QAOASolver(
            hamiltonian=hamiltonian,
            p_layers=1,
            optimizer='COBYLA'
        )
        
        result = solver.run(maxiter=30)
        
        # Optimal value should be close to -1
        assert result.optimal_value < 0
        assert result.optimal_bitstring in ['01', '10']
