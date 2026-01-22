"""Tests for quantum solvers."""

import pytest
import numpy as np
from qiskit.quantum_info import SparsePauliOp
from quantum_protein_folding.quantum.vqe import VQESolver
from quantum_protein_folding.quantum.qaoa import QAOASolver


class TestVQESolver:
    """Test VQE implementation."""
    
    def test_simple_hamiltonian(self):
        """Test VQE on simple Hamiltonian."""
        # Simple Hamiltonian: H = Z
        H = SparsePauliOp.from_list([('Z', 1.0)])
        
        solver = VQESolver(
            hamiltonian=H,
            n_qubits=1,
            ansatz_depth=1,
            shots=1024
        )
        
        result = solver.run(maxiter=50)
        
        # Ground state of Z is |1> with energy -1
        assert result.optimal_value < 0
        assert len(result.optimal_params) > 0
    
    def test_two_qubit_system(self):
        """Test VQE on two-qubit system."""
        # H = -ZZ (ground state |00> or |11>)
        H = SparsePauliOp.from_list([('ZZ', -1.0)])
        
        solver = VQESolver(
            hamiltonian=H,
            n_qubits=2,
            ansatz_depth=2
        )
        
        result = solver.run(maxiter=100)
        
        # Energy should be close to -1
        assert result.optimal_value < -0.5
    
    def test_convergence_history(self):
        """Test convergence tracking."""
        H = SparsePauliOp.from_list([('Z', 1.0)])
        
        solver = VQESolver(hamiltonian=H, n_qubits=1, ansatz_depth=1)
        result = solver.run(maxiter=30)
        
        assert len(result.convergence_history) > 0
        assert result.n_iterations > 0
    
    def test_bitstring_output(self):
        """Test bitstring is valid."""
        H = SparsePauliOp.from_list([('ZZ', 1.0)])
        
        solver = VQESolver(hamiltonian=H, n_qubits=2, ansatz_depth=1)
        result = solver.run(maxiter=30)
        
        assert len(result.optimal_bitstring) == 2
        assert all(b in '01' for b in result.optimal_bitstring)


class TestQAOASolver:
    """Test QAOA implementation."""
    
    def test_simple_qaoa(self):
        """Test QAOA on simple problem."""
        # MaxCut on 2 vertices: H = (I - Z_0 Z_1) / 2
        H = SparsePauliOp.from_list([('II', 0.5), ('ZZ', -0.5)])
        
        solver = QAOASolver(
            hamiltonian=H,
            p_layers=1,
            shots=1024
        )
        
        result = solver.run(maxiter=50)
        
        assert result.optimal_value < 0.6  # Should find good cut
        assert len(result.optimal_bitstring) == 2
    
    def test_multiple_layers(self):
        """Test QAOA with multiple layers."""
        H = SparsePauliOp.from_list([('ZZ', 1.0), ('Z', -0.5)])
        
        solver = QAOASolver(
            hamiltonian=H,
            p_layers=3
        )
        
        result = solver.run(maxiter=50)
        
        # Should have 2*p parameters
        assert len(result.optimal_params) == 2 * 3
    
    def test_solution_distribution(self):
        """Test QAOA returns probability distribution."""
        H = SparsePauliOp.from_list([('Z', 1.0)])
        
        solver = QAOASolver(hamiltonian=H, p_layers=1)
        result = solver.run(maxiter=30)
        
        assert isinstance(result.solution_distribution, dict)
        assert len(result.solution_distribution) > 0
        
        # Probabilities should sum to ~1
        total_prob = sum(result.solution_distribution.values())
        assert 0.9 <= total_prob <= 1.1
    
    def test_evaluate_bitstring(self):
        """Test direct bitstring evaluation."""
        H = SparsePauliOp.from_list([('Z', 1.0)])
        
        solver = QAOASolver(hamiltonian=H, p_layers=1)
        
        # Evaluate |0> and |1>
        energy_0 = solver.evaluate_solution('0')
        energy_1 = solver.evaluate_solution('1')
        
        assert isinstance(energy_0, (float, np.floating))
        assert isinstance(energy_1, (float, np.floating))
        # |1> should have lower energy for H=Z
        assert energy_1 < energy_0
