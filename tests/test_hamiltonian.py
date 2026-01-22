"""Tests for Hamiltonian construction."""

import pytest
import numpy as np
from qiskit.quantum_info import SparsePauliOp

from quantum_protein_folding.data.loaders import load_hp_sequence
from quantum_protein_folding.data.preprocess import map_to_lattice
from quantum_protein_folding.quantum.hamiltonian import (
    build_hamiltonian,
    hamiltonian_to_matrix,
    compute_exact_ground_state,
)


class TestHamiltonianConstruction:
    """Test Hamiltonian building."""
    
    def test_build_hamiltonian(self):
        """Test basic Hamiltonian construction."""
        sequence = load_hp_sequence("HPHP")
        encoding = map_to_lattice(sequence, lattice_dim=2)
        
        H = encoding.hamiltonian
        
        assert isinstance(H, SparsePauliOp)
        assert H.num_qubits == encoding.n_qubits
    
    def test_hamiltonian_hermitian(self):
        """Test that Hamiltonian is Hermitian."""
        sequence = load_hp_sequence("HP")
        encoding = map_to_lattice(sequence, lattice_dim=2)
        
        H_matrix = hamiltonian_to_matrix(encoding.hamiltonian)
        
        # Check Hermiticity
        assert np.allclose(H_matrix, H_matrix.conj().T)
    
    def test_hamiltonian_real_eigenvalues(self):
        """Test that eigenvalues are real."""
        sequence = load_hp_sequence("HPH")
        encoding = map_to_lattice(sequence, lattice_dim=2)
        
        H_matrix = hamiltonian_to_matrix(encoding.hamiltonian)
        eigenvalues = np.linalg.eigvalsh(H_matrix)
        
        # All eigenvalues should be real
        assert np.all(np.isreal(eigenvalues))


class TestExactDiagonalization:
    """Test exact ground state computation."""
    
    def test_exact_ground_state(self):
        """Test exact diagonalization for small system."""
        sequence = load_hp_sequence("HP")
        encoding = map_to_lattice(sequence, lattice_dim=2)
        
        if encoding.n_qubits <= 12:
            energy, state = compute_exact_ground_state(encoding.hamiltonian)
            
            assert isinstance(energy, float)
            assert len(state) == 2 ** encoding.n_qubits
            assert np.isclose(np.linalg.norm(state), 1.0)  # Normalized
