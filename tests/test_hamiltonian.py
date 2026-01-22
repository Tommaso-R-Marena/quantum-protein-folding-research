"""Tests for Hamiltonian construction."""

import pytest
import numpy as np
from qiskit.quantum_info import SparsePauliOp

from quantum_protein_folding.data.loaders import load_hp_sequence
from quantum_protein_folding.quantum.hamiltonian import (
    build_hamiltonian,
    hamiltonian_to_matrix,
    compute_exact_ground_state,
)


class TestHamiltonianConstruction:
    """Test Hamiltonian building."""
    
    def test_build_hamiltonian(self):
        """Test basic Hamiltonian construction."""
        seq = load_hp_sequence("HPH")
        n_qubits = 4  # Small system
        
        H = build_hamiltonian(
            sequence=seq,
            n_qubits=n_qubits,
            lattice_dim=2,
            lattice_size=5,
            encoding_type='turn_direction'
        )
        
        assert isinstance(H, SparsePauliOp)
        assert H.num_qubits == n_qubits
    
    def test_hamiltonian_hermitian(self):
        """Hamiltonian must be Hermitian."""
        seq = load_hp_sequence("HP")
        n_qubits = 2
        
        H = build_hamiltonian(seq, n_qubits, 2, 3, 'turn_direction')
        H_matrix = hamiltonian_to_matrix(H)
        
        assert np.allclose(H_matrix, H_matrix.conj().T)
    
    def test_hamiltonian_real_eigenvalues(self):
        """Hermitian matrix has real eigenvalues."""
        seq = load_hp_sequence("HP")
        n_qubits = 2
        
        H = build_hamiltonian(seq, n_qubits, 2, 3, 'turn_direction')
        H_matrix = hamiltonian_to_matrix(H)
        
        eigenvalues = np.linalg.eigvalsh(H_matrix)
        assert np.all(np.isreal(eigenvalues))


class TestGroundState:
    """Test exact ground state computation."""
    
    def test_ground_state_normalized(self):
        """Ground state should be normalized."""
        seq = load_hp_sequence("HP")
        n_qubits = 2
        
        H = build_hamiltonian(seq, n_qubits, 2, 3, 'turn_direction')
        energy, state = compute_exact_ground_state(H)
        
        # Check normalization
        norm = np.linalg.norm(state)
        assert abs(norm - 1.0) < 1e-10
    
    def test_ground_energy_lowest(self):
        """Ground state energy should be the lowest."""
        seq = load_hp_sequence("HP")
        n_qubits = 2
        
        H = build_hamiltonian(seq, n_qubits, 2, 3, 'turn_direction')
        ground_energy, _ = compute_exact_ground_state(H)
        
        # All eigenvalues
        H_matrix = hamiltonian_to_matrix(H)
        all_eigenvalues = np.linalg.eigvalsh(H_matrix)
        
        # Ground energy should be minimum
        assert abs(ground_energy - np.min(all_eigenvalues)) < 1e-10
