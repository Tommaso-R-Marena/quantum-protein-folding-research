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
        sequence = load_hp_sequence("HPH")
        n_qubits = 4
        
        H = build_hamiltonian(
            sequence=sequence,
            n_qubits=n_qubits,
            lattice_dim=2,
            lattice_size=4,
            encoding_type='turn_direction'
        )
        
        assert isinstance(H, SparsePauliOp)
        assert H.num_qubits == n_qubits
    
    def test_hamiltonian_hermitian(self):
        """Test Hamiltonian is Hermitian."""
        sequence = load_hp_sequence("HPH")
        
        H = build_hamiltonian(
            sequence=sequence,
            n_qubits=4,
            lattice_dim=2,
            lattice_size=4
        )
        
        H_matrix = hamiltonian_to_matrix(H)
        
        assert np.allclose(H_matrix, H_matrix.conj().T)
    
    def test_hamiltonian_real(self):
        """Test Hamiltonian is real-valued."""
        sequence = load_hp_sequence("HPHPH")
        
        H = build_hamiltonian(
            sequence=sequence,
            n_qubits=6,
            lattice_dim=2,
            lattice_size=5
        )
        
        H_matrix = hamiltonian_to_matrix(H)
        
        assert np.allclose(H_matrix.imag, 0)
    
    def test_different_weights(self):
        """Test different constraint weights affect Hamiltonian."""
        sequence = load_hp_sequence("HPH")
        
        H1 = build_hamiltonian(
            sequence, n_qubits=4, lattice_dim=2, lattice_size=4,
            constraint_weight=1.0
        )
        
        H2 = build_hamiltonian(
            sequence, n_qubits=4, lattice_dim=2, lattice_size=4,
            constraint_weight=10.0
        )
        
        # Hamiltonians should be different
        assert not np.allclose(
            hamiltonian_to_matrix(H1),
            hamiltonian_to_matrix(H2)
        )


class TestExactDiagonalization:
    """Test exact ground state computation."""
    
    def test_compute_ground_state(self):
        """Test exact ground state calculation."""
        sequence = load_hp_sequence("HP")
        
        H = build_hamiltonian(
            sequence, n_qubits=2, lattice_dim=2, lattice_size=3
        )
        
        energy, state = compute_exact_ground_state(H)
        
        assert isinstance(energy, (float, np.floating))
        assert len(state) == 2 ** 2
        assert np.isclose(np.linalg.norm(state), 1.0)
    
    def test_ground_state_energy_real(self):
        """Test ground state energy is real."""
        sequence = load_hp_sequence("HPH")
        
        H = build_hamiltonian(
            sequence, n_qubits=3, lattice_dim=2, lattice_size=4
        )
        
        energy, _ = compute_exact_ground_state(H)
        
        assert np.isreal(energy)
