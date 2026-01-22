"""Tests for preprocessing and lattice encoding."""

import pytest
import numpy as np

from quantum_protein_folding.data.loaders import load_hp_sequence
from quantum_protein_folding.data.preprocess import (
    encode_binary_positions,
    encode_turn_directions,
    map_to_lattice,
    decode_conformation,
    check_valid_conformation,
)


class TestLatticeEncoding:
    """Tests for lattice encoding schemes."""
    
    def test_binary_position_encoding_2d(self):
        """Test binary position encoding in 2D."""
        seq = load_hp_sequence("HPHH")
        n_qubits, qubit_map = encode_binary_positions(seq, lattice_dim=2, lattice_size=6)
        
        # 4 residues * 2 dimensions * ceil(log2(6)) bits = 4*2*3 = 24
        assert n_qubits == 24
        assert len(qubit_map) == n_qubits
    
    def test_turn_direction_encoding_2d(self):
        """Test turn direction encoding in 2D."""
        seq = load_hp_sequence("HPHH")
        n_qubits, qubit_map = encode_turn_directions(seq, lattice_dim=2)
        
        # 3 bonds * ceil(log2(4)) = 3 * 2 = 6
        assert n_qubits == 6
        assert len(qubit_map) == n_qubits
    
    def test_turn_direction_encoding_3d(self):
        """Test turn direction encoding in 3D."""
        seq = load_hp_sequence("HPH")
        n_qubits, qubit_map = encode_turn_directions(seq, lattice_dim=3)
        
        # 2 bonds * ceil(log2(6)) = 2 * 3 = 6
        assert n_qubits == 6


class TestHamiltonianConstruction:
    """Tests for Hamiltonian construction."""
    
    def test_hamiltonian_creation(self):
        """Test Hamiltonian is created correctly."""
        seq = load_hp_sequence("HPHH")
        encoding = map_to_lattice(seq, lattice_dim=2, encoding_type='turn_direction')
        
        assert encoding.hamiltonian is not None
        assert encoding.hamiltonian.num_qubits == encoding.n_qubits
    
    def test_hamiltonian_hermitian(self):
        """Test Hamiltonian is Hermitian."""
        seq = load_hp_sequence("HPH")
        encoding = map_to_lattice(seq, lattice_dim=2)
        
        H_matrix = encoding.hamiltonian.to_matrix()
        assert np.allclose(H_matrix, H_matrix.conj().T)


class TestConformationDecoding:
    """Tests for bitstring decoding."""
    
    def test_decode_turn_direction(self):
        """Test decoding turn-based encoding."""
        seq = load_hp_sequence("HPHH")
        encoding = map_to_lattice(seq, lattice_dim=2, encoding_type='turn_direction')
        
        # Simple bitstring (all zeros)
        bitstring = '0' * encoding.n_qubits
        conformation = decode_conformation(bitstring, encoding)
        
        assert conformation.shape == (4, 2)
        assert conformation[0, 0] == 0 and conformation[0, 1] == 0  # Origin
    
    def test_conformation_validation_valid(self):
        """Test valid conformation passes."""
        seq = load_hp_sequence("HPHH")
        
        # Valid self-avoiding walk
        conformation = np.array([
            [0, 0],
            [1, 0],
            [1, 1],
            [0, 1]
        ])
        
        is_valid, message = check_valid_conformation(conformation, seq)
        assert is_valid
    
    def test_conformation_validation_overlap(self):
        """Test overlapping residues detected."""
        seq = load_hp_sequence("HPHH")
        
        # Invalid: residues 0 and 2 overlap
        conformation = np.array([
            [0, 0],
            [1, 0],
            [0, 0],  # Overlaps with residue 0
            [1, 1]
        ])
        
        is_valid, message = check_valid_conformation(conformation, seq)
        assert not is_valid
        assert "overlap" in message.lower()
