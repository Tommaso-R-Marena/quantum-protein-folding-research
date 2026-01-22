"""Tests for lattice preprocessing."""

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


class TestEncoding:
    """Test lattice encoding schemes."""
    
    def test_binary_position_encoding(self):
        """Test binary position encoding."""
        seq = load_hp_sequence("HPHP")
        n_qubits, qubit_map = encode_binary_positions(seq, lattice_dim=2, lattice_size=8)
        
        # Each coordinate needs ceil(log2(8)) = 3 bits
        # 4 residues * 2 dimensions * 3 bits = 24 qubits
        assert n_qubits == 24
        assert len(qubit_map) == 24
    
    def test_turn_direction_encoding(self):
        """Test turn direction encoding."""
        seq = load_hp_sequence("HPHP")
        n_qubits, qubit_map = encode_turn_directions(seq, lattice_dim=2)
        
        # 2D: 4 directions need 2 bits per turn
        # 4 residues = 3 bonds = 3 turns
        # 3 turns * 2 bits = 6 qubits
        assert n_qubits == 6
        assert len(qubit_map) == 6
    
    def test_turn_encoding_3d(self):
        """Test 3D turn encoding."""
        seq = load_hp_sequence("HPHP")
        n_qubits, qubit_map = encode_turn_directions(seq, lattice_dim=3)
        
        # 3D: 6 directions need 3 bits per turn
        # 3 turns * 3 bits = 9 qubits
        assert n_qubits == 9


class TestLatticeMapping:
    """Test complete lattice mapping."""
    
    def test_map_to_lattice(self):
        """Test full lattice mapping pipeline."""
        seq = load_hp_sequence("HPH")
        encoding = map_to_lattice(seq, lattice_dim=2)
        
        assert encoding.n_qubits > 0
        assert encoding.lattice_dim == 2
        assert encoding.sequence == seq
        assert encoding.hamiltonian is not None
    
    def test_hamiltonian_hermitian(self):
        """Hamiltonian should be Hermitian."""
        seq = load_hp_sequence("HPH")
        encoding = map_to_lattice(seq, lattice_dim=2)
        
        # Small system: check Hermiticity
        if encoding.n_qubits <= 10:
            H_matrix = encoding.hamiltonian.to_matrix()
            assert np.allclose(H_matrix, H_matrix.conj().T)


class TestDecoding:
    """Test bitstring decoding."""
    
    def test_decode_turn_direction(self):
        """Test decoding turn direction encoding."""
        seq = load_hp_sequence("HPH")
        encoding = map_to_lattice(seq, lattice_dim=2, encoding_type='turn_direction')
        
        # Create valid bitstring (all zeros = all east moves)
        bitstring = '0' * encoding.n_qubits
        conformation = decode_conformation(bitstring, encoding)
        
        assert conformation.shape == (seq.length, 2)
        # All-east should give linear chain
        assert np.array_equal(conformation[:, 1], np.zeros(seq.length))
    
    def test_conformation_validation(self):
        """Test conformation validation."""
        seq = load_hp_sequence("HPH")
        
        # Valid linear conformation
        valid_conf = np.array([[0, 0], [1, 0], [2, 0]])
        is_valid, msg = check_valid_conformation(valid_conf, seq)
        assert is_valid
        
        # Invalid: bond length != 1
        invalid_conf = np.array([[0, 0], [2, 0], [3, 0]])  # Gap in chain
        is_valid, msg = check_valid_conformation(invalid_conf, seq)
        assert not is_valid
        assert 'length' in msg.lower()
        
        # Invalid: overlap
        overlap_conf = np.array([[0, 0], [1, 0], [1, 0]])  # Residues 1 and 2 overlap
        is_valid, msg = check_valid_conformation(overlap_conf, seq)
        assert not is_valid
        assert 'overlap' in msg.lower()
