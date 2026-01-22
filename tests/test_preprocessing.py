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
    
    def test_turn_encoding_qubits(self):
        """Test turn encoding qubit count."""
        sequence = load_hp_sequence("HPHPH")
        n_qubits, qubit_map = encode_turn_directions(sequence, lattice_dim=2)
        
        # For 2D: 4 directions = 2 bits per turn
        # For N=5: 4 bonds, 2 bits each = 8 qubits
        assert n_qubits == 8
        assert len(qubit_map) == 8
    
    def test_binary_encoding_qubits(self):
        """Test binary position encoding qubit count."""
        sequence = load_hp_sequence("HPH")
        n_qubits, qubit_map = encode_binary_positions(
            sequence, lattice_dim=2, lattice_size=4
        )
        
        # 3 residues, 2 coords, 2 bits per coord = 12 qubits
        assert n_qubits == 12
    
    def test_encoding_3d(self):
        """Test 3D encoding."""
        sequence = load_hp_sequence("HPH")
        n_qubits_2d, _ = encode_turn_directions(sequence, lattice_dim=2)
        n_qubits_3d, _ = encode_turn_directions(sequence, lattice_dim=3)
        
        # 3D should require more qubits (6 directions vs 4)
        assert n_qubits_3d > n_qubits_2d


class TestLatticeMapping:
    """Test full lattice mapping."""
    
    def test_map_to_lattice(self):
        """Test complete lattice mapping."""
        sequence = load_hp_sequence("HPHPH")
        encoding = map_to_lattice(sequence, lattice_dim=2)
        
        assert encoding.n_qubits > 0
        assert encoding.hamiltonian is not None
        assert encoding.sequence == sequence
    
    def test_hamiltonian_hermitian(self):
        """Test Hamiltonian is Hermitian."""
        sequence = load_hp_sequence("HPH")
        encoding = map_to_lattice(sequence, lattice_dim=2)
        
        H_matrix = encoding.hamiltonian.to_matrix()
        
        assert np.allclose(H_matrix, H_matrix.conj().T)
    
    def test_different_constraints(self):
        """Test different constraint weights."""
        sequence = load_hp_sequence("HPHPH")
        
        enc1 = map_to_lattice(sequence, constraint_weight=1.0)
        enc2 = map_to_lattice(sequence, constraint_weight=10.0)
        
        # Should produce different Hamiltonians
        assert not np.allclose(
            enc1.hamiltonian.to_matrix(),
            enc2.hamiltonian.to_matrix()
        )


class TestDecoding:
    """Test bitstring decoding."""
    
    def test_decode_turn_direction(self):
        """Test decoding turn-based encoding."""
        sequence = load_hp_sequence("HPH")
        encoding = map_to_lattice(
            sequence, lattice_dim=2, encoding_type='turn_direction'
        )
        
        # Simple bitstring (all zeros = straight line)
        n_bits = encoding.n_qubits
        bitstring = '0' * n_bits
        
        coords = decode_conformation(bitstring, encoding)
        
        assert coords.shape == (3, 2)  # 3 residues, 2D
    
    def test_decode_validity(self):
        """Test decoded conformation validity."""
        sequence = load_hp_sequence("HPH")
        encoding = map_to_lattice(sequence, lattice_dim=2)
        
        # Try several random bitstrings
        for _ in range(10):
            bitstring = ''.join(
                str(np.random.randint(2)) 
                for _ in range(encoding.n_qubits)
            )
            
            coords = decode_conformation(bitstring, encoding)
            
            # Should have correct shape
            assert coords.shape == (sequence.length, 2)


class TestValidation:
    """Test conformation validation."""
    
    def test_valid_straight_line(self):
        """Test valid straight-line conformation."""
        sequence = load_hp_sequence("HPH")
        coords = np.array([
            [0, 0],
            [1, 0],
            [2, 0]
        ])
        
        is_valid, msg = check_valid_conformation(coords, sequence)
        assert is_valid
    
    def test_invalid_overlap(self):
        """Test detection of overlapping residues."""
        sequence = load_hp_sequence("HPH")
        coords = np.array([
            [0, 0],
            [1, 0],
            [1, 0]  # Overlap!
        ])
        
        is_valid, msg = check_valid_conformation(coords, sequence)
        assert not is_valid
        assert "overlap" in msg.lower()
    
    def test_invalid_bond_length(self):
        """Test detection of invalid bond lengths."""
        sequence = load_hp_sequence("HPH")
        coords = np.array([
            [0, 0],
            [2, 0],  # Bond length = 2 (should be 1)
            [3, 0]
        ])
        
        is_valid, msg = check_valid_conformation(coords, sequence)
        assert not is_valid
        assert "length" in msg.lower()
