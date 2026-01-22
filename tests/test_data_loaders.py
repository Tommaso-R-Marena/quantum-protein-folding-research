"""Tests for data loading functionality."""

import pytest
import numpy as np
from quantum_protein_folding.data.loaders import (
    load_hp_sequence,
    load_fasta_sequence,
    ProteinSequence,
)
from quantum_protein_folding.data.mj_matrix import MJ_MATRIX


class TestHPLoader:
    """Test HP sequence loading."""
    
    def test_load_simple_sequence(self):
        """Test loading basic HP sequence."""
        seq = load_hp_sequence("HPHPH")
        
        assert seq.length == 5
        assert seq.sequence == "HPHPH"
        assert seq.contact_matrix.shape == (5, 5)
    
    def test_contact_energies(self):
        """Test HP contact energy values."""
        seq = load_hp_sequence("HPH")
        
        # H-H contact should be -1
        assert seq.get_contact_energy(0, 2) == -1.0
        
        # H-P or P-H should be 0
        assert seq.get_contact_energy(0, 1) == 0.0
        assert seq.get_contact_energy(1, 2) == 0.0
    
    def test_invalid_sequence(self):
        """Test error on invalid characters."""
        with pytest.raises(ValueError):
            load_hp_sequence("HPXH")  # X is invalid
    
    def test_empty_sequence(self):
        """Test empty sequence handling."""
        with pytest.raises(ValueError):
            load_hp_sequence("")


class TestProteinSequence:
    """Test ProteinSequence class."""
    
    def test_initialization(self):
        """Test sequence initialization."""
        contact_matrix = np.array([
            [0, 0, -1],
            [0, 0, 0],
            [-1, 0, 0]
        ])
        
        seq = ProteinSequence(
            sequence="HPH",
            contact_matrix=contact_matrix
        )
        
        assert seq.length == 3
        assert seq.sequence == "HPH"
    
    def test_contact_matrix_symmetry(self):
        """Test contact matrix is symmetric."""
        seq = load_hp_sequence("HPHPH")
        
        assert np.allclose(
            seq.contact_matrix,
            seq.contact_matrix.T
        )
    
    def test_diagonal_zero(self):
        """Test contact matrix diagonal is zero."""
        seq = load_hp_sequence("HPHPH")
        
        assert np.allclose(
            np.diag(seq.contact_matrix),
            np.zeros(seq.length)
        )


class TestMJMatrix:
    """Test Miyazawa-Jernigan matrix."""
    
    def test_mj_symmetry(self):
        """Test MJ matrix is symmetric."""
        for aa1 in MJ_MATRIX:
            for aa2 in MJ_MATRIX:
                assert MJ_MATRIX[aa1][aa2] == MJ_MATRIX[aa2][aa1]
    
    def test_mj_values(self):
        """Test some known MJ values."""
        # Hydrophobic pairs should be negative
        assert MJ_MATRIX['L']['L'] < 0
        assert MJ_MATRIX['I']['V'] < 0
        
        # Values should be in reasonable range
        for aa1 in MJ_MATRIX:
            for aa2 in MJ_MATRIX:
                energy = MJ_MATRIX[aa1][aa2]
                assert -10 <= energy <= 5
