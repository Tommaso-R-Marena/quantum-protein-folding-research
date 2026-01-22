"""Tests for data loading modules."""

import pytest
import numpy as np

from quantum_protein_folding.data.loaders import (
    load_hp_sequence,
    load_fasta_sequence,
    ProteinSequence,
)


class TestHPSequenceLoader:
    """Tests for HP model sequence loading."""
    
    def test_load_simple_sequence(self):
        """Test loading simple HP sequence."""
        seq = load_hp_sequence("HPHH")
        
        assert seq.sequence == "HPHH"
        assert seq.length == 4
        assert seq.model_type == "HP"
    
    def test_hp_contact_energies(self):
        """Test HP contact energy calculation."""
        seq = load_hp_sequence("HPH")
        
        # H-H contact
        energy_hh = seq.get_contact_energy(0, 2)
        assert energy_hh == -1.0
        
        # H-P contact
        energy_hp = seq.get_contact_energy(0, 1)
        assert energy_hp == 0.0
    
    def test_contact_matrix_symmetry(self):
        """Test contact matrix is symmetric."""
        seq = load_hp_sequence("HPHH")
        
        assert np.allclose(seq.contact_matrix, seq.contact_matrix.T)
    
    def test_invalid_sequence(self):
        """Test error handling for invalid sequence."""
        with pytest.raises(ValueError):
            load_hp_sequence("HXYZ")  # Invalid characters


class TestProteinSequence:
    """Tests for ProteinSequence class."""
    
    def test_sequence_creation(self):
        """Test direct sequence creation."""
        contact_matrix = np.array([
            [0, 0, -1],
            [0, 0, 0],
            [-1, 0, 0]
        ])
        
        seq = ProteinSequence(
            sequence="HPH",
            contact_matrix=contact_matrix,
            model_type="HP"
        )
        
        assert seq.length == 3
        assert seq.get_contact_energy(0, 2) == -1.0
    
    def test_sequence_length_property(self):
        """Test length property."""
        seq = load_hp_sequence("HPHPPHHPHH")
        assert seq.length == 10
        assert seq.length == len(seq.sequence)
