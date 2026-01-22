"""Tests for data loaders."""

import pytest
import numpy as np
from quantum_protein_folding.data.loaders import (
    load_hp_sequence,
    load_fasta_sequence,
    ProteinSequence,
)


class TestHPLoader:
    """Test HP model sequence loading."""
    
    def test_load_hp_sequence(self):
        """Test basic HP sequence loading."""
        seq = load_hp_sequence("HPHP")
        
        assert seq.sequence == "HPHP"
        assert seq.length == 4
        assert seq.model == "HP"
    
    def test_contact_matrix_shape(self):
        """Test contact matrix dimensions."""
        seq = load_hp_sequence("HPHPPHHP")
        
        assert seq.contact_matrix.shape == (8, 8)
        assert np.allclose(seq.contact_matrix, seq.contact_matrix.T)  # Symmetric
    
    def test_contact_energies(self):
        """Test HP contact energy values."""
        seq = load_hp_sequence("HP")
        
        # H-H contact should be negative (favorable)
        hh_energy = seq.get_contact_energy(0, 0)  # Both H
        assert hh_energy < 0
        
        # H-P contact should be less favorable
        hp_energy = seq.get_contact_energy(0, 1)
        assert hp_energy > hh_energy
    
    def test_invalid_sequence(self):
        """Test that invalid characters raise error."""
        with pytest.raises(ValueError):
            load_hp_sequence("HPXP")  # X is not valid


class TestProteinSequence:
    """Test ProteinSequence dataclass."""
    
    def test_sequence_creation(self):
        """Test manual sequence creation."""
        contact_matrix = np.array([[0, -1], [-1, 0]])
        
        seq = ProteinSequence(
            sequence="HP",
            contact_matrix=contact_matrix,
            model="HP"
        )
        
        assert seq.length == 2
        assert seq.get_contact_energy(0, 1) == -1
