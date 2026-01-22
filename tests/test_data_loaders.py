"""Tests for data loading modules."""

import pytest
import numpy as np
from quantum_protein_folding.data.loaders import (
    load_hp_sequence,
    load_fasta_sequence,
    ProteinSequence,
)
from quantum_protein_folding.data.contact_potentials import MiyazawaJernigan


class TestHPSequence:
    """Test HP sequence loading."""
    
    def test_load_simple_sequence(self):
        """Test loading a simple HP sequence."""
        seq = load_hp_sequence("HPHPPH")
        
        assert seq.length == 6
        assert seq.sequence == "HPHPPH"
        assert seq.source == "HP"
    
    def test_contact_matrix_symmetric(self):
        """Contact matrix should be symmetric."""
        seq = load_hp_sequence("HPHPPH")
        
        # Check symmetry
        assert np.allclose(seq.contact_matrix, seq.contact_matrix.T)
    
    def test_contact_energy_hh(self):
        """H-H contacts should have negative energy."""
        seq = load_hp_sequence("HH")
        energy = seq.get_contact_energy(0, 1)
        
        # HP model: H-H = -1, others = 0
        assert energy == -1.0
    
    def test_contact_energy_pp(self):
        """P-P contacts should have zero energy."""
        seq = load_hp_sequence("PP")
        energy = seq.get_contact_energy(0, 1)
        
        assert energy == 0.0
    
    def test_invalid_sequence(self):
        """Invalid characters should raise error."""
        with pytest.raises(ValueError):
            load_hp_sequence("HPXPH")  # X is invalid


class TestFASTASequence:
    """Test FASTA sequence loading."""
    
    def test_single_sequence(self):
        """Test loading single FASTA entry."""
        fasta_content = ">TestProtein\nMKLLIVLCLAVAALA\n"
        
        # Create temporary file
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as f:
            f.write(fasta_content)
            fasta_path = f.name
        
        try:
            seq = load_fasta_sequence(fasta_path)
            assert seq.length == 15
            assert seq.sequence == "MKLLIVLCLAVAALA"
        finally:
            import os
            os.unlink(fasta_path)


class TestMJPotential:
    """Test Miyazawa-Jernigan contact potentials."""
    
    def test_mj_matrix_symmetric(self):
        """MJ matrix should be symmetric."""
        mj = MiyazawaJernigan()
        
        for aa1 in mj.amino_acids:
            for aa2 in mj.amino_acids:
                energy1 = mj.get_contact_energy(aa1, aa2)
                energy2 = mj.get_contact_energy(aa2, aa1)
                assert abs(energy1 - energy2) < 1e-10
    
    def test_known_values(self):
        """Test some known MJ values."""
        mj = MiyazawaJernigan()
        
        # Hydrophobic pairs should be favorable (negative)
        ile_ile = mj.get_contact_energy('I', 'I')
        assert ile_ile < 0
        
        # Charged pairs of opposite sign should be favorable
        lys_glu = mj.get_contact_energy('K', 'E')
        assert lys_glu < 0
    
    def test_invalid_amino_acid(self):
        """Invalid amino acid should raise error."""
        mj = MiyazawaJernigan()
        
        with pytest.raises(ValueError):
            mj.get_contact_energy('X', 'A')  # X is invalid
