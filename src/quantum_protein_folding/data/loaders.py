"""Data loaders for protein sequences from various formats."""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union
import re

try:
    from Bio import SeqIO
    from Bio.PDB import PDBParser
    HAS_BIOPYTHON = True
except ImportError:
    HAS_BIOPYTHON = False

import numpy as np


@dataclass
class ProteinSequence:
    """Container for protein sequence data.
    
    Attributes:
        sequence: Amino acid sequence (single-letter codes)
        name: Sequence identifier
        source: Source file or database ID
        metadata: Additional metadata dictionary
    """
    sequence: str
    name: str
    source: str = ""
    metadata: dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        self.sequence = self.sequence.upper().strip()
    
    @property
    def length(self) -> int:
        """Return sequence length."""
        return len(self.sequence)
    
    def to_hp_model(self) -> str:
        """Convert to hydrophobic-polar (HP) model.
        
        H: Hydrophobic (A, V, I, L, M, F, Y, W)
        P: Polar/Hydrophilic (all others)
        """
        hydrophobic = set("AVIL MFYW")
        return "".join("H" if aa in hydrophobic else "P" for aa in self.sequence)
    
    def get_residue_types(self) -> np.ndarray:
        """Return array of residue type indices (0-19 for 20 amino acids)."""
        AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
        indices = [AA_ALPHABET.index(aa) if aa in AA_ALPHABET else 0 
                   for aa in self.sequence]
        return np.array(indices, dtype=np.int32)


def load_hp_sequence(sequence: str, name: str = "HP_seq") -> ProteinSequence:
    """Load hydrophobic-polar (HP) sequence.
    
    Args:
        sequence: String of 'H' (hydrophobic) and 'P' (polar) characters
        name: Sequence identifier
        
    Returns:
        ProteinSequence object
        
    Raises:
        ValueError: If sequence contains invalid characters
        
    Examples:
        >>> seq = load_hp_sequence("HPHPPHHPHH")
        >>> seq.length
        10
    """
    sequence = sequence.upper().strip()
    if not all(c in "HP" for c in sequence):
        raise ValueError(f"HP sequence must contain only 'H' and 'P' characters, got: {sequence}")
    
    return ProteinSequence(
        sequence=sequence,
        name=name,
        source="HP_model",
        metadata={"model": "hydrophobic_polar"}
    )


def load_fasta_sequence(
    filepath: Union[str, Path],
    index: int = 0
) -> ProteinSequence:
    """Load protein sequence from FASTA file.
    
    Args:
        filepath: Path to FASTA file
        index: Index of sequence to load (if multiple sequences)
        
    Returns:
        ProteinSequence object
        
    Raises:
        ImportError: If BioPython is not installed
        FileNotFoundError: If file does not exist
        IndexError: If index is out of range
    """
    if not HAS_BIOPYTHON:
        raise ImportError(
            "BioPython is required for FASTA loading. "
            "Install with: pip install biopython"
        )
    
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"FASTA file not found: {filepath}")
    
    sequences = list(SeqIO.parse(filepath, "fasta"))
    if not sequences:
        raise ValueError(f"No sequences found in {filepath}")
    
    if index >= len(sequences):
        raise IndexError(
            f"Index {index} out of range for {len(sequences)} sequences"
        )
    
    record = sequences[index]
    return ProteinSequence(
        sequence=str(record.seq),
        name=record.id,
        source=str(filepath),
        metadata={"description": record.description}
    )


def load_pdb_sequence(
    filepath: Union[str, Path],
    chain_id: str = "A"
) -> ProteinSequence:
    """Load protein sequence from PDB structure file.
    
    Args:
        filepath: Path to PDB file
        chain_id: Chain identifier to extract
        
    Returns:
        ProteinSequence object with sequence from specified chain
        
    Raises:
        ImportError: If BioPython is not installed
        FileNotFoundError: If file does not exist
        ValueError: If chain not found
    """
    if not HAS_BIOPYTHON:
        raise ImportError(
            "BioPython is required for PDB loading. "
            "Install with: pip install biopython"
        )
    
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"PDB file not found: {filepath}")
    
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(filepath.stem, filepath)
    
    # Three-letter to one-letter amino acid code mapping
    aa_dict = {
        'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
        'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
        'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
        'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
    }
    
    # Extract sequence from specified chain
    for model in structure:
        for chain in model:
            if chain.id == chain_id:
                sequence = ""
                for residue in chain:
                    if residue.id[0] == ' ':  # Standard residue
                        res_name = residue.get_resname()
                        if res_name in aa_dict:
                            sequence += aa_dict[res_name]
                
                if not sequence:
                    raise ValueError(f"No valid residues found in chain {chain_id}")
                
                return ProteinSequence(
                    sequence=sequence,
                    name=f"{filepath.stem}_{chain_id}",
                    source=str(filepath),
                    metadata={
                        "pdb_id": filepath.stem,
                        "chain_id": chain_id,
                        "n_residues": len(sequence)
                    }
                )
    
    raise ValueError(f"Chain '{chain_id}' not found in PDB structure")


def get_benchmark_sequences() -> dict:
    """Return dictionary of standard HP benchmark sequences.
    
    Returns:
        Dictionary mapping sequence names to HP sequences
    """
    return {
        "2d_6mer": "HPPHPP",
        "2d_8mer": "HPHPPHHP",
        "2d_10mer": "HPHPPHHPHH",
        "2d_20mer": "HPHPPHHPHPPHPHHPPHPH",
        "2d_24mer": "HHPPHPPHPPHPPHPPHPPHPPHH",
        "3d_8mer": "HPHPPHHP",
        "3d_13mer": "PPHPPHHPPHHPP",
        "3d_21mer": "PHHPPHHPPHHPPHHPPHHPP",
        "tortilla": "PPPHHPPHHPPPPPHHHHHHHPPHHPPPPHHPPHPP",
    }
