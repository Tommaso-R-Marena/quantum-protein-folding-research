"""Protein sequence loaders for various file formats.

Supports:
    - PDB files with BioPython parsing
    - FASTA format sequences
    - HP (Hydrophobic-Polar) model sequences
    - Real protein databases
"""

import os
from pathlib import Path
from typing import List, Optional, Union
import warnings

try:
    from Bio import SeqIO
    from Bio.PDB import PDBParser
    BIOPYTHON_AVAILABLE = True
except ImportError:
    BIOPYTHON_AVAILABLE = False
    warnings.warn(
        "BioPython not installed. PDB parsing will be limited. "
        "Install with: pip install biopython"
    )


class ProteinSequence:
    """Container for protein sequence data.
    
    Attributes:
        sequence: Single-letter amino acid codes
        name: Protein identifier
        source: Source file or database
        metadata: Additional information
    """
    
    def __init__(
        self,
        sequence: str,
        name: str = "unknown",
        source: str = "unknown",
        metadata: Optional[dict] = None
    ):
        self.sequence = sequence.upper()
        self.name = name
        self.source = source
        self.metadata = metadata or {}
    
    def __len__(self) -> int:
        return len(self.sequence)
    
    def __str__(self) -> str:
        return self.sequence
    
    def __repr__(self) -> str:
        return f"ProteinSequence(name='{self.name}', length={len(self)}, sequence='{self.sequence[:20]}...')"
    
    def to_hp_model(self) -> str:
        """Convert to hydrophobic-polar (HP) model.
        
        Hydrophobic residues: A, F, I, L, M, P, V, W
        Polar residues: All others
        
        Returns:
            HP sequence string
        """
        hydrophobic = set('AFILVMPW')
        return ''.join('H' if aa in hydrophobic else 'P' for aa in self.sequence)


def load_pdb_sequence(
    pdb_file: Union[str, Path],
    chain_id: Optional[str] = None
) -> ProteinSequence:
    """Load protein sequence from PDB file.
    
    Args:
        pdb_file: Path to PDB file
        chain_id: Specific chain to extract (None = first chain)
    
    Returns:
        ProteinSequence object
    
    Raises:
        FileNotFoundError: If PDB file doesn't exist
        ImportError: If BioPython not installed
        ValueError: If PDB parsing fails
    
    Example:
        >>> seq = load_pdb_sequence("data/raw/pdb/1N9L.pdb")
        >>> print(seq.sequence)
        DRVYIHPFHL
    """
    if not BIOPYTHON_AVAILABLE:
        raise ImportError(
            "BioPython required for PDB parsing. Install with: pip install biopython"
        )
    
    pdb_file = Path(pdb_file)
    if not pdb_file.exists():
        raise FileNotFoundError(f"PDB file not found: {pdb_file}")
    
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_file.stem, str(pdb_file))
    
    # Get first model
    model = structure[0]
    
    # Get specified chain or first chain
    if chain_id:
        if chain_id not in model:
            raise ValueError(f"Chain {chain_id} not found in PDB structure")
        chain = model[chain_id]
    else:
        chain = next(model.get_chains())
    
    # Extract sequence from residues
    three_to_one = {
        'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D',
        'CYS': 'C', 'GLN': 'Q', 'GLU': 'E', 'GLY': 'G',
        'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K',
        'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S',
        'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
    }
    
    sequence = []
    for residue in chain.get_residues():
        if residue.get_id()[0] == ' ':  # Standard residue
            res_name = residue.get_resname()
            if res_name in three_to_one:
                sequence.append(three_to_one[res_name])
    
    if not sequence:
        raise ValueError(f"No valid residues found in {pdb_file}")
    
    return ProteinSequence(
        sequence=''.join(sequence),
        name=pdb_file.stem,
        source=str(pdb_file),
        metadata={'chain_id': chain.get_id()}
    )


def load_fasta_sequence(
    fasta_file: Union[str, Path],
    index: int = 0
) -> ProteinSequence:
    """Load protein sequence from FASTA file.
    
    Args:
        fasta_file: Path to FASTA file
        index: Which sequence to load (for multi-FASTA files)
    
    Returns:
        ProteinSequence object
    
    Raises:
        FileNotFoundError: If FASTA file doesn't exist
        IndexError: If index out of range
    
    Example:
        >>> seq = load_fasta_sequence("data/raw/fasta/angiotensin.fasta")
        >>> print(seq.name)
        sp|P01019|ANGT_HUMAN
    """
    fasta_file = Path(fasta_file)
    if not fasta_file.exists():
        raise FileNotFoundError(f"FASTA file not found: {fasta_file}")
    
    if BIOPYTHON_AVAILABLE:
        # Use BioPython for robust parsing
        records = list(SeqIO.parse(str(fasta_file), "fasta"))
        if not records:
            raise ValueError(f"No sequences found in {fasta_file}")
        if index >= len(records):
            raise IndexError(f"Index {index} out of range (found {len(records)} sequences)")
        
        record = records[index]
        return ProteinSequence(
            sequence=str(record.seq),
            name=record.id,
            source=str(fasta_file),
            metadata={'description': record.description}
        )
    else:
        # Fallback: simple manual parsing
        with open(fasta_file, 'r') as f:
            lines = f.readlines()
        
        sequences = []
        current_name = ""
        current_seq = []
        
        for line in lines:
            line = line.strip()
            if line.startswith('>'):
                if current_seq:
                    sequences.append((current_name, ''.join(current_seq)))
                current_name = line[1:].split()[0]
                current_seq = []
            elif line:
                current_seq.append(line)
        
        if current_seq:
            sequences.append((current_name, ''.join(current_seq)))
        
        if not sequences:
            raise ValueError(f"No sequences found in {fasta_file}")
        if index >= len(sequences):
            raise IndexError(f"Index {index} out of range (found {len(sequences)} sequences)")
        
        name, sequence = sequences[index]
        return ProteinSequence(
            sequence=sequence,
            name=name,
            source=str(fasta_file)
        )


def load_hp_sequence(hp_string: str) -> ProteinSequence:
    """Load HP (Hydrophobic-Polar) model sequence.
    
    Args:
        hp_string: String of 'H' (hydrophobic) and 'P' (polar) characters
    
    Returns:
        ProteinSequence object
    
    Raises:
        ValueError: If string contains invalid characters
    
    Example:
        >>> seq = load_hp_sequence("HPHPPHHPHH")
        >>> len(seq)
        10
    """
    hp_string = hp_string.upper().strip()
    
    # Validate HP string
    valid_chars = set('HP')
    invalid = set(hp_string) - valid_chars
    if invalid:
        raise ValueError(
            f"Invalid characters in HP sequence: {invalid}. "
            f"Only 'H' and 'P' allowed."
        )
    
    return ProteinSequence(
        sequence=hp_string,
        name=f"HP_{len(hp_string)}",
        source="hp_model",
        metadata={'model': 'hydrophobic_polar'}
    )


# Standard HP benchmark sequences from literature
HP_BENCHMARKS = {
    "fibonacci_6": "HPHHPH",
    "fibonacci_13": "HPHPPHHPPHHPH",
    "fibonacci_21": "HPHPPHHPPHHPPHPPHHPH",
    "s1": "HPPHPPHPPHPPHPPHPPHP",  # 2D: E=-9
    "s2": "HPHPPHHPPHHPPHPPHHPH",  # 2D: E=-9
    "s3": "PPHPPHHPPHHPPHHPPHHPH",  # 3D: E=-8
    "s4": "PPHPPHHPPHHPPPPPPHHPPPPHHPPPHHHP",  # 3D: E=-14
}


def get_hp_benchmark(name: str) -> ProteinSequence:
    """Get standard HP benchmark sequence.
    
    Args:
        name: Benchmark name (e.g., 'fibonacci_13', 's1')
    
    Returns:
        ProteinSequence object
    
    Raises:
        KeyError: If benchmark name not found
    
    Example:
        >>> seq = get_hp_benchmark("fibonacci_13")
        >>> print(seq.sequence)
        HPHPPHHPPHHPH
    """
    if name not in HP_BENCHMARKS:
        available = ', '.join(HP_BENCHMARKS.keys())
        raise KeyError(
            f"Unknown benchmark '{name}'. Available: {available}"
        )
    
    return load_hp_sequence(HP_BENCHMARKS[name])
