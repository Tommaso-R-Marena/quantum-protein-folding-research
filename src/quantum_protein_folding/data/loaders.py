"""Data loaders for protein sequences from various formats.

This module provides functions to load protein sequences from:
- HP (Hydrophobic-Polar) model strings
- FASTA files
- PDB (Protein Data Bank) files

All loaders return standardized sequence representations with residue types
and contact potential matrices.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass


# Miyazawa-Jernigan (MJ) contact energy matrix (kJ/mol)
# Reduced 20x20 matrix for common amino acids
# Source: Miyazawa & Jernigan, Macromolecules 18, 534-552 (1985)
MJ_MATRIX = {
    ('A', 'A'): -0.62, ('A', 'C'): -1.11, ('A', 'D'): -0.75, ('A', 'E'): -0.91,
    ('A', 'F'): -1.36, ('A', 'G'): -0.60, ('A', 'H'): -1.05, ('A', 'I'): -1.30,
    ('A', 'K'): -0.84, ('A', 'L'): -1.24, ('A', 'M'): -1.17, ('A', 'N'): -0.76,
    ('A', 'P'): -0.69, ('A', 'Q'): -0.88, ('A', 'R'): -0.88, ('A', 'S'): -0.67,
    ('A', 'T'): -0.69, ('A', 'V'): -1.02, ('A', 'W'): -1.35, ('A', 'Y'): -1.06,
    
    ('C', 'C'): -3.52, ('C', 'D'): -2.41, ('C', 'E'): -2.27, ('C', 'F'): -2.66,
    ('C', 'G'): -1.52, ('C', 'H'): -2.16, ('C', 'I'): -2.59, ('C', 'K'): -1.95,
    ('C', 'L'): -2.59, ('C', 'M'): -2.95, ('C', 'N'): -1.88, ('C', 'P'): -1.66,
    ('C', 'Q'): -2.15, ('C', 'R'): -2.27, ('C', 'S'): -1.66, ('C', 'T'): -1.77,
    ('C', 'V'): -2.34, ('C', 'W'): -3.01, ('C', 'Y'): -2.41,
    
    ('D', 'D'): -0.56, ('D', 'E'): -0.52, ('D', 'F'): -1.21, ('D', 'G'): -0.58,
    ('D', 'H'): -0.80, ('D', 'I'): -1.02, ('D', 'K'): -0.35, ('D', 'L'): -1.08,
    ('D', 'M'): -1.06, ('D', 'N'): -0.52, ('D', 'P'): -0.44, ('D', 'Q'): -0.54,
    ('D', 'R'): -0.59, ('D', 'S'): -0.48, ('D', 'T'): -0.53, ('D', 'V'): -0.89,
    ('D', 'W'): -1.26, ('D', 'Y'): -0.93,
    
    ('E', 'E'): -0.44, ('E', 'F'): -1.30, ('E', 'G'): -0.67, ('E', 'H'): -0.83,
    ('E', 'I'): -1.16, ('E', 'K'): -0.37, ('E', 'L'): -1.18, ('E', 'M'): -1.20,
    ('E', 'N'): -0.59, ('E', 'P'): -0.52, ('E', 'Q'): -0.60, ('E', 'R'): -0.61,
    ('E', 'S'): -0.56, ('E', 'T'): -0.62, ('E', 'V'): -1.00, ('E', 'W'): -1.37,
    ('E', 'Y'): -1.04,
    
    ('F', 'F'): -2.17, ('F', 'G'): -1.15, ('F', 'H'): -1.61, ('F', 'I'): -1.96,
    ('F', 'K'): -1.24, ('F', 'L'): -1.95, ('F', 'M'): -2.12, ('F', 'N'): -1.21,
    ('F', 'P'): -1.15, ('F', 'Q'): -1.37, ('F', 'R'): -1.38, ('F', 'S'): -1.08,
    ('F', 'T'): -1.21, ('F', 'V'): -1.71, ('F', 'W'): -2.25, ('F', 'Y'): -1.87,
    
    ('G', 'G'): -0.41, ('G', 'H'): -0.79, ('G', 'I'): -0.96, ('G', 'K'): -0.62,
    ('G', 'L'): -1.02, ('G', 'M'): -0.98, ('G', 'N'): -0.56, ('G', 'P'): -0.42,
    ('G', 'Q'): -0.65, ('G', 'R'): -0.69, ('G', 'S'): -0.52, ('G', 'T'): -0.56,
    ('G', 'V'): -0.81, ('G', 'W'): -1.18, ('G', 'Y'): -0.92,
    
    ('H', 'H'): -1.35, ('H', 'I'): -1.44, ('H', 'K'): -0.78, ('H', 'L'): -1.49,
    ('H', 'M'): -1.60, ('H', 'N'): -0.78, ('H', 'P'): -0.72, ('H', 'Q'): -0.93,
    ('H', 'R'): -0.87, ('H', 'S'): -0.74, ('H', 'T'): -0.82, ('H', 'V'): -1.23,
    ('H', 'W'): -1.74, ('H', 'Y'): -1.39,
    
    ('I', 'I'): -1.82, ('I', 'K'): -1.08, ('I', 'L'): -1.87, ('I', 'M'): -1.94,
    ('I', 'N'): -1.02, ('I', 'P'): -0.95, ('I', 'Q'): -1.19, ('I', 'R'): -1.15,
    ('I', 'S'): -0.94, ('I', 'T'): -1.05, ('I', 'V'): -1.56, ('I', 'W'): -2.09,
    ('I', 'Y'): -1.58,
    
    ('K', 'K'): -0.34, ('K', 'L'): -1.15, ('K', 'M'): -1.15, ('K', 'N'): -0.44,
    ('K', 'P'): -0.48, ('K', 'Q'): -0.53, ('K', 'R'): -0.41, ('K', 'S'): -0.47,
    ('K', 'T'): -0.52, ('K', 'V'): -0.97, ('K', 'W'): -1.35, ('K', 'Y'): -1.02,
    
    ('L', 'L'): -1.92, ('L', 'M'): -2.04, ('L', 'N'): -1.08, ('L', 'P'): -1.02,
    ('L', 'Q'): -1.23, ('L', 'R'): -1.22, ('L', 'S'): -0.99, ('L', 'T'): -1.08,
    ('L', 'V'): -1.63, ('L', 'W'): -2.16, ('L', 'Y'): -1.64,
    
    ('M', 'M'): -2.49, ('M', 'N'): -1.13, ('M', 'P'): -1.05, ('M', 'Q'): -1.30,
    ('M', 'R'): -1.27, ('M', 'S'): -1.03, ('M', 'T'): -1.13, ('M', 'V'): -1.70,
    ('M', 'W'): -2.45, ('M', 'Y'): -1.80,
    
    ('N', 'N'): -0.48, ('N', 'P'): -0.41, ('N', 'Q'): -0.56, ('N', 'R'): -0.54,
    ('N', 'S'): -0.47, ('N', 'T'): -0.50, ('N', 'V'): -0.85, ('N', 'W'): -1.19,
    ('N', 'Y'): -0.89,
    
    ('P', 'P'): -0.31, ('P', 'Q'): -0.53, ('P', 'R'): -0.57, ('P', 'S'): -0.42,
    ('P', 'T'): -0.46, ('P', 'V'): -0.76, ('P', 'W'): -1.13, ('P', 'Y'): -0.85,
    
    ('Q', 'Q'): -0.62, ('Q', 'R'): -0.64, ('Q', 'S'): -0.55, ('Q', 'T'): -0.60,
    ('Q', 'V'): -1.04, ('Q', 'W'): -1.49, ('Q', 'Y'): -1.11,
    
    ('R', 'R'): -0.45, ('R', 'S'): -0.55, ('R', 'T'): -0.60, ('R', 'V'): -1.03,
    ('R', 'W'): -1.53, ('R', 'Y'): -1.11,
    
    ('S', 'S'): -0.40, ('S', 'T'): -0.46, ('S', 'V'): -0.75, ('S', 'W'): -1.02,
    ('S', 'Y'): -0.80,
    
    ('T', 'T'): -0.49, ('T', 'V'): -0.83, ('T', 'W'): -1.17, ('T', 'Y'): -0.89,
    
    ('V', 'V'): -1.35, ('V', 'W'): -1.88, ('V', 'Y'): -1.42,
    
    ('W', 'W'): -2.85, ('W', 'Y'): -2.17,
    
    ('Y', 'Y'): -1.74,
}

# Make MJ matrix symmetric
MJ_MATRIX_FULL = {}
for (a, b), energy in MJ_MATRIX.items():
    MJ_MATRIX_FULL[(a, b)] = energy
    MJ_MATRIX_FULL[(b, a)] = energy


# HP model: Simplified hydrophobic-polar representation
HP_CONTACT_ENERGY = {
    ('H', 'H'): -2.3,  # Hydrophobic-hydrophobic attraction
    ('H', 'P'): 0.0,   # No interaction
    ('P', 'H'): 0.0,
    ('P', 'P'): 0.0,
}


@dataclass
class ProteinSequence:
    """Standardized protein sequence representation.
    
    Attributes:
        sequence: Amino acid sequence as single-letter codes
        residue_types: List of residue types
        contact_matrix: N×N matrix of contact energies
        metadata: Additional sequence information (PDB ID, etc.)
    """
    sequence: str
    residue_types: List[str]
    contact_matrix: np.ndarray
    metadata: Dict[str, any]
    
    @property
    def length(self) -> int:
        return len(self.sequence)
    
    def get_contact_energy(self, i: int, j: int) -> float:
        """Get contact energy between residues i and j."""
        return self.contact_matrix[i, j]


def load_hp_sequence(sequence: str) -> ProteinSequence:
    """Load hydrophobic-polar (HP) lattice model sequence.
    
    Args:
        sequence: String of 'H' (hydrophobic) and 'P' (polar) residues
        
    Returns:
        ProteinSequence with HP contact energies
        
    Example:
        >>> seq = load_hp_sequence("HPHPPHHPHH")
        >>> seq.length
        10
        >>> seq.get_contact_energy(0, 2)
        0.0
    """
    sequence = sequence.upper().strip()
    
    # Validate sequence
    if not all(c in 'HP' for c in sequence):
        raise ValueError(f"Invalid HP sequence: {sequence}. Only 'H' and 'P' allowed.")
    
    n = len(sequence)
    residue_types = list(sequence)
    
    # Build contact energy matrix
    contact_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            ri, rj = sequence[i], sequence[j]
            contact_matrix[i, j] = HP_CONTACT_ENERGY[(ri, rj)]
            contact_matrix[j, i] = contact_matrix[i, j]
    
    return ProteinSequence(
        sequence=sequence,
        residue_types=residue_types,
        contact_matrix=contact_matrix,
        metadata={'model': 'HP', 'source': 'string'}
    )


def load_fasta_sequence(
    fasta_path: str,
    potential_type: str = 'miyazawa_jernigan'
) -> ProteinSequence:
    """Load protein sequence from FASTA file.
    
    Args:
        fasta_path: Path to FASTA file
        potential_type: Contact potential type ('miyazawa_jernigan' or 'hp')
        
    Returns:
        ProteinSequence with specified contact energies
        
    Example:
        >>> seq = load_fasta_sequence('data/raw/fasta/1N9L.fasta')
        >>> seq.metadata['pdb_id']
        '1N9L'
    """
    fasta_path = Path(fasta_path)
    
    if not fasta_path.exists():
        raise FileNotFoundError(f"FASTA file not found: {fasta_path}")
    
    # Parse FASTA
    with open(fasta_path, 'r') as f:
        lines = f.readlines()
    
    header = lines[0].strip()
    sequence = ''.join(line.strip() for line in lines[1:])
    sequence = sequence.upper()
    
    # Extract PDB ID from header if present
    pdb_id = None
    if header.startswith('>'):
        parts = header[1:].split('|')
        if len(parts) >= 2:
            pdb_id = parts[1].strip()
    
    # Build contact matrix based on potential type
    n = len(sequence)
    residue_types = list(sequence)
    
    if potential_type == 'miyazawa_jernigan':
        contact_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                ri, rj = sequence[i], sequence[j]
                if (ri, rj) in MJ_MATRIX_FULL:
                    contact_matrix[i, j] = MJ_MATRIX_FULL[(ri, rj)]
                else:
                    # Use average MJ energy for unknown residue pairs
                    contact_matrix[i, j] = -1.0
                contact_matrix[j, i] = contact_matrix[i, j]
    elif potential_type == 'hp':
        # Map to HP model (crude approximation)
        hp_sequence = _map_to_hp(sequence)
        contact_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                ri, rj = hp_sequence[i], hp_sequence[j]
                contact_matrix[i, j] = HP_CONTACT_ENERGY[(ri, rj)]
                contact_matrix[j, i] = contact_matrix[i, j]
    else:
        raise ValueError(f"Unknown potential type: {potential_type}")
    
    return ProteinSequence(
        sequence=sequence,
        residue_types=residue_types,
        contact_matrix=contact_matrix,
        metadata={
            'source': 'fasta',
            'file': str(fasta_path),
            'pdb_id': pdb_id,
            'potential_type': potential_type,
        }
    )


def load_pdb_sequence(
    pdb_path: str,
    chain_id: Optional[str] = None,
    potential_type: str = 'miyazawa_jernigan'
) -> ProteinSequence:
    """Load protein sequence from PDB file.
    
    Args:
        pdb_path: Path to PDB file
        chain_id: Specific chain to extract (default: first chain)
        potential_type: Contact potential type
        
    Returns:
        ProteinSequence with 3D structure metadata
    """
    pdb_path = Path(pdb_path)
    
    if not pdb_path.exists():
        raise FileNotFoundError(f"PDB file not found: {pdb_path}")
    
    # Simple PDB parser (for production, use BioPython)
    sequence = []
    ca_coords = []
    current_chain = None
    
    # Standard 3-letter to 1-letter amino acid codes
    aa_codes = {
        'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
        'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
        'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
        'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
    }
    
    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith('ATOM'):
                atom_name = line[12:16].strip()
                res_name = line[17:20].strip()
                chain = line[21].strip()
                
                # Select chain
                if chain_id is None and current_chain is None:
                    current_chain = chain
                elif chain_id is not None:
                    current_chain = chain_id
                
                if chain != current_chain:
                    continue
                
                # Extract CA (alpha-carbon) atoms
                if atom_name == 'CA' and res_name in aa_codes:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    
                    sequence.append(aa_codes[res_name])
                    ca_coords.append([x, y, z])
    
    if not sequence:
        raise ValueError(f"No valid protein sequence found in {pdb_path}")
    
    sequence_str = ''.join(sequence)
    ca_coords = np.array(ca_coords)
    
    # Build contact matrix
    n = len(sequence_str)
    residue_types = list(sequence_str)
    
    if potential_type == 'miyazawa_jernigan':
        contact_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                ri, rj = sequence_str[i], sequence_str[j]
                if (ri, rj) in MJ_MATRIX_FULL:
                    contact_matrix[i, j] = MJ_MATRIX_FULL[(ri, rj)]
                else:
                    contact_matrix[i, j] = -1.0
                contact_matrix[j, i] = contact_matrix[i, j]
    else:
        raise ValueError(f"Unsupported potential type for PDB: {potential_type}")
    
    return ProteinSequence(
        sequence=sequence_str,
        residue_types=residue_types,
        contact_matrix=contact_matrix,
        metadata={
            'source': 'pdb',
            'file': str(pdb_path),
            'chain_id': current_chain,
            'ca_coordinates': ca_coords,
            'potential_type': potential_type,
        }
    )


def _map_to_hp(sequence: str) -> str:
    """Map amino acid sequence to HP model.
    
    Hydrophobic residues: A, F, I, L, M, V, W, Y
    Polar residues: C, D, E, G, H, K, N, P, Q, R, S, T
    """
    hydrophobic = set('AFILMVWY')
    hp_seq = ''
    for aa in sequence:
        hp_seq += 'H' if aa in hydrophobic else 'P'
    return hp_seq
