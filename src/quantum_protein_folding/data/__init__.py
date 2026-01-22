"""Data loading and preprocessing for protein sequences.

This module provides utilities for:
    - Loading protein sequences from PDB and FASTA files
    - Preprocessing sequences for lattice models
    - Miyazawa-Jernigan contact potential matrices
    - Hydrophobic-polar (HP) model sequences
"""

from quantum_protein_folding.data.loaders import (
    load_pdb_sequence,
    load_fasta_sequence,
    load_hp_sequence,
)
from quantum_protein_folding.data.preprocess import (
    map_to_lattice,
    encode_sequence,
    LatticeData,
)
from quantum_protein_folding.data.mj_matrix import get_mj_matrix

__all__ = [
    "load_pdb_sequence",
    "load_fasta_sequence",
    "load_hp_sequence",
    "map_to_lattice",
    "encode_sequence",
    "LatticeData",
    "get_mj_matrix",
]
