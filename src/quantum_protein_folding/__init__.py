"""Quantum Protein Folding: NISQ-compatible variational algorithms for structure prediction.

This package provides production-ready implementations of VQE and QAOA algorithms
for lattice-based protein folding problems.
"""

__version__ = "0.1.0"
__author__ = "Tommaso R. Marena"
__email__ = "tommaso.marena@example.com"

from quantum_protein_folding.models import VQEFoldingModel, QAOAFoldingModel
from quantum_protein_folding.data.loaders import (
    load_hp_sequence,
    load_pdb_sequence,
    load_fasta_sequence,
)

__all__ = [
    "VQEFoldingModel",
    "QAOAFoldingModel",
    "load_hp_sequence",
    "load_pdb_sequence",
    "load_fasta_sequence",
]
