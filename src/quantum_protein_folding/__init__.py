"""Quantum Protein Folding: NISQ-Compatible Variational Algorithms.

A production-ready framework for quantum protein structure prediction using
Variational Quantum Eigensolver (VQE) and Quantum Approximate Optimization
Algorithm (QAOA) on near-term quantum devices.
"""

__version__ = "0.1.0"
__author__ = "Tommaso R. Marena"
__email__ = "marena@cua.edu"

from quantum_protein_folding.models.vqe_model import VQEFoldingModel
from quantum_protein_folding.models.qaoa_model import QAOAFoldingModel
from quantum_protein_folding.data.loaders import (
    load_hp_sequence,
    load_fasta_sequence,
    load_pdb_sequence,
)

__all__ = [
    "VQEFoldingModel",
    "QAOAFoldingModel",
    "load_hp_sequence",
    "load_fasta_sequence",
    "load_pdb_sequence",
]
