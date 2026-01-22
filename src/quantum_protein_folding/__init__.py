"""Quantum Protein Folding: NISQ-Compatible Variational Algorithms.

A production-ready implementation of quantum algorithms for protein structure
prediction on Near-term Intermediate-Scale Quantum (NISQ) devices.

Key Components:
    - VQE and QAOA implementations for lattice protein folding
    - Real protein data loaders (PDB, FASTA)
    - Miyazawa-Jernigan contact potentials
    - Classical baselines and benchmarking
    - Comprehensive analysis and visualization tools

Example:
    >>> from quantum_protein_folding.models import VQEFoldingModel
    >>> model = VQEFoldingModel(
    ...     sequence="HPHPPHHPHH",
    ...     lattice_dim=2,
    ...     lattice_size=5
    ... )
    >>> result = model.run(maxiter=200)
    >>> print(f"Energy: {result.optimal_value:.4f}")

Author: Tommaso R. Marena
Institution: The Catholic University of America
License: MIT
"""

__version__ = "1.0.0"
__author__ = "Tommaso R. Marena"
__email__ = "marena@cua.edu"

# Core imports for easy access
from quantum_protein_folding.quantum.vqe import VQE
from quantum_protein_folding.quantum.qaoa import QAOA
from quantum_protein_folding.quantum.hamiltonian import ProteinHamiltonian

__all__ = [
    "VQE",
    "QAOA",
    "ProteinHamiltonian",
    "__version__",
]
