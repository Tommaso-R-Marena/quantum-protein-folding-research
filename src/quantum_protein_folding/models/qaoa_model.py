"""QAOA-based protein folding model API."""

import numpy as np
from typing import Union, Optional, Dict

from quantum_protein_folding.data.loaders import (
    ProteinSequence,
    load_hp_sequence,
)
from quantum_protein_folding.data.preprocess import (
    map_to_lattice,
    decode_conformation,
)
from quantum_protein_folding.quantum.qaoa import QAOASolver, QAOAResult


class QAOAFoldingModel:
    """High-level API for QAOA-based protein folding.
    
    Example:
        >>> model = QAOAFoldingModel(
        ...     sequence="HPHPPHHPHH",
        ...     p_layers=3,
        ...     lattice_dim=2
        ... )
        >>> result = model.run(maxiter=100)
        >>> print(f"Best solution: {result.optimal_bitstring}")
    """
    
    def __init__(
        self,
        sequence: Union[str, ProteinSequence],
        p_layers: int = 1,
        lattice_dim: int = 2,
        lattice_size: Optional[int] = None,
        encoding_type: str = 'turn_direction',
        optimizer: str = 'COBYLA',
        backend: str = 'aer_simulator',
        shots: int = 1024,
    ):
        """Initialize QAOA folding model.
        
        Args:
            sequence: Protein sequence
            p_layers: Number of QAOA layers
            lattice_dim: Lattice dimension
            lattice_size: Lattice size
            encoding_type: Encoding scheme
            optimizer: Classical optimizer
            backend: Quantum backend
            shots: Measurement shots
        """
        # Load sequence
        if isinstance(sequence, str):
            self.sequence = load_hp_sequence(sequence)
        else:
            self.sequence = sequence
        
        # Encode to lattice
        self.encoding = map_to_lattice(
            self.sequence,
            lattice_dim=lattice_dim,
            lattice_size=lattice_size,
            encoding_type=encoding_type,
        )
        
        # Create QAOA solver
        self.solver = QAOASolver(
            hamiltonian=self.encoding.hamiltonian,
            p_layers=p_layers,
            optimizer=optimizer,
            backend=backend,
            shots=shots,
        )
    
    def run(
        self,
        maxiter: int = 100,
        initial_params: Optional[np.ndarray] = None
    ) -> QAOAResult:
        """Run QAOA optimization.
        
        Args:
            maxiter: Maximum iterations
            initial_params: Initial parameter guess
            
        Returns:
            QAOAResult with optimal solution
        """
        return self.solver.run(maxiter=maxiter, initial_params=initial_params)
    
    def decode_conformation(self, bitstring: str) -> np.ndarray:
        """Decode bitstring to coordinates."""
        return decode_conformation(bitstring, self.encoding)
    
    def evaluate_energy(self, conformation: np.ndarray) -> float:
        """Evaluate energy of conformation."""
        from quantum_protein_folding.classical.energy import compute_energy
        return compute_energy(conformation, self.sequence, self.encoding.lattice_dim)
