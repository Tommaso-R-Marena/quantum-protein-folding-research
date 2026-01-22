"""VQE-based protein folding model API."""

import numpy as np
from typing import Union, Optional
from dataclasses import dataclass

from quantum_protein_folding.data.loaders import (
    ProteinSequence,
    load_hp_sequence,
    load_fasta_sequence,
    load_pdb_sequence,
)
from quantum_protein_folding.data.preprocess import (
    map_to_lattice,
    decode_conformation,
    check_valid_conformation,
)
from quantum_protein_folding.quantum.vqe import VQESolver, VQEResult


class VQEFoldingModel:
    """High-level API for VQE-based protein folding.
    
    Example:
        >>> model = VQEFoldingModel(
        ...     sequence="HPHPPHHPHH",
        ...     lattice_dim=2,
        ...     lattice_size=5,
        ...     ansatz_type="hardware_efficient",
        ...     ansatz_depth=3
        ... )
        >>> result = model.run(maxiter=200)
        >>> print(f"Energy: {result.optimal_value:.4f}")
    """
    
    def __init__(
        self,
        sequence: Union[str, ProteinSequence],
        lattice_dim: int = 2,
        lattice_size: Optional[int] = None,
        encoding_type: str = 'turn_direction',
        ansatz_type: str = 'hardware_efficient',
        ansatz_depth: int = 3,
        optimizer: str = 'COBYLA',
        backend: str = 'aer_simulator',
        shots: int = 1024,
        constraint_weight: float = 10.0,
        bias_weight: float = 0.1,
    ):
        """Initialize VQE folding model.
        
        Args:
            sequence: Protein sequence (string or ProteinSequence)
            lattice_dim: Lattice dimension (2 or 3)
            lattice_size: Lattice size (auto if None)
            encoding_type: 'turn_direction' or 'binary_position'
            ansatz_type: 'hardware_efficient' or 'problem_inspired'
            ansatz_depth: Number of ansatz layers
            optimizer: Classical optimizer
            backend: Quantum backend
            shots: Measurement shots
            constraint_weight: Backbone constraint weight
            bias_weight: Compactness bias weight
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
            constraint_weight=constraint_weight,
            bias_weight=bias_weight,
        )
        
        # Create VQE solver
        self.solver = VQESolver(
            hamiltonian=self.encoding.hamiltonian,
            n_qubits=self.encoding.n_qubits,
            ansatz_type=ansatz_type,
            ansatz_depth=ansatz_depth,
            optimizer=optimizer,
            backend=backend,
            shots=shots,
        )
    
    def run(self, maxiter: int = 200) -> VQEResult:
        """Run VQE optimization.
        
        Args:
            maxiter: Maximum iterations
            
        Returns:
            VQEResult with optimized parameters
        """
        return self.solver.run(maxiter=maxiter)
    
    def decode_conformation(self, bitstring: str) -> np.ndarray:
        """Decode bitstring to 3D coordinates.
        
        Args:
            bitstring: Measurement outcome
            
        Returns:
            (N, d) array of residue positions
        """
        return decode_conformation(bitstring, self.encoding)
    
    def evaluate_energy(self, conformation: np.ndarray) -> float:
        """Evaluate folding energy of a conformation.
        
        Args:
            conformation: (N, d) residue positions
            
        Returns:
            Total energy
        """
        from quantum_protein_folding.classical.energy import compute_energy
        
        return compute_energy(conformation, self.sequence, self.encoding.lattice_dim)
    
    def validate_conformation(self, conformation: np.ndarray) -> bool:
        """Check if conformation satisfies lattice constraints.
        
        Args:
            conformation: (N, d) residue positions
            
        Returns:
            True if valid
        """
        is_valid, _ = check_valid_conformation(conformation, self.sequence)
        return is_valid
