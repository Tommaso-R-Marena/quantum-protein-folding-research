"""Classical energy calculation for protein conformations."""

import numpy as np
from typing import Optional

from quantum_protein_folding.data.loaders import ProteinSequence


def compute_energy(
    conformation: np.ndarray,
    sequence: ProteinSequence,
    lattice_dim: int = 2,
    constraint_weight: float = 10.0,
    bias_weight: float = 0.1
) -> float:
    """Compute total folding energy of a conformation.
    
    E = E_contact + lambda * E_backbone + mu * E_bias
    
    Args:
        conformation: (N, d) array of residue positions
        sequence: Protein sequence with contact matrix
        lattice_dim: Lattice dimension
        constraint_weight: Backbone constraint weight
        bias_weight: Compactness bias weight
        
    Returns:
        Total energy
    """
    e_contact = compute_contact_energy(conformation, sequence)
    e_backbone = compute_backbone_energy(conformation)
    e_bias = compute_bias_energy(conformation)
    
    total_energy = (
        e_contact +
        constraint_weight * e_backbone +
        bias_weight * e_bias
    )
    
    return total_energy


def compute_contact_energy(
    conformation: np.ndarray,
    sequence: ProteinSequence
) -> float:
    """Compute contact energy.
    
    Sum of pairwise contact energies for residues at distance 1.
    
    Args:
        conformation: Residue positions
        sequence: Sequence with contact matrix
        
    Returns:
        Contact energy
    """
    n = sequence.length
    energy = 0.0
    
    # Only non-local contacts (|i-j| > 2)
    for i in range(n - 3):
        for j in range(i + 3, n):
            # Check if in contact (lattice distance = 1)
            dist = np.linalg.norm(conformation[i] - conformation[j])
            
            if abs(dist - 1.0) < 1e-6:  # Contact
                energy += sequence.get_contact_energy(i, j)
    
    return energy


def compute_backbone_energy(
    conformation: np.ndarray
) -> float:
    """Compute backbone constraint violations.
    
    Penalizes:
    - Bond lengths != 1
    - Overlapping residues
    
    Args:
        conformation: Residue positions
        
    Returns:
        Constraint penalty
    """
    n = len(conformation)
    energy = 0.0
    
    # Bond length constraints
    for i in range(n - 1):
        bond_length = np.linalg.norm(conformation[i+1] - conformation[i])
        deviation = abs(bond_length - 1.0)
        energy += deviation ** 2
    
    # Self-avoidance
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(conformation[i] - conformation[j])
            if dist < 1e-6:  # Overlap
                energy += 1000.0  # Large penalty
    
    return energy


def compute_bias_energy(
    conformation: np.ndarray
) -> float:
    """Compute compactness bias.
    
    Sum of squared distances from origin.
    
    Args:
        conformation: Residue positions
        
    Returns:
        Bias energy
    """
    return np.sum(conformation ** 2)
