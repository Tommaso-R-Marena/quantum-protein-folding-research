"""Classical baseline algorithms for protein folding."""

import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass
import itertools

from quantum_protein_folding.data.loaders import ProteinSequence
from quantum_protein_folding.data.preprocess import LatticeEncoding, check_valid_conformation
from quantum_protein_folding.classical.energy import compute_energy


@dataclass
class ClassicalFoldingResult:
    """Result of classical folding algorithm.
    
    Attributes:
        conformation: Optimal residue positions
        energy: Minimum energy found
        n_iterations: Number of iterations
        method: Algorithm name
    """
    conformation: np.ndarray
    energy: float
    n_iterations: int
    method: str


def simulated_annealing_fold(
    encoding: LatticeEncoding,
    max_iterations: int = 10000,
    initial_temp: float = 10.0,
    cooling_rate: float = 0.95,
    seed: Optional[int] = None
) -> ClassicalFoldingResult:
    """Simulated annealing for protein folding.
    
    Args:
        encoding: Lattice encoding with sequence
        max_iterations: Maximum iterations
        initial_temp: Initial temperature
        cooling_rate: Temperature decay rate
        seed: Random seed
        
    Returns:
        ClassicalFoldingResult with best conformation
    """
    if seed is not None:
        np.random.seed(seed)
    
    sequence = encoding.sequence
    lattice_dim = encoding.lattice_dim
    n = sequence.length
    
    # Random initial conformation
    current_conf = _generate_random_walk(n, lattice_dim)
    current_energy = compute_energy(current_conf, sequence, lattice_dim)
    
    best_conf = current_conf.copy()
    best_energy = current_energy
    
    temperature = initial_temp
    
    for iteration in range(max_iterations):
        # Propose move: randomly perturb one residue
        new_conf = _propose_move(current_conf, lattice_dim)
        
        # Check validity
        is_valid, _ = check_valid_conformation(new_conf, sequence)
        
        if not is_valid:
            continue  # Reject invalid moves
        
        # Compute energy
        new_energy = compute_energy(new_conf, sequence, lattice_dim)
        
        # Accept/reject
        delta_e = new_energy - current_energy
        
        if delta_e < 0 or np.random.rand() < np.exp(-delta_e / temperature):
            current_conf = new_conf
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_conf = current_conf.copy()
                best_energy = current_energy
        
        # Cool down
        temperature *= cooling_rate
    
    return ClassicalFoldingResult(
        conformation=best_conf,
        energy=best_energy,
        n_iterations=max_iterations,
        method='simulated_annealing'
    )


def exact_enumeration_fold(
    encoding: LatticeEncoding,
    max_conformations: int = 100000
) -> ClassicalFoldingResult:
    """Exact enumeration of all valid conformations.
    
    Only feasible for very short sequences (N <= 10).
    
    Args:
        encoding: Lattice encoding
        max_conformations: Maximum conformations to check
        
    Returns:
        ClassicalFoldingResult with optimal conformation
    """
    sequence = encoding.sequence
    lattice_dim = encoding.lattice_dim
    n = sequence.length
    
    if n > 10:
        raise ValueError(f"Exact enumeration infeasible for N={n} (max 10)")
    
    best_conf = None
    best_energy = float('inf')
    n_checked = 0
    
    # Generate all possible turn sequences
    n_directions = 2 * lattice_dim
    n_bonds = n - 1
    
    for turn_sequence in itertools.product(range(n_directions), repeat=n_bonds):
        if n_checked >= max_conformations:
            break
        
        # Build conformation from turns
        conf = _turns_to_conformation(turn_sequence, lattice_dim)
        
        # Check validity
        is_valid, _ = check_valid_conformation(conf, sequence)
        
        if not is_valid:
            continue
        
        # Compute energy
        energy = compute_energy(conf, sequence, lattice_dim)
        
        if energy < best_energy:
            best_energy = energy
            best_conf = conf.copy()
        
        n_checked += 1
    
    if best_conf is None:
        raise ValueError("No valid conformations found")
    
    return ClassicalFoldingResult(
        conformation=best_conf,
        energy=best_energy,
        n_iterations=n_checked,
        method='exact_enumeration'
    )


def _generate_random_walk(n: int, dim: int) -> np.ndarray:
    """Generate random self-avoiding walk."""
    directions_2d = [np.array([1, 0]), np.array([0, 1]), 
                     np.array([-1, 0]), np.array([0, -1])]
    directions_3d = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1]),
                     np.array([-1, 0, 0]), np.array([0, -1, 0]), np.array([0, 0, -1])]
    
    directions = directions_2d if dim == 2 else directions_3d
    
    conformation = np.zeros((n, dim), dtype=int)
    occupied = {tuple(conformation[0])}
    
    for i in range(1, n):
        # Try random directions until valid
        attempts = 0
        while attempts < 100:
            direction = directions[np.random.randint(len(directions))]
            new_pos = conformation[i-1] + direction
            
            if tuple(new_pos) not in occupied:
                conformation[i] = new_pos
                occupied.add(tuple(new_pos))
                break
            
            attempts += 1
        
        if attempts >= 100:
            # Restart if stuck
            return _generate_random_walk(n, dim)
    
    return conformation


def _propose_move(conformation: np.ndarray, dim: int) -> np.ndarray:
    """Propose a local move (corner flip or crankshaft)."""
    n = len(conformation)
    new_conf = conformation.copy()
    
    # Randomly choose a residue to move (not endpoints)
    if n <= 2:
        return new_conf
    
    idx = np.random.randint(1, n - 1)
    
    # Try a small displacement
    directions_2d = [np.array([1, 0]), np.array([0, 1]), 
                     np.array([-1, 0]), np.array([0, -1])]
    directions_3d = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1]),
                     np.array([-1, 0, 0]), np.array([0, -1, 0]), np.array([0, 0, -1])]
    
    directions = directions_2d if dim == 2 else directions_3d
    
    displacement = directions[np.random.randint(len(directions))]
    new_conf[idx] += displacement
    
    return new_conf


def _turns_to_conformation(turns: Tuple[int, ...], dim: int) -> np.ndarray:
    """Convert turn sequence to conformation."""
    n = len(turns) + 1
    
    if dim == 2:
        direction_map = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
        }
    else:
        direction_map = {
            0: np.array([1, 0, 0]),
            1: np.array([0, 1, 0]),
            2: np.array([0, 0, 1]),
            3: np.array([-1, 0, 0]),
            4: np.array([0, -1, 0]),
            5: np.array([0, 0, -1]),
        }
    
    conformation = np.zeros((n, dim), dtype=int)
    current_pos = np.zeros(dim, dtype=int)
    conformation[0] = current_pos
    
    for i, turn in enumerate(turns):
        current_pos = current_pos + direction_map[turn]
        conformation[i + 1] = current_pos
    
    return conformation
