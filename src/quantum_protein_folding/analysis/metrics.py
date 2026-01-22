"""Metrics for protein folding analysis."""

import numpy as np
from typing import List, Dict, Tuple


def compute_rmsd(
    conformation1: np.ndarray,
    conformation2: np.ndarray,
    align: bool = True
) -> float:
    """Compute Root Mean Square Deviation between two conformations.
    
    RMSD = sqrt(1/N * sum_i ||r1_i - r2_i||^2)
    
    Args:
        conformation1: First conformation (N, d)
        conformation2: Second conformation (N, d)
        align: Whether to align structures first
        
    Returns:
        RMSD value in lattice units
    """
    conf1 = conformation1.astype(float)
    conf2 = conformation2.astype(float)
    
    if conf1.shape != conf2.shape:
        raise ValueError(f"Shape mismatch: {conf1.shape} vs {conf2.shape}")
    
    n = len(conf1)
    
    if align:
        # Center both structures
        conf1 = conf1 - np.mean(conf1, axis=0)
        conf2 = conf2 - np.mean(conf2, axis=0)
        
        # Optimal rotation via SVD (Kabsch algorithm)
        H = conf1.T @ conf2
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        
        # Ensure proper rotation (det = 1)
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        
        conf1 = conf1 @ R
    
    # Compute RMSD
    squared_deviations = np.sum((conf1 - conf2) ** 2, axis=1)
    rmsd = np.sqrt(np.mean(squared_deviations))
    
    return float(rmsd)


def compute_energy_gap(
    quantum_energy: float,
    classical_energy: float
) -> float:
    """Compute energy gap between quantum and classical solutions.
    
    Args:
        quantum_energy: Energy from quantum algorithm
        classical_energy: Energy from classical baseline
        
    Returns:
        Relative energy gap
    """
    if abs(classical_energy) < 1e-10:
        return abs(quantum_energy - classical_energy)
    
    gap = (quantum_energy - classical_energy) / abs(classical_energy)
    return float(gap)


def compute_overlap(
    state1: np.ndarray,
    state2: np.ndarray
) -> float:
    """Compute quantum state overlap |<psi1|psi2>|^2.
    
    Args:
        state1: First state vector
        state2: Second state vector
        
    Returns:
        Overlap (fidelity)
    """
    overlap = np.abs(np.vdot(state1, state2)) ** 2
    return float(overlap)


def analyze_convergence(
    history: List[float]
) -> Dict[str, float]:
    """Analyze convergence from optimization history.
    
    Args:
        history: List of objective values vs iteration
        
    Returns:
        Dictionary with convergence metrics
    """
    if not history:
        return {}
    
    history_array = np.array(history)
    
    # Final value
    final_value = history_array[-1]
    
    # Best value
    best_value = np.min(history_array)
    
    # Convergence rate (exponential fit)
    n = len(history_array)
    if n > 10:
        # Fit to a * exp(-b * x) + c
        x = np.arange(n)
        y = history_array - best_value
        
        # Simple estimate: measure decay
        half_life_idx = np.argmin(np.abs(y - y[0] / 2))
        convergence_rate = np.log(2) / (half_life_idx + 1)
    else:
        convergence_rate = 0.0
    
    # Variance in final 10%
    final_portion = history_array[-max(1, n//10):]
    final_variance = np.var(final_portion)
    
    return {
        'final_value': float(final_value),
        'best_value': float(best_value),
        'convergence_rate': float(convergence_rate),
        'final_variance': float(final_variance),
        'n_iterations': n,
    }


def compute_contact_map(
    conformation: np.ndarray,
    cutoff: float = 1.5
) -> np.ndarray:
    """Compute contact map from conformation.
    
    Args:
        conformation: Residue positions (N, d)
        cutoff: Contact distance cutoff
        
    Returns:
        (N, N) binary contact matrix
    """
    n = len(conformation)
    contact_map = np.zeros((n, n), dtype=int)
    
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(conformation[i] - conformation[j])
            if dist <= cutoff:
                contact_map[i, j] = 1
                contact_map[j, i] = 1
    
    return contact_map


def compare_contact_maps(
    map1: np.ndarray,
    map2: np.ndarray
) -> float:
    """Compute contact map similarity (F1 score).
    
    Args:
        map1: First contact map
        map2: Second contact map
        
    Returns:
        F1 score (0-1)
    """
    # True positives, false positives, false negatives
    tp = np.sum((map1 == 1) & (map2 == 1))
    fp = np.sum((map1 == 1) & (map2 == 0))
    fn = np.sum((map1 == 0) & (map2 == 1))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return float(f1)
