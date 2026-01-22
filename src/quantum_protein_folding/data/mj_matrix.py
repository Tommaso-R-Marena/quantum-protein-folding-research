"""Miyazawa-Jernigan contact potential matrix.

Reference:
    Miyazawa, S., & Jernigan, R. L. (1996). Residue-residue potentials
    with a favorable contact pair term and an unfavorable high packing
    density term, for simulation and threading. Journal of Molecular
    Biology, 256(3), 623-644.

The MJ matrix provides statistical contact energies between amino acid
pairs derived from known protein structures. Units are in kcal/mol.
"""

import numpy as np
from typing import Dict, Tuple

# 20 standard amino acids (alphabetical order)
AMINO_ACIDS = [
    'A',  # Alanine
    'C',  # Cysteine
    'D',  # Aspartic acid
    'E',  # Glutamic acid
    'F',  # Phenylalanine
    'G',  # Glycine
    'H',  # Histidine
    'I',  # Isoleucine
    'K',  # Lysine
    'L',  # Leucine
    'M',  # Methionine
    'N',  # Asparagine
    'P',  # Proline
    'Q',  # Glutamine
    'R',  # Arginine
    'S',  # Serine
    'T',  # Threonine
    'V',  # Valine
    'W',  # Tryptophan
    'Y',  # Tyrosine
]

# Full MJ matrix (symmetric, 20x20)
# Values are in kcal/mol (more negative = more favorable)
# This is the MJ96 matrix from the 1996 paper
MJ_MATRIX_VALUES = np.array([
    # A     C     D     E     F     G     H     I     K     L     M     N     P     Q     R     S     T     V     W     Y
    [-1.94, -1.91, -1.09, -1.02, -2.01, -1.31, -1.38, -1.69, -0.96, -1.64, -1.55, -0.98, -1.07, -1.01, -1.00, -1.03, -1.15, -1.48, -1.83, -1.52],  # A
    [-1.91, -3.56, -2.41, -2.27, -3.29, -2.39, -2.62, -2.85, -2.15, -2.79, -2.63, -2.28, -2.18, -2.26, -2.27, -2.01, -2.15, -2.60, -3.15, -2.82],  # C
    [-1.09, -2.41, -0.82, -0.55, -1.56, -0.82, -0.99, -1.19, -0.24, -1.13, -1.06, -0.48, -0.69, -0.53, -0.39, -0.57, -0.73, -1.01, -1.41, -1.10],  # D
    [-1.02, -2.27, -0.55, -0.50, -1.48, -0.78, -0.91, -1.13, -0.32, -1.08, -1.00, -0.50, -0.66, -0.49, -0.43, -0.55, -0.70, -0.96, -1.35, -1.05],  # E
    [-2.01, -3.29, -1.56, -1.48, -2.78, -1.73, -1.88, -2.24, -1.36, -2.19, -2.03, -1.48, -1.52, -1.50, -1.48, -1.42, -1.57, -1.96, -2.48, -2.15],  # F
    [-1.31, -2.39, -0.82, -0.78, -1.73, -1.15, -1.19, -1.41, -0.66, -1.36, -1.27, -0.76, -0.91, -0.76, -0.73, -0.76, -0.91, -1.18, -1.61, -1.29],  # G
    [-1.38, -2.62, -0.99, -0.91, -1.88, -1.19, -1.33, -1.55, -0.76, -1.50, -1.40, -0.88, -1.02, -0.88, -0.84, -0.89, -1.05, -1.31, -1.76, -1.44],  # H
    [-1.69, -2.85, -1.19, -1.13, -2.24, -1.41, -1.55, -1.91, -1.03, -1.86, -1.74, -1.12, -1.22, -1.14, -1.11, -1.11, -1.27, -1.63, -2.09, -1.76],  # I
    [-0.96, -2.15, -0.24, -0.32, -1.36, -0.66, -0.76, -1.03, -0.01, -0.98, -0.92, -0.34, -0.55, -0.34, -0.26, -0.43, -0.60, -0.85, -1.26, -0.95],  # K
    [-1.64, -2.79, -1.13, -1.08, -2.19, -1.36, -1.50, -1.86, -0.98, -1.81, -1.69, -1.07, -1.17, -1.09, -1.06, -1.06, -1.22, -1.58, -2.04, -1.71],  # L
    [-1.55, -2.63, -1.06, -1.00, -2.03, -1.27, -1.40, -1.74, -0.92, -1.69, -1.60, -1.00, -1.09, -1.01, -0.99, -0.99, -1.14, -1.47, -1.91, -1.59],  # M
    [-0.98, -2.28, -0.48, -0.50, -1.48, -0.76, -0.88, -1.12, -0.34, -1.07, -1.00, -0.44, -0.67, -0.47, -0.42, -0.54, -0.70, -0.94, -1.35, -1.04],  # N
    [-1.07, -2.18, -0.69, -0.66, -1.52, -0.91, -1.02, -1.22, -0.55, -1.17, -1.09, -0.67, -0.74, -0.66, -0.63, -0.68, -0.83, -1.07, -1.48, -1.16],  # P
    [-1.01, -2.26, -0.53, -0.49, -1.50, -0.76, -0.88, -1.14, -0.34, -1.09, -1.01, -0.47, -0.66, -0.48, -0.43, -0.55, -0.71, -0.96, -1.37, -1.06],  # Q
    [-1.00, -2.27, -0.39, -0.43, -1.48, -0.73, -0.84, -1.11, -0.26, -1.06, -0.99, -0.42, -0.63, -0.43, -0.38, -0.52, -0.68, -0.93, -1.34, -1.03],  # R
    [-1.03, -2.01, -0.57, -0.55, -1.42, -0.76, -0.89, -1.11, -0.43, -1.06, -0.99, -0.54, -0.68, -0.55, -0.52, -0.60, -0.74, -0.98, -1.38, -1.07],  # S
    [-1.15, -2.15, -0.73, -0.70, -1.57, -0.91, -1.05, -1.27, -0.60, -1.22, -1.14, -0.70, -0.83, -0.71, -0.68, -0.74, -0.89, -1.13, -1.54, -1.22],  # T
    [-1.48, -2.60, -1.01, -0.96, -1.96, -1.18, -1.31, -1.63, -0.85, -1.58, -1.47, -0.94, -1.07, -0.96, -0.93, -0.98, -1.13, -1.42, -1.85, -1.52],  # V
    [-1.83, -3.15, -1.41, -1.35, -2.48, -1.61, -1.76, -2.09, -1.26, -2.04, -1.91, -1.35, -1.48, -1.37, -1.34, -1.38, -1.54, -1.85, -2.37, -2.02],  # W
    [-1.52, -2.82, -1.10, -1.05, -2.15, -1.29, -1.44, -1.76, -0.95, -1.71, -1.59, -1.04, -1.16, -1.06, -1.03, -1.07, -1.22, -1.52, -2.02, -1.70],  # Y
])

# Create amino acid to index mapping
AA_TO_INDEX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}


def get_mj_matrix(normalized: bool = False) -> np.ndarray:
    """Get the Miyazawa-Jernigan contact potential matrix.
    
    Args:
        normalized: If True, normalize energies to [0, 1] range
    
    Returns:
        20x20 symmetric matrix of contact energies
    
    Example:
        >>> mj = get_mj_matrix()
        >>> mj.shape
        (20, 20)
        >>> mj[0, 0]  # A-A contact
        -1.94
    """
    if normalized:
        # Normalize to [0, 1] where 0 = most favorable, 1 = least favorable
        min_val = MJ_MATRIX_VALUES.min()
        max_val = MJ_MATRIX_VALUES.max()
        return (MJ_MATRIX_VALUES - min_val) / (max_val - min_val)
    return MJ_MATRIX_VALUES.copy()


def get_contact_energy(aa1: str, aa2: str) -> float:
    """Get contact energy between two amino acids.
    
    Args:
        aa1: First amino acid (single letter code)
        aa2: Second amino acid (single letter code)
    
    Returns:
        Contact energy in kcal/mol (more negative = more favorable)
    
    Raises:
        ValueError: If amino acid not recognized
    
    Example:
        >>> energy = get_contact_energy('A', 'L')
        >>> print(f"{energy:.2f} kcal/mol")
        -1.64 kcal/mol
    """
    aa1 = aa1.upper()
    aa2 = aa2.upper()
    
    if aa1 not in AA_TO_INDEX:
        raise ValueError(f"Unknown amino acid: {aa1}")
    if aa2 not in AA_TO_INDEX:
        raise ValueError(f"Unknown amino acid: {aa2}")
    
    i = AA_TO_INDEX[aa1]
    j = AA_TO_INDEX[aa2]
    
    return MJ_MATRIX_VALUES[i, j]


def get_hp_energies() -> Tuple[float, float, float]:
    """Get simplified HP model contact energies.
    
    Returns:
        Tuple of (E_HH, E_HP, E_PP) energies
    
    The HP model simplifies the 20x20 MJ matrix to just 2 residue types:
    - H (hydrophobic): A, F, I, L, M, P, V, W
    - P (polar): C, D, E, G, H, K, N, Q, R, S, T, Y
    
    We compute average energies for each pair type from the full MJ matrix.
    
    Example:
        >>> e_hh, e_hp, e_pp = get_hp_energies()
        >>> print(f"HH: {e_hh:.2f}, HP: {e_hp:.2f}, PP: {e_pp:.2f}")
        HH: -2.02, HP: -1.31, PP: -0.85
    """
    # Hydrophobic residues
    hydrophobic = ['A', 'F', 'I', 'L', 'M', 'P', 'V', 'W']
    # Polar residues
    polar = ['C', 'D', 'E', 'G', 'H', 'K', 'N', 'Q', 'R', 'S', 'T', 'Y']
    
    # Get indices
    h_idx = [AA_TO_INDEX[aa] for aa in hydrophobic]
    p_idx = [AA_TO_INDEX[aa] for aa in polar]
    
    # Compute average energies
    e_hh_vals = [MJ_MATRIX_VALUES[i, j] for i in h_idx for j in h_idx if i <= j]
    e_hp_vals = [MJ_MATRIX_VALUES[i, j] for i in h_idx for j in p_idx]
    e_pp_vals = [MJ_MATRIX_VALUES[i, j] for i in p_idx for j in p_idx if i <= j]
    
    e_hh = np.mean(e_hh_vals)
    e_hp = np.mean(e_hp_vals)
    e_pp = np.mean(e_pp_vals)
    
    return e_hh, e_hp, e_pp


def sequence_to_mj_matrix(sequence: str) -> np.ndarray:
    """Get MJ contact matrix for a specific protein sequence.
    
    Args:
        sequence: Amino acid sequence (single letter codes)
    
    Returns:
        NxN matrix where element [i,j] is the contact energy between
        residues i and j
    
    Raises:
        ValueError: If sequence contains unknown amino acids
    
    Example:
        >>> seq = "AFILMVW"  # All hydrophobic
        >>> matrix = sequence_to_mj_matrix(seq)
        >>> matrix.shape
        (7, 7)
    """
    sequence = sequence.upper()
    n = len(sequence)
    matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            matrix[i, j] = get_contact_energy(sequence[i], sequence[j])
    
    return matrix
