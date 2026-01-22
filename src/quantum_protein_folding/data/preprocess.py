"""Lattice mapping and preprocessing for protein sequences.

Converts protein sequences to lattice representations suitable for
quantum algorithm encoding.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Union
from enum import Enum

from quantum_protein_folding.data.loaders import ProteinSequence
from quantum_protein_folding.data.mj_matrix import (
    sequence_to_mj_matrix,
    get_hp_energies,
)


class LatticeType(Enum):
    """Supported lattice types for protein folding."""
    CUBIC_2D = "cubic_2d"  # Square lattice
    CUBIC_3D = "cubic_3d"  # Cubic lattice
    TETRAHEDRAL = "tetrahedral"  # Tetrahedral lattice (3D)


class EncodingType(Enum):
    """Qubit encoding strategies."""
    POSITION = "position"  # Binary encoding of absolute positions
    DIRECTION = "direction"  # Binary encoding of bond directions
    TURN = "turn"  # Encoding of turns (left/right/straight)


@dataclass
class LatticeData:
    """Container for lattice-encoded protein data.
    
    Attributes:
        sequence: Original protein sequence
        lattice_type: Type of lattice
        lattice_dim: Lattice dimension (2 or 3)
        lattice_size: Maximum extent in each dimension
        contact_matrix: NxN contact energy matrix
        n_qubits: Total number of qubits required
        encoding_type: Qubit encoding strategy
        hamiltonian: Quantum Hamiltonian operator (if constructed)
        metadata: Additional information
    """
    sequence: str
    lattice_type: LatticeType
    lattice_dim: int
    lattice_size: int
    contact_matrix: np.ndarray
    n_qubits: int
    encoding_type: EncodingType
    hamiltonian: Optional[object] = None
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


def calculate_n_qubits(
    chain_length: int,
    lattice_dim: int,
    lattice_size: int,
    encoding_type: EncodingType
) -> int:
    """Calculate number of qubits needed for encoding.
    
    Args:
        chain_length: Number of residues
        lattice_dim: Dimension of lattice (2 or 3)
        lattice_size: Maximum extent in each dimension
        encoding_type: Encoding strategy
    
    Returns:
        Total number of qubits
    
    Mathematical Details:
        Position encoding:
            - Each coordinate needs log2(lattice_size) qubits
            - Total: chain_length * lattice_dim * ceil(log2(lattice_size))
        
        Direction encoding:
            - 2D: 4 directions (2 qubits per bond)
            - 3D: 6 directions (3 qubits per bond)
            - Total: (chain_length - 1) * qubits_per_direction
        
        Turn encoding:
            - 3 turn types: straight, left, right
            - 2 qubits per turn
            - Total: (chain_length - 2) * 2
    
    Example:
        >>> n_q = calculate_n_qubits(10, 2, 8, EncodingType.POSITION)
        >>> print(n_q)
        60  # 10 * 2 * 3 (log2(8) = 3)
    """
    if encoding_type == EncodingType.POSITION:
        bits_per_coord = int(np.ceil(np.log2(lattice_size)))
        return chain_length * lattice_dim * bits_per_coord
    
    elif encoding_type == EncodingType.DIRECTION:
        if lattice_dim == 2:
            bits_per_direction = 2  # 4 directions: up, down, left, right
        else:  # 3D
            bits_per_direction = 3  # 6 directions: ±x, ±y, ±z
        return (chain_length - 1) * bits_per_direction
    
    elif encoding_type == EncodingType.TURN:
        # Encode relative turns between consecutive bonds
        return (chain_length - 2) * 2  # 2 qubits encode 3 turn types
    
    else:
        raise ValueError(f"Unknown encoding type: {encoding_type}")


def encode_sequence(
    sequence: Union[str, ProteinSequence],
    lattice_dim: int = 2,
    lattice_size: int = None,
    encoding_type: EncodingType = EncodingType.DIRECTION,
) -> np.ndarray:
    """Encode protein sequence for lattice representation.
    
    Args:
        sequence: Protein sequence or ProteinSequence object
        lattice_dim: Lattice dimension (2 or 3)
        lattice_size: Maximum lattice extent (default: 2 * len(sequence))
        encoding_type: Encoding strategy
    
    Returns:
        Integer array encoding residue properties
    
    Example:
        >>> encoded = encode_sequence("HPHPH", lattice_dim=2)
        >>> encoded.shape
        (5,)
    """
    if isinstance(sequence, ProteinSequence):
        seq_str = str(sequence)
    else:
        seq_str = sequence
    
    if lattice_size is None:
        lattice_size = 2 * len(seq_str)
    
    # For now, return simple integer encoding
    # H -> 0, P -> 1 for HP model
    # Or map amino acids to indices
    if all(c in 'HP' for c in seq_str.upper()):
        return np.array([0 if c == 'H' else 1 for c in seq_str.upper()])
    else:
        # Use ASCII values as simple encoding
        return np.array([ord(c) for c in seq_str.upper()])


def map_to_lattice(
    sequence: Union[str, ProteinSequence],
    lattice_dim: int = 2,
    lattice_size: Optional[int] = None,
    lattice_type: LatticeType = LatticeType.CUBIC_2D,
    encoding_type: EncodingType = EncodingType.DIRECTION,
    potential_type: str = "hp",
    **kwargs
) -> LatticeData:
    """Map protein sequence to lattice representation.
    
    Args:
        sequence: Protein sequence string or ProteinSequence object
        lattice_dim: Dimension of lattice (2 or 3)
        lattice_size: Maximum extent in each dimension (auto if None)
        lattice_type: Type of lattice geometry
        encoding_type: Qubit encoding strategy
        potential_type: Contact potential ('hp' or 'miyazawa_jernigan')
        **kwargs: Additional parameters
    
    Returns:
        LatticeData object with all preprocessing completed
    
    Example:
        >>> from quantum_protein_folding.data.loaders import load_hp_sequence
        >>> seq = load_hp_sequence("HPHPPHHPHH")
        >>> lattice_data = map_to_lattice(
        ...     seq,
        ...     lattice_dim=2,
        ...     lattice_type=LatticeType.CUBIC_2D,
        ...     potential_type="hp"
        ... )
        >>> print(lattice_data.n_qubits)
        18  # (10-1) * 2 for direction encoding in 2D
    """
    # Extract sequence string
    if isinstance(sequence, ProteinSequence):
        seq_str = str(sequence)
        seq_name = sequence.name
    else:
        seq_str = str(sequence)
        seq_name = "unknown"
    
    chain_length = len(seq_str)
    
    # Auto-determine lattice size if not provided
    if lattice_size is None:
        # Rule of thumb: lattice should be large enough to accommodate
        # fully extended chain with some buffer
        lattice_size = max(8, chain_length + 2)
    
    # Validate lattice type and dimension consistency
    if lattice_type == LatticeType.CUBIC_2D and lattice_dim != 2:
        raise ValueError("CUBIC_2D requires lattice_dim=2")
    if lattice_type in [LatticeType.CUBIC_3D, LatticeType.TETRAHEDRAL] and lattice_dim != 3:
        raise ValueError(f"{lattice_type.value} requires lattice_dim=3")
    
    # Calculate number of qubits
    n_qubits = calculate_n_qubits(
        chain_length, lattice_dim, lattice_size, encoding_type
    )
    
    # Build contact energy matrix
    if potential_type == "hp" or all(c in 'HP' for c in seq_str.upper()):
        # HP model: use simplified energies
        e_hh, e_hp, e_pp = get_hp_energies()
        contact_matrix = np.zeros((chain_length, chain_length))
        
        for i in range(chain_length):
            for j in range(chain_length):
                if i != j:
                    aa_i = seq_str[i].upper()
                    aa_j = seq_str[j].upper()
                    
                    if aa_i == 'H' and aa_j == 'H':
                        contact_matrix[i, j] = e_hh
                    elif aa_i == 'P' and aa_j == 'P':
                        contact_matrix[i, j] = e_pp
                    else:
                        contact_matrix[i, j] = e_hp
    
    elif potential_type == "miyazawa_jernigan" or potential_type == "mj":
        # Full MJ matrix
        contact_matrix = sequence_to_mj_matrix(seq_str)
    
    else:
        raise ValueError(
            f"Unknown potential type: {potential_type}. "
            f"Use 'hp' or 'miyazawa_jernigan'"
        )
    
    # Create LatticeData object
    lattice_data = LatticeData(
        sequence=seq_str,
        lattice_type=lattice_type,
        lattice_dim=lattice_dim,
        lattice_size=lattice_size,
        contact_matrix=contact_matrix,
        n_qubits=n_qubits,
        encoding_type=encoding_type,
        metadata={
            'sequence_name': seq_name,
            'chain_length': chain_length,
            'potential_type': potential_type,
            **kwargs
        }
    )
    
    return lattice_data


def get_neighbor_directions(lattice_dim: int) -> List[Tuple[int, ...]]:
    """Get possible neighbor directions for a lattice.
    
    Args:
        lattice_dim: Dimension of lattice (2 or 3)
    
    Returns:
        List of direction tuples
    
    Example:
        >>> dirs_2d = get_neighbor_directions(2)
        >>> len(dirs_2d)
        4  # up, down, left, right
        >>> dirs_3d = get_neighbor_directions(3)
        >>> len(dirs_3d)
        6  # ±x, ±y, ±z
    """
    if lattice_dim == 2:
        return [(0, 1), (0, -1), (1, 0), (-1, 0)]  # Up, down, right, left
    elif lattice_dim == 3:
        return [
            (1, 0, 0), (-1, 0, 0),  # ±x
            (0, 1, 0), (0, -1, 0),  # ±y
            (0, 0, 1), (0, 0, -1),  # ±z
        ]
    else:
        raise ValueError(f"Unsupported dimension: {lattice_dim}")


def bitstring_to_conformation(
    bitstring: str,
    lattice_data: LatticeData,
    start_position: Optional[Tuple[int, ...]] = None
) -> np.ndarray:
    """Decode bitstring to lattice conformation.
    
    Args:
        bitstring: Binary string from quantum measurement
        lattice_data: LatticeData object with encoding info
        start_position: Starting position (default: lattice center)
    
    Returns:
        (chain_length, lattice_dim) array of positions
    
    Example:
        >>> lattice_data = map_to_lattice("HPHPH", lattice_dim=2)
        >>> bitstring = "01001101"  # Example measurement
        >>> positions = bitstring_to_conformation(bitstring, lattice_data)
        >>> positions.shape
        (5, 2)
    """
    chain_length = len(lattice_data.sequence)
    lattice_dim = lattice_data.lattice_dim
    encoding_type = lattice_data.encoding_type
    
    if start_position is None:
        # Start at center of lattice
        start_position = tuple([lattice_data.lattice_size // 2] * lattice_dim)
    
    if encoding_type == EncodingType.DIRECTION:
        # Decode direction encoding
        directions = get_neighbor_directions(lattice_dim)
        n_dirs = len(directions)
        bits_per_direction = int(np.ceil(np.log2(n_dirs)))
        
        # Build conformation by following encoded directions
        positions = np.zeros((chain_length, lattice_dim), dtype=int)
        positions[0] = start_position
        
        for i in range(chain_length - 1):
            # Extract bits for this bond
            start_bit = i * bits_per_direction
            end_bit = start_bit + bits_per_direction
            direction_bits = bitstring[start_bit:end_bit]
            
            # Convert to direction index
            dir_idx = int(direction_bits, 2) % n_dirs
            direction = directions[dir_idx]
            
            # Move to next position
            positions[i + 1] = positions[i] + np.array(direction)
    
    elif encoding_type == EncodingType.POSITION:
        # Decode absolute position encoding
        bits_per_coord = int(np.ceil(np.log2(lattice_data.lattice_size)))
        bits_per_residue = lattice_dim * bits_per_coord
        
        positions = np.zeros((chain_length, lattice_dim), dtype=int)
        
        for i in range(chain_length):
            for d in range(lattice_dim):
                start_bit = i * bits_per_residue + d * bits_per_coord
                end_bit = start_bit + bits_per_coord
                coord_bits = bitstring[start_bit:end_bit]
                positions[i, d] = int(coord_bits, 2) % lattice_data.lattice_size
    
    else:
        raise NotImplementedError(f"Decoding for {encoding_type} not implemented")
    
    return positions


def check_self_avoidance(positions: np.ndarray) -> bool:
    """Check if conformation satisfies self-avoidance constraint.
    
    Args:
        positions: (chain_length, lattice_dim) array of positions
    
    Returns:
        True if no self-overlaps, False otherwise
    
    Example:
        >>> positions = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])
        >>> check_self_avoidance(positions)
        True
        >>> positions_overlap = np.array([[0, 0], [0, 1], [0, 0]])
        >>> check_self_avoidance(positions_overlap)
        False
    """
    # Convert positions to tuples for set comparison
    position_tuples = [tuple(pos) for pos in positions]
    return len(position_tuples) == len(set(position_tuples))


def check_connectivity(positions: np.ndarray) -> bool:
    """Check if conformation satisfies chain connectivity.
    
    Args:
        positions: (chain_length, lattice_dim) array of positions
    
    Returns:
        True if all consecutive residues are neighbors, False otherwise
    
    Example:
        >>> positions = np.array([[0, 0], [0, 1], [1, 1]])
        >>> check_connectivity(positions)
        True
        >>> positions_broken = np.array([[0, 0], [0, 2], [1, 1]])
        >>> check_connectivity(positions_broken)
        False
    """
    for i in range(len(positions) - 1):
        distance = np.linalg.norm(positions[i + 1] - positions[i])
        if not np.isclose(distance, 1.0):
            return False
    return True


def is_valid_conformation(positions: np.ndarray) -> bool:
    """Check if conformation is physically valid.
    
    Args:
        positions: (chain_length, lattice_dim) array of positions
    
    Returns:
        True if conformation satisfies all constraints
    
    Example:
        >>> positions = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])
        >>> is_valid_conformation(positions)
        True
    """
    return check_self_avoidance(positions) and check_connectivity(positions)
