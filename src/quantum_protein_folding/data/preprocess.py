"""Lattice encoding and preprocessing for quantum algorithms.

This module implements:
1. Binary position encoding: Map residues to lattice sites with binary qubits
2. Turn-based encoding: Encode conformations as sequence of turns
3. Lattice constraint enforcement: Self-avoidance and connectivity
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import itertools

from quantum_protein_folding.data.loaders import ProteinSequence


@dataclass
class LatticeEncoding:
    """Encoded protein conformation on lattice.
    
    Attributes:
        n_qubits: Total number of qubits required
        encoding_type: 'binary_position' or 'turn_direction'
        lattice_dim: Lattice dimension (2 or 3)
        lattice_size: Size of lattice in each dimension
        qubit_map: Mapping from qubit indices to physical meaning
        hamiltonian: Qiskit SparsePauliOp representing the energy
    """
    n_qubits: int
    encoding_type: str
    lattice_dim: int
    lattice_size: int
    qubit_map: Dict[int, str]
    hamiltonian: 'SparsePauliOp'  # type: ignore
    sequence: ProteinSequence
    

def encode_binary_positions(
    sequence: ProteinSequence,
    lattice_dim: int = 2,
    lattice_size: int = None
) -> Tuple[int, Dict[int, Tuple[int, int]]]:
    """Encode residue positions with binary qubits.
    
    Each residue position (x, y) or (x, y, z) is encoded in binary.
    For dimension d and lattice size L:
        - Each coordinate requires ceil(log2(L)) qubits
        - Total qubits: N * d * ceil(log2(L))
    
    Args:
        sequence: Protein sequence
        lattice_dim: Lattice dimension (2 or 3)
        lattice_size: Lattice size (default: sequence length + 2)
        
    Returns:
        n_qubits: Total qubits needed
        qubit_map: Maps qubit index to (residue_idx, coordinate, bit_position)
    """
    if lattice_size is None:
        lattice_size = sequence.length + 2
    
    n_bits_per_coord = int(np.ceil(np.log2(lattice_size)))
    n_qubits = sequence.length * lattice_dim * n_bits_per_coord
    
    qubit_map = {}
    qubit_idx = 0
    
    for residue in range(sequence.length):
        for coord in range(lattice_dim):
            for bit in range(n_bits_per_coord):
                qubit_map[qubit_idx] = (residue, coord, bit)
                qubit_idx += 1
    
    return n_qubits, qubit_map


def encode_turn_directions(
    sequence: ProteinSequence,
    lattice_dim: int = 2
) -> Tuple[int, Dict[int, Tuple[int, str]]]:
    """Encode conformation as sequence of turn directions.
    
    For 2D lattice: 4 directions (forward, left, right, back)
    For 3D lattice: 6 directions
    
    Each turn requires ceil(log2(n_directions)) qubits.
    Total qubits: (N-1) * ceil(log2(n_directions))
    
    Args:
        sequence: Protein sequence
        lattice_dim: Lattice dimension
        
    Returns:
        n_qubits: Total qubits needed
        qubit_map: Maps qubit index to (bond_idx, direction_bit)
    """
    n_directions = 2 * lattice_dim
    n_bits_per_turn = int(np.ceil(np.log2(n_directions)))
    n_bonds = sequence.length - 1
    n_qubits = n_bonds * n_bits_per_turn
    
    qubit_map = {}
    qubit_idx = 0
    
    for bond in range(n_bonds):
        for bit in range(n_bits_per_turn):
            qubit_map[qubit_idx] = (bond, f"direction_bit_{bit}")
            qubit_idx += 1
    
    return n_qubits, qubit_map


def map_to_lattice(
    sequence: ProteinSequence,
    lattice_dim: int = 2,
    lattice_size: Optional[int] = None,
    encoding_type: str = 'turn_direction',
    constraint_weight: float = 10.0,
    bias_weight: float = 0.1
) -> LatticeEncoding:
    """Map protein sequence to lattice with quantum Hamiltonian.
    
    This is the main preprocessing function that:
    1. Chooses encoding scheme
    2. Builds quantum Hamiltonian with contact, backbone, and bias terms
    3. Returns complete lattice encoding ready for VQE/QAOA
    
    Args:
        sequence: Protein sequence with contact matrix
        lattice_dim: Lattice dimension (2 or 3)
        lattice_size: Lattice size (auto-determined if None)
        encoding_type: 'binary_position' or 'turn_direction'
        constraint_weight: Weight for backbone constraints (lambda)
        bias_weight: Weight for compactness bias (mu)
        
    Returns:
        LatticeEncoding with Hamiltonian
    """
    from quantum_protein_folding.quantum.hamiltonian import build_hamiltonian
    
    if lattice_size is None:
        lattice_size = sequence.length + 2
    
    # Choose encoding
    if encoding_type == 'binary_position':
        n_qubits, qubit_map = encode_binary_positions(
            sequence, lattice_dim, lattice_size
        )
    elif encoding_type == 'turn_direction':
        n_qubits, qubit_map = encode_turn_directions(sequence, lattice_dim)
    else:
        raise ValueError(f"Unknown encoding type: {encoding_type}")
    
    # Build Hamiltonian
    hamiltonian = build_hamiltonian(
        sequence=sequence,
        n_qubits=n_qubits,
        lattice_dim=lattice_dim,
        lattice_size=lattice_size,
        encoding_type=encoding_type,
        constraint_weight=constraint_weight,
        bias_weight=bias_weight
    )
    
    return LatticeEncoding(
        n_qubits=n_qubits,
        encoding_type=encoding_type,
        lattice_dim=lattice_dim,
        lattice_size=lattice_size,
        qubit_map=qubit_map,
        hamiltonian=hamiltonian,
        sequence=sequence
    )


def decode_conformation(
    bitstring: str,
    encoding: LatticeEncoding
) -> np.ndarray:
    """Decode bitstring to lattice conformation.
    
    Args:
        bitstring: Measurement outcome (binary string)
        encoding: Lattice encoding specification
        
    Returns:
        coordinates: (N, d) array of residue positions
    """
    n = encoding.sequence.length
    d = encoding.lattice_dim
    
    if encoding.encoding_type == 'binary_position':
        return _decode_binary_position(bitstring, n, d, encoding.lattice_size)
    elif encoding.encoding_type == 'turn_direction':
        return _decode_turn_direction(bitstring, n, d)
    else:
        raise ValueError(f"Unknown encoding: {encoding.encoding_type}")


def _decode_binary_position(
    bitstring: str,
    n_residues: int,
    lattice_dim: int,
    lattice_size: int
) -> np.ndarray:
    """Decode binary position encoding to coordinates."""
    n_bits_per_coord = int(np.ceil(np.log2(lattice_size)))
    coords = np.zeros((n_residues, lattice_dim), dtype=int)
    
    bit_idx = 0
    for i in range(n_residues):
        for d in range(lattice_dim):
            # Extract bits for this coordinate
            coord_bits = bitstring[bit_idx:bit_idx + n_bits_per_coord]
            coords[i, d] = int(coord_bits, 2) % lattice_size
            bit_idx += n_bits_per_coord
    
    return coords


def _decode_turn_direction(
    bitstring: str,
    n_residues: int,
    lattice_dim: int
) -> np.ndarray:
    """Decode turn direction encoding to coordinates."""
    n_directions = 2 * lattice_dim
    n_bits_per_turn = int(np.ceil(np.log2(n_directions)))
    
    # Direction vectors for 2D
    if lattice_dim == 2:
        direction_map = {
            0: np.array([1, 0]),   # East
            1: np.array([0, 1]),   # North
            2: np.array([-1, 0]),  # West
            3: np.array([0, -1]),  # South
        }
    else:  # 3D
        direction_map = {
            0: np.array([1, 0, 0]),
            1: np.array([-1, 0, 0]),
            2: np.array([0, 1, 0]),
            3: np.array([0, -1, 0]),
            4: np.array([0, 0, 1]),
            5: np.array([0, 0, -1]),
        }
    
    # Start at origin
    coords = np.zeros((n_residues, lattice_dim), dtype=int)
    current_pos = np.zeros(lattice_dim, dtype=int)
    coords[0] = current_pos
    
    # Decode turns
    bit_idx = 0
    for bond in range(n_residues - 1):
        turn_bits = bitstring[bit_idx:bit_idx + n_bits_per_turn]
        direction_idx = int(turn_bits, 2) % n_directions
        
        current_pos = current_pos + direction_map[direction_idx]
        coords[bond + 1] = current_pos
        bit_idx += n_bits_per_turn
    
    return coords


def check_valid_conformation(
    coords: np.ndarray,
    sequence: ProteinSequence
) -> Tuple[bool, str]:
    """Check if conformation satisfies lattice constraints.
    
    Args:
        coords: (N, d) array of residue positions
        sequence: Protein sequence
        
    Returns:
        is_valid: True if conformation is valid
        message: Description of violation (if any)
    """
    n = sequence.length
    
    # Check connectivity: adjacent residues must be neighbors
    for i in range(n - 1):
        dist = np.linalg.norm(coords[i+1] - coords[i])
        if abs(dist - 1.0) > 1e-6:
            return False, f"Bond {i}-{i+1} has length {dist:.3f} (should be 1)"
    
    # Check self-avoidance: no two residues at same position
    for i in range(n):
        for j in range(i + 1, n):
            if np.allclose(coords[i], coords[j]):
                return False, f"Residues {i} and {j} overlap at {coords[i]}"
    
    return True, "Valid conformation"
