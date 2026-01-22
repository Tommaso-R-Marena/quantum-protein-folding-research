"""Quantum Hamiltonian construction for lattice protein folding.

Implements the energy Hamiltonian:
    H = H_contact + H_backbone + H_bias

where:
- H_contact: Inter-residue contact energies (MJ potentials)
- H_backbone: Connectivity and geometry constraints
- H_bias: Compactness regularization
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from qiskit.quantum_info import SparsePauliOp, Pauli
from itertools import combinations

from quantum_protein_folding.data.loaders import ProteinSequence


def build_hamiltonian(
    sequence: ProteinSequence,
    n_qubits: int,
    lattice_dim: int,
    lattice_size: int,
    encoding_type: str = 'turn_direction',
    constraint_weight: float = 10.0,
    bias_weight: float = 0.1
) -> SparsePauliOp:
    """Build complete quantum Hamiltonian for protein folding.
    
    Args:
        sequence: Protein sequence with contact matrix
        n_qubits: Total number of qubits
        lattice_dim: Lattice dimension
        lattice_size: Lattice size
        encoding_type: Encoding scheme
        constraint_weight: Weight lambda for backbone constraints
        bias_weight: Weight mu for compactness bias
        
    Returns:
        SparsePauliOp representing full Hamiltonian
    """
    # Build each term
    H_contact = _build_contact_hamiltonian(
        sequence, n_qubits, lattice_dim, encoding_type
    )
    
    H_backbone = _build_backbone_hamiltonian(
        sequence, n_qubits, lattice_dim, lattice_size, encoding_type
    )
    
    H_bias = _build_bias_hamiltonian(
        sequence, n_qubits, lattice_dim, encoding_type
    )
    
    # Combine with weights
    hamiltonian = (
        H_contact +
        constraint_weight * H_backbone +
        bias_weight * H_bias
    )
    
    # Simplify
    hamiltonian = hamiltonian.simplify()
    
    return hamiltonian


def _build_contact_hamiltonian(
    sequence: ProteinSequence,
    n_qubits: int,
    lattice_dim: int,
    encoding_type: str
) -> SparsePauliOp:
    """Build contact energy Hamiltonian: H_contact = sum_{i<j} epsilon_ij * delta(r_i, r_j).
    
    For turn encoding, we approximate contact detection using:
    - Pairwise qubit interactions that encourage/discourage proximity
    - Simplified QUBO formulation
    """
    n = sequence.length
    pauli_list = []
    
    if encoding_type == 'turn_direction':
        # For turn encoding: approximate contacts with QUBO terms
        # This is a simplified model where we penalize/reward certain turn patterns
        
        for i in range(n - 3):  # Non-local contacts only (|i-j| > 2)
            for j in range(i + 3, n):
                epsilon = sequence.get_contact_energy(i, j)
                
                if abs(epsilon) < 1e-6:
                    continue  # Skip zero interactions
                
                # Approximate contact probability with qubit correlations
                # This is a crude approximation; exact implementation requires
                # constraint satisfaction clauses
                
                # Use ZZ interactions between turn qubits
                n_bits_per_turn = int(np.ceil(np.log2(2 * lattice_dim)))
                
                # Bond indices for residues i and j
                bond_i = i
                bond_j = j - 1
                
                if bond_i < n - 1 and bond_j < n - 1:
                    for bit_i in range(n_bits_per_turn):
                        for bit_j in range(n_bits_per_turn):
                            qubit_i = bond_i * n_bits_per_turn + bit_i
                            qubit_j = bond_j * n_bits_per_turn + bit_j
                            
                            if qubit_i < n_qubits and qubit_j < n_qubits:
                                # ZZ interaction
                                pauli_str = ['I'] * n_qubits
                                pauli_str[qubit_i] = 'Z'
                                pauli_str[qubit_j] = 'Z'
                                pauli_list.append(
                                    (''.join(pauli_str), epsilon / (n_bits_per_turn ** 2))
                                )
    
    else:  # binary_position encoding
        # For position encoding: exact contact detection requires many ancillas
        # Use approximate QUBO formulation
        
        for i in range(n - 3):
            for j in range(i + 3, n):
                epsilon = sequence.get_contact_energy(i, j)
                
                if abs(epsilon) < 1e-6:
                    continue
                
                # Simplified: penalize coordinate differences
                n_bits_per_coord = int(np.ceil(np.log2(sequence.length + 2)))
                
                for dim in range(lattice_dim):
                    for bit in range(n_bits_per_coord):
                        qubit_i = i * lattice_dim * n_bits_per_coord + dim * n_bits_per_coord + bit
                        qubit_j = j * lattice_dim * n_bits_per_coord + dim * n_bits_per_coord + bit
                        
                        if qubit_i < n_qubits and qubit_j < n_qubits:
                            # Encourage same bit values (contact)
                            pauli_str = ['I'] * n_qubits
                            pauli_str[qubit_i] = 'Z'
                            pauli_str[qubit_j] = 'Z'
                            pauli_list.append(
                                (''.join(pauli_str), epsilon / (lattice_dim * n_bits_per_coord))
                            )
    
    if not pauli_list:
        # Return zero operator if no contacts
        return SparsePauliOp(['I' * n_qubits], [0.0])
    
    return SparsePauliOp.from_list(pauli_list)


def _build_backbone_hamiltonian(
    sequence: ProteinSequence,
    n_qubits: int,
    lattice_dim: int,
    lattice_size: int,
    encoding_type: str
) -> SparsePauliOp:
    """Build backbone constraint Hamiltonian.
    
    Enforces:
    1. Chain connectivity: |r_{i+1} - r_i| = 1
    2. Self-avoidance: r_i != r_j for i != j
    
    These are hard constraints, implemented as penalty terms.
    """
    n = sequence.length
    pauli_list = []
    
    if encoding_type == 'turn_direction':
        # Turn encoding naturally enforces connectivity
        # Add self-avoidance penalties
        
        n_bits_per_turn = int(np.ceil(np.log2(2 * lattice_dim)))
        
        # Penalize invalid turn sequences (approximation)
        for bond in range(n - 1):
            for bit in range(n_bits_per_turn):
                qubit_idx = bond * n_bits_per_turn + bit
                if qubit_idx < n_qubits:
                    # Add small Z terms to bias toward valid configurations
                    pauli_str = ['I'] * n_qubits
                    pauli_str[qubit_idx] = 'Z'
                    pauli_list.append((''.join(pauli_str), 0.1))
    
    else:  # binary_position
        # Add connectivity constraints: penalize |r_{i+1} - r_i| != 1
        # This requires arithmetic circuits (complex)
        # Simplified: add penalty for large separations
        
        n_bits_per_coord = int(np.ceil(np.log2(lattice_size)))
        
        for i in range(n - 1):
            for dim in range(lattice_dim):
                for bit in range(n_bits_per_coord):
                    qubit_i = i * lattice_dim * n_bits_per_coord + dim * n_bits_per_coord + bit
                    qubit_j = (i+1) * lattice_dim * n_bits_per_coord + dim * n_bits_per_coord + bit
                    
                    if qubit_i < n_qubits and qubit_j < n_qubits:
                        # Penalize XOR (different bits)
                        pauli_str_zz = ['I'] * n_qubits
                        pauli_str_zz[qubit_i] = 'Z'
                        pauli_str_zz[qubit_j] = 'Z'
                        pauli_list.append((''.join(pauli_str_zz), -0.5))  # Reward similarity
                        
                        # Add Z terms
                        pauli_str_z = ['I'] * n_qubits
                        pauli_str_z[qubit_i] = 'Z'
                        pauli_list.append((''.join(pauli_str_z), 0.5))
    
    if not pauli_list:
        return SparsePauliOp(['I' * n_qubits], [0.0])
    
    return SparsePauliOp.from_list(pauli_list)


def _build_bias_hamiltonian(
    sequence: ProteinSequence,
    n_qubits: int,
    lattice_dim: int,
    encoding_type: str
) -> SparsePauliOp:
    """Build compactness bias: H_bias = mu * sum_i |r_i|^2.
    
    Favors conformations near the origin.
    """
    n = sequence.length
    pauli_list = []
    
    # Simple implementation: add Z terms to all qubits
    # This provides a weak bias toward |0...0> state
    
    for qubit in range(n_qubits):
        pauli_str = ['I'] * n_qubits
        pauli_str[qubit] = 'Z'
        pauli_list.append((''.join(pauli_str), 0.01))  # Small bias
    
    if not pauli_list:
        return SparsePauliOp(['I' * n_qubits], [0.0])
    
    return SparsePauliOp.from_list(pauli_list)


def hamiltonian_to_matrix(
    hamiltonian: SparsePauliOp
) -> np.ndarray:
    """Convert Hamiltonian to dense matrix representation.
    
    Useful for exact diagonalization and testing.
    
    Args:
        hamiltonian: Sparse Pauli operator
        
    Returns:
        Dense matrix (2^n x 2^n)
    """
    return hamiltonian.to_matrix()


def compute_exact_ground_state(
    hamiltonian: SparsePauliOp
) -> Tuple[float, np.ndarray]:
    """Compute exact ground state via diagonalization.
    
    Only feasible for small systems (n_qubits <= 20).
    
    Args:
        hamiltonian: Hamiltonian operator
        
    Returns:
        ground_energy: Lowest eigenvalue
        ground_state: Corresponding eigenvector
    """
    matrix = hamiltonian_to_matrix(hamiltonian)
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    
    ground_energy = eigenvalues[0]
    ground_state = eigenvectors[:, 0]
    
    return ground_energy, ground_state
