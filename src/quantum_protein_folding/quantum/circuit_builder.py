"""Quantum circuit construction for variational algorithms.

Implements:
1. Hardware-efficient ansatz
2. Problem-inspired ansatz (lattice topology)
3. QAOA mixer and cost operators
"""

import numpy as np
from typing import List, Optional, Tuple
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterVector
from qiskit.quantum_info import SparsePauliOp


def build_hardware_efficient_ansatz(
    n_qubits: int,
    depth: int = 3,
    entanglement: str = 'linear',
    parameter_prefix: str = 'θ'
) -> Tuple[QuantumCircuit, List[Parameter]]:
    """Build hardware-efficient variational ansatz.
    
    Circuit structure per layer:
    1. Single-qubit rotations: Ry(θ) ⊗ Rz(φ) on each qubit
    2. Entangling gates: CNOT according to topology
    
    Args:
        n_qubits: Number of qubits
        depth: Number of ansatz layers
        entanglement: 'linear', 'circular', or 'full'
        parameter_prefix: Prefix for parameter names
        
    Returns:
        circuit: Parameterized quantum circuit
        parameters: List of variational parameters
    """
    # Create parameter vectors
    params_ry = ParameterVector(f'{parameter_prefix}_ry', length=n_qubits * depth)
    params_rz = ParameterVector(f'{parameter_prefix}_rz', length=n_qubits * depth)
    
    circuit = QuantumCircuit(n_qubits)
    
    # Initial layer: Hadamard on all qubits
    for qubit in range(n_qubits):
        circuit.h(qubit)
    
    param_idx = 0
    
    for layer in range(depth):
        # Single-qubit rotations
        for qubit in range(n_qubits):
            circuit.ry(params_ry[param_idx], qubit)
            circuit.rz(params_rz[param_idx], qubit)
            param_idx += 1
        
        # Entangling layer
        if entanglement == 'linear':
            for qubit in range(n_qubits - 1):
                circuit.cx(qubit, qubit + 1)
        elif entanglement == 'circular':
            for qubit in range(n_qubits - 1):
                circuit.cx(qubit, qubit + 1)
            circuit.cx(n_qubits - 1, 0)  # Wrap around
        elif entanglement == 'full':
            for i in range(n_qubits):
                for j in range(i + 1, n_qubits):
                    circuit.cx(i, j)
        else:
            raise ValueError(f"Unknown entanglement: {entanglement}")
        
        # Barrier for visualization
        circuit.barrier()
    
    # Final rotation layer
    for qubit in range(n_qubits):
        circuit.ry(params_ry[param_idx], qubit)
        circuit.rz(params_rz[param_idx], qubit)
        param_idx += 1
    
    all_params = list(params_ry) + list(params_rz)
    
    return circuit, all_params


def build_problem_inspired_ansatz(
    n_qubits: int,
    lattice_dim: int,
    depth: int = 2,
    parameter_prefix: str = 'θ'
) -> Tuple[QuantumCircuit, List[Parameter]]:
    """Build problem-inspired ansatz based on lattice topology.
    
    Groups qubits according to lattice structure and applies
    rotations that respect geometric constraints.
    
    Args:
        n_qubits: Number of qubits
        lattice_dim: Lattice dimension (affects grouping)
        depth: Number of ansatz layers
        parameter_prefix: Parameter prefix
        
    Returns:
        circuit: Parameterized circuit
        parameters: Variational parameters
    """
    # For simplicity, use hardware-efficient with structured entanglement
    # In a full implementation, this would encode lattice geometry
    
    circuit, params = build_hardware_efficient_ansatz(
        n_qubits=n_qubits,
        depth=depth,
        entanglement='linear',
        parameter_prefix=parameter_prefix
    )
    
    return circuit, params


def build_qaoa_circuit(
    hamiltonian: SparsePauliOp,
    p_layers: int = 1,
    mixer_hamiltonian: Optional[SparsePauliOp] = None
) -> Tuple[QuantumCircuit, List[Parameter]]:
    """Build QAOA circuit with alternating cost and mixer layers.
    
    Circuit:
        |ψ(β, γ)⟩ = ∏_{p=1}^P e^{-iβ_p H_M} e^{-iγ_p H_C} |+⟩^⊗n
    
    Args:
        hamiltonian: Cost Hamiltonian H_C
        p_layers: Number of QAOA layers
        mixer_hamiltonian: Mixer H_M (default: X mixer)
        
    Returns:
        circuit: QAOA circuit
        parameters: [β_1, ..., β_P, γ_1, ..., γ_P]
    """
    n_qubits = hamiltonian.num_qubits
    
    # Default mixer: sum of X operators
    if mixer_hamiltonian is None:
        mixer_ops = [('I' * i + 'X' + 'I' * (n_qubits - i - 1), 1.0) 
                     for i in range(n_qubits)]
        mixer_hamiltonian = SparsePauliOp.from_list(mixer_ops)
    
    # Parameters
    betas = ParameterVector('β', length=p_layers)
    gammas = ParameterVector('γ', length=p_layers)
    
    circuit = QuantumCircuit(n_qubits)
    
    # Initialize in |+⟩^⊗n
    for qubit in range(n_qubits):
        circuit.h(qubit)
    
    # QAOA layers
    for p in range(p_layers):
        # Cost layer: e^{-iγ H_C}
        _apply_pauli_evolution(circuit, hamiltonian, gammas[p])
        
        # Mixer layer: e^{-iβ H_M}
        _apply_pauli_evolution(circuit, mixer_hamiltonian, betas[p])
        
        circuit.barrier()
    
    parameters = list(betas) + list(gammas)
    
    return circuit, parameters


def _apply_pauli_evolution(
    circuit: QuantumCircuit,
    hamiltonian: SparsePauliOp,
    parameter: Parameter
) -> None:
    """Apply e^{-i*parameter*hamiltonian} to circuit.
    
    Uses Pauli rotation gates for each term in the Hamiltonian.
    """
    for pauli, coeff in hamiltonian.to_list():
        # Create rotation for this Pauli string
        _apply_pauli_rotation(circuit, pauli, parameter * coeff)


def _apply_pauli_rotation(
    circuit: QuantumCircuit,
    pauli_string: str,
    angle: Parameter
) -> None:
    """Apply rotation e^{-i*angle*P} where P is a Pauli string.
    
    Implements the standard decomposition using CNOT ladders.
    """
    n_qubits = len(pauli_string)
    
    # Find non-identity Pauli operators
    active_qubits = []
    pauli_gates = []
    
    for i, pauli in enumerate(pauli_string):
        if pauli != 'I':
            active_qubits.append(i)
            pauli_gates.append(pauli)
    
    if not active_qubits:
        return  # Identity operator
    
    # Change basis for X and Y operators
    for qubit, gate in zip(active_qubits, pauli_gates):
        if gate == 'X':
            circuit.h(qubit)
        elif gate == 'Y':
            circuit.sdg(qubit)
            circuit.h(qubit)
    
    # CNOT ladder
    for i in range(len(active_qubits) - 1):
        circuit.cx(active_qubits[i], active_qubits[i + 1])
    
    # Rz rotation on last qubit
    circuit.rz(2 * angle, active_qubits[-1])
    
    # Reverse CNOT ladder
    for i in range(len(active_qubits) - 2, -1, -1):
        circuit.cx(active_qubits[i], active_qubits[i + 1])
    
    # Reverse basis change
    for qubit, gate in zip(active_qubits, pauli_gates):
        if gate == 'X':
            circuit.h(qubit)
        elif gate == 'Y':
            circuit.h(qubit)
            circuit.s(qubit)


def count_circuit_resources(
    circuit: QuantumCircuit
) -> dict:
    """Count circuit resources (gates, depth, etc.).
    
    Returns:
        Dictionary with circuit statistics
    """
    ops = circuit.count_ops()
    
    return {
        'n_qubits': circuit.num_qubits,
        'n_parameters': circuit.num_parameters,
        'depth': circuit.depth(),
        'gate_counts': ops,
        'total_gates': sum(ops.values()),
        'cx_count': ops.get('cx', 0),
    }
