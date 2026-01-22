"""Quantum circuit builders for variational ansätze.

Provides various ansatz designs:
    - Hardware-efficient ansatz
    - Problem-inspired ansatz (lattice-aware)
    - Custom parameterized circuits
"""

import numpy as np
from typing import List, Tuple, Optional, Union, Callable
from enum import Enum

try:
    from qiskit import QuantumCircuit, QuantumRegister
    from qiskit.circuit import Parameter, ParameterVector
    from qiskit.circuit.library import TwoLocal, RealAmplitudes, EfficientSU2
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    import warnings
    warnings.warn("Qiskit not installed")


class AnsatzType(Enum):
    """Supported ansatz types."""
    HARDWARE_EFFICIENT = "hardware_efficient"
    PROBLEM_INSPIRED = "problem_inspired"
    REAL_AMPLITUDES = "real_amplitudes"
    EFFICIENT_SU2 = "efficient_su2"
    CUSTOM = "custom"


class EntanglementPattern(Enum):
    """Entanglement connection patterns."""
    LINEAR = "linear"  # Chain: 0-1, 1-2, 2-3, ...
    FULL = "full"  # All-to-all connections
    CIRCULAR = "circular"  # Ring: includes n-1 to 0
    PAIRWISE = "pairwise"  # Adjacent pairs: (0,1), (2,3), ...


class CircuitBuilder:
    """Builder for variational quantum circuits.
    
    Constructs parameterized circuits for VQE and QAOA algorithms.
    
    Attributes:
        n_qubits: Number of qubits
        ansatz_type: Type of ansatz to build
        depth: Number of ansatz layers
        entanglement: Entanglement pattern
        rotation_blocks: Rotation gates for each layer
        entanglement_blocks: Entanglement gates
    
    Example:
        >>> builder = CircuitBuilder(
        ...     n_qubits=6,
        ...     ansatz_type=AnsatzType.HARDWARE_EFFICIENT,
        ...     depth=3
        ... )
        >>> circuit, params = builder.build()
        >>> print(f"Parameters: {len(params)}")
        36  # 6 qubits * 2 rotations * 3 layers
    """
    
    def __init__(
        self,
        n_qubits: int,
        ansatz_type: AnsatzType = AnsatzType.HARDWARE_EFFICIENT,
        depth: int = 1,
        entanglement: EntanglementPattern = EntanglementPattern.LINEAR,
        rotation_blocks: Optional[List[str]] = None,
        entanglement_blocks: Optional[List[str]] = None,
        insert_barriers: bool = False,
    ):
        """Initialize circuit builder.
        
        Args:
            n_qubits: Number of qubits
            ansatz_type: Type of ansatz
            depth: Number of repetitions of rotation + entanglement layers
            entanglement: Pattern for entangling gates
            rotation_blocks: Gates for rotation layer (default: ['ry', 'rz'])
            entanglement_blocks: Gates for entanglement (default: ['cx'])
            insert_barriers: Whether to insert barriers between layers
        """
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit required. Install with: pip install qiskit")
        
        self.n_qubits = n_qubits
        self.ansatz_type = ansatz_type
        self.depth = depth
        self.entanglement = entanglement
        self.rotation_blocks = rotation_blocks or ['ry', 'rz']
        self.entanglement_blocks = entanglement_blocks or ['cx']
        self.insert_barriers = insert_barriers
    
    def build(self) -> Tuple[QuantumCircuit, ParameterVector]:
        """Build the variational circuit.
        
        Returns:
            Tuple of (circuit, parameters)
        
        Raises:
            NotImplementedError: If ansatz type not supported
        """
        if self.ansatz_type == AnsatzType.HARDWARE_EFFICIENT:
            return self._build_hardware_efficient()
        elif self.ansatz_type == AnsatzType.PROBLEM_INSPIRED:
            return self._build_problem_inspired()
        elif self.ansatz_type == AnsatzType.REAL_AMPLITUDES:
            return self._build_real_amplitudes()
        elif self.ansatz_type == AnsatzType.EFFICIENT_SU2:
            return self._build_efficient_su2()
        else:
            raise NotImplementedError(f"Ansatz {self.ansatz_type} not implemented")
    
    def _build_hardware_efficient(self) -> Tuple[QuantumCircuit, ParameterVector]:
        """Build hardware-efficient ansatz.
        
        Structure for each layer:
            1. Single-qubit rotations (RY and RZ on each qubit)
            2. Entangling layer (CX gates according to pattern)
        
        Returns:
            Tuple of (circuit, parameters)
        """
        n_params_per_layer = self.n_qubits * len(self.rotation_blocks)
        total_params = n_params_per_layer * self.depth
        
        # Create parameter vector
        params = ParameterVector('θ', total_params)
        
        # Create circuit
        qc = QuantumCircuit(self.n_qubits)
        
        param_idx = 0
        for layer in range(self.depth):
            # Rotation layer
            for rot_gate in self.rotation_blocks:
                for qubit in range(self.n_qubits):
                    if rot_gate == 'ry':
                        qc.ry(params[param_idx], qubit)
                    elif rot_gate == 'rz':
                        qc.rz(params[param_idx], qubit)
                    elif rot_gate == 'rx':
                        qc.rx(params[param_idx], qubit)
                    else:
                        raise ValueError(f"Unknown rotation gate: {rot_gate}")
                    param_idx += 1
            
            # Entanglement layer
            if layer < self.depth:  # No entanglement after last layer
                self._add_entanglement_layer(qc)
            
            if self.insert_barriers:
                qc.barrier()
        
        return qc, params
    
    def _build_problem_inspired(self) -> Tuple[QuantumCircuit, ParameterVector]:
        """Build problem-inspired ansatz.
        
        For protein folding, this respects the lattice structure by
        entangling qubits that correspond to nearby residues.
        
        Returns:
            Tuple of (circuit, parameters)
        """
        # Similar to hardware-efficient but with custom entanglement
        # that matches the protein chain topology
        
        n_params_per_layer = self.n_qubits * 2  # RY + RZ
        total_params = n_params_per_layer * self.depth
        
        params = ParameterVector('θ', total_params)
        qc = QuantumCircuit(self.n_qubits)
        
        param_idx = 0
        for layer in range(self.depth):
            # Rotation layer
            for qubit in range(self.n_qubits):
                qc.ry(params[param_idx], qubit)
                param_idx += 1
                qc.rz(params[param_idx], qubit)
                param_idx += 1
            
            # Chain-like entanglement (respecting protein backbone)
            for qubit in range(self.n_qubits - 1):
                qc.cx(qubit, qubit + 1)
            
            # Add some long-range entanglement for non-local contacts
            if layer % 2 == 1 and self.n_qubits > 4:
                for qubit in range(0, self.n_qubits - 2, 2):
                    qc.cx(qubit, qubit + 2)
            
            if self.insert_barriers:
                qc.barrier()
        
        return qc, params
    
    def _build_real_amplitudes(self) -> Tuple[QuantumCircuit, ParameterVector]:
        """Build RealAmplitudes ansatz from Qiskit library.
        
        Returns:
            Tuple of (circuit, parameters)
        """
        ansatz = RealAmplitudes(
            self.n_qubits,
            reps=self.depth,
            entanglement=self._get_qiskit_entanglement(),
            insert_barriers=self.insert_barriers
        )
        
        params = ParameterVector('θ', ansatz.num_parameters)
        qc = ansatz.assign_parameters(params)
        
        return qc, params
    
    def _build_efficient_su2(self) -> Tuple[QuantumCircuit, ParameterVector]:
        """Build EfficientSU2 ansatz from Qiskit library.
        
        Returns:
            Tuple of (circuit, parameters)
        """
        ansatz = EfficientSU2(
            self.n_qubits,
            reps=self.depth,
            entanglement=self._get_qiskit_entanglement(),
            insert_barriers=self.insert_barriers
        )
        
        params = ParameterVector('θ', ansatz.num_parameters)
        qc = ansatz.assign_parameters(params)
        
        return qc, params
    
    def _add_entanglement_layer(self, qc: QuantumCircuit) -> None:
        """Add entanglement gates to circuit.
        
        Args:
            qc: Circuit to modify in-place
        """
        if self.entanglement == EntanglementPattern.LINEAR:
            # Chain: 0-1, 1-2, 2-3, ...
            for i in range(self.n_qubits - 1):
                qc.cx(i, i + 1)
        
        elif self.entanglement == EntanglementPattern.CIRCULAR:
            # Ring: includes connection from last to first
            for i in range(self.n_qubits - 1):
                qc.cx(i, i + 1)
            qc.cx(self.n_qubits - 1, 0)
        
        elif self.entanglement == EntanglementPattern.FULL:
            # All-to-all
            for i in range(self.n_qubits):
                for j in range(i + 1, self.n_qubits):
                    qc.cx(i, j)
        
        elif self.entanglement == EntanglementPattern.PAIRWISE:
            # Adjacent pairs
            for i in range(0, self.n_qubits - 1, 2):
                qc.cx(i, i + 1)
        
        else:
            raise ValueError(f"Unknown entanglement pattern: {self.entanglement}")
    
    def _get_qiskit_entanglement(self) -> str:
        """Convert entanglement pattern to Qiskit format.
        
        Returns:
            Qiskit entanglement string
        """
        mapping = {
            EntanglementPattern.LINEAR: 'linear',
            EntanglementPattern.FULL: 'full',
            EntanglementPattern.CIRCULAR: 'circular',
            EntanglementPattern.PAIRWISE: 'pairwise',
        }
        return mapping.get(self.entanglement, 'linear')


def initialize_parameters(
    n_params: int,
    method: str = "random",
    seed: Optional[int] = None,
    scale: float = 0.1
) -> np.ndarray:
    """Initialize variational parameters.
    
    Args:
        n_params: Number of parameters
        method: Initialization method ('random', 'zeros', 'small_random')
        seed: Random seed for reproducibility
        scale: Scale for random initialization
    
    Returns:
        Array of initial parameter values
    
    Example:
        >>> params = initialize_parameters(36, method='random', seed=42)
        >>> params.shape
        (36,)
    """
    if seed is not None:
        np.random.seed(seed)
    
    if method == "zeros":
        return np.zeros(n_params)
    elif method == "random":
        return np.random.uniform(-np.pi, np.pi, n_params)
    elif method == "small_random":
        return np.random.uniform(-scale, scale, n_params)
    elif method == "normal":
        return np.random.normal(0, scale, n_params)
    else:
        raise ValueError(f"Unknown initialization method: {method}")


def build_qaoa_circuit(
    hamiltonian,
    p_layers: int,
    initial_state: Optional[QuantumCircuit] = None
) -> Tuple[QuantumCircuit, ParameterVector, ParameterVector]:
    """Build QAOA circuit.
    
    Args:
        hamiltonian: Problem Hamiltonian (SparsePauliOp)
        p_layers: Number of QAOA layers
        initial_state: Optional custom initial state (default: |+⟩^n)
    
    Returns:
        Tuple of (circuit, beta_params, gamma_params)
    
    Mathematical Structure:
        U(β, γ) = ∏_{k=1}^p e^{-iβ_k H_M} e^{-iγ_k H_C}
    
    Example:
        >>> from qiskit.quantum_info import SparsePauliOp
        >>> H = SparsePauliOp(['ZZ', 'Z'], [1.0, 0.5])
        >>> qc, beta, gamma = build_qaoa_circuit(H, p_layers=2)
    """
    if not QISKIT_AVAILABLE:
        raise ImportError("Qiskit required")
    
    n_qubits = hamiltonian.num_qubits
    
    # Create parameter vectors
    beta = ParameterVector('β', p_layers)
    gamma = ParameterVector('γ', p_layers)
    
    # Create circuit
    qc = QuantumCircuit(n_qubits)
    
    # Initial state (uniform superposition if not specified)
    if initial_state is None:
        for qubit in range(n_qubits):
            qc.h(qubit)
    else:
        qc.compose(initial_state, inplace=True)
    
    # QAOA layers
    for p in range(p_layers):
        # Apply problem Hamiltonian: e^{-iγ H_C}
        qc.compose(
            hamiltonian.exp_i().power(gamma[p]),
            inplace=True
        )
        
        # Apply mixer Hamiltonian: e^{-iβ H_M} = ∏_i e^{-iβ X_i}
        for qubit in range(n_qubits):
            qc.rx(2 * beta[p], qubit)  # RX(θ) = e^{-iθ/2 X}
    
    return qc, beta, gamma
