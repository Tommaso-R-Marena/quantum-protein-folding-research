"""Variational Quantum Eigensolver (VQE) implementation.

Production-ready VQE solver for protein folding Hamiltonians.
"""

import numpy as np
from typing import Optional, Callable, List, Tuple
from dataclasses import dataclass

from qiskit import QuantumCircuit, transpile
from qiskit.primitives import Estimator, BackendEstimator
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel

from quantum_protein_folding.quantum.circuit_builder import (
    build_hardware_efficient_ansatz,
    build_problem_inspired_ansatz,
)
from quantum_protein_folding.quantum.optimizer import VariationalOptimizer, OptimizationResult


@dataclass
class VQEResult:
    """VQE optimization result.
    
    Attributes:
        optimal_value: Minimum energy found
        optimal_params: Optimal circuit parameters
        optimal_bitstring: Most likely measurement outcome
        n_iterations: Optimization iterations
        convergence_history: Energy vs iteration
        ansatz_circuit: Optimized quantum circuit
        eigenstate: Optimal state vector (if available)
    """
    optimal_value: float
    optimal_params: np.ndarray
    optimal_bitstring: str
    n_iterations: int
    convergence_history: List[float]
    ansatz_circuit: QuantumCircuit
    eigenstate: Optional[np.ndarray] = None


class VQESolver:
    """Variational Quantum Eigensolver.
    
    Minimizes expectation value ⟨ψ(θ)|H|ψ(θ)⟩ over parameters θ.
    
    Example:
        >>> from qiskit.quantum_info import SparsePauliOp
        >>> hamiltonian = SparsePauliOp.from_list([('ZZ', 1.0), ('XX', -1.0)])
        >>> solver = VQESolver(hamiltonian, n_qubits=2)
        >>> result = solver.run(maxiter=100)
        >>> print(f"Ground state energy: {result.optimal_value:.4f}")
    """
    
    def __init__(
        self,
        hamiltonian: SparsePauliOp,
        n_qubits: int,
        ansatz_type: str = 'hardware_efficient',
        ansatz_depth: int = 3,
        optimizer: str = 'COBYLA',
        backend: str = 'aer_simulator',
        shots: int = 1024,
        noise_model: Optional[NoiseModel] = None,
        initial_params: Optional[np.ndarray] = None,
    ):
        """Initialize VQE solver.
        
        Args:
            hamiltonian: Problem Hamiltonian (SparsePauliOp)
            n_qubits: Number of qubits
            ansatz_type: 'hardware_efficient' or 'problem_inspired'
            ansatz_depth: Number of ansatz layers
            optimizer: Classical optimizer ('COBYLA', 'SPSA', 'SLSQP')
            backend: Qiskit backend name or 'aer_simulator'
            shots: Number of measurement shots
            noise_model: Optional noise model for simulation
            initial_params: Initial parameter guess
        """
        self.hamiltonian = hamiltonian
        self.n_qubits = n_qubits
        self.ansatz_type = ansatz_type
        self.ansatz_depth = ansatz_depth
        self.shots = shots
        self.noise_model = noise_model
        
        # Build ansatz
        if ansatz_type == 'hardware_efficient':
            self.ansatz, self.params = build_hardware_efficient_ansatz(
                n_qubits, depth=ansatz_depth
            )
        elif ansatz_type == 'problem_inspired':
            self.ansatz, self.params = build_problem_inspired_ansatz(
                n_qubits, lattice_dim=2, depth=ansatz_depth
            )
        else:
            raise ValueError(f"Unknown ansatz type: {ansatz_type}")
        
        self.n_params = len(self.params)
        
        # Initial parameters
        if initial_params is not None:
            self.initial_params = initial_params
        else:
            # Random initialization
            self.initial_params = 2 * np.pi * np.random.rand(self.n_params)
        
        # Set up backend
        if backend == 'aer_simulator' or backend == 'simulator':
            self.backend = AerSimulator(noise_model=noise_model)
        else:
            # For real backends, would use IBMQ provider
            self.backend = AerSimulator()
        
        # Set up estimator (primitive for expectation values)
        self.estimator = Estimator()
        
        # Classical optimizer
        self.classical_optimizer = VariationalOptimizer(method=optimizer)
    
    def run(
        self,
        maxiter: int = 200,
        tol: float = 1e-6,
        callback: Optional[Callable] = None
    ) -> VQEResult:
        """Run VQE optimization.
        
        Args:
            maxiter: Maximum optimization iterations
            tol: Convergence tolerance
            callback: Optional callback function(iteration, energy)
            
        Returns:
            VQEResult with optimized parameters and energy
        """
        self.classical_optimizer.maxiter = maxiter
        self.classical_optimizer.tol = tol
        
        # Objective function: compute ⟨H⟩
        def objective(params_array):
            energy = self._compute_expectation_value(params_array)
            
            if callback is not None:
                callback(self.classical_optimizer.iteration_count, energy)
            
            return energy
        
        # Run optimization
        opt_result = self.classical_optimizer.optimize(
            objective_function=objective,
            initial_params=self.initial_params,
            bounds=[(0, 2*np.pi)] * self.n_params
        )
        
        # Get optimal bitstring by sampling
        optimal_bitstring = self._sample_bitstring(opt_result.optimal_params)
        
        # Build optimal circuit
        optimal_circuit = self.ansatz.assign_parameters(
            dict(zip(self.params, opt_result.optimal_params))
        )
        
        return VQEResult(
            optimal_value=opt_result.optimal_value,
            optimal_params=opt_result.optimal_params,
            optimal_bitstring=optimal_bitstring,
            n_iterations=opt_result.n_iterations,
            convergence_history=opt_result.convergence_history,
            ansatz_circuit=optimal_circuit,
        )
    
    def _compute_expectation_value(self, params_array: np.ndarray) -> float:
        """Compute ⟨ψ(θ)|H|ψ(θ)⟩ via quantum measurements.
        
        Uses Estimator primitive for efficient expectation value calculation.
        """
        # Bind parameters
        param_dict = dict(zip(self.params, params_array))
        bound_circuit = self.ansatz.assign_parameters(param_dict)
        
        # Use Estimator to compute expectation
        job = self.estimator.run(
            circuits=[bound_circuit],
            observables=[self.hamiltonian],
            shots=self.shots
        )
        
        result = job.result()
        expectation_value = result.values[0]
        
        return float(expectation_value)
    
    def _sample_bitstring(self, params_array: np.ndarray) -> str:
        """Sample most likely bitstring from parameterized circuit.
        
        Args:
            params_array: Circuit parameters
            
        Returns:
            Most probable measurement outcome
        """
        from qiskit import QuantumCircuit
        from qiskit.primitives import Sampler
        
        # Bind parameters
        param_dict = dict(zip(self.params, params_array))
        bound_circuit = self.ansatz.assign_parameters(param_dict)
        
        # Add measurements
        qc = QuantumCircuit(self.n_qubits, self.n_qubits)
        qc.compose(bound_circuit, inplace=True)
        qc.measure_all()
        
        # Sample
        sampler = Sampler()
        job = sampler.run(qc, shots=self.shots)
        result = job.result()
        
        # Get most common outcome
        quasi_dist = result.quasi_dists[0]
        most_likely = max(quasi_dist, key=quasi_dist.get)
        
        # Convert to bitstring
        bitstring = format(most_likely, f'0{self.n_qubits}b')
        
        return bitstring
