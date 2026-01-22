"""Quantum Approximate Optimization Algorithm (QAOA) implementation.

QAOA for QUBO-formulated protein folding problems.
"""

import numpy as np
from typing import Optional, Callable, List, Dict
from dataclasses import dataclass

from qiskit import QuantumCircuit
from qiskit.primitives import Estimator, Sampler
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel

from quantum_protein_folding.quantum.circuit_builder import build_qaoa_circuit
from quantum_protein_folding.quantum.optimizer import VariationalOptimizer


@dataclass
class QAOAResult:
    """QAOA optimization result.
    
    Attributes:
        optimal_value: Minimum cost found
        optimal_params: Optimal [β, γ] parameters
        optimal_bitstring: Best solution bitstring
        solution_distribution: Distribution over bitstrings
        n_iterations: Optimization iterations
        convergence_history: Cost vs iteration
        qaoa_circuit: Optimized QAOA circuit
    """
    optimal_value: float
    optimal_params: np.ndarray
    optimal_bitstring: str
    solution_distribution: Dict[str, float]
    n_iterations: int
    convergence_history: List[float]
    qaoa_circuit: QuantumCircuit


class QAOASolver:
    """Quantum Approximate Optimization Algorithm.
    
    Approximates solution to combinatorial optimization problems
    via parameterized quantum circuits.
    
    Example:
        >>> hamiltonian = SparsePauliOp.from_list([('ZZ', 1.0), ('Z', -0.5)])
        >>> solver = QAOASolver(hamiltonian, p_layers=3)
        >>> result = solver.run(maxiter=100)
        >>> print(f"Best solution: {result.optimal_bitstring}")
    """
    
    def __init__(
        self,
        hamiltonian: SparsePauliOp,
        p_layers: int = 1,
        mixer_hamiltonian: Optional[SparsePauliOp] = None,
        optimizer: str = 'COBYLA',
        backend: str = 'aer_simulator',
        shots: int = 1024,
        noise_model: Optional[NoiseModel] = None,
    ):
        """Initialize QAOA solver.
        
        Args:
            hamiltonian: Cost Hamiltonian H_C
            p_layers: Number of QAOA layers (depth)
            mixer_hamiltonian: Mixer H_M (default: X mixer)
            optimizer: Classical optimizer
            backend: Quantum backend
            shots: Measurement shots
            noise_model: Optional noise model
        """
        self.hamiltonian = hamiltonian
        self.n_qubits = hamiltonian.num_qubits
        self.p_layers = p_layers
        self.mixer_hamiltonian = mixer_hamiltonian
        self.shots = shots
        self.noise_model = noise_model
        
        # Build QAOA circuit
        self.circuit, self.params = build_qaoa_circuit(
            hamiltonian, p_layers, mixer_hamiltonian
        )
        
        self.n_params = len(self.params)
        
        # Backend
        if backend == 'aer_simulator' or backend == 'simulator':
            self.backend = AerSimulator(noise_model=noise_model)
        else:
            self.backend = AerSimulator()
        
        # Primitives
        self.estimator = Estimator()
        self.sampler = Sampler()
        
        # Classical optimizer
        self.classical_optimizer = VariationalOptimizer(method=optimizer)
    
    def run(
        self,
        maxiter: int = 100,
        initial_params: Optional[np.ndarray] = None,
        callback: Optional[Callable] = None
    ) -> QAOAResult:
        """Run QAOA optimization.
        
        Args:
            maxiter: Maximum iterations
            initial_params: Initial parameter guess (default: random)
            callback: Optional callback(iteration, cost)
            
        Returns:
            QAOAResult with optimal solution
        """
        self.classical_optimizer.maxiter = maxiter
        
        # Initial parameters
        if initial_params is None:
            # Common initialization: small random values
            initial_params = np.random.rand(self.n_params) * 0.1
        
        # Objective: compute cost function expectation
        def objective(params):
            cost = self._compute_cost(params)
            
            if callback is not None:
                callback(self.classical_optimizer.iteration_count, cost)
            
            return cost
        
        # Optimize
        opt_result = self.classical_optimizer.optimize(
            objective_function=objective,
            initial_params=initial_params,
        )
        
        # Sample solution distribution
        solution_dist = self._sample_distribution(opt_result.optimal_params)
        
        # Get best bitstring
        best_bitstring = max(solution_dist, key=solution_dist.get)
        
        # Build optimal circuit
        optimal_circuit = self.circuit.assign_parameters(
            dict(zip(self.params, opt_result.optimal_params))
        )
        
        return QAOAResult(
            optimal_value=opt_result.optimal_value,
            optimal_params=opt_result.optimal_params,
            optimal_bitstring=best_bitstring,
            solution_distribution=solution_dist,
            n_iterations=opt_result.n_iterations,
            convergence_history=opt_result.convergence_history,
            qaoa_circuit=optimal_circuit,
        )
    
    def _compute_cost(self, params: np.ndarray) -> float:
        """Compute cost function expectation ⟨H_C⟩."""
        # Bind parameters
        param_dict = dict(zip(self.params, params))
        bound_circuit = self.circuit.assign_parameters(param_dict)
        
        # Compute expectation
        job = self.estimator.run(
            circuits=[bound_circuit],
            observables=[self.hamiltonian],
            shots=self.shots
        )
        
        result = job.result()
        cost = result.values[0]
        
        return float(cost)
    
    def _sample_distribution(self, params: np.ndarray) -> Dict[str, float]:
        """Sample bitstring distribution from QAOA circuit.
        
        Returns:
            Dictionary mapping bitstrings to probabilities
        """
        # Bind parameters
        param_dict = dict(zip(self.params, params))
        bound_circuit = self.circuit.assign_parameters(param_dict)
        
        # Add measurements
        qc = QuantumCircuit(self.n_qubits, self.n_qubits)
        qc.compose(bound_circuit, inplace=True)
        qc.measure_all()
        
        # Sample
        job = self.sampler.run(qc, shots=self.shots)
        result = job.result()
        
        quasi_dist = result.quasi_dists[0]
        
        # Convert to bitstring distribution
        distribution = {}
        for outcome, prob in quasi_dist.items():
            bitstring = format(outcome, f'0{self.n_qubits}b')
            distribution[bitstring] = prob
        
        return distribution
    
    def evaluate_solution(
        self,
        bitstring: str
    ) -> float:
        """Evaluate cost of a specific bitstring.
        
        Args:
            bitstring: Binary string solution
            
        Returns:
            Cost value
        """
        # Convert bitstring to computational basis state
        state_vector = np.zeros(2 ** self.n_qubits, dtype=complex)
        state_index = int(bitstring, 2)
        state_vector[state_index] = 1.0
        
        # Compute ⟨ψ|H|ψ⟩
        H_matrix = self.hamiltonian.to_matrix()
        cost = np.real(state_vector.conj() @ H_matrix @ state_vector)
        
        return float(cost)
