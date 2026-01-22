# API Reference

## High-Level Models

### VQEFoldingModel

```python
class VQEFoldingModel:
    """
    High-level API for VQE-based protein folding.
    
    Combines sequence loading, lattice encoding, Hamiltonian construction,
    and VQE optimization into a single convenient interface.
    """
```

#### Constructor

```python
__init__(
    sequence: Union[str, ProteinSequence],
    lattice_dim: int = 2,
    lattice_size: Optional[int] = None,
    encoding_type: str = 'turn_direction',
    ansatz_type: str = 'hardware_efficient',
    ansatz_depth: int = 3,
    optimizer: str = 'COBYLA',
    backend: str = 'aer_simulator',
    shots: int = 1024,
    constraint_weight: float = 10.0,
    bias_weight: float = 0.1,
)
```

**Parameters**:
- `sequence`: Protein sequence (string like "HPHPPH" or ProteinSequence object)
- `lattice_dim`: Lattice dimension (2 or 3)
- `lattice_size`: Lattice size (auto-determined if None)
- `encoding_type`: 'turn_direction' or 'binary_position'
- `ansatz_type`: 'hardware_efficient' or 'problem_inspired'
- `ansatz_depth`: Number of ansatz layers (circuit depth)
- `optimizer`: Classical optimizer ('COBYLA', 'SPSA', 'SLSQP', 'L-BFGS-B')
- `backend`: Qiskit backend ('aer_simulator', 'statevector_simulator', or device name)
- `shots`: Number of measurement shots per expectation value
- `constraint_weight`: Weight λ for backbone constraints
- `bias_weight`: Weight μ for compactness bias

#### Methods

**run(maxiter: int = 200) -> VQEResult**

Run VQE optimization.

- **Returns**: `VQEResult` with:
  - `optimal_value`: Minimum energy found
  - `optimal_params`: Optimal circuit parameters
  - `optimal_bitstring`: Most likely measurement outcome
  - `n_iterations`: Number of optimization iterations
  - `convergence_history`: Energy vs iteration
  - `ansatz_circuit`: Optimized quantum circuit

**decode_conformation(bitstring: str) -> np.ndarray**

Decode bitstring to 3D coordinates.

- **Parameters**: Binary string from measurement
- **Returns**: (N, d) array of residue positions

**evaluate_energy(conformation: np.ndarray) -> float**

Evaluate classical energy of a conformation.

**validate_conformation(conformation: np.ndarray) -> bool**

Check if conformation satisfies lattice constraints.

---

### QAOAFoldingModel

```python
class QAOAFoldingModel:
    """
    High-level API for QAOA-based protein folding.
    """
```

#### Constructor

```python
__init__(
    sequence: Union[str, ProteinSequence],
    p_layers: int = 1,
    lattice_dim: int = 2,
    lattice_size: Optional[int] = None,
    encoding_type: str = 'turn_direction',
    optimizer: str = 'COBYLA',
    backend: str = 'aer_simulator',
    shots: int = 1024,
)
```

**Parameters**:
- `p_layers`: Number of QAOA layers (circuit depth)
- Other parameters same as VQEFoldingModel

#### Methods

**run(maxiter: int = 100, initial_params: Optional[np.ndarray] = None) -> QAOAResult**

- **Returns**: `QAOAResult` with:
  - `optimal_value`: Minimum cost
  - `optimal_params`: Optimal [β, γ] parameters
  - `optimal_bitstring`: Best solution
  - `solution_distribution`: Dict[bitstring -> probability]
  - `convergence_history`: Cost vs iteration

---

## Data Loading

### load_hp_sequence

```python
def load_hp_sequence(sequence: str) -> ProteinSequence:
    """
    Load HP model sequence.
    
    Args:
        sequence: String of H (hydrophobic) and P (polar) residues
    
    Returns:
        ProteinSequence with HP contact matrix
    
    Example:
        >>> seq = load_hp_sequence("HPHPPH")
        >>> seq.length
        6
    """
```

### load_fasta_sequence

```python
def load_fasta_sequence(
    file_path: str,
    contact_potential: str = 'MJ'
) -> ProteinSequence:
    """
    Load protein from FASTA file.
    
    Args:
        file_path: Path to FASTA file
        contact_potential: 'MJ' for Miyazawa-Jernigan
    
    Returns:
        ProteinSequence with MJ contact matrix
    """
```

---

## Classical Baselines

### simulated_annealing_fold

```python
def simulated_annealing_fold(
    encoding: LatticeEncoding,
    max_iterations: int = 10000,
    initial_temp: float = 10.0,
    cooling_rate: float = 0.95,
    seed: Optional[int] = None
) -> ClassicalFoldingResult:
    """
    Simulated annealing for protein folding.
    
    Returns:
        ClassicalFoldingResult with:
        - conformation: Optimal residue positions
        - energy: Minimum energy
        - n_iterations: Number of iterations
        - method: 'simulated_annealing'
    """
```

### exact_enumeration_fold

```python
def exact_enumeration_fold(
    encoding: LatticeEncoding,
    max_conformations: int = 100000
) -> ClassicalFoldingResult:
    """
    Exact enumeration (brute force).
    
    Only feasible for N ≤ 10.
    """
```

---

## Analysis

### Metrics

```python
def compute_rmsd(
    conformation1: np.ndarray,
    conformation2: np.ndarray,
    align: bool = True
) -> float:
    """Compute Root Mean Square Deviation."""

def compute_energy_gap(
    quantum_energy: float,
    classical_energy: float
) -> float:
    """Compute relative energy gap."""

def analyze_convergence(
    history: List[float]
) -> Dict[str, float]:
    """
    Analyze convergence metrics.
    
    Returns dict with:
    - final_value
    - best_value
    - convergence_rate
    - final_variance
    """
```

### Plotting

```python
def plot_convergence(
    history: List[float],
    title: str = "Optimization Convergence",
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """Plot optimization convergence."""

def plot_conformation_2d(
    conformation: np.ndarray,
    sequence: Optional[str] = None,
    title: str = "Protein Conformation",
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """Plot 2D protein structure."""

def plot_scaling_analysis(
    results: Dict[int, Dict[str, float]],
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """Plot scaling analysis (qubits, time, energy vs N)."""
```

---

## Advanced: Low-Level APIs

### Hamiltonian Construction

```python
from quantum_protein_folding.quantum.hamiltonian import build_hamiltonian

H = build_hamiltonian(
    sequence: ProteinSequence,
    n_qubits: int,
    lattice_dim: int,
    lattice_size: int,
    encoding_type: str = 'turn_direction',
    constraint_weight: float = 10.0,
    bias_weight: float = 0.1
) -> SparsePauliOp
```

### Direct Solver Usage

```python
from quantum_protein_folding.quantum.vqe import VQESolver
from qiskit.quantum_info import SparsePauliOp

H = SparsePauliOp.from_list([('ZZ', 1.0), ('XX', -1.0)])

solver = VQESolver(
    hamiltonian=H,
    n_qubits=2,
    ansatz_depth=3
)

result = solver.run(maxiter=100)
```

---

## Type Definitions

### VQEResult

```python
@dataclass
class VQEResult:
    optimal_value: float
    optimal_params: np.ndarray
    optimal_bitstring: str
    n_iterations: int
    convergence_history: List[float]
    ansatz_circuit: QuantumCircuit
    eigenstate: Optional[np.ndarray] = None
```

### QAOAResult

```python
@dataclass
class QAOAResult:
    optimal_value: float
    optimal_params: np.ndarray
    optimal_bitstring: str
    solution_distribution: Dict[str, float]
    n_iterations: int
    convergence_history: List[float]
    qaoa_circuit: QuantumCircuit
```

### LatticeEncoding

```python
@dataclass
class LatticeEncoding:
    n_qubits: int
    encoding_type: str
    lattice_dim: int
    lattice_size: int
    qubit_map: Dict[int, str]
    hamiltonian: SparsePauliOp
    sequence: ProteinSequence
```
