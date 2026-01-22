# Quantum Protein Folding: NISQ-Compatible Variational Algorithms

[![CI](https://github.com/Tommaso-R-Marena/quantum-protein-folding-research/workflows/CI/badge.svg)](https://github.com/Tommaso-R-Marena/quantum-protein-folding-research/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

## Overview

This repository provides a **production-ready, publication-grade implementation** of quantum algorithms for protein structure prediction, specifically designed for **Near-term Intermediate-Scale Quantum (NISQ)** devices. The framework implements both **Variational Quantum Eigensolver (VQE)** and **Quantum Approximate Optimization Algorithm (QAOA)** approaches to the lattice protein folding problem, with rigorous mathematical foundations and comprehensive benchmarking capabilities.

### Key Features

- **Rigorous lattice-based protein models** with real contact potentials (Miyazawa-Jernigan MJ10)
- **Production VQE implementation** with hardware-efficient and problem-inspired ansätze
- **Full QAOA stack** for QUBO-encoded folding problems
- **Hybrid quantum-classical optimization** with noise-aware strategies
- **Real protein data integration** (PDB, FASTA formats)
- **Comprehensive benchmarking suite** with classical baselines
- **Publication-ready visualization** and analysis tools
- **Full test coverage** with CI/CD integration

---

## Mathematical Formulation

### Lattice Protein Model

We employ a **coarse-grained lattice representation** where each amino acid residue \( i \) occupies a unique lattice site \( \mathbf{r}_i \in \mathbb{Z}^d \) (typically \( d=2 \) or \( d=3 \)). The protein conformation is subject to:

1. **Chain connectivity**: \( \|\mathbf{r}_{i+1} - \mathbf{r}_i\| = 1 \) for all \( i \)
2. **Self-avoidance**: \( \mathbf{r}_i \neq \mathbf{r}_j \) for \( i \neq j \)

### Energy Hamiltonian

The folding energy is modeled as:

\[
H = H_{\text{contact}} + H_{\text{backbone}} + H_{\text{bias}}
\]

where:

#### Contact Energy
\[
H_{\text{contact}} = \sum_{\substack{i < j \\ |i-j| > 2}} \epsilon_{a_i, a_j} \cdot \delta(\|\mathbf{r}_i - \mathbf{r}_j\|, 1)
\]

- \( \epsilon_{a_i, a_j} \): Miyazawa-Jernigan contact energy matrix for residue types \( a_i, a_j \)
- \( \delta(x, y) \): Kronecker delta function
- Non-local contacts (\( |i-j| > 2 \)) contribute to structure stabilization

#### Backbone Geometry
\[
H_{\text{backbone}} = \lambda_1 \sum_{i=1}^{N-1} (\|\mathbf{r}_{i+1} - \mathbf{r}_i\| - 1)^2 + \lambda_2 \sum_{i=1}^{N-2} P(\theta_i)
\]

- \( P(\theta_i) \): Bond angle penalty enforcing realistic backbone geometry
- \( \lambda_1, \lambda_2 \): Constraint weights

#### Bias Term
\[
H_{\text{bias}} = \mu \sum_{i=1}^N \|\mathbf{r}_i\|^2
\]

Regularization favoring compact structures (\( \mu \ll 1 \)).

### Qubit Encoding

For a protein of length \( N \) on a lattice of dimension \( L^d \):

1. **Binary position encoding**: Each residue position requires \( d \lceil \log_2 L \rceil \) qubits
2. **Direction encoding** (alternative): Each bond direction encoded in \( \lceil \log_2(2d) \rceil \) qubits
3. **Total qubit requirement**: \( n_q = N \cdot d \lceil \log_2 L \rceil \) (position) or \( n_q = (N-1) \lceil \log_2(2d) \rceil \) (direction)

The Hamiltonian is mapped to a qubit operator:

\[
H \rightarrow \sum_{\alpha} h_\alpha P_\alpha
\]

where \( P_\alpha \in \{I, X, Y, Z\}^{\otimes n_q} \) are Pauli strings and \( h_\alpha \in \mathbb{R} \).

---

## Algorithms

### Variational Quantum Eigensolver (VQE)

**Objective**: Minimize \( \langle \psi(\boldsymbol{\theta}) | H | \psi(\boldsymbol{\theta}) \rangle \)

**Ansatz**: Hardware-efficient ansatz with problem-inspired structure:

\[
|\psi(\boldsymbol{\theta})\rangle = \prod_{\ell=1}^L U_\ell(\boldsymbol{\theta}_\ell) |\mathbf{0}\rangle
\]

where each layer \( U_\ell \) consists of:
- Single-qubit rotations: \( R_y(\theta_{i,\ell}) \otimes R_z(\phi_{i,\ell}) \)
- Entangling gates: \( \text{CNOT}_{i,j} \) according to lattice topology

**Optimization Loop**:
1. Prepare \( |\psi(\boldsymbol{\theta})\rangle \) on quantum device
2. Measure \( \langle H \rangle = \sum_\alpha h_\alpha \langle P_\alpha \rangle \)
3. Classical optimizer updates \( \boldsymbol{\theta} \leftarrow \boldsymbol{\theta} - \eta \nabla_{\boldsymbol{\theta}} \langle H \rangle \)
4. Repeat until convergence

### Quantum Approximate Optimization Algorithm (QAOA)

**Problem Hamiltonian**: QUBO formulation of \( H \)

**QAOA Circuit**:
\[
U(\boldsymbol{\beta}, \boldsymbol{\gamma}) = \prod_{p=1}^P e^{-i \beta_p H_M} e^{-i \gamma_p H_C}
\]

- \( H_C \): Cost Hamiltonian (folding energy)
- \( H_M = \sum_i X_i \): Mixer Hamiltonian
- \( P \): Number of QAOA layers

**State Preparation**:
\[
|\psi(\boldsymbol{\beta}, \boldsymbol{\gamma})\rangle = U(\boldsymbol{\beta}, \boldsymbol{\gamma}) |+\rangle^{\otimes n_q}
\]

**Measurement**: Sample bitstrings \( \{x_k\} \) and decode to conformations

---

## Installation

### Prerequisites

- Python ≥ 3.9
- pip or conda
- (Optional) IBM Quantum account for hardware access

### Quick Start

```bash
# Clone repository
git clone https://github.com/Tommaso-R-Marena/quantum-protein-folding-research.git
cd quantum-protein-folding-research

# Install dependencies
pip install -r requirements.txt

# Or use conda
conda env create -f environment.yml
conda activate qpf

# Run tests to verify installation
pytest tests/ -v
```

### Development Installation

```bash
pip install -e .
pip install -r requirements-dev.txt
```

---

## Usage

### Basic Example: HP Lattice Model

```python
from quantum_protein_folding.models import VQEFoldingModel
from quantum_protein_folding.data.loaders import load_hp_sequence

# Load hydrophobic-polar sequence
sequence = load_hp_sequence("HPHPPHHPHH")

# Initialize VQE model
model = VQEFoldingModel(
    sequence=sequence,
    lattice_dim=2,
    lattice_size=5,
    ansatz_type="hardware_efficient",
    ansatz_depth=3,
    optimizer="COBYLA",
    backend="aer_simulator",
    shots=1024
)

# Run optimization
result = model.run(maxiter=200)

# Extract results
optimal_energy = result.optimal_value
optimal_params = result.optimal_params
conformation = model.decode_conformation(result.optimal_bitstring)

print(f"Ground state energy: {optimal_energy:.4f}")
print(f"Optimal conformation: {conformation}")
```

### Real Protein Example: Angiotensin

```python
from quantum_protein_folding.models import QAOAFoldingModel
from quantum_protein_folding.data.loaders import load_pdb_sequence
from quantum_protein_folding.data.preprocess import map_to_lattice

# Load real protein sequence
sequence = load_pdb_sequence("data/raw/pdb/1N9L.pdb")  # Angiotensin
sequence_reduced = sequence[:8]  # Use first 8 residues

# Map to lattice with MJ potentials
lattice_data = map_to_lattice(
    sequence_reduced,
    potential_type="miyazawa_jernigan",
    lattice_dim=3
)

# Initialize QAOA model
model = QAOAFoldingModel(
    hamiltonian=lattice_data.hamiltonian,
    n_qubits=lattice_data.n_qubits,
    p_layers=3,
    optimizer="SPSA",
    backend="ibmq_manila",  # Real hardware
    shots=8192
)

# Run QAOA
result = model.run(maxiter=100)

# Analyze results
from quantum_protein_folding.analysis.metrics import compute_rmsd
from quantum_protein_folding.classical.baseline import simulated_annealing_fold

classical_result = simulated_annealing_fold(lattice_data)
rmsd = compute_rmsd(result.conformation, classical_result.conformation)

print(f"QAOA energy: {result.optimal_value:.4f}")
print(f"Classical energy: {classical_result.energy:.4f}")
print(f"RMSD: {rmsd:.4f} Å")
```

### Scaling Analysis

```python
from quantum_protein_folding.analysis.benchmarking import scaling_benchmark
import numpy as np

chain_lengths = np.arange(4, 12, 2)
results = scaling_benchmark(
    chain_lengths=chain_lengths,
    algorithm="VQE",
    lattice_dim=2,
    n_trials=10,
    backend="aer_simulator"
)

# Plot scaling behavior
from quantum_protein_folding.analysis.plots import plot_scaling_analysis
plot_scaling_analysis(results, save_path="figures/scaling_analysis.png")
```

---

## Repository Structure

```
quantum-protein-folding-research/
├── README.md                    # This file
├── LICENSE                      # MIT License
├── requirements.txt             # Python dependencies
├── environment.yml              # Conda environment
├── setup.py                     # Package installation
├── .gitignore
├── .github/
│   └── workflows/
│       └── ci.yml               # CI/CD pipeline
├── src/
│   └── quantum_protein_folding/
│       ├── __init__.py
│       ├── data/                # Data loading and preprocessing
│       ├── quantum/             # Quantum algorithms (VQE, QAOA)
│       ├── classical/           # Classical baselines
│       ├── models/              # High-level model APIs
│       ├── analysis/            # Metrics and visualization
│       └── utils/               # Utilities and configuration
├── notebooks/                   # Jupyter notebooks with examples
├── data/                        # Protein sequences and processed data
├── tests/                       # Comprehensive test suite
├── configs/                     # Experiment configurations
└── docs/                        # Documentation
```

---

## Experiments and Reproducibility

### Included Experiments

1. **HP Lattice Benchmark** (`notebooks/experiment_01_hp_lattice.ipynb`)
   - Standard HP sequences from literature
   - Comparison with exact enumeration
   - VQE vs QAOA performance

2. **Angiotensin Folding** (`notebooks/experiment_02_angiotensin.ipynb`)
   - Real peptide with MJ potentials
   - Hardware vs simulator comparison
   - Noise resilience analysis

3. **Scaling Analysis** (`notebooks/experiment_03_scaling.ipynb`)
   - Qubit scaling: \( n_q \) vs \( N \)
   - Time-to-solution vs chain length
   - Parameter count vs ansatz depth

### Reproducing Results

All experiments are fully reproducible:

```bash
# Run all benchmarks
python -m quantum_protein_folding.analysis.benchmarking --config configs/benchmark_suite.yaml

# Run specific experiment
jupyter notebook notebooks/experiment_01_hp_lattice.ipynb
```

---

## Testing

### Run Full Test Suite

```bash
pytest tests/ -v --cov=quantum_protein_folding --cov-report=html
```

### Test Coverage

- **Unit tests**: Individual components (Hamiltonian, circuits, encoders)
- **Integration tests**: End-to-end workflows
- **Regression tests**: Numerical accuracy and convergence
- **Hardware tests**: (Optional) Real device validation

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{marena2026quantum,
  title={Quantum Variational Algorithms for Protein Structure Prediction on NISQ Devices},
  author={Marena, Tommaso R.},
  journal={Nature Computational Science},
  year={2026},
  note={In preparation}
}
```

### Related Work

- Robert, A., et al. "Resource-efficient quantum algorithm for protein folding." *npj Quantum Information* 7, 38 (2021).
- Perdomo-Ortiz, A., et al. "Finding low-energy conformations of lattice protein models by quantum annealing." *Scientific Reports* 2, 571 (2012).
- Fingerhuth, M., et al. "A quantum alternating operator ansatz with hard and soft constraints for lattice protein folding." *arXiv:1810.13411* (2018).

See `docs/bibliography.md` for full reference list.

---

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Please ensure:
- All tests pass (`pytest tests/`)
- Code follows style guide (`black src/ tests/`)
- Documentation is updated

---

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file.

---

## Contact

**Tommaso R. Marena**  
The Catholic University of America  
Email: [your-email]@cua.edu  
GitHub: [@Tommaso-R-Marena](https://github.com/Tommaso-R-Marena)

---

## Acknowledgments

- IBM Quantum for hardware access
- Qiskit development team
- Dr. Katherine Havanki for research guidance
- CUA Department of Chemistry for computational resources
