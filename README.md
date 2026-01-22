# QuantumFold-Advantage: Quantum Protein Folding Research

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Qiskit](https://img.shields.io/badge/Qiskit-1.0+-6133BD.svg)](https://qiskit.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Platform-agnostic quantum algorithms for protein structure prediction, exploring quantum advantage in computational biology.**

## 🧬 Overview

QuantumFold-Advantage is a research platform implementing variational quantum eigensolver (VQE) and quantum approximate optimization algorithm (QAOA) approaches to the lattice protein folding problem. This project bridges quantum computing and computational biology, providing:

- **Production-ready implementations** of VQE and QAOA for protein folding
- **Multiple encoding schemes**: turn-based and binary position encodings
- **Classical baselines**: simulated annealing and exact enumeration for benchmarking
- **Comprehensive analysis tools**: energy metrics, RMSD calculations, and visualization
- **Hardware-efficient ansätze** optimized for NISQ devices
- **Flexible backend support**: Qiskit Aer simulators and real quantum hardware

### Key Features

✨ **Quantum Algorithms**
- Variational Quantum Eigensolver (VQE) with customizable ansatz
- Quantum Approximate Optimization Algorithm (QAOA) with tunable depth
- Support for problem-inspired and hardware-efficient circuit designs

🧪 **Protein Models**
- HP (Hydrophobic-Polar) lattice model
- FASTA sequence parsing
- PDB structure loading
- Miyazawa-Jernigan contact energy matrix

📊 **Analysis & Benchmarking**
- RMSD calculations with structural alignment
- Energy landscape visualization
- Convergence analysis and optimization tracking
- Quantum vs. classical performance comparison

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Examples](#usage-examples)
- [Project Structure](#project-structure)
- [Notebooks](#notebooks)
- [Documentation](#documentation)
- [Research Context](#research-context)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip or conda package manager

### Install from Source

```bash
# Clone the repository
git clone https://github.com/Tommaso-R-Marena/quantum-protein-folding-research.git
cd quantum-protein-folding-research

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Dependencies

Core requirements:
- `qiskit >= 1.0.0`
- `qiskit-aer >= 0.13.0`
- `numpy >= 1.21.0`
- `scipy >= 1.7.0`
- `matplotlib >= 3.4.0`
- `biopython >= 1.79` (for FASTA/PDB parsing)

## ⚡ Quick Start

### VQE for Protein Folding

```python
from quantum_protein_folding.models import VQEFoldingModel

# Define HP sequence
sequence = "HPHPPHHPHH"

# Create VQE model
model = VQEFoldingModel(
    sequence=sequence,
    lattice_dim=2,
    ansatz_depth=3,
    optimizer='COBYLA'
)

# Run optimization
result = model.run(maxiter=200)

print(f"Optimal energy: {result.optimal_value:.4f}")
print(f"Best bitstring: {result.optimal_bitstring}")

# Decode to 3D structure
conformation = model.decode_conformation(result.optimal_bitstring)
print(f"Conformation:\n{conformation}")
```

### QAOA for Protein Folding

```python
from quantum_protein_folding.models import QAOAFoldingModel

# Create QAOA model with 3 layers
model = QAOAFoldingModel(
    sequence="HPHPPHHPHH",
    p_layers=3,
    lattice_dim=2
)

# Run optimization
result = model.run(maxiter=100)

print(f"Best solution: {result.optimal_bitstring}")
print(f"Cost: {result.optimal_value:.4f}")
```

### Classical Baseline Comparison

```python
from quantum_protein_folding.classical import simulated_annealing_fold
from quantum_protein_folding.data.preprocess import map_to_lattice
from quantum_protein_folding.data.loaders import load_hp_sequence

# Load sequence and create encoding
sequence = load_hp_sequence("HPHPPHHPHH")
encoding = map_to_lattice(sequence, lattice_dim=2)

# Run simulated annealing
classical_result = simulated_annealing_fold(
    encoding,
    max_iterations=10000,
    initial_temp=10.0
)

print(f"Classical energy: {classical_result.energy:.4f}")
print(f"Quantum energy: {result.optimal_value:.4f}")
print(f"Energy gap: {result.optimal_value - classical_result.energy:.4f}")
```

## 📚 Usage Examples

### Example 1: Custom Hamiltonian Construction

```python
from quantum_protein_folding.quantum.hamiltonian import build_hamiltonian
from quantum_protein_folding.data.loaders import load_hp_sequence

sequence = load_hp_sequence("HPHPPHHPHH")

hamiltonian = build_hamiltonian(
    sequence=sequence,
    n_qubits=18,
    lattice_dim=2,
    lattice_size=12,
    encoding_type='turn_direction',
    constraint_weight=10.0,
    bias_weight=0.1
)

print(f"Hamiltonian: {hamiltonian}")
print(f"Number of terms: {len(hamiltonian)}")
```

### Example 2: Visualization and Analysis

```python
from quantum_protein_folding.analysis import (
    plot_convergence,
    plot_conformation_2d,
    compute_rmsd
)

# Plot optimization convergence
plot_convergence(
    history=result.convergence_history,
    title="VQE Optimization",
    save_path="results/convergence.png"
)

# Visualize protein conformation
conformation = model.decode_conformation(result.optimal_bitstring)
plot_conformation_2d(
    conformation=conformation,
    sequence=sequence,
    title="Optimal Folding",
    save_path="results/structure.png"
)

# Compare structures
rmsd = compute_rmsd(conformation1, conformation2, align=True)
print(f"RMSD between structures: {rmsd:.3f}")
```

### Example 3: Load Real Protein from PDB

```python
from quantum_protein_folding.data.loaders import load_pdb_sequence

# Load from PDB file
sequence = load_pdb_sequence("data/1CRN.pdb")

print(f"Sequence: {sequence.sequence}")
print(f"Length: {sequence.length}")
print(f"Contact matrix shape: {sequence.contact_matrix.shape}")
```

## 📁 Project Structure

```
quantum-protein-folding-research/
├── src/
│   └── quantum_protein_folding/
│       ├── data/              # Data loaders and preprocessing
│       │   ├── loaders.py     # HP, FASTA, PDB parsers
│       │   └── preprocess.py  # Lattice encoding
│       ├── quantum/           # Quantum algorithms
│       │   ├── hamiltonian.py # Hamiltonian construction
│       │   ├── vqe.py         # VQE implementation
│       │   ├── qaoa.py        # QAOA implementation
│       │   ├── circuit_builder.py  # Ansatz circuits
│       │   └── optimizer.py   # Classical optimizers
│       ├── classical/         # Classical baselines
│       │   ├── energy.py      # Energy calculations
│       │   └── baseline.py    # SA, exact enumeration
│       ├── models/            # High-level APIs
│       │   ├── vqe_model.py
│       │   └── qaoa_model.py
│       └── analysis/          # Analysis tools
│           ├── metrics.py     # RMSD, energy gap
│           └── plots.py       # Visualization
├── notebooks/                 # Jupyter notebooks
│   ├── 01_quick_start.ipynb
│   ├── 02_vqe_tutorial.ipynb
│   ├── 03_qaoa_tutorial.ipynb
│   └── 04_benchmarking.ipynb
├── tests/                     # Unit tests
├── data/                      # Example datasets
├── results/                   # Output directory
├── requirements.txt
├── setup.py
└── README.md
```

## 📓 Notebooks

Interactive Jupyter notebooks demonstrating key functionality:

1. **[Quick Start Guide](notebooks/01_quick_start.ipynb)** - Basic usage and examples
2. **[VQE Tutorial](notebooks/02_vqe_tutorial.ipynb)** - Deep dive into VQE for protein folding
3. **[QAOA Tutorial](notebooks/03_qaoa_tutorial.ipynb)** - QAOA implementation and optimization
4. **[Benchmarking Suite](notebooks/04_benchmarking.ipynb)** - Quantum vs. classical comparison

To run notebooks:

```bash
jupyter notebook notebooks/
```

## 📖 Documentation

### Algorithm Overview

**VQE (Variational Quantum Eigensolver)**

Minimizes the expectation value of the folding Hamiltonian:

```
E(θ) = ⟨ψ(θ)|H|ψ(θ)⟩
```

where `H = H_contact + λ·H_backbone + μ·H_bias`

**QAOA (Quantum Approximate Optimization Algorithm)**

Approximates the ground state via alternating cost and mixer unitaries:

```
|ψ(β, γ)⟩ = ∏ₚ e^(-iβₚH_M) e^(-iγₚH_C) |+⟩^⊗n
```

### Encoding Schemes

1. **Turn-Based Encoding**: Encodes conformations as sequences of directional turns
   - Qubits: `(N-1) · ⌈log₂(2d)⌉` for dimension `d`
   - Natural for lattice walks

2. **Binary Position Encoding**: Encodes each residue position in binary
   - Qubits: `N · d · ⌈log₂(L)⌉` for lattice size `L`
   - Direct spatial representation

### Energy Model

Total energy functional:

```
E = ∑ᵢⱼ εᵢⱼ·δ(rᵢ,rⱼ) + λ·∑ᵢ(|rᵢ₊₁-rᵢ|-1)² + μ·∑ᵢ|rᵢ|²
```

- **Contact term**: Miyazawa-Jernigan potentials
- **Backbone term**: Connectivity constraints
- **Bias term**: Compactness regularization

## 🔬 Research Context

This project investigates the potential for quantum advantage in protein structure prediction, a fundamental problem in computational biology. Key research questions:

1. **Scalability**: How do quantum algorithms scale compared to classical methods?
2. **Accuracy**: Can quantum approaches find lower-energy conformations?
3. **NISQ Viability**: Are current/near-term quantum devices sufficient?
4. **Encoding Efficiency**: Which lattice encoding schemes are most effective?

### Related Publications

*Preprint and publications coming soon*

## 🤝 Contributing

Contributions are welcome! Areas of interest:

- Additional encoding schemes
- Novel ansatz designs
- Noise mitigation strategies
- Integration with real protein databases
- Performance optimizations

Please open an issue or pull request on GitHub.

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@software{quantumfold_advantage,
  author = {Marena, Tommaso R.},
  title = {QuantumFold-Advantage: Quantum Algorithms for Protein Folding},
  year = {2026},
  url = {https://github.com/Tommaso-R-Marena/quantum-protein-folding-research},
  institution = {The Catholic University of America}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

**Tommaso R. Marena**
- Email: [marena@cua.edu](mailto:marena@cua.edu)
- GitHub: [@Tommaso-R-Marena](https://github.com/Tommaso-R-Marena)
- Institution: The Catholic University of America

---

*This research is part of ongoing work at The Catholic University of America exploring quantum computing applications in computational biology and chemistry.*
