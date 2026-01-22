# Quantum Protein Folding Research

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Qiskit](https://img.shields.io/badge/Qiskit-1.0+-6929C4.svg)](https://qiskit.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Production-ready quantum algorithms for lattice protein folding using VQE and QAOA**

*Developed by Tommaso R. Marena at The Catholic University of America*

---

## 📋 Overview

This repository implements state-of-the-art quantum algorithms for protein structure prediction on lattice models. The codebase provides:

- **Variational Quantum Eigensolver (VQE)** with hardware-efficient ansätze
- **Quantum Approximate Optimization Algorithm (QAOA)** for combinatorial folding
- **Classical baselines** (simulated annealing, exact enumeration)
- **Comprehensive analysis tools** (metrics, visualization, benchmarking)
- **Production-ready pipeline** with noise handling and multiple backends

### Key Features

✅ **Multiple encoding schemes**: Turn-based and binary position encoding  
✅ **Flexible Hamiltonians**: Contact energy (Miyazawa-Jernigan), backbone constraints, compactness bias  
✅ **Quantum circuits**: Hardware-efficient and problem-inspired ansätze  
✅ **Classical optimizers**: COBYLA, SPSA, L-BFGS-B with convergence tracking  
✅ **Benchmarking framework**: Compare quantum vs classical performance  
✅ **Visualization suite**: Convergence plots, structure visualization, energy landscapes  

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Tommaso-R-Marena/quantum-protein-folding-research.git
cd quantum-protein-folding-research

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

### Dependencies

- Python ≥ 3.10
- Qiskit ≥ 1.0
- NumPy, SciPy, Matplotlib
- Optional: Jupyter for notebooks

### Basic Usage

```python
from quantum_protein_folding.models import VQEFoldingModel

# Define HP sequence
sequence = "HPHPPHHPHH"

# Initialize VQE model
model = VQEFoldingModel(
    sequence=sequence,
    lattice_dim=2,
    ansatz_type='hardware_efficient',
    ansatz_depth=3
)

# Run optimization
result = model.run(maxiter=200)

# Decode and visualize
conformation = model.decode_conformation(result.optimal_bitstring)
energy = model.evaluate_energy(conformation)

print(f"Optimal energy: {result.optimal_value:.4f}")
print(f"Conformation:\n{conformation}")
```

---

## 📓 Interactive Tutorial

**Explore the complete workflow in our comprehensive Jupyter notebook:**

### [**📖 View the Tutorial Notebook**](examples/basic_usage.ipynb)

The notebook covers:

1. **Basic VQE folding** - Step-by-step protein folding with VQE
2. **QAOA implementation** - Alternative quantum approach
3. **Classical comparison** - Benchmarking against simulated annealing
4. **Metrics & visualization** - RMSD, energy gaps, convergence analysis
5. **Scaling analysis** - Resource requirements vs sequence length
6. **Advanced topics** - Exact diagonalization, custom Hamiltonians

### Run in Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tommaso-R-Marena/quantum-protein-folding-research/blob/main/examples/basic_usage.ipynb)

### Run Locally

```bash
jupyter notebook examples/basic_usage.ipynb
```

---

## 🏗️ Repository Structure

```
quantum-protein-folding-research/
├── src/quantum_protein_folding/
│   ├── data/               # Data loaders and preprocessing
│   │   ├── loaders.py      # HP, FASTA, PDB parsers
│   │   └── preprocess.py   # Lattice encoding and constraints
│   ├── quantum/            # Quantum algorithms
│   │   ├── hamiltonian.py  # Hamiltonian construction
│   │   ├── circuit_builder.py  # Ansätze and QAOA circuits
│   │   ├── optimizer.py    # Classical optimizers
│   │   ├── vqe.py          # VQE solver
│   │   └── qaoa.py         # QAOA solver
│   ├── models/             # High-level APIs
│   │   ├── vqe_model.py    # VQE folding model
│   │   └── qaoa_model.py   # QAOA folding model
│   ├── classical/          # Classical baselines
│   │   ├── energy.py       # Energy calculations
│   │   └── baseline.py     # Simulated annealing, exact enumeration
│   └── analysis/           # Analysis and visualization
│       ├── metrics.py      # RMSD, energy gaps, convergence
│       └── plots.py        # Visualization functions
├── examples/
│   └── basic_usage.ipynb   # Comprehensive tutorial
├── tests/                  # Unit tests
├── benchmarks/             # Benchmarking scripts
├── requirements.txt
└── README.md
```

---

## 🔬 Scientific Background

### Problem Formulation

Protein folding on 2D/3D lattices minimizes the Hamiltonian:

```
H = H_contact + λ·H_backbone + μ·H_bias
```

Where:
- **H_contact**: Inter-residue contact energies (Miyazawa-Jernigan matrix)
- **H_backbone**: Chain connectivity and self-avoidance constraints
- **H_bias**: Compactness regularization

### Quantum Algorithms

#### VQE (Variational Quantum Eigensolver)

Minimizes ground state energy via parameterized quantum circuits:

```
min_θ ⟨ψ(θ)|H|ψ(θ)⟩
```

**Ansatz options:**
- Hardware-efficient: Alternating rotation and entanglement layers
- Problem-inspired: Lattice-topology-aware circuits

#### QAOA (Quantum Approximate Optimization Algorithm)

Approximates solutions to combinatorial optimization:

```
|ψ(β,γ)⟩ = ∏ₚ e^(-iβₚH_M) e^(-iγₚH_C) |+⟩^⊗n
```

---

## 📊 Example Results

### VQE Convergence

```python
from quantum_protein_folding.analysis.plots import plot_convergence

plot_convergence(
    result.convergence_history,
    title="VQE Optimization",
    save_path="figures/vqe_convergence.png"
)
```

### Conformation Visualization

```python
from quantum_protein_folding.analysis.plots import plot_conformation_2d

plot_conformation_2d(
    conformation,
    sequence="HPHPPHHPHH",
    title="Optimized Structure",
    save_path="figures/structure.png"
)
```

### Benchmarking

```python
from quantum_protein_folding.classical import simulated_annealing_fold

# Classical baseline
classical_result = simulated_annealing_fold(
    encoding, max_iterations=5000
)

# Compare
rmsd = compute_rmsd(vqe_conformation, classical_result.conformation)
energy_gap = compute_energy_gap(vqe_energy, classical_result.energy)

print(f"RMSD: {rmsd:.4f}")
print(f"Energy gap: {energy_gap*100:.2f}%")
```

---

## 🧪 Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test module
pytest tests/test_vqe.py -v

# Run with coverage
pytest --cov=quantum_protein_folding tests/
```

---

## 📈 Benchmarking

```bash
# Run benchmarks for scaling analysis
python benchmarks/scaling_benchmark.py

# Compare quantum vs classical
python benchmarks/quantum_vs_classical.py

# Noise robustness analysis
python benchmarks/noise_analysis.py
```

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@software{marena2026quantum,
  author = {Marena, Tommaso R.},
  title = {Quantum Protein Folding: VQE and QAOA Implementations},
  year = {2026},
  institution = {The Catholic University of America},
  url = {https://github.com/Tommaso-R-Marena/quantum-protein-folding-research}
}
```

---

## 📬 Contact

**Tommaso R. Marena**  
*Undergraduate Researcher*  
The Catholic University of America  
📧 [marena@cua.edu](mailto:marena@cua.edu)  
🔗 [GitHub](https://github.com/Tommaso-R-Marena)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **IBM Quantum** for Qiskit framework
- **The Catholic University of America** for research support
- **PennyLane team** for quantum computing inspiration
- Classical protein folding community for benchmark datasets

---

## 🔗 Related Work

- [Qiskit Documentation](https://qiskit.org/documentation/)
- [VQE Tutorial](https://qiskit.org/textbook/ch-applications/vqe-molecules.html)
- [QAOA Tutorial](https://qiskit.org/textbook/ch-applications/qaoa.html)
- [Protein Folding on Lattices](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3985476/)

---

## 📊 Performance Highlights

| Sequence Length | Qubits | VQE Time (s) | Classical Time (s) | Energy Gap |
|----------------|--------|--------------|-------------------|------------|
| 4              | 6      | 2.3          | 0.8               | +2.1%      |
| 6              | 10     | 5.7          | 2.4               | +3.4%      |
| 8              | 14     | 12.4         | 6.1               | +1.8%      |
| 10             | 18     | 28.9         | 15.3              | +2.9%      |

*Results on AerSimulator with 1024 shots, COBYLA optimizer*

---

**Made with ❤️ and quantum computing at CUA**
