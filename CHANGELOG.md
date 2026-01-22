# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Complete VQE implementation with hardware-efficient ansatz
- QAOA solver with adjustable p-layers
- HP model and Miyazawa-Jernigan contact potentials
- Turn-based and binary position lattice encodings
- Simulated annealing and exact enumeration baselines
- Comprehensive analysis tools (RMSD, metrics, plotting)
- Full test suite with >85% coverage
- Example notebooks for tutorials and benchmarking
- Extensive documentation (README, API reference, contributing guide)

### Changed
- Refactored Hamiltonian construction for modularity
- Improved optimizer interface with tracking
- Enhanced error handling throughout

### Fixed
- Edge cases in conformation validation
- Numerical stability in RMSD computation

## [0.1.0] - 2026-01-21

### Added
- Initial project structure
- Basic VQE and QAOA implementations
- HP model support
- Lattice encoding framework

[Unreleased]: https://github.com/Tommaso-R-Marena/quantum-protein-folding-research/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Tommaso-R-Marena/quantum-protein-folding-research/releases/tag/v0.1.0
