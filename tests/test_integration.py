"""Integration tests for complete workflows."""

import pytest
import numpy as np

from quantum_protein_folding.models import VQEFoldingModel, QAOAFoldingModel
from quantum_protein_folding.classical import simulated_annealing_fold


class TestEndToEndWorkflow:
    """End-to-end integration tests."""
    
    @pytest.mark.slow
    def test_vqe_complete_workflow(self):
        """Test complete VQE workflow."""
        # Create model
        model = VQEFoldingModel(
            sequence="HPHH",
            lattice_dim=2,
            ansatz_depth=2,
            optimizer='COBYLA'
        )
        
        # Run optimization
        result = model.run(maxiter=30)
        
        # Decode solution
        conformation = model.decode_conformation(result.optimal_bitstring)
        
        # Validate
        is_valid = model.validate_conformation(conformation)
        
        # Evaluate energy
        energy = model.evaluate_energy(conformation)
        
        # Assertions
        assert result.optimal_value is not None
        assert conformation.shape[0] == 4
        assert np.isfinite(energy)
    
    @pytest.mark.slow
    def test_qaoa_complete_workflow(self):
        """Test complete QAOA workflow."""
        model = QAOAFoldingModel(
            sequence="HPHH",
            p_layers=2,
            lattice_dim=2
        )
        
        result = model.run(maxiter=20)
        
        conformation = model.decode_conformation(result.optimal_bitstring)
        energy = model.evaluate_energy(conformation)
        
        assert result.optimal_value is not None
        assert len(result.solution_distribution) > 0
        assert np.isfinite(energy)
    
    def test_classical_workflow(self):
        """Test classical baseline workflow."""
        model = VQEFoldingModel(sequence="HPHH", lattice_dim=2)
        
        result = simulated_annealing_fold(
            model.encoding,
            max_iterations=500,
            seed=42
        )
        
        is_valid = model.validate_conformation(result.conformation)
        energy = model.evaluate_energy(result.conformation)
        
        assert result.energy is not None
        assert np.isfinite(energy)
    
    @pytest.mark.slow
    def test_comparison_workflow(self):
        """Test comparing multiple methods."""
        sequence = "HPHH"
        
        # VQE
        vqe_model = VQEFoldingModel(sequence, lattice_dim=2, ansatz_depth=1)
        vqe_result = vqe_model.run(maxiter=20)
        
        # QAOA
        qaoa_model = QAOAFoldingModel(sequence, p_layers=1, lattice_dim=2)
        qaoa_result = qaoa_model.run(maxiter=20)
        
        # Classical
        classical_result = simulated_annealing_fold(
            vqe_model.encoding,
            max_iterations=500,
            seed=42
        )
        
        # All should produce valid results
        assert vqe_result.optimal_value is not None
        assert qaoa_result.optimal_value is not None
        assert classical_result.energy is not None
        
        # Energies should be in reasonable range
        all_energies = [
            vqe_result.optimal_value,
            qaoa_result.optimal_value,
            classical_result.energy
        ]
        assert all(np.isfinite(e) for e in all_energies)
