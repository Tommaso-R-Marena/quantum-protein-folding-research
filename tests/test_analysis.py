"""Tests for analysis tools."""

import pytest
import numpy as np
import matplotlib.pyplot as plt

from quantum_protein_folding.analysis import (
    compute_rmsd,
    compute_energy_gap,
    compute_overlap,
    analyze_convergence,
    plot_convergence,
    plot_conformation_2d,
)


class TestMetrics:
    """Test analysis metrics."""
    
    def test_rmsd_identical(self):
        """RMSD of identical structures should be zero."""
        conf = np.array([[0, 0], [1, 0], [2, 0]])
        
        rmsd = compute_rmsd(conf, conf)
        
        assert abs(rmsd) < 1e-10
    
    def test_rmsd_translation(self):
        """RMSD should handle translation."""
        conf1 = np.array([[0, 0], [1, 0], [2, 0]])
        conf2 = np.array([[5, 5], [6, 5], [7, 5]])  # Translated
        
        rmsd = compute_rmsd(conf1, conf2, align=True)
        
        # After alignment, should be zero
        assert abs(rmsd) < 1e-10
    
    def test_energy_gap(self):
        """Test energy gap calculation."""
        gap = compute_energy_gap(-5.0, -10.0)
        
        # Gap = (quantum - classical) / |classical|
        expected = (-5.0 - (-10.0)) / 10.0
        assert abs(gap - expected) < 1e-10
    
    def test_overlap(self):
        """Test quantum state overlap."""
        state1 = np.array([1.0, 0.0, 0.0, 0.0])
        state2 = np.array([0.0, 1.0, 0.0, 0.0])
        
        overlap = compute_overlap(state1, state2)
        
        # Orthogonal states
        assert abs(overlap) < 1e-10
        
        # Same state
        overlap_same = compute_overlap(state1, state1)
        assert abs(overlap_same - 1.0) < 1e-10
    
    def test_convergence_analysis(self):
        """Test convergence analysis."""
        history = [10.0, 8.0, 6.0, 5.0, 4.5, 4.2, 4.1, 4.0]
        
        stats = analyze_convergence(history)
        
        assert stats['final_value'] == 4.0
        assert stats['best_value'] == 4.0
        assert stats['n_iterations'] == len(history)


class TestPlotting:
    """Test plotting functions."""
    
    def test_plot_convergence(self):
        """Test convergence plotting."""
        history = [10.0, 8.0, 6.0, 5.0, 4.5]
        
        fig = plot_convergence(history, show=False)
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_conformation_2d(self):
        """Test 2D conformation plotting."""
        conf = np.array([[0, 0], [1, 0], [1, 1]])
        
        fig = plot_conformation_2d(conf, sequence="HPH", show=False)
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_3d_raises(self):
        """3D conformation should raise error in 2D plot."""
        conf_3d = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]])
        
        with pytest.raises(ValueError):
            plot_conformation_2d(conf_3d, show=False)
