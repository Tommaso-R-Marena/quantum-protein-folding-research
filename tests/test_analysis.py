"""Tests for analysis and metrics."""

import pytest
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

from quantum_protein_folding.analysis import (
    compute_rmsd,
    compute_energy_gap,
    analyze_convergence,
    compute_contact_map,
    plot_convergence,
    plot_conformation_2d,
)


class TestMetrics:
    """Tests for analysis metrics."""
    
    def test_rmsd_identical(self):
        """Test RMSD of identical structures is zero."""
        conf = np.array([[0, 0], [1, 0], [1, 1]])
        
        rmsd = compute_rmsd(conf, conf, align=False)
        
        assert rmsd < 1e-10
    
    def test_rmsd_translation(self):
        """Test RMSD with translation."""
        conf1 = np.array([[0, 0], [1, 0], [1, 1]])
        conf2 = np.array([[5, 5], [6, 5], [6, 6]])  # Translated
        
        rmsd_aligned = compute_rmsd(conf1, conf2, align=True)
        rmsd_unaligned = compute_rmsd(conf1, conf2, align=False)
        
        assert rmsd_aligned < 1e-6  # Should be ~0 after alignment
        assert rmsd_unaligned > 5.0  # Large before alignment
    
    def test_energy_gap(self):
        """Test energy gap calculation."""
        gap = compute_energy_gap(-1.0, -2.0)
        assert gap == 0.5  # (-1 - (-2)) / |-2| = 0.5
    
    def test_analyze_convergence(self):
        """Test convergence analysis."""
        history = [10.0, 5.0, 3.0, 2.5, 2.1, 2.0, 2.0]
        
        analysis = analyze_convergence(history)
        
        assert 'final_value' in analysis
        assert 'best_value' in analysis
        assert analysis['best_value'] == 2.0
        assert analysis['n_iterations'] == len(history)
    
    def test_contact_map(self):
        """Test contact map computation."""
        conformation = np.array([
            [0, 0],
            [1, 0],
            [1, 1],
            [0, 1]
        ])
        
        contact_map = compute_contact_map(conformation, cutoff=1.5)
        
        assert contact_map.shape == (4, 4)
        assert np.all(contact_map >= 0)
        assert np.all(contact_map <= 1)
        # Check symmetry
        assert np.allclose(contact_map, contact_map.T)


class TestPlotting:
    """Tests for plotting functions."""
    
    def test_plot_convergence(self, tmp_path):
        """Test convergence plotting."""
        history = [10.0, 5.0, 3.0, 2.0, 1.5, 1.0]
        
        save_path = tmp_path / "convergence.png"
        
        fig = plot_convergence(
            history,
            save_path=str(save_path),
            show=False
        )
        
        assert save_path.exists()
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_conformation_2d(self, tmp_path):
        """Test 2D conformation plotting."""
        conformation = np.array([
            [0, 0],
            [1, 0],
            [1, 1],
            [0, 1]
        ])
        
        save_path = tmp_path / "structure.png"
        
        fig = plot_conformation_2d(
            conformation,
            sequence="HPHH",
            save_path=str(save_path),
            show=False
        )
        
        assert save_path.exists()
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_conformation_3d_error(self):
        """Test error for 3D conformation in 2D plot."""
        conformation_3d = np.array([[0, 0, 0], [1, 0, 0]])
        
        with pytest.raises(ValueError, match="2D"):
            plot_conformation_2d(conformation_3d, show=False)
