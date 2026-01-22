"""Tests for analysis tools."""

import pytest
import numpy as np
from quantum_protein_folding.analysis.metrics import (
    compute_rmsd,
    compute_energy_gap,
    compute_contact_map,
    analyze_convergence,
)


class TestMetrics:
    """Test analysis metrics."""
    
    def test_rmsd_identical(self):
        """Test RMSD of identical structures is zero."""
        coords = np.array([[0, 0], [1, 0], [2, 0]])
        
        rmsd = compute_rmsd(coords, coords)
        
        assert np.isclose(rmsd, 0.0, atol=1e-6)
    
    def test_rmsd_translation(self):
        """Test RMSD with aligned structures."""
        coords1 = np.array([[0, 0], [1, 0], [2, 0]])
        coords2 = np.array([[5, 5], [6, 5], [7, 5]])  # Translated
        
        rmsd = compute_rmsd(coords1, coords2, align=True)
        
        # After alignment, should be zero
        assert np.isclose(rmsd, 0.0, atol=1e-6)
    
    def test_rmsd_without_alignment(self):
        """Test RMSD without alignment."""
        coords1 = np.array([[0, 0], [1, 0]])
        coords2 = np.array([[1, 0], [2, 0]])
        
        rmsd = compute_rmsd(coords1, coords2, align=False)
        
        assert rmsd > 0
    
    def test_energy_gap(self):
        """Test energy gap calculation."""
        gap = compute_energy_gap(-1.0, -2.0)
        
        assert isinstance(gap, (float, np.floating))
        # (-1 - (-2)) / |-2| = 1/2 = 0.5
        assert np.isclose(gap, 0.5)
    
    def test_contact_map(self):
        """Test contact map computation."""
        coords = np.array([
            [0, 0],
            [1, 0],
            [2, 0]
        ])
        
        contact_map = compute_contact_map(coords, cutoff=1.5)
        
        assert contact_map.shape == (3, 3)
        assert contact_map[0, 1] == 1  # Adjacent residues
        assert contact_map[1, 2] == 1
        assert contact_map[0, 2] == 0  # Distance = 2 > cutoff
    
    def test_contact_map_symmetry(self):
        """Test contact map is symmetric."""
        coords = np.array([[0, 0], [1, 1], [2, 0]])
        
        contact_map = compute_contact_map(coords)
        
        assert np.allclose(contact_map, contact_map.T)


class TestConvergenceAnalysis:
    """Test convergence analysis."""
    
    def test_analyze_simple_convergence(self):
        """Test convergence analysis on simple history."""
        history = [10.0, 5.0, 2.0, 1.0, 0.5, 0.5, 0.5]
        
        metrics = analyze_convergence(history)
        
        assert 'final_value' in metrics
        assert 'best_value' in metrics
        assert 'n_iterations' in metrics
        
        assert np.isclose(metrics['final_value'], 0.5)
        assert np.isclose(metrics['best_value'], 0.5)
        assert metrics['n_iterations'] == 7
    
    def test_convergence_empty_history(self):
        """Test handling of empty history."""
        metrics = analyze_convergence([])
        
        assert metrics == {}
