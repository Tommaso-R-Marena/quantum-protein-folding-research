"""Visualization functions for protein folding analysis."""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Dict, Tuple
from pathlib import Path


def plot_convergence(
    history: List[float],
    title: str = "Optimization Convergence",
    xlabel: str = "Iteration",
    ylabel: str = "Energy",
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """Plot optimization convergence history.
    
    Args:
        history: List of objective values
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        save_path: Path to save figure
        show: Whether to display plot
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    iterations = np.arange(len(history))
    ax.plot(iterations, history, 'b-', linewidth=2, alpha=0.7)
    
    # Mark best value
    best_idx = np.argmin(history)
    best_value = history[best_idx]
    ax.plot(best_idx, best_value, 'r*', markersize=15, label=f'Best: {best_value:.4f}')
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig


def plot_conformation_2d(
    conformation: np.ndarray,
    sequence: Optional[str] = None,
    title: str = "Protein Conformation",
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """Plot 2D protein conformation.
    
    Args:
        conformation: (N, 2) residue positions
        sequence: Optional sequence string for coloring
        title: Plot title
        save_path: Path to save figure
        show: Whether to display
        
    Returns:
        Matplotlib figure
    """
    if conformation.shape[1] != 2:
        raise ValueError("This function only plots 2D conformations")
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot bonds
    for i in range(len(conformation) - 1):
        ax.plot(
            [conformation[i, 0], conformation[i+1, 0]],
            [conformation[i, 1], conformation[i+1, 1]],
            'k-', linewidth=2, alpha=0.5
        )
    
    # Plot residues
    if sequence is not None:
        # Color by residue type (H=red, P=blue for HP model)
        colors = ['red' if res == 'H' else 'blue' for res in sequence]
    else:
        colors = 'green'
    
    ax.scatter(
        conformation[:, 0],
        conformation[:, 1],
        c=colors,
        s=200,
        edgecolors='black',
        linewidth=2,
        zorder=10
    )
    
    # Label residues
    for i, pos in enumerate(conformation):
        ax.text(
            pos[0], pos[1], str(i),
            ha='center', va='center',
            fontsize=10, fontweight='bold',
            color='white'
        )
    
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig


def plot_energy_landscape(
    energies: Dict[str, float],
    title: str = "Energy Landscape",
    save_path: Optional[str] = None,
    show: bool = True,
    max_display: int = 20
) -> plt.Figure:
    """Plot energy landscape (bar chart of conformations).
    
    Args:
        energies: Dict mapping bitstring -> energy
        title: Plot title
        save_path: Save path
        show: Whether to display
        max_display: Maximum conformations to show
        
    Returns:
        Figure
    """
    # Sort by energy
    sorted_items = sorted(energies.items(), key=lambda x: x[1])[:max_display]
    
    bitstrings = [item[0][:8] + '...' if len(item[0]) > 8 else item[0] 
                  for item in sorted_items]
    energy_values = [item[1] for item in sorted_items]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x_pos = np.arange(len(bitstrings))
    bars = ax.bar(x_pos, energy_values, alpha=0.7, edgecolor='black')
    
    # Color lowest energy
    bars[0].set_color('green')
    bars[0].set_alpha(0.9)
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(bitstrings, rotation=45, ha='right')
    ax.set_xlabel('Conformation (bitstring)', fontsize=12)
    ax.set_ylabel('Energy', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig


def plot_scaling_analysis(
    results: Dict[int, Dict[str, float]],
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """Plot scaling analysis (qubits, time, energy vs chain length).
    
    Args:
        results: Dict[chain_length -> {'n_qubits', 'time', 'energy', ...}]
        save_path: Save path
        show: Whether to display
        
    Returns:
        Figure with subplots
    """
    chain_lengths = sorted(results.keys())
    
    n_qubits = [results[n]['n_qubits'] for n in chain_lengths]
    times = [results[n].get('time', 0) for n in chain_lengths]
    energies = [results[n].get('energy', 0) for n in chain_lengths]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Qubits scaling
    axes[0].plot(chain_lengths, n_qubits, 'o-', linewidth=2, markersize=8)
    axes[0].set_xlabel('Chain Length (N)', fontsize=11)
    axes[0].set_ylabel('Number of Qubits', fontsize=11)
    axes[0].set_title('Qubit Scaling', fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Time scaling
    axes[1].plot(chain_lengths, times, 's-', linewidth=2, markersize=8, color='orange')
    axes[1].set_xlabel('Chain Length (N)', fontsize=11)
    axes[1].set_ylabel('Time (s)', fontsize=11)
    axes[1].set_title('Time-to-Solution', fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    # Energy scaling
    axes[2].plot(chain_lengths, energies, '^-', linewidth=2, markersize=8, color='green')
    axes[2].set_xlabel('Chain Length (N)', fontsize=11)
    axes[2].set_ylabel('Ground State Energy', fontsize=11)
    axes[2].set_title('Energy Scaling', fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig
