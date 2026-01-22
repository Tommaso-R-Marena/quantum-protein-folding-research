#!/usr/bin/env python3
"""Quick start example for quantum protein folding.

Author: Tommaso R. Marena (marena@cua.edu)
Institution: The Catholic University of America
"""

import sys
from pathlib import Path

# Add src to path if running from examples directory
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from quantum_protein_folding.models import VQEFoldingModel, QAOAFoldingModel
from quantum_protein_folding.analysis.plots import plot_convergence, plot_conformation_2d
from quantum_protein_folding.analysis.metrics import compute_rmsd, analyze_convergence
from quantum_protein_folding.classical import simulated_annealing_fold


def main():
    """Run a complete quantum protein folding workflow."""
    
    print("="*60)
    print("Quantum Protein Folding: Quick Start Demo")
    print("Tommaso R. Marena - The Catholic University of America")
    print("="*60)
    
    # Define sequence
    sequence = "HPHPPHHPHH"
    print(f"\nSequence: {sequence}")
    print(f"Length: {len(sequence)} residues")
    
    # ========== VQE ==========
    print("\n" + "="*60)
    print("1. VQE OPTIMIZATION")
    print("="*60)
    
    vqe_model = VQEFoldingModel(
        sequence=sequence,
        lattice_dim=2,
        ansatz_type='hardware_efficient',
        ansatz_depth=2,
        optimizer='COBYLA'
    )
    
    print(f"\nQubits required: {vqe_model.encoding.n_qubits}")
    print(f"Parameters: {vqe_model.solver.n_params}")
    print("\nRunning VQE (this may take 1-2 minutes)...")
    
    vqe_result = vqe_model.run(maxiter=100)
    
    print(f"\n✅ VQE Complete!")
    print(f"  Energy: {vqe_result.optimal_value:.4f}")
    print(f"  Iterations: {vqe_result.n_iterations}")
    print(f"  Bitstring: {vqe_result.optimal_bitstring}")
    
    # Analyze convergence
    conv_stats = analyze_convergence(vqe_result.convergence_history)
    print(f"\n  Convergence rate: {conv_stats['convergence_rate']:.4f}")
    print(f"  Final variance: {conv_stats['final_variance']:.6f}")
    
    # Decode conformation
    vqe_conf = vqe_model.decode_conformation(vqe_result.optimal_bitstring)
    print(f"\n  Conformation:\n{vqe_conf}")
    
    # ========== QAOA ==========
    print("\n" + "="*60)
    print("2. QAOA OPTIMIZATION")
    print("="*60)
    
    qaoa_model = QAOAFoldingModel(
        sequence=sequence,
        p_layers=2,
        lattice_dim=2,
        optimizer='COBYLA'
    )
    
    print(f"\nQAOA depth (p): {qaoa_model.solver.p_layers}")
    print("Running QAOA...")
    
    qaoa_result = qaoa_model.run(maxiter=60)
    
    print(f"\n✅ QAOA Complete!")
    print(f"  Cost: {qaoa_result.optimal_value:.4f}")
    print(f"  Bitstring: {qaoa_result.optimal_bitstring}")
    
    qaoa_conf = qaoa_model.decode_conformation(qaoa_result.optimal_bitstring)
    
    # ========== CLASSICAL BASELINE ==========
    print("\n" + "="*60)
    print("3. CLASSICAL BASELINE (Simulated Annealing)")
    print("="*60)
    
    print("\nRunning simulated annealing...")
    classical_result = simulated_annealing_fold(
        vqe_model.encoding,
        max_iterations=3000,
        seed=42
    )
    
    print(f"\n✅ Classical Complete!")
    print(f"  Energy: {classical_result.energy:.4f}")
    print(f"  Iterations: {classical_result.n_iterations}")
    
    # ========== COMPARISON ==========
    print("\n" + "="*60)
    print("4. COMPARATIVE ANALYSIS")
    print("="*60)
    
    # RMSD
    rmsd_vqe = compute_rmsd(vqe_conf, classical_result.conformation)
    rmsd_qaoa = compute_rmsd(qaoa_conf, classical_result.conformation)
    
    print(f"\nEnergies:")
    print(f"  VQE:       {vqe_result.optimal_value:.4f}")
    print(f"  QAOA:      {qaoa_result.optimal_value:.4f}")
    print(f"  Classical: {classical_result.energy:.4f}")
    
    print(f"\nRMSD vs Classical (lattice units):")
    print(f"  VQE:  {rmsd_vqe:.4f}")
    print(f"  QAOA: {rmsd_qaoa:.4f}")
    
    # Determine best method
    best_energy = min(
        vqe_result.optimal_value,
        qaoa_result.optimal_value,
        classical_result.energy
    )
    
    if abs(vqe_result.optimal_value - best_energy) < 1e-6:
        best_method = "VQE"
    elif abs(qaoa_result.optimal_value - best_energy) < 1e-6:
        best_method = "QAOA"
    else:
        best_method = "Classical"
    
    print(f"\n🏆 Best method: {best_method}")
    
    # ========== VISUALIZATION ==========
    print("\n" + "="*60)
    print("5. VISUALIZATION")
    print("="*60)
    
    print("\nGenerating plots...")
    
    # Create output directory
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    # Plot convergence
    plot_convergence(
        vqe_result.convergence_history,
        title="VQE Convergence",
        save_path=output_dir / "vqe_convergence.png",
        show=False
    )
    print(f"  ✓ Saved: {output_dir / 'vqe_convergence.png'}")
    
    # Plot conformations
    plot_conformation_2d(
        vqe_conf,
        sequence=sequence,
        title=f"VQE Structure (E={vqe_result.optimal_value:.4f})",
        save_path=output_dir / "vqe_structure.png",
        show=False
    )
    print(f"  ✓ Saved: {output_dir / 'vqe_structure.png'}")
    
    plot_conformation_2d(
        qaoa_conf,
        sequence=sequence,
        title=f"QAOA Structure (E={qaoa_result.optimal_value:.4f})",
        save_path=output_dir / "qaoa_structure.png",
        show=False
    )
    print(f"  ✓ Saved: {output_dir / 'qaoa_structure.png'}")
    
    plot_conformation_2d(
        classical_result.conformation,
        sequence=sequence,
        title=f"Classical Structure (E={classical_result.energy:.4f})",
        save_path=output_dir / "classical_structure.png",
        show=False
    )
    print(f"  ✓ Saved: {output_dir / 'classical_structure.png'}")
    
    # ========== SUMMARY ==========
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"\n✅ Successfully folded sequence: {sequence}")
    print(f"\n📈 Results saved to: {output_dir.absolute()}")
    print(f"\n🔬 For questions: marena@cua.edu")
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
