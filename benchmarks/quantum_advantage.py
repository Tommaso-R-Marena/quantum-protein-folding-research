"""Quantum advantage analysis.

Compare quantum algorithms against classical baselines to identify
regimes where quantum methods show advantage.

Usage:
    python benchmarks/quantum_advantage.py --sequences data/benchmark_sequences.txt
"""

import argparse
import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

from quantum_protein_folding.models import VQEFoldingModel, QAOAFoldingModel
from quantum_protein_folding.classical import simulated_annealing_fold, exact_enumeration_fold
from quantum_protein_folding.data import load_hp_sequence
from quantum_protein_folding.data.preprocess import map_to_lattice
from quantum_protein_folding.analysis import compute_energy_gap


class QuantumAdvantageAnalyzer:
    """Analyze quantum advantage for protein folding."""
    
    def __init__(self, vqe_maxiter: int = 150, qaoa_p: int = 3):
        self.vqe_maxiter = vqe_maxiter
        self.qaoa_p = qaoa_p
        self.results = []
    
    def benchmark_sequence(self, sequence: str, run_exact: bool = False) -> Dict:
        """Benchmark single sequence with all methods.
        
        Args:
            sequence: HP sequence string
            run_exact: Whether to run exact enumeration (only for N <= 10)
            
        Returns:
            Dictionary with results for all methods
        """
        print(f"\nBenchmarking: {sequence} (N={len(sequence)})")
        
        seq_obj = load_hp_sequence(sequence)
        encoding = map_to_lattice(seq_obj, lattice_dim=2)
        
        result = {
            'sequence': sequence,
            'length': len(sequence),
            'n_qubits': encoding.n_qubits,
        }
        
        # Exact solution (if feasible)
        if run_exact and len(sequence) <= 10:
            print("  Running exact enumeration...")
            try:
                start = time.time()
                exact_result = exact_enumeration_fold(encoding, max_conformations=100000)
                exact_time = time.time() - start
                
                result['exact'] = {
                    'energy': float(exact_result.energy),
                    'time': exact_time,
                    'conformations_checked': exact_result.n_iterations
                }
                print(f"    Exact energy: {exact_result.energy:.4f} ({exact_time:.2f}s)")
            except Exception as e:
                print(f"    Exact failed: {e}")
        
        # Classical SA
        print("  Running Simulated Annealing...")
        try:
            start = time.time()
            sa_result = simulated_annealing_fold(encoding, max_iterations=10000, seed=42)
            sa_time = time.time() - start
            
            result['classical_sa'] = {
                'energy': float(sa_result.energy),
                'time': sa_time,
            }
            print(f"    SA energy: {sa_result.energy:.4f} ({sa_time:.2f}s)")
        except Exception as e:
            print(f"    SA failed: {e}")
            return result
        
        # VQE
        print("  Running VQE...")
        try:
            start = time.time()
            vqe_model = VQEFoldingModel(
                sequence=sequence,
                lattice_dim=2,
                ansatz_depth=3,
                optimizer='COBYLA'
            )
            vqe_result = vqe_model.run(maxiter=self.vqe_maxiter)
            vqe_time = time.time() - start
            
            result['vqe'] = {
                'energy': float(vqe_result.optimal_value),
                'time': vqe_time,
                'n_params': vqe_model.solver.n_params,
                'iterations': vqe_result.n_iterations,
            }
            
            # Compute gap
            gap = compute_energy_gap(vqe_result.optimal_value, sa_result.energy)
            result['vqe']['energy_gap_vs_sa'] = float(gap)
            
            print(f"    VQE energy: {vqe_result.optimal_value:.4f} ({vqe_time:.2f}s)")
            print(f"    Gap vs SA: {gap*100:.2f}%")
        except Exception as e:
            print(f"    VQE failed: {e}")
        
        # QAOA
        print("  Running QAOA...")
        try:
            start = time.time()
            qaoa_model = QAOAFoldingModel(
                sequence=sequence,
                p_layers=self.qaoa_p,
                lattice_dim=2,
                optimizer='COBYLA'
            )
            qaoa_result = qaoa_model.run(maxiter=self.vqe_maxiter)
            qaoa_time = time.time() - start
            
            result['qaoa'] = {
                'energy': float(qaoa_result.optimal_value),
                'time': qaoa_time,
                'n_params': qaoa_model.solver.n_params,
                'iterations': qaoa_result.n_iterations,
            }
            
            # Compute gap
            gap = compute_energy_gap(qaoa_result.optimal_value, sa_result.energy)
            result['qaoa']['energy_gap_vs_sa'] = float(gap)
            
            print(f"    QAOA energy: {qaoa_result.optimal_value:.4f} ({qaoa_time:.2f}s)")
            print(f"    Gap vs SA: {gap*100:.2f}%")
        except Exception as e:
            print(f"    QAOA failed: {e}")
        
        self.results.append(result)
        return result
    
    def analyze_advantage(self) -> Dict:
        """Analyze quantum advantage across all benchmarks.
        
        Returns:
            Summary statistics
        """
        if not self.results:
            return {}
        
        vqe_wins = 0
        qaoa_wins = 0
        classical_wins = 0
        
        vqe_gaps = []
        qaoa_gaps = []
        
        for result in self.results:
            if 'vqe' not in result or 'qaoa' not in result or 'classical_sa' not in result:
                continue
            
            vqe_e = result['vqe']['energy']
            qaoa_e = result['qaoa']['energy']
            sa_e = result['classical_sa']['energy']
            
            best_e = min(vqe_e, qaoa_e, sa_e)
            
            if abs(vqe_e - best_e) < 1e-3:
                vqe_wins += 1
            if abs(qaoa_e - best_e) < 1e-3:
                qaoa_wins += 1
            if abs(sa_e - best_e) < 1e-3:
                classical_wins += 1
            
            if 'energy_gap_vs_sa' in result['vqe']:
                vqe_gaps.append(result['vqe']['energy_gap_vs_sa'])
            if 'energy_gap_vs_sa' in result['qaoa']:
                qaoa_gaps.append(result['qaoa']['energy_gap_vs_sa'])
        
        return {
            'total_benchmarks': len(self.results),
            'vqe_wins': vqe_wins,
            'qaoa_wins': qaoa_wins,
            'classical_wins': classical_wins,
            'vqe_avg_gap': float(np.mean(vqe_gaps)) if vqe_gaps else None,
            'qaoa_avg_gap': float(np.mean(qaoa_gaps)) if qaoa_gaps else None,
            'vqe_best_gap': float(min(vqe_gaps)) if vqe_gaps else None,
            'qaoa_best_gap': float(min(qaoa_gaps)) if qaoa_gaps else None,
        }
    
    def save_results(self, output_file: str):
        """Save results to JSON file."""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'benchmarks': self.results,
            'summary': self.analyze_advantage()
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\n\nResults saved to: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Quantum advantage analysis')
    parser.add_argument('--sequences', type=str, help='File with benchmark sequences (one per line)')
    parser.add_argument('--output', type=str, default='results/quantum_advantage.json',
                        help='Output file')
    parser.add_argument('--vqe_maxiter', type=int, default=150, help='VQE max iterations')
    parser.add_argument('--qaoa_p', type=int, default=3, help='QAOA layers')
    parser.add_argument('--run_exact', action='store_true', help='Run exact enumeration for small sequences')
    
    args = parser.parse_args()
    
    # Default benchmark sequences
    if args.sequences:
        with open(args.sequences) as f:
            sequences = [line.strip() for line in f if line.strip()]
    else:
        # Standard benchmarks
        sequences = [
            'HPHPH',
            'HPHPPHHPHH',
            'HPHPHPHPHPHP',
            'HPHPPHHPPHHPPHPH',
        ]
    
    analyzer = QuantumAdvantageAnalyzer(
        vqe_maxiter=args.vqe_maxiter,
        qaoa_p=args.qaoa_p
    )
    
    print("=" * 60)
    print("Quantum Advantage Analysis")
    print("=" * 60)
    
    for seq in sequences:
        analyzer.benchmark_sequence(seq, run_exact=args.run_exact)
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    summary = analyzer.analyze_advantage()
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    # Save results
    analyzer.save_results(args.output)
