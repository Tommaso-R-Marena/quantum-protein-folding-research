"""Scaling study: Performance vs sequence length.

Usage:
    python benchmarks/scaling_study.py --min_length 5 --max_length 15 --output results/scaling.json
"""

import argparse
import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, List

from quantum_protein_folding.models import VQEFoldingModel, QAOAFoldingModel
from quantum_protein_folding.data import load_hp_sequence
from quantum_protein_folding.classical import simulated_annealing_fold
from quantum_protein_folding.analysis import plot_scaling_analysis


def generate_test_sequences(min_length: int, max_length: int) -> Dict[int, str]:
    """Generate test HP sequences of varying lengths.
    
    Returns:
        Dict mapping length -> sequence
    """
    sequences = {}
    
    for length in range(min_length, max_length + 1):
        # Generate alternating HP pattern with some variation
        if length % 2 == 0:
            seq = 'HP' * (length // 2)
        else:
            seq = 'HP' * (length // 2) + 'H'
        
        # Add some randomness
        seq_list = list(seq)
        for i in range(length // 4):
            idx = np.random.randint(0, length)
            seq_list[idx] = 'P' if seq_list[idx] == 'H' else 'H'
        
        sequences[length] = ''.join(seq_list)
    
    return sequences


def benchmark_vqe(sequence: str, maxiter: int = 100) -> Dict:
    """Benchmark VQE on given sequence."""
    print(f"  VQE (N={len(sequence)})...", end=' ', flush=True)
    
    start_time = time.time()
    
    model = VQEFoldingModel(
        sequence=sequence,
        lattice_dim=2,
        ansatz_depth=3,
        optimizer='COBYLA',
        shots=1024
    )
    
    result = model.run(maxiter=maxiter)
    
    elapsed = time.time() - start_time
    
    print(f"{elapsed:.2f}s")
    
    return {
        'n_qubits': model.encoding.n_qubits,
        'n_params': model.solver.n_params,
        'energy': float(result.optimal_value),
        'iterations': result.n_iterations,
        'time': elapsed,
    }


def benchmark_qaoa(sequence: str, p_layers: int = 3, maxiter: int = 100) -> Dict:
    """Benchmark QAOA on given sequence."""
    print(f"  QAOA (N={len(sequence)})...", end=' ', flush=True)
    
    start_time = time.time()
    
    model = QAOAFoldingModel(
        sequence=sequence,
        p_layers=p_layers,
        lattice_dim=2,
        optimizer='COBYLA',
        shots=1024
    )
    
    result = model.run(maxiter=maxiter)
    
    elapsed = time.time() - start_time
    
    print(f"{elapsed:.2f}s")
    
    return {
        'n_qubits': model.encoding.n_qubits,
        'n_params': model.solver.n_params,
        'energy': float(result.optimal_value),
        'iterations': result.n_iterations,
        'time': elapsed,
    }


def benchmark_classical(sequence: str, encoding) -> Dict:
    """Benchmark classical simulated annealing."""
    print(f"  Classical SA (N={len(sequence)})...", end=' ', flush=True)
    
    start_time = time.time()
    
    result = simulated_annealing_fold(
        encoding,
        max_iterations=5000,
        seed=42
    )
    
    elapsed = time.time() - start_time
    
    print(f"{elapsed:.2f}s")
    
    return {
        'energy': float(result.energy),
        'iterations': result.n_iterations,
        'time': elapsed,
    }


def run_scaling_study(
    min_length: int = 5,
    max_length: int = 12,
    methods: List[str] = ['vqe', 'qaoa', 'classical'],
    output_file: str = None,
    seed: int = 42
):
    """Run complete scaling study.
    
    Args:
        min_length: Minimum sequence length
        max_length: Maximum sequence length
        methods: List of methods to benchmark
        output_file: Path to save results (JSON)
        seed: Random seed
    """
    np.random.seed(seed)
    
    print(f"Scaling Study: N = {min_length} to {max_length}")
    print(f"Methods: {methods}\n")
    
    # Generate sequences
    sequences = generate_test_sequences(min_length, max_length)
    
    results = {}
    
    for length, sequence in sequences.items():
        print(f"\nLength {length}: {sequence}")
        
        results[length] = {'sequence': sequence}
        
        # Create encoding for classical method
        if 'classical' in methods:
            from quantum_protein_folding.data.preprocess import map_to_lattice
            seq_obj = load_hp_sequence(sequence)
            encoding = map_to_lattice(seq_obj, lattice_dim=2)
        
        # VQE
        if 'vqe' in methods:
            try:
                results[length]['vqe'] = benchmark_vqe(sequence)
            except Exception as e:
                print(f"    VQE failed: {e}")
                results[length]['vqe'] = {'error': str(e)}
        
        # QAOA
        if 'qaoa' in methods:
            try:
                results[length]['qaoa'] = benchmark_qaoa(sequence)
            except Exception as e:
                print(f"    QAOA failed: {e}")
                results[length]['qaoa'] = {'error': str(e)}
        
        # Classical
        if 'classical' in methods:
            try:
                results[length]['classical'] = benchmark_classical(sequence, encoding)
            except Exception as e:
                print(f"    Classical failed: {e}")
                results[length]['classical'] = {'error': str(e)}
    
    # Save results
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n\nResults saved to: {output_path}")
    
    # Generate plots
    print("\nGenerating plots...")
    
    # Prepare data for plotting
    plot_data = {}
    for length in results:
        if isinstance(results[length].get('vqe'), dict) and 'n_qubits' in results[length]['vqe']:
            plot_data[length] = {
                'n_qubits': results[length]['vqe']['n_qubits'],
                'time': results[length]['vqe']['time'],
                'energy': results[length]['vqe']['energy'],
            }
    
    if plot_data:
        plot_scaling_analysis(
            plot_data,
            save_path=str(output_path).replace('.json', '_scaling.png') if output_file else None,
            show=True
        )
    
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run scaling study')
    parser.add_argument('--min_length', type=int, default=5, help='Minimum sequence length')
    parser.add_argument('--max_length', type=int, default=12, help='Maximum sequence length')
    parser.add_argument('--methods', nargs='+', default=['vqe', 'qaoa', 'classical'],
                        help='Methods to benchmark')
    parser.add_argument('--output', type=str, default='results/scaling_study.json',
                        help='Output file for results')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    run_scaling_study(
        min_length=args.min_length,
        max_length=args.max_length,
        methods=args.methods,
        output_file=args.output,
        seed=args.seed
    )
