import torch
from test import run_morbo_optimization, build_parser
from pathlib import Path
import json
import matplotlib.pyplot as plt

def run_comparison(args):
    """Run both HV and scalarization approaches"""
    
    results = {}
    
    # ============================================
    # 1. Hypervolume-based MORBO (Current approach)
    # ============================================
    print("="*80)
    print("RUNNING: HYPERVOLUME-BASED MORBO")
    print("="*80)
    
    args.outdir = Path(args.outdir) / "hypervolume"
    args_hv = args
    
    from morbo.trust_region import TurboHParams
    
    # Configure for noisy objectives
    tr_hparams_hv = TurboHParams(
        hypervolume=True,
        use_noisy_trbo=True,  # Enable noisy handling
        infer_reference_point=True,  # Auto-adjust reference
        batch_size=args.batch_size,
        n_trust_regions=args.n_trust_regions,
        raw_samples=2048,
        n_initial_points=max(20, args.n_trust_regions * args.batch_size * 2),
    )
    
    trbo_state_hv, hv_history_hv = run_morbo_optimization(args_hv)
    
    results['hypervolume'] = {
        'hv_history': hv_history_hv,
        'final_hv': hv_history_hv[-1] if hv_history_hv else 0.0,
        'pareto_size': trbo_state_hv.pareto_X.shape[0],
        'n_evals': trbo_state_hv.n_evals.item(),
    }
    
    # ============================================
    # 2. Scalarization-based MORBO
    # ============================================
    print("\n" + "="*80)
    print("RUNNING: SCALARIZATION-BASED MORBO")
    print("="*80)
    
    args.outdir = Path(args.outdir).parent / "scalarization"
    args_scal = args
    
    tr_hparams_scal = TurboHParams(
        hypervolume=False,  # Use scalarization
        use_noisy_trbo=True,
        fixed_scalarization=False,  # Random weights per TR
        batch_size=args.batch_size,
        n_trust_regions=args.n_trust_regions,
        raw_samples=2048,
        n_initial_points=max(20, args.n_trust_regions * args.batch_size * 2),
    )
    
    trbo_state_scal, hv_history_scal = run_morbo_optimization(args_scal)
    
    results['scalarization'] = {
        'hv_history': hv_history_scal,
        'final_hv': hv_history_scal[-1] if hv_history_scal else 0.0,
        'pareto_size': trbo_state_scal.pareto_X.shape[0],
        'n_evals': trbo_state_scal.n_evals.item(),
    }
    
    # ============================================
    # 3. Comparison & Plotting
    # ============================================
    compare_and_plot(results, args.outdir.parent)
    
    return results


def compare_and_plot(results, output_dir):
    """Compare HV vs Scalarization results"""
    
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)
    
    hv_final = results['hypervolume']['final_hv']
    scal_final = results['scalarization']['final_hv']
    
    improvement = ((hv_final - scal_final) / scal_final * 100) if scal_final > 0 else float('inf')
    
    print(f"\nHypervolume Approach:")
    print(f"  Final HV: {hv_final:.4f}")
    print(f"  Pareto Size: {results['hypervolume']['pareto_size']}")
    print(f"  Evaluations: {results['hypervolume']['n_evals']}")
    
    print(f"\nScalarization Approach:")
    print(f"  Final HV: {scal_final:.4f}")
    print(f"  Pareto Size: {results['scalarization']['pareto_size']}")
    print(f"  Evaluations: {results['scalarization']['n_evals']}")
    
    print(f"\nImprovement: {improvement:.2f}%")
    print(f"Winner: {'Hypervolume' if hv_final > scal_final else 'Scalarization'}")
    
    # Plot convergence
    plt.figure(figsize=(10, 6))
    plt.plot(results['hypervolume']['hv_history'], 'b-', linewidth=2, label='Hypervolume-based')
    plt.plot(results['scalarization']['hv_history'], 'r--', linewidth=2, label='Scalarization-based')
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Hypervolume', fontsize=12)
    plt.title('MORBO: Hypervolume vs Scalarization Comparison', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_path = output_dir / 'comparison_plot.png'
    plt.savefig(plot_path, dpi=300)
    print(f"\nPlot saved to: {plot_path}")
    
    # Save comparison results
    with open(output_dir / 'comparison_results.json', 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    
    results = run_comparison(args)
