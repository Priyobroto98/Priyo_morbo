"""
MORBO implementation for Digital Twin Multi-Objective Optimization
Adapted for: 5 design variables (resource allocations + MaxRC) and 5 KPIs
"""
from pathlib import Path
import argparse
import csv
import os
import time
import warnings
import torch
import json
from typing import List, Optional
from botorch.utils.sampling import draw_sobol_samples
from botorch.utils.transforms import unnormalize
from torch import Tensor
from morbo.run_one_replication import run_one_replication
from morbo.state import TRBOState, TRGenStatus
from morbo.trust_region import TurboHParams, HypervolumeTrustRegion
from morbo.gen import TS_select_batch_MORBO
from morbo.benchmark_function import BenchmarkFunction
import sim_utils  
from botorch.utils.multi_objective.box_decompositions.dominated import DominatedPartitioning

tkwargs = {"dtype": torch.double, "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")}

class DTObjectiveFunction:
    """Wrapper to make DT simulation compatible with MORBO framework"""
    
    def __init__(self, args, results_dir, output_dir, kpi_means, kpi_stds):
        self.args = args
        self.results_dir = results_dir
        self.output_dir = output_dir
        self.kpi_means = kpi_means
        self.kpi_stds = kpi_stds
        self.n_calls = 0
        
    def __call__(self, X: Tensor) -> Tensor:
        """
        Evaluate DT at design points X
        X: batch_size x 5 tensor (unnormalized)
        Returns: batch_size x 5 tensor of objectives
        """
        batch_size = X.shape[0]
        obj_fn_values = []
        inputs = []
        total_res_list = []
        
        for i in range(batch_size):
            chute_east = torch.ceil(X[i, 0]).item()
            chute_west = torch.ceil(X[i, 1]).item()
            infeed_north = torch.ceil(X[i, 2]).item()
            infeed_south = torch.ceil(X[i, 3]).item()
            max_rc = torch.ceil(X[i, 4]).item()
            
            total_resources = chute_east + chute_west + infeed_north + infeed_south
            total_res_list.append(total_resources)
            
            new_input = {
                "chute_east_zone_cluster_res": chute_east,
                "chute_west_zone_cluster_res": chute_west,
                "infeed_north_zone_cluster_res": infeed_north,
                "infeed_south_zone_cluster_res": infeed_south,
                "max_rc": max_rc,
            }
            inputs.append(new_input)
        
        # Run simulation
        obj_fn_values = sim_utils.run_simulation(
            args=self.args,
            results_dir=self.results_dir,
            output_dir=self.output_dir,
            batch_size=batch_size,
            DT_SIMULATIONS=self.n_calls,
            total_res_list=total_res_list,
            inputs=inputs,
        )
        
        self.n_calls += batch_size
        Y = torch.tensor(obj_fn_values, **tkwargs)
        
        # Normalize objectives for better GP modeling
        Y_normalized = (Y - self.kpi_means) / self.kpi_stds
        
        return Y_normalized

def initialize_morbo_state(
    dim: int,
    bounds: Tensor,
    max_evals: int,
    n_trust_regions: int,
    batch_size: int,
    reference_point: List[float],
) -> TRBOState:
    
    base_min = n_trust_regions * batch_size
    stricter_min = n_trust_regions * (batch_size + 1)
    min_tr_size_target = max(2 * base_min, stricter_min)
    
    # Calculate initial points
    n_initial_points_raw = min_tr_size_target + max(1, batch_size)
    
    # SAFEGUARD: Ensure n_initial_points leaves room for optimization
    max_allowed_initial = max_evals // 2  
    n_initial_points = min(n_initial_points_raw, max_allowed_initial)
    
    # Ensure minimum viable number
    n_initial_points = max(n_initial_points, n_trust_regions + batch_size)
    
    print(
        f"[initialize_morbo_state] "
        f"calculated n_initial={n_initial_points_raw}, "
        f"capped at {n_initial_points} (max_evals={max_evals})"
    )
    
    # Update min_tr_size accordingly
    min_tr_size_target = min(min_tr_size_target, n_initial_points - batch_size)
    
    tr_hparams = TurboHParams(
        n_trust_regions=n_trust_regions,
        batch_size=batch_size,
        n_initial_points=n_initial_points,
        min_tr_size=min_tr_size_target,
        length_min=0.01,
        length_max=1.6,
        success_streak=3,
        failure_streak=max(1, batch_size),
        hypervolume=True,
        use_ard=True,
        max_cholesky_size=2000,
        raw_samples=2048,
        qmc=True, #Better space coverage
        track_history=True,
        sample_subset_d=True,
        winsor_pct=5.0, # Cuts extreme values to stabilize GP.
        max_reference_point=reference_point,
        tabu_tenure=5,
        verbose=True,

    )

    # Create the state
    trbo_state = TRBOState(
        dim=dim,
        num_outputs=5,           # 5 KPIs
        num_objectives=5,
        bounds=bounds,
        max_evals=max_evals,
        tr_hparams=tr_hparams,
        objective=None,        
        constraints=None,
    )

    return trbo_state

def run_morbo_optimization(args):
    """Main MORBO optimization loop for DT"""

    NUM_DESIGN_VARS = 5

    # -----------------------------
    # Setup directories (robust)
    # -----------------------------
    # Coerce to Path (handles both str and Path inputs)
    outdir = args.outdir if isinstance(args.outdir, Path) else Path(args.outdir)

    # Choose a base directory depending on environment
    if args.env == "welkin":
        base_results = Path("/data/ML/Priyobroto/Digital-twin/Koge/Results/MORBO")
    else:
        base_results = Path("./Results/MORBO")

    # If user provided an absolute outdir, use it as-is; otherwise put it under the base.
    results_dir = outdir if outdir.is_absolute() else (base_results / outdir)
    output_dir = results_dir / "simulation_outputs"

    # Create directories
    results_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # Problem bounds
    # -----------------------------
    problem_bounds = torch.ones(2, NUM_DESIGN_VARS, **tkwargs)
    problem_bounds[1, 0:2] = 120  # chute resources
    problem_bounds[1, 2:4] = 40   # infeed resources
    problem_bounds[1, 4] = 50     # MaxRC
    print(f"DEBUG: problem_bounds shape={problem_bounds.shape},values=\n{problem_bounds}")
    # -----------------------------
    # KPI normalization statistics
    # -----------------------------
    kpi_means = torch.tensor([0.58, -40, -1.3, -0.09, -125], **tkwargs)
    kpi_stds  = torch.tensor([0.17,  13,  1.36,  0.09,   50], **tkwargs)

    # Reference point (normalized space)
    reference_point_raw = torch.tensor([0.6, -70, -1, -0.2, -300], **tkwargs)
    reference_point_normalized = (reference_point_raw - kpi_means) / kpi_stds

    # -----------------------------
    # Initialize objective function
    # -----------------------------
    dt_objective = DTObjectiveFunction(
        args,
        str(results_dir),
        str(output_dir),
        kpi_means,
        kpi_stds,
    )

    # -----------------------------
    # Initialize MORBO state
    # -----------------------------
    trbo_state = initialize_morbo_state(
        dim=NUM_DESIGN_VARS,
        bounds=problem_bounds,
        max_evals=args.max_evals,
        n_trust_regions=args.n_trust_regions,
        batch_size=args.batch_size,
        reference_point=reference_point_normalized.tolist(),
    )

    # -----------------------------
    # Generate initial Sobol samples
    # -----------------------------
    print(f"Generating {trbo_state.tr_hparams.n_initial_points} initial points...")
    X_init = draw_sobol_samples(
        bounds=problem_bounds,
        n=trbo_state.tr_hparams.n_initial_points,
        q=1,
    ).squeeze(1).to(**tkwargs)

    # Evaluate initial points
    Y_init = dt_objective(X_init)

    # Update state with initial data
    trbo_state.X_history = X_init
    trbo_state.Y_history = Y_init
    trbo_state.n_evals[...] = X_init.shape[0]


    # ============================================================================
    # Initialize ref_point and empty Pareto sets
    # The state.update() method only sets ref_point when Pareto points exist
    # We need it initialized even when no initial points exceed reference
    # ============================================================================
    if trbo_state.ref_point is None:
        trbo_state.ref_point = reference_point_normalized.clone()
        print(f"Initialized reference point: {trbo_state.ref_point.tolist()}")

    # Initialize empty Pareto sets if None
    if trbo_state.pareto_Y_better_than_ref is None:
        trbo_state.pareto_Y_better_than_ref = torch.empty(0, 5, **tkwargs)
    if trbo_state.pareto_X_better_than_ref is None:
        trbo_state.pareto_X_better_than_ref = torch.empty(0, NUM_DESIGN_VARS, **tkwargs)
    # ============================================================================

    # Initialize trust regions
    print(f"Initializing {args.n_trust_regions} trust regions...")
    for tr_idx in range(args.n_trust_regions):
        trbo_state.initialize_standard(
            tr_idx=tr_idx,
            X_init=X_init,
            Y_init=Y_init,
        )

    # Compute initial hypervolume
    if trbo_state.pareto_Y_better_than_ref is not None and trbo_state.pareto_Y_better_than_ref.shape[0] > 0:
        bd = DominatedPartitioning(
            ref_point=reference_point_normalized,
            Y=trbo_state.pareto_Y_better_than_ref,
        )
        initial_hv = bd.compute_hypervolume().item()
        print(f"Initial hypervolume: {initial_hv:.4f}")
    else:
        initial_hv = 0.0
        print("Initial hypervolume: 0.0 (no points better than reference)")

    # Save hypervolume history
    hv_history = [initial_hv]



    # -----------------------------
    # Main optimization loop
    # -----------------------------
    iteration = 0
    while trbo_state.n_evals < args.max_evals:
        iteration += 1
        print(f"\n{'='*80}")
        print(f"MORBO Iteration {iteration} | Evaluations: {trbo_state.n_evals}/{args.max_evals}")
        print(f"{'='*80}")

        # Generate candidates using Thompson Sampling
        print("Generating candidates via Thompson Sampling...")
        t_start = time.time()
        candidate_output = TS_select_batch_MORBO(trbo_state)
        X_cand = candidate_output.X_cand
        tr_indices = candidate_output.tr_indices
        t_gen = time.time() - t_start
        print(f"Candidate generation time: {t_gen:.2f}s")

        # Evaluate candidates
        print("Evaluating candidates...")
        t_eval_start = time.time()
        Y_cand = dt_objective(X_cand)
        t_eval = time.time() - t_eval_start
        print(f"Evaluation time: {t_eval:.2f}s")

        # Update state - THIS ALREADY UPDATES PARETO FRONTIER
        trbo_state.update(
            X=X_cand,
            Y=Y_cand,
            new_ind=tr_indices,
        )

        # Update trust regions
        should_restart = trbo_state.update_trust_regions_and_log(
            X_cand=X_cand,
            Y_cand=Y_cand,
            tr_indices=tr_indices,
            batch_size=args.batch_size,
            verbose=True,
        )

        # Restart trust regions if needed
        for tr_idx, restart in enumerate(should_restart):
            if restart:
                print(f"Restarting trust region {tr_idx}")
                trbo_state.initialize_standard(
                    tr_idx=tr_idx,
                    restart=True,
                )

        # Compute hypervolume (Pareto frontier already updated by trbo_state.update())
        if trbo_state.pareto_Y_better_than_ref is not None and trbo_state.pareto_Y_better_than_ref.shape[0] > 0:
            bd = DominatedPartitioning(
                ref_point=reference_point_normalized,
                Y=trbo_state.pareto_Y_better_than_ref,
            )
            current_hv = bd.compute_hypervolume().item()
            hv_history.append(current_hv)
            
            print(f"Current hypervolume: {current_hv:.4f} (Δ={current_hv - hv_history[-2]:.4f})")
            print(f"Pareto frontier size: {trbo_state.pareto_X.shape[0]}")
        else:
            print("No Pareto points better than reference point yet")
            hv_history.append(hv_history[-1])  # Keep previous HV


        # Save intermediate results every 5 iterations
        if iteration % 5 == 0:
            save_results(trbo_state, str(results_dir), hv_history, kpi_means, kpi_stds)

    # Final save
    print("\nOptimization complete!")
    save_results(trbo_state, str(results_dir), hv_history, kpi_means, kpi_stds)

    return trbo_state, hv_history




def save_results(trbo_state, results_dir, hv_history, kpi_means, kpi_stds):
    """Save optimization results"""
    
    # Denormalize objectives
    Y_denormalized = trbo_state.Y_history * kpi_stds + kpi_means
    
    # Save tensors
    torch.save(trbo_state.X_history, f"{results_dir}/X_history.pt")
    torch.save(Y_denormalized, f"{results_dir}/Y_history.pt")
    torch.save(trbo_state.pareto_X, f"{results_dir}/pareto_X.pt")
    
    # Save hypervolume history
    with open(f"{results_dir}/hypervolume_history.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["Iteration", "Hypervolume"])
        for i, hv in enumerate(hv_history):
            writer.writerow([i, hv])
    
    print(f"Results saved to {results_dir}")



import argparse
from pathlib import Path

# --- top of test.py or wherever you build the parser ---
from pathlib import Path
import argparse

def build_parser():
    parser = argparse.ArgumentParser(description="MORBO optimization for DT")

    # General optimization / run configuration
    parser.add_argument("--env", type=str, choices=["local", "welkin"], default="local",
                        help="Runtime environment")
    parser.add_argument("--max_evals", type=int, default=200,
                        help="Total evaluation budget")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size per iteration")
    parser.add_argument("--n_trust_regions", type=int, default=5,
                        help="Number of trust regions")
    parser.add_argument("--outdir", type=Path, default=Path("morbo_dt_optimization"),
                        help="Output directory; absolute path is used as-is; relative is placed under ./Results/MORBO")

    # Optional short-run (5-tick) mode in your Excel workbook edits
    parser.add_argument("--testing", action="store_true",
                        help="If set, run short 5-tick DT simulations (write Config!B5=5)")

    # DT simulation I/O
    # Flexible: allow multiple input templates; if fewer than batch_size, we replicate
    parser.add_argument(
        "--input-files",
        type=Path,
        nargs="+",
        help="One or more Excel input templates (one per batch). If fewer than batch_size, the first is replicated."
    )
    # Backward-compatible single template (optional)
    parser.add_argument("--input-file1", type=Path,
                        help="(Deprecated) Single input Excel template (used if --input-files not supplied)")

    # Additional DT resources (used by dt_evaluation)
    parser.add_argument("--output-file", type=Path, required=True,
                        help="Path to DT output workbook (e.g., Output.xlsx)")
    parser.add_argument("--parcel-mapping", type=Path, required=True,
                        help="Path to parcel_chute_cage_mapping.xlsx")
    parser.add_argument("--parcel-data", type=Path, required=True,
                        help="Path to s1-Parcel_data.csv")

    return parser

def main():
    parser = build_parser()
    args = parser.parse_args()

    # Normalize input templates for downstream code:
    if getattr(args, "input_files", None) is None:
        # Fallback if only --input-file1 provided
        if getattr(args, "input_file1", None) is not None:
            args.input_files = [args.input_file1]
        else:
            # You can choose to error or allow later code to handle it
            raise SystemExit("No input workbook provided. Use --input-files or --input-file1.")

    run_morbo_optimization(args)

if __name__ == "__main__":
    # Windows multiprocessing requires the main-guard
    main()
