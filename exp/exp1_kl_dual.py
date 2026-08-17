"""
Exp 1: KL-Dual Robust Prior Evaluation for DRBC.
Validates that the inner KL-robust prior problem reduces to 1D optimisation
over the dual variable lambda.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from priors import FiniteSupportPrior
from kl_dual import optimize_dual, dual_objective, bayesian_value
from primal_validation import primal_robust_value, primal_dual_gap
from value_estimators import estimate_Z_merton_finite


# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------

SUPPORT = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
PROBS   = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
DELTAS  = [0.0, 0.01, 0.05, 0.10, 0.20, 0.50]
N_INNER = 2000
SEED    = 42

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')


# ---------------------------------------------------------------------------
# Step 1 — Build prior and compute Z values
# ---------------------------------------------------------------------------

def compute_Z_values(support, n_inner, seed):
    """Compute Z_pi(b_i) for each support point using Merton estimator."""
    rng = np.random.default_rng(seed)
    Z_vals = np.array([
        estimate_Z_merton_finite(b, n_inner=n_inner, rng=rng)
        for b in support
    ])
    return Z_vals


# ---------------------------------------------------------------------------
# Step 2 — Run experiment for all delta values
# ---------------------------------------------------------------------------

def run_experiment(Z_vals, probs, deltas):
    """Return list of result dicts for each delta."""
    bay_val = bayesian_value(Z_vals, probs)
    rows = []

    for delta in deltas:
        dual_res   = optimize_dual(Z_vals, probs, delta)
        primal_res = primal_robust_value(Z_vals, probs, delta)
        gap        = primal_dual_gap(primal_res["primal_value"],
                                     dual_res["robust_value"])

        rows.append({
            "delta":         delta,
            "lambda_star":   dual_res["lambda_star"],
            "dual_value":    dual_res["robust_value"],
            "primal_value":  primal_res["primal_value"],
            "gap":           gap,
            "bayesian_value": bay_val,
            # Store grid for plotting
            "_grid_lambdas": dual_res["grid_lambdas"],
            "_grid_values":  dual_res["grid_values"],
        })

    return rows


# ---------------------------------------------------------------------------
# Step 3 — Print table
# ---------------------------------------------------------------------------

def print_table(rows):
    header = (
        f"{'delta':>8s}  {'lambda*':>12s}  {'dual_val':>12s}  "
        f"{'primal_val':>12s}  {'gap':>10s}  {'bayes_val':>12s}"
    )
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for r in rows:
        print(
            f"{r['delta']:8.4f}  {r['lambda_star']:12.4f}  "
            f"{r['dual_value']:12.6f}  {r['primal_value']:12.6f}  "
            f"{r['gap']:10.2e}  {r['bayesian_value']:12.6f}"
        )
    print(sep)

    # Delta=0 check
    r0 = next(r for r in rows if r["delta"] == 0.0)
    print(
        f"\ndelta=0: dual_value={r0['dual_value']:.6f}, "
        f"bayesian_value={r0['bayesian_value']:.6f}, "
        f"diff={abs(r0['dual_value'] - r0['bayesian_value']):.2e}"
    )


# ---------------------------------------------------------------------------
# Step 4 — Plot
# ---------------------------------------------------------------------------

def make_plot(rows, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # (a) Dual objective vs lambda for a selection of deltas
    ax = axes[0]
    plot_deltas = [d for d in [0.01, 0.05, 0.10, 0.20, 0.50]
                   if any(r["delta"] == d for r in rows)]
    cmap = plt.get_cmap("viridis")
    colors = cmap(np.linspace(0.15, 0.85, len(plot_deltas)))

    for col, delta in zip(colors, plot_deltas):
        r = next(r for r in rows if r["delta"] == delta)
        lam_vals = r["_grid_lambdas"]
        obj_vals = r["_grid_values"]
        ax.semilogx(lam_vals, obj_vals, color=col, label=f"δ={delta}")
        ax.axvline(r["lambda_star"], color=col, linestyle="--", alpha=0.5)

    ax.set_xlabel("λ (dual variable)", fontsize=12)
    ax.set_ylabel("Dual objective f(λ)", fontsize=12)
    ax.set_title("(a) Dual Objective vs λ", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, which="both", linestyle=":", alpha=0.5)

    # (b) Robust value and Bayesian value vs delta
    ax = axes[1]
    deltas_arr   = np.array([r["delta"]       for r in rows])
    dual_arr     = np.array([r["dual_value"]   for r in rows])
    primal_arr   = np.array([r["primal_value"] for r in rows])
    bay_arr      = np.array([r["bayesian_value"] for r in rows])

    ax.plot(deltas_arr, dual_arr,   "o-", color="steelblue",  label="Dual robust value")
    ax.plot(deltas_arr, primal_arr, "s--", color="darkorange", label="Primal robust value")
    ax.axhline(bay_arr[0], color="grey", linestyle=":", label="Bayesian value")

    ax.set_xlabel("δ (KL radius)", fontsize=12)
    ax.set_ylabel("Value", fontsize=12)
    ax.set_title("(b) Robust vs Bayesian Value", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.5)

    fig.suptitle("Exp 1: KL-Dual Robust Prior Evaluation", fontsize=14, y=1.01)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved → {out_path}")


# ---------------------------------------------------------------------------
# Step 5 — Save CSV
# ---------------------------------------------------------------------------

def save_csv(rows, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fieldnames = ["delta", "lambda_star", "dual_value", "primal_value", "gap"]
    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r[k] for k in fieldnames})
    print(f"Results saved → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Exp 1: KL-Dual Robust Prior Evaluation")
    print("=" * 60)

    prior = FiniteSupportPrior(SUPPORT, PROBS)
    print(f"\nPrior support : {prior.support}")
    print(f"Prior probs   : {prior.probs}")

    print(f"\nComputing Z_pi(b_i) for each support point (n_inner={N_INNER}) ...")
    Z_vals = compute_Z_values(SUPPORT, N_INNER, SEED)
    print(f"Z_vals = {np.array2string(Z_vals, precision=4)}")

    print("\nRunning dual/primal optimisation for each delta ...")
    rows = run_experiment(Z_vals, PROBS, DELTAS)

    print("\n--- Results Table ---")
    print_table(rows)

    out_fig = os.path.join(RESULTS_DIR, "exp1_kl_dual.png")
    out_csv = os.path.join(RESULTS_DIR, "exp1_results.csv")
    make_plot(rows, out_fig)
    save_csv(rows, out_csv)

    print("\n--- Summary ---")
    print(f"  delta=0 dual ≈ Bayesian value: "
          f"{abs(rows[0]['dual_value'] - rows[0]['bayesian_value']):.2e} gap")
    print(f"  Max primal-dual gap across all deltas: "
          f"{max(r['gap'] for r in rows):.2e}")
    print("Exp 1 complete.")


if __name__ == "__main__":
    main()
