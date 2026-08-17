"""
Exp 3: Synthetic LQ DRBC robustness under prior misspecification.

Compares oracle (true A), plug-in (estimated A), and DRBC-LQ (analytical
Riccati-based adversarial reweighting) over multiple independent runs.

Paper context:
- True prior: theta* ~ N(0, 0.5^2 I_p)
- Misspecified training prior: theta ~ N(0, 1.0^2 I_p)
- DRBC KL radii delta in {0.01, 0.05, 0.10}
- 20 independent runs (paper: 100), 256 eval rollouts.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lq_system import LQSystem, LQIdentifier
from riccati import solve_riccati_lq, compute_optimal_gain
from drbc_lq_policy import DRBCLQPolicy, simulate_lqr_policy, simulate_drbc_policy

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
N_RUNS       = 20         # paper uses 100; 20 for tractability
DELTA_VALUES = [0.01, 0.05, 0.10]
N_EVAL       = 256        # Monte Carlo eval rollouts per method per run
N_THETA      = 32         # outer prior samples for DRBC (paper: 32)
TRUE_STD     = 0.5        # true prior std
MISSPEC_STD  = 1.0        # misspecified training prior std
GLOBAL_SEED  = 0

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_exploration_gain(du: int, dx: int, rng: np.random.Generator) -> np.ndarray:
    """Random exploration feedback gain K_rand ~ N(0, 0.3^2)."""
    return rng.normal(0.0, 0.3, (du, dx))


def run_single_run(
    run_id: int,
    lq: LQSystem,
    identifier: LQIdentifier,
    rng_run: np.random.Generator,
) -> dict:
    """One independent run: identify theta, build oracle/plug-in/DRBC policies, evaluate."""
    result = {"run_id": run_id}

    # 1. Sample true theta* from N(0, 0.5^2 I_p)
    theta_star = rng_run.normal(0.0, TRUE_STD, lq.dx)
    A_true = lq.compute_A(theta_star)

    # 2. Exploration trajectory
    K_rand = make_exploration_gain(lq.du, lq.dx, rng_run)
    x0_id = rng_run.standard_normal(lq.dx)
    X_id, U_id = lq.simulate_trajectory(A_true, K_rand, x0_id, rng_run)

    # 3. GLS estimation
    theta_hat = identifier.gls_estimate(
        X_id, U_id, lq.A0, lq.basis_matrices, lq.G, lq.Sigma, lq.dt
    )

    # 4. Oracle Riccati (using theta_star) — evaluated under true A
    P_oracle = solve_riccati_lq(A_true, lq.G, lq.Q, lq.R, lq.Q_f, lq.T, lq.dt)
    K_oracle = compute_optimal_gain(P_oracle, lq.G, lq.R)
    J_oracle = simulate_lqr_policy(K_oracle, lq, A_true, N_EVAL, rng_run)

    # 5. Plug-in Riccati (using theta_hat) — evaluated under true A
    A_hat = lq.compute_A(theta_hat)
    P_plugin = solve_riccati_lq(A_hat, lq.G, lq.Q, lq.R, lq.Q_f, lq.T, lq.dt)
    K_plugin = compute_optimal_gain(P_plugin, lq.G, lq.R)
    J_plugin = simulate_lqr_policy(K_plugin, lq, A_true, N_EVAL, rng_run)

    result["oracle_cost"]  = J_oracle
    result["plugin_cost"]  = J_plugin
    result["oracle_utility"] = -J_oracle
    result["plugin_utility"] = -J_plugin
    result["plugin_gap"]     = J_plugin - J_oracle

    # 6. DRBC analytical policy for each delta (evaluated under true A)
    for delta in DELTA_VALUES:
        drbc_rng = np.random.default_rng(run_id * 1000 + int(delta * 1000))
        policy = DRBCLQPolicy(
            lq_system=lq,
            theta_hat=theta_hat,
            delta=delta,
            N_theta=N_THETA,
            misspec_std=MISSPEC_STD,
        )
        policy.train_policy(rng=drbc_rng)
        eval_rng = np.random.default_rng(run_id * 2000 + int(delta * 1000))
        J_drbc = simulate_drbc_policy(policy, lq, theta_star, n_eval=N_EVAL, rng=eval_rng)
        result[f"drbc_cost_d{delta}"]    = J_drbc
        result[f"drbc_utility_d{delta}"] = -J_drbc
        result[f"drbc_gap_d{delta}"]     = J_drbc - J_oracle

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 65)
    print("Exp 3: Synthetic LQ DRBC — Prior Misspecification Robustness")
    print("=" * 65)
    print(f"N_RUNS={N_RUNS}, N_EVAL={N_EVAL}, N_THETA={N_THETA}")

    master_rng = np.random.default_rng(GLOBAL_SEED)
    lq = LQSystem(dx=10, du=5, T=2.0, dt=0.02, sigma_scale=0.7, seed=12345)
    identifier = LQIdentifier()

    records = []
    for run_id in range(N_RUNS):
        print(f"\n  Run {run_id + 1}/{N_RUNS} …", flush=True)
        rng_run = np.random.default_rng(master_rng.integers(0, 2**32))
        rec = run_single_run(run_id, lq, identifier, rng_run)
        records.append(rec)
        print(f"    oracle_cost={rec['oracle_cost']:.3f}, plugin_cost={rec['plugin_cost']:.3f}")
        for delta in DELTA_VALUES:
            print(f"    DRBC(delta={delta}): cost={rec[f'drbc_cost_d{delta}']:.3f}")

    df = pd.DataFrame(records)

    # ── Report table ──────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("Results (utility = -cost, gap = cost_method - cost_oracle)")
    print("=" * 65)

    def fmt(arr: np.ndarray) -> str:
        return f"{arr.mean():.2f} ± {arr.std():.2f}"

    print(f"{'Method':<22} {'Utility':>14} {'Gap':>14}")
    print("-" * 52)
    print(f"{'Oracle':<22} {fmt(df['oracle_utility'].values):>14} {'—':>14}")
    print(f"{'Plug-in':<22} {fmt(df['plugin_utility'].values):>14} {fmt(df['plugin_gap'].values):>14}")
    for delta in DELTA_VALUES:
        tag = f"DRBC(delta={delta})"
        print(
            f"{tag:<22} "
            f"{fmt(df[f'drbc_utility_d{delta}'].values):>14} "
            f"{fmt(df[f'drbc_gap_d{delta}'].values):>14}"
        )

    # ── Save CSV ──────────────────────────────────────────────────────────
    csv_path = os.path.join(RESULTS_DIR, "exp3_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    # ── Summary row (for RESULTS.md) ─────────────────────────────────────
    summary_rows = []
    summary_rows.append({
        "method": "Oracle",
        "mean_utility": df["oracle_utility"].mean(),
        "std_utility": df["oracle_utility"].std(),
        "mean_gap": 0.0,
        "std_gap": 0.0,
    })
    summary_rows.append({
        "method": "Plug-in",
        "mean_utility": df["plugin_utility"].mean(),
        "std_utility": df["plugin_utility"].std(),
        "mean_gap": df["plugin_gap"].mean(),
        "std_gap": df["plugin_gap"].std(),
    })
    for delta in DELTA_VALUES:
        summary_rows.append({
            "method": f"DRBC(d={delta})",
            "mean_utility": df[f"drbc_utility_d{delta}"].mean(),
            "std_utility": df[f"drbc_utility_d{delta}"].std(),
            "mean_gap": df[f"drbc_gap_d{delta}"].mean(),
            "std_gap": df[f"drbc_gap_d{delta}"].std(),
        })
    summary_df = pd.DataFrame(summary_rows)

    # ── Plot ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    methods = ["Oracle", "Plug-in"] + [f"DRBC(d={d})" for d in DELTA_VALUES]
    means   = [r["mean_utility"] for r in summary_rows]
    stds    = [r["std_utility"]  for r in summary_rows]
    colors  = ["green"] + ["red"] + ["steelblue"] * len(DELTA_VALUES)

    ax = axes[0]
    ax.bar(methods, means, yerr=stds, color=colors, alpha=0.7, capsize=5)
    ax.set_title("Exp 3: Achieved Utility (mean ± std)")
    ax.set_ylabel("Utility (−cost)")
    ax.tick_params(axis="x", rotation=30)

    gaps   = [r["mean_gap"] for r in summary_rows]
    g_stds = [r["std_gap"]  for r in summary_rows]
    ax2 = axes[1]
    ax2.bar(methods[1:], gaps[1:], yerr=g_stds[1:], color=colors[1:], alpha=0.7, capsize=5)
    ax2.set_title("Exp 3: Gap to Oracle (mean ± std)")
    ax2.set_ylabel("Gap (cost_method − cost_oracle)")
    ax2.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    png_path = os.path.join(RESULTS_DIR, "exp3_utilities.png")
    plt.savefig(png_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to {png_path}")


if __name__ == "__main__":
    main()
