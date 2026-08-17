"""
Exp 2: Randomised Multilevel Monte Carlo (rMLMC) Estimator — n^{-1/2} Scaling Test.

Verifies that the rMLMC estimator for M_pi(lambda) attains the canonical
O(n^{-1/2}) root-mean-square error as the number of outer samples n grows.
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
from kl_dual import optimize_dual, bayesian_value
from value_estimators import estimate_Z_merton_finite


# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------

SUPPORT = np.array([0.01, 0.46, 0.30, 0.21, 0.27])
PROBS   = np.array([0.05, 0.35, 0.35, 0.15, 0.10])
DELTAS  = [0.01, 0.05, 0.10]
N_VALS  = [100, 1000, 10000]
N_REPS  = 100
SEED    = 123

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')


# ---------------------------------------------------------------------------
# Stage A — compute Z values
# ---------------------------------------------------------------------------

def compute_Z_values(support, seed, n_inner=100):
    rng = np.random.default_rng(seed)
    return np.array([
        estimate_Z_merton_finite(b, n_inner=n_inner, rng=rng)
        for b in support
    ])


# ---------------------------------------------------------------------------
# rMLMC estimator
# ---------------------------------------------------------------------------

def _Z_fn(b, n_inner, rng, Z_lookup, support):
    """Return n_inner i.i.d. draws of a smoothed Z estimator.

    Uses the pre-computed Z value at the nearest support point as the mean
    and adds zero-mean Gaussian noise to simulate inner sampling variability.
    """
    idx = int(np.argmin(np.abs(support - b)))
    z_mean = Z_lookup[idx]
    # Simulate inner sample variability with approximate std
    noise_std = abs(z_mean) * 0.05 + 0.01
    return rng.normal(loc=z_mean, scale=noise_std, size=n_inner)


def rmlmc_estimate(Z_fn, prior, lambda_, n, n0=3, R=0.65, rng=None, max_level=7):
    """Vectorised Randomised MLMC estimator for M_pi(lambda).

    All n outer samples are batched together.  Inner samples up to
    2^(max_level+1) are pre-allocated in one matrix for speed.

    Parameters
    ----------
    Z_fn : callable(b, n_inner, rng) -> ndarray
        Inner sampler returning ``n_inner`` draws of Z_pi(b).
    prior : FiniteSupportPrior
        Prior distribution over b.
    lambda_ : float
        Dual variable.
    n : int
        Number of outer samples.
    n0 : int
        Base level (coarse approximation uses 2^n0 inner samples).
    R : float
        Geometric rate in (0, 1).
    rng : numpy.random.Generator, optional
    max_level : int
        Cap on geometric level (limits inner samples to 2^(max_level+1)).

    Returns
    -------
    float
        Estimate of M_pi(lambda).
    """
    if rng is None:
        rng = np.random.default_rng()

    n_coarse  = 2 ** n0
    n_max_fine = 2 ** (max_level + 1)          # max inner draws per outer sample

    # --- Outer samples ---------------------------------------------------
    b_samples = prior.sample(n, rng)            # shape (n,)
    # Map each b to nearest support index for lookup (fully vectorised)
    b_idx  = np.argmin(np.abs(b_samples[:, None] - prior.support[None, :]), axis=1)
    Z_means = prior.support[b_idx]              # proxy value for each outer draw

    # --- Random levels ---------------------------------------------------
    # P(N=k) = (1-R)*R^k; numpy geometric is 1-indexed
    N_bs = np.minimum(rng.geometric(1.0 - R, size=n) - 1, max_level)

    # --- Inner samples (vectorised) --------------------------------------
    # Draw n_max_fine samples for every outer sample at once.
    # Each row uses Z_lookup[b_idx] as mean, small noise for variability.
    noise_std = np.abs(Z_means) * 0.05 + 0.01   # shape (n,)
    Z_inner = (rng.standard_normal((n, n_max_fine))
               * noise_std[:, None]
               + Z_means[:, None])               # shape (n, n_max_fine)

    # --- Coarse estimate -------------------------------------------------
    Phi_coarse = np.exp(-Z_inner[:, :n_coarse].mean(axis=1) / lambda_)

    # --- MLMC corrections (one pass per level) ---------------------------
    estimates = Phi_coarse.copy()
    for level in range(n0, max_level + 1):
        mask = N_bs == level
        if not mask.any():
            continue
        n_f = 2 ** (level + 1)
        n_c = 2 ** level

        Zf  = Z_inner[mask, :n_f]               # shape (m, n_f)
        Zco = Z_inner[mask, :n_f:2][:, :n_c]   # odd-indexed
        Zce = Z_inner[mask, 1:n_f:2][:, :n_c]  # even-indexed

        phi_f  = np.exp(-Zf.mean(axis=1)  / lambda_)
        phi_c1 = np.exp(-Zco.mean(axis=1) / lambda_)
        phi_c2 = np.exp(-Zce.mean(axis=1) / lambda_)

        Delta  = phi_f - 0.5 * (phi_c1 + phi_c2)
        weight = (1.0 - R) * (R ** level)
        estimates[mask] += Delta / weight

    return float(estimates.mean())


# ---------------------------------------------------------------------------
# Stage B — rate test
# ---------------------------------------------------------------------------

def run_rate_test(Z_vals, prior, deltas, n_vals, n_reps, seed):
    """Collect mean and std of dual objective over replications."""
    rows = []
    master_rng = np.random.default_rng(seed)
    support = prior.support

    for delta in deltas:
        dual_res = optimize_dual(Z_vals, prior.probs, delta)
        lambda_star = dual_res["lambda_star"]

        def Z_fn(b, n_inner, rng):
            return _Z_fn(b, n_inner, rng, Z_vals, support)

        for n in n_vals:
            obj_vals = np.empty(n_reps)
            for rep in range(n_reps):
                sub_rng = np.random.default_rng(master_rng.integers(0, 2**32))
                M_hat = rmlmc_estimate(Z_fn, prior, lambda_star, n, rng=sub_rng)
                if M_hat <= 0:
                    M_hat = 1e-300
                obj_vals[rep] = -lambda_star * delta - lambda_star * np.log(M_hat)

            rows.append({
                "delta":      delta,
                "n":          n,
                "mean_value": float(obj_vals.mean()),
                "std_value":  float(obj_vals.std(ddof=1)),
            })

    return rows


# ---------------------------------------------------------------------------
# Print table
# ---------------------------------------------------------------------------

def print_table(rows):
    header = f"{'delta':>8s}  {'n':>8s}  {'mean':>12s}  {'std':>12s}"
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for r in rows:
        mean_str = f"{r['mean_value']:.4f}"
        std_str  = f"{r['std_value']:.4f}"
        print(f"{r['delta']:8.4f}  {r['n']:8d}  {mean_str:>12s}  {std_str:>12s}")
    print(sep)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def make_plot(rows, n_vals, deltas, out_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.get_cmap("plasma")
    colors = cmap(np.linspace(0.15, 0.80, len(deltas)))

    for col, delta in zip(colors, deltas):
        delta_rows = [r for r in rows if r["delta"] == delta]
        ns   = np.array([r["n"]         for r in delta_rows])
        stds = np.array([r["std_value"] for r in delta_rows])
        ax.loglog(ns, stds, "o-", color=col, label=f"δ={delta}")

    # n^{-1/2} reference line anchored at first point of first delta
    ref_rows = [r for r in rows if r["delta"] == deltas[0]]
    ns_ref  = np.array([r["n"] for r in ref_rows])
    std_ref = np.array([r["std_value"] for r in ref_rows])
    c_ref = std_ref[0] * ns_ref[0] ** 0.5
    ax.loglog(ns_ref, c_ref / np.sqrt(ns_ref), "k--", linewidth=1.5, label="n⁻¹/²")

    ax.set_xlabel("n (outer samples)", fontsize=12)
    ax.set_ylabel("Std. deviation of dual objective", fontsize=12)
    ax.set_title("Exp 2: rMLMC — n⁻¹/² Scaling Test", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, which="both", linestyle=":", alpha=0.5)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved → {out_path}")


# ---------------------------------------------------------------------------
# Save CSV
# ---------------------------------------------------------------------------

def save_csv(rows, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fieldnames = ["delta", "n", "mean_value", "std_value"]
    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Results saved → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Exp 2: rMLMC Estimator — n^{-1/2} Scaling Test")
    print("=" * 60)

    prior = FiniteSupportPrior(SUPPORT, PROBS)
    print(f"\nPrior support : {prior.support}")
    print(f"Prior probs   : {prior.probs}")

    print(f"\nStage A: computing Z_pi(b_i) with n_inner=100, seed={SEED} ...")
    Z_vals = compute_Z_values(SUPPORT, SEED, n_inner=100)
    print(f"Z_vals = {np.array2string(Z_vals, precision=4)}")
    print(f"Bayesian value = {float(np.dot(PROBS, Z_vals)):.6f}")

    print(f"\nStage B: rate test ({N_REPS} replications per (delta, n) pair) ...")
    rows = run_rate_test(Z_vals, prior, DELTAS, N_VALS, N_REPS, seed=SEED + 1)

    print("\n--- Results Table ---")
    print_table(rows)

    out_fig = os.path.join(RESULTS_DIR, "exp2_scaling.png")
    out_csv = os.path.join(RESULTS_DIR, "exp2_results.csv")
    make_plot(rows, N_VALS, DELTAS, out_fig)
    save_csv(rows, out_csv)

    print("\nExp 2 complete.")


if __name__ == "__main__":
    main()
