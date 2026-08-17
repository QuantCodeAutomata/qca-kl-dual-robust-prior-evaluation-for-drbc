"""
Exp 4: Synthetic Merton DRBC vs DRC radius ablation.

Compares Bayesian, DRBC, and DRC policies in the Merton portfolio problem
as ambiguity radius delta varies. The true drift follows B_t = (B0/2)*(1+cos(kappa*t)),
creating model misspecification relative to the policy prior.

Parameters (documented as assumptions where not fully specified in text):
- T=1.0, sigma=0.4, r=0.05, x0=1.0
- Power utility alpha=0.5: U(x) = x^0.5 / 0.5
- B0=0.3, kappa=pi (one full cosine period)
- Finite-support prior: b_support=[0.05, 0.10, 0.15, 0.20, 0.25], p=[0.1,0.2,0.4,0.2,0.1]
- Monte Carlo: 500 trajectories, K=100 steps
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import logsumexp

# ─────────────────────────────────────────────────────────────────────────────
# Environment parameters (assumptions documented)
# ─────────────────────────────────────────────────────────────────────────────
T      = 1.0
SIGMA  = 0.4
R      = 0.05
X0     = 1.0
ALPHA  = 0.5   # power utility: U(x) = x^alpha / alpha
B0     = 0.3   # assumption: true drift amplitude
KAPPA  = np.pi # assumption: one cosine period over [0, T]
K      = 100   # time steps
DT     = T / K

# Finite-support prior (baseline)
B_SUPPORT = np.array([0.05, 0.10, 0.15, 0.20, 0.25])
P_BASE    = np.array([0.10, 0.20, 0.40, 0.20, 0.10])

# Delta grid
DELTA_GRID = [0.0, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30]
N_TRAJ     = 500  # Monte Carlo trajectories for evaluation
SEED       = 42

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# True drift path
# ─────────────────────────────────────────────────────────────────────────────

def true_drift_path(times: np.ndarray) -> np.ndarray:
    """B_t = (B0/2)*(1 + cos(kappa*t)) — time-varying true drift."""
    return (B0 / 2.0) * (1.0 + np.cos(KAPPA * times))


# ─────────────────────────────────────────────────────────────────────────────
# Merton optimal fraction (static b)
# ─────────────────────────────────────────────────────────────────────────────

def merton_fraction(b: float, r: float = R, sigma: float = SIGMA, alpha: float = ALPHA) -> float:
    """Optimal Merton fraction pi* = (b-r)/(sigma^2*(1-alpha))."""
    denom = sigma**2 * (1.0 - alpha)
    return (b - r) / denom


# ─────────────────────────────────────────────────────────────────────────────
# Simulation of wealth under a *constant* Merton fraction
# ─────────────────────────────────────────────────────────────────────────────

def simulate_wealth(
    pi_val: float,
    true_B_path: np.ndarray,
    times: np.ndarray,
    rng: np.random.Generator,
    n_traj: int = N_TRAJ,
) -> np.ndarray:
    """Simulate n_traj wealth trajectories under constant fraction pi_val.

    dX_t = r*X_t*dt + pi*X_t*(B_t-r)*dt + pi*X_t*sigma*dW_t
    => multiplicative, no bankruptcy with X0=1 and moderate pi.

    Returns
    -------
    X_T : (n_traj,) terminal wealth.
    """
    X = np.ones(n_traj) * X0
    for k in range(K):
        B_t = true_B_path[k]
        dW  = rng.standard_normal(n_traj) * np.sqrt(DT)
        drift_term = (R + pi_val * (B_t - R)) * DT
        diff_term  = pi_val * SIGMA * dW
        X = X * np.exp(drift_term - 0.5 * pi_val**2 * SIGMA**2 * DT + diff_term)
    return X


def expected_utility(X_T: np.ndarray) -> float:
    """E[X_T^alpha / alpha]."""
    X_T_clipped = np.clip(X_T, 1e-10, None)
    return float(np.mean(X_T_clipped**ALPHA / ALPHA))


# ─────────────────────────────────────────────────────────────────────────────
# Bayesian policy (prior-weighted fraction, delta-independent)
# ─────────────────────────────────────────────────────────────────────────────

def bayesian_fraction(b_support: np.ndarray, probs: np.ndarray) -> float:
    """Bayesian Merton fraction = sum_i p_i * merton_fraction(b_i)."""
    return float(np.dot(probs, [merton_fraction(b) for b in b_support]))


# ─────────────────────────────────────────────────────────────────────────────
# DRBC policy: optimize adversarial q* on simplex, then use weighted fraction
# ─────────────────────────────────────────────────────────────────────────────

def compute_Z_merton(b_support: np.ndarray, r: float = R, sigma: float = SIGMA,
                     alpha: float = ALPHA, T: float = T) -> np.ndarray:
    """Analytical Merton value Z_i = V_Merton(b_i) for power utility.

    V_Merton(b) = (x0^alpha/alpha) * exp(alpha*(r + (b-r)^2/(2*sigma^2*(1-alpha)))*T)
    """
    Z = np.array([
        (X0**ALPHA / ALPHA) * np.exp(
            ALPHA * (r + (b - r)**2 / (2 * sigma**2 * (1 - alpha))) * T
        )
        for b in b_support
    ])
    return Z


def drbc_q_star(
    Z_vals: np.ndarray,
    p_base: np.ndarray,
    lambda_dual: float,
    n_iter: int = 500,
    lr: float = 1e-3,
) -> np.ndarray:
    """Find adversarial q* via projected gradient descent (minimise sum q_i Z_i + lambda*KL).

    Uses logsumexp-softmax trick for analytical solution:
    q*_i ∝ p_i * exp(-Z_i / lambda_dual)
    """
    # Analytical solution for the KL-penalised problem
    log_q = np.log(p_base + 1e-300) - Z_vals / lambda_dual
    log_q -= logsumexp(log_q)  # normalise
    return np.exp(log_q)


def drbc_fraction(
    b_support: np.ndarray,
    p_base: np.ndarray,
    delta: float,
    lambda_C: float = 1.0,
) -> float:
    """DRBC Merton fraction: use q*(lambda_dual) with lambda_dual = lambda_C/sqrt(delta)."""
    if delta <= 0.0:
        return bayesian_fraction(b_support, p_base)
    Z_vals = compute_Z_merton(b_support)
    lambda_dual = lambda_C / np.sqrt(delta)
    q_star = drbc_q_star(Z_vals, p_base, lambda_dual)
    return float(np.dot(q_star, [merton_fraction(b) for b in b_support]))


# ─────────────────────────────────────────────────────────────────────────────
# DRC policy: robustify full dynamics (conservative formula)
# ─────────────────────────────────────────────────────────────────────────────

def drc_fraction(b_support: np.ndarray, p_base: np.ndarray, delta: float) -> float:
    """DRC Merton fraction.

    Conservative formula: pi_DRC = (mu_nominal - r - penalty) / (sigma^2*(1-alpha))
    where penalty = delta^0.5 * sigma (robust adjustment to nominal drift).
    Clipped to [0, pi_max].
    """
    mu_nominal = float(np.dot(p_base, b_support))
    penalty = np.sqrt(delta) * SIGMA
    pi_drc = (mu_nominal - R - penalty) / (SIGMA**2 * (1.0 - ALPHA))
    return float(np.clip(pi_drc, 0.0, 20.0))


# ─────────────────────────────────────────────────────────────────────────────
# Oracle (numerical): grid search over constant fractions
# ─────────────────────────────────────────────────────────────────────────────

def compute_oracle_utility(times: np.ndarray, true_B: np.ndarray, rng: np.random.Generator) -> tuple:
    """Grid search for best constant fraction under true B_t."""
    pi_grid = np.linspace(-2.0, 10.0, 61)
    best_pi, best_util = pi_grid[0], -np.inf
    for pi in pi_grid:
        X_T = simulate_wealth(pi, true_B, times, rng, n_traj=200)
        u = expected_utility(X_T)
        if u > best_util:
            best_util = u
            best_pi = pi
    return best_pi, best_util


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("Exp 4: Merton DRBC vs DRC — Ambiguity Radius Ablation")
    print("=" * 60)
    print(f"B0={B0}, kappa=pi, sigma={SIGMA}, r={R}, alpha={ALPHA}, T={T}")
    print(f"Prior: b_support={B_SUPPORT}, p={P_BASE}")

    rng = np.random.default_rng(SEED)
    times = np.linspace(0.0, T, K + 1)
    true_B = true_drift_path(times)

    # Oracle
    print("\nComputing oracle (grid search over true drift path) …")
    pi_oracle, util_oracle = compute_oracle_utility(times, true_B, np.random.default_rng(SEED + 1))
    print(f"  Oracle pi*={pi_oracle:.4f}, utility={util_oracle:.6f}")

    records = []
    for delta in DELTA_GRID:
        print(f"\n  delta={delta:.3f}")

        # 1. Bayesian
        pi_bayes = bayesian_fraction(B_SUPPORT, P_BASE)
        X_T_bayes = simulate_wealth(pi_bayes, true_B, times, rng, N_TRAJ)
        u_bayes = expected_utility(X_T_bayes)
        gap_bayes = util_oracle - u_bayes

        # 2. DRBC
        pi_drbc = drbc_fraction(B_SUPPORT, P_BASE, delta)
        X_T_drbc = simulate_wealth(pi_drbc, true_B, times, rng, N_TRAJ)
        u_drbc = expected_utility(X_T_drbc)
        gap_drbc = util_oracle - u_drbc

        # 3. DRC
        pi_drc = drc_fraction(B_SUPPORT, P_BASE, delta)
        X_T_drc = simulate_wealth(pi_drc, true_B, times, rng, N_TRAJ)
        u_drc = expected_utility(X_T_drc)
        gap_drc = util_oracle - u_drc

        print(
            f"    Bayes pi={pi_bayes:.3f} util={u_bayes:.5f} gap={gap_bayes:.5f} | "
            f"DRBC pi={pi_drbc:.3f} util={u_drbc:.5f} gap={gap_drbc:.5f} | "
            f"DRC pi={pi_drc:.3f} util={u_drc:.5f} gap={gap_drc:.5f}"
        )
        records.append(
            {"delta": delta, "method": "Bayesian",  "pi": pi_bayes,  "utility": u_bayes,  "gap": gap_bayes}
        )
        records.append(
            {"delta": delta, "method": "DRBC",      "pi": pi_drbc,   "utility": u_drbc,   "gap": gap_drbc}
        )
        records.append(
            {"delta": delta, "method": "DRC",        "pi": pi_drc,    "utility": u_drc,    "gap": gap_drc}
        )

    df = pd.DataFrame(records)

    # ── Save CSV ──────────────────────────────────────────────────────────
    csv_path = os.path.join(RESULTS_DIR, "exp4_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    # ── Plot ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for method, color, ls in [
        ("Bayesian", "tab:orange", "--"),
        ("DRBC",     "tab:blue",   "-"),
        ("DRC",      "tab:red",    "-."),
    ]:
        sub = df[df["method"] == method].sort_values("delta")
        axes[0].plot(sub["delta"], sub["utility"], label=method, color=color, ls=ls, marker="o")
        axes[1].plot(sub["delta"], sub["gap"],     label=method, color=color, ls=ls, marker="o")

    axes[0].axhline(util_oracle, color="green", ls=":", label="Oracle")
    axes[0].set_title("Exp 4: Expected Utility vs Delta")
    axes[0].set_xlabel("KL radius delta")
    axes[0].set_ylabel("Expected Utility E[U(X_T)]")
    axes[0].legend()

    axes[1].set_title("Exp 4: Utility Gap vs Delta")
    axes[1].set_xlabel("KL radius delta")
    axes[1].set_ylabel("Utility Gap (Oracle − Method)")
    axes[1].legend()

    plt.tight_layout()
    png_path = os.path.join(RESULTS_DIR, "exp4_merton_ablation.png")
    plt.savefig(png_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to {png_path}")

    # ── Qualitative check ─────────────────────────────────────────────────
    print("\n── Qualitative check (DRBC gap < DRC gap?) ──")
    for delta in DELTA_GRID[1:]:  # skip delta=0
        sub = df[df["delta"] == delta]
        g_drbc = sub[sub["method"] == "DRBC"]["gap"].values[0]
        g_drc  = sub[sub["method"] == "DRC"]["gap"].values[0]
        status = "✓" if g_drbc < g_drc else "✗"
        print(f"  delta={delta:.2f}: DRBC gap={g_drbc:.5f}, DRC gap={g_drc:.5f}  {status}")


if __name__ == "__main__":
    main()
