"""
Exp 6: Finite-Support DRBC Merton Learning.

Solves the distributionally robust Bayesian control (DRBC) problem for the
Merton portfolio with a finite-support prior.  The adversarial distribution q*
is found by gradient descent on the Lagrangian

    L(q) = V_tilde(q) + lambda_dual * KL(q || p)

where V_tilde(q) = sum_i q_i V_merton(b_i) is the analytically computed
Bayesian Merton value under distribution q.
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
from primal_validation import kl_divergence


# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------

B_SUPPORT   = np.array([0.01, 0.46, 0.30, 0.21, 0.27])
P_BASELINE  = np.array([0.05, 0.35, 0.35, 0.15, 0.10])
LAMBDA_DUALS = [10, 50, 100, 500]

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')


# ---------------------------------------------------------------------------
# Analytical Merton value
# ---------------------------------------------------------------------------

def merton_value(b, r=0.05, sigma=0.4, T=1.0, x0=1.0, alpha=0.5):
    """Closed-form Merton value for power utility U(x) = x^alpha / alpha.

    V_merton(b) = (x0^alpha / alpha) * exp(alpha * [(b-r)^2 / (2*sigma^2*(1-alpha)) + r] * T)

    Parameters
    ----------
    b : float or array-like
        Excess-return parameter.
    r : float
        Risk-free rate.
    sigma : float
        Risky-asset volatility.
    T : float
        Investment horizon.
    x0 : float
        Initial wealth.
    alpha : float
        Utility exponent (must be < 1 and != 0).

    Returns
    -------
    float or ndarray
    """
    b = np.asarray(b, dtype=float)
    exponent = alpha * (((b - r) ** 2) / (2.0 * sigma ** 2 * (1.0 - alpha)) + r) * T
    return (x0 ** alpha / alpha) * np.exp(exponent)


def bayesian_merton_value_finite(b_support, q, r=0.05, sigma=0.4,
                                  T=1.0, x0=1.0, alpha=0.5):
    """Compute V_tilde(q) = sum_i q_i * V_merton(b_i).

    Parameters
    ----------
    b_support : array-like of shape (K,)
        Support points.
    q : array-like of shape (K,)
        Distribution over support points.
    r, sigma, T, x0, alpha : float
        Merton model parameters.

    Returns
    -------
    float
    """
    b_support = np.asarray(b_support, dtype=float)
    q = np.asarray(q, dtype=float)
    V = merton_value(b_support, r=r, sigma=sigma, T=T, x0=x0, alpha=alpha)
    return float(np.dot(q, V))


# ---------------------------------------------------------------------------
# Simplex projection
# ---------------------------------------------------------------------------

def _project_simplex(v):
    """Project vector v onto the probability simplex.

    Uses the O(K log K) algorithm of Duchi et al. (2008).

    Parameters
    ----------
    v : ndarray of shape (K,)

    Returns
    -------
    ndarray of shape (K,)
    """
    K = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, K + 1) > (cssv - 1))[0][-1]
    theta = (cssv[rho] - 1.0) / (rho + 1.0)
    return np.maximum(v - theta, 0.0)


# ---------------------------------------------------------------------------
# q-optimiser
# ---------------------------------------------------------------------------

def q_optimizer(b_support, p_baseline, lambda_dual,
                r=0.05, sigma=0.4, T=1.0, x0=1.0, alpha=0.5,
                lr=1e-5, h=1e-6, max_iter=10000, tol=1e-5):
    """Minimise V_tilde(q) + lambda_dual * KL(q || p) over the simplex.

    Uses projected gradient descent with central finite-difference gradients.

    Parameters
    ----------
    b_support : array-like of shape (K,)
        Support points.
    p_baseline : array-like of shape (K,)
        Reference distribution.
    lambda_dual : float
        KL penalty weight (larger => stay closer to p, less robust).
    r, sigma, T, x0, alpha : float
        Merton model parameters.
    lr : float
        Gradient-descent learning rate.
    h : float
        Finite-difference step size.
    max_iter : int
        Maximum number of iterations.
    tol : float
        Convergence tolerance on ||q_new - q||_2^2.

    Returns
    -------
    dict with keys:
        ``q_star``           : ndarray — optimal distribution.
        ``objective_history``: list — objective values per iteration.
        ``n_iters``          : int  — number of iterations performed.
        ``converged``        : bool — True if tol was reached.
    """
    b_support  = np.asarray(b_support, dtype=float)
    p_baseline = np.asarray(p_baseline, dtype=float)
    K = len(b_support)
    EPS = 1e-10

    def objective(q):
        kl = kl_divergence(np.clip(q, EPS, None), p_baseline)
        return bayesian_merton_value_finite(b_support, q, r, sigma, T, x0, alpha) \
               + lambda_dual * kl

    q = p_baseline.copy()
    history = [objective(q)]
    converged = False

    for it in range(max_iter):
        # Central finite-difference gradient
        grad = np.zeros(K)
        for k in range(K):
            q_plus  = q.copy(); q_plus[k]  += h; q_plus  = np.clip(q_plus, EPS, None)
            q_minus = q.copy(); q_minus[k] -= h; q_minus = np.clip(q_minus, EPS, None)
            grad[k] = (objective(q_plus) - objective(q_minus)) / (2.0 * h)

        q_new = _project_simplex(q - lr * grad)
        q_new = np.clip(q_new, EPS, None)
        q_new /= q_new.sum()

        obj_new = objective(q_new)
        history.append(obj_new)

        delta_q = float(np.sum((q_new - q) ** 2))
        q = q_new

        if delta_q < tol:
            converged = True
            break

    return {
        "q_star":            q,
        "objective_history": history,
        "n_iters":           it + 1,
        "converged":         converged,
    }


# ---------------------------------------------------------------------------
# Print helpers
# ---------------------------------------------------------------------------

def print_result(lambda_dual, q_star, b_support, p_baseline, r, sigma, T, x0, alpha):
    V_val = bayesian_merton_value_finite(b_support, q_star, r, sigma, T, x0, alpha)
    kl    = kl_divergence(q_star, p_baseline)
    q_str = np.array2string(q_star, precision=4, suppress_small=True)
    print(f"  lambda_dual={lambda_dual:>6d}: q*={q_str}, V={V_val:.6f}, KL={kl:.6f}")


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def make_plot(results, b_support, p_baseline, out_path):
    n_lambdas = len(results)
    fig, axes = plt.subplots(1, n_lambdas + 1, figsize=(4 * (n_lambdas + 1), 4),
                             sharey=False)
    x = np.arange(len(b_support))
    width = 0.35
    labels = [f"b={b:.2f}" for b in b_support]

    # Baseline
    ax0 = axes[0]
    ax0.bar(x, p_baseline, color="grey", alpha=0.7, label="p_baseline")
    ax0.set_xticks(x)
    ax0.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax0.set_title("Baseline p", fontsize=11)
    ax0.set_ylabel("Probability", fontsize=10)
    ax0.set_ylim(0, None)

    cmap = plt.get_cmap("tab10")
    for idx, (lambda_dual, res) in enumerate(results):
        ax = axes[idx + 1]
        q_star = res["q_star"]
        ax.bar(x - width / 2, p_baseline, width, color="grey", alpha=0.5, label="p")
        ax.bar(x + width / 2, q_star,    width, color=cmap(idx), alpha=0.8, label="q*")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        V_val = bayesian_merton_value_finite(b_support, q_star)
        kl    = kl_divergence(q_star, p_baseline)
        ax.set_title(f"λ={lambda_dual}\nV={V_val:.3f}, KL={kl:.4f}", fontsize=10)
        ax.legend(fontsize=8)
        ax.set_ylim(0, None)

    fig.suptitle("Exp 6: Adversarial q* Distributions (Finite-Support DRBC)", fontsize=12)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved → {out_path}")


# ---------------------------------------------------------------------------
# Save CSV
# ---------------------------------------------------------------------------

def save_csv(results, b_support, p_baseline, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    K = len(b_support)
    fieldnames = (
        ["lambda_dual"]
        + [f"q_{k}" for k in range(K)]
        + ["V_tilde", "KL"]
    )
    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for lambda_dual, res in results:
            q_star = res["q_star"]
            V_val  = bayesian_merton_value_finite(b_support, q_star)
            kl     = kl_divergence(q_star, p_baseline)
            row    = {"lambda_dual": lambda_dual, "V_tilde": V_val, "KL": kl}
            row.update({f"q_{k}": q_star[k] for k in range(K)})
            writer.writerow(row)
    print(f"Results saved → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Exp 6: Finite-Support DRBC Merton Learning")
    print("=" * 60)

    prior = FiniteSupportPrior(B_SUPPORT, P_BASELINE)
    print(f"\nSupport : {prior.support}")
    print(f"Baseline: {prior.probs}")

    V_base = bayesian_merton_value_finite(B_SUPPORT, P_BASELINE)
    KL_self = kl_divergence(P_BASELINE, P_BASELINE)
    print(f"\nBaseline V_tilde(p): {V_base:.6f}  KL(p||p): {KL_self:.2e}")

    results = []
    print("\nOptimising q* for each lambda_dual ...")
    for lambda_dual in LAMBDA_DUALS:
        # Scale lr inversely with lambda_dual for gradient stability:
        # The KL Hessian eigenvalue is lambda/min(p), so lr < 2*min(p)/lambda.
        lr_safe = 0.04 / lambda_dual
        res = q_optimizer(
            B_SUPPORT, P_BASELINE, lambda_dual,
            lr=lr_safe, h=1e-6, max_iter=50000, tol=1e-10,
        )
        results.append((lambda_dual, res))
        print_result(lambda_dual, res["q_star"], B_SUPPORT, P_BASELINE,
                     0.05, 0.4, 1.0, 1.0, 0.5)
        status = "converged" if res["converged"] else f"stopped at {res['n_iters']} iters"
        print(f"    ({status})")

    # Verification: large lambda => q* ≈ p
    print("\nVerification: lambda_dual=5000 (should give q* ≈ p) ...")
    res_inf = q_optimizer(B_SUPPORT, P_BASELINE, 5000,
                          lr=0.04 / 5000, h=1e-6, max_iter=50000, tol=1e-10)
    q_inf = res_inf["q_star"]
    diff  = float(np.max(np.abs(q_inf - P_BASELINE)))
    print(f"  max|q* - p| = {diff:.4e}  (should be small)")

    out_fig = os.path.join(RESULTS_DIR, "exp6_q_distributions.png")
    out_csv = os.path.join(RESULTS_DIR, "exp6_results.csv")
    make_plot(results, B_SUPPORT, P_BASELINE, out_fig)
    save_csv(results, B_SUPPORT, P_BASELINE, out_csv)

    print("\nExp 6 complete.")


if __name__ == "__main__":
    main()
