"""
kl_dual.py — KL-dual robust prior evaluation for DRBC.

Implements the dual formulation of the distributionally robust objective:

    R_delta(pi) = sup_{lambda >= 0} { -lambda*delta - lambda*log(M_pi(lambda)) }

where

    M_pi(lambda) = E_{B~mu}[exp(-Z_pi(B) / lambda)]

All exponential computations use logsumexp for numerical stability.
"""

import numpy as np
from scipy.special import logsumexp
from scipy.optimize import minimize_scalar


# ---------------------------------------------------------------------------
# Core building blocks
# ---------------------------------------------------------------------------

def compute_M_pi(Z_vals, probs, lambda_, use_logsumexp=True):
    """Compute the moment-generating-function proxy M_pi(lambda).

    For a discrete (or Monte-Carlo) distribution:

        M_pi(lambda) = sum_i p_i * exp(-Z_i / lambda)

    Parameters
    ----------
    Z_vals : array-like of shape (K,)
        Value estimates Z_pi(b_i) at each support point.
    probs : array-like of shape (K,)
        Probability weights (must be non-negative and sum to 1).
    lambda_ : float
        Dual variable; must be strictly positive.
    use_logsumexp : bool, optional
        When True (default) the computation is done in log-space via
        ``scipy.special.logsumexp`` for numerical stability.

    Returns
    -------
    float
        M_pi(lambda) > 0.

    Raises
    ------
    ValueError
        If *lambda_* is not strictly positive.
    """
    if lambda_ <= 0:
        raise ValueError(f"lambda_ must be > 0, got {lambda_}.")

    Z_vals = np.asarray(Z_vals, dtype=float)
    probs = np.asarray(probs, dtype=float)

    if use_logsumexp:
        log_weights = np.log(np.clip(probs, 1e-300, None)) + (-Z_vals / lambda_)
        log_M = logsumexp(log_weights)
        return float(np.exp(log_M))
    else:
        return float(np.sum(probs * np.exp(-Z_vals / lambda_)))


def dual_objective(lambda_, Z_vals, probs, delta):
    """Dual objective f(lambda) = -lambda*delta - lambda*log(M_pi(lambda)).

    Parameters
    ----------
    lambda_ : float
        Dual variable (> 0).
    Z_vals : array-like of shape (K,)
        Value estimates at each support point.
    probs : array-like of shape (K,)
        Probability weights.
    delta : float
        KL-ball radius (>= 0).

    Returns
    -------
    float
        f(lambda).  Returns ``-inf`` if *lambda_* <= 0 or M_pi is non-positive.
    """
    if lambda_ <= 0:
        return -np.inf

    Z_vals = np.asarray(Z_vals, dtype=float)
    probs = np.asarray(probs, dtype=float)

    # Compute log M_pi in log-space for stability
    log_weights = np.log(np.clip(probs, 1e-300, None)) + (-Z_vals / lambda_)
    log_M = logsumexp(log_weights)

    return float(-lambda_ * delta - lambda_ * log_M)


def optimize_dual(Z_vals, probs, delta, lambda_min=1e-6, lambda_max=1e4, n_grid=200):
    """Maximise the dual objective over lambda in [lambda_min, lambda_max].

    Two complementary strategies are used:

    1. **Grid search** over ``n_grid`` log-spaced values of lambda.
    2. **Bounded scalar minimisation** (``scipy.optimize.minimize_scalar``) using
       the grid-search maximum as a warm-start bracket.

    Parameters
    ----------
    Z_vals : array-like of shape (K,)
        Value estimates at each support point.
    probs : array-like of shape (K,)
        Probability weights.
    delta : float
        KL-ball radius (>= 0).
    lambda_min : float, optional
        Lower bound for lambda search (default 1e-6).
    lambda_max : float, optional
        Upper bound for lambda search (default 1e4).
    n_grid : int, optional
        Number of grid points (default 200).

    Returns
    -------
    dict with keys:
        ``lambda_star`` : float — optimal dual variable.
        ``robust_value`` : float — maximum of the dual objective.
        ``grid_lambdas`` : numpy.ndarray — lambda values used in grid search.
        ``grid_values``  : numpy.ndarray — dual objective values on the grid.
    """
    Z_vals = np.asarray(Z_vals, dtype=float)
    probs = np.asarray(probs, dtype=float)

    grid_lambdas = np.logspace(
        np.log10(lambda_min), np.log10(lambda_max), n_grid
    )
    grid_values = np.array(
        [dual_objective(lam, Z_vals, probs, delta) for lam in grid_lambdas]
    )

    best_grid_idx = int(np.argmax(grid_values))
    lambda_grid_best = grid_lambdas[best_grid_idx]

    # Bounded scalar maximisation (minimise negative objective)
    def neg_obj(lam):
        return -dual_objective(lam, Z_vals, probs, delta)

    result = minimize_scalar(
        neg_obj,
        bounds=(lambda_min, lambda_max),
        method="bounded",
        options={"xatol": 1e-10, "maxiter": 5000},
    )

    if result.success and (-result.fun) >= grid_values[best_grid_idx]:
        lambda_star = float(result.x)
        robust_value = float(-result.fun)
    else:
        lambda_star = float(lambda_grid_best)
        robust_value = float(grid_values[best_grid_idx])

    return {
        "lambda_star": lambda_star,
        "robust_value": robust_value,
        "grid_lambdas": grid_lambdas,
        "grid_values": grid_values,
    }


def bayesian_value(Z_vals, probs):
    """Compute the plain Bayesian (non-robust) value E_mu[Z_pi(B)].

    Parameters
    ----------
    Z_vals : array-like of shape (K,)
        Value estimates at each support point.
    probs : array-like of shape (K,)
        Probability weights.

    Returns
    -------
    float
        sum_i p_i * Z_i.
    """
    Z_vals = np.asarray(Z_vals, dtype=float)
    probs = np.asarray(probs, dtype=float)
    return float(np.dot(probs, Z_vals))
