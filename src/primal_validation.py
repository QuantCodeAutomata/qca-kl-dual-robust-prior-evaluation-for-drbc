"""
primal_validation.py — Primal KL-ball optimisation for DRBC validation.

Solves the primal problem

    min_{q in Delta_K} sum_i q_i Z_i   s.t.  KL(q || p) <= delta

via SLSQP and provides utilities for comparing with the dual value.
"""

import numpy as np
from scipy.optimize import minimize


# ---------------------------------------------------------------------------
# KL divergence
# ---------------------------------------------------------------------------

def kl_divergence(q, p):
    """Compute KL(q || p) = sum_i q_i log(q_i / p_i).

    Entries where q_i == 0 contribute 0 (convention 0*log(0) = 0).

    Parameters
    ----------
    q : array-like of shape (K,)
        Numerator distribution (non-negative, need not sum to 1 for raw call).
    p : array-like of shape (K,)
        Denominator distribution (strictly positive entries).

    Returns
    -------
    float
        KL divergence.  Returns ``+inf`` if p_i == 0 for some i where q_i > 0.
    """
    q = np.asarray(q, dtype=float)
    p = np.asarray(p, dtype=float)

    kl = 0.0
    for qi, pi in zip(q, p):
        if qi == 0.0:
            continue
        if pi <= 0.0:
            return np.inf
        kl += qi * np.log(qi / pi)
    return float(kl)


# ---------------------------------------------------------------------------
# Primal optimisation
# ---------------------------------------------------------------------------

def primal_robust_value(Z_vals, p_baseline, delta):
    """Solve the primal KL-robust minimisation problem.

    Finds:

        q* = argmin_{q in Delta_K} sum_i q_i Z_i   s.t.  KL(q||p) <= delta

    using ``scipy.optimize.minimize`` with the SLSQP method.

    Parameters
    ----------
    Z_vals : array-like of shape (K,)
        Value estimates at each support point.
    p_baseline : array-like of shape (K,)
        Reference (baseline) distribution.
    delta : float
        KL-ball radius.  When delta == 0 the constraint forces q == p, so the
        primal value equals the Bayesian value.

    Returns
    -------
    dict with keys:
        ``primal_value`` : float — optimal objective value sum_i q*_i Z_i.
        ``q_opt``        : numpy.ndarray — optimal distribution q*.
        ``success``      : bool — True if the optimiser converged.
    """
    Z_vals = np.asarray(Z_vals, dtype=float)
    p_baseline = np.asarray(p_baseline, dtype=float)
    K = len(Z_vals)

    def objective(q):
        return float(np.dot(q, Z_vals))

    def grad_objective(q):
        return Z_vals.copy()

    # Constraint: KL(q||p) - delta <= 0
    def kl_constraint(q):
        return delta - kl_divergence(q, p_baseline)

    constraints = [
        {"type": "ineq", "fun": kl_constraint},
        {"type": "eq",   "fun": lambda q: q.sum() - 1.0},
    ]
    bounds = [(1e-12, 1.0)] * K

    # Warm-start: feasible point is p_baseline itself
    q0 = p_baseline.copy()

    result = minimize(
        objective,
        q0,
        jac=grad_objective,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 2000},
    )

    q_opt = np.clip(result.x, 0.0, None)
    # Re-normalise to correct for any tiny numerical drift
    q_sum = q_opt.sum()
    if q_sum > 0:
        q_opt /= q_sum

    return {
        "primal_value": float(np.dot(q_opt, Z_vals)),
        "q_opt": q_opt,
        "success": bool(result.success),
    }


# ---------------------------------------------------------------------------
# Gap
# ---------------------------------------------------------------------------

def primal_dual_gap(primal_val, dual_val):
    """Absolute difference between primal and dual values.

    By strong duality the gap should be zero (or very small numerically).

    Parameters
    ----------
    primal_val : float
    dual_val   : float

    Returns
    -------
    float
        ``|primal_val - dual_val|``.
    """
    return float(abs(primal_val - dual_val))
