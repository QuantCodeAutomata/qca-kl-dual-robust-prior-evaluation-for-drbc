"""
tests/test_exp1_exp2_exp6.py

Pytest suite for the DRBC modules:
  src/priors.py, src/kl_dual.py, src/primal_validation.py,
  src/value_estimators.py, exp/exp6_finite_support_drbc.py
"""

import sys
import os

# Allow imports from src/ and exp/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'exp'))

import numpy as np
import pytest

from priors import FiniteSupportPrior, GaussianPrior
from kl_dual import (
    compute_M_pi,
    dual_objective,
    optimize_dual,
    bayesian_value,
)
from primal_validation import (
    kl_divergence,
    primal_robust_value,
    primal_dual_gap,
)
from value_estimators import estimate_Z_bandit, estimate_Z_merton_finite
from exp6_finite_support_drbc import (
    bayesian_merton_value_finite,
    q_optimizer,
    B_SUPPORT,
    P_BASELINE,
)
from exp2_rmlmc import rmlmc_estimate, _Z_fn, SUPPORT as EXP2_SUPPORT


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

SUPPORT_SMALL = np.array([0.1, 0.5, 0.9])
PROBS_SMALL   = np.array([0.2, 0.5, 0.3])
Z_SMALL       = np.array([1.0, 2.0, 3.0])


# ===========================================================================
# 1. test_kl_divergence_zero
# ===========================================================================

def test_kl_divergence_zero():
    """KL(p || p) must equal zero for any valid distribution p."""
    p = np.array([0.1, 0.4, 0.3, 0.2])
    result = kl_divergence(p, p)
    assert abs(result) < 1e-12, f"KL(p||p) should be 0, got {result}"


# ===========================================================================
# 2. test_dual_equals_bayesian_at_delta_zero
# ===========================================================================

def test_dual_equals_bayesian_at_delta_zero():
    """At delta=0 the dual value must equal the Bayesian value E_mu[Z]."""
    Z_vals = Z_SMALL
    probs  = PROBS_SMALL

    bay = bayesian_value(Z_vals, probs)
    res = optimize_dual(Z_vals, probs, delta=0.0)
    dual_val = res["robust_value"]

    # At delta=0 the dual supremum equals E_mu[Z]
    assert abs(dual_val - bay) < 1e-4, (
        f"Dual at delta=0 ({dual_val:.6f}) should ≈ Bayesian ({bay:.6f})"
    )


# ===========================================================================
# 3. test_robust_value_decreases_with_delta
# ===========================================================================

def test_robust_value_decreases_with_delta():
    """The dual robust value must be non-increasing in delta.

    A larger KL ball gives the adversary more freedom, so the worst-case
    value can only be lower (or equal).
    """
    Z_vals = Z_SMALL
    probs  = PROBS_SMALL
    deltas = [0.0, 0.01, 0.05, 0.10, 0.30]
    vals   = [optimize_dual(Z_vals, probs, d)["robust_value"] for d in deltas]

    for i in range(len(vals) - 1):
        assert vals[i] >= vals[i + 1] - 1e-8, (
            f"Robust value should be non-increasing: "
            f"delta={deltas[i]} gives {vals[i]:.6f}, "
            f"delta={deltas[i+1]} gives {vals[i+1]:.6f}"
        )


# ===========================================================================
# 4. test_primal_dual_agreement
# ===========================================================================

def test_primal_dual_agreement():
    """Primal and dual values must agree within 1e-3 for a small discrete case."""
    Z_vals = Z_SMALL
    probs  = PROBS_SMALL
    delta  = 0.10

    dual_res   = optimize_dual(Z_vals, probs, delta)
    primal_res = primal_robust_value(Z_vals, probs, delta)
    gap        = primal_dual_gap(primal_res["primal_value"], dual_res["robust_value"])

    assert gap < 1e-3, (
        f"Primal-dual gap {gap:.2e} exceeds 1e-3 for delta={delta}"
    )


# ===========================================================================
# 5. test_m_pi_positive
# ===========================================================================

def test_m_pi_positive():
    """M_pi(lambda) must be strictly positive for any lambda > 0."""
    Z_vals = Z_SMALL
    probs  = PROBS_SMALL

    for lam in [0.01, 0.1, 1.0, 10.0, 1000.0]:
        M = compute_M_pi(Z_vals, probs, lam)
        assert M > 0, f"M_pi({lam}) should be > 0, got {M}"
        assert np.isfinite(M), f"M_pi({lam}) should be finite, got {M}"


# ===========================================================================
# 6. test_rmlmc_positive_variance
# ===========================================================================

def test_rmlmc_positive_variance():
    """rMLMC estimator must have positive and finite variance across replications."""
    prior = FiniteSupportPrior(EXP2_SUPPORT, np.array([0.05, 0.35, 0.35, 0.15, 0.10]))

    rng_master = np.random.default_rng(7)
    Z_lookup = np.array([0.5, 1.2, 0.9, 0.8, 1.0])

    def Z_fn(b, n_inner, rng):
        return _Z_fn(b, n_inner, rng, Z_lookup, EXP2_SUPPORT)

    lambda_ = 1.0
    estimates = np.array([
        rmlmc_estimate(Z_fn, prior, lambda_, n=50,
                       rng=np.random.default_rng(rng_master.integers(0, 2**32)))
        for _ in range(30)
    ])

    var = float(estimates.var())
    assert var > 0, "rMLMC variance should be positive (estimator is stochastic)."
    assert np.isfinite(var), f"rMLMC variance should be finite, got {var}."
    assert all(np.isfinite(estimates)), "All rMLMC estimates should be finite."


# ===========================================================================
# 7. test_simplex_constraint_q_optimizer
# ===========================================================================

def test_simplex_constraint_q_optimizer():
    """q* returned by q_optimizer must lie on the simplex (sum=1, all>0)."""
    res = q_optimizer(B_SUPPORT, P_BASELINE, lambda_dual=100,
                      lr=0.01, h=1e-6, max_iter=500, tol=1e-6)
    q_star = res["q_star"]

    assert abs(q_star.sum() - 1.0) < 1e-8, (
        f"q_star must sum to 1, got {q_star.sum()}"
    )
    assert all(q_star > 0), f"All q_star entries must be > 0, got {q_star}"


# ===========================================================================
# 8. test_prior_validation
# ===========================================================================

def test_prior_validation():
    """FiniteSupportPrior must raise ValueError for invalid inputs."""
    # Negative probability
    with pytest.raises(ValueError, match="non-negative"):
        FiniteSupportPrior([0.1, 0.9], [-0.1, 1.1])

    # Probabilities don't sum to 1
    with pytest.raises(ValueError, match="sum to 1"):
        FiniteSupportPrior([0.1, 0.9], [0.3, 0.3])

    # Mismatched lengths
    with pytest.raises(ValueError, match="match"):
        FiniteSupportPrior([0.1, 0.5, 0.9], [0.5, 0.5])


# ===========================================================================
# 9. test_bayesian_value_finite_support
# ===========================================================================

def test_bayesian_value_finite_support():
    """bayesian_value must match sum_i p_i Z_i exactly."""
    Z_vals = Z_SMALL
    probs  = PROBS_SMALL

    expected = float(np.dot(probs, Z_vals))
    result   = bayesian_value(Z_vals, probs)

    assert abs(result - expected) < 1e-15, (
        f"bayesian_value {result} != dot(probs, Z) {expected}"
    )


# ===========================================================================
# 10. test_dual_objective_finite
# ===========================================================================

def test_dual_objective_finite():
    """dual_objective must return a finite float for valid inputs."""
    Z_vals = Z_SMALL
    probs  = PROBS_SMALL

    for lam in [0.01, 0.5, 2.0, 100.0]:
        for delta in [0.0, 0.1, 1.0]:
            val = dual_objective(lam, Z_vals, probs, delta)
            assert np.isfinite(val), (
                f"dual_objective(lambda={lam}, delta={delta}) = {val} is not finite."
            )
            assert isinstance(val, float), (
                f"dual_objective should return float, got {type(val)}."
            )
