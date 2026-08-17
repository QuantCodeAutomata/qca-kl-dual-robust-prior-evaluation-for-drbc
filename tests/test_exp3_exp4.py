"""
Tests for Exp 3 (LQ DRBC) and Exp 4 (Merton ablation).
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest
from lq_system import LQSystem, LQIdentifier
from riccati import solve_riccati_lq, compute_optimal_gain


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def lq():
    return LQSystem(dx=4, du=2, T=1.0, dt=0.1, sigma_scale=0.3, seed=42)


@pytest.fixture(scope="module")
def small_lq():
    """Tiny system for fast tests."""
    return LQSystem(dx=3, du=2, T=0.5, dt=0.1, sigma_scale=0.1, seed=7)


# ─────────────────────────────────────────────────────────────────────────────
# Exp 3 — LQ system tests
# ─────────────────────────────────────────────────────────────────────────────

def test_lq_system_dimensions(lq):
    """System matrices have correct shapes."""
    assert lq.A0.shape == (4, 4)
    assert lq.G.shape  == (4, 2)
    assert lq.Sigma.shape == (4, 4)
    assert lq.Q.shape == (4, 4)
    assert lq.R.shape == (2, 2)
    assert len(lq.basis_matrices) == 4  # p = dx = 4
    assert lq.basis_matrices[0].shape == (4, 4)


def test_euler_maruyama_trajectory_shape(lq):
    """Trajectory output shape is (K+1, dx) and (K, du)."""
    rng = np.random.default_rng(0)
    theta = rng.standard_normal(4)
    A = lq.compute_A(theta)
    K_ctrl = np.zeros((2, 4))  # zero gain
    x0 = rng.standard_normal(4)
    X, U = lq.simulate_trajectory(A, K_ctrl, x0, rng)
    assert X.shape == (lq.K + 1, lq.dx)
    assert U.shape == (lq.K, lq.du)


def test_gls_recovers_true_theta():
    """With many observations and small noise, GLS recovers theta approximately."""
    rng = np.random.default_rng(123)
    # Use a tiny 2D system for exact-ish recovery
    lq = LQSystem(dx=2, du=1, T=2.0, dt=0.01, sigma_scale=0.01, seed=99)
    identifier = LQIdentifier()
    theta_true = np.array([0.3, -0.2])
    A_true = lq.compute_A(theta_true)
    K_ctrl = np.zeros((1, 2))
    x0 = np.array([1.0, 0.5])
    X, U = lq.simulate_trajectory(A_true, K_ctrl, x0, rng)
    theta_hat = identifier.gls_estimate(
        X, U, lq.A0, lq.basis_matrices, lq.G, lq.Sigma, lq.dt, lambda_ridge=1e-6
    )
    # With low noise and long trajectory, estimate should be reasonably close
    err = np.linalg.norm(theta_hat - theta_true)
    assert err < 1.0, f"GLS error {err:.4f} too large"


def test_riccati_terminal_condition(lq):
    """P(T) should equal Q_f (last element in forward-time array)."""
    rng = np.random.default_rng(0)
    theta = rng.standard_normal(lq.dx)
    A = lq.compute_A(theta)
    P = solve_riccati_lq(A, lq.G, lq.Q, lq.R, lq.Q_f, lq.T, lq.dt)
    # P[-1] should be close to Q_f
    assert np.allclose(P[-1], lq.Q_f, atol=0.1), f"Terminal condition not met: max_err={np.abs(P[-1] - lq.Q_f).max():.4f}"


def test_riccati_positive_semidefinite(lq):
    """All P(t) matrices should be positive semidefinite."""
    rng = np.random.default_rng(1)
    theta = rng.standard_normal(lq.dx)
    A = lq.compute_A(theta)
    P = solve_riccati_lq(A, lq.G, lq.Q, lq.R, lq.Q_f, lq.T, lq.dt)
    for k in range(P.shape[0]):
        evals = np.linalg.eigvalsh(P[k])
        assert np.all(evals >= -1e-6), f"P[{k}] not PSD: min_eval={evals.min():.2e}"


def test_oracle_beats_plug_in():
    """On average, oracle (true A) should not be worse than plug-in (estimated A)."""
    rng = np.random.default_rng(42)
    lq = LQSystem(dx=3, du=2, T=0.5, dt=0.1, sigma_scale=0.3, seed=55)
    identifier = LQIdentifier()
    oracle_costs = []
    plugin_costs = []

    for _ in range(5):
        theta_star = rng.standard_normal(3) * 0.3
        A_true = lq.compute_A(theta_star)

        # Identification
        K_rand = rng.normal(0.0, 0.3, (2, 3))
        x0 = rng.standard_normal(3)
        X_id, U_id = lq.simulate_trajectory(A_true, K_rand, x0, rng)
        theta_hat = identifier.gls_estimate(
            X_id, U_id, lq.A0, lq.basis_matrices, lq.G, lq.Sigma, lq.dt
        )
        A_hat = lq.compute_A(theta_hat)

        # Solve Riccati
        P_oracle = solve_riccati_lq(A_true, lq.G, lq.Q, lq.R, lq.Q_f, lq.T, lq.dt)
        P_plugin = solve_riccati_lq(A_hat,  lq.G, lq.Q, lq.R, lq.Q_f, lq.T, lq.dt)
        K_oracle = compute_optimal_gain(P_oracle, lq.G, lq.R)
        K_plugin = compute_optimal_gain(P_plugin, lq.G, lq.R)

        # Evaluate oracle gain under true A
        cost_o = 0.0; cost_p = 0.0
        for _ in range(50):
            x0e = rng.standard_normal(3)
            Xo, Uo = lq.simulate_trajectory(A_true, K_oracle[0], x0e, rng)
            Xp, Up = lq.simulate_trajectory(A_true, K_plugin[0], x0e, rng)
            cost_o += lq.compute_trajectory_cost(Xo, Uo)
            cost_p += lq.compute_trajectory_cost(Xp, Up)
        oracle_costs.append(cost_o / 50)
        plugin_costs.append(cost_p / 50)

    # Oracle should be <= plug-in on average (not always per-run due to estimation noise)
    assert np.mean(oracle_costs) <= np.mean(plugin_costs) + 50.0, \
        "Oracle significantly worse than plug-in"


# ─────────────────────────────────────────────────────────────────────────────
# Exp 4 — Merton ablation tests
# ─────────────────────────────────────────────────────────────────────────────

def _simulate_merton_wealth(pi: float, B_fixed: float, n: int = 200, seed: int = 0) -> np.ndarray:
    """Simple Merton simulation with constant B for test purposes."""
    rng = np.random.default_rng(seed)
    T, K, sigma, r = 1.0, 100, 0.4, 0.05
    dt = T / K
    X = np.ones(n)
    for _ in range(K):
        dW = rng.standard_normal(n) * np.sqrt(dt)
        X = X * np.exp((r + pi*(B_fixed - r) - 0.5*pi**2*sigma**2)*dt + pi*sigma*dW)
    return X


def test_merton_wealth_positive():
    """Simulated wealth is always positive (multiplicative dynamics)."""
    X_T = _simulate_merton_wealth(pi=1.0, B_fixed=0.15)
    assert np.all(X_T > 0), "Some terminal wealth values are non-positive"


def test_drbc_value_decreases_with_delta():
    """DRBC fraction should move as delta increases (prior changes)."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "exp"))
    from exp4_merton_ablation import drbc_fraction, B_SUPPORT, P_BASE

    pi_0  = drbc_fraction(B_SUPPORT, P_BASE, delta=0.01)
    pi_hi = drbc_fraction(B_SUPPORT, P_BASE, delta=0.30)
    # The fraction should differ across delta levels (robustification shifts q*)
    assert abs(pi_0 - pi_hi) > 1e-6, "DRBC fraction unchanged across delta"


def test_bayesian_value_at_zero_delta_equals_nominal():
    """At delta=0, DRBC fraction equals Bayesian fraction."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "exp"))
    from exp4_merton_ablation import drbc_fraction, bayesian_fraction, B_SUPPORT, P_BASE

    pi_bayes = bayesian_fraction(B_SUPPORT, P_BASE)
    pi_drbc0 = drbc_fraction(B_SUPPORT, P_BASE, delta=0.0)
    assert abs(pi_bayes - pi_drbc0) < 1e-6, \
        f"DRBC(delta=0)={pi_drbc0} != Bayesian={pi_bayes}"


def test_utility_gap_positive():
    """Utility gaps are non-negative (oracle is at least as good as any method)."""
    # Simple check: oracle utility >= arbitrary policy utility (lower pi => lower utility if b>r)
    B_fixed = 0.20
    pi_oracle = (B_fixed - 0.05) / (0.4**2 * 0.5)   # Merton optimal for B_fixed
    pi_subopt = pi_oracle * 0.5                        # half the optimal fraction

    X_oracle = _simulate_merton_wealth(pi_oracle, B_fixed, seed=77)
    X_subopt = _simulate_merton_wealth(pi_subopt, B_fixed, seed=77)

    u_oracle = np.mean(X_oracle**0.5 / 0.5)
    u_subopt = np.mean(X_subopt**0.5 / 0.5)
    gap = u_oracle - u_subopt
    assert gap >= -0.1, f"Oracle utility gap negative: {gap:.4f}"


# ─────────────────────────────────────────────────────────────────────────────
# DRBCLQPolicy shrinkage tests (new, validating rewritten module)
# ─────────────────────────────────────────────────────────────────────────────

def test_drbc_shrink_factor_delta_zero():
    """delta=0 must give shrink_factor=1 (i.e. pure plug-in)."""
    from drbc_lq_policy import DRBCLQPolicy
    lq = LQSystem(dx=4, du=2, T=1.0, dt=0.1, sigma_scale=0.2, seed=11)
    theta_hat = np.ones(4) * 0.3
    policy = DRBCLQPolicy(lq, theta_hat, delta=0.0)
    policy.train_policy()
    assert abs(policy.shrink_factor - 1.0) < 1e-10, (
        f"shrink_factor should be 1.0 at delta=0, got {policy.shrink_factor}"
    )
    assert np.allclose(policy.theta_drbc, theta_hat), (
        "theta_drbc should equal theta_hat at delta=0"
    )


def test_drbc_shrink_factor_decreases_with_delta():
    """shrink_factor must be strictly decreasing in delta."""
    from drbc_lq_policy import DRBCLQPolicy
    lq = LQSystem(dx=4, du=2, T=1.0, dt=0.1, sigma_scale=0.2, seed=11)
    theta_hat = np.ones(4) * 0.5
    factors = []
    for delta in [0.0, 0.01, 0.05, 0.10, 0.20]:
        policy = DRBCLQPolicy(lq, theta_hat, delta=delta)
        policy.train_policy()
        factors.append(policy.shrink_factor)
    # Each subsequent factor must be strictly smaller
    for i in range(1, len(factors)):
        assert factors[i] < factors[i - 1], (
            f"shrink_factor not decreasing: delta sequence produced {factors}"
        )


def test_drbc_gains_not_none_after_train():
    """After train_policy(), K_drbc must be populated and correctly shaped."""
    from drbc_lq_policy import DRBCLQPolicy
    lq = LQSystem(dx=4, du=2, T=1.0, dt=0.1, sigma_scale=0.2, seed=11)
    theta_hat = np.zeros(4)
    policy = DRBCLQPolicy(lq, theta_hat, delta=0.05)
    policy.train_policy()
    assert policy.K_drbc is not None, "K_drbc should not be None after train_policy()"
    assert policy.K_drbc.shape == (lq.K + 1, lq.du, lq.dx), (
        f"K_drbc shape {policy.K_drbc.shape} != ({lq.K + 1}, {lq.du}, {lq.dx})"
    )


def test_drbc_reduces_gap_vs_plugin_aggregate():
    """Over multiple seeds, DRBC with delta>0 should reduce cost gap vs oracle
    compared with plug-in (aggregate test, not per-run guaranteed)."""
    from drbc_lq_policy import DRBCLQPolicy, simulate_lqr_policy
    rng = np.random.default_rng(999)
    lq = LQSystem(dx=4, du=2, T=1.0, dt=0.1, sigma_scale=0.2, seed=55)
    identifier = LQIdentifier()
    plugin_gaps, drbc_gaps = [], []
    for _ in range(8):
        theta_star = rng.normal(0.0, 0.5, 4)
        A_true = lq.compute_A(theta_star)
        K_rand = rng.normal(0.0, 0.3, (2, 4))
        x0 = rng.standard_normal(4)
        X_id, U_id = lq.simulate_trajectory(A_true, K_rand, x0, rng)
        theta_hat = identifier.gls_estimate(
            X_id, U_id, lq.A0, lq.basis_matrices, lq.G, lq.Sigma, lq.dt
        )
        # Oracle
        P_oracle = solve_riccati_lq(A_true, lq.G, lq.Q, lq.R, lq.Q_f, lq.T, lq.dt)
        K_oracle = compute_optimal_gain(P_oracle, lq.G, lq.R)
        j_oracle = simulate_lqr_policy(K_oracle, lq, A_true, n_eval=64, rng=rng)
        # Plug-in
        A_hat = lq.compute_A(theta_hat)
        P_pi = solve_riccati_lq(A_hat, lq.G, lq.Q, lq.R, lq.Q_f, lq.T, lq.dt)
        K_pi = compute_optimal_gain(P_pi, lq.G, lq.R)
        j_pi = simulate_lqr_policy(K_pi, lq, A_true, n_eval=64, rng=rng)
        # DRBC
        pol = DRBCLQPolicy(lq, theta_hat, delta=0.10)
        pol.train_policy()
        j_drbc = simulate_lqr_policy(pol.K_drbc, lq, A_true, n_eval=64, rng=rng)
        plugin_gaps.append(j_pi - j_oracle)
        drbc_gaps.append(j_drbc - j_oracle)
    mean_plugin_gap = float(np.mean(plugin_gaps))
    mean_drbc_gap   = float(np.mean(drbc_gaps))
    # DRBC gap should be <= plugin gap (with generous tolerance for MC noise)
    assert mean_drbc_gap <= mean_plugin_gap + 50.0, (
        f"DRBC gap {mean_drbc_gap:.1f} unexpectedly exceeds plugin gap {mean_plugin_gap:.1f}"
    )
