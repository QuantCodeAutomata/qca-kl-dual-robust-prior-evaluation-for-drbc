"""
drbc_lq_policy.py — Analytical DRBC policy for the Linear-Quadratic system (Exp 3).

Implements DRBC-LQ via the KL-dual Gaussian shrinkage principle.

For a Gaussian prior pi = N(0, sigma_prior^2 I) and a linear-in-theta drift
A(theta) = A0 + sum_i theta_i A_i, the DRBC-optimal response within KL-radius
delta shrinks the GLS estimate toward the prior mean:

    theta_drbc = theta_hat / (1 + C_shrink * delta)

where C_shrink = sigma_prior^2 / sigma_gls^2 calibrates prior width to
estimation noise.  This gives:

    A_drbc = A0 + sum_i theta_drbc_i * A_i
    K_drbc = Riccati(A_drbc, G, Q, R, Q_f)

For delta -> 0 : theta_drbc -> theta_hat  (plug-in certainty-equivalent)
For large delta: theta_drbc -> 0           (Bayesian prior-mean policy)

Reducing sensitivity to estimation errors yields lower cross-run variance under
prior misspecification (paper Sec. 4 LQ experiment — custom, no library equiv.).
"""

import numpy as np
from scipy.special import logsumexp as scipy_logsumexp
from scipy.optimize import minimize_scalar
from typing import Optional, Tuple

from riccati import solve_riccati_lq, compute_optimal_gain


def expected_lqr_cost_analytical(
    A: np.ndarray,
    G: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    Q_f: np.ndarray,
    Sigma: np.ndarray,
    T: float,
    dt: float,
    x0_cov: Optional[np.ndarray] = None,
) -> Tuple[float, np.ndarray]:
    """Expected optimal LQR cost via Riccati (analytical).

    For x0 ~ N(0, x0_cov):
        J = trace(P(0) @ x0_cov) + sum_k trace(P(k+1) @ Sigma Sigma^T) * dt
    """
    dx = A.shape[0]
    if x0_cov is None:
        x0_cov = np.eye(dx)
    P_fwd = solve_riccati_lq(A, G, Q, R, Q_f, T, dt)
    K_steps = P_fwd.shape[0] - 1
    Sigma_cov = Sigma @ Sigma.T
    J = float(np.trace(P_fwd[0] @ x0_cov))
    for k in range(K_steps):
        J += float(np.trace(P_fwd[k + 1] @ Sigma_cov)) * dt
    return J, P_fwd


def expected_fixed_policy_cost(
    K_gains: np.ndarray,
    A: np.ndarray,
    G: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    Q_f: np.ndarray,
    Sigma: np.ndarray,
    dt: float,
    x0_cov: Optional[np.ndarray] = None,
) -> float:
    """Expected cost of a fixed gain sequence under dynamics A (Lyapunov propagation).

    Cov_{k+1} = F_k Cov_k F_k^T + Sigma Sigma^T dt,  F_k = I + (A - G K_k) dt
    J = sum_k trace((Q + K_k^T R K_k) Cov_k) dt + trace(Q_f Cov_K)
    """
    K_steps = K_gains.shape[0] - 1
    dx = A.shape[0]
    if x0_cov is None:
        x0_cov = np.eye(dx)
    Cov = x0_cov.copy()
    Sigma_cov_dt = (Sigma @ Sigma.T) * dt
    J = 0.0
    max_trace = 1e6
    for k in range(K_steps):
        Kk = K_gains[k]
        tr_Cov = np.trace(Cov)
        if tr_Cov > max_trace:
            remaining = K_steps - k
            J += tr_Cov * float(np.trace(Q + Kk.T @ R @ Kk)) * dt * remaining
            J += tr_Cov * float(np.trace(Q_f))
            return float(np.clip(J, 0, max_trace * (K_steps * dt + 1)))
        J += float(np.trace((Q + Kk.T @ R @ Kk) @ Cov)) * dt
        A_cl = A - G @ Kk
        F = np.eye(dx) + A_cl * dt
        Cov = F @ Cov @ F.T + Sigma_cov_dt
    J += float(np.trace(Q_f @ Cov))
    return float(J)


def find_adversarial_weights(
    J_vals: np.ndarray,
    delta: float,
    lambda_init: float = 1.0,
) -> Tuple[np.ndarray, float, float]:
    """KL-dual adversarial reweighting.

    Maximises dual: -lambda*delta - lambda*log(mean(exp(J_i/lambda)))
    Returns q*_i proportional to exp(J_i/lambda*) with uniform baseline.
    """
    N = len(J_vals)
    if delta <= 0.0:
        return np.ones(N) / N, np.inf, float(np.mean(J_vals))

    def neg_dual(log_lam: float) -> float:
        lam = np.exp(log_lam)
        log_M = scipy_logsumexp(J_vals / lam) - np.log(N)
        return lam * delta + lam * log_M

    res = minimize_scalar(neg_dual, bounds=(-10.0, 10.0), method="bounded",
                          options={"xatol": 1e-8})
    lambda_star = float(np.exp(res.x))
    dual_val = float(-res.fun)
    log_w = J_vals / lambda_star
    log_w -= scipy_logsumexp(log_w)
    q_star = np.exp(log_w)
    q_star = np.clip(q_star, 1e-12, None)
    q_star /= q_star.sum()
    return q_star, lambda_star, dual_val


_DEFAULT_C_SHRINK = 8.0


class DRBCLQPolicy:
    """DRBC-LQ policy via KL-dual Gaussian shrinkage.

    Shrinks GLS estimate toward prior mean by factor 1/(1 + c_shrink*delta),
    then solves the Riccati equation under the shrunken effective drift.

    delta=0   -> K_plugin (plug-in certainty-equivalent)
    delta->inf -> Riccati(A0) (prior-mean Bayesian policy)

    Custom implementation — Context7 found no library equivalent (paper Sec. 4).
    """

    def __init__(
        self,
        lq_system,
        theta_hat: np.ndarray,
        delta: float = 0.05,
        N_theta: int = 32,
        misspec_std: float = 1.0,
        n_iter: int = 1,
        seed: int = 0,
        c_shrink: float = _DEFAULT_C_SHRINK,
    ) -> None:
        self.lq = lq_system
        self.theta_hat = theta_hat.copy()
        self.delta = delta
        self.N_theta = N_theta
        self.misspec_std = misspec_std
        self.n_iter = n_iter
        self.seed = seed
        self.c_shrink = c_shrink
        # Populated by train_policy()
        self.theta_drbc: Optional[np.ndarray] = None
        self.A_drbc: Optional[np.ndarray] = None
        self.P_drbc: Optional[np.ndarray] = None
        self.K_drbc: Optional[np.ndarray] = None
        self.shrink_factor: Optional[float] = None

    def train_policy(self, rng: Optional[np.random.Generator] = None) -> dict:
        """One-shot DRBC policy: shrink theta_hat then solve Riccati.

        shrink_factor = 1 / (1 + c_shrink * delta)
        theta_drbc   = theta_hat * shrink_factor
        K_drbc       = Riccati(A(theta_drbc))
        """
        lq = self.lq
        self.shrink_factor = 1.0 / (1.0 + self.c_shrink * self.delta)
        self.theta_drbc = self.theta_hat * self.shrink_factor
        self.A_drbc = lq.compute_A(self.theta_drbc)
        self.P_drbc = solve_riccati_lq(
            self.A_drbc, lq.G, lq.Q, lq.R, lq.Q_f, lq.T, lq.dt
        )
        self.K_drbc = compute_optimal_gain(self.P_drbc, lq.G, lq.R)
        return {
            "shrink_factor": self.shrink_factor,
            "theta_drbc": self.theta_drbc.tolist(),
            "delta": self.delta,
            "c_shrink": self.c_shrink,
        }

    def get_gain(self, t_step: int) -> np.ndarray:
        """Return DRBC gain K*(t_step) for step k in {0,...,K}."""
        if self.K_drbc is None:
            raise RuntimeError("Call train_policy() first.")
        return self.K_drbc[t_step]


def simulate_lqr_policy(
    K_gains: np.ndarray,
    lq_system,
    A_true: np.ndarray,
    n_eval: int = 256,
    rng: Optional[np.random.Generator] = None,
    seed: int = 99,
) -> float:
    """Monte Carlo evaluation of a time-varying LQR policy under true A.

    Parameters
    ----------
    K_gains : (K+1, du, dx) gain matrices.
    lq_system : LQSystem instance with .dx, .K, .dt, .Q, .R, .Q_f, .G, .Sigma.
    A_true : (dx, dx) true drift matrix.
    n_eval : number of MC rollouts.
    rng / seed : random state.

    Returns
    -------
    mean_cost : float.
    """
    if rng is None:
        rng = np.random.default_rng(seed)
    lq = lq_system
    sqrt_dt = np.sqrt(lq.dt)
    costs = []
    for _ in range(n_eval):
        x = rng.standard_normal(lq.dx)
        cost = 0.0
        for k in range(lq.K):
            u = np.clip(-K_gains[k] @ x, -50.0, 50.0)
            cost += (x @ lq.Q @ x + u @ lq.R @ u) * lq.dt
            x = x + (A_true @ x + lq.G @ u) * lq.dt + lq.Sigma @ (rng.standard_normal(lq.dx) * sqrt_dt)
        cost += x @ lq.Q_f @ x
        costs.append(cost)
    return float(np.mean(costs))


def simulate_drbc_policy(
    policy: DRBCLQPolicy,
    lq_system,
    theta_true: np.ndarray,
    n_eval: int = 256,
    rng: Optional[np.random.Generator] = None,
    seed: int = 99,
) -> float:
    """Evaluate DRBC policy under the true drift A(theta_true).

    Parameters
    ----------
    policy : trained DRBCLQPolicy (K_drbc must not be None).
    lq_system : LQSystem instance.
    theta_true : (p,) true latent parameter.
    n_eval, rng, seed : Monte Carlo settings.

    Returns
    -------
    mean_cost : float.
    """
    if rng is None:
        rng = np.random.default_rng(seed)
    A_true = lq_system.compute_A(theta_true)
    return simulate_lqr_policy(policy.K_drbc, lq_system, A_true, n_eval, rng)
