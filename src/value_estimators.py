"""
value_estimators.py — Model-agnostic value estimators for DRBC.

Provides Monte-Carlo estimators for Z_pi(b) used in the KL-dual and rMLMC
computations.

    * ``estimate_Z_bandit``          — simple linear bandit reward.
    * ``estimate_Z_merton_finite``   — Merton wealth process under power utility.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Bandit estimator
# ---------------------------------------------------------------------------

def estimate_Z_bandit(b, n_inner, rng, reward_mean_fn=None):
    """Estimate Z_pi(b) for a simple bandit policy.

    The default policy uses a linear reward: r_t = b + noise, and Z_pi(b) is
    estimated as the sample mean of *n_inner* independent draws.

    Parameters
    ----------
    b : float
        Parameter value (context / bandit arm mean).
    n_inner : int
        Number of inner MC draws.
    rng : numpy.random.Generator
        Random number generator.
    reward_mean_fn : callable, optional
        Function mapping b to the expected reward.  Defaults to the identity
        ``lambda b: b``.

    Returns
    -------
    float
        Monte-Carlo estimate of Z_pi(b) = E[reward | b].
    """
    if reward_mean_fn is None:
        reward_mean_fn = lambda x: x  # noqa: E731

    mean_r = reward_mean_fn(b)
    rewards = mean_r + rng.standard_normal(n_inner)
    return float(rewards.mean())


# ---------------------------------------------------------------------------
# Merton estimator
# ---------------------------------------------------------------------------

def estimate_Z_merton_finite(
    b,
    r=0.05,
    sigma=0.4,
    T=1.0,
    x0=1.0,
    alpha=0.5,
    n_inner=1000,
    rng=None,
):
    """Estimate E[X_T^alpha / alpha] for the Merton portfolio under power utility.

    The optimal Merton fraction is

        pi*(b) = (b - r) / (sigma^2 * (1 - alpha))

    The wealth SDE is integrated with Euler-Maruyama using 100 time steps:

        X_{t+dt} = X_t * (1 + (r + pi*(b)*(b - r)) * dt
                           + pi*(b) * sigma * sqrt(dt) * Z)

    where Z ~ N(0, 1).

    Parameters
    ----------
    b : float
        Excess-return parameter (drift of the risky asset).
    r : float
        Risk-free rate (default 0.05).
    sigma : float
        Volatility of the risky asset (default 0.4).
    T : float
        Investment horizon (default 1.0).
    x0 : float
        Initial wealth (default 1.0).
    alpha : float
        Risk-aversion exponent for power utility U(x) = x^alpha / alpha
        (default 0.5; must be < 1 and != 0).
    n_inner : int
        Number of simulated paths (default 1000).
    rng : numpy.random.Generator, optional
        Random number generator.  Created fresh if not provided.

    Returns
    -------
    float
        Monte-Carlo estimate of E[X_T^alpha / alpha].
    """
    if rng is None:
        rng = np.random.default_rng()

    n_steps = 100
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)

    # Merton optimal fraction
    pi_star = (b - r) / (sigma ** 2 * (1.0 - alpha))

    # Portfolio drift and diffusion coefficients
    mu_port = r + pi_star * (b - r)
    sig_port = pi_star * sigma

    # Simulate n_inner paths simultaneously
    X = np.full(n_inner, x0, dtype=float)
    for _ in range(n_steps):
        Z = rng.standard_normal(n_inner)
        X = X * (1.0 + mu_port * dt + sig_port * sqrt_dt * Z)
        X = np.maximum(X, 1e-300)  # keep wealth positive

    utility = (X ** alpha) / alpha
    return float(utility.mean())
