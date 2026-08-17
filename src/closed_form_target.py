"""
Closed-form benchmark for the Merton terminal-wealth learner.
Appendix H.4.1 analytical solution.
"""
import numpy as np
import pandas as pd
from scipy import stats

# Paper parameters
SIGMA = 0.4
SIGMA_0 = 2.0
R_VALS = [0.05, 0.10]
B_0 = 0.1
T = 1.0
MU_0 = 0.1
B_VALS = [0.1, 0.3]
X_0 = 1.0
ALPHA = 0.5  # CRRA exponent gamma = alpha = 0.5


def merton_optimal_fraction(b, r, sigma, gamma):
    """Return Merton optimal risky fraction pi* = (b - r) / (sigma^2 * (1 - gamma))."""
    return (b - r) / (sigma**2 * (1.0 - gamma))


def merton_expected_utility(b, r, sigma, gamma, x0, T):
    """
    Return closed-form expected utility E[U(X_T*)] under optimal Merton policy.

    U(x) = x^gamma / gamma  (power / CRRA utility).

    E[U(X_T*)] = (x0^gamma / gamma) * exp(gamma * (r + (b-r)^2 /
                  (2*sigma^2*(1-gamma))) * T)
    """
    exponent = gamma * (r + (b - r) ** 2 / (2.0 * sigma**2 * (1.0 - gamma))) * T
    return (x0**gamma / gamma) * np.exp(exponent)


def merton_terminal_wealth_samples(b, r, sigma, gamma, x0, T, W_T_samples):
    """
    Return optimal terminal wealth X_T* for given Brownian-motion draws W_T.

    X_T* = x0 * exp((r + pi*(b-r) - 0.5*pi^2*sigma^2)*T + pi*sigma*W_T)

    Parameters
    ----------
    b, r, sigma, gamma, x0, T : float
        Model parameters.
    W_T_samples : array-like
        Realisations of W_T ~ N(0, T).

    Returns
    -------
    np.ndarray
        Array of X_T* values, same length as W_T_samples.
    """
    W_T_samples = np.asarray(W_T_samples, dtype=float)
    pi = merton_optimal_fraction(b, r, sigma, gamma)
    drift = (r + pi * (b - r) - 0.5 * pi**2 * sigma**2) * T
    diffusion = pi * sigma * W_T_samples
    return x0 * np.exp(drift + diffusion)


def compute_closed_form_table(r_vals, b_vals, sigma, gamma, x0, T):
    """
    Return DataFrame with closed-form expected utilities for all (r, b) pairs.

    Columns: r, b, utility_cf
    """
    rows = []
    for r in r_vals:
        for b in b_vals:
            utility = merton_expected_utility(b, r, sigma, gamma, x0, T)
            rows.append({"r": r, "b": b, "utility_cf": utility})
    return pd.DataFrame(rows)
