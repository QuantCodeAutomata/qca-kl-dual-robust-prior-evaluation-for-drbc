"""
riccati.py — Backward matrix Riccati ODE solver for LQ control.

Riccati ODE (backward in time):
    -dP/dt = A^T P + P A - P G R^{-1} G^T P + Q
    P(T) = Q_f

The optimal feedback gain is:
    K*(t) = R^{-1} G^T P(t)
"""

import numpy as np
from scipy.integrate import solve_ivp
from typing import Tuple


def solve_riccati_lq(
    A: np.ndarray,
    G: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    Q_f: np.ndarray,
    T: float,
    dt: float,
) -> np.ndarray:
    """Solve the backward matrix Riccati ODE.

    Parameters
    ----------
    A  : (dx, dx) drift matrix.
    G  : (dx, du) control loading.
    Q  : (dx, dx) state cost matrix.
    R  : (du, du) control cost matrix.
    Q_f: (dx, dx) terminal cost matrix.
    T  : horizon.
    dt : time step for output grid.

    Returns
    -------
    P_fwd : (K+1, dx, dx) solution in *forward* time order,
            i.e. P_fwd[0] = P(0), P_fwd[-1] = P(T) = Q_f.
    """
    dx = A.shape[0]
    R_inv = np.linalg.inv(R)
    GRGt = G @ R_inv @ G.T  # (dx, dx)

    def riccati_ode(tau: float, p_flat: np.ndarray) -> np.ndarray:
        """ODE in backward time tau = T - t, so dP/dtau = A^T P + P A - P G R^{-1} G^T P + Q."""
        P = p_flat.reshape(dx, dx)
        dP = A.T @ P + P @ A - P @ GRGt @ P + Q
        return dP.ravel()

    # Solve in backward time: tau in [0, T] with P(tau=0) = Q_f
    p0 = Q_f.ravel()
    K = int(round(T / dt))
    tau_eval = np.linspace(0.0, T, K + 1)

    sol = solve_ivp(
        riccati_ode,
        [0.0, T],
        p0,
        t_eval=tau_eval,
        method="RK45",
        rtol=1e-6,
        atol=1e-8,
        max_step=dt,
    )

    # sol.y has shape (dx*dx, K+1); columns correspond to tau_eval
    P_back = sol.y.T.reshape(K + 1, dx, dx)  # P_back[0]=P(T), P_back[-1]=P(0)

    # Reverse to forward time: P_fwd[0] = P(t=0), P_fwd[-1] = P(T)
    P_fwd = P_back[::-1].copy()

    # Symmetrise and clip negative eigenvalues to 0 for numerical safety
    for k in range(K + 1):
        P_fwd[k] = 0.5 * (P_fwd[k] + P_fwd[k].T)
        evals, evecs = np.linalg.eigh(P_fwd[k])
        evals = np.clip(evals, 0.0, None)
        P_fwd[k] = evecs @ np.diag(evals) @ evecs.T

    return P_fwd


def compute_optimal_gain(
    P: np.ndarray,
    G: np.ndarray,
    R: np.ndarray,
) -> np.ndarray:
    """Compute time-varying optimal gain K*(t) = R^{-1} G^T P(t).

    Parameters
    ----------
    P : (K+1, dx, dx) Riccati solution.
    G : (dx, du) control loading.
    R : (du, du) control cost.

    Returns
    -------
    K_opt : (K+1, du, dx) optimal gains.
    """
    R_inv = np.linalg.inv(R)
    K_opt = np.einsum("ij,kjl->kil", R_inv @ G.T, P)  # (K+1, du, dx)
    return K_opt
