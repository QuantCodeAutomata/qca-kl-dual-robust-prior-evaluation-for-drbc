"""
lq_system.py — Linear-Quadratic control system for DRBC experiment (Exp 3).

System dynamics:
    dX_t = (A(theta) X_t + G u_t) dt + Sigma dW_t

Cost:
    J = sum_k [ X_k^T Q X_k + U_k^T R U_k ] * dt + X_K^T Q_f X_K
"""

import numpy as np
from typing import Optional, Tuple


class LQSystem:
    """Linear-Quadratic system specification."""

    def __init__(
        self,
        dx: int = 10,
        du: int = 5,
        T: float = 2.0,
        dt: float = 0.02,
        sigma_scale: float = 0.7,
        seed: int = 12345,
    ) -> None:
        self.dx = dx
        self.du = du
        self.T = T
        self.dt = dt
        self.K = int(round(T / dt))  # 100 steps
        self.sigma_scale = sigma_scale
        self.seed = seed

        # Build fixed system matrices once
        rng = np.random.default_rng(seed)
        self.Q = 3.0 * np.eye(dx)
        self.Q_f = 3.0 * np.eye(dx)
        self.R = np.eye(du)
        self.A0 = self._build_A0()
        self.basis_matrices = self._build_basis_matrices(p=dx, rng=rng)
        self.G = self._build_G(rng)
        self.Sigma = self._build_Sigma()
        self.Sigma_inv = np.linalg.inv(self.Sigma)

    # ------------------------------------------------------------------
    # System matrix builders
    # ------------------------------------------------------------------

    def _build_A0(self) -> np.ndarray:
        """Tridiagonal A0: diagonal=0.6, upper=0.15, lower=-0.1."""
        A = np.zeros((self.dx, self.dx))
        np.fill_diagonal(A, 0.6)
        for i in range(self.dx - 1):
            A[i, i + 1] = 0.15   # upper diagonal
            A[i + 1, i] = -0.1   # lower diagonal
        return A

    def _build_basis_matrices(self, p: int, rng: np.random.Generator) -> list:
        """p random dx×dx drift basis matrices."""
        mats = []
        for _ in range(p):
            M = rng.normal(0.0, 0.05, (self.dx, self.dx))
            np.fill_diagonal(M, rng.normal(0.0, 0.2, self.dx))
            mats.append(M)
        return mats

    def _build_G(self, rng: np.random.Generator) -> np.ndarray:
        """dx×du control loading matrix."""
        return rng.normal(0.0, 0.3, (self.dx, self.du))

    def _build_Sigma(self) -> np.ndarray:
        """Diffusion matrix: sigma_scale * I_dx."""
        return self.sigma_scale * np.eye(self.dx)

    # ------------------------------------------------------------------
    # Derived quantities
    # ------------------------------------------------------------------

    def compute_A(self, theta: np.ndarray) -> np.ndarray:
        """Drift matrix A(theta) = A0 + sum_i theta_i A_i."""
        A = self.A0.copy()
        for i, Ai in enumerate(self.basis_matrices):
            A = A + theta[i] * Ai
        return A

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def simulate_trajectory(
        self,
        A: np.ndarray,
        K_ctrl: np.ndarray,
        x0: np.ndarray,
        rng: np.random.Generator,
        G: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Euler-Maruyama simulation of LQ dynamics.

        Parameters
        ----------
        A : (dx, dx) drift matrix.
        K_ctrl : (du, dx) feedback gain, u_k = -K_ctrl @ x_k.
        x0 : (dx,) initial state.
        rng : random number generator.
        G : (dx, du) control loading; defaults to self.G.

        Returns
        -------
        X_traj : (K+1, dx) state trajectory.
        U_traj : (K, du) control trajectory.
        """
        if G is None:
            G = self.G
        K = self.K
        dx, du = self.dx, self.du
        sqrt_dt = np.sqrt(self.dt)

        X = np.zeros((K + 1, dx))
        U = np.zeros((K, du))
        X[0] = x0

        for k in range(K):
            u_k = -K_ctrl @ X[k]
            u_k = np.clip(u_k, -50.0, 50.0)
            dW = rng.standard_normal(dx) * sqrt_dt
            X[k + 1] = X[k] + (A @ X[k] + G @ u_k) * self.dt + self.Sigma @ dW
            U[k] = u_k

        return X, U

    def compute_trajectory_cost(self, X: np.ndarray, U: np.ndarray) -> float:
        """Running cost + terminal cost.

        Parameters
        ----------
        X : (K+1, dx) state trajectory.
        U : (K, du) control trajectory.

        Returns
        -------
        float total cost J.
        """
        K = self.K
        cost = 0.0
        for k in range(K):
            cost += (X[k] @ self.Q @ X[k] + U[k] @ self.R @ U[k]) * self.dt
        cost += X[K] @ self.Q_f @ X[K]
        return float(cost)


class LQIdentifier:
    """GLS estimator for the latent drift parameter theta."""

    def gls_estimate(
        self,
        X_traj: np.ndarray,
        U_traj: np.ndarray,
        A0: np.ndarray,
        basis_matrices: list,
        G: np.ndarray,
        Sigma: np.ndarray,
        dt: float,
        lambda_ridge: float = 1e-4,
    ) -> np.ndarray:
        """Estimate theta via GLS.

        Parameters
        ----------
        X_traj : (K+1, dx) state trajectory.
        U_traj : (K, du) control trajectory.
        A0, basis_matrices, G, Sigma : system matrices.
        dt : time step.
        lambda_ridge : ridge regularisation for ill-conditioned Gram matrix.

        Returns
        -------
        theta_hat : (p,) parameter estimate.
        """
        K, dx = U_traj.shape[0], X_traj.shape[1]
        p = len(basis_matrices)
        Sigma_inv = np.linalg.inv(Sigma)

        # Accumulate normal equations
        XtX = np.zeros((p, p))
        Xty = np.zeros(p)

        for k in range(K):
            # Corrected increment
            delta_y = X_traj[k + 1] - X_traj[k] - G @ U_traj[k] * dt
            r_k = delta_y / dt - A0 @ X_traj[k]
            # Feature matrix row: (dx, p)
            Phi_k = np.column_stack([Ai @ X_traj[k] for Ai in basis_matrices])

            # Whiten
            r_tilde = Sigma_inv @ r_k           # (dx,)
            Phi_tilde = Sigma_inv @ Phi_k       # (dx, p)

            XtX += Phi_tilde.T @ Phi_tilde
            Xty += Phi_tilde.T @ r_tilde

        # Ridge-regularised solution
        A_reg = XtX + lambda_ridge * np.eye(p)
        theta_hat = np.linalg.solve(A_reg, Xty)
        return theta_hat
