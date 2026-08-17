"""
priors.py — Prior distributions for DRBC experiments.

Provides FiniteSupportPrior and GaussianPrior, both validated on construction
and usable for sampling and log-probability evaluation.
"""

import numpy as np
from scipy.stats import multivariate_normal


class FiniteSupportPrior:
    """Discrete prior over a finite set of support points.

    Parameters
    ----------
    support : array-like of shape (K,) or (K, d)
        The support points of the prior.  For scalar parameters K 1-D values
        are accepted; for vector parameters a 2-D array of shape (K, d).
    probs : array-like of shape (K,)
        Non-negative probability weights that must sum to 1.

    Raises
    ------
    ValueError
        If ``probs`` contains negative entries, does not sum to 1 (within 1e-8),
        or the number of support points does not match the length of ``probs``.
    """

    def __init__(self, support, probs):
        support = np.asarray(support, dtype=float)
        probs = np.asarray(probs, dtype=float)

        if probs.ndim != 1:
            raise ValueError("probs must be a 1-D array.")
        if support.shape[0] != probs.shape[0]:
            raise ValueError(
                f"Number of support points ({support.shape[0]}) must match "
                f"len(probs) ({probs.shape[0]})."
            )
        if np.any(probs < 0):
            raise ValueError("All entries of probs must be non-negative.")
        if not np.isclose(probs.sum(), 1.0, atol=1e-8):
            raise ValueError(
                f"probs must sum to 1, got {probs.sum():.10f}."
            )

        self._support = support
        self._probs = probs

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def support(self):
        """numpy.ndarray: Support points of the prior."""
        return self._support

    @property
    def probs(self):
        """numpy.ndarray: Probability weights of the prior."""
        return self._probs

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def sample(self, n, rng=None):
        """Draw *n* i.i.d. samples from the prior.

        Parameters
        ----------
        n : int
            Number of samples to draw.
        rng : numpy.random.Generator, optional
            Random number generator.  Defaults to ``numpy.random.default_rng()``.

        Returns
        -------
        numpy.ndarray
            Shape ``(n,)`` for scalar support or ``(n, d)`` for vector support.
        """
        if rng is None:
            rng = np.random.default_rng()
        indices = rng.choice(len(self._probs), size=n, p=self._probs)
        return self._support[indices]

    def log_prob(self, b):
        """Log-probability of a support point *b*.

        Parameters
        ----------
        b : scalar or array-like
            A single support point value.  Must match exactly one entry in
            ``self.support`` (compared element-wise).

        Returns
        -------
        float
            ``log(P(B = b))``.  Returns ``-inf`` if *b* is not in the support.
        """
        b = np.asarray(b, dtype=float)
        for i, s in enumerate(self._support):
            if np.allclose(b, s):
                p = self._probs[i]
                return float(np.log(p)) if p > 0 else -np.inf
        return -np.inf


class GaussianPrior:
    """Multivariate Gaussian prior N(mean, cov).

    Parameters
    ----------
    mean : array-like of shape (d,)
        Mean vector.
    cov : array-like of shape (d, d)
        Positive-semidefinite covariance matrix.

    Raises
    ------
    ValueError
        If dimensions of *mean* and *cov* are inconsistent, or if *cov* is not
        square.
    """

    def __init__(self, mean, cov):
        mean = np.asarray(mean, dtype=float)
        cov = np.asarray(cov, dtype=float)

        if mean.ndim != 1:
            raise ValueError("mean must be a 1-D array.")
        if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
            raise ValueError("cov must be a square 2-D array.")
        if cov.shape[0] != mean.shape[0]:
            raise ValueError(
                f"cov shape {cov.shape} is incompatible with mean of length {mean.shape[0]}."
            )

        self._mean = mean
        self._cov = cov
        self._rv = multivariate_normal(mean=mean, cov=cov, allow_singular=True)

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def sample(self, n, rng=None):
        """Draw *n* i.i.d. samples from N(mean, cov).

        Parameters
        ----------
        n : int
            Number of samples.
        rng : numpy.random.Generator, optional
            Random number generator.  When provided its ``multivariate_normal``
            method is used; otherwise scipy's frozen distribution is used.

        Returns
        -------
        numpy.ndarray of shape (n, d)
        """
        if rng is None:
            rng = np.random.default_rng()
        return rng.multivariate_normal(self._mean, self._cov, size=n)

    def log_prob(self, b):
        """Log-probability density at *b*.

        Parameters
        ----------
        b : array-like of shape (d,)

        Returns
        -------
        float
        """
        return float(self._rv.logpdf(np.asarray(b, dtype=float)))
