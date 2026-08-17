# DRBC Repository — Agent Memory

## Project Overview
Distributionally Robust Bayesian Control (DRBC) experiments implemented in Python.
All experiments live in `exp/`, core library modules in `src/`, data utilities in `data/`.

## Directory Layout
```
src/                  Core library modules (pure Python / NumPy / PyTorch)
  closed_form_target.py   Merton closed-form benchmark (Appendix H.4.1)
  terminal_wealth_net.py  Neural terminal-wealth learner + trainer (Exp 5)
  kl_dual.py              KL-divergence dual utilities (Exp 1)
  primal_validation.py    Primal / MC validation helpers (Exp 2)
  priors.py               Prior support / weight definitions
  value_estimators.py     Value-function estimators

data/
  data_fetcher.py     Yahoo Finance / FRED data helpers + S&P 500 proxy universe

exp/
  exp5_terminal_wealth.py   Exp 5: neural Merton benchmark vs closed-form
  exp7_sp500_backtest.py    Exp 7: S&P 500 rolling backtest (2019-2023)
  exp1_kl_dual.py
  exp2_rmlmc.py
  exp6_finite_support_drbc.py

tests/
  test_exp5_exp7.py   10 pytest tests for Exp 5 and Exp 7
  test_exp1_exp2_exp6.py

results/              Output artefacts (CSV, PNG); gitignored where large
```

## Key Parameters (paper)
| Symbol | Value | Meaning |
|--------|-------|---------|
| sigma  | 0.4   | volatility |
| sigma_0| 2.0   | prior std |
| gamma / alpha | 0.5 | CRRA exponent |
| T      | 1.0   | horizon |
| x0     | 1.0   | initial wealth |
| r_vals | [0.05, 0.10] | risk-free rates |
| b_vals | [0.1, 0.3]   | drift scenarios |
| delta  | 0.15  | robustness radius |

## Import Convention
All `exp/` scripts must prepend `src/` (and `data/` where needed) to `sys.path`:
```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'data'))
```
`src/terminal_wealth_net.py` imports from `closed_form_target` without a package prefix
(both live in the same `src/` directory; the exp scripts add `src/` to `sys.path`).

## Matplotlib Notes
- Use `matplotlib.use("Agg")` before importing `pyplot` in non-interactive scripts.
- `ax.boxplot()` keyword: use `tick_labels=` (not `labels=`) in matplotlib ≥ 3.9.

## TerminalWealthNet
- Architecture: Input(2) → [Linear→LeakyReLU(0.01)] × 3 hidden [128,256,256] → Linear(1)
- Input columns: [W_T, B] — Brownian terminal value and drift parameter.
- Training: MSE between h_theta(W_T, B) and analytical X_T* from `merton_terminal_wealth_samples`.
- LR schedule: 10-epoch warmup → 140-epoch stable → linear decay to 3e-4.

## MertonBenchmarkTrainer
- `generate_samples(n, seed)` fixes B = b1 for all samples (Q^{b1} measure).
- `compute_loss` accepts `measure='b1'|'b2'` to select the drift for the target.
- `evaluate_utility` estimates E[h^gamma / gamma] via MC, clamping h >= 1e-8.

## Data Fetcher
- `fetch_sp500_universe()` returns a hardcoded 50-ticker proxy list.
- `fetch_price_data` caches to `data/cache/prices_{start}_{end}.csv`; falls back to
  synthetic GBM paths when Yahoo Finance is unavailable.
- `fetch_risk_free_rate` downloads FRED TB3MS; falls back to 2 % constant on error.

## Backtest (Exp 7)
- Period: 2019-01-01 to 2023-12-31 (fetch from 2018-01-01 for 252-day warm-up).
- Methods: Merton, Bayes_P1/P2, DRC_P1/P2, DRBC_P1/P2.
- DRBC weights: projected gradient descent on simplex (100 iter, lr=1e-4),
  minimising E_q[Z_i] where Z_i = (b_i-r)^2 / (2σ²(1-α)) · T.
- DRC uses mu_nominal = E_{P}[b] — fixed prior mean, not stock-specific estimate.
- All pi values clipped to [-5, 5].

## Testing
Run: `python -m pytest tests/ -v`
All tests use `sys.path` manipulation — no package install required.
Test 9 (`test_risk_free_rate_positive`) uses a mock pd.Series — no network call.

## Module Notes: src/priors.py
- `FiniteSupportPrior(support, probs)`: validates non-negative probs summing to 1 and
  matching lengths.  `log_prob(b)` uses `np.allclose` for support lookup.
- `GaussianPrior(mean, cov)`: wraps `scipy.stats.multivariate_normal` with
  `allow_singular=True`; sampling via `rng.multivariate_normal`.

## Module Notes: src/kl_dual.py
- All exponentials use `scipy.special.logsumexp` for numerical stability.
- `optimize_dual` combines a 200-point log-spaced grid search with
  `scipy.optimize.minimize_scalar` (bounded) for fine refinement.
- At delta=0, the dual supremum is attained as lambda→∞ and equals E_mu[Z]
  (Bayesian value); numerically the grid max at lambda_max≈1e4 gives <1e-3 gap.

## Module Notes: src/primal_validation.py
- `primal_robust_value` uses SLSQP with both an equality constraint (sum=1) and
  an inequality constraint (KL(q||p) <= delta).  Bounds are [1e-12, 1] per component.
- SLSQP may emit RuntimeWarning about values clipped to bounds — this is benign;
  the returned q_opt is re-normalised.

## Module Notes: src/value_estimators.py
- `estimate_Z_merton_finite`: Euler-Maruyama with 100 steps, vectorised over
  n_inner paths.  Wealth clipped to 1e-300 to prevent log(0) in power utility.
- Merton optimal fraction: pi*(b) = (b-r)/(sigma^2*(1-alpha)).

## Experiment Notes: exp/exp6_finite_support_drbc.py
- `q_optimizer` uses projected gradient descent on the simplex via the Duchi et al.
  O(K log K) projection algorithm.
- STABILITY: the KL Hessian eigenvalue at q≈p is lambda/min(p_i).  For stable
  convergence, use lr < 2*min(p)/lambda.  In main(), lr = 0.04/lambda_dual.
- The analytic optimum is q*_i ∝ p_i * exp(-V(b_i)/lambda), which the gradient
  descent recovers numerically.
- Finite-difference step h=1e-6 for gradient computation; central differences.
