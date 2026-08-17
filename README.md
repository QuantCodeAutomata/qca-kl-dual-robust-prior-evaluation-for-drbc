# KL-Dual Robust Prior Evaluation for DRBC

A Python implementation of Distributionally Robust Bayesian Control (DRBC) experiments, validating the KL-dual robust prior framework and its applications to portfolio optimization and linear-quadratic control.

## Overview

This repository implements and validates seven experiments from the DRBC paper:

| Experiment | Description |
|-----------|-------------|
| **exp_1** | KL-Dual Robust Prior Evaluation — validates 1D dual optimization |
| **exp_2** | rMLMC Policy-Evaluation Estimator — verifies n⁻¹/² convergence rate |
| **exp_3** | Synthetic LQ DRBC — robustness under prior misspecification |
| **exp_4** | Merton DRBC vs DRC — ambiguity radius ablation |
| **exp_5** | Terminal-Wealth Neural Learner — closed-form benchmark validation |
| **exp_6** | Finite-Support DRBC Merton Learning — zero-order simplex optimization |
| **exp_7** | S&P 500 Rolling Backtest — empirical portfolio comparison |

## Repository Structure

```
src/                   Core library modules
  priors.py             Finite-support and Gaussian prior classes
  kl_dual.py            KL-dual robust objective (dual form, logsumexp-stable)
  primal_validation.py  Primal KL-ball optimization for validation
  value_estimators.py   Model-agnostic Z_pi(b) estimators
  rmlmc.py              (in exp2_rmlmc.py) rMLMC unbiased estimator
  lq_system.py          Linear-Quadratic system + GLS identifier
  riccati.py            Backward matrix Riccati ODE solver
  drbc_lq_policy.py     DRBC neural gain policy for LQ control
  closed_form_target.py Merton closed-form benchmark (Appendix H.4.1)
  terminal_wealth_net.py Neural terminal-wealth learner (Theorem 4.1)

data/
  data_fetcher.py       Yahoo Finance / FRED data fetching with cache

exp/
  exp1_kl_dual.py       KL dual vs primal validation experiment
  exp2_rmlmc.py         rMLMC n^{-1/2} scaling rate experiment
  exp3_lq_drbc.py       LQ system oracle/plug-in/DRBC comparison
  exp4_merton_ablation.py Merton DRBC vs DRC ablation over delta
  exp5_terminal_wealth.py Neural learner benchmark experiment
  exp6_finite_support_drbc.py Finite-support DRBC Merton optimization
  exp7_sp500_backtest.py Rolling backtest on S&P 500 proxy universe

tests/
  test_exp1_exp2_exp6.py Tests for KL dual, rMLMC, finite-support modules
  test_exp3_exp4.py      Tests for LQ system and Merton ablation
  test_exp5_exp7.py      Tests for neural learner and backtest

results/
  RESULTS.md            Summary of all experiment findings
  *.png                 Figures
  *.csv                 Numerical results
```

## Key Theory

### KL-Dual Representation (exp_1)
The robust prior evaluation objective:
```
R_delta(pi) = inf_{nu: KL(nu||mu) <= delta} integral Z_pi(b) nu(db)
            = sup_{lambda >= 0} { -lambda*delta - lambda*log(M_pi(lambda)) }
```
where `M_pi(lambda) = E_{B~mu}[exp(-Z_pi(B)/lambda)]`.

### rMLMC Estimator (exp_2)
Unbiased estimation of `M_pi(lambda)` using randomized multilevel Monte Carlo, achieving O(n^{-1/2}) rate.

### Merton Policies
- **Bayesian**: `pi = sum_i p_i * (b_i - r) / (sigma^2 * (1-alpha))`
- **DRBC**: `pi = sum_i q*_i * (b_i - r) / (sigma^2 * (1-alpha))` where `q*` solves the KL-robust problem
- **DRC**: Conservative adjustment of nominal drift

## Installation

```bash
pip install numpy scipy pandas matplotlib torch quantstats
```

## Running Experiments

```bash
# Run all experiments
python exp/exp1_kl_dual.py
python exp/exp2_rmlmc.py
python exp/exp3_lq_drbc.py
python exp/exp4_merton_ablation.py
python exp/exp5_terminal_wealth.py
python exp/exp6_finite_support_drbc.py
python exp/exp7_sp500_backtest.py

# Run all tests
python -m pytest tests/ -v
```

## Key Results

See `results/RESULTS.md` for a full summary of findings.

## Notes on Parameter Assumptions

Several parameters are not fully specified in the paper text and have been documented as assumptions in the code:
- Merton ablation: `B0=0.3`, `kappa=pi`, consistent with a one-period cosine drift cycle
- LQ system: `G` matrix uses `N(0, 0.3^2)` entries, consistent with exploration policy scaling
- S&P 500 backtest: Uses Yahoo Finance data (2019-2023) with 50-ticker proxy universe instead of WRDS (institutional access required)
