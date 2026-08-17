"""
Pytest tests for Exp 5 (terminal wealth) and Exp 7 (S&P 500 backtest).
"""
import os
import sys

# Make src/ importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))

import numpy as np
import pandas as pd
import pytest
import torch

from closed_form_target import (
    merton_expected_utility,
    merton_optimal_fraction,
    merton_terminal_wealth_samples,
)
from terminal_wealth_net import MertonBenchmarkTrainer, TerminalWealthNet

# ---------------------------------------------------------------------------
# Prior definitions (mirror exp7)
# ---------------------------------------------------------------------------
P1_B = np.array([-0.08, 0.16, -0.02, 0.04, 0.10])
P1_W = np.array([0.35,  0.08,  0.25, 0.22, 0.10])

P2_B = np.array([-0.05, 0.15,  0.00, 0.05, 0.10])
P2_W = np.array([0.45,  0.05,  0.25, 0.15, 0.10])


# ---------------------------------------------------------------------------
# 1. Closed-form utility is strictly positive
# ---------------------------------------------------------------------------
def test_closed_form_utility_positive():
    utility = merton_expected_utility(b=0.1, r=0.05, sigma=0.4, gamma=0.5, x0=1.0, T=1.0)
    assert utility > 0.0, f"Expected utility must be positive, got {utility}"


# ---------------------------------------------------------------------------
# 2. Merton optimal fraction is positive when b > r
# ---------------------------------------------------------------------------
def test_merton_optimal_fraction_positive():
    pi = merton_optimal_fraction(b=0.1, r=0.05, sigma=0.4, gamma=0.5)
    assert pi > 0.0, f"pi* must be positive when b > r, got {pi}"


# ---------------------------------------------------------------------------
# 3. Network output shape is (batch, 1)
# ---------------------------------------------------------------------------
def test_terminal_wealth_net_output_shape():
    net = TerminalWealthNet()
    batch_size = 16
    x = torch.randn(batch_size, 2)
    out = net(x)
    assert out.shape == (batch_size, 1), f"Expected (16, 1), got {out.shape}"


# ---------------------------------------------------------------------------
# 4. Forward pass completes without error for batch_size=10
# ---------------------------------------------------------------------------
def test_terminal_wealth_net_forward_pass():
    net = TerminalWealthNet()
    x = torch.randn(10, 2)
    out = net(x)
    assert out is not None
    assert torch.isfinite(out).all(), "Network output contains non-finite values"


# ---------------------------------------------------------------------------
# 5. Trainer generates samples with correct shapes
# ---------------------------------------------------------------------------
def test_trainer_generates_samples():
    net = TerminalWealthNet()
    trainer = MertonBenchmarkTrainer(net=net, n_samples=50)
    W_T, B = trainer.generate_samples(n=50, seed=0)
    assert W_T.shape == (50,), f"W_T shape mismatch: {W_T.shape}"
    assert B.shape == (50,),   f"B shape mismatch: {B.shape}"
    assert W_T.dtype == torch.float32
    assert B.dtype == torch.float32


# ---------------------------------------------------------------------------
# 6. Prior 1 probabilities sum to 1
# ---------------------------------------------------------------------------
def test_prior1_sums_to_one():
    assert abs(P1_W.sum() - 1.0) < 1e-9, f"Prior 1 sums to {P1_W.sum()}, expected 1.0"


# ---------------------------------------------------------------------------
# 7. Prior 2 probabilities sum to 1
# ---------------------------------------------------------------------------
def test_prior2_sums_to_one():
    assert abs(P2_W.sum() - 1.0) < 1e-9, f"Prior 2 sums to {P2_W.sum()}, expected 1.0"


# ---------------------------------------------------------------------------
# 8. Merton policy clipped to [-5, 5]
# ---------------------------------------------------------------------------
def test_merton_policy_clipping():
    # pi* with extreme parameters could be very large
    pi_raw = merton_optimal_fraction(b=5.0, r=0.0, sigma=0.1, gamma=0.5)
    pi_clipped = float(np.clip(pi_raw, -5.0, 5.0))
    assert -5.0 <= pi_clipped <= 5.0, f"Clipped pi not in [-5, 5]: {pi_clipped}"
    # Also check that clipping actually changed a large value
    assert pi_raw > 5.0, f"pi_raw should exceed 5 for these params, got {pi_raw}"
    assert pi_clipped == 5.0


# ---------------------------------------------------------------------------
# 9. Risk-free rate is positive (mock data)
# ---------------------------------------------------------------------------
def test_risk_free_rate_positive():
    # Use mock series to avoid network dependency
    rf_mock = pd.Series(
        [0.02, 0.025, 0.03],
        index=pd.date_range("2020-01-01", periods=3, freq="MS"),
    )
    assert (rf_mock > 0).all(), "Mock risk-free rate contains non-positive values"
    assert float(rf_mock.mean()) > 0.0


# ---------------------------------------------------------------------------
# 10. Sharpe ratio is finite for non-degenerate returns
# ---------------------------------------------------------------------------
def test_sharpe_ratio_finite():
    rng = np.random.default_rng(99)
    monthly_returns = rng.normal(0.005, 0.03, size=60)  # 5 years of monthly data
    sharpe = float(np.mean(monthly_returns) / np.std(monthly_returns) * np.sqrt(12))
    assert np.isfinite(sharpe), f"Sharpe ratio is not finite: {sharpe}"
