"""
Exp 7: Real-data S&P 500 rolling backtest for DRBC portfolio allocation.
Uses Yahoo Finance data (WRDS not available) for 2019-01-01 to 2023-12-31 window.
"""
import os
import sys
import warnings

# Ensure src/ and data/ are importable
_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "data"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_fetcher import fetch_price_data, fetch_risk_free_rate, fetch_sp500_universe

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
START_DATE = "2018-01-01"   # extra year for warm-up window
END_DATE = "2023-12-31"
BACKTEST_START = "2019-01-01"
EST_WINDOW = 252            # trading days
SIGMA_ASSUMED = 0.4
ALPHA = 0.5
DELTA = 0.15

# Prior 1
P1_B = np.array([-0.08, 0.16, -0.02, 0.04, 0.10])
P1_W = np.array([0.35,  0.08,  0.25, 0.22, 0.10])

# Prior 2
P2_B = np.array([-0.05, 0.15,  0.00, 0.05, 0.10])
P2_W = np.array([0.45,  0.05,  0.25, 0.15, 0.10])

RESULTS_DIR = os.path.join(_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

METHODS = ["Merton", "Bayes_P1", "Bayes_P2", "DRC_P1", "DRC_P2", "DRBC_P1", "DRBC_P2"]


# ---------------------------------------------------------------------------
# Utility: simplex projection
# ---------------------------------------------------------------------------

def _project_simplex(v: np.ndarray) -> np.ndarray:
    """Project v onto the probability simplex."""
    n = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho_arr = np.where(u * np.arange(1, n + 1) > (cssv - 1))[0]
    rho = int(rho_arr[-1])
    theta = (cssv[rho] - 1.0) / (rho + 1)
    return np.maximum(v - theta, 0.0)


# ---------------------------------------------------------------------------
# DRBC: worst-case distribution on support via simplex gradient descent
# ---------------------------------------------------------------------------

def _compute_drbc_weights(
    b_support: np.ndarray,
    probs: np.ndarray,
    r: float,
    sigma: float,
    alpha: float,
    delta: float,
) -> np.ndarray:
    """KL-dual adversarial reweighting for DRBC.

    Solves  min_{q in simplex, KL(q||probs) <= delta}  E_q[b_i]
    using the closed-form dual:  q*_i ∝ probs_i * exp(-b_i / lambda*)
    where lambda* satisfies  KL(q*||probs) = delta.

    For a long Merton position (positive allocation), lower drift is worse;
    the adversary therefore concentrates mass on the most negative drift states.
    """
    from scipy.optimize import brentq
    from scipy.special import logsumexp

    if delta < 1e-12:
        return probs.copy().astype(float)

    log_p = np.log(np.maximum(probs, 1e-300))

    def _q_from_lam(lam: float) -> np.ndarray:
        log_q_unnorm = log_p - b_support / lam
        log_q_unnorm -= logsumexp(log_q_unnorm)
        return np.exp(log_q_unnorm)

    def _kl(lam: float) -> float:
        q = _q_from_lam(lam)
        mask = q > 1e-300
        return float(np.sum(q[mask] * np.log(q[mask] / probs[mask]))) - delta

    # At lambda→∞ the distribution q→probs and KL→0.
    # At lambda→0+ the distribution concentrates on min(b) and KL→large.
    try:
        lam_star = brentq(_kl, 1e-8, 1e6, xtol=1e-10, maxiter=200)
        return _q_from_lam(lam_star)
    except Exception:
        # Fallback: concentrate entirely on the worst (most negative) drift
        q = np.zeros_like(probs, dtype=float)
        q[np.argmin(b_support)] = 1.0
        return q


# ---------------------------------------------------------------------------
# Policy functions
# ---------------------------------------------------------------------------

def _merton_pi(mu_est: float, r: float, sigma: float, alpha: float) -> float:
    denom = sigma**2 * (1.0 - alpha)
    return (mu_est - r) / denom if denom != 0 else 0.0


def _bayesian_posterior(
    b_support: np.ndarray,
    probs: np.ndarray,
    mu_ann: float,
    sigma: float,
    T: float = 1.0,
) -> np.ndarray:
    """Compute Bayesian posterior weights for the finite-support prior.

    Under a GBM with drift b_i, the observed annualised log-return y_obs = mu_ann
    follows N(b_i - 0.5*sigma^2, sigma^2/T).  The posterior is:
        q_i ∝ p_i * likelihood(y_obs | b_i)
    """
    y_obs = mu_ann
    # log-likelihood for each support point
    log_likes = -0.5 * ((y_obs - (b_support - 0.5 * sigma ** 2)) ** 2) / (sigma ** 2 / T)
    log_likes -= log_likes.max()          # numerical stability
    post = probs * np.exp(log_likes)
    post_sum = post.sum()
    if post_sum < 1e-300:
        return probs.copy()               # degenerate: fall back to prior
    return post / post_sum


def _bayes_pi(b_support, probs, mu_ann: float, r: float, sigma: float, alpha: float) -> float:
    """Bayesian Merton: posterior-mean drift → Merton fraction."""
    post = _bayesian_posterior(b_support, probs, mu_ann, sigma)
    mu_bayes = float(np.dot(post, b_support))
    return _merton_pi(mu_bayes, r, sigma, alpha)


def _drc_pi(b_support, probs, mu_ann: float, r: float, sigma: float, alpha: float, delta: float) -> float:
    """DRC: Bayesian posterior mean with an aggressive worst-case drift penalty.

    The DRC penalty sqrt(2*delta)*sigma is the continuous-time KL-ball drift
    shift derived from the classical distributionally robust control theory.
    Unlike DRBC, DRC robustifies the full dynamics at every instant, resulting
    in a larger penalty and potentially negative (short) allocations.
    """
    post = _bayesian_posterior(b_support, probs, mu_ann, sigma)
    mu_nom = float(np.dot(post, b_support))
    drc_penalty = np.sqrt(2.0 * delta) * sigma       # continuous-time DRC shift
    denom = sigma ** 2 * (1.0 - alpha)
    return (mu_nom - r - drc_penalty) / denom if denom != 0 else 0.0  # allow negatives (short)


def _drbc_pi(b_support, probs, mu_ann: float, r: float, sigma: float, alpha: float, delta: float) -> float:
    """DRBC: adversarially reweight Bayesian posterior under KL ball, then apply Merton."""
    post = _bayesian_posterior(b_support, probs, mu_ann, sigma)
    q_star = _compute_drbc_weights(b_support, post, r, sigma, alpha, delta)
    mu_drbc = float(np.dot(q_star, b_support))
    return _merton_pi(mu_drbc, r, sigma, alpha)


def _clip(pi: float, lo: float = -5.0, hi: float = 5.0) -> float:
    return float(np.clip(pi, lo, hi))


# ---------------------------------------------------------------------------
# Rolling backtest core
# ---------------------------------------------------------------------------

def _run_backtest(prices: pd.DataFrame, rf_series: pd.Series) -> pd.DataFrame:
    """
    Return DataFrame(index=month_end, columns=method_ticker) of monthly returns.
    """
    # Compute daily log-returns
    log_ret = np.log(prices / prices.shift(1)).dropna()
    trading_days = log_ret.index

    # Identify month-end rebalance dates on or after BACKTEST_START
    rebal_dates = (
        log_ret.loc[BACKTEST_START:]
        .resample("ME")
        .last()
        .index
    )

    records = []  # list of dicts {date, method, ticker, monthly_return}

    for rebal_dt in rebal_dates:
        # Estimation window: last EST_WINDOW days before rebal_dt
        past = trading_days[trading_days <= rebal_dt]
        if len(past) < EST_WINDOW + 1:
            continue
        window_idx = past[-(EST_WINDOW + 1) :]
        window_ret = log_ret.loc[window_idx[1:]]  # EST_WINDOW rows

        # Risk-free rate for this period (annualised decimal)
        rf_window = rf_series.loc[:rebal_dt]
        r_annual = float(rf_window.iloc[-1]) if not rf_window.empty else 0.02
        r_monthly = r_annual / 12.0

        # Identify next-month return dates
        future = trading_days[trading_days > rebal_dt]
        next_month_end = rebal_dt + pd.offsets.MonthEnd(1)
        next_ret_days = future[future <= next_month_end]

        if len(next_ret_days) == 0:
            continue
        next_log_ret = log_ret.loc[next_ret_days]

        for ticker in prices.columns:
            # Estimate annualised parameters for this stock
            col = window_ret[ticker].dropna()
            if len(col) < 20:
                continue
            mu_daily = float(col.mean())
            sigma_daily = float(col.std())
            if sigma_daily < 1e-8:
                continue
            mu_ann = mu_daily * 252
            sigma_ann = sigma_daily * np.sqrt(252)
            sigma_used = max(sigma_ann, 0.05)  # floor

            # Next-month compounded return (simple approx from log-returns)
            next_monthly_log = float(next_log_ret[ticker].dropna().sum())
            next_monthly_simple = np.expm1(next_monthly_log)

            # Compute pi for each method
            # Bayesian / DRC / DRBC all take mu_ann to update the prior via
            # Bayes' theorem before computing the robust allocation.
            pi_vals = {
                "Merton":   _merton_pi(mu_ann, r_annual, sigma_used, ALPHA),
                "Bayes_P1": _bayes_pi(P1_B, P1_W, mu_ann, r_annual, sigma_used, ALPHA),
                "Bayes_P2": _bayes_pi(P2_B, P2_W, mu_ann, r_annual, sigma_used, ALPHA),
                "DRC_P1":   _drc_pi(P1_B, P1_W, mu_ann, r_annual, sigma_used, ALPHA, DELTA),
                "DRC_P2":   _drc_pi(P2_B, P2_W, mu_ann, r_annual, sigma_used, ALPHA, DELTA),
                "DRBC_P1":  _drbc_pi(P1_B, P1_W, mu_ann, r_annual, sigma_used, ALPHA, DELTA),
                "DRBC_P2":  _drbc_pi(P2_B, P2_W, mu_ann, r_annual, sigma_used, ALPHA, DELTA),
            }

            for method, pi_raw in pi_vals.items():
                pi = _clip(pi_raw)
                # Excess monthly return: pi*(R_asset - R_rf) — used for Sharpe
                # Equivalent to total return minus risk-free rate, which is the
                # standard numerator/denominator for the annualised Sharpe ratio.
                excess_ret = pi * (next_monthly_simple - r_monthly)
                records.append(
                    {
                        "date": rebal_dt,
                        "method": method,
                        "ticker": ticker,
                        "monthly_return": excess_ret,
                    }
                )

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Sharpe ratio helper
# ---------------------------------------------------------------------------

def _sharpe(returns: np.ndarray) -> float:
    if len(returns) < 2 or np.std(returns) < 1e-10:
        return 0.0
    return float(np.mean(returns) / np.std(returns) * np.sqrt(12))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Exp 7 – S&P 500 DRBC Rolling Backtest (2019–2023)")
    print("=" * 60)

    # ---- fetch universe ----
    tickers = fetch_sp500_universe()
    print(f"\nUniverse: {len(tickers)} tickers")

    # ---- price data ----
    cache_dir = os.path.join(_ROOT, "data", "cache")
    print("Fetching price data …")
    prices = fetch_price_data(tickers, START_DATE, END_DATE, cache_dir=cache_dir)
    if prices.empty:
        warnings.warn("Price data unavailable; generating synthetic data.")
        rng = np.random.default_rng(42)
        dates = pd.date_range(START_DATE, END_DATE, freq="B")
        prices = pd.DataFrame(
            100 * np.cumprod(1 + rng.normal(0.0004, 0.012, size=(len(dates), len(tickers))), axis=0),
            index=dates,
            columns=tickers,
        )
    print(f"Prices shape: {prices.shape}")

    # ---- risk-free rate ----
    print("Fetching risk-free rate …")
    rf = fetch_risk_free_rate(START_DATE, END_DATE)

    # ---- rolling backtest ----
    print("Running rolling backtest …")
    records_df = _run_backtest(prices, rf)

    if records_df.empty:
        warnings.warn("No backtest records generated; check data availability.")
        sharpe_df = pd.DataFrame(
            {"method": METHODS, "mean_sharpe": [0.0] * len(METHODS), "std_sharpe": [0.0] * len(METHODS)}
        )
    else:
        # ---- per-ticker Sharpe ratios ----
        sharpe_rows = []
        for method in METHODS:
            sub = records_df[records_df.method == method]
            ticker_sharpes = []
            for ticker, grp in sub.groupby("ticker"):
                s = _sharpe(grp["monthly_return"].values)
                if np.isfinite(s):
                    ticker_sharpes.append(s)
            sharpe_rows.append(
                {
                    "method": method,
                    "mean_sharpe": float(np.mean(ticker_sharpes)) if ticker_sharpes else 0.0,
                    "std_sharpe": float(np.std(ticker_sharpes)) if ticker_sharpes else 0.0,
                    "_all": ticker_sharpes,
                }
            )
        sharpe_df = pd.DataFrame(sharpe_rows)

    # ---- print table ----
    print("\nSharpe Ratio Summary (cross-sectional mean ± std):")
    print(f"{'Method':<12} {'Mean Sharpe':>14} {'Std Sharpe':>12}")
    print("-" * 42)
    for _, row in sharpe_df.iterrows():
        print(f"{row['method']:<12} {row['mean_sharpe']:>14.4f} {row['std_sharpe']:>12.4f}")

    # ---- boxplot ----
    fig, ax = plt.subplots(figsize=(10, 5))
    if "_all" in sharpe_df.columns:
        data_to_plot = [row["_all"] if row["_all"] else [0.0] for _, row in sharpe_df.iterrows()]
        ax.boxplot(data_to_plot, tick_labels=sharpe_df["method"].tolist(), patch_artist=True)
    else:
        ax.bar(sharpe_df["method"], sharpe_df["mean_sharpe"])
    ax.set_ylabel("Annualised Sharpe Ratio")
    ax.set_title("S&P 500 Backtest – Sharpe Ratio by Method (2019–2023)")
    ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    boxplot_path = os.path.join(RESULTS_DIR, "exp7_sharpe_boxplot.png")
    plt.savefig(boxplot_path, dpi=120)
    plt.close(fig)
    print(f"\nSaved Sharpe boxplot → {boxplot_path}")

    # ---- save CSV ----
    save_df = sharpe_df.drop(columns=["_all"], errors="ignore")
    csv_path = os.path.join(RESULTS_DIR, "exp7_results.csv")
    save_df.to_csv(csv_path, index=False)
    print(f"Saved results CSV → {csv_path}")
    print("\nExp 7 complete.\n")


if __name__ == "__main__":
    main()
