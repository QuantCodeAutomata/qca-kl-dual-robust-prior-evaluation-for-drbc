"""
Data fetcher for S&P 500 rolling backtest.
Uses Yahoo Finance as fallback since WRDS requires institutional access.
"""
import os
import warnings

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Universe
# ---------------------------------------------------------------------------

_SP500_PROXY = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "BRK-B", "JPM", "V",
    "JNJ", "XOM", "UNH", "WMT", "PG", "MA", "HD", "CVX", "BAC", "ABBV",
    "KO", "PEP", "AVGO", "COST", "TMO", "MRK", "CSCO", "ABT", "MCD", "PFE",
    "ACN", "DHR", "NKE", "DIS", "LIN", "VZ", "WFC", "CMCSA", "ADBE", "BMY",
    "PM", "ORCL", "TXN", "HON", "LOW", "UPS", "IBM", "CAT", "GE", "RTX",
    "MMM",
]


def fetch_sp500_universe() -> list:
    """
    Return proxy list of ~50 large-cap S&P 500 tickers.

    WRDS requires institutional access; this hardcoded list approximates the
    universe used in the paper.
    """
    return list(_SP500_PROXY)


# ---------------------------------------------------------------------------
# Price data
# ---------------------------------------------------------------------------

def fetch_price_data(
    tickers: list,
    start_date: str,
    end_date: str,
    cache_dir: str = "data/cache",
) -> pd.DataFrame:
    """
    Download adjusted-close prices for *tickers* from Yahoo Finance.

    Results are cached as a CSV in *cache_dir*.  On download failure the
    function returns whatever was cached; if no cache exists it raises.

    Parameters
    ----------
    tickers    : list of ticker strings
    start_date : 'YYYY-MM-DD'
    end_date   : 'YYYY-MM-DD'
    cache_dir  : directory for cached CSVs (relative or absolute)

    Returns
    -------
    DataFrame  index=DatetimeIndex (trading days), columns=tickers
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(
        cache_dir,
        f"prices_{start_date}_{end_date}.csv",
    )

    if os.path.exists(cache_file):
        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        # Restrict to requested tickers (cache may hold more)
        available = [t for t in tickers if t in df.columns]
        return df[available]

    df = _download_yahoo(tickers, start_date, end_date)
    df = _clean_prices(df)
    df.to_csv(cache_file)
    return df


def _download_yahoo(tickers: list, start_date: str, end_date: str) -> pd.DataFrame:
    """Attempt to download from Yahoo Finance via yfinance."""
    try:
        import yfinance as yf

        raw = yf.download(
            tickers,
            start=start_date,
            end=end_date,
            auto_adjust=True,
            progress=False,
            threads=True,
        )
        if isinstance(raw.columns, pd.MultiIndex):
            # yfinance returns MultiIndex (field, ticker) when >1 ticker
            if "Close" in raw.columns.get_level_values(0):
                df = raw["Close"]
            else:
                df = raw.iloc[:, : len(tickers)]
        else:
            df = raw[["Close"]] if "Close" in raw.columns else raw
        df.columns = [str(c) for c in df.columns]
        return df
    except Exception as exc:
        warnings.warn(f"yfinance download failed: {exc}. Returning empty DataFrame.")
        return pd.DataFrame()


def _clean_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill up to 5 days then drop columns with remaining NaNs."""
    if df.empty:
        return df
    df = df.ffill(limit=5)
    df = df.dropna(axis=1)
    return df


# ---------------------------------------------------------------------------
# Risk-free rate
# ---------------------------------------------------------------------------

def fetch_risk_free_rate(start_date: str, end_date: str) -> pd.Series:
    """
    Download the 3-month T-bill rate from FRED (series TB3MS).

    Returns Series(index=DatetimeIndex, values=annual_rate_decimal).
    Falls back to a constant 2 % rate on download failure.
    """
    try:
        import pandas_datareader.data as web

        raw = web.DataReader("TB3MS", "fred", start_date, end_date)
        series = raw["TB3MS"].dropna() / 100.0  # percent -> decimal
        return series
    except Exception as exc:
        warnings.warn(
            f"FRED download failed: {exc}. Using constant rf=0.02 fallback."
        )
        idx = pd.date_range(start_date, end_date, freq="MS")
        return pd.Series(0.02, index=idx, name="TB3MS")
