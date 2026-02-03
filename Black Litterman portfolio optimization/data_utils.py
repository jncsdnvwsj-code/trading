"""
Data utilities for loading and processing financial data.
"""
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta


def load_historical_data(tickers, start_date=None, end_date=None, period='5y'):
    """
    Load historical price data from Yahoo Finance.
    
    Parameters
    ----------
    tickers : list
        List of ticker symbols (e.g., ['AAPL', 'MSFT', 'GOOGL'])
    start_date : str, optional
        Start date in 'YYYY-MM-DD' format
    end_date : str, optional
        End date in 'YYYY-MM-DD' format
    period : str, optional
        Period to load if dates not specified ('5y', '10y', etc.)
    
    Returns
    -------
    pd.DataFrame
        DataFrame with adjusted closing prices
    """
    if start_date is None:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * int(period[0]))
    
    data = yf.download(tickers, start=start_date, end=end_date, progress=False)
    
    if len(tickers) == 1:
        prices = data[['Adj Close']]
        prices.columns = tickers
    else:
        prices = data['Adj Close']
    
    return prices.dropna()


def calculate_returns(prices, freq='daily'):
    """
    Calculate returns from prices.
    
    Parameters
    ----------
    prices : pd.DataFrame
        DataFrame with price data
    freq : str, optional
        'daily', 'weekly', 'monthly', 'yearly'
    
    Returns
    -------
    pd.DataFrame
        DataFrame with returns
    """
    returns = prices.pct_change().dropna()
    
    if freq == 'weekly':
        returns = prices.resample('W').last().pct_change().dropna()
    elif freq == 'monthly':
        returns = prices.resample('M').last().pct_change().dropna()
    elif freq == 'yearly':
        returns = prices.resample('Y').last().pct_change().dropna()
    
    return returns


def calculate_cov_matrix(returns, annualize=True):
    """
    Calculate covariance matrix from returns.
    
    Parameters
    ----------
    returns : pd.DataFrame
        DataFrame with returns
    annualize : bool
        Whether to annualize the covariance (assuming 252 trading days)
    
    Returns
    -------
    np.ndarray
        Covariance matrix
    """
    cov = returns.cov().values
    
    if annualize:
        cov = cov * 252
    
    return cov


def calculate_expected_returns(returns, annualize=True):
    """
    Calculate expected returns (mean).
    
    Parameters
    ----------
    returns : pd.DataFrame
        DataFrame with returns
    annualize : bool
        Whether to annualize the returns
    
    Returns
    -------
    np.ndarray
        Expected returns vector
    """
    exp_returns = returns.mean().values
    
    if annualize:
        exp_returns = exp_returns * 252
    
    return exp_returns


def get_asset_names(returns):
    """Get asset names from returns DataFrame."""
    return returns.columns.tolist()
