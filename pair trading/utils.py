"""
Utility functions for pairs trading system.
"""
import numpy as np
import pandas as pd
from typing import Tuple, Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def normalize_prices(prices: pd.Series) -> pd.Series:
    """Normalize price series to start at 1."""
    return prices / prices.iloc[0]


def calculate_returns(prices: pd.Series, periods: int = 1) -> pd.Series:
    """Calculate log returns."""
    return np.log(prices / prices.shift(periods))


def calculate_spread(price1: pd.Series, price2: pd.Series, hedge_ratio: float = None) -> pd.Series:
    """
    Calculate spread between two price series.
    If hedge_ratio provided, uses it; otherwise assumes 1:1 ratio.
    """
    if hedge_ratio is None:
        hedge_ratio = 1.0
    return price1 - (hedge_ratio * price2)


def calculate_zscore(series: pd.Series, window: int = 20) -> pd.Series:
    """Calculate Z-score of a series with rolling window."""
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()
    return (series - rolling_mean) / rolling_std


def calculate_half_life(spread: pd.Series) -> float:
    """
    Calculate half-life of mean reversion using Ornstein-Uhlenbeck process.
    Estimates the average time for spread to revert to mean.
    """
    spread = spread.dropna()
    if len(spread) < 2:
        return np.nan
    
    # Fit AR(1) model: spread_t = c + lambda * spread_{t-1} + epsilon
    from numpy.polynomial import polynomial as P
    
    y = spread.iloc[1:].values
    X = spread.iloc[:-1].values
    
    # Add constant
    X_with_const = np.column_stack([np.ones(len(X)), X])
    
    # OLS regression
    try:
        coeffs = np.linalg.lstsq(X_with_const, y, rcond=None)[0]
        lambda_coeff = coeffs[1]
        
        if lambda_coeff >= 1 or lambda_coeff <= 0:
            return np.nan
        
        half_life = -np.log(2) / np.log(lambda_coeff)
        return half_life
    except:
        return np.nan


def load_price_data(symbol1: str, symbol2: str, start_date: str, 
                   end_date: str, data_source: str = 'yfinance') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load historical price data for two symbols.
    
    Parameters:
    -----------
    symbol1, symbol2 : str
        Ticker symbols
    start_date, end_date : str
        Date range (YYYY-MM-DD)
    data_source : str
        'yfinance' or 'csv'
    
    Returns:
    --------
    prices1, prices2 : pd.DataFrame
        OHLCV data for each symbol
    """
    if data_source == 'yfinance':
        try:
            import yfinance as yf
            prices1 = yf.download(symbol1, start=start_date, end=end_date, progress=False)
            prices2 = yf.download(symbol2, start=start_date, end=end_date, progress=False)
            return prices1, prices2
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    else:
        raise NotImplementedError(f"Data source {data_source} not implemented")


def validate_data(prices1: pd.Series, prices2: pd.Series, min_length: int = 252) -> bool:
    """Validate that price series have sufficient length and no excessive NaNs."""
    if len(prices1) < min_length or len(prices2) < min_length:
        logger.warning(f"Insufficient data: {len(prices1)}, {len(prices2)} samples")
        return False
    
    nan_ratio1 = prices1.isna().sum() / len(prices1)
    nan_ratio2 = prices2.isna().sum() / len(prices2)
    
    if nan_ratio1 > 0.1 or nan_ratio2 > 0.1:
        logger.warning(f"Excessive NaNs: {nan_ratio1:.2%}, {nan_ratio2:.2%}")
        return False
    
    return True


def resample_data(prices: pd.DataFrame, freq: str = 'D') -> pd.DataFrame:
    """
    Resample OHLCV data to different frequency.
    freq: 'D' (daily), 'H' (hourly), 'T' (minute), etc.
    """
    if not isinstance(prices.index, pd.DatetimeIndex):
        prices.index = pd.to_datetime(prices.index)
    
    resampler = prices.resample(freq)
    
    return pd.DataFrame({
        'Open': resampler['Open'].first(),
        'High': resampler['High'].max(),
        'Low': resampler['Low'].min(),
        'Close': resampler['Close'].last(),
        'Volume': resampler['Volume'].sum(),
        'Adj Close': resampler['Adj Close'].last()
    })


def forward_fill_data(prices: pd.DataFrame, limit: int = 5) -> pd.DataFrame:
    """Forward fill missing values up to a limit."""
    return prices.fillna(method='ffill', limit=limit).dropna()
