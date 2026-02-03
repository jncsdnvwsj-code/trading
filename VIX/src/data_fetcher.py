"""
Data fetcher for VIX and related market data
Fetches historical VIX data, S&P 500 data, and VIX futures/options data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from typing import Tuple, Dict, Optional
import warnings

warnings.filterwarnings('ignore')


class VIXDataFetcher:
    """Fetch and process VIX and market data"""
    
    def __init__(self, start_date: str = None, end_date: str = None):
        """
        Initialize data fetcher
        
        Parameters:
        -----------
        start_date : str
            Start date in format YYYY-MM-DD
        end_date : str
            End date in format YYYY-MM-DD
        """
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        self.start_date = start_date or (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')
        
    def fetch_vix_data(self) -> pd.DataFrame:
        """
        Fetch VIX index data from Yahoo Finance
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with VIX data (Close, High, Low, Volume)
        """
        print(f"Fetching VIX data from {self.start_date} to {self.end_date}...")
        try:
            vix_data = yf.download('^VIX', start=self.start_date, end=self.end_date, 
                                   progress=False, interval='1d')
            vix_data = vix_data[['Close', 'High', 'Low', 'Volume']].copy()
            vix_data.columns = ['VIX_Close', 'VIX_High', 'VIX_Low', 'VIX_Volume']
            return vix_data
        except Exception as e:
            print(f"Error fetching VIX data: {e}")
            return pd.DataFrame()
    
    def fetch_sp500_data(self) -> pd.DataFrame:
        """
        Fetch S&P 500 data
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with S&P 500 data
        """
        print(f"Fetching S&P 500 data from {self.start_date} to {self.end_date}...")
        try:
            sp500_data = yf.download('^GSPC', start=self.start_date, end=self.end_date, 
                                     progress=False, interval='1d')
            sp500_data = sp500_data[['Close', 'High', 'Low', 'Volume']].copy()
            sp500_data.columns = ['SPX_Close', 'SPX_High', 'SPX_Low', 'SPX_Volume']
            return sp500_data
        except Exception as e:
            print(f"Error fetching S&P 500 data: {e}")
            return pd.DataFrame()
    
    def fetch_combined_data(self) -> pd.DataFrame:
        """
        Fetch and combine VIX and S&P 500 data
        
        Returns:
        --------
        pd.DataFrame
            Combined DataFrame with both VIX and S&P 500 data
        """
        vix_data = self.fetch_vix_data()
        sp500_data = self.fetch_sp500_data()
        
        combined_data = pd.concat([vix_data, sp500_data], axis=1)
        combined_data = combined_data.dropna()
        
        print(f"Combined data shape: {combined_data.shape}")
        print(f"Date range: {combined_data.index[0]} to {combined_data.index[-1]}")
        
        return combined_data
    
    @staticmethod
    def calculate_returns(data: pd.DataFrame, column: str, periods: list = [1, 5, 21]) -> pd.DataFrame:
        """
        Calculate log returns for specified periods
        
        Parameters:
        -----------
        data : pd.DataFrame
            Input data
        column : str
            Column name to calculate returns for
        periods : list
            List of periods for returns calculation
            
        Returns:
        --------
        pd.DataFrame
            Data with added return columns
        """
        for period in periods:
            data[f'{column}_Return_{period}d'] = np.log(data[column] / data[column].shift(period))
        return data
    
    @staticmethod
    def generate_sample_data(days: int = 1260) -> pd.DataFrame:
        """
        Generate synthetic market data for testing (5 years of trading days)
        
        Parameters:
        -----------
        days : int
            Number of days to generate
            
        Returns:
        --------
        pd.DataFrame
            Synthetic market data
        """
        np.random.seed(42)
        dates = pd.date_range(end=datetime.now(), periods=days, freq='B')
        
        # Generate synthetic VIX data with mean reversion
        vix = []
        vix_level = 15
        for _ in range(days):
            # Mean reversion to 15
            vix_level = vix_level + 0.05 * (15 - vix_level) + np.random.normal(0, 0.8)
            vix_level = max(8, min(80, vix_level))  # Keep VIX between 8 and 80
            vix.append(vix_level)
        
        vix = np.array(vix)
        
        # Generate synthetic S&P 500 data (inverse correlation with VIX)
        spx = []
        spx_price = 4000
        for i in range(days):
            # S&P 500 returns inversely correlated with VIX changes
            vix_return = (vix[i] - vix[i-1]) / vix[i-1] if i > 0 else 0
            sp_return = 0.0003 - 0.02 * vix_return + np.random.normal(0, 0.012)
            spx_price = spx_price * (1 + sp_return)
            spx.append(spx_price)
        
        spx = np.array(spx)
        
        # Create DataFrame
        data = pd.DataFrame({
            'VIX_Close': vix,
            'VIX_High': vix * (1 + np.abs(np.random.normal(0, 0.01, days))),
            'VIX_Low': vix * (1 - np.abs(np.random.normal(0, 0.01, days))),
            'VIX_Volume': np.random.uniform(10e6, 50e6, days),
            'SPX_Close': spx,
            'SPX_High': spx * (1 + np.abs(np.random.normal(0, 0.005, days))),
            'SPX_Low': spx * (1 - np.abs(np.random.normal(0, 0.005, days))),
            'SPX_Volume': np.random.uniform(1e9, 5e9, days),
        }, index=dates)
        
        return data


if __name__ == '__main__':
    # Example usage
    fetcher = VIXDataFetcher(start_date='2020-01-01')
    
    # Try to fetch real data, fall back to synthetic if needed
    try:
        data = fetcher.fetch_combined_data()
        print("\nFetched real data")
    except Exception as e:
        print(f"Could not fetch real data: {e}")
        print("Generating synthetic data...")
        data = fetcher.generate_sample_data(1260)
    
    print("\nData summary:")
    print(data.head())
    print(f"\nData statistics:")
    print(data.describe())
    
    # Save data
    data.to_csv('data/vix_market_data.csv')
    print("\nData saved to data/vix_market_data.csv")
