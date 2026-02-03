"""
VIX Trading Strategies
Implement various volatility trading strategies
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
from abc import ABC, abstractmethod
import warnings

warnings.filterwarnings('ignore')


class BaseStrategy(ABC):
    """Base class for all strategies"""
    
    def __init__(self, data: pd.DataFrame, initial_capital: float = 100000,
                 transaction_cost: float = 0.001):
        """
        Initialize strategy
        
        Parameters:
        -----------
        data : pd.DataFrame
            Market data with required columns
        initial_capital : float
            Starting capital in dollars
        transaction_cost : float
            Transaction cost as percentage (0.1% = 0.001)
        """
        self.data = data.copy()
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.positions = pd.DataFrame(index=data.index)
        self.returns = pd.DataFrame(index=data.index)
        
    @abstractmethod
    def generate_signals(self) -> pd.Series:
        """Generate trading signals"""
        pass
    
    def calculate_returns(self, position_series: pd.Series) -> pd.Series:
        """Calculate strategy returns from positions"""
        pass


class MeanReversionStrategy(BaseStrategy):
    """
    Mean Reversion Strategy: VIX reverts to long-term average
    - Short VIX futures when VIX is high (above MA + threshold * std)
    - Long VIX futures when VIX is low (below MA - threshold * std)
    - Close positions when VIX reverts to mean
    """
    
    def __init__(self, data: pd.DataFrame, vix_column: str = 'VIX_Close',
                 ma_window: int = 60, threshold: float = 1.5,
                 position_size: float = 1.0, **kwargs):
        """
        Initialize Mean Reversion Strategy
        
        Parameters:
        -----------
        data : pd.DataFrame
            Market data
        vix_column : str
            Column name for VIX prices
        ma_window : int
            Moving average window
        threshold : float
            Number of standard deviations for entry
        position_size : float
            Position size as percentage of capital (0-1)
        """
        super().__init__(data, **kwargs)
        self.vix_column = vix_column
        self.ma_window = ma_window
        self.threshold = threshold
        self.position_size = position_size
        
    def generate_signals(self) -> pd.DataFrame:
        """
        Generate trading signals based on mean reversion
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with signal, entry_price, exit_price columns
        """
        vix = self.data[self.vix_column]
        ma = vix.rolling(self.ma_window).mean()
        std = vix.rolling(self.ma_window).std()
        
        # Z-score
        z_score = (vix - ma) / std
        
        # Signals: 1 = short (high vol), -1 = long (low vol), 0 = neutral
        signals = pd.Series(0, index=self.data.index)
        
        # Short when VIX is very high
        signals[z_score > self.threshold] = 1
        
        # Long when VIX is very low
        signals[z_score < -self.threshold] = -1
        
        # Close positions when reverted (z-score between -0.5 and 0.5)
        signals[(np.abs(z_score) <= 0.5) & (signals.shift(1) != 0)] = 0
        
        # Forward fill to maintain positions
        signals_filled = signals.copy()
        current_signal = 0
        for i in range(len(signals)):
            if signals.iloc[i] != 0:
                current_signal = signals.iloc[i]
            elif current_signal != 0 and signals.iloc[i] == 0:
                signals_filled.iloc[i] = current_signal
            else:
                signals_filled.iloc[i] = current_signal
        
        self.positions['Signal'] = signals_filled
        self.positions['VIX'] = vix
        self.positions['MA'] = ma
        self.positions['ZScore'] = z_score
        
        return self.positions
    
    def calculate_pnl(self) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Calculate P&L for the strategy
        
        Returns:
        --------
        Tuple of (returns series, detailed P&L dataframe)
        """
        signals = self.generate_signals()
        vix = self.data[self.vix_column]
        
        # Position changes
        position_change = signals['Signal'].diff().fillna(0)
        
        # Entry and exit prices
        entry_prices = vix.where(position_change != 0, np.nan)
        
        # For each position, track entry price
        current_entry_price = np.nan
        entry_prices_filled = []
        
        for i in range(len(vix)):
            if position_change.iloc[i] != 0:
                current_entry_price = vix.iloc[i]
            entry_prices_filled.append(current_entry_price)
        
        # Calculate P&L
        pnl = pd.Series(index=self.data.index, dtype=float)
        
        for i in range(len(vix)):
            if signals['Signal'].iloc[i] == 0:
                pnl.iloc[i] = 0
            else:
                # For short positions (Signal=1): gain when VIX goes down
                if signals['Signal'].iloc[i] == 1:
                    pnl.iloc[i] = (entry_prices_filled[i] - vix.iloc[i]) / entry_prices_filled[i]
                # For long positions (Signal=-1): gain when VIX goes up
                else:
                    pnl.iloc[i] = (vix.iloc[i] - entry_prices_filled[i]) / entry_prices_filled[i]
        
        # Apply transaction costs only on position changes
        pnl = pnl - (np.abs(position_change) * self.transaction_cost)
        
        return pnl, signals


class TrendFollowingStrategy(BaseStrategy):
    """
    Trend Following Strategy: Follow VIX momentum
    - Long VIX when it's rising (positive momentum)
    - Short VIX when it's falling (negative momentum)
    - Use 2-week and 4-week moving averages for trend confirmation
    """
    
    def __init__(self, data: pd.DataFrame, vix_column: str = 'VIX_Close',
                 short_ma: int = 10, long_ma: int = 21,
                 position_size: float = 1.0, **kwargs):
        """
        Initialize Trend Following Strategy
        
        Parameters:
        -----------
        data : pd.DataFrame
            Market data
        vix_column : str
            Column name for VIX prices
        short_ma : int
            Short moving average window (days)
        long_ma : int
            Long moving average window (days)
        position_size : float
            Position size as percentage of capital
        """
        super().__init__(data, **kwargs)
        self.vix_column = vix_column
        self.short_ma = short_ma
        self.long_ma = long_ma
        self.position_size = position_size
        
    def generate_signals(self) -> pd.DataFrame:
        """
        Generate trend following signals
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with signals
        """
        vix = self.data[self.vix_column]
        ma_short = vix.rolling(self.short_ma).mean()
        ma_long = vix.rolling(self.long_ma).mean()
        
        # Signal: 1 = uptrend (short MA > long MA), -1 = downtrend
        signals = pd.Series(0, index=self.data.index)
        signals[ma_short > ma_long] = 1  # Uptrend = short VIX
        signals[ma_short < ma_long] = -1  # Downtrend = long VIX
        
        self.positions['Signal'] = signals
        self.positions['VIX'] = vix
        self.positions['MA_Short'] = ma_short
        self.positions['MA_Long'] = ma_long
        
        return self.positions
    
    def calculate_pnl(self) -> Tuple[pd.Series, pd.DataFrame]:
        """Calculate P&L for trend following strategy"""
        signals = self.generate_signals()
        vix = self.data[self.vix_column]
        
        # P&L: daily change * position
        vix_returns = np.log(vix / vix.shift(1))
        
        # VIX inverts positions: short = betting on decline, long = betting on rise
        # For simplicity: signal direction = position direction on VIX returns
        pnl = signals['Signal'].shift(1) * vix_returns
        
        # Apply transaction costs
        position_change = signals['Signal'].diff().fillna(0)
        pnl = pnl - (np.abs(position_change) * self.transaction_cost)
        
        return pnl, signals


class VolatilityOfVolatilityStrategy(BaseStrategy):
    """
    Vol of Vol Strategy: Trade based on volatility of volatility
    - When volatility is calm (low vol of vol): go long VIX-like instruments
    - When volatility is choppy (high vol of vol): reduce positions or hedge
    - Useful for long-vol strategies to reduce drawdowns
    """
    
    def __init__(self, data: pd.DataFrame, vix_column: str = 'VIX_Close',
                 vol_of_vol_window: int = 20, vol_threshold_pct: float = 75,
                 position_size: float = 1.0, **kwargs):
        """
        Initialize Vol of Vol Strategy
        
        Parameters:
        -----------
        data : pd.DataFrame
            Market data
        vix_column : str
            Column name for VIX
        vol_of_vol_window : int
            Window for calculating vol of vol
        vol_threshold_pct : float
            Percentile threshold (0-100) for determining high/low vol of vol
        position_size : float
            Position size as percentage
        """
        super().__init__(data, **kwargs)
        self.vix_column = vix_column
        self.vol_of_vol_window = vol_of_vol_window
        self.vol_threshold_pct = vol_threshold_pct
        self.position_size = position_size
        
    def generate_signals(self) -> pd.DataFrame:
        """Generate Vol of Vol signals"""
        vix = self.data[self.vix_column]
        vix_returns = np.log(vix / vix.shift(1))
        vol_of_vol = vix_returns.rolling(self.vol_of_vol_window).std() * np.sqrt(252)
        
        # Percentile of vol of vol
        vol_of_vol_pct = vol_of_vol.rolling(63).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1]
        ) * 100
        
        # Signals: 1 = low vol of vol (go long), -1 = high vol of vol (go short/hedge)
        signals = pd.Series(0, index=self.data.index)
        signals[vol_of_vol_pct < (100 - self.vol_threshold_pct)] = 1
        signals[vol_of_vol_pct > self.vol_threshold_pct] = -1
        
        self.positions['Signal'] = signals
        self.positions['VIX'] = vix
        self.positions['VolOfVol'] = vol_of_vol
        self.positions['VolOfVolPct'] = vol_of_vol_pct
        
        return self.positions
    
    def calculate_pnl(self) -> Tuple[pd.Series, pd.DataFrame]:
        """Calculate P&L"""
        signals = self.generate_signals()
        vix = self.data[self.vix_column]
        vix_returns = np.log(vix / vix.shift(1))
        
        pnl = signals['Signal'].shift(1) * vix_returns
        
        position_change = signals['Signal'].diff().fillna(0)
        pnl = pnl - (np.abs(position_change) * self.transaction_cost)
        
        return pnl, signals


class HedgedVolatilityStrategy(BaseStrategy):
    """
    Hedged Volatility Strategy: Combine mean reversion with hedging
    - Long equity (S&P 500) with VIX mean reversion hedge
    - When VIX is elevated: reduce equity exposure, increase hedge
    - When VIX is depressed: increase equity exposure, reduce hedge
    """
    
    def __init__(self, data: pd.DataFrame, vix_column: str = 'VIX_Close',
                 spx_column: str = 'SPX_Close', long_threshold: float = 2.0,
                 short_threshold: float = 1.0, **kwargs):
        """
        Initialize Hedged Volatility Strategy
        
        Parameters:
        -----------
        data : pd.DataFrame
            Market data with VIX and S&P 500
        vix_column : str
            Column name for VIX
        spx_column : str
            Column name for S&P 500
        long_threshold : float
            Z-score threshold for long equity
        short_threshold : float
            Z-score threshold for reducing exposure
        """
        super().__init__(data, **kwargs)
        self.vix_column = vix_column
        self.spx_column = spx_column
        self.long_threshold = long_threshold
        self.short_threshold = short_threshold
        
    def generate_signals(self) -> pd.DataFrame:
        """Generate hedged strategy signals"""
        vix = self.data[self.vix_column]
        spx = self.data[self.spx_column]
        
        ma = vix.rolling(60).mean()
        std = vix.rolling(60).std()
        z_score = (vix - ma) / std
        
        # Equity allocation based on VIX level
        equity_weight = pd.Series(1.0, index=self.data.index)
        
        # Reduce exposure when VIX is very elevated (z > 1.5)
        equity_weight[z_score > self.short_threshold] = 0.5
        
        # Increase hedge when VIX is extremely elevated (z > 2)
        equity_weight[z_score > self.long_threshold] = 0.2
        
        # Fully invested when VIX is depressed (z < -1)
        equity_weight[z_score < -1.0] = 1.2
        
        self.positions['EquityWeight'] = equity_weight
        self.positions['VIX'] = vix
        self.positions['ZScore'] = z_score
        self.positions['SPX'] = spx
        
        return self.positions
    
    def calculate_pnl(self) -> Tuple[pd.Series, pd.DataFrame]:
        """Calculate portfolio P&L"""
        signals = self.generate_signals()
        spx = self.data[self.spx_column]
        spx_returns = np.log(spx / spx.shift(1))
        
        # Portfolio return = equity return * weight
        pnl = signals['EquityWeight'].shift(1) * spx_returns
        
        # Rebalancing costs when weight changes
        weight_change = signals['EquityWeight'].diff().fillna(0)
        pnl = pnl - (np.abs(weight_change) * self.transaction_cost)
        
        return pnl, signals


if __name__ == '__main__':
    # Example usage
    from data_fetcher import VIXDataFetcher
    
    fetcher = VIXDataFetcher(start_date='2020-01-01')
    try:
        data = fetcher.fetch_combined_data()
    except:
        print("Generating synthetic data...")
        data = fetcher.generate_sample_data(1260)
    
    print("Testing strategies...")
    
    # Test Mean Reversion
    mr_strategy = MeanReversionStrategy(data)
    mr_pnl, mr_signals = mr_strategy.calculate_pnl()
    print(f"\nMean Reversion Strategy")
    print(f"  Total Return: {(mr_pnl.sum() * 100):.2f}%")
    
    # Test Trend Following
    tf_strategy = TrendFollowingStrategy(data)
    tf_pnl, tf_signals = tf_strategy.calculate_pnl()
    print(f"\nTrend Following Strategy")
    print(f"  Total Return: {(tf_pnl.sum() * 100):.2f}%")
    
    # Test Vol of Vol
    vv_strategy = VolatilityOfVolatilityStrategy(data)
    vv_pnl, vv_signals = vv_strategy.calculate_pnl()
    print(f"\nVol of Vol Strategy")
    print(f"  Total Return: {(vv_pnl.sum() * 100):.2f}%")
    
    # Test Hedged
    hedged_strategy = HedgedVolatilityStrategy(data)
    hedged_pnl, hedged_signals = hedged_strategy.calculate_pnl()
    print(f"\nHedged Volatility Strategy")
    print(f"  Total Return: {(hedged_pnl.sum() * 100):.2f}%")
