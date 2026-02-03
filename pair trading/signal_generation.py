"""
Signal generation for pairs trading based on mean reversion.

Signal types:
- Z-score based signals
- Bollinger Bands
- Half-life adjusted thresholds
- Volatility-adjusted signals
"""
import numpy as np
import pandas as pd
from enum import Enum
from typing import Dict, Tuple, List
import logging

logger = logging.getLogger(__name__)


class SignalType(Enum):
    """Types of trading signals."""
    BUY = 1      # Go long pair 1, short pair 2
    SELL = -1    # Go short pair 1, long pair 2
    HOLD = 0     # No signal


class SignalStrength(Enum):
    """Signal strength levels."""
    WEAK = 1
    MODERATE = 2
    STRONG = 3


class MeanReversionSignalGenerator:
    """Generate trading signals based on mean reversion."""
    
    def __init__(self, 
                 z_score_entry: float = 2.0,
                 z_score_exit: float = 0.5,
                 lookback_window: int = 20,
                 min_half_life: float = 5):
        """
        Parameters:
        -----------
        z_score_entry : float
            Z-score threshold for entry signal
        z_score_exit : float
            Z-score threshold for exit signal
        lookback_window : int
            Period for rolling mean/std calculation
        min_half_life : float
            Minimum half-life for position (trading periods)
        """
        self.z_score_entry = z_score_entry
        self.z_score_exit = z_score_exit
        self.lookback_window = lookback_window
        self.min_half_life = min_half_life
    
    def calculate_zscore(self, spread: pd.Series) -> pd.Series:
        """Calculate Z-score of spread."""
        rolling_mean = spread.rolling(window=self.lookback_window).mean()
        rolling_std = spread.rolling(window=self.lookback_window).std()
        zscore = (spread - rolling_mean) / rolling_std
        return zscore
    
    def generate_signals(self, spread: pd.Series) -> pd.DataFrame:
        """
        Generate signals based on Z-score of spread.
        
        Returns:
        --------
        DataFrame with columns: 'zscore', 'signal', 'strength'
        """
        zscore = self.calculate_zscore(spread)
        
        signals = pd.DataFrame({
            'zscore': zscore,
            'spread': spread
        })
        
        signals['signal'] = SignalType.HOLD.value
        signals['strength'] = 0
        
        # Entry signals (when spread deviates from mean)
        signals.loc[zscore > self.z_score_entry, 'signal'] = SignalType.SELL.value
        signals.loc[zscore < -self.z_score_entry, 'signal'] = SignalType.BUY.value
        
        # Exit signals (when spread returns toward mean)
        signals.loc[(zscore > 0) & (zscore < self.z_score_exit), 'signal'] = SignalType.HOLD.value
        signals.loc[(zscore < 0) & (zscore > -self.z_score_exit), 'signal'] = SignalType.HOLD.value
        
        # Signal strength
        signals['strength'] = np.where(
            np.abs(signals['zscore']) > self.z_score_entry * 1.5,
            SignalStrength.STRONG.value,
            np.where(
                np.abs(signals['zscore']) > self.z_score_entry,
                SignalStrength.MODERATE.value,
                SignalStrength.WEAK.value
            )
        )
        
        return signals
    
    def generate_adaptive_signals(self, spread: pd.Series, 
                                 half_life: float) -> pd.DataFrame:
        """
        Generate adaptive signals based on half-life of mean reversion.
        Adjust thresholds based on expected reversion speed.
        """
        # Scale thresholds by half-life
        half_life_factor = np.sqrt(20 / half_life) if half_life > 0 else 1.0
        half_life_factor = np.clip(half_life_factor, 0.5, 2.0)
        
        # Temporarily adjust thresholds
        original_entry = self.z_score_entry
        original_exit = self.z_score_exit
        
        self.z_score_entry *= half_life_factor
        self.z_score_exit *= half_life_factor / 2
        
        try:
            signals = self.generate_signals(spread)
        finally:
            self.z_score_entry = original_entry
            self.z_score_exit = original_exit
        
        signals['half_life_factor'] = half_life_factor
        return signals


class BollingerBandSignalGenerator:
    """Generate signals based on Bollinger Bands."""
    
    def __init__(self, period: int = 20, std_dev: float = 2.0):
        """
        Parameters:
        -----------
        period : int
            Period for moving average
        std_dev : float
            Number of standard deviations for bands
        """
        self.period = period
        self.std_dev = std_dev
    
    def generate_signals(self, spread: pd.Series) -> pd.DataFrame:
        """
        Generate signals when spread touches Bollinger Bands.
        
        Returns:
        --------
        DataFrame with BB bands and signals
        """
        sma = spread.rolling(window=self.period).mean()
        std = spread.rolling(window=self.period).std()
        
        upper_band = sma + (self.std_dev * std)
        lower_band = sma - (self.std_dev * std)
        
        signals = pd.DataFrame({
            'spread': spread,
            'sma': sma,
            'upper_band': upper_band,
            'lower_band': lower_band,
            'signal': SignalType.HOLD.value
        })
        
        # Touch upper band -> SELL (short pair1, long pair2)
        signals.loc[spread >= upper_band, 'signal'] = SignalType.SELL.value
        
        # Touch lower band -> BUY (long pair1, short pair2)
        signals.loc[spread <= lower_band, 'signal'] = SignalType.BUY.value
        
        # Exit at moving average
        signals.loc[
            ((signals['signal'].shift(1) == SignalType.BUY) & (spread >= sma)) |
            ((signals['signal'].shift(1) == SignalType.SELL) & (spread <= sma)),
            'signal'
        ] = SignalType.HOLD.value
        
        return signals


class OrnsteinUhlenbeckSignalGenerator:
    """Generate signals using Ornstein-Uhlenbeck process assumptions."""
    
    def __init__(self, entry_threshold: float = 2.0, 
                 exit_threshold: float = 0.5,
                 lambda_coeff: float = None):
        """
        Parameters:
        -----------
        entry_threshold : float
            Entry threshold in standard deviations
        exit_threshold : float
            Exit threshold
        lambda_coeff : float
            Mean reversion speed coefficient. If None, estimate from data.
        """
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.lambda_coeff = lambda_coeff
    
    def estimate_lambda(self, spread: pd.Series, lookback: int = 252) -> float:
        """Estimate mean reversion coefficient from historical data."""
        spread_clean = spread.dropna().tail(lookback)
        
        if len(spread_clean) < 2:
            return 0.01
        
        # AR(1) regression: spread_t = c + lambda * spread_{t-1} + epsilon
        y = spread_clean.iloc[1:].values
        X = spread_clean.iloc[:-1].values
        X_with_const = np.column_stack([np.ones(len(X)), X])
        
        try:
            coeffs = np.linalg.lstsq(X_with_const, y, rcond=None)[0]
            lambda_coeff = coeffs[1]
            
            if lambda_coeff < 0 or lambda_coeff >= 1:
                return 0.01
            
            return lambda_coeff
        except:
            return 0.01
    
    def generate_signals(self, spread: pd.Series, 
                        half_life: float = None) -> pd.DataFrame:
        """
        Generate OU process-based signals.
        """
        spread_clean = spread.dropna()
        
        if len(spread_clean) < 20:
            return pd.DataFrame()
        
        # Calculate rolling statistics
        rolling_mean = spread_clean.rolling(window=20).mean()
        rolling_std = spread_clean.rolling(window=20).std()
        
        # OU expected value at each point
        lambda_coeff = self.lambda_coeff or self.estimate_lambda(spread)
        
        deviation = (spread_clean - rolling_mean) / rolling_std
        
        signals = pd.DataFrame({
            'spread': spread_clean,
            'deviation': deviation,
            'rolling_mean': rolling_mean,
            'rolling_std': rolling_std,
            'signal': SignalType.HOLD.value
        }, index=spread_clean.index)
        
        # Entry signals
        signals.loc[deviation > self.entry_threshold, 'signal'] = SignalType.SELL.value
        signals.loc[deviation < -self.entry_threshold, 'signal'] = SignalType.BUY.value
        
        # Exit signals
        signals.loc[
            (signals['signal'].shift(1) != SignalType.HOLD) & 
            (np.abs(deviation) < self.exit_threshold),
            'signal'
        ] = SignalType.HOLD.value
        
        signals['lambda'] = lambda_coeff
        
        return signals


class VolatilityAdjustedSignalGenerator:
    """Adjust signals based on market volatility."""
    
    def __init__(self, base_generator: MeanReversionSignalGenerator,
                 vol_window: int = 20):
        self.base_generator = base_generator
        self.vol_window = vol_window
    
    def generate_signals(self, spread: pd.Series, 
                        returns: pd.Series = None) -> pd.DataFrame:
        """
        Generate volatility-adjusted signals.
        Lower volatility -> wider entry bands
        Higher volatility -> tighter entry bands
        """
        signals = self.base_generator.generate_signals(spread)
        
        if returns is not None:
            volatility = returns.rolling(window=self.vol_window).std()
            volatility_zscore = (volatility - volatility.mean()) / volatility.std()
            
            # Adjust signal strength based on volatility
            vol_adjustment = 1.0 + (0.2 * volatility_zscore)
            vol_adjustment = np.clip(vol_adjustment, 0.5, 1.5)
            
            signals['volatility_adjustment'] = vol_adjustment
            signals['adjusted_zscore'] = signals['zscore'] / vol_adjustment
        
        return signals


class SignalFilter:
    """Filter signals based on market conditions."""
    
    def __init__(self, min_signal_strength: SignalStrength = SignalStrength.MODERATE):
        self.min_signal_strength = min_signal_strength
    
    def filter_signals(self, signals: pd.DataFrame, 
                      min_volume: float = None,
                      price_levels: pd.DataFrame = None) -> pd.DataFrame:
        """
        Apply filters to trading signals.
        
        Parameters:
        -----------
        signals : pd.DataFrame
            Generated signals
        min_volume : float
            Minimum volume requirement
        price_levels : pd.DataFrame
            Price data for additional filters
        
        Returns:
        --------
        Filtered signals
        """
        filtered = signals.copy()
        
        # Filter by signal strength
        if 'strength' in filtered.columns:
            filtered.loc[
                filtered['strength'] < self.min_signal_strength.value,
                'signal'
            ] = SignalType.HOLD.value
        
        # Filter by volume if provided
        if min_volume is not None and 'volume' in price_levels.columns:
            filtered.loc[price_levels['Volume'] < min_volume, 'signal'] = SignalType.HOLD.value
        
        return filtered
