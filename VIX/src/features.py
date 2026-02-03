"""
Feature engineering and volatility regime detection
Build features to identify different market volatility regimes
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy import stats
from typing import Tuple, Dict
import warnings

warnings.filterwarnings('ignore')


class VolatilityFeatures:
    """Calculate volatility-related features"""
    
    @staticmethod
    def calculate_rolling_volatility(data: pd.DataFrame, column: str, 
                                     windows: list = [5, 10, 21, 63]) -> pd.DataFrame:
        """
        Calculate rolling volatility (standard deviation of returns)
        
        Parameters:
        -----------
        data : pd.DataFrame
            Input data
        column : str
            Column name for returns calculation
        windows : list
            List of rolling windows in days
            
        Returns:
        --------
        pd.DataFrame
            Data with rolling volatility columns
        """
        returns = np.log(data[column] / data[column].shift(1)).dropna()
        
        for window in windows:
            data[f'{column}_RollingVol_{window}d'] = returns.rolling(window).std() * np.sqrt(252)
        
        return data
    
    @staticmethod
    def calculate_vix_term_structure(vix_close: pd.Series) -> pd.DataFrame:
        """
        Simulate VIX term structure features using VIX spot with lags
        In real implementation, would use actual VIX futures prices
        
        Parameters:
        -----------
        vix_close : pd.Series
            VIX closing prices
            
        Returns:
        --------
        pd.DataFrame
            Term structure features
        """
        df = pd.DataFrame(index=vix_close.index)
        
        # Simulate different contract months using VIX lags
        df['VIX_F1'] = vix_close  # Front contract (spot)
        df['VIX_F2'] = vix_close.rolling(2).mean()  # 2-month out (simplified)
        df['VIX_F3'] = vix_close.rolling(3).mean()  # 3-month out (simplified)
        df['VIX_F4'] = vix_close.rolling(4).mean()  # 4-month out (simplified)
        
        # Calculate term structure slope and curve
        df['VIX_Slope_F12'] = df['VIX_F2'] - df['VIX_F1']  # F2-F1 spread
        df['VIX_Slope_F23'] = df['VIX_F3'] - df['VIX_F2']  # F3-F2 spread
        df['VIX_Slope_F34'] = df['VIX_F4'] - df['VIX_F3']  # F4-F3 spread
        df['VIX_Slope_F14'] = df['VIX_F4'] - df['VIX_F1']  # F4-F1 overall slope
        
        # Convexity (curvature)
        df['VIX_Convexity'] = df['VIX_F1'] + df['VIX_F3'] - 2 * df['VIX_F2']
        
        return df
    
    @staticmethod
    def calculate_mean_reversion_features(data: pd.DataFrame, column: str,
                                          windows: list = [20, 60, 120]) -> pd.DataFrame:
        """
        Calculate mean reversion features
        
        Parameters:
        -----------
        data : pd.DataFrame
            Input data
        column : str
            Column to analyze
        windows : list
            Windows for moving averages
            
        Returns:
        --------
        pd.DataFrame
            Data with mean reversion features
        """
        for window in windows:
            ma = data[column].rolling(window).mean()
            std = data[column].rolling(window).std()
            
            # Z-score (deviation from mean)
            data[f'{column}_ZScore_{window}d'] = (data[column] - ma) / std
            
            # Distance to moving average in percentage
            data[f'{column}_DistMA_{window}d'] = ((data[column] - ma) / ma) * 100
            
            # Moving average slope (momentum)
            data[f'{column}_MomentumMA_{window}d'] = ma.diff()
        
        return data
    
    @staticmethod
    def calculate_vix_momentum(vix_close: pd.Series, 
                               periods: list = [5, 10, 21]) -> pd.DataFrame:
        """
        Calculate VIX momentum indicators
        
        Parameters:
        -----------
        vix_close : pd.Series
            VIX prices
        periods : list
            Momentum periods
            
        Returns:
        --------
        pd.DataFrame
            VIX momentum features
        """
        df = pd.DataFrame(index=vix_close.index)
        
        for period in periods:
            # Rate of change
            df[f'VIX_ROC_{period}d'] = (vix_close - vix_close.shift(period)) / vix_close.shift(period)
            
            # Momentum (difference)
            df[f'VIX_Momentum_{period}d'] = vix_close.diff(period)
            
            # Relative Strength Index (RSI)
            delta = vix_close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            df[f'VIX_RSI_{period}d'] = 100 - (100 / (1 + rs))
        
        return df
    
    @staticmethod
    def calculate_volatility_of_volatility(vix_close: pd.Series, 
                                           window: int = 20) -> pd.Series:
        """
        Calculate volatility of VIX (Vol of Vol)
        
        Parameters:
        -----------
        vix_close : pd.Series
            VIX prices
        window : int
            Rolling window
            
        Returns:
        --------
        pd.Series
            Volatility of volatility
        """
        vix_returns = np.log(vix_close / vix_close.shift(1))
        vol_of_vol = vix_returns.rolling(window).std() * np.sqrt(252)
        return vol_of_vol
    
    @staticmethod
    def calculate_vix_sp500_relationship(data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate features related to VIX-S&P 500 relationship
        
        Parameters:
        -----------
        data : pd.DataFrame
            Data with VIX_Close and SPX_Close
            
        Returns:
        --------
        pd.DataFrame
            Relationship features
        """
        # Calculate returns
        vix_return = np.log(data['VIX_Close'] / data['VIX_Close'].shift(1))
        spx_return = np.log(data['SPX_Close'] / data['SPX_Close'].shift(1))
        
        # Rolling correlation
        rolling_corr = vix_return.rolling(20).corr(spx_return)
        data['VIX_SPX_Correlation_20d'] = rolling_corr
        
        # VIX-SPX relationship: ratio
        data['VIX_SPX_Ratio'] = data['VIX_Close'] / (data['SPX_Close'] / 100)
        
        # Divergence: when VIX and SPX move differently than usual
        data['VIX_SPX_Divergence'] = vix_return + spx_return  # Should be ~0 in normal correlation
        
        return data


class RegimeDetector:
    """Detect and classify volatility regimes"""
    
    @staticmethod
    def detect_regimes_vix_percentile(vix_close: pd.Series, 
                                     window: int = 60,
                                     low_threshold: float = 0.33,
                                     high_threshold: float = 0.67) -> pd.Series:
        """
        Detect regimes based on VIX percentile within rolling window
        
        Parameters:
        -----------
        vix_close : pd.Series
            VIX closing prices
        window : int
            Rolling window for percentile calculation
        low_threshold : float
            Percentile threshold for low volatility (0-1)
        high_threshold : float
            Percentile threshold for high volatility (0-1)
            
        Returns:
        --------
        pd.Series
            Regime classification (0=Low Vol, 1=Normal, 2=High Vol)
        """
        percentiles = []
        
        for i in range(len(vix_close)):
            if i < window:
                percentiles.append(np.nan)
            else:
                window_data = vix_close.iloc[i-window:i]
                pct = stats.percentileofscore(window_data, vix_close.iloc[i]) / 100
                percentiles.append(pct)
        
        percentiles = pd.Series(percentiles, index=vix_close.index)
        
        # Classify regimes
        regimes = pd.Series(1, index=vix_close.index)  # Default to Normal
        regimes[percentiles < low_threshold] = 0  # Low volatility
        regimes[percentiles > high_threshold] = 2  # High volatility
        
        return regimes
    
    @staticmethod
    def detect_regimes_zscore(data: pd.DataFrame, column: str = 'VIX_Close',
                             window: int = 60,
                             threshold: float = 1.0) -> pd.Series:
        """
        Detect regimes based on Z-score of VIX
        
        Parameters:
        -----------
        data : pd.DataFrame
            Input data
        column : str
            Column to analyze
        window : int
            Rolling window for mean/std calculation
        threshold : float
            Z-score threshold for regime classification
            
        Returns:
        --------
        pd.Series
            Regime classification
        """
        ma = data[column].rolling(window).mean()
        std = data[column].rolling(window).std()
        z_score = (data[column] - ma) / std
        
        regimes = pd.Series(1, index=data.index)  # Default to Normal
        regimes[z_score < -threshold] = 0  # Low volatility (z < -1)
        regimes[z_score > threshold] = 2  # High volatility (z > 1)
        
        return regimes
    
    @staticmethod
    def detect_regimes_kmeans(data: pd.DataFrame, features: list, n_regimes: int = 3):
        """
        Detect regimes using K-Means clustering
        
        Parameters:
        -----------
        data : pd.DataFrame
            Input data with feature columns
        features : list
            List of feature column names to use
        n_regimes : int
            Number of regimes to detect
            
        Returns:
        --------
        pd.Series
            Regime classification
        """
        from sklearn.cluster import KMeans
        
        # Prepare features
        feature_data = data[features].dropna()
        
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(feature_data)
        
        # Cluster
        kmeans = KMeans(n_clusters=n_regimes, random_state=42, n_init=10)
        regimes = kmeans.fit_predict(features_scaled)
        
        # Create regime series with proper index
        regime_series = pd.Series(np.nan, index=data.index)
        regime_series.loc[feature_data.index] = regimes
        
        # Sort regimes by average VIX level
        regime_mapping = {}
        for regime in range(n_regimes):
            avg_vix = data.loc[regime_series == regime, 'VIX_Close'].mean()
            regime_mapping[regime] = (regime, avg_vix)
        
        # Sort by VIX level and remap: 0=Low, 1=Normal, 2=High
        sorted_regimes = sorted(regime_mapping.items(), key=lambda x: x[1][1])
        remap = {old: new for new, (old, _) in enumerate(sorted_regimes)}
        regime_series = regime_series.map(remap)
        
        return regime_series
    
    @staticmethod
    def detect_volatility_spikes(vix_close: pd.Series, threshold_std: float = 2.0,
                                 window: int = 20) -> pd.Series:
        """
        Detect sudden volatility spikes
        
        Parameters:
        -----------
        vix_close : pd.Series
            VIX closing prices
        threshold_std : float
            Number of standard deviations for spike detection
        window : int
            Rolling window
            
        Returns:
        --------
        pd.Series
            Boolean series indicating spikes
        """
        vix_returns = np.log(vix_close / vix_close.shift(1))
        rolling_std = vix_returns.rolling(window).std()
        rolling_mean = vix_returns.rolling(window).mean()
        
        spikes = np.abs(vix_returns - rolling_mean) > (threshold_std * rolling_std)
        
        return spikes


if __name__ == '__main__':
    # Example usage
    from data_fetcher import VIXDataFetcher
    
    # Load data
    fetcher = VIXDataFetcher(start_date='2020-01-01')
    try:
        data = fetcher.fetch_combined_data()
    except:
        print("Generating synthetic data...")
        data = fetcher.generate_sample_data(1260)
    
    # Calculate features
    print("Calculating volatility features...")
    data = VolatilityFeatures.calculate_rolling_volatility(data, 'VIX_Close')
    data = VolatilityFeatures.calculate_mean_reversion_features(data, 'VIX_Close')
    data = VolatilityFeatures.calculate_vix_momentum(data, 'VIX_Close')
    data = VolatilityFeatures.calculate_vix_sp500_relationship(data)
    
    # Detect regimes
    print("Detecting volatility regimes...")
    data['Regime_Percentile'] = RegimeDetector.detect_regimes_vix_percentile(data['VIX_Close'])
    data['Regime_ZScore'] = RegimeDetector.detect_regimes_zscore(data, 'VIX_Close')
    data['VolSpikes'] = RegimeDetector.detect_volatility_spikes(data['VIX_Close'])
    
    print("\nFeatures calculated successfully!")
    print(data.head())
    
    # Save features
    data.to_csv('data/vix_features.csv')
    print("\nFeatures saved to data/vix_features.csv")
