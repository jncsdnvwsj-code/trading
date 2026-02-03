"""
Backtesting engine for volatility trading strategies
Calculate performance metrics and analyze strategy results
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


class Backtester:
    """Complete backtesting engine for trading strategies"""
    
    def __init__(self, initial_capital: float = 100000, risk_free_rate: float = 0.04):
        """
        Initialize backtester
        
        Parameters:
        -----------
        initial_capital : float
            Starting capital
        risk_free_rate : float
            Risk-free rate for Sharpe ratio calculation (annualized)
        """
        self.initial_capital = initial_capital
        self.risk_free_rate = risk_free_rate
        
    def backtest_strategy(self, returns: pd.Series) -> Dict:
        """
        Run backtest on a strategy
        
        Parameters:
        -----------
        returns : pd.Series
            Daily returns of the strategy
            
        Returns:
        --------
        Dict
            Dictionary with all performance metrics
        """
        # Calculate cumulative returns and equity curve
        cumulative_returns = (1 + returns).cumprod()
        equity_curve = self.initial_capital * cumulative_returns
        
        # Basic metrics
        total_return = cumulative_returns.iloc[-1] - 1
        annual_return = (1 + total_return) ** (252 / len(returns)) - 1
        
        # Volatility metrics
        daily_vol = returns.std()
        annual_vol = daily_vol * np.sqrt(252)
        
        # Sharpe ratio
        sharpe_ratio = (annual_return - self.risk_free_rate) / annual_vol if annual_vol > 0 else 0
        
        # Drawdown metrics
        running_max = equity_curve.expanding().max()
        drawdown = (equity_curve - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Drawdown duration
        drawdown_duration = self._calculate_drawdown_duration(drawdown)
        
        # Win rate and profit factor
        win_rate = len(returns[returns > 0]) / len(returns[returns != 0]) if len(returns[returns != 0]) > 0 else 0
        positive_returns = returns[returns > 0].sum()
        negative_returns = abs(returns[returns < 0].sum())
        profit_factor = positive_returns / negative_returns if negative_returns > 0 else 0
        
        # Risk metrics
        sortino_ratio = self._calculate_sortino_ratio(returns, self.risk_free_rate / 252)
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Monthly analysis
        monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        
        metrics = {
            'Total Return': total_return,
            'Annual Return': annual_return,
            'Volatility': annual_vol,
            'Sharpe Ratio': sharpe_ratio,
            'Sortino Ratio': sortino_ratio,
            'Max Drawdown': max_drawdown,
            'Max Drawdown Duration (Days)': drawdown_duration,
            'Win Rate': win_rate,
            'Profit Factor': profit_factor,
            'Calmar Ratio': calmar_ratio,
            'Monthly Win Rate': len(monthly_returns[monthly_returns > 0]) / len(monthly_returns[monthly_returns != 0]) if len(monthly_returns[monthly_returns != 0]) > 0 else 0,
            'Best Day': returns.max(),
            'Worst Day': returns.min(),
            'Best Month': monthly_returns.max(),
            'Worst Month': monthly_returns.min(),
            'Positive Months': len(monthly_returns[monthly_returns > 0]),
            'Total Months': len(monthly_returns),
            'Avg Daily Return': returns.mean(),
            'Avg Monthly Return': monthly_returns.mean(),
            'Cumulative Return': cumulative_returns.iloc[-1] - 1,
        }
        
        return metrics
    
    def analyze_by_regime(self, returns: pd.Series, regimes: pd.Series) -> Dict:
        """
        Analyze strategy performance by market regime
        
        Parameters:
        -----------
        returns : pd.Series
            Daily returns
        regimes : pd.Series
            Regime classification (0=Low Vol, 1=Normal, 2=High Vol)
            
        Returns:
        --------
        Dict
            Performance metrics by regime
        """
        regime_performance = {}
        regime_names = {0: 'Low Volatility', 1: 'Normal', 2: 'High Volatility'}
        
        for regime in [0, 1, 2]:
            mask = regimes == regime
            if mask.sum() == 0:
                continue
            
            regime_returns = returns[mask]
            cumulative_returns = (1 + regime_returns).cumprod()
            total_return = cumulative_returns.iloc[-1] - 1 if len(regime_returns) > 0 else 0
            annual_return = (1 + total_return) ** (252 / len(regime_returns)) - 1 if len(regime_returns) > 0 else 0
            annual_vol = regime_returns.std() * np.sqrt(252)
            sharpe = (annual_return - self.risk_free_rate) / annual_vol if annual_vol > 0 else 0
            
            regime_performance[regime_names[regime]] = {
                'Days': mask.sum(),
                'Total Return': total_return,
                'Annual Return': annual_return,
                'Volatility': annual_vol,
                'Sharpe Ratio': sharpe,
                'Win Rate': len(regime_returns[regime_returns > 0]) / len(regime_returns[regime_returns != 0]) if len(regime_returns[regime_returns != 0]) > 0 else 0,
            }
        
        return regime_performance
    
    @staticmethod
    def _calculate_drawdown_duration(drawdown: pd.Series) -> int:
        """Calculate maximum drawdown duration in days"""
        drawdown_periods = (drawdown != 0).astype(int)
        drawdown_changes = drawdown_periods.diff()
        
        start_indices = np.where(drawdown_changes == 1)[0]
        end_indices = np.where(drawdown_changes == -1)[0]
        
        if len(start_indices) == 0:
            return 0
        
        if len(end_indices) == 0:
            end_indices = np.array([len(drawdown) - 1])
        
        max_duration = 0
        for start in start_indices:
            end = end_indices[end_indices > start]
            if len(end) > 0:
                duration = end[0] - start
                max_duration = max(max_duration, duration)
        
        return int(max_duration)
    
    @staticmethod
    def _calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float) -> float:
        """Calculate Sortino ratio (uses downside deviation)"""
        excess_returns = returns - risk_free_rate
        downside_returns = excess_returns[excess_returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252)
        
        annual_return = excess_returns.mean() * 252
        
        if downside_std == 0:
            return 0
        
        return annual_return / downside_std
    
    def compare_strategies(self, strategies_dict: Dict[str, pd.Series]) -> pd.DataFrame:
        """
        Compare multiple strategies
        
        Parameters:
        -----------
        strategies_dict : Dict[str, pd.Series]
            Dictionary with strategy names and their returns
            
        Returns:
        --------
        pd.DataFrame
            Comparison of all metrics across strategies
        """
        results = {}
        
        for strategy_name, returns in strategies_dict.items():
            results[strategy_name] = self.backtest_strategy(returns)
        
        return pd.DataFrame(results).T


class RollingPerformance:
    """Analyze rolling performance metrics"""
    
    @staticmethod
    def calculate_rolling_sharpe(returns: pd.Series, window: int = 60, 
                                 risk_free_rate: float = 0.04) -> pd.Series:
        """
        Calculate rolling Sharpe ratio
        
        Parameters:
        -----------
        returns : pd.Series
            Daily returns
        window : int
            Rolling window in days
        risk_free_rate : float
            Annual risk-free rate
            
        Returns:
        --------
        pd.Series
            Rolling Sharpe ratio
        """
        rolling_return = returns.rolling(window).mean() * 252
        rolling_vol = returns.rolling(window).std() * np.sqrt(252)
        rolling_sharpe = (rolling_return - risk_free_rate) / rolling_vol
        
        return rolling_sharpe
    
    @staticmethod
    def calculate_rolling_max_drawdown(returns: pd.Series, window: int = 60) -> pd.Series:
        """Calculate rolling maximum drawdown"""
        cumulative = (1 + returns).cumprod()
        
        rolling_max_dd = pd.Series(index=returns.index, dtype=float)
        
        for i in range(len(returns)):
            if i < window:
                continue
            
            window_cumulative = cumulative.iloc[i-window:i+1]
            running_max = window_cumulative.expanding().max()
            drawdowns = (window_cumulative - running_max) / running_max
            rolling_max_dd.iloc[i] = drawdowns.min()
        
        return rolling_max_dd
    
    @staticmethod
    def calculate_rolling_return(returns: pd.Series, window: int = 60) -> pd.Series:
        """Calculate rolling cumulative return"""
        return (1 + returns).rolling(window).apply(lambda x: x.prod() - 1)


class StressTestAnalyzer:
    """Analyze strategy performance under stress scenarios"""
    
    @staticmethod
    def analyze_tail_periods(returns: pd.Series, vix: pd.Series, 
                            vix_threshold_pct: float = 90) -> Dict:
        """
        Analyze performance during high volatility periods
        
        Parameters:
        -----------
        returns : pd.Series
            Strategy returns
        vix : pd.Series
            VIX index
        vix_threshold_pct : float
            Percentile threshold for high VIX
            
        Returns:
        --------
        Dict
            Performance during tail events
        """
        vix_threshold = np.percentile(vix, vix_threshold_pct)
        tail_mask = vix > vix_threshold
        
        normal_returns = returns[~tail_mask]
        tail_returns = returns[tail_mask]
        
        return {
            'Normal Period Performance': {
                'Days': (~tail_mask).sum(),
                'Total Return': (1 + normal_returns).prod() - 1,
                'Sharpe Ratio': (normal_returns.mean() * 252 - 0.04) / (normal_returns.std() * np.sqrt(252)),
                'Max Drawdown': ((1 + normal_returns).cumprod() / (1 + normal_returns).cumprod().expanding().max() - 1).min(),
            },
            'Tail Period Performance': {
                'Days': tail_mask.sum(),
                'Total Return': (1 + tail_returns).prod() - 1,
                'Sharpe Ratio': (tail_returns.mean() * 252 - 0.04) / (tail_returns.std() * np.sqrt(252)) if tail_returns.std() > 0 else 0,
                'Max Drawdown': ((1 + tail_returns).cumprod() / (1 + tail_returns).cumprod().expanding().max() - 1).min(),
            }
        }
    
    @staticmethod
    def monte_carlo_simulation(returns: pd.Series, n_simulations: int = 1000, 
                               n_periods: int = 252) -> Dict:
        """
        Run Monte Carlo simulation of strategy
        
        Parameters:
        -----------
        returns : pd.Series
            Historical returns
        n_simulations : int
            Number of simulations
        n_periods : int
            Number of periods to simulate
            
        Returns:
        --------
        Dict
            Simulation statistics
        """
        mu = returns.mean()
        sigma = returns.std()
        
        simulations = np.random.normal(mu, sigma, (n_simulations, n_periods))
        cumulative_simulations = (1 + simulations).cumprod(axis=1)
        
        return {
            'Final Value P5': np.percentile(cumulative_simulations[:, -1], 5),
            'Final Value P25': np.percentile(cumulative_simulations[:, -1], 25),
            'Final Value P50': np.percentile(cumulative_simulations[:, -1], 50),
            'Final Value P75': np.percentile(cumulative_simulations[:, -1], 75),
            'Final Value P95': np.percentile(cumulative_simulations[:, -1], 95),
            'Worst Case': cumulative_simulations[:, -1].min(),
            'Best Case': cumulative_simulations[:, -1].max(),
        }


if __name__ == '__main__':
    print("Backtesting module loaded successfully")
