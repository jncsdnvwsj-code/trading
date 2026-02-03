"""
Scenario analysis for stress testing portfolios under varying market conditions.
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


class ScenarioAnalyzer:
    """
    Stress test portfolios under various market scenarios.
    """
    
    def __init__(self, assets, expected_returns, cov_matrix):
        """
        Initialize scenario analyzer.
        
        Parameters
        ----------
        assets : list
            Asset names
        expected_returns : np.ndarray
            Expected returns
        cov_matrix : np.ndarray
            Covariance matrix
        """
        self.assets = assets
        self.expected_returns = expected_returns
        self.cov_matrix = cov_matrix
        self.n_assets = len(assets)
    
    def normal_scenario(self, weights, periods=252):
        """
        Simulate normal market conditions.
        
        Parameters
        ----------
        weights : np.ndarray
            Portfolio weights
        periods : int
            Number of periods to simulate
        
        Returns
        -------
        dict
            Scenario results
        """
        portfolio_return = np.dot(weights, self.expected_returns)
        portfolio_vol = np.sqrt(np.dot(weights, np.dot(self.cov_matrix, weights)))
        
        # Generate returns
        returns = np.random.normal(portfolio_return / 252, portfolio_vol / np.sqrt(252), periods)
        cumulative = (1 + returns).cumprod()
        
        return {
            'name': 'Normal',
            'returns': returns,
            'cumulative': cumulative,
            'final_value': cumulative[-1],
            'max_return': returns.max(),
            'min_return': returns.min(),
            'avg_return': returns.mean(),
            'volatility': returns.std()
        }
    
    def bull_market_scenario(self, weights, periods=252, shift=0.5):
        """
        Simulate bull market (positive shock).
        
        Parameters
        ----------
        weights : np.ndarray
            Portfolio weights
        periods : int
            Number of periods
        shift : float
            Return shift factor (e.g., 0.5 for 50% increase)
        
        Returns
        -------
        dict
            Scenario results
        """
        bull_returns = self.expected_returns * (1 + shift)
        portfolio_return = np.dot(weights, bull_returns)
        portfolio_vol = np.sqrt(np.dot(weights, np.dot(self.cov_matrix, weights)))
        
        returns = np.random.normal(portfolio_return / 252, portfolio_vol / np.sqrt(252), periods)
        cumulative = (1 + returns).cumprod()
        
        return {
            'name': 'Bull Market',
            'returns': returns,
            'cumulative': cumulative,
            'final_value': cumulative[-1],
            'max_return': returns.max(),
            'min_return': returns.min(),
            'avg_return': returns.mean(),
            'volatility': returns.std()
        }
    
    def bear_market_scenario(self, weights, periods=252, shift=-0.5):
        """
        Simulate bear market (negative shock).
        
        Parameters
        ----------
        weights : np.ndarray
            Portfolio weights
        periods : int
            Number of periods
        shift : float
            Return shift factor (e.g., -0.5 for 50% decrease)
        
        Returns
        -------
        dict
            Scenario results
        """
        bear_returns = self.expected_returns * (1 + shift)
        bear_returns = np.maximum(bear_returns, -0.3)  # Cap at -30%
        portfolio_return = np.dot(weights, bear_returns)
        portfolio_vol = np.sqrt(np.dot(weights, np.dot(self.cov_matrix, weights)))
        
        returns = np.random.normal(portfolio_return / 252, portfolio_vol / np.sqrt(252), periods)
        cumulative = (1 + returns).cumprod()
        
        return {
            'name': 'Bear Market',
            'returns': returns,
            'cumulative': cumulative,
            'final_value': cumulative[-1],
            'max_return': returns.max(),
            'min_return': returns.min(),
            'avg_return': returns.mean(),
            'volatility': returns.std()
        }
    
    def volatility_spike_scenario(self, weights, periods=252, vol_multiplier=2.0):
        """
        Simulate volatility spike.
        
        Parameters
        ----------
        weights : np.ndarray
            Portfolio weights
        periods : int
            Number of periods
        vol_multiplier : float
            Volatility multiplier (e.g., 2.0 for 2x volatility)
        
        Returns
        -------
        dict
            Scenario results
        """
        portfolio_return = np.dot(weights, self.expected_returns)
        portfolio_vol = np.sqrt(np.dot(weights, np.dot(self.cov_matrix, weights))) * vol_multiplier
        
        returns = np.random.normal(portfolio_return / 252, portfolio_vol / np.sqrt(252), periods)
        cumulative = (1 + returns).cumprod()
        
        return {
            'name': f'Volatility Spike ({vol_multiplier}x)',
            'returns': returns,
            'cumulative': cumulative,
            'final_value': cumulative[-1],
            'max_return': returns.max(),
            'min_return': returns.min(),
            'avg_return': returns.mean(),
            'volatility': returns.std()
        }
    
    def crisis_scenario(self, weights, periods=252, correlation_shock=0.5):
        """
        Simulate financial crisis with correlation increase.
        
        Parameters
        ----------
        weights : np.ndarray
            Portfolio weights
        periods : int
            Number of periods
        correlation_shock : float
            Additional correlation (0-1)
        
        Returns
        -------
        dict
            Scenario results
        """
        # Increase all correlations
        crisis_cov = self.cov_matrix.copy()
        std_devs = np.sqrt(np.diag(crisis_cov))
        corr = crisis_cov / np.outer(std_devs, std_devs)
        corr = np.minimum(corr + correlation_shock, 0.99)
        crisis_cov = np.diag(std_devs) @ corr @ np.diag(std_devs)
        
        # Lower returns
        crisis_returns = self.expected_returns * 0.3
        portfolio_return = np.dot(weights, crisis_returns)
        portfolio_vol = np.sqrt(np.dot(weights, np.dot(crisis_cov, weights)))
        
        returns = np.random.normal(portfolio_return / 252, portfolio_vol / np.sqrt(252), periods)
        cumulative = (1 + returns).cumprod()
        
        return {
            'name': 'Crisis',
            'returns': returns,
            'cumulative': cumulative,
            'final_value': cumulative[-1],
            'max_return': returns.max(),
            'min_return': returns.min(),
            'avg_return': returns.mean(),
            'volatility': returns.std()
        }
    
    def sector_rotation_scenario(self, weights, periods=252):
        """
        Simulate sector rotation (some assets outperform, others underperform).
        
        Parameters
        ----------
        weights : np.ndarray
            Portfolio weights
        periods : int
            Number of periods
        
        Returns
        -------
        dict
            Scenario results
        """
        # Randomly assign rotation (half outperform, half underperform)
        rotation_factors = np.ones(self.n_assets)
        rotation_factors[:self.n_assets // 2] *= 1.5
        rotation_factors[self.n_assets // 2:] *= 0.5
        
        rotated_returns = self.expected_returns * rotation_factors
        portfolio_return = np.dot(weights, rotated_returns)
        portfolio_vol = np.sqrt(np.dot(weights, np.dot(self.cov_matrix, weights)))
        
        returns = np.random.normal(portfolio_return / 252, portfolio_vol / np.sqrt(252), periods)
        cumulative = (1 + returns).cumprod()
        
        return {
            'name': 'Sector Rotation',
            'returns': returns,
            'cumulative': cumulative,
            'final_value': cumulative[-1],
            'max_return': returns.max(),
            'min_return': returns.min(),
            'avg_return': returns.mean(),
            'volatility': returns.std()
        }
    
    def liquidity_crisis_scenario(self, weights, periods=252, slippage_increase=0.02):
        """
        Simulate liquidity crisis with wider spreads.
        
        Parameters
        ----------
        weights : np.ndarray
            Portfolio weights
        periods : int
            Number of periods
        slippage_increase : float
            Additional slippage cost
        
        Returns
        -------
        dict
            Scenario results
        """
        portfolio_return = np.dot(weights, self.expected_returns) - slippage_increase * 252
        portfolio_vol = np.sqrt(np.dot(weights, np.dot(self.cov_matrix, weights))) * 1.5
        
        returns = np.random.normal(portfolio_return / 252, portfolio_vol / np.sqrt(252), periods)
        cumulative = (1 + returns).cumprod()
        
        return {
            'name': 'Liquidity Crisis',
            'returns': returns,
            'cumulative': cumulative,
            'final_value': cumulative[-1],
            'max_return': returns.max(),
            'min_return': returns.min(),
            'avg_return': returns.mean(),
            'volatility': returns.std()
        }
    
    def run_all_scenarios(self, weights, periods=252, n_simulations=100):
        """
        Run all scenarios for portfolio stress testing.
        
        Parameters
        ----------
        weights : np.ndarray
            Portfolio weights
        periods : int
            Number of periods per scenario
        n_simulations : int
            Number of simulations per scenario
        
        Returns
        -------
        dict
            Summary of all scenarios
        """
        scenarios_results = {}
        
        scenario_methods = [
            self.normal_scenario,
            self.bull_market_scenario,
            self.bear_market_scenario,
            self.volatility_spike_scenario,
            self.crisis_scenario,
            self.sector_rotation_scenario,
            self.liquidity_crisis_scenario
        ]
        
        for method in scenario_methods:
            final_values = []
            for _ in range(n_simulations):
                result = method(weights, periods)
                final_values.append(result['final_value'])
            
            final_values = np.array(final_values)
            
            scenarios_results[result['name']] = {
                'avg_final_value': final_values.mean(),
                'median_final_value': np.median(final_values),
                'std_final_value': final_values.std(),
                'min_final_value': final_values.min(),
                'max_final_value': final_values.max(),
                'worst_case_loss': 1 - final_values.min(),
                'prob_loss': (final_values < 1).mean()
            }
        
        return scenarios_results
