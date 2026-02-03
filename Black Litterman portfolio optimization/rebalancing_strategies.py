"""
Advanced rebalancing strategies and dynamic weight optimization.
"""
import numpy as np
import pandas as pd
from mean_variance import MeanVarianceOptimizer
from data_utils import calculate_returns, calculate_cov_matrix, calculate_expected_returns


class DynamicRebalancer:
    """
    Dynamic portfolio rebalancing with rolling optimization.
    """
    
    def __init__(self, prices, window_size=252, step_size=20):
        """
        Initialize dynamic rebalancer.
        
        Parameters
        ----------
        prices : pd.DataFrame
            Asset prices
        window_size : int
            Rolling window size for optimization
        step_size : int
            Rebalance frequency (every N days)
        """
        self.prices = prices
        self.window_size = window_size
        self.step_size = step_size
        self.rebalance_dates = []
        self.weights_schedule = {}
    
    def generate_rolling_weights(self):
        """
        Generate weights schedule using rolling optimization.
        
        Returns
        -------
        dict
            Dictionary mapping dates to optimal weights
        """
        n = len(self.prices)
        
        for i in range(self.window_size, n, self.step_size):
            current_date = self.prices.index[i]
            window_prices = self.prices.iloc[i-self.window_size:i]
            
            # Calculate returns and statistics
            returns = calculate_returns(window_prices)
            exp_returns = calculate_expected_returns(returns, annualize=True)
            cov = calculate_cov_matrix(returns, annualize=True)
            
            # Optimize
            optimizer = MeanVarianceOptimizer(exp_returns, cov, risk_free_rate=0.02)
            result = optimizer.optimize_max_sharpe()
            
            self.rebalance_dates.append(current_date)
            self.weights_schedule[current_date] = result['weights']
        
        return self.weights_schedule
    
    def generate_momentum_weights(self, momentum_period=60):
        """
        Generate weights based on momentum strategy.
        
        Parameters
        ----------
        momentum_period : int
            Period for momentum calculation
        
        Returns
        -------
        dict
            Dictionary mapping dates to momentum-based weights
        """
        n = len(self.prices)
        
        for i in range(momentum_period, n, self.step_size):
            current_date = self.prices.index[i]
            momentum_prices = self.prices.iloc[i-momentum_period:i]
            momentum_returns = momentum_prices.iloc[-1] / momentum_prices.iloc[0] - 1
            
            # Allocate more to high momentum assets
            momentum_returns = np.maximum(momentum_returns, 0)
            weights = momentum_returns / (momentum_returns.sum() + 1e-8)
            
            self.rebalance_dates.append(current_date)
            self.weights_schedule[current_date] = weights.values
        
        return self.weights_schedule
    
    def generate_mean_reversion_weights(self, zscore_period=60):
        """
        Generate weights based on mean reversion strategy.
        
        Parameters
        ----------
        zscore_period : int
            Period for Z-score calculation
        
        Returns
        -------
        dict
            Dictionary mapping dates to mean-reversion weights
        """
        n = len(self.prices)
        
        for i in range(zscore_period, n, self.step_size):
            current_date = self.prices.index[i]
            window_prices = self.prices.iloc[i-zscore_period:i]
            
            # Calculate Z-scores
            means = window_prices.mean()
            stds = window_prices.std()
            current_prices = self.prices.iloc[i]
            zscores = (current_prices - means) / stds
            
            # Allocate to oversold (negative Z-score)
            weights = -zscores
            weights = np.maximum(weights, 0)
            weights = weights / (weights.sum() + 1e-8)
            
            self.rebalance_dates.append(current_date)
            self.weights_schedule[current_date] = weights.values
        
        return self.weights_schedule
    
    def get_weights_at_date(self, date):
        """
        Get portfolio weights at specific date.
        
        Parameters
        ----------
        date : pd.Timestamp
            Target date
        
        Returns
        -------
        np.ndarray or None
            Portfolio weights on that date
        """
        # Find closest rebalance date before or on target date
        valid_dates = [d for d in self.rebalance_dates if d <= date]
        if not valid_dates:
            return None
        
        closest_date = max(valid_dates)
        return self.weights_schedule[closest_date]


class ConstrainedRebalancer:
    """
    Rebalancing with portfolio constraints (minimum/maximum weights, turnover limits).
    """
    
    def __init__(self, n_assets, min_weight=0, max_weight=1, max_turnover=0.2):
        """
        Initialize constrained rebalancer.
        
        Parameters
        ----------
        n_assets : int
            Number of assets
        min_weight : float
            Minimum weight per asset
        max_weight : float
            Maximum weight per asset
        max_turnover : float
            Maximum allowed turnover
        """
        self.n_assets = n_assets
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.max_turnover = max_turnover
    
    def get_constraints(self, current_weights=None):
        """
        Get optimization constraints.
        
        Parameters
        ----------
        current_weights : np.ndarray, optional
            Current portfolio weights for turnover constraint
        
        Returns
        -------
        list
            List of constraint dictionaries
        """
        from scipy.optimize import LinearConstraint, Bounds
        
        constraints = []
        
        # Sum to 1 constraint
        constraints.append({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        
        # Turnover constraint
        if current_weights is not None:
            def turnover_constraint(x):
                return self.max_turnover - np.sum(np.abs(x - current_weights)) / 2
            constraints.append({'type': 'ineq', 'fun': turnover_constraint})
        
        return constraints
    
    def get_bounds(self):
        """
        Get weight bounds.
        
        Returns
        -------
        Bounds
            Scipy Bounds object
        """
        from scipy.optimize import Bounds
        return Bounds(self.min_weight, self.max_weight)
    
    def optimize_with_constraints(self, expected_returns, cov_matrix, current_weights=None):
        """
        Optimize portfolio with constraints.
        
        Parameters
        ----------
        expected_returns : np.ndarray
            Expected returns
        cov_matrix : np.ndarray
            Covariance matrix
        current_weights : np.ndarray, optional
            Current weights for turnover constraint
        
        Returns
        -------
        dict
            Optimization result
        """
        optimizer = MeanVarianceOptimizer(expected_returns, cov_matrix, risk_free_rate=0.02)
        constraints = self.get_constraints(current_weights)
        bounds = self.get_bounds()
        
        return optimizer.optimize_max_sharpe(constraints=constraints, bounds=bounds)


class AdaptiveRebalancer:
    """
    Adaptive rebalancing based on market regime detection.
    """
    
    def __init__(self, prices, lookback_period=252):
        """
        Initialize adaptive rebalancer.
        
        Parameters
        ----------
        prices : pd.DataFrame
            Asset prices
        lookback_period : int
            Lookback period for regime detection
        """
        self.prices = prices
        self.lookback_period = lookback_period
    
    def detect_market_regime(self, date_idx):
        """
        Detect current market regime.
        
        Parameters
        ----------
        date_idx : int
            Current index
        
        Returns
        -------
        str
            Regime: 'bull', 'bear', or 'sideways'
        """
        if date_idx < self.lookback_period:
            return 'bull'
        
        window = self.prices.iloc[date_idx-self.lookback_period:date_idx]
        returns = window.pct_change().mean() * 252
        
        avg_return = returns.mean()
        
        if avg_return > 0.1:
            return 'bull'
        elif avg_return < -0.05:
            return 'bear'
        else:
            return 'sideways'
    
    def get_regime_weights(self, expected_returns, cov_matrix, regime):
        """
        Get portfolio weights based on market regime.
        
        Parameters
        ----------
        expected_returns : np.ndarray
            Expected returns
        cov_matrix : np.ndarray
            Covariance matrix
        regime : str
            Market regime
        
        Returns
        -------
        np.ndarray
            Regime-adapted weights
        """
        optimizer = MeanVarianceOptimizer(expected_returns, cov_matrix, risk_free_rate=0.02)
        
        if regime == 'bull':
            # Higher equity allocation
            result = optimizer.optimize_max_sharpe()
        elif regime == 'bear':
            # Conservative allocation
            result = optimizer.optimize_min_variance()
        else:
            # Balanced allocation
            n_assets = len(expected_returns)
            result = {
                'weights': np.ones(n_assets) / n_assets,
                'return': np.dot(np.ones(n_assets) / n_assets, expected_returns),
                'volatility': np.sqrt(np.dot(np.ones(n_assets) / n_assets, 
                                            np.dot(cov_matrix, np.ones(n_assets) / n_assets)))
            }
        
        return result['weights']


class CostAwareRebalancer:
    """
    Rebalancing with explicit transaction cost modeling.
    """
    
    def __init__(self, transaction_cost=0.001, slippage=0.0005):
        """
        Initialize cost-aware rebalancer.
        
        Parameters
        ----------
        transaction_cost : float
            Transaction cost percentage
        slippage : float
            Slippage percentage
        """
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.total_cost = 0
    
    def calculate_rebalancing_cost(self, old_weights, new_weights, portfolio_value):
        """
        Calculate rebalancing cost.
        
        Parameters
        ----------
        old_weights : np.ndarray
            Current weights
        new_weights : np.ndarray
            Target weights
        portfolio_value : float
            Current portfolio value
        
        Returns
        -------
        float
            Rebalancing cost
        """
        changes = np.abs(new_weights - old_weights)
        cost_rate = self.transaction_cost + self.slippage
        total_cost = np.sum(changes) * portfolio_value * cost_rate / 2
        
        self.total_cost += total_cost
        return total_cost
    
    def adjust_weights_for_costs(self, optimal_weights, current_weights, portfolio_value):
        """
        Adjust target weights considering costs.
        
        If cost is too high, reduce the deviation from current weights.
        
        Parameters
        ----------
        optimal_weights : np.ndarray
            Optimal weights from optimization
        current_weights : np.ndarray
            Current weights
        portfolio_value : float
            Current portfolio value
        
        Returns
        -------
        tuple
            (adjusted_weights, cost)
        """
        cost = self.calculate_rebalancing_cost(current_weights, optimal_weights, portfolio_value)
        
        # If cost exceeds 0.5% of portfolio, use smaller moves
        if cost > portfolio_value * 0.005:
            # Move 50% of the way to optimal
            adjusted_weights = current_weights + 0.5 * (optimal_weights - current_weights)
            adjusted_weights = adjusted_weights / adjusted_weights.sum()
            cost = self.calculate_rebalancing_cost(current_weights, adjusted_weights, portfolio_value)
        else:
            adjusted_weights = optimal_weights
        
        return adjusted_weights, cost
