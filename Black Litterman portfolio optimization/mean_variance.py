"""
Mean-Variance Portfolio Optimization using Modern Portfolio Theory.
"""
import numpy as np
from scipy.optimize import minimize, LinearConstraint, Bounds
import warnings

warnings.filterwarnings('ignore')


class MeanVarianceOptimizer:
    """
    Mean-Variance portfolio optimization using Markowitz theory.
    """
    
    def __init__(self, expected_returns, cov_matrix, risk_free_rate=0.02):
        """
        Initialize the optimizer.
        
        Parameters
        ----------
        expected_returns : np.ndarray
            Vector of expected returns (n_assets,)
        cov_matrix : np.ndarray
            Covariance matrix (n_assets, n_assets)
        risk_free_rate : float
            Risk-free rate (default 2%)
        """
        self.expected_returns = expected_returns
        self.cov_matrix = cov_matrix
        self.risk_free_rate = risk_free_rate
        self.n_assets = len(expected_returns)
        
    def portfolio_performance(self, weights):
        """
        Calculate portfolio return, volatility, and Sharpe ratio.
        
        Parameters
        ----------
        weights : np.ndarray
            Portfolio weights
        
        Returns
        -------
        tuple
            (return, volatility, sharpe_ratio)
        """
        portfolio_return = np.dot(weights, self.expected_returns)
        portfolio_std = np.sqrt(np.dot(weights, np.dot(self.cov_matrix, weights)))
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / (portfolio_std + 1e-8)
        
        return portfolio_return, portfolio_std, sharpe_ratio
    
    def negative_sharpe(self, weights):
        """Objective function: negative Sharpe ratio (for minimization)."""
        return -self.portfolio_performance(weights)[2]
    
    def portfolio_volatility(self, weights):
        """Objective function: portfolio volatility."""
        return np.sqrt(np.dot(weights, np.dot(self.cov_matrix, weights)))
    
    def optimize_max_sharpe(self, constraints=None, bounds=None):
        """
        Find maximum Sharpe ratio portfolio.
        
        Parameters
        ----------
        constraints : dict, optional
            Optimization constraints
        bounds : list, optional
            Bounds for weights
        
        Returns
        -------
        dict
            Optimization result with weights and metrics
        """
        if constraints is None:
            constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        
        if bounds is None:
            bounds = Bounds(0, 1)  # Long-only portfolio
        
        x0 = np.array([1 / self.n_assets] * self.n_assets)
        
        result = minimize(
            self.negative_sharpe,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        weights = result.x
        ret, vol, sharpe = self.portfolio_performance(weights)
        
        return {
            'weights': weights,
            'return': ret,
            'volatility': vol,
            'sharpe_ratio': sharpe,
            'success': result.success
        }
    
    def optimize_min_variance(self, constraints=None, bounds=None):
        """
        Find minimum variance portfolio.
        
        Parameters
        ----------
        constraints : dict, optional
            Optimization constraints
        bounds : list, optional
            Bounds for weights
        
        Returns
        -------
        dict
            Optimization result with weights and metrics
        """
        if constraints is None:
            constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        
        if bounds is None:
            bounds = Bounds(0, 1)  # Long-only portfolio
        
        x0 = np.array([1 / self.n_assets] * self.n_assets)
        
        result = minimize(
            self.portfolio_volatility,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        weights = result.x
        ret, vol, sharpe = self.portfolio_performance(weights)
        
        return {
            'weights': weights,
            'return': ret,
            'volatility': vol,
            'sharpe_ratio': sharpe,
            'success': result.success
        }
    
    def optimize_target_return(self, target_return, constraints=None, bounds=None):
        """
        Find minimum variance portfolio with target return.
        
        Parameters
        ----------
        target_return : float
            Target return level
        constraints : dict, optional
            Additional constraints
        bounds : list, optional
            Bounds for weights
        
        Returns
        -------
        dict
            Optimization result with weights and metrics
        """
        if bounds is None:
            bounds = Bounds(0, 1)
        
        if constraints is None:
            constraints = []
        else:
            constraints = [constraints] if isinstance(constraints, dict) else constraints
        
        constraints.append({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        constraints.append({'type': 'eq', 'fun': lambda x: np.dot(x, self.expected_returns) - target_return})
        
        x0 = np.array([1 / self.n_assets] * self.n_assets)
        
        result = minimize(
            self.portfolio_volatility,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        if result.success:
            weights = result.x
            ret, vol, sharpe = self.portfolio_performance(weights)
            return {
                'weights': weights,
                'return': ret,
                'volatility': vol,
                'sharpe_ratio': sharpe,
                'success': True
            }
        else:
            return {
                'weights': None,
                'return': None,
                'volatility': None,
                'sharpe_ratio': None,
                'success': False
            }
    
    def efficient_frontier(self, n_points=100, constraints=None, bounds=None):
        """
        Calculate efficient frontier.
        
        Parameters
        ----------
        n_points : int
            Number of points on the frontier
        constraints : dict, optional
            Optimization constraints
        bounds : list, optional
            Bounds for weights
        
        Returns
        -------
        tuple
            (returns, volatilities, weights_list)
        """
        if bounds is None:
            bounds = Bounds(0, 1)
        
        min_return = self.expected_returns.min()
        max_return = self.expected_returns.max()
        
        target_returns = np.linspace(min_return, max_return, n_points)
        
        frontier_returns = []
        frontier_vols = []
        frontier_weights = []
        
        for target in target_returns:
            result = self.optimize_target_return(target, constraints, bounds)
            if result['success']:
                frontier_returns.append(result['return'])
                frontier_vols.append(result['volatility'])
                frontier_weights.append(result['weights'])
        
        return (
            np.array(frontier_returns),
            np.array(frontier_vols),
            frontier_weights
        )
