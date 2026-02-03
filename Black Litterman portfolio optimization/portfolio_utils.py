"""
Portfolio utility functions and helpers.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm


def sharpe_ratio(returns, risk_free_rate=0.02):
    """
    Calculate Sharpe ratio.
    
    Parameters
    ----------
    returns : np.ndarray or pd.Series
        Series of returns
    risk_free_rate : float
        Risk-free rate
    
    Returns
    -------
    float
        Sharpe ratio
    """
    excess_returns = returns - risk_free_rate / 252
    return excess_returns.mean() / (excess_returns.std() + 1e-8) * np.sqrt(252)


def information_ratio(strategy_returns, benchmark_returns):
    """
    Calculate information ratio.
    
    Parameters
    ----------
    strategy_returns : np.ndarray or pd.Series
        Strategy returns
    benchmark_returns : np.ndarray or pd.Series
        Benchmark returns
    
    Returns
    -------
    float
        Information ratio
    """
    excess_returns = strategy_returns - benchmark_returns
    return excess_returns.mean() / (excess_returns.std() + 1e-8) * np.sqrt(252)


def sortino_ratio(returns, target_return=0, risk_free_rate=0.02):
    """
    Calculate Sortino ratio (uses downside volatility).
    
    Parameters
    ----------
    returns : np.ndarray or pd.Series
        Series of returns
    target_return : float
        Target return threshold
    risk_free_rate : float
        Risk-free rate
    
    Returns
    -------
    float
        Sortino ratio
    """
    excess_returns = returns - target_return
    downside_returns = np.minimum(excess_returns, 0)
    downside_volatility = np.sqrt(np.mean(downside_returns ** 2))
    return (returns.mean() - risk_free_rate) / (downside_volatility + 1e-8) * np.sqrt(252)


def max_drawdown(returns):
    """
    Calculate maximum drawdown.
    
    Parameters
    ----------
    returns : np.ndarray or pd.Series
        Series of returns
    
    Returns
    -------
    float
        Maximum drawdown (negative)
    """
    cumulative = (1 + returns).cumprod()
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    return drawdown.min()


def calmar_ratio(returns, periods_per_year=252):
    """
    Calculate Calmar ratio (annual return / max drawdown).
    
    Parameters
    ----------
    returns : np.ndarray or pd.Series
        Series of returns
    periods_per_year : int
        Periods per year (252 for daily)
    
    Returns
    -------
    float
        Calmar ratio
    """
    annual_return = returns.mean() * periods_per_year
    max_dd = max_drawdown(returns)
    return annual_return / (abs(max_dd) + 1e-8)


def value_at_risk(returns, confidence_level=0.95):
    """
    Calculate Value at Risk (VaR).
    
    Parameters
    ----------
    returns : np.ndarray or pd.Series
        Series of returns
    confidence_level : float
        Confidence level (e.g., 0.95 for 95%)
    
    Returns
    -------
    float
        VaR at given confidence level
    """
    return np.percentile(returns, (1 - confidence_level) * 100)


def conditional_value_at_risk(returns, confidence_level=0.95):
    """
    Calculate Conditional Value at Risk (CVaR / Expected Shortfall).
    
    Parameters
    ----------
    returns : np.ndarray or pd.Series
        Series of returns
    confidence_level : float
        Confidence level
    
    Returns
    -------
    float
        CVaR at given confidence level
    """
    var = value_at_risk(returns, confidence_level)
    return returns[returns <= var].mean()


def portfolio_variance(weights, cov_matrix):
    """
    Calculate portfolio variance.
    
    Parameters
    ----------
    weights : np.ndarray
        Portfolio weights
    cov_matrix : np.ndarray
        Covariance matrix
    
    Returns
    -------
    float
        Portfolio variance
    """
    return np.dot(weights, np.dot(cov_matrix, weights))


def portfolio_std(weights, cov_matrix):
    """
    Calculate portfolio standard deviation.
    
    Parameters
    ----------
    weights : np.ndarray
        Portfolio weights
    cov_matrix : np.ndarray
        Covariance matrix
    
    Returns
    -------
    float
        Portfolio standard deviation
    """
    return np.sqrt(portfolio_variance(weights, cov_matrix))


def portfolio_return(weights, expected_returns):
    """
    Calculate portfolio expected return.
    
    Parameters
    ----------
    weights : np.ndarray
        Portfolio weights
    expected_returns : np.ndarray
        Expected returns
    
    Returns
    -------
    float
        Portfolio expected return
    """
    return np.dot(weights, expected_returns)


def calculate_turnover(old_weights, new_weights):
    """
    Calculate portfolio turnover.
    
    Parameters
    ----------
    old_weights : np.ndarray
        Previous portfolio weights
    new_weights : np.ndarray
        New portfolio weights
    
    Returns
    -------
    float
        Turnover (sum of absolute changes)
    """
    return np.sum(np.abs(new_weights - old_weights)) / 2


def calculate_tracking_error(strategy_returns, benchmark_returns):
    """
    Calculate tracking error.
    
    Parameters
    ----------
    strategy_returns : np.ndarray or pd.Series
        Strategy returns
    benchmark_returns : np.ndarray or pd.Series
        Benchmark returns
    
    Returns
    -------
    float
        Tracking error (annualized)
    """
    excess_returns = strategy_returns - benchmark_returns
    return excess_returns.std() * np.sqrt(252)


def diversification_ratio(weights, cov_matrix):
    """
    Calculate diversification ratio.
    
    Higher ratio indicates better diversification.
    
    Parameters
    ----------
    weights : np.ndarray
        Portfolio weights
    cov_matrix : np.ndarray
        Covariance matrix
    
    Returns
    -------
    float
        Diversification ratio
    """
    stds = np.sqrt(np.diag(cov_matrix))
    weighted_volatility = np.dot(weights, stds)
    portfolio_volatility = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
    return weighted_volatility / (portfolio_volatility + 1e-8)


def herfindahl_index(weights):
    """
    Calculate Herfindahl-Hirschman Index (concentration measure).
    
    Lower values indicate better diversification.
    Range: [1/N, 1] where N is number of assets
    
    Parameters
    ----------
    weights : np.ndarray
        Portfolio weights
    
    Returns
    -------
    float
        HHI
    """
    return np.sum(weights ** 2)


def effective_n_assets(weights):
    """
    Calculate effective number of assets in portfolio.
    
    Parameters
    ----------
    weights : np.ndarray
        Portfolio weights
    
    Returns
    -------
    float
        Effective number of assets
    """
    n = len(weights)
    hhi = herfindahl_index(weights)
    return 1 / hhi if hhi > 0 else n


def rolling_performance(returns, window=252, metric='sharpe'):
    """
    Calculate rolling performance metrics.
    
    Parameters
    ----------
    returns : pd.Series
        Series of returns
    window : int
        Rolling window size
    metric : str
        'sharpe', 'return', 'volatility'
    
    Returns
    -------
    pd.Series
        Rolling metrics
    """
    if metric == 'sharpe':
        return returns.rolling(window).mean() / returns.rolling(window).std() * np.sqrt(252)
    elif metric == 'return':
        return returns.rolling(window).mean() * 252
    elif metric == 'volatility':
        return returns.rolling(window).std() * np.sqrt(252)


def portfolio_constraint_weights(weights, min_weight=0, max_weight=1):
    """
    Apply weight constraints to portfolio.
    
    Parameters
    ----------
    weights : np.ndarray
        Portfolio weights
    min_weight : float
        Minimum weight per asset
    max_weight : float
        Maximum weight per asset
    
    Returns
    -------
    np.ndarray
        Constrained weights (renormalized to sum to 1)
    """
    constrained = np.clip(weights, min_weight, max_weight)
    return constrained / constrained.sum()


class PerformanceReport:
    """Generate comprehensive performance report."""
    
    def __init__(self, returns, benchmark_returns=None, name='Strategy'):
        """
        Initialize report.
        
        Parameters
        ----------
        returns : pd.Series
            Strategy returns
        benchmark_returns : pd.Series, optional
            Benchmark returns
        name : str
            Strategy name
        """
        self.returns = returns
        self.benchmark_returns = benchmark_returns
        self.name = name
    
    def generate(self):
        """Generate full report."""
        report = {
            'name': self.name,
            'total_return': (1 + self.returns).prod() - 1,
            'annual_return': self.returns.mean() * 252,
            'annual_volatility': self.returns.std() * np.sqrt(252),
            'sharpe_ratio': sharpe_ratio(self.returns),
            'sortino_ratio': sortino_ratio(self.returns),
            'max_drawdown': max_drawdown(self.returns),
            'calmar_ratio': calmar_ratio(self.returns),
            'var_95': value_at_risk(self.returns, 0.95),
            'cvar_95': conditional_value_at_risk(self.returns, 0.95),
        }
        
        if self.benchmark_returns is not None:
            report['information_ratio'] = information_ratio(
                self.returns, self.benchmark_returns
            )
            report['tracking_error'] = calculate_tracking_error(
                self.returns, self.benchmark_returns
            )
        
        return report
    
    def print_report(self):
        """Print formatted report."""
        report = self.generate()
        
        print(f"\nPerformance Report: {report['name']}")
        print("=" * 50)
        print(f"Total Return:        {report['total_return']:>10.2%}")
        print(f"Annual Return:       {report['annual_return']:>10.2%}")
        print(f"Annual Volatility:   {report['annual_volatility']:>10.2%}")
        print(f"Sharpe Ratio:        {report['sharpe_ratio']:>10.4f}")
        print(f"Sortino Ratio:       {report['sortino_ratio']:>10.4f}")
        print(f"Max Drawdown:        {report['max_drawdown']:>10.2%}")
        print(f"Calmar Ratio:        {report['calmar_ratio']:>10.4f}")
        print(f"VaR (95%):           {report['var_95']:>10.2%}")
        print(f"CVaR (95%):          {report['cvar_95']:>10.2%}")
        
        if 'information_ratio' in report:
            print(f"Information Ratio:   {report['information_ratio']:>10.4f}")
            print(f"Tracking Error:      {report['tracking_error']:>10.2%}")
