"""
Backtesting framework for portfolio strategies with transaction costs.
"""
import numpy as np
import pandas as pd
from datetime import datetime


class PortfolioBacktester:
    """
    Backtest portfolio strategies with rebalancing and transaction costs.
    """
    
    def __init__(self, prices, initial_capital=100000):
        """
        Initialize the backtester.
        
        Parameters
        ----------
        prices : pd.DataFrame
            DataFrame with asset prices indexed by date
        initial_capital : float
            Initial portfolio value
        """
        self.prices = prices
        self.initial_capital = initial_capital
        self.dates = prices.index
        self.n_assets = len(prices.columns)
        self.asset_names = prices.columns.tolist()
        
        self.portfolio_values = []
        self.portfolio_weights = []
        self.trades = []
        self.returns = []
        
    def run_backtest(self, weights_rebalance, rebalance_frequency='monthly', 
                     transaction_cost=0.001, slippage=0.0):
        """
        Run backtest with given weights schedule.
        
        Parameters
        ----------
        weights_rebalance : dict or callable
            If dict: {date: weights}
            If callable: function(current_date, prices) -> weights
        rebalance_frequency : str
            'daily', 'weekly', 'monthly', 'quarterly', 'yearly'
        transaction_cost : float
            Transaction cost as percentage (e.g., 0.001 = 0.1%)
        slippage : float
            Slippage as percentage
        
        Returns
        -------
        pd.DataFrame
            Backtest results
        """
        # Get rebalance dates
        if isinstance(weights_rebalance, dict):
            rebalance_dates = sorted(weights_rebalance.keys())
        else:
            rebalance_dates = self._get_rebalance_dates(rebalance_frequency)
        
        # Initialize portfolio
        current_weights = np.ones(self.n_assets) / self.n_assets
        current_value = self.initial_capital
        current_shares = current_value * current_weights / self.prices.iloc[0].values
        
        self.portfolio_values = [current_value]
        self.portfolio_weights = [current_weights.copy()]
        self.trades = []
        
        rebalance_idx = 0
        
        for t in range(1, len(self.prices)):
            current_date = self.dates[t]
            prev_date = self.dates[t-1]
            
            # Update portfolio value
            current_prices = self.prices.iloc[t].values
            current_value = np.dot(current_shares, current_prices)
            current_weights = (current_shares * current_prices) / current_value
            
            self.portfolio_values.append(current_value)
            self.portfolio_weights.append(current_weights.copy())
            
            # Check if rebalance needed
            rebalance = False
            
            if isinstance(weights_rebalance, dict):
                if current_date in weights_rebalance:
                    target_weights = np.array(weights_rebalance[current_date])
                    rebalance = True
            else:
                # Callable weights
                if rebalance_idx < len(rebalance_dates) and current_date >= rebalance_dates[rebalance_idx]:
                    target_weights = weights_rebalance(current_date, self.prices.iloc[:t])
                    rebalance = True
                    rebalance_idx += 1
            
            # Execute rebalance
            if rebalance:
                # Calculate target shares
                target_shares = current_value * target_weights / current_prices
                
                # Calculate transaction costs
                share_changes = target_shares - current_shares
                cost_basis = np.abs(share_changes) * current_prices
                costs = cost_basis * (transaction_cost + slippage)
                total_cost = costs.sum()
                
                # Reduce portfolio value by costs
                current_value -= total_cost
                
                # Record trade
                self.trades.append({
                    'date': current_date,
                    'old_weights': current_weights.copy(),
                    'new_weights': target_weights,
                    'transaction_cost': total_cost,
                    'portfolio_value_before': current_value + total_cost,
                    'portfolio_value_after': current_value
                })
                
                # Update shares
                current_shares = current_value * target_weights / current_prices
        
        return self._compile_results()
    
    def run_buy_and_hold(self, initial_weights, transaction_cost=0.001):
        """
        Run simple buy-and-hold backtest.
        
        Parameters
        ----------
        initial_weights : np.ndarray
            Initial portfolio weights
        transaction_cost : float
            Initial transaction cost
        
        Returns
        -------
        pd.DataFrame
            Backtest results
        """
        # Dictionary with single rebalance at first date
        weights_dict = {self.dates[0]: initial_weights}
        
        return self.run_backtest(weights_dict, transaction_cost=transaction_cost)
    
    def _get_rebalance_dates(self, frequency):
        """Get rebalance dates based on frequency."""
        if frequency == 'daily':
            return self.dates
        elif frequency == 'weekly':
            return self.dates[self.dates.dayofweek == 4]  # Fridays
        elif frequency == 'monthly':
            return self.dates[self.dates.is_month_end]
        elif frequency == 'quarterly':
            months = self.dates.month
            quarters = (months - 1) // 3
            is_quarter_end = (months % 3 == 0)
            return self.dates[is_quarter_end]
        elif frequency == 'yearly':
            return self.dates[self.dates.is_year_end]
        else:
            return self.dates
    
    def _compile_results(self):
        """Compile backtest results into DataFrame."""
        self.portfolio_values = np.array(self.portfolio_values)
        daily_returns = np.diff(self.portfolio_values) / self.portfolio_values[:-1]
        
        results = pd.DataFrame({
            'date': self.dates,
            'portfolio_value': self.portfolio_values,
            'daily_return': np.concatenate([[np.nan], daily_returns])
        })
        
        return results
    
    def calculate_metrics(self, results):
        """
        Calculate backtest performance metrics.
        
        Parameters
        ----------
        results : pd.DataFrame
            Backtest results
        
        Returns
        -------
        dict
            Performance metrics
        """
        returns = results['daily_return'].dropna()
        portfolio_vals = results['portfolio_value'].values
        
        total_return = (portfolio_vals[-1] - self.initial_capital) / self.initial_capital
        annual_return = total_return ** (252 / len(returns)) - 1
        annual_vol = returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / (annual_vol + 1e-8)
        
        # Drawdown
        cumulative = (1 + returns).cumprod()
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Information ratio (relative to buy-and-hold)
        days_held = len(returns)
        cumulative_return = (1 + returns).prod() - 1
        
        metrics = {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_vol,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'days_held': days_held,
            'final_value': portfolio_vals[-1],
            'transactions': len(self.trades)
        }
        
        return metrics
    
    def get_weights_history(self):
        """Get portfolio weights over time."""
        weights_df = pd.DataFrame(
            self.portfolio_weights,
            columns=self.asset_names,
            index=self.dates
        )
        return weights_df
