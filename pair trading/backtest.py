"""
Backtesting framework for pairs trading strategies.

Features:
- Portfolio tracking
- P&L calculation
- Performance metrics
- Transaction costs
- Risk metrics
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Represents a trading position."""
    symbol1: str
    symbol2: str
    entry_date: datetime
    entry_price1: float
    entry_price2: float
    hedge_ratio: float
    quantity1: int
    quantity2: int
    direction: int  # 1 for long, -1 for short
    
    def __repr__(self):
        return f"Pos({self.symbol1}/{self.symbol2} {self.direction} @ {self.entry_date})"


@dataclass
class Trade:
    """Represents a completed trade."""
    symbol1: str
    symbol2: str
    entry_date: datetime
    exit_date: datetime
    entry_price1: float
    exit_price1: float
    entry_price2: float
    exit_price2: float
    hedge_ratio: float
    quantity1: int
    quantity2: int
    direction: int
    pnl: float
    pnl_pct: float
    holding_period: int


class BacktestEngine:
    """Backtesting engine for pairs trading."""
    
    def __init__(self, initial_capital: float = 100000,
                 transaction_cost: float = 0.001,  # 0.1%
                 slippage: float = 0.0005):  # 0.05%
        """
        Parameters:
        -----------
        initial_capital : float
            Starting capital
        transaction_cost : float
            Transaction cost as fraction (e.g., 0.001 = 0.1%)
        slippage : float
            Slippage as fraction of price
        """
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        
        # State variables
        self.current_capital = initial_capital
        self.positions: List[Position] = []
        self.trades: List[Trade] = []
        self.portfolio_values = []
        self.cash_values = []
        self.timestamps = []
    
    def calculate_position_value(self, prices1: float, prices2: float,
                                position: Position) -> float:
        """Calculate current position value."""
        pnl1 = position.quantity1 * (prices1 - position.entry_price1)
        pnl2 = -position.quantity2 * (prices2 - position.entry_price2)
        return pnl1 + pnl2
    
    def open_position(self, date: datetime, price1: float, price2: float,
                     symbol1: str, symbol2: str, hedge_ratio: float,
                     quantity1: int, direction: int = 1) -> float:
        """
        Open a new position.
        
        Returns:
        --------
        Cost of opening position (transaction cost)
        """
        quantity2 = int(quantity1 * hedge_ratio)
        
        cost1 = quantity1 * price1 * (1 + self.slippage) * (1 + self.transaction_cost)
        cost2 = quantity2 * price2 * (1 + self.slippage) * (1 + self.transaction_cost)
        total_cost = cost1 + cost2
        
        if total_cost > self.current_capital:
            logger.warning(f"Insufficient capital for position: {total_cost:.2f}")
            return 0
        
        position = Position(
            symbol1=symbol1,
            symbol2=symbol2,
            entry_date=date,
            entry_price1=price1,
            entry_price2=price2,
            hedge_ratio=hedge_ratio,
            quantity1=quantity1,
            quantity2=quantity2,
            direction=direction
        )
        
        self.positions.append(position)
        self.current_capital -= total_cost
        
        logger.info(f"Opened position: {position}")
        return total_cost
    
    def close_position(self, position_idx: int, date: datetime, 
                      price1: float, price2: float) -> Tuple[float, Trade]:
        """
        Close an existing position.
        
        Returns:
        --------
        Tuple of (proceeds, Trade record)
        """
        position = self.positions[position_idx]
        
        # Calculate P&L
        pnl1 = position.quantity1 * (price1 - position.entry_price1)
        pnl2 = position.quantity2 * (position.entry_price2 - price2)
        pnl = pnl1 + pnl2
        
        # Calculate proceeds (with transaction cost)
        proceeds1 = position.quantity1 * price1 * (1 - self.slippage) * (1 - self.transaction_cost)
        proceeds2 = position.quantity2 * price2 * (1 - self.slippage) * (1 - self.transaction_cost)
        proceeds = proceeds1 + proceeds2
        
        # Calculate returns
        cost = position.quantity1 * position.entry_price1 + \
               position.quantity2 * position.entry_price2
        pnl_pct = pnl / cost if cost > 0 else 0
        
        # Holding period in days
        holding_period = (date - position.entry_date).days
        
        # Create trade record
        trade = Trade(
            symbol1=position.symbol1,
            symbol2=position.symbol2,
            entry_date=position.entry_date,
            exit_date=date,
            entry_price1=position.entry_price1,
            exit_price1=price1,
            entry_price2=position.entry_price2,
            exit_price2=price2,
            hedge_ratio=position.hedge_ratio,
            quantity1=position.quantity1,
            quantity2=position.quantity2,
            direction=position.direction,
            pnl=pnl,
            pnl_pct=pnl_pct,
            holding_period=holding_period
        )
        
        self.trades.append(trade)
        self.current_capital += proceeds
        self.positions.pop(position_idx)
        
        logger.info(f"Closed position: {trade.symbol1}/{trade.symbol2} PnL: {pnl:.2f} ({pnl_pct:.2%})")
        return proceeds, trade
    
    def run_backtest(self, prices1: pd.DataFrame, prices2: pd.DataFrame,
                    signals: pd.DataFrame, hedge_ratio: float = 1.0,
                    symbol1: str = "Asset1", symbol2: str = "Asset2",
                    quantity1: int = 100) -> Dict:
        """
        Run backtest on price data and signals.
        
        Parameters:
        -----------
        prices1, prices2 : pd.DataFrame
            OHLCV data with DatetimeIndex
        signals : pd.DataFrame
            Trading signals from signal generator
        hedge_ratio : float
            Hedge ratio for position sizing
        symbol1, symbol2 : str
            Asset names
        quantity1 : int
            Quantity of first asset
        
        Returns:
        --------
        Dictionary with backtest results
        """
        # Align data
        common_dates = prices1.index.intersection(prices2.index).intersection(signals.index)
        
        prices1_aligned = prices1.loc[common_dates, 'Close']
        prices2_aligned = prices2.loc[common_dates, 'Close']
        signals_aligned = signals.loc[common_dates]
        
        # Track portfolio
        portfolio_values = []
        cash_values = []
        position_values = []
        
        for i, date in enumerate(common_dates):
            # Get prices
            price1 = prices1_aligned.iloc[i]
            price2 = prices2_aligned.iloc[i]
            signal = signals_aligned.iloc[i].get('signal', 0)
            
            # Update position value
            pv = sum(self.calculate_position_value(price1, price2, pos) 
                    for pos in self.positions)
            position_values.append(pv)
            
            # Calculate portfolio value
            portfolio_value = self.current_capital + pv
            portfolio_values.append(portfolio_value)
            cash_values.append(self.current_capital)
            
            # Signal handling
            if signal == 1 and len(self.positions) == 0:  # BUY signal
                self.open_position(date, price1, price2, symbol1, symbol2, 
                                 hedge_ratio, quantity1, direction=1)
            
            elif signal == -1 and len(self.positions) == 0:  # SELL signal
                self.open_position(date, price1, price2, symbol1, symbol2,
                                 hedge_ratio, quantity1, direction=-1)
            
            elif signal == 0 and len(self.positions) > 0:  # EXIT signal
                for pos_idx in range(len(self.positions) - 1, -1, -1):
                    self.close_position(pos_idx, date, price1, price2)
            
            self.timestamps.append(date)
        
        # Compile results
        results = {
            'portfolio_values': portfolio_values,
            'cash_values': cash_values,
            'timestamps': self.timestamps,
            'trades': self.trades,
            'final_value': portfolio_values[-1] if portfolio_values else self.initial_capital,
            'total_return': (portfolio_values[-1] - self.initial_capital) / self.initial_capital if portfolio_values else 0
        }
        
        return results
    
    def get_performance_metrics(self, results: Dict) -> Dict:
        """Calculate performance metrics from backtest results."""
        portfolio_values = np.array(results['portfolio_values'])
        timestamps = pd.DatetimeIndex(results['timestamps'])
        
        # Returns
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        # Metrics
        total_return = (portfolio_values[-1] - self.initial_capital) / self.initial_capital
        annual_return = (portfolio_values[-1] / self.initial_capital) ** (252 / len(returns)) - 1
        
        # Volatility
        daily_vol = np.std(returns)
        annual_vol = daily_vol * np.sqrt(252)
        
        # Sharpe ratio (assuming 0% risk-free rate)
        sharpe = annual_return / annual_vol if annual_vol > 0 else 0
        
        # Max drawdown
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        # Win rate
        trades = results.get('trades', [])
        if trades:
            wins = sum(1 for t in trades if t.pnl > 0)
            win_rate = wins / len(trades)
            avg_win = np.mean([t.pnl for t in trades if t.pnl > 0]) if wins > 0 else 0
            avg_loss = np.mean([t.pnl for t in trades if t.pnl < 0]) if (len(trades) - wins) > 0 else 0
            profit_factor = abs(avg_win * wins / (avg_loss * (len(trades) - wins))) if avg_loss != 0 else np.inf
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_vol,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'num_trades': len(trades),
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_trade_pnl': np.mean([t.pnl for t in trades]) if trades else 0,
            'avg_holding_period': np.mean([t.holding_period for t in trades]) if trades else 0
        }


class PortfolioAnalyzer:
    """Analyze portfolio performance and risk."""
    
    @staticmethod
    def calculate_var(returns: pd.Series, confidence: float = 0.95) -> float:
        """Calculate Value at Risk."""
        return np.percentile(returns, (1 - confidence) * 100)
    
    @staticmethod
    def calculate_cvar(returns: pd.Series, confidence: float = 0.95) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)."""
        var = PortfolioAnalyzer.calculate_var(returns, confidence)
        return returns[returns <= var].mean()
    
    @staticmethod
    def calculate_calmar_ratio(returns: pd.Series, annual_return: float) -> float:
        """Calculate Calmar ratio (annual return / max drawdown)."""
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = abs(np.min(drawdown))
        
        return annual_return / max_drawdown if max_drawdown > 0 else 0
    
    @staticmethod
    def calculate_sortino_ratio(returns: pd.Series, target_return: float = 0, 
                               periods_per_year: int = 252) -> float:
        """Calculate Sortino ratio (return / downside volatility)."""
        excess_returns = returns - target_return / periods_per_year
        downside_std = np.sqrt(np.mean(np.minimum(excess_returns, 0) ** 2))
        annual_return = np.mean(returns) * periods_per_year
        
        return annual_return / (downside_std * np.sqrt(periods_per_year)) if downside_std > 0 else 0
