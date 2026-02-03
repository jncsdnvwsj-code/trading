"""
Complete pairs trading strategy pipeline.
"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
import logging
from cointegration import CointegrationAnalyzer, PairsSelector
from signal_generation import (
    MeanReversionSignalGenerator,
    BollingerBandSignalGenerator,
    OrnsteinUhlenbeckSignalGenerator
)
from backtest import BacktestEngine, PortfolioAnalyzer
from live_trading import (
    LiveTradingManager,
    MockPriceDataProvider,
    MockBrokerAPI
)
from utils import load_price_data, validate_data, calculate_half_life

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PairsStrategyPipeline:
    """Complete pairs trading strategy pipeline."""
    
    def __init__(self, 
                 z_score_entry: float = 2.0,
                 z_score_exit: float = 0.5,
                 lookback_window: int = 20,
                 initial_capital: float = 100000,
                 transaction_cost: float = 0.001):
        """
        Parameters:
        -----------
        z_score_entry : float
            Z-score threshold for entry
        z_score_exit : float
            Z-score threshold for exit
        lookback_window : int
            Period for mean/std calculation
        initial_capital : float
            Starting capital for backtest
        transaction_cost : float
            Transaction cost as fraction
        """
        self.z_score_entry = z_score_entry
        self.z_score_exit = z_score_exit
        self.lookback_window = lookback_window
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        
        self.coint_analyzer = CointegrationAnalyzer()
        self.pairs_selector = PairsSelector()
        self.signal_generator = MeanReversionSignalGenerator(
            z_score_entry=z_score_entry,
            z_score_exit=z_score_exit,
            lookback_window=lookback_window
        )
        self.backtest_engine = BacktestEngine(
            initial_capital=initial_capital,
            transaction_cost=transaction_cost
        )
    
    def discover_pairs(self, symbols_data: Dict[str, pd.Series],
                      top_n: int = 10) -> List[Dict]:
        """
        Discover cointegrated pairs from universe of symbols.
        
        Parameters:
        -----------
        symbols_data : Dict[str, pd.Series]
            Dictionary of symbol -> normalized price series
        top_n : int
            Return top N pairs
        
        Returns:
        --------
        List of cointegrated pairs with metadata
        """
        logger.info(f"Discovering cointegrated pairs from {len(symbols_data)} symbols...")
        
        cointegrated = self.coint_analyzer.find_cointegrated_pairs(symbols_data, top_n=top_n*3)
        
        if not cointegrated:
            logger.warning("No cointegrated pairs found")
            return []
        
        best_pairs = self.pairs_selector.select_best_pairs(cointegrated, top_n=top_n)
        
        logger.info(f"Selected {len(best_pairs)} best pairs for trading")
        return best_pairs
    
    def backtest_pair(self, symbol1: str, symbol2: str,
                     prices1: pd.DataFrame, prices2: pd.DataFrame,
                     hedge_ratio: float = 1.0,
                     quantity1: int = 100) -> Dict:
        """
        Backtest a single pair.
        
        Parameters:
        -----------
        symbol1, symbol2 : str
            Asset symbols
        prices1, prices2 : pd.DataFrame
            OHLCV data
        hedge_ratio : float
            Hedge ratio
        quantity1 : int
            Position size for first asset
        
        Returns:
        --------
        Backtest results with performance metrics
        """
        # Align and prepare data
        common_dates = prices1.index.intersection(prices2.index)
        prices1_aligned = prices1.loc[common_dates]
        prices2_aligned = prices2.loc[common_dates]
        
        # Calculate spread
        spread = prices1_aligned['Close'] - (hedge_ratio * prices2_aligned['Close'])
        
        # Generate signals
        signals = self.signal_generator.generate_signals(spread)
        
        # Reset backtest engine
        self.backtest_engine = BacktestEngine(
            initial_capital=self.initial_capital,
            transaction_cost=self.transaction_cost
        )
        
        # Run backtest
        results = self.backtest_engine.run_backtest(
            prices1_aligned,
            prices2_aligned,
            signals,
            hedge_ratio=hedge_ratio,
            symbol1=symbol1,
            symbol2=symbol2,
            quantity1=quantity1
        )
        
        # Calculate metrics
        metrics = self.backtest_engine.get_performance_metrics(results)
        
        results['metrics'] = metrics
        results['pair'] = f"{symbol1}/{symbol2}"
        
        logger.info(f"Backtest {symbol1}/{symbol2}: Return {metrics['total_return']:.2%}, Sharpe {metrics['sharpe_ratio']:.2f}")
        
        return results
    
    def backtest_portfolio(self, pairs: List[Dict],
                          symbols_data: Dict[str, pd.DataFrame]) -> Dict:
        """
        Backtest a portfolio of pairs.
        
        Parameters:
        -----------
        pairs : List[Dict]
            List of pairs to trade
        symbols_data : Dict[str, pd.DataFrame]
            OHLCV data for all symbols
        
        Returns:
        --------
        Portfolio backtest results
        """
        pair_results = []
        
        for pair in pairs:
            symbol1 = pair['symbol1']
            symbol2 = pair['symbol2']
            hedge_ratio = pair['hedge_ratio']
            
            if symbol1 not in symbols_data or symbol2 not in symbols_data:
                logger.warning(f"Missing data for {symbol1}/{symbol2}")
                continue
            
            result = self.backtest_pair(
                symbol1, symbol2,
                symbols_data[symbol1],
                symbols_data[symbol2],
                hedge_ratio=hedge_ratio
            )
            
            pair_results.append(result)
        
        # Aggregate results
        if pair_results:
            total_returns = [r['metrics']['total_return'] for r in pair_results]
            sharpe_ratios = [r['metrics']['sharpe_ratio'] for r in pair_results]
            num_trades = [r['metrics']['num_trades'] for r in pair_results]
            
            portfolio_result = {
                'num_pairs': len(pair_results),
                'avg_return': np.mean(total_returns),
                'avg_sharpe': np.mean(sharpe_ratios),
                'total_trades': sum(num_trades),
                'pairs': pair_results
            }
        else:
            portfolio_result = {'num_pairs': 0, 'avg_return': 0}
        
        return portfolio_result
    
    def setup_live_trading(self, symbols_data: Dict[str, pd.DataFrame],
                          initial_balance: float = 100000) -> LiveTradingManager:
        """
        Setup live trading manager with mock broker.
        
        Parameters:
        -----------
        symbols_data : Dict[str, pd.DataFrame]
            OHLCV data for testing
        initial_balance : float
            Initial account balance
        
        Returns:
        --------
        Configured LiveTradingManager
        """
        # Create mock providers
        price_provider = MockPriceDataProvider(symbols_data)
        broker_api = MockBrokerAPI(initial_balance=initial_balance)
        
        # Create trading manager
        trading_manager = LiveTradingManager(
            price_provider=price_provider,
            broker_api=broker_api,
            position_size=0.1,
            max_positions=5,
            stop_loss_pct=0.05
        )
        
        logger.info("Live trading manager configured")
        return trading_manager
    
    def print_summary(self, results: Dict):
        """Print backtest summary."""
        if 'metrics' in results:
            metrics = results['metrics']
            print(f"\n{'='*50}")
            print(f"Backtest Summary: {results.get('pair', 'Portfolio')}")
            print(f"{'='*50}")
            print(f"Total Return:      {metrics['total_return']:>10.2%}")
            print(f"Annual Return:     {metrics['annual_return']:>10.2%}")
            print(f"Annual Volatility: {metrics['annual_volatility']:>10.2%}")
            print(f"Sharpe Ratio:      {metrics['sharpe_ratio']:>10.2f}")
            print(f"Max Drawdown:      {metrics['max_drawdown']:>10.2%}")
            print(f"Win Rate:          {metrics['win_rate']:>10.2%}")
            print(f"Num Trades:        {metrics['num_trades']:>10d}")
            print(f"Avg Holding Days:  {metrics['avg_holding_period']:>10.1f}")
            print(f"{'='*50}\n")
        else:
            print(f"\nPortfolio Summary")
            print(f"{'='*50}")
            print(f"Num Pairs:    {results.get('num_pairs', 0)}")
            print(f"Avg Return:   {results.get('avg_return', 0):.2%}")
            print(f"Avg Sharpe:   {results.get('avg_sharpe', 0):.2f}")
            print(f"Total Trades: {results.get('total_trades', 0)}")
            print(f"{'='*50}\n")


def run_example():
    """Run example backtest."""
    print("Pairs Trading Strategy Example")
    print("="*50)
    
    # Initialize pipeline
    pipeline = PairsStrategyPipeline(
        z_score_entry=2.0,
        z_score_exit=0.5,
        initial_capital=100000
    )
    
    # Example: Load data for specific pairs
    try:
        print("\nDownloading price data...")
        symbols = ['AAPL', 'MSFT', 'GOOG', 'TSLA', 'META']
        symbols_data = {}
        
        for symbol in symbols:
            try:
                prices = __import__('yfinance').download(
                    symbol,
                    start='2023-01-01',
                    end='2024-12-31',
                    progress=False
                )
                if len(prices) > 0:
                    symbols_data[symbol] = prices
                    print(f"✓ Loaded {symbol}: {len(prices)} days")
            except Exception as e:
                print(f"✗ Failed to load {symbol}: {e}")
        
        if len(symbols_data) < 2:
            print("\nNote: Need at least 2 symbols with data for pairs trading.")
            print("To run a real example, ensure yfinance is installed:")
            print("  pip install yfinance")
            return
        
        # Discover pairs
        print("\nDiscovering cointegrated pairs...")
        from utils import normalize_prices
        
        normalized_data = {}
        for symbol, prices in symbols_data.items():
            normalized_data[symbol] = normalize_prices(prices['Close'])
        
        pairs = pipeline.discover_pairs(normalized_data, top_n=3)
        
        if not pairs:
            print("No cointegrated pairs found.")
            return
        
        # Backtest
        print("\nRunning backtest...")
        for i, pair in enumerate(pairs, 1):
            symbol1 = pair['symbol1']
            symbol2 = pair['symbol2']
            print(f"\n({i}) {symbol1}/{symbol2}")
            print(f"   Cointegration p-value: {pair['p_value']:.4f}")
            print(f"   Hedge ratio: {pair['hedge_ratio']:.4f}")
            
            result = pipeline.backtest_pair(
                symbol1, symbol2,
                symbols_data[symbol1],
                symbols_data[symbol2],
                hedge_ratio=pair['hedge_ratio']
            )
            
            pipeline.print_summary(result)
    
    except ImportError:
        print("\nNote: yfinance not installed.")
        print("Install with: pip install yfinance")
        print("\nExample structure loaded successfully!")
        print("To run with real data, ensure yfinance is available.")


if __name__ == "__main__":
    run_example()
