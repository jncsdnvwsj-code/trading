"""
Example scripts for pairs trading strategy.
"""

import pandas as pd
import numpy as np
from typing import Dict, List
from datetime import datetime, timedelta
import logging

# Import modules
from utils import load_price_data, normalize_prices, calculate_half_life
from cointegration import CointegrationAnalyzer, PairsSelector
from signal_generation import (
    MeanReversionSignalGenerator,
    BollingerBandSignalGenerator,
    SignalType
)
from backtest import BacktestEngine
from live_trading import MockPriceDataProvider, MockBrokerAPI, LiveTradingManager
from strategy import PairsStrategyPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# EXAMPLE 1: Basic Cointegration Analysis
# ============================================================================

def example_cointegration_analysis():
    """
    Demonstrate basic cointegration testing between two assets.
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: Cointegration Analysis")
    print("="*70)
    
    try:
        import yfinance as yf
        
        # Load data
        print("\nDownloading price data for AAPL and MSFT...")
        aapl = yf.download('AAPL', start='2023-01-01', end='2024-12-31', progress=False)
        msft = yf.download('MSFT', start='2023-01-01', end='2024-12-31', progress=False)
        
        # Test cointegration
        analyzer = CointegrationAnalyzer()
        result = analyzer.test_cointegration(
            aapl['Close'],
            msft['Close'],
            'AAPL',
            'MSFT'
        )
        
        print(f"\nResults:")
        print(f"  Cointegrated: {result['is_cointegrated']}")
        print(f"  P-value: {result['p_value']:.6f}")
        print(f"  Hedge Ratio: {result['hedge_ratio']:.4f}")
        print(f"  Spread Half-life: {result['half_life']:.2f} days")
        print(f"  Spread Mean: {result['spread_mean']:.2f}")
        print(f"  Spread Std: {result['spread_std']:.2f}")
        
        # Stationarity test
        print(f"\nStationarity Test:")
        adf1 = analyzer.test_stationarity(aapl['Close'], 'AAPL')
        adf2 = analyzer.test_stationarity(msft['Close'], 'MSFT')
        
        print(f"  {adf1['interpretation']}")
        print(f"  {adf2['interpretation']}")
        
    except ImportError:
        print("\nNote: yfinance required. Install with: pip install yfinance")


# ============================================================================
# EXAMPLE 2: Signal Generation
# ============================================================================

def example_signal_generation():
    """
    Demonstrate signal generation from a spread.
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: Signal Generation")
    print("="*70)
    
    try:
        import yfinance as yf
        
        # Load data
        print("\nDownloading price data...")
        aapl = yf.download('AAPL', start='2023-01-01', end='2024-01-31', progress=False)
        msft = yf.download('MSFT', start='2023-01-01', end='2024-01-31', progress=False)
        
        # Calculate spread
        hedge_ratio = 1.0
        spread = aapl['Close'] - (hedge_ratio * msft['Close'])
        
        # Generate Z-score signals
        print("\nGenerating Z-score signals...")
        gen_zscore = MeanReversionSignalGenerator(z_score_entry=2.0, z_score_exit=0.5)
        signals_zscore = gen_zscore.generate_signals(spread)
        
        # Generate Bollinger Band signals
        print("Generating Bollinger Band signals...")
        gen_bb = BollingerBandSignalGenerator(period=20, std_dev=2.0)
        signals_bb = gen_bb.generate_signals(spread)
        
        # Print sample signals
        print("\nSample signals (last 10 days):")
        print("\nZ-Score Signals:")
        print(signals_zscore[['zscore', 'signal', 'strength']].tail(10))
        
        print("\nBollinger Band Signals:")
        print(signals_bb[['signal']].tail(10))
        
        # Count signal types
        n_buy = (signals_zscore['signal'] == SignalType.BUY.value).sum()
        n_sell = (signals_zscore['signal'] == SignalType.SELL.value).sum()
        n_hold = (signals_zscore['signal'] == SignalType.HOLD.value).sum()
        
        print(f"\nSignal Summary:")
        print(f"  Buy signals: {n_buy}")
        print(f"  Sell signals: {n_sell}")
        print(f"  Hold signals: {n_hold}")
        
    except ImportError:
        print("\nNote: yfinance required.")


# ============================================================================
# EXAMPLE 3: Backtesting a Pair
# ============================================================================

def example_backtesting():
    """
    Demonstrate backtesting a single pair.
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: Backtesting")
    print("="*70)
    
    try:
        import yfinance as yf
        
        # Load data
        print("\nDownloading data...")
        aapl = yf.download('AAPL', start='2023-01-01', end='2024-12-31', progress=False)
        msft = yf.download('MSFT', start='2023-01-01', end='2024-12-31', progress=False)
        
        # Setup
        pipeline = PairsStrategyPipeline(
            initial_capital=100000,
            transaction_cost=0.001
        )
        
        # Run backtest
        print("\nRunning backtest...")
        result = pipeline.backtest_pair(
            'AAPL', 'MSFT',
            aapl, msft,
            hedge_ratio=1.0,
            quantity1=100
        )
        
        # Print results
        pipeline.print_summary(result)
        
        # Trade details
        trades = result.get('trades', [])
        if trades:
            print(f"\nTrade Details (first 5):")
            for i, trade in enumerate(trades[:5]):
                print(f"  {i+1}. {trade.entry_date.date()} -> {trade.exit_date.date()}")
                print(f"     PnL: ${trade.pnl:.2f} ({trade.pnl_pct:+.2%})")
                print(f"     Holding: {trade.holding_period} days")
        
    except ImportError:
        print("\nNote: yfinance required.")


# ============================================================================
# EXAMPLE 4: Paper Trading with Mock Broker
# ============================================================================

def example_paper_trading():
    """
    Demonstrate live trading with mock broker.
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: Paper Trading")
    print("="*70)
    
    try:
        import yfinance as yf
        
        # Load data
        print("\nDownloading data...")
        symbols = ['AAPL', 'MSFT']
        price_data = {}
        for sym in symbols:
            price_data[sym] = yf.download(
                sym,
                start='2024-01-01',
                end='2024-12-31',
                progress=False
            )
        
        # Setup trading
        print("\nSetting up mock trading...")
        price_provider = MockPriceDataProvider(price_data)
        broker = MockBrokerAPI(initial_balance=100000)
        trader = LiveTradingManager(price_provider, broker)
        
        # Simulate some trades
        print("\nSimulating trades...")
        signals_to_execute = [
            {
                'symbol1': 'AAPL',
                'symbol2': 'MSFT',
                'signal': 1,  # BUY
                'hedge_ratio': 0.95,
                'strength': 2
            },
            {
                'symbol1': 'AAPL',
                'symbol2': 'MSFT',
                'signal': 0,  # CLOSE
                'hedge_ratio': 0.95,
                'strength': 0
            }
        ]
        
        for signal in signals_to_execute:
            success, msg = trader.execute_signal(signal)
            print(f"  {'✓' if success else '✗'} {msg}")
        
        # Get portfolio status
        status = trader.get_portfolio_status()
        print(f"\nPortfolio Status:")
        print(f"  Account Balance: ${status['account_balance']:.2f}")
        print(f"  Active Positions: {status['active_positions']}")
        print(f"  Unrealized P&L: ${status['total_unrealized_pnl']:.2f}")
        
    except ImportError:
        print("\nNote: yfinance required.")


# ============================================================================
# EXAMPLE 5: Pairs Discovery
# ============================================================================

def example_pairs_discovery():
    """
    Discover cointegrated pairs from a universe of stocks.
    """
    print("\n" + "="*70)
    print("EXAMPLE 5: Pairs Discovery")
    print("="*70)
    
    try:
        import yfinance as yf
        
        # Download data for multiple stocks
        symbols = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'TSLA']
        print(f"\nDownloading data for {len(symbols)} stocks...")
        
        symbol_data = {}
        for sym in symbols:
            try:
                data = yf.download(sym, start='2023-01-01', end='2024-12-31', progress=False)
                if len(data) > 0:
                    symbol_data[sym] = data
                    print(f"  ✓ {sym}")
            except:
                print(f"  ✗ {sym}")
        
        if len(symbol_data) < 2:
            print("\nInsufficient data for pairs analysis")
            return
        
        # Normalize prices
        print("\nNormalizing prices...")
        normalized_data = {}
        for sym, data in symbol_data.items():
            normalized_data[sym] = normalize_prices(data['Close'])
        
        # Find pairs
        print("\nFinding cointegrated pairs...")
        analyzer = CointegrationAnalyzer()
        pairs = analyzer.find_cointegrated_pairs(normalized_data, top_n=5)
        
        if not pairs:
            print("No cointegrated pairs found.")
            return
        
        print(f"\nTop {len(pairs)} cointegrated pairs:")
        for i, pair in enumerate(pairs, 1):
            print(f"\n  {i}. {pair['symbol1']}/{pair['symbol2']}")
            print(f"     P-value: {pair['p_value']:.6f}")
            print(f"     Hedge Ratio: {pair['hedge_ratio']:.4f}")
            print(f"     Half-life: {pair['half_life']:.2f} days")
            print(f"     Spread Std: {pair['spread_std']:.4f}")
        
    except ImportError:
        print("\nNote: yfinance required. Install with: pip install yfinance")


# ============================================================================
# EXAMPLE 6: Performance Analysis
# ============================================================================

def example_performance_analysis():
    """
    Analyze backtest performance metrics.
    """
    print("\n" + "="*70)
    print("EXAMPLE 6: Performance Analysis")
    print("="*70)
    
    try:
        import yfinance as yf
        
        # Load data
        print("\nDownloading data...")
        aapl = yf.download('AAPL', start='2023-01-01', end='2024-12-31', progress=False)
        msft = yf.download('MSFT', start='2023-01-01', end='2024-12-31', progress=False)
        
        # Run backtest
        pipeline = PairsStrategyPipeline()
        result = pipeline.backtest_pair('AAPL', 'MSFT', aapl, msft)
        
        metrics = result.get('metrics', {})
        
        print("\nDetailed Performance Metrics:")
        print(f"  Total Return:       {metrics.get('total_return', 0):>12.2%}")
        print(f"  Annual Return:      {metrics.get('annual_return', 0):>12.2%}")
        print(f"  Annual Volatility:  {metrics.get('annual_volatility', 0):>12.2%}")
        print(f"  Sharpe Ratio:       {metrics.get('sharpe_ratio', 0):>12.2f}")
        print(f"  Sortino Ratio:      {metrics.get('sortino_ratio', 0):>12.2f}")
        print(f"  Max Drawdown:       {metrics.get('max_drawdown', 0):>12.2%}")
        print(f"  Win Rate:           {metrics.get('win_rate', 0):>12.2%}")
        print(f"  Profit Factor:      {metrics.get('profit_factor', 0):>12.2f}")
        print(f"  Num Trades:         {metrics.get('num_trades', 0):>12.0f}")
        print(f"  Avg Trade PnL:      ${metrics.get('avg_trade_pnl', 0):>11.2f}")
        print(f"  Avg Holding Days:   {metrics.get('avg_holding_period', 0):>12.1f}")
        
    except ImportError:
        print("\nNote: yfinance required.")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run all examples."""
    examples = [
        ("Cointegration Analysis", example_cointegration_analysis),
        ("Signal Generation", example_signal_generation),
        ("Backtesting", example_backtesting),
        ("Paper Trading", example_paper_trading),
        ("Pairs Discovery", example_pairs_discovery),
        ("Performance Analysis", example_performance_analysis),
    ]
    
    print("\n" + "="*70)
    print("PAIRS TRADING STRATEGY - EXAMPLES")
    print("="*70)
    print("\nAvailable Examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")
    
    print("\nRunning all examples...\n")
    
    for name, func in examples:
        try:
            func()
        except Exception as e:
            print(f"\nError in {name}: {e}")
            logger.exception(f"Error in {name}")
    
    print("\n" + "="*70)
    print("Examples completed!")
    print("="*70)


if __name__ == "__main__":
    main()
