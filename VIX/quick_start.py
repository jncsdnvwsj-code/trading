#!/usr/bin/env python
"""
Quick Start Example - VIX Trading Strategy
Run this script to generate sample analysis without Jupyter
"""

import sys
sys.path.insert(0, './src')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from data_fetcher import VIXDataFetcher
from features import VolatilityFeatures, RegimeDetector
from strategies import (MeanReversionStrategy, TrendFollowingStrategy, 
                        VolatilityOfVolatilityStrategy, HedgedVolatilityStrategy)
from backtester import Backtester

def main():
    print("="*80)
    print("VIX VOLATILITY TRADING STRATEGY - QUICK START")
    print("="*80)
    
    # Step 1: Load Data
    print("\n[1] Loading historical data...")
    fetcher = VIXDataFetcher(start_date='2020-01-01')
    
    try:
        data = fetcher.fetch_combined_data()
        print(f"✓ Fetched real data: {len(data)} trading days")
    except Exception as e:
        print(f"⚠ Could not fetch real data ({e})")
        print("  Generating synthetic data instead...")
        data = fetcher.generate_sample_data(1260)
        print(f"✓ Generated synthetic data: {len(data)} trading days")
    
    print(f"  Date range: {data.index[0].date()} to {data.index[-1].date()}")
    
    # Step 2: Calculate Features
    print("\n[2] Calculating volatility features...")
    data = VolatilityFeatures.calculate_rolling_volatility(data, 'VIX_Close', windows=[5, 10, 21, 63])
    data = VolatilityFeatures.calculate_mean_reversion_features(data, 'VIX_Close', windows=[20, 60, 120])
    vix_momentum = VolatilityFeatures.calculate_vix_momentum(data['VIX_Close'], periods=[5, 10, 21])
    data = pd.concat([data, vix_momentum], axis=1)
    data['VolOfVol'] = VolatilityFeatures.calculate_volatility_of_volatility(data['VIX_Close'], window=20)
    data = VolatilityFeatures.calculate_vix_sp500_relationship(data)
    print(f"✓ Calculated {len(data.columns)} total columns (features + data)")
    
    # Step 3: Detect Regimes
    print("\n[3] Detecting volatility regimes...")
    data['Regime'] = RegimeDetector.detect_regimes_vix_percentile(data['VIX_Close'])
    regime_counts = data['Regime'].value_counts().sort_index()
    regime_names = {0: 'Low Vol', 1: 'Normal', 2: 'High Vol'}
    
    for regime, count in regime_counts.items():
        pct = count / len(data) * 100
        print(f"  {regime_names[regime]:15s}: {count:4d} days ({pct:5.1f}%)")
    
    # Step 4: Initialize Strategies
    print("\n[4] Initializing trading strategies...")
    
    mr_strategy = MeanReversionStrategy(data)
    print("  ✓ Mean Reversion Strategy")
    
    tf_strategy = TrendFollowingStrategy(data)
    print("  ✓ Trend Following Strategy")
    
    vv_strategy = VolatilityOfVolatilityStrategy(data)
    print("  ✓ Volatility of Volatility Strategy")
    
    hedged_strategy = HedgedVolatilityStrategy(data)
    print("  ✓ Hedged Volatility Strategy")
    
    # Step 5: Backtest Strategies
    print("\n[5] Backtesting strategies...")
    
    mr_pnl, _ = mr_strategy.calculate_pnl()
    tf_pnl, _ = tf_strategy.calculate_pnl()
    vv_pnl, _ = vv_strategy.calculate_pnl()
    hedged_pnl, _ = hedged_strategy.calculate_pnl()
    spx_returns = np.log(data['SPX_Close'] / data['SPX_Close'].shift(1))
    
    strategies_dict = {
        'Mean Reversion': mr_pnl,
        'Trend Following': tf_pnl,
        'Vol of Vol': vv_pnl,
        'Hedged Volatility': hedged_pnl,
        'S&P 500': spx_returns
    }
    
    print(f"✓ Backtested {len(strategies_dict)} strategies")
    
    # Step 6: Calculate Performance Metrics
    print("\n[6] Calculating performance metrics...")
    
    backtester = Backtester(initial_capital=100000)
    comparison_df = backtester.compare_strategies(strategies_dict)
    
    # Print results
    print("\n" + "="*80)
    print("PERFORMANCE SUMMARY")
    print("="*80)
    
    key_metrics = ['Total Return', 'Annual Return', 'Volatility', 'Sharpe Ratio', 'Max Drawdown']
    
    print(f"\n{'Strategy':<20} {'Total Return':>15} {'Ann. Return':>15} {'Volatility':>12} {'Sharpe':>10}")
    print("-"*80)
    
    for strategy in strategies_dict.keys():
        total_ret = comparison_df.loc[strategy, 'Total Return']
        annual_ret = comparison_df.loc[strategy, 'Annual Return']
        vol = comparison_df.loc[strategy, 'Volatility']
        sharpe = comparison_df.loc[strategy, 'Sharpe Ratio']
        
        print(f"{strategy:<20} {total_ret*100:>14.2f}% {annual_ret*100:>14.2f}% {vol*100:>11.2f}% {sharpe:>9.3f}")
    
    print("\n" + "-"*80)
    print(f"\n{'Metric':<20} {'Value':>20}")
    print("-"*80)
    
    # Find best strategy by Sharpe ratio
    best_sharpe = comparison_df['Sharpe Ratio'].idxmax()
    print(f"Best Risk-Adj Return:  {best_sharpe}")
    
    # Find best total return
    best_return = comparison_df['Total Return'].idxmax()
    print(f"Best Total Return:     {best_return}")
    
    # Find lowest drawdown
    best_dd = comparison_df['Max Drawdown'].idxmin()
    print(f"Lowest Max Drawdown:   {best_dd}")
    
    # Step 7: Regime Analysis
    print("\n" + "="*80)
    print("PERFORMANCE BY VOLATILITY REGIME")
    print("="*80)
    
    for strategy_name in ['Mean Reversion', 'Trend Following']:
        returns = strategies_dict[strategy_name]
        regime_perf = backtester.analyze_by_regime(returns, data['Regime'])
        
        print(f"\n{strategy_name}:")
        print("-"*60)
        
        for regime_name, metrics in regime_perf.items():
            print(f"  {regime_name:<15} Total Return: {metrics['Total Return']*100:>7.2f}%  " +
                  f"Sharpe: {metrics['Sharpe Ratio']:>6.3f}")
    
    # Step 8: Monthly Statistics
    print("\n" + "="*80)
    print("MONTHLY STATISTICS")
    print("="*80)
    
    mr_monthly = mr_pnl.resample('M').apply(lambda x: (1 + x).prod() - 1)
    
    print(f"\nMean Reversion Strategy - Monthly Breakdown:")
    print(f"  Positive Months: {(mr_monthly > 0).sum()}/{len(mr_monthly)}")
    print(f"  Best Month: {mr_monthly.max()*100:7.2f}%")
    print(f"  Worst Month: {mr_monthly.min()*100:7.2f}%")
    print(f"  Avg Monthly Return: {mr_monthly.mean()*100:7.3f}%")
    
    # Summary
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\n✓ Generated {len(data)} data points")
    print(f"✓ Calculated {len(data.columns)} features")
    print(f"✓ Backtested {len(strategies_dict)} strategies")
    print(f"✓ Analyzed {len(data['Regime'].unique())} volatility regimes")
    
    print("\nFor detailed visualizations, please run the Jupyter notebook:")
    print("  jupyter notebook notebooks/VIX_Trading_Strategy.ipynb")
    
    print("\nFor configuration options, see config.py")
    
    # Save results
    print(f"\n✓ Saving results to results/")
    comparison_df.to_csv('results/strategy_comparison_quick.csv')
    data.to_csv('results/vix_data_with_features.csv')
    
    print("\nDone! ✓")

if __name__ == '__main__':
    main()
