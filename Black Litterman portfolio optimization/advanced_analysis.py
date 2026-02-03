"""
Advanced portfolio analysis with scenario testing and stress scenarios.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from data_utils import (load_historical_data, calculate_returns,
                        calculate_cov_matrix, calculate_expected_returns)
from mean_variance import MeanVarianceOptimizer
from black_litterman import BlackLittermanModel
from backtester import PortfolioBacktester
from scenario_analysis import ScenarioAnalyzer
from portfolio_utils import PerformanceReport, diversification_ratio
from rebalancing_strategies import DynamicRebalancer, AdaptiveRebalancer

print("=" * 80)
print("ADVANCED PORTFOLIO ANALYSIS: Stress Testing and Scenario Analysis")
print("=" * 80)

# Configuration
TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'JPM', 'XOM', 'JNJ']
INITIAL_CAPITAL = 1000000

# ============================================================================
# Data Loading and Preparation
# ============================================================================
print("\n[1] Loading Data and Computing Statistics...")
prices = load_historical_data(TICKERS, period='5y')
returns = calculate_returns(prices)
expected_returns = calculate_expected_returns(returns, annualize=True)
cov_matrix = calculate_cov_matrix(returns, annualize=True)

print(f"    Data loaded: {len(prices)} trading days")
print(f"    Time period: {prices.index[0].date()} to {prices.index[-1].date()}\n")

# ============================================================================
# Optimize Portfolios
# ============================================================================
print("[2] Portfolio Optimization...")

# Mean-Variance
mv_optimizer = MeanVarianceOptimizer(expected_returns, cov_matrix)
mv_result = mv_optimizer.optimize_max_sharpe()
print(f"    MV Sharpe: {mv_result['sharpe_ratio']:.4f}")
print(f"    Diversification Ratio: {diversification_ratio(mv_result['weights'], cov_matrix):.4f}")

# Black-Litterman
market_weights = np.array([3.0, 2.8, 2.5, 2.2, 1.8, 0.8, 0.6, 0.5])
market_weights = market_weights / market_weights.sum()

bl_model = BlackLittermanModel(cov_matrix, risk_aversion=2.5)
bl_model.set_market_weights(market_weights)
bl_model.calculate_equilibrium_returns()

# Add views
bl_model.add_view([1, -1, 0, 0, 0, 0, 0, 0], 0.03, confidence=0.7)  # AAPL > MSFT
bl_model.add_view([0, 0, 0, 0, 1, -1, 0, 0], 0.02, confidence=0.6)  # NVDA > JPM

bl_model.fit(use_equilibrium=True)
bl_optimizer = MeanVarianceOptimizer(bl_model.posterior_returns, cov_matrix)
bl_result = bl_optimizer.optimize_max_sharpe()
print(f"    BL Sharpe: {bl_result['sharpe_ratio']:.4f}")
print(f"    Diversification Ratio: {diversification_ratio(bl_result['weights'], cov_matrix):.4f}\n")

# ============================================================================
# Stress Testing - Scenario Analysis
# ============================================================================
print("[3] Stress Testing: Scenario Analysis...")
print("-" * 70)

scenario_analyzer = ScenarioAnalyzer(TICKERS, expected_returns, cov_matrix)

print("\nMean-Variance Portfolio Stress Test (100 simulations each):")
mv_scenarios = scenario_analyzer.run_all_scenarios(mv_result['weights'], 
                                                    periods=252, n_simulations=100)

for scenario_name, metrics in mv_scenarios.items():
    print(f"\n  {scenario_name}:")
    print(f"    Avg Final Value:     {metrics['avg_final_value']:>10.4f} (1 year)")
    print(f"    Worst Case Loss:     {metrics['worst_case_loss']:>10.2%}")
    print(f"    Probability of Loss: {metrics['prob_loss']:>10.2%}")

print("\n" + "-" * 70)
print("\nBlack-Litterman Portfolio Stress Test (100 simulations each):")
bl_scenarios = scenario_analyzer.run_all_scenarios(bl_result['weights'], 
                                                    periods=252, n_simulations=100)

for scenario_name, metrics in bl_scenarios.items():
    print(f"\n  {scenario_name}:")
    print(f"    Avg Final Value:     {metrics['avg_final_value']:>10.4f} (1 year)")
    print(f"    Worst Case Loss:     {metrics['worst_case_loss']:>10.2%}")
    print(f"    Probability of Loss: {metrics['prob_loss']:>10.2%}")

# ============================================================================
# Backtesting with Different Rebalancing Frequencies
# ============================================================================
print("\n" + "=" * 80)
print("[4] Rebalancing Frequency Analysis")
print("=" * 80)

monthly_dates = prices.index[prices.index.is_month_end]

rebalance_frequencies = [
    ('no_rebalance', [prices.index[0]]),
    ('quarterly', prices.index[prices.index.is_month_end][::3]),
    ('monthly', prices.index[prices.index.is_month_end]),
    ('weekly', prices.index[prices.index.dayofweek == 4])  # Fridays
]

results_by_frequency = {}

print("\nMean-Variance Strategy:")
print(f"{'Frequency':<15} {'Total Return':>15} {'Annual Return':>15} {'Sharpe':>10} {'Transactions':>12}")
print("-" * 70)

for freq_name, freq_dates in rebalance_frequencies:
    weights_dict = {}
    for date in freq_dates:
        weights_dict[date] = mv_result['weights']
    
    backtest = PortfolioBacktester(prices, INITIAL_CAPITAL)
    results = backtest.run_backtest(weights_dict, transaction_cost=0.001)
    metrics = backtest.calculate_metrics(results)
    results_by_frequency[f'MV_{freq_name}'] = metrics
    
    print(f"{freq_name:<15} {metrics['total_return']:>14.2%} "
          f"{metrics['annual_return']:>14.2%} {metrics['sharpe_ratio']:>10.4f} "
          f"{metrics['transactions']:>12d}")

print("\nBlack-Litterman Strategy:")
print(f"{'Frequency':<15} {'Total Return':>15} {'Annual Return':>15} {'Sharpe':>10} {'Transactions':>12}")
print("-" * 70)

for freq_name, freq_dates in rebalance_frequencies:
    weights_dict = {}
    for date in freq_dates:
        weights_dict[date] = bl_result['weights']
    
    backtest = PortfolioBacktester(prices, INITIAL_CAPITAL)
    results = backtest.run_backtest(weights_dict, transaction_cost=0.001)
    metrics = backtest.calculate_metrics(results)
    results_by_frequency[f'BL_{freq_name}'] = metrics
    
    print(f"{freq_name:<15} {metrics['total_return']:>14.2%} "
          f"{metrics['annual_return']:>14.2%} {metrics['sharpe_ratio']:>10.4f} "
          f"{metrics['transactions']:>12d}")

# ============================================================================
# Cost Impact Analysis
# ============================================================================
print("\n" + "=" * 80)
print("[5] Transaction Cost Impact Analysis")
print("=" * 80)

cost_scenarios = [0.0001, 0.0005, 0.001, 0.002, 0.005]
print(f"\n{'Cost':<8} {'MV Return':>12} {'MV Sharpe':>12} {'BL Return':>12} {'BL Sharpe':>12}")
print("-" * 55)

for cost in cost_scenarios:
    # Mean-Variance
    weights_dict = {}
    for date in monthly_dates:
        weights_dict[date] = mv_result['weights']
    
    backtest_mv = PortfolioBacktester(prices, INITIAL_CAPITAL)
    results_mv = backtest_mv.run_backtest(weights_dict, transaction_cost=cost)
    metrics_mv = backtest_mv.calculate_metrics(results_mv)
    
    # Black-Litterman
    weights_dict = {}
    for date in monthly_dates:
        weights_dict[date] = bl_result['weights']
    
    backtest_bl = PortfolioBacktester(prices, INITIAL_CAPITAL)
    results_bl = backtest_bl.run_backtest(weights_dict, transaction_cost=cost)
    metrics_bl = backtest_bl.calculate_metrics(results_bl)
    
    print(f"{cost:<.4%}   {metrics_mv['annual_return']:>11.2%} {metrics_mv['sharpe_ratio']:>12.4f} "
          f"{metrics_bl['annual_return']:>11.2%} {metrics_bl['sharpe_ratio']:>12.4f}")

# ============================================================================
# Dynamic Rebalancing Analysis
# ============================================================================
print("\n" + "=" * 80)
print("[6] Dynamic Rebalancing Strategy")
print("=" * 80)

print("\nGenerating rolling weights (12-month lookback, monthly rebalance)...")
dynamic_rebalancer = DynamicRebalancer(prices, window_size=252, step_size=20)
rolling_weights = dynamic_rebalancer.generate_rolling_weights()

print(f"Generated {len(rolling_weights)} rebalance dates")

# Backtest dynamic strategy
backtest_dynamic = PortfolioBacktester(prices, INITIAL_CAPITAL)
results_dynamic = backtest_dynamic.run_backtest(
    rolling_weights,
    transaction_cost=0.001
)
metrics_dynamic = backtest_dynamic.calculate_metrics(results_dynamic)

print(f"\nDynamic Rebalancing Results:")
print(f"    Total Return:        {metrics_dynamic['total_return']:>8.2%}")
print(f"    Annual Return:       {metrics_dynamic['annual_return']:>8.2%}")
print(f"    Annual Volatility:   {metrics_dynamic['annual_volatility']:>8.2%}")
print(f"    Sharpe Ratio:        {metrics_dynamic['sharpe_ratio']:>8.4f}")
print(f"    Max Drawdown:        {metrics_dynamic['max_drawdown']:>8.2%}")

# Compare with static strategies
print(f"\nComparison with Static Strategies:")
print(f"{'Strategy':<25} {'Annual Return':>15} {'Sharpe':>10} {'Max DD':>10}")
print("-" * 62)
print(f"{'Dynamic Rebalance':<25} {metrics_dynamic['annual_return']:>14.2%} "
      f"{metrics_dynamic['sharpe_ratio']:>10.4f} {metrics_dynamic['max_drawdown']:>10.2%}")

# Monthly MV
weights_dict = {}
for date in monthly_dates:
    weights_dict[date] = mv_result['weights']
backtest_mv_monthly = PortfolioBacktester(prices, INITIAL_CAPITAL)
results_mv_monthly = backtest_mv_monthly.run_backtest(weights_dict, transaction_cost=0.001)
metrics_mv_monthly = backtest_mv_monthly.calculate_metrics(results_mv_monthly)
print(f"{'MV Max Sharpe (Monthly)':<25} {metrics_mv_monthly['annual_return']:>14.2%} "
      f"{metrics_mv_monthly['sharpe_ratio']:>10.4f} {metrics_mv_monthly['max_drawdown']:>10.2%}")

# Monthly BL
weights_dict = {}
for date in monthly_dates:
    weights_dict[date] = bl_result['weights']
backtest_bl_monthly = PortfolioBacktester(prices, INITIAL_CAPITAL)
results_bl_monthly = backtest_bl_monthly.run_backtest(weights_dict, transaction_cost=0.001)
metrics_bl_monthly = backtest_bl_monthly.calculate_metrics(results_bl_monthly)
print(f"{'BL (Monthly)':<25} {metrics_bl_monthly['annual_return']:>14.2%} "
      f"{metrics_bl_monthly['sharpe_ratio']:>10.4f} {metrics_bl_monthly['max_drawdown']:>10.2%}")

# ============================================================================
# Visualization
# ============================================================================
print("\n[7] Generating Visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Scenario Analysis Comparison
ax = axes[0, 0]
scenario_names = list(mv_scenarios.keys())
mv_probs = [mv_scenarios[s]['prob_loss'] for s in scenario_names]
bl_probs = [bl_scenarios[s]['prob_loss'] for s in scenario_names]

x = np.arange(len(scenario_names))
width = 0.35
ax.bar(x - width/2, mv_probs, width, label='Mean-Variance')
ax.bar(x + width/2, bl_probs, width, label='Black-Litterman')
ax.set_ylabel('Probability of Loss')
ax.set_title('Stress Test: Probability of Loss by Scenario')
ax.set_xticks(x)
ax.set_xticklabels([s.replace(' ', '\n') for s in scenario_names], fontsize=8)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Plot 2: Rebalancing Frequency Impact
ax = axes[0, 1]
freq_names = [f[0] for f in rebalance_frequencies]
mv_sharpes = [results_by_frequency[f'MV_{name}']['sharpe_ratio'] for name in freq_names]
bl_sharpes = [results_by_frequency[f'BL_{name}']['sharpe_ratio'] for name in freq_names]

x = np.arange(len(freq_names))
ax.bar(x - width/2, mv_sharpes, width, label='Mean-Variance')
ax.bar(x + width/2, bl_sharpes, width, label='Black-Litterman')
ax.set_ylabel('Sharpe Ratio')
ax.set_title('Impact of Rebalancing Frequency')
ax.set_xticks(x)
ax.set_xticklabels(freq_names)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Plot 3: Cost Impact
ax = axes[1, 0]
costs_pct = [c * 100 for c in cost_scenarios]
mv_costs = []
bl_costs = []

for cost in cost_scenarios:
    weights_dict = {}
    for date in monthly_dates:
        weights_dict[date] = mv_result['weights']
    backtest = PortfolioBacktester(prices, INITIAL_CAPITAL)
    results = backtest.run_backtest(weights_dict, transaction_cost=cost)
    metrics = backtest.calculate_metrics(results)
    mv_costs.append(metrics['annual_return'])
    
    weights_dict = {}
    for date in monthly_dates:
        weights_dict[date] = bl_result['weights']
    backtest = PortfolioBacktester(prices, INITIAL_CAPITAL)
    results = backtest.run_backtest(weights_dict, transaction_cost=cost)
    metrics = backtest.calculate_metrics(results)
    bl_costs.append(metrics['annual_return'])

ax.plot(costs_pct, mv_costs, marker='o', label='Mean-Variance')
ax.plot(costs_pct, bl_costs, marker='s', label='Black-Litterman')
ax.set_xlabel('Transaction Cost (%)')
ax.set_ylabel('Annual Return (%)')
ax.set_title('Impact of Transaction Costs on Returns')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: Strategy Comparison
ax = axes[1, 1]
strategies = ['Dynamic\nRebalance', 'MV\nMonthly', 'BL\nMonthly']
returns = [metrics_dynamic['annual_return'], 
           metrics_mv_monthly['annual_return'],
           metrics_bl_monthly['annual_return']]
sharpes = [metrics_dynamic['sharpe_ratio'],
           metrics_mv_monthly['sharpe_ratio'],
           metrics_bl_monthly['sharpe_ratio']]

ax2 = ax.twinx()
bars = ax.bar(strategies, returns, color='steelblue', alpha=0.7, label='Annual Return')
line = ax2.plot(strategies, sharpes, color='red', marker='o', linewidth=2, markersize=8, label='Sharpe Ratio')

ax.set_ylabel('Annual Return (%)', color='steelblue')
ax2.set_ylabel('Sharpe Ratio', color='red')
ax.set_title('Strategy Comparison: Return vs Risk-Adjusted Return')
ax.tick_params(axis='y', labelcolor='steelblue')
ax2.tick_params(axis='y', labelcolor='red')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('advanced_portfolio_analysis.png', dpi=300, bbox_inches='tight')
print("Saved: advanced_portfolio_analysis.png")
plt.show()

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
