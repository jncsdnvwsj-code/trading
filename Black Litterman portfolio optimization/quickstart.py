"""
Quick-Start Example: Simple portfolio optimization with basic comparison.
Run this first to understand the main concepts.
"""
import numpy as np
import matplotlib.pyplot as plt
from data_utils import (load_historical_data, calculate_returns,
                        calculate_cov_matrix, calculate_expected_returns)
from mean_variance import MeanVarianceOptimizer
from black_litterman import BlackLittermanModel
from backtester import PortfolioBacktester

print("=" * 70)
print("QUICK START: Portfolio Optimization Example")
print("=" * 70)

# Step 1: Load Data
print("\nStep 1: Loading Historical Data...")
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']
prices = load_historical_data(tickers, period='3y')
print(f"✓ Loaded {len(prices)} days of price data")
print(f"  Assets: {', '.join(tickers)}")
print(f"  Period: {prices.index[0].date()} to {prices.index[-1].date()}")

# Step 2: Calculate Statistics
print("\nStep 2: Calculating Expected Returns and Covariance...")
returns = calculate_returns(prices)
expected_returns = calculate_expected_returns(returns, annualize=True)
cov_matrix = calculate_cov_matrix(returns, annualize=True)

print("✓ Expected Annual Returns:")
for ticker, ret in zip(tickers, expected_returns):
    print(f"  {ticker}: {ret:7.2%}")

# Step 3: Mean-Variance Optimization
print("\nStep 3: Mean-Variance Optimization...")
mv_optimizer = MeanVarianceOptimizer(expected_returns, cov_matrix)

# Get max Sharpe ratio portfolio
max_sharpe = mv_optimizer.optimize_max_sharpe()
print("✓ Maximum Sharpe Ratio Portfolio:")
print(f"  Expected Return: {max_sharpe['return']:6.2%}")
print(f"  Volatility:      {max_sharpe['volatility']:6.2%}")
print(f"  Sharpe Ratio:    {max_sharpe['sharpe_ratio']:6.4f}")
print("  Allocation:")
for ticker, weight in zip(tickers, max_sharpe['weights']):
    if weight > 0.01:
        print(f"    {ticker}: {weight:6.2%}")

# Get minimum variance portfolio
min_var = mv_optimizer.optimize_min_variance()
print("\n✓ Minimum Variance Portfolio:")
print(f"  Expected Return: {min_var['return']:6.2%}")
print(f"  Volatility:      {min_var['volatility']:6.2%}")
print(f"  Sharpe Ratio:    {min_var['sharpe_ratio']:6.4f}")

# Step 4: Black-Litterman Model
print("\nStep 4: Black-Litterman Model with Investor Views...")

# Set market weights (equal weight for simplicity)
market_weights = np.ones(len(tickers)) / len(tickers)

bl_model = BlackLittermanModel(cov_matrix, risk_aversion=2.5)
bl_model.set_market_weights(market_weights)
bl_model.calculate_equilibrium_returns()

# Add a simple view: NVDA will outperform AAPL by 3%
view_P = np.zeros(len(tickers))
view_P[tickers.index('NVDA')] = 1
view_P[tickers.index('AAPL')] = -1
bl_model.add_view(view_P, 0.03, confidence=0.7)
print("✓ Added view: NVDA outperforms AAPL by 3% (70% confidence)")

# Fit the model
posterior_returns = bl_model.fit(use_equilibrium=True)
print("\n✓ Posterior Expected Returns:")
for ticker, ret in zip(tickers, posterior_returns):
    print(f"  {ticker}: {ret:7.2%}")

# Optimize with BL returns
bl_optimizer = MeanVarianceOptimizer(posterior_returns, cov_matrix)
bl_result = bl_optimizer.optimize_max_sharpe()
print("\n✓ Black-Litterman Optimized Portfolio:")
print(f"  Expected Return: {bl_result['return']:6.2%}")
print(f"  Volatility:      {bl_result['volatility']:6.2%}")
print(f"  Sharpe Ratio:    {bl_result['sharpe_ratio']:6.4f}")
print("  Allocation:")
for ticker, weight in zip(tickers, bl_result['weights']):
    if weight > 0.01:
        print(f"    {ticker}: {weight:6.2%}")

# Step 5: Backtesting
print("\nStep 5: Backtesting Both Strategies...")

initial_capital = 1000000
monthly_dates = prices.index[prices.index.is_month_end]

# Mean-Variance Strategy
print("\n✓ Mean-Variance Strategy (Monthly Rebalancing):")
weights_mv = {date: max_sharpe['weights'] for date in monthly_dates}
backtest_mv = PortfolioBacktester(prices, initial_capital)
results_mv = backtest_mv.run_backtest(weights_mv, transaction_cost=0.001)
metrics_mv = backtest_mv.calculate_metrics(results_mv)

print(f"  Total Return:      {metrics_mv['total_return']:8.2%}")
print(f"  Annual Return:     {metrics_mv['annual_return']:8.2%}")
print(f"  Annual Volatility: {metrics_mv['annual_volatility']:8.2%}")
print(f"  Sharpe Ratio:      {metrics_mv['sharpe_ratio']:8.4f}")
print(f"  Max Drawdown:      {metrics_mv['max_drawdown']:8.2%}")
print(f"  Final Value:       ${metrics_mv['final_value']:>12,.0f}")

# Black-Litterman Strategy
print("\n✓ Black-Litterman Strategy (Monthly Rebalancing):")
weights_bl = {date: bl_result['weights'] for date in monthly_dates}
backtest_bl = PortfolioBacktester(prices, initial_capital)
results_bl = backtest_bl.run_backtest(weights_bl, transaction_cost=0.001)
metrics_bl = backtest_bl.calculate_metrics(results_bl)

print(f"  Total Return:      {metrics_bl['total_return']:8.2%}")
print(f"  Annual Return:     {metrics_bl['annual_return']:8.2%}")
print(f"  Annual Volatility: {metrics_bl['annual_volatility']:8.2%}")
print(f"  Sharpe Ratio:      {metrics_bl['sharpe_ratio']:8.4f}")
print(f"  Max Drawdown:      {metrics_bl['max_drawdown']:8.2%}")
print(f"  Final Value:       ${metrics_bl['final_value']:>12,.0f}")

# Step 6: Visualization
print("\nStep 6: Creating Visualizations...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Portfolio Weights
ax1 = axes[0]
x = np.arange(len(tickers))
width = 0.35
equal_weight = np.ones(len(tickers)) / len(tickers)
ax1.bar(x - width, max_sharpe['weights'], width, label='MV Max Sharpe')
ax1.bar(x, bl_result['weights'], width, label='BL Optimal')
ax1.bar(x + width, equal_weight, width, label='Equal Weight', alpha=0.5)
ax1.set_ylabel('Weight')
ax1.set_title('Portfolio Allocations')
ax1.set_xticks(x)
ax1.set_xticklabels(tickers)
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')

# Plot 2: Backtest Performance
ax2 = axes[1]
ax2.plot(results_mv['date'], results_mv['portfolio_value'], label='Mean-Variance', linewidth=2)
ax2.plot(results_bl['date'], results_bl['portfolio_value'], label='Black-Litterman', linewidth=2)
ax2.set_xlabel('Date')
ax2.set_ylabel('Portfolio Value ($)')
ax2.set_title('Backtest Performance Comparison')
ax2.legend()
ax2.grid(True, alpha=0.3)
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

plt.tight_layout()
plt.savefig('quickstart_comparison.png', dpi=200, bbox_inches='tight')
print("✓ Saved visualization to: quickstart_comparison.png")

plt.show()

# Step 7: Key Insights
print("\n" + "=" * 70)
print("KEY INSIGHTS")
print("=" * 70)
print("\n1. Comparison Summary:")
print(f"   MV Sharpe Ratio:    {metrics_mv['sharpe_ratio']:.4f}")
print(f"   BL Sharpe Ratio:    {metrics_bl['sharpe_ratio']:.4f}")
if metrics_bl['sharpe_ratio'] > metrics_mv['sharpe_ratio']:
    improvement = (metrics_bl['sharpe_ratio'] / metrics_mv['sharpe_ratio'] - 1) * 100
    print(f"   → BL improves Sharpe by {improvement:.1f}%")
else:
    degradation = (1 - metrics_bl['sharpe_ratio'] / metrics_mv['sharpe_ratio']) * 100
    print(f"   → BL decreases Sharpe by {degradation:.1f}%")

print("\n2. Risk Comparison:")
print(f"   MV Volatility:      {metrics_mv['annual_volatility']:.2%}")
print(f"   BL Volatility:      {metrics_bl['annual_volatility']:.2%}")

print("\n3. Drawdown Comparison:")
print(f"   MV Max Drawdown:    {metrics_mv['max_drawdown']:.2%}")
print(f"   BL Max Drawdown:    {metrics_bl['max_drawdown']:.2%}")

print("\n4. Allocation Differences:")
print("   Strategy           | AAPL  | MSFT  | GOOGL | AMZN  | NVDA")
print("   " + "-" * 60)
print(f"   MV Max Sharpe      | {max_sharpe['weights'][0]:5.1%}| {max_sharpe['weights'][1]:5.1%}| {max_sharpe['weights'][2]:5.1%}| {max_sharpe['weights'][3]:5.1%}| {max_sharpe['weights'][4]:5.1%}")
print(f"   BL Optimal         | {bl_result['weights'][0]:5.1%}| {bl_result['weights'][1]:5.1%}| {bl_result['weights'][2]:5.1%}| {bl_result['weights'][3]:5.1%}| {bl_result['weights'][4]:5.1%}")

print("\n" + "=" * 70)
print("Next Steps:")
print("  1. Run 'python main_analysis.py' for comprehensive analysis")
print("  2. Run 'python advanced_analysis.py' for stress testing")
print("  3. Modify investor views in this script to test different scenarios")
print("=" * 70)
