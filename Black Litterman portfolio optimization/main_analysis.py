"""
Comprehensive portfolio optimization example combining Mean-Variance and 
Black-Litterman approaches with backtesting and transaction cost analysis.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data_utils import (load_historical_data, calculate_returns, 
                        calculate_cov_matrix, calculate_expected_returns,
                        get_asset_names)
from mean_variance import MeanVarianceOptimizer
from black_litterman import BlackLittermanModel
from backtester import PortfolioBacktester

# Set random seed for reproducibility
np.random.seed(42)

# Configuration
TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'JPM', 'XOM', 'JNJ']
INITIAL_CAPITAL = 1000000
RISK_FREE_RATE = 0.02

print("=" * 80)
print("PORTFOLIO OPTIMIZATION: Mean-Variance vs Black-Litterman Model")
print("=" * 80)

# ============================================================================
# SECTION 1: Load Historical Data
# ============================================================================
print("\n[1] Loading Historical Data...")
print(f"    Tickers: {', '.join(TICKERS)}")

prices = load_historical_data(TICKERS, period='5y')
returns = calculate_returns(prices)
expected_returns = calculate_expected_returns(returns, annualize=True)
cov_matrix = calculate_cov_matrix(returns, annualize=True)

print(f"    Data Period: {prices.index[0].date()} to {prices.index[-1].date()}")
print(f"    Total Trading Days: {len(returns)}")
print(f"    Number of Assets: {len(TICKERS)}\n")

# Display summary statistics
print("Expected Annual Returns:")
for ticker, ret in zip(TICKERS, expected_returns):
    print(f"    {ticker:6s}: {ret:7.2%}")

print(f"\nAnnualized Volatility:")
for ticker, vol in zip(TICKERS, np.sqrt(np.diag(cov_matrix))):
    print(f"    {ticker:6s}: {vol:7.2%}")

# ============================================================================
# SECTION 2: Mean-Variance Optimization
# ============================================================================
print("\n" + "=" * 80)
print("[2] Mean-Variance Optimization")
print("=" * 80)

mv_optimizer = MeanVarianceOptimizer(expected_returns, cov_matrix, RISK_FREE_RATE)

# Find efficient frontier
print("\nCalculating Efficient Frontier...")
frontier_returns, frontier_vols, frontier_weights = mv_optimizer.efficient_frontier(n_points=50)

# Find optimal portfolios
print("\nOptimal Portfolios:")

max_sharpe_result = mv_optimizer.optimize_max_sharpe()
print(f"\n  Maximum Sharpe Ratio Portfolio:")
print(f"    Expected Return: {max_sharpe_result['return']:.2%}")
print(f"    Volatility:      {max_sharpe_result['volatility']:.2%}")
print(f"    Sharpe Ratio:    {max_sharpe_result['sharpe_ratio']:.4f}")
print(f"    Weights:")
for ticker, weight in zip(TICKERS, max_sharpe_result['weights']):
    if weight > 0.01:
        print(f"      {ticker:6s}: {weight:6.2%}")

min_var_result = mv_optimizer.optimize_min_variance()
print(f"\n  Minimum Variance Portfolio:")
print(f"    Expected Return: {min_var_result['return']:.2%}")
print(f"    Volatility:      {min_var_result['volatility']:.2%}")
print(f"    Sharpe Ratio:    {min_var_result['sharpe_ratio']:.4f}")
print(f"    Weights:")
for ticker, weight in zip(TICKERS, min_var_result['weights']):
    if weight > 0.01:
        print(f"      {ticker:6s}: {weight:6.2%}")

# ============================================================================
# SECTION 3: Black-Litterman Model
# ============================================================================
print("\n" + "=" * 80)
print("[3] Black-Litterman Model with Investor Views")
print("=" * 80)

# Set market weights based on market cap (approximate)
market_weights = np.array([3.0, 2.8, 2.5, 2.2, 1.8, 0.8, 0.6, 0.5])  # Relative market caps
market_weights = market_weights / market_weights.sum()

bl_model = BlackLittermanModel(cov_matrix, risk_aversion=2.5, risk_free_rate=RISK_FREE_RATE)
bl_model.set_market_weights(market_weights)

print(f"\nMarket Weights (Equilibrium Portfolio):")
for ticker, weight in zip(TICKERS, market_weights):
    print(f"    {ticker:6s}: {weight:6.2%}")

# Calculate implied equilibrium returns
eq_returns = bl_model.calculate_equilibrium_returns()
print(f"\nImplied Equilibrium Returns:")
for ticker, ret in zip(TICKERS, eq_returns):
    print(f"    {ticker:6s}: {ret:7.2%}")

# Add investor views
print(f"\nAdding Investor Views:")

# View 1: NVDA will outperform AAPL by 3%
view_P_1 = np.zeros(len(TICKERS))
view_P_1[TICKERS.index('NVDA')] = 1
view_P_1[TICKERS.index('AAPL')] = -1
bl_model.add_view(view_P_1, 0.03, confidence=0.7)
print("    View 1: NVDA outperforms AAPL by 3% (70% confidence)")

# View 2: Tech sector (AAPL, MSFT, GOOGL, NVDA) average return will be 12%
view_P_2 = np.zeros(len(TICKERS))
tech_indices = [TICKERS.index(t) for t in ['AAPL', 'MSFT', 'GOOGL', 'NVDA']]
for idx in tech_indices:
    view_P_2[idx] = 0.25
bl_model.add_view(view_P_2, 0.12, confidence=0.6)
print("    View 2: Tech sector average return 12% (60% confidence)")

# View 3: Energy (XOM) will outperform Finance (JPM) by 2%
view_P_3 = np.zeros(len(TICKERS))
view_P_3[TICKERS.index('XOM')] = 1
view_P_3[TICKERS.index('JPM')] = -1
bl_model.add_view(view_P_3, 0.02, confidence=0.5)
print("    View 3: XOM outperforms JPM by 2% (50% confidence)")

# Fit the model
posterior_returns = bl_model.fit(use_equilibrium=True)
print(f"\nPosterior Expected Returns (Black-Litterman):")
for ticker, ret in zip(TICKERS, posterior_returns):
    print(f"    {ticker:6s}: {ret:7.2%}")

# Optimize using Black-Litterman returns
bl_optimizer = MeanVarianceOptimizer(posterior_returns, cov_matrix, RISK_FREE_RATE)
bl_sharpe_result = bl_optimizer.optimize_max_sharpe()

print(f"\nBlack-Litterman Optimal Portfolio:")
print(f"    Expected Return: {bl_sharpe_result['return']:.2%}")
print(f"    Volatility:      {bl_sharpe_result['volatility']:.2%}")
print(f"    Sharpe Ratio:    {bl_sharpe_result['sharpe_ratio']:.4f}")
print(f"    Weights:")
for ticker, weight in zip(TICKERS, bl_sharpe_result['weights']):
    if weight > 0.01:
        print(f"      {ticker:6s}: {weight:6.2%}")

# ============================================================================
# SECTION 4: Backtesting
# ============================================================================
print("\n" + "=" * 80)
print("[4] Backtesting with Transaction Costs")
print("=" * 80)

# Create rebalance date dictionary (monthly rebalancing)
monthly_dates = prices.index[prices.index.is_month_end]
rebalance_dates = monthly_dates[::1]  # Every month

# Strategy 1: Mean-Variance Max Sharpe
print("\nStrategy 1: Mean-Variance Max Sharpe (Monthly Rebalancing)")
print("-" * 60)

backtest_mv = PortfolioBacktester(prices, INITIAL_CAPITAL)

# Create weights dict with monthly rebalancing
weights_mv = {}
for date in rebalance_dates:
    weights_mv[date] = max_sharpe_result['weights']

results_mv = backtest_mv.run_backtest(
    weights_mv,
    rebalance_frequency='monthly',
    transaction_cost=0.001,
    slippage=0.0005
)

metrics_mv = backtest_mv.calculate_metrics(results_mv)
print(f"  Total Return:        {metrics_mv['total_return']:>8.2%}")
print(f"  Annual Return:       {metrics_mv['annual_return']:>8.2%}")
print(f"  Annual Volatility:   {metrics_mv['annual_volatility']:>8.2%}")
print(f"  Sharpe Ratio:        {metrics_mv['sharpe_ratio']:>8.4f}")
print(f"  Max Drawdown:        {metrics_mv['max_drawdown']:>8.2%}")
print(f"  Final Value:         ${metrics_mv['final_value']:>12,.0f}")
print(f"  Rebalances:          {metrics_mv['transactions']:>8d}")

# Strategy 2: Black-Litterman
print("\nStrategy 2: Black-Litterman with Views (Monthly Rebalancing)")
print("-" * 60)

backtest_bl = PortfolioBacktester(prices, INITIAL_CAPITAL)

weights_bl = {}
for date in rebalance_dates:
    weights_bl[date] = bl_sharpe_result['weights']

results_bl = backtest_bl.run_backtest(
    weights_bl,
    rebalance_frequency='monthly',
    transaction_cost=0.001,
    slippage=0.0005
)

metrics_bl = backtest_bl.calculate_metrics(results_bl)
print(f"  Total Return:        {metrics_bl['total_return']:>8.2%}")
print(f"  Annual Return:       {metrics_bl['annual_return']:>8.2%}")
print(f"  Annual Volatility:   {metrics_bl['annual_volatility']:>8.2%}")
print(f"  Sharpe Ratio:        {metrics_bl['sharpe_ratio']:>8.4f}")
print(f"  Max Drawdown:        {metrics_bl['max_drawdown']:>8.2%}")
print(f"  Final Value:         ${metrics_bl['final_value']:>12,.0f}")
print(f"  Rebalances:          {metrics_bl['transactions']:>8d}")

# Strategy 3: Buy and Hold (Equal Weight)
print("\nStrategy 3: Buy and Hold - Equal Weight (No Rebalancing)")
print("-" * 60)

equal_weights = np.ones(len(TICKERS)) / len(TICKERS)
backtest_bh = PortfolioBacktester(prices, INITIAL_CAPITAL)
results_bh = backtest_bh.run_buy_and_hold(equal_weights, transaction_cost=0.001)

metrics_bh = backtest_bh.calculate_metrics(results_bh)
print(f"  Total Return:        {metrics_bh['total_return']:>8.2%}")
print(f"  Annual Return:       {metrics_bh['annual_return']:>8.2%}")
print(f"  Annual Volatility:   {metrics_bh['annual_volatility']:>8.2%}")
print(f"  Sharpe Ratio:        {metrics_bh['sharpe_ratio']:>8.4f}")
print(f"  Max Drawdown:        {metrics_bh['max_drawdown']:>8.2%}")
print(f"  Final Value:         ${metrics_bh['final_value']:>12,.0f}")

# ============================================================================
# SECTION 5: Transaction Cost Analysis
# ============================================================================
print("\n" + "=" * 80)
print("[5] Transaction Cost Impact Analysis")
print("=" * 80)

cost_levels = [0.0001, 0.0005, 0.001, 0.002, 0.005]
print(f"\nMean-Variance Strategy Impact:")
print(f"{'Transaction Cost':>18} {'Total Return':>15} {'Annual Ret':>15} {'Sharpe':>10}")
print("-" * 60)

for cost in cost_levels:
    backtest_temp = PortfolioBacktester(prices, INITIAL_CAPITAL)
    results_temp = backtest_temp.run_backtest(
        weights_mv,
        transaction_cost=cost,
        slippage=cost*0.5
    )
    metrics_temp = backtest_temp.calculate_metrics(results_temp)
    print(f"{cost:18.4%} {metrics_temp['total_return']:>14.2%} "
          f"{metrics_temp['annual_return']:>14.2%} {metrics_temp['sharpe_ratio']:>10.4f}")

print(f"\nBlack-Litterman Strategy Impact:")
print(f"{'Transaction Cost':>18} {'Total Return':>15} {'Annual Ret':>15} {'Sharpe':>10}")
print("-" * 60)

for cost in cost_levels:
    backtest_temp = PortfolioBacktester(prices, INITIAL_CAPITAL)
    results_temp = backtest_temp.run_backtest(
        weights_bl,
        transaction_cost=cost,
        slippage=cost*0.5
    )
    metrics_temp = backtest_temp.calculate_metrics(results_temp)
    print(f"{cost:18.4%} {metrics_temp['total_return']:>14.2%} "
          f"{metrics_temp['annual_return']:>14.2%} {metrics_temp['sharpe_ratio']:>10.4f}")

# ============================================================================
# SECTION 6: Visualization
# ============================================================================
print("\n" + "=" * 80)
print("[6] Generating Visualizations...")
print("=" * 80)

fig = plt.figure(figsize=(16, 12))

# Plot 1: Efficient Frontier
ax1 = plt.subplot(2, 3, 1)
ax1.plot(frontier_vols, frontier_returns, 'b-', linewidth=2, label='Efficient Frontier')
ax1.scatter(max_sharpe_result['volatility'], max_sharpe_result['return'], 
           marker='*', color='red', s=500, label='Max Sharpe (MV)')
ax1.scatter(min_var_result['volatility'], min_var_result['return'], 
           marker='s', color='green', s=100, label='Min Variance')
ax1.scatter(bl_sharpe_result['volatility'], bl_sharpe_result['return'], 
           marker='^', color='orange', s=100, label='Max Sharpe (BL)')
ax1.set_xlabel('Volatility')
ax1.set_ylabel('Expected Return')
ax1.set_title('Efficient Frontier: MV vs Black-Litterman')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Portfolio Weights Comparison
ax2 = plt.subplot(2, 3, 2)
x = np.arange(len(TICKERS))
width = 0.25
ax2.bar(x - width, max_sharpe_result['weights'], width, label='Max Sharpe (MV)')
ax2.bar(x, bl_sharpe_result['weights'], width, label='Max Sharpe (BL)')
ax2.bar(x + width, equal_weights, width, label='Equal Weight')
ax2.set_xlabel('Assets')
ax2.set_ylabel('Weight')
ax2.set_title('Portfolio Allocations Comparison')
ax2.set_xticks(x)
ax2.set_xticklabels(TICKERS, rotation=45)
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

# Plot 3: Cumulative Returns
ax3 = plt.subplot(2, 3, 3)
ax3.plot(results_mv['date'], results_mv['portfolio_value'], label='Mean-Variance', linewidth=2)
ax3.plot(results_bl['date'], results_bl['portfolio_value'], label='Black-Litterman', linewidth=2)
ax3.plot(results_bh['date'], results_bh['portfolio_value'], label='Buy & Hold (EW)', linewidth=2)
ax3.set_xlabel('Date')
ax3.set_ylabel('Portfolio Value ($)')
ax3.set_title('Backtest: Cumulative Performance')
ax3.legend()
ax3.grid(True, alpha=0.3)
plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)

# Plot 4: Returns Distribution
ax4 = plt.subplot(2, 3, 4)
returns_mv = results_mv['daily_return'].dropna()
returns_bl = results_bl['daily_return'].dropna()
returns_bh = results_bh['daily_return'].dropna()
ax4.hist(returns_mv * 252, bins=50, alpha=0.5, label='Mean-Variance', density=True)
ax4.hist(returns_bl * 252, bins=50, alpha=0.5, label='Black-Litterman', density=True)
ax4.hist(returns_bh * 252, bins=50, alpha=0.5, label='Buy & Hold', density=True)
ax4.set_xlabel('Annualized Daily Return')
ax4.set_ylabel('Density')
ax4.set_title('Returns Distribution')
ax4.legend()
ax4.grid(True, alpha=0.3)

# Plot 5: Covariance Matrix Heatmap
ax5 = plt.subplot(2, 3, 5)
corr_matrix = cov_matrix / np.outer(np.sqrt(np.diag(cov_matrix)), 
                                     np.sqrt(np.diag(cov_matrix)))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
           xticklabels=TICKERS, yticklabels=TICKERS, ax=ax5, vmin=-1, vmax=1)
ax5.set_title('Correlation Matrix')
plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45)
plt.setp(ax5.yaxis.get_majorticklabels(), rotation=0)

# Plot 6: Performance Metrics
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')
metrics_text = f"""
PERFORMANCE COMPARISON

Mean-Variance Max Sharpe:
  Annual Return:    {metrics_mv['annual_return']:>8.2%}
  Annual Vol:       {metrics_mv['annual_volatility']:>8.2%}
  Sharpe Ratio:     {metrics_mv['sharpe_ratio']:>8.4f}
  Max Drawdown:     {metrics_mv['max_drawdown']:>8.2%}

Black-Litterman:
  Annual Return:    {metrics_bl['annual_return']:>8.2%}
  Annual Vol:       {metrics_bl['annual_volatility']:>8.2%}
  Sharpe Ratio:     {metrics_bl['sharpe_ratio']:>8.4f}
  Max Drawdown:     {metrics_bl['max_drawdown']:>8.2%}

Buy & Hold (Equal Weight):
  Annual Return:    {metrics_bh['annual_return']:>8.2%}
  Annual Vol:       {metrics_bh['annual_volatility']:>8.2%}
  Sharpe Ratio:     {metrics_bh['sharpe_ratio']:>8.4f}
  Max Drawdown:     {metrics_bh['max_drawdown']:>8.2%}
"""
ax6.text(0.1, 0.95, metrics_text, transform=ax6.transAxes, fontsize=10,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('portfolio_optimization_analysis.png', dpi=300, bbox_inches='tight')
print("Saved: portfolio_optimization_analysis.png")
plt.show()

# Additional visualization: Expected returns comparison
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(TICKERS))
width = 0.25
ax.bar(x - width, expected_returns, width, label='Historical Mean', alpha=0.8)
ax.bar(x, eq_returns, width, label='BL Equilibrium', alpha=0.8)
ax.bar(x + width, posterior_returns, width, label='BL Posterior (with views)', alpha=0.8)
ax.set_xlabel('Assets')
ax.set_ylabel('Expected Return')
ax.set_title('Expected Returns: Historical vs Black-Litterman Models')
ax.set_xticks(x)
ax.set_xticklabels(TICKERS, rotation=45)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('expected_returns_comparison.png', dpi=300, bbox_inches='tight')
print("Saved: expected_returns_comparison.png")
plt.show()

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
