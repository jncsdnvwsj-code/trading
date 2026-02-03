# Portfolio Optimization Project - Complete Index

## 📋 Quick Navigation

### 🚀 START HERE
- **First Time?** → Read [README.md](README.md)
- **Want Quick Demo?** → Run `python quickstart.py`
- **Full Analysis?** → Run `python main_analysis.py`
- **Stress Tests?** → Run `python advanced_analysis.py`

---

## 📚 Documentation Files

### [README.md](README.md) - 500+ lines
**Complete project documentation with:**
- Overview and features
- Installation instructions  
- 5 usage examples with code
- Technical details and formulas
- Performance optimization tips
- References and citations

### [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) - 400+ lines
**Executive summary with:**
- Project completion status
- Feature checklist
- File contents overview
- Usage examples
- Expected results
- Learning path
- Quality checklist

---

## 🔧 Core Modules (7 files, 2000+ lines of code)

### [data_utils.py](data_utils.py) - 200 lines
**Data loading and processing utilities**

Functions:
- `load_historical_data()` - Download from Yahoo Finance
- `calculate_returns()` - Convert prices to returns
- `calculate_cov_matrix()` - Annualized covariance
- `calculate_expected_returns()` - Annualized expected returns
- `get_asset_names()` - Extract asset labels

Key Features:
- Multiple return frequencies (daily, weekly, monthly)
- Yahoo Finance integration
- Automatic data cleaning

Example:
```python
prices = load_historical_data(['AAPL', 'MSFT', 'GOOGL'], period='5y')
returns = calculate_returns(prices)
cov_matrix = calculate_cov_matrix(returns, annualize=True)
```

---

### [mean_variance.py](mean_variance.py) - 220 lines
**Markowitz portfolio optimization**

Class: `MeanVarianceOptimizer`

Methods:
- `optimize_max_sharpe()` - Maximum Sharpe ratio portfolio
- `optimize_min_variance()` - Minimum variance portfolio
- `optimize_target_return()` - Specific return objective
- `efficient_frontier()` - Calculate frontier (100+ points)
- `portfolio_performance()` - Calculate return/vol/Sharpe
- `portfolio_volatility()` - Calculate portfolio std dev
- `negative_sharpe()` - Objective function for optimization

Key Features:
- SLSQP optimization algorithm
- Long-only constraints
- Flexible constraints support
- Numerically stable

Example:
```python
optimizer = MeanVarianceOptimizer(expected_returns, cov_matrix)
result = optimizer.optimize_max_sharpe()
print(f"Return: {result['return']:.2%}")
print(f"Volatility: {result['volatility']:.2%}")
print(f"Sharpe: {result['sharpe_ratio']:.4f}")
print(f"Weights: {result['weights']}")
```

---

### [black_litterman.py](black_litterman.py) - 280 lines
**Black-Litterman portfolio model**

Class: `BlackLittermanModel`

Methods:
- `set_market_weights()` - Set equilibrium weights
- `calculate_equilibrium_returns()` - Implied returns
- `add_view()` - Add investor view with confidence
- `fit()` - Fit model with views
- `get_posterior_covariance()` - Adjusted covariance
- `optimize_bl_portfolio()` - Optimize using posterior returns

Key Features:
- Multiple views support
- Confidence-weighted view incorporation
- Equilibrium return calculation
- Posterior covariance adjustment

Example:
```python
bl_model = BlackLittermanModel(cov_matrix, risk_aversion=2.5)
bl_model.set_market_weights(market_weights)
bl_model.calculate_equilibrium_returns()

# Add view: NVDA outperforms AAPL by 3%
view_P = np.array([1, -1, 0, 0, 0, 0, 0, 0])
bl_model.add_view(view_P, 0.03, confidence=0.7)

posterior_returns = bl_model.fit(use_equilibrium=True)
```

---

### [backtester.py](backtester.py) - 250 lines
**Portfolio backtesting engine**

Class: `PortfolioBacktester`

Methods:
- `run_backtest()` - Run strategy with weights schedule
- `run_buy_and_hold()` - Simple buy-and-hold test
- `calculate_metrics()` - Compute performance metrics
- `get_weights_history()` - Extract weight timeseries

Key Features:
- Flexible rebalancing frequencies
- Transaction cost modeling
- Slippage calculation
- Trade tracking
- Daily return calculation

Metrics Calculated:
- Total return, annual return
- Volatility, Sharpe ratio
- Maximum drawdown
- Final portfolio value
- Transaction count

Example:
```python
backtest = PortfolioBacktester(prices, initial_capital=1000000)
weights_dict = {date: optimal_weights for date in monthly_dates}
results = backtest.run_backtest(weights_dict, transaction_cost=0.001)
metrics = backtest.calculate_metrics(results)
print(f"Final Value: ${metrics['final_value']:,.0f}")
print(f"Annual Return: {metrics['annual_return']:.2%}")
```

---

### [portfolio_utils.py](portfolio_utils.py) - 350 lines
**Performance metrics and utility functions**

Ratio Functions:
- `sharpe_ratio()` - Risk-adjusted return
- `sortino_ratio()` - Downside risk focused
- `information_ratio()` - Benchmark relative
- `calmar_ratio()` - Return/max drawdown

Risk Metrics:
- `max_drawdown()` - Largest peak-to-trough decline
- `value_at_risk()` - VaR at confidence level
- `conditional_value_at_risk()` - Expected shortfall
- `portfolio_variance()`, `portfolio_std()` - Volatility

Diversification Metrics:
- `diversification_ratio()` - Diversification measure
- `herfindahl_index()` - Concentration measure
- `effective_n_assets()` - Effective diversification

Portfolio Functions:
- `portfolio_return()` - Expected portfolio return
- `calculate_turnover()` - Weight change magnitude
- `calculate_tracking_error()` - Tracking deviation

Class: `PerformanceReport`
- Generate comprehensive metrics report
- Print formatted performance summary

Example:
```python
from portfolio_utils import PerformanceReport
report = PerformanceReport(returns, benchmark_returns, name='Strategy')
report.print_report()
```

---

### [scenario_analysis.py](scenario_analysis.py) - 350 lines
**Stress testing framework**

Class: `ScenarioAnalyzer`

Scenario Methods:
- `normal_scenario()` - Historical statistics
- `bull_market_scenario()` - +50% returns
- `bear_market_scenario()` - -50% returns  
- `volatility_spike_scenario()` - 2x volatility
- `crisis_scenario()` - Correlation shock
- `sector_rotation_scenario()` - Sector outperformance
- `liquidity_crisis_scenario()` - 2% slippage
- `run_all_scenarios()` - All 7 scenarios combined

Outputs per Scenario:
- Average final value
- Median final value
- Std deviation
- Worst case loss
- Probability of loss
- 100 simulations each

Example:
```python
analyzer = ScenarioAnalyzer(assets, expected_returns, cov_matrix)
scenarios = analyzer.run_all_scenarios(weights, periods=252, n_simulations=100)
for name, metrics in scenarios.items():
    print(f"{name}: {metrics['prob_loss']:.2%} loss probability")
```

---

### [rebalancing_strategies.py](rebalancing_strategies.py) - 300 lines
**Dynamic rebalancing strategies**

Classes:

1. **DynamicRebalancer**
   - `generate_rolling_weights()` - 12-month rolling optimization
   - `generate_momentum_weights()` - Momentum strategy
   - `generate_mean_reversion_weights()` - Mean reversion strategy
   - `get_weights_at_date()` - Lookup function

2. **ConstrainedRebalancer**
   - `get_constraints()` - Weight and turnover constraints
   - `get_bounds()` - Min/max weight bounds
   - `optimize_with_constraints()` - Constrained optimization

3. **AdaptiveRebalancer**
   - `detect_market_regime()` - Bull/bear/sideways
   - `get_regime_weights()` - Regime-dependent allocation

4. **CostAwareRebalancer**
   - `calculate_rebalancing_cost()` - Compute costs
   - `adjust_weights_for_costs()` - Cost-optimal adjustment

Example:
```python
rebalancer = DynamicRebalancer(prices, window_size=252, step_size=20)
rolling_weights = rebalancer.generate_rolling_weights()
backtest = PortfolioBacktester(prices)
results = backtest.run_backtest(rolling_weights, transaction_cost=0.001)
```

---

### [config.py](config.py) - 250 lines
**Centralized configuration**

Configuration Sections:
- `DEFAULT_ASSETS` - Asset universe
- `MV_CONFIG` - Mean-variance parameters
- `BL_CONFIG` - Black-Litterman settings
- `CONSTRAINTS` - Portfolio constraints
- `BACKTEST_CONFIG` - Backtesting parameters
- `SCENARIO_CONFIG` - Stress test settings
- `REBALANCING_CONFIG` - Dynamic parameters
- `PLOT_CONFIG` - Visualization settings

Presets:
- `conservative` - Risk-averse parameters
- `aggressive` - Risk-seeking parameters
- `balanced` - Moderate parameters

Functions:
- `validate_config()` - Check configuration consistency
- `print_config_summary()` - Display configuration
- `load_preset()` - Load configuration preset

Example:
```python
from config import CONSERVATIVE_PRESET, print_config_summary
print_config_summary()
config = load_preset('conservative')
```

---

## 🎯 Example Scripts (3 files, 1200+ lines)

### [quickstart.py](quickstart.py) - 300 lines
**Beginner-friendly introduction**

Content:
1. Load historical data (3 years)
2. Calculate statistics
3. Mean-variance optimization
4. Black-Litterman with 1 view
5. Basic backtesting
6. Simple visualization
7. Key insights

Run: `python quickstart.py`

Output:
- Portfolio allocations
- Performance comparison
- Visualization (quickstart_comparison.png)
- Interpretation guidance

Time: ~2-5 minutes

---

### [main_analysis.py](main_analysis.py) - 500+ lines
**Comprehensive optimization analysis**

Content:
1. Load 5 years of data (8 assets)
2. Statistical analysis
3. Mean-variance optimization
4. Black-Litterman model (3 views)
5. Portfolio comparison
6. Backtesting (3 strategies)
7. Transaction cost analysis
8. Comprehensive visualizations

Run: `python main_analysis.py`

Output:
- Efficient frontier plot
- Portfolio weight comparison
- Cumulative performance
- Return distribution
- Correlation heatmap
- Cost sensitivity tables
- portfolio_optimization_analysis.png
- expected_returns_comparison.png

Time: ~5-10 minutes

Strategies Tested:
1. MV Max Sharpe (monthly rebalancing)
2. Black-Litterman (monthly rebalancing)
3. Buy-and-Hold Equal Weight

---

### [advanced_analysis.py](advanced_analysis.py) - 400+ lines
**Stress testing and advanced analysis**

Content:
1. Data loading and optimization
2. Stress testing (7 scenarios, 100 sims each)
3. Rebalancing frequency analysis (no, quarterly, monthly, weekly)
4. Transaction cost impact (5 cost levels)
5. Dynamic rebalancing comparison
6. Regime-based strategies
7. Comprehensive comparisons

Run: `python advanced_analysis.py`

Output:
- Scenario stress test results
- Rebalancing frequency tables
- Cost sensitivity analysis
- Dynamic vs static comparison
- Visualizations (advanced_portfolio_analysis.png)
- Probability of loss by scenario

Time: ~10-15 minutes

---

## 📊 Output Files Generated

### From quickstart.py
- `quickstart_comparison.png` - Portfolio allocation and performance

### From main_analysis.py
- `portfolio_optimization_analysis.png` - 6-panel analysis
- `expected_returns_comparison.png` - Return comparison

### From advanced_analysis.py
- `advanced_portfolio_analysis.png` - 4-panel stress testing

### All Scripts
Console output with detailed statistics and tables

---

## 🎓 Learning Sequence

### Level 1: Beginner (15 minutes)
1. Read README.md Introduction
2. Review project structure
3. Run quickstart.py
4. View generated visualization

### Level 2: Intermediate (45 minutes)
1. Read README.md Technical Details
2. Study main_analysis.py code
3. Run main_analysis.py
4. Understand outputs and metrics

### Level 3: Advanced (2+ hours)
1. Study each module in detail
2. Run advanced_analysis.py
3. Modify investor views in code
4. Experiment with parameters

### Level 4: Expert (ongoing)
1. Customize for your assets
2. Add additional constraints
3. Implement custom views
4. Integration into production systems

---

## 🔧 Customization Guide

### Change Asset Universe
```python
# In config.py or directly
TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'JPM', 'XOM', 'JNJ']
```

### Modify Investor Views
```python
# In main_analysis.py or advanced_analysis.py
bl_model.add_view(view_P, view_return, confidence)
# view_P is array showing position (long/short)
# view_return is expected excess return
# confidence is 0-1 scale
```

### Change Optimization Parameters
```python
# In config.py
MV_CONFIG = {
    'risk_aversion': 3.0,  # More conservative
    'risk_free_rate': 0.03,  # Higher rate
    ...
}
```

### Modify Backtesting Parameters
```python
backtest = PortfolioBacktester(prices, initial_capital=5000000)
results = backtest.run_backtest(
    weights,
    transaction_cost=0.002,  # 0.2%
    slippage=0.001  # 0.1%
)
```

---

## 📈 Key Formulas Reference

### Sharpe Ratio
$$\text{Sharpe} = \frac{\mu_p - r_f}{\sigma_p}$$

### Portfolio Return
$$\mu_p = \mathbf{w}^T \boldsymbol{\mu}$$

### Portfolio Variance
$$\sigma_p^2 = \mathbf{w}^T \boldsymbol{\Sigma} \mathbf{w}$$

### Black-Litterman Posterior
$$\boldsymbol{\mu}_{BL} = \boldsymbol{\mu}_{eq} + \boldsymbol{\Sigma} \mathbf{P}^T (\mathbf{P} \boldsymbol{\Sigma} \mathbf{P}^T + \boldsymbol{\Omega})^{-1} (\mathbf{Q} - \mathbf{P} \boldsymbol{\mu}_{eq})$$

### Transaction Cost
$$\text{Cost} = \sum_i |w_{new,i} - w_{old,i}| \times V \times (c + s) / 2$$

---

## ⚙️ System Requirements

### Python
- Python 3.8 or higher

### Dependencies
All listed in requirements.txt:
- numpy==1.24.3
- pandas==2.0.3
- scipy==1.11.1
- matplotlib==3.7.2
- seaborn==0.12.2
- scikit-learn==1.3.0
- yfinance==0.2.28

### Hardware
- Minimum: 4GB RAM
- Recommended: 8GB RAM
- Typical execution: <15 minutes

---

## 🐛 Troubleshooting

### No Data Downloaded
```
Error: "No data found for ticker"
Solution: Check ticker symbol (e.g., GOOGL not GOOGLE)
```

### Optimization Failed
```
Error: "Optimization not successful"
Solution: Check that cov_matrix is positive definite
          Add small regularization to diagonal
```

### Memory Issues
```
Solution: Reduce data period or number of assets
          Use data_utils.load_historical_data(period='2y')
```

### Import Errors
```
Solution: pip install -r requirements.txt
          Ensure all modules in same directory
```

---

## 📞 Support & Resources

### Documentation
- README.md - Complete guide
- PROJECT_SUMMARY.md - Executive summary
- This file - Navigation and index

### Code Comments
- Every module has detailed docstrings
- Functions explain parameters and returns
- Examples included in main modules

### External References
- Markowitz, H. (1952) - Portfolio Selection
- Black & Litterman (1992) - Global Portfolio Optimization
- Litterman, R. (2003) - Modern Investment Management

---

## ✅ Verification Checklist

- ✓ All 14 files present and readable
- ✓ Code modules syntactically correct
- ✓ Examples executable
- ✓ Documentation comprehensive
- ✓ Configuration centralized
- ✓ 7 stress scenarios implemented
- ✓ 15+ performance metrics
- ✓ Multiple rebalancing strategies
- ✓ Production-ready code
- ✓ Ready for deployment

---

**Project Version**: 1.0 Complete
**Last Updated**: February 2026
**Status**: Production Ready
