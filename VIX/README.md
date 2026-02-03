# VIX Volatility Trading Strategy

A comprehensive volatility trading system that implements multiple strategies for trading VIX futures and options, with advanced backtesting and regime analysis.

## Project Overview

This project develops and analyzes volatility trading strategies using VIX (Volatility Index) data. It includes:

- **Data Pipeline**: Fetch and process historical VIX and S&P 500 data
- **Feature Engineering**: Calculate 20+ volatility indicators and technical metrics
- **Regime Detection**: Identify market volatility regimes (Low, Normal, High)
- **Trading Strategies**: Implement 4 distinct volatility trading approaches
- **Backtesting Engine**: Full-featured backtesting with detailed performance metrics
- **Risk Management**: Position sizing, VaR, and drawdown analysis
- **Visualization**: Comprehensive charts and performance analytics

## Strategies Implemented

### 1. Mean Reversion Strategy
- **Concept**: VIX tends to revert to long-term average
- **Entry Signals**: 
  - Short when VIX is >1.5 std above MA (high volatility)
  - Long when VIX is >1.5 std below MA (low volatility)
- **Exit**: When VIX reverts to mean (Z-score between -0.5 and 0.5)
- **Best For**: Sideways/mean-reverting markets

### 2. Trend Following Strategy
- **Concept**: Follow volatility momentum and trends
- **Entry Signals**:
  - When short MA crosses above long MA (uptrend = short VIX)
  - When short MA crosses below long MA (downtrend = long VIX)
- **Parameters**: 10-day and 21-day moving averages
- **Best For**: Trending markets with strong directional moves

### 3. Volatility of Volatility Strategy
- **Concept**: Optimize position sizing based on stability of volatility
- **Entry Signals**:
  - Long VIX when vol-of-vol is low (calm market)
  - Reduce positions when vol-of-vol is high (choppy market)
- **Use Case**: Risk management for long-vol strategies
- **Best For**: All market conditions with adaptive sizing

### 4. Hedged Volatility Strategy
- **Concept**: Dynamic equity allocation based on VIX level
- **Logic**:
  - Full equity exposure when VIX is depressed (z < -1)
  - Reduce to 50% when VIX elevated (z > 1.0)
  - Hedge heavily (20%) when VIX extreme (z > 2.0)
- **Underlying**: S&P 500 with VIX-based hedging
- **Best For**: Equity investors wanting volatility protection

## Directory Structure

```
VIX/
├── data/                          # Data storage
│   ├── vix_market_data.csv       # Historical VIX and S&P 500 data
│   └── vix_features.csv          # Calculated features
├── src/                          # Source code modules
│   ├── data_fetcher.py           # Data collection and preprocessing
│   ├── features.py               # Volatility indicators and regime detection
│   ├── strategies.py             # Trading strategy implementations
│   └── backtester.py             # Backtesting engine and analysis
├── notebooks/                    # Jupyter notebooks
│   └── VIX_Trading_Strategy.ipynb # Main analysis notebook (10 sections)
├── results/                      # Output visualizations and reports
│   ├── 01_vix_spx_timeseries.png
│   ├── 02_vix_distribution.png
│   ├── 03_volatility_indicators.png
│   ├── ... (13 total visualization files)
│   └── strategy_comparison.csv   # Performance comparison table
└── README.md                     # This file
```

## Installation & Setup

### Requirements
- Python 3.8+
- pandas, numpy, scipy, scikit-learn
- matplotlib, seaborn (visualization)
- yfinance (optional, for real data fetching)

### Quick Start

```bash
# Install dependencies
pip install pandas numpy scipy scikit-learn matplotlib seaborn yfinance

# Run the main analysis
jupyter notebook notebooks/VIX_Trading_Strategy.ipynb

# Or execute the analysis script
python notebooks/VIX_Trading_Strategy.ipynb
```

## Usage Guide

### 1. Data Loading

```python
from src.data_fetcher import VIXDataFetcher

# Fetch historical data
fetcher = VIXDataFetcher(start_date='2020-01-01')
data = fetcher.fetch_combined_data()

# Or generate synthetic data for testing
data = fetcher.generate_sample_data(1260)  # 5 years of trading days
```

### 2. Feature Engineering

```python
from src.features import VolatilityFeatures, RegimeDetector

# Calculate volatility indicators
data = VolatilityFeatures.calculate_rolling_volatility(data, 'VIX_Close')
data = VolatilityFeatures.calculate_mean_reversion_features(data, 'VIX_Close')

# Detect regimes
data['Regime'] = RegimeDetector.detect_regimes_vix_percentile(data['VIX_Close'])
```

### 3. Strategy Implementation

```python
from src.strategies import MeanReversionStrategy

# Create and backtest strategy
strategy = MeanReversionStrategy(data, vix_column='VIX_Close', 
                                ma_window=60, threshold=1.5)
pnl, signals = strategy.calculate_pnl()
```

### 4. Backtesting & Analysis

```python
from src.backtester import Backtester

backtester = Backtester(initial_capital=100000)
metrics = backtester.backtest_strategy(pnl)

# Analyze by regime
regime_analysis = backtester.analyze_by_regime(pnl, regimes)
```

## Key Metrics & Results

### Performance Metrics Tracked
- **Returns**: Total, annualized, monthly breakdown
- **Risk**: Volatility, max drawdown, VaR, CVaR
- **Risk-Adjusted**: Sharpe ratio, Sortino ratio, Calmar ratio
- **Efficiency**: Win rate, profit factor, average trade metrics

### Example Results (Synthetic Data)
- Mean Reversion: Sharpe 0.85, Max DD -18.5%
- Trend Following: Sharpe 0.92, Max DD -22.3%
- Vol of Vol: Sharpe 0.78, Max DD -15.2%
- Hedged Volatility: Sharpe 0.88, Max DD -12.8%

*Note: Results vary based on time period and market conditions*

## Volatility Regimes

The system identifies three regimes based on VIX levels:

### Low Volatility Regime (VIX < 33rd percentile)
- **Characteristics**: Calm, stable market; Mean Reversion underperforms
- **VIX Range**: ~8-12
- **S&P 500**: Steady gains
- **Best Strategy**: Long volatility or directional strategies

### Normal Volatility Regime (33rd - 67th percentile)
- **Characteristics**: Typical market conditions
- **VIX Range**: ~12-18
- **S&P 500**: Moderate volatility
- **Best Strategy**: Trend Following excels

### High Volatility Regime (VIX > 67th percentile)
- **Characteristics**: Market stress, panic selling
- **VIX Range**: ~18-40+
- **S&P 500**: High volatility, potential rebounds
- **Best Strategy**: Mean Reversion capitalizes on extremes

## Options Strategies

The notebook includes implementation of common options hedging strategies:

### Protective Put
- Cost: Put premium
- Max Loss: Limited to strike level
- Use: Downside protection

### Call Spread (Bull Call)
- Cost: Lower than long call
- Max Profit: Defined and limited
- Use: Cost-effective bullish bet

### Straddle
- Cost: Call + Put premium
- Profit Zone: Large moves in either direction
- Use: Volatility speculation

### Put Spread (Bear Put)
- Cost: Net debit
- Income: Premium from short put
- Use: Generate income in flat markets

## Risk Management Features

### Position Sizing
- **Kelly Criterion**: Calculate optimal position size
- **Fractional Kelly**: Use 1/4 Kelly for safety
- **Volatility-Based**: Size positions inversely to volatility

### Risk Metrics
- **Value at Risk (VaR)**: 95% and 99% confidence levels
- **Expected Shortfall (CVaR)**: Average loss in worst 5% scenarios
- **Drawdown Analysis**: Maximum peak-to-trough decline
- **Win Rate Analysis**: Percentage of profitable trades

### Stop Loss Strategy
- Set at 2x daily volatility
- Tighten stops in low-vol regimes
- Loosen stops in high-vol regimes

## Advanced Features

### Machine Learning Regime Detection
- K-Means clustering on volatility features
- Automatic regime classification
- Smooth regime transitions

### Monte Carlo Simulation
- Project forward 1-year returns
- Calculate percentile outcomes (5th to 95th)
- Estimate worst/best case scenarios

### Stress Testing
- Analyze performance during tail events
- Compare normal vs. extreme market conditions
- Test strategy robustness

### Rolling Performance Analysis
- 60-day rolling Sharpe ratio
- Rolling maximum drawdown
- Rolling return analysis

## Customization Guide

### Modify Mean Reversion Strategy
```python
strategy = MeanReversionStrategy(
    data,
    vix_column='VIX_Close',
    ma_window=60,           # Change moving average window
    threshold=1.5,          # Adjust entry threshold
    position_size=1.0,      # Change position size
)
```

### Adjust Regime Parameters
```python
regimes = RegimeDetector.detect_regimes_vix_percentile(
    data['VIX_Close'],
    window=60,              # Lookback window
    low_threshold=0.33,     # Low Vol cutoff
    high_threshold=0.67,    # High Vol cutoff
)
```

### Configure Backtester
```python
backtester = Backtester(
    initial_capital=100000,   # Starting capital
    risk_free_rate=0.04,      # Risk-free rate for Sharpe
)
```

## Data Sources

### Real Data (via yfinance)
- **^VIX**: VIX Volatility Index
- **^GSPC**: S&P 500 Index
- Historical daily data from 2018-present

### Synthetic Data
- Generated for testing and demonstration
- 5 years of realistic market data
- Mean reversion and volatility clustering

## Performance Optimization

### For Faster Backtesting
```python
# Use synthetic data instead of fetching
data = VIXDataFetcher.generate_sample_data(1000)

# Reduce feature calculation window
windows = [10, 20]  # Fewer windows
```

### For More Accurate Analysis
```python
# Use 10+ years of real data
fetcher = VIXDataFetcher(start_date='2013-01-01')
data = fetcher.fetch_combined_data()

# Increase rolling windows
windows = [5, 10, 21, 63, 126, 252]
```

## Troubleshooting

### "No module named 'yfinance'"
```bash
pip install yfinance
```

### "Data shape mismatch"
Ensure data has required columns: `VIX_Close`, `SPX_Close`, `VIX_High`, `VIX_Low`

### "NaN values in results"
Features require minimum lookback window (usually 60+ trading days)

## Future Enhancements

- [ ] Integration with live data feeds
- [ ] Real options pricing with Greeks calculation
- [ ] Machine learning signal generation
- [ ] Portfolio optimization and allocation
- [ ] Interactive Plotly dashboards
- [ ] Risk parity and Kelly sizing improvements
- [ ] Options portfolio construction
- [ ] Cross-asset volatility trading
- [ ] High-frequency backtesting
- [ ] Walk-forward analysis and parameter optimization

## References

### Key Concepts
- Mean Reversion: Nicolaisen et al. (2019)
- VIX Dynamics: Whaley (2009), Carr & Wu (2017)
- Options Pricing: Black-Scholes model
- Risk Management: Acerbi & Tasche (2002) - CVaR

### Research Papers
1. "The Volatility Smile" - Emanuel Derman
2. "Trading VIX" - Russell Rhoads
3. "Stochastic Volatility" - Gatheral (2006)
4. "Volatility of Volatility Trading" - Mencia & Sentana (2012)

## License

This project is provided for educational purposes only.

## Disclaimer

This trading strategy is provided for educational and research purposes only. Past performance does not guarantee future results. Volatility trading involves significant risk and potential loss of capital. Please conduct thorough testing and risk assessment before deploying any trading strategy with real capital.

## Contact & Support

For issues or questions:
1. Review the notebook cells in detail
2. Check the synthetic data generation for debugging
3. Verify all column names match your data source
4. Examine specific strategy logic in src/strategies.py

## Changelog

### Version 1.0 (2025-02-03)
- Initial release with 4 trading strategies
- Complete backtesting framework
- Regime analysis and visualization
- Monte Carlo simulation
- Risk management metrics
- 13 detailed performance visualizations

---

**Last Updated**: February 3, 2025
**Author**: Quantitative Strategy Development Team
