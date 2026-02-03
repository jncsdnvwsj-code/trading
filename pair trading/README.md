# Pairs Trading & Statistical Arbitrage Strategy

A comprehensive framework for developing, backtesting, and deploying pairs trading and statistical arbitrage strategies.

## Overview

This system implements a complete pairs trading pipeline using cointegration analysis and mean reversion signals. It can:

- Discover cointegrated asset pairs from large universes
- Generate trading signals based on mean reversion
- Backtest strategies with realistic transaction costs
- Execute live trades with risk management
- Monitor portfolio performance and risk metrics

## Key Components

### 1. **Cointegration Analysis** (`cointegration.py`)

Identifies stable long-term relationships between assets.

**Key Classes:**
- `CointegrationAnalyzer`: Tests stationarity and cointegration
  - Augmented Dickey-Fuller (ADF) test for stationarity
  - Engle-Granger cointegration test
  - Johansen multivariate cointegration test
  - Half-life calculation for mean reversion speed

- `PairsSelector`: Selects optimal pairs based on multiple criteria
  - Cointegration strength
  - Mean reversion half-life
  - Spread volatility

**Example:**
```python
analyzer = CointegrationAnalyzer()
result = analyzer.test_cointegration(price1, price2, "Stock1", "Stock2")
# result contains p_value, hedge_ratio, half_life, spread statistics
```

### 2. **Signal Generation** (`signal_generation.py`)

Generates trading signals from mean-reverting spreads.

**Signal Generators:**

**MeanReversionSignalGenerator** (Primary)
- Z-score based entry/exit
- Adaptive thresholds based on half-life
- Signal strength classification
```python
generator = MeanReversionSignalGenerator(z_score_entry=2.0, z_score_exit=0.5)
signals = generator.generate_signals(spread)
# Signals: BUY (1), SELL (-1), HOLD (0)
```

**BollingerBandSignalGenerator**
- Upper/lower band breakouts
- Touch-and-revert logic
```python
bb_generator = BollingerBandSignalGenerator(period=20, std_dev=2.0)
signals = bb_generator.generate_signals(spread)
```

**OrnsteinUhlenbeckSignalGenerator**
- Mean reversion speed estimation
- OU process-based thresholds
```python
ou_generator = OrnsteinUhlenbeckSignalGenerator()
signals = ou_generator.generate_signals(spread)
```

**VolatilityAdjustedSignalGenerator**
- Adjusts entry thresholds based on market volatility
- Tighter bands in low volatility, wider in high

### 3. **Backtesting Framework** (`backtest.py`)

Full backtest engine with realistic market conditions.

**Features:**
- Position management
- Transaction costs and slippage
- P&L calculation
- Trade history tracking
- Performance metrics calculation

**Key Classes:**
- `BacktestEngine`: Main backtesting engine
- `PortfolioAnalyzer`: Risk and performance metrics

**Metrics Calculated:**
- Total and annual returns
- Volatility (annual)
- Sharpe ratio
- Max drawdown
- Win rate
- Profit factor
- Average holding period
- Value at Risk (VaR)
- Conditional VaR (Expected Shortfall)
- Calmar ratio
- Sortino ratio

**Example:**
```python
engine = BacktestEngine(initial_capital=100000)
results = engine.run_backtest(prices1, prices2, signals)
metrics = engine.get_performance_metrics(results)

print(f"Return: {metrics['total_return']:.2%}")
print(f"Sharpe: {metrics['sharpe_ratio']:.2f}")
```

### 4. **Live Trading Integration** (`live_trading.py`)

Execute signals on live market data with broker integration.

**Key Classes:**
- `PriceDataProvider`: Abstract price feed interface
- `BrokerAPI`: Abstract broker API
- `MockPriceDataProvider`: Paper trading with simulated data
- `MockBrokerAPI`: Simulated broker for testing
- `LiveTradingManager`: Orchestrates signal execution

**Features:**
- Position sizing based on account balance
- Order placement and management
- Risk monitoring
- Position tracking
- P&L monitoring

**Example:**
```python
price_provider = MockPriceDataProvider(data)
broker = MockBrokerAPI(initial_balance=100000)
trader = LiveTradingManager(price_provider, broker)

success, msg = trader.execute_signal({
    'symbol1': 'AAPL',
    'symbol2': 'MSFT',
    'signal': 1,  # BUY
    'hedge_ratio': 0.95
})

status = trader.get_portfolio_status()
print(f"Active positions: {status['active_positions']}")
print(f"Unrealized P&L: {status['total_unrealized_pnl']}")
```

### 5. **Complete Pipeline** (`strategy.py`)

Orchestrates entire workflow from discovery to live trading.

**Key Class:**
- `PairsStrategyPipeline`: End-to-end strategy management

**Workflow:**
1. Discover cointegrated pairs
2. Backtest individual pairs
3. Aggregate portfolio results
4. Setup live trading

**Example:**
```python
pipeline = PairsStrategyPipeline(
    z_score_entry=2.0,
    z_score_exit=0.5,
    initial_capital=100000
)

# Discover pairs
pairs = pipeline.discover_pairs(symbols_data, top_n=10)

# Backtest
for pair in pairs:
    result = pipeline.backtest_pair(
        pair['symbol1'],
        pair['symbol2'],
        symbols_data[pair['symbol1']],
        symbols_data[pair['symbol2']],
        hedge_ratio=pair['hedge_ratio']
    )
    pipeline.print_summary(result)

# Setup live trading
trader = pipeline.setup_live_trading(symbols_data)
```

### 6. **Utilities** (`utils.py`)

Helper functions for data processing.

**Key Functions:**
- `normalize_prices()`: Scale prices to base 100
- `calculate_returns()`: Log returns calculation
- `calculate_spread()`: Spread with optional hedge ratio
- `calculate_zscore()`: Rolling Z-score
- `calculate_half_life()`: Mean reversion half-life
- `load_price_data()`: Download historical data
- `validate_data()`: Check data quality

## Installation

```bash
# Install dependencies
pip install numpy pandas statsmodels scipy

# Optional: for live data
pip install yfinance

# Optional: for broker integration (requires account)
# pip install alpaca-trade-api  # Alpaca
# pip install ccxt              # Crypto exchanges
```

## Quick Start

### 1. Basic Pairs Trading

```python
from strategy import PairsStrategyPipeline
from utils import load_price_data, normalize_prices

# Create pipeline
pipeline = PairsStrategyPipeline()

# Load data
price1, price2 = load_price_data('AAPL', 'MSFT', '2023-01-01', '2024-12-31')

# Backtest
result = pipeline.backtest_pair(
    'AAPL', 'MSFT',
    price1, price2,
    hedge_ratio=0.95
)

pipeline.print_summary(result)
```

### 2. Discover Multiple Pairs

```python
from cointegration import CointegrationAnalyzer

# Load universe of stocks
symbols = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'TSLA']
data = {sym: load_price_data(sym)[0] for sym in symbols}

# Normalize
normalized = {sym: normalize_prices(data[sym]['Close']) for sym in symbols}

# Find cointegrated pairs
analyzer = CointegrationAnalyzer()
pairs = analyzer.find_cointegrated_pairs(normalized, top_n=10)

for pair in pairs:
    print(f"{pair['symbol1']}/{pair['symbol2']}: p={pair['p_value']:.4f}")
```

### 3. Live Trading Simulation

```python
from live_trading import MockPriceDataProvider, MockBrokerAPI, LiveTradingManager
from signal_generation import MeanReversionSignalGenerator

# Setup
price_provider = MockPriceDataProvider(price_data)
broker = MockBrokerAPI(initial_balance=100000)
trader = LiveTradingManager(price_provider, broker)

# Generate and execute signals
for signal in signals:
    success, msg = trader.execute_signal(signal)
    if success:
        print(msg)

# Monitor
status = trader.get_portfolio_status()
for pos in status['positions']:
    print(f"{pos['pair']}: PnL {pos['unrealized_pnl']:.2f}")
```

## Key Concepts

### Cointegration
Two assets are cointegrated if they have a long-term equilibrium relationship. When prices diverge, mean reversion creates profitable trading opportunities.

- **Engle-Granger Test**: Two-step method for pairwise testing
- **Johansen Test**: Multivariate method for portfolio analysis

### Mean Reversion
An asset's spread tends to revert to its historical mean. Trading capitalizes on this.

- **Z-score**: Standardized deviation from mean
- **Half-life**: Expected time for 50% reversion
- **Spread**: The mean-reverting series (e.g., price1 - β×price2)

### Hedge Ratio
The relative sizing of positions to minimize spread variance.

Calculated as β from regression: `price1 = α + β×price2 + ε`

### Performance Metrics

**Sharpe Ratio** = (Annual Return) / (Annual Volatility)
- Higher is better; >1.0 considered good

**Sortino Ratio** = (Annual Return) / (Downside Volatility)
- Only penalizes negative volatility

**Max Drawdown** = Maximum peak-to-trough decline
- Measures worst-case scenario

**Calmar Ratio** = (Annual Return) / (Max Drawdown)
- Return per unit of downside risk

## Configuration

Create `config.py` to customize parameters:

```python
# Cointegration
CONFIDENCE_LEVEL = 0.95
ADF_THRESHOLD = 0.05
COINT_THRESHOLD = 0.05

# Signal Generation
Z_SCORE_ENTRY = 2.0
Z_SCORE_EXIT = 0.5
LOOKBACK_WINDOW = 20
MIN_HALF_LIFE = 5
MAX_HALF_LIFE = 252

# Backtesting
INITIAL_CAPITAL = 100000
TRANSACTION_COST = 0.001  # 0.1%
SLIPPAGE = 0.0005         # 0.05%

# Position Management
MAX_POSITIONS = 5
POSITION_SIZE = 0.1  # 10% of capital per trade
STOP_LOSS_PCT = 0.05
```

## Extending the System

### Custom Signal Generators

```python
from signal_generation import MeanReversionSignalGenerator

class CustomSignalGenerator(MeanReversionSignalGenerator):
    def generate_signals(self, spread):
        # Your custom logic
        signals = super().generate_signals(spread)
        # Modify signals
        return signals
```

### Custom Brokers

```python
from live_trading import BrokerAPI

class CustomBroker(BrokerAPI):
    def place_order(self, symbol, quantity, price):
        # Connect to your broker API
        pass
    
    def get_position(self, symbol):
        # Fetch current position
        pass
```

### Real-Time Price Feeds

```python
from live_trading import PriceDataProvider

class RealtimePriceFeed(PriceDataProvider):
    def get_latest_price(self, symbol):
        # Connect to websocket or API
        pass
```

## Risk Management

The system includes built-in risk controls:

1. **Position Sizing**: Limits capital per trade
2. **Max Positions**: Limits concurrent trades
3. **Stop Loss**: Automatic position closure
4. **Portfolio Monitoring**: Real-time P&L tracking
5. **Transaction Costs**: Realistic slippage modeling

## Performance Considerations

- **Computational**: Cointegration analysis scales as O(n²) for n assets
- **Data**: Recommend minimum 252 days (1 year) of data
- **Signal Frequency**: Daily optimal for stable pairs

## Troubleshooting

**No cointegrated pairs found**
- Increase time period
- Lower cointegration threshold
- Use more liquid assets

**High transaction costs reducing returns**
- Use longer holding periods
- Lower entry thresholds
- Focus on higher-liquidity pairs

**High drawdowns**
- Increase Z-score entry threshold
- Reduce position size
- Add stop-loss limits

## References

- Engle, R. F., & Granger, C. W. (1987). Co-integration and error correction
- Vidyamurthy, G. (2004). Pairs Trading: Quantitative Methods and Analysis
- Aronson, D. (2007). Evidence-Based Technical Analysis

## License

This code is provided as educational material.

## Disclaimer

Past performance does not guarantee future results. Statistical arbitrage involves substantial risk. Use this framework for research and education. Always backtest thoroughly before deploying real capital.
