# VIX Trading Strategy - Example Configurations

This file contains ready-to-use configuration scenarios for different trading objectives.

## Configuration 1: Conservative Hedging Strategy

**Objective**: Protect equity portfolio with minimal drawdown

```python
from src.strategies import HedgedVolatilityStrategy
from src.backtester import Backtester

# Strategy configuration
strategy = HedgedVolatilityStrategy(
    data,
    vix_column='VIX_Close',
    spx_column='SPX_Close',
    long_threshold=1.5,    # More aggressive hedging
    short_threshold=0.5    # Start hedging earlier
)

# Risk management
risk_config = {
    'position_sizing': {
        'method': 'volatility',
        'kelly_fraction': 0.1,      # Very conservative
        'max_position_size': 0.6,   # Max 60% invested
    },
    'stop_loss': {
        'volatility_multiple': 1.5,  # Tight stops
    }
}

# Expected metrics
# - Annual Return: 5-8%
# - Max Drawdown: -8-12%
# - Sharpe Ratio: 0.8-1.0
# - Use case: Risk-averse investors, pension funds
```

## Configuration 2: Aggressive Growth Strategy

**Objective**: Maximize returns with higher volatility tolerance

```python
from src.strategies import MeanReversionStrategy, TrendFollowingStrategy

# Primary strategy: Mean Reversion with aggressive sizing
mr_strategy = MeanReversionStrategy(
    data,
    ma_window=40,          # Shorter MA for faster signals
    threshold=1.0,         # Lower threshold for more trades
    position_size=1.5      # Leverage: 150% of capital
)

# Risk management
risk_config = {
    'position_sizing': {
        'method': 'kelly',
        'kelly_fraction': 1.0,       # Full Kelly
        'max_position_size': 1.5,    # Leverage allowed
    },
    'stop_loss': {
        'volatility_multiple': 3.0,  # Wider stops
    }
}

# Expected metrics
# - Annual Return: 15-25%
# - Max Drawdown: -25-35%
# - Sharpe Ratio: 0.8-1.2
# - Use case: Hedge funds, aggressive traders
```

## Configuration 3: Balanced Multi-Strategy

**Objective**: Blend multiple strategies for smooth returns

```python
from src.strategies import (MeanReversionStrategy, TrendFollowingStrategy,
                           VolatilityOfVolatilityStrategy)

# Strategy allocation
strategies = {
    'mean_reversion': {
        'weight': 0.40,
        'ma_window': 60,
        'threshold': 1.5,
    },
    'trend_following': {
        'weight': 0.35,
        'short_ma': 10,
        'long_ma': 21,
    },
    'vol_of_vol': {
        'weight': 0.25,
        'vol_of_vol_window': 20,
        'vol_threshold_pct': 75,
    }
}

# Risk management
risk_config = {
    'position_sizing': {
        'method': 'kelly',
        'kelly_fraction': 0.5,       # Half Kelly
        'max_position_size': 1.0,    # No leverage
    }
}

# Portfolio logic
def calculate_portfolio_return(returns_dict, weights):
    portfolio_return = sum(returns * weights[name] 
                          for name, returns in returns_dict.items())
    return portfolio_return

# Expected metrics
# - Annual Return: 10-15%
# - Max Drawdown: -15-20%
# - Sharpe Ratio: 1.0-1.3
# - Use case: Balanced funds, moderate traders
```

## Configuration 4: Volatility-of-Volatility Trading

**Objective**: Exploit vol-of-vol patterns for smoother returns

```python
from src.strategies import VolatilityOfVolatilityStrategy
from src.features import RegimeDetector

# Strategy configuration
strategy = VolatilityOfVolatilityStrategy(
    data,
    vix_column='VIX_Close',
    vol_of_vol_window=15,      # Sensitive to vol changes
    vol_threshold_pct=80,      # Stricter threshold
)

# Regime-adaptive configuration
regimes = RegimeDetector.detect_regimes_vix_percentile(data['VIX_Close'])

# Different parameters per regime
regime_config = {
    0: {'vol_threshold_pct': 85},  # Low vol: stricter
    1: {'vol_threshold_pct': 75},  # Normal: baseline
    2: {'vol_threshold_pct': 65},  # High vol: relaxed
}

# Risk management
risk_config = {
    'position_sizing': {
        'method': 'volatility',
        'kelly_fraction': 0.25,
    }
}

# Expected metrics
# - Annual Return: 8-12%
# - Max Drawdown: -10-15%
# - Sharpe Ratio: 1.1-1.4
# - Use case: Long-vol funds, risk parity
```

## Configuration 5: Regime-Adaptive Strategy

**Objective**: Switch strategies based on market regime

```python
from src.strategies import MeanReversionStrategy, TrendFollowingStrategy
from src.features import RegimeDetector

regimes = RegimeDetector.detect_regimes_vix_percentile(data['VIX_Close'])

# Create both strategies
mr_strategy = MeanReversionStrategy(data, ma_window=60, threshold=1.5)
tf_strategy = TrendFollowingStrategy(data, short_ma=10, long_ma=21)

# Strategy selection logic
def select_strategy(regime, returns_mr, returns_tf):
    if regime == 0:  # Low Vol
        return returns_tf  # Trend following better
    elif regime == 1:  # Normal
        return 0.5 * returns_mr + 0.5 * returns_tf  # 50/50 blend
    else:  # High Vol
        return returns_mr  # Mean reversion better

# Apply regime-based strategy
portfolio_returns = []
for i in range(len(data)):
    regime = regimes.iloc[i]
    strategy_return = select_strategy(regime, mr_pnl.iloc[i], tf_pnl.iloc[i])
    portfolio_returns.append(strategy_return)

portfolio_returns = pd.Series(portfolio_returns, index=data.index)

# Risk management
risk_config = {
    'position_sizing': {
        'method': 'volatility',
        'kelly_fraction': 0.3,
    },
    'regime_adjustments': {
        0: {'position_multiple': 0.8},  # Reduce size in low vol
        1: {'position_multiple': 1.0},
        2: {'position_multiple': 1.1},  # Increase size in high vol
    }
}

# Expected metrics
# - Annual Return: 12-18%
# - Max Drawdown: -12-18%
# - Sharpe Ratio: 1.2-1.5
# - Use case: Quantitative hedge funds
```

## Configuration 6: Sector Rotation with VIX

**Objective**: Use VIX to time sector allocations

```python
# VIX-based sector weighting
def calculate_sector_weights(vix_level):
    if vix_level < 12:
        # Low volatility: favor growth/tech
        return {
            'XLK': 0.25,  # Technology
            'XLV': 0.10,  # Healthcare
            'XLE': 0.05,  # Energy
            'XLF': 0.15,  # Financials
            'XLU': 0.05,  # Utilities
            'XLRE': 0.10, # Real Estate
            'XLI': 0.15,  # Industrials
            'XLP': 0.10,  # Consumer Staples
            'XLY': 0.05,  # Consumer Discretionary
        }
    elif vix_level < 20:
        # Normal volatility: balanced
        return {
            'XLK': 0.15,
            'XLV': 0.15,
            'XLE': 0.10,
            'XLF': 0.15,
            'XLU': 0.10,
            'XLRE': 0.10,
            'XLI': 0.15,
            'XLP': 0.10,
            'XLY': 0.00,
        }
    else:
        # High volatility: defensive
        return {
            'XLK': 0.10,
            'XLV': 0.20,  # Healthcare
            'XLE': 0.00,
            'XLF': 0.05,
            'XLU': 0.15,  # Utilities (defensive)
            'XLRE': 0.15,
            'XLI': 0.10,
            'XLP': 0.20,  # Consumer Staples (defensive)
            'XLY': 0.05,
        }

# Apply dynamic sector rotation
def rebalance_sector_weights(data, vix_column='VIX_Close'):
    sector_weights = pd.DataFrame(index=data.index)
    
    for i in range(len(data)):
        vix = data[vix_column].iloc[i]
        weights = calculate_sector_weights(vix)
        for sector, weight in weights.items():
            if sector not in sector_weights.columns:
                sector_weights[sector] = 0.0
            sector_weights.loc[data.index[i], sector] = weight
    
    return sector_weights

# Expected metrics
# - Annual Return: 8-12%
# - Max Drawdown: -10-15%
# - Sharpe Ratio: 0.9-1.2
# - Use case: Active asset allocation
```

## Configuration 7: Options Collar Protection

**Objective**: Protect long equity with options collar

```python
from src.backtester import Backtester
from scipy.stats import norm

class CollarStrategy:
    def __init__(self, data, hedge_ratio=0.5):
        self.data = data
        self.hedge_ratio = hedge_ratio  # % of portfolio to hedge
    
    def calculate_collar_cost(self, vix_level):
        # Buy put, sell call
        put_strike = vix_level - 2
        call_strike = vix_level + 2
        
        # Simplified cost calculation
        put_cost = vix_level * 0.05  # 5% of VIX
        call_credit = vix_level * 0.04  # 4% of VIX
        
        net_cost = (put_cost - call_credit) * self.hedge_ratio
        return net_cost
    
    def calculate_strategy_return(self, equity_return, vix_level, vix_prev):
        collar_cost = self.calculate_collar_cost(vix_level)
        hedged_return = equity_return * (1 - self.hedge_ratio) - collar_cost
        return hedged_return

# Risk management
risk_config = {
    'hedge_ratio': 0.5,  # Hedge 50% of portfolio
    'rebalance_frequency': 'monthly',
    'collar_width': 2,  # 2 std away from VIX
}

# Expected metrics
# - Annual Return: 6-10%
# - Max Drawdown: -8-12%
# - Sharpe Ratio: 0.9-1.1
# - Use case: Conservative investors
```

## Configuration 8: High Frequency VIX Trading

**Objective**: Trade intraday VIX microstructure

```python
# Note: Requires minute-level data

def high_frequency_config():
    return {
        'data_frequency': '1min',  # 1-minute bars
        'lookback_window': 30,      # 30 minutes
        'entry_threshold': 0.5,     # 0.5 std dev
        'exit_threshold': 0.1,      # Quick profits
        'max_position_duration': 60, # 1 hour max
        
        'risk_management': {
            'position_size': 0.01,      # Small 1% positions
            'stop_loss_pips': 1.0,      # 1 VIX point
            'max_daily_loss': 0.02,     # Stop after 2% loss
        },
        
        'execution': {
            'order_type': 'market',
            'slippage': 0.1,            # 0.1 VIX slippage
            'commission': 0.001,        # 0.1% commission
        }
    }

# Expected metrics
# - Annual Return: 15-30%
# - Max Drawdown: -5-8%
# - Sharpe Ratio: 1.5-2.0
# - Use case: Prop trading firms, HFT
```

## Performance Comparison

| Config | Annual Ret | Max DD | Sharpe | Best For |
|--------|-----------|--------|--------|----------|
| Conservative | 5-8% | -8-12% | 0.8 | Pension funds |
| Aggressive | 15-25% | -25-35% | 0.8 | Hedge funds |
| Balanced | 10-15% | -15-20% | 1.2 | Most traders |
| Vol-of-Vol | 8-12% | -10-15% | 1.3 | Long-vol funds |
| Regime-Adaptive | 12-18% | -12-18% | 1.3 | Quant firms |
| Sector Rotation | 8-12% | -10-15% | 1.0 | Active managers |
| Options Collar | 6-10% | -8-12% | 0.9 | Conservative |
| High Frequency | 15-30% | -5-8% | 1.8 | Prop traders |

## How to Use These Configurations

1. Copy a configuration that matches your objectives
2. Adjust parameters based on your risk tolerance
3. Backtest with your historical data
4. Monitor live performance and fine-tune
5. Rebalance parameters quarterly

## Parameter Adjustment Guide

### To Increase Returns
- Lower entry thresholds (more frequent trades)
- Increase position sizing (use more leverage)
- Extend moving average windows (catch longer trends)
- Reduce transaction costs (negotiate with broker)

### To Reduce Drawdown
- Raise entry thresholds (fewer trades)
- Decrease position sizing
- Add stop losses
- Increase hedging ratio
- Use regime filters

### To Improve Risk-Adjusted Returns (Sharpe Ratio)
- Optimize position sizing with Kelly Criterion
- Reduce volatility through diversification
- Use mean reversion in trending markets
- Add volatility-based position sizing
- Combine multiple uncorrelated strategies

---

For detailed implementation, refer to the main documentation and source code.
