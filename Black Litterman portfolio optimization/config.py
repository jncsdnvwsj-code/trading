"""
Configuration and settings for portfolio optimization.
Centralized configuration management.
"""

# ============================================================================
# DATA CONFIGURATION
# ============================================================================

# Asset Universe
DEFAULT_ASSETS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'JPM', 'XOM', 'JNJ']

# Data loading
DATA_PERIOD = '5y'  # Default historical period
RISK_FREE_RATE = 0.02  # 2% annual risk-free rate
TRADING_DAYS_PER_YEAR = 252

# ============================================================================
# OPTIMIZATION CONFIGURATION
# ============================================================================

# Mean-Variance Optimizer
MV_CONFIG = {
    'risk_aversion': 2.5,
    'risk_free_rate': RISK_FREE_RATE,
    'optimization_method': 'SLSQP',
    'max_iterations': 1000,
}

# Black-Litterman Model
BL_CONFIG = {
    'risk_aversion': 2.5,
    'risk_free_rate': RISK_FREE_RATE,
    'confidence_default': 0.7,  # Default confidence for views
    'use_equilibrium': True,    # Use market equilibrium as prior
}

# ============================================================================
# PORTFOLIO CONSTRAINTS
# ============================================================================

CONSTRAINTS = {
    'sum_to_one': True,  # Weights sum to 1
    'long_only': True,   # No short selling
    'min_weight': 0.0,   # Minimum weight per asset
    'max_weight': 1.0,   # Maximum weight per asset
    'max_turnover': None,  # No turnover limit if None
}

# Sector concentration limits (optional)
SECTOR_LIMITS = {
    'tech': 0.40,      # Max 40% in tech
    'financials': 0.25,
    'energy': 0.15,
}

# ============================================================================
# BACKTESTING CONFIGURATION
# ============================================================================

BACKTEST_CONFIG = {
    'initial_capital': 1000000,
    'transaction_cost': 0.001,  # 0.1% transaction cost
    'slippage': 0.0005,         # 0.05% slippage
    'rebalance_frequency': 'monthly',  # 'daily', 'weekly', 'monthly', 'quarterly', 'yearly'
    'track_turnover': True,
}

# Rebalancing Frequencies (for reference)
REBALANCING_OPTIONS = {
    'daily': 1,
    'weekly': 5,
    'monthly': 21,
    'quarterly': 63,
    'yearly': 252,
}

# ============================================================================
# SCENARIO TESTING CONFIGURATION
# ============================================================================

SCENARIO_CONFIG = {
    'n_simulations': 100,  # Simulations per scenario
    'periods_per_scenario': 252,  # Trading days
    'scenarios': [
        'normal',
        'bull_market',
        'bear_market',
        'volatility_spike',
        'crisis',
        'sector_rotation',
        'liquidity_crisis',
    ],
}

# Scenario Parameters
SCENARIO_PARAMS = {
    'bull_market_shift': 0.5,           # 50% return increase
    'bear_market_shift': -0.5,          # 50% return decrease
    'volatility_spike_multiplier': 2.0, # 2x volatility
    'crisis_correlation_shock': 0.5,    # Correlation increase
    'liquidity_crisis_slippage': 0.02,  # 2% additional slippage
}

# ============================================================================
# DYNAMIC REBALANCING CONFIGURATION
# ============================================================================

REBALANCING_CONFIG = {
    'window_size': 252,          # 1 year lookback for optimization
    'step_size': 20,             # Rebalance every 20 trading days
    'momentum_period': 60,       # 3 months for momentum calculation
    'zscore_period': 60,         # 3 months for mean reversion
    'adaptive_lookback': 252,    # 1 year for regime detection
}

# ============================================================================
# PERFORMANCE METRICS CONFIGURATION
# ============================================================================

METRICS_CONFIG = {
    'periods_per_year': TRADING_DAYS_PER_YEAR,
    'confidence_level': 0.95,  # For VaR calculation
    'rolling_window': 252,     # 1 year rolling window
}

# ============================================================================
# VISUALIZATION CONFIGURATION
# ============================================================================

PLOT_CONFIG = {
    'figsize_single': (10, 6),
    'figsize_double': (14, 10),
    'figsize_triple': (16, 12),
    'dpi': 300,
    'style': 'seaborn-v0_8-darkgrid',  # Matplotlib style
    'colormap': 'viridis',
    'font_size': 10,
}

# Color scheme
COLORS = {
    'mean_variance': '#1f77b4',  # Blue
    'black_litterman': '#ff7f0e',  # Orange
    'benchmark': '#2ca02c',  # Green
    'optimal': '#d62728',  # Red
    'buy_hold': '#9467bd',  # Purple
    'dynamic': '#8c564b',  # Brown
}

# ============================================================================
# REPORTING CONFIGURATION
# ============================================================================

REPORT_CONFIG = {
    'decimal_places': 4,
    'percentage_format': '.2%',
    'currency_format': ',.0f',
    'include_vif': True,  # Include Variance Inflation Factor analysis
    'include_correlation': True,
}

# ============================================================================
# UTILITY FUNCTIONS FOR CONFIGURATION
# ============================================================================

def get_asset_count():
    """Get number of assets in default universe."""
    return len(DEFAULT_ASSETS)

def get_sector_for_asset(asset):
    """Get sector classification for asset."""
    sector_map = {
        'AAPL': 'tech',
        'MSFT': 'tech',
        'GOOGL': 'tech',
        'AMZN': 'tech',
        'NVDA': 'tech',
        'JPM': 'financials',
        'XOM': 'energy',
        'JNJ': 'healthcare',
    }
    return sector_map.get(asset, 'other')

def get_constraint_bounds():
    """Get weight bounds from constraints."""
    return (CONSTRAINTS['min_weight'], CONSTRAINTS['max_weight'])

def validate_config():
    """Validate configuration consistency."""
    warnings = []
    
    if CONSTRAINTS['sum_to_one'] and CONSTRAINTS['min_weight'] * len(DEFAULT_ASSETS) > 1:
        warnings.append("Warning: Min weight constraint may violate sum-to-one constraint")
    
    if CONSTRAINTS['max_weight'] * len(DEFAULT_ASSETS) < 1:
        warnings.append("Warning: Max weight constraint may violate sum-to-one constraint")
    
    if BL_CONFIG['risk_aversion'] <= 0:
        warnings.append("Warning: Risk aversion must be positive")
    
    if BACKTEST_CONFIG['transaction_cost'] < 0 or BACKTEST_CONFIG['slippage'] < 0:
        warnings.append("Warning: Transaction costs must be non-negative")
    
    return warnings

def print_config_summary():
    """Print configuration summary."""
    print("=" * 70)
    print("PORTFOLIO OPTIMIZATION CONFIGURATION")
    print("=" * 70)
    
    print("\nAssets:")
    print(f"  Universe: {', '.join(DEFAULT_ASSETS)}")
    print(f"  Count: {len(DEFAULT_ASSETS)}")
    
    print("\nOptimization:")
    print(f"  Risk Aversion: {MV_CONFIG['risk_aversion']}")
    print(f"  Risk-Free Rate: {RISK_FREE_RATE:.2%}")
    print(f"  Method: {MV_CONFIG['optimization_method']}")
    
    print("\nConstraints:")
    print(f"  Long-Only: {CONSTRAINTS['long_only']}")
    print(f"  Min Weight: {CONSTRAINTS['min_weight']:.2%}")
    print(f"  Max Weight: {CONSTRAINTS['max_weight']:.2%}")
    if CONSTRAINTS['max_turnover']:
        print(f"  Max Turnover: {CONSTRAINTS['max_turnover']:.2%}")
    
    print("\nBacktesting:")
    print(f"  Initial Capital: ${BACKTEST_CONFIG['initial_capital']:,.0f}")
    print(f"  Transaction Cost: {BACKTEST_CONFIG['transaction_cost']:.4%}")
    print(f"  Slippage: {BACKTEST_CONFIG['slippage']:.4%}")
    print(f"  Rebalance Frequency: {BACKTEST_CONFIG['rebalance_frequency']}")
    
    print("\nScenario Testing:")
    print(f"  Simulations per Scenario: {SCENARIO_CONFIG['n_simulations']}")
    print(f"  Periods per Scenario: {SCENARIO_CONFIG['periods_per_scenario']} days")
    print(f"  Number of Scenarios: {len(SCENARIO_CONFIG['scenarios'])}")
    
    print("\nValidation:")
    warnings = validate_config()
    if warnings:
        for warning in warnings:
            print(f"  ⚠ {warning}")
    else:
        print("  ✓ Configuration is valid")
    
    print("=" * 70)

# ============================================================================
# PRESET CONFIGURATIONS
# ============================================================================

# Conservative Portfolio
CONSERVATIVE_PRESET = {
    'risk_aversion': 4.0,
    'max_weight': 0.3,
    'constraints': {**CONSTRAINTS, 'max_weight': 0.3},
    'rebalance_frequency': 'quarterly',
    'transaction_cost': 0.002,  # Higher assumed cost for smaller positions
}

# Aggressive Portfolio
AGGRESSIVE_PRESET = {
    'risk_aversion': 1.5,
    'max_weight': 1.0,
    'constraints': {**CONSTRAINTS, 'max_weight': 1.0},
    'rebalance_frequency': 'monthly',
    'transaction_cost': 0.0005,  # Lower cost from larger positions
}

# Balanced Portfolio
BALANCED_PRESET = {
    'risk_aversion': 2.5,
    'max_weight': 0.5,
    'constraints': {**CONSTRAINTS, 'max_weight': 0.5},
    'rebalance_frequency': 'monthly',
    'transaction_cost': 0.001,
}

PRESETS = {
    'conservative': CONSERVATIVE_PRESET,
    'aggressive': AGGRESSIVE_PRESET,
    'balanced': BALANCED_PRESET,
}

def load_preset(preset_name):
    """Load configuration preset."""
    if preset_name in PRESETS:
        return PRESETS[preset_name]
    else:
        raise ValueError(f"Unknown preset: {preset_name}")

# ============================================================================
# IF SCRIPT RUN DIRECTLY
# ============================================================================

if __name__ == '__main__':
    print_config_summary()
