"""
Configuration file for VIX Trading Strategy
Modify parameters here to customize the strategy behavior
"""

# ==================== DATA CONFIGURATION ====================
DATA_CONFIG = {
    # Historical data range
    'start_date': '2018-01-01',
    'end_date': None,  # None = today
    'data_source': 'yfinance',  # 'yfinance' or 'synthetic'
    
    # Data columns
    'vix_column': 'VIX_Close',
    'spx_column': 'SPX_Close',
    
    # Synthetic data parameters
    'synthetic': {
        'num_days': 1260,  # 5 years of trading days
        'initial_vix': 15,
        'initial_spx': 4000,
    }
}

# ==================== BACKTESTING CONFIGURATION ====================
BACKTEST_CONFIG = {
    'initial_capital': 100000,
    'transaction_cost': 0.001,  # 0.1% transaction cost
    'risk_free_rate': 0.04,  # 4% annual rate
    'slippage': 0.0,  # Additional slippage cost
}

# ==================== STRATEGY PARAMETERS ====================

# Mean Reversion Strategy
MEAN_REVERSION_CONFIG = {
    'ma_window': 60,  # Moving average window in days
    'threshold': 1.5,  # Z-score threshold for entry
    'position_size': 1.0,  # 0-1, fraction of capital
}

# Trend Following Strategy
TREND_FOLLOWING_CONFIG = {
    'short_ma': 10,  # Short moving average (days)
    'long_ma': 21,   # Long moving average (days)
    'position_size': 1.0,
}

# Volatility of Volatility Strategy
VOL_OF_VOL_CONFIG = {
    'vol_of_vol_window': 20,  # Window for calculating vol-of-vol
    'vol_threshold_pct': 75,  # Percentile threshold
    'position_size': 1.0,
}

# Hedged Volatility Strategy
HEDGED_CONFIG = {
    'base_equity_weight': 1.0,  # Baseline equity allocation
    'long_threshold': 2.0,  # Z-score for heavy hedge (20% equity)
    'short_threshold': 1.0,  # Z-score for moderate hedge (50% equity)
}

# ==================== VOLATILITY REGIME DETECTION ====================
REGIME_CONFIG = {
    'method': 'percentile',  # 'percentile', 'zscore', or 'kmeans'
    'percentile': {
        'window': 60,
        'low_threshold': 0.33,  # Bottom 33rd percentile = Low Vol
        'high_threshold': 0.67,  # Top 67th percentile = High Vol
    },
    'zscore': {
        'window': 60,
        'threshold': 1.0,  # ±1 std dev
    },
    'kmeans': {
        'n_regimes': 3,
        'features': ['VIX_Close_ZScore_60d', 'VIX_Close_RollingVol_21d', 'VolOfVol']
    }
}

# ==================== FEATURE ENGINEERING ====================
FEATURES_CONFIG = {
    'rolling_volatility': {
        'windows': [5, 10, 21, 63],  # Days for rolling volatility
    },
    'mean_reversion': {
        'windows': [20, 60, 120],  # Days for mean reversion indicators
    },
    'momentum': {
        'periods': [5, 10, 21],  # Days for momentum calculation
    },
    'vix_spike_detection': {
        'threshold_std': 2.0,  # Std deviations for spike detection
        'window': 20,
    }
}

# ==================== OPTIONS CONFIGURATION ====================
OPTIONS_CONFIG = {
    'spot_price_column': 'VIX_Close',
    'risk_free_rate': 0.04,
    'dividend_yield': 0.0,
    'time_to_expiration_days': 30,
    'min_strike_offset': -5,  # Strike K = S + offset
    'max_strike_offset': 5,
}

# ==================== RISK MANAGEMENT ====================
RISK_CONFIG = {
    'position_sizing': {
        'method': 'kelly',  # 'fixed', 'kelly', or 'volatility'
        'kelly_fraction': 0.25,  # Use 1/4 Kelly for safety
        'max_position_size': 1.0,  # Max 100% of capital
        'min_position_size': 0.0,  # Min 0% of capital
    },
    'stop_loss': {
        'enabled': True,
        'method': 'volatility',  # 'fixed_pct', 'atr', 'volatility'
        'volatility_multiple': 2.0,  # Stop at 2x daily volatility
    },
    'drawdown_limits': {
        'max_drawdown': 0.25,  # 25% max drawdown
        'stop_trading_after_dd': True,
    },
    'var_confidence_levels': [0.95, 0.99],  # VaR at 95% and 99%
}

# ==================== VISUALIZATION CONFIGURATION ====================
VISUALIZATION_CONFIG = {
    'figsize': (14, 8),
    'dpi': 300,
    'style': 'seaborn-v0_8-darkgrid',
    'save_plots': True,
    'results_directory': '../results/',
    'color_scheme': {
        'vix': 'red',
        'spx': 'blue',
        'long_signal': 'green',
        'short_signal': 'red',
        'regime_low': 'green',
        'regime_normal': 'yellow',
        'regime_high': 'red',
    }
}

# ==================== ANALYSIS CONFIGURATION ====================
ANALYSIS_CONFIG = {
    'rolling_window_days': 60,  # For rolling Sharpe, drawdown, etc.
    'monthly_analysis': True,
    'regime_analysis': True,
    'stress_test': {
        'enabled': True,
        'vix_threshold_percentile': 90,  # Top 10% VIX values = tail events
    },
    'monte_carlo': {
        'enabled': True,
        'num_simulations': 1000,
        'num_periods': 252,  # 1 year of trading days
    },
    'correlation_analysis': True,
    'drawdown_analysis': True,
}

# ==================== OUTPUT CONFIGURATION ====================
OUTPUT_CONFIG = {
    'save_data': True,
    'save_features': True,
    'save_results': True,
    'output_formats': ['csv', 'pickle'],  # Save as CSV and pickle
    'create_summary_report': True,
    'export_performance_metrics': True,
}

# ==================== LOGGING CONFIGURATION ====================
LOGGING_CONFIG = {
    'verbose': True,
    'log_file': '../results/strategy_log.txt',
    'log_level': 'INFO',  # 'DEBUG', 'INFO', 'WARNING', 'ERROR'
}

# ==================== MODEL PARAMETERS ====================
MODEL_CONFIG = {
    'random_seed': 42,
    'sklearn_random_state': 42,
}

# ==================== ALERTS & NOTIFICATIONS ====================
ALERTS_CONFIG = {
    'email_alerts': False,
    'email_address': '',
    'alert_on_extreme_vix': True,
    'extreme_vix_threshold': 30,
    'alert_on_regime_change': True,
    'alert_on_new_high_drawdown': True,
}

# ==================== HELPER FUNCTIONS ====================

def get_config(section: str) -> dict:
    """Get configuration for a specific section"""
    configs = {
        'data': DATA_CONFIG,
        'backtest': BACKTEST_CONFIG,
        'mean_reversion': MEAN_REVERSION_CONFIG,
        'trend_following': TREND_FOLLOWING_CONFIG,
        'vol_of_vol': VOL_OF_VOL_CONFIG,
        'hedged': HEDGED_CONFIG,
        'regime': REGIME_CONFIG,
        'features': FEATURES_CONFIG,
        'options': OPTIONS_CONFIG,
        'risk': RISK_CONFIG,
        'visualization': VISUALIZATION_CONFIG,
        'analysis': ANALYSIS_CONFIG,
        'output': OUTPUT_CONFIG,
        'logging': LOGGING_CONFIG,
        'model': MODEL_CONFIG,
        'alerts': ALERTS_CONFIG,
    }
    return configs.get(section, {})

def print_all_configs():
    """Print all configurations"""
    print("="*80)
    print("VIX TRADING STRATEGY - CONFIGURATION")
    print("="*80)
    
    for config_name in [
        'data', 'backtest', 'mean_reversion', 'trend_following', 
        'vol_of_vol', 'hedged', 'regime', 'features', 'options', 'risk'
    ]:
        config = get_config(config_name)
        print(f"\n{config_name.upper()}:")
        print("-"*80)
        for key, value in config.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for sub_key, sub_value in value.items():
                    print(f"    {sub_key}: {sub_value}")
            else:
                print(f"  {key}: {value}")
    
    print("\n" + "="*80)

if __name__ == '__main__':
    print_all_configs()
