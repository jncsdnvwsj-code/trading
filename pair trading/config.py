"""
Configuration file for pairs trading strategy.
"""

# ============================================================================
# COINTEGRATION ANALYSIS PARAMETERS
# ============================================================================

# Confidence level for statistical tests
CONFIDENCE_LEVEL = 0.95

# P-value threshold for stationarity tests (ADF)
ADF_THRESHOLD = 0.05

# P-value threshold for cointegration tests
COINT_THRESHOLD = 0.05

# Half-life constraints for pair selection
MIN_HALF_LIFE = 5        # Minimum in trading periods (too fast = high turnover)
MAX_HALF_LIFE = 252      # Maximum in trading periods (too slow = inefficient)


# ============================================================================
# SIGNAL GENERATION PARAMETERS
# ============================================================================

# Z-score entry threshold (signal when |zscore| > this)
Z_SCORE_ENTRY = 2.0

# Z-score exit threshold (close when |zscore| < this)
Z_SCORE_EXIT = 0.5

# Lookback window for rolling mean/std (in trading periods)
LOOKBACK_WINDOW = 20

# Signal strength thresholds
WEAK_SIGNAL_THRESHOLD = 1.0  # |zscore| > ENTRY
MODERATE_SIGNAL_THRESHOLD = 1.5 * Z_SCORE_ENTRY
STRONG_SIGNAL_THRESHOLD = 1.5 * Z_SCORE_ENTRY

# Bollinger Bands parameters
BB_PERIOD = 20
BB_STD_DEV = 2.0

# OU Process parameters
OU_ENTRY_THRESHOLD = 2.0
OU_EXIT_THRESHOLD = 0.5


# ============================================================================
# BACKTESTING PARAMETERS
# ============================================================================

# Initial capital for backtest
INITIAL_CAPITAL = 100000

# Transaction cost as fraction (0.001 = 0.1%)
TRANSACTION_COST = 0.001

# Slippage as fraction of price
SLIPPAGE = 0.0005

# Minimum data length (in days)
MIN_DATA_LENGTH = 252

# Performance metrics
RISK_FREE_RATE = 0.0
PERIODS_PER_YEAR = 252


# ============================================================================
# LIVE TRADING PARAMETERS
# ============================================================================

# Position management
MAX_POSITIONS = 5               # Maximum concurrent positions
POSITION_SIZE = 0.1             # 10% of capital per position
POSITION_SIZE_STRATEGY = "fixed"  # "fixed", "kelly", "volatility_adjusted"

# Risk management
STOP_LOSS_PCT = 0.05           # 5% stop loss
TAKE_PROFIT_PCT = None          # None = no take profit

# Order parameters
DEFAULT_ORDER_TYPE = "market"   # "market" or "limit"
ORDER_TIMEOUT = 300             # seconds


# ============================================================================
# DATA PARAMETERS
# ============================================================================

# Data source
DATA_SOURCE = "yfinance"        # "yfinance", "csv", "api"

# Minimum volume requirement (in shares per day)
MIN_VOLUME = 1000000

# Maximum bid-ask spread tolerance (%)
MAX_SPREAD = 0.1

# Frequency resampling
RESAMPLE_FREQ = "D"             # "D" (daily), "H" (hourly), "T" (minute)

# Data alignment
FORWARD_FILL_LIMIT = 5          # Max bars to forward fill


# ============================================================================
# PORTFOLIO PARAMETERS
# ============================================================================

# Pair selection
TOP_N_PAIRS = 10               # Number of pairs to trade

# Diversification
SECTOR_CONCENTRATION = 0.5      # Max % in single sector
CORRELATION_MAX = 0.7           # Max correlation between pairs

# Rebalancing
REBALANCE_FREQUENCY = "monthly"  # "daily", "weekly", "monthly"


# ============================================================================
# REPORTING PARAMETERS
# ============================================================================

# Report frequency
REPORT_FREQUENCY = "daily"      # "real-time", "hourly", "daily"

# Metrics to track
METRICS_TO_TRACK = [
    'total_return',
    'annual_return',
    'sharpe_ratio',
    'sortino_ratio',
    'max_drawdown',
    'win_rate',
    'num_trades',
    'profit_factor'
]

# Plot settings
PLOT_EQUITY_CURVE = True
PLOT_DRAWDOWN = True
PLOT_SPREAD = True
PLOT_SIGNALS = True


# ============================================================================
# STRATEGY-SPECIFIC PARAMETERS
# ============================================================================

class StrategyConfig:
    """Strategy-specific configurations."""
    
    MOMENTUM_THRESHOLD = 0.02      # 2% momentum threshold
    VOLATILITY_ADJUSTMENT = True   # Adjust entry based on volatility
    ADAPTIVE_THRESHOLDS = True    # Adjust thresholds based on market conditions
    
    # Market regime detection
    REGIME_DETECTION = True
    HIGH_VOL_THRESHOLD = 1.5      # 1.5x normal volatility
    LOW_VOL_THRESHOLD = 0.5       # 0.5x normal volatility
    
    # Mean reversion assumptions
    ASSUME_OU_PROCESS = True      # Assume OU process for mean reversion
    ESTIMATE_LAMBDA = True        # Estimate mean reversion speed


# ============================================================================
# BROKER-SPECIFIC PARAMETERS
# ============================================================================

class BrokerConfig:
    """Broker API configurations."""
    
    # Alpaca
    ALPACA_BASE_URL = "https://paper-api.alpaca.markets"
    ALPACA_API_KEY = ""
    ALPACA_SECRET_KEY = ""
    
    # Interactive Brokers
    IB_HOST = "127.0.0.1"
    IB_PORT = 7497
    IB_CLIENT_ID = 1
    
    # Crypto (CCXT)
    CCXT_EXCHANGE = "binance"
    CCXT_API_KEY = ""
    CCXT_SECRET_KEY = ""


# ============================================================================
# LOGGING PARAMETERS
# ============================================================================

LOG_LEVEL = "INFO"
LOG_FILE = "pairs_trading.log"
CONSOLE_OUTPUT = True

# Debug settings
DEBUG_MODE = False
VERBOSE_SIGNALS = True
PRINT_TRADES = True
