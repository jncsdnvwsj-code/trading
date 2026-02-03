"""
VIX Volatility Trading Strategy Package
Complete system for trading VIX futures and options
"""

__version__ = "1.0.0"
__author__ = "Quantitative Strategy Development Team"
__description__ = "Comprehensive volatility trading strategy with backtesting framework"

# Import main components
from .data_fetcher import VIXDataFetcher
from .features import VolatilityFeatures, RegimeDetector
from .strategies import (
    MeanReversionStrategy,
    TrendFollowingStrategy,
    VolatilityOfVolatilityStrategy,
    HedgedVolatilityStrategy
)
from .backtester import Backtester, RollingPerformance, StressTestAnalyzer

__all__ = [
    'VIXDataFetcher',
    'VolatilityFeatures',
    'RegimeDetector',
    'MeanReversionStrategy',
    'TrendFollowingStrategy',
    'VolatilityOfVolatilityStrategy',
    'HedgedVolatilityStrategy',
    'Backtester',
    'RollingPerformance',
    'StressTestAnalyzer',
]

print(f"VIX Trading Strategy v{__version__} loaded successfully!")
