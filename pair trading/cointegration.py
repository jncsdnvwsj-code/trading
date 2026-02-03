"""
Cointegration analysis for pairs trading.

Implements:
- Augmented Dickey-Fuller test
- Johansen cointegration test
- Pairs selection based on cointegration
"""
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, coint, grangercausalitytests
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from scipy import stats
import logging
from typing import Tuple, Dict, List
from utils import calculate_half_life, normalize_prices

logger = logging.getLogger(__name__)


class CointegrationAnalyzer:
    """Analyze cointegration between asset pairs."""
    
    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level
        self.adf_threshold = 0.05  # p-value threshold
        self.coint_threshold = 0.05
        
    def test_stationarity(self, series: pd.Series, name: str = "Series") -> Dict:
        """
        Augmented Dickey-Fuller test for stationarity.
        
        Returns:
        --------
        Dict with test results
        """
        try:
            result = adfuller(series.dropna(), maxlag="AIC", regression="c", autolag="AIC")
            
            return {
                'test_statistic': result[0],
                'p_value': result[1],
                'n_lags': result[2],
                'n_obs': result[3],
                'critical_values': result[4],
                'is_stationary': result[1] < self.adf_threshold,
                'interpretation': f"{name}: {'Stationary' if result[1] < self.adf_threshold else 'Non-stationary'} (p={result[1]:.4f})"
            }
        except Exception as e:
            logger.error(f"ADF test failed for {name}: {e}")
            return {'is_stationary': False, 'error': str(e)}
    
    def test_cointegration(self, price1: pd.Series, price2: pd.Series, 
                          name1: str = "Asset1", name2: str = "Asset2") -> Dict:
        """
        Engle-Granger cointegration test.
        
        H0: No cointegration
        H1: Cointegration exists
        
        Returns:
        --------
        Dict with cointegration test results and optimal hedge ratio
        """
        try:
            # Clean data
            data = pd.DataFrame({'p1': price1, 'p2': price2}).dropna()
            
            if len(data) < 50:
                logger.warning(f"Insufficient data for cointegration test: {len(data)} samples")
                return {'is_cointegrated': False, 'error': 'Insufficient data'}
            
            # Engle-Granger test
            score, pvalue, _ = coint(data['p1'], data['p2'])
            
            # Calculate optimal hedge ratio (beta from regression)
            hedge_ratio = np.polyfit(data['p2'], data['p1'], 1)[0]
            
            # Calculate spread with optimal hedge ratio
            spread = data['p1'] - hedge_ratio * data['p2']
            
            # Test if spread is stationary
            adf_result = adfuller(spread, autolag="AIC")
            spread_pvalue = adf_result[1]
            
            is_cointegrated = pvalue < self.coint_threshold
            
            # Calculate half-life of mean reversion
            half_life = calculate_half_life(pd.Series(spread))
            
            return {
                'is_cointegrated': is_cointegrated,
                'test_statistic': score,
                'p_value': pvalue,
                'hedge_ratio': hedge_ratio,
                'spread_pvalue': spread_pvalue,
                'half_life': half_life,
                'spread_mean': spread.mean(),
                'spread_std': spread.std(),
                'interpretation': f"{name1} & {name2}: {'Cointegrated' if is_cointegrated else 'Not cointegrated'} (p={pvalue:.4f}, hedge_ratio={hedge_ratio:.4f})"
            }
        except Exception as e:
            logger.error(f"Cointegration test failed: {e}")
            return {'is_cointegrated': False, 'error': str(e)}
    
    def johansen_test(self, data: pd.DataFrame, det_order: int = 0, 
                     k_ar_diff: int = 1) -> Dict:
        """
        Johansen cointegration test for multivariate analysis.
        Better for finding cointegrating relationships in > 2 assets.
        
        Returns:
        --------
        Dict with eigenvalues and cointegrating vectors
        """
        try:
            data_clean = data.dropna()
            
            if len(data_clean) < 50:
                return {'error': 'Insufficient data'}
            
            result = coint_johansen(data_clean, det_order=det_order, k_ar_diff=k_ar_diff)
            
            # Trace test
            trace_stat = result.lr1[:, 0]  # Trace test statistic
            trace_crit = result.cvt[:, 1]  # 95% critical value
            
            # Maximum eigenvalue test
            eigen_stat = result.lr2[:, 0]
            eigen_crit = result.cvm[:, 1]
            
            n_cointegrating = sum(trace_stat > trace_crit)
            
            return {
                'n_cointegrating': n_cointegrating,
                'trace_stat': trace_stat,
                'trace_crit': trace_crit,
                'eigen_stat': eigen_stat,
                'eigen_crit': eigen_crit,
                'evec': result.evec,  # Cointegrating vectors
                'eval': result.evals  # Eigenvalues
            }
        except Exception as e:
            logger.error(f"Johansen test failed: {e}")
            return {'error': str(e)}
    
    def find_cointegrated_pairs(self, symbols: Dict[str, pd.Series], 
                               top_n: int = 10) -> List[Tuple]:
        """
        Find all cointegrated pairs from a list of symbols.
        
        Parameters:
        -----------
        symbols : Dict[str, pd.Series]
            Dictionary of symbol -> price series
        top_n : int
            Return top N cointegrated pairs
        
        Returns:
        --------
        List of (symbol1, symbol2, p_value, hedge_ratio) tuples
        """
        symbol_names = list(symbols.keys())
        n = len(symbol_names)
        cointegrated = []
        
        for i in range(n):
            for j in range(i + 1, n):
                sym1, sym2 = symbol_names[i], symbol_names[j]
                price1, price2 = symbols[sym1], symbols[sym2]
                
                # Normalize prices
                price1_norm = normalize_prices(price1)
                price2_norm = normalize_prices(price2)
                
                result = self.test_cointegration(price1_norm, price2_norm, sym1, sym2)
                
                if result.get('is_cointegrated'):
                    cointegrated.append({
                        'symbol1': sym1,
                        'symbol2': sym2,
                        'p_value': result['p_value'],
                        'hedge_ratio': result['hedge_ratio'],
                        'half_life': result['half_life'],
                        'spread_std': result['spread_std']
                    })
        
        # Sort by p-value (strongest cointegration)
        cointegrated.sort(key=lambda x: x['p_value'])
        
        logger.info(f"Found {len(cointegrated)} cointegrated pairs out of {n*(n-1)//2} possible")
        return cointegrated[:top_n]


class PairsSelector:
    """Select optimal pairs for trading."""
    
    def __init__(self, min_half_life: float = 5, max_half_life: float = 252):
        """
        min_half_life: Minimum half-life in trading periods (too quick reversion = high turnover)
        max_half_life: Maximum half-life (too slow = capital inefficient)
        """
        self.min_half_life = min_half_life
        self.max_half_life = max_half_life
    
    def score_pair(self, coint_result: Dict) -> float:
        """Score a pair based on multiple criteria."""
        score = 0.0
        
        # P-value contribution (lower is better)
        p_value = coint_result.get('p_value', 1.0)
        score -= p_value * 100  # More negative = stronger cointegration
        
        # Half-life contribution (should be in optimal range)
        half_life = coint_result.get('half_life', np.inf)
        if self.min_half_life <= half_life <= self.max_half_life:
            score += 50  # Bonus for optimal half-life
        else:
            score -= abs(half_life - (self.min_half_life + self.max_half_life) / 2) / 10
        
        # Spread volatility (lower is better for mean reversion)
        spread_std = coint_result.get('spread_std', 1.0)
        score -= spread_std * 10
        
        return score
    
    def select_best_pairs(self, cointegrated_pairs: List[Dict], 
                         top_n: int = 5) -> List[Dict]:
        """Select best pairs based on scoring criteria."""
        scored = []
        for pair in cointegrated_pairs:
            pair['score'] = self.score_pair(pair)
            scored.append(pair)
        
        scored.sort(key=lambda x: x['score'], reverse=True)
        return scored[:top_n]
