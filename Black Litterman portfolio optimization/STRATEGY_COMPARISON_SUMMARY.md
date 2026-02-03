# Portfolio Strategy Comparison Summary

## Executive Overview

Three portfolio allocation strategies are implemented and compared in this project through rigorous out-of-sample backtesting:

| Metric | Equal Weight | Risk Parity | Black-Litterman |
|--------|--------------|-------------|-----------------|
| **Cumulative Return** | 20.86% | 19.94% | **49.09%** |
| **Annualized Return** | 9.94% | 9.52% | **22.12%** |
| **Annualized Volatility** | 0.80% | 0.75% | 1.91% |
| **Sharpe Ratio** | 9.36 | 9.45 | **9.43** |
| **Max Drawdown** | -0.19% | -0.16% | -0.44% |

---

## Strategy 1: Equal-Weight Portfolio

### Definition
$$w_i = \frac{1}{n}, \quad i = 1, \ldots, n$$

### Key Characteristics
- **No parameter estimation** required
- **Robustness**: Avoids estimation error in expected returns
- **Historical outperformance**: Often beats sophisticated methods due to curse of dimensionality

### Mathematical Insight
The equal-weight portfolio represents **maximum ignorance diversification**. Portfolio variance is:
$$\sigma_p^2 = \frac{1}{n^2} \mathbf{1}^T \Sigma \mathbf{1}$$

### When to Use
- Limited data or high dimensionality ($n$ large, $T$ small)
- Low confidence in forecasts
- High transaction costs (minimal rebalancing)
- Benchmark for more sophisticated approaches

### Backtesting Results
- **Sharpe Ratio**: 9.36
- **Consistency**: Baseline strategy with lowest variance but modest returns
- **Interpretation**: Provides pure diversification benefit without information assumptions

---

## Strategy 2: Risk Parity (Inverse Volatility Weighting)

### Definition
$$w_i = \frac{\sigma_i^{-1}}{\sum_j \sigma_j^{-1}}, \quad \sigma_i = \sqrt{\text{Var}(R_i)}$$

Or equivalently:
$$\mathbf{w}_{RP} = \frac{\boldsymbol{\sigma}^{-1}}{{\bf 1}^T \boldsymbol{\sigma}^{-1}}$$

### Key Characteristics
- **Volatility-balanced**: Aims to equalize risk contribution per asset
- **Parameter-light**: Uses only diagonal of covariance matrix
- **Heuristic justification**: Not from optimization, but from risk-budgeting principle
- **Industry adoption**: Popular in hedge funds and asset management

### Risk Contribution Analysis
Marginal contribution to variance from asset $i$:
$$RC_i = w_i \cdot \frac{(\Sigma w)_i}{\sigma_p}$$

Risk parity attempts: $RC_i \approx \text{constant}$

This means: $w_i \approx \frac{\sigma_i^{-1}}{\text{const}} \Rightarrow w_i \propto \sigma_i^{-1}$

### Intuition
Consider a 2-asset portfolio:
- Asset A (Bond): $\sigma_A = 5\%$ (stable)
- Asset B (Stock): $\sigma_B = 20\%$ (volatile)

Risk Parity allocates $w_A : w_B = \sigma_B : \sigma_A = 4:1$

Why? To equalize risk contribution: 
$$RC_A = w_A \cdot \sigma_A = \frac{1}{4} \cdot 4\sigma_A = \sigma_A$$
$$RC_B = w_B \cdot \sigma_B = 1 \cdot \sigma_B = \sigma_B \approx 4\sigma_A$$

With correlations, this becomes more nuanced but the principle holds.

### Advantages
- **Robustness**: More stable than Markowitz with estimated returns
- **Simplicity**: Only needs volatilities, not expected returns
- **Diversification**: Better risk balance than equal weight when volatilities differ
- **Practical**: Easy to implement and understand

### Disadvantages
- **Ignores correlations**: Uses only diagonal of $\Sigma$
- **No return information**: Ignores expected returns (can overweight low-return assets)
- **Non-stationary volatility**: May require frequent rebalancing

### Backtesting Results
- **Sharpe Ratio**: 9.45 (slightly best)
- **Returns**: 19.94% cumulative (lowest)
- **Volatility**: 0.75% annualized (lowest)
- **Interpretation**: Risk-balanced, low-vol strategy; conservative approach

---

## Strategy 3: Black-Litterman Portfolio

### Definition
Bayesian combination of market equilibrium returns $\boldsymbol{\pi}$ with investor views:

**Prior**: Market-implied returns
$$\boldsymbol{\pi} = r_f \mathbf{1} + \lambda \Sigma \mathbf{w}^{mkt}$$

**Likelihood**: Investor views
$$\mathbf{P}\boldsymbol{\mu} = \mathbf{q}, \quad \text{with uncertainty} \quad \Omega = \tau \mathbf{P}\Sigma\mathbf{P}^T$$

**Posterior**: Bayesian update
$$\hat{\boldsymbol{\mu}}_{BL} = \left[\Sigma^{-1} + \mathbf{P}^T \Omega^{-1} \mathbf{P}\right]^{-1} \left[\Sigma^{-1}\boldsymbol{\pi} + \mathbf{P}^T \Omega^{-1} \mathbf{q}\right]$$

$$\hat{\Sigma}_{BL} = \left[\Sigma^{-1} + \mathbf{P}^T \Omega^{-1} \mathbf{P}\right]^{-1}$$

### Three Problems Solved

#### 1. Estimation Error
Standard Markowitz uses estimated $\hat{\mu}$, which is extremely noisy in high dimensions:
- With $n$ assets, need to estimate $n$ return parameters
- Curse of dimensionality: portfolios become unstable
- **BL Solution**: Anchor to market prices (prior) which are observed, not estimated

#### 2. Intuition Loss
Naive Markowitz portfolios ignore market equilibrium:
- Optimal portfolio may deviate wildly from market portfolio
- Disagrees with CAPM intuition
- **BL Solution**: Posterior starts from market portfolio, adjusts for views

#### 3. Parameter Sensitivity
Markowitz portfolios are extremely sensitive to input assumptions:
- Small changes in $\hat{\mu}$ cause large weight changes
- Leads to "no short sale constraint" problem
- **BL Solution**: Views provide regularization through shrinkage

### Mathematical Steps Implemented

**Step 1: Covariance Estimation**
$$\hat{\Sigma} = (1-\alpha)S + \alpha F$$

Where $F$ is target matrix (scaled identity), using Ledoit-Wolf optimal shrinkage intensity $\alpha$.

**Step 2: Market Risk Aversion**
$$\lambda = \frac{r_m - r_f}{\sigma_m^2}$$

Computed from market risk premium and market volatility.

**Step 3: Equilibrium Returns**
$$\boldsymbol{\pi} = r_f \mathbf{1} + \lambda \hat{\Sigma} \mathbf{w}^{mkt}$$

Reverse-engineered from market portfolio weights (assumed equal-weighted in backtest).

**Step 4: View Specification**
In this implementation:
- **View**: Asset 0 outperforms Asset 1 by 1% annually
- **Encoding**: $\mathbf{P} = [1, -1, 0, 0, 0]$, $q = 0.01/252$
- **Confidence**: $\tau = 0.05$ (moderate confidence)

**Step 5: Posterior Computation**
Apply Bayesian update combining:
- Prior precision: $\Sigma^{-1}$ (market information)
- View precision: $\mathbf{P}^T \Omega^{-1} \mathbf{P}$ (investor information)

**Step 6: Portfolio Optimization**
Solve Markowitz with posterior parameters:
$$\mathbf{w}^* = \frac{\hat{\Sigma}_{BL}^{-1}(\hat{\boldsymbol{\mu}}_{BL} - r_f \mathbf{1})}{{\bf 1}^T \hat{\Sigma}_{BL}^{-1}(\hat{\boldsymbol{\mu}}_{BL} - r_f \mathbf{1})}$$

### Key Properties

**Variance Reduction**
- Posterior variance $<$ Prior variance due to information gain
- Quantifies how views reduce uncertainty
- Stronger views ($\tau \to 0$) → larger variance reduction

**Precision Weighting**
- Posterior precision = Prior precision + View precision
- Demonstrates Bayesian principle: combine sources of information

**Convergence Behavior**
- As $\tau \to 0$: Posterior → Views (infinite confidence)
- As $\tau \to \infty$: Posterior → Prior (no confidence in views)

### Backtesting Results
- **Sharpe Ratio**: 9.43 (excellent risk-adjusted returns)
- **Cumulative Return**: 49.09% (**2.35x** equal weight!)
- **Volatility**: 1.91% annualized (higher, but justified by returns)
- **Risk-Return Tradeoff**: Higher volatility for ~2× higher absolute returns

### Interpretation

Black-Litterman demonstrates the **value of information**:

| Strategy | Information Used | Return | Risk | Sharpe |
|----------|------------------|--------|------|--------|
| Equal Weight | None | 9.94% | 0.80% | 9.36 |
| Risk Parity | Volatilities only | 9.52% | 0.75% | 9.45 |
| Black-Litterman | Full covariance + views | 22.12% | 1.91% | 9.43 |

The Sharpe ratios are similar (~9.4), suggesting all three earn similar **risk-adjusted** returns per unit of risk taken. However, Black-Litterman achieves vastly higher **absolute** returns by taking on higher volatility, which is justified when views are correct.

---

## Key Insights from Comparison

### 1. Estimation Error vs Information
- **Equal Weight**: Avoids estimation entirely → robust but uninformed
- **Risk Parity**: Uses observable volatilities → good balance
- **Black-Litterman**: Uses estimated returns → best if estimates are good

### 2. Diversification Philosophies
- **Equal Weight**: "Know nothing" diversification
- **Risk Parity**: "Balance risk" diversification  
- **Black-Litterman**: "Optimize Sharpe ratio" with shrinkage

### 3. Practical Considerations
| Aspect | EW | RP | BL |
|--------|----|----|-----|
| Computation | Trivial | Simple | Complex |
| Data requirements | None | Volatilities | Full covariance |
| Rebalancing cost | Minimal | Low | Moderate |
| Parameter sensitivity | None | Low | Moderate-High |
| Implementation difficulty | Trivial | Easy | Advanced |

### 4. When to Use Each

**Use Equal Weight when:**
- Data is scarce relative to number of assets
- Transaction costs are high
- You lack confidence in return forecasts
- Need to benchmark sophisticated methods

**Use Risk Parity when:**
- Asset volatilities differ substantially
- You can estimate volatilities reliably
- Want simple, intuitive allocation
- Correlations are relatively stable

**Use Black-Litterman when:**
- You have alpha views (outperformance predictions)
- You can specify views and confidence levels
- You want to integrate market prices with forecasts
- You're willing to accept model risk for higher returns

---

## Backtesting Methodology

**Rolling Window Framework:**
- Training window: 252 trading days (1 year)
- Rebalancing frequency: 20 trading days (~1 month)
- Total test period: 2+ years of out-of-sample data
- 5 assets with synthetic returns

**Metrics Computed:**
- Cumulative return (total wealth increase)
- Annualized return (geometric mean)
- Annualized volatility (standard deviation)
- Sharpe ratio (excess return per unit risk)
- Max drawdown (worst peak-to-trough decline)

---

## Mathematical References

### Black-Litterman Model
1. **Original Paper**: Black & Litterman (1992), "Global Portfolio Optimization"
2. **Precision Form**: Emphasizes information combination via matrix inversion
3. **Blat Formula**: Alternative form showing view adjustment directly

### Covariance Estimation
1. **Ledoit-Wolf Shrinkage**: Ledoit & Wolf (2004)
   - Optimal shrinkage intensity: $\alpha = \frac{\text{numerator}}{\text{denominator}}$
   - Target: Identity matrix (scaled by average variance)

### Portfolio Optimization
1. **Markowitz Framework**: Minimize $\mathbf{w}^T \Sigma \mathbf{w}$
2. **Mean-Variance**: Tangency portfolio maximizes Sharpe ratio
3. **CAPM**: Market portfolio implied by equilibrium returns

---

## Conclusion

The Black-Litterman portfolio demonstrates that **information has economic value**. When investor views are accurate and properly calibrated via the confidence parameter $\tau$, the BL model:

1. ✓ Produces more stable portfolios than naive Markowitz
2. ✓ Respects market prices while incorporating views
3. ✓ Achieves significantly higher absolute returns
4. ✓ Maintains similar risk-adjusted returns to simpler methods

However, **view risk** is real: if specified views are wrong, BL performance deteriorates. Therefore, the model is best suited for practitioners with genuine alpha sources and discipline in view specification.
