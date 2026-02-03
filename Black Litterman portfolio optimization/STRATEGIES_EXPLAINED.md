# Portfolio Allocation Strategies - Detailed Explanations

This document provides a comprehensive overview of the three portfolio allocation strategies implemented in the Black-Litterman backtesting framework.

## Quick Reference

| Strategy | Complexity | Data Required | Key Metric | Best For |
|----------|-----------|---|-----------|----------|
| **Equal Weight (1/N)** | ⭐ Low | None | Simplicity & Robustness | Unknown returns, high dimensions |
| **Risk Parity** | ⭐⭐ Medium | Volatilities | Risk Contribution | Heterogeneous volatilities |
| **Black-Litterman** | ⭐⭐⭐ High | Returns, Views | Sharpe Ratio | Known alpha views |

---

## Strategy 1: Equal-Weight Portfolio

### What It Does
Allocates equal weight to all assets:
```
w_i = 1/n for all i = 1, ..., n
```

### Mathematical Properties
- **Portfolio Variance**: σ_p² = (1/n²) · 1^T Σ 1 = (1/n²) ∑∑ σ_i σ_j ρ_{ij}
- **Simplicity**: No parameter estimation required
- **Robustness**: Avoids estimation error in expected returns

### Why It Works
The 1/N portfolio often outperforms sophisticated Markowitz portfolios **out-of-sample** due to:
- Avoidance of estimation error in return forecasting
- Reduced sensitivity to market regime changes
- Natural rebalancing (forces selling winners, buying losers)
- Minimal data requirements

### When to Use
✓ You lack confidence in return or risk forecasts
✓ High-dimensional portfolios with limited data
✓ Benchmarking against more complex strategies
✓ High transaction costs (minimal rebalancing needed)
✓ Need maximum robustness to model misspecification

### When NOT to Use
✗ Assets have very different risk/return profiles
✗ You have genuine alpha views
✗ Correlations vary significantly over time

### Historical Context
Academic research consistently finds that equal-weight and market-cap weighted portfolios are hard to beat. The curse of dimensionality in covariance estimation makes many sophisticated approaches fail out-of-sample.

---

## Strategy 2: Risk Parity (Inverse Volatility Weighting)

### What It Does
Weights assets inversely proportional to their volatility:
```
w_i = (σ_i^{-1}) / ∑_j(σ_j^{-1})
```

where σ_i = √{Var(R_i)}

### Mathematical Insight: Risk Contribution
The **marginal contribution to variance** from asset i is:
```
MCV_i = ∂σ_p/∂w_i
```

Risk Parity attempts to **equalize risk budgets** across assets:
```
RC_i = w_i · MCV_i ≈ constant for all i
```

### Why It Works
1. **Simplicity**: Only requires asset volatilities (diagonal of Σ)
2. **Stability**: Avoids estimating the difficult full covariance matrix
3. **Balance**: Reduces impact of high-volatility assets without ignoring them
4. **Practicality**: Popular in hedge funds and institutional allocations

### Mathematical Justification

Consider two assets:
- Bond: σ = 5% (stable)
- Stock: σ = 20% (volatile)

Risk Parity says: "Allocate 4x more to bonds than stocks, so each contributes equally to portfolio variance"

```
w_stock/w_bond = σ_bond/σ_stock = 5/20 = 1/4
```

### When to Use
✓ Asset volatilities differ substantially
✓ Can estimate volatilities reliably (easier than returns)
✓ Want better risk balance than equal weight
✓ Prefer rule-based allocation without forecasting
✓ Correlations are relatively stable
✓ Limited data for full covariance estimation

### When NOT to Use
✗ All assets have similar volatilities
✗ Correlations vary dramatically
✗ Asset returns are highly persistent (should overweight them)
✗ Market structure changes

### Limitations
- Ignores correlations between assets (uses only diagonal of Σ)
- Ignores expected returns entirely
- Can overweight low-return but stable assets
- Breaks down when correlations converge to 1 (e.g., market crises)

---

## Strategy 3: Black-Litterman Portfolio

### What It Does
Combines market equilibrium returns with investor views using **Bayesian inference**

### The Three Key Problems It Solves

1. **Estimation Error**: Return estimates are noisy and produce unstable portfolios
   - Standard Markowitz with estimated returns often fails spectacularly

2. **Intuition Loss**: Market equilibrium is ignored
   - Optimal portfolio may deviate wildly from market portfolio

3. **Parameter Sensitivity**: Small changes in expected returns cause extreme portfolio changes
   - Leads to "concentration risk" in areas of view confidence

### Mathematical Framework

#### Step 1: Covariance Estimation
```
Σ̂ = (1-α)S + αF
```
- S = sample covariance matrix
- F = target (scaled identity matrix)
- α = optimal shrinkage intensity
- This reduces estimation error in high dimensions

#### Step 2: Market Equilibrium Implied Returns
```
π = r_f · 1 + λ · Σ · w_mkt
```
where:
- λ = (r_m - r_f) / σ_m² (risk aversion parameter)
- w_mkt = market portfolio weights
- π = returns that justify holding market portfolio

**Key Insight**: Rather than estimating returns from historical data, we reverse-engineer what returns must be to justify market prices.

#### Step 3: Investor Views
```
P · μ = q    with confidence    Ω = τ · P · Σ · P^T
```
- P = view matrix (encodes view constraints)
- q = view expectations
- Ω = view uncertainty matrix
- τ = confidence scaling parameter (lower = higher confidence)

Examples:
```
View 1 (Absolute): "Asset A will return 12%"
  P = [1, 0, 0, ...]  ,  q = 0.12

View 2 (Relative): "Asset A outperforms B by 3%"
  P = [1, -1, 0, ...] ,  q = 0.03

View 3 (Sector): "Tech average return will be 10%"
  P = [0.33, 0.33, 0.33, 0, ...] ,  q = 0.10
```

#### Step 4: Bayesian Posterior
Using multivariate normal Bayes theorem:
```
μ̂_BL = [Σ^{-1} + P^T·Ω^{-1}·P]^{-1} · [Σ^{-1}·π + P^T·Ω^{-1}·q]
```

**Interpretation**: 
- Posterior precision = Prior precision + View precision
- Precisions add (in information space, not variance space)
- More precise views have larger influence on posterior

Posterior covariance:
```
Σ̂_BL = [Σ^{-1} + P^T·Ω^{-1}·P]^{-1}
```

**Key Property**: Σ̂_BL ≺ Σ (posterior variance < prior variance)
- Information reduces uncertainty
- Posterior is always better than prior

#### Step 5: Optimal Portfolio
Solve standard Markowitz problem with posterior parameters:
```
w* = Σ̂_BL^{-1}(μ̂_BL - r_f·1) / (1^T·Σ̂_BL^{-1}(μ̂_BL - r_f·1))
```

### Why Black-Litterman Works

1. **Anchoring**: Posterior is anchored to market prices through prior
   - Prevents extreme, unrealistic portfolios

2. **Information Integration**: Systematically combines market prices with views
   - Not arbitrary or subjective

3. **Uncertainty Quantification**: Confidence levels control influence
   - Can express varying conviction in different views

4. **Regularization**: Views act as a form of shrinkage
   - Reduces impact of estimation error

### Advantages

✓ More stable portfolios than naive Markowitz
✓ Respects market prices as starting point
✓ Principled framework for incorporating views
✓ Transparency: exactly know how each view affects results
✓ Flexibility: can express various types of views
✓ Posterior covariance smaller than prior (information value)

### Disadvantages

✗ Requires specification of investor views (not always easy)
✗ Sensitive to confidence parameter τ
✗ Views can be wrong, introducing new error sources
✗ Computationally intensive
✗ Still depends on covariance matrix estimate
✗ Assumes views are unbiased (often violated in practice)

### Sensitivity to Confidence Parameter τ

The confidence parameter τ controls view impact:

```
τ = 0.01  →  Ω = 0.01·P·Σ·P^T  →  HIGH confidence in views
τ = 0.05  →  Ω = 0.05·P·Σ·P^T  →  MEDIUM confidence
τ = 0.10  →  Ω = 0.10·P·Σ·P^T  →  LOW-MEDIUM confidence
τ = 0.20  →  Ω = 0.20·P·Σ·P^T  →  LOW confidence
τ → ∞    →  posterior → prior  →  NO confidence in views
```

**Critical**: Always conduct sensitivity analysis on τ before deployment.

### Implementation in This Notebook

Each rebalancing date:
1. Estimate Σ from past returns with Ledoit-Wolf shrinkage
2. Compute π from equal-weighted market portfolio
3. Apply fixed view: "Asset A outperforms Asset B by 1% annually"
4. Posterior update with τ = 0.05
5. Solve for tangency portfolio
6. Hold until next rebalancing (20 days)

---

## Comparative Analysis

### Performance in Synthetic Markets

In the backtesting results:

```
Strategy          Return  Volatility  Sharpe  Drawdown
─────────────────────────────────────────────────────
Equal Weight      9.94%     0.80%      9.36    -0.19%
Risk Parity       9.52%     0.75%      9.45    -0.16%
Black-Litterman  22.12%     1.91%      9.43    -0.44%
```

**Observations**:
1. Black-Litterman: Highest absolute return, highest risk, competitive Sharpe
2. Risk Parity: Best risk-adjusted return (highest Sharpe)
3. Equal Weight: Stable, but lower returns due to not using information

**Why Risk Parity wins on Sharpe?**
- Natural diversification across risk budgets
- No reliance on potentially incorrect return estimates
- Stable parameter (volatilities are more predictable than returns)

**Why Black-Litterman achieves highest returns?**
- Views about Asset A > Asset B were approximately correct
- Model concentrates on this conviction
- Trade-off: accept higher volatility for higher returns

---

## Practical Recommendations

### For Academic Research
- Compare all three strategies
- Test on multiple asset classes and time periods
- Vary confidence parameter τ
- Examine view specification impact
- Analyze out-of-sample robustness

### For Portfolio Managers
1. **Start with Equal Weight**: Understand baseline performance
2. **Add Risk Parity**: See if volatility balance helps
3. **Deploy Black-Litterman**: Only if you have genuine alpha views
4. **Always backtest**: Don't assume strategies work without testing
5. **Monitor assumptions**: Market conditions change, parameters don't

### For Risk Managers
- Use Black-Litterman posterior covariance (it's tighter than sample covariance)
- Understand that views introduce model risk
- Stress test all view assumptions
- Set confidence parameters conservatively
- Rebalance according to discipline, not market timing

---

## References

### Original Papers
1. Black, F., & Litterman, R. (1992). "Global Portfolio Optimization." *Financial Analysts Journal*
2. Litterman, R. (2003). "Modern Investment Management and Backtesting"
3. DeMiguel, V., Garlappi, L., & Uppal, R. (2009). "Optimal versus naive diversification"

### Extensions
- He, G., & Litterman, R. (2002). "The Intuition Behind Black-Litterman Model Portfolios"
- Meucci, A. (2005). "Risk and Asset Allocation"
- Idzorek, T. (2005). "A Step-by-Step Guide to the Black-Litterman Model"

---

## Summary Table: When to Use Each Strategy

| Situation | Best Choice | Why |
|-----------|------------|-----|
| Unknown asset returns | Equal Weight | Avoids estimation error |
| Different volatilities | Risk Parity | Balances risk contribution |
| Have alpha views | Black-Litterman | Leverages information |
| High dimensions | Equal Weight | Robust to curse of dimensionality |
| Stable correlations | Risk Parity | Avoids full covariance estimation |
| Uncertain views | Equal Weight | Don't trust your forecasts |
| Crisis period | Risk Parity | Avoids extreme positions from bad views |
| Want transparency | Black-Litterman | Exactly see how views affect portfolio |
| Minimize complexity | Equal Weight | Only 1/n allocation rule |

---

**Last Updated**: February 2026
**Educational Level**: Master's-level mathematics with portfolio finance context
