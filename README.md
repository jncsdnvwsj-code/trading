Multi-Asset Exotic Option Pricing via GARCH-Copula Pipeline
Project Overview
This project implements a high-fidelity pricing engine for a "Worst-of" Put Option on a basket of mega-cap equities (AAPL, MSFT, GOOGL). Unlike standard Black-Scholes models that assume constant volatility and linear correlations, this framework accounts for two critical market realities: Volatility Clustering and Asymmetric Tail Dependence.

Local Volatility Surface Calibration via Dupire PDE
Project Overview
This project implements a high-fidelity Local Volatility Surface ($\sigma_{loc}$) calibration engine. Unlike the standard Black-Scholes model, which assumes volatility is constant, this framework uses the Dupire Equation to extract a state-dependent volatility surface directly from market-implied option prices.This model is essential for the accurate pricing of path-dependent exotic derivatives (such as Asian or Barrier options) where the "volatility smile" dynamics significantly impact the expected payoff.
