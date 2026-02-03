"""
Black-Litterman Model for portfolio optimization combining market equilibrium
with investor views.
"""
import numpy as np
from scipy.optimize import minimize
import warnings

warnings.filterwarnings('ignore')


class BlackLittermanModel:
    """
    Black-Litterman portfolio optimization model.
    
    Combines market equilibrium expected returns with investor views
    to produce posterior expected returns for portfolio optimization.
    """
    
    def __init__(self, cov_matrix, risk_aversion=2.5, risk_free_rate=0.02):
        """
        Initialize the Black-Litterman model.
        
        Parameters
        ----------
        cov_matrix : np.ndarray
            Covariance matrix of asset returns (n_assets, n_assets)
        risk_aversion : float
            Risk aversion coefficient (default 2.5)
        risk_free_rate : float
            Risk-free rate (default 2%)
        """
        self.cov_matrix = cov_matrix
        self.risk_aversion = risk_aversion
        self.risk_free_rate = risk_free_rate
        self.n_assets = cov_matrix.shape[0]
        self.market_weights = None
        self.equilibrium_returns = None
        self.posterior_returns = None
        
    def set_market_weights(self, market_weights):
        """
        Set market capitalization weights.
        
        Parameters
        ----------
        market_weights : np.ndarray
            Market weights of assets
        """
        self.market_weights = market_weights / market_weights.sum()
    
    def calculate_equilibrium_returns(self):
        """
        Calculate implied equilibrium returns from market weights.
        
        Returns
        -------
        np.ndarray
            Implied equilibrium returns
        """
        if self.market_weights is None:
            raise ValueError("Market weights not set")
        
        # Implied returns = risk_aversion * Cov * market_weights
        self.equilibrium_returns = (
            self.risk_aversion * 
            np.dot(self.cov_matrix, self.market_weights)
        )
        
        return self.equilibrium_returns
    
    def add_view(self, view_matrix, view_return, confidence=1.0):
        """
        Add investor views to the model.
        
        Parameters
        ----------
        view_matrix : np.ndarray
            View matrix P (m_views, n_assets)
            Each row represents a view
            E.g., [1, -1, 0] means asset 1 outperforms asset 2
        view_return : np.ndarray or float
            Expected return from the view
        confidence : float or np.ndarray
            Confidence in the view (0-1) or uncertainty matrix diagonal
        
        Returns
        -------
        None
        """
        if not hasattr(self, 'views'):
            self.views = []
        
        view_matrix = np.atleast_2d(view_matrix)
        view_return = np.atleast_1d(view_return)
        
        self.views.append({
            'P': view_matrix,
            'Q': view_return,
            'confidence': confidence
        })
    
    def fit(self, views=None, view_confidences=None, use_equilibrium=True):
        """
        Fit the Black-Litterman model with views.
        
        Parameters
        ----------
        views : list, optional
            List of (P, Q) tuples representing views
        view_confidences : list, optional
            Confidence levels for each view
        use_equilibrium : bool
            Whether to use equilibrium returns as prior
        
        Returns
        -------
        np.ndarray
            Posterior expected returns
        """
        if use_equilibrium and self.equilibrium_returns is None:
            self.calculate_equilibrium_returns()
        
        prior_returns = self.equilibrium_returns if use_equilibrium else np.zeros(self.n_assets)
        
        if not hasattr(self, 'views') or len(self.views) == 0:
            self.posterior_returns = prior_returns
            return prior_returns
        
        # Combine all views
        P_list = [v['P'] for v in self.views]
        Q_list = [v['Q'] for v in self.views]
        
        P = np.vstack(P_list) if P_list else np.zeros((1, self.n_assets))
        Q = np.concatenate(Q_list) if Q_list else np.array([0])
        
        # Uncertainty in views - based on confidence
        if view_confidences is None:
            view_confidences = [v['confidence'] for v in self.views]
        
        # Omega: uncertainty matrix (diagonal)
        # Omega[i,i] = P[i] * Cov * P[i].T / confidence[i]
        omega_diag = []
        for i, (p_row, conf) in enumerate(zip(P, view_confidences)):
            view_variance = np.dot(p_row, np.dot(self.cov_matrix, p_row))
            omega_i = view_variance / (conf if conf > 0 else 1)
            omega_diag.append(omega_i)
        
        omega = np.diag(omega_diag)
        
        # Black-Litterman formula
        # Posterior mean = Prior + Cov * P.T * (P * Cov * P.T + Omega)^-1 * (Q - P * Prior)
        
        P_cov_PT = np.dot(P, np.dot(self.cov_matrix, P.T))
        
        try:
            inv_term = np.linalg.inv(P_cov_PT + omega)
            adjustment = np.dot(
                np.dot(self.cov_matrix, P.T),
                np.dot(inv_term, Q - np.dot(P, prior_returns))
            )
            self.posterior_returns = prior_returns + adjustment
        except np.linalg.LinAlgError:
            # If matrix is singular, return prior
            self.posterior_returns = prior_returns
        
        return self.posterior_returns
    
    def get_posterior_covariance(self):
        """
        Calculate posterior covariance matrix.
        
        Returns
        -------
        np.ndarray
            Posterior covariance matrix
        """
        if not hasattr(self, 'views') or len(self.views) == 0:
            return self.cov_matrix
        
        # Combine views
        P_list = [v['P'] for v in self.views]
        P = np.vstack(P_list) if P_list else np.zeros((1, self.n_assets))
        
        # Omega calculation
        view_confidences = [v['confidence'] for v in self.views]
        omega_diag = []
        for p_row, conf in zip(P, view_confidences):
            view_variance = np.dot(p_row, np.dot(self.cov_matrix, p_row))
            omega_i = view_variance / (conf if conf > 0 else 1)
            omega_diag.append(omega_i)
        
        omega = np.diag(omega_diag)
        
        # Posterior covariance is adjusted from prior
        P_cov_PT = np.dot(P, np.dot(self.cov_matrix, P.T))
        
        try:
            inv_term = np.linalg.inv(P_cov_PT + omega)
            adjustment = np.dot(P, np.dot(self.cov_matrix, P.T))
            adjustment = np.dot(adjustment, np.dot(inv_term, adjustment))
            posterior_cov = self.cov_matrix - adjustment / (1 + self.risk_aversion)
        except np.linalg.LinAlgError:
            posterior_cov = self.cov_matrix
        
        return posterior_cov
    
    def optimize_bl_portfolio(self, constraints=None, bounds=None):
        """
        Optimize portfolio using posterior returns.
        
        Parameters
        ----------
        constraints : dict, optional
            Optimization constraints
        bounds : Bounds, optional
            Weight bounds
        
        Returns
        -------
        dict
            Optimization result
        """
        if self.posterior_returns is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        from mean_variance import MeanVarianceOptimizer
        
        optimizer = MeanVarianceOptimizer(
            self.posterior_returns,
            self.cov_matrix,
            self.risk_free_rate
        )
        
        return optimizer.optimize_max_sharpe(constraints, bounds)
