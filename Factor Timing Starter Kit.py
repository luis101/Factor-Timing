# Factor Timing Starter Kit

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt

# Parameters
rolling_window = 240  # 20 years of monthly data
validation_window = 12  # 1 year validation
shrinkage_grid = np.linspace(0.1, 10, 20)  # Shrinkage values

# Simulated placeholder data (replace with real data)
# factor_returns: T x K dataframe
# predictors: T x J dataframe
factor_returns = pd.read_csv("factor_returns.csv", index_col=0, parse_dates=True)
predictors = pd.read_csv("predictors.csv", index_col=0, parse_dates=True)

# Create timing portfolios (K * J)
def create_timing_portfolios(factor_returns, predictors):
    timing_returns = {}
    for f_name in factor_returns.columns:
        for p_name in predictors.columns:
            timing_returns[f"{f_name}_{p_name}"] = factor_returns[f_name] * predictors[p_name].shift(1)
    return pd.DataFrame(timing_returns)

# Compute shrunk covariance matrix using Ledoit-Wolf
def compute_shrunk_cov(X):
    lw = LedoitWolf().fit(X.dropna())
    return lw.covariance_, lw.location_

# Ridge-style weight solution
def compute_ridge_weights(mu, Sigma, shrink_k, diag_penalty):
    T = len(mu)
    penalty_matrix = np.zeros_like(Sigma)
    np.fill_diagonal(penalty_matrix, diag_penalty)
    Sigma_shrunk = Sigma + shrink_k / T * penalty_matrix
    weights = np.linalg.solve(Sigma_shrunk, mu)
    return weights

# Sharpe ratio calculation
def compute_sharpe(returns):
    return returns.mean() / returns.std()

# Main rolling loop
portfolio_returns = []
for t in range(rolling_window, len(factor_returns) - validation_window):
    train_idx = slice(t - rolling_window, t)
    val_idx = slice(t, t + validation_window)
    test_idx = t + validation_window

    # Create training data
    G = create_timing_portfolios(factor_returns.iloc[train_idx], predictors.iloc[train_idx])
    Sigma, mu = compute_shrunk_cov(G)
    diag_penalty = np.diag(Sigma)

    # Validation loop to choose k
    best_k = None
    best_sr = -np.inf
    for k in shrinkage_grid:
        w = compute_ridge_weights(mu, Sigma, k, diag_penalty)
        val_G = create_timing_portfolios(factor_returns.iloc[val_idx], predictors.iloc[val_idx])
        r_val = val_G @ w
        sr = compute_sharpe(r_val)
        if sr > best_sr:
            best_sr = sr
            best_k = k

    # Final weight with best k
    w_final = compute_ridge_weights(mu, Sigma, best_k, diag_penalty)
    G_test = create_timing_portfolios(factor_returns.iloc[[test_idx]], predictors.iloc[[test_idx]])
    r_test = G_test @ w_final
    portfolio_returns.append(r_test.iloc[0])

# Convert to series and analyze performance
returns_series = pd.Series(portfolio_returns, index=factor_returns.index[rolling_window+validation_window:])
print("Sharpe Ratio:", compute_sharpe(returns_series))
returns_series.cumsum().plot(title="Cumulative Return of Timing Strategy")
plt.show()
