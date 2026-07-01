import numpy as np

def monte_carlo(S, K, T, r, sigma, n_simulations=10000, n_steps=252,
                option_type="call", seed=42):
    """
    Monte Carlo option pricer using Geometric Brownian Motion.

    S : float — spot price
    K : float — strike price
    T : float — time to expiry in years
    r : float — risk-free rate
    sigma : float — volatility
    n_simulations : int — number of simulated paths
    n_steps : int — number of time steps per path
    option_type : str — 'call' or 'put'
    seed : int — random seed for reproducibility
    
    Returns
    dict — price, standard error, and 95% confidence interval
    """
    rng = np.random.default_rng(seed)
    dt  = T / n_steps

    # --- Step 1: simulate log returns ---
    # GBM: dS = S(r dt + sigma dW)
    # in log space: d(lnS) = (r - 0.5*sigma^2)dt + sigma*sqrt(dt)*Z
    Z = rng.standard_normal((n_simulations, n_steps))
    log_returns = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z

    # --- Step 2: compute terminal stock prices ---
    log_paths       = np.cumsum(log_returns, axis=1)
    terminal_prices = S * np.exp(log_paths[:, -1])

    # --- Step 3: compute discounted payoffs ---
    if option_type == "call":
        payoffs = np.maximum(terminal_prices - K, 0)
    else:
        payoffs = np.maximum(K - terminal_prices, 0)

    discounted_payoffs = np.exp(-r * T) * payoffs

    # --- Step 4: estimate price and confidence interval ---
    price = np.mean(discounted_payoffs)
    se    = np.std(discounted_payoffs) / np.sqrt(n_simulations)

    return {
        "price": price,
        "std_error": se,
        "ci_lower": price - 1.96 * se,
        "ci_upper": price + 1.96 * se
    }


def monte_carlo_antithetic(S, K, T, r, sigma, n_simulations=10000,
                            n_steps=252, option_type="call", seed=42):
    """
    Monte Carlo with antithetic variates for variance reduction.
    For every random path Z, also simulate -Z. The two are
    negatively correlated, which reduces the standard error.
    
    The key implementation detail is that standard error is computed
    across pair averages (Z + -Z)/2, not across individual payoffs.
    This is what actually captures the variance reduction — treating
    all payoffs as independent ignores the pairing and gives the same
    standard error as plain Monte Carlo.
    """
    rng = np.random.default_rng(seed)
    dt  = T / n_steps

    # generate half the paths, pair with their negatives
    Z        = rng.standard_normal((n_simulations // 2, n_steps))
    Z_paired = np.vstack([Z, -Z])

    log_returns     = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z_paired
    terminal_prices = S * np.exp(np.cumsum(log_returns, axis=1)[:, -1])

    if option_type == "call":
        payoffs = np.maximum(terminal_prices - K, 0)
    else:
        payoffs = np.maximum(K - terminal_prices, 0)

    discounted_payoffs = np.exp(-r * T) * payoffs

    # average each (Z, -Z) pair first, then compute statistics across pair averages
    # this is the critical step — variance reduction only shows up when you
    # account for the pairing structure rather than treating all paths as independent
    paired_means = discounted_payoffs.reshape(2, n_simulations // 2).mean(axis=0)

    price = paired_means.mean()
    se    = paired_means.std() / np.sqrt(n_simulations // 2)

    return {
        "price": price,
        "std_error": se,
        "ci_lower": price - 1.96 * se,
        "ci_upper": price + 1.96 * se
    }