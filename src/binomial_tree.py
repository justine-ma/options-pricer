import numpy as np

def binomial_tree(S, K, T, r, sigma, N, option_type="call", american=False):
    """
    Cox-Ross-Rubinstein binomial tree pricer.
    S : float  — spot price
    K : float  — strike price
    T : float  — time to expiry in years
    r : float  — risk-free rate
    sigma : float — volatility
    N : int   — number of time steps
    option_type : str — 'call' or 'put'
    american : bool — True for American, False for European
    
    Returns
    float — option price
    """
    dt = T / N                          # length of each time step
    u  = np.exp(sigma * np.sqrt(dt))    # up factor
    d  = 1 / u                          # down factor (ensures recombining tree)
    q  = (np.exp(r * dt) - d) / (u - d) # risk-neutral probability of up move

    # at step N, stock has moved up j times and down (N-j) times
    stock_at_expiry = S * (u ** np.arange(N, -1, -1)) * (d ** np.arange(0, N+1, 1))

    if option_type == "call":
        option_values = np.maximum(stock_at_expiry - K, 0)
    else:
        option_values = np.maximum(K - stock_at_expiry, 0)

    discount = np.exp(-r * dt)
    for i in range(N - 1, -1, -1):
        # expected value = probability-weighted average of up and down nodes
        option_values = discount * (q * option_values[:-1] + (1 - q) * option_values[1:])

        if american:
            # at each node, check if early exercise is worth more than holding
            stock_at_node = S * (u ** np.arange(i, -1, -1)) * (d ** np.arange(0, i+1, 1))
            if option_type == "call":
                intrinsic = np.maximum(stock_at_node - K, 0)
            else:
                intrinsic = np.maximum(K - stock_at_node, 0)
            option_values = np.maximum(option_values, intrinsic)

    return option_values[0]