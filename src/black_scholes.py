import numpy as np
from scipy.stats import norm

def bsm_price(S, K, T, r, sigma, option_type="call"):
    #Black-Scholes-Merton pricer. 
    '''
    S - spot price
    K - strike price
    T - time to expiry 
    r - risk-free rate
    sigma - volatility

    returns option price
    '''
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        price = K * np.exp(-r * T) * norm.cdf( -d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("option type must be 'call' or 'put'")
    
    return price 

def put_call_parity_check(S, K, T, r, sigma):
    #verifies put-call parity
    call = bsm_price(S, K, T, r, sigma, "call")
    put  = bsm_price(S, K, T, r, sigma, "put")
    lhs  = call - put
    rhs  = S - K * np.exp(-r * T)
    return {"call": call, "put": put, "lhs": lhs, "rhs": rhs, "diff": abs(lhs - rhs)}
