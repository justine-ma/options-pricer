import numpy as np
from src.black_scholes import bsm_price
from src.greeks import vega


def implied_vol_newton(market_price, S, K, T, r, option_type="call",
                       initial_guess=0.2, max_iterations=100, tolerance=1e-6):
    """
    Compute implied volatility using Newton-Raphson.

    Solves g(sigma) = BSM_price(sigma) - market_price = 0.
    Uses vega as the derivative since d(BSM_price)/d(sigma) = vega.

    Parameters
    ----------
    market_price : float — observed market option price
    S : float — spot price
    K : float — strike price
    T : float — time to expiry in years
    r : float — risk-free rate
    option_type : str — 'call' or 'put'
    initial_guess : float — starting sigma (default 0.2)
    max_iterations : int — iteration cap before giving up
    tolerance : float — convergence threshold

    Returns
    -------
    float or None — implied volatility, or None if failed to converge
    """
    sigma = initial_guess

    for i in range(max_iterations):
        price = bsm_price(S, K, T, r, sigma, option_type)
        diff  = price - market_price

        if abs(diff) < tolerance:
            return sigma

        v = vega(S, K, T, r, sigma) / 0.01  # vega is per 1% move, convert back

        if abs(v) < 1e-10:
            # vega too small — Newton step would blow up, give up
            return None

        sigma = sigma - diff / v

        if sigma <= 0:
            # stepped into invalid territory
            return None

    return None  # did not converge


def implied_vol_bisection(market_price, S, K, T, r, option_type="call",
                          sigma_low=1e-4, sigma_high=5.0, tolerance=1e-6,
                          max_iterations=1000):
    """
    Compute implied volatility using bisection.

    Guaranteed to converge as long as the root lies within
    [sigma_low, sigma_high], which holds for all practical option prices.

    Parameters
    ----------
    market_price : float — observed market option price
    S : float — spot price
    K : float — strike price
    T : float — time to expiry in years
    r : float — risk-free rate
    option_type : str — 'call' or 'put'
    sigma_low : float — lower bound for sigma search
    sigma_high : float — upper bound for sigma search
    tolerance : float — convergence threshold
    max_iterations : int — iteration cap

    Returns
    -------
    float or None — implied volatility, or None if no root in bracket
    """
    price_low  = bsm_price(S, K, T, r, sigma_low,  option_type)
    price_high = bsm_price(S, K, T, r, sigma_high, option_type)

    if price_low > market_price:
        # market price below minimum possible BSM price
        return None
    if price_high < market_price:
        # market price above maximum possible BSM price
        return None

    for i in range(max_iterations):
        sigma_mid  = (sigma_low + sigma_high) / 2
        price_mid  = bsm_price(S, K, T, r, sigma_mid, option_type)
        diff       = price_mid - market_price

        if abs(diff) < tolerance:
            return sigma_mid

        if diff > 0:
            sigma_high = sigma_mid
        else:
            sigma_low  = sigma_mid

    return (sigma_low + sigma_high) / 2


def implied_vol(market_price, S, K, T, r, option_type="call"):
    """
    Compute implied volatility using Newton-Raphson with bisection fallback.

    Tries Newton-Raphson first for speed. Falls back to bisection
    if Newton-Raphson fails to converge or steps into invalid territory.

    Parameters
    ----------
    market_price : float — observed market option price
    S : float — spot price
    K : float — strike price
    T : float — time to expiry in years
    r : float — risk-free rate
    option_type : str — 'call' or 'put'

    Returns
    -------
    float or None — implied volatility, or None if both methods fail
    """
    # basic sanity check — option price must exceed intrinsic value
    if option_type == "call":
        intrinsic = max(S - K * np.exp(-r * T), 0)
    else:
        intrinsic = max(K * np.exp(-r * T) - S, 0)

    if market_price <= intrinsic:
        return None

    # try Newton-Raphson first
    iv = implied_vol_newton(market_price, S, K, T, r, option_type)

    if iv is not None and 1e-4 < iv < 5.0:
        return iv

    # fall back to bisection
    return implied_vol_bisection(market_price, S, K, T, r, option_type)