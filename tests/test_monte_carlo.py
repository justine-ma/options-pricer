from src.monte_carlo import monte_carlo, monte_carlo_antithetic
from src.black_scholes import bsm_price

def test_mc_call_close_to_bsm():
    bsm = bsm_price(100, 100, 1, 0.05, 0.2, "call")
    mc  = monte_carlo(100, 100, 1, 0.05, 0.2, n_simulations=50000)
    assert abs(mc["price"] - bsm) < 0.1

def test_mc_put_close_to_bsm():
    bsm = bsm_price(100, 100, 1, 0.05, 0.2, "put")
    mc  = monte_carlo(100, 100, 1, 0.05, 0.2, n_simulations=50000, option_type="put")
    assert abs(mc["price"] - bsm) < 0.1

def test_antithetic_lower_std_error():
    # antithetic should always produce a lower standard error
    mc_standard   = monte_carlo(100, 100, 1, 0.05, 0.2, n_simulations=10000)
    mc_antithetic = monte_carlo_antithetic(100, 100, 1, 0.05, 0.2, n_simulations=10000)
    assert mc_antithetic["std_error"] < mc_standard["std_error"]

def test_bsm_within_mc_confidence_interval():
    bsm = bsm_price(100, 100, 1, 0.05, 0.2, "call")
    mc  = monte_carlo(100, 100, 1, 0.05, 0.2, n_simulations=50000)
    assert mc["ci_lower"] < bsm < mc["ci_upper"]