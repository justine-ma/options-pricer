from src.binomial_tree import binomial_tree
from src.black_scholes import bsm_price

def test_european_call_converges_to_bsm():
    # with enough steps, binomial tree should match BSM closely
    bsm   = bsm_price(100, 100, 1, 0.05, 0.2, "call")
    tree  = binomial_tree(100, 100, 1, 0.05, 0.2, N=1000, option_type="call")
    assert abs(tree - bsm) < 0.01

def test_european_put_converges_to_bsm():
    bsm  = bsm_price(100, 100, 1, 0.05, 0.2, "put")
    tree = binomial_tree(100, 100, 1, 0.05, 0.2, N=1000, option_type="put")
    assert abs(tree - bsm) < 0.01

def test_american_put_greater_than_european():
    # American put should always be worth at least as much as European
    european = binomial_tree(100, 100, 1, 0.05, 0.2, N=200, american=False, option_type="put")
    american = binomial_tree(100, 100, 1, 0.05, 0.2, N=200, american=True,  option_type="put")
    assert american >= european

def test_american_call_no_early_exercise():
    # American call on non-dividend stock should equal European call
    european = binomial_tree(100, 100, 1, 0.05, 0.2, N=200, american=False, option_type="call")
    american = binomial_tree(100, 100, 1, 0.05, 0.2, N=200, american=True,  option_type="call")
    assert abs(american - european) < 0.001