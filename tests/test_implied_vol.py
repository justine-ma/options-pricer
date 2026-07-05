import pytest
import numpy as np
from src.black_scholes import bsm_price
from src.implied_vol import implied_vol, implied_vol_newton, implied_vol_bisection

# round-trip test — compute BSM price then recover sigma
def test_round_trip_call():
    true_sigma = 0.25
    price = bsm_price(100, 100, 1, 0.05, true_sigma, "call")
    iv    = implied_vol(price, 100, 100, 1, 0.05, "call")
    assert abs(iv - true_sigma) < 1e-5

def test_round_trip_put():
    true_sigma = 0.30
    price = bsm_price(100, 100, 1, 0.05, true_sigma, "put")
    iv    = implied_vol(price, 100, 100, 1, 0.05, "put")
    assert abs(iv - true_sigma) < 1e-5

def test_round_trip_otm_call():
    # out of the money
    true_sigma = 0.20
    price = bsm_price(100, 120, 1, 0.05, true_sigma, "call")
    iv    = implied_vol(price, 100, 120, 1, 0.05, "call")
    assert abs(iv - true_sigma) < 1e-5

def test_round_trip_itm_put():
    # in the money
    true_sigma = 0.20
    price = bsm_price(100, 120, 1, 0.05, true_sigma, "put")
    iv    = implied_vol(price, 100, 120, 1, 0.05, "put")
    assert abs(iv - true_sigma) < 1e-5

def test_newton_and_bisection_agree():
    price = bsm_price(100, 100, 1, 0.05, 0.20, "call")
    iv_nr = implied_vol_newton(price, 100, 100, 1, 0.05)
    iv_bs = implied_vol_bisection(price, 100, 100, 1, 0.05)
    assert abs(iv_nr - iv_bs) < 1e-4

def test_below_intrinsic_returns_none():
    # price below intrinsic value has no implied vol
    iv = implied_vol(0.01, 100, 50, 1, 0.05, "call")
    assert iv is None

def test_various_volatilities():
    # implied vol should recover correctly across a range of true vols
    for true_sigma in [0.10, 0.20, 0.30, 0.50, 0.80]:
        price = bsm_price(100, 100, 1, 0.05, true_sigma, "call")
        iv    = implied_vol(price, 100, 100, 1, 0.05, "call")
        assert abs(iv - true_sigma) < 1e-5, f"Failed for sigma={true_sigma}"