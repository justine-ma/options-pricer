import pytest
from src.black_scholes import bsm_price, put_call_parity_check

# Known values to check against (from textbooks)
def test_call_price_known_value():
    price = bsm_price(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type="call")
    assert abs(price - 10.4506) < 0.001

def test_put_price_known_value():
    price = bsm_price(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type="put")
    assert abs(price - 5.5735) < 0.001

def test_put_call_parity():
    result = put_call_parity_check(S=100, K=100, T=1, r=0.05, sigma=0.2)
    assert result["diff"] < 1e-10

def test_call_intrinsic_value():
    # Deep in-the-money call should approach intrinsic value
    price = bsm_price(S=200, K=100, T=0.001, r=0.05, sigma=0.2, option_type="call")
    assert abs(price - 100) < 1.0

def test_invalid_option_type():
    with pytest.raises(ValueError):
        bsm_price(100, 100, 1, 0.05, 0.2, option_type="banana")