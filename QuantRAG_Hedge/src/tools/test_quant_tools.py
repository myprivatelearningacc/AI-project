from src.tools.black_scholes import black_scholes_call, black_scholes_put
from src.tools.greeks import calculate_delta, calculate_gamma, calculate_vega
from src.tools.monte_carlo import simulate_gbm, monte_carlo_price
from src.tools.delta_hedging import hedge_shares_needed


def test_black_scholes_call():
    price = black_scholes_call(S=100, K=100, T=1, r=0.05, sigma=0.2)
    assert round(price, 4) == 10.4506


def test_black_scholes_put():
    price = black_scholes_put(S=100, K=100, T=1, r=0.05, sigma=0.2)
    assert round(price, 4) == 5.5735


def test_delta_call():
    delta = calculate_delta(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type="call")
    assert round(delta, 4) == 0.6368


def test_delta_put():
    delta = calculate_delta(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type="put")
    assert round(delta, 4) == -0.3632


def test_gamma():
    gamma = calculate_gamma(S=100, K=100, T=1, r=0.05, sigma=0.2)
    assert round(gamma, 4) == 0.0188


def test_vega():
    vega = calculate_vega(S=100, K=100, T=1, r=0.05, sigma=0.2)
    assert round(vega, 4) == 37.5240


def test_simulate_gbm_shape():
    paths = simulate_gbm(
        S0=100,
        mu=0.05,
        sigma=0.2,
        T=1,
        n_steps=252,
        n_paths=1000,
        seed=42
    )
    assert paths.shape == (1000, 253)


def test_monte_carlo_price_call_positive():
    price = monte_carlo_price(
        S0=100,
        K=100,
        T=1,
        r=0.05,
        sigma=0.2,
        option_type="call",
        n_steps=252,
        n_paths=10000,
        seed=42
    )
    assert price > 0


def test_hedge_shares_needed():
    shares = hedge_shares_needed(
        option_delta=0.6,
        number_of_options=10,
        contract_size=100
    )
    assert shares == 600