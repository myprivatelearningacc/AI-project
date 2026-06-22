import numpy as np
from scipy.stats import norm


def validate_bs_inputs(S: float, K: float, T: float, sigma: float) -> None:
    """
    Validate core Black-Scholes inputs.
    """
    if S <= 0:
        raise ValueError("Stock price S must be greater than 0.")
    if K <= 0:
        raise ValueError("Strike price K must be greater than 0.")
    if T <= 0:
        raise ValueError("Time to maturity T must be greater than 0.")
    if sigma <= 0:
        raise ValueError("Volatility sigma must be greater than 0.")


def calculate_d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
    validate_bs_inputs(S, K, T, sigma)
    return (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))


def calculate_d2(S: float, K: float, T: float, r: float, sigma: float) -> float:
    d1 = calculate_d1(S, K, T, r, sigma)
    return d1 - sigma * np.sqrt(T)


def black_scholes_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
) -> float:
    """
    Calculate Black-Scholes option price for European call or put.
    """
    validate_bs_inputs(S, K, T, sigma)

    option_type = option_type.lower()
    d1 = calculate_d1(S, K, T, r, sigma)
    d2 = calculate_d2(S, K, T, r, sigma)

    if option_type == "call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be either 'call' or 'put'.")

    return float(price)


def black_scholes_delta(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
) -> float:
    """
    Delta measures sensitivity of option price to stock price.
    """
    validate_bs_inputs(S, K, T, sigma)

    option_type = option_type.lower()
    d1 = calculate_d1(S, K, T, r, sigma)

    if option_type == "call":
        delta = norm.cdf(d1)
    elif option_type == "put":
        delta = norm.cdf(d1) - 1
    else:
        raise ValueError("option_type must be either 'call' or 'put'.")

    return float(delta)


def black_scholes_gamma(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
) -> float:
    """
    Gamma measures sensitivity of delta to stock price.
    Same for call and put in Black-Scholes.
    """
    validate_bs_inputs(S, K, T, sigma)

    d1 = calculate_d1(S, K, T, r, sigma)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))

    return float(gamma)


def black_scholes_vega(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
) -> float:
    """
    Vega measures sensitivity of option price to volatility.
    Returned as price change for 1.00 change in volatility.
    Divide by 100 if interpreting as change per 1% volatility.
    """
    validate_bs_inputs(S, K, T, sigma)

    d1 = calculate_d1(S, K, T, r, sigma)
    vega = S * norm.pdf(d1) * np.sqrt(T)

    return float(vega)


def black_scholes_theta(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
) -> float:
    """
    Theta measures sensitivity of option price to time decay.
    Returned as annual theta.
    """
    validate_bs_inputs(S, K, T, sigma)

    option_type = option_type.lower()
    d1 = calculate_d1(S, K, T, r, sigma)
    d2 = calculate_d2(S, K, T, r, sigma)

    first_term = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))

    if option_type == "call":
        theta = first_term - r * K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        theta = first_term + r * K * np.exp(-r * T) * norm.cdf(-d2)
    else:
        raise ValueError("option_type must be either 'call' or 'put'.")

    return float(theta)


def calculate_all_greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
) -> dict:
    """
    Return price and main Greeks in one dictionary.
    """
    return {
        "price": black_scholes_price(S, K, T, r, sigma, option_type),
        "delta": black_scholes_delta(S, K, T, r, sigma, option_type),
        "gamma": black_scholes_gamma(S, K, T, r, sigma),
        "vega": black_scholes_vega(S, K, T, r, sigma),
        "theta": black_scholes_theta(S, K, T, r, sigma, option_type),
    }