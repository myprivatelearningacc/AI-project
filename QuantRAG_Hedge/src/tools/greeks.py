import math
from src.tools.black_scholes import calculate_d1, norm_cdf


def norm_pdf(x: float) -> float:
    """
    Standard normal probability density function.
    """
    return (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * x ** 2)


def calculate_delta(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call"
) -> float:
    """
    Calculate Black-Scholes delta.

    For call:
        Delta = N(d1)

    For put:
        Delta = N(d1) - 1
    """
    d1 = calculate_d1(S, K, T, r, sigma)

    if option_type.lower() == "call":
        return norm_cdf(d1)
    elif option_type.lower() == "put":
        return norm_cdf(d1) - 1.0
    else:
        raise ValueError("option_type must be either 'call' or 'put'.")


def calculate_gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Calculate Black-Scholes gamma.

    Gamma is the same for European calls and puts.
    """
    d1 = calculate_d1(S, K, T, r, sigma)
    return norm_pdf(d1) / (S * sigma * math.sqrt(T))


def calculate_vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Calculate Black-Scholes vega.

    This returns vega for a 1.00 change in volatility.
    For example, if this returns 37.52, then a 0.01 increase in volatility
    changes the option price by roughly 0.3752.
    """
    d1 = calculate_d1(S, K, T, r, sigma)
    return S * norm_pdf(d1) * math.sqrt(T)