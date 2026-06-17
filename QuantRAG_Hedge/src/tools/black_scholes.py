import math


def norm_cdf(x: float) -> float:
    """
    Standard normal cumulative distribution function.
    Uses math.erf so we do not need scipy.
    """
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def calculate_d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Calculate d1 in the Black-Scholes formula.

    Parameters:
    S: current stock price
    K: strike price
    T: time to maturity in years
    r: continuously compounded risk-free rate
    sigma: annualized volatility
    """
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        raise ValueError("S, K, T, and sigma must be positive.")

    return (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))


def calculate_d2(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Calculate d2 in the Black-Scholes formula.
    """
    d1 = calculate_d1(S, K, T, r, sigma)
    return d1 - sigma * math.sqrt(T)


def black_scholes_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Price a European call option using the Black-Scholes formula.
    """
    d1 = calculate_d1(S, K, T, r, sigma)
    d2 = calculate_d2(S, K, T, r, sigma)

    call_price = S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
    return call_price


def black_scholes_put(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Price a European put option using the Black-Scholes formula.
    """
    d1 = calculate_d1(S, K, T, r, sigma)
    d2 = calculate_d2(S, K, T, r, sigma)

    put_price = K * math.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)
    return put_price