import numpy as np
from scipy.stats import norm


class BlackScholesModel:
    """
    Black-Scholes option pricing model for European call and put options.

    Parameters
    ----------
    S : float
        Current underlying asset price.
    K : float
        Strike price.
    T : float
        Time to maturity in years.
    r : float
        Risk-free interest rate.
    sigma : float
        Volatility of the underlying asset.
    """

    def __init__(self, S: float, K: float, T: float, r: float, sigma: float):
        self.S = float(S)
        self.K = float(K)
        self.T = float(T)
        self.r = float(r)
        self.sigma = float(sigma)

        if self.S <= 0:
            raise ValueError("Underlying price S must be positive.")
        if self.K <= 0:
            raise ValueError("Strike price K must be positive.")
        if self.T <= 0:
            raise ValueError("Time to maturity T must be positive.")
        if self.sigma <= 0:
            raise ValueError("Volatility sigma must be positive.")

    def d1(self) -> float:
        return (
            np.log(self.S / self.K)
            + (self.r + 0.5 * self.sigma ** 2) * self.T
        ) / (self.sigma * np.sqrt(self.T))

    def d2(self) -> float:
        return self.d1() - self.sigma * np.sqrt(self.T)

    def call_price(self) -> float:
        d1 = self.d1()
        d2 = self.d2()

        return self.S * norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(d2)

    def put_price(self) -> float:
        d1 = self.d1()
        d2 = self.d2()

        return self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) - self.S * norm.cdf(-d1)

    def delta(self, option_type: str = "call") -> float:
        d1 = self.d1()

        if option_type == "call":
            return norm.cdf(d1)
        elif option_type == "put":
            return norm.cdf(d1) - 1
        else:
            raise ValueError("option_type must be 'call' or 'put'.")

    def gamma(self) -> float:
        d1 = self.d1()
        return norm.pdf(d1) / (self.S * self.sigma * np.sqrt(self.T))

    def vega(self) -> float:
        d1 = self.d1()
        return self.S * norm.pdf(d1) * np.sqrt(self.T) / 100

    def theta(self, option_type: str = "call") -> float:
        d1 = self.d1()
        d2 = self.d2()

        first_term = -(
            self.S * norm.pdf(d1) * self.sigma
        ) / (2 * np.sqrt(self.T))

        if option_type == "call":
            second_term = self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
            return (first_term - second_term) / 365
        elif option_type == "put":
            second_term = self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-d2)
            return (first_term + second_term) / 365
        else:
            raise ValueError("option_type must be 'call' or 'put'.")

    def rho(self, option_type: str = "call") -> float:
        d2 = self.d2()

        if option_type == "call":
            return self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(d2) / 100
        elif option_type == "put":
            return -self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(-d2) / 100
        else:
            raise ValueError("option_type must be 'call' or 'put'.")

    def summary(self) -> dict:
        return {
            "call_price": self.call_price(),
            "put_price": self.put_price(),
            "call_delta": self.delta("call"),
            "put_delta": self.delta("put"),
            "gamma": self.gamma(),
            "vega": self.vega(),
            "call_theta": self.theta("call"),
            "put_theta": self.theta("put"),
            "call_rho": self.rho("call"),
            "put_rho": self.rho("put"),
            "d1": self.d1(),
            "d2": self.d2(),
        }