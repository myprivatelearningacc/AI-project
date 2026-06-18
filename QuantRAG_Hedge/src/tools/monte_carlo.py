import numpy as np


def simulate_gbm(
    S0: float,
    mu: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_paths: int,
    seed: int | None = None
) -> np.ndarray:
    """
    Simulate stock price paths under Geometric Brownian Motion.

    dS_t = mu * S_t * dt + sigma * S_t * dW_t

    Returns:
        paths: numpy array with shape (n_paths, n_steps + 1)
    """
    if S0 <= 0 or sigma < 0 or T <= 0:
        raise ValueError("S0 and T must be positive, sigma must be non-negative.")
    if n_steps <= 0 or n_paths <= 0:
        raise ValueError("n_steps and n_paths must be positive integers.")

    rng = np.random.default_rng(seed)

    dt = T / n_steps
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = S0

    for t in range(1, n_steps + 1):
        z = rng.standard_normal(n_paths)

        paths[:, t] = paths[:, t - 1] * np.exp(
            (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z
        )

    return paths


def monte_carlo_price(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    n_steps: int = 252,
    n_paths: int = 10000,
    seed: int | None = 42
) -> float:
    """
    Price a European option using Monte Carlo simulation.

    Under risk-neutral pricing, we simulate GBM using mu = r.
    """
    paths = simulate_gbm(
        S0=S0,
        mu=r,
        sigma=sigma,
        T=T,
        n_steps=n_steps,
        n_paths=n_paths,
        seed=seed
    )

    final_prices = paths[:, -1]

    if option_type.lower() == "call":
        payoffs = np.maximum(final_prices - K, 0)
    elif option_type.lower() == "put":
        payoffs = np.maximum(K - final_prices, 0)
    else:
        raise ValueError("option_type must be either 'call' or 'put'.")

    discounted_price = np.exp(-r * T) * np.mean(payoffs)
    return float(discounted_price)