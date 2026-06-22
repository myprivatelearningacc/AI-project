import numpy as np


def simulate_gbm_paths(
    S0: float,
    mu: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_paths: int,
    seed: int | None = None,
) -> np.ndarray:
    """
    Simulate stock price paths using Geometric Brownian Motion.

    Parameters
    ----------
    S0 : float
        Initial stock price.
    mu : float
        Expected return / drift.
    sigma : float
        Volatility.
    T : float
        Time horizon in years.
    n_steps : int
        Number of rebalancing steps.
    n_paths : int
        Number of simulated paths.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    paths : np.ndarray
        Shape: (n_paths, n_steps + 1)
    """
    if S0 <= 0:
        raise ValueError("S0 must be greater than 0.")
    if sigma <= 0:
        raise ValueError("sigma must be greater than 0.")
    if T <= 0:
        raise ValueError("T must be greater than 0.")
    if n_steps <= 0:
        raise ValueError("n_steps must be greater than 0.")
    if n_paths <= 0:
        raise ValueError("n_paths must be greater than 0.")

    rng = np.random.default_rng(seed)

    dt = T / n_steps
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = S0

    random_shocks = rng.normal(0, 1, size=(n_paths, n_steps))

    for t in range(1, n_steps + 1):
        paths[:, t] = paths[:, t - 1] * np.exp(
            (mu - 0.5 * sigma ** 2) * dt
            + sigma * np.sqrt(dt) * random_shocks[:, t - 1]
        )

    return paths