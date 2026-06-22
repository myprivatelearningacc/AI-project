import numpy as np
import pandas as pd

from src.tools.greeks import (
    black_scholes_price,
    black_scholes_delta,
)
from src.tools.gbm_simulator import simulate_gbm_paths


def option_payoff(S_T: np.ndarray, K: float, option_type: str = "call") -> np.ndarray:
    """
    Calculate European option payoff at maturity.
    """
    option_type = option_type.lower()

    if option_type == "call":
        return np.maximum(S_T - K, 0)
    elif option_type == "put":
        return np.maximum(K - S_T, 0)
    else:
        raise ValueError("option_type must be either 'call' or 'put'.")


def simulate_delta_hedge(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    n_steps: int,
    n_paths: int,
    transaction_cost_rate: float = 0.0,
    option_type: str = "call",
    mu: float | None = None,
    seed: int | None = None,
) -> dict:
    """
    Simulate Black-Scholes delta hedging strategy.

    The simulation assumes we are short one European option and use delta hedging
    to reduce payoff risk.

    Parameters
    ----------
    S0, K, T, r, sigma : float
        Black-Scholes parameters.
    n_steps : int
        Number of rebalancing steps.
    n_paths : int
        Number of simulated stock paths.
    transaction_cost_rate : float
        Proportional transaction cost per stock traded.
        Example: 0.001 means 0.1%.
    option_type : str
        'call' or 'put'.
    mu : float or None
        Drift used for GBM stock simulation.
        If None, risk-free rate r is used.
    seed : int or None
        Random seed.

    Returns
    -------
    dict
        Contains paths, no hedge errors, delta hedge errors, transaction costs,
        summary metrics, and path-level dataframe.
    """
    if mu is None:
        mu = r

    if transaction_cost_rate < 0:
        raise ValueError("transaction_cost_rate cannot be negative.")

    dt = T / n_steps

    stock_paths = simulate_gbm_paths(
        S0=S0,
        mu=mu,
        sigma=sigma,
        T=T,
        n_steps=n_steps,
        n_paths=n_paths,
        seed=seed,
    )

    initial_option_price = black_scholes_price(
        S=S0,
        K=K,
        T=T,
        r=r,
        sigma=sigma,
        option_type=option_type,
    )

    # No hedge baseline:
    # We sell the option and keep premium in cash until maturity.
    final_cash_no_hedge = initial_option_price * np.exp(r * T)
    payoff = option_payoff(stock_paths[:, -1], K, option_type)
    no_hedge_error = final_cash_no_hedge - payoff

    delta_hedge_error = np.zeros(n_paths)
    total_transaction_cost = np.zeros(n_paths)
    final_portfolio_value = np.zeros(n_paths)

    for i in range(n_paths):
        path = stock_paths[i]

        # Initial delta
        remaining_T = T
        delta = black_scholes_delta(
            S=path[0],
            K=K,
            T=remaining_T,
            r=r,
            sigma=sigma,
            option_type=option_type,
        )

        # We sell option, receive premium, then buy delta shares.
        # cash = premium - cost of shares - transaction cost
        initial_trade_value = delta * path[0]
        initial_tc = transaction_cost_rate * abs(delta) * path[0]

        cash = initial_option_price - initial_trade_value - initial_tc
        total_tc = initial_tc

        current_delta = delta

        # Rebalance through time, excluding final maturity point
        for t in range(1, n_steps):
            # Cash grows at risk-free rate
            cash *= np.exp(r * dt)

            remaining_T = T - t * dt

            new_delta = black_scholes_delta(
                S=path[t],
                K=K,
                T=remaining_T,
                r=r,
                sigma=sigma,
                option_type=option_type,
            )

            delta_change = new_delta - current_delta
            trade_value = delta_change * path[t]
            tc = transaction_cost_rate * abs(delta_change) * path[t]

            # If delta_change > 0, buy more stock, cash decreases.
            # If delta_change < 0, sell stock, cash increases.
            cash -= trade_value
            cash -= tc

            total_tc += tc
            current_delta = new_delta

        # Final cash accrual to maturity
        cash *= np.exp(r * dt)

        # Liquidate stock position at maturity
        stock_position_value = current_delta * path[-1]
        liquidation_tc = transaction_cost_rate * abs(current_delta) * path[-1]

        final_value = cash + stock_position_value - liquidation_tc
        total_tc += liquidation_tc

        final_portfolio_value[i] = final_value
        total_transaction_cost[i] = total_tc
        delta_hedge_error[i] = final_value - payoff[i]

    path_results = pd.DataFrame(
        {
            "terminal_stock_price": stock_paths[:, -1],
            "payoff": payoff,
            "no_hedge_error": no_hedge_error,
            "delta_hedge_error": delta_hedge_error,
            "total_transaction_cost": total_transaction_cost,
            "final_portfolio_value": final_portfolio_value,
        }
    )

    summary = {
        "initial_option_price": float(initial_option_price),

        "no_hedge_mean_error": float(np.mean(no_hedge_error)),
        "no_hedge_std_error": float(np.std(no_hedge_error)),
        "no_hedge_mae": float(np.mean(np.abs(no_hedge_error))),

        "delta_hedge_mean_error": float(np.mean(delta_hedge_error)),
        "delta_hedge_std_error": float(np.std(delta_hedge_error)),
        "delta_hedge_mae": float(np.mean(np.abs(delta_hedge_error))),

        "mean_transaction_cost": float(np.mean(total_transaction_cost)),
        "total_transaction_cost": float(np.sum(total_transaction_cost)),
        "number_of_rebalances": int(n_steps - 1),
    }

    return {
        "stock_paths": stock_paths,
        "payoff": payoff,
        "no_hedge_error": no_hedge_error,
        "delta_hedge_error": delta_hedge_error,
        "total_transaction_cost": total_transaction_cost,
        "path_results": path_results,
        "summary": summary,
    }