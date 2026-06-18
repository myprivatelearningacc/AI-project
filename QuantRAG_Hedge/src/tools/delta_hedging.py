from src.tools.greeks import calculate_delta


def hedge_shares_needed(
    option_delta: float,
    number_of_options: int,
    contract_size: int = 100
) -> float:
    """
    Calculate how many shares are needed to delta hedge an option position.

    Example:
    If we are short 10 call contracts, each contract controls 100 shares,
    and call delta = 0.6, then we need to buy:

        0.6 * 10 * 100 = 600 shares

    This function returns the absolute hedge quantity.
    """
    if contract_size <= 0:
        raise ValueError("contract_size must be positive.")

    return option_delta * number_of_options * contract_size


def calculate_option_delta_for_hedge(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call"
) -> float:
    """
    Wrapper function to calculate the option delta used for hedging.
    """
    return calculate_delta(S, K, T, r, sigma, option_type)