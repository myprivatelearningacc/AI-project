import numpy as np

from src.tools.greeks import black_scholes_price
from src.tools.gbm_simulator import simulate_gbm_paths


class OptionHedgingEnv:
    """
    A simple Gym-style environment for option hedging.

    The agent is assumed to be short one European option.
    At each time step, the agent chooses a target hedge position.

    State:
        (moneyness_bin, time_step, position_bin)

    Action:
        target hedge position from a discrete action list, e.g.
        [-1.0, -0.5, 0.0, 0.5, 1.0]

    Reward:
        intermediate reward = - transaction_cost
        terminal reward = - transaction_cost - hedging_error^2

    step(action) returns:
        next_state, reward, done, info
    """

    def __init__(
        self,
        S0: float = 100.0,
        K: float = 100.0,
        T: float = 1.0,
        r: float = 0.05,
        sigma: float = 0.20,
        mu: float | None = None,
        n_steps: int = 50,
        transaction_cost_rate: float = 0.001,
        option_type: str = "call",
        actions: list[float] | None = None,
        moneyness_bins: list[float] | None = None,
        seed: int | None = None,
    ):
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.mu = r if mu is None else mu
        self.n_steps = n_steps
        self.dt = T / n_steps
        self.transaction_cost_rate = transaction_cost_rate
        self.option_type = option_type.lower()

        if actions is None:
            actions = [-1.0, -0.5, 0.0, 0.5, 1.0]
        self.actions = np.array(actions, dtype=float)

        if moneyness_bins is None:
            moneyness_bins = [0.7, 0.85, 0.95, 1.05, 1.15, 1.3]
        self.moneyness_bins = np.array(moneyness_bins, dtype=float)

        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.current_step = None
        self.stock_path = None
        self.cash = None
        self.hedge_position = None
        self.initial_option_price = None
        self.total_transaction_cost = None
        self.done = None
        self.history = None

        self._validate_inputs()

    def _validate_inputs(self) -> None:
        if self.S0 <= 0:
            raise ValueError("S0 must be greater than 0.")
        if self.K <= 0:
            raise ValueError("K must be greater than 0.")
        if self.T <= 0:
            raise ValueError("T must be greater than 0.")
        if self.sigma <= 0:
            raise ValueError("sigma must be greater than 0.")
        if self.n_steps <= 0:
            raise ValueError("n_steps must be greater than 0.")
        if self.transaction_cost_rate < 0:
            raise ValueError("transaction_cost_rate cannot be negative.")
        if self.option_type not in ["call", "put"]:
            raise ValueError("option_type must be either 'call' or 'put'.")
        if len(self.actions) == 0:
            raise ValueError("actions cannot be empty.")

    def reset(self, seed: int | None = None):
        """
        Start a new episode.

        Returns
        -------
        state : tuple
            (moneyness_bin, time_step, position_bin)
        """
        if seed is not None:
            self.seed = seed
            self.rng = np.random.default_rng(seed)

        path_seed = int(self.rng.integers(0, 1_000_000_000))

        self.stock_path = simulate_gbm_paths(
            S0=self.S0,
            mu=self.mu,
            sigma=self.sigma,
            T=self.T,
            n_steps=self.n_steps,
            n_paths=1,
            seed=path_seed,
        )[0]

        self.initial_option_price = black_scholes_price(
            S=self.S0,
            K=self.K,
            T=self.T,
            r=self.r,
            sigma=self.sigma,
            option_type=self.option_type,
        )

        # Short one option, receive option premium.
        # Initially hold zero shares.
        self.cash = self.initial_option_price
        self.hedge_position = 0.0
        self.current_step = 0
        self.total_transaction_cost = 0.0
        self.done = False

        self.history = []

        state = self._get_state()
        self._record_history(
            reward=0.0,
            transaction_cost=0.0,
            portfolio_value=self._current_portfolio_value(),
            option_payoff=0.0,
            hedging_error=0.0,
            action=None,
        )

        return state

    def step(self, action):
        """
        Execute one hedging step.

        Parameters
        ----------
        action : int or float
            If int, treated as action index.
            If float, treated as target hedge position.

        Returns
        -------
        next_state : tuple
        reward : float
        done : bool
        info : dict
        """
        if self.done:
            raise RuntimeError("Episode is already done. Call reset() before stepping again.")

        target_position = self._parse_action(action)
        current_stock_price = self.stock_path[self.current_step]

        # Rebalance hedge position at current stock price.
        position_change = target_position - self.hedge_position
        trade_value = position_change * current_stock_price
        transaction_cost = self.transaction_cost_rate * abs(position_change) * current_stock_price

        self.cash -= trade_value
        self.cash -= transaction_cost
        self.hedge_position = target_position
        self.total_transaction_cost += transaction_cost

        reward = -transaction_cost

        # Move to next time step.
        self.current_step += 1

        # Cash earns risk-free interest during the interval.
        self.cash *= np.exp(self.r * self.dt)

        done = self.current_step == self.n_steps

        option_payoff = 0.0
        hedging_error = 0.0
        liquidation_cost = 0.0

        if done:
            terminal_stock_price = self.stock_path[self.current_step]

            # Liquidate stock position at maturity.
            liquidation_value = self.hedge_position * terminal_stock_price
            liquidation_cost = (
                self.transaction_cost_rate
                * abs(self.hedge_position)
                * terminal_stock_price
            )

            self.cash += liquidation_value
            self.cash -= liquidation_cost
            self.total_transaction_cost += liquidation_cost

            option_payoff = self._option_payoff(terminal_stock_price)

            # Since we are short one option:
            # hedging_error = final hedging portfolio value - option payoff
            portfolio_value = self.cash
            hedging_error = portfolio_value - option_payoff

            reward -= liquidation_cost
            reward -= hedging_error ** 2

            self.hedge_position = 0.0
            self.done = True
        else:
            portfolio_value = self._current_portfolio_value()

        next_state = self._get_state()

        info = {
            "stock_price": float(self.stock_path[self.current_step]),
            "hedge_position": float(self.hedge_position),
            "cash": float(self.cash),
            "portfolio_value": float(portfolio_value),
            "option_payoff": float(option_payoff),
            "hedging_error": float(hedging_error),
            "transaction_cost": float(transaction_cost + liquidation_cost),
            "total_transaction_cost": float(self.total_transaction_cost),
            "time_step": int(self.current_step),
            "done": bool(done),
        }

        self._record_history(
            reward=reward,
            transaction_cost=transaction_cost + liquidation_cost,
            portfolio_value=portfolio_value,
            option_payoff=option_payoff,
            hedging_error=hedging_error,
            action=target_position,
        )

        return next_state, float(reward), bool(done), info

    def _get_state(self):
        """
        Convert continuous variables into discrete tabular state.

        State:
            (moneyness_bin, time_step, position_bin)
        """
        stock_price = self.stock_path[self.current_step]
        moneyness = stock_price / self.K

        moneyness_bin = int(np.digitize(moneyness, self.moneyness_bins))
        time_bin = int(self.current_step)
        position_bin = self._position_to_bin(self.hedge_position)

        return (moneyness_bin, time_bin, position_bin)

    def _position_to_bin(self, position: float) -> int:
        """
        Map current hedge position to nearest action index.
        """
        distances = np.abs(self.actions - position)
        return int(np.argmin(distances))

    def _parse_action(self, action) -> float:
        """
        Accept either action index or direct target hedge position.

        Example:
            action = 3 means self.actions[3]
            action = 0.5 means target hedge position 0.5
        """
        if isinstance(action, (int, np.integer)):
            if action < 0 or action >= len(self.actions):
                raise ValueError(f"Action index must be between 0 and {len(self.actions) - 1}.")
            return float(self.actions[action])

        action = float(action)

        if action not in self.actions:
            nearest_action = self.actions[np.argmin(np.abs(self.actions - action))]
            return float(nearest_action)

        return action

    def _option_payoff(self, stock_price: float) -> float:
        if self.option_type == "call":
            return float(max(stock_price - self.K, 0.0))
        elif self.option_type == "put":
            return float(max(self.K - stock_price, 0.0))
        else:
            raise ValueError("option_type must be either 'call' or 'put'.")

    def _current_portfolio_value(self) -> float:
        stock_price = self.stock_path[self.current_step]
        return float(self.cash + self.hedge_position * stock_price)

    def _record_history(
        self,
        reward: float,
        transaction_cost: float,
        portfolio_value: float,
        option_payoff: float,
        hedging_error: float,
        action,
    ) -> None:
        stock_price = self.stock_path[self.current_step]

        record = {
            "time_step": int(self.current_step),
            "stock_price": float(stock_price),
            "moneyness": float(stock_price / self.K),
            "hedge_position": float(self.hedge_position),
            "cash": float(self.cash),
            "portfolio_value": float(portfolio_value),
            "option_payoff": float(option_payoff),
            "hedging_error": float(hedging_error),
            "transaction_cost": float(transaction_cost),
            "total_transaction_cost": float(self.total_transaction_cost),
            "reward": float(reward),
            "action": None if action is None else float(action),
            "state": self._get_state(),
        }

        self.history.append(record)

    def sample_random_action(self) -> int:
        """
        Return a random action index.
        Useful for testing before implementing Q-learning.
        """
        return int(self.rng.integers(0, len(self.actions)))

    def get_history(self):
        """
        Return episode history as a list of dictionaries.
        """
        return self.history

    def get_action_space(self):
        """
        Return available hedge positions.
        """
        return self.actions.copy()

    def get_state_space_info(self):
        """
        Return useful information about discrete state design.
        """
        return {
            "moneyness_bins": self.moneyness_bins.copy(),
            "n_moneyness_states": len(self.moneyness_bins) + 1,
            "n_time_states": self.n_steps + 1,
            "actions": self.actions.copy(),
            "n_position_states": len(self.actions),
        }