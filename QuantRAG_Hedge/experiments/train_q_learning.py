"""
Train a tabular Q-learning hedging agent under a Black-Scholes market simulation.

Run from project root:
    python experiments/train_q_learning.py

Example variations:
    python experiments/train_q_learning.py --reward_type final_error
    python experiments/train_q_learning.py --reward_type final_error_cost
    python experiments/train_q_learning.py --state_type with_position --cost_rate 0.005
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Allow running this script directly from the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.q_learning_agent import QLearningAgent


@dataclass
class HedgingConfig:
    # Market / option parameters
    S0: float = 100.0
    K: float = 100.0
    r: float = 0.01
    sigma: float = 0.20
    T: float = 1.0
    n_steps: int = 30
    option_type: str = "call"

    # Discretisation
    n_moneyness_bins: int = 10
    n_time_bins: int = 10
    n_position_bins: int = 9
    state_type: str = "basic"  # basic or with_position

    # Action grid: target hedge ratio / shares
    n_actions: int = 9
    min_position: float = 0.0
    max_position: float = 1.0

    # Reward / cost settings
    reward_type: str = "final_error"  # final_error, final_error_cost, final_error_lambda_cost
    cost_rate: float = 0.001
    cost_lambda: float = 1.0
    reward_scale: float = 100.0

    # Q-learning parameters
    alpha: float = 0.1
    gamma: float = 0.99
    epsilon: float = 1.0
    epsilon_decay: float = 0.995
    epsilon_min: float = 0.05
    episodes: int = 5000
    seed: int = 42


class BlackScholesHedgingEnv:
    """
    Discrete-action hedging environment for a short European option.

    The agent receives the option premium at t=0 and chooses a target hedge
    position in the underlying. At maturity, the final hedging error is:
        cash + shares * S_T - option_payoff

    The learning objective is to make this error close to zero while controlling
    transaction cost.
    """

    def __init__(self, config: HedgingConfig):
        self.config = config
        self.dt = config.T / config.n_steps
        self.actions = np.linspace(config.min_position, config.max_position, config.n_actions)
        self.rng = np.random.default_rng(config.seed)
        self.reset()

    def reset(self) -> Tuple[int, ...]:
        self.t = 0
        self.path = self._simulate_gbm_path()
        self.position = 0.0
        self.cash = self._bs_price(self.config.S0, self.config.T)
        self.total_transaction_cost = 0.0
        self.done = False
        return self._state()

    def step(self, action_idx: int) -> Tuple[Optional[Tuple[int, ...]], float, bool, Dict[str, float]]:
        if self.done:
            raise RuntimeError("Episode is done. Call reset() before step().")
        if action_idx < 0 or action_idx >= len(self.actions):
            raise ValueError("Invalid action index.")

        S_t = float(self.path[self.t])
        new_position = float(self.actions[action_idx])
        trade_size = new_position - self.position
        transaction_cost = self.config.cost_rate * abs(trade_size) * S_t

        # Rebalance the hedge at current price.
        self.cash -= trade_size * S_t
        self.cash -= transaction_cost
        self.total_transaction_cost += transaction_cost
        self.position = new_position

        # Move to next time step and accrue interest on cash.
        self.cash *= math.exp(self.config.r * self.dt)
        self.t += 1

        if self.t >= self.config.n_steps:
            self.done = True
            S_T = float(self.path[-1])
            payoff = self._option_payoff(S_T)
            final_portfolio = self.cash + self.position * S_T - payoff
            final_error = final_portfolio
            reward = self._terminal_reward(final_error, self.total_transaction_cost)
            info = {
                "S_T": S_T,
                "payoff": payoff,
                "final_error": final_error,
                "abs_final_error": abs(final_error),
                "squared_final_error": final_error**2,
                "transaction_cost": self.total_transaction_cost,
                "reward": reward,
            }
            return None, reward, True, info

        return self._state(), 0.0, False, {
            "transaction_cost": transaction_cost,
            "total_transaction_cost": self.total_transaction_cost,
        }

    def _simulate_gbm_path(self) -> np.ndarray:
        cfg = self.config
        z = self.rng.normal(size=cfg.n_steps)
        increments = (cfg.r - 0.5 * cfg.sigma**2) * self.dt + cfg.sigma * math.sqrt(self.dt) * z
        log_path = np.log(cfg.S0) + np.cumsum(np.insert(increments, 0, 0.0))
        return np.exp(log_path)

    def _state(self) -> Tuple[int, ...]:
        S_t = float(self.path[self.t])
        moneyness = S_t / self.config.K
        m_bin = discretize(moneyness, low=0.70, high=1.30, n_bins=self.config.n_moneyness_bins)
        tau = (self.config.n_steps - self.t) / self.config.n_steps
        time_bin = discretize(tau, low=0.0, high=1.0, n_bins=self.config.n_time_bins)

        if self.config.state_type == "basic":
            return (m_bin, time_bin)
        if self.config.state_type == "with_position":
            p_bin = discretize(
                self.position,
                low=self.config.min_position,
                high=self.config.max_position,
                n_bins=self.config.n_position_bins,
            )
            return (m_bin, time_bin, p_bin)
        raise ValueError("state_type must be 'basic' or 'with_position'.")

    def _terminal_reward(self, final_error: float, transaction_cost: float) -> float:
        cfg = self.config
        error_penalty = final_error**2

        if cfg.reward_type == "final_error":
            reward = -error_penalty
        elif cfg.reward_type == "final_error_cost":
            reward = -error_penalty - transaction_cost
        elif cfg.reward_type == "final_error_lambda_cost":
            reward = -error_penalty - cfg.cost_lambda * transaction_cost
        else:
            raise ValueError(
                "reward_type must be 'final_error', 'final_error_cost', or 'final_error_lambda_cost'."
            )

        return reward / cfg.reward_scale

    def _option_payoff(self, S: float) -> float:
        if self.config.option_type == "call":
            return max(S - self.config.K, 0.0)
        if self.config.option_type == "put":
            return max(self.config.K - S, 0.0)
        raise ValueError("option_type must be 'call' or 'put'.")

    def _bs_price(self, S: float, tau: float) -> float:
        cfg = self.config
        if tau <= 0:
            return self._option_payoff(S)
        d1 = (math.log(S / cfg.K) + (cfg.r + 0.5 * cfg.sigma**2) * tau) / (cfg.sigma * math.sqrt(tau))
        d2 = d1 - cfg.sigma * math.sqrt(tau)
        if cfg.option_type == "call":
            return S * normal_cdf(d1) - cfg.K * math.exp(-cfg.r * tau) * normal_cdf(d2)
        if cfg.option_type == "put":
            return cfg.K * math.exp(-cfg.r * tau) * normal_cdf(-d2) - S * normal_cdf(-d1)
        raise ValueError("option_type must be 'call' or 'put'.")


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def discretize(value: float, low: float, high: float, n_bins: int) -> int:
    if n_bins <= 1:
        return 0
    clipped = min(max(value, low), high)
    edges = np.linspace(low, high, n_bins + 1)
    return int(np.clip(np.digitize(clipped, edges[1:-1], right=False), 0, n_bins - 1))


def train_agent(config: HedgingConfig, output_dir: Path, run_name: str = "q_learning") -> Tuple[QLearningAgent, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    env = BlackScholesHedgingEnv(config)
    agent = QLearningAgent(
        n_actions=config.n_actions,
        alpha=config.alpha,
        gamma=config.gamma,
        epsilon=config.epsilon,
        epsilon_decay=config.epsilon_decay,
        epsilon_min=config.epsilon_min,
        seed=config.seed,
    )

    records: List[Dict[str, float]] = []

    for episode in range(1, config.episodes + 1):
        state = env.reset()
        done = False
        episode_reward = 0.0
        steps = 0
        final_info: Dict[str, float] = {}

        while not done:
            action = agent.choose_action(state, training=True)
            next_state, reward, done, info = env.step(action)
            agent.update(state, action, reward, next_state, done)
            episode_reward += reward
            steps += 1
            if done:
                final_info = info
            else:
                state = next_state  # type: ignore[assignment]

        agent.decay_epsilon()

        records.append(
            {
                "episode": episode,
                "reward": episode_reward,
                "epsilon": agent.epsilon,
                "steps": steps,
                "final_error": final_info.get("final_error", np.nan),
                "abs_final_error": final_info.get("abs_final_error", np.nan),
                "squared_final_error": final_info.get("squared_final_error", np.nan),
                "transaction_cost": final_info.get("transaction_cost", np.nan),
                "S_T": final_info.get("S_T", np.nan),
            }
        )

    history = pd.DataFrame(records)
    history["rolling_reward_100"] = history["reward"].rolling(100, min_periods=1).mean()
    history["rolling_abs_error_100"] = history["abs_final_error"].rolling(100, min_periods=1).mean()
    history["rolling_cost_100"] = history["transaction_cost"].rolling(100, min_periods=1).mean()

    history_path = output_dir / f"{run_name}_history.csv"
    model_path = output_dir / f"{run_name}_q_table.json"
    config_path = output_dir / f"{run_name}_config.json"
    plot_path = output_dir / f"{run_name}_learning_curve.png"

    history.to_csv(history_path, index=False)
    agent.save(model_path)
    pd.Series(asdict(config)).to_json(config_path, indent=2)
    plot_learning_curve(history, plot_path, title=f"Q-learning hedging: {run_name}")

    return agent, history


def evaluate_agent(config: HedgingConfig, agent: QLearningAgent, n_eval_episodes: int = 1000, seed_offset: int = 10_000) -> Dict[str, float]:
    eval_config = HedgingConfig(**asdict(config))
    eval_config.seed = config.seed + seed_offset
    env = BlackScholesHedgingEnv(eval_config)

    rows = []
    old_epsilon = agent.epsilon
    agent.set_epsilon(0.0)

    for _ in range(n_eval_episodes):
        state = env.reset()
        done = False
        final_info = {}
        while not done:
            action = agent.choose_action(state, training=False)
            next_state, _, done, info = env.step(action)
            if done:
                final_info = info
            else:
                state = next_state  # type: ignore[assignment]
        rows.append(final_info)

    agent.set_epsilon(old_epsilon)
    df = pd.DataFrame(rows)
    return {
        "eval_mean_abs_error": float(df["abs_final_error"].mean()),
        "eval_rmse_error": float(np.sqrt(df["squared_final_error"].mean())),
        "eval_mean_transaction_cost": float(df["transaction_cost"].mean()),
        "eval_mean_reward": float(df["reward"].mean()),
        "eval_error_std": float(df["final_error"].std()),
    }


def plot_learning_curve(history: pd.DataFrame, output_path: Path, title: str) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(history["episode"], history["rolling_reward_100"])
    plt.xlabel("Episode")
    plt.ylabel("Rolling mean reward, window=100")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()

    error_path = output_path.with_name(output_path.stem + "_abs_error.png")
    plt.figure(figsize=(10, 5))
    plt.plot(history["episode"], history["rolling_abs_error_100"])
    plt.xlabel("Episode")
    plt.ylabel("Rolling mean absolute hedging error, window=100")
    plt.title(title.replace("reward", "absolute error"))
    plt.tight_layout()
    plt.savefig(error_path, dpi=160)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Q-learning hedging agent.")
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--epsilon_decay", type=float, default=0.995)
    parser.add_argument("--epsilon_min", type=float, default=0.05)
    parser.add_argument("--reward_type", type=str, default="final_error", choices=["final_error", "final_error_cost", "final_error_lambda_cost"])
    parser.add_argument("--cost_rate", type=float, default=0.001)
    parser.add_argument("--cost_lambda", type=float, default=1.0)
    parser.add_argument("--state_type", type=str, default="basic", choices=["basic", "with_position"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_name", type=str, default="q_learning")
    parser.add_argument("--output_dir", type=str, default="outputs/milestone8_q_learning")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = HedgingConfig(
        episodes=args.episodes,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon=args.epsilon,
        epsilon_decay=args.epsilon_decay,
        epsilon_min=args.epsilon_min,
        reward_type=args.reward_type,
        cost_rate=args.cost_rate,
        cost_lambda=args.cost_lambda,
        state_type=args.state_type,
        seed=args.seed,
    )

    output_dir = Path(args.output_dir)
    agent, history = train_agent(config, output_dir=output_dir, run_name=args.run_name)
    eval_metrics = evaluate_agent(config, agent, n_eval_episodes=1000)

    summary = {
        "run_name": args.run_name,
        **asdict(config),
        "train_final_100_reward": float(history["reward"].tail(100).mean()),
        "train_final_100_abs_error": float(history["abs_final_error"].tail(100).mean()),
        "train_final_100_transaction_cost": float(history["transaction_cost"].tail(100).mean()),
        **eval_metrics,
        "visited_states": len(agent.q_table),
    }
    summary_path = output_dir / f"{args.run_name}_summary.csv"
    pd.DataFrame([summary]).to_csv(summary_path, index=False)

    print("Training completed.")
    print(pd.Series(summary).to_string())
    print(f"\nSaved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
