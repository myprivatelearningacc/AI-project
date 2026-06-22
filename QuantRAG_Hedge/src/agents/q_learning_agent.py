"""
Q-learning agent for discrete hedging policies.

This agent is intentionally environment-agnostic:
- state can be any hashable object, usually a tuple of discrete bins
- action is an integer index into a discrete action grid
- Q-values are stored in a dictionary-backed Q-table

Update rule:
Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Hashable, Iterable, List, Optional, Tuple

import numpy as np


State = Hashable


class QLearningAgent:
    """Tabular Q-learning agent with epsilon-greedy exploration."""

    def __init__(
        self,
        n_actions: int,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.05,
        seed: Optional[int] = None,
    ) -> None:
        if n_actions <= 0:
            raise ValueError("n_actions must be positive.")
        if not 0 < alpha <= 1:
            raise ValueError("alpha should be in (0, 1].")
        if not 0 <= gamma <= 1:
            raise ValueError("gamma should be in [0, 1].")
        if not 0 <= epsilon <= 1:
            raise ValueError("epsilon should be in [0, 1].")
        if not 0 <= epsilon_min <= 1:
            raise ValueError("epsilon_min should be in [0, 1].")
        if not 0 < epsilon_decay <= 1:
            raise ValueError("epsilon_decay should be in (0, 1].")

        self.n_actions = int(n_actions)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.epsilon_decay = float(epsilon_decay)
        self.epsilon_min = float(epsilon_min)

        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

        # q_table[state] -> np.array of length n_actions
        self.q_table: Dict[State, np.ndarray] = defaultdict(
            lambda: np.zeros(self.n_actions, dtype=float)
        )

    def get_q_values(self, state: State) -> np.ndarray:
        """Return Q-values for a state."""
        return self.q_table[state]

    def choose_action(self, state: State, training: bool = True) -> int:
        """
        Choose an action using epsilon-greedy exploration during training.
        During evaluation, choose the greedy action.
        """
        if training and self.rng.random() < self.epsilon:
            return self.rng.randrange(self.n_actions)

        q_values = self.get_q_values(state)
        max_q = np.max(q_values)
        best_actions = np.flatnonzero(np.isclose(q_values, max_q))
        return int(self.np_rng.choice(best_actions))

    def update(
        self,
        state: State,
        action: int,
        reward: float,
        next_state: Optional[State],
        done: bool,
    ) -> float:
        """
        Apply one Q-learning update and return the temporal-difference error.
        """
        if action < 0 or action >= self.n_actions:
            raise ValueError(f"Invalid action index {action}.")

        current_q = self.q_table[state][action]
        if done or next_state is None:
            target = reward
        else:
            target = reward + self.gamma * float(np.max(self.q_table[next_state]))

        td_error = target - current_q
        self.q_table[state][action] = current_q + self.alpha * td_error
        return float(td_error)

    def decay_epsilon(self) -> float:
        """Decay epsilon after each episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return self.epsilon

    def set_epsilon(self, epsilon: float) -> None:
        """Manually set epsilon, useful before evaluation."""
        if not 0 <= epsilon <= 1:
            raise ValueError("epsilon should be in [0, 1].")
        self.epsilon = float(epsilon)

    def policy(self) -> Dict[str, int]:
        """
        Export greedy action index per visited state.
        Keys are JSON strings because states are usually tuples.
        """
        return {json.dumps(_state_to_jsonable(s)): int(np.argmax(q)) for s, q in self.q_table.items()}

    def save(self, path: str | Path) -> None:
        """Save the Q-table and parameters to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "n_actions": self.n_actions,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "epsilon_decay": self.epsilon_decay,
            "epsilon_min": self.epsilon_min,
            "q_table": [
                {
                    "state": _state_to_jsonable(state),
                    "q_values": q_values.tolist(),
                }
                for state, q_values in self.q_table.items()
            ],
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path, seed: Optional[int] = None) -> "QLearningAgent":
        """Load a saved Q-learning agent from JSON."""
        path = Path(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        agent = cls(
            n_actions=payload["n_actions"],
            alpha=payload["alpha"],
            gamma=payload["gamma"],
            epsilon=payload["epsilon"],
            epsilon_decay=payload["epsilon_decay"],
            epsilon_min=payload["epsilon_min"],
            seed=seed,
        )
        for row in payload["q_table"]:
            state = _jsonable_to_state(row["state"])
            agent.q_table[state] = np.array(row["q_values"], dtype=float)
        return agent


def _state_to_jsonable(state: Any) -> Any:
    if isinstance(state, tuple):
        return [_state_to_jsonable(x) for x in state]
    if isinstance(state, (np.integer,)):
        return int(state)
    if isinstance(state, (np.floating,)):
        return float(state)
    return state


def _jsonable_to_state(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_jsonable_to_state(x) for x in value)
    return value
