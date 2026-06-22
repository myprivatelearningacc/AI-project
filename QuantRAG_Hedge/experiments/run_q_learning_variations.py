"""
Run Q-learning hedging variations for Milestone 8.

Variations included:
1. Reward design:
   A: - final_error^2
   B: - final_error^2 - transaction_cost
   C: - final_error^2 - lambda * transaction_cost

2. State representation:
   A: (moneyness_bin, time_bin)
   B: (moneyness_bin, time_bin, position_bin)

3. Transaction cost level:
   cost = 0.0%, 0.1%, 0.5%, 1.0%

4. Optional hyperparameters:
   alpha = 0.05, 0.1, 0.2
   epsilon_decay = 0.99, 0.995, 0.999

Run from project root:
    python experiments/run_q_learning_variations.py

Faster test run:
    python experiments/run_q_learning_variations.py --episodes 1000 --skip_hyperparams
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.train_q_learning import HedgingConfig, evaluate_agent, train_agent


def run_one(config: HedgingConfig, output_dir: Path, run_name: str, group: str) -> Dict[str, float]:
    print(f"\n=== Running {group}: {run_name} ===")
    agent, history = train_agent(config=config, output_dir=output_dir / group, run_name=run_name)
    eval_metrics = evaluate_agent(config, agent, n_eval_episodes=500)

    result = {
        "group": group,
        "run_name": run_name,
        **asdict(config),
        "train_final_100_reward": float(history["reward"].tail(100).mean()),
        "train_final_100_abs_error": float(history["abs_final_error"].tail(100).mean()),
        "train_final_100_transaction_cost": float(history["transaction_cost"].tail(100).mean()),
        **eval_metrics,
        "visited_states": len(agent.q_table),
    }
    print(pd.Series(result)[[
        "train_final_100_abs_error",
        "train_final_100_transaction_cost",
        "eval_mean_abs_error",
        "eval_rmse_error",
        "eval_mean_transaction_cost",
        "visited_states",
    ]].to_string())
    return result


def run_all_variations(base_config: HedgingConfig, output_dir: Path, include_hyperparams: bool = True) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    results: List[Dict[str, float]] = []

    # Variation 1: reward design
    reward_variants = [
        ("reward_A_error_only", {"reward_type": "final_error", "cost_lambda": 1.0}),
        ("reward_B_error_plus_cost", {"reward_type": "final_error_cost", "cost_lambda": 1.0}),
        ("reward_C_error_plus_lambda_cost", {"reward_type": "final_error_lambda_cost", "cost_lambda": 5.0}),
    ]
    for run_name, changes in reward_variants:
        cfg = clone_config(base_config, **changes)
        results.append(run_one(cfg, output_dir, run_name, group="variation_1_reward"))

    # Variation 2: state representation
    state_variants = [
        ("state_A_moneyness_time", {"state_type": "basic"}),
        ("state_B_moneyness_time_position", {"state_type": "with_position"}),
    ]
    for run_name, changes in state_variants:
        cfg = clone_config(base_config, **changes)
        results.append(run_one(cfg, output_dir, run_name, group="variation_2_state"))

    # Variation 3: transaction cost levels
    cost_variants = [
        ("cost_0_0pct", 0.000),
        ("cost_0_1pct", 0.001),
        ("cost_0_5pct", 0.005),
        ("cost_1_0pct", 0.010),
    ]
    for run_name, cost_rate in cost_variants:
        cfg = clone_config(base_config, cost_rate=cost_rate, reward_type="final_error_cost")
        results.append(run_one(cfg, output_dir, run_name, group="variation_3_cost"))

    # Variation 4: optional hyperparameters
    if include_hyperparams:
        for alpha in [0.05, 0.1, 0.2]:
            cfg = clone_config(base_config, alpha=alpha)
            results.append(run_one(cfg, output_dir, f"alpha_{alpha}", group="variation_4_hyperparams"))

        for epsilon_decay in [0.99, 0.995, 0.999]:
            cfg = clone_config(base_config, epsilon_decay=epsilon_decay)
            results.append(run_one(cfg, output_dir, f"epsilon_decay_{epsilon_decay}", group="variation_4_hyperparams"))

    summary = pd.DataFrame(results)
    summary_path = output_dir / "q_learning_variation_summary.csv"
    summary.to_csv(summary_path, index=False)
    plot_summary(summary, output_dir)
    return summary


def clone_config(config: HedgingConfig, **changes) -> HedgingConfig:
    data = asdict(config)
    data.update(changes)
    return HedgingConfig(**data)


def plot_summary(summary: pd.DataFrame, output_dir: Path) -> None:
    metric = "eval_mean_abs_error"
    ordered = summary.sort_values(metric)

    plt.figure(figsize=(12, 6))
    plt.bar(ordered["run_name"], ordered[metric])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Evaluation mean absolute hedging error")
    plt.title("Q-learning variation comparison")
    plt.tight_layout()
    plt.savefig(output_dir / "variation_eval_abs_error_comparison.png", dpi=160)
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.bar(ordered["run_name"], ordered["eval_mean_transaction_cost"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Evaluation mean transaction cost")
    plt.title("Transaction cost comparison across variations")
    plt.tight_layout()
    plt.savefig(output_dir / "variation_transaction_cost_comparison.png", dpi=160)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Milestone 8 Q-learning variations.")
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="outputs/milestone8_q_learning_variations")
    parser.add_argument("--skip_hyperparams", action="store_true", help="Skip optional hyperparameter variation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_config = HedgingConfig(
        episodes=args.episodes,
        seed=args.seed,
        alpha=0.1,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.05,
        cost_rate=0.001,
        reward_type="final_error",
        state_type="basic",
    )
    summary = run_all_variations(
        base_config=base_config,
        output_dir=Path(args.output_dir),
        include_hyperparams=not args.skip_hyperparams,
    )
    print("\nAll variations completed.")
    print(summary[[
        "group",
        "run_name",
        "eval_mean_abs_error",
        "eval_rmse_error",
        "eval_mean_transaction_cost",
        "visited_states",
    ]].sort_values("eval_mean_abs_error").to_string(index=False))
    print(f"\nSaved summary to: {Path(args.output_dir) / 'q_learning_variation_summary.csv'}")


if __name__ == "__main__":
    main()
