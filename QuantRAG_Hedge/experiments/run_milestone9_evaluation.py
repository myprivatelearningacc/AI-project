import math
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# 1. Global config
# =========================

RESULT_DIR = Path("results/milestone9")
RESULT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 123, 2026, 777, 999]

ALGORITHMS = [
    "no_hedge",
    "bs_delta",
    "q_learning_basic",
    "q_learning_reward_variation",
    "q_learning_state_variation",
]


@dataclass
class MarketConfig:
    S0: float = 100.0
    K: float = 100.0
    r: float = 0.01
    sigma: float = 0.20
    T: float = 30 / 365
    n_steps: int = 30
    transaction_cost: float = 0.001
    option_type: str = "call"


@dataclass
class QConfig:
    episodes: int = 5000
    alpha: float = 0.10
    gamma: float = 0.99
    epsilon: float = 1.0
    epsilon_decay: float = 0.995
    epsilon_min: float = 0.05


# =========================
# 2. Black-Scholes utilities
# =========================

def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_price(S, K, r, sigma, tau, option_type="call"):
    if tau <= 0:
        if option_type == "call":
            return max(S - K, 0.0)
        return max(K - S, 0.0)

    if S <= 0 or K <= 0 or sigma <= 0:
        return 0.0

    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * tau) / (sigma * math.sqrt(tau))
    d2 = d1 - sigma * math.sqrt(tau)

    if option_type == "call":
        return S * norm_cdf(d1) - K * math.exp(-r * tau) * norm_cdf(d2)
    else:
        return K * math.exp(-r * tau) * norm_cdf(-d2) - S * norm_cdf(-d1)


def bs_delta(S, K, r, sigma, tau, option_type="call"):
    if tau <= 0:
        if option_type == "call":
            return 1.0 if S > K else 0.0
        return -1.0 if S < K else 0.0

    if S <= 0 or K <= 0 or sigma <= 0:
        return 0.0

    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * tau) / (sigma * math.sqrt(tau))

    if option_type == "call":
        return norm_cdf(d1)
    else:
        return norm_cdf(d1) - 1.0


def payoff(S, K, option_type="call"):
    if option_type == "call":
        return max(S - K, 0.0)
    return max(K - S, 0.0)


# =========================
# 3. Market simulation
# =========================

def generate_gbm_path(cfg: MarketConfig, rng: np.random.Generator):
    dt = cfg.T / cfg.n_steps
    prices = np.zeros(cfg.n_steps + 1)
    prices[0] = cfg.S0

    for t in range(cfg.n_steps):
        z = rng.normal()
        prices[t + 1] = prices[t] * math.exp(
            (cfg.r - 0.5 * cfg.sigma ** 2) * dt + cfg.sigma * math.sqrt(dt) * z
        )

    return prices


# =========================
# 4. Baseline evaluation
# =========================

def evaluate_fixed_policy(cfg: MarketConfig, seed: int, policy_name: str, n_paths: int = 1000):
    rng = np.random.default_rng(seed)
    rows = []

    dt = cfg.T / cfg.n_steps
    premium = bs_price(cfg.S0, cfg.K, cfg.r, cfg.sigma, cfg.T, cfg.option_type)

    for path_id in range(n_paths):
        S_path = generate_gbm_path(cfg, rng)

        cash = premium
        phi = 0.0
        total_tc = 0.0
        n_trades = 0

        for t in range(cfg.n_steps):
            S_t = S_path[t]
            tau = cfg.T - t * dt

            if policy_name == "no_hedge":
                target_phi = 0.0
            elif policy_name == "bs_delta":
                target_phi = bs_delta(S_t, cfg.K, cfg.r, cfg.sigma, tau, cfg.option_type)
            else:
                raise ValueError(f"Unknown fixed policy: {policy_name}")

            dphi = target_phi - phi
            tc = cfg.transaction_cost * abs(dphi) * S_t

            if abs(dphi) > 1e-8:
                n_trades += 1

            cash -= dphi * S_t
            cash -= tc
            total_tc += tc

            cash *= math.exp(cfg.r * dt)
            phi = target_phi

        S_T = S_path[-1]
        final_payoff = payoff(S_T, cfg.K, cfg.option_type)
        portfolio_value = cash + phi * S_T

        hedging_error = portfolio_value - final_payoff
        final_reward = -abs(hedging_error) - total_tc

        rows.append({
            "seed": seed,
            "path_id": path_id,
            "algorithm": policy_name,
            "hedging_error": hedging_error,
            "abs_hedging_error": abs(hedging_error),
            "transaction_cost": total_tc,
            "n_trades": n_trades,
            "final_reward": final_reward,
        })

    return pd.DataFrame(rows)


# =========================
# 5. Q-learning agent
# =========================

class QLearningHedger:
    def __init__(self, cfg: MarketConfig, qcfg: QConfig, variant: str, seed: int):
        self.cfg = cfg
        self.qcfg = qcfg
        self.variant = variant
        self.rng = np.random.default_rng(seed)

        self.actions = np.array([0.0, 0.25, 0.50, 0.75, 1.0])
        self.q_table = {}

        self.moneyness_bins = np.array([0.70, 0.80, 0.90, 0.97, 1.03, 1.10, 1.20, 1.30])
        self.position_bins = np.array([0.125, 0.375, 0.625, 0.875])
        self.move_bins = np.array([0.98, 1.00, 1.02])

    def _bin(self, value, bins):
        return int(np.digitize(value, bins))

    def state(self, S_t, t, phi, prev_S=None):
        moneyness = S_t / self.cfg.K
        time_bin = int(10 * t / self.cfg.n_steps)
        money_bin = self._bin(moneyness, self.moneyness_bins)

        if self.variant == "q_learning_state_variation":
            pos_bin = self._bin(phi, self.position_bins)

            if prev_S is None:
                move_bin = 1
            else:
                move_bin = self._bin(S_t / prev_S, self.move_bins)

            return (money_bin, time_bin, pos_bin, move_bin)

        return (money_bin, time_bin)

    def get_q(self, state):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(self.actions))
        return self.q_table[state]

    def choose_action(self, state, epsilon):
        if self.rng.random() < epsilon:
            return int(self.rng.integers(len(self.actions)))
        return int(np.argmax(self.get_q(state)))

    def reward_function(self, mark_error, transaction_cost, dphi, terminal=False):
        if self.variant == "q_learning_reward_variation":
            reward = -abs(mark_error) - 5.0 * transaction_cost - 0.05 * abs(dphi)

            if terminal:
                reward -= 2.0 * abs(mark_error)

            return reward

        return -abs(mark_error) - transaction_cost

    def train(self):
        episode_rewards = []
        epsilon = self.qcfg.epsilon
        dt = self.cfg.T / self.cfg.n_steps

        for ep in range(self.qcfg.episodes):
            S_path = generate_gbm_path(self.cfg, self.rng)

            premium = bs_price(
                self.cfg.S0,
                self.cfg.K,
                self.cfg.r,
                self.cfg.sigma,
                self.cfg.T,
                self.cfg.option_type,
            )

            cash = premium
            phi = 0.0
            total_reward = 0.0

            for t in range(self.cfg.n_steps):
                S_t = S_path[t]
                S_next = S_path[t + 1]
                prev_S = S_path[t - 1] if t > 0 else None

                tau = self.cfg.T - t * dt
                next_tau = max(self.cfg.T - (t + 1) * dt, 0.0)

                s = self.state(S_t, t, phi, prev_S)
                action_idx = self.choose_action(s, epsilon)
                target_phi = self.actions[action_idx]

                dphi = target_phi - phi
                tc = self.cfg.transaction_cost * abs(dphi) * S_t

                cash -= dphi * S_t
                cash -= tc
                cash *= math.exp(self.cfg.r * dt)
                phi = target_phi

                portfolio_next = cash + phi * S_next

                if t == self.cfg.n_steps - 1:
                    option_value_next = payoff(S_next, self.cfg.K, self.cfg.option_type)
                    terminal = True
                else:
                    option_value_next = bs_price(
                        S_next,
                        self.cfg.K,
                        self.cfg.r,
                        self.cfg.sigma,
                        next_tau,
                        self.cfg.option_type,
                    )
                    terminal = False

                mark_error = portfolio_next - option_value_next
                reward = self.reward_function(mark_error, tc, dphi, terminal)

                s_next = self.state(S_next, t + 1, phi, S_t)
                q_current = self.get_q(s)[action_idx]
                q_next_max = 0.0 if terminal else np.max(self.get_q(s_next))

                td_target = reward + self.qcfg.gamma * q_next_max
                self.q_table[s][action_idx] += self.qcfg.alpha * (td_target - q_current)

                total_reward += reward

            epsilon = max(self.qcfg.epsilon_min, epsilon * self.qcfg.epsilon_decay)
            episode_rewards.append(total_reward)

        return pd.DataFrame({
            "episode": np.arange(1, self.qcfg.episodes + 1),
            "episode_reward": episode_rewards,
            "variant": self.variant,
        })

    def evaluate(self, seed: int, n_paths: int = 1000):
        rng = np.random.default_rng(seed)
        rows = []

        dt = self.cfg.T / self.cfg.n_steps
        premium = bs_price(self.cfg.S0, self.cfg.K, self.cfg.r, self.cfg.sigma, self.cfg.T, self.cfg.option_type)

        for path_id in range(n_paths):
            S_path = generate_gbm_path(self.cfg, rng)

            cash = premium
            phi = 0.0
            total_tc = 0.0
            n_trades = 0

            for t in range(self.cfg.n_steps):
                S_t = S_path[t]
                prev_S = S_path[t - 1] if t > 0 else None

                s = self.state(S_t, t, phi, prev_S)
                action_idx = int(np.argmax(self.get_q(s)))
                target_phi = self.actions[action_idx]

                dphi = target_phi - phi
                tc = self.cfg.transaction_cost * abs(dphi) * S_t

                if abs(dphi) > 1e-8:
                    n_trades += 1

                cash -= dphi * S_t
                cash -= tc
                total_tc += tc

                cash *= math.exp(self.cfg.r * dt)
                phi = target_phi

            S_T = S_path[-1]
            final_payoff = payoff(S_T, self.cfg.K, self.cfg.option_type)
            portfolio_value = cash + phi * S_T

            hedging_error = portfolio_value - final_payoff
            final_reward = -abs(hedging_error) - total_tc

            rows.append({
                "seed": seed,
                "path_id": path_id,
                "algorithm": self.variant,
                "hedging_error": hedging_error,
                "abs_hedging_error": abs(hedging_error),
                "transaction_cost": total_tc,
                "n_trades": n_trades,
                "final_reward": final_reward,
            })

        return pd.DataFrame(rows)

    def save_policy_heatmap(self, output_path: Path):
        records = []

        fixed_phi = 0.50

        for money_bin_id in range(len(self.moneyness_bins) + 1):
            for time_bin in range(11):
                state = (money_bin_id, time_bin)

                q_values = self.get_q(state)
                best_action = self.actions[int(np.argmax(q_values))]

                records.append({
                    "moneyness_bin": money_bin_id,
                    "time_bin": time_bin,
                    "best_hedge_ratio": best_action,
                })

        pd.DataFrame(records).to_csv(output_path, index=False)


# =========================
# 6. Metrics
# =========================

def cvar_95_abs_error(abs_errors):
    threshold = np.percentile(abs_errors, 95)
    tail = abs_errors[abs_errors >= threshold]
    return float(np.mean(tail))


def summarize_metrics(df):
    rows = []

    for algorithm, g in df.groupby("algorithm"):
        rows.append({
            "algorithm": algorithm,
            "mean_abs_hedging_error": g["abs_hedging_error"].mean(),
            "std_hedging_error": g["hedging_error"].std(),
            "cvar_95_abs_error": cvar_95_abs_error(g["abs_hedging_error"].values),
            "mean_transaction_cost": g["transaction_cost"].mean(),
            "mean_number_of_trades": g["n_trades"].mean(),
            "mean_final_reward": g["final_reward"].mean(),
        })

    summary = pd.DataFrame(rows)
    return summary.sort_values("mean_abs_hedging_error")


# =========================
# 7. Plots
# =========================

def plot_learning_curve(learning_df):
    plt.figure(figsize=(9, 5))

    for variant, g in learning_df.groupby("variant"):
        curve = g.groupby("episode")["episode_reward"].mean()
        smooth = curve.rolling(100, min_periods=1).mean()
        plt.plot(smooth.index, smooth.values, label=variant)

    plt.xlabel("Episode")
    plt.ylabel("Episode reward, rolling mean 100")
    plt.title("Learning curve: Q-learning episode reward")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "01_learning_curve.png", dpi=200)
    plt.close()


def plot_hedging_error_distribution(df):
    selected = df[df["algorithm"].isin(["bs_delta", "q_learning_basic"])]

    plt.figure(figsize=(9, 5))

    for algorithm, g in selected.groupby("algorithm"):
        plt.hist(
            g["hedging_error"],
            bins=40,
            alpha=0.55,
            density=True,
            label=algorithm,
        )

    plt.xlabel("Final hedging error")
    plt.ylabel("Density")
    plt.title("Hedging error distribution: delta hedge vs Q-learning")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "02_hedging_error_distribution.png", dpi=200)
    plt.close()


def plot_transaction_cost_vs_error(df):
    plt.figure(figsize=(8, 5))

    for algorithm, g in df.groupby("algorithm"):
        sample = g.sample(min(len(g), 700), random_state=42)
        plt.scatter(
            sample["transaction_cost"],
            sample["abs_hedging_error"],
            alpha=0.35,
            label=algorithm,
        )

    plt.xlabel("Transaction cost")
    plt.ylabel("Absolute hedging error")
    plt.title("Transaction cost vs hedging error")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "03_transaction_cost_vs_error.png", dpi=200)
    plt.close()


def plot_bar_mean_abs_error(summary_df):
    ordered = summary_df.sort_values("mean_abs_hedging_error")

    plt.figure(figsize=(9, 5))
    plt.bar(ordered["algorithm"], ordered["mean_abs_hedging_error"])
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Mean absolute hedging error")
    plt.title("Mean absolute hedging error by algorithm")
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "04_bar_mean_abs_error.png", dpi=200)
    plt.close()


def plot_policy_heatmap(policy_csv):
    policy_df = pd.read_csv(policy_csv)

    pivot = policy_df.pivot(
        index="moneyness_bin",
        columns="time_bin",
        values="best_hedge_ratio",
    )

    plt.figure(figsize=(9, 5))
    plt.imshow(pivot.values, aspect="auto", origin="lower")
    plt.colorbar(label="Best hedge ratio")
    plt.xlabel("Time bin")
    plt.ylabel("Moneyness bin")
    plt.title("Q-learning policy heatmap: action by moneyness and time")
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "05_policy_heatmap.png", dpi=200)
    plt.close()


# =========================
# 8. Main experiment
# =========================

def main():
    cfg = MarketConfig()
    qcfg = QConfig()

    all_eval_results = []
    all_learning_curves = []

    print("Running fixed baselines...")

    for seed in SEEDS:
        all_eval_results.append(evaluate_fixed_policy(cfg, seed, "no_hedge"))
        all_eval_results.append(evaluate_fixed_policy(cfg, seed, "bs_delta"))

    q_variants = [
        "q_learning_basic",
        "q_learning_reward_variation",
        "q_learning_state_variation",
    ]

    print("Training and evaluating Q-learning agents...")

    for variant in q_variants:
        for seed in SEEDS:
            print(f"Training {variant}, seed={seed}")

            agent = QLearningHedger(cfg, qcfg, variant=variant, seed=seed)
            learning_df = agent.train()
            learning_df["seed"] = seed

            eval_df = agent.evaluate(seed=seed, n_paths=1000)

            all_learning_curves.append(learning_df)
            all_eval_results.append(eval_df)

            with open(RESULT_DIR / f"{variant}_seed_{seed}_qtable.pkl", "wb") as f:
                pickle.dump(agent.q_table, f)

            if variant == "q_learning_basic" and seed == 42:
                agent.save_policy_heatmap(RESULT_DIR / "policy_heatmap_data_q_basic_seed42.csv")

    eval_df = pd.concat(all_eval_results, ignore_index=True)
    learning_df = pd.concat(all_learning_curves, ignore_index=True)

    summary_df = summarize_metrics(eval_df)

    eval_df.to_csv(RESULT_DIR / "evaluation_results_raw.csv", index=False)
    learning_df.to_csv(RESULT_DIR / "learning_curves.csv", index=False)
    summary_df.to_csv(RESULT_DIR / "summary_metrics.csv", index=False)

    plot_learning_curve(learning_df)
    plot_hedging_error_distribution(eval_df)
    plot_transaction_cost_vs_error(eval_df)
    plot_bar_mean_abs_error(summary_df)
    plot_policy_heatmap(RESULT_DIR / "policy_heatmap_data_q_basic_seed42.csv")

    print("\nDone. Summary metrics:")
    print(summary_df.to_string(index=False))

    print(f"\nSaved outputs to: {RESULT_DIR}")


if __name__ == "__main__":
    main()