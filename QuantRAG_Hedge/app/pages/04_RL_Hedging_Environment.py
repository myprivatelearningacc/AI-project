import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.graph_objects as go


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


from src.envs.option_hedging_env import OptionHedgingEnv


st.set_page_config(
    page_title="RL Hedging Environment",
    layout="wide",
)

st.title("RL Hedging Environment")

st.markdown(
    """
This page tests the custom Gym-style environment for option hedging.

The environment uses a discrete state representation suitable for tabular Q-learning:

```text
state = (moneyness_bin, time_step, position_bin)
The agent chooses a target hedge position from a discrete action space:

actions = [-1.0, -0.5, 0.0, 0.5, 1.0]

"""
)

st.sidebar.header("Environment Parameters")

option_type = st.sidebar.selectbox("Option Type", ["call", "put"])

S0 = st.sidebar.number_input("Initial Stock Price S₀", min_value=1.0, value=100.0)
K = st.sidebar.number_input("Strike Price K", min_value=1.0, value=100.0)
T = st.sidebar.number_input("Time to Maturity T", min_value=0.01, value=1.0)
r = st.sidebar.number_input("Risk-free Rate r", value=0.05, step=0.01, format="%.4f")
sigma = st.sidebar.number_input("Volatility σ", min_value=0.01, value=0.20, step=0.01, format="%.4f")
mu = st.sidebar.number_input("GBM Drift μ", value=0.05, step=0.01, format="%.4f")

n_steps = st.sidebar.slider("Number of Steps", min_value=5, max_value=252, value=30, step=5)
transaction_cost_rate = st.sidebar.number_input(
"Transaction Cost Rate",
min_value=0.0,
value=0.001,
step=0.0005,
format="%.4f",
)

seed = st.sidebar.number_input("Random Seed", min_value=0, value=42, step=1)

policy_type = st.sidebar.selectbox(
"Demo Policy",
[
"Random policy",
"Always no hedge",
"Always 0.5 shares",
"Always 1.0 share",
],
)

def choose_action(env, policy_name):
    if policy_name == "Random policy":
        return env.sample_random_action()
    elif policy_name == "Always no hedge":
        return 2 # action 0.0
    elif policy_name == "Always 0.5 shares":
        return 3 # action 0.5
    elif policy_name == "Always 1.0 share":
        return 4 # action 1.0
    else:
        return env.sample_random_action()

if st.button("Run One Episode"):
    env = OptionHedgingEnv(
    S0=S0,
    K=K,
    T=T,
    r=r,
    sigma=sigma,
    mu=mu,
    n_steps=n_steps,
    transaction_cost_rate=transaction_cost_rate,
    option_type=option_type,
    seed=int(seed),
    )

    state = env.reset()

    done = False
    total_reward = 0.0
    final_info = None

    while not done:
        action = choose_action(env, policy_type)
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        state = next_state
        final_info = info

    history = pd.DataFrame(env.get_history())

    st.subheader("Final Episode Result")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Reward", f"{total_reward:.4f}")
    col2.metric("Final Hedging Error", f"{final_info['hedging_error']:.4f}")
    col3.metric("Total Transaction Cost", f"{final_info['total_transaction_cost']:.4f}")
    col4.metric("Final Stock Price", f"{final_info['stock_price']:.4f}")

    st.subheader("Discrete State Space Info")

    state_info = env.get_state_space_info()

    state_info_df = pd.DataFrame(
        {
            "Item": [
                "Number of moneyness states",
                "Number of time states",
                "Number of position states",
                "Action space",
                "Moneyness bins",
            ],
            "Value": [
                state_info["n_moneyness_states"],
                state_info["n_time_states"],
                state_info["n_position_states"],
                str(state_info["actions"]),
                str(state_info["moneyness_bins"]),
            ],
        }
    )

    st.dataframe(state_info_df, use_container_width=True)

    st.subheader("Episode History")

    st.dataframe(history, use_container_width=True)

    fig_stock = go.Figure()

    fig_stock.add_trace(
        go.Scatter(
            x=history["time_step"],
            y=history["stock_price"],
            mode="lines+markers",
            name="Stock Price",
        )
    )

    fig_stock.update_layout(
        title="Stock Price Path",
        xaxis_title="Time Step",
        yaxis_title="Stock Price",
        height=400,
    )

    st.plotly_chart(fig_stock, use_container_width=True)

    fig_position = go.Figure()

    fig_position.add_trace(
        go.Scatter(
            x=history["time_step"],
            y=history["hedge_position"],
            mode="lines+markers",
            name="Hedge Position",
        )
    )

    fig_position.update_layout(
        title="Hedge Position Over Time",
        xaxis_title="Time Step",
        yaxis_title="Hedge Position",
        height=400,
    )

    st.plotly_chart(fig_position, use_container_width=True)

    fig_reward = go.Figure()

    fig_reward.add_trace(
        go.Bar(
            x=history["time_step"],
            y=history["reward"],
            name="Reward",
        )
    )

    fig_reward.update_layout(
        title="Reward Per Step",
        xaxis_title="Time Step",
        yaxis_title="Reward",
        height=400,
    )

    st.plotly_chart(fig_reward, use_container_width=True)