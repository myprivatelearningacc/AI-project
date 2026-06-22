import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px


# Make project root importable
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


from src.tools.greeks import calculate_all_greeks
from src.tools.gbm_simulator import simulate_gbm_paths
from src.tools.delta_hedging import simulate_delta_hedge


st.set_page_config(
    page_title="Greeks & Delta Hedging",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Greeks & Delta Hedging Simulator")

st.markdown(
    """
This page implements a classical Black-Scholes hedging baseline.

It includes:

1. Black-Scholes Greeks calculator  
2. Geometric Brownian Motion stock path simulator  
3. Delta hedging simulator with transaction costs  

This baseline will later be compared with Q-learning or reinforcement learning hedging strategies.
"""
)


# =========================
# Sidebar inputs
# =========================

st.sidebar.header("Option Parameters")

option_type = st.sidebar.selectbox(
    "Option Type",
    ["call", "put"],
    index=0,
)

S0 = st.sidebar.number_input(
    "Initial Stock Price S₀",
    min_value=1.0,
    value=100.0,
    step=1.0,
)

K = st.sidebar.number_input(
    "Strike Price K",
    min_value=1.0,
    value=100.0,
    step=1.0,
)

T = st.sidebar.number_input(
    "Time to Maturity T, in years",
    min_value=0.01,
    value=1.0,
    step=0.05,
)

r = st.sidebar.number_input(
    "Risk-free Rate r",
    value=0.05,
    step=0.01,
    format="%.4f",
)

sigma = st.sidebar.number_input(
    "Volatility σ",
    min_value=0.01,
    value=0.20,
    step=0.01,
    format="%.4f",
)


st.sidebar.header("Simulation Parameters")

mu = st.sidebar.number_input(
    "GBM Drift μ",
    value=0.05,
    step=0.01,
    format="%.4f",
)

n_steps = st.sidebar.slider(
    "Number of Rebalancing Steps",
    min_value=5,
    max_value=252,
    value=50,
    step=5,
)

n_paths = st.sidebar.slider(
    "Number of Simulated Paths",
    min_value=100,
    max_value=10000,
    value=1000,
    step=100,
)

sample_paths_to_show = st.sidebar.slider(
    "Sample Paths to Show",
    min_value=5,
    max_value=20,
    value=10,
    step=1,
)

transaction_cost_rate = st.sidebar.number_input(
    "Transaction Cost Rate",
    min_value=0.0,
    value=0.001,
    step=0.0005,
    format="%.4f",
)

seed = st.sidebar.number_input(
    "Random Seed",
    min_value=0,
    value=42,
    step=1,
)


# =========================
# Section 1: Greeks Calculator
# =========================

st.header("1. Black-Scholes Greeks Calculator")

try:
    greeks = calculate_all_greeks(
        S=S0,
        K=K,
        T=T,
        r=r,
        sigma=sigma,
        option_type=option_type,
    )

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Option Price", f"{greeks['price']:.4f}")
    col2.metric("Delta", f"{greeks['delta']:.4f}")
    col3.metric("Gamma", f"{greeks['gamma']:.6f}")
    col4.metric("Vega", f"{greeks['vega']:.4f}")
    col5.metric("Theta", f"{greeks['theta']:.4f}")

    st.info(
        """
Interpretation:

- **Delta**: sensitivity of option price to stock price.
- **Gamma**: sensitivity of delta to stock price.
- **Vega**: sensitivity of option price to volatility.
- **Theta**: sensitivity of option price to time decay.
"""
    )

except Exception as e:
    st.error(f"Greeks calculation error: {e}")


# =========================
# Section 2: GBM Simulator
# =========================

st.header("2. Geometric Brownian Motion Stock Simulator")

try:
    stock_paths = simulate_gbm_paths(
        S0=S0,
        mu=mu,
        sigma=sigma,
        T=T,
        n_steps=n_steps,
        n_paths=n_paths,
        seed=int(seed),
    )

    time_grid = np.linspace(0, T, n_steps + 1)

    fig_paths = go.Figure()

    for i in range(min(sample_paths_to_show, n_paths)):
        fig_paths.add_trace(
            go.Scatter(
                x=time_grid,
                y=stock_paths[i],
                mode="lines",
                name=f"Path {i + 1}",
                showlegend=False,
            )
        )

    fig_paths.update_layout(
        title="Sample Simulated Stock Paths",
        xaxis_title="Time",
        yaxis_title="Stock Price",
        height=450,
    )

    st.plotly_chart(fig_paths, use_container_width=True)

    terminal_prices = stock_paths[:, -1]

    fig_terminal = px.histogram(
        x=terminal_prices,
        nbins=50,
        title="Terminal Stock Price Distribution",
        labels={"x": "Terminal Stock Price", "y": "Frequency"},
    )

    st.plotly_chart(fig_terminal, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Mean Terminal Price", f"{np.mean(terminal_prices):.4f}")
    col2.metric("Std Terminal Price", f"{np.std(terminal_prices):.4f}")
    col3.metric("Median Terminal Price", f"{np.median(terminal_prices):.4f}")

except Exception as e:
    st.error(f"GBM simulation error: {e}")


# =========================
# Section 3: Delta Hedging
# =========================

st.header("3. Delta Hedging Simulator")

st.markdown(
    """
The simulation assumes the trader sells one European option and receives the Black-Scholes premium.

Then we compare:

- **No Hedge**: keep the option premium in cash until maturity.
- **Delta Hedge**: rebalance stock holdings according to Black-Scholes delta.

The final hedging error is:

```text
hedging error = final portfolio value - option payoff
A better hedge should usually have a smaller error distribution.
"""
)

run_hedge = st.button("Run Delta Hedging Simulation")

if run_hedge:
    try:
        results = simulate_delta_hedge(
        S0=S0,
        K=K,
        T=T,
        r=r,
        sigma=sigma,
        n_steps=n_steps,
        n_paths=n_paths,
        transaction_cost_rate=transaction_cost_rate,
        option_type=option_type,
        mu=mu,
        seed=int(seed),
        )

        summary = results["summary"]
        path_results = results["path_results"]

        st.subheader("Hedging Summary Metrics")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Initial Option Price", f"{summary['initial_option_price']:.4f}")
        col2.metric("Rebalances", summary["number_of_rebalances"])
        col3.metric("Mean Transaction Cost", f"{summary['mean_transaction_cost']:.4f}")
        col4.metric("Total Transaction Cost", f"{summary['total_transaction_cost']:.4f}")

        st.subheader("No Hedge vs Delta Hedge")

        metrics_df = pd.DataFrame(
            {
                "Strategy": ["No Hedge", "Delta Hedge"],
                "Mean Error": [
                    summary["no_hedge_mean_error"],
                    summary["delta_hedge_mean_error"],
                ],
                "Std Error": [
                    summary["no_hedge_std_error"],
                    summary["delta_hedge_std_error"],
                ],
                "Mean Absolute Error": [
                    summary["no_hedge_mae"],
                    summary["delta_hedge_mae"],
                ],
            }
        )

        st.dataframe(metrics_df, use_container_width=True)

        fig_error = go.Figure()

        fig_error.add_trace(
            go.Histogram(
                x=results["no_hedge_error"],
                name="No Hedge",
                opacity=0.60,
                nbinsx=60,
            )
        )

        fig_error.add_trace(
            go.Histogram(
                x=results["delta_hedge_error"],
                name="Delta Hedge",
                opacity=0.60,
                nbinsx=60,
            )
        )

        fig_error.update_layout(
            title="Hedging Error Distribution",
            xaxis_title="Hedging Error",
            yaxis_title="Frequency",
            barmode="overlay",
            height=450,
        )

        st.plotly_chart(fig_error, use_container_width=True)

        fig_tc_error = px.scatter(
            path_results,
            x="total_transaction_cost",
            y="delta_hedge_error",
            title="Transaction Cost vs Delta Hedging Error",
            labels={
                "total_transaction_cost": "Total Transaction Cost",
                "delta_hedge_error": "Delta Hedging Error",
            },
            opacity=0.5,
        )

        st.plotly_chart(fig_tc_error, use_container_width=True)

        st.subheader("Path-level Results Sample")

        st.dataframe(path_results.head(20), use_container_width=True)

        csv = path_results.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Hedging Results CSV",
            data=csv,
            file_name="delta_hedging_results.csv",
            mime="text/csv",
        )

    except Exception as e:
        st.error(f"Delta hedging simulation error: {e}")