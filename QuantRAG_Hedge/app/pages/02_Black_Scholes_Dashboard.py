import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

APP_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = APP_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.black_scholes import BlackScholesModel


st.set_page_config(
    page_title="Black-Scholes Dashboard",
    layout="wide",
)


st.title("Milestone 5 — Black-Scholes Pricing Baseline")

st.markdown(
    """
This dashboard implements the **Black-Scholes model** as a classical pricing baseline.
It supports European call and put options, computes Greeks, and provides visual sensitivity analysis.

In later milestones, this module will serve as the baseline pricing engine for delta hedging experiments.
"""
)


# =========================
# Sidebar inputs
# =========================

st.sidebar.header("Model Inputs")

option_type = st.sidebar.radio(
    "Option Type",
    ["Call", "Put"],
    index=0,
)

S = st.sidebar.number_input(
    "Current Stock Price (S)",
    min_value=0.01,
    max_value=10000.0,
    value=100.0,
    step=1.0,
)

K = st.sidebar.number_input(
    "Strike Price (K)",
    min_value=0.01,
    max_value=10000.0,
    value=100.0,
    step=1.0,
)

T = st.sidebar.number_input(
    "Time to Maturity (T, years)",
    min_value=0.0001,
    max_value=30.0,
    value=1.0,
    step=0.1,
)

r_percent = st.sidebar.number_input(
    "Risk-free Rate (%)",
    min_value=-50.0,
    max_value=100.0,
    value=5.0,
    step=0.1,
)

sigma_percent = st.sidebar.number_input(
    "Volatility (%)",
    min_value=0.01,
    max_value=500.0,
    value=20.0,
    step=0.5,
)

r = r_percent / 100
sigma = sigma_percent / 100

option_type_lower = option_type.lower()


# =========================
# Input validation
# =========================

validation_errors = []

if S <= 0:
    validation_errors.append("Current stock price S must be greater than 0.")

if K <= 0:
    validation_errors.append("Strike price K must be greater than 0.")

if T <= 0:
    validation_errors.append("Time to maturity T must be greater than 0.")

if sigma <= 0:
    validation_errors.append("Volatility sigma must be greater than 0.")

# r can be any float, including negative values.

if validation_errors:
    st.error("Invalid input detected.")
    for error in validation_errors:
        st.write(f"- {error}")
    st.stop()


# =========================
# Model calculation
# =========================

try:
    model = BlackScholesModel(
        S=S,
        K=K,
        T=T,
        r=r,
        sigma=sigma,
    )

    summary = model.summary()

except ValueError as e:
    st.error(f"Model error: {e}")
    st.stop()


call_price = summary["call_price"]
put_price = summary["put_price"]

selected_price = call_price if option_type_lower == "call" else put_price
selected_delta = summary["call_delta"] if option_type_lower == "call" else summary["put_delta"]
selected_theta = summary["call_theta"] if option_type_lower == "call" else summary["put_theta"]
selected_rho = summary["call_rho"] if option_type_lower == "call" else summary["put_rho"]


# =========================
# Main KPI cards
# =========================

st.subheader("Option Price Summary")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(f"{option_type} Price", f"${selected_price:,.4f}")

with col2:
    st.metric("Call Price", f"${call_price:,.4f}")

with col3:
    st.metric("Put Price", f"${put_price:,.4f}")

with col4:
    moneyness = S / K
    st.metric("Moneyness S/K", f"{moneyness:.4f}")


col5, col6, col7, col8 = st.columns(4)

with col5:
    st.metric("d1", f"{summary['d1']:.4f}")

with col6:
    st.metric("d2", f"{summary['d2']:.4f}")

with col7:
    intrinsic_value = max(S - K, 0) if option_type_lower == "call" else max(K - S, 0)
    st.metric("Intrinsic Value", f"${intrinsic_value:,.4f}")

with col8:
    time_value = selected_price - intrinsic_value
    st.metric("Time Value", f"${time_value:,.4f}")


# =========================
# Greeks table
# =========================

st.subheader("Greeks")

greeks_df = pd.DataFrame(
    {
        "Greek": ["Delta", "Gamma", "Vega", "Theta", "Rho"],
        "Selected Option": [
            selected_delta,
            summary["gamma"],
            summary["vega"],
            selected_theta,
            selected_rho,
        ],
        "Call": [
            summary["call_delta"],
            summary["gamma"],
            summary["vega"],
            summary["call_theta"],
            summary["call_rho"],
        ],
        "Put": [
            summary["put_delta"],
            summary["gamma"],
            summary["vega"],
            summary["put_theta"],
            summary["put_rho"],
        ],
        "Meaning": [
            "Sensitivity to stock price movement",
            "Sensitivity of Delta to stock price movement",
            "Sensitivity to volatility change",
            "Time decay per day",
            "Sensitivity to interest rate change",
        ],
    }
)

st.dataframe(
    greeks_df.style.format(
        {
            "Selected Option": "{:.6f}",
            "Call": "{:.6f}",
            "Put": "{:.6f}",
        }
    ),
    use_container_width=True,
    hide_index=True,
)


# =========================
# Payoff chart
# =========================

st.subheader(f"{option_type} Payoff and Profit at Expiration")

S_range = np.linspace(max(0.01, S * 0.4), S * 1.8, 250)

if option_type_lower == "call":
    payoff = np.maximum(S_range - K, 0)
else:
    payoff = np.maximum(K - S_range, 0)

profit = payoff - selected_price

fig_payoff = go.Figure()

fig_payoff.add_trace(
    go.Scatter(
        x=S_range,
        y=payoff,
        mode="lines",
        name=f"{option_type} Payoff",
    )
)

fig_payoff.add_trace(
    go.Scatter(
        x=S_range,
        y=profit,
        mode="lines",
        name=f"{option_type} Profit",
    )
)

fig_payoff.add_hline(
    y=0,
    line_dash="dash",
    annotation_text="Zero P&L",
    annotation_position="bottom right",
)

fig_payoff.add_vline(
    x=K,
    line_dash="dash",
    annotation_text="Strike K",
    annotation_position="top",
)

fig_payoff.add_vline(
    x=S,
    line_dash="dot",
    annotation_text="Current S",
    annotation_position="bottom",
)

fig_payoff.update_layout(
    xaxis_title="Stock Price at Expiration",
    yaxis_title="Payoff / Profit",
    hovermode="x unified",
    height=500,
)

st.plotly_chart(fig_payoff, use_container_width=True)


# =========================
# Option price sensitivity
# =========================

st.subheader(f"{option_type} Price Sensitivity to Stock Price")

selected_prices = []
call_prices = []
put_prices = []

for price in S_range:
    temp_model = BlackScholesModel(
        S=price,
        K=K,
        T=T,
        r=r,
        sigma=sigma,
    )

    temp_call = temp_model.call_price()
    temp_put = temp_model.put_price()

    call_prices.append(temp_call)
    put_prices.append(temp_put)

    if option_type_lower == "call":
        selected_prices.append(temp_call)
    else:
        selected_prices.append(temp_put)

fig_price = go.Figure()

fig_price.add_trace(
    go.Scatter(
        x=S_range,
        y=selected_prices,
        mode="lines",
        name=f"{option_type} Price",
    )
)

fig_price.add_trace(
    go.Scatter(
        x=S_range,
        y=call_prices,
        mode="lines",
        name="Call Price",
        visible="legendonly" if option_type_lower == "put" else True,
    )
)

fig_price.add_trace(
    go.Scatter(
        x=S_range,
        y=put_prices,
        mode="lines",
        name="Put Price",
        visible="legendonly" if option_type_lower == "call" else True,
    )
)

fig_price.add_vline(
    x=S,
    line_dash="dot",
    annotation_text="Current S",
    annotation_position="top",
)

fig_price.add_vline(
    x=K,
    line_dash="dash",
    annotation_text="Strike K",
    annotation_position="bottom",
)

fig_price.update_layout(
    xaxis_title="Underlying Stock Price",
    yaxis_title="Option Price",
    hovermode="x unified",
    height=500,
)

st.plotly_chart(fig_price, use_container_width=True)


# =========================
# Greeks sensitivity curves
# =========================

st.subheader(f"{option_type} Greeks across Stock Prices")

greek_rows = []

for price in S_range:
    temp_model = BlackScholesModel(
        S=price,
        K=K,
        T=T,
        r=r,
        sigma=sigma,
    )

    greek_rows.append(
        {
            "stock_price": price,
            "delta": temp_model.delta(option_type_lower),
            "gamma": temp_model.gamma(),
            "vega": temp_model.vega(),
            "theta": temp_model.theta(option_type_lower),
            "rho": temp_model.rho(option_type_lower),
        }
    )

greeks_curve_df = pd.DataFrame(greek_rows)

selected_greeks = st.multiselect(
    "Choose Greeks to visualize",
    ["delta", "gamma", "vega", "theta", "rho"],
    default=["delta", "gamma", "vega"],
)

fig_greeks = go.Figure()

for greek in selected_greeks:
    fig_greeks.add_trace(
        go.Scatter(
            x=greeks_curve_df["stock_price"],
            y=greeks_curve_df[greek],
            mode="lines",
            name=greek.capitalize(),
        )
    )

fig_greeks.add_vline(
    x=K,
    line_dash="dash",
    annotation_text="Strike K",
    annotation_position="top",
)

fig_greeks.add_vline(
    x=S,
    line_dash="dot",
    annotation_text="Current S",
    annotation_position="bottom",
)

fig_greeks.update_layout(
    xaxis_title="Underlying Stock Price",
    yaxis_title="Greek Value",
    hovermode="x unified",
    height=520,
)

st.plotly_chart(fig_greeks, use_container_width=True)


# =========================
# Heatmap
# =========================

st.subheader(f"{option_type} Price Heatmap")

stock_grid = np.linspace(max(0.01, S * 0.5), S * 1.5, 40)
vol_grid = np.linspace(0.01, max(0.8, sigma * 2), 40)

heatmap_values = []

for vol in vol_grid:
    row = []
    for stock_price in stock_grid:
        temp_model = BlackScholesModel(
            S=stock_price,
            K=K,
            T=T,
            r=r,
            sigma=vol,
        )

        if option_type_lower == "call":
            row.append(temp_model.call_price())
        else:
            row.append(temp_model.put_price())

    heatmap_values.append(row)

fig_heatmap = go.Figure(
    data=go.Heatmap(
        z=heatmap_values,
        x=stock_grid,
        y=vol_grid * 100,
        colorbar=dict(title=f"{option_type} Price"),
    )
)

fig_heatmap.update_layout(
    xaxis_title="Stock Price",
    yaxis_title="Volatility (%)",
    height=550,
)

st.plotly_chart(fig_heatmap, use_container_width=True)


# =========================
# Explanation section
# =========================

st.subheader("Model Explanation")

with st.expander("Black-Scholes Formula"):
    st.markdown(
        r"""
For a European call option:

$$
C = S N(d_1) - K e^{-rT} N(d_2)
$$

For a European put option:

$$
P = K e^{-rT} N(-d_2) - S N(-d_1)
$$

where:

$$
d_1 = \frac{\ln(S/K) + (r + \frac{1}{2}\sigma^2)T}{\sigma \sqrt{T}}
$$

$$
d_2 = d_1 - \sigma \sqrt{T}
$$

Variable meanings:

- \(S\): current stock price  
- \(K\): strike price  
- \(T\): time to maturity  
- \(r\): risk-free rate  
- \(\sigma\): volatility  
- \(N(\cdot)\): cumulative normal distribution  
"""
    )

with st.expander("Role in this project"):
    st.markdown(
        """
In this project, the Black-Scholes model is not only a pricing calculator.
It works as the **classical baseline** for later hedging experiments.

In Milestone 6, the dashboard uses Black-Scholes Greeks, especially Delta, to construct
a dynamic delta hedging simulator. This allows us to compare theoretical option pricing
with practical hedging performance under simulated market paths.
"""
    )

with st.expander("Assumptions and Limitations"):
    st.markdown(
        """
Main assumptions:

1. The option is European and can only be exercised at maturity.
2. The underlying asset follows a lognormal price process.
3. Volatility is constant.
4. The risk-free rate is constant.
5. There are no transaction costs or taxes.
6. Markets are frictionless.
7. No arbitrage opportunities exist.

Limitations:

- Real markets have changing volatility.
- Transaction costs and liquidity constraints exist.
- Interest rates may be negative or time-varying.
- The model is less suitable for American options.
- It does not capture volatility smile or volatility skew.
"""
    )