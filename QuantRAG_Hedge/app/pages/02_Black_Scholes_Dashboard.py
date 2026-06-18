import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Allow importing from src/
ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
sys.path.append(str(SRC_DIR))

from src.black_scholes import BlackScholesModel


st.set_page_config(
    page_title="Black-Scholes Dashboard",
    layout="wide"
)


st.title("Milestone 5: Black-Scholes Option Pricing Dashboard")

st.markdown(
    """
This dashboard implements the **Black-Scholes model** for European call and put options.
It allows users to interactively change market assumptions and observe how option prices,
payoff profiles, and Greeks respond.
"""
)


# =========================
# Sidebar inputs
# =========================

st.sidebar.header("Model Inputs")

S = st.sidebar.number_input(
    "Current Stock Price (S)",
    min_value=1.0,
    max_value=1000.0,
    value=100.0,
    step=1.0
)

K = st.sidebar.number_input(
    "Strike Price (K)",
    min_value=1.0,
    max_value=1000.0,
    value=100.0,
    step=1.0
)

T = st.sidebar.slider(
    "Time to Maturity (T, years)",
    min_value=0.01,
    max_value=5.0,
    value=1.0,
    step=0.01
)

r_percent = st.sidebar.slider(
    "Risk-free Rate (%)",
    min_value=0.0,
    max_value=20.0,
    value=5.0,
    step=0.1
)

sigma_percent = st.sidebar.slider(
    "Volatility (%)",
    min_value=1.0,
    max_value=100.0,
    value=20.0,
    step=0.5
)

r = r_percent / 100
sigma = sigma_percent / 100


# =========================
# Model calculation
# =========================

model = BlackScholesModel(S=S, K=K, T=T, r=r, sigma=sigma)
summary = model.summary()

call_price = summary["call_price"]
put_price = summary["put_price"]


# =========================
# Main KPI cards
# =========================

st.subheader("Option Price Summary")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Call Price", f"${call_price:,.4f}")

with col2:
    st.metric("Put Price", f"${put_price:,.4f}")

with col3:
    st.metric("d1", f"{summary['d1']:.4f}")

with col4:
    st.metric("d2", f"{summary['d2']:.4f}")


# =========================
# Greeks table
# =========================

st.subheader("Greeks")

greeks_df = pd.DataFrame(
    {
        "Greek": ["Delta", "Gamma", "Vega", "Theta", "Rho"],
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
        "Interpretation": [
            "Sensitivity to stock price movement",
            "Sensitivity of Delta to stock price movement",
            "Sensitivity to volatility change",
            "Sensitivity to time decay per day",
            "Sensitivity to interest rate change",
        ],
    }
)

st.dataframe(
    greeks_df.style.format(
        {
            "Call": "{:.6f}",
            "Put": "{:.6f}",
        }
    ),
    use_container_width=True
)


# =========================
# Payoff chart
# =========================

st.subheader("Payoff Diagram at Expiration")

S_range = np.linspace(max(1, S * 0.4), S * 1.8, 200)

call_payoff = np.maximum(S_range - K, 0)
put_payoff = np.maximum(K - S_range, 0)

call_profit = call_payoff - call_price
put_profit = put_payoff - put_price

fig_payoff = go.Figure()

fig_payoff.add_trace(
    go.Scatter(
        x=S_range,
        y=call_profit,
        mode="lines",
        name="Call Profit",
    )
)

fig_payoff.add_trace(
    go.Scatter(
        x=S_range,
        y=put_profit,
        mode="lines",
        name="Put Profit",
    )
)

fig_payoff.add_hline(
    y=0,
    line_dash="dash",
    annotation_text="Break-even line",
    annotation_position="bottom right"
)

fig_payoff.add_vline(
    x=K,
    line_dash="dash",
    annotation_text="Strike Price",
    annotation_position="top"
)

fig_payoff.update_layout(
    xaxis_title="Stock Price at Expiration",
    yaxis_title="Profit / Loss",
    hovermode="x unified",
    height=500
)

st.plotly_chart(fig_payoff, use_container_width=True)


# =========================
# Option price sensitivity
# =========================

st.subheader("Option Price Sensitivity to Stock Price")

call_prices = []
put_prices = []

for price in S_range:
    temp_model = BlackScholesModel(
        S=price,
        K=K,
        T=T,
        r=r,
        sigma=sigma
    )
    call_prices.append(temp_model.call_price())
    put_prices.append(temp_model.put_price())

fig_price = go.Figure()

fig_price.add_trace(
    go.Scatter(
        x=S_range,
        y=call_prices,
        mode="lines",
        name="Call Price"
    )
)

fig_price.add_trace(
    go.Scatter(
        x=S_range,
        y=put_prices,
        mode="lines",
        name="Put Price"
    )
)

fig_price.add_vline(
    x=S,
    line_dash="dash",
    annotation_text="Current S",
    annotation_position="top"
)

fig_price.add_vline(
    x=K,
    line_dash="dot",
    annotation_text="Strike K",
    annotation_position="bottom"
)

fig_price.update_layout(
    xaxis_title="Underlying Stock Price",
    yaxis_title="Option Price",
    hovermode="x unified",
    height=500
)

st.plotly_chart(fig_price, use_container_width=True)


# =========================
# Heatmap
# =========================

st.subheader("Call Price Heatmap")

stock_grid = np.linspace(max(1, S * 0.5), S * 1.5, 40)
vol_grid = np.linspace(0.05, 0.8, 40)

heatmap_values = []

for vol in vol_grid:
    row = []
    for stock_price in stock_grid:
        temp_model = BlackScholesModel(
            S=stock_price,
            K=K,
            T=T,
            r=r,
            sigma=vol
        )
        row.append(temp_model.call_price())
    heatmap_values.append(row)

fig_heatmap = go.Figure(
    data=go.Heatmap(
        z=heatmap_values,
        x=stock_grid,
        y=vol_grid * 100,
        colorbar=dict(title="Call Price")
    )
)

fig_heatmap.update_layout(
    xaxis_title="Stock Price",
    yaxis_title="Volatility (%)",
    height=550
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

with st.expander("Financial Intuition"):
    st.markdown(
        """
The Black-Scholes model estimates the fair value of a European option under a set of assumptions.

A **call option** becomes more valuable when the stock price increases because the holder has the right to buy the asset at the fixed strike price.

A **put option** becomes more valuable when the stock price decreases because the holder has the right to sell the asset at the fixed strike price.

Higher volatility usually increases both call and put values because more uncertainty creates more upside potential for option holders.

Time to maturity also matters. More time gives the asset more opportunity to move, which usually increases option value.
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
- Interest rates are not always constant.
- The model is less suitable for American options.
- It does not fully capture volatility smile or market microstructure effects.
"""
    )