# 01 Options Basics

## 1. What is an option?

An **option** is a financial derivative that gives its holder the right, but not the obligation, to buy or sell an underlying asset at a fixed price before or at a specific future date.

The underlying asset can be a stock, index, ETF, currency, commodity, or another tradable instrument.

There are two basic types of options:

- **Call option**: gives the holder the right to buy the underlying asset.
- **Put option**: gives the holder the right to sell the underlying asset.

The fixed price is called the **strike price**, usually denoted by \(K\). The final date is called the **maturity** or **expiration date**, denoted by \(T\). The underlying asset price at time \(t\) is denoted by \(S_t\).

## 2. Basic notation

| Symbol | Meaning |
|---|---|
| \(S_t\) | Underlying asset price at time \(t\) |
| \(K\) | Strike price |
| \(T\) | Maturity time |
| \(r\) | Risk-free interest rate |
| \(\sigma\) | Volatility of the underlying asset |
| \(C\) | Call option price |
| \(P\) | Put option price |

## 3. Payoff of a European call option

A European call option can only be exercised at maturity. Its payoff is:

\[
\text{Call payoff} = \max(S_T - K, 0)
\]

Intuition:

- If \(S_T > K\), the holder can buy the asset at \(K\) and immediately own something worth \(S_T\). The payoff is \(S_T - K\).
- If \(S_T \le K\), exercising the option is not useful. The payoff is 0.

Example:

Suppose \(K = 100\).

| Final stock price \(S_T\) | Call payoff \(\max(S_T-K,0)\) |
|---:|---:|
| 80 | 0 |
| 100 | 0 |
| 120 | 20 |
| 150 | 50 |

## 4. Payoff of a European put option

A European put option can only be exercised at maturity. Its payoff is:

\[
\text{Put payoff} = \max(K - S_T, 0)
\]

Intuition:

- If \(S_T < K\), the holder can sell the asset for \(K\), even though the market price is only \(S_T\). The payoff is \(K - S_T\).
- If \(S_T \ge K\), exercising the put is not useful. The payoff is 0.

Example:

Suppose \(K = 100\).

| Final stock price \(S_T\) | Put payoff \(\max(K-S_T,0)\) |
|---:|---:|
| 80 | 20 |
| 100 | 0 |
| 120 | 0 |
| 150 | 0 |

## 5. Moneyness

**Moneyness** describes the relationship between the current asset price \(S_t\) and the strike price \(K\).

For a call option:

- **In the money (ITM)**: \(S_t > K\)
- **At the money (ATM)**: \(S_t \approx K\)
- **Out of the money (OTM)**: \(S_t < K\)

For a put option:

- **In the money (ITM)**: \(S_t < K\)
- **At the money (ATM)**: \(S_t \approx K\)
- **Out of the money (OTM)**: \(S_t > K\)

Intuition:

Moneyness tells us whether the option would have positive exercise value if it were exercised immediately. However, option price is not only payoff. Before maturity, an option also has **time value**, because the underlying price may move favorably in the future.

## 6. Intrinsic value and time value

The price of an option can be decomposed conceptually into:

\[
\text{Option price} = \text{Intrinsic value} + \text{Time value}
\]

For a call:

\[
\text{Intrinsic value} = \max(S_t - K, 0)
\]

For a put:

\[
\text{Intrinsic value} = \max(K - S_t, 0)
\]

Time value reflects the possibility that the option becomes more valuable before expiration.

Example:

Suppose a call option has \(S_t = 105\), \(K = 100\), and market price \(C = 8\).

The intrinsic value is:

\[
\max(105 - 100, 0) = 5
\]

The time value is:

\[
8 - 5 = 3
\]

## 7. European vs American options

A **European option** can only be exercised at maturity.

An **American option** can be exercised at any time before or at maturity.

Black-Scholes pricing in its basic form is designed for European options. American options are harder to price because early exercise may be optimal, especially for dividend-paying stocks or deep in-the-money puts.

## 8. Long and short positions

Buying an option creates a **long option** position. Selling or writing an option creates a **short option** position.

For a long call:

\[
\text{Profit} = \max(S_T-K,0) - C_0
\]

For a short call:

\[
\text{Profit} = C_0 - \max(S_T-K,0)
\]

where \(C_0\) is the premium paid at the start.

Intuition:

- The option buyer pays premium upfront and receives nonlinear upside.
- The option seller receives premium upfront but takes on the obligation to pay the option payoff.

## 9. Put-call parity

For European options on a non-dividend-paying stock, put-call parity is:

\[
C - P = S_0 - K e^{-rT}
\]

Equivalently:

\[
C + K e^{-rT} = P + S_0
\]

Intuition:

A portfolio of one call plus cash equal to the present value of the strike has the same final payoff as a portfolio of one put plus one share of stock.

At maturity:

\[
\max(S_T-K,0)+K = \max(K-S_T,0)+S_T
\]

Both sides are equal to \(\max(S_T,K)\).

## 10. Main assumptions in basic option theory

Basic option pricing usually assumes:

1. The underlying asset can be traded continuously.
2. There are no transaction costs.
3. There are no taxes or liquidity constraints.
4. Investors can borrow and lend at the risk-free rate.
5. There is no arbitrage.
6. The option contract terms are fixed and known.

These assumptions are idealized. Real markets have transaction costs, bid-ask spreads, discrete trading, market impact, and liquidity constraints. This is one reason why hedging in practice can differ from textbook formulas.

## 11. Why options matter for hedging

Options are useful because they create nonlinear exposure. The payoff is not a straight line like holding a stock. This makes them powerful for risk management but also difficult to hedge.

For example, a call option becomes more sensitive to the stock price as the stock price increases. A put option becomes more sensitive as the stock price falls. This changing sensitivity motivates the study of **Greeks**, especially **Delta**, and the design of hedging strategies such as **delta hedging**.

## 12. Mini example: option payoff and profit

Suppose an investor buys a European call with:

- \(S_0 = 100\)
- \(K = 100\)
- Premium \(C_0 = 6\)

At maturity, if \(S_T = 115\):

\[
\text{Payoff} = \max(115-100,0)=15
\]

\[
\text{Profit} = 15 - 6 = 9
\]

If \(S_T = 95\):

\[
\text{Payoff} = \max(95-100,0)=0
\]

\[
\text{Profit} = 0 - 6 = -6
\]

The maximum loss for the long call buyer is the premium paid, but the upside can be large if the stock price rises significantly.

## 13. Retrieval questions for RAG testing

1. What is the payoff of a European call option?
2. What is the payoff of a European put option?
3. What is the difference between intrinsic value and time value?
4. What does it mean for a call option to be in the money?
5. What is put-call parity?
6. Why are options harder to hedge than stocks?
