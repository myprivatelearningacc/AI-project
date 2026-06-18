# 02 Black-Scholes Model

## 1. Purpose of the Black-Scholes model

The **Black-Scholes model** is a classical mathematical model for pricing European call and put options. It gives a closed-form formula for the fair price of an option under idealized market assumptions.

The model is important because it connects option prices to:

- current stock price,
- strike price,
- time to maturity,
- risk-free interest rate,
- volatility.

It also provides the foundation for Greeks and delta hedging.

## 2. Core modeling assumption

The Black-Scholes model assumes that the stock price follows a **geometric Brownian motion**:

\[
dS_t = \mu S_t dt + \sigma S_t dW_t
\]

where:

| Symbol | Meaning |
|---|---|
| \(S_t\) | Stock price at time \(t\) |
| \(\mu\) | Expected return or drift |
| \(\sigma\) | Volatility |
| \(W_t\) | Brownian motion |
| \(dt\) | Small time interval |

Intuition:

The term \(\mu S_t dt\) represents the expected direction of movement. The term \(\sigma S_t dW_t\) represents random market shocks. Because the randomness is proportional to \(S_t\), the model keeps stock prices positive.

## 3. Risk-neutral idea

One surprising result in Black-Scholes pricing is that the expected return \(\mu\) of the stock does not appear in the final option pricing formula.

Under the **risk-neutral measure**, the stock price evolves as:

\[
dS_t = r S_t dt + \sigma S_t dW_t^{\mathbb{Q}}
\]

where \(r\) is the risk-free rate.

Intuition:

In a no-arbitrage market, we can price derivatives by discounting their expected payoff under a risk-neutral probability measure. This does not mean investors are actually risk-neutral. It is a mathematical pricing tool.

## 4. Black-Scholes formula for a European call

For a non-dividend-paying stock, the Black-Scholes price of a European call option is:

\[
C = S_0 N(d_1) - K e^{-rT} N(d_2)
\]

where:

\[
d_1 = \frac{\ln(S_0/K) + \left(r + \frac{1}{2}\sigma^2\right)T}{\sigma\sqrt{T}}
\]

\[
d_2 = d_1 - \sigma\sqrt{T}
\]

and \(N(x)\) is the cumulative distribution function of the standard normal distribution.

## 5. Black-Scholes formula for a European put

The Black-Scholes price of a European put option is:

\[
P = K e^{-rT} N(-d_2) - S_0 N(-d_1)
\]

This is consistent with put-call parity:

\[
C - P = S_0 - K e^{-rT}
\]

## 6. Interpretation of \(d_1\) and \(d_2\)

The quantities \(d_1\) and \(d_2\) are standardized terms that combine moneyness, interest rate, volatility, and time.

A useful intuition:

- \(N(d_2)\) is related to the risk-neutral probability that the call option finishes in the money.
- \(N(d_1)\) is the call option's Delta in the non-dividend-paying Black-Scholes model.

They are not just arbitrary symbols. They summarize the option's position relative to the strike after adjusting for uncertainty and time.

## 7. Assumptions of the Black-Scholes model

The standard Black-Scholes model assumes:

1. The option is European.
2. The underlying stock pays no dividends.
3. The stock price follows geometric Brownian motion.
4. Volatility \(\sigma\) is constant.
5. The risk-free rate \(r\) is constant.
6. Trading is continuous.
7. There are no transaction costs or taxes.
8. The market is frictionless and liquid.
9. Investors can borrow and lend at the risk-free rate.
10. There are no arbitrage opportunities.

These assumptions simplify the market. In real trading, volatility changes over time, trading is discrete, liquidity is limited, and transaction costs matter.

## 8. Why volatility matters

Volatility measures uncertainty in the future stock price. Higher volatility usually increases both call and put option prices.

Intuition:

- A call option benefits from large upward movements but has limited downside because payoff cannot go below zero.
- A put option benefits from large downward movements but also has limited downside for the buyer.
- Therefore, more uncertainty increases the value of optionality.

## 9. Example: calculating a European call price

Suppose:

- \(S_0 = 100\)
- \(K = 100\)
- \(r = 0.05\)
- \(\sigma = 0.20\)
- \(T = 1\) year

First compute:

\[
d_1 = \frac{\ln(100/100) + (0.05 + 0.5 \times 0.20^2) \times 1}{0.20\sqrt{1}}
\]

\[
d_1 = \frac{0 + (0.05 + 0.02)}{0.20} = 0.35
\]

\[
d_2 = 0.35 - 0.20 = 0.15
\]

Using standard normal values:

\[
N(d_1) \approx N(0.35) \approx 0.6368
\]

\[
N(d_2) \approx N(0.15) \approx 0.5596
\]

The call price is:

\[
C = 100(0.6368) - 100e^{-0.05}(0.5596)
\]

Since \(e^{-0.05} \approx 0.9512\):

\[
C \approx 63.68 - 53.23 = 10.45
\]

So the Black-Scholes call price is approximately:

\[
C \approx 10.45
\]

## 10. Example: calculating a European put price

Using the same inputs, the put price is:

\[
P = 100e^{-0.05}N(-0.15) - 100N(-0.35)
\]

Using:

\[
N(-0.15) \approx 0.4404
\]

\[
N(-0.35) \approx 0.3632
\]

Then:

\[
P \approx 100(0.9512)(0.4404) - 100(0.3632)
\]

\[
P \approx 41.89 - 36.32 = 5.57
\]

So the Black-Scholes put price is approximately:

\[
P \approx 5.57
\]

We can check put-call parity:

\[
C - P \approx 10.45 - 5.57 = 4.88
\]

\[
S_0 - Ke^{-rT} = 100 - 100e^{-0.05} \approx 100 - 95.12 = 4.88
\]

The values are consistent.

## 11. The Black-Scholes PDE

The Black-Scholes option price \(V(S,t)\) satisfies the partial differential equation:

\[
\frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + rS\frac{\partial V}{\partial S} - rV = 0
\]

with terminal condition:

For a call:

\[
V(S,T) = \max(S-K,0)
\]

For a put:

\[
V(S,T) = \max(K-S,0)
\]

Intuition:

The PDE comes from constructing a continuously rebalanced portfolio of the option and the stock so that the random component cancels out. The resulting riskless portfolio must earn the risk-free rate.

## 12. Limitations of Black-Scholes

Black-Scholes is elegant but not fully realistic. Important limitations include:

1. Volatility is not constant in real markets.
2. Asset returns often have fat tails.
3. Trading is discrete, not continuous.
4. Transaction costs and bid-ask spreads exist.
5. Market liquidity is not infinite.
6. Large trades can move prices.
7. Jumps and crashes are not captured by simple geometric Brownian motion.

These limitations motivate more practical hedging methods, including transaction-cost-aware hedging and reinforcement learning hedging.

## 13. Retrieval questions for RAG testing

1. What is the Black-Scholes formula for a European call option?
2. What is the Black-Scholes formula for a European put option?
3. What assumptions does the Black-Scholes model make?
4. Why does volatility increase option value?
5. What is the Black-Scholes PDE?
6. Why does the drift \(\mu\) not appear in the final Black-Scholes option price?
