# 03 Greeks

## 1. What are Greeks?

**Greeks** measure how sensitive an option price is to changes in different input variables.

If the option price is written as:

\[
V = V(S, t, \sigma, r)
\]

then Greeks are partial derivatives of \(V\) with respect to these variables.

They are called Greeks because many of them are denoted by Greek letters, such as Delta \(\Delta\), Gamma \(\Gamma\), Theta \(\Theta\), Vega, and Rho \(\rho\).

Greeks are essential for risk management because they tell traders how an option portfolio reacts to market changes.

## 2. Delta

Delta measures the sensitivity of an option price to the underlying asset price:

\[
\Delta = \frac{\partial V}{\partial S}
\]

Intuition:

Delta tells us approximately how much the option price changes when the stock price increases by 1 unit.

Approximation:

\[
\Delta V \approx \Delta \cdot \Delta S
\]

For a European call option under Black-Scholes with no dividends:

\[
\Delta_{call} = N(d_1)
\]

For a European put option:

\[
\Delta_{put} = N(d_1) - 1
\]

Typical ranges:

| Option type | Delta range |
|---|---:|
| Call | 0 to 1 |
| Put | -1 to 0 |

Example:

If a call option has \(\Delta = 0.60\), then a 1 dollar increase in the stock price approximately increases the call price by 0.60 dollars.

## 3. Gamma

Gamma measures how Delta changes when the underlying asset price changes:

\[
\Gamma = \frac{\partial^2 V}{\partial S^2}
\]

Equivalently:

\[
\Gamma = \frac{\partial \Delta}{\partial S}
\]

Under Black-Scholes, call and put options have the same Gamma:

\[
\Gamma = \frac{N'(d_1)}{S\sigma\sqrt{T}}
\]

where \(N'(d_1)\) is the standard normal probability density function.

Intuition:

Gamma tells us how unstable the Delta is. If Gamma is high, the hedge ratio changes quickly as the stock price moves.

This matters because delta hedging assumes that we can rebalance the hedge. High Gamma means we may need to rebalance more often.

## 4. Theta

Theta measures the sensitivity of an option price to the passage of time:

\[
\Theta = \frac{\partial V}{\partial t}
\]

In many trading contexts, Theta is interpreted as the option's time decay.

Intuition:

Options usually lose time value as maturity approaches. All else equal, a shorter time to maturity gives the option less opportunity to move into the money.

For a non-dividend-paying European call under Black-Scholes:

\[
\Theta_{call} = -\frac{S N'(d_1)\sigma}{2\sqrt{T}} - rKe^{-rT}N(d_2)
\]

For a European put:

\[
\Theta_{put} = -\frac{S N'(d_1)\sigma}{2\sqrt{T}} + rKe^{-rT}N(-d_2)
\]

Note:

Different sources may define Theta with respect to calendar time \(t\) or time to maturity \(T\). The sign convention should always be checked.

## 5. Vega

Vega measures the sensitivity of an option price to volatility:

\[
\text{Vega} = \frac{\partial V}{\partial \sigma}
\]

Under Black-Scholes, call and put options have the same Vega:

\[
\text{Vega} = S N'(d_1)\sqrt{T}
\]

Intuition:

Higher volatility increases the probability of large favorable movements. Since option buyers have limited downside but benefit from favorable extremes, higher volatility generally increases option value.

Example:

If an option has Vega \(= 25\), then a 0.01 increase in volatility approximately increases the option price by:

\[
25 \times 0.01 = 0.25
\]

## 6. Rho

Rho measures the sensitivity of an option price to the risk-free interest rate:

\[
\rho = \frac{\partial V}{\partial r}
\]

For a European call:

\[
\rho_{call} = KT e^{-rT}N(d_2)
\]

For a European put:

\[
\rho_{put} = -KT e^{-rT}N(-d_2)
\]

Intuition:

A higher interest rate reduces the present value of the strike price. This tends to increase call prices and decrease put prices.

## 7. Greek summary table

| Greek | Formula idea | Measures sensitivity to | Main risk meaning |
|---|---|---|---|
| Delta | \(\partial V/\partial S\) | Stock price | Directional exposure |
| Gamma | \(\partial^2 V/\partial S^2\) | Stock price curvature | Delta instability |
| Theta | \(\partial V/\partial t\) | Time | Time decay |
| Vega | \(\partial V/\partial \sigma\) | Volatility | Volatility exposure |
| Rho | \(\partial V/\partial r\) | Interest rate | Rate exposure |

## 8. Assumptions behind Black-Scholes Greeks

The closed-form Greek formulas above assume the Black-Scholes setting:

1. The option is European.
2. The underlying asset follows geometric Brownian motion.
3. Volatility is constant.
4. The risk-free rate is constant.
5. Trading is continuous.
6. There are no transaction costs.
7. The stock pays no dividends.

If these assumptions fail, the formulas may still be useful approximations, but they may not fully capture real market risk.

## 9. Example: Delta approximation

Suppose a call option has:

- Current price \(C = 10\)
- Delta \(\Delta = 0.55\)
- Stock price change \(\Delta S = +2\)

The approximate option price change is:

\[
\Delta C \approx 0.55 \times 2 = 1.10
\]

So the new option price is approximately:

\[
C_{new} \approx 10 + 1.10 = 11.10
\]

This is only a first-order approximation. If the stock movement is large, Gamma becomes important.

## 10. Example: Delta-Gamma approximation

A better approximation includes Gamma:

\[
\Delta V \approx \Delta \Delta S + \frac{1}{2}\Gamma(\Delta S)^2
\]

Suppose:

- \(\Delta = 0.55\)
- \(\Gamma = 0.04\)
- \(\Delta S = 2\)

Then:

\[
\Delta V \approx 0.55(2) + \frac{1}{2}(0.04)(2^2)
\]

\[
\Delta V \approx 1.10 + 0.08 = 1.18
\]

The Gamma term adjusts for the curvature of the option price.

## 11. Why Greeks matter for hedging

Delta is the starting point for hedging because it measures directional exposure. If a trader is short one call option with Delta \(0.60\), they can hedge the position by buying approximately \(0.60\) shares of the underlying stock.

However, Delta changes over time and with stock price movement. Gamma explains why the hedge must be updated. Vega explains why a portfolio can lose money even if Delta is hedged, because implied volatility may change. Theta explains why time decay affects the value of the option position.

A complete risk management system usually monitors multiple Greeks, not Delta alone.

## 12. Retrieval questions for RAG testing

1. What does Delta measure?
2. What is the Black-Scholes Delta of a European call?
3. What does Gamma measure?
4. Why does high Gamma make hedging harder?
5. What does Vega measure?
6. Why can a delta-hedged portfolio still lose money?
