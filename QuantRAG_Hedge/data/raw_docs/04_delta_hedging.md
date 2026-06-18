# 04 Delta Hedging

## 1. What is delta hedging?

**Delta hedging** is a strategy that tries to reduce or eliminate the directional risk of an option position by trading the underlying asset.

Delta measures the sensitivity of the option value to the underlying price:

\[
\Delta = \frac{\partial V}{\partial S}
\]

If an option position has Delta exposure, a trader can take an opposite position in the underlying asset to make the total portfolio approximately delta-neutral.

## 2. Delta-neutral portfolio

A portfolio is **delta-neutral** if its total Delta is zero:

\[
\Delta_{portfolio} = 0
\]

Suppose a trader is short one call option. The call has Delta \(\Delta_C\). The short call position has Delta:

\[
-\Delta_C
\]

To hedge this short call, the trader buys \(\Delta_C\) shares of the underlying stock. The stock position has Delta:

\[
+\Delta_C
\]

The total Delta is:

\[
-\Delta_C + \Delta_C = 0
\]

Intuition:

If the stock price goes up slightly, the short call loses money, but the long stock position gains money. If the stock price goes down slightly, the short call gains money, but the long stock position loses money.

## 3. Example: hedging a short call

Suppose a trader sells one European call option with:

- Option Delta \(\Delta_C = 0.60\)
- One option represents one share for simplicity

The short call has Delta:

\[
-0.60
\]

To delta hedge, the trader buys:

\[
0.60 \text{ shares}
\]

The total Delta becomes:

\[
-0.60 + 0.60 = 0
\]

If the stock price increases by 1 dollar, then approximately:

- short call loses \(0.60\),
- stock hedge gains \(0.60\),
- net first-order change is approximately 0.

## 4. Rebalancing

Delta is not constant. It changes when:

1. the stock price changes,
2. time passes,
3. volatility changes,
4. interest rates change.

Therefore, a delta hedge must be updated repeatedly. This is called **rebalancing**.

If the new option Delta becomes \(0.70\), the trader must increase the stock hedge from \(0.60\) shares to \(0.70\) shares.

The required trade is:

\[
0.70 - 0.60 = 0.10 \text{ shares}
\]

## 5. Connection to Gamma

Gamma measures how fast Delta changes as the stock price changes:

\[
\Gamma = \frac{\partial \Delta}{\partial S}
\]

If Gamma is high, Delta changes quickly. This makes delta hedging more difficult because the hedge becomes outdated quickly.

The Delta-Gamma approximation is:

\[
\Delta V \approx \Delta \Delta S + \frac{1}{2}\Gamma(\Delta S)^2
\]

A delta hedge removes the first-order term, but the second-order Gamma term remains.

This is why a delta-neutral portfolio can still gain or lose money when the underlying price moves significantly.

## 6. Ideal continuous-time delta hedging

In the Black-Scholes model, if markets are frictionless and trading is continuous, a trader can theoretically replicate an option payoff by continuously rebalancing a portfolio of stock and cash.

For an option value \(V(S,t)\), the replicating portfolio holds:

\[
\Delta_t = \frac{\partial V}{\partial S}
\]

shares of the underlying asset.

The cash account is adjusted so that the portfolio value matches the option value.

Intuition:

The option can be dynamically replicated because the random stock price risk is canceled by holding the correct number of shares. This is the foundation of the Black-Scholes PDE.

## 7. Discrete-time hedging

In real markets, traders cannot rebalance continuously. Hedging is done at discrete times, such as every minute, hour, day, or week.

Discrete hedging creates **hedging error** because the stock price may move between rebalancing times.

The hedging error can be affected by:

1. rebalancing frequency,
2. volatility,
3. Gamma,
4. transaction costs,
5. jumps in the underlying price,
6. model error in estimating Delta.

## 8. Transaction costs

In textbook delta hedging, transaction costs are usually ignored. In practice, every rebalancing trade may incur costs such as:

- bid-ask spread,
- brokerage fees,
- market impact,
- slippage.

If a trader rebalances too frequently, transaction costs can become large. If the trader rebalances too rarely, hedging error can become large.

This creates a trade-off:

\[
\text{Total cost} = \text{Hedging error} + \text{Transaction cost}
\]

A practical hedging strategy must balance both terms.

## 9. Example: discrete rebalancing

Suppose a trader is short one call option.

At time \(t_0\):

- Call Delta = 0.50
- Trader buys 0.50 shares

At time \(t_1\):

- Stock price increases
- Call Delta becomes 0.65
- Trader buys additional 0.15 shares

At time \(t_2\):

- Stock price decreases
- Call Delta becomes 0.40
- Trader sells 0.25 shares

This repeated buying and selling keeps the portfolio closer to delta-neutral, but it creates transaction costs.

## 10. Assumptions of classical delta hedging

Classical delta hedging assumes:

1. The option pricing model is correct.
2. Delta can be computed accurately.
3. The underlying asset can be traded freely.
4. Trading can occur continuously or very frequently.
5. There are no transaction costs.
6. There are no liquidity constraints.
7. The market has no jumps.
8. The risk-free rate and volatility are known or estimated accurately.

These assumptions are strong. Real-world hedging often violates several of them.

## 11. Why delta hedging is not perfect

Delta hedging is not perfect because:

1. Delta changes over time.
2. Trading is discrete.
3. Transaction costs exist.
4. Volatility is uncertain.
5. The pricing model may be wrong.
6. Asset prices may jump.
7. Liquidity may be limited.

Therefore, delta hedging should be seen as risk reduction, not risk elimination.

## 12. Link to reinforcement learning hedging

Classical delta hedging gives a formula-based strategy:

\[
a_t = \Delta_t
\]

where \(a_t\) is the number of shares to hold at time \(t\).

Reinforcement learning hedging treats hedging as a sequential decision-making problem. Instead of following the Black-Scholes Delta exactly, an RL agent learns a trading policy that may account for:

- transaction costs,
- discrete rebalancing,
- risk preferences,
- market frictions,
- model misspecification.

This is why delta hedging is a natural baseline for an RL hedging project.

## 13. Retrieval questions for RAG testing

1. What does it mean for a portfolio to be delta-neutral?
2. How do you hedge a short call option using Delta?
3. Why does delta hedging require rebalancing?
4. Why does Gamma make delta hedging harder?
5. What are the main sources of hedging error?
6. Why is delta hedging an important baseline for reinforcement learning hedging?
