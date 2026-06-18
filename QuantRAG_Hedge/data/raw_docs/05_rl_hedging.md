# 05 Reinforcement Learning for Hedging

## 1. Why use reinforcement learning for hedging?

Classical delta hedging is based on mathematical formulas from models such as Black-Scholes. It works well under ideal assumptions, including continuous trading, no transaction costs, constant volatility, and correct model specification.

Real markets are different. Trading is discrete, transaction costs exist, volatility changes, and model assumptions may fail.

**Reinforcement learning (RL)** is useful because hedging is naturally a sequential decision-making problem. At each time step, the hedger observes the market and portfolio state, chooses a trading action, receives a cost or reward, and then moves to the next time step.

## 2. Hedging as a Markov decision process

A reinforcement learning problem is often modeled as a **Markov decision process (MDP)**:

\[
(\mathcal{S}, \mathcal{A}, P, R, \gamma)
\]

where:

| Component | Meaning in hedging |
|---|---|
| \(\mathcal{S}\) | State space: market and portfolio information |
| \(\mathcal{A}\) | Action space: trading decisions |
| \(P\) | Transition dynamics of market states |
| \(R\) | Reward or negative cost function |
| \(\gamma\) | Discount factor |

The goal is to learn a policy:

\[
\pi(a_t \mid s_t)
\]

that chooses an action \(a_t\) given the current state \(s_t\).

## 3. State design for RL hedging

A state should include information that helps the agent decide how to hedge.

Possible state variables include:

1. Current underlying price \(S_t\)
2. Strike price \(K\)
3. Time to maturity \(\tau = T-t\)
4. Current option value \(V_t\)
5. Current Delta \(\Delta_t\)
6. Current hedge position \(h_t\)
7. Previous action \(a_{t-1}\)
8. Cash account or portfolio value
9. Realized volatility estimate
10. Implied volatility estimate, if available

A compact example state is:

\[
s_t = \left(\frac{S_t}{K}, \tau, \Delta_t, h_t\right)
\]

Intuition:

- \(S_t/K\) describes moneyness.
- \(\tau\) describes how much time remains.
- \(\Delta_t\) gives the classical hedge target.
- \(h_t\) tells the agent its current hedge position.

## 4. Action design

The action can represent either the target hedge position or the trade size.

### Target-position action

The agent chooses the number of shares to hold:

\[
a_t = h_t
\]

For example, \(a_t = 0.60\) means the agent wants to hold 0.60 shares.

### Trade-size action

The agent chooses how much to buy or sell:

\[
a_t = h_t - h_{t-1}
\]

For example, \(a_t = 0.10\) means the agent buys 0.10 additional shares.

In practice, the action space can be continuous or discrete.

## 5. Portfolio dynamics

Suppose the hedger holds \(h_t\) shares of the underlying asset at time \(t\). The portfolio value can be written as:

\[
\Pi_t = h_t S_t + B_t - V_t
\]

where:

- \(h_tS_t\) is the value of the stock hedge,
- \(B_t\) is the cash account,
- \(V_t\) is the option liability value if the hedger is short the option.

A self-financing update without transaction costs is approximately:

\[
B_{t+1} = B_t e^{r\Delta t} - (h_{t+1}-h_t)S_t
\]

With proportional transaction cost \(\lambda\), the update becomes:

\[
B_{t+1} = B_t e^{r\Delta t} - (h_{t+1}-h_t)S_t - \lambda S_t |h_{t+1}-h_t|
\]

Intuition:

Changing the hedge position costs money. The term \(\lambda S_t |h_{t+1}-h_t|\) penalizes frequent or large trades.

## 6. Reward design

The reward should encourage good hedging and discourage unnecessary trading.

A simple reward can be the negative squared hedging error:

\[
r_t = -\left(\Pi_{t+1} - \Pi_t\right)^2
\]

A terminal reward can penalize final replication error:

\[
r_T = -\left(\Pi_T\right)^2
\]

If transaction costs are included:

\[
r_t = -\left(\text{hedging error}_t^2 + \lambda S_t |h_t-h_{t-1}|\right)
\]

Alternative risk-sensitive objectives include:

\[
\mathbb{E}[\text{PnL}] - \eta \text{Var}(\text{PnL})
\]

or minimizing conditional value at risk:

\[
\text{CVaR}_{\alpha}(\text{loss})
\]

Intuition:

The reward function determines what the agent learns. If we only penalize hedging error, the agent may trade too much. If we include transaction costs, the agent learns to balance risk reduction and trading cost.

## 7. Policy objective

The RL agent tries to maximize expected cumulative reward:

\[
J(\pi) = \mathbb{E}_{\pi}\left[\sum_{t=0}^{T} \gamma^t r_t\right]
\]

In episodic hedging, one episode can represent one simulated option life from initial time to maturity.

The learned policy is:

\[
\pi^* = \arg\max_{\pi} J(\pi)
\]

## 8. Baseline: Black-Scholes delta hedging

A natural baseline policy is the Black-Scholes delta hedge:

\[
h_t = \Delta_t^{BS}
\]

For a short European call, this means holding:

\[
h_t = N(d_1)
\]

shares of the underlying asset.

The RL policy can be compared against this baseline using metrics such as:

1. mean hedging error,
2. standard deviation of PnL,
3. mean transaction cost,
4. tail loss,
5. CVaR,
6. final replication error.

## 9. Example: simple RL hedging environment

Consider a simplified environment:

- One European call option
- \(S_0 = 100\)
- \(K = 100\)
- \(T = 30\) days
- Daily rebalancing
- Proportional transaction cost \(\lambda = 0.001\)

At each day \(t\):

1. The agent observes \(s_t = (S_t/K, \tau, \Delta_t, h_t)\).
2. The agent chooses a new hedge position \(h_{t+1}\).
3. The environment updates the stock price.
4. The portfolio value and transaction cost are computed.
5. The agent receives a reward based on hedging error and trading cost.

A good RL agent should learn not only to follow Delta, but also to avoid excessive rebalancing when the benefit is small relative to transaction cost.

## 10. Assumptions in RL hedging experiments

An RL hedging project usually makes assumptions such as:

1. Market data can be simulated or sampled from historical data.
2. The training environment is representative of the test environment.
3. The state contains enough information for good decisions.
4. The reward function correctly reflects the hedging objective.
5. Transaction costs can be modeled explicitly.
6. The action constraints are realistic.
7. The evaluation metrics reflect practical hedging performance.

These assumptions should be stated clearly because RL models can overfit to the simulated environment.

## 11. Advantages of RL hedging

RL hedging can be useful because it can:

1. handle transaction costs directly,
2. learn under discrete rebalancing,
3. optimize non-standard objectives,
4. adapt to simulated market frictions,
5. learn policies without requiring a closed-form solution.

## 12. Limitations of RL hedging

RL hedging also has important limitations:

1. It requires many training episodes.
2. It may overfit to the simulator.
3. It can be unstable to train.
4. The learned policy may be hard to interpret.
5. Poor reward design can produce undesirable behavior.
6. Real market regime changes can reduce performance.

Therefore, RL hedging should be compared carefully against strong baselines such as Black-Scholes delta hedging and no-transaction bands.

## 13. Evaluation metrics

Useful evaluation metrics include:

### Mean final hedging error

\[
\frac{1}{N}\sum_{i=1}^{N} \Pi_T^{(i)}
\]

### Mean squared final hedging error

\[
\frac{1}{N}\sum_{i=1}^{N} \left(\Pi_T^{(i)}\right)^2
\]

### PnL volatility

\[
\sqrt{\text{Var}(\text{PnL})}
\]

### Average transaction cost

\[
\frac{1}{N}\sum_{i=1}^{N}\sum_{t=0}^{T-1}\lambda S_t^{(i)}|h_{t+1}^{(i)}-h_t^{(i)}|
\]

### CVaR of loss

\[
\text{CVaR}_{\alpha}(L) = \mathbb{E}[L \mid L \ge \text{VaR}_{\alpha}(L)]
\]

where \(L\) is the loss variable.

## 14. Link to the QuantRAG Hedge project

In a QuantRAG Hedge project, the knowledge base can help answer conceptual and mathematical questions such as:

- What is Delta?
- Why does Gamma matter for hedging?
- What is the Black-Scholes hedge ratio?
- Why does transaction cost affect rebalancing frequency?
- How can RL formulate hedging as an MDP?

The RAG system should retrieve relevant chunks from options theory, Greeks, delta hedging, and RL hedging documents before generating answers.

## 15. Retrieval questions for RAG testing

1. How can hedging be formulated as a Markov decision process?
2. What can be included in the RL hedging state?
3. What is a transaction-cost-aware reward function?
4. Why is Black-Scholes delta hedging a baseline for RL hedging?
5. What metrics can evaluate an RL hedging strategy?
6. What are the limitations of reinforcement learning for hedging?
