# MDP Formulation for Option Hedging

- State: moneyness, time to maturity, current hedge position
- Action: discrete hedge ratio or hedge adjustment
- Transition: GBM stock-price movement
- Reward: negative hedging error minus transaction cost
- Policy: mapping from market state to hedge action

# Q-learning Update

Q(s,a) <- Q(s,a) + alpha [r + gamma max_a' Q(s',a') - Q(s,a)]

# Why Q-learning fits hedging

Hedging is sequential, uncertain, and cost-sensitive.