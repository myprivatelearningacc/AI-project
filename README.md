````markdown
# QuantRAG-Hedge: Tool-Augmented AI Assistant and Reinforcement Learning for Option Hedging

QuantRAG-Hedge is an Artificial Intelligence course project that combines classical quantitative finance, reinforcement learning, and a tool-augmented RAG assistant. The main evaluated AI algorithm is **tabular Q-learning**, applied to the problem of dynamic option hedging under transaction costs and discrete rebalancing.

The project implements Black-Scholes option pricing, Greeks calculation, geometric Brownian motion simulation, delta hedging, a custom MDP-style hedging environment, Q-learning agents, experimental evaluation across multiple seeds, and a Streamlit dashboard for interactive exploration.

---

## Project Overview

Option hedging is a sequential decision-making problem. A hedger observes the market state, chooses a hedge position, pays transaction costs when rebalancing, and aims to minimize the final hedging error at maturity.

This project compares classical and learning-based hedging strategies:

1. **No Hedge**  
   Holds only the option premium and does not rebalance.

2. **Black-Scholes Delta Hedge**  
   Rebalances the hedge position using the analytical Black-Scholes delta.

3. **Q-learning Basic**  
   Learns a discrete hedge ratio from moneyness and time bins.

4. **Q-learning Reward Variation**  
   Uses a modified reward with stronger transaction-cost and position-change penalties.

5. **Q-learning State Variation**  
   Adds hedge-position and recent price-movement information to the tabular state.

The final evaluation uses five random seeds:

```python
[42, 123, 2026, 777, 999]
````

and reports multiple metrics, including mean absolute hedging error, standard deviation of hedging error, 95% CVaR, transaction cost, number of trades, and final reward.

---

## Main Results

The final experiment shows that:

* Black-Scholes delta hedging achieves the lowest hedging error in the simulated Black-Scholes environment.
* Q-learning improves substantially over the no-hedge baseline.
* The reward-variation Q-learning agent slightly improves over the basic Q-learning agent.
* The state-variation agent performs worse because the larger tabular state space is harder to explore under a fixed training budget.
* The learned Q-learning policy heatmap shows interpretable hedging behavior across moneyness and time bins.

Summary of final metrics:

| Algorithm                   |   MAE | Std. Error | CVaR 95% |  Cost | Trades | Reward |
| --------------------------- | ----: | ---------: | -------: | ----: | -----: | -----: |
| Black-Scholes delta hedge   | 0.329 |      0.367 |    1.125 | 0.223 | 29.756 | -0.552 |
| Q-learning reward variation | 1.398 |      1.781 |    4.586 | 0.289 |  7.638 | -1.687 |
| Q-learning basic            | 1.463 |      1.889 |    5.012 | 0.319 |  8.014 | -1.781 |
| Q-learning state variation  | 2.031 |      2.434 |    6.654 | 0.997 | 21.487 | -3.029 |
| No hedge                    | 2.748 |      3.511 |   10.149 | 0.000 |  0.000 | -2.748 |

---

## Repository Structure

```text
QuantRAG_Hedge/
│
├── app/
│   ├── 01_Home.py
│   └── pages/
│       ├── 02_Black_Scholes_Dashboard.py
│       ├── 03_Greeks_and_Hedging.py
│       └── 04_RL_Hedging_Environment.py
│
├── data/
│   ├── raw_docs/
│   │   ├── 01_options_basics.md
│   │   ├── 02_black_scholes.md
│   │   ├── 03_greeks.md
│   │   ├── 04_delta_hedging.md
│   │   ├── 05_rl_hedging.md
│   │   ├── 06_mdp_q_learning_hedging.md
│   │   └── 07_experiment_design.md
│   │
│   └── processed/
│       └── basic_vector_store/
│           ├── chunks.json
│           └── embeddings.npy
│
├── experiments/
│   ├── train_q_learning.py
│   ├── run_q_learning_variations.py
│   └── run_milestone9_evaluation.py
│
├── outputs/
│   └── milestone8_q_learning/
│       ├── q_learning_config.json
│       ├── q_learning_history.csv
│       ├── q_learning_learning_curve.png
│       ├── q_learning_learning_curve_abs_error.png
│       ├── q_learning_q_table.json
│       └── q_learning_summary.csv
│
├── results/
│   └── milestone9/
│       ├── summary_metrics.csv
│       ├── evaluation_results_raw.csv
│       ├── learning_curves.csv
│       ├── policy_heatmap_data_q_basic_seed42.csv
│       ├── 01_learning_curve.png
│       ├── 02_hedging_error_distribution.png
│       ├── 03_transaction_cost_vs_error.png
│       ├── 04_bar_mean_abs_error.png
│       ├── 05_policy_heatmap.png
│       └── q_learning_*_qtable.pkl
│
├── scripts/
│   └── run_basic_rag.py
│
├── src/
│   ├── agents/
│   │   └── q_learning_agent.py
│   │
│   ├── envs/
│   │   └── option_hedging_env.py
│   │
│   ├── tools/
│   │   ├── black_scholes.py
│   │   ├── greeks.py
│   │   ├── gbm_simulator.py
│   │   ├── monte_carlo.py
│   │   ├── delta_hedging.py
│   │   └── test_quant_tools.py
│   │
│   ├── black_scholes.py
│   ├── chunker.py
│   ├── document_loader.py
│   ├── embedder.py
│   ├── generator.py
│   ├── prompt_builder.py
│   ├── rag_pipeline.py
│   ├── retriever.py
│   └── vector_store.py
│
├── tests/
│   └── test_m7_env.py
│
├── requirements.txt
└── README.md
```

---

## Key Components

### 1. Black-Scholes Pricing and Greeks

Implemented in:

```text
src/tools/black_scholes.py
src/tools/greeks.py
```

These modules compute European call and put prices and Greeks such as delta, gamma, vega, and theta. They provide the analytical foundation for the delta hedging baseline.

---

### 2. GBM Market Simulation

Implemented in:

```text
src/tools/gbm_simulator.py
```

The underlying stock price is simulated using geometric Brownian motion:

```math
S_{t+\Delta t} = S_t \exp\left((\mu - \frac{1}{2}\sigma^2)\Delta t + \sigma\sqrt{\Delta t}Z_t\right)
```

where (Z_t \sim N(0,1)).

---

### 3. Delta Hedging Baseline

Implemented in:

```text
src/tools/delta_hedging.py
```

The delta hedge strategy dynamically rebalances the stock position using the Black-Scholes delta. Transaction costs are applied whenever the hedge position changes.

---

### 4. RL Hedging Environment

Implemented in:

```text
src/envs/option_hedging_env.py
```

The hedging problem is formulated as a Markov Decision Process:

* **State:** moneyness bin, time bin, and optionally hedge-position/recent movement information
* **Action:** discrete target hedge ratio
* **Reward:** negative hedging error and transaction cost penalty
* **Goal:** minimize final hedging error while controlling trading cost

---

### 5. Q-learning Agent

Implemented in:

```text
src/agents/q_learning_agent.py
```

The agent uses the standard Q-learning update:

```math
Q(s_t,a_t) \leftarrow Q(s_t,a_t) + \alpha
\left[
r_t + \gamma \max_{a'} Q(s_{t+1},a') - Q(s_t,a_t)
\right]
```

Three Q-learning settings are tested:

* basic Q-learning
* reward variation
* state variation

---

### 6. RAG and Tool-Augmented Assistant

Implemented in:

```text
src/document_loader.py
src/chunker.py
src/embedder.py
src/vector_store.py
src/retriever.py
src/prompt_builder.py
src/generator.py
src/rag_pipeline.py
scripts/run_basic_rag.py
```

The RAG component retrieves relevant explanations from the project knowledge base in `data/raw_docs/`. It supports the broader educational interface of the project. Numerical finance calculations are handled by deterministic Python tools rather than relying only on an LLM.

---

### 7. Streamlit Dashboard

Implemented in:

```text
app/01_Home.py
app/pages/02_Black_Scholes_Dashboard.py
app/pages/03_Greeks_and_Hedging.py
app/pages/04_RL_Hedging_Environment.py
```

The dashboard allows interactive exploration of:

* Black-Scholes option pricing
* Greeks
* GBM simulation
* delta hedging
* reinforcement learning hedging environment

---

## Installation

Clone the repository:

```bash
git clone https://github.com/<your-username>/QuantRAG_Hedge.git
cd QuantRAG_Hedge
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Streamlit App

From the project root, run:

```bash
streamlit run app/01_Home.py
```

Then open the local URL shown in the terminal.

---

## Running the Q-learning Experiments

Train the basic Q-learning agent:

```bash
python experiments/train_q_learning.py
```

Run Q-learning variations:

```bash
python experiments/run_q_learning_variations.py
```

Run the final multi-seed experimental evaluation:

```bash
python experiments/run_milestone9_evaluation.py
```

The final results will be saved to:

```text
results/milestone9/
```

---

## Output Files

The final evaluation produces:

```text
summary_metrics.csv
evaluation_results_raw.csv
learning_curves.csv
policy_heatmap_data_q_basic_seed42.csv
01_learning_curve.png
02_hedging_error_distribution.png
03_transaction_cost_vs_error.png
04_bar_mean_abs_error.png
05_policy_heatmap.png
```

These files are used to generate the tables and figures in the final report.

---

## Example Figures

### Q-learning Learning Curve

```text
results/milestone9/01_learning_curve.png
```

### Hedging Error Distribution

```text
results/milestone9/02_hedging_error_distribution.png
```

### Transaction Cost vs Hedging Error

```text
results/milestone9/03_transaction_cost_vs_error.png
```

### Mean Absolute Hedging Error by Algorithm

```text
results/milestone9/04_bar_mean_abs_error.png
```

### Learned Q-learning Policy Heatmap

```text
results/milestone9/05_policy_heatmap.png
```

---

## Testing

Run the available tests with:

```bash
pytest
```

or run a specific test file:

```bash
pytest tests/test_m7_env.py
```

---

## Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Streamlit
* scikit-learn / vector utilities
* Tabular Q-learning
* Black-Scholes option pricing
* Geometric Brownian Motion simulation
* Retrieval-Augmented Generation pipeline

---

## Notes on Academic Integrity

This repository was developed for an Artificial Intelligence programming project. The implementation includes:

* original implementation work,
* AI-assisted development,
* standard formulas and algorithms adapted from course materials and cited references.

A separate `statement.pdf` is included in the official submission to declare which parts were written independently, which parts were AI-assisted, and which parts were based on existing sources.

---

## References

The project is based on concepts from option pricing, reinforcement learning, and retrieval-augmented generation, including:

1. Black, F. and Scholes, M. (1973). *The Pricing of Options and Corporate Liabilities.*
2. Merton, R. C. (1973). *Theory of Rational Option Pricing.*
3. Sutton, R. S. and Barto, A. G. (2018). *Reinforcement Learning: An Introduction.*
4. Buehler, H., Gonon, L., Teichmann, J., and Wood, B. (2019). *Deep Hedging.*
5. Halperin, I. (2017). *QLBS: Q-Learner in the Black-Scholes(-Merton) Worlds.*
6. Lewis, P. et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.*

---

## Author

**Nguyen Khanh Ngoc**
College of Engineering and Computer Science
VinUniversity

```
```
