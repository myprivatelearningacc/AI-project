from src.envs.option_hedging_env import OptionHedgingEnv


env = OptionHedgingEnv(
    S0=100,
    K=100,
    T=1,
    r=0.05,
    sigma=0.2,
    n_steps=10,
    transaction_cost_rate=0.001,
    option_type="call",
    seed=42,
)

state = env.reset()
print("Initial state:", state)
print("Actions:", env.get_action_space())

done = False
total_reward = 0

while not done:
    action = env.sample_random_action()
    next_state, reward, done, info = env.step(action)

    total_reward += reward

    print("Action:", action)
    print("Next state:", next_state)
    print("Reward:", reward)
    print("Done:", done)
    print("Info:", info)
    print("-" * 60)

print("Total reward:", total_reward)
print("Episode history length:", len(env.get_history()))