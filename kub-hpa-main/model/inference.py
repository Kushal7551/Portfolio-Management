import sys
import pandas as pd
import numpy as np
from environment import StockTradingEnv
from training import DQNAgent


def evaluate_and_log(env, agent, weights_path, output_csv_path):
    agent.load_weights(weights_path)
    results = []
    state = env.reset()
    state = np.reshape(state, [1, agent.state_size])

    done = False
    while not done: 
        action = agent.act(state)

        current_step = env.current_step
        current_price = env.data.loc[current_step, 'close']
        open_price = env.data.loc[current_step, 'open']
        stock_held_before = env.stock_held

        next_state, _, done, info = env.step(action)

        target_stock_value = action[0] * (env.cash + env.stock_held * current_price)
        target_stock_quantity = target_stock_value / current_price
        quantity = 0
        if target_stock_quantity > stock_held_before:
            action_type = "buy"
            quantity = target_stock_quantity - stock_held_before
        elif target_stock_quantity < stock_held_before:
            action_type = "sell"
            quantity = stock_held_before - target_stock_quantity
        else:
            action_type = "hold"

        results.append({
            "time_step": current_step,
            "open_price": open_price,
            "close_price": current_price,
            "cash": env.cash,
            "stock_held": env.stock_held,
            "net_worth": info["net_worth"],
            "action": action[0],
            "action_type": action_type,
            "quantity": quantity,
        })

        state = np.reshape(next_state, [1, agent.state_size])

    results_df = pd.DataFrame(results)
    results2 = env.data
    results2['signal']=results_df['action_type']
    results2['quantity']=results_df['stock_held']
    results2['netWorth']=results_df['net_worth']
    results2['balance']=results_df['cash']
    results2.to_csv(output_csv_path, index=False)
    print(f"Results saved to {output_csv_path}")
    if len(sys.argv) != 2:
        print("Usage: python evaluate_model.py <input_csv_filename>")
        sys.exit(1)

# Get the input CSV filename
input_csv_filename = sys.argv[1]
input_csv_path = f"/{input_csv_filename}"  # Input file path
output_csv_path = f"/{input_csv_filename.replace('.csv', '_results.csv')}"  # Output file path
weights_path = "/app/weights/dqn_final.weights.h5"  # Path to the model weights

# Load the data
data = pd.read_csv(input_csv_path)

# Initialize the environment and agent
env = StockTradingEnv(data)
state_size = env.observation_space.shape[0]
action_size = env.action_space.shape[0]
agent = DQNAgent(state_size, action_size)

# Evaluate and log results
evaluate_and_log(env, agent, weights_path, output_csv_path)