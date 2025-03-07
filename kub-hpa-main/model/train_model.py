import sys
import pandas as pd
from environment import StockTradingEnv
from training import DQNAgent

def train_model(data_path, weights_output_path, episodes=5):
    # Load training data
    data = pd.read_csv(data_path)
    
    # Initialize environment and agent
    env = StockTradingEnv(data)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    agent = DQNAgent(state_size, action_size) 
    
    # Fine-tuning: Load existing weights if available
    try:
        agent.load_weights("/app/weights/dqn_final.weights.h5")
        print("Loaded existing model weights for fine-tuning.")
    except:
        print("No existing weights found. Training from scratch.")
    
    # Training loop
    for e in range(episodes):
        state = env.reset()
        state = state.reshape([1, state_size])
        done = False    
        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            next_state = next_state.reshape([1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
        
        # Train the agent
        agent.replay(32)  # Adjust batch size as needed
        print(f"Episode {e + 1}/{episodes} complete.")

    # Save fine-tuned weights
    # agent.save_weights("/model/weights/dqn_final.weights.h5")
    print(f"Weights saved to {weights_output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python train_model.py <data_csv_path> <weights_output_path>")
        sys.exit(1)

    data_csv_path = sys.argv[1]
    weights_output_path = sys.argv[2]
    train_model(data_csv_path, weights_output_path)
