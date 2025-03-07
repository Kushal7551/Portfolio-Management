import pytest
import pandas as pd
import numpy as np
from environment import StockTradingEnv
from training import DQNAgent

@pytest.fixture
def sample_data():
    """Fixture to create a small sample dataset for testing."""
    data = {
        'Date': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05'],
        'Prev Close': [100, 102, 101, 103, 105],
        'open': [101, 102, 102, 104, 106],
        'high': [102, 104, 103, 106, 108],
        'low': [99, 101, 100, 102, 104],
        'Last': [101, 103, 102, 105, 107],
        'close': [102, 103, 101, 104, 106],
        'VWAP': [100.5, 102.3, 101.2, 103.4, 105.6],
        'Volume': [1000, 1200, 1100, 1300, 1500]
    }
    return pd.DataFrame(data)

def test_environment_initialization(sample_data):
    """Test if the StockTradingEnv initializes correctly."""
    env = StockTradingEnv(sample_data)
    obs = env.reset()

    assert obs.shape[0] == sample_data.shape[1] - 3 + 4, "Observation shape mismatch."
    assert env.cash == 100000, "Initial cash is not set correctly."
    assert env.stock_held == 0, "Initial stock held is not set correctly."

def test_environment_step(sample_data):
    """Test if the environment step function updates state correctly."""
    env = StockTradingEnv(sample_data)
    obs = env.reset()

    initial_cash = env.cash
    action = [0.5]  # Allocate 50% of the net worth to stock
    obs, reward, done, info = env.step(action)

    assert env.cash <= initial_cash, "Cash did not reduce after a buy action."
    assert env.stock_held >= 0, "Stock held is invalid after a buy action."
    assert 'net_worth' in info, "'net_worth' key missing in the info dictionary."

def test_agent_initialization(sample_data):
    """Test if the DQNAgent initializes correctly with the environment."""
    env = StockTradingEnv(sample_data)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    agent = DQNAgent(state_size, action_size)

    assert agent.model is not None, "DQNAgent model is not initialized."

def test_agent_action(sample_data):
    """Test if the agent can take a valid action."""
    env = StockTradingEnv(sample_data)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    agent = DQNAgent(state_size, action_size)

    state = env.reset()
    state = state.reshape([1, state_size])
    action = agent.act(state)

    assert 0 <= action < action_size, "Agent produced an invalid action."

def test_model_training(sample_data):
    """Test the training process with a DQNAgent."""
    env = StockTradingEnv(sample_data)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    agent = DQNAgent(state_size, action_size)

    # Simulate training for a small number of steps
    for _ in range(5):
        state = env.reset()
        state = state.reshape([1, state_size])
        done = False

        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            next_state = next_state.reshape([1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state

