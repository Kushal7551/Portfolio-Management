import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from collections import deque
import random
from environment import StockTradingEnv
import pandas as pd


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential([
            Dense(24, input_dim=self.state_size, activation='relu'),
            Dense(24, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                      loss='mse')
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.rand(self.action_size)  # Random action
        q_values = self.model.predict(state, verbose=0)
        return q_values[0]

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target += self.gamma * np.amax(self.model.predict(next_state, verbose=0)[0])
            target_f = self.model.predict(state, verbose=0)
            target_f[0] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_weights(self, filepath):
        self.model.save_weights(filepath)
        print(f"Weights saved to {filepath}.")

    def load_weights(self, filepath):
        """Load model weights from the specified filepath."""
        self.model.load_weights(filepath)
        print(f"Weights loaded from {filepath}.")


def train_dqn(env, agent, episodes=10, batch_size=32, save_interval=50):
    for e in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, agent.state_size])
        for time in range(200):  # Max steps per episode
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, agent.state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print(f"Episode {e+1}/{episodes} finished in {time} steps")
                break
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)

        if (e + 1) % save_interval == 0:
            agent.save_weights(f"dqn_weights_episode_{e+1}.h5")


# data = pd.read_csv("ADANI.csv")  # Replace with your data file
# env = StockTradingEnv(data)
# state_size = env.observation_space.shape[0]
# action_size = env.action_space.shape[0]
# agent = DQNAgent(state_size, action_size)
# train_dqn(env, agent, episodes=10, batch_size=32, save_interval=50)
# agent.save_weights("dqn_final.weights.h5")