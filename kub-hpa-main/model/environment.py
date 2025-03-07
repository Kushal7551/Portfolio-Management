import numpy as np
from gym import Env
from gym.spaces import Box


class StockTradingEnv(Env):
    def __init__(self, data):
        super(StockTradingEnv, self).__init__()
        self.data = data
        self.current_step = 0
        self.cash = 100000  # Starting cash 
        self.stock_held = 0
        self.transaction_cost_pct = 0.001  # 0.1% transaction cost
        self.numeric_data = self.data.drop(columns=['Date', 'Symbol', 'Series'], errors='ignore')

        self.action_space = Box(low=0, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(self.numeric_data.shape[1] + 2,), dtype=np.float32
        )

    def reset(self):
        self.current_step = 0
        self.cash = 100000
        self.stock_held = 0
        return self._get_observation()

    def step(self, action):
        action = np.clip(action, 0, 1)[0]
        current_price = self.data.loc[self.current_step, 'close']
        net_worth = self.cash + self.stock_held * current_price
        target_stock_value = action * net_worth
        target_stock_quantity = target_stock_value / current_price

        if target_stock_quantity > self.stock_held:  # Buy
            buy_quantity = target_stock_quantity - self.stock_held
            buy_cost = buy_quantity * current_price * (1 + self.transaction_cost_pct)
            if buy_cost <= self.cash:
                self.stock_held += buy_quantity
                self.cash -= buy_cost
        elif target_stock_quantity < self.stock_held:  # Sell
            sell_quantity = self.stock_held - target_stock_quantity
            sell_revenue = sell_quantity * current_price * (1 - self.transaction_cost_pct)
            self.stock_held -= sell_quantity
            self.cash += sell_revenue

        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        reward = self._calculate_reward()

        return self._get_observation(), reward, done, {"net_worth": net_worth}

    def _get_observation(self):
        obs = self.numeric_data.iloc[self.current_step].values
        return np.append(obs, [self.cash, self.stock_held])

    def _calculate_reward(self):
        if self.current_step == 0:
            return 0
        prev_price = self.data.loc[self.current_step - 1, 'close']
        current_price = self.data.loc[self.current_step, 'close']
        portfolio_value = self.cash + self.stock_held * current_price
        prev_portfolio_value = self.cash + self.stock_held * prev_price
        return np.log(portfolio_value / prev_portfolio_value)