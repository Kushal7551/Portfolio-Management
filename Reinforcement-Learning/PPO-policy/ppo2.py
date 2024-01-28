from collections import deque
import random
import json
import gym
from gym import spaces
import pandas as pd
import numpy as np
# from utils import TradingGraph, Write_to_file
import math
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from indicators import BolingerBands, RSI, PSAR, SMA
import random
warnings.filterwarnings("ignore")
TRANSACTION_COST=0.015
MAX_ACCOUNT_BALANCE = 2147483647
MAX_NUM_SHARES = 2147483647
MAX_SHARE_PRICE = 200000
MAX_STEPS = 2000000
MAX_CAPITAL = 220000
MIN_CAPITAL= 220000

INITIAL_ACCOUNT_BALANCE = 100000
AVAILABLE_TO_TRADE=INITIAL_ACCOUNT_BALANCE
LOOK_BACK=10
FEATURES=5

def RSI(data, n=5):
    price = pd.DataFrame(data)
    delta = price.diff()
    delta[0][0] = delta[0][1]
    dUp, dDown = delta.copy(), delta.copy()
    dUp[dUp < 0] = 0
    dDown[dDown > 0] = 0
    RolUp = dUp.ewm(span=n).mean()
    RolDown = dDown.ewm(span=n).mean().abs()
    RS = RolUp / RolDown
    rsi = 100.0 - (100.0 / (1.0 + RS))
    t_rsi = np.array(rsi.replace(np.nan, 0)).tolist()
    return [l[0] for l in t_rsi]

class StockTradingEnv(gym.Env):

    def __init__(self, df):
        super(StockTradingEnv, self).__init__()

        self.df = df
        self.Render_range = 1000
        self.number_of_trades=0
        self.las_net_worth=100000
        self.reward_range = (-MAX_ACCOUNT_BALANCE, MAX_ACCOUNT_BALANCE)

        # Actions of the format Buy x%, Sell x%, Hold, etc.
        self.action_space = spaces.Box(
            low=np.array([0, 0]), high=np.array([3, 1]), dtype=np.float16)

        # Prices contains the OHCL values for the last five prices
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(FEATURES+1, LOOK_BACK), dtype=np.float16)
        self.portfolio_value=[INITIAL_ACCOUNT_BALANCE]
    
    def printAverageNetworth(self):
        return self.portfolio_value

    def compute_sharpe_ratio(self, returns):
        mean_return = np.mean(returns)
        risk_free_rate = -0.005  # You may want to adjust this based on your scenario
        std_dev_return = np.std(returns)

        if std_dev_return == 0:
            return 0  # Avoid division by zero

        sharpe_ratio = (mean_return - risk_free_rate) / std_dev_return
        return sharpe_ratio
    
    def _next_observation(self):
        test=self.df.loc[self.current_step - LOOK_BACK: self.current_step -1, 'Close'].values
        self.current_step=max(self.current_step,LOOK_BACK+10)
        frame = np.array([
            self.df.loc[self.current_step - LOOK_BACK: self.current_step -1, 'Open'].values / MAX_SHARE_PRICE,
            self.df.loc[self.current_step - LOOK_BACK: self.current_step -1, 'High'].values / MAX_SHARE_PRICE,
            self.df.loc[self.current_step - LOOK_BACK: self.current_step -1, 'Low'].values / MAX_SHARE_PRICE,
            self.df.loc[self.current_step - LOOK_BACK: self.current_step -1, 'Close'].values / MAX_SHARE_PRICE,
            self.df.loc[self.current_step - LOOK_BACK: self.current_step -1, 'Volume'].values / MAX_NUM_SHARES,
            # self.df.loc[self.current_step - LOOK_BACK: self.current_step -1, 'BB_up'].values / MAX_SHARE_PRICE,
            # self.df.loc[self.current_step - LOOK_BACK: self.current_step -1, 'BB_dn'].values / MAX_SHARE_PRICE,
            # self.df.loc[self.current_step - LOOK_BACK: self.current_step -1, 'SMA'].values / MAX_SHARE_PRICE,

        ])
        lol=np.zeros((1,LOOK_BACK))
        lol[0][0]=self.balance / MAX_ACCOUNT_BALANCE
        lol[0][1]=self.max_net_worth / MAX_ACCOUNT_BALANCE
        lol[0][2]=self.shares_held / MAX_NUM_SHARES
        lol[0][3]=self.cost_basis / MAX_SHARE_PRICE
        lol[0][4]=self.total_shares_sold / MAX_NUM_SHARES
        lol[0][5]=self.total_sales_value / (MAX_NUM_SHARES * MAX_SHARE_PRICE)
        obs1 = np.append(frame, lol, axis=0)
        obs =obs1
        return obs

    def _take_action(self, action):
        # Set the current price to a random price within the time step
        current_price = random.uniform(
            self.df.loc[self.current_step, "Open"], self.df.loc[self.current_step, "Close"])

        action_type = action[0]
        amount = action[1]
        if action_type < 1:
            # Buy amount % of balance in shares
          
            total_possible = (min(self.balance,AVAILABLE_TO_TRADE) / current_price)
            shares_bought = (total_possible * amount)
            if (shares_bought>0.00001):
                prev_cost = self.cost_basis * self.shares_held
                additional_cost = shares_bought * current_price
                self.number_of_trades+=1
                self.balance -= additional_cost
                self.cost_basis = (
                    prev_cost + additional_cost) / (self.shares_held + shares_bought)
                self.shares_held += shares_bought
                Date = self.df.loc[self.current_step, 'Date']
                self.trades.append({'Date' : Date, 'High' : self.df.loc[self.current_step, "Close"], 'Low' : self.df.loc[self.current_step, "Open"], 'total': shares_bought, 'type': "buy"})
            

        elif action_type < 2:
            # Sell amount % of shares held
            shares_sold = (self.shares_held * amount)
            if (shares_sold>0.00001):
                self.balance += (shares_sold * current_price )
                self.shares_held -= shares_sold
                self.total_shares_sold += shares_sold
                self.total_sales_value += shares_sold * current_price
                Date = self.df.loc[self.current_step, 'Date']
                self.number_of_trades+=1
                self.trades.append({'Date' : Date, 'High' : self.df.loc[self.current_step, "Close"], 'Low' : self.df.loc[self.current_step, "Open"], 'total': shares_sold, 'type': "sell"})

        self.net_worth = self.balance + self.shares_held * current_price

        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth

        if self.shares_held == 0:
            self.cost_basis = 0

    def step(self, action):
        # Execute one time step within the environment
        self._take_action(action)
        self.portfolio_value.append(self.net_worth)
        self.current_step += 1
        if self.current_step > len(self.df.loc[:, 'Open'].values) - LOOK_BACK:
            self.current_step = 0

        delay_modifier = (self.current_step / MAX_STEPS)
        daily_returns = np.diff(self.portfolio_value) / self.portfolio_value[:-1]
        sharpe_ratio = self.compute_sharpe_ratio(daily_returns)
       
        action_type = action[0]
        done = self.net_worth <= 0.9*self.max_net_worth
        reward=(self.net_worth - self.las_net_worth)*100
        self.las_net_worth=self.net_worth
        obs = self._next_observation()
        # print(self.current_step)

        return obs, reward, done, {}

    def reset(self):
        # Reset the state of the environment to an initial state
        self.balance = INITIAL_ACCOUNT_BALANCE
        self.net_worth = INITIAL_ACCOUNT_BALANCE
        self.max_net_worth = INITIAL_ACCOUNT_BALANCE
        self.shares_held = 0
        self.cost_basis = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0
        # self.visualization = TradingGraph(Render_range=self.Render_range) # init visualization
        self.trades = deque(maxlen=self.Render_range)

        # Set the current step to a random point within the data frame
        self.current_step = random.randint(
            LOOK_BACK+10, len(self.df.loc[:, 'Open'].values) -LOOK_BACK- 10)

        return self._next_observation()
    
    def test_reset(self,df):
        # Reset the state of the environment to an initial state
        self.df=df
        # print(df)
        self.balance = INITIAL_ACCOUNT_BALANCE
        self.net_worth = INITIAL_ACCOUNT_BALANCE
        self.max_net_worth = INITIAL_ACCOUNT_BALANCE
        self.number_of_trades=0
        self.shares_held = 0
        self.cost_basis = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0

        # Set the current step to a random point within the data frame
        self.current_step =LOOK_BACK+16
        return self._next_observation()

    def render(self, mode='human'):
        # Render the environment to the screen
        profit = self.net_worth - INITIAL_ACCOUNT_BALANCE

        Date = self.df.loc[self.current_step, 'Date']
        Open = self.df.loc[self.current_step, 'Open']
        Close = self.df.loc[self.current_step, 'Close']
        High = self.df.loc[self.current_step, 'High']
        Low = self.df.loc[self.current_step, 'Low']
        Volume = self.df.loc[self.current_step, 'Volume']

        # Render the environment to the screen
        # self.visualization.render(Date, Open, High, Low, Close, Volume, self.net_worth, self.trades)

        print(f'Step: {self.current_step}')
        print(f'Balance: {self.balance}')
        print(
            f'Shares held: {self.shares_held} (Total sold: {self.total_shares_sold})')
        print(
            f'Avg cost for held shares: {self.cost_basis} (Total sales value: {self.total_sales_value})')
        print(
            f'Net worth: {self.net_worth} (Max net worth: {self.max_net_worth})')
        print(f'Profit: {profit}')
        print(f'Number of Trades {self.number_of_trades}')
        return self.balance,self.current_step,profit,self.net_worth

import gym
import json
import datetime as dt

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO,DDPG
import pandas as pd
path='mrf.csv'
df = pd.read_csv(path)
df.dropna(inplace=True)
df = df.reset_index()

boll=BolingerBands(data=df, period=10, std=1)
df=boll.compute()
print(df)

p=int(np.floor(0.75*len(df)))
env=StockTradingEnv(df[0:p])
test_df=pd.read_csv(path)
test_df=test_df[p:]
test_df.reset_index(inplace=True)
from sb3_contrib import RecurrentPPO
model = RecurrentPPO("MlpLstmPolicy", env, verbose=1)
model.learn(total_timesteps=10000)
print("testing")
model.save('model.h5')
obs = env.test_reset(test_df)
columns = ['Balance', 'step', 'profit','net_worth']
data = pd.DataFrame(columns=columns)
for i in range(len(df)-p):
    action, _states = model.predict(obs)
    print('Action',action)
    # print('states',_states)
    obs, rewards, done, info = env.step(action)
    balance, step,profit,net_worth=env.render(mode='human')
    print(balance,profit,net_worth)
    new_row = {'Balance': balance, 'step': step,'profit':profit,'net_worth':net_worth}
    data.loc[len(data)]=new_row
print(np.mean(env.printAverageNetworth()))
data.to_csv('example.csv', index=False)
test_df.to_csv('market.csv',index=False)



