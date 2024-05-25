import numpy as np
from gym import spaces
import random
import pandas as pd 
import os

REWARD_LOOKBACK = 100

class State():
    def __init__(self,dataframes_array,features,lookback,time):
        self.features=np.array([dataframes_array[i][features][time-lookback:time] for i in range(len(dataframes_array))])

    def print(self):
        print(self.features.shape) 
        print(self.features) 


class Action():
    def __init__(self,weights,holdings):
        self.action=np.array(holdings)
        self.holdings=holdings
    def print(self):
        print(self.action.shape)
        print(self.action)
    

class MultiTradingEnv():
    def generate_random(self,size):
        random_numbers = np.random.rand(size)
        normalized_array = random_numbers / random_numbers.sum()
        return normalized_array
        
    def generate(self,size):
        random_numbers = np.random.rand(size)
        normalized_array = random_numbers
        return normalized_array
    
    def __init__(self,dataframes_array,norm_dataframes,total_timesteps,initial_cap,features,lookback,positions,wt_combs):
        self.features=features
        self.dataframes_array=dataframes_array
        self.norm_dataframes=norm_dataframes
        self.num_stocks=len(dataframes_array)
        self.action_space = spaces.Discrete(self.num_stocks , len(positions))
        self.total_timesteps=total_timesteps
        self.initial_cap=initial_cap
        self.current_step=0
        self.balance=initial_cap
        self.holdings=np.zeros(shape=(self.num_stocks))
        self.shares_held=np.zeros(shape=(self.num_stocks))
        self.history=[]
        self.lookback=lookback
        self.max_net_worth=initial_cap
        self.reward_history=[]
        self.positions=positions
        self.history_array = []
        self.history_array_episode = []
        self.weight_combs=wt_combs
        self.total_reward=0
        self.prev_shares_held = np.zeros(shape=(self.num_stocks))
        self.buy_hold=np.zeros(shape=len(self.dataframes_array[0]))

    def computeBH(self):
        tmp=np.zeros(shape=len(self.dataframes_array[0]))
        cnt=0
        wts=np.ones(self.num_stocks)/self.num_stocks
        for i in self.dataframes_array:
            num=(self.initial_cap*wts[cnt])/np.array(i['Open'][0])
            ar=np.array(i['Open'])*num
            tmp+=ar
            cnt+=1
        tmp=tmp[5:-15]
        tmp/=tmp[0]
        tmp*=self.initial_cap
        self.buy_hold=tmp
  
    def select_action(self):
        action=Action(weights=self.generate_random(self.num_stocks),holdings=self.generate(self.num_stocks))
        return action

    
    def reward(self):
      if(np.array_equal(self.prev_shares_held,self.shares_held)==False):
        if(len(self.history_array_episode)):
            reward = 0
            for i in range(min(REWARD_LOOKBACK,len(self.history_array_episode))):
              reward+=np.log(self.net_worth/self.history_array_episode[len(self.history_array_episode)-i-1][3])
            return reward/min(REWARD_LOOKBACK,len(self.history_array_episode))
        else:
            return np.log(self.net_worth/self.initial_cap)
      else:
        return 0.0
    def rewardBH(self):
      if(np.array_equal(self.prev_shares_held,self.shares_held)==False):
        return (self.net_worth-self.buy_hold[self.current_step])/self.net_worth
      else:
        return 0.0

    
    def step(self,action,weight):
        if(self.current_step>=self.total_timesteps-15):
            terminated=True
        else:   
            terminated=False
        indices=action.cpu().numpy()
        self.holdings=np.array([self.positions[i] for i in indices][0])
        current_prices=np.array([self.dataframes_array[i]['Open'].iloc[self.current_step+1] for i in range(len(self.dataframes_array))])
        self.net_worth=np.sum(self.shares_held*current_prices)+self.balance
        self.prev_shares_held = [i for i in self.shares_held]
        current_cash_flow=0      
        weight_array = np.array(self.weight_combs[weight])  
        for i in range(len(self.holdings)):
            net_allot=weight_array[i]*self.net_worth
            stocks=self.holdings[i]*net_allot//current_prices[i]
            current_cash_flow+=(net_allot-stocks*current_prices[i])
            self.shares_held[i]=stocks
        reward=self.reward()
        # reward=self.rewardBH()
        self.balance = current_cash_flow
        self.current_step+=1
        self.max_net_worth=max(self.max_net_worth,self.net_worth)
        state=State(dataframes_array=self.norm_dataframes,features=self.features,lookback=self.lookback,time=self.current_step)
        return state,reward,terminated
            
    def reset(self):
        self.balance = self.initial_cap
        self.net_worth = self.initial_cap
        self.max_net_worth = self.initial_cap
        self.shares_held = np.zeros(shape=(self.num_stocks)) 
        self.capital=self.initial_cap
        # self.current_step = random.randint(0,self.total_timesteps)
        self.history_array_episode = []
        self.current_step=self.lookback+1
        state=State(dataframes_array=self.norm_dataframes,features=self.features,lookback=self.lookback,time=self.current_step)
        return state
    
    # def save_log():
    #     print()
    
        
    def render(self,episode_num,reward):
        # print(f'Balance {self.balance}')
        # print(f'Holdings {self.holdings}')
        # print(f'Weights {self.weights}')
        # print(f'Shares {self.shares_held}')
        # print(f'Net worth {self.net_worth}')
        self.total_reward+=reward
        self.history_array_episode.append([self.balance, self.holdings, self.shares_held, self.net_worth])
    
    def render_logs(self):
        self.history_array.append(self.history_array_episode)
        self.history.append(self.net_worth)
        self.reward_history.append(self.total_reward)
        self.total_reward=0


