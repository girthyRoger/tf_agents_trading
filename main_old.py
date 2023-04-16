
# Optimal condition: portfolio value is as high as possible
# Episode length: 5 years
# Actions: Buy, Do nothing, Sell
# Task: Build a model that maximises the portfolio value at the end of the episode.

# Gym stuff
import gym
import gym_anytrading

# Stable baselines stuff
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import A2C

# processing libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


# import data
df = pd.read_csv("data/QAN.csv")

# Clean data
df["Date"] = pd.to_datetime(df["Date"])
print(df.dtypes)
df.set_index("Date", inplace=True)
df.sort_index(ascending=True, inplace=True)
print(df.head())

env = gym.make("stocks-v0", df=df, frame_bound=(10,100), window_size=5)

# state = env.reset()
# while True:
#     action = env.action_space.sample()
#     n_state, reward, done, info = env.step(action)
#     if done:
#         print("info:",info)
#         break
# plt.figure(figsize=(15,6))
# plt.cla()
# env.render_all()
# plt.show()


env_maker = lambda: gym.make("stocks-v0", df=df, frame_bound=(10,100), window_size=5)
env = DummyVecEnv([env_maker])

model = A2C("MlpLstmPolicy", env, verbose=1)
model.learn(total_timesteps=100000)

# # buying shares

# class Asset():
#     def __init__(self, buy_price, quantity) -> None:
#         self.buy_price: int = buy_price
#         self.quantity = int = quantity

# class Portfolio():
#     def __init__(self, starting_cash) -> None:
#         self.cash = 0
#         self.starting_cash = starting_cash
#         self.stocks = []
#         self.asset_value = 0
#         self.position = self.asset_value + self.cash

#     def set_portfolio(self):
#         self.cash = self.starting_cash
#         self.stocks =[]
    
#     def get_position(self, current_price):
#         self.asset_value = current_price*sum(self.stocks)/len(self.stocks)
#         return self.position
    
#     def buy_stock(self, current_price):
#         if current_price <= self.cash:
#             self.stocks.append(current_price)
#             self.cash -= current_price
        
#     def sell_stock(self, current_price):
#         if len(self.stocks)>0:
#             self.stocks.pop(0)
#             self.cash += current_price
        
