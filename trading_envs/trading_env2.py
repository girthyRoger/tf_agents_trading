from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import tensorflow as tf
import numpy as np
import pandas as pd

from tf_agents.environments import py_environment
from tf_agents.environments import validate_py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types

import trading_envs.yfinance_utils as yf_u
import trading_envs.transaction_manager as transaction_manager

class TradingEnv(py_environment.PyEnvironment):
    # TODO: Implement a render and reward tracking so I can see what's going on.
    def __init__(self, max_position:int=10) -> None:
        # params
        self._max_position:int = max_position
        
        # specifications
        # self._observation_space = ("Date", "Open", "High", "Low", "Close", "Volume", "Vwap", "Unrealised Gain","Position")
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(9,), 
            dtype= np.float32, 
            minimum= [0, 0, 0, 0, 0, 0, 0, -np.inf, -1], 
            maximum= [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 1]
        )
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(1,),
            dtype=np.float32,
            minimum=[-1],
            maximum=[1]
            )
        
        # episode properties - set in self._reset()
        self._total_profit = 0
        self._episode_observations:pd.DataFrame = pd.DataFrame()
        self._max_steps = 0
        self._current_step = 0
        self._episode_ended = False
        self._tm = transaction_manager.TransactionManager(self._max_position)

    def action_spec(self):
        return self._action_spec
    
    def observation_spec(self):
        return self._observation_spec
        
    def _get_observations(self, step:int):
        position = self._tm.get_position()
        
        # process ticker dataframe and spit out observations from current step
        yf_observations = self._episode_observations.iloc[step,:]
        vwap = (yf_observations.High + yf_observations.Low + yf_observations.Close)/3 # daily vwap only
        unrealised_gain = self._tm.get_gain_at_position(position=0,close=yf_observations.Close)

        observation = np.concatenate([yf_observations, [vwap], [unrealised_gain], [position]], dtype=np.float32)
        
        return observation
    
    def _reset(self):
        # set episode properties to starting values
        self._total_profit = 0
        self._episode_observations= yf_u.get_random_ticker_history()
        self._max_steps = len(self._episode_observations)-1
        self._episode_ended = False
        self._tm = transaction_manager.TransactionManager(self._max_position)
        self._current_step = 0
        observation = self._get_observations(self._current_step)
        
        return ts.restart(observation=observation)
        
    def _step(self, action):
        # Reset if the episode has ended
        if self._episode_ended:
            return self.reset()

        # Process actions agains observation provided for step s
        s_timestep = self.current_time_step()        
        s_close = s_timestep.observation[4]

        reward = self._tm.add(action=action[0], close=s_close)

        # Prepare changes for step s+1
        self._current_step += 1
        observation = self._get_observations(self._current_step)

        if self._current_step == self._max_steps:
            self._episode_ended = True
            return ts.termination(observation=observation, reward=reward)
        else:    
            return ts.transition(observation=observation, reward=reward, discount=0.99)
        
    
def validate():
    env = TradingEnv(10)
    validate_py_environment(env)