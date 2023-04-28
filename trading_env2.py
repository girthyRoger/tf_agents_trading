from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import tensorflow as tf
import numpy as np
import pandas as pd

from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts

from lib import yfinance_utils as yf_u

class TradingEnv(py_environment.PyEnvironment):
    def __init__(self, max_position:int=10) -> None:
        # params
        self._max_position:int = max_position
        
        # specifications
        self._observation_space = ("Date", "Open", "High", "Low", "Close", "Volume", "Position", "Vwap", "Unrealised Gain")
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(9,), 
            dtype= np.float32, 
            minimum= [0, 0, 0, 0, 0, 0, 0, 0, -1], 
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
        self._episode_ended = False
    
    def _get_observations(self, step:int):
        # process ticker dataframe and spit out observations from current step
        yf_observations = self._episode_observations.iloc[step,:]
        print(yf_observations)
        vwap = 0
        unrealised_gain = 0
        position = 0
        
        observation = np.concatenate([yf_observations, [vwap], [unrealised_gain], [position]])
        
        return observation
    
    def _reset(self):
        # set episode properties to starting values
        self._total_profit = 0
        self._episode_observations= yf_u.get_random_ticker_history()
        self._max_steps = len(self._episode_observations)-1
        self._episode_ended = False
        observation = self._get_observations(0)
        
        return ts.restart(observation=observation)
        
    def _step(self, action):
        s_timestep = self.current_time_step()
        s_position= s_timestep.observation[6]
        d_position = action - s_position
        
        d_shares = round(self._max_position*d_position)
        
        
        
        # How do we keep track of the gains
        pass
        
    def test(self):
        self._reset()
    
    
env = TradingEnv()

obs = env.test()