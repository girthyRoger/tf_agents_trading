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

from enum import Enum

class Action(Enum):
    HOLD = 0
    BUY = 1
    SELL = 2

class TradingEnv(py_environment.PyEnvironment):
    # TODO: Implement render()
    
    def __init__(self, df: pd.DataFrame, df_boundary: tuple[int, int], window_size, starting_cash, handle_auto_reset: bool = False):
        super().__init__(handle_auto_reset)
        
        # inputs
        self._df = df
        self._window_size = window_size
        self._df_boundary = df_boundary
        self._starting_cash = starting_cash
        
        # Env specs
        self._action_spec= array_spec.BoundedArraySpec(
            shape=(1,), dtype=np.int32, minimum=0, maximum=2)
        self._observation_spec= array_spec.BoundedArraySpec(
            shape=(4,self._window_size,), dtype=np.float32, minimum=[[0], [-np.inf], [0], [0]], maximum=[[np.inf]])
        self._episode_ended= False
        
        # Episode
        self.prices, self.signal_features = self._process_data()
        self._cash, self._share_value = None, None
        self._portfolio_value = None
        self._holdings = None
        self._max_step = self._df_boundary[1] - self._df_boundary[0] - 1
        self._current_step = None #! There HAS to be a way to infer this from ts.timestep!!!! In the meantime we use this low IQ counter
        
        
    def action_spec(self):
        return self._action_spec
    
    def observation_spec(self):
        return self._observation_spec
    
    def _reset(self) -> ts.TimeStep:
        self._episode_ended = False
        
        self._cash = np.full((self._window_size), fill_value= self._starting_cash, dtype=np.float32)
        self._share_value = np.zeros((self._window_size), dtype=np.float32)
        self._holdings = 0
        self._current_step = 0 
        self._portfolio_value = 0
        
        observations = self._get_observations()
        
        return ts.restart(observations)

    
    def _step(self, action):
        # reset for new episode
        if self._episode_ended:
            return self.reset()
        
        # setup for action processing
        action = Action(action)
        timestep = self.current_time_step()
        observations = np.asarray(timestep.observation)
        current_price, current_cash = observations[0,-1], observations[2,-1]
              
        if action == Action.HOLD:
            pass
        elif action == Action.BUY:
            if current_cash >= current_price:
                current_cash -= current_price
                self._holdings +=1       
        elif action == Action.SELL:
            if self._holdings > 0:
                current_cash += current_price
                self._holdings -=1

        # calculate new values
        current_share_value = self._holdings*current_price        
        self._share_value = np.delete(np.append(self._share_value,[current_share_value]),0,0)
        self._cash = np.delete(np.append(self._cash,[current_cash]),0,0)
        self._portfolio_value = current_cash + current_share_value
        
        # Calculate reward
        reward = np.array(self._portfolio_value, dtype=np.float32)
        
        # advance timestep and get observations
        self._current_step += 1
        observations = self._get_observations()
        if self._current_step == self._max_step:
            self._episode_ended = True
        #return new timestep
        
        if self._episode_ended:
            print(
                f"Episode Complete!\n\
            My final portfolio value is: ${self._portfolio_value}\n\
            My cash balance is: ${current_cash}\n\
            My shares balance is: ${current_share_value}\n\
            My holdings have {self._holdings} shares." 
            )
            return ts.termination(observation= observations, reward= reward)
        else:
            return ts.transition(observation= observations, reward= reward, discount=1.0)
                                
    def _get_observations(self) -> np.ndarray[np.float32]:
        signal_obs = self.signal_features[self._current_step:self._window_size+self._current_step]
        price_obs, delta_obs = signal_obs[:,0], signal_obs[:,1]
        
        observations = np.array([price_obs, delta_obs, self._cash, self._share_value], dtype = np.float32) # ? Should cash and share value be calculated from the current timestep?
        return observations
    
    def _process_data(self):
        prices = self._df.loc[:,"Close"].to_numpy()
        prices[self._df_boundary[0] - self._window_size] # Validation
        
        start_index = self._df_boundary[0] - self._window_size
        end_index = self._df_boundary[1]
        prices= prices[start_index : end_index]
        
        diff = np.insert(np.diff(prices), 0, 0)
        signal_features = np.column_stack((prices, diff))
        
        return prices, signal_features
          
    
    #! Delete dis!
    def test_process(self):
        prices, signal_features = self._process_data()
        
        self._reset() 
        self.step(0)