from tf_agents.networks import actor_distribution_network
from tf_agents.agents.reinforce import reinforce_agent
from tensorflow import keras
import tensorflow as tf
from tf_agents.policies import py_tf_eager_policy
from tf_agents.environments import py_environment

import numpy as np


# Hyperparams

num_iterations = 250 # @param {type:"integer"}
collect_episodes_per_iteration = 2 # @param {type:"integer"}


log_interval = 25 # @param {type:"integer"}
num_eval_episodes = 10 # @param {type:"integer"}
eval_interval = 50 # @param {type:"integer"}



class REINFORCE():
    def __init__(self, observation_spec, action_spec, time_step_spec) -> None:
        # Hyperparams
        self._fc_layer_params = (100,)
        self._learning_rate = 1e-3 # @param {type:"number"}
        
        # actor network
        self.actor_net = actor_distribution_network.ActorDistributionNetwork(
            input_tensor_spec=observation_spec, 
            output_tensor_spec=action_spec, 
            fc_layer_params=self._fc_layer_params
            )
        
        # optimiser
        self.optimiser = keras.optimizers.Adam(learning_rate=self._learning_rate)
        
        # counter to track how many times the network is updated
        self.train_step_counter = tf.Variable(0)
        
        # Agent
        self.tf_agent = reinforce_agent.ReinforceAgent(
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            actor_network=self.actor_net,
            optimizer=self.optimiser,
            normalize_returns=True,
            train_step_counter=self.train_step_counter
        )
        self.tf_agent.initialize()
        
        self.eval_policy = self.tf_agent.policy
        self.collect_policy = self.tf_agent.collect_policy
        
    def compute_average_return(self, environment:py_environment.PyEnvironment, num_episodes=10):
        total_return= 0.0
        
        for _ in range(num_episodes):
            time_step = environment.reset()
            episode_return= 0.0
            
            while not time_step.is_last():
                action_step = self.eval_policy.action(time_step)
                time_step = environment.step(action_step.action)
                episode_return += time_step.reward()
            
            total_return += episode_return
        
        avg_return= total_return/num_episodes
        return avg_return.numpy()[0]
                