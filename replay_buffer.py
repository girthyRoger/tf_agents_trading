from tf_agents.agents.reinforce import reinforce_agent
from tf_agents.specs import tensor_spec
from tf_agents.replay_buffers import reverb_replay_buffer, reverb_utils
from tf_agents.policies import py_tf_eager_policy
from tf_agents.drivers import py_driver



import reverb


class ReplayBuffer():
    def __init__(self, tf_agent:reinforce_agent.ReinforceAgent ) -> None:
        # hyperparams
        self.replay_buffer_capacity = 2000 # @param {type:"integer"}

        # configuration
        self.table_name = 'uniform_table'
        
        self.replay_buffer_signature = tensor_spec.from_spec(
            tf_agent.collect_data_spec)
        self.replay_buffer_signature = tensor_spec.add_outer_dim(
            self.replay_buffer_signature)
        
        self.table = reverb.Table(
            self.table_name,
            max_size=self.replay_buffer_capacity,
            sampler=reverb.selectors.Uniform(),
            remover=reverb.selectors.Fifo(),
            rate_limiter=reverb.rate_limiters.MinSize(1),
            signature=self.replay_buffer_signature)

        self.reverb_server = reverb.Server([self.table])

        self.replay_buffer = reverb_replay_buffer.ReverbReplayBuffer(
            tf_agent.collect_data_spec,
            table_name=self.table_name,
            sequence_length=None,
            local_server=self.reverb_server)

        self.rb_observer = reverb_utils.ReverbAddEpisodeObserver(
            self.replay_buffer.py_client,
            self.table_name,
            self.replay_buffer_capacity
        )
        
    def collect_episode(self,environment, policy, num_episodes):
        driver = py_driver.PyDriver(
            environment,
            py_tf_eager_policy.PyTFEagerPolicy(
            policy, use_tf_function=True),
            [self.rb_observer],
            max_episodes=num_episodes)
        initial_time_step = environment.reset()
        driver.run(initial_time_step)


