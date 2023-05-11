from trading_envs import Action, TradingEnv
from tf_agents.environments import utils
from tf_agents.utils import common
from tf_agents.environments import tf_py_environment

from REINFORCE import REINFORCE
from replay_buffer import ReplayBuffer

import pandas as pd
import matplotlib.pyplot as plt

def main():
    # import data
    df = pd.read_csv("data/QAN.csv")

    # Clean data
    df["Date"] = pd.to_datetime(df["Date"])
    print(df.dtypes)
    df.set_index("Date", inplace=True)
    df.sort_index(ascending=True, inplace=True)
        
    # hyperparams
    num_eval_episodes = 10 # @param {type:"integer"}
    num_iterations = 250 # @param {type:"integer"}
    collect_episodes_per_iteration = 2 # @param {type:"integer"}
    log_interval = 25 # @param {type:"integer"}
    eval_interval = 50 # @param {type:"integer"}
    
    train_env_py = TradingEnv(df=df, df_boundary=(10,100), window_size=5, starting_cash=100)
    
    utils.validate_py_environment(train_env_py, episodes=5)
    
    train_env = tf_py_environment.TFPyEnvironment(train_env_py)
    eval_env = tf_py_environment.TFPyEnvironment(TradingEnv(df=df, df_boundary=(99,129), window_size=5, starting_cash=100)) 
    
    print(train_env.time_step_spec())
    
    reinforce = REINFORCE(
        observation_spec=train_env.observation_spec(), 
        action_spec=train_env.action_spec(),
        time_step_spec=train_env.time_step_spec()
        )
    
    buffer = ReplayBuffer(tf_agent=reinforce.tf_agent)
    
    # reinforce.tf_agent.train() = common.function(reinforce.tf_agent.train())
    
    # Reset the train step
    reinforce.tf_agent.train_step_counter.assign(0)
    
    # evaluate agent policy once before training
    avg_return = reinforce.compute_average_return(environment=eval_env, num_episodes=num_eval_episodes)
    returns = [avg_return]
    
    for _ in range(num_iterations):
        buffer.collect_episode(
            environment=train_env,
            policy=reinforce.tf_agent.collect_policy,
            num_episodes=collect_episodes_per_iteration
        )
        iterator = iter(buffer.replay_buffer.as_dataset(sample_batch_size=1))
        trajectories, _ = next(iterator)
        train_loss = reinforce.tf_agent.train(experience=trajectories)
        
        buffer.replay_buffer.clear()
        
        step = reinforce.tf_agent.train_step_counter.numpy()
        
        if step % log_interval == 0:
            print(f"Step: {step}: loss: {train_loss.loss}")
        
        if step % eval_interval == 0:
            avg_return = reinforce.compute_average_return(environment=eval_env, num_episodes=num_eval_episodes)
            print(f"Step: {step}: Average Return: {avg_return}")
            returns.append(avg_return)
            
    steps = range(0, num_iterations + 1, eval_interval)
    plt.plot(steps, returns)
    plt.ylabel('Average Return')
    plt.xlabel('Step')
    plt.ylim(top=250)


    
    




if __name__ == "__main__":
    main()