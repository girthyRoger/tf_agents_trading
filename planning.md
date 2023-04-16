# Planning
## Environment

### Observations 
The observation will include:
- Close Price
- Change in price from last step
- Cash
- Share value

The observation space will include the observations from the last 5 timesteps.

### Rewards and Goal

The goal of the agent is to maximise reward from selling long positions. The reward is calculated by checking the total portfolio value at any given step.

- Total Portfolio value = reward

### Actions
The agent will have buy, sell, and hold actions.

### Step
Each step, the environment will:
- Perform an action
- Calculate new observables based on the action
- Calculate the reward
- Advance the tick by a day
- Return the new observation and reward.

### Results
We will store the results in a format that we can save.

