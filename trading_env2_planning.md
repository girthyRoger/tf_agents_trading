# Planning for trading env 2

## Overview
This trading env is going to be structured as follows.

### Observations
The observation space will use a larger representation of the price data. The price data will be vectorised as follows:

Obs = (date, open, high, low, close, volume, position, unrealised_gain)

Position is a fraction of the 


### Actions
Actions will take the following shape

Action = (1) with min -1 and max 1. 

The action will be the potential position