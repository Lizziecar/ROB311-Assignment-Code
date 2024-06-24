# part1_2.py: Project 4 Part 1 script
#
# --
# Artificial Intelligence
# ROB 311
# Programming Project 4

import numpy as np
from mdp_env import mdp_env
from mdp_agent import mdp_agent

from part1_1 import get_transition_model
from mdp_cleaning_task import cleaning_env
from mdp_grid_task import grid_env
from mdp_agent import mdp_agent

## WARNING: DO NOT CHANGE THE NAME OF THIS FILE, ITS FUNCTION SIGNATURE OR IMPORT STATEMENTS

"""
INSTRUCTIONS
-----------------
  - Complete the value_iteration method below
  - Please write abundant comments and write neat code
  - You can write any number of local functions
  - More implementation details in the Function comments
"""

def value_iteration(env: mdp_env, agent: mdp_agent, eps: float, max_iter = 1000) -> np.ndarray:
    """
    value_iteration method implements VALUE ITERATION MDP solver,
    shown in AIMA (4ed pg 653). The goal is to produce an optimal policy
    for any given mdp environment.

    Inputs
    ---------------
        agent: The MDP solving agent (mdp_agent)
        env:   The MDP environment (mdp_env)
        eps:   Max error allowed in the utility of a state
        max_iter: Max iterations for the algorithm

    Outputs
    ---------------
        policy: A list/array of actions for each state
                (Terminal states can have any random action)
       <agent>  Implicitly, you are populating the utlity matrix of
                the mdp agent. Do not return this function.
    """
    policy = np.empty_like(env.states)
    agent.utility = np.zeros([len(env.states), 1])
    U_astrix = np.zeros([len(env.states), 1])

    ## START: Student code
    for i in range(max_iter): 
      agent.utility = U_astrix.copy()
      delta = 0

      for s in env.states:

        # Summation formula in the bellman update formula
        bell_upt = np.zeros([len(env.actions), 1])

        for action in env.actions:
          value = 0

          for s_to in env.states:
            value += env.transition_model[s,s_to, action] * U_astrix[s_to]
          
          bell_upt[action] = value
        
        bell_max = np.max(bell_upt)

        U_astrix[s] = env.rewards[s] + agent.gamma * bell_max

        if np.abs(U_astrix[s] - agent.utility[s]) > delta:
          delta = np.abs(U_astrix[s] - agent.utility[s])

      # Exit condition
      if delta < (eps*(1-agent.gamma))/agent.gamma:
        break
    
    for s in env.states:
      # Summation formula in the bellman update formula
      policy_upt = np.zeros([len(env.actions), 1])

      for action in env.actions:
        value = 0

        for s_to in env.states:
          value += env.transition_model[s,s_to, action] * U_astrix[s_to]
          
        policy_upt[action] = value
      
      policy[s] = np.argmax(policy_upt)

    ## END Student code
    return policy