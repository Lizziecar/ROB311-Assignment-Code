# part2.py: Project 4 Part 2 script
#
# --
# Artificial Intelligence
# ROB 311
# Programming Project 4

import numpy as np
from mdp_env import mdp_env
from mdp_agent import mdp_agent


## WARNING: DO NOT CHANGE THE NAME OF THIS FILE, ITS FUNCTION SIGNATURE OR IMPORT STATEMENTS

"""
INSTRUCTIONS
-----------------
  - Complete the policy_iteration method below
  - Please write abundant comments and write neat code
  - You can write any number of local functions
  - More implementation details in the Function comments
"""

def all_action_summation(env, agent, s):
  # Summation formula in the bellman update formula
  bell_upt = np.zeros([len(env.actions), 1])

  for action in env.actions:
    value = 0

    for s_to in env.states:
      
      value += env.transition_model[s, s_to, action] * agent.utility[s_to]
    
    bell_upt[action] = value
  
  return bell_upt

def policy_action_summation(env, agent, s, policy):
  # Summation formula in the bellman update formula for particular policy

  bell_value = 0

  for s_to in env.states:
    bell_value += env.transition_model[s,s_to, policy[s]] * agent.utility[s_to]

  return bell_value

def policy_evaluation(env, agent, s, policy):
  utility = env.rewards[s]

  value = 0
  for s_to in env.states:
    value += env.transition_model[s, s_to, policy[s]] * agent.utility[s_to]
  
  utility += agent.gamma*value

  return utility


def policy_iteration(env: mdp_env, agent: mdp_agent, max_iter = 1000) -> np.ndarray:
    """
    policy_iteration method implements VALUE ITERATION MDP solver,
    shown in AIMA (4ed pg 657). The goal is to produce an optimal policy
    for any given mdp environment.

    Inputs-
        agent: The MDP solving agent (mdp_agent)
        env:   The MDP environment (mdp_env)
        max_iter: Max iterations for the algorithm

    Outputs -
        policy: A list/array of actions for each state
                (Terminal states can have any random action)
        <agent>  Implicitly, you are populating the utlity matrix of
                the mdp agent. Do not return this function.
    """
    np.random.seed(1) # TODO: Remove this

    policy = np.zeros(len(env.states)).astype(int)
    agent.utility = np.zeros([len(env.states), 1])

    ## START: Student code
    for i in range(max_iter): 
      for s in env.states:
        agent.utility[s] = policy_evaluation(env, agent, s, policy)

      unchanged = True

      for s in env.states:
        if np.max(all_action_summation(env, agent, s)) > policy_action_summation(env, agent, s, policy):
          policy[s] = np.argmax(all_action_summation(env, agent, s))
          unchanged = False
      
      if unchanged:
        break

    ## END: Student code
    return policy