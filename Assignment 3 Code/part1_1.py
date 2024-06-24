# part1_1.py: Project 4 Part 1 script
#
# --
# Artificial Intelligence
# ROB 311
# Programming Project 4

import numpy as np
from mdp_cleaning_task import cleaning_env

## WARNING: DO NOT CHANGE THE NAME OF THIS FILE, ITS FUNCTION SIGNATURE OR IMPORT STATEMENTS

"""
INSTRUCTIONS
-----------------
  - Complete the method  get_transition_model which creates the
    transition probability matrix for the cleanign robot problem desribed in the
    project document.
  - Please write abundant comments and write neat code
  - You can write any number of local functions
  - More implementation details in the Function comments
"""

def get_transition_model(env: cleaning_env) -> np.ndarray:
    """
    get_transition_model method creates a table of size (SxSxA) that represents the
    probability of the agent going from s1 to s2 while taking action a
    e.g. P[s1,s2,a] = 0.5
    This is the method that will be used by the cleaning environment (described in the
    project document) for populating its transition probability table

    Inputs
    --------------
        env: The cleaning environment

    Outputs
    --------------
        P: Matrix of size (SxSxA) specifying all of the transition probabilities.
    """

    P = np.zeros([len(env.states), len(env.states), len(env.actions)])

    ## START: Student Code
    # Terminal states can be left as zero since they end

    # L = 1
    # R = -1

    for state_from in env.states:
      if state_from != 0 and state_from != 5:
        for state_to in env.states:
          for action in env.actions:
            if action == 0 and state_from - state_to == 1:
              P[state_from, state_to, action] = 0.8
            if action == 1 and state_from - state_to == -1:
              P[state_from, state_to, action] = 0.8
            if action == 0 and state_from - state_to == -1:
              P[state_from, state_to, action] = 0.05
            if action == 1 and state_from - state_to == 1:
              P[state_from, state_to, action] = 0.05

            # Same state:
            if state_from == state_to:
              P[state_from, state_to, action] = 0.15

    ## END: Student code
    return P