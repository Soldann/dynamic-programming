"""
 Solver.py

 Python function template to solve the stochastic
 shortest path problem.

 Dynamic Programming and Optimal Control
 Fall 2024
 Programming Exercise

 Contact: Antonio Terpin aterpin@ethz.ch

 Authors: Maximilian Stralz, Philip Pawlowsky, Antonio Terpin

 --
 ETH Zurich
 Institute for Dynamic Systems and Control
 --
"""

import numpy as np
from utils import *


def solution(P, Q, Constants):
    """Computes the optimal cost and the optimal control input for each
    state of the state space solving the stochastic shortest
    path problem by:
            - Value Iteration;
            - Policy Iteration;
            - Linear Programming;
            - or a combination of these.

    Args:
        P  (np.array): A (K x K x L)-matrix containing the transition probabilities
                       between all states in the state space for all control inputs.
                       The entry P(i, j, l) represents the transition probability
                       from state i to state j if control input l is applied
        Q  (np.array): A (K x L)-matrix containing the expected stage costs of all states
                       in the state space for all control inputs. The entry G(i, l)
                       represents the cost if we are in state i and apply control
                       input l
        Constants: The constants describing the problem instance.

    Returns:
        np.array: The optimal cost to go for the stochastic SPP
        np.array: The optimal control policy for the stochastic SPP

    """

    J_opt = np.zeros(Constants.K)
    u_opt = np.zeros(Constants.K)

    value_func = np.zeros(Constants.K)
    while True:
        max_diff = 0
        for state in range(len(value_func)):
            newval = np.min(Q[state,:] + np.sum(P[state,:,:]*np.tile(value_func,(9,1)).T, axis=0))
            diff = np.abs(newval - value_func[state])
            max_diff = max(diff, max_diff)
            value_func[state] = newval

        if max_diff <= 1e-5:
            break
    
    J_opt = value_func
    # TODO implement Value Iteration, Policy Iteration,
    #      Linear Programming or a combination of these

    return J_opt, u_opt
