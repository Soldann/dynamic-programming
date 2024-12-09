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
from collections import deque

def get_neighbours() -> np.array:
    """
    For the BFS jumpstart: get a np.array that shows what the neighbours 
    of points are
    
    ### Returns
    - np.array() dtype int
        - Each of the K rows represents a current postiion x_k. Each row
          contains 4 elements that are the approximate neighbours: nodes
          x_k-1 that could have been used to transition to node x_k. When
          these nodes would be outside the grid or they are occupied by 
          a statinoary drone, np.nan is inserted.
    """

    # An array of shape (n*m x 8): each row stands for a drone state, the 
    # elements of each row are the indices of the 8 drone states that are
    # neighbours of this drone state. When there are less than 8 neighbours
    # they are padded with np.nan

    # Create a grid of indices
    indices = np.arange(Constants.N * Constants.M).reshape(Constants.N, Constants.M)
    # Create padded array with -1 on the borders
    padded_indices = np.pad(indices, pad_width=1, mode='constant', constant_values=-1)
    # Collect the 8 neighbors using slicing
    neighbours = np.stack([
        padded_indices[:-2, 1:-1],
        padded_indices[1:-1, :-2], padded_indices[1:-1, 2:],
        padded_indices[2:, 1:-1],
    ], axis=-1).reshape(-1, 4)
    # replace -1 by np.nan:
    neighbours = np.where(neighbours == -1, np.nan, neighbours)

    # The grid positions with stationary drones should not be available neighbours for 
    # any grid position
    drone_pos = np.ravel_multi_index(Constants.DRONE_POS.T, (Constants.N, Constants.M))
    neighbours = np.where(np.isin(neighbours, drone_pos), np.nan, neighbours)

    return neighbours


def bfs_policy() -> np.array: 
    """
    Run a bfs traversal and return a jumpstart policy neglecting the sawn, 
    the current and the stochastic transitions

    ### Returns
    - np.array, dtype int
        - A map from the current state to the control input that will
          (in an unweighted shortest path problem) lead the path to the 
          goal
    """

    # Initialise datastructure for BFS traversal
    g = np.ravel_multi_index(Constants.GOAL_POS, (Constants.N, Constants.M))
    visited = np.zeros(Constants.N * Constants.M).astype("bool")
    visited[g] = True       # Boolean array: True where index was visited
    queue = deque((g,))     # The queue as a deque initialised with the goal
                            # Lookup table for the policy (init -1):
    policy = -1 * np.ones(Constants.N * Constants.M)
    neighbours = get_neighbours()
    actions = np.array([1, 5, 3, 7])  # Actions S, R, L, N

    # BFS traversal
    while bool(queue):      # While the queue is not empty

        i = queue.pop()     # The current position index i

        # Setting the policy's action in the neighbour nodes s.t. the drone
        # moves towards the BFS parent
        nan_mask = ~np.isnan(neighbours[i])  # Mask of genuine neighbours
                                             # Genuine neighbours of i
        i_dash = neighbours[i][nan_mask].astype("int")
        a = actions[nan_mask]                # Genuine actions at i
        nv_mask = ~visited[i_dash]           # Mask of non-visited i_dash
        i_dash_nv = i_dash[nv_mask]          # Non-visited i_dash
        a_nv = a[nv_mask]                    # Actions for the i_dash_nv
        policy[i_dash_nv] = a_nv             # Hence: actions at those indices

        # Marking the neighbours as visited
        visited[i_dash_nv] = True

        # Inserting the non-visited neighbours into the queue
        queue.extendleft(i_dash_nv)

    return policy


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

    # Value iteration vanilla
    J_opt_new = np.min(Q + np.sum(P * J_opt.reshape(1, -1, 1), axis=1), axis=1)
    i = 0
    while not np.allclose(J_opt, J_opt_new, rtol=1e-5, atol=1e-8):
        J_opt = J_opt_new
        J_opt_new = np.min(Q + np.sum(P * J_opt.reshape(1, -1, 1), axis=1), axis=1)
        i += 1

    return J_opt, u_opt

if __name__ == "__main__":
    p = bfs_policy()
    print("Yippie!")