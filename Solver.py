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

# Remove before submitting:
from line_profiler import profile
from matplotlib import pyplot as plt
COMPARISON_GRAPHICS = True  # Plot the jumpstarted vs the final value function


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
    drone_pos = np.ravel_multi_index(np.flip(Constants.DRONE_POS, axis=1).T, 
                                     (Constants.N, Constants.M))
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
    g = np.ravel_multi_index(np.flip(Constants.GOAL_POS), (Constants.N, Constants.M))
    visited = np.zeros(Constants.N * Constants.M).astype("bool")
    visited[g] = True       # Boolean array: True where index was visited
    queue = deque((g,))     # The queue as a deque initialised with the goal
                            # Lookup table for the policy (init -1):
    policy = np.zeros(Constants.N * Constants.M, dtype=int)
    neighbours = get_neighbours()
    actions = np.array([7, 5, 3, 1], dtype=int)  # Actions N, E, W, S

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


def jumpstart_v(policy: np.array, p: np.array) -> np.array:
    """
    Create a jumpstart value function based on a jumpstart policy (BFS) by
    performing fixed-point policy evaluation. To reduce the state space, 
    swan position is not accounted for but all other aspectss of the problem
    are.

    ### Parameters
    - policy : 1D np.array of M*N elements
        - Each element specifies an integer in the input space. For undefned
          states (e.g. on the cell of static drones) the policy is 0
    - p : 3D np.array (K, K, L)
        - Transition probabilities

    ### Returns
    - 2D np.array (K, K)
        - Value function for every state. Across the swan states the value
          function will be identical
    """

    ### --- SET SWAN POSITION --- ###

    # To keep the jumpstart efficient, the swan's position is kept constant
    # while evaluating the jumpstart policy. It is positioned close to the 
    # goal node. The jumpstart value function for this swan position will be
    # copied for all swan positions; since the swan was close to the goal
    # node, the overestimation of the cost in that cell should be removable
    # with few iterations of value iteration.

    swan_range = 1  # how far the swan may be from the goal

    lower_n = max(0, Constants.GOAL_POS[1] - swan_range)
    upper_n = min(Constants.N - 1, Constants.GOAL_POS[1] + swan_range) + 1
    lower_m = max(0, Constants.GOAL_POS[0] - swan_range)
    upper_m = min(Constants.M - 1, Constants.GOAL_POS[0] + swan_range) + 1

    # Positioning the swan even when the goal in a corner / at the edge
    patch = np.zeros((upper_n - lower_n, upper_m - lower_m))
    patch[Constants.GOAL_POS[1] - lower_n, 
          Constants.GOAL_POS[0] - lower_m] = np.nan
    legal_pos = np.where(~np.isnan(patch))
    swan_pos = legal_pos[0][0] + lower_n, legal_pos[1][0] + lower_m
    swan_idx = np.ravel_multi_index(swan_pos, (Constants.N, Constants.M))

    # Getting the indices range of drone states with this swan state
    lower_drone_idx = swan_idx * Constants.N * Constants.M
    upper_drone_idx = (swan_idx + 1) * Constants.N * Constants.M

    ### --- POLICY EVALUATION --- ###

    # Select transition probabilities: constant swan position
    p_drone = p[lower_drone_idx:upper_drone_idx, 
                lower_drone_idx:upper_drone_idx, :]

    print("jo")
    

def full_v_jumpstart(policy: np.array, p: np.array, q: np.array) -> np.array:
    """
    As jumpstart v but taking account of the swan positions as well
    """

    # Transition probabilities according to pi
    # The same policy is applied regardless of the swan position
    nm = Constants.N * Constants.M
    pi = np.tile(policy, nm)
    p_pi = p[np.arange(nm**2), :, pi]  # select actions according to policy

    # Expected stage cost according to pi:
    q_pi = q[np.arange(nm**2), pi]     # select actions according to policy

    # Policy evalutation: initialisation
    v_opt = np.zeros(nm**2)
    v_opt_new = q_pi + p_pi @ v_opt
    i = 0
    while not np.allclose(v_opt, v_opt_new, rtol=1e-5, atol=1e-8):
        v_opt = v_opt_new
        v_opt_new = q_pi + p_pi @ v_opt
        i += 1

    return v_opt_new

@profile
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

    ### --- POLICY EVALUATION OF JUMPSTART --- ###

    # Find (approximate) value function to the policy provided by the
    # jumpstart graph traversal. Finds the value function only for one swan
    # position

    J_opt = full_v_jumpstart(bfs_policy(), P, Q)
    u_opt = np.zeros(Constants.M**2 * Constants.N**2, dtype=int)

    if COMPARISON_GRAPHICS:
        J_init = J_opt.copy()

    # Value iteration vanilla
    J_opt_new = np.min(Q + np.sum(P * J_opt.reshape(1, -1, 1), axis=1), axis=1)
    i = 0
    while not np.allclose(J_opt, J_opt_new, rtol=1e-5, atol=1e-8):
        J_opt = J_opt_new
        J_opt_new = np.min(Q + np.sum(P * J_opt.reshape(1, -1, 1), axis=1), axis=1)
        i += 1

    if COMPARISON_GRAPHICS:
        mn = Constants.M * Constants.N
        fig, axs = plt.subplots(nrows=1, ncols=2)
        vmin = min(J_init.min(), J_opt.min())
        vmax = max(J_init.max(), J_opt.max())
        ims = axs[0].imshow(J_init[:mn].reshape(Constants.N, Constants.M), 
                            vmin=vmin, vmax=vmax)
        axs[0].set_title("Jumpstart V")
        cax = fig.add_axes([0.2, 0.1, 0.6, 0.03])
        cbar = fig.colorbar(ims, cax=cax, orientation="horizontal")
        axs[1].imshow(J_opt[:mn].reshape(Constants.N, Constants.M), 
                      vmin=vmin, vmax=vmax)
        axs[1].set_title("Final V")
        plt.show()

    return J_opt, u_opt


if __name__ == "__main__":
    pass