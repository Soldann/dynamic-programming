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
import heapq

# Remove before submitting:
from line_profiler import profile
from matplotlib import pyplot as plt
COMPARISON_GRAPHICS = False  # Plot the jumpstarted vs the final value function


def get_neighbours(use: str) -> np.array:
    """
    For the BFS jumpstart: get a np.array that shows what the neighbours 
    of points are

    ### Params
    - use : str
        - One of ["bfs", "dijkstra", "vi"]. For "bfs", the four immediate 
          neighbours are returned. For "dijkstra" 8 neighbours are returned 
          (incl. the diagonals). For "vi" the central node is also returned
    
    ### Returns
    - np.array() dtype int
        - Each of the K rows represents a current postiion x_k. Each row
          contains 4 or 8 elements that are the approximate neighbours: 
          nodes x_k-1 that could have been used to transition to node x_k. 
          When these nodes would be outside the grid or they are occupied by 
          a statinoary drone, np.nan is inserted.
    """

    assert use in ("bfs", "dijkstra", "vi") 

    # An array of shape (n*m x 8) or (n*m x 4): each row stands for a drone 
    # state, the elements of each row are the indices of the 8 drone states 
    # that are neighbours of this drone state. When there are less than 8 
    # neighbours they are padded with np.nan

    # Create a grid of indices
    indices = np.arange(Constants.N * Constants.M).reshape(Constants.N, Constants.M)
    # Collect the 8 or 4 neighbors using slicing
    # Create padded array with -1 on the borders
    padded_indices = np.pad(indices, pad_width=1, mode='constant', constant_values=-1)
    if use == "bfs":
        neighbours = np.stack([
            padded_indices[:-2, 1:-1],
            padded_indices[1:-1, :-2], padded_indices[1:-1, 2:],
            padded_indices[2:, 1:-1],
        ], axis=-1).reshape(-1, 4)
    elif use == "dijkstra":
        neighbours = np.stack([
            padded_indices[:-2, :-2], padded_indices[:-2, 1:-1], padded_indices[:-2, 2:],
            padded_indices[1:-1, :-2], padded_indices[1:-1, 2:],
            padded_indices[2:, :-2], padded_indices[2:, 1:-1], padded_indices[2:, 2:]
        ], axis=-1).reshape(-1, 8)
    elif use == "vi":
        neighbours = np.stack([
            padded_indices[:-2, :-2], padded_indices[:-2, 1:-1], padded_indices[:-2, 2:],
            padded_indices[1:-1, :-2], padded_indices[1:-1, 1:-1], padded_indices[1:-1, 2:],
            padded_indices[2:, :-2], padded_indices[2:, 1:-1], padded_indices[2:, 2:]
        ], axis=-1).reshape(-1, 9)
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
    neighbours = get_neighbours("bfs")
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

# TODO: extend Dijkstra to deal with Kamikaze problem (transitions to start)
def dijkstra() -> np.array:
    """
    Run a best-first (dijkstra) traversal and return a jumpstart policy
    neglecting the swan, the current and the stochastic transitions.
    Using the actual thruster, time, and drone costs gives a good approxi-
    mation of the value function of this policy (later used by
    jumpstart_v()). 

    ### Returns
    - np.array, dtype int
        - A map from the current state to the control input that will
          (in an unweighted shortest path problem) lead the path to the 
          goal
    """

    ### --- INITIALISE DATASTRUCTURES --- ###

    # For Dijkstra traversal:
    g = np.ravel_multi_index(np.flip(Constants.GOAL_POS),     # goal node 
                             (Constants.N, Constants.M))
    start = np.ravel_multi_index(np.flip(Constants.START_POS),
                             (Constants.N, Constants.M))
    visited = np.zeros(Constants.N * Constants.M, dtype=bool)
    cost = np.empty(Constants.N * Constants.M, dtype=float)   # cost to go
    cost[:] = np.inf
    cost[g] = 0.
    policy = np.zeros(Constants.N * Constants.M, dtype=int)
    empty_queue = False                                       # break loop
    visited[g] = True                                         # visited nodes
    neighbours = get_neighbours("dijkstra")

    # For computing actions and costs
    actions = np.array([8, 7, 6, 5, 3, 2, 1, 0])      # actions per neighbour
    c_diag = (Constants.TIME_COST                     # cost diagonal motion
              + 2 * Constants.THRUSTER_COST)
    c_straight = (Constants.TIME_COST                 # cost straight motion
                  + Constants.THRUSTER_COST)
    motion_cost = np.array([c_diag, c_straight,       # cost array: cost of
                            c_diag, c_straight,       # corresponding actions
                            c_straight, c_diag,       # from the actions array
                            c_straight, c_diag], dtype=float)
    motion_drone_cost = (motion_cost                  # Cost when also sinking
                         + Constants.DRONE_COST)
    
    ### ------ INITIALISE PRIORITY QUEUE ------ ###

    # A heapified list of entries. Each entry is a list. entry[0]: cost;
    # entry[1]: nodeID; entry[2]: "p" for "present" or "r" for "removed"
    entry_0 = [0., g, "p"]
    queue = [entry_0]
    # Array for retrieving the mutible entry (list) based on the node ID (its
    # index). Also denotes what nodes are in the priority queue
    entry_finder = np.empty(Constants.N * Constants.M, dtype=object)
    entry_finder[g] = entry_0

    ### ------ BEST-FIRST TRAVERSAL ------ ###

    while bool(queue):  # while the queue is not empty

        ## --- POP AN ENTRY --- ##

        entry = heapq.heappop(queue)
        while entry[-1] != "p":  # current entry has actually been removed
            if bool(queue):      # queue non-empty: pop
                entry = heapq.heappop(queue)
            else:                # empty queue: break
                empty_queue = True
                break
        if empty_queue:          # empty cueue: break outer loop too
            break
        i = entry[1]             # the node ID (cell index)
        visited[i] = True        # popped from queue means visited
        entry_finder[i] = None   # remove popped nodes from dict

        ## --- PROCESS AN ENTRY wHEN NOT THE STARTING NODE --- ##

        # if i != start:

        # The genuine neighbours i_d of node i:
        nan_mask = ~np.isnan(neighbours[i])  # Mask of genuine neighbours
                                                # True where not nan
        i_d = neighbours[i][nan_mask].astype("int")  # neighbour idx
        i_d_motion_cost = motion_cost[nan_mask]      # cost getting there
        a = actions[nan_mask]                # Genuine actions at i

        # Which of these genuine neighbours have not yet been visited?
        nv_mask = ~visited[i_d]              # Mask of non-visited i_d

        # Did we find a cheaper way too any neighbour i_d?
        cheaper_mask = (cost[i] + i_d_motion_cost < cost[i_d])

        # Only when both are true would we like to update these neighbours
        update_mask = nv_mask & cheaper_mask
        i_to_update = i_d[update_mask]
        a_to_update = a[update_mask]
        c_to_update = i_d_motion_cost[update_mask]

        """
        i_d_nv = i_d[nv_mask]                # Non-visited i_dash
        a_nv = a[nv_mask]                    # Actions to the i_dash_nv
        cost_nv = i_d_motion_cost[nv_mask]   # ...cost of those actions

        # Did we find a cheaper way to reach any of the i_d_nv?
        cheaper_mask = (cost_nv + cost[i] < cost[i_d_nv])
        i_to_update = i_d_nv[cheaper_mask]
        a_to_update = a_nv[cheaper_mask]
        c_to_update = cost_nv[cheaper_mask] + cost[i]  # new cost: stage
                                                        # cost + parent c.
        """

        # Update the policy and cost of those neighbours i_to_update that
        # have not yet been visited and whose cost-to-go is reduced:
        policy[i_to_update] = a_to_update
        cost[i_to_update] = cost[i] + c_to_update

        # Book-keeping: 
        #  - for those i_to_update that are alreay in the priority queue:
        #    update their cost-to-go and heapify
        #  - for those i_to_update that are newly discovered: add them to
        #    the queue and heapify
        for j, new_c in zip(i_to_update, c_to_update):
            
            # When in queue: invalidate entry by chaning "p" to "r"
            if entry_finder[j] is not None:
                entry_finder[j][-1] = "r"

            # In any case: add new entry to the queue:
            q_entry = [new_c, j, "p"]
            heapq.heappush(queue, q_entry)
            entry_finder[j] = q_entry
                
        # else:  # We are dealing with the last node
    
    # Remove the inf values in the final cost values:
    cost = np.where(cost == np.inf, 0, cost)

    return policy, cost


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
    start = np.ravel_multi_index(np.flip(Constants.START_POS),
                                 (Constants.N, Constants.M))

    ### --- POLICY EVALUATION OF JUMPSTART --- ###

    # Find (approximate) value function to the policy provided by the
    # jumpstart graph traversal.
    jumpstart_policy, jumpstart_v = dijkstra()
    nm = Constants.N * Constants.M
    J_opt =  np.tile(jumpstart_v, nm)

    # If desired: visually compare the jumpstart V and the final V.
    if COMPARISON_GRAPHICS:
        J_init = J_opt.copy()

    ### --- FOR LOOP VALUE ITERATION --- ###

    # Advantages for loop: (1) asynchronous value iteration, (2) for each
    # state, consider only those next states than can actually be reached

    # Finding the possible next states of the drone
    drone_neighbours = get_neighbours("vi")          # Neighbours in range 1
    flow = Constants.FLOW_FIELD.reshape(-1, 2, order="F")  # flattened flow f.
    x, y = np.meshgrid(np.arange(Constants.M), np.arange(Constants.N))
    xy = np.stack((x.reshape(-1), y.reshape(-1)), axis=1)
    flow_drone_xy = xy + flow                        # Drone pos. with current
    flow_drone_i = np.ravel_multi_index((flow_drone_xy[:, 1], 
                                         flow_drone_xy[:, 0]), 
                                         (Constants.N, Constants.M),
                                         mode="clip")  # Flattened c. position
    all_neigh = np.hstack((drone_neighbours, 
                          drone_neighbours[flow_drone_i], 
                          np.c_[[start] * nm]))        # Possible next pos.
    np.nan_to_num(all_neigh, copy=False, nan=start) 
    all_neigh = np.array([np.unique(all_neigh[i]).astype(int)  # Remove repe-
                          for i in range(nm)], dtype=object)   # ted next pos.
    all_neigh_sw = np.array([np.hstack([all_neigh[i] + nm * j  # Allow any 
                                        for j in range(nm)])   # next swan p.
                             for i in range(nm)], dtype=object)

    # Accelerate value lookup:
    nn = [all_neigh_sw[i % nm] for i in range(Constants.K)]
    pp = [P[i, nn[i], :] for i in range(Constants.K)]

    # first iteration
    J_opt_new = J_opt.copy()
    for i in range(Constants.K):  # for each state
        J_opt_new[i] = np.min(Q[i] + J_opt_new[np.newaxis, nn[i]] @ pp[i])
        
    # VI until convergence:
    j = 0
    while not np.allclose(J_opt, J_opt_new, rtol=1e-5, atol=1e-8):
        J_opt = J_opt_new.copy()
        for i in range(Constants.K):  # for each state
            J_opt_new[i] = np.min(Q[i] + J_opt_new[np.newaxis, nn[i]] @ pp[i])
        j += 1
    
    
    # Find the optimal policy that belongs to this optimal vlaue function
    u_opt = np.empty(Constants.K, dtype=int)
    for i in range(Constants.K):
        u_opt[i] = np.argmin(Q[i] + J_opt_new[np.newaxis, nn[i]] @ pp[i])

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
    from ComputeExpectedStageCosts import compute_expected_stage_cost
    from ComputeTransitionProbabilities import compute_transition_probabilities
    P = compute_transition_probabilities(Constants)
    Q = compute_expected_stage_cost(Constants)
    solution(P, Q, Constants)