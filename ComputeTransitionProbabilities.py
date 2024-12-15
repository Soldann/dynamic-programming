"""
 ComputeTransitionProbabilities.py

 Python function template to compute the transition probability matrix.

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
import multiprocessing
import os
import copy

# Upon calling either compute_transition_probabilities() or 
# compute_expected_stage_costs() the superfunction is called and computes
# bot P and Q. They are stored in these variables.
P = None
Q = None
CachedConstants = None

def compute_transition_probabilities(Constants):
    """Computes the transition probability matrix P.

    It is of size (K,K,L) where:
        - K is the size of the state space;
        - L is the size of the input space; and
        - P[i,j,l] corresponds to the probability of transitioning
            from the state i to the state j when input l is applied.

    Args:
        Constants: The constants describing the problem instance.

    Returns:
        np.array: Transition probability matrix of shape (K,K,L).
    """
    global P

    ComputeValuesParallel(Constants)
    return P

##### -------------------------------------------------------- #####
##### -------------------- SUPERFUNCTION --------------------- #####
##### -------------------------------------------------------- #####

class CacheableConstants:
        # Feel free to tweak these to test your solution.
    # ----- World -----

    def __init__(self, constants_to_cache):
        self.M = constants_to_cache.M
        self.N = constants_to_cache.N
        self.START_POS = constants_to_cache.START_POS.copy()
        self.GOAL_POS = constants_to_cache.GOAL_POS.copy()
        self.DRONE_POS = constants_to_cache.DRONE_POS.copy()
        self.K = constants_to_cache.K
        self.INPUT_SPACE = constants_to_cache.INPUT_SPACE.copy()
        self.L = constants_to_cache.L
        self.THRUSTER_COST = constants_to_cache.THRUSTER_COST
        self.TIME_COST = constants_to_cache.TIME_COST
        self.DRONE_COST = constants_to_cache.DRONE_COST
        self.SWAN_PROB = constants_to_cache.SWAN_PROB
        self.CURRENT_PROB = constants_to_cache.CURRENT_PROB.copy()
        self.FLOW_FIELD = constants_to_cache.FLOW_FIELD.copy()

    def __eq__(self, other):
        if not isinstance(other, CacheableConstants):
            return False
        return (self.M == other.M and
                self.N == other.N and
                np.array_equal(self.START_POS, other.START_POS) and
                np.array_equal(self.GOAL_POS, other.GOAL_POS) and
                np.array_equal(self.DRONE_POS, other.DRONE_POS) and
                self.K == other.K and
                np.array_equal(self.INPUT_SPACE, other.INPUT_SPACE) and
                self.L == other.L and
                self.THRUSTER_COST == other.THRUSTER_COST and
                self.TIME_COST == other.TIME_COST and
                self.DRONE_COST == other.DRONE_COST and
                self.SWAN_PROB == other.SWAN_PROB and
                np.array_equal(self.CURRENT_PROB, other.CURRENT_PROB) and
                np.array_equal(self.FLOW_FIELD, other.FLOW_FIELD))

def process_state(state, Constants):

    P = np.zeros((Constants.K, Constants.L))
    Q = np.ones((Constants.L)) * np.inf

    starting_state = idx2state_with_constant(state, Constants).astype(np.int32)

    robot_x, robot_y, swan_x, swan_y = starting_state

    if (robot_x == swan_x and robot_y == swan_y # if we have collided with the swan
        or np.any((Constants.DRONE_POS == [robot_x, robot_y]).all(axis=1)) # or if we have collided with a static drone
        or robot_x == Constants.GOAL_POS[0] and robot_y == Constants.GOAL_POS[1]): # or we have reached the goal state
        Q[:] = 0
        return P, Q

    w_current_x, w_current_y = Constants.FLOW_FIELD[robot_x, robot_y]
    current_prob = Constants.CURRENT_PROB[robot_x,robot_y]

    swan_theta = np.arctan2(robot_y - swan_y, robot_x - swan_x)
    if -np.pi / 8 <= swan_theta < np.pi / 8:
        potential_swan_move_x = swan_x + 1
        potential_swan_move_y = swan_y
    elif np.pi / 8 <= swan_theta < 3 * np.pi / 8:
        potential_swan_move_x = swan_x + 1
        potential_swan_move_y = swan_y + 1
    elif 3 * np.pi / 8 <= swan_theta < 5 * np.pi / 8:
        potential_swan_move_x = swan_x
        potential_swan_move_y = swan_y + 1
    elif 5 * np.pi / 8 <= swan_theta < 7 * np.pi / 8:
        potential_swan_move_x = swan_x - 1
        potential_swan_move_y = swan_y + 1
    elif swan_theta >= 7 * np.pi / 8   or swan_theta < -7 * np.pi / 8:
        potential_swan_move_x = swan_x - 1
        potential_swan_move_y = swan_y
    elif -7 * np.pi / 8 <= swan_theta < -5 * np.pi / 8:
        potential_swan_move_x = swan_x - 1
        potential_swan_move_y = swan_y - 1
    elif -5 * np.pi / 8 <= swan_theta < -3 * np.pi / 8:
        potential_swan_move_x = swan_x
        potential_swan_move_y = swan_y - 1
    elif -3 * np.pi / 8 <= swan_theta < -np.pi / 8:
        potential_swan_move_x = swan_x + 1
        potential_swan_move_y = swan_y - 1
    else:
        raise RuntimeError("Invalid angle between swan and robot")

    for input in range(Constants.L):
        control_x, control_y = Constants.INPUT_SPACE[input]

        new_robot_x = robot_x + control_x
        new_robot_y = robot_y + control_y

        a = 0
        b = 0
        c = 0
        d = 0
        # For any control input, there are 4 things that could happen

        # Case 1: No current, swan doesn't move

        robot_out_of_bounds = new_robot_x < 0 or new_robot_x >= Constants.M or new_robot_y < 0 or new_robot_y >= Constants.N

        path_to_end = bresenham((robot_x, robot_y), (new_robot_x, new_robot_y))
        path_to_end_hit_drone =  np.any((Constants.DRONE_POS[:, None] == path_to_end).all(axis=-1).any(axis=1))  # check if path to end intersects any drone position

        if (robot_out_of_bounds
            or path_to_end_hit_drone
            or new_robot_x == swan_x and new_robot_y == swan_y):
            # The drone has gone off the edge or hit another drone/swan, so we reset                
            a = 1
        else:
            ending_idx = state2idx_with_constant([new_robot_x, new_robot_y, swan_x, swan_y], Constants)
            P[ending_idx, input] += (1 - current_prob) * (1 - Constants.SWAN_PROB)

    
        #  Case 2: No current, swan moves
        if (robot_out_of_bounds
            or path_to_end_hit_drone
            or new_robot_x == potential_swan_move_x and new_robot_y == potential_swan_move_y):
            # The drone has gone off the edge or hit another drone/swan, so we reset
            b = 1
        else:
            ending_idx = state2idx_with_constant([new_robot_x, new_robot_y, potential_swan_move_x, potential_swan_move_y], Constants)
            P[ending_idx, input] += (1 - current_prob) * (Constants.SWAN_PROB)

        # Case 3: Current, swan doesn't move
        new_robot_x = robot_x + control_x + w_current_x
        new_robot_y = robot_y + control_y + w_current_y

        robot_out_of_bounds = new_robot_x < 0 or new_robot_x >= Constants.M or new_robot_y < 0 or new_robot_y >= Constants.N

        path_to_end = bresenham((robot_x, robot_y), (new_robot_x, new_robot_y))
        path_to_end_hit_drone =  np.any((Constants.DRONE_POS[:, None] == path_to_end).all(axis=-1).any(axis=1))  # check if path to end intersects any drone position
        
        if (robot_out_of_bounds
            or path_to_end_hit_drone
            or new_robot_x == swan_x and new_robot_y == swan_y):
            # The drone has gone off the edge or hit another drone/swan, so we reset        
            c = 1        
        else:
            ending_idx = state2idx_with_constant([new_robot_x, new_robot_y, swan_x, swan_y], Constants)
            P[ending_idx, input] += (current_prob) * (1 - Constants.SWAN_PROB)
        
        # Case 4: Current, swan

        if (robot_out_of_bounds
            or path_to_end_hit_drone
            or new_robot_x == potential_swan_move_x and new_robot_y == potential_swan_move_y):
            # The drone has gone off the edge or hit another drone/swan, so we reset
            d =  1
        else:
            ending_idx = state2idx_with_constant([new_robot_x, new_robot_y, potential_swan_move_x, potential_swan_move_y], Constants)
            P[ending_idx, input] += (current_prob) * (Constants.SWAN_PROB)
        
        # Set value in case of crash:
        crash_value = a*(1-current_prob)*(1-Constants.SWAN_PROB) + b*(1-current_prob)*(Constants.SWAN_PROB) + c*(current_prob)*(1-Constants.SWAN_PROB) + d*(current_prob)*(Constants.SWAN_PROB)
        P[:, input].reshape((Constants.M, Constants.N, Constants.M, Constants.N), order='F')[Constants.START_POS[0], Constants.START_POS[1],:] += np.full((Constants.M, Constants.N), crash_value/(Constants.M*Constants.N-1))

        # Reset value where swan starts at start position
        ending_idx = state2idx_with_constant([*Constants.START_POS, *Constants.START_POS], Constants)
        P[ending_idx, input] = 0
        thruster_val = Constants.INPUT_SPACE[input]
        Q[input] = Constants.TIME_COST + Constants.THRUSTER_COST*(np.sum(np.abs(thruster_val))) + crash_value*Constants.DRONE_COST

    return P, Q


def ComputeValuesParallel(Constants) -> None:
    """
        Computes both transition probabilities and expected stage costs for a 
        given problem. The function modifies the global variables P, Q, and
        CachedConstants. This way, if P or Q are requested for the same 
        Constants class again, no new computation has to be performed.
    """
    
    global CachedConstants
    constants_to_cache = CacheableConstants(Constants)
    if CachedConstants == constants_to_cache:
        return
    CachedConstants = constants_to_cache

    global P
    global Q

    with multiprocessing.Pool(os.cpu_count()) as cpu_pool:

        results = cpu_pool.starmap(process_state, [(i, constants_to_cache) for i in range(Constants.K)])
        # TODO fill the transition probability matrix P here
        P = np.array([result[0] for result in results])
        Q = np.array([result[1] for result in results])
        cpu_pool.close()
        cpu_pool.join()
