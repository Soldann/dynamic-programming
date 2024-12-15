"""
 utils.py

 Helper functions that are used in multiple files. Feel free to add more functions.

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

# APPROACH:
# Computing the transition probabilities and the expected stage cost require
# very similar computations. To make the code more efficient, we have decided
# to do one computation for both P and Q. compute_transition_probabilities and
# compute_expected_stage_costs() then only return the result that was pre-
# computed. P and Q are actually calculated here, in the lower section of this
# file. 

import numpy as np
from Constants import Constants
import multiprocessing
import os
import ctypes

# P and Q store the results computed in this file. CachedConstants is used
# to check if the Constants passed to compute_transition_probabilities()
# and compute_expected_stage_costs() are the same.
P = None
Q = None
CachedConstants = None

def bresenham(start, end):
    """
    Generates the coordinates of a line between two points using Bresenham's algorithm.

    Parameters:
        start (tuple or list): The starting point (x0, y0).
        end (tuple or list): The ending point (x1, y1).

    Returns:
        List[Tuple[int, int]]: A list of (x, y) coordinates.

    Example:
        >>> bresenham((2, 3), (10, 8))
        [(2, 3), (3, 4), (4, 4), (5, 5), (6, 6), (7, 6), (8, 7), (9, 7), (10, 8)]
    """
    x0, y0 = start
    x1, y1 = end

    points = []

    dx = x1 - x0
    dy = y1 - y0

    x_sign = 1 if dx > 0 else -1 if dx < 0 else 0
    y_sign = 1 if dy > 0 else -1 if dy < 0 else 0

    dx = abs(dx)
    dy = abs(dy)

    if dx > dy:
        xx, xy, yx, yy = x_sign, 0, 0, y_sign
    else:
        dx, dy = dy, dx
        xx, xy, yx, yy = 0, y_sign, x_sign, 0

    D = 2 * dy - dx
    y = 0

    for x in range(dx + 1):
        px = x0 + x * xx + y * yx
        py = y0 + x * xy + y * yy
        points.append((px, py))
        if D >= 0:
            y += 1
            D -= 2 * dx
        D += 2 * dy

    return points

def idx2state(idx):
    """Converts a given index into the corresponding state.

    Args:
        idx (int): index of the entry whose state is required

    Returns:
        np.array: (x,y,x,y) state corresponding to the given index
    """
    state = np.empty(4)

    for i, j in enumerate(
        [
            Constants.M,
            Constants.N,
            Constants.M,
            Constants.N,
        ]
    ):
        state[i] = idx % j
        idx = idx // j
    return state


def state2idx(state):
    """Converts a given state into the corresponding index.

    Args:
        state (np.array): (x,y,x,y) entry in the state space

    Returns:
        int: index corresponding to the given state
    """
    idx = 0

    factor = 1
    for i, j in enumerate([Constants.M, Constants.N, Constants.M, Constants.N]):
        idx += state[i] * factor
        factor *= j

    return idx

def idx2state_with_constant(idx, Constants):
    """Converts a given index into the corresponding state.

    Args:
        idx (int): index of the entry whose state is required

    Returns:
        np.array: (x,y,x,y) state corresponding to the given index
    """
    state = np.empty(4)

    for i, j in enumerate(
        [
            Constants.M,
            Constants.N,
            Constants.M,
            Constants.N,
        ]
    ):
        state[i] = idx % j
        idx = idx // j
    return state


def state2idx_with_constant(state, Constants):
    """Converts a given state into the corresponding index.

    Args:
        state (np.array): (x,y,x,y) entry in the state space

    Returns:
        int: index corresponding to the given state
    """
    idx = 0

    factor = 1
    for i, j in enumerate([Constants.M, Constants.N, Constants.M, Constants.N]):
        idx += state[i] * factor
        factor *= j

    return idx

##### ---------------------------------------------------------- #####
##### -------------------- COMPUTE P AND Q --------------------- #####
##### ---------------------------------------------------------- #####

class CacheableConstants:
    """
    Class to compare the attributes of the passed Constants class. If all the
    attributes are the same, P and Q do not have to be re-computed. If they
    are not, a new computation will be triggered.
    """

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

def process_state(state):
    """
    Function used to get P and Q for each individual state. Is called by 
    ComputeValuesParallel()
    """
    global Constants

    global P
    global Q

    starting_state = idx2state_with_constant(state, Constants).astype(np.int32)

    robot_x, robot_y, swan_x, swan_y = starting_state

    if (robot_x == swan_x and robot_y == swan_y # if we have collided with the swan
        or np.any((Constants.DRONE_POS == [robot_x, robot_y]).all(axis=1)) # or if we have collided with a static drone
        or robot_x == Constants.GOAL_POS[0] and robot_y == Constants.GOAL_POS[1]): # or we have reached the goal state
        Q[state, :] = 0
        return

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
            P[state, ending_idx, input] += (1 - current_prob) * (1 - Constants.SWAN_PROB)

    
        #  Case 2: No current, swan moves
        if (robot_out_of_bounds
            or path_to_end_hit_drone
            or new_robot_x == potential_swan_move_x and new_robot_y == potential_swan_move_y):
            # The drone has gone off the edge or hit another drone/swan, so we reset
            b = 1
        else:
            ending_idx = state2idx_with_constant([new_robot_x, new_robot_y, potential_swan_move_x, potential_swan_move_y], Constants)
            P[state, ending_idx, input] += (1 - current_prob) * (Constants.SWAN_PROB)

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
            P[state, ending_idx, input] += (current_prob) * (1 - Constants.SWAN_PROB)
        
        # Case 4: Current, swan

        if (robot_out_of_bounds
            or path_to_end_hit_drone
            or new_robot_x == potential_swan_move_x and new_robot_y == potential_swan_move_y):
            # The drone has gone off the edge or hit another drone/swan, so we reset
            d =  1
        else:
            ending_idx = state2idx_with_constant([new_robot_x, new_robot_y, potential_swan_move_x, potential_swan_move_y], Constants)
            P[state, ending_idx, input] += (current_prob) * (Constants.SWAN_PROB)
        
        # Set value in case of crash:
        crash_value = a*(1-current_prob)*(1-Constants.SWAN_PROB) + b*(1-current_prob)*(Constants.SWAN_PROB) + c*(current_prob)*(1-Constants.SWAN_PROB) + d*(current_prob)*(Constants.SWAN_PROB)
        P[state, :, input].reshape((Constants.M, Constants.N, Constants.M, Constants.N), order='F')[Constants.START_POS[0], Constants.START_POS[1],:] += np.full((Constants.M, Constants.N), crash_value/(Constants.M*Constants.N-1))

        # Reset value where swan starts at start position
        ending_idx = state2idx_with_constant([*Constants.START_POS, *Constants.START_POS], Constants)
        P[state, ending_idx, input] = 0
        thruster_val = Constants.INPUT_SPACE[input]
        Q[state, input] = Constants.TIME_COST + Constants.THRUSTER_COST*(np.sum(np.abs(thruster_val))) + crash_value*Constants.DRONE_COST


def initialize_worker(shared_P_array, shared_Q_array, new_consts):
    """
        Initializes the workers with shared variables for process_state
    """
    global Constants
    Constants = new_consts
    global P
    global Q
    P = np.frombuffer(shared_P_array, dtype=np.float32).reshape((Constants.K, Constants.K, Constants.L))
    Q = np.frombuffer(shared_Q_array, dtype=np.float32).reshape((Constants.K, Constants.L))


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

    shared_P_array = multiprocessing.RawArray(ctypes.c_float, Constants.K * Constants.K * Constants.L)
    shared_Q_array = multiprocessing.RawArray(ctypes.c_float, [np.inf] * Constants.K * Constants.L)
    P = np.ctypeslib.as_array(shared_P_array).reshape((Constants.K, Constants.K, Constants.L))
    Q = np.ctypeslib.as_array(shared_Q_array).reshape((Constants.K, Constants.L))

    with multiprocessing.Pool(os.cpu_count(), initializer=initialize_worker, initargs=(shared_P_array, shared_Q_array, CachedConstants)) as cpu_pool:

        results = cpu_pool.map(process_state, range(Constants.K))
        # TODO fill the transition probability matrix P here
        cpu_pool.close()
        cpu_pool.join()
