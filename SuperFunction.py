import numpy as np
import multiprocessing
import os
import ctypes
import copy
from utils import idx2state, state2idx, bresenham

P = None
Q = None

def initialize_worker(shared_P_array, shared_Q_array, shape_P, shape_Q):
    global P
    global Q
    # global Constants
    # Constants = consts
    print(shape_P)
    P = np.frombuffer(shared_P_array, dtype=np.float64).reshape(shape_P)
    Q = np.frombuffer(shared_Q_array, dtype=np.float64).reshape(shape_Q)


def process_state(state, Constants):
    global P
    global Q

    # P = P.reshape((Constants.K, Constants.K, Constants.L))
    # Q = Q.reshape((Constants.K, Constants.L))

    starting_state = idx2state(state).astype(np.int32)

    robot_x, robot_y, swan_x, swan_y = starting_state

    if (robot_x == swan_x and robot_y == swan_y # if we have collided with the swan
        or np.any((Constants.DRONE_POS == [robot_x, robot_y]).all(axis=1)) # or if we have collided with a static drone
        or robot_x == Constants.GOAL_POS[0] and robot_y == Constants.GOAL_POS[1]): # or we have reached the goal state
        # print("Skipping:")
        # print(robot_x, robot_y)
        # print(Constants.DRONE_POS)
        # print(swan_x, swan_y)
        Q[state,:] = 0
        return

    w_current_x, w_current_y = Constants.FLOW_FIELD[robot_x, robot_y]
    current_prob = Constants.CURRENT_PROB[robot_x,robot_y]

    swan_theta = np.arctan2(robot_y - swan_y, robot_x - swan_x)
    # print("theta", swan_theta)
    # print(swan_x, swan_y)
    # print(robot_x, robot_y)
    # print(robot_y - swan_y, robot_x - swan_x)
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

    # print(potential_swan_move_x, potential_swan_move_y)

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
            # P[state, :, input].reshape((Constants.M, Constants.N, Constants.M, Constants.N), order='F')[Constants.START_POS[0], Constants.START_POS[1],:]+= np.full((Constants.M, Constants.N), 1/(Constants.M*Constants.N-1)) * (1 - current_prob) * (1 - Constants.SWAN_PROB)
            a = 1
        else:
            ending_idx = state2idx([new_robot_x, new_robot_y, swan_x, swan_y])
            P[state, ending_idx, input] += (1 - current_prob) * (1 - Constants.SWAN_PROB)
            # print("adding 1", [new_robot_x, new_robot_y, swan_x, swan_y])
        
        # print(P[state, :, input].reshape((Constants.M, Constants.N, Constants.M, Constants.N))[new_robot_x, new_robot_y,:])
        # print(np.sum(P[state, :, input]))
        
        # print((1 - current_prob) * (1 - Constants.SWAN_PROB))
        # assert np.sum(P[state, :, input]) == (1 - current_prob) * (1 - Constants.SWAN_PROB)
        # thing = bresenham([robot_x, robot_y], [new_robot_x,new_robot_y])

        # print(thing)
        # print(Constants.DRONE_POS[:, None])
        # print(Constants.DRONE_POS[:, None].shape)
        # print(np.array(thing).shape)
        # print("INtersect", ([[1, 0], [3, 0], [0, 0]] == Constants.DRONE_POS[:]))
        # print("INtersect", )
    
        #  Case 2: No current, swan moves
        if (robot_out_of_bounds
            or path_to_end_hit_drone
            or new_robot_x == potential_swan_move_x and new_robot_y == potential_swan_move_y):
            # The drone has gone off the edge or hit another drone/swan, so we reset
            b = 1
            # P[state, :, input].reshape((Constants.M, Constants.N, Constants.M, Constants.N), order='F')[Constants.START_POS[0], Constants.START_POS[1],:]+= np.full((Constants.M, Constants.N), 1/(Constants.M*Constants.N-1)) * (1 - current_prob) * (Constants.SWAN_PROB)
        else:
            ending_idx = state2idx([new_robot_x, new_robot_y, potential_swan_move_x, potential_swan_move_y])
            P[state, ending_idx, input] += (1 - current_prob) * (Constants.SWAN_PROB)
            # print("adding 2", [new_robot_x, new_robot_y, potential_swan_move_x, potential_swan_move_y])

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
            # P[state, :, input].reshape((Constants.M, Constants.N, Constants.M, Constants.N), order='F')[Constants.START_POS[0], Constants.START_POS[1],:]+= np.full((Constants.M, Constants.N), 1/(Constants.M*Constants.N-1)) * (current_prob) * (1 - Constants.SWAN_PROB)
        else:
            ending_idx = state2idx([new_robot_x, new_robot_y, swan_x, swan_y])
            P[state, ending_idx, input] += (current_prob) * (1 - Constants.SWAN_PROB)
            # print("adding 3", [new_robot_x, new_robot_y, swan_x, swan_y])
        
        # Case 4: Current, swan

        if (robot_out_of_bounds
            or path_to_end_hit_drone
            or new_robot_x == potential_swan_move_x and new_robot_y == potential_swan_move_y):
            # The drone has gone off the edge or hit another drone/swan, so we reset
            d =  1
            # P[state, :, input].reshape((Constants.M, Constants.N, Constants.M, Constants.N), order='F')[Constants.START_POS[0], Constants.START_POS[1],:]+= np.full((Constants.M, Constants.N), 1/(Constants.M*Constants.N-1)) * (current_prob) * (Constants.SWAN_PROB)
        else:
            ending_idx = state2idx([new_robot_x, new_robot_y, potential_swan_move_x, potential_swan_move_y])
            P[state, ending_idx, input] += (current_prob) * (Constants.SWAN_PROB)
            # print("adding 4", [new_robot_x, new_robot_y, potential_swan_move_x, potential_swan_move_y])

        
        # Set value in case of crash:
        crash_value = a*(1-current_prob)*(1-Constants.SWAN_PROB) + b*(1-current_prob)*(Constants.SWAN_PROB) + c*(current_prob)*(1-Constants.SWAN_PROB) + d*(current_prob)*(Constants.SWAN_PROB)
        P[state, :, input].reshape((Constants.M, Constants.N, Constants.M, Constants.N), order='F')[Constants.START_POS[0], Constants.START_POS[1],:] += np.full((Constants.M, Constants.N), crash_value/(Constants.M*Constants.N-1))

        # Reset value where swan starts at start position
        ending_idx = state2idx([*Constants.START_POS, *Constants.START_POS])
        P[state, ending_idx, input] = 0
        thruster_val = Constants.INPUT_SPACE[input]
        Q[state, input] = Constants.TIME_COST + Constants.THRUSTER_COST*(np.sum(np.abs(thruster_val))) + crash_value*Constants.DRONE_COST

        # print(P[state, :, input].reshape((Constants.M, Constants.N, Constants.M, Constants.N), order='F')[Constants.START_POS[0], Constants.START_POS[1],:])
        # print(crash_value / (Constants.M * Constants.N - 1))
        # print(np.sum(P[state, :, input]))

        # print(P[state, :, input][np.nonzero(P[state, :, input])])
        # print([idx2state(id) for id in np.argwhere(P[state, :, input])])
        # print((1 - current_prob)*(1 - Constants.SWAN_PROB), (1- current_prob)*(Constants.SWAN_PROB), (current_prob)*(1-Constants.SWAN_PROB), (current_prob)*(Constants.SWAN_PROB))
        # assert np.isclose(np.sum(P[state, :, input]), 1)

    # s = np.ravel_multi_index([10000000, 0, 0, 0], (Constants.M, Constants.N, Constants.M, Constants.N), mode='clip', order='F')
    # ravel = np.ravel_multi_index(starting_state, (Constants.M, Constants.N, Constants.M, Constants.N), mode='clip', order='F')
    # assert ravel == state

def ComputeValues(Constants):
    """
        Computes both transition probabilities and expected stage costs for a given problem
    """
    # if Constants == new_constants:
    #     return
    global P
    global Q
    # P = np.zeros((Constants.K, Constants.K, Constants.L))
    Q_initializer = np.ones((Constants.K, Constants.L)) * np.inf
    shared_P_array = multiprocessing.RawArray(ctypes.c_double, Constants.K * Constants.K * Constants.L)
    shared_Q_array = multiprocessing.RawArray(ctypes.c_double, Constants.K * Constants.L)
    P = np.ctypeslib.as_array(shared_P_array).reshape((Constants.K, Constants.K, Constants.L))
    Q = np.ctypeslib.as_array(shared_Q_array).reshape((Constants.K, Constants.L))
    np.copyto(Q, Q_initializer)
    with multiprocessing.Pool(os.cpu_count(), initializer=initialize_worker, initargs=(shared_P_array, shared_Q_array, (Constants.K, Constants.K, Constants.L), (Constants.K, Constants.L))) as cpu_pool:
        # cpu_pool = multiprocessing.Pool(1)

        cpu_pool.starmap(process_state, [(i, copy.deepcopy(Constants)) for i in range(Constants.K)])
        # TODO fill the transition probability matrix P here
        cpu_pool.close()
        cpu_pool.join()
