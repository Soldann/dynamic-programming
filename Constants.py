"""
 Constants.py

 Python script containg the definition of the class Constants
 that holds all the problem constants.

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

import itertools

import numpy as np

seed = 0  # Feel free to change it
np.random.seed(seed)


def generate_unique_coordinates(n, M, N):
    rng = np.random.default_rng(seed=seed)
    # Create a grid of all possible coordinates
    all_coords = np.array(np.meshgrid(np.arange(M), np.arange(N))).T.reshape(-1, 2)
    # Use the Generator.choice method to select n+2 unique coordinates without replacement
    selected_coords = rng.choice(
        all_coords, size=n, replace=False, axis=0, shuffle=True
    )

    return selected_coords[0], selected_coords[1], selected_coords[2:]


class Constants:
    # Feel free to tweak these to test your solution.
    # ----- World -----
    # State space constants
    M = 5  # Size of the x axis (west to east)
    N = 5  # Size of the y axis (south to north)

    # Map constants
    N_DRONES = 3  # Number of static drones
    # Generate unique coordinates for the start, goal and drone positions
    START_POS, GOAL_POS, DRONE_POS = generate_unique_coordinates(N_DRONES + 2, M, N)

    # State space
    _y = np.arange(0, N)
    _x = np.arange(0, M)
    STATE_SPACE = np.array(list(itertools.product(_y, _x, _y, _x)), dtype=int)[
        :, [3, 2, 1, 0]
    ]
    K = len(STATE_SPACE)

    # input space
    INPUT_SPACE = np.array(list(itertools.product([-1, 0, 1], repeat=2)), dtype=int)[
        :, [1, 0]
    ]
    L = len(INPUT_SPACE)

    # ----- Cost -----
    # Stage cost factors
    THRUSTER_COST = 1  # Cost of using one thruster
    TIME_COST = 10  # Cost of a time step
    DRONE_COST = 100  # Cost of sending a new drone

    # ----- Disturbances -----
    SWAN_PROB = 0.8
    CURRENT_PROB = np.random.uniform(0, 0.1, (M, N))  # Drift of the current
    FLOW_FIELD = np.random.choice(
        [-2, -1, 0, 1, 2], size=(M, N, 2)
    )  # Flow field of the current
