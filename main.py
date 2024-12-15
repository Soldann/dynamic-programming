"""
 main.py

 Python script that calls all the functions for computing the optimal cost
 and policy of the given problem.

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

import sys, os

import numpy as np
from ComputeExpectedStageCosts import compute_expected_stage_cost
from ComputeTransitionProbabilities import compute_transition_probabilities
from Constants import Constants
from Solver import solution

if __name__ == "__main__":
    # Set the following to True as you progress with the files
    TRANSITION_PROBABILITIES_IMPLEMENTED = True
    STAGE_COSTS_IMPLEMENTED = True
    SOLUTION_IMPLEMENTED = True

    # Compute transition probabilities
    if TRANSITION_PROBABILITIES_IMPLEMENTED:
        sys.stdout.write("[ ] Computing transition probabilities...")

        # TODO implement this function in ComputeTransitionProbabilities.py
        P = compute_transition_probabilities(Constants)

        print("\r[X] Transition probabilities computed.     ")
    else:
        print(
            "[ ] Transition probabilities not implemented. If this is unexpected, check the boolean 'TRANSITION_PROBABILITIES_IMPLEMENTED'."
        )
        P = np.zeros((Constants.K, Constants.K, Constants.L))

    # Compute expected stage costs
    if STAGE_COSTS_IMPLEMENTED:
        sys.stdout.write("[ ] Computing expected stage costs...")

        # TODO implement this function in ComputeExpectedStageCosts.py
        Q = compute_expected_stage_cost(Constants)

        print("\r[X] Expected stage costs computed.            ")
    else:
        print(
            "[ ] Expected stage costs not implemented. If this is unexpected, check the boolean 'STAGE_COSTS_IMPLEMENTED'."
        )
        Q = np.ones((Constants.K, Constants.L)) * np.inf

    # Solve the stochastic shortest path problem
    if SOLUTION_IMPLEMENTED:
        sys.stdout.write("[ ] Solving discounted stochastic shortest path problem...")

        # TODO implement this function in Solver.py
        J_opt, u_opt = solution(P, Q, Constants)

        assert J_opt.shape[0] == Constants.K, "J_opt dimensions do not match the world."
        assert u_opt.shape[0] == Constants.K, "u_opt dimensions do not match the world."

        print("\r[X] Discounted stochastic shortest path problem solved.    ")
    else:
        print(
            "[ ] Solution of the discounted stochastic shortest path problem not implemented. If this is unexpected, check the boolean 'SOLUTION_IMPLEMENTED'."
        )
        J_opt = np.inf * np.ones(Constants.K)
        u_opt = np.zeros(Constants.K)

    os.makedirs("workspaces", exist_ok=True)
    np.save("workspaces/J_opt.npy", J_opt)
    np.save("workspaces/u_opt.npy", (u_opt).astype(int))