"""
Author: Cody Duong
KUID: 3050266
Creation Date: 2024-11-26
Purpose:

This script implements the RL Policy Iteration and Value Iteration algorithms for a 5x5 Gridworld task.

Inputs:
- None, the parameters for the grid and iterations are defined in the script.

Outputs:
- Optimal policy arrays for Policy Iteration and Value Iteration.
- Plots showing error vs. iterations for both methods.
- Explanations for convergence methods.

Sources:
- ChatGPT (debugging)
"""

from typing import Any
import numpy as np
import matplotlib.pyplot as plt

# Constants for the Gridworld
GRID_SIZE = 5
TERMINAL_STATES = [(0, 0), (4, 4)]
ACTIONS = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # Left, Right, Up, Down
GAMMA = 1.0  # Discount factor
REWARD = -1
CONVERGENCE_THRESHOLD = 1e-4


def initialize_policy_value(grid_size) -> tuple[Any, Any]:
    """Initialize value and policy arrays."""
    values = np.zeros((grid_size, grid_size))
    policy = np.random.choice(len(ACTIONS), size=(grid_size, grid_size))
    return values, policy


def is_terminal(state):
    """Check if a state is terminal."""
    return state in TERMINAL_STATES


def policy_iteration() -> tuple[Any, Any, list[Any], int]:
    """Perform Policy Iteration."""
    values, policy = initialize_policy_value(GRID_SIZE)
    iteration = 0
    errors = []

    while True:
        # Policy Evaluation
        delta = 0
        new_values = values.copy()
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                if is_terminal((i, j)):
                    continue
                action = ACTIONS[policy[i, j]]
                next_i, next_j = i + action[0], j + action[1]
                if 0 <= next_i < GRID_SIZE and 0 <= next_j < GRID_SIZE:
                    reward = REWARD
                else:
                    next_i, next_j = i, j  # Stay in place if hitting a wall
                    reward = REWARD
                new_values[i, j] = reward + GAMMA * values[next_i, next_j]
                delta = max(delta, abs(new_values[i, j] - values[i, j]))
        values = new_values
        errors.append(delta)

        # Policy Improvement
        stable = True
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                if is_terminal((i, j)):
                    continue
                action_values = []
                for action in ACTIONS:
                    next_i, next_j = i + action[0], j + action[1]
                    if 0 <= next_i < GRID_SIZE and 0 <= next_j < GRID_SIZE:
                        reward = REWARD
                    else:
                        next_i, next_j = i, j
                        reward = REWARD
                    action_values.append(reward + GAMMA * values[next_i, next_j])
                best_action = np.argmax(action_values)
                if policy[i, j] != best_action:
                    stable = False
                policy[i, j] = best_action
        iteration += 1
        if stable or delta < CONVERGENCE_THRESHOLD:
            break
    return values, policy, errors, iteration


def value_iteration() -> tuple[Any, Any, list[Any], int]:
    """Perform Value Iteration."""
    values, _ = initialize_policy_value(GRID_SIZE)
    iteration = 0
    errors = []

    while True:
        delta = 0
        new_values = values.copy()
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                if is_terminal((i, j)):
                    continue
                action_values = []
                for action in ACTIONS:
                    next_i, next_j = i + action[0], j + action[1]
                    if 0 <= next_i < GRID_SIZE and 0 <= next_j < GRID_SIZE:
                        reward = REWARD
                    else:
                        next_i, next_j = i, j
                        reward = REWARD
                    action_values.append(reward + GAMMA * values[next_i, next_j])
                new_values[i, j] = max(action_values)
                delta = max(delta, abs(new_values[i, j] - values[i, j]))
        values = new_values
        errors.append(delta)
        iteration += 1
        if delta < CONVERGENCE_THRESHOLD:
            break

    # Extract policy from values
    policy = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            if is_terminal((i, j)):
                continue
            action_values = []
            for action in ACTIONS:
                next_i, next_j = i + action[0], j + action[1]
                if 0 <= next_i < GRID_SIZE and 0 <= next_j < GRID_SIZE:
                    reward = REWARD
                else:
                    next_i, next_j = i, j
                    reward = REWARD
                action_values.append(reward + GAMMA * values[next_i, next_j])
            policy[i, j] = np.argmax(action_values)

    return values, policy, errors, iteration


def plot_errors(errors, title) -> None:
    """Plot error vs. iterations."""
    plt.plot(errors)
    plt.title(title)
    plt.xlabel("Iterations")
    plt.ylabel("Error")
    plt.grid()
    plt.show()


def main() -> None:
    # Part 1: Policy Iteration
    pi_values, pi_policy, pi_errors, pi_iterations = policy_iteration()
    print("Policy Iteration:")
    print(f"Optimal Policy:\n{pi_policy}")
    print(f"Iterations: {pi_iterations}")
    plot_errors(pi_errors, "Policy Iteration Error vs. Iterations")

    # Part 2: Value Iteration
    vi_values, vi_policy, vi_errors, vi_iterations = value_iteration()
    print("Value Iteration:")
    print(f"Optimal Policy:\n{vi_policy}")
    print(f"Iterations: {vi_iterations}")
    plot_errors(vi_errors, "Value Iteration Error vs. Iterations")


if __name__ == "__main__":
    main()
