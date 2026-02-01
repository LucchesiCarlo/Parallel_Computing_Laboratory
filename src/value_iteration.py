import numpy as np

from src.grid_world import GridWorld, Action


def sync_optimality_bellman(input: np.ndarray, output: np.ndarray, world: GridWorld, start = 0, end = -1, gamma = 0.95):
    if end == -1:
        end = len(input)
    for i in range(start, end):
        row = i // world.width
        col = i % world.width

        actions = [a for a in Action]
        values = np.zeros_like(actions, dtype = float)

        for j, action in enumerate(actions):
            distribution = world.markov_transition(row, col, action)
            reward = world.state_reward(row, col)
            for state, value in distribution.items():
                if value != 0:
                    reward += (value * input[state[0] * world.width + state[1]]) * gamma
            values[j] = reward

        output[i] = np.max(values)