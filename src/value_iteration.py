import numpy as np

from src.grid_world import GridWorld, Action


def sync_optimality_bellman(input: np.ndarray, output: np.ndarray, world: GridWorld, start = 0, end = -1, gamma = 0.95):
    if end == -1 or end > len(input):
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

    return output[start:end]

def async_optimality_bellman(input: np.ndarray, world: GridWorld, gamma = 0.95):
    i = np.random.randint(0, len(input))
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

    input[i] = np.max(values)

    return (i, input[i])

def async_optimality_bellman_locks(input: np.ndarray, world: GridWorld, locks, gamma = 0.95):
    i = np.random.randint(0, len(input))
    row = i // world.width
    col = i % world.width

    actions = [a for a in Action]
    values = np.zeros_like(actions, dtype = float)

    for x in range(-1, 2):
        for y in range(-1, 2):
            if (0 <= (y + row) < world.height) and (0 <= (x + col) < world.width):
                locks[(y + row) * world.width + x + col].acquire()

    for j, action in enumerate(actions):
        distribution = world.markov_transition(row, col, action)
        reward = world.state_reward(row, col)
        for state, value in distribution.items():
            if value != 0:
                reward += (value * input[state[0] * world.width + state[1]]) * gamma
        values[j] = reward

    input[i] = np.max(values)
    for x in range(-1, 2):
        for y in range(-1, 2):
            if (0 <= (y + row) < world.height) and (0 <= (x + col) < world.width):
                locks[(y + row) * world.width + x + col].release()

    return (i, input[i])