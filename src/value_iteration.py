import numpy as np

from src.grid_world import GridWorld, Action

#Different implementations of iteration for different Value Iteration algorithm

def sync_optimality_bellman(input: np.ndarray, output: np.ndarray, world: GridWorld, start: int = 0, end: int = -1, gamma: float = 0.95):
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

def async_optimality_bellman(input: np.ndarray, world: GridWorld, gamma: float = 0.95):
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

def async_optimality_bellman_locks(input: np.ndarray, world: GridWorld, locks, gamma: float = 0.95):
    i = np.random.randint(0, len(input))
    row = i // world.width
    col = i % world.width

    actions = [a for a in Action]
    values = np.zeros_like(actions, dtype = float)

    to_lock = [(0, -1), (-1, 0), (0, 0), (1, 0), (0, 1)]
    for x, y in to_lock:
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
    for x, y in to_lock:
            if (0 <= (y + row) < world.height) and (0 <= (x + col) < world.width):
                locks[(y + row) * world.width + x + col].release()

    return (i, input[i])