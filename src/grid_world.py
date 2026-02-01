import numpy as np
from enum import Enum

class Cell(float, Enum):
    Free = 0
    Goal = 1
    Trap = -1
    Wall = np.NaN

class Action(Enum):
    UP, DOWN, LEFT, RIGHT = range(4)


class GridWorld(object):

    def __init__(self, width, height, side_prob: float = 0.1):
        self.map = None
        self.width = width
        self.height = height
        if side_prob > 0.5:
            side_prob = 0.1
        self.side_prob = side_prob
        self.reset_map()

    def randomize(self, num_walls = 0, num_traps = 1, num_goal = 1, seed = None):
        if num_walls + num_traps + num_goal > self.width * self.height :
            return None

        if seed is not None:
            np.random.seed(seed)
        self.map = np.zeros((self.height, self.width), dtype=float)

        for i in range(num_walls):
            x = np.random.randint(0, self.width - 1)
            y = np.random.randint(0, self.height - 1)
            while self.map[y][x] != 0:
                x = np.random.randint(0, self.width - 1)
                y = np.random.randint(0, self.height - 1)
            self.map[y][x] = Cell.Wall.value

        for i in range(num_traps):
            x = np.random.randint(0, self.width - 1)
            y = np.random.randint(0, self.height - 1)
            while self.map[y][x] != 0:
                x = np.random.randint(0, self.width - 1)
                y = np.random.randint(0, self.height - 1)
            self.map[y][x] = Cell.Trap.value

        for i in range(num_goal):
            x = np.random.randint(0, self.width - 1)
            y = np.random.randint(0, self.height - 1)
            while self.map[y][x] != 0:
                x = np.random.randint(0, self.width - 1)
                y = np.random.randint(0, self.height - 1)
            self.map[y][x] = Cell.Goal.value

        return None

    def set_element(self, row, col, cell: Cell):
        if 0 <= row < self.height and 0 <= col < self.width:
            self.map[row][col] = cell.value

    def reset_map(self):
        self.map = np.zeros((self.height, self.width), dtype=float)

    def state_reward(self, row, col):
        if row < 0 or row >= self.height or col < 0 or col >= self.width:
            return Cell.Wall.value
        else:
            return self.map[row][col]

    def markov_transition(self, row, col, action):
        distribution = {}
        if self.state_reward(row, col) == Cell.Goal.value or self.state_reward(row, col) == Cell.Trap.value:
            return distribution

        main_move = self.__apply_action(row, col, action)
        if action == Action.UP or action == Action.DOWN:
            side_move_1 = self.__apply_action(row, col, Action.LEFT)
            side_move_2 = self.__apply_action(row, col, Action.RIGHT)
        else:
            side_move_1 = self.__apply_action(row, col, Action.UP)
            side_move_2 = self.__apply_action(row, col, Action.DOWN)

        distribution[main_move] = 0
        distribution[side_move_1] = 0
        distribution[side_move_2] = 0
        distribution[(row, col)] = 0

        partial_prob = 0

        if not np.isnan(self.state_reward(side_move_1[0], side_move_1[1])):
            distribution[side_move_1] = self.side_prob
        else:
            distribution[(row, col)] += self.side_prob

        if not np.isnan(self.state_reward(side_move_2[0], side_move_2[1])):
            distribution[side_move_2] = self.side_prob
        else:
            distribution[(row, col)] += self.side_prob

        partial_prob += 2 * self.side_prob

        if not np.isnan(self.state_reward(main_move[0], main_move[1])):
            distribution[main_move] = 1 - partial_prob
        else:
            distribution[(row, col)] += 1 - partial_prob

        return distribution

    @staticmethod
    def __apply_action(row, col, action):
        if action == Action.UP:
            row -= 1
        if action == Action.DOWN:
            row += 1
        if action == Action.LEFT:
            col -= 1
        if action == Action.RIGHT:
            col += 1
        return row, col

    def copy_map(self):
        return np.copy(self.map)