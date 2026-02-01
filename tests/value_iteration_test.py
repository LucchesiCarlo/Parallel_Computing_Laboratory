import unittest
import numpy as np
from src import grid_world as gw
from src import value_iteration as vi
from tests import nan_equality


class ValueIterationTest(unittest.TestCase):

    def setUp(self):
        self.world = gw.GridWorld(4, 3)
        self.world.reset_map()

        self.world.set_element(1, 1, gw.Cell.Wall)
        self.world.set_element(0, 3, gw.Cell.Goal)
        self.world.set_element(1, 3, gw.Cell.Trap)

        self.input = self.world.copy_map().flatten()
        self.output = np.zeros_like(self.input)

    def test_value_iteration_sync(self):
        supposed_output = np.array([0, 0, 0.76, 1,
                                          0, np.NaN, 0, -1,
                                          0, 0, 0, 0] , dtype = float)
        vi.sync_optimality_bellman(self.input, self.output, self.world)

        for i in range(12):
            self.assertTrue(nan_equality(self.output[i], supposed_output[i]), msg = f"Bellman Optimality Equations are not well implemented (error in index {i}).")  # add assertion here

    def test_value_iteration_sync_two_spet(self):
        supposed_output = np.array([0, 0.5776, 0.8322, 1,
                                          0, np.NaN, 0.4826, -1,
                                          0, 0, 0, 0] , dtype = float)
        vi.sync_optimality_bellman(self.input, self.output, self.world)
        vi.sync_optimality_bellman(self.output, self.input, self.world)

        for i in range(12):
            self.assertTrue(nan_equality(self.input[i], supposed_output[i]), msg = f"Bellman Optimality Equations are not well implemented (error in index {i}).")

if __name__ == '__main__':
    unittest.main()
