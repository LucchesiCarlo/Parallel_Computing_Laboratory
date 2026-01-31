import unittest
import numpy as np
from src import grid_world as gw

class GridWorldTest(unittest.TestCase):

    def setUp(self):
        self.world = gw.GridWorld(4, 3)
        self.world.reset_map()

        self.world.set_element(1, 1, gw.Cell.Wall)
        self.world.set_element(0, 3, gw.Cell.Goal)
        self.world.set_element(0, 2, gw.Cell.Trap)

    def test_cell_enum(self):
        self.assertAlmostEqual(gw.Cell.Goal.value, 1, msg="Goal cell should be 1")
        self.assertAlmostEqual(gw.Cell.Trap.value, -1, msg="Trap cell should be -1")
        self.assertAlmostEqual(gw.Cell.Free.value, 0, msg="Free cell should be 1")
        self.assertTrue(np.isnan(gw.Cell.Wall.value), msg="Wall cell should be Nan")

    def test_correct_world_set(self):
        self.assertAlmostEqual(self.world.state_reward(0,0), gw.Cell.Free.value, msg = "The map is not well initialized.")
        self.assertAlmostEqual(self.world.state_reward(0,3), gw.Cell.Goal.value, msg = "The map didn't set correctly the goal.")
        self.assertAlmostEqual(self.world.state_reward(0,2), gw.Cell.Trap.value, msg = "The map didn't set correctly the trap.")

        self.assertTrue(np.isnan(self.world.state_reward(1,1)), msg = "The map didn't set correctly the wall.")

    def test_markov_transition(self):
        distribution_1 = {
            (1, 2): 0.8,
            (2, 1): 0.1,
            (2, 3): 0.1,
            (2, 2): 0,
        }

        self.assertEqual(self.world.markov_transition(2, 2, gw.Action.UP), distribution_1, msg = "The markov transition density is wrong if these isn't any wall.")

    def test_markov_transition_wall(self):
        distribution_1 = {
            (1, 1): 0,
            (2, 0): 0.1,
            (2, 2): 0.1,
            (2, 1): 0.8,
        }
        self.assertEqual(self.world.markov_transition(2, 1, gw.Action.UP), distribution_1, msg = "The markov transition density is wrong if these is any wall.")

    def test_markov_transition_outside(self):
        distribution_1 = {
            (-1, 0): 0,
            (0, -1): 0,
            (0, 0): 0.9,
            (1, 0): 0.1,
        }
        self.assertEqual(self.world.markov_transition(0, 0, gw.Action.LEFT), distribution_1, msg = "The markov transition density is wrong if agent tries to escape the world.")

if __name__ == '__main__':
    unittest.main()
