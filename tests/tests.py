import unittest

from tests.grid_world_test import GridWorldTest
from tests.value_iteration_test import ValueIterationTest


def suite():
    suite = unittest.TestSuite()
    suite.addTest(GridWorldTest())
    suite.addTest(ValueIterationTest())
    return suite

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())