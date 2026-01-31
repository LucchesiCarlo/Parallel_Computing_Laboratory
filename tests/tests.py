import unittest

from tests.grid_world_test import GridWorldTest


def suite():
    suite = unittest.TestSuite()
    suite.addTest(GridWorldTest())
    return suite

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())