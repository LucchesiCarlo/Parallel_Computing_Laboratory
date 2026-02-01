import time

import numpy as np
from src import grid_world as gw
from src import value_iteration as vi
from src import utils
from src.utils import save_on_csv


def solve_problem(output_file, size, wall_ratio = 0.05):
    EPSILON = 0.0001

    world = gw.GridWorld(size, size)
    world.randomize(num_walls= round((size**2)* wall_ratio))

    input = world.copy_map().flatten()
    output = np.zeros_like(input)

    start_time = time.perf_counter()
    while utils.nan_norm(input, output) > EPSILON:
        #print(nan_norm(input, output))
        vi.sync_optimality_bellman(input, output, world)
        input, output = output, input

    end_time = time.perf_counter()
    time_taken = end_time - start_time
    print("Completion Time: ", time_taken)
    print("Termination meet with: ", utils.nan_norm(input, output))

    save_on_csv(output_file, time_taken, 1, len(input[~np.isnan(input)]))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    solve_problem("output.csv", 10)
