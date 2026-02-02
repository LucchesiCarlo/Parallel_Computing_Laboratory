import time

import numpy as np
from src import grid_world as gw
from src import value_iteration as vi
from src import utils
from src.utils import save_on_csv
import concurrent.futures
from joblib import Parallel, delayed

def solve_problem(output_file, size, wall_ratio = 0.05, seed = None, EPSILON = 0.0001):
    start_time = time.perf_counter()

    world = gw.GridWorld(size, size)
    world.randomize(seed = seed, num_walls= round((size**2)* wall_ratio))

    input = world.copy_map().flatten()
    output = np.zeros_like(input)

    while utils.nan_norm(input, output) > EPSILON:
        vi.sync_optimality_bellman(input, output, world)
        input, output = output, input

    end_time = time.perf_counter()
    time_taken = end_time - start_time
    print("Completion Time: ", time_taken)
    print("Termination meet with: ", utils.nan_norm(input, output))

    save_on_csv(output_file, time_taken, 1, len(input[~np.isnan(input)]))

def thread_solve_problem(output_file, size, wall_ratio = 0.05, threads = 8, seed = None, EPSILON = 0.0001):
    start_time = time.perf_counter()

    world = gw.GridWorld(size, size)
    world.randomize(seed = seed, num_walls= round((size**2)* wall_ratio))

    input = world.copy_map().flatten()
    output = np.zeros_like(input)

    chunks = (len(input) // threads) + 1

    with concurrent.futures.ThreadPoolExecutor(threads) as executor:
        while utils.nan_norm(input, output) > EPSILON:
            results = [executor.submit(vi.sync_optimality_bellman, input, output, world, start = i * chunks, end = (i + 1) * chunks) for i in range(threads)]
            #thread_list = [threading.Thread(target = vi.sync_optimality_bellman, args = (input, output, world), kwargs = {"start" : i * chunks, "end": (i + 1) * chunks}) for i in range(threads)]
            for r in results:
                r.result()
            input, output = output, input

    end_time = time.perf_counter()
    time_taken = end_time - start_time
    print("Completion Time: ", time_taken)
    print("Termination meet with: ", utils.nan_norm(input, output))

    save_on_csv(output_file, time_taken, threads, len(input[~np.isnan(input)]))


def process_solve_problem(output_file, size, wall_ratio = 0.05, threads = 8, seed = None, EPSILON = 0.0001):
    start_time = time.perf_counter()

    world = gw.GridWorld(size, size)
    world.randomize(seed=seed, num_walls=round((size ** 2) * wall_ratio))

    input = world.copy_map().flatten()
    output = np.zeros_like(input)

    chunks = (len(input) // threads) + 1
    while utils.nan_norm(input, output) > EPSILON:
        args = (input, output, world)
        kwargs = [{"start": i * chunks, "end": (i + 1) * chunks} for i in range(threads)]
        results = Parallel(n_jobs=threads, prefer="threads")(delayed(vi.sync_optimality_bellman)(*args, **kwargs[i]) for i in range(threads))
        output = np.concatenate(results)
        input, output = output, input

    end_time = time.perf_counter()
    time_taken = end_time - start_time
    print("Completion Time: ", time_taken)
    print("Termination meet with: ", utils.nan_norm(input, output))

    save_on_csv(output_file, time_taken, threads, len(input[~np.isnan(input)]))


if __name__ == '__main__':
    size = 50
    seed = 671278
    solve_problem("sequential.csv", size, seed = seed)
    #process_solve_problem("process.csv", size, threads = 4, seed = seed)
    #for t in range(1, 1):
    thread_solve_problem("thread.csv", size, threads = 4, seed = seed)
