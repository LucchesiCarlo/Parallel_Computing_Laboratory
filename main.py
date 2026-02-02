import itertools

from src import solvers
from src import async_test
from src import process_test
from src import threads_test

if __name__ == '__main__':
    sizes = range(50, 151, 25)
    seed = 671278
    MAX_THREADS = 8
    threads = range(1, MAX_THREADS + 1)
    ITER = 6
    for size in sizes:
        reference = solvers.solve_problem("results/reference.csv", size, seed = seed)
        for t in threads:
            for _ in range(ITER):
                process_test.process_test(size, t, seed)
                threads_test.threads_test(size, t, seed)
                async_test.async_test(size, t, seed, reference)
