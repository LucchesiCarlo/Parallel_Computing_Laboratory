import itertools

from src import solvers

if __name__ == '__main__':
    sizes = range(50, 101, 10)
    seed = 671278
    MAX_THREADS = 8
    threads = range(1, MAX_THREADS + 1)
    ITER = 6
    #This loop repetition makes the same code to be executed 6 time in a row,
    # so that every version should benefit from caching in the same way
    for size in sizes:
        reference = None
        for _ in range(ITER):
            reference = solvers.solve_problem("results/sequential.csv", size, seed = seed)
        for _ in range(ITER):
            solvers.solve_problem_async("results/async.csv", size, reference, seed=seed, EPSILON=0.01)
        for t in threads:
            for _ in range(ITER):
                solvers.joblib_solve_problem("results/thread_joblib.csv", size, threads = t, seed = seed, backend="threading")
            for _ in range(ITER):
                solvers.joblib_solve_problem("results/process.csv", size, threads = t, seed = seed)
            for _ in range(ITER):
                solvers.thread_solve_problem("results/thread.csv", size, threads = t, seed = seed)
            for _ in range(ITER):
                solvers.solve_problem_async_threads("results/async_locks.csv", size, reference, threads=t, seed=seed, EPSILON=0.01)
            for _ in range(ITER):
                solvers.solve_problem_async_race("results/async_race.csv", size, reference, threads=t, seed=seed, EPSILON=0.01)

