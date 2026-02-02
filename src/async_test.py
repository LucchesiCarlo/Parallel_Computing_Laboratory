from src import solvers

def async_test(size, threads, seed, reference):
        solvers.solve_problem_async("results/async.csv", size, reference, seed=seed, EPSILON=0.01)
        solvers.solve_problem_async_threads("results/async_locks.csv", size, reference, threads=threads, seed=seed,
                                                           EPSILON=0.01)
        solvers.solve_problem_async_race("results/async_race.csv", size, reference, threads=threads, seed=seed,
                                                        EPSILON=0.01)
