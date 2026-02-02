from src import solvers
from src.utils import nan_norm

if __name__ == '__main__':
    size = 100
    seed = 671278
    reference = solvers.solve_problem("sequential.csv", size, seed = seed)
    #solvers.process_solve_problem("process.csv", size, threads = 4, seed = seed)
    #solvers.thread_solve_problem("thread.csv", size, threads = 4, seed = seed)

    result_1 = solvers.solve_problem_async("async.csv", size, reference, seed = seed, EPSILON = 0.01)
    result_2 = solvers.solve_problem_async_threads("async_locks.csv", size, reference, threads = 4, seed = seed, EPSILON = 0.01)
    result_3 = solvers.solve_problem_async_race("async_race.csv", size, reference, threads = 4, seed = seed, EPSILON = 0.01)

    print(f"Difference with locks: {nan_norm(result_1, result_2)}")
    print(f"Difference without locks: {nan_norm(result_1, result_3)}")