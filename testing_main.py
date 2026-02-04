from src import solvers

if __name__ == '__main__':
    size = 100
    seed = 671278

    threads = 8
    print("Sequential")
    reference = solvers.solve_problem("trash.csv", size, seed = seed)
    print("Joblib (Process)")
    solvers.process_solve_problem("trash.csv", size, threads = threads, seed = seed)
    print("Threads")
    solvers.thread_solve_problem("trash.csv", size, threads = threads, seed = seed)

    print("Async")
    solvers.solve_problem_async("trash.csv", size, reference, seed = seed, EPSILON = 0.01)
    print("Async Threads")
    solvers.solve_problem_async_threads("trash.csv", size, reference, seed = seed, EPSILON = 0.01)
    print("Async Threads (Locks)")
    solvers.solve_problem_async_race("trash.csv", size, reference, seed = seed, EPSILON = 0.01)
