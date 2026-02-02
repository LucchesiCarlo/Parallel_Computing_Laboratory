from src import solvers

if __name__ == '__main__':
    size = 200
    seed = 671278
    reference = solvers.solve_problem("sequential.csv", size, seed = seed, EPSILON = 0.000001)
    #process_solve_problem("process.csv", size, threads = 4, seed = seed)
    #for t in range(1, 1):
    #thread_solve_problem("thread.csv", size, threads = 4, seed = seed)
    solvers.solve_problem_async("async_sequential.csv", size, reference, seed = seed, EPSILON = 0.001)
