from src import solvers

def threads_test(size, threads, seed):
    solvers.thread_solve_problem("results/thread.csv", size, threads = threads, seed = seed)
