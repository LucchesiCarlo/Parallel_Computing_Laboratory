from src import solvers

def process_test(size, threads, seed):
    solvers.solve_problem("results/sequential.csv", size, seed = seed)
    solvers.process_solve_problem("results/process.csv", size, threads = threads, seed = seed)
