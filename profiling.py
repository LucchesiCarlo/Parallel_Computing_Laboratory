from src import solvers
import cProfile

if __name__ == '__main__':
    size = 100
    seed = 671278
    pr = cProfile.Profile()
    pr.enable()
    reference = solvers.solve_problem("profile.csv", size, seed = seed)
    pr.disable()
    pr.print_stats(sort="cumtime")
