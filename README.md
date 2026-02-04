# Parallel Implementation of Value Iteration Algorithm for Machine Learning

This repository contains the code for the Parallel Computing Laboratory (3 CFU).

## Task to Parallelize
One basic approach to solve a problem of Reinforcement Learning (RL) is the algorithm of Value Iteration. 
It can be used when the world state is finite, and small enough to be explicitly represented in memory.

Value iteration converges to the optimal Value function applying in a synchronous way the Bellman Optimality Equations,
given any possible start.
Due to the synchronous update, two arrays for the Value function is need: one that contains the current result, and another that contains the result ov the iteration.
This update strategy makes the result easy to parallelize.

However, Asynchronous Value Iteration that updates only the value of one state, it's preferred. 
In fact, even if it requires more iterations, it takes less to converge.
Also, it needs only one array, increasing the maximum world usable.
However, this type of update has data flow dependencies, and it's harder to parallelize.

In this project work, there will be presented several parallel version with the goal to see what can get better performance.

### GIL Free environment
To reproduce the python environment used in this project work, it can be reproduced using the `requirements.txt` file.
Due to some issued not resolved, Joblib is not reported correctly. So, it's possible that the code will crash due to a not proper installation of the `distutils`.
This can ben easily be resolved updating Joblib using pip:
```commandline
pip install --upgrade joblib
```

## Run the Experiments
All the tests done in this experiments can be simply done executing `main.py` in a GIL free environment.
```commandline
python main.py
```

**Attention:** tests will require several hours.

## Prints results
The `plot_results.ipynb` notebook is used to plot all the results. To generate the graphs present inside the report, they are inside the `results.zip` file.
