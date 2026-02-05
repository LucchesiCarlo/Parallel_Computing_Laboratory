import csv

import numpy as np
import os

#Allows to calculate a distance base on the norm of vectors that has NaN values.
def nan_dist(a, b):
    return nan_norm(a - b)


#Allows to calculate the norm of vectors that has NaN values.
def nan_norm(c):
    return np.linalg.norm(c[~np.isnan(c)])

def save_on_csv(filename, time, threads, size):
    exists = os.path.exists(filename)

    with open(filename, 'a') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',',)
        if not exists:
            csvwriter.writerow(["time", "threads", "size"])
        csvwriter.writerow([time, threads, size])