import csv

import numpy as np
import os

def nan_norm(a, b):
    c = a - b
    return np.linalg.norm(c[~np.isnan(c)])

def save_on_csv(filename, time, threads, size):
    exists = os.path.exists(filename)

    with open(filename, 'a') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',',)
        if not exists:
            csvwriter.writerow(["time", "threads", "size"])
        csvwriter.writerow([time, threads, size])