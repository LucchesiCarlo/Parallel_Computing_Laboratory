import numpy as np

def nan_equality(a, b):
    return round(a - b, 7) == 0 or (np.isnan(a)) and (np.isnan(b))