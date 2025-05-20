import numpy as np

# Геометрия расширяющегося канала
L = 3.0
h0 = 1.0
h1 = 4.0

def top_wall(x):
    return h0 + (h1 - h0) * (x / L)

def wall_angle():
    return np.arctan((h1 - h0) / L)
