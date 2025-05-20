import numpy as np
from scipy.interpolate import interp1d

eps = 1e-8

def sound_speed(p, rho, gamma: float = 1.4):
    return np.sqrt(gamma * np.maximum(p, eps) / np.maximum(rho, eps))

def enthalpy(p, rho, gamma: float = 1.4):
    return gamma/(gamma-1) * p / np.maximum(rho, eps)

def interp_y(field, yg,y_val):
    f = interp1d(yg, field, kind='quadratic',
                 fill_value='extrapolate', assume_sorted=True)
    return float(f(y_val))
