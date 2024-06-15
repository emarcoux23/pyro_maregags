# -*- coding: utf-8 -*-

from abcd1234 import f
import numpy as np

q = (0, 0, 0, 0, 0, 0)
r_expected = (0.000, 0.039, 0.518)

#q = (30, 28.583, 11.745, 299.627, 20.978, 0.01)
#r_expected = (0.063, 0.098, 0.587)

q = [np.deg2rad(q[0]), 
     np.deg2rad(q[1]), 
     np.deg2rad(q[2]), 
     np.deg2rad(q[3]), 
     np.deg2rad(q[4]), 
     q[5]]

r = f(q)

# Formatting r
r = tuple(f"{value:.3f}" for value in r)
r = r_float = tuple(float(value) for value in r)

print("r expected:   " + str(r_expected))
print("r calculated: " + str(r))