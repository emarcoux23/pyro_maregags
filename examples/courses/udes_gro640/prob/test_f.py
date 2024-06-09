# -*- coding: utf-8 -*-

from abcd1234 import f, dhs2T
import numpy as np

q = [0,0,0,0,0,0,0]

r = f( q )

print(r)

r =     (0.05,      0.08,       0.03,       0.04,       0.06)
d =     (0.03,      0.06,       0.01,       0.02,       0.04)
theta = (np.pi/3,   np.pi/2,    np.pi/4,    np.pi/2,    np.pi/4)
alpha = (np.pi/2,   np.pi/3,    np.pi/3,    np.pi/3,    np.pi/3)

print("")
print(dhs2T(r, d, theta, alpha))
print("")