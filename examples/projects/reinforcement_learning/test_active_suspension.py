#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 22:27:47 2022

@author: alex
"""

import numpy as np

from pyro.dynamic  import suspension

import dynamic_programming as dprog
import discretizer
import costfunction

sys  = suspension.QuarterCarOnRoughTerrain()

sys.mass = 1.0
sys.b    = 2.0
sys.k    = 10.0

sys.vx = 1.0

# Set domain
sys.x_ub = np.array([+2, +2, +20])
sys.x_lb = np.array([-2, -2, +10])

sys.u_ub = np.array([+20])
sys.u_lb = np.array([-20])

# Discrete world
grid_sys = discretizer.GridDynamicSystem(sys, (51, 51, 51), [11], 0.05)

# Cost Function
qcf = costfunction.QuadraticCostFunction.from_sys(sys)

qcf.xbar = np.array([ 0 , 1.0, 20 ]) # target
qcf.INF  = 100000
qcf.EPS  = 0.5

qcf.Q[0,0] = 2.0
qcf.Q[1,1] = 500.0
qcf.Q[2,2] = 0.0

qcf.R[0,0] = 0.1

qcf.S[0,0] = 0.0
qcf.S[1,1] = 0.0
qcf.S[2,2] = 0.0

# DP algo
dp = dprog.DynamicProgrammingWithLookUpTable( grid_sys, qcf)

dp.alpha = 0.99
dp.solve_bellman_equation()


ctl = dp.get_lookup_table_controller()

cl_sys = ctl + sys

# Simulation and animation
cl_sys = ctl + sys
cl_sys.x0   = np.array([0,0,10])
cl_sys.compute_trajectory( 10, 10001, 'euler')
cl_sys.plot_trajectory('xu')
cl_sys.plot_phase_plane_trajectory()
cl_sys.animate_simulation()