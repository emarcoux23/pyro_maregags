#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 10 12:50:30 2022

@author: alex
"""

import numpy as np

from pyro.dynamic.rocket                  import Rocket
from pyro.analysis.costfunction           import QuadraticCostFunction
from pyro.dynamic.statespace              import linearize
from pyro.planning.trajectoryoptimisation import DirectCollocationTrajectoryOptimisation

# Non-linear model
sys = Rocket()

sys.inertia = 400

sys.xbar =  np.array([0,2.2,0,0,0,0])
sys.ubar =  np.array([1,0]) * sys.mass * sys.gravity # Nominal trust = weight

# Linear model
ss  = linearize( sys , 0.01 )
    

# Cost function
cf  = QuadraticCostFunction.from_sys( sys )

cf.Q[0,0] = 1
cf.Q[1,1] = 10000
cf.Q[2,2] = 0.1
cf.Q[3,3] = 0
cf.Q[4,4] = 10000
cf.Q[5,5] = 0

cf.R[0,0] = 0.01
cf.R[1,1] = 10.0

ss.cost_function = cf


planner = DirectCollocationTrajectoryOptimisation( ss )

planner.x_start = np.array([0,0,0,0,0,0])
planner.x_goal  = np.array([0,10,0,0,0,0])

planner.compute_optimal_trajectory()
planner.show_solution()
planner.animate_solution()