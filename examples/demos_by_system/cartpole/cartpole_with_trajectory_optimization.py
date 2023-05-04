#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  4 10:01:30 2023

@author: alex
"""

###############################################################################
import numpy as np
###############################################################################
from pyro.dynamic   import cartpole
from pyro.analysis  import costfunction
from pyro.planning  import trajectoryoptimisation
###############################################################################


# sys
sys  = cartpole.CartPole()

sys.xbar[1] = np.pi # Up-right position

#Max/Min torque
sys.u_ub[0] = +20
sys.u_lb[0] = -20


# Cost function
cf  = costfunction.QuadraticCostFunction.from_sys( sys )

cf.Q[0,0] = 1.0
cf.Q[0,0] = 10.0
cf.R[0,0] = 100

planner = trajectoryoptimisation.DirectCollocationTrajectoryOptimisation( sys , dt= 0.2, grid = 20 )


planner.x_start = np.array([0,0,0,0])
planner.x_goal  = np.array([0,np.pi,0,0])

planner.maxiter = 500
planner.compute_optimal_trajectory()
planner.show_solution()
planner.animate_solution()