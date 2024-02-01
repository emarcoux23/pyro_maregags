#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  3 08:27:06 2021

@author: alex
"""

import numpy as np

from boat2 import Boat2D

from pyro.analysis.costfunction import QuadraticCostFunction
from pyro.dynamic.statespace    import linearize
from pyro.control.lqr           import synthesize_lqr_controller


# Non-linear model
sys = Boat2D()


# from pyro.planning.trajectoryoptimisation import DirectCollocationTrajectoryOptimisation

# planner = DirectCollocationTrajectoryOptimisation( sys , 0.1 , 20 )


# planner.init_dynamic_plot()

# planner.x_start = np.array([-5,0,0,2,0,0])
# planner.x_goal  = np.array([0,0,0,0,0,0])

# planner.set_linear_initial_guest()

# planner.maxiter = 500

# planner.compute_optimal_trajectory()

sys.xbar[3] = 0.0
sys.ubar[0] = 0.0

    
# Linear model
ss  = linearize( sys , 0.01 )

# Cost function
cf  = QuadraticCostFunction.from_sys( sys )
cf.Q[0,0] = 1
cf.Q[1,1] = 1
cf.Q[2,2] = 1
cf.Q[3,3] = 1
cf.Q[4,4] = 1
cf.Q[5,5] = 1

cf.R[0,0] = 0.001
cf.R[1,1] = 0.001

# LQR controller
ctl = synthesize_lqr_controller( ss , cf , sys.xbar , sys.ubar )

print(ctl.K)

# Simulation Closed-Loop Non-linear with LQR controller
cl_sys = ctl + sys
cl_sys.x0 = np.array([+1.2,-1.2,-0.2,0,0,0])
cl_sys.compute_trajectory(20)
cl_sys.plot_trajectory('xu')
cl_sys.animate_simulation( time_factor_video = 1.0 )