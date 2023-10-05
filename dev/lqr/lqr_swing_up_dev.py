#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  4 10:01:30 2023

@author: alex
"""

###############################################################################
import numpy as np
###############################################################################
from pyro.dynamic   import pendulum
from pyro.analysis  import costfunction
from pyro.planning  import trajectoryoptimisation
from pyro.control   import lqr
from pyro.analysis  import simulation
###############################################################################

import lqr_dev as lqr


# # sys
# sys  = pendulum.InvertedPendulum()
sys = pendulum.SinglePendulum()


# #Max/Min torque
# sys.u_ub[0] = +4
# sys.u_lb[0] = -6


# Cost function
cf  = costfunction.QuadraticCostFunction.from_sys( sys )

cf.Q[0,0] = 1.0
cf.Q[1,1] = 1.0
cf.R[0,0] = 0.1

# planner = trajectoryoptimisation.DirectCollocationTrajectoryOptimisation( 
#             sys , 
#             dt = 0.2 , 
#             grid = 20 ,
#             cost_function = cf 
#             )


# planner.x_start = np.array([+np.pi,0])
# planner.x_goal  = np.array([0,0])

# planner.init_dynamic_plot()
# planner.maxiter = 500
# planner.compute_optimal_trajectory()



#Max/Min torque
sys.u_ub[0] = +4
sys.u_lb[0] = -6

planner = trajectoryoptimisation.DirectCollocationTrajectoryOptimisation( sys )

planner.x_start = np.array([0,0])
planner.x_goal  = np.array([-3.14,0])

planner.compute_optimal_trajectory()
planner.show_solution()
planner.animate_solution()


# planner.show_solution()
# planner.animate_solution()

traj = planner.traj

# # traj = simulation.Trajectory.load('trajectory.npz')

ctl = lqr.TrajectoryLQRController( sys , traj , cf )


# Simulation
cl_sys = ctl + sys

cl_sys.x0[0] = 0.0
cl_sys.x0[1] = 0.0
    
cl_sys.compute_trajectory( tf = 5.0 , n  = 20001, solver = 'euler')
cl_sys.plot_trajectory('xu')
cl_sys.animate_simulation( time_factor_video=1.0 )

