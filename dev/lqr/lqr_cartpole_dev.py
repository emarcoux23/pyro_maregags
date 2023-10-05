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

from pyro.analysis  import simulation
###############################################################################

import lqr_dev as lqr


# sys
sys  = cartpole.CartPole()

sys.xbar[1] = np.pi # Up-right position

#Max/Min torque
sys.u_ub[0] = +20
sys.u_lb[0] = -20


# Cost function
cf  = costfunction.QuadraticCostFunction.from_sys( sys )

cf.Q[0,0] = 1.0
cf.Q[1,1] = 1.0
cf.R[0,0] = 1.0

planner = trajectoryoptimisation.DirectCollocationTrajectoryOptimisation( 
            sys , 
            dt = 0.2 , 
            grid = 20 ,
            cost_function = cf 
            )


planner.x_start = np.array([0,0,0,0])
planner.x_goal  = np.array([0,np.pi,0,0])

planner.init_dynamic_plot()
planner.maxiter = 500
planner.compute_optimal_trajectory()
# planner.show_solution()
# planner.animate_solution()

traj = planner.traj

# traj = simulation.Trajectory.load('trajectory.npz')
# traj = simulation.Trajectory.load('cartpole_hidef_swingup_traj.npy')

cf.R[0,0] = 0.1

ctl = lqr.TrajectoryLQRController( sys , traj , cf )


# Simulation
cl_sys = ctl + sys

cl_sys.x0[0] = 0.0
cl_sys.x0[1] = 0.0
    
cl_sys.compute_trajectory( tf = 5.0 , n  = 20001, solver = 'euler')
cl_sys.plot_trajectory('xu')
cl_sys.animate_simulation( time_factor_video=1.0 )

