#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 14:04:51 2021

@author: alex
"""

import numpy as np

from pyro.dynamic.pendulum                import Acrobot
from pyro.planning.trajectoryoptimisation import DirectCollocationTrajectoryOptimisation
from pyro.planning.randomtree             import RRT


sys = Acrobot()

#Max/Min torque
sys.u_ub[0] = +30
sys.u_lb[0] = -30

# RRT

x_start = np.array([-3.14,0,0,0])
x_goal  = np.array([0,0,0,0])

rrt = RRT( sys)
    
rrt.u_options = [ sys.u_ub , sys.u_lb ]

rrt.x_start = x_start
rrt.x_goal  = x_goal

rrt.goal_radius          = 1.5
rrt.dt                   = 0.1
rrt.max_nodes            = 10000
rrt.max_solution_time    = 3.0
rrt.max_distance_compute = 1000
rrt.dyna_plot            = False

rrt.find_path_to_goal()

planner = DirectCollocationTrajectoryOptimisation( sys , 0.1 , 20 )

planner.x_start = x_start
planner.x_goal  = x_goal

planner.set_initial_trajectory_guest( rrt.traj )

planner.maxiter = 500
planner.compute_optimal_trajectory()
planner.show_solution()
planner.animate_solution()