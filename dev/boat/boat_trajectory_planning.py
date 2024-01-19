#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  3 08:27:06 2021

@author: alex
"""

import numpy as np

from boat import Boat

from pyro.planning.trajectoryoptimisation import DirectCollocationTrajectoryOptimisation

# DO NOT CONVERGE!!!

sys = Boat()

planner = DirectCollocationTrajectoryOptimisation( sys , 1.0 , 10 )


planner.init_dynamic_plot()

planner.x_start = np.array([0,-10,0,0,2,0])
planner.x_goal  = np.array([0,0,0,0,0,0])

planner.set_linear_initial_guest()
# planner.set_linear_initial_guest( True )

planner.maxiter = 500

planner.compute_optimal_trajectory()
# planner.show_solution()
# planner.animate_solution()

