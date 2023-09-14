#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 10:38:30 2023

@author: alex
"""

import numpy as np

from pyro.dynamic import plane

from pyro.planning.trajectoryoptimisation import DirectCollocationTrajectoryOptimisation

sys = plane.Plane2D()

sys.x0   = np.array([0,0,0,20,0,0])



def t2u(t):
    
    u = np.array([ 20 , -0.1  ])
    
    return u
    
sys.t2u = t2u

#sys.gravity = 0

sys.compute_trajectory( 2 , 2001 , 'euler' )
#sys.plot_trajectory('x')
sys.animate_simulation( time_factor_video=0.5 )

planner = DirectCollocationTrajectoryOptimisation( sys )

planner.x_start = sys.x0
planner.x_goal  = sys.traj.x[-1,:]

planner.grid    = 20
planner.dt      = 0.1
planner.maxiter = 400

planner.set_initial_trajectory_guest( sys.traj )

planner.compute_optimal_trajectory()
planner.show_solution()
planner.animate_solution()