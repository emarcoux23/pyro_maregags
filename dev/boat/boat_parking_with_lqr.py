#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  3 08:27:06 2021

@author: alex
"""

import numpy as np

from boat import Boat

from pyro.analysis.costfunction import QuadraticCostFunction
from pyro.dynamic.statespace    import linearize
from pyro.control.lqr           import synthesize_lqr_controller


# Non-linear model
sys = Boat()

sys.ubar =  np.array([100,0]) 
    
# Linear model
ss  = linearize( sys , 0.01 )

# Cost function
cf  = QuadraticCostFunction.from_sys( sys )
cf.Q[0,0] = 10000
cf.Q[1,1] = 0
cf.Q[2,2] = 100
cf.Q[3,3] = 0
cf.Q[4,4] = 0
cf.Q[5,5] = 0

cf.R[0,0] = 1
cf.R[1,1] = 10000

# LQR controller
ctl = synthesize_lqr_controller( ss , cf , sys.xbar , sys.ubar )

# Instable when trust reverse!!!


# Simulation Closed-Loop Non-linear with LQR controller
cl_sys = ctl + sys
cl_sys.x0 = np.array([1,-10,-0.2,0,0,0])
cl_sys.x0 = np.array([-2.0,-10,-0.5,0,0,0])
cl_sys.compute_trajectory(20)
cl_sys.plot_trajectory('xu')
ani = cl_sys.animate_simulation( time_factor_video = 1.0 )
# cl_sys.animate_simulation( save = True , file_name = 'boat_parking_with_lqr.gif' )
# ani.save( 'test.mp4', writer='imagemagick', fps=30)
# ani.save( 'test.gif')