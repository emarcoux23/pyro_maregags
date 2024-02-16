#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  3 08:27:06 2021

@author: alex
"""

import numpy as np

from pyro.dynamic.boat          import Boat2D
from pyro.analysis.costfunction import QuadraticCostFunction
from pyro.dynamic.statespace    import linearize
from pyro.control.lqr           import synthesize_lqr_controller


# Non-linear model
sys = Boat2D()

sys.mass    = 1000.0
sys.inertia = 2000.0
sys.l_t     = 2.0
sys.damping_coef = np.array([ 200.0, 2000.0, 1000.0 ])

sys.xbar[3] = 0.0
sys.ubar[0] = 0.0
    
# Linear model
ss  = linearize( sys , 0.01 )

ss.A[ abs(ss.A) < 0.00001 ] = 0.0
ss.B[ abs(ss.B) < 0.00001 ] = 0.0
print(ss.A)
print(ss.B)

CA = np.hstack( (ss.B, 
                 ss.A @ ss.B,
                 ss.A @ ss.A @ ss.B,
                 ss.A @ ss.A @ ss.A @ ss.B,
                 ss.A @ ss.A @ ss.A @ ss.A @ ss.B,
                 ss.A @ ss.A @ ss.A @ ss.A @ ss.A @ ss.B) )

rank = np.linalg.matrix_rank(CA)
print('Controllability rank = ',rank)
# Linearized boat is never controllable arround zero velocity

# Cost function
cf  = QuadraticCostFunction.from_sys( sys )
cf.Q[0,0] = 1E3
cf.Q[1,1] = 1E3
cf.Q[2,2] = 1E3
cf.Q[3,3] = 1E3
cf.Q[4,4] = 1E3
cf.Q[5,5] = 1E3

cf.R[0,0] = 1E-6
cf.R[1,1] = 1E-6

# LQR controller
ctl = synthesize_lqr_controller( ss , cf , sys.xbar , sys.ubar )

ctl.K[ abs(ctl.K) < 0.00001 ] = 0.0
print(ctl.K)

# Simulation Closed-Loop Non-linear with LQR controller
# cl_sys = ctl + sys
# cl_sys.x0 = np.array([+1.0,+0.2,-0.2,0,0,0])
# cl_sys.compute_trajectory( tf = 1000 , n = 10001 )
# cl_sys.plot_trajectory('xu')
# cl_sys.animate_simulation( time_factor_video = 100.0 )