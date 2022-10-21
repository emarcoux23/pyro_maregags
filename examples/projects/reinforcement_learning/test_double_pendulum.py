#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 22:27:47 2022

@author: alex
"""

import numpy as np

from pyro.dynamic  import pendulum
from pyro.control  import controller
import dynamic_programming as dprog
import discretizer
import costfunction

sys  = pendulum.DoublePendulum()

dx = 0.1

sys.x_ub = np.array([  dx ,  dx,  dx ,  dx ])
sys.x_lb = np.array([ -dx , -dx, -dx , -dx ])

sys.u_ub = np.array([  20.0 ,  10.0 ])
sys.u_lb = np.array([ -20.0 , -10.0 ])

# Discrete world 
grid_sys = discretizer.GridDynamicSystem( sys , [21,21,21,21] , [3,3] , 0.02 )

# Cost Function
qcf = costfunction.QuadraticCostFunction.from_sys(sys)

qcf.xbar = np.array([ 0 , 0, 0 , 0 ]) # target
qcf.INF  = 1000
qcf.EPS  = 0.05

qcf.Q[0,0] = 0.1
qcf.Q[1,1] = 0.1
qcf.Q[2,2] = 0.1
qcf.Q[3,3] = 0.1

qcf.S[0,0] = 10000
qcf.S[1,1] = 10000
qcf.S[2,2] = 10000
qcf.S[3,3] = 10000


# DP algo
dp = dprog.DynamicProgrammingWithLookUpTable( grid_sys, qcf )


dp.compute_steps( 10 , animate_cost2go=True )


"""
ctl = dprog.LookUpTableController( grid_sys , dp.pi )

ctl.plot_control_law( sys = sys , n = 100)


#asign controller
cl_sys = controller.ClosedLoopSystem( sys , ctl )

##############################################################################

# Simulation and animation
cl_sys.x0   = np.array([0,0.1,0,0.1])
cl_sys.compute_trajectory( 10, 10001, 'euler')
cl_sys.plot_trajectory('xu')
cl_sys.plot_phase_plane_trajectory()
cl_sys.animate_simulation()
"""