#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 22:27:47 2022

Note: Is this exemple working or a work in progress???

@author: alex
"""

import numpy as np

from pyro.dynamic  import pendulum
from pyro.control  import controller
from pyro.analysis import costfunction
from pyro.planning import dynamicprogramming 
from pyro.planning import discretizer

sys  = pendulum.DoublePendulum()

sys.I1 = 5
sys.I2 = 5

dx1 =  4.0
dx2 =  4.0
ddx1 = 5.0
ddx2 = 5.0

sys.x_ub = np.array([  dx1 ,  dx2,  ddx1 ,  ddx2 ])
sys.x_lb = np.array([ -dx1 , -dx2, -ddx1 , -ddx2 ])

sys.u_ub = np.array([  10.0 ,  5.0 ])
sys.u_lb = np.array([ -10.0 , -5.0 ])

# Discrete world 
grid_sys = discretizer.GridDynamicSystem( sys , [41,41,21,21] , [3,3] , 0.2 )

# Cost Function
qcf = costfunction.QuadraticCostFunction.from_sys(sys)

qcf.xbar = np.array([ 0 , 0, 0 , 0 ]) # target
qcf.INF  = 100000
qcf.EPS  = 0.2

qcf.Q[0,0] = 0.1
qcf.Q[1,1] = 0.1
qcf.Q[2,2] = 0.1
qcf.Q[3,3] = 0.1

qcf.S[0,0] = 1
qcf.S[1,1] = 1
qcf.S[2,2] = 1
qcf.S[3,3] = 1


# DP algo
dp = dynamicprogramming.DynamicProgrammingWithLookUpTable( grid_sys, qcf )

#dp.plot_cost2go()
#dp.compute_steps( 10 , animate_policy = True )
dp.compute_steps(100)


ctl = dp.get_lookup_table_controller()



#asign controller
cl_sys = controller.ClosedLoopSystem( sys , ctl )

##############################################################################

# Simulation and animation
cl_sys.x0   = np.array([0,0.1,0,0.0])
cl_sys.compute_trajectory( 10, 10001, 'euler')
cl_sys.plot_trajectory('xu')
cl_sys.plot_phase_plane_trajectory()
cl_sys.animate_simulation()
