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

sys  = pendulum.TwoIndependentSinglePendulum()

sys.x_ub = np.array([ 4.0 , 4.0, 5.0 , 5.0 ])
sys.x_lb = np.array([ -4.0 , -4.0, -5.0 , -5.0 ])

# Discrete world 
grid_sys = discretizer.GridDynamicSystem( sys , [41,41,21,21] , [3,3] , 0.2 )

# Cost Function
qcf = costfunction.QuadraticCostFunction.from_sys(sys)

qcf.xbar = np.array([ -3.14 , -3.14, 0 , 0 ]) # target
qcf.INF  = 1000000
qcf.EPS  = 2.0


# DP algo
dp = dynamicprogramming.DynamicProgrammingWithLookUpTable( grid_sys, qcf)
#dp = dynamicprogramming .DynamicProgrammingFast2DGrid(grid_sys, qcf)


#dp.interpol_method = 'nearest' #12 sec
#dp.interpol_method = 'linear'  #18 sec
#dp.interpol_method =  'linear' #

#dp.plot_dynamic_cost2go = False
#dp.compute_steps(1,True)
dp.compute_steps(80)
#dp.save_latest('test4d_2')

ctl = dp.get_lookup_table_controller()

cl_sys = ctl + sys

cl_sys.x0   = np.array([-0.5,0.1,0,0])
cl_sys.compute_trajectory( 30, 10001, 'euler')
cl_sys.plot_trajectory('xu')
cl_sys.animate_simulation()

