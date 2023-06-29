# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 20:28:17 2018

@author: Alexandre
"""
##############################################################################
import numpy as np
##############################################################################
from pyro.dynamic  import pendulum
from pyro.planning import discretizer
from pyro.analysis import costfunction
from pyro.planning import dynamicprogramming
##############################################################################

sys  = pendulum.InvertedPendulum()


m = 1
g = 10
l = 1

t_max_star  = 0.35
q_star      = 0.1

theta_star  = 2.0 * np.pi
dtheta_star = 1.0 * np.pi
time_star   = 2.0 * np.pi * 10.0


omega = np.sqrt( ( g / l  ) )
mgl   = m * g * l

t_max  = t_max_star * mgl
q      = q_star * mgl
theta  = theta_star
dtheta = dtheta_star * omega
time   = time_star / omega
J_max  = mgl**2 / omega * time_star * ( ( q_star * theta_star )**2 + t_max_star**2 )
J_min  = J_max / 4000


# kinematic
sys.lc1 = l

sys.l1       = sys.lc1
sys.l_domain = sys.lc1 * 2

# dynamic
sys.m1       = m
sys.I1       = 0
sys.gravity  = g
sys.d1       = 0

sys.u_ub[0]  = + t_max
sys.u_lb[0]  = - t_max

sys.x_ub = np.array([ + theta , + dtheta ])
sys.x_lb = np.array([ - theta , - dtheta ])

dt = 0.05

# Discrete world 
grid_sys = discretizer.GridDynamicSystem( sys , [301,301] , [21] , dt )

# Cost Function
qcf = costfunction.QuadraticCostFunction.from_sys(sys)

qcf.xbar = np.array([ 0 , 0 ]) # target
qcf.INF  = J_max


qcf.Q[0,0] = q ** 2
qcf.Q[1,1] = 0.0

qcf.R[0,0] = 1.0

qcf.S[0,0] = 0.0
qcf.S[1,1] = 0.0


# DP algo
dp = dynamicprogramming.DynamicProgrammingWithLookUpTable( grid_sys, qcf )


steps = int( time / dt) 

dp.compute_steps( steps )
#dp.solve_bellman_equation( tol = J_min )


dp.clean_infeasible_set()
dp.plot_cost2go_3D()
dp.plot_policy()

ctl = dp.get_lookup_table_controller()

# Simulation
cl_sys = ctl + sys
cl_sys.x0   = np.array([-3.14, 0.])
cl_sys.compute_trajectory( 10, 10001, 'euler')
cl_sys.plot_trajectory('xu')
cl_sys.plot_phase_plane_trajectory()
#cl_sys.animate_simulation()