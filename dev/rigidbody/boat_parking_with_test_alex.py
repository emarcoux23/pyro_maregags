#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  3 08:27:06 2021

@author: alex
"""

import numpy as np

from scipy.linalg  import solve_continuous_are

from boat2 import Boat2D

from pyro.analysis.costfunction import QuadraticCostFunction
from pyro.dynamic.statespace    import linearize
from pyro.control.lqr           import synthesize_lqr_controller


from pyro.control import controller




###############################################################################

class BoatController( controller.StaticController ) :
    
    ############################
    def __init__( self , sys ):
        """ """
        
        # Dimensions
        self.k   = 3 
        self.m   = 2
        self.p   = 6 
        
        super().__init__(self.k, self.m, self.p)
        
        # Label
        self.name = 'Boat Controller'
        
        
        self.sys = sys
        
        ss  = linearize( sys , 0.01 )
        
        # Velocity only
        self.A = ss.A[3:,3:]
        self.B = ss.B[3:,:]
        
        # Cost function
        cf  = QuadraticCostFunction(3,2)
        cf.Q[0,0] = 1000
        cf.Q[1,1] = 1000
        cf.Q[2,2] = 10000

        cf.R[0,0] = 0.001
        cf.R[1,1] = 0.001
        
        S = solve_continuous_are( self.A , self.B , cf.Q , cf.R )
        
        # Matrix gain
        BTS   = np.dot( self.B.T , S )
        R_inv = np.linalg.inv( cf.R )
        self.K     = np.dot( R_inv  , BTS )
        
        self.v_d = np.array([1,0,-0.5])
        
        self.q_d = np.array([0,0,0.0])
        
        self.KP = np.array([[ 0.5 , 0   , 0],
                            [ 0   , 0.5 , 0],
                            [ 0   , 0   , 2]])
        
        self.t_max = np.array([10000,1000])
        self.t_min = np.array([-1000,-1000])
        self.v_max = np.array([5.0,1.0,1.0])
        self.v_min = np.array([-1.0,-1.0,-1.0])
        
    
    #############################
    def c( self , y , r , t = 0 ):

        q = y[0:3]
        v = y[3:]
        
        q_e = self.q_d - q
        
        if q_e[0]**2>2.0 or q_e[1]**2>2.0:
            q_e[2] = np.arctan2(q_e[1],q_e[0]) - q[2]
        
        # v_d = self.v_d
        v_d = self.sys.N( q ).T @ self.KP @ q_e
        
        v_d = np.clip( v_d , self.v_min , self.v_max )
        
        v_e = v_d - v
        
        u = self.K @ v_e
        
        u = np.clip( u , self.t_min , self.t_max )

        
        return u


# Non-linear model
sys = Boat2D()



ctl = BoatController( sys )

print(ctl.K)

# Simulation Closed-Loop Non-linear with LQR controller
cl_sys = ctl + sys

cl_sys.x0 = np.array([3,3,1.0,5,0,0])
cl_sys.compute_trajectory(20)

# cl_sys.x0 = np.array([-20,10,-2.5,0,0,0])
# cl_sys.compute_trajectory(20)

# cl_sys.x0 = np.array([50,50,0.0,0,0,0])
# cl_sys.compute_trajectory(50)

cl_sys.plot_trajectory('xu')
cl_sys.animate_simulation( time_factor_video = 1.0 )