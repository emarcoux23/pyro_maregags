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
def na( theta ):
    """ 
    Normalize angle to [-pi,pi]
    """
    
    theta = ( theta + np.pi )  % (2*np.pi) - np.pi
        
    return theta


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
        print(ss.A)
        
        # Velocity only
        self.A = ss.A[3:,3:]
        self.B = ss.B[3:,:]
        
        # Cost function
        cf  = QuadraticCostFunction(3,2)
        cf.Q[0,0] = 1000
        cf.Q[1,1] = 1000
        cf.Q[2,2] = 5000

        cf.R[0,0] = 0.0001
        cf.R[1,1] = 0.0001
        
        S = solve_continuous_are( self.A , self.B , cf.Q , cf.R )
        
        # Matrix gain
        BTS   = np.dot( self.B.T , S )
        R_inv = np.linalg.inv( cf.R )
        self.K     = np.dot( R_inv  , BTS )
        print(self.K)
        
        self.KP = np.array([[ 0.5 , 0   , 0],
                            [ 0   , 0.5 , 0],
                            [ 0   , 0   , 2]])
        
        self.t_max = np.array([10000,1000])
        self.t_min = np.array([-1000,-1000])
        self.v_max = np.array([5.0,1.0,1.0])
        self.v_min = np.array([-1.0,-1.0,-1.0])

        self.trajectory_following = False
        self.d_max = 2.0


    #############################
    def q_d( self , t = 0 ):
        """ Return the desired position """

        if self.trajectory_following:

            # vx = 5.0
            a = 10.
            w = 0.3

            q_d = np.array([ a * np.cos(w*t) , a * np.sin(w*t), 0.0 ]) 
            
            q_d[2] = np.arctan2( a * w * np.cos(w * t ), a * w * - np.sin(w*t))

        else:

            q_d = np.array([0,0,0.0])

        return q_d
    
    #############################
    def dq_d( self , t = 0 ):
        """ Return the desired position """

        if self.trajectory_following:

            a = 10.0
            w = 0.3

            dq_d = np.array([ a * w * - np.sin(w*t) , a * w * np.cos(w * t ), 0 ])

        else:

            dq_d = np.array([0,0,0.0])

        return dq_d
        
    
    #############################
    def c( self , y , r , t = 0 ):

        q = y[0:3]
        v = y[3:]

        q_d  = self.q_d(t)
        dq_d = self.dq_d(t)
        
        # Configuration error
        q_e = q_d - q
        q_e[2] = na( na(q_d[2]) - na(q[2]) )   # withtout cyclic fuck-up
        d_e = np.linalg.norm(q_e[0:2])         # distance to target

        # Dynamic heading ref
        # If far from target, head to target
        # If close to target, head to ref orientation
        if d_e > self.d_max:
            actual  = na( q[2] )
            desired = na( np.arctan2(q_e[1],q_e[0]) )
            q_e[2]  = na( desired - actual )

        # Position outter loop in inertial frame
        dq_r = self.KP @ q_e + dq_d * 0.9

        # Desired velocity in body frame
        v_d = self.sys.N( q ).T @ dq_r

        # Direct Velocity control for debugging
        # v_d = np.array([1,0,-0.5])
        
        # Velocity setpoint limits
        v_d = np.clip( v_d , self.v_min , self.v_max )
        
        # Velocity error
        v_e = v_d - v
        
        # Velocity inner loop
        u = self.K @ v_e
        
        # Max/min thrust
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

cl_sys.plot_trajectory('xu')
cl_sys.animate_simulation( time_factor_video = 1.0 )

cl_sys.x0 = np.array([-20,10,-2.5,0,0,0])
cl_sys.compute_trajectory(20)

cl_sys.plot_trajectory('xu')
cl_sys.animate_simulation( time_factor_video = 1.0 )

cl_sys.x0 = np.array([50,50,0.0,0,0,0])
cl_sys.compute_trajectory(50)

cl_sys.plot_trajectory('xu')
cl_sys.animate_simulation( time_factor_video = 1.0 )
# cl_sys.animate_simulation( time_factor_video = 1.0 , save = True , file_name = 'boat2' , show = False)


cl_sys.x0 = np.array([0,0,0,0,0,0])
ctl.trajectory_following = True
# ctl.d_max                = 0.0
cl_sys.compute_trajectory(40)

cl_sys.plot_trajectory('xu')
cl_sys.animate_simulation( time_factor_video = 1.0 )
# cl_sys.animate_simulation( time_factor_video = 1.0 , save = True , file_name = 'boat1' , show = False)
