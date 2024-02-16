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
        # Dimensions of signals
        self.k   = 3 
        self.m   = 2
        self.p   = 6 
        
        super().__init__(self.k, self.m, self.p)
        
        # Label
        self.name = 'Boat Controller'
        
        # Dynamic model available to the controller
        self.sys = sys
        
        # Velocity inner loop parameters
        self.reference_velocity = np.array([ 5.0, 0.0, 0.0])

        # Linearized model
        ss     = linearize( sys , 0.01 )
        self.A = ss.A[3:,3:]
        self.B = ss.B[3:,:]
        print('Velocity linearized dynamic')
        print('----------------------------')
        print('A =\n', self.A)
        print('B =\n', self.B)
        
        # Cost function
        cf  = QuadraticCostFunction(3,2)
        cf.Q[0,0] = 1000
        cf.Q[1,1] = 1000
        cf.Q[2,2] = 5000
        cf.R[0,0] = 0.0001
        cf.R[1,1] = 0.0001
        
        # LQR solution
        self.S = solve_continuous_are( self.A , self.B , cf.Q , cf.R )
        self.K = np.linalg.inv( cf.R ) @ self.B.T @ self.S
        print('Velocity inner-loop LQR gain matrix =\n', self.K)
        
        # Outer position loop parameters
        self.position_control_isactivated = True
        self.KP = np.array([[ 0.5 , 0   , 0],
                            [ 0   , 0.5 , 0],
                            [ 0   , 0   , 2]])
        self.d_max = 2.0

        # Trajectory following parameters
        self.trajectory_following_isactivated = False
        self.traj_amplitude = 10.0
        self.traj_frequency = 0.3

        # Saturations
        self.f_max = np.array([10000,1000])
        self.f_min = np.array([-1000,-1000])
        self.v_max = np.array([5.0,1.0,1.0])
        self.v_min = np.array([-1.0,-1.0,-1.0])

    #############################
    def q_d( self , t = 0 ):
        """ Return the desired position """

        if self.trajectory_following_isactivated:

            a = self.traj_amplitude
            w = self.traj_frequency

            q_d = np.array([ a * np.cos(w*t) , a * np.sin(w*t), 0.0 ]) 
            
            q_d[2] = np.arctan2( a * w * np.cos(w * t ), a * w * - np.sin(w*t))

        else:

            q_d = np.array([0,0,0.0])

        return q_d
    
    #############################
    def dq_d( self , t = 0 ):
        """ Return the time derivative of the desired position """

        if self.trajectory_following_isactivated:

            a = self.traj_amplitude
            w = self.traj_frequency

            dq_d = np.array([ a * w * - np.sin(w*t) , a * w * np.cos(w * t ), 0 ])

            # TODO: dq_d[2] = ...

        else:

            dq_d = np.array([0,0,0.0])

        return dq_d
        
    
    #############################
    def c( self , y , r , t = 0 ):
        """ 
        Control law
        """

        # Avilable observations
        q = y[0:3] # position feedback
        v = y[3:]  # velocity feedback

        if self.position_control_isactivated:

            # Desired position and velocity
            q_d  = self.q_d(t)
            dq_d = self.dq_d(t)
            
            # Configuration error
            q_e = q_d - q
            d_e = np.linalg.norm(q_e[0:2])  # absolute distance to target

            # Angular position error withtout cyclic fuck-up
            q_e[2] = na( na(q_d[2]) - na(q[2]) )  

            # Dynamic heading ref
            # If far from target, reference orientation is overided to
            # instead making the boat head toward the desired position
            if d_e > self.d_max:
                actual  = na( q[2] )
                desired = na( np.arctan2( q_e[1] , q_e[0] )  )
                q_e[2]  = na( desired - actual )

            # Position outter loop in inertial frame
            dq_r = self.KP @ q_e + dq_d * 1.0 
            # TODO 0.9 is a quick n dirty fix for the boat to always lag behind the target

            # Desired velocity in body frame
            v_d = self.sys.N( q ).T @ dq_r

            # Velocity setpoint limits
            v_d = np.clip( v_d , self.v_min , self.v_max )

        else:

            # Direct Velocity control for debugging
            v_d = self.reference_velocity
        
        # Velocity error
        v_e = v_d - v
        
        # Velocity inner loop
        u = self.K @ v_e
        
        # Max/min thrust
        u = np.clip( u , self.f_min , self.f_max )

        return u
    
    #########################################################################
    def forward_kinematic_lines_plus( self, x , u , t ):
        """  
        Graphical output for the controller
        -----------------------------------
        plot the desired boat pose

        x,u,t are the state, input and time of the global closed-loop system

        """

        # desired boat pose, from model forward kinematics
        pts, style, color = self.sys.forward_kinematic_lines( self.q_d(t) )

        # Change the line style and color
        style[0] = '--'
        color[0] = 'c'
        color[1] = 'c'

        return pts, style, color


# Non-linear model
sys = Boat2D()

# Cascade controller
ctl = BoatController( sys )

# Simulation Closed-Loop Non-linear with LQR controller
cl_sys = ctl + sys

cl_sys.x0 = np.array([3,3,1.0,5,0,0])
cl_sys.compute_trajectory(10)

cl_sys.plot_trajectory('xu')
cl_sys.animate_simulation( time_factor_video = 1.0 )

# cl_sys.x0 = np.array([-20,10,-2.5,0,0,0])
# cl_sys.compute_trajectory(20)

# # cl_sys.plot_trajectory('xu')
# cl_sys.animate_simulation( time_factor_video = 1.0 )

# cl_sys.x0 = np.array([50,50,0.0,0,0,0])
# cl_sys.compute_trajectory(50)

# # cl_sys.plot_trajectory('xu')
# # cl_sys.animate_simulation( time_factor_video = 1.0 )
# cl_sys.animate_simulation( time_factor_video = 1.0 , save = True , file_name = 'boat2' , show = False)


# cl_sys.x0 = np.array([0,0,0,0,0,0])
ctl.trajectory_following_isactivated = True
# # ctl.d_max                = 0.0
cl_sys.compute_trajectory(40)

cl_sys.plot_trajectory('xu')
cl_sys.animate_simulation( time_factor_video = 1.0 )
# cl_sys.animate_simulation( time_factor_video = 1.0 , save = True , file_name = 'boat1' , show = False)
