#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 11 10:11:31 2023

@author: alex
"""


##############################################################################
# 1 DoF Car Model
##############################################################################
        
class MountainCar( system.ContinuousDynamicSystem ):
    """ 
    Equations of Motion
    -------------------------
    TBD
    """
    
    ############################
    def __init__(self):
        """ """
    
        # Dimensions
        self.n = 2   
        self.m = 1   
        self.p = 2
        
        # initialize standard params
        system.ContinuousDynamicSystem.__init__( self, self.n, self.m, self.p)
        
        # Labels
        self.name = 'Mountain Car'
        self.state_label = ['x','dx']
        self.input_label = ['throttle']
        self.output_label = ['x','dx']
        
        # Units
        self.state_units = ['[m]','[m/sec]']
        self.input_units = ['[N]']
        self.output_units = ['[m]','[m/sec]']
        
        # State working range
        self.x_ub = np.array([+0.5,+0.07,])
        self.x_lb = np.array([-1.2,-0.07])
        
        # Input working range
        self.u_ub = np.array([  1.0])
        self.u_lb = np.array([ -1.0])
        
        # Model param
        self.mass    = 1.0          # total car mass [kg]
        self.gravity = 1.0       # gravity constant [N/kg]
        
        # Relief curve
        self.a   = 0.0025
        self.b   = 0.001
        self.w   = 0.0025
        
        # Graphic output parameters 
        self.dynamic_domain  = False
        
        
    #############################################
    def f(self, x, u, t = 0 ):

        # TODO

        return dx
        
        