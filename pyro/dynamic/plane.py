#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 10:29:50 2023

@author: alex
"""

###############################################################################
import numpy as np
import matplotlib.pyplot as plt
###############################################################################
from pyro.dynamic import system
from pyro.dynamic import mechanical
###############################################################################




##############################################################################
# 2D planar drone
##############################################################################
        
class Plane2D( mechanical.MechanicalSystem ):
    
    """ 
    Equations of Motion
    -------------------------
    
    """
    
    ############################
    def __init__(self):
        """ """
        
        # initialize standard params
        mechanical.MechanicalSystemWithPositionInputs.__init__( self, 3 , 1 , 1 )
        
        # Labels
        self.name = '2D plane model'
        self.state_label = ['x','y','theta','vx','vy','w']
        self.input_label = ['Trust', 'delta']
        self.output_label = self.state_label
        
        # Units
        self.state_units = ['[m]','[m]','[rad]','[m/sec]','[m/sec]','[rad/sec]']
        self.input_units = ['[N]', '[Rad]']
        self.output_units = self.state_units
        
        # State working range
        self.x_ub = np.array([+50,+100,+2,10,10,10])
        self.x_lb = np.array([-50,-0,-2,-10,-10,-10])
        
        # Model param
        self.mass           = 1000
        self.inertia        = 100
        self.gravity        = 9.8
        
        # Kinematic param
        
        
        # Graphic output parameters 
        self.dynamic_domain  = True
        self.dynamic_range   = 10
        
        
        
    ###########################################################################
    def H(self, q ):
        """ 
        Inertia matrix 
        ----------------------------------
        dim( H ) = ( dof , dof )
        
        such that --> Kinetic Energy = 0.5 * dq^T * H(q) * dq
        
        """  
        
        H = np.zeros((3,3))
        
        H[0,0] = self.mass
        H[1,1] = self.mass
        H[2,2] = self.inertia
        
        return H
    
    
    ###########################################################################
    def C(self, q , dq ):
        """ 
        Corriolis and Centrifugal Matrix 
        ------------------------------------
        dim( C ) = ( dof , dof )
        
        such that --> d H / dt =  C + C^T
        
        
        """ 
        
        C = np.zeros((3,3))

        return C
    
    
    ###########################################################################
    def g(self, q ):
        """ 
        Gravitationnal forces vector : dof x 1
        """
        
        g = np.zeros(3)
        
        g[1] =  self.mass * self.gravity

        return g
    
    