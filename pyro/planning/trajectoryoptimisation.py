#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 05:49:06 2021

@author: alex
"""

import numpy as np
from scipy.optimize import minimize



###############################################################################
class DirectCollocationTrajectoryOptimisation:
    """ 
    Trajectory optimisation based on fixed-time-steps direct collocation
    ---------------------------------------------------------------------
    sys  : dynamical system class
    dt   : time step size
    grid : number of time step (discretization number)
    
    
    """
    
    ############################
    def __init__(self, sys , dt = 0.2 , grid = 20):
        
        self.sys = sys          # Dynamic system class
        
        self.cost_function = sys.cost_function # default is quadratic cost
        
        self.x_start = sys.x0
        self.x_goal  = sys.xbar
        
        self.dt    = dt
        self.grid  = grid
        
        
    
    ############################
    def decisionvariables2xu(self, dec):
        """ Unpack decision variable vector into x and u trajectory matrices """
        
        n    = self.sys.n   # number of states
        m    = self.sys.m   # number of inputs
        grid = self.grid    # number of time steps
    
        x = np.zeros((n,grid)) 
        u = np.zeros((m,grid))
        
        for i in range(self.sys.n):
            x[i,:] = dec[ i * grid : (i+1) * grid ]
            
        for j in range(self.sys.m):
            k = n + j
            u[j,:] = dec[ k * grid : (k+1) * grid ]
        
        return x,u
        
    
    ############################
    def cost(self, dec):
        """ Compute cost for given decision variable using trapez integration approx """
        
        x,u = self.decisionvariables2xu( dec )
        
        J = 0
        
        for i in range(self.grid -1):
            #i
            x_i = x[:,i]
            u_i = u[:,i]
            t_i = i*self.dt
            y_i = self.sys.h(x_i,u_i,t_i)
            dJi = self.cost_function.g( x_i , u_i,  y_i, t_i )
            
            #i+1
            x_i1 = x[:,i+1]
            u_i1 = u[:,i+1]
            t_i1 = (i+1)*self.dt
            y_i1 = self.sys.h(x_i1,u_i1,t_i1)
            dJi1 = self.cost_function.g( x_i1 , u_i1,  y_i1, t_i1 )
            
            #trapez
            dJ = 0.5 * ( dJi + dJi1 )
            
            #integral
            J = J + dJ * self.dt
            
        return J
    
    
    ########################
    def dynamic_constraints(self, dec):
        """ Compute residues of dynamic constraints """
    
        x,u = self.decisionvariables2xu( dec )
        
        residues_vec = np.zeros( (self.grid-1) * self.sys.n )
        
        for i in range(self.grid-1):
            
            #i
            x_i = x[:,i]
            u_i = u[:,i]
            t_i = i*self.dt
            dx_i = self.sys.f(x_i,u_i,t_i) # analytical state derivatives
            
            #i+1
            x_i1 = x[:,i+1]
            u_i1 = u[:,i+1]
            t_i1 = (i+1)*self.dt
            dx_i1 = self.sys.f(x_i1,u_i1,t_i1) # analytical state derivatives
            
            #trapez
            delta_x_eqs = 0.5 * self.dt * (dx_i + dx_i1)
            
            #num diff
            delta_x_num = x[:,i+1] - x[:,i] # numerical delta in trajectory data
            
            diff = delta_x_num - delta_x_eqs
            
            for j in range(self.sys.n):
                #TODO numpy manip for replacing slow for loop
                residues_vec[i + (self.grid-1) * j ] = diff[j]
            
        return residues_vec



'''
#################################################################
##################          Main                         ########
#################################################################
'''


if __name__ == "__main__":     
    """ MAIN TEST """
    
    from pyro.dynamic import pendulum
    

    sys  = pendulum.SinglePendulum()
    
    x_start = np.array([0.1,0])
    x_goal  = np.array([-3.14,0])