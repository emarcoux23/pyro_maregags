#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  9 09:07:29 2022

@author: alex
"""

import numpy as np


###############################################################################
class MarkovSystem:
    """ 
    Mother class for discrete stochastic dynamical systems
    ----------------------------------------------
    n : number of finite states
    m : number of control action
    p : number of outputs
    k : time index
    ---------------------------------------
    x_k+1 = fk( x_k , u_k , k )
    y_k   = hk( x_k , u_k , k )
    
    optionnal: 
    u_k = policy( y_k , k ) : action autonomous decision
    
    
    """
    ###########################################################################
    # The two following functions needs to be implemented by child classes
    ###########################################################################
    
    ############################
    def __init__(self, n = 1, m = 1):
        """ 
        The __init__ method of the Mother class can be used to fill-in default
        labels, units, bounds, and base values.
        """
        
        #############################
        # Parameters
        #############################
        
        p = n

        # Dimensions
        self.n = n   
        self.m = m   
        self.p = p
        
        # Labels
        self.name = 'ContinuousDynamicSystem'
        self.state_label  = []
        self.input_label  = []
        self.output_label = []
        
        # Default Label and units
        for i in range(n):
            self.state_label.append('State '+str(i))
        for i in range(m):
            self.input_label.append('Action '+str(i))
        for i in range(p):
            self.output_label.append('Output '+str(i))
        
        # Default state and inputs values    
        self.xbar = 0
        self.ubar = 0
        
        # Plot params
        self.domain     = [ (-10,10) , (-10,10) , (-10,10) ]
        self.linestyle  = 'o-'
        self.is_3d      =  False  # Use 2d plot by default
        
        
        ################################
        # Transition Probabilities
        ################################
        
        self.T_ija = np.zeros((n,n,m))
        
        for a in range(m):
            
            self.T_ija[:,:,a] = np.diag(np.ones((n)))
        
        
        ################################
        # Variables
        ################################
        
        # Initial value for simulations
        self.x0    = np.zeros( n )
        self.x0[0] = 1
        
        # Result of last simulation
        self.traj = None
        
        # Cost function for evaluation
        # TODO

    
    #############################
    def fk( self , x , u , k = 0 ):
        """ 
        compute the evolution of the probability distribution
        
        """
        
        T_ij = self.T_ija[:,:,u] # transition matrix of given action
        
        x_k1 = np.dot( T_ij , x )
        
        return x_k1
    
    
    #############################
    def check_probability_matrix( self ):
        """ 
        check if transition prob sums to 1
        
        """
        
        print( self.T_ija.sum( axis = 0 ) )
        
        return self.T_ija.sum( axis = 0 ) # should be all ones
    
    
    ###########################################################################
    # The following functions can be overloaded when necessary by child classes
    ###########################################################################
    
    #############################
    def h( self , x , k = 0 ):
        """ 
        Output fonction y = h(x,u,t)
        
        """
        
        y = x      # default output is state
        
        return y
    
    #############################
    def policy( self , y , k ):
        """ 
        
        """
        
        # Default action
        u = self.ubar
        
        return u
    
        
    #############################
    def simulation_of_density_probability( self , N = 10 , plot = True ):
        """ 
        N = number of steps
        """
        
        x_k = self.x0
        
        for k in range(N):
            
            y_k  = self.h( x_k , k )
            u_k  = self.policy( y_k , k )
            x_k1 = self.fk( x_k, u_k , k )
            
            x_k = x_k1
            
            if plot:
                print(x_k1)
            
        
        return x_k1
    

    
    
    


'''
#################################################################
##################          Main                         ########
#################################################################
'''


if __name__ == "__main__":     
    """ MAIN TEST """
    
    m = MarkovSystem(3,2)
    
    m.T_ija[0,0,0] = 0.5
    m.T_ija[1,0,0] = 0.5
    
    m.check_probability_matrix()
    