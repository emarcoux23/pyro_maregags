#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 20:48:32 2022

@author: alex
"""

import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import RectBivariateSpline as interpol2D
from scipy.interpolate import RegularGridInterpolator as rgi

from pyro.control  import controller


###############################################################################
### DP controllers
###############################################################################

class tabular_controller( controller.StaticController ):

    ############################
    def __init__(self, grid_sys , a_star ):
        """ 
        grid_sys : pyro class of discretized continuous system
        a_star   : list of optimal action id by node id
        """
        
        if grid_sys.nodes_n != a_star.size:
            raise ValueError("Grid size does not match optimal action table size")
        
        k = 1                   # Ref signal dim
        m = grid_sys.sys.m      # control input signal dim
        p = grid_sys.sys.n      # output signal dim (state feedback)
        
        super().__init__(k, m, p)
        
        # Grid sys
        self.grid_sys = grid_sys
        
        # Table of optimal actions
        self.a_star = a_star
        
        # Label
        self.name = 'Tabular Controller'
        
        # Options
        self.interpolation_scheme = 'nearest'
        
    
    #############################
    def optimal_action_selection( self , x ):
        """  select the optimal u given actual cost map """
        
        if self.interpolation_scheme == 'nearest':
            
            s = self.grid_sys.get_nearest_node_id_from_state( x )
            a = self.a_star[ s ]
            u = self.grid_sys.input_from_action_id[ a ]
            
        else:
            
            raise NotImplementedError
        
        return u
    

    #############################
    def c( self , y , r , t = 0 ):
        """  State feedback (y=x) - no reference - time independent """
        x = y
        u = self.optimal_action_selection( x )
        
        return u
    
    
##################################################################
class epsilon_greedy_controller( tabular_controller ):
    
    ############################
    def __init__(self,  grid_sys , a_star ):
        """ """
        super().__init__( grid_sys , a_star )
        

        self.name = 'Epsilon Greedy Controller'
        
        self.epsilon = 0.7
        
        
    #############################
    def c( self , y , r , t = 0 ):
        """  State feedback (y=x) - no reference - time independent """
        x = y
        
        if np.random.uniform(0,1) < self.epsilon:
    
            # greedy behavior
            u = self.optimal_action_selection( x )
    
        else:
        
            # Random exploration
            random_index = int(np.random.uniform( 0 , self.grid_sys.actions_n ))
            u = self.grid_sys.input_from_action_id[ random_index ]
            
            # TODO add domain check for random actions?
        
        return u
    
    
    
    
class DynamicProgramming:
    """ Dynamic programming on a grid sys """
    
    ############################
    def __init__(self, grid_sys , cost_function , final_time = 0 ):
        
        # Dynamic system
        self.grid_sys  = grid_sys         # Discretized Dynamic system class
        self.sys       = grid_sys.sys     # Base Dynamic system class
        
        # Cost function
        self.cf  = cost_function
        self.tf  = final_time
        
        # Options
        self.uselookuptable       = True
        self.interpolation_scheme = 'nearest'
        
        # Memory
        self.t = self.tf
        self.k = 0
        
        # Init
        self.initialize()
        
        
    ##############################
    def initialize(self):
        """ initialize cost-to-go and policy """

        self.J_next = np.zeros( self.grid_sys.nodes_n , dtype = float )
        
        self.J_list  = []
        self.t_list  = []
        self.pi_list = []

        # Initial cost-to-go evaluation       
        for s in range( self.grid_sys.nodes_n ):  
            
                x = self.grid_sys.state_from_node_id[ s , : ]
                
                # Final Cost
                self.J_next[ s ] = self.cf.h( x )
        
        
        self.J_list.append( self.J_next  )
        self.t_list.append( self.tf )
        self.pi_list.append( None )
                        
                
    ###############################
    def compute_step(self):
        """ One step of value iteration """
        
        # New iteration backward dt in time
        self.k = self.k + 1
        self.t = self.t - self.grid_sys.dt
        
        # New Cost-to-go and policy to be computed
        J  = self.J_next.copy()
        pi = np.zeros( self.grid_sys.nodes_n , dtype = int   )
        
        # For all state nodes        
        for s in range( self.grid_sys.nodes_n ):  
            
                x = self.grid_sys.state_from_node_id[ s , : ]

                # One steps costs - Q values
                Q = np.zeros( self.grid_sys.actions_n  ) 
                
                # For all control actions
                for a in range( self.grid_sys.actions_n ):
                    
                    u = self.grid_sys.input_from_action_id[ a , : ]
                    
                    y = self.sys.h(x, u, 0) # TODO remove y from cost                    
                    
                    if self.uselookuptable:
                        
                        # This is only for time-independents
                        x_next        = self.grid_sys.x_next_table[s,a,:]
                        action_isok   = self.grid_sys.action_isok[s,a]
                        
                    else:
                        
                        raise NotImplementedError
                    
                    if action_isok:
                        
                        # Cost-to-go of a given action
                        J_next = self.get_J_from_state( x_next )
                        Q[ a ] = self.cf.g(x, u, y, self.t ) * self.grid_sys.dt + J_next
                        
                    else:
                        # Not allowable input at this state
                        Q[ a ] = self.cf.INF
                        
                        
                J[ s ]  = Q.min()
                pi[ s ] = Q.argmin()
                
                # Impossible situation ( unaceptable situation for any control actions )
                if J[ s ] > (self.cf.INF-1) :
                    pi[ s ] = -1
        
        
        # Convergence check        
        delta = J - self.J_next
        j_max     = J.max()
        delta_max = delta.max()
        delta_min = delta.min()
        print(self.k,' t:',self.t,'max:',j_max, 'Deltas:',delta_max,delta_min)
        
        # Update J_next
        self.J_next = J
        
        # List in memory
        self.J_list.append( J  )
        self.t_list.append( self.t )
        self.pi_list.append( pi )
        
    
    ################################
    def get_J_from_state( self, x ):
        """
        
        Parameters
        ----------
        x : n x 1 numpy array
            state of the dynamic system

        Returns
        -------
        J(x) : scalar
               Cost-to-go according to the internal J_next table

        """
        
        if self.interpolation_scheme == 'nearest':
            
            s = self.grid_sys.get_nearest_node_id_from_state( x )
            J = self.J_next[ s ]
            
        else:
            
            raise NotImplementedError
            
        return J

    
    
    ################################
    def compute_steps(self, l = 50):
        """ compute number of step """
               
        for i in range(l):
            self.compute_step()
            
            
            


'''
#################################################################
##################          Main                         ########
#################################################################
'''


if __name__ == "__main__":     
    """ MAIN TEST """
    
    import numpy as np

    from pyro.dynamic  import pendulum
    import discretizer
    from pyro.analysis import costfunction

    sys  = pendulum.SinglePendulum()

    # Discrete world 
    grid_sys = discretizer.GridDynamicSystem( sys )

    # Cost Function
    qcf = sys.cost_function

    qcf.xbar = np.array([ -3.14 , 0 ]) # target
    qcf.INF  = 10000

    # DP algo
    dp = DynamicProgramming( grid_sys, qcf )
    
    dp.compute_steps(1)
    
    # not validated!!
    
    grid_sys.plot_grid_value( dp.J_next )
    grid_sys.plot_control_input_from_policy( dp.pi_list[-1] , 0)
    
    """
    a = dp.pi_list[ -1 ]
    
    ctl = tabular_controller( grid_sys , a )
    
    ctl.plot_control_law( sys = sys , n = 100)
    
    
    #asign controller
    cl_sys = controller.ClosedLoopSystem( sys , ctl )
    
    ##############################################################################
    
    # Simulation and animation
    cl_sys.x0   = np.array([0,0])
    cl_sys.compute_trajectory( 10, 10001, 'euler')
    cl_sys.plot_trajectory('xu')
    cl_sys.plot_phase_plane_trajectory()
    cl_sys.animate_simulation()
    """