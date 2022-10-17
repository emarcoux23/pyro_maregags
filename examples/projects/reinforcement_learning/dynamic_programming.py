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

class LookUpTableController( controller.StaticController ):

    ############################
    def __init__(self, grid_sys , pi ):
        """ 
        grid_sys : pyro class of discretized continuous system
        a_star   : list of optimal action id by node id
        """
        
        if grid_sys.nodes_n != pi.size:
            raise ValueError("Grid size does not match optimal action table size")
        
        k = 1                   # Ref signal dim
        m = grid_sys.sys.m      # control input signal dim
        p = grid_sys.sys.n      # output signal dim (state feedback)
        
        super().__init__(k, m, p)
        
        # Grid sys
        self.grid_sys = grid_sys
        
        # Table of actions
        self.pi = pi
        
        # Label
        self.name = 'Tabular Controller'
        
        # Interpolation Options
        self.interpol_method = []
        
        for k in range(self.m):
            self.interpol_method.append('linear') # "linear”, “nearest”, “slinear”, “cubic”, and “quintic”
            
        self.compute_interpol_functions()
        
    
    #############################
    def compute_interpol_functions( self  ):
        """  """
        
        self.u_interpol = [] 
        
        for k in range(self.m):
            
            u_k      = self.grid_sys.input_from_policy_array( self.pi, k)
            self.u_interpol.append( self.grid_sys.compute_interpolation_function( u_k , self.interpol_method[k] ) )
        
    
    #############################
    def lookup_table_selection( self , x ):
        """  select the optimal u given actual cost map """
        
        u = np.zeros( self.m )
        
        for k in range(self.m):
            
            u[k] = self.u_interpol[k]( x )
            
        return u
    

    #############################
    def c( self , y , r , t = 0 ):
        """  State feedback (y=x) - no reference - time independent """
        
        x = y
        
        u = self.lookup_table_selection( x )
        
        return u
    
    
##################################################################
class EpsilonGreedyController( LookUpTableController ):
    
    ############################
    def __init__(self,  grid_sys , pi_star ):
        """ """
        super().__init__( grid_sys , pi_star )
        

        self.name = 'Epsilon Greedy Controller'
        
        self.epsilon = 0.7
        
        
    #############################
    def c( self , y , r , t = 0 ):
        """  State feedback (y=x) - no reference - time independent """
        x = y
        
        if np.random.uniform(0,1) < self.epsilon:
    
            # greedy behavior
            u = self.lookup_table_selection( x )
    
        else:
        
            # Random exploration
            random_index = int(np.random.uniform( 0 , self.grid_sys.actions_n ))
            u = self.grid_sys.input_from_action_id[ random_index ]
            
            # TODO add domain check for random actions?
        
        return u
    

###############################################################################
### DP controllers
###############################################################################

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
        
        self.interpol_method ='linear' # "linear”, “nearest”, “slinear”, “cubic”, and “quintic”
        
        
        # Memory
        self.t = self.tf
        self.k = 0
        
        # Lists of time, cost-to-go and policy
        self.t_list  = []
        self.J_list  = []
        self.pi_list = []
        
        # Final cost
        self.evaluate_terminal_cost()
        
        # Final value in lists
        self.J_list.append( self.J_next  )
        self.t_list.append( self.tf )
        self.pi_list.append( None )
        
        
    ##############################
    def evaluate_terminal_cost(self):
        """ initialize cost-to-go and policy """

        self.J_next  = np.zeros( self.grid_sys.nodes_n , dtype = float )

        # Initial cost-to-go evaluation       
        for s in range( self.grid_sys.nodes_n ):  
            
                x = self.grid_sys.state_from_node_id[ s , : ]
                
                # Final Cost
                self.J_next[ s ] = self.cf.h( x )
                
    
    ###############################
    def initialize_backward_step(self):
        """ One step of value iteration """
        
        # Update values
        self.k = self.k + 1                  # index backward in time
        self.t = self.t - self.grid_sys.dt   # time
        
        # New Cost-to-go and policy array to be computed
        self.J  = np.zeros( self.grid_sys.nodes_n , dtype = float )
        self.pi = np.zeros( self.grid_sys.nodes_n , dtype = int   )
        
        # Create interpol function
        self.J_interpol = self.grid_sys.compute_interpolation_function( self.J_next , self.interpol_method )
                        
                
    ###############################
    def compute_backward_step(self):
        """ One step of value iteration """
        
        # For all state nodes        
        for s in range( self.grid_sys.nodes_n ):  
            
                x = self.grid_sys.state_from_node_id[ s , : ]
                
                Q = np.zeros( self.grid_sys.actions_n  ) 
                
                # For all control actions
                for a in range( self.grid_sys.actions_n ):
                    
                    u = self.grid_sys.input_from_action_id[ a , : ]
                    
                    y = self.sys.h(x, u, 0) # TODO remove y from cost                    
                        
                    # If action is in allowable set
                    if self.sys.isavalidinput(x,u):
                        
                        x_next = self.sys.f(x,u,self.t ) * self.grid_sys.dt + x
                        
                        # if the next state is not out-of-bound
                        if self.sys.isavalidstate(x_next):

                            # Cost-to-go of a given action
                            J_next = self.J_interpol( x_next )
                            Q[ a ] = self.cf.g(x, u, y, self.t ) * self.grid_sys.dt + J_next
                            
                        else:
                            
                            # Out of bound cost
                            Q[ a ] = self.cf.INF # TODO add option to customize this
                        
                    else:
                        # Invalide control input at this state
                        Q[ a ] = self.cf.INF
                        
                self.J[ s ]  = Q.min()
                self.pi[ s ] = Q.argmin()
                
                # Impossible situation ( unaceptable situation for any control actions )
                if self.J[ s ] > (self.cf.INF-1) :
                    self.pi[ s ] = -1
                    
    
    ###############################
    def finalize_backward_step(self):
        """ One step of value iteration """
        
        # Convergence check        
        delta = self.J - self.J_next
        j_max     = self.J.max()
        delta_max = delta.max()
        delta_min = delta.min()
        print(self.k,' t:',self.t,'max:',j_max, 'Deltas:',delta_max,delta_min)
        
        # Update J_next
        self.J_next = self.J
        
        # List in memory
        self.J_list.append( self.J  )
        self.t_list.append( self.t )
        self.pi_list.append( self.pi )

    
    ################################
    def compute_steps(self, l = 50):
        """ compute number of step """
               
        for i in range(l):
            self.initialize_backward_step()
            self.compute_backward_step()
            self.finalize_backward_step()
            


###############################################################################
    
class DynamicProgrammingWithLookUpTable( DynamicProgramming ):
    """ Dynamic programming on a grid sys """
    
                
    ###############################
    def compute_backward_step(self):
        """ One step of value iteration """

        # For all state nodes        
        for s in range( self.grid_sys.nodes_n ):  
            
                x = self.grid_sys.state_from_node_id[ s , : ]

                # One steps costs - Q values
                Q = np.zeros( self.grid_sys.actions_n  ) 
                
                # For all control actions
                for a in range( self.grid_sys.actions_n ):
                    
                    # If action is in allowable set
                    if self.grid_sys.action_isok[s,a]:
                        
                        # if the next state is not out-of-bound
                        if self.grid_sys.x_next_isok[s,a]:
                            
                            u = self.grid_sys.input_from_action_id[ a , : ]
                            
                            y = self.sys.h(x, u, 0) # TODO remove y from cost                    
                                
                            # This is only for time-independents
                            x_next        = self.grid_sys.x_next_table[s,a,:]

                            # Cost-to-go of a given action
                            J_next = self.J_interpol( x_next )
                            Q[ a ] = self.cf.g(x, u, y, self.t ) * self.grid_sys.dt + J_next
                            
                        else:
                            
                            # Out of bound cost
                            Q[ a ] = self.cf.INF # TODO add option to customize this
                        
                    else:
                        # Not allowable input at this state
                        Q[ a ] = self.cf.INF
                        
                        
                self.J[ s ]  = Q.min()
                self.pi[ s ] = Q.argmin()
                
                # Impossible situation ( unaceptable situation for any control actions )
                if self.J[ s ] > (self.cf.INF-1) :
                    self.pi[ s ] = -1
                    

###############################################################################

class DynamicProgrammingFast2DGrid( DynamicProgramming ):
    """ Dynamic programming on a grid sys """
    
    ###############################
    def initialize_backward_step(self):
        """ One step of value iteration """
        
        # Update values
        self.k = self.k + 1                  # index backward in time
        self.t = self.t - self.grid_sys.dt   # time
        
        # New Cost-to-go and policy array to be computed
        self.J  = np.zeros( self.grid_sys.nodes_n , dtype = float )
        self.pi = np.zeros( self.grid_sys.nodes_n , dtype = int   )
        
        # Create interpol function
        self.J_interpol = self.grid_sys.compute_bivariatespline_2D_interpolation_function( self.J_next )
    
                
    ###############################
    def compute_backward_step(self):
        """ One step of value iteration """

        # For all state nodes        
        for s in range( self.grid_sys.nodes_n ):  
            
                x = self.grid_sys.state_from_node_id[ s , : ]

                # One steps costs - Q values
                Q = np.zeros( self.grid_sys.actions_n  ) 
                
                # For all control actions
                for a in range( self.grid_sys.actions_n ):
                    
                    # If action is in allowable set
                    if self.grid_sys.action_isok[s,a]:
                        
                        # if the next state is not out-of-bound
                        if self.grid_sys.x_next_isok[s,a]:
                            
                            u = self.grid_sys.input_from_action_id[ a , : ]
                            
                            y = self.sys.h(x, u, 0) # TODO remove y from cost                    
                                
                            # This is only for time-independents
                            x_next        = self.grid_sys.x_next_table[s,a,:]

                            # Cost-to-go of a given action
                            J_next = self.J_interpol( x_next[0] , x_next[1])
                            Q[ a ] = self.cf.g(x, u, y, self.t ) * self.grid_sys.dt + J_next
                            
                        else:
                            
                            # Out of bound cost
                            Q[ a ] = self.cf.INF # TODO add option to customize this
                        
                    else:
                        # Not allowable input at this state
                        Q[ a ] = self.cf.INF
                        
                        
                self.J[ s ]  = Q.min()
                self.pi[ s ] = Q.argmin()
                
                # Impossible situation ( unaceptable situation for any control actions )
                if self.J[ s ] > (self.cf.INF-1) :
                    self.pi[ s ] = -1

        
    

            
            
            


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
    grid_sys = discretizer.GridDynamicSystem( sys , [101,101] , [3] )

    # Cost Function
    qcf = sys.cost_function

    qcf.xbar = np.array([ -3.14 , 0 ]) # target
    qcf.INF  = 10000

    # DP algo
    #dp = DynamicProgramming( grid_sys, qcf )
    dp = DynamicProgrammingWithLookUpTable( grid_sys, qcf)
    
    dp.compute_steps(50)
    
    # not validated!!
    
    grid_sys.plot_grid_value( dp.J_next )
    grid_sys.plot_control_input_from_policy( dp.pi , 0)
    
    
    interpol = grid_sys.compute_interpolation_function( dp.J_next ) 
    
    
    a = dp.pi_list[ -1 ]
    
    ctl = LookUpTableController( grid_sys , a )
    
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
    