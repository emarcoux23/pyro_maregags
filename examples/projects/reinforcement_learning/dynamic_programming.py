#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 20:48:32 2022

@author: alex
"""

import numpy as np
import matplotlib.pyplot as plt
import time

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
        Pyro controller based on a discretized lookpup table of control inputs

        Parameters
        ----------
        grid_sys : pyro GridDynamicSystem class
            A discretized dynamic system
        pi : numpy array, dim =  self.grid_sys.nodes_n , dtype = int
            A list of action index for each node id
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
            
            u_k      = self.grid_sys.get_input_from_policy( self.pi, k)
            self.u_interpol.append( self.grid_sys.compute_interpolation_function( u_k , 
                                                                                 self.interpol_method[k],
                                                                                 bounds_error = False   , 
                                                                                 fill_value = 0  ) )
        
    
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
    
    

###############################################################################
### DP Algo
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
        self.alpha                = 1.0 # facteur d'oubli exponentiel
        self.interpol_method      ='linear' # "linear”, “nearest”, “slinear”, “cubic”, and “quintic”
        self.save_time_history    = True
        
        # Memory
        self.t = self.tf
        self.k = 0
        self.start_time = time.time()
        
        # Final cost
        self.evaluate_terminal_cost()
        
        
        if self.save_time_history:

            self.t_list  = []
            self.J_list  = []
            self.pi_list = []
            
            # Final value in lists
            self.J_list.append( self.J_next  )
            self.t_list.append( self.tf )
            self.pi_list.append( None )
        
        
    ##############################
    def evaluate_terminal_cost(self):
        """ initialize cost-to-go and policy """

        self.J_next  = np.zeros( self.grid_sys.nodes_n , dtype = float )
        self.pi      = None

        # Initial cost-to-go evaluation       
        for s in range( self.grid_sys.nodes_n ):  
            
                xf = self.grid_sys.state_from_node_id[ s , : ]
                
                # Final Cost
                self.J_next[ s ] = self.cf.h( xf , self.tf )
                
    
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
        self.J_interpol = self.grid_sys.compute_interpolation_function( self.J_next               , 
                                                                        self.interpol_method      , 
                                                                        bounds_error = False      , 
                                                                        fill_value = 0  )
                        
                
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
                        
                    # If action is in allowable set
                    if self.sys.isavalidinput(x,u):
                        
                        x_next = self.sys.f(x,u,self.t ) * self.grid_sys.dt + x
                        
                        # if the next state is not out-of-bound
                        if self.sys.isavalidstate(x_next):

                            # Cost-to-go of a given action
                            J_next = self.J_interpol( x_next )
                            Q[ a ] = self.cf.g(x, u, self.t ) * self.grid_sys.dt + self.alpha * J_next
                            
                        else:
                            
                            # Out of bound terminal cost
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
        
        # Computation time
        elapsed_time = time.time() - self.start_time
        
        # Convergence check        
        delta = self.J - self.J_next
        j_max     = self.J.max()
        delta_max = delta.max()
        delta_min = delta.min()
        
        print('%d t:%.2f Elasped:%.2f max: %.2f dmax:%.2f dmin:%.2f' % (self.k,self.t,elapsed_time,j_max,delta_max,delta_min) )
        
        # Update J_next
        self.J_next = self.J
        
        # List in memory
        if self.save_time_history:
            self.J_list.append( self.J  )
            self.t_list.append( self.t )
            self.pi_list.append( self.pi )

    
    ################################
    def compute_steps(self, l = 50 , animate_iteration = False ):
        """ compute number of step """
        
        if animate_iteration: self.plot_cost2go()
               
        for i in range(l):
            self.initialize_backward_step()
            self.compute_backward_step()
            self.finalize_backward_step()
            if animate_iteration: self.update_cost2go_plot()
            
            
    ################################
    def plot_cost2go(self , jmax = 1000 , i = 0 , j = 1 ):
               
        fig, ax, pcm = self.grid_sys.plot_grid_value( self.J_next , 'Cost-to-go' , i , j , jmax , 0 )
        
        text = ax.text(0.05, 0.05, '', transform=ax.transAxes, fontsize = 8 )
        
        self.cost2go_fig = [fig, ax, pcm, text]
        
        plt.pause( 0.001 )
        plt.ion()
        
        
    ################################
    def update_cost2go_plot(self):
        
        J_grid = self.grid_sys.get_grid_from_array( self.J_next )
        
        J_2d = self.grid_sys.get_2D_slice_of_grid( J_grid , 0 , 1 )
               
        self.cost2go_fig[2].set_array( np.ravel( J_2d.T ) )
        self.cost2go_fig[3].set_text('Optimal cost2go at time = %4.2f' % ( self.t ))
        
        plt.pause( 0.001 )
        
        
    ################################
    def save_latest(self, name = 'test_data'):
        """ save cost2go and policy of the latest iteration (further back in time) """
        
        np.save(name + '_J_inf', self.J_next)
        np.save(name + '_pi_inf', self.pi.astype(int) )
        
    
    ################################
    def load_J_next(self, name = 'test_data'):
        """ Load J_next from file """
        
        try:

            self.J_next = np.load( name + '_J_inf'   + '.npy' )
            #self.pi     = np.load( name + '_pi_inf'  + '.npy' ).astype(int)
            
        except:
            
            print('Failed to load J_next ' )
            


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
                                
                            # This is only for time-independents
                            x_next        = self.grid_sys.x_next_table[s,a,:]

                            # Cost-to-go of a given action
                            J_next = self.J_interpol( x_next )
                            Q[ a ] = self.cf.g(x, u, self.t ) * self.grid_sys.dt + self.alpha * J_next
                            
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
    
class DynamicProgrammingWithLookUpTable2( DynamicProgramming ):
    """ Dynamic programming on a grid sys """
    
    ############################
    def __init__(self, grid_sys , cost_function , final_time = 0 ):
        
        DynamicProgramming.__init__(self, grid_sys, cost_function, final_time)
        
        self.compute_cost_lookuptable()
    
    
    ###############################
    def compute_cost_lookuptable(self):
        """ One step of value iteration """
        
        self.G = np.zeros( ( self.grid_sys.nodes_n , self.grid_sys.actions_n ) , dtype = float )

        # For all state nodes        
        for s in range( self.grid_sys.nodes_n ):  
            
                x = self.grid_sys.state_from_node_id[ s , : ]
                
                # For all control actions
                for a in range( self.grid_sys.actions_n ):
                    
                    # If action is in allowable set
                    if self.grid_sys.action_isok[s,a]:
                        
                        # if the next state is not out-of-bound
                        if self.grid_sys.x_next_isok[s,a]:
                            
                            u = self.grid_sys.input_from_action_id[ a , : ]  
                            
                            self.G[ s , a ] = self.cf.g(x, u, self.t ) * self.grid_sys.dt
                        
                        else:
                            # Out of bound cost
                            self.G[ s , a ] = self.cf.INF
                    
                    else:
                        # Not allowable input at this state
                        self.G[ s , a ] = self.cf.INF
    
                
    ###############################
    def compute_backward_step(self):
        """ One step of value iteration """
        
        self.Q       = np.zeros( ( self.grid_sys.nodes_n , self.grid_sys.actions_n ) , dtype = float )
        self.Jx_next = np.zeros( ( self.grid_sys.nodes_n , self.grid_sys.actions_n ) , dtype = float )
        
        # Computing the J_next of all x_next in the look-up table
        self.Jx_next = self.J_interpol( self.grid_sys.x_next_table )
        
        # Matrix version of computing all Q values
        self.Q       = self.G + self.alpha * self.Jx_next
                        
        self.J  = self.Q.min( axis = 1 )
        self.pi = self.Q.argmin( axis = 1 )
                

                    
                    

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
                                
                            # This is only for time-independents
                            x_next        = self.grid_sys.x_next_table[s,a,:]

                            # Cost-to-go of a given action
                            J_next = self.J_interpol( x_next[0] , x_next[1])
                            Q[ a ] = self.cf.g(x, u, self.t ) * self.grid_sys.dt + self.alpha * J_next
                            
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
    import costfunction

    sys  = pendulum.SinglePendulum()

    # Discrete world 
    grid_sys = discretizer.GridDynamicSystem( sys , [101,101] , [3] )

    # Cost Function
    qcf = costfunction.QuadraticCostFunction.from_sys(sys)

    qcf.xbar = np.array([ -3.14 , 0 ]) # target
    qcf.INF  = 10000

    # DP algo
    #dp = DynamicProgramming( grid_sys, qcf )
    #dp2 = DynamicProgrammingWithLookUpTable2( grid_sys, qcf)

    