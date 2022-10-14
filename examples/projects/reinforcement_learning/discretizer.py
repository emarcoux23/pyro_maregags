# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 10:02:12 2017

@author: alxgr
"""

import numpy as np

'''
################################################################################
'''


class GridDynamicSystem:
    """ Create a discrete gird state-action space for a continous dynamic system """
    
    ############################
    def __init__(self, sys , x_grid_dim = ( 101 , 101 ), u_grid_dim = ( 11 , 1 ) , dt = 0.05 , lookup = True ):
        
        # Dynamic system class
        self.sys = sys 
        
        # Discretization Parameters
        
        # time discretization
        self.dt    = dt        
        
        # Grid size
        self.x_grid_dim = x_grid_dim
        self.u_grid_dim = u_grid_dim
        
        # Options
        self.uselookuptable = lookup
        
        # Initialize
        self.compute()  
        
    ##############################
    def compute(self):
        """  """

        self.discretize_state_space()
        self.discretize_input_space() 
        
        print('\nDiscretization:\n---------------------------------')
        print('State space dimensions:', self.sys.n , ' Input space dimension:', self.sys.m )
        print('Number of nodes:', self.nodes_n , ' Number of actions:', self.actions_n )
        print('Number of node-action pairs:', self.nodes_n * self.actions_n )
        
        self.generate_nodes()
        self.generate_actions()
        
        if self.uselookuptable:
            self.compute_lookuptable()
            
        
    #############################
    def discretize_state_space(self):
        """ Grid the state space """
                        
        self.x_level  = []
        self.nodes_n  = 1
        
        # linespace for each x-axis and total number of nodes
        for i in range(self.sys.n):
            self.x_level.append(  np.linspace( self.sys.x_lb[i]  , self.sys.x_ub[i]  , self.x_grid_dim[i]  ) )
            self.nodes_n        = self.nodes_n * self.x_grid_dim[i]
        
        # 1-D List of nodes
        self.state_from_node_id = np.zeros(( self.nodes_n , self.sys.n ), dtype = float )  # Number of nodes x state dimensions
        self.index_from_node_id = np.zeros(( self.nodes_n , self.sys.n ), dtype = int   )  # Number of nodes x state dimensions
        
        
    #############################
    def discretize_input_space(self):
        """ Grid the input space """
        
        self.u_level    = []
        self.actions_n  = 1
        
        # linespace for each u-axis and total number of actions
        for i in range(self.sys.m):
            self.u_level.append(  np.linspace( self.sys.u_lb[i]  , self.sys.u_ub[i]  , self.u_grid_dim[i]  ) )
            self.actions_n       = self.actions_n * self.u_grid_dim[i]
        
        # 1-D List of actions
        self.input_from_action_id = np.zeros(( self.actions_n , self.sys.m ), dtype = float )  # Number of actions x inputs dimensions
        self.index_from_action_id = np.zeros(( self.actions_n , self.sys.m ), dtype = int   )  # Number of actions x inputs dimensions
        
        
    ##############################
    def generate_nodes(self):
        """ Compute 1-D list of nodes """
        
        # n-D grid of node ID
        self.node_id_from_index = np.zeros( self.x_grid_dim , dtype = int )     # grid of node ID
        
        # For all state nodes
        node_id = 0
        
        if self.sys.n == 2 :
            
            for i in range(self.x_grid_dim[0]):
                for j in range(self.x_grid_dim[1]):
                    
                    # State
                    x = np.array([ self.x_level[0][i]  ,  self.x_level[1][j] ])
                    
                    # State and grid index based on node id
                    self.state_from_node_id[ node_id , : ] = x
                    self.index_from_node_id[ node_id , : ] = np.array([i,j])
                    
                    # Node # based on index ij
                    self.node_id_from_index[i,j] = node_id
    
                    # Increment node number
                    node_id = node_id + 1
                    
                    
        elif self.sys.n == 3:
            
            for i in range(self.x_grid_dim[0]):
                for j in range(self.x_grid_dim[1]):
                    for k in range(self.x_grid_dim[2]):
                    
                        # State
                        x = np.array([ self.x_level[0][i]  ,  self.x_level[1][j]  , self.x_level[2][k] ])
                        
                        # State and grid index based on node #
                        self.state_from_node_id[ node_id , : ] = x
                        self.index_from_node_id[ node_id , : ] = np.array([i,j,k])
                        
                        # Node # based on index ijk
                        self.node_id_from_index[i,j,k] = node_id
        
                        # Increment node number
                        node_id = node_id + 1
                        
                        
                        
        elif self.sys.n == 4:
            
            # NOT yet validated!!!
            
            for i in range(self.x_grid_dim[0]):
                for j in range(self.x_grid_dim[1]):
                    for k in range(self.x_grid_dim[2]):
                        for l in range(self.x_grid_dim[3]):
                    
                            # State
                            x = np.array([ self.x_level[0][i]  ,  self.x_level[1][j]  , self.x_level[2][k] , self.x_level[3][l]])
                            
                            # State and grid index based on node #
                            self.state_from_node_id[ node_id , : ] = x
                            self.index_from_node_id[ node_id , : ] = np.array([i,j,k,l])
                            
                            # Node # based on index ijkl
                            self.node_id_from_index[i,j,k,l] = node_id
            
                            # Increment node number
                            node_id = node_id + 1
                    
        else:
            
            raise NotImplementedError
            
                
    ##############################
    def generate_actions(self):
        """ Compute 1-D list of actions """
        
        # m-D grid of action ID
        self.action_id_from_index = np.zeros( self.u_grid_dim , dtype = int )     # grid of node ID
        
        # For all state nodes
        action_id = 0
        
        # Single input
        
        if self.sys.m == 1 :
        
            for k in range(self.u_grid_dim[0]):
                    
                u = np.array([ self.u_level[0][k] ])
                
                # State and grid index based on node #
                self.input_from_action_id[ action_id , : ] = u
                self.index_from_action_id[ action_id , : ] = k
                
                # Action # based on index k
                self.action_id_from_index[k] = action_id
    
                # Increment node number
                action_id = action_id + 1
                
        elif self.sys.m == 2 :
            
            for k in range(self.u_grid_dim[0]):
                for l in range(self.u_grid_dim[1]):
                    
                    u = np.array([ self.u_level[0][k] , self.u_level[1][l] ])
                    
                    # State and grid index based on node #
                    self.input_from_action_id[ action_id , : ] = u
                    self.index_from_action_id[ action_id , : ] = np.array([k,l])
                    
                    # Action # based on index k
                    self.action_id_from_index[k,l] = action_id
        
                    # Increment node number
                    action_id = action_id + 1
        
        else:
            
            raise NotImplementedError
            
            
    ##############################
    def compute_lookuptable(self):
        """ Compute lookup table for faster evaluation """
            
        # Evaluation lookup tables      
        self.action_isok   = np.zeros( ( self.nodes_n , self.actions_n ) , dtype = bool )
        self.x_next        = np.zeros( ( self.nodes_n , self.actions_n , self.sys.n ) , dtype = float ) # lookup table for dynamic
        
        # For all state nodes        
        for node_id in range( self.nodes_n ):  
            
                x = self.state_from_node_id[ node_id , : ]
            
                # For all control actions
                for action_id in range( self.actions_n ):
                    
                    u = self.input_from_action_id[ action_id , : ]
                    
                    # Compute next state for all inputs
                    x_next = self.sys.f( x , u ) * self.dt + x
                    
                    # validity of the options
                    x_ok = self.sys.isavalidstate(x_next)
                    u_ok = self.sys.isavalidinput(x,u)
                    
                    self.x_next[ node_id ,  action_id , : ] = x_next
                    self.action_isok[ node_id , action_id ] = ( u_ok & x_ok )
                        
                        
    
    ##############################
    ### Quick shorcut 
    ##############################
    
    ##############################
    def x2node(self, x ):
        """  """
        raise NotImplementedError
        
        s = 0
        
        return s
    
    ##############################
    def x2index(self, x ):
        """  """
        raise NotImplementedError
        
        i = 0
        j = 0
        
        return (i,j)
    
    ##############################
    def node2x(self, x ):
        """  """
        raise NotImplementedError
        
        s = 0
        
        return s
    
    
    ##############################
    def index2x(self, u ):
        """  """
        raise NotImplementedError
        
        a = 0
        
        return a
    
    ##############################
    def u2index(self, u ):
        """  """
        raise NotImplementedError
        
        k = 0
        
        return k
            
                
                



'''
#################################################################
##################          Main                         ########
#################################################################
'''


if __name__ == "__main__":     
    """ MAIN TEST """
    
    pass