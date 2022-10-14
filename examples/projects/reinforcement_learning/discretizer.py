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
    def __init__(self, sys , x_grid_dim = [ 101 , 101 ], u_grid_dim = [ 11 ] , dt = 0.05 , lookup = True ):
        
        # Dynamic system class
        self.sys = sys 
        
        # Discretization Parameters
        
        # time discretization
        self.dt    = dt        
        
        # Grid size
        self.x_grid_dim = np.array( x_grid_dim )
        self.u_grid_dim = np.array( u_grid_dim )
        
        # Range
        self.x_range     = self.sys.x_ub - self.sys.x_lb
        self.u_range     = self.sys.u_ub - self.sys.u_lb
        
        # spatial step size
        self.x_step_size = self.x_range / ( self.x_grid_dim - 1 )
        self.u_step_size = self.u_range / ( self.u_grid_dim - 1 )
        
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
            self.compute_xnext_table()
            self.compute_action_set_table()
            self.compute_nearest_snext_table()
            
        
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
    def compute_action_set_table(self):
        """ Compute a boolen table describing the action set for each node """
            
        # Evaluation lookup tables      
        self.action_isok   = np.zeros( ( self.nodes_n , self.actions_n ) , dtype = bool )
        
        # For all state nodes        
        for node_id in range( self.nodes_n ):  
            
                x = self.state_from_node_id[ node_id , : ]
            
                # For all control actions
                for action_id in range( self.actions_n ):
                    
                    u = self.input_from_action_id[ action_id , : ]

                    u_ok = self.sys.isavalidinput(x,u)

                    self.action_isok[ node_id , action_id ] = u_ok
                    
                    
    ##############################
    def compute_xnext_table(self):
        """ Compute a x_next lookup table for the forward dynamics """
            
        # Evaluation lookup tables
        self.x_next_table = np.zeros( ( self.nodes_n , self.actions_n , self.sys.n ) , dtype = float ) # lookup table for dynamic
        
        # For all state nodes        
        for node_id in range( self.nodes_n ):  
            
                x = self.state_from_node_id[ node_id , : ]
            
                # For all control actions
                for action_id in range( self.actions_n ):
                    
                    u = self.input_from_action_id[ action_id , : ]
                    
                    # Compute next state for all inputs
                    x_next = self.sys.f( x , u ) * self.dt + x
                    
                    self.x_next_table[ node_id ,  action_id , : ] = x_next
                    
    
    ##############################
    def compute_nearest_snext_table(self):
        """ Compute s_next lookup table for the forward dynamics """
            
        # Evaluation lookup tables
        self.s_next_table = np.zeros( ( self.nodes_n , self.actions_n ) , dtype = int ) # lookup table for dynamic
        
        # For all state nodes        
        for node_id in range( self.nodes_n ):  
            
                x = self.state_from_node_id[ node_id , : ]
            
                # For all control actions
                for action_id in range( self.actions_n ):
                    
                    # Compute the control input
                    u = self.input_from_action_id[ action_id , : ]
                    
                    # Compute next state
                    x_next = self.sys.f( x , u ) * self.dt + x
                    
                    # Compute nearest node
                    s_next = self.get_nearest_node_id_from_state( x_next )
                    
                    # Put in the lookup table
                    self.s_next_table[ node_id ,  action_id ] = s_next
                    
                    
                        
                        
    
    ##############################
    ### Quick shorcut 
    ##############################
    
    ##############################
    def get_index_from_state(self, x ):
        """  
        Return state position on the grid in terms of fractionnal indexes 
        """
        
        indexes = np.zeros( self.sys.n , dtype = float )
        
        # for all state dimensions
        for i in range( self.sys.n ):
            
            indexes[i] = ( x[i] - self.sys.x_lb[i] ) / self.x_range[i] * ( self.x_grid_dim[i] - 1 )
        
        return indexes
    
    
    ##############################
    def get_nearest_index_from_state(self, x ):
        """  
        Return nearest indexes on the state-space grid from a state
        """
        
        # Round the indexes to the nearest integer
        nearest_indexes = np.rint( self.get_index_from_state( x ) ).astype(int)
        
        clipped_indexes = np.clip( nearest_indexes , 0 , self.x_grid_dim - 1 )
        
        # SHould we return -1 for out of bounds indexes??
        
        return clipped_indexes
    
    
    ##############################
    def get_nearest_node_id_from_state(self, x ):
        """  
        Return the node id that is the closest on the grid from x
        """
        
        indexes = tuple( self.get_nearest_index_from_state( x ) )
        
        node_id = self.node_id_from_index[ indexes ]
        
        return node_id
            
                
                



'''
#################################################################
##################          Main                         ########
#################################################################
'''


if __name__ == "__main__":     
    """ MAIN TEST """

    from pyro.dynamic  import pendulum

    sys  = pendulum.SinglePendulum()
    
    G =  GridDynamicSystem( sys )
    
    sys.x_ub = np.array([2.0,2.0])
    sys.x_lb = np.array([-2.0,-2.0])
    sys.u_ub = np.array([1.0,1.0])
    sys.u_lb = np.array([0.0,0.0])
    
    g = GridDynamicSystem( sys , [ 5, 5] , [2] )
    
    
