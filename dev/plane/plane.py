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


###############################################################################
def Transformation_Matrix_2D_from_base_angle( theta , bx , by ):
    
    s = np.sin( theta )
    c = np.cos( theta )
    
    T = np.array([ [ c   , -s ,  bx ] , 
                   [ s   ,  c ,  by ] ,
                   [ 0   ,  0 ,  1  ]   ])
    
    return T


###############################################################################
def Transform_2D_Pts( pts , T ):
    
    pts_transformed = np.zeros( pts.shape )
    
    for i in range(pts.shape[0]):
        
        pts_transformed[ i ] = T @ pts[ i ]
    
    return pts_transformed


###############################################################################
def arrow_from_base_angle( l , theta , bx , by ):

    d = l * 0.15          # length of arrow secondary lines
    
    pts_local = np.array([ [ 0   ,  0 ,  1 ] , 
                           [ l   ,  0 ,  1 ] ,
                           [ l-d ,  d ,  1 ] ,
                           [ l   ,  0 ,  1 ] ,
                           [ l-d , -d ,  1 ] ])
    
    T = Transformation_Matrix_2D_from_base_angle( theta , bx , by )
    
    pts_global = Transform_2D_Pts( pts_local , T )
    
    
    return pts_global


###############################################################################
def arrow_from_tip_angle( l , theta , bx , by ):

    d = l * 0.15          # length of arrow secondary lines
    
    pts_local = np.array([ [ -l  ,  0 ,  1 ] , 
                           [ 0   ,  0 ,  1 ] ,
                           [ -d  ,  d ,  1 ] ,
                           [ 0   ,  0 ,  1 ] ,
                           [  -d , -d ,  1 ] ])
    
    T = Transformation_Matrix_2D_from_base_angle( theta , bx , by )
    
    pts_global = Transform_2D_Pts( pts_local , T )
    
    
    return pts_global




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
        self.width           = 1
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
    
    
    ###########################################################################
    def forward_kinematic_domain(self, q ):
        """ 
        """
        l = self.dynamic_range
        
        x = q[0] + self.width * 5
        y = q[1] + self.width * 1.5
        z = 0
        
        if self.dynamic_domain:
        
            domain  = [ ( -l + x , l + x ) ,
                        ( -l + y , l + y ) ,
                        ( -l + z , l + z ) ]
        else:
            
            domain  = [ ( -l , l ) ,
                        ( -l , l ) ,
                        ( -l , l ) ]#
            
                
        return domain
    
    
    ###########################################################################
    def forward_kinematic_lines(self, q ):
        """ 
        Compute points p = [x;y;z] positions given config q 
        ----------------------------------------------------
        - points of interest for ploting
        
        Outpus:
        lines_pts = [] : a list of array (n_pts x 3) for each lines
        
        """
        
        lines_pts   = [] # list of array (n_pts x 3) for each lines
        lines_style = []
        lines_color = []
        
        
        ###########################
        # Dimensions
        ###########################
        
        w = self.width  # body width
        l = w * 10      # body lenght
        
        ###########################
        # Body
        ###########################
        
        pts      = np.zeros(( 5 , 3 ))
        
        x = q[0]
        y = q[1]
        theta = q[2]
        
        
        body_pts_local = np.array([ [ 0   ,  0   ,  1 ] , 
                                    [ l   ,  0   ,  1 ] ,
                                    [ l-w ,  w   ,  1 ] ,
                                    [ 2*w ,  w   ,  1 ] ,
                                    [ w   ,  3*w ,  1 ] ,
                                    [ 0   ,  3*w ,  1 ] ,
                                    [ 0   ,  0   ,  1 ] ])
        
        T_body = Transformation_Matrix_2D_from_base_angle( theta , x , y )
        
        body_pts_global = Transform_2D_Pts( body_pts_local , T_body )
        
        lines_pts.append( body_pts_global )
        lines_style.append( '-')
        lines_color.append( 'b')
        
        ###########################
        # Wings
        ###########################
        
        pts      = np.zeros(( 2 , 3 ))

        
        wings_pts_local = np.array([ [ 3*w   ,  0.5 * w   ,  1 ] , 
                                    [ 6*w   ,  0.5 * w   ,  1 ] ])
        
        
        wings_pts_global = Transform_2D_Pts( wings_pts_local , T_body )
        
        lines_pts.append( wings_pts_global )
        lines_style.append( '-')
        lines_color.append( 'b')
        
        ###########################
        # bottom line
        ###########################
        
        pts = np.zeros((2,3))
        
        pts[0,0] = -10000
        pts[1,0] = 10000
        pts[0,1] = 0
        pts[1,1] = 0
        
        lines_pts.append( pts )
        lines_style.append('--')
        lines_color.append('k')
        


        
            
        return lines_pts , lines_style , lines_color
    
    
    ###########################################################################
    def forward_kinematic_lines_plus(self, x , u , t ):
        """ 
        plots the force vector
        
        """
        
        lines_pts = [] # list of array (n_pts x 3) for each lines
        lines_style = []
        lines_color = []
        
        w = self.width
        
        q, dq = self.x2q(x)
        
        bx = q[0]
        by = q[1]
        theta = q[2]
        
        trust_vector_lenght = u[0] * 10 * self.width / (self.u_ub[0] - self.u_lb[0])
        
        delta = u[1]
        
        ###########################
        # Trust vector
        ###########################
        
        pts  = arrow_from_tip_angle( trust_vector_lenght , theta , bx , by )
        
        
        lines_pts.append( pts )
        lines_style.append( '-')
        lines_color.append( 'r')
        
        ###########################
        # Control surface
        ###########################
        
        ctl_pts_local = np.array([ [ 0    ,  0   ,  1 ] , 
                                    [ -2*w ,  0   ,  1 ] ])
        
        b_T_c = Transformation_Matrix_2D_from_base_angle( delta , w ,  0.5 * w )
        a_T_b = Transformation_Matrix_2D_from_base_angle( theta , bx , by )
        
        ctl_pts_global = Transform_2D_Pts( ctl_pts_local , a_T_b @ b_T_c )
        
        lines_pts.append( ctl_pts_global )

        lines_style.append( '-')
        lines_color.append( 'b')
        
                
        return lines_pts , lines_style , lines_color



    
'''
#################################################################
##################          Main                         ########
#################################################################
'''


if __name__ == "__main__":     
    """ MAIN TEST """
    
    
    if True:
    
        sys = Plane2D()
        
        sys.x0   = np.array([10,20,0.5,1,1,-0.2])
        
        
        
        def t2u(t):
            
            u = np.array([ t , -0.2 + t * 0.1 ])
            
            return u
            
        sys.t2u = t2u
        #sys.ubar = np.array([ 3 , -0.05 ])
        
        sys.compute_trajectory()
        
        sys.animate_simulation()