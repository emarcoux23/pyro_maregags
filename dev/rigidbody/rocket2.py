# -*- coding: utf-8 -*-

###############################################################################
import numpy as np
###############################################################################
from pyro.dynamic import system
from pyro.kinematic import geometry
from pyro.kinematic import drawing
###############################################################################

from rigidbody import RigidBody2D

##############################################################################

###############################################################################
        
class Rocket2D( RigidBody2D ):    
    """

    """

    ############################
    def __init__(self):
        """ """

        RigidBody2D.__init__( self , force_inputs = 1, other_inputs = 1)

        self.input_label = ['T','delta']
        self.input_units = ['[N]','[rad]']

         # State working range
        self.x_ub = np.array([+50,+100,+2,10,10,10])
        self.x_lb = np.array([-50,-0,-2,-10,-10,-10])

        self.u_ub = np.array([+5000,+1])
        self.u_lb = np.array([+100,-1])

        # Dynamic properties
        self.mass     = 1000.0
        self.inertia  = 1000.0
        self.l_t      = 1.0 # Distance between CG and Thrust

        self.gravity        = 9.8

        self.damping_coef   = np.array([ [ 1.0, 1.0, 1.0 ] ,  
                                         [ 1.0, 1.0, 1.0 ] ])

        # Kinematic param
        self.width  = 0.2
        self.height = 2.0
        
        # rocket drawing

        # Graphic output parameters 
        self.dynamic_domain  = True
        self.dynamic_range   = 10
        
        pts = np.zeros(( 10 , 3 ))
        l   = self.height
        w   = self.width
        
        pts[0,:] = np.array([ -l, 0,0])
        pts[1,:] = np.array([-l, -w,0])
        pts[2,:] = np.array([+l, -w,0])
        pts[3,:] = np.array([l+w,0,0])
        pts[4,:] = np.array([+l, +w,0])
        pts[5,:] = np.array([-l, +w,0])
        pts[6,:] = pts[0,:]
        pts[7,:] = pts[0,:] + np.array([-w,-w,0])
        pts[8,:] = pts[0,:] + np.array([-w,+w,0])
        pts[9,:] = pts[0,:]
        
        self.drawing_body_pts = pts
    
    ###########################################################################
    def B(self, q , u ):
        """ 
        Actuator Matrix  : dof x m
        """
        
        B = np.zeros((3,1))
        
        delta = u[1]
        
        B[0,0] =  np.cos( delta )
        B[1,0] =  np.sin( delta )
        B[2,0] = - self.l_t * np.sin( delta )
        
        return B
    
    
    ###########################################################################
    def g(self, q ):
        """ 
        Gravitationnal forces vector : dof x 1
        """
        
        # Gravity in inertial frame
        g_a    = np.zeros( self.dof ) 
        g_a[1] = self.mass * self.gravity

        # Gravity in body frame
        g_body = self.N( q ).T @ g_a
        
        return g_body
    
        
    ###########################################################################
    def d(self, q , v , u ):
        """ 
        State-dependent dissipative forces : dof x 1
        """
        
        d = np.zeros(self.dof ) 

        C = self.damping_coef

        # quadratic + linear damping based on 6 coefficients
        d = v*np.abs(v) * C[0,:] + v * C[1,:]
        
        return d
    
    ###########################################################################
    def forward_kinematic_domain(self, q ):

        l = self.height * 3
        
        x = q[0]
        y = q[1]
        z = 0
        
        if self.dynamic_domain:
        
            domain  = [ ( -l + x , l + x ) ,
                        ( -l + y , l + y ) ,
                        ( -l + z , l + z ) ]#  
        else:
            
            domain  = [ ( -l , l ) ,
                        ( -l , l ) ,
                        ( -l , l ) ]#
                
        return domain
    
    
    ###########################################################################
    def forward_kinematic_lines(self, q ):
        
        lines_pts = [] # list of array (n_pts x 3) for each lines
        lines_style = []
        lines_color = []
        
        ###############################
        # ground line
        ###############################
        
        pts      = np.zeros(( 2 , 3 ))
        pts[0,:] = np.array([-10,0,0])
        pts[1,:] = np.array([+10,0,0])
        
        lines_pts.append( pts )
        lines_style.append( '--')
        lines_color.append( 'k' )
        
        ###########################
        #  body
        ###########################
        
        x     = q[0]
        y     = q[1]
        theta = q[2]
        
        W_T_B    = geometry.transformation_matrix_2D( theta , x , y )
        
        pts_B    = self.drawing_body_pts
        pts_W    = drawing.transform_points_2D( W_T_B , pts_B )

        lines_pts.append( pts_W )
        lines_style.append( '-')
        lines_color.append( 'b' )
        
        ###########################
        #  C.G.
        ###########################
        
        pts      = np.zeros(( 1 , 3 ))
        pts[0,:] = np.array([x,y,0])
        
        lines_pts.append( pts )
        lines_style.append( 'o')
        lines_color.append( 'b' )
                
        return lines_pts , lines_style , lines_color
    
    ###########################################################################
    def forward_kinematic_lines_plus(self, x , u , t ):
        """ 
        show trust vectors
        
        
        """
        
        lines_pts = [] # list of array (n_pts x 3) for each lines
        lines_style = []
        lines_color = []
        
        ###########################
        # trust force vector
        ###########################
        
        length  = u[0] / self.u_ub[0] * self.height
        delta   = u[1] 
        offset = -self.height - self.width
        
        pts_body = drawing.arrow_from_length_angle( length, delta, x = offset, origin = 'tip' )
        W_T_B    = geometry.transformation_matrix_2D( x[2], x[0] , x[1] )
        pts_W    = drawing.transform_points_2D( W_T_B , pts_body )
        
        lines_pts.append( pts_W )
        lines_style.append( '-')
        lines_color.append( 'r' )
                
        return lines_pts , lines_style , lines_color

    
'''
#################################################################
##################          Main                         ########
#################################################################
'''


if __name__ == "__main__":     
    """ MAIN TEST """
    
    sys = Rocket2D()

    sys.gravity = 9.81
    
    sys.x0[0] = 0
    sys.x0[1] = 0
    sys.x0[2] = np.pi/2

    sys.x0[3] = 0.0
    sys.x0[4] = 0.0
    sys.x0[5] = 0.0
    
    sys.ubar[0] = sys.gravity * sys.mass * 1.1
    sys.ubar[1] = -0.01
    
    sys.plot_trajectory('xu')
    sys.animate_simulation()