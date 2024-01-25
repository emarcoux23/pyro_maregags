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
        
class Boat2D( RigidBody2D ):    
    """

    """

    ############################
    def __init__(self):
        """ """

        RigidBody2D.__init__( self , force_inputs = 2, other_inputs = 0)

        self.input_label = ['Tx','Ty']
        self.input_units = ['[N]','[N]']

         # State working range
        self.x_ub = np.array([+50,+100,+2,10,10,10])
        self.x_lb = np.array([-50,-0,-2,-10,-10,-10])

        self.u_ub = np.array([+100,+100])
        self.u_lb = np.array([+100,-100])

        # Dynamic properties
        self.mass     = 1000.0
        self.inertia  = 1000.0
        self.l_t      = 2.0     # Distance between CG and Thrust vector

        self.gravity        = 9.8

        self.damping_coef   = np.array([ [ 1.0, 1.0, 1.0 ] ,  
                                         [ 1.0, 1.0, 1.0 ] ])

        # Kinematic param
        self.width  = 1.0
        self.height = self.l_t
        
        # rocket drawing

        # Graphic output parameters 
        self.dynamic_domain  = True
        self.dynamic_range   = 10
        
        pts = np.zeros(( 6 , 3 ))
        l   = self.height
        w   = self.width
        
        pts[0,:] = np.array([-l, +w,0])
        pts[1,:] = np.array([-l, -w,0])
        pts[2,:] = np.array([+l, -w,0])
        pts[3,:] = np.array([l+w,0,0])
        pts[4,:] = np.array([+l, +w,0])
        pts[5,:] = np.array([-l, +w,0])
        
        self.drawing_body_pts = pts

    ###########################################################################
    def B(self, q , u ):
        """ 
        Actuator Matrix  : dof x m
        """
        
        B = np.zeros((3,2))
        
        B[0,0] = 1
        B[1,1] = 1
        B[2,1] = - self.l_t 
        
        return B
    
        
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
        
        vx  = u[0] / self.u_ub[0] * self.height
        vy  = u[1] / self.u_ub[1] * self.height
        offset = -self.height
        
        pts_body = drawing.arrow_from_components( vx , vy , x = offset, origin = 'tip'  )    
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
    
    sys = Boat2D()

    sys.x0[0] = 0
    sys.x0[1] = 0
    sys.x0[2] = 0

    sys.x0[3] = 1.0
    sys.x0[4] = 0.0
    sys.x0[5] = 0.0
    
    sys.ubar[0] = 100
    sys.ubar[1] = 100
    
    sys.plot_trajectory('xu')
    sys.animate_simulation()