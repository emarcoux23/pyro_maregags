#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 14 23:19:16 2020

@author: alex
------------------------------------


Fichier d'amorce pour les livrables de la problématique GRO640'
"""

# ABSOLUTE PATH TO PYRO LIBRARY
import sys; sys.path.insert(0, "C:/PythonLib/pyro_maregags")

import numpy as np

from pyro.control  import robotcontrollers
from pyro.control.robotcontrollers import EndEffectorPD
from pyro.control.robotcontrollers import EndEffectorKinematicController


###################
# Part 1
###################

def dh2T( r , d , theta, alpha ):
    """
    Parameters
    ----------
    r     : float 1x1
    d     : float 1x1
    theta : float 1x1
    alpha : float 1x1
    
    4 paramètres de DH

    Returns
    -------
    T     : float 4x4 (numpy array)
            Matrice de transformation

    """

    ###################
    # Votre code ici
    ###################

    T = np.array([[np.cos(theta),   -np.sin(theta) * np.cos(alpha),  np.sin(theta) * np.sin(alpha), r * np.cos(theta)],
                  [np.sin(theta),    np.cos(theta) * np.cos(alpha), -np.cos(theta) * np.sin(alpha), r * np.sin(theta)],
                  [0,                np.sin(alpha),                  np.cos(alpha),                 d],
                  [0,                0,                              0,                             1]])

    return T

def dhs2T( r , d , theta, alpha ):
    """
    Parameters
    ----------
    r     : float nx1
    d     : float nx1
    theta : float nx1
    alpha : float nx1
    
    Colonnes de paramètre de DH

    Returns
    -------
    WTT     : float 4x4 (numpy array)
              Matrice de transformation totale de l'outil
    """
    
    ###################
    # Votre code ici
    ###################

    # POUR S'ASSURER QUE LES INPUTS SOIENT CORRECTS
    #TODO corrigé ça
    if len(r) != len(d) or len(r) != len(theta) or len(r) != len(alpha):
        raise ValueError("Lengths of input arrays (r, d, theta, alpha) must be equal.")
    if len(r) <= 1 or len(r) >= 7:
        raise ValueError("Lengths of input arrays (r, d, theta, alpha) must be of size 2 to 7.")

    # MATRICE DE TRANSFORMATION ENTRE LES 2 PREMIERS JOINTS
    WTT = dh2T(r[0], d[0], theta[0], alpha[0])

    # CONSTRUCTION DE LA MATRICE DE TRANSFORMATION POUR TOUS LES JOINTS RESTANTS
    for i in range(1, len(r)):
        new_WTT = dh2T(r[i], d[i], theta[i], alpha[i])
        WTT = np.dot(WTT, new_WTT)

    return WTT

def f(q):
    """
        Parameters
    ----------
    q : float 6x1
        Joint space coordinates

    Returns
    -------
    r : float 3x1 
        Effector (x,y,z) position
    """
    
    ###################
    # Votre code ici
    ###################

    # DH PARAMETERS OF KUKA ROBOT
    r =     [0, -33, 155, 135, 80.5, 9.5]
    d =     [72, 75, 0, 0, 0, 136.6]
    theta = q
    alpha = [0, -np.pi/2, 0, 0, -np.pi/2, np.pi/2]
    
    # CREATE THE TRANSFORMATION MATRIXES FROM ALL BASES TO W
    w_T_a = dh2T (r[0],  d[0],  theta[0],  alpha[0])
    a_T_b = dhs2T(r[:2], d[:2], theta[:2], alpha[:2])
    a_T_c = dhs2T(r[:3], d[:3], theta[:3], alpha[:3])
    a_T_d = dhs2T(r[:4], d[:4], theta[:4], alpha[:4])
    a_T_e = dhs2T(r[:5], d[:5], theta[:5], alpha[:5])
    a_T_tool = dhs2T(r[:6], d[:6], theta[:6], alpha[:6])

    # CREATE THE 1X4 POSITON VECTORS IN THEIR RESPECTIVE BASES
    r_W_to_A_in_A = np.array([[r[0]], [0], [d[0]], [1]])
    r_A_to_B_in_B = np.array([[r[1]], [0], [d[1]], [1]])
    r_B_to_C_in_C = np.array([[r[2]], [0], [d[2]], [1]])
    r_C_to_D_in_D = np.array([[r[3]], [0], [d[3]], [1]])
    r_D_to_E_in_E = np.array([[r[4]], [0], [d[4]], [1]])
    r_E_to_tool_in_tool = np.array([[r[5]], [0], [d[5]], [1]])

    # TRANSFORM ALL THE POSITION VECTORS IN THEIR RESPECTIVE
    # BASES TO THE SAME W BASE
    r_W_to_A_in_W = np.dot(w_T_a, r_W_to_A_in_A)
    r_A_to_B_in_W = np.dot(a_T_b, r_A_to_B_in_B)
    r_B_to_C_in_W = np.dot(a_T_c, r_B_to_C_in_C)
    r_C_to_D_in_W = np.dot(a_T_d, r_C_to_D_in_D)
    r_D_to_E_in_W = np.dot(a_T_e, r_D_to_E_in_E)
    r_E_to_tool_in_W = np.dot(a_T_tool, r_E_to_tool_in_tool)

    # ADD ALL POSITION VECTORS IN THE SAME W BASE
    r_W_to_tool_in_W = r_W_to_A_in_W + r_A_to_B_in_W + r_B_to_C_in_W + r_C_to_D_in_W + r_D_to_E_in_W + r_E_to_tool_in_W

    # 1x4 MATRIX TO 1X3 MATRIX TO ONLY KEEP XYZ
    r = r_W_to_tool_in_W[:-1]
    
    return r


###################
# Part 2
###################
    
class CustomPositionController( EndEffectorKinematicController ) :
    
    ############################
    def __init__(self, manipulator ):
        """ """
        
        EndEffectorKinematicController.__init__( self, manipulator, 1)
        
        ###################################################
        # Vos paramètres de loi de commande ici !!
        ###################################################
        
    
    #############################
    def c( self , y , r , t = 0 ):
        """ 
        Feedback law: u = c(y,r,t)
        
        INPUTS
        y = q   : sensor signal vector  = joint angular positions      dof x 1
        r = r_d : reference signal vector  = desired effector position   e x 1
        t       : time                                                   1 x 1
        
        OUPUTS
        u = dq  : control inputs vector =  joint velocities             dof x 1
        
        """
        
        # Feedback from sensors
        q = y
        
        # Jacobian computation
        J = self.J( q )
        
        # Ref
        r_desired   = r
        r_actual    = self.fwd_kin( q )
        
        # Error
        e  = r_desired - r_actual
        
        ################
        dq = np.zeros( self.m )  # place-holder de bonne dimension
        
        ##################################
        # Votre loi de commande ici !!!
        ##################################

        lamda = 2
        K = 5
        JT = np.transpose(J)
        I = np.identity(3)

        dq = np.linalg.inv((JT @ J) + (lamda*lamda*I)) @ JT @ e * K

        return dq
    
    
###################
# Part 3
###################
        

        
class CustomDrillingController( robotcontrollers.RobotController ) :
    """ 

    """
    
    ############################
    def __init__(self, robot_model ):
        """ """
        
        super().__init__( dof = 3 )
        
        self.robot_model = robot_model
        
        # Label
        self.name = 'Custom Drilling Controller'
        
        
    #############################
    def c( self , y , r , t = 0 ):
        """ 
        Feedback static computation u = c(y,r,t)
        
        INPUTS
        y  : sensor signal vector     p x 1
        r  : reference signal vector  k x 1
        t  : time                     1 x 1
        
        OUPUTS
        u  : control inputs vector    m x 1
        
        """
        
        # Ref
        f_e = r
        
        # Feedback from sensors
        x = y
        [ q , dq ] = self.x2q( x )
        
        # Robot model
        r = self.robot_model.forward_kinematic_effector( q ) # End-effector actual position
        J = self.robot_model.J( q )      # Jacobian matrix
        g = self.robot_model.g( q )      # Gravity vector
        H = self.robot_model.H( q )      # Inertia matrix
        C = self.robot_model.C( q , dq ) # Coriolis matrix
            
        ##################################
        # Votre loi de commande ici !!!
        ##################################
        
        u = np.zeros(self.m)  # place-holder de bonne dimension
        
        return u
        
    
###################
# Part 4
###################
        
    
def goal2r( r_0 , r_f , t_f ):
    """
    
    Parameters
    ----------
    r_0 : numpy array float 3 x 1
        effector initial position
    r_f : numpy array float 3 x 1
        effector final position
    t_f : float
        time 

    Returns
    -------
    r   : numpy array float 3 x l
    dr  : numpy array float 3 x l
    ddr : numpy array float 3 x l

    """
    # Time discretization
    l = 1000 # nb of time steps
    
    # Number of DoF for the effector only
    m = 3
    
    r = np.zeros((m,l))
    dr = np.zeros((m,l))
    ddr = np.zeros((m,l))
    
    #################################
    # Votre code ici !!!
    ##################################
    
    
    return r, dr, ddr


def r2q( r, dr, ddr , manipulator ):
    """

    Parameters
    ----------
    r   : numpy array float 3 x l
    dr  : numpy array float 3 x l
    ddr : numpy array float 3 x l
    
    manipulator : pyro object 

    Returns
    -------
    q   : numpy array float 3 x l
    dq  : numpy array float 3 x l
    ddq : numpy array float 3 x l

    """
    # Time discretization
    l = r.shape[1]
    
    # Number of DoF
    n = 3
    
    # Output dimensions
    q = np.zeros((n,l))
    dq = np.zeros((n,l))
    ddq = np.zeros((n,l))
    
    #################################
    # Votre code ici !!!
    ##################################
    
    
    return q, dq, ddq



def q2torque( q, dq, ddq , manipulator ):
    """

    Parameters
    ----------
    q   : numpy array float 3 x l
    dq  : numpy array float 3 x l
    ddq : numpy array float 3 x l
    
    manipulator : pyro object 

    Returns
    -------
    tau   : numpy array float 3 x l

    """
    # Time discretization
    l = q.shape[1]
    
    # Number of DoF
    n = 3
    
    # Output dimensions
    tau = np.zeros((n,l))
    
    #################################
    # Votre code ici !!!
    ##################################
    
    
    return tau