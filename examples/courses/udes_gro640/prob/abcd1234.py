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
    T = np.array([[np.cos(theta), -np.sin(theta) * np.cos(alpha),  np.sin(theta) * np.sin(alpha), r * np.cos(theta)],
                  [np.sin(theta),  np.cos(theta) * np.cos(alpha), -np.cos(theta) * np.sin(alpha), r * np.sin(theta)],
                  [0,              np.sin(alpha),                  np.cos(alpha),                 d],
                  [0,              0,                              0,                             1]])
    return T

def dhs2T( r , d , theta, alpha ):

    # POUR S'ASSURER QUE LES INPUTS SOIENT DE LA MÊME TAILLE
    if len(r) != len(d) or len(r) != len(theta) or len(r) != len(alpha):
        raise ValueError("Lengths of input arrays (r, d, theta, alpha) must be equal.")

    # MATRICE DE TRANSFORMATION ENTRE LES 2 PREMIERS JOINTS
    WTT = dh2T(r[0], d[0], theta[0], alpha[0])

    # CONSTRUCTION DE LA MATRICE DE TRANSFORMATION POUR TOUS LES JOINTS RESTANTS
    for i in range(1, len(r)):
        new_WTT = dh2T(r[i], d[i], theta[i], alpha[i])
        WTT = np.dot(WTT, new_WTT)

    return WTT

def f(q):

    # Pour la clarté :)
    q1 = q[0]; q2 = q[1]; q3 = q[2]; q4 = q[3]; q5 = q[4]

    # Paramètres DH
    d =     [0.072, 0.075,        0,            0,     0           ]
    theta = [q1,    q2 + np.pi/2, q3 + np.pi/2, q4,    q5 - np.pi/2]
    r =     [0,     0.033,        0.155,        0.135, 0.081       ]
    alpha = [0,     np.pi/2,      0,            0,     np.pi/2     ]
    
    # Matrice de transformation de E vers le world
    w_T_e = dhs2T(r[:5], d[:5], theta[:5], alpha[:5])

    # Vecteur du world vers le Tool en base World extrait de la matrice de rotation
    r = w_T_e[:3, 3]
    
    return r


###################
# Part 2
###################
    
class CustomPositionController( EndEffectorKinematicController ) :
    
    ############################
    def __init__(self, manipulator ):

        EndEffectorKinematicController.__init__( self, manipulator, 1)      
    
    #############################
    def c( self , y , r , t = 0 ):
        
        # Feedback from sensors
        q = y
        
        # Jacobian computation
        J = self.J( q )
        
        # Ref
        r_desired   = r
        r_actual    = self.fwd_kin( q )
        
        # Error
        e  = r_desired - r_actual
        
        dq = np.zeros( self.m )  # place-holder de bonne dimension
        
        ##################################
        # Votre loi de commande ici !!!
        ##################################

        # Paramètres de la loi de commande
        lamda = 2
        K = 5

        # Éléments utiles pour la loi de commande
        JT = np.transpose(J)
        I = np.identity(3)
        lamba2_I = (lamda*lamda*I)

        # Loi de commande qui pénalise les grandes vitesses
        dq = np.linalg.inv((JT @ J) + (lamba2_I)) @ JT @ e * K

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

        self.step = 0
        
        # Label
        self.name = 'Custom Drilling Controller'
        
        
    #############################
    def c( self , y , r , t = 0 ):

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

        dr = self.robot_model.J(q) @ dq  # End-effector actual speed
            
        ##################################
        # Votre loi de commande ici !!!
        ##################################

       
        if self.step == 0:      # Approche rapide au dessus du trou
            r_d = np.array([0.25, 0.25, 0.43])
            k = 100
            b = 40
        elif self.step == 1:    # Approche précise (linéaire) au dessus du trou
            r_d = np.array([0.25, 0.25, 0.401])
            k = 60
            b = 40
        elif self.step == 2:     # Percage
            r_d = np.array([0.25, 0.25, 0.20])
            k = 10
            b = 100
        elif self.step == 3:     # Retrait linéaire au dessus du trou
            r_d = np.array([0.25, 0.25, 0.43])
            k = 1000
            b = 50
        elif self.step == 4:     # Retour position neutre
            r_d = np.array([0.5, 0.0, 0.50])
            k = 100
            b = 40
        else:                    # Retour position neutre
            r_d = np.array([0.5, 0.0, 0.50])
            k = 100
            b = 40

        # Matrice de constantes Kp et Kd
        K = np.diag([k, k, k])
        B = np.diag([b, b, b])

        # Vitesse désirée de 0 m/s à la fin du mouvement
        dr_d = np.array([0, 0, 0])

        # Erreur en position et en vitesse
        r_e = r_d - r
        dr_e = dr_d - dr
        
        # Force à appliquer à l'effecteur
        f_e = np.dot(K, r_e) + np.dot(B, dr_e)

        # 200N en Z- si on est à l'étape du percage
        if self.step == 2:
            f_e[2] = -200

        # Calcul de la commande en torque aux joints
        u = np.dot(J.T, f_e) + g

        # Calcul des normes du vecteur erreur position et vitesse
        norm_r_e  = np.linalg.norm(r_e)
        norm_dr_e = np.linalg.norm(dr_e)

        # Si la position est atteinte durant l'étape de percage, on passe à la prochaine étape
        if norm_r_e < 0.01 and self.step == 2:
            self.step += 1

        # Si la position est atteinte et que la vitesse est nulle, on passe à la prochaine étape
        elif norm_r_e < 0.01 and norm_dr_e < 0.01 and self.step < 4:
            self.step += 1
        
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

    a_max = 1
    v_max = 1
    T = t_f
    t = np.linspace(0, T, l)
    
    for i in range(l):
        if t[i] < 0 and t[i] <= v_max/a_max:
            s = 1/2*(a_max*t[i]**2)
            s_dot = a_max*t[i]
            s_dot_dot = a_max
        if t[i] > v_max/a_max and t[i] <= T - v_max/a_max:
            s = v_max*t[i] - (v_max**2)/(2*a_max)
            s_dot = v_max
            s_dot_dot = 0
        if (T - v_max/a_max) < t[i] and t[i] <= T:
            s = ((2*a_max*v_max*T)-(2*v_max**2)-((a_max**2)*((t[i]-T)**2)))/(2*a_max) 
            s_dot = a_max*(T-t[i])
            s_dot_dot = -a_max
        else :
            s = None
            s_dot = None
            s_dot_dot = None

    
    ## ajouter le code p.169 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    
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
    J = self.J(q)
    J_inv = np.linalg.inv(J)
    J_dot = np.diff(J)

    # add for loop
    #q = 
    dq = np.dot(J_inv, dr)
    ddq = np.dot(J_inv, (ddr - (np.dot(J_dot, dq))))
    
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
    
    for i in range(l):
        tau[:, i] = manipulator.inverse_dynamics(q[:, i], dq[:, i], ddq[:, i])
    
    return tau