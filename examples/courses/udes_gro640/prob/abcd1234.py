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

from scipy.optimize import fsolve
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
    q1 = q[0]; q2 = q[1]; q3 = q[2]; q4 = q[3]; q5 = q[4]; q6 = q[5]

    # Paramètres DH
    d =     [0.14700079,        0,              0,          0.00859888,         0.217,      0.009 + q6  ]
    theta = [-(q1 + np.pi/2),   -q2 + np.pi/2,  -q3,        -(q4 + np.pi/2),    -q5,        0           ]
    r =     [-0.03877670,       -0.155,         -0.135,     0,                  0,          0           ]
    alpha = [-np.pi/2,          0,              0,          np.pi/2,            -np.pi/2,   0           ]
    
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
        J = self.J(q)
        
        # Ref
        r_desired = r
        r_actual  = self.fwd_kin(q)
        
        # Error
        e = r_desired - r_actual

        # Paramètres de la loi de commande
        lamda = 3
        K = 15

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
    
def goal2r(r_0, r_f, t_f):

    # Time discretization
    l = 1000 # nb of time steps
    
    # Number of DoF for the effector only
    m = 3
    
    r = np.zeros((m,l))
    dr = np.zeros((m,l))
    ddr = np.zeros((m,l))

    # Discrétisation du temps total sur l pas discrets
    t = np.linspace(0, t_f, l)

    # On trouve v_max et a_max qui respecte ces contraintes:
    # Éq. 10.7: tf = (a_max + v_max^2) / (a_max * v_max)
    # Fig. 10.4: a_max > v_max^2 
    v_max = 0.5
    a_max = 0.5

    # On bâti le profil temporel de type trapézoidal selon Fig. 10.3
    for i in range(l):

        # Phase d'accélération selon Éq. 10.8
        if 0 <= t[i] <= v_max / a_max:
            s = 0.5 * a_max * t[i] * t[i]
            ds = a_max * t[i]
            dds = a_max

        # Phase de vitesse constante selon Éq. 10.9
        elif v_max / a_max < t[i] <= t_f - v_max / a_max:
            s = v_max * t[i] - (v_max**2) / (2 * a_max)
            ds = v_max
            dds = 0

        # Phase de decceleration selon Éq. 10.10
        elif t_f - v_max / a_max < t[i] <= t_f:
            s = ((2 * a_max * v_max * t_f) - (2 * v_max * v_max) -((a_max * a_max) * (t[i] - t_f)**2))/(2 * a_max)
            ds = a_max * (t_f - t[i])
            dds = -a_max

        # Création d'un chemin en ligne droite dans le domaine de l'effecteur: Éq. 10.26
        r[:,i]   = r_0 + (r_f - r_0) * s    # Éq. 10.27
        dr[:,i]  =       (r_f - r_0) * ds   # Éq. 10.28
        ddr[:,i] =       (r_f - r_0) * dds  # Éq. 10.29
    
    return r, dr, ddr

def r2q(r, dr, ddr, manipulator):

    # Time discretization
    l = r.shape[1]
    
    # Number of DoF
    n = 3
    
    # Output dimensions
    q = np.zeros((n,l))
    dq = np.zeros((n,l))
    ddq = np.zeros((n,l))

    # Longueurs des joints du bras
    l1 = manipulator.l1
    l2 = manipulator.l2
    l3 = manipulator.l3

    # Pas de temps avec tf = 3
    dt = 3 / l

    # Pour tous les points discrétiser de la trajectoire
    for i in range(l):

        # On prend le x, y et z du point discrétisé
        x = r[0, i]
        y = r[1, i]
        z = r[2, i]

        # Système d'équation de la cinématique directe trouvé à la main
        # Les équations pour x, y et z sont mises à zéro pour pouvoir solve
        def equations(vars):
            theta_1, theta_2, theta_3 = vars
            equ_1 = x - (l2 * np.cos(theta_2) + l3 * np.cos(theta_2 + theta_3)) * np.cos(theta_1)
            equ_2 = y - (l2 * np.cos(theta_2) + l3 * np.cos(theta_2 + theta_3)) * np.sin(theta_1)
            equ_3 = z - (l1 + l2 * np.sin(theta_2) + l3 * np.sin(theta_2 + theta_3))
            return [equ_1, equ_2, equ_3]

        # Dernière valeur de q trouvée, sinon zéro
        if i < 1:
            last_q = [0, 0, 0]
        else:
            last_q = q[:,i-1]

        # Méthode numérique trouvée pour solve les équations en x, y, z et
        # ainsi retrouver q1, q2, q3. Le last_q permet à la méthode de mieux
        # trouver la prochaine réponse
        q_solved = fsolve(equations, last_q)

        # Les angles trouvées par la méthode ne sont pas toujours normalisées
        # sur le cercle trigonométrique. Le modulo assure un range de 0 à 2pi
        q_solved = [angle % (2 * np.pi) for angle in q_solved]

        # Calcul de la jacobienne inverse
        J_inv = np.linalg.inv(manipulator.J(q_solved))

        # Calcul de q1_dot, q2_dot, q3_dot selon la cinématique différentielle inverse
        dq_solved = J_inv @ dr[:,i]

        # On rempli la matrice des q et dq discret aux bons index
        q[:,i] = q_solved
        dq[:,i] = dq_solved

        # Pour l'accélération, on utilise le pas de temps, le dq présent et le dq précédent
        # pour dériver la vitesse et ainsi trouver l'accélération
        if i > 0:
            ddq[:,i] = (dq_solved - dq[:,i-1]) / dt

    return q, dq, ddq

def q2torque(q, dq, ddq, manipulator):

    # Time discretization
    l = q.shape[1]
    
    # Number of DoF
    n = 3
    
    # Output dimensions
    tau = np.zeros((n,l))
    
    # Pour tous les q, dq et dqq discrétisés
    for i in range(l):

        # Matrice d'inertie
        H = manipulator.H(q[:,i])

        # Matrice des forces de Coriolis
        C = manipulator.C(q[:,i], dq[:,i])

        # Vecteur de froces dissipatrices
        d = manipulator.d(q[:,i], dq[:,i]) 

        # Vecteur de forces conservatrices
        g = manipulator.g(q[:,i])

        # Matrice actionneurs -> forces généralisées
        B = manipulator.B(q[:,i])

        # Selon Éq. 6.4: H*ddq + C*dq + d + g = B*u + J_T*f_RE
        # À préciser que f_RE est la force appliquée à l'effecteur et égale à zéro, donc ignoré
        # À préciser qu'on isole u, qui est ici égal à tau, le torque des 3 joints et donc on 
        # doit multiplier la gauche de l'équation par l'inverse de B
        tau[:, i] = ( (H @ ddq[:,i]) + (C @ dq[:,i]) + d + g + d ) @ np.linalg.inv(B)
    
    return tau