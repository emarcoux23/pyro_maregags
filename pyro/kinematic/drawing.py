#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 09:12:12 2023

@author: alex
"""

###############################################################################
import numpy as np
import matplotlib.pyplot as plt
###############################################################################
from pyro.kinematic import geometry
###############################################################################



###############################################################################
def transform_points_2D( A_T_B , pts_B ):
    """
    
    Take a list of pts in a given frame B and express them in frame A base on 
    transformation matrix A_T_B

    Parameters
    ----------
    pts : TYPE
        DESCRIPTION.
    T : TYPE
        DESCRIPTION.

    Returns
    -------
    pts_transformed : TYPE
        DESCRIPTION.

    """
    
    pts_A = np.zeros( pts_B.shape )
    
    # For all pts in the list
    for i in range(pts_B.shape[0]):
        
        pts_A[ i ] = A_T_B @ pts_B[ i ]
    
    return pts_A
