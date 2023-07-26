#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 18 12:30:17 2023

@author: alex
"""

import numpy as np

from pyro.dynamic  import drone
from pyro.planning import dynamicprogramming

sys  = drone.SpeedControlledDrone2D()


sys.obstacles = [
         np.array([ 0.  , 20. ]) ,
         np.array([ 0.  , 25. ]) ,
         np.array([ 0.  , 30. ]) ,
         np.array([ 5.  , 20. ]) ,
         np.array([ 5.  , 25. ]) ,
         np.array([ 5.  , 30. ]) ,
         np.array([ 10. , 20. ]) ,
         np.array([ 10. , 25. ]) ,
         np.array([ 10. , 30. ]) ,
         np.array([ 2.5 , 35  ]) ,
         np.array([ 7.5 , 35  ]) ,
        ]
        
x_start = np.array([8.0,0])
x_goal  = np.array([0.0,0])