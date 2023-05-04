# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 12:05:08 2018

@author: Alexandre
"""


import matplotlib
import matplotlib.pyplot as plt
import sys as python_system

###############################################################################
import numpy as np
###############################################################################
from pyro.dynamic import pendulum
###############################################################################

#matplotlib.use('Qt5Agg')
#plt.ion()

print('The current matplotlib backend is:', matplotlib.get_backend() )
print('Matplotlib interactive mode is activated: ', plt.isinteractive() )
print('The python script is running interactive', hasattr(python_system, 'ps1') )

sys = pendulum.DoublePendulum()

# Simultation
sys.x0  = np.array([-0.1,0,0,0])
sys.plot_trajectory()
sys.plot_phase_plane_trajectory(0, 2)
sys.animate_simulation()
#sys.animate_simulation( is_3d = True )
sys.show3( [1,1] )

