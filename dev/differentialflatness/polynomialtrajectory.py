# -*- coding: utf-8 -*-

###############################################################################
import numpy as np
import matplotlib.pyplot as plt
from pyro.analysis  import graphical
###############################################################################

p = np.array([ 0 , 0 , 0 , 0 , 0 , 0 ])

# Time
t = np.linspace(0,10,100)

# Position
x = p[0] + p[1]*t + p[2]*t**2 + p[3]*t**3 + p[4]*t**4 + p[5]*t**5
dx = p[1] + 2*p[2]*t + 3*p[3]*t**2 + 4*p[4]*t**3 + 5*p[5]*t**4
ddx = 2*p[2] + 6*p[3]*t + 12*p[4]*t**2 + 20*p[5]*t**3
dddx = 6*p[3] + 24*p[4]*t + 60*p[5]*t**2
ddddx = 24*p[4] + 120*p[5]*t


fig , ax = plt.subplots(6, figsize=graphical.default_figsize, dpi= graphical.default_dpi, frameon=True)
        

ax[0].plot( t , x , 'b')
ax[0].set_ylabel('x', fontsize=graphical.default_fontsize)
ax[0].set_xlabel('v', fontsize=graphical.default_fontsize )
ax[0].tick_params( labelsize = graphical.default_fontsize )
ax[0].grid(True)

ax[1].plot( t , dx , 'b')
ax[1].set_ylabel('dx', fontsize=graphical.default_fontsize)
ax[1].set_xlabel('t', fontsize=graphical.default_fontsize )
ax[1].tick_params( labelsize = graphical.default_fontsize )
ax[1].grid(True)

ax[2].plot( t , ddx , 'b')
ax[2].set_ylabel('ddx', fontsize=graphical.default_fontsize)
ax[2].set_xlabel('t', fontsize=graphical.default_fontsize )
ax[2].tick_params( labelsize = graphical.default_fontsize )
ax[2].grid(True)

ax[3].plot( t , dddx , 'b')
ax[3].set_ylabel('dddx', fontsize=graphical.default_fontsize)
ax[3].set_xlabel('t', fontsize=graphical.default_fontsize )
ax[3].tick_params( labelsize = graphical.default_fontsize )
ax[3].grid(True)

ax[4].plot( t , ddddx , 'b')
ax[4].set_ylabel('ddddx', fontsize=graphical.default_fontsize)
ax[4].set_xlabel('t', fontsize=graphical.default_fontsize )
ax[4].tick_params( labelsize = graphical.default_fontsize )
ax[4].grid(True)

fig.tight_layout()
fig.canvas.draw()

plt.show()