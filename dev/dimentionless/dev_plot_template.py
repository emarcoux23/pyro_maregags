#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 13:26:10 2023

@author: alex
"""
import matplotlib.pyplot as plt

##################################
# Dimensionless policy plot
##################################

fig = plt.figure(figsize= (4, 3), dpi=300, frameon=True)
fig.canvas.manager.set_window_title( 'dimentionless policy' )
ax  = fig.add_subplot(1, 1, 1)

xname = r'$\theta^*$'#self.sys.state_label[x] #+ ' ' + self.sys.state_units[x]
yname = r'$\dot{\theta}^* = \frac{\dot{\theta}}{\omega}$'#self.sys.state_label[y] #+ ' ' + self.sys.state_units[y]
zname = r'$\tau^*=\frac{\tau}{mgl}$'

ax.set_ylabel(yname, fontsize=10)
ax.set_xlabel(xname, fontsize=10)

x_level = grid_sys.x_level[ 0 ] * 1
y_level = grid_sys.x_level[ 1 ] * (1 / omega)

##################################
### Create grid of data and plot
#################################

u = grid_sys.get_input_from_policy( dp.pi , 0 )

u2 =  u * (1/mgl)

J_grid_nd = grid_sys.get_grid_from_array( u2 ) 

J_grid_2d = grid_sys.get_2D_slice_of_grid( J_grid_nd , 0 , 1 )

mesh = ax.pcolormesh( x_level, y_level, J_grid_2d.T, 
               shading='gouraud' , cmap = 'bwr') #, norm = colors.LogNorm()

#mesh.set_clim(vmin=jmin, vmax=jmax)

##################################
# Figure param
##################################

ax.tick_params( labelsize = 10 )
ax.grid(True)

cbar = fig.colorbar( mesh )

cbar.set_label(zname, fontsize=10 , rotation=90)

fig.tight_layout()
fig.show()