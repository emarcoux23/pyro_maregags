# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 20:28:17 2018

@author: Alexandre
"""
##############################################################################
import numpy as np
import matplotlib.pyplot as plt
##############################################################################
from pyro.dynamic  import longitudinal_vehicule
from pyro.planning import discretizer
from pyro.analysis import costfunction
from pyro.planning import dynamicprogramming
from pyro.analysis import graphical
##############################################################################


m = 1
g = 1
l = 1
t_max_star = 1
q_star = 1
case_name = 'test'
show = True
res = 'std'
legend = 1

# Additionnal fixed domain dimentionless parameters
theta_star  = 2.0 * np.pi
dtheta_star = 1.0 * np.pi
time_star   = 2.0 * np.pi * 20.0

# Combined system parameters
omega = np.sqrt( ( g / l  ) )
mgl   = m * g * l

# Dimentional parameters
t_max  = t_max_star * mgl
q      = q_star * mgl
theta  = theta_star
dtheta = dtheta_star * omega
time   = time_star / omega
J_max  = mgl**2 / omega * time_star * ( ( q_star * theta_star )**2 + t_max_star**2 )

print('\n\nCase :' + case_name )
print('----------------------------------------------------')
print(' m=',m,' g=',g,' l=',l,' t_max=', t_max, ' q=', q)

################################
# Dynamic system definition
################################

sys  = longitudinal_vehicule.

# kinematic
sys.lc1 = l

sys.l1       = sys.lc1
sys.l_domain = sys.lc1 * 2

# dynamic
sys.m1       = m
sys.I1       = 0
sys.gravity  = g
sys.d1       = 0

sys.u_ub[0]  = + t_max
sys.u_lb[0]  = - t_max

sys.x_ub = np.array([ + theta , + dtheta ])
sys.x_lb = np.array([ - theta , - dtheta ])

################################
# Discritized grid
################################

if res == 'test' :

    dt = 0.5
    nx = 21
    nu = 3
    
elif res == 'plus' :
    
    dt = 0.05
    nx = 301
    nu = 101
    
elif res == 'hi' :
    
    dt = 0.025
    nx = 501
    nu = 101
    
else:
    
    dt = 0.05
    nx = 101
    nu = 11
        
grid_sys = discretizer.GridDynamicSystem( sys , [nx,nx] , [nu] , dt , True )

################################
# Cost function
################################

qcf = costfunction.QuadraticCostFunction.from_sys(sys)

qcf.xbar = np.array([ 0 , 0 ]) # target
qcf.INF  = J_max

qcf.Q[0,0] = q ** 2
qcf.Q[1,1] = 0.0

qcf.R[0,0] = 1.0

################################
# Computing optimal policy
################################

dp = dynamicprogramming.DynamicProgrammingWithLookUpTable( grid_sys, qcf )

steps = int( time / dt ) 

dp.compute_steps( steps )


#grid_sys.fontsize = 10
qcf.INF  = 0.1 * J_max
dp.clean_infeasible_set()


##################################
# Fig param
##################################

dpi      = 300
fontsize = 10
figsize  = (4, 3)

##################################
# Dimensional policy plot
##################################

fig = plt.figure( figsize = figsize, dpi=dpi, frameon=True)
fig.canvas.manager.set_window_title( 'dimentionless policy' )
ax  = fig.add_subplot(1, 1, 1)

xname = r'$\theta  \; [rad]$'
yname = r'$\dot{\theta} \; [rad/sec]$'
zname = r'$\tau \; [Nm]$'

sys.state_label[0] = r'$\theta$'
sys.state_label[1] = r'$\dot{\theta}$'
sys.input_label[0] = r'$\tau$'

xrange = 2.0 * np.pi
yrange = np.pi * np.sqrt( 10 / 1. )
zrange = 20.

ax.set_ylabel( yname, fontsize = fontsize )
ax.set_xlabel( xname, fontsize = fontsize )

x_level = grid_sys.x_level[ 0 ] 
y_level = grid_sys.x_level[ 1 ] 


u         = grid_sys.get_input_from_policy( dp.pi , 0 )
J_grid_nd = grid_sys.get_grid_from_array( u ) 
J_grid_2d = grid_sys.get_2D_slice_of_grid( J_grid_nd , 0 , 1 )

mesh = ax.pcolormesh( x_level, 
                      y_level,
                      J_grid_2d.T, 
                      shading='gouraud', 
                      cmap = 'bwr', 
                      vmin = -zrange, 
                      vmax = zrange,
                      rasterized = True ) 

ax.tick_params( labelsize = fontsize )
ax.grid(True)
ax.set_ylim( -yrange, +yrange)
ax.set_xlim( -xrange, xrange)

cbar = fig.colorbar( mesh )

cbar.set_label(zname, fontsize = fontsize , rotation = 90 )

fig.tight_layout()
#fig.show()
fig.savefig( case_name + '_policy.pdf')
fig.savefig( case_name + '_policy.png')
fig.savefig( case_name + '_policy.jpg')

if show:
    plt.show()
else:
    plt.close( fig )


##################################
# Trajectory plot
##################################

ctl = dp.get_lookup_table_controller()

# Simulation
cl_sys = ctl + sys
cl_sys.x0   = np.array([-3.14, 0.])
cl_sys.compute_trajectory( 10 , 6001, 'euler')

tp = graphical.TrajectoryPlotter( sys )
tp.fontsize = fontsize
tp.plot( cl_sys.traj , 'xu' , show = False )
tp.plots[1].set_ylim([-5.5, 5.5])
tp.plots[2].set_ylim([-zrange, zrange])
tp.fig.savefig( case_name + '_traj.pdf')
tp.fig.savefig( case_name + '_traj.png')
tp.fig.savefig( case_name + '_traj.jpg')

if show:
    plt.show()
else:
    plt.close( tp.fig )


##################################
# Dimensionless policy plot
##################################

fig = plt.figure( figsize= figsize, dpi=dpi, frameon=True)
fig.canvas.manager.set_window_title( 'dimentionless policy' )
ax  = fig.add_subplot(1, 1, 1)

xname = r'$\theta^*$'#self.sys.state_label[x] #+ ' ' + self.sys.state_units[x]
yname = r'$\dot{\theta}^* = \frac{\dot{\theta}}{\omega}$'#self.sys.state_label[y] #+ ' ' + self.sys.state_units[y]
zname = r'$\tau^*=\frac{\tau}{mgl}$'

ax.set_ylabel(yname, fontsize = fontsize )
ax.set_xlabel(xname, fontsize = fontsize )

x_level = grid_sys.x_level[ 0 ] * 1
y_level = grid_sys.x_level[ 1 ] * (1 / omega)


u = grid_sys.get_input_from_policy( dp.pi , 0 )

u2 =  u * (1/mgl)

J_grid_nd = grid_sys.get_grid_from_array( u2 ) 

J_grid_2d = grid_sys.get_2D_slice_of_grid( J_grid_nd , 0 , 1 )

mesh = ax.pcolormesh( x_level, 
                      y_level, 
                      J_grid_2d.T, 
                      shading='gouraud', 
                      cmap = 'bwr',
                      rasterized = True ) 

ax.tick_params( labelsize = fontsize )
ax.grid(True)

cbar = fig.colorbar( mesh )

cbar.set_label(zname, fontsize = fontsize , rotation = 90 )

fig.tight_layout()
#fig.show()
fig.savefig( case_name + '_dimpolicy.pdf')
fig.savefig( case_name + '_dimpolicy.png')
fig.savefig( case_name + '_dimpolicy.jpg')

if show:
    plt.show()
else:
    plt.close( fig )

    

    


    
    