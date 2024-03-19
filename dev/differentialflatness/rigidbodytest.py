# -*- coding: utf-8 -*-

###############################################################################
import numpy as np
import matplotlib.pyplot as plt


from pyro.analysis import graphical
from pyro.planning import trajectorygeneration
from pyro.dynamic import rigidbody
from pyro.planning import plan

###############################################################################

# fixed initial position for now
# initial angular velocity is related to jerk of trajectory
x0 = -10.0
y0 = -2.0
z0 = 0.0

# fixed final position for now
xf = 0.0
yf = 0.0
zf = 0.0 #np.pi / 2
tf = 10

ddx0 = 0.0
ddy0 = -1.0#ddx0 * np.tan(z0)

ddxf = 1.0
ddyf =+0.0 # ddxf * np.tan(zf)

bc_y = np.array([y0, 0, ddy0, yf, 0, ddyf])

gex = trajectorygeneration.SingleAxisTrajectoryGenerator(N=9)
gex.bc_t0_N = 5
gex.bc_tf_N = 5
gex.x0 = np.array([x0, 0, ddx0,0,0,0])
gex.xf = np.array([xf, 0, ddxf,0,0,0])
gex.solve()
x = gex.X[0, :]
dx = gex.X[1, :]
ax = gex.X[2, :]
dax = gex.X[3, :]
ddax = gex.X[4, :]
t = gex.t

gey = trajectorygeneration.SingleAxisTrajectoryGenerator(N=9)
gey.bc_t0_N = 5
gey.bc_tf_N = 5
gey.x0 = np.array([y0, 0, ddy0,0,0,0])
gey.xf = np.array([yf, 0, ddyf,0,0,0])
gey.solve()
y = gey.X[0, :]
dy = gey.X[1, :]
ay = gey.X[2, :]
day = gey.X[3, :]
dday = gex.X[4, :]

# Position theta
theta = np.arctan2(ay, ax)
# theta = np.arctan( (ay/ax))
s = np.sin(theta)
c = np.cos(theta)
dtheta = (day * c - dax * s) / (ax * s + ay * c)
ddtheta = (
    s * (-ddax + ax * dtheta**2 - 2 * day * dtheta)
    + c * (dday - ay * dtheta**2 - 2 * dax * dtheta)
) / (
    ax * c + ay * s
)  # TODO check analytical equation, seems wrong

dtheta_num = np.diff(theta, n=1, prepend=0.0)
ddtheta_num = np.diff(dtheta, n=1, prepend=0.0)

# dtheta = dtheta_num
# ddtheta = ddtheta_num

# Create traj
steps = len(t)
xs = np.zeros((steps, 6))
ys = np.zeros((steps, 6))
us = np.zeros((steps, 2))
dxs = np.zeros((steps, 6))

sys = rigidbody.RigidBody2D()

sys.mass = 0.8
sys.inertia = 1.0
sys.l_t = 1.0

m = sys.mass
J = sys.inertia
r = sys.l_t

x_cg = x - J / (m * r) * np.cos(theta)
y_cg = y - J / (m * r) * np.sin(theta)

xs[:, 0] = x_cg
xs[:, 1] = y_cg
xs[:, 2] = theta

M = np.array([[m, 0], [0, m]])

ax_cg = ax + J / (m * r) * np.sin(theta) * ddtheta + J / (m * r) * np.cos(theta) * dtheta**2
ay_cg = ay - J / (m * r) * np.cos(theta) * ddtheta + J / (m * r) * np.sin(theta) * dtheta**2

# COmpute forces
for i in range(steps):
    R = np.array([[np.cos(theta[i]), -np.sin(theta[i])], [np.sin(theta[i]), np.cos(theta[i])]])
    a_cg = np.array([ax_cg[i], ay_cg[i]])
    us[i, :] = np.linalg.inv(R) @ M @ a_cg


traj = plan.Trajectory(xs, us, t, dxs, ys)

fig, axes = plt.subplots(
    2, figsize=graphical.default_figsize, dpi=graphical.default_dpi, frameon=True
)

axes[0].plot(t, ax_cg, "b")
axes[0].set_ylabel("ax_cg", fontsize=graphical.default_fontsize)
axes[0].set_xlabel("t", fontsize=graphical.default_fontsize)
axes[0].tick_params(labelsize=graphical.default_fontsize)
axes[0].grid(True)

axes[1].plot(t, ay_cg, "b")
axes[1].set_ylabel("ay_cg", fontsize=graphical.default_fontsize)
axes[1].set_xlabel("t", fontsize=graphical.default_fontsize)
axes[1].tick_params(labelsize=graphical.default_fontsize)
axes[1].grid(True)


fig, axes = plt.subplots(
    3, figsize=graphical.default_figsize, dpi=graphical.default_dpi, frameon=True
)

axes[0].plot(t, theta, "b")
axes[0].set_ylabel("Theta", fontsize=graphical.default_fontsize)
axes[0].set_xlabel("v", fontsize=graphical.default_fontsize)
axes[0].tick_params(labelsize=graphical.default_fontsize)
axes[0].grid(True)

axes[1].plot(t, dtheta, "b")
# axes[1].plot(t, dtheta_num, "r")
axes[1].set_ylabel("w", fontsize=graphical.default_fontsize)
axes[1].set_xlabel("t", fontsize=graphical.default_fontsize)
axes[1].tick_params(labelsize=graphical.default_fontsize)
axes[1].grid(True)

axes[2].plot(t, ddtheta, "b")
# axes[2].plot(t, ddtheta_num, "r")
axes[2].set_ylabel("dw", fontsize=graphical.default_fontsize)
axes[2].set_xlabel("t", fontsize=graphical.default_fontsize)
axes[2].tick_params(labelsize=graphical.default_fontsize)
axes[2].grid(True)

fig.tight_layout()
fig.canvas.draw()

plt.show()


fig, axes = plt.subplots(
    1, figsize=graphical.default_figsize, dpi=graphical.default_dpi, frameon=True
)

axes.plot(x, y, "r")
axes.plot(x_cg, y_cg, "b")
axes.set_ylabel("y", fontsize=graphical.default_fontsize)
axes.set_xlabel("x", fontsize=graphical.default_fontsize)
axes.axis("equal")
axes.set(xlim=(-10, 10), ylim=(-10, 10))
axes.tick_params(labelsize=graphical.default_fontsize)
axes.grid(True)

fig.tight_layout()
fig.canvas.draw()

plt.show()


sys.traj = traj

sys.animate_simulation()
