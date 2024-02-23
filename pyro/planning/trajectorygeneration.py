#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 22 Feb 2024

@author: alex
"""

import numpy as np
import matplotlib.pyplot as plt

# from scipy.optimize import minimize

from pyro.analysis import graphical


###############################################################################
class SingleAxisTrajectoryGenerator:
    """
    This class is a tool to generate a point-to-point trajectory for a
    single axis based on boundary conditions (position and higher order derivative)

    Polynomial of order N

    if boundary conditions do not fully specify the profile parameter,
    then an optimization is conducted (TODO)

    """

    ################################################
    def __init__(self, tf=10, N=5, x0=None, xf=None):

        self.tf = tf
        self.poly_N = N
        self.diff_N = N
        self.x0 = x0
        self.xf = xf

        self.boundary_condition_N = 3

        self.labels = [
            'pos',
            "vel",
            "acc",
            "jerk (3th)",
            "snap (4th)",
            "crac (5th)",
            "pop (6th)",
            "7th",
            "8th",
            "9th",
            "10th",
        ]

    ################################################
    def compute_b(self):

        x0 = self.x0
        xf = self.xf
        N = self.boundary_condition_N

        b = np.hstack((x0[:N], xf[:N]))

        print(r"[x(0), \dot{x}(x0), \ddot{x}(x0), ...] = ", x0)
        print(r"[x(tf), \dot{x}(tf), \ddot{x}(x0), ...] = ", xf)
        print("Boundary condition vector: \n", b)

        self.b = b

    ################################################
    def compute_A(self):

        x0 = self.x0
        xf = self.xf
        N = self.boundary_condition_N

        # TODO specific poly 5 + N = 3
        A = np.array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 2, 0, 0, 0],
                [1, tf, tf**2, tf**3, tf**4, tf**5],
                [0, 1, 2 * tf, 3 * tf**2, 4 * tf**3, 5 * tf**4],
                [0, 0, 2, 6 * tf, 12 * tf**2, 20 * tf**3],
            ]
        )

        print("Boundary condition matrix: \n", A)
        #print("Boundary condition matrix: \n", A2)

        self.A = A

    ################################################
    def compute_parameters(self):

        A = self.A
        b = self.b

        p = np.linalg.solve(A, b)

        print("Polynomial parameters: \n", p)

        self.p = p

    ################################################
    def generate_trajectory2(self, dt=0.01):

        p = self.p
        n = int(self.tf / dt)
        t = np.linspace(0, self.tf, n)

        # TODO: N-order version
        x = p[0] + p[1] * t + p[2] * t**2 + p[3] * t**3 + p[4] * t**4 + p[5] * t**5
        dx = p[1] + 2 * p[2] * t + 3 * p[3] * t**2 + 4 * p[4] * t**3 + 5 * p[5] * t**4
        ddx = 2 * p[2] + 6 * p[3] * t + 12 * p[4] * t**2 + 20 * p[5] * t**3
        dddx = 6 * p[3] + 24 * p[4] * t + 60 * p[5] * t**2
        ddddx = 24 * p[4] + 120 * p[5] * t

        X = np.vstack((x, dx, ddx, dddx, ddddx))

        self.X = X
        self.t = t

    ################################################
    def generate_trajectory(self, dt=0.01):

        p = self.p

        N = self.poly_N # order of polynomial

        steps = int(self.tf / dt)
        ts = np.linspace(0, self.tf, steps)
        
        m = self.diff_N # number of derivative to compute

        X = np.zeros((m,steps))

        # For all jth derivative of the signal
        for j in range(m):
            # For all time steps
            for i in range(steps):
                t = ts[i]
                x = 0
                # For all terms of the polynomical
                for n in range(j,N+1):
                    p_n = p[n]
                    exp = n - j
                    mul = 1
                    for k in range(j):
                        mul = mul * ( n - k )
                    x = x + mul * p_n * t ** exp

                X[j,i] = x

        self.X = X
        self.t = ts

    ################################################
    def plot_trajectory(self, n_fig=None):

        X = self.X
        t = self.t

        n_max = X.shape[0]

        if n_fig is None:
            n = n_max
        elif n_fig < n_max:
            n = n_fig
        else:
            n = n_max

        fig, ax = plt.subplots(
            n,
            figsize=graphical.default_figsize,
            dpi=graphical.default_dpi,
            frameon=True,
        )

        if n == 1:
            ax = [ax]

        for i in range(n):

            ax[i].plot(t, X[i, :], "b")
            ax[i].set_ylabel(self.labels[i], fontsize=graphical.default_fontsize)
            ax[i].tick_params(labelsize=graphical.default_fontsize)
            ax[i].grid(True)

        ax[-1].set_xlabel("Time[sec]", fontsize=graphical.default_fontsize)
        #fig.tight_layout()
        fig.canvas.draw()

        plt.show()

    ################################################
    def solve(self):

        self.compute_b()
        self.compute_A()

        self.compute_parameters()
        self.generate_trajectory()
        self.plot_trajectory()


"""
#################################################################
##################          Main                         ########
#################################################################
"""


if __name__ == "__main__":
    """MAIN TEST"""

    ge = SingleAxisTrajectoryGenerator()

    ge.x0 = np.array([-1, -1, 0, 10])
    ge.xf = np.array([1, 0, 0])

    ge.diff_N = 20

    ge.solve()
