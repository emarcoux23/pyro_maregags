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

    Multiple methods are implemented:
    - Trapezoidal (acceleration profil is discontinuous)
    - Polynomial of order N

    if boundary conditions do not fully specify the profile parameter,
    then an optimization is conducted (TODO)

    """

    ################################################
    def __init__(self, tf=10, method="poly", N=5, x0=None, xf=None):

        self.tf = tf

        self.method = method
        self.poly_N = N

        if method == "trapz":
            self.max_order = 3

        self.boundary_condition_N = 3

        self.labels = [
            "position",
            "velocity",
            "acceleration",
            "jerk",
            "snap",
            "d5",
            "d6",
            "d7",
        ]

    ################################################
    def set_initial_conditions(self, x0):

        N = self.boundary_condition_N

        self.x0 = np.zeros(N)
        self.x0[:N] = x0[:N]  # First N values only

    ################################################
    def set_final_conditions(self, xf):

        N = self.boundary_condition_N

        self.xf = np.zeros(N)
        self.xf[:N] = xf[:N]  # First N values only

    ################################################
    def generate_polynomial_parameters(self):

        N = self.poly_N
        tf = self.tf
        x0 = self.x0
        xf = self.xf

        # TODO: N-order version
        # Place holder for 5-order

        b = np.hstack((x0, xf))
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

        p = np.linalg.solve(A, b)

        return p

    ################################################
    def generate_trajectory(self, p, dt=0.01):

        n = int(self.tf / dt)
        t = np.linspace(0, self.tf, n)

        # TODO: N-order version
        x = p[0] + p[1] * t + p[2] * t**2 + p[3] * t**3 + p[4] * t**4 + p[5] * t**5
        dx = p[1] + 2 * p[2] * t + 3 * p[3] * t**2 + 4 * p[4] * t**3 + 5 * p[5] * t**4
        ddx = 2 * p[2] + 6 * p[3] * t + 12 * p[4] * t**2 + 20 * p[5] * t**3
        dddx = 6 * p[3] + 24 * p[4] * t + 60 * p[5] * t**2
        ddddx = 24 * p[4] + 120 * p[5] * t

        X = np.vstack((x, dx, ddx, dddx, ddddx))

        return (X, t)

    ################################################
    def plot_trajectory(self, X, t, n_fig = None):

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

        if n==1: ax = [ax]

        for i in range(n):

            ax[i].plot(t, X[i, :], "b")
            ax[i].set_ylabel(self.labels[i], fontsize=graphical.default_fontsize)
            ax[i].set_xlabel("Time[sec]", fontsize=graphical.default_fontsize)
            ax[i].tick_params(labelsize=graphical.default_fontsize)
            ax[i].grid(True)

        fig.tight_layout()
        fig.canvas.draw()

        plt.show()

    ################################################
    def solve(self):

        p = self.generate_polynomial_parameters()
        X, t = self.generate_trajectory(p)

        self.plot_trajectory(X, t)

        return (p, X, t)


"""
#################################################################
##################          Main                         ########
#################################################################
"""


if __name__ == "__main__":
    """MAIN TEST"""

    ge = SingleAxisTrajectoryGenerator()

    ge.set_initial_conditions(np.array([-1, -1, -1]))
    ge.set_final_conditions(np.array([1, 1, 1]))

    p = ge.generate_polynomial_parameters()
    X, t = ge.generate_trajectory(p)

    ge.plot_trajectory(X, t)
