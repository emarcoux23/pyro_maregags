#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 22 Feb 2024

@author: alex
"""

import numpy as np
import matplotlib.pyplot as plt

import warnings


from scipy.optimize import minimize

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

        self.bc_t0_N = 3
        self.bc_tf_N = 3

        self.labels = [
            "pos",
            "vel",
            "acc",
            "jerk",
            "snap",
            "crac",
            "pop",
            "7th",
            "8th",
            "9th",
            "10th",
        ]

    ################################################
    def compute_b(self):

        x0 = self.x0
        xf = self.xf
        N0 = self.bc_t0_N
        Nf = self.bc_tf_N

        b = np.hstack((x0[:N0], xf[:Nf]))

        print(r"[x(0), \dot{x}(x0), \ddot{x}(x0), ...] = ", x0)
        print(r"[x(tf), \dot{x}(tf), \ddot{x}(x0), ...] = ", xf)
        print("Boundary condition vector: \n", b)

        self.b = b

    ################################################
    def compute_A(self):

        tf = self.tf
        x0 = self.x0
        xf = self.xf
        N0 = self.bc_t0_N
        Nf = self.bc_tf_N
        N = self.poly_N

        A = np.zeros((N0 + Nf, N + 1))

        # For all jth derivative of the initial conditions
        t0 = 0
        for j in range(N0):
            # For all terms of the polynomical
            for n in range(j, N + 1):
                exp = n - j
                mul = 1
                for k in range(j):
                    mul = mul * (n - k)
                A[j, n] = mul * t0**exp

        # For all jth derivative of the final conditions
        for j in range(Nf):
            # For all terms of the polynomical
            for n in range(j, N + 1):
                exp = n - j
                mul = 1
                for k in range(j):
                    mul = mul * (n - k)
                A[N0 + j, n] = mul * tf**exp

        print("Boundary condition matrix: \n", A)

        self.A = A

    ################################################
    def compute_Q(self):

        Q = np.zeros((self.poly_N + 1, self.poly_N + 1))

        for i in range(self.poly_N + 1):
            Q[i, i] = 1.0 * i * i

        # TODO compute real Q
        # see https://groups.csail.mit.edu/rrg/papers/BryIJRR15.pdf

        self.Q = Q

    ################################################
    def constraints(self, p):

        res = self.A @ p - self.b

        return res

    ################################################
    def cost(self, p):

        Q = self.Q

        # TODO compute real Q
        # see https://groups.csail.mit.edu/rrg/papers/BryIJRR15.pdf

        J = p.T @ Q @ p

        # Min Jerk test
        self.p = p
        self.generate_trajectory()
        jerk = self.X[3, :]
        snap = self.X[4, :]

        # #J = np.abs(jerk).max()
        J = np.abs(snap).max()

        return J

    ################################################
    def compute_parameters(self):

        A = self.A
        b = self.b

        if A.shape[0] == A.shape[1]:

            print("Fully constrained trajectory parameters")
            p = np.linalg.solve(A, b)

        elif A.shape[0] > A.shape[1]:

            warnings.warn(
                "Warning! : impossible to respect all boundary condition, raise the order of the polynomial"
            )
            print(
                "Overconstrained boundary consitions: solving for best solution in the least-square sense"
            )
            p = np.linalg.lstsq(A, b)[0]

        else:

            print("Optimization over free decision variables")

            self.compute_Q()

            p0 = np.zeros(self.poly_N + 1)

            constraints = {"type": "eq", "fun": self.constraints}

            res = minimize(
                self.cost,
                p0,
                method="SLSQP",
                constraints=constraints,
                options={"disp": True, "maxiter": 500},
            )

            p = res.x

            # p = np.linalg.lstsq(A, b)[0]

        print("Polynomial parameters: \n", p)

        self.p = p

    ################################################
    def generate_trajectory(self, dt=0.01):

        p = self.p

        N = self.poly_N  # order of polynomial

        steps = int(self.tf / dt)
        ts = np.linspace(0, self.tf, steps)

        m = self.diff_N  # number of derivative to compute

        X = np.zeros((m, steps))

        # For all jth derivative of the signal
        for j in range(m):
            # For all time steps
            for i in range(steps):
                t = ts[i]
                x = 0
                # For all terms of the polynomical
                # TODO could replace this with A(t) generic code
                for n in range(j, N + 1):
                    p_n = p[n]
                    exp = n - j
                    mul = 1
                    for k in range(j):
                        mul = mul * (n - k)
                    x = x + mul * p_n * t**exp

                X[j, i] = x

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
        # fig.tight_layout()
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

    ge.x0 = np.array([0,  0, 0, 0, 0, 0, 0, 0])
    ge.xf = np.array([10, 0, 0, 0, 0, 0, 0, 0])

    # ge.bc_t0_N = 2
    # ge.bc_tf_N = 2
    # ge.poly_N = 3
    # ge.diff_N = 3

    # ge.solve()

    ge.bc_t0_N = 3
    ge.bc_tf_N = 3
    ge.poly_N = 5
    ge.diff_N = 7

    ge.solve()

    # ge.bc_t0_N = 4
    # ge.bc_tf_N = 4
    # ge.poly_N = 7
    # ge.diff_N = 7

    # ge.solve()

    # ge.bc_t0_N = 5
    # ge.bc_tf_N = 5
    # ge.poly_N = 9
    # ge.diff_N = 7

    # ge.solve()

    # ge.bc_t0_N = 6
    # ge.bc_tf_N = 6
    # ge.poly_N = 11
    # ge.diff_N = 7

    # ge.solve()

    ge.bc_t0_N = 7
    ge.bc_tf_N = 7
    ge.poly_N = 13
    ge.diff_N = 7

    ge.solve()

    ge.bc_t0_N = 1
    ge.bc_tf_N = 1
    ge.poly_N = 3
    ge.diff_N = 7

    ge.solve()
