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

# Import standard graphical parameters if part of pyro
try:
    from pyro.analysis import graphical

    default_figsize = graphical.default_figsize
    default_dpi = graphical.default_dpi
    default_fontsize = graphical.default_fontsize
except:
    default_figsize = (10, 6)
    default_dpi = 100
    default_fontsize = 12


###############################################################################
class SingleAxisPolynomialTrajectoryGenerator:
    """
    This class is a tool to generate a point-to-point trajectory for a
    single axis based on boundary conditions (position and higher order derivative)

    Polynomial of order N

    x(t) = p0 + p1*t + p2*t^2 + ... + pN*t^N

    if boundary conditions do not fully specify the parameters of the polynomial,
    then an optimization is conducted to minimize the cost function which is defined
    as a weighted sum of the integral of the square of the ith derivative of the profile.

    Parameters:
    -----------
    tf : float
        duration of the trajectory
    poly_N : int
        order of the polynomial
    diff_N : int
        order of the highest derivative to compute
    x0 : array
        initial conditions (position, velocity, acceleration, jerk, snap, crackle, pop, ...)
    xf : array
        final conditions (position, velocity, acceleration, jerk, snap, crackle, pop, ...)
    x0_N : int
        number of initial conditions to impose (higher order derivative of the initial conditions)
    xf_N : int
        number of final conditions to impose (higher order derivative of the final conditions)
    Rs : array
        weights for the cost function penalizing the ith polynomial parameters directly
    Ws : array
        weights for the cost function penalizing the ith derivative of the profile
    dt : float
        time step for the numerical solution of the trajectory

    Output:
    -------
    p : array
        polynomial parameters
    X : array
        profile of the trajectory X[i, j] is the ith derivative of the profile at time t[j]
    t : array
        time vector

    """

    ################################################
    def __init__(
        self,
        tf=10,
        poly_N=5,
        diff_N=7,
        x0=np.array([0.0, 0.0, 0.0]),
        xf=np.array([0.0, 0.0, 10.0]),
        dt=0.01,
    ):

        self.tf = tf
        self.poly_N = poly_N
        self.diff_N = diff_N
        self.x0 = x0
        self.xf = xf
        self.x0_N = x0.shape[0]
        self.xf_N = xf.shape[0]
        self.Rs = np.zeros((self.poly_N + 1))
        self.Ws = np.zeros((self.diff_N))
        self.dt = dt

        # Outputs
        self.t = None
        self.X = None
        self.p = None

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
    def compute_b(self, x0, xf, N0, Nf):
        """Compute the boundary condition vector b = [x0;xf] which represents the initial and final conditions on the trajectory and its derivatives"""

        b = np.hstack((x0[:N0], xf[:Nf]))

        print("Boundary condition vector b = [x0;xf]: \n", b)

        return b

    ################################################
    def compute_A(self, tf, N0, Nf, poly_N):
        """Compute the boundary condition matrix A which represents on the polynomial parameters are related to the boundary conditions"""

        A = np.zeros((N0 + Nf, poly_N + 1))

        # For all jth derivative of the initial conditions
        t0 = 0
        for j in range(N0):
            # For all terms of the polynomical
            for n in range(j, poly_N + 1):
                exp = n - j
                mul = 1
                for k in range(j):
                    mul = mul * (n - k)
                A[j, n] = mul * t0**exp

        # For all jth derivative of the final conditions
        for j in range(Nf):
            # For all terms of the polynomical
            for n in range(j, poly_N + 1):
                exp = n - j
                mul = 1
                for k in range(j):
                    mul = mul * (n - k)
                A[N0 + j, n] = mul * tf**exp

        print("Boundary condition matrix: \n", A)

        return A

    ################################################
    def compute_Q(self, poly_N, diff_N, tf, Ws, Rs):
        """Compute the cost function matrix Q, only used if the boundary conditions do not fully specify the parameters of the polynomial"""

        # Quadratic cost matrix
        Q = np.zeros((poly_N + 1, poly_N + 1))

        # Quadratic cost matrix for each derivative
        Qs = np.zeros((poly_N + 1, poly_N + 1, diff_N))

        # Qs are weight corresponding to computing the integral of the square of the ith derivative of the profile
        # J = p.T @ Qs[i] @ p = integral( [ d_dt(ith)x(t) ]^2 dt)
        # see https://groups.csail.mit.edu/rrg/papers/BryIJRR15.pdf
        for r in range(diff_N):
            for i in range(poly_N + 1):
                for l in range(poly_N + 1):
                    if (i >= r) and (l >= r):
                        mul = 1
                        for m in range(r):
                            mul = mul * (i - m) * (l - m)
                        exp = i + l - 2 * r + 1
                        Qs[i, l, r] = 2 * mul * tf**exp / (i + l - 2 * r + 1)
                    else:
                        Qs[i, l, r] = 0

        # Total cost for all derivatives
        for r in range(diff_N):
            Q = Q + Ws[r] * Qs[:, :, r]

        # Regulation term penalizing the polynomial parameters directly
        Q = Q + np.diag(Rs[: (poly_N + 1)])

        return Q

    ################################################
    def solve_for_polynomial_parameters(self, A, b, Q):
        """Solve for the polynomial parameters pç

        Parameters:
        -----------
        A : array
            boundary condition matrix
        b : array
            boundary condition vector
        Q : array
            cost function matrix

        Output:
        -------
        p : array
            polynomial parameters

        """

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

            p0 = np.zeros(A.shape[1])

            constraints = {"type": "eq", "fun": lambda p: A @ p - b}
            cost = lambda p: p.T @ Q @ p
            grad = lambda p: 2 * p.T @ Q
            hess = lambda p: 2 * Q

            # TODO: Change to a solver specifc to quadratic optimization
            res = minimize(
                cost,
                p0,
                method="SLSQP",
                jac=grad,
                hess=hess,
                constraints=constraints,
                options={"disp": True, "maxiter": 5000},
            )

            p = res.x

        print("Computed polynomial parameters: \n", p)

        return p

    ################################################
    def generate_trajectory(self, tf, p, diff_N, dt=0.01):
        """Generate a numerical trajectory based on the polynomial parameters"""

        Np1 = p.shape[0]  # order of polynomial
        steps = int(tf / dt)  # number of time steps
        ts = np.linspace(0, tf, steps)
        X = np.zeros((diff_N, steps))

        # For all jth derivative of the signal
        for j in range(diff_N):
            # For all time steps
            for i in range(steps):
                t = ts[i]
                x = 0
                # For all terms of the polynomical
                # TODO could replace this with A(t) generic code
                for n in range(j, Np1):
                    p_n = p[n]
                    exp = n - j
                    mul = 1
                    for k in range(j):
                        mul = mul * (n - k)
                    x = x + mul * p_n * t**exp

                X[j, i] = x

        return X, ts

    ################################################
    def get_trajectory(self, j, t, p):
        """Get the jth derivative of the trajectory at time t based on the polynomial parameters p"""

        Np1 = p.shape[0]  # order of polynomial
        x = 0

        # For all terms of the polynomical
        for n in range(j, Np1):
            p_n = p[n]
            exp = n - j
            mul = 1
            for k in range(j):
                mul = mul * (n - k)
            x = x + mul * p_n * t**exp

        return x

    ################################################
    def plot_trajectory(self, X, t, n_fig=None):

        # Number of derivatives to plot
        n_max = X.shape[0]
        if n_fig is None:
            n = n_max
        elif n_fig < n_max:
            n = n_fig
        else:
            n = n_max

        fig, ax = plt.subplots(
            n,
            figsize=default_figsize,
            dpi=default_dpi,
            frameon=True,
        )

        if n == 1:
            ax = [ax]

        for i in range(n):

            ax[i].plot(t, X[i, :], "b")
            ax[i].set_ylabel(self.labels[i], fontsize=default_fontsize)
            ax[i].tick_params(labelsize=default_fontsize)
            ax[i].grid(True)

        ax[-1].set_xlabel("Time[sec]", fontsize=default_fontsize)
        # fig.tight_layout()
        fig.canvas.draw()

        plt.show()

    ################################################
    def solve(self, show=True):

        tf = self.tf
        x0 = self.x0
        xf = self.xf
        N0 = self.x0_N
        Nf = self.xf_N
        Np = self.poly_N
        Nd = self.diff_N
        Ws = self.Ws
        Rs = self.Rs
        dt = self.dt

        b = self.compute_b(x0, xf, N0, Nf)
        A = self.compute_A(tf, N0, Nf, Np)
        Q = self.compute_Q(Np, Nd, tf, Ws, Rs)

        p = self.solve_for_polynomial_parameters(A, b, Q)

        X, t = self.generate_trajectory(tf, p, Nd, dt)

        if show:
            self.plot_trajectory(X, t)

        return p, X, t


"""
#################################################################
##################          Main                         ########
#################################################################
"""


if __name__ == "__main__":
    """MAIN TEST"""

    x0 = np.array([0, 0, 0, 0, 0, 0, 0, 0])
    xf = np.array([10, 0, 0, 0, 0, 0, 0, 0])

    ge = SingleAxisPolynomialTrajectoryGenerator(
        x0=x0, xf=xf, tf=10, poly_N=5, diff_N=7, dt=0.01
    )

    #############################
    ### Fully constrained order 3
    #############################

    # ge.x0_N = 2
    # ge.xf_N = 2
    # ge.poly_N = 3
    # ge.diff_N = 3

    # ge.solve()

    #############################
    ### Fully constrained order 5
    #############################

    ge.x0_N = 3
    ge.xf_N = 3
    ge.poly_N = 5
    ge.diff_N = 7

    ge.solve()  # order 5 fully constrained

    ###########################################
    ### Optimization on polynomial parameters
    ###########################################

    ge.poly_N = 12
    ge.Rs = 0.0 * np.ones(ge.poly_N + 1)
    ge.Ws = np.array([0, 0.0, 10.0, 1.0, 1.0, 1.0, 1.0])

    p, X, t = ge.solve()  # order 12 with optimization on polynomial parameters

    #############################
    ### Fully constrained order 7
    #############################

    # ge = SingleAxisPolynomialTrajectoryGenerator(
    #     x0=x0, xf=xf, tf=10, poly_N=7, diff_N=7, dt=0.01
    # )

    # ge.x0_N = 4
    # ge.xf_N = 4

    # ge.solve()

    #############################
    ### Fully constrained order 9
    #############################

    # ge = SingleAxisPolynomialTrajectoryGenerator(
    #     x0=x0, xf=xf, tf=10, poly_N=9, diff_N=7, dt=0.01
    # )

    # ge.x0_N = 5
    # ge.xf_N = 5

    # ge.solve()

    #############################
    ### Overconstrained order 3
    #############################

    # ge = SingleAxisPolynomialTrajectoryGenerator(
    #     x0=x0, xf=xf, tf=10, poly_N=3, diff_N=7, dt=0.01
    # )

    # ge.x0_N = 3
    # ge.xf_N = 3

    # ge.solve()
