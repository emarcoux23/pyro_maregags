import numpy as np

from scipy import linalg
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp

from pyro.dynamic  import ContinuousDynamicSystem
from pyro.analysis import simulation


###############################################################################
class StateSpaceSystem(ContinuousDynamicSystem):
    """Time-invariant state space representation of dynamic system

    f = A x + B u
    h = C x + D u

    Parameters
    ----------
    A, B, C, D : array_like
        The matrices which define the system

    """
    ############################################
    def __init__(self, A, B, C, D):
        
        self.A = A
        self.B = B
        self.C = C
        self.D = D

        self._check_dimensions()

        n = A.shape[1]
        m = B.shape[1]
        p = C.shape[0]
        
        ContinuousDynamicSystem.__init__( self, n, m, p)
        
    ############################################
    def _check_dimensions(self):
        
        if self.A.shape[0] != self.A.shape[1]:
            raise ValueError("A must be square")

        if self.B.shape[0] != self.A.shape[0]:
            raise ValueError("Number of rows in B does not match A")

        if self.C.shape[1] != self.A.shape[0]:
            raise ValueError("Number of columns in C does not match A")

        if self.D.shape[1] != self.B.shape[1]:
            raise ValueError("Number of columns in D does not match B")

        if self.C.shape[0] != self.D.shape[0]:
            raise ValueError("Number of rows in C does not match D")
    
    #############################################
    def f(self, x, u, t):

        dx = np.dot(self.A, x) + np.dot(self.B, u)

        return dx
    
    #############################################
    def h(self, x, u, t):
        
        y = np.dot(self.C, x) + np.dot(self.D, u)
        
        return y
    
    
    ############################################
    def compute_eigen_modes(self):
        
        D,V = linalg.eig( self.A )
        
        self.poles = D
        self.modes = V
        
        return (D,V)
    
    ############################################
    def compute_eigen_mode_traj(self, i = 0 ):
        """ 
        Simulation of time evolution of the system on mode i
        ------------------------------------------------
        i : mode index
        """
        
        #Time scaling for the mode
        norm = np.sqrt(self.poles[i].real**2 + self.poles[i].imag**2)
        
        if norm > 0.001:
            tf = 2. / norm * 2 * np.pi + 1
            tf = np.clip(tf , 1 , 30)
        else:
            tf = 5
            
        n  = 2001

        sim = simulation.Simulator(self, tf, n)
        
        sim.x0 = self.modes[:,i].real + self.xbar

        traj   = sim.compute() # save the result in the instance

        return traj
    
    ############################################
    def animate_eigen_mode(self, i = 0 , is_3d = False):
        """ 
        Simulation of time evolution of the system on mode i
        ------------------------------------------------
        i : mode index
        """
        
        # Compute eigen decomposition
        self.compute_eigen_modes()
        
        # Simulate one mode
        traj = self.compute_eigen_mode_traj( i )
        
        # Animate mode
        animator       = self.get_animator()
        
        template = 'Mode %i \n%0.1f+%0.1fj'
        label    = template % (i, self.poles[i].real, self.poles[i].imag)
        
        animator.top_right_label = label
        animator.animate_simulation( traj, 3.0, is_3d)

    
    
    
    

################################################################
def _approx_jacobian(func, xbar, epsilons):
    """ Numerically approximate the jacobian of a function

    Parameters
    ----------
    func : callable
        Function for which to approximate the jacobian. Must accept an array of
        dimension ``n`` and return an array of dimension ``m``.
    xbar : array_like (dimension ``n``)
        Input around which the jacobian will be evaluated.
    epsilons : array_like (dimension ``n``)
        Step size to use for each input when approximating the jacobian

    Returns
    -------
    jac : array_like
        Jacobian matrix with dimensions m x n
    """

    n  = xbar.shape[0]
    ybar = func(xbar)
    m  = ybar.shape[0]

    J = np.zeros((m, n))
    
    for i in range(n):
        # Forward evaluation
        xf    = np.copy(xbar)
        xf[i] = xbar[i] + epsilons[i]
        yf    = func(xf)

        # Backward evaluation
        xb    = np.copy(xbar)
        xb[i] = xbar[i] - epsilons[i]
        yb    = func(xb)
        
        # Slope
        delta = yf - yb

        J[:, i] = delta / (2.0 * epsilons[i])

    return J


#################################################################
def linearize(sys, epsilon_x=0.001, epsilon_u=None):
    """Generate linear state-space model by linearizing any system.

    The system to be linearized is assumed to be time-invariant.

    Parameters
    ----------
    sys : `pyro.dynamic.ContinuousDynamicSystem`
        The system to linearize
    xbar : array_like
        State array arround which the system will be linearized
    epsilon : float
        Step size to use for numerical gradient approximation

    Returns
    -------
    instance of `StateSpaceSystem`

    """
    
    xbar = sys.xbar.astype(float)
    ubar = sys.ubar.astype(float)

    epsilon_x = np.asarray(epsilon_x)

    if epsilon_u is None:
        if epsilon_x.size > 1:
            raise ValueError("If epsilon_u is not provided, epsilon_x must be scalar")
        epsilon_u = epsilon_x

    epsilon_u = np.asarray(epsilon_u)

    if epsilon_u.size == 1:
        epsilon_u = np.ones(sys.m) * epsilon_u

    if epsilon_x.size == 1:
        epsilon_x = np.ones(sys.n) * epsilon_x
        

    def f_x(x):
        return sys.f(x, ubar, 0)

    def f_u(u):
        return sys.f(xbar, u, 0)

    def h_x(x):
        return sys.h(x, ubar, 0)

    def h_u(u):
        return sys.h(xbar, u, 0)

    A = _approx_jacobian(f_x, xbar, epsilon_x)
    B = _approx_jacobian(f_u, ubar, epsilon_u)
    C = _approx_jacobian(h_x, xbar, epsilon_x)
    D = _approx_jacobian(h_u, ubar, epsilon_u)
    
    ss = StateSpaceSystem(A, B, C, D)
    
    #############
    # Labels
    #############
    
    for i in range(sys.n):
        ss.state_label[i]  = 'Delta ' + sys.state_label[i]
    
    ss.state_units  = sys.state_units
    
    for i in range(sys.p):
        ss.output_label[i] = 'Delta ' + sys.output_label[i]
        
    ss.output_units = sys.output_units
    
    for i in range(sys.m):
        ss.input_label[i]  = 'Delta ' + sys.input_label[i]
        
    ss.input_units  = sys.input_units
    
    ss.name = 'Linearized ' + sys.name
    
    #############
    # Graphical
    #############
    
    # New fonction from delta_states to configuration space
    def new_xut2q( x, u, t):
        
        x = x + sys.xbar
        u = u + sys.ubar
        
        return sys.xut2q( x, u, t)
    
    ss.xut2q                     = new_xut2q
    
    # Using the non-linear sys graphical kinematic
    ss.linestyle                = sys.linestyle
    ss.forward_kinematic_domain = sys.forward_kinematic_domain
    ss.forward_kinematic_lines  = sys.forward_kinematic_lines

    return ss


class StateObserver(StateSpaceSystem):
    """Linear time-invariant continuous-time state observer

    f = d(x_est)/dt = A x_est + B u + L(y - y_est)
    h = C x + D u

    Where x_est is the estimate of x, the state vector and y_est is the
    estimate of y, the output vector.

    Parameters
    ----------
    sys: instance of `ContinuousDynamicSystem`
        The "real" system model that is observed.

    A, B, C, D: array-like
        Matrices that correspond to the "plant" model used by the observer to
        approximate the real system. The shapes of the matrices must be coherent
        with the number of states, inputs and outputs (n, m, p) of `sys`.

    L: n x p array-like
        Observer gain matrix. n and p refer respectively to the number of states and the
        number of outputs of `sys`.

    """

    def __init__(self, sys, A, B, C, D, L):
        self.A = np.array(A, ndmin=2, dtype=np.float64)
        self.B = np.array(B, ndmin=2, dtype=np.float64)
        self.C = np.array(C, ndmin=2, dtype=np.float64)
        self.D = np.array(D, ndmin=2, dtype=np.float64)
        self.L = np.array(L, ndmin=2, dtype=np.float64)

        self.realsys = sys # Keep a reference to real system model for simulation

        self._check_dimensions()

        n = self.A.shape[1] * 2 # Number states = Real states + estimated states
        m = self.B.shape[1]     # Same inputs as original system
        p = self.A.shape[1]     # Outputs of observer = estimated states

        ContinuousDynamicSystem.__init__( self, n, m, p)


    def _check_dimensions(self):
        super()._check_dimensions()

        # L must be n x p of sys
        if self.L.shape[0] != self.realsys.n or self.L.shape[1] != self.realsys.p:
            raise ValueError("Dimensions of gain matrix L do not match system ss.")

        # A, B, C, D must correspond to sys n,m,p
        if self.A.shape[0] != self.realsys.n:
            raise ValueError("Shape of A must correspond to number of states n of sys")

        if not (self.B.shape[1] == self.realsys.m):
            raise ValueError("Shape of B must correspond to number of inputs m of sys")

        if not (self.C.shape[0] == self.realsys.p):
            raise ValueError("Shape of C must correspond to number of outputs p of sys")


    @classmethod
    def from_ss(cls, ss, L):
        """Create a state observer based on an existing state-space system"""
        return cls(ss, ss.A, ss.B, ss.C, ss.D, L)


    @classmethod
    def kalman(cls, sys, A, B, C, D, Q, R, G=None):
        """ Create a state observer by calculating the Kalman gain matrix.

        This method calculates the Kalman gain matrix L_Kalman for the system:

        dx/dt = Ax + Bu + Gw
        y = Cx + Du + v

        Where w and v are normally distributed random vectors with 0 mean and
        covariance matrices Q and V.

        Notes on matrix G:

            - In the case where the noise process w is additive onto the system inputs,
              dx/dt = Ax + B(u + w), we have B(u + w) = Bu + Bw and therefore `G` = `B`.
              This is the default case when `G` is left unspecified or `None`.

            - If the noise process is additive onto the system states,
              dx/dt = Ax + Bu + w, then `G=I` should be passed as an argument, where `I`
              is the identity matrix with the same shape as A (n x n).

        Parameters
        ----------

        A : array-like      n x n
            Systems dynamics (state transition) matrix of the filter plant model
        B : array-like      n x m
            Input matrix of the filter plant model
        C : array-like      p x n
            State-Output matrix of the filter plant model
        D : array-like      p x m
            Input-Output matrix of the filter plant model
        Q : array-like      q x q
            Covariance matrix of the noise process w (q x 1)
        R : array-like      p x p
            Covariance matrix of the noise process v (m x 1)
        G : array-like      n x q
            Input matrix for the noise process w. By default (`G=None`), it is assumed
            that the noise process w is additive on the input u, therefore G = B and
            q = m.

        Returns
        ----------

        Instance of `StateObserver` with L, the Kalman gain matrix.

        """

        A = np.array(A, ndmin=2, dtype=np.float64)
        B = np.array(B, ndmin=2, dtype=np.float64)
        C = np.array(C, ndmin=2, dtype=np.float64)
        D = np.array(D, ndmin=2, dtype=np.float64)
        Q = np.array(Q, ndmin=2, dtype=np.float64)
        R = np.array(R, ndmin=2, dtype=np.float64)

        if G is None:
            G = B
        else:
            G = np.array(G, ndmin=2, dtype=np.float64)

        L = np.zeros([A.shape[0], C.shape[0]]) # temporary
        obs = cls(sys, A, B, C, D, L)

        # Check dimensions of Q, R and G matrices
        if not Q.shape[0] == Q.shape[1]:
            raise ValueError("Q must be square")
        if not R.shape[0] == R.shape[1]:
            raise ValueError("R must be square")
        if not G.shape[0] == A.shape[0]:
            raise ValueError("Shape[0] of G does not match shape of A")
        if not G.shape[1] == Q.shape[0]:
            raise ValueError("Shape[1] of G does not match shape of Q")
        if not R.shape[0] == C.shape[0]:
            raise ValueError("Shape of R must match number of outputs of C")

        P = linalg.solve_continuous_are(a=A.T, b=C.T, q=(G @ Q @ G.T), r=R)
        LT = np.linalg.solve(R.T, (C @ P.T))
        if LT.ndim < 2:
            LT = LT[:, np.newaxis]
        L_kalm = LT.T
        assert L_kalm.shape == obs.L.shape

        obs.L = L_kalm
        return obs


    @classmethod
    def kalman_from_ss(cls, ss, Q, R, G=None):
        """Create a state observer by calculating the Kalman gain matrix.

        See documentation for `kalman(...)`. This method uses the A, B, C, D matrices
        from the system `ss`.

        Returns
        ----------

        Instance of `StateObserver` with L, the Kalman gain matrix.

        """

        return cls.kalman(ss, ss.A, ss.B, ss.C, ss.D, Q, R, G)


    def f(self, x, u, t):
        assert u.size == self.m
        assert x.size == self.n

        n_orig = self.n / 2
        x_orig, x_est = x[n_orig:], x[:n_orig]

        dx_orig = self.sys.f(x_orig, u, t)
        y_orig = self.sys.h(x_orig, u, t)

        dx_est = self.f_est(x_est, u, y_orig)

        dx = np.concatenate([dx_orig, dx_est], axis=0)
        assert dx.shape[0] == self.n and dx.size == self.n

        return dx

    def h(self, x, u, t):
        # Output of observer system is the full vector of estimated states
        n_orig = self.n / 2
        x_est = x[:n_orig]
        return x_est

    def t2u(self, t):
        return self.realsys.t2u(t)

    def f_est(self, x_est, u, y_orig):
        y_est = np.dot(self.C, x_est) + np.dot(self.D, u)
        dx_est = np.dot(self.A, x_est) \
                 + np.dot(self.B, u) \
                 + np.dot(self.L, (y_orig - y_est))
        return dx_est

    def compute_estimates_from_outputs(self, x_est_0, y, u, t):
        """
        Compute observer estimates based on time-series data of system inputs and
        outputs.
        """
        y = np.asarray(y)
        t = np.asarray(t)

        if not y.size == t.size:
            raise ValueError("Shapes of y and t do not match")

        if u is None:
            def get_u(t):
                return self.t2u(t)
        elif np.isscalar(u):
            def get_u(t):
                return u
        elif np.asarray(u).size == np.asarray(t).size:
            get_u = interp1d(t, u)
        else:
            raise ValueError(
                "u must be either None, a scalar, or a vector with identical shape as t"
            )

        y_interp = interp1d(t, y)

        def fsim(t, x_est):
            uu = get_u(t)
            yy = y_interp(t)
            return self.f_est(x_est, uu, yy)

        sol = solve_ivp(fsim, [t[0], t[-1]], x_est_0, t_eval=t)

        return sol.y



'''
#################################################################
##################          Main                         ########
#################################################################
'''


if __name__ == "__main__":     
    """ MAIN TEST """
    from pyro.dynamic import pendulum
    
    non_linear_sys = pendulum.SinglePendulum()
    non_linear_sys.xbar = np.array([0.,0.])
    
    EPS = 0.001
    
    linearized_sys = linearize( non_linear_sys , EPS )
    
    print('\nA:\n',linearized_sys.A)
    print('\nB:\n',linearized_sys.B)
    print('\nC:\n',linearized_sys.C)
    print('\nD:\n',linearized_sys.D)
    
    # Small oscillations
    non_linear_sys.x0 = np.array([0.1,0])
    linearized_sys.x0 = np.array([0.1,0])
    
    non_linear_sys.compute_trajectory()
    linearized_sys.compute_trajectory()
    
    non_linear_sys.plot_trajectory()
    linearized_sys.plot_trajectory()
    
    # Large oscillations
    non_linear_sys.x0 = np.array([1.8,0])
    linearized_sys.x0 = np.array([1.8,0])
    
    non_linear_sys.compute_trajectory()
    linearized_sys.compute_trajectory()
    
    non_linear_sys.plot_trajectory()
    linearized_sys.plot_trajectory()
    
    
