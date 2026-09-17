"""Extended Kalman Filter, generic over tm objects, numpy arrays, and scalars."""
import numpy as np

from .state_utils import as_matrix, from_vector, numerical_jacobian, to_vector


class ExtendedKalmanFilter:
    """
    An Extended Kalman Filter (EKF) for nonlinear process/measurement models.

    Like `KalmanFilter`, state is tracked internally as a flat vector and exposed in
    whichever representation (tm, numpy array, or float) it was initialized with.
    Unlike `KalmanFilter`, the process and measurement models are arbitrary functions
    operating on states in their native representation - so a tm-based motion or
    observation model can freely use tm's overloaded operators - rather than fixed
    matrices. Jacobians are estimated numerically by default, or may be supplied
    analytically for speed/accuracy.
    """

    def __init__(self, initial_state, initial_covariance=1.0, process_noise=1.0,
            jacobian_step=1e-6):
        """
        Create a new Extended Kalman Filter.

        Args:
            initial_state: tm, np.ndarray, list/tuple, or float initial state estimate.
            initial_covariance: initial state covariance (n, n), or scalar shorthand.
            process_noise: default process noise covariance Q (n, n), or scalar
                shorthand, used by `predict` when no explicit `Q` is supplied.
            jacobian_step (float): default perturbation size used for numerically
                estimated Jacobians.
        """
        self._template = initial_state
        self.x = to_vector(initial_state)
        self.n = len(self.x)
        self.P = as_matrix(initial_covariance, self.n)
        self.default_process_noise = as_matrix(process_noise, self.n)
        self.jacobian_step = jacobian_step

    @property
    def state(self):
        """Return the current state estimate, in the representation it was created with."""
        return from_vector(self.x, self._template)

    def predict(self, f, u=None, F_jacobian=None, Q=None):
        """
        Propagate the state and covariance forward through a nonlinear process model.

        Args:
            f: process model. Called as `f(state)`, or `f(state, u)` if `u` is given,
                where `state` is in the filter's native representation; must return a
                new state in that same representation.
            u: optional control input passed through to `f` and `F_jacobian`.
            F_jacobian: optional function `F_jacobian(state)` (or `(state, u)` if `u`
                is given) returning the (n, n) Jacobian of `f` with respect to state.
                If omitted, the Jacobian is estimated numerically from `f`.
            Q: process noise covariance (n, n), or scalar shorthand. Defaults to the
                `process_noise` supplied at construction.

        Returns:
            The predicted state, in the filter's native representation.
        """
        def call_f(state):
            return f(state, u) if u is not None else f(state)

        def f_vec(vector):
            return to_vector(call_f(from_vector(vector, self._template)))

        if F_jacobian is None:
            F = numerical_jacobian(f_vec, self.x, self.jacobian_step)
        else:
            jacobian = F_jacobian(self.state, u) if u is not None else F_jacobian(self.state)
            F = as_matrix(jacobian, self.n)

        Q = self.default_process_noise if Q is None else as_matrix(Q, self.n)

        self.x = f_vec(self.x)
        self.P = F @ self.P @ F.T + Q
        return self.state

    def update(self, measurement, h, H_jacobian=None, R=1.0):
        """
        Incorporate a new measurement through a nonlinear measurement model.

        Args:
            measurement: tm, np.ndarray, list/tuple, or float measurement.
            h: measurement model. Called as `h(state)`, where `state` is in the
                filter's native representation; must return a predicted measurement in
                the same representation as `measurement`.
            H_jacobian: optional function `H_jacobian(state)` returning the (m, n)
                Jacobian of `h` with respect to state. If omitted, the Jacobian is
                estimated numerically from `h`.
            R: measurement noise covariance (m, m), or scalar shorthand.

        Returns:
            The updated state, in the filter's native representation.
        """
        z = to_vector(measurement)
        m = len(z)

        def h_vec(vector):
            return to_vector(h(from_vector(vector, self._template)))

        predicted_z = h_vec(self.x)
        if H_jacobian is None:
            H = numerical_jacobian(h_vec, self.x, self.jacobian_step)
        else:
            H = as_matrix(H_jacobian(self.state), m, self.n)

        R = as_matrix(R, m)

        innovation = z - predicted_z
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ innovation
        self.P = (np.eye(self.n) - K @ H) @ self.P
        return self.state
