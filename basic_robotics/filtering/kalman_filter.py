"""Linear Kalman Filter, generic over tm objects, numpy arrays, and scalars."""
import numpy as np

from .state_utils import as_matrix, from_vector, to_vector


class KalmanFilter:
    """
    A standard linear Kalman Filter.

    The filter's state may be represented natively as a basic_robotics tm object, a
    numpy array, or a raw float/double; internally the state is tracked as a flat
    vector so the standard predict/update linear algebra applies uniformly, and it is
    converted back to the native representation whenever `state` is read.

    Process/measurement models (F, B, H) and noise covariances (Q, R) are always
    expressed as plain numpy matrices, or as a scalar shorthand for an isotropic
    `scalar * identity(n)` matrix.
    """

    def __init__(self, initial_state, initial_covariance=1.0, process_noise=1.0):
        """
        Create a new Kalman Filter.

        Args:
            initial_state: tm, np.ndarray, list/tuple, or float initial state estimate.
            initial_covariance: initial state covariance. An (n, n) matrix, or a
                scalar shorthand for `initial_covariance * identity(n)`.
            process_noise: default process noise covariance Q used by `predict` when
                no explicit `Q` is supplied. An (n, n) matrix, or a scalar shorthand.
        """
        self._template = initial_state
        self.x = to_vector(initial_state)
        self.n = len(self.x)
        self.P = as_matrix(initial_covariance, self.n)
        self.default_process_noise = as_matrix(process_noise, self.n)

    @property
    def state(self):
        """Return the current state estimate, in the representation it was created with."""
        return from_vector(self.x, self._template)

    def predict(self, F=1.0, Q=None, B=None, u=None):
        """
        Propagate the state and covariance forward one step: x = Fx (+ Bu), P = FPF' + Q.

        Args:
            F: state transition matrix (n, n), or scalar shorthand for `F * identity(n)`.
            Q: process noise covariance (n, n), or scalar shorthand. Defaults to the
                `process_noise` supplied at construction.
            B: optional control input matrix (n, k), or scalar shorthand (requires k == n).
            u: optional control input (tm, np.ndarray, list/tuple, or float) of length k.

        Returns:
            The predicted state, in the filter's native representation.
        """
        F = as_matrix(F, self.n)
        Q = self.default_process_noise if Q is None else as_matrix(Q, self.n)

        self.x = F @ self.x
        if u is not None:
            u_vec = to_vector(u)
            B = as_matrix(1.0 if B is None else B, self.n, len(u_vec))
            self.x = self.x + B @ u_vec
        self.P = F @ self.P @ F.T + Q
        return self.state

    def update(self, measurement, H=1.0, R=1.0):
        """
        Incorporate a new measurement into the state estimate.

        Args:
            measurement: tm, np.ndarray, list/tuple, or float measurement.
            H: measurement matrix (m, n) mapping state to measurement space, or scalar
                shorthand for `H * identity(n)` (requires the measurement to be the
                same dimension as the state).
            R: measurement noise covariance (m, m), or scalar shorthand.

        Returns:
            The updated state, in the filter's native representation.
        """
        z = to_vector(measurement)
        m = len(z)
        H = as_matrix(H, m, self.n)
        R = as_matrix(R, m)

        innovation = z - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ innovation
        self.P = (np.eye(self.n) - K @ H) @ self.P
        return self.state
