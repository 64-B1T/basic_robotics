"""
Generic filtering and state estimation tools.

Provides Low Pass, (linear) Kalman, Extended Kalman, and Particle filter
implementations that operate uniformly over basic_robotics tm objects, numpy
arrays, and raw floats/doubles - each filter tracks its state internally as a
flat vector for the underlying linear algebra, and hands it back in whichever
representation it was created with.
"""
from .state_utils import as_matrix, from_vector, numerical_jacobian, to_vector
from .low_pass_filter import LowPassFilter
from .kalman_filter import KalmanFilter
from .extended_kalman_filter import ExtendedKalmanFilter
from .particle_filter import ParticleFilter
