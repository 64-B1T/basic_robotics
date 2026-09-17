import unittest

import numpy as np

from basic_robotics.general import tm
from basic_robotics.filtering import ExtendedKalmanFilter


def constant_model(x, u=None):
    return x


def identity_measurement(x):
    return x


class test_filtering_extended_kalman_filter(unittest.TestCase):

    def test_converges_on_noisy_constant_float_with_numerical_jacobian(self):
        rng = np.random.default_rng(0)
        true_value = 3.0
        ekf = ExtendedKalmanFilter(0.0, initial_covariance=1.0, process_noise=1e-5)
        for _ in range(150):
            ekf.predict(constant_model)
            z = true_value + rng.normal(scale=0.3)
            ekf.update(z, identity_measurement, R=0.1)
        self.assertAlmostEqual(ekf.state, true_value, delta=0.3)

    def test_matches_linear_kalman_filter_on_a_linear_model(self):
        # For a linear model, the EKF's numerically-estimated Jacobian should
        # reduce to the same behavior as an explicit linear Kalman Filter.
        from basic_robotics.filtering import KalmanFilter

        rng = np.random.default_rng(1)
        kf = KalmanFilter(0.0, initial_covariance=1.0, process_noise=1e-4)
        ekf = ExtendedKalmanFilter(0.0, initial_covariance=1.0, process_noise=1e-4)

        for _ in range(50):
            z = 2.0 + rng.normal(scale=0.2)
            kf.predict(F=1.0)
            ekf.predict(constant_model)
            kf.update(z, H=1.0, R=0.04)
            ekf.update(z, identity_measurement, R=0.04)

        self.assertAlmostEqual(kf.state, ekf.state, places=4)

    def test_explicit_jacobians_are_used_when_supplied(self):
        calls = {'F': 0, 'H': 0}

        def f(x, u=None):
            return 2.0 * x

        def f_jacobian(x, u=None):
            calls['F'] += 1
            return 2.0

        def h(x):
            return x

        def h_jacobian(x):
            calls['H'] += 1
            return 1.0

        ekf = ExtendedKalmanFilter(1.0, initial_covariance=1.0, process_noise=0.0)
        ekf.predict(f, F_jacobian=f_jacobian)
        self.assertAlmostEqual(ekf.state, 2.0)
        ekf.update(2.0, h, H_jacobian=h_jacobian, R=1e-6)
        self.assertEqual(calls['F'], 1)
        self.assertEqual(calls['H'], 1)

    def test_tm_state_with_tm_native_motion_model(self):
        rng = np.random.default_rng(2)

        def rotate_and_advance(x, u=None):
            return x @ tm([0.05, 0, 0, 0, 0, 0.02])

        ekf = ExtendedKalmanFilter(tm(), initial_covariance=0.05, process_noise=1e-5)
        for _ in range(30):
            ekf.predict(rotate_and_advance)
            noisy = tm(ekf.state.gTAA().flatten() + rng.normal(scale=0.01, size=6))
            ekf.update(noisy, identity_measurement, R=0.01)

        self.assertIsInstance(ekf.state, tm)


if __name__ == '__main__':
    unittest.main()
