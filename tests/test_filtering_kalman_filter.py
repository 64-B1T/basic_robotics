import unittest

import numpy as np

from basic_robotics.general import tm
from basic_robotics.filtering import KalmanFilter


class test_filtering_kalman_filter(unittest.TestCase):

    def test_converges_on_noisy_constant_float(self):
        rng = np.random.default_rng(0)
        true_value = 5.0
        kf = KalmanFilter(0.0, initial_covariance=1.0, process_noise=1e-4)
        for _ in range(200):
            z = true_value + rng.normal(scale=0.5)
            kf.predict(F=1.0)
            kf.update(z, H=1.0, R=0.25)
        self.assertAlmostEqual(kf.state, true_value, delta=0.3)

    def test_state_matches_initial_type_float(self):
        kf = KalmanFilter(1.0)
        self.assertIsInstance(kf.state, float)

    def test_tm_state_tracks_noisy_pose(self):
        rng = np.random.default_rng(1)
        true_pose = tm([1.0, 2.0, 0.5, 0, 0, 0])
        kf = KalmanFilter(tm(), initial_covariance=1.0, process_noise=1e-4)
        for _ in range(200):
            noisy = tm(true_pose.gTAA().flatten() + rng.normal(scale=0.1, size=6))
            kf.predict(F=1.0)
            kf.update(noisy, H=1.0, R=0.01)
        self.assertIsInstance(kf.state, tm)
        np.testing.assert_allclose(
                kf.state.gTAA().flatten(), true_pose.gTAA().flatten(), atol=0.3)

    def test_ndarray_constant_velocity_model(self):
        rng = np.random.default_rng(2)
        dt = 1.0
        F = np.array([[1, dt], [0, 1]])
        H = np.array([[1, 0]])
        kf = KalmanFilter(
                np.array([0.0, 1.0]), initial_covariance=np.eye(2), process_noise=np.eye(2) * 0.01)
        true_pos, true_vel = 0.0, 1.0
        for _ in range(200):
            true_pos += true_vel * dt
            z = np.array([true_pos + rng.normal(scale=0.2)])
            kf.predict(F=F)
            kf.update(z, H=H, R=0.04)
        self.assertIsInstance(kf.state, np.ndarray)
        self.assertAlmostEqual(kf.state[0], true_pos, delta=2.0)
        self.assertAlmostEqual(kf.state[1], true_vel, delta=0.3)

    def test_control_input_shifts_state(self):
        kf = KalmanFilter(0.0, initial_covariance=1.0, process_noise=0.0)
        kf.predict(F=1.0, B=1.0, u=5.0, Q=0.0)
        self.assertAlmostEqual(kf.state, 5.0)

    def test_scalar_shorthand_matches_explicit_identity(self):
        kf_scalar = KalmanFilter(np.array([1.0, 1.0]), initial_covariance=2.0, process_noise=1.0)
        kf_matrix = KalmanFilter(
                np.array([1.0, 1.0]), initial_covariance=2.0 * np.eye(2), process_noise=np.eye(2))
        kf_scalar.predict(F=1.0)
        kf_matrix.predict(F=np.eye(2))
        np.testing.assert_allclose(kf_scalar.P, kf_matrix.P)

    def test_rejects_non_square_scalar_shorthand(self):
        kf = KalmanFilter(np.array([1.0, 2.0, 3.0]))
        with self.assertRaises(ValueError):
            # measurement is length 1, state is length 3: scalar H can't be inferred
            kf.update(np.array([1.0]), H=1.0)


if __name__ == '__main__':
    unittest.main()
