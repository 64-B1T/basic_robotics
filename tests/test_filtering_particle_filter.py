import unittest

import numpy as np

from basic_robotics.general import tm
from basic_robotics.filtering import ParticleFilter


def constant_model(x, u=None):
    return x


def identity_measurement(x):
    return x


class test_filtering_particle_filter(unittest.TestCase):

    def test_converges_on_noisy_constant_float(self):
        rng = np.random.default_rng(0)
        true_value = 4.0
        pf = ParticleFilter(0.0, initial_covariance=1.0, num_particles=300, seed=0)
        for _ in range(150):
            pf.predict(constant_model, process_noise=1e-3)
            z = true_value + rng.normal(scale=0.4)
            pf.update(z, identity_measurement, measurement_noise=0.16)
        self.assertAlmostEqual(pf.state, true_value, delta=0.5)

    def test_state_matches_initial_type(self):
        pf = ParticleFilter(1.0, num_particles=50, seed=0)
        self.assertIsInstance(pf.state, float)
        pf_arr = ParticleFilter(np.array([1.0, 2.0]), num_particles=50, seed=0)
        self.assertIsInstance(pf_arr.state, np.ndarray)

    def test_weights_stay_normalized_after_update(self):
        pf = ParticleFilter(0.0, num_particles=100, seed=0)
        pf.update(0.1, identity_measurement, measurement_noise=1.0)
        self.assertAlmostEqual(np.sum(pf.weights), 1.0, places=8)

    def test_resample_replaces_degenerate_weights_with_uniform(self):
        pf = ParticleFilter(0.0, num_particles=50, seed=0)
        pf.weights = np.zeros(50)
        pf.weights[0] = 1.0
        pf._resample_if_needed(threshold=1.0)
        np.testing.assert_allclose(pf.weights, np.full(50, 1.0 / 50))

    def test_recovers_from_degenerate_zero_likelihood(self):
        pf = ParticleFilter(0.0, initial_covariance=1e-6, num_particles=20, seed=0)
        # A measurement enormously far from every particle drives every likelihood
        # to ~0; the filter should reset to uniform weights instead of raising.
        pf.update(1e6, identity_measurement, measurement_noise=1e-6)
        self.assertAlmostEqual(np.sum(pf.weights), 1.0, places=8)
        self.assertFalse(np.any(np.isnan(pf.weights)))

    def test_tm_state_with_tm_native_motion_model(self):
        rng = np.random.default_rng(1)

        def rotate_and_advance(x, u=None):
            return x @ tm([0.05, 0, 0, 0, 0, 0.02])

        pf = ParticleFilter(tm(), initial_covariance=0.05, num_particles=300, seed=1)
        for _ in range(20):
            pf.predict(rotate_and_advance, process_noise=1e-4)
            noisy = tm(pf.state.gTAA().flatten() + rng.normal(scale=0.02, size=6))
            pf.update(noisy, identity_measurement, measurement_noise=0.02)

        self.assertIsInstance(pf.state, tm)


if __name__ == '__main__':
    unittest.main()
