import unittest

import numpy as np

from basic_robotics.general import tm
from basic_robotics.filtering import LowPassFilter


class test_filtering_low_pass_filter(unittest.TestCase):

    def test_seeds_from_first_measurement_when_unset(self):
        lpf = LowPassFilter(0.5)
        self.assertIsNone(lpf.state)
        result = lpf.update(3.0)
        self.assertEqual(result, 3.0)

    def test_rejects_invalid_alpha(self):
        with self.assertRaises(ValueError):
            LowPassFilter(1.5)
        with self.assertRaises(ValueError):
            LowPassFilter(-0.1)

    def test_float_smoothing(self):
        lpf = LowPassFilter(0.5, initial_state=0.0)
        result = lpf.update(10.0)
        self.assertAlmostEqual(result, 5.0)
        result = lpf.update(10.0)
        self.assertAlmostEqual(result, 7.5)

    def test_converges_towards_constant_input(self):
        lpf = LowPassFilter(0.2, initial_state=0.0)
        for _ in range(200):
            result = lpf.update(10.0)
        self.assertAlmostEqual(result, 10.0, places=3)

    def test_numpy_array_state(self):
        lpf = LowPassFilter(0.5, initial_state=np.zeros(3))
        result = lpf.update(np.array([2.0, 4.0, 6.0]))
        self.assertIsInstance(result, np.ndarray)
        np.testing.assert_allclose(result, [1.0, 2.0, 3.0])

    def test_list_input_is_coerced_to_array(self):
        lpf = LowPassFilter(0.5, initial_state=[0.0, 0.0])
        result = lpf.update([2.0, 4.0])
        self.assertIsInstance(result, np.ndarray)
        np.testing.assert_allclose(result, [1.0, 2.0])

    def test_tm_state(self):
        lpf = LowPassFilter(0.5, initial_state=tm())
        result = lpf.update(tm([2.0, 0, 0, 0, 0, 0]))
        self.assertIsInstance(result, tm)
        np.testing.assert_allclose(result.gTAA().flatten(), [1.0, 0, 0, 0, 0, 0], atol=1e-8)

    def test_reset(self):
        lpf = LowPassFilter(0.5, initial_state=5.0)
        lpf.update(10.0)
        lpf.reset()
        self.assertIsNone(lpf.state)
        lpf.reset(2.0)
        self.assertEqual(lpf.state, 2.0)


if __name__ == '__main__':
    unittest.main()
