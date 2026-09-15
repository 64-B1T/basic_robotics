import unittest

import numpy as np

from basic_robotics.general import tm
from basic_robotics.general.faser_twist import Twist
from basic_robotics.general.faser_screw import Screw


class test_general_twist(unittest.TestCase):

    def test_general_twist_construction_defaults_frame(self):
        twist = Twist(np.array([0.1, 0.2, 0.3, 1.0, 0.0, 0.0]).reshape((6, 1)))
        np.testing.assert_allclose(
                twist.data.flatten(), [0.1, 0.2, 0.3, 1.0, 0.0, 0.0])
        np.testing.assert_allclose(twist.frame_applied.gTAA().flatten(), np.zeros(6))

    def test_general_twist_construction_with_frame(self):
        frame = tm([1, 2, 3, 0, 0, 0])
        twist = Twist(np.array([0, 0, 0, 1, 0, 0]).reshape((6, 1)), frame)
        np.testing.assert_allclose(twist.frame_applied.gTAA().flatten(), frame.gTAA().flatten())

    def test_general_twist_fromTM_and_toTM_round_trip(self):
        original = tm([1, 2, 0, 0, 0, np.pi / 4])
        twist = Twist.fromTM(original)

        self.assertIsInstance(twist, Twist)

        reconstructed = twist.toTM()
        np.testing.assert_allclose(
                reconstructed.gTM(), original.gTM(), atol=1e-8)

    def test_general_twist_fromTM_pure_translation(self):
        original = tm([1, 0, 0, 0, 0, 0])
        twist = Twist.fromTM(original)
        # A pure-translation transform's twist has zero angular component.
        np.testing.assert_allclose(twist.data.flatten()[0:3], [0, 0, 0], atol=1e-8)
        np.testing.assert_allclose(twist.data.flatten()[3:6], [1, 0, 0], atol=1e-8)

    def test_general_twist_twistMatrix_shape_and_structure(self):
        twist = Twist(np.array([0.1, 0.2, 0.3, 1.0, 2.0, 3.0]).reshape((6, 1)))
        mat = twist.twistMatrix()

        self.assertEqual(mat.shape, (4, 4))
        # Bottom row of an se(3) matrix is always zero.
        np.testing.assert_allclose(mat[3, :], [0, 0, 0, 0])
        # Upper-left 3x3 block is skew-symmetric (angular velocity).
        np.testing.assert_allclose(mat[0:3, 0:3], -mat[0:3, 0:3].T, atol=1e-8)
        # Last column (minus the bottom entry) is the linear velocity.
        np.testing.assert_allclose(mat[0:3, 3], [0.1, 0.2, 0.3])

    def test_general_twist_toScrew_general_case(self):
        twist = Twist(np.array([0.1, 0.2, 0.3, 1.0, 0.0, 0.0]).reshape((6, 1)))
        screw = twist.toScrew()

        self.assertIsInstance(screw, Screw)
        w = screw.data.flatten()[0:3]
        self.assertAlmostEqual(float(np.linalg.norm(w)), 1.0, places=6)

    def test_general_twist_toScrew_pure_rotation_case(self):
        # Zero linear component triggers the pure-rotation branch, which
        # returns a screw aligned with the angular axis and zero-point.
        twist = Twist(np.array([0, 0, 1, 0, 0, 0]).reshape((6, 1)))
        screw = twist.toScrew()

        np.testing.assert_allclose(screw.data.flatten()[0:3], [0, 0, 1], atol=1e-8)
        np.testing.assert_allclose(screw.data.flatten()[3:6], [0, 0, 0], atol=1e-8)


if __name__ == '__main__':
    unittest.main()
