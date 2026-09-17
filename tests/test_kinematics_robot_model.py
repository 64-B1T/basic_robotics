import unittest

import numpy as np

from basic_robotics.general import tm, Wrench
from basic_robotics.kinematics.robot_model import Robot


class _JacobianOnlyRobot(Robot):
    """Minimal Robot subclass defining only jacobian(), to exercise the
    base class's default inverseJacobian()/jacobianBody() implementations."""

    def __init__(self):
        super().__init__("JacobianOnlyRobot")
        self._end_effector_pos_global = tm()

    def jacobian(self, *args, **kwargs):
        return np.eye(6) * 2.0


class _InverseJacobianOnlyRobot(Robot):
    """Minimal Robot subclass defining only inverseJacobian(), to exercise
    the base class's default jacobian() implementation."""

    def __init__(self):
        super().__init__("InverseJacobianOnlyRobot")
        self._end_effector_pos_global = tm()

    def inverseJacobian(self, *args, **kwargs):
        return np.eye(6) * 0.5


class test_kinematics_robot_model(unittest.TestCase):

    def test_robot_model_getActuatorForces(self):
        robot = Robot()
        robot._last_tau = np.array([1.0, 2.0, 3.0])
        forces = robot.getActuatorForces()
        np.testing.assert_allclose(forces, [1.0, 2.0, 3.0])
        # Confirm it's a copy, not the same array.
        forces[0] = 99.0
        self.assertEqual(robot._last_tau[0], 1.0)

    def test_robot_model_getGrav(self):
        robot = Robot()
        grav = robot.getGrav()
        np.testing.assert_allclose(grav, [0, 0, -9.81])
        grav[0] = 99.0
        self.assertEqual(robot.grav[0], 0)

    def test_robot_model_setGrav(self):
        robot = Robot()
        robot.setGrav(np.array([0, 0, -1.62]))
        np.testing.assert_allclose(robot.grav, [0, 0, -1.62])

    def test_robot_model_getEEPos_and_getBasePos(self):
        robot = Robot()
        robot._end_effector_pos_global = tm([1, 2, 3, 0, 0, 0])
        robot._base_pos_global = tm([4, 5, 6, 0, 0, 0])
        np.testing.assert_allclose(
                robot.getEEPos().gTAA().flatten(), [1, 2, 3, 0, 0, 0])
        np.testing.assert_allclose(
                robot.getBasePos().gTAA().flatten(), [4, 5, 6, 0, 0, 0])

    def test_robot_model_unimplemented_stubs_return_none(self):
        robot = Robot()
        self.assertIsNone(robot.FK())
        self.assertIsNone(robot.IK())
        self.assertIsNone(robot.randomPos())
        self.assertIsNone(robot.move(tm()))
        self.assertIsNone(robot.draw())

    def test_robot_model_default_inverseJacobian_from_jacobian(self):
        robot = _JacobianOnlyRobot()
        inv_jac = robot.inverseJacobian()
        np.testing.assert_allclose(inv_jac, np.eye(6) * 0.5, atol=1e-8)

    def test_robot_model_default_jacobian_from_inverseJacobian(self):
        robot = _InverseJacobianOnlyRobot()
        jac = robot.jacobian()
        np.testing.assert_allclose(jac, np.eye(6) * 2.0, atol=1e-8)

    def test_robot_model_velocityAtJoints(self):
        robot = _JacobianOnlyRobot()
        # inverseJacobian defaults to pinv(jacobian) = pinv(2*I) = 0.5*I
        twist = np.ones(6)
        joint_vels = robot.velocityAtJoints(twist)
        np.testing.assert_allclose(joint_vels.flatten(), np.ones(6) * 0.5, atol=1e-8)

    def test_robot_model_staticForcesInvBody(self):
        robot = _JacobianOnlyRobot()
        forces = np.ones(6)
        wrench = robot.staticForcesInvBody(forces)
        self.assertIsInstance(wrench, Wrench)
        np.testing.assert_allclose(robot._last_tau, forces)


if __name__ == '__main__':
    unittest.main()
