import unittest
from io import StringIO
from contextlib import redirect_stdout

import numpy as np

from basic_robotics.workspace.robot_link import RobotLink


class FakeRobot:
    """A minimal duck-typed robot, used to prove RobotLink is not coupled to
    any particular kinematics implementation."""

    def jacobianBody(self, theta):
        return np.eye(6) * theta[0]


class test_workspace_robot_link(unittest.TestCase):

    def setUp(self):
        self.robot = FakeRobot()
        self.link = RobotLink(self.robot)

    def test_starts_unready_and_reports_unbound_methods(self):
        self.assertFalse(self.link.is_ready())
        out = StringIO()
        with redirect_stdout(out):
            self.link.print_unbound()
        printed = out.getvalue()
        self.assertIn('FK', printed)
        self.assertIn('IK', printed)
        self.assertIn('EE', printed)

    def test_becomes_ready_once_all_methods_bound(self):
        self.link.bind_fk(lambda theta: ('fk', theta))
        self.link.bind_ik(lambda goal: ('ik', goal))
        self.link.bind_ee(lambda: 'ee')
        self.link.bind_jt(lambda: 'jt')
        self.assertTrue(self.link.is_ready())

    def test_bound_methods_delegate_with_correct_arguments(self):
        calls = {}
        self.link.bind_fk(lambda theta: calls.setdefault('fk', theta))
        self.link.bind_ik(lambda goal: calls.setdefault('ik', goal))
        self.link.bind_ee(lambda: 'ee_result')
        self.link.bind_jt(lambda: 'jt_result')

        self.link.FK(np.array([1, 2, 3]))
        self.link.IK('goal_tm')
        self.assertEqual(self.link.getEE(), 'ee_result')
        self.assertEqual(self.link.getJointTransforms(), 'jt_result')

        np.testing.assert_array_equal(calls['fk'], np.array([1, 2, 3]))
        self.assertEqual(calls['ik'], 'goal_tm')

    def test_jacobian_body_delegates_to_wrapped_robot(self):
        result = self.link.jacobianBody(np.array([3.0]))
        np.testing.assert_array_equal(result, np.eye(6) * 3.0)


if __name__ == '__main__':
    unittest.main()
