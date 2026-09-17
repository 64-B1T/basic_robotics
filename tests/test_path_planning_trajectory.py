import unittest

import numpy as np

from basic_robotics.general import tm, fsr
from basic_robotics.kinematics import Arm
from basic_robotics.path_planning.trajectory import (
    TrapezoidalProfile, SCurveProfile, timeScaleProfile,
    JointTrajectory, CartesianTrajectory,
)


def _make_test_arm():
    Base_T = tm()
    L1, L2, L3, W = 4.5, 3.75, 3.75, 0.1
    basic_arm_end_effector_home = fsr.TAAtoTM(np.array([[L2+L3+W+W+W], [0], [L1], [0], [0], [0]]))
    basic_arm_joint_axes = np.array(
            [[0, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 0], [0, 1, 0], [1, 0, 0]]).conj().T
    basic_arm_joint_homes = np.array(
            [[0, 0, 0], [0, 0, L1], [L2, 0, L1], [L2+L3, 0, L1],
             [L2+L3+W, 0, L1], [L2+L3+2*W, 0, L1]]).conj().T
    basic_arm_screw_list = np.zeros((6, 6))
    for i in range(6):
        basic_arm_screw_list[0:6, i] = np.hstack((
                basic_arm_joint_axes[0:3, i],
                np.cross(basic_arm_joint_homes[0:3, i], basic_arm_joint_axes[0:3, i])))
    arm = Arm(Base_T, basic_arm_screw_list, basic_arm_end_effector_home,
            basic_arm_joint_homes, basic_arm_joint_axes)
    return arm


class test_trapezoidal_profile(unittest.TestCase):

    def test_reaches_v_max_on_a_long_move(self):
        p = TrapezoidalProfile(10.0, v_max=2.0, a_max=1.0)
        self.assertAlmostEqual(p.v_peak, 2.0)
        self.assertGreater(p.t_flat, 0.0)

    def test_triangular_on_a_short_move(self):
        p = TrapezoidalProfile(1.0, v_max=100.0, a_max=1.0)
        self.assertLess(p.v_peak, 100.0)
        self.assertEqual(p.t_flat, 0.0)

    def test_boundary_conditions(self):
        p = TrapezoidalProfile(5.0, v_max=2.0, a_max=1.0)
        self.assertAlmostEqual(p.position(0.0), 0.0)
        self.assertAlmostEqual(p.position(p.duration), 5.0, places=6)
        self.assertAlmostEqual(p.velocity(0.0), 0.0)
        self.assertAlmostEqual(p.velocity(p.duration), 0.0, places=6)

    def test_negative_distance_is_mirrored(self):
        p = TrapezoidalProfile(-5.0, v_max=2.0, a_max=1.0)
        self.assertAlmostEqual(p.position(p.duration), -5.0, places=6)

    def test_never_exceeds_limits(self):
        p = TrapezoidalProfile(7.0, v_max=1.5, a_max=0.7)
        ts = np.linspace(0, p.duration, 500)
        vels = [p.velocity(t) for t in ts]
        accels = [p.acceleration(t) for t in ts]
        self.assertLessEqual(max(vels), 1.5 + 1e-9)
        self.assertLessEqual(max(np.abs(accels)), 0.7 + 1e-9)

    def test_rejects_infinite_limits(self):
        with self.assertRaises(ValueError):
            TrapezoidalProfile(1.0, v_max=np.inf, a_max=1.0)
        with self.assertRaises(ValueError):
            TrapezoidalProfile(1.0, v_max=1.0, a_max=0.0)


class test_scurve_profile(unittest.TestCase):

    def test_boundary_conditions_long_move(self):
        p = SCurveProfile(10.0, v_max=2.0, a_max=1.0, j_max=5.0)
        self.assertAlmostEqual(p.position(0.0), 0.0)
        self.assertAlmostEqual(p.position(p.duration), 10.0, places=5)
        self.assertAlmostEqual(p.velocity(0.0), 0.0, places=6)
        self.assertAlmostEqual(p.velocity(p.duration), 0.0, places=5)
        self.assertAlmostEqual(p.acceleration(0.0), 0.0, places=6)
        self.assertAlmostEqual(p.acceleration(p.duration), 0.0, places=5)

    def test_boundary_conditions_short_move_no_cruise(self):
        # Distance too small to ever reach v_max.
        p = SCurveProfile(0.05, v_max=10.0, a_max=5.0, j_max=20.0)
        self.assertAlmostEqual(p.position(0.0), 0.0)
        self.assertAlmostEqual(p.position(p.duration), 0.05, places=5)
        self.assertAlmostEqual(p.velocity(p.duration), 0.0, places=5)
        self.assertLess(max(p.velocity(t) for t in np.linspace(0, p.duration, 200)), 10.0)

    def test_never_exceeds_limits(self):
        p = SCurveProfile(8.0, v_max=1.5, a_max=0.8, j_max=2.0)
        ts = np.linspace(0, p.duration, 1000)
        vels = [p.velocity(t) for t in ts]
        accels = [p.acceleration(t) for t in ts]
        self.assertLessEqual(max(vels), 1.5 + 1e-6)
        self.assertLessEqual(max(np.abs(accels)), 0.8 + 1e-6)

    def test_takes_at_least_as_long_as_trapezoidal(self):
        trap = TrapezoidalProfile(6.0, v_max=2.0, a_max=1.0)
        scurve = SCurveProfile(6.0, v_max=2.0, a_max=1.0, j_max=3.0)
        self.assertGreaterEqual(scurve.duration, trap.duration - 1e-9)

    def test_velocity_is_continuous(self):
        p = SCurveProfile(3.0, v_max=1.0, a_max=0.5, j_max=1.0)
        ts = np.linspace(0, p.duration, 400)
        vels = np.array([p.velocity(t) for t in ts])
        self.assertTrue(np.all(np.abs(np.diff(vels)) < 0.05))


class test_time_scale_profile_factory(unittest.TestCase):

    def test_none_jerk_gives_trapezoidal(self):
        p = timeScaleProfile(3.0, 1.0, 1.0, None)
        self.assertIsInstance(p, TrapezoidalProfile)

    def test_finite_jerk_gives_scurve(self):
        p = timeScaleProfile(3.0, 1.0, 1.0, 2.0)
        self.assertIsInstance(p, SCurveProfile)

    def test_infinite_jerk_gives_trapezoidal(self):
        p = timeScaleProfile(3.0, 1.0, 1.0, np.inf)
        self.assertIsInstance(p, TrapezoidalProfile)


class test_joint_trajectory(unittest.TestCase):

    def test_endpoints_match_waypoints(self):
        wps = [np.array([0.0, 0.0]), np.array([1.0, -2.0]), np.array([1.0, 3.0])]
        traj = JointTrajectory(wps, max_vel=1.0, max_accel=1.0)
        np.testing.assert_allclose(traj.position(0.0), wps[0])
        np.testing.assert_allclose(traj.position(traj.duration), wps[-1], atol=1e-6)

    def test_velocity_zero_at_each_waypoint_boundary(self):
        wps = [np.array([0.0]), np.array([2.0]), np.array([-1.0])]
        traj = JointTrajectory(wps, max_vel=1.0, max_accel=1.0)
        t_mid = traj._segments[0][1]
        np.testing.assert_allclose(traj.velocity(0.0), [0.0], atol=1e-6)
        np.testing.assert_allclose(traj.velocity(t_mid), [0.0], atol=1e-6)
        np.testing.assert_allclose(traj.velocity(traj.duration), [0.0], atol=1e-6)

    def test_axes_are_synchronized_within_a_segment(self):
        # A far-moving axis and a near-moving axis should still start/stop together.
        wps = [np.array([0.0, 0.0]), np.array([10.0, 0.5])]
        traj = JointTrajectory(wps, max_vel=[2.0, 2.0], max_accel=[1.0, 1.0])
        dur = traj._segments[0][1]
        np.testing.assert_allclose(traj.velocity(dur), [0.0, 0.0], atol=1e-6)

    def test_per_axis_limits_are_respected(self):
        wps = [np.array([0.0, 0.0]), np.array([5.0, -3.0])]
        max_vel, max_accel = np.array([1.0, 0.5]), np.array([0.5, 0.25])
        traj = JointTrajectory(wps, max_vel, max_accel)
        times, _, vels, accels = traj.sample(0.01)
        self.assertTrue(np.all(np.abs(vels) <= max_vel + 1e-6))
        self.assertTrue(np.all(np.abs(accels) <= max_accel + 1e-6))

    def test_jerk_limited_respects_limits(self):
        wps = [np.array([0.0]), np.array([4.0])]
        traj = JointTrajectory(wps, max_vel=1.0, max_accel=1.0, max_jerk=2.0)
        _, _, vels, accels = traj.sample(0.01)
        self.assertTrue(np.all(np.abs(vels) <= 1.0 + 1e-6))
        self.assertTrue(np.all(np.abs(accels) <= 1.0 + 1e-6))

    def test_requires_at_least_two_waypoints(self):
        with self.assertRaises(ValueError):
            JointTrajectory([np.array([0.0])], max_vel=1.0, max_accel=1.0)

    def test_requires_finite_limits(self):
        with self.assertRaises(ValueError):
            JointTrajectory(
                    [np.array([0.0]), np.array([1.0])],
                    max_vel=np.inf, max_accel=1.0)


class test_cartesian_trajectory(unittest.TestCase):

    def test_endpoints_match_waypoints_pure_translation(self):
        X0, X1 = tm(), tm([1.0, 0.0, 0.0, 0, 0, 0])
        traj = CartesianTrajectory([X0, X1], v_max=1.0, a_max=1.0)
        np.testing.assert_allclose(traj.position(0.0).gTM(), X0.gTM(), atol=1e-6)
        np.testing.assert_allclose(traj.position(traj.duration).gTM(), X1.gTM(), atol=1e-5)

    def test_endpoints_match_waypoints_pure_rotation(self):
        X0, X1 = tm(), tm([0, 0, 0, 0, 0, np.pi / 2])
        traj = CartesianTrajectory([X0, X1], v_max=1.0, a_max=1.0)
        np.testing.assert_allclose(traj.position(0.0).gTM(), X0.gTM(), atol=1e-6)
        np.testing.assert_allclose(traj.position(traj.duration).gTM(), X1.gTM(), atol=1e-5)

    def test_multi_segment_path_endpoints(self):
        X0 = tm()
        X1 = tm([1.0, 0.0, 0.0, 0, 0, 0])
        X2 = tm([1.0, 1.0, 0.0, 0, 0, np.pi / 4])
        traj = CartesianTrajectory([X0, X1, X2], v_max=0.5, a_max=0.5)
        np.testing.assert_allclose(traj.position(0.0).gTM(), X0.gTM(), atol=1e-6)
        np.testing.assert_allclose(traj.position(traj.duration).gTM(), X2.gTM(), atol=1e-5)
        self.assertGreater(traj.duration, 0.0)

    def test_jerk_limited_cartesian(self):
        X0, X1 = tm(), tm([2.0, 0.0, 0.0, 0, 0, 0])
        traj = CartesianTrajectory([X0, X1], v_max=1.0, a_max=1.0, j_max=2.0)
        np.testing.assert_allclose(traj.position(traj.duration).gTM(), X1.gTM(), atol=1e-5)


class test_arm_time_parametrize_path(unittest.TestCase):

    def test_uses_arm_configured_limits(self):
        arm = _make_test_arm()
        arm.setJointProperties(
                max_vels=np.ones(6) * 1.0,
                max_accels=np.ones(6) * 0.5)
        path = [np.zeros(6), np.array([0.3, -0.2, 0.1, 0.0, 0.2, -0.1])]
        traj = arm.timeParametrizePath(path)
        self.assertIsInstance(traj, JointTrajectory)
        self.assertGreater(traj.duration, 0.0)
        np.testing.assert_allclose(traj.position(traj.duration), path[-1], atol=1e-6)

    def test_override_limits_without_mutating_arm(self):
        arm = _make_test_arm()
        arm.setJointProperties(max_vels=np.ones(6), max_accels=np.ones(6))
        path = [np.zeros(6), np.ones(6) * 0.5]
        traj_default = arm.timeParametrizePath(path)
        traj_fast = arm.timeParametrizePath(path, max_vels=np.ones(6) * 10, max_accels=np.ones(6) * 10)
        self.assertLess(traj_fast.duration, traj_default.duration)
        np.testing.assert_allclose(arm.max_vels, np.ones(6))

    def test_requires_finite_limits_by_default(self):
        arm = _make_test_arm()  # max_vels/max_accels default to inf
        path = [np.zeros(6), np.ones(6) * 0.1]
        with self.assertRaises(ValueError):
            arm.timeParametrizePath(path)


if __name__ == '__main__':
    unittest.main()
