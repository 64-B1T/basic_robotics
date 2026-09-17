import os
import unittest

import numpy as np

from basic_robotics.general import tm, fsr
from basic_robotics.kinematics import Arm
from basic_robotics.kinematics.visual_info import vis_info
from basic_robotics.workspace.alpha_shape import AlphaShape
from basic_robotics.workspace.robot_link import RobotLink
from basic_robotics.workspace.analyzer import (
    WorkspaceAnalyzer,
    calculate_manipulability_score,
    gen_manip_sphere,
    get_collision_data,
    ignore_close_points,
    inside_alpha_shape,
    maximize_manipulability_at_point,
    moller_trumbore_ray_intersection,
    moller_trumbore_ray_intersection_array,
    optimize_robot_for_goals,
    process_empty,
    process_point,
    setup_collision_manager,
)


def make_test_arm():
    """Small 6-DOF arm (same proportions as the kinematics/collision test
    fixtures elsewhere in the suite), with box visual properties (so it can
    be wrapped in a ColliderArm) and mass properties (so link-mass statics
    can be exercised)."""
    base_t = tm()
    l1, l2, l3, w = 1.5, 1.25, 1.25, 0.1
    ee_home = fsr.TAAtoTM(np.array([[l2 + l3 + w + w + w], [0], [l1], [0], [0], [0]]))
    joint_axes = np.array(
            [[0, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 0], [0, 1, 0], [1, 0, 0]]).conj().T
    joint_homes = np.array(
            [[0, 0, 0], [0, 0, l1], [l2, 0, l1], [l2 + l3, 0, l1],
             [l2 + l3 + w, 0, l1], [l2 + l3 + 2 * w, 0, l1]]).conj().T
    screw_list = np.zeros((6, 6))
    for i in range(6):
        screw_list[0:6, i] = np.hstack((
                joint_axes[0:3, i], np.cross(joint_homes[0:3, i], joint_axes[0:3, i])))
    box_dims = np.array(
            [[w, w, l1], [l2, w, w], [l3, w, w], [w, w, w], [w, w, w], [w, w, w]]).conj().T

    arm = Arm(base_t, screw_list, ee_home, joint_homes, joint_axes)

    vis_props = []
    for i in range(6):
        info = vis_info()
        info.geo_type = 'box'
        info.box_size = box_dims[:, i]
        vis_props.append(info)
    eef_info = vis_info()
    eef_info.geo_type = 'box'
    eef_info.box_size = [w, w, w]
    vis_props.append(eef_info)
    arm.setVisColProperties(vis_props=vis_props)

    masses = np.array([5.0, 5.0, 5.0, 1.0, 1.0, 1.0, 0.1])
    mass_centers = [tm() for _ in range(7)]
    arm.setMassProperties(masses, mass_centers)
    return arm


def make_robot_link(arm):
    link = RobotLink(arm)
    link.bind_fk(lambda theta: (arm.FK(theta), True))
    link.bind_ik(lambda goal: arm.IK(goal, protect=True))
    link.bind_ee(arm.getEEPos)
    link.bind_jt(arm.getJointTransforms)
    link.joint_mins = arm.joint_mins
    link.joint_maxs = arm.joint_maxs
    link.link_names = arm.link_names
    return link


# A configuration well clear of the arm's singular (fully outstretched) pose,
# so Jacobian-based manipulability scores stay real-valued.
NON_SINGULAR_POSE = np.array([0.3, 0.4, -0.3, 0.2, 0.5, 0.1])

MESH_FILE = os.path.join(
        os.path.dirname(__file__), 'test_helpers', 'ur_description',
        'meshes', 'ur5', 'collision', 'base.stl')


class test_workspace_analyzer(unittest.TestCase):

    def setUp(self):
        self.arm = make_test_arm()
        self.link = make_robot_link(self.arm)
        self.analyzer = WorkspaceAnalyzer(self.link)

    # -- module-level helper functions ---------------------------------

    def test_gen_manip_sphere_shape(self):
        sphere, true_rez = gen_manip_sphere(9)
        self.assertEqual(sphere.shape[1], 3)
        self.assertEqual(len(sphere), true_rez)
        # The sphere always carries an extra "no rotation" sample at the origin.
        self.assertTrue(np.any(np.all(sphere == 0, axis=1)))

    def test_moller_trumbore_single_point_hit_and_miss(self):
        triangle = [np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])]
        hit = moller_trumbore_ray_intersection(np.array([0.1, 0.1, -1.0]), triangle, ray_dir=[0, 0, 1])
        miss = moller_trumbore_ray_intersection(np.array([5.0, 5.0, -1.0]), triangle, ray_dir=[0, 0, 1])
        self.assertTrue(hit)
        self.assertFalse(miss)

    def test_moller_trumbore_array_matches_single_point_version(self):
        triangle = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        points = np.array([[0.1, 0.1, -1.0], [5.0, 5.0, -1.0]])
        result = moller_trumbore_ray_intersection_array(points, triangle, ray=np.array([0.0, 0.0, 1.0]))
        np.testing.assert_array_equal(result, [True, False])

    def test_inside_alpha_shape_separates_interior_and_exterior_points(self):
        rng = np.random.default_rng(7)
        cloud = rng.random((100, 3)) * 2.0
        shape = AlphaShape(cloud, alpha=1.5, mode=1)
        center = cloud.mean(axis=0)
        far_away = np.array([1000.0, 1000.0, 1000.0])
        # A single-point batch trips a ZeroDivisionError in the shared
        # progressBar helper (unrelated pre-existing bug), so test both
        # points in one batch rather than one point cloud at a time.
        result = inside_alpha_shape(shape, np.array([center, far_away]))
        self.assertEqual(len(result), 1)
        np.testing.assert_allclose(result[0], center)

    def test_process_empty(self):
        result = process_empty('p')
        self.assertEqual(result, ['p', 0, [], [], []])

    def test_ignore_close_points(self):
        seen = [np.array([0.0, 0.0, 0.0])]
        empty_results = []
        continuance, empty_results = ignore_close_points(seen, empty_results, np.array([0.01, 0, 0]), 0.5)
        self.assertTrue(continuance)
        self.assertEqual(len(empty_results), 1)

        continuance, empty_results = ignore_close_points(seen, empty_results, np.array([10.0, 0, 0]), 0.5)
        self.assertFalse(continuance)

    def test_calculate_manipulability_score_is_real_valued_away_from_singularity(self):
        self.arm.FK(NON_SINGULAR_POSE)
        rv, rw = calculate_manipulability_score(self.arm, NON_SINGULAR_POSE)
        self.assertGreater(rv.real, 0)
        self.assertGreater(rw.real, 0)
        self.assertAlmostEqual(rv.imag, 0)
        self.assertAlmostEqual(rw.imag, 0)

    def test_maximize_manipulability_at_point_returns_reachable_pose(self):
        self.arm.FK(NON_SINGULAR_POSE)
        target = self.arm.getEEPos().TAA.flatten()[0:3]
        score, pose, theta = maximize_manipulability_at_point(self.arm, target)
        self.assertGreater(score, 0)
        np.testing.assert_allclose(pose.TAA.flatten()[0:3], target, atol=1e-3)

    def test_process_point_jacobian_mode(self):
        self.arm.FK(NON_SINGULAR_POSE)
        target = self.arm.getEEPos().TAA.flatten()[0:3]
        result = process_point(target, None, None, self.arm, use_jacobian=True)
        point, score, successes, thetas = result
        self.assertGreater(score, 0)
        self.assertEqual(len(successes), 1)
        self.assertEqual(len(thetas), 1)

    def test_process_point_unit_sphere_mode(self):
        self.arm.FK(NON_SINGULAR_POSE)
        target = self.arm.getEEPos().TAA.flatten()[0:3]
        sphere, true_rez = gen_manip_sphere(9)
        result = process_point(target, sphere, true_rez, self.arm, use_jacobian=False)
        point, score, successes, thetas = result
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 1)

    def test_setup_collision_manager_reports_no_collision_for_extended_arm(self):
        manager = setup_collision_manager(self.arm)
        manager.update()
        self.assertFalse(get_collision_data(manager))

    def test_setup_collision_manager_unwraps_robot_link(self):
        # Regression test: ColliderArm needs the concrete Arm (for its
        # `_col_props`/`_vis_props`/`link_names` and `getJointTransforms(bool)`)
        # not the generic RobotLink adapter, which doesn't expose that shape.
        via_link = setup_collision_manager(self.link)
        via_arm = setup_collision_manager(self.arm)
        via_link.update()
        via_arm.update()
        self.assertEqual(get_collision_data(via_link), get_collision_data(via_arm))

    # -- WorkspaceAnalyzer methods ---------------------------------------

    def test_analyze_task_space_distinguishes_reachable_from_unreachable(self):
        # Regression test: the original code checked truthiness of the raw
        # (theta, success) tuple IK returns, which is always truthy.
        self.arm.FK(NON_SINGULAR_POSE)
        reachable = self.arm.getEEPos()
        unreachable = tm([1000.0, 1000.0, 1000.0, 0, 0, 0])
        num_successful, successful_poses = self.analyzer.analyze_task_space(
                [reachable, unreachable])
        self.assertEqual(num_successful, 1)
        self.assertEqual(len(successful_poses), 1)

    def _two_reachable_targets(self):
        # A single-point batch trips a ZeroDivisionError in the shared
        # progressBar helper (unrelated pre-existing bug), so tests use two
        # points rather than one.
        self.arm.FK(NON_SINGULAR_POSE)
        first = self.arm.getEEPos()
        second = first @ tm([0.05, 0, 0, 0, 0, 0])
        return [first, second]

    def test_analyze_task_space_manipulability_jacobian_mode(self):
        targets = self._two_reachable_targets()
        results = self.analyzer.analyze_task_space_manipulability(
                targets, manip_resolution=9, use_jacobian=True)
        self.assertEqual(len(results), 2)
        self.assertGreater(results[0][1], 0)

    def test_analyze_task_space_manipulability_unit_sphere_mode(self):
        targets = self._two_reachable_targets()
        results = self.analyzer.analyze_task_space_manipulability(
                targets, manip_resolution=9, use_jacobian=False)
        self.assertEqual(len(results), 2)
        self.assertGreaterEqual(results[0][1], 0)

    def test_analyze_task_space_manipulability_with_collision_detect(self):
        # self.analyzer.bot is a RobotLink here, exercising the same unwrap
        # path as test_setup_collision_manager_unwraps_robot_link but through
        # the full non-parallel analysis method.
        targets = self._two_reachable_targets()
        results = self.analyzer.analyze_task_space_manipulability(
                targets, manip_resolution=9, use_jacobian=True, collision_detect=True)
        self.assertEqual(len(results), 2)

    def test_analyze_total_workspace_exhaustive_point_cloud_small(self):
        cloud = self.analyzer.analyze_total_workspace_exhaustive_point_cloud(num_iterations=2)
        self.assertGreater(len(cloud), 0)
        for pose in cloud:
            self.assertIsInstance(pose, tm)

    def test_analyze_total_workspace_functional_small(self):
        cloud = self.analyzer.analyze_total_workspace_functional(num_spread=4)
        self.assertGreater(len(cloud), 0)

    def test_analyze_brute_manipulability_on_joints_small(self):
        # No joints excluded, all six vary at the smallest useful resolution
        # (2^6 = 64 FK evaluations - keeps this fast without needing to lock
        # any joints at a value that could coincide with a real kinematic
        # singularity for this fixture's geometry).
        results = self.analyzer.analyze_brute_manipulability_on_joints(
                resolution=2, joint_indexes=[])
        self.assertEqual(len(results), 64)
        for ee_pos, score, theta, manip in results:
            self.assertIsInstance(ee_pos, tm)
            self.assertTrue(np.isfinite(score))
            self.assertGreaterEqual(score, 0)

    def test_analyze_6dof_manipulability_small(self):
        # Regression test: fsr.unitSphere(n) now returns a bare ndarray (not a
        # tuple), so the old `fsr.unitSphere(n)[0]` silently grabbed a single
        # point instead of the whole sphere.
        results = self.analyzer.analyze_6dof_manipulability(
                shell_range=1.5, num_shells=2, points_per_shell=4)
        self.assertGreater(len(results), 4)

    def test_analyze_manipulability_within_volume_small(self):
        rng = np.random.default_rng(3)
        cloud = rng.random((60, 3)) * 1.0 + np.array([0.5, -0.5, 1.0])
        shape = AlphaShape(cloud, alpha=1.5, mode=1)
        results = self.analyzer.analyze_manipulability_within_volume(
                shape, grid_resolution=0.5, manip_resolution=9, use_jacobian=True)
        self.assertIsInstance(results, list)

    def test_analyze_manipulability_over_trajectory_small(self):
        self.arm.FK(NON_SINGULAR_POSE)
        base = self.arm.getEEPos()
        waypoints = [base, base @ tm([0.1, 0, 0, 0, 0, 0])]
        results = self.analyzer.analyze_manipulability_over_trajectory(
                waypoints, manipulability_mode=1, manip_resolution=9)
        self.assertEqual(len(results), 2)

    def test_analyze_matching_joint_torques(self):
        # Regression test: current Arm exposes `staticForcesWithLinkMasses`
        # (plural, `(wrench, theta)` order), not the old
        # `staticForceWithLinkMasses(theta, wrench)`.
        targets = self._two_reachable_targets()
        manipulability_space = self.analyzer.analyze_task_space_manipulability(
                targets, manip_resolution=9, use_jacobian=True)
        results = self.analyzer.analyze_matching_joint_torques(
                manipulability_space, mass_cg=tm(), mass=1.0,
                grav_vector=np.array([0, 0, -9.81]))
        self.assertEqual(len(results), 2)
        point, torque_results = results[0]
        self.assertEqual(len(torque_results), 1)
        joint_config, torques = torque_results[0]
        self.assertEqual(len(torques), 6)

    def test_analyze_manipulability_on_object_surface(self):
        # `minimum_dist` thins the mesh's ~280 vertices down to a handful of
        # well-separated sample points so this stays fast, while still
        # exercising mesh loading/scaling/pose, distance filtering,
        # deduplication, and the mesh-aware collision-detect path (a
        # different branch of setup_collision_manager than the other tests
        # exercise, since it also binds an obstacle).
        # use_jacobian=False (unit-sphere sampling) is pinned explicitly: this
        # test is about the mesh/collision/filtering pipeline, not about
        # which manipulability mode is used, and the two modes give genuinely
        # different reachability guarantees (unit-sphere enumerates discrete
        # orientations; jacobian mode locally optimizes from a fixed guess
        # and can legitimately fail to find any working orientation for a
        # point that unit-sphere sampling would have found by enumeration).
        object_pose = tm([1.2, 0, 1.5, 0, 0, 0])
        results, mesh = self.analyzer.analyze_manipulability_on_object_surface(
                MESH_FILE, object_scale=1.0, object_pose=object_pose,
                manip_resolution=9, collision_detect=True, exempt_ee=True,
                use_jacobian=False, minimum_dist=0.05)
        self.assertGreater(len(results), 0)
        # process_point results are 4-tuples, but process_empty (used for
        # points discarded before evaluation) returns a 5-tuple, so index
        # rather than unpack a fixed arity across the combined list.
        for result in results:
            self.assertGreaterEqual(result[1], 0)
            self.assertLessEqual(result[1], 1)
        # At least one point should have been close enough to actually be
        # evaluated (as opposed to summarily discarded as out of reach).
        self.assertTrue(any(result[1] > 0 for result in results))

    def test_analyze_joint_related_to_end_effector_vals(self):
        # Regression test: current Robot/Arm expose `velocityAtEndEffector`,
        # not the removed `_jointsToEndEffectorJacobian`.
        targets = self._two_reachable_targets()
        manipulability_space = self.analyzer.analyze_task_space_manipulability(
                targets, manip_resolution=9, use_jacobian=True)
        results = self.analyzer.analyze_joint_related_to_end_effector_vals(
                manipulability_space, joint_goal_norm=0.5)
        self.assertEqual(len(results), 2)


class test_workspace_optimize_robot_for_goals(unittest.TestCase):
    """optimize_robot_for_goals only needs a duck-typed builder function
    (build_robot(x) -> object with .link_home_positions and .IK), so it is
    tested independently of any concrete kinematics model."""

    def test_optimize_robot_for_goals_converges_on_trivial_problem(self):
        class FakeBot:
            def __init__(self, scale):
                self.link_home_positions = [0, 0]  # length -> num_dofs = 3
                self.scale = scale if scale is not None else np.ones(3)

            def IK(self, goal):
                # Always succeeds; the optimizer should therefore just settle
                # near its initial guess since there's no failure penalty.
                return None, True

        results = optimize_robot_for_goals(FakeBot, goals_list=[tm()], init=np.array([1.0, 1.0, 1.0]))
        self.assertEqual(len(results), 3)


if __name__ == '__main__':
    unittest.main()
