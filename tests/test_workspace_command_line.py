import os
import tempfile
import unittest

import numpy as np

from basic_robotics.workspace.command_line import CommandExecutor


IRB_2400_URDF = os.path.join(
        os.path.dirname(__file__), 'test_helpers', 'irb_2400.urdf')
MESH_FILE = os.path.join(
        os.path.dirname(__file__), 'test_helpers', 'ur_description',
        'meshes', 'ur5', 'collision', 'base.stl')


class test_workspace_command_line(unittest.TestCase):

    def setUp(self):
        self.executor = CommandExecutor()

    def test_constructor_initializes_expected_state(self):
        # Regression test: __init was a typo for __init__, so this
        # initializer never actually ran and CommandExecutor() came back
        # with none of these attributes set.
        self.assertIsNone(self.executor.analyzer)
        self.assertFalse(self.executor.ready)
        self.assertFalse(self.executor.done)
        self.assertEqual(self.executor.sequence, [])

    def test_cmd_prepare_splits_on_spaces_and_equals(self):
        parsed = self.executor.cmd_prepare('analyzeBruteManipulability -numIterations=2 -plot')
        self.assertEqual(parsed, ['analyzeBruteManipulability', '-numIterations', '2', '-plot'])

    def test_save_results_flag(self):
        self.assertEqual(self.executor.save_results_flag(['cmd']), (False, ''))
        self.assertEqual(
                self.executor.save_results_flag(['cmd', '-o', 'out.dat']), (True, 'out.dat'))

    def test_unrecognized_command_does_not_raise(self):
        self.executor.cmd_parser('notARealCommand')

    def test_analysis_command_requires_a_loaded_robot_first(self):
        self.assertFalse(self.executor.ready)
        # Should print a message and return without raising or requiring a robot.
        self.executor.cmd_parser('analyzeBruteManipulability -numIterations=2')

    def test_cmd_load_robot_from_urdf_builds_a_ready_analyzer(self):
        link = self.executor.cmd_load_robot(['loadRobot', '-fromURDF', IRB_2400_URDF])
        self.assertTrue(self.executor.ready)
        self.assertIsNotNone(self.executor.analyzer)
        self.assertTrue(link.is_ready())
        # Regression test: Arm exposes `_vis_props`/`_col_props` (private),
        # not the `vis_props`/`col_props` this loader used to read - reading
        # the wrong (nonexistent) attribute would raise AttributeError, which
        # not raising here already rules out. Confirm it copied the right thing.
        self.assertIs(link.vis_props, link.robot._vis_props)
        self.assertIs(link.col_props, link.robot._col_props)

    def test_full_dispatch_runs_brute_manipulability_end_to_end(self):
        self.executor.cmd_parser('loadRobot -fromURDF=' + IRB_2400_URDF)
        results = self.executor.cmd_parser('analyzeBruteManipulability -numIterations=2')
        self.assertEqual(len(results), 64)  # 2 positions ^ 6 dof

    def test_full_dispatch_runs_exhaustive_workspace_end_to_end(self):
        self.executor.cmd_parser('loadRobot -fromURDF=' + IRB_2400_URDF)
        pose_cloud = self.executor.cmd_parser('exhaustiveMethodTotalWorkspace -numIterations=2')
        self.assertGreater(len(pose_cloud), 0)
        self.assertIs(self.executor.exhaustive_pose_cloud, pose_cloud)

    def _write_point_cloud_file(self, points):
        with tempfile.NamedTemporaryFile('w', suffix='.txt', delete=False) as f:
            for p in points:
                f.write('[' + ', '.join(str(v) for v in p) + ']\n')
            return f.name

    def test_full_dispatch_runs_task_space_manipulability_end_to_end(self):
        link = self.executor.cmd_load_robot(['loadRobot', '-fromURDF', IRB_2400_URDF])
        reachable = link.robot.FK(np.array([0.1, 0.2, -0.1, 0.1, 0.1, 0.1])).TAA.flatten()
        fname = self._write_point_cloud_file([reachable, reachable])
        try:
            results = self.executor.cmd_parser(
                    'analyzeTaskSpaceManipulability -f=' + fname + ' -manipulationResolution=9')
        finally:
            os.remove(fname)
        self.assertEqual(len(results), 2)
        self.assertIs(self.executor.pose_results_manipulability_volume, results)

    def test_full_dispatch_runs_unit_shell_manipulability_end_to_end(self):
        # Regression test: the -numShells handler used to check the -range
        # flag a second time instead of -numShells, so -numShells was
        # silently ignored and this always used shells_range's value instead.
        # (-range is parsed with int(), so it must be a whole number.)
        self.executor.cmd_parser('loadRobot -fromURDF=' + IRB_2400_URDF)
        results = self.executor.cmd_parser(
                'unitShellManipulability -range=2 -numShells=2 -numShellPoints=4')
        self.assertGreater(len(results), 4)
        self.assertIs(self.executor.pose_results_6dof_shells, results)

    def test_full_dispatch_runs_object_surface_manipulability_end_to_end(self):
        # This fixture URDF carries no visual/collision geometry, so collision
        # detection is left off here; the collision-aware branch of object
        # surface analysis is exercised directly in test_workspace_analyzer.
        self.executor.cmd_parser('loadRobot -fromURDF=' + IRB_2400_URDF)
        results = self.executor.cmd_parser(
                'objectSurfaceManipulability -f=' + MESH_FILE
                + ' -pose=[1.2,0,1.5,0,0,0] -manipulationResolution=9 -minDist=0.05')
        self.assertGreater(len(results), 0)
        self.assertIs(self.executor.pose_results_object_surface, results)

if __name__ == '__main__':
    unittest.main()
